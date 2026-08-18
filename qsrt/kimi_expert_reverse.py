"""Per-expert gate/up curvature and objective-gradient accumulation.

The accumulator consumes official-model expert inputs and concatenated gate/up
cotangents. It moves both tensors into the coupled QSRT W1/W3 coordinates before
forming block output-Fisher factors or a bilateral low-rank weight-gradient
sketch. The sketch reconstructs individual 16x16 gradient tiles on demand, so a
full dense gradient matrix is never required on disk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch

from qsrt.qsrt_coupled import (
    CoupledHadamardExecution,
    CoupledHadamardSpec,
    block_hadamard,
    rotation_signs,
)
from qsrt.segmented_fisher import (
    add_segmented_fisher_128_,
    transform_coupled_preactivation_128,
)


@dataclass(frozen=True)
class CoupledUpstreamReverseConfig:
    """Geometry and storage policy for one routed MoE layer."""

    num_experts: int
    hidden_dimension: int
    intermediate_dimension: int
    spec: CoupledHadamardSpec
    intermediate_draws: tuple[int, ...] | None = None
    output_factor_block_size: int = 128
    gradient_sketch_rank: int = 32
    gradient_sketch_seed: int = 0

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError("expert count must be positive")
        if self.hidden_dimension <= 0 or self.intermediate_dimension <= 0:
            raise ValueError("expert dimensions must be positive")
        output_dimension = 2 * self.intermediate_dimension
        if self.output_factor_block_size <= 0 or (
            output_dimension % self.output_factor_block_size
        ):
            raise ValueError("output-factor block size must divide gate/up width")
        if not 1 <= self.gradient_sketch_rank <= min(
            self.hidden_dimension, output_dimension
        ):
            raise ValueError("gradient sketch rank is incompatible with expert geometry")
        if self.intermediate_draws is not None:
            if len(self.intermediate_draws) != self.num_experts:
                raise ValueError("intermediate draw table must cover every expert")
            if any(not 0 <= int(draw) < 8 for draw in self.intermediate_draws):
                raise ValueError("intermediate draws must lie in 0..7")

    def spec_for_expert(self, expert: int) -> CoupledHadamardSpec:
        """Return the frozen coupled transform used by one stored expert."""

        if not 0 <= expert < self.num_experts:
            raise IndexError("routed expert index is out of range")
        if self.intermediate_draws is None:
            return self.spec
        return CoupledHadamardSpec(
            residual_block_size=self.spec.residual_block_size,
            preactivation_block_size=self.spec.preactivation_block_size,
            postactivation_block_size=self.spec.postactivation_block_size,
            residual_draw=self.spec.residual_draw,
            intermediate_draw=int(self.intermediate_draws[expert]),
        )


def _rademacher_projection(
    rows: int,
    columns: int,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed & ((1 << 63) - 1))
    return (
        torch.randint(0, 2, (rows, columns), generator=generator)
        .mul_(2)
        .sub_(1)
        .to(device=device, dtype=torch.float32)
        .div_(math.sqrt(columns))
    )


class CoupledUpstreamReverseAccumulator:
    """Capture W1/W3 Fisher blocks and deterministic gradient sketches."""

    def __init__(
        self,
        config: CoupledUpstreamReverseConfig,
        *,
        device: torch.device | str,
        capture_fisher: bool = True,
        capture_objective_gradient: bool = True,
    ):
        target = torch.device(device)
        if target.type != "cuda" and target.type != "cpu":
            raise ValueError("reverse accumulation requires a CPU or CUDA device")
        if not capture_fisher and not capture_objective_gradient:
            raise ValueError("at least one reverse quantity must be captured")
        self.config = config
        self.device = target
        self._executions: dict[int, CoupledHadamardExecution] = {}
        output_dimension = 2 * config.intermediate_dimension
        blocks = output_dimension // config.output_factor_block_size
        self.output_factor_sums = (
            torch.zeros(
                (
                    config.num_experts,
                    blocks,
                    config.output_factor_block_size,
                    config.output_factor_block_size,
                ),
                dtype=torch.float32,
                device=target,
            )
            if capture_fisher
            else None
        )
        self.output_factor_rows = torch.zeros(
            config.num_experts,
            dtype=torch.int64,
            device=target,
        )
        self._draws = torch.tensor(
            (
                (config.spec.intermediate_draw,) * config.num_experts
                if config.intermediate_draws is None
                else config.intermediate_draws
            ),
            dtype=torch.long,
            device=target,
        )
        self._preactivation_signs: torch.Tensor | None = None

        rank = config.gradient_sketch_rank
        self.omega_input = (
            _rademacher_projection(
                config.hidden_dimension,
                rank,
                seed=config.gradient_sketch_seed ^ 0x243F6A8885A308D3,
                device=target,
            )
            if capture_objective_gradient
            else None
        )
        self.omega_output = (
            _rademacher_projection(
                output_dimension,
                rank,
                seed=config.gradient_sketch_seed ^ 0x13198A2E03707344,
                device=target,
            )
            if capture_objective_gradient
            else None
        )
        self.gradient_left = (
            torch.zeros(
                (config.num_experts, output_dimension, rank),
                dtype=torch.float32,
                device=target,
            )
            if capture_objective_gradient
            else None
        )
        self.gradient_right = (
            torch.zeros(
                (config.num_experts, rank, config.hidden_dimension),
                dtype=torch.float32,
                device=target,
            )
            if capture_objective_gradient
            else None
        )
        self.gradient_rows = torch.zeros(
            config.num_experts,
            dtype=torch.int64,
            device=target,
        )

    def _validate_expert(self, expert: int) -> None:
        if not 0 <= expert < self.config.num_experts:
            raise IndexError("routed expert index is out of range")

    def _sign_table(self) -> torch.Tensor:
        signs = self._preactivation_signs
        if signs is None:
            output_dimension = 2 * self.config.intermediate_dimension
            signs = torch.stack(
                [
                    rotation_signs(
                        output_dimension,
                        draw=draw,
                        axis=1,
                        device=self.device,
                    )
                    for draw in range(8)
                ]
            )
            self._preactivation_signs = signs
        return signs

    def execution_for_expert(self, expert: int) -> CoupledHadamardExecution:
        """Return a cached transform for the expert's intermediate draw."""

        spec = self.config.spec_for_expert(expert)
        execution = self._executions.get(spec.intermediate_draw)
        if execution is None:
            execution = CoupledHadamardExecution(
                self.config.hidden_dimension,
                self.config.intermediate_dimension,
                spec,
            )
            self._executions[spec.intermediate_draw] = execution
        return execution

    def _transform(
        self,
        expert: int,
        input_rows: torch.Tensor,
        gate_up_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_rows.device != self.device or gate_up_gradient.device != self.device:
            raise ValueError("reverse callback tensors are on the wrong device")
        if input_rows.ndim != 2 or input_rows.shape[1] != self.config.hidden_dimension:
            raise ValueError("routed expert inputs have incompatible geometry")
        expected_output = 2 * self.config.intermediate_dimension
        if (
            gate_up_gradient.ndim != 2
            or gate_up_gradient.shape != (input_rows.shape[0], expected_output)
        ):
            raise ValueError("gate/up cotangents have incompatible geometry")
        execution = self.execution_for_expert(expert)
        inputs = execution.transform_inputs(input_rows.detach()).float()
        gradients = execution.transform_preactivation_gradients(
            gate_up_gradient.detach()
        ).float()
        return inputs, gradients

    def _transform_fisher_gradient(
        self,
        expert: int,
        input_rows: torch.Tensor,
        gate_up_gradient: torch.Tensor,
    ) -> torch.Tensor:
        if input_rows.device != self.device or gate_up_gradient.device != self.device:
            raise ValueError("reverse callback tensors are on the wrong device")
        if input_rows.ndim != 2 or input_rows.shape[1] != self.config.hidden_dimension:
            raise ValueError("routed expert inputs have incompatible geometry")
        expected_output = 2 * self.config.intermediate_dimension
        if (
            gate_up_gradient.ndim != 2
            or gate_up_gradient.shape != (input_rows.shape[0], expected_output)
        ):
            raise ValueError("gate/up cotangents have incompatible geometry")
        return self.execution_for_expert(expert).transform_preactivation_gradients(
            gate_up_gradient.detach()
        ).float()

    @torch.no_grad()
    def add_fisher(
        self,
        expert: int,
        input_rows: torch.Tensor,
        gate_up_gradient: torch.Tensor,
    ) -> None:
        """Accumulate block-diagonal output Fisher for one expert occurrence."""

        self._validate_expert(expert)
        if self.output_factor_sums is None:
            raise RuntimeError("output-Fisher capture is disabled")
        gradients = self._transform_fisher_gradient(
            expert,
            input_rows,
            gate_up_gradient,
        )
        block_size = self.config.output_factor_block_size
        blocked = gradients.reshape(gradients.shape[0], -1, block_size)
        self.output_factor_sums[expert].add_(
            torch.einsum("nbi,nbj->bij", blocked, blocked)
        )
        self.output_factor_rows[expert] += gradients.shape[0]

    @torch.no_grad()
    def add_grouped_fisher(
        self,
        sorted_experts: torch.Tensor,
        offsets: torch.Tensor,
        input_rows: torch.Tensor,
        gate_up_gradient: torch.Tensor,
    ) -> None:
        """Accumulate all routed experts from one grouped dispatch."""

        if self.output_factor_sums is None:
            raise RuntimeError("output-Fisher capture is disabled")
        if self.device.type != "cuda" or self.config.output_factor_block_size != 128:
            self._add_grouped_fallback(
                offsets,
                input_rows,
                gate_up_gradient,
                channel="fisher",
            )
            return
        transformed = transform_coupled_preactivation_128(
            gate_up_gradient,
            sorted_experts,
            self._draws,
            self._sign_table(),
        )
        blocked = transformed.reshape(
            transformed.shape[0],
            -1,
            self.config.output_factor_block_size,
        )
        add_segmented_fisher_128_(self.output_factor_sums, blocked, offsets)
        counts = offsets.to(torch.int64).clone()
        counts[1:] -= offsets[:-1]
        self.output_factor_rows.add_(counts)

    def _add_grouped_fallback(
        self,
        offsets: torch.Tensor,
        input_rows: torch.Tensor,
        gate_up_gradient: torch.Tensor,
        *,
        channel: str,
    ) -> None:
        boundaries = offsets.detach().cpu().tolist()
        begin = 0
        for expert, end in enumerate(boundaries):
            if end > begin:
                if channel == "fisher":
                    self.add_fisher(
                        expert,
                        input_rows[begin:end],
                        gate_up_gradient[begin:end],
                    )
                else:
                    self.add_objective_gradient(
                        expert,
                        input_rows[begin:end],
                        gate_up_gradient[begin:end],
                    )
            begin = end

    @torch.no_grad()
    def add_grouped(
        self,
        sorted_experts: torch.Tensor,
        offsets: torch.Tensor,
        input_rows: torch.Tensor,
        gate_up_gradient: torch.Tensor,
        *,
        channel: str | None,
    ) -> None:
        """Dispatch a grouped reverse callback without changing its objective."""

        if channel is None:
            return
        expected_rows = int(gate_up_gradient.shape[0])
        if (
            sorted_experts.device != self.device
            or offsets.device != self.device
            or input_rows.device != self.device
            or gate_up_gradient.device != self.device
        ):
            raise ValueError("grouped reverse tensors are on the wrong device")
        if sorted_experts.shape != (expected_rows,) or offsets.shape != (
            self.config.num_experts,
        ):
            raise ValueError("grouped reverse routing has incompatible geometry")
        if offsets.dtype != torch.int32:
            raise TypeError("grouped reverse offsets must be int32")
        if channel == "fisher":
            self.add_grouped_fisher(
                sorted_experts,
                offsets,
                input_rows,
                gate_up_gradient,
            )
        elif channel == "objective":
            self._add_grouped_fallback(
                offsets,
                input_rows,
                gate_up_gradient,
                channel=channel,
            )
        else:
            raise ValueError(f"unsupported reverse channel {channel!r}")

    @torch.no_grad()
    def add_objective_gradient(
        self,
        expert: int,
        input_rows: torch.Tensor,
        gate_up_gradient: torch.Tensor,
    ) -> None:
        """Accumulate a bilateral sketch of the coupled W1/W3 gradient."""

        self._validate_expert(expert)
        if (
            self.omega_input is None
            or self.omega_output is None
            or self.gradient_left is None
            or self.gradient_right is None
        ):
            raise RuntimeError("objective-gradient capture is disabled")
        inputs, gradients = self._transform(expert, input_rows, gate_up_gradient)
        self.gradient_left[expert].add_(
            gradients.T @ (inputs @ self.omega_input)
        )
        self.gradient_right[expert].add_(
            (gradients @ self.omega_output).T @ inputs
        )
        self.gradient_rows[expert] += gradients.shape[0]

    @torch.no_grad()
    def reconstructed_gradient(self, expert: int) -> torch.Tensor:
        """Reconstruct one full coupled-coordinate gradient for diagnostics."""

        self._validate_expert(expert)
        if (
            self.omega_output is None
            or self.gradient_left is None
            or self.gradient_right is None
        ):
            raise RuntimeError("objective-gradient capture is disabled")
        left = self.gradient_left[expert]
        right = self.gradient_right[expert]
        core = self.omega_output.T @ left
        return (left @ torch.linalg.pinv(core) @ right).contiguous()

    @torch.no_grad()
    def gradient_tile(
        self,
        expert: int,
        output_begin: int,
        input_begin: int,
        *,
        tile_size: int = 16,
    ) -> torch.Tensor:
        """Reconstruct one Viterbi-sized objective-gradient tile."""

        self._validate_expert(expert)
        output_dimension = 2 * self.config.intermediate_dimension
        if (
            tile_size <= 0
            or output_begin < 0
            or input_begin < 0
            or output_begin + tile_size > output_dimension
            or input_begin + tile_size > self.config.hidden_dimension
        ):
            raise ValueError("gradient tile lies outside the coupled matrix")
        if (
            self.omega_output is None
            or self.gradient_left is None
            or self.gradient_right is None
        ):
            raise RuntimeError("objective-gradient capture is disabled")
        left = self.gradient_left[expert]
        right = self.gradient_right[expert]
        core_inverse = torch.linalg.pinv(self.omega_output.T @ left)
        return (
            left[output_begin : output_begin + tile_size]
            @ core_inverse
            @ right[:, input_begin : input_begin + tile_size]
        ).contiguous()

    def allocated_bytes(self) -> int:
        """Return persistent tensor storage owned by this accumulator."""

        tensors = (
            self.output_factor_sums,
            self.omega_input,
            self.omega_output,
            self.gradient_left,
            self.gradient_right,
        )
        return sum(
            value.numel() * value.element_size()
            for value in tensors
            if value is not None
        )


class CoupledDownReverseAccumulator:
    """Capture deterministic W2 gradients in stored coupled coordinates."""

    def __init__(
        self,
        config: CoupledUpstreamReverseConfig,
        *,
        device: torch.device | str,
    ):
        target = torch.device(device)
        if target.type not in ("cpu", "cuda"):
            raise ValueError("reverse accumulation requires a CPU or CUDA device")
        rank = config.gradient_sketch_rank
        if rank > min(config.hidden_dimension, config.intermediate_dimension):
            raise ValueError("gradient sketch rank is incompatible with W2 geometry")
        self.config = config
        self.device = target
        self._executions: dict[int, CoupledHadamardExecution] = {}
        self.omega_input = _rademacher_projection(
            config.intermediate_dimension,
            rank,
            seed=config.gradient_sketch_seed ^ 0xA4093822299F31D0,
            device=target,
        )
        self.omega_output = _rademacher_projection(
            config.hidden_dimension,
            rank,
            seed=config.gradient_sketch_seed ^ 0x082EFA98EC4E6C89,
            device=target,
        )
        self.gradient_left = torch.zeros(
            (config.num_experts, config.hidden_dimension, rank),
            dtype=torch.float32,
            device=target,
        )
        self.gradient_right = torch.zeros(
            (config.num_experts, rank, config.intermediate_dimension),
            dtype=torch.float32,
            device=target,
        )
        self.gradient_rows = torch.zeros(
            config.num_experts,
            dtype=torch.int64,
            device=target,
        )

    @torch.no_grad()
    def add_grouped_objective_gradient(
        self,
        sorted_experts: torch.Tensor,
        offsets: torch.Tensor,
        postactivation_rows: torch.Tensor,
        output_gradient: torch.Tensor,
    ) -> None:
        """Accumulate grouped W2 gradients through the exact scalar fallback."""

        del sorted_experts
        boundaries = offsets.detach().cpu().tolist()
        begin = 0
        for expert, end in enumerate(boundaries):
            if end > begin:
                self.add_objective_gradient(
                    expert,
                    postactivation_rows[begin:end],
                    output_gradient[begin:end],
                )
            begin = end

    def _validate_expert(self, expert: int) -> None:
        if not 0 <= expert < self.config.num_experts:
            raise IndexError("routed expert index is out of range")

    def execution_for_expert(self, expert: int) -> CoupledHadamardExecution:
        """Return a cached transform for the expert's intermediate draw."""

        spec = self.config.spec_for_expert(expert)
        execution = self._executions.get(spec.intermediate_draw)
        if execution is None:
            execution = CoupledHadamardExecution(
                self.config.hidden_dimension,
                self.config.intermediate_dimension,
                spec,
            )
            self._executions[spec.intermediate_draw] = execution
        return execution

    def _transform(
        self,
        expert: int,
        postactivation_rows: torch.Tensor,
        output_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            postactivation_rows.device != self.device
            or output_gradient.device != self.device
        ):
            raise ValueError("reverse callback tensors are on the wrong device")
        if (
            postactivation_rows.ndim != 2
            or postactivation_rows.shape[1] != self.config.intermediate_dimension
        ):
            raise ValueError("post-SiTU rows have incompatible geometry")
        if output_gradient.ndim != 2 or output_gradient.shape != (
            postactivation_rows.shape[0],
            self.config.hidden_dimension,
        ):
            raise ValueError("expert-output cotangents have incompatible geometry")
        execution = self.execution_for_expert(expert)
        inputs = execution.transform_postactivation_rows(
            postactivation_rows.detach()
        ).float()
        gradients = execution.transform_expert_output_gradients(
            output_gradient.detach()
        ).float()
        return inputs, gradients

    @torch.no_grad()
    def add_objective_gradient(
        self,
        expert: int,
        postactivation_rows: torch.Tensor,
        output_gradient: torch.Tensor,
    ) -> None:
        """Accumulate a bilateral sketch of one expert's W2 gradient."""

        self._validate_expert(expert)
        inputs, gradients = self._transform(
            expert,
            postactivation_rows,
            output_gradient,
        )
        self.gradient_left[expert].add_(
            gradients.T @ (inputs @ self.omega_input)
        )
        self.gradient_right[expert].add_(
            (gradients @ self.omega_output).T @ inputs
        )
        self.gradient_rows[expert] += gradients.shape[0]

    @torch.no_grad()
    def reconstructed_gradient(self, expert: int) -> torch.Tensor:
        """Reconstruct one full stored-coordinate W2 gradient for diagnostics."""

        self._validate_expert(expert)
        left = self.gradient_left[expert]
        right = self.gradient_right[expert]
        core = self.omega_output.T @ left
        return (left @ torch.linalg.pinv(core) @ right).contiguous()

    @torch.no_grad()
    def gradient_tile(
        self,
        expert: int,
        output_begin: int,
        input_begin: int,
        *,
        tile_size: int = 16,
    ) -> torch.Tensor:
        """Reconstruct one Viterbi-sized stored-coordinate W2 gradient tile."""

        self._validate_expert(expert)
        if (
            tile_size <= 0
            or output_begin < 0
            or input_begin < 0
            or output_begin + tile_size > self.config.hidden_dimension
            or input_begin + tile_size > self.config.intermediate_dimension
        ):
            raise ValueError("gradient tile lies outside W2")
        left = self.gradient_left[expert]
        right = self.gradient_right[expert]
        core_inverse = torch.linalg.pinv(self.omega_output.T @ left)
        return (
            left[output_begin : output_begin + tile_size]
            @ core_inverse
            @ right[:, input_begin : input_begin + tile_size]
        ).contiguous()

    def allocated_bytes(self) -> int:
        """Return persistent tensor storage owned by this accumulator."""

        tensors = (
            self.omega_input,
            self.omega_output,
            self.gradient_left,
            self.gradient_right,
        )
        return sum(value.numel() * value.element_size() for value in tensors)


class ExpertReverseChannelRouter:
    """Dispatch retained-graph VJPs into Fisher or objective accumulators."""

    def __init__(
        self,
        *,
        upstream: CoupledUpstreamReverseAccumulator,
        down: CoupledDownReverseAccumulator | None = None,
        routed_fisher_add: Callable[[torch.Tensor], None] | None = None,
    ):
        if down is not None and (
            upstream.config != down.config or upstream.device != down.device
        ):
            raise ValueError("expert reverse accumulators must share one geometry")
        self.upstream = upstream
        self.down = down
        self.routed_fisher_add = routed_fisher_add
        self.channel: str | None = None

    def select_channel(self, channel: str | None) -> None:
        if channel not in (None, "fisher", "objective"):
            raise ValueError(f"unsupported reverse channel {channel!r}")
        self.channel = channel

    def routed_output(self, gradient: torch.Tensor) -> None:
        if self.channel == "fisher" and self.routed_fisher_add is not None:
            self.routed_fisher_add(gradient)

    def expert_preactivation(
        self,
        expert: int,
        input_rows: torch.Tensor,
        gate_up_gradient: torch.Tensor,
    ) -> None:
        if self.channel == "fisher":
            self.upstream.add_fisher(expert, input_rows, gate_up_gradient)
        elif self.channel == "objective":
            self.upstream.add_objective_gradient(
                expert,
                input_rows,
                gate_up_gradient,
            )

    def grouped_expert_preactivation(
        self,
        sorted_experts: torch.Tensor,
        offsets: torch.Tensor,
        input_rows: torch.Tensor,
        gate_up_gradient: torch.Tensor,
    ) -> None:
        """Accumulate one grouped expert-dispatch reverse channel."""

        self.upstream.add_grouped(
            sorted_experts,
            offsets,
            input_rows,
            gate_up_gradient,
            channel=self.channel,
        )

    def expert_output(
        self,
        expert: int,
        postactivation_rows: torch.Tensor,
        output_gradient: torch.Tensor,
    ) -> None:
        if self.channel == "objective" and self.down is not None:
            self.down.add_objective_gradient(
                expert,
                postactivation_rows,
                output_gradient,
            )

    def grouped_expert_output(
        self,
        sorted_experts: torch.Tensor,
        offsets: torch.Tensor,
        postactivation_rows: torch.Tensor,
        output_gradient: torch.Tensor,
    ) -> None:
        """Accumulate grouped W2 objective gradients when requested."""

        if self.channel == "objective" and self.down is not None:
            self.down.add_grouped_objective_gradient(
                sorted_experts,
                offsets,
                postactivation_rows,
                output_gradient,
            )

    def install(self, adapter: Any, module: Any) -> None:
        """Install all reverse taps on one materialized decoder layer."""

        if not adapter.enable_routed_output_gradients(module, self.routed_output):
            raise ValueError("decoder layer has no routed expert output")
        if not adapter.enable_expert_preactivation_gradients(
            module,
            (
                self.grouped_expert_preactivation
                if getattr(
                    getattr(module, "block_sparse_moe", None),
                    "_qsrt_grouped_expert_dispatch",
                    False,
                )
                else self.expert_preactivation
            ),
        ):
            raise ValueError("decoder layer has no gate/up preactivation tap")
        if self.down is not None and not adapter.enable_expert_output_gradients(
            module,
            (
                self.grouped_expert_output
                if getattr(
                    getattr(module, "block_sparse_moe", None),
                    "_qsrt_grouped_expert_dispatch",
                    False,
                )
                else self.expert_output
            ),
        ):
            raise ValueError("decoder layer has no expert W2 output tap")


__all__ = [
    "CoupledDownReverseAccumulator",
    "CoupledUpstreamReverseAccumulator",
    "CoupledUpstreamReverseConfig",
    "ExpertReverseChannelRouter",
]

"""Distributed segment replay for final-output Kimi-K3 curvature.

Kimi-K3's 93 decoder layers use a 12-layer attention-residual block. A replay
segment assigns those 12 consecutive layers to distinct CUDA devices. The
segment input and all earlier block-boundary states are independent leaves;
autograd therefore reduces every chain and attention-residual contribution
from the segment into one input cotangent and one cotangent per earlier block.

The implementation retains no parameter gradients. Routed expert output hooks
accumulate the empirical output-side Fisher factor used by two-sided W2
rounding while the ordinary input VJP continues toward earlier segments.
"""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections.abc import Callable, Mapping
from typing import Any, Protocol, Sequence

import torch

from qsrt.kimi_boundary_slabs import KimiBoundarySlabArchive
from qsrt.kimi_cotangent_slabs import (
    CudaBf16SlabWriter,
    KimiCotangentSlabWorkspace,
)
from qsrt.kimi_output_factors import (
    KimiOutputFactorArchive,
    OutputFactorSums,
    document_factor_split,
)


class KimiReversePipelineAdapter(Protocol):
    """Decoder operations required by a differentiable segment replay."""

    def load_layer(
        self,
        layer: int,
        device: torch.device,
    ) -> tuple[Any, object | None]: ...

    def forward_layer(
        self,
        module: Any,
        *,
        layer: int,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def release_layer(self, module: Any) -> None: ...

    def enable_routed_output_gradients(
        self,
        module: Any,
        callback: Any,
    ) -> bool: ...

    def enable_expert_preactivation_gradients(
        self,
        module: Any,
        callback: Any,
    ) -> bool: ...

    def enable_expert_output_gradients(
        self,
        module: Any,
        callback: Any,
    ) -> bool: ...


class EmpiricalOutputFactor:
    """FP32 sum of output-gradient outer products on one CUDA device."""

    def __init__(self, dimension: int, *, device: torch.device | str):
        target = torch.device(device)
        if target.type != "cuda" or target.index is None:
            raise ValueError("output-factor accumulation requires an indexed CUDA device")
        if dimension <= 0:
            raise ValueError("output-factor dimension must be positive")
        self.dimension = int(dimension)
        self.device = target
        self.sum = torch.zeros(
            (self.dimension, self.dimension),
            dtype=torch.float32,
            device=target,
        )
        self.rows = 0
        self._lock = threading.Lock()

    def add(self, gradient: torch.Tensor) -> None:
        """Accumulate one routed-latent gradient batch without TF32."""

        if gradient.device != self.device or not gradient.is_floating_point():
            raise ValueError("routed-output gradient is on the wrong device or dtype")
        if gradient.shape[-1] != self.dimension:
            raise ValueError("routed-output gradient has the wrong dimension")
        if torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError(
                "output-factor capture requires TF32 matmul to be disabled "
                "before worker threads start"
            )
        rows = gradient.detach().reshape(-1, self.dimension).float()
        self.sum.addmm_(rows.T, rows)
        with self._lock:
            self.rows += int(rows.shape[0])

    def mean(self, *, damping_ratio: float = 0.0) -> torch.Tensor:
        """Return the symmetric mean factor with trace-scaled damping."""

        if self.rows <= 0:
            raise RuntimeError("output factor contains no gradient rows")
        if not math.isfinite(damping_ratio) or damping_ratio < 0.0:
            raise ValueError("damping ratio must be finite and nonnegative")
        factor = self.sum / float(self.rows)
        factor = (factor + factor.T) * 0.5
        if damping_ratio:
            diagonal_mean = torch.diagonal(factor).mean()
            factor = factor.clone()
            factor.diagonal().add_(float(damping_ratio) * diagonal_mean)
        if not bool(torch.all(torch.isfinite(factor))):
            raise FloatingPointError("output factor contains non-finite values")
        return factor.contiguous()

    def cpu_sum(self) -> torch.Tensor:
        """Return the undamped outer-product sum after queued work completes."""

        return self.sum.detach().cpu().contiguous()


@dataclass(frozen=True)
class SegmentReplayResult:
    """Cotangents produced by one complete decoder segment."""

    input_gradient: torch.Tensor
    residual_gradients: tuple[torch.Tensor, ...]
    output: torch.Tensor


@dataclass(frozen=True)
class MultiChannelSegmentReplayResult:
    """Independent VJPs evaluated against one decoder forward graph."""

    channels: dict[str, SegmentReplayResult]
    output: torch.Tensor


@dataclass(frozen=True)
class ReverseLayerRecord:
    """Materialization and factor support for one replayed decoder layer."""

    layer: int
    device: str
    load_seconds: float
    split_a_rows: int
    split_b_rows: int
    load_receipt: object | None


@dataclass(frozen=True)
class ReverseSegmentRecord:
    """Measured work for one committed reverse-replay segment."""

    first_layer: int
    end_layer: int
    documents: int
    tokens: int
    elapsed_seconds: float
    layers: tuple[ReverseLayerRecord, ...]


@dataclass(frozen=True)
class KimiReversePipelineResult:
    """Committed output factors and cotangent state for every decoder layer."""

    segments: tuple[ReverseSegmentRecord, ...]
    elapsed_seconds: float


@dataclass
class _LoadedReverseLayer:
    layer: int
    device: torch.device
    module: Any
    receipt: object | None
    load_seconds: float
    split_a: EmpiricalOutputFactor | None = None
    split_b: EmpiricalOutputFactor | None = None

    def select_split(self, split: str) -> EmpiricalOutputFactor:
        value = self.split_a if split == "a" else self.split_b
        if value is None:
            raise RuntimeError(f"decoder layer {self.layer} has no routed output factor")
        return value


def _validate_cuda_tensor(
    value: torch.Tensor,
    *,
    name: str,
    dimension: int,
    check_finite: bool,
) -> None:
    if value.device.type != "cuda" or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point CUDA tensor")
    if value.shape[-1] != dimension:
        raise ValueError(f"{name} has the wrong hidden dimension")
    if check_finite and not bool(torch.all(torch.isfinite(value))):
        raise FloatingPointError(f"{name} contains non-finite values")


def replay_decoder_segment_channels(
    *,
    adapter: KimiReversePipelineAdapter,
    modules: Sequence[Any],
    first_layer: int,
    input_hidden: torch.Tensor,
    residual_inputs: Sequence[torch.Tensor],
    output_gradients: Mapping[str, torch.Tensor],
    devices: Sequence[torch.device | str],
    residual_block_size: int,
    select_channel: Callable[[str | None], None] | None = None,
    validate_numerics: bool = True,
) -> MultiChannelSegmentReplayResult:
    """Recompute one aligned segment and evaluate multiple exact VJPs.

    ``input_hidden`` has shape ``[1, tokens, hidden]``. Every residual input has
    shape ``[tokens, hidden]`` and represents a preceding decoder boundary.
    The number of residual leaves must equal ``first_layer / block_size``.
    The segment may be shorter only at the end of the model.

    Every output cotangent is differentiated through the same forward graph.
    ``select_channel`` runs immediately before each VJP so dynamically
    dispatched hooks can accumulate channel-specific Fisher factors or
    objective gradients without recomputing decoder activations.
    """

    if first_layer < 0 or residual_block_size <= 0:
        raise ValueError("segment layer and residual block size are invalid")
    if first_layer % residual_block_size:
        raise ValueError("segment must begin on an attention-residual boundary")
    if not modules or len(modules) != len(devices):
        raise ValueError("every segment module requires one CUDA device")
    if len(modules) > residual_block_size:
        raise ValueError("segment cannot cross an attention-residual boundary")
    normalized = tuple(torch.device(value) for value in devices)
    if any(value.type != "cuda" or value.index is None for value in normalized):
        raise ValueError("segment devices must be indexed CUDA devices")
    if len(set(normalized)) != len(normalized):
        raise ValueError("segment devices must be unique")
    if not output_gradients:
        raise ValueError("segment replay requires at least one output-gradient channel")
    for channel in output_gradients:
        if not isinstance(channel, str) or not channel:
            raise ValueError("output-gradient channel names must be nonempty strings")
    if input_hidden.ndim != 3 or input_hidden.shape[0] != 1:
        raise ValueError("segment input must have shape [1, tokens, hidden]")
    hidden_dimension = int(input_hidden.shape[-1])
    token_count = int(input_hidden.shape[1])
    _validate_cuda_tensor(
        input_hidden,
        name="segment input",
        dimension=hidden_dimension,
        check_finite=validate_numerics,
    )
    if input_hidden.device != normalized[0]:
        raise ValueError("segment input must begin on the first device")
    expected_residuals = len(range(0, first_layer, residual_block_size))
    if len(residual_inputs) != expected_residuals:
        raise ValueError(
            f"segment requires {expected_residuals} residual inputs, "
            f"received {len(residual_inputs)}"
        )
    for index, value in enumerate(residual_inputs):
        if tuple(value.shape) != (token_count, hidden_dimension):
            raise ValueError(f"residual input {index} has the wrong shape")
        _validate_cuda_tensor(
            value,
            name=f"residual input {index}",
            dimension=hidden_dimension,
            check_finite=validate_numerics,
        )
        if value.device != normalized[0]:
            raise ValueError("residual inputs must begin on the first device")

    hidden_leaf = input_hidden.detach().requires_grad_(True)
    residual_leaves = tuple(
        value.detach().requires_grad_(True) for value in residual_inputs
    )
    if residual_leaves:
        preceding_residual = torch.stack(residual_leaves, dim=1)
    else:
        preceding_residual = torch.empty(
            (token_count, 0, hidden_dimension),
            dtype=input_hidden.dtype,
            device=normalized[0],
        )
    # The first aligned layer appends the segment input to the persistent
    # residual prefix. Broadcast that immutable prefix to every later stage up
    # front, allowing the live hidden activation alone to traverse the device
    # pipeline. Autograd retains the copy edges back to the segment leaves.
    complete_residual = torch.cat(
        (preceding_residual, hidden_leaf.reshape(-1, hidden_dimension).unsqueeze(1)),
        dim=1,
    )
    residual_by_stage = [preceding_residual]
    residual_by_stage.extend(
        complete_residual.to(device=device, non_blocking=True)
        for device in normalized[1:]
    )
    hidden = hidden_leaf
    with torch.enable_grad():
        for stage, (module, device) in enumerate(zip(modules, normalized)):
            if hidden.device != device:
                hidden = hidden.to(device=device, non_blocking=True)
            hidden, observed_residual = adapter.forward_layer(
                module,
                layer=first_layer + stage,
                hidden_states=hidden,
                block_residual=residual_by_stage[stage],
            )
            if observed_residual.shape[1] != complete_residual.shape[1]:
                raise ValueError("segment layer returned the wrong residual prefix")

        results: dict[str, SegmentReplayResult] = {}
        channel_items = tuple(output_gradients.items())
        try:
            for index, (channel, output_gradient) in enumerate(channel_items):
                if output_gradient.shape != hidden.shape:
                    raise ValueError(
                        f"segment output gradient for {channel!r} has the wrong shape"
                    )
                if output_gradient.device != hidden.device:
                    output_gradient = output_gradient.to(
                        hidden.device, non_blocking=True
                    )
                _validate_cuda_tensor(
                    output_gradient,
                    name=f"segment output gradient for {channel!r}",
                    dimension=hidden_dimension,
                    check_finite=validate_numerics,
                )
                if select_channel is not None:
                    select_channel(channel)
                gradients = torch.autograd.grad(
                    hidden,
                    (hidden_leaf, *residual_leaves),
                    grad_outputs=output_gradient,
                    allow_unused=False,
                    retain_graph=index + 1 < len(channel_items),
                )
                input_gradient = gradients[0]
                residual_gradients = tuple(gradients[1:])
                if validate_numerics:
                    if not bool(torch.all(torch.isfinite(input_gradient))):
                        raise FloatingPointError(
                            f"segment input gradient for {channel!r} is non-finite"
                        )
                    if not all(
                        bool(torch.all(torch.isfinite(value)))
                        for value in residual_gradients
                    ):
                        raise FloatingPointError(
                            f"segment residual gradient for {channel!r} is non-finite"
                        )
                results[channel] = SegmentReplayResult(
                    input_gradient=input_gradient,
                    residual_gradients=residual_gradients,
                    output=hidden.detach(),
                )
        finally:
            if select_channel is not None:
                select_channel(None)
    return MultiChannelSegmentReplayResult(
        channels=results,
        output=hidden.detach(),
    )


def replay_decoder_segment(
    *,
    adapter: KimiReversePipelineAdapter,
    modules: Sequence[Any],
    first_layer: int,
    input_hidden: torch.Tensor,
    residual_inputs: Sequence[torch.Tensor],
    output_gradient: torch.Tensor,
    devices: Sequence[torch.device | str],
    residual_block_size: int,
    validate_numerics: bool = True,
) -> SegmentReplayResult:
    """Recompute one aligned segment and return one exact input VJP."""

    result = replay_decoder_segment_channels(
        adapter=adapter,
        modules=modules,
        first_layer=first_layer,
        input_hidden=input_hidden,
        residual_inputs=residual_inputs,
        output_gradients={"default": output_gradient},
        devices=devices,
        residual_block_size=residual_block_size,
        validate_numerics=validate_numerics,
    )
    return result.channels["default"]


class KimiReversePipeline:
    """Replay official decoder segments and capture final-output curvature.

    Each committed segment advances the double-buffered cotangent workspace
    exactly one attention-residual block toward decoder input. The matching
    routed-output factors remain pending until that cotangent commit succeeds,
    so interruption cannot publish factors from an unusable reverse state.
    """

    def __init__(
        self,
        *,
        adapter: KimiReversePipelineAdapter,
        boundary_archive: KimiBoundarySlabArchive,
        cotangent_workspace: KimiCotangentSlabWorkspace,
        output_factors: KimiOutputFactorArchive,
        devices: Sequence[torch.device | str],
        slab_buffer_tokens: int = 256,
        direct_io: bool = True,
        validate_numerics: bool = True,
    ):
        normalized = tuple(torch.device(value) for value in devices)
        if not normalized or any(
            value.type != "cuda" or value.index is None for value in normalized
        ):
            raise ValueError("reverse replay requires indexed CUDA devices")
        if len(set(normalized)) != len(normalized):
            raise ValueError("reverse replay devices must be unique")
        if len(normalized) < boundary_archive.attn_res_block_size:
            raise ValueError(
                "reverse replay requires one device per attention-residual layer"
            )
        if slab_buffer_tokens <= 0:
            raise ValueError("cotangent slab buffer size must be positive")
        if not boundary_archive.complete:
            raise ValueError("reverse replay requires a sealed boundary archive")
        if (
            cotangent_workspace.token_count != boundary_archive.token_count
            or cotangent_workspace.hidden_dimension != boundary_archive.hidden_dimension
            or cotangent_workspace.num_layers != boundary_archive.num_layers
            or cotangent_workspace.residual_block_size
            != boundary_archive.attn_res_block_size
        ):
            raise ValueError("cotangent workspace and boundary archive disagree")
        if (
            output_factors.num_layers != boundary_archive.num_layers
        ):
            raise ValueError("output-factor archive and boundary archive disagree")
        chain_boundary = cotangent_workspace.manifest.get("chain_boundary")
        if chain_boundary is None:
            raise ValueError("final-output suffix cotangents are not initialized")
        if not 0 <= int(chain_boundary) <= boundary_archive.num_layers:
            raise ValueError("cotangent chain boundary is invalid")
        self.adapter = adapter
        self.boundaries = boundary_archive
        self.workspace = cotangent_workspace
        self.output_factors = output_factors
        self.devices = normalized
        self.slab_buffer_tokens = int(slab_buffer_tokens)
        self.direct_io = bool(direct_io)
        self.validate_numerics = bool(validate_numerics)
        self.documents = boundary_archive.load_documents()

    def _load_one(self, layer: int, device: torch.device) -> _LoadedReverseLayer:
        started = time.monotonic()
        module, receipt = self.adapter.load_layer(layer, device)
        return _LoadedReverseLayer(
            layer=layer,
            device=device,
            module=module,
            receipt=receipt,
            load_seconds=time.monotonic() - started,
        )

    def _load_segment(
        self,
        first_layer: int,
        end_layer: int,
    ) -> list[_LoadedReverseLayer]:
        layers = tuple(range(first_layer, end_layer))
        with ThreadPoolExecutor(
            max_workers=len(layers),
            thread_name_prefix=f"kimi-reverse-load-{first_layer:03d}",
        ) as executor:
            futures = [
                executor.submit(self._load_one, layer, device)
                for layer, device in zip(layers, self.devices, strict=False)
            ]
            loaded = [future.result() for future in futures]
        expected = set(self.output_factors.expected_layers)
        for item in loaded:
            enabled = self.adapter.enable_routed_output_gradients(
                item.module,
                None,
            )
            if enabled != (item.layer in expected):
                raise ValueError(
                    f"routed-output factor contract differs at decoder layer {item.layer}"
                )
            if enabled:
                item.split_a = EmpiricalOutputFactor(
                    self.output_factors.dimension,
                    device=item.device,
                )
                item.split_b = EmpiricalOutputFactor(
                    self.output_factors.dimension,
                    device=item.device,
                )
        return loaded

    def _release_segment(self, loaded: Sequence[_LoadedReverseLayer]) -> None:
        for item in loaded:
            torch.cuda.set_device(item.device)
            self.adapter.release_layer(item.module)
            item.module = None
        for device in {item.device for item in loaded}:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    def _cuda_writer(self, update, role: str) -> CudaBf16SlabWriter:
        return CudaBf16SlabWriter(
            update.writer(role, direct=self.direct_io),
            device=self.devices[0],
            buffer_tokens=self.slab_buffer_tokens,
        )

    def _run_segment(self, first_layer: int) -> ReverseSegmentRecord:
        end_layer = min(
            first_layer + self.boundaries.attn_res_block_size,
            self.boundaries.num_layers,
        )
        started = time.monotonic()
        loaded = self._load_segment(first_layer, end_layer)
        modules = tuple(item.module for item in loaded)
        devices = tuple(item.device for item in loaded)
        factor_writer = self.output_factors.begin_segment(first_layer, end_layer)
        cotangent_update = self.workspace.begin_segment(first_layer)
        cotangent_writers = {
            role: self._cuda_writer(cotangent_update, role)
            for role in cotangent_update.roles
        }
        residual_boundaries = self.boundaries.residual_boundaries_before(first_layer)
        original_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for document in range(self.documents.document_count):
                first_token, end_token = self.documents.document_extent(document)
                split = document_factor_split(self.documents.identifiers[document])
                for item in loaded:
                    if item.split_a is not None:
                        accumulator = item.select_split(split)
                        self.adapter.enable_routed_output_gradients(
                            item.module,
                            accumulator.add,
                        )

                input_cpu = self.boundaries.read_cpu(
                    first_layer,
                    first_token,
                    end_token,
                    direct=self.direct_io,
                    pin_memory=True,
                )
                residual_cpu = [
                    self.boundaries.read_cpu(
                        boundary,
                        first_token,
                        end_token,
                        direct=self.direct_io,
                        pin_memory=True,
                    )
                    for boundary in residual_boundaries
                ]
                chain_cpu = self.workspace.read_chain(
                    first_token,
                    end_token,
                    direct=self.direct_io,
                    pin_memory=True,
                )
                input_residual_cpu = self.workspace.read_residual(
                    first_layer,
                    first_token,
                    end_token,
                    direct=self.direct_io,
                    pin_memory=True,
                )
                earlier_cotangent_cpu = [
                    self.workspace.read_residual(
                        boundary,
                        first_token,
                        end_token,
                        direct=self.direct_io,
                        pin_memory=True,
                    )
                    for boundary in residual_boundaries
                ]

                first_device = devices[0]
                with torch.cuda.device(first_device):
                    input_hidden = input_cpu.to(
                        device=first_device,
                        non_blocking=True,
                    ).unsqueeze(0)
                    residual_inputs = tuple(
                        value.to(device=first_device, non_blocking=True)
                        for value in residual_cpu
                    )
                result = replay_decoder_segment(
                    adapter=self.adapter,
                    modules=modules,
                    first_layer=first_layer,
                    input_hidden=input_hidden,
                    residual_inputs=residual_inputs,
                    output_gradient=chain_cpu.unsqueeze(0),
                    devices=devices,
                    residual_block_size=self.boundaries.attn_res_block_size,
                    validate_numerics=self.validate_numerics,
                )
                input_residual = input_residual_cpu.to(
                    device=first_device,
                    non_blocking=True,
                )
                next_chain = (
                    result.input_gradient.reshape(-1, self.boundaries.hidden_dimension).float()
                    + input_residual.float()
                ).to(torch.bfloat16)
                cotangent_writers["chain"].append(next_chain)
                for boundary, contribution, existing in zip(
                    residual_boundaries,
                    result.residual_gradients,
                    earlier_cotangent_cpu,
                    strict=True,
                ):
                    updated = (
                        contribution.float()
                        + existing.to(device=first_device, non_blocking=True).float()
                    ).to(torch.bfloat16)
                    cotangent_writers[
                        self.workspace.residual_role(boundary)
                    ].append(updated)

            layer_records: list[ReverseLayerRecord] = []
            for item in loaded:
                if item.split_a is None or item.split_b is None:
                    continue
                sums = OutputFactorSums(
                    split_a=item.split_a.cpu_sum(),
                    split_a_rows=item.split_a.rows,
                    split_b=item.split_b.cpu_sum(),
                    split_b_rows=item.split_b.rows,
                )
                factor_writer.add(item.layer, sums)
                layer_records.append(
                    ReverseLayerRecord(
                        layer=item.layer,
                        device=str(item.device),
                        load_seconds=item.load_seconds,
                        split_a_rows=item.split_a.rows,
                        split_b_rows=item.split_b.rows,
                        load_receipt=item.receipt,
                    )
                )
            factor_writer.seal()
            for role, writer in cotangent_writers.items():
                cotangent_update.record(writer.finish())
            cotangent_update.commit()
            factor_writer.commit()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = original_tf32
            for writer in cotangent_writers.values():
                writer.close()
            self._release_segment(loaded)
        return ReverseSegmentRecord(
            first_layer=first_layer,
            end_layer=end_layer,
            documents=self.documents.document_count,
            tokens=self.documents.token_count,
            elapsed_seconds=time.monotonic() - started,
            layers=tuple(layer_records),
        )

    def run(self) -> KimiReversePipelineResult:
        """Resume at the durable cotangent boundary and replay to decoder input."""

        started = time.monotonic()
        completed = tuple(
            str(value["operation"])
            for value in self.workspace.manifest.get("completed_operations", [])
        )
        self.output_factors.discard_uncommitted_pending(completed)
        self.output_factors.recover_pending(completed)
        records: list[ReverseSegmentRecord] = []
        while int(self.workspace.manifest["chain_boundary"]) > 0:
            end_layer = int(self.workspace.manifest["chain_boundary"])
            first_layer = (
                (end_layer - 1) // self.boundaries.attn_res_block_size
            ) * self.boundaries.attn_res_block_size
            record = self._run_segment(first_layer)
            records.append(record)
        self.output_factors.seal()
        return KimiReversePipelineResult(
            segments=tuple(records),
            elapsed_seconds=time.monotonic() - started,
        )


__all__ = [
    "EmpiricalOutputFactor",
    "KimiReversePipeline",
    "KimiReversePipelineResult",
    "KimiReversePipelineAdapter",
    "MultiChannelSegmentReplayResult",
    "ReverseLayerRecord",
    "ReverseSegmentRecord",
    "SegmentReplayResult",
    "replay_decoder_segment",
    "replay_decoder_segment_channels",
]

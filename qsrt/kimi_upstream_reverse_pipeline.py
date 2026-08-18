"""Distributed reverse replay for coupled-basis W1/W3 Fisher factors.

Each routed decoder layer owns one expert-static coupled Hadamard draw table.
Reverse replay differentiates the official model's final-logit Fisher samples
through a 12-layer decoder segment, transforms every gate/up cotangent into
the exact stored expert coordinates, and accumulates 128-coordinate output
blocks. Factor publication is tied to the matching cotangent-segment commit so
an interrupted run can resume without mixing incompatible reverse states.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from qsrt.kimi_boundary_slabs import KimiBoundarySlabArchive
from qsrt.kimi_cotangent_slabs import (
    CudaBf16SlabWriter,
    KimiCotangentSlabWorkspace,
)
from qsrt.kimi_expert_reverse import (
    CoupledUpstreamReverseAccumulator,
    CoupledUpstreamReverseConfig,
    ExpertReverseChannelRouter,
)
from qsrt.kimi_reverse_pipeline import (
    KimiReversePipelineAdapter,
    replay_decoder_segment_channels,
)
from qsrt.kimi_upstream_factors import (
    KimiUpstreamFactorArchive,
    UpstreamFactorSums,
)
from qsrt.qsrt_coupled import CoupledHadamardSpec


@dataclass(frozen=True)
class UpstreamReverseLayerRecord:
    """Materialization and Fisher support for one routed decoder layer."""

    layer: int
    device: str
    load_seconds: float
    supported_experts: int
    gradient_rows: int
    accumulator_bytes: int
    load_receipt: object | None
    worker_seconds: float = 0.0
    queue_wait_seconds: float = 0.0
    boundary_read_seconds: float = 0.0
    host_dispatch_seconds: float = 0.0
    gpu_active_seconds: float = 0.0


@dataclass(frozen=True)
class UpstreamReverseSegmentRecord:
    """Measured work for one committed upstream-Fisher segment."""

    first_layer: int
    end_layer: int
    documents: int
    tokens: int
    elapsed_seconds: float
    layers: tuple[UpstreamReverseLayerRecord, ...]
    timings: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class KimiUpstreamReversePipelineResult:
    """Committed coupled-basis factors and cotangent state."""

    segments: tuple[UpstreamReverseSegmentRecord, ...]
    elapsed_seconds: float


@dataclass
class _LoadedUpstreamLayer:
    layer: int
    device: torch.device
    module: Any
    receipt: object | None
    load_seconds: float
    accumulator: CoupledUpstreamReverseAccumulator | None = None
    router: ExpertReverseChannelRouter | None = None


class KimiUpstreamReversePipeline:
    """Replay final-logit Fisher cotangents into W1/W3 stored coordinates."""

    def __init__(
        self,
        *,
        adapter: KimiReversePipelineAdapter,
        boundary_archive: KimiBoundarySlabArchive,
        cotangent_workspace: KimiCotangentSlabWorkspace,
        upstream_factors: KimiUpstreamFactorArchive,
        intermediate_draws: Mapping[int, Sequence[int]],
        devices: Sequence[torch.device | str],
        slab_buffer_tokens: int = 256,
        direct_io: bool = True,
        validate_numerics: bool = True,
        gradient_sketch_seed: int = 0,
        coupled_spec: CoupledHadamardSpec = CoupledHadamardSpec(),
    ):
        normalized = tuple(torch.device(value) for value in devices)
        if not normalized or any(
            value.type != "cuda" or value.index is None for value in normalized
        ):
            raise ValueError("upstream reverse replay requires indexed CUDA devices")
        if len(set(normalized)) != len(normalized):
            raise ValueError("upstream reverse replay devices must be unique")
        if len(normalized) < boundary_archive.attn_res_block_size:
            raise ValueError(
                "upstream reverse replay requires one device per residual-block layer"
            )
        if slab_buffer_tokens <= 0:
            raise ValueError("cotangent slab buffer size must be positive")
        if not boundary_archive.complete:
            raise ValueError("upstream reverse replay requires sealed boundaries")
        if (
            cotangent_workspace.token_count != boundary_archive.token_count
            or cotangent_workspace.hidden_dimension
            != boundary_archive.hidden_dimension
            or cotangent_workspace.num_layers != boundary_archive.num_layers
            or cotangent_workspace.residual_block_size
            != boundary_archive.attn_res_block_size
        ):
            raise ValueError("cotangent workspace and boundary archive disagree")
        if upstream_factors.num_layers != boundary_archive.num_layers:
            raise ValueError("upstream-factor and boundary layer counts disagree")
        chain_boundary = cotangent_workspace.manifest.get("chain_boundary")
        if chain_boundary is None:
            raise ValueError("final-output Fisher cotangents are not initialized")
        if not 0 <= int(chain_boundary) <= boundary_archive.num_layers:
            raise ValueError("cotangent chain boundary is invalid")

        expected = set(upstream_factors.expected_layers)
        supplied = {int(layer) for layer in intermediate_draws}
        if supplied != expected:
            raise ValueError(
                "coupled draw inventory differs from routed layers: "
                f"missing={sorted(expected - supplied)}, "
                f"unexpected={sorted(supplied - expected)}"
            )
        normalized_draws: dict[int, tuple[int, ...]] = {}
        for layer, values in intermediate_draws.items():
            draws = tuple(int(value) for value in values)
            if len(draws) != upstream_factors.num_experts or any(
                not 0 <= draw < 8 for draw in draws
            ):
                raise ValueError(
                    f"decoder layer {layer} lacks one valid draw per routed expert"
                )
            normalized_draws[int(layer)] = draws

        self.adapter = adapter
        self.boundaries = boundary_archive
        self.workspace = cotangent_workspace
        self.upstream_factors = upstream_factors
        self.intermediate_draws = normalized_draws
        self.devices = normalized
        self.slab_buffer_tokens = int(slab_buffer_tokens)
        self.direct_io = bool(direct_io)
        self.validate_numerics = bool(validate_numerics)
        self.gradient_sketch_seed = int(gradient_sketch_seed)
        self.coupled_spec = coupled_spec
        self.documents = boundary_archive.load_documents()

    def _load_one(self, layer: int, device: torch.device) -> _LoadedUpstreamLayer:
        started = time.monotonic()
        module, receipt = self.adapter.load_layer(layer, device)
        return _LoadedUpstreamLayer(
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
    ) -> list[_LoadedUpstreamLayer]:
        layers = tuple(range(first_layer, end_layer))
        with ThreadPoolExecutor(
            max_workers=len(layers),
            thread_name_prefix=f"kimi-upstream-load-{first_layer:03d}",
        ) as executor:
            stage_devices = self.devices[: len(layers)]
            futures = [
                executor.submit(self._load_one, layer, device)
                for layer, device in zip(layers, stage_devices, strict=True)
            ]
            loaded = [future.result() for future in futures]

        expected = set(self.upstream_factors.expected_layers)
        for item in loaded:
            if item.layer not in expected:
                if self.adapter.enable_routed_output_gradients(item.module, None):
                    raise ValueError(
                        f"decoder layer {item.layer} is routed but has no factor target"
                    )
                continue
            config = CoupledUpstreamReverseConfig(
                num_experts=self.upstream_factors.num_experts,
                hidden_dimension=self.upstream_factors.hidden_dimension,
                intermediate_dimension=self.upstream_factors.intermediate_dimension,
                spec=self.coupled_spec,
                intermediate_draws=self.intermediate_draws[item.layer],
                output_factor_block_size=self.upstream_factors.block_size,
                gradient_sketch_rank=self.upstream_factors.gradient_rank,
                gradient_sketch_seed=(
                    self.gradient_sketch_seed ^ (item.layer * 0x9E3779B1)
                ),
            )
            item.accumulator = CoupledUpstreamReverseAccumulator(
                config,
                device=item.device,
                capture_fisher=True,
                capture_objective_gradient=False,
            )
            item.router = ExpertReverseChannelRouter(upstream=item.accumulator)
            item.router.install(self.adapter, item.module)
        return loaded

    def _release_segment(self, loaded: Sequence[_LoadedUpstreamLayer]) -> None:
        for item in loaded:
            torch.cuda.set_device(item.device)
            self.adapter.release_layer(item.module)
            item.module = None
            item.accumulator = None
            item.router = None
        for device in {item.device for item in loaded}:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    def _cuda_writer(self, update, role: str) -> CudaBf16SlabWriter:
        return CudaBf16SlabWriter(
            update.writer(role, direct=self.direct_io),
            device=self.devices[0],
            buffer_tokens=self.slab_buffer_tokens,
        )

    @staticmethod
    def _select_channel(
        loaded: Sequence[_LoadedUpstreamLayer], channel: str | None
    ) -> None:
        for item in loaded:
            if item.router is not None:
                item.router.select_channel(channel)

    def _run_segment(self, first_layer: int) -> UpstreamReverseSegmentRecord:
        end_layer = min(
            first_layer + self.boundaries.attn_res_block_size,
            self.boundaries.num_layers,
        )
        started = time.monotonic()
        loaded = self._load_segment(first_layer, end_layer)
        modules = tuple(item.module for item in loaded)
        devices = tuple(item.device for item in loaded)
        factor_writer = self.upstream_factors.begin_segment(first_layer, end_layer)
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
                result = replay_decoder_segment_channels(
                    adapter=self.adapter,
                    modules=modules,
                    first_layer=first_layer,
                    input_hidden=input_hidden,
                    residual_inputs=residual_inputs,
                    output_gradients={"fisher": chain_cpu.unsqueeze(0)},
                    devices=devices,
                    residual_block_size=self.boundaries.attn_res_block_size,
                    select_channel=lambda channel: self._select_channel(
                        loaded, channel
                    ),
                    validate_numerics=self.validate_numerics,
                ).channels["fisher"]
                input_residual = input_residual_cpu.to(
                    device=first_device,
                    non_blocking=True,
                )
                next_chain = (
                    result.input_gradient.reshape(
                        -1, self.boundaries.hidden_dimension
                    ).float()
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
                        + existing.to(
                            device=first_device,
                            non_blocking=True,
                        ).float()
                    ).to(torch.bfloat16)
                    cotangent_writers[
                        self.workspace.residual_role(boundary)
                    ].append(updated)

            records: list[UpstreamReverseLayerRecord] = []
            for item in loaded:
                accumulator = item.accumulator
                if accumulator is None:
                    continue
                assert accumulator.output_factor_sums is not None
                draws = torch.tensor(
                    self.intermediate_draws[item.layer], dtype=torch.uint8
                )
                sums = UpstreamFactorSums(
                    output_hessian_blocks=(
                        accumulator.output_factor_sums.detach().cpu().contiguous()
                    ),
                    output_hessian_rows=(
                        accumulator.output_factor_rows.detach().cpu().contiguous()
                    ),
                    intermediate_draws=draws,
                )
                factor_writer.add(item.layer, sums)
                records.append(
                    UpstreamReverseLayerRecord(
                        layer=item.layer,
                        device=str(item.device),
                        load_seconds=item.load_seconds,
                        supported_experts=int(
                            torch.count_nonzero(
                                accumulator.output_factor_rows
                            ).item()
                        ),
                        gradient_rows=int(
                            accumulator.output_factor_rows.sum().item()
                        ),
                        accumulator_bytes=accumulator.allocated_bytes(),
                        load_receipt=item.receipt,
                    )
                )
            factor_writer.seal()
            for writer in cotangent_writers.values():
                cotangent_update.record(writer.finish())
            cotangent_update.commit()
            factor_writer.commit()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = original_tf32
            for writer in cotangent_writers.values():
                writer.close()
            self._release_segment(loaded)
        return UpstreamReverseSegmentRecord(
            first_layer=first_layer,
            end_layer=end_layer,
            documents=self.documents.document_count,
            tokens=self.documents.token_count,
            elapsed_seconds=time.monotonic() - started,
            layers=tuple(records),
        )

    def run(self) -> KimiUpstreamReversePipelineResult:
        """Resume from the durable cotangent boundary and replay to layer zero."""

        started = time.monotonic()
        completed = tuple(
            str(value["operation"])
            for value in self.workspace.manifest.get("completed_operations", [])
        )
        self.upstream_factors.discard_uncommitted_pending(completed)
        self.upstream_factors.recover_pending(completed)
        records: list[UpstreamReverseSegmentRecord] = []
        while int(self.workspace.manifest["chain_boundary"]) > 0:
            end_layer = int(self.workspace.manifest["chain_boundary"])
            first_layer = (
                (end_layer - 1) // self.boundaries.attn_res_block_size
            ) * self.boundaries.attn_res_block_size
            records.append(self._run_segment(first_layer))
        self.upstream_factors.seal()
        return KimiUpstreamReversePipelineResult(
            segments=tuple(records),
            elapsed_seconds=time.monotonic() - started,
        )


__all__ = [
    "KimiUpstreamReversePipeline",
    "KimiUpstreamReversePipelineResult",
    "UpstreamReverseLayerRecord",
    "UpstreamReverseSegmentRecord",
]

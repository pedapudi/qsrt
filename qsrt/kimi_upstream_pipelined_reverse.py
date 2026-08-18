"""Document-pipelined reverse replay for coupled W1/W3 Fisher factors.

Every decoder layer in an attention-residual segment resides on a different
GPU.  Stored boundary activations let each layer recompute its local autograd
graph only after the matching output cotangent arrives.  Complete documents
therefore move backward through the devices as a bounded pipeline without
retaining more than one decoder graph per stage.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
from qsrt.kimi_reverse_pipeline import KimiReversePipelineAdapter
from qsrt.kimi_upstream_factors import (
    KimiUpstreamFactorArchive,
    UpstreamFactorSums,
)
from qsrt.kimi_upstream_reverse_pipeline import (
    KimiUpstreamReversePipelineResult,
    UpstreamReverseLayerRecord,
    UpstreamReverseSegmentRecord,
)
from qsrt.qsrt_coupled import CoupledHadamardSpec


@dataclass
class _ReverseDocument:
    document: int
    first_token: int
    end_token: int
    hidden_gradient: torch.Tensor
    residual_gradient: torch.Tensor
    objective_hidden_gradient: torch.Tensor | None = None
    objective_residual_gradient: torch.Tensor | None = None
    ready: torch.cuda.Event | None = None


@dataclass
class _LoadedLayer:
    layer: int
    device: torch.device
    module: Any
    receipt: object | None
    load_seconds: float
    accumulator: CoupledUpstreamReverseAccumulator | None = None
    router: ExpertReverseChannelRouter | None = None
    worker_seconds: float = 0.0
    queue_wait_seconds: float = 0.0
    boundary_read_seconds: float = 0.0
    host_dispatch_seconds: float = 0.0
    gpu_active_seconds: float = 0.0


@dataclass(frozen=True)
class _WorkerFailure:
    layer: int
    message: str
    traceback: str


class _PipelineStopped(RuntimeError):
    pass


class KimiPipelinedUpstreamReverse:
    """Replay one document per pipeline stage under bounded GPU memory."""

    def __init__(
        self,
        *,
        adapter: KimiReversePipelineAdapter,
        boundary_archive: KimiBoundarySlabArchive,
        cotangent_workspace: KimiCotangentSlabWorkspace,
        objective_workspace: KimiCotangentSlabWorkspace | None = None,
        upstream_factors: KimiUpstreamFactorArchive,
        intermediate_draws: Mapping[int, Sequence[int]],
        devices: Sequence[torch.device | str],
        queue_depth: int = 1,
        slab_buffer_tokens: int = 2048,
        direct_io: bool = True,
        validation_documents: int = 1,
        gradient_sketch_seed: int = 0,
        coupled_spec: CoupledHadamardSpec = CoupledHadamardSpec(),
    ):
        normalized = tuple(torch.device(value) for value in devices)
        if not normalized or any(
            value.type != "cuda" or value.index is None for value in normalized
        ):
            raise ValueError("reverse replay requires indexed CUDA devices")
        if len(set(normalized)) != len(normalized):
            raise ValueError("reverse replay devices must be unique")
        if len(normalized) < boundary_archive.attn_res_block_size:
            raise ValueError("one GPU is required per attention-residual layer")
        if queue_depth <= 0 or slab_buffer_tokens <= 0:
            raise ValueError("pipeline queue and slab buffers must be positive")
        if validation_documents < 0:
            raise ValueError("validation document count cannot be negative")
        if not boundary_archive.complete:
            raise ValueError("reverse replay requires sealed boundary activations")
        if (
            cotangent_workspace.token_count != boundary_archive.token_count
            or cotangent_workspace.hidden_dimension
            != boundary_archive.hidden_dimension
            or cotangent_workspace.num_layers != boundary_archive.num_layers
            or cotangent_workspace.residual_block_size
            != boundary_archive.attn_res_block_size
        ):
            raise ValueError("cotangent workspace and boundary archive disagree")
        chain_boundary = cotangent_workspace.manifest.get("chain_boundary")
        if chain_boundary is None or not 0 <= int(chain_boundary) <= boundary_archive.num_layers:
            raise ValueError("cotangent chain boundary is invalid")
        if objective_workspace is not None:
            if (
                objective_workspace.token_count != boundary_archive.token_count
                or objective_workspace.hidden_dimension
                != boundary_archive.hidden_dimension
                or objective_workspace.num_layers != boundary_archive.num_layers
                or objective_workspace.residual_block_size
                != boundary_archive.attn_res_block_size
            ):
                raise ValueError(
                    "objective workspace and boundary archive disagree"
                )
            objective_boundary = objective_workspace.manifest.get("chain_boundary")
            if objective_boundary != chain_boundary:
                raise ValueError(
                    "Fisher and objective cotangents are at different boundaries"
                )
        if upstream_factors.num_layers != boundary_archive.num_layers:
            raise ValueError("factor archive and boundary archive disagree")

        expected = set(upstream_factors.expected_layers)
        supplied = {int(layer) for layer in intermediate_draws}
        if supplied != expected:
            raise ValueError("coupled draw inventory differs from routed layers")
        draws: dict[int, tuple[int, ...]] = {}
        for layer, values in intermediate_draws.items():
            normalized_values = tuple(int(value) for value in values)
            if len(normalized_values) != upstream_factors.num_experts or any(
                not 0 <= value < 8 for value in normalized_values
            ):
                raise ValueError(f"decoder layer {layer} has invalid coupled draws")
            draws[int(layer)] = normalized_values

        self.adapter = adapter
        self.boundaries = boundary_archive
        self.workspace = cotangent_workspace
        self.objective_workspace = objective_workspace
        self.upstream_factors = upstream_factors
        self.intermediate_draws = draws
        self.devices = normalized
        self.queue_depth = int(queue_depth)
        self.slab_buffer_tokens = int(slab_buffer_tokens)
        self.direct_io = bool(direct_io)
        self.validation_documents = int(validation_documents)
        self.gradient_sketch_seed = int(gradient_sketch_seed)
        self.coupled_spec = coupled_spec
        self.documents = boundary_archive.load_documents()
        self._abort = threading.Event()
        self._failures: queue.Queue[_WorkerFailure] = queue.Queue()

    def _load_one(self, layer: int, device: torch.device) -> _LoadedLayer:
        started = time.monotonic()
        try:
            module, receipt = self.adapter.load_layer(layer, device)
        except BaseException as error:
            raise RuntimeError(
                f"failed to load decoder layer {layer} on {device}: "
                f"{type(error).__name__}: {error}"
            ) from error
        return _LoadedLayer(
            layer=layer,
            device=device,
            module=module,
            receipt=receipt,
            load_seconds=time.monotonic() - started,
        )

    def _load_segment(self, first_layer: int, end_layer: int) -> list[_LoadedLayer]:
        layers = tuple(range(first_layer, end_layer))
        devices = self.devices[: len(layers)]
        with ThreadPoolExecutor(
            max_workers=len(layers),
            thread_name_prefix=f"k3-reverse-load-{first_layer:03d}",
        ) as executor:
            futures = [
                executor.submit(self._load_one, layer, device)
                for layer, device in zip(layers, devices, strict=True)
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
                capture_objective_gradient=self.objective_workspace is not None,
            )
            item.router = ExpertReverseChannelRouter(upstream=item.accumulator)
            item.router.install(self.adapter, item.module)
            item.router.select_channel("fisher")
        return loaded

    def _release_segment(self, loaded: Sequence[_LoadedLayer]) -> None:
        for item in loaded:
            torch.cuda.set_device(item.device)
            self.adapter.release_layer(item.module)
            item.module = None
            item.accumulator = None
            item.router = None
        for device in {item.device for item in loaded}:
            with torch.cuda.device(device):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    def _put(self, target: queue.Queue[_ReverseDocument], item: _ReverseDocument) -> None:
        while not self._abort.is_set():
            try:
                target.put(item, timeout=0.1)
                return
            except queue.Full:
                continue
        raise _PipelineStopped("reverse pipeline stopped while publishing a document")

    def _get(self, source: queue.Queue[_ReverseDocument]) -> _ReverseDocument:
        while not self._abort.is_set():
            try:
                return source.get(timeout=0.1)
            except queue.Empty:
                continue
        raise _PipelineStopped("reverse pipeline stopped while awaiting a document")

    def _fail(self, layer: int, error: BaseException) -> None:
        if isinstance(error, _PipelineStopped) and self._abort.is_set():
            return
        self._failures.put(
            _WorkerFailure(
                layer=layer,
                message=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
        )
        self._abort.set()

    def _validate_document(
        self,
        item: _ReverseDocument,
        *,
        document: int,
        output_blocks: int,
    ) -> None:
        first, end = self.documents.document_extent(document)
        if (item.document, item.first_token, item.end_token) != (document, first, end):
            raise ValueError("reverse pipeline document order changed")
        expected_hidden = (1, end - first, self.boundaries.hidden_dimension)
        expected_residual = (
            end - first,
            output_blocks,
            self.boundaries.hidden_dimension,
        )
        if item.hidden_gradient.dtype != torch.bfloat16 or (
            tuple(item.hidden_gradient.shape) != expected_hidden
        ):
            raise ValueError("hidden cotangent has incompatible geometry")
        if item.residual_gradient.dtype != torch.bfloat16 or (
            tuple(item.residual_gradient.shape) != expected_residual
        ):
            raise ValueError("residual cotangent has incompatible geometry")
        objective_values = (
            item.objective_hidden_gradient,
            item.objective_residual_gradient,
        )
        if self.objective_workspace is None:
            if any(value is not None for value in objective_values):
                raise ValueError("unexpected objective cotangent channel")
        else:
            objective_hidden, objective_residual = objective_values
            if objective_hidden is None or objective_residual is None:
                raise ValueError("objective cotangent channel is incomplete")
            if objective_hidden.dtype != torch.bfloat16 or (
                tuple(objective_hidden.shape) != expected_hidden
            ):
                raise ValueError("objective hidden cotangent has incompatible geometry")
            if objective_residual.dtype != torch.bfloat16 or (
                tuple(objective_residual.shape) != expected_residual
            ):
                raise ValueError(
                    "objective residual cotangent has incompatible geometry"
                )

    @staticmethod
    def _finite(*values: torch.Tensor) -> bool:
        return all(bool(torch.all(torch.isfinite(value))) for value in values)

    def _read_residual_values(
        self,
        boundaries: Sequence[int],
        first_token: int,
        end_token: int,
    ) -> list[torch.Tensor]:
        return [
            self.boundaries.read_cpu(
                boundary,
                first_token,
                end_token,
                direct=self.direct_io,
                pin_memory=True,
            )
            for boundary in boundaries
        ]

    def _worker(
        self,
        *,
        item: _LoadedLayer,
        source: queue.Queue[_ReverseDocument],
        target: queue.Queue[_ReverseDocument],
    ) -> None:
        layer = item.layer
        device = item.device
        input_boundaries = self.boundaries.residual_boundaries_before(layer)
        input_blocks = len(input_boundaries)
        output_blocks = input_blocks + int(
            layer % self.boundaries.attn_res_block_size == 0
        )
        torch.cuda.set_device(device)
        stream = torch.cuda.Stream(device=device)
        worker_started = time.monotonic()
        gpu_ranges: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        try:
            for document in range(self.documents.document_count):
                wait_started = time.monotonic()
                incoming = self._get(source)
                item.queue_wait_seconds += time.monotonic() - wait_started
                self._validate_document(
                    incoming,
                    document=document,
                    output_blocks=output_blocks,
                )
                first, end = self.documents.document_extent(document)
                read_started = time.monotonic()
                hidden_cpu = self.boundaries.read_cpu(
                    layer,
                    first,
                    end,
                    direct=self.direct_io,
                    pin_memory=True,
                )
                residual_cpu = self._read_residual_values(
                    input_boundaries,
                    first,
                    end,
                )
                item.boundary_read_seconds += time.monotonic() - read_started
                dispatch_started = time.monotonic()
                with torch.cuda.device(device), torch.cuda.stream(stream):
                    if incoming.ready is not None:
                        stream.wait_event(incoming.ready)
                    gpu_started = torch.cuda.Event(enable_timing=True)
                    gpu_finished = torch.cuda.Event(enable_timing=True)
                    gpu_started.record(stream)
                    hidden_gradient = incoming.hidden_gradient.to(
                        device=device,
                        non_blocking=True,
                    )
                    residual_gradient = incoming.residual_gradient.to(
                        device=device,
                        non_blocking=True,
                    )
                    objective_hidden_gradient = (
                        None
                        if incoming.objective_hidden_gradient is None
                        else incoming.objective_hidden_gradient.to(
                            device=device,
                            non_blocking=True,
                        )
                    )
                    objective_residual_gradient = (
                        None
                        if incoming.objective_residual_gradient is None
                        else incoming.objective_residual_gradient.to(
                            device=device,
                            non_blocking=True,
                        )
                    )
                    incoming.hidden_gradient.record_stream(stream)
                    incoming.residual_gradient.record_stream(stream)
                    if incoming.objective_hidden_gradient is not None:
                        incoming.objective_hidden_gradient.record_stream(stream)
                    if incoming.objective_residual_gradient is not None:
                        incoming.objective_residual_gradient.record_stream(stream)
                    hidden_source = hidden_cpu.to(
                        device=device,
                        non_blocking=True,
                    )
                    if residual_cpu:
                        residual_source = torch.stack(
                            [
                                value.to(device=device, non_blocking=True)
                                for value in residual_cpu
                            ],
                            dim=1,
                        )
                    else:
                        residual_source = torch.empty(
                            (
                                end - first,
                                0,
                                self.boundaries.hidden_dimension,
                            ),
                            dtype=torch.bfloat16,
                            device=device,
                        )

                    def reverse_channel(
                        channel: str,
                        channel_hidden_gradient: torch.Tensor,
                        channel_residual_gradient: torch.Tensor,
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                        hidden_leaf = (
                            hidden_source.unsqueeze(0)
                            .detach()
                            .requires_grad_(True)
                        )
                        residual_leaf = residual_source.detach().requires_grad_(True)
                        if item.router is not None:
                            item.router.select_channel(channel)
                        with torch.enable_grad():
                            output, output_residual = self.adapter.forward_layer(
                                item.module,
                                layer=layer,
                                hidden_states=hidden_leaf,
                                block_residual=residual_leaf,
                            )
                        if tuple(output_residual.shape) != (
                            end - first,
                            output_blocks,
                            self.boundaries.hidden_dimension,
                        ):
                            raise ValueError(
                                "decoder layer returned the wrong residual prefix"
                            )
                        input_gradients = torch.autograd.grad(
                            (output, output_residual),
                            (hidden_leaf, residual_leaf),
                            grad_outputs=(
                                channel_hidden_gradient,
                                channel_residual_gradient,
                            ),
                            allow_unused=False,
                        )
                        return output, input_gradients[0], input_gradients[1]

                    output, hidden_input_gradient, residual_input_gradient = (
                        reverse_channel(
                            "fisher",
                            hidden_gradient,
                            residual_gradient,
                        )
                    )
                    objective_hidden_input_gradient = None
                    objective_residual_input_gradient = None
                    if objective_hidden_gradient is not None:
                        assert objective_residual_gradient is not None
                        (
                            _objective_output,
                            objective_hidden_input_gradient,
                            objective_residual_input_gradient,
                        ) = reverse_channel(
                            "objective",
                            objective_hidden_gradient,
                            objective_residual_gradient,
                        )
                    if item.router is not None:
                        item.router.select_channel(None)
                    if document < self.validation_documents and not self._finite(
                        output,
                        hidden_input_gradient,
                        residual_input_gradient,
                    ):
                        raise FloatingPointError(
                            f"decoder layer {layer} produced non-finite reverse values"
                        )
                    hidden_input_gradient = hidden_input_gradient.to(torch.bfloat16)
                    residual_input_gradient = residual_input_gradient.to(torch.bfloat16)
                    if objective_hidden_input_gradient is not None:
                        assert objective_residual_input_gradient is not None
                        if document < self.validation_documents and not self._finite(
                            objective_hidden_input_gradient,
                            objective_residual_input_gradient,
                        ):
                            raise FloatingPointError(
                                f"decoder layer {layer} produced non-finite objective "
                                "reverse values"
                            )
                        objective_hidden_input_gradient = (
                            objective_hidden_input_gradient.to(torch.bfloat16)
                        )
                        objective_residual_input_gradient = (
                            objective_residual_input_gradient.to(torch.bfloat16)
                        )
                    gpu_finished.record(stream)
                    gpu_ranges.append((gpu_started, gpu_finished))
                    ready = torch.cuda.Event()
                    ready.record(stream)
                item.host_dispatch_seconds += time.monotonic() - dispatch_started
                publish_started = time.monotonic()
                self._put(
                    target,
                    _ReverseDocument(
                        document=document,
                        first_token=first,
                        end_token=end,
                        hidden_gradient=hidden_input_gradient,
                        residual_gradient=residual_input_gradient,
                        objective_hidden_gradient=(
                            objective_hidden_input_gradient
                        ),
                        objective_residual_gradient=(
                            objective_residual_input_gradient
                        ),
                        ready=ready,
                    ),
                )
                item.queue_wait_seconds += time.monotonic() - publish_started
            if gpu_ranges:
                gpu_ranges[-1][1].synchronize()
                item.gpu_active_seconds = sum(
                    started.elapsed_time(finished) for started, finished in gpu_ranges
                ) / 1000.0
            item.worker_seconds = time.monotonic() - worker_started
        except BaseException as error:
            self._fail(layer, error)

    def _feed(
        self,
        *,
        end_layer: int,
        target: queue.Queue[_ReverseDocument],
        device: torch.device,
    ) -> None:
        output_boundaries = tuple(
            range(0, end_layer, self.boundaries.attn_res_block_size)
        )
        torch.cuda.set_device(device)
        stream = torch.cuda.Stream(device=device)
        try:
            for document in range(self.documents.document_count):
                first, end = self.documents.document_extent(document)
                hidden_cpu = self.workspace.read_chain(
                    first,
                    end,
                    direct=self.direct_io,
                    pin_memory=True,
                )
                residual_cpu = [
                    self.workspace.read_residual(
                        boundary,
                        first,
                        end,
                        direct=self.direct_io,
                        pin_memory=True,
                    )
                    for boundary in output_boundaries
                ]
                objective_hidden_cpu = (
                    None
                    if self.objective_workspace is None
                    else self.objective_workspace.read_chain(
                        first,
                        end,
                        direct=self.direct_io,
                        pin_memory=True,
                    )
                )
                objective_residual_cpu = (
                    []
                    if self.objective_workspace is None
                    else [
                        self.objective_workspace.read_residual(
                            boundary,
                            first,
                            end,
                            direct=self.direct_io,
                            pin_memory=True,
                        )
                        for boundary in output_boundaries
                    ]
                )
                with torch.cuda.device(device), torch.cuda.stream(stream):
                    hidden = hidden_cpu.to(device=device, non_blocking=True).unsqueeze(0)
                    if residual_cpu:
                        residual = torch.stack(
                            [
                                value.to(device=device, non_blocking=True)
                                for value in residual_cpu
                            ],
                            dim=1,
                        )
                    else:
                        residual = torch.empty(
                            (end - first, 0, self.boundaries.hidden_dimension),
                            dtype=torch.bfloat16,
                            device=device,
                        )
                    objective_hidden = (
                        None
                        if objective_hidden_cpu is None
                        else objective_hidden_cpu.to(
                            device=device,
                            non_blocking=True,
                        ).unsqueeze(0)
                    )
                    if objective_residual_cpu:
                        objective_residual = torch.stack(
                            [
                                value.to(device=device, non_blocking=True)
                                for value in objective_residual_cpu
                            ],
                            dim=1,
                        )
                    elif self.objective_workspace is not None:
                        objective_residual = torch.empty(
                            (end - first, 0, self.boundaries.hidden_dimension),
                            dtype=torch.bfloat16,
                            device=device,
                        )
                    else:
                        objective_residual = None
                    ready = torch.cuda.Event()
                    ready.record(stream)
                self._put(
                    target,
                    _ReverseDocument(
                        document=document,
                        first_token=first,
                        end_token=end,
                        hidden_gradient=hidden,
                        residual_gradient=residual,
                        objective_hidden_gradient=objective_hidden,
                        objective_residual_gradient=objective_residual,
                        ready=ready,
                    ),
                )
        except BaseException as error:
            self._fail(end_layer, error)

    def _cuda_writer(self, update, role: str, device: torch.device) -> CudaBf16SlabWriter:
        return CudaBf16SlabWriter(
            update.writer(role, direct=self.direct_io),
            device=device,
            buffer_tokens=self.slab_buffer_tokens,
        )

    def _copy_factor_sums(self, item: _LoadedLayer) -> UpstreamFactorSums:
        accumulator = item.accumulator
        if accumulator is None or accumulator.output_factor_sums is None:
            raise RuntimeError(f"decoder layer {item.layer} has no Fisher accumulator")
        torch.cuda.set_device(item.device)
        factor_sums = accumulator.output_factor_sums.detach()
        factor_rows = accumulator.output_factor_rows.detach()
        factor_sums.div_(
            factor_rows.clamp_min(1)
            .to(dtype=factor_sums.dtype)
            .reshape(-1, 1, 1, 1)
        )
        gradient_values: dict[str, object] = {}
        if self.objective_workspace is not None:
            if (
                accumulator.gradient_left is None
                or accumulator.gradient_right is None
                or accumulator.omega_output is None
            ):
                raise RuntimeError(
                    f"decoder layer {item.layer} has no objective-gradient accumulator"
                )
            gradient_values = {
                "gradient_left": (
                    accumulator.gradient_left.detach().cpu().contiguous()
                ),
                "gradient_right": (
                    accumulator.gradient_right.detach().cpu().contiguous()
                ),
                "gradient_rows": (
                    accumulator.gradient_rows.detach().cpu().contiguous()
                ),
                "gradient_output_projection": (
                    accumulator.omega_output.detach().cpu().contiguous()
                ),
                "objective_normalizer": float(self.boundaries.token_count),
            }
        return UpstreamFactorSums(
            output_hessian_blocks=factor_sums.cpu().contiguous(),
            output_hessian_rows=factor_rows.cpu().contiguous(),
            intermediate_draws=torch.tensor(
                self.intermediate_draws[item.layer],
                dtype=torch.uint8,
            ),
            output_hessian_normalized=True,
            **gradient_values,
        )

    def _run_segment(self, first_layer: int) -> UpstreamReverseSegmentRecord:
        end_layer = min(
            first_layer + self.boundaries.attn_res_block_size,
            self.boundaries.num_layers,
        )
        started = time.monotonic()
        loaded = self._load_segment(first_layer, end_layer)
        loaded_at = time.monotonic()
        timings: dict[str, float] = {"load": loaded_at - started}
        factor_writer = self.upstream_factors.begin_segment(first_layer, end_layer)
        update = self.workspace.begin_segment(first_layer)
        objective_update = (
            None
            if self.objective_workspace is None
            else self.objective_workspace.begin_segment(first_layer)
        )
        first_device = loaded[0].device
        writers = {
            role: self._cuda_writer(update, role, first_device)
            for role in update.roles
        }
        objective_writers = (
            {}
            if objective_update is None
            else {
                role: self._cuda_writer(objective_update, role, first_device)
                for role in objective_update.roles
            }
        )
        queues = [
            queue.Queue[_ReverseDocument](maxsize=self.queue_depth)
            for _ in range(len(loaded) + 1)
        ]
        workers = [
            threading.Thread(
                target=self._worker,
                kwargs={
                    "item": item,
                    "source": queues[index + 1],
                    "target": queues[index],
                },
                name=f"k3-reverse-layer-{item.layer:03d}",
                daemon=True,
            )
            for index, item in enumerate(loaded)
        ]
        feeder = threading.Thread(
            target=self._feed,
            kwargs={
                "end_layer": end_layer,
                "target": queues[-1],
                "device": loaded[-1].device,
            },
            name=f"k3-reverse-feed-{first_layer:03d}",
            daemon=True,
        )
        try:
            for worker in workers:
                worker.start()
            feeder.start()
            output_boundaries = tuple(
                range(0, first_layer, self.boundaries.attn_res_block_size)
            )
            for document in range(self.documents.document_count):
                try:
                    result = self._get(queues[0])
                except _PipelineStopped as error:
                    if self._failures.empty():
                        raise
                    failure = self._failures.get()
                    raise RuntimeError(
                        f"reverse layer {failure.layer} failed: {failure.message}\n"
                        f"{failure.traceback}"
                    ) from error
                self._validate_document(
                    result,
                    document=document,
                    output_blocks=len(output_boundaries),
                )
                for writer in writers.values():
                    if result.ready is not None:
                        writer.copy_stream.wait_event(result.ready)
                writers["chain"].append(
                    result.hidden_gradient.reshape(
                        -1,
                        self.boundaries.hidden_dimension,
                    )
                )
                for index, boundary in enumerate(output_boundaries):
                    writers[self.workspace.residual_role(boundary)].append(
                        result.residual_gradient[:, index, :]
                    )
                if self.objective_workspace is not None:
                    if (
                        result.objective_hidden_gradient is None
                        or result.objective_residual_gradient is None
                    ):
                        raise RuntimeError(
                            "reverse pipeline dropped the objective cotangent channel"
                        )
                    for writer in objective_writers.values():
                        if result.ready is not None:
                            writer.copy_stream.wait_event(result.ready)
                    objective_writers["chain"].append(
                        result.objective_hidden_gradient.reshape(
                            -1,
                            self.boundaries.hidden_dimension,
                        )
                    )
                    for index, boundary in enumerate(output_boundaries):
                        objective_writers[
                            self.objective_workspace.residual_role(boundary)
                        ].append(
                            result.objective_residual_gradient[:, index, :]
                        )
            feeder.join()
            for worker in workers:
                worker.join()
            if not self._failures.empty():
                failure = self._failures.get()
                raise RuntimeError(
                    f"reverse layer {failure.layer} failed: {failure.message}\n"
                    f"{failure.traceback}"
                )

            replayed_at = time.monotonic()
            timings["reverse_replay"] = replayed_at - loaded_at
            factor_items = [
                item for item in loaded if item.accumulator is not None
            ]
            copies: list[UpstreamFactorSums] = []
            if factor_items:
                with ThreadPoolExecutor(
                    max_workers=len(factor_items),
                    thread_name_prefix=f"k3-factor-copy-{first_layer:03d}",
                ) as executor:
                    copies = list(executor.map(self._copy_factor_sums, factor_items))
            copied_at = time.monotonic()
            timings["factor_copy"] = copied_at - replayed_at
            stored = []
            if factor_items:
                with ThreadPoolExecutor(
                    max_workers=min(8, len(factor_items)),
                    thread_name_prefix=f"k3-factor-write-{first_layer:03d}",
                ) as executor:
                    stored = list(
                        executor.map(
                            lambda values: factor_writer.add(*values),
                            (
                                (item.layer, sums)
                                for item, sums in zip(
                                    factor_items, copies, strict=True
                                )
                            ),
                        )
                    )
            if len(stored) != len(factor_items):
                raise RuntimeError("upstream-factor segment write was incomplete")
            written_at = time.monotonic()
            timings["factor_write"] = written_at - copied_at
            records: list[UpstreamReverseLayerRecord] = []
            for item, sums in zip(factor_items, copies, strict=True):
                assert item.accumulator is not None
                records.append(
                    UpstreamReverseLayerRecord(
                        layer=item.layer,
                        device=str(item.device),
                        load_seconds=item.load_seconds,
                        supported_experts=int(
                            torch.count_nonzero(sums.output_hessian_rows).item()
                        ),
                        gradient_rows=int(sums.output_hessian_rows.sum().item()),
                        accumulator_bytes=item.accumulator.allocated_bytes(),
                        load_receipt=item.receipt,
                        worker_seconds=item.worker_seconds,
                        queue_wait_seconds=item.queue_wait_seconds,
                        boundary_read_seconds=item.boundary_read_seconds,
                        host_dispatch_seconds=item.host_dispatch_seconds,
                        gpu_active_seconds=item.gpu_active_seconds,
                    )
                )
            factor_writer.seal()
            for writer in writers.values():
                update.record(writer.finish())
            if objective_update is not None:
                for writer in objective_writers.values():
                    objective_update.record(writer.finish())
            update.commit()
            if objective_update is not None:
                objective_update.commit()
            factor_writer.commit()
            committed_at = time.monotonic()
            timings["commit"] = committed_at - written_at
        finally:
            release_started = time.monotonic()
            self._abort.set()
            feeder.join(timeout=1.0)
            for worker in workers:
                worker.join(timeout=1.0)
            for writer in writers.values():
                writer.close()
            for writer in objective_writers.values():
                writer.close()
            self._release_segment(loaded)
            self._abort.clear()
            while not self._failures.empty():
                self._failures.get_nowait()
            timings["release"] = time.monotonic() - release_started
        return UpstreamReverseSegmentRecord(
            first_layer=first_layer,
            end_layer=end_layer,
            documents=self.documents.document_count,
            tokens=self.documents.token_count,
            elapsed_seconds=time.monotonic() - started,
            layers=tuple(records),
            timings=timings,
        )

    def run(self) -> KimiUpstreamReversePipelineResult:
        """Resume at a segment boundary and replay to decoder input."""

        started = time.monotonic()
        completed = tuple(
            str(value["operation"])
            for value in self.workspace.manifest.get("completed_operations", [])
        )
        if self.objective_workspace is not None:
            objective_completed = tuple(
                str(value["operation"])
                for value in self.objective_workspace.manifest.get(
                    "completed_operations", []
                )
            )
            if objective_completed != completed:
                raise ValueError(
                    "Fisher and objective workspaces contain different segments"
                )
        self.upstream_factors.discard_uncommitted_pending(completed)
        self.upstream_factors.recover_pending(completed)
        records: list[UpstreamReverseSegmentRecord] = []
        while int(self.workspace.manifest["chain_boundary"]) > 0:
            if self.objective_workspace is not None and (
                self.objective_workspace.manifest["chain_boundary"]
                != self.workspace.manifest["chain_boundary"]
            ):
                raise ValueError(
                    "Fisher and objective cotangents are at different boundaries"
                )
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


__all__ = ["KimiPipelinedUpstreamReverse"]

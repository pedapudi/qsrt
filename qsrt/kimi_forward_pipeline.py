"""Segmented multi-GPU execution for exact Kimi-K3 boundary capture.

Each segment holds one consecutive decoder layer per CUDA device. Complete
documents move peer-to-peer through that acyclic device pipeline. The final
device writes the segment boundary to the archive; the following segment reads
that boundary on its first device. This bounded disk handoff avoids retaining
the complete corpus on a wraparound GPU while still loading every weight layer
exactly once.

After a device drains its layer, it releases those weights and preloads the
corresponding layer in the following segment while later devices continue to
drain. Every decoder boundary is written as an ordered BF16 slab. CUDA events
carry producer completion across devices, so peer copies and slab writes do
not force device-wide synchronization.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Collection, Iterable, Protocol, Sequence

import torch

from qsrt.kimi_boundary_slabs import KimiBoundarySlabArchive


@dataclass
class PipelineActivation:
    """One complete document at a decoder boundary."""

    document: int
    first_token: int
    end_token: int
    hidden_states: torch.Tensor
    block_residual: torch.Tensor
    ready: torch.cuda.Event | None = None

    @property
    def token_count(self) -> int:
        return self.end_token - self.first_token


class KimiForwardPipelineAdapter(Protocol):
    """Model operations required by the device pipeline."""

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


@dataclass(frozen=True)
class LayerPipelineRecord:
    """Measured work for one materialized decoder layer."""

    layer: int
    stage: int
    device: str
    load_seconds: float
    compute_seconds: float
    tokens: int
    load_receipt: object | None


@dataclass(frozen=True)
class KimiForwardPipelineResult:
    """Completed layer records and wall-clock duration."""

    records: tuple[LayerPipelineRecord, ...]
    elapsed_seconds: float


class _PipelineStopped(RuntimeError):
    pass


@dataclass(frozen=True)
class _WorkerFailure:
    stage: int
    message: str
    traceback: str


@dataclass
class _LoadedLayer:
    module: Any
    receipt: object | None
    load_seconds: float
    ready: torch.cuda.Event


def stage_layers(
    *,
    stage: int,
    stage_count: int,
    num_layers: int,
) -> tuple[int, ...]:
    """Return the decoder layers assigned to one cyclic stage."""

    if stage_count <= 0 or num_layers <= 0:
        raise ValueError("stage and layer counts must be positive")
    if not 0 <= stage < stage_count:
        raise ValueError("stage index is outside the pipeline")
    return tuple(range(stage, num_layers, stage_count))


class KimiForwardPipeline:
    """Execute and archive a complete Kimi-K3 forward pass."""

    def __init__(
        self,
        *,
        adapter: KimiForwardPipelineAdapter,
        archive: KimiBoundarySlabArchive,
        devices: Sequence[torch.device | str],
        queue_depth: int = 2,
        slab_buffer_tokens: int = 2048,
        direct_io: bool = True,
        retained_boundaries: Collection[int] | None = None,
    ):
        if archive.complete:
            raise ValueError("a completed boundary archive cannot be written")
        if not devices:
            raise ValueError("the pipeline requires at least one CUDA device")
        normalized = tuple(torch.device(value) for value in devices)
        if any(value.type != "cuda" or value.index is None for value in normalized):
            raise ValueError("every pipeline device must be an indexed CUDA device")
        if len(set(normalized)) != len(normalized):
            raise ValueError("pipeline devices must be unique")
        if queue_depth <= 0 or slab_buffer_tokens <= 0:
            raise ValueError("queue depth and slab buffer size must be positive")
        if archive.num_layers < len(normalized):
            raise ValueError("device count cannot exceed decoder layer count")

        self.adapter = adapter
        self.archive = archive
        self.devices = normalized
        self.queue_depth = int(queue_depth)
        self.slab_buffer_tokens = int(slab_buffer_tokens)
        self.direct_io = bool(direct_io)
        archive_retained = frozenset(archive.retained_boundaries)
        full_boundaries = frozenset(range(archive.num_layers + 1))
        requested_retained = (
            None
            if retained_boundaries is None
            else frozenset(int(value) for value in retained_boundaries)
        )
        if requested_retained is not None and requested_retained != archive_retained:
            raise ValueError(
                "pipeline retained boundaries differ from the archive contract"
            )
        self.retained_boundaries = (
            None if archive_retained == full_boundaries else archive_retained
        )
        if self.retained_boundaries is not None:
            if any(
                value < 0 or value > archive.num_layers
                for value in self.retained_boundaries
            ):
                raise ValueError("a retained boundary is outside the decoder geometry")
            required = {0}
            for boundary in range(len(normalized), archive.num_layers, len(normalized)):
                required.add(boundary)
                required.update(archive.residual_boundaries_before(boundary))
            missing = required - self.retained_boundaries
            if missing:
                raise ValueError(
                    "selective execution is missing required handoff boundaries: "
                    f"{sorted(missing)}"
                )
        self.documents = archive.load_documents()
        self._abort = threading.Event()
        self._failures: queue.Queue[_WorkerFailure] = queue.Queue()
        self._records: list[LayerPipelineRecord] = []
        self._records_lock = threading.Lock()

    def _put(
        self,
        target: queue.Queue[PipelineActivation],
        item: PipelineActivation,
    ) -> None:
        while not self._abort.is_set():
            try:
                target.put(item, timeout=0.1)
                return
            except queue.Full:
                continue
        raise _PipelineStopped("pipeline stopped while publishing an activation")

    def _get(
        self,
        source: queue.Queue[PipelineActivation],
    ) -> PipelineActivation:
        while not self._abort.is_set():
            try:
                return source.get(timeout=0.1)
            except queue.Empty:
                continue
        raise _PipelineStopped("pipeline stopped while awaiting an activation")

    def _fail(self, stage: int, error: BaseException) -> None:
        if isinstance(error, _PipelineStopped) and self._abort.is_set():
            return
        self._failures.put(
            _WorkerFailure(
                stage=stage,
                message=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
        )
        self._abort.set()

    def _validate_item(
        self,
        item: PipelineActivation,
        *,
        expected_document: int,
    ) -> None:
        first, end = self.documents.document_extent(expected_document)
        expected = (expected_document, first, end)
        observed = (item.document, item.first_token, item.end_token)
        if observed != expected:
            raise ValueError(
                f"pipeline document order mismatch: {observed} != {expected}"
            )
        hidden = item.hidden_states
        residual = item.block_residual
        if hidden.dtype != torch.bfloat16 or hidden.ndim != 3:
            raise TypeError("pipeline hidden states must have shape [1, tokens, H] in BF16")
        if tuple(hidden.shape) != (1, end - first, self.archive.hidden_dimension):
            raise ValueError("pipeline hidden states do not match the document extent")
        if residual.dtype != torch.bfloat16 or residual.ndim != 3:
            raise TypeError("pipeline block residual must have shape [tokens, blocks, H]")
        if tuple(residual.shape[::2]) != (end - first, self.archive.hidden_dimension):
            raise ValueError("pipeline block residual does not match the document extent")
        if residual.device != hidden.device:
            raise ValueError("hidden states and block residual must share one device")

    @staticmethod
    def _move_to_stage(
        item: PipelineActivation,
        *,
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.cuda.device(device), torch.cuda.stream(stream):
            if item.ready is not None:
                stream.wait_event(item.ready)
            hidden = item.hidden_states
            residual = item.block_residual
            if hidden.device != device:
                hidden = hidden.to(device=device, non_blocking=True)
            if residual.device != device:
                residual = residual.to(device=device, non_blocking=True)
            item.hidden_states.record_stream(stream)
            item.block_residual.record_stream(stream)
        return hidden, residual

    def _load_layer(self, layer: int, device: torch.device) -> _LoadedLayer:
        started = time.monotonic()
        module, receipt = self.adapter.load_layer(layer, device)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(device))
        return _LoadedLayer(
            module=module,
            receipt=receipt,
            load_seconds=time.monotonic() - started,
            ready=ready,
        )

    def _release_loaded(self, loaded: _LoadedLayer, device: torch.device) -> None:
        torch.cuda.set_device(device)
        self.adapter.release_layer(loaded.module)
        del loaded.module
        torch.cuda.empty_cache()

    def _worker(
        self,
        *,
        stage: int,
        layer: int,
        end_layer: int,
        source: queue.Queue[PipelineActivation],
        target: queue.Queue[PipelineActivation] | None,
        loaded: _LoadedLayer | None,
        preloaded: dict[int, _LoadedLayer],
        preloaded_lock: threading.Lock,
    ) -> None:
        device = self.devices[stage]
        torch.cuda.set_device(device)
        compute_stream = torch.cuda.Stream(device=device)
        current = loaded
        try:
            if current is None:
                current = self._load_layer(layer, device)
            compute_stream.wait_event(current.ready)
            compute_seconds = 0.0
            processed_tokens = 0
            started_event: torch.cuda.Event | None = None
            finished_event: torch.cuda.Event | None = None
            writer_id = f"stage-{stage:02d}"
            persist_output = (
                self.retained_boundaries is None
                or layer + 1 in self.retained_boundaries
            )
            writer_context = (
                self.archive.cuda_extent_writer(
                    layer + 1,
                    writer_id=writer_id,
                    first_token=0,
                    end_token=self.archive.token_count,
                    device=device,
                    buffer_tokens=self.slab_buffer_tokens,
                    direct=self.direct_io,
                )
                if persist_output
                else nullcontext(None)
            )
            with writer_context as slab_writer:
                for document in range(self.documents.document_count):
                    item = self._get(source)
                    self._validate_item(item, expected_document=document)
                    hidden, residual = self._move_to_stage(
                        item,
                        device=device,
                        stream=compute_stream,
                    )
                    expected_blocks = len(
                        range(0, layer, self.archive.attn_res_block_size)
                    )
                    if residual.shape[1] != expected_blocks:
                        raise ValueError(
                            f"layer {layer} received {residual.shape[1]} "
                            f"attention-residual blocks, expected {expected_blocks}"
                        )
                    with torch.inference_mode(), torch.cuda.stream(compute_stream):
                        if started_event is None:
                            started_event = torch.cuda.Event(enable_timing=True)
                            started_event.record(compute_stream)
                        output, output_residual = self.adapter.forward_layer(
                            current.module,
                            layer=layer,
                            hidden_states=hidden,
                            block_residual=residual,
                        )
                        self._validate_item(
                            PipelineActivation(
                                document=document,
                                first_token=item.first_token,
                                end_token=item.end_token,
                                hidden_states=output,
                                block_residual=output_residual,
                            ),
                            expected_document=document,
                        )
                        expected_output_blocks = expected_blocks + int(
                            layer % self.archive.attn_res_block_size == 0
                        )
                        if output_residual.shape[1] != expected_output_blocks:
                            raise ValueError(
                                f"layer {layer} produced "
                                f"{output_residual.shape[1]} attention-residual "
                                f"blocks, expected {expected_output_blocks}"
                            )
                        if slab_writer is not None:
                            slab_writer.append(
                                output.reshape(-1, self.archive.hidden_dimension)
                            )
                        ready = torch.cuda.Event()
                        ready.record(compute_stream)
                    processed_tokens += item.token_count
                    if target is not None:
                        self._put(
                            target,
                            PipelineActivation(
                                document=document,
                                first_token=item.first_token,
                                end_token=item.end_token,
                                hidden_states=output,
                                block_residual=output_residual,
                                ready=ready,
                            ),
                        )
                with torch.cuda.stream(compute_stream):
                    finished_event = torch.cuda.Event(enable_timing=True)
                    finished_event.record(compute_stream)
                if slab_writer is not None:
                    slab_writer.finish()

            compute_stream.synchronize()
            assert started_event is not None and finished_event is not None
            compute_seconds = started_event.elapsed_time(finished_event) / 1000.0
            with self._records_lock:
                self._records.append(
                    LayerPipelineRecord(
                        layer=layer,
                        stage=stage,
                        device=str(device),
                        load_seconds=current.load_seconds,
                        compute_seconds=compute_seconds,
                        tokens=processed_tokens,
                        load_receipt=current.receipt,
                    )
                )

            self._release_loaded(current, device)
            current = None
            next_layer = layer + len(self.devices)
            if next_layer < end_layer and not self._abort.is_set():
                following = self._load_layer(next_layer, device)
                with preloaded_lock:
                    preloaded[stage] = following
        except BaseException as error:
            self._fail(stage, error)
        finally:
            if current is not None:
                try:
                    compute_stream.synchronize()
                finally:
                    self._release_loaded(current, device)

    def _feed(
        self,
        inputs: Iterable[PipelineActivation],
        *,
        target: queue.Queue[PipelineActivation],
        write_boundary_zero: bool,
    ) -> None:
        device = self.devices[0]
        torch.cuda.set_device(device)
        try:
            writer_context = (
                self.archive.cuda_extent_writer(
                    0,
                    writer_id="input",
                    first_token=0,
                    end_token=self.archive.token_count,
                    device=device,
                    buffer_tokens=self.slab_buffer_tokens,
                    direct=self.direct_io,
                )
                if write_boundary_zero
                else nullcontext(None)
            )
            with writer_context as slab_writer:
                observed = 0
                for document, item in enumerate(inputs):
                    if self._abort.is_set():
                        raise _PipelineStopped("pipeline stopped while reading inputs")
                    self._validate_item(item, expected_document=document)
                    if item.hidden_states.device != device:
                        raise ValueError("input activations must originate on the first device")
                    if write_boundary_zero and item.block_residual.shape[1] != 0:
                        raise ValueError("decoder input must have an empty residual prefix")
                    current_stream = torch.cuda.current_stream(device)
                    if item.ready is not None:
                        current_stream.wait_event(item.ready)
                    if slab_writer is not None:
                        slab_writer.append(
                            item.hidden_states.reshape(
                                -1, self.archive.hidden_dimension
                            )
                        )
                    if item.ready is None:
                        item.ready = torch.cuda.Event()
                        item.ready.record(current_stream)
                    self._put(target, item)
                    observed += 1
                if observed != self.documents.document_count:
                    raise ValueError(
                        f"input source produced {observed} documents, expected "
                        f"{self.documents.document_count}"
                    )
                if slab_writer is not None:
                    slab_writer.finish()
        except BaseException as error:
            self._fail(-1, error)

    def _archive_inputs(self, boundary: int) -> Iterable[PipelineActivation]:
        device = self.devices[0]
        torch.cuda.set_device(device)
        copy_stream = torch.cuda.Stream(device=device)
        residual_boundaries = self.archive.residual_boundaries_before(boundary)
        for document in range(self.documents.document_count):
            first, end = self.documents.document_extent(document)
            hidden_cpu = self.archive.read_cpu(
                boundary,
                first,
                end,
                direct=self.direct_io,
                pin_memory=True,
            )
            residual_cpu = [
                self.archive.read_cpu(
                    residual_boundary,
                    first,
                    end,
                    direct=self.direct_io,
                    pin_memory=True,
                )
                for residual_boundary in residual_boundaries
            ]
            with torch.cuda.stream(copy_stream):
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
                        (end - first, 0, self.archive.hidden_dimension),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                ready = torch.cuda.Event()
                ready.record(copy_stream)
            yield PipelineActivation(
                document=document,
                first_token=first,
                end_token=end,
                hidden_states=hidden,
                block_residual=residual,
                ready=ready,
            )

    def _run_segment(
        self,
        *,
        first_layer: int,
        end_layer: int,
        inputs: Iterable[PipelineActivation],
        preloaded: dict[int, _LoadedLayer],
        write_boundary_zero: bool,
    ) -> dict[int, _LoadedLayer]:
        layer_count = min(len(self.devices), end_layer - first_layer)
        links = [
            queue.Queue[PipelineActivation](maxsize=self.queue_depth)
            for _ in range(layer_count)
        ]
        following: dict[int, _LoadedLayer] = {}
        following_lock = threading.Lock()
        workers = [
            threading.Thread(
                target=self._worker,
                kwargs={
                    "stage": stage,
                    "layer": first_layer + stage,
                    "end_layer": end_layer,
                    "source": links[stage],
                    "target": links[stage + 1] if stage + 1 < layer_count else None,
                    "loaded": preloaded.pop(stage, None),
                    "preloaded": following,
                    "preloaded_lock": following_lock,
                },
                name=f"kimi-forward-layer-{first_layer + stage:03d}",
            )
            for stage in range(layer_count)
        ]
        feeder = threading.Thread(
            target=self._feed,
            kwargs={
                "inputs": inputs,
                "target": links[0],
                "write_boundary_zero": write_boundary_zero,
            },
            name=f"kimi-forward-input-{first_layer:03d}",
        )
        for worker in workers:
            worker.start()
        feeder.start()
        feeder.join()
        for worker in workers:
            worker.join()
        for stage, loaded in preloaded.items():
            self._release_loaded(loaded, self.devices[stage])
        preloaded.clear()
        return following

    def run(
        self,
        inputs: Iterable[PipelineActivation] | None,
        *,
        start_layer: int = 0,
        end_layer: int | None = None,
    ) -> KimiForwardPipelineResult:
        """Run a decoder interval and seal its declared boundary archive."""

        execution_end = self.archive.num_layers if end_layer is None else int(end_layer)
        if not 0 <= start_layer <= execution_end <= self.archive.num_layers:
            raise ValueError("execution interval is outside the decoder geometry")
        if start_layer == 0 and inputs is None and 0 not in self.archive.sealed_boundary_prefix():
            raise ValueError("decoder inputs are required when boundary zero is unsealed")
        if start_layer > 0 and start_layer not in self.archive.sealed_boundary_prefix():
            raise ValueError("the restart boundary must already be sealed")

        started = time.monotonic()
        prepared_boundaries = (
            range(self.archive.num_layers + 1)
            if self.retained_boundaries is None
            else sorted(self.retained_boundaries)
        )
        for boundary in prepared_boundaries:
            self.archive.prepare_boundary(boundary)

        preloaded: dict[int, _LoadedLayer] = {}
        try:
            for first_layer in range(
                start_layer,
                execution_end,
                len(self.devices),
            ):
                segment_inputs = (
                    inputs
                    if first_layer == 0 and inputs is not None
                    else self._archive_inputs(first_layer)
                )
                preloaded = self._run_segment(
                    first_layer=first_layer,
                    end_layer=execution_end,
                    inputs=segment_inputs,
                    preloaded=preloaded,
                    write_boundary_zero=(
                        first_layer == 0
                        and inputs is not None
                        and (
                            self.retained_boundaries is None
                            or 0 in self.retained_boundaries
                        )
                    ),
                )
                failures: list[_WorkerFailure] = []
                while not self._failures.empty():
                    failures.append(self._failures.get())
                if failures:
                    first = failures[0]
                    raise RuntimeError(
                        f"Kimi forward pipeline failed at stage {first.stage}: "
                        f"{first.message}\n{first.traceback}"
                    )
                if (
                    first_layer == 0
                    and inputs is not None
                    and (
                        self.retained_boundaries is None
                        or 0 in self.retained_boundaries
                    )
                ):
                    self.archive.seal_boundary(0)
                segment_end = min(
                    first_layer + len(self.devices),
                    execution_end,
                )
                for boundary in range(first_layer + 1, segment_end + 1):
                    if (
                        self.retained_boundaries is None
                        or boundary in self.retained_boundaries
                    ):
                        self.archive.seal_boundary(boundary)
        finally:
            for stage, loaded in preloaded.items():
                self._release_loaded(loaded, self.devices[stage])
        self.archive.seal()
        return KimiForwardPipelineResult(
            records=tuple(sorted(self._records, key=lambda value: value.layer)),
            elapsed_seconds=time.monotonic() - started,
        )


__all__ = [
    "KimiForwardPipeline",
    "KimiForwardPipelineAdapter",
    "KimiForwardPipelineResult",
    "LayerPipelineRecord",
    "PipelineActivation",
    "stage_layers",
]

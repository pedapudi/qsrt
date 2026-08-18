"""Layer-resident replay for exact final-KL expert-weight gradients."""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from qsrt.kimi_boundary_slabs import KimiBoundarySlabArchive
from qsrt.kimi_cotangent_slabs import KimiCotangentSlabWorkspace
from qsrt.kimi_dense_objective_gradients import (
    DenseObjectiveGradientAccumulator,
    DenseObjectiveGradientArchive,
    DenseObjectiveGradientConfig,
    DenseObjectiveGradientRouter,
)
from qsrt.kimi_upstream_pipelined_reverse import (
    KimiPipelinedUpstreamReverse,
    _LoadedLayer,
    _PipelineStopped,
    _ReverseDocument,
)
from qsrt.qsrt_coupled import CoupledHadamardSpec


@dataclass(frozen=True)
class DenseGradientShardRecord:
    """Timing and support for one expert-shard replay."""

    expert_begin: int
    expert_end: int
    elapsed_seconds: float
    replay_seconds: float
    write_seconds: float
    layer_rows: Mapping[int, Mapping[str, int]]


@dataclass(frozen=True)
class DenseGradientReplayResult:
    """Completed dense-gradient replay over all expert shards."""

    first_layer: int
    end_layer: int
    target_layers: tuple[int, ...]
    records: tuple[DenseGradientShardRecord, ...]
    elapsed_seconds: float


class KimiDenseObjectiveGradientReplay(KimiPipelinedUpstreamReverse):
    """Replay one residual segment repeatedly while its decoder layers stay resident."""

    def __init__(
        self,
        *,
        adapter: Any,
        boundary_archive: KimiBoundarySlabArchive,
        objective_workspace: KimiCotangentSlabWorkspace,
        gradient_archive: DenseObjectiveGradientArchive,
        intermediate_draws: Mapping[int, Sequence[int]],
        target_layers: Sequence[int],
        devices: Sequence[torch.device | str],
        queue_depth: int = 1,
        direct_io: bool = True,
        validation_documents: int = 1,
        coupled_spec: CoupledHadamardSpec = CoupledHadamardSpec(),
    ):
        normalized = tuple(torch.device(value) for value in devices)
        if not normalized or any(
            value.type != "cuda" or value.index is None for value in normalized
        ):
            raise ValueError("dense reverse replay requires indexed CUDA devices")
        if len(set(normalized)) != len(normalized):
            raise ValueError("dense reverse replay devices must be unique")
        if not boundary_archive.complete:
            raise ValueError("dense reverse replay requires sealed boundary activations")
        if objective_workspace.manifest.get("chain_boundary") != boundary_archive.num_layers:
            raise ValueError("objective cotangents must remain at the final decoder boundary")
        if (
            objective_workspace.token_count != boundary_archive.token_count
            or objective_workspace.hidden_dimension != boundary_archive.hidden_dimension
            or objective_workspace.num_layers != boundary_archive.num_layers
            or objective_workspace.residual_block_size
            != boundary_archive.attn_res_block_size
        ):
            raise ValueError("objective cotangents and boundary activations disagree")
        layers = tuple(sorted({int(value) for value in target_layers}))
        if not layers:
            raise ValueError("at least one decoder layer must receive dense gradients")
        first_layer = (
            (layers[0] // boundary_archive.attn_res_block_size)
            * boundary_archive.attn_res_block_size
        )
        end_layer = min(
            first_layer + boundary_archive.attn_res_block_size,
            boundary_archive.num_layers,
        )
        if any(not first_layer <= layer < end_layer for layer in layers):
            raise ValueError("dense-gradient target layers must share one residual segment")
        if len(normalized) < end_layer - first_layer:
            raise ValueError("one CUDA device is required per replayed decoder layer")
        archive_layers = tuple(int(value) for value in gradient_archive.manifest["layers"])
        if layers != archive_layers:
            raise ValueError("gradient archive target layers differ from replay targets")
        num_experts = int(gradient_archive.manifest["num_experts"])
        supplied_draws = {int(layer): tuple(int(v) for v in values) for layer, values in intermediate_draws.items()}
        if set(supplied_draws) != set(layers):
            raise ValueError("coupled draws must cover exactly the gradient target layers")
        if any(
            len(values) != num_experts or any(not 0 <= value < 8 for value in values)
            for values in supplied_draws.values()
        ):
            raise ValueError("coupled draws are incompatible with the expert inventory")
        if queue_depth <= 0 or validation_documents < 0:
            raise ValueError("pipeline queue and validation counts must be nonnegative")

        self.adapter = adapter
        self.boundaries = boundary_archive
        # The inherited worker labels its only channel "fisher". Here that
        # channel carries the deterministic final-KL cotangent.
        self.workspace = objective_workspace
        self.objective_workspace = None
        self.gradient_archive = gradient_archive
        self.intermediate_draws = supplied_draws
        self.target_layers = layers
        self.first_layer = first_layer
        self.end_layer = end_layer
        self.devices = normalized[: end_layer - first_layer]
        self.queue_depth = int(queue_depth)
        self.direct_io = bool(direct_io)
        self.validation_documents = int(validation_documents)
        self.coupled_spec = coupled_spec
        self.documents = boundary_archive.load_documents()
        self._abort = threading.Event()
        self._failures = queue.Queue()

    def _load_segment(self) -> list[_LoadedLayer]:
        layers = tuple(range(self.first_layer, self.end_layer))
        with ThreadPoolExecutor(
            max_workers=len(layers),
            thread_name_prefix=f"k3-dense-gradient-load-{self.first_layer:03d}",
        ) as executor:
            futures = [
                executor.submit(self._load_one, layer, device)
                for layer, device in zip(layers, self.devices, strict=True)
            ]
            return [future.result() for future in futures]

    def _attach_shard(
        self,
        loaded: Sequence[_LoadedLayer],
        *,
        expert_begin: int,
        expert_end: int,
    ) -> None:
        for item in loaded:
            if item.layer not in self.target_layers:
                continue
            config = DenseObjectiveGradientConfig(
                num_experts=int(self.gradient_archive.manifest["num_experts"]),
                expert_begin=expert_begin,
                expert_end=expert_end,
                hidden_dimension=int(
                    self.gradient_archive.manifest["hidden_dimension"]
                ),
                intermediate_dimension=int(
                    self.gradient_archive.manifest["intermediate_dimension"]
                ),
                intermediate_draws=self.intermediate_draws[item.layer],
                spec=self.coupled_spec,
            )
            item.accumulator = DenseObjectiveGradientAccumulator(
                config,
                device=item.device,
            )
            item.router = DenseObjectiveGradientRouter(item.accumulator)
            item.router.install(self.adapter, item.module)

    def _detach_shard(self, loaded: Sequence[_LoadedLayer]) -> None:
        devices: set[torch.device] = set()
        for item in loaded:
            if item.router is not None:
                item.router.uninstall(self.adapter, item.module)
                item.router = None
            if item.accumulator is not None:
                devices.add(item.device)
                item.accumulator = None
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    def _run_loaded_pass(self, loaded: Sequence[_LoadedLayer]) -> None:
        self._abort.clear()
        while not self._failures.empty():
            self._failures.get_nowait()
        for item in loaded:
            item.worker_seconds = 0.0
            item.queue_wait_seconds = 0.0
            item.boundary_read_seconds = 0.0
            item.host_dispatch_seconds = 0.0
            item.gpu_active_seconds = 0.0
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
                name=f"k3-dense-gradient-layer-{item.layer:03d}",
                daemon=True,
            )
            for index, item in enumerate(loaded)
        ]
        feeder = threading.Thread(
            target=self._feed,
            kwargs={
                "end_layer": self.end_layer,
                "target": queues[-1],
                "device": loaded[-1].device,
            },
            name=f"k3-dense-gradient-feed-{self.first_layer:03d}",
            daemon=True,
        )
        drain_stream = torch.cuda.Stream(device=loaded[0].device)
        try:
            for worker in workers:
                worker.start()
            feeder.start()
            output_blocks = len(
                range(0, self.first_layer, self.boundaries.attn_res_block_size)
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
                    output_blocks=output_blocks,
                )
                with torch.cuda.device(loaded[0].device), torch.cuda.stream(drain_stream):
                    if result.ready is not None:
                        drain_stream.wait_event(result.ready)
                    result.hidden_gradient.record_stream(drain_stream)
                    result.residual_gradient.record_stream(drain_stream)
            feeder.join()
            for worker in workers:
                worker.join()
            if not self._failures.empty():
                failure = self._failures.get()
                raise RuntimeError(
                    f"reverse layer {failure.layer} failed: {failure.message}\n"
                    f"{failure.traceback}"
                )
            drain_stream.synchronize()
        finally:
            self._abort.set()
            feeder.join(timeout=1.0)
            for worker in workers:
                worker.join(timeout=1.0)
            self._abort.clear()

    def _write_shards(self, loaded: Sequence[_LoadedLayer]) -> None:
        existing = set(dict(self.gradient_archive.manifest.get("shards", {})))
        items = [
            item
            for item in loaded
            if item.accumulator is not None
            and self.gradient_archive.shard_key(
                item.layer,
                item.accumulator.config.expert_begin,
                item.accumulator.config.expert_end,
            )
            not in existing
        ]

        def write(item: _LoadedLayer) -> None:
            assert isinstance(item.accumulator, DenseObjectiveGradientAccumulator)
            torch.cuda.set_device(item.device)
            self.gradient_archive.write_shard(
                layer=item.layer,
                accumulator=item.accumulator,
            )

        with ThreadPoolExecutor(
            max_workers=len(items),
            thread_name_prefix="k3-dense-gradient-write",
        ) as executor:
            list(executor.map(write, items))

    def run(self) -> DenseGradientReplayResult:
        """Capture every configured expert shard without reloading decoder layers."""

        started = time.monotonic()
        loaded = self._load_segment()
        records: list[DenseGradientShardRecord] = []
        num_experts = int(self.gradient_archive.manifest["num_experts"])
        shard_experts = int(self.gradient_archive.manifest["shard_experts"])
        try:
            for expert_begin in range(0, num_experts, shard_experts):
                expert_end = expert_begin + shard_experts
                existing = set(
                    dict(self.gradient_archive.manifest.get("shards", {}))
                )
                expected = {
                    self.gradient_archive.shard_key(
                        layer,
                        expert_begin,
                        expert_end,
                    )
                    for layer in self.target_layers
                }
                if expected <= existing:
                    continue
                shard_started = time.monotonic()
                self._attach_shard(
                    loaded,
                    expert_begin=expert_begin,
                    expert_end=expert_end,
                )
                replay_started = time.monotonic()
                self._run_loaded_pass(loaded)
                replay_finished = time.monotonic()
                layer_rows = {
                    item.layer: {
                        "upstream": int(item.accumulator.upstream_rows.sum().item()),
                        "down": int(item.accumulator.down_rows.sum().item()),
                    }
                    for item in loaded
                    if isinstance(
                        item.accumulator,
                        DenseObjectiveGradientAccumulator,
                    )
                }
                self._write_shards(loaded)
                write_finished = time.monotonic()
                records.append(
                    DenseGradientShardRecord(
                        expert_begin=expert_begin,
                        expert_end=expert_end,
                        elapsed_seconds=write_finished - shard_started,
                        replay_seconds=replay_finished - replay_started,
                        write_seconds=write_finished - replay_finished,
                        layer_rows=layer_rows,
                    )
                )
                self._detach_shard(loaded)
            self.gradient_archive.seal()
        finally:
            self._detach_shard(loaded)
            self._release_segment(loaded)
        return DenseGradientReplayResult(
            first_layer=self.first_layer,
            end_layer=self.end_layer,
            target_layers=self.target_layers,
            records=tuple(records),
            elapsed_seconds=time.monotonic() - started,
        )


__all__ = [
    "DenseGradientReplayResult",
    "DenseGradientShardRecord",
    "KimiDenseObjectiveGradientReplay",
]

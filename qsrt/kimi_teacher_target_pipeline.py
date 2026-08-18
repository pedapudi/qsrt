"""Data-parallel construction of frozen teacher LM-head input targets."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from qsrt.instanttensor_kimi import InstantTensorLoadConfig
from qsrt.kimi_boundary_slabs import DocumentPartition, KimiBoundarySlabArchive
from qsrt.kimi_cotangent_slabs import CudaBf16SlabWriter
from qsrt.kimi_official_fisher import OfficialKimiFisherSuffix
from qsrt.kimi_suffix_training_archive import KimiSuffixTrainingArchive


@dataclass(frozen=True)
class TeacherTargetWorkerRecord:
    """Completed document extent on one CUDA device."""

    device: str
    first_document: int
    end_document: int
    first_token: int
    end_token: int
    elapsed_seconds: float


@dataclass(frozen=True)
class TeacherTargetPipelineResult:
    """Sealed teacher targets and worker timing."""

    workers: tuple[TeacherTargetWorkerRecord, ...]
    elapsed_seconds: float


class KimiTeacherTargetPipeline:
    """Reduce exact teacher decoder state to one normalized target slab."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        teacher_boundaries: KimiBoundarySlabArchive,
        destination: KimiSuffixTrainingArchive,
        devices: Sequence[torch.device | str],
        epsilon: float,
        vocabulary_size: int,
        logit_scale: float = 1.0,
        slab_buffer_tokens: int = 256,
        replay_chunk_tokens: int = 2048,
        direct_io: bool = True,
        load_config: InstantTensorLoadConfig | None = None,
    ):
        normalized = tuple(torch.device(value) for value in devices)
        if not normalized or any(
            value.type != "cuda" or value.index is None for value in normalized
        ):
            raise ValueError("teacher target pipeline requires indexed CUDA devices")
        if len(set(normalized)) != len(normalized):
            raise ValueError("teacher target devices must be unique")
        if not teacher_boundaries.complete:
            raise ValueError("teacher boundary archive must be sealed")
        if destination.complete:
            raise ValueError("suffix-training archive is already complete")
        if (
            teacher_boundaries.num_layers != destination.num_layers
            or teacher_boundaries.hidden_dimension != destination.hidden_dimension
            or teacher_boundaries.attn_res_block_size
            != destination.attn_res_block_size
            or teacher_boundaries.token_count != destination.token_count
            or teacher_boundaries.document_count != destination.document_count
        ):
            raise ValueError("teacher and suffix-training archives disagree")
        residual_boundaries = tuple(
            range(
                0,
                teacher_boundaries.num_layers,
                teacher_boundaries.attn_res_block_size,
            )
        )
        required = set(residual_boundaries) | {teacher_boundaries.num_layers}
        missing = required - set(teacher_boundaries.retained_boundaries)
        if missing:
            raise ValueError(
                f"teacher archive is missing required boundaries: {sorted(missing)}"
            )
        teacher_documents = teacher_boundaries.load_documents()
        destination_documents = destination.load_documents()
        if (
            teacher_documents.identifiers != destination_documents.identifiers
            or not torch.equal(teacher_documents.offsets, destination_documents.offsets)
            or not torch.equal(
                teacher_documents.input_ids,
                destination_documents.input_ids,
            )
        ):
            raise ValueError("teacher and student archives contain different documents")
        if slab_buffer_tokens <= 0:
            raise ValueError("teacher target slab buffer must be positive")
        if replay_chunk_tokens <= 0:
            raise ValueError("teacher target replay chunk must be positive")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.teacher = teacher_boundaries
        self.destination = destination
        self.devices = normalized
        self.epsilon = float(epsilon)
        self.vocabulary_size = int(vocabulary_size)
        self.logit_scale = float(logit_scale)
        self.slab_buffer_tokens = int(slab_buffer_tokens)
        self.replay_chunk_tokens = int(replay_chunk_tokens)
        self.direct_io = bool(direct_io)
        self.load_config = load_config or InstantTensorLoadConfig()
        self.documents = teacher_documents
        self.residual_boundaries = residual_boundaries

    def _worker(
        self,
        *,
        partition: DocumentPartition,
        device: torch.device,
        suffix: OfficialKimiFisherSuffix,
    ) -> TeacherTargetWorkerRecord:
        started = time.monotonic()
        torch.cuda.set_device(device)
        writer_id = f"device-{device.index:02d}"
        writer = CudaBf16SlabWriter(
            self.destination.teacher_target_writer(
                writer_id=writer_id,
                first_token=partition.first_token,
                end_token=partition.end_token,
                direct=self.direct_io,
            ),
            device=device,
            buffer_tokens=self.slab_buffer_tokens,
        )
        try:
            for document in range(
                partition.first_document,
                partition.end_document,
            ):
                first, end = self.documents.document_extent(document)
                for chunk_first in range(first, end, self.replay_chunk_tokens):
                    chunk_end = min(chunk_first + self.replay_chunk_tokens, end)
                    final_cpu = self.teacher.read_cpu(
                        self.teacher.num_layers,
                        chunk_first,
                        chunk_end,
                        direct=self.direct_io,
                        pin_memory=True,
                    )
                    residual_cpu = tuple(
                        self.teacher.read_cpu(
                            boundary,
                            chunk_first,
                            chunk_end,
                            direct=self.direct_io,
                            pin_memory=True,
                        )
                        for boundary in self.residual_boundaries
                    )
                    final = final_cpu.to(device=device, non_blocking=True).unsqueeze(0)
                    residual = tuple(
                        value.to(device=device, non_blocking=True)
                        for value in residual_cpu
                    )
                    normalized = suffix.normalized_hidden(
                        final_boundary=final,
                        residual_inputs=residual,
                    )
                    writer.append(
                        normalized.reshape(-1, self.teacher.hidden_dimension)
                    )
            receipt = writer.finish()
            self.destination.record_teacher_target(receipt)
        finally:
            writer.close()
        return TeacherTargetWorkerRecord(
            device=str(device),
            first_document=partition.first_document,
            end_document=partition.end_document,
            first_token=partition.first_token,
            end_token=partition.end_token,
            elapsed_seconds=time.monotonic() - started,
        )

    def run(self) -> TeacherTargetPipelineResult:
        """Construct and seal the complete normalized teacher target."""

        started = time.monotonic()
        partitions = self.documents.contiguous_partitions(len(self.devices))
        # InstantTensor selects CUDA destinations through process-global state
        # during handle creation. Materialize replicas serially, then replay
        # documents in parallel.
        suffixes = tuple(
            OfficialKimiFisherSuffix(
                checkpoint=self.checkpoint,
                device=device,
                hidden_dimension=self.teacher.hidden_dimension,
                vocabulary_size=self.vocabulary_size,
                residual_block_count=len(self.residual_boundaries),
                epsilon=self.epsilon,
                logit_scale=self.logit_scale,
                load_config=self.load_config,
            )
            for device in self.devices
        )
        records = []
        with ThreadPoolExecutor(
            max_workers=len(self.devices),
            thread_name_prefix="kimi-teacher-target",
        ) as executor:
            futures = [
                executor.submit(
                    self._worker,
                    partition=partition,
                    device=device,
                    suffix=suffix,
                )
                for partition, device, suffix in zip(
                    partitions,
                    self.devices,
                    suffixes,
                    strict=True,
                )
            ]
            for future in as_completed(futures):
                records.append(future.result())
        self.destination.seal_teacher_target()
        return TeacherTargetPipelineResult(
            workers=tuple(sorted(records, key=lambda value: value.first_token)),
            elapsed_seconds=time.monotonic() - started,
        )


__all__ = [
    "KimiTeacherTargetPipeline",
    "TeacherTargetPipelineResult",
    "TeacherTargetWorkerRecord",
]

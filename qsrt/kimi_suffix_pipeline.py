"""Data-parallel final-output Fisher initialization for Kimi-K3 replay."""

from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from qsrt.instanttensor_kimi import InstantTensorLoadConfig
from qsrt.kimi_boundary_slabs import DocumentPartition, KimiBoundarySlabArchive
from qsrt.kimi_cotangent_slabs import (
    CudaBf16SlabWriter,
    KimiCotangentSlabWorkspace,
    SlabWriteReceipt,
)
from qsrt.kimi_official_fisher import OfficialKimiFisherSuffix


SAMPLE_FILENAME = "fisher-token-pairs.i32"


def document_fisher_seed(
    base_seed: int,
    *,
    document: int,
    identifier: str,
) -> int:
    """Derive a batching- and worker-independent seed for one document."""

    digest = hashlib.sha256(
        f"Kimi-K3 final-output Fisher v1\n{int(base_seed)}\n"
        f"{int(document)}\n{identifier}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SuffixWorkerRecord:
    """Completed document and token extent on one CUDA device."""

    device: str
    first_document: int
    end_document: int
    first_token: int
    end_token: int
    elapsed_seconds: float


@dataclass(frozen=True)
class SuffixPipelineResult:
    """Sealed suffix cotangents and deterministic vocabulary samples."""

    workers: tuple[SuffixWorkerRecord, ...]
    sample_path: Path
    sample_sha256: str
    elapsed_seconds: float
    objective_kl_sum: float | None = None
    objective_tokens: int = 0


class _SamplePairWriter:
    def __init__(
        self,
        path: Path,
        *,
        first_token: int,
        end_token: int,
    ):
        self.path = path
        self.first_token = int(first_token)
        self.end_token = int(end_token)
        self.next_token = self.first_token
        self.fd = os.open(path, os.O_WRONLY)

    def append(self, first: torch.Tensor, second: torch.Tensor) -> None:
        if first.device.type != "cpu" or second.device.type != "cpu":
            raise ValueError("sample-pair writes require CPU tensors")
        if first.ndim != 1 or first.shape != second.shape:
            raise ValueError("sample token pairs have incompatible geometry")
        value = torch.stack((first, second), dim=1).to(torch.int32).contiguous()
        rows = int(value.shape[0])
        if self.next_token + rows > self.end_token:
            raise ValueError("sample-pair write exceeds the token extent")
        byte_view = memoryview(value.view(torch.uint8).numpy())
        offset = self.next_token * 2 * torch.int32.itemsize
        written = 0
        while written < byte_view.nbytes:
            count = os.pwrite(self.fd, byte_view[written:], offset + written)
            if count <= 0:
                raise OSError("sample-pair write made no progress")
            written += count
        self.next_token += rows

    def finish(self) -> None:
        if self.next_token != self.end_token:
            raise ValueError("sample-pair writer did not cover its token extent")
        os.fsync(self.fd)
        os.close(self.fd)
        self.fd = -1

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


class KimiSuffixPipeline:
    """Initialize reverse cotangents on document-balanced CUDA workers."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        boundary_archive: KimiBoundarySlabArchive,
        workspace: KimiCotangentSlabWorkspace,
        objective_workspace: KimiCotangentSlabWorkspace | None = None,
        teacher_boundary_archive: KimiBoundarySlabArchive | None = None,
        devices: Sequence[torch.device | str],
        epsilon: float,
        vocabulary_size: int,
        logit_scale: float = 1.0,
        base_seed: int = 20260815,
        lm_head_chunk_tokens: int = 128,
        slab_buffer_tokens: int = 256,
        direct_io: bool = True,
        load_config: InstantTensorLoadConfig | None = None,
    ):
        normalized = tuple(torch.device(value) for value in devices)
        if not normalized or any(
            value.type != "cuda" or value.index is None for value in normalized
        ):
            raise ValueError("suffix pipeline requires indexed CUDA devices")
        if len(set(normalized)) != len(normalized):
            raise ValueError("suffix pipeline devices must be unique")
        if workspace.manifest.get("chain_boundary") is not None:
            raise ValueError("suffix workspace is already initialized")
        if (objective_workspace is None) != (teacher_boundary_archive is None):
            raise ValueError(
                "objective cotangents require both a workspace and teacher boundaries"
            )
        if (
            workspace.token_count != boundary_archive.token_count
            or workspace.hidden_dimension != boundary_archive.hidden_dimension
            or workspace.num_layers != boundary_archive.num_layers
        ):
            raise ValueError("suffix workspace and boundary archive disagree")
        if objective_workspace is not None:
            if objective_workspace.manifest.get("chain_boundary") is not None:
                raise ValueError("objective suffix workspace is already initialized")
            if (
                objective_workspace.token_count != boundary_archive.token_count
                or objective_workspace.hidden_dimension
                != boundary_archive.hidden_dimension
                or objective_workspace.num_layers != boundary_archive.num_layers
                or objective_workspace.residual_block_size
                != boundary_archive.attn_res_block_size
            ):
                raise ValueError(
                    "objective workspace and anchor boundary archive disagree"
                )
        if teacher_boundary_archive is not None:
            if (
                not teacher_boundary_archive.complete
                or teacher_boundary_archive.token_count
                != boundary_archive.token_count
                or teacher_boundary_archive.hidden_dimension
                != boundary_archive.hidden_dimension
                or teacher_boundary_archive.num_layers != boundary_archive.num_layers
                or teacher_boundary_archive.attn_res_block_size
                != boundary_archive.attn_res_block_size
            ):
                raise ValueError("teacher and anchor boundary archives disagree")
        if lm_head_chunk_tokens <= 0 or slab_buffer_tokens <= 0:
            raise ValueError("suffix pipeline buffers must be positive")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.archive = boundary_archive
        self.workspace = workspace
        self.objective_workspace = objective_workspace
        self.teacher_archive = teacher_boundary_archive
        self.devices = normalized
        self.epsilon = float(epsilon)
        self.vocabulary_size = int(vocabulary_size)
        self.logit_scale = float(logit_scale)
        self.base_seed = int(base_seed)
        self.lm_head_chunk_tokens = int(lm_head_chunk_tokens)
        self.slab_buffer_tokens = int(slab_buffer_tokens)
        self.direct_io = bool(direct_io)
        self.load_config = load_config or InstantTensorLoadConfig()
        self.documents = boundary_archive.load_documents()
        if teacher_boundary_archive is not None:
            teacher_documents = teacher_boundary_archive.load_documents()
            if (
                teacher_documents.identifiers != self.documents.identifiers
                or not torch.equal(teacher_documents.offsets, self.documents.offsets)
                or not torch.equal(teacher_documents.input_ids, self.documents.input_ids)
            ):
                raise ValueError(
                    "teacher and anchor archives contain different token documents"
                )

    def _worker(
        self,
        *,
        partition: DocumentPartition,
        device: torch.device,
        suffix: OfficialKimiFisherSuffix,
        update,
        objective_update,
        sample_path: Path,
    ) -> tuple[
        SuffixWorkerRecord,
        list[SlabWriteReceipt],
        list[SlabWriteReceipt],
        float,
        int,
    ]:
        started = time.monotonic()
        torch.cuda.set_device(device)
        writer_id = f"device-{device.index:02d}"
        writers = {
            role: CudaBf16SlabWriter(
                update.writer(
                    role,
                    direct=self.direct_io,
                    writer_id=writer_id,
                    first_token=partition.first_token,
                    end_token=partition.end_token,
                ),
                device=device,
                buffer_tokens=self.slab_buffer_tokens,
            )
            for role in update.roles
        }
        objective_writers = (
            {
                role: CudaBf16SlabWriter(
                    objective_update.writer(
                        role,
                        direct=self.direct_io,
                        writer_id=writer_id,
                        first_token=partition.first_token,
                        end_token=partition.end_token,
                    ),
                    device=device,
                    buffer_tokens=self.slab_buffer_tokens,
                )
                for role in objective_update.roles
            }
            if objective_update is not None
            else {}
        )
        sample_writer = _SamplePairWriter(
            sample_path,
            first_token=partition.first_token,
            end_token=partition.end_token,
        )
        receipts: list[SlabWriteReceipt] = []
        objective_receipts: list[SlabWriteReceipt] = []
        objective_kl_sum = 0.0
        objective_tokens = 0
        try:
            for document in range(
                partition.first_document,
                partition.end_document,
            ):
                first, end = self.documents.document_extent(document)
                hidden_cpu = self.archive.read_cpu(
                    self.archive.num_layers,
                    first,
                    end,
                    direct=self.direct_io,
                    pin_memory=True,
                )
                residual_cpu = [
                    self.archive.read_cpu(
                        boundary,
                        first,
                        end,
                        direct=self.direct_io,
                        pin_memory=True,
                    )
                    for boundary in self.workspace.residual_boundaries
                ]
                hidden = hidden_cpu.to(device=device, non_blocking=True).unsqueeze(0)
                residuals = tuple(
                    value.to(device=device, non_blocking=True)
                    for value in residual_cpu
                )
                seed = document_fisher_seed(
                    self.base_seed,
                    document=document,
                    identifier=self.documents.identifiers[document],
                )
                objective_result = None
                if self.teacher_archive is None:
                    result = suffix.vjp(
                        final_boundary=hidden,
                        residual_inputs=residuals,
                        seed=seed,
                        lm_head_chunk_tokens=self.lm_head_chunk_tokens,
                    )
                else:
                    teacher_hidden_cpu = self.teacher_archive.read_cpu(
                        self.teacher_archive.num_layers,
                        first,
                        end,
                        direct=self.direct_io,
                        pin_memory=True,
                    )
                    teacher_residual_cpu = [
                        self.teacher_archive.read_cpu(
                            boundary,
                            first,
                            end,
                            direct=self.direct_io,
                            pin_memory=True,
                        )
                        for boundary in self.workspace.residual_boundaries
                    ]
                    teacher_normalized = suffix.normalized_hidden(
                        final_boundary=teacher_hidden_cpu.to(
                            device=device,
                            non_blocking=True,
                        ).unsqueeze(0),
                        residual_inputs=tuple(
                            value.to(device=device, non_blocking=True)
                            for value in teacher_residual_cpu
                        ),
                    )
                    channels = suffix.vjp_channels(
                        final_boundary=hidden,
                        residual_inputs=residuals,
                        teacher_normalized=teacher_normalized,
                        seed=seed,
                        lm_head_chunk_tokens=self.lm_head_chunk_tokens,
                    )
                    result = channels.fisher
                    objective_result = channels.objective
                writers["chain"].append(
                    result.chain_gradient.reshape(-1, self.archive.hidden_dimension)
                )
                for boundary, gradient in zip(
                    self.workspace.residual_boundaries,
                    result.residual_gradients,
                    strict=True,
                ):
                    writers[self.workspace.residual_role(boundary)].append(gradient)
                sample_writer.append(
                    result.first_tokens.cpu(),
                    result.second_tokens.cpu(),
                )
                if objective_result is not None:
                    objective_writers["chain"].append(
                        objective_result.chain_gradient.reshape(
                            -1,
                            self.archive.hidden_dimension,
                        )
                    )
                    for boundary, gradient in zip(
                        self.workspace.residual_boundaries,
                        objective_result.residual_gradients,
                        strict=True,
                    ):
                        objective_writers[
                            self.objective_workspace.residual_role(boundary)
                        ].append(gradient)
                    objective_kl_sum += objective_result.kl_sum
                    objective_tokens += objective_result.token_count
            for writer in writers.values():
                receipts.append(writer.finish())
            for writer in objective_writers.values():
                objective_receipts.append(writer.finish())
            sample_writer.finish()
        finally:
            for writer in writers.values():
                writer.close()
            for writer in objective_writers.values():
                writer.close()
            sample_writer.close()
        return (
            SuffixWorkerRecord(
                device=str(device),
                first_document=partition.first_document,
                end_document=partition.end_document,
                first_token=partition.first_token,
                end_token=partition.end_token,
                elapsed_seconds=time.monotonic() - started,
            ),
            receipts,
            objective_receipts,
            objective_kl_sum,
            objective_tokens,
        )

    def run(self) -> SuffixPipelineResult:
        """Compute every final-output Fisher seed and seal suffix cotangents."""

        started = time.monotonic()
        update = self.workspace.begin_suffix()
        objective_update = (
            self.objective_workspace.begin_suffix()
            if self.objective_workspace is not None
            else None
        )
        sample_path = self.workspace.root / SAMPLE_FILENAME
        fd = os.open(sample_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.posix_fallocate(
                fd,
                0,
                self.archive.token_count * 2 * torch.int32.itemsize,
            )
        finally:
            os.close(fd)
        partitions = self.documents.contiguous_partitions(len(self.devices))
        total_objective_kl_sum = 0.0
        total_objective_tokens = 0
        # InstantTensor's CUDA destination selection is process-global during
        # handle construction. Load each replica before starting worker
        # threads; document replay and slab writes remain data parallel.
        suffixes = tuple(
            OfficialKimiFisherSuffix(
                checkpoint=self.checkpoint,
                device=device,
                hidden_dimension=self.archive.hidden_dimension,
                vocabulary_size=self.vocabulary_size,
                residual_block_count=len(self.workspace.residual_boundaries),
                epsilon=self.epsilon,
                logit_scale=self.logit_scale,
                load_config=self.load_config,
            )
            for device in self.devices
        )
        records: list[SuffixWorkerRecord] = []
        with ThreadPoolExecutor(
            max_workers=len(self.devices),
            thread_name_prefix="kimi-fisher-suffix",
        ) as executor:
            futures = [
                executor.submit(
                    self._worker,
                    partition=partition,
                    device=device,
                    suffix=suffix,
                    update=update,
                    objective_update=objective_update,
                    sample_path=sample_path,
                )
                for partition, device, suffix in zip(
                    partitions,
                    self.devices,
                    suffixes,
                    strict=True,
                )
            ]
            for future in as_completed(futures):
                (
                    record,
                    receipts,
                    objective_receipts,
                    objective_kl_sum,
                    objective_tokens,
                ) = future.result()
                records.append(record)
                for receipt in receipts:
                    update.record(receipt)
                if objective_update is not None:
                    for receipt in objective_receipts:
                        objective_update.record(receipt)
                total_objective_kl_sum += objective_kl_sum
                total_objective_tokens += objective_tokens
        update.commit()
        if objective_update is not None:
            objective_update.commit()
        return SuffixPipelineResult(
            workers=tuple(sorted(records, key=lambda value: value.first_token)),
            sample_path=sample_path,
            sample_sha256=_sha256(sample_path),
            elapsed_seconds=time.monotonic() - started,
            objective_kl_sum=(
                total_objective_kl_sum
                if objective_update is not None
                else None
            ),
            objective_tokens=total_objective_tokens,
        )


__all__ = [
    "KimiSuffixPipeline",
    "SAMPLE_FILENAME",
    "SuffixPipelineResult",
    "SuffixWorkerRecord",
    "document_fisher_seed",
]

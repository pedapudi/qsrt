"""Exact student suffix state and frozen teacher distribution targets.

The student portion is a selective decoder-boundary archive containing the
suffix cut and every persistent residual prefix needed to resume execution at
that cut. The teacher portion is one BF16 slab containing the normalized input
to the frozen teacher LM head. A complete archive is therefore sufficient for
document-exact suffix replay without retaining teacher decoder boundaries.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import torch

from qsrt.kimi_boundary_slabs import DocumentIndex, KimiBoundarySlabArchive
from qsrt.kimi_cotangent_slabs import SequentialBf16SlabWriter, SlabWriteReceipt
from qsrt.suffix_recovery_training import (
    SuffixState,
    SuffixTrainingDocument,
)


KIND = "Kimi-K3 exact suffix-training archive"
SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
PARTIAL_MANIFEST_FILENAME = ".manifest.partial.json"
STUDENT_DIRECTORY = "student-boundaries"
TEACHER_TARGET_FILENAME = "teacher-normalized-lm-head-input.bf16"
TEACHER_RECEIPT_DIRECTORY = ".teacher-target-receipts"
HIDDEN_DTYPE = torch.bfloat16


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def _json_object(value: Mapping[str, object] | None) -> dict[str, object]:
    encoded = json.dumps({} if value is None else dict(value), sort_keys=True)
    result = json.loads(encoded)
    if not isinstance(result, dict):
        raise TypeError("suffix-training provenance must be a JSON object")
    return result


def _receipt_filename(writer_id: str) -> str:
    if not writer_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in writer_id
    ):
        raise ValueError("writer identity contains unsupported characters")
    return f"{writer_id}.json"


class KimiSuffixTrainingArchive:
    """Validated on-disk inputs for exact suffix recovery training."""

    def __init__(self, root: str | Path, *, require_complete: bool = False):
        self.root = Path(root).expanduser().resolve()
        complete_path = self.root / MANIFEST_FILENAME
        partial_path = self.root / PARTIAL_MANIFEST_FILENAME
        if complete_path.is_file():
            manifest_path = complete_path
        elif partial_path.is_file() and not require_complete:
            manifest_path = partial_path
        else:
            raise FileNotFoundError(
                complete_path if require_complete else partial_path
            )
        document = json.loads(manifest_path.read_text())
        if document.get("kind") != KIND:
            raise ValueError(f"{manifest_path}: unexpected archive kind")
        if int(document.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"{manifest_path}: unsupported schema version")
        self.manifest_path = manifest_path
        self.manifest: dict[str, Any] = document
        self.num_layers = int(document["num_layers"])
        self.hidden_dimension = int(document["hidden_dimension"])
        self.attn_res_block_size = int(document["attn_res_block_size"])
        self.cut_layer = int(document["cut_layer"])
        self.token_count = int(document["token_count"])
        self.document_count = int(document["document_count"])
        self.row_bytes = self.hidden_dimension * HIDDEN_DTYPE.itemsize
        if int(document["row_bytes"]) != self.row_bytes:
            raise ValueError("suffix-training archive row size is inconsistent")
        self.student = KimiBoundarySlabArchive(
            self.root / STUDENT_DIRECTORY,
            require_complete=require_complete,
        )
        if (
            self.student.num_layers != self.num_layers
            or self.student.hidden_dimension != self.hidden_dimension
            or self.student.attn_res_block_size != self.attn_res_block_size
            or self.student.token_count != self.token_count
            or self.student.document_count != self.document_count
        ):
            raise ValueError("student boundary archive geometry does not match")
        expected = tuple(range(0, self.cut_layer, self.attn_res_block_size)) + (
            self.cut_layer,
        )
        if self.student.retained_boundaries != expected:
            raise ValueError("student boundary archive has the wrong retained state")

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        documents: DocumentIndex,
        num_layers: int,
        hidden_dimension: int,
        attn_res_block_size: int,
        cut_layer: int,
        provenance: Mapping[str, object] | None = None,
    ) -> "KimiSuffixTrainingArchive":
        if (
            num_layers <= 0
            or hidden_dimension <= 0
            or attn_res_block_size <= 0
            or not 0 < cut_layer < num_layers
            or cut_layer % attn_res_block_size
        ):
            raise ValueError("suffix-training archive geometry is invalid")
        destination = Path(root).expanduser().resolve()
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(
                f"suffix-training destination is not empty: {destination}"
            )
        destination.mkdir(parents=True, exist_ok=True)
        (destination / TEACHER_RECEIPT_DIRECTORY).mkdir()
        retained = tuple(range(0, cut_layer, attn_res_block_size)) + (cut_layer,)
        student = KimiBoundarySlabArchive.create(
            destination / STUDENT_DIRECTORY,
            documents=documents,
            num_layers=num_layers,
            hidden_dimension=hidden_dimension,
            attn_res_block_size=attn_res_block_size,
            retained_boundaries=retained,
            provenance={
                "purpose": "student suffix replay state",
                "suffix_archive": str(destination),
            },
        )
        row_bytes = hidden_dimension * HIDDEN_DTYPE.itemsize
        target_bytes = documents.token_count * row_bytes
        target_path = destination / TEACHER_TARGET_FILENAME
        descriptor = os.open(
            target_path,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            os.posix_fallocate(descriptor, 0, target_bytes)
        except BaseException:
            os.close(descriptor)
            target_path.unlink(missing_ok=True)
            raise
        os.close(descriptor)
        manifest = {
            "kind": KIND,
            "schema_version": SCHEMA_VERSION,
            "complete": False,
            "num_layers": int(num_layers),
            "hidden_dimension": int(hidden_dimension),
            "hidden_dtype": "bfloat16",
            "row_bytes": row_bytes,
            "attn_res_block_size": int(attn_res_block_size),
            "cut_layer": int(cut_layer),
            "token_count": documents.token_count,
            "document_count": documents.document_count,
            "student_archive": STUDENT_DIRECTORY,
            "student_retained_boundaries": list(retained),
            "teacher_target": {
                "semantic_role": "normalized input to the frozen teacher LM head",
                "file": TEACHER_TARGET_FILENAME,
                "bytes": target_bytes,
                "sealed": False,
            },
            "provenance": _json_object(provenance),
        }
        _atomic_json(destination / PARTIAL_MANIFEST_FILENAME, manifest)
        del student
        return cls(destination)

    @property
    def complete(self) -> bool:
        return bool(self.manifest.get("complete", False))

    @property
    def teacher_target_path(self) -> Path:
        return self.root / TEACHER_TARGET_FILENAME

    @property
    def expected_slab_bytes(self) -> int:
        return self.token_count * self.row_bytes

    def load_documents(self) -> DocumentIndex:
        return self.student.load_documents()

    def teacher_target_writer(
        self,
        *,
        writer_id: str,
        first_token: int,
        end_token: int,
        direct: bool = True,
    ) -> SequentialBf16SlabWriter:
        if self.complete or bool(self.manifest["teacher_target"]["sealed"]):
            raise RuntimeError("teacher target cannot be modified after sealing")
        _receipt_filename(writer_id)
        return SequentialBf16SlabWriter(
            self.teacher_target_path,
            role="teacher-normalized",
            slot=0,
            token_count=self.token_count,
            hidden_dimension=self.hidden_dimension,
            direct=direct,
            writer_id=writer_id,
            first_token=first_token,
            end_token=end_token,
        )

    def record_teacher_target(self, receipt: SlabWriteReceipt) -> Path:
        if receipt.role != "teacher-normalized" or receipt.slot != 0:
            raise ValueError("receipt does not describe a teacher target extent")
        if receipt.bytes != (receipt.end_token - receipt.first_token) * self.row_bytes:
            raise ValueError("teacher target receipt has the wrong byte count")
        path = self.root / TEACHER_RECEIPT_DIRECTORY / _receipt_filename(
            receipt.writer_id
        )
        if path.exists():
            existing = SlabWriteReceipt(**json.loads(path.read_text()))
            if existing != receipt:
                raise FileExistsError(
                    f"incompatible teacher-target receipt already exists: {path}"
                )
        else:
            _atomic_json(path, receipt.to_json())
        return path

    def _teacher_receipts(self) -> tuple[SlabWriteReceipt, ...]:
        values = []
        for path in sorted((self.root / TEACHER_RECEIPT_DIRECTORY).glob("*.json")):
            value = json.loads(path.read_text())
            values.append(
                SlabWriteReceipt(
                    role=str(value["role"]),
                    slot=int(value["slot"]),
                    writer_id=str(value["writer_id"]),
                    first_token=int(value["first_token"]),
                    end_token=int(value["end_token"]),
                    bytes=int(value["bytes"]),
                    sha256=str(value["sha256"]),
                )
            )
        return tuple(values)

    def discard_teacher_target_receipts(self) -> tuple[str, ...]:
        """Discard unsealed target receipts before replaying complete extents."""

        if self.complete:
            raise RuntimeError("completed archive cannot be modified")
        if bool(self.manifest["teacher_target"]["sealed"]):
            raise RuntimeError("sealed teacher target cannot be modified")
        directory = self.root / TEACHER_RECEIPT_DIRECTORY
        discarded = tuple(path.name for path in sorted(directory.glob("*.json")))
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir()
        return discarded

    def seal_teacher_target(self) -> None:
        receipts = sorted(
            self._teacher_receipts(),
            key=lambda value: value.first_token,
        )
        if not receipts:
            raise ValueError("teacher target has no writer receipts")
        cursor = 0
        for receipt in receipts:
            if (
                receipt.role != "teacher-normalized"
                or receipt.slot != 0
                or receipt.first_token != cursor
                or receipt.end_token <= receipt.first_token
                or receipt.bytes
                != (receipt.end_token - receipt.first_token) * self.row_bytes
            ):
                raise ValueError(
                    f"teacher target receipt coverage breaks at token {cursor}"
                )
            cursor = receipt.end_token
        if cursor != self.token_count:
            raise ValueError(
                f"teacher target receipts end at {cursor}, expected {self.token_count}"
            )
        target = dict(self.manifest["teacher_target"])
        target["sealed"] = True
        target["extents"] = [receipt.to_json() for receipt in receipts]
        self.manifest["teacher_target"] = target
        _atomic_json(self.root / PARTIAL_MANIFEST_FILENAME, self.manifest)

    def seal(self) -> Path:
        if not self.student.complete:
            raise ValueError("student boundary archive is not sealed")
        if not bool(self.manifest["teacher_target"]["sealed"]):
            raise ValueError("teacher normalized target is not sealed")
        if self.teacher_target_path.stat().st_size != self.expected_slab_bytes:
            raise ValueError("teacher normalized target has the wrong size")
        self.manifest["student_manifest_sha256"] = _sha256(
            self.student.manifest_path
        )
        self.manifest["complete"] = True
        destination = self.root / MANIFEST_FILENAME
        _atomic_json(destination, self.manifest)
        (self.root / PARTIAL_MANIFEST_FILENAME).unlink()
        self.manifest_path = destination
        return destination

    def read_teacher_normalized(
        self,
        first_token: int,
        end_token: int,
        *,
        direct: bool = True,
        pin_memory: bool | None = None,
    ) -> torch.Tensor:
        if not 0 <= first_token < end_token <= self.token_count:
            raise ValueError("teacher target read is outside the archive")
        if pin_memory is None:
            pin_memory = direct
        value = torch.empty(
            (end_token - first_token, self.hidden_dimension),
            dtype=HIDDEN_DTYPE,
            pin_memory=pin_memory,
        )
        flags = os.O_RDONLY
        if direct:
            if not hasattr(os, "O_DIRECT"):
                raise RuntimeError("O_DIRECT is unavailable on this platform")
            flags |= os.O_DIRECT
            if value.data_ptr() % 512:
                raise ValueError("direct-I/O target buffers must be 512-byte aligned")
        descriptor = os.open(self.teacher_target_path, flags)
        try:
            byte_view = memoryview(value.view(torch.uint8).numpy())
            offset = first_token * self.row_bytes
            read = 0
            while read < byte_view.nbytes:
                count = os.preadv(descriptor, [byte_view[read:]], offset + read)
                if count <= 0:
                    raise EOFError("teacher target slab read ended early")
                read += count
        finally:
            os.close(descriptor)
        return value

    def load_document(
        self,
        document: int,
        *,
        direct: bool = True,
    ) -> SuffixTrainingDocument:
        documents = self.load_documents()
        first, end = documents.document_extent(document)
        hidden = self.student.read_cpu(
            self.cut_layer,
            first,
            end,
            direct=direct,
            pin_memory=direct,
        )
        residual = torch.stack(
            [
                self.student.read_cpu(
                    boundary,
                    first,
                    end,
                    direct=direct,
                    pin_memory=direct,
                )
                for boundary in self.student.residual_boundaries_before(
                    self.cut_layer
                )
            ],
            dim=1,
        )
        return SuffixTrainingDocument(
            identifier=documents.identifiers[document],
            student_boundary=SuffixState(hidden, residual),
            teacher_normalized=self.read_teacher_normalized(
                first,
                end,
                direct=direct,
                pin_memory=direct,
            ),
        )


__all__ = ["KimiSuffixTrainingArchive"]

"""Crash-safe BF16 cotangent storage for segmented Kimi-K3 replay.

One reusable chain slab carries the derivative at the decoder boundary being
reversed. One slab per 12-layer attention-residual boundary carries the
derivative contributed through that persistent skip input. Every reverse
segment writes inactive slots and switches all affected slots in one atomic
manifest update, so an interrupted segment leaves the preceding state usable.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from qsrt.kimi_boundary_slabs import KimiBoundarySlabArchive


KIND = "Kimi-K3 final-output Fisher cotangent workspace"
SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
SLOT_DIRECTORY = "slabs"
HIDDEN_DTYPE = torch.bfloat16


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _json_object(value: Mapping[str, object] | None) -> dict[str, object]:
    encoded = json.dumps({} if value is None else dict(value), sort_keys=True)
    result = json.loads(encoded)
    if not isinstance(result, dict):
        raise TypeError("cotangent provenance must be a JSON object")
    return result


def _slot_filename(role: str, slot: int) -> str:
    if slot not in (0, 1):
        raise ValueError("cotangent slot must be zero or one")
    return f"{role}-{slot}.bf16"


@dataclass(frozen=True)
class SlabWriteReceipt:
    """Digest and geometry of one completely rewritten slab slot."""

    role: str
    slot: int
    writer_id: str
    first_token: int
    end_token: int
    bytes: int
    sha256: str

    def to_json(self) -> dict[str, object]:
        return {
            "role": self.role,
            "slot": self.slot,
            "writer_id": self.writer_id,
            "first_token": self.first_token,
            "end_token": self.end_token,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


class SequentialBf16SlabWriter:
    """Write a complete BF16 slab in increasing token order."""

    def __init__(
        self,
        path: Path,
        *,
        role: str,
        slot: int,
        token_count: int,
        hidden_dimension: int,
        direct: bool,
        writer_id: str = "all",
        first_token: int = 0,
        end_token: int | None = None,
    ):
        self.path = path
        self.role = role
        self.slot = slot
        self.token_count = int(token_count)
        self.writer_id = str(writer_id)
        self.first_token = int(first_token)
        self.end_token = self.token_count if end_token is None else int(end_token)
        if (
            not self.writer_id
            or not 0 <= self.first_token < self.end_token <= self.token_count
        ):
            raise ValueError("cotangent writer extent or identity is invalid")
        self.hidden_dimension = int(hidden_dimension)
        self.row_bytes = self.hidden_dimension * HIDDEN_DTYPE.itemsize
        self.direct = bool(direct)
        flags = os.O_WRONLY
        if self.direct:
            if not hasattr(os, "O_DIRECT"):
                raise RuntimeError("O_DIRECT is unavailable on this platform")
            flags |= os.O_DIRECT
        self._fd = os.open(path, flags)
        self._next_token = self.first_token
        self._digest = hashlib.sha256()
        self._finished = False

    def __enter__(self) -> "SequentialBf16SlabWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def append(self, value: torch.Tensor) -> None:
        if self._finished:
            raise RuntimeError("cotangent slab writer is finished")
        if value.device.type != "cpu" or value.dtype != HIDDEN_DTYPE:
            raise TypeError("cotangent slab writes require CPU BF16 values")
        if value.ndim != 2 or value.shape[1] != self.hidden_dimension:
            raise ValueError("cotangent slab write has incompatible geometry")
        owned = value.contiguous()
        rows = int(owned.shape[0])
        if self._next_token + rows > self.end_token:
            raise ValueError("cotangent slab write exceeds the token extent")
        byte_view = memoryview(owned.view(torch.uint8).numpy())
        if self.direct and owned.data_ptr() % 512:
            raise ValueError("direct-I/O cotangent buffers must be 512-byte aligned")
        if self.direct and byte_view.nbytes % 512:
            raise ValueError("direct-I/O cotangent writes must be 512-byte aligned")
        offset = self._next_token * self.row_bytes
        written = 0
        while written < byte_view.nbytes:
            count = os.pwrite(self._fd, byte_view[written:], offset + written)
            if count <= 0:
                raise OSError("cotangent slab write made no progress")
            written += count
        self._digest.update(byte_view)
        self._next_token += rows

    def finish(self) -> SlabWriteReceipt:
        if self._finished:
            raise RuntimeError("cotangent slab writer is already finished")
        if self._next_token != self.end_token:
            raise ValueError(
                f"cotangent slab stopped at token {self._next_token}, "
                f"expected {self.end_token}"
            )
        os.fsync(self._fd)
        os.close(self._fd)
        self._finished = True
        return SlabWriteReceipt(
            role=self.role,
            slot=self.slot,
            writer_id=self.writer_id,
            first_token=self.first_token,
            end_token=self.end_token,
            bytes=(self.end_token - self.first_token) * self.row_bytes,
            sha256=self._digest.hexdigest(),
        )

    def close(self) -> None:
        if not self._finished:
            os.close(self._fd)
            self._finished = True


@dataclass
class _CudaWriteSlot:
    buffer: torch.Tensor
    producer_event: torch.cuda.Event
    event: torch.cuda.Event
    future: Future[None] | None = None
    source: torch.Tensor | None = None


class CudaBf16SlabWriter:
    """Overlap BF16 device-to-host copies with sequential slab writes."""

    def __init__(
        self,
        writer: SequentialBf16SlabWriter,
        *,
        device: torch.device | str,
        buffer_tokens: int,
        buffer_count: int = 2,
    ):
        target = torch.device(device)
        if target.type != "cuda" or target.index is None:
            raise ValueError("CUDA cotangent writer requires an indexed device")
        if buffer_tokens <= 0 or buffer_count < 2:
            raise ValueError("CUDA cotangent writer requires double buffering")
        self.writer = writer
        self.device = target
        self.copy_stream = torch.cuda.Stream(device=target)
        self._slots = [
            _CudaWriteSlot(
                buffer=torch.empty(
                    (buffer_tokens, writer.hidden_dimension),
                    dtype=HIDDEN_DTYPE,
                    device="cpu",
                    pin_memory=True,
                ),
                producer_event=torch.cuda.Event(),
                event=torch.cuda.Event(),
            )
            for _ in range(buffer_count)
        ]
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"cotangent-{writer.role}-{writer.slot}",
        )
        self._next_slot = 0
        self._finished = False

    @staticmethod
    def _write(
        writer: SequentialBf16SlabWriter,
        event: torch.cuda.Event,
        buffer: torch.Tensor,
        rows: int,
    ) -> None:
        event.synchronize()
        writer.append(buffer[:rows])

    @staticmethod
    def _wait(slot: _CudaWriteSlot) -> None:
        if slot.future is not None:
            slot.future.result()
            slot.future = None
        slot.source = None

    def append(self, value: torch.Tensor) -> None:
        if self._finished:
            raise RuntimeError("CUDA cotangent writer is finished")
        if value.device != self.device or value.dtype != HIDDEN_DTYPE:
            raise TypeError("CUDA cotangent write has the wrong device or dtype")
        if value.ndim != 2 or value.shape[1] != self.writer.hidden_dimension:
            raise ValueError("CUDA cotangent write has incompatible geometry")
        first = 0
        while first < value.shape[0]:
            slot = self._slots[self._next_slot]
            self._wait(slot)
            rows = min(int(slot.buffer.shape[0]), int(value.shape[0]) - first)
            source = value[first : first + rows]
            producer_stream = torch.cuda.current_stream(self.device)
            slot.producer_event.record(producer_stream)
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(slot.producer_event)
                slot.buffer[:rows].copy_(source, non_blocking=True)
                source.record_stream(self.copy_stream)
                slot.event.record(self.copy_stream)
            slot.source = source
            slot.future = self._executor.submit(
                self._write,
                self.writer,
                slot.event,
                slot.buffer,
                rows,
            )
            self._next_slot = (self._next_slot + 1) % len(self._slots)
            first += rows

    def finish(self) -> SlabWriteReceipt:
        if self._finished:
            raise RuntimeError("CUDA cotangent writer is already finished")
        for slot in self._slots:
            self._wait(slot)
        receipt = self.writer.finish()
        self._finished = True
        return receipt

    def close(self) -> None:
        for slot in self._slots:
            self._wait(slot)
        self._executor.shutdown(wait=True)
        self.writer.close()


class CotangentUpdate:
    """One atomic suffix initialization or reverse-segment update."""

    def __init__(
        self,
        workspace: "KimiCotangentSlabWorkspace",
        *,
        operation: str,
        output_chain_boundary: int,
        roles: Sequence[str],
    ):
        self.workspace = workspace
        self.operation = operation
        self.output_chain_boundary = int(output_chain_boundary)
        self.roles = tuple(roles)
        self.target_slots = {
            role: 0 if role not in workspace.active_slots else 1 - workspace.active_slots[role]
            for role in self.roles
        }
        self._receipts: dict[str, list[SlabWriteReceipt]] = {}
        self._committed = False
        for role, slot in self.target_slots.items():
            workspace._prepare_slot(role, slot)

    def writer(
        self,
        role: str,
        *,
        direct: bool = True,
        writer_id: str = "all",
        first_token: int = 0,
        end_token: int | None = None,
    ) -> SequentialBf16SlabWriter:
        if role not in self.target_slots:
            raise KeyError(role)
        return SequentialBf16SlabWriter(
            self.workspace.slot_path(role, self.target_slots[role]),
            role=role,
            slot=self.target_slots[role],
            token_count=self.workspace.token_count,
            hidden_dimension=self.workspace.hidden_dimension,
            direct=direct,
            writer_id=writer_id,
            first_token=first_token,
            end_token=end_token,
        )

    def record(self, receipt: SlabWriteReceipt) -> None:
        expected = self.target_slots.get(receipt.role)
        if expected is None or receipt.slot != expected:
            raise ValueError("cotangent receipt does not belong to this update")
        existing = self._receipts.setdefault(receipt.role, [])
        if any(value.writer_id == receipt.writer_id for value in existing):
            raise ValueError("cotangent writer identity was used more than once")
        existing.append(receipt)

    def commit(self) -> Path:
        if self._committed:
            raise RuntimeError("cotangent update is already committed")
        missing = set(self.roles) - set(self._receipts)
        if missing:
            raise ValueError(f"cotangent update is missing complete roles: {sorted(missing)}")
        for role, receipts in self._receipts.items():
            cursor = 0
            for receipt in sorted(receipts, key=lambda value: value.first_token):
                if receipt.first_token != cursor:
                    raise ValueError(
                        f"cotangent role {role} coverage breaks at token {cursor}"
                    )
                cursor = receipt.end_token
            if cursor != self.workspace.token_count:
                raise ValueError(
                    f"cotangent role {role} ends at token {cursor}, "
                    f"expected {self.workspace.token_count}"
                )
        self.workspace._commit(self)
        self._committed = True
        return self.workspace.manifest_path


class KimiCotangentSlabWorkspace:
    """Validated double-buffered reverse-replay state."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.manifest_path = self.root / MANIFEST_FILENAME
        document = json.loads(self.manifest_path.read_text())
        if document.get("kind") != KIND or document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{self.manifest_path}: incompatible cotangent workspace")
        self.manifest: dict[str, Any] = document
        self.token_count = int(document["token_count"])
        self.hidden_dimension = int(document["hidden_dimension"])
        self.num_layers = int(document["num_layers"])
        self.residual_block_size = int(document["residual_block_size"])
        self.row_bytes = self.hidden_dimension * HIDDEN_DTYPE.itemsize
        self.slab_bytes = self.token_count * self.row_bytes
        self.residual_boundaries = tuple(
            int(value) for value in document["residual_boundaries"]
        )
        self.active_slots = {
            str(role): int(slot)
            for role, slot in document.get("active_slots", {}).items()
        }

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        boundary_archive: KimiBoundarySlabArchive,
        provenance: Mapping[str, object] | None = None,
    ) -> "KimiCotangentSlabWorkspace":
        if not boundary_archive.complete:
            raise ValueError("cotangent replay requires a sealed boundary archive")
        destination = Path(root).expanduser().resolve()
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"cotangent workspace is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / SLOT_DIRECTORY).mkdir()
        residual_boundaries = list(
            range(
                0,
                boundary_archive.num_layers,
                boundary_archive.attn_res_block_size,
            )
        )
        document = {
            "kind": KIND,
            "schema_version": SCHEMA_VERSION,
            "boundary_archive": str(boundary_archive.root),
            "boundary_manifest": str(boundary_archive.manifest_path),
            "token_count": boundary_archive.token_count,
            "hidden_dimension": boundary_archive.hidden_dimension,
            "hidden_dtype": "bfloat16",
            "row_bytes": boundary_archive.row_bytes,
            "slab_bytes": boundary_archive.expected_slab_bytes,
            "num_layers": boundary_archive.num_layers,
            "residual_block_size": boundary_archive.attn_res_block_size,
            "residual_boundaries": residual_boundaries,
            "chain_boundary": None,
            "active_slots": {},
            "slot_receipts": {},
            "completed_operations": [],
            "provenance": _json_object(provenance),
        }
        _atomic_json(destination / MANIFEST_FILENAME, document)
        return cls(destination)

    @staticmethod
    def residual_role(boundary: int) -> str:
        return f"residual-{boundary:03d}"

    def slot_path(self, role: str, slot: int) -> Path:
        return self.root / SLOT_DIRECTORY / _slot_filename(role, slot)

    def _prepare_slot(self, role: str, slot: int) -> Path:
        path = self.slot_path(role, slot)
        if path.exists():
            if path.stat().st_size != self.slab_bytes:
                raise ValueError(f"cotangent slab has the wrong size: {path}")
            return path
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.posix_fallocate(fd, 0, self.slab_bytes)
        except BaseException:
            os.close(fd)
            path.unlink(missing_ok=True)
            raise
        os.close(fd)
        return path

    def begin_suffix(self) -> CotangentUpdate:
        if self.manifest.get("chain_boundary") is not None:
            raise RuntimeError("cotangent suffix is already initialized")
        roles = ["chain"] + [
            self.residual_role(boundary) for boundary in self.residual_boundaries
        ]
        return CotangentUpdate(
            self,
            operation="suffix",
            output_chain_boundary=self.num_layers,
            roles=roles,
        )

    def begin_segment(self, first_layer: int) -> CotangentUpdate:
        if first_layer not in self.residual_boundaries:
            raise ValueError("reverse segment must begin on a residual boundary")
        end_layer = min(first_layer + self.residual_block_size, self.num_layers)
        if self.manifest.get("chain_boundary") != end_layer:
            raise ValueError(
                f"cotangent chain is at boundary {self.manifest.get('chain_boundary')}, "
                f"not required boundary {end_layer}"
            )
        touched = [
            self.residual_role(boundary)
            for boundary in self.residual_boundaries
            if boundary < first_layer
        ]
        return CotangentUpdate(
            self,
            operation=f"segment-{first_layer:03d}-{end_layer:03d}",
            output_chain_boundary=first_layer,
            roles=("chain", *touched),
        )

    def _commit(self, update: CotangentUpdate) -> None:
        active = dict(self.active_slots)
        receipts = dict(self.manifest.get("slot_receipts", {}))
        for role, extents in update._receipts.items():
            slot = update.target_slots[role]
            active[role] = slot
            receipts[f"{role}:{slot}"] = {
                "bytes": self.slab_bytes,
                "extents": [
                    value.to_json()
                    for value in sorted(extents, key=lambda item: item.first_token)
                ],
            }
        operations = list(self.manifest.get("completed_operations", []))
        operations.append(
            {
                "operation": update.operation,
                "chain_boundary": update.output_chain_boundary,
                "roles": list(update.roles),
            }
        )
        self.manifest["active_slots"] = active
        self.manifest["slot_receipts"] = receipts
        self.manifest["chain_boundary"] = update.output_chain_boundary
        self.manifest["completed_operations"] = operations
        _atomic_json(self.manifest_path, self.manifest)
        self.active_slots = active

    def _read_role(
        self,
        role: str,
        first_token: int,
        end_token: int,
        *,
        direct: bool,
        pin_memory: bool | None,
    ) -> torch.Tensor:
        if role not in self.active_slots:
            raise RuntimeError(f"cotangent role is not initialized: {role}")
        if not 0 <= first_token < end_token <= self.token_count:
            raise ValueError("cotangent read is outside the token extent")
        if pin_memory is None:
            pin_memory = direct
        value = torch.empty(
            (end_token - first_token, self.hidden_dimension),
            dtype=HIDDEN_DTYPE,
            device="cpu",
            pin_memory=pin_memory,
        )
        flags = os.O_RDONLY
        if direct:
            if not hasattr(os, "O_DIRECT"):
                raise RuntimeError("O_DIRECT is unavailable on this platform")
            flags |= os.O_DIRECT
            if value.data_ptr() % 512:
                raise ValueError("direct-I/O cotangent buffers must be 512-byte aligned")
        fd = os.open(self.slot_path(role, self.active_slots[role]), flags)
        try:
            byte_view = memoryview(value.view(torch.uint8).numpy())
            offset = first_token * self.row_bytes
            read = 0
            while read < byte_view.nbytes:
                count = os.preadv(fd, [byte_view[read:]], offset + read)
                if count <= 0:
                    raise EOFError("cotangent slab read ended early")
                read += count
        finally:
            os.close(fd)
        return value

    def read_chain(
        self,
        first_token: int,
        end_token: int,
        *,
        direct: bool = True,
        pin_memory: bool | None = None,
    ) -> torch.Tensor:
        return self._read_role(
            "chain",
            first_token,
            end_token,
            direct=direct,
            pin_memory=pin_memory,
        )

    def read_residual(
        self,
        boundary: int,
        first_token: int,
        end_token: int,
        *,
        direct: bool = True,
        pin_memory: bool | None = None,
    ) -> torch.Tensor:
        if boundary not in self.residual_boundaries:
            raise ValueError("unknown residual boundary")
        return self._read_role(
            self.residual_role(boundary),
            first_token,
            end_token,
            direct=direct,
            pin_memory=pin_memory,
        )


__all__ = [
    "CotangentUpdate",
    "CudaBf16SlabWriter",
    "KimiCotangentSlabWorkspace",
    "SequentialBf16SlabWriter",
    "SlabWriteReceipt",
]

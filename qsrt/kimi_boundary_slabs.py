"""Exact Kimi-K3 layer-boundary storage for out-of-core reverse replay.

The archive stores every decoder boundary as one token-major BF16 slab.  A
separate document index defines sequence boundaries and preserves the input
tokens.  Kimi-K3's attention-residual prefix is derived from decoder inputs at
layers divisible by ``attn_res_block_size``; it is therefore reconstructed
from boundary slabs instead of being serialized a second time.

Large slabs use a header-free raw layout so readers can issue aligned direct
I/O into pinned memory without parsing or faulting a multi-terabyte mapping.
The manifest, document metadata, and token index remain small structured
files.  A boundary is usable only after disjoint writer receipts cover the
complete token range and the coordinator seals it in the manifest.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file


KIND = "Kimi-K3 exact layer-boundary slab archive"
SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
PARTIAL_MANIFEST_FILENAME = ".manifest.partial.json"
DOCUMENT_TENSORS_FILENAME = "document-tensors.safetensors"
DOCUMENT_METADATA_FILENAME = "documents.json"
BOUNDARY_DIRECTORY = "boundaries"
RECEIPT_DIRECTORY = ".receipts"
HIDDEN_DTYPE = torch.bfloat16


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_safetensors(
    path: Path,
    tensors: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file(dict(tensors), temporary, metadata=dict(metadata))
    os.replace(temporary, path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_object(value: Mapping[str, object] | None) -> dict[str, object]:
    result = {} if value is None else dict(value)
    encoded = json.dumps(result, sort_keys=True)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError("archive provenance must be a JSON object")
    return decoded


def _boundary_filename(boundary: int) -> str:
    return f"hidden-{boundary:03d}.bf16"


def _receipt_filename(writer_id: str) -> str:
    if not writer_id or writer_id in {".", ".."}:
        raise ValueError("writer_id must be nonempty")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in writer_id):
        raise ValueError("writer_id may contain only letters, digits, '-' and '_'")
    return f"{writer_id}.json"


@dataclass(frozen=True)
class DocumentPartition:
    """One worker's contiguous range of complete documents and tokens."""

    index: int
    first_document: int
    end_document: int
    first_token: int
    end_token: int

    @property
    def document_count(self) -> int:
        return self.end_document - self.first_document

    @property
    def token_count(self) -> int:
        return self.end_token - self.first_token

    def to_json(self) -> dict[str, int]:
        return {
            "index": self.index,
            "first_document": self.first_document,
            "end_document": self.end_document,
            "first_token": self.first_token,
            "end_token": self.end_token,
        }


@dataclass(frozen=True)
class DocumentIndex:
    """Tokenized complete documents in one deterministic global order."""

    input_ids: torch.Tensor
    offsets: torch.Tensor
    identifiers: tuple[str, ...]

    def __post_init__(self) -> None:
        input_ids = self.input_ids
        offsets = self.offsets
        if input_ids.device.type != "cpu" or offsets.device.type != "cpu":
            raise ValueError("document index tensors must be on CPU")
        if input_ids.ndim != 1:
            raise ValueError("input_ids must be one-dimensional")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must be int32 or int64")
        if offsets.ndim != 1 or offsets.dtype != torch.int64:
            raise TypeError("offsets must be one-dimensional int64")
        if offsets.numel() < 2:
            raise ValueError("the archive requires at least one document")
        values = offsets.tolist()
        if values[0] != 0 or values[-1] != input_ids.numel():
            raise ValueError("document offsets must span every input token")
        if any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError("every document must contain at least one token")
        if len(self.identifiers) != len(values) - 1:
            raise ValueError("document identifier count does not match offsets")
        if len(set(self.identifiers)) != len(self.identifiers):
            raise ValueError("document identifiers must be unique")
        if any(not identifier for identifier in self.identifiers):
            raise ValueError("document identifiers must be nonempty")
        if input_ids.numel() and int(input_ids.min()) < 0:
            raise ValueError("input token IDs must be nonnegative")

    @property
    def document_count(self) -> int:
        return self.offsets.numel() - 1

    @property
    def token_count(self) -> int:
        return self.input_ids.numel()

    def document_extent(self, index: int) -> tuple[int, int]:
        if not 0 <= index < self.document_count:
            raise IndexError(index)
        return int(self.offsets[index]), int(self.offsets[index + 1])

    def contiguous_partitions(self, count: int) -> tuple[DocumentPartition, ...]:
        """Balance tokens while assigning each worker complete documents."""

        if not 1 <= count <= self.document_count:
            raise ValueError("partition count must be in 1..document_count")
        offsets = self.offsets.tolist()
        boundaries = [0]
        for partition in range(1, count):
            remaining = count - partition
            lower = boundaries[-1] + 1
            upper = self.document_count - remaining
            token_cursor = offsets[boundaries[-1]]
            target = token_cursor + (
                self.token_count - token_cursor
            ) / (remaining + 1)
            insertion = bisect.bisect_left(offsets, target, lower, upper + 1)
            candidates = {
                max(lower, min(upper, insertion)),
                max(lower, min(upper, insertion - 1)),
            }
            chosen = min(
                candidates,
                key=lambda value: (abs(offsets[value] - target), value),
            )
            boundaries.append(chosen)
        boundaries.append(self.document_count)
        return tuple(
            DocumentPartition(
                index=index,
                first_document=first,
                end_document=end,
                first_token=offsets[first],
                end_token=offsets[end],
            )
            for index, (first, end) in enumerate(
                zip(boundaries, boundaries[1:])
            )
        )


@dataclass(frozen=True)
class SlabExtentReceipt:
    """Completed byte range written by one archive worker."""

    writer_id: str
    first_token: int
    end_token: int
    bytes: int
    sha256: str

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "SlabExtentReceipt":
        return cls(
            writer_id=str(value["writer_id"]),
            first_token=int(value["first_token"]),
            end_token=int(value["end_token"]),
            bytes=int(value["bytes"]),
            sha256=str(value["sha256"]),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "writer_id": self.writer_id,
            "first_token": self.first_token,
            "end_token": self.end_token,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


class BoundaryExtentWriter:
    """Append one ordered token extent to a preallocated boundary slab."""

    def __init__(
        self,
        archive: "KimiBoundarySlabArchive",
        *,
        boundary: int,
        writer_id: str,
        first_token: int,
        end_token: int,
        direct: bool,
    ):
        archive._validate_boundary(boundary)
        if not 0 <= first_token < end_token <= archive.token_count:
            raise ValueError("writer token extent is outside the archive")
        self.archive = archive
        self.boundary = boundary
        self.writer_id = writer_id
        self.first_token = first_token
        self.end_token = end_token
        self.next_token = first_token
        self.direct = bool(direct)
        self._digest = hashlib.sha256()
        path = archive.boundary_path(boundary)
        if not path.is_file():
            raise FileNotFoundError(path)
        flags = os.O_WRONLY
        if self.direct:
            if not hasattr(os, "O_DIRECT"):
                raise RuntimeError("O_DIRECT is unavailable on this platform")
            flags |= os.O_DIRECT
        self._fd = os.open(path, flags)
        self._closed = False

    def __enter__(self) -> "BoundaryExtentWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def append(self, hidden_states: torch.Tensor) -> None:
        if self._closed:
            raise RuntimeError("boundary extent writer is closed")
        if hidden_states.device.type != "cpu":
            raise ValueError("append expects a CPU tensor")
        if hidden_states.dtype != HIDDEN_DTYPE or hidden_states.ndim != 2:
            raise TypeError("hidden states must be a two-dimensional BF16 tensor")
        if hidden_states.shape[1] != self.archive.hidden_dimension:
            raise ValueError("hidden-state width does not match the archive")
        value = hidden_states.contiguous()
        token_count = int(value.shape[0])
        if self.next_token + token_count > self.end_token:
            raise ValueError("hidden-state append exceeds the writer extent")
        byte_view = memoryview(value.view(torch.uint8).numpy())
        if self.direct and value.data_ptr() % 512:
            raise ValueError("direct-I/O buffers must be at least 512-byte aligned")
        if self.direct and byte_view.nbytes % 512:
            raise ValueError("direct-I/O writes must be a multiple of 512 bytes")
        offset = self.next_token * self.archive.row_bytes
        written = 0
        while written < byte_view.nbytes:
            count = os.pwrite(self._fd, byte_view[written:], offset + written)
            if count <= 0:
                raise OSError("boundary slab write made no progress")
            written += count
        self._digest.update(byte_view)
        self.next_token += token_count

    def finish(self) -> SlabExtentReceipt:
        if self._closed:
            raise RuntimeError("boundary extent writer is closed")
        if self.next_token != self.end_token:
            raise ValueError(
                f"writer stopped at token {self.next_token}, expected {self.end_token}"
            )
        os.fsync(self._fd)
        receipt = SlabExtentReceipt(
            writer_id=self.writer_id,
            first_token=self.first_token,
            end_token=self.end_token,
            bytes=(self.end_token - self.first_token) * self.archive.row_bytes,
            sha256=self._digest.hexdigest(),
        )
        path = self.archive.receipt_path(self.boundary, self.writer_id)
        if path.exists():
            existing = SlabExtentReceipt.from_json(json.loads(path.read_text()))
            if existing != receipt:
                raise FileExistsError(f"incompatible boundary receipt already exists: {path}")
        else:
            _atomic_json(path, receipt.to_json())
        return receipt

    def close(self) -> None:
        if not self._closed:
            os.close(self._fd)
            self._closed = True


@dataclass
class _CudaWriteSlot:
    buffer: torch.Tensor
    producer_event: torch.cuda.Event
    event: torch.cuda.Event
    future: Future[None] | None = None
    source: torch.Tensor | None = None


class CudaBoundaryExtentWriter:
    """Overlap CUDA-to-host transfer with ordered direct slab writes."""

    def __init__(
        self,
        writer: BoundaryExtentWriter,
        *,
        device: torch.device | str,
        buffer_tokens: int,
        buffer_count: int = 2,
    ):
        if buffer_tokens <= 0 or buffer_count < 2:
            raise ValueError("CUDA slab writer requires positive double buffering")
        self.writer = writer
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("CUDA slab writer requires an indexed CUDA device")
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self._slots = [
            _CudaWriteSlot(
                buffer=torch.empty(
                    (buffer_tokens, writer.archive.hidden_dimension),
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
            thread_name_prefix=f"boundary-{writer.boundary:03d}-{writer.writer_id}",
        )
        self._next_slot = 0
        self._finished = False

    def __enter__(self) -> "CudaBoundaryExtentWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            if exc_type is None and not self._finished:
                self.finish()
        finally:
            self.close()

    @staticmethod
    def _write_slot(
        writer: BoundaryExtentWriter,
        event: torch.cuda.Event,
        buffer: torch.Tensor,
        token_count: int,
    ) -> None:
        event.synchronize()
        writer.append(buffer[:token_count])

    @staticmethod
    def _wait_slot(slot: _CudaWriteSlot) -> None:
        if slot.future is not None:
            slot.future.result()
            slot.future = None
        slot.source = None

    def append(self, hidden_states: torch.Tensor) -> None:
        if self._finished:
            raise RuntimeError("CUDA slab writer is finished")
        if hidden_states.device != self.device:
            raise ValueError("hidden states are on the wrong CUDA device")
        if hidden_states.dtype != HIDDEN_DTYPE or hidden_states.ndim != 2:
            raise TypeError("hidden states must be a two-dimensional BF16 tensor")
        if hidden_states.shape[1] != self.writer.archive.hidden_dimension:
            raise ValueError("hidden-state width does not match the archive")

        first = 0
        while first < hidden_states.shape[0]:
            slot = self._slots[self._next_slot]
            self._wait_slot(slot)
            token_count = min(
                int(slot.buffer.shape[0]), int(hidden_states.shape[0]) - first
            )
            source = hidden_states[first : first + token_count]
            producer_stream = torch.cuda.current_stream(self.device)
            slot.producer_event.record(producer_stream)
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(slot.producer_event)
                slot.buffer[:token_count].copy_(source, non_blocking=True)
                source.record_stream(self.copy_stream)
                slot.event.record(self.copy_stream)
            slot.source = source
            slot.future = self._executor.submit(
                self._write_slot,
                self.writer,
                slot.event,
                slot.buffer,
                token_count,
            )
            self._next_slot = (self._next_slot + 1) % len(self._slots)
            first += token_count

    def finish(self) -> SlabExtentReceipt:
        if self._finished:
            raise RuntimeError("CUDA slab writer is already finished")
        for slot in self._slots:
            self._wait_slot(slot)
        receipt = self.writer.finish()
        self._finished = True
        return receipt

    def close(self) -> None:
        for slot in self._slots:
            self._wait_slot(slot)
        self._executor.shutdown(wait=True)
        self.writer.close()


class KimiBoundarySlabArchive:
    """Manifest and direct-I/O access for exact decoder-boundary slabs."""

    def __init__(self, root: Path, *, require_complete: bool = False):
        self.root = root.expanduser().resolve()
        completed = self.root / MANIFEST_FILENAME
        partial = self.root / PARTIAL_MANIFEST_FILENAME
        if completed.is_file():
            manifest_path = completed
        elif partial.is_file() and not require_complete:
            manifest_path = partial
        else:
            raise FileNotFoundError(completed if require_complete else partial)
        document = json.loads(manifest_path.read_text())
        if document.get("kind") != KIND:
            raise ValueError(f"{manifest_path}: unexpected archive kind")
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{manifest_path}: unsupported schema version")
        self.manifest_path = manifest_path
        self.manifest: dict[str, Any] = document
        self.num_layers = int(document["num_layers"])
        self.hidden_dimension = int(document["hidden_dimension"])
        self.attn_res_block_size = int(document["attn_res_block_size"])
        self.token_count = int(document["token_count"])
        self.document_count = int(document["document_count"])
        self.row_bytes = self.hidden_dimension * HIDDEN_DTYPE.itemsize
        if int(document["row_bytes"]) != self.row_bytes:
            raise ValueError("manifest row size is inconsistent with BF16 geometry")

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        documents: DocumentIndex,
        num_layers: int,
        hidden_dimension: int,
        attn_res_block_size: int,
        retained_boundaries: Collection[int] | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> "KimiBoundarySlabArchive":
        if num_layers <= 0 or hidden_dimension <= 0 or attn_res_block_size <= 0:
            raise ValueError("archive geometry must be positive")
        destination = root.expanduser().resolve()
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"archive destination is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / BOUNDARY_DIRECTORY).mkdir()
        (destination / RECEIPT_DIRECTORY).mkdir()
        retained = (
            tuple(range(num_layers + 1))
            if retained_boundaries is None
            else tuple(sorted({int(value) for value in retained_boundaries}))
        )
        if not retained:
            raise ValueError("an archive must retain at least one decoder boundary")
        if retained[0] < 0 or retained[-1] > num_layers:
            raise ValueError("a retained boundary is outside the archive geometry")

        input_ids = documents.input_ids.to(torch.int32).contiguous()
        offsets = documents.offsets.contiguous()
        token_path = destination / DOCUMENT_TENSORS_FILENAME
        _atomic_safetensors(
            token_path,
            {"input_ids": input_ids, "document_offsets": offsets},
            metadata={
                "kind": "Kimi-K3 replay document tensor index",
                "schema_version": str(SCHEMA_VERSION),
            },
        )
        metadata_document = {
            "kind": "Kimi-K3 replay document metadata",
            "schema_version": SCHEMA_VERSION,
            "identifiers": list(documents.identifiers),
        }
        _atomic_json(destination / DOCUMENT_METADATA_FILENAME, metadata_document)
        token_index_sha256 = hashlib.sha256(token_path.read_bytes()).hexdigest()
        document_metadata_sha256 = _sha256_bytes(
            (destination / DOCUMENT_METADATA_FILENAME).read_bytes()
        )
        manifest = {
            "kind": KIND,
            "schema_version": SCHEMA_VERSION,
            "complete": False,
            "num_layers": int(num_layers),
            "boundary_count": len(retained),
            "retained_boundaries": list(retained),
            "hidden_dimension": int(hidden_dimension),
            "hidden_dtype": "bfloat16",
            "row_bytes": int(hidden_dimension) * HIDDEN_DTYPE.itemsize,
            "attn_res_block_size": int(attn_res_block_size),
            "token_count": documents.token_count,
            "document_count": documents.document_count,
            "document_tensors": DOCUMENT_TENSORS_FILENAME,
            "document_tensors_sha256": token_index_sha256,
            "document_metadata": DOCUMENT_METADATA_FILENAME,
            "document_metadata_sha256": document_metadata_sha256,
            "provenance": _json_object(provenance),
            "sealed_boundaries": {},
        }
        _atomic_json(destination / PARTIAL_MANIFEST_FILENAME, manifest)
        return cls(destination)

    @property
    def expected_slab_bytes(self) -> int:
        return self.token_count * self.row_bytes

    @property
    def retained_boundaries(self) -> tuple[int, ...]:
        raw = self.manifest.get("retained_boundaries")
        if raw is None:
            return tuple(range(self.num_layers + 1))
        if not isinstance(raw, list):
            raise TypeError("retained_boundaries must be a JSON array")
        values = tuple(int(value) for value in raw)
        if (
            not values
            or tuple(sorted(set(values))) != values
            or values[0] < 0
            or values[-1] > self.num_layers
        ):
            raise ValueError("retained_boundaries is not canonical")
        if int(self.manifest["boundary_count"]) != len(values):
            raise ValueError("boundary_count does not match retained_boundaries")
        return values

    @property
    def complete(self) -> bool:
        return bool(self.manifest.get("complete", False))

    def load_documents(self) -> DocumentIndex:
        tensors = load_file(self.root / DOCUMENT_TENSORS_FILENAME, device="cpu")
        metadata = json.loads((self.root / DOCUMENT_METADATA_FILENAME).read_text())
        return DocumentIndex(
            input_ids=tensors["input_ids"],
            offsets=tensors["document_offsets"],
            identifiers=tuple(str(value) for value in metadata["identifiers"]),
        )

    def _validate_boundary(self, boundary: int) -> None:
        if not 0 <= boundary <= self.num_layers:
            raise IndexError(f"boundary must be in 0..{self.num_layers}")

    def boundary_path(self, boundary: int) -> Path:
        self._validate_boundary(boundary)
        return self.root / BOUNDARY_DIRECTORY / _boundary_filename(boundary)

    def receipt_path(self, boundary: int, writer_id: str) -> Path:
        self._validate_boundary(boundary)
        return (
            self.root
            / RECEIPT_DIRECTORY
            / f"boundary-{boundary:03d}"
            / _receipt_filename(writer_id)
        )

    def prepare_boundary(self, boundary: int) -> Path:
        if self.complete:
            raise RuntimeError("completed archive cannot be modified")
        if boundary not in self.retained_boundaries:
            raise ValueError(f"boundary {boundary} is not retained by this archive")
        path = self.boundary_path(boundary)
        if path.exists():
            if path.stat().st_size != self.expected_slab_bytes:
                raise ValueError(f"existing boundary has the wrong size: {path}")
            return path
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        fd = os.open(path, flags, 0o600)
        try:
            os.posix_fallocate(fd, 0, self.expected_slab_bytes)
        except BaseException:
            os.close(fd)
            path.unlink(missing_ok=True)
            raise
        os.close(fd)
        return path

    def sealed_boundary_prefix(self) -> tuple[int, ...]:
        """Return the retained-boundary prefix committed to the archive."""

        raw = self.manifest.get("sealed_boundaries", {})
        if not isinstance(raw, dict):
            raise TypeError("sealed_boundaries must be a JSON object")
        try:
            sealed = tuple(sorted(int(value) for value in raw))
        except ValueError as error:
            raise ValueError("sealed boundary keys must be integers") from error
        if any(boundary < 0 or boundary > self.num_layers for boundary in sealed):
            raise ValueError("sealed boundary key is outside the archive geometry")
        expected = self.retained_boundaries[: len(sealed)]
        if sealed != expected:
            raise ValueError(
                "sealed boundaries must form a prefix of the retained boundaries"
            )
        return sealed

    def discard_unsealed_receipts(self) -> tuple[int, ...]:
        """Discard incomplete writer receipts before recomputing unsealed boundaries."""

        if self.complete:
            raise RuntimeError("completed archive cannot be modified")
        sealed = set(self.sealed_boundary_prefix())
        discarded: list[int] = []
        for boundary in range(self.num_layers + 1):
            if boundary in sealed:
                continue
            directory = self.root / RECEIPT_DIRECTORY / f"boundary-{boundary:03d}"
            if directory.exists():
                shutil.rmtree(directory)
                discarded.append(boundary)
        return tuple(discarded)

    def extent_writer(
        self,
        boundary: int,
        *,
        writer_id: str,
        first_token: int,
        end_token: int,
        direct: bool = True,
    ) -> BoundaryExtentWriter:
        return BoundaryExtentWriter(
            self,
            boundary=boundary,
            writer_id=writer_id,
            first_token=first_token,
            end_token=end_token,
            direct=direct,
        )

    def cuda_extent_writer(
        self,
        boundary: int,
        *,
        writer_id: str,
        first_token: int,
        end_token: int,
        device: torch.device | str,
        buffer_tokens: int,
        buffer_count: int = 2,
        direct: bool = True,
    ) -> CudaBoundaryExtentWriter:
        """Create an asynchronous CUDA-to-slab writer for one token extent."""

        return CudaBoundaryExtentWriter(
            self.extent_writer(
                boundary,
                writer_id=writer_id,
                first_token=first_token,
                end_token=end_token,
                direct=direct,
            ),
            device=device,
            buffer_tokens=buffer_tokens,
            buffer_count=buffer_count,
        )

    def _load_receipts(self, boundary: int) -> list[SlabExtentReceipt]:
        directory = self.root / RECEIPT_DIRECTORY / f"boundary-{boundary:03d}"
        if not directory.is_dir():
            return []
        return [
            SlabExtentReceipt.from_json(json.loads(path.read_text()))
            for path in sorted(directory.glob("*.json"))
        ]

    def seal_boundary(self, boundary: int) -> dict[str, object]:
        if self.complete:
            raise RuntimeError("completed archive cannot be modified")
        if boundary not in self.retained_boundaries:
            raise ValueError(f"boundary {boundary} is not retained by this archive")
        path = self.boundary_path(boundary)
        if path.stat().st_size != self.expected_slab_bytes:
            raise ValueError(f"boundary slab has the wrong size: {path}")
        receipts = sorted(
            self._load_receipts(boundary), key=lambda receipt: receipt.first_token
        )
        if not receipts:
            raise ValueError(f"boundary {boundary} has no writer receipts")
        cursor = 0
        for receipt in receipts:
            if receipt.first_token != cursor:
                raise ValueError(
                    f"boundary {boundary} receipt coverage breaks at token {cursor}"
                )
            expected = (receipt.end_token - receipt.first_token) * self.row_bytes
            if receipt.bytes != expected or receipt.end_token <= receipt.first_token:
                raise ValueError(f"boundary {boundary} has an invalid receipt")
            cursor = receipt.end_token
        if cursor != self.token_count:
            raise ValueError(
                f"boundary {boundary} receipts end at {cursor}, expected {self.token_count}"
            )
        record: dict[str, object] = {
            "file": str(path.relative_to(self.root)),
            "bytes": self.expected_slab_bytes,
            "extents": [receipt.to_json() for receipt in receipts],
        }
        sealed = dict(self.manifest.get("sealed_boundaries", {}))
        key = str(boundary)
        existing = sealed.get(key)
        if existing is not None and existing != record:
            raise ValueError(f"boundary {boundary} was already sealed differently")
        sealed[key] = record
        self.manifest["sealed_boundaries"] = sealed
        _atomic_json(self.root / PARTIAL_MANIFEST_FILENAME, self.manifest)
        return record

    def seal(self) -> Path:
        expected = {str(boundary) for boundary in self.retained_boundaries}
        actual = set(self.manifest.get("sealed_boundaries", {}))
        if actual != expected:
            missing = sorted(expected - actual, key=int)
            raise ValueError(f"archive is missing sealed boundaries: {missing[:8]}")
        self.manifest["complete"] = True
        destination = self.root / MANIFEST_FILENAME
        _atomic_json(destination, self.manifest)
        (self.root / PARTIAL_MANIFEST_FILENAME).unlink()
        self.manifest_path = destination
        return destination

    def read_cpu(
        self,
        boundary: int,
        first_token: int,
        end_token: int,
        *,
        direct: bool = True,
        pin_memory: bool | None = None,
    ) -> torch.Tensor:
        self._validate_boundary(boundary)
        if not 0 <= first_token < end_token <= self.token_count:
            raise ValueError("read token extent is outside the archive")
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
                raise ValueError("direct-I/O buffers must be at least 512-byte aligned")
        fd = os.open(self.boundary_path(boundary), flags)
        try:
            byte_view = memoryview(value.view(torch.uint8).numpy())
            offset = first_token * self.row_bytes
            read = 0
            while read < byte_view.nbytes:
                count = os.preadv(fd, [byte_view[read:]], offset + read)
                if count <= 0:
                    raise EOFError("boundary slab read ended early")
                read += count
        finally:
            os.close(fd)
        return value

    def residual_boundaries_before(self, layer: int) -> tuple[int, ...]:
        if not 0 <= layer < self.num_layers:
            raise IndexError(layer)
        return tuple(range(0, layer, self.attn_res_block_size))

    def reconstruct_block_residual(
        self,
        *,
        layer: int,
        first_token: int,
        end_token: int,
        device: torch.device | str,
        direct: bool = True,
    ) -> torch.Tensor:
        boundaries = self.residual_boundaries_before(layer)
        if not boundaries:
            return torch.empty(
                (end_token - first_token, 0, self.hidden_dimension),
                dtype=HIDDEN_DTYPE,
                device=device,
            )
        values = [
            self.read_cpu(
                boundary,
                first_token,
                end_token,
                direct=direct,
                pin_memory=True,
            ).to(device=device, non_blocking=True)
            for boundary in boundaries
        ]
        return torch.stack(values, dim=1)


def verify_partition_cover(
    partitions: Iterable[DocumentPartition],
    *,
    document_count: int,
    token_count: int,
) -> None:
    """Validate complete ordered coverage by document and token extents."""

    items = sorted(partitions, key=lambda item: item.index)
    document_cursor = token_cursor = 0
    for expected_index, item in enumerate(items):
        if item.index != expected_index:
            raise ValueError("partition indices must be contiguous from zero")
        if item.first_document != document_cursor or item.first_token != token_cursor:
            raise ValueError("partition coverage is not contiguous")
        if item.document_count <= 0 or item.token_count <= 0:
            raise ValueError("every partition must contain documents and tokens")
        document_cursor = item.end_document
        token_cursor = item.end_token
    if document_cursor != document_count or token_cursor != token_count:
        raise ValueError("partitions do not cover the complete archive")


__all__ = [
    "BoundaryExtentWriter",
    "DocumentIndex",
    "DocumentPartition",
    "KimiBoundarySlabArchive",
    "SlabExtentReceipt",
    "verify_partition_cover",
]

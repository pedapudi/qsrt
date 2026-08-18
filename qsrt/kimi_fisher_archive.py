"""Replayable Kimi-K3 layer-boundary states for curvature capture.

The archive stores one hidden-state tensor at every decoder boundary.  Kimi's
attention-residual state is append-only, so only the newly appended vector is
stored at a boundary; a reader reconstructs the residual prefix required by
any layer.  This keeps the archive exact without repeatedly serializing the
growing residual tensor.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from qsrt.kimi_stream import StreamState, tensor_digest


KIND = "Kimi-K3 layer-boundary replay archive"
SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
PARTIAL_MANIFEST_FILENAME = ".manifest.partial.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_safetensors(
    path: Path,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file(dict(tensors), temporary, metadata=dict(metadata))
    os.replace(temporary, path)


def _json_object(value: Mapping[str, object] | None) -> dict[str, object]:
    result = {} if value is None else dict(value)
    try:
        encoded = json.dumps(result, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("archive provenance must be JSON serializable") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError("archive provenance must be a JSON object")
    return decoded


def _boundary_filename(next_layer: int) -> str:
    return f"boundary-{next_layer:03d}.safetensors"


@dataclass(frozen=True)
class BoundaryRecord:
    """Inventory entry for one decoder-boundary file."""

    next_layer: int
    residual_count: int
    file: str
    sha256: str
    bytes: int
    hidden_sha256: str
    appended_residual_sha256: str | None

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "BoundaryRecord":
        appended = value.get("appended_residual_sha256")
        if appended is not None and not isinstance(appended, str):
            raise TypeError("appended residual digest must be a string or null")
        return cls(
            next_layer=int(value["next_layer"]),
            residual_count=int(value["residual_count"]),
            file=str(value["file"]),
            sha256=str(value["sha256"]),
            bytes=int(value["bytes"]),
            hidden_sha256=str(value["hidden_sha256"]),
            appended_residual_sha256=appended,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "next_layer": self.next_layer,
            "residual_count": self.residual_count,
            "file": self.file,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "hidden_sha256": self.hidden_sha256,
            "appended_residual_sha256": self.appended_residual_sha256,
        }


class KimiBoundaryArchiveWriter:
    """Incrementally write one rectangular Kimi-K3 replay batch."""

    def __init__(
        self,
        root: Path,
        *,
        num_layers: int,
        residual_block_size: int,
        provenance: Mapping[str, object] | None = None,
        resume: bool = False,
    ):
        if num_layers <= 0:
            raise ValueError("archive layer count must be positive")
        if residual_block_size <= 0:
            raise ValueError("attention-residual block size must be positive")
        self.root = root.expanduser().resolve()
        self.num_layers = int(num_layers)
        self.residual_block_size = int(residual_block_size)
        self.provenance = _json_object(provenance)
        self.records: list[BoundaryRecord] = []
        self._input_ids: torch.Tensor | None = None
        self._batch_shape: tuple[int, int] | None = None
        self._hidden_dimension: int | None = None
        self._dtype: torch.dtype | None = None

        completed = self.root / MANIFEST_FILENAME
        partial = self.root / PARTIAL_MANIFEST_FILENAME
        if completed.exists():
            raise FileExistsError(f"completed boundary archive already exists: {completed}")
        if partial.exists():
            if not resume:
                raise FileExistsError(
                    f"partial boundary archive already exists: {partial}; use resume"
                )
            self._restore_partial(partial)
        else:
            if resume:
                raise FileNotFoundError(partial)
            self.root.mkdir(parents=True, exist_ok=True)
            unexpected = sorted(self.root.glob("boundary-*.safetensors"))
            if unexpected:
                raise FileExistsError(
                    f"boundary archive directory contains untracked files: {unexpected[:3]}"
                )

    @property
    def next_layer(self) -> int:
        return 0 if not self.records else self.records[-1].next_layer + 1

    def _manifest(self, *, complete: bool) -> dict[str, object]:
        return {
            "kind": KIND,
            "schema_version": SCHEMA_VERSION,
            "complete": complete,
            "num_layers": self.num_layers,
            "residual_block_size": self.residual_block_size,
            "batch_shape": None if self._batch_shape is None else list(self._batch_shape),
            "hidden_dimension": self._hidden_dimension,
            "hidden_dtype": None if self._dtype is None else str(self._dtype),
            "input_ids_sha256": (
                None if self._input_ids is None else tensor_digest(self._input_ids)
            ),
            "provenance": self.provenance,
            "boundaries": [record.to_json() for record in self.records],
        }

    def _restore_partial(self, path: Path) -> None:
        document = json.loads(path.read_text())
        expected = {
            "kind": KIND,
            "schema_version": SCHEMA_VERSION,
            "complete": False,
            "num_layers": self.num_layers,
            "residual_block_size": self.residual_block_size,
            "provenance": self.provenance,
        }
        for key, value in expected.items():
            if document.get(key) != value:
                raise ValueError(f"partial boundary archive has incompatible {key}")
        records = document.get("boundaries")
        if not isinstance(records, list) or not records:
            raise ValueError("partial boundary archive contains no boundaries")
        self.records = [BoundaryRecord.from_json(record) for record in records]
        self._validate_inventory(verify_hashes=True)
        first = load_file(self.root / self.records[0].file, device="cpu")
        self._input_ids = first["input_ids"]
        hidden = first["hidden_states"]
        self._batch_shape = tuple(map(int, hidden.shape[:2]))
        self._hidden_dimension = int(hidden.shape[2])
        self._dtype = hidden.dtype

    def _validate_inventory(self, *, verify_hashes: bool) -> None:
        expected_next = 0
        expected_residuals = 0
        for record in self.records:
            if record.next_layer != expected_next:
                raise ValueError("boundary archive layer inventory is not contiguous")
            expected_count = (
                expected_next + self.residual_block_size - 1
            ) // self.residual_block_size
            if record.residual_count != expected_count:
                raise ValueError("boundary archive residual count violates Kimi geometry")
            path = self.root / record.file
            if not path.is_file() or path.stat().st_size != record.bytes:
                raise ValueError(f"boundary archive file is missing or truncated: {path}")
            if verify_hashes and _sha256(path) != record.sha256:
                raise ValueError(f"boundary archive file hash mismatch: {path}")
            appended = record.appended_residual_sha256 is not None
            if appended != (record.residual_count == expected_residuals + 1):
                raise ValueError("boundary archive residual append inventory is invalid")
            expected_residuals = record.residual_count
            expected_next += 1

    def append(self, state: StreamState) -> BoundaryRecord:
        """Store the next state, including only a newly appended residual vector."""

        state.validate()
        expected_layer = 0 if not self.records else self.records[-1].next_layer + 1
        if state.next_layer != expected_layer:
            raise ValueError(
                f"archive expected boundary {expected_layer}, got {state.next_layer}"
            )
        if state.next_layer > self.num_layers:
            raise ValueError("archive received a state past the configured model depth")

        hidden = state.hidden_states.detach().contiguous().cpu()
        input_ids = state.input_ids.detach().contiguous().cpu()
        residual = state.block_residual.detach().contiguous().cpu()
        batch_shape = tuple(map(int, hidden.shape[:2]))
        hidden_dimension = int(hidden.shape[2])
        if self._input_ids is None:
            self._input_ids = input_ids
            self._batch_shape = batch_shape
            self._hidden_dimension = hidden_dimension
            self._dtype = hidden.dtype
        else:
            if not torch.equal(input_ids, self._input_ids):
                raise ValueError("input IDs changed between archived boundaries")
            if batch_shape != self._batch_shape or hidden_dimension != self._hidden_dimension:
                raise ValueError("hidden-state geometry changed between boundaries")
            if hidden.dtype != self._dtype:
                raise ValueError("hidden-state dtype changed between boundaries")

        expected_residuals = (
            state.next_layer + self.residual_block_size - 1
        ) // self.residual_block_size
        if residual.shape[1] != expected_residuals:
            raise ValueError(
                "attention-residual count does not match the decoder boundary"
            )
        previous_residuals = 0 if not self.records else self.records[-1].residual_count
        if expected_residuals - previous_residuals not in (0, 1):
            raise ValueError("attention-residual state is not append-only")

        tensors: dict[str, torch.Tensor] = {"hidden_states": hidden}
        if state.next_layer == 0:
            tensors["input_ids"] = input_ids
        appended_digest = None
        if expected_residuals > previous_residuals:
            appended = residual[:, -1, :].contiguous()
            tensors["appended_residual"] = appended
            appended_digest = tensor_digest(appended)

        filename = _boundary_filename(state.next_layer)
        path = self.root / filename
        if path.exists():
            raise FileExistsError(path)
        metadata = {
            "kind": KIND,
            "schema_version": str(SCHEMA_VERSION),
            "next_layer": str(state.next_layer),
            "residual_count": str(expected_residuals),
        }
        _atomic_safetensors(path, tensors, metadata)
        record = BoundaryRecord(
            next_layer=state.next_layer,
            residual_count=expected_residuals,
            file=filename,
            sha256=_sha256(path),
            bytes=path.stat().st_size,
            hidden_sha256=tensor_digest(hidden),
            appended_residual_sha256=appended_digest,
        )
        self.records.append(record)
        _atomic_json(self.root / PARTIAL_MANIFEST_FILENAME, self._manifest(complete=False))
        return record

    def seal(self) -> Path:
        """Seal a complete boundary inventory after all decoder layers exist."""

        if len(self.records) != self.num_layers + 1:
            raise ValueError(
                f"boundary archive has {len(self.records)} of {self.num_layers + 1} states"
            )
        self._validate_inventory(verify_hashes=False)
        destination = self.root / MANIFEST_FILENAME
        _atomic_json(destination, self._manifest(complete=True))
        (self.root / PARTIAL_MANIFEST_FILENAME).unlink()
        return destination


class KimiBoundaryArchive:
    """Validated read-only access to one complete replay batch."""

    def __init__(self, root: Path, *, verify_hashes: bool = False):
        self.root = root.expanduser().resolve()
        manifest_path = self.root / MANIFEST_FILENAME
        document = json.loads(manifest_path.read_text())
        expected = {
            "kind": KIND,
            "schema_version": SCHEMA_VERSION,
            "complete": True,
        }
        for key, value in expected.items():
            if document.get(key) != value:
                raise ValueError(f"boundary archive has invalid {key}")
        self.manifest: dict[str, Any] = document
        self.num_layers = int(document["num_layers"])
        self.residual_block_size = int(document["residual_block_size"])
        raw_records = document.get("boundaries")
        if not isinstance(raw_records, list):
            raise TypeError("boundary archive inventory must be a list")
        self.records = [BoundaryRecord.from_json(record) for record in raw_records]
        if len(self.records) != self.num_layers + 1:
            raise ValueError("boundary archive inventory is incomplete")
        self._record_by_layer = {record.next_layer: record for record in self.records}
        if len(self._record_by_layer) != len(self.records):
            raise ValueError("boundary archive contains duplicate layers")
        self._validate_inventory(verify_hashes=verify_hashes)
        self._input_ids: torch.Tensor | None = None
        self._residual_appends: list[torch.Tensor] | None = None

    def _validate_inventory(self, *, verify_hashes: bool) -> None:
        residual_count = 0
        for next_layer, record in enumerate(self.records):
            if record.next_layer != next_layer:
                raise ValueError("boundary archive layer inventory is not contiguous")
            expected_count = (
                next_layer + self.residual_block_size - 1
            ) // self.residual_block_size
            if record.residual_count != expected_count:
                raise ValueError("boundary archive residual count is invalid")
            path = self.root / record.file
            if not path.is_file() or path.stat().st_size != record.bytes:
                raise ValueError(f"boundary archive file is missing or truncated: {path}")
            if verify_hashes and _sha256(path) != record.sha256:
                raise ValueError(f"boundary archive file hash mismatch: {path}")
            appended = record.appended_residual_sha256 is not None
            if appended != (expected_count == residual_count + 1):
                raise ValueError("boundary archive residual append inventory is invalid")
            residual_count = expected_count

    def _load_boundary(self, next_layer: int) -> dict[str, torch.Tensor]:
        try:
            record = self._record_by_layer[next_layer]
        except KeyError as exc:
            raise IndexError(f"archive has no decoder boundary {next_layer}") from exc
        path = self.root / record.file
        with safe_open(path, framework="pt", device="cpu") as reader:
            metadata = reader.metadata() or {}
            if metadata.get("next_layer") != str(next_layer):
                raise ValueError(f"boundary file has wrong layer metadata: {path}")
            keys = set(reader.keys())
        expected = {"hidden_states"}
        if next_layer == 0:
            expected.add("input_ids")
        if record.appended_residual_sha256 is not None:
            expected.add("appended_residual")
        if keys != expected:
            raise ValueError(f"boundary file has unexpected tensors: {path}")
        tensors = load_file(path, device="cpu")
        if tensor_digest(tensors["hidden_states"]) != record.hidden_sha256:
            raise ValueError(f"boundary hidden-state digest mismatch: {path}")
        if "appended_residual" in tensors and (
            tensor_digest(tensors["appended_residual"])
            != record.appended_residual_sha256
        ):
            raise ValueError(f"boundary residual digest mismatch: {path}")
        return tensors

    @property
    def input_ids(self) -> torch.Tensor:
        if self._input_ids is None:
            self._input_ids = self._load_boundary(0)["input_ids"]
            expected = self.manifest.get("input_ids_sha256")
            if tensor_digest(self._input_ids) != expected:
                raise ValueError("boundary archive input-ID digest mismatch")
        return self._input_ids

    def _all_residual_appends(self) -> list[torch.Tensor]:
        if self._residual_appends is None:
            self._residual_appends = []
            for record in self.records:
                if record.appended_residual_sha256 is not None:
                    tensors = self._load_boundary(record.next_layer)
                    self._residual_appends.append(tensors["appended_residual"])
        return self._residual_appends

    def state(
        self,
        next_layer: int,
        *,
        device: torch.device | str = "cpu",
    ) -> StreamState:
        """Reconstruct the exact inter-layer state at ``next_layer``."""

        tensors = self._load_boundary(next_layer)
        hidden = tensors["hidden_states"]
        record = self._record_by_layer[next_layer]
        appends = self._all_residual_appends()[: record.residual_count]
        tokens = hidden.shape[0] * hidden.shape[1]
        if appends:
            residual = torch.stack(appends, dim=1)
        else:
            residual = hidden.new_zeros(tokens, 0, hidden.shape[2])
        state = StreamState(
            next_layer=next_layer,
            input_ids=self.input_ids,
            hidden_states=hidden,
            block_residual=residual,
        )
        state.validate()
        target = torch.device(device)
        return StreamState(
            next_layer=next_layer,
            input_ids=state.input_ids.to(target),
            hidden_states=state.hidden_states.to(target),
            block_residual=state.block_residual.to(target),
        )


__all__ = [
    "BoundaryRecord",
    "KIND",
    "KimiBoundaryArchive",
    "KimiBoundaryArchiveWriter",
    "MANIFEST_FILENAME",
    "SCHEMA_VERSION",
]

"""Layer-shared output curvature for official Kimi-K3 routed experts.

Each routed MoE layer stores the empirical second moment of the gradient at
the route-weighted W2 sum before the latent RMSNorm and output projection.
Two document-disjoint sums are retained so factor stability can be measured
without repeating decoder replay.

Factors are committed by decoder segment. A sealed pending segment can be
promoted after the cotangent workspace commits the corresponding reverse
operation, making interrupted replay recoverable at segment boundaries.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file


KIND = "Kimi-K3 routed-output empirical Fisher archive"
SEGMENT_KIND = "Kimi-K3 routed-output empirical Fisher segment"
SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
SEGMENT_DIRECTORY = "segments"
PENDING_DIRECTORY = ".pending"


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
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
        raise TypeError("output-factor provenance must be a JSON object")
    return result


def _segment_name(first_layer: int, end_layer: int) -> str:
    return f"segment-{first_layer:03d}-{end_layer:03d}"


def _layer_filename(layer: int) -> str:
    return f"layer-{layer:03d}.safetensors"


def document_factor_split(identifier: str) -> str:
    """Assign a complete document to one stable output-factor split."""

    if not identifier:
        raise ValueError("document identifier must be nonempty")
    digest = hashlib.sha256(
        f"Kimi-K3 routed-output Fisher split v1\n{identifier}".encode()
    ).digest()
    return "a" if digest[0] & 1 == 0 else "b"


@dataclass(frozen=True)
class OutputFactorSums:
    """Document-disjoint FP32 gradient outer-product sums for one layer."""

    split_a: torch.Tensor
    split_a_rows: int
    split_b: torch.Tensor
    split_b_rows: int

    def validate(self, *, dimension: int) -> None:
        for name, value in (("split_a", self.split_a), ("split_b", self.split_b)):
            if value.device.type != "cpu" or value.dtype != torch.float32:
                raise TypeError(f"{name} factor sum must be CPU FP32")
            if tuple(value.shape) != (dimension, dimension):
                raise ValueError(f"{name} factor sum has incompatible geometry")
            if not bool(torch.all(torch.isfinite(value))):
                raise FloatingPointError(f"{name} factor sum contains non-finite values")
        if self.split_a_rows <= 0 or self.split_b_rows <= 0:
            raise ValueError("both output-factor document splits require gradient rows")


@dataclass(frozen=True)
class OutputFactorLayerRecord:
    """Stored tensors and support for one routed MoE layer."""

    layer: int
    file: str
    sha256: str
    bytes: int
    split_a_rows: int
    split_b_rows: int

    @property
    def rows(self) -> int:
        return self.split_a_rows + self.split_b_rows

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "OutputFactorLayerRecord":
        return cls(
            layer=int(value["layer"]),
            file=str(value["file"]),
            sha256=str(value["sha256"]),
            bytes=int(value["bytes"]),
            split_a_rows=int(value["split_a_rows"]),
            split_b_rows=int(value["split_b_rows"]),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "file": self.file,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "split_a_rows": self.split_a_rows,
            "split_b_rows": self.split_b_rows,
            "rows": self.rows,
        }


class OutputFactorSegmentWriter:
    """Build one durable, not-yet-promoted reverse-segment result."""

    def __init__(
        self,
        archive: "KimiOutputFactorArchive",
        *,
        first_layer: int,
        end_layer: int,
    ):
        if not 0 <= first_layer < end_layer <= archive.num_layers:
            raise ValueError("output-factor segment range is invalid")
        self.archive = archive
        self.first_layer = int(first_layer)
        self.end_layer = int(end_layer)
        self.operation = f"segment-{first_layer:03d}-{end_layer:03d}"
        self.name = _segment_name(first_layer, end_layer)
        self.path = archive.root / PENDING_DIRECTORY / self.name
        committed = archive.root / SEGMENT_DIRECTORY / self.name
        if committed.exists():
            raise FileExistsError(f"output-factor segment is already committed: {committed}")
        if self.path.exists():
            raise FileExistsError(f"pending output-factor segment already exists: {self.path}")
        self.path.mkdir(parents=True)
        self.records: list[OutputFactorLayerRecord] = []
        self._sealed = False
        self._committed = False

    def add(self, layer: int, sums: OutputFactorSums) -> OutputFactorLayerRecord:
        if self._sealed:
            raise RuntimeError("output-factor segment is sealed")
        if not self.first_layer <= layer < self.end_layer:
            raise ValueError("output-factor layer lies outside the segment")
        if any(record.layer == layer for record in self.records):
            raise ValueError(f"output-factor layer {layer} was added twice")
        if layer not in self.archive.expected_layers:
            raise ValueError(f"decoder layer {layer} has no routed MoE output")
        sums.validate(dimension=self.archive.dimension)
        split_a = (sums.split_a / float(sums.split_a_rows)).contiguous()
        split_b = (sums.split_b / float(sums.split_b_rows)).contiguous()
        rows = sums.split_a_rows + sums.split_b_rows
        combined = (
            (sums.split_a + sums.split_b) / float(rows)
        ).contiguous()
        tensors = {
            "output_hessian": (combined + combined.T).mul_(0.5),
            "output_hessian_split_a": (split_a + split_a.T).mul_(0.5),
            "output_hessian_split_b": (split_b + split_b.T).mul_(0.5),
        }
        path = self.path / _layer_filename(layer)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        save_file(
            tensors,
            temporary,
            metadata={
                "kind": KIND,
                "schema_version": str(SCHEMA_VERSION),
                "layer": str(layer),
                "semantic_point": (
                    "route-weighted expert W2 sum before routed latent "
                    "RMSNorm and output projection"
                ),
                "split_a_rows": str(sums.split_a_rows),
                "split_b_rows": str(sums.split_b_rows),
                "rows": str(rows),
                "damping": "none",
            },
        )
        os.replace(temporary, path)
        record = OutputFactorLayerRecord(
            layer=layer,
            file=path.name,
            sha256=_sha256(path),
            bytes=path.stat().st_size,
            split_a_rows=sums.split_a_rows,
            split_b_rows=sums.split_b_rows,
        )
        self.records.append(record)
        return record

    def seal(self) -> Path:
        if self._sealed:
            raise RuntimeError("output-factor segment is already sealed")
        expected = {
            layer
            for layer in self.archive.expected_layers
            if self.first_layer <= layer < self.end_layer
        }
        actual = {record.layer for record in self.records}
        if actual != expected:
            raise ValueError(
                f"output-factor segment layers differ: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        manifest = {
            "kind": SEGMENT_KIND,
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "operation": self.operation,
            "first_layer": self.first_layer,
            "end_layer": self.end_layer,
            "layers": [
                record.to_json() for record in sorted(self.records, key=lambda item: item.layer)
            ],
        }
        path = self.path / MANIFEST_FILENAME
        _atomic_json(path, manifest)
        self._sealed = True
        return path

    def commit(self) -> Path:
        if not self._sealed:
            raise RuntimeError("output-factor segment must be sealed before commit")
        if self._committed:
            raise RuntimeError("output-factor segment is already committed")
        destination = self.archive.root / SEGMENT_DIRECTORY / self.name
        os.replace(self.path, destination)
        self.archive._refresh_manifest()
        self._committed = True
        return destination


class KimiOutputFactorArchive:
    """Segment-atomic routed-output factor inventory."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.manifest_path = self.root / MANIFEST_FILENAME
        document = json.loads(self.manifest_path.read_text())
        if document.get("kind") != KIND or document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{self.manifest_path}: incompatible output-factor archive")
        self.manifest = document
        self.num_layers = int(document["num_layers"])
        self.dimension = int(document["dimension"])
        self.expected_layers = tuple(int(value) for value in document["expected_layers"])

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        num_layers: int,
        dimension: int,
        expected_layers: Sequence[int],
        provenance: Mapping[str, object] | None = None,
    ) -> "KimiOutputFactorArchive":
        layers = tuple(sorted(set(int(value) for value in expected_layers)))
        if num_layers <= 0 or dimension <= 0 or not layers:
            raise ValueError("output-factor archive geometry must be positive")
        if layers[0] < 0 or layers[-1] >= num_layers:
            raise ValueError("expected output-factor layer lies outside the model")
        destination = Path(root).expanduser().resolve()
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"output-factor destination is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / SEGMENT_DIRECTORY).mkdir()
        (destination / PENDING_DIRECTORY).mkdir()
        document = {
            "kind": KIND,
            "schema_version": SCHEMA_VERSION,
            "complete": False,
            "num_layers": int(num_layers),
            "dimension": int(dimension),
            "expected_layers": list(layers),
            "semantic_point": (
                "route-weighted expert W2 sum before routed latent RMSNorm "
                "and output projection"
            ),
            "factor_dtype": "float32",
            "damping": "none",
            "segments": [],
            "provenance": _json_object(provenance),
        }
        _atomic_json(destination / MANIFEST_FILENAME, document)
        return cls(destination)

    def begin_segment(
        self,
        first_layer: int,
        end_layer: int,
    ) -> OutputFactorSegmentWriter:
        return OutputFactorSegmentWriter(
            self,
            first_layer=first_layer,
            end_layer=end_layer,
        )

    def layer_path(self, layer: int, *, verify_hash: bool = True) -> Path:
        """Resolve one committed layer factor and verify its inventory entry."""

        requested = int(layer)
        if requested not in self.expected_layers:
            raise ValueError(f"decoder layer {requested} has no routed output factor")
        for segment in self.manifest.get("segments", []):
            directory = self.root / str(segment["directory"])
            for value in segment.get("layers", []):
                record = OutputFactorLayerRecord.from_json(value)
                if record.layer != requested:
                    continue
                path = directory / record.file
                if not path.is_file() or path.stat().st_size != record.bytes:
                    raise ValueError(f"output-factor tensor is missing or truncated: {path}")
                if verify_hash and _sha256(path) != record.sha256:
                    raise ValueError(f"output-factor tensor hash mismatch: {path}")
                return path
        raise FileNotFoundError(
            f"output-factor archive has not committed decoder layer {requested}"
        )

    def _segment_records(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        observed_layers: set[int] = set()
        for directory in sorted((self.root / SEGMENT_DIRECTORY).glob("segment-*-*")):
            path = directory / MANIFEST_FILENAME
            document = json.loads(path.read_text())
            if (
                document.get("kind") != SEGMENT_KIND
                or document.get("schema_version") != SCHEMA_VERSION
                or document.get("complete") is not True
            ):
                raise ValueError(f"invalid output-factor segment manifest: {path}")
            layers = document.get("layers")
            if not isinstance(layers, list):
                raise TypeError(f"output-factor segment has no layer inventory: {path}")
            for value in layers:
                layer = int(value["layer"])
                if layer in observed_layers:
                    raise ValueError(f"output-factor layer {layer} appears in two segments")
                observed_layers.add(layer)
                tensor_path = directory / str(value["file"])
                if (
                    not tensor_path.is_file()
                    or tensor_path.stat().st_size != int(value["bytes"])
                    or _sha256(tensor_path) != value["sha256"]
                ):
                    raise ValueError(f"output-factor tensor is missing or corrupt: {tensor_path}")
                with safe_open(tensor_path, framework="pt", device="cpu") as reader:
                    expected_keys = {
                        "output_hessian",
                        "output_hessian_split_a",
                        "output_hessian_split_b",
                    }
                    if set(reader.keys()) != expected_keys:
                        raise ValueError(f"output-factor tensor inventory is invalid: {tensor_path}")
                    if any(
                        tuple(reader.get_slice(key).get_shape())
                        != (self.dimension, self.dimension)
                        for key in expected_keys
                    ):
                        raise ValueError(f"output-factor geometry is invalid: {tensor_path}")
            records.append(
                {
                    "directory": str(directory.relative_to(self.root)),
                    "manifest_sha256": _sha256(path),
                    "operation": str(document["operation"]),
                    "first_layer": int(document["first_layer"]),
                    "end_layer": int(document["end_layer"]),
                    "layers": layers,
                }
            )
        return records

    def _refresh_manifest(self) -> None:
        segments = self._segment_records()
        layers = {
            int(layer["layer"])
            for segment in segments
            for layer in segment["layers"]
        }
        self.manifest["segments"] = segments
        self.manifest["complete"] = layers == set(self.expected_layers)
        _atomic_json(self.manifest_path, self.manifest)

    def recover_pending(self, completed_operations: Sequence[str]) -> tuple[Path, ...]:
        """Promote sealed factors whose cotangent segment already committed."""

        completed = set(str(value) for value in completed_operations)
        promoted: list[Path] = []
        for directory in sorted((self.root / PENDING_DIRECTORY).glob("segment-*-*")):
            manifest_path = directory / MANIFEST_FILENAME
            if not manifest_path.is_file():
                continue
            document = json.loads(manifest_path.read_text())
            operation = str(document.get("operation"))
            if operation not in completed:
                continue
            destination = self.root / SEGMENT_DIRECTORY / directory.name
            if destination.exists():
                raise FileExistsError(destination)
            os.replace(directory, destination)
            promoted.append(destination)
        if promoted:
            self._refresh_manifest()
        return tuple(promoted)

    def discard_uncommitted_pending(
        self,
        completed_operations: Sequence[str],
    ) -> tuple[Path, ...]:
        """Remove segment output that has no matching cotangent commit."""

        completed = set(str(value) for value in completed_operations)
        discarded: list[Path] = []
        for directory in sorted((self.root / PENDING_DIRECTORY).glob("segment-*-*")):
            manifest_path = directory / MANIFEST_FILENAME
            operation = None
            if manifest_path.is_file():
                try:
                    document = json.loads(manifest_path.read_text())
                    operation = str(document.get("operation"))
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    operation = None
            if operation in completed:
                continue
            shutil.rmtree(directory)
            discarded.append(directory)
        return tuple(discarded)

    def seal(self) -> Path:
        self._refresh_manifest()
        if self.manifest.get("complete") is not True:
            observed = {
                int(layer["layer"])
                for segment in self.manifest["segments"]
                for layer in segment["layers"]
            }
            raise ValueError(
                f"output-factor archive is missing layers: "
                f"{sorted(set(self.expected_layers) - observed)}"
            )
        return self.manifest_path


__all__ = [
    "KimiOutputFactorArchive",
    "OutputFactorLayerRecord",
    "OutputFactorSegmentWriter",
    "OutputFactorSums",
    "document_factor_split",
]

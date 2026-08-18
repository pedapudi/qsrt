"""Coupled-basis W1/W3 curvature and objective-gradient storage.

Each routed expert stores block-diagonal output-Fisher factors for the
interleaved gate/up matrix after the coupled preactivation transform. Optional
bilateral sketches represent an anchor-relative full-model KL gradient in the
same coordinates. The input-side factor remains the transformed H13 used by
the encoder.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file


KIND = "Kimi-K3 coupled W1/W3 reverse-factor archive"
SEGMENT_KIND = "Kimi-K3 coupled W1/W3 reverse-factor segment"
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
        raise TypeError("upstream-factor provenance must be a JSON object")
    return result


def _segment_name(first_layer: int, end_layer: int) -> str:
    return f"segment-{first_layer:03d}-{end_layer:03d}"


def _layer_filename(layer: int) -> str:
    return f"layer-{layer:03d}.safetensors"


@dataclass(frozen=True)
class UpstreamFactorSums:
    """Reverse quantities for one routed MoE layer before serialization."""

    output_hessian_blocks: torch.Tensor
    output_hessian_rows: torch.Tensor
    intermediate_draws: torch.Tensor
    output_hessian_normalized: bool = False
    gradient_left: torch.Tensor | None = None
    gradient_right: torch.Tensor | None = None
    gradient_rows: torch.Tensor | None = None
    gradient_output_projection: torch.Tensor | None = None
    objective_normalizer: float | None = None

    def validate(
        self,
        *,
        num_experts: int,
        hidden_dimension: int,
        intermediate_dimension: int,
        block_size: int,
        gradient_rank: int,
    ) -> None:
        blocks = 2 * intermediate_dimension // block_size
        expected_factor = (num_experts, blocks, block_size, block_size)
        if self.output_hessian_blocks.device.type != "cpu":
            raise TypeError("output-Hessian blocks must reside on CPU")
        if self.output_hessian_blocks.dtype != torch.float32:
            raise TypeError("output-Hessian blocks must be FP32")
        if tuple(self.output_hessian_blocks.shape) != expected_factor:
            raise ValueError("output-Hessian blocks have incompatible geometry")
        if self.output_hessian_rows.device.type != "cpu" or (
            self.output_hessian_rows.dtype != torch.int64
        ):
            raise TypeError("output-Hessian support must be CPU int64")
        if tuple(self.output_hessian_rows.shape) != (num_experts,):
            raise ValueError("output-Hessian support has incompatible geometry")
        if bool(torch.any(self.output_hessian_rows < 0)):
            raise ValueError("output-Hessian support cannot be negative")
        if self.intermediate_draws.device.type != "cpu" or (
            self.intermediate_draws.dtype != torch.uint8
        ):
            raise TypeError("intermediate draws must be CPU uint8")
        if tuple(self.intermediate_draws.shape) != (num_experts,) or bool(
            torch.any(self.intermediate_draws >= 8)
        ):
            raise ValueError("intermediate draws have incompatible values")
        if not bool(torch.all(torch.isfinite(self.output_hessian_blocks))):
            raise FloatingPointError("output-Hessian blocks contain non-finite values")
        if not isinstance(self.output_hessian_normalized, bool):
            raise TypeError("output-Hessian normalization state must be Boolean")

        gradients = (
            self.gradient_left,
            self.gradient_right,
            self.gradient_rows,
            self.gradient_output_projection,
        )
        if all(value is None for value in gradients):
            if self.objective_normalizer is not None:
                raise ValueError("gradient normalization is present without a gradient")
            return
        if any(value is None for value in gradients):
            raise ValueError("objective-gradient sketch is incomplete")
        assert self.gradient_left is not None
        assert self.gradient_right is not None
        assert self.gradient_rows is not None
        assert self.gradient_output_projection is not None
        expected = {
            "gradient_left": (num_experts, 2 * intermediate_dimension, gradient_rank),
            "gradient_right": (num_experts, gradient_rank, hidden_dimension),
            "gradient_rows": (num_experts,),
            "gradient_output_projection": (2 * intermediate_dimension, gradient_rank),
        }
        for name, value in (
            ("gradient_left", self.gradient_left),
            ("gradient_right", self.gradient_right),
            ("gradient_rows", self.gradient_rows),
            ("gradient_output_projection", self.gradient_output_projection),
        ):
            if value.device.type != "cpu" or tuple(value.shape) != expected[name]:
                raise ValueError(f"{name} has incompatible storage or geometry")
        if self.gradient_left.dtype != torch.float32 or (
            self.gradient_right.dtype != torch.float32
        ) or self.gradient_output_projection.dtype != torch.float32:
            raise TypeError("objective-gradient sketch tensors must be FP32")
        if self.gradient_rows.dtype != torch.int64 or bool(
            torch.any(self.gradient_rows < 0)
        ):
            raise TypeError("objective-gradient support must be nonnegative int64")
        if self.objective_normalizer is None or not (
            float(self.objective_normalizer) > 0.0
        ):
            raise ValueError("objective-gradient normalization must be positive")


@dataclass(frozen=True)
class KimiUpstreamGradientLayer:
    """One layer's normalized low-rank KL-gradient sketch."""

    left: torch.Tensor
    right: torch.Tensor
    rows: torch.Tensor
    intermediate_draws: torch.Tensor
    output_projection: torch.Tensor
    intermediate_dimension: int

    def expert_factors(
        self,
        expert: int,
        *,
        matrix: str,
        core_rcond: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return factors whose product is one source-coordinate gradient."""

        if matrix not in ("w1", "w3"):
            raise ValueError("upstream gradient matrix must be w1 or w3")
        full_left, right = self.expert_joint_factors(
            expert,
            core_rcond=core_rcond,
        )
        offset = 0 if matrix == "w1" else self.intermediate_dimension
        return (
            full_left[offset : offset + self.intermediate_dimension].contiguous(),
            right,
        )

    def expert_joint_factors(
        self,
        expert: int,
        *,
        core_rcond: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one stabilized joint gate/up gradient factorization."""

        if not 0 <= expert < self.left.shape[0]:
            raise IndexError("routed expert index is out of range")
        if core_rcond is not None and not 0.0 < float(core_rcond) <= 1.0:
            raise ValueError("gradient core rcond must lie in (0, 1]")
        full_left = self.left[expert]
        core = self.output_projection.T @ full_left
        core_inverse = (
            torch.linalg.pinv(core)
            if core_rcond is None
            else torch.linalg.pinv(core, rtol=float(core_rcond))
        )
        return (
            (full_left @ core_inverse).contiguous(),
            self.right[expert].contiguous(),
        )


@dataclass(frozen=True)
class UpstreamFactorLayerRecord:
    """Stored reverse-factor tensors and support for one decoder layer."""

    layer: int
    file: str
    sha256: str
    bytes: int
    supported_experts: int
    gradient: bool

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "UpstreamFactorLayerRecord":
        return cls(
            layer=int(value["layer"]),
            file=str(value["file"]),
            sha256=str(value["sha256"]),
            bytes=int(value["bytes"]),
            supported_experts=int(value["supported_experts"]),
            gradient=bool(value["gradient"]),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "file": self.file,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "supported_experts": self.supported_experts,
            "gradient": self.gradient,
        }


class UpstreamFactorSegmentWriter:
    """Build one durable reverse-segment result before cotangent commit."""

    def __init__(
        self,
        archive: "KimiUpstreamFactorArchive",
        *,
        first_layer: int,
        end_layer: int,
    ):
        if not 0 <= first_layer < end_layer <= archive.num_layers:
            raise ValueError("upstream-factor segment range is invalid")
        self.archive = archive
        self.first_layer = int(first_layer)
        self.end_layer = int(end_layer)
        self.operation = _segment_name(first_layer, end_layer)
        self.name = self.operation
        self.path = archive.root / PENDING_DIRECTORY / self.name
        committed = archive.root / SEGMENT_DIRECTORY / self.name
        if committed.exists() or self.path.exists():
            raise FileExistsError(f"upstream-factor segment already exists: {self.name}")
        self.path.mkdir(parents=True)
        self.records: list[UpstreamFactorLayerRecord] = []
        self._pending_layers: set[int] = set()
        self._lock = threading.Lock()
        self._sealed = False
        self._committed = False

    def add(self, layer: int, sums: UpstreamFactorSums) -> UpstreamFactorLayerRecord:
        with self._lock:
            if self._sealed:
                raise RuntimeError("upstream-factor segment is sealed")
            if layer not in self.archive.expected_layers or not (
                self.first_layer <= layer < self.end_layer
            ):
                raise ValueError("upstream-factor layer lies outside the segment")
            if layer in self._pending_layers or any(
                record.layer == layer for record in self.records
            ):
                raise ValueError(f"upstream-factor layer {layer} was added twice")
            self._pending_layers.add(layer)
        try:
            sums.validate(
                num_experts=self.archive.num_experts,
                hidden_dimension=self.archive.hidden_dimension,
                intermediate_dimension=self.archive.intermediate_dimension,
                block_size=self.archive.block_size,
                gradient_rank=self.archive.gradient_rank,
            )
            rows = sums.output_hessian_rows
            if sums.output_hessian_normalized:
                factors = sums.output_hessian_blocks.contiguous()
            else:
                denominators = rows.clamp_min(1).to(torch.float32).reshape(
                    -1, 1, 1, 1
                )
                factors = sums.output_hessian_blocks.div(denominators).contiguous()
            tensors = {
                "intermediate_draws": sums.intermediate_draws.contiguous(),
                "output_hessian_blocks": factors,
                "output_hessian_rows": rows.contiguous(),
            }
            gradient = sums.gradient_left is not None
            if gradient:
                assert sums.gradient_left is not None
                assert sums.gradient_right is not None
                assert sums.gradient_rows is not None
                assert sums.gradient_output_projection is not None
                assert sums.objective_normalizer is not None
                scale = 1.0 / float(sums.objective_normalizer)
                tensors.update(
                    {
                        "gradient_left": sums.gradient_left.mul(scale).contiguous(),
                        "gradient_right": sums.gradient_right.mul(scale).contiguous(),
                        "gradient_rows": sums.gradient_rows.contiguous(),
                        "gradient_output_projection": (
                            sums.gradient_output_projection.contiguous()
                        ),
                    }
                )
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
                        "interleaved gate/up preactivation in stored coupled coordinates"
                    ),
                    "gradient": str(gradient).lower(),
                    "objective_normalizer": (
                        "none"
                        if sums.objective_normalizer is None
                        else repr(float(sums.objective_normalizer))
                    ),
                    "damping": "none",
                },
            )
            os.replace(temporary, path)
            record = UpstreamFactorLayerRecord(
                layer=layer,
                file=path.name,
                sha256=_sha256(path),
                bytes=path.stat().st_size,
                supported_experts=int(torch.count_nonzero(rows).item()),
                gradient=gradient,
            )
        except BaseException:
            with self._lock:
                self._pending_layers.discard(layer)
            raise
        with self._lock:
            self._pending_layers.remove(layer)
            self.records.append(record)
        return record

    def seal(self) -> Path:
        with self._lock:
            if self._pending_layers:
                raise RuntimeError("upstream-factor segment has unfinished layer writes")
            expected = {
                layer
                for layer in self.archive.expected_layers
                if self.first_layer <= layer < self.end_layer
            }
            actual = {record.layer for record in self.records}
            if actual != expected:
                raise ValueError(
                    "upstream-factor segment layers differ: "
                    f"missing={sorted(expected - actual)}, "
                    f"unexpected={sorted(actual - expected)}"
                )
            records = sorted(self.records, key=lambda item: item.layer)
            self._sealed = True
        manifest = {
            "kind": SEGMENT_KIND,
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "operation": self.operation,
            "first_layer": self.first_layer,
            "end_layer": self.end_layer,
            "layers": [record.to_json() for record in records],
        }
        path = self.path / MANIFEST_FILENAME
        _atomic_json(path, manifest)
        return path

    def commit(self) -> Path:
        if not self._sealed or self._committed:
            raise RuntimeError("upstream-factor segment is not commit-ready")
        destination = self.archive.root / SEGMENT_DIRECTORY / self.name
        os.replace(self.path, destination)
        self.archive._refresh_manifest()
        self._committed = True
        return destination


class KimiUpstreamFactorArchive:
    """Segment-atomic inventory of coupled W1/W3 reverse factors."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.manifest_path = self.root / MANIFEST_FILENAME
        document = json.loads(self.manifest_path.read_text())
        if document.get("kind") != KIND or document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{self.manifest_path}: incompatible upstream-factor archive")
        self.manifest = document
        self.num_layers = int(document["num_layers"])
        self.num_experts = int(document["num_experts"])
        self.hidden_dimension = int(document["hidden_dimension"])
        self.intermediate_dimension = int(document["intermediate_dimension"])
        self.block_size = int(document["block_size"])
        self.gradient_rank = int(document["gradient_rank"])
        self.expected_layers = tuple(int(value) for value in document["expected_layers"])
        self._verified_paths: set[Path] = set()

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        num_layers: int,
        num_experts: int,
        hidden_dimension: int,
        intermediate_dimension: int,
        block_size: int,
        gradient_rank: int,
        expected_layers: Sequence[int],
        provenance: Mapping[str, object] | None = None,
    ) -> "KimiUpstreamFactorArchive":
        layers = tuple(sorted(set(int(value) for value in expected_layers)))
        if min(
            num_layers,
            num_experts,
            hidden_dimension,
            intermediate_dimension,
            block_size,
            gradient_rank,
        ) <= 0 or not layers:
            raise ValueError("upstream-factor archive geometry must be positive")
        if 2 * intermediate_dimension % block_size:
            raise ValueError("block size must divide the interleaved gate/up dimension")
        if layers[0] < 0 or layers[-1] >= num_layers:
            raise ValueError("expected upstream-factor layer lies outside the model")
        destination = Path(root).expanduser().resolve()
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"upstream-factor destination is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / SEGMENT_DIRECTORY).mkdir()
        (destination / PENDING_DIRECTORY).mkdir()
        _atomic_json(
            destination / MANIFEST_FILENAME,
            {
                "kind": KIND,
                "schema_version": SCHEMA_VERSION,
                "complete": False,
                "num_layers": int(num_layers),
                "num_experts": int(num_experts),
                "hidden_dimension": int(hidden_dimension),
                "intermediate_dimension": int(intermediate_dimension),
                "block_size": int(block_size),
                "gradient_rank": int(gradient_rank),
                "expected_layers": list(layers),
                "semantic_point": (
                    "interleaved gate/up preactivation in stored coupled coordinates"
                ),
                "factor_dtype": "float32",
                "damping": "none",
                "segments": [],
                "provenance": _json_object(provenance),
            },
        )
        return cls(destination)

    def begin_segment(self, first_layer: int, end_layer: int) -> UpstreamFactorSegmentWriter:
        return UpstreamFactorSegmentWriter(
            self,
            first_layer=first_layer,
            end_layer=end_layer,
        )

    def layer_path(self, layer: int, *, verify_hash: bool = True) -> Path:
        requested = int(layer)
        if requested not in self.expected_layers:
            raise ValueError(f"decoder layer {requested} has no upstream factors")
        for segment in self.manifest.get("segments", []):
            directory = self.root / str(segment["directory"])
            for value in segment.get("layers", []):
                record = UpstreamFactorLayerRecord.from_json(value)
                if record.layer != requested:
                    continue
                path = directory / record.file
                if not path.is_file() or path.stat().st_size != record.bytes:
                    raise ValueError(f"upstream-factor tensor is missing or truncated: {path}")
                if verify_hash and path not in self._verified_paths:
                    if _sha256(path) != record.sha256:
                        raise ValueError(f"upstream-factor tensor hash mismatch: {path}")
                    self._verified_paths.add(path)
                return path
        raise FileNotFoundError(f"upstream factors have not committed layer {requested}")

    def load_expert_output_blocks(
        self,
        layer: int,
        expert: int,
        *,
        matrix: str,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, int, int]:
        """Load one stored W1 or W3 block factor without reading other experts."""

        if matrix not in ("w1", "w3"):
            raise ValueError("upstream factor matrix must be w1 or w3")
        if not 0 <= expert < self.num_experts:
            raise IndexError("routed expert index is out of range")
        blocks_per_matrix = self.intermediate_dimension // self.block_size
        begin = 0 if matrix == "w1" else blocks_per_matrix
        end = begin + blocks_per_matrix
        with safe_open(self.layer_path(layer), framework="pt", device="cpu") as reader:
            blocks = reader.get_slice("output_hessian_blocks")[expert, begin:end]
            rows = int(reader.get_tensor("output_hessian_rows")[expert])
            draw = int(reader.get_tensor("intermediate_draws")[expert])
        return blocks.to(device=device), rows, draw

    def load_layer_output_blocks(
        self,
        layer: int,
        *,
        device: torch.device | str = "cpu",
        verify_hash: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load every expert factor and draw from one layer-file open."""

        target = torch.device(device)
        with safe_open(
            self.layer_path(layer, verify_hash=verify_hash),
            framework="pt",
            device=str(target),
        ) as reader:
            blocks = reader.get_tensor("output_hessian_blocks")
            rows = reader.get_tensor("output_hessian_rows")
            draws = reader.get_tensor("intermediate_draws")
        expected_blocks = (
            self.num_experts,
            2 * self.intermediate_dimension // self.block_size,
            self.block_size,
            self.block_size,
        )
        if blocks.dtype != torch.float32 or tuple(blocks.shape) != expected_blocks:
            raise ValueError("stored layer output factors have incompatible geometry")
        if rows.dtype != torch.int64 or tuple(rows.shape) != (self.num_experts,):
            raise ValueError("stored layer factor support has incompatible geometry")
        if draws.dtype != torch.uint8 or tuple(draws.shape) != (self.num_experts,):
            raise ValueError("stored layer factor draws have incompatible geometry")
        return blocks, rows, draws

    def load_expert_gradient(
        self,
        layer: int,
        expert: int,
        *,
        matrix: str,
        device: torch.device | str = "cpu",
        core_rcond: float | None = None,
    ) -> torch.Tensor:
        """Reconstruct one low-rank anchor-gradient matrix for Viterbi costs."""

        sketch = self.load_layer_gradient(layer, device=device)
        left, right = sketch.expert_factors(
            expert,
            matrix=matrix,
            core_rcond=core_rcond,
        )
        return (left @ right).contiguous()

    def load_layer_gradient(
        self,
        layer: int,
        *,
        device: torch.device | str = "cpu",
        verify_hash: bool = True,
    ) -> KimiUpstreamGradientLayer:
        """Load one layer's gradient sketch without its output-Fisher blocks."""

        path = self.layer_path(layer)
        if verify_hash:
            path = self.layer_path(layer, verify_hash=True)
        target = torch.device(device)
        with safe_open(path, framework="pt", device=str(target)) as reader:
            required = {
                "gradient_left",
                "gradient_right",
                "gradient_rows",
                "gradient_output_projection",
                "intermediate_draws",
            }
            if not required.issubset(reader.keys()):
                raise ValueError(f"upstream-factor layer has no objective gradient: {path}")
            left = reader.get_tensor("gradient_left")
            right = reader.get_tensor("gradient_right")
            rows = reader.get_tensor("gradient_rows")
            draws = reader.get_tensor("intermediate_draws")
            projection = reader.get_tensor("gradient_output_projection")
        expected = {
            "left": (
                self.num_experts,
                2 * self.intermediate_dimension,
                self.gradient_rank,
            ),
            "right": (
                self.num_experts,
                self.gradient_rank,
                self.hidden_dimension,
            ),
            "rows": (self.num_experts,),
            "draws": (self.num_experts,),
            "projection": (
                2 * self.intermediate_dimension,
                self.gradient_rank,
            ),
        }
        for name, value in (
            ("left", left),
            ("right", right),
            ("rows", rows),
            ("draws", draws),
            ("projection", projection),
        ):
            if tuple(value.shape) != expected[name]:
                raise ValueError(f"stored {name} gradient tensor has wrong geometry")
        if left.dtype != torch.float32 or right.dtype != torch.float32 or (
            projection.dtype != torch.float32
        ):
            raise TypeError("stored objective-gradient factors must be FP32")
        if rows.dtype != torch.int64:
            raise TypeError("stored objective-gradient support must be int64")
        if draws.dtype != torch.uint8:
            raise TypeError("stored intermediate draws must be uint8")
        return KimiUpstreamGradientLayer(
            left=left,
            right=right,
            rows=rows,
            intermediate_draws=draws,
            output_projection=projection,
            intermediate_dimension=self.intermediate_dimension,
        )

    def _segment_records(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        observed: set[int] = set()
        for directory in sorted((self.root / SEGMENT_DIRECTORY).glob("segment-*-*")):
            path = directory / MANIFEST_FILENAME
            document = json.loads(path.read_text())
            if document.get("kind") != SEGMENT_KIND or document.get("complete") is not True:
                raise ValueError(f"invalid upstream-factor segment manifest: {path}")
            layers = document.get("layers")
            if not isinstance(layers, list):
                raise TypeError(f"upstream-factor segment has no layer inventory: {path}")
            for value in layers:
                record = UpstreamFactorLayerRecord.from_json(value)
                if record.layer in observed:
                    raise ValueError(f"upstream-factor layer {record.layer} appears twice")
                observed.add(record.layer)
                tensor_path = directory / record.file
                if not tensor_path.is_file() or tensor_path.stat().st_size != record.bytes:
                    raise ValueError(f"upstream-factor tensor is missing: {tensor_path}")
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
        observed = {
            int(value["layer"])
            for segment in segments
            for value in segment["layers"]
        }
        self.manifest["segments"] = segments
        self.manifest["complete"] = observed == set(self.expected_layers)
        _atomic_json(self.manifest_path, self.manifest)

    def recover_pending(self, completed_operations: Sequence[str]) -> tuple[Path, ...]:
        completed = set(str(value) for value in completed_operations)
        promoted: list[Path] = []
        for directory in sorted((self.root / PENDING_DIRECTORY).glob("segment-*-*")):
            path = directory / MANIFEST_FILENAME
            if not path.is_file():
                continue
            document = json.loads(path.read_text())
            if str(document.get("operation")) not in completed:
                continue
            destination = self.root / SEGMENT_DIRECTORY / directory.name
            os.replace(directory, destination)
            promoted.append(destination)
        if promoted:
            self._refresh_manifest()
        return tuple(promoted)

    def discard_uncommitted_pending(
        self,
        completed_operations: Sequence[str],
    ) -> tuple[Path, ...]:
        completed = set(str(value) for value in completed_operations)
        discarded: list[Path] = []
        for directory in sorted((self.root / PENDING_DIRECTORY).glob("segment-*-*")):
            path = directory / MANIFEST_FILENAME
            operation = None
            if path.is_file():
                try:
                    operation = str(json.loads(path.read_text()).get("operation"))
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
                int(value["layer"])
                for segment in self.manifest["segments"]
                for value in segment["layers"]
            }
            raise ValueError(
                f"upstream-factor archive is missing layers: "
                f"{sorted(set(self.expected_layers) - observed)}"
            )
        return self.manifest_path


__all__ = [
    "KimiUpstreamFactorArchive",
    "KimiUpstreamGradientLayer",
    "UpstreamFactorLayerRecord",
    "UpstreamFactorSegmentWriter",
    "UpstreamFactorSums",
]

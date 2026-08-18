"""Canonical all-row calibration capture container.

One capture stores the input, routed expert choices, applied route weights, and
the post-allreduce routed output for every observed token at every routed-MoE
layer.  Rank zero owns the tensors.  Other tensor-parallel ranks contribute
receipts but do not duplicate row data.

Layer chunks are independently checksummed safetensors files.  Their row ranges
must be contiguous and identical across layers.  A capture is readable for
candidate scoring only after the root manifest is finalized.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load, load_file, save_file

from qsrt import constants as C


ALL_ROW_CAPTURE_KIND = "qsrt_all_routed_rows"
ALL_ROW_CAPTURE_SCHEMA_VERSION = 1
DEFAULT_CHUNK_ROWS = 16_384
REQUIRED_TENSORS = (
    "input",
    "expert_indices",
    "route_weights",
    "routed_output",
    "request_index",
    "document_id",
    "token_offset",
    "role",
)


@dataclass(frozen=True)
class AllRowCaptureGeometry:
    """Logical tensor geometry for one routed-MoE capture."""

    layers: tuple[int, ...] = tuple(C.MOE_LAYERS)
    input_size: int = C.LATENT
    top_k: int = C.TOP_K

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, object]) -> "AllRowCaptureGeometry":
        value = manifest.get("geometry")
        if not isinstance(value, Mapping):
            raise ValueError("all-row capture manifest lacks geometry")
        raw_layers = value.get("layers")
        if not isinstance(raw_layers, list):
            raise ValueError("all-row capture geometry lacks layer identities")
        geometry = cls(
            layers=tuple(int(layer) for layer in raw_layers),
            input_size=int(value.get("input_size", 0)),
            top_k=int(value.get("top_k", 0)),
        )
        geometry.validate()
        return geometry

    def validate(self) -> None:
        if not self.layers or tuple(sorted(set(self.layers))) != self.layers:
            raise ValueError("capture layers must be sorted and unique")
        if min(self.layers) < 0 or self.input_size <= 0 or self.top_k <= 0:
            raise ValueError("capture geometry contains a nonpositive dimension")

    def manifest(self) -> dict[str, object]:
        self.validate()
        return {
            "layers": list(self.layers),
            "input_size": self.input_size,
            "top_k": self.top_k,
        }


@dataclass(frozen=True)
class AllRowChunk:
    """One verified layer chunk."""

    layer: int
    index: int
    row_begin: int
    row_end: int
    path: Path
    receipt_path: Path
    sha256: str

    @property
    def rows(self) -> int:
        return self.row_end - self.row_begin

    def load(
        self,
        *,
        verify_checksum: bool = False,
        fields: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        requested = REQUIRED_TENSORS if fields is None else tuple(dict.fromkeys(fields))
        unknown = set(requested) - set(REQUIRED_TENSORS)
        if not requested or unknown:
            raise ValueError(f"invalid capture chunk fields: {sorted(unknown)}")
        if verify_checksum:
            payload = self.path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != self.sha256:
                raise ValueError(f"capture chunk checksum differs: {self.path}")
            tensors = load(payload)
        else:
            with safe_open(self.path, framework="pt", device="cpu") as handle:
                tensors = {name: handle.get_tensor(name) for name in requested}
        if fields is None:
            _validate_chunk_tensors(tensors, rows=self.rows, geometry=None)
        else:
            for name, tensor in tensors.items():
                if tensor.shape[0] != self.rows or not tensor.is_contiguous():
                    raise ValueError(f"capture tensor {name!r} has invalid row geometry")
        return {name: tensors[name] for name in requested}


@dataclass(frozen=True)
class ExpertRowIndex:
    """Positions and top-k slots for one expert in a materialized layer."""

    row_indices: torch.Tensor
    route_slots: torch.Tensor

    def __post_init__(self) -> None:
        if self.row_indices.dtype != torch.int64 or self.route_slots.dtype != torch.uint8:
            raise TypeError("expert row indices require int64 rows and uint8 route slots")
        if self.row_indices.ndim != 1 or self.route_slots.shape != self.row_indices.shape:
            raise ValueError("expert row positions and route slots must be aligned vectors")
        if not self.row_indices.is_contiguous() or not self.route_slots.is_contiguous():
            raise ValueError("expert row indices must be contiguous")

    @property
    def rows(self) -> int:
        return int(self.row_indices.numel())


@dataclass(frozen=True)
class MappedExpertRowIndex:
    """Expert occurrences without duplicating the layer input matrix."""

    row_indices: torch.Tensor
    route_slots: torch.Tensor
    route_weights: torch.Tensor

    def __post_init__(self) -> None:
        if (
            self.row_indices.dtype != torch.int64
            or self.route_slots.dtype != torch.uint8
            or self.route_weights.dtype != torch.float32
        ):
            raise TypeError(
                "mapped expert indices require int64 rows, uint8 slots, and "
                "FP32 route weights"
            )
        if not (
            self.row_indices.ndim == 1
            and self.route_slots.shape == self.row_indices.shape
            and self.route_weights.shape == self.row_indices.shape
        ):
            raise ValueError("mapped expert occurrence tensors must be aligned vectors")

    @property
    def rows(self) -> int:
        return int(self.row_indices.numel())


@dataclass(frozen=True)
class MappedLayerRows:
    """Memory-mapped layer inputs with a compact routed-occurrence index.

    Safetensors input tensors remain backed by the capture files.  This avoids
    one 28-GiB private input copy per scoring worker while retaining direct,
    batched access to every naturally routed expert occurrence.
    """

    layer: int
    population_rows: int
    input_chunks: tuple[torch.Tensor, ...]
    chunk_begins: torch.Tensor
    chunk_ends: torch.Tensor
    expert_index: tuple[MappedExpertRowIndex, ...]

    def __post_init__(self) -> None:
        if (
            self.layer < 0
            or self.population_rows <= 0
            or not self.input_chunks
            or not self.expert_index
        ):
            raise ValueError("mapped layer rows require complete layer data")
        chunks = len(self.input_chunks)
        if (
            self.chunk_begins.shape != (chunks,)
            or self.chunk_ends.shape != (chunks,)
            or self.chunk_begins.dtype != torch.int64
            or self.chunk_ends.dtype != torch.int64
            or int(self.chunk_begins[0]) != 0
            or int(self.chunk_ends[-1]) != self.population_rows
            or not torch.equal(self.chunk_begins[1:], self.chunk_ends[:-1])
        ):
            raise ValueError("mapped layer chunk ranges are not contiguous")
        if any(
            tensor.ndim != 2
            or tensor.dtype != torch.bfloat16
            or tensor.shape[0] != int(self.chunk_ends[index] - self.chunk_begins[index])
            for index, tensor in enumerate(self.input_chunks)
        ):
            raise ValueError("mapped layer input chunks have invalid geometry")
        if sum(index.rows for index in self.expert_index) <= 0:
            raise ValueError("mapped layer occurrence index is empty")

    @property
    def rows(self) -> int:
        return self.population_rows

    def _gather_inputs(self, rows: torch.Tensor) -> torch.Tensor:
        if rows.dtype != torch.int64 or rows.ndim != 1:
            raise TypeError("mapped input rows must be an int64 vector")
        if rows.numel() == 0:
            return torch.empty(
                (0, self.input_chunks[0].shape[1]), dtype=torch.bfloat16
            )
        chunk_ids = torch.bucketize(rows, self.chunk_ends, right=True)
        parts: list[torch.Tensor] = []
        begin = 0
        while begin < rows.numel():
            chunk = int(chunk_ids[begin])
            end = begin + 1
            while end < rows.numel() and int(chunk_ids[end]) == chunk:
                end += 1
            local = rows[begin:end] - self.chunk_begins[chunk]
            parts.append(self.input_chunks[chunk].index_select(0, local))
            begin = end
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    def expert_batches(
        self,
        expert: int,
        *,
        batch_rows: int,
        row_limit: int | None = None,
        fields: Sequence[str] = ("input",),
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield one expert's occurrences from the mapped layer inputs."""

        if tuple(fields) != ("input",):
            raise ValueError("mapped layer scoring exposes only the input field")
        if not 0 <= expert < len(self.expert_index):
            raise ValueError("expert is outside the captured routing geometry")
        if batch_rows <= 0:
            raise ValueError("batch_rows must be positive")
        if row_limit is not None and not 0 <= row_limit <= self.population_rows:
            raise ValueError("row_limit is outside the mapped layer")
        positions = self.expert_index[expert]
        count = positions.rows
        if row_limit is not None:
            count = int(torch.searchsorted(positions.row_indices, row_limit).item())
        for begin in range(0, count, batch_rows):
            end = min(begin + batch_rows, count)
            rows = positions.row_indices[begin:end]
            yield {
                "input": self._gather_inputs(rows),
                "row_index": rows,
                "route_slot": positions.route_slots[begin:end],
                "route_weight": positions.route_weights[begin:end],
            }


@dataclass(frozen=True)
class MaterializedLayerRows:
    """One layer's canonical rows and an occurrence index for routed experts."""

    layer: int
    tensors: Mapping[str, torch.Tensor]
    expert_index: tuple[ExpertRowIndex, ...]

    def __post_init__(self) -> None:
        if self.layer < 0 or not self.tensors or not self.expert_index:
            raise ValueError("materialized layer rows require data and expert identities")
        rows = {int(value.shape[0]) for value in self.tensors.values()}
        if len(rows) != 1:
            raise ValueError("materialized layer tensors do not share one row count")
        total_rows = rows.pop()
        top_k = self.tensors["expert_indices"].shape[1]
        if sum(index.rows for index in self.expert_index) != total_rows * top_k:
            raise ValueError("expert occurrence index does not cover every routed slot")

    @property
    def rows(self) -> int:
        return int(self.tensors["input"].shape[0])

    def expert_batches(
        self,
        expert: int,
        *,
        batch_rows: int,
        row_limit: int | None = None,
        fields: Sequence[str] = ("input",),
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield all naturally routed occurrences for one expert.

        ``route_weight`` is extracted from the expert's actual top-k slot.  A
        row limit selects a prefix of the calibration population, which is
        useful for measuring whether candidate choices stabilize with more
        captured rows.
        """

        if not 0 <= expert < len(self.expert_index):
            raise ValueError("expert is outside the captured routing geometry")
        if batch_rows <= 0:
            raise ValueError("batch_rows must be positive")
        if row_limit is not None and not 0 <= row_limit <= self.rows:
            raise ValueError("row_limit is outside the materialized layer")
        unknown = set(fields) - set(self.tensors)
        if unknown:
            raise ValueError(f"unknown materialized fields: {sorted(unknown)}")
        positions = self.expert_index[expert]
        if row_limit is None:
            selected_rows = positions.row_indices
            selected_slots = positions.route_slots
        else:
            keep = positions.row_indices < row_limit
            selected_rows = positions.row_indices[keep].contiguous()
            selected_slots = positions.route_slots[keep].contiguous()
        for begin in range(0, selected_rows.numel(), batch_rows):
            end = min(begin + batch_rows, selected_rows.numel())
            rows = selected_rows[begin:end]
            slots = selected_slots[begin:end].long()
            batch = {
                name: self.tensors[name].index_select(0, rows)
                for name in fields
            }
            batch["row_index"] = rows
            batch["route_slot"] = selected_slots[begin:end]
            batch["route_weight"] = self.tensors["route_weights"][rows, slots]
            yield batch


@dataclass(frozen=True)
class MaterializedExpertRows:
    """Captured occurrences for one expert with global population row identities."""

    layer: int
    expert: int
    population_rows: int
    tensors: Mapping[str, torch.Tensor]
    row_indices: torch.Tensor
    route_slots: torch.Tensor
    route_weights: torch.Tensor

    @property
    def rows(self) -> int:
        return self.population_rows

    def expert_batches(
        self,
        expert: int,
        *,
        batch_rows: int,
        row_limit: int | None = None,
        fields: Sequence[str] = ("input",),
    ) -> Iterator[dict[str, torch.Tensor]]:
        if expert != self.expert:
            raise ValueError("materialized expert rows belong to another expert")
        if batch_rows <= 0:
            raise ValueError("batch_rows must be positive")
        if row_limit is not None and not 0 <= row_limit <= self.population_rows:
            raise ValueError("row_limit is outside the captured population")
        unknown = set(fields) - set(self.tensors)
        if unknown:
            raise ValueError(f"unknown materialized fields: {sorted(unknown)}")
        count = self.row_indices.numel()
        if row_limit is not None:
            count = int(torch.searchsorted(self.row_indices, row_limit).item())
        for begin in range(0, count, batch_rows):
            end = min(begin + batch_rows, count)
            batch = {name: self.tensors[name][begin:end] for name in fields}
            batch["row_index"] = self.row_indices[begin:end]
            batch["route_slot"] = self.route_slots[begin:end]
            batch["route_weight"] = self.route_weights[begin:end]
            yield batch


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"capture is missing {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def initialize_all_row_capture(
    root: str | Path,
    *,
    run_id: str,
    model: str,
    revision: str,
    resident_checkpoint: str,
    corpus_manifest_sha256: str,
    geometry: AllRowCaptureGeometry = AllRowCaptureGeometry(),
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    expected_rows: int | None = None,
    tp_world_size: int,
) -> Path:
    """Create or verify an incomplete all-row capture root."""

    if not run_id or not model or not revision or not resident_checkpoint:
        raise ValueError("capture identity fields must be nonempty")
    if len(corpus_manifest_sha256) != 64:
        raise ValueError("corpus manifest identity must be a SHA-256 digest")
    if chunk_rows <= 0 or tp_world_size <= 0:
        raise ValueError("chunk rows and TP world size must be positive")
    if expected_rows is not None and expected_rows <= 0:
        raise ValueError("expected capture rows must be positive")
    geometry.validate()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "kind": ALL_ROW_CAPTURE_KIND,
        "schema_version": ALL_ROW_CAPTURE_SCHEMA_VERSION,
        "complete": False,
        "run_id": run_id,
        "model": model,
        "revision": revision,
        "resident_checkpoint": resident_checkpoint,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "geometry": geometry.manifest(),
        "chunk_rows": int(chunk_rows),
        "expected_rows": expected_rows,
        "tp_world_size": int(tp_world_size),
        "canonical_tensor_rank": 0,
        "route_weight_convention": "applied_gate; squared_once_in_sse",
        "row_identity": "request_index,document_id,token_offset,role",
    }
    path = root / "manifest.json"
    if path.exists():
        existing = _read_json(path)
        if bool(existing.get("complete", False)):
            raise ValueError(f"capture {root} is already complete")
        if existing != manifest:
            raise ValueError(f"capture {root} does not match the requested identity")
    else:
        _atomic_json(path, manifest)
    for layer in geometry.layers:
        (root / f"layer-{layer:05d}").mkdir(exist_ok=True)
    return root


def _validate_chunk_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    rows: int,
    geometry: AllRowCaptureGeometry | None,
) -> None:
    if set(tensors) != set(REQUIRED_TENSORS):
        missing = sorted(set(REQUIRED_TENSORS) - set(tensors))
        extra = sorted(set(tensors) - set(REQUIRED_TENSORS))
        raise ValueError(f"capture chunk tensor keys differ: missing={missing}, extra={extra}")
    if rows <= 0:
        raise ValueError("capture chunks must contain rows")
    input_size = geometry.input_size if geometry else tensors["input"].shape[1]
    top_k = geometry.top_k if geometry else tensors["expert_indices"].shape[1]
    expected = {
        "input": ((rows, input_size), torch.bfloat16),
        "expert_indices": ((rows, top_k), torch.int32),
        "route_weights": ((rows, top_k), torch.float32),
        "routed_output": ((rows, input_size), torch.bfloat16),
        "request_index": ((rows,), torch.int64),
        "document_id": ((rows,), torch.int64),
        "token_offset": ((rows,), torch.int32),
        "role": ((rows,), torch.uint8),
    }
    for key, (shape, dtype) in expected.items():
        tensor = tensors[key]
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"capture tensor {key!r} has {tuple(tensor.shape)} {tensor.dtype}; "
                f"expected {shape} {dtype}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"capture tensor {key!r} must be contiguous")
    if not torch.all(torch.isfinite(tensors["route_weights"])):
        raise ValueError("route weights must be finite")
    if torch.any(tensors["expert_indices"] < 0):
        raise ValueError("expert indices must be nonnegative")
    if torch.any(tensors["token_offset"] < 0):
        raise ValueError("token offsets must be nonnegative")
    if torch.any(tensors["role"] > 1):
        raise ValueError("row roles must be 0 for prompt or 1 for decode")


def write_all_row_chunk(
    root: str | Path,
    *,
    layer: int,
    index: int,
    row_begin: int,
    tensors: Mapping[str, torch.Tensor],
) -> AllRowChunk:
    """Atomically persist one canonical rank-zero layer chunk."""

    root = Path(root)
    manifest = _read_json(root / "manifest.json")
    if bool(manifest.get("complete", False)):
        raise ValueError("cannot append to a finalized capture")
    geometry = AllRowCaptureGeometry.from_manifest(manifest)
    if layer not in geometry.layers or index < 0 or row_begin < 0:
        raise ValueError("capture chunk identity is outside the manifest geometry")
    rows = int(tensors["input"].shape[0]) if "input" in tensors else 0
    canonical = {key: value.detach().cpu().contiguous() for key, value in tensors.items()}
    _validate_chunk_tensors(canonical, rows=rows, geometry=geometry)
    layer_root = root / f"layer-{layer:05d}"
    layer_root.mkdir(parents=True, exist_ok=True)
    path = layer_root / f"chunk-{index:08d}.safetensors"
    receipt_path = layer_root / f"chunk-{index:08d}.json"
    if path.exists() or receipt_path.exists():
        raise ValueError(f"capture chunk {layer}:{index} already exists")
    temporary = path.with_name(f".{path.name}.tmp")
    save_file(canonical, str(temporary))
    temporary.replace(path)
    digest = _sha256(path)
    receipt = {
        "kind": "qsrt_all_routed_rows_chunk",
        "schema_version": ALL_ROW_CAPTURE_SCHEMA_VERSION,
        "layer": layer,
        "index": index,
        "row_begin": row_begin,
        "row_end": row_begin + rows,
        "rows": rows,
        "file": path.name,
        "sha256": digest,
        "request_index_first": int(canonical["request_index"][0]),
        "request_index_last": int(canonical["request_index"][-1]),
        "token_offset_first": int(canonical["token_offset"][0]),
        "token_offset_last": int(canonical["token_offset"][-1]),
    }
    _atomic_json(receipt_path, receipt)
    return AllRowChunk(
        layer=layer,
        index=index,
        row_begin=row_begin,
        row_end=row_begin + rows,
        path=path,
        receipt_path=receipt_path,
        sha256=digest,
    )


def layer_chunks(
    root: str | Path,
    layer: int,
    *,
    verify_hashes: bool = True,
) -> tuple[AllRowChunk, ...]:
    """Load and validate one layer's contiguous chunk index."""

    root = Path(root)
    manifest = _read_json(root / "manifest.json")
    geometry = AllRowCaptureGeometry.from_manifest(manifest)
    if layer not in geometry.layers:
        raise ValueError(f"layer {layer} is not in the capture geometry")
    receipts = sorted((root / f"layer-{layer:05d}").glob("chunk-*.json"))
    chunks: list[AllRowChunk] = []
    next_row = 0
    for expected_index, receipt_path in enumerate(receipts):
        receipt = _read_json(receipt_path)
        index = int(receipt.get("index", -1))
        row_begin = int(receipt.get("row_begin", -1))
        row_end = int(receipt.get("row_end", -1))
        if index != expected_index or row_begin != next_row or row_end <= row_begin:
            raise ValueError(
                f"layer {layer} chunk index has a gap, overlap, or out-of-order receipt"
            )
        path = receipt_path.parent / str(receipt.get("file", ""))
        digest = str(receipt.get("sha256", ""))
        if not path.is_file():
            raise ValueError(f"capture chunk file is missing: {path}")
        if verify_hashes and _sha256(path) != digest:
            raise ValueError(f"capture chunk checksum differs: {path}")
        chunk = AllRowChunk(
            layer=layer,
            index=index,
            row_begin=row_begin,
            row_end=row_end,
            path=path,
            receipt_path=receipt_path,
            sha256=digest,
        )
        chunks.append(chunk)
        next_row = row_end
    return tuple(chunks)


def finalize_all_row_capture(
    root: str | Path,
    *,
    rank_receipts: Mapping[int, str],
) -> dict[str, object]:
    """Verify every layer and seal the root manifest."""

    root = Path(root)
    manifest = _read_json(root / "manifest.json")
    if bool(manifest.get("complete", False)):
        raise ValueError("capture is already finalized")
    geometry = AllRowCaptureGeometry.from_manifest(manifest)
    world = int(manifest.get("tp_world_size", 0))
    if sorted(rank_receipts) != list(range(world)):
        raise ValueError("rank receipts do not cover the complete TP world")
    if any(len(value) != 64 for value in rank_receipts.values()):
        raise ValueError("rank receipts must be SHA-256 digests")
    rows_by_layer: dict[str, int] = {}
    for layer in geometry.layers:
        chunks = layer_chunks(root, layer)
        if not chunks:
            raise ValueError(f"layer {layer} contains no capture chunks")
        rows_by_layer[str(layer)] = chunks[-1].row_end
    distinct_rows = set(rows_by_layer.values())
    if len(distinct_rows) != 1:
        raise ValueError("capture layers do not contain identical row counts")
    rows = distinct_rows.pop()
    expected_rows = manifest.get("expected_rows")
    if expected_rows is not None and rows != int(expected_rows):
        raise ValueError(f"capture contains {rows} rows; expected {expected_rows}")
    manifest.update(
        {
            "complete": True,
            "rows": rows,
            "rows_by_layer": rows_by_layer,
            "rank_receipts": {str(rank): digest for rank, digest in sorted(rank_receipts.items())},
        }
    )
    _atomic_json(root / "manifest.json", manifest)
    return manifest


def load_all_row_capture(
    root: str | Path,
    *,
    verify_hashes: bool = True,
) -> tuple[dict[str, object], AllRowCaptureGeometry, dict[int, tuple[AllRowChunk, ...]]]:
    """Open a finalized capture and validate its complete chunk index."""

    root = Path(root)
    manifest = _read_json(root / "manifest.json")
    if manifest.get("kind") != ALL_ROW_CAPTURE_KIND:
        raise ValueError("capture kind is not the canonical all-row schema")
    if int(manifest.get("schema_version", 0)) != ALL_ROW_CAPTURE_SCHEMA_VERSION:
        raise ValueError("unsupported all-row capture schema")
    if not bool(manifest.get("complete", False)):
        raise ValueError("all-row capture is not finalized")
    geometry = AllRowCaptureGeometry.from_manifest(manifest)
    chunks = {
        layer: layer_chunks(root, layer, verify_hashes=verify_hashes)
        for layer in geometry.layers
    }
    expected_rows = int(manifest.get("rows", -1))
    if expected_rows <= 0 or any(values[-1].row_end != expected_rows for values in chunks.values()):
        raise ValueError("all-row capture row count does not close")
    return manifest, geometry, chunks


def iter_layer_rows(
    root: str | Path,
    layer: int,
    *,
    verify_hashes: bool = True,
) -> Iterator[dict[str, torch.Tensor]]:
    """Stream verified canonical rows for one layer."""

    for chunk in layer_chunks(root, layer, verify_hashes=verify_hashes):
        yield chunk.load()


def materialize_layer_rows(
    root: str | Path,
    layer: int,
    *,
    fields: Sequence[str] = REQUIRED_TENSORS,
    num_experts: int = C.NUM_EXPERTS,
    verify_hashes: bool = True,
) -> MaterializedLayerRows:
    """Load one layer once and build an expert-occurrence index.

    A complete 4,000,000-row Kimi layer occupies about 59 GiB with all fields
    and about 28 GiB when only the input and routing tensors are retained.  The
    one-time materialization avoids rereading every chunk for each of 896
    experts while keeping memory bounded to one layer.
    """

    requested = tuple(dict.fromkeys(fields))
    if not requested:
        raise ValueError("materialization requires at least one tensor field")
    unknown = set(requested) - set(REQUIRED_TENSORS)
    if unknown:
        raise ValueError(f"unknown capture fields: {sorted(unknown)}")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    required = tuple(
        dict.fromkeys((*requested, "expert_indices", "route_weights"))
    )
    chunks = layer_chunks(root, layer, verify_hashes=verify_hashes)
    if not chunks:
        raise ValueError(f"layer {layer} contains no capture chunks")
    total_rows = chunks[-1].row_end
    tensors: dict[str, torch.Tensor] = {}
    cursor = 0
    for chunk in chunks:
        loaded = chunk.load(fields=required)
        if not tensors:
            for name in required:
                source = loaded[name]
                tensors[name] = torch.empty(
                    (total_rows, *source.shape[1:]),
                    dtype=source.dtype,
                    device="cpu",
                )
        rows = loaded["input"].shape[0]
        for name in required:
            tensors[name][cursor : cursor + rows].copy_(loaded[name])
        cursor += rows
    if cursor != total_rows:
        raise AssertionError("materialized row count differs from the chunk index")

    ids = tensors["expert_indices"].long()
    if bool(torch.any(ids >= num_experts)):
        raise ValueError("captured expert identity exceeds num_experts")
    rows, top_k = ids.shape
    flat_ids = ids.reshape(-1)
    order = torch.argsort(flat_ids, stable=True)
    counts = torch.bincount(flat_ids, minlength=num_experts)
    flat_rows = torch.arange(rows, dtype=torch.int64).repeat_interleave(top_k)
    flat_slots = torch.arange(top_k, dtype=torch.uint8).repeat(rows)
    sorted_rows = flat_rows.index_select(0, order)
    sorted_slots = flat_slots.index_select(0, order)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    expert_index = tuple(
        ExpertRowIndex(
            sorted_rows[offsets[expert] : offsets[expert + 1]].contiguous(),
            sorted_slots[offsets[expert] : offsets[expert + 1]].contiguous(),
        )
        for expert in range(num_experts)
    )
    return MaterializedLayerRows(
        layer=layer,
        tensors=tensors,
        expert_index=expert_index,
    )


def map_layer_rows(
    root: str | Path,
    layer: int,
    *,
    num_experts: int = C.NUM_EXPERTS,
    verify_hashes: bool = True,
) -> MappedLayerRows:
    """Map one layer's inputs and build a compact expert occurrence index.

    The capture files remain the backing storage for input rows.  Only global
    row identities, route slots, and applied route weights are copied into the
    process.  This is the appropriate representation when several GPU workers
    score different layers concurrently on a finite-memory host.
    """

    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    chunks = layer_chunks(root, layer, verify_hashes=verify_hashes)
    if not chunks:
        raise ValueError(f"layer {layer} contains no capture chunks")

    loaded_chunks: list[dict[str, torch.Tensor]] = []
    counts = torch.zeros(num_experts, dtype=torch.int64)
    top_k: int | None = None
    for chunk in chunks:
        loaded = chunk.load(fields=("input", "expert_indices", "route_weights"))
        ids = loaded["expert_indices"]
        if top_k is None:
            top_k = int(ids.shape[1])
        elif ids.shape[1] != top_k:
            raise ValueError("mapped layer chunks disagree on top-k geometry")
        if bool(torch.any(ids >= num_experts)):
            raise ValueError("captured expert identity exceeds num_experts")
        counts.add_(torch.bincount(ids.reshape(-1).long(), minlength=num_experts))
        loaded_chunks.append(loaded)
    assert top_k is not None

    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    total_occurrences = int(offsets[-1])
    row_indices = torch.empty(total_occurrences, dtype=torch.int64)
    route_slots = torch.empty(total_occurrences, dtype=torch.uint8)
    route_weights = torch.empty(total_occurrences, dtype=torch.float32)
    cursors = torch.zeros(num_experts, dtype=torch.int64)

    for chunk, loaded in zip(chunks, loaded_chunks):
        flat_ids = loaded["expert_indices"].reshape(-1).long()
        order = torch.argsort(flat_ids, stable=True)
        ordered_ids = flat_ids.index_select(0, order)
        chunk_counts = torch.bincount(flat_ids, minlength=num_experts)
        chunk_offsets = torch.cat(
            (torch.zeros(1, dtype=torch.int64), chunk_counts.cumsum(0))
        )
        local_rank = torch.arange(order.numel(), dtype=torch.int64) - torch.repeat_interleave(
            chunk_offsets[:-1], chunk_counts
        )
        destinations = (
            offsets.index_select(0, ordered_ids)
            + cursors.index_select(0, ordered_ids)
            + local_rank
        )
        row_indices.index_copy_(
            0,
            destinations,
            order.div(top_k, rounding_mode="floor").add(chunk.row_begin),
        )
        route_slots.index_copy_(0, destinations, order.remainder(top_k).to(torch.uint8))
        route_weights.index_copy_(
            0,
            destinations,
            loaded["route_weights"].reshape(-1).index_select(0, order),
        )
        cursors.add_(chunk_counts)
    if not torch.equal(cursors, counts):
        raise AssertionError("mapped layer expert occurrence counts did not close")

    expert_index = tuple(
        MappedExpertRowIndex(
            row_indices[offsets[expert] : offsets[expert + 1]],
            route_slots[offsets[expert] : offsets[expert + 1]],
            route_weights[offsets[expert] : offsets[expert + 1]],
        )
        for expert in range(num_experts)
    )
    return MappedLayerRows(
        layer=layer,
        population_rows=chunks[-1].row_end,
        input_chunks=tuple(loaded["input"] for loaded in loaded_chunks),
        chunk_begins=torch.tensor([chunk.row_begin for chunk in chunks], dtype=torch.int64),
        chunk_ends=torch.tensor([chunk.row_end for chunk in chunks], dtype=torch.int64),
        expert_index=expert_index,
    )


def materialize_expert_rows(
    root: str | Path,
    layer: int,
    expert: int,
    *,
    fields: Sequence[str] = ("input",),
    verify_hashes: bool = True,
) -> MaterializedExpertRows:
    """Load only one expert's naturally routed occurrences from a layer."""

    requested = tuple(dict.fromkeys(fields))
    if not requested:
        raise ValueError("materialization requires at least one tensor field")
    unknown = set(requested) - set(REQUIRED_TENSORS)
    if unknown:
        raise ValueError(f"unknown capture fields: {sorted(unknown)}")
    chunks = layer_chunks(root, layer, verify_hashes=verify_hashes)
    if not chunks:
        raise ValueError(f"layer {layer} contains no capture chunks")
    tensors: dict[str, list[torch.Tensor]] = {name: [] for name in requested}
    row_indices: list[torch.Tensor] = []
    route_slots: list[torch.Tensor] = []
    route_weights: list[torch.Tensor] = []
    for chunk in chunks:
        loaded = chunk.load(fields=tuple(dict.fromkeys((*requested, "expert_indices", "route_weights"))))
        matches = torch.nonzero(loaded["expert_indices"] == expert, as_tuple=False)
        if matches.numel() == 0:
            continue
        local_rows = matches[:, 0]
        slots = matches[:, 1]
        for name in requested:
            tensors[name].append(loaded[name].index_select(0, local_rows))
        row_indices.append(local_rows.add(chunk.row_begin))
        route_slots.append(slots.to(torch.uint8))
        route_weights.append(loaded["route_weights"][local_rows, slots])
    if not row_indices:
        raise ValueError(f"expert {expert} has no routed occurrences in layer {layer}")
    return MaterializedExpertRows(
        layer=layer,
        expert=expert,
        population_rows=chunks[-1].row_end,
        tensors={name: torch.cat(values, dim=0).contiguous() for name, values in tensors.items()},
        row_indices=torch.cat(row_indices).contiguous(),
        route_slots=torch.cat(route_slots).contiguous(),
        route_weights=torch.cat(route_weights).contiguous(),
    )


__all__ = [
    "ALL_ROW_CAPTURE_KIND",
    "ALL_ROW_CAPTURE_SCHEMA_VERSION",
    "AllRowCaptureGeometry",
    "AllRowChunk",
    "ExpertRowIndex",
    "MaterializedLayerRows",
    "MaterializedExpertRows",
    "MappedExpertRowIndex",
    "MappedLayerRows",
    "DEFAULT_CHUNK_ROWS",
    "REQUIRED_TENSORS",
    "finalize_all_row_capture",
    "initialize_all_row_capture",
    "iter_layer_rows",
    "layer_chunks",
    "load_all_row_capture",
    "materialize_layer_rows",
    "materialize_expert_rows",
    "map_layer_rows",
    "write_all_row_chunk",
]

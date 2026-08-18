"""Exact top-k route capture and comparison for streamed Kimi execution."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from qsrt.kimi_forward_pipeline import KimiForwardPipelineAdapter


SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
PARTIAL_MANIFEST_FILENAME = "manifest.partial.json"


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _layer_filename(layer: int) -> str:
    return f"layer-{layer:03d}.i16"


class _RouteWriter:
    def __init__(self, archive: "KimiRouteArchive", layer: int):
        self.archive = archive
        self.layer = int(layer)
        self.path = archive.root / _layer_filename(layer)
        self.temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        self.stream = self.temporary.open("wb")
        self.rows = 0
        self.closed = False

    def append(self, indices: torch.Tensor) -> None:
        if self.closed:
            raise RuntimeError("route writer is closed")
        if indices.ndim != 2 or indices.shape[1] != self.archive.top_k:
            raise ValueError("route indices have incompatible top-k geometry")
        if indices.numel() and (
            int(indices.min().item()) < 0
            or int(indices.max().item()) >= self.archive.num_experts
        ):
            raise ValueError("route indices are outside the expert range")
        value = indices.detach().to(device="cpu", dtype=torch.int16).contiguous()
        self.stream.write(memoryview(value.numpy()).cast("B"))
        self.rows += int(value.shape[0])

    def finish(self) -> None:
        if self.closed:
            return
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        self.closed = True
        if self.rows != self.archive.token_count:
            return
        os.replace(self.temporary, self.path)
        self.archive._record_layer(self.layer, self.path)


class KimiRouteArchive:
    """One raw int16 top-k matrix per decoder layer."""

    def __init__(self, root: str | Path, *, require_complete: bool = False):
        self.root = Path(root).expanduser().resolve()
        manifest = self.root / MANIFEST_FILENAME
        partial = self.root / PARTIAL_MANIFEST_FILENAME
        path = manifest if manifest.is_file() else partial
        if require_complete and not manifest.is_file():
            raise FileNotFoundError(manifest)
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read route archive {path}: {error}") from error
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path}: unsupported route archive schema")
        self.manifest_path = path
        self.manifest = document
        self.token_count = int(document["token_count"])
        self.num_layers = int(document["num_layers"])
        self.first_layer = int(document["first_layer"])
        self.num_experts = int(document["num_experts"])
        self.top_k = int(document["top_k"])
        self._lock = threading.Lock()

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        token_count: int,
        num_layers: int,
        first_layer: int = 1,
        num_experts: int,
        top_k: int,
        provenance: Mapping[str, object] | None = None,
    ) -> "KimiRouteArchive":
        if min(token_count, num_layers, num_experts, top_k) <= 0:
            raise ValueError("route archive geometry must be positive")
        if not 0 <= first_layer < num_layers:
            raise ValueError("route archive layer range is invalid")
        if num_experts > torch.iinfo(torch.int16).max:
            raise ValueError("route archive int16 storage cannot represent expert IDs")
        destination = Path(root).expanduser().resolve()
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"route archive destination is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            destination / PARTIAL_MANIFEST_FILENAME,
            {
                "schema_version": SCHEMA_VERSION,
                "complete": False,
                "token_count": int(token_count),
                "num_layers": int(num_layers),
                "first_layer": int(first_layer),
                "num_experts": int(num_experts),
                "top_k": int(top_k),
                "dtype": "int16",
                "layers": {},
                "provenance": dict(provenance or {}),
            },
        )
        return cls(destination)

    @property
    def expected_layer_bytes(self) -> int:
        return self.token_count * self.top_k * torch.int16.itemsize

    def writer(self, layer: int) -> _RouteWriter:
        if not self.first_layer <= layer < self.num_layers:
            raise IndexError(layer)
        return _RouteWriter(self, layer)

    def _record_layer(self, layer: int, path: Path) -> None:
        if path.stat().st_size != self.expected_layer_bytes:
            raise ValueError(f"route layer {layer} has the wrong byte count")
        with self._lock:
            layers = self.manifest.setdefault("layers", {})
            if str(layer) in layers:
                raise ValueError(f"route layer {layer} was finalized twice")
            layers[str(layer)] = {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            _atomic_json(self.root / PARTIAL_MANIFEST_FILENAME, self.manifest)

    def seal(self) -> None:
        expected = set(range(self.first_layer, self.num_layers))
        observed = {int(value) for value in self.manifest.get("layers", {})}
        if observed != expected:
            raise ValueError(
                f"route archive is incomplete; missing={sorted(expected - observed)[:8]}"
            )
        self.manifest["complete"] = True
        _atomic_json(self.root / MANIFEST_FILENAME, self.manifest)
        (self.root / PARTIAL_MANIFEST_FILENAME).unlink()
        self.manifest_path = self.root / MANIFEST_FILENAME

    def read_layer(self, layer: int) -> torch.Tensor:
        if not self.first_layer <= layer < self.num_layers:
            raise IndexError(layer)
        path = self.root / _layer_filename(layer)
        if path.stat().st_size != self.expected_layer_bytes:
            raise ValueError(f"route layer {layer} has the wrong byte count")
        return torch.from_file(
            str(path),
            shared=False,
            size=self.token_count * self.top_k,
            dtype=torch.int16,
        ).reshape(self.token_count, self.top_k)


class RouteCapturingAdapter:
    """Add top-k route recording to a streamed decoder adapter."""

    def __init__(self, adapter: KimiForwardPipelineAdapter, archive: KimiRouteArchive):
        self.adapter = adapter
        self.archive = archive
        self._state: dict[int, tuple[Any, _RouteWriter]] = {}

    def load_layer(self, layer: int, device: torch.device) -> tuple[Any, object | None]:
        module, receipt = self.adapter.load_layer(layer, device)
        block = getattr(module, "block_sparse_moe", None)
        gate = None if block is None else getattr(block, "gate", None)
        if gate is None:
            if layer < self.archive.first_layer:
                return module, receipt
            self.adapter.release_layer(module)
            raise TypeError(f"decoder layer {layer} has no routed MoE gate")
        writer = self.archive.writer(layer)

        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            if not isinstance(output, tuple) or len(output) != 2:
                raise TypeError("Kimi gate did not return route indices and weights")
            writer.append(output[0])

        handle = gate.register_forward_hook(capture)
        self._state[id(module)] = (handle, writer)
        return module, receipt

    def forward_layer(
        self,
        module: Any,
        *,
        layer: int,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.adapter.forward_layer(
            module,
            layer=layer,
            hidden_states=hidden_states,
            block_residual=block_residual,
        )

    def release_layer(self, module: Any) -> None:
        state = self._state.pop(id(module), None)
        if state is not None:
            handle, writer = state
            handle.remove()
            writer.finish()
        self.adapter.release_layer(module)


@dataclass(frozen=True)
class LayerRouteAgreement:
    layer: int
    mean_topk_overlap: float
    exact_topk_set_agreement: float
    marginal_total_variation: float


def compare_route_archives(
    teacher: KimiRouteArchive,
    student: KimiRouteArchive,
    *,
    chunk_tokens: int = 8192,
) -> dict[str, object]:
    """Compare route sets and expert-selection marginals layer by layer."""

    geometry = (
        teacher.token_count,
        teacher.num_layers,
        teacher.first_layer,
        teacher.num_experts,
        teacher.top_k,
    )
    if geometry != (
        student.token_count,
        student.num_layers,
        student.first_layer,
        student.num_experts,
        student.top_k,
    ):
        raise ValueError("teacher and student route archives have different geometry")
    if chunk_tokens <= 0:
        raise ValueError("route comparison chunk size must be positive")
    records: list[LayerRouteAgreement] = []
    for layer in range(teacher.first_layer, teacher.num_layers):
        left = teacher.read_layer(layer)
        right = student.read_layer(layer)
        overlap_sum = 0
        exact_sum = 0
        left_counts = torch.zeros(teacher.num_experts, dtype=torch.int64)
        right_counts = torch.zeros_like(left_counts)
        for first in range(0, teacher.token_count, chunk_tokens):
            end = min(first + chunk_tokens, teacher.token_count)
            a = left[first:end].to(torch.int64)
            b = right[first:end].to(torch.int64)
            intersection = (a.unsqueeze(2) == b.unsqueeze(1)).any(dim=2).sum(dim=1)
            overlap_sum += int(intersection.sum().item())
            exact_sum += int((intersection == teacher.top_k).sum().item())
            left_counts += torch.bincount(a.reshape(-1), minlength=teacher.num_experts)
            right_counts += torch.bincount(b.reshape(-1), minlength=teacher.num_experts)
        selections = teacher.token_count * teacher.top_k
        records.append(
            LayerRouteAgreement(
                layer=layer,
                mean_topk_overlap=overlap_sum / selections,
                exact_topk_set_agreement=exact_sum / teacher.token_count,
                marginal_total_variation=(
                    0.5 * float((left_counts - right_counts).abs().sum().item()) / selections
                ),
            )
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "teacher": str(teacher.root),
        "student": str(student.root),
        "token_count": teacher.token_count,
        "top_k": teacher.top_k,
        "layers": [asdict(record) for record in records],
        "mean_topk_overlap": sum(record.mean_topk_overlap for record in records)
        / len(records),
        "mean_exact_topk_set_agreement": sum(
            record.exact_topk_set_agreement for record in records
        )
        / len(records),
        "mean_marginal_total_variation": sum(
            record.marginal_total_variation for record in records
        )
        / len(records),
    }


__all__ = [
    "KimiRouteArchive",
    "LayerRouteAgreement",
    "RouteCapturingAdapter",
    "compare_route_archives",
]

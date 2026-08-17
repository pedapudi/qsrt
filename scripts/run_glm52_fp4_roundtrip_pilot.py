#!/usr/bin/env python3
"""Measure MXFP4 and NVFP4 distortion on the frozen GLM-5.2 K4 panel."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

import torch

from qsrt.fp4_roundtrip import mxfp4_roundtrip, nvfp4_roundtrip
from qsrt.glm52_pilot import (
    K4_PANEL,
    PROJECTIONS,
    SOURCE_INDEX_SHA256,
    SOURCE_REVISION,
    IndexedTensorStore,
    atomic_write_json,
    metric_terms,
    panel_cells,
    source_tensor_name,
)


PILOT_KIND = "qsrt_glm52_fp4_roundtrip_pilot_v1"
FORMATS = {
    "mxfp4": {"nominal_bpw": 4.25, "roundtrip": mxfp4_roundtrip},
    "nvfp4": {"nominal_bpw": 4.5, "roundtrip": nvfp4_roundtrip},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser


def _manifest(source: Path, device: torch.device) -> dict[str, Any]:
    try:
        flashinfer_version = importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError:
        flashinfer_version = importlib.metadata.version("flashinfer")
    return {
        "kind": PILOT_KIND,
        "source": {
            "root": str(source.resolve()),
            "revision": SOURCE_REVISION,
            "index_sha256": SOURCE_INDEX_SHA256,
        },
        "panel": {str(layer): list(experts) for layer, experts in K4_PANEL.items()},
        "formats": {
            name: {"nominal_bpw": spec["nominal_bpw"]}
            for name, spec in FORMATS.items()
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "flashinfer": flashinfer_version,
        },
    }


def _prepare_dest(dest: Path, manifest: dict[str, Any], resume: bool) -> None:
    manifest_path = dest / "manifest.json"
    if dest.exists() and not resume:
        raise FileExistsError(f"destination already exists: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        stored = json.loads(manifest_path.read_text())
        if stored != manifest:
            raise ValueError("resume manifest differs from the existing study")
    else:
        atomic_write_json(manifest_path, manifest)


def _record_path(dest: Path, layer: int, expert: int) -> Path:
    return dest / "experts" / f"layer-{layer:03d}-expert-{expert:03d}.json"


def _sum_format(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    energy = sum(record["source_energy"] for record in records)
    sse = sum(record["formats"][name]["sse"] for record in records)
    weight_count = sum(record["weight_count"] for record in records)
    payload_bytes = sum(record["formats"][name]["payload_bytes"] for record in records)
    return {
        "source_energy": energy,
        "sse": sse,
        "relative_sse": sse / energy,
        "weight_count": weight_count,
        "payload_bytes": payload_bytes,
        "effective_bpw": 8.0 * payload_bytes / weight_count,
        "nominal_bpw": FORMATS[name]["nominal_bpw"],
    }


def _run_expert(
    source: IndexedTensorStore,
    *,
    layer: int,
    expert: int,
    device: torch.device,
) -> dict[str, Any]:
    projections: dict[str, Any] = {}
    for projection in PROJECTIONS:
        tensor_name = source_tensor_name(layer, expert, projection.name)
        source_weight = source.get(tensor_name)
        if source_weight.dtype != torch.bfloat16:
            raise TypeError(f"{tensor_name} has dtype {source_weight.dtype}, expected BF16")
        if tuple(source_weight.shape) != projection.source_shape:
            raise ValueError(
                f"{tensor_name} has shape {tuple(source_weight.shape)}, "
                f"expected {projection.source_shape}"
            )
        source_cuda = source_weight.to(device)
        source_energy, _ = metric_terms(source_weight, source_weight)
        format_metrics: dict[str, Any] = {}
        for name, spec in FORMATS.items():
            result = spec["roundtrip"](source_cuda)
            reconstruction = result.reconstruction
            energy_check, sse = metric_terms(source_weight, reconstruction)
            if energy_check != source_energy:
                raise AssertionError("source energy changed between format arms")
            format_metrics[name] = {
                "sse": sse,
                "relative_sse": sse / source_energy,
                "payload_bytes": result.payload_bytes,
                "value_bytes": result.value_bytes,
                "block_scale_bytes": result.block_scale_bytes,
                "tensor_scale_bytes": result.tensor_scale_bytes,
                "effective_bpw": result.effective_bpw(source_weight.numel()),
            }
            del reconstruction, result
        projections[projection.name] = {
            "source_energy": source_energy,
            "weight_count": source_weight.numel(),
            "formats": format_metrics,
        }
        del source_cuda, source_weight

    energy = sum(item["source_energy"] for item in projections.values())
    weight_count = sum(item["weight_count"] for item in projections.values())
    formats = {}
    for name in FORMATS:
        sse = sum(item["formats"][name]["sse"] for item in projections.values())
        payload_bytes = sum(
            item["formats"][name]["payload_bytes"] for item in projections.values()
        )
        formats[name] = {
            "sse": sse,
            "relative_sse": sse / energy,
            "payload_bytes": payload_bytes,
            "effective_bpw": 8.0 * payload_bytes / weight_count,
        }
    return {
        "kind": PILOT_KIND,
        "layer": layer,
        "expert": expert,
        "source_energy": energy,
        "weight_count": weight_count,
        "formats": formats,
        "projections": projections,
    }


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("FlashInfer FP4 pilot requires a CUDA device")
    manifest = _manifest(args.source, device)
    _prepare_dest(args.dest, manifest, args.resume)
    source = IndexedTensorStore(args.source)
    records: list[dict[str, Any]] = []
    total = len(panel_cells(K4_PANEL))
    for ordinal, (layer, expert) in enumerate(panel_cells(K4_PANEL), start=1):
        path = _record_path(args.dest, layer, expert)
        if path.is_file():
            record = json.loads(path.read_text())
        else:
            started = time.monotonic()
            record = _run_expert(
                source, layer=layer, expert=expert, device=device
            )
            record["wall_seconds"] = time.monotonic() - started
            atomic_write_json(path, record)
        records.append(record)
        print(
            f"[{ordinal:02d}/{total}] layer {layer} expert {expert}: "
            f"MXFP4={record['formats']['mxfp4']['relative_sse']:.8f} "
            f"NVFP4={record['formats']['nvfp4']['relative_sse']:.8f}",
            flush=True,
        )
        torch.cuda.empty_cache()

    report = {
        "kind": PILOT_KIND,
        "status": "complete",
        "expert_count": len(records),
        "matrix_count": len(records) * len(PROJECTIONS),
        "aggregate": {name: _sum_format(records, name) for name in FORMATS},
        "experts": records,
    }
    atomic_write_json(args.dest / "report.json", report)
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

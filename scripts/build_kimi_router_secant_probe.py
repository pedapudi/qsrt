#!/usr/bin/env python3
"""Construct one bounded router-bias secant calibration probe."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from qsrt.router_bias_secant import build_secant_bias_update


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _load(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    return load_file(path, device="cpu"), json.loads(path.with_suffix(".json").read_text())


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {}
    source = values.to(torch.float64)
    return {
        name: float(torch.quantile(source, probability))
        for name, probability in (
            ("minimum", 0.0),
            ("p01", 0.01),
            ("p10", 0.10),
            ("median", 0.50),
            ("p90", 0.90),
            ("p99", 0.99),
            ("maximum", 1.0),
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--slope-before", type=Path, required=True)
    parser.add_argument("--slope-after", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--noise-sigma", type=float, default=2.5)
    parser.add_argument("--margin-multiple", type=float, default=32.0)
    args = parser.parse_args()

    paths = {
        "teacher": args.teacher.expanduser().resolve(),
        "slope_before": args.slope_before.expanduser().resolve(),
        "slope_after": args.slope_after.expanduser().resolve(),
        "base": args.base.expanduser().resolve(),
    }
    loaded = {name: _load(path) for name, path in paths.items()}
    tensors = {name: value[0] for name, value in loaded.items()}
    reports = {name: value[1] for name, value in loaded.items()}
    reference_population = reports["teacher"]["population"]
    reference_hash = reference_population["corpus_plan_sha256"]
    tokens = int(reference_population["tokens"])
    top_k = int(reports["teacher"]["decoder"]["top_k"])
    active = tensors["teacher"]["active_layers"].to(torch.bool)
    geometry = tensors["teacher"]["selection_counts"].shape
    for name in paths:
        report = reports[name]
        value = tensors[name]
        if (
            report["population"]["corpus_plan_sha256"] != reference_hash
            or int(report["population"]["tokens"]) != tokens
            or int(report["decoder"]["top_k"]) != top_k
            or value["selection_counts"].shape != geometry
            or not torch.equal(value["active_layers"].to(torch.bool), active)
        ):
            raise ValueError(f"{name} does not match the teacher capture contract")
    if not 0 <= args.layer < geometry[0] or not bool(active[args.layer]):
        raise ValueError("requested layer is not an active routed layer")
    for name in ("slope_before", "slope_after", "base"):
        if tensors[name]["biases"].shape != geometry:
            raise ValueError(f"{name} has invalid bias geometry")
    for name in paths:
        if int(tensors[name]["selection_counts"][args.layer].sum()) != tokens * top_k:
            raise ValueError(f"{name} selection counts do not close")

    base_layers = {int(row["layer"]): row for row in reports["base"]["layers"]}
    median_margin = float(base_layers[args.layer]["margin_p50"])
    result = build_secant_bias_update(
        teacher_counts=tensors["teacher"]["selection_counts"][args.layer],
        slope_before_counts=tensors["slope_before"]["selection_counts"][args.layer],
        slope_after_counts=tensors["slope_after"]["selection_counts"][args.layer],
        base_counts=tensors["base"]["selection_counts"][args.layer],
        slope_before_bias=tensors["slope_before"]["biases"][args.layer],
        slope_after_bias=tensors["slope_after"]["biases"][args.layer],
        tokens=tokens,
        median_margin=median_margin,
        noise_sigma=args.noise_sigma,
        margin_multiple=args.margin_multiple,
    )
    biases = tensors["base"]["biases"].to(torch.float32).clone()
    biases[args.layer] += result.update
    output = args.output.expanduser().resolve()
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    save_file(
        {
            "biases": biases.contiguous(),
            "active_layers": active.contiguous(),
            "secant_update": result.update.contiguous(),
            "secant_slopes": result.slopes.contiguous(),
            "secant_valid": result.valid.contiguous(),
        },
        str(temporary),
        metadata={
            "kind": "Kimi-K3 layer-local router-bias secant calibration probe",
            "layer": str(args.layer),
            "base_capture": str(paths["base"]),
            "corpus_plan_sha256": str(reference_hash),
        },
    )
    os.replace(temporary, output)
    valid_proposals = result.raw_proposal[result.valid]
    valid_updates = result.bounded_proposal[result.valid]
    report = {
        "kind": "Kimi-K3 layer-local router-bias secant calibration probe",
        "status": "research-only pre-registered experiment",
        "sources": {name: str(path) for name, path in paths.items()},
        "output": str(output),
        "corpus_plan_sha256": reference_hash,
        "tokens": tokens,
        "top_k": top_k,
        "layer": args.layer,
        "noise_sigma": args.noise_sigma,
        "margin_multiple": args.margin_multiple,
        "median_selection_margin": median_margin,
        "absolute_update_clamp": result.clamp,
        "resolved_frequency_movements": int(result.resolved.sum()),
        "valid_positive_slopes": int(result.valid.sum()),
        "raw_proposal": _quantiles(valid_proposals),
        "bounded_causal_proposal": _quantiles(valid_updates),
        "update_mean": float(result.update.to(torch.float64).mean()),
        "update_l2": float(torch.linalg.vector_norm(result.update)),
        "update_max_abs": float(result.update.abs().max()),
        "unchanged_layers": [
            layer for layer in torch.where(active)[0].tolist() if layer != args.layer
        ],
    }
    _atomic_json(output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

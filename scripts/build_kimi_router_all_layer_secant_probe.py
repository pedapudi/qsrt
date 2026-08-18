#!/usr/bin/env python3
"""Construct one bounded all-layer router-bias secant probe."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from qsrt.router_bias_secant import build_least_squares_secant_bias_update


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


def _tv(residual: torch.Tensor, *, top_k: int) -> float:
    return 0.5 * float(residual.to(torch.float64).abs().sum()) / top_k


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--round", dest="rounds", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise-sigma", type=float, default=2.5)
    parser.add_argument("--margin-multiple", type=float, default=64.0)
    args = parser.parse_args()
    if len(args.rounds) != 5:
        raise ValueError("the registered experiment requires exactly five rounds")

    teacher_path = args.teacher.expanduser().resolve()
    round_paths = [path.expanduser().resolve() for path in args.rounds]
    teacher, teacher_report = _load(teacher_path)
    loaded_rounds = [_load(path) for path in round_paths]
    round_tensors = [value[0] for value in loaded_rounds]
    round_reports = [value[1] for value in loaded_rounds]

    reference_population = teacher_report["population"]
    reference_hash = reference_population["corpus_plan_sha256"]
    tokens = int(reference_population["tokens"])
    top_k = int(teacher_report["decoder"]["top_k"])
    active = teacher["active_layers"].to(torch.bool)
    geometry = teacher["selection_counts"].shape
    for index, (path, tensors, report) in enumerate(
        zip(round_paths, round_tensors, round_reports, strict=True)
    ):
        if (
            report["population"]["corpus_plan_sha256"] != reference_hash
            or int(report["population"]["tokens"]) != tokens
            or int(report["decoder"]["top_k"]) != top_k
            or tensors["selection_counts"].shape != geometry
            or tensors["biases"].shape != geometry
            or not torch.equal(tensors["active_layers"].to(torch.bool), active)
        ):
            raise ValueError(f"round {index} does not match the teacher contract: {path}")
    for layer in torch.where(active)[0].tolist():
        if int(teacher["selection_counts"][layer].sum()) != tokens * top_k:
            raise ValueError(f"teacher selection counts do not close at layer {layer}")
        for index, tensors in enumerate(round_tensors):
            if int(tensors["selection_counts"][layer].sum()) != tokens * top_k:
                raise ValueError(
                    f"round {index} selection counts do not close at layer {layer}"
                )

    base = round_tensors[-1]
    base_layers = {int(row["layer"]): row for row in round_reports[-1]["layers"]}
    biases = base["biases"].to(torch.float32).clone()
    updates = torch.zeros_like(biases)
    slopes = torch.zeros_like(biases)
    slope_resolved = torch.zeros(geometry, dtype=torch.bool)
    residual_resolved = torch.zeros_like(slope_resolved)
    valid = torch.zeros_like(slope_resolved)
    predicted_frequencies = torch.zeros_like(biases)
    layer_records: list[dict[str, Any]] = []
    teacher_frequency = teacher["selection_counts"].to(torch.float64) / tokens
    base_frequency = base["selection_counts"].to(torch.float64) / tokens

    for layer in torch.where(active)[0].tolist():
        median_margin = float(base_layers[layer]["margin_p50"])
        result = build_least_squares_secant_bias_update(
            teacher_counts=teacher["selection_counts"][layer],
            round_counts=torch.stack(
                [value["selection_counts"][layer] for value in round_tensors]
            ),
            round_biases=torch.stack([value["biases"][layer] for value in round_tensors]),
            tokens=tokens,
            median_margin=median_margin,
            noise_sigma=args.noise_sigma,
            margin_multiple=args.margin_multiple,
        )
        biases[layer] += result.update
        updates[layer] = result.update
        slopes[layer] = result.slopes
        slope_resolved[layer] = result.slope_resolved
        residual_resolved[layer] = result.residual_resolved
        valid[layer] = result.valid
        predicted_frequencies[layer] = result.predicted_frequency

        base_residual = base_frequency[layer] - teacher_frequency[layer]
        predicted_residual = (
            result.predicted_frequency.to(torch.float64) - teacher_frequency[layer]
        )
        layer_records.append(
            {
                "layer": layer,
                "median_selection_margin": median_margin,
                "absolute_update_clamp": result.clamp,
                "resolved_positive_slopes": int(result.slope_resolved.sum()),
                "noise_resolved_residuals": int(result.residual_resolved.sum()),
                "updated_experts": int(result.valid.sum()),
                "base_marginal_total_variation": _tv(base_residual, top_k=top_k),
                "predicted_marginal_total_variation": _tv(
                    predicted_residual, top_k=top_k
                ),
                "base_noise_resolved_marginal_total_variation": _tv(
                    torch.where(result.residual_resolved, base_residual, 0.0),
                    top_k=top_k,
                ),
                "predicted_raw_residual_sign_flips": int(
                    ((base_residual * predicted_residual) < 0).sum()
                ),
                "predicted_updated_residual_sign_flips": int(
                    (((base_residual * predicted_residual) < 0) & result.valid).sum()
                ),
                "update_l2": float(torch.linalg.vector_norm(result.update)),
                "update_max_abs": float(result.update.abs().max()),
                "slope": _quantiles(result.slopes[result.slope_resolved]),
                "raw_proposal": _quantiles(result.raw_proposal[result.valid]),
                "bounded_proposal": _quantiles(
                    result.bounded_proposal[result.valid]
                ),
            }
        )

    output = args.output.expanduser().resolve()
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    save_file(
        {
            "biases": biases.contiguous(),
            "active_layers": active.contiguous(),
            "secant_update": updates.contiguous(),
            "secant_slopes": slopes.contiguous(),
            "secant_slope_resolved": slope_resolved.contiguous(),
            "secant_residual_resolved": residual_resolved.contiguous(),
            "secant_valid": valid.contiguous(),
            "predicted_frequencies": predicted_frequencies.contiguous(),
        },
        str(temporary),
        metadata={
            "kind": "Kimi-K3 all-layer router-bias secant closure probe",
            "base_capture": str(round_paths[-1]),
            "corpus_plan_sha256": str(reference_hash),
            "margin_multiple": str(args.margin_multiple),
        },
    )
    os.replace(temporary, output)
    base_values = [row["base_marginal_total_variation"] for row in layer_records]
    predicted_values = [
        row["predicted_marginal_total_variation"] for row in layer_records
    ]
    report = {
        "kind": "Kimi-K3 all-layer router-bias secant closure probe",
        "status": "research-only pre-registered experiment",
        "teacher": str(teacher_path),
        "rounds": [str(path) for path in round_paths],
        "base": str(round_paths[-1]),
        "output": str(output),
        "corpus_plan_sha256": reference_hash,
        "tokens": tokens,
        "top_k": top_k,
        "noise_sigma": args.noise_sigma,
        "margin_multiple": args.margin_multiple,
        "active_layers": int(active.sum()),
        "resolved_positive_slopes": int(slope_resolved.sum()),
        "updated_experts": int(valid.sum()),
        "base_mean_marginal_total_variation": sum(base_values) / len(base_values),
        "predicted_mean_marginal_total_variation": (
            sum(predicted_values) / len(predicted_values)
        ),
        "layers": layer_records,
    }
    _atomic_json(output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

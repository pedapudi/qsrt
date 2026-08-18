#!/usr/bin/env python
"""Measure document-split stability of Kimi-K3 routed-output Fisher factors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

_REPOSITORY = Path(__file__).resolve().parents[1]
if str(_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY))

from qsrt.kimi_output_factors import KimiOutputFactorArchive


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def _parse_damping_ratios(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "damping ratios must be comma-separated numbers"
        ) from error
    if not result or any(not math.isfinite(item) or item < 0.0 for item in result):
        raise argparse.ArgumentTypeError(
            "damping ratios must be finite and nonnegative"
        )
    return result


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = torch.sum(left * right)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    value = float((numerator / denominator).item())
    return max(-1.0, min(1.0, value))


def _damped(value: torch.Tensor, ratio: float) -> torch.Tensor:
    if ratio == 0.0:
        return value
    result = value.clone()
    diagonal_mean = torch.diagonal(result).mean()
    result.diagonal().add_(ratio * diagonal_mean)
    return result


def _component_cosines(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[float, float]:
    left_diagonal = torch.diagonal(left)
    right_diagonal = torch.diagonal(right)
    total_dot = torch.sum(left * right)
    left_total_squared = torch.sum(left * left)
    right_total_squared = torch.sum(right * right)
    diagonal_dot = torch.sum(left_diagonal * right_diagonal)
    left_diagonal_squared = torch.sum(left_diagonal * left_diagonal)
    right_diagonal_squared = torch.sum(right_diagonal * right_diagonal)
    off_diagonal_cosine = (
        (total_dot - diagonal_dot)
        / torch.sqrt(
            (left_total_squared - left_diagonal_squared)
            * (right_total_squared - right_diagonal_squared)
        )
    )
    dimension = left.shape[0]
    left_trace = torch.sum(left_diagonal)
    right_trace = torch.sum(right_diagonal)
    anisotropic_cosine = (
        (total_dot - left_trace * right_trace / dimension)
        / torch.sqrt(
            (left_total_squared - left_trace.square() / dimension)
            * (right_total_squared - right_trace.square() / dimension)
        )
    )
    return (
        max(-1.0, min(1.0, float(off_diagonal_cosine.item()))),
        max(-1.0, min(1.0, float(anisotropic_cosine.item()))),
    )


def _anisotropic_shrinkage_estimate(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    left_rows: int,
    right_rows: int,
) -> dict[str, float | None]:
    """Estimate full-sample Fisher shrinkage from independent document splits.

    Each split factor is modeled as the same anisotropic signal plus
    independent sample noise whose power is inversely proportional to its row
    count. The cross-split inner product estimates signal power. Excess power
    in each split estimates the per-sample noise coefficient, which is then
    scaled to the combined row count. Adding ``r * trace(H) / d * I`` and
    ignoring the objective's irrelevant common scale shrinks anisotropy by
    ``1 / (1 + r)``, so the estimated MSE-optimal damping ratio is the
    full-sample noise-to-signal ratio.
    """

    if left_rows <= 0 or right_rows <= 0:
        raise ValueError("split row counts must be positive")
    dimension = left.shape[0]
    left_diagonal = torch.diagonal(left)
    right_diagonal = torch.diagonal(right)
    left_trace = torch.sum(left_diagonal)
    right_trace = torch.sum(right_diagonal)
    signal_power = (
        torch.sum(left * right) - left_trace * right_trace / dimension
    )
    left_power = torch.sum(left * left) - left_trace.square() / dimension
    right_power = torch.sum(right * right) - right_trace.square() / dimension
    signal = float(signal_power.item())
    left_noise = max(0.0, float((left_power - signal_power).item()))
    right_noise = max(0.0, float((right_power - signal_power).item()))
    per_sample_noise = 0.5 * (
        left_rows * left_noise + right_rows * right_noise
    )
    full_noise = per_sample_noise / (left_rows + right_rows)
    if signal <= 0.0:
        return {
            "signal_power": signal,
            "estimated_full_sample_noise_power": full_noise,
            "estimated_anisotropic_shrinkage": 0.0,
            "estimated_identity_damping_ratio": None,
        }
    damping_ratio = full_noise / signal
    return {
        "signal_power": signal,
        "estimated_full_sample_noise_power": full_noise,
        "estimated_anisotropic_shrinkage": 1.0 / (1.0 + damping_ratio),
        "estimated_identity_damping_ratio": damping_ratio,
    }


def _layer_inventory(archive: KimiOutputFactorArchive) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    for segment in archive.manifest.get("segments", []):
        for value in segment.get("layers", []):
            result[int(value["layer"])] = value
    return result


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "minimum": ordered[0],
        "p10": percentile(0.10),
        "median": statistics.median(ordered),
        "p90": percentile(0.90),
        "maximum": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--damping-ratios",
        type=_parse_damping_ratios,
        default=_parse_damping_ratios("0,0.1,1,10,30"),
    )
    args = parser.parse_args()

    archive = KimiOutputFactorArchive(args.archive)
    if archive.manifest.get("complete") is not True:
        raise ValueError("output-factor archive must be complete before analysis")
    device = torch.device(args.device)
    if device.type == "cuda" and (
        device.index is None or device.index >= torch.cuda.device_count()
    ):
        raise ValueError(f"analysis device is unavailable: {device}")
    inventory = _layer_inventory(archive)
    records: list[dict[str, object]] = []
    for layer in archive.expected_layers:
        tensors = load_file(archive.layer_path(layer), device=str(device))
        full = tensors["output_hessian"].float()
        split_a = tensors["output_hessian_split_a"].float()
        split_b = tensors["output_hessian_split_b"].float()
        support = inventory[layer]
        rows_a = int(support["split_a_rows"])
        rows_b = int(support["split_b_rows"])
        reconstructed = (split_a * rows_a + split_b * rows_b) / (rows_a + rows_b)
        full_norm = torch.linalg.vector_norm(full)
        closure = float(
            (torch.linalg.vector_norm(full - reconstructed) / full_norm).item()
        )
        diagonal = torch.diagonal(full)
        frobenius_squared = torch.sum(full * full)
        off_diagonal_fraction = float(
            (1.0 - torch.sum(diagonal * diagonal) / frobenius_squared).item()
        )
        effective_rank = float(
            ((torch.sum(diagonal) ** 2) / frobenius_squared).item()
        )
        cosines = {
            str(ratio): _cosine(_damped(split_a, ratio), _damped(split_b, ratio))
            for ratio in args.damping_ratios
        }
        off_diagonal_cosine, anisotropic_cosine = _component_cosines(
            split_a,
            split_b,
        )
        shrinkage = _anisotropic_shrinkage_estimate(
            split_a,
            split_b,
            left_rows=rows_a,
            right_rows=rows_b,
        )
        records.append(
            {
                "layer": layer,
                "split_a_rows": rows_a,
                "split_b_rows": rows_b,
                "split_cosine_by_damping_ratio": cosines,
                "split_off_diagonal_cosine": off_diagonal_cosine,
                "split_anisotropic_cosine": anisotropic_cosine,
                **shrinkage,
                "split_relative_difference": float(
                    (
                        torch.linalg.vector_norm(split_a - split_b)
                        / torch.sqrt(
                            0.5
                            * (
                                torch.sum(split_a * split_a)
                                + torch.sum(split_b * split_b)
                            )
                        )
                    ).item()
                ),
                "off_diagonal_energy_fraction": off_diagonal_fraction,
                "effective_rank": effective_rank,
                "full_split_closure_relative": closure,
            }
        )
        del tensors, full, split_a, split_b, reconstructed

    damping_summary = {
        str(ratio): _summary(
            [
                float(record["split_cosine_by_damping_ratio"][str(ratio)])
                for record in records
            ]
        )
        for ratio in args.damping_ratios
    }
    result: dict[str, object] = {
        "archive": str(archive.root),
        "archive_manifest": str(archive.manifest_path),
        "archive_manifest_sha256": _sha256(archive.manifest_path),
        "layer_count": len(records),
        "dimension": archive.dimension,
        "damping_ratios": list(args.damping_ratios),
        "split_cosine_by_damping_ratio": damping_summary,
        "split_relative_difference": _summary(
            [float(record["split_relative_difference"]) for record in records]
        ),
        "split_off_diagonal_cosine": _summary(
            [float(record["split_off_diagonal_cosine"]) for record in records]
        ),
        "split_anisotropic_cosine": _summary(
            [float(record["split_anisotropic_cosine"]) for record in records]
        ),
        "estimated_anisotropic_shrinkage": _summary(
            [
                float(record["estimated_anisotropic_shrinkage"])
                for record in records
            ]
        ),
        "estimated_identity_damping_ratio": _summary(
            [
                float(record["estimated_identity_damping_ratio"])
                for record in records
                if record["estimated_identity_damping_ratio"] is not None
            ]
        ),
        "off_diagonal_energy_fraction": _summary(
            [float(record["off_diagonal_energy_fraction"]) for record in records]
        ),
        "effective_rank": _summary(
            [float(record["effective_rank"]) for record in records]
        ),
        "full_split_closure_relative_maximum": max(
            float(record["full_split_closure_relative"]) for record in records
        ),
        "layers": records,
    }
    _atomic_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

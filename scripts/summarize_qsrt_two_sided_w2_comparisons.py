#!/usr/bin/env python3
"""Summarize canonical-W2 BlockLDLQ and BaKron comparison receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Mapping


METRICS: dict[str, tuple[str, ...]] = {
    "optimized_damped_two_sided_work": ("two_sided_work_sse",),
    "undamped_fisher_source": ("two_sided_source_sse",),
    "one_sided_h2": ("input_hessian_sse",),
    "canonical_expert_output": ("canonical_down_expert_output_sse",),
    "canonical_mapped_exact": (
        "canonical_down_routed_output",
        "mapped_exact_sse",
    ),
    "full_expert_output": ("expert_output_sse",),
    "full_mapped_exact": ("routed_output", "mapped_exact_sse"),
}

OPTIONAL_METRICS: dict[str, tuple[str, ...]] = {
    "independent_fisher_source": ("scoring_two_sided_source_sse",),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _nested_float(value: Mapping[str, Any], path: tuple[str, ...]) -> float:
    cursor: Any = value
    for component in path:
        if not isinstance(cursor, Mapping) or component not in cursor:
            raise ValueError(f"result lacks metric {'.'.join(path)}")
        cursor = cursor[component]
    result = float(cursor)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"metric {'.'.join(path)} is invalid: {cursor!r}")
    return result


def _summary(
    experts: Mapping[str, Any],
    metric_path: tuple[str, ...],
) -> dict[str, object]:
    baseline_values: list[float] = []
    candidate_values: list[float] = []
    reductions: list[float] = []
    wins = 0
    per_expert: dict[str, float] = {}
    for expert, result in sorted(experts.items(), key=lambda item: int(item[0])):
        if not isinstance(result, Mapping):
            raise TypeError(f"expert {expert} result is not an object")
        w2 = result.get("w2")
        if not isinstance(w2, Mapping):
            raise ValueError(f"expert {expert} lacks W2 results")
        baseline = w2.get("baseline")
        candidate = w2.get("two_sided")
        if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
            raise ValueError(f"expert {expert} lacks paired W2 candidates")
        baseline_value = _nested_float(baseline, metric_path)
        candidate_value = _nested_float(candidate, metric_path)
        if baseline_value == 0.0:
            if candidate_value != 0.0:
                raise ValueError(f"expert {expert} has a zero baseline metric")
            reduction = 0.0
        else:
            reduction = 100.0 * (1.0 - candidate_value / baseline_value)
        baseline_values.append(baseline_value)
        candidate_values.append(candidate_value)
        reductions.append(reduction)
        per_expert[expert] = reduction
        wins += int(candidate_value < baseline_value)
    pooled_baseline = math.fsum(baseline_values)
    pooled_candidate = math.fsum(candidate_values)
    pooled_reduction = (
        0.0
        if pooled_baseline == 0.0
        else 100.0 * (1.0 - pooled_candidate / pooled_baseline)
    )
    return {
        "direction": "positive reduction is better",
        "pooled_baseline": pooled_baseline,
        "pooled_candidate": pooled_candidate,
        "pooled_reduction_percent": pooled_reduction,
        "median_expert_reduction_percent": statistics.median(reductions),
        "minimum_expert_reduction_percent": min(reductions),
        "maximum_expert_reduction_percent": max(reductions),
        "wins": wins,
        "experts": len(reductions),
        "per_expert_reduction_percent": per_expert,
    }


def _receipt(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if value.get("complete") is not True:
        raise ValueError(f"comparison receipt is incomplete: {path}")
    signature = value.get("signature")
    experts = value.get("experts")
    if not isinstance(signature, Mapping) or not isinstance(experts, Mapping):
        raise ValueError(
            f"comparison receipt has no signature or expert results: {path}"
        )
    if signature.get("kind") != "qsrt_uniform_k2_two_sided_w2_pilot":
        raise ValueError(f"comparison receipt has the wrong kind: {path}")
    damping_ratio = float(signature["output_damping_ratio"])
    metrics = {
        name: _summary(experts, metric_path)
        for name, metric_path in METRICS.items()
    }
    first_expert = next(iter(experts.values()))
    first_baseline = first_expert["w2"]["baseline"]
    for name, metric_path in OPTIONAL_METRICS.items():
        if metric_path[0] in first_baseline:
            metrics[name] = _summary(experts, metric_path)
    coordinate_closure = None
    if damping_ratio == 0.0:
        coordinate_closure = abs(
            float(
                metrics["optimized_damped_two_sided_work"][
                    "pooled_reduction_percent"
                ]
            )
            - float(
                metrics["undamped_fisher_source"][
                    "pooled_reduction_percent"
                ]
            )
        )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "layer": int(signature["layer"]),
        "expert_ids": sorted(int(expert) for expert in experts),
        "output_damping_ratio": damping_ratio,
        "output_hessian_key": signature.get("output_hessian_key"),
        "scoring_output_hessian_key": signature.get(
            "scoring_output_hessian_key"
        ),
        "output_factor_archive": signature.get("output_factor_archive"),
        "output_factor_archive_manifest_sha256": signature.get(
            "output_factor_archive_manifest_sha256"
        ),
        "candidate_pool_manifest_sha256": signature.get(
            "candidate_pool_manifest_sha256"
        ),
        "fit_cache_manifest_sha256": signature.get("fit_cache_manifest_sha256"),
        "evaluation_manifest_sha256": signature.get(
            "evaluation_manifest_sha256"
        ),
        "seconds": float(value["seconds"]),
        "zero_damping_coordinate_reduction_closure_percent_points": (
            coordinate_closure
        ),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [_receipt(path.expanduser().resolve()) for path in args.receipts]
    reference = records[0]
    invariant_keys = (
        "layer",
        "expert_ids",
        "candidate_pool_manifest_sha256",
        "fit_cache_manifest_sha256",
        "evaluation_manifest_sha256",
    )
    for record in records[1:]:
        for key in invariant_keys:
            if record[key] != reference[key]:
                raise ValueError(f"comparison receipts disagree on {key}")
    result = {
        "kind": "qsrt_uniform_k2_two_sided_w2_comparison_summary",
        "schema_version": 1,
        "metric_direction": "positive reduction is better",
        "receipts": records,
    }
    if args.output is not None:
        _atomic_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

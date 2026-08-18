#!/usr/bin/env python3
"""Aggregate per-expert uniform-K2 distortion-transfer receipts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from itertools import combinations
from pathlib import Path


STAGES = (
    "raw_weight",
    "gate_preactivation",
    "up_preactivation",
    "post_situ",
    "expert_output",
    "routed_expert",
    "mapped_linear",
    "mapped_exact",
)
MATRICES = ("w1", "w3", "w2")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _hybrid_key(selected: frozenset[int]) -> str:
    return "".join("s" if index in selected else "m" for index in range(3))


def _shapley(hybrid_sse: dict[str, float]) -> dict[str, float]:
    result = {matrix: 0.0 for matrix in MATRICES}
    factorial = math.factorial
    for index, matrix in enumerate(MATRICES):
        others = tuple(item for item in range(3) if item != index)
        for count in range(3):
            weight = factorial(count) * factorial(2 - count) / factorial(3)
            for subset_tuple in combinations(others, count):
                subset = frozenset(subset_tuple)
                before = hybrid_sse[_hybrid_key(subset)]
                after = hybrid_sse[_hybrid_key(subset | {index})]
                result[matrix] += weight * (before - after)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(
        path
        for path in args.receipts.glob("layer-*.json")
        if ".failed-" not in path.name
    )
    receipts = [json.loads(path.read_text()) for path in paths]
    if not receipts or not all(bool(value.get("complete")) for value in receipts):
        raise ValueError("receipts must contain at least one complete expert result")

    stages: dict[str, object] = {}
    for stage in STAGES:
        mcg = sum(value["sqg_vs_mcg"][stage]["mcg_sse"] for value in receipts)
        sqg = sum(value["sqg_vs_mcg"][stage]["sqg_sse"] for value in receipts)
        reductions = [
            value["sqg_vs_mcg"][stage]["sqg_relative_reduction"]
            for value in receipts
        ]
        stages[stage] = {
            "mcg_pooled_sse": mcg,
            "sqg_pooled_sse": sqg,
            "sqg_pooled_relative_reduction": 1.0 - sqg / mcg,
            "median_expert_relative_reduction": statistics.median(reductions),
            "minimum_expert_relative_reduction": min(reductions),
            "maximum_expert_relative_reduction": max(reductions),
            "sqg_wins": sum(value > 0 for value in reductions),
            "experts": len(reductions),
        }

    dense: dict[str, object] = {}
    for matrix in MATRICES:
        field = (
            "source_hidden_covariance_sse"
            if matrix == "w2"
            else "encoding_covariance_sse"
        )
        mcg = sum(
            value["candidates"]["mcg"]["matrices"][matrix][field]
            for value in receipts
        )
        sqg = sum(
            value["candidates"]["sqg_xor_cheb_t12"]["matrices"][matrix][field]
            for value in receipts
        )
        dense[matrix] = {
            "metric": (
                "shared source-hidden covariance"
                if matrix == "w2"
                else "shared layer-global input covariance"
            ),
            "mcg_pooled_sse": mcg,
            "sqg_pooled_sse": sqg,
            "sqg_pooled_relative_reduction": 1.0 - sqg / mcg,
        }

    hybrid_sse = {
        key: sum(
            value["hybrids"][key]["residual_geometry"]["mapped_exact_sse"]
            for value in receipts
        )
        for key in ("mmm", "mms", "msm", "mss", "smm", "sms", "ssm", "sss")
    }
    shapley = _shapley(hybrid_sse)
    total_improvement = hybrid_sse["mmm"] - hybrid_sse["sss"]
    summary = {
        "kind": "qsrt_uniform_k2_distortion_transfer_summary",
        "schema_version": 1,
        "receipts": [str(path.resolve()) for path in paths],
        "experts": [
            {
                "layer": value["signature"]["layer"],
                "expert": value["signature"]["expert"],
                "fit_occurrences": value["fit_support"]["sampled_occurrences"],
                "evaluation_occurrences": value["evaluation_support"]["occurrences"],
                "evaluation_documents": value["evaluation_support"]["documents"],
            }
            for value in receipts
        ],
        "stages": stages,
        "common_dense_covariance": dense,
        "mapped_exact_hybrid_sse": hybrid_sse,
        "mapped_exact_shapley_attribution": {
            matrix: {
                "absolute_sse_reduction": amount,
                "share_of_sqg_improvement": amount / total_improvement,
            }
            for matrix, amount in shapley.items()
        },
        "scope": {
            "full_model_kld": False,
            "natural_downstream_propagation": False,
            "checkpoint_materialized": False,
            "conclusion": (
                "diagnostic expert and immediate downstream attribution only"
            ),
        },
    }
    _atomic_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

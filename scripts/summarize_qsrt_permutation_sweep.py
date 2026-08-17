#!/usr/bin/env python3
"""Compare tile-funded neuron-permutation policies on absolute expert SSE."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


CONTRACT_KEYS = (
    "experiment_implementation_sha256",
    "capture",
    "sample_cache",
    "training_report",
    "hessians",
    "codebook",
    "codebook_sha256",
    "official_revision",
    "ldlq_tf32",
    "w2_hessian",
    "allocation_search",
    "allocation_coordinates",
)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} does not contain a JSON object")
    if not value.get("complete"):
        raise ValueError(f"{path} is not complete")
    return value


def _contract_identity(contract: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(
        json.dumps(contract.get(key), sort_keys=True)
        if isinstance(contract.get(key), (dict, list))
        else contract.get(key)
        for key in CONTRACT_KEYS
    )


def _absolute_scores(result: Mapping[str, object]) -> tuple[float, float, str]:
    qsrt = result.get("qsrt_308")
    if not isinstance(qsrt, Mapping):
        raise ValueError("result does not contain a qsrt_308 experiment")
    serial = qsrt.get("serial_validation")
    if not isinstance(serial, Mapping):
        raise ValueError("result lacks isolated serial candidate closure")
    selected_name = serial.get("selected_on_serial_fit")
    selected = serial.get("serial_fit_selected")
    if not isinstance(selected_name, str) or not isinstance(selected, Mapping):
        raise ValueError("result lacks a fit-selected serial re-encode")
    fit = selected.get("fit")
    confirmation = selected.get("confirmation")
    if not isinstance(fit, Mapping) or not isinstance(confirmation, Mapping):
        raise ValueError("serial candidate lacks fit or confirmation totals")
    fit_sse = float(fit["sse"])
    confirmation_sse = float(confirmation["sse"])
    if fit_sse < 0 or confirmation_sse < 0:
        raise ValueError("SSE must be nonnegative")
    permutation = result.get("permutation_sha256")
    if not isinstance(permutation, str) or len(permutation) != 64:
        raise ValueError("result lacks a frozen permutation identity")
    return fit_sse, confirmation_sse, permutation


def summarize(
    paths: Sequence[Path],
    *,
    baseline: str = "h2_reverse",
    selection_experts: frozenset[tuple[int, int]] | None = None,
) -> dict:
    rows: dict[tuple[int, int], dict[str, dict[str, object]]] = defaultdict(dict)
    contract_identity: tuple[object, ...] | None = None
    for path in paths:
        payload = _read(path)
        contract = payload.get("contract")
        if not isinstance(contract, Mapping):
            raise ValueError(f"{path} lacks an experiment contract")
        identity = _contract_identity(contract)
        if contract_identity is None:
            contract_identity = identity
        elif identity != contract_identity:
            raise ValueError(f"{path} does not match the sweep contract")
        layer = int(contract["layer"])
        policy = contract.get("permutation_policy")
        if not isinstance(policy, str):
            raise ValueError(f"{path} lacks a permutation policy")
        results = payload.get("results")
        if not isinstance(results, Mapping):
            raise ValueError(f"{path} lacks expert results")
        for expert_text, result in results.items():
            if not isinstance(result, Mapping) or result.get("skipped"):
                raise ValueError(f"{path} has a skipped or malformed expert result")
            expert = int(expert_text)
            fit_sse, confirmation_sse, permutation = _absolute_scores(result)
            key = (layer, expert)
            if policy in rows[key]:
                raise ValueError(f"duplicate {policy} result for layer {layer} expert {expert}")
            rows[key][policy] = {
                "fit_sse": fit_sse,
                "confirmation_sse": confirmation_sse,
                "permutation_sha256": permutation,
                "source": str(path.resolve()),
            }
    if not rows:
        raise ValueError("the sweep contains no expert results")
    policy_sets = {frozenset(value) for value in rows.values()}
    if len(policy_sets) != 1:
        raise ValueError("every expert must have the same permutation policy set")
    policies = tuple(sorted(next(iter(policy_sets))))
    if baseline not in policies:
        raise ValueError(f"baseline policy {baseline!r} is absent")
    all_experts = frozenset(rows)
    if selection_experts is None:
        selection_experts = all_experts
        evaluation_experts = all_experts
        expert_partition = "shared_experts_document_disjoint"
    else:
        missing = selection_experts - all_experts
        if missing:
            raise ValueError(f"selection experts are absent: {sorted(missing)}")
        evaluation_experts = all_experts - selection_experts
        if not selection_experts or not evaluation_experts:
            raise ValueError("expert-disjoint selection needs two nonempty partitions")
        expert_partition = "expert_and_document_disjoint"

    pooled = {}
    for policy in policies:
        pooled[policy] = {
            "fit_sse": sum(float(value[policy]["fit_sse"]) for value in rows.values()),
            "confirmation_sse": sum(
                float(value[policy]["confirmation_sse"]) for value in rows.values()
            ),
        }
    selected_policy = min(
        policies,
        key=lambda policy: sum(
            float(rows[key][policy]["fit_sse"]) for key in selection_experts
        ),
    )
    selected_evaluation_confirmation = sum(
        float(rows[key][selected_policy]["confirmation_sse"])
        for key in evaluation_experts
    )
    baseline_evaluation_confirmation = sum(
        float(rows[key][baseline]["confirmation_sse"]) for key in evaluation_experts
    )

    per_expert = {}
    fit_selected_confirmation = 0.0
    baseline_per_expert_confirmation = 0.0
    for (layer, expert), value in sorted(rows.items()):
        selected = min(policies, key=lambda policy: float(value[policy]["fit_sse"]))
        selected_confirmation = float(value[selected]["confirmation_sse"])
        local_baseline = float(value[baseline]["confirmation_sse"])
        fit_selected_confirmation += selected_confirmation
        baseline_per_expert_confirmation += local_baseline
        per_expert[f"{layer}/{expert}"] = {
            "selected_on_fit": selected,
            "confirmation_sse": selected_confirmation,
            "confirmation_relative_to_baseline": selected_confirmation / local_baseline - 1.0,
            "policies": value,
        }

    return {
        "kind": "qsrt_permutation_sweep_summary",
        "schema_version": 1,
        "comparison": "absolute_fit_selected_serial_reencode_sse",
        "selection_partition": "fit",
        "evaluation_partition": "document_disjoint_confirmation",
        "expert_partition": expert_partition,
        "selection_experts": [f"{layer}/{expert}" for layer, expert in sorted(selection_experts)],
        "evaluation_experts": [
            f"{layer}/{expert}" for layer, expert in sorted(evaluation_experts)
        ],
        "contract_identity": dict(
            zip(CONTRACT_KEYS, contract_identity or (), strict=True)
        ),
        "policies": list(policies),
        "baseline_policy": baseline,
        "pooled": pooled,
        "pooled_selected_on_fit": selected_policy,
        "pooled_selected_confirmation_relative_to_baseline": (
            selected_evaluation_confirmation / baseline_evaluation_confirmation - 1.0
        ),
        "per_expert_fit_selected_confirmation_relative_to_baseline": (
            fit_selected_confirmation / baseline_per_expert_confirmation - 1.0
        ),
        "experts": per_expert,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--baseline", default="h2_reverse")
    parser.add_argument(
        "--selection-experts",
        help=(
            "comma-separated layer/expert keys used to choose one global policy; "
            "all remaining experts form the evaluation set"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection_experts = None
    if args.selection_experts:
        selection_experts = frozenset(
            tuple(map(int, item.split("/")))
            for item in args.selection_experts.split(",")
        )
        if any(len(item) != 2 for item in selection_experts):
            raise ValueError("selection experts must use layer/expert syntax")
    payload = summarize(
        args.inputs,
        baseline=args.baseline,
        selection_experts=selection_experts,
    )
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()

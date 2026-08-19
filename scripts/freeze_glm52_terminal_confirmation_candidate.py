#!/usr/bin/env python3
"""Freeze a screened GLM-5.2 correction before confirmation access.

The script accepts the layer-3 expert-103 candidate only when its eight
document screening mean improves, at least six documents improve, and pooled
CVaR1% does not regress. It extracts the registered BF16 factors into a
standalone safetensors payload, charges every serialized byte in that payload,
and binds the payload, runtime mode, screening report, and teacher-reference
plan into the authorization required for confirmation-logit generation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from qsrt.correctness import sha256_file
from qsrt.glm52_document_disjoint_confirmation import (
    validate_frozen_low_rank_candidate,
)
from qsrt.glm52_expert_intervention_runtime import (
    MATERIALIZED_LOW_RANK_CANDIDATE_MODE,
    validate_intervention_artifact,
)
from qsrt.glm52_pilot import atomic_write_json
from qsrt.glm52_terminal_teacher_reference import (
    validate_terminal_teacher_reference_plan,
)
from qsrt.qsrt_codec_pilot import tensor_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervention-artifact", type=Path, required=True)
    parser.add_argument("--candidate-registration", type=Path, required=True)
    parser.add_argument("--screening-report", type=Path, required=True)
    parser.add_argument("--teacher-reference-plan", type=Path, required=True)
    parser.add_argument("--frozen-at-utc", required=True)
    parser.add_argument("--dest", type=Path, required=True)
    return parser


def _screening_gate(
    report: dict[str, Any],
    *,
    manifest_sha256: str,
    teacher_reference_plan_sha256: str,
) -> dict[str, Any]:
    if (
        report.get("schema")
        != "qsrt_glm52_document_disjoint_candidate_evaluation"
        or report.get("status") != "complete"
        or report.get("evaluation_tier") != "screening"
        or report.get("measurement_controls", {}).get("passed") is not True
        or report.get("intervention_artifact", {}).get("manifest_sha256")
        != manifest_sha256
        or report.get("teacher_references", {}).get("plan_sha256")
        != teacher_reference_plan_sha256
    ):
        raise ValueError("terminal-reference screening report identity differs")
    summary = report.get("summary")
    if not isinstance(summary, dict) or summary.get("document_count") != 8:
        raise ValueError("terminal-reference screening requires eight documents")
    equal_document = summary.get("equal_document_weight")
    outcomes = summary.get("document_outcomes")
    tails = summary.get("tail_metrics")
    if (
        not isinstance(equal_document, dict)
        or not isinstance(outcomes, dict)
        or not isinstance(tails, dict)
        or not isinstance(tails.get("baseline"), dict)
        or not isinstance(tails.get("candidate"), dict)
    ):
        raise ValueError("terminal-reference screening statistics are incomplete")
    mean_difference = float(
        equal_document["candidate_minus_baseline_mean_forward_kld"]
    )
    improved_documents = int(outcomes["candidate_better"])
    equal_documents = int(outcomes["candidate_equal"])
    regressed_documents = int(outcomes["candidate_worse"])
    baseline_cvar1 = float(tails["baseline"]["cvar1"])
    candidate_cvar1 = float(tails["candidate"]["cvar1"])
    if (
        not all(
            math.isfinite(value)
            for value in (mean_difference, baseline_cvar1, candidate_cvar1)
        )
        or min(improved_documents, equal_documents, regressed_documents) < 0
        or improved_documents + equal_documents + regressed_documents != 8
    ):
        raise ValueError("terminal-reference screening statistics are invalid")
    passed = (
        mean_difference < 0.0
        and improved_documents >= 6
        and candidate_cvar1 <= baseline_cvar1
    )
    if not passed:
        raise ValueError(
            "candidate failed the pre-registered mean, document-sign, or CVaR screen"
        )
    return {
        "equal_document_mean_candidate_minus_resident_forward_kld": mean_difference,
        "improved_document_count": improved_documents,
        "required_improved_document_count": 6,
        "baseline_pooled_cvar1": baseline_cvar1,
        "candidate_pooled_cvar1": candidate_cvar1,
        "cvar_rule": "candidate pooled CVaR1% must not exceed resident CVaR1%",
        "screening_false_pass_probability_for_six_of_eight_signs_under_null": (
            37 / 256
        ),
        "passed": True,
    }


def _validated_frozen_at_utc(value: str) -> str:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("frozen-at time must be an ISO-8601 UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError("frozen-at time must be an ISO-8601 UTC timestamp")
    return parsed.isoformat()


def _write_factor_payload(
    *, artifact: dict[str, Any], frozen: dict[str, Any], destination: Path
) -> dict[str, Any]:
    records = artifact["report"]["experts"]
    matches = [record for record in records if record["expert"] == frozen["expert"]]
    if len(matches) != 1:
        raise ValueError("registered expert is not unique in the intervention")
    record = matches[0]
    source_path = Path(artifact["root"]) / "experts" / record["dense_endpoint_file"]
    with safe_open(source_path, framework="pt", device="cpu") as handle:
        factor_a = handle.get_tensor("adapter.down.a").contiguous()
        factor_b = handle.get_tensor("adapter.down.b").contiguous()
    if (
        factor_a.dtype != torch.bfloat16
        or factor_b.dtype != torch.bfloat16
        or list(factor_a.shape) != [2048, frozen["rank"]]
        or list(factor_b.shape) != [6144, frozen["rank"]]
        or tensor_sha256(factor_a) != frozen["factor_a_sha256"]
        or tensor_sha256(factor_b) != frozen["factor_b_sha256"]
    ):
        raise ValueError("registered BF16 factors differ from the runtime artifact")

    output_path = destination / "layer-003-expert-103-down-rank-4-bf16.safetensors"
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.partial")
    metadata = {
        "schema": "qsrt_glm52_bf16_low_rank_down_correction",
        "model_layer": str(frozen["layer"]),
        "expert": str(frozen["expert"]),
        "matrix": "down_proj",
        "rank": str(frozen["rank"]),
        "factor_dtype": "BF16",
        "artifact_manifest_sha256": artifact["manifest_sha256"],
    }
    try:
        save_file(
            {"factor_a": factor_a, "factor_b": factor_b},
            str(temporary),
            metadata=metadata,
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": output_path.name,
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "metadata": metadata,
        "factor_a_shape": list(factor_a.shape),
        "factor_b_shape": list(factor_b.shape),
        "factor_a_sha256": tensor_sha256(factor_a),
        "factor_b_sha256": tensor_sha256(factor_b),
        "logical_factor_bytes": sum(
            tensor.numel() * tensor.element_size()
            for tensor in (factor_a, factor_b)
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.dest.exists():
        raise FileExistsError(args.dest)
    artifact = validate_intervention_artifact(args.intervention_artifact)
    if artifact["artifact_kind"] != "single_layer" or artifact["model_layer"] != 3:
        raise ValueError("the registered confirmation candidate must be at layer 3")
    registration = json.loads(args.candidate_registration.read_text())
    frozen = validate_frozen_low_rank_candidate(registration, artifact)
    if (frozen["expert"], frozen["rank"], frozen["factor_dtype"]) != (
        103,
        4,
        "BF16",
    ):
        raise ValueError("confirmation registration is not the frozen expert-103 arm")
    screening_report = json.loads(args.screening_report.read_text())
    teacher_reference_plan = json.loads(args.teacher_reference_plan.read_text())
    validate_terminal_teacher_reference_plan(teacher_reference_plan)
    teacher_reference_plan_sha256 = sha256_file(args.teacher_reference_plan)
    screening_gate = _screening_gate(
        screening_report,
        manifest_sha256=artifact["manifest_sha256"],
        teacher_reference_plan_sha256=teacher_reference_plan_sha256,
    )
    frozen_at_utc = _validated_frozen_at_utc(args.frozen_at_utc)

    args.dest.mkdir(parents=True)
    payload = _write_factor_payload(
        artifact=artifact, frozen=frozen, destination=args.dest
    )
    manifest = {
        "schema": "qsrt_glm52_frozen_bf16_low_rank_down_correction",
        "schema_version": 1,
        "status": "frozen_before_confirmation_reference_access",
        "candidate": frozen,
        "candidate_runtime_mode": MATERIALIZED_LOW_RANK_CANDIDATE_MODE,
        "intervention_artifact_manifest_sha256": artifact["manifest_sha256"],
        "candidate_registration_sha256": sha256_file(
            args.candidate_registration
        ),
        "screening_report_sha256": sha256_file(args.screening_report),
        "teacher_reference_plan_sha256": teacher_reference_plan_sha256,
        "screening_gate": screening_gate,
        "serialized_correction": payload,
        "byte_accounting": {
            "total_charged_bytes": payload["bytes"],
            "scope": (
                "every byte in the standalone BF16 factor payload, including its "
                "safetensors header and metadata"
            ),
            "complete_checkpoint_size_status": (
                "unresolved until the GLM-native QSRT container is materialized"
            ),
        },
    }
    atomic_write_json(args.dest / "manifest.json", manifest)
    freeze = {
        "schema": "qsrt_glm52_terminal_reference_confirmation_freeze",
        "schema_version": 1,
        "status": "frozen_before_confirmation_reference_access",
        "artifact_manifest_sha256": artifact["manifest_sha256"],
        "candidate_runtime_mode": MATERIALIZED_LOW_RANK_CANDIDATE_MODE,
        "teacher_reference_plan_sha256": teacher_reference_plan_sha256,
        "screening_report_sha256": sha256_file(args.screening_report),
        "total_charged_bytes": payload["bytes"],
        "frozen_at_utc": frozen_at_utc,
        "confirmation_quality_gate": {
            "minimum_document_count": 32,
            "sampling_unit": "document",
            "mean_rule": (
                "paired document-bootstrap one-sided 95% upper bound is below zero"
            ),
            "tail_metric": "pooled position CVaR1%",
            "maximum_absolute_tail_increase": 0.0,
            "tail_rule": (
                "candidate pooled position CVaR1% must not exceed resident CVaR1%"
            ),
        },
        "frozen_candidate": {
            **frozen,
            "serialized_correction_file": payload["path"],
            "serialized_correction_sha256": payload["sha256"],
            "serialized_correction_bytes": payload["bytes"],
        },
    }
    atomic_write_json(args.dest / "confirmation-freeze.json", freeze)
    print(json.dumps({"manifest": manifest, "freeze": freeze}, indent=2))


if __name__ == "__main__":
    main()

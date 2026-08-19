from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from qsrt.glm52_document_disjoint_confirmation import (
    validate_terminal_reference_confirmation_freeze,
)
from qsrt.glm52_expert_intervention import INTERVENTION_ARTIFACT_KIND
from qsrt.glm52_expert_intervention_runtime import (
    MATERIALIZED_LOW_RANK_CANDIDATE_MODE,
    validate_intervention_artifact,
)
from qsrt.qsrt_codec_pilot import tensor_sha256
from scripts.freeze_glm52_terminal_confirmation_candidate import (
    _screening_gate,
    main,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PLAN = (
    ROOT / "experiments/glm52_terminal_hidden_teacher_reference_plan.json"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_factor_artifact(root: Path) -> tuple[dict, dict]:
    expert_root = root / "experts"
    expert_root.mkdir(parents=True)
    factor_a = torch.arange(2048 * 4, dtype=torch.float32).reshape(2048, 4)
    factor_a = (factor_a / 8192).to(torch.bfloat16)
    factor_b = torch.arange(6144 * 4, dtype=torch.float32).reshape(6144, 4)
    factor_b = (factor_b / 24576).to(torch.bfloat16)
    endpoint = expert_root / "layer-003-expert-103.safetensors"
    save_file(
        {"adapter.down.a": factor_a, "adapter.down.b": factor_b},
        str(endpoint),
    )
    manifest = {
        "kind": f"{INTERVENTION_ARTIFACT_KIND}_manifest",
        "candidate": {"tensor_prefix": "candidate"},
        "exl3_endpoint_identity": {"layer": 3},
        "panel": {"3": [103]},
    }
    manifest_sha256 = _canonical_sha256(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n")
    record = {
        "layer": 3,
        "expert": 103,
        "dense_endpoint_file": endpoint.name,
        "dense_endpoint_file_bytes": endpoint.stat().st_size,
        "dense_endpoint_file_sha256": hashlib.sha256(
            endpoint.read_bytes()
        ).hexdigest(),
        "rank": 4,
        "factor_dtype": "BF16",
        "selected_ridge_factor": 0.001,
        "factor_a_sha256": tensor_sha256(factor_a),
        "factor_b_sha256": tensor_sha256(factor_b),
        "logical_adapter_bytes": 65536,
        "materialized_down_sha256": "c" * 64,
    }
    report = {
        "kind": INTERVENTION_ARTIFACT_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "layer": 3,
        "expert_count": 1,
        "dense_endpoint_bytes": endpoint.stat().st_size,
        "experts": [record],
    }
    (root / "report.json").write_text(json.dumps(report) + "\n")
    return report, record


def _screening_report(*, manifest_sha256: str, plan_sha256: str) -> dict:
    return {
        "schema": "qsrt_glm52_document_disjoint_candidate_evaluation",
        "schema_version": 1,
        "status": "complete",
        "evaluation_tier": "screening",
        "measurement_controls": {"passed": True},
        "intervention_artifact": {"manifest_sha256": manifest_sha256},
        "teacher_references": {"plan_sha256": plan_sha256},
        "summary": {
            "document_count": 8,
            "equal_document_weight": {
                "candidate_minus_baseline_mean_forward_kld": -0.001
            },
            "document_outcomes": {
                "candidate_better": 6,
                "candidate_equal": 1,
                "candidate_worse": 1,
            },
            "tail_metrics": {
                "baseline": {"cvar1": 0.2},
                "candidate": {"cvar1": 0.19},
            },
        },
    }


def test_screening_gate_rejects_a_tail_regression() -> None:
    report = _screening_report(
        manifest_sha256="a" * 64, plan_sha256="b" * 64
    )
    report["summary"]["tail_metrics"]["candidate"]["cvar1"] = 0.21

    with pytest.raises(ValueError, match="failed the pre-registered"):
        _screening_gate(
            report,
            manifest_sha256="a" * 64,
            teacher_reference_plan_sha256="b" * 64,
        )


def test_freeze_extracts_the_registered_factors_and_authorizes_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifact"
    report, record = _write_factor_artifact(artifact_root)
    artifact = validate_intervention_artifact(artifact_root)
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(
        json.dumps(
            {
                "schema": "qsrt_glm52_low_rank_down_confirmation_registration",
                "schema_version": 1,
                "status": "frozen_before_document_disjoint_confirmation",
                "frozen_correction": {
                    "layer": 3,
                    "expert": 103,
                    "rank": 4,
                    "factor_dtype": "BF16",
                    "selected_ridge_factor": 0.001,
                    "factor_a_sha256": record["factor_a_sha256"],
                    "factor_b_sha256": record["factor_b_sha256"],
                    "logical_factor_bytes": 65536,
                    "materialized_down_sha256": "c" * 64,
                },
            }
        )
        + "\n"
    )
    plan_sha256 = hashlib.sha256(REFERENCE_PLAN.read_bytes()).hexdigest()
    screening_path = tmp_path / "screening-report.json"
    screening_path.write_text(
        json.dumps(
            _screening_report(
                manifest_sha256=report["manifest_sha256"],
                plan_sha256=plan_sha256,
            )
        )
        + "\n"
    )
    destination = tmp_path / "frozen-candidate"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_glm52_terminal_confirmation_candidate.py",
            "--intervention-artifact",
            str(artifact_root),
            "--candidate-registration",
            str(registration_path),
            "--screening-report",
            str(screening_path),
            "--teacher-reference-plan",
            str(REFERENCE_PLAN),
            "--frozen-at-utc",
            "2026-08-19T05:30:00Z",
            "--dest",
            str(destination),
        ],
    )

    main()

    freeze_path = destination / "confirmation-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    payload = destination / freeze["frozen_candidate"]["serialized_correction_file"]
    assert freeze["frozen_at_utc"] == "2026-08-19T05:30:00+00:00"
    assert freeze["total_charged_bytes"] == payload.stat().st_size
    assert freeze["total_charged_bytes"] >= 65536
    with safe_open(payload, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == {"factor_a", "factor_b"}
        assert tensor_sha256(handle.get_tensor("factor_a")) == record[
            "factor_a_sha256"
        ]
        assert tensor_sha256(handle.get_tensor("factor_b")) == record[
            "factor_b_sha256"
        ]
    authorized = validate_terminal_reference_confirmation_freeze(
        freeze,
        artifact=artifact,
        candidate_runtime_mode=MATERIALIZED_LOW_RANK_CANDIDATE_MODE,
        teacher_reference_plan_sha256=plan_sha256,
        screening_report_path=screening_path,
    )
    assert authorized["total_charged_bytes"] == payload.stat().st_size

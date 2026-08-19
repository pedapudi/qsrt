from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from qsrt.glm52_model_kld_candidate_selection import (
    SELECTION_PLAN_SCHEMA,
    summarize_selection_document_groups,
    validate_candidate_subset_selection_plan,
)


def _canonical_json_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def test_error_blind_plan_builder_declares_singletons_and_complete_panel(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    experts = [130, 74, 172, 83, 249, 84, 214, 232]
    manifest = {
        "kind": "qsrt_glm52_dense_expert_intervention_v1_manifest",
        "panel": {"55": experts},
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    (artifact / "report.json").write_text(
        json.dumps(
            {
                "kind": "qsrt_glm52_dense_expert_intervention_v1",
                "status": "complete",
                "manifest_sha256": _canonical_json_sha256(manifest),
                "layer": 55,
                "experts": [{"expert": expert} for expert in experts],
            }
        )
    )
    destination = tmp_path / "selection-plan.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_glm52_error_blind_candidate_subset_plan.py",
            "--artifact",
            str(artifact),
            "--dest",
            str(destination),
            "--candidate-description",
            "uniform K3",
            "--include-complete-panel",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    plan = json.loads(destination.read_text())
    arms = validate_candidate_subset_selection_plan(
        plan,
        artifact={
            "manifest_sha256": _canonical_json_sha256(manifest),
            "model_layer": 55,
            "expert_ids": experts,
        },
    )
    assert [arm["selected_experts"] for arm in arms[:-1]] == [
        [expert] for expert in sorted(experts)
    ]
    assert arms[-1]["selected_experts"] == sorted(experts)
    assert all("error" not in arm for arm in plan["candidate_arms"])


def _artifact() -> dict[str, object]:
    return {
        "manifest_sha256": "a" * 64,
        "model_layer": 63,
        "expert_ids": [29, 118, 123, 164, 199, 215],
    }


def _plan() -> dict[str, object]:
    return {
        "schema": SELECTION_PLAN_SCHEMA,
        "schema_version": 1,
        "status": "frozen_before_candidate_subset_kld_measurement",
        "artifact_manifest_sha256": "a" * 64,
        "model_layer": 63,
        "selection_protocol": {
            "document_order_source": "selected_chunks in public-reference plan order",
            "screening_document_count": 8,
            "selection_check_document_count": 8,
        },
        "candidate_arms": [
            {
                "name": "accepted-experts-without-164",
                "selected_experts": [215, 29, 118],
                "reason": "Test whether the other accepted experts offset expert 164.",
            },
            {
                "name": "expert-123-alone",
                "selected_experts": [123],
                "reason": "Measure one expert without assuming additive effects.",
            },
        ],
    }


def test_candidate_subset_selection_plan_normalizes_expert_order() -> None:
    arms = validate_candidate_subset_selection_plan(_plan(), artifact=_artifact())
    assert arms[0]["selected_experts"] == [29, 118, 215]
    assert arms[1]["selected_experts"] == [123]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "draft", "not frozen"),
        ("artifact_manifest_sha256", "b" * 64, "artifact identity"),
        ("model_layer", 64, "layer mismatch"),
    ],
)
def test_candidate_subset_selection_plan_closes_identity(
    field: str, value: object, message: str
) -> None:
    plan = _plan()
    plan[field] = value
    with pytest.raises(ValueError, match=message):
        validate_candidate_subset_selection_plan(plan, artifact=_artifact())


def test_candidate_subset_selection_plan_rejects_duplicate_subsets() -> None:
    plan = _plan()
    plan["candidate_arms"][1]["selected_experts"] = [118, 29, 215]  # type: ignore[index]
    with pytest.raises(ValueError, match="distinct expert subsets"):
        validate_candidate_subset_selection_plan(plan, artifact=_artifact())


def test_candidate_subset_selection_plan_rejects_unknown_expert() -> None:
    plan = copy.deepcopy(_plan())
    plan["candidate_arms"][0]["selected_experts"] = [254]  # type: ignore[index]
    with pytest.raises(ValueError, match="outside the artifact"):
        validate_candidate_subset_selection_plan(plan, artifact=_artifact())


def test_candidate_subset_selection_plan_rejects_unsafe_name() -> None:
    plan = copy.deepcopy(_plan())
    plan["candidate_arms"][0]["name"] = "Expert 29"  # type: ignore[index]
    with pytest.raises(ValueError, match="path-safe lowercase"):
        validate_candidate_subset_selection_plan(plan, artifact=_artifact())


def _selection_report() -> dict[str, object]:
    arms: dict[str, object] = {}
    documents: list[dict[str, object]] = [
        {"candidate_arms": {}} for _ in range(16)
    ]
    for arm_index, arm in enumerate(_plan()["candidate_arms"]):  # type: ignore[index]
        first_delta = -0.2 if arm_index == 0 else -0.1
        second_delta = -0.1 if arm_index == 0 else 0.2
        arms[arm["name"]] = {  # type: ignore[index]
            "selected_experts": arm["selected_experts"],  # type: ignore[index]
            "summary": {
                # Production summaries sort documents by identity. The frozen
                # two-group decision must use the top-level execution order.
                "per_document": [
                    {"candidate_minus_baseline_mean_forward_kld": second_delta}
                    for _ in range(8)
                ]
                + [
                    {"candidate_minus_baseline_mean_forward_kld": first_delta}
                    for _ in range(8)
                ]
            },
        }
        for document_index, document in enumerate(documents):
            delta = first_delta if document_index < 8 else second_delta
            document["candidate_arms"][arm["name"]] = {  # type: ignore[index]
                "candidate_minus_resident_mean_forward_kld": delta
            }
    return {
        "schema": "qsrt_glm52_document_disjoint_model_kld_candidate_selection",
        "schema_version": 1,
        "status": "complete",
        "candidate_arms": arms,
        "documents": documents,
    }


def test_selection_document_groups_require_improvement_in_both_halves() -> None:
    summaries = summarize_selection_document_groups(_selection_report(), plan=_plan())
    assert summaries[0] == {
        "name": "accepted-experts-without-164",
        "selected_experts": [215, 29, 118],
        "screening_document_mean_delta": pytest.approx(-0.2),
        "selection_check_document_mean_delta": pytest.approx(-0.1),
        "all_document_mean_delta": pytest.approx(-0.15),
        "retained": True,
    }
    assert summaries[1]["retained"] is False


def test_selection_summary_script_records_retained_arms(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    report_path = tmp_path / "report.json"
    destination = tmp_path / "decision.json"
    plan_path.write_text(json.dumps(_plan()))
    report_path.write_text(json.dumps(_selection_report()))
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_glm52_candidate_subset_selection.py",
            "--plan",
            str(plan_path),
            "--report",
            str(report_path),
            "--dest",
            str(destination),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    decision = json.loads(destination.read_text())
    assert decision["status"] == "complete"
    assert decision["retained_arm_names"] == ["accepted-experts-without-164"]
    assert decision["arms"][0]["screening_document_mean_delta"] == pytest.approx(-0.2)
    assert decision["arms"][0]["selection_check_document_mean_delta"] == pytest.approx(-0.1)


def test_selection_document_groups_reject_extra_report_arm() -> None:
    report = _selection_report()
    report["candidate_arms"]["unregistered-arm"] = copy.deepcopy(  # type: ignore[index]
        report["candidate_arms"]["expert-123-alone"]  # type: ignore[index]
    )
    with pytest.raises(ValueError, match="outside the frozen plan"):
        summarize_selection_document_groups(report, plan=_plan())


@pytest.mark.parametrize(
    ("layer", "manifest_sha256", "expert_ids"),
    [
        (
            52,
            "dddf3a9b238201b7718792a00f6c43e52bdd623796b0ef9a79be1381978e217d",
            [29, 36, 68, 96, 116, 186, 235, 254],
        ),
        (
            60,
            "13fdde80894d5728d3f78b28e52c22fc266a6295b22cb2b8dd7d5138bf21eabc",
            [28, 78, 125, 136, 142, 180, 186, 230],
        ),
        (
            63,
            "0ced0fdc2898e5091ce5afe0c3c744dbea1b57240d76169a283293a4c18b6d2e",
            [29, 32, 118, 123, 149, 164, 199, 215],
        ),
        (
            64,
            "84fabf919b13455aa88348868a1c93cb3133fc19388cc45c917d6a738cf7765b",
            [76, 85, 90, 106, 155, 210, 241, 253],
        ),
    ],
)
def test_frozen_down_refit_singleton_plans_cover_each_panel_expert(
    layer: int, manifest_sha256: str, expert_ids: list[int]
) -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / f"glm52_layer{layer}_down_refit_singleton_model_kld_selection_plan.json"
    )
    plan = json.loads(path.read_text())
    arms = validate_candidate_subset_selection_plan(
        plan,
        artifact={
            "manifest_sha256": manifest_sha256,
            "model_layer": layer,
            "expert_ids": expert_ids,
        },
    )
    assert [arm["selected_experts"][0] for arm in arms] == expert_ids
    assert all(len(arm["selected_experts"]) == 1 for arm in arms)


@pytest.mark.parametrize(
    ("layer", "manifest_sha256", "artifact_experts", "selected_experts"),
    [
        (
            52,
            "095b8975c67dd3fa9c4d0364d8c0babd62be3ecddb21187266c044c3d30ea7b6",
            [29, 36, 68, 96, 116, 186, 235, 254],
            [29, 36, 68, 96, 116, 186, 235],
        ),
        (
            60,
            "11bbe93f3aa781e05bf4ae762c23ef5a58a3ceeade9fcae906a07847f89eb896",
            [28, 78, 125, 136, 142, 180, 186, 230],
            [28, 78, 125, 136, 180, 230],
        ),
        (
            64,
            "1c87bdfb18d583b5831dc76b51e3ecd4456bb7b990d502a36cda5f1aaeb8a82d",
            [76, 85, 90, 106, 155, 210, 241, 253],
            [85, 90, 106, 155, 210, 241],
        ),
    ],
)
def test_frozen_rank4_low_rank_down_singleton_plans_cover_only_unmeasured_accepted_corrections(
    layer: int,
    manifest_sha256: str,
    artifact_experts: list[int],
    selected_experts: list[int],
) -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / f"glm52_layer{layer}_rank4_low_rank_down_singleton_model_kld_selection_plan.json"
    )
    plan = json.loads(path.read_text())
    arms = validate_candidate_subset_selection_plan(
        plan,
        artifact={
            "manifest_sha256": manifest_sha256,
            "model_layer": layer,
            "expert_ids": artifact_experts,
        },
    )
    assert [arm["selected_experts"][0] for arm in arms] == selected_experts
    assert all(len(arm["selected_experts"]) == 1 for arm in arms)


def test_frozen_layer63_retained_composition_closes_selected_experts() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "glm52_layer63_down_refit_model_kld_retained_composition_plan.json"
    )
    plan = json.loads(path.read_text())
    arms = validate_candidate_subset_selection_plan(
        plan,
        artifact={
            "manifest_sha256": (
                "0ced0fdc2898e5091ce5afe0c3c744dbea1b57240d76169a283293a4c18b6d2e"
            ),
            "model_layer": 63,
            "expert_ids": [29, 32, 118, 123, 149, 164, 199, 215],
        },
    )
    assert arms == [
        {
            "name": "experts-149-and-164",
            "selected_experts": [149, 164],
            "reason": (
                "Both singleton candidates improved equal-document mean KLD in "
                "the first eight and the remaining eight public documents. "
                "Measuring the pair tests non-additive model effects."
            ),
        }
    ]


def test_layer63_retained_composition_single_reference_registration_is_frozen() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "glm52_layer63_down_refit_model_kld_retained_composition_single_reference_registration.json"
    )
    registration = json.loads(path.read_text())

    assert registration["schema"] == "qsrt_glm52_single_reference_candidate_registration"
    assert registration["status"] == "frozen_before_single_reference_measurement"
    assert registration["candidate"]["selected_experts"] == [149, 164]
    assert registration["candidate"]["candidate_subset_plan_sha256"] == (
        "c4e5787501d1f6a2caa92031f50e61cfd1605abd58bb94d45f8e19f8edaccebe"
    )
    assert registration["reference"]["context_length"] == 2048
    assert registration["reference"]["model_revision"] == (
        "4d67f66cc64d3219133b767c253b2ad1425c6c88"
    )
    assert registration["reference"]["corpus_separation_plan_sha256"] == (
        "b694ac0a1aeb09f7c61a20b5f72289f3e791d616ff7471b5894d857f8c363b55"
    )
    assert registration["reference"]["reference_fit_document_overlap"] == 0
    assert registration["reference"]["reference_selection_document_overlap"] == 0
    assert "cannot" in registration["evidence_boundary"]

"""Validate frozen GLM-5.2 candidate subsets for model-KLD selection.

The model runner can switch a prebuilt intervention artifact on for any subset
of its experts without reloading the resident checkpoint.  A selection plan
freezes those subsets before the corresponding KLD values are computed.  The
documents used by this runner are selection data, not confirmation data.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


SELECTION_PLAN_SCHEMA = "qsrt_glm52_model_kld_candidate_subset_selection"


def validate_candidate_subset_selection_plan(
    plan: Mapping[str, Any], *, artifact: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return normalized candidate arms after closing plan and artifact identity."""

    if plan.get("schema") != SELECTION_PLAN_SCHEMA or plan.get("schema_version") != 1:
        raise ValueError("candidate-subset selection plan schema mismatch")
    if plan.get("status") != "frozen_before_candidate_subset_kld_measurement":
        raise ValueError("candidate-subset selection plan is not frozen")
    if plan.get("artifact_manifest_sha256") != artifact.get("manifest_sha256"):
        raise ValueError("candidate-subset selection plan artifact identity mismatch")
    if plan.get("model_layer") != artifact.get("model_layer"):
        raise ValueError("candidate-subset selection plan layer mismatch")

    protocol = plan.get("selection_protocol")
    if protocol is not None:
        if not isinstance(protocol, Mapping):
            raise TypeError("selection protocol must be an object")
        if protocol.get("document_order_source") != (
            "selected_chunks in public-reference plan order"
        ):
            raise ValueError("selection protocol document order is unsupported")
        if protocol.get("screening_document_count") != 8:
            raise ValueError("selection protocol must use eight screening documents")
        if protocol.get("selection_check_document_count") != 8:
            raise ValueError("selection protocol must use eight check documents")

    artifact_experts = set(artifact.get("expert_ids", ()))
    if not artifact_experts:
        raise ValueError("intervention artifact contains no experts")
    raw_arms = plan.get("candidate_arms")
    if not isinstance(raw_arms, list) or not raw_arms:
        raise ValueError("candidate-subset selection plan must contain candidate arms")
    if len(raw_arms) > 16:
        raise ValueError("candidate-subset selection plan exceeds the 16-arm limit")

    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    subsets: set[tuple[int, ...]] = set()
    for raw in raw_arms:
        if not isinstance(raw, Mapping):
            raise TypeError("each candidate arm must be an object")
        name = raw.get("name")
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 80
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in name)
        ):
            raise ValueError("candidate arm names must be path-safe lowercase identifiers")
        if name in names:
            raise ValueError("candidate arm names must be unique")
        names.add(name)

        raw_experts = raw.get("selected_experts")
        if not isinstance(raw_experts, list) or not raw_experts:
            raise ValueError("each candidate arm must select at least one expert")
        if any(type(expert) is not int for expert in raw_experts):
            raise TypeError("selected expert IDs must be integers")
        experts = tuple(sorted(raw_experts))
        if len(experts) != len(set(experts)):
            raise ValueError("a candidate arm must not repeat an expert")
        if not set(experts).issubset(artifact_experts):
            raise ValueError("candidate arm selects an expert outside the artifact")
        if experts in subsets:
            raise ValueError("candidate arms must contain distinct expert subsets")
        subsets.add(experts)

        reason = raw.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("each candidate arm must state its selection reason")
        normalized.append(
            {
                "name": name,
                "selected_experts": list(experts),
                "reason": reason.strip(),
            }
        )
    return normalized


def summarize_selection_document_groups(
    report: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Apply a frozen eight-document plus eight-document retention rule."""

    if report.get("schema") != (
        "qsrt_glm52_document_disjoint_model_kld_candidate_selection"
    ) or report.get("schema_version") != 1:
        raise ValueError("candidate-subset selection report schema mismatch")
    if report.get("status") != "complete":
        raise ValueError("candidate-subset selection report is incomplete")
    protocol = plan.get("selection_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("candidate-subset selection plan has no selection protocol")
    screening_count = protocol.get("screening_document_count")
    check_count = protocol.get("selection_check_document_count")
    if screening_count != 8 or check_count != 8:
        raise ValueError("candidate-subset selection protocol must split eight and eight")

    report_arms = report.get("candidate_arms")
    report_documents = report.get("documents")
    plan_arms = plan.get("candidate_arms")
    if (
        not isinstance(report_arms, Mapping)
        or not isinstance(report_documents, list)
        or len(report_documents) != screening_count + check_count
        or not isinstance(plan_arms, list)
    ):
        raise ValueError("candidate-subset selection arms are malformed")

    summaries: list[dict[str, Any]] = []
    expected_names: list[str] = []
    for plan_arm in plan_arms:
        if not isinstance(plan_arm, Mapping) or not isinstance(plan_arm.get("name"), str):
            raise ValueError("candidate-subset selection plan arm is malformed")
        name = plan_arm["name"]
        expected_names.append(name)
        report_arm = report_arms.get(name)
        if not isinstance(report_arm, Mapping):
            raise ValueError(f"candidate-subset report is missing arm {name}")
        deltas: list[float] = []
        for document in report_documents:
            if not isinstance(document, Mapping):
                raise ValueError("candidate-subset report has a malformed document")
            document_arms = document.get("candidate_arms")
            if not isinstance(document_arms, Mapping):
                raise ValueError("candidate-subset report document has no candidate arms")
            document_arm = document_arms.get(name)
            if not isinstance(document_arm, Mapping):
                raise ValueError(
                    f"candidate-subset report document is missing arm {name}"
                )
            delta = document_arm.get("candidate_minus_resident_mean_forward_kld")
            if not isinstance(delta, (int, float)) or not math.isfinite(float(delta)):
                raise ValueError(f"candidate-subset report arm {name} has a non-finite delta")
            deltas.append(float(delta))
        screening_mean = sum(deltas[:screening_count]) / screening_count
        check_mean = sum(deltas[screening_count:]) / check_count
        overall_mean = sum(deltas) / len(deltas)
        summaries.append(
            {
                "name": name,
                "selected_experts": list(report_arm.get("selected_experts", ())),
                "screening_document_mean_delta": screening_mean,
                "selection_check_document_mean_delta": check_mean,
                "all_document_mean_delta": overall_mean,
                "retained": screening_mean < 0.0 and check_mean < 0.0,
            }
        )
    if set(report_arms) != set(expected_names):
        raise ValueError("candidate-subset report contains arms outside the frozen plan")
    return summaries


__all__ = [
    "SELECTION_PLAN_SCHEMA",
    "summarize_selection_document_groups",
    "validate_candidate_subset_selection_plan",
]

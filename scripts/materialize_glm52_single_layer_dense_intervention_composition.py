#!/usr/bin/env python3
"""Compose registered dense expert endpoints from one GLM-5.2 layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Any

from qsrt.correctness import sha256_file
from qsrt.glm52_expert_intervention import INTERVENTION_ARTIFACT_KIND
from qsrt.glm52_expert_intervention_runtime import (
    LEGACY_K3_CANDIDATE_MODE,
    validate_dense_intervention_artifact,
)
from qsrt.glm52_pilot import atomic_write_json


REGISTRATION_SCHEMA = (
    "qsrt_glm52_single_layer_dense_intervention_composition_registration"
)


def _canonical_json_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _safe_directory_name(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in (".", "..")
    ):
        raise ValueError(f"{field} must be one directory name")
    return value


def _validate_registration(value: Any) -> tuple[int, list[dict[str, Any]]]:
    if not isinstance(value, dict):
        raise TypeError("dense intervention composition registration must be an object")
    if value.get("schema") != REGISTRATION_SCHEMA or value.get("schema_version") != 1:
        raise ValueError("dense intervention composition registration schema mismatch")
    frozen_at_utc = value.get("frozen_at_utc")
    if not isinstance(frozen_at_utc, str) or not frozen_at_utc:
        raise ValueError("dense intervention composition has no freeze time")
    model_layer = value.get("model_layer")
    if (
        isinstance(model_layer, bool)
        or not isinstance(model_layer, int)
        or not 3 <= model_layer <= 77
    ):
        raise ValueError(
            "dense intervention composition model_layer must be from 3 through 77"
        )
    components = value.get("components")
    if not isinstance(components, list) or len(components) < 2:
        raise ValueError("dense intervention composition needs at least two components")

    normalized: list[dict[str, Any]] = []
    seen_experts: set[int] = set()
    for component in components:
        if not isinstance(component, dict):
            raise TypeError("dense intervention composition component must be an object")
        source_name = _safe_directory_name(
            component.get("source_artifact_name"), field="source_artifact_name"
        )
        expert_ids = component.get("expert_ids")
        if (
            not isinstance(expert_ids, list)
            or not expert_ids
            or any(type(expert) is not int or not 0 <= expert < 256 for expert in expert_ids)
            or len(expert_ids) != len(set(expert_ids))
            or bool(seen_experts.intersection(expert_ids))
        ):
            raise ValueError(
                "component expert_ids must be disjoint IDs from 0 through 255"
            )
        logical_adapter_bytes = component.get("logical_adapter_bytes", 0)
        if (
            isinstance(logical_adapter_bytes, bool)
            or not isinstance(logical_adapter_bytes, int)
            or logical_adapter_bytes < 0
        ):
            raise ValueError("component logical_adapter_bytes must be nonnegative")
        seen_experts.update(expert_ids)
        normalized.append(
            {
                **component,
                "source_artifact_name": source_name,
                "expert_ids": sorted(expert_ids),
                "logical_adapter_bytes": logical_adapter_bytes,
            }
        )
    return model_layer, normalized


def _validate_selection_evidence(
    *,
    component: dict[str, Any],
    results_root: Path,
    source_manifest_sha256: str,
) -> dict[str, Any] | None:
    evidence = component.get("selection_evidence")
    if evidence is None:
        return None
    if not isinstance(evidence, dict):
        raise TypeError("component selection_evidence must be an object")
    result_name = _safe_directory_name(
        evidence.get("selection_result_name"), field="selection_result_name"
    )
    arm_name = evidence.get("candidate_arm")
    if not isinstance(arm_name, str) or not arm_name:
        raise ValueError("component selection evidence has no candidate arm")
    report_path = results_root / result_name / "report.json"
    expected_report_sha256 = evidence.get("selection_report_sha256")
    if (
        not isinstance(expected_report_sha256, str)
        or sha256_file(report_path) != expected_report_sha256
    ):
        raise ValueError("component selection report identity mismatch")
    report = json.loads(report_path.read_text())
    if report.get("status") != "complete" or not report.get(
        "measurement_controls", {}
    ).get("passed"):
        raise ValueError("component selection report controls did not pass")
    intervention = report.get("intervention_artifact")
    if (
        not isinstance(intervention, dict)
        or intervention.get("manifest_sha256") != source_manifest_sha256
    ):
        raise ValueError("component selection report belongs to another artifact")
    documents = report.get("documents")
    if not isinstance(documents, list) or len(documents) != 16:
        raise ValueError("component selection report must contain sixteen documents")
    try:
        deltas = [
            float(
                document["candidate_arms"][arm_name][
                    "candidate_minus_resident_mean_forward_kld"
                ]
            )
            for document in documents
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("component selection arm evidence is incomplete") from error
    measured = {
        "first_eight_mean_candidate_minus_resident_kld": statistics.fmean(
            deltas[:8]
        ),
        "second_eight_mean_candidate_minus_resident_kld": statistics.fmean(
            deltas[8:]
        ),
        "all_document_mean_candidate_minus_resident_kld": statistics.fmean(deltas),
    }
    for field, actual in measured.items():
        expected = evidence.get(field)
        if not isinstance(expected, (int, float)) or not math.isclose(
            actual, float(expected), rel_tol=0.0, abs_tol=1e-18
        ):
            raise ValueError(f"component selection metric {field} changed")
    if (
        measured["first_eight_mean_candidate_minus_resident_kld"] >= 0.0
        or measured["second_eight_mean_candidate_minus_resident_kld"] >= 0.0
    ):
        raise ValueError("component did not improve both selection document groups")
    return {
        "result_name": result_name,
        "report_sha256": expected_report_sha256,
        "candidate_arm": arm_name,
        **measured,
    }


def materialize_single_layer_composition(
    *, registration_path: Path, results_root: Path, destination: Path
) -> dict[str, Any]:
    """Copy disjoint registered endpoints into one hash-bound layer artifact."""

    registration_path = registration_path.resolve(strict=True)
    results_root = results_root.resolve(strict=True)
    registration = json.loads(registration_path.read_text())
    model_layer, components = _validate_registration(registration)
    destination.mkdir(parents=True, exist_ok=False)
    expert_root = destination / "experts"
    expert_root.mkdir()
    copied_records: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    tensor_prefix: str | None = None
    total_dense_endpoint_bytes = 0
    total_logical_adapter_bytes = 0
    try:
        for component in components:
            source_root = results_root / component["source_artifact_name"]
            source = validate_dense_intervention_artifact(source_root)
            if source["model_layer"] != model_layer:
                raise ValueError("composition source belongs to a different model layer")
            if tensor_prefix is None:
                tensor_prefix = source["candidate_tensor_prefix"]
            elif tensor_prefix != source["candidate_tensor_prefix"]:
                raise ValueError("composition sources use different tensor prefixes")
            selected = component["expert_ids"]
            if not set(selected).issubset(source["expert_ids"]):
                raise ValueError("composition selects an unavailable source expert")
            selection_evidence = _validate_selection_evidence(
                component=component,
                results_root=results_root,
                source_manifest_sha256=source["manifest_sha256"],
            )
            records_by_expert = {
                int(record["expert"]): record for record in source["report"]["experts"]
            }
            for expert in selected:
                record = dict(records_by_expert[expert])
                filename = record["dense_endpoint_file"]
                source_file = source_root / "experts" / filename
                destination_file = expert_root / filename
                if destination_file.exists():
                    raise ValueError("composition endpoint filenames collide")
                shutil.copyfile(source_file, destination_file)
                if (
                    destination_file.stat().st_size
                    != record["dense_endpoint_file_bytes"]
                    or sha256_file(destination_file)
                    != record["dense_endpoint_file_sha256"]
                ):
                    raise RuntimeError("copied endpoint failed its source identity")
                copied_records.append(record)
                total_dense_endpoint_bytes += destination_file.stat().st_size
            source_receipt = {
                "root": str(source_root),
                "manifest_sha256": source["manifest_sha256"],
                "report_sha256": sha256_file(source_root / "report.json"),
                "experts": selected,
                "logical_adapter_bytes": component["logical_adapter_bytes"],
            }
            if selection_evidence is not None:
                source_receipt["selection_evidence"] = selection_evidence
            source_receipts.append(source_receipt)
            total_logical_adapter_bytes += component["logical_adapter_bytes"]

        copied_records.sort(key=lambda record: int(record["expert"]))
        experts = [int(record["expert"]) for record in copied_records]
        registration_sha256 = sha256_file(registration_path)
        manifest = {
            "kind": f"{INTERVENTION_ARTIFACT_KIND}_manifest",
            "candidate": {
                "profile": "registered_single_layer_dense_endpoint_composition",
                "tensor_prefix": tensor_prefix or LEGACY_K3_CANDIDATE_MODE,
            },
            "input_intervention_artifact": {
                "root": str(registration_path),
                "manifest_sha256": registration_sha256,
                "report_sha256": registration_sha256,
            },
            "input_artifacts": source_receipts,
            "panel": {str(model_layer): experts},
            "registration": {
                "path": str(registration_path),
                "sha256": registration_sha256,
                "frozen_at_utc": registration["frozen_at_utc"],
            },
        }
        manifest_sha256 = _canonical_json_sha256(manifest)
        report = {
            "kind": INTERVENTION_ARTIFACT_KIND,
            "status": "complete",
            "manifest_sha256": manifest_sha256,
            "layer": model_layer,
            "expert_count": len(experts),
            "dense_endpoint_bytes": total_dense_endpoint_bytes,
            "logical_adapter_bytes": total_logical_adapter_bytes,
            "experts": copied_records,
            "evidence_boundary": (
                "This artifact copies registered dense experiment endpoints. "
                "It is not a serialized QSRT checkpoint."
            ),
        }
        atomic_write_json(destination / "manifest.json", manifest)
        atomic_write_json(destination / "report.json", report)
        validated = validate_dense_intervention_artifact(destination)
        if validated["manifest_sha256"] != manifest_sha256:
            raise RuntimeError("materialized composition identity changed")
        return report
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = materialize_single_layer_composition(
        registration_path=args.registration,
        results_root=args.results_root,
        destination=args.dest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

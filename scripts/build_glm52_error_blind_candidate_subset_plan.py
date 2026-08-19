#!/usr/bin/env python3
"""Freeze error-blind singleton and complete-panel model-KLD arms.

The input artifact already contains one candidate endpoint for every expert in
one source-controlled panel. This tool reads only artifact identities and panel
membership. It does not read local reconstruction errors or model-KLD values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACT_KIND = "qsrt_glm52_dense_expert_intervention_v1"
MANIFEST_KIND = f"{ARTIFACT_KIND}_manifest"
PLAN_SCHEMA = "qsrt_glm52_model_kld_candidate_subset_selection"


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument(
        "--candidate-description",
        required=True,
        help="plain-language description of the candidate construction",
    )
    parser.add_argument(
        "--include-complete-panel",
        action="store_true",
        help="measure all eight candidate experts together as a predeclared arm",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact = args.artifact.resolve(strict=True)
    manifest = _read_object(artifact / "manifest.json")
    report = _read_object(artifact / "report.json")
    manifest_sha256 = _canonical_json_sha256(manifest)
    if manifest.get("kind") != MANIFEST_KIND:
        raise ValueError("intervention manifest kind mismatch")
    if report.get("kind") != ARTIFACT_KIND or report.get("status") != "complete":
        raise ValueError("intervention report is incomplete or has the wrong kind")
    if report.get("manifest_sha256") != manifest_sha256:
        raise ValueError("intervention report and manifest identities differ")
    layer = report.get("layer")
    if type(layer) is not int or not 3 <= layer <= 77:
        raise ValueError("intervention report has an invalid GLM model layer")
    records = report.get("experts")
    if not isinstance(records, list) or len(records) != 8:
        raise ValueError("intervention report must contain one eight-expert panel")
    experts: list[int] = []
    for record in records:
        if not isinstance(record, dict) or type(record.get("expert")) is not int:
            raise TypeError("intervention expert receipt is malformed")
        expert = int(record["expert"])
        if not 0 <= expert < 256 or expert in experts:
            raise ValueError("intervention expert IDs must be unique values from 0 to 255")
        experts.append(expert)
    panel = manifest.get("panel")
    if panel != {str(layer): experts}:
        raise ValueError("intervention manifest panel and report order differ")

    description = args.candidate_description.strip()
    if not description:
        raise ValueError("candidate description must contain text")
    arms = [
        {
            "name": f"expert-{expert}-alone",
            "selected_experts": [expert],
            "reason": (
                f"Measure expert {expert}'s {description} endpoint without any "
                "other candidate expert active."
            ),
        }
        for expert in sorted(experts)
    ]
    if args.include_complete_panel:
        arms.append(
            {
                "name": "complete-eight-expert-panel",
                "selected_experts": sorted(experts),
                "reason": (
                    f"Measure the complete frozen eight-expert {description} panel "
                    "as one predeclared composition."
                ),
            }
        )

    plan = {
        "schema": PLAN_SCHEMA,
        "schema_version": 1,
        "status": "frozen_before_candidate_subset_kld_measurement",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": (
            f"Measure the model-KLD effect of each frozen layer-{layer} {description} "
            "expert without using local complete-expert error as an authorization rule."
        ),
        "evidence_boundary": (
            "The 16 public 512-token documents are candidate-selection data. "
            "Any retained singleton or composition requires a new document-disjoint "
            "reference tier before it can support a checkpoint-quality claim."
        ),
        "artifact_manifest_sha256": manifest_sha256,
        "model_layer": layer,
        "selection_protocol": {
            "document_order_source": "selected_chunks in public-reference plan order",
            "screening_document_count": 8,
            "selection_check_document_count": 8,
            "retention_rule": (
                "Retain an arm only when candidate-minus-resident equal-document mean "
                "KLD is negative on both ordered document groups."
            ),
            "composition_boundary": (
                "A composition chosen after reading singleton results is a new selection "
                "candidate and requires a separate document-disjoint confirmation tier."
            ),
        },
        "candidate_arms": arms,
    }
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    if args.dest.exists():
        raise FileExistsError(args.dest)
    temporary = args.dest.with_name(f".{args.dest.name}.partial-{os.getpid()}")
    temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.dest)
    print(json.dumps({"dest": str(args.dest), "sha256": hashlib.sha256(args.dest.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()

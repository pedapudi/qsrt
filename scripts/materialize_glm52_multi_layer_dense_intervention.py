#!/usr/bin/env python3
"""Materialize one self-contained GLM-5.2 intervention spanning several layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from qsrt.correctness import sha256_file
from qsrt.glm52_expert_intervention import INTERVENTION_ARTIFACT_KIND
from qsrt.glm52_expert_intervention_runtime import (
    MULTI_LAYER_INTERVENTION_ARTIFACT_KIND,
    validate_dense_intervention_artifact,
    validate_multi_layer_intervention_artifact,
)
from qsrt.glm52_pilot import atomic_write_json


REGISTRATION_SCHEMA = "qsrt_glm52_multi_layer_intervention_registration"


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


def _validate_registration(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        raise TypeError("multi-layer intervention registration must be an object")
    if value.get("schema") != REGISTRATION_SCHEMA or value.get("schema_version") != 1:
        raise ValueError("multi-layer intervention registration schema mismatch")
    frozen_at_utc = value.get("frozen_at_utc")
    if not isinstance(frozen_at_utc, str) or not frozen_at_utc:
        raise ValueError("multi-layer intervention registration has no freeze time")
    components = value.get("components")
    if not isinstance(components, list) or len(components) < 2:
        raise ValueError("multi-layer intervention registration needs two components")
    normalized: list[dict[str, Any]] = []
    seen_layers: set[int] = set()
    for component in components:
        if not isinstance(component, dict):
            raise TypeError("multi-layer intervention component must be an object")
        model_layer = component.get("model_layer")
        if (
            isinstance(model_layer, bool)
            or not isinstance(model_layer, int)
            or not 3 <= model_layer <= 77
            or model_layer in seen_layers
        ):
            raise ValueError("multi-layer intervention layers must be unique 3..77")
        source_name = _safe_directory_name(
            component.get("source_artifact_name"), field="source_artifact_name"
        )
        expert_ids = component.get("expert_ids")
        if (
            not isinstance(expert_ids, list)
            or not expert_ids
            or any(type(expert) is not int or not 0 <= expert < 256 for expert in expert_ids)
            or len(expert_ids) != len(set(expert_ids))
        ):
            raise ValueError("component expert_ids must be distinct IDs from 0 through 255")
        seen_layers.add(model_layer)
        normalized.append(
            {
                **component,
                "model_layer": model_layer,
                "source_artifact_name": source_name,
                "expert_ids": sorted(expert_ids),
            }
        )
    if [record["model_layer"] for record in normalized] != sorted(seen_layers):
        raise ValueError("multi-layer intervention components must be layer ordered")
    return normalized


def materialize_multi_layer_intervention(
    *, registration_path: Path, results_root: Path, destination: Path
) -> dict[str, Any]:
    """Copy registered expert endpoints into a hash-bound multi-layer artifact."""

    registration_path = registration_path.resolve(strict=True)
    results_root = results_root.resolve(strict=True)
    registration = json.loads(registration_path.read_text())
    components = _validate_registration(registration)
    destination.mkdir(parents=True, exist_ok=False)
    component_receipts: list[dict[str, Any]] = []
    total_dense_endpoint_bytes = 0
    total_logical_additional_bytes = 0
    total_experts = 0
    try:
        for registered in components:
            model_layer = registered["model_layer"]
            source_root = results_root / registered["source_artifact_name"]
            source = validate_dense_intervention_artifact(source_root)
            if source["model_layer"] != model_layer:
                raise ValueError(
                    f"registered layer {model_layer} differs from its source artifact"
                )
            selected = registered["expert_ids"]
            if not set(selected).issubset(source["expert_ids"]):
                raise ValueError(
                    f"layer {model_layer} registration selects an unavailable expert"
                )
            source_manifest = json.loads((source_root / "manifest.json").read_text())
            source_report = source["report"]
            records_by_expert = {
                int(record["expert"]): record for record in source_report["experts"]
            }
            relative_root = f"components/layer-{model_layer:03d}"
            component_root = destination / relative_root
            expert_root = component_root / "experts"
            expert_root.mkdir(parents=True)
            copied_records: list[dict[str, Any]] = []
            component_bytes = 0
            for expert in selected:
                record = dict(records_by_expert[expert])
                filename = record["dense_endpoint_file"]
                source_file = source_root / "experts" / filename
                destination_file = expert_root / filename
                shutil.copyfile(source_file, destination_file)
                if (
                    destination_file.stat().st_size != record["dense_endpoint_file_bytes"]
                    or sha256_file(destination_file) != record["dense_endpoint_file_sha256"]
                ):
                    raise RuntimeError("copied dense endpoint failed its source identity")
                copied_records.append(record)
                component_bytes += destination_file.stat().st_size

            component_manifest = {
                "kind": f"{INTERVENTION_ARTIFACT_KIND}_manifest",
                "candidate": source_manifest["candidate"],
                "input_intervention_artifact": {
                    "root": str(source_root),
                    "manifest_sha256": source["manifest_sha256"],
                    "report_sha256": sha256_file(source_root / "report.json"),
                },
                "panel": {str(model_layer): selected},
            }
            component_manifest_sha256 = _canonical_json_sha256(component_manifest)
            component_report = {
                "kind": INTERVENTION_ARTIFACT_KIND,
                "status": "complete",
                "manifest_sha256": component_manifest_sha256,
                "layer": model_layer,
                "expert_count": len(selected),
                "dense_endpoint_bytes": component_bytes,
                "experts": copied_records,
                "evidence_boundary": (
                    "This component copies only the experts frozen by the "
                    "multi-layer model-KLD selection. Its dense endpoints are "
                    "an experiment representation, not serialized QSRT payloads."
                ),
            }
            atomic_write_json(component_root / "manifest.json", component_manifest)
            atomic_write_json(component_root / "report.json", component_report)
            validated_component = validate_dense_intervention_artifact(component_root)
            if validated_component["manifest_sha256"] != component_manifest_sha256:
                raise RuntimeError("materialized component identity changed")
            component_receipts.append(
                {
                    "model_layer": model_layer,
                    "relative_root": relative_root,
                    "manifest_sha256": component_manifest_sha256,
                    "expert_ids": selected,
                    "source_artifact": {
                        "root": str(source_root),
                        "manifest_sha256": source["manifest_sha256"],
                        "report_sha256": sha256_file(source_root / "report.json"),
                    },
                }
            )
            total_dense_endpoint_bytes += component_bytes
            total_logical_additional_bytes += int(
                registered.get("logical_adapter_bytes", 0)
            )
            total_experts += len(selected)

        manifest = {
            "kind": f"{MULTI_LAYER_INTERVENTION_ARTIFACT_KIND}_manifest",
            "registration": {
                "path": str(registration_path),
                "sha256": sha256_file(registration_path),
                "frozen_at_utc": registration["frozen_at_utc"],
            },
            "components": component_receipts,
            "logical_additional_bytes": total_logical_additional_bytes,
        }
        manifest_sha256 = _canonical_json_sha256(manifest)
        model_layers = [record["model_layer"] for record in component_receipts]
        report = {
            "kind": MULTI_LAYER_INTERVENTION_ARTIFACT_KIND,
            "status": "complete",
            "manifest_sha256": manifest_sha256,
            "model_layers": model_layers,
            "expert_ids_by_layer": {
                str(record["model_layer"]): record["expert_ids"]
                for record in component_receipts
            },
            "expert_count": total_experts,
            "dense_endpoint_bytes": total_dense_endpoint_bytes,
            "logical_additional_bytes": total_logical_additional_bytes,
            "evidence_boundary": (
                "This artifact measures the registered cross-layer composition "
                "inside the resident EXL3 model. It does not establish a complete "
                "QSRT checkpoint or a serialized model-size result."
            ),
        }
        atomic_write_json(destination / "manifest.json", manifest)
        atomic_write_json(destination / "report.json", report)
        validated = validate_multi_layer_intervention_artifact(destination)
        if validated["manifest_sha256"] != manifest_sha256:
            raise RuntimeError("multi-layer intervention identity changed")
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
    report = materialize_multi_layer_intervention(
        registration_path=args.registration,
        results_root=args.results_root,
        destination=args.dest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

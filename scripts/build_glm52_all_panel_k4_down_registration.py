#!/usr/bin/env python3
"""Freeze a smaller-than-EXL3 all-panel K4-down recovery map.

Every panel expert uses QSRT K3 for gate and up and K4 for down.  An accepted
reconstructed-activation refit supplies the down target; an expert whose refit
fell back keeps the source-weight target.  The registration reads no K4 error
or model-KLD result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

UNIFORM_K3_PANEL_BYTES = 113_643_520
ONE_PROJECTION_BIT_INCREMENT_BYTES = 1_572_864
REGISTERED_PARTIAL_RATE_MAP_SCHEMA = (
    "qsrt_glm52_registered_partial_down_refit_k3_k4_intervention"
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--down-refit", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    panel_path = args.panel_manifest.resolve(strict=True)
    refit_root = args.down_refit.resolve(strict=True)
    refit_manifest_path = refit_root / "manifest.json"
    refit_report_path = refit_root / "report.json"
    panel = _read_object(panel_path)
    refit_manifest = _read_object(refit_manifest_path)
    refit_report = _read_object(refit_report_path)

    if panel.get("schema") != "qsrt_glm52_real_weight_panel":
        raise ValueError("GLM panel schema mismatch")
    layer = panel.get("layer")
    if type(layer) is not int or not 3 <= layer <= 77:
        raise ValueError("GLM panel layer is invalid")
    raw_experts = panel.get("experts")
    if not isinstance(raw_experts, list) or len(raw_experts) != 8:
        raise ValueError("GLM panel must contain eight experts")
    expert_order: list[int] = []
    comparison_rates: dict[str, dict[str, int]] = {}
    comparison_rate_sum = 0
    for record in raw_experts:
        if not isinstance(record, dict) or type(record.get("expert")) is not int:
            raise TypeError("GLM panel expert record is malformed")
        expert = record["expert"]
        rates = record.get("exl3_rates")
        if (
            not isinstance(rates, list)
            or len(rates) != 3
            or any(type(rate) is not int or rate not in (3, 4, 5) for rate in rates)
        ):
            raise ValueError("GLM panel EXL3 rates are invalid")
        if expert in expert_order:
            raise ValueError("GLM panel repeats an expert")
        expert_order.append(expert)
        normalized = {
            "gate_proj": rates[0],
            "up_proj": rates[1],
            "down_proj": rates[2],
        }
        comparison_rates[str(expert)] = normalized
        comparison_rate_sum += sum(normalized.values())

    if refit_report.get("status") != "complete" or refit_report.get("layer") != layer:
        raise ValueError("down-refit artifact is incomplete or has the wrong layer")
    if refit_report.get("manifest_sha256") != hashlib.sha256(
        json.dumps(
            refit_manifest, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest():
        raise ValueError("down-refit manifest identity mismatch")
    refit_experts = [record.get("expert") for record in refit_report.get("experts", [])]
    if refit_experts != expert_order:
        raise ValueError("down-refit artifact and frozen panel order differ")

    candidate_rate_sum = len(expert_order) * 10
    projection_count = len(expert_order) * 3
    comparison_bytes = UNIFORM_K3_PANEL_BYTES + (
        comparison_rate_sum - 3 * projection_count
    ) * ONE_PROJECTION_BIT_INCREMENT_BYTES
    candidate_bytes = UNIFORM_K3_PANEL_BYTES + (
        candidate_rate_sum - 3 * projection_count
    ) * ONE_PROJECTION_BIT_INCREMENT_BYTES
    if candidate_bytes >= comparison_bytes:
        raise ValueError("all-panel K4-down map is not smaller than EXL3")

    comparison_checkpoint = panel.get("comparison_checkpoint")
    source = panel.get("source")
    if not isinstance(comparison_checkpoint, dict) or not isinstance(source, dict):
        raise TypeError("GLM panel lacks source or comparison identity")
    registration = {
        "schema": REGISTERED_PARTIAL_RATE_MAP_SCHEMA,
        "schema_version": 1,
        "status": "frozen_before_candidate_k4_measurement",
        "objective": (
            f"Measure every frozen layer-{layer} expert with QSRT K3 gate and up "
            "and K4 down while keeping the complete panel smaller than EXL3."
        ),
        "model_layer": layer,
        "source_weights": {
            "model_id": source["model_id"],
            "revision": source["revision"],
            "configuration_sha256": source["config_sha256"],
            "tensor_index_sha256": source["index_sha256"],
            "complete_checkpoint_required": False,
        },
        "comparison_checkpoint": {
            "model_id": comparison_checkpoint["model_id"],
            "revision": comparison_checkpoint["revision"],
            "root_manifest_sha256": comparison_checkpoint["manifest_sha256"],
        },
        "comparison_panel": {
            "path": str(panel_path),
            "manifest_sha256": _sha256(panel_path),
            "expert_order": expert_order,
        },
        "down_refit_base": {
            "artifact_path": str(refit_root),
            "manifest_file_sha256": _sha256(refit_manifest_path),
            "report_file_sha256": _sha256(refit_report_path),
        },
        "candidate_construction": {
            "gate_rate": 3,
            "up_rate": 3,
            "down_rate": 4,
            "down_target_policy": (
                "use the reconstructed-activation refit when the K3 refit artifact "
                "accepted it; otherwise use that artifact's source-target fallback"
            ),
            "allow_source_target_fallback": True,
            "k4_candidate_measurements_used": False,
        },
        "comparison_exl3_rates": comparison_rates,
        "registered_replacements": [
            {
                "expert": expert,
                "candidate_rates": {
                    "gate_proj": 3,
                    "up_proj": 3,
                    "down_proj": 4,
                },
                "reason": (
                    "error-blind all-panel K4-down rate map frozen before K4 "
                    "candidate construction or model-KLD measurement"
                ),
            }
            for expert in expert_order
        ],
        "logical_byte_contract": {
            "uniform_k3_panel_bytes": UNIFORM_K3_PANEL_BYTES,
            "one_projection_bit_increment_bytes": ONE_PROJECTION_BIT_INCREMENT_BYTES,
            "comparison_exl3_rate_sum": comparison_rate_sum,
            "comparison_exl3_panel_bytes": comparison_bytes,
            "registered_candidate_rate_sum": candidate_rate_sum,
            "registered_candidate_panel_bytes": candidate_bytes,
            "logical_margin_bytes": comparison_bytes - candidate_bytes,
            "boundary": (
                "The ledger covers the complete comparison panel's trellis payload "
                "and scale storage. It is not a complete serialized checkpoint ledger."
            ),
        },
        "measurement_order": {
            "selection": (
                "measure eight singleton experts and the predeclared complete panel "
                "on two ordered groups of public BF16-reference documents"
            ),
            "development_screen": (
                "freeze retained arms before opening the separate 2,048-token reference"
            ),
        },
    }
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    if args.dest.exists():
        raise FileExistsError(args.dest)
    temporary = args.dest.with_name(f".{args.dest.name}.partial-{os.getpid()}")
    temporary.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.dest)
    print(json.dumps({"dest": str(args.dest), "sha256": _sha256(args.dest)}))


if __name__ == "__main__":
    main()

"""Materialize a frozen mixed-K3/K4 GLM-5.2 intervention artifact.

The base artifact supplies K3 gate/up tensors and the already-selected
reconstructed-activation K3 down refit.  A separate source-target K4 artifact
supplies every possible K4 projection.  This module applies only an allocation
that was frozen before the K4 measurements existed; it does not inspect the
reporting KLD context or choose rates from weight error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open

from qsrt.correctness import sha256_file
from qsrt.glm52_expert_intervention import (
    INTERVENTION_ARTIFACT_KIND,
    _atomic_save_tensors,
    _dense_expert_path,
)
from qsrt.glm52_expert_intervention_runtime import (
    validate_dense_intervention_artifact,
)
from qsrt.glm52_pilot import PROJECTIONS, _expert_path, atomic_write_json, prepare_destination
from qsrt.qsrt_codec_pilot import tensor_sha256


MIXED_ALLOCATION_EXPERIMENT = "qsrt_glm52_fixed_mixed_k3_k4_down_refit_v1"
ALLOCATION_PRE_REGISTRATION_IDENTITIES = {
    "qsrt_glm52_layer3_k3_k4_allocation_pre_registration": (
        "frozen_before_k4_candidate_measurement"
    ),
    "qsrt_glm52_layer3_rate_preserving_down_refit_k3_k4_pre_registration": (
        "frozen_before_rate_preserving_k4_candidate_measurement"
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def frozen_k4_rate_map(pre_registration: Mapping[str, Any]) -> dict[int, set[str]]:
    """Return the immutable fixed allocation as expert-to-projection sets."""

    schema = pre_registration.get("schema")
    if (
        schema not in ALLOCATION_PRE_REGISTRATION_IDENTITIES
        or pre_registration.get("status")
        != ALLOCATION_PRE_REGISTRATION_IDENTITIES[schema]
    ):
        raise ValueError("mixed-rate pre-registration identity mismatch")
    allocation = pre_registration.get("fixed_exl3_rate_stratified_allocation")
    if not isinstance(allocation, dict) or allocation.get("candidate_measurements_used") is not False:
        raise ValueError("fixed allocation must be independent of candidate measurements")
    entries = allocation.get("k4_projections")
    if not isinstance(entries, list):
        raise TypeError("fixed K4 projections must be a list")
    allowed = {spec.name for spec in PROJECTIONS}
    rate_map: dict[int, set[str]] = {}
    seen: set[tuple[int, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("fixed K4 projection entry must be an object")
        expert = entry.get("expert")
        projection = entry.get("projection")
        key = (expert, projection)
        if (
            isinstance(expert, bool)
            or not isinstance(expert, int)
            or not 0 <= expert < 256
            or projection not in allowed
            or key in seen
        ):
            raise ValueError("fixed K4 projection entries must be unique valid cells")
        seen.add(key)
        rate_map.setdefault(expert, set()).add(projection)
    maximum = pre_registration["rate_contract"]["maximum_k4_projection_count"]
    if len(seen) != int(maximum):
        raise ValueError("fixed allocation does not spend the frozen K4 budget")
    return rate_map


def _records_by_expert(artifact: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {
        int(record["expert"]): record
        for record in artifact["report"]["experts"]
    }


def _load_endpoint(path: Path, prefix: str) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        required = {
            f"{endpoint}.{spec.name}"
            for endpoint in ("exl3", prefix)
            for spec in PROJECTIONS
        }
        if not required.issubset(keys):
            raise ValueError(f"{path} lacks required endpoint tensors")
        return {key: handle.get_tensor(key) for key in required}


def materialize_fixed_mixed_artifact(
    *,
    base_root: Path,
    uniform_k4_root: Path,
    pre_registration_path: Path,
    dest: Path,
) -> dict[str, Any]:
    """Combine frozen K3/down-refit and K4 tensors without selecting on KLD."""

    pre_registration = _read_json(pre_registration_path)
    pre_registration_sha256 = sha256_file(pre_registration_path)
    rate_map = frozen_k4_rate_map(pre_registration)
    base = validate_dense_intervention_artifact(base_root)
    k4 = validate_dense_intervention_artifact(uniform_k4_root)
    if k4["candidate_tensor_prefix"] != "qsrt_k4":
        raise ValueError("uniform-K4 artifact does not expose qsrt_k4 tensors")
    panel_order = tuple(pre_registration["expert_panel"]["expert_order"])
    if set(panel_order) != set(base["expert_ids"]) or set(panel_order) != set(k4["expert_ids"]):
        raise ValueError("mixed-rate inputs do not cover the frozen expert panel")
    base_manifest_path = Path(base["root"]) / "manifest.json"
    base_report_path = Path(base["root"]) / "report.json"
    expected_base = pre_registration["base_representation"]
    if (
        sha256_file(base_manifest_path) != expected_base["artifact_manifest_file_sha256"]
        or sha256_file(base_report_path) != expected_base["artifact_report_file_sha256"]
    ):
        raise ValueError("base down-refit artifact differs from the pre-registration")

    logical_bytes = int(
        pre_registration["logical_byte_gate"]["mixed_qsrt_bytes_at_twelve_k4_projections"]
    )
    comparison_bytes = int(pre_registration["logical_byte_gate"]["comparison_exl3_bytes"])
    if logical_bytes >= comparison_bytes:
        raise ValueError("frozen mixed allocation is not logically smaller than EXL3")
    manifest = {
        "kind": f"{INTERVENTION_ARTIFACT_KIND}_manifest",
        "candidate": {
            "profile": "qsrt_sqg_e4m3",
            "tensor_prefix": "candidate",
            "variant": "fixed_mixed_k3_k4_over_reconstructed_activation_down_refit",
            "allocation_source": "pre_registered_EXL3_rate_stratification",
            "candidate_measurements_used": False,
        },
        "input_intervention_artifact": {
            "root": base["root"],
            "manifest_sha256": base["manifest_sha256"],
        },
        "uniform_k4_artifact": {
            "root": k4["root"],
            "manifest_sha256": k4["manifest_sha256"],
        },
        "pre_registration": {
            "path": str(pre_registration_path.resolve()),
            "sha256": pre_registration_sha256,
        },
        "panel": {"3": list(panel_order)},
        "logical_byte_accounting": {
            "mixed_qsrt_bytes": logical_bytes,
            "comparison_exl3_bytes": comparison_bytes,
            "logical_margin_bytes": comparison_bytes - logical_bytes,
            "serialized_container_gate_passed": False,
        },
        "resident_endpoint_dtype": "FP16",
        "resident_coordinate_basis": "sealed_R7_permuted_middle_coordinates",
        "evidence_boundary": (
            "the dense intervention measures a frozen layer-3 mechanism; logical "
            "rate accounting is not a complete serialized QSRT container byte result"
        ),
        "model_downloads_required": False,
        "complete_bf16_checkpoint_required": False,
    }
    manifest_sha256 = prepare_destination(dest, manifest, resume=False)
    base_records = _records_by_expert(base)
    k4_records = _records_by_expert(k4)
    records: list[dict[str, Any]] = []
    for expert in panel_order:
        base_record = base_records[expert]
        k4_record = k4_records[expert]
        base_path = Path(base["root"]) / "experts" / base_record["dense_endpoint_file"]
        k4_path = Path(k4["root"]) / "experts" / k4_record["dense_endpoint_file"]
        base_tensors = _load_endpoint(base_path, base["candidate_tensor_prefix"])
        k4_tensors = _load_endpoint(k4_path, "qsrt_k4")
        output: dict[str, torch.Tensor] = {}
        rates: dict[str, int] = {}
        hashes: dict[str, str] = {}
        for spec in PROJECTIONS:
            exl3_key = f"exl3.{spec.name}"
            if not torch.equal(base_tensors[exl3_key], k4_tensors[exl3_key]):
                raise ValueError(f"expert {expert} {spec.name} EXL3 endpoints differ")
            output[exl3_key] = base_tensors[exl3_key]
            use_k4 = spec.name in rate_map.get(expert, set())
            source_prefix = "qsrt_k4" if use_k4 else base["candidate_tensor_prefix"]
            source_tensors = k4_tensors if use_k4 else base_tensors
            candidate = source_tensors[f"{source_prefix}.{spec.name}"]
            output[f"candidate.{spec.name}"] = candidate
            rates[spec.name] = 4 if use_k4 else 3
            hashes[spec.name] = tensor_sha256(candidate)
        output_path = _dense_expert_path(dest, 3, expert)
        _atomic_save_tensors(output_path, output)
        record = {
            "kind": f"{INTERVENTION_ARTIFACT_KIND}_expert",
            "complete": True,
            "manifest_sha256": manifest_sha256,
            "layer": 3,
            "expert": expert,
            "dense_endpoint_file": output_path.name,
            "dense_endpoint_file_bytes": output_path.stat().st_size,
            "dense_endpoint_file_sha256": sha256_file(output_path),
            "rates": rates,
            "candidate_tensor_sha256": hashes,
            "base_dense_endpoint_sha256": base_record["dense_endpoint_file_sha256"],
            "uniform_k4_dense_endpoint_sha256": k4_record["dense_endpoint_file_sha256"],
        }
        atomic_write_json(_expert_path(dest, 3, expert), record)
        records.append(record)
    report = {
        "kind": INTERVENTION_ARTIFACT_KIND,
        "experiment": MIXED_ALLOCATION_EXPERIMENT,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "layer": 3,
        "expert_count": len(records),
        "panel": manifest["panel"],
        "k4_projection_count": sum(
            rate == 4 for record in records for rate in record["rates"].values()
        ),
        "logical_byte_accounting": manifest["logical_byte_accounting"],
        "dense_endpoint_bytes": sum(record["dense_endpoint_file_bytes"] for record in records),
        "experts": records,
        "evidence_boundary": manifest["evidence_boundary"],
    }
    atomic_write_json(dest / "report.json", report)
    validate_dense_intervention_artifact(dest)
    return report


__all__ = [
    "MIXED_ALLOCATION_EXPERIMENT",
    "frozen_k4_rate_map",
    "materialize_fixed_mixed_artifact",
]

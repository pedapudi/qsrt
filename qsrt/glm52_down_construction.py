"""Compare down-projection input metrics and targets for GLM-5.2 experts.

Gate and up remain at their uniform-K3 reconstructions.  The two experiment
axes are the metric used while encoding the down matrix and the continuous
matrix presented to that encoder:

* identity or reconstructed-activation covariance; and
* original source weights or a ridge-fitted target that reproduces the source
  expert output from reconstructed gate/up activations.

The fitted target is independent of the encoding metric.  This makes the
four-cell comparison capable of separating conditioning from target refitting.
All targets are fit on activation-fit documents and all encoded candidates are
ranked on separate candidate-selection documents.  Full-model KLD on untouched
documents remains the acceptance measurement.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open

from qsrt.correctness import sha256_file
from qsrt.exl3_loader import load_qsrt_encoder
from qsrt.exl3_reference import CODEBOOK_SQG_XOR_CHEB_T12
from qsrt.glm52_down_refit import (
    _expert_hidden,
    _read_capture_rows,
    _teacher_output,
    solve_down_correction,
)
from qsrt.glm52_expert_intervention import (
    INTERVENTION_ARTIFACT_KIND,
    _atomic_save_tensors,
    _dense_expert_path,
)
from qsrt.glm52_expert_intervention_runtime import (
    validate_dense_intervention_artifact,
)
from qsrt.glm52_pilot import (
    PROJECTIONS,
    IndexedTensorStore,
    _expert_path,
    _transform_seeds,
    atomic_write_json,
    prepare_destination,
    source_tensor_name,
)
from qsrt.glm52_real_weight_benchmark import (
    load_frozen_real_weight_panel,
    select_frozen_panel_slice,
    validate_bounded_source_window,
)
from qsrt.glm52_routed_input_curvature import routed_input_hessian
from qsrt.ldlq import SIGMA_REG
from qsrt.qsrt_codec_pilot import encode_uniform_candidate, tensor_sha256
from qsrt.sqg_quantizer import install_sqg_quantizer


DOWN_CONSTRUCTION_COMPARISON_KIND = "qsrt_glm52_down_construction_comparison_v1"
INPUT_METRIC_POLICIES = ("identity", "reconstructed_input_covariance")
TARGET_POLICIES = ("source_weights", "reconstructed_activation_refit")


def down_construction_name(*, input_metric: str, target: str) -> str:
    """Return a stable, self-descriptive name for one comparison cell."""

    if input_metric not in INPUT_METRIC_POLICIES:
        raise ValueError(f"unsupported down input metric {input_metric!r}")
    if target not in TARGET_POLICIES:
        raise ValueError(f"unsupported down target {target!r}")
    return f"{input_metric}__{target}"


def route_weighted_output_error_statistics(
    teacher: torch.Tensor,
    candidate: torch.Tensor,
    route_weights: torch.Tensor,
) -> dict[str, float | int]:
    """Summarize complete-expert output error, including its routed-row tail."""

    if teacher.ndim != 2 or candidate.shape != teacher.shape:
        raise ValueError("teacher and candidate outputs must share one matrix shape")
    if route_weights.ndim != 1 or route_weights.numel() != teacher.shape[0]:
        raise ValueError("route weights must contain one value per output row")
    if teacher.shape[0] < 1:
        raise ValueError("output error requires at least one routed row")
    weights = route_weights.double().clamp_min(0.0).unsqueeze(1)
    row_error = ((teacher.double() - candidate.double()) * weights).square().sum(1)
    row_reference = (teacher.double() * weights).square().sum(1)
    error_sum = row_error.sum()
    reference_sum = row_reference.sum().clamp_min(1e-30)
    ordered = torch.sort(row_error).values
    row_count = int(ordered.numel())
    tail_count = max(1, math.ceil(row_count * 0.01))
    p99_index = min(row_count - 1, math.ceil(row_count * 0.99) - 1)
    return {
        "row_count": row_count,
        "weighted_error_sum": float(error_sum.item()),
        "weighted_reference_sum": float(reference_sum.item()),
        "weighted_relative_sse": float(error_sum.div(reference_sum).item()),
        "row_error_p99": float(ordered[p99_index].item()),
        "row_error_cvar1": float(ordered[-tail_count:].mean().item()),
        "row_error_max": float(ordered[-1].item()),
        "cvar1_row_count": tail_count,
    }


def refit_passes_local_fallback(
    *,
    baseline: Mapping[str, float | int],
    candidate: Mapping[str, float | int],
    tail_relative_tolerance: float,
) -> bool:
    """Apply the pre-model local mean and routed-row-tail fallback rule."""

    tolerance = float(tail_relative_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tail_relative_tolerance must be finite and nonnegative")
    baseline_mean = float(baseline["weighted_relative_sse"])
    candidate_mean = float(candidate["weighted_relative_sse"])
    baseline_tail = float(baseline["row_error_cvar1"])
    candidate_tail = float(candidate["row_error_cvar1"])
    mean_epsilon = max(1e-15, baseline_mean * 1e-9)
    tail_epsilon = max(1e-15, baseline_tail * 1e-9)
    return (
        candidate_mean < baseline_mean - mean_epsilon
        and candidate_tail
        <= baseline_tail * (1.0 + tolerance) + tail_epsilon
    )


def _load_uniform_tensors(
    root: Path, record: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    path = root / "experts" / record["dense_endpoint_file"]
    if sha256_file(path) != record["dense_endpoint_file_sha256"]:
        raise ValueError(f"input endpoint for expert {record['expert']} failed closure")
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {
            key: handle.get_tensor(key)
            for endpoint in ("exl3", "qsrt_k3")
            for key in (
                f"{endpoint}.gate_proj",
                f"{endpoint}.up_proj",
                f"{endpoint}.down_proj",
            )
        }


@torch.no_grad()
def build_down_construction_for_expert(
    *,
    source: IndexedTensorStore,
    input_artifact_root: Path,
    input_record: Mapping[str, Any],
    routed_rows: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    layer: int,
    expert: int,
    input_metric_policy: str,
    target_policy: str,
    ridge_factors: Sequence[float],
    covariance_identity_shrinkage: float,
    tail_relative_tolerance: float,
    device: torch.device,
    quantizer_module: Any,
    dest: Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Build one complete expert for one down-construction comparison cell."""

    construction = down_construction_name(
        input_metric=input_metric_policy, target=target_policy
    )
    factors = tuple(float(value) for value in ridge_factors)
    if target_policy == "reconstructed_activation_refit" and (
        not factors
        or any(not math.isfinite(value) or value <= 0.0 for value in factors)
    ):
        raise ValueError("ridge factors must be finite positive values")
    tensors = _load_uniform_tensors(input_artifact_root, input_record)
    source_weights = {
        spec.name: source.get(source_tensor_name(layer, expert, spec.name))
        for spec in PROJECTIONS
    }
    source_gpu = {
        name: value.to(device=device, dtype=torch.bfloat16).contiguous()
        for name, value in source_weights.items()
    }
    uniform_gpu = {
        name: tensors[f"qsrt_k3.{name}"].to(device).half().contiguous()
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    prepared: dict[str, dict[str, torch.Tensor]] = {}
    for collection in ("activation_fit", "candidate_selection"):
        x_cpu, route_weights_cpu = routed_rows[collection]
        x = x_cpu.to(device)
        hidden = _expert_hidden(
            x, uniform_gpu["gate_proj"], uniform_gpu["up_proj"]
        )
        prepared[collection] = {
            "hidden": hidden,
            "teacher": _teacher_output(
                x,
                source_gpu["gate_proj"],
                source_gpu["up_proj"],
                source_gpu["down_proj"],
            ),
            "route_weights": route_weights_cpu.to(device).float(),
        }

    fit = prepared["activation_fit"]
    selection = prepared["candidate_selection"]
    input_hessian: torch.Tensor | None = None
    input_hessian_record: dict[str, Any]
    if input_metric_policy == "identity":
        input_hessian_record = {"policy": "identity_control"}
    else:
        input_hessian, covariance_record = routed_input_hessian(
            fit["hidden"],
            fit["route_weights"],
            identity_shrinkage=covariance_identity_shrinkage,
        )
        input_hessian_record = {
            "policy": "reconstructed_gate_up_activation_covariance",
            **covariance_record,
            "sha256": tensor_sha256(input_hessian),
        }

    down_spec = next(spec for spec in PROJECTIONS if spec.name == "down_proj")
    input_seed, output_seed = _transform_seeds(layer, down_spec)

    def encode_down(target: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        encoded = encode_uniform_candidate(
            target.detach().cpu(),
            bits=3,
            codebook=CODEBOOK_SQG_XOR_CHEB_T12,
            device=device,
            quantizer_module=quantizer_module,
            input_sign_seed=input_seed,
            output_sign_seed=output_seed,
            sigma_reg=SIGMA_REG,
            tailbite_context=128,
            ldlq_tf32=True,
            input_hessian=input_hessian,
        )
        return encoded.pop("reconstruction").to(device).half(), encoded["payload"]

    # Identity/source is the already-validated uniform-K3 endpoint.  Reusing it
    # makes that cell a byte-exact control rather than a second encoder run.
    if input_metric_policy == "identity":
        source_endpoint = uniform_gpu["down_proj"]
        source_payload = input_record["projections"]["down_proj"]["qsrt_k3"][
            "payload"
        ]
        source_reused = True
    else:
        source_endpoint, source_payload = encode_down(source_weights["down_proj"])
        source_reused = False
    source_output = F.linear(selection["hidden"], source_endpoint).float()
    source_metrics = route_weighted_output_error_statistics(
        selection["teacher"], source_output, selection["route_weights"]
    )

    candidates: list[dict[str, Any]] = []
    selected_endpoint = source_endpoint
    selected_payload = source_payload
    selected_target_sha256 = tensor_sha256(source_weights["down_proj"].float())
    selected_ridge_factor: float | None = None
    accepted_refit = False
    if target_policy == "reconstructed_activation_refit":
        fit_baseline = F.linear(
            fit["hidden"], uniform_gpu["down_proj"]
        ).float()
        residual = fit["teacher"] - fit_baseline
        for ridge_factor in factors:
            correction, solver = solve_down_correction(
                fit["hidden"],
                residual,
                fit["route_weights"],
                ridge_factor=ridge_factor,
            )
            target = uniform_gpu["down_proj"].float() + correction
            endpoint, payload = encode_down(target)
            output = F.linear(selection["hidden"], endpoint).float()
            metrics = route_weighted_output_error_statistics(
                selection["teacher"], output, selection["route_weights"]
            )
            candidates.append(
                {
                    "ridge_factor": ridge_factor,
                    "solver": solver,
                    "continuous_target_sha256": tensor_sha256(target),
                    "payload": payload,
                    "dense_tensor_sha256": tensor_sha256(endpoint),
                    "candidate_selection": metrics,
                    "passes_local_fallback": refit_passes_local_fallback(
                        baseline=source_metrics,
                        candidate=metrics,
                        tail_relative_tolerance=tail_relative_tolerance,
                    ),
                    "endpoint": endpoint,
                }
            )
        passing = [item for item in candidates if item["passes_local_fallback"]]
        if passing:
            selected = min(
                passing,
                key=lambda item: (
                    item["candidate_selection"]["weighted_relative_sse"],
                    item["candidate_selection"]["row_error_cvar1"],
                    item["ridge_factor"],
                ),
            )
            selected_endpoint = selected["endpoint"]
            selected_payload = selected["payload"]
            selected_target_sha256 = selected["continuous_target_sha256"]
            selected_ridge_factor = float(selected["ridge_factor"])
            accepted_refit = True

    selected_output = F.linear(selection["hidden"], selected_endpoint).float()
    selected_metrics = route_weighted_output_error_statistics(
        selection["teacher"], selected_output, selection["route_weights"]
    )
    output_tensors = dict(tensors)
    output_tensors["qsrt_k3.down_proj"] = selected_endpoint.cpu()
    output_path = _dense_expert_path(dest, layer, expert)
    _atomic_save_tensors(output_path, output_tensors)
    candidate_records = [
        {key: value for key, value in item.items() if key != "endpoint"}
        for item in candidates
    ]
    return {
        "kind": f"{INTERVENTION_ARTIFACT_KIND}_expert",
        "complete": True,
        "manifest_sha256": manifest_sha256,
        "layer": layer,
        "expert": expert,
        "construction": construction,
        "dense_endpoint_file": output_path.name,
        "dense_endpoint_file_bytes": output_path.stat().st_size,
        "dense_endpoint_file_sha256": sha256_file(output_path),
        "input_dense_endpoint_sha256": input_record["dense_endpoint_file_sha256"],
        "routed_rows": {
            collection: int(values["hidden"].shape[0])
            for collection, values in prepared.items()
        },
        "input_metric": input_hessian_record,
        "target_policy": target_policy,
        "source_target": {
            "reused_uniform_k3_endpoint": source_reused,
            "payload": source_payload,
            "dense_tensor_sha256": tensor_sha256(source_endpoint),
            "candidate_selection": source_metrics,
        },
        "refit_candidates": candidate_records,
        "accepted_refit": accepted_refit,
        "selected_ridge_factor": selected_ridge_factor,
        "selected_continuous_target_sha256": selected_target_sha256,
        "selected_payload": selected_payload,
        "selected_candidate_selection": selected_metrics,
        "materialized_down_tensor_sha256": tensor_sha256(
            output_tensors["qsrt_k3.down_proj"]
        ),
    }


def run_down_construction_panel(
    *,
    source_root: Path,
    source_inventory_path: Path,
    input_artifact_root: Path,
    capture_root: Path,
    panel_manifest_path: Path,
    dest: Path,
    layer: int,
    expert_count: int,
    panel_offset: int,
    input_metric_policy: str,
    target_policy: str,
    ridge_factors: Sequence[float],
    covariance_identity_shrinkage: float,
    tail_relative_tolerance: float,
    device: torch.device,
    exllamav3_root: Path,
    verify_source_shard_hashes: bool = True,
) -> dict[str, Any]:
    """Build one disjoint expert slice for one down-construction cell."""

    construction = down_construction_name(
        input_metric=input_metric_policy, target=target_policy
    )
    factors = tuple(float(value) for value in ridge_factors)
    frozen = load_frozen_real_weight_panel(panel_manifest_path, layer=layer)
    experts = select_frozen_panel_slice(
        frozen, offset=panel_offset, expert_count=expert_count
    )
    input_artifact = validate_dense_intervention_artifact(input_artifact_root)
    if input_artifact["candidate_tensor_prefix"] != "qsrt_k3":
        raise ValueError("down construction requires a uniform-K3 input artifact")
    if not set(experts).issubset(input_artifact["expert_ids"]):
        raise ValueError("input intervention artifact does not cover the panel slice")
    input_records = {
        int(record["expert"]): record
        for record in input_artifact["report"]["experts"]
    }
    source_inventory = validate_bounded_source_window(
        source_root,
        source_inventory_path,
        panel={layer: experts},
        verify_shard_hashes=verify_source_shard_hashes,
    )
    capture_manifest_path = capture_root / "manifest.json"
    capture_manifest_sha256 = sha256_file(capture_manifest_path)
    routed = _read_capture_rows(capture_root, experts=experts)
    manifest = {
        "kind": f"{INTERVENTION_ARTIFACT_KIND}_manifest",
        "candidate": {
            "profile": "qsrt_sqg_e4m3",
            "uniform_rate": 3,
            "tensor_prefix": "qsrt_k3",
            "variant": construction,
            "input_metric_policy": input_metric_policy,
            "target_policy": target_policy,
            "ridge_factors": list(factors),
            "covariance_identity_shrinkage": covariance_identity_shrinkage,
            "local_tail_relative_tolerance": tail_relative_tolerance,
            "fallback": "source-target endpoint under the same input metric",
        },
        "source": source_inventory,
        "input_intervention_artifact": {
            "root": str(input_artifact_root.resolve()),
            "manifest_sha256": input_artifact["manifest_sha256"],
        },
        "activation_capture": {
            "root": str(capture_root.resolve()),
            "manifest_sha256": capture_manifest_sha256,
        },
        "frozen_panel": {
            "path": frozen["path"],
            "sha256": frozen["sha256"],
            "selected_offset": panel_offset,
            "selected_count": expert_count,
        },
        "panel": {str(layer): list(experts)},
        "resident_endpoint_dtype": "FP16",
        "resident_coordinate_basis": "sealed_R7_permuted_middle_coordinates",
        "device": str(device),
        "fit_numeric_policy": {
            "float32_matmul_precision": "highest",
            "route_weight_dtype": "FP32",
            "teacher_weight_dtype": "BF16",
            "candidate_weight_dtype": "FP16",
            "target_fit_basis": "uniform_K3_reconstructed_gate_up_activations",
            "target_fit_is_independent_of_input_metric": True,
        },
        "evidence_boundary": (
            "activation-fit documents fit metrics and targets; separate candidate-"
            "selection documents apply local mean and routed-row-tail fallback; "
            "only untouched multi-document BF16-reference KLD can accept a "
            "down-construction rule"
        ),
        "model_downloads_required": False,
        "complete_bf16_checkpoint_required": False,
    }
    manifest_sha256 = prepare_destination(dest, manifest, resume=False)
    source = IndexedTensorStore(source_root)
    quantizer_module = load_qsrt_encoder(exllamav3_root)
    install_sqg_quantizer(quantizer_module)
    records: list[dict[str, Any]] = []
    for ordinal, expert in enumerate(experts, start=1):
        started = time.monotonic()
        record = build_down_construction_for_expert(
            source=source,
            input_artifact_root=input_artifact_root,
            input_record=input_records[expert],
            routed_rows={
                collection: routed[collection][expert]
                for collection in routed
            },
            layer=layer,
            expert=expert,
            input_metric_policy=input_metric_policy,
            target_policy=target_policy,
            ridge_factors=factors,
            covariance_identity_shrinkage=covariance_identity_shrinkage,
            tail_relative_tolerance=tail_relative_tolerance,
            device=device,
            quantizer_module=quantizer_module,
            dest=dest,
            manifest_sha256=manifest_sha256,
        )
        record["wall_seconds"] = time.monotonic() - started
        atomic_write_json(_expert_path(dest, layer, expert), record)
        records.append(record)
        print(
            f"[{ordinal:02d}/{len(experts)}] layer {layer} expert {expert}: "
            f"construction={construction} refit={record['accepted_refit']} "
            f"selection={record['selected_candidate_selection']['weighted_relative_sse']:.6g}",
            flush=True,
        )
        torch.cuda.empty_cache()
    report = {
        "kind": INTERVENTION_ARTIFACT_KIND,
        "experiment": DOWN_CONSTRUCTION_COMPARISON_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "layer": layer,
        "construction": construction,
        "expert_count": len(records),
        "accepted_refit_count": sum(
            bool(record["accepted_refit"]) for record in records
        ),
        "panel": manifest["panel"],
        "dense_endpoint_bytes": sum(
            int(record["dense_endpoint_file_bytes"]) for record in records
        ),
        "experts": records,
        "evidence_boundary": manifest["evidence_boundary"],
    }
    atomic_write_json(dest / "report.json", report)
    return report


__all__ = [
    "DOWN_CONSTRUCTION_COMPARISON_KIND",
    "INPUT_METRIC_POLICIES",
    "TARGET_POLICIES",
    "down_construction_name",
    "refit_passes_local_fallback",
    "route_weighted_output_error_statistics",
    "run_down_construction_panel",
]

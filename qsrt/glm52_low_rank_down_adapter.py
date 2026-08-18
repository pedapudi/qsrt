"""Fit bounded activation-weighted low-rank corrections to GLM-5.2 down projections.

The correction is fitted around an existing dense intervention artifact.  Its
input rows are reconstructed by that artifact's gate and up projections, and
its target is the complete official BF16 expert output.  The stored BF16
factors are also materialized into the dense FP16 endpoint used by the bounded
KLD harness.  That dense endpoint is a mechanism-screening view; a release
artifact still needs a factor-aware container and serving kernel.
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
from qsrt.glm52_down_construction import route_weighted_output_error_statistics
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
from qsrt.glm52_expert_intervention_runtime import validate_dense_intervention_artifact
from qsrt.glm52_pilot import (
    PROJECTIONS,
    IndexedTensorStore,
    _expert_path,
    atomic_write_json,
    prepare_destination,
    source_tensor_name,
)
from qsrt.glm52_real_weight_benchmark import (
    load_frozen_real_weight_panel,
    select_frozen_panel_slice,
    validate_bounded_source_window,
)
from qsrt.low_rank_adapters import fit_weighted_error_adapter
from qsrt.qsrt_codec_pilot import tensor_sha256


GLM52_LOW_RANK_DOWN_EXPERIMENT = "qsrt_glm52_activation_weighted_down_adapter_v1"
BASE_CONSTRUCTIONS = (
    "uniform_k3",
    "reconstructed_activation_down_refit",
)
FACTOR_DTYPE = torch.bfloat16


def _load_base_tensors(
    root: Path, record: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    path = root / "experts" / str(record["dense_endpoint_file"])
    if sha256_file(path) != record["dense_endpoint_file_sha256"]:
        raise ValueError(f"base endpoint for expert {record['expert']} failed closure")
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


def materialize_bf16_down_adapter(
    base_down: torch.Tensor,
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Balance storage at BF16 and return its dense FP16 screening endpoint."""

    if base_down.ndim != 2 or factor_a.ndim != 2 or factor_b.ndim != 2:
        raise ValueError("base down weights and adapter factors must be matrices")
    if factor_a.shape[1] != factor_b.shape[1]:
        raise ValueError("adapter factors must have the same rank")
    if base_down.shape != (factor_b.shape[0], factor_a.shape[0]):
        raise ValueError("adapter factors do not match the down matrix")
    a = factor_a.to(dtype=FACTOR_DTYPE).contiguous()
    b = factor_b.to(dtype=FACTOR_DTYPE).contiguous()
    correction = b.float() @ a.float().T
    dense = (base_down.float() + correction).to(torch.float16).contiguous()
    if not bool(torch.isfinite(dense).all()):
        raise ValueError("materialized adapter endpoint contains non-finite values")
    return a, b, dense


@torch.no_grad()
def fit_functional_down_adapter(
    *,
    base_down: torch.Tensor,
    fit_hidden: torch.Tensor,
    fit_teacher: torch.Tensor,
    fit_route_weights: torch.Tensor,
    selection_hidden: torch.Tensor,
    selection_teacher: torch.Tensor,
    selection_route_weights: torch.Tensor,
    rank: int,
    ridge_factors: Sequence[float],
    oversampling: int,
    power_iterations: int,
    batch_rows: int,
    seed: int,
) -> dict[str, Any]:
    """Fit and locally rank BF16 adapter candidates for one frozen base matrix."""

    factors = tuple(float(value) for value in ridge_factors)
    if not factors or any(not math.isfinite(value) or value <= 0.0 for value in factors):
        raise ValueError("ridge factors must be finite positive values")
    if rank <= 0:
        raise ValueError("adapter rank must be positive")
    baseline_fit = F.linear(fit_hidden, base_down).float()
    baseline_selection = F.linear(selection_hidden, base_down).float()
    baseline_metrics = route_weighted_output_error_statistics(
        selection_teacher, baseline_selection, selection_route_weights
    )
    candidates: list[dict[str, Any]] = []
    for index, ridge_factor in enumerate(factors):
        full_correction, solver = solve_down_correction(
            fit_hidden,
            fit_teacher - baseline_fit,
            fit_route_weights,
            ridge_factor=ridge_factor,
        )
        fitted = fit_weighted_error_adapter(
            full_correction,
            fit_hidden,
            fit_route_weights,
            rank=rank,
            oversampling=oversampling,
            power_iterations=power_iterations,
            batch_rows=batch_rows,
            seed=seed + index,
        )
        factor_a, factor_b, dense = materialize_bf16_down_adapter(
            base_down, fitted.a, fitted.b
        )
        fit_output = F.linear(fit_hidden, dense).float()
        selection_output = F.linear(selection_hidden, dense).float()
        candidates.append(
            {
                "ridge_factor": ridge_factor,
                "solver": solver,
                "factor_a": factor_a,
                "factor_b": factor_b,
                "dense": dense,
                "fit_metrics": route_weighted_output_error_statistics(
                    fit_teacher, fit_output, fit_route_weights
                ),
                "selection_metrics": route_weighted_output_error_statistics(
                    selection_teacher,
                    selection_output,
                    selection_route_weights,
                ),
                "weighted_low_rank_objective_total": fitted.objective_total,
                "weighted_low_rank_objective_captured": float(
                    fitted.objective_captured[-1].item()
                ),
                "singular_values": [
                    float(value) for value in fitted.singular_values.tolist()
                ],
            }
        )
    selected = min(
        candidates,
        key=lambda item: (
            item["selection_metrics"]["weighted_relative_sse"],
            item["selection_metrics"]["row_error_cvar1"],
            item["ridge_factor"],
        ),
    )
    return {
        "baseline_selection_metrics": baseline_metrics,
        "selected": selected,
        "candidates": candidates,
    }


@torch.no_grad()
def build_low_rank_down_for_expert(
    *,
    source: IndexedTensorStore,
    input_artifact_root: Path,
    input_record: Mapping[str, Any],
    routed_rows: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    layer: int,
    expert: int,
    base_construction: str,
    rank: int,
    ridge_factors: Sequence[float],
    oversampling: int,
    power_iterations: int,
    batch_rows: int,
    seed: int,
    device: torch.device,
    dest: Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Fit one down-only adapter and materialize its bounded dense endpoint."""

    tensors = _load_base_tensors(input_artifact_root, input_record)
    source_weights = {
        spec.name: source.get(source_tensor_name(layer, expert, spec.name))
        for spec in PROJECTIONS
    }
    source_gpu = {
        name: value.to(device=device, dtype=torch.bfloat16).contiguous()
        for name, value in source_weights.items()
    }
    base_gpu = {
        name: tensors[f"qsrt_k3.{name}"].to(device=device, dtype=torch.float16)
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    prepared: dict[str, dict[str, torch.Tensor]] = {}
    for collection in ("activation_fit", "candidate_selection"):
        x_cpu, route_weights_cpu = routed_rows[collection]
        x = x_cpu.to(device)
        prepared[collection] = {
            "hidden": _expert_hidden(x, base_gpu["gate_proj"], base_gpu["up_proj"]),
            "teacher": _teacher_output(
                x,
                source_gpu["gate_proj"],
                source_gpu["up_proj"],
                source_gpu["down_proj"],
            ),
            "route_weights": route_weights_cpu.to(device=device, dtype=torch.float32),
        }
    fit = prepared["activation_fit"]
    selection = prepared["candidate_selection"]
    result = fit_functional_down_adapter(
        base_down=base_gpu["down_proj"],
        fit_hidden=fit["hidden"],
        fit_teacher=fit["teacher"],
        fit_route_weights=fit["route_weights"],
        selection_hidden=selection["hidden"],
        selection_teacher=selection["teacher"],
        selection_route_weights=selection["route_weights"],
        rank=rank,
        ridge_factors=ridge_factors,
        oversampling=oversampling,
        power_iterations=power_iterations,
        batch_rows=batch_rows,
        seed=seed,
    )
    selected = result["selected"]
    output_tensors = dict(tensors)
    output_tensors["adapter.down.base"] = tensors["qsrt_k3.down_proj"].clone()
    output_tensors["qsrt_k3.down_proj"] = selected["dense"].cpu()
    output_tensors["adapter.down.a"] = selected["factor_a"].cpu()
    output_tensors["adapter.down.b"] = selected["factor_b"].cpu()
    output_path = _dense_expert_path(dest, layer, expert)
    _atomic_save_tensors(output_path, output_tensors)
    factor_bytes = sum(
        value.numel() * value.element_size()
        for value in (selected["factor_a"], selected["factor_b"])
    )
    candidate_records = []
    for candidate in result["candidates"]:
        candidate_records.append(
            {
                key: value
                for key, value in candidate.items()
                if key not in {"factor_a", "factor_b", "dense"}
            }
        )
    baseline = result["baseline_selection_metrics"]
    selected_metrics = selected["selection_metrics"]
    return {
        "kind": f"{INTERVENTION_ARTIFACT_KIND}_expert",
        "complete": True,
        "manifest_sha256": manifest_sha256,
        "layer": layer,
        "expert": expert,
        "experiment": GLM52_LOW_RANK_DOWN_EXPERIMENT,
        "base_construction": base_construction,
        "rank": rank,
        "factor_dtype": "BF16",
        "factor_shapes": {
            "a": list(selected["factor_a"].shape),
            "b": list(selected["factor_b"].shape),
        },
        "logical_adapter_bytes": factor_bytes,
        "selected_ridge_factor": float(selected["ridge_factor"]),
        "factor_a_sha256": tensor_sha256(selected["factor_a"]),
        "factor_b_sha256": tensor_sha256(selected["factor_b"]),
        "base_down_sha256": tensor_sha256(tensors["qsrt_k3.down_proj"]),
        "materialized_down_sha256": tensor_sha256(selected["dense"]),
        "input_dense_endpoint_sha256": input_record["dense_endpoint_file_sha256"],
        "dense_endpoint_file": output_path.name,
        "dense_endpoint_file_bytes": output_path.stat().st_size,
        "dense_endpoint_file_sha256": sha256_file(output_path),
        "routed_rows": {
            collection: int(values["hidden"].shape[0])
            for collection, values in prepared.items()
        },
        "baseline_candidate_selection": baseline,
        "selected_candidate_selection": selected_metrics,
        "selection_weighted_relative_sse_reduction": 1.0
        - float(selected_metrics["weighted_relative_sse"])
        / float(baseline["weighted_relative_sse"]),
        "ridge_candidates": candidate_records,
    }


def run_low_rank_down_panel(
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
    base_construction: str,
    rank: int,
    ridge_factors: Sequence[float],
    oversampling: int,
    power_iterations: int,
    batch_rows: int,
    seed: int,
    device: torch.device,
    verify_source_shard_hashes: bool = True,
) -> dict[str, Any]:
    """Build one disjoint expert slice for the down-only adapter screen."""

    if base_construction not in BASE_CONSTRUCTIONS:
        raise ValueError(f"unsupported base construction {base_construction!r}")
    if rank not in (2, 4):
        raise ValueError("the bounded screen supports rank two or rank four")
    frozen = load_frozen_real_weight_panel(panel_manifest_path, layer=layer)
    experts = select_frozen_panel_slice(
        frozen, offset=panel_offset, expert_count=expert_count
    )
    input_artifact = validate_dense_intervention_artifact(input_artifact_root)
    if input_artifact["candidate_tensor_prefix"] != "qsrt_k3":
        raise ValueError("low-rank fitting requires a QSRT-K3 base artifact")
    if not set(experts).issubset(input_artifact["expert_ids"]):
        raise ValueError("base artifact does not cover the requested panel slice")
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
    capture_manifest = json.loads(capture_manifest_path.read_text())
    routed = _read_capture_rows(capture_root, experts=experts)
    manifest = {
        "kind": f"{INTERVENTION_ARTIFACT_KIND}_manifest",
        "source": source_inventory,
        "input_intervention_artifact": {
            "manifest_sha256": input_artifact["manifest_sha256"],
            "report_sha256": sha256_file(input_artifact_root / "report.json"),
            "candidate_tensor_prefix": input_artifact["candidate_tensor_prefix"],
            "base_construction": base_construction,
        },
        "activation_capture": {
            "manifest_sha256": sha256_file(capture_manifest_path),
            "corpus_plan_sha256": capture_manifest["corpus_plan_sha256"],
            "collections": capture_manifest["collections"],
        },
        "frozen_panel": {
            "path": frozen["path"],
            "sha256": frozen["sha256"],
            "selected_offset": panel_offset,
            "selected_count": expert_count,
        },
        "panel": {str(layer): list(experts)},
        "candidate": {
            "profile": "qsrt_sqg_e4m3_plus_activation_weighted_down_adapter",
            "tensor_prefix": "qsrt_k3",
            "base_construction": base_construction,
            "adapter_matrix": "down_proj",
            "adapter_rank": rank,
            "factor_dtype": "BF16",
            "screening_endpoint_dtype": "FP16",
            "factorized_runtime_contract": {
                "base_down_tensor": "adapter.down.base",
                "factor_a_tensor": "adapter.down.a",
                "factor_b_tensor": "adapter.down.b",
                "preferred_execution": (
                    "materialize_fp16_down_from_base_and_factors_at_expert_load"
                ),
                "rejected_control_execution": "base_down_plus_two_bf16_factor_gemms",
            },
        },
        "fit_numeric_policy": {
            "ridge_factors": [float(value) for value in ridge_factors],
            "oversampling": oversampling,
            "power_iterations": power_iterations,
            "batch_rows": batch_rows,
            "seed": seed,
            "route_weight_power": 2,
        },
        "resident_endpoint_dtype": "FP16",
        "resident_coordinate_basis": (
            "sealed per-expert R7 middle-coordinate permutation; functionally "
            "equivalent to the official source expert"
        ),
        "device": str(device),
        "evidence_boundary": (
            "the artifact stores the BF16 factors, their K3 base matrix, and a "
            "materialized FP16 endpoint for bounded model-KLD screening; checkpoint "
            "size and serving claims require a runtime that reconstructs the endpoint "
            "from the stored base and factors"
        ),
        "model_downloads_required": False,
        "complete_bf16_checkpoint_required": False,
    }
    manifest_sha256 = prepare_destination(dest, manifest, resume=False)
    source = IndexedTensorStore(source_root)
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for ordinal, expert in enumerate(experts, start=1):
        expert_started = time.monotonic()
        record = build_low_rank_down_for_expert(
            source=source,
            input_artifact_root=input_artifact_root,
            input_record=input_records[expert],
            routed_rows={
                collection: routed[collection][expert]
                for collection in ("activation_fit", "candidate_selection")
            },
            layer=layer,
            expert=expert,
            base_construction=base_construction,
            rank=rank,
            ridge_factors=ridge_factors,
            oversampling=oversampling,
            power_iterations=power_iterations,
            batch_rows=batch_rows,
            seed=seed + expert * 17,
            device=device,
            dest=dest,
            manifest_sha256=manifest_sha256,
        )
        record["wall_seconds"] = time.monotonic() - expert_started
        atomic_write_json(_expert_path(dest, layer, expert), record)
        records.append(record)
        print(
            f"[{ordinal:02d}/{len(experts)}] layer {layer} expert {expert}: "
            f"selection reduction="
            f"{record['selection_weighted_relative_sse_reduction']:.6%}",
            flush=True,
        )
    report = {
        "kind": INTERVENTION_ARTIFACT_KIND,
        "experiment": GLM52_LOW_RANK_DOWN_EXPERIMENT,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "layer": layer,
        "expert_count": len(records),
        "panel": manifest["panel"],
        "base_construction": base_construction,
        "rank": rank,
        "logical_adapter_bytes": sum(
            int(record["logical_adapter_bytes"]) for record in records
        ),
        "dense_endpoint_bytes": sum(
            int(record["dense_endpoint_file_bytes"]) for record in records
        ),
        "experts": records,
        "wall_seconds": time.monotonic() - started,
        "evidence_boundary": manifest["evidence_boundary"],
    }
    atomic_write_json(dest / "report.json", report)
    return report


__all__ = [
    "BASE_CONSTRUCTIONS",
    "GLM52_LOW_RANK_DOWN_EXPERIMENT",
    "build_low_rank_down_for_expert",
    "fit_functional_down_adapter",
    "materialize_bf16_down_adapter",
    "run_low_rank_down_panel",
]

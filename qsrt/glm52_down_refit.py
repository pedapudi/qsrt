"""Refit GLM-5.2 down projections after reconstructing gate and up weights.

The fit target is the official BF16 expert function from the bounded layer-3
source window.  Inputs, route IDs, and applied route weights come from the
document-disjoint resident-EXL3 capture.  A ridge-regularized correction is
fit around the existing QSRT-K3 down endpoint, selected on different articles,
and encoded through the unchanged K3 trellis.  The original down endpoint is
retained whenever the re-encoded correction does not improve selection error.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open

from qsrt.correctness import sha256_file
from qsrt.exl3_loader import load_qsrt_encoder
from qsrt.exl3_reference import CODEBOOK_SQG_XOR_CHEB_T12
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
from qsrt.ldlq import SIGMA_REG
from qsrt.qsrt_codec_pilot import encode_uniform_candidate, tensor_sha256
from qsrt.sqg_quantizer import install_sqg_quantizer


DOWN_REFIT_KIND = "qsrt_glm52_reconstructed_activation_down_refit_v1"


def weighted_relative_sse(
    teacher: torch.Tensor,
    candidate: torch.Tensor,
    route_weights: torch.Tensor,
) -> float:
    """Measure squared error after applying each routed expert weight."""

    if teacher.shape != candidate.shape or teacher.ndim != 2:
        raise ValueError("teacher and candidate outputs must share one matrix shape")
    if route_weights.ndim != 1 or route_weights.numel() != teacher.shape[0]:
        raise ValueError("route weights must contain one value per output row")
    weight = route_weights.double().unsqueeze(1)
    error = (teacher.double() - candidate.double()) * weight
    reference = teacher.double() * weight
    return float(
        error.square().sum().div(reference.square().sum().clamp_min(1e-30)).item()
    )


def solve_down_correction(
    reconstructed_hidden: torch.Tensor,
    teacher_residual: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    ridge_factor: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Solve a route-weighted ridge correction in the reconstructed basis."""

    if reconstructed_hidden.ndim != 2 or teacher_residual.ndim != 2:
        raise ValueError("ridge inputs must be matrices")
    if reconstructed_hidden.shape[0] != teacher_residual.shape[0]:
        raise ValueError("ridge hidden and residual row counts differ")
    if route_weights.ndim != 1 or route_weights.numel() != reconstructed_hidden.shape[0]:
        raise ValueError("ridge route weights must match the row count")
    if not math.isfinite(ridge_factor) or ridge_factor <= 0.0:
        raise ValueError("ridge_factor must be finite and positive")
    hidden = reconstructed_hidden.float()
    residual = teacher_residual.float()
    weights = route_weights.float().clamp_min(0.0).unsqueeze(1)
    weighted_hidden = hidden * weights
    weighted_residual = residual * weights
    gram = weighted_hidden.T @ weighted_hidden
    gram_scale = float(gram.diagonal().mean().item())
    if not math.isfinite(gram_scale) or gram_scale <= 0.0:
        raise ValueError("routed reconstructed activations have a degenerate Gram matrix")
    ridge = ridge_factor * gram_scale
    gram.diagonal().add_(ridge)
    right_hand_side = weighted_hidden.T @ weighted_residual
    factor, information = torch.linalg.cholesky_ex(gram)
    if int(information.max().item()) != 0:
        raise RuntimeError("ridge Gram matrix was not positive definite")
    correction_transposed = torch.cholesky_solve(right_hand_side, factor)
    return correction_transposed.T.contiguous(), {
        "ridge_factor": float(ridge_factor),
        "gram_diagonal_mean": gram_scale,
        "ridge_absolute": ridge,
    }


def _read_capture_rows(
    capture_root: Path,
    *,
    experts: Sequence[int],
    model_layer: int = 3,
) -> dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    manifest = json.loads((capture_root / "manifest.json").read_text())
    if (
        manifest.get("schema") != "qsrt_glm52_layer_input_capture_manifest"
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("model_layer") != model_layer
    ):
        raise ValueError("layer-input capture manifest identity mismatch")
    plan_sha256 = manifest.get("corpus_plan_sha256")
    if (
        not isinstance(plan_sha256, str)
        or len(plan_sha256) != 64
        or any(character not in "0123456789abcdef" for character in plan_sha256)
    ):
        raise ValueError("layer-input capture has an invalid corpus-plan identity")
    selected = tuple(int(expert) for expert in experts)
    hidden: dict[str, dict[int, list[torch.Tensor]]] = {
        collection: {expert: [] for expert in selected}
        for collection in ("activation_fit", "candidate_selection")
    }
    weights: dict[str, dict[int, list[torch.Tensor]]] = {
        collection: {expert: [] for expert in selected}
        for collection in hidden
    }
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("layer-input capture manifest has no records")
    seen_files: set[str] = set()
    seen_generations: set[int] = set()
    collection_counts = {collection: 0 for collection in hidden}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("layer-input capture record must be an object")
        collection = record["collection"]
        if collection not in hidden:
            raise ValueError(f"unknown capture collection {collection!r}")
        filename = record.get("capture_file")
        generation = record.get("control_generation")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or filename in seen_files
        ):
            raise ValueError("capture filenames must be unique basenames")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
            or generation in seen_generations
        ):
            raise ValueError("capture control generations must be unique integers")
        seen_files.add(filename)
        seen_generations.add(generation)
        collection_counts[collection] += 1
        path = capture_root / filename
        if (
            path.stat().st_size != record["capture_file_bytes"]
            or sha256_file(path) != record["capture_file_sha256"]
        ):
            raise ValueError(f"capture file {path.name} failed byte closure")
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata != {
                "schema": "qsrt_glm52_layer_input_capture_v1",
                "model_layer": str(model_layer),
                "control_generation": str(generation),
                "corpus_plan_sha256": plan_sha256,
            }:
                raise ValueError(f"capture file {path.name} metadata mismatch")
            x = handle.get_tensor("hidden_states")
            ids = handle.get_tensor("topk_ids")
            route_weights = handle.get_tensor("topk_weights")
        if (
            x.dtype != torch.bfloat16
            or x.ndim != 2
            or x.shape[1] != 6144
            or ids.dtype != torch.int32
            or tuple(ids.shape) != (x.shape[0], 8)
            or route_weights.dtype != torch.float32
            or route_weights.shape != ids.shape
            or int(record.get("token_count", -1)) != x.shape[0]
        ):
            raise ValueError(f"capture file {path.name} tensor contract mismatch")
        if (
            not bool(torch.isfinite(x).all())
            or not bool(torch.isfinite(route_weights).all())
            or bool((route_weights < 0.0).any())
            or bool((ids < 0).any())
            or bool((ids >= 256).any())
        ):
            raise ValueError(f"capture file {path.name} contains invalid values")
        for expert in selected:
            positions = (ids == expert).nonzero(as_tuple=False)
            if positions.numel() == 0:
                continue
            token_ids = positions[:, 0]
            route_ids = positions[:, 1]
            if torch.unique(token_ids).numel() != token_ids.numel():
                raise ValueError(f"capture routes expert {expert} more than once per token")
            hidden[collection][expert].append(x.index_select(0, token_ids))
            weights[collection][expert].append(
                route_weights[token_ids, route_ids].float()
            )
    if manifest.get("collections") != collection_counts:
        raise ValueError("layer-input capture collection counts do not match records")
    result: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    for collection in hidden:
        result[collection] = {}
        for expert in selected:
            if not hidden[collection][expert]:
                raise ValueError(
                    f"expert {expert} has no routed rows in {collection}"
                )
            result[collection][expert] = (
                torch.cat(hidden[collection][expert], dim=0).contiguous(),
                torch.cat(weights[collection][expert], dim=0).contiguous(),
            )
    return result


def _expert_hidden(
    x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> torch.Tensor:
    x = x.to(device=gate.device, dtype=gate.dtype)
    return (F.silu(F.linear(x, gate)) * F.linear(x, up)).contiguous()


def _teacher_output(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    x = x.to(device=gate.device, dtype=gate.dtype)
    hidden = F.silu(F.linear(x, gate)) * F.linear(x, up)
    return F.linear(hidden, down).float()


@torch.no_grad()
def refit_one_expert(
    *,
    source: IndexedTensorStore,
    input_artifact_root: Path,
    input_record: Mapping[str, Any],
    routed_rows: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    layer: int,
    expert: int,
    ridge_factors: Sequence[float],
    device: torch.device,
    quantizer_module: Any,
    dest: Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Fit, select, re-encode, and conditionally accept one down projection."""

    input_path = input_artifact_root / "experts" / input_record["dense_endpoint_file"]
    if sha256_file(input_path) != input_record["dense_endpoint_file_sha256"]:
        raise ValueError(f"input endpoint for expert {expert} failed byte closure")
    with safe_open(input_path, framework="pt", device="cpu") as handle:
        tensors = {
            key: handle.get_tensor(key)
            for endpoint in ("exl3", "qsrt_k3")
            for key in (
                f"{endpoint}.gate_proj",
                f"{endpoint}.up_proj",
                f"{endpoint}.down_proj",
            )
        }
    source_weights = {
        spec.name: source.get(source_tensor_name(layer, expert, spec.name))
        for spec in PROJECTIONS
    }
    source_gpu = {
        name: value.to(device=device, dtype=torch.bfloat16).contiguous()
        for name, value in source_weights.items()
    }
    candidate_gpu = {
        name: value.to(device=device, dtype=torch.float16).contiguous()
        for name, value in (
            ("gate_proj", tensors["qsrt_k3.gate_proj"]),
            ("up_proj", tensors["qsrt_k3.up_proj"]),
            ("down_proj", tensors["qsrt_k3.down_proj"]),
        )
    }
    prepared: dict[str, dict[str, torch.Tensor]] = {}
    for collection in ("activation_fit", "candidate_selection"):
        x_cpu, route_weight_cpu = routed_rows[collection]
        x = x_cpu.to(device)
        teacher = _teacher_output(
            x,
            source_gpu["gate_proj"],
            source_gpu["up_proj"],
            source_gpu["down_proj"],
        )
        hidden = _expert_hidden(
            x, candidate_gpu["gate_proj"], candidate_gpu["up_proj"]
        )
        baseline = F.linear(hidden, candidate_gpu["down_proj"]).float()
        prepared[collection] = {
            "hidden": hidden,
            "teacher": teacher,
            "baseline": baseline,
            "route_weights": route_weight_cpu.to(device).float(),
        }

    baseline_metrics = {
        collection: weighted_relative_sse(
            values["teacher"], values["baseline"], values["route_weights"]
        )
        for collection, values in prepared.items()
    }
    fit = prepared["activation_fit"]
    selection = prepared["candidate_selection"]
    residual = fit["teacher"] - fit["baseline"]
    dense_candidates: list[dict[str, Any]] = []
    baseline_down = candidate_gpu["down_proj"].float()
    for ridge_factor in ridge_factors:
        correction, solver = solve_down_correction(
            fit["hidden"], residual, fit["route_weights"], ridge_factor=ridge_factor
        )
        refitted_down = baseline_down + correction
        fit_output = F.linear(fit["hidden"].float(), refitted_down).float()
        selection_output = F.linear(
            selection["hidden"].float(), refitted_down
        ).float()
        dense_candidates.append(
            {
                "ridge_factor": float(ridge_factor),
                "solver": solver,
                "down": refitted_down,
                "activation_fit_weighted_relative_sse": weighted_relative_sse(
                    fit["teacher"], fit_output, fit["route_weights"]
                ),
                "candidate_selection_weighted_relative_sse": weighted_relative_sse(
                    selection["teacher"],
                    selection_output,
                    selection["route_weights"],
                ),
            }
        )
    selected = min(
        dense_candidates,
        key=lambda item: (
            item["candidate_selection_weighted_relative_sse"],
            item["ridge_factor"],
        ),
    )
    down_spec = next(spec for spec in PROJECTIONS if spec.name == "down_proj")
    input_seed, output_seed = _transform_seeds(layer, down_spec)
    encoded = encode_uniform_candidate(
        selected["down"].cpu(),
        bits=3,
        codebook=CODEBOOK_SQG_XOR_CHEB_T12,
        device=device,
        quantizer_module=quantizer_module,
        input_sign_seed=input_seed,
        output_sign_seed=output_seed,
        sigma_reg=SIGMA_REG,
        tailbite_context=128,
        ldlq_tf32=True,
    )
    encoded_down = encoded.pop("reconstruction").to(device).half()
    encoded_metrics = {
        collection: weighted_relative_sse(
            values["teacher"],
            F.linear(values["hidden"], encoded_down).float(),
            values["route_weights"],
        )
        for collection, values in prepared.items()
    }
    tolerance = max(1e-15, baseline_metrics["candidate_selection"] * 1e-9)
    accepted = (
        encoded_metrics["candidate_selection"]
        < baseline_metrics["candidate_selection"] - tolerance
    )
    output_tensors = dict(tensors)
    if accepted:
        output_tensors["qsrt_k3.down_proj"] = encoded_down.cpu()
    output_path = _dense_expert_path(dest, layer, expert)
    _atomic_save_tensors(output_path, output_tensors)
    candidate_summaries = [
        {key: value for key, value in item.items() if key != "down"}
        for item in dense_candidates
    ]
    return {
        "kind": f"{INTERVENTION_ARTIFACT_KIND}_expert",
        "complete": True,
        "manifest_sha256": manifest_sha256,
        "layer": layer,
        "expert": expert,
        "dense_endpoint_file": output_path.name,
        "dense_endpoint_file_bytes": output_path.stat().st_size,
        "dense_endpoint_file_sha256": sha256_file(output_path),
        "input_dense_endpoint_sha256": input_record["dense_endpoint_file_sha256"],
        "accepted": accepted,
        "routed_rows": {
            collection: int(values["hidden"].shape[0])
            for collection, values in prepared.items()
        },
        "baseline_weighted_relative_sse": baseline_metrics,
        "dense_refit_candidates": candidate_summaries,
        "selected_ridge_factor": selected["ridge_factor"],
        "reencoded_weighted_relative_sse": encoded_metrics,
        "reencoded_down_payload": encoded["payload"],
        "materialized_down_tensor_sha256": tensor_sha256(
            output_tensors["qsrt_k3.down_proj"]
        ),
    }


def run_down_refit_panel(
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
    ridge_factors: Sequence[float],
    device: torch.device,
    exllamav3_root: Path,
    verify_source_shard_hashes: bool = True,
) -> dict[str, Any]:
    """Run a disjoint expert slice of reconstructed-activation down refits."""

    factors = tuple(float(value) for value in ridge_factors)
    if not factors or any(not math.isfinite(value) or value <= 0 for value in factors):
        raise ValueError("ridge factors must be finite positive values")
    frozen = load_frozen_real_weight_panel(panel_manifest_path, layer=layer)
    experts = select_frozen_panel_slice(
        frozen, offset=panel_offset, expert_count=expert_count
    )
    input_artifact = validate_dense_intervention_artifact(input_artifact_root)
    if not set(experts).issubset(input_artifact["expert_ids"]):
        raise ValueError("input intervention artifact does not cover the refit panel")
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
    routed = _read_capture_rows(
        capture_root, experts=experts, model_layer=layer
    )
    manifest = {
        "kind": f"{INTERVENTION_ARTIFACT_KIND}_manifest",
        "candidate": {
            "profile": "qsrt_sqg_e4m3",
            "uniform_rate": 3,
            "variant": "reconstructed_activation_down_refit",
            "ridge_factors": list(factors),
            "fallback": "retain original QSRT-K3 down endpoint",
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
        },
        "evidence_boundary": (
            "down refits use document-disjoint fit and selection activations; "
            "only untouched BF16-reference full-model KLD can accept the panel"
        ),
        "model_downloads_required": False,
        "complete_bf16_checkpoint_required": False,
    }
    manifest_sha256 = prepare_destination(dest, manifest, resume=False)
    source = IndexedTensorStore(source_root)
    quantizer_module = load_qsrt_encoder(exllamav3_root)
    install_sqg_quantizer(quantizer_module)
    records: list[dict[str, Any]] = []
    for index, expert in enumerate(experts, start=1):
        started = time.monotonic()
        record = refit_one_expert(
            source=source,
            input_artifact_root=input_artifact_root,
            input_record=input_records[expert],
            routed_rows={
                collection: routed[collection][expert]
                for collection in routed
            },
            layer=layer,
            expert=expert,
            ridge_factors=factors,
            device=device,
            quantizer_module=quantizer_module,
            dest=dest,
            manifest_sha256=manifest_sha256,
        )
        record["wall_seconds"] = time.monotonic() - started
        atomic_write_json(_expert_path(dest, layer, expert), record)
        records.append(record)
        print(
            f"[{index:02d}/{len(experts)}] layer {layer} expert {expert}: "
            f"accepted={record['accepted']} selection="
            f"{record['baseline_weighted_relative_sse']['candidate_selection']:.6g}->"
            f"{record['reencoded_weighted_relative_sse']['candidate_selection']:.6g}",
            flush=True,
        )
        torch.cuda.empty_cache()
    report = {
        "kind": INTERVENTION_ARTIFACT_KIND,
        "experiment": DOWN_REFIT_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "layer": layer,
        "expert_count": len(records),
        "accepted_expert_count": sum(bool(record["accepted"]) for record in records),
        "panel": manifest["panel"],
        "dense_endpoint_bytes": sum(record["dense_endpoint_file_bytes"] for record in records),
        "experts": records,
        "evidence_boundary": manifest["evidence_boundary"],
    }
    atomic_write_json(dest / "report.json", report)
    return report


__all__ = [
    "DOWN_REFIT_KIND",
    "run_down_refit_panel",
    "solve_down_correction",
    "weighted_relative_sse",
]

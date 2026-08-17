"""Test routed-input curvature as a trellis path-selection control on GLM-5.2.

This experiment replaces the identity input metric used by the initial GLM
codec benchmark with expert-routed activation covariances.  It is deliberately
one-sided: it measures the established activation-aware quantization control,
not the full downstream-loss curvature required by model-preserving adaptive
rounding.  Candidate matrices are fit on activation-fit articles and selected
by complete expert-output error on separate candidate-selection articles.
"""

from __future__ import annotations

import itertools
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
    weighted_relative_sse,
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
from qsrt.ldlq import SIGMA_REG
from qsrt.qsrt_codec_pilot import encode_uniform_candidate, tensor_sha256
from qsrt.sqg_quantizer import install_sqg_quantizer


ROUTED_INPUT_CURVATURE_KIND = "qsrt_glm52_routed_input_curvature_control_v1"


def upstream_curvature_basis_name(
    *, gate_curvature: bool, up_curvature: bool
) -> str:
    """Name the gate/up reconstruction pair used to build a down metric."""

    if not isinstance(gate_curvature, bool) or not isinstance(up_curvature, bool):
        raise TypeError("gate_curvature and up_curvature must be booleans")
    return (
        f"gate_{'curvature' if gate_curvature else 'baseline'}_"
        f"up_{'curvature' if up_curvature else 'baseline'}"
    )


def routed_input_hessian(
    values: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    identity_shrinkage: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build a route-weighted covariance with explicit identity shrinkage."""

    if values.ndim != 2 or route_weights.ndim != 1:
        raise ValueError("curvature values must be a matrix with one weight per row")
    if route_weights.numel() != values.shape[0]:
        raise ValueError("curvature route weights do not match the row count")
    if (
        not math.isfinite(identity_shrinkage)
        or not 0.0 < identity_shrinkage <= 1.0
    ):
        raise ValueError("identity_shrinkage must be in (0, 1]")
    weights = route_weights.float().clamp_min(0.0)
    mass = weights.square().sum()
    if float(mass.item()) <= 0.0:
        raise ValueError("curvature route weights have zero squared mass")
    weighted = values.float() * weights.unsqueeze(1)
    hessian = (weighted.T @ weighted).div_(mass)
    diagonal_mean = float(hessian.diagonal().mean().item())
    if not math.isfinite(diagonal_mean) or diagonal_mean <= 0.0:
        raise ValueError("curvature covariance has a nonpositive mean diagonal")
    hessian.mul_(1.0 - identity_shrinkage)
    hessian.diagonal().add_(identity_shrinkage * diagonal_mean)
    return hessian.contiguous(), {
        "row_count": int(values.shape[0]),
        "squared_route_mass": float(mass.item()),
        "unshrunk_diagonal_mean": diagonal_mean,
        "identity_shrinkage": float(identity_shrinkage),
    }


def _candidate_expert_output(
    x: torch.Tensor,
    *,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    hidden = _expert_hidden(x, gate, up)
    return F.linear(hidden, down).float()


@torch.no_grad()
def encode_curvature_candidate_for_expert(
    *,
    source: IndexedTensorStore,
    input_artifact_root: Path,
    input_record: Mapping[str, Any],
    routed_rows: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    layer: int,
    expert: int,
    identity_shrinkage: float,
    device: torch.device,
    quantizer_module: Any,
    dest: Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Encode three routed-curvature matrices and select a complete expert."""

    input_path = input_artifact_root / "experts" / input_record["dense_endpoint_file"]
    if sha256_file(input_path) != input_record["dense_endpoint_file_sha256"]:
        raise ValueError(f"input endpoint for expert {expert} failed byte closure")
    with safe_open(input_path, framework="pt", device="cpu") as handle:
        input_tensors = {
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
    fit_x_cpu, fit_route_cpu = routed_rows["activation_fit"]
    selection_x_cpu, selection_route_cpu = routed_rows["candidate_selection"]
    fit_x = fit_x_cpu.to(device)
    fit_route = fit_route_cpu.to(device).float()
    selection_x = selection_x_cpu.to(device)
    selection_route = selection_route_cpu.to(device).float()
    input_hessian, input_hessian_record = routed_input_hessian(
        fit_x, fit_route, identity_shrinkage=identity_shrinkage
    )

    encoded: dict[str, dict[str, Any]] = {}
    for spec in PROJECTIONS[:2]:
        input_seed, output_seed = _transform_seeds(layer, spec)
        result = encode_uniform_candidate(
            source_weights[spec.name],
            bits=3,
            codebook=CODEBOOK_SQG_XOR_CHEB_T12,
            device=device,
            quantizer_module=quantizer_module,
            input_sign_seed=input_seed,
            output_sign_seed=output_seed,
            scale_scope_key=(
                ROUTED_INPUT_CURVATURE_KIND,
                layer,
                expert,
                spec.name,
            ),
            g_scale_into_sv=True,
            sigma_reg=SIGMA_REG,
            tailbite_context=128,
            ldlq_tf32=True,
            input_hessian=input_hessian,
        )
        encoded[spec.name] = result
    del input_hessian
    torch.cuda.empty_cache()

    base_gpu = {
        name: input_tensors[f"qsrt_k3.{name}"].to(device).half().contiguous()
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    curvature_upstream_gpu = {
        name: encoded[name]["reconstruction"].to(device).half().contiguous()
        for name in ("gate_proj", "up_proj")
    }
    down_spec = PROJECTIONS[2]
    input_seed, output_seed = _transform_seeds(layer, down_spec)
    down_by_upstream: dict[str, dict[str, Any]] = {}
    down_hessian_records: dict[str, dict[str, float]] = {}
    for gate_curvature, up_curvature in itertools.product((False, True), repeat=2):
        upstream_basis = upstream_curvature_basis_name(
            gate_curvature=gate_curvature,
            up_curvature=up_curvature,
        )
        fit_hidden = _expert_hidden(
            fit_x,
            (
                curvature_upstream_gpu["gate_proj"]
                if gate_curvature
                else base_gpu["gate_proj"]
            ),
            (
                curvature_upstream_gpu["up_proj"]
                if up_curvature
                else base_gpu["up_proj"]
            ),
        )
        down_hessian, down_hessian_record = routed_input_hessian(
            fit_hidden, fit_route, identity_shrinkage=identity_shrinkage
        )
        down_by_upstream[upstream_basis] = encode_uniform_candidate(
            source_weights["down_proj"],
            bits=3,
            codebook=CODEBOOK_SQG_XOR_CHEB_T12,
            device=device,
            quantizer_module=quantizer_module,
            input_sign_seed=input_seed,
            output_sign_seed=output_seed,
            sigma_reg=SIGMA_REG,
            tailbite_context=128,
            ldlq_tf32=True,
            input_hessian=down_hessian,
        )
        down_hessian_records[upstream_basis] = down_hessian_record
        del down_hessian, fit_hidden
        torch.cuda.empty_cache()

    source_gpu = {
        name: value.to(device=device, dtype=torch.bfloat16).contiguous()
        for name, value in source_weights.items()
    }
    teacher = _teacher_output(
        selection_x,
        source_gpu["gate_proj"],
        source_gpu["up_proj"],
        source_gpu["down_proj"],
    )
    curvature_down_gpu = {
        upstream_basis: result["reconstruction"].to(device).half().contiguous()
        for upstream_basis, result in down_by_upstream.items()
    }
    combinations: list[dict[str, Any]] = []
    projection_names = ("gate_proj", "up_proj", "down_proj")
    for choices in itertools.product((False, True), repeat=3):
        gate_curvature, up_curvature, down_curvature = choices
        upstream_basis = upstream_curvature_basis_name(
            gate_curvature=gate_curvature,
            up_curvature=up_curvature,
        )
        selected_weights = {
            "gate_proj": (
                curvature_upstream_gpu["gate_proj"]
                if gate_curvature
                else base_gpu["gate_proj"]
            ),
            "up_proj": (
                curvature_upstream_gpu["up_proj"]
                if up_curvature
                else base_gpu["up_proj"]
            ),
            "down_proj": (
                curvature_down_gpu[upstream_basis]
                if down_curvature
                else base_gpu["down_proj"]
            ),
        }
        output = _candidate_expert_output(
            selection_x,
            gate=selected_weights["gate_proj"],
            up=selected_weights["up_proj"],
            down=selected_weights["down_proj"],
        )
        combinations.append(
            {
                "changed_projections": [
                    name
                    for name, use_curvature in zip(
                        projection_names, choices, strict=True
                    )
                    if use_curvature
                ],
                "down_curvature_upstream_basis": (
                    upstream_basis if down_curvature else None
                ),
                "candidate_selection_weighted_relative_sse": weighted_relative_sse(
                    teacher, output, selection_route
                ),
            }
        )
    selected = min(
        combinations,
        key=lambda item: (
            item["candidate_selection_weighted_relative_sse"],
            len(item["changed_projections"]),
            item["changed_projections"],
        ),
    )
    output_tensors = dict(input_tensors)
    if "gate_proj" in selected["changed_projections"]:
        output_tensors["qsrt_k3.gate_proj"] = encoded["gate_proj"][
            "reconstruction"
        ].half()
    if "up_proj" in selected["changed_projections"]:
        output_tensors["qsrt_k3.up_proj"] = encoded["up_proj"][
            "reconstruction"
        ].half()
    if "down_proj" in selected["changed_projections"]:
        selected_down_basis = selected["down_curvature_upstream_basis"]
        output_tensors["qsrt_k3.down_proj"] = down_by_upstream[
            selected_down_basis
        ]["reconstruction"].half()
    output_path = _dense_expert_path(dest, layer, expert)
    _atomic_save_tensors(output_path, output_tensors)
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
        "routed_rows": {
            "activation_fit": int(fit_x.shape[0]),
            "candidate_selection": int(selection_x.shape[0]),
        },
        "input_hessian": input_hessian_record,
        "down_hessians_by_upstream_basis": down_hessian_records,
        "selection_combinations": combinations,
        "selected_changed_projections": selected["changed_projections"],
        "selected_down_curvature_upstream_basis": selected[
            "down_curvature_upstream_basis"
        ],
        "selected_candidate_selection_weighted_relative_sse": selected[
            "candidate_selection_weighted_relative_sse"
        ],
        "curvature_payloads": {
            "gate_proj": encoded["gate_proj"]["payload"],
            "up_proj": encoded["up_proj"]["payload"],
            "down_proj_by_upstream_basis": {
                upstream_basis: result["payload"]
                for upstream_basis, result in down_by_upstream.items()
            },
        },
        "materialized_tensor_sha256": {
            name: tensor_sha256(output_tensors[f"qsrt_k3.{name}"])
            for name in projection_names
        },
    }


def run_routed_input_curvature_panel(
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
    identity_shrinkage: float,
    device: torch.device,
    exllamav3_root: Path,
    verify_source_shard_hashes: bool = True,
) -> dict[str, Any]:
    """Run a disjoint expert slice of routed-input curvature candidates."""

    frozen = load_frozen_real_weight_panel(panel_manifest_path, layer=layer)
    experts = select_frozen_panel_slice(
        frozen, offset=panel_offset, expert_count=expert_count
    )
    input_artifact = validate_dense_intervention_artifact(input_artifact_root)
    if not set(experts).issubset(input_artifact["expert_ids"]):
        raise ValueError("input intervention artifact does not cover the curvature panel")
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
    capture_manifest_sha256 = sha256_file(capture_root / "manifest.json")
    routed = _read_capture_rows(capture_root, experts=experts)
    manifest = {
        "kind": f"{INTERVENTION_ARTIFACT_KIND}_manifest",
        "candidate": {
            "profile": "qsrt_sqg_e4m3",
            "uniform_rate": 3,
            "variant": "routed_input_curvature_selected_by_expert_output_error",
            "identity_shrinkage": identity_shrinkage,
            "fallback": "retain identity-H QSRT-K3 projection",
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
            "one-sided routed-input curvature control selected by local complete-"
            "expert output error; not downstream-loss curvature or full-model KLD"
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
        record = encode_curvature_candidate_for_expert(
            source=source,
            input_artifact_root=input_artifact_root,
            input_record=input_records[expert],
            routed_rows={
                collection: routed[collection][expert]
                for collection in routed
            },
            layer=layer,
            expert=expert,
            identity_shrinkage=identity_shrinkage,
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
            f"changed={record['selected_changed_projections']}",
            flush=True,
        )
        torch.cuda.empty_cache()
    report = {
        "kind": INTERVENTION_ARTIFACT_KIND,
        "experiment": ROUTED_INPUT_CURVATURE_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "layer": layer,
        "expert_count": len(records),
        "changed_expert_count": sum(
            bool(record["selected_changed_projections"]) for record in records
        ),
        "panel": manifest["panel"],
        "dense_endpoint_bytes": sum(record["dense_endpoint_file_bytes"] for record in records),
        "experts": records,
        "evidence_boundary": manifest["evidence_boundary"],
    }
    atomic_write_json(dest / "report.json", report)
    return report


__all__ = [
    "ROUTED_INPUT_CURVATURE_KIND",
    "routed_input_hessian",
    "run_routed_input_curvature_panel",
    "upstream_curvature_basis_name",
]

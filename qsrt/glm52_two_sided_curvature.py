"""Encode GLM-5.2 K3 candidates with two-sided downstream-loss curvature.

The input is the already-qualified dense uniform-K3 intervention artifact and
an expert-local factor artifact built from disjoint fit documents. For each
projection, this module first recreates the uniform-K3 endpoint with its
recorded global scale. It then changes only the input/output curvature and the
resulting BlockLDLQ/Viterbi decisions. The experiment fails closed unless the
control reconstruction and both persisted scale vectors close exactly.

The lower Kronecker-curvature reconstruction is selected independently for
gate, up, and down. This is a fit-set decision. The resulting complete expert
must still pass the separate full-model, document-disjoint forward-KLD gate.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open

from qsrt.correctness import sha256_file
from qsrt.exl3_loader import load_qsrt_encoder
from qsrt.exl3_reference import CODEBOOK_SQG_XOR_CHEB_T12
from qsrt.glm52_downstream_curvature import (
    DOWNSTREAM_CURVATURE_FACTOR_KIND,
    load_expert_curvature_factors,
    validate_downstream_curvature_factor_artifact,
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


TWO_SIDED_CURVATURE_EXPERIMENT_KIND = (
    "qsrt_glm52_two_sided_downstream_curvature_k3_v1"
)


def _canonical_curvature_loss(
    source: torch.Tensor,
    reconstruction: torch.Tensor,
    input_metric: torch.Tensor,
    output_metric: torch.Tensor,
    *,
    device: torch.device,
) -> float:
    """Evaluate ``tr(H_out E H_in E.T)`` in the source tensor basis."""

    if source.shape != reconstruction.shape or source.ndim != 2:
        raise ValueError("curvature loss requires matching source matrices")
    output_dimension, input_dimension = source.shape
    if input_metric.shape != (input_dimension, input_dimension):
        raise ValueError("input metric does not match the source input dimension")
    if output_metric.shape != (output_dimension, output_dimension):
        raise ValueError("output metric does not match the source output dimension")
    error = (source.float() - reconstruction.float()).to(device)
    input_gpu = input_metric.to(device=device, dtype=torch.float32)
    output_gpu = output_metric.to(device=device, dtype=torch.float32)
    # For symmetric input_metric, this contraction is the same scalar as
    # sum((H_out @ E @ H_in) * E) and avoids retaining that chained product.
    value = torch.sum((output_gpu @ error) * (error @ input_gpu), dtype=torch.float64)
    result = float(value.item())
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("curvature loss must be finite and nonnegative")
    return result


def _baseline_global_scale(input_record: Mapping[str, Any], projection: str) -> float:
    try:
        value = input_record["projections"][projection]["qsrt_k3"]["payload"][
            "g_scale"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"uniform-K3 receipt does not record the {projection} global scale"
        ) from error
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"uniform-K3 {projection} global scale must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"uniform-K3 {projection} global scale is invalid")
    return result


def _scale_vectors_close(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    return all(
        control.get(key) == candidate.get(key)
        for key in ("suh_sha256", "svh_sha256", "scale_bytes", "g_scale")
    )


@torch.no_grad()
def encode_two_sided_candidate_for_expert(
    *,
    source: IndexedTensorStore,
    input_artifact_root: Path,
    input_record: Mapping[str, Any],
    curvature_factor_root: Path,
    curvature_record: Mapping[str, Any],
    layer: int,
    expert: int,
    device: torch.device,
    quantizer_module: Any,
    dest: Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Recreate the control, encode two-sided paths, and select projections."""

    input_path = input_artifact_root / "experts" / input_record["dense_endpoint_file"]
    if (
        input_path.stat().st_size != input_record["dense_endpoint_file_bytes"]
        or sha256_file(input_path) != input_record["dense_endpoint_file_sha256"]
    ):
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
    factors = load_expert_curvature_factors(
        curvature_factor_root, curvature_record
    )
    output_tensors = dict(input_tensors)
    projection_records: dict[str, Any] = {}
    selected_changed_projections: list[str] = []

    for spec in PROJECTIONS:
        projection_name = spec.name
        source_weight = source.get(source_tensor_name(layer, expert, projection_name))
        baseline_weight = input_tensors[f"qsrt_k3.{projection_name}"]
        global_scale = _baseline_global_scale(input_record, projection_name)
        input_seed, output_seed = _transform_seeds(layer, spec)
        scale_scope_key = (
            INTERVENTION_ARTIFACT_KIND,
            layer,
            expert,
            projection_name,
        ) if projection_name in ("gate_proj", "up_proj") else None
        common_arguments = {
            "source": source_weight,
            "bits": 3,
            "codebook": CODEBOOK_SQG_XOR_CHEB_T12,
            "device": device,
            "quantizer_module": quantizer_module,
            "input_sign_seed": input_seed,
            "output_sign_seed": output_seed,
            "scale_scope_key": scale_scope_key,
            "g_scale_into_sv": projection_name in ("gate_proj", "up_proj"),
            "sigma_reg": SIGMA_REG,
            "tailbite_context": 128,
            "ldlq_tf32": True,
            "g_scale_override": global_scale,
            "return_trellis_diagnostics": True,
        }
        control = encode_uniform_candidate(**common_arguments)
        control_reconstruction = control.pop("reconstruction")
        if not torch.equal(control_reconstruction.half(), baseline_weight.half()):
            raise ValueError(
                f"frozen-scale uniform-K3 control for {projection_name} does not "
                "reproduce the input endpoint"
            )
        metric_pair = factors[projection_name]
        candidate = encode_uniform_candidate(
            **common_arguments,
            input_hessian=metric_pair["input_metric"],
            output_hessian=metric_pair["output_metric"],
        )
        candidate_reconstruction = candidate.pop("reconstruction")
        if not _scale_vectors_close(control["payload"], candidate["payload"]):
            raise ValueError(
                f"two-sided curvature changed the {projection_name} scale plane"
            )
        baseline_loss = _canonical_curvature_loss(
            source_weight,
            baseline_weight,
            metric_pair["input_metric"],
            metric_pair["output_metric"],
            device=device,
        )
        candidate_loss = _canonical_curvature_loss(
            source_weight,
            candidate_reconstruction,
            metric_pair["input_metric"],
            metric_pair["output_metric"],
            device=device,
        )
        accept_candidate = candidate_loss < baseline_loss
        if accept_candidate:
            selected_changed_projections.append(projection_name)
            output_tensors[f"qsrt_k3.{projection_name}"] = (
                candidate_reconstruction.half()
            )
        projection_records[projection_name] = {
            "source_tensor": source_tensor_name(layer, expert, projection_name),
            "factor_input_metric_sha256": tensor_sha256(
                metric_pair["input_metric"]
            ),
            "factor_output_metric_sha256": tensor_sha256(
                metric_pair["output_metric"]
            ),
            "frozen_global_scale": global_scale,
            "scale_plane_closed": True,
            "uniform_control_dense_tensor_sha256": tensor_sha256(
                control_reconstruction.half()
            ),
            "input_uniform_dense_tensor_sha256": tensor_sha256(
                baseline_weight.half()
            ),
            "candidate_dense_tensor_sha256": tensor_sha256(
                candidate_reconstruction.half()
            ),
            "uniform_control_curvature_loss": baseline_loss,
            "candidate_curvature_loss": candidate_loss,
            "candidate_curvature_loss_reduction": (
                0.0 if baseline_loss == 0.0 else 1.0 - candidate_loss / baseline_loss
            ),
            "accepted_on_fit_curvature": accept_candidate,
            "uniform_control_payload": control["payload"],
            "two_sided_candidate_payload": candidate["payload"],
            "uniform_control_trellis_diagnostics": control[
                "trellis_diagnostics"
            ],
            "two_sided_candidate_trellis_diagnostics": candidate[
                "trellis_diagnostics"
            ],
        }
        del control_reconstruction, candidate_reconstruction
        torch.cuda.empty_cache()

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
        "curvature_factor_file_sha256": curvature_record["factor_file_sha256"],
        "selected_changed_projections": selected_changed_projections,
        "projection_results": projection_records,
        "materialized_tensor_sha256": {
            spec.name: tensor_sha256(output_tensors[f"qsrt_k3.{spec.name}"])
            for spec in PROJECTIONS
        },
        "evidence_boundary": (
            "fit-curvature selection only; full-model forward KLD remains unmeasured"
        ),
    }


def run_two_sided_curvature_panel(
    *,
    source_root: Path,
    source_inventory_path: Path,
    input_artifact_root: Path,
    curvature_factor_root: Path,
    panel_manifest_path: Path,
    dest: Path,
    layer: int,
    expert_count: int,
    panel_offset: int,
    device: torch.device,
    exllamav3_root: Path,
    verify_source_shard_hashes: bool = True,
) -> dict[str, Any]:
    """Encode one frozen expert-panel slice with two-sided curvature."""

    frozen = load_frozen_real_weight_panel(panel_manifest_path, layer=layer)
    experts = select_frozen_panel_slice(
        frozen, offset=panel_offset, expert_count=expert_count
    )
    input_artifact = validate_dense_intervention_artifact(input_artifact_root)
    factor_artifact = validate_downstream_curvature_factor_artifact(
        curvature_factor_root
    )
    if not set(experts).issubset(input_artifact["expert_ids"]):
        raise ValueError("input intervention artifact does not cover the selected panel")
    if not set(experts).issubset(factor_artifact["expert_ids"]):
        raise ValueError("curvature factor artifact does not cover the selected panel")
    input_records = {
        int(record["expert"]): record
        for record in input_artifact["report"]["experts"]
    }
    curvature_records = {
        int(record["expert"]): record
        for record in factor_artifact["report"]["experts"]
    }
    source_inventory = validate_bounded_source_window(
        source_root,
        source_inventory_path,
        panel={layer: experts},
        verify_shard_hashes=verify_source_shard_hashes,
    )
    manifest = {
        "kind": f"{INTERVENTION_ARTIFACT_KIND}_manifest",
        "candidate": {
            "profile": "qsrt_sqg_e4m3",
            "uniform_rate": 3,
            "variant": "two_sided_downstream_curvature_feedback",
            "selection": "lower_fit_kronecker_curvature_per_projection",
            "fallback": "retain_reproduced_uniform_k3_projection",
            "global_scale_policy": "freeze_uniform_k3_receipt",
            "scale_plane_policy": "require_identical_suh_and_svh_hashes",
        },
        "source": source_inventory,
        "input_intervention_artifact": {
            "root": str(input_artifact_root.resolve()),
            "manifest_sha256": input_artifact["manifest_sha256"],
        },
        "downstream_curvature_factor_artifact": {
            "root": str(curvature_factor_root.resolve()),
            "manifest_sha256": factor_artifact["manifest_sha256"],
            "kind": DOWNSTREAM_CURVATURE_FACTOR_KIND,
        },
        "frozen_panel": {
            "path": frozen["path"],
            "sha256": frozen["sha256"],
            "selected_offset": panel_offset,
            "selected_count": expert_count,
        },
        "panel": {str(layer): list(experts)},
        "resident_endpoint_dtype": "FP16",
        "device": str(device),
        "model_downloads_required": False,
        "complete_bf16_checkpoint_required": False,
        "evidence_boundary": (
            "format-preserving K3 factor-fit candidate; document-disjoint full-"
            "model KLD and complete serialized checkpoint size remain required"
        ),
    }
    manifest_sha256 = prepare_destination(dest, manifest, resume=False)
    source = IndexedTensorStore(source_root)
    quantizer_module = load_qsrt_encoder(exllamav3_root)
    install_sqg_quantizer(quantizer_module)
    records: list[dict[str, Any]] = []
    for index, expert in enumerate(experts, start=1):
        started = time.monotonic()
        record = encode_two_sided_candidate_for_expert(
            source=source,
            input_artifact_root=input_artifact_root,
            input_record=input_records[expert],
            curvature_factor_root=curvature_factor_root,
            curvature_record=curvature_records[expert],
            layer=layer,
            expert=expert,
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
    report = {
        "kind": INTERVENTION_ARTIFACT_KIND,
        "experiment": TWO_SIDED_CURVATURE_EXPERIMENT_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "layer": layer,
        "expert_count": len(records),
        "changed_expert_count": sum(
            bool(record["selected_changed_projections"]) for record in records
        ),
        "changed_projection_count": sum(
            len(record["selected_changed_projections"]) for record in records
        ),
        "panel": manifest["panel"],
        "dense_endpoint_bytes": sum(
            record["dense_endpoint_file_bytes"] for record in records
        ),
        "experts": records,
        "evidence_boundary": manifest["evidence_boundary"],
    }
    atomic_write_json(dest / "report.json", report)
    validate_dense_intervention_artifact(dest)
    return report


__all__ = [
    "TWO_SIDED_CURVATURE_EXPERIMENT_KIND",
    "encode_two_sided_candidate_for_expert",
    "run_two_sided_curvature_panel",
]

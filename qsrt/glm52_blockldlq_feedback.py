"""Build frozen-scale GLM-5.2 BlockLDLQ feedback ablations.

The input artifact contains the qualified uniform-K3 dense endpoints. Every
projection is re-encoded twice from the same BF16 source tensor: once with the
ordinary feedback multiplier of one and once with a requested multiplier.
Both calls reuse the recorded global scale. The ablation fails unless the
ordinary call reproduces the stored endpoint and the two calls persist
identical input/output scale vectors. The resulting artifact therefore changes
only feedback-adjusted trellis targets and the path bits selected from them.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch
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
from qsrt.glm52_two_sided_curvature import (
    _baseline_global_scale,
    _scale_vectors_close,
)
from qsrt.ldlq import SIGMA_REG
from qsrt.qsrt_codec_pilot import encode_uniform_candidate, tensor_sha256
from qsrt.sqg_quantizer import install_sqg_quantizer


BLOCKLDLQ_FEEDBACK_EXPERIMENT_KIND = "qsrt_glm52_frozen_scale_blockldlq_feedback_v1"


def _validate_feedback_multiplier(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) < 1.0
    ):
        raise ValueError("ablation feedback multiplier must be in [0, 1)")
    return float(value)


@torch.no_grad()
def encode_feedback_ablation_for_expert(
    *,
    source: IndexedTensorStore,
    input_artifact_root: Path,
    input_record: Mapping[str, Any],
    layer: int,
    expert: int,
    feedback_multiplier: float,
    device: torch.device,
    quantizer_module: Any,
    dest: Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Reproduce one expert and change only its BlockLDLQ feedback strength."""

    multiplier = _validate_feedback_multiplier(feedback_multiplier)
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
    output_tensors = dict(input_tensors)
    projection_records: dict[str, Any] = {}
    for spec in PROJECTIONS:
        projection_name = spec.name
        source_weight = source.get(source_tensor_name(layer, expert, projection_name))
        stored_control = input_tensors[f"qsrt_k3.{projection_name}"]
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
        control = encode_uniform_candidate(
            **common_arguments,
            ldlq_feedback_multiplier=1.0,
        )
        control_reconstruction = control.pop("reconstruction")
        if not torch.equal(control_reconstruction.half(), stored_control.half()):
            raise ValueError(
                f"frozen-scale BlockLDLQ control for {projection_name} does not "
                "reproduce the input endpoint"
            )
        ablation = encode_uniform_candidate(
            **common_arguments,
            ldlq_feedback_multiplier=multiplier,
        )
        ablation_reconstruction = ablation.pop("reconstruction")
        if not _scale_vectors_close(control["payload"], ablation["payload"]):
            raise ValueError(
                f"feedback ablation changed the {projection_name} scale plane"
            )
        source_float = source_weight.float()
        control_sse = float(
            torch.sum(
                (source_float - control_reconstruction.float()).double().square()
            ).item()
        )
        ablation_sse = float(
            torch.sum(
                (source_float - ablation_reconstruction.float()).double().square()
            ).item()
        )
        output_tensors[f"qsrt_k3.{projection_name}"] = ablation_reconstruction.half()
        projection_records[projection_name] = {
            "source_tensor": source_tensor_name(layer, expert, projection_name),
            "frozen_global_scale": global_scale,
            "scale_plane_closed": True,
            "ordinary_feedback_dense_tensor_sha256": tensor_sha256(
                control_reconstruction.half()
            ),
            "stored_uniform_k3_dense_tensor_sha256": tensor_sha256(
                stored_control.half()
            ),
            "ablation_dense_tensor_sha256": tensor_sha256(
                ablation_reconstruction.half()
            ),
            "ordinary_feedback_source_sse": control_sse,
            "ablation_source_sse": ablation_sse,
            "ablation_source_sse_reduction": (
                0.0 if control_sse == 0.0 else 1.0 - ablation_sse / control_sse
            ),
            "ordinary_feedback_payload": control["payload"],
            "ablation_payload": ablation["payload"],
            "ordinary_feedback_trellis_diagnostics": control[
                "trellis_diagnostics"
            ],
            "ablation_trellis_diagnostics": ablation["trellis_diagnostics"],
        }
        del control_reconstruction, ablation_reconstruction
        torch.cuda.empty_cache()
    output_path = _dense_expert_path(dest, layer, expert)
    _atomic_save_tensors(output_path, output_tensors)
    return {
        "kind": f"{INTERVENTION_ARTIFACT_KIND}_expert",
        "complete": True,
        "manifest_sha256": manifest_sha256,
        "layer": layer,
        "expert": expert,
        "feedback_multiplier": multiplier,
        "dense_endpoint_file": output_path.name,
        "dense_endpoint_file_bytes": output_path.stat().st_size,
        "dense_endpoint_file_sha256": sha256_file(output_path),
        "input_dense_endpoint_sha256": input_record["dense_endpoint_file_sha256"],
        "projection_results": projection_records,
        "materialized_tensor_sha256": {
            spec.name: tensor_sha256(output_tensors[f"qsrt_k3.{spec.name}"])
            for spec in PROJECTIONS
        },
        "evidence_boundary": (
            "frozen-scale feedback ablation; full-model forward KLD remains "
            "required before interpreting source error"
        ),
    }


def run_feedback_ablation_panel(
    *,
    source_root: Path,
    source_inventory_path: Path,
    input_artifact_root: Path,
    panel_manifest_path: Path,
    dest: Path,
    layer: int,
    expert_count: int,
    panel_offset: int,
    feedback_multiplier: float,
    device: torch.device,
    exllamav3_root: Path,
    verify_source_shard_hashes: bool = True,
) -> dict[str, Any]:
    """Build one frozen expert slice with a controlled feedback multiplier."""

    multiplier = _validate_feedback_multiplier(feedback_multiplier)
    frozen = load_frozen_real_weight_panel(panel_manifest_path, layer=layer)
    experts = select_frozen_panel_slice(
        frozen, offset=panel_offset, expert_count=expert_count
    )
    input_artifact = validate_dense_intervention_artifact(input_artifact_root)
    if not set(experts).issubset(input_artifact["expert_ids"]):
        raise ValueError("input intervention artifact does not cover the selected panel")
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
    manifest = {
        "kind": f"{INTERVENTION_ARTIFACT_KIND}_manifest",
        "candidate": {
            "profile": "qsrt_sqg_e4m3",
            "uniform_rate": 3,
            "variant": "frozen_scale_blockldlq_feedback_ablation",
            "feedback_multiplier": multiplier,
            "global_scale_policy": "freeze_uniform_k3_receipt",
            "scale_plane_policy": "require_identical_suh_and_svh_hashes",
        },
        "source": source_inventory,
        "input_intervention_artifact": {
            "root": str(input_artifact_root.resolve()),
            "manifest_sha256": input_artifact["manifest_sha256"],
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
            "one-variable K3 feedback ablation with frozen scales; document-"
            "disjoint full-model KLD remains the decision metric"
        ),
    }
    manifest_sha256 = prepare_destination(dest, manifest, resume=False)
    source = IndexedTensorStore(source_root)
    quantizer_module = load_qsrt_encoder(exllamav3_root)
    install_sqg_quantizer(quantizer_module)
    records: list[dict[str, Any]] = []
    for index, expert in enumerate(experts, start=1):
        started = time.monotonic()
        record = encode_feedback_ablation_for_expert(
            source=source,
            input_artifact_root=input_artifact_root,
            input_record=input_records[expert],
            layer=layer,
            expert=expert,
            feedback_multiplier=multiplier,
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
            f"feedback={multiplier:.3f}",
            flush=True,
        )
    report = {
        "kind": INTERVENTION_ARTIFACT_KIND,
        "experiment": BLOCKLDLQ_FEEDBACK_EXPERIMENT_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "layer": layer,
        "feedback_multiplier": multiplier,
        "expert_count": len(records),
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
    "BLOCKLDLQ_FEEDBACK_EXPERIMENT_KIND",
    "encode_feedback_ablation_for_expert",
    "run_feedback_ablation_panel",
]

"""Compare frozen GLM expert candidates on the untouched reporting context.

The input capture contains the layer-3 residual-stream vectors, route IDs, and
applied route weights from the published BF16-reference context. The scorer
reconstructs each selected expert with official BF16 weights, uniform QSRT K3,
and one already-frozen comparison artifact. It measures complete expert
functions and the route-weighted sum contributed by the complete panel.

The reporting context may diagnose whether a previously selected local metric
agrees with full-model KLD. It must never select a path, rate, refit, shrinkage
setting, or allocation candidate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open

from qsrt.correctness import sha256_file
from qsrt.glm52_expert_intervention_runtime import (
    validate_dense_intervention_artifact,
)
from qsrt.glm52_pilot import (
    HIDDEN_SIZE,
    IndexedTensorStore,
    PROJECTIONS,
    atomic_write_json,
    source_tensor_name,
)
from qsrt.glm52_real_weight_benchmark import validate_bounded_source_window


REPORTING_CAPTURE_SCHEMA = (
    "qsrt_glm52_reporting_layer_input_capture_manifest"
)
REPORTING_OUTPUT_REPORT_SCHEMA = (
    "qsrt_glm52_reporting_complete_expert_output_comparison"
)
PUBLISHED_REFERENCE_MANIFEST_SHA256 = (
    "985120136741037918bcd4dc8da9813c1f6268b35a730302f99cf6b3eebb7606"
)


def squared_error_receipt(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    """Return absolute and reference-normalized squared error."""

    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("reference and candidate outputs must share a rank-two shape")
    reference64 = reference.double()
    error64 = candidate.double() - reference64
    error_sum = float(error64.square().sum().item())
    reference_sum = float(reference64.square().sum().item())
    return {
        "squared_error_sum": error_sum,
        "reference_squared_sum": reference_sum,
        "relative_squared_error": error_sum / max(reference_sum, 1e-30),
    }


def evaluate_complete_expert(
    expert_input: torch.Tensor,
    *,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one complete SwiGLU expert in its supplied weight dtype."""

    x = expert_input.to(device=gate.device, dtype=gate.dtype)
    hidden = F.silu(F.linear(x, gate)) * F.linear(x, up)
    return F.linear(hidden, down).float()


def _validate_capture(
    capture_root: Path,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    root = capture_root.resolve(strict=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != REPORTING_CAPTURE_SCHEMA
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("model_layer") != 3
        or manifest.get("collections") != {"untouched_reporting_context": 1}
        or manifest.get("reference_manifest_sha256")
        != PUBLISHED_REFERENCE_MANIFEST_SHA256
    ):
        raise ValueError("reporting layer-input capture manifest mismatch")
    reuse_policy = manifest.get("reuse_policy")
    if not isinstance(reuse_policy, str) or "must not select" not in reuse_policy:
        raise ValueError("reporting capture does not prohibit candidate selection")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("reporting capture must contain one sequence record")
    record = records[0]
    filename = record.get("capture_file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("reporting capture filename is unsafe")
    path = root / filename
    if (
        not path.is_file()
        or path.stat().st_size != record.get("capture_file_bytes")
        or sha256_file(path) != record.get("capture_file_sha256")
    ):
        raise ValueError("reporting capture file failed byte closure")
    generation = record.get("control_generation")
    plan_sha256 = manifest.get("corpus_plan_sha256")
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata != {
            "schema": "qsrt_glm52_layer_input_capture_v1",
            "model_layer": "3",
            "control_generation": str(generation),
            "corpus_plan_sha256": plan_sha256,
        }:
            raise ValueError("reporting capture tensor metadata mismatch")
        hidden_states = handle.get_tensor("hidden_states")
        topk_ids = handle.get_tensor("topk_ids")
        topk_weights = handle.get_tensor("topk_weights")
    if (
        hidden_states.dtype != torch.bfloat16
        or hidden_states.ndim != 2
        or hidden_states.shape[1] != HIDDEN_SIZE
        or topk_ids.dtype != torch.int32
        or topk_ids.shape != (hidden_states.shape[0], 8)
        or topk_weights.dtype != torch.float32
        or topk_weights.shape != topk_ids.shape
        or record.get("token_count") != hidden_states.shape[0]
    ):
        raise ValueError("reporting capture tensor contract mismatch")
    if (
        not bool(torch.isfinite(hidden_states).all())
        or not bool(torch.isfinite(topk_weights).all())
        or bool((topk_weights < 0).any())
        or bool((topk_ids < 0).any())
        or bool((topk_ids >= 256).any())
    ):
        raise ValueError("reporting capture contains invalid tensor values")
    return manifest, hidden_states, topk_ids, topk_weights


def _artifact_records(artifact: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    records = artifact["report"]["experts"]
    return {int(record["expert"]): record for record in records}


def _load_endpoint_tensors(
    root: Path, record: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    path = root / "experts" / str(record["dense_endpoint_file"])
    if sha256_file(path) != record["dense_endpoint_file_sha256"]:
        raise ValueError(f"dense endpoint {path.name} failed byte closure")
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


def _validate_kld_report(
    path: Path,
    *,
    expected_artifact_manifest_sha256: str,
) -> dict[str, Any]:
    report = json.loads(path.read_text())
    paired = report.get("paired")
    if (
        report.get("schema") != "qsrt_glm52_paired_expert_intervention_kld"
        or report.get("schema_version") != 2
        or report.get("status") != "complete"
        or report.get("measurement_controls_passed") is not True
        or report.get("intervention_artifact", {}).get("manifest_sha256")
        != expected_artifact_manifest_sha256
        or not isinstance(paired, dict)
        or paired.get("position_count") != 2047
    ):
        raise ValueError(f"KLD report {path} failed identity or control closure")
    return report


def _runtime_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    runtime = dict(report.get("runtime", {}))
    runtime.pop("model_load_seconds", None)
    return runtime


@torch.no_grad()
def compare_reporting_expert_outputs(
    *,
    source_root: Path,
    source_inventory_path: Path,
    uniform_artifact_root: Path,
    comparison_artifact_root: Path,
    reporting_capture_root: Path,
    uniform_kld_report_path: Path,
    comparison_kld_report_path: Path,
    dest: Path,
    device: torch.device,
    verify_source_shard_hashes: bool = True,
) -> dict[str, Any]:
    """Compare two frozen artifacts against official BF16 expert functions."""

    if dest.exists():
        raise FileExistsError(dest)
    uniform_root = uniform_artifact_root.resolve(strict=True)
    comparison_root = comparison_artifact_root.resolve(strict=True)
    uniform = validate_dense_intervention_artifact(uniform_root)
    comparison = validate_dense_intervention_artifact(comparison_root)
    experts = tuple(int(value) for value in uniform["expert_ids"])
    if comparison["expert_ids"] != experts:
        raise ValueError("uniform and comparison artifacts cover different experts")
    uniform_manifest = json.loads((uniform_root / "manifest.json").read_text())
    comparison_manifest = json.loads((comparison_root / "manifest.json").read_text())
    if uniform_manifest.get("panel") != comparison_manifest.get("panel"):
        raise ValueError("uniform and comparison artifacts use different panels")
    if uniform_manifest.get("source_identity") != comparison_manifest.get(
        "source_identity"
    ):
        raise ValueError("uniform and comparison artifacts use different sources")
    if comparison_manifest.get("input_intervention_artifact", {}).get(
        "manifest_sha256"
    ) != uniform["manifest_sha256"]:
        raise ValueError("comparison artifact is not derived from the uniform artifact")
    source_identity = validate_bounded_source_window(
        source_root,
        source_inventory_path,
        panel={3: experts},
        verify_shard_hashes=verify_source_shard_hashes,
    )
    capture_manifest, hidden_cpu, ids, route_weights = _validate_capture(
        reporting_capture_root
    )
    uniform_kld = _validate_kld_report(
        uniform_kld_report_path,
        expected_artifact_manifest_sha256=uniform["manifest_sha256"],
    )
    comparison_kld = _validate_kld_report(
        comparison_kld_report_path,
        expected_artifact_manifest_sha256=comparison["manifest_sha256"],
    )
    uniform_baseline_kld = float(
        uniform_kld["paired"]["baseline_mean_forward_kld"]
    )
    comparison_baseline_kld = float(
        comparison_kld["paired"]["baseline_mean_forward_kld"]
    )
    if uniform_baseline_kld != comparison_baseline_kld:
        raise ValueError("KLD reports use different resident baselines")
    if (
        uniform_kld.get("reference_manifest")
        != comparison_kld.get("reference_manifest")
        or uniform_kld.get("attention_contract")
        != comparison_kld.get("attention_contract")
        or _runtime_contract(uniform_kld) != _runtime_contract(comparison_kld)
    ):
        raise ValueError("KLD reports use different reference or runtime contracts")

    source = IndexedTensorStore(source_root)
    uniform_records = _artifact_records(uniform)
    comparison_records = _artifact_records(comparison)
    token_count = int(hidden_cpu.shape[0])
    panel_reference = torch.zeros(
        (token_count, HIDDEN_SIZE), dtype=torch.float32, device=device
    )
    panel_uniform = torch.zeros_like(panel_reference)
    panel_comparison = torch.zeros_like(panel_reference)
    selected_token_mask = torch.zeros(token_count, dtype=torch.bool, device=device)
    expert_receipts: list[dict[str, Any]] = []

    for expert in experts:
        positions = (ids == expert).nonzero(as_tuple=False)
        if positions.numel() == 0:
            expert_receipts.append(
                {
                    "expert": expert,
                    "routed_row_count": 0,
                    "uniform": None,
                    "comparison": None,
                    "reporting_preference": "unobserved",
                }
            )
            continue
        token_indices_cpu = positions[:, 0]
        route_indices_cpu = positions[:, 1]
        if torch.unique(token_indices_cpu).numel() != token_indices_cpu.numel():
            raise ValueError(f"reporting context routes expert {expert} twice per token")
        expert_input = hidden_cpu.index_select(0, token_indices_cpu).to(device)
        applied_weights = route_weights[
            token_indices_cpu, route_indices_cpu
        ].to(device)
        source_weights = {
            spec.name: source.get(source_tensor_name(3, expert, spec.name))
            .to(device=device, dtype=torch.bfloat16)
            .contiguous()
            for spec in PROJECTIONS
        }
        uniform_tensors = _load_endpoint_tensors(
            uniform_root, uniform_records[expert]
        )
        comparison_tensors = _load_endpoint_tensors(
            comparison_root, comparison_records[expert]
        )
        for projection in ("gate_proj", "up_proj", "down_proj"):
            if not torch.equal(
                uniform_tensors[f"exl3.{projection}"],
                comparison_tensors[f"exl3.{projection}"],
            ):
                raise ValueError(
                    f"expert {expert} EXL3 endpoint differs between artifacts"
                )
        uniform_weights = {
            projection: uniform_tensors[f"qsrt_k3.{projection}"]
            .to(device)
            .contiguous()
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        comparison_weights = {
            projection: comparison_tensors[f"qsrt_k3.{projection}"]
            .to(device)
            .contiguous()
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        reference_output = evaluate_complete_expert(
            expert_input,
            gate=source_weights["gate_proj"],
            up=source_weights["up_proj"],
            down=source_weights["down_proj"],
        )
        uniform_output = evaluate_complete_expert(
            expert_input,
            gate=uniform_weights["gate_proj"],
            up=uniform_weights["up_proj"],
            down=uniform_weights["down_proj"],
        )
        comparison_output = evaluate_complete_expert(
            expert_input,
            gate=comparison_weights["gate_proj"],
            up=comparison_weights["up_proj"],
            down=comparison_weights["down_proj"],
        )
        weighted_reference = reference_output * applied_weights.unsqueeze(1)
        weighted_uniform = uniform_output * applied_weights.unsqueeze(1)
        weighted_comparison = comparison_output * applied_weights.unsqueeze(1)
        token_indices = token_indices_cpu.to(device)
        panel_reference.index_add_(0, token_indices, weighted_reference)
        panel_uniform.index_add_(0, token_indices, weighted_uniform)
        panel_comparison.index_add_(0, token_indices, weighted_comparison)
        selected_token_mask[token_indices] = True
        uniform_metric = squared_error_receipt(
            weighted_reference, weighted_uniform
        )
        comparison_metric = squared_error_receipt(
            weighted_reference, weighted_comparison
        )
        if comparison_metric["relative_squared_error"] < uniform_metric[
            "relative_squared_error"
        ]:
            preference = "comparison"
        elif comparison_metric["relative_squared_error"] > uniform_metric[
            "relative_squared_error"
        ]:
            preference = "uniform_k3"
        else:
            preference = "equal"
        expert_receipts.append(
            {
                "expert": expert,
                "routed_row_count": int(token_indices.numel()),
                "applied_route_weight_sum": float(applied_weights.sum().item()),
                "uniform": uniform_metric,
                "comparison": comparison_metric,
                "reporting_preference": preference,
                "comparison_changed_projections": comparison_records[expert].get(
                    "selected_changed_projections"
                ),
            }
        )
        del (
            expert_input,
            source_weights,
            uniform_tensors,
            comparison_tensors,
            uniform_weights,
            comparison_weights,
            reference_output,
            uniform_output,
            comparison_output,
            weighted_reference,
            weighted_uniform,
            weighted_comparison,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    panel_reference = panel_reference[selected_token_mask]
    panel_uniform = panel_uniform[selected_token_mask]
    panel_comparison = panel_comparison[selected_token_mask]
    uniform_panel_metric = squared_error_receipt(panel_reference, panel_uniform)
    comparison_panel_metric = squared_error_receipt(
        panel_reference, panel_comparison
    )
    local_preference = (
        "comparison"
        if comparison_panel_metric["relative_squared_error"]
        < uniform_panel_metric["relative_squared_error"]
        else "uniform_k3"
        if comparison_panel_metric["relative_squared_error"]
        > uniform_panel_metric["relative_squared_error"]
        else "equal"
    )
    uniform_mean_kld = float(uniform_kld["paired"]["candidate_mean_forward_kld"])
    comparison_mean_kld = float(
        comparison_kld["paired"]["candidate_mean_forward_kld"]
    )
    kld_preference = (
        "comparison"
        if comparison_mean_kld < uniform_mean_kld
        else "uniform_k3"
        if comparison_mean_kld > uniform_mean_kld
        else "equal"
    )
    report = {
        "schema": REPORTING_OUTPUT_REPORT_SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "source": source_identity,
        "uniform_artifact": {
            "root": str(uniform_root),
            "manifest_sha256": uniform["manifest_sha256"],
        },
        "comparison_artifact": {
            "root": str(comparison_root),
            "manifest_sha256": comparison["manifest_sha256"],
            "variant": comparison_manifest.get("candidate", {}).get("variant"),
        },
        "reporting_capture": {
            "root": str(reporting_capture_root.resolve()),
            "manifest_sha256": sha256_file(
                reporting_capture_root / "manifest.json"
            ),
            "corpus_plan_sha256": capture_manifest["corpus_plan_sha256"],
            "reference_manifest_sha256": capture_manifest[
                "reference_manifest_sha256"
            ],
            "token_count": token_count,
            "selected_panel_token_count": int(selected_token_mask.sum().item()),
            "reuse_policy": capture_manifest["reuse_policy"],
        },
        "panel_route_weighted_output": {
            "uniform": uniform_panel_metric,
            "comparison": comparison_panel_metric,
            "local_metric_preference": local_preference,
        },
        "full_model_kld": {
            "resident_exl3_mean_forward_kld": uniform_baseline_kld,
            "uniform_mean_forward_kld": uniform_mean_kld,
            "comparison_mean_forward_kld": comparison_mean_kld,
            "kld_preference": kld_preference,
            "uniform_report_path": str(uniform_kld_report_path.resolve()),
            "uniform_report_sha256": sha256_file(uniform_kld_report_path),
            "comparison_report_path": str(comparison_kld_report_path.resolve()),
            "comparison_report_sha256": sha256_file(comparison_kld_report_path),
        },
        "local_metric_and_kld_order_agree": local_preference == kld_preference,
        "experts": expert_receipts,
        "evidence_boundary": (
            "complete-expert and selected-panel output error on one untouched "
            "reporting context; the candidate was frozen before this context "
            "was captured; this diagnostic must not select later candidates; "
            "the paired KLD values still lack document-level replication"
        ),
        "model_downloads_performed": False,
        "complete_bf16_checkpoint_required": False,
    }
    atomic_write_json(dest / "report.json", report)
    return report


__all__ = [
    "REPORTING_CAPTURE_SCHEMA",
    "REPORTING_OUTPUT_REPORT_SCHEMA",
    "PUBLISHED_REFERENCE_MANIFEST_SHA256",
    "compare_reporting_expert_outputs",
    "evaluate_complete_expert",
    "squared_error_receipt",
]

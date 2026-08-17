"""Build expert-local two-sided curvature factors for GLM-5.2.

Model-Preserving Adaptive Rounding approximates a linear layer's downstream
loss with a Kronecker product of an input metric and an output metric. Its
sequence-gradient construction forms the complete weight gradient for each
calibration sequence and accumulates the two Gram matrices of that gradient.

GLM-5.2 exposes three linear projections inside each routed SwiGLU expert.
This module needs only two full-model quantities for a selected expert: the
input to the expert and the loss gradient at the surrounding mixture-of-
experts output. The ordinary chain rule derives the gate-, up-, and down-
projection gradients locally from the bounded official source tensors. The
complete BF16 checkpoint is therefore not required.

The capture schema is intentionally independent of vLLM. A capture producer
must join tensor-parallel hidden coordinates, retain one record per source
sequence, and store the gradient before the MoE output is added to the
residual stream. The reader multiplies that shared gradient by the recorded
route coefficient to obtain the gradient of one routed expert's output.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from qsrt.correctness import sha256_file
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
from qsrt.qsrt_codec_pilot import tensor_sha256


MOE_OUTPUT_GRADIENT_CAPTURE_SCHEMA = (
    "qsrt_glm52_moe_output_gradient_capture_manifest"
)
MOE_OUTPUT_GRADIENT_FILE_SCHEMA = "qsrt_glm52_moe_output_gradient_capture"
DOWNSTREAM_CURVATURE_FACTOR_KIND = (
    "qsrt_glm52_expert_downstream_curvature_factors_v1"
)
CURVATURE_PROJECTION_NAMES = tuple(spec.name for spec in PROJECTIONS)


@dataclass(frozen=True)
class RoutedGradientSequence:
    """One source sequence's rows for a naturally routed expert."""

    sequence_identity: str
    expert_inputs: torch.Tensor
    expert_output_gradients: torch.Tensor


def _validate_shrinkage(value: float, *, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{role} identity shrinkage must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"{role} identity shrinkage must be in (0, 1]")
    return result


def _normalize_and_shrink_metric(
    metric: torch.Tensor,
    *,
    identity_shrinkage: float,
    role: str,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError(f"{role} metric must be square")
    if not torch.is_floating_point(metric) or not bool(torch.isfinite(metric).all()):
        raise ValueError(f"{role} metric must be finite and floating point")
    shrinkage = _validate_shrinkage(identity_shrinkage, role=role)
    metric = ((metric.float() + metric.float().T) * 0.5).contiguous()
    unnormalized_diagonal_mean = float(metric.diagonal().mean().item())
    if (
        not math.isfinite(unnormalized_diagonal_mean)
        or unnormalized_diagonal_mean <= 0.0
    ):
        raise ValueError(f"{role} metric has a nonpositive mean diagonal")
    metric.div_(unnormalized_diagonal_mean)
    metric.mul_(1.0 - shrinkage)
    metric.diagonal().add_(shrinkage)
    return metric.contiguous(), {
        "dimension": int(metric.shape[0]),
        "identity_shrinkage": shrinkage,
        "unnormalized_diagonal_mean": unnormalized_diagonal_mean,
        "normalized_diagonal_mean": float(metric.diagonal().mean().item()),
        "normalized_frobenius_norm": float(torch.linalg.vector_norm(metric).item()),
    }


@torch.no_grad()
def sequence_gradient_weight_curvature(
    projection_sequences: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    input_identity_shrinkage: float,
    output_identity_shrinkage: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Estimate Kronecker factors from complete per-sequence weight gradients.

    Every sequence member contains the linear input ``X`` and the loss
    gradient ``G`` at that linear's output. The complete weight gradient is
    ``G.T @ X``. The returned factors are the averages of its right and left
    Gram matrices, normalized to unit mean diagonal and shrunk toward the
    identity. This is the first, simultaneously initialized iteration of the
    paper's sequence-gradient Kronecker estimator.
    """

    if not projection_sequences:
        raise ValueError("curvature estimation requires at least one sequence")
    input_shrinkage = _validate_shrinkage(
        input_identity_shrinkage, role="input"
    )
    output_shrinkage = _validate_shrinkage(
        output_identity_shrinkage, role="output"
    )
    input_dimension: int | None = None
    output_dimension: int | None = None
    input_sum: torch.Tensor | None = None
    output_sum: torch.Tensor | None = None
    row_count = 0
    nonzero_sequence_count = 0
    sequence_gradient_norms: list[float] = []

    for sequence_index, (inputs, output_gradients) in enumerate(
        projection_sequences
    ):
        if (
            inputs.ndim != 2
            or output_gradients.ndim != 2
            or inputs.shape[0] != output_gradients.shape[0]
            or inputs.shape[0] == 0
        ):
            raise ValueError(
                f"curvature sequence {sequence_index} must contain matching "
                "nonempty input and output-gradient rows"
            )
        if (
            not torch.is_floating_point(inputs)
            or not torch.is_floating_point(output_gradients)
            or not bool(torch.isfinite(inputs).all())
            or not bool(torch.isfinite(output_gradients).all())
        ):
            raise ValueError(
                f"curvature sequence {sequence_index} contains invalid values"
            )
        if input_dimension is None:
            input_dimension = int(inputs.shape[1])
            output_dimension = int(output_gradients.shape[1])
            input_sum = torch.zeros(
                (input_dimension, input_dimension),
                dtype=torch.float32,
                device=device,
            )
            output_sum = torch.zeros(
                (output_dimension, output_dimension),
                dtype=torch.float32,
                device=device,
            )
        elif (
            inputs.shape[1] != input_dimension
            or output_gradients.shape[1] != output_dimension
        ):
            raise ValueError("curvature sequences changed projection dimensions")

        inputs_gpu = inputs.to(device=device, dtype=torch.float32)
        gradients_gpu = output_gradients.to(device=device, dtype=torch.float32)
        gradient_weight = gradients_gpu.T @ inputs_gpu
        gradient_norm = float(torch.linalg.vector_norm(gradient_weight).item())
        sequence_gradient_norms.append(gradient_norm)
        if gradient_norm > 0.0:
            nonzero_sequence_count += 1
        assert input_sum is not None and output_sum is not None
        input_sum.addmm_(gradient_weight.T, gradient_weight)
        output_sum.addmm_(gradient_weight, gradient_weight.T)
        row_count += int(inputs.shape[0])
        del inputs_gpu, gradients_gpu, gradient_weight

    if nonzero_sequence_count == 0:
        raise ValueError("all complete sequence weight gradients are zero")
    assert input_sum is not None and output_sum is not None
    sequence_count = len(projection_sequences)
    input_sum.div_(sequence_count)
    output_sum.div_(sequence_count)
    input_metric, input_record = _normalize_and_shrink_metric(
        input_sum,
        identity_shrinkage=input_shrinkage,
        role="input",
    )
    output_metric, output_record = _normalize_and_shrink_metric(
        output_sum,
        identity_shrinkage=output_shrinkage,
        role="output",
    )
    record = {
        "estimator": "complete_sequence_weight_gradient_kronecker_grams",
        "sequence_count": sequence_count,
        "nonzero_sequence_count": nonzero_sequence_count,
        "routed_row_count": row_count,
        "sequence_gradient_norm_min": min(sequence_gradient_norms),
        "sequence_gradient_norm_max": max(sequence_gradient_norms),
        "sequence_gradient_norm_mean": sum(sequence_gradient_norms)
        / sequence_count,
        "input_metric": input_record,
        "output_metric": output_record,
    }
    return input_metric.cpu(), output_metric.cpu(), record


def _silu_derivative(values: torch.Tensor) -> torch.Tensor:
    sigmoid = torch.sigmoid(values)
    return sigmoid * (1.0 + values * (1.0 - sigmoid))


@torch.no_grad()
def derive_projection_sequences(
    sequences: Sequence[RoutedGradientSequence],
    *,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    device: torch.device,
) -> dict[str, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Apply the SwiGLU chain rule to one expert's routed output gradients."""

    if not sequences:
        raise ValueError("projection derivation requires at least one sequence")
    if (
        gate_weight.ndim != 2
        or up_weight.shape != gate_weight.shape
        or down_weight.ndim != 2
        or down_weight.shape[1] != gate_weight.shape[0]
    ):
        raise ValueError("gate, up, and down weights do not form a SwiGLU expert")
    hidden_dimension = int(gate_weight.shape[1])
    intermediate_dimension = int(gate_weight.shape[0])
    if down_weight.shape[0] != hidden_dimension:
        raise ValueError("down projection does not return the expert hidden dimension")

    gate = gate_weight.to(device=device, dtype=torch.float32)
    up = up_weight.to(device=device, dtype=torch.float32)
    down = down_weight.to(device=device, dtype=torch.float32)
    result: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
        name: [] for name in CURVATURE_PROJECTION_NAMES
    }
    seen_sequence_identities: set[str] = set()
    for sequence in sequences:
        if (
            not isinstance(sequence.sequence_identity, str)
            or not sequence.sequence_identity
            or sequence.sequence_identity in seen_sequence_identities
        ):
            raise ValueError("curvature sequence identities must be unique strings")
        seen_sequence_identities.add(sequence.sequence_identity)
        x = sequence.expert_inputs
        expert_output_gradient = sequence.expert_output_gradients
        if (
            x.ndim != 2
            or x.shape[1] != hidden_dimension
            or expert_output_gradient.shape != x.shape
            or x.shape[0] == 0
            or not torch.is_floating_point(x)
            or not torch.is_floating_point(expert_output_gradient)
            or not bool(torch.isfinite(x).all())
            or not bool(torch.isfinite(expert_output_gradient).all())
        ):
            raise ValueError(
                f"curvature sequence {sequence.sequence_identity!r} has invalid rows"
            )
        x_gpu = x.to(device=device, dtype=torch.float32)
        output_gradient_gpu = expert_output_gradient.to(
            device=device, dtype=torch.float32
        )
        gate_values = F.linear(x_gpu, gate)
        up_values = F.linear(x_gpu, up)
        activated_gate = F.silu(gate_values)
        down_inputs = activated_gate * up_values
        hidden_gradient = output_gradient_gpu @ down
        gate_gradients = hidden_gradient * up_values * _silu_derivative(gate_values)
        up_gradients = hidden_gradient * activated_gate
        result["gate_proj"].append((x_gpu.cpu(), gate_gradients.cpu()))
        result["up_proj"].append((x_gpu.cpu(), up_gradients.cpu()))
        result["down_proj"].append((down_inputs.cpu(), output_gradient_gpu.cpu()))
        del (
            x_gpu,
            output_gradient_gpu,
            gate_values,
            up_values,
            activated_gate,
            down_inputs,
            hidden_gradient,
            gate_gradients,
            up_gradients,
        )
    return result


def _validate_sha256(value: object, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{role} must be a lowercase SHA-256 digest")
    return value


def read_moe_output_gradient_capture(
    capture_root: Path,
    *,
    layer: int,
    experts: Sequence[int],
) -> tuple[dict[int, list[RoutedGradientSequence]], dict[str, Any]]:
    """Read joined full-hidden MoE output gradients for selected experts."""

    root = capture_root.resolve(strict=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != MOE_OUTPUT_GRADIENT_CAPTURE_SCHEMA
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("model_layer") != layer
        or manifest.get("gradient_location")
        != "moe_output_before_residual_addition"
        or manifest.get("tensor_parallel_hidden_join_complete") is not True
    ):
        raise ValueError("MoE output-gradient capture manifest identity mismatch")
    plan_sha256 = _validate_sha256(
        manifest.get("corpus_plan_sha256"), role="corpus plan identity"
    )
    selected = tuple(int(expert) for expert in experts)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected experts must be a nonempty unique sequence")
    routed: dict[int, list[RoutedGradientSequence]] = {
        expert: [] for expert in selected
    }
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("MoE output-gradient capture has no sequence records")
    seen_files: set[str] = set()
    seen_sequence_identities: set[str] = set()
    seen_document_identities: set[str] = set()
    total_tokens = 0
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("MoE output-gradient capture record must be an object")
        filename = record.get("capture_file")
        sequence_identity = record.get("sequence_identity")
        document_identity = record.get("document_identity")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or filename in seen_files
        ):
            raise ValueError("gradient capture filenames must be unique basenames")
        if (
            not isinstance(sequence_identity, str)
            or not sequence_identity
            or sequence_identity in seen_sequence_identities
        ):
            raise ValueError("gradient capture sequence identities must be unique")
        if (
            not isinstance(document_identity, str)
            or not document_identity
            or document_identity in seen_document_identities
        ):
            raise ValueError(
                "gradient capture records must use distinct document identities"
            )
        seen_files.add(filename)
        seen_sequence_identities.add(sequence_identity)
        seen_document_identities.add(document_identity)
        path = root / filename
        if (
            path.stat().st_size != record.get("capture_file_bytes")
            or sha256_file(path) != record.get("capture_file_sha256")
        ):
            raise ValueError(f"gradient capture file {filename} failed byte closure")
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            expected_metadata = {
                "schema": f"{MOE_OUTPUT_GRADIENT_FILE_SCHEMA}_v1",
                "model_layer": str(layer),
                "corpus_plan_sha256": plan_sha256,
                "sequence_identity": sequence_identity,
                "document_identity": document_identity,
            }
            if metadata != expected_metadata:
                raise ValueError(f"gradient capture file {filename} metadata mismatch")
            hidden_states = handle.get_tensor("hidden_states")
            topk_ids = handle.get_tensor("topk_ids")
            topk_weights = handle.get_tensor("topk_weights")
            moe_output_gradients = handle.get_tensor("moe_output_gradients")
        token_count = int(hidden_states.shape[0]) if hidden_states.ndim == 2 else -1
        if (
            hidden_states.dtype != torch.bfloat16
            or tuple(hidden_states.shape) != (token_count, 6144)
            or topk_ids.dtype != torch.int32
            or tuple(topk_ids.shape) != (token_count, 8)
            or topk_weights.dtype != torch.float32
            or topk_weights.shape != topk_ids.shape
            or moe_output_gradients.dtype not in (torch.bfloat16, torch.float32)
            or tuple(moe_output_gradients.shape) != (token_count, 6144)
            or record.get("token_count") != token_count
            or token_count <= 0
        ):
            raise ValueError(f"gradient capture file {filename} tensor contract mismatch")
        if (
            not bool(torch.isfinite(hidden_states).all())
            or not bool(torch.isfinite(topk_weights).all())
            or not bool(torch.isfinite(moe_output_gradients).all())
            or bool((topk_ids < 0).any())
            or bool((topk_ids >= 256).any())
            or bool((topk_weights < 0).any())
        ):
            raise ValueError(f"gradient capture file {filename} contains invalid values")
        total_tokens += token_count
        for expert in selected:
            positions = (topk_ids == expert).nonzero(as_tuple=False)
            if positions.numel() == 0:
                continue
            token_ids = positions[:, 0]
            route_ids = positions[:, 1]
            if torch.unique(token_ids).numel() != token_ids.numel():
                raise ValueError(f"gradient capture routes expert {expert} twice per token")
            route_weights = topk_weights[token_ids, route_ids].float().unsqueeze(1)
            routed[expert].append(
                RoutedGradientSequence(
                    sequence_identity=sequence_identity,
                    expert_inputs=hidden_states.index_select(0, token_ids).contiguous(),
                    expert_output_gradients=(
                        moe_output_gradients.index_select(0, token_ids).float()
                        * route_weights
                    ).contiguous(),
                )
            )
    for expert, sequences in routed.items():
        if not sequences:
            raise ValueError(f"expert {expert} has no routed output-gradient rows")
    return routed, {
        "root": str(root),
        "manifest_sha256": sha256_file(manifest_path),
        "corpus_plan_sha256": plan_sha256,
        "sequence_count": len(records),
        "document_count": len(seen_document_identities),
        "token_count": total_tokens,
        "gradient_location": manifest["gradient_location"],
        "route_coefficient_application": (
            "reader_multiplied_moe_output_gradient_by_natural_route_weight"
        ),
    }


def _atomic_save_factor_tensors(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        save_file(
            {
                name: value.detach().cpu().float().contiguous()
                for name, value in tensors.items()
            },
            temporary,
            metadata={"format": DOWNSTREAM_CURVATURE_FACTOR_KIND},
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def curvature_factor_path(root: Path, layer: int, expert: int) -> Path:
    return root / "experts" / f"layer-{layer:03d}-expert-{expert:03d}.safetensors"


@torch.no_grad()
def build_expert_curvature_factors(
    *,
    source: IndexedTensorStore,
    sequences: Sequence[RoutedGradientSequence],
    layer: int,
    expert: int,
    input_identity_shrinkage: float,
    output_identity_shrinkage: float,
    device: torch.device,
    dest: Path,
) -> dict[str, Any]:
    """Derive and store all three projection factor pairs for one expert."""

    source_weights = {
        spec.name: source.get(source_tensor_name(layer, expert, spec.name))
        for spec in PROJECTIONS
    }
    projection_sequences = derive_projection_sequences(
        sequences,
        gate_weight=source_weights["gate_proj"],
        up_weight=source_weights["up_proj"],
        down_weight=source_weights["down_proj"],
        device=device,
    )
    tensors: dict[str, torch.Tensor] = {}
    projection_records: dict[str, Any] = {}
    for projection_name in CURVATURE_PROJECTION_NAMES:
        input_metric, output_metric, record = sequence_gradient_weight_curvature(
            projection_sequences[projection_name],
            input_identity_shrinkage=input_identity_shrinkage,
            output_identity_shrinkage=output_identity_shrinkage,
            device=device,
        )
        tensors[f"{projection_name}.input_metric"] = input_metric
        tensors[f"{projection_name}.output_metric"] = output_metric
        projection_records[projection_name] = {
            **record,
            "source_tensor": source_tensor_name(layer, expert, projection_name),
            "input_metric_sha256": tensor_sha256(input_metric),
            "output_metric_sha256": tensor_sha256(output_metric),
        }
        torch.cuda.empty_cache()
    path = curvature_factor_path(dest, layer, expert)
    _atomic_save_factor_tensors(path, tensors)
    return {
        "layer": layer,
        "expert": expert,
        "factor_file": path.name,
        "factor_file_bytes": path.stat().st_size,
        "factor_file_sha256": sha256_file(path),
        "sequence_identities": [
            sequence.sequence_identity for sequence in sequences
        ],
        "projection_factors": projection_records,
    }


def load_expert_curvature_factors(
    root: Path,
    record: Mapping[str, Any],
) -> dict[str, dict[str, torch.Tensor]]:
    """Load one expert's factors and verify file and tensor identities."""

    filename = record.get("factor_file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("curvature factor filename must be a basename")
    path = root / "experts" / filename
    if (
        path.stat().st_size != record.get("factor_file_bytes")
        or sha256_file(path) != record.get("factor_file_sha256")
    ):
        raise ValueError(f"curvature factor file {filename} failed byte closure")
    result: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        if (handle.metadata() or {}) != {"format": DOWNSTREAM_CURVATURE_FACTOR_KIND}:
            raise ValueError(f"curvature factor file {filename} metadata mismatch")
        for projection_name in CURVATURE_PROJECTION_NAMES:
            input_metric = handle.get_tensor(f"{projection_name}.input_metric")
            output_metric = handle.get_tensor(f"{projection_name}.output_metric")
            projection_record = record["projection_factors"][projection_name]
            if (
                input_metric.dtype != torch.float32
                or output_metric.dtype != torch.float32
                or input_metric.ndim != 2
                or input_metric.shape[0] != input_metric.shape[1]
                or output_metric.ndim != 2
                or output_metric.shape[0] != output_metric.shape[1]
                or not bool(torch.isfinite(input_metric).all())
                or not bool(torch.isfinite(output_metric).all())
                or tensor_sha256(input_metric)
                != projection_record.get("input_metric_sha256")
                or tensor_sha256(output_metric)
                != projection_record.get("output_metric_sha256")
            ):
                raise ValueError(
                    f"curvature factors for {projection_name} failed tensor closure"
                )
            result[projection_name] = {
                "input_metric": input_metric,
                "output_metric": output_metric,
            }
    return result


def validate_downstream_curvature_factor_artifact(
    root: Path,
) -> dict[str, Any]:
    """Validate a complete expert-local factor artifact without loading tensors."""

    resolved = root.resolve(strict=True)
    manifest_path = resolved / "manifest.json"
    report_path = resolved / "report.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha256 = sha256_file(manifest_path)
    report = json.loads(report_path.read_text())
    if (
        manifest.get("kind") != f"{DOWNSTREAM_CURVATURE_FACTOR_KIND}_manifest"
        or report.get("kind") != DOWNSTREAM_CURVATURE_FACTOR_KIND
        or report.get("status") != "complete"
        or report.get("manifest_sha256") != manifest_sha256
    ):
        raise ValueError("downstream-curvature factor artifact identity mismatch")
    records = report.get("experts")
    if not isinstance(records, list) or not records:
        raise ValueError("downstream-curvature factor report has no experts")
    experts: set[int] = set()
    total_bytes = 0
    for record in records:
        expert = record.get("expert")
        if (
            isinstance(expert, bool)
            or not isinstance(expert, int)
            or expert < 0
            or expert in experts
        ):
            raise ValueError("downstream-curvature factor report repeats an expert")
        experts.add(expert)
        filename = record.get("factor_file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("downstream-curvature factor filename is invalid")
        path = resolved / "experts" / filename
        if (
            path.stat().st_size != record.get("factor_file_bytes")
            or sha256_file(path) != record.get("factor_file_sha256")
        ):
            raise ValueError(f"curvature factor for expert {expert} failed closure")
        total_bytes += path.stat().st_size
    if (
        report.get("expert_count") != len(records)
        or report.get("factor_file_bytes") != total_bytes
    ):
        raise ValueError("downstream-curvature factor report totals do not close")
    return {
        "root": str(resolved),
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "report": report,
        "expert_ids": tuple(sorted(experts)),
        "factor_file_bytes": total_bytes,
    }


def run_downstream_curvature_factor_build(
    *,
    source_root: Path,
    source_inventory_path: Path,
    output_gradient_capture_root: Path,
    panel_manifest_path: Path,
    dest: Path,
    layer: int,
    expert_count: int,
    panel_offset: int,
    input_identity_shrinkage: float,
    output_identity_shrinkage: float,
    device: torch.device,
    verify_source_shard_hashes: bool = True,
) -> dict[str, Any]:
    """Build an immutable factor artifact for a frozen expert-panel slice."""

    frozen = load_frozen_real_weight_panel(panel_manifest_path, layer=layer)
    experts = select_frozen_panel_slice(
        frozen, offset=panel_offset, expert_count=expert_count
    )
    source_inventory = validate_bounded_source_window(
        source_root,
        source_inventory_path,
        panel={layer: experts},
        verify_shard_hashes=verify_source_shard_hashes,
    )
    routed, capture_record = read_moe_output_gradient_capture(
        output_gradient_capture_root,
        layer=layer,
        experts=experts,
    )
    manifest = {
        "kind": f"{DOWNSTREAM_CURVATURE_FACTOR_KIND}_manifest",
        "layer": layer,
        "source": source_inventory,
        "output_gradient_capture": capture_record,
        "frozen_panel": {
            "path": frozen["path"],
            "sha256": frozen["sha256"],
            "selected_offset": panel_offset,
            "selected_count": expert_count,
        },
        "panel": {str(layer): list(experts)},
        "estimator": {
            "name": "complete_sequence_weight_gradient_kronecker_grams",
            "input_identity_shrinkage": _validate_shrinkage(
                input_identity_shrinkage, role="input"
            ),
            "output_identity_shrinkage": _validate_shrinkage(
                output_identity_shrinkage, role="output"
            ),
            "projection_gradients": "exact_local_swiglu_chain_rule",
            "route_sampling": "natural_routes_only",
        },
        "device": str(device),
        "model_downloads_required": False,
        "complete_bf16_checkpoint_required": False,
        "evidence_boundary": (
            "fit-document curvature factors only; candidate selection and full-"
            "model KLD require separate document-disjoint inputs"
        ),
    }
    manifest_sha256 = prepare_destination(dest, manifest, resume=False)
    source = IndexedTensorStore(source_root)
    records: list[dict[str, Any]] = []
    for expert in experts:
        record = build_expert_curvature_factors(
            source=source,
            sequences=routed[expert],
            layer=layer,
            expert=expert,
            input_identity_shrinkage=input_identity_shrinkage,
            output_identity_shrinkage=output_identity_shrinkage,
            device=device,
            dest=dest,
        )
        record["manifest_sha256"] = manifest_sha256
        atomic_write_json(_expert_path(dest, layer, expert), record)
        records.append(record)
        print(
            f"[{len(records):02d}/{len(experts)}] layer {layer} expert {expert}: "
            f"{record['factor_file_bytes']:,} factor bytes",
            flush=True,
        )
    report = {
        "kind": DOWNSTREAM_CURVATURE_FACTOR_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "layer": layer,
        "expert_count": len(records),
        "factor_file_bytes": sum(record["factor_file_bytes"] for record in records),
        "panel": manifest["panel"],
        "experts": records,
        "evidence_boundary": manifest["evidence_boundary"],
    }
    atomic_write_json(dest / "report.json", report)
    validate_downstream_curvature_factor_artifact(dest)
    return report


__all__ = [
    "CURVATURE_PROJECTION_NAMES",
    "DOWNSTREAM_CURVATURE_FACTOR_KIND",
    "MOE_OUTPUT_GRADIENT_CAPTURE_SCHEMA",
    "RoutedGradientSequence",
    "build_expert_curvature_factors",
    "curvature_factor_path",
    "derive_projection_sequences",
    "load_expert_curvature_factors",
    "read_moe_output_gradient_capture",
    "run_downstream_curvature_factor_build",
    "sequence_gradient_weight_curvature",
    "validate_downstream_curvature_factor_artifact",
]

"""Numerical helpers for paired GLM-5.2 teacher-to-candidate KLD tests."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def forward_kld_per_position(
    reference_logits: torch.Tensor,
    model_logits: torch.Tensor,
    *,
    chunk_rows: int = 16,
    compute_device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return ``KL(reference || model)`` using bounded full-vocabulary chunks."""

    if (
        reference_logits.ndim != 2
        or model_logits.ndim != 2
        or reference_logits.shape != model_logits.shape
    ):
        raise ValueError("reference and model logits must have the same rank-two shape")
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")
    device = torch.device(compute_device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("KLD compute device must be a CPU or CUDA device")
    result = torch.empty(reference_logits.shape[0], dtype=torch.float64)
    with torch.inference_mode():
        for start in range(0, reference_logits.shape[0], chunk_rows):
            stop = min(reference_logits.shape[0], start + chunk_rows)
            reference = reference_logits[start:stop].to(
                device=device, dtype=torch.float32
            )
            model = model_logits[start:stop].to(device=device, dtype=torch.float32)
            log_reference = F.log_softmax(reference, dim=-1)
            log_model = F.log_softmax(model, dim=-1)
            values = F.kl_div(
                log_model,
                log_reference,
                reduction="none",
                log_target=True,
            ).sum(dim=-1)
            result[start:stop] = values.double().cpu()
    return result


def target_layer_routes(
    routed_experts: np.ndarray,
    *,
    model_layer: int,
    total_decoder_layers: int,
    first_moe_layer: int,
) -> np.ndarray:
    """Select one decoder layer from either dense-inclusive or MoE-only routes."""

    routes = np.asarray(routed_experts)
    if routes.ndim != 3:
        raise ValueError("routed experts must have shape [tokens, layers, top-k]")
    if routes.shape[1] == total_decoder_layers:
        layer_index = model_layer
    elif routes.shape[1] == total_decoder_layers - first_moe_layer:
        layer_index = model_layer - first_moe_layer
    else:
        raise ValueError(
            f"routed-expert layer axis has length {routes.shape[1]}, expected "
            f"{total_decoder_layers} or {total_decoder_layers - first_moe_layer}"
        )
    if not 0 <= layer_index < routes.shape[1]:
        raise ValueError("requested model layer is outside the routed-expert array")
    return np.ascontiguousarray(routes[:, layer_index, :])


def route_support_summary(
    routes: np.ndarray,
    *,
    selected_experts: Sequence[int],
) -> dict[str, Any]:
    """Count tokens and route entries that use the selected expert panel."""

    values = np.asarray(routes)
    if values.ndim != 2:
        raise ValueError("one layer's routes must have shape [tokens, top-k]")
    selected = np.asarray(tuple(int(value) for value in selected_experts), dtype=values.dtype)
    if selected.size == 0 or np.unique(selected).size != selected.size:
        raise ValueError("selected experts must be a nonempty unique sequence")
    selected_mask = np.isin(values, selected)
    counts = {
        str(expert): int(np.count_nonzero(values == expert))
        for expert in selected.tolist()
    }
    return {
        "token_count": int(values.shape[0]),
        "top_k": int(values.shape[1]),
        "selected_experts": selected.astype(int).tolist(),
        "selected_route_count": int(np.count_nonzero(selected_mask)),
        "selected_token_count": int(np.count_nonzero(np.any(selected_mask, axis=1))),
        "route_count_by_expert": counts,
    }


def paired_kld_summary(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, Any]:
    """Summarize paired per-position KLD without treating tokens as documents."""

    if baseline.ndim != 1 or candidate.shape != baseline.shape or baseline.numel() == 0:
        raise ValueError("paired KLD arrays must be nonempty vectors of equal length")
    baseline = baseline.double()
    candidate = candidate.double()
    if not bool(torch.isfinite(baseline).all()) or not bool(torch.isfinite(candidate).all()):
        raise ValueError("paired KLD arrays must be finite")
    difference = candidate - baseline
    baseline_mean = float(baseline.mean().item())
    candidate_mean = float(candidate.mean().item())
    return {
        "position_count": baseline.numel(),
        "baseline_mean_forward_kld": baseline_mean,
        "candidate_mean_forward_kld": candidate_mean,
        "candidate_minus_baseline_mean_forward_kld": float(difference.mean().item()),
        "relative_forward_kld_reduction": (
            1.0 - candidate_mean / baseline_mean if baseline_mean > 0.0 else None
        ),
        "candidate_better_position_count": int(torch.count_nonzero(difference < 0.0)),
        "candidate_equal_position_count": int(torch.count_nonzero(difference == 0.0)),
        "candidate_worse_position_count": int(torch.count_nonzero(difference > 0.0)),
        "paired_difference_standard_deviation": float(
            difference.std(unbiased=True).item() if difference.numel() > 1 else 0.0
        ),
        "evidence_boundary": (
            "paired token-position statistics from one reference context; token "
            "positions are correlated and do not constitute document-level replicates"
        ),
    }


__all__ = [
    "forward_kld_per_position",
    "paired_kld_summary",
    "route_support_summary",
    "target_layer_routes",
]

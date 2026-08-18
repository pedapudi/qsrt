"""Kimi-K3 layer-92 suffix gradients for output-side curvature capture."""

from __future__ import annotations

import math

import torch


def _require_last_dimension(
    value: torch.Tensor,
    dimension: int,
    name: str,
) -> None:
    if not value.is_floating_point() or value.ndim < 1:
        raise TypeError(f"{name} must be a floating-point tensor")
    if value.shape[-1] != dimension:
        raise ValueError(f"{name} has the wrong trailing dimension")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must be finite")


def rms_norm_pullback(
    inputs: torch.Tensor,
    output_gradient: torch.Tensor,
    weight: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Pull a gradient through ``weight * inputs / rms(inputs)``."""

    if inputs.shape != output_gradient.shape:
        raise ValueError("RMSNorm input and output gradient shapes must match")
    dimension = int(inputs.shape[-1])
    _require_last_dimension(inputs, dimension, "RMSNorm input")
    _require_last_dimension(output_gradient, dimension, "RMSNorm output gradient")
    _require_last_dimension(weight, dimension, "RMSNorm weight")
    if weight.ndim != 1:
        raise ValueError("RMSNorm weight must be one-dimensional")
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("RMSNorm epsilon must be finite and nonnegative")

    reciprocal_rms = torch.rsqrt(inputs.square().mean(dim=-1, keepdim=True) + epsilon)
    weighted_gradient = output_gradient * weight
    radial = (weighted_gradient * inputs).mean(dim=-1, keepdim=True)
    return reciprocal_rms * (
        weighted_gradient - inputs * reciprocal_rms.square() * radial
    )


def attention_residual_prefix_pullback(
    updated_prefix: torch.Tensor,
    mixed_output: torch.Tensor,
    prefix_weight: torch.Tensor,
    output_gradient: torch.Tensor,
    score_query: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Pull a final attention-residual gradient into its updated prefix source."""

    if not (
        updated_prefix.shape == mixed_output.shape == output_gradient.shape
    ):
        raise ValueError("attention-residual vector shapes must match")
    dimension = int(updated_prefix.shape[-1])
    _require_last_dimension(updated_prefix, dimension, "updated prefix")
    _require_last_dimension(mixed_output, dimension, "mixed output")
    _require_last_dimension(output_gradient, dimension, "mixed-output gradient")
    _require_last_dimension(score_query, dimension, "attention-residual score query")
    if score_query.ndim != 1:
        raise ValueError("attention-residual score query must be one-dimensional")
    if prefix_weight.shape != updated_prefix.shape[:-1]:
        raise ValueError("prefix weight shape must match the vector batch shape")
    if not prefix_weight.is_floating_point():
        raise TypeError("prefix weights must be floating point")
    if not bool(torch.all(torch.isfinite(prefix_weight))):
        raise ValueError("prefix weights must be finite")
    if not bool(torch.all((prefix_weight >= 0) & (prefix_weight <= 1))):
        raise ValueError("prefix weights must lie in [0, 1]")
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("attention-residual epsilon must be finite and nonnegative")

    reciprocal_rms = torch.rsqrt(
        updated_prefix.square().mean(dim=-1, keepdim=True) + epsilon
    )
    query_dot = (updated_prefix * score_query).sum(dim=-1, keepdim=True)
    score_gradient = reciprocal_rms * score_query - (
        updated_prefix
        * reciprocal_rms.pow(3)
        * query_dot
        / float(dimension)
    )
    mixture_dot = (
        (updated_prefix - mixed_output) * output_gradient
    ).sum(dim=-1, keepdim=True)
    return prefix_weight.unsqueeze(-1) * (
        output_gradient + score_gradient * mixture_dot
    )


def routed_output_pullback(
    routed_latent: torch.Tensor,
    hidden_gradient: torch.Tensor,
    norm_weight: torch.Tensor,
    projection_weight: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Pull a hidden-space gradient through the routed RMSNorm/up projection."""

    if projection_weight.ndim != 2 or not projection_weight.is_floating_point():
        raise TypeError("routed output projection must be a floating-point matrix")
    hidden_dimension, latent_dimension = map(int, projection_weight.shape)
    _require_last_dimension(routed_latent, latent_dimension, "routed latent")
    _require_last_dimension(hidden_gradient, hidden_dimension, "hidden gradient")
    if routed_latent.shape[:-1] != hidden_gradient.shape[:-1]:
        raise ValueError("routed latent and hidden gradient batches must match")
    _require_last_dimension(norm_weight, latent_dimension, "routed RMSNorm weight")
    if norm_weight.ndim != 1:
        raise ValueError("routed RMSNorm weight must be one-dimensional")
    if not bool(torch.all(torch.isfinite(projection_weight))):
        raise ValueError("routed output projection must be finite")

    normalized_gradient = hidden_gradient @ projection_weight
    return rms_norm_pullback(
        routed_latent,
        normalized_gradient,
        norm_weight,
        epsilon=epsilon,
    )


def kimi_layer92_routed_fisher_pullback(
    routed_latent: torch.Tensor,
    updated_prefix: torch.Tensor,
    final_mixed: torch.Tensor,
    prefix_weight: torch.Tensor,
    normalized_hidden_gradient: torch.Tensor,
    *,
    final_norm_weight: torch.Tensor,
    final_norm_epsilon: float,
    attention_score_query: torch.Tensor,
    attention_epsilon: float,
    routed_norm_weight: torch.Tensor,
    routed_norm_epsilon: float,
    routed_projection_weight: torch.Tensor,
) -> torch.Tensor:
    """Pull a pre-LM-head gradient into the aggregated routed W2 output."""

    mixed_gradient = rms_norm_pullback(
        final_mixed,
        normalized_hidden_gradient,
        final_norm_weight,
        epsilon=final_norm_epsilon,
    )
    prefix_gradient = attention_residual_prefix_pullback(
        updated_prefix,
        final_mixed,
        prefix_weight,
        mixed_gradient,
        attention_score_query,
        epsilon=attention_epsilon,
    )
    return routed_output_pullback(
        routed_latent,
        prefix_gradient,
        routed_norm_weight,
        routed_projection_weight,
        epsilon=routed_norm_epsilon,
    )


def paired_lm_head_fisher_gradients(
    lm_head_weight: torch.Tensor,
    first_tokens: torch.Tensor,
    second_tokens: torch.Tensor,
    *,
    logit_scale: float = 1.0,
) -> torch.Tensor:
    """Return unbiased two-sample softmax-Fisher gradients before the LM head."""

    if lm_head_weight.ndim != 2 or not lm_head_weight.is_floating_point():
        raise TypeError("LM-head weight must be a floating-point matrix")
    if first_tokens.shape != second_tokens.shape:
        raise ValueError("paired token-index shapes must match")
    if first_tokens.dtype != torch.long or second_tokens.dtype != torch.long:
        raise TypeError("paired token indices must use torch.long")
    if not math.isfinite(logit_scale):
        raise ValueError("logit scale must be finite")
    vocabulary = int(lm_head_weight.shape[0])
    if bool(torch.any((first_tokens < 0) | (first_tokens >= vocabulary))):
        raise ValueError("first token index lies outside the LM-head vocabulary")
    if bool(torch.any((second_tokens < 0) | (second_tokens >= vocabulary))):
        raise ValueError("second token index lies outside the LM-head vocabulary")

    first = lm_head_weight.index_select(0, first_tokens.reshape(-1))
    second = lm_head_weight.index_select(0, second_tokens.reshape(-1))
    gradients = (first - second) * (float(logit_scale) / math.sqrt(2.0))
    return gradients.reshape(*first_tokens.shape, lm_head_weight.shape[1])


def empirical_fisher_factor(
    gradients: torch.Tensor,
    *,
    damping_ratio: float = 0.0,
) -> torch.Tensor:
    """Build a mean outer-product factor with trace-scaled diagonal damping."""

    if gradients.ndim != 2 or not gradients.is_floating_point():
        raise TypeError("Fisher gradients must be a floating-point matrix")
    if gradients.shape[0] == 0:
        raise ValueError("Fisher gradients must contain at least one row")
    if not bool(torch.all(torch.isfinite(gradients))):
        raise ValueError("Fisher gradients must be finite")
    if not math.isfinite(damping_ratio) or damping_ratio < 0.0:
        raise ValueError("damping ratio must be finite and nonnegative")

    work = gradients.float()
    factor = work.T @ work
    factor /= float(work.shape[0])
    factor = (factor + factor.T) * 0.5
    if damping_ratio:
        diagonal_mean = torch.diagonal(factor).mean()
        factor.diagonal().add_(float(damping_ratio) * diagonal_mean)
    return factor.contiguous()


def _sketch_a_rows(
    inputs: torch.Tensor,
    output_gradients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.ndim != 2 or not inputs.is_floating_point():
        raise TypeError("Sketch-A inputs must be a floating-point matrix")
    if output_gradients.ndim != 2 or not output_gradients.is_floating_point():
        raise TypeError(
            "Sketch-A output gradients must be a floating-point matrix"
        )
    if inputs.shape[0] == 0 or output_gradients.shape[0] != inputs.shape[0]:
        raise ValueError("Sketch-A inputs and gradients must contain aligned rows")
    if inputs.device != output_gradients.device:
        raise ValueError("Sketch-A inputs and gradients must share one device")
    if not bool(torch.all(torch.isfinite(inputs))) or not bool(
        torch.all(torch.isfinite(output_gradients))
    ):
        raise ValueError("Sketch-A inputs and gradients must be finite")
    work_dtype = (
        torch.float64
        if torch.float64 in (inputs.dtype, output_gradients.dtype)
        else torch.float32
    )
    return inputs.to(dtype=work_dtype), output_gradients.to(dtype=work_dtype)


def _sketch_a_factor(
    factor: torch.Tensor,
    *,
    dimension: int,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if (
        factor.ndim != 2
        or tuple(factor.shape) != (dimension, dimension)
        or not factor.is_floating_point()
    ):
        raise ValueError(f"{name} must be a square factor matching its rows")
    if factor.device != device:
        raise ValueError(f"{name} must share the sample device")
    work = factor.to(dtype=dtype)
    if not bool(torch.all(torch.isfinite(work))):
        raise ValueError(f"{name} must be finite")
    scale = float(work.abs().max())
    tolerance = 1e-10 if dtype == torch.float64 else 1e-5
    if not torch.allclose(
        work,
        work.T,
        rtol=tolerance,
        atol=tolerance * max(scale, 1.0),
    ):
        raise ValueError(f"{name} must be symmetric")
    return ((work + work.T) * 0.5).contiguous()


def _sketch_a_update(
    rows: torch.Tensor,
    quadratic_weights: torch.Tensor,
    factor_norm_squared: torch.Tensor,
    *,
    damping_ratio: float,
) -> torch.Tensor:
    if not math.isfinite(damping_ratio) or damping_ratio < 0.0:
        raise ValueError("Sketch-A damping ratio must be finite and nonnegative")
    weight_scale = float(quadratic_weights.abs().max())
    tolerance = (1e-10 if rows.dtype == torch.float64 else 2e-5) * max(
        weight_scale, 1.0
    )
    if float(quadratic_weights.min()) < -tolerance:
        raise ValueError("Sketch-A conditioning factor is not positive semidefinite")
    weights = quadratic_weights.clamp_min(0)
    if not bool(torch.isfinite(factor_norm_squared)) or float(
        factor_norm_squared
    ) <= 0.0:
        raise ValueError("Sketch-A conditioning factor has zero Frobenius norm")
    updated = rows.T @ (rows * weights[:, None])
    updated /= float(rows.shape[0]) * factor_norm_squared
    updated = (updated + updated.T) * 0.5
    if damping_ratio:
        diagonal_mean = torch.diagonal(updated).mean()
        updated.diagonal().add_(float(damping_ratio) * diagonal_mean)
    return updated.contiguous()


def sketch_a_input_factor_update(
    inputs: torch.Tensor,
    output_gradients: torch.Tensor,
    output_factor: torch.Tensor,
    *,
    damping_ratio: float = 0.0,
) -> torch.Tensor:
    """Apply one YAQA Sketch-A power step to the input-side factor.

    Each row represents one paired linear-layer input and output gradient.
    Any sample weighting, including an applied MoE route weight, must already
    be present in exactly one of those two row matrices.
    """

    inputs, output_gradients = _sketch_a_rows(inputs, output_gradients)
    factor = _sketch_a_factor(
        output_factor,
        dimension=int(output_gradients.shape[1]),
        name="Sketch-A output factor",
        dtype=inputs.dtype,
        device=inputs.device,
    )
    weights = torch.einsum(
        "bi,ij,bj->b", output_gradients, factor, output_gradients
    )
    return _sketch_a_update(
        inputs,
        weights,
        factor.square().sum(),
        damping_ratio=damping_ratio,
    )


def sketch_a_output_factor_update(
    inputs: torch.Tensor,
    output_gradients: torch.Tensor,
    input_factor: torch.Tensor,
    *,
    damping_ratio: float = 0.0,
) -> torch.Tensor:
    """Apply one YAQA Sketch-A power step to the output-side factor."""

    inputs, output_gradients = _sketch_a_rows(inputs, output_gradients)
    factor = _sketch_a_factor(
        input_factor,
        dimension=int(inputs.shape[1]),
        name="Sketch-A input factor",
        dtype=inputs.dtype,
        device=inputs.device,
    )
    weights = torch.einsum("bi,ij,bj->b", inputs, factor, inputs)
    return _sketch_a_update(
        output_gradients,
        weights,
        factor.square().sum(),
        damping_ratio=damping_ratio,
    )

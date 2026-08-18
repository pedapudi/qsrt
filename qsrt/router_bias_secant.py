"""Secant-calibrated selection-bias updates from measured router frequencies."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SecantBiasUpdate:
    update: torch.Tensor
    bounded_proposal: torch.Tensor
    slopes: torch.Tensor
    resolved: torch.Tensor
    valid: torch.Tensor
    standard_error: torch.Tensor
    raw_proposal: torch.Tensor
    clamp: float


@dataclass(frozen=True)
class LeastSquaresSecantBiasUpdate:
    update: torch.Tensor
    bounded_proposal: torch.Tensor
    slopes: torch.Tensor
    slope_standard_error: torch.Tensor
    slope_resolved: torch.Tensor
    residual_resolved: torch.Tensor
    valid: torch.Tensor
    residual_standard_error: torch.Tensor
    raw_proposal: torch.Tensor
    predicted_frequency: torch.Tensor
    clamp: float


def build_secant_bias_update(
    *,
    teacher_counts: torch.Tensor,
    slope_before_counts: torch.Tensor,
    slope_after_counts: torch.Tensor,
    base_counts: torch.Tensor,
    slope_before_bias: torch.Tensor,
    slope_after_bias: torch.Tensor,
    tokens: int,
    median_margin: float,
    noise_sigma: float = 2.5,
    margin_multiple: float = 32.0,
) -> SecantBiasUpdate:
    """Build one bounded secant update for a single router layer."""

    tensors = (
        teacher_counts,
        slope_before_counts,
        slope_after_counts,
        base_counts,
        slope_before_bias,
        slope_after_bias,
    )
    if any(value.ndim != 1 for value in tensors):
        raise ValueError("secant inputs must be one-dimensional")
    if len({value.shape for value in tensors}) != 1:
        raise ValueError("secant inputs have different expert geometry")
    if tokens <= 0 or noise_sigma < 0 or margin_multiple <= 0:
        raise ValueError("tokens and secant bounds must be positive")
    if median_margin <= 0:
        raise ValueError("median selection margin must be positive")

    counts0 = slope_before_counts.to(torch.float64)
    counts1 = slope_after_counts.to(torch.float64)
    teacher_frequency = teacher_counts.to(torch.float64) / tokens
    base_frequency = base_counts.to(torch.float64) / tokens
    frequency0 = counts0 / tokens
    frequency1 = counts1 / tokens
    frequency_delta = frequency1 - frequency0
    standard_error = torch.sqrt(
        (
            frequency0 * (1.0 - frequency0)
            + frequency1 * (1.0 - frequency1)
        )
        / tokens
    )
    resolved = frequency_delta.abs() > noise_sigma * standard_error

    bias_delta = (
        slope_after_bias.to(torch.float64) - slope_before_bias.to(torch.float64)
    )
    bias_delta -= bias_delta.mean()
    nonzero_bias = bias_delta.abs() > torch.finfo(torch.float64).eps
    slopes = torch.zeros_like(frequency_delta)
    slopes[nonzero_bias] = frequency_delta[nonzero_bias] / bias_delta[nonzero_bias]
    valid = resolved & nonzero_bias & torch.isfinite(slopes) & (slopes > 0)

    raw_proposal = torch.zeros_like(slopes)
    raw_proposal[valid] = -(
        base_frequency[valid] - teacher_frequency[valid]
    ) / slopes[valid]
    bound = float(margin_multiple * median_margin)
    bounded_proposal = torch.zeros_like(slopes)
    bounded_proposal[valid] = torch.clamp(
        raw_proposal[valid], min=-bound, max=bound
    )
    # A common bias shift is exactly selection-null. Remove it after bounding
    # the causal per-expert proposals so the serialized layer has zero drift.
    update = bounded_proposal - bounded_proposal.mean()
    return SecantBiasUpdate(
        update=update.to(torch.float32),
        bounded_proposal=bounded_proposal.to(torch.float32),
        slopes=slopes.to(torch.float32),
        resolved=resolved,
        valid=valid,
        standard_error=standard_error.to(torch.float32),
        raw_proposal=raw_proposal.to(torch.float32),
        clamp=bound,
    )


def build_least_squares_secant_bias_update(
    *,
    teacher_counts: torch.Tensor,
    round_counts: torch.Tensor,
    round_biases: torch.Tensor,
    tokens: int,
    median_margin: float,
    noise_sigma: float = 2.5,
    margin_multiple: float = 64.0,
) -> LeastSquaresSecantBiasUpdate:
    """Fit one bounded per-expert update from repeated bias/frequency pairs."""

    if teacher_counts.ndim != 1:
        raise ValueError("teacher counts must be one-dimensional")
    if round_counts.ndim != 2 or round_biases.ndim != 2:
        raise ValueError("round tensors must be two-dimensional")
    if round_counts.shape != round_biases.shape:
        raise ValueError("round count and bias tensors have different geometry")
    if round_counts.shape[1:] != teacher_counts.shape:
        raise ValueError("teacher and round tensors have different expert geometry")
    if round_counts.shape[0] < 3:
        raise ValueError("least-squares secant fitting requires at least three rounds")
    if tokens <= 0 or noise_sigma < 0 or margin_multiple <= 0:
        raise ValueError("tokens and secant bounds must be positive")
    if median_margin <= 0:
        raise ValueError("median selection margin must be positive")

    counts = round_counts.to(torch.float64)
    frequencies = counts / tokens
    biases = round_biases.to(torch.float64)
    # A common shift of every expert bias is selection-null. Remove it from
    # each measured round before estimating a causal response.
    biases = biases - biases.mean(dim=1, keepdim=True)
    x = biases - biases.mean(dim=0, keepdim=True)
    y = frequencies - frequencies.mean(dim=0, keepdim=True)
    denominator = (x * x).sum(dim=0)
    nonzero = denominator > torch.finfo(torch.float64).eps
    slopes = torch.zeros_like(denominator)
    slopes[nonzero] = (x[:, nonzero] * y[:, nonzero]).sum(dim=0) / denominator[nonzero]

    frequency_variance = frequencies * (1.0 - frequencies) / tokens
    slope_variance = torch.zeros_like(denominator)
    slope_variance[nonzero] = (
        (x[:, nonzero].square() * frequency_variance[:, nonzero]).sum(dim=0)
        / denominator[nonzero].square()
    )
    slope_standard_error = torch.sqrt(torch.clamp_min(slope_variance, 0.0))
    slope_resolved = (
        nonzero
        & torch.isfinite(slopes)
        & torch.isfinite(slope_standard_error)
        & (slopes > noise_sigma * slope_standard_error)
    )

    teacher_frequency = teacher_counts.to(torch.float64) / tokens
    base_frequency = frequencies[-1]
    residual = base_frequency - teacher_frequency
    residual_standard_error = torch.sqrt(
        (
            base_frequency * (1.0 - base_frequency)
            + teacher_frequency * (1.0 - teacher_frequency)
        )
        / tokens
    )
    residual_resolved = residual.abs() > noise_sigma * residual_standard_error
    valid = slope_resolved & residual_resolved

    raw_proposal = torch.zeros_like(slopes)
    raw_proposal[valid] = -residual[valid] / slopes[valid]
    bound = float(margin_multiple * median_margin)
    bounded_proposal = torch.zeros_like(slopes)
    bounded_proposal[valid] = torch.clamp(
        raw_proposal[valid], min=-bound, max=bound
    )
    predicted_frequency = base_frequency + slopes * bounded_proposal
    return LeastSquaresSecantBiasUpdate(
        update=bounded_proposal.to(torch.float32),
        bounded_proposal=bounded_proposal.to(torch.float32),
        slopes=slopes.to(torch.float32),
        slope_standard_error=slope_standard_error.to(torch.float32),
        slope_resolved=slope_resolved,
        residual_resolved=residual_resolved,
        valid=valid,
        residual_standard_error=residual_standard_error.to(torch.float32),
        raw_proposal=raw_proposal.to(torch.float32),
        predicted_frequency=predicted_frequency.to(torch.float32),
        clamp=bound,
    )

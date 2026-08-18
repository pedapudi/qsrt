"""Functional distortion diagnostics for routed expert reconstructions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qsrt.coupled_expert_study import RoutedOutputMetric


@dataclass(frozen=True)
class ErrorGeometry:
    """Residual-stream error before and after the following output mapping."""

    residual_sse: float
    radial_sse: float
    tangential_sse: float
    mapped_linear_sse: float
    mapped_radial_sse: float
    mapped_tangential_sse: float
    mapped_exact_sse: float


def quadratic_matrix_sse(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    covariance: torch.Tensor,
) -> tuple[float, float]:
    """Return ``tr(E H E^T)`` and the corresponding reference energy."""

    if reference.ndim != 2 or candidate.shape != reference.shape:
        raise ValueError("matrix distortion requires matching rank-two weights")
    if covariance.ndim != 2 or tuple(covariance.shape) != (
        reference.shape[1],
        reference.shape[1],
    ):
        raise ValueError("matrix distortion covariance has the wrong shape")
    if not all(
        bool(torch.all(torch.isfinite(value)))
        for value in (reference, candidate, covariance)
    ):
        raise ValueError("matrix distortion inputs must be finite")
    dtype = torch.float64
    device = covariance.device
    source = reference.to(device=device, dtype=dtype)
    error = candidate.to(device=device, dtype=dtype) - source
    hessian = covariance.to(device=device, dtype=dtype)
    numerator = torch.sum((error @ hessian) * error)
    denominator = torch.sum((source @ hessian) * source)
    return float(numerator), float(denominator)


def two_sided_encoder_sse(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    input_hessian: torch.Tensor,
    output_hessian: torch.Tensor,
) -> tuple[float, float]:
    """Return the two-sided quadratic loss for ``[input, output]`` weights.

    The numerator is ``tr(H_O E.T H_I E)``.  The denominator applies the
    same metric to ``reference`` so callers can form a relative distortion.
    """

    if reference.ndim != 2 or candidate.shape != reference.shape:
        raise ValueError("two-sided distortion requires matching rank-two weights")
    expected_input = (reference.shape[0], reference.shape[0])
    expected_output = (reference.shape[1], reference.shape[1])
    if input_hessian.ndim != 2 or tuple(input_hessian.shape) != expected_input:
        raise ValueError("input Hessian has the wrong shape")
    if output_hessian.ndim != 2 or tuple(output_hessian.shape) != expected_output:
        raise ValueError("output Hessian has the wrong shape")
    if not all(
        bool(torch.all(torch.isfinite(value)))
        for value in (reference, candidate, input_hessian, output_hessian)
    ):
        raise ValueError("two-sided distortion inputs must be finite")
    device = input_hessian.device
    dtype = torch.float64
    source = reference.to(device=device, dtype=dtype)
    error = candidate.to(device=device, dtype=dtype) - source
    h_input = input_hessian.to(device=device, dtype=dtype)
    h_output = output_hessian.to(device=device, dtype=dtype)

    def objective(value: torch.Tensor) -> torch.Tensor:
        return torch.sum(value * (h_input @ value @ h_output))

    return float(objective(error)), float(objective(source))


def mapped_output_hessian(
    aggregate: torch.Tensor,
    metric: RoutedOutputMetric,
    *,
    row_weights: torch.Tensor | None = None,
    chunk_rows: int = 4096,
    accumulation_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Average the exact local output metric after RMSNorm and projection.

    For aggregate row ``u`` and perturbation ``v``, the returned matrix ``H``
    satisfies ``v.T @ H @ v = mean(||J(u) @ v||^2)``.  Optional row weights
    are applied exactly once and normalized by their total mass.
    """

    if aggregate.ndim != 2 or aggregate.shape[1] != metric.gain.numel():
        raise ValueError("aggregate has the wrong output-metric shape")
    if not bool(torch.all(torch.isfinite(aggregate))):
        raise ValueError("aggregate must be finite")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if accumulation_dtype not in (torch.float32, torch.float64):
        raise ValueError("accumulation_dtype must be float32 or float64")
    if row_weights is None:
        row_weights = torch.ones(
            aggregate.shape[0], device=aggregate.device, dtype=accumulation_dtype
        )
    elif row_weights.ndim != 1 or row_weights.shape[0] != aggregate.shape[0]:
        raise ValueError("row weights must align with aggregate rows")
    elif not bool(torch.all(torch.isfinite(row_weights))) or bool(
        torch.any(row_weights < 0)
    ):
        raise ValueError("row weights must be finite and nonnegative")

    device = aggregate.device
    gain = metric.gain.to(device=device, dtype=accumulation_dtype)
    projection = metric.projection.to(device=device, dtype=accumulation_dtype)
    scaled_projection = projection * gain[None, :]
    base_metric = scaled_projection.T @ scaled_projection
    dimension = aggregate.shape[1]
    weighted_ab_outer = torch.zeros_like(base_metric)
    weighted_quadratic_outer = torch.zeros_like(base_metric)
    weighted_a2 = torch.zeros((), device=device, dtype=accumulation_dtype)
    total_weight = torch.zeros((), device=device, dtype=accumulation_dtype)

    for begin in range(0, aggregate.shape[0], chunk_rows):
        end = min(begin + chunk_rows, aggregate.shape[0])
        rows = aggregate[begin:end].to(dtype=accumulation_dtype)
        weights = row_weights[begin:end].to(
            device=device, dtype=accumulation_dtype
        )
        rms = torch.sqrt(rows.square().mean(dim=1) + metric.epsilon)
        inverse_rms = rms.reciprocal()
        rank_one_scale = (dimension * rms.pow(3)).reciprocal()
        ab_weights = weights * inverse_rms * rank_one_scale
        weighted_ab_outer.add_(rows.T @ (rows * ab_weights[:, None]))

        projected_rows = (rows * gain[None, :]) @ projection.T
        row_quadratic = projected_rows.square().sum(dim=1)
        quadratic_weights = weights * rank_one_scale.square() * row_quadratic
        weighted_quadratic_outer.add_(
            rows.T @ (rows * quadratic_weights[:, None])
        )
        weighted_a2.add_(torch.sum(weights * inverse_rms.square()))
        total_weight.add_(weights.sum())

    if not bool(total_weight > 0):
        raise ValueError("row weights must have positive mass")
    cross = weighted_ab_outer @ base_metric
    result = (
        weighted_a2 * base_metric
        - cross
        - cross.T
        + weighted_quadratic_outer
    ) / total_weight
    return ((result + result.T) * 0.5).contiguous()


def routed_error_geometry(
    aggregate: torch.Tensor,
    routed_error: torch.Tensor,
    metric: RoutedOutputMetric,
) -> ErrorGeometry:
    """Split routed error into aggregate-radial and tangential components."""

    if aggregate.ndim != 2 or routed_error.shape != aggregate.shape:
        raise ValueError("aggregate and routed error must be aligned rank-two tensors")
    if not all(
        bool(torch.all(torch.isfinite(value)))
        for value in (aggregate, routed_error)
    ):
        raise ValueError("routed error geometry requires finite tensors")
    source = aggregate.float()
    error = routed_error.float()
    source_square = source.square().sum(dim=1, keepdim=True).clamp_min(1e-30)
    radial = source * ((source * error).sum(dim=1, keepdim=True) / source_square)
    tangential = error - radial
    mapped = metric.jacobian_vectors(source, error)
    mapped_radial = metric.jacobian_vectors(source, radial)
    mapped_tangential = metric.jacobian_vectors(source, tangential)
    exact = metric.exact_delta(source, error)
    return ErrorGeometry(
        residual_sse=float(error.double().square().sum()),
        radial_sse=float(radial.double().square().sum()),
        tangential_sse=float(tangential.double().square().sum()),
        mapped_linear_sse=float(mapped.double().square().sum()),
        mapped_radial_sse=float(mapped_radial.double().square().sum()),
        mapped_tangential_sse=float(mapped_tangential.double().square().sum()),
        mapped_exact_sse=float(exact.double().square().sum()),
    )


__all__ = [
    "ErrorGeometry",
    "mapped_output_hessian",
    "quadratic_matrix_sse",
    "routed_error_geometry",
    "two_sided_encoder_sse",
]

"""Linear objective guidance for fixed-format trellis quantization.

The decoder and payload remain unchanged. A gradient measured at an identified
anchor changes only the offline quantization target. Complete-square target
shifts let the existing Viterbi and two-sided rounding implementations optimize
the quadratic-plus-linear objective without another CUDA trellis kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ViterbiGradientGuidance:
    """A deterministic objective gradient and the anchor that defines it."""

    gradient: torch.Tensor
    anchor: torch.Tensor
    anchor_id: str
    objective_id: str
    strength: float = 1.0

    def validate(self, shape: torch.Size | tuple[int, ...]) -> None:
        expected = tuple(shape)
        if tuple(self.gradient.shape) != expected or tuple(self.anchor.shape) != expected:
            raise ValueError("gradient guidance tensors do not match the encoded matrix")
        if not self.gradient.is_floating_point() or not self.anchor.is_floating_point():
            raise TypeError("gradient guidance tensors must be floating point")
        if not bool(torch.all(torch.isfinite(self.gradient))):
            raise ValueError("objective gradient must be finite")
        if not bool(torch.all(torch.isfinite(self.anchor))):
            raise ValueError("gradient anchor must be finite")
        if not self.anchor_id or not self.objective_id:
            raise ValueError("gradient guidance requires anchor and objective identities")
        if not math.isfinite(self.strength) or self.strength < 0.0:
            raise ValueError("gradient strength must be finite and nonnegative")


@dataclass(frozen=True)
class LowRankViterbiGradientGuidance:
    """A factorized objective gradient and its full decoded anchor.

    ``left @ right`` has the same shape and semantics as the dense gradient in
    :class:`ViterbiGradientGuidance`.  Keeping the rank factors avoids
    materializing a 44 MiB W1 or W3 gradient for every routed expert.
    """

    left: torch.Tensor
    right: torch.Tensor
    anchor: torch.Tensor
    anchor_id: str
    objective_id: str
    strength: float = 1.0

    def validate(self, shape: torch.Size | tuple[int, ...]) -> None:
        expected = tuple(shape)
        if len(expected) != 2:
            raise ValueError("low-rank gradient guidance requires a matrix")
        rows, columns = expected
        if self.left.ndim != 2 or self.right.ndim != 2:
            raise ValueError("low-rank gradient factors must be matrices")
        rank = self.left.shape[1]
        if (
            self.left.shape[0] != rows
            or self.right.shape != (rank, columns)
            or self.anchor.shape != expected
        ):
            raise ValueError("low-rank gradient guidance has incompatible geometry")
        if rank <= 0:
            raise ValueError("low-rank gradient guidance must have positive rank")
        if not all(
            value.is_floating_point()
            for value in (self.left, self.right, self.anchor)
        ):
            raise TypeError("gradient guidance tensors must be floating point")
        if not all(
            bool(torch.all(torch.isfinite(value)))
            for value in (self.left, self.right, self.anchor)
        ):
            raise ValueError("gradient guidance tensors must be finite")
        if not self.anchor_id or not self.objective_id:
            raise ValueError("gradient guidance requires anchor and objective identities")
        if not math.isfinite(self.strength) or self.strength < 0.0:
            raise ValueError("gradient strength must be finite and nonnegative")

    def rows(
        self,
        row_slice: slice,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Materialize only the requested gradient rows."""

        left = self.left[row_slice].to(device=device, dtype=dtype)
        right = self.right.to(device=device, dtype=dtype)
        return left @ right

    def linear_term(
        self,
        candidate: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> float:
        """Evaluate ``strength * <left @ right, candidate - anchor>``."""

        delta = candidate.to(dtype=dtype) - self.anchor.to(
            device=candidate.device,
            dtype=dtype,
        )
        left = self.left.to(device=candidate.device, dtype=dtype)
        right = self.right.to(device=candidate.device, dtype=dtype)
        # <L R, D> = tr(L.T D R.T).  Contract the low-rank dimension last so
        # the full gradient matrix is never materialized.
        value = torch.sum((left.T @ delta) * right)
        return float(float(self.strength) * value.item())


def shift_elementwise_target(
    target: torch.Tensor,
    gradient: torch.Tensor,
    *,
    strength: float,
    quadratic_weight: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Complete the square for separable quadratic Viterbi edge costs.

    The returned target has the same minimizer as

    ``a * (q - target)^2 + strength * gradient * q``.
    """

    if target.shape != gradient.shape:
        raise ValueError("target and gradient must have matching shapes")
    if not target.is_floating_point() or not gradient.is_floating_point():
        raise TypeError("target and gradient must be floating point")
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("gradient strength must be finite and nonnegative")
    weight = torch.as_tensor(
        quadratic_weight,
        device=target.device,
        dtype=torch.float32,
    )
    if weight.ndim and weight.shape != target.shape:
        raise ValueError("quadratic weights must be scalar or match the target")
    if not bool(torch.all(torch.isfinite(weight))) or not bool(torch.all(weight > 0)):
        raise ValueError("quadratic weights must be finite and positive")
    return target.float() - 0.5 * float(strength) * gradient.float() / weight


def shift_two_sided_target(
    target: torch.Tensor,
    gradient: torch.Tensor,
    row_hessian: torch.Tensor,
    column_hessian: torch.Tensor,
    *,
    strength: float,
) -> torch.Tensor:
    """Complete the square under ``tr(E.T H_row E H_column)``.

    ``target`` and ``gradient`` use matrix orientation ``[row, column]``.
    Linear solves are used instead of explicitly forming either inverse.
    """

    if target.ndim != 2 or gradient.shape != target.shape:
        raise ValueError("two-sided target and gradient must be matching matrices")
    rows, columns = target.shape
    if tuple(row_hessian.shape) != (rows, rows):
        raise ValueError("row Hessian has incompatible geometry")
    if tuple(column_hessian.shape) != (columns, columns):
        raise ValueError("column Hessian has incompatible geometry")
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("gradient strength must be finite and nonnegative")
    device = target.device
    dtype = torch.float64 if target.dtype == torch.float64 else torch.float32
    matrix = gradient.to(device=device, dtype=dtype)
    row = row_hessian.to(device=device, dtype=dtype)
    column = column_hessian.to(device=device, dtype=dtype)
    left_solved = torch.linalg.solve(row, matrix)
    correction = torch.linalg.solve(column.T, left_solved.T).T
    if not bool(torch.all(torch.isfinite(correction))):
        raise FloatingPointError("two-sided gradient target shift is non-finite")
    return target.to(dtype=dtype) - 0.5 * float(strength) * correction


def two_sided_guided_objective(
    target: torch.Tensor,
    candidate: torch.Tensor,
    anchor: torch.Tensor,
    gradient: torch.Tensor,
    row_hessian: torch.Tensor,
    column_hessian: torch.Tensor,
    *,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return quadratic, linear, and combined candidate objectives."""

    if not (target.shape == candidate.shape == anchor.shape == gradient.shape):
        raise ValueError("guided objective matrices must have matching shapes")
    error = candidate - target
    quadratic = torch.sum(error * (row_hessian @ error @ column_hessian))
    linear = float(strength) * torch.sum(gradient * (candidate - anchor))
    return quadratic, linear, quadratic + linear


__all__ = [
    "LowRankViterbiGradientGuidance",
    "ViterbiGradientGuidance",
    "shift_elementwise_target",
    "shift_two_sided_target",
    "two_sided_guided_objective",
]

"""Executable algebraic contracts for QSRT dense-H BlockLDLQ.

The production encoder is deliberately GPU-oriented.  This module contains a
small FP64 reference for the claims that must hold independently of CUDA:

* a weighted Gram matrix is exactly the linear-output SSE metric;
* diagonal damping has the stated scaled-identity interpretation;
* the block factorization reconstructs the damped covariance;
* the reverse BlockLDLQ feedback residual is ``L.T @ error``;
* an input permutation is a congruence transform of the covariance; and
* a decoded upstream gate/up pair defines the exact conditional down metric.

These helpers do not implement a second quantizer.  They are a proof oracle
against which the production encoder and captured artifacts are checked.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


_U64_MASK = (1 << 64) - 1
_CAPTURE_SPLIT_XOR = 0x6A09E667F3BCC909


def splitmix64(value: int) -> int:
    """Bit-exact scalar oracle for the capture sidecar's unsigned hash."""

    value = int(value) & _U64_MASK
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _U64_MASK
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _U64_MASK
    return (value ^ (value >> 31)) & _U64_MASK


def capture_sample_selected(observation: int, sample_rate: int) -> bool:
    """Return whether an input row should be retained by the GPU sampler."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    observation = int(observation) & _U64_MASK
    epoch = observation >> 32
    token = observation & 0xFFFFFFFF
    return splitmix64(epoch ^ token) % int(sample_rate) == 0


def capture_validation_split(observation: int, modulus: int) -> int:
    """Return the deterministic capture split label for an observation."""

    if modulus < 2:
        raise ValueError("validation modulus must be at least two")
    key = (int(observation) ^ _CAPTURE_SPLIT_XOR) & _U64_MASK
    return int(splitmix64(key) % int(modulus) == 0)


def _finite_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 2 or not value.numel():
        raise ValueError(f"{name} must be a nonempty matrix")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must be finite")
    return value


def _square(value: torch.Tensor, name: str) -> torch.Tensor:
    _finite_matrix(value, name)
    if value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be square")
    return value


def relative_error(actual: torch.Tensor | float, expected: torch.Tensor | float) -> float:
    """Return an FP64 relative error with a stable zero denominator."""

    actual_tensor = torch.as_tensor(actual, dtype=torch.float64)
    expected_tensor = torch.as_tensor(expected, dtype=torch.float64)
    numerator = torch.linalg.vector_norm(actual_tensor - expected_tensor)
    denominator = torch.linalg.vector_norm(expected_tensor)
    return float(numerator / max(float(denominator), torch.finfo(torch.float64).tiny))


def weighted_gram(rows: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute ``sum_i weights[i] * x_i x_i.T / sum_i weights[i]`` in FP64."""

    _finite_matrix(rows, "rows")
    if weights.ndim != 1 or weights.shape[0] != rows.shape[0]:
        raise ValueError("weights must align with rows")
    if not bool(torch.all(torch.isfinite(weights))) or bool(torch.any(weights < 0)):
        raise ValueError("weights must be finite and nonnegative")
    weights64 = weights.to(dtype=torch.float64)
    denominator = weights64.sum()
    if denominator <= 0:
        raise ValueError("weights must have positive mass")
    rows64 = rows.to(dtype=torch.float64)
    result = rows64.T @ (rows64 * weights64[:, None]) / denominator
    return ((result + result.T) * 0.5).contiguous()


def quadratic_error(error: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
    """Return ``trace(error @ H @ error.T)`` for ``error=[out, in]``."""

    _finite_matrix(error, "error")
    _square(hessian, "hessian")
    if error.shape[1] != hessian.shape[0]:
        raise ValueError("error input dimension does not match hessian")
    error64 = error.to(dtype=torch.float64)
    hessian64 = hessian.to(dtype=torch.float64)
    return torch.einsum("oi,ij,oj->", error64, hessian64, error64)


def explicit_weighted_output_error(
    rows: torch.Tensor,
    weights: torch.Tensor,
    error: torch.Tensor,
) -> torch.Tensor:
    """Return weighted mean ``||x_i @ error.T||^2`` in FP64."""

    _finite_matrix(rows, "rows")
    _finite_matrix(error, "error")
    if error.shape[1] != rows.shape[1]:
        raise ValueError("error input dimension does not match rows")
    if weights.ndim != 1 or weights.shape[0] != rows.shape[0]:
        raise ValueError("weights must align with rows")
    rows64 = rows.to(dtype=torch.float64)
    weights64 = weights.to(dtype=torch.float64)
    denominator = weights64.sum()
    if denominator <= 0:
        raise ValueError("weights must have positive mass")
    outputs = rows64 @ error.to(dtype=torch.float64).T
    return (weights64[:, None] * outputs.square()).sum() / denominator


def damp_hessian(hessian: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply the production diagonal damping rule in FP64."""

    _square(hessian, "hessian")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    result = hessian.to(dtype=torch.float64, copy=True)
    scale = torch.diagonal(result).mean()
    result.diagonal().add_(sigma * scale)
    return ((result + result.T) * 0.5).contiguous()


@dataclass(frozen=True)
class BlockLDLFactors:
    """FP64 reference factors satisfying ``H = L @ D @ L.T``."""

    lower: torch.Tensor
    block_diagonal: torch.Tensor
    block_size: int


def block_ldl_reference(hessian: torch.Tensor, block_size: int = 16) -> BlockLDLFactors:
    """Factor a positive-definite matrix like the production block routine."""

    _square(hessian, "hessian")
    if block_size <= 0 or hessian.shape[0] % block_size:
        raise ValueError("block size must divide the hessian dimension")
    hessian64 = hessian.to(dtype=torch.float64)
    cholesky = torch.linalg.cholesky(hessian64)
    dimension = int(hessian.shape[0])
    lower = cholesky.clone()
    block_diagonal = torch.zeros_like(hessian64)
    for begin in range(0, dimension, block_size):
        end = begin + block_size
        diagonal_cholesky = cholesky[begin:end, begin:end]
        # Right solve is more stable than explicitly materializing an inverse:
        # X @ C_ii = C[:, i]  =>  X = C[:, i] @ inv(C_ii).
        lower[:, begin:end] = torch.linalg.solve_triangular(
            diagonal_cholesky.T,
            cholesky[:, begin:end].T,
            upper=True,
        ).T
        lower[begin:end, begin:end] = torch.eye(
            block_size, dtype=torch.float64, device=hessian.device
        )
        block_diagonal[begin:end, begin:end] = (
            diagonal_cholesky @ diagonal_cholesky.T
        )
    return BlockLDLFactors(lower.contiguous(), block_diagonal.contiguous(), block_size)


def block_feedback_targets(
    weight: torch.Tensor,
    quantized: torch.Tensor,
    lower: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Recreate the target presented to each reverse BlockLDLQ block.

    ``weight`` and ``quantized`` use the encoder orientation ``[input, out]``.
    The quantized tensor may be arbitrary: the identity being checked is the
    feedback recursion itself, not the behavior of a particular codebook.
    """

    _finite_matrix(weight, "weight")
    _finite_matrix(quantized, "quantized")
    _square(lower, "lower")
    if quantized.shape != weight.shape or lower.shape[0] != weight.shape[0]:
        raise ValueError("feedback tensor shapes do not align")
    if block_size <= 0 or weight.shape[0] % block_size:
        raise ValueError("block size must divide the input dimension")
    weight64 = weight.to(dtype=torch.float64)
    quantized64 = quantized.to(dtype=torch.float64)
    lower64 = lower.to(dtype=torch.float64)
    error = weight64 - quantized64
    targets = torch.empty_like(weight64)
    for end in range(weight.shape[0], 0, -block_size):
        begin = end - block_size
        targets[begin:end] = weight64[begin:end]
        if end < weight.shape[0]:
            targets[begin:end].add_(lower64[end:, begin:end].T @ error[end:])
    return targets


def block_objective_from_feedback(
    error: torch.Tensor, factors: BlockLDLFactors
) -> torch.Tensor:
    """Evaluate the decomposed objective from ``Z = L.T @ error``."""

    _finite_matrix(error, "error")
    if error.shape[0] != factors.lower.shape[0]:
        raise ValueError("error input dimension does not match factors")
    transformed = factors.lower.T @ error.to(dtype=torch.float64)
    return torch.einsum(
        "io,ij,jo->",
        transformed,
        factors.block_diagonal,
        transformed,
    )


def permute_input_metric(
    error: torch.Tensor,
    hessian: torch.Tensor,
    permutation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return encoder-oriented error and covariance after an input permutation."""

    _finite_matrix(error, "error")
    _square(hessian, "hessian")
    if permutation.ndim != 1 or permutation.numel() != hessian.shape[0]:
        raise ValueError("permutation has the wrong shape")
    permutation = permutation.to(dtype=torch.long, device=hessian.device)
    if not torch.equal(
        torch.sort(permutation).values,
        torch.arange(hessian.shape[0], device=hessian.device),
    ):
        raise ValueError("permutation must contain every input index once")
    encoder_error = error.index_select(1, permutation).T.contiguous()
    encoder_hessian = hessian.index_select(0, permutation).index_select(
        1, permutation
    )
    return encoder_error, encoder_hessian


def congruence_metric(hessian: torch.Tensor, decode_input: torch.Tensor) -> torch.Tensor:
    """Transform ``H`` when canonical error is ``decode_input @ work_error``."""

    _square(hessian, "hessian")
    _square(decode_input, "decode_input")
    if hessian.shape != decode_input.shape:
        raise ValueError("decode transform and hessian dimensions differ")
    hessian64 = hessian.to(dtype=torch.float64)
    transform64 = decode_input.to(dtype=torch.float64)
    return (transform64.T @ hessian64 @ transform64).contiguous()


def situ_middle(
    inputs: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> torch.Tensor:
    """Replay Kimi's decoded gate/up pair into canonical post-SiTU rows."""

    _finite_matrix(inputs, "inputs")
    _finite_matrix(gate_weight, "gate_weight")
    _finite_matrix(up_weight, "up_weight")
    if gate_weight.shape != up_weight.shape or gate_weight.shape[1] != inputs.shape[1]:
        raise ValueError("gate/up shapes do not align with inputs")
    gate = F.linear(inputs, gate_weight)
    up = F.linear(inputs, up_weight)
    return (F.silu(gate) * up).contiguous()


def conditional_h2(
    inputs: torch.Tensor,
    gates: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return decoded upstream rows and their gate-square covariance."""

    if gates.ndim != 1 or gates.shape[0] != inputs.shape[0]:
        raise ValueError("gates must align with inputs")
    middle = situ_middle(inputs, gate_weight, up_weight)
    return middle, weighted_gram(middle, gates.square())


__all__ = [
    "BlockLDLFactors",
    "block_feedback_targets",
    "block_ldl_reference",
    "block_objective_from_feedback",
    "conditional_h2",
    "congruence_metric",
    "capture_sample_selected",
    "capture_validation_split",
    "damp_hessian",
    "explicit_weighted_output_error",
    "permute_input_metric",
    "quadratic_error",
    "relative_error",
    "situ_middle",
    "splitmix64",
    "weighted_gram",
]

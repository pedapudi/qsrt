"""High-rate SQG graphs and FP16 reconstruction laws.

This module is deliberately separate from the frozen K2/K3/K4 QSRT runtime
profile. ``sqg_fp16_d3l`` is the canonical uniform K5/K6 reconstruction
profile. It combines the primary carry-mixed SQG graph with a frozen 416-byte
FP16 dyadic-linear approximation to the shared Gaussian rank law. The other
graph allocations in this module remain numerical research controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import torch

from qsrt.sqg_e4m3 import sqg_xor_rank_permutation


HighRateAllocation = Literal["native_strata", "q8_phase"]
QuantilePosition = Literal["high", "low"]
_TRANSITIONS = 1 << 16
_D3L_SUBDIVISION_BITS = 3
SQG_FP16_D3L = "sqg_fp16_d3l"
SQG_FP16_D3L_DESCRIPTOR_BYTES = 416
SQG_FP16_D3L_DESCRIPTOR_SHA256 = (
    "17cf4ca9ef1e3a07c3354c12f7ac887b4e081b1668bea61eb37d8f2b410bb968"
)


@dataclass(frozen=True)
class FP16DyadicLinearDescriptors:
    """Base/slope descriptors for a dyadic approximation to an FP16 rank law."""

    subdivision_bits: int
    base: torch.Tensor
    slope: torch.Tensor

    @property
    def descriptor_count(self) -> int:
        return int(self.base.numel())

    @property
    def storage_bytes(self) -> int:
        return self.descriptor_count * 4


def _validate_bits(bits: int) -> None:
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in (5, 6):
        raise ValueError("experimental high-rate SQG supports only integer K5/K6")


def _reverse_low_bits(values: torch.Tensor, bits: int) -> torch.Tensor:
    result = torch.zeros_like(values)
    for index in range(bits):
        result |= ((values >> index) & 1) << (bits - 1 - index)
    return result


def _dyadic_descriptor_coordinates(
    magnitude_rank: torch.Tensor,
    *,
    subdivision_bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return descriptor indices and descending local coordinates.

    ``magnitude_rank`` is zero at the center and 32,767 at the outer tail.
    The construction allocates progressively smaller segments toward the tail,
    where the inverse-normal law is steepest.
    """

    if (
        isinstance(subdivision_bits, bool)
        or not isinstance(subdivision_bits, int)
        or subdivision_bits not in range(1, 9)
    ):
        raise ValueError("subdivision_bits must be an integer from 1 through 8")
    values = torch.as_tensor(magnitude_rank, dtype=torch.int64, device="cpu")
    if values.numel() and (
        int(values.min().item()) < 0 or int(values.max().item()) >= _TRANSITIONS // 2
    ):
        raise ValueError("magnitude ranks must be in [0, 32767]")
    odd_tail_coordinate = 65535 - 2 * values
    exponent = torch.floor(torch.log2(odd_tail_coordinate.to(torch.float64))).to(
        torch.int64
    )
    descriptor = torch.empty_like(exponent)
    local = torch.zeros_like(exponent)

    exact = exponent <= subdivision_bits
    exact_exponent = exponent[exact]
    exact_coordinate = odd_tail_coordinate[exact]
    descriptor[exact] = torch.where(
        exact_exponent == 0,
        torch.zeros_like(exact_exponent),
        (1 << (exact_exponent - 1))
        + ((exact_coordinate - (1 << exact_exponent) - 1) >> 1),
    )

    fitted_exponent = exponent[~exact]
    fitted_coordinate = odd_tail_coordinate[~exact]
    shift = fitted_exponent - subdivision_bits
    subdivision = (fitted_coordinate - (1 << fitted_exponent)) >> shift
    descriptor[~exact] = (shift << subdivision_bits) + subdivision
    maximum_odd = (
        (1 << fitted_exponent)
        + (subdivision << shift)
        + (1 << shift)
        - 1
    )
    local[~exact] = (maximum_odd - fitted_coordinate) >> 1
    return descriptor.contiguous(), local.contiguous()


def fit_fp16_dyadic_linear_rank_law(
    target: torch.Tensor,
    *,
    subdivision_bits: int = _D3L_SUBDIVISION_BITS,
) -> FP16DyadicLinearDescriptors:
    """Fit FP16 base/slope descriptors to a symmetric 65,536-rank law.

    Fitting is an offline operation. Reconstruction emulates one FP16 FMA per
    rank after the integer dyadic coordinate calculation used by the proposed
    decoder.
    """

    law = torch.as_tensor(target, device="cpu")
    if law.dtype != torch.float16 or tuple(law.shape) != (_TRANSITIONS,):
        raise ValueError("target must contain 65,536 FP16 rank values")
    if not bool(torch.isfinite(law).all()):
        raise ValueError("target rank law must contain only finite values")
    if not bool(torch.all(law[1:] >= law[:-1])):
        raise ValueError("target rank law must be monotone")

    magnitude = torch.arange(_TRANSITIONS // 2, dtype=torch.int64)
    descriptor, local = _dyadic_descriptor_coordinates(
        magnitude, subdivision_bits=subdivision_bits
    )
    descriptor_count = (16 - subdivision_bits) * (1 << subdivision_bits)
    positive = law[_TRANSITIONS // 2 :].to(torch.float32).numpy()
    descriptor_numpy = descriptor.numpy()
    local_numpy = local.numpy()
    base = np.empty(descriptor_count, dtype=np.float16)
    slope = np.empty(descriptor_count, dtype=np.float16)
    for index in range(descriptor_count):
        locations = np.flatnonzero(descriptor_numpy == index)
        x = local_numpy[locations].astype(np.float64)
        y = positive[locations].astype(np.float64)
        if locations.size == 1:
            intercept, gradient = float(y[0]), 0.0
        else:
            design = np.stack((np.ones_like(x), x), axis=1)
            intercept, gradient = np.linalg.lstsq(design, y, rcond=None)[0]
        base[index] = np.float16(intercept)
        slope[index] = np.float16(gradient)
    return FP16DyadicLinearDescriptors(
        subdivision_bits=subdivision_bits,
        base=torch.from_numpy(base.copy()),
        slope=torch.from_numpy(slope.copy()),
    )


def decode_fp16_dyadic_linear_rank_law(
    descriptors: FP16DyadicLinearDescriptors,
) -> torch.Tensor:
    """Expand dyadic descriptors using the proposed decoder's FP16 FMA law."""

    if (
        descriptors.base.dtype != torch.float16
        or descriptors.slope.dtype != torch.float16
        or descriptors.base.ndim != 1
        or descriptors.slope.ndim != 1
        or descriptors.base.shape != descriptors.slope.shape
    ):
        raise ValueError("dyadic descriptors must be paired one-dimensional FP16 tables")
    expected = (16 - descriptors.subdivision_bits) * (
        1 << descriptors.subdivision_bits
    )
    if descriptors.descriptor_count != expected:
        raise ValueError("dyadic descriptor count does not match subdivision_bits")
    ranks = torch.arange(_TRANSITIONS, dtype=torch.int64)
    negative = ranks < _TRANSITIONS // 2
    magnitude = torch.where(
        negative,
        _TRANSITIONS // 2 - 1 - ranks,
        ranks - _TRANSITIONS // 2,
    )
    descriptor, local = _dyadic_descriptor_coordinates(
        magnitude, subdivision_bits=descriptors.subdivision_bits
    )
    # CUDA __hfma performs the multiply/add at FP32 precision and rounds the
    # fused result once to FP16. Integer local coordinates are first rounded to
    # FP16, exactly as in the proposed decoder.
    local_fp32 = local.to(torch.float16).to(torch.float32)
    magnitude_values = (
        descriptors.slope.index_select(0, descriptor).to(torch.float32)
        * local_fp32
        + descriptors.base.index_select(0, descriptor).to(torch.float32)
    ).to(torch.float16)
    return torch.where(negative, -magnitude_values, magnitude_values).contiguous()


@lru_cache(maxsize=None)
def sqg_high_rate_rank_permutation(
    bits: int,
    allocation: HighRateAllocation,
    *,
    quantile_position: QuantilePosition = "high",
) -> torch.Tensor:
    """Map physical EXL states to quantile/phase ranks bijectively.

    EXL varies the high K predecessor bits within one Viterbi menu.  The
    ``native_strata`` control spends all K bits on ``2**K`` strata.  The
    ``q8_phase`` allocation spends three bits on eight strata and the other
    K-3 bits on H=4/H=8 phase alternatives at K5/K6.
    """

    _validate_bits(bits)
    if allocation not in ("native_strata", "q8_phase"):
        raise ValueError(f"unsupported high-rate allocation {allocation!r}")
    if quantile_position not in ("high", "low"):
        raise ValueError("quantile_position must be 'high' or 'low'")
    quantile_bits = bits if allocation == "native_strata" else 3
    phase_select_bits = bits - quantile_bits
    outgoing_width = 16 - bits
    outgoing_mask = (1 << outgoing_width) - 1
    states = torch.arange(_TRANSITIONS, dtype=torch.int64)
    predecessor = states >> outgoing_width
    outgoing_edge = states & outgoing_mask

    if phase_select_bits == 0:
        quantile = predecessor
        phase_select = torch.zeros_like(predecessor)
    elif quantile_position == "high":
        quantile = predecessor >> phase_select_bits
        phase_select = predecessor & ((1 << phase_select_bits) - 1)
    else:
        quantile = predecessor & ((1 << quantile_bits) - 1)
        phase_select = predecessor >> quantile_bits

    phase_width = 16 - quantile_bits
    phase_mask = (1 << phase_width) - 1
    phase_input = (phase_select << outgoing_width) | outgoing_edge
    mixed = phase_input ^ (phase_input >> 11)
    mixed ^= (mixed << 11) & phase_mask
    product = (0x3FA7D929 * mixed + 0xC928FD8E) & 0xFFFFFFFF
    phase = product & phase_mask
    syndrome = product >> (32 - quantile_bits)
    stratum = _reverse_low_bits(quantile, quantile_bits) ^ syndrome
    return ((stratum << phase_width) | phase).contiguous()


@lru_cache(maxsize=1)
def sqg_normal_rank_fp16() -> torch.Tensor:
    """Return a 65,536-rank normal law without finite-E4M3 rounding."""

    ranks = torch.arange(_TRANSITIONS, dtype=torch.float64)
    probability = (ranks + 0.5) / _TRANSITIONS
    result = (1.5 * torch.special.ndtri(probability)).to(torch.float16)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("high-rate SQG FP16 law contains a non-finite value")
    return result.contiguous()


@lru_cache(maxsize=1)
def sqg_normal_d3l_descriptors() -> FP16DyadicLinearDescriptors:
    """Return the canonical 104-descriptor D3L approximation."""

    return fit_fp16_dyadic_linear_rank_law(
        sqg_normal_rank_fp16(), subdivision_bits=_D3L_SUBDIVISION_BITS
    )


@lru_cache(maxsize=1)
def sqg_normal_d3l_descriptor_bytes() -> torch.Tensor:
    """Return the canonical interleaved ``(base, slope)`` FP16 payload."""

    descriptors = sqg_normal_d3l_descriptors()
    payload = torch.stack((descriptors.base, descriptors.slope), dim=1)
    result = payload.contiguous().view(torch.uint8)
    if result.numel() != SQG_FP16_D3L_DESCRIPTOR_BYTES:
        raise AssertionError("D3L descriptor payload has the wrong byte count")
    return result


@lru_cache(maxsize=1)
def sqg_normal_rank_fp16_d3l() -> torch.Tensor:
    """Return the normal rank law reconstructed from the 416-byte D3L table."""

    return decode_fp16_dyadic_linear_rank_law(sqg_normal_d3l_descriptors())


@lru_cache(maxsize=None)
def sqg_fp16_d3l_codebook(bits: int) -> torch.Tensor:
    """Return the canonical state-indexed ``sqg_fp16_d3l`` K5/K6 codebook."""

    _validate_bits(bits)
    ranks = sqg_xor_rank_permutation(bits)
    return sqg_normal_rank_fp16_d3l().index_select(0, ranks).contiguous()


@lru_cache(maxsize=None)
def sqg_high_rate_fp16_codebook(
    bits: int,
    allocation: HighRateAllocation,
    *,
    quantile_position: QuantilePosition = "high",
) -> torch.Tensor:
    """Return a state-indexed FP16 codebook for a K5/K6 research graph."""

    ranks = sqg_high_rate_rank_permutation(
        bits, allocation, quantile_position=quantile_position
    )
    return sqg_normal_rank_fp16().index_select(0, ranks).contiguous()


@lru_cache(maxsize=None)
def sqg_high_rate_fp16_d3l_codebook(
    bits: int,
    allocation: HighRateAllocation,
    *,
    quantile_position: QuantilePosition = "high",
) -> torch.Tensor:
    """Return a state-indexed K5/K6 codebook reconstructed from D3L."""

    ranks = sqg_high_rate_rank_permutation(
        bits, allocation, quantile_position=quantile_position
    )
    return sqg_normal_rank_fp16_d3l().index_select(0, ranks).contiguous()

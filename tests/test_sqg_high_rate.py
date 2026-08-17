from __future__ import annotations

import pytest
import torch

from qsrt.sqg_high_rate import (
    SQG_FP16_D3L_DESCRIPTOR_BYTES,
    SQG_FP16_D3L_DESCRIPTOR_SHA256,
    decode_fp16_dyadic_linear_rank_law,
    fit_fp16_dyadic_linear_rank_law,
    sqg_fp16_d3l_codebook,
    sqg_high_rate_fp16_codebook,
    sqg_high_rate_fp16_d3l_codebook,
    sqg_high_rate_rank_permutation,
    sqg_normal_d3l_descriptor_bytes,
    sqg_normal_d3l_descriptors,
    sqg_normal_rank_fp16,
    sqg_normal_rank_fp16_d3l,
)
from qsrt.qsrt_codec_pilot import tensor_sha256
from qsrt.sqg_e4m3 import sqg_xor_rank_permutation


@pytest.mark.parametrize("bits", (5, 6))
@pytest.mark.parametrize("allocation", ("native_strata", "q8_phase"))
def test_high_rate_rank_maps_are_bijective_and_match_actual_exl_menus(
    bits: int, allocation: str
) -> None:
    ranks = sqg_high_rate_rank_permutation(bits, allocation)
    assert torch.equal(torch.sort(ranks).values, torch.arange(1 << 16))
    quantile_bits = bits if allocation == "native_strata" else 3
    phase_choices = 1 << (bits - quantile_bits)
    outgoing_width = 16 - bits
    strata = (ranks >> (16 - quantile_bits)).reshape(
        1 << bits, 1 << outgoing_width
    ).T
    counts = torch.stack(
        [(strata == stratum).sum(dim=1) for stratum in range(1 << quantile_bits)]
    )
    assert torch.equal(counts, torch.full_like(counts, phase_choices))


def test_high_rate_normal_fp16_law_is_finite_monotone_and_richer_than_e4m3() -> None:
    law = sqg_normal_rank_fp16()
    assert law.dtype == torch.float16
    assert law.shape == (1 << 16,)
    assert bool(torch.isfinite(law).all())
    assert bool(torch.all(law[1:] >= law[:-1]))
    assert torch.unique(law).numel() > 10_000


@pytest.mark.parametrize("bits", (5, 6))
@pytest.mark.parametrize("allocation", ("native_strata", "q8_phase"))
def test_high_rate_fp16_codebook_uses_the_shared_rank_law(
    bits: int, allocation: str
) -> None:
    expected = sqg_normal_rank_fp16().index_select(
        0, sqg_high_rate_rank_permutation(bits, allocation)
    )
    assert torch.equal(sqg_high_rate_fp16_codebook(bits, allocation), expected)


def test_d3l_is_a_416_byte_tail_adaptive_approximation() -> None:
    descriptors = sqg_normal_d3l_descriptors()
    assert descriptors.subdivision_bits == 3
    assert descriptors.descriptor_count == 104
    assert descriptors.storage_bytes == 416
    assert descriptors.base.dtype == torch.float16
    assert descriptors.slope.dtype == torch.float16

    target = sqg_normal_rank_fp16()
    reconstructed = sqg_normal_rank_fp16_d3l()
    assert torch.equal(
        reconstructed,
        decode_fp16_dyadic_linear_rank_law(descriptors),
    )
    error = reconstructed.float() - target.float()
    assert float(torch.sqrt(torch.mean(error.square()))) < 7e-4
    assert float(error.abs().max()) <= 4e-3
    assert bool(torch.isfinite(reconstructed).all())
    assert torch.equal(reconstructed, -reconstructed.flip(0))
    steps = reconstructed[1:].float() - reconstructed[:-1].float()
    assert int(torch.count_nonzero(steps < 0)) == 10
    assert float(steps.min()) >= -2e-3


def test_d3l_descriptor_payload_has_frozen_identity() -> None:
    payload = sqg_normal_d3l_descriptor_bytes()
    assert payload.dtype == torch.uint8
    assert payload.numel() == SQG_FP16_D3L_DESCRIPTOR_BYTES
    assert tensor_sha256(payload) == SQG_FP16_D3L_DESCRIPTOR_SHA256


@pytest.mark.parametrize("bits", (5, 6))
def test_official_d3l_codebook_uses_primary_carry_mixed_graph(bits: int) -> None:
    expected = sqg_normal_rank_fp16_d3l().index_select(
        0, sqg_xor_rank_permutation(bits)
    )
    assert torch.equal(sqg_fp16_d3l_codebook(bits), expected)


@pytest.mark.parametrize("bits", (5, 6))
def test_high_rate_d3l_codebook_uses_the_shared_rank_map(bits: int) -> None:
    expected = sqg_normal_rank_fp16_d3l().index_select(
        0, sqg_high_rate_rank_permutation(bits, "native_strata")
    )
    assert torch.equal(
        sqg_high_rate_fp16_d3l_codebook(bits, "native_strata"), expected
    )


def test_dyadic_fitter_rejects_non_monotone_rank_laws() -> None:
    target = sqg_normal_rank_fp16().clone()
    target[100], target[101] = target[101].clone(), target[100].clone()
    target[100] = target[101] + torch.tensor(1.0, dtype=torch.float16)
    with pytest.raises(ValueError, match="monotone"):
        fit_fp16_dyadic_linear_rank_law(target)


def test_high_rate_controls_reject_production_rates() -> None:
    with pytest.raises(ValueError, match="K5/K6"):
        sqg_high_rate_rank_permutation(4, "q8_phase")

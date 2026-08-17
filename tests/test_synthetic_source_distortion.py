"""Tests for the per-symbol synthetic-source distortion harness.

The harness's value rests on two transcriptions being exact (ExLlamaV3's
computed codebooks and the shared trellis convention) and on the Viterbi
recursion being a true exact search.  Each test below pins one of those
anchors to an independent source of truth: the encoder's own hard-coded
codebook constant, the CPU reference trellis reconstruction, closed-form
scalar quantization, and the frozen SQG tables.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qsrt import exl3_reference
from qsrt import sqg_e4m3
from qsrt import synthetic_source_distortion as ssd


def test_codebook_transcriptions_match_encoder_constants() -> None:
    # The encoder hard-codes codebook 0's standard deviation; an exact
    # transcription must reproduce it.  MCG's distinct-value count is the
    # matching fingerprint for the production comparison codebook.
    cb0 = ssd.exl3_cb0_values()
    assert cb0.std() == pytest.approx(ssd.EXL3_CB0_SIGMA, rel=1e-7)
    mcg = ssd.exl3_mcg_values()
    assert len(np.unique(mcg)) == ssd.EXL3_MCG_DISTINCT_VALUES
    assert np.isfinite(mcg).all()
    assert mcg.std() == pytest.approx(ssd.EXL3_CB0_SIGMA, rel=1e-3)


def test_mul1_and_cb0_tables_are_finite_and_centered() -> None:
    for values in (ssd.exl3_mul1_values(), ssd.exl3_cb0_values()):
        assert values.shape == (ssd.N_CODEWORDS,)
        assert np.isfinite(values).all()
        assert abs(values.mean()) < 0.05
        assert 0.5 < values.std() < 2.0


@pytest.mark.parametrize("bits", (2, 3, 4))
def test_predecessor_index_matches_reference_trellis(bits: int) -> None:
    # Walk a random edge stream through the CPU reference reconstruction and
    # check that consecutive 16-bit windows are linked by the same
    # predecessor map the Viterbi uses.
    generator = torch.Generator().manual_seed(7)
    edges = torch.randint(0, 1 << bits, (1, 256), generator=generator)
    windows = exl3_reference.reconstruct_trellis_states(edges, bits)[0].to(
        torch.int64
    ) & 0xFFFF
    pred = ssd.predecessor_index(bits)
    n_states = 1 << (16 - bits)
    # The reference tile is cyclic, so only interior transitions (where the
    # full 16-bit window has been populated) are checked.
    for i in range(16, 255):
        assert pred[windows[i + 1]] == windows[i] & (n_states - 1)


def test_degenerate_trellis_collapses_to_scalar_quantization() -> None:
    # A table that depends only on the menu-free bits gives the Viterbi an
    # unconstrained scalar choice at every step, so the exact search must
    # reproduce nearest-neighbour scalar quantization on the scored window.
    bits = 2
    levels = np.asarray(ssd.LLOYD_MAX_2BIT_LEVELS)
    free = np.arange(ssd.N_CODEWORDS) >> (16 - bits)
    table = torch.from_numpy(levels[free]).float()
    source = ssd.gaussian_sequences(2, 64, seed=3)
    window = (16, 48)
    scored = ssd.viterbi_windowed_sse(source, table, bits, window=window)
    span = window[1] - window[0]
    expected = ssd.scalar_nearest_mse(levels, source[:, window[0] : window[1]])
    assert float(scored.sum()) / (2 * span) == pytest.approx(expected, rel=1e-5)


def test_lloyd_max_reference_mse() -> None:
    samples = ssd.gaussian_sequences(1, 200_000, seed=11)
    mse = ssd.scalar_nearest_mse(np.asarray(ssd.LLOYD_MAX_2BIT_LEVELS), samples)
    assert mse == pytest.approx(ssd.LLOYD_MAX_2BIT_MSE, abs=0.002)


def test_trellis_beats_scalar_at_k2() -> None:
    source = ssd.gaussian_sequences(4, 128, seed=5)
    window = (32, 96)
    span = window[1] - window[0]
    table = torch.from_numpy(ssd.reconstruction_table("sqg_t12_e4m3", 2)).float()
    scale = 0.6624  # fitted value at the production operating point
    scored = ssd.viterbi_windowed_sse(source / scale, table, 2, window=window)
    trellis_mse = float(scored.mean()) * scale**2 / span
    scalar_mse = ssd.scalar_nearest_mse(
        np.asarray(ssd.LLOYD_MAX_2BIT_LEVELS), source[:, window[0] : window[1]]
    )
    # The 14-bit-history trellis must show a clear shaping gain over the
    # optimal memoryless scalar quantizer at the same rate.
    assert trellis_mse < 0.85 * scalar_mse


def test_production_sqg_table_matches_frozen_source() -> None:
    values = ssd.reconstruction_table("sqg_t12_e4m3", 2)
    frozen = (
        sqg_e4m3.sqg_xor_cheb_t12_bytes(2)
        .view(torch.float8_e4m3fn)
        .float()
        .numpy()
        .astype(np.float64)
    )
    assert np.array_equal(values, frozen)
    assert len(np.unique(values)) == 151
    assert 0.0 in values


@pytest.mark.parametrize(
    ("bits", "expected"),
    ((2, 2.0437), (3, 7.9062)),
)
def test_frozen_graph_menu_stratum_diversity(bits: int, expected: float) -> None:
    # The frozen rank construction derives strata from branch bits, which a
    # Viterbi menu pins, so menus expose far fewer distinct strata than the
    # nominal 2**K at K2.  These constants are deterministic properties of
    # the frozen graph; a change here means the graph changed.
    stats = ssd.menu_statistics("sqg_exact_e4m3", bits)
    assert stats["mean_distinct_strata_per_menu"] == pytest.approx(
        expected, abs=5e-3
    )
    assert stats["nominal_strata"] == 1 << bits


@pytest.mark.parametrize("bits", (2, 3))
def test_menu_oriented_rank_is_a_bijection_with_full_menus(bits: int) -> None:
    ranks = ssd.menu_oriented_rank(bits)
    assert len(np.unique(ranks)) == ssd.N_CODEWORDS
    stats = ssd.menu_statistics("sqg_menu_oriented_e4m3", bits)
    assert stats["mean_distinct_strata_per_menu"] == pytest.approx(1 << bits)


def test_measure_code_smoke() -> None:
    result = ssd.measure_code(
        "sqg_t12_e4m3", 2, sequences=2, steps=96, window=(32, 64)
    )
    assert result.bits == 2
    assert 0.0 < result.mse < ssd.LLOYD_MAX_2BIT_MSE
    assert result.measured_symbols == 2 * 32
    payload = result.as_dict()
    assert payload["gaussian_rd_bound"] == pytest.approx(0.0625)
    assert len(payload["per_sequence_mse"]) == 2

from __future__ import annotations

import pytest
import torch

from qsrt.qsrt_codec_pilot import (
    CODEBOOK_MCG,
    UNIFORM_CODEBOOKS,
    FixedAverageRateGeometry,
    encode_uniform_candidate,
    permute_shared_axis,
    pack_uniform_trellis_edges,
    restore_shared_axis,
    replay_uniform_candidate,
    select_coupled_modes,
    shared_axis_weight_energy_order,
    summarize_mode_selections,
    unpack_uniform_trellis_states,
)
from qsrt.qsrt import pack_trellis_edges
from qsrt.exl3_reference import reconstruct_trellis_states
from qsrt.exl3_reference import decode_exl3_weight


def test_uniform_candidate_rejects_an_invalid_feedback_multiplier_before_cuda() -> None:
    with pytest.raises(ValueError, match="ldlq_feedback_multiplier"):
        encode_uniform_candidate(
            torch.zeros((16, 128)),
            bits=3,
            codebook=CODEBOOK_MCG,
            device=torch.device("cpu"),
            quantizer_module=object(),
            input_sign_seed=0,
            output_sign_seed=0,
            ldlq_feedback_multiplier=1.5,
        )


def test_uniform_candidate_rejects_conflicting_output_feedback_controls() -> None:
    source = torch.zeros((16, 128))
    with pytest.raises(ValueError, match="cannot be combined"):
        encode_uniform_candidate(
            source,
            bits=3,
            codebook=CODEBOOK_MCG,
            device=torch.device("cpu"),
            quantizer_module=object(),
            input_sign_seed=0,
            output_sign_seed=0,
            output_hessian=torch.eye(16),
            use_two_sided_traversal_without_output_feedback=True,
        )
    with pytest.raises(TypeError, match="must be a boolean"):
        encode_uniform_candidate(
            source,
            bits=3,
            codebook=CODEBOOK_MCG,
            device=torch.device("cpu"),
            quantizer_module=object(),
            input_sign_seed=0,
            output_sign_seed=0,
            use_two_sided_traversal_without_output_feedback=1,  # type: ignore[arg-type]
        )


def test_fixed_average_geometry_is_shape_generic_and_exact_3bpw() -> None:
    glm = FixedAverageRateGeometry(axis_channels=2048)
    assert glm.record_count == 16
    assert glm.tiles_per_record == 8
    assert glm.record_bits(0) == (3,) * 16
    assert glm.record_bits(1) == (2,) + (3,) * 14 + (4,)
    assert glm.record_bits(2) == (2, 2) + (3,) * 12 + (4, 4)
    for mode in glm.mode_ids:
        assert len(glm.tile_bits(mode)) == 128
        assert sum(glm.tile_bits(mode)) / len(glm.tile_bits(mode)) == 3
    assert glm.logical_trellis_bytes((2048, 6144)) == 4_718_592

    other = FixedAverageRateGeometry(
        axis_channels=1024,
        record_channels=64,
        mode_ids=(0, 1, 2, 3),
    )
    assert other.record_count == 16
    assert sum(other.record_bits(3)) == 48


def test_fixed_average_geometry_rejects_model_or_rate_mismatch() -> None:
    with pytest.raises(ValueError, match="whole coding records"):
        FixedAverageRateGeometry(axis_channels=2050)
    with pytest.raises(ValueError, match="too many"):
        FixedAverageRateGeometry(axis_channels=256, mode_ids=(0, 2))
    with pytest.raises(ValueError, match="K2/K3/K4"):
        FixedAverageRateGeometry(axis_channels=2048, donor_bits=1)


def test_shared_energy_order_couples_row_and_column_roles() -> None:
    gate = torch.zeros((8, 3), dtype=torch.float32)
    up = torch.zeros_like(gate)
    down = torch.zeros((3, 8), dtype=torch.float32)
    # Four-channel group 0 has combined energy 4; group 1 has energy 12.
    gate[:4] = 1.0
    up[4:] = 1.0
    down[:, 4:] = 1.0
    permutation, scores = shared_axis_weight_energy_order(
        [(gate, 0), (up, 0), (down, 1)], group_channels=4
    )
    assert scores.tolist() == [12.0, 24.0]
    assert permutation.tolist() == list(range(8))

    gate[:4] = 3.0
    permutation, scores = shared_axis_weight_energy_order(
        [(gate, 0), (up, 0), (down, 1)], group_channels=4
    )
    assert scores[0] > scores[1]
    assert permutation.tolist() == [4, 5, 6, 7, 0, 1, 2, 3]


@pytest.mark.parametrize("axis", (0, 1))
def test_shared_axis_permutation_round_trips(axis: int) -> None:
    source = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    length = source.shape[axis]
    permutation = torch.arange(length - 1, -1, -1)
    ordered = permute_shared_axis(source, permutation, axis=axis)
    restored = restore_shared_axis(ordered, permutation, axis=axis)
    torch.testing.assert_close(restored, source, rtol=0, atol=0)


def test_coupled_mode_selection_partitions_matrices_and_ties_to_lower_rate_shift() -> None:
    matrix_sse = {
        "gate": {0: 10.0, 1: 8.0, 2: 9.0},
        "up": {0: 5.0, 1: 8.0, 2: 6.0},
        "down": {0: 7.0, 1: 6.0, 2: 6.0},
    }
    selected = select_coupled_modes(
        matrix_sse, {"r13": ("gate", "up"), "r2": ("down",)}
    )
    # gate+up ties R0 and R2 at 15, so the deterministic lower mode wins.
    assert selected["r13"]["selected_mode"] == 0
    assert selected["r2"]["selected_mode"] == 1

    with pytest.raises(ValueError, match="partition"):
        select_coupled_modes(matrix_sse, {"r13": ("gate", "up")})


def test_mode_selection_summary_counts_any_r1_plus() -> None:
    records = [
        {
            "rate_selection": {
                "r13": {"selected_mode": 0},
                "r2": {"selected_mode": 0},
            }
        },
        {
            "rate_selection": {
                "r13": {"selected_mode": 1},
                "r2": {"selected_mode": 0},
            }
        },
        {
            "rate_selection": {
                "r13": {"selected_mode": 2},
                "r2": {"selected_mode": 1},
            }
        },
    ]
    summary = summarize_mode_selections(records, ("r13", "r2"))
    assert summary["any_family_r1_plus"] == 2
    assert summary["all_families_r1_plus"] == 1
    assert summary["all_r0"] == 1
    assert summary["family_histograms"] == {
        "r13": {"R0": 1, "R1": 1, "R2": 1},
        "r2": {"R0": 2, "R1": 1},
    }


@pytest.mark.parametrize("bits", (2, 3, 4, 5, 6))
def test_generic_uniform_unpack_closes_native_exl_words(bits: int) -> None:
    generator = torch.Generator().manual_seed(100 + bits)
    edges = torch.randint(
        0, 1 << bits, (2, 3, 256), dtype=torch.int16, generator=generator
    )
    packed = pack_uniform_trellis_edges(edges, bits)
    actual = unpack_uniform_trellis_states(packed, bits)
    expected = reconstruct_trellis_states(edges, bits)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert CODEBOOK_MCG == "mcg"


@pytest.mark.parametrize("bits", (2, 3, 4))
def test_experimental_uniform_pack_matches_frozen_qsrt_rates(bits: int) -> None:
    edges = torch.arange(8 * 8 * 256, dtype=torch.int16).reshape(8, 8, 256)
    assert torch.equal(
        pack_uniform_trellis_edges(edges, bits), pack_trellis_edges(edges, bits)
    )


def test_uniform_pack_rejects_rates_outside_experimental_range() -> None:
    edges = torch.zeros((1, 1, 256), dtype=torch.int16)
    with pytest.raises(ValueError, match="K2 through K6"):
        pack_uniform_trellis_edges(edges, 7)
    with pytest.raises(ValueError, match="K2 through K6"):
        unpack_uniform_trellis_states(
            torch.zeros((1, 1, 16), dtype=torch.int16), True
        )


def test_uniform_control_codebooks_include_exact_offline_control() -> None:
    assert "sqg-cheb-normal-e4m3" in UNIFORM_CODEBOOKS


def test_uniform_candidate_replay_replaces_only_reconstruction_labels() -> None:
    bits = 3
    edges = torch.arange(8 * 8 * 256, dtype=torch.int16).reshape(8, 8, 256)
    trellis = pack_uniform_trellis_edges(edges, bits)
    suh = torch.linspace(0.5, 1.0, 128, dtype=torch.float16)
    svh = torch.linspace(1.0, 1.5, 128, dtype=torch.float16)
    codebook = torch.linspace(-2.0, 2.0, 65536, dtype=torch.float16)
    replay = {
        "bits": bits,
        "trellis": trellis,
        "suh": suh,
        "svh": svh,
    }

    actual = replay_uniform_candidate(replay, codebook_values=codebook)
    states = unpack_uniform_trellis_states(trellis, bits)
    expected = decode_exl3_weight(
        states, suh, svh, codebook_values=codebook
    ).half().T.contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_uniform_candidate_replay_rejects_malformed_inputs() -> None:
    good = {
        "bits": 3,
        "trellis": torch.zeros((1, 1, 48), dtype=torch.int16),
        "suh": torch.ones(16, dtype=torch.float16),
        "svh": torch.ones(16, dtype=torch.float16),
    }
    codebook = torch.zeros(65536, dtype=torch.float16)
    with pytest.raises(ValueError, match="K2 through K6"):
        replay_uniform_candidate({**good, "bits": 7}, codebook_values=codebook)
    with pytest.raises(TypeError, match="trellis, suh, and svh"):
        replay_uniform_candidate({**good, "svh": None}, codebook_values=codebook)
    with pytest.raises(ValueError, match="65,536 finite FP16"):
        replay_uniform_candidate(good, codebook_values=codebook.float())

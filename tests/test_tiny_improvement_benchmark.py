from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from qsrt.tiny_improvement_benchmark import (
    BIT_LIMIT,
    BITS_AT_PLAY,
    PAYLOAD_BITS,
    PSEUDO_VOCABULARY_READOUT,
    SQG_XOR_CHEB_T12_SHA256,
    _forward_kld,
    _sqg_edge_rank,
    _sqg_edge_rank_for_rate,
    closed_paths,
    choose_path,
    make_normalized_pair_table,
    make_problem,
    refit_down_target,
    run_benchmark,
    run_sweep,
    sqg_xor_cheb_t12_bytes,
)


def test_benchmark_has_a_multi_neuron_expert_under_the_bit_limit() -> None:
    report = run_benchmark()

    assert PAYLOAD_BITS == 16
    assert BITS_AT_PLAY == 30
    assert BITS_AT_PLAY <= BIT_LIMIT == 32
    assert report["bit_budget"] == {
        "source_weight_count": 8,
        "gate_up_stream_bits": 8,
        "down_stream_bits": 8,
        "payload_bits": 16,
        "payload_bits_per_weight": 2.0,
        "pair_working_state_bits": 12,
        "scalar_k2_working_state_bits": 14,
        "pair_bits_at_play": 28,
        "scalar_k2_bits_at_play": 30,
        "bits_at_play": 30,
        "limit": 32,
    }
    assert report["problem"]["input_dimensions"] == 1
    assert report["problem"]["intermediate_neurons"] == 2
    assert report["problem"]["output_dimensions"] == 2
    assert report["problem"]["fit_rows"] == 48
    assert report["problem"]["heldout_rows"] == 48
    assert report["problem"]["fit_and_heldout_rows_disjoint"] is True
    assert "4,096 finite-E4M3 pair entries selected by rank >> 4" in report[
        "problem"
    ]["trellis_graph"]["pair_table"]
    assert len(report["problem"]["upstream_scales"]) == 2
    assert report["problem"]["upstream_scales"][0] != report["problem"]["upstream_scales"][1]
    assert report["problem"]["comparator"]["name"] == "matched_payload_scalar_K2_control"
    assert report["problem"]["comparator"]["not_exl3"] is True
    assert report["problem"]["comparator"]["scalar_table"] == (
        "exact frozen 4,096-entry sqg_xor_cheb_t12 rank table"
    )
    assert "cannot establish EXL3 dominance" in report["kld_evidence_boundary"]
    assert "cannot establish smaller total model size" in report["size_evidence_boundary"]
    assert report["problem"]["blockldlq"].startswith("omitted")
    assert len(report["problem"]["down_scales"]) == 2
    assert len(report["problem"]["source_down_k2_paths"]) == 2
    assert all(
        0 <= index < 4096
        for path in report["problem"]["source_down_k2_paths"]
        for index in path["table_indices"]
    )
    assert len(report["stages"]["refit_down_target"]["down_target_scales"]) == 2
    assert len(report["stages"]["refit_down_target"]["down_candidate_k2_paths"]) == 2


def test_embedded_t12_bytes_match_the_production_table() -> None:
    import hashlib

    from qsrt.sqg_e4m3 import sqg_xor_cheb_t12_rank_lut_bytes

    actual = sqg_xor_cheb_t12_bytes()
    expected = sqg_xor_cheb_t12_rank_lut_bytes().cpu().numpy()
    np.testing.assert_array_equal(actual, expected)
    assert hashlib.sha256(actual.tobytes()).hexdigest() == SQG_XOR_CHEB_T12_SHA256


def test_pair_table_retains_phase_entries_in_every_joint_quartile() -> None:
    table = make_normalized_pair_table()
    path_indices = [indices for _branches, _states, indices in closed_paths()]
    reconstruction_sequences = {
        tuple(table[np.asarray(indices)].reshape(-1)) for indices in path_indices
    }

    assert table.shape == (4096, 2)
    assert np.isfinite(table).all()
    assert all(
        np.unique(table[stratum * 256 : (stratum + 1) * 256], axis=0).shape[0]
        > 1
        for stratum in range(16)
    )
    assert len(reconstruction_sequences) == 256


def test_pseudo_vocabulary_readout_is_centered_full_rank_and_sees_both_directions() -> None:
    assert PSEUDO_VOCABULARY_READOUT.shape[0] >= 4
    np.testing.assert_allclose(PSEUDO_VOCABULARY_READOUT.sum(axis=0), 0.0)
    assert np.linalg.matrix_rank(PSEUDO_VOCABULARY_READOUT) == 2
    teacher = np.zeros((4, 2), dtype=np.float64)
    assert _forward_kld(teacher, teacher + np.array((1.0, 0.0))) > 0.0
    assert _forward_kld(teacher, teacher + np.array((0.0, 1.0))) > 0.0


def test_fit_and_heldout_rows_are_distinct_arrays() -> None:
    problem = make_problem()

    assert problem.fit_inputs.shape == problem.heldout_inputs.shape == (48, 1)
    assert not np.array_equal(problem.fit_inputs, problem.heldout_inputs)
    assert not np.shares_memory(problem.fit_inputs, problem.heldout_inputs)


def test_pair_trellis_has_two_hundred_fifty_six_closed_paths() -> None:
    paths = list(closed_paths())

    assert len(paths) == 256
    assert all(len(branches) == 2 for branches, _states, _indices in paths)
    assert all(len(states) == 2 for _branches, states, _indices in paths)
    assert all(0 <= state < 4096 for _branches, states, _indices in paths for state in states)
    assert len({branches for branches, _states, _indices in paths}) == 256
    assert len({indices for _branches, _states, indices in paths}) == 256


@pytest.mark.parametrize("bits", [2, 4])
def test_all_carry_mixed_ranks_match_production(bits: int) -> None:
    from qsrt.sqg_e4m3 import sqg_xor_rank_permutation

    expected = sqg_xor_rank_permutation(bits).cpu().numpy()
    width = 16 - bits
    actual = np.fromiter(
        (_sqg_edge_rank_for_rate(codeword >> bits, codeword & ((1 << bits) - 1), bits)
         for codeword in range(65536)), dtype=np.int64, count=65536
    )
    np.testing.assert_array_equal(actual, expected)
    if bits == 4:
        assert _sqg_edge_rank(0xABC, 9) == actual[(0xABC << 4) | 9]


def test_selection_and_refit_use_fit_rows_but_report_heldout_rows() -> None:
    report = run_benchmark()
    stages = report["stages"]
    trained = stages["trained_pair_table"]
    output_aware = stages["output_aware_path"]
    refit = stages["refit_down_target"]

    assert output_aware["selection_split"] == "fit_rows_only"
    assert output_aware["fit_forward_kld"] <= trained["fit_forward_kld"]
    assert output_aware["heldout_status"] in {"improved", "regressed", "unchanged"}
    assert output_aware["heldout_forward_kld_status"] in {
        "improved", "regressed", "unchanged"
    }
    assert refit["selection_split"] == "fit_rows_only"
    assert np.asarray(refit["continuous_down_target"]).shape == (2, 2)
    assert np.asarray(refit["quantized_down_candidate"]).shape == (2, 2)
    assert refit["candidate_direction_changed"] is True
    if refit["down_target_accepted"]:
        assert refit["fit_forward_kld"] < output_aware["fit_forward_kld"]
    else:
        assert refit["fit_forward_kld"] == output_aware["fit_forward_kld"]
    assert isinstance(refit["candidate_heldout_output_mean_squared_error"], float)
    assert all(
        stage["fit_forward_kld"] >= 0.0
        and stage["heldout_forward_kld"] >= 0.0
        for stage in stages.values()
    )


def test_heldout_mutation_cannot_change_fit_selected_path_or_refit() -> None:
    problem = make_problem()
    mutated = replace(
        problem,
        heldout_inputs=problem.heldout_inputs * -37.0 + 11.0,
        heldout_source_output=problem.heldout_source_output * 19.0 - 5.0,
    )
    selected = choose_path(problem, problem.trained_pair_table, objective="fit_forward_kld")
    mutated_selected = choose_path(
        mutated, mutated.trained_pair_table, objective="fit_forward_kld"
    )
    assert selected.branches == mutated_selected.branches
    refit = refit_down_target(problem, selected)
    mutated_refit = refit_down_target(mutated, mutated_selected)
    assert refit.accepted_on_fit == mutated_refit.accepted_on_fit
    np.testing.assert_array_equal(refit.quantized_candidate, mutated_refit.quantized_candidate)
    np.testing.assert_array_equal(refit.selected_down, mutated_refit.selected_down)


def test_report_exposes_actual_reconstruction_path_differences() -> None:
    report = run_benchmark()
    paths = [
        report["stages"][stage]["path"]["branches"]
        for stage in ("scalar_k2_control", "trained_pair_table", "output_aware_path")
    ]
    comparison = report["path_comparison"]

    assert comparison["unique_reconstruction_paths"] == len({tuple(path) for path in paths})
    assert comparison["unique_reconstruction_paths"] > 1
    assert comparison["pairwise_branch_hamming_distance"] == {
        "scalar_k2_control_vs_trained_pair_table": sum(a != b for a, b in zip(paths[0], paths[1])),
        "scalar_k2_control_vs_output_aware_path": sum(a != b for a, b in zip(paths[0], paths[2])),
        "trained_pair_table_vs_output_aware_path": sum(a != b for a, b in zip(paths[1], paths[2])),
    }


@pytest.mark.parametrize("expert_count", [4, 8])
def test_shared_table_sweep_is_bounded_and_surfaces_regressions(expert_count: int) -> None:
    started = time.perf_counter()
    report = run_sweep(expert_count)

    assert time.perf_counter() - started < 60.0
    assert report["expert_count"] == expert_count
    assert report["shared_pair_table"] is True
    assert report["sequential_execution"] is True
    assert report["peak_bits_at_play"] == 30
    assert report["aggregate_payload_bits"] == expert_count * PAYLOAD_BITS
    assert report["experts_with_multiple_reconstruction_paths"] > 0
    assert report["experts_with_output_aware_path_change"] > 0
    assert report["down_refit_direction_changes"] > 0
    for split in (
        "fit_transitions", "transitions", "bf16_transitions",
        "fit_forward_kld_transitions", "heldout_forward_kld_transitions",
    ):
        for summary in report[split].values():
            assert sum(
                summary[f"{status}_experts"]
                for status in ("improved", "unchanged", "regressed")
            ) == expert_count
    if expert_count == 8:
        assert any(
            summary["regressed_experts"] > 0
            for summary in report["heldout_forward_kld_transitions"].values()
        )


def test_benchmark_is_deterministic() -> None:
    assert run_benchmark() == run_benchmark()


def test_mismatched_pair_correlation_surfaces_regressions() -> None:
    report = run_sweep(4, pair_table_correlation=-0.7)

    assert report["pair_table_calibration_correlation"] == -0.7
    for transition in report["heldout_forward_kld_transitions"].values():
        assert sum(transition[f"{status}_experts"] for status in (
            "improved", "unchanged", "regressed"
        )) == 4
    assert any(
        transition["regressed_experts"] > 0
        for transition in report["heldout_forward_kld_transitions"].values()
    )


def test_command_defaults_to_eight_experts() -> None:
    script = Path(__file__).parents[1] / "scripts" / "benchmark_qsrt_tiny_improvements.py"
    completed = subprocess.run(
        [sys.executable, str(script)], check=True, capture_output=True, text=True
    )

    report = json.loads(completed.stdout)
    assert report["expert_count"] == 8
    assert report["peak_bits_at_play"] == 30
    assert run_sweep()["expert_count"] == 8

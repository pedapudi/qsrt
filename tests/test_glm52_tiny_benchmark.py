from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from qsrt.glm52_tiny_benchmark import (
    BITS_AT_PLAY,
    BIT_LIMIT,
    PAYLOAD_BITS,
    PSEUDO_VOCABULARY_READOUT,
    SOURCE_WEIGHT_COUNT,
    STAGE_NAMES,
    BenchmarkConfig,
    apply_reciprocal_balance,
    expert_output,
    make_problem,
    run_problem,
    run_sweep,
    scalar_k2_paths,
)


def test_problem_is_a_complete_glm_swiglu_expert_under_the_bit_limit() -> None:
    report = run_sweep(1)

    assert SOURCE_WEIGHT_COUNT == 8
    assert PAYLOAD_BITS == 16
    assert BITS_AT_PLAY == 30
    assert BITS_AT_PLAY <= BIT_LIMIT == 32
    assert report["bit_budget"] == {
        "source_weight_count": 8,
        "payload_bits": 16,
        "payload_bits_per_weight": 2.0,
        "scalar_k2_history_bits": 14,
        "bits_at_play": 30,
        "limit": 32,
        "aggregate_payload_bits": 16,
    }
    assert report["problem"]["expert_equation"] == (
        "down(SiLU(gate(input)) * up(input))"
    )
    assert report["problem"]["gate_weights"] == 2
    assert report["problem"]["up_weights"] == 2
    assert report["problem"]["down_weights"] == 4
    assert "pair" not in " ".join(report["stage_names"])


def test_scalar_paths_use_the_frozen_production_k2_t12_mapping() -> None:
    paths = scalar_k2_paths(np.asarray((1.0, -0.5)))

    assert paths.reconstructions.shape == (16, 2)
    assert len(set(paths.branches)) == 16
    assert all(0 <= state < 2**14 for states in paths.states for state in states)
    assert all(0 <= index < 4096 for indices in paths.table_indices for index in indices)
    assert all(
        rank >> 4 == index
        for ranks, indices in zip(paths.ranks, paths.table_indices)
        for rank, index in zip(ranks, indices)
    )


def test_scalar_path_ranks_match_the_production_sqg_graph() -> None:
    from qsrt.sqg_e4m3 import sqg_xor_rank_permutation

    paths = scalar_k2_paths(np.asarray((1.0, -0.5)))
    production_ranks = sqg_xor_rank_permutation(2).cpu().numpy()
    for branches, states, ranks in zip(paths.branches, paths.states, paths.ranks):
        for branch, state, rank in zip(branches, states, ranks):
            assert rank == int(production_ranks[(state << 2) | branch])


def test_reciprocal_balance_preserves_the_full_precision_expert() -> None:
    problem = make_problem(7)
    gauge = np.asarray((0.5, 2.0))
    gate, up, down = apply_reciprocal_balance(
        problem.source_gate, problem.source_up, problem.source_down, gauge
    )

    expected = expert_output(
        problem.fit_inputs,
        problem.source_gate,
        problem.source_up,
        problem.source_down,
    )
    actual = expert_output(problem.fit_inputs, gate, up, down)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


def test_fit_and_heldout_rows_are_independent_and_selection_ignores_heldout() -> None:
    config = BenchmarkConfig(gauge_values=(0.5, 1.0, 2.0))
    problem = make_problem(3, config=config)
    original = run_problem(problem, config=config)
    mutated = replace(
        problem,
        heldout_inputs=problem.heldout_inputs * -41.0 + 7.0,
        heldout_source_output=problem.heldout_source_output * 23.0 - 9.0,
    )
    changed = run_problem(mutated, config=config)

    assert not np.array_equal(problem.fit_inputs, problem.heldout_inputs)
    assert not np.shares_memory(problem.fit_inputs, problem.heldout_inputs)
    for stage in STAGE_NAMES:
        left = original["stages"][stage]
        right = changed["stages"][stage]
        assert left["reciprocal_gauge"] == right["reciprocal_gauge"]
        assert left["gate_k2_path"] == right["gate_k2_path"]
        assert left["up_k2_path"] == right["up_k2_path"]
        assert left["down_k2_paths"] == right["down_k2_paths"]


def test_readout_is_centered_full_rank_and_observes_both_output_directions() -> None:
    np.testing.assert_allclose(PSEUDO_VOCABULARY_READOUT.sum(axis=0), 0.0)
    assert np.linalg.matrix_rank(PSEUDO_VOCABULARY_READOUT) == 2
    assert PSEUDO_VOCABULARY_READOUT.shape == (4, 2)


def test_report_preserves_expert_level_regressions_and_selection_contracts() -> None:
    report = run_sweep(4, config=BenchmarkConfig(gauge_values=(0.5, 1.0, 2.0)))

    assert report["selection_rows"] == "fit rows only"
    assert "independently generated" in report["reporting_rows"]
    assert "cannot establish checkpoint dominance" in report["kld_evidence_boundary"]
    assert "cannot establish serialized model size" in report["size_evidence_boundary"]
    assert report["expert_count"] == 4
    assert report["stage_names"] == list(STAGE_NAMES)
    for stage in STAGE_NAMES[1:]:
        for summaries in (
            report["heldout_transitions"],
            report["heldout_forward_kld_transitions"],
        ):
            summary = summaries[stage]
            assert sum(
                summary[f"{status}_experts"]
                for status in ("improved", "unchanged", "regressed")
            ) == 4
    for expert in report["experts"]:
        for stage in STAGE_NAMES:
            assert expert["stages"][stage]["runtime_metadata_bits_for_gauge"] == 0
        refit = expert["stages"][STAGE_NAMES[-1]]["down_refit"]
        assert refit["upstream_gate_path_frozen"] is True
        assert refit["upstream_up_path_frozen"] is True
        assert refit["gauge_frozen"] is True
        control = expert["stages"][STAGE_NAMES[0]]["fit_forward_kld"]
        balanced = expert["stages"][STAGE_NAMES[1]]["fit_forward_kld"]
        fit_selected = expert["stages"][STAGE_NAMES[2]]["fit_forward_kld"]
        composed = expert["stages"][STAGE_NAMES[3]]["fit_forward_kld"]
        refitted = expert["stages"][STAGE_NAMES[4]]["fit_forward_kld"]
        assert balanced <= control + 1e-15
        assert fit_selected <= control + 1e-15
        assert composed <= min(balanced, fit_selected) + 1e-15
        assert refitted <= composed + 1e-15


@pytest.mark.parametrize("expert_count", [0, 257])
def test_expert_count_is_bounded(expert_count: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 256"):
        run_sweep(expert_count)


def test_default_eight_expert_sweep_is_fast_and_deterministic() -> None:
    started = time.perf_counter()
    first = run_sweep()
    elapsed = time.perf_counter() - started

    assert elapsed < 5.0
    assert first["expert_count"] == 8
    assert first == run_sweep()


def test_cli_defaults_to_eight_experts() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "benchmark_glm52_tiny_quantization.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--gauge-values", "0.5,1,2"],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert report["expert_count"] == 8
    assert report["bit_budget"]["bits_at_play"] == 30
    assert report["wall_seconds"] < 10.0

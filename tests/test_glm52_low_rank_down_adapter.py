from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from qsrt.glm52_low_rank_down_adapter import (
    fit_functional_down_adapter,
    materialize_bf16_down_adapter,
)


def test_materialized_adapter_checks_geometry() -> None:
    with pytest.raises(ValueError, match="do not match"):
        materialize_bf16_down_adapter(
            torch.zeros(5, 3), torch.zeros(4, 2), torch.zeros(5, 2)
        )


def test_functional_adapter_recovers_predictable_rank_two_error() -> None:
    torch.manual_seed(29)
    rows = torch.randn(160, 7, dtype=torch.float16)
    selection_rows = torch.randn(96, 7, dtype=torch.float16)
    base = torch.randn(11, 7, dtype=torch.float16) * 0.1
    factor_a = torch.randn(7, 2) * 0.2
    factor_b = torch.randn(11, 2) * 0.2
    target = base.float() + factor_b @ factor_a.T
    teacher = rows.float() @ target.T
    selection_teacher = selection_rows.float() @ target.T
    weights = torch.linspace(0.2, 1.0, rows.shape[0])
    selection_weights = torch.linspace(0.3, 1.0, selection_rows.shape[0])

    result = fit_functional_down_adapter(
        base_down=base,
        fit_hidden=rows,
        fit_teacher=teacher,
        fit_route_weights=weights,
        selection_hidden=selection_rows,
        selection_teacher=selection_teacher,
        selection_route_weights=selection_weights,
        rank=2,
        ridge_factors=(1e-5,),
        oversampling=4,
        power_iterations=2,
        batch_rows=31,
        seed=7,
    )

    before = result["baseline_selection_metrics"]["weighted_relative_sse"]
    after = result["selected"]["selection_metrics"]["weighted_relative_sse"]
    assert after < before * 0.002
    assert result["selected"]["factor_a"].dtype == torch.bfloat16
    assert result["selected"]["factor_b"].dtype == torch.bfloat16
    assert result["selected"]["dense"].dtype == torch.float16


def test_functional_adapter_rejects_invalid_ridge_grid() -> None:
    rows = torch.ones(8, 3, dtype=torch.float16)
    with pytest.raises(ValueError, match="ridge factors"):
        fit_functional_down_adapter(
            base_down=torch.zeros(4, 3, dtype=torch.float16),
            fit_hidden=rows,
            fit_teacher=torch.zeros(8, 4),
            fit_route_weights=torch.ones(8),
            selection_hidden=rows,
            selection_teacher=torch.zeros(8, 4),
            selection_route_weights=torch.ones(8),
            rank=2,
            ridge_factors=(0.0,),
            oversampling=2,
            power_iterations=1,
            batch_rows=4,
            seed=0,
        )


def test_per_expert_attribution_uses_a_distinct_runtime_control() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    launcher = (
        repository_root
        / "experiments"
        / "run_glm52_candidate_kld_chunked_full_vocabulary_on_kossel.sh"
    ).read_text()

    assert 'result_suffix="-per-expert-attribution"' in launcher
    assert (
        'control_root="${experiment_root}/runtime-control/'
        '${artifact_name}${result_suffix}-per-expert-correctness"'
        in launcher
    )


def test_attribution_selected_subsets_are_frozen_in_the_launcher() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    launcher = (
        repository_root
        / "experiments"
        / "run_glm52_candidate_kld_chunked_full_vocabulary_on_kossel.sh"
    ).read_text()

    assert "low-rank-down-refit-r2-attribution-selected-subsets" in launcher
    assert '"all_individually_improving_experts":[89,103,208]' in launcher
    assert '"strongest_pair":[89,208]' in launcher
    assert '"expert_89_and_103":[89,103]' in launcher
    assert '"expert_103_and_208":[103,208]' in launcher
    assert '"${candidate_subset_arguments[@]}"' in launcher

    assert "low-rank-down-refit-r4-attribution-selected-subsets" in launcher
    assert '"rank2_helpful_expert_89":[89]' in launcher
    assert '"rank2_helpful_expert_103":[103]' in launcher
    assert '"rank2_helpful_expert_208":[208]' in launcher
    assert '"all_rank2_individually_improving_experts":[89,103,208]' in launcher


def test_rank_four_expert_103_confirmation_registration_is_frozen() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    registration = json.loads(
        (
            repository_root
            / "experiments"
            / "glm52_layer3_rank4_expert103_low_rank_down_confirmation_registration.json"
        ).read_text()
    )

    assert registration["status"] == "frozen_before_document_disjoint_confirmation"
    correction = registration["frozen_correction"]
    assert (correction["layer"], correction["expert"], correction["rank"]) == (
        3,
        103,
        4,
    )
    assert correction["factor_dtype"] == "BF16"
    assert correction["logical_factor_bytes"] == 65_536
    assert len(correction["factor_a_sha256"]) == 64
    assert len(correction["factor_b_sha256"]) == 64

    byte_screen = registration["logical_byte_screen"]
    assert (
        byte_screen["comparison_panel_bytes"]
        - byte_screen["candidate_panel_bytes"]
        == byte_screen["candidate_margin_bytes"]
    )
    assert registration["one_context_measurement"]["candidate_mean_kld"] < 0.059
    assert registration["confirmation_contract"]["minimum_document_count"] >= 32

from __future__ import annotations

import pytest
import torch

from qsrt.glm52_down_construction import (
    down_construction_name,
    refit_passes_local_fallback,
    route_weighted_output_error_statistics,
)


def test_down_construction_names_state_both_experiment_axes() -> None:
    assert (
        down_construction_name(
            input_metric="reconstructed_input_covariance",
            target="reconstructed_activation_refit",
        )
        == "reconstructed_input_covariance__reconstructed_activation_refit"
    )
    with pytest.raises(ValueError, match="unsupported down input metric"):
        down_construction_name(input_metric="mystery", target="source_weights")


def test_route_weighted_output_error_reports_mean_and_tail() -> None:
    teacher = torch.zeros(100, 2)
    candidate = torch.zeros_like(teacher)
    candidate[:2, 0] = torch.tensor([10.0, 3.0])
    teacher[:, 1] = 1.0
    candidate[:, 1] = 1.0
    statistics = route_weighted_output_error_statistics(
        teacher, candidate, torch.ones(100)
    )
    assert statistics["weighted_relative_sse"] == pytest.approx(1.09)
    assert statistics["row_error_max"] == 100.0
    assert statistics["row_error_cvar1"] == 100.0
    assert statistics["cvar1_row_count"] == 1


def test_zero_route_weight_removes_a_row_from_error_but_not_row_count() -> None:
    teacher = torch.ones(2, 1)
    candidate = torch.tensor([[100.0], [0.0]])
    statistics = route_weighted_output_error_statistics(
        teacher, candidate, torch.tensor([0.0, 1.0])
    )
    assert statistics["row_count"] == 2
    assert statistics["weighted_relative_sse"] == 1.0


def test_local_fallback_requires_mean_gain_and_tail_noninferiority() -> None:
    baseline = {"weighted_relative_sse": 1.0, "row_error_cvar1": 4.0}
    assert refit_passes_local_fallback(
        baseline=baseline,
        candidate={"weighted_relative_sse": 0.9, "row_error_cvar1": 4.0},
        tail_relative_tolerance=0.0,
    )
    assert not refit_passes_local_fallback(
        baseline=baseline,
        candidate={"weighted_relative_sse": 0.9, "row_error_cvar1": 4.1},
        tail_relative_tolerance=0.0,
    )
    assert not refit_passes_local_fallback(
        baseline=baseline,
        candidate={"weighted_relative_sse": 1.1, "row_error_cvar1": 3.0},
        tail_relative_tolerance=0.0,
    )
    assert refit_passes_local_fallback(
        baseline=baseline,
        candidate={"weighted_relative_sse": 0.9, "row_error_cvar1": 4.1},
        tail_relative_tolerance=0.05,
    )


def test_local_fallback_rejects_invalid_tail_tolerance() -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        refit_passes_local_fallback(
            baseline={"weighted_relative_sse": 1.0, "row_error_cvar1": 1.0},
            candidate={"weighted_relative_sse": 0.5, "row_error_cvar1": 0.5},
            tail_relative_tolerance=-0.1,
        )

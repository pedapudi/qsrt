from __future__ import annotations

import torch

from qsrt.router_bias_secant import (
    build_least_squares_secant_bias_update,
    build_secant_bias_update,
)


def test_secant_update_is_bounded_zero_mean_and_correctly_signed() -> None:
    tokens = 1_000_000
    teacher = torch.tensor([80_000, 90_000, 100_000, 110_000])
    before = torch.tensor([70_000, 100_000, 90_000, 120_000])
    after = torch.tensor([75_000, 95_000, 95_000, 115_000])
    base = after.clone()
    bias0 = torch.zeros(4)
    bias1 = torch.tensor([0.001, -0.001, 0.001, -0.001])
    result = build_secant_bias_update(
        teacher_counts=teacher,
        slope_before_counts=before,
        slope_after_counts=after,
        base_counts=base,
        slope_before_bias=bias0,
        slope_after_bias=bias1,
        tokens=tokens,
        median_margin=1e-4,
        noise_sigma=0.0,
        margin_multiple=8.0,
    )
    assert result.valid.all()
    assert float(result.bounded_proposal.abs().max()) <= 8e-4 + 1e-9
    assert abs(float(result.update.mean())) < 1e-9
    residual = base.to(torch.float64) - teacher.to(torch.float64)
    assert torch.all(result.bounded_proposal.to(torch.float64) * residual <= 0)


def test_secant_update_excludes_unresolved_or_negative_slopes() -> None:
    tokens = 1_000_000
    result = build_secant_bias_update(
        teacher_counts=torch.tensor([100_000, 100_000, 100_000]),
        slope_before_counts=torch.tensor([100_000, 100_000, 100_000]),
        slope_after_counts=torch.tensor([100_001, 99_999, 100_000]),
        base_counts=torch.tensor([110_000, 90_000, 100_000]),
        slope_before_bias=torch.zeros(3),
        slope_after_bias=torch.tensor([0.001, 0.001, -0.002]),
        tokens=tokens,
        median_margin=1e-4,
        noise_sigma=2.5,
        margin_multiple=32.0,
    )
    assert not result.resolved.any()
    assert not result.valid.any()
    assert torch.equal(result.update, torch.zeros_like(result.update))


def test_least_squares_secant_recovers_positive_frequency_response() -> None:
    tokens = 1_000_000
    teacher_frequency = torch.tensor([0.20, 0.20, 0.20, 0.20], dtype=torch.float64)
    expert_direction = torch.tensor([0.001, -0.001, 0.002, -0.002])
    response = torch.tensor([2.0, 2.0, 1.0, 1.0], dtype=torch.float64)
    round_scale = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])[:, None]
    round_biases = round_scale * expert_direction[None, :]
    round_frequency = teacher_frequency[None, :] + response[None, :] * round_biases
    round_counts = torch.round(round_frequency * tokens).to(torch.int64)
    result = build_least_squares_secant_bias_update(
        teacher_counts=torch.round(teacher_frequency * tokens).to(torch.int64),
        round_counts=round_counts,
        round_biases=round_biases,
        tokens=tokens,
        median_margin=0.001,
        noise_sigma=2.5,
        margin_multiple=64.0,
    )
    assert result.slope_resolved.all()
    assert result.residual_resolved.all()
    assert result.valid.all()
    assert torch.allclose(result.slopes, response.to(torch.float32), atol=1e-5)
    assert torch.allclose(
        result.update,
        (-2.0 * expert_direction).to(torch.float32),
        atol=1e-6,
    )
    assert torch.allclose(
        result.predicted_frequency,
        teacher_frequency.to(torch.float32),
        atol=1e-6,
    )


def test_least_squares_secant_rejects_negative_slope_and_noise_residual() -> None:
    tokens = 1_000_000
    round_biases = torch.tensor(
        [
            [-0.002, 0.002],
            [-0.001, 0.001],
            [0.0, 0.0],
            [0.001, -0.001],
            [0.002, -0.002],
        ]
    )
    round_counts = torch.tensor(
        [
            [98_000, 98_000],
            [99_000, 99_000],
            [100_000, 100_000],
            [101_000, 101_000],
            [102_000, 102_000],
        ]
    )
    result = build_least_squares_secant_bias_update(
        teacher_counts=torch.tensor([102_001, 102_001]),
        round_counts=round_counts,
        round_biases=round_biases,
        tokens=tokens,
        median_margin=0.001,
    )
    assert result.slope_resolved.tolist() == [True, False]
    assert result.residual_resolved.tolist() == [False, False]
    assert not result.valid.any()
    assert torch.equal(result.update, torch.zeros_like(result.update))

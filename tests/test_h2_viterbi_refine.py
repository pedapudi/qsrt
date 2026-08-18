from __future__ import annotations

import torch

from qsrt.h2_viterbi_refine import (
    H2ViterbiRefineConfig,
    dense_h_guided_objective,
    dense_h_objective,
    refine_h2_viterbi_paths,
)
from qsrt.gradient_guided_viterbi import (
    LowRankViterbiGradientGuidance,
    ViterbiGradientGuidance,
)


class NearestGridQuantizer:
    def __init__(self, levels: torch.Tensor):
        self.levels = levels

    def __call__(self, tiles: torch.Tensor):
        levels = self.levels.to(device=tiles.device, dtype=tiles.dtype)
        payload = (tiles[..., None] - levels).abs().argmin(dim=-1)
        return levels[payload], payload.to(torch.int16)


def _make_spd(size: int, condition: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    source = torch.randn((size, size), generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(source)
    log_condition = torch.log(torch.tensor(condition, dtype=torch.float64))
    eigenvalues = torch.exp(
        torch.linspace(
            -0.5 * log_condition,
            0.5 * log_condition,
            size,
            dtype=torch.float64,
        )
    )
    hessian = basis @ torch.diag(eigenvalues) @ basis.T
    hessian += 0.025 * torch.eye(size, dtype=torch.float64)
    return (hessian / torch.diag(hessian).mean()).float()


def _independent_quantization(
    weight: torch.Tensor, quantizer: NearestGridQuantizer
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = weight.shape
    flattened = (
        weight.reshape(rows // 16, 16, columns // 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(-1, 256)
    )
    reconstruction, payload = quantizer(flattened)
    reconstruction = (
        reconstruction.reshape(rows // 16, columns // 16, 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(rows, columns)
    )
    return reconstruction, payload.reshape(rows // 16, columns // 16, 256)


def test_dense_h_objective_matches_direct_expression() -> None:
    torch.manual_seed(1)
    weight = torch.randn(32, 16)
    reconstruction = torch.randn(32, 16)
    hessian = _make_spd(32, 3.0, 2)
    error = weight - reconstruction
    direct = float(torch.sum(error * (hessian @ error)))
    measured = dense_h_objective(weight, reconstruction, hessian)
    assert abs(measured - direct) < 1.0e-4


def test_refinement_is_monotone_and_preserves_shapes() -> None:
    torch.manual_seed(3)
    weight = torch.randn(64, 32)
    hessian = _make_spd(64, 5.0, 4)
    quantizer = NearestGridQuantizer(torch.tensor([-1.5, -0.5, 0.5, 1.5]))
    initial, payload = _independent_quantization(weight, quantizer)
    before = dense_h_objective(weight, initial, hessian)
    result = refine_h2_viterbi_paths(
        weight,
        hessian,
        initial,
        payload,
        quantizer,
        H2ViterbiRefineConfig(
            sweeps=2,
            dither_scales=(0.02, 0.04),
            num_dither_patterns=2,
            score_dtype=torch.float64,
        ),
    )
    after = dense_h_objective(weight, result.weight_q, hessian)
    assert result.weight_q.shape == initial.shape
    assert result.encoded.shape == payload.shape
    assert after <= before + 1.0e-6
    assert all(
        stats.objective_after <= stats.objective_before + 1.0e-6
        for stats in result.sweep_stats
    )


def test_identity_hessian_is_a_no_op() -> None:
    torch.manual_seed(5)
    weight = torch.randn(32, 16)
    quantizer = NearestGridQuantizer(torch.tensor([-1.0, 0.0, 1.0]))
    initial, payload = _independent_quantization(weight, quantizer)
    result = refine_h2_viterbi_paths(
        weight,
        torch.eye(32),
        initial,
        payload,
        quantizer,
        H2ViterbiRefineConfig(),
    )
    assert result.sweep_stats[0].accepted_tiles == 0
    assert torch.equal(result.weight_q, initial)
    assert torch.equal(result.encoded, payload)


def test_packed_payload_shape_is_preserved() -> None:
    torch.manual_seed(7)
    weight = torch.randn(32, 16)
    hessian = _make_spd(32, 2.0, 8)

    class PackedQuantizer(NearestGridQuantizer):
        def __call__(self, tiles: torch.Tensor):
            reconstruction, payload = super().__call__(tiles)
            return reconstruction, payload.reshape(payload.shape[0], 128, 2)

    quantizer = PackedQuantizer(torch.tensor([-1.0, 0.0, 1.0]))
    initial, scalar_payload = _independent_quantization(weight, quantizer)
    payload = scalar_payload.reshape(2, 1, 128, 2)
    result = refine_h2_viterbi_paths(
        weight,
        hessian,
        initial,
        payload,
        quantizer,
        H2ViterbiRefineConfig(
            dither_scales=(0.02,), num_dither_patterns=1
        ),
    )
    assert result.encoded.shape == payload.shape


def test_column_weights_match_explicit_two_sided_scaling() -> None:
    torch.manual_seed(9)
    weight = torch.randn(32, 16)
    reconstruction = torch.randn(32, 16)
    source_hessian = _make_spd(32, 3.0, 10)
    input_scale = torch.exp(torch.randn(32) * 0.2)
    output_scale = torch.exp(torch.randn(16) * 0.2)
    source_error = (
        input_scale[:, None]
        * (weight - reconstruction)
        * output_scale[None]
    )
    explicit = float(torch.sum(source_error * (source_hessian @ source_error)))
    encoder_hessian = (
        input_scale[:, None] * source_hessian * input_scale[None, :]
    )
    measured = dense_h_objective(
        weight,
        reconstruction,
        encoder_hessian,
        column_weights=output_scale.square(),
    )
    assert abs(measured - explicit) / abs(explicit) < 1.0e-6


def test_zero_strength_guidance_matches_dense_h_refinement() -> None:
    torch.manual_seed(11)
    weight = torch.randn(48, 32)
    hessian = _make_spd(48, 4.0, 12)
    quantizer = NearestGridQuantizer(torch.tensor([-1.5, -0.5, 0.5, 1.5]))
    initial, payload = _independent_quantization(weight, quantizer)
    config = H2ViterbiRefineConfig(sweeps=2, score_dtype=torch.float64)
    control = refine_h2_viterbi_paths(
        weight, hessian, initial, payload, quantizer, config
    )
    guided = refine_h2_viterbi_paths(
        weight,
        hessian,
        initial,
        payload,
        quantizer,
        config,
        guidance=ViterbiGradientGuidance(
            gradient=torch.randn_like(weight),
            anchor=initial,
            anchor_id="test-anchor",
            objective_id="test-objective",
            strength=0.0,
        ),
    )
    assert torch.equal(guided.weight_q, control.weight_q)
    assert torch.equal(guided.encoded, control.encoded)
    assert guided.stats_dict() == control.stats_dict()


def test_gradient_guidance_is_monotone_from_the_anchor() -> None:
    torch.manual_seed(13)
    weight = torch.randn(32, 32)
    hessian = _make_spd(32, 2.5, 14)
    quantizer = NearestGridQuantizer(torch.tensor([-1.0, 0.0, 1.0]))
    initial, payload = _independent_quantization(weight, quantizer)
    guidance = ViterbiGradientGuidance(
        gradient=torch.randn_like(weight),
        anchor=initial,
        anchor_id="test-anchor",
        objective_id="test-objective",
        strength=0.35,
    )
    before = dense_h_guided_objective(weight, initial, hessian, guidance)
    result = refine_h2_viterbi_paths(
        weight,
        hessian,
        initial,
        payload,
        quantizer,
        H2ViterbiRefineConfig(sweeps=3, score_dtype=torch.float64),
        guidance=guidance,
    )
    after = dense_h_guided_objective(weight, result.weight_q, hessian, guidance)
    assert after[2] <= before[2] + 1.0e-6
    assert all(
        stats.objective_after <= stats.objective_before + 1.0e-6
        for stats in result.sweep_stats
    )


def test_guided_conditional_target_respects_column_weights() -> None:
    torch.manual_seed(15)
    weight = torch.randn(16, 16)
    hessian = _make_spd(16, 2.0, 16)
    column_weights = torch.linspace(0.5, 2.0, 16)
    quantizer = NearestGridQuantizer(torch.tensor([-1.0, 0.0, 1.0]))
    initial, payload = _independent_quantization(weight, quantizer)
    guidance = ViterbiGradientGuidance(
        gradient=torch.randn_like(weight),
        anchor=initial,
        anchor_id="test-anchor",
        objective_id="test-objective",
        strength=0.2,
    )
    before = dense_h_guided_objective(
        weight,
        initial,
        hessian,
        guidance,
        column_weights=column_weights,
    )
    result = refine_h2_viterbi_paths(
        weight,
        hessian,
        initial,
        payload,
        quantizer,
        H2ViterbiRefineConfig(score_dtype=torch.float64),
        column_weights=column_weights,
        guidance=guidance,
    )
    after = dense_h_guided_objective(
        weight,
        result.weight_q,
        hessian,
        guidance,
        column_weights=column_weights,
    )
    assert after[2] <= before[2] + 1.0e-6


def test_low_rank_guidance_matches_dense_guidance() -> None:
    torch.manual_seed(17)
    weight = torch.randn(32, 32)
    hessian = _make_spd(32, 3.0, 18)
    quantizer = NearestGridQuantizer(torch.tensor([-1.0, 0.0, 1.0]))
    initial, payload = _independent_quantization(weight, quantizer)
    left = torch.randn(32, 5)
    right = torch.randn(5, 32)
    common = {
        "anchor": initial,
        "anchor_id": "test-anchor",
        "objective_id": "test-objective",
        "strength": 0.07,
    }
    dense = ViterbiGradientGuidance(gradient=left @ right, **common)
    low_rank = LowRankViterbiGradientGuidance(
        left=left,
        right=right,
        **common,
    )
    config = H2ViterbiRefineConfig(sweeps=2, score_dtype=torch.float64)
    dense_result = refine_h2_viterbi_paths(
        weight,
        hessian,
        initial,
        payload,
        quantizer,
        config,
        guidance=dense,
    )
    low_rank_result = refine_h2_viterbi_paths(
        weight,
        hessian,
        initial,
        payload,
        quantizer,
        config,
        guidance=low_rank,
    )
    assert torch.equal(low_rank_result.weight_q, dense_result.weight_q)
    assert torch.equal(low_rank_result.encoded, dense_result.encoded)
    dense_objective = dense_h_guided_objective(
        weight,
        dense_result.weight_q,
        hessian,
        dense,
    )
    low_rank_objective = dense_h_guided_objective(
        weight,
        low_rank_result.weight_q,
        hessian,
        low_rank,
    )
    torch.testing.assert_close(
        torch.tensor(low_rank_objective),
        torch.tensor(dense_objective),
    )

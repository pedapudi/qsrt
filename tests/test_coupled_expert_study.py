from __future__ import annotations

import math

import pytest
import torch

from qsrt.coupled_expert_study import (
    CoupledTriplet,
    RateComponent,
    RoutedOutputMetric,
    allocate_rate_options,
    apply_common_input_gauge,
    apply_output_rotation,
    apply_permutation_sign_gauge,
    apply_postactivation_scale,
    apply_w3_w2_sign_draw,
    apply_w3_w2_scale_gauge,
    block_hadamard,
    blockwise_codebook_quantize,
    conditional_entropy_bits,
    effective_bpw,
    encode_coupled_block_hadamard,
    encode_two_sided_linear,
    entropy_bits,
    execute_coupled_block_hadamard,
    execute_two_sided_linear,
    expert_hidden,
    expert_output,
    fit_cross_matrix_predictor,
    fit_function_space_correction,
    fit_metric_codebook,
    local_triplet_metrics,
    pair_activation_metric,
    pair_residual_decomposition,
    quantize_metric_codebook,
    radial_tangent_decomposition,
    ridge_refit_down,
    route_error_covariance,
    search_expert_output_gain,
    select_corouted_candidate_modes,
    situ_component_geometry,
    situ_derivatives,
    situ_value,
    sparse_fingerprint_alignment,
    temperature_scaled_situ,
)


def _triplet(*, seed: int = 0, hidden: int = 5, intermediate: int = 7) -> CoupledTriplet:
    generator = torch.Generator().manual_seed(seed)
    return CoupledTriplet(
        torch.randn(intermediate, hidden, generator=generator) * 0.2,
        torch.randn(intermediate, hidden, generator=generator) * 0.2,
        torch.randn(hidden, intermediate, generator=generator) * 0.2,
    )


def _orthogonal(width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(width, width, generator=generator)).Q


def test_situ_derivatives_match_finite_difference() -> None:
    generator = torch.Generator().manual_seed(1)
    gate = torch.randn(4, 6, generator=generator)
    up = torch.randn(4, 6, generator=generator)
    d_gate, d_up = situ_derivatives(gate, up)
    epsilon = 2e-3
    numeric_gate = (situ_value(gate + epsilon, up) - situ_value(gate - epsilon, up)) / (
        2 * epsilon
    )
    numeric_up = (situ_value(gate, up + epsilon) - situ_value(gate, up - epsilon)) / (
        2 * epsilon
    )
    torch.testing.assert_close(d_gate, numeric_gate, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(d_up, numeric_up, rtol=2e-3, atol=2e-3)


def test_situ_component_geometry_reconstructs_activation_and_derivatives() -> None:
    generator = torch.Generator().manual_seed(10101)
    gate = torch.randn(4, 6, generator=generator) * 3
    up = torch.randn(4, 6, generator=generator) * 12
    geometry = situ_component_geometry(gate, up)
    torch.testing.assert_close(
        geometry.gate_factor * geometry.up_factor,
        situ_value(gate, up),
    )
    d_gate, d_up = situ_derivatives(gate, up)
    torch.testing.assert_close(
        geometry.up_factor * geometry.gate_derivative,
        d_gate,
    )
    torch.testing.assert_close(
        geometry.gate_factor * geometry.up_derivative,
        d_up,
    )
    assert torch.all((geometry.up_derivative >= 0) & (geometry.up_derivative <= 1))


def test_rms_output_jacobian_and_radial_suppression() -> None:
    generator = torch.Generator().manual_seed(2)
    aggregate = torch.randn(5, 4, generator=generator)
    vectors = torch.randn(5, 3, 4, generator=generator)
    metric = RoutedOutputMetric(torch.rand(4, generator=generator) + 0.5, torch.randn(6, 4, generator=generator))
    mapped = metric.jacobian_vectors(aggregate, vectors)
    epsilon = 2e-3
    numeric = torch.stack(
        [
            (metric.output(aggregate + epsilon * vectors[:, item]) - metric.output(aggregate - epsilon * vectors[:, item]))
            / (2 * epsilon)
            for item in range(vectors.shape[1])
        ],
        dim=1,
    )
    torch.testing.assert_close(mapped, numeric, rtol=2e-3, atol=2e-3)

    nearly_scale_invariant = metric.jacobian_vectors(aggregate, aggregate)
    ordinary = metric.jacobian_vectors(aggregate, vectors[:, 0])
    assert nearly_scale_invariant.square().mean() < ordinary.square().mean() * 1e-6


def test_exact_parameter_gauges_close() -> None:
    triplet = _triplet(seed=3)
    inputs = torch.randn(9, triplet.hidden, generator=torch.Generator().manual_seed(4))
    reference = expert_output(inputs, triplet)

    permutation = torch.randperm(triplet.intermediate, generator=torch.Generator().manual_seed(5))
    signs = torch.where(torch.arange(triplet.intermediate) % 2 == 0, 1.0, -1.0)
    gauged = apply_permutation_sign_gauge(triplet, permutation, signs)
    torch.testing.assert_close(expert_output(inputs, gauged), reference, rtol=2e-5, atol=2e-5)

    transform = torch.randn(triplet.hidden, triplet.hidden, generator=torch.Generator().manual_seed(6))
    transform += torch.eye(triplet.hidden) * 3
    input_gauged = apply_common_input_gauge(triplet, transform)
    torch.testing.assert_close(
        expert_output(inputs @ transform.T, input_gauged), reference, rtol=3e-5, atol=3e-5
    )

    rotation = _orthogonal(triplet.hidden, 7)
    output_gauged = apply_output_rotation(triplet, rotation)
    torch.testing.assert_close(
        expert_output(inputs, output_gauged), reference @ rotation.T, rtol=2e-5, atol=2e-5
    )

    scale = torch.linspace(0.5, 1.5, triplet.intermediate)
    hidden = expert_hidden(inputs, triplet)
    scaled = apply_postactivation_scale(triplet, scale)
    torch.testing.assert_close(
        (hidden * scale) @ scaled.down.T, reference, rtol=2e-5, atol=2e-5
    )


def test_temperature_and_two_sided_closures() -> None:
    generator = torch.Generator().manual_seed(8)
    gate = torch.randn(3, 6, generator=generator)
    up = torch.randn(3, 6, generator=generator)
    gate_scale = torch.linspace(0.7, 1.3, 6)
    up_scale = torch.linspace(1.4, 0.8, 6)
    transformed = temperature_scaled_situ(
        gate * gate_scale,
        up * up_scale,
        gate_scale,
        up_scale,
    )
    torch.testing.assert_close(
        transformed,
        situ_value(gate, up) * gate_scale * up_scale,
        rtol=2e-5,
        atol=2e-5,
    )

    weight = torch.randn(5, 4, generator=generator)
    inputs = torch.randn(7, 4, generator=generator)
    left = _orthogonal(5, 9)
    right = _orthogonal(4, 10)
    encoded = encode_two_sided_linear(weight, left, right)
    torch.testing.assert_close(
        execute_two_sided_linear(inputs, encoded, left, right),
        inputs @ weight.T,
        rtol=2e-5,
        atol=2e-5,
    )

    blocked = torch.randn(3, 16, generator=generator)
    original_blocked = blocked.clone()
    torch.testing.assert_close(
        block_hadamard(block_hadamard(blocked, block_size=8), block_size=8),
        blocked,
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(blocked, original_blocked)
    quantized = blockwise_codebook_quantize(blocked, block_size=8)
    assert quantized.shape == blocked.shape
    assert torch.isfinite(quantized).all()
    assert (quantized - blocked).square().sum() < blocked.square().sum()


def test_coupled_block_hadamard_closes_across_activation() -> None:
    triplet = _triplet(seed=101, hidden=16, intermediate=16)
    inputs = torch.randn(7, 16, generator=torch.Generator().manual_seed(102))
    for residual_draw, intermediate_draw in ((0, 0), (1, 0), (0, 7), (2, 5)):
        encoded = encode_coupled_block_hadamard(
            triplet,
            block_size=8,
            residual_rotation_draw=residual_draw,
            intermediate_rotation_draw=intermediate_draw,
        )
        torch.testing.assert_close(
            execute_coupled_block_hadamard(
                inputs,
                encoded,
                block_size=8,
                residual_rotation_draw=residual_draw,
                intermediate_rotation_draw=intermediate_draw,
            ),
            expert_output(inputs, triplet),
            rtol=3e-5,
            atol=3e-5,
        )


def test_coupled_block_hadamard_closes_with_distinct_boundary_widths() -> None:
    triplet = _triplet(seed=103, hidden=16, intermediate=16)
    inputs = torch.randn(7, 16, generator=torch.Generator().manual_seed(104))
    encoded = encode_coupled_block_hadamard(
        triplet,
        block_size=8,
        preactivation_block_size=32,
        postactivation_block_size=16,
    )
    torch.testing.assert_close(
        execute_coupled_block_hadamard(
            inputs,
            encoded,
            block_size=8,
            preactivation_block_size=32,
            postactivation_block_size=16,
        ),
        expert_output(inputs, triplet),
        rtol=3e-5,
        atol=3e-5,
    )


def test_w3_w2_sign_draw_is_exact_and_deterministic() -> None:
    triplet = _triplet(seed=117, hidden=16, intermediate=8)
    inputs = torch.randn(5, 16, generator=torch.Generator().manual_seed(118))
    expected = expert_output(inputs, triplet)
    first = apply_w3_w2_sign_draw(triplet, draw=3)
    second = apply_w3_w2_sign_draw(triplet, draw=3)
    assert torch.equal(first.up, second.up)
    assert torch.equal(first.down, second.down)
    torch.testing.assert_close(expert_output(inputs, first), expected)
    identity = apply_w3_w2_sign_draw(triplet, draw=0)
    assert torch.equal(identity.up, triplet.up)
    assert torch.equal(identity.down, triplet.down)


def test_w3_w2_scale_gauge_is_bounded_and_validated() -> None:
    triplet = _triplet(seed=119, hidden=16, intermediate=8)
    gauged = apply_w3_w2_scale_gauge(
        triplet,
        policy="down_rms",
        strength=0.5,
    )
    scale = gauged.up / triplet.up
    finite_scale = scale[torch.isfinite(scale)]
    assert torch.all(finite_scale.abs() >= 0.5)
    assert torch.all(finite_scale.abs() <= 2.0)
    assert torch.allclose(gauged.up * gauged.down.T, triplet.up * triplet.down.T)
    assert apply_w3_w2_scale_gauge(
        triplet,
        policy="identity",
        strength=0.0,
    ) is triplet
    with pytest.raises(ValueError, match="unknown"):
        apply_w3_w2_scale_gauge(triplet, policy="bad", strength=0.5)


def test_pair_metric_and_residual_cross_term() -> None:
    generator = torch.Generator().manual_seed(11)
    gate = torch.randn(10, 4, generator=generator)
    up = torch.randn(10, 4, generator=generator)
    summary = pair_activation_metric(gate, up)
    assert summary.metric.shape == (4, 2, 2)
    assert torch.all(torch.linalg.eigvalsh(summary.metric) >= -1e-10)
    assert torch.all((summary.small_eigenvalue_fraction >= 0) & (summary.small_eigenvalue_fraction <= 0.5))

    source = _triplet(seed=12, hidden=4, intermediate=5)
    candidate = CoupledTriplet(
        source.gate + 0.01,
        source.up - 0.01,
        source.down,
    )
    inputs = torch.randn(8, 4, generator=generator)
    decomposition = pair_residual_decomposition(inputs, source, candidate)
    assert decomposition["joint_linear_sse"] == pytest.approx(
        decomposition["separate_linear_sse"] + decomposition["cross_term"], rel=1e-6
    )


def test_local_triplet_metric_matches_finite_difference() -> None:
    generator = torch.Generator().manual_seed(13)
    source = _triplet(seed=14, hidden=4, intermediate=5)
    inputs = torch.randn(6, 4, generator=generator)
    route_gates = torch.rand(6, generator=generator)
    aggregate = torch.randn(6, 4, generator=generator)
    output_metric = RoutedOutputMetric(torch.rand(4, generator=generator) + 0.5, torch.randn(7, 4, generator=generator))
    neuron = 2
    coordinate = 1
    actual = local_triplet_metrics(
        inputs,
        source,
        route_gates=route_gates,
        aggregate=aggregate,
        output_metric=output_metric,
        neuron_indices=torch.tensor([neuron]),
        coordinate_indices=torch.tensor([coordinate]),
    )[0]

    epsilon = 2e-3
    numeric_vectors = []
    for matrix in ("gate", "up", "down"):
        plus = [value.clone() for value in source.tensors()]
        minus = [value.clone() for value in source.tensors()]
        matrix_index = {"gate": 0, "up": 1, "down": 2}[matrix]
        index = (neuron, coordinate) if matrix != "down" else (coordinate, neuron)
        plus[matrix_index][index] += epsilon
        minus[matrix_index][index] -= epsilon
        output_plus = expert_output(inputs, CoupledTriplet(*plus))
        output_minus = expert_output(inputs, CoupledTriplet(*minus))
        derivative = route_gates[:, None] * (output_plus - output_minus) / (2 * epsilon)
        numeric_vectors.append(output_metric.jacobian_vectors(aggregate, derivative))
    numeric = torch.stack(numeric_vectors, dim=1).double()
    expected = torch.einsum("rqi,rsi->qs", numeric, numeric) / inputs.shape[0]
    torch.testing.assert_close(actual, expected, rtol=4e-3, atol=3e-5)


def test_routed_radial_and_cross_expert_decompositions() -> None:
    generator = torch.Generator().manual_seed(15)
    aggregate = torch.randn(8, 5, generator=generator)
    error = torch.randn(8, 5, generator=generator)
    split = radial_tangent_decomposition(aggregate, error)
    torch.testing.assert_close(split["radial"] + split["tangent"], error)
    assert torch.max(torch.abs((split["tangent"] * aggregate).sum(dim=1))) < 2e-6

    expert_errors = torch.randn(8, 3, 5, generator=generator)
    gates = torch.rand(8, 3, generator=generator)
    metric = RoutedOutputMetric(torch.ones(5), torch.eye(5))
    result = route_error_covariance(aggregate, expert_errors, gates, metric)
    assert result["total_sse"] == pytest.approx(
        result["diagonal_sse"] + result["cross_term"], rel=1e-10, abs=1e-10
    )


def test_corouted_candidate_selection_finds_cancelling_modes() -> None:
    expert_ids = torch.tensor([[0, 1], [0, 1]])
    errors = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, -1.0]]],
            [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, -1.0]]],
        ]
    )
    selected = select_corouted_candidate_modes(expert_ids, errors)
    torch.testing.assert_close(selected["selection"], torch.tensor([1, 1]))
    assert selected["objective"] == pytest.approx(0.0)
    assert selected["selected_unary"] == pytest.approx(4.0)
    assert selected["cross_term"] == pytest.approx(-4.0)


def test_corouted_candidate_selection_respects_unary_filter() -> None:
    expert_ids = torch.tensor([[0, 1]])
    errors = torch.tensor(
        [[[[1.0], [10.0]], [[1.0], [-10.0]]]]
    )
    unrestricted = select_corouted_candidate_modes(expert_ids, errors)
    assert unrestricted["objective"] == pytest.approx(0.0)
    restricted = select_corouted_candidate_modes(
        expert_ids,
        errors,
        valid_modes=torch.tensor(
            [[True, True], [True, True], [True, False], [True, False]]
        ),
        unary_relative_slack=0.1,
    )
    torch.testing.assert_close(
        restricted["selection"], torch.tensor([0, 0, -1, -1])
    )
    assert restricted["objective"] == pytest.approx(4.0)


def test_alignment_metric_vq_and_predictors() -> None:
    generator = torch.Generator().manual_seed(16)
    template = torch.randn(20, 6, generator=generator)
    template = template / template.norm(dim=1, keepdim=True)
    permutation = torch.randperm(20, generator=generator)
    source = template.index_select(0, permutation)
    alignment = sparse_fingerprint_alignment(source, template, candidate_counts=(4, 8, 20))
    torch.testing.assert_close(source.index_select(0, alignment), template)

    values = torch.cat(
        (torch.randn(100, 2, generator=generator) * 0.1 - 1, torch.randn(100, 2, generator=generator) * 0.1 + 1)
    )
    metrics = torch.eye(2).repeat(values.shape[0], 1, 1)
    codebook = fit_metric_codebook(values, 2, metrics=metrics, seed=17)
    quantized, _ = quantize_metric_codebook(values, codebook, metrics=metrics)
    assert (values - quantized).square().mean() < values.square().mean() * 0.05

    gate = torch.randn(7, 5, generator=generator)
    up = torch.randn(7, 5, generator=generator)
    coefficients = torch.randn(7, 2, generator=generator)
    target = coefficients[:, :1] * gate + coefficients[:, 1:] * up
    triplet = CoupledTriplet(gate, up, target.T)
    predicted = fit_cross_matrix_predictor(triplet, per="neuron")
    assert predicted["residual_fraction"] < 1e-10


def test_refits_corrections_gain_entropy_and_rate() -> None:
    generator = torch.Generator().manual_seed(18)
    hidden = torch.randn(12, 5, generator=generator)
    quantized_hidden = hidden + torch.randn(12, 5, generator=generator) * 0.03
    down = torch.randn(4, 5, generator=generator)
    refit = ridge_refit_down(down, hidden, quantized_hidden, regularization=1e-3)
    before = (quantized_hidden @ down.T - hidden @ down.T).square().sum()
    after = (quantized_hidden @ refit.T - hidden @ down.T).square().sum()
    assert after < before

    features = torch.randn(20, 6, generator=generator)
    left_true = torch.randn(6, 2, generator=generator)
    right_true = torch.randn(2, 7, generator=generator)
    target = features @ left_true @ right_true
    left, right = fit_function_space_correction(features, target, rank=2, regularization=1e-5)
    assert (features @ left @ right - target).square().mean() < 1e-7

    aggregate = torch.randn(10, 4, generator=generator)
    source_output = torch.randn(10, 4, generator=generator)
    candidate_output = source_output / 1.2
    gates = torch.rand(10, generator=generator)
    metric = RoutedOutputMetric(torch.ones(4), torch.eye(4))
    gain = search_expert_output_gain(
        aggregate,
        source_output,
        candidate_output,
        gates,
        metric,
        torch.linspace(0.8, 1.4, 31),
    )
    assert gain["gain"] == pytest.approx(1.2, abs=0.021)
    assert gain["sse"] < gain["unit_gain_sse"] * 1e-6

    symbols = torch.tensor([0, 0, 1, 1])
    contexts = torch.tensor([0, 0, 1, 1])
    assert entropy_bits(symbols) == pytest.approx(1.0)
    assert conditional_entropy_bits(symbols, contexts) == pytest.approx(0.0)
    rate = effective_bpw(100, [RateComponent("payload", 190), RateComponent("metadata", 5)])
    assert rate["effective_bpw"] == pytest.approx(1.95)

    distortion = torch.tensor([[4.0, 1.0], [3.0, 1.0], [5.0, 1.0]])
    bits = torch.tensor([[1, 3], [1, 3], [1, 3]], dtype=torch.int64)
    allocation = allocate_rate_options(distortion, bits, target_bits=5)
    assert allocation["bits"] <= 5
    assert allocation["distortion"] < 12.0
    assert math.isfinite(allocation["penalty"])

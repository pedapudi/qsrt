import math

import torch

from qsrt.suffix_fisher import (
    attention_residual_prefix_pullback,
    empirical_fisher_factor,
    kimi_layer92_routed_fisher_pullback,
    paired_lm_head_fisher_gradients,
    rms_norm_pullback,
    sketch_a_input_factor_update,
    sketch_a_output_factor_update,
)


def _rms_norm(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    reciprocal_rms = torch.rsqrt(
        inputs.square().mean(dim=-1, keepdim=True) + epsilon
    )
    return inputs * reciprocal_rms * weight


def test_rms_norm_pullback_matches_autograd() -> None:
    generator = torch.Generator().manual_seed(1)
    inputs = torch.randn((3, 7), generator=generator, dtype=torch.float64)
    inputs.requires_grad_()
    weight = torch.randn(7, generator=generator, dtype=torch.float64)
    gradient = torch.randn((3, 7), generator=generator, dtype=torch.float64)
    output = _rms_norm(inputs, weight, 2e-5)
    expected = torch.autograd.grad((output * gradient).sum(), inputs)[0]
    actual = rms_norm_pullback(inputs.detach(), gradient, weight, epsilon=2e-5)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_attention_residual_prefix_pullback_matches_autograd() -> None:
    generator = torch.Generator().manual_seed(2)
    blocks = torch.randn((4, 3, 9), generator=generator, dtype=torch.float64)
    prefix = torch.randn((4, 9), generator=generator, dtype=torch.float64)
    prefix.requires_grad_()
    query = torch.randn(9, generator=generator, dtype=torch.float64)
    gradient = torch.randn((4, 9), generator=generator, dtype=torch.float64)
    sources = torch.cat((blocks, prefix[:, None, :]), dim=1)
    reciprocal_rms = torch.rsqrt(sources.square().mean(dim=-1) + 1e-5)
    scores = (sources * query).sum(dim=-1) * reciprocal_rms
    weights = torch.softmax(scores, dim=1)
    mixed = (weights[..., None] * sources).sum(dim=1)
    expected = torch.autograd.grad((mixed * gradient).sum(), prefix)[0]
    actual = attention_residual_prefix_pullback(
        prefix.detach(),
        mixed.detach(),
        weights[:, -1].detach(),
        gradient,
        query,
        epsilon=1e-5,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)


def test_kimi_layer92_routed_pullback_matches_autograd() -> None:
    generator = torch.Generator().manual_seed(3)
    batch, latent, hidden, blocks_count = 3, 5, 8, 4
    routed = torch.randn((batch, latent), generator=generator, dtype=torch.float64)
    routed.requires_grad_()
    routed_norm = torch.randn(latent, generator=generator, dtype=torch.float64)
    routed_projection = torch.randn(
        (hidden, latent), generator=generator, dtype=torch.float64
    )
    shared = torch.randn((batch, hidden), generator=generator, dtype=torch.float64)
    prefix = torch.randn((batch, hidden), generator=generator, dtype=torch.float64)
    blocks = torch.randn(
        (batch, blocks_count, hidden), generator=generator, dtype=torch.float64
    )
    attention_query = torch.randn(hidden, generator=generator, dtype=torch.float64)
    final_norm = torch.randn(hidden, generator=generator, dtype=torch.float64)
    normalized_gradient = torch.randn(
        (batch, hidden), generator=generator, dtype=torch.float64
    )

    routed_hidden = _rms_norm(routed, routed_norm, 2e-5) @ routed_projection.T
    updated_prefix = prefix + routed_hidden + shared
    sources = torch.cat((blocks, updated_prefix[:, None, :]), dim=1)
    reciprocal_rms = torch.rsqrt(sources.square().mean(dim=-1) + 3e-5)
    scores = (sources * attention_query).sum(dim=-1) * reciprocal_rms
    weights = torch.softmax(scores, dim=1)
    mixed = (weights[..., None] * sources).sum(dim=1)
    normalized = _rms_norm(mixed, final_norm, 4e-5)
    expected = torch.autograd.grad(
        (normalized * normalized_gradient).sum(), routed
    )[0]

    actual = kimi_layer92_routed_fisher_pullback(
        routed.detach(),
        updated_prefix.detach(),
        mixed.detach(),
        weights[:, -1].detach(),
        normalized_gradient,
        final_norm_weight=final_norm,
        final_norm_epsilon=4e-5,
        attention_score_query=attention_query,
        attention_epsilon=3e-5,
        routed_norm_weight=routed_norm,
        routed_norm_epsilon=2e-5,
        routed_projection_weight=routed_projection,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-12)


def test_paired_lm_head_samples_reproduce_softmax_fisher() -> None:
    generator = torch.Generator().manual_seed(4)
    weight = torch.randn((5, 3), generator=generator, dtype=torch.float64)
    probabilities = torch.softmax(
        torch.randn(5, generator=generator, dtype=torch.float64), dim=0
    )
    first, second = torch.meshgrid(torch.arange(5), torch.arange(5), indexing="ij")
    gradients = paired_lm_head_fisher_gradients(
        weight,
        first.reshape(-1),
        second.reshape(-1),
        logit_scale=1.7,
    )
    pair_probabilities = (
        probabilities[:, None] * probabilities[None, :]
    ).reshape(-1)
    sampled = torch.einsum("n,ni,nj->ij", pair_probabilities, gradients, gradients)
    centered = weight - probabilities @ weight
    exact = 1.7**2 * torch.einsum(
        "n,ni,nj->ij", probabilities, centered, centered
    )
    torch.testing.assert_close(sampled, exact, rtol=1e-12, atol=1e-12)


def test_empirical_fisher_factor_adds_trace_scaled_damping() -> None:
    gradients = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    factor = empirical_fisher_factor(gradients, damping_ratio=0.25)
    undamped = gradients.T @ gradients / 2
    expected = undamped + 0.25 * torch.diagonal(undamped).mean() * torch.eye(2)
    torch.testing.assert_close(factor, expected)


def test_sketch_a_updates_match_explicit_kronecker_contractions() -> None:
    generator = torch.Generator().manual_seed(5)
    inputs = torch.randn((7, 3), generator=generator, dtype=torch.float64)
    gradients = torch.randn((7, 2), generator=generator, dtype=torch.float64)
    input_factor = torch.randn((3, 3), generator=generator, dtype=torch.float64)
    input_factor = input_factor @ input_factor.T
    output_factor = torch.randn((2, 2), generator=generator, dtype=torch.float64)
    output_factor = output_factor @ output_factor.T

    fisher = torch.einsum(
        "ba,bi,bc,bj->aicj", gradients, inputs, gradients, inputs
    ) / float(inputs.shape[0])
    expected_input = torch.einsum("ac,aicj->ij", output_factor, fisher)
    expected_input /= output_factor.square().sum()
    expected_output = torch.einsum("ij,aicj->ac", input_factor, fisher)
    expected_output /= input_factor.square().sum()

    actual_input = sketch_a_input_factor_update(
        inputs, gradients, output_factor
    )
    actual_output = sketch_a_output_factor_update(
        inputs, gradients, input_factor
    )
    torch.testing.assert_close(actual_input, expected_input, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        actual_output, expected_output, rtol=1e-12, atol=1e-12
    )


def test_sketch_a_factor_scaling_preserves_the_kronecker_product() -> None:
    generator = torch.Generator().manual_seed(6)
    inputs = torch.randn((11, 3), generator=generator, dtype=torch.float64)
    gradients = torch.randn((11, 2), generator=generator, dtype=torch.float64)
    input_factor = torch.randn((3, 3), generator=generator, dtype=torch.float64)
    input_factor = input_factor @ input_factor.T

    output = sketch_a_output_factor_update(inputs, gradients, input_factor)
    scaled_output = sketch_a_output_factor_update(
        inputs, gradients, 7.0 * input_factor
    )
    torch.testing.assert_close(
        torch.kron(output, input_factor),
        torch.kron(scaled_output, 7.0 * input_factor),
        rtol=1e-12,
        atol=1e-12,
    )


def test_sketch_a_route_weight_can_enter_either_factor_once() -> None:
    generator = torch.Generator().manual_seed(7)
    inputs = torch.randn((5, 3), generator=generator, dtype=torch.float64)
    gradients = torch.randn((5, 2), generator=generator, dtype=torch.float64)
    route_weights = torch.rand(5, generator=generator, dtype=torch.float64)
    weighted_inputs = inputs * route_weights[:, None]
    weighted_gradients = gradients * route_weights[:, None]
    fisher_from_inputs = torch.einsum(
        "ba,bi,bc,bj->aicj",
        gradients,
        weighted_inputs,
        gradients,
        weighted_inputs,
    )
    fisher_from_gradients = torch.einsum(
        "ba,bi,bc,bj->aicj",
        weighted_gradients,
        inputs,
        weighted_gradients,
        inputs,
    )
    torch.testing.assert_close(fisher_from_inputs, fisher_from_gradients)

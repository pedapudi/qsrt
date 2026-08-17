from __future__ import annotations

import pytest
import torch

from qsrt.blockldlq_proof import (
    block_feedback_targets,
    block_ldl_reference,
    block_objective_from_feedback,
    capture_sample_selected,
    capture_validation_split,
    conditional_h2,
    congruence_metric,
    damp_hessian,
    explicit_weighted_output_error,
    permute_input_metric,
    quadratic_error,
    relative_error,
    weighted_gram,
)


def _positive_definite(dimension: int, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    samples = torch.randn((3 * dimension, dimension), generator=generator)
    return weighted_gram(samples, torch.ones(samples.shape[0])) + 0.1 * torch.eye(
        dimension, dtype=torch.float64
    )


def test_weighted_hessian_is_exact_linear_output_sse() -> None:
    generator = torch.Generator().manual_seed(2)
    rows = torch.randn((41, 19), generator=generator)
    weights = torch.rand(41, generator=generator).square()
    error = torch.randn((7, 19), generator=generator)
    hessian = weighted_gram(rows, weights)

    proxy = quadratic_error(error, hessian)
    direct = explicit_weighted_output_error(rows, weights, error)

    assert relative_error(proxy, direct) < 2e-15


def test_diagonal_damping_is_scaled_identity_regularization() -> None:
    hessian = _positive_definite(32)
    error = torch.randn((5, 32), generator=torch.Generator().manual_seed(3))
    sigma = 0.025
    damped = damp_hessian(hessian, sigma)
    expected_penalty = (
        sigma
        * torch.diagonal(hessian).mean()
        * error.to(torch.float64).square().sum()
    )

    actual_penalty = quadratic_error(error, damped) - quadratic_error(error, hessian)

    assert relative_error(actual_penalty, expected_penalty) < 2e-14


@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_block_ldl_reconstructs_hessian_and_feedback_objective(block_size: int) -> None:
    dimension = 32
    hessian = _positive_definite(dimension, seed=4)
    factors = block_ldl_reference(hessian, block_size)
    reconstructed = factors.lower @ factors.block_diagonal @ factors.lower.T
    assert relative_error(reconstructed, hessian) < 2e-14

    generator = torch.Generator().manual_seed(5)
    weight = torch.randn((dimension, 11), generator=generator)
    quantized = torch.randn((dimension, 11), generator=generator)
    error = weight.double() - quantized.double()
    targets = block_feedback_targets(weight, quantized, factors.lower, block_size)
    transformed = factors.lower.T @ error

    # The target residual seen by every reverse block is exactly its block of
    # L.T @ (W-Q), including arbitrary earlier/future quantization errors.
    torch.testing.assert_close(
        targets.double() - quantized.double(), transformed, rtol=1e-13, atol=1e-13
    )
    dense = torch.einsum("io,ij,jo->", error, hessian, error)
    decomposed = block_objective_from_feedback(error, factors)
    assert relative_error(decomposed, dense) < 3e-14


def test_w2_input_permutation_is_an_exact_congruence() -> None:
    generator = torch.Generator().manual_seed(6)
    dimension = 32
    error = torch.randn((13, dimension), generator=generator)
    hessian = _positive_definite(dimension, seed=7)
    permutation = torch.randperm(dimension, generator=generator)

    encoder_error, encoder_hessian = permute_input_metric(
        error, hessian, permutation
    )

    canonical = quadratic_error(error, hessian)
    encoded = torch.einsum(
        "io,ij,jo->",
        encoder_error.double(),
        encoder_hessian.double(),
        encoder_error.double(),
    )
    assert relative_error(encoded, canonical) < 2e-15


def test_input_conditioning_requires_hessian_congruence() -> None:
    generator = torch.Generator().manual_seed(8)
    dimension = 16
    hessian = _positive_definite(dimension, seed=9)
    work_error = torch.randn((dimension, 5), generator=generator).double()
    scales = torch.linspace(0.55, 1.45, dimension, dtype=torch.float64)
    decode_input = torch.diag(scales)
    canonical_error = decode_input @ work_error
    exact = torch.einsum("io,ij,jo->", canonical_error, hessian, canonical_error)
    transformed_h = congruence_metric(hessian, decode_input)
    work_metric = torch.einsum("io,ij,jo->", work_error, transformed_h, work_error)

    assert relative_error(work_metric, exact) < 2e-15
    # Omitting the congruence is not an equivalent objective.  This assertion
    # makes the production-conditioning audit an explicit gate.
    omitted = torch.einsum("io,ij,jo->", work_error, hessian, work_error)
    assert relative_error(omitted, exact) > 1e-2


def test_candidate_conditioned_h2_is_exact_down_projection_sse() -> None:
    generator = torch.Generator().manual_seed(10)
    rows = torch.randn((53, 17), generator=generator)
    gates = torch.rand(53, generator=generator)
    gate_weight = torch.randn((23, 17), generator=generator)
    up_weight = torch.randn((23, 17), generator=generator)
    down_error = torch.randn((9, 23), generator=generator)

    middle, h2 = conditional_h2(rows, gates, gate_weight, up_weight)
    proxy = quadratic_error(down_error, h2)
    direct = explicit_weighted_output_error(middle, gates.square(), down_error)

    assert relative_error(proxy, direct) < 3e-15


def test_candidate_conditioned_h2_changes_with_decoded_upstream() -> None:
    generator = torch.Generator().manual_seed(11)
    rows = torch.randn((71, 13), generator=generator)
    gates = torch.rand(71, generator=generator)
    gate_weight = torch.randn((29, 13), generator=generator)
    up_weight = torch.randn((29, 13), generator=generator)
    _, baseline = conditional_h2(rows, gates, gate_weight, up_weight)
    perturbed_gate = gate_weight.clone()
    perturbed_gate[0].mul_(1.5)
    _, candidate = conditional_h2(rows, gates, perturbed_gate, up_weight)

    assert relative_error(candidate, baseline) > 1e-4


def test_capture_hash_oracles_match_fixed_vectors() -> None:
    # Values were independently evaluated from the Triton unsigned operations
    # in b12x.moe.calibration.  Keeping fixed observations here catches signed
    # overflow or accidental changes to the capture's statistical contract.
    cases = {
        0x0000000100000000: (False, 0),
        0x0000000100000001: (True, 0),
        0x000000070000002A: (False, 0),
        0x000004D200000011: (False, 0),
    }
    for observation, (selected, split) in cases.items():
        assert capture_sample_selected(observation, 64) is selected
        assert capture_validation_split(observation, 16) == split

    selected_observation = 0x0000000200000002
    validation_observation = 0x0000000100000008
    assert capture_sample_selected(selected_observation, 64)
    assert capture_validation_split(validation_observation, 16) == 1

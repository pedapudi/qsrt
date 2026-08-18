from __future__ import annotations

import pytest
import torch

from qsrt.coupled_expert_study import RoutedOutputMetric
from qsrt.distortion_transfer import (
    mapped_output_hessian,
    quadratic_matrix_sse,
    routed_error_geometry,
    two_sided_encoder_sse,
)


def test_quadratic_matrix_sse_matches_diagonal_weighting() -> None:
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    candidate = source + torch.tensor([[1.0, -1.0], [2.0, 0.5]])
    covariance = torch.diag(torch.tensor([2.0, 3.0]))
    numerator, denominator = quadratic_matrix_sse(source, candidate, covariance)
    assert numerator == pytest.approx(13.75)
    assert denominator == pytest.approx(80.0)


def test_routed_error_geometry_separates_radial_and_tangential_error() -> None:
    aggregate = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    error = torch.tensor([[1.0, 4.0], [5.0, 2.0]])
    metric = RoutedOutputMetric(torch.ones(2), torch.eye(2), epsilon=1e-5)
    result = routed_error_geometry(aggregate, error, metric)
    assert result.radial_sse == pytest.approx(5.0)
    assert result.tangential_sse == pytest.approx(41.0)
    assert result.residual_sse == pytest.approx(46.0)
    assert result.mapped_exact_sse > 0.0


def test_two_sided_encoder_sse_matches_explicit_kronecker_metric() -> None:
    generator = torch.Generator().manual_seed(3)
    source = torch.randn((5, 4), generator=generator)
    candidate = source + torch.randn((5, 4), generator=generator) * 0.1
    input_root = torch.randn((8, 5), generator=generator)
    output_root = torch.randn((7, 4), generator=generator)
    input_hessian = input_root.T @ input_root + 0.2 * torch.eye(5)
    output_hessian = output_root.T @ output_root + 0.2 * torch.eye(4)

    numerator, denominator = two_sided_encoder_sse(
        source, candidate, input_hessian, output_hessian
    )
    error = (candidate - source).double()
    expected_numerator = torch.trace(
        output_hessian.double() @ error.T @ input_hessian.double() @ error
    )
    expected_denominator = torch.trace(
        output_hessian.double()
        @ source.double().T
        @ input_hessian.double()
        @ source.double()
    )
    assert numerator == pytest.approx(float(expected_numerator), rel=1e-12)
    assert denominator == pytest.approx(float(expected_denominator), rel=1e-12)


def test_mapped_output_hessian_closes_against_explicit_jacobian_vectors() -> None:
    generator = torch.Generator().manual_seed(4)
    aggregate = torch.randn((19, 6), generator=generator)
    gain = torch.randn(6, generator=generator)
    projection = torch.randn((9, 6), generator=generator)
    vector = torch.randn(6, generator=generator)
    weights = torch.rand(19, generator=generator)
    metric = RoutedOutputMetric(gain, projection, epsilon=2e-5)

    hessian = mapped_output_hessian(
        aggregate,
        metric,
        row_weights=weights,
        chunk_rows=5,
    )
    vectors = vector.expand_as(aggregate)
    mapped = metric.jacobian_vectors(aggregate, vectors)
    expected = (weights.double() * mapped.double().square().sum(dim=1)).sum()
    expected /= weights.double().sum()
    actual = vector.double() @ hessian @ vector.double()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-8)


def test_mapped_output_hessian_is_symmetric_positive_semidefinite() -> None:
    generator = torch.Generator().manual_seed(5)
    metric = RoutedOutputMetric(
        torch.randn(7, generator=generator),
        torch.randn((11, 7), generator=generator),
    )
    aggregate = torch.randn((31, 7), generator=generator)
    hessian = mapped_output_hessian(aggregate, metric, chunk_rows=8)

    torch.testing.assert_close(hessian, hessian.T, rtol=0, atol=1e-12)
    assert float(torch.linalg.eigvalsh(hessian).min()) >= -1e-10

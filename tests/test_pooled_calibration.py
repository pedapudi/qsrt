import pytest
import torch

from qsrt.pooled_calibration import (
    blockwise_upstream_conditioning_coefficients,
    CandidateHiddenStatistics,
    candidate_h2,
    collect_coupled_hidden_statistics,
    collect_upstream_functional_statistics,
    decoded_down_sse,
    decoded_down_sse_difference,
    evaluate_coupled_candidate_portfolio_batches,
    evaluate_coupled_expert_batches,
    ridge_refit_down_from_statistics,
)
from qsrt.coupled_expert_study import CoupledTriplet
from qsrt.qsrt_coupled import (
    CoupledHadamardExecution,
    CoupledHadamardSpec,
    encode_coupled_weights,
)
from qsrt.tp_simulator import situ


def _samples() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7819)
    source = torch.randn(37, 7, generator=generator, dtype=torch.float64)
    candidate = source + 0.08 * torch.randn(
        37, 7, generator=generator, dtype=torch.float64
    )
    gates = torch.rand(37, generator=generator, dtype=torch.float64)
    down = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    return source, candidate, gates, down


def test_pooled_down_sse_matches_explicit_weighted_outputs() -> None:
    source, candidate, gates, down = _samples()
    decoded = down + 0.04
    stats = CandidateHiddenStatistics.zeros(7)
    stats.update(candidate, source, gates)

    actual = decoded_down_sse(stats, decoded, down)
    explicit = (
        gates[:, None]
        * (candidate @ decoded - source @ down)
    ).square().sum()

    torch.testing.assert_close(actual, explicit, rtol=2e-12, atol=2e-12)


def test_chunked_statistics_match_one_pass() -> None:
    source, candidate, gates, _ = _samples()
    full = CandidateHiddenStatistics.zeros(7)
    full.update(candidate, source, gates)
    chunked = CandidateHiddenStatistics.zeros(7)
    for begin, end in ((0, 5), (5, 19), (19, 37)):
        part = CandidateHiddenStatistics.zeros(7)
        part.update(candidate[begin:end], source[begin:end], gates[begin:end])
        chunked.merge_(part)

    torch.testing.assert_close(chunked.candidate_gram, full.candidate_gram)
    torch.testing.assert_close(
        chunked.candidate_source_cross, full.candidate_source_cross
    )
    torch.testing.assert_close(chunked.source_gram, full.source_gram)
    torch.testing.assert_close(
        chunked.candidate_residual_cross,
        full.candidate_residual_cross,
    )
    torch.testing.assert_close(
        chunked.hidden_residual_gram,
        full.hidden_residual_gram,
    )
    assert chunked.weight_sum == pytest.approx(full.weight_sum)
    assert chunked.weight_square_sum == pytest.approx(full.weight_square_sum)
    assert chunked.rows == full.rows


def test_candidate_h2_uses_only_candidate_gram_and_scaled_identity() -> None:
    source, candidate, gates, _ = _samples()
    stats = CandidateHiddenStatistics.zeros(7)
    stats.update(candidate, source, gates)
    h2, evidence = candidate_h2(stats)

    local = stats.candidate_gram / stats.weight_sum
    identity = torch.eye(7, dtype=torch.float64) * torch.trace(local) / 7
    expected = torch.lerp(identity, local, evidence["local_alpha"])
    torch.testing.assert_close(h2, expected)
    torch.testing.assert_close(torch.trace(h2), torch.trace(local))


def test_candidate_h2_oas_matches_the_closed_form_estimator() -> None:
    generator = torch.Generator().manual_seed(9173)
    candidate = torch.randn(4096, 11, generator=generator, dtype=torch.float64)
    gates = torch.ones(4096, dtype=torch.float64)
    stats = CandidateHiddenStatistics.zeros(11)
    stats.update(candidate, candidate, gates)
    _h2, evidence = candidate_h2(stats, max_local_alpha=1.0)

    covariance = stats.candidate_gram / stats.weight_sum
    trace_square = torch.trace(covariance).square()
    frobenius_square = covariance.square().sum()
    dimension = covariance.shape[0]
    expected = min(
        1.0,
        float(
            ((1.0 - 2.0 / dimension) * frobenius_square + trace_square)
            / (
                (stats.effective_sample_size + 1.0 - 2.0 / dimension)
                * (frobenius_square - trace_square / dimension)
            )
        ),
    )
    assert evidence["oas_shrinkage"] == pytest.approx(expected)


def test_statistics_ridge_refit_improves_the_pooled_objective() -> None:
    source, candidate, gates, down = _samples()
    stats = CandidateHiddenStatistics.zeros(7)
    stats.update(candidate, source, gates)
    refit, evidence = ridge_refit_down_from_statistics(
        stats,
        down,
        regularization_ratio=1e-5,
    )

    assert evidence["regularization"] > 0
    assert decoded_down_sse(stats, refit, down) < decoded_down_sse(
        stats, down, down
    )
    sparse = CandidateHiddenStatistics.zeros(7, retain_source_gram=False)
    sparse.update(candidate, source, gates)
    difference = decoded_down_sse_difference(sparse, refit, down, down)
    explicit_difference = decoded_down_sse(stats, refit, down) - decoded_down_sse(
        stats, down, down
    )
    torch.testing.assert_close(difference, explicit_difference)


def test_decoded_down_sse_avoids_absolute_energy_cancellation() -> None:
    generator = torch.Generator().manual_seed(13084)
    source = 16.0 * torch.randn(2048, 64, generator=generator)
    candidate = source + 2e-3 * torch.randn(
        2048, 64, generator=generator
    )
    gates = torch.rand(2048, generator=generator)
    source_down = 8.0 * torch.randn(64, 48, generator=generator)
    candidate_down = source_down + 1e-3 * torch.randn(
        64, 48, generator=generator
    )
    statistics = CandidateHiddenStatistics.zeros(64, dtype=torch.float32)
    statistics.update(candidate, source, gates)

    actual = decoded_down_sse(statistics, candidate_down, source_down)
    explicit = (
        gates[:, None]
        * (candidate @ candidate_down - source @ source_down)
    ).square().sum(dtype=torch.float64)

    torch.testing.assert_close(
        actual.double(),
        explicit,
        rtol=2e-4,
        atol=2e-4,
    )


def test_decoded_down_sse_difference_avoids_cross_gram_cancellation() -> None:
    generator = torch.Generator().manual_seed(11839)
    source = 16.0 * torch.randn(2048, 64, generator=generator)
    hidden = source + 2e-3 * torch.randn(2048, 64, generator=generator)
    gates = torch.rand(2048, generator=generator)
    source_down = 8.0 * torch.randn(64, 48, generator=generator)
    baseline_down = source_down + 2e-3 * torch.randn(
        64, 48, generator=generator
    )
    candidate_down = baseline_down + 1e-3 * torch.randn(
        64, 48, generator=generator
    )
    statistics = CandidateHiddenStatistics.zeros(
        64,
        dtype=torch.float32,
        retain_source_gram=False,
    )
    statistics.update(hidden, source, gates)

    actual = decoded_down_sse_difference(
        statistics,
        candidate_down,
        baseline_down,
        source_down,
    )
    candidate_error = hidden @ candidate_down - source @ source_down
    baseline_error = hidden @ baseline_down - source @ source_down
    explicit = (
        gates[:, None].square()
        * (candidate_error.square() - baseline_error.square())
    ).sum(dtype=torch.float64)

    torch.testing.assert_close(
        actual.double(),
        explicit,
        rtol=2e-4,
        atol=2e-4,
    )


def test_pooled_coupled_evaluation_matches_explicit_routed_function() -> None:
    generator = torch.Generator().manual_seed(4051)
    source = CoupledTriplet(
        torch.randn(3, 4, generator=generator),
        torch.randn(3, 4, generator=generator),
        torch.randn(4, 3, generator=generator),
    )
    spec = CoupledHadamardSpec(
        residual_block_size=2,
        preactivation_block_size=2,
        postactivation_block_size=1,
        intermediate_draw=3,
    )
    execution = CoupledHadamardExecution(hidden=4, intermediate=3, spec=spec)
    encoded_source = CoupledTriplet(*encode_coupled_weights(source.tensors(), spec))
    candidate = CoupledTriplet(
        encoded_source.gate + 0.015,
        encoded_source.up - 0.02,
        encoded_source.down + 0.01,
    )
    inputs = torch.randn(11, 4, generator=generator)
    gates = torch.rand(11, generator=generator)
    row_indices = torch.tensor([0, 2, 3, 7, 8, 11, 13, 18, 20, 22, 29])
    batches = [
        {
            "input": inputs[:6],
            "route_weight": gates[:6],
            "row_index": row_indices[:6],
        },
        {
            "input": inputs[6:],
            "route_weight": gates[6:],
            "row_index": row_indices[6:],
        },
    ]

    result = evaluate_coupled_expert_batches(
        batches,
        source=source,
        candidate_coordinates=candidate,
        execution=execution,
        prefix_row_limits=(10, 21, 30),
        statistics_dtype=torch.float64,
        retain_source_gram=True,
    )
    source_output = situ(inputs @ source.gate.T, inputs @ source.up.T) @ source.down.T
    candidate_output = execution.execute(inputs, candidate.tensors())
    explicit_sse = ((candidate_output - source_output) * gates[:, None]).square().sum()
    explicit_energy = (source_output * gates[:, None]).square().sum()

    assert result.routed_occurrences == 11
    assert result.sse == pytest.approx(float(explicit_sse), rel=2e-6)
    assert result.source_energy == pytest.approx(float(explicit_energy), rel=2e-6)
    assert result.prefix_scores[10][2] == 5
    assert result.prefix_scores[21][2] == 9
    assert result.prefix_scores[30][2] == 11
    torch.testing.assert_close(
        decoded_down_sse(
            result.statistics,
            candidate.down.T,
            encoded_source.down.T,
        ),
        ((
            execution.decode_middle(
                execution.transform_inputs(inputs), candidate.gate, candidate.up
            )
            @ candidate.down.T
            - execution.decode_middle(
                execution.transform_inputs(inputs),
                encoded_source.gate,
                encoded_source.up,
            )
            @ encoded_source.down.T
        ) * gates[:, None]).square().sum(),
        rtol=2e-6,
        atol=2e-6,
        check_dtype=False,
    )

    hidden_only = collect_coupled_hidden_statistics(
        batches,
        source=source,
        candidate_coordinates=candidate,
        execution=execution,
        statistics_dtype=torch.float64,
        retain_source_gram=True,
    )
    torch.testing.assert_close(
        hidden_only.candidate_gram,
        result.statistics.candidate_gram,
    )
    torch.testing.assert_close(
        hidden_only.candidate_source_cross,
        result.statistics.candidate_source_cross,
    )
    torch.testing.assert_close(
        hidden_only.source_gram,
        result.statistics.source_gram,
    )
    torch.testing.assert_close(
        hidden_only.candidate_residual_cross,
        result.statistics.candidate_residual_cross,
    )
    torch.testing.assert_close(
        hidden_only.hidden_residual_gram,
        result.statistics.hidden_residual_gram,
    )
    assert hidden_only.rows == result.statistics.rows
    assert hidden_only.weight_sum == pytest.approx(result.statistics.weight_sum)


def test_pooled_portfolio_matches_independent_candidate_scores() -> None:
    generator = torch.Generator().manual_seed(1987)
    source = CoupledTriplet(
        torch.randn(3, 4, generator=generator),
        torch.randn(3, 4, generator=generator),
        torch.randn(4, 3, generator=generator),
    )
    inputs = torch.randn(13, 4, generator=generator)
    route_weights = torch.rand(13, generator=generator)
    row_indices = torch.arange(2, 28, 2, dtype=torch.int64)
    batches = [
        {
            "input": inputs[:7],
            "route_weight": route_weights[:7],
            "row_index": row_indices[:7],
        },
        {
            "input": inputs[7:],
            "route_weight": route_weights[7:],
            "row_index": row_indices[7:],
        },
    ]
    candidates = {}
    executions = {}
    for draw, offset in ((0, 0.01), (3, -0.015)):
        spec = CoupledHadamardSpec(
            residual_block_size=2,
            preactivation_block_size=2,
            postactivation_block_size=1,
            intermediate_draw=draw,
        )
        encoded = CoupledTriplet(*encode_coupled_weights(source.tensors(), spec))
        candidates[str(draw)] = CoupledTriplet(
            encoded.gate + offset,
            encoded.up - 0.5 * offset,
            encoded.down + 0.25 * offset,
        )
        executions[str(draw)] = CoupledHadamardExecution(4, 3, spec)

    portfolio = evaluate_coupled_candidate_portfolio_batches(
        iter(batches),
        source=source,
        candidate_coordinates=candidates,
        executions=executions,
    )
    for name, candidate in candidates.items():
        independent = evaluate_coupled_expert_batches(
            iter(batches),
            source=source,
            candidate_coordinates=candidate,
            execution=executions[name],
        )
        assert portfolio.candidate_sse[name] == pytest.approx(
            independent.sse, rel=2e-6
        )
        assert portfolio.source_energy == pytest.approx(
            independent.source_energy, rel=2e-6
        )
        assert portfolio.routed_occurrences == independent.routed_occurrences


def test_upstream_functional_statistics_match_explicit_derivative_outer_products() -> None:
    generator = torch.Generator().manual_seed(903)
    source = CoupledTriplet(
        torch.randn(3, 4, generator=generator),
        torch.randn(3, 4, generator=generator),
        torch.randn(4, 3, generator=generator),
    )
    inputs = torch.randn(9, 4, generator=generator)
    route_weights = torch.rand(9, generator=generator)
    batches = [
        {
            "input": inputs,
            "route_weight": route_weights,
            "row_index": torch.arange(9, dtype=torch.int64),
        }
    ]
    statistics = collect_upstream_functional_statistics(
        batches, source=source, device="cpu"
    )
    gate = inputs @ source.gate.T
    up = inputs @ source.up.T
    from qsrt.coupled_expert_study import situ_derivatives

    d_gate, d_up = situ_derivatives(gate, up)
    derivatives = torch.stack((d_gate, d_up), dim=2).double()
    weights = route_weights.double().square()
    down_energy = source.down.double().square().sum(dim=0)
    expected = torch.einsum(
        "r,rni,rnj,n->nij", weights, derivatives, derivatives, down_energy
    )
    torch.testing.assert_close(statistics.derivative_metric, expected)
    torch.testing.assert_close(
        statistics.hidden_energy,
        (situ(gate, up).double().square() * weights[:, None]).sum(dim=0),
    )
    assert statistics.rows == 9
    assert statistics.effective_sample_size > 0


def test_blockwise_upstream_conditioning_coefficients_sum_metrics_before_ratio() -> None:
    metric = torch.tensor(
        [
            [[4.0, 2.0], [2.0, 9.0]],
            [[1.0, -0.5], [-0.5, 1.0]],
            [[2.0, 1.0], [1.0, 8.0]],
            [[6.0, 3.0], [3.0, 4.0]],
        ],
        dtype=torch.float64,
    )
    gate_from_up, up_from_gate, evidence = (
        blockwise_upstream_conditioning_coefficients(metric, block_size=2)
    )
    torch.testing.assert_close(
        gate_from_up,
        torch.tensor((0.3, 0.3, 0.5, 0.5), dtype=torch.float64),
    )
    torch.testing.assert_close(
        up_from_gate,
        torch.tensor((0.15, 0.15, 1.0 / 3.0, 1.0 / 3.0), dtype=torch.float64),
    )
    assert evidence["block_size"] == 2


def test_blockwise_upstream_conditioning_coefficients_reject_non_psd_metric() -> None:
    metric = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        blockwise_upstream_conditioning_coefficients(metric, block_size=1)

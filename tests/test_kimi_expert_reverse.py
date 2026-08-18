from __future__ import annotations

import torch

from qsrt.kimi_expert_reverse import (
    CoupledDownReverseAccumulator,
    CoupledUpstreamReverseAccumulator,
    CoupledUpstreamReverseConfig,
    ExpertReverseChannelRouter,
)
from qsrt.qsrt_coupled import CoupledHadamardSpec


def _accumulator(rank: int = 4) -> CoupledUpstreamReverseAccumulator:
    return CoupledUpstreamReverseAccumulator(
        CoupledUpstreamReverseConfig(
            num_experts=2,
            hidden_dimension=8,
            intermediate_dimension=4,
            spec=CoupledHadamardSpec(
                residual_block_size=4,
                preactivation_block_size=4,
                postactivation_block_size=4,
                intermediate_draw=3,
            ),
            output_factor_block_size=4,
            gradient_sketch_rank=rank,
            gradient_sketch_seed=77,
        ),
        device="cpu",
    )


def test_upstream_reverse_accumulates_coupled_fisher_blocks() -> None:
    generator = torch.Generator().manual_seed(42)
    accumulator = _accumulator()
    inputs = torch.randn((5, 8), generator=generator)
    gradients = torch.randn((5, 8), generator=generator)
    accumulator.add_fisher(1, inputs, gradients)

    transformed = accumulator.execution_for_expert(
        1
    ).transform_preactivation_gradients(gradients)
    blocked = transformed.reshape(5, 2, 4)
    expected = torch.einsum("nbi,nbj->bij", blocked, blocked)
    torch.testing.assert_close(accumulator.output_factor_sums[1], expected)
    assert accumulator.output_factor_rows.tolist() == [0, 5]


def test_upstream_reverse_sketch_recovers_low_rank_gradient() -> None:
    generator = torch.Generator().manual_seed(108)
    accumulator = _accumulator(rank=4)
    inputs = torch.randn((3, 8), generator=generator)
    gradients = torch.randn((3, 8), generator=generator)
    accumulator.add_objective_gradient(0, inputs, gradients)

    execution = accumulator.execution_for_expert(0)
    transformed_inputs = execution.transform_inputs(inputs)
    transformed_gradients = execution.transform_preactivation_gradients(gradients)
    expected = transformed_gradients.T @ transformed_inputs
    observed = accumulator.reconstructed_gradient(0)
    torch.testing.assert_close(observed, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(
        accumulator.gradient_tile(0, 0, 0, tile_size=4),
        expected[:4, :4],
        rtol=2e-4,
        atol=2e-4,
    )
    assert accumulator.gradient_rows.tolist() == [3, 0]


def test_down_reverse_sketch_recovers_low_rank_gradient() -> None:
    generator = torch.Generator().manual_seed(109)
    config = _accumulator(rank=4).config
    accumulator = CoupledDownReverseAccumulator(config, device="cpu")
    postactivation = torch.randn((3, 4), generator=generator)
    gradients = torch.randn((3, 8), generator=generator)
    accumulator.add_objective_gradient(1, postactivation, gradients)

    execution = accumulator.execution_for_expert(1)
    transformed_inputs = execution.transform_postactivation_rows(postactivation)
    transformed_gradients = execution.transform_expert_output_gradients(gradients)
    expected = transformed_gradients.T @ transformed_inputs
    torch.testing.assert_close(
        accumulator.reconstructed_gradient(1),
        expected,
        rtol=2e-4,
        atol=2e-4,
    )
    torch.testing.assert_close(
        accumulator.gradient_tile(1, 0, 0, tile_size=4),
        expected[:4, :4],
        rtol=2e-4,
        atol=2e-4,
    )
    assert accumulator.gradient_rows.tolist() == [0, 3]


def test_reverse_channel_router_separates_fisher_and_objective_vjps() -> None:
    upstream = _accumulator(rank=4)
    down = CoupledDownReverseAccumulator(upstream.config, device="cpu")
    routed: list[torch.Tensor] = []
    router = ExpertReverseChannelRouter(
        upstream=upstream,
        down=down,
        routed_fisher_add=lambda value: routed.append(value.clone()),
    )
    generator = torch.Generator().manual_seed(110)
    inputs = torch.randn((3, 8), generator=generator)
    gate_up = torch.randn((3, 8), generator=generator)
    postactivation = torch.randn((3, 4), generator=generator)
    output = torch.randn((3, 8), generator=generator)

    router.select_channel("fisher")
    router.routed_output(output)
    router.expert_preactivation(0, inputs, gate_up)
    router.expert_output(0, postactivation, output)
    assert upstream.output_factor_rows.tolist() == [3, 0]
    assert upstream.gradient_rows.tolist() == [0, 0]
    assert down.gradient_rows.tolist() == [0, 0]
    assert len(routed) == 1

    router.select_channel("objective")
    router.routed_output(output)
    router.expert_preactivation(0, inputs, gate_up)
    router.expert_output(0, postactivation, output)
    assert upstream.output_factor_rows.tolist() == [3, 0]
    assert upstream.gradient_rows.tolist() == [3, 0]
    assert down.gradient_rows.tolist() == [3, 0]
    assert len(routed) == 1


def test_upstream_reverse_uses_each_experts_frozen_draw() -> None:
    config = CoupledUpstreamReverseConfig(
        num_experts=2,
        hidden_dimension=8,
        intermediate_dimension=4,
        spec=CoupledHadamardSpec(
            residual_block_size=4,
            preactivation_block_size=4,
            postactivation_block_size=4,
        ),
        intermediate_draws=(0, 6),
        output_factor_block_size=4,
        gradient_sketch_rank=4,
    )
    accumulator = CoupledUpstreamReverseAccumulator(
        config,
        device="cpu",
        capture_objective_gradient=False,
    )
    inputs = torch.ones((2, 8))
    gradients = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    accumulator.add_fisher(0, inputs, gradients)
    accumulator.add_fisher(1, inputs, gradients)

    expected = []
    for expert in (0, 1):
        transformed = accumulator.execution_for_expert(
            expert
        ).transform_preactivation_gradients(gradients)
        blocked = transformed.reshape(2, 2, 4)
        expected.append(torch.einsum("nbi,nbj->bij", blocked, blocked))
    torch.testing.assert_close(accumulator.output_factor_sums, torch.stack(expected))
    assert config.spec_for_expert(0).intermediate_draw == 0
    assert config.spec_for_expert(1).intermediate_draw == 6


def test_reverse_channel_router_can_capture_upstream_only() -> None:
    upstream = _accumulator(rank=4)
    router = ExpertReverseChannelRouter(upstream=upstream)
    inputs = torch.ones((2, 8))
    gradients = torch.ones((2, 8))
    router.select_channel("fisher")
    router.expert_preactivation(0, inputs, gradients)
    router.expert_output(0, torch.ones((2, 4)), gradients)
    assert upstream.output_factor_rows.tolist() == [2, 0]


@torch.no_grad()
def test_grouped_cuda_fisher_matches_individual_expert_accumulation() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda", 0)
    config = CoupledUpstreamReverseConfig(
        num_experts=8,
        hidden_dimension=128,
        intermediate_dimension=128,
        spec=CoupledHadamardSpec(
            residual_block_size=128,
            preactivation_block_size=128,
            postactivation_block_size=128,
        ),
        intermediate_draws=tuple(range(8)),
        output_factor_block_size=128,
        gradient_sketch_rank=8,
    )
    grouped = CoupledUpstreamReverseAccumulator(
        config,
        device=device,
        capture_objective_gradient=False,
    )
    individual = CoupledUpstreamReverseAccumulator(
        config,
        device=device,
        capture_objective_gradient=False,
    )
    generator = torch.Generator(device="cpu").manual_seed(113)
    expert_ids = torch.randint(8, (96,), generator=generator).sort().values.to(device)
    offsets = (
        torch.bincount(expert_ids, minlength=8)
        .to(torch.int32)
        .cumsum(0, dtype=torch.int32)
    )
    inputs = torch.randn(
        (96, 128), generator=generator, dtype=torch.bfloat16
    ).to(device)
    gradients = torch.randn(
        (96, 256), generator=generator, dtype=torch.bfloat16
    ).to(device)

    grouped.add_grouped_fisher(expert_ids, offsets, inputs, gradients)
    begin = 0
    for expert, end in enumerate(offsets.cpu().tolist()):
        if end > begin:
            individual.add_fisher(
                expert,
                inputs[begin:end],
                gradients[begin:end],
            )
        begin = end

    assert torch.equal(grouped.output_factor_sums, individual.output_factor_sums)
    assert torch.equal(grouped.output_factor_rows, individual.output_factor_rows)

from __future__ import annotations

import json

import numpy as np
import torch

from qsrt.kimi_dense_objective_gradients import (
    DenseObjectiveGradientAccumulator,
    DenseObjectiveGradientArchive,
    DenseObjectiveGradientConfig,
)
from qsrt.qsrt_coupled import CoupledHadamardSpec


def _config() -> DenseObjectiveGradientConfig:
    return DenseObjectiveGradientConfig(
        num_experts=4,
        expert_begin=1,
        expert_end=3,
        hidden_dimension=8,
        intermediate_dimension=4,
        intermediate_draws=(0, 1, 2, 0),
        spec=CoupledHadamardSpec(
            residual_block_size=4,
            preactivation_block_size=4,
            postactivation_block_size=4,
        ),
    )


def test_dense_gradient_accumulator_uses_coupled_coordinates() -> None:
    config = _config()
    accumulator = DenseObjectiveGradientAccumulator(config, device="cpu")
    generator = torch.Generator().manual_seed(17)
    inputs = torch.randn(5, 8, generator=generator)
    gate_up_gradient = torch.randn(5, 8, generator=generator)
    postactivation = torch.randn(5, 4, generator=generator)
    output_gradient = torch.randn(5, 8, generator=generator)

    accumulator.add_upstream(1, inputs, gate_up_gradient)
    accumulator.add_down(1, postactivation, output_gradient)

    execution = accumulator.execution_for_expert(1)
    expected_upstream = (
        execution.transform_preactivation_gradients(gate_up_gradient).T
        @ execution.transform_inputs(inputs)
    )
    expected_down = (
        execution.transform_expert_output_gradients(output_gradient).T
        @ execution.transform_postactivation_rows(postactivation)
    )
    torch.testing.assert_close(accumulator.upstream[0], expected_upstream)
    torch.testing.assert_close(accumulator.down[0], expected_down)
    assert accumulator.upstream_rows.tolist() == [5, 0]
    assert accumulator.down_rows.tolist() == [5, 0]


def test_grouped_accumulation_filters_exact_expert_shard() -> None:
    config = _config()
    accumulator = DenseObjectiveGradientAccumulator(config, device="cpu")
    generator = torch.Generator().manual_seed(23)
    counts = (2, 3, 1, 2)
    experts = torch.repeat_interleave(
        torch.arange(4, dtype=torch.int64),
        torch.tensor(counts),
    )
    offsets = torch.tensor(counts, dtype=torch.int32).cumsum(0)
    inputs = torch.randn(8, 8, generator=generator)
    gate_up_gradient = torch.randn(8, 8, generator=generator)
    postactivation = torch.randn(8, 4, generator=generator)
    output_gradient = torch.randn(8, 8, generator=generator)

    accumulator.add_grouped_upstream(
        experts,
        offsets,
        inputs,
        gate_up_gradient,
    )
    accumulator.add_grouped_down(
        experts,
        offsets,
        postactivation,
        output_gradient,
    )

    assert accumulator.upstream_rows.tolist() == [3, 1]
    assert accumulator.down_rows.tolist() == [3, 1]
    for local, expert in enumerate((1, 2)):
        begin = int(offsets[expert - 1])
        end = int(offsets[expert])
        execution = accumulator.execution_for_expert(expert)
        expected = (
            execution.transform_preactivation_gradients(
                gate_up_gradient[begin:end]
            ).T
            @ execution.transform_inputs(inputs[begin:end])
        )
        torch.testing.assert_close(accumulator.upstream[local], expected)


def test_dense_gradient_archive_round_trip(tmp_path) -> None:
    config = _config()
    accumulator = DenseObjectiveGradientAccumulator(config, device="cpu")
    accumulator.upstream.copy_(
        torch.arange(accumulator.upstream.numel(), dtype=torch.float32).reshape_as(
            accumulator.upstream
        )
    )
    accumulator.down.fill_(3.5)
    accumulator.upstream_rows.copy_(torch.tensor((7, 9)))
    accumulator.down_rows.copy_(torch.tensor((5, 8)))
    archive = DenseObjectiveGradientArchive.create(
        tmp_path / "gradients",
        layers=(89,),
        num_experts=4,
        shard_experts=2,
        hidden_dimension=8,
        intermediate_dimension=4,
        objective_normalizer=100.0,
        provenance={"objective": "KL(reference || candidate)"},
    )
    archive.write_shard(layer=89, accumulator=accumulator)

    upstream = np.memmap(
        archive.root
        / "layer-089/experts-0001-0003/upstream.f32",
        mode="r",
        dtype=np.float32,
        shape=tuple(accumulator.upstream.shape),
    )
    np.testing.assert_array_equal(upstream, accumulator.upstream.numpy())
    receipt = json.loads(
        (
            archive.root
            / "layer-089/experts-0001-0003/receipt.json"
        ).read_text()
    )
    assert receipt["files"]["down_rows"]["bytes"] == 16
    assert receipt["sum_squares"]["upstream"] == float(
        accumulator.upstream.square().sum(dtype=torch.float64)
    )
    assert receipt["sum_squares"]["down"] == float(
        accumulator.down.square().sum(dtype=torch.float64)
    )

from __future__ import annotations

import torch

from qsrt.coupled_expert_study import (
    CoupledTriplet,
    encode_coupled_block_hadamard,
    execute_coupled_block_hadamard,
)
from qsrt.qsrt_coupled import (
    CoupledHadamardSpec,
    block_hadamard,
    coupled_execution,
    encode_coupled_weights,
    rotation_signs,
    signed_block_hadamard,
)
from qsrt.tp_simulator import situ


def test_block_hadamard_preserves_empty_routed_batches() -> None:
    values = torch.empty((0, 7, 16), dtype=torch.float16)
    transformed = block_hadamard(values, block_size=8, dim=2)

    assert transformed.shape == values.shape
    assert transformed.dtype == torch.float32
    assert transformed.numel() == 0


def test_production_coupled_transform_closes_and_matches_research_oracle() -> None:
    generator = torch.Generator().manual_seed(871)
    rows, hidden, intermediate = 9, 16, 12
    weights = (
        torch.randn(intermediate, hidden, generator=generator),
        torch.randn(intermediate, hidden, generator=generator),
        torch.randn(hidden, intermediate, generator=generator),
    )
    inputs = torch.randn(rows, hidden, generator=generator)
    spec = CoupledHadamardSpec(
        residual_block_size=8,
        preactivation_block_size=8,
        postactivation_block_size=4,
        intermediate_draw=3,
    )

    encoded = encode_coupled_weights(weights, spec)
    execution = coupled_execution(encoded, spec)
    output = execution.execute(inputs, encoded)
    reference = torch.nn.functional.linear(
        situ(
            torch.nn.functional.linear(inputs, weights[0]),
            torch.nn.functional.linear(inputs, weights[1]),
        ),
        weights[2],
    )
    research_encoded = encode_coupled_block_hadamard(
        CoupledTriplet(*weights),
        block_size=8,
        preactivation_block_size=8,
        postactivation_block_size=4,
        intermediate_rotation_draw=3,
    )
    research_output = execute_coupled_block_hadamard(
        inputs,
        research_encoded,
        block_size=8,
        preactivation_block_size=8,
        postactivation_block_size=4,
        intermediate_rotation_draw=3,
    )

    for actual, expected in zip(encoded, research_encoded.tensors(), strict=True):
        assert torch.equal(actual, expected)
    assert torch.allclose(output, reference, rtol=2e-5, atol=2e-5)
    assert torch.allclose(output, research_output, rtol=2e-5, atol=2e-5)


def test_coupled_hessian_transforms_preserve_quadratic_forms() -> None:
    generator = torch.Generator().manual_seed(91)
    hidden, intermediate = 16, 12
    weights = (
        torch.randn(intermediate, hidden, generator=generator),
        torch.randn(intermediate, hidden, generator=generator),
        torch.randn(hidden, intermediate, generator=generator),
    )
    spec = CoupledHadamardSpec(
        residual_block_size=8,
        preactivation_block_size=8,
        postactivation_block_size=4,
        intermediate_draw=5,
    )
    execution = coupled_execution(encode_coupled_weights(weights, spec), spec)
    rows = torch.randn(11, hidden, generator=generator)
    hidden_rows = torch.randn(11, intermediate, generator=generator)
    h13 = rows.T @ rows
    h2 = hidden_rows.T @ hidden_rows

    transformed_rows = execution.transform_inputs(rows)
    post_signs = rotation_signs(
        intermediate, draw=spec.intermediate_draw, axis=2, device=rows.device
    )
    transformed_hidden_rows = signed_block_hadamard(
        hidden_rows,
        block_size=spec.postactivation_block_size,
        signs=post_signs,
        dim=1,
    )
    transformed_hidden = execution.transform_h2(h2)
    assert torch.allclose(
        execution.transform_h13(h13),
        transformed_rows.T @ transformed_rows,
        rtol=2e-5,
        atol=2e-5,
    )
    assert torch.allclose(
        transformed_hidden,
        transformed_hidden_rows.T @ transformed_hidden_rows,
        rtol=2e-5,
        atol=2e-5,
    )

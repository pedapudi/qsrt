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
    decode_coupled_weights,
    encode_coupled_down_weight,
    encode_coupled_upstream_weights,
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
    encoded_upstream = encode_coupled_upstream_weights(weights[0], weights[1], spec)
    encoded_down = encode_coupled_down_weight(weights[2], spec)
    assert torch.equal(encoded_upstream[0], encoded[0])
    assert torch.equal(encoded_upstream[1], encoded[1])
    assert torch.equal(encoded_down, encoded[2])
    decoded = decode_coupled_weights(encoded, spec)
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
    for actual, expected in zip(decoded, weights, strict=True):
        assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-5)
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
    transformed_outputs = torch.randn(11, hidden, generator=generator)
    h13 = rows.T @ rows
    h2 = hidden_rows.T @ hidden_rows
    output_root = torch.randn(hidden + 3, hidden, generator=generator)
    output_hessian = output_root.T @ output_root

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
    ordinary_outputs = execution.decode_output(transformed_outputs)
    transformed_output_hessian = execution.transform_output_hessian(output_hessian)
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
    ordinary_objective = torch.einsum(
        "bi,ij,bj->", ordinary_outputs, output_hessian, ordinary_outputs
    )
    transformed_objective = torch.einsum(
        "bi,ij,bj->",
        transformed_outputs,
        transformed_output_hessian,
        transformed_outputs,
    )
    assert torch.allclose(
        transformed_objective,
        ordinary_objective,
        rtol=3e-5,
        atol=3e-5,
    )


def test_coupled_upstream_gradient_transform_preserves_directional_derivative() -> None:
    generator = torch.Generator().manual_seed(883)
    hidden, intermediate = 16, 12
    source = (
        torch.randn(intermediate, hidden, generator=generator),
        torch.randn(intermediate, hidden, generator=generator),
        torch.randn(hidden, intermediate, generator=generator),
    )
    spec = CoupledHadamardSpec(
        residual_block_size=8,
        preactivation_block_size=8,
        postactivation_block_size=4,
        intermediate_draw=6,
    )
    execution = coupled_execution(encode_coupled_weights(source, spec), spec)
    source_gradients = (
        torch.randn(intermediate, hidden, generator=generator),
        torch.randn(intermediate, hidden, generator=generator),
    )
    stored_displacements = (
        torch.randn(intermediate, hidden, generator=generator),
        torch.randn(intermediate, hidden, generator=generator),
    )
    decoded_displacements = decode_coupled_weights(
        (
            *stored_displacements,
            torch.zeros(hidden, intermediate),
        ),
        spec,
    )
    stored_gradients = execution.transform_upstream_weight_gradients(
        *source_gradients
    )
    source_derivative = sum(
        torch.sum(gradient * displacement)
        for gradient, displacement in zip(
            source_gradients, decoded_displacements[:2], strict=True
        )
    )
    stored_derivative = sum(
        torch.sum(gradient * displacement)
        for gradient, displacement in zip(
            stored_gradients, stored_displacements, strict=True
        )
    )
    torch.testing.assert_close(source_derivative, stored_derivative)


def test_coupled_activation_cotangents_produce_stored_weight_gradient() -> None:
    generator = torch.Generator().manual_seed(112)
    rows, hidden, intermediate = 7, 16, 12
    source = (
        torch.randn(intermediate, hidden, generator=generator),
        torch.randn(intermediate, hidden, generator=generator),
        torch.randn(hidden, intermediate, generator=generator),
    )
    spec = CoupledHadamardSpec(
        residual_block_size=8,
        preactivation_block_size=8,
        postactivation_block_size=4,
        intermediate_draw=2,
    )
    execution = coupled_execution(encode_coupled_weights(source, spec), spec)
    inputs = torch.randn(rows, hidden, generator=generator)
    gate_up_gradients = torch.randn(rows, 2 * intermediate, generator=generator)
    source_w1_gradient = gate_up_gradients[:, :intermediate].T @ inputs
    source_w3_gradient = gate_up_gradients[:, intermediate:].T @ inputs
    expected = execution.transform_upstream_weight_gradients(
        source_w1_gradient,
        source_w3_gradient,
    )

    transformed_inputs = execution.transform_inputs(inputs)
    transformed_gradients = execution.transform_preactivation_gradients(
        gate_up_gradients
    )
    observed = transformed_gradients.T @ transformed_inputs
    torch.testing.assert_close(observed[:intermediate], expected[0])
    torch.testing.assert_close(observed[intermediate:], expected[1])


def test_coupled_down_cotangents_produce_stored_weight_gradient() -> None:
    generator = torch.Generator().manual_seed(113)
    rows, hidden, intermediate = 7, 16, 12
    source = (
        torch.randn(intermediate, hidden, generator=generator),
        torch.randn(intermediate, hidden, generator=generator),
        torch.randn(hidden, intermediate, generator=generator),
    )
    spec = CoupledHadamardSpec(
        residual_block_size=8,
        preactivation_block_size=8,
        postactivation_block_size=4,
        intermediate_draw=2,
    )
    execution = coupled_execution(encode_coupled_weights(source, spec), spec)
    postactivation = torch.randn(rows, intermediate, generator=generator)
    output_gradients = torch.randn(rows, hidden, generator=generator)
    source_gradient = output_gradients.T @ postactivation
    expected = execution.transform_down_weight_gradient(source_gradient)
    transformed_inputs = execution.transform_postactivation_rows(postactivation)
    transformed_gradients = execution.transform_expert_output_gradients(
        output_gradients
    )
    torch.testing.assert_close(
        transformed_gradients.T @ transformed_inputs,
        expected,
    )

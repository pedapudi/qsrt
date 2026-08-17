from __future__ import annotations

from pathlib import Path

import torch

from qsrt.correctness import sha256_file
from qsrt.glm52_downstream_curvature import (
    DOWNSTREAM_CURVATURE_FACTOR_KIND,
    RoutedGradientSequence,
    _atomic_save_factor_tensors,
    derive_projection_sequences,
    load_expert_curvature_factors,
    sequence_gradient_weight_curvature,
)
from qsrt.qsrt_codec_pilot import tensor_sha256


def test_sequence_gradient_weight_curvature_matches_complete_gradient_grams() -> None:
    sequences = [
        (
            torch.tensor([[1.0, 2.0], [0.5, -1.0]]),
            torch.tensor([[2.0, -1.0], [1.5, 0.25]]),
        ),
        (
            torch.tensor([[-0.5, 1.0], [2.0, 0.25], [1.0, -2.0]]),
            torch.tensor([[0.5, 1.0], [-1.0, 2.0], [0.25, -0.75]]),
        ),
    ]

    input_metric, output_metric, record = sequence_gradient_weight_curvature(
        sequences,
        input_identity_shrinkage=1.0,
        output_identity_shrinkage=1.0,
        device=torch.device("cpu"),
    )

    gradients = [output.T @ values for values, output in sequences]
    raw_input = sum(gradient.T @ gradient for gradient in gradients) / 2
    raw_output = sum(gradient @ gradient.T for gradient in gradients) / 2
    assert torch.equal(input_metric, torch.eye(2))
    assert torch.equal(output_metric, torch.eye(2))
    assert record["input_metric"]["unnormalized_diagonal_mean"] == float(
        raw_input.diagonal().mean()
    )
    assert record["output_metric"]["unnormalized_diagonal_mean"] == float(
        raw_output.diagonal().mean()
    )
    assert record["sequence_count"] == 2
    assert record["routed_row_count"] == 5


def test_sequence_gradient_weight_curvature_preserves_correlations_before_shrinkage() -> None:
    values = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    gradients = torch.tensor([[1.0, 2.0], [0.5, 1.0]])

    input_metric, output_metric, _ = sequence_gradient_weight_curvature(
        [(values, gradients)],
        input_identity_shrinkage=0.25,
        output_identity_shrinkage=0.25,
        device=torch.device("cpu"),
    )

    gradient_weight = gradients.T @ values
    expected_input = gradient_weight.T @ gradient_weight
    expected_input /= expected_input.diagonal().mean()
    expected_input *= 0.75
    expected_input.diagonal().add_(0.25)
    expected_output = gradient_weight @ gradient_weight.T
    expected_output /= expected_output.diagonal().mean()
    expected_output *= 0.75
    expected_output.diagonal().add_(0.25)
    assert torch.allclose(input_metric, expected_input)
    assert torch.allclose(output_metric, expected_output)


def test_swiglu_chain_rule_derives_each_projection_gradient() -> None:
    gate = torch.tensor([[0.5, -0.25], [0.75, 0.125]])
    up = torch.tensor([[0.25, 0.5], [-0.5, 0.75]])
    down = torch.tensor([[0.5, -0.75], [0.25, 0.5]])
    x = torch.tensor([[1.0, -2.0], [0.5, 0.25]])
    expert_output_gradient = torch.tensor([[0.2, -0.4], [0.75, 0.5]])

    derived = derive_projection_sequences(
        [
            RoutedGradientSequence(
                sequence_identity="document-a",
                expert_inputs=x,
                expert_output_gradients=expert_output_gradient,
            )
        ],
        gate_weight=gate,
        up_weight=up,
        down_weight=down,
        device=torch.device("cpu"),
    )

    x_reference = x.clone().requires_grad_(True)
    gate_reference = gate.clone().requires_grad_(True)
    up_reference = up.clone().requires_grad_(True)
    down_reference = down.clone().requires_grad_(True)
    gate_values = torch.nn.functional.linear(x_reference, gate_reference)
    up_values = torch.nn.functional.linear(x_reference, up_reference)
    hidden = torch.nn.functional.silu(gate_values) * up_values
    output = torch.nn.functional.linear(hidden, down_reference)
    output.backward(expert_output_gradient)

    gate_input, gate_gradient = derived["gate_proj"][0]
    up_input, up_gradient = derived["up_proj"][0]
    down_input, down_gradient = derived["down_proj"][0]
    assert torch.equal(gate_input, x)
    assert torch.equal(up_input, x)
    assert torch.allclose(down_input, hidden.detach())
    assert torch.allclose(gate_gradient.T @ gate_input, gate_reference.grad)
    assert torch.allclose(up_gradient.T @ up_input, up_reference.grad)
    assert torch.allclose(down_gradient.T @ down_input, down_reference.grad)


def test_factor_file_loader_closes_file_and_tensor_hashes(tmp_path: Path) -> None:
    tensors = {
        f"{projection}.{role}_metric": torch.eye(2, dtype=torch.float32)
        for projection in ("gate_proj", "up_proj", "down_proj")
        for role in ("input", "output")
    }
    factor_path = tmp_path / "experts" / "layer-003-expert-007.safetensors"
    _atomic_save_factor_tensors(factor_path, tensors)
    record = {
        "factor_file": factor_path.name,
        "factor_file_bytes": factor_path.stat().st_size,
        "factor_file_sha256": sha256_file(factor_path),
        "projection_factors": {
            projection: {
                "input_metric_sha256": tensor_sha256(
                    tensors[f"{projection}.input_metric"]
                ),
                "output_metric_sha256": tensor_sha256(
                    tensors[f"{projection}.output_metric"]
                ),
            }
            for projection in ("gate_proj", "up_proj", "down_proj")
        },
    }

    loaded = load_expert_curvature_factors(tmp_path, record)

    assert set(loaded) == {"gate_proj", "up_proj", "down_proj"}
    assert all(
        torch.equal(metrics["input_metric"], torch.eye(2))
        and torch.equal(metrics["output_metric"], torch.eye(2))
        for metrics in loaded.values()
    )
    assert DOWNSTREAM_CURVATURE_FACTOR_KIND

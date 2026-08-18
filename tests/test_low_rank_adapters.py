from __future__ import annotations

import torch
from safetensors.torch import save_file

from qsrt.low_rank_adapters import (
    fit_plain_error_adapter,
    fit_weighted_error_adapter,
    load_sparse_expert_adapter_banks,
)


def test_plain_error_adapter_recovers_low_rank_matrix() -> None:
    torch.manual_seed(7)
    left = torch.randn(11, 3)
    right = torch.randn(9, 3)
    error = left @ right.T
    fit = fit_plain_error_adapter(error, rank=3, oversampling=3, power_iterations=2)
    recovered = fit.b @ fit.a.T
    torch.testing.assert_close(recovered, error, rtol=2e-4, atol=2e-4)


def test_weighted_error_adapter_recovers_supported_output_subspace() -> None:
    torch.manual_seed(11)
    left = torch.randn(13, 2)
    right = torch.randn(7, 2)
    error = left @ right.T
    rows = torch.randn(64, 7)
    weights = torch.linspace(0.1, 1.0, 64)
    fit = fit_weighted_error_adapter(
        error,
        rows,
        weights,
        rank=2,
        oversampling=3,
        power_iterations=2,
        batch_rows=9,
        seed=3,
    )
    recovered = fit.b @ fit.a.T
    torch.testing.assert_close(recovered, error, rtol=3e-4, atol=3e-4)
    assert fit.objective_captured[-1] / fit.objective_total > 0.999


def test_sparse_factor_loader_expands_selected_experts(tmp_path) -> None:
    tensors = {}
    shapes = {"w1": (5, 3), "w2": (3, 5), "w3": (5, 3)}
    for expert in (1, 3):
        for matrix, (output_dimension, input_dimension) in shapes.items():
            tensors[f"experts.{expert}.{matrix}.weighted.rank_2.a"] = torch.full(
                (input_dimension, 2), expert + 0.25, dtype=torch.bfloat16
            )
            tensors[f"experts.{expert}.{matrix}.weighted.rank_2.b"] = torch.full(
                (output_dimension, 2), expert + 0.5, dtype=torch.bfloat16
            )
            tensors[f"experts.{expert}.{matrix}.plain.rank_2.a"] = torch.zeros(
                (input_dimension, 2), dtype=torch.bfloat16
            )
            tensors[f"experts.{expert}.{matrix}.plain.rank_2.b"] = torch.zeros(
                (output_dimension, 2), dtype=torch.bfloat16
            )
    path = tmp_path / "factors.safetensors"
    save_file(tensors, path, metadata={"layer": "84"})

    banks, experts, metadata = load_sparse_expert_adapter_banks(
        path,
        variant="weighted",
        rank=2,
        matrix_shapes=shapes,
        num_experts=4,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert experts == (1, 3)
    assert metadata == {"layer": "84"}
    for matrix in shapes:
        a, b = banks[matrix]
        assert torch.count_nonzero(a[0]) == 0
        assert torch.count_nonzero(b[2]) == 0
        assert torch.all(a[1] == torch.tensor(1.25, dtype=torch.bfloat16))
        assert torch.all(b[3] == torch.tensor(3.5, dtype=torch.bfloat16))

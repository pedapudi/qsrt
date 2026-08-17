from __future__ import annotations

import pytest
import torch

from qsrt.glm52_routed_input_curvature import (
    routed_input_hessian,
    upstream_curvature_basis_name,
)


def test_each_gate_up_choice_has_a_distinct_down_curvature_basis() -> None:
    names = {
        upstream_curvature_basis_name(
            gate_curvature=gate_curvature,
            up_curvature=up_curvature,
        )
        for gate_curvature in (False, True)
        for up_curvature in (False, True)
    }

    assert names == {
        "gate_baseline_up_baseline",
        "gate_baseline_up_curvature",
        "gate_curvature_up_baseline",
        "gate_curvature_up_curvature",
    }


def test_routed_input_hessian_is_weighted_symmetric_and_shrunk() -> None:
    values = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    weights = torch.tensor([1.0, 0.5, 0.0])
    hessian, record = routed_input_hessian(
        values, weights, identity_shrinkage=0.1
    )
    assert torch.allclose(hessian, hessian.T)
    assert torch.linalg.eigvalsh(hessian).min() > 0.0
    assert record["row_count"] == 3
    assert record["squared_route_mass"] == pytest.approx(1.25)
    assert record["identity_shrinkage"] == 0.1


def test_routed_input_hessian_rejects_invalid_weights_and_shrinkage() -> None:
    with pytest.raises(ValueError, match="do not match"):
        routed_input_hessian(
            torch.ones(2, 3), torch.ones(3), identity_shrinkage=0.1
        )
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        routed_input_hessian(
            torch.ones(2, 3), torch.ones(2), identity_shrinkage=0.0
        )
    with pytest.raises(ValueError, match="zero squared mass"):
        routed_input_hessian(
            torch.ones(2, 3), torch.zeros(2), identity_shrinkage=0.1
        )

from __future__ import annotations

import pytest
import torch

from qsrt.glm52_two_sided_curvature import (
    _baseline_global_scale,
    _canonical_curvature_loss,
    _scale_vectors_close,
)


def test_canonical_curvature_loss_matches_explicit_kronecker_form() -> None:
    source = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    reconstruction = torch.tensor([[0.75, -1.0], [0.25, 2.5]])
    input_metric = torch.tensor([[2.0, 0.25], [0.25, 1.0]])
    output_metric = torch.tensor([[1.5, -0.1], [-0.1, 0.75]])

    actual = _canonical_curvature_loss(
        source,
        reconstruction,
        input_metric,
        output_metric,
        device=torch.device("cpu"),
    )

    error = (source - reconstruction).reshape(-1)
    expected = error @ torch.kron(output_metric, input_metric) @ error
    assert actual == pytest.approx(float(expected), abs=1e-7)


def test_baseline_global_scale_reads_the_uniform_k3_receipt() -> None:
    record = {
        "projections": {
            "gate_proj": {"qsrt_k3": {"payload": {"g_scale": 1.125}}}
        }
    }

    assert _baseline_global_scale(record, "gate_proj") == 1.125


def test_scale_vector_closure_requires_both_hashes_bytes_and_scale() -> None:
    control = {
        "suh_sha256": "a" * 64,
        "svh_sha256": "b" * 64,
        "scale_bytes": 24,
        "g_scale": 0.875,
    }
    assert _scale_vectors_close(control, dict(control))
    changed = dict(control)
    changed["svh_sha256"] = "c" * 64
    assert not _scale_vectors_close(control, changed)

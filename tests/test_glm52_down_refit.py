from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from qsrt.correctness import sha256_file
from qsrt.glm52_down_refit import (
    _read_capture_rows,
    solve_down_correction,
    weighted_relative_sse,
)


def _write_capture(
    root: Path, *, invalid_plan_metadata: bool = False
) -> None:
    root.mkdir()
    plan_sha256 = "a" * 64
    records = []
    for generation, collection in enumerate(
        ("activation_fit", "candidate_selection"), start=1
    ):
        filename = f"layer-003-input-chunk-{generation:06d}.safetensors"
        path = root / filename
        hidden = torch.arange(2 * 6144, dtype=torch.bfloat16).reshape(2, 6144)
        ids = torch.tensor(
            [[7, 1, 2, 3, 4, 5, 6, 8], [9, 10, 7, 11, 12, 13, 14, 15]],
            dtype=torch.int32,
        )
        weights = torch.full((2, 8), 0.125, dtype=torch.float32)
        save_file(
            {"hidden_states": hidden, "topk_ids": ids, "topk_weights": weights},
            path,
            metadata={
                "schema": "qsrt_glm52_layer_input_capture_v1",
                "model_layer": "3",
                "control_generation": str(generation),
                "corpus_plan_sha256": (
                    "b" * 64 if invalid_plan_metadata else plan_sha256
                ),
            },
        )
        records.append(
            {
                "collection": collection,
                "window_id": f"article-{generation}",
                "token_count": 2,
                "control_generation": generation,
                "capture_file": filename,
                "capture_file_bytes": path.stat().st_size,
                "capture_file_sha256": sha256_file(path),
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "qsrt_glm52_layer_input_capture_manifest",
                "schema_version": 1,
                "status": "complete",
                "model_layer": 3,
                "corpus_plan_sha256": plan_sha256,
                "records": records,
                "collections": {"activation_fit": 1, "candidate_selection": 1},
            }
        )
    )


def test_ridge_correction_recovers_a_small_linear_residual() -> None:
    torch.manual_seed(7)
    hidden = torch.randn(32, 4)
    expected = torch.randn(3, 4)
    residual = hidden @ expected.T
    correction, metadata = solve_down_correction(
        hidden,
        residual,
        torch.ones(32),
        ridge_factor=1e-8,
    )
    assert torch.allclose(correction, expected, atol=1e-5, rtol=1e-5)
    assert metadata["ridge_absolute"] > 0.0


def test_route_weighted_error_ignores_zero_weight_rows() -> None:
    teacher = torch.tensor([[1.0, 0.0], [10.0, 10.0]])
    candidate = torch.tensor([[0.0, 0.0], [-100.0, 100.0]])
    value = weighted_relative_sse(
        teacher, candidate, torch.tensor([1.0, 0.0])
    )
    assert value == 1.0


def test_ridge_and_metric_validate_shapes_and_regularization() -> None:
    with pytest.raises(ValueError, match="row counts differ"):
        solve_down_correction(
            torch.ones(3, 2),
            torch.ones(4, 2),
            torch.ones(3),
            ridge_factor=0.1,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        solve_down_correction(
            torch.ones(3, 2),
            torch.ones(3, 2),
            torch.ones(3),
            ridge_factor=0.0,
        )
    with pytest.raises(ValueError, match="one value per output row"):
        weighted_relative_sse(torch.ones(2, 2), torch.ones(2, 2), torch.ones(3))


def test_capture_reader_revalidates_metadata_shapes_and_selected_routes(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "capture"
    _write_capture(capture)

    rows = _read_capture_rows(capture, experts=[7])

    for collection in ("activation_fit", "candidate_selection"):
        hidden, weights = rows[collection][7]
        assert tuple(hidden.shape) == (2, 6144)
        assert torch.equal(weights, torch.full((2,), 0.125))


def test_capture_reader_rejects_mixed_corpus_plan_metadata(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    _write_capture(capture, invalid_plan_metadata=True)

    with pytest.raises(ValueError, match="metadata mismatch"):
        _read_capture_rows(capture, experts=[7])

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from qsrt.glm52_reporting_output import (
    PUBLISHED_REFERENCE_MANIFEST_SHA256,
    REPORTING_CAPTURE_SCHEMA,
    _runtime_contract,
    _validate_capture,
    evaluate_complete_expert,
    squared_error_receipt,
)


def _write_reporting_capture(root: Path) -> tuple[Path, Path]:
    root.mkdir()
    tensor_path = root / "layer-003-input-chunk-000000.safetensors"
    plan_sha256 = "1" * 64
    hidden = torch.zeros((3, 6144), dtype=torch.bfloat16)
    topk_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 7, 6, 5, 4, 3, 2, 1],
            [9, 10, 11, 12, 13, 14, 15, 16],
        ],
        dtype=torch.int32,
    )
    topk_weights = torch.full((3, 8), 0.125, dtype=torch.float32)
    save_file(
        {
            "hidden_states": hidden,
            "topk_ids": topk_ids,
            "topk_weights": topk_weights,
        },
        tensor_path,
        metadata={
            "schema": "qsrt_glm52_layer_input_capture_v1",
            "model_layer": "3",
            "control_generation": "900001",
            "corpus_plan_sha256": plan_sha256,
        },
    )
    tensor_sha256 = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": REPORTING_CAPTURE_SCHEMA,
                "schema_version": 1,
                "status": "complete",
                "model_layer": 3,
                "corpus_plan_sha256": plan_sha256,
                "reference_manifest_sha256": (
                    PUBLISHED_REFERENCE_MANIFEST_SHA256
                ),
                "collections": {"untouched_reporting_context": 1},
                "reuse_policy": (
                    "reporting analysis only; this context must not select paths, "
                    "rates, refits, shrinkage, or allocation settings"
                ),
                "records": [
                    {
                        "token_count": 3,
                        "control_generation": 900001,
                        "capture_file": tensor_path.name,
                        "capture_file_bytes": tensor_path.stat().st_size,
                        "capture_file_sha256": tensor_sha256,
                    }
                ],
            }
        )
    )
    return manifest_path, tensor_path


def test_squared_error_receipt_uses_reference_energy() -> None:
    reference = torch.tensor([[1.0, 2.0]])
    candidate = torch.tensor([[2.0, 0.0]])

    receipt = squared_error_receipt(reference, candidate)

    assert receipt == {
        "squared_error_sum": 5.0,
        "reference_squared_sum": 5.0,
        "relative_squared_error": 1.0,
    }
    with pytest.raises(ValueError, match="rank-two"):
        squared_error_receipt(reference, candidate.squeeze(0))


def test_complete_expert_matches_explicit_swiglu_equation() -> None:
    expert_input = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    gate = torch.tensor([[0.5, 1.0], [-1.0, 0.25]], dtype=torch.float32)
    up = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    down = torch.tensor([[1.0, -0.5], [0.25, 2.0]], dtype=torch.float32)

    actual = evaluate_complete_expert(
        expert_input, gate=gate, up=up, down=down
    )
    gate_values = expert_input @ gate.T
    up_values = expert_input @ up.T
    expected = (torch.nn.functional.silu(gate_values) * up_values) @ down.T

    torch.testing.assert_close(actual, expected)


def test_reporting_capture_closes_bytes_metadata_and_tensor_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "capture"
    _write_reporting_capture(root)

    manifest, hidden, topk_ids, topk_weights = _validate_capture(root)

    assert manifest["schema"] == REPORTING_CAPTURE_SCHEMA
    assert hidden.shape == (3, 6144)
    assert topk_ids.shape == (3, 8)
    assert topk_weights.shape == (3, 8)


def test_reporting_capture_rejects_candidate_selection_permission(
    tmp_path: Path,
) -> None:
    root = tmp_path / "capture"
    manifest_path, _ = _write_reporting_capture(root)
    manifest = json.loads(manifest_path.read_text())
    manifest["reuse_policy"] = "candidate selection is permitted"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="does not prohibit"):
        _validate_capture(root)


def test_reporting_capture_rejects_changed_tensor_bytes(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    _, tensor_path = _write_reporting_capture(root)
    with tensor_path.open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(ValueError, match="byte closure"):
        _validate_capture(root)


def test_runtime_contract_ignores_only_load_duration() -> None:
    first = {"runtime": {"dtype": "bfloat16", "model_load_seconds": 10.0}}
    second = {"runtime": {"dtype": "bfloat16", "model_load_seconds": 99.0}}

    assert _runtime_contract(first) == _runtime_contract(second)
    second["runtime"]["dtype"] = "float16"
    assert _runtime_contract(first) != _runtime_contract(second)

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from qsrt.continuous_recovery import (
    audit_continuous_recovery,
    suffix_capture_storage,
)
from qsrt.kimi_capture_documents import load_token_suite_document_index
from qsrt.kimi_routes import KimiRouteArchive, compare_route_archives


def _checkpoint(root: Path) -> Path:
    root.mkdir()
    tensors: dict[str, torch.Tensor] = {}
    for layer in range(2, 4):
        prefix = f"language_model.model.layers.{layer}"
        tensors[f"{prefix}.block_sparse_moe.gate.weight"] = torch.zeros(
            3, 8, dtype=torch.bfloat16
        )
        tensors[
            f"{prefix}.block_sparse_moe.gate.e_score_correction_bias"
        ] = torch.zeros(3, dtype=torch.float32)
        tensors[f"{prefix}.self_attn.q_proj.weight"] = torch.zeros(
            8, 8, dtype=torch.float8_e4m3fn
        )
        tensors[f"{prefix}.self_attn.q_proj.weight_scale"] = torch.zeros(
            8, 1, dtype=torch.uint8
        )
        tensors[f"{prefix}.self_attn.A_log"] = torch.zeros(128, dtype=torch.float32)
        tensors[f"{prefix}.input_layernorm.weight"] = torch.ones(
            8, dtype=torch.bfloat16
        )
    tensors.update(
        {
            "language_model.lm_head.weight": torch.zeros(12, 8, dtype=torch.bfloat16),
            "language_model.model.norm.weight": torch.ones(8, dtype=torch.bfloat16),
            "language_model.model.output_attn_res_norm.weight": torch.ones(
                8, dtype=torch.bfloat16
            ),
            "language_model.model.output_attn_res_proj.weight": torch.ones(
                1, 8, dtype=torch.bfloat16
            ),
        }
    )
    shard = "model.safetensors"
    save_file(tensors, root / shard)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}})
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "num_hidden_layers": 4,
                    "hidden_size": 8,
                    "attn_res_block_size": 2,
                    "linear_attn_config": {"num_heads": 6},
                }
            }
        )
    )
    return root


def test_suffix_capture_storage_includes_residual_prefixes() -> None:
    storage = suffix_capture_storage(
        10,
        hidden_dimension=8,
        first_layer=6,
        attention_residual_block_size=2,
    )
    assert storage.student_hidden_vectors_per_token == 4
    assert storage.student_bytes == 4 * 10 * 8 * 2
    assert storage.teacher_bytes == 10 * 8 * 2
    assert storage.total_bytes == 5 * 10 * 8 * 2


def test_suffix_capture_rejects_midsegment_cut() -> None:
    with pytest.raises(ValueError, match="segment boundary"):
        suffix_capture_storage(
            10,
            hidden_dimension=8,
            first_layer=5,
            attention_residual_block_size=2,
        )


def test_inventory_excludes_router_bias_and_mxfp8_scales(tmp_path: Path) -> None:
    report = audit_continuous_recovery(
        _checkpoint(tmp_path / "checkpoint"),
        first_layer=2,
        end_layer=4,
        capture_token_count=10,
    )
    tensors = {item["name"]: item for item in report["trainable"]["tensors"]}
    q_proj = tensors["language_model.model.layers.2.self_attn.q_proj.weight"]
    assert q_proj["source_dtype"] == "F8_E4M3"
    assert q_proj["runtime_dtype"] == "BF16"
    assert q_proj["parameter_bytes"] == 8 * 8 * 2
    a_log = tensors["language_model.model.layers.2.self_attn.A_log"]
    assert a_log["source_shape"] == (128,)
    assert a_log["runtime_shape"] == (6,)
    excluded = {item["name"]: item["reason"] for item in report["excluded"]["tensors"]}
    assert any(name.endswith("weight_scale") for name in excluded)
    assert any(name.endswith("e_score_correction_bias") for name in excluded)
    assert report["capture_storage"]["student_hidden_vectors_per_token"] == 2


def test_token_suite_loader_preserves_stored_ids(tmp_path: Path) -> None:
    root = tmp_path / "suite"
    (root / "tokens").mkdir(parents=True)
    contexts = []
    for index, values in ((2, [7, 11]), (5, [13, 17, 19])):
        path = root / "tokens" / f"context-{index:04d}.json"
        path.write_text(json.dumps(values, separators=(",", ":")))
        contexts.append(
            {
                "context_index": index,
                "token_file": str(path.relative_to(root)),
                "token_ids_json_sha256": hashlib.sha256(
                    json.dumps(values, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "num_tokens": len(values),
            }
        )
    (root / "suite-manifest.json").write_text(json.dumps({"contexts": contexts}))
    documents = load_token_suite_document_index(root, (5, 2))
    assert documents.input_ids.tolist() == [13, 17, 19, 7, 11]
    assert documents.offsets.tolist() == [0, 3, 5]


def test_token_suite_loader_accepts_window_manifest(tmp_path: Path) -> None:
    root = tmp_path / "suite"
    (root / "tokens").mkdir(parents=True)
    values = [23, 29, 31]
    path = root / "tokens" / "window-000-prose.json"
    path.write_text(json.dumps(values, separators=(",", ":")))
    digest = hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (root / "suite-manifest.json").write_text(
        json.dumps(
            {
                "windows": [
                    {
                        "index": 0,
                        "token_file": str(path.relative_to(root)),
                        "token_ids_json_sha256": digest,
                        "num_tokens": len(values),
                    }
                ]
            }
        )
    )
    documents = load_token_suite_document_index(root, (0,))
    assert documents.input_ids.tolist() == values
    assert documents.offsets.tolist() == [0, 3]


def _routes(root: Path, values: dict[int, torch.Tensor]) -> KimiRouteArchive:
    archive = KimiRouteArchive.create(
        root,
        token_count=3,
        num_layers=3,
        first_layer=1,
        num_experts=8,
        top_k=2,
    )
    for layer, routes in values.items():
        writer = archive.writer(layer)
        writer.append(routes)
        writer.finish()
    archive.seal()
    return KimiRouteArchive(root, require_complete=True)


def test_route_comparison_is_set_based(tmp_path: Path) -> None:
    teacher_values = {
        1: torch.tensor([[1, 2], [3, 4], [5, 6]]),
        2: torch.tensor([[0, 1], [2, 3], [4, 5]]),
    }
    student_values = {
        1: torch.tensor([[2, 1], [3, 7], [6, 5]]),
        2: teacher_values[2].flip(1),
    }
    report = compare_route_archives(
        _routes(tmp_path / "teacher", teacher_values),
        _routes(tmp_path / "student", student_values),
        chunk_tokens=2,
    )
    layers = {item["layer"]: item for item in report["layers"]}
    assert layers[1]["mean_topk_overlap"] == pytest.approx(5 / 6)
    assert layers[1]["exact_topk_set_agreement"] == pytest.approx(2 / 3)
    assert layers[2]["mean_topk_overlap"] == 1.0
    assert layers[2]["exact_topk_set_agreement"] == 1.0

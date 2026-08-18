import importlib.util
from pathlib import Path

import torch
from safetensors.torch import save_file


def _capture_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "capture_kimi_k3_suffix_fisher.py"
    )
    spec = importlib.util.spec_from_file_location(
        "capture_kimi_k3_suffix_fisher", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_suffix_capture_schema_closes_raw_and_consolidated_rows(
    tmp_path: Path,
) -> None:
    capture = _capture_module()
    request_dir = tmp_path / "request"
    request_dir.mkdir()
    raw = request_dir / "suffix.rows-000000-000004.safetensors"
    rows = torch.tensor([0, 2], dtype=torch.int64)
    tensors = {
        "expert_indices": torch.tensor([[1, 3], [2, 4]], dtype=torch.int32),
        "expert_input": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
        "final_mixed": torch.arange(10, dtype=torch.bfloat16).reshape(2, 5),
        "prefix_weight": torch.tensor([0.25, 0.75], dtype=torch.float32),
        "route_weights": torch.tensor(
            [[0.4, 0.6], [0.7, 0.3]], dtype=torch.float32
        ),
        "routed_latent": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        "row_index": rows,
        "updated_prefix": torch.arange(10, dtype=torch.bfloat16).reshape(2, 5),
    }
    raw_metadata = {
        "format_version": "2",
        "row_start": "0",
        "row_end": "4",
        "row_stride": "2",
        "row_offset": "0",
        "expert_input_dimension": "3",
        "experts_per_token": "2",
        "latent_dimension": "4",
        "hidden_dimension": "5",
        "semantic_point": "kimi_k3_layer_92_common_suffix",
        "route_weight_semantics": "applied_moe_weight",
    }
    save_file(tensors, raw, metadata=raw_metadata)

    loaded = capture._load_chunks(
        request_dir,
        scored_rows=4,
        row_stride=2,
        row_offset=0,
        expert_input_dimension=3,
        experts_per_token=2,
        latent_dimension=4,
        hidden_dimension=5,
        num_experts=8,
    )
    for key, value in tensors.items():
        torch.testing.assert_close(loaded[key], value)

    consolidated = tmp_path / "suffix_0000.safetensors"
    token_hash = "a" * 64
    save_file(
        loaded,
        consolidated,
        metadata={
            "format_version": "2",
            "token_ids_json_sha256": token_hash,
            "semantic_point": "kimi_k3_layer_92_common_suffix",
            "route_weight_semantics": "applied_moe_weight",
            "expert_input_dimension": "3",
            "experts_per_token": "2",
            "latent_dimension": "4",
            "hidden_dimension": "5",
            "num_experts": "8",
        },
    )
    capture._validate_output(
        consolidated,
        token_hash=token_hash,
        expected_rows=rows,
        expert_input_dimension=3,
        experts_per_token=2,
        latent_dimension=4,
        hidden_dimension=5,
        num_experts=8,
    )

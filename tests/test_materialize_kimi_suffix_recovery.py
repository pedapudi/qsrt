from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def test_materializer_replaces_mxfp8_and_removes_its_scale(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    shared = (
        "language_model.model.layers.84.block_sparse_moe.shared_experts."
        "gate_proj.weight"
    )
    scale = f"{shared}_scale"
    norm = "language_model.model.layers.84.input_layernorm.weight"
    unrelated = "language_model.model.layers.1.input_layernorm.weight"
    first_shard = "model-00001-of-00002.safetensors"
    second_shard = "model-00002-of-00002.safetensors"
    save_file(
        {
            shared: torch.zeros(2, 32, dtype=torch.float8_e4m3fn),
            scale: torch.zeros(2, 1, dtype=torch.uint8),
            norm: torch.ones(2, dtype=torch.bfloat16),
        },
        anchor / first_shard,
    )
    save_file(
        {unrelated: torch.ones(3, dtype=torch.bfloat16)},
        anchor / second_shard,
    )
    weight_map = {
        shared: first_shard,
        scale: first_shard,
        norm: first_shard,
        unrelated: second_shard,
    }
    (anchor / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map})
    )
    (anchor / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "quantization_config": {
                        "dense_format": "mxfp8",
                        "ignored_layers": ["kv_b_proj"],
                    }
                }
            }
        )
    )
    overlay = tmp_path / "overlay.safetensors"
    replacements = {
        shared: torch.randn(2, 32, dtype=torch.bfloat16),
        norm: torch.randn(2, dtype=torch.bfloat16),
    }
    save_file(replacements, overlay)
    destination = tmp_path / "materialized"

    subprocess.run(
        [
            sys.executable,
            "scripts/materialize_kimi_suffix_recovery.py",
            "--anchor",
            str(anchor),
            "--overlay",
            str(overlay),
            "--dest",
            str(destination),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    index = json.loads((destination / "model.safetensors.index.json").read_text())
    assert scale not in index["weight_map"]
    assert index["metadata"]["total_size"] == 2 * 32 * 2 + 2 * 2 + 3 * 2
    with safe_open(
        destination / first_shard,
        framework="pt",
        device="cpu",
    ) as reader:
        assert scale not in reader.keys()
        for name, expected in replacements.items():
            torch.testing.assert_close(reader.get_tensor(name), expected)
    with safe_open(anchor / first_shard, framework="pt", device="cpu") as reader:
        assert scale in reader.keys()
        assert reader.get_tensor(shared).dtype == torch.float8_e4m3fn
    assert os.stat(anchor / second_shard).st_ino == os.stat(
        destination / second_shard
    ).st_ino
    config = json.loads((destination / "config.json").read_text())
    assert config["text_config"]["quantization_config"]["ignored_layers"] == [
        "kv_b_proj",
        shared.removesuffix(".weight"),
    ]
    anchor_config = json.loads((anchor / "config.json").read_text())
    assert anchor_config["text_config"]["quantization_config"]["ignored_layers"] == [
        "kv_b_proj"
    ]

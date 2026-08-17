from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from qsrt.glm52_expert_intervention import (
    INTERVENTION_ARTIFACT_KIND,
    _shared_scale_key,
    decode_r7_projection,
    merge_dense_intervention_artifacts,
)
from qsrt.correctness import sha256_file
from qsrt.glm52_pilot import PROJECTIONS


class _TensorHandle:
    def __init__(self, tensors: dict[str, torch.Tensor]):
        self.tensors = tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.tensors[name]


class _FakeExtension:
    def reconstruct(
        self,
        output: torch.Tensor,
        trellis: torch.Tensor,
        bits: int,
        unpermute: bool,
        mul1: bool,
    ) -> None:
        assert bits == 3
        assert unpermute is True
        assert mul1 is False
        output.fill_(0.25)


class _FakeQuantizer:
    ext = _FakeExtension()

    @staticmethod
    def preapply_had_l(value: torch.Tensor, block: int) -> torch.Tensor:
        assert block == 128
        return value

    @staticmethod
    def preapply_had_r(value: torch.Tensor, block: int) -> torch.Tensor:
        assert block == 128
        return value


def _small_gate_spec():
    return replace(
        PROJECTIONS[0],
        source_shape=(32, 128),
        encoder_shape=(128, 32),
    )


def test_r7_gate_decode_uses_shared_input_and_expert_output_scales() -> None:
    spec = _small_gate_spec()
    prefix = "model.layers.3.mlp.experts.7.gate_proj"
    handle = _TensorHandle(
        {
            f"{prefix}.trellis": torch.zeros((8, 2, 48), dtype=torch.int16),
            f"{prefix}.svh": torch.full((32,), 3.0, dtype=torch.float16),
            f"{prefix}.mcg": torch.tensor(0xCBAC1FED - 2**32, dtype=torch.int32),
            _shared_scale_key(3, "gate_proj"): torch.full(
                (128,), 2.0, dtype=torch.float16
            ),
        }
    )

    decoded, payload = decode_r7_projection(
        handle,
        layer=3,
        expert=7,
        spec=spec,
        bits=3,
        device=torch.device("cpu"),
        quantizer_module=_FakeQuantizer(),
    )

    assert decoded.dtype == torch.float16
    assert tuple(decoded.shape) == spec.source_shape
    assert torch.all(decoded == 1.5)
    assert payload["trellis_bpw"] == 3.0
    assert payload["mcg_multiplier"] == 0xCBAC1FED


def test_r7_decode_rejects_wrong_rate_and_marker() -> None:
    spec = _small_gate_spec()
    with pytest.raises(ValueError, match="only K3, K4, or K5"):
        decode_r7_projection(
            _TensorHandle({}),
            layer=3,
            expert=7,
            spec=spec,
            bits=2,
            device=torch.device("cpu"),
            quantizer_module=_FakeQuantizer(),
        )

    prefix = "model.layers.3.mlp.experts.7.gate_proj"
    handle = _TensorHandle(
        {
            f"{prefix}.trellis": torch.zeros((8, 2, 48), dtype=torch.int16),
            f"{prefix}.svh": torch.ones((32,), dtype=torch.float16),
            f"{prefix}.mcg": torch.tensor(1, dtype=torch.int32),
            _shared_scale_key(3, "gate_proj"): torch.ones(
                (128,), dtype=torch.float16
            ),
        }
    )
    with pytest.raises(ValueError, match="unexpected multiplier"):
        decode_r7_projection(
            handle,
            layer=3,
            expert=7,
            spec=spec,
            bits=3,
            device=torch.device("cpu"),
            quantizer_module=_FakeQuantizer(),
        )


def test_r7_shared_scale_names_are_projection_specific() -> None:
    assert _shared_scale_key(3, "gate_proj").endswith(".gate_up_suh")
    assert _shared_scale_key(3, "up_proj").endswith(".gate_up_suh")
    assert _shared_scale_key(3, "down_proj").endswith(".down_svh")
    with pytest.raises(ValueError, match="unsupported expert projection"):
        _shared_scale_key(3, "router")


def _write_merge_slice(root: Path, *, expert: int) -> None:
    experts = root / "experts"
    experts.mkdir(parents=True)
    filename = f"layer-003-expert-{expert:03d}.safetensors"
    dense_path = experts / filename
    dense_path.write_bytes(f"expert-{expert}".encode())
    manifest = {
        "kind": f"{INTERVENTION_ARTIFACT_KIND}_manifest",
        "source": {
            "model_id": "zai-org/GLM-5.2",
            "revision": "source-revision",
            "config_sha256": "a" * 64,
            "index_sha256": "b" * 64,
            "source_inventory_sha256": "c" * 64,
        },
        "exl3_endpoint": {
            "model_id": "brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78",
            "revision": "endpoint-revision",
            "manifest_sha256": "d" * 64,
            "manifest_json_sha256": "e" * 64,
            "layer": 3,
            "sidecar_sha256": "f" * 64,
            "shard": "r7-experts-layer-003.safetensors",
            "shard_sha256": "1" * 64,
            "allocation_bpw": 3.5,
        },
        "candidate": {"uniform_rate": 3},
        "resident_endpoint_dtype": "FP16",
        "resident_coordinate_basis": "sealed permutation",
        "device": "cuda:0",
        "evidence_boundary": "test fixture",
        "model_downloads_required": False,
        "complete_bf16_checkpoint_required": False,
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    record = {
        "kind": f"{INTERVENTION_ARTIFACT_KIND}_expert",
        "complete": True,
        "manifest_sha256": manifest_sha256,
        "layer": 3,
        "expert": expert,
        "dense_endpoint_file": filename,
        "dense_endpoint_file_bytes": dense_path.stat().st_size,
        "dense_endpoint_file_sha256": sha256_file(dense_path),
    }
    report = {
        "kind": INTERVENTION_ARTIFACT_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "layer": 3,
        "expert_count": 1,
        "dense_endpoint_bytes": dense_path.stat().st_size,
        "experts": [record],
    }
    (root / "report.json").write_text(json.dumps(report))


def test_merge_preserves_dense_bytes_and_frozen_panel_order(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_merge_slice(first, expert=64)
    _write_merge_slice(second, expert=208)
    dest = tmp_path / "merged"

    report = merge_dense_intervention_artifacts(
        inputs=[first, second],
        dest=dest,
        panel_manifest_path=Path("experiments/glm52_layer3_rate_pattern_panel.json"),
        layer=3,
    )

    assert report["expert_count"] == 2
    assert [record["expert"] for record in report["experts"]] == [64, 208]
    for record in report["experts"]:
        merged = dest / "experts" / record["dense_endpoint_file"]
        assert sha256_file(merged) == record["dense_endpoint_file_sha256"]
        assert record["manifest_sha256"] == report["manifest_sha256"]

from __future__ import annotations

import pytest

from qsrt import constants as C
from qsrt.exl3_reference import CODEBOOK_SQG_XOR_CHEB_T12
from qsrt.pack.package_helpers import (
    DEFAULT_NONEXPERT,
    _validate_mxfp8_multimodal_specs,
)
from qsrt.pack.qsrt_allocation import (
    QSRT_ALLOCATION_KIND,
    QSRT_ALLOCATION_SCHEMA_VERSION,
)
from qsrt.pack.qsrt_model_view import (
    qsrt_atoms_v2_quantization_config,
    qsrt_hybrid_bit_map,
    qsrt_quantization_config,
)
from qsrt.qsrt import FORMAT_X4T, PHASE1_MODE_IDS


def _multimodal_mxfp8_specs() -> dict[str, tuple[str, tuple[int, ...]]]:
    weights = {
        "mm_projector.proj.0.weight",
        "mm_projector.proj.2.weight",
    }
    for block in range(27):
        prefix = f"vision_tower.encoder.blocks.{block}"
        weights.update(
            {
                f"{prefix}.wqkv.weight",
                f"{prefix}.wo.weight",
                f"{prefix}.mlp.fc0.weight",
                f"{prefix}.mlp.fc1.weight",
            }
        )
    specs: dict[str, tuple[str, tuple[int, ...]]] = {}
    for name in weights:
        specs[name] = ("F8_E4M3", (64, 96))
        specs[f"{name}_scale"] = ("U8", (64, 3))
    return specs


def _allocation() -> dict:
    layers = {}
    for layer in C.MOE_LAYERS:
        codes = [0] * C.NUM_EXPERTS
        codes[layer % C.NUM_EXPERTS] = FORMAT_X4T
        x4t = [layer % C.NUM_EXPERTS]
        layers[str(layer)] = {
            "format_codes": codes,
            "x4t": x4t,
            "compressed": [
                expert for expert in range(C.NUM_EXPERTS) if expert not in x4t
            ],
        }
    return {
        "kind": QSRT_ALLOCATION_KIND,
        "schema_version": QSRT_ALLOCATION_SCHEMA_VERSION,
        "meta": {
            "codec": "QSRT",
            "high_tier_storage": "x4t",
            "candidate_codebook": CODEBOOK_SQG_XOR_CHEB_T12,
            "candidate_mode_ids": list(PHASE1_MODE_IDS),
        },
        "layers": layers,
    }


def test_qsrt_model_view_config_is_tp_independent() -> None:
    allocation = _allocation()
    bit_map = qsrt_hybrid_bit_map(allocation)
    config = qsrt_quantization_config(allocation)

    assert bit_map["1"].count(4) == 1
    assert bit_map["1"].count(3) == C.NUM_EXPERTS - 1
    assert config["demoted_format"] == "qsrt_sqg_e4m3"
    assert config["kept_storage"] == "x4t"
    assert config["qsrt"]["codebook"] == "sqg_xor_cheb_t12"
    assert "vision_tower" not in config["ignored_layers"]
    assert "mm_projector" not in config["ignored_layers"]
    assert "tp_size" not in config


def test_qsrt_atoms_v2_model_view_is_all_qsrt_and_tp_independent() -> None:
    config = qsrt_atoms_v2_quantization_config()
    assert set(config["hybrid_bit_map"]) == {
        str(layer) for layer in C.MOE_LAYERS
    }
    assert all(
        bits == [3] * C.NUM_EXPERTS
        for bits in config["hybrid_bit_map"].values()
    )
    assert config["qsrt"] == {
        "schema": "qsrt_kimi_k3_qsrt_atoms_v2",
        "storage_format": "qsrt_atoms_v2",
        "encoding": "qsrt_sqg_e4m3",
        "codebook": "sqg_xor_cheb_t12",
        "artifact_manifest": "qsrt-manifest.json",
        "profile": "k3x22_k4x2",
    }
    assert "vision_tower" not in config["ignored_layers"]
    assert "mm_projector" not in config["ignored_layers"]
    assert "tp_size" not in config


def test_qsrt_atoms_v2_pure_k2_model_view_contract() -> None:
    config = qsrt_atoms_v2_quantization_config("k2_coupled_h512_h128")
    assert all(
        bits == [2] * C.NUM_EXPERTS
        for bits in config["hybrid_bit_map"].values()
    )
    assert config["qsrt"]["profile"] == "k2_coupled_h512_h128"


def test_default_nonexpert_overlay_includes_mxfp8_multimodal_weights() -> None:
    assert DEFAULT_NONEXPERT.name == "Kimi-K3-mxfp8-nonexpert-mm"


def test_mxfp8_multimodal_contract_accepts_complete_overlay() -> None:
    _validate_mxfp8_multimodal_specs(_multimodal_mxfp8_specs())


def test_mxfp8_multimodal_contract_rejects_bf16_weight() -> None:
    specs = _multimodal_mxfp8_specs()
    specs["mm_projector.proj.0.weight"] = ("BF16", (64, 96))
    with pytest.raises(ValueError, match="must be a rank-two E4M3 tensor"):
        _validate_mxfp8_multimodal_specs(specs)

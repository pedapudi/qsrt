"""Offline GLM-5.2 codec-distortion pilot adapter.

GLM-specific code here is limited to immutable checkpoint identities, tensor
roles and shapes, and the frozen expert panel.  Reusable fixed-average-rate
QSRT experiment machinery lives in :mod:`qsrt.qsrt_codec_pilot`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from safetensors import safe_open

from qsrt.correctness import git_state, sha256_file
from qsrt.exl3_loader import load_qsrt_encoder
from qsrt.exl3_reference import (
    CODEBOOK_SQG_CHEB_NORMAL_E4M3,
    CODEBOOK_SQG_XOR_CHEB_T12,
    decode_exl3_weight,
)
from qsrt.ldlq import SIGMA_REG, make_shared_h
from qsrt.qsrt import unpack_trellis_states
from qsrt.qsrt_codec_pilot import (
    CODEBOOK_MCG,
    FixedAverageRateGeometry,
    encode_uniform_candidate,
    encode_fixed_rate_candidates,
    select_coupled_modes,
    shared_axis_weight_energy_order,
    summarize_mode_selections,
    tensor_sha256,
)
from qsrt.sqg_e4m3 import (
    sqg_cheb_normal_e4m3_bytes,
    sqg_cheb_normal_rank_e4m3_bytes,
    sqg_xor_cheb_t12_bytes,
    sqg_xor_cheb_t12_rank_lut_bytes,
)
from qsrt.sqg_quantizer import install_sqg_quantizer
from qsrt.sqg_high_rate import (
    SQG_FP16_D3L,
    SQG_FP16_D3L_DESCRIPTOR_SHA256,
    sqg_fp16_d3l_codebook,
)


PILOT_KIND = "qsrt_glm52_k3_codec_pilot_v1"
RATE_SHIFT_PILOT_KIND = "qsrt_glm52_qsrt_codec_pilot_v1"
UNIFORM_RATE_PILOT_KIND = "qsrt_glm52_uniform_rate_codec_pilot_v1"
UNIFORM_HIGH_RATE_PILOT_KIND = "qsrt_glm52_uniform_k5_k6_codec_pilot_v1"
PILOT_SEED = "glm52-qsrt-k3-pilot-v1"
UNIFORM_RATE_PILOT_SEED = "glm52-uniform-rate-codec-pilot-v1"
UNIFORM_HIGH_RATE_BITS = (5, 6)
SOURCE_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
BASELINE_REVISION = "a350292cb2038f2c31732569a711a89e5d72fd46"
SOURCE_CONFIG_SHA256 = (
    "185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a"
)
SOURCE_INDEX_SHA256 = (
    "5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e"
)
BASELINE_MANIFEST_SHA256 = (
    "a737e9b23023c0de3ef282fceae28ef14b79056ba5cf677ed546fe089c1fec21"
)

HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 2048
EXPERTS_PER_LAYER = 256
TP_RANKS = 4
RANK_INTERMEDIATE_SIZE = INTERMEDIATE_SIZE // TP_RANKS
BITS = 3
TAILBITE_CONTEXT = 128
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_808
BASELINE_STORED_ERROR_ATOL = 1e-6
RATE_GEOMETRY = FixedAverageRateGeometry(
    axis_channels=INTERMEDIATE_SIZE,
    record_channels=128,
    tile_channels=16,
    mode_ids=(0, 1, 2),
)
SHARED_AXIS_BY_PROJECTION = {
    "gate_proj": 0,
    "up_proj": 0,
    "down_proj": 1,
}
RATE_FAMILIES = {
    "r13": ("gate_proj", "up_proj"),
    "r2": ("down_proj",),
}

PANEL: dict[int, tuple[int, ...]] = {
    3: (239, 32, 217, 254),
    10: (252, 56, 110, 8),
    16: (188, 87, 134, 7),
    23: (59, 154, 116, 130),
    30: (137, 98, 40, 80),
    37: (126, 113, 57, 51),
    43: (186, 66, 181, 225),
    50: (206, 3, 107, 17),
    57: (48, 223, 52, 252),
    64: (33, 95, 25, 194),
    70: (218, 94, 165, 0),
    77: (251, 97, 172, 15),
}


def production_codec_microbenchmark_panel(
    expert_count: int = 8,
) -> dict[int, tuple[int, ...]]:
    """Select a depth-spread prefix of the frozen real-expert panel.

    The returned experts keep the error-blind, source-pinned selection used by
    the 48-expert codec pilot.  Counts smaller than one complete twelve-layer
    round are spread from the first to the last sampled mixture-of-experts
    layer.  Larger counts add the next independently hashed expert from each
    sampled layer.  No weight or reconstruction error enters the selection.
    """

    maximum = sum(len(experts) for experts in PANEL.values())
    if (
        isinstance(expert_count, bool)
        or not isinstance(expert_count, int)
        or not 1 <= expert_count <= maximum
    ):
        raise ValueError(f"expert_count must be an integer between 1 and {maximum}")

    layers = tuple(sorted(PANEL))
    selected: dict[int, list[int]] = {}
    remaining = expert_count
    for expert_slot in range(max(len(experts) for experts in PANEL.values())):
        eligible_layers = tuple(
            layer for layer in layers if expert_slot < len(PANEL[layer])
        )
        take = min(remaining, len(eligible_layers))
        if take == 0:
            break
        if take == 1:
            positions = (len(eligible_layers) // 2,)
        else:
            positions = tuple(
                int(round(index * (len(eligible_layers) - 1) / (take - 1)))
                for index in range(take)
            )
        if len(set(positions)) != take:
            raise AssertionError("depth-spread panel selection produced duplicate layers")
        for position in positions:
            layer = eligible_layers[position]
            selected.setdefault(layer, []).append(PANEL[layer][expert_slot])
        remaining -= take

    if remaining:
        raise AssertionError("frozen panel did not contain the requested experts")
    return {
        layer: tuple(experts)
        for layer, experts in sorted(selected.items())
    }

K4_PANEL: dict[int, tuple[int, ...]] = {
    3: (51, 45, 38, 64),
    10: (112, 27, 82, 135),
    16: (111, 104, 179, 190),
    23: (255, 13, 158, 24),
    30: (253, 116, 7, 196),
    37: (164, 21, 142, 248),
    43: (63, 243, 189, 160),
    50: (154, 70, 111, 19),
    57: (214, 227, 128, 113),
    64: (183, 101, 245, 146),
    70: (244, 189, 40, 21),
    77: (152, 98, 55, 148),
}

_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ProjectionSpec:
    name: str
    index: int
    source_shape: tuple[int, int]
    encoder_shape: tuple[int, int]
    rank_encoder_shape: tuple[int, int]
    rank_trellis_shape: tuple[int, int, int]
    concat_dim: int
    expert_scale: str
    shared_scale: str


PROJECTIONS: tuple[ProjectionSpec, ...] = (
    ProjectionSpec(
        name="gate_proj",
        index=0,
        source_shape=(INTERMEDIATE_SIZE, HIDDEN_SIZE),
        encoder_shape=(HIDDEN_SIZE, INTERMEDIATE_SIZE),
        rank_encoder_shape=(HIDDEN_SIZE, RANK_INTERMEDIATE_SIZE),
        rank_trellis_shape=(HIDDEN_SIZE // 16, RANK_INTERMEDIATE_SIZE // 16, 48),
        concat_dim=0,
        expert_scale="svh",
        shared_scale="suh",
    ),
    ProjectionSpec(
        name="up_proj",
        index=1,
        source_shape=(INTERMEDIATE_SIZE, HIDDEN_SIZE),
        encoder_shape=(HIDDEN_SIZE, INTERMEDIATE_SIZE),
        rank_encoder_shape=(HIDDEN_SIZE, RANK_INTERMEDIATE_SIZE),
        rank_trellis_shape=(HIDDEN_SIZE // 16, RANK_INTERMEDIATE_SIZE // 16, 48),
        concat_dim=0,
        expert_scale="svh",
        shared_scale="suh",
    ),
    ProjectionSpec(
        name="down_proj",
        index=2,
        source_shape=(HIDDEN_SIZE, INTERMEDIATE_SIZE),
        encoder_shape=(INTERMEDIATE_SIZE, HIDDEN_SIZE),
        rank_encoder_shape=(RANK_INTERMEDIATE_SIZE, HIDDEN_SIZE),
        rank_trellis_shape=(RANK_INTERMEDIATE_SIZE // 16, HIDDEN_SIZE // 16, 48),
        concat_dim=1,
        expert_scale="suh",
        shared_scale="svh",
    ),
)


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _canonical_json_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    """Atomically write deterministic JSON without leaving a partial receipt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def panel_cells(
    panel: Mapping[int, Sequence[int]] = PANEL,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(layer), int(expert))
        for layer, experts in panel.items()
        for expert in experts
    )


def select_hashed_experts(
    eligible: Iterable[int],
    layer: int,
    count: int,
    *,
    seed: str = PILOT_SEED,
) -> tuple[int, ...]:
    """Select an error-blind panel by a frozen content-independent hash order."""

    values = tuple(int(expert) for expert in eligible)
    if count <= 0:
        raise ValueError("selection count must be positive")
    if len(set(values)) != len(values):
        raise ValueError("eligible expert IDs must be unique")
    if any(expert < 0 or expert >= EXPERTS_PER_LAYER for expert in values):
        raise ValueError("eligible expert ID is outside 0..255")
    if len(values) < count:
        raise ValueError("not enough eligible experts")

    def key(expert: int) -> bytes:
        return hashlib.sha256(
            f"{seed}:{int(layer)}:{expert}".encode("ascii")
        ).digest()

    return tuple(sorted(values, key=key)[:count])


def validate_and_select_panel(
    tier_bitmap: Mapping[str, Any],
) -> dict[int, tuple[int, ...]]:
    """Validate tier metadata and rederive the checked-in 48-expert panel."""

    return validate_and_select_rate_panel(
        tier_bitmap,
        panel=PANEL,
        eligible_rate=BITS,
        seed=PILOT_SEED,
    )


def validate_and_select_rate_panel(
    tier_bitmap: Mapping[str, Any],
    *,
    panel: Mapping[int, Sequence[int]],
    eligible_rate: int,
    seed: str,
) -> dict[int, tuple[int, ...]]:
    """Rederive any frozen, error-blind panel from checkpoint tier metadata."""

    actual: dict[int, tuple[int, ...]] = {}
    for layer, raw_expected in panel.items():
        expected = tuple(map(int, raw_expected))
        entry = tier_bitmap.get(str(layer))
        if not isinstance(entry, dict):
            raise ValueError(f"tier bitmap is missing layer {layer}")
        rates = entry.get("k")
        errors = entry.get("expert_rel_rt_mse")
        if not isinstance(rates, list) or len(rates) != EXPERTS_PER_LAYER:
            raise ValueError(f"layer {layer} has an invalid K tier vector")
        if not isinstance(errors, list) or len(errors) != EXPERTS_PER_LAYER:
            raise ValueError(f"layer {layer} has an invalid round-trip error vector")
        if any(isinstance(rate, bool) or int(rate) not in (3, 4) for rate in rates):
            raise ValueError(f"layer {layer} has a rate outside K3/K4")
        eligible = [
            expert
            for expert, rate in enumerate(rates)
            if int(rate) == eligible_rate
        ]
        selected = select_hashed_experts(
            eligible, layer, len(expected), seed=seed
        )
        if selected != expected:
            raise ValueError(
                f"layer {layer} panel drifted: selected {selected}, expected {expected}"
            )
        actual[layer] = selected
    if len(panel_cells(panel)) != 48:
        raise AssertionError("the GLM pilot panel must contain exactly 48 experts")
    return actual


def source_tensor_name(layer: int, expert: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"


def _baseline_prefix(layer: int, expert: int, projection: str, rank: int) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.rank{rank}"


def _shared_prefix(layer: int, projection: str, rank: int) -> str:
    return f"model.layers.{layer}.mlp.experts.shared_h.{projection}.rank{rank}"


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read only a safetensors header, never any tensor payload."""

    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"truncated safetensors header in {path}")
        size = int.from_bytes(raw_size, "little")
        if size <= 0 or size > path.stat().st_size - 8:
            raise ValueError(f"invalid safetensors header size in {path}")
        raw = handle.read(size)
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise TypeError(f"invalid safetensors header in {path}")
    header.pop("__metadata__", None)
    return header


def _validate_header_tensor(
    header: Mapping[str, Any],
    name: str,
    *,
    dtype: str,
    shape: Sequence[int],
) -> None:
    entry = header.get(name)
    if not isinstance(entry, dict):
        raise KeyError(f"missing tensor {name}")
    if entry.get("dtype") != dtype:
        raise TypeError(
            f"{name} has dtype {entry.get('dtype')!r}, expected {dtype!r}"
        )
    actual_shape = entry.get("shape")
    if actual_shape != list(shape):
        raise ValueError(f"{name} has shape {actual_shape}, expected {list(shape)}")


class IndexedTensorStore:
    """Header-validated streaming reader for an indexed safetensors model."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        index_path = self.root / "model.safetensors.index.json"
        index = _read_json_object(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid or empty weight map in {index_path}")
        self.weight_map: dict[str, str] = {
            str(name): str(filename) for name, filename in weight_map.items()
        }
        self._headers: dict[str, dict[str, Any]] = {}

    def filename(self, name: str) -> str:
        try:
            return self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"source index does not contain {name}") from exc

    def header(self, filename: str) -> dict[str, Any]:
        if filename not in self._headers:
            self._headers[filename] = _read_safetensors_header(self.root / filename)
        return self._headers[filename]

    def validate_tensor(
        self, name: str, *, dtype: str, shape: Sequence[int]
    ) -> None:
        filename = self.filename(name)
        _validate_header_tensor(
            self.header(filename), name, dtype=dtype, shape=shape
        )

    def get(self, name: str) -> torch.Tensor:
        filename = self.filename(name)
        with safe_open(self.root / filename, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)


def _parse_sha256_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line:
            continue
        digest, separator, name = line.partition("  ")
        if not separator or not _HEX_SHA256.fullmatch(digest) or not name:
            raise ValueError(f"malformed line {line_number} in {path}")
        if name in result:
            raise ValueError(f"duplicate manifest entry {name}")
        result[name] = digest
    if not result:
        raise ValueError(f"empty SHA-256 manifest {path}")
    return result


def _file_identity(path: Path, logical_name: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    blob = resolved.name if _HEX_SHA256.fullmatch(resolved.name) else None
    return {
        "name": logical_name,
        "size": path.stat().st_size,
        "resolved_blob_sha256": blob,
    }


def _verify_manifest_entry(path: Path, expected: str) -> str:
    resolved = path.resolve(strict=True)
    # LFS payloads are normally cached under their raw SHA-256, while small
    # Git/Xet metadata files may use a different content-addressing scheme.
    # A matching blob name closes the large-file identity without rereading
    # hundreds of GiB; otherwise fall back to hashing the actual contents.
    actual = resolved.name
    if actual != expected:
        actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")
    return actual


def validate_architecture_configs(
    source_config: Mapping[str, Any], baseline_config: Mapping[str, Any]
) -> None:
    expected = {
        "hidden_size": HIDDEN_SIZE,
        "moe_intermediate_size": INTERMEDIATE_SIZE,
        "n_routed_experts": EXPERTS_PER_LAYER,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 78,
        "first_k_dense_replace": 3,
        "num_nextn_predict_layers": 1,
        "n_shared_experts": 1,
        "hidden_act": "silu",
    }
    for label, config in (("source", source_config), ("baseline", baseline_config)):
        for field, value in expected.items():
            if config.get(field) != value:
                raise ValueError(
                    f"{label} config field {field!r} is {config.get(field)!r}, "
                    f"expected {value!r}"
                )
        if config.get("architectures") != ["GlmMoeDsaForCausalLM"]:
            raise ValueError(f"{label} config has an unexpected architecture")

    hybrid = baseline_config.get("hybrid_tr3_tail")
    if not isinstance(hybrid, dict):
        raise ValueError("baseline config is missing hybrid_tr3_tail")
    required = {
        "format": "exl3-trellis",
        "codebook": "mcg",
        "source_format": "BF16",
        "experts_per_layer": EXPERTS_PER_LAYER,
        "tp": TP_RANKS,
        "tier_bitmap": "tier_bitmap.json",
        "source_config_sha256": SOURCE_CONFIG_SHA256,
        "source_index_sha256": SOURCE_INDEX_SHA256,
    }
    for field, value in required.items():
        if hybrid.get(field) != value:
            raise ValueError(
                f"baseline hybrid field {field!r} is {hybrid.get(field)!r}, "
                f"expected {value!r}"
            )
    if hybrid.get("k_values") != [3, 4]:
        raise ValueError("baseline K values must be exactly [3, 4]")
    if hybrid.get("moe_layers") != [3, 78]:
        raise ValueError("baseline MoE layer range must be 3..78")
    if int(hybrid.get("mcg_multiplier", -1)) != 0xCBAC1FED:
        raise ValueError("baseline MCG multiplier identity changed")


def _validate_baseline_header(
    header: Mapping[str, Any],
    layer: int,
    experts: Sequence[int],
    *,
    bits: int = BITS,
) -> None:
    if bits not in (3, 4):
        raise ValueError("the materialized GLM comparison contains only K3/K4")
    for spec in PROJECTIONS:
        for rank in range(TP_RANKS):
            shared = _shared_prefix(layer, spec.name, rank)
            shared_length = (
                spec.rank_encoder_shape[0]
                if spec.shared_scale == "suh"
                else spec.rank_encoder_shape[1]
            )
            _validate_header_tensor(
                header,
                f"{shared}.{spec.shared_scale}",
                dtype="F16",
                shape=(shared_length,),
            )
            for expert in experts:
                prefix = _baseline_prefix(layer, expert, spec.name, rank)
                expert_length = (
                    spec.rank_encoder_shape[0]
                    if spec.expert_scale == "suh"
                    else spec.rank_encoder_shape[1]
                )
                _validate_header_tensor(
                    header,
                    f"{prefix}.trellis",
                    dtype="I16",
                    shape=(
                        spec.rank_trellis_shape[0],
                        spec.rank_trellis_shape[1],
                        16 * bits,
                    ),
                )
                _validate_header_tensor(
                    header,
                    f"{prefix}.{spec.expert_scale}",
                    dtype="F16",
                    shape=(expert_length,),
                )
                _validate_header_tensor(
                    header, f"{prefix}.mcg", dtype="I32", shape=()
                )


def validate_inventory(
    source_root: Path,
    baseline_root: Path,
    *,
    panel: Mapping[int, Sequence[int]] = PANEL,
    panel_rate: int = BITS,
    panel_seed: str = PILOT_SEED,
) -> dict[str, Any]:
    """Close immutable identities and all selected tensor headers."""

    source_root = source_root.resolve()
    baseline_root = baseline_root.resolve()
    source_config_path = source_root / "config.json"
    source_index_path = source_root / "model.safetensors.index.json"
    baseline_config_path = baseline_root / "config.json"
    baseline_manifest_path = baseline_root / "MANIFEST.sha256"
    tier_path = baseline_root / "tier_bitmap.json"
    for path in (
        source_config_path,
        source_index_path,
        baseline_config_path,
        baseline_manifest_path,
        tier_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(source_config_path) != SOURCE_CONFIG_SHA256:
        raise ValueError("official GLM-5.2 config identity mismatch")
    if sha256_file(source_index_path) != SOURCE_INDEX_SHA256:
        raise ValueError("official GLM-5.2 tensor index identity mismatch")
    if sha256_file(baseline_manifest_path) != BASELINE_MANIFEST_SHA256:
        raise ValueError("comparison checkpoint manifest identity mismatch")

    source_config = _read_json_object(source_config_path)
    baseline_config = _read_json_object(baseline_config_path)
    validate_architecture_configs(source_config, baseline_config)

    manifest = _parse_sha256_manifest(baseline_manifest_path)
    missing_manifest_files = [
        name for name in manifest if not (baseline_root / name).is_file()
    ]
    if missing_manifest_files:
        raise FileNotFoundError(
            f"comparison checkpoint is missing {len(missing_manifest_files)} files"
        )
    consumed_baseline_names = {
        "config.json",
        "tier_bitmap.json",
        *(f"model-layer-{layer:03d}.safetensors" for layer in panel),
    }
    for name in sorted(consumed_baseline_names):
        try:
            digest = manifest[name]
        except KeyError as exc:
            raise KeyError(f"comparison manifest does not contain {name}") from exc
        _verify_manifest_entry(baseline_root / name, digest)

    tier_bitmap = _read_json_object(tier_path)
    validate_and_select_rate_panel(
        tier_bitmap,
        panel=panel,
        eligible_rate=panel_rate,
        seed=panel_seed,
    )
    source = IndexedTensorStore(source_root)
    missing_source_shards = sorted(
        filename
        for filename in set(source.weight_map.values())
        if not (source_root / filename).is_file()
    )
    if missing_source_shards:
        raise FileNotFoundError(
            f"official checkpoint is missing {len(missing_source_shards)} shards"
        )

    selected_source_files: set[str] = set()
    baseline_files: list[dict[str, Any]] = []
    for layer, raw_experts in panel.items():
        experts = tuple(map(int, raw_experts))
        baseline_name = f"model-layer-{layer:03d}.safetensors"
        baseline_path = baseline_root / baseline_name
        header = _read_safetensors_header(baseline_path)
        _validate_baseline_header(header, layer, experts, bits=panel_rate)
        baseline_files.append(_file_identity(baseline_path, baseline_name))
        for expert in experts:
            for spec in PROJECTIONS:
                name = source_tensor_name(layer, expert, spec.name)
                source.validate_tensor(name, dtype="BF16", shape=spec.source_shape)
                selected_source_files.add(source.filename(name))

    return {
        "source": {
            "declared_revision": SOURCE_REVISION,
            "root": str(source_root),
            "config_sha256": SOURCE_CONFIG_SHA256,
            "index_sha256": SOURCE_INDEX_SHA256,
            "indexed_tensor_count": len(source.weight_map),
            "indexed_shard_count": len(set(source.weight_map.values())),
            "selected_shards": [
                _file_identity(source_root / name, name)
                for name in sorted(selected_source_files)
            ],
        },
        "baseline": {
            "declared_revision": BASELINE_REVISION,
            "root": str(baseline_root),
            "manifest_sha256": BASELINE_MANIFEST_SHA256,
            "manifest_entry_count": len(manifest),
            "selected_layer_files": baseline_files,
        },
    }


def metric_terms(
    source: torch.Tensor, reconstruction: torch.Tensor
) -> tuple[float, float]:
    """Return source energy and SSE with FP64 reductions."""

    if source.shape != reconstruction.shape:
        raise ValueError(
            f"metric shape mismatch: {tuple(source.shape)} != "
            f"{tuple(reconstruction.shape)}"
        )
    source_f = source.float()
    reconstruction_f = reconstruction.float()
    if not bool(torch.isfinite(source_f).all()):
        raise ValueError("source tensor contains non-finite values")
    if not bool(torch.isfinite(reconstruction_f).all()):
        raise ValueError("reconstruction contains non-finite values")
    energy = float(source_f.square().sum(dtype=torch.float64).item())
    sse = float(
        (reconstruction_f - source_f).square().sum(dtype=torch.float64).item()
    )
    if not math.isfinite(energy) or energy <= 0.0:
        raise ValueError("source tensor has invalid energy")
    if not math.isfinite(sse) or sse < 0.0:
        raise ValueError("reconstruction has invalid SSE")
    return energy, sse


def _validate_runtime_tensor(
    value: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    shape: Sequence[int],
) -> None:
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(
            f"{name} has shape {tuple(value.shape)}, expected {tuple(shape)}"
        )
    if not value.is_contiguous():
        raise ValueError(f"{name} is not contiguous")


@torch.no_grad()
def decode_baseline_projection(
    handle: Any,
    *,
    layer: int,
    expert: int,
    spec: ProjectionSpec,
    device: torch.device,
    quantizer_module: Any,
    bits: int = BITS,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Decode and join the exact TP4 MCG artifact for one projection."""

    pieces: list[torch.Tensor] = []
    trellis_bytes = 0
    expert_scale_bytes = 0
    shared_scale_bytes = 0
    marker_bytes = 0
    markers: list[int] = []
    for rank in range(TP_RANKS):
        prefix = _baseline_prefix(layer, expert, spec.name, rank)
        shared = _shared_prefix(layer, spec.name, rank)
        trellis = handle.get_tensor(f"{prefix}.trellis").to(device).contiguous()
        expert_scale = (
            handle.get_tensor(f"{prefix}.{spec.expert_scale}")
            .to(device)
            .contiguous()
        )
        shared_scale = (
            handle.get_tensor(f"{shared}.{spec.shared_scale}")
            .to(device)
            .contiguous()
        )
        marker = handle.get_tensor(f"{prefix}.mcg")
        _validate_runtime_tensor(
            trellis,
            name=f"{prefix}.trellis",
            dtype=torch.int16,
            shape=(
                spec.rank_trellis_shape[0],
                spec.rank_trellis_shape[1],
                16 * bits,
            ),
        )
        expected_expert_length = (
            spec.rank_encoder_shape[0]
            if spec.expert_scale == "suh"
            else spec.rank_encoder_shape[1]
        )
        expected_shared_length = (
            spec.rank_encoder_shape[0]
            if spec.shared_scale == "suh"
            else spec.rank_encoder_shape[1]
        )
        _validate_runtime_tensor(
            expert_scale,
            name=f"{prefix}.{spec.expert_scale}",
            dtype=torch.float16,
            shape=(expected_expert_length,),
        )
        _validate_runtime_tensor(
            shared_scale,
            name=f"{shared}.{spec.shared_scale}",
            dtype=torch.float16,
            shape=(expected_shared_length,),
        )
        if marker.dtype != torch.int32 or marker.ndim != 0:
            raise TypeError(f"{prefix}.mcg must be a scalar I32 tensor")
        marker_unsigned = int(marker.item()) & 0xFFFF_FFFF
        if marker_unsigned != 0xCBAC1FED:
            raise ValueError(f"{prefix}.mcg has an unexpected multiplier")
        markers.append(marker_unsigned)

        suh = shared_scale if spec.shared_scale == "suh" else expert_scale
        svh = shared_scale if spec.shared_scale == "svh" else expert_scale
        weight = torch.empty(
            spec.rank_encoder_shape, dtype=torch.float16, device=device
        )
        quantizer_module.ext.reconstruct(weight, trellis, bits, True, False)
        weight = quantizer_module.preapply_had_l(weight, 128)
        weight *= suh.unsqueeze(1)
        weight = quantizer_module.preapply_had_r(weight, 128)
        weight *= svh.unsqueeze(0)
        pieces.append(weight.T.contiguous().cpu())

        trellis_bytes += trellis.numel() * trellis.element_size()
        expert_scale_bytes += expert_scale.numel() * expert_scale.element_size()
        shared_scale_bytes += shared_scale.numel() * shared_scale.element_size()
        marker_bytes += marker.numel() * marker.element_size()

    reconstruction = torch.cat(pieces, dim=spec.concat_dim).contiguous()
    if tuple(reconstruction.shape) != spec.source_shape:
        raise ValueError(
            f"joined baseline {spec.name} has shape {tuple(reconstruction.shape)}, "
            f"expected {spec.source_shape}"
        )
    weight_count = math.prod(spec.source_shape)
    trellis_bpw = trellis_bytes * 8.0 / weight_count
    if trellis_bpw != float(bits):
        raise ValueError(f"baseline {spec.name} trellis rate is {trellis_bpw} bpw")
    return reconstruction, {
        "codebook": "mcg",
        "tp_ranks": TP_RANKS,
        "trellis_bytes": trellis_bytes,
        "trellis_bpw": trellis_bpw,
        "expert_scale_bytes": expert_scale_bytes,
        "shared_scale_bytes_physical": shared_scale_bytes,
        "shared_scale_bytes_amortized_per_expert": shared_scale_bytes
        / EXPERTS_PER_LAYER,
        "marker_bytes": marker_bytes,
        "mcg_multiplier": markers[0],
    }


def _transform_seeds(layer: int, spec: ProjectionSpec) -> tuple[int, int]:
    input_seed = int(layer) * 1_000_000 + spec.index
    return input_seed, input_seed + 499_979


@torch.no_grad()
def encode_sqg_projection(
    source: torch.Tensor,
    *,
    layer: int,
    expert: int,
    spec: ProjectionSpec,
    device: torch.device,
    quantizer_module: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Uniform-K3 encode followed by an independent stored-payload decode."""

    if source.dtype != torch.bfloat16 or tuple(source.shape) != spec.source_shape:
        raise TypeError(
            f"official {spec.name} must be BF16 {spec.source_shape}, got "
            f"{source.dtype} {tuple(source.shape)}"
        )
    weight = source.T.float().contiguous().to(device)
    shared_h = make_shared_h(weight.shape[0], device)
    input_seed, output_seed = _transform_seeds(layer, spec)
    quant_args: dict[str, Any] = {
        "K": BITS,
        "seed": input_seed,
        "sv_seed": output_seed,
        "sigma_reg": SIGMA_REG,
        "devices": [str(device)],
        "device_ratios": None,
        "apply_out_scales": False,
        "ldlq_tf32": True,
        "tailbite_context": TAILBITE_CONTEXT,
        "sqg_e4m3_lut": sqg_xor_cheb_t12_bytes(BITS),
    }
    if spec.name in ("gate_proj", "up_proj"):
        # This pilot measures the numerical endpoint, so the scale profile is
        # tensor-local. The unique key enables the production placement of the
        # fitted global scale in sv without sharing it with another expert.
        quant_args["shared_input_scales_key"] = (
            PILOT_KIND,
            layer,
            expert,
            spec.name,
        )
        quant_args["g_scale_into_sv"] = True

    torch.cuda.synchronize(device)
    started = time.monotonic()
    _, proxy_error, tensors = quantizer_module.quantize_qsrt(
        weight,
        shared_h,
        quant_args,
        True,
        progress_str="",
    )
    torch.cuda.synchronize(device)
    encode_seconds = time.monotonic() - started
    trellis = tensors["trellis"].contiguous()
    suh = tensors["suh"].contiguous()
    svh = tensors["svh"].contiguous()
    expected_trellis_shape = (
        spec.encoder_shape[0] // 16,
        spec.encoder_shape[1] // 16,
        48,
    )
    _validate_runtime_tensor(
        trellis,
        name=f"SQG {spec.name} trellis",
        dtype=torch.int16,
        shape=expected_trellis_shape,
    )
    _validate_runtime_tensor(
        suh,
        name=f"SQG {spec.name} suh",
        dtype=torch.float16,
        shape=(spec.encoder_shape[0],),
    )
    _validate_runtime_tensor(
        svh,
        name=f"SQG {spec.name} svh",
        dtype=torch.float16,
        shape=(spec.encoder_shape[1],),
    )
    states = unpack_trellis_states(trellis, BITS)
    stored = decode_exl3_weight(
        states,
        suh,
        svh,
        codebook=CODEBOOK_SQG_XOR_CHEB_T12,
        bits=BITS,
    )
    # EXL's actual reconstruction endpoint is FP16. Use the same endpoint for
    # SQG before comparing either codec with the official BF16 tensor.
    reconstruction = stored.half().T.contiguous().cpu()
    if tuple(reconstruction.shape) != spec.source_shape:
        raise ValueError(
            f"SQG {spec.name} has shape {tuple(reconstruction.shape)}, "
            f"expected {spec.source_shape}"
        )
    trellis_bytes = trellis.numel() * trellis.element_size()
    weight_count = math.prod(spec.source_shape)
    trellis_bpw = trellis_bytes * 8.0 / weight_count
    if trellis_bpw != 3.0:
        raise ValueError(f"SQG {spec.name} trellis rate is {trellis_bpw} bpw")
    return reconstruction, {
        "profile": "qsrt_sqg_e4m3",
        "codebook": CODEBOOK_SQG_XOR_CHEB_T12,
        "rate": BITS,
        "trellis_bytes": trellis_bytes,
        "trellis_bpw": trellis_bpw,
        "scale_bytes": suh.numel() * suh.element_size()
        + svh.numel() * svh.element_size(),
        "proxy_relative_error": float(proxy_error),
        "encode_seconds": encode_seconds,
        "input_sign_seed": input_seed,
        "output_sign_seed": output_seed,
    }


def run_expert(
    *,
    source: IndexedTensorStore,
    baseline_handle: Any,
    tier_entry: Mapping[str, Any],
    layer: int,
    expert: int,
    device: torch.device,
    quantizer_module: Any,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Measure one complete gate/up/down expert and close the stored baseline."""

    rates = tier_entry.get("k")
    stored_errors = tier_entry.get("expert_rel_rt_mse")
    if not isinstance(rates, list) or int(rates[expert]) != BITS:
        raise ValueError(f"layer {layer} expert {expert} is not a whole-expert K3 cell")
    if not isinstance(stored_errors, list):
        raise ValueError(f"layer {layer} has no stored round-trip error vector")

    projection_records: dict[str, Any] = {}
    total_energy = 0.0
    total_baseline_sse = 0.0
    total_sqg_sse = 0.0
    for spec in PROJECTIONS:
        name = source_tensor_name(layer, expert, spec.name)
        source_weight = source.get(name)
        if source_weight.dtype != torch.bfloat16:
            raise TypeError(f"{name} is {source_weight.dtype}, expected BF16")
        if tuple(source_weight.shape) != spec.source_shape:
            raise ValueError(
                f"{name} has shape {tuple(source_weight.shape)}, "
                f"expected {spec.source_shape}"
            )
        baseline_weight, baseline_payload = decode_baseline_projection(
            baseline_handle,
            layer=layer,
            expert=expert,
            spec=spec,
            device=device,
            quantizer_module=quantizer_module,
        )
        sqg_weight, sqg_payload = encode_sqg_projection(
            source_weight,
            layer=layer,
            expert=expert,
            spec=spec,
            device=device,
            quantizer_module=quantizer_module,
        )
        energy, baseline_sse = metric_terms(source_weight, baseline_weight)
        sqg_energy, sqg_sse = metric_terms(source_weight, sqg_weight)
        if sqg_energy != energy:
            raise AssertionError("source energy changed between codec arms")
        baseline_relative = baseline_sse / energy
        sqg_relative = sqg_sse / energy
        ratio = sqg_relative / baseline_relative
        projection_records[spec.name] = {
            "source_energy": energy,
            "baseline_sse": baseline_sse,
            "baseline_relative_sse": baseline_relative,
            "sqg_sse": sqg_sse,
            "sqg_relative_sse": sqg_relative,
            "sqg_over_baseline": ratio,
            "relative_reduction": 1.0 - ratio,
            "baseline_payload": baseline_payload,
            "sqg_payload": sqg_payload,
        }
        total_energy += energy
        total_baseline_sse += baseline_sse
        total_sqg_sse += sqg_sse
        del source_weight, baseline_weight, sqg_weight

    baseline_relative = total_baseline_sse / total_energy
    sqg_relative = total_sqg_sse / total_energy
    ratio = sqg_relative / baseline_relative
    stored_error = float(stored_errors[expert])
    stored_error_abs_delta = abs(baseline_relative - stored_error)
    if stored_error_abs_delta > BASELINE_STORED_ERROR_ATOL:
        raise ValueError(
            f"layer {layer} expert {expert} baseline decode failed stored-error "
            f"closure: {baseline_relative} vs {stored_error}"
        )
    return {
        "kind": f"{PILOT_KIND}_expert",
        "manifest_sha256": manifest_sha256,
        "complete": True,
        "layer": layer,
        "expert": expert,
        "rate": BITS,
        "source_energy": total_energy,
        "baseline_sse": total_baseline_sse,
        "baseline_relative_sse": baseline_relative,
        "baseline_stored_relative_sse": stored_error,
        "baseline_stored_error_abs_delta": stored_error_abs_delta,
        "sqg_sse": total_sqg_sse,
        "sqg_relative_sse": sqg_relative,
        "sqg_over_baseline": ratio,
        "relative_reduction": 1.0 - ratio,
        "projections": projection_records,
    }


def _validate_selected_k3_cell(
    tier_entry: Mapping[str, Any], *, layer: int, expert: int
) -> float:
    rates = tier_entry.get("k")
    stored_errors = tier_entry.get("expert_rel_rt_mse")
    if not isinstance(rates, list) or int(rates[expert]) != BITS:
        raise ValueError(f"layer {layer} expert {expert} is not a whole-expert K3 cell")
    if not isinstance(stored_errors, list):
        raise ValueError(f"layer {layer} has no stored round-trip error vector")
    return float(stored_errors[expert])


@torch.no_grad()
def run_rate_shift_expert(
    *,
    source: IndexedTensorStore,
    baseline_handle: Any,
    tier_entry: Mapping[str, Any],
    layer: int,
    expert: int,
    device: torch.device,
    quantizer_module: Any,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Measure one expert using generic fixed-average-rate QSRT candidates."""

    stored_error = _validate_selected_k3_cell(
        tier_entry, layer=layer, expert=expert
    )
    source_weights: dict[str, torch.Tensor] = {}
    for spec in PROJECTIONS:
        name = source_tensor_name(layer, expert, spec.name)
        weight = source.get(name)
        if weight.dtype != torch.bfloat16 or tuple(weight.shape) != spec.source_shape:
            raise TypeError(
                f"{name} must be BF16 {spec.source_shape}, got "
                f"{weight.dtype} {tuple(weight.shape)}"
            )
        source_weights[spec.name] = weight

    permutation, group_energy = shared_axis_weight_energy_order(
        [
            (source_weights[spec.name], SHARED_AXIS_BY_PROJECTION[spec.name])
            for spec in PROJECTIONS
        ],
        group_channels=4,
    )
    projection_records: dict[str, Any] = {}
    matrix_sse: dict[str, dict[int, float]] = {}
    total_energy = 0.0
    total_baseline_sse = 0.0
    for spec in PROJECTIONS:
        source_weight = source_weights[spec.name]
        baseline_weight, baseline_payload = decode_baseline_projection(
            baseline_handle,
            layer=layer,
            expert=expert,
            spec=spec,
            device=device,
            quantizer_module=quantizer_module,
        )
        energy, baseline_sse = metric_terms(source_weight, baseline_weight)
        input_seed, output_seed = _transform_seeds(layer, spec)
        shared_axis = SHARED_AXIS_BY_PROJECTION[spec.name]
        candidates = encode_fixed_rate_candidates(
            source_weight,
            shared_axis=shared_axis,
            permutation=permutation,
            geometry=RATE_GEOMETRY,
            device=device,
            quantizer_module=quantizer_module,
            input_sign_seed=input_seed,
            output_sign_seed=output_seed,
            scale_scope_key=(RATE_SHIFT_PILOT_KIND, layer, expert, spec.name)
            if shared_axis == 0
            else None,
            g_scale_into_sv=shared_axis == 0,
            sigma_reg=SIGMA_REG,
            tailbite_context=TAILBITE_CONTEXT,
            ldlq_tf32=True,
        )
        candidate_records: dict[str, Any] = {}
        mode_sse: dict[int, float] = {}
        for candidate in candidates:
            mode_id = int(candidate["mode_id"])
            reconstruction = candidate.pop("reconstruction")
            candidate_energy, candidate_sse = metric_terms(
                source_weight, reconstruction
            )
            if candidate_energy != energy:
                raise AssertionError("source energy changed between codec candidates")
            mode_sse[mode_id] = candidate_sse
            candidate_records[f"R{mode_id}"] = {
                "sse": candidate_sse,
                "relative_sse": candidate_sse / energy,
                "payload": candidate["payload"],
            }
            del reconstruction
        matrix_sse[spec.name] = mode_sse
        projection_records[spec.name] = {
            "source_energy": energy,
            "baseline_sse": baseline_sse,
            "baseline_relative_sse": baseline_sse / energy,
            "baseline_payload": baseline_payload,
            "candidates": candidate_records,
        }
        total_energy += energy
        total_baseline_sse += baseline_sse
        del baseline_weight

    rate_selection = select_coupled_modes(matrix_sse, RATE_FAMILIES)
    family_for_matrix = {
        matrix: family
        for family, matrices in RATE_FAMILIES.items()
        for matrix in matrices
    }
    total_sqg_sse = 0.0
    total_r0_sse = 0.0
    for spec in PROJECTIONS:
        projection = projection_records[spec.name]
        family = family_for_matrix[spec.name]
        selected_mode = int(rate_selection[family]["selected_mode"])
        selected = projection["candidates"][f"R{selected_mode}"]
        r0 = projection["candidates"]["R0"]
        energy = float(projection["source_energy"])
        baseline_sse = float(projection["baseline_sse"])
        projection.update(
            {
                "selected_mode": selected_mode,
                "sqg_sse": selected["sse"],
                "sqg_relative_sse": selected["relative_sse"],
                "sqg_over_baseline": selected["sse"] / baseline_sse,
                "sqg_r0_sse": r0["sse"],
                "sqg_r0_relative_sse": r0["relative_sse"],
                "sqg_r0_over_baseline": r0["sse"] / baseline_sse,
                "source_energy_check": energy,
            }
        )
        total_sqg_sse += float(selected["sse"])
        total_r0_sse += float(r0["sse"])

    for family in rate_selection.values():
        selected_mode = int(family["selected_mode"])
        r0_sse = float(family["sse_by_mode"]["R0"])
        selected_sse = float(family["sse_by_mode"][f"R{selected_mode}"])
        family["selected_sse"] = selected_sse
        family["r0_sse"] = r0_sse
        family["relative_reduction_from_r0"] = 1.0 - selected_sse / r0_sse

    baseline_relative = total_baseline_sse / total_energy
    sqg_relative = total_sqg_sse / total_energy
    r0_relative = total_r0_sse / total_energy
    stored_error_abs_delta = abs(baseline_relative - stored_error)
    if stored_error_abs_delta > BASELINE_STORED_ERROR_ATOL:
        raise ValueError(
            f"layer {layer} expert {expert} baseline decode failed stored-error "
            f"closure: {baseline_relative} vs {stored_error}"
        )
    return {
        "kind": f"{RATE_SHIFT_PILOT_KIND}_expert",
        "manifest_sha256": manifest_sha256,
        "complete": True,
        "layer": layer,
        "expert": expert,
        "rate": BITS,
        "source_energy": total_energy,
        "baseline_sse": total_baseline_sse,
        "baseline_relative_sse": baseline_relative,
        "baseline_stored_relative_sse": stored_error,
        "baseline_stored_error_abs_delta": stored_error_abs_delta,
        "sqg_sse": total_sqg_sse,
        "sqg_relative_sse": sqg_relative,
        "sqg_over_baseline": total_sqg_sse / total_baseline_sse,
        "relative_reduction": 1.0 - total_sqg_sse / total_baseline_sse,
        "sqg_r0_sse": total_r0_sse,
        "sqg_r0_relative_sse": r0_relative,
        "sqg_r0_over_baseline": total_r0_sse / total_baseline_sse,
        "rate_selection": rate_selection,
        "ordering": {
            "policy": "coupled_source_weight_energy",
            "group_channels": 4,
            "permutation_sha256": tensor_sha256(permutation),
            "group_energy_sha256": tensor_sha256(group_energy),
            "group_energy_min": float(group_energy.min().item()),
            "group_energy_max": float(group_energy.max().item()),
        },
        "projections": projection_records,
    }


@torch.no_grad()
def run_uniform_rate_expert(
    *,
    source: IndexedTensorStore,
    baseline_handle: Any,
    tier_entry: Mapping[str, Any],
    layer: int,
    expert: int,
    device: torch.device,
    quantizer_module: Any,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Measure fresh K2 MCG/SQG and artifact-MCG/SQG K4 for one expert."""

    rates = tier_entry.get("k")
    stored_errors = tier_entry.get("expert_rel_rt_mse")
    if not isinstance(rates, list) or int(rates[expert]) != 4:
        raise ValueError(f"layer {layer} expert {expert} is not a real K4 cell")
    if not isinstance(stored_errors, list):
        raise ValueError(f"layer {layer} has no stored round-trip error vector")
    stored_k4_error = float(stored_errors[expert])

    rate_records: dict[str, dict[str, Any]] = {
        "K2": {
            "rate": 2,
            "baseline_origin": "fresh_offline_mcg_encode",
            "projections": {},
            "source_energy": 0.0,
            "baseline_sse": 0.0,
            "sqg_sse": 0.0,
        },
        "K4": {
            "rate": 4,
            "baseline_origin": "materialized_willfalco_artifact",
            "projections": {},
            "source_energy": 0.0,
            "baseline_sse": 0.0,
            "sqg_sse": 0.0,
        },
    }
    for spec in PROJECTIONS:
        name = source_tensor_name(layer, expert, spec.name)
        source_weight = source.get(name)
        if source_weight.dtype != torch.bfloat16 or tuple(source_weight.shape) != spec.source_shape:
            raise TypeError(
                f"{name} must be BF16 {spec.source_shape}, got "
                f"{source_weight.dtype} {tuple(source_weight.shape)}"
            )
        input_seed, output_seed = _transform_seeds(layer, spec)
        k2_mcg = encode_uniform_candidate(
            source_weight,
            bits=2,
            codebook=CODEBOOK_MCG,
            device=device,
            quantizer_module=quantizer_module,
            input_sign_seed=input_seed,
            output_sign_seed=output_seed,
            scale_scope_key=(
                UNIFORM_RATE_PILOT_KIND,
                "mcg",
                2,
                layer,
                expert,
                spec.name,
            )
            if spec.name in ("gate_proj", "up_proj")
            else None,
            g_scale_into_sv=spec.name in ("gate_proj", "up_proj"),
            sigma_reg=SIGMA_REG,
            tailbite_context=TAILBITE_CONTEXT,
            ldlq_tf32=True,
        )
        k2_sqg = encode_uniform_candidate(
            source_weight,
            bits=2,
            codebook=CODEBOOK_SQG_XOR_CHEB_T12,
            device=device,
            quantizer_module=quantizer_module,
            input_sign_seed=input_seed,
            output_sign_seed=output_seed,
            scale_scope_key=(
                UNIFORM_RATE_PILOT_KIND,
                "sqg",
                2,
                layer,
                expert,
                spec.name,
            )
            if spec.name in ("gate_proj", "up_proj")
            else None,
            g_scale_into_sv=spec.name in ("gate_proj", "up_proj"),
            sigma_reg=SIGMA_REG,
            tailbite_context=TAILBITE_CONTEXT,
            ldlq_tf32=True,
        )
        k4_artifact_weight, k4_artifact_payload = decode_baseline_projection(
            baseline_handle,
            layer=layer,
            expert=expert,
            spec=spec,
            device=device,
            quantizer_module=quantizer_module,
            bits=4,
        )
        k4_sqg = encode_uniform_candidate(
            source_weight,
            bits=4,
            codebook=CODEBOOK_SQG_XOR_CHEB_T12,
            device=device,
            quantizer_module=quantizer_module,
            input_sign_seed=input_seed,
            output_sign_seed=output_seed,
            scale_scope_key=(
                UNIFORM_RATE_PILOT_KIND,
                "sqg",
                4,
                layer,
                expert,
                spec.name,
            )
            if spec.name in ("gate_proj", "up_proj")
            else None,
            g_scale_into_sv=spec.name in ("gate_proj", "up_proj"),
            sigma_reg=SIGMA_REG,
            tailbite_context=TAILBITE_CONTEXT,
            ldlq_tf32=True,
        )

        arms = {
            "K2": (
                k2_mcg.pop("reconstruction"),
                k2_mcg["payload"],
                k2_sqg.pop("reconstruction"),
                k2_sqg["payload"],
            ),
            "K4": (
                k4_artifact_weight,
                k4_artifact_payload,
                k4_sqg.pop("reconstruction"),
                k4_sqg["payload"],
            ),
        }
        reference_energy: float | None = None
        for label, (
            baseline_weight,
            baseline_payload,
            sqg_weight,
            sqg_payload,
        ) in arms.items():
            energy, baseline_sse = metric_terms(source_weight, baseline_weight)
            sqg_energy, sqg_sse = metric_terms(source_weight, sqg_weight)
            if sqg_energy != energy:
                raise AssertionError("source energy changed between uniform codecs")
            if reference_energy is None:
                reference_energy = energy
            elif reference_energy != energy:
                raise AssertionError("source energy changed between rates")
            projection = {
                "source_energy": energy,
                "baseline_sse": baseline_sse,
                "baseline_relative_sse": baseline_sse / energy,
                "sqg_sse": sqg_sse,
                "sqg_relative_sse": sqg_sse / energy,
                "sqg_over_baseline": sqg_sse / baseline_sse,
                "relative_reduction": 1.0 - sqg_sse / baseline_sse,
                "baseline_payload": baseline_payload,
                "sqg_payload": sqg_payload,
            }
            rate_records[label]["projections"][spec.name] = projection
            rate_records[label]["source_energy"] += energy
            rate_records[label]["baseline_sse"] += baseline_sse
            rate_records[label]["sqg_sse"] += sqg_sse
            del baseline_weight, sqg_weight
        del arms
        del source_weight

    for label, rate_record in rate_records.items():
        energy = float(rate_record["source_energy"])
        baseline_sse = float(rate_record["baseline_sse"])
        sqg_sse = float(rate_record["sqg_sse"])
        rate_record.update(
            {
                "baseline_relative_sse": baseline_sse / energy,
                "sqg_relative_sse": sqg_sse / energy,
                "sqg_over_baseline": sqg_sse / baseline_sse,
                "relative_reduction": 1.0 - sqg_sse / baseline_sse,
            }
        )
    k4_delta = abs(
        float(rate_records["K4"]["baseline_relative_sse"]) - stored_k4_error
    )
    if k4_delta > BASELINE_STORED_ERROR_ATOL:
        raise ValueError(
            f"layer {layer} expert {expert} K4 artifact failed stored-error "
            f"closure: {rate_records['K4']['baseline_relative_sse']} vs "
            f"{stored_k4_error}"
        )
    rate_records["K4"]["baseline_stored_relative_sse"] = stored_k4_error
    rate_records["K4"]["baseline_stored_error_abs_delta"] = k4_delta
    return {
        "kind": f"{UNIFORM_RATE_PILOT_KIND}_expert",
        "manifest_sha256": manifest_sha256,
        "complete": True,
        "layer": layer,
        "expert": expert,
        "rates": rate_records,
    }


@torch.no_grad()
def run_fresh_uniform_rate_expert(
    *,
    source: IndexedTensorStore,
    layer: int,
    expert: int,
    bits: Sequence[int],
    device: torch.device,
    quantizer_module: Any,
    manifest_sha256: str,
    pilot_kind: str = UNIFORM_HIGH_RATE_PILOT_KIND,
    sqg_codebook: str = CODEBOOK_SQG_XOR_CHEB_T12,
    sqg_codebook_values_by_bits: Mapping[int, torch.Tensor] | None = None,
    return_trellis_diagnostics: bool = False,
) -> dict[str, Any]:
    """Measure freshly encoded MCG and SQG at arbitrary uniform pilot rates."""

    rates = tuple(int(value) for value in bits)
    if (
        not rates
        or tuple(sorted(set(rates))) != rates
        or any(value not in range(2, 7) for value in rates)
    ):
        raise ValueError("fresh uniform rates must be unique ordered K2 through K6")
    if sqg_codebook_values_by_bits is not None and set(
        sqg_codebook_values_by_bits
    ) != set(rates):
        raise ValueError("custom SQG codebook tables must cover every requested rate")
    rate_records: dict[str, dict[str, Any]] = {
        f"K{rate}": {
            "rate": rate,
            "baseline_origin": "fresh_offline_mcg_encode",
            "projections": {},
            "source_energy": 0.0,
            "baseline_sse": 0.0,
            "sqg_sse": 0.0,
        }
        for rate in rates
    }
    for spec in PROJECTIONS:
        name = source_tensor_name(layer, expert, spec.name)
        source_weight = source.get(name)
        if source_weight.dtype != torch.bfloat16 or tuple(source_weight.shape) != spec.source_shape:
            raise TypeError(
                f"{name} must be BF16 {spec.source_shape}, got "
                f"{source_weight.dtype} {tuple(source_weight.shape)}"
            )
        input_seed, output_seed = _transform_seeds(layer, spec)
        reference_energy: float | None = None
        for rate in rates:
            arms = {}
            for arm, codebook in (
                ("baseline", CODEBOOK_MCG),
                ("sqg", sqg_codebook),
            ):
                encoded = encode_uniform_candidate(
                    source_weight,
                    bits=rate,
                    codebook=codebook,
                    device=device,
                    quantizer_module=quantizer_module,
                    input_sign_seed=input_seed,
                    output_sign_seed=output_seed,
                    scale_scope_key=(
                        pilot_kind,
                        arm,
                        rate,
                        layer,
                        expert,
                        spec.name,
                    )
                    if spec.name in ("gate_proj", "up_proj")
                    else None,
                    g_scale_into_sv=spec.name in ("gate_proj", "up_proj"),
                    sigma_reg=SIGMA_REG,
                    tailbite_context=TAILBITE_CONTEXT,
                    ldlq_tf32=True,
                    codebook_values=(
                        sqg_codebook_values_by_bits[rate]
                        if arm == "sqg" and sqg_codebook_values_by_bits is not None
                        else None
                    ),
                    return_trellis_diagnostics=(
                        return_trellis_diagnostics and arm == "sqg"
                    ),
                )
                arms[arm] = (
                    encoded.pop("reconstruction"),
                    encoded["payload"],
                    encoded.get("trellis_diagnostics"),
                )
            baseline_weight, baseline_payload, baseline_diagnostics = arms["baseline"]
            sqg_weight, sqg_payload, sqg_diagnostics = arms["sqg"]
            if baseline_diagnostics is not None:
                raise AssertionError("MCG control unexpectedly returned SQG diagnostics")
            energy, baseline_sse = metric_terms(source_weight, baseline_weight)
            sqg_energy, sqg_sse = metric_terms(source_weight, sqg_weight)
            if sqg_energy != energy:
                raise AssertionError("source energy changed between uniform codecs")
            if reference_energy is None:
                reference_energy = energy
            elif reference_energy != energy:
                raise AssertionError("source energy changed between rates")
            projection = {
                "source_energy": energy,
                "baseline_sse": baseline_sse,
                "baseline_relative_sse": baseline_sse / energy,
                "sqg_sse": sqg_sse,
                "sqg_relative_sse": sqg_sse / energy,
                "sqg_over_baseline": sqg_sse / baseline_sse,
                "relative_reduction": 1.0 - sqg_sse / baseline_sse,
                "baseline_payload": baseline_payload,
                "sqg_payload": sqg_payload,
            }
            if return_trellis_diagnostics:
                if sqg_diagnostics is None:
                    raise RuntimeError("SQG arm omitted requested trellis diagnostics")
                projection["sqg_trellis_diagnostics"] = sqg_diagnostics
            record = rate_records[f"K{rate}"]
            record["projections"][spec.name] = projection
            record["source_energy"] += energy
            record["baseline_sse"] += baseline_sse
            record["sqg_sse"] += sqg_sse
            del baseline_weight, sqg_weight, arms
        del source_weight

    for rate_record in rate_records.values():
        energy = float(rate_record["source_energy"])
        baseline_sse = float(rate_record["baseline_sse"])
        sqg_sse = float(rate_record["sqg_sse"])
        rate_record.update(
            {
                "baseline_relative_sse": baseline_sse / energy,
                "sqg_relative_sse": sqg_sse / energy,
                "sqg_over_baseline": sqg_sse / baseline_sse,
                "relative_reduction": 1.0 - sqg_sse / baseline_sse,
            }
        )
    return {
        "kind": f"{pilot_kind}_expert",
        "manifest_sha256": manifest_sha256,
        "complete": True,
        "layer": layer,
        "expert": expert,
        "rates": rate_records,
    }


@torch.no_grad()
def run_k2_sqg_control_expert(
    *,
    source: IndexedTensorStore,
    base_record: Mapping[str, Any],
    layer: int,
    expert: int,
    device: torch.device,
    quantizer_module: Any,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Encode exact Cheb-normal controls alongside native K2/K4 arms."""

    if int(base_record.get("layer", -1)) != layer or int(
        base_record.get("expert", -1)
    ) != expert:
        raise ValueError("base-study expert does not match the requested cell")
    base_rates = base_record.get("rates")
    if not isinstance(base_rates, Mapping) or set(base_rates) != {"K2", "K4"}:
        raise ValueError("base-study expert has an invalid rate table")

    result_rates: dict[str, dict[str, Any]] = {
        "K2": {"rate": 2, "projections": {}, "arms": {}},
        "K4": {"rate": 4, "projections": {}, "arms": {}},
    }
    for spec in PROJECTIONS:
        name = source_tensor_name(layer, expert, spec.name)
        source_weight = source.get(name)
        if source_weight.dtype != torch.bfloat16 or tuple(source_weight.shape) != spec.source_shape:
            raise TypeError(
                f"{name} must be BF16 {spec.source_shape}, got "
                f"{source_weight.dtype} {tuple(source_weight.shape)}"
            )
        input_seed, output_seed = _transform_seeds(layer, spec)
        logical_rate_axis = (
            "n" if SHARED_AXIS_BY_PROJECTION[spec.name] == 0 else "k"
        )
        exact_by_bits: dict[int, dict[str, Any]] = {}
        for bits in (2, 4):
            exact_by_bits[bits] = encode_uniform_candidate(
                source_weight,
                bits=bits,
                codebook=CODEBOOK_SQG_CHEB_NORMAL_E4M3,
                rate_axis=logical_rate_axis,
                device=device,
                quantizer_module=quantizer_module,
                input_sign_seed=input_seed,
                output_sign_seed=output_seed,
                scale_scope_key=(
                    K2_CONTROL_PILOT_KIND,
                    "cheb_normal",
                    bits,
                    layer,
                    expert,
                    spec.name,
                )
                if spec.name in ("gate_proj", "up_proj")
                else None,
                g_scale_into_sv=spec.name in ("gate_proj", "up_proj"),
                sigma_reg=SIGMA_REG,
                tailbite_context=TAILBITE_CONTEXT,
                ldlq_tf32=True,
            )
        for label, bits in (("K2", 2), ("K4", 4)):
            base_projection = base_rates[label]["projections"][spec.name]
            energy = float(base_projection["source_energy"])
            measured_energy, exact_sse = metric_terms(
                source_weight, exact_by_bits[bits].pop("reconstruction")
            )
            if measured_energy != energy:
                raise ValueError("base study and exact control source energy differ")
            arms = {
                "mcg": {
                    "sse": float(base_projection["baseline_sse"]),
                    "relative_sse": float(base_projection["baseline_relative_sse"]),
                    "payload": base_projection["baseline_payload"],
                    "origin": base_rates[label]["baseline_origin"],
                },
                "t12": {
                    "sse": float(base_projection["sqg_sse"]),
                    "relative_sse": float(base_projection["sqg_relative_sse"]),
                    "payload": base_projection["sqg_payload"],
                    "origin": "matched_base_study",
                },
                "cheb_normal": {
                    "sse": exact_sse,
                    "relative_sse": exact_sse / energy,
                    "payload": exact_by_bits[bits]["payload"],
                    "origin": "fresh_offline_control_encode",
                },
            }
            result_rates[label]["projections"][spec.name] = {
                "source_energy": energy,
                "arms": arms,
            }
        del source_weight

    for label, rate in result_rates.items():
        arm_names = ("mcg", "t12", "cheb_normal")
        energy = sum(
            float(projection["source_energy"])
            for projection in rate["projections"].values()
        )
        for arm_name in arm_names:
            sse = sum(
                float(projection["arms"][arm_name]["sse"])
                for projection in rate["projections"].values()
            )
            rate["arms"][arm_name] = {
                "sse": sse,
                "relative_sse": sse / energy,
            }
        rate["source_energy"] = energy
    return {
        "kind": f"{K2_CONTROL_PILOT_KIND}_expert",
        "manifest_sha256": manifest_sha256,
        "complete": True,
        "layer": layer,
        "expert": expert,
        "rates": result_rates,
    }


def _group_terms(
    records: Sequence[Mapping[str, Any]], *, sqg_sse_field: str = "sqg_sse"
) -> dict[str, float]:
    energy = sum(float(record["source_energy"]) for record in records)
    baseline_sse = sum(float(record["baseline_sse"]) for record in records)
    sqg_sse = sum(float(record[sqg_sse_field]) for record in records)
    if energy <= 0.0 or baseline_sse <= 0.0:
        raise ValueError("aggregate metric has a nonpositive denominator")
    baseline_relative = baseline_sse / energy
    sqg_relative = sqg_sse / energy
    ratio = sqg_relative / baseline_relative
    return {
        "source_energy": energy,
        "baseline_sse": baseline_sse,
        "baseline_relative_sse": baseline_relative,
        "sqg_sse": sqg_sse,
        "sqg_relative_sse": sqg_relative,
        "sqg_over_baseline": ratio,
        "relative_reduction": 1.0 - ratio,
    }


def cluster_bootstrap_ratio(
    records: Sequence[Mapping[str, Any]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    sqg_sse_field: str = "sqg_sse",
) -> dict[str, Any]:
    """Paired layer-cluster bootstrap for the aggregate SSE ratio."""

    if replicates <= 0:
        raise ValueError("bootstrap replicate count must be positive")
    by_layer: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        by_layer.setdefault(int(record["layer"]), []).append(record)
    layers = tuple(sorted(by_layer))
    if not layers:
        raise ValueError("bootstrap requires at least one layer")
    layer_baseline = np.asarray(
        [sum(float(item["baseline_sse"]) for item in by_layer[layer]) for layer in layers],
        dtype=np.float64,
    )
    layer_sqg = np.asarray(
        [
            sum(float(item[sqg_sse_field]) for item in by_layer[layer])
            for layer in layers
        ],
        dtype=np.float64,
    )
    if np.any(layer_baseline <= 0.0):
        raise ValueError("bootstrap baseline SSE must be positive")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(layers), size=(replicates, len(layers)))
    baseline = layer_baseline[draws].sum(axis=1)
    sqg = layer_sqg[draws].sum(axis=1)
    ratios = sqg / baseline
    lower, upper = np.quantile(ratios, (0.025, 0.975))
    return {
        "method": "paired_layer_cluster_percentile",
        "layer_clusters": len(layers),
        "replicates": replicates,
        "seed": seed,
        "ratio_ci95": [float(lower), float(upper)],
    }


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    sqg_sse_field: str = "sqg_sse",
    panel: Mapping[int, Sequence[int]] = PANEL,
) -> dict[str, Any]:
    expected_cells = set(panel_cells(panel))
    actual_cells = {(int(record["layer"]), int(record["expert"])) for record in records}
    if actual_cells != expected_cells or len(records) != len(expected_cells):
        raise ValueError("expert records do not exactly cover the frozen panel")
    overall = _group_terms(records, sqg_sse_field=sqg_sse_field)
    expert_ratios = [
        float(record[sqg_sse_field]) / float(record["baseline_sse"])
        for record in records
    ]
    overall.update(
        {
            "expert_count": len(records),
            "expert_wins": sum(ratio < 1.0 for ratio in expert_ratios),
            "expert_ties": sum(ratio == 1.0 for ratio in expert_ratios),
            "expert_losses": sum(ratio > 1.0 for ratio in expert_ratios),
            "median_expert_ratio": float(np.median(expert_ratios)),
            "geometric_mean_expert_ratio": float(
                np.exp(np.mean(np.log(np.asarray(expert_ratios, dtype=np.float64))))
            ),
        }
    )
    bootstrap = cluster_bootstrap_ratio(records, sqg_sse_field=sqg_sse_field)
    lower, upper = bootstrap["ratio_ci95"]
    if upper < 1.0:
        classification = "clear_lower_distortion"
    elif lower > 1.0:
        classification = "clear_regression"
    else:
        classification = "inconclusive"

    by_layer = {
        str(layer): _group_terms(
            [record for record in records if int(record["layer"]) == layer],
            sqg_sse_field=sqg_sse_field,
        )
        for layer in panel
    }
    by_projection: dict[str, Any] = {}
    for spec in PROJECTIONS:
        projection_records = []
        for record in records:
            projection = record["projections"][spec.name]
            projection_records.append(
                {
                    "source_energy": projection["source_energy"],
                    "baseline_sse": projection["baseline_sse"],
                    sqg_sse_field: projection[sqg_sse_field],
                }
            )
        by_projection[spec.name] = _group_terms(
            projection_records, sqg_sse_field=sqg_sse_field
        )
    return {
        "overall": overall,
        "bootstrap": bootstrap,
        "classification": classification,
        "by_layer": by_layer,
        "by_projection": by_projection,
    }


def aggregate_rate_shift_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate the selected rate-shift arm and its matched R0 control."""

    return {
        "rate_shifted": aggregate_records(records),
        "uniform_r0_control": aggregate_records(
            records, sqg_sse_field="sqg_r0_sse"
        ),
        "mode_selection": summarize_mode_selections(
            records, tuple(RATE_FAMILIES)
        ),
    }


def aggregate_uniform_rate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    rate_labels: Sequence[str] = ("K2", "K4"),
    panel: Mapping[int, Sequence[int]] = K4_PANEL,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    labels = tuple(rate_labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("uniform-rate labels must be nonempty and unique")
    for label in labels:
        views = []
        for record in records:
            rate = record["rates"][label]
            views.append(
                {
                    "layer": record["layer"],
                    "expert": record["expert"],
                    "source_energy": rate["source_energy"],
                    "baseline_sse": rate["baseline_sse"],
                    "sqg_sse": rate["sqg_sse"],
                    "projections": rate["projections"],
                }
            )
        result[label] = aggregate_records(views, panel=panel)
    return result


def _aggregate_control_pair(
    records: Sequence[Mapping[str, Any]],
    *,
    rate_label: str,
    baseline_arm: str,
    candidate_arm: str,
) -> dict[str, Any]:
    views = []
    for record in records:
        rate = record["rates"][rate_label]
        projections = {}
        for spec in PROJECTIONS:
            projection = rate["projections"][spec.name]
            projections[spec.name] = {
                "source_energy": projection["source_energy"],
                "baseline_sse": projection["arms"][baseline_arm]["sse"],
                "sqg_sse": projection["arms"][candidate_arm]["sse"],
            }
        views.append(
            {
                "layer": record["layer"],
                "expert": record["expert"],
                "source_energy": rate["source_energy"],
                "baseline_sse": rate["arms"][baseline_arm]["sse"],
                "sqg_sse": rate["arms"][candidate_arm]["sse"],
                "projections": projections,
            }
        )
    return aggregate_records(views, panel=K4_PANEL)


def aggregate_k2_sqg_control_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "K2": {
            "vs_mcg": {
                arm: _aggregate_control_pair(
                    records,
                    rate_label="K2",
                    baseline_arm="mcg",
                    candidate_arm=arm,
                )
                for arm in ("t12", "cheb_normal")
            },
            "cheb_normal_vs_t12": _aggregate_control_pair(
                records,
                rate_label="K2",
                baseline_arm="t12",
                candidate_arm="cheb_normal",
            ),
        },
        "K4": {
            "vs_mcg": {
                arm: _aggregate_control_pair(
                    records,
                    rate_label="K4",
                    baseline_arm="mcg",
                    candidate_arm=arm,
                )
                for arm in ("t12", "cheb_normal")
            },
            "cheb_normal_vs_t12": _aggregate_control_pair(
                records,
                rate_label="K4",
                baseline_arm="t12",
                candidate_arm="cheb_normal",
            ),
        },
    }


def _runtime_identity(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or device.index is None:
        raise ValueError("the GLM codec pilot requires one explicit CUDA device")
    if device.index < 0 or device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {device.index} is not enumerated")
    properties = torch.cuda.get_device_properties(device)
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
    }


def build_manifest(
    *,
    source_root: Path,
    baseline_root: Path,
    device: torch.device,
    inventory: Mapping[str, Any],
    panel: Mapping[int, Sequence[int]] = PANEL,
) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    rank_lut = sqg_xor_cheb_t12_rank_lut_bytes()
    k3_lut = sqg_xor_cheb_t12_bytes(BITS)
    return {
        "kind": f"{PILOT_KIND}_manifest",
        "source_root": str(source_root.resolve()),
        "baseline_root": str(baseline_root.resolve()),
        "inventory": inventory,
        "panel_seed": PILOT_SEED,
        "panel": {str(layer): list(experts) for layer, experts in panel.items()},
        "panel_contract": {
            "selection": "content hash without error ranking",
            "complete_frozen_panel_experts": len(panel_cells(PANEL)),
            "executed_experts": len(panel_cells(panel)),
        },
        "quantization": {
            "profile": "qsrt_sqg_e4m3",
            "codebook": CODEBOOK_SQG_XOR_CHEB_T12,
            "rate": BITS,
            "hessian": "identity",
            "neuron_order": "identity",
            "sigma_reg": SIGMA_REG,
            "tailbite_context": TAILBITE_CONTEXT,
            "apply_out_scales": False,
            "ldlq_tf32": True,
            "folded_scale_power": 0,
            "rate_shifts": False,
            "scale_scope": "tensor_local",
            "seed_formula": "input=L*1000000+projection_index; output=input+499979",
        },
        "codebook_identity": {
            "t12_rank_lut_sha256": _tensor_sha256(rank_lut),
            "k3_direct_lut_sha256": _tensor_sha256(k3_lut),
        },
        "metric": {
            "name": "source_relative_sse",
            "source_dtype": "BF16",
            "reconstruction_endpoint": "FP16",
            "accumulation": "FP64",
            "baseline_stored_error_atol": BASELINE_STORED_ERROR_ATOL,
        },
        "bootstrap": {
            "method": "paired_layer_cluster_percentile",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "runtime": _runtime_identity(device),
        "qsrt": git_state(repo),
    }


def build_rate_shift_manifest(
    *,
    source_root: Path,
    baseline_root: Path,
    device: torch.device,
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable manifest for the headline fixed-rate experiment."""

    repo = Path(__file__).resolve().parents[1]
    rank_lut = sqg_xor_cheb_t12_rank_lut_bytes()
    rate_luts = {bits: sqg_xor_cheb_t12_bytes(bits) for bits in (2, 3, 4)}
    return {
        "kind": f"{RATE_SHIFT_PILOT_KIND}_manifest",
        "source_root": str(source_root.resolve()),
        "baseline_root": str(baseline_root.resolve()),
        "inventory": inventory,
        "panel_seed": PILOT_SEED,
        "panel": {str(layer): list(experts) for layer, experts in PANEL.items()},
        "quantization": {
            "profile": "qsrt_sqg_e4m3",
            "codebook": CODEBOOK_SQG_XOR_CHEB_T12,
            "rates": [2, 3, 4],
            "mean_trellis_bpw": BITS,
            "mode_ids": list(RATE_GEOMETRY.mode_ids),
            "axis_channels": RATE_GEOMETRY.axis_channels,
            "record_channels": RATE_GEOMETRY.record_channels,
            "records": RATE_GEOMETRY.record_count,
            "tiles_per_record": RATE_GEOMETRY.tiles_per_record,
            "rate_identity": "K_donor + K_recipient = 2 * K_baseline",
            "hessian": "identity",
            "ordering": "coupled_source_weight_energy",
            "ordering_group_channels": 4,
            "sigma_reg": SIGMA_REG,
            "tailbite_context": TAILBITE_CONTEXT,
            "apply_out_scales": False,
            "ldlq_tf32": True,
            "folded_scale_power": 0,
            "rate_shifts": True,
            "scale_scope": "tensor_local",
            "seed_formula": "input=L*1000000+projection_index; output=input+499979",
        },
        "model_adapter": {
            "shared_coordinate_axis": "expert intermediate neuron",
            "source_axes": dict(SHARED_AXIS_BY_PROJECTION),
            "rate_families": {
                family: list(matrices)
                for family, matrices in RATE_FAMILIES.items()
            },
            "symmetry": (
                "one permutation P is applied to gate/up output rows and down "
                "input columns, then undone after reconstruction"
            ),
        },
        "mode_selection": {
            "metric": "raw_source_weight_sse",
            "coupled_gate_up": True,
            "independent_down": True,
            "tie_break": "lowest_mode_id",
            "scope": "codec_endpoint_optimization_not_quality_validation",
        },
        "codebook_identity": {
            "t12_rank_lut_sha256": _tensor_sha256(rank_lut),
            **{
                f"k{bits}_direct_lut_sha256": _tensor_sha256(lut)
                for bits, lut in rate_luts.items()
            },
        },
        "metric": {
            "name": "source_relative_sse",
            "source_dtype": "BF16",
            "reconstruction_endpoint": "FP16",
            "accumulation": "FP64",
            "baseline_stored_error_atol": BASELINE_STORED_ERROR_ATOL,
        },
        "bootstrap": {
            "method": "paired_layer_cluster_percentile",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "runtime": _runtime_identity(device),
        "qsrt": git_state(repo),
    }


def build_uniform_rate_manifest(
    *,
    source_root: Path,
    baseline_root: Path,
    device: torch.device,
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    rank_lut = sqg_xor_cheb_t12_rank_lut_bytes()
    return {
        "kind": f"{UNIFORM_RATE_PILOT_KIND}_manifest",
        "source_root": str(source_root.resolve()),
        "baseline_root": str(baseline_root.resolve()),
        "inventory": inventory,
        "panel_seed": UNIFORM_RATE_PILOT_SEED,
        "panel": {
            str(layer): list(experts) for layer, experts in K4_PANEL.items()
        },
        "panel_contract": {
            "eligible_checkpoint_tier": 4,
            "experts_per_layer": 4,
            "selection": "content_hash_without_error_ranking",
            "shared_by_k2_and_k4": True,
        },
        "quantization": {
            "rates": [2, 4],
            "uniform_rate": True,
            "rate_shifts": False,
            "sqg_profile": SQG_FP16_D3L,
            "sqg_codebook": SQG_FP16_D3L,
            "mcg_codebook": CODEBOOK_MCG,
            "k2_mcg_origin": "fresh_offline_encode",
            "k4_mcg_origin": "materialized_willfalco_artifact",
            "hessian": "identity",
            "neuron_order": "identity",
            "sigma_reg": SIGMA_REG,
            "tailbite_context": TAILBITE_CONTEXT,
            "apply_out_scales": False,
            "ldlq_tf32": True,
            "scale_scope": "tensor_local_for_fresh_encodes",
            "seed_formula": "input=L*1000000+projection_index; output=input+499979",
        },
        "codebook_identity": {
            "t12_rank_lut_sha256": _tensor_sha256(rank_lut),
            "k2_direct_lut_sha256": _tensor_sha256(
                sqg_xor_cheb_t12_bytes(2)
            ),
            "k4_direct_lut_sha256": _tensor_sha256(
                sqg_xor_cheb_t12_bytes(4)
            ),
            "mcg_multiplier": 0xCBAC1FED,
        },
        "metric": {
            "name": "source_relative_sse",
            "source_dtype": "BF16",
            "reconstruction_endpoint": "FP16",
            "accumulation": "FP64",
            "baseline_stored_error_atol": BASELINE_STORED_ERROR_ATOL,
        },
        "bootstrap": {
            "method": "paired_layer_cluster_percentile",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "runtime": _runtime_identity(device),
        "qsrt": git_state(repo),
    }


def build_uniform_high_rate_manifest(
    *,
    source_root: Path,
    baseline_root: Path,
    device: torch.device,
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe the matched fresh-MCG/SQG K5/K6 extension study."""

    repo = Path(__file__).resolve().parents[1]
    return {
        "kind": f"{UNIFORM_HIGH_RATE_PILOT_KIND}_manifest",
        "source_root": str(source_root.resolve()),
        "baseline_root": str(baseline_root.resolve()),
        "inventory": inventory,
        "panel_seed": UNIFORM_RATE_PILOT_SEED,
        "panel": {
            str(layer): list(experts) for layer, experts in K4_PANEL.items()
        },
        "panel_contract": {
            "eligible_checkpoint_tier": 4,
            "experts_per_layer": 4,
            "selection": "content_hash_without_error_ranking",
            "reuses_uniform_k2_k4_panel": True,
        },
        "quantization": {
            "rates": list(UNIFORM_HIGH_RATE_BITS),
            "uniform_rate": True,
            "rate_shifts": False,
            "sqg_profile": "qsrt_sqg_e4m3",
            "sqg_codebook": CODEBOOK_SQG_XOR_CHEB_T12,
            "mcg_codebook": CODEBOOK_MCG,
            "mcg_origin": "fresh_offline_encode",
            "sqg_origin": "fresh_offline_encode",
            "hessian": "identity",
            "neuron_order": "identity",
            "sigma_reg": SIGMA_REG,
            "tailbite_context": TAILBITE_CONTEXT,
            "apply_out_scales": False,
            "ldlq_tf32": True,
            "scale_scope": "tensor_local_for_fresh_encodes",
            "seed_formula": "input=L*1000000+projection_index; output=input+499979",
        },
        "codebook_identity": {
            "d3l_descriptor_sha256": SQG_FP16_D3L_DESCRIPTOR_SHA256,
            **{
                f"k{bits}_direct_lut_sha256": _tensor_sha256(
                    sqg_fp16_d3l_codebook(bits)
                )
                for bits in UNIFORM_HIGH_RATE_BITS
            },
            "mcg_multiplier": 0xCBAC1FED,
        },
        "metric": {
            "name": "source_relative_sse",
            "source_dtype": "BF16",
            "reconstruction_endpoint": "FP16",
            "accumulation": "FP64",
        },
        "bootstrap": {
            "method": "paired_layer_cluster_percentile",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "runtime": _runtime_identity(device),
        "qsrt": git_state(repo),
    }


def load_uniform_rate_base_study(
    base_root: Path,
) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    report_path = base_root / "report.json"
    manifest_path = base_root / "manifest.json"
    report = _read_json_object(report_path)
    manifest = _read_json_object(manifest_path)
    if report.get("kind") != UNIFORM_RATE_PILOT_KIND or report.get("status") != "complete":
        raise ValueError("K2 control base study is not a complete uniform-rate pilot")
    if manifest.get("kind") != f"{UNIFORM_RATE_PILOT_KIND}_manifest":
        raise ValueError("K2 control base study has a foreign manifest")
    if report.get("manifest_sha256") != _canonical_json_sha256(manifest):
        raise ValueError("K2 control base report does not match its manifest")
    expected_panel = {
        str(layer): list(experts) for layer, experts in K4_PANEL.items()
    }
    if manifest.get("panel") != expected_panel:
        raise ValueError("K2 control base study used a different expert panel")
    raw_records = report.get("experts")
    if not isinstance(raw_records, list):
        raise ValueError("K2 control base report has no expert records")
    records = {
        (int(record["layer"]), int(record["expert"])): record
        for record in raw_records
    }
    if set(records) != set(panel_cells(K4_PANEL)) or len(records) != 48:
        raise ValueError("K2 control base report does not cover the frozen panel")
    identity = {
        "root": str(base_root.resolve()),
        "report_sha256": sha256_file(report_path),
        "manifest_sha256": sha256_file(manifest_path),
        "content_manifest_sha256": report["manifest_sha256"],
    }
    return report, records, identity


def build_k2_control_manifest(
    *,
    source_root: Path,
    baseline_root: Path,
    device: torch.device,
    inventory: Mapping[str, Any],
    base_study: Mapping[str, Any],
) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    exact_rank = sqg_cheb_normal_rank_e4m3_bytes()
    return {
        "kind": f"{K2_CONTROL_PILOT_KIND}_manifest",
        "source_root": str(source_root.resolve()),
        "baseline_root": str(baseline_root.resolve()),
        "inventory": inventory,
        "base_study": dict(base_study),
        "panel_seed": UNIFORM_RATE_PILOT_SEED,
        "panel": {
            str(layer): list(experts) for layer, experts in K4_PANEL.items()
        },
        "controls": {
            "rates": [2, 4],
            "t12": "matched_base_study",
            "mcg": "matched_base_study",
            "cheb_normal": {
                "codebook": CODEBOOK_SQG_CHEB_NORMAL_E4M3,
                "graph": "native_rate_strata",
                "rates": [2, 4],
            },
            "purpose": "measure the T12 staircase approximation cost",
        },
        "quantization": {
            "hessian": "identity",
            "neuron_order": "identity",
            "sigma_reg": SIGMA_REG,
            "tailbite_context": TAILBITE_CONTEXT,
            "apply_out_scales": False,
            "ldlq_tf32": True,
            "scale_scope": "tensor_local",
            "seed_formula": "input=L*1000000+projection_index; output=input+499979",
        },
        "codebook_identity": {
            "cheb_normal_rank_sha256": _tensor_sha256(exact_rank),
            "native_k2_sha256": _tensor_sha256(
                sqg_cheb_normal_e4m3_bytes(2)
            ),
            "native_k4_sha256": _tensor_sha256(
                sqg_cheb_normal_e4m3_bytes(4)
            ),
        },
        "metric": {
            "name": "source_relative_sse",
            "source_dtype": "BF16",
            "reconstruction_endpoint": "FP16",
            "accumulation": "FP64",
        },
        "bootstrap": {
            "method": "paired_layer_cluster_percentile",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "runtime": _runtime_identity(device),
        "qsrt": git_state(repo),
    }


def prepare_destination(dest: Path, manifest: dict[str, Any], *, resume: bool) -> str:
    manifest_path = dest / "manifest.json"
    if dest.exists():
        if not resume:
            raise FileExistsError(
                f"destination {dest} exists; use --resume only for an identical run"
            )
        if not manifest_path.is_file():
            raise ValueError(f"resume destination {dest} has no manifest.json")
        existing = _read_json_object(manifest_path)
        if existing != manifest:
            raise ValueError("resume destination manifest does not match this run")
    else:
        dest.mkdir(parents=True)
        (dest / "experts").mkdir()
        atomic_write_json(manifest_path, manifest)
    experts_dir = dest / "experts"
    if not experts_dir.is_dir():
        raise FileNotFoundError(experts_dir)
    return _canonical_json_sha256(manifest)


def _expert_path(dest: Path, layer: int, expert: int) -> Path:
    return dest / "experts" / f"layer-{layer:03d}-expert-{expert:03d}.json"


def _load_completed_expert(
    path: Path,
    *,
    layer: int,
    expert: int,
    manifest_sha256: str,
    pilot_kind: str = PILOT_KIND,
) -> dict[str, Any]:
    record = _read_json_object(path)
    if (
        record.get("kind") != f"{pilot_kind}_expert"
        or record.get("complete") is not True
        or record.get("manifest_sha256") != manifest_sha256
        or record.get("layer") != layer
        or record.get("expert") != expert
    ):
        raise ValueError(f"invalid or foreign expert receipt {path}")
    return record


def summary_markdown(report: Mapping[str, Any]) -> str:
    aggregate = report["aggregate"]
    overall = aggregate["overall"]
    lower, upper = aggregate["bootstrap"]["ratio_ci95"]
    lines = [
        "# GLM-5.2 K3 codec-distortion pilot",
        "",
        f"- Classification: `{aggregate['classification']}`",
        f"- Experts: {overall['expert_count']}",
        f"- EXL3 relative SSE: {overall['baseline_relative_sse']:.9g}",
        f"- SQG relative SSE: {overall['sqg_relative_sse']:.9g}",
        f"- SQG / EXL3: {overall['sqg_over_baseline']:.6f}",
        f"- Relative reduction: {100.0 * overall['relative_reduction']:.3f}%",
        f"- Layer-cluster 95% CI: [{lower:.6f}, {upper:.6f}]",
        f"- Expert wins/ties/losses: {overall['expert_wins']}/"
        f"{overall['expert_ties']}/{overall['expert_losses']}",
        "",
        "## Per projection",
        "",
        "| Projection | EXL3 relative SSE | SQG relative SSE | SQG / EXL3 |",
        "|---|---:|---:|---:|",
    ]
    for spec in PROJECTIONS:
        item = aggregate["by_projection"][spec.name]
        lines.append(
            f"| {spec.name} | {item['baseline_relative_sse']:.9g} | "
            f"{item['sqg_relative_sse']:.9g} | {item['sqg_over_baseline']:.6f} |"
        )
    lines.extend(
        [
            "",
            "This is a raw weight-distortion pilot against the materialized artifact. "
            "It does not isolate the codebook from Hessian, transform, or scale-policy "
            "differences and is not an end-to-end quality result.",
            "",
        ]
    )
    return "\n".join(lines)


def run_pilot(
    *,
    source_root: Path,
    baseline_root: Path,
    dest: Path,
    device: torch.device,
    resume: bool = False,
    exllamav3_root: Path = Path("/home/luke/projects/exllamav3"),
    panel: Mapping[int, Sequence[int]] = PANEL,
) -> dict[str, Any]:
    """Execute or resume a source-pinned, production-encoder expert panel."""

    panel = {
        int(layer): tuple(map(int, experts))
        for layer, experts in panel.items()
    }
    if not panel or not panel_cells(panel):
        raise ValueError("the GLM codec panel must contain at least one expert")
    inventory = validate_inventory(source_root, baseline_root, panel=panel)
    tier_bitmap = _read_json_object(baseline_root / "tier_bitmap.json")
    manifest = build_manifest(
        source_root=source_root,
        baseline_root=baseline_root,
        device=device,
        inventory=inventory,
        panel=panel,
    )
    manifest_sha256 = prepare_destination(dest, manifest, resume=resume)
    source = IndexedTensorStore(source_root)
    records: list[dict[str, Any]] = []
    pending = []
    for layer, expert in panel_cells(panel):
        path = _expert_path(dest, layer, expert)
        if path.is_file():
            records.append(
                _load_completed_expert(
                    path,
                    layer=layer,
                    expert=expert,
                    manifest_sha256=manifest_sha256,
                )
            )
        else:
            pending.append((layer, expert))

    if pending:
        quantizer_module = load_qsrt_encoder(exllamav3_root)
        install_sqg_quantizer(quantizer_module)
        pending_set = set(pending)
        completed = len(records)
        total = len(panel_cells(panel))
        for layer, experts in panel.items():
            layer_pending = [expert for expert in experts if (layer, expert) in pending_set]
            if not layer_pending:
                continue
            layer_path = baseline_root / f"model-layer-{layer:03d}.safetensors"
            with safe_open(layer_path, framework="pt", device="cpu") as handle:
                for expert in layer_pending:
                    started = time.monotonic()
                    record = run_expert(
                        source=source,
                        baseline_handle=handle,
                        tier_entry=tier_bitmap[str(layer)],
                        layer=layer,
                        expert=expert,
                        device=device,
                        quantizer_module=quantizer_module,
                        manifest_sha256=manifest_sha256,
                    )
                    record["wall_seconds"] = time.monotonic() - started
                    atomic_write_json(_expert_path(dest, layer, expert), record)
                    records.append(record)
                    completed += 1
                    print(
                        f"[{completed:02d}/{total}] layer {layer} expert {expert}: "
                        f"ratio={record['sqg_over_baseline']:.6f} "
                        f"reduction={100.0 * record['relative_reduction']:.2f}%",
                        flush=True,
                    )
                    torch.cuda.empty_cache()

    records = [
        _load_completed_expert(
            _expert_path(dest, layer, expert),
            layer=layer,
            expert=expert,
            manifest_sha256=manifest_sha256,
        )
        for layer, expert in panel_cells(panel)
    ]
    aggregate = aggregate_records(records, panel=panel)
    report = {
        "kind": PILOT_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "expert_count": len(records),
        "matrix_count": len(records) * len(PROJECTIONS),
        "panel": {str(layer): list(experts) for layer, experts in panel.items()},
        "production_shape": {
            "source_dtype": "BF16",
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "trellis_values_per_tile": 256,
            "tailbite_context": TAILBITE_CONTEXT,
            "uses_production_cuda_encoder": True,
            "uses_production_scale_search": True,
        },
        "aggregate": aggregate,
        "experts": records,
    }
    atomic_write_json(dest / "report.json", report)
    summary_path = dest / "summary.md"
    temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(summary_markdown(report))
        temporary.replace(summary_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return report


def rate_shift_summary_markdown(report: Mapping[str, Any]) -> str:
    aggregate = report["aggregate"]
    shifted = aggregate["rate_shifted"]
    overall = shifted["overall"]
    r0 = aggregate["uniform_r0_control"]["overall"]
    modes = aggregate["mode_selection"]
    lower, upper = shifted["bootstrap"]["ratio_ci95"]
    lines = [
        "# GLM-5.2 fixed-3bpw QSRT codec-distortion pilot",
        "",
        f"- Classification: `{shifted['classification']}`",
        f"- Experts: {overall['expert_count']}",
        f"- Experts selecting R1+ on either family: "
        f"{modes['any_family_r1_plus']}",
        f"- EXL3 relative SSE: {overall['baseline_relative_sse']:.9g}",
        f"- QSRT rate-shifted relative SSE: {overall['sqg_relative_sse']:.9g}",
        f"- QSRT matched R0 relative SSE: {r0['sqg_relative_sse']:.9g}",
        f"- Rate-shifted QSRT / EXL3: {overall['sqg_over_baseline']:.6f}",
        f"- Rate-shifted relative reduction: "
        f"{100.0 * overall['relative_reduction']:.3f}%",
        f"- Layer-cluster 95% CI: [{lower:.6f}, {upper:.6f}]",
        f"- Expert wins/ties/losses: {overall['expert_wins']}/"
        f"{overall['expert_ties']}/{overall['expert_losses']}",
        "",
        "## Rate selections",
        "",
    ]
    for family, histogram in modes["family_histograms"].items():
        rendered = ", ".join(
            f"{label}={count}" for label, count in histogram.items()
        )
        lines.append(f"- {family}: {rendered}")
    lines.extend(
        [
            "",
            "The headline arm uses equal K2 donors and K4 recipients over a "
            "GLM-native 16-record axis, so every matrix remains exactly three "
            "trellis bits per weight. The rate codec, ordering, and selection "
            "machinery are model-agnostic; this adapter supplies GLM tensor roles.",
            "",
            "This is a raw weight-distortion pilot against the materialized "
            "artifact. Mode selection optimizes this codec endpoint on the source "
            "weights themselves; it is not an end-to-end quality result.",
            "",
        ]
    )
    return "\n".join(lines)


def run_rate_shift_pilot(
    *,
    source_root: Path,
    baseline_root: Path,
    dest: Path,
    device: torch.device,
    resume: bool = False,
    exllamav3_root: Path = Path("/home/luke/projects/exllamav3"),
) -> dict[str, Any]:
    """Execute or resume the fixed 48-expert rate-shifted QSRT pilot."""

    inventory = validate_inventory(source_root, baseline_root)
    tier_bitmap = _read_json_object(baseline_root / "tier_bitmap.json")
    validate_and_select_panel(tier_bitmap)
    manifest = build_rate_shift_manifest(
        source_root=source_root,
        baseline_root=baseline_root,
        device=device,
        inventory=inventory,
    )
    manifest_sha256 = prepare_destination(dest, manifest, resume=resume)
    source = IndexedTensorStore(source_root)
    records: list[dict[str, Any]] = []
    pending = []
    for layer, expert in panel_cells():
        path = _expert_path(dest, layer, expert)
        if path.is_file():
            records.append(
                _load_completed_expert(
                    path,
                    layer=layer,
                    expert=expert,
                    manifest_sha256=manifest_sha256,
                    pilot_kind=RATE_SHIFT_PILOT_KIND,
                )
            )
        else:
            pending.append((layer, expert))

    if pending:
        quantizer_module = load_qsrt_encoder(exllamav3_root)
        install_sqg_quantizer(quantizer_module)
        pending_set = set(pending)
        completed = len(records)
        total = len(panel_cells())
        for layer, experts in PANEL.items():
            layer_pending = [
                expert for expert in experts if (layer, expert) in pending_set
            ]
            if not layer_pending:
                continue
            layer_path = baseline_root / f"model-layer-{layer:03d}.safetensors"
            with safe_open(layer_path, framework="pt", device="cpu") as handle:
                for expert in layer_pending:
                    started = time.monotonic()
                    record = run_rate_shift_expert(
                        source=source,
                        baseline_handle=handle,
                        tier_entry=tier_bitmap[str(layer)],
                        layer=layer,
                        expert=expert,
                        device=device,
                        quantizer_module=quantizer_module,
                        manifest_sha256=manifest_sha256,
                    )
                    record["wall_seconds"] = time.monotonic() - started
                    atomic_write_json(_expert_path(dest, layer, expert), record)
                    records.append(record)
                    completed += 1
                    r13 = record["rate_selection"]["r13"]["selected_mode"]
                    r2 = record["rate_selection"]["r2"]["selected_mode"]
                    print(
                        f"[{completed:02d}/{total}] layer {layer} expert {expert}: "
                        f"R{r13}/R{r2} ratio={record['sqg_over_baseline']:.6f} "
                        f"reduction={100.0 * record['relative_reduction']:.2f}%",
                        flush=True,
                    )
                    torch.cuda.empty_cache()

    records = [
        _load_completed_expert(
            _expert_path(dest, layer, expert),
            layer=layer,
            expert=expert,
            manifest_sha256=manifest_sha256,
            pilot_kind=RATE_SHIFT_PILOT_KIND,
        )
        for layer, expert in panel_cells()
    ]
    aggregate = aggregate_rate_shift_records(records)
    report = {
        "kind": RATE_SHIFT_PILOT_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "expert_count": len(records),
        "matrix_count": len(records) * len(PROJECTIONS),
        "aggregate": aggregate,
        "experts": records,
    }
    atomic_write_json(dest / "report.json", report)
    summary_path = dest / "summary.md"
    temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(rate_shift_summary_markdown(report))
        temporary.replace(summary_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return report


def uniform_rate_summary_markdown(report: Mapping[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# GLM-5.2 uniform-rate codec-distortion pilot",
        "",
        "The same 48 error-blind, K4-eligible experts are used at both rates. "
        "K2 MCG and SQG are freshly encoded; K4 MCG is the materialized "
        "willfalco artifact and K4 SQG is freshly encoded.",
        "",
        "| Rate | Classification | MCG relative SSE | SQG relative SSE | "
        "SQG / MCG | Reduction | Wins/ties/losses |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label in ("K2", "K4"):
        item = aggregate[label]
        overall = item["overall"]
        lines.append(
            f"| {label} | `{item['classification']}` | "
            f"{overall['baseline_relative_sse']:.9g} | "
            f"{overall['sqg_relative_sse']:.9g} | "
            f"{overall['sqg_over_baseline']:.6f} | "
            f"{100.0 * overall['relative_reduction']:.3f}% | "
            f"{overall['expert_wins']}/{overall['expert_ties']}/"
            f"{overall['expert_losses']} |"
        )
    lines.extend(
        [
            "",
            "This is a uniform-rate raw weight-distortion study. It is not a "
            "rate-shift selection study or an end-to-end quality result.",
            "",
        ]
    )
    return "\n".join(lines)


def run_uniform_rate_pilot(
    *,
    source_root: Path,
    baseline_root: Path,
    dest: Path,
    device: torch.device,
    resume: bool = False,
    exllamav3_root: Path = Path("/home/luke/projects/exllamav3"),
) -> dict[str, Any]:
    """Execute or resume the matched uniform-K2 and uniform-K4 study."""

    inventory = validate_inventory(
        source_root,
        baseline_root,
        panel=K4_PANEL,
        panel_rate=4,
        panel_seed=UNIFORM_RATE_PILOT_SEED,
    )
    tier_bitmap = _read_json_object(baseline_root / "tier_bitmap.json")
    validate_and_select_rate_panel(
        tier_bitmap,
        panel=K4_PANEL,
        eligible_rate=4,
        seed=UNIFORM_RATE_PILOT_SEED,
    )
    manifest = build_uniform_rate_manifest(
        source_root=source_root,
        baseline_root=baseline_root,
        device=device,
        inventory=inventory,
    )
    manifest_sha256 = prepare_destination(dest, manifest, resume=resume)
    source = IndexedTensorStore(source_root)
    records: list[dict[str, Any]] = []
    pending = []
    for layer, expert in panel_cells(K4_PANEL):
        path = _expert_path(dest, layer, expert)
        if path.is_file():
            records.append(
                _load_completed_expert(
                    path,
                    layer=layer,
                    expert=expert,
                    manifest_sha256=manifest_sha256,
                    pilot_kind=UNIFORM_RATE_PILOT_KIND,
                )
            )
        else:
            pending.append((layer, expert))

    if pending:
        quantizer_module = load_qsrt_encoder(exllamav3_root)
        install_sqg_quantizer(quantizer_module)
        pending_set = set(pending)
        completed = len(records)
        total = len(panel_cells(K4_PANEL))
        for layer, experts in K4_PANEL.items():
            layer_pending = [
                expert for expert in experts if (layer, expert) in pending_set
            ]
            if not layer_pending:
                continue
            layer_path = baseline_root / f"model-layer-{layer:03d}.safetensors"
            with safe_open(layer_path, framework="pt", device="cpu") as handle:
                for expert in layer_pending:
                    started = time.monotonic()
                    record = run_uniform_rate_expert(
                        source=source,
                        baseline_handle=handle,
                        tier_entry=tier_bitmap[str(layer)],
                        layer=layer,
                        expert=expert,
                        device=device,
                        quantizer_module=quantizer_module,
                        manifest_sha256=manifest_sha256,
                    )
                    record["wall_seconds"] = time.monotonic() - started
                    atomic_write_json(_expert_path(dest, layer, expert), record)
                    records.append(record)
                    completed += 1
                    k2 = record["rates"]["K2"]["sqg_over_baseline"]
                    k4 = record["rates"]["K4"]["sqg_over_baseline"]
                    print(
                        f"[{completed:02d}/{total}] layer {layer} expert {expert}: "
                        f"K2={k2:.6f} K4={k4:.6f}",
                        flush=True,
                    )
                    torch.cuda.empty_cache()

    records = [
        _load_completed_expert(
            _expert_path(dest, layer, expert),
            layer=layer,
            expert=expert,
            manifest_sha256=manifest_sha256,
            pilot_kind=UNIFORM_RATE_PILOT_KIND,
        )
        for layer, expert in panel_cells(K4_PANEL)
    ]
    aggregate = aggregate_uniform_rate_records(records)
    report = {
        "kind": UNIFORM_RATE_PILOT_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "expert_count": len(records),
        "matrix_count": len(records) * len(PROJECTIONS),
        "rates": [2, 4],
        "aggregate": aggregate,
        "experts": records,
    }
    atomic_write_json(dest / "report.json", report)
    summary_path = dest / "summary.md"
    temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(uniform_rate_summary_markdown(report))
        temporary.replace(summary_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return report


def uniform_high_rate_summary_markdown(report: Mapping[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# GLM-5.2 uniform-K5/K6 codec-distortion pilot",
        "",
        "The same frozen 48-expert panel as the uniform K2/K4 study is used. "
        "Both MCG and SQG are freshly encoded from the official BF16 source at "
        "each rate.",
        "",
        "| Rate | Classification | MCG relative SSE | SQG relative SSE | "
        "SQG / MCG | Reduction | Wins/ties/losses |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for bits in UNIFORM_HIGH_RATE_BITS:
        label = f"K{bits}"
        item = aggregate[label]
        overall = item["overall"]
        lines.append(
            f"| {label} | `{item['classification']}` | "
            f"{overall['baseline_relative_sse']:.9g} | "
            f"{overall['sqg_relative_sse']:.9g} | "
            f"{overall['sqg_over_baseline']:.6f} | "
            f"{100.0 * overall['relative_reduction']:.3f}% | "
            f"{overall['expert_wins']}/{overall['expert_ties']}/"
            f"{overall['expert_losses']} |"
        )
    lines.extend(
        [
            "",
            "This is a uniform-rate raw weight-distortion study. It does not "
            "change the production QSRT rate set or evaluate end-to-end quality.",
            "",
        ]
    )
    return "\n".join(lines)


def run_uniform_high_rate_pilot(
    *,
    source_root: Path,
    baseline_root: Path,
    dest: Path,
    device: torch.device,
    resume: bool = False,
    exllamav3_root: Path = Path("/home/luke/projects/exllamav3"),
) -> dict[str, Any]:
    """Execute or resume the matched uniform-K5 and uniform-K6 study."""

    inventory = validate_inventory(
        source_root,
        baseline_root,
        panel=K4_PANEL,
        panel_rate=4,
        panel_seed=UNIFORM_RATE_PILOT_SEED,
    )
    tier_bitmap = _read_json_object(baseline_root / "tier_bitmap.json")
    validate_and_select_rate_panel(
        tier_bitmap,
        panel=K4_PANEL,
        eligible_rate=4,
        seed=UNIFORM_RATE_PILOT_SEED,
    )
    manifest = build_uniform_high_rate_manifest(
        source_root=source_root,
        baseline_root=baseline_root,
        device=device,
        inventory=inventory,
    )
    manifest_sha256 = prepare_destination(dest, manifest, resume=resume)
    source = IndexedTensorStore(source_root)
    pending: list[tuple[int, int]] = []
    for layer, expert in panel_cells(K4_PANEL):
        path = _expert_path(dest, layer, expert)
        if path.is_file():
            _load_completed_expert(
                path,
                layer=layer,
                expert=expert,
                manifest_sha256=manifest_sha256,
                pilot_kind=UNIFORM_HIGH_RATE_PILOT_KIND,
            )
        else:
            pending.append((layer, expert))

    if pending:
        quantizer_module = load_qsrt_encoder(exllamav3_root)
        install_sqg_quantizer(quantizer_module)
        pending_set = set(pending)
        completed = len(panel_cells(K4_PANEL)) - len(pending)
        total = len(panel_cells(K4_PANEL))
        for layer, experts in K4_PANEL.items():
            for expert in experts:
                if (layer, expert) not in pending_set:
                    continue
                started = time.monotonic()
                record = run_fresh_uniform_rate_expert(
                    source=source,
                    layer=layer,
                    expert=expert,
                    bits=UNIFORM_HIGH_RATE_BITS,
                    device=device,
                    quantizer_module=quantizer_module,
                    manifest_sha256=manifest_sha256,
                    sqg_codebook=SQG_FP16_D3L,
                )
                record["wall_seconds"] = time.monotonic() - started
                atomic_write_json(_expert_path(dest, layer, expert), record)
                completed += 1
                ratios = " ".join(
                    f"K{bits}={record['rates'][f'K{bits}']['sqg_over_baseline']:.6f}"
                    for bits in UNIFORM_HIGH_RATE_BITS
                )
                print(
                    f"[{completed:02d}/{total}] layer {layer} expert {expert}: "
                    f"{ratios}",
                    flush=True,
                )
                torch.cuda.empty_cache()

    records = [
        _load_completed_expert(
            _expert_path(dest, layer, expert),
            layer=layer,
            expert=expert,
            manifest_sha256=manifest_sha256,
            pilot_kind=UNIFORM_HIGH_RATE_PILOT_KIND,
        )
        for layer, expert in panel_cells(K4_PANEL)
    ]
    rate_labels = tuple(f"K{bits}" for bits in UNIFORM_HIGH_RATE_BITS)
    aggregate = aggregate_uniform_rate_records(records, rate_labels=rate_labels)
    report = {
        "kind": UNIFORM_HIGH_RATE_PILOT_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "expert_count": len(records),
        "matrix_count": len(records) * len(PROJECTIONS),
        "rates": list(UNIFORM_HIGH_RATE_BITS),
        "aggregate": aggregate,
        "experts": records,
    }
    atomic_write_json(dest / "report.json", report)
    summary_path = dest / "summary.md"
    temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(uniform_high_rate_summary_markdown(report))
        temporary.replace(summary_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return report


def k2_control_summary_markdown(report: Mapping[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# GLM-5.2 SQG Cheb-normal reconstruction controls",
        "",
        "| Rate | Arm | Arm / MCG | Relative reduction vs MCG | Wins |",
        "|---|---|---:|---:|---:|",
    ]
    for label in ("K2", "K4"):
        for arm, item in aggregate[label]["vs_mcg"].items():
            overall = item["overall"]
            lines.append(
                f"| {label} | {arm} | {overall['sqg_over_baseline']:.6f} | "
                f"{100.0 * overall['relative_reduction']:.4f}% | "
                f"{overall['expert_wins']}/{overall['expert_count']} |"
            )
    k2_exact = aggregate["K2"]["cheb_normal_vs_t12"]["overall"]
    k4_exact = aggregate["K4"]["cheb_normal_vs_t12"]["overall"]
    lines.extend(
        [
            "",
            "## Direct controls",
            "",
            f"- K2 exact Cheb-normal / T12: "
            f"{k2_exact['sqg_over_baseline']:.6f} "
            f"({100.0 * k2_exact['relative_reduction']:.4f}% lower is better)",
            f"- K4 exact Cheb-normal / T12: "
            f"{k4_exact['sqg_over_baseline']:.6f} "
            f"({100.0 * k4_exact['relative_reduction']:.4f}% lower is better)",
            "",
        ]
    )
    return "\n".join(lines)


def run_k2_sqg_control_pilot(
    *,
    source_root: Path,
    baseline_root: Path,
    base_study_root: Path,
    dest: Path,
    device: torch.device,
    resume: bool = False,
    exllamav3_root: Path = Path("/home/luke/projects/exllamav3"),
) -> dict[str, Any]:
    """Run exact Cheb-normal versus T12 controls at K2 and K4."""

    _, base_records, base_identity = load_uniform_rate_base_study(
        base_study_root
    )
    inventory = validate_inventory(
        source_root,
        baseline_root,
        panel=K4_PANEL,
        panel_rate=4,
        panel_seed=UNIFORM_RATE_PILOT_SEED,
    )
    manifest = build_k2_control_manifest(
        source_root=source_root,
        baseline_root=baseline_root,
        device=device,
        inventory=inventory,
        base_study=base_identity,
    )
    manifest_sha256 = prepare_destination(dest, manifest, resume=resume)
    source = IndexedTensorStore(source_root)
    records: list[dict[str, Any]] = []
    pending = []
    for layer, expert in panel_cells(K4_PANEL):
        path = _expert_path(dest, layer, expert)
        if path.is_file():
            records.append(
                _load_completed_expert(
                    path,
                    layer=layer,
                    expert=expert,
                    manifest_sha256=manifest_sha256,
                    pilot_kind=K2_CONTROL_PILOT_KIND,
                )
            )
        else:
            pending.append((layer, expert))

    if pending:
        quantizer_module = load_qsrt_encoder(exllamav3_root)
        install_sqg_quantizer(quantizer_module)
        completed = len(records)
        total = len(panel_cells(K4_PANEL))
        for layer, expert in pending:
            started = time.monotonic()
            record = run_k2_sqg_control_expert(
                source=source,
                base_record=base_records[(layer, expert)],
                layer=layer,
                expert=expert,
                device=device,
                quantizer_module=quantizer_module,
                manifest_sha256=manifest_sha256,
            )
            record["wall_seconds"] = time.monotonic() - started
            atomic_write_json(_expert_path(dest, layer, expert), record)
            records.append(record)
            completed += 1
            k2 = record["rates"]["K2"]["arms"]
            k4 = record["rates"]["K4"]["arms"]
            print(
                f"[{completed:02d}/{total}] layer {layer} expert {expert}: "
                f"K2 exact/T12={k2['cheb_normal']['sse'] / k2['t12']['sse']:.6f} "
                f"K4 exact/T12={k4['cheb_normal']['sse'] / k4['t12']['sse']:.6f}",
                flush=True,
            )
            torch.cuda.empty_cache()

    records = [
        _load_completed_expert(
            _expert_path(dest, layer, expert),
            layer=layer,
            expert=expert,
            manifest_sha256=manifest_sha256,
            pilot_kind=K2_CONTROL_PILOT_KIND,
        )
        for layer, expert in panel_cells(K4_PANEL)
    ]
    aggregate = aggregate_k2_sqg_control_records(records)
    report = {
        "kind": K2_CONTROL_PILOT_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "base_study": base_identity,
        "expert_count": len(records),
        "matrix_count": len(records) * len(PROJECTIONS),
        "aggregate": aggregate,
        "experts": records,
    }
    atomic_write_json(dest / "report.json", report)
    summary_path = dest / "summary.md"
    temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(k2_control_summary_markdown(report))
        temporary.replace(summary_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return report

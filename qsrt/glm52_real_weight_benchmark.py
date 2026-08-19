"""Real-weight GLM-5.2 codec benchmark with a bounded BF16 source window.

The refreshed 3.5-bpw EXL3 checkpoint stores routed experts in one R7 shard
per layer.  The older GLM experiment adapter predates that layout and requires
the complete 282-shard BF16 checkpoint.  This module validates the R7 layout,
uses its rate map only to stratify an error-independent expert sample, and
freshly encodes matched MCG and QSRT candidates from the official BF16 source.

The resulting measurements are raw weight-distortion controls.  They do not
measure full-model forward KLD and must not be presented as model-quality
evidence.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from qsrt.correctness import sha256_file
from qsrt.exl3_loader import load_qsrt_encoder
from qsrt.glm52_pilot import (
    EXPERTS_PER_LAYER,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    PROJECTIONS,
    SOURCE_CONFIG_SHA256,
    SOURCE_INDEX_SHA256,
    SOURCE_REVISION,
    IndexedTensorStore,
    _expert_path,
    _load_completed_expert,
    _read_json_object,
    _read_safetensors_header,
    _validate_header_tensor,
    _verify_manifest_entry,
    aggregate_uniform_rate_records,
    atomic_write_json,
    panel_cells,
    prepare_destination,
    run_fresh_uniform_rate_expert,
    source_tensor_name,
)
from qsrt.sqg_quantizer import install_sqg_quantizer


REAL_WEIGHT_CODEC_BENCHMARK_KIND = (
    "qsrt_glm52_real_weight_codec_benchmark_v1"
)
REAL_WEIGHT_PANEL_SCHEMA = "qsrt_glm52_real_weight_panel"
EXL3_ENDPOINT_REVISION = "9ab9579774cc432df91567a36f6e9e863e0d4c9f"
EXL3_CONFIG_SHA256 = (
    "fabb73eb513ec64f3a365da396b38de8d55b3930edfb11baeecbf34ecafa6126"
)
EXL3_ROOT_MANIFEST_SHA256 = (
    "c81f3129e418683b6e37c17b8198681c11324f3918fdbd844c1be346114c387b"
)
EXL3_MANIFEST_JSON_SHA256 = (
    "df2f4c87b22c21c5234ef216149f5b5adc556820bb97ae4bc6dd7f4f0647b8db"
)
SOURCE_INVENTORY_FILE_SHA256 = (
    "7472012f8fb42968d78a73a765bf3d729fb66c017a2fecca1a9919495d61cfca"
)
SOURCE_INVENTORY_CONTENT_SHA256 = (
    "3b50d0ddceebaa849fe1153917b5ac9332b27ce7a445a960a6132a513998335f"
)
R7_LAYER_SIDECAR_SHA256 = {
    3: "36e623438c17bf8d143e3d81f712856421ab9fa8155c76ccb7482576c2eb410c",
    52: "dbdd253ffb2e6204c1577addc65bbad3c3939ee2eb9efb774bde6d477f3f429d",
    55: "a90e97c91c7a5362c006a80ef4e35e0df43ebee4daa25d5be8687e257fd61fb8",
    56: "04203db9b9aaee1d6688d39f71cec5e4af46b4f48d61fcce5890f14802b1dac6",
    57: "b61ff67b3e9aaf309621523fa2526ca2d63a3377b2413301bb3873bc902b1867",
    58: "88af2a8b8495df42f2e145b6c7a924a4a455b39455aab6037899fe0d28909305",
    60: "9ea7033e3dcafc296d8979f98e248d6827a37f037de8c7154439de4d6e8f3660",
    63: "0a753793cdb899642388a1d5801073adb34f2cf8099d290f818039f4c81e8722",
    64: "599b2f3d691a6e2d88d79cc6520bc0a758bf5a31c0cf2426cc7cb5c53f05bbe6",
}
R7_MARKER = "CODEX_ROUND7"
R7_RECIPE_VERSION = "tr3-v4-r7-draft-1"
R7_SCHEMA_VERSION = 2
R7_RATE_SELECTION_SEED = "glm52-r7-rate-pattern-stratified-real-weight-v1"


def load_frozen_real_weight_panel(path: Path, *, layer: int) -> dict[str, Any]:
    """Load a source-controlled expert list frozen before candidate scoring."""

    path = path.resolve()
    panel = _read_json_object(path)
    required = {
        "schema": REAL_WEIGHT_PANEL_SCHEMA,
        "schema_version": 1,
        "selection_status": "frozen_before_candidate_measurement",
        "selection_seed": R7_RATE_SELECTION_SEED,
        "layer": layer,
    }
    for field, expected in required.items():
        if panel.get(field) != expected:
            raise ValueError(
                f"frozen panel field {field!r} is {panel.get(field)!r}, "
                f"expected {expected!r}"
            )
    expected_source = {
        "model_id": "zai-org/GLM-5.2",
        "revision": SOURCE_REVISION,
        "config_sha256": SOURCE_CONFIG_SHA256,
        "index_sha256": SOURCE_INDEX_SHA256,
    }
    if panel.get("source") != expected_source:
        raise ValueError("frozen panel source identity mismatch")
    expected_comparison = {
        "model_id": "brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78",
        "revision": EXL3_ENDPOINT_REVISION,
        "manifest_sha256": EXL3_ROOT_MANIFEST_SHA256,
        "layer_sidecar_sha256": R7_LAYER_SIDECAR_SHA256[layer],
    }
    if panel.get("comparison_checkpoint") != expected_comparison:
        raise ValueError("frozen panel comparison-checkpoint identity mismatch")
    entries = panel.get("experts")
    if not isinstance(entries, list) or not entries:
        raise ValueError("frozen panel must contain at least one expert")
    experts: list[int] = []
    rate_patterns: dict[int, tuple[int, int, int]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TypeError(f"frozen panel expert entry {index} must be an object")
        expert = entry.get("expert")
        rates = entry.get("exl3_rates")
        if (
            isinstance(expert, bool)
            or not isinstance(expert, int)
            or not 0 <= expert < EXPERTS_PER_LAYER
        ):
            raise ValueError(f"frozen panel expert entry {index} has an invalid ID")
        if (
            not isinstance(rates, list)
            or len(rates) != len(PROJECTIONS)
            or any(
                isinstance(rate, bool)
                or not isinstance(rate, int)
                or rate not in (3, 4, 5)
                for rate in rates
            )
        ):
            raise ValueError(
                f"frozen panel expert {expert} has invalid gate, up, and down rates"
            )
        if expert in rate_patterns:
            raise ValueError(f"frozen panel repeats expert {expert}")
        experts.append(expert)
        rate_patterns[expert] = tuple(rates)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "experts": tuple(experts),
        "rate_patterns": rate_patterns,
        "evidence_role": panel.get("evidence_role"),
    }


def select_frozen_panel_slice(
    frozen_panel: Mapping[str, Any], *, offset: int, expert_count: int
) -> tuple[int, ...]:
    """Select a disjoint contiguous slice from a source-controlled panel."""

    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("panel offset must be a nonnegative integer")
    if (
        isinstance(expert_count, bool)
        or not isinstance(expert_count, int)
        or expert_count < 1
    ):
        raise ValueError("expert_count must be a positive integer")
    experts = tuple(int(expert) for expert in frozen_panel["experts"])
    stop = offset + expert_count
    if stop > len(experts):
        raise ValueError("the requested slice exceeds the frozen panel length")
    return experts[offset:stop]


def _validate_glm_source_config(config: Mapping[str, Any]) -> None:
    expected = {
        "architectures": ["GlmMoeDsaForCausalLM"],
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
    for field, expected_value in expected.items():
        if config.get(field) != expected_value:
            raise ValueError(
                f"official source config field {field!r} is "
                f"{config.get(field)!r}, expected {expected_value!r}"
            )


def _normalized_panel(
    panel: Mapping[int, Sequence[int]],
) -> dict[int, tuple[int, ...]]:
    result: dict[int, tuple[int, ...]] = {}
    for raw_layer, raw_experts in panel.items():
        layer = int(raw_layer)
        experts = tuple(int(expert) for expert in raw_experts)
        if layer < 3 or layer > 77:
            raise ValueError(f"GLM routed-expert layer {layer} is outside 3..77")
        if not experts:
            raise ValueError(f"GLM layer {layer} has no selected experts")
        if len(set(experts)) != len(experts):
            raise ValueError(f"GLM layer {layer} repeats an expert ID")
        if any(expert < 0 or expert >= EXPERTS_PER_LAYER for expert in experts):
            raise ValueError(f"GLM layer {layer} has an expert outside 0..255")
        result[layer] = experts
    if not result:
        raise ValueError("the real-weight benchmark panel is empty")
    return dict(sorted(result.items()))


def validate_bounded_source_window(
    source_root: Path,
    source_inventory_path: Path,
    *,
    panel: Mapping[int, Sequence[int]],
    verify_shard_hashes: bool = True,
) -> dict[str, Any]:
    """Validate only the official BF16 shards needed by ``panel``.

    The official index still closes the complete tensor namespace.  The sealed
    source inventory closes file and tensor identities for the bounded window,
    so unrelated source shards may remain absent.
    """

    selected_panel = _normalized_panel(panel)
    source_root = source_root.resolve()
    source_inventory_path = source_inventory_path.resolve()
    config_path = source_root / "config.json"
    index_path = source_root / "model.safetensors.index.json"
    for path in (config_path, index_path, source_inventory_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(config_path) != SOURCE_CONFIG_SHA256:
        raise ValueError("official GLM-5.2 config identity mismatch")
    if sha256_file(index_path) != SOURCE_INDEX_SHA256:
        raise ValueError("official GLM-5.2 tensor index identity mismatch")
    if sha256_file(source_inventory_path) != SOURCE_INVENTORY_FILE_SHA256:
        raise ValueError("official GLM-5.2 source inventory identity mismatch")

    config = _read_json_object(config_path)
    _validate_glm_source_config(config)
    inventory = _read_json_object(source_inventory_path)
    required_inventory_fields = {
        "marker": R7_MARKER,
        "recipe_version": R7_RECIPE_VERSION,
        "schema": "r7-checkpoint-inventory-v1",
        "role": "bf16-source",
        "config_sha256": SOURCE_CONFIG_SHA256,
        "index_sha256": SOURCE_INDEX_SHA256,
        "inventory_sha256": SOURCE_INVENTORY_CONTENT_SHA256,
        "routed_bf16_validated": True,
    }
    for field, expected_value in required_inventory_fields.items():
        if inventory.get(field) != expected_value:
            raise ValueError(
                f"source inventory field {field!r} is {inventory.get(field)!r}, "
                f"expected {expected_value!r}"
            )
    inventory_entries = inventory.get("entries")
    inventory_shards = inventory.get("shards")
    if not isinstance(inventory_entries, dict) or not isinstance(
        inventory_shards, dict
    ):
        raise TypeError("source inventory must contain entry and shard objects")

    source = IndexedTensorStore(source_root)
    selected_shards: set[str] = set()
    selected_tensor_count = 0
    for layer, experts in selected_panel.items():
        for expert in experts:
            for spec in PROJECTIONS:
                name = source_tensor_name(layer, expert, spec.name)
                filename = source.filename(name)
                entry = inventory_entries.get(name)
                if not isinstance(entry, dict):
                    raise KeyError(f"source inventory does not contain {name}")
                expected_nbytes = spec.source_shape[0] * spec.source_shape[1] * 2
                expected_entry = {
                    "dtype": "BF16",
                    "shape": list(spec.source_shape),
                    "nbytes": expected_nbytes,
                    "shard": filename,
                }
                for field, expected_value in expected_entry.items():
                    if entry.get(field) != expected_value:
                        raise ValueError(
                            f"source inventory {name} field {field!r} is "
                            f"{entry.get(field)!r}, expected {expected_value!r}"
                        )
                source.validate_tensor(name, dtype="BF16", shape=spec.source_shape)
                selected_shards.add(filename)
                selected_tensor_count += 1

    shard_receipts: list[dict[str, Any]] = []
    for filename in sorted(selected_shards):
        shard_metadata = inventory_shards.get(filename)
        if not isinstance(shard_metadata, dict):
            raise KeyError(f"source inventory does not contain shard {filename}")
        path = source_root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_size = int(shard_metadata.get("size", -1))
        if path.stat().st_size != expected_size:
            raise ValueError(
                f"source shard {filename} is {path.stat().st_size} bytes, "
                f"expected {expected_size}"
            )
        expected_sha256 = shard_metadata.get("file_sha256")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError(f"source shard {filename} has no valid SHA-256")
        if verify_shard_hashes:
            _verify_manifest_entry(path, expected_sha256)
        shard_receipts.append(
            {
                "name": filename,
                "size": expected_size,
                "sha256": expected_sha256,
                "sha256_verified": verify_shard_hashes,
            }
        )

    return {
        "model_id": "zai-org/GLM-5.2",
        "revision": SOURCE_REVISION,
        "root": str(source_root),
        "config_sha256": SOURCE_CONFIG_SHA256,
        "index_sha256": SOURCE_INDEX_SHA256,
        "source_inventory_path": str(source_inventory_path),
        "source_inventory_sha256": SOURCE_INVENTORY_FILE_SHA256,
        "selected_tensor_count": selected_tensor_count,
        "selected_shards": shard_receipts,
        "complete_checkpoint_required": False,
    }


def _r7_projection_key(layer: int, expert: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}"


def validate_r7_rate_map(
    sidecar: Mapping[str, Any], *, layer: int
) -> dict[int, tuple[int, int, int]]:
    """Validate one complete R7 layer rate map and return expert patterns."""

    required = {
        "marker": R7_MARKER,
        "schema_version": R7_SCHEMA_VERSION,
        "recipe_version": R7_RECIPE_VERSION,
        "layer": layer,
        "allocation_bit_units": 2688,
        "shard": f"r7-experts-layer-{layer:03d}.safetensors",
    }
    for field, expected_value in required.items():
        if sidecar.get(field) != expected_value:
            raise ValueError(
                f"R7 layer {layer} field {field!r} is {sidecar.get(field)!r}, "
                f"expected {expected_value!r}"
            )
    raw_target_bpw = sidecar.get("allocation_target_bpw")
    try:
        target_bpw = float(raw_target_bpw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"R7 layer {layer} has an invalid allocation_target_bpw"
        ) from exc
    if isinstance(raw_target_bpw, bool) or target_bpw != 3.5:
        raise ValueError(
            f"R7 layer {layer} allocation_target_bpw is {raw_target_bpw!r}, "
            "expected 3.5"
        )
    bit_map = sidecar.get("bit_map")
    if not isinstance(bit_map, dict):
        raise TypeError(f"R7 layer {layer} bit_map must be an object")
    expected_keys = {
        _r7_projection_key(layer, expert, spec.name)
        for expert in range(EXPERTS_PER_LAYER)
        for spec in PROJECTIONS
    }
    if set(bit_map) != expected_keys:
        missing = len(expected_keys - set(bit_map))
        extra = len(set(bit_map) - expected_keys)
        raise ValueError(
            f"R7 layer {layer} rate map has {missing} missing and {extra} extra entries"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value not in (3, 4, 5)
        for value in bit_map.values()
    ):
        raise ValueError(f"R7 layer {layer} contains a rate outside K3/K4/K5")
    if sum(int(value) for value in bit_map.values()) != 2688:
        raise ValueError(f"R7 layer {layer} rate map does not close to 3.5 bpw")

    return {
        expert: tuple(
            int(bit_map[_r7_projection_key(layer, expert, spec.name)])
            for spec in PROJECTIONS
        )
        for expert in range(EXPERTS_PER_LAYER)
    }


def select_rate_pattern_stratified_experts(
    patterns: Mapping[int, Sequence[int]],
    *,
    layer: int,
    expert_count: int,
    seed: str = R7_RATE_SELECTION_SEED,
) -> tuple[int, ...]:
    """Select deterministic experts while covering distinct R7 rate patterns."""

    if (
        isinstance(expert_count, bool)
        or not isinstance(expert_count, int)
        or not 1 <= expert_count <= EXPERTS_PER_LAYER
    ):
        raise ValueError("expert_count must be an integer between 1 and 256")
    if set(patterns) != set(range(EXPERTS_PER_LAYER)):
        raise ValueError("rate patterns must cover expert IDs 0..255 exactly")
    grouped: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for expert, raw_pattern in patterns.items():
        pattern = tuple(int(value) for value in raw_pattern)
        if len(pattern) != len(PROJECTIONS) or any(
            value not in (3, 4, 5) for value in pattern
        ):
            raise ValueError(f"expert {expert} has an invalid R7 rate pattern")
        grouped[pattern].append(int(expert))

    def expert_key(expert: int) -> bytes:
        return hashlib.sha256(
            f"{seed}:{layer}:{expert}".encode("ascii")
        ).digest()

    for experts in grouped.values():
        experts.sort(key=expert_key)
    ordered_patterns = tuple(
        sorted(grouped, key=lambda pattern: (-len(grouped[pattern]), pattern))
    )
    selected: list[int] = []
    offset = 0
    while len(selected) < expert_count:
        added = False
        for pattern in ordered_patterns:
            experts = grouped[pattern]
            if offset < len(experts):
                selected.append(experts[offset])
                added = True
                if len(selected) == expert_count:
                    break
        if not added:
            raise AssertionError("R7 expert groups could not satisfy the requested count")
        offset += 1
    return tuple(selected)


def validate_r7_endpoint_layer(
    endpoint_root: Path,
    *,
    layer: int,
    expert_count: int,
    selected_experts: Sequence[int] | None = None,
    verify_shard_hash: bool = True,
) -> dict[str, Any]:
    """Validate the corrected R7 endpoint and select one layer's benchmark panel."""

    endpoint_root = endpoint_root.resolve()
    try:
        expected_sidecar_sha256 = R7_LAYER_SIDECAR_SHA256[layer]
    except KeyError as exc:
        raise ValueError(
            f"no pinned R7 sidecar identity is registered for layer {layer}"
        ) from exc
    paths = {
        "config": endpoint_root / "config.json",
        "root_manifest": endpoint_root / "MANIFEST.sha256",
        "manifest": endpoint_root / "MANIFEST.json",
        "sidecar": endpoint_root / f"r7-experts-layer-{layer:03d}.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_identities = {
        "config": EXL3_CONFIG_SHA256,
        "root_manifest": EXL3_ROOT_MANIFEST_SHA256,
        "manifest": EXL3_MANIFEST_JSON_SHA256,
        "sidecar": expected_sidecar_sha256,
    }
    for role, expected_sha256 in expected_identities.items():
        actual_sha256 = sha256_file(paths[role])
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"EXL3 {role} identity mismatch: {actual_sha256} != {expected_sha256}"
            )
    root_manifest_lines = paths["root_manifest"].read_text().splitlines()
    if root_manifest_lines != [f"{EXL3_MANIFEST_JSON_SHA256}  MANIFEST.json"]:
        raise ValueError("EXL3 root manifest does not bind only MANIFEST.json")

    config = _read_json_object(paths["config"])
    _validate_glm_source_config(config)
    manifest = _read_json_object(paths["manifest"])
    expected_manifest_fields = {
        "marker": R7_MARKER,
        "schema": "r7-complete-v2-checkpoint-v1",
        "recipe_version": R7_RECIPE_VERSION,
        "requires_loader_feature": "r7-asymmetric-two-stack",
    }
    for field, expected_value in expected_manifest_fields.items():
        if manifest.get(field) != expected_value:
            raise ValueError(
                f"EXL3 manifest field {field!r} is {manifest.get(field)!r}, "
                f"expected {expected_value!r}"
            )
    layer_entry = manifest.get("layers", {}).get(str(layer))
    expected_layer_entry = {
        "bit_units": 2688,
        "manifest": paths["sidecar"].name,
        "shard": f"r7-experts-layer-{layer:03d}.safetensors",
    }
    if layer_entry != expected_layer_entry:
        raise ValueError(f"EXL3 manifest layer {layer} entry is inconsistent")

    sidecar = _read_json_object(paths["sidecar"])
    patterns = validate_r7_rate_map(sidecar, layer=layer)
    if selected_experts is None:
        experts = select_rate_pattern_stratified_experts(
            patterns, layer=layer, expert_count=expert_count
        )
    else:
        experts = tuple(int(expert) for expert in selected_experts)
        if len(experts) != expert_count:
            raise ValueError(
                "the frozen expert list length does not match expert_count"
            )
        if len(set(experts)) != len(experts):
            raise ValueError("the frozen expert list repeats an expert")
        if any(expert < 0 or expert >= EXPERTS_PER_LAYER for expert in experts):
            raise ValueError("the frozen expert list contains an invalid expert ID")
    shard_name = str(sidecar["shard"])
    shard_path = endpoint_root / shard_name
    if not shard_path.is_file():
        raise FileNotFoundError(shard_path)
    shard_sha256 = sidecar.get("shard_sha256")
    if not isinstance(shard_sha256, str) or len(shard_sha256) != 64:
        raise ValueError(f"R7 layer {layer} sidecar has no valid shard SHA-256")
    if verify_shard_hash:
        _verify_manifest_entry(shard_path, shard_sha256)

    header = _read_safetensors_header(shard_path)
    for expert in experts:
        for spec, bits in zip(PROJECTIONS, patterns[expert], strict=True):
            prefix = _r7_projection_key(layer, expert, spec.name)
            _validate_header_tensor(
                header,
                f"{prefix}.trellis",
                dtype="I16",
                shape=(
                    spec.encoder_shape[0] // 16,
                    spec.encoder_shape[1] // 16,
                    16 * bits,
                ),
            )
            scale_length = (
                spec.encoder_shape[0]
                if spec.expert_scale == "suh"
                else spec.encoder_shape[1]
            )
            _validate_header_tensor(
                header,
                f"{prefix}.{spec.expert_scale}",
                dtype="F16",
                shape=(scale_length,),
            )
            _validate_header_tensor(
                header, f"{prefix}.mcg", dtype="I32", shape=()
            )

    pattern_counts = Counter(patterns.values())
    return {
        "model_id": "brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78",
        "revision": EXL3_ENDPOINT_REVISION,
        "root": str(endpoint_root),
        "manifest_sha256": EXL3_ROOT_MANIFEST_SHA256,
        "manifest_json_sha256": EXL3_MANIFEST_JSON_SHA256,
        "layer": layer,
        "sidecar_sha256": expected_sidecar_sha256,
        "shard": shard_name,
        "shard_sha256": shard_sha256,
        "shard_sha256_verified": verify_shard_hash,
        "allocation_bpw": 3.5,
        "rate_pattern_counts": {
            "/".join(map(str, pattern)): count
            for pattern, count in sorted(pattern_counts.items())
        },
        "selected_experts": list(experts),
        "selected_rate_patterns": {
            str(expert): list(patterns[expert]) for expert in experts
        },
        "selection_role": (
            "coverage stratification only; the EXL3 rate map is not a QSRT "
            "quality target or population weight"
        ),
    }


def run_real_weight_codec_benchmark(
    *,
    source_root: Path,
    source_inventory_path: Path,
    exl3_endpoint_root: Path,
    dest: Path,
    layer: int,
    expert_count: int,
    bits: Sequence[int],
    device: torch.device,
    exllamav3_root: Path,
    panel_manifest_path: Path | None = None,
    panel_offset: int = 0,
    resume: bool = False,
    verify_source_shard_hashes: bool = True,
    verify_exl3_shard_hash: bool = True,
    return_trellis_diagnostics: bool = False,
) -> dict[str, Any]:
    """Freshly compare matched MCG and QSRT encodes on real BF16 experts."""

    rates = tuple(int(value) for value in bits)
    if (
        not rates
        or tuple(sorted(set(rates))) != rates
        or any(value not in range(2, 7) for value in rates)
    ):
        raise ValueError("benchmark rates must be unique ordered K2 through K6")
    frozen_panel = (
        load_frozen_real_weight_panel(panel_manifest_path, layer=layer)
        if panel_manifest_path is not None
        else None
    )
    selected_frozen_experts = (
        select_frozen_panel_slice(
            frozen_panel, offset=panel_offset, expert_count=expert_count
        )
        if frozen_panel is not None
        else None
    )
    if frozen_panel is None and panel_offset != 0:
        raise ValueError("panel_offset requires a frozen panel manifest")
    endpoint = validate_r7_endpoint_layer(
        exl3_endpoint_root,
        layer=layer,
        expert_count=expert_count,
        selected_experts=selected_frozen_experts,
        verify_shard_hash=verify_exl3_shard_hash,
    )
    if frozen_panel is not None:
        endpoint_patterns = {
            int(expert): tuple(rates)
            for expert, rates in endpoint["selected_rate_patterns"].items()
        }
        expected_patterns = {
            expert: frozen_panel["rate_patterns"][expert]
            for expert in selected_frozen_experts
        }
        if endpoint_patterns != expected_patterns:
            raise ValueError(
                "the frozen panel rate patterns do not match the endpoint sidecar"
            )
    panel = {layer: tuple(int(value) for value in endpoint["selected_experts"])}
    source_inventory = validate_bounded_source_window(
        source_root,
        source_inventory_path,
        panel=panel,
        verify_shard_hashes=verify_source_shard_hashes,
    )
    manifest = {
        "kind": f"{REAL_WEIGHT_CODEC_BENCHMARK_KIND}_manifest",
        "source": source_inventory,
        "exl3_endpoint": endpoint,
        "panel": {str(layer): list(panel[layer])},
        "frozen_panel": (
            {
                "path": frozen_panel["path"],
                "sha256": frozen_panel["sha256"],
                "evidence_role": frozen_panel["evidence_role"],
                "selected_offset": panel_offset,
                "selected_count": expert_count,
            }
            if frozen_panel is not None
            else None
        ),
        "rates": list(rates),
        "codec_arms": {
            "control": "fresh MCG encode from official BF16",
            "candidate": "fresh sqg_xor_cheb_t12 encode from official BF16",
            "same_transform_feedback_and_scale_search": True,
        },
        "trellis_diagnostics": {
            "enabled": return_trellis_diagnostics,
            "scope": (
                "exact normalized SQG tiles presented to Viterbi after "
                "BlockLDLQ feedback"
            ),
        },
        "device": str(device),
        "evidence_boundary": (
            "raw weight distortion on a rate-pattern-stratified real-weight "
            "panel; not full-model KLD, task quality, or a population estimate"
        ),
    }
    manifest_sha256 = prepare_destination(dest, manifest, resume=resume)
    source = IndexedTensorStore(source_root)
    records: list[dict[str, Any]] = []
    pending: list[tuple[int, int]] = []
    for panel_layer, expert in panel_cells(panel):
        path = _expert_path(dest, panel_layer, expert)
        if path.is_file():
            records.append(
                _load_completed_expert(
                    path,
                    layer=panel_layer,
                    expert=expert,
                    manifest_sha256=manifest_sha256,
                    pilot_kind=REAL_WEIGHT_CODEC_BENCHMARK_KIND,
                )
            )
        else:
            pending.append((panel_layer, expert))

    if pending:
        quantizer_module = load_qsrt_encoder(exllamav3_root)
        install_sqg_quantizer(quantizer_module)
        completed = len(records)
        for panel_layer, expert in pending:
            started = time.monotonic()
            record = run_fresh_uniform_rate_expert(
                source=source,
                layer=panel_layer,
                expert=expert,
                bits=rates,
                device=device,
                quantizer_module=quantizer_module,
                manifest_sha256=manifest_sha256,
                pilot_kind=REAL_WEIGHT_CODEC_BENCHMARK_KIND,
                return_trellis_diagnostics=return_trellis_diagnostics,
            )
            record["wall_seconds"] = time.monotonic() - started
            atomic_write_json(_expert_path(dest, panel_layer, expert), record)
            records.append(record)
            completed += 1
            ratios = " ".join(
                f"K{rate}={record['rates'][f'K{rate}']['sqg_over_baseline']:.6f}"
                for rate in rates
            )
            print(
                f"[{completed:02d}/{len(panel_cells(panel))}] layer "
                f"{panel_layer} expert {expert}: {ratios}",
                flush=True,
            )
            torch.cuda.empty_cache()

    records = [
        _load_completed_expert(
            _expert_path(dest, panel_layer, expert),
            layer=panel_layer,
            expert=expert,
            manifest_sha256=manifest_sha256,
            pilot_kind=REAL_WEIGHT_CODEC_BENCHMARK_KIND,
        )
        for panel_layer, expert in panel_cells(panel)
    ]
    rate_labels = tuple(f"K{rate}" for rate in rates)
    report = {
        "kind": REAL_WEIGHT_CODEC_BENCHMARK_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "expert_count": len(records),
        "matrix_count": len(records) * len(PROJECTIONS),
        "rates": list(rates),
        "panel": manifest["panel"],
        "aggregate": aggregate_uniform_rate_records(
            records, rate_labels=rate_labels, panel=panel
        ),
        "experts": records,
        "evidence_boundary": manifest["evidence_boundary"],
    }
    atomic_write_json(dest / "report.json", report)
    return report

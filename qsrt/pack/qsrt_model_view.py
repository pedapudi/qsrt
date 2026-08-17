"""Publish a canonical QSRT artifact as a local Hugging Face model view.

The view contains no prepared tensor-parallel checkpoint data.  It links the
TP-independent QSRT atom/X4T files, the reusable MXFP8 non-expert overlay, and
the immutable model metadata into one directory that vLLM can open directly.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from qsrt import constants as C
from qsrt.exl3_reference import CODEBOOK_SQG_XOR_CHEB_T12
from qsrt.io.hf_cache import resolve
from qsrt.pack.package_helpers import (
    DEFAULT_NONEXPERT,
    IGNORED_DENSE_LAYERS,
    _atomic_json,
    _auxiliary_sources,
    _fsync_directory,
    _link,
    _nonexpert_weight_map,
    _read_json,
    _require_exact_link,
    _scan_nonexpert_weight_map,
)
from qsrt.pack.qsrt_allocation import (
    HIGH_TIER_STORAGE,
    QSRT_ALLOCATION_KIND,
    QSRT_ALLOCATION_SCHEMA_VERSION,
)
from qsrt.pack.qsrt_pool import load_qsrt_candidate_pool
from qsrt.pack.qsrt_materialize import (
    QSRT_ALLOCATION_COPY_FILENAME,
    QSRT_MANIFEST_FILENAME,
)
from qsrt.pack.qsrt_validate import validate_qsrt_artifact
from qsrt.qsrt import FORMAT_X4T, H308, K2, PHASE1_MODE_IDS, ExpertFormatSpec
from qsrt.qsrt_coupled_plan import CoupledRotationPlan
from qsrt.qsrt_atoms_v2 import COUPLED_H308_PROFILE as ATOMS_V2_COUPLED_H308_PROFILE
from qsrt.qsrt_atoms_v2 import PROFILE as ATOMS_V2_PROFILE
from qsrt.qsrt_atoms_v2 import PURE_K2_PROFILE as ATOMS_V2_PURE_K2_PROFILE
from qsrt.qsrt_atoms_v2 import SCHEMA as ATOMS_V2_SCHEMA
from qsrt.qsrt_atoms_v2 import SUPPORTED_PROFILES as ATOMS_V2_PROFILES
from qsrt.pack.qsrt_atoms_v2 import (
    QSRTAtomsV2Reader,
    layer_filename,
    materialize_atoms_v2_layer,
)


QSRT_MODEL_VIEW_KIND = "qsrt_kimi_k3_qsrt_model_view"
QSRT_MODEL_VIEW_SCHEMA_VERSION = 1
QSRT_MODEL_VIEW_MANIFEST = "qsrt-model-view.json"


def qsrt_hybrid_bit_map(allocation: dict) -> dict[str, list[int]]:
    """Derive the vLLM tier map from an authenticated QSRT allocation."""

    if allocation.get("kind") != QSRT_ALLOCATION_KIND:
        raise ValueError("QSRT allocation kind mismatch")
    if allocation.get("schema_version") != QSRT_ALLOCATION_SCHEMA_VERSION:
        raise ValueError("QSRT allocation schema mismatch")
    meta = allocation.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("QSRT allocation has no meta object")
    expected_meta = {
        "codec": "QSRT",
        "high_tier_storage": HIGH_TIER_STORAGE,
        "candidate_codebook": CODEBOOK_SQG_XOR_CHEB_T12,
        "candidate_mode_ids": list(PHASE1_MODE_IDS),
    }
    for name, expected in expected_meta.items():
        if meta.get(name) != expected:
            raise ValueError(f"QSRT allocation {name} drifted")
    layers = allocation.get("layers")
    if not isinstance(layers, dict) or set(layers) != {
        str(layer) for layer in C.MOE_LAYERS
    }:
        raise ValueError("QSRT allocation must contain exactly 92 MoE layers")

    result: dict[str, list[int]] = {}
    for layer in C.MOE_LAYERS:
        entry = layers[str(layer)]
        if not isinstance(entry, dict):
            raise ValueError(f"QSRT layer {layer} allocation is malformed")
        codes = entry.get("format_codes")
        if not isinstance(codes, list) or len(codes) != C.NUM_EXPERTS:
            raise ValueError(
                f"QSRT layer {layer} must contain {C.NUM_EXPERTS} format codes"
            )
        bits: list[int] = []
        for raw_code in codes:
            if isinstance(raw_code, bool) or not isinstance(raw_code, int):
                raise ValueError(f"QSRT layer {layer} has a non-integer code")
            spec = ExpertFormatSpec.from_code(raw_code)
            bits.append(4 if spec.is_x4t else 3)
        expected_x4t = [expert for expert, bit in enumerate(bits) if bit == 4]
        expected_compressed = [
            expert for expert, bit in enumerate(bits) if bit == 3
        ]
        if (
            entry.get("x4t") != expected_x4t
            or entry.get("compressed") != expected_compressed
        ):
            raise ValueError(
                f"QSRT layer {layer} tiers disagree with its format codes"
            )
        if any(
            (code == FORMAT_X4T) != (bit == 4)
            for code, bit in zip(codes, bits, strict=True)
        ):
            raise AssertionError("QSRT tier-map derivation drifted")
        result[str(layer)] = bits
    return result


def qsrt_quantization_config(allocation: dict) -> dict:
    """Return the explicit TP-independent QSRT/X4T runtime contract."""

    return {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "group_size": 16,
        "hybrid_bit_map": qsrt_hybrid_bit_map(allocation),
        "kept_format": "mxfp4_e8m0k32",
        "kept_storage": "x4t",
        "demoted_format": "qsrt_sqg_e4m3",
        "qsrt": {
            "schema": "qsrt_kimi_k3_qsrt_atoms_v1",
            "storage_format": "qsrt_atoms_v1",
            "encoding": "qsrt_sqg_e4m3",
            "codebook": "sqg_xor_cheb_t12",
            "artifact_manifest": QSRT_MANIFEST_FILENAME,
        },
        "dense_format": "mxfp8",
        "ignored_layers": list(IGNORED_DENSE_LAYERS),
    }


def qsrt_atoms_v2_quantization_config(
    profile: str = ATOMS_V2_PROFILE,
) -> dict:
    """Return the runtime contract for one all-QSRT atoms-v2 profile."""

    if profile not in ATOMS_V2_PROFILES:
        raise ValueError(f"unsupported QSRT atoms-v2 profile {profile!r}")
    bits = 2 if profile == ATOMS_V2_PURE_K2_PROFILE else 3

    return {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "group_size": 16,
        "hybrid_bit_map": {
            str(layer): [bits] * C.NUM_EXPERTS for layer in C.MOE_LAYERS
        },
        "kept_format": "mxfp4_e8m0k32",
        "kept_storage": "x4t",
        "demoted_format": "qsrt_sqg_e4m3",
        "qsrt": {
            "schema": ATOMS_V2_SCHEMA,
            "storage_format": "qsrt_atoms_v2",
            "encoding": "qsrt_sqg_e4m3",
            "codebook": "sqg_xor_cheb_t12",
            "artifact_manifest": QSRT_MANIFEST_FILENAME,
            "profile": profile,
        },
        "dense_format": "mxfp8",
        "ignored_layers": list(IGNORED_DENSE_LAYERS),
    }


def _files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while left_chunk := left_handle.read(64 << 20):
            if left_chunk != right_handle.read(len(left_chunk)):
                return False
        return right_handle.read(1) == b""


def validate_qsrt_atoms_v2_artifact(
    root: str | Path,
    *,
    validate_candidate_payload_headers: bool = True,
    verify_payloads: bool = False,
) -> dict:
    """Validate the complete all-QSRT atoms-v2 layer inventory."""

    root = Path(root).resolve()
    manifest = _read_json(root / QSRT_MANIFEST_FILENAME)
    profile = manifest.get("profile")
    if profile not in ATOMS_V2_PROFILES:
        raise ValueError(f"unsupported QSRT atoms-v2 profile {profile!r}")
    expected = {
        "kind": "qsrt_kimi_k3_qsrt_artifact",
        "schema_version": 2,
        "codec": "QSRT",
        "complete": True,
        "storage_schema": ATOMS_V2_SCHEMA,
        "storage_format": "qsrt_atoms_v2",
        "profile": profile,
        "tensor_parallel_independent": True,
        "all_experts_qsrt": True,
        "layer_count": C.NUM_MOE_LAYERS,
        "compressed_experts": C.NUM_MOE_LAYERS * C.NUM_EXPERTS,
        "x4t_experts": 0,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"QSRT atoms-v2 manifest {name} mismatch")
    layers = manifest.get("layers")
    if not isinstance(layers, dict) or set(layers) != {
        str(layer) for layer in C.MOE_LAYERS
    }:
        raise ValueError("QSRT atoms-v2 manifest must contain exactly 92 layers")
    total_bytes = 0
    for layer in C.MOE_LAYERS:
        entry = layers[str(layer)]
        name = layer_filename(layer)
        if not isinstance(entry, dict) or entry.get("qsrt_atoms") != name:
            raise ValueError(f"QSRT atoms-v2 layer {layer} entry is malformed")
        path = root / name
        with QSRTAtomsV2Reader(path) as reader:
            if reader.header.layer != layer:
                raise ValueError(f"QSRT atoms-v2 layer {layer} identity drifted")
            if reader.header.layout.profile != profile:
                raise ValueError(f"QSRT atoms-v2 layer {layer} profile drifted")
            disk_bytes = reader.header.layout.disk_bytes
        if path.stat().st_size != disk_bytes or entry.get("atom_disk_bytes") != disk_bytes:
            raise ValueError(f"QSRT atoms-v2 layer {layer} byte count drifted")
        total_bytes += disk_bytes
    if manifest.get("container_bytes") != total_bytes:
        raise ValueError("QSRT atoms-v2 container byte count drifted")

    candidate_digest = manifest.get("candidate_pool_content_sha256")
    payloads_verified = False
    if verify_payloads:
        candidate_root = Path(str(manifest.get("candidate_pool", ""))).resolve()
        pool = load_qsrt_candidate_pool(
            candidate_root,
            validate_payload_headers=validate_candidate_payload_headers,
            require_completion=True,
            verify_completion_hashes=True,
        )
        if pool.content_sha256 != candidate_digest:
            raise ValueError("QSRT atoms-v2 candidate-pool identity drifted")
        if pool.codebook != CODEBOOK_SQG_XOR_CHEB_T12:
            raise ValueError("QSRT atoms-v2 candidate codebook drifted")

        if profile == ATOMS_V2_COUPLED_H308_PROFILE:
            mode = H308
            rotation_plan = CoupledRotationPlan.from_json(
                manifest.get("coupled_rotation_plan")
            )
        elif profile == ATOMS_V2_PURE_K2_PROFILE:
            mode = K2
            rotation_plan = CoupledRotationPlan.from_json(
                manifest.get("coupled_rotation_plan")
            )
        else:
            mode = H308
            rotation_plan = None
        if pool.mode_ids != (mode.mode_id,):
            raise ValueError("QSRT atoms-v2 candidate mode drifted")
        if rotation_plan is None:
            if pool.coupled_rotation_draws is not None:
                raise ValueError("uncoupled atoms-v2 candidate has rotation draws")
        else:
            if pool.coupled_rotation_draws is None:
                raise ValueError("coupled atoms-v2 candidate has no rotation draws")
            for layer in C.MOE_LAYERS:
                expected_draws = tuple(
                    int(draw) for draw in pool.coupled_rotation_draws[layer - 1]
                )
                if rotation_plan.for_layer(layer) != expected_draws:
                    raise ValueError(
                        f"QSRT atoms-v2 layer {layer} rotation plan drifted"
                    )

        with tempfile.TemporaryDirectory(
            prefix=".qsrt-atoms-v2-verify-", dir=root.parent
        ) as temporary:
            temporary_root = Path(temporary)
            for layer in C.MOE_LAYERS:
                rebuilt = temporary_root / layer_filename(layer)
                materialize_atoms_v2_layer(
                    pool.root,
                    rebuilt,
                    layer,
                    batch_size=64,
                    profile=profile,
                    rotation_draws=(
                        None
                        if rotation_plan is None
                        else rotation_plan.for_layer(layer)
                    ),
                )
                if not _files_equal(root / layer_filename(layer), rebuilt):
                    raise ValueError(
                        f"QSRT atoms-v2 layer {layer} payload disagrees with "
                        "its sealed candidate"
                    )
                rebuilt.unlink()
        payloads_verified = True
    return {
        "kind": "qsrt_kimi_k3_qsrt_artifact_validation",
        "schema_version": 2,
        "artifact": str(root),
        "complete": True,
        "layers": C.NUM_MOE_LAYERS,
        "compressed_experts": C.NUM_MOE_LAYERS * C.NUM_EXPERTS,
        "x4t_experts": 0,
        "container_bytes": total_bytes,
        "storage_schema": ATOMS_V2_SCHEMA,
        "profile": profile,
        "tensor_parallel_independent": True,
        "candidate_pool_content_sha256": candidate_digest,
        "candidate_payloads_verified": payloads_verified,
    }


def _artifact_runtime_contract(artifact: Path) -> tuple[dict, dict]:
    manifest = _read_json(artifact / QSRT_MANIFEST_FILENAME)
    if manifest.get("storage_schema") == ATOMS_V2_SCHEMA:
        return validate_qsrt_atoms_v2_artifact(artifact), (
            qsrt_atoms_v2_quantization_config(str(manifest.get("profile")))
        )
    validation = validate_qsrt_artifact(
        artifact,
        validate_candidate_payload_headers=True,
        verify_payloads=False,
    )
    allocation = _read_json(artifact / QSRT_ALLOCATION_COPY_FILENAME)
    return validation, qsrt_quantization_config(allocation)


def validate_qsrt_model_view(root: str | Path) -> dict:
    """Close a model view against its canonical artifact and linked sources."""

    root = Path(root).resolve()
    view = _read_json(root / QSRT_MODEL_VIEW_MANIFEST)
    expected_scalars = {
        "kind": QSRT_MODEL_VIEW_KIND,
        "schema_version": QSRT_MODEL_VIEW_SCHEMA_VERSION,
        "layers": C.NUM_MOE_LAYERS,
    }
    for name, expected in expected_scalars.items():
        if view.get(name) != expected:
            raise ValueError(f"QSRT model view {name} mismatch")

    artifact = Path(str(view.get("artifact", ""))).resolve()
    nonexpert = Path(str(view.get("nonexpert_source", ""))).resolve()
    snapshot = Path(str(view.get("official_snapshot", ""))).resolve()
    artifact_validation, quantization = _artifact_runtime_contract(artifact)
    if view.get("artifact_validation") != artifact_validation:
        raise ValueError("QSRT model view artifact verdict drifted")

    artifact_names = sorted(
        path.name for path in artifact.iterdir() if path.is_file()
    )
    if view.get("artifact_files") != artifact_names:
        raise ValueError("QSRT model view artifact inventory drifted")
    for name in artifact_names:
        _require_exact_link(root / name, artifact / name)

    config = _read_json(root / "config.json")
    text_config = config.get("text_config")
    if (
        not isinstance(text_config, dict)
        or text_config.get("quantization_config") != quantization
        or "quantization_config" in config
    ):
        raise ValueError("QSRT model-view quantization config drifted")

    weight_map, total_bytes, nonexpert_names = _scan_nonexpert_weight_map(
        nonexpert
    )
    if view.get("nonexpert_shards") != nonexpert_names:
        raise ValueError("QSRT model view non-expert inventory drifted")
    for name in nonexpert_names:
        _require_exact_link(root / name, nonexpert / name)
    expected_index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": dict(sorted(weight_map.items())),
    }
    if _read_json(root / "model.safetensors.index.json") != expected_index:
        raise ValueError("QSRT model view tensor index drifted")

    auxiliary = _auxiliary_sources(snapshot)
    auxiliary_names = [source.name for source in auxiliary]
    if view.get("auxiliary_files") != auxiliary_names:
        raise ValueError("QSRT model view auxiliary inventory drifted")
    for source in auxiliary:
        _require_exact_link(root / source.name, source)

    expected_names = {
        *artifact_names,
        *nonexpert_names,
        *auxiliary_names,
        "config.json",
        "model.safetensors.index.json",
        QSRT_MODEL_VIEW_MANIFEST,
    }
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != expected_names:
        raise ValueError(
            "QSRT model-view inventory does not close; "
            f"missing={sorted(expected_names - actual_names)[:3]}, "
            f"extra={sorted(actual_names - expected_names)[:3]}"
        )
    return {
        "kind": QSRT_MODEL_VIEW_KIND,
        "schema_version": QSRT_MODEL_VIEW_SCHEMA_VERSION,
        "model_view": str(root),
        "complete": True,
        "layers": C.NUM_MOE_LAYERS,
        "indexed_tensors": len(weight_map),
        "indexed_safetensors_bytes": total_bytes,
        "tensor_parallel_independent": True,
    }


def publish_qsrt_model_view(
    artifact: str | Path,
    destination: str | Path,
    *,
    nonexpert_source: str | Path = DEFAULT_NONEXPERT,
    official_snapshot: str | Path | None = None,
) -> dict:
    """Atomically publish a symlink-only vLLM view of one QSRT artifact."""

    artifact = Path(artifact).resolve()
    destination = Path(destination).resolve()
    nonexpert = Path(nonexpert_source).resolve()
    snapshot = (
        Path(official_snapshot).resolve()
        if official_snapshot is not None
        else Path(resolve().snapshot_dir).resolve()
    )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    artifact_validation, quantization = _artifact_runtime_contract(artifact)
    source_config = _read_json(snapshot / "config.json")
    text_config = source_config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("official model config has no text_config object")
    source_config.pop("quantization_config", None)
    text_config["quantization_config"] = quantization

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.partial")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    staging.mkdir()
    try:
        artifact_names = sorted(
            path.name for path in artifact.iterdir() if path.is_file()
        )
        for name in artifact_names:
            _link(artifact / name, staging / name)
        weight_map, total_bytes, nonexpert_names = _nonexpert_weight_map(
            nonexpert,
            staging,
        )
        auxiliary = _auxiliary_sources(snapshot)
        for source in auxiliary:
            _link(source, staging / source.name)
        _atomic_json(staging / "config.json", source_config)
        _atomic_json(
            staging / "model.safetensors.index.json",
            {
                "metadata": {"total_size": total_bytes},
                "weight_map": dict(sorted(weight_map.items())),
            },
        )
        _atomic_json(
            staging / QSRT_MODEL_VIEW_MANIFEST,
            {
                "kind": QSRT_MODEL_VIEW_KIND,
                "schema_version": QSRT_MODEL_VIEW_SCHEMA_VERSION,
                "artifact": str(artifact),
                "artifact_validation": artifact_validation,
                "artifact_files": artifact_names,
                "nonexpert_source": str(nonexpert),
                "nonexpert_shards": nonexpert_names,
                "official_snapshot": str(snapshot),
                "auxiliary_files": [source.name for source in auxiliary],
                "layers": C.NUM_MOE_LAYERS,
            },
        )
        _fsync_directory(staging)
        staging.replace(destination)
        _fsync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_qsrt_model_view(destination)


__all__ = [
    "QSRT_MODEL_VIEW_KIND",
    "QSRT_MODEL_VIEW_MANIFEST",
    "QSRT_MODEL_VIEW_SCHEMA_VERSION",
    "publish_qsrt_model_view",
    "qsrt_atoms_v2_quantization_config",
    "qsrt_hybrid_bit_map",
    "qsrt_quantization_config",
    "validate_qsrt_atoms_v2_artifact",
    "validate_qsrt_model_view",
]

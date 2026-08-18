"""Validation and raw-keep diagnostics for QSRT candidate pools.

The candidate encoder scores the complete gated expert function against the
decoded official MXFP4 source.  Those scores already follow natural captured
routes and already include the applied router gate squared.  Consequently the
stored excess SSE is directly the benefit of retaining that assignment in
MXFP4; applying traffic weights again would double-count routing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from qsrt import constants as C
from qsrt.exl3_reference import CODEBOOK_SQG_XOR_CHEB_T12, QSRT_CODEBOOKS
from qsrt.io.hf_cache import read_safetensors_header
from qsrt.qsrt import (
    EXPERTS_PER_LAYER,
    EXPERT_TRELLIS_BYTES,
    FIXED_HIGH_RATE_TRELLIS_BYTES,
    H308,
    INTERMEDIATE_CHANNELS,
    K2,
    LATENT_CHANNELS,
    LOGICAL_CANDIDATE_SCHEMAS,
    MATRIX_TRELLIS_BYTES,
    PURE_K2_MATRIX_TRELLIS_BYTES,
    R0,
    SCHEMA as QSRT_SCHEMA,
    ExpertFormatSpec,
)
from qsrt.qsrt_storage import ATOM_SCALE_BYTES, ATOMS_PER_EXPERT, QSRTLayerLayout
from qsrt.qsrt_coupled_plan import (
    CoupledDrawSelection,
    CoupledRotationPlan,
    POOLED_ALL_ROW_SELECTION,
    PRODUCTION_SELECTION,
    select_coupled_draw,
)
from qsrt.pooled_selection import PooledCandidateScore, select_pooled_candidate
from qsrt.pack.qsrt_candidates import (
    CANDIDATE_POOL_KIND,
    CANDIDATE_POOL_SCHEMA_VERSION,
    OFFICIAL_SOURCE_DAMAGE_METRIC,
)


RAW_KEEP_ALLOCATION_KIND = "qsrt_kimi_k3_qsrt_raw_keep_diagnostic"
ALLOCATION_SCHEMA_VERSION = 3
RAW_KEEP_PROMOTION_BYTES = (
    3 * (INTERMEDIATE_CHANNELS * LATENT_CHANNELS // 2)
    + 3 * (INTERMEDIATE_CHANNELS * LATENT_CHANNELS // 32)
    - (EXPERT_TRELLIS_BYTES + ATOMS_PER_EXPERT * ATOM_SCALE_BYTES)
)
CERTIFIED_ALLOCATION_OPTIMALITY = "globally_optimal_equal_promotion_cost"
UNCERTIFIED_ALLOCATION_OPTIMALITY = "uncertified_greedy_alignment_repair"
CANDIDATE_POOL_COMPLETION_KIND = "qsrt_kimi_k3_qsrt_candidate_completion"
CANDIDATE_POOL_COMPLETION_SCHEMA_VERSION = 2
CANDIDATE_POOL_COMPLETION_FILENAME = "qsrt-candidate-completion.json"
POOLED_FIXED_PROFILE_METRICS_KIND = "qsrt_pooled_routed_fixed_profile_metrics"
POOLED_FIXED_PROFILE_METRICS_SCHEMA_VERSION = 1


def _is_fixed_high_rate(mode_ids: tuple[int, ...]) -> bool:
    return mode_ids == (H308.mode_id,)


def _fixed_mode(mode_ids: tuple[int, ...]):
    if mode_ids == (R0.mode_id,):
        return R0
    if mode_ids == (H308.mode_id,):
        return H308
    if mode_ids == (K2.mode_id,):
        return K2
    return None


def _format_label(r13: int, r2: int) -> str:
    if r13 == r2 and r13 in (H308.mode_id, K2.mode_id):
        return H308.name if r13 == H308.mode_id else K2.name
    return f"R{r13}/R{r2}"


def _matrix_trellis_bytes(mode_ids: tuple[int, ...]) -> int:
    fixed_mode = _fixed_mode(mode_ids)
    if fixed_mode == H308:
        return FIXED_HIGH_RATE_TRELLIS_BYTES
    if fixed_mode == K2:
        return PURE_K2_MATRIX_TRELLIS_BYTES
    return MATRIX_TRELLIS_BYTES


def _selection_contract(mode_ids: tuple[int, ...]) -> dict[str, object]:
    fixed_mode = _fixed_mode(mode_ids)
    if fixed_mode is not None:
        return {
            "format_grid": f"fixed_{fixed_mode.name.lower()}",
            "shared_r": True,
            "candidate_construction_fold": "fit",
            "mode_selection_fold": "none",
            "mode_proposal_metric": "none",
            "mode_acceptance": (
                "fixed_22k3_2k4_high_rate_profile"
                if fixed_mode == H308
                else (
                    "fixed_uniform_k2_profile"
                    if fixed_mode == K2
                    else "fixed_uniform_k3_profile"
                )
            ),
            "external_validation_used": False,
        }
    return {
        "format_grid": "cartesian_r13_r2",
        "shared_r": False,
        "candidate_construction_fold": "fit",
        "mode_selection_fold": "confirmation",
        "mode_proposal_metric": "confirmation_routed_functional_sse",
        "mode_acceptance": "paired_document_bootstrap_lower_bound_vs_r0",
        "external_validation_used": False,
    }


def pooled_fixed_profile_selection_contract(
    mode_ids: tuple[int, ...],
) -> dict[str, object]:
    """Describe fixed-profile selection over one pooled routed population."""

    fixed_mode = _fixed_mode(mode_ids)
    if fixed_mode is None:
        raise ValueError("pooled fixed-profile metrics require one fixed mode")
    return {
        "format_grid": f"fixed_{fixed_mode.name.lower()}",
        "shared_r": True,
        "candidate_construction_population": "all_captured_natural_routes",
        "candidate_selection_population": "all_captured_natural_routes",
        "candidate_selection_metric": OFFICIAL_SOURCE_DAMAGE_METRIC,
        "candidate_selection_rule": "minimum_serialized_pooled_routed_sse",
        "route_weighting": "applied_gate_squared_once",
        "external_validation_used": False,
        "metric_contract": {
            "kind": POOLED_FIXED_PROFILE_METRICS_KIND,
            "schema_version": POOLED_FIXED_PROFILE_METRICS_SCHEMA_VERSION,
        },
    }


@dataclass(frozen=True)
class QSRTCandidatePool:
    root: Path
    manifest: dict
    damage: np.ndarray
    selected_r13: np.ndarray
    selected_r2: np.ndarray
    proposed_r13: np.ndarray
    proposed_r2: np.ndarray
    evaluated_format_counts: np.ndarray
    format_histogram: dict[str, int]
    damage_metric: str = OFFICIAL_SOURCE_DAMAGE_METRIC
    damage_weighting: str = (
        "natural-routing token-uniform selection-corpus rows with applied gate squared"
    )
    damage_provenance: dict | None = None
    completion: dict | None = None
    content_sha256: str | None = None
    trellis_schema: str | None = None
    mode_ids: tuple[int, ...] = (0, 1, 2)
    codebook: str = CODEBOOK_SQG_XOR_CHEB_T12
    coupled_rotation_draws: np.ndarray | None = None


@dataclass(frozen=True)
class QSRTRawKeepAllocation:
    """Legacy-cost diagnostic used to compare calibration damage rankings.

    The production QSRT allocator is :class:`QSRTAllocation` in
    :mod:`qsrt.pack.qsrt_allocation`; it uses measured X4T bytes. This class
    intentionally models a fixed-size raw-MXFP4 keep tier so old and new
    ranking fields can be compared at the same retained-expert count.
    """
    keep_mask: np.ndarray
    layer_orders: tuple[np.ndarray, ...]
    target_container_bytes: int | None
    container_bytes: int
    raw_target_keep_count: int
    retained_experts: int
    alignment_repair_swaps: int
    alignment_repair_damage_cost: float


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "file": path.name,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _candidate_content_sha256(
    manifest_sha256: str,
    layers: Mapping[str, object],
    logical_trellis_schema: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"qsrt-qsrt-candidate-pool-v2\0")
    digest.update(bytes.fromhex(manifest_sha256))
    schema_bytes = logical_trellis_schema.encode("utf-8")
    digest.update(len(schema_bytes).to_bytes(4, "little"))
    digest.update(schema_bytes)
    for layer in C.MOE_LAYERS:
        entry = layers.get(str(layer))
        if not isinstance(entry, dict):
            raise ValueError(f"candidate completion is missing layer {layer}")
        for role in ("payload", "metrics", "selection"):
            component = entry.get(role)
            if not isinstance(component, dict):
                raise ValueError(
                    f"candidate completion layer {layer} is missing {role}"
                )
            relative = f"candidates/{component.get('file', '')}".encode("utf-8")
            sha256 = component.get("sha256")
            if not isinstance(sha256, str) or len(sha256) != 64:
                raise ValueError(
                    f"candidate completion layer {layer} {role} has no SHA-256"
                )
            try:
                raw_sha256 = bytes.fromhex(sha256)
            except ValueError as exc:
                raise ValueError(
                    f"candidate completion layer {layer} {role} SHA-256 is invalid"
                ) from exc
            digest.update(len(relative).to_bytes(4, "little"))
            digest.update(relative)
            digest.update(raw_sha256)
    return digest.hexdigest()


def _canonical_layer_paths(root: Path, layer: int) -> tuple[Path, Path, Path]:
    stem = f"qsrt-layer-{layer:05d}"
    candidate_dir = root / "candidates"
    return (
        candidate_dir / f"{stem}.safetensors",
        candidate_dir / f"{stem}.metrics.safetensors",
        candidate_dir / f"{stem}.selection.json",
    )


def _indexed_candidate_file(path: Path) -> dict[str, int | str]:
    before = _file_identity(path)
    sha256 = _sha256(path)
    after = _file_identity(path)
    if before != after:
        raise ValueError(f"candidate file changed while hashing: {path}")
    return {**after, "sha256": sha256}


def candidate_pool_completion_document(
    root: str | Path,
    *,
    jobs: int = 1,
    logical_trellis_schema: str = QSRT_SCHEMA,
) -> dict:
    """Hash one complete atomic candidate pool into an immutable index."""

    root = Path(root).resolve()
    if isinstance(jobs, bool) or jobs < 1:
        raise ValueError("candidate hash jobs must be a positive integer")
    if logical_trellis_schema not in LOGICAL_CANDIDATE_SCHEMAS:
        raise ValueError("candidate logical trellis schema is unsupported")
    manifest_path = root / "qsrt-candidate-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    paths: list[Path] = []
    for layer in C.MOE_LAYERS:
        layer_paths = _canonical_layer_paths(root, layer)
        for path in layer_paths:
            if not path.is_file() or path.stat().st_size <= 0:
                raise ValueError(f"candidate pool is incomplete; missing {path}")
        paths.extend(layer_paths)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        indexed = list(executor.map(_indexed_candidate_file, paths))
    layers: dict[str, dict[str, dict[str, int | str]]] = {}
    cursor = 0
    for layer in C.MOE_LAYERS:
        layers[str(layer)] = {
            role: indexed[cursor + offset]
            for offset, role in enumerate(("payload", "metrics", "selection"))
        }
        cursor += 3
    manifest = _read_json(manifest_path)
    codebook = manifest.get("codebook")
    if codebook not in QSRT_CODEBOOKS:
        raise ValueError("candidate manifest codebook is unsupported")
    manifest_sha256 = _sha256(manifest_path)
    content_sha256 = _candidate_content_sha256(
        manifest_sha256,
        layers,
        logical_trellis_schema,
    )
    return {
        "kind": CANDIDATE_POOL_COMPLETION_KIND,
        "schema_version": CANDIDATE_POOL_COMPLETION_SCHEMA_VERSION,
        "complete": True,
        "candidate_schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
        "manifest": manifest_path.name,
        "manifest_sha256": manifest_sha256,
        "logical_trellis_schema": logical_trellis_schema,
        "codebook": codebook,
        "content_sha256": content_sha256,
        "layer_count": C.NUM_MOE_LAYERS,
        "layers": layers,
    }


def validate_candidate_pool_completion(
    root: str | Path,
    *,
    verify_hashes: bool = False,
) -> dict:
    """Validate the content index and optionally rehash all candidate files."""

    root = Path(root).resolve()
    completion_path = root / CANDIDATE_POOL_COMPLETION_FILENAME
    completion = _read_json(completion_path)
    expected_scalars = {
        "kind": CANDIDATE_POOL_COMPLETION_KIND,
        "schema_version": CANDIDATE_POOL_COMPLETION_SCHEMA_VERSION,
        "complete": True,
        "candidate_schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
        "manifest": "qsrt-candidate-manifest.json",
        "layer_count": C.NUM_MOE_LAYERS,
    }
    for name, expected in expected_scalars.items():
        if completion.get(name) != expected:
            raise ValueError(
                f"candidate completion {name} mismatch: "
                f"{completion.get(name)!r} != {expected!r}"
            )
    logical_trellis_schema = completion.get("logical_trellis_schema")
    if logical_trellis_schema not in LOGICAL_CANDIDATE_SCHEMAS:
        raise ValueError("candidate completion logical trellis schema is unsupported")
    manifest_path = root / str(completion["manifest"])
    if completion.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("candidate completion manifest hash drifted")
    manifest = _read_json(manifest_path)
    manifest_codebook = manifest.get("codebook")
    completion_codebook = completion.get("codebook")
    if (
        manifest_codebook not in QSRT_CODEBOOKS
        or completion_codebook != manifest_codebook
    ):
        raise ValueError("candidate completion codebook disagrees with its manifest")
    layers = completion.get("layers")
    if not isinstance(layers, dict) or set(layers) != {
        str(layer) for layer in C.MOE_LAYERS
    }:
        raise ValueError("candidate completion must contain exactly 92 layers")
    for layer in C.MOE_LAYERS:
        expected_paths = _canonical_layer_paths(root, layer)
        entry = layers[str(layer)]
        if not isinstance(entry, dict) or set(entry) != {
            "payload",
            "metrics",
            "selection",
        }:
            raise ValueError(
                f"candidate completion layer {layer} inventory drifted"
            )
        for role, path in zip(
            ("payload", "metrics", "selection"),
            expected_paths,
            strict=True,
        ):
            indexed = entry[role]
            if not isinstance(indexed, dict):
                raise ValueError(
                    f"candidate completion layer {layer} {role} is malformed"
                )
            sha256 = indexed.get("sha256")
            identity = {key: value for key, value in indexed.items() if key != "sha256"}
            if identity != _file_identity(path):
                raise ValueError(
                    f"candidate completion layer {layer} {role} identity drifted"
                )
            if verify_hashes and sha256 != _sha256(path):
                raise ValueError(
                    f"candidate completion layer {layer} {role} hash drifted"
                )
    expected_content = _candidate_content_sha256(
        str(completion["manifest_sha256"]),
        layers,
        str(logical_trellis_schema),
    )
    if completion.get("content_sha256") != expected_content:
        raise ValueError("candidate completion content digest does not close")
    return completion


def write_candidate_pool_completion(
    root: str | Path,
    document: dict,
) -> Path:
    """Atomically publish a freshly computed candidate completion index."""

    root = Path(root).resolve()
    path = root / CANDIDATE_POOL_COMPLETION_FILENAME
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(document, handle, indent=1, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _require_tensor(
    metrics: Mapping[str, torch.Tensor],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    try:
        value = metrics[name]
    except KeyError as exc:
        raise ValueError(f"candidate metrics are missing {name}") from exc
    if tuple(value.shape) != shape:
        raise ValueError(
            f"candidate metric {name} has shape {tuple(value.shape)}, expected {shape}"
        )
    if value.dtype != dtype:
        raise ValueError(
            f"candidate metric {name} has dtype {value.dtype}, expected {dtype}"
        )
    return value


def validate_layer_metrics(
    metrics: Mapping[str, torch.Tensor],
    *,
    mode_ids: tuple[int, ...],
    fit_documents: int,
    confirmation_documents: int,
    expert_ids: tuple[int, ...] = tuple(range(EXPERTS_PER_LAYER)),
    min_fit_documents: int | None = None,
    min_confirmation_documents: int | None = None,
    minimum_improvement: float | None = None,
) -> dict[str, torch.Tensor]:
    """Validate one layer ledger and recompute its selected functional damage."""

    experts = len(expert_ids)
    modes = len(mode_ids)
    actual_experts = _require_tensor(
        metrics, "expert_ids", shape=(experts,), dtype=torch.int16
    )
    if actual_experts.tolist() != list(expert_ids):
        raise ValueError("candidate expert IDs are not in canonical order")
    actual_modes = _require_tensor(
        metrics, "mode_ids", shape=(modes,), dtype=torch.uint8
    )
    if actual_modes.tolist() != list(mode_ids):
        raise ValueError("candidate mode IDs do not match the pool manifest")

    selected_r13 = _require_tensor(
        metrics, "selected_r13", shape=(experts,), dtype=torch.uint8
    )
    selected_r2 = _require_tensor(
        metrics, "selected_r2", shape=(experts,), dtype=torch.uint8
    )
    proposed_r13 = _require_tensor(
        metrics, "proposed_r13", shape=(experts,), dtype=torch.uint8
    )
    proposed_r2 = _require_tensor(
        metrics, "proposed_r2", shape=(experts,), dtype=torch.uint8
    )
    evaluated = _require_tensor(
        metrics,
        "format_evaluated",
        shape=(experts, modes, modes),
        dtype=torch.bool,
    )
    fit_sse = _require_tensor(
        metrics,
        "fit_sse",
        shape=(experts, modes, modes, fit_documents),
        dtype=torch.float64,
    )
    confirmation_sse = _require_tensor(
        metrics,
        "confirmation_sse",
        shape=(experts, modes, modes, confirmation_documents),
        dtype=torch.float64,
    )
    fit_reference = _require_tensor(
        metrics,
        "fit_reference_energy",
        shape=(experts, fit_documents),
        dtype=torch.float64,
    )
    confirmation_reference = _require_tensor(
        metrics,
        "confirmation_reference_energy",
        shape=(experts, confirmation_documents),
        dtype=torch.float64,
    )
    fit_counts = _require_tensor(
        metrics,
        "fit_counts",
        shape=(experts, fit_documents),
        dtype=torch.int32,
    )
    confirmation_counts = _require_tensor(
        metrics,
        "confirmation_counts",
        shape=(experts, confirmation_documents),
        dtype=torch.int32,
    )
    stored_damage = _require_tensor(
        metrics,
        OFFICIAL_SOURCE_DAMAGE_METRIC,
        shape=(experts,),
        dtype=torch.float64,
    )

    mode_column = {mode: column for column, mode in enumerate(mode_ids)}
    try:
        selected_r13_columns = torch.tensor(
            [mode_column[int(mode)] for mode in selected_r13.tolist()],
            dtype=torch.long,
        )
        selected_r2_columns = torch.tensor(
            [mode_column[int(mode)] for mode in selected_r2.tolist()],
            dtype=torch.long,
        )
        _ = [mode_column[int(mode)] for mode in proposed_r13.tolist()]
        _ = [mode_column[int(mode)] for mode in proposed_r2.tolist()]
    except KeyError as exc:
        raise ValueError(
            "selected/proposed rate is absent from the mode table"
        ) from exc
    rows = torch.arange(experts)
    if not bool(
        torch.all(evaluated[rows, selected_r13_columns, selected_r2_columns])
    ):
        raise ValueError("a selected format was not evaluated")

    for name, values in (
        ("fit_sse", fit_sse[evaluated]),
        ("confirmation_sse", confirmation_sse[evaluated]),
        ("fit_reference_energy", fit_reference),
        ("confirmation_reference_energy", confirmation_reference),
        (OFFICIAL_SOURCE_DAMAGE_METRIC, stored_damage),
    ):
        if not bool(torch.all(torch.isfinite(values))) or bool(torch.any(values < 0)):
            raise ValueError(f"candidate metric {name} must be finite and non-negative")
    if bool(torch.any(fit_counts < 0)) or bool(torch.any(confirmation_counts < 0)):
        raise ValueError("candidate document counts must be non-negative")

    selection_parameters = (
        min_fit_documents,
        min_confirmation_documents,
        minimum_improvement,
    )
    if any(value is not None for value in selection_parameters):
        if any(value is None for value in selection_parameters):
            raise ValueError(
                "selection validation parameters must be provided together"
            )
        fixed_mode = _fixed_mode(mode_ids)
        if fixed_mode is not None:
            if not bool(torch.all(evaluated)):
                raise ValueError(
                    f"fixed {fixed_mode.name} profile must be evaluated for every expert"
                )
            expected = torch.full(
                (experts,), fixed_mode.mode_id, dtype=torch.uint8
            )
            if not (
                torch.equal(proposed_r13, expected)
                and torch.equal(proposed_r2, expected)
                and torch.equal(selected_r13, expected)
                and torch.equal(selected_r2, expected)
            ):
                raise ValueError(
                    f"fixed {fixed_mode.name} profile was not selected exactly"
                )
            improvement = _require_tensor(
                metrics,
                "confirmation_improvement",
                shape=(experts,),
                dtype=torch.float64,
            )
            ci95 = _require_tensor(
                metrics,
                "confirmation_ci95",
                shape=(experts, 2),
                dtype=torch.float64,
            )
            if not bool(torch.all(torch.isnan(improvement))) or not bool(
                torch.all(torch.isnan(ci95))
            ):
                raise ValueError(
                    f"fixed {fixed_mode.name} profile must not carry selection evidence"
                )
        else:
            _validate_selection_semantics(
                metrics,
                mode_ids=mode_ids,
                evaluated=evaluated,
                fit_sse=fit_sse,
                confirmation_sse=confirmation_sse,
                fit_counts=fit_counts,
                confirmation_counts=confirmation_counts,
                proposed_r13=proposed_r13,
                proposed_r2=proposed_r2,
                selected_r13=selected_r13,
                selected_r2=selected_r2,
                min_fit_documents=int(min_fit_documents),
                min_confirmation_documents=int(min_confirmation_documents),
                minimum_improvement=float(minimum_improvement),
            )

    recomputed = (
        fit_sse[rows, selected_r13_columns, selected_r2_columns].sum(dim=1)
        + confirmation_sse[
            rows, selected_r13_columns, selected_r2_columns
        ].sum(dim=1)
    )
    if not torch.equal(recomputed, stored_damage):
        raise ValueError(
            f"{OFFICIAL_SOURCE_DAMAGE_METRIC} does not equal selected "
            "fit+confirmation SSE"
        )
    return {
        "damage": stored_damage,
        "selected_r13": selected_r13,
        "selected_r2": selected_r2,
        "proposed_r13": proposed_r13,
        "proposed_r2": proposed_r2,
        "evaluated_format_count": evaluated.sum(dim=(1, 2)),
    }


def validate_pooled_fixed_profile_metrics(
    metrics: Mapping[str, torch.Tensor],
    *,
    mode_ids: tuple[int, ...],
    expert_ids: tuple[int, ...] = tuple(range(EXPERTS_PER_LAYER)),
    expected_coupled_draws: tuple[int, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Validate all-row routed metrics for one immutable fixed-rate profile."""

    fixed_mode = _fixed_mode(mode_ids)
    if fixed_mode is None:
        raise ValueError("pooled fixed-profile metrics require one fixed mode")
    expected_names = {
        "expert_ids",
        "mode_ids",
        "selected_r13",
        "selected_r2",
        "proposed_r13",
        "proposed_r2",
        "format_evaluated",
        "pooled_baseline_sse",
        "pooled_replacement_sse",
        "pooled_selected_sse",
        "pooled_source_energy",
        OFFICIAL_SOURCE_DAMAGE_METRIC,
        "routed_occurrences",
        "effective_sample_size",
        "selected_functional_refit",
        "coupled_draw_selected",
    }
    if set(metrics) != expected_names:
        raise ValueError(
            "pooled candidate metric inventory drifted: "
            f"missing={sorted(expected_names - set(metrics))}, "
            f"extra={sorted(set(metrics) - expected_names)}"
        )
    experts = len(expert_ids)
    actual_experts = _require_tensor(
        metrics, "expert_ids", shape=(experts,), dtype=torch.int16
    )
    if actual_experts.tolist() != list(expert_ids):
        raise ValueError("pooled candidate expert IDs are not in canonical order")
    actual_modes = _require_tensor(
        metrics, "mode_ids", shape=(1,), dtype=torch.uint8
    )
    if actual_modes.tolist() != [fixed_mode.mode_id]:
        raise ValueError("pooled candidate mode ID does not match the manifest")

    expected_mode = torch.full((experts,), fixed_mode.mode_id, dtype=torch.uint8)
    rates: dict[str, torch.Tensor] = {}
    for name in ("selected_r13", "selected_r2", "proposed_r13", "proposed_r2"):
        value = _require_tensor(metrics, name, shape=(experts,), dtype=torch.uint8)
        if not torch.equal(value, expected_mode):
            raise ValueError(f"pooled fixed profile has invalid {name}")
        rates[name] = value
    evaluated = _require_tensor(
        metrics, "format_evaluated", shape=(experts, 1, 1), dtype=torch.bool
    )
    if not bool(torch.all(evaluated)):
        raise ValueError("pooled fixed profile was not evaluated for every expert")

    baseline = _require_tensor(
        metrics, "pooled_baseline_sse", shape=(experts,), dtype=torch.float64
    )
    replacement = _require_tensor(
        metrics, "pooled_replacement_sse", shape=(experts,), dtype=torch.float64
    )
    selected = _require_tensor(
        metrics, "pooled_selected_sse", shape=(experts,), dtype=torch.float64
    )
    source_energy = _require_tensor(
        metrics, "pooled_source_energy", shape=(experts,), dtype=torch.float64
    )
    damage = _require_tensor(
        metrics, OFFICIAL_SOURCE_DAMAGE_METRIC, shape=(experts,), dtype=torch.float64
    )
    occurrences = _require_tensor(
        metrics, "routed_occurrences", shape=(experts,), dtype=torch.int64
    )
    effective_samples = _require_tensor(
        metrics, "effective_sample_size", shape=(experts,), dtype=torch.float64
    )
    selected_refit = _require_tensor(
        metrics, "selected_functional_refit", shape=(experts,), dtype=torch.bool
    )
    coupled_draws = _require_tensor(
        metrics, "coupled_draw_selected", shape=(experts,), dtype=torch.uint8
    )

    for name, values in (
        ("pooled_baseline_sse", baseline),
        ("pooled_replacement_sse", replacement),
        ("pooled_selected_sse", selected),
        ("pooled_source_energy", source_energy),
        (OFFICIAL_SOURCE_DAMAGE_METRIC, damage),
        ("effective_sample_size", effective_samples),
    ):
        if not bool(torch.all(torch.isfinite(values))) or bool(torch.any(values < 0)):
            raise ValueError(f"pooled candidate metric {name} must be finite and non-negative")
    if bool(torch.any(occurrences <= 0)) or bool(torch.any(effective_samples <= 0)):
        raise ValueError("pooled candidate support must be positive")
    if bool(torch.any(effective_samples > occurrences.to(torch.float64))):
        raise ValueError("pooled effective sample size exceeds routed occurrences")
    if bool(torch.any(source_energy <= 0)):
        raise ValueError("pooled source energy must be positive")

    expected_refit = replacement < baseline
    expected_selected = torch.minimum(baseline, replacement)
    if not torch.equal(selected_refit, expected_refit):
        raise ValueError("pooled refit selection disagrees with exact SSE")
    if not torch.equal(selected, expected_selected):
        raise ValueError("pooled selected SSE is not the exact minimum")
    if not torch.equal(damage, selected):
        raise ValueError(
            f"{OFFICIAL_SOURCE_DAMAGE_METRIC} does not equal pooled selected SSE"
        )
    if expected_coupled_draws is not None:
        if len(expected_coupled_draws) != experts or coupled_draws.tolist() != list(
            expected_coupled_draws
        ):
            raise ValueError("pooled coupled draws disagree with the model plan")

    return {
        "damage": damage,
        **rates,
        "evaluated_format_count": evaluated.sum(dim=(1, 2)),
        "coupled_draw_selected": coupled_draws,
    }


def validate_coupled_rotation_metrics(
    metrics: Mapping[str, torch.Tensor],
    contract: Mapping[str, object] | None,
    *,
    layer: int,
    expert_ids: tuple[int, ...],
    min_fit_documents: int,
    min_confirmation_documents: int,
    minimum_improvement: float,
    ledger: Mapping[str, object] | None = None,
) -> dict[str, torch.Tensor] | None:
    """Validate the selected coupled-Hadamard draw and its fold evidence."""

    metric_names = {
        "coupled_draw_evaluated",
        "coupled_draw_fit_sse",
        "coupled_draw_confirmation_sse",
        "coupled_draw_proposed",
        "coupled_draw_selected",
        "coupled_draw_confirmation_improvement",
    }
    pooled_metric_names = {
        "coupled_draw_pooled_sse",
        "coupled_draw_pooled_source_energy",
        "coupled_draw_pooled_occurrences",
    }
    present = metric_names.intersection(metrics)
    pooled_present = pooled_metric_names.intersection(metrics)
    if contract is None:
        if present or pooled_present:
            raise ValueError("non-coupled candidate pool carries coupled draw metrics")
        return None
    if present != metric_names:
        raise ValueError("coupled candidate metrics are incomplete")
    if not isinstance(contract, Mapping):
        raise ValueError("coupled rotation contract must be an object")
    pooled_selection = contract.get("selection") == POOLED_ALL_ROW_SELECTION
    if pooled_selection and pooled_present != pooled_metric_names:
        raise ValueError("pooled coupled candidate metrics are incomplete")
    if not pooled_selection and pooled_present:
        raise ValueError("non-pooled coupled candidate carries pooled metrics")

    experts = len(expert_ids)
    evaluated = _require_tensor(
        metrics, "coupled_draw_evaluated", shape=(experts, 8), dtype=torch.bool
    )
    fit_sse = _require_tensor(
        metrics, "coupled_draw_fit_sse", shape=(experts, 8), dtype=torch.float64
    )
    confirmation_sse = _require_tensor(
        metrics,
        "coupled_draw_confirmation_sse",
        shape=(experts, 8),
        dtype=torch.float64,
    )
    proposed = _require_tensor(
        metrics, "coupled_draw_proposed", shape=(experts,), dtype=torch.uint8
    )
    selected = _require_tensor(
        metrics, "coupled_draw_selected", shape=(experts,), dtype=torch.uint8
    )
    improvement = _require_tensor(
        metrics,
        "coupled_draw_confirmation_improvement",
        shape=(experts,),
        dtype=torch.float64,
    )
    pooled_sse = (
        _require_tensor(
            metrics,
            "coupled_draw_pooled_sse",
            shape=(experts, 8),
            dtype=torch.float64,
        )
        if pooled_selection
        else None
    )
    pooled_source_energy = (
        _require_tensor(
            metrics,
            "coupled_draw_pooled_source_energy",
            shape=(experts,),
            dtype=torch.float64,
        )
        if pooled_selection
        else None
    )
    pooled_occurrences = (
        _require_tensor(
            metrics,
            "coupled_draw_pooled_occurrences",
            shape=(experts,),
            dtype=torch.int64,
        )
        if pooled_selection
        else None
    )
    fit_counts = metrics["fit_counts"]
    confirmation_counts = metrics["confirmation_counts"]
    format_fit_sse = metrics["fit_sse"]
    format_confirmation_sse = metrics["confirmation_sse"]

    source = contract.get("source")
    if source == "model_rotation_plan":
        plan = CoupledRotationPlan.from_json(contract.get("plan"))
        layer_draws = plan.for_layer(layer)
        expected_draws = tuple((layer_draws[expert],) for expert in expert_ids)
        dynamic_draws: tuple[int, ...] | None = None
    elif source == "candidate_pool_selection":
        if contract.get("selection") not in (
            PRODUCTION_SELECTION,
            POOLED_ALL_ROW_SELECTION,
        ):
            raise ValueError("coupled draw selection policy is unsupported")
        raw_draws = contract.get("draw_candidates")
        if not isinstance(raw_draws, list) or any(
            isinstance(draw, bool) or not isinstance(draw, int) for draw in raw_draws
        ):
            raise ValueError("coupled draw candidates are malformed")
        dynamic_draws = tuple(raw_draws)
        if (
            not dynamic_draws
            or dynamic_draws[0] != 0
            or len(set(dynamic_draws)) != len(dynamic_draws)
            or any(not 0 <= draw < 8 for draw in dynamic_draws)
        ):
            raise ValueError("coupled draw candidates must be unique and begin at zero")
        expected_draws = (dynamic_draws,) * experts
    else:
        raise ValueError("coupled rotation source is unsupported")

    selections = None if ledger is None else ledger.get("selections")
    if ledger is not None and not isinstance(selections, Mapping):
        raise ValueError("coupled selection ledger has no expert evidence")

    for row, expert in enumerate(expert_ids):
        draws = expected_draws[row]
        expected_mask = torch.zeros(8, dtype=torch.bool)
        expected_mask[list(draws)] = True
        if not torch.equal(evaluated[row], expected_mask):
            raise ValueError(f"expert {expert} coupled draw evaluation set drifted")
        for values, name in (
            (fit_sse[row], "fit"),
            (confirmation_sse[row], "confirmation"),
        ):
            if not bool(torch.all(torch.isfinite(values[expected_mask]))) or bool(
                torch.any(values[expected_mask] < 0)
            ):
                raise ValueError(f"expert {expert} coupled {name} SSE is invalid")
            if not bool(torch.all(torch.isnan(values[~expected_mask]))):
                raise ValueError(
                    f"expert {expert} unevaluated coupled {name} SSE must be NaN"
                )
        fit_documents = int(torch.count_nonzero(fit_counts[row]))
        confirmation_documents = int(
            torch.count_nonzero(confirmation_counts[row])
        )
        fit_map = {draw: float(fit_sse[row, draw]) for draw in draws}
        confirmation_map = {
            draw: float(confirmation_sse[row, draw]) for draw in draws
        }
        pooled_map: dict[int, float] | None = None
        if dynamic_draws is not None and pooled_selection:
            assert (
                pooled_sse is not None
                and pooled_source_energy is not None
                and pooled_occurrences is not None
            )
            if (
                not bool(torch.all(torch.isfinite(pooled_sse[row, expected_mask])))
                or bool(torch.any(pooled_sse[row, expected_mask] < 0))
                or not bool(torch.all(torch.isnan(pooled_sse[row, ~expected_mask])))
                or not math.isfinite(float(pooled_source_energy[row]))
                or float(pooled_source_energy[row]) <= 0
                or int(pooled_occurrences[row]) <= 0
            ):
                raise ValueError(f"expert {expert} pooled coupled score is invalid")
            pooled_map = {
                draw: float(pooled_sse[row, draw]) for draw in dynamic_draws
            }
            pooled_decision = select_pooled_candidate(
                PooledCandidateScore(
                    name=str(draw),
                    sse=pooled_map[draw],
                    source_energy=float(pooled_source_energy[row]),
                    metadata_bytes=0,
                )
                for draw in dynamic_draws
            )
            pooled_draw = int(pooled_decision.selected.name)
            decision = CoupledDrawSelection(
                evaluated_draws=dynamic_draws,
                proposed_draw=pooled_draw,
                selected_draw=pooled_draw,
                fit_documents=fit_documents,
                confirmation_documents=confirmation_documents,
                confirmation_relative_improvement=None,
                accepted=pooled_draw != 0,
                reason="pooled_all_routed_rows_minimum_sse",
            )
            expected_improvement = None
        elif dynamic_draws is not None:
            decision = select_coupled_draw(
                dynamic_draws,
                fit_map,
                confirmation_map,
                fit_documents=fit_documents,
                confirmation_documents=confirmation_documents,
                min_fit_documents=min_fit_documents,
                min_confirmation_documents=min_confirmation_documents,
                minimum_improvement=minimum_improvement,
            )
            expected_improvement = decision.confirmation_relative_improvement
        else:
            draw = draws[0]
            decision = CoupledDrawSelection(
                evaluated_draws=draws,
                proposed_draw=draw,
                selected_draw=draw,
                fit_documents=fit_documents,
                confirmation_documents=confirmation_documents,
                confirmation_relative_improvement=None,
                accepted=False,
                reason="fixed_model_rotation_plan",
            )
            expected_improvement = None
        if int(proposed[row]) != decision.proposed_draw or int(selected[row]) != decision.selected_draw:
            raise ValueError(f"expert {expert} coupled draw decision drifted")
        if expected_improvement is None:
            if not math.isnan(float(improvement[row])):
                raise ValueError(f"expert {expert} coupled improvement must be NaN")
        elif not math.isclose(
            float(improvement[row]), expected_improvement, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError(f"expert {expert} coupled improvement drifted")
        draw = decision.selected_draw
        if not torch.equal(format_fit_sse[row, 0, 0].sum(), fit_sse[row, draw]):
            raise ValueError(f"expert {expert} selected coupled fit SSE does not close")
        if not torch.equal(
            format_confirmation_sse[row, 0, 0].sum(), confirmation_sse[row, draw]
        ):
            raise ValueError(
                f"expert {expert} selected coupled confirmation SSE does not close"
            )
        if selections is not None:
            evidence = selections[str(expert)].get("coupled_rotation")
            expected_evidence = {
                **decision.to_json(),
                "fit_post_projection_sse": {
                    str(draw): fit_map[draw] for draw in draws
                },
                "confirmation_post_projection_sse": {
                    str(draw): confirmation_map[draw] for draw in draws
                },
                **(
                    {
                        "pooled_post_projection_sse": {
                            str(draw): pooled_map[draw] for draw in draws
                        },
                        "pooled_source_energy": float(pooled_source_energy[row]),
                        "pooled_routed_occurrences": int(
                            pooled_occurrences[row]
                        ),
                    }
                    if pooled_map is not None
                    else {}
                ),
            }
            if evidence != expected_evidence:
                raise ValueError(f"expert {expert} coupled JSON evidence drifted")
    return {
        "selected": selected,
        "proposed": proposed,
        "evaluated_count": evaluated.sum(dim=1),
    }


def _validate_selection_semantics(
    metrics: Mapping[str, torch.Tensor],
    *,
    mode_ids: tuple[int, ...],
    evaluated: torch.Tensor,
    fit_sse: torch.Tensor,
    confirmation_sse: torch.Tensor,
    fit_counts: torch.Tensor,
    confirmation_counts: torch.Tensor,
    proposed_r13: torch.Tensor,
    proposed_r2: torch.Tensor,
    selected_r13: torch.Tensor,
    selected_r2: torch.Tensor,
    min_fit_documents: int,
    min_confirmation_documents: int,
    minimum_improvement: float,
) -> None:
    """Re-derive the phase-1 proposal and confirmation acceptance contract."""

    if min(min_fit_documents, min_confirmation_documents) < 1:
        raise ValueError("selection support thresholds must be positive")
    if not np.isfinite(minimum_improvement):
        raise ValueError("selection improvement margin must be finite")
    experts, modes, r2_modes = evaluated.shape
    if r2_modes != modes:
        raise ValueError("evaluated format grid must be square")
    mode_column = {mode: column for column, mode in enumerate(mode_ids)}
    r0_column = mode_column[0]
    fit_support = (fit_counts > 0).sum(dim=1)
    confirmation_support = (confirmation_counts > 0).sum(dim=1)
    expected_full = (fit_support >= min_fit_documents) & (
        confirmation_support >= min_confirmation_documents
    )
    full = evaluated.all(dim=(1, 2))
    r0_only = evaluated[:, r0_column, r0_column] & (
        evaluated.sum(dim=(1, 2)) == 1
    )
    if not torch.equal(full, expected_full) or not bool(
        torch.all(full | r0_only)
    ):
        raise ValueError("evaluated formats disagree with document support policy")
    if bool(torch.any(fit_sse[~evaluated] != 0)) or bool(
        torch.any(confirmation_sse[~evaluated] != 0)
    ):
        raise ValueError("unevaluated mode metrics must remain zero")

    confirmation_totals = confirmation_sse.sum(dim=3)
    expected_proposed_r13 = torch.zeros(experts, dtype=torch.uint8)
    expected_proposed_r2 = torch.zeros(experts, dtype=torch.uint8)
    for row in range(experts):
        if not bool(expected_full[row]):
            continue
        winner_r13, winner_r2 = min(
            ((r13, r2) for r13 in mode_ids for r2 in mode_ids),
            key=lambda rate_pair: (
                float(
                    confirmation_totals[
                        row,
                        mode_column[rate_pair[0]],
                        mode_column[rate_pair[1]],
                    ]
                ),
                rate_pair[0] + rate_pair[1],
                rate_pair[0],
                rate_pair[1],
            ),
        )
        expected_proposed_r13[row] = winner_r13
        expected_proposed_r2[row] = winner_r2
    if not torch.equal(proposed_r13, expected_proposed_r13) or not torch.equal(
        proposed_r2, expected_proposed_r2
    ):
        raise ValueError(
            "proposed formats do not equal the supported confirmation argmin"
        )

    improvement = _require_tensor(
        metrics,
        "confirmation_improvement",
        shape=(experts,),
        dtype=torch.float64,
    )
    ci95 = _require_tensor(
        metrics,
        "confirmation_ci95",
        shape=(experts, 2),
        dtype=torch.float64,
    )
    expected_selected_r13 = torch.zeros(experts, dtype=torch.uint8)
    expected_selected_r2 = torch.zeros(experts, dtype=torch.uint8)
    for row in range(experts):
        r13 = int(proposed_r13[row])
        r2 = int(proposed_r2[row])
        if (r13, r2) == (0, 0):
            if not bool(torch.isnan(improvement[row])) or not bool(
                torch.all(torch.isnan(ci95[row]))
            ):
                raise ValueError(
                    "R0/R0 proposals must not carry confirmation evidence"
                )
            continue
        baseline = float(
            confirmation_sse[row, r0_column, r0_column].sum()
        )
        trial = float(
            confirmation_sse[
                row, mode_column[r13], mode_column[r2]
            ].sum()
        )
        observed = 1.0 - trial / baseline if baseline > 0 else None
        if observed is None:
            if not bool(torch.isnan(improvement[row])) or not bool(
                torch.all(torch.isnan(ci95[row]))
            ):
                raise ValueError(
                    "zero-baseline proposals have invalid confirmation evidence"
                )
            continue
        if not math.isclose(
            float(improvement[row]), observed, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError("stored confirmation improvement does not close")
        low, high = (float(value) for value in ci95[row])
        if not np.isfinite((low, high)).all() or low > high:
            raise ValueError("nonzero proposals must have a finite ordered CI")
        if low > minimum_improvement:
            expected_selected_r13[row] = r13
            expected_selected_r2[row] = r2
    if not torch.equal(selected_r13, expected_selected_r13) or not torch.equal(
        selected_r2, expected_selected_r2
    ):
        raise ValueError("selected formats disagree with the confirmation gate")


def _expected_payload_metadata(
    layer: int,
    *,
    codebook: str,
    tailbite_context: int,
) -> dict[str, str]:
    if (
        isinstance(tailbite_context, bool)
        or not isinstance(tailbite_context, int)
        or not 1 <= tailbite_context <= 128
    ):
        raise ValueError("tailbite_context must be an integer in 1..128")
    return {
        "kind": CANDIDATE_POOL_KIND,
        "schema_version": str(CANDIDATE_POOL_SCHEMA_VERSION),
        "layer": str(layer),
        "codebook": codebook,
        "tailbite_context": str(tailbite_context),
    }


def _validate_payload_header(
    path: Path,
    layer: int,
    *,
    codebook: str,
    tailbite_context: int,
    trellis_bytes: int,
) -> None:
    header = read_safetensors_header(path)
    metadata = header.pop("__metadata__", None)
    expected_metadata = _expected_payload_metadata(
        layer,
        codebook=codebook,
        tailbite_context=tailbite_context,
    )
    if metadata != expected_metadata:
        raise ValueError(
            f"layer {layer} payload metadata mismatch: "
            f"{metadata!r} != {expected_metadata!r}"
        )
    expected_prefix = (
        f"language_model.model.layers.{layer}.block_sparse_moe.experts."
    )
    seen: set[tuple[int, str, str]] = set()
    with safe_open(path, framework="pt", device="cpu") as handle:
        names = set(handle.keys())
        if names != set(header):
            raise ValueError(f"layer {layer} raw and parsed payload headers disagree")
        for name in names:
            if not name.startswith(expected_prefix):
                raise ValueError(f"layer {layer} payload has foreign tensor {name}")
            suffix = name[len(expected_prefix) :]
            fields = suffix.split(".")
            if len(fields) != 3 or not fields[0].isdigit():
                raise ValueError(f"layer {layer} payload has malformed tensor {name}")
            expert = int(fields[0])
            matrix = fields[1]
            part_prefix = "qsrt_"
            if not fields[2].startswith(part_prefix):
                raise ValueError(f"layer {layer} payload has malformed tensor {name}")
            part = fields[2][len(part_prefix) :]
            key = (expert, matrix, part)
            if not 0 <= expert < EXPERTS_PER_LAYER:
                raise ValueError(f"layer {layer} payload has invalid expert {expert}")
            if matrix not in ("w1", "w3", "w2") or part not in (
                "trellis",
                "suh",
                "svh",
            ):
                raise ValueError(f"layer {layer} payload has invalid component {name}")
            _validate_payload_component(
                name,
                header[name],
                matrix=matrix,
                part=part,
                trellis_bytes=trellis_bytes,
            )
            seen.add(key)
    expected = {
        (expert, matrix, part)
        for expert in range(EXPERTS_PER_LAYER)
        for matrix in ("w1", "w3", "w2")
        for part in ("trellis", "suh", "svh")
    }
    if seen != expected:
        missing = sorted(expected - seen)[:4]
        extra = sorted(seen - expected)[:4]
        raise ValueError(
            f"layer {layer} payload components do not close; "
            f"missing={missing}, extra={extra}"
        )


def validate_candidate_layer_payload(
    path: str | Path,
    *,
    layer: int,
    codebook: str,
    tailbite_context: int,
    mode_ids: tuple[int, ...],
) -> None:
    """Validate one canonical all-expert candidate payload."""

    _validate_payload_header(
        Path(path),
        layer,
        codebook=codebook,
        tailbite_context=tailbite_context,
        trellis_bytes=_matrix_trellis_bytes(mode_ids),
    )


def validate_selection_ledger_evidence(
    ledger: Mapping[str, object],
    metrics: Mapping[str, torch.Tensor],
    *,
    mode_ids: tuple[int, ...],
    expert_ids: tuple[int, ...] = tuple(range(EXPERTS_PER_LAYER)),
    logical_trellis_schema: str = QSRT_SCHEMA,
) -> str:
    """Bind duplicated JSON selection evidence to the tensor ledger.

    The JSON evidence is intentionally descriptive, while the tensor sidecar
    is the allocator's numeric source.  Persisted candidate probes also need
    the JSON-selected mode to interpret the descriptor-free trellis payload,
    so a pool must not be sealed unless both representations agree exactly.
    """

    experts = len(expert_ids)
    selections = ledger.get("selections")
    expected_keys = {str(expert) for expert in expert_ids}
    if not isinstance(selections, dict) or set(selections) != expected_keys:
        raise ValueError("selection evidence does not cover exactly the ledger experts")

    if logical_trellis_schema not in LOGICAL_CANDIDATE_SCHEMAS:
        raise ValueError("selection evidence logical trellis schema is unsupported")
    ledger_trellis_schema = ledger.get("logical_trellis_schema")
    if ledger_trellis_schema is not None:
        if ledger_trellis_schema not in LOGICAL_CANDIDATE_SCHEMAS:
            raise ValueError(
                "selection ledger logical trellis schema is unsupported"
            )
        if ledger_trellis_schema != logical_trellis_schema:
            raise ValueError(
                "selection ledger logical trellis schema disagrees with the pool"
            )
    selected_r13 = _require_tensor(
        metrics, "selected_r13", shape=(experts,), dtype=torch.uint8
    )
    selected_r2 = _require_tensor(
        metrics, "selected_r2", shape=(experts,), dtype=torch.uint8
    )
    proposed_r13 = _require_tensor(
        metrics, "proposed_r13", shape=(experts,), dtype=torch.uint8
    )
    proposed_r2 = _require_tensor(
        metrics, "proposed_r2", shape=(experts,), dtype=torch.uint8
    )
    evaluated = _require_tensor(
        metrics,
        "format_evaluated",
        shape=(experts, len(mode_ids), len(mode_ids)),
        dtype=torch.bool,
    )
    fit_counts = metrics.get("fit_counts")
    confirmation_counts = metrics.get("confirmation_counts")
    if (
        fit_counts is None
        or fit_counts.dtype != torch.int32
        or fit_counts.ndim != 2
        or fit_counts.shape[0] != experts
        or confirmation_counts is None
        or confirmation_counts.dtype != torch.int32
        or confirmation_counts.ndim != 2
        or confirmation_counts.shape[0] != experts
    ):
        raise ValueError("selection support tensors are malformed")
    improvement = _require_tensor(
        metrics,
        "confirmation_improvement",
        shape=(experts,),
        dtype=torch.float64,
    )
    ci95 = _require_tensor(
        metrics,
        "confirmation_ci95",
        shape=(experts, 2),
        dtype=torch.float64,
    )
    fit_support = (fit_counts > 0).sum(dim=1)
    confirmation_support = (confirmation_counts > 0).sum(dim=1)
    matrix_rates = {
        "w1": ("n", 0),
        "w3": ("n", 0),
        "w2": ("k", 1),
    }

    def require_optional_float(
        actual: object,
        expected: float,
        label: str,
    ) -> None:
        if math.isnan(expected):
            if actual is not None:
                raise ValueError(f"{label} must be null")
            return
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise ValueError(f"{label} must be a finite number")
        if not math.isfinite(float(actual)) or not math.isclose(
            float(actual), expected, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError(f"{label} disagrees with the tensor ledger")

    for row, expert in enumerate(expert_ids):
        evidence = selections[str(expert)]
        if not isinstance(evidence, dict):
            raise ValueError(f"selection evidence for expert {expert} is malformed")
        selection = evidence.get("selection")
        if not isinstance(selection, dict):
            raise ValueError(f"selection decision for expert {expert} is malformed")
        expected_scalars = {
            "selected_r13": int(selected_r13[row]),
            "selected_r2": int(selected_r2[row]),
            "proposed_r13": int(proposed_r13[row]),
            "proposed_r2": int(proposed_r2[row]),
            "fit_documents": int(fit_support[row]),
            "confirmation_documents": int(confirmation_support[row]),
        }
        for name, expected in expected_scalars.items():
            actual = selection.get(name)
            if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
                raise ValueError(
                    f"selection evidence expert {expert} {name} disagrees "
                    "with the tensor ledger"
                )
        expected_accepted = (
            int(selected_r13[row]), int(selected_r2[row])
        ) != (0, 0)
        if selection.get("accepted") is not expected_accepted:
            raise ValueError(
                f"selection evidence expert {expert} accepted flag drifted"
            )
        if not isinstance(selection.get("reason"), str) or not selection["reason"]:
            raise ValueError(f"selection evidence expert {expert} has no reason")
        valid_replicates = selection.get("bootstrap_replicates_valid")
        if (
            isinstance(valid_replicates, bool)
            or not isinstance(valid_replicates, int)
            or valid_replicates < 0
        ):
            raise ValueError(
                f"selection evidence expert {expert} has invalid bootstrap support"
            )
        require_optional_float(
            selection.get("confirmation_relative_improvement"),
            float(improvement[row]),
            f"selection evidence expert {expert} confirmation improvement",
        )
        raw_ci = selection.get("confirmation_ci95")
        if not isinstance(raw_ci, list) or len(raw_ci) != 2:
            raise ValueError(f"selection evidence expert {expert} CI is malformed")
        for column, actual in enumerate(raw_ci):
            require_optional_float(
                actual,
                float(ci95[row, column]),
                f"selection evidence expert {expert} CI[{column}]",
            )

        expected_evaluated_modes = [
            mode
            for column, mode in enumerate(mode_ids)
            if bool(evaluated[row, column].any())
            or bool(evaluated[row, :, column].any())
        ]
        if evidence.get("evaluated_modes") != expected_evaluated_modes:
            raise ValueError(
                f"selection evidence expert {expert} evaluated modes drifted"
            )
        expected_formats = [
            (r13, r2)
            for r13_column, r13 in enumerate(mode_ids)
            for r2_column, r2 in enumerate(mode_ids)
            if bool(evaluated[row, r13_column, r2_column])
        ]
        mode_coding = evidence.get("mode_coding")
        if not isinstance(mode_coding, dict) or set(mode_coding) != {
            _format_label(r13, r2) for r13, r2 in expected_formats
        }:
            raise ValueError(
                f"selection evidence expert {expert} mode coding drifted"
            )
        for rate_pair in expected_formats:
            format_name = _format_label(*rate_pair)
            coding = mode_coding[format_name]
            if not isinstance(coding, dict):
                raise ValueError(
                    f"selection evidence expert {expert} {format_name} coding is malformed"
                )
            expected_trellis_bytes = 3 * _matrix_trellis_bytes(mode_ids)
            expected_scale_bytes = 6 * (INTERMEDIATE_CHANNELS + LATENT_CHANNELS)
            expected_coding_scalars = {
                "trellis_bytes": expected_trellis_bytes,
                "scale_bytes": expected_scale_bytes,
                "total_bytes_before_layer_deduplication": (
                    expected_trellis_bytes + expected_scale_bytes
                ),
            }
            for name, expected in expected_coding_scalars.items():
                if coding.get(name) != expected:
                    raise ValueError(
                        f"selection evidence expert {expert} {format_name} "
                        f"{name} drifted"
                    )
            matrices = coding.get("matrices")
            if not isinstance(matrices, dict) or set(matrices) != set(
                matrix_rates
            ):
                raise ValueError(
                    f"selection evidence expert {expert} {format_name} matrices drifted"
                )
            for matrix, (rate_axis, pair_index) in matrix_rates.items():
                matrix_coding = matrices[matrix]
                if not isinstance(matrix_coding, dict):
                    raise ValueError(
                        f"selection evidence expert {expert} {format_name} "
                        f"{matrix} coding is malformed"
                    )
                expected_mode = rate_pair[pair_index]
                expected_matrix = {
                    "mode": (
                        H308.name
                        if expected_mode == H308.mode_id
                        else K2.name
                        if expected_mode == K2.mode_id
                        else f"R{expected_mode}"
                    ),
                    "mode_id": expected_mode,
                    "rate_axis": rate_axis,
                }
                for name, expected in expected_matrix.items():
                    if matrix_coding.get(name) != expected:
                        raise ValueError(
                            f"selection evidence expert {expert} {format_name} "
                            f"{matrix} {name} drifted"
                        )
                proxy = matrix_coding.get("proxy")
                if (
                    isinstance(proxy, bool)
                    or not isinstance(proxy, (int, float))
                    or not math.isfinite(float(proxy))
                    or float(proxy) < 0
                ):
                    raise ValueError(
                        f"selection evidence expert {expert} {format_name} "
                        f"{matrix} proxy is invalid"
                    )
    return logical_trellis_schema


def validate_pooled_fixed_profile_ledger_evidence(
    ledger: Mapping[str, object],
    metrics: Mapping[str, torch.Tensor],
    *,
    expert_ids: tuple[int, ...] = tuple(range(EXPERTS_PER_LAYER)),
    logical_trellis_schema: str = QSRT_SCHEMA,
) -> str:
    """Bind pooled fixed-profile JSON evidence to its tensor sidecar."""

    if logical_trellis_schema not in LOGICAL_CANDIDATE_SCHEMAS:
        raise ValueError("pooled selection logical trellis schema is unsupported")
    if ledger.get("logical_trellis_schema") != logical_trellis_schema:
        raise ValueError("pooled selection logical trellis schema drifted")
    selections = ledger.get("selections")
    expected_keys = {str(expert) for expert in expert_ids}
    if not isinstance(selections, dict) or set(selections) != expected_keys:
        raise ValueError("pooled selection evidence does not cover exactly the experts")

    fields = {
        "baseline_sse": metrics["pooled_baseline_sse"],
        "replacement_sse": metrics["pooled_replacement_sse"],
        "selected_sse": metrics["pooled_selected_sse"],
        "source_energy": metrics["pooled_source_energy"],
        "effective_sample_size": metrics["effective_sample_size"],
    }
    occurrences = metrics["routed_occurrences"]
    selected_refit = metrics["selected_functional_refit"]
    draws = metrics["coupled_draw_selected"]
    for row, expert in enumerate(expert_ids):
        evidence = selections[str(expert)]
        if not isinstance(evidence, dict):
            raise ValueError(f"pooled evidence for expert {expert} is malformed")
        if evidence.get("selected_down_target") != (
            "functional_ridge" if bool(selected_refit[row]) else "source_pool"
        ):
            raise ValueError(f"pooled expert {expert} down-target decision drifted")
        if evidence.get("routed_occurrences") != int(occurrences[row]):
            raise ValueError(f"pooled expert {expert} routed support drifted")
        if evidence.get("coupled_draw") != int(draws[row]):
            raise ValueError(f"pooled expert {expert} coupled draw drifted")
        for name, values in fields.items():
            actual = evidence.get(name)
            expected = float(values[row])
            if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isclose(
                float(actual), expected, rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError(f"pooled expert {expert} {name} drifted")
    return logical_trellis_schema


def _validate_payload_component(
    name: str,
    entry: object,
    *,
    matrix: str,
    part: str,
    trellis_bytes: int = MATRIX_TRELLIS_BYTES,
) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"candidate payload component {name} has no tensor header")
    if part == "trellis":
        expected_dtype = "I16"
        expected_shape = [trellis_bytes // torch.int16.itemsize]
    else:
        expected_dtype = "F16"
        if part == "suh":
            channels = INTERMEDIATE_CHANNELS if matrix == "w2" else LATENT_CHANNELS
        else:
            channels = LATENT_CHANNELS if matrix == "w2" else INTERMEDIATE_CHANNELS
        expected_shape = [channels]
    if entry.get("dtype") != expected_dtype or entry.get("shape") != expected_shape:
        raise ValueError(
            f"candidate payload component {name} has "
            f"{entry.get('dtype')} {entry.get('shape')}, expected "
            f"{expected_dtype} {expected_shape}"
        )


def load_qsrt_candidate_pool(
    root: str | Path,
    *,
    validate_payload_headers: bool = True,
    require_completion: bool = True,
    verify_completion_hashes: bool = False,
) -> QSRTCandidatePool:
    """Load and strictly validate a complete 92-layer candidate pool."""

    root = Path(root).resolve()
    manifest_path = root / "qsrt-candidate-manifest.json"
    manifest = _read_json(manifest_path)
    expected_manifest = {
        "kind": CANDIDATE_POOL_KIND,
        "schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
        "source_model": C.MODEL_ID,
        "source_revision": C.REVISION,
        "damage_metric": OFFICIAL_SOURCE_DAMAGE_METRIC,
        "damage_already_natural_route_and_gate_weighted": True,
    }
    for name, expected in expected_manifest.items():
        if manifest.get(name) != expected:
            raise ValueError(
                f"candidate manifest {name} mismatch: "
                f"{manifest.get(name)!r} != {expected!r}"
            )
    mode_ids = tuple(int(mode) for mode in manifest.get("mode_ids", ()))
    coupled_rotation_contract = manifest.get("coupled_rotation")
    fixed_profile = _fixed_mode(mode_ids) is not None
    conventional_grid = (
        bool(mode_ids)
        and mode_ids[0] == 0
        and mode_ids == tuple(sorted(set(mode_ids)))
        and all(mode in (0, 1, 2) for mode in mode_ids)
    )
    if not (fixed_profile or conventional_grid):
        raise ValueError("candidate pool has an invalid rate-transfer mode policy")
    if mode_ids == (K2.mode_id,) and not isinstance(
        coupled_rotation_contract, Mapping
    ):
        raise ValueError("fixed K2 requires one coupled rotation contract")
    if coupled_rotation_contract is not None and mode_ids not in (
        (R0.mode_id,),
        (H308.mode_id,),
        (K2.mode_id,),
    ):
        raise ValueError(
            "coupled candidate pools require a fixed K2, K3, or "
            "22-K3/2-K4 schedule"
        )
    expected_contract = _selection_contract(mode_ids)
    for name in ("format_grid", "shared_r"):
        if manifest.get(name) != expected_contract[name]:
            raise ValueError(f"candidate manifest {name} drifted")
    codebook = manifest.get("codebook")
    if codebook not in QSRT_CODEBOOKS:
        raise ValueError("candidate manifest codebook is unsupported")
    tailbite_context = manifest.get("tailbite_context")
    if (
        not isinstance(tailbite_context, int)
        or isinstance(tailbite_context, bool)
        or not 1 <= tailbite_context <= 128
    ):
        raise ValueError("candidate manifest tailbite context is invalid")
    declared_trellis_schema = manifest.get("logical_trellis_schema")
    if declared_trellis_schema not in LOGICAL_CANDIDATE_SCHEMAS:
        raise ValueError("candidate manifest logical trellis schema is unsupported")
    completion = (
        validate_candidate_pool_completion(
            root,
            verify_hashes=verify_completion_hashes,
        )
        if require_completion
        else None
    )

    damage = np.empty((C.NUM_MOE_LAYERS, C.NUM_EXPERTS), dtype=np.float64)
    selected_r13 = np.empty_like(damage, dtype=np.uint8)
    selected_r2 = np.empty_like(damage, dtype=np.uint8)
    proposed_r13 = np.empty_like(damage, dtype=np.uint8)
    proposed_r2 = np.empty_like(damage, dtype=np.uint8)
    evaluated_counts = np.empty_like(damage, dtype=np.uint8)
    coupled_rotation_draws = (
        np.empty_like(damage, dtype=np.uint8)
        if coupled_rotation_contract is not None
        else None
    )
    histogram = {
        _format_label(r13, r2): 0 for r13 in mode_ids for r2 in mode_ids
    }
    trellis_schema: str | None = None
    for layer in C.MOE_LAYERS:
        payload_path, metrics_path, ledger_path = _canonical_layer_paths(root, layer)
        for path in (payload_path, metrics_path, ledger_path):
            if not path.is_file() or path.stat().st_size <= 0:
                raise ValueError(f"candidate pool is incomplete; missing {path}")
        ledger = _read_json(ledger_path)
        expected_ledger = {
            "kind": CANDIDATE_POOL_KIND,
            "schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
            "complete": True,
            "layer": layer,
            "all_experts": True,
            "payload": payload_path.name,
            "metrics": metrics_path.name,
            "source_revision": C.REVISION,
        }
        for name, expected in expected_ledger.items():
            if ledger.get(name) != expected:
                raise ValueError(
                    f"layer {layer} ledger {name} mismatch: "
                    f"{ledger.get(name)!r} != {expected!r}"
                )
        if ledger.get("experts") != list(range(C.NUM_EXPERTS)):
            raise ValueError(f"layer {layer} ledger is not a canonical all-expert pool")
        if ledger.get("codebook") != codebook:
            raise ValueError(f"layer {layer} codebook drifted")
        if ledger.get("tailbite_context") != tailbite_context:
            raise ValueError(f"layer {layer} tailbite context drifted")
        contract = ledger.get("selection_contract", {})
        if tuple(contract.get("mode_ids", ())) != mode_ids:
            raise ValueError(f"layer {layer} selection mode table drifted")
        metric_contract = contract.get("metric_contract")
        pooled_fixed_profile = metric_contract == {
            "kind": POOLED_FIXED_PROFILE_METRICS_KIND,
            "schema_version": POOLED_FIXED_PROFILE_METRICS_SCHEMA_VERSION,
        }
        expected_layer_rotation = (
            {
                "source": "model_rotation_plan",
                "plan_sha256": _json_sha256(coupled_rotation_contract.get("plan")),
            }
            if pooled_fixed_profile
            and isinstance(coupled_rotation_contract, Mapping)
            and coupled_rotation_contract.get("source") == "model_rotation_plan"
            else coupled_rotation_contract
        )
        if ledger.get("coupled_rotation") != expected_layer_rotation:
            raise ValueError(f"layer {layer} coupled rotation contract drifted")
        expected_selection_contract = (
            pooled_fixed_profile_selection_contract(mode_ids)
            if pooled_fixed_profile
            else _selection_contract(mode_ids)
        )
        for name, expected in expected_selection_contract.items():
            if contract.get(name) != expected:
                raise ValueError(
                    f"layer {layer} selection contract {name} drifted"
                )
        damage_contract = ledger.get("damage_contract", {})
        if damage_contract.get("metric") != OFFICIAL_SOURCE_DAMAGE_METRIC:
            raise ValueError(f"layer {layer} damage contract drifted")

        metrics = load_file(str(metrics_path), device="cpu")
        if pooled_fixed_profile:
            if not isinstance(coupled_rotation_contract, Mapping) or coupled_rotation_contract.get(
                "source"
            ) != "model_rotation_plan":
                raise ValueError(
                    f"layer {layer} pooled fixed profile requires a model rotation plan"
                )
            rotation_plan = CoupledRotationPlan.from_json(
                coupled_rotation_contract.get("plan")
            )
            expected_draws = rotation_plan.for_layer(layer)
            validated = validate_pooled_fixed_profile_metrics(
                metrics,
                mode_ids=mode_ids,
                expected_coupled_draws=expected_draws,
            )
            coupled_validated = {
                "selected": validated["coupled_draw_selected"],
                "proposed": validated["coupled_draw_selected"],
                "evaluated_count": torch.ones(C.NUM_EXPERTS, dtype=torch.int64),
            }
            layer_trellis_schema = validate_pooled_fixed_profile_ledger_evidence(
                ledger,
                metrics,
                logical_trellis_schema=str(declared_trellis_schema),
            )
        else:
            fit_documents = int(contract.get("fit_documents", -1))
            confirmation_documents = int(
                contract.get("confirmation_documents", -1)
            )
            if min(fit_documents, confirmation_documents) < 1:
                raise ValueError(f"layer {layer} has invalid document partitions")
            min_fit_documents = int(contract.get("min_fit_documents", -1))
            min_confirmation_documents = int(
                contract.get("min_confirmation_documents", -1)
            )
            minimum_improvement = float(
                contract.get("minimum_improvement", float("nan"))
            )
            validated = validate_layer_metrics(
                metrics,
                mode_ids=mode_ids,
                fit_documents=fit_documents,
                confirmation_documents=confirmation_documents,
                min_fit_documents=min_fit_documents,
                min_confirmation_documents=min_confirmation_documents,
                minimum_improvement=minimum_improvement,
            )
            coupled_validated = validate_coupled_rotation_metrics(
                metrics,
                coupled_rotation_contract,
                layer=layer,
                expert_ids=tuple(range(C.NUM_EXPERTS)),
                min_fit_documents=min_fit_documents,
                min_confirmation_documents=min_confirmation_documents,
                minimum_improvement=minimum_improvement,
                ledger=ledger,
            )
            layer_trellis_schema = validate_selection_ledger_evidence(
                ledger,
                metrics,
                mode_ids=mode_ids,
                logical_trellis_schema=str(declared_trellis_schema),
            )
        if trellis_schema is None:
            trellis_schema = layer_trellis_schema
        elif layer_trellis_schema != trellis_schema:
            raise ValueError("candidate pool mixes logical trellis schemas")
        row = layer - 1
        damage[row] = validated["damage"].numpy()
        selected_r13[row] = validated["selected_r13"].numpy()
        selected_r2[row] = validated["selected_r2"].numpy()
        proposed_r13[row] = validated["proposed_r13"].numpy()
        proposed_r2[row] = validated["proposed_r2"].numpy()
        evaluated_counts[row] = validated["evaluated_format_count"].numpy()
        if coupled_rotation_draws is not None:
            if coupled_validated is None:
                raise AssertionError("coupled rotation validation disappeared")
            coupled_rotation_draws[row] = coupled_validated["selected"].numpy()
            actual_draw_histogram = {
                str(draw): int(
                    (coupled_validated["selected"] == draw).sum()
                )
                for draw in range(8)
                if bool((coupled_validated["selected"] == draw).any())
            }
            if ledger.get("selected_coupled_draw_histogram") != actual_draw_histogram:
                raise ValueError(
                    f"layer {layer} selected coupled-draw histogram does not close"
                )
        actual_histogram = {
            _format_label(r13, r2): int(
                (
                    (validated["selected_r13"] == r13)
                    & (validated["selected_r2"] == r2)
                ).sum()
            )
            for r13 in mode_ids
            for r2 in mode_ids
        }
        if ledger.get("selected_format_histogram") != actual_histogram:
            raise ValueError(f"layer {layer} selected-format histogram does not close")
        for name, count in actual_histogram.items():
            histogram[name] += count
        if validate_payload_headers:
            _validate_payload_header(
                payload_path,
                layer,
                codebook=codebook,
                tailbite_context=tailbite_context,
                trellis_bytes=_matrix_trellis_bytes(mode_ids),
            )

    if completion is not None and completion.get(
        "logical_trellis_schema"
    ) != trellis_schema:
        raise ValueError(
            "candidate completion logical trellis schema disagrees with "
            "the selection ledgers"
        )
    if (
        declared_trellis_schema is not None
        and declared_trellis_schema != trellis_schema
    ):
        raise ValueError(
            "candidate manifest logical trellis schema disagrees with "
            "the selection ledgers"
        )

    return QSRTCandidatePool(
        root=root,
        manifest=manifest,
        damage=damage,
        selected_r13=selected_r13,
        selected_r2=selected_r2,
        proposed_r13=proposed_r13,
        proposed_r2=proposed_r2,
        evaluated_format_counts=evaluated_counts,
        format_histogram=histogram,
        completion=completion,
        content_sha256=(
            str(completion["content_sha256"])
            if completion is not None
            else None
        ),
        trellis_schema=trellis_schema,
        mode_ids=mode_ids,
        codebook=codebook,
        coupled_rotation_draws=coupled_rotation_draws,
    )


def raw_keep_container_bytes(keep_mask: np.ndarray) -> int:
    """Return bytes for the legacy equal-cost retained-tier diagnostic.

    ``QSRTLayerLayout`` owns only the compressed trellis slab because the
    production high tier lives in an external X4T sidecar.  This older ranking
    diagnostic deliberately models a fixed-size raw MXFP4 payload per retained
    expert; omitting that payload would make a promotion appear to *remove*
    bytes.  Production QSRT/X4T allocation uses
    :func:`qsrt.pack.qsrt_allocation.qsrt_total_container_bytes` with the
    measured per-expert X4T costs instead.
    """

    expected = (C.NUM_MOE_LAYERS, C.NUM_EXPERTS)
    if keep_mask.shape != expected or keep_mask.dtype != np.bool_:
        raise ValueError(f"keep mask must be bool with shape {expected}")
    return sum(
        _nominal_raw_keep_layer_bytes(int(keep_mask[row].sum()))
        for row in range(C.NUM_MOE_LAYERS)
    )


def _nominal_raw_keep_layer_bytes(keep_count: int) -> int:
    """Compressed slab plus fixed-size raw MXFP4 bytes for a legacy keep tier."""

    return (
        QSRTLayerLayout(
            compressed_experts=C.NUM_EXPERTS - keep_count,
            x4t_experts=keep_count,
        ).disk_bytes
        + keep_count
        * 3
        * (
            INTERMEDIATE_CHANNELS * LATENT_CHANNELS // 2
            + INTERMEDIATE_CHANNELS * LATENT_CHANNELS // 32
        )
    )


def keep_mask_from_allocation_document(document: dict) -> np.ndarray:
    """Parse a raw-keep diagnostic or the production EXL3 keep map."""

    try:
        layers = document["layers"]
    except (KeyError, TypeError) as exc:
        raise ValueError("allocation is missing its layers object") from exc
    if not isinstance(layers, dict):
        raise ValueError("allocation layers must be an object")
    mask = np.zeros((C.NUM_MOE_LAYERS, C.NUM_EXPERTS), dtype=np.bool_)
    universe = set(range(C.NUM_EXPERTS))
    for row, layer in enumerate(C.MOE_LAYERS):
        try:
            entry = layers[str(layer)]
        except KeyError as exc:
            raise ValueError(f"allocation is missing MoE layer {layer}") from exc
        if not isinstance(entry, dict):
            raise ValueError(f"allocation layer {layer} must be an object")
        keep = entry.get("keep")
        compressed = entry.get("compressed", entry.get("exl3"))
        if not isinstance(keep, list) or not isinstance(compressed, list):
            raise ValueError(f"allocation layer {layer} tiers must be lists")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in keep + compressed
        ):
            raise ValueError(f"allocation layer {layer} has a non-integer expert ID")
        keep_set, compressed_set = set(keep), set(compressed)
        if (
            len(keep_set) != len(keep)
            or len(compressed_set) != len(compressed)
            or keep_set & compressed_set
            or keep_set | compressed_set != universe
        ):
            raise ValueError(f"allocation layer {layer} tiers do not form a partition")
        mask[row, keep] = True
    return mask


def _validate_damage(damage: np.ndarray) -> np.ndarray:
    damage = np.asarray(damage, dtype=np.float64)
    expected = (C.NUM_MOE_LAYERS, C.NUM_EXPERTS)
    if damage.shape != expected:
        raise ValueError(f"damage must have shape {expected}")
    if not np.isfinite(damage).all() or np.any(damage < 0):
        raise ValueError("damage must contain finite non-negative values")
    return damage


def _layer_orders(damage: np.ndarray) -> tuple[np.ndarray, ...]:
    # Stable ordering gives deterministic layer/expert tie-breaking.
    return tuple(
        np.argsort(-damage[row], kind="stable")
        for row in range(damage.shape[0])
    )


def _prefix_mask(orders: tuple[np.ndarray, ...], counts: np.ndarray) -> np.ndarray:
    mask = np.zeros((len(orders), len(orders[0])), dtype=np.bool_)
    for row, (order, count) in enumerate(zip(orders, counts, strict=True)):
        mask[row, order[: int(count)]] = True
    return mask


def _global_top_counts(damage: np.ndarray, keep_count: int) -> np.ndarray:
    flat_order = np.argsort(-damage.reshape(-1), kind="stable")
    selected = flat_order[:keep_count]
    return np.bincount(selected // damage.shape[1], minlength=damage.shape[0])


def choose_qsrt_raw_keep_allocation(
    damage: np.ndarray,
    *,
    keep_fraction: float | None = None,
    keep_count: int | None = None,
    target_container_bytes: int | None = None,
) -> QSRTRawKeepAllocation:
    """Choose retained experts and account the exact aligned container bytes.

    With a count/fraction target this is the exact top-damage allocation.  A
    byte target first chooses the maximum count permitted by the unaligned
    marginal cost, then repairs any 4-KiB layer-alignment overrun using the
    minimum-damage one-expert cross-layer moves available at each step.  If no
    alignment-reducing move exists, one promotion is conservatively removed.
    """

    damage = _validate_damage(damage)
    specified = sum(
        value is not None
        for value in (keep_fraction, keep_count, target_container_bytes)
    )
    if specified != 1:
        raise ValueError(
            "specify exactly one of keep_fraction, keep_count, or "
            "target_container_bytes"
        )
    total = damage.size
    all_compressed = C.NUM_MOE_LAYERS * _nominal_raw_keep_layer_bytes(0)
    layer_base = _nominal_raw_keep_layer_bytes(0)
    if any(
        _nominal_raw_keep_layer_bytes(keep)
        < layer_base + keep * RAW_KEEP_PROMOTION_BYTES
        for keep in range(C.NUM_EXPERTS + 1)
    ):
        raise AssertionError(
            "allocation optimality proof requires non-negative alignment residuals"
        )
    all_kept = C.NUM_MOE_LAYERS * _nominal_raw_keep_layer_bytes(C.NUM_EXPERTS)
    if keep_fraction is not None:
        if not 0.0 <= keep_fraction <= 1.0:
            raise ValueError("keep_fraction must lie in [0, 1]")
        raw_target = round(keep_fraction * total)
    elif keep_count is not None:
        if isinstance(keep_count, bool) or not 0 <= keep_count <= total:
            raise ValueError(f"keep_count must lie in 0..{total}")
        raw_target = keep_count
    else:
        assert target_container_bytes is not None
        if isinstance(target_container_bytes, bool):
            raise TypeError("target_container_bytes must be an integer")
        if target_container_bytes < all_compressed:
            raise ValueError(
                f"byte target {target_container_bytes} is below all-compressed size "
                f"{all_compressed}"
            )
        if target_container_bytes >= all_kept:
            raw_target = total
        else:
            raw_target = min(
                total,
                (target_container_bytes - all_compressed) // RAW_KEEP_PROMOTION_BYTES,
            )

    orders = _layer_orders(damage)
    counts = _global_top_counts(damage, int(raw_target))
    keep_mask = _prefix_mask(orders, counts)
    actual_bytes = raw_keep_container_bytes(keep_mask)
    repair_swaps = 0
    repair_damage = 0.0
    if target_container_bytes is not None and actual_bytes > target_container_bytes:
        while actual_bytes > target_container_bytes:
            best: tuple[float, int, int, int] | None = None
            for source in range(C.NUM_MOE_LAYERS):
                source_count = int(counts[source])
                if source_count == 0:
                    continue
                source_before = _nominal_raw_keep_layer_bytes(source_count)
                source_after = _nominal_raw_keep_layer_bytes(source_count - 1)
                removed_damage = damage[source, orders[source][source_count - 1]]
                for destination in range(C.NUM_MOE_LAYERS):
                    destination_count = int(counts[destination])
                    if destination == source or destination_count == C.NUM_EXPERTS:
                        continue
                    destination_before = _nominal_raw_keep_layer_bytes(
                        destination_count
                    )
                    destination_after = _nominal_raw_keep_layer_bytes(
                        destination_count + 1
                    )
                    byte_delta = (
                        source_after
                        + destination_after
                        - source_before
                        - destination_before
                    )
                    if byte_delta >= 0:
                        continue
                    added_damage = damage[
                        destination, orders[destination][destination_count]
                    ]
                    quality_loss = float(removed_damage - added_damage)
                    candidate = (quality_loss, source, destination, byte_delta)
                    if best is None or candidate < best:
                        best = candidate
            if best is None:
                # Removing one global promotion creates more slack than the
                # maximum total alignment overhead, so this always terminates.
                raw_target -= 1
                counts = _global_top_counts(damage, int(raw_target))
                keep_mask = _prefix_mask(orders, counts)
                actual_bytes = raw_keep_container_bytes(keep_mask)
                repair_swaps = 0
                repair_damage = 0.0
                continue
            quality_loss, source, destination, byte_delta = best
            counts[source] -= 1
            counts[destination] += 1
            actual_bytes += byte_delta
            repair_swaps += 1
            repair_damage += quality_loss
        keep_mask = _prefix_mask(orders, counts)
        if raw_keep_container_bytes(keep_mask) != actual_bytes:
            raise AssertionError("allocation byte accounting drifted during repair")

    return QSRTRawKeepAllocation(
        keep_mask=keep_mask,
        layer_orders=orders,
        target_container_bytes=target_container_bytes,
        container_bytes=actual_bytes,
        raw_target_keep_count=int(raw_target),
        retained_experts=int(keep_mask.sum()),
        alignment_repair_swaps=repair_swaps,
        alignment_repair_damage_cost=repair_damage,
    )


def raw_keep_allocation_optimality(allocation: QSRTRawKeepAllocation) -> str:
    """State whether the returned keep set has a global optimality proof."""

    if allocation.target_container_bytes is None:
        return CERTIFIED_ALLOCATION_OPTIMALITY
    all_compressed = C.NUM_MOE_LAYERS * _nominal_raw_keep_layer_bytes(0)
    maximum_count = min(
        C.NUM_MOE_LAYERS * C.NUM_EXPERTS,
        (
            allocation.target_container_bytes - all_compressed
        )
        // RAW_KEEP_PROMOTION_BYTES,
    )
    if (
        allocation.retained_experts == maximum_count
        and allocation.alignment_repair_swaps == 0
    ):
        # Layer-alignment residuals are non-negative. No allocation can fit
        # another equal-cost promotion, and the global top-damage set at this
        # count already fits, so this is the exact optimum.
        return CERTIFIED_ALLOCATION_OPTIMALITY
    return UNCERTIFIED_ALLOCATION_OPTIMALITY


def allocation_document(
    pool: QSRTCandidatePool, allocation: QSRTRawKeepAllocation
) -> dict:
    keep_mask = allocation.keep_mask
    if (
        pool.damage.shape != keep_mask.shape
        or pool.selected_r13.shape != keep_mask.shape
        or pool.selected_r2.shape != keep_mask.shape
    ):
        raise ValueError("candidate pool and allocation shapes disagree")
    layers: dict[str, dict] = {}
    for row, layer in enumerate(C.MOE_LAYERS):
        keep = np.flatnonzero(keep_mask[row]).tolist()
        compressed = np.flatnonzero(~keep_mask[row]).tolist()
        format_codes = [
            ExpertFormatSpec.x4t().code
            if keep_mask[row, expert]
            else ExpertFormatSpec.compressed(
                int(pool.selected_r13[row, expert]),
                int(pool.selected_r2[row, expert]),
            ).code
            for expert in range(C.NUM_EXPERTS)
        ]
        layers[str(layer)] = {
            "keep": keep,
            "compressed": compressed,
            "format_codes": format_codes,
        }
    total_damage = float(pool.damage.sum())
    retained_damage = float(pool.damage[keep_mask].sum())
    compressed_damage = float(pool.damage[~keep_mask].sum())
    if not np.isclose(total_damage, retained_damage + compressed_damage):
        raise AssertionError("allocation damage ledger does not close")
    parameters = C.NUM_MOE_LAYERS * C.NUM_EXPERTS * 3 * 3072 * 3584
    return {
        "kind": RAW_KEEP_ALLOCATION_KIND,
        "schema_version": ALLOCATION_SCHEMA_VERSION,
        "meta": {
            "candidate_pool": str(pool.root),
            "candidate_schema_version": pool.manifest["schema_version"],
            "candidate_pool_content_sha256": pool.content_sha256,
            "candidate_logical_trellis_schema": pool.trellis_schema,
            "candidate_codebook": pool.codebook,
            "candidate_mode_ids": list(pool.mode_ids),
            "source_revision": pool.manifest["source_revision"],
            "damage_metric": pool.damage_metric,
            "damage_weighting": pool.damage_weighting,
            "damage_provenance": pool.damage_provenance,
            "retained_experts": allocation.retained_experts,
            "compressed_experts": int(keep_mask.size - allocation.retained_experts),
            "realized_keep_fraction": allocation.retained_experts / keep_mask.size,
            "raw_target_keep_count": allocation.raw_target_keep_count,
            "target_container_bytes": allocation.target_container_bytes,
            "container_bytes": allocation.container_bytes,
            "effective_bits_per_original_expert_weight": (
                allocation.container_bytes * 8 / parameters
            ),
            "all_compressed_damage": total_damage,
            "retained_damage_avoided": retained_damage,
            "remaining_compressed_damage": compressed_damage,
            "alignment_repair_swaps": allocation.alignment_repair_swaps,
            "alignment_repair_damage_cost": allocation.alignment_repair_damage_cost,
            "raw_keep_allocation_optimality": raw_keep_allocation_optimality(allocation),
            "format_histogram_before_keep": pool.format_histogram,
        },
        "layers": layers,
    }


def write_allocation(path: str | Path, document: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, indent=1, sort_keys=True))
    temporary.replace(path)
    return path

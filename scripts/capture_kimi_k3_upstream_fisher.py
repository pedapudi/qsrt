#!/usr/bin/env python
"""Capture final-logit Fisher factors for Kimi-K3 QSRT W1/W3 rounding."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from instanttensor import Backend
from safetensors import safe_open

_REPOSITORY = Path(__file__).resolve().parents[1]
if str(_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY))

from qsrt.instanttensor_kimi import InstantTensorLoadConfig
from qsrt.kimi_boundary_slabs import KimiBoundarySlabArchive
from qsrt.kimi_cotangent_slabs import KimiCotangentSlabWorkspace
from qsrt.kimi_official_forward import (
    OfficialKimiForwardAdapter,
    load_official_kimi_runtime,
    new_meta_decoder_layer,
)
from qsrt.kimi_quantized_forward import QSRTAnchorPayload, QSRTKimiForwardAdapter
from qsrt.kimi_suffix_pipeline import KimiSuffixPipeline
from qsrt.kimi_upstream_factors import KimiUpstreamFactorArchive
from qsrt.kimi_upstream_pipelined_reverse import KimiPipelinedUpstreamReverse
from qsrt.qsrt_atoms_v2 import unpack_atoms_v2_format_section


DEFAULT_WEIGHT_CHECKPOINT = Path(
    "/data/cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
)
DEFAULT_CODE_CHECKPOINT = Path(
    "/data/cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721"
)
DEFAULT_PROFILE = Path(
    "/data/releases/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-v1"
)
RUN_FILENAME = "upstream-fisher-run.json"
FAILURE_FILENAME = "upstream-fisher-failure.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _parse_devices(value: str) -> tuple[torch.device, ...]:
    try:
        indices = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "devices must be comma-separated integers"
        ) from error
    if not indices or len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("devices must be nonempty and unique")
    return tuple(torch.device("cuda", index) for index in indices)


def _parse_instanttensor_backend(value: str) -> Backend:
    try:
        return Backend[value.strip().upper()]
    except KeyError as error:
        names = ", ".join(item.name.lower() for item in Backend)
        raise argparse.ArgumentTypeError(
            f"InstantTensor backend must be one of: {names}"
        ) from error


def _stored_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(path)
        candidate = candidate.parent
    return candidate


def _require_values(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    for key, value in expected.items():
        if observed.get(key) != value:
            raise ValueError(
                f"{label} {key} differs: stored={observed.get(key)!r}, "
                f"requested={value!r}"
            )


def _routed_geometry(runtime) -> tuple[tuple[int, ...], int, int, int]:
    layers: list[int] = []
    geometries: set[tuple[int, int, int]] = set()
    for layer in range(int(runtime.text_config.num_hidden_layers)):
        module = new_meta_decoder_layer(runtime, layer)
        block = getattr(module, "block_sparse_moe", None)
        if block is None:
            continue
        experts = tuple(block.experts)
        if not experts:
            raise ValueError(f"decoder layer {layer} has no routed experts")
        w1 = getattr(experts[0], "w1", None)
        if w1 is None or not hasattr(w1, "weight") or w1.weight.ndim != 2:
            raise ValueError(f"decoder layer {layer} has no routed W1 matrix")
        intermediate, hidden = (int(value) for value in w1.weight.shape)
        geometries.add((len(experts), hidden, intermediate))
        layers.append(layer)
    if not layers or len(geometries) != 1:
        raise ValueError("routed W1 geometry is inconsistent across decoder layers")
    experts, hidden, intermediate = geometries.pop()
    return tuple(layers), experts, hidden, intermediate


def _load_profile_draws(
    profile: Path,
    *,
    layers: Sequence[int],
    num_experts: int,
) -> tuple[dict[int, tuple[int, ...]], str, str]:
    completion = profile / "qsrt-completion.json"
    if not completion.is_file():
        raise FileNotFoundError(f"QSRT profile lacks its completion record: {completion}")
    document = json.loads(completion.read_text())
    if document.get("complete") is not True:
        raise ValueError("QSRT profile is not complete")
    if int(document.get("layer_count", len(document.get("layers", [])))) != len(layers):
        raise ValueError("QSRT profile layer count differs from the routed model")

    draws_by_layer: dict[int, tuple[int, ...]] = {}
    profile_names: set[str] = set()
    for layer in layers:
        path = profile / f"qsrt-layer-{layer:05d}.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"QSRT profile is missing decoder layer {layer}")
        with safe_open(path, framework="pt", device="cpu") as reader:
            metadata = reader.metadata()
            if metadata is None or "profile" not in metadata:
                raise ValueError(f"QSRT layer {layer} lacks its atoms profile")
            profile_name = str(metadata["profile"])
            formats, draws = unpack_atoms_v2_format_section(
                profile_name,
                reader.get_tensor("_qsrt_format_section"),
            )
        if draws is None or len(draws) != num_experts:
            raise ValueError(f"QSRT layer {layer} lacks coupled-Hadamard draws")
        if len(formats) != num_experts or any(value != "K2" for value in formats):
            raise ValueError(f"QSRT layer {layer} is not uniform K2")
        draws_by_layer[int(layer)] = tuple(int(value) for value in draws)
        profile_names.add(profile_name)
    if len(profile_names) != 1:
        raise ValueError("QSRT profile identity changes between decoder layers")
    return draws_by_layer, profile_names.pop(), _sha256(completion)


def _validate_workspace(
    workspace: KimiCotangentSlabWorkspace,
    *,
    boundaries: KimiBoundarySlabArchive,
    boundary_manifest_sha256: str,
    base_seed: int,
) -> None:
    residual_boundaries = tuple(
        range(0, boundaries.num_layers, boundaries.attn_res_block_size)
    )
    _require_values(
        workspace.manifest,
        {
            "token_count": boundaries.token_count,
            "hidden_dimension": boundaries.hidden_dimension,
            "num_layers": boundaries.num_layers,
            "residual_block_size": boundaries.attn_res_block_size,
            "residual_boundaries": list(residual_boundaries),
        },
        label="cotangent workspace",
    )
    if Path(str(workspace.manifest["boundary_archive"])).resolve() != boundaries.root:
        raise ValueError("cotangent workspace belongs to another boundary archive")
    provenance = workspace.manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("cotangent workspace provenance must be a JSON object")
    _require_values(
        provenance,
        {
            "purpose": "final-logit Fisher replay for coupled W1/W3",
            "boundary_manifest_sha256": boundary_manifest_sha256,
            "base_seed": int(base_seed),
        },
        label="cotangent provenance",
    )
    boundary = workspace.manifest.get("chain_boundary")
    valid = {None, boundaries.num_layers, *residual_boundaries}
    if boundary not in valid:
        raise ValueError(f"cotangent chain boundary is invalid: {boundary!r}")


def _validate_objective_workspace(
    workspace: KimiCotangentSlabWorkspace,
    *,
    boundaries: KimiBoundarySlabArchive,
    boundary_manifest_sha256: str,
    teacher_manifest_sha256: str,
    anchor_id: str,
    objective_id: str,
) -> None:
    residual_boundaries = tuple(
        range(0, boundaries.num_layers, boundaries.attn_res_block_size)
    )
    _require_values(
        workspace.manifest,
        {
            "token_count": boundaries.token_count,
            "hidden_dimension": boundaries.hidden_dimension,
            "num_layers": boundaries.num_layers,
            "residual_block_size": boundaries.attn_res_block_size,
            "residual_boundaries": list(residual_boundaries),
        },
        label="objective cotangent workspace",
    )
    if Path(str(workspace.manifest["boundary_archive"])).resolve() != boundaries.root:
        raise ValueError("objective cotangents belong to another anchor archive")
    provenance = workspace.manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("objective cotangent provenance must be a JSON object")
    _require_values(
        provenance,
        {
            "purpose": "deterministic final-logit KL gradient",
            "anchor_boundary_manifest_sha256": boundary_manifest_sha256,
            "teacher_boundary_manifest_sha256": teacher_manifest_sha256,
            "anchor_id": anchor_id,
            "objective_id": objective_id,
        },
        label="objective cotangent provenance",
    )
    boundary = workspace.manifest.get("chain_boundary")
    valid = {None, boundaries.num_layers, *residual_boundaries}
    if boundary not in valid:
        raise ValueError(f"objective cotangent chain boundary is invalid: {boundary!r}")


def _validate_factors(
    factors: KimiUpstreamFactorArchive,
    *,
    boundaries: KimiBoundarySlabArchive,
    workspace: KimiCotangentSlabWorkspace,
    objective_workspace: KimiCotangentSlabWorkspace | None,
    gradient_anchor_id: str | None,
    gradient_objective_id: str | None,
    layers: Sequence[int],
    num_experts: int,
    hidden_dimension: int,
    intermediate_dimension: int,
    block_size: int,
    gradient_rank: int,
    boundary_manifest_sha256: str,
    fisher_sample_sha256: str,
    weight_revision: str,
    profile_completion_sha256: str,
) -> None:
    _require_values(
        factors.manifest,
        {
            "num_layers": boundaries.num_layers,
            "num_experts": num_experts,
            "hidden_dimension": hidden_dimension,
            "intermediate_dimension": intermediate_dimension,
            "block_size": block_size,
            "gradient_rank": gradient_rank,
            "expected_layers": [int(value) for value in layers],
            "damping": "none",
        },
        label="upstream-factor archive",
    )
    provenance = factors.manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("upstream-factor provenance must be a JSON object")
    _require_values(
        provenance,
        {
            "purpose": "two-sided canonical-W1/W3 rounding",
            "weight_revision": weight_revision,
            "boundary_manifest_sha256": boundary_manifest_sha256,
            "cotangent_manifest": str(workspace.manifest_path),
            "fisher_sample_sha256": fisher_sample_sha256,
            "profile_completion_sha256": profile_completion_sha256,
            "gradient_anchor_id": gradient_anchor_id,
            "gradient_objective_id": gradient_objective_id,
            "objective_cotangent_manifest": (
                None
                if objective_workspace is None
                else str(objective_workspace.manifest_path)
            ),
            "damping": "none",
        },
        label="upstream-factor provenance",
    )
    completed = {
        str(value["operation"])
        for value in workspace.manifest.get("completed_operations", [])
    }
    uncommitted = [
        str(value.get("operation"))
        for value in factors.manifest.get("segments", [])
        if str(value.get("operation")) not in completed
    ]
    if uncommitted:
        raise ValueError(
            "upstream-factor segments have no matching cotangent commit: "
            f"{uncommitted}"
        )


def _preflight(
    *,
    boundaries: KimiBoundarySlabArchive,
    cotangent_path: Path,
    objective_cotangent_path: Path | None,
    factor_path: Path,
    factor_layers: int,
    num_experts: int,
    intermediate_dimension: int,
    block_size: int,
    devices: Sequence[torch.device],
    load_config: InstantTensorLoadConfig,
    filesystem_reserve_bytes: int,
) -> dict[str, object]:
    if len(devices) < boundaries.attn_res_block_size:
        raise ValueError("one CUDA device is required per residual-block layer")
    if any(
        device.index is None or device.index >= torch.cuda.device_count()
        for device in devices
    ):
        raise ValueError("a requested CUDA device is unavailable")
    reverse_devices = tuple(devices[: boundaries.attn_res_block_size])
    peer_links = []
    for left, right in zip(reverse_devices, reverse_devices[1:]):
        accessible = torch.cuda.can_device_access_peer(left.index, right.index)
        peer_links.append(
            {"from": left.index, "to": right.index, "accessible": accessible}
        )
        if not accessible:
            raise RuntimeError(f"CUDA peer access is unavailable from {left} to {right}")

    blocks = 2 * intermediate_dimension // block_size
    accumulator_bytes = num_experts * blocks * block_size * block_size * 4
    required_gpu_bytes = (70 << 30) + load_config.buffer_size + accumulator_bytes
    gpu_memory = []
    for device in reverse_devices:
        with torch.cuda.device(device):
            free, total = torch.cuda.mem_get_info(device)
        gpu_memory.append(
            {"device": str(device), "free_bytes": free, "total_bytes": total}
        )
        if free < required_gpu_bytes:
            raise RuntimeError(
                f"{device} has {free:,} free bytes, expected at least "
                f"{required_gpu_bytes:,}"
            )

    residual_count = len(
        range(0, boundaries.num_layers, boundaries.attn_res_block_size)
    )
    cotangent_upper_bound = 2 * (
        1 + residual_count
    ) * boundaries.expected_slab_bytes
    factor_bytes = factor_layers * accumulator_bytes
    storage_requirements = [
        (cotangent_path, cotangent_upper_bound),
        (factor_path, factor_bytes),
    ]
    if objective_cotangent_path is not None:
        storage_requirements.append(
            (objective_cotangent_path, cotangent_upper_bound)
        )
    filesystems: dict[str, dict[str, int]] = {}
    for path, expected_bytes in storage_requirements:
        root = _nearest_existing_parent(path.parent)
        entry = filesystems.setdefault(
            str(root),
            {
                "free_bytes": shutil.disk_usage(root).free,
                "expected_total_bytes": 0,
                "existing_bytes": 0,
                "additional_required_bytes": 0,
            },
        )
        existing = _stored_bytes(path)
        entry["expected_total_bytes"] += expected_bytes
        entry["existing_bytes"] += existing
        entry["additional_required_bytes"] += max(0, expected_bytes - existing)
    reserve = filesystem_reserve_bytes
    for root, values in filesystems.items():
        if values["free_bytes"] < values["additional_required_bytes"] + reserve:
            raise RuntimeError(
                f"capture requires {values['additional_required_bytes']:,} bytes plus "
                f"{reserve:,} bytes reserve on {root}, but only "
                f"{values['free_bytes']:,} bytes are free"
            )
    return {
        "devices": [str(value) for value in devices],
        "reverse_devices": [str(value) for value in reverse_devices],
        "peer_links": peer_links,
        "gpu_memory": gpu_memory,
        "accumulator_bytes_per_layer": accumulator_bytes,
        "cotangent_upper_bound_bytes": cotangent_upper_bound,
        "factor_bytes": factor_bytes,
        "filesystems": filesystems,
        "filesystem_reserve_bytes": reserve,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--cotangents", type=Path, required=True)
    parser.add_argument("--objective-cotangents", type=Path)
    parser.add_argument("--teacher-boundaries", type=Path)
    parser.add_argument("--gradient-anchor-id")
    parser.add_argument("--gradient-objective-id")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--weight-checkpoint", type=Path, default=DEFAULT_WEIGHT_CHECKPOINT)
    parser.add_argument("--code-checkpoint", type=Path, default=DEFAULT_CODE_CHECKPOINT)
    parser.add_argument(
        "--quantized-anchor-model",
        type=Path,
        help="served checkpoint supplying the exact non-expert anchor tensors",
    )
    parser.add_argument(
        "--quantized-anchor-candidate-pool",
        type=Path,
        help="sealed candidate pool supplying the anchor expert payload",
    )
    parser.add_argument(
        "--quantized-anchor-overlay-root",
        type=Path,
        action="append",
        default=[],
        help="ordered full-layer payload overlay root; later roots take precedence",
    )
    parser.add_argument(
        "--devices",
        type=_parse_devices,
        default=_parse_devices(",".join(str(index) for index in range(12))),
    )
    parser.add_argument("--base-seed", type=int, default=20260815)
    parser.add_argument("--factor-block-size", type=int, default=128)
    parser.add_argument("--gradient-sketch-rank", type=int, default=32)
    parser.add_argument("--lm-head-chunk-tokens", type=int, default=128)
    parser.add_argument("--slab-buffer-tokens", type=int, default=2048)
    parser.add_argument("--pipeline-queue-depth", type=int, default=1)
    parser.add_argument("--validation-documents", type=int, default=1)
    parser.add_argument("--instant-buffer-gib", type=int, default=4)
    parser.add_argument(
        "--instant-backend",
        type=_parse_instanttensor_backend,
        default=Backend.CUFILE,
    )
    parser.add_argument("--instant-chunk-mib", type=int, default=128)
    parser.add_argument("--instant-concurrency", type=int, default=4)
    parser.add_argument("--instant-io-depth", type=int, default=8)
    parser.add_argument("--filesystem-reserve-gib", type=int, default=128)
    parser.add_argument("--buffered-io", action="store_true")
    parser.add_argument("--stop-after-suffix", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if (
        args.factor_block_size <= 0
        or args.gradient_sketch_rank <= 0
        or args.lm_head_chunk_tokens <= 0
        or args.slab_buffer_tokens <= 0
        or args.pipeline_queue_depth <= 0
        or args.validation_documents < 0
        or args.instant_buffer_gib <= 0
        or args.instant_chunk_mib <= 0
        or args.instant_concurrency <= 0
        or args.instant_io_depth <= 0
        or args.filesystem_reserve_gib < 0
    ):
        raise ValueError("capture dimensions and buffer sizes must be positive")
    objective_values = (
        args.objective_cotangents,
        args.teacher_boundaries,
        args.gradient_anchor_id,
        args.gradient_objective_id,
    )
    if any(value is not None for value in objective_values) and not all(
        value is not None for value in objective_values
    ):
        raise ValueError(
            "objective capture requires cotangents, teacher boundaries, "
            "anchor identity, and objective identity"
        )
    capture_objective = args.objective_cotangents is not None
    if (args.quantized_anchor_model is None) != (
        args.quantized_anchor_candidate_pool is None
    ):
        raise ValueError(
            "quantized anchor model and candidate pool must be supplied together"
        )
    if args.quantized_anchor_overlay_root and args.quantized_anchor_model is None:
        raise ValueError("quantized anchor overlays require a quantized anchor model")
    if capture_objective and args.quantized_anchor_model is None:
        raise ValueError(
            "deterministic KL gradients require replay from an exact quantized anchor"
        )
    boundary_path = args.boundaries.expanduser().resolve()
    cotangent_path = args.cotangents.expanduser().resolve()
    objective_cotangent_path = (
        None
        if args.objective_cotangents is None
        else args.objective_cotangents.expanduser().resolve()
    )
    factor_path = args.dest.expanduser().resolve()
    profile_path = args.profile.expanduser().resolve()
    boundaries = KimiBoundarySlabArchive(boundary_path, require_complete=True)
    quantized_anchor = None
    if args.quantized_anchor_model is not None:
        quantized_anchor = {
            "model_checkpoint": str(
                args.quantized_anchor_model.expanduser().resolve()
            ),
            "candidate_pool": str(
                args.quantized_anchor_candidate_pool.expanduser().resolve()
            ),
            "overlay_roots": [
                str(path.expanduser().resolve())
                for path in args.quantized_anchor_overlay_root
            ],
        }
        boundary_provenance = boundaries.manifest.get("provenance")
        if not isinstance(boundary_provenance, dict):
            raise TypeError("boundary archive provenance must be a JSON object")
        if boundary_provenance.get("quantized_anchor") != quantized_anchor:
            raise ValueError(
                "boundary archive was not captured from the requested quantized anchor"
            )
    teacher_boundaries = (
        None
        if args.teacher_boundaries is None
        else KimiBoundarySlabArchive(
            args.teacher_boundaries.expanduser().resolve(),
            require_complete=True,
        )
    )
    runtime = load_official_kimi_runtime(
        weight_checkpoint=args.weight_checkpoint,
        code_checkpoint=args.code_checkpoint,
    )
    if (
        boundaries.num_layers != int(runtime.text_config.num_hidden_layers)
        or boundaries.hidden_dimension != int(runtime.text_config.hidden_size)
        or boundaries.attn_res_block_size
        != int(runtime.text_config.attn_res_block_size)
    ):
        raise ValueError("boundary archive and official model geometry disagree")
    routed_layers, num_experts, hidden_dimension, intermediate_dimension = (
        _routed_geometry(runtime)
    )
    if 2 * intermediate_dimension % args.factor_block_size:
        raise ValueError("factor block size does not divide coupled gate/up width")
    draws, atoms_profile, profile_completion_sha256 = _load_profile_draws(
        profile_path,
        layers=routed_layers,
        num_experts=num_experts,
    )
    load_config = InstantTensorLoadConfig(
        buffer_size=args.instant_buffer_gib << 30,
        chunk_size=args.instant_chunk_mib << 20,
        concurrency=args.instant_concurrency,
        io_depth=args.instant_io_depth,
        backend=args.instant_backend,
    )
    preflight = _preflight(
        boundaries=boundaries,
        cotangent_path=cotangent_path,
        objective_cotangent_path=objective_cotangent_path,
        factor_path=factor_path,
        factor_layers=0 if args.stop_after_suffix else len(routed_layers),
        num_experts=num_experts,
        intermediate_dimension=intermediate_dimension,
        block_size=args.factor_block_size,
        devices=args.devices,
        load_config=load_config,
        filesystem_reserve_bytes=args.filesystem_reserve_gib << 30,
    )
    boundary_manifest_sha256 = _sha256(boundaries.manifest_path)
    teacher_manifest_sha256 = (
        None
        if teacher_boundaries is None
        else _sha256(teacher_boundaries.manifest_path)
    )
    summary: dict[str, object] = {
        "boundary_archive": str(boundary_path),
        "boundary_manifest_sha256": boundary_manifest_sha256,
        "cotangent_workspace": str(cotangent_path),
        "objective_cotangent_workspace": (
            None
            if objective_cotangent_path is None
            else str(objective_cotangent_path)
        ),
        "teacher_boundary_archive": (
            None if teacher_boundaries is None else str(teacher_boundaries.root)
        ),
        "gradient_anchor_id": args.gradient_anchor_id,
        "gradient_objective_id": args.gradient_objective_id,
        "quantized_anchor": quantized_anchor,
        "upstream_factors": str(factor_path),
        "profile": str(profile_path),
        "profile_completion_sha256": profile_completion_sha256,
        "atoms_profile": atoms_profile,
        "weight_checkpoint": str(runtime.weight_checkpoint),
        "weight_revision": runtime.weight_checkpoint.name,
        "code_checkpoint": str(runtime.code_checkpoint),
        "code_revision": runtime.code_checkpoint.name,
        "documents": boundaries.load_documents().document_count,
        "tokens": boundaries.token_count,
        "decoder_layers": boundaries.num_layers,
        "routed_layers": list(routed_layers),
        "num_experts": num_experts,
        "expert_hidden_dimension": hidden_dimension,
        "intermediate_dimension": intermediate_dimension,
        "factor_block_size": args.factor_block_size,
        "gradient_sketch_rank": args.gradient_sketch_rank,
        "instanttensor": {
            "backend": load_config.backend.name.lower(),
            "buffer_bytes": load_config.buffer_size,
            "chunk_bytes": load_config.chunk_size,
            "concurrency": load_config.concurrency,
            "io_depth": load_config.io_depth,
            "whole_shard": load_config.whole_shard,
        },
        "preflight": preflight,
    }

    existing_workspace = None
    if cotangent_path.exists():
        existing_workspace = KimiCotangentSlabWorkspace(cotangent_path)
        _validate_workspace(
            existing_workspace,
            boundaries=boundaries,
            boundary_manifest_sha256=boundary_manifest_sha256,
            base_seed=args.base_seed,
        )
    existing_factors = None
    existing_objective_workspace = None
    if objective_cotangent_path is not None:
        assert teacher_manifest_sha256 is not None
        if objective_cotangent_path.exists():
            existing_objective_workspace = KimiCotangentSlabWorkspace(
                objective_cotangent_path
            )
            _validate_objective_workspace(
                existing_objective_workspace,
                boundaries=boundaries,
                boundary_manifest_sha256=boundary_manifest_sha256,
                teacher_manifest_sha256=teacher_manifest_sha256,
                anchor_id=args.gradient_anchor_id,
                objective_id=args.gradient_objective_id,
            )
    if factor_path.exists():
        if existing_workspace is None:
            raise ValueError(
                "an upstream-factor archive cannot resume without its cotangents"
            )
        sample_path = existing_workspace.root / "fisher-token-pairs.i32"
        if not sample_path.is_file():
            raise FileNotFoundError("upstream factors lack their Fisher sample identity")
        existing_factors = KimiUpstreamFactorArchive(factor_path)
        _validate_factors(
            existing_factors,
            boundaries=boundaries,
            workspace=existing_workspace,
            objective_workspace=existing_objective_workspace,
            gradient_anchor_id=args.gradient_anchor_id,
            gradient_objective_id=args.gradient_objective_id,
            layers=routed_layers,
            num_experts=num_experts,
            hidden_dimension=hidden_dimension,
            intermediate_dimension=intermediate_dimension,
            block_size=args.factor_block_size,
            gradient_rank=args.gradient_sketch_rank,
            boundary_manifest_sha256=boundary_manifest_sha256,
            fisher_sample_sha256=_sha256(sample_path),
            weight_revision=runtime.weight_checkpoint.name,
            profile_completion_sha256=profile_completion_sha256,
        )
    summary["resume_state"] = {
        "cotangent_workspace": existing_workspace is not None,
        "chain_boundary": (
            None
            if existing_workspace is None
            else existing_workspace.manifest.get("chain_boundary")
        ),
        "upstream_factor_archive": existing_factors is not None,
        "objective_cotangent_workspace": existing_objective_workspace is not None,
        "committed_factor_segments": (
            0
            if existing_factors is None
            else len(existing_factors.manifest.get("segments", []))
        ),
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    started = time.monotonic()
    workspace = existing_workspace or KimiCotangentSlabWorkspace.create(
        cotangent_path,
        boundary_archive=boundaries,
        provenance={
            "purpose": "final-logit Fisher replay for coupled W1/W3",
            "boundary_manifest_sha256": boundary_manifest_sha256,
            "base_seed": args.base_seed,
        },
    )
    _validate_workspace(
        workspace,
        boundaries=boundaries,
        boundary_manifest_sha256=boundary_manifest_sha256,
        base_seed=args.base_seed,
    )
    objective_workspace = existing_objective_workspace
    if objective_cotangent_path is not None and objective_workspace is None:
        assert teacher_manifest_sha256 is not None
        objective_workspace = KimiCotangentSlabWorkspace.create(
            objective_cotangent_path,
            boundary_archive=boundaries,
            provenance={
                "purpose": "deterministic final-logit KL gradient",
                "anchor_boundary_manifest_sha256": boundary_manifest_sha256,
                "teacher_boundary_manifest_sha256": teacher_manifest_sha256,
                "anchor_id": args.gradient_anchor_id,
                "objective_id": args.gradient_objective_id,
            },
        )
    if objective_workspace is not None:
        assert teacher_manifest_sha256 is not None
        _validate_objective_workspace(
            objective_workspace,
            boundaries=boundaries,
            boundary_manifest_sha256=boundary_manifest_sha256,
            teacher_manifest_sha256=teacher_manifest_sha256,
            anchor_id=args.gradient_anchor_id,
            objective_id=args.gradient_objective_id,
        )
    try:
        suffix_checkpoint = (
            runtime.weight_checkpoint
            if args.quantized_anchor_model is None
            else args.quantized_anchor_model.expanduser().resolve()
        )
        if workspace.manifest.get("chain_boundary") is None:
            suffix_record: object = _jsonable(
                KimiSuffixPipeline(
                    checkpoint=suffix_checkpoint,
                    boundary_archive=boundaries,
                    workspace=workspace,
                    objective_workspace=objective_workspace,
                    teacher_boundary_archive=teacher_boundaries,
                    devices=args.devices,
                    epsilon=float(runtime.text_config.rms_norm_eps),
                    vocabulary_size=int(runtime.text_config.vocab_size),
                    base_seed=args.base_seed,
                    lm_head_chunk_tokens=args.lm_head_chunk_tokens,
                    slab_buffer_tokens=args.slab_buffer_tokens,
                    direct_io=not args.buffered_io,
                    load_config=load_config,
                ).run()
            )
        else:
            suffix_record = {
                "resumed": True,
                "chain_boundary": workspace.manifest["chain_boundary"],
            }
        sample_path = workspace.root / "fisher-token-pairs.i32"
        fisher_sample_sha256 = _sha256(sample_path)
        if args.stop_after_suffix:
            record = summary | {
                "complete": True,
                "elapsed_seconds": time.monotonic() - started,
                "suffix": suffix_record,
                "cotangent_manifest_sha256": _sha256(workspace.manifest_path),
                "objective_cotangent_manifest_sha256": (
                    None
                    if objective_workspace is None
                    else _sha256(objective_workspace.manifest_path)
                ),
            }
            _atomic_json(cotangent_path / RUN_FILENAME, record)
            print(json.dumps(record, indent=2))
            return
        factors = existing_factors or KimiUpstreamFactorArchive.create(
            factor_path,
            num_layers=boundaries.num_layers,
            num_experts=num_experts,
            hidden_dimension=hidden_dimension,
            intermediate_dimension=intermediate_dimension,
            block_size=args.factor_block_size,
            gradient_rank=args.gradient_sketch_rank,
            expected_layers=routed_layers,
            provenance={
                "purpose": "two-sided canonical-W1/W3 rounding",
                "semantic_point": (
                    "gate/up preactivation gradient after the expert-static "
                    "coupled output transform"
                ),
                "weight_revision": runtime.weight_checkpoint.name,
                "profile": str(profile_path),
                "profile_completion_sha256": profile_completion_sha256,
                "gradient_anchor_id": args.gradient_anchor_id,
                "gradient_objective_id": args.gradient_objective_id,
                "objective_cotangent_manifest": (
                    None
                    if objective_workspace is None
                    else str(objective_workspace.manifest_path)
                ),
                "atoms_profile": atoms_profile,
                "boundary_manifest_sha256": boundary_manifest_sha256,
                "cotangent_manifest": str(workspace.manifest_path),
                "fisher_sample_sha256": fisher_sample_sha256,
                "torch_version": torch.__version__,
                "instanttensor_version": importlib.metadata.version("instanttensor"),
                "damping": "none",
            },
        )
        _validate_factors(
            factors,
            boundaries=boundaries,
            workspace=workspace,
            objective_workspace=objective_workspace,
            gradient_anchor_id=args.gradient_anchor_id,
            gradient_objective_id=args.gradient_objective_id,
            layers=routed_layers,
            num_experts=num_experts,
            hidden_dimension=hidden_dimension,
            intermediate_dimension=intermediate_dimension,
            block_size=args.factor_block_size,
            gradient_rank=args.gradient_sketch_rank,
            boundary_manifest_sha256=boundary_manifest_sha256,
            fisher_sample_sha256=fisher_sample_sha256,
            weight_revision=runtime.weight_checkpoint.name,
            profile_completion_sha256=profile_completion_sha256,
        )
        if args.quantized_anchor_model is None:
            reverse_adapter = OfficialKimiForwardAdapter(
                runtime,
                load_config=load_config,
                validate_outputs=False,
                grouped_expert_dispatch=True,
            )
        else:
            reverse_adapter = QSRTKimiForwardAdapter(
                runtime,
                model_checkpoint=args.quantized_anchor_model,
                expert_payload=QSRTAnchorPayload(
                    args.quantized_anchor_candidate_pool,
                    overlay_roots=args.quantized_anchor_overlay_root,
                ),
                load_config=load_config,
                validate_outputs=False,
            )
        reverse = KimiPipelinedUpstreamReverse(
            adapter=reverse_adapter,
            boundary_archive=boundaries,
            cotangent_workspace=workspace,
            objective_workspace=objective_workspace,
            upstream_factors=factors,
            intermediate_draws=draws,
            devices=args.devices[: boundaries.attn_res_block_size],
            queue_depth=args.pipeline_queue_depth,
            slab_buffer_tokens=args.slab_buffer_tokens,
            direct_io=not args.buffered_io,
            validation_documents=args.validation_documents,
            gradient_sketch_seed=args.base_seed,
        ).run()
        record = summary | {
            "complete": True,
            "elapsed_seconds": time.monotonic() - started,
            "suffix": suffix_record,
            "reverse": _jsonable(reverse),
            "cotangent_manifest_sha256": _sha256(workspace.manifest_path),
            "upstream_factor_manifest_sha256": _sha256(factors.manifest_path),
        }
        _atomic_json(factor_path / RUN_FILENAME, record)
    except BaseException as error:
        record = summary | {
            "complete": False,
            "elapsed_seconds": time.monotonic() - started,
            "error": f"{type(error).__name__}: {error}",
        }
        failure_path = (
            factor_path / FAILURE_FILENAME
            if factor_path.is_dir()
            else cotangent_path / FAILURE_FILENAME
        )
        _atomic_json(failure_path, record)
        raise
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

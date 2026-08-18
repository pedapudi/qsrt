#!/usr/bin/env python
"""Capture final-logit Fisher factors for every Kimi-K3 routed MoE layer."""

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
from typing import Any, Mapping, Sequence

import torch

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
from qsrt.kimi_output_factors import KimiOutputFactorArchive
from qsrt.kimi_reverse_pipeline import KimiReversePipeline
from qsrt.kimi_suffix_pipeline import KimiSuffixPipeline


DEFAULT_WEIGHT_CHECKPOINT = Path(
    "/data/cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
)
DEFAULT_CODE_CHECKPOINT = Path(
    "/data/cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721"
)
RUN_FILENAME = "output-fisher-run.json"
FAILURE_FILENAME = "output-fisher-failure.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
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


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(path)
        candidate = candidate.parent
    return candidate


def _stored_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


def _routed_layer_geometry(runtime) -> tuple[tuple[int, ...], int]:
    layers: list[int] = []
    output_dimensions: set[int] = set()
    for layer in range(int(runtime.text_config.num_hidden_layers)):
        module = new_meta_decoder_layer(runtime, layer)
        block = getattr(module, "block_sparse_moe", None)
        if block is None:
            continue
        layers.append(layer)
        parameter = dict(module.named_parameters()).get(
            "block_sparse_moe.experts.0.w2.weight"
        )
        if parameter is None or parameter.ndim != 2:
            raise ValueError(f"decoder layer {layer} has no routed W2 matrix")
        output_dimensions.add(int(parameter.shape[0]))
    if not layers or len(output_dimensions) != 1:
        raise ValueError("routed MoE output geometry is inconsistent")
    return tuple(layers), output_dimensions.pop()


def _resolved_manifest_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} is missing from the stored capture identity")
    return Path(value).expanduser().resolve()


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


def _validate_workspace_identity(
    workspace: KimiCotangentSlabWorkspace,
    *,
    boundaries: KimiBoundarySlabArchive,
    boundary_manifest_sha256: str,
    base_seed: int,
) -> None:
    manifest = workspace.manifest
    residual_boundaries = tuple(
        range(
            0,
            boundaries.num_layers,
            boundaries.attn_res_block_size,
        )
    )
    _require_values(
        manifest,
        {
            "token_count": boundaries.token_count,
            "hidden_dimension": boundaries.hidden_dimension,
            "num_layers": boundaries.num_layers,
            "residual_block_size": boundaries.attn_res_block_size,
            "residual_boundaries": list(residual_boundaries),
        },
        label="cotangent workspace",
    )
    if _resolved_manifest_path(
        manifest.get("boundary_archive"),
        field="cotangent boundary_archive",
    ) != boundaries.root:
        raise ValueError("cotangent workspace belongs to a different boundary archive")
    if _resolved_manifest_path(
        manifest.get("boundary_manifest"),
        field="cotangent boundary_manifest",
    ) != boundaries.manifest_path:
        raise ValueError("cotangent workspace records a different boundary manifest")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("cotangent workspace provenance must be a JSON object")
    _require_values(
        provenance,
        {
            "purpose": "final-logit empirical Fisher reverse replay",
            "boundary_manifest_sha256": boundary_manifest_sha256,
            "base_seed": int(base_seed),
        },
        label="cotangent provenance",
    )
    chain_boundary = manifest.get("chain_boundary")
    valid_boundaries = {
        None,
        boundaries.num_layers,
        *residual_boundaries,
    }
    if chain_boundary not in valid_boundaries:
        raise ValueError(f"cotangent chain boundary is invalid: {chain_boundary!r}")
    operations = manifest.get("completed_operations")
    if not isinstance(operations, list):
        raise TypeError("cotangent completed_operations must be a JSON list")
    if chain_boundary is None:
        if operations:
            raise ValueError("uninitialized cotangent workspace contains committed operations")
    elif not operations or int(operations[-1]["chain_boundary"]) != chain_boundary:
        raise ValueError("cotangent chain boundary differs from its final committed operation")


def _validate_factor_identity(
    factors: KimiOutputFactorArchive,
    *,
    boundaries: KimiBoundarySlabArchive,
    workspace: KimiCotangentSlabWorkspace,
    expected_layers: Sequence[int],
    dimension: int,
    boundary_manifest_sha256: str,
    fisher_sample_sha256: str,
    weight_revision: str,
) -> None:
    _require_values(
        factors.manifest,
        {
            "num_layers": boundaries.num_layers,
            "dimension": int(dimension),
            "expected_layers": [int(value) for value in expected_layers],
            "damping": "none",
        },
        label="output-factor archive",
    )
    provenance = factors.manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("output-factor provenance must be a JSON object")
    _require_values(
        provenance,
        {
            "purpose": "two-sided canonical-W2 rounding",
            "weight_revision": weight_revision,
            "boundary_manifest_sha256": boundary_manifest_sha256,
            "cotangent_manifest": str(workspace.manifest_path),
            "fisher_sample_sha256": fisher_sample_sha256,
            "damping": "none",
        },
        label="output-factor provenance",
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
            "output-factor segments have no matching cotangent commit: "
            f"{uncommitted}"
        )


def _preflight(
    *,
    boundaries: KimiBoundarySlabArchive,
    cotangent_path: Path,
    factor_path: Path,
    factor_dimension: int,
    factor_layers: int,
    devices: tuple[torch.device, ...],
    load_config: InstantTensorLoadConfig,
) -> dict[str, object]:
    if len(devices) < boundaries.attn_res_block_size:
        raise ValueError(
            "one CUDA device is required for every layer in an attention-residual block"
        )
    if any(
        device.index is None or device.index >= torch.cuda.device_count()
        for device in devices
    ):
        raise ValueError("a requested CUDA device is unavailable")
    reverse_devices = devices[: boundaries.attn_res_block_size]
    peer_links = []
    for left, right in zip(reverse_devices, reverse_devices[1:]):
        accessible = torch.cuda.can_device_access_peer(left.index, right.index)
        peer_links.append(
            {"from": left.index, "to": right.index, "accessible": accessible}
        )
        if not accessible:
            raise RuntimeError(f"CUDA peer access is unavailable from {left} to {right}")

    required_gpu_bytes = (70 << 30) + load_config.buffer_size
    gpu_memory = []
    for device in devices:
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

    residual_count = len(range(
        0,
        boundaries.num_layers,
        boundaries.attn_res_block_size,
    ))
    cotangent_upper_bound = (
        2 * (1 + residual_count) * boundaries.expected_slab_bytes
    )
    factor_bytes = factor_layers * 3 * factor_dimension * factor_dimension * 4
    filesystems: dict[str, dict[str, int]] = {}
    for path, expected_bytes in (
        (cotangent_path, cotangent_upper_bound),
        (factor_path, factor_bytes),
    ):
        root = _nearest_existing_parent(path.parent)
        free = shutil.disk_usage(root).free
        key = str(root)
        entry = filesystems.setdefault(
            key,
            {
                "free_bytes": free,
                "expected_total_bytes": 0,
                "existing_bytes": 0,
                "additional_required_bytes": 0,
            },
        )
        existing_bytes = _stored_bytes(path)
        entry["expected_total_bytes"] += expected_bytes
        entry["existing_bytes"] += existing_bytes
        entry["additional_required_bytes"] += max(
            0,
            expected_bytes - existing_bytes,
        )
    reserve = 128 << 30
    for root, value in filesystems.items():
        if value["free_bytes"] < value["additional_required_bytes"] + reserve:
            raise RuntimeError(
                f"capture requires {value['additional_required_bytes']:,} "
                f"additional bytes plus "
                f"{reserve:,} bytes reserve on {root}, but only "
                f"{value['free_bytes']:,} bytes are free"
            )
    return {
        "devices": [str(value) for value in devices],
        "reverse_devices": [str(value) for value in reverse_devices],
        "peer_links": peer_links,
        "gpu_memory": gpu_memory,
        "cotangent_upper_bound_bytes": cotangent_upper_bound,
        "factor_bytes": factor_bytes,
        "filesystems": filesystems,
        "filesystem_reserve_bytes": reserve,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--cotangents", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--weight-checkpoint", type=Path, default=DEFAULT_WEIGHT_CHECKPOINT)
    parser.add_argument("--code-checkpoint", type=Path, default=DEFAULT_CODE_CHECKPOINT)
    parser.add_argument(
        "--devices",
        type=_parse_devices,
        default=_parse_devices(",".join(str(index) for index in range(12))),
    )
    parser.add_argument("--base-seed", type=int, default=20260815)
    parser.add_argument("--lm-head-chunk-tokens", type=int, default=128)
    parser.add_argument("--slab-buffer-tokens", type=int, default=256)
    parser.add_argument("--instant-buffer-gib", type=int, default=4)
    parser.add_argument("--instant-chunk-mib", type=int, default=16)
    parser.add_argument("--instant-io-depth", type=int, default=256)
    parser.add_argument("--buffered-io", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.lm_head_chunk_tokens <= 0 or args.slab_buffer_tokens <= 0:
        raise ValueError("capture buffer sizes must be positive")
    boundary_path = args.boundaries.expanduser().resolve()
    cotangent_path = args.cotangents.expanduser().resolve()
    factor_path = args.dest.expanduser().resolve()
    boundaries = KimiBoundarySlabArchive(boundary_path, require_complete=True)
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
    routed_layers, factor_dimension = _routed_layer_geometry(runtime)
    load_config = InstantTensorLoadConfig(
        buffer_size=args.instant_buffer_gib << 30,
        chunk_size=args.instant_chunk_mib << 20,
        concurrency=1,
        io_depth=args.instant_io_depth,
    )
    preflight = _preflight(
        boundaries=boundaries,
        cotangent_path=cotangent_path,
        factor_path=factor_path,
        factor_dimension=factor_dimension,
        factor_layers=len(routed_layers),
        devices=args.devices,
        load_config=load_config,
    )
    boundary_manifest_sha256 = _sha256(boundaries.manifest_path)
    summary: dict[str, object] = {
        "boundary_archive": str(boundary_path),
        "boundary_manifest_sha256": boundary_manifest_sha256,
        "cotangent_workspace": str(cotangent_path),
        "output_factors": str(factor_path),
        "weight_checkpoint": str(runtime.weight_checkpoint),
        "weight_revision": runtime.weight_checkpoint.name,
        "code_checkpoint": str(runtime.code_checkpoint),
        "code_revision": runtime.code_checkpoint.name,
        "documents": boundaries.load_documents().document_count,
        "tokens": boundaries.token_count,
        "decoder_layers": boundaries.num_layers,
        "routed_layers": list(routed_layers),
        "routed_output_dimension": factor_dimension,
        "preflight": preflight,
    }
    existing_workspace: KimiCotangentSlabWorkspace | None = None
    existing_factors: KimiOutputFactorArchive | None = None
    if cotangent_path.exists():
        existing_workspace = KimiCotangentSlabWorkspace(cotangent_path)
        _validate_workspace_identity(
            existing_workspace,
            boundaries=boundaries,
            boundary_manifest_sha256=boundary_manifest_sha256,
            base_seed=args.base_seed,
        )
    if factor_path.exists():
        if existing_workspace is None:
            raise ValueError(
                "an output-factor archive cannot be resumed without its "
                "cotangent workspace"
            )
        sample_path = existing_workspace.root / "fisher-token-pairs.i32"
        if not sample_path.is_file():
            raise FileNotFoundError(
                "output-factor archive has no Fisher sample identity: "
                f"{sample_path}"
            )
        existing_factors = KimiOutputFactorArchive(factor_path)
        _validate_factor_identity(
            existing_factors,
            boundaries=boundaries,
            workspace=existing_workspace,
            expected_layers=routed_layers,
            dimension=factor_dimension,
            boundary_manifest_sha256=boundary_manifest_sha256,
            fisher_sample_sha256=_sha256(sample_path),
            weight_revision=runtime.weight_checkpoint.name,
        )
    summary["resume_state"] = {
        "cotangent_workspace": existing_workspace is not None,
        "chain_boundary": (
            None
            if existing_workspace is None
            else existing_workspace.manifest.get("chain_boundary")
        ),
        "output_factor_archive": existing_factors is not None,
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
    if existing_workspace is not None:
        workspace = existing_workspace
    else:
        workspace = KimiCotangentSlabWorkspace.create(
            cotangent_path,
            boundary_archive=boundaries,
            provenance={
                "purpose": "final-logit empirical Fisher reverse replay",
                "boundary_manifest_sha256": boundary_manifest_sha256,
                "base_seed": args.base_seed,
            },
        )
    _validate_workspace_identity(
        workspace,
        boundaries=boundaries,
        boundary_manifest_sha256=boundary_manifest_sha256,
        base_seed=args.base_seed,
    )
    try:
        suffix_record: object
        if workspace.manifest.get("chain_boundary") is None:
            suffix = KimiSuffixPipeline(
                checkpoint=runtime.weight_checkpoint,
                boundary_archive=boundaries,
                workspace=workspace,
                devices=args.devices,
                epsilon=float(runtime.text_config.rms_norm_eps),
                vocabulary_size=int(runtime.text_config.vocab_size),
                base_seed=args.base_seed,
                lm_head_chunk_tokens=args.lm_head_chunk_tokens,
                slab_buffer_tokens=args.slab_buffer_tokens,
                direct_io=not args.buffered_io,
                load_config=load_config,
            ).run()
            suffix_record = _jsonable(suffix)
        else:
            suffix_record = {
                "resumed": True,
                "chain_boundary": workspace.manifest["chain_boundary"],
            }

        sample_path = workspace.root / "fisher-token-pairs.i32"
        fisher_sample_sha256 = _sha256(sample_path)
        if existing_factors is not None:
            factors = existing_factors
        else:
            factors = KimiOutputFactorArchive.create(
                factor_path,
                num_layers=boundaries.num_layers,
                dimension=factor_dimension,
                expected_layers=routed_layers,
                provenance={
                    "purpose": "two-sided canonical-W2 rounding",
                    "semantic_point": (
                        "gradient at the route-weighted expert W2 sum before "
                        "routed latent RMSNorm and output projection"
                    ),
                    "weight_revision": runtime.weight_checkpoint.name,
                    "boundary_manifest_sha256": boundary_manifest_sha256,
                    "cotangent_manifest": str(workspace.manifest_path),
                    "fisher_sample_sha256": fisher_sample_sha256,
                    "torch_version": torch.__version__,
                    "instanttensor_version": importlib.metadata.version("instanttensor"),
                    "damping": "none",
                },
            )
        _validate_factor_identity(
            factors,
            boundaries=boundaries,
            workspace=workspace,
            expected_layers=routed_layers,
            dimension=factor_dimension,
            boundary_manifest_sha256=boundary_manifest_sha256,
            fisher_sample_sha256=fisher_sample_sha256,
            weight_revision=runtime.weight_checkpoint.name,
        )
        reverse = KimiReversePipeline(
            adapter=OfficialKimiForwardAdapter(runtime, load_config=load_config),
            boundary_archive=boundaries,
            cotangent_workspace=workspace,
            output_factors=factors,
            devices=args.devices[: boundaries.attn_res_block_size],
            slab_buffer_tokens=args.slab_buffer_tokens,
            direct_io=not args.buffered_io,
        ).run()
        record = summary | {
            "complete": True,
            "elapsed_seconds": time.monotonic() - started,
            "suffix": suffix_record,
            "reverse": _jsonable(reverse),
            "cotangent_manifest_sha256": _sha256(workspace.manifest_path),
            "output_factor_manifest_sha256": _sha256(factors.manifest_path),
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

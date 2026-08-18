#!/usr/bin/env python
"""Capture every official Kimi-K3 decoder boundary for Fisher replay."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

_REPOSITORY = Path(__file__).resolve().parents[1]
if str(_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY))

from qsrt.instanttensor_kimi import InstantTensorLoadConfig
from qsrt.kimi_boundary_slabs import KimiBoundarySlabArchive
from qsrt.kimi_capture_documents import (
    load_corpus_document_index,
    load_token_suite_document_index,
)
from qsrt.kimi_forward_pipeline import KimiForwardPipeline
from qsrt.kimi_official_forward import (
    OfficialKimiEmbeddingInputs,
    OfficialKimiForwardAdapter,
    load_official_kimi_runtime,
)
from qsrt.kimi_quantized_forward import QSRTAnchorPayload, QSRTKimiForwardAdapter
from qsrt.kimi_routes import KimiRouteArchive, RouteCapturingAdapter


DEFAULT_WEIGHT_CHECKPOINT = Path(
    "/data/cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
)
DEFAULT_CODE_CHECKPOINT = Path(
    "/data/cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721"
)
DEFAULT_CORPUS_REPORT = Path(
    "/data/datasets/kquant/captures/k3-all-routed-4m-v1-corpus.json"
)
RUN_FILENAME = "forward-run.json"
FAILURE_FILENAME = "forward-failure.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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
        raise argparse.ArgumentTypeError("devices must be comma-separated integers") from error
    if not indices or len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("devices must be nonempty and unique")
    return tuple(torch.device("cuda", index) for index in indices)


def _parse_indices(value: str) -> tuple[int, ...]:
    result: list[int] = []
    try:
        for field in value.split(","):
            field = field.strip()
            if not field:
                continue
            if "-" in field:
                first_text, end_text = field.split("-", 1)
                first, end = int(first_text), int(end_text)
                if end < first:
                    raise ValueError
                result.extend(range(first, end + 1))
            else:
                result.append(int(field))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "context indices must be comma-separated integers or inclusive ranges"
        ) from error
    if not result or len(set(result)) != len(result) or min(result) < 0:
        raise argparse.ArgumentTypeError("context indices must be nonnegative and unique")
    return tuple(result)


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(path)
        candidate = candidate.parent
    return candidate


def _tokenizer_hashes(root: Path) -> dict[str, str]:
    names = (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    )
    return {name: _sha256(root / name) for name in names if (root / name).is_file()}


def _preflight(
    *,
    destination: Path,
    devices: tuple[torch.device, ...],
    token_count: int,
    hidden_dimension: int,
    boundary_count: int,
    load_config: InstantTensorLoadConfig,
    resume: bool,
) -> dict[str, object]:
    if destination.exists() and not resume:
        raise FileExistsError(f"capture destination already exists: {destination}")
    if not destination.exists() and resume:
        raise FileNotFoundError(f"capture destination does not exist: {destination}")
    if any(device.index is None or device.index >= torch.cuda.device_count() for device in devices):
        raise ValueError("a requested CUDA device is unavailable")
    peer_links = []
    for left, right in zip(devices, devices[1:]):
        accessible = torch.cuda.can_device_access_peer(left.index, right.index)
        peer_links.append({"from": left.index, "to": right.index, "accessible": accessible})
        if not accessible:
            raise RuntimeError(f"CUDA peer access is unavailable from {left} to {right}")

    required_gpu_bytes = (66 << 30) + load_config.buffer_size
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

    slab_bytes = token_count * hidden_dimension * torch.bfloat16.itemsize
    archive_bytes = boundary_count * slab_bytes
    filesystem = _nearest_existing_parent(destination.parent)
    free_disk = shutil.disk_usage(filesystem).free
    reserve = 64 << 30
    additional_archive_bytes = 0 if resume else archive_bytes
    if free_disk < additional_archive_bytes + reserve:
        raise RuntimeError(
            f"capture requires {additional_archive_bytes:,} additional bytes plus "
            f"{reserve:,} bytes "
            f"reserve, but {filesystem} has {free_disk:,} bytes free"
        )
    return {
        "devices": [str(device) for device in devices],
        "peer_links": peer_links,
        "gpu_memory": gpu_memory,
        "slab_bytes": slab_bytes,
        "archive_bytes": archive_bytes,
        "additional_archive_bytes": additional_archive_bytes,
        "filesystem": str(filesystem),
        "filesystem_free_bytes": free_disk,
        "filesystem_reserve_bytes": reserve,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--corpus-report", type=Path)
    parser.add_argument(
        "--token-suite",
        type=Path,
        help="distribution-fidelity suite whose stored token IDs are replayed directly",
    )
    parser.add_argument(
        "--token-suite-contexts",
        type=_parse_indices,
        help="context IDs from --token-suite, such as 0-31 or 0,2,7",
    )
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
        "--tokenizer",
        type=Path,
        help="tokenizer directory; defaults to model_dir in the corpus report",
    )
    parser.add_argument("--devices", type=_parse_devices, default=_parse_devices(",".join(str(i) for i in range(12))))
    parser.add_argument("--queue-depth", type=int, default=2)
    parser.add_argument("--slab-buffer-tokens", type=int, default=2048)
    parser.add_argument("--instant-buffer-gib", type=int, default=4)
    parser.add_argument("--instant-chunk-mib", type=int, default=16)
    parser.add_argument("--instant-io-depth", type=int, default=256)
    parser.add_argument("--buffered-io", action="store_true")
    parser.add_argument(
        "--route-dest",
        type=Path,
        help="record every layer's exact top-k expert selections",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from the last contiguous sealed decoder boundary",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.queue_depth <= 0 or args.slab_buffer_tokens <= 0:
        raise ValueError("queue depth and slab buffer tokens must be positive")
    if (args.quantized_anchor_model is None) != (
        args.quantized_anchor_candidate_pool is None
    ):
        raise ValueError(
            "quantized anchor model and candidate pool must be supplied together"
        )
    if args.quantized_anchor_overlay_root and args.quantized_anchor_model is None:
        raise ValueError("quantized anchor overlays require a quantized anchor model")
    if args.corpus_report is not None and args.token_suite is not None:
        raise ValueError("corpus report and token suite are mutually exclusive")
    if args.token_suite is None and args.token_suite_contexts is not None:
        raise ValueError("token-suite contexts require --token-suite")
    if args.token_suite is not None and args.token_suite_contexts is None:
        raise ValueError("--token-suite requires --token-suite-contexts")
    report_path: Path | None = None
    report: dict[str, Any] | None = None
    tokenizer_root: Path | None = None
    tokenizer_output = io.StringIO()
    if args.token_suite is not None:
        token_suite = args.token_suite.expanduser().resolve()
        documents = load_token_suite_document_index(
            token_suite,
            args.token_suite_contexts,
        )
        population_provenance: dict[str, object] = {
            "kind": "distribution-fidelity stored-token subset",
            "suite": str(token_suite),
            "suite_manifest_sha256": _sha256(token_suite / "suite-manifest.json"),
            "context_indices": list(args.token_suite_contexts),
        }
    else:
        report_path = (
            args.corpus_report or DEFAULT_CORPUS_REPORT
        ).expanduser().resolve()
        report = json.loads(report_path.read_text())
        tokenizer_root = (
            args.tokenizer.expanduser().resolve()
            if args.tokenizer is not None
            else Path(str(report["model_dir"])).expanduser().resolve()
        )
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError("run this capture with the vLLM Python environment") from error
        with (
            contextlib.redirect_stdout(tokenizer_output),
            contextlib.redirect_stderr(tokenizer_output),
        ):
            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_root), trust_remote_code=True
            )
            documents = load_corpus_document_index(report_path, tokenizer)
        population_provenance = {
            "kind": "authenticated corpus report",
            "corpus_report": str(report_path),
            "corpus_report_sha256": _sha256(report_path),
            "corpus_plan_sha256": report["plan_sha256"],
            "tokenizer": str(tokenizer_root),
            "tokenizer_hashes": _tokenizer_hashes(tokenizer_root),
            "tokenizer_diagnostic_lines": len(tokenizer_output.getvalue().splitlines()),
        }

    runtime = load_official_kimi_runtime(
        weight_checkpoint=args.weight_checkpoint,
        code_checkpoint=args.code_checkpoint,
    )
    num_layers = int(runtime.text_config.num_hidden_layers)
    hidden_dimension = int(runtime.text_config.hidden_size)
    attn_res_block_size = int(runtime.text_config.attn_res_block_size)
    load_config = InstantTensorLoadConfig(
        buffer_size=args.instant_buffer_gib << 30,
        chunk_size=args.instant_chunk_mib << 20,
        concurrency=1,
        io_depth=args.instant_io_depth,
    )
    destination = args.dest.expanduser().resolve()
    preflight = _preflight(
        destination=destination,
        devices=args.devices,
        token_count=documents.token_count,
        hidden_dimension=hidden_dimension,
        boundary_count=num_layers + 1,
        load_config=load_config,
        resume=args.resume,
    )
    provenance = {
        "purpose": "final-output Fisher and Kronecker-factor replay",
        "weight_checkpoint": str(runtime.weight_checkpoint),
        "weight_revision": runtime.weight_checkpoint.name,
        "weight_config_sha256": _sha256(runtime.weight_checkpoint / "config.json"),
        "weight_index_sha256": _sha256(
            runtime.weight_checkpoint / "model.safetensors.index.json"
        ),
        "code_checkpoint": str(runtime.code_checkpoint),
        "code_revision": runtime.code_checkpoint.name,
        "code_config_sha256": _sha256(runtime.code_checkpoint / "config.json"),
        "population": population_provenance,
        "torch_version": torch.__version__,
        "instanttensor_version": importlib.metadata.version("instanttensor"),
        "load_config": {
            "buffer_size": load_config.buffer_size,
            "chunk_size": load_config.chunk_size,
            "concurrency": load_config.concurrency,
            "io_depth": load_config.io_depth,
            "backend": str(load_config.backend),
        },
        "preflight": preflight,
    }
    if args.quantized_anchor_model is not None:
        provenance["quantized_anchor"] = {
            "model_checkpoint": str(args.quantized_anchor_model.resolve()),
            "candidate_pool": str(args.quantized_anchor_candidate_pool.resolve()),
            "overlay_roots": [
                str(path.resolve()) for path in args.quantized_anchor_overlay_root
            ],
        }
    summary = {
        "destination": str(destination),
        "documents": documents.document_count,
        "tokens": documents.token_count,
        "num_layers": num_layers,
        "hidden_dimension": hidden_dimension,
        "preflight": preflight,
        "resume_requested": args.resume,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    started = time.monotonic()
    if args.resume:
        archive = KimiBoundarySlabArchive(destination)
        if archive.complete:
            raise ValueError(f"capture archive is already complete: {destination}")
        if (
            archive.num_layers != num_layers
            or archive.hidden_dimension != hidden_dimension
            or archive.attn_res_block_size != attn_res_block_size
        ):
            raise ValueError("capture archive geometry does not match the official model")
        stored_documents = archive.load_documents()
        if (
            not torch.equal(stored_documents.input_ids, documents.input_ids)
            or not torch.equal(stored_documents.offsets, documents.offsets)
            or stored_documents.identifiers != documents.identifiers
        ):
            raise ValueError("capture archive documents do not match the corpus report")
        stored_provenance = dict(archive.manifest.get("provenance", {}))
        expected_provenance = dict(provenance)
        stored_provenance.pop("preflight", None)
        expected_provenance.pop("preflight", None)
        if stored_provenance != expected_provenance:
            raise ValueError("capture archive provenance does not match this invocation")
        sealed_prefix = archive.sealed_boundary_prefix()
        discarded_boundaries = archive.discard_unsealed_receipts()
        start_layer = sealed_prefix[-1] if sealed_prefix else 0
        use_archived_input = bool(sealed_prefix)
    else:
        archive = KimiBoundarySlabArchive.create(
            destination,
            documents=documents,
            num_layers=num_layers,
            hidden_dimension=hidden_dimension,
            attn_res_block_size=attn_res_block_size,
            provenance=provenance,
        )
        sealed_prefix = ()
        discarded_boundaries = ()
        start_layer = 0
        use_archived_input = False
    if args.quantized_anchor_model is None:
        adapter = OfficialKimiForwardAdapter(runtime, load_config=load_config)
    else:
        adapter = QSRTKimiForwardAdapter(
            runtime,
            model_checkpoint=args.quantized_anchor_model,
            expert_payload=QSRTAnchorPayload(
                args.quantized_anchor_candidate_pool,
                overlay_roots=args.quantized_anchor_overlay_root,
            ),
            load_config=load_config,
        )
    route_archive: KimiRouteArchive | None = None
    if args.route_dest is not None:
        route_archive = KimiRouteArchive.create(
            args.route_dest,
            token_count=documents.token_count,
            num_layers=num_layers,
            num_experts=int(runtime.text_config.num_experts),
            top_k=int(runtime.text_config.num_experts_per_token),
            provenance={
                "weight_checkpoint": str(runtime.weight_checkpoint),
                "capture_destination": str(destination),
                "population": population_provenance,
                "quantized_anchor": provenance.get("quantized_anchor"),
            },
        )
        adapter = RouteCapturingAdapter(adapter, route_archive)
    inputs = None
    if not use_archived_input:
        inputs = OfficialKimiEmbeddingInputs(
            runtime=runtime,
            documents=documents,
            device=args.devices[0],
            load_config=load_config,
        )
    try:
        result = KimiForwardPipeline(
            adapter=adapter,
            archive=archive,
            devices=args.devices,
            queue_depth=args.queue_depth,
            slab_buffer_tokens=args.slab_buffer_tokens,
            direct_io=not args.buffered_io,
        ).run(inputs, start_layer=start_layer)
        if route_archive is not None:
            route_archive.seal()
        record = summary | {
            "complete": True,
            "elapsed_seconds": result.elapsed_seconds,
            "embedding_load_seconds": None if inputs is None else inputs.load_seconds,
            "resumed_from_boundary": start_layer if args.resume else None,
            "sealed_boundary_count_at_resume": len(sealed_prefix),
            "discarded_unsealed_receipts": list(discarded_boundaries),
            "layers": [_jsonable(item) for item in result.records],
            "route_archive": None
            if route_archive is None
            else str(route_archive.root),
        }
        _atomic_json(destination / RUN_FILENAME, record)
    except BaseException as error:
        record = summary | {
            "complete": False,
            "elapsed_seconds": time.monotonic() - started,
            "error": f"{type(error).__name__}: {error}",
        }
        _atomic_json(destination / FAILURE_FILENAME, record)
        raise
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

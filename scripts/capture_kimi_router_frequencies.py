#!/usr/bin/env python3
"""Measure exact Kimi router marginals without retaining decoder slabs."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

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
from qsrt.kimi_router_frequencies import (
    FrequencyCapturingAdapter,
    RouterFrequencyCollector,
)


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


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return {
            key: _jsonable(item)
            for key, item in dataclasses.asdict(value).items()
        }
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


def _retained_boundaries(
    *,
    num_layers: int,
    attn_res_block_size: int,
    stage_count: int,
) -> tuple[int, ...]:
    retained = {0}
    for boundary in range(stage_count, num_layers, stage_count):
        retained.add(boundary)
        retained.update(range(0, boundary, attn_res_block_size))
    return tuple(sorted(retained))


def _load_bias_overrides(
    path: Path | None,
    *,
    num_layers: int,
    num_experts: int,
) -> dict[int, torch.Tensor]:
    if path is None:
        return {}
    tensors = load_file(path, device="cpu")
    key = "biases" if "biases" in tensors else "correction_bias"
    value = tensors[key].to(torch.float32)
    if value.shape != (num_layers, num_experts):
        raise ValueError(
            f"{path}: bias tensor has shape {tuple(value.shape)}, expected "
            f"{(num_layers, num_experts)}"
        )
    return {layer: value[layer] for layer in range(1, num_layers)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--corpus-report", type=Path)
    parser.add_argument(
        "--token-suite",
        type=Path,
        help="distribution-fidelity token suite replayed without retokenization",
    )
    parser.add_argument(
        "--token-suite-contexts",
        type=_parse_indices,
        help="context IDs from --token-suite, such as 0-31",
    )
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--weight-checkpoint", type=Path, default=DEFAULT_WEIGHT_CHECKPOINT)
    parser.add_argument("--code-checkpoint", type=Path, default=DEFAULT_CODE_CHECKPOINT)
    parser.add_argument("--quantized-anchor-model", type=Path)
    parser.add_argument("--quantized-anchor-candidate-pool", type=Path)
    parser.add_argument(
        "--quantized-anchor-overlay-root", type=Path, action="append", default=[]
    )
    parser.add_argument("--bias-overrides", type=Path)
    parser.add_argument(
        "--devices",
        type=_parse_devices,
        default=_parse_devices(",".join(str(index) for index in range(12))),
    )
    parser.add_argument("--queue-depth", type=int, default=2)
    parser.add_argument("--slab-buffer-tokens", type=int, default=2048)
    parser.add_argument("--instant-buffer-gib", type=int, default=4)
    parser.add_argument("--instant-chunk-mib", type=int, default=16)
    parser.add_argument("--instant-io-depth", type=int, default=256)
    parser.add_argument("--buffered-io", action="store_true")
    parser.add_argument(
        "--grouped-expert-dispatch",
        action="store_true",
        help="execute official MXFP4 experts with three grouped BF16 GEMMs",
    )
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if (args.quantized_anchor_model is None) != (
        args.quantized_anchor_candidate_pool is None
    ):
        raise ValueError("quantized anchor model and candidate pool must be supplied together")
    if args.quantized_anchor_overlay_root and args.quantized_anchor_model is None:
        raise ValueError("quantized anchor overlays require a quantized anchor model")
    if args.bias_overrides is not None and args.quantized_anchor_model is None:
        raise ValueError("bias overrides are valid only for a quantized anchor run")
    if args.grouped_expert_dispatch and args.quantized_anchor_model is not None:
        raise ValueError("grouped-expert-dispatch applies only to the official teacher")
    if args.corpus_report is not None and args.token_suite is not None:
        raise ValueError("corpus report and token suite are mutually exclusive")
    if args.token_suite is None and args.token_suite_contexts is not None:
        raise ValueError("token-suite contexts require --token-suite")
    if args.token_suite is not None and args.token_suite_contexts is None:
        raise ValueError("--token-suite requires --token-suite-contexts")
    destination = args.dest.expanduser().resolve()
    scratch = args.scratch.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if scratch.exists():
        raise FileExistsError(scratch)

    if args.token_suite is not None:
        suite_path = args.token_suite.expanduser().resolve()
        documents = load_token_suite_document_index(
            suite_path,
            args.token_suite_contexts,
        )
        population = {
            "kind": "distribution-fidelity stored-token subset",
            "suite": str(suite_path),
            "suite_manifest_sha256": _sha256(suite_path / "suite-manifest.json"),
            "context_indices": list(args.token_suite_contexts),
            "documents": documents.document_count,
            "tokens": documents.token_count,
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
            raise RuntimeError("transformers is required to reconstruct the corpus") from error
        tokenizer_output = io.StringIO()
        with (
            contextlib.redirect_stdout(tokenizer_output),
            contextlib.redirect_stderr(tokenizer_output),
        ):
            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_root), trust_remote_code=True
            )
            documents = load_corpus_document_index(report_path, tokenizer)
        population = {
            "kind": "authenticated corpus report",
            "corpus_report": str(report_path),
            "corpus_report_sha256": _sha256(report_path),
            "corpus_plan_sha256": report["plan_sha256"],
            "documents": documents.document_count,
            "tokens": documents.token_count,
        }

    runtime = load_official_kimi_runtime(
        weight_checkpoint=args.weight_checkpoint,
        code_checkpoint=args.code_checkpoint,
    )
    num_layers = int(runtime.text_config.num_hidden_layers)
    hidden_dimension = int(runtime.text_config.hidden_size)
    num_experts = int(runtime.text_config.num_experts)
    top_k = int(runtime.text_config.num_experts_per_token)
    attn_res_block_size = int(runtime.text_config.attn_res_block_size)
    retained = _retained_boundaries(
        num_layers=num_layers,
        attn_res_block_size=attn_res_block_size,
        stage_count=len(args.devices),
    )
    slab_bytes = documents.token_count * hidden_dimension * torch.bfloat16.itemsize
    required_bytes = len(retained) * slab_bytes
    summary = {
        "population": population,
        "decoder": {
            "layers": num_layers,
            "hidden_dimension": hidden_dimension,
            "experts": num_experts,
            "top_k": top_k,
        },
        "devices": [str(value) for value in args.devices],
        "retained_transient_boundaries": list(retained),
        "grouped_expert_dispatch": bool(args.grouped_expert_dispatch),
        "transient_slab_bytes": required_bytes,
        "destination": str(destination),
        "scratch": str(scratch),
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return
    if any(
        device.index is None or device.index >= torch.cuda.device_count()
        for device in args.devices
    ):
        raise ValueError("a requested CUDA device is unavailable")
    free_disk = shutil.disk_usage(scratch.parent).free
    if free_disk < required_bytes + (64 << 30):
        raise RuntimeError(
            f"transient execution needs {required_bytes:,} bytes plus reserve; "
            f"only {free_disk:,} bytes are free"
        )

    load_config = InstantTensorLoadConfig(
        buffer_size=args.instant_buffer_gib << 30,
        chunk_size=args.instant_chunk_mib << 20,
        concurrency=1,
        io_depth=args.instant_io_depth,
    )
    archive = KimiBoundarySlabArchive.create(
        scratch,
        documents=documents,
        num_layers=num_layers,
        hidden_dimension=hidden_dimension,
        attn_res_block_size=attn_res_block_size,
        retained_boundaries=retained,
        provenance={
            "purpose": "transient router-frequency execution handoff",
            "retained_boundaries": list(retained),
            "population": population,
        },
    )
    if args.quantized_anchor_model is None:
        base_adapter = OfficialKimiForwardAdapter(
            runtime,
            load_config=load_config,
            grouped_expert_dispatch=args.grouped_expert_dispatch,
        )
        model_kind = "official MXFP4 teacher"
    else:
        base_adapter = QSRTKimiForwardAdapter(
            runtime,
            model_checkpoint=args.quantized_anchor_model,
            expert_payload=QSRTAnchorPayload(
                args.quantized_anchor_candidate_pool,
                overlay_roots=args.quantized_anchor_overlay_root,
            ),
            load_config=load_config,
        )
        model_kind = "quantized QSRT student"
    overrides = _load_bias_overrides(
        args.bias_overrides,
        num_layers=num_layers,
        num_experts=num_experts,
    )
    collector = RouterFrequencyCollector(
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
    )
    adapter = FrequencyCapturingAdapter(
        base_adapter,
        collector,
        bias_overrides=overrides,
    )
    inputs = OfficialKimiEmbeddingInputs(
        runtime=runtime,
        documents=documents,
        device=args.devices[0],
        load_config=load_config,
    )
    started = time.monotonic()
    complete = False
    try:
        result = KimiForwardPipeline(
            adapter=adapter,
            archive=archive,
            devices=args.devices,
            queue_depth=args.queue_depth,
            slab_buffer_tokens=args.slab_buffer_tokens,
            direct_io=not args.buffered_io,
            retained_boundaries=retained,
        ).run(inputs)
        collector.validate_complete(first_layer=1)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        save_file(
            {
                "selection_counts": collector.counts.contiguous(),
                "biases": collector.biases.contiguous(),
                "active_layers": collector.active.contiguous(),
            },
            str(temporary),
            metadata={
                "kind": "Kimi-K3 router-frequency capture",
                "model_kind": model_kind,
                "population_kind": str(population["kind"]),
                "tokens": str(documents.token_count),
            },
        )
        os.replace(temporary, destination)
        run = summary | {
            "complete": True,
            "model_kind": model_kind,
            "bias_overrides": None
            if args.bias_overrides is None
            else str(args.bias_overrides.resolve()),
            "elapsed_seconds": time.monotonic() - started,
            "pipeline_elapsed_seconds": result.elapsed_seconds,
            "embedding_load_seconds": inputs.load_seconds,
            "layer_pipeline_records": _jsonable(result.records),
            "layer_load_seconds_sum": sum(
                record.load_seconds for record in result.records
            ),
            "layer_compute_lane_seconds_sum": sum(
                record.compute_seconds for record in result.records
            ),
            "layers": collector.report(),
            "output_sha256": _sha256(destination),
        }
        _atomic_json(destination.with_suffix(".json"), run)
        complete = True
    finally:
        if not args.keep_scratch and complete:
            shutil.rmtree(scratch, ignore_errors=False)
        elif not complete:
            print(
                f"router-frequency capture failed; retained scratch at {scratch}",
                file=sys.stderr,
            )
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Capture exact student suffix state and frozen teacher distribution targets."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from qsrt.instanttensor_kimi import InstantTensorLoadConfig
from qsrt.kimi_boundary_slabs import DocumentIndex, KimiBoundarySlabArchive
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
from qsrt.kimi_suffix_training_archive import KimiSuffixTrainingArchive
from qsrt.kimi_teacher_target_pipeline import KimiTeacherTargetPipeline


DEFAULT_WEIGHT_CHECKPOINT = Path(
    "/data/cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
)
DEFAULT_CODE_CHECKPOINT = Path(
    "/data/cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721"
)
RUN_FILENAME = "capture-run.json"
TEACHER_RUN_FILENAME = "teacher-target-run.json"
FAILURE_FILENAME = "capture-failure.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def _parse_indices(value: str) -> tuple[int, ...]:
    result = []
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
            "context indices must be comma-separated integers or ranges"
        ) from error
    if not result or len(set(result)) != len(result) or min(result) < 0:
        raise argparse.ArgumentTypeError("context indices must be unique and nonnegative")
    return tuple(result)


def _existing_ancestor(path: Path) -> Path:
    value = path
    while not value.exists():
        parent = value.parent
        if parent == value:
            raise FileNotFoundError(f"no existing ancestor for {path}")
        value = parent
    return value


def _storage_preflight(
    *,
    destination: Path,
    destination_bytes: int,
    scratch: Path,
    scratch_bytes: int,
    reserve_bytes: int,
) -> list[dict[str, object]]:
    destination_mount = _existing_ancestor(destination.parent)
    scratch_mount = _existing_ancestor(scratch.parent)
    if destination_mount.stat().st_dev == scratch_mount.stat().st_dev:
        required = destination_bytes + scratch_bytes
        free = shutil.disk_usage(destination_mount).free
        if free < required + reserve_bytes:
            raise RuntimeError(
                f"capture needs {required:,} bytes plus reserve on "
                f"{destination_mount}; only {free:,} bytes are free"
            )
        return [
            {
                "path": str(destination_mount),
                "required_bytes": required,
                "free_bytes": free,
                "reserve_bytes": reserve_bytes,
            }
        ]

    result = []
    for role, mount, required in (
        ("destination", destination_mount, destination_bytes),
        ("teacher scratch", scratch_mount, scratch_bytes),
    ):
        free = shutil.disk_usage(mount).free
        if free < required + reserve_bytes:
            raise RuntimeError(
                f"{role} needs {required:,} bytes plus reserve on {mount}; "
                f"only {free:,} bytes are free"
            )
        result.append(
            {
                "role": role,
                "path": str(mount),
                "required_bytes": required,
                "free_bytes": free,
                "reserve_bytes": reserve_bytes,
            }
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--teacher-scratch", type=Path, required=True)
    parser.add_argument(
        "--teacher-route-dest",
        type=Path,
        help="capture teacher top-k routes during a fresh teacher waterfall",
    )
    parser.add_argument(
        "--component",
        choices=("all", "teacher", "student"),
        default="all",
        help=(
            "capture both components, only the bias-independent teacher target, "
            "or only student boundaries into an archive with a sealed teacher target"
        ),
    )
    parser.add_argument("--corpus-report", type=Path)
    parser.add_argument("--token-suite", type=Path)
    parser.add_argument("--token-suite-contexts", type=_parse_indices)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--student-candidate-pool", type=Path, required=True)
    parser.add_argument(
        "--student-overlay-root",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--weight-checkpoint",
        type=Path,
        default=DEFAULT_WEIGHT_CHECKPOINT,
    )
    parser.add_argument(
        "--code-checkpoint",
        type=Path,
        default=DEFAULT_CODE_CHECKPOINT,
    )
    parser.add_argument("--cut-layer", type=int, default=84)
    parser.add_argument(
        "--devices",
        type=_parse_devices,
        default=_parse_devices(",".join(str(index) for index in range(12))),
    )
    parser.add_argument("--queue-depth", type=int, default=2)
    parser.add_argument("--slab-buffer-tokens", type=int, default=2048)
    parser.add_argument("--target-buffer-tokens", type=int, default=256)
    parser.add_argument("--target-replay-chunk-tokens", type=int, default=2048)
    parser.add_argument("--instant-buffer-gib", type=int, default=4)
    parser.add_argument("--instant-chunk-mib", type=int, default=16)
    parser.add_argument("--instant-io-depth", type=int, default=256)
    parser.add_argument("--buffered-io", action="store_true")
    parser.add_argument("--keep-teacher-boundaries", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the last sealed retained boundary",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _documents_match(left: DocumentIndex, right: DocumentIndex) -> bool:
    return (
        left.identifiers == right.identifiers
        and torch.equal(left.offsets, right.offsets)
        and torch.equal(left.input_ids, right.input_ids)
    )


def _remaining_allocated_bytes(path: Path, expected_bytes: int) -> int:
    if not path.exists():
        return expected_bytes
    stat = path.stat()
    if stat.st_size != expected_bytes:
        raise ValueError(f"existing slab has the wrong size: {path}")
    return max(0, expected_bytes - stat.st_blocks * 512)


def _remaining_boundary_bytes(
    archive: KimiBoundarySlabArchive,
) -> int:
    return sum(
        _remaining_allocated_bytes(
            archive.boundary_path(boundary),
            archive.expected_slab_bytes,
        )
        for boundary in archive.retained_boundaries
    )


def main() -> None:
    args = _parser().parse_args()
    if (args.corpus_report is None) == (args.token_suite is None):
        raise ValueError("supply exactly one corpus report or token suite")
    if args.teacher_route_dest is not None and args.component == "student":
        raise ValueError("teacher route capture requires the teacher component")
    if args.teacher_route_dest is not None and args.resume:
        raise ValueError("teacher route capture requires a fresh teacher waterfall")
    if (args.token_suite is None) != (args.token_suite_contexts is None):
        raise ValueError("token-suite contexts are required exactly with a token suite")
    if (
        args.queue_depth <= 0
        or args.slab_buffer_tokens <= 0
        or args.target_buffer_tokens <= 0
        or args.target_replay_chunk_tokens <= 0
    ):
        raise ValueError("pipeline queue and slab buffers must be positive")
    destination = args.dest.expanduser().resolve()
    teacher_scratch = args.teacher_scratch.expanduser().resolve()
    teacher_route_destination = (
        None
        if args.teacher_route_dest is None
        else args.teacher_route_dest.expanduser().resolve()
    )
    if args.resume:
        if not destination.exists():
            raise FileNotFoundError(destination)
    else:
        if args.component == "student":
            raise ValueError("student-only capture requires --resume")
        if destination.exists():
            raise FileExistsError(destination)
        if teacher_scratch.exists():
            raise FileExistsError(teacher_scratch)
        if teacher_route_destination is not None and teacher_route_destination.exists():
            raise FileExistsError(teacher_route_destination)

    if args.token_suite is not None:
        suite = args.token_suite.expanduser().resolve()
        documents = load_token_suite_document_index(
            suite,
            args.token_suite_contexts,
        )
        population = {
            "kind": "distribution-fidelity stored-token subset",
            "suite": str(suite),
            "suite_manifest_sha256": _sha256(suite / "suite-manifest.json"),
            "context_indices": list(args.token_suite_contexts),
            "documents": documents.document_count,
            "tokens": documents.token_count,
        }
    else:
        report_path = args.corpus_report.expanduser().resolve()
        report = json.loads(report_path.read_text())
        tokenizer_root = (
            args.tokenizer.expanduser().resolve()
            if args.tokenizer is not None
            else Path(str(report["model_dir"])).expanduser().resolve()
        )
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "run corpus capture with the vLLM Python environment"
            ) from error
        tokenizer_output = io.StringIO()
        with (
            contextlib.redirect_stdout(tokenizer_output),
            contextlib.redirect_stderr(tokenizer_output),
        ):
            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_root),
                trust_remote_code=True,
            )
            documents = load_corpus_document_index(report_path, tokenizer)
        population = {
            "kind": "authenticated corpus report",
            "corpus_report": str(report_path),
            "corpus_report_sha256": _sha256(report_path),
            "corpus_plan_sha256": report["plan_sha256"],
            "tokenizer": str(tokenizer_root),
            "documents": documents.document_count,
            "tokens": documents.token_count,
        }

    runtime = load_official_kimi_runtime(
        weight_checkpoint=args.weight_checkpoint,
        code_checkpoint=args.code_checkpoint,
    )
    num_layers = int(runtime.text_config.num_hidden_layers)
    hidden_dimension = int(runtime.text_config.hidden_size)
    residual_block_size = int(runtime.text_config.attn_res_block_size)
    if (
        not 0 < args.cut_layer < num_layers
        or args.cut_layer % residual_block_size
    ):
        raise ValueError("suffix cut must be an internal residual-segment boundary")
    residual_boundaries = tuple(range(0, num_layers, residual_block_size))
    student_boundaries = tuple(
        range(0, args.cut_layer, residual_block_size)
    ) + (args.cut_layer,)
    teacher_boundaries = residual_boundaries + (num_layers,)
    slab_bytes = documents.token_count * hidden_dimension * torch.bfloat16.itemsize
    destination_bytes = (len(student_boundaries) + 1) * slab_bytes
    scratch_bytes = len(teacher_boundaries) * slab_bytes
    summary = {
        "destination": str(destination),
        "teacher_scratch": str(teacher_scratch),
        "population": population,
        "cut_layer": args.cut_layer,
        "student_boundaries": list(student_boundaries),
        "teacher_transient_boundaries": list(teacher_boundaries),
        "slab_bytes": slab_bytes,
        "permanent_slab_count": len(student_boundaries) + 1,
        "permanent_slab_bytes": destination_bytes,
        "transient_teacher_slab_count": len(teacher_boundaries),
        "transient_teacher_slab_bytes": scratch_bytes,
        "peak_slab_count": len(student_boundaries) + 1 + len(teacher_boundaries),
        "peak_slab_bytes": destination_bytes + scratch_bytes,
        "devices": [str(value) for value in args.devices],
        "teacher_grouped_expert_dispatch": True,
        "teacher_route_destination": (
            None
            if teacher_route_destination is None
            else str(teacher_route_destination)
        ),
        "component": args.component,
        "resume": bool(args.resume),
    }
    if args.dry_run:
        summary["storage_preflight"] = _storage_preflight(
            destination=destination,
            destination_bytes=destination_bytes,
            scratch=teacher_scratch,
            scratch_bytes=scratch_bytes,
            reserve_bytes=64 << 30,
        )
        print(json.dumps(summary, indent=2))
        return
    if any(
        device.index is None or device.index >= torch.cuda.device_count()
        for device in args.devices
    ):
        raise ValueError("a requested CUDA device is unavailable")

    load_config = InstantTensorLoadConfig(
        buffer_size=args.instant_buffer_gib << 30,
        chunk_size=args.instant_chunk_mib << 20,
        concurrency=1,
        io_depth=args.instant_io_depth,
    )
    provenance = {
        "purpose": "continuous-recovery suffix training",
        "population": population,
        "student_model": str(args.student_model.resolve()),
        "student_candidate_pool": str(args.student_candidate_pool.resolve()),
        "student_overlay_roots": [
            str(path.resolve()) for path in args.student_overlay_root
        ],
        "official_weight_checkpoint": str(runtime.weight_checkpoint),
        "official_code_checkpoint": str(runtime.code_checkpoint),
    }
    if args.resume:
        archive = KimiSuffixTrainingArchive(destination)
        if (
            archive.num_layers != num_layers
            or archive.hidden_dimension != hidden_dimension
            or archive.attn_res_block_size != residual_block_size
            or archive.cut_layer != args.cut_layer
            or archive.manifest.get("provenance") != provenance
            or not _documents_match(archive.load_documents(), documents)
        ):
            raise ValueError("resume request does not match the suffix archive")
        if archive.complete:
            record_path = destination / RUN_FILENAME
            if record_path.is_file():
                print(record_path.read_text(), end="")
                return
            record = summary | {
                "complete": True,
                "elapsed_seconds": 0.0,
                "recovered_from_sealed_archive": True,
                "archive_manifest_sha256": _sha256(archive.manifest_path),
                "student_pipeline": {"status": "already complete"},
                "teacher_pipeline": {"status": "already complete"},
                "teacher_target_pipeline": {"status": "already complete"},
            }
            _atomic_json(record_path, record)
            print(json.dumps(record, indent=2))
            return
    else:
        archive = KimiSuffixTrainingArchive.create(
            destination,
            documents=documents,
            num_layers=num_layers,
            hidden_dimension=hidden_dimension,
            attn_res_block_size=residual_block_size,
            cut_layer=args.cut_layer,
            provenance=provenance,
        )

    teacher_provenance = {
        "purpose": "transient teacher state for normalized LM-head targets",
        "population": population,
    }
    existing_teacher = None
    if teacher_scratch.exists():
        if not args.resume:
            raise FileExistsError(teacher_scratch)
        existing_teacher = KimiBoundarySlabArchive(teacher_scratch)
        if (
            existing_teacher.num_layers != num_layers
            or existing_teacher.hidden_dimension != hidden_dimension
            or existing_teacher.attn_res_block_size != residual_block_size
            or existing_teacher.retained_boundaries != teacher_boundaries
            or existing_teacher.manifest.get("provenance") != teacher_provenance
            or not _documents_match(existing_teacher.load_documents(), documents)
        ):
            raise ValueError("resume request does not match the teacher archive")

    remaining_destination_bytes = (
        _remaining_boundary_bytes(archive.student)
        + _remaining_allocated_bytes(
            archive.teacher_target_path,
            archive.expected_slab_bytes,
        )
    )
    target_is_sealed = bool(archive.manifest["teacher_target"]["sealed"])
    if args.component == "student" and not target_is_sealed:
        raise ValueError("student-only capture requires a sealed teacher target")
    remaining_scratch_bytes = (
        0
        if target_is_sealed
        else (
            scratch_bytes
            if existing_teacher is None
            else _remaining_boundary_bytes(existing_teacher)
        )
    )
    summary["remaining_permanent_allocation_bytes"] = remaining_destination_bytes
    summary["remaining_transient_allocation_bytes"] = remaining_scratch_bytes
    summary["storage_preflight"] = _storage_preflight(
        destination=destination,
        destination_bytes=remaining_destination_bytes,
        scratch=teacher_scratch,
        scratch_bytes=remaining_scratch_bytes,
        reserve_bytes=64 << 30,
    )
    started = time.monotonic()
    try:
        if args.component == "teacher":
            student_result: object = {"status": "not requested"}
            student_start_layer = None
        elif archive.student.complete:
            student_result: object = {"status": "already complete"}
            student_start_layer = args.cut_layer
        else:
            archive.student.discard_unsealed_receipts()
            student_prefix = archive.student.sealed_boundary_prefix()
            student_start_layer = student_prefix[-1] if student_prefix else 0
            if student_start_layer == args.cut_layer:
                archive.student.seal()
                student_result = {"status": "sealed existing boundaries"}
            else:
                student_adapter = QSRTKimiForwardAdapter(
                    runtime,
                    model_checkpoint=args.student_model,
                    expert_payload=QSRTAnchorPayload(
                        args.student_candidate_pool,
                        overlay_roots=args.student_overlay_root,
                    ),
                    load_config=load_config,
                )
                student_inputs = (
                    OfficialKimiEmbeddingInputs(
                        runtime=runtime,
                        documents=documents,
                        device=args.devices[0],
                        load_config=load_config,
                    )
                    if not student_prefix
                    else None
                )
                student_result = KimiForwardPipeline(
                    adapter=student_adapter,
                    archive=archive.student,
                    devices=args.devices,
                    queue_depth=args.queue_depth,
                    slab_buffer_tokens=args.slab_buffer_tokens,
                    direct_io=not args.buffered_io,
                    retained_boundaries=student_boundaries,
                ).run(
                    student_inputs,
                    start_layer=student_start_layer,
                    end_layer=args.cut_layer,
                )

        teacher_archive = existing_teacher
        if args.component == "student":
            teacher_result: object = {"status": "not requested; target is sealed"}
            teacher_start_layer = None
            target_result: object = {"status": "already complete"}
        elif target_is_sealed:
            teacher_result: object = {"status": "not required; target is sealed"}
            teacher_start_layer = num_layers
            target_result: object = {"status": "already complete"}
        else:
            if teacher_archive is None:
                teacher_archive = KimiBoundarySlabArchive.create(
                    teacher_scratch,
                    documents=documents,
                    num_layers=num_layers,
                    hidden_dimension=hidden_dimension,
                    attn_res_block_size=residual_block_size,
                    retained_boundaries=teacher_boundaries,
                    provenance=teacher_provenance,
                )
            if teacher_archive.complete:
                teacher_result = {"status": "already complete"}
                teacher_start_layer = num_layers
            else:
                teacher_archive.discard_unsealed_receipts()
                teacher_prefix = teacher_archive.sealed_boundary_prefix()
                teacher_start_layer = teacher_prefix[-1] if teacher_prefix else 0
                if teacher_start_layer == num_layers:
                    teacher_archive.seal()
                    teacher_result = {"status": "sealed existing boundaries"}
                else:
                    teacher_inputs = (
                        OfficialKimiEmbeddingInputs(
                            runtime=runtime,
                            documents=documents,
                            device=args.devices[0],
                            load_config=load_config,
                        )
                        if not teacher_prefix
                        else None
                    )
                    teacher_adapter: object = OfficialKimiForwardAdapter(
                            runtime,
                            load_config=load_config,
                            grouped_expert_dispatch=True,
                        )
                    teacher_route_archive = None
                    if teacher_route_destination is not None:
                        teacher_route_archive = KimiRouteArchive.create(
                            teacher_route_destination,
                            token_count=documents.token_count,
                            num_layers=num_layers,
                            num_experts=int(runtime.text_config.num_experts),
                            top_k=int(runtime.text_config.num_experts_per_token),
                            provenance={
                                "weight_checkpoint": str(runtime.weight_checkpoint),
                                "population": population,
                                "capture_destination": str(destination),
                            },
                        )
                        teacher_adapter = RouteCapturingAdapter(
                            teacher_adapter,
                            teacher_route_archive,
                        )
                    teacher_result = KimiForwardPipeline(
                        adapter=teacher_adapter,
                        archive=teacher_archive,
                        devices=args.devices,
                        queue_depth=args.queue_depth,
                        slab_buffer_tokens=args.slab_buffer_tokens,
                        direct_io=not args.buffered_io,
                        retained_boundaries=teacher_boundaries,
                    ).run(
                        teacher_inputs,
                        start_layer=teacher_start_layer,
                    )
                    if teacher_route_archive is not None:
                        teacher_route_archive.seal()
            if args.resume:
                archive.discard_teacher_target_receipts()
            target_result = KimiTeacherTargetPipeline(
                checkpoint=runtime.weight_checkpoint,
                teacher_boundaries=teacher_archive,
                destination=archive,
                devices=args.devices,
                epsilon=float(runtime.text_config.rms_norm_eps),
                vocabulary_size=int(runtime.text_config.vocab_size),
                slab_buffer_tokens=args.target_buffer_tokens,
                replay_chunk_tokens=args.target_replay_chunk_tokens,
                direct_io=not args.buffered_io,
                load_config=load_config,
            ).run()
        if archive.student.complete and bool(archive.manifest["teacher_target"]["sealed"]):
            archive.seal()
        archive_complete = archive.complete
        record = summary | {
            "complete": archive_complete,
            "component_complete": True,
            "elapsed_seconds": time.monotonic() - started,
            "student_start_layer": student_start_layer,
            "teacher_start_layer": teacher_start_layer,
            "student_pipeline": _jsonable(student_result),
            "teacher_pipeline": _jsonable(teacher_result),
            "teacher_target_pipeline": _jsonable(target_result),
        }
        record_filename = RUN_FILENAME if archive_complete else TEACHER_RUN_FILENAME
        _atomic_json(destination / record_filename, record)
        if (
            args.component != "student"
            and not args.keep_teacher_boundaries
            and bool(archive.manifest["teacher_target"]["sealed"])
            and teacher_scratch.exists()
        ):
            shutil.rmtree(teacher_scratch)
        (destination / FAILURE_FILENAME).unlink(missing_ok=True)
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

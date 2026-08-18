#!/usr/bin/env python3
"""Measure frozen expert-error adapters by exact Kimi-K3 suffix replay."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path
from typing import Iterable, Sequence

import torch

from qsrt.kimi_official_forward import (
    install_grouped_low_rank_adapters,
    load_official_kimi_runtime,
)
from qsrt.kimi_quantized_forward import QSRTAnchorPayload, QSRTKimiForwardAdapter
from qsrt.kimi_suffix_recovery_model import (
    KimiSuffixDecoderStage,
    load_kimi_suffix_decoder_stage,
    load_kimi_suffix_output_modules,
)
from qsrt.kimi_suffix_training_archive import KimiSuffixTrainingArchive
from qsrt.low_rank_adapters import load_sparse_expert_adapter_banks
from qsrt.suffix_recovery_training import (
    DenseDistributionKLLossHead,
    SuffixReplayTrainer,
    SuffixTrainingDocument,
)


DEFAULT_CODE_CHECKPOINT = Path(
    "/data/cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721"
)
DEFAULT_CANDIDATE_POOL = Path(
    "/data/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-CANDIDATES-v1"
)
DEFAULT_OVERLAYS = (
    Path("/data/kquant/research/k3-uniform-k2-direct-viterbi-w2-fixed-v1"),
    Path(
        "/data/kquant/research/k3-uniform-k2-direct-viterbi-all-linears-v1/"
        "upstream-overlays"
    ),
)
FIRST_LAYER = 84
END_LAYER = 93


def _parse_devices(value: str) -> tuple[torch.device, ...]:
    try:
        indices = tuple(int(field) for field in value.split(",") if field.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "devices must be comma-separated integers"
        ) from error
    if len(indices) != END_LAYER - FIRST_LAYER or len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError(
            f"suffix replay requires {END_LAYER - FIRST_LAYER} unique devices"
        )
    return tuple(torch.device("cuda", index) for index in indices)


def _parse_names(value: str) -> tuple[str, ...]:
    names = tuple(field.strip() for field in value.split(",") if field.strip())
    if not names or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("values must be unique and nonempty")
    return names


def _parse_ranks(value: str) -> tuple[int, ...]:
    try:
        ranks = tuple(int(field) for field in value.split(",") if field.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("ranks must be comma-separated integers") from error
    if not ranks or min(ranks) <= 0 or len(set(ranks)) != len(ranks):
        raise argparse.ArgumentTypeError("ranks must be unique and positive")
    return ranks


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _document_batches(
    archive: KimiSuffixTrainingArchive,
    *,
    target_tokens: int,
) -> tuple[tuple[int, ...], ...]:
    documents = archive.load_documents()
    batches: list[tuple[int, ...]] = []
    active: list[int] = []
    active_tokens = 0
    for index in range(documents.document_count):
        first, end = documents.document_extent(index)
        tokens = end - first
        if active and active_tokens + tokens > target_tokens:
            batches.append(tuple(active))
            active = []
            active_tokens = 0
        active.append(index)
        active_tokens += tokens
    if active:
        batches.append(tuple(active))
    return tuple(batches)


def _load_documents(
    archive: KimiSuffixTrainingArchive,
    indices: Sequence[int],
) -> tuple[SuffixTrainingDocument, ...]:
    return tuple(
        archive.load_document(index, direct=True).causal_positions()
        for index in indices
    )


def _prefetched_batches(
    archive: KimiSuffixTrainingArchive,
    batches: Sequence[Sequence[int]],
) -> Iterable[tuple[SuffixTrainingDocument, ...]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_load_documents, archive, batches[0])
        for index in range(len(batches)):
            documents = future.result()
            if index + 1 < len(batches):
                future = executor.submit(
                    _load_documents,
                    archive,
                    batches[index + 1],
                )
            yield documents


def _evaluate(
    trainer: SuffixReplayTrainer,
    archive: KimiSuffixTrainingArchive,
    *,
    batch_tokens: int,
) -> dict[str, object]:
    started = time.monotonic()
    kl_sum = 0.0
    token_count = 0
    batches = _document_batches(archive, target_tokens=batch_tokens)
    for documents in _prefetched_batches(archive, batches):
        result = trainer.evaluate(documents)
        kl_sum += result.kl_sum
        token_count += result.token_count
    return {
        "mean_kl": kl_sum / token_count,
        "kl_sum": kl_sum,
        "token_count": token_count,
        "batch_count": len(batches),
        "elapsed_seconds": time.monotonic() - started,
    }


def _load_stages(
    adapter: QSRTKimiForwardAdapter,
    devices: Sequence[torch.device],
) -> tuple[tuple[KimiSuffixDecoderStage, ...], list[dict[str, object]]]:
    def load(item: tuple[int, torch.device]):
        layer, device = item
        stage, stats, _selected = load_kimi_suffix_decoder_stage(
            adapter,
            layer=layer,
            device=device,
        )
        stage.requires_grad_(False)
        return layer, stage, stats

    loaded = []
    layers = tuple(range(FIRST_LAYER, END_LAYER))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(layers)) as executor:
        loaded.extend(executor.map(load, zip(layers, devices, strict=True)))
    loaded.sort(key=lambda item: item[0])
    stages = tuple(item[1] for item in loaded)
    report = [
        {
            "layer": layer,
            "device": str(device),
            "load_stats": stats.to_json() if hasattr(stats, "to_json") else vars(stats),
        }
        for (layer, _stage, stats), device in zip(loaded, devices, strict=True)
    ]
    return stages, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-archive", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--factor-report", type=Path)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_CANDIDATE_POOL)
    parser.add_argument("--overlay-root", type=Path, action="append", default=None)
    parser.add_argument("--code-checkpoint", type=Path, default=DEFAULT_CODE_CHECKPOINT)
    parser.add_argument(
        "--stage-devices",
        type=_parse_devices,
        default=_parse_devices("0,1,2,3,4,5,6,7,8"),
    )
    parser.add_argument("--student-output-device", type=int, default=9)
    parser.add_argument("--teacher-output-device", type=int, default=10)
    parser.add_argument("--variants", type=_parse_names, default=("plain", "weighted"))
    parser.add_argument("--ranks", type=_parse_ranks, default=(2, 4, 8, 16))
    parser.add_argument("--null-rank", type=int, default=8)
    parser.add_argument("--batch-tokens", type=int, default=32_768)
    parser.add_argument("--loss-chunk-tokens", type=int, default=128)
    return parser


def main() -> None:
    args = _parser().parse_args()
    destination = args.dest.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"evaluation destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    if args.batch_tokens <= 0 or args.loss_chunk_tokens <= 0:
        raise ValueError("evaluation batch and chunk sizes must be positive")
    if any(variant not in {"plain", "weighted"} for variant in args.variants):
        raise ValueError("adapter variants must be plain or weighted")

    archive = KimiSuffixTrainingArchive(
        args.screening_archive,
        require_complete=True,
    )
    if archive.cut_layer != FIRST_LAYER or archive.num_layers != END_LAYER:
        raise ValueError("screening archive does not describe layers 84-92")
    overlays = tuple(DEFAULT_OVERLAYS if args.overlay_root is None else args.overlay_root)
    factor_report_path = (
        args.factors.parent / "result.json"
        if args.factor_report is None
        else args.factor_report
    ).expanduser().resolve()
    factor_report = json.loads(factor_report_path.read_text())
    if int(factor_report.get("layer", -1)) != FIRST_LAYER:
        raise ValueError("factor report does not describe decoder layer 84")
    if Path(str(factor_report.get("candidate_pool"))).resolve() != args.candidate_pool.resolve():
        raise ValueError("factor report uses a different candidate pool")
    reported_overlays = tuple(
        Path(str(path)).resolve() for path in factor_report.get("overlay_roots", ())
    )
    if reported_overlays != tuple(path.resolve() for path in overlays):
        raise ValueError("factor report uses different expert payload overlays")

    runtime = load_official_kimi_runtime(
        weight_checkpoint=args.model,
        code_checkpoint=args.code_checkpoint,
    )
    payload = QSRTAnchorPayload(args.candidate_pool, overlay_roots=overlays)
    adapter = QSRTKimiForwardAdapter(
        runtime,
        model_checkpoint=args.model,
        expert_payload=payload,
        validate_outputs=False,
    )
    stages, stage_report = _load_stages(adapter, args.stage_devices)
    student_output, teacher_output = load_kimi_suffix_output_modules(
        checkpoint=args.model,
        student_device=torch.device("cuda", args.student_output_device),
        teacher_device=torch.device("cuda", args.teacher_output_device),
        epsilon=float(runtime.text_config.rms_norm_eps),
    )
    loss_head = DenseDistributionKLLossHead(
        student=student_output,
        teacher=teacher_output,
        chunk_tokens=args.loss_chunk_tokens,
        logit_scale=float(getattr(runtime.text_config, "logit_scale", 1.0)),
    )
    trainer = SuffixReplayTrainer(
        stages=stages,
        loss_head=loss_head,
        queue_depth=1,
        checkpoint_stages=False,
    )

    layer_stage = stages[FIRST_LAYER - FIRST_LAYER]
    block = layer_stage.decoder.block_sparse_moe
    banks = block._qsrt_expert_weight_banks
    shapes = {
        matrix: (int(bank.shape[1]), int(bank.shape[2]))
        for matrix, bank in banks.items()
    }
    records = [{"name": "anchor", **_evaluate(trainer, archive, batch_tokens=args.batch_tokens)}]
    null_banks, null_experts, _null_metadata = load_sparse_expert_adapter_banks(
        args.factors,
        variant=args.variants[0],
        rank=args.null_rank,
        matrix_shapes=shapes,
        num_experts=int(banks["w1"].shape[0]),
        device=args.stage_devices[0],
        dtype=banks["w1"].dtype,
    )
    for factors in null_banks.values():
        for factor in factors:
            factor.zero_()
    install_grouped_low_rank_adapters(layer_stage.decoder, null_banks)
    null_result = _evaluate(trainer, archive, batch_tokens=args.batch_tokens)
    null_result.update(
        {
            "name": f"zero-adapter-rank-{args.null_rank}",
            "rank": args.null_rank,
            "experts": list(null_experts),
        }
    )
    if null_result["kl_sum"] != records[0]["kl_sum"]:
        raise RuntimeError("zero-factor adapter changed suffix-replay KL")
    records.append(null_result)
    for variant in args.variants:
        for rank in args.ranks:
            factor_banks, experts, metadata = load_sparse_expert_adapter_banks(
                args.factors,
                variant=variant,
                rank=rank,
                matrix_shapes=shapes,
                num_experts=int(banks["w1"].shape[0]),
                device=args.stage_devices[0],
                dtype=banks["w1"].dtype,
            )
            if int(metadata.get("layer", -1)) != FIRST_LAYER:
                raise ValueError("factor metadata does not describe decoder layer 84")
            install_grouped_low_rank_adapters(layer_stage.decoder, factor_banks)
            result = _evaluate(trainer, archive, batch_tokens=args.batch_tokens)
            result.update(
                {
                    "name": f"{variant}-rank-{rank}",
                    "variant": variant,
                    "rank": rank,
                    "experts": list(experts),
                }
            )
            records.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    install_grouped_low_rank_adapters(layer_stage.decoder, None)

    baseline = float(records[0]["mean_kl"])
    for record in records:
        record["relative_kl_change_from_anchor"] = float(record["mean_kl"]) / baseline - 1.0
    report = {
        "kind": "Kimi-K3 frozen expert-error adapter suffix-replay evaluation",
        "status": "research_only",
        "screening_archive": str(archive.root),
        "model": str(args.model.expanduser().resolve()),
        "candidate_pool": str(args.candidate_pool.expanduser().resolve()),
        "overlay_roots": [str(path.expanduser().resolve()) for path in overlays],
        "factor_file": str(args.factors.expanduser().resolve()),
        "factor_report": str(factor_report_path),
        "stage_loads": stage_report,
        "records": records,
    }
    _atomic_json(destination / "result.json", report)


if __name__ == "__main__":
    main()

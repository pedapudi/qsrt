#!/usr/bin/env python3
"""Train Kimi-K3 suffix shared experts and normalization tensors."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Iterable, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from qsrt.instanttensor_kimi import InstantTensorLoadConfig
from qsrt.kimi_official_forward import load_official_kimi_runtime
from qsrt.kimi_quantized_forward import QSRTAnchorPayload, QSRTKimiForwardAdapter
from qsrt.kimi_routes import KimiRouteArchive
from qsrt.kimi_suffix_recovery_model import (
    KimiSuffixDecoderStage,
    KimiSuffixStudentOutput,
    load_kimi_suffix_decoder_stage,
    load_kimi_suffix_output_modules,
)
from qsrt.kimi_suffix_training_archive import KimiSuffixTrainingArchive
from qsrt.suffix_recovery_training import (
    DenseDistributionKLLossHead,
    FP32AdamWConfig,
    FP32MasterAdamW,
    SuffixReplayTrainer,
    SuffixTrainingDocument,
    combined_gradient_norm,
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


class _RouteAgreementAccumulator:
    """Compare suffix gate results with teacher routes on identical documents."""

    def __init__(
        self,
        *,
        archive: KimiSuffixTrainingArchive,
        teacher: KimiRouteArchive,
        layers: Sequence[int],
    ):
        if archive.token_count != teacher.token_count:
            raise ValueError("screening and teacher route archives have different rows")
        archive_population = archive.manifest.get("provenance", {}).get("population")
        route_population = teacher.manifest.get("provenance", {}).get("population")
        if archive_population != route_population:
            raise ValueError(
                "screening and teacher route archives describe different populations"
            )
        if min(layers) < teacher.first_layer or max(layers) >= teacher.num_layers:
            raise ValueError("teacher route archive does not cover the suffix layers")
        documents = archive.load_documents()
        self.extents = {
            identifier: documents.document_extent(index)
            for index, identifier in enumerate(documents.identifiers)
        }
        self.teacher = {
            layer: teacher.read_layer(layer)
            for layer in layers
        }
        self.num_experts = teacher.num_experts
        self.top_k = teacher.top_k
        self.overlap = {layer: 0 for layer in layers}
        self.exact = {layer: 0 for layer in layers}
        self.tokens = {layer: 0 for layer in layers}
        self.teacher_counts = {
            layer: torch.zeros(self.num_experts, dtype=torch.int64)
            for layer in layers
        }
        self.student_counts = {
            layer: torch.zeros(self.num_experts, dtype=torch.int64)
            for layer in layers
        }

    def observe(
        self,
        _stage_index: int,
        document: SuffixTrainingDocument,
        module: torch.nn.Module,
    ) -> None:
        if not isinstance(module, KimiSuffixDecoderStage):
            raise TypeError("route agreement requires Kimi suffix decoder stages")
        student = module.take_captured_routes().reshape(-1, self.top_k)
        first, end = self.extents[document.identifier]
        teacher = self.teacher[module.layer][first : end - 1]
        if student.shape != teacher.shape:
            raise ValueError("student and teacher routes have different document geometry")
        student64 = student.to(torch.int64)
        teacher64 = teacher.to(torch.int64)
        matches = (student64.unsqueeze(2) == teacher64.unsqueeze(1)).any(dim=2)
        self.overlap[module.layer] += int(matches.sum().item())
        exact = torch.equal(student64.sort(dim=1).values, teacher64.sort(dim=1).values)
        if exact:
            self.exact[module.layer] += int(student64.shape[0])
        else:
            self.exact[module.layer] += int(
                (
                    student64.sort(dim=1).values
                    == teacher64.sort(dim=1).values
                ).all(dim=1).sum().item()
            )
        self.tokens[module.layer] += int(student64.shape[0])
        self.student_counts[module.layer] += torch.bincount(
            student64.flatten(),
            minlength=self.num_experts,
        )
        self.teacher_counts[module.layer] += torch.bincount(
            teacher64.flatten(),
            minlength=self.num_experts,
        )

    def report(self) -> list[dict[str, object]]:
        values = []
        for layer in sorted(self.tokens):
            tokens = self.tokens[layer]
            if tokens <= 0:
                raise RuntimeError(f"no routes observed for layer {layer}")
            selections = tokens * self.top_k
            marginal_tv = 0.5 * float(
                (
                    self.student_counts[layer] / selections
                    - self.teacher_counts[layer] / selections
                ).abs().sum().item()
            )
            values.append(
                {
                    "layer": layer,
                    "mean_topk_overlap": self.overlap[layer] / selections,
                    "exact_topk_set_agreement": self.exact[layer] / tokens,
                    "marginal_total_variation": marginal_tv,
                    "token_count": tokens,
                }
            )
        return values


def _parse_devices(value: str) -> tuple[torch.device, ...]:
    try:
        indices = tuple(int(field) for field in value.split(",") if field.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "devices must be comma-separated integers"
        ) from error
    if not indices or len(indices) != len(set(indices)) or min(indices) < 0:
        raise argparse.ArgumentTypeError("devices must be unique and nonnegative")
    return tuple(torch.device("cuda", index) for index in indices)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_json(path: Path, value: object) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def _named_trainables(module: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def _document_batches(
    archive: KimiSuffixTrainingArchive,
    *,
    target_tokens: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    if target_tokens <= 0:
        raise ValueError("target batch size must be positive")
    documents = archive.load_documents()
    order = list(range(documents.document_count))
    random.Random(seed).shuffle(order)
    batches: list[tuple[int, ...]] = []
    active: list[int] = []
    active_tokens = 0
    for index in order:
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


def _load_batch(
    archive: KimiSuffixTrainingArchive,
    indices: Sequence[int],
) -> tuple[SuffixTrainingDocument, ...]:
    return tuple(
        archive.load_document(index, direct=True).causal_positions()
        for index in indices
    )


def _split_batch_indices(
    archive: KimiSuffixTrainingArchive,
    indices: Sequence[int],
    *,
    target_tokens: int,
) -> tuple[tuple[int, ...], ...]:
    """Partition one optimizer batch into whole-document replay microbatches."""

    if target_tokens <= 0:
        raise ValueError("target microbatch size must be positive")
    documents = archive.load_documents()
    batches: list[tuple[int, ...]] = []
    active: list[int] = []
    active_tokens = 0
    for index in indices:
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


def _prefetched_batches(
    archive: KimiSuffixTrainingArchive,
    batches: Sequence[Sequence[int]],
) -> Iterable[tuple[Sequence[int], tuple[SuffixTrainingDocument, ...]]]:
    """Overlap one bounded archive read with the active GPU batch."""

    if not batches:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_load_batch, archive, batches[0])
        for index, indices in enumerate(batches):
            documents = future.result()
            if index + 1 < len(batches):
                future = executor.submit(_load_batch, archive, batches[index + 1])
            yield indices, documents


def _stage_overlay_name(layer: int, parameter_name: str) -> str:
    prefix = "decoder."
    if not parameter_name.startswith(prefix):
        raise ValueError(f"suffix stage parameter lacks decoder prefix: {parameter_name}")
    return (
        f"language_model.model.layers.{layer}."
        f"{parameter_name[len(prefix):]}"
    )


def _save_overlay(
    path: Path,
    *,
    stages: Sequence[KimiSuffixDecoderStage],
    student_output: KimiSuffixStudentOutput,
    metadata: dict[str, str],
    verify: bool = False,
) -> dict[str, object]:
    tensors: dict[str, torch.Tensor] = {}
    for stage in stages:
        for name, parameter in _named_trainables(stage).items():
            tensors[_stage_overlay_name(stage.layer, name)] = (
                parameter.detach().to(device="cpu").contiguous()
            )
    for name, parameter in _named_trainables(student_output).items():
        try:
            checkpoint_name = student_output.CHECKPOINT_PARAMETER_NAMES[name]
        except KeyError as error:
            raise KeyError(f"unmapped suffix output parameter {name}") from error
        tensors[checkpoint_name] = parameter.detach().to(device="cpu").contiguous()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file(tensors, temporary, metadata=metadata)
    os.replace(temporary, path)
    tensor_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
    if verify:
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            if set(reader.keys()) != set(tensors):
                raise ValueError("saved suffix overlay has the wrong tensor set")
            for name, expected in tensors.items():
                actual = reader.get_tensor(name)
                if actual.dtype != expected.dtype or not torch.equal(actual, expected):
                    raise ValueError(
                        f"saved suffix overlay differs from runtime tensor {name}"
                    )
    return {
        "path": str(path),
        "tensor_count": len(tensors),
        "tensor_bytes": tensor_bytes,
        "bit_identical_to_runtime": bool(verify),
    }


def _load_stages(
    adapter: QSRTKimiForwardAdapter,
    devices: Sequence[torch.device],
) -> tuple[tuple[KimiSuffixDecoderStage, ...], list[dict[str, object]]]:
    layers = tuple(range(FIRST_LAYER, END_LAYER))
    if len(devices) != len(layers):
        raise ValueError(f"suffix replay requires exactly {len(layers)} stage devices")

    def load(item: tuple[int, torch.device]):
        layer, device = item
        stage, stats, selected = load_kimi_suffix_decoder_stage(
            adapter,
            layer=layer,
            device=device,
        )
        return layer, stage, stats, selected

    loaded = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(layers)) as executor:
        for result in executor.map(load, zip(layers, devices, strict=True)):
            loaded.append(result)
    loaded.sort(key=lambda item: item[0])
    stages = tuple(item[1] for item in loaded)
    report = [
        {
            "layer": layer,
            "device": str(device),
            "selected_tensors": list(selected),
            "load_stats": (
                stats.to_json() if hasattr(stats, "to_json") else vars(stats)
            ),
        }
        for (layer, _stage, stats, selected), device in zip(
            loaded,
            devices,
            strict=True,
        )
    ]
    return stages, report


def _evaluate(
    trainer: SuffixReplayTrainer,
    archive: KimiSuffixTrainingArchive,
    *,
    target_tokens: int,
    teacher_routes: KimiRouteArchive | None = None,
    stages: Sequence[KimiSuffixDecoderStage] = (),
) -> dict[str, object]:
    started = time.monotonic()
    kl_sum = 0.0
    token_count = 0
    route_agreement = None
    if teacher_routes is not None:
        if not stages:
            raise ValueError("route agreement requires suffix decoder stages")
        route_agreement = _RouteAgreementAccumulator(
            archive=archive,
            teacher=teacher_routes,
            layers=[stage.layer for stage in stages],
        )
        for stage in stages:
            stage.enable_route_capture()
    batches = _document_batches(archive, target_tokens=target_tokens, seed=0)
    try:
        for batch_index, (_indices, documents) in enumerate(
            _prefetched_batches(archive, batches)
        ):
            result = trainer.evaluate(
                documents,
                stage_observer=(
                    None if route_agreement is None else route_agreement.observe
                ),
            )
            kl_sum += result.kl_sum
            token_count += result.token_count
            del result
    finally:
        if route_agreement is not None:
            for stage in stages:
                stage.disable_route_capture()
    report = {
        "mean_kl": kl_sum / token_count,
        "kl_sum": kl_sum,
        "token_count": token_count,
        "batch_count": len(batches),
        "elapsed_seconds": time.monotonic() - started,
    }
    if route_agreement is not None:
        report["route_agreement"] = route_agreement.report()
    return report


def _step_learning_rate(base: float, *, step: int, warmup_steps: int) -> float:
    if step <= 0:
        raise ValueError("optimizer step must be positive")
    if warmup_steps <= 0 or step >= warmup_steps:
        return base
    return base * step / warmup_steps


def _update_norm_groups(
    stages: Sequence[KimiSuffixDecoderStage],
    optimizer_steps: Sequence[object],
) -> dict[str, float]:
    squared = {
        "shared_experts": 0.0,
        "decoder_norms": 0.0,
        "output_norms": 0.0,
    }
    for stage, result in zip(stages, optimizer_steps[:-1], strict=True):
        del stage
        for name, norm in result.parameter_update_norms.items():
            group = "shared_experts" if ".shared_experts." in name else "decoder_norms"
            squared[group] += float(norm) ** 2
    for norm in optimizer_steps[-1].parameter_update_norms.values():
        squared["output_norms"] += float(norm) ** 2
    return {name: math.sqrt(value) for name, value in squared.items()}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-archive", type=Path, required=True)
    parser.add_argument("--screening-archive", type=Path, required=True)
    parser.add_argument(
        "--screening-teacher-routes",
        type=Path,
        help="teacher top-k routes captured on the screening archive population",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_CANDIDATE_POOL)
    parser.add_argument(
        "--overlay-root",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--code-checkpoint", type=Path, default=DEFAULT_CODE_CHECKPOINT)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--stage-devices", type=_parse_devices, default=_parse_devices("0,1,2,3,4,5,6,7,8"))
    parser.add_argument("--student-output-device", type=int, default=9)
    parser.add_argument("--teacher-output-device", type=int, default=10)
    parser.add_argument("--batch-tokens", type=int, default=32_768)
    parser.add_argument(
        "--microbatch-tokens",
        type=int,
        help=(
            "whole-document replay limit used for FP32 gradient accumulation; "
            "defaults to --batch-tokens"
        ),
    )
    parser.add_argument("--loss-chunk-tokens", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--screen-every",
        type=int,
        default=0,
        help="screen every N optimizer steps; zero disables step-based screening",
    )
    parser.add_argument("--screen-every-epoch", action="store_true")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--queue-depth", type=int, default=1)
    parser.add_argument("--no-checkpoint-stages", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    destination = args.dest.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"training destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    if args.epochs <= 0 or args.screen_every < 0 or args.warmup_steps < 0:
        raise ValueError("epochs must be positive and schedule counts nonnegative")
    if args.max_steps is not None and args.max_steps < 0:
        raise ValueError("maximum step count cannot be negative")
    if args.batch_tokens <= 0:
        raise ValueError("optimizer batch size must be positive")
    if args.microbatch_tokens is None:
        args.microbatch_tokens = args.batch_tokens
    if args.microbatch_tokens <= 0 or args.microbatch_tokens > args.batch_tokens:
        raise ValueError(
            "microbatch size must be positive and no larger than optimizer batch size"
        )
    if not math.isfinite(args.learning_rate) or args.learning_rate < 0.0:
        raise ValueError("learning rate must be finite and nonnegative")

    training = KimiSuffixTrainingArchive(args.training_archive, require_complete=True)
    screening = KimiSuffixTrainingArchive(args.screening_archive, require_complete=True)
    screening_teacher_routes = (
        None
        if args.screening_teacher_routes is None
        else KimiRouteArchive(args.screening_teacher_routes, require_complete=True)
    )
    for archive in (training, screening):
        if archive.cut_layer != FIRST_LAYER or archive.num_layers != END_LAYER:
            raise ValueError("suffix archive does not describe layers 84-92")

    runtime = load_official_kimi_runtime(
        weight_checkpoint=args.model,
        code_checkpoint=args.code_checkpoint,
    )
    overlays = tuple(DEFAULT_OVERLAYS if args.overlay_root is None else args.overlay_root)
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
        load_config=InstantTensorLoadConfig(),
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
        queue_depth=args.queue_depth,
        checkpoint_stages=not args.no_checkpoint_stages,
    )
    optimizer_config = FP32AdamWConfig(learning_rate=args.learning_rate)
    optimizers = [
        FP32MasterAdamW(_named_trainables(stage), optimizer_config)
        for stage in stages
    ]
    output_optimizer = FP32MasterAdamW(
        _named_trainables(student_output),
        optimizer_config,
    )
    all_optimizers = (*optimizers, output_optimizer)

    manifest = {
        "semantic_role": "Kimi-K3 suffix shared-expert and normalization training",
        "training_archive": str(training.root),
        "screening_archive": str(screening.root),
        "screening_teacher_routes": (
            None
            if screening_teacher_routes is None
            else str(screening_teacher_routes.root)
        ),
        "model": str(args.model.expanduser().resolve()),
        "candidate_pool": str(args.candidate_pool.expanduser().resolve()),
        "overlay_roots": [str(path.expanduser().resolve()) for path in overlays],
        "layers": [FIRST_LAYER, END_LAYER],
        "stage_devices": [str(device) for device in args.stage_devices],
        "student_output_device": args.student_output_device,
        "teacher_output_device": args.teacher_output_device,
        "batch_tokens": args.batch_tokens,
        "microbatch_tokens": args.microbatch_tokens,
        "loss_chunk_tokens": args.loss_chunk_tokens,
        "learning_rate": args.learning_rate,
        "optimizer": {
            "name": "AdamW",
            "beta1": optimizer_config.beta1,
            "beta2": optimizer_config.beta2,
            "epsilon": optimizer_config.epsilon,
            "weight_decay": optimizer_config.weight_decay_toward_initial,
            "accumulation_dtype": "float32",
        },
        "warmup_steps": args.warmup_steps,
        "max_gradient_norm": args.max_gradient_norm,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "screen_every": args.screen_every,
        "screen_every_epoch": bool(args.screen_every_epoch),
        "seed": args.seed,
        "stage_loads": stage_report,
    }
    _atomic_json(destination / "run.json", manifest)
    zero_overlay = destination / "overlay-step-000000.safetensors"
    zero_overlay_identity = _save_overlay(
        zero_overlay,
        stages=stages,
        student_output=student_output,
        metadata={"step": "0", "model": manifest["model"]},
        verify=True,
    )
    _append_json(
        destination / "progress.jsonl",
        {"event": "zero_step_overlay", "step": 0, **zero_overlay_identity},
    )
    initial_screening = _evaluate(
        trainer,
        screening,
        target_tokens=args.batch_tokens,
        teacher_routes=screening_teacher_routes,
        stages=stages,
    )
    _append_json(
        destination / "progress.jsonl",
        {"event": "screening", "step": 0, **initial_screening},
    )
    best_screening = initial_screening
    best_overlay = zero_overlay
    screened_steps = {0}

    def screen(step: int) -> None:
        nonlocal best_screening, best_overlay
        if step in screened_steps:
            return
        overlay = destination / f"overlay-step-{step:06d}.safetensors"
        _save_overlay(
            overlay,
            stages=stages,
            student_output=student_output,
            metadata={"step": str(step), "model": manifest["model"]},
        )
        result = _evaluate(
            trainer,
            screening,
            target_tokens=args.batch_tokens,
            teacher_routes=screening_teacher_routes,
            stages=stages,
        )
        _append_json(
            destination / "progress.jsonl",
            {"event": "screening", "step": step, **result},
        )
        screened_steps.add(step)
        if result["mean_kl"] < best_screening["mean_kl"]:
            best_screening = result
            best_overlay = overlay

    step = 0
    stop = args.max_steps == 0
    for epoch in range(args.epochs):
        if stop:
            break
        batches = _document_batches(
            training,
            target_tokens=args.batch_tokens,
            seed=args.seed + epoch,
        )
        for batch_index, indices in enumerate(batches):
            started = time.monotonic()
            microbatches = _split_batch_indices(
                training,
                indices,
                target_tokens=args.microbatch_tokens,
            )
            batch_kl_sum = 0.0
            batch_token_count = 0
            for _microbatch_index, (_micro_indices, documents) in enumerate(
                _prefetched_batches(training, microbatches)
            ):
                gradients = trainer.gradients(
                    documents,
                    retain_input_gradients=False,
                )
                for optimizer, values in zip(
                    optimizers,
                    gradients.stage_parameter_gradients,
                    strict=True,
                ):
                    optimizer.accumulate(values)
                output_optimizer.accumulate(gradients.output_parameter_gradients)
                batch_kl_sum += gradients.kl_sum
                batch_token_count += gradients.token_count
                del gradients
            gradient_scale = 1.0 / batch_token_count
            global_norm = combined_gradient_norm(
                all_optimizers,
                gradient_scale=gradient_scale,
            )
            rate = _step_learning_rate(
                args.learning_rate,
                step=step + 1,
                warmup_steps=args.warmup_steps,
            )
            optimizer_steps = [
                optimizer.step(
                    gradient_scale=gradient_scale,
                    max_gradient_norm=args.max_gradient_norm,
                    global_gradient_norm=global_norm,
                    learning_rate=rate,
                )
                for optimizer in all_optimizers
            ]
            step += 1
            _append_json(
                destination / "progress.jsonl",
                {
                    "event": "train",
                    "epoch": epoch,
                    "batch": batch_index,
                    "step": step,
                    "mean_kl": batch_kl_sum / batch_token_count,
                    "token_count": batch_token_count,
                    "microbatch_count": len(microbatches),
                    "global_gradient_norm": global_norm,
                    "learning_rate": rate,
                    "clipping_scale": optimizer_steps[0].clipping_scale,
                    "parameter_update_norms": _update_norm_groups(
                        stages,
                        optimizer_steps,
                    ),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            if args.screen_every and step % args.screen_every == 0:
                screen(step)
            if args.max_steps is not None and step >= args.max_steps:
                stop = True
                break
        if args.screen_every_epoch:
            screen(step)

    final_overlay = destination / f"overlay-step-{step:06d}-final.safetensors"
    _save_overlay(
        final_overlay,
        stages=stages,
        student_output=student_output,
        metadata={"step": str(step), "model": manifest["model"]},
    )
    _atomic_json(
        destination / "complete.json",
        {
            "step": step,
            "initial_screening": initial_screening,
            "best_screening": best_screening,
            "best_overlay": str(best_overlay),
            "final_overlay": str(final_overlay),
        },
    )


if __name__ == "__main__":
    main()

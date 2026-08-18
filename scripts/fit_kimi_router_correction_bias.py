#!/usr/bin/env python3
"""Fit Kimi selection-only router biases by teacher-frequency feedback."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from qsrt.kimi_boundary_slabs import KimiBoundarySlabArchive
from qsrt.kimi_routes import KimiRouteArchive


DEFAULT_SCORES = Path(
    "/data/kquant/research/qsrt-continuous-recovery-m0/"
    "layer-012-student-router-scores.safetensors"
)
DEFAULT_BOUNDARIES = Path(
    "/data/kquant/research/qsrt-continuous-recovery-m0/student-legacy32-boundaries"
)
DEFAULT_TEACHER_ROUTES = Path(
    "/data/kquant/research/qsrt-continuous-recovery-m0/teacher-legacy32-routes"
)
DEFAULT_STUDENT_ROUTES = Path(
    "/data/kquant/research/qsrt-continuous-recovery-m0/student-legacy32-routes"
)
DEFAULT_OUTPUT = Path(
    "/data/kquant/research/qsrt-continuous-recovery-m0/"
    "layer-012-frequency-matched-router-bias.safetensors"
)
DEFAULT_REPORT = Path(
    "/data/kquant/research/qsrt-continuous-recovery-m0/"
    "layer-012-frequency-matched-router-bias.json"
)


@dataclass(frozen=True)
class Schedule:
    name: str
    phases: tuple[tuple[float, int], ...]


SCHEDULES = (
    Schedule("fine", ((0.002, 150), (0.001, 150), (0.0005, 150))),
    Schedule("medium", ((0.005, 100), (0.002, 100), (0.001, 100), (0.0005, 100))),
    Schedule("coarse", ((0.01, 100), (0.005, 100), (0.002, 100), (0.001, 100))),
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _token_indices(
    archive: KimiBoundarySlabArchive,
    *,
    parity: int | None,
) -> torch.Tensor:
    documents = archive.load_documents()
    parts = []
    for document in range(documents.document_count):
        if parity is not None and document % 2 != parity:
            continue
        first, end = documents.document_extent(document)
        parts.append(torch.arange(first, end, dtype=torch.int64))
    return torch.cat(parts)


def _metrics(
    scores: torch.Tensor,
    bias: torch.Tensor,
    teacher_ids: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor]:
    top_k = int(teacher_ids.shape[1])
    candidate_ids = torch.topk(scores + bias, top_k, dim=1, sorted=False).indices
    expert_count = int(scores.shape[1])
    teacher_counts = torch.bincount(
        teacher_ids.reshape(-1), minlength=expert_count
    ).to(torch.float32)
    candidate_counts = torch.bincount(
        candidate_ids.reshape(-1), minlength=expert_count
    ).to(torch.float32)
    selections = teacher_ids.numel()
    intersection = (
        candidate_ids.unsqueeze(2) == teacher_ids.unsqueeze(1)
    ).any(dim=2).sum(dim=1)
    metrics = {
        "marginal_total_variation": float(
            0.5 * (candidate_counts - teacher_counts).abs().sum() / selections
        ),
        "mean_topk_overlap": float(intersection.to(torch.float32).mean() / top_k),
        "exact_topk_set_agreement": float(
            (intersection == top_k).to(torch.float32).mean()
        ),
    }
    frequency_gradient = (
        candidate_counts - teacher_counts
    ) / int(scores.shape[0])
    return metrics, frequency_gradient


def _fit(
    scores: torch.Tensor,
    teacher_ids: torch.Tensor,
    initial_bias: torch.Tensor,
    schedule: Schedule,
) -> tuple[torch.Tensor, dict[str, object]]:
    bias = initial_bias.clone()
    initial_mean = initial_bias.mean()
    best_bias = bias.clone()
    best_metrics, _ = _metrics(scores, bias, teacher_ids)
    phases = []
    iteration = 0
    for learning_rate, steps in schedule.phases:
        for _ in range(steps):
            metrics, gradient = _metrics(scores, bias, teacher_ids)
            if metrics["marginal_total_variation"] < best_metrics[
                "marginal_total_variation"
            ]:
                best_metrics = metrics
                best_bias.copy_(bias)
            bias.add_(gradient, alpha=-learning_rate)
            bias.add_(initial_mean - bias.mean())
            iteration += 1
        phase_metrics, _ = _metrics(scores, bias, teacher_ids)
        phases.append(
            {
                "learning_rate": learning_rate,
                "steps": steps,
                "ending_iteration": iteration,
                "ending_metrics": phase_metrics,
            }
        )
    final_metrics, _ = _metrics(scores, bias, teacher_ids)
    if final_metrics["marginal_total_variation"] < best_metrics[
        "marginal_total_variation"
    ]:
        best_metrics = final_metrics
        best_bias.copy_(bias)
    return best_bias, {"best_fit_metrics": best_metrics, "phases": phases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARIES)
    parser.add_argument("--teacher-routes", type=Path, default=DEFAULT_TEACHER_ROUTES)
    parser.add_argument("--student-routes", type=Path, default=DEFAULT_STUDENT_ROUTES)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("device must be an indexed CUDA device")
    torch.cuda.set_device(device)
    score_tensors = load_file(args.scores, device="cpu")
    scores = score_tensors["scores"].to(device=device, dtype=torch.float32)
    initial_bias = score_tensors["correction_bias"].to(device=device, dtype=torch.float32)
    teacher_all = KimiRouteArchive(args.teacher_routes).read_layer(args.layer).to(
        device=device, dtype=torch.int64
    )
    student_all = KimiRouteArchive(args.student_routes).read_layer(args.layer).to(
        device=device, dtype=torch.int64
    )
    boundaries = KimiBoundarySlabArchive(args.boundaries)
    if scores.shape[0] != teacher_all.shape[0] or teacher_all.shape != student_all.shape:
        raise ValueError("score and route archives have different token geometry")

    baseline_ids = torch.topk(
        scores + initial_bias, int(student_all.shape[1]), dim=1, sorted=False
    ).indices
    baseline_mismatch = int(
        (
            baseline_ids.sort(dim=1).values
            != student_all.sort(dim=1).values
        ).any(dim=1).sum()
    )
    if baseline_mismatch:
        raise RuntimeError(
            f"stored scores disagree with {baseline_mismatch} archived student routes"
        )

    fit_indices = _token_indices(boundaries, parity=0).to(device)
    validation_indices = _token_indices(boundaries, parity=1).to(device)
    all_indices = _token_indices(boundaries, parity=None).to(device)
    split_results = []
    candidates: dict[str, torch.Tensor] = {}
    for schedule in SCHEDULES:
        bias, fit_receipt = _fit(
            scores[fit_indices],
            teacher_all[fit_indices],
            initial_bias,
            schedule,
        )
        validation_metrics, _ = _metrics(
            scores[validation_indices], bias, teacher_all[validation_indices]
        )
        candidates[schedule.name] = bias
        split_results.append(
            {
                "schedule": schedule.name,
                "phases": [list(value) for value in schedule.phases],
                **fit_receipt,
                "validation_metrics": validation_metrics,
            }
        )

    selected = min(
        split_results,
        key=lambda value: value["validation_metrics"]["marginal_total_variation"],
    )
    selected_schedule = next(
        schedule for schedule in SCHEDULES if schedule.name == selected["schedule"]
    )
    final_bias, final_fit_receipt = _fit(
        scores[all_indices],
        teacher_all[all_indices],
        initial_bias,
        selected_schedule,
    )
    baseline_metrics, _ = _metrics(scores, initial_bias, teacher_all)
    final_metrics, _ = _metrics(scores, final_bias, teacher_all)
    delta = final_bias - initial_bias

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    save_file(
        {
            "initial_correction_bias": initial_bias.cpu().contiguous(),
            "correction_bias": final_bias.cpu().contiguous(),
        },
        str(temporary),
        metadata={
            "kind": "Kimi-K3 teacher-frequency-matched router correction bias",
            "layer": str(args.layer),
            "score_archive": str(args.scores.resolve()),
            "teacher_route_archive": str(args.teacher_routes.resolve()),
            "selected_schedule": selected_schedule.name,
        },
    )
    temporary.replace(args.output)
    report = {
        "kind": "Kimi-K3 router correction-bias frequency matching",
        "layer": args.layer,
        "score_archive": str(args.scores.resolve()),
        "teacher_route_archive": str(args.teacher_routes.resolve()),
        "student_route_archive": str(args.student_routes.resolve()),
        "boundary_archive": str(args.boundaries.resolve()),
        "token_count": int(scores.shape[0]),
        "document_split": {
            "fit": "even document indices",
            "validation": "odd document indices",
            "fit_tokens": int(fit_indices.numel()),
            "validation_tokens": int(validation_indices.numel()),
        },
        "baseline_score_route_mismatch_tokens": baseline_mismatch,
        "baseline_metrics": baseline_metrics,
        "schedule_selection": split_results,
        "selected_schedule": selected_schedule.name,
        "all_document_refit": final_fit_receipt,
        "all_document_metrics": final_metrics,
        "bias_update": {
            "l2": float(torch.linalg.vector_norm(delta)),
            "max_abs": float(delta.abs().max()),
            "mean": float(delta.mean()),
            "standard_deviation": float(delta.std()),
        },
        "output": str(args.output.resolve()),
        "qualification": "research-only until a disjoint full-model KLD screen passes",
    }
    _atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

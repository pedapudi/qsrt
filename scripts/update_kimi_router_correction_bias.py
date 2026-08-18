#!/usr/bin/env python3
"""Apply one margin-scaled teacher-frequency router-bias update."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _load_capture(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors = load_file(path, device="cpu")
    report = json.loads(path.with_suffix(".json").read_text())
    return tensors, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--step-fraction",
        type=float,
        default=1.0,
        help="maximum per-expert bias update as a fraction of the layer median margin",
    )
    parser.add_argument("--min-eta", type=float, default=1e-4)
    parser.add_argument("--max-eta", type=float, default=0.05)
    parser.add_argument("--tv-floor", type=float, default=2e-4)
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=2.5,
        help=(
            "zero per-expert frequency differences within this many conservative "
            "independent-binomial standard errors"
        ),
    )
    args = parser.parse_args()
    if (
        args.step_fraction <= 0
        or not 0 < args.min_eta <= args.max_eta
        or args.noise_sigma < 0
    ):
        raise ValueError("step and eta bounds must be positive")

    teacher, teacher_report = _load_capture(args.teacher.resolve())
    student, student_report = _load_capture(args.student.resolve())
    teacher_counts = teacher["selection_counts"].to(torch.int64)
    student_counts = student["selection_counts"].to(torch.int64)
    biases = student["biases"].to(torch.float32).clone()
    active = student["active_layers"].to(torch.bool)
    if teacher_counts.shape != student_counts.shape or biases.shape != teacher_counts.shape:
        raise ValueError("teacher and student captures have different layer geometry")
    if not torch.equal(active, teacher["active_layers"].to(torch.bool)):
        raise ValueError("teacher and student active-layer masks differ")
    teacher_population = teacher_report["population"]
    student_population = student_report["population"]
    if teacher_population["corpus_plan_sha256"] != student_population["corpus_plan_sha256"]:
        raise ValueError("teacher and student captures use different corpus populations")
    tokens = int(teacher_population["tokens"])
    top_k = int(teacher_report["decoder"]["top_k"])
    if tokens != int(student_population["tokens"]):
        raise ValueError("teacher and student token counts differ")
    if any(
        int(counts[layer].sum()) != tokens * top_k
        for counts in (teacher_counts, student_counts)
        for layer in torch.where(active)[0].tolist()
    ):
        raise ValueError("a router frequency row does not close")

    student_layers = {int(row["layer"]): row for row in student_report["layers"]}
    layer_records = []
    for layer in torch.where(active)[0].tolist():
        difference = (student_counts[layer] - teacher_counts[layer]).to(torch.float64)
        raw_gradient = difference / tokens
        teacher_probability = teacher_counts[layer].to(torch.float64) / tokens
        student_probability = student_counts[layer].to(torch.float64) / tokens
        standard_error = torch.sqrt(
            (
                teacher_probability * (1.0 - teacher_probability)
                + student_probability * (1.0 - student_probability)
            )
            / tokens
        )
        resolved = raw_gradient.abs() > args.noise_sigma * standard_error
        gradient = torch.where(resolved, raw_gradient, 0.0).to(torch.float32)
        tv = 0.5 * float(difference.abs().sum()) / (tokens * top_k)
        resolved_tv = 0.5 * float(
            torch.where(resolved, difference, 0.0).abs().sum()
        ) / (tokens * top_k)
        median_margin = float(student_layers[layer]["margin_p50"])
        max_gradient = float(gradient.abs().max())
        if resolved_tv <= args.tv_floor or max_gradient == 0:
            eta = 0.0
            update = torch.zeros_like(gradient)
        else:
            eta = args.step_fraction * median_margin / max_gradient
            eta = max(args.min_eta, min(args.max_eta, eta))
            update = -eta * gradient
            biases[layer] += update
        layer_records.append(
            {
                "layer": layer,
                "marginal_total_variation": tv,
                "noise_resolved_marginal_total_variation": resolved_tv,
                "noise_resolved_experts": int(resolved.sum()),
                "maximum_frequency_standard_error": float(standard_error.max()),
                "median_threshold_margin": median_margin,
                "max_abs_frequency_gradient": max_gradient,
                "eta_b": eta,
                "update_l2": float(torch.linalg.vector_norm(update)),
                "update_max_abs": float(update.abs().max()),
            }
        )

    active_records = [row for row in layer_records if row["eta_b"] > 0]
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    save_file(
        {
            "biases": biases.contiguous(),
            "active_layers": active.contiguous(),
        },
        str(temporary),
        metadata={
            "kind": "Kimi-K3 margin-scaled router correction biases",
            "teacher_frequency_capture": str(args.teacher.resolve()),
            "student_frequency_capture": str(args.student.resolve()),
            "corpus_plan_sha256": str(teacher_population["corpus_plan_sha256"]),
        },
    )
    os.replace(temporary, output)
    tv_values = [row["marginal_total_variation"] for row in layer_records]
    layer12 = next(row for row in layer_records if row["layer"] == 12)
    report = {
        "kind": "Kimi-K3 router correction-bias frequency-feedback update",
        "teacher_frequency_capture": str(args.teacher.resolve()),
        "student_frequency_capture": str(args.student.resolve()),
        "corpus_plan_sha256": teacher_population["corpus_plan_sha256"],
        "tokens": tokens,
        "documents": teacher_population["documents"],
        "rule": "bias -= eta_b * (student_frequency - teacher_frequency)",
        "eta_scaling": (
            "eta_b is chosen per layer so its largest proposed expert update equals "
            "step_fraction times that layer's median 16th-to-17th score margin, then "
            "clamped to the declared eta interval"
        ),
        "step_fraction": args.step_fraction,
        "min_eta": args.min_eta,
        "max_eta": args.max_eta,
        "tv_floor": args.tv_floor,
        "noise_sigma": args.noise_sigma,
        "noise_model": (
            "independent-binomial teacher/student frequency difference; conservative "
            "because both captures replay the same documents"
        ),
        "layers_updated": len(active_records),
        "mean_marginal_total_variation": sum(tv_values) / len(tv_values),
        "maximum_marginal_total_variation": max(tv_values),
        "layer_12": layer12,
        "layers": layer_records,
        "output": str(output),
    }
    _atomic_json(output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

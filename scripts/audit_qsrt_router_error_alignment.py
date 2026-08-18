#!/usr/bin/env python3
"""Measure boundary error aligned with a Kimi routed-expert gate row-space."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from qsrt.instanttensor_kimi import OfficialKimiLayerShards
from qsrt.kimi_boundary_slabs import KimiBoundarySlabArchive


DEFAULT_TEACHER = Path(
    "/data/kquant/research/qsrt-continuous-recovery-m0/teacher-legacy32-boundaries"
)
DEFAULT_STUDENT = Path(
    "/data/kquant/research/qsrt-continuous-recovery-m0/student-legacy32-boundaries"
)
DEFAULT_CHECKPOINT = Path(
    "/data/cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
)
DEFAULT_OUTPUT = Path(
    "/data/kquant/research/qsrt-continuous-recovery-m0/"
    "layer-012-router-error-alignment.json"
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


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    probabilities = torch.tensor(
        [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0], dtype=torch.float64
    )
    result = torch.quantile(values.to(torch.float64), probabilities)
    return {
        label: float(value)
        for label, value in zip(
            ("min", "p01", "p05", "median", "p95", "p99", "max"),
            result.tolist(),
            strict=True,
        )
    }


def _load_router_weight(checkpoint: Path, layer: int) -> tuple[str, torch.Tensor]:
    name = (
        f"language_model.model.layers.{layer}.block_sparse_moe.gate.weight"
    )
    index = OfficialKimiLayerShards(checkpoint)
    shard = index.tensor_shard(name)
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        value = handle.get_tensor(name).clone()
    if value.ndim != 2:
        raise ValueError(f"{name} is not a matrix")
    return name, value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--student", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.chunk_tokens <= 0:
        raise ValueError("chunk-tokens must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("device must be an indexed CUDA device")

    teacher = KimiBoundarySlabArchive(args.teacher)
    student = KimiBoundarySlabArchive(args.student)
    if (
        teacher.token_count != student.token_count
        or teacher.hidden_dimension != student.hidden_dimension
    ):
        raise ValueError("teacher and student boundary archives have different geometry")

    tensor_name, router_cpu = _load_router_weight(args.checkpoint, args.layer)
    output_dimension, hidden_dimension = map(int, router_cpu.shape)
    if hidden_dimension != teacher.hidden_dimension:
        raise ValueError("router input width differs from boundary width")

    torch.cuda.set_device(device)
    router = router_cpu.to(device=device, dtype=torch.float32)
    basis, upper = torch.linalg.qr(router.T, mode="reduced")
    diagonal = torch.diagonal(upper).abs()
    threshold = float(diagonal.max()) * max(router.shape) * torch.finfo(torch.float32).eps
    numerical_rank = int((diagonal > threshold).sum())
    if numerical_rank != output_dimension:
        basis = basis[:, :numerical_rank]
    rank = int(basis.shape[1])

    total_error_energy = 0.0
    total_projected_energy = 0.0
    squared_error_energy_sum = 0.0
    token_fractions: list[torch.Tensor] = []
    zero_error_tokens = 0
    for first in range(0, teacher.token_count, args.chunk_tokens):
        end = min(first + args.chunk_tokens, teacher.token_count)
        teacher_chunk = teacher.read_cpu(first_token=first, end_token=end, boundary=args.layer)
        student_chunk = student.read_cpu(first_token=first, end_token=end, boundary=args.layer)
        error = student_chunk.to(device=device, dtype=torch.float32)
        error.sub_(teacher_chunk.to(device=device, dtype=torch.float32))
        error_energy = error.square().sum(dim=1)
        projected_energy = (error @ basis).square().sum(dim=1)
        nonzero = error_energy > 0
        zero_error_tokens += int((~nonzero).sum())
        token_fractions.append(
            (projected_energy[nonzero] / error_energy[nonzero]).cpu()
        )
        total_error_energy += float(error_energy.sum(dtype=torch.float64))
        total_projected_energy += float(projected_energy.sum(dtype=torch.float64))
        squared_error_energy_sum += float(error_energy.square().sum(dtype=torch.float64))

    fractions = torch.cat(token_fractions)
    aggregate_fraction = total_projected_energy / total_error_energy
    isotropic_mean = rank / hidden_dimension
    isotropic_single_direction_std = math.sqrt(
        2.0
        * rank
        * (hidden_dimension - rank)
        / (hidden_dimension**2 * (hidden_dimension + 2))
    )
    effective_token_count = (
        total_error_energy**2 / squared_error_energy_sum
        if squared_error_energy_sum
        else 0.0
    )
    report = {
        "kind": "Kimi-K3 router row-space boundary-error alignment",
        "layer": args.layer,
        "boundary": args.layer,
        "teacher_archive": str(args.teacher.resolve()),
        "student_archive": str(args.student.resolve()),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "router_tensor": tensor_name,
        "router_shape": [output_dimension, hidden_dimension],
        "router_numerical_rank": rank,
        "qr_rank_threshold": threshold,
        "qr_diagonal_min": float(diagonal.min()),
        "qr_diagonal_max": float(diagonal.max()),
        "token_count": teacher.token_count,
        "zero_error_tokens": zero_error_tokens,
        "total_error_energy": total_error_energy,
        "total_projected_energy": total_projected_energy,
        "aggregate_projected_fraction": aggregate_fraction,
        "per_token_projected_fraction": _quantiles(fractions),
        "isotropic_random_direction_baseline": {
            "expected_projected_fraction": isotropic_mean,
            "single_direction_standard_deviation": isotropic_single_direction_std,
            "error_energy_effective_token_count": effective_token_count,
            "aggregate_to_expected_ratio": aggregate_fraction / isotropic_mean,
        },
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

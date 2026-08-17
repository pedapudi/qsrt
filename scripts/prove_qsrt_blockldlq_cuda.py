#!/usr/bin/env python3
"""Prove the production QSRT Hessian congruence on CUDA.

The encoder quantizes a Hadamard- and scale-conditioned weight.  Its dense-H
metric must therefore be transformed by the same complete input conditioning,
including the nonuniform magnitudes chosen by ``regularize``.  This witness
calls the production preparation/factorization code and compares its work-space
quadratic form with the corresponding canonical-space quadratic form.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qsrt.exl3_loader import load_qsrt_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--sigma-reg", type=float, default=0.025)
    parser.add_argument(
        "--exllamav3-root", type=Path, default=Path("/home/luke/projects/exllamav3")
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the production congruence witness")
    device = torch.device(args.device)
    backend = load_qsrt_encoder(args.exllamav3_root)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    dimension = 128
    outputs = 96
    rows = 521
    samples = torch.randn(
        rows, dimension, generator=generator, device=device, dtype=torch.float32
    )
    capture_h = samples.T @ samples
    count = float(rows)
    normalized_h = capture_h / count
    normalized_h = normalized_h.clone()
    normalized_h.diagonal().add_(args.sigma_reg * normalized_h.diagonal().mean())

    # A deliberately nonuniform positive conditioning vector makes a missing
    # magnitude congruence easy to detect; signs are included as in production.
    signs = torch.where(
        torch.arange(dimension, device=device) % 3 == 0,
        -torch.ones((), device=device),
        torch.ones((), device=device),
    )
    magnitudes = torch.linspace(0.35, 2.4, dimension, device=device)
    conditioning = (signs * magnitudes).float().unsqueeze(1)
    work_error = torch.randn(
        dimension, outputs, generator=generator, device=device, dtype=torch.float32
    )

    h_data = {
        "H": capture_h.clone(),
        "count": count,
        "finalized": False,
        "device": device,
    }
    quant_args = {"sigma_reg": args.sigma_reg}
    fallback, _, _, _ = backend.prepare_capture_H_for_conditioning(
        h_data, quant_args, False
    )
    if fallback:
        raise AssertionError("the positive-definite proof Hessian fell back")
    fallback, transformed_h, lower, _, _ = (
        backend.finalize_capture_H_with_conditioning(
            h_data, conditioning, quant_args, False
        )
    )
    if fallback or lower is None:
        raise AssertionError("production BlockLDLQ factorization fell back")
    transformed_h = transformed_h.to(device)

    canonical_error = backend.preapply_had_l(work_error, backend.had_k)
    canonical_error *= conditioning
    expected = torch.einsum(
        "ik,ij,jk->", canonical_error, normalized_h, canonical_error
    )
    actual = torch.einsum("ik,ij,jk->", work_error, transformed_h, work_error)
    relative_error = float((actual - expected).abs() / expected.abs().clamp_min(1e-30))
    if relative_error > 2e-6:
        raise AssertionError(
            f"conditioned dense-H metric mismatch: relative error {relative_error:.3e}"
        )

    receipt = {
        "kind": "qsrt_blockldlq_cuda_congruence_proof",
        "schema_version": 1,
        "device": str(device),
        "seed": args.seed,
        "dimension": dimension,
        "outputs": outputs,
        "rows": rows,
        "sigma_reg": args.sigma_reg,
        "conditioning_min_abs": float(conditioning.abs().min()),
        "conditioning_max_abs": float(conditioning.abs().max()),
        "canonical_metric": float(expected),
        "work_metric": float(actual),
        "relative_error": relative_error,
        "lower_shape": list(lower.shape),
        "passed": True,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(json.dumps(receipt, indent=1, sort_keys=True))
        temporary.replace(args.output)
    print(json.dumps(receipt, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate two-sided curvature and frozen-scale feedback on one CUDA matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qsrt.exl3_loader import load_qsrt_encoder
from qsrt.exl3_reference import CODEBOOK_SQG_XOR_CHEB_T12
from qsrt.glm52_pilot import atomic_write_json
from qsrt.qsrt_codec_pilot import encode_uniform_candidate, tensor_sha256
from qsrt.sqg_quantizer import install_sqg_quantizer


def _spd(dimension: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    basis = torch.randn((dimension, 12), generator=generator)
    metric = basis @ basis.T / basis.shape[1]
    metric.diagonal().add_(0.25 * metric.diagonal().mean())
    return metric.contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--exllamav3-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.dest.exists():
        raise FileExistsError(args.dest)
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(20_260_817)
    source = torch.randn((128, 128), generator=generator, dtype=torch.float32)
    input_metric = _spd(128, seed=131)
    output_metric = _spd(128, seed=197)
    quantizer = load_qsrt_encoder(args.exllamav3_root)
    install_sqg_quantizer(quantizer)
    common = {
        "source": source,
        "bits": 3,
        "codebook": CODEBOOK_SQG_XOR_CHEB_T12,
        "device": device,
        "quantizer_module": quantizer,
        "input_sign_seed": 17,
        "output_sign_seed": 29,
        "tailbite_context": 128,
        "ldlq_tf32": True,
        "return_trellis_diagnostics": True,
    }
    ordinary = encode_uniform_candidate(**common)
    global_scale = ordinary["payload"]["g_scale"]
    ordinary_frozen = encode_uniform_candidate(
        **common,
        g_scale_override=global_scale,
    )
    two_sided = encode_uniform_candidate(
        **common,
        g_scale_override=global_scale,
        input_hessian=input_metric,
        output_hessian=output_metric,
    )
    direct = encode_uniform_candidate(
        **common,
        g_scale_override=global_scale,
        ldlq_feedback_multiplier=0.0,
    )
    scale_keys = ("suh_sha256", "svh_sha256", "scale_bytes", "g_scale")
    scale_closure = {
        "ordinary_frozen": all(
            ordinary["payload"][key] == ordinary_frozen["payload"][key]
            for key in scale_keys
        ),
        "two_sided": all(
            ordinary["payload"][key] == two_sided["payload"][key]
            for key in scale_keys
        ),
        "direct": all(
            ordinary["payload"][key] == direct["payload"][key]
            for key in scale_keys
        ),
    }
    reconstruction_hashes = {
        name: tensor_sha256(result["reconstruction"])
        for name, result in (
            ("ordinary", ordinary),
            ("ordinary_frozen", ordinary_frozen),
            ("two_sided", two_sided),
            ("direct", direct),
        )
    }
    report = {
        "schema": "qsrt_two_sided_curvature_cuda_closure",
        "schema_version": 1,
        "status": "passed",
        "device": str(device),
        "source_shape": list(source.shape),
        "source_sha256": tensor_sha256(source),
        "input_metric_sha256": tensor_sha256(input_metric),
        "output_metric_sha256": tensor_sha256(output_metric),
        "global_scale": global_scale,
        "scale_closure": scale_closure,
        "ordinary_replay_exact": (
            reconstruction_hashes["ordinary"]
            == reconstruction_hashes["ordinary_frozen"]
        ),
        "reconstruction_sha256": reconstruction_hashes,
        "path_changed": {
            "two_sided": (
                ordinary["payload"]["trellis_sha256"]
                != two_sided["payload"]["trellis_sha256"]
            ),
            "direct": (
                ordinary["payload"]["trellis_sha256"]
                != direct["payload"]["trellis_sha256"]
            ),
        },
        "proxy_relative_error": {
            name: result["payload"]["proxy_relative_error"]
            for name, result in (
                ("ordinary", ordinary),
                ("two_sided", two_sided),
                ("direct", direct),
            )
        },
        "trellis_diagnostics": {
            name: result["trellis_diagnostics"]
            for name, result in (
                ("ordinary", ordinary),
                ("two_sided", two_sided),
                ("direct", direct),
            )
        },
        "evidence_boundary": (
            "synthetic 128-by-128 CUDA closure; no GLM source weight, routed "
            "gradient factor, expert output, or full-model KLD was measured"
        ),
    }
    if not all(scale_closure.values()) or not report["ordinary_replay_exact"]:
        report["status"] = "failed"
        atomic_write_json(args.dest, report)
        raise RuntimeError("two-sided CUDA scale or replay closure failed")
    atomic_write_json(args.dest, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

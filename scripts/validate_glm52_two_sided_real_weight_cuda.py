#!/usr/bin/env python3
"""Run two-sided K3 closure on one complete real GLM-5.2 gate matrix."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from qsrt.exl3_loader import load_qsrt_encoder
from qsrt.exl3_reference import CODEBOOK_SQG_XOR_CHEB_T12
from qsrt.glm52_expert_intervention import INTERVENTION_ARTIFACT_KIND
from qsrt.glm52_expert_intervention_runtime import (
    validate_dense_intervention_artifact,
)
from qsrt.glm52_pilot import (
    PROJECTIONS,
    IndexedTensorStore,
    _transform_seeds,
    atomic_write_json,
    source_tensor_name,
)
from qsrt.glm52_real_weight_benchmark import validate_bounded_source_window
from qsrt.glm52_two_sided_curvature import (
    _baseline_global_scale,
    _scale_vectors_close,
)
from qsrt.ldlq import SIGMA_REG
from qsrt.qsrt_codec_pilot import encode_uniform_candidate, tensor_sha256
from qsrt.sqg_quantizer import install_sqg_quantizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-inventory", required=True, type=Path)
    parser.add_argument("--input-artifact", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--exllamav3-root", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-source-shard-hashes", action="store_true")
    args = parser.parse_args()
    if args.dest.exists():
        raise FileExistsError(args.dest)
    device = torch.device(args.device)
    validate_bounded_source_window(
        args.source_root,
        args.source_inventory,
        panel={args.layer: (args.expert,)},
        verify_shard_hashes=not args.skip_source_shard_hashes,
    )
    artifact = validate_dense_intervention_artifact(args.input_artifact)
    if args.expert not in artifact["expert_ids"]:
        raise ValueError("input artifact does not contain the requested expert")
    record = next(
        item
        for item in artifact["report"]["experts"]
        if int(item["expert"]) == args.expert
    )
    endpoint_path = args.input_artifact / "experts" / record["dense_endpoint_file"]
    with safe_open(endpoint_path, framework="pt", device="cpu") as handle:
        stored_uniform = handle.get_tensor("qsrt_k3.gate_proj")
    source_store = IndexedTensorStore(args.source_root)
    source = source_store.get(
        source_tensor_name(args.layer, args.expert, "gate_proj")
    )
    gate_spec = PROJECTIONS[0]
    input_seed, output_seed = _transform_seeds(args.layer, gate_spec)
    global_scale = _baseline_global_scale(record, "gate_proj")
    quantizer = load_qsrt_encoder(args.exllamav3_root)
    install_sqg_quantizer(quantizer)
    common = {
        "source": source,
        "bits": 3,
        "codebook": CODEBOOK_SQG_XOR_CHEB_T12,
        "device": device,
        "quantizer_module": quantizer,
        "input_sign_seed": input_seed,
        "output_sign_seed": output_seed,
        "scale_scope_key": (
            INTERVENTION_ARTIFACT_KIND,
            args.layer,
            args.expert,
            "gate_proj",
        ),
        "g_scale_into_sv": True,
        "sigma_reg": SIGMA_REG,
        "tailbite_context": 128,
        "ldlq_tf32": True,
        "g_scale_override": global_scale,
    }
    torch.cuda.reset_peak_memory_stats()
    ordinary_started = time.monotonic()
    ordinary = encode_uniform_candidate(**common)
    ordinary_wall_seconds = time.monotonic() - ordinary_started
    ordinary_peak_memory_bytes = torch.cuda.max_memory_allocated()
    if not torch.equal(ordinary["reconstruction"].half(), stored_uniform.half()):
        raise RuntimeError("ordinary frozen-scale gate control did not replay")
    input_metric = torch.eye(source.shape[1], dtype=torch.float32)
    output_metric = torch.eye(source.shape[0], dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats()
    two_sided_started = time.monotonic()
    two_sided = encode_uniform_candidate(
        **common,
        input_hessian=input_metric,
        output_hessian=output_metric,
    )
    two_sided_wall_seconds = time.monotonic() - two_sided_started
    two_sided_peak_memory_bytes = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    traversal_control_started = time.monotonic()
    traversal_control = encode_uniform_candidate(
        **common,
        input_hessian=input_metric,
        use_two_sided_traversal_without_output_feedback=True,
    )
    traversal_control_wall_seconds = (
        time.monotonic() - traversal_control_started
    )
    traversal_control_peak_memory_bytes = torch.cuda.max_memory_allocated()
    scale_closed = _scale_vectors_close(
        ordinary["payload"], two_sided["payload"]
    )
    traversal_control_scale_closed = _scale_vectors_close(
        ordinary["payload"], traversal_control["payload"]
    )
    traversal_control_path_exact = (
        ordinary["payload"]["trellis_sha256"]
        == traversal_control["payload"]["trellis_sha256"]
    )
    traversal_control_reconstruction_exact = torch.equal(
        ordinary["reconstruction"], traversal_control["reconstruction"]
    )
    source_float = source.float()
    source_energy = float(
        torch.sum(source_float.square(), dtype=torch.float64).item()
    )
    ordinary_source_sse = float(
        torch.sum(
            (source_float - ordinary["reconstruction"].float()).square(),
            dtype=torch.float64,
        ).item()
    )
    two_sided_source_sse = float(
        torch.sum(
            (source_float - two_sided["reconstruction"].float()).square(),
            dtype=torch.float64,
        ).item()
    )
    report = {
        "schema": "qsrt_glm52_two_sided_real_weight_cuda_closure",
        "schema_version": 1,
        "status": "passed",
        "layer": args.layer,
        "expert": args.expert,
        "projection": "gate_proj",
        "source_tensor": source_tensor_name(
            args.layer, args.expert, "gate_proj"
        ),
        "source_shape": list(source.shape),
        "source_sha256": tensor_sha256(source),
        "input_metric": "identity_control",
        "output_metric": "source_basis_identity_control",
        "source_basis_identity_output_metric_maps_to_work_basis": (
            "output Hadamard and persisted nonuniform output scales apply a "
            "congruence transform, so source-basis identity is generally not "
            "the ordinary encoder's zero-output-feedback metric"
        ),
        "frozen_global_scale": global_scale,
        "ordinary_replay_exact": True,
        "scale_plane_closed": scale_closed,
        "path_changed": (
            ordinary["payload"]["trellis_sha256"]
            != two_sided["payload"]["trellis_sha256"]
        ),
        "ordinary_dense_tensor_sha256": tensor_sha256(
            ordinary["reconstruction"]
        ),
        "two_sided_dense_tensor_sha256": tensor_sha256(
            two_sided["reconstruction"]
        ),
        "zero_output_feedback_traversal_control": {
            "scale_plane_closed": traversal_control_scale_closed,
            "trellis_path_exact": traversal_control_path_exact,
            "reconstruction_exact": traversal_control_reconstruction_exact,
            "dense_tensor_sha256": tensor_sha256(
                traversal_control["reconstruction"]
            ),
            "payload": traversal_control["payload"],
            "wall_seconds": traversal_control_wall_seconds,
            "peak_cuda_memory_bytes": traversal_control_peak_memory_bytes,
        },
        "ordinary_proxy_relative_error": ordinary["payload"][
            "proxy_relative_error"
        ],
        "two_sided_proxy_relative_error": two_sided["payload"][
            "proxy_relative_error"
        ],
        "source_energy": source_energy,
        "ordinary_source_sse": ordinary_source_sse,
        "two_sided_source_sse": two_sided_source_sse,
        "ordinary_source_relative_sse": ordinary_source_sse / source_energy,
        "two_sided_source_relative_sse": two_sided_source_sse / source_energy,
        "ordinary_wall_seconds": ordinary_wall_seconds,
        "two_sided_wall_seconds": two_sided_wall_seconds,
        "ordinary_peak_cuda_memory_bytes": ordinary_peak_memory_bytes,
        "two_sided_peak_cuda_memory_bytes": two_sided_peak_memory_bytes,
        "cuda_device_index": torch.cuda.current_device(),
        "cuda_device_name": torch.cuda.get_device_name(),
        "timing_boundary": (
            "the ordinary arm initializes and warms the encoder before the "
            "two-sided arm, so wall times are dimensional closure data rather "
            "than a performance comparison"
        ),
        "ordinary_payload": ordinary["payload"],
        "two_sided_payload": two_sided["payload"],
        "evidence_boundary": (
            "complete real GLM-5.2 source matrix and production K3 codec, but "
            "identity output curvature; this tests dimensions and CUDA closure, "
            "not downstream-loss prediction or full-model KLD"
        ),
    }
    if not scale_closed:
        report["status"] = "failed"
        atomic_write_json(args.dest, report)
        raise RuntimeError("two-sided real-weight scale closure failed")
    if not (
        traversal_control_scale_closed
        and traversal_control_path_exact
        and traversal_control_reconstruction_exact
    ):
        report["status"] = "failed"
        atomic_write_json(args.dest, report)
        raise RuntimeError(
            "zero-output-feedback two-sided traversal did not reproduce "
            "ordinary BlockLDLQ"
        )
    atomic_write_json(args.dest, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "source_shape",
                    "ordinary_replay_exact",
                    "scale_plane_closed",
                    "path_changed",
                    "ordinary_proxy_relative_error",
                    "two_sided_proxy_relative_error",
                    "ordinary_source_relative_sse",
                    "two_sided_source_relative_sse",
                    "zero_output_feedback_traversal_control",
                    "ordinary_wall_seconds",
                    "two_sided_wall_seconds",
                    "ordinary_peak_cuda_memory_bytes",
                    "two_sided_peak_cuda_memory_bytes",
                    "evidence_boundary",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

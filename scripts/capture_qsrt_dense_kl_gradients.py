#!/usr/bin/env python
"""Capture dense final-KL gradients for selected QSRT expert layers."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from instanttensor import Backend

_REPOSITORY = Path(__file__).resolve().parents[1]
if str(_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY))

from qsrt.instanttensor_kimi import InstantTensorLoadConfig
from qsrt.kimi_boundary_slabs import KimiBoundarySlabArchive
from qsrt.kimi_cotangent_slabs import KimiCotangentSlabWorkspace
from qsrt.kimi_dense_gradient_replay import KimiDenseObjectiveGradientReplay
from qsrt.kimi_dense_objective_gradients import DenseObjectiveGradientArchive
from qsrt.kimi_official_forward import load_official_kimi_runtime
from qsrt.kimi_quantized_forward import QSRTAnchorPayload, QSRTKimiForwardAdapter

from scripts.capture_kimi_k3_upstream_fisher import (
    DEFAULT_CODE_CHECKPOINT,
    DEFAULT_PROFILE,
    DEFAULT_WEIGHT_CHECKPOINT,
    _load_profile_draws,
    _parse_devices,
    _parse_instanttensor_backend,
    _routed_geometry,
    _sha256,
)


DEFAULT_BOUNDARIES = Path(
    "/data/datasets/kquant/captures/"
    "k3-qsrt-k2-direct-viterbi-final-kl-gradient-100k-v1-boundaries"
)
DEFAULT_OBJECTIVE_COTANGENTS = Path(
    "/data/kquant/research/"
    "k3-qsrt-k2-direct-viterbi-final-kl-gradient-100k-v1-objective-cotangents"
)
DEFAULT_GRADIENTS = Path(
    "/data/kquant/research/qsrt-fp32-kl-refinement-layers89-92/gradients"
)
DEFAULT_MODEL = Path(
    "/data/models/Kimi-K3-QSRT-K2-DIRECT-VITERBI-ALL-LINEARS-v1-model"
)
DEFAULT_CANDIDATES = Path(
    "/data/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-CANDIDATES-v1"
)
DEFAULT_UPSTREAM_OVERLAYS = Path(
    "/data/kquant/research/"
    "k3-uniform-k2-direct-viterbi-all-linears-v1/upstream-overlays"
)
DEFAULT_W2_OVERLAYS = Path(
    "/data/kquant/research/"
    "k3-uniform-k2-direct-viterbi-all-linears-v1/w2-overlays"
)


def _parse_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(sorted({int(item) for item in value.split(",") if item}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from error
    if not layers:
        raise argparse.ArgumentTypeError("at least one target layer is required")
    return layers


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARIES)
    parser.add_argument(
        "--objective-cotangents",
        type=Path,
        default=DEFAULT_OBJECTIVE_COTANGENTS,
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_GRADIENTS)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--overlay-root",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--weight-checkpoint", type=Path, default=DEFAULT_WEIGHT_CHECKPOINT)
    parser.add_argument("--code-checkpoint", type=Path, default=DEFAULT_CODE_CHECKPOINT)
    parser.add_argument("--layers", type=_parse_layers, default=(89, 90, 91, 92))
    parser.add_argument("--shard-experts", type=int, default=112)
    parser.add_argument(
        "--devices",
        type=_parse_devices,
        default=_parse_devices(",".join(str(index) for index in range(9))),
    )
    parser.add_argument("--pipeline-queue-depth", type=int, default=1)
    parser.add_argument("--validation-documents", type=int, default=1)
    parser.add_argument("--instant-buffer-gib", type=int, default=4)
    parser.add_argument(
        "--instant-backend",
        type=_parse_instanttensor_backend,
        default=Backend.CUFILE,
    )
    parser.add_argument("--instant-chunk-mib", type=int, default=128)
    parser.add_argument("--instant-concurrency", type=int, default=4)
    parser.add_argument("--instant-io-depth", type=int, default=8)
    parser.add_argument("--buffered-io", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if (
        args.shard_experts <= 0
        or args.pipeline_queue_depth <= 0
        or args.validation_documents < 0
        or args.instant_buffer_gib <= 0
        or args.instant_chunk_mib <= 0
        or args.instant_concurrency <= 0
        or args.instant_io_depth <= 0
    ):
        raise ValueError("capture dimensions and buffer sizes are invalid")
    boundaries = KimiBoundarySlabArchive(
        args.boundaries.expanduser().resolve(),
        require_complete=True,
    )
    objective = KimiCotangentSlabWorkspace(
        args.objective_cotangents.expanduser().resolve()
    )
    runtime = load_official_kimi_runtime(
        weight_checkpoint=args.weight_checkpoint,
        code_checkpoint=args.code_checkpoint,
    )
    routed_layers, num_experts, hidden, intermediate = _routed_geometry(runtime)
    if num_experts % args.shard_experts:
        raise ValueError("expert shard width must divide the routed-expert count")
    if any(layer not in routed_layers for layer in args.layers):
        raise ValueError("every target must be a routed expert layer")
    draws, atoms_profile, completion_sha256 = _load_profile_draws(
        args.profile.expanduser().resolve(),
        layers=routed_layers,
        num_experts=num_experts,
    )
    overlays = tuple(
        path.expanduser().resolve()
        for path in (
            args.overlay_root
            if args.overlay_root is not None
            else (DEFAULT_UPSTREAM_OVERLAYS, DEFAULT_W2_OVERLAYS)
        )
    )
    model = args.model.expanduser().resolve()
    candidate_pool = args.candidate_pool.expanduser().resolve()
    quantized_anchor = {
        "model_checkpoint": str(model),
        "candidate_pool": str(candidate_pool),
        "overlay_roots": [str(path) for path in overlays],
    }
    provenance = boundaries.manifest.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("quantized_anchor") != quantized_anchor:
        raise ValueError("boundary activations belong to another quantized anchor")
    if objective.manifest.get("chain_boundary") != boundaries.num_layers:
        raise ValueError("objective cotangents are not at the final decoder boundary")

    destination = args.dest.expanduser().resolve()
    if destination.exists():
        archive = DenseObjectiveGradientArchive(destination)
    else:
        archive = DenseObjectiveGradientArchive.create(
            destination,
            layers=args.layers,
            num_experts=num_experts,
            shard_experts=args.shard_experts,
            hidden_dimension=hidden,
            intermediate_dimension=intermediate,
            objective_normalizer=float(boundaries.token_count),
            provenance={
                "objective": "token-summed KL(official MXFP4 || direct-Viterbi K2)",
                "boundary_archive": str(boundaries.root),
                "boundary_manifest_sha256": _sha256(boundaries.manifest_path),
                "objective_cotangent_workspace": str(objective.root),
                "objective_cotangent_manifest_sha256": _sha256(
                    objective.manifest_path
                ),
                "quantized_anchor": quantized_anchor,
                "official_revision": runtime.weight_checkpoint.name,
                "profile": str(args.profile.expanduser().resolve()),
                "profile_completion_sha256": completion_sha256,
                "atoms_profile": atoms_profile,
            },
        )
    expected_archive = {
        "layers": list(args.layers),
        "num_experts": num_experts,
        "shard_experts": args.shard_experts,
        "hidden_dimension": hidden,
        "intermediate_dimension": intermediate,
    }
    for key, expected in expected_archive.items():
        if archive.manifest.get(key) != expected:
            raise ValueError(
                f"gradient archive {key} differs: "
                f"stored={archive.manifest.get(key)!r}, requested={expected!r}"
            )

    load_config = InstantTensorLoadConfig(
        buffer_size=args.instant_buffer_gib << 30,
        chunk_size=args.instant_chunk_mib << 20,
        concurrency=args.instant_concurrency,
        io_depth=args.instant_io_depth,
        backend=args.instant_backend,
    )
    adapter = QSRTKimiForwardAdapter(
        runtime,
        model_checkpoint=model,
        expert_payload=QSRTAnchorPayload(candidate_pool, overlay_roots=overlays),
        load_config=load_config,
        validate_outputs=False,
    )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    result = KimiDenseObjectiveGradientReplay(
        adapter=adapter,
        boundary_archive=boundaries,
        objective_workspace=objective,
        gradient_archive=archive,
        intermediate_draws={layer: draws[layer] for layer in args.layers},
        target_layers=args.layers,
        devices=args.devices,
        queue_depth=args.pipeline_queue_depth,
        direct_io=not args.buffered_io,
        validation_documents=args.validation_documents,
    ).run()
    record = {
        "complete": True,
        "elapsed_seconds": time.monotonic() - started,
        "result": asdict(result),
    }
    _atomic_json(destination / "dense-gradient-run.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

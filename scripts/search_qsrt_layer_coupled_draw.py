#!/usr/bin/env python3
"""Encode one layer under one coupled-Hadamard draw and measure distortion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

from qsrt import constants as C
from qsrt.exl3_loader import load_qsrt_encoder
from qsrt.instanttensor_kimi import InstantTensorKimiLayerLoader
from qsrt.pack.qsrt_encoder import qsrt_transform_seed_draw
from qsrt.qsrt import matrix_rate_axis
from qsrt.qsrt_coupled import CoupledHadamardSpec, encode_coupled_weights
from qsrt.source_weights import OfficialMXFP4Store
from qsrt.sqg_quantizer import install_sqg_quantizer
from qsrt.two_sided_qsrt import encode_uniform_sqg_direct_batch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--draw", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--exllamav3-root",
        type=Path,
        default=Path("/home/luke/projects/exllamav3"),
    )
    parser.add_argument("--expert-batch-size", type=int, default=4)
    parser.add_argument("--expert-limit", type=int)
    parser.add_argument("--tailbite-context", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def at(fraction: float) -> float:
        return ordered[round(fraction * (len(ordered) - 1))]

    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p90": at(0.90),
        "p95": at(0.95),
        "p99": at(0.99),
        "maximum": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def main() -> None:
    args = _parser().parse_args()
    if args.layer not in C.MOE_LAYERS:
        raise ValueError("--layer must identify a routed decoder layer")
    if not 0 <= args.draw < 8:
        raise ValueError("--draw must lie in 0..7")
    if args.expert_batch_size < 1:
        raise ValueError("--expert-batch-size must be positive")
    if args.expert_limit is not None and not 1 <= args.expert_limit <= C.NUM_EXPERTS:
        raise ValueError("--expert-limit must lie in 1..896")
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("--device must be an indexed CUDA device")

    started = time.monotonic()
    source_store = OfficialMXFP4Store()
    loader = InstantTensorKimiLayerLoader(source_store.root, device=device)
    banks, load_stats = loader.load_expert_banks(
        layer=args.layer,
        matrices=("w1", "w3", "w2"),
    )
    backend = load_qsrt_encoder(args.exllamav3_root)
    install_sqg_quantizer(backend)
    expert_count = args.expert_limit or C.NUM_EXPERTS
    records: list[dict[str, object]] = []
    totals = {
        matrix: {"sse": 0.0, "source_energy": 0.0}
        for matrix in C.EXPERT_MATRICES
    }

    for begin in range(0, expert_count, args.expert_batch_size):
        experts = tuple(
            range(begin, min(begin + args.expert_batch_size, expert_count))
        )
        transformed = {matrix: [] for matrix in C.EXPERT_MATRICES}
        for expert in experts:
            triplet = encode_coupled_weights(
                tuple(banks[matrix][expert] for matrix in C.EXPERT_MATRICES),
                CoupledHadamardSpec(intermediate_draw=args.draw),
            )
            for matrix, source in zip(C.EXPERT_MATRICES, triplet, strict=True):
                transformed[matrix].append(source)

        batch_records = {
            expert: {
                "expert": expert,
                "draw": args.draw,
                "sse": 0.0,
                "source_energy": 0.0,
                "matrices": {},
            }
            for expert in experts
        }
        for matrix in C.EXPERT_MATRICES:
            sources = torch.stack(transformed[matrix])
            seeds = qsrt_transform_seed_draw(args.layer, matrix)
            results = encode_uniform_sqg_direct_batch(
                sources,
                bits=2,
                device=device,
                quantizer_module=backend,
                input_sign_seed=seeds.input_sign,
                output_sign_seed=seeds.output_sign,
                rate_axis=matrix_rate_axis(matrix),
                scale_scope_key=(
                    ("direct-viterbi", args.layer, matrix)
                    if matrix in ("w1", "w3")
                    else None
                ),
                shared_scale_axis=(
                    "input" if matrix in ("w1", "w3") else None
                ),
                tailbite_context=args.tailbite_context,
            )
            for expert, source, result in zip(experts, sources, results, strict=True):
                reconstruction = result.candidate.reconstruction.float()
                target = source.float()
                error = reconstruction - target
                sse = float(error.square().double().sum().item())
                energy = float(target.square().double().sum().item())
                matrix_record = {
                    "sse": sse,
                    "source_energy": energy,
                    "relative_sse": sse / energy,
                    "relative_l2": (sse / energy) ** 0.5,
                    "global_scale": result.global_scale,
                }
                batch_records[expert]["matrices"][matrix] = matrix_record
                batch_records[expert]["sse"] += sse
                batch_records[expert]["source_energy"] += energy
                totals[matrix]["sse"] += sse
                totals[matrix]["source_energy"] += energy
            del sources, results

        for expert in experts:
            record = batch_records[expert]
            record["relative_sse"] = record["sse"] / record["source_energy"]
            record["relative_l2"] = record["relative_sse"] ** 0.5
            records.append(record)
        if len(records) % 64 == 0:
            print(
                json.dumps(
                    {
                        "layer": args.layer,
                        "draw": args.draw,
                        "encoded_experts": len(records),
                        "experts": expert_count,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    total_sse = sum(value["sse"] for value in totals.values())
    total_energy = sum(value["source_energy"] for value in totals.values())
    for value in totals.values():
        value["relative_sse"] = value["sse"] / value["source_energy"]
        value["relative_l2"] = value["relative_sse"] ** 0.5
    report = {
        "kind": "qsrt_uniform_k2_coupled_draw_search",
        "layer": args.layer,
        "draw": args.draw,
        "experts": expert_count,
        "source_checkpoint": str(source_store.root),
        "source_load": {
            "serialized_bytes": load_stats.serialized_bytes,
            "dense_bytes": load_stats.dense_bytes,
            "seconds": load_stats.elapsed_seconds,
        },
        "total": {
            "sse": total_sse,
            "source_energy": total_energy,
            "relative_sse": total_sse / total_energy,
            "relative_l2": (total_sse / total_energy) ** 0.5,
        },
        "matrices": totals,
        "expert_relative_sse": _quantiles(
            [float(record["relative_sse"]) for record in records]
        ),
        "expert_sse": _quantiles([float(record["sse"]) for record in records]),
        "elapsed_seconds": time.monotonic() - started,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "records"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

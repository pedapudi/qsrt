#!/usr/bin/env python3
"""Measure source-to-decoded distortion for one uniform-K2 expert layer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

from safetensors import safe_open
import torch

from qsrt import constants as C
from qsrt.instanttensor_kimi import InstantTensorKimiLayerLoader
from qsrt.pack.qsrt_validation import decode_candidate_matrix
from qsrt.qsrt import K2
from qsrt.qsrt_coupled import CoupledHadamardSpec, encode_coupled_weights
from qsrt.source_weights import OfficialMXFP4Store


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--upstream-overlay", type=Path, required=True)
    parser.add_argument("--down-overlay", type=Path, required=True)
    parser.add_argument("--draw-result", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
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


def _draws(path: Path, layer: int) -> tuple[int, ...]:
    result = json.loads(path.read_text())
    if int(result.get("layer", -1)) != layer:
        raise ValueError("draw result belongs to a different layer")
    experts = result.get("experts")
    if not isinstance(experts, dict) or len(experts) != C.NUM_EXPERTS:
        raise ValueError("draw result does not contain all experts")
    return tuple(int(experts[str(expert)]["intermediate_draw"]) for expert in range(C.NUM_EXPERTS))


def main() -> None:
    args = _parser().parse_args()
    if args.layer not in C.MOE_LAYERS:
        raise ValueError("--layer must identify a routed decoder layer")
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("--device must be an indexed CUDA device")
    draws = _draws(args.draw_result, args.layer)

    started = time.monotonic()
    source_store = OfficialMXFP4Store()
    loader = InstantTensorKimiLayerLoader(source_store.root, device=device)
    banks, load_stats = loader.load_expert_banks(
        layer=args.layer,
        matrices=("w1", "w3", "w2"),
    )
    records: list[dict[str, object]] = []
    totals = {
        matrix: {"sse": 0.0, "source_energy": 0.0}
        for matrix in C.EXPERT_MATRICES
    }

    with safe_open(args.upstream_overlay, framework="pt", device="cpu") as upstream:
        with safe_open(args.down_overlay, framework="pt", device="cpu") as down:
            for expert in range(C.NUM_EXPERTS):
                transformed = encode_coupled_weights(
                    tuple(banks[matrix][expert] for matrix in C.EXPERT_MATRICES),
                    CoupledHadamardSpec(intermediate_draw=draws[expert]),
                )
                matrix_records: dict[str, dict[str, float]] = {}
                expert_sse = 0.0
                expert_energy = 0.0
                for matrix, source in zip(C.EXPERT_MATRICES, transformed, strict=True):
                    decoded = decode_candidate_matrix(
                        upstream if matrix in ("w1", "w3") else down,
                        layer=args.layer,
                        expert=expert,
                        matrix=matrix,
                        mode_id=K2.mode_id,
                        device=device,
                    )
                    source_physical = source.T.float()
                    error = decoded.float() - source_physical
                    sse = float(error.square().double().sum().item())
                    energy = float(source_physical.square().double().sum().item())
                    matrix_records[matrix] = {
                        "sse": sse,
                        "source_energy": energy,
                        "relative_sse": sse / energy,
                        "relative_l2": (sse / energy) ** 0.5,
                        "max_abs": float(error.abs().max().item()),
                    }
                    totals[matrix]["sse"] += sse
                    totals[matrix]["source_energy"] += energy
                    expert_sse += sse
                    expert_energy += energy
                    del decoded, source_physical, error
                records.append(
                    {
                        "expert": expert,
                        "intermediate_draw": draws[expert],
                        "sse": expert_sse,
                        "source_energy": expert_energy,
                        "relative_sse": expert_sse / expert_energy,
                        "relative_l2": (expert_sse / expert_energy) ** 0.5,
                        "matrices": matrix_records,
                    }
                )
                if (expert + 1) % 64 == 0:
                    print(
                        json.dumps(
                            {
                                "layer": args.layer,
                                "decoded_experts": expert + 1,
                                "experts": C.NUM_EXPERTS,
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
        "kind": "qsrt_uniform_k2_layer_payload_distortion",
        "layer": args.layer,
        "source_checkpoint": str(source_store.root),
        "upstream_overlay": str(args.upstream_overlay.resolve()),
        "down_overlay": str(args.down_overlay.resolve()),
        "draw_result": str(args.draw_result.resolve()),
        "experts": C.NUM_EXPERTS,
        "draw_counts": {
            str(draw): draws.count(draw) for draw in sorted(set(draws))
        },
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
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

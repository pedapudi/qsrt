#!/usr/bin/env python3
"""Compare matched MCG and QSRT production encodes on real GLM-5.2 weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qsrt.glm52_real_weight_benchmark import run_real_weight_codec_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exl3-endpoint", type=Path, required=True)
    parser.add_argument(
        "--source-inventory",
        type=Path,
        help=(
            "sealed source_inventory.json; defaults to the inventory bundled "
            "with the EXL3 endpoint"
        ),
    )
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument(
        "--panel-manifest",
        type=Path,
        default=Path("experiments/glm52_layer3_rate_pattern_panel.json"),
        help="source-controlled expert list frozen before candidate measurement",
    )
    parser.add_argument(
        "--experts",
        type=int,
        default=8,
        help="number of layer-local experts sampled across R7 rate patterns, 1..256",
    )
    parser.add_argument(
        "--panel-offset",
        type=int,
        default=0,
        help="zero-based start within the frozen panel for disjoint GPU workers",
    )
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument(
        "--bits",
        default="3",
        help="comma-separated matched rates from 2 through 6; default: 3",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--exllamav3-root",
        type=Path,
        default=Path("/opt/exllamav3"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-source-shard-hashes",
        action="store_true",
        help="skip rereading BF16 shard payloads after a separately recorded verification",
    )
    parser.add_argument(
        "--skip-exl3-shard-hash",
        action="store_true",
        help="skip rereading the R7 shard after a separately recorded verification",
    )
    parser.add_argument(
        "--trellis-diagnostics",
        action="store_true",
        help=(
            "record post-BlockLDLQ Viterbi-input distribution, residual, "
            "adjacency, tails, and table occupancy for the SQG arm"
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        bits = tuple(int(value) for value in args.bits.split(","))
    except ValueError:
        parser.error("--bits must be a comma-separated list of integers")
    source_inventory = args.source_inventory or (
        args.exl3_endpoint
        / "reproducibility"
        / "r10"
        / "inventories"
        / "source_inventory.json"
    )
    report = run_real_weight_codec_benchmark(
        source_root=args.source,
        source_inventory_path=source_inventory,
        exl3_endpoint_root=args.exl3_endpoint,
        dest=args.dest,
        layer=args.layer,
        expert_count=args.experts,
        bits=bits,
        device=torch.device(args.device),
        resume=args.resume,
        exllamav3_root=args.exllamav3_root,
        panel_manifest_path=args.panel_manifest,
        panel_offset=args.panel_offset,
        verify_source_shard_hashes=not args.skip_source_shard_hashes,
        verify_exl3_shard_hash=not args.skip_exl3_shard_hash,
        return_trellis_diagnostics=args.trellis_diagnostics,
    )
    print(
        json.dumps(
            {
                "expert_count": report["expert_count"],
                "rates": report["rates"],
                "panel": report["panel"],
                "aggregate": report["aggregate"],
                "evidence_boundary": report["evidence_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build dense EXL3 and QSRT expert endpoints for reversible KLD tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qsrt.glm52_expert_intervention import build_dense_intervention_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--exl3-endpoint", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument(
        "--panel-manifest",
        type=Path,
        default=Path("experiments/glm52_layer3_rate_pattern_panel.json"),
    )
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--candidate-bits", type=int, choices=(3, 4), default=3)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--panel-offset", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exllamav3-root", type=Path, default=Path("/opt/exllamav3"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-source-shard-hashes", action="store_true")
    parser.add_argument("--skip-exl3-shard-hash", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_dense_intervention_artifacts(
        source_root=args.source,
        source_inventory_path=args.source_inventory,
        exl3_endpoint_root=args.exl3_endpoint,
        panel_manifest_path=args.panel_manifest,
        dest=args.dest,
        layer=args.layer,
        expert_count=args.experts,
        panel_offset=args.panel_offset,
        device=torch.device(args.device),
        exllamav3_root=args.exllamav3_root,
        resume=args.resume,
        verify_source_shard_hashes=not args.skip_source_shard_hashes,
        verify_exl3_shard_hash=not args.skip_exl3_shard_hash,
        candidate_bits=args.candidate_bits,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "layer": report["layer"],
                "expert_count": report["expert_count"],
                "dense_endpoint_bytes": report["dense_endpoint_bytes"],
                "evidence_boundary": report["evidence_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

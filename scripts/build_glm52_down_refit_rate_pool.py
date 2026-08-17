#!/usr/bin/env python3
"""Build a GLM-5.2 K3/K4 pool that preserves each fitted down target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qsrt.glm52_down_refit_rate_pool import build_down_refit_rate_pool_slice


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--uniform-k3", type=Path, required=True)
    parser.add_argument("--down-refit", type=Path, required=True)
    parser.add_argument("--uniform-k4", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--panel-offset", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exllamav3-root", type=Path, default=Path("/opt/exllamav3"))
    parser.add_argument("--skip-source-shard-hashes", action="store_true")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    report = build_down_refit_rate_pool_slice(
        source_root=args.source,
        source_inventory_path=args.source_inventory,
        uniform_k3_root=args.uniform_k3,
        down_refit_root=args.down_refit,
        uniform_k4_root=args.uniform_k4,
        capture_root=args.capture,
        panel_manifest_path=args.panel_manifest,
        dest=args.dest,
        layer=args.layer,
        expert_count=args.experts,
        panel_offset=args.panel_offset,
        device=torch.device(args.device),
        exllamav3_root=args.exllamav3_root,
        verify_source_shard_hashes=not args.skip_source_shard_hashes,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

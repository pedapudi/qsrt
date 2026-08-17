#!/usr/bin/env python3
"""Encode a frozen GLM-5.2 panel with two-sided downstream curvature."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qsrt.glm52_two_sided_curvature import run_two_sided_curvature_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-inventory", required=True, type=Path)
    parser.add_argument("--input-artifact", required=True, type=Path)
    parser.add_argument("--curvature-factors", required=True, type=Path)
    parser.add_argument("--panel-manifest", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--exllamav3-root", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--experts", type=int, default=1)
    parser.add_argument("--panel-offset", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-source-shard-hashes", action="store_true")
    args = parser.parse_args()
    report = run_two_sided_curvature_panel(
        source_root=args.source_root,
        source_inventory_path=args.source_inventory,
        input_artifact_root=args.input_artifact,
        curvature_factor_root=args.curvature_factors,
        panel_manifest_path=args.panel_manifest,
        dest=args.dest,
        layer=args.layer,
        expert_count=args.experts,
        panel_offset=args.panel_offset,
        device=torch.device(args.device),
        exllamav3_root=args.exllamav3_root,
        verify_source_shard_hashes=not args.skip_source_shard_hashes,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "expert_count": report["expert_count"],
                "changed_expert_count": report["changed_expert_count"],
                "changed_projection_count": report["changed_projection_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

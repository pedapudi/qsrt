#!/usr/bin/env python3
"""Build GLM-5.2 expert-local input/output curvature factor pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qsrt.glm52_downstream_curvature import (
    run_downstream_curvature_factor_build,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-inventory", required=True, type=Path)
    parser.add_argument("--output-gradient-capture", required=True, type=Path)
    parser.add_argument("--panel-manifest", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--experts", type=int, default=1)
    parser.add_argument("--panel-offset", type=int, default=0)
    parser.add_argument("--input-identity-shrinkage", type=float, default=0.10)
    parser.add_argument("--output-identity-shrinkage", type=float, default=0.10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-source-shard-hashes", action="store_true")
    args = parser.parse_args()
    report = run_downstream_curvature_factor_build(
        source_root=args.source_root,
        source_inventory_path=args.source_inventory,
        output_gradient_capture_root=args.output_gradient_capture,
        panel_manifest_path=args.panel_manifest,
        dest=args.dest,
        layer=args.layer,
        expert_count=args.experts,
        panel_offset=args.panel_offset,
        input_identity_shrinkage=args.input_identity_shrinkage,
        output_identity_shrinkage=args.output_identity_shrinkage,
        device=torch.device(args.device),
        verify_source_shard_hashes=not args.skip_source_shard_hashes,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "expert_count": report["expert_count"],
                "factor_file_bytes": report["factor_file_bytes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Encode GLM-5.2 QSRT candidates with routed-input curvature metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qsrt.glm52_routed_input_curvature import run_routed_input_curvature_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--input-artifact", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--panel-offset", type=int, default=0)
    parser.add_argument("--identity-shrinkage", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exllamav3-root", type=Path, default=Path("/opt/exllamav3"))
    parser.add_argument("--skip-source-shard-hashes", action="store_true")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    report = run_routed_input_curvature_panel(
        source_root=args.source,
        source_inventory_path=args.source_inventory,
        input_artifact_root=args.input_artifact,
        capture_root=args.capture,
        panel_manifest_path=args.panel_manifest,
        dest=args.dest,
        layer=args.layer,
        expert_count=args.experts,
        panel_offset=args.panel_offset,
        identity_shrinkage=args.identity_shrinkage,
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
                "dense_endpoint_bytes": report["dense_endpoint_bytes"],
                "evidence_boundary": report["evidence_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

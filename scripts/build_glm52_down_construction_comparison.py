#!/usr/bin/env python3
"""Build one GLM-5.2 down-projection metric-and-target comparison cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qsrt.glm52_down_construction import (
    INPUT_METRIC_POLICIES,
    TARGET_POLICIES,
    run_down_construction_panel,
)


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
    parser.add_argument(
        "--input-metric", choices=INPUT_METRIC_POLICIES, required=True
    )
    parser.add_argument("--target", choices=TARGET_POLICIES, required=True)
    parser.add_argument("--ridge-factors", default="0.001,0.01,0.1,1.0")
    parser.add_argument("--covariance-identity-shrinkage", type=float, default=0.01)
    parser.add_argument("--local-tail-relative-tolerance", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exllamav3-root", type=Path, default=Path("/opt/exllamav3"))
    parser.add_argument("--skip-source-shard-hashes", action="store_true")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    factors = tuple(float(value) for value in args.ridge_factors.split(","))
    report = run_down_construction_panel(
        source_root=args.source,
        source_inventory_path=args.source_inventory,
        input_artifact_root=args.input_artifact,
        capture_root=args.capture,
        panel_manifest_path=args.panel_manifest,
        dest=args.dest,
        layer=args.layer,
        expert_count=args.experts,
        panel_offset=args.panel_offset,
        input_metric_policy=args.input_metric,
        target_policy=args.target,
        ridge_factors=factors,
        covariance_identity_shrinkage=args.covariance_identity_shrinkage,
        tail_relative_tolerance=args.local_tail_relative_tolerance,
        device=torch.device(args.device),
        exllamav3_root=args.exllamav3_root,
        verify_source_shard_hashes=not args.skip_source_shard_hashes,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "construction": report["construction"],
                "expert_count": report["expert_count"],
                "accepted_refit_count": report["accepted_refit_count"],
                "dense_endpoint_bytes": report["dense_endpoint_bytes"],
                "evidence_boundary": report["evidence_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

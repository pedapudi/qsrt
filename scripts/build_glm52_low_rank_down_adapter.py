#!/usr/bin/env python3
"""Build one GLM-5.2 down-only activation-weighted low-rank adapter slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qsrt.glm52_low_rank_down_adapter import (
    BASE_CONSTRUCTIONS,
    run_low_rank_down_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--input-artifact", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--base-construction", choices=BASE_CONSTRUCTIONS, required=True)
    parser.add_argument("--rank", type=int, choices=(2, 4), required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--panel-offset", type=int, default=0)
    parser.add_argument("--ridge-factors", default="0.001,0.01,0.1,1.0")
    parser.add_argument("--oversampling", type=int, default=8)
    parser.add_argument("--power-iterations", type=int, default=2)
    parser.add_argument("--batch-rows", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-source-shard-hashes", action="store_true")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    report = run_low_rank_down_panel(
        source_root=args.source,
        source_inventory_path=args.source_inventory,
        input_artifact_root=args.input_artifact,
        capture_root=args.capture,
        panel_manifest_path=args.panel_manifest,
        dest=args.dest,
        layer=args.layer,
        expert_count=args.experts,
        panel_offset=args.panel_offset,
        base_construction=args.base_construction,
        rank=args.rank,
        ridge_factors=tuple(float(value) for value in args.ridge_factors.split(",")),
        oversampling=args.oversampling,
        power_iterations=args.power_iterations,
        batch_rows=args.batch_rows,
        seed=args.seed,
        device=torch.device(args.device),
        verify_source_shard_hashes=not args.skip_source_shard_hashes,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "base_construction": report["base_construction"],
                "rank": report["rank"],
                "expert_count": report["expert_count"],
                "logical_adapter_bytes": report["logical_adapter_bytes"],
                "dense_endpoint_bytes": report["dense_endpoint_bytes"],
                "evidence_boundary": report["evidence_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

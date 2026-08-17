#!/usr/bin/env python3
"""Build a frozen-scale GLM-5.2 BlockLDLQ feedback ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qsrt.glm52_blockldlq_feedback import run_feedback_ablation_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-inventory", required=True, type=Path)
    parser.add_argument("--input-artifact", required=True, type=Path)
    parser.add_argument("--panel-manifest", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--exllamav3-root", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--experts", type=int, default=1)
    parser.add_argument("--panel-offset", type=int, default=0)
    parser.add_argument("--feedback-multiplier", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-source-shard-hashes", action="store_true")
    args = parser.parse_args()
    report = run_feedback_ablation_panel(
        source_root=args.source_root,
        source_inventory_path=args.source_inventory,
        input_artifact_root=args.input_artifact,
        panel_manifest_path=args.panel_manifest,
        dest=args.dest,
        layer=args.layer,
        expert_count=args.experts,
        panel_offset=args.panel_offset,
        feedback_multiplier=args.feedback_multiplier,
        device=torch.device(args.device),
        exllamav3_root=args.exllamav3_root,
        verify_source_shard_hashes=not args.skip_source_shard_hashes,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "expert_count": report["expert_count"],
                "feedback_multiplier": report["feedback_multiplier"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

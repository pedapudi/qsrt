#!/usr/bin/env python3
"""Compare frozen GLM expert artifacts on an untouched reporting context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qsrt.glm52_reporting_output import compare_reporting_expert_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--uniform-artifact", type=Path, required=True)
    parser.add_argument("--comparison-artifact", type=Path, required=True)
    parser.add_argument("--reporting-capture", type=Path, required=True)
    parser.add_argument("--uniform-kld-report", type=Path, required=True)
    parser.add_argument("--comparison-kld-report", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--skip-source-shard-hashes",
        action="store_true",
        help=(
            "validate source headers and recorded hashes without rereading each "
            "multi-gigabyte shard; reserved for a repeated run after a sealed "
            "hash-verifying run"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = compare_reporting_expert_outputs(
        source_root=args.source_root,
        source_inventory_path=args.source_inventory,
        uniform_artifact_root=args.uniform_artifact,
        comparison_artifact_root=args.comparison_artifact,
        reporting_capture_root=args.reporting_capture,
        uniform_kld_report_path=args.uniform_kld_report,
        comparison_kld_report_path=args.comparison_kld_report,
        dest=args.dest,
        device=torch.device(args.device),
        verify_source_shard_hashes=not args.skip_source_shard_hashes,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

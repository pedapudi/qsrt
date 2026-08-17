#!/usr/bin/env python3
"""Run matched GLM-5.2 uniform-K2 and uniform-K4 codec pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qsrt.glm52_pilot import run_uniform_rate_pilot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--exllamav3-root",
        type=Path,
        default=Path("/home/luke/projects/exllamav3"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_uniform_rate_pilot(
        source_root=args.source,
        baseline_root=args.baseline,
        dest=args.dest,
        device=torch.device(args.device),
        resume=args.resume,
        exllamav3_root=args.exllamav3_root,
    )
    print(
        json.dumps(
            {
                label: {
                    "classification": report["aggregate"][label]["classification"],
                    "overall": report["aggregate"][label]["overall"],
                    "bootstrap": report["aggregate"][label]["bootstrap"],
                }
                for label in ("K2", "K4")
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

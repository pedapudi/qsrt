#!/usr/bin/env python3
"""Run the frozen 48-expert GLM-5.2 fixed-3bpw QSRT codec pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qsrt.glm52_pilot import run_rate_shift_pilot


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
    report = run_rate_shift_pilot(
        source_root=args.source,
        baseline_root=args.baseline,
        dest=args.dest,
        device=torch.device(args.device),
        resume=args.resume,
        exllamav3_root=args.exllamav3_root,
    )
    aggregate = report["aggregate"]
    print(
        json.dumps(
            {
                "classification": aggregate["rate_shifted"]["classification"],
                "overall": aggregate["rate_shifted"]["overall"],
                "uniform_r0_control": aggregate["uniform_r0_control"]["overall"],
                "mode_selection": aggregate["mode_selection"],
                "bootstrap": aggregate["rate_shifted"]["bootstrap"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the bounded matched-payload QSRT proposal benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from qsrt.tiny_improvement_benchmark import run_benchmark, run_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="write the JSON report to this path instead of standard output",
    )
    parser.add_argument(
        "--experts",
        type=int,
        default=8,
        help="run this many experts sequentially with one shared table (default: 8)",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="first source seed in a multi-expert sweep (default: 0)",
    )
    parser.add_argument(
        "--pair-table-correlation",
        type=float,
        default=0.7,
        help=(
            "correlation used to train the shared synthetic pair table; "
            "must be between -0.99 and 0.99 (default: 0.7)"
        ),
    )
    args = parser.parse_args()
    if args.experts <= 0:
        parser.error("--experts must be positive")
    if not -0.99 < args.pair_table_correlation < 0.99:
        parser.error("--pair-table-correlation must be between -0.99 and 0.99")
    started = time.perf_counter()
    report = (
        run_benchmark(
            args.start_seed,
            pair_table_correlation=args.pair_table_correlation,
        )
        if args.experts == 1
        else run_sweep(
            args.experts,
            start_seed=args.start_seed,
            pair_table_correlation=args.pair_table_correlation,
        )
    )
    report["wall_seconds"] = time.perf_counter() - started
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the CPU-only two-bit GLM-5.2 expert mechanism benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from qsrt.glm52_tiny_benchmark import (
    DEFAULT_EXPERT_COUNT,
    MAXIMUM_EXPERT_COUNT,
    SOURCE_FAMILIES,
    BenchmarkConfig,
    run_sweep,
)


def _positive_gauge_values(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "gauge values must be comma-separated numbers"
        ) from exc
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("gauge values must be positive")
    if not any(abs(value - 1.0) <= 1e-12 for value in values):
        raise argparse.ArgumentTypeError("gauge values must include 1.0")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experts",
        type=int,
        default=DEFAULT_EXPERT_COUNT,
        help=f"number of sequential experts, 1..{MAXIMUM_EXPERT_COUNT} (default: 8)",
    )
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument(
        "--source-family", choices=SOURCE_FAMILIES, default="mixed"
    )
    parser.add_argument("--gate-up-correlation", type=float, default=0.6)
    parser.add_argument("--tail-degrees-of-freedom", type=float, default=3.5)
    parser.add_argument("--weight-scale", type=float, default=1.0)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--fit-rows", type=int, default=48)
    parser.add_argument("--heldout-rows", type=int, default=48)
    parser.add_argument(
        "--gauge-values",
        type=_positive_gauge_values,
        default=_positive_gauge_values("0.5,0.7071067811865476,1,1.4142135623730951,2"),
        help="positive reciprocal-balance factors including 1.0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON to this path instead of standard output",
    )
    args = parser.parse_args()
    if not 1 <= args.experts <= MAXIMUM_EXPERT_COUNT:
        parser.error(f"--experts must be between 1 and {MAXIMUM_EXPERT_COUNT}")
    config = BenchmarkConfig(
        fit_rows=args.fit_rows,
        heldout_rows=args.heldout_rows,
        source_family=args.source_family,
        gate_up_correlation=args.gate_up_correlation,
        tail_degrees_of_freedom=args.tail_degrees_of_freedom,
        weight_scale=args.weight_scale,
        input_scale=args.input_scale,
        gauge_values=args.gauge_values,
    )
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    started = time.perf_counter()
    report = run_sweep(args.experts, start_seed=args.start_seed, config=config)
    report["wall_seconds"] = time.perf_counter() - started
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()

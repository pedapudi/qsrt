#!/usr/bin/env python3
"""Compare teacher and student top-k routing archives."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from qsrt.kimi_routes import KimiRouteArchive, compare_route_archives


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-tokens", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_route_archives(
        KimiRouteArchive(args.teacher, require_complete=True),
        KimiRouteArchive(args.student, require_complete=True),
        chunk_tokens=args.chunk_tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "tokens": report["token_count"],
                "mean_topk_overlap": report["mean_topk_overlap"],
                "mean_exact_topk_set_agreement": report[
                    "mean_exact_topk_set_agreement"
                ],
                "mean_marginal_total_variation": report[
                    "mean_marginal_total_variation"
                ],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

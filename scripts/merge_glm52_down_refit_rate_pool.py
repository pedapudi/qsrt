#!/usr/bin/env python3
"""Merge disjoint GLM-5.2 down-refit K3/K4 rate-pool slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qsrt.glm52_down_refit_rate_pool import merge_down_refit_rate_pool_slices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    args = parser.parse_args()
    report = merge_down_refit_rate_pool_slices(
        inputs=args.inputs,
        dest=args.dest,
        panel_manifest_path=args.panel_manifest,
        layer=args.layer,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

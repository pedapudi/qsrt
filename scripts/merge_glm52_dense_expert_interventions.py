#!/usr/bin/env python3
"""Merge disjoint GLM-5.2 dense-endpoint slices into one checked artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qsrt.glm52_expert_intervention import merge_dense_intervention_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    args = parser.parse_args()
    report = merge_dense_intervention_artifacts(
        inputs=args.inputs,
        dest=args.dest,
        panel_manifest_path=args.panel_manifest,
        layer=args.layer,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "expert_count": report["expert_count"],
                "dense_endpoint_bytes": report["dense_endpoint_bytes"],
                "composition": report["composition"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

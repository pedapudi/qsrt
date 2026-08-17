#!/usr/bin/env python3
"""Materialize the pre-registered GLM-5.2 mixed-K3/K4 panel artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qsrt.glm52_k3_k4_allocation import materialize_fixed_mixed_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--uniform-k4", type=Path, required=True)
    parser.add_argument("--pre-registration", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    report = materialize_fixed_mixed_artifact(
        base_root=args.base,
        uniform_k4_root=args.uniform_k4,
        pre_registration_path=args.pre_registration,
        dest=args.dest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Materialize a frozen partial GLM-5.2 K3/K4 down-refit rate map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qsrt.glm52_down_refit_rate_pool import (
    materialize_registered_partial_rate_map,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate-pool", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    report = materialize_registered_partial_rate_map(
        rate_pool_root=args.rate_pool,
        registration_path=args.registration,
        dest=args.dest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

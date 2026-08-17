#!/usr/bin/env python3
"""Materialize a fixed or selection-data GLM-5.2 mixed-rate candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qsrt.glm52_down_refit_rate_pool import (
    materialize_down_refit_rate_pool_allocation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate-pool", type=Path, required=True)
    parser.add_argument("--pre-registration", type=Path, required=True)
    parser.add_argument(
        "--allocation-kind",
        choices=("fixed_rate_stratified", "selection_data_complete_expert"),
        required=True,
    )
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    report = materialize_down_refit_rate_pool_allocation(
        rate_pool_root=args.rate_pool,
        pre_registration_path=args.pre_registration,
        allocation_kind=args.allocation_kind,
        dest=args.dest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

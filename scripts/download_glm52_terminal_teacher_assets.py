#!/usr/bin/env python3
"""Download only the GLM-5.2 tensors needed for frozen teacher references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from qsrt.glm52_terminal_teacher_assets import (
    build_terminal_teacher_asset_download_contract,
    download_terminal_teacher_assets,
)
from qsrt.glm52_terminal_teacher_reference import (
    validate_terminal_teacher_reference_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--token-environment-variable",
        default="HF_TOKEN",
        help="Environment variable containing an optional Hugging Face token.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.retries < 0:
        raise ValueError("retries must not be negative")
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    validate_terminal_teacher_reference_plan(plan)
    contract = build_terminal_teacher_asset_download_contract(
        plan=plan,
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
    )
    report = download_terminal_teacher_assets(
        contract=contract,
        destination=args.dest,
        token=os.environ.get(args.token_environment_variable),
        jobs=args.jobs,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

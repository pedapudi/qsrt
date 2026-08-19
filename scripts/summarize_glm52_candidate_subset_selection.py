#!/usr/bin/env python3
"""Apply the frozen two-group retention rule to a model-KLD selection report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from qsrt.glm52_model_kld_candidate_selection import (
    summarize_selection_document_groups,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plan = json.loads(args.plan.read_text())
    report = json.loads(args.report.read_text())
    arms = summarize_selection_document_groups(report, plan=plan)
    retained = [arm for arm in arms if arm["retained"]]
    decision = {
        "schema": "qsrt_glm52_candidate_subset_selection_decision",
        "schema_version": 1,
        "status": "complete",
        "acceptance_rule": (
            "candidate-minus-resident equal-document mean KLD must be negative "
            "in both ordered eight-document groups"
        ),
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "arms": arms,
        "retained_arm_names": [arm["name"] for arm in retained],
    }
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    if args.dest.exists():
        raise FileExistsError(args.dest)
    temporary = args.dest.with_name(f".{args.dest.name}.partial-{os.getpid()}")
    temporary.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.dest)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

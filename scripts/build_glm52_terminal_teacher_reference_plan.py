#!/usr/bin/env python3
"""Freeze GLM-5.2 screening and confirmation documents before logit generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from qsrt.glm52_pilot import atomic_write_json
from qsrt.glm52_terminal_teacher_reference import (
    build_terminal_teacher_reference_plan,
    validate_terminal_teacher_reference_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-document-plan", type=Path, required=True)
    parser.add_argument("--calibration-corpus", type=Path, required=True)
    parser.add_argument("--frozen-at-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = build_terminal_teacher_reference_plan(
        capture_plan_bytes=args.canonical_document_plan.read_bytes(),
        corpus_bytes=args.calibration_corpus.read_bytes(),
        frozen_at_utc=args.frozen_at_utc,
    )
    validate_terminal_teacher_reference_plan(plan)
    atomic_write_json(args.output, plan)


if __name__ == "__main__":
    main()

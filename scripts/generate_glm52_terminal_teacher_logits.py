#!/usr/bin/env python3
"""Generate document-level GLM-5.2 teacher logits from terminal hidden rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from qsrt.glm52_document_disjoint_confirmation import (
    validate_terminal_reference_confirmation_generation_authorization,
)
from qsrt.glm52_terminal_teacher_assets import (
    build_terminal_teacher_asset_download_contract,
    validate_downloaded_terminal_teacher_assets,
)
from qsrt.glm52_terminal_teacher_logits import (
    generate_terminal_teacher_references,
    selected_document_token_receipts,
    tokenizer_file_identity,
)
from qsrt.glm52_terminal_teacher_reference import (
    validate_terminal_teacher_reference_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--evaluation-tier",
        choices=("screening", "confirmation"),
        required=True,
    )
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--closure-rows", type=int, default=8)
    parser.add_argument("--confirmation-freeze", type=Path)
    parser.add_argument("--screening-report", type=Path)
    parser.add_argument("--dest", type=Path, required=True)
    return parser


def _devices(value: str) -> list[torch.device]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("CUDA device list must contain distinct device numbers")
    devices = [torch.device(f"cuda:{int(name)}") for name in names]
    if any(
        device.index is None or device.index >= torch.cuda.device_count()
        for device in devices
    ):
        raise ValueError("requested CUDA device is unavailable")
    return devices


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("GLM-5.2 teacher-logit generation requires CUDA")
    torch.use_deterministic_algorithms(True)
    devices = _devices(args.devices)
    plan_bytes = args.plan.read_bytes()
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    plan = json.loads(plan_bytes)
    validate_terminal_teacher_reference_plan(plan)
    confirmation_authorization = None
    if args.evaluation_tier == "confirmation":
        if args.confirmation_freeze is None or args.screening_report is None:
            raise PermissionError(
                "confirmation generation requires a freeze record and screening report"
            )
        confirmation_authorization = (
            validate_terminal_reference_confirmation_generation_authorization(
                json.loads(args.confirmation_freeze.read_text()),
                teacher_reference_plan_sha256=plan_sha256,
                screening_report_path=args.screening_report,
            )
        )
    elif args.confirmation_freeze is not None or args.screening_report is not None:
        raise ValueError("screening generation cannot consume confirmation inputs")
    asset_contract = build_terminal_teacher_asset_download_contract(
        plan=plan, plan_sha256=plan_sha256
    )
    complete = validate_downloaded_terminal_teacher_assets(
        contract=asset_contract, destination=args.assets
    )
    complete_path = args.assets / "complete.json"
    complete_sha256 = hashlib.sha256(complete_path.read_bytes()).hexdigest()
    if complete["asset_count"] != asset_contract["asset_count"]:
        raise ValueError("teacher-asset complete receipt count differs")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=True,
    )
    token_receipts = selected_document_token_receipts(
        plan=plan,
        corpus_path=args.assets / "metadata/reap_recall_calib.jsonl",
        tokenizer=tokenizer,
        evaluation_tier=args.evaluation_tier,
    )
    manifest = generate_terminal_teacher_references(
        plan=plan,
        plan_sha256=plan_sha256,
        asset_contract=asset_contract,
        assets_root=args.assets,
        asset_complete_sha256=complete_sha256,
        token_receipts=token_receipts,
        tokenizer_identity={
            **tokenizer_file_identity(args.tokenizer),
            "class": type(tokenizer).__name__,
            "vocabulary_size": int(tokenizer.vocab_size),
        },
        evaluation_tier=args.evaluation_tier,
        devices=devices,
        closure_rows=args.closure_rows,
        destination=args.dest,
        confirmation_authorization=confirmation_authorization,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

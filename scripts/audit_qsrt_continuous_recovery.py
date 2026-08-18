#!/usr/bin/env python3
"""Audit continuous-recovery parameters and exact suffix-capture storage."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from qsrt.continuous_recovery import audit_continuous_recovery


DEFAULT_CHECKPOINT = Path(
    "/data/models/Kimi-K3-QSRT-K2-DIRECT-VITERBI-ALL-LINEARS-v1-model"
)


def _gib(value: int) -> str:
    return f"{value / (1 << 30):,.3f} GiB"


def _tib(value: int) -> str:
    return f"{value / (1 << 40):,.3f} TiB"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--first-layer", type=int, default=84)
    parser.add_argument("--end-layer", type=int, default=93)
    parser.add_argument("--capture-tokens", type=int, default=50_000_000)
    parser.add_argument("--data-root", type=Path, default=Path("/data"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_continuous_recovery(
        args.checkpoint,
        first_layer=args.first_layer,
        end_layer=args.end_layer,
        capture_token_count=args.capture_tokens,
    )
    usage = shutil.disk_usage(args.data_root)
    report["data_filesystem"] = {
        "path": str(args.data_root.resolve()),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_after_capture_bytes": usage.free
        - int(report["capture_storage"]["total_bytes"]),
    }
    if args.output is not None:
        _atomic_json(args.output, report)

    total = report["trainable"]["total"]
    capture = report["capture_storage"]
    print(f"checkpoint: {report['checkpoint']}")
    print(
        f"suffix: layers {report['suffix']['first_layer']}-"
        f"{report['suffix']['end_layer'] - 1}"
    )
    print(
        f"trainable: {total['tensor_count']} tensors, "
        f"{total['parameters']:,} parameters"
    )
    print(f"resident trainables: {_gib(total['parameter_bytes'])}")
    print(f"trainables + gradients + FP32 AdamW: {_gib(total['training_bytes'])}")
    for group, values in report["trainable"]["by_group"].items():
        print(
            f"  {group}: {values['parameters']:,} params, "
            f"{_gib(values['training_bytes'])} training state"
        )
    print(
        "student replay state: "
        f"{capture['student_hidden_vectors_per_token']} BF16 hidden vectors/token, "
        f"{_tib(capture['student_bytes'])}"
    )
    print(
        f"teacher final hiddens: {_tib(capture['teacher_bytes'])}; "
        f"capture total: {_tib(capture['total_bytes'])}"
    )
    print(
        f"{args.data_root} free: {_tib(usage.free)}; "
        f"after uncompressed capture: "
        f"{_tib(usage.free - int(capture['total_bytes']))}"
    )
    if args.output is not None:
        print(f"report: {args.output.resolve()}")


if __name__ == "__main__":
    main()

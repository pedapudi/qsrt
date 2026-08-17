#!/usr/bin/env python3
"""Publish a TP-independent QSRT artifact as a direct local model view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qsrt.pack.package_helpers import DEFAULT_NONEXPERT
from qsrt.pack.qsrt_model_view import publish_qsrt_model_view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--nonexpert-source", type=Path, default=DEFAULT_NONEXPERT)
    parser.add_argument("--official-snapshot", type=Path)
    args = parser.parse_args()
    result = publish_qsrt_model_view(
        args.artifact,
        args.destination,
        nonexpert_source=args.nonexpert_source,
        official_snapshot=args.official_snapshot,
    )
    print(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()

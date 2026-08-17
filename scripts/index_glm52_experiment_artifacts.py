#!/usr/bin/env python3
"""Classify experiment files and hash every file readable by the caller."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_NAME = "ARTIFACT_INDEX.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _category(relative_path: Path) -> tuple[str, str]:
    first = relative_path.parts[0] if relative_path.parts else ""
    categories = {
        "reference": ("published_evaluation_reference", "retain"),
        "results": ("experiment_result", "retain"),
        "preflight": ("input_identity_preflight", "retain"),
        "dependencies": ("reproducible_dependency_copy", "removable_after_release"),
        "runtime-cache": ("runtime_cache", "removable_after_release"),
        "corpus-cache": ("evaluation_corpus_cache", "removable_after_release"),
        "corpus": ("corpus_download_attempt", "review_before_cleanup"),
        "source": ("source_working_tree_copy", "retain_until_source_revision_is_frozen"),
        "model-cache": ("local_model_cache", "removable_after_experiments"),
        "captures": ("routed_activation_capture", "retain_until_results_are_consolidated"),
        "launch-records": ("experiment_launch_record", "retain"),
        "runtime-control": ("runtime_control", "removable_after_experiments"),
        "logs": ("experiment_log", "retain"),
    }
    return categories.get(first, ("uncategorized_experiment_file", "review_before_cleanup"))


def build_index(root: Path, output: Path) -> dict[str, Any]:
    resolved_root = root.resolve(strict=True)
    resolved_output = output.resolve()
    entries: list[dict[str, Any]] = []
    for path in sorted(resolved_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if path.resolve() == resolved_output:
            continue
        relative = path.relative_to(resolved_root)
        category, cleanup_policy = _category(relative)
        metadata = path.stat()
        entry = {
            "relative_path": relative.as_posix(),
            "size_bytes": metadata.st_size,
            "category": category,
            "cleanup_policy": cleanup_policy,
        }
        try:
            entry["sha256"] = _sha256(path)
        except OSError as error:
            entry["sha256"] = None
            entry["hash_error"] = f"{type(error).__name__}: {error.strerror}"
        entries.append(entry)
    unhashed_file_count = sum(entry["sha256"] is None for entry in entries)
    return {
        "schema": "glm52_experiment_artifact_index",
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "root": str(resolved_root),
        "excluded_output_path": str(resolved_output),
        "regular_file_count": len(entries),
        "regular_file_bytes": sum(entry["size_bytes"] for entry in entries),
        "unhashed_file_count": unhashed_file_count,
        "files": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help=f"JSON destination; defaults to ROOT/{DEFAULT_OUTPUT_NAME}",
    )
    args = parser.parse_args()
    output = args.output or args.root / DEFAULT_OUTPUT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    index = build_index(args.root, output)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                key: index[key]
                for key in (
                    "regular_file_count",
                    "regular_file_bytes",
                    "unhashed_file_count",
                )
            }
        )
    )


if __name__ == "__main__":
    main()

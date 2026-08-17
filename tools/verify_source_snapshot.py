#!/usr/bin/env python3
"""Create or verify the publication-safe QSRT source snapshot manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "SOURCE_SNAPSHOT_MANIFEST.json"
CHECKSUM_PATH = ROOT / "SOURCE_SNAPSHOT_MANIFEST.sha256"
MAX_FILE_BYTES = 10 * 1024 * 1024
IGNORED_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "out",
}
IGNORED_NAMES = {MANIFEST_PATH.name, CHECKSUM_PATH.name}
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".gguf",
    ".kqcapture",
    ".kqhess",
    ".kqsamples",
    ".kqstats",
    ".logits",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".safetensors",
}
SECRET_PATTERNS = {
    "private_key": re.compile(rb"BEGIN (?:RSA |OPENSSH |EC |DSA |PGP )?PRIVATE KEY"),
    "github_token": re.compile(rb"(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{20,}"),
    "hugging_face_token": re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    "aws_access_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excluded(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.name in IGNORED_NAMES
        or any(part in IGNORED_PARTS for part in relative.parts)
    )


def inventory() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if excluded(path):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.is_symlink():
            failures.append(f"symlink is forbidden: {relative} -> {os.readlink(path)}")
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            failures.append(f"file exceeds {MAX_FILE_BYTES} bytes: {relative} ({size})")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"generated-artifact suffix is forbidden: {relative}")
        data = path.read_bytes()
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(data):
                failures.append(f"possible {name} in {relative}")
        records.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    if failures:
        raise SystemExit("\n".join(failures))
    return records


def source_git_state(source: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(source), *args], text=True
        ).strip()

    status_lines = run("status", "--porcelain=v1").splitlines()
    modified_tracked_files = run("ls-files", "-m").splitlines()
    untracked_files = run("ls-files", "-o", "--exclude-standard").splitlines()
    return {
        "origin": run("remote", "get-url", "origin"),
        "base_commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "upstream": run("rev-parse", "--abbrev-ref", "@{upstream}"),
        "ahead_behind": run("rev-list", "--left-right", "--count", "@{upstream}...HEAD"),
        "status_porcelain_v1": status_lines,
        "modified_tracked_files": modified_tracked_files,
        "untracked_files": untracked_files,
        "modified_tracked_count": len(modified_tracked_files),
        "untracked_file_count": len(untracked_files),
    }


def create(source: Path, timestamp: str) -> None:
    records = inventory()
    document = {
        "schema": "qsrt-source-snapshot-manifest-v1",
        "snapshot_timestamp_utc": timestamp,
        "source_git": source_git_state(source),
        "publication_additions": [
            "CONTINUATION_GUIDE.md",
            "publication safeguards appended to .gitignore",
            "tools/verify_source_snapshot.py",
            "SOURCE_SNAPSHOT_MANIFEST.json",
            "SOURCE_SNAPSHOT_MANIFEST.sha256",
        ],
        "excluded_classes": [
            "source Git history",
            "virtual environments and Python caches",
            "generated out directories",
            "model checkpoints and tensor payloads",
            "reference logits, activation captures, and Hessian bundles",
            "container and OCI images or layers",
            "runtime and package caches",
            "credentials and authentication state",
            "regular files larger than ten mebibytes",
        ],
        "manifest_self_exclusions": [
            MANIFEST_PATH.name,
            CHECKSUM_PATH.name,
        ],
        "maximum_file_bytes": MAX_FILE_BYTES,
        "file_count": len(records),
        "total_file_bytes": sum(record["bytes"] for record in records),
        "files": records,
    }
    MANIFEST_PATH.write_text(json.dumps(document, indent=2) + "\n")
    digest = sha256(MANIFEST_PATH)
    CHECKSUM_PATH.write_text(f"{digest}  {MANIFEST_PATH.name}\n")
    print(
        json.dumps(
            {
                "manifest": str(MANIFEST_PATH),
                "manifest_sha256": digest,
                "file_count": len(records),
                "total_file_bytes": document["total_file_bytes"],
            },
            indent=2,
        )
    )


def verify() -> None:
    expected = json.loads(MANIFEST_PATH.read_text())
    expected_records = {record["path"]: record for record in expected["files"]}
    actual_records = {record["path"]: record for record in inventory()}
    failures: list[str] = []
    for path in sorted(expected_records.keys() - actual_records.keys()):
        failures.append(f"missing file: {path}")
    for path in sorted(actual_records.keys() - expected_records.keys()):
        failures.append(f"unexpected file: {path}")
    for path in sorted(expected_records.keys() & actual_records.keys()):
        if expected_records[path] != actual_records[path]:
            failures.append(f"changed file: {path}")
    checksum_digest, checksum_name = CHECKSUM_PATH.read_text().strip().split()
    if checksum_name != MANIFEST_PATH.name:
        failures.append(f"unexpected checksum target: {checksum_name}")
    if checksum_digest != sha256(MANIFEST_PATH):
        failures.append("manifest checksum mismatch")
    if failures:
        raise SystemExit("\n".join(failures))
    print(
        json.dumps(
            {
                "status": "verified",
                "file_count": len(actual_records),
                "total_file_bytes": sum(record["bytes"] for record in actual_records.values()),
                "manifest_sha256": checksum_digest,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--timestamp-utc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.create:
        if args.source is None or args.timestamp_utc is None:
            raise SystemExit("--create requires --source and --timestamp-utc")
        create(args.source.resolve(), args.timestamp_utc)
    else:
        verify()


if __name__ == "__main__":
    main()

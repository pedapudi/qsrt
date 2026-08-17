#!/usr/bin/env python3
"""Validate a host-local copy of the immutable GLM-5.2 EXL3 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ALLOWED_DESTINATION_EXTRAS = {"COPY_RSYNC.log"}
SMALL_FILE_HASH_LIMIT = 64 * 1024 * 1024
R7_HASH_WORKERS = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_files_and_symlinks(root: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            entries[relative] = {"kind": "symlink", "target": os.readlink(path)}
        elif path.is_file():
            stat = path.stat()
            entries[relative] = {
                "kind": "file",
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
    return entries


def _is_r7_expert_shard(relative_path: str) -> bool:
    name = Path(relative_path).name
    return (
        relative_path == name
        and name.startswith("r7-experts-layer-")
        and name.endswith(".safetensors")
    )


def validate_copy(source: Path, destination: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    destination = destination.resolve(strict=True)
    if source == destination:
        raise ValueError("source and destination must be different directories")

    source_entries = _regular_files_and_symlinks(source)
    destination_entries = _regular_files_and_symlinks(destination)
    missing = sorted(set(source_entries) - set(destination_entries))
    unexpected = sorted(
        set(destination_entries) - set(source_entries) - ALLOWED_DESTINATION_EXTRAS
    )
    mismatched: list[dict[str, Any]] = []
    for relative in sorted(set(source_entries) & set(destination_entries)):
        source_entry = source_entries[relative]
        destination_entry = destination_entries[relative]
        if source_entry != destination_entry:
            mismatched.append(
                {
                    "relative_path": relative,
                    "source": source_entry,
                    "destination": destination_entry,
                }
            )
    if missing or unexpected or mismatched:
        raise ValueError(
            "host-local copy tree mismatch: "
            f"{len(missing)} missing, {len(unexpected)} unexpected, "
            f"{len(mismatched)} metadata mismatches"
        )

    partial_files = sorted(
        relative
        for relative in destination_entries
        if relative.endswith((".partial", ".incomplete"))
        or "/.rsync-partial/" in f"/{relative}"
    )
    if partial_files:
        raise ValueError(f"host-local copy retains partial files: {partial_files[:8]}")

    small_file_hashes: list[dict[str, Any]] = []
    for relative, entry in source_entries.items():
        if entry["kind"] != "file" or entry["size_bytes"] > SMALL_FILE_HASH_LIMIT:
            continue
        source_sha256 = sha256_file(source / relative)
        destination_sha256 = sha256_file(destination / relative)
        if source_sha256 != destination_sha256:
            raise ValueError(f"small copied file failed SHA-256 closure: {relative}")
        small_file_hashes.append(
            {
                "relative_path": relative,
                "size_bytes": entry["size_bytes"],
                "sha256": destination_sha256,
            }
        )

    large_non_r7_hashes: list[dict[str, Any]] = []
    for relative, entry in source_entries.items():
        if (
            entry["kind"] != "file"
            or entry["size_bytes"] <= SMALL_FILE_HASH_LIMIT
            or _is_r7_expert_shard(relative)
        ):
            continue
        source_sha256 = sha256_file(source / relative)
        destination_sha256 = sha256_file(destination / relative)
        if source_sha256 != destination_sha256:
            raise ValueError(
                f"large non-R7 copied file failed SHA-256 closure: {relative}"
            )
        large_non_r7_hashes.append(
            {
                "relative_path": relative,
                "size_bytes": entry["size_bytes"],
                "sha256": destination_sha256,
            }
        )

    manifest_checksum = (destination / "MANIFEST.sha256").read_text().split()
    if len(manifest_checksum) != 2 or manifest_checksum[1] != "MANIFEST.json":
        raise ValueError("MANIFEST.sha256 has an invalid record")
    if sha256_file(destination / "MANIFEST.json") != manifest_checksum[0]:
        raise ValueError("destination MANIFEST.json failed its sealed SHA-256")

    def validate_r7_shard(sidecar_path: Path) -> dict[str, Any]:
        sidecar = json.loads(sidecar_path.read_text())
        shard_name = sidecar.get("shard")
        expected_sha256 = sidecar.get("shard_sha256")
        layer = sidecar.get("layer")
        if (
            not isinstance(shard_name, str)
            or Path(shard_name).name != shard_name
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or isinstance(layer, bool)
            or not isinstance(layer, int)
        ):
            raise ValueError(f"invalid R7 sidecar identity: {sidecar_path.name}")
        shard_path = destination / shard_name
        actual_sha256 = sha256_file(shard_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"R7 shard failed sealed SHA-256: {shard_name}")
        return {
            "layer": layer,
            "sidecar": sidecar_path.name,
            "shard": shard_name,
            "size_bytes": shard_path.stat().st_size,
            "sha256": actual_sha256,
        }

    sidecar_paths = sorted(destination.glob("r7-experts-layer-*.json"))
    if not sidecar_paths:
        raise ValueError("destination contains no R7 shard sidecars")
    with ThreadPoolExecutor(
        max_workers=min(R7_HASH_WORKERS, len(sidecar_paths))
    ) as executor:
        r7_shards = sorted(
            executor.map(validate_r7_shard, sidecar_paths),
            key=lambda item: item["layer"],
        )
    if len(r7_shards) != 75 or [item["layer"] for item in r7_shards] != list(
        range(3, 78)
    ):
        raise ValueError("destination must contain sealed R7 shards for layers 3..77")

    copied_file_bytes = sum(
        entry["size_bytes"]
        for entry in source_entries.values()
        if entry["kind"] == "file"
    )
    return {
        "schema": "glm52_exl3_host_local_copy_validation",
        "schema_version": 1,
        "status": "passed",
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "source": str(source),
        "destination": str(destination),
        "network_transfer": False,
        "source_regular_file_and_symlink_count": len(source_entries),
        "copied_file_bytes": copied_file_bytes,
        "allowed_destination_extras": sorted(ALLOWED_DESTINATION_EXTRAS),
        "small_file_sha256_count": len(small_file_hashes),
        "small_file_sha256_bytes": sum(item["size_bytes"] for item in small_file_hashes),
        "large_non_r7_sha256_count": len(large_non_r7_hashes),
        "large_non_r7_sha256_bytes": sum(
            item["size_bytes"] for item in large_non_r7_hashes
        ),
        "large_non_r7_files": large_non_r7_hashes,
        "manifest_sha256": manifest_checksum[0],
        "r7_shard_count": len(r7_shards),
        "r7_shard_bytes": sum(item["size_bytes"] for item in r7_shards),
        "r7_hash_workers": min(R7_HASH_WORKERS, len(r7_shards)),
        "r7_shards": r7_shards,
        "evidence_boundary": (
            "all source paths match destination type, size, and modification time; "
            "all files up to 64 MiB and every larger non-R7 file match source "
            "SHA-256; every R7 expert shard matches its sealed destination "
            "sidecar SHA-256"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = validate_copy(args.source, args.destination)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "source_regular_file_and_symlink_count",
                    "copied_file_bytes",
                    "small_file_sha256_count",
                    "large_non_r7_sha256_count",
                    "large_non_r7_sha256_bytes",
                    "r7_shard_count",
                    "r7_shard_bytes",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

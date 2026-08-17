from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "validate_glm52_exl3_host_local_copy.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_glm52_exl3_host_local_copy", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _make_checkpoint(root: Path) -> None:
    root.mkdir()
    manifest = b'{"schema":"test"}\n'
    (root / "MANIFEST.json").write_bytes(manifest)
    (root / "MANIFEST.sha256").write_text(
        f"{_sha256(manifest)}  MANIFEST.json\n"
    )
    (root / "config.json").write_text("{}\n")
    for layer in range(3, 78):
        shard_name = f"r7-experts-layer-{layer:03d}.safetensors"
        shard = f"layer {layer}\n".encode()
        (root / shard_name).write_bytes(shard)
        (root / f"r7-experts-layer-{layer:03d}.json").write_text(
            json.dumps(
                {
                    "layer": layer,
                    "shard": shard_name,
                    "shard_sha256": _sha256(shard),
                }
            )
            + "\n"
        )


def _copy_checkpoint(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    (destination / "COPY_RSYNC.log").write_text("local copy\n")


def test_validator_closes_tree_small_files_manifest_and_all_r7_shards(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _make_checkpoint(source)
    _copy_checkpoint(source, destination)

    report = MODULE.validate_copy(source, destination)

    assert report["status"] == "passed"
    assert report["network_transfer"] is False
    assert report["r7_shard_count"] == 75
    assert report["r7_hash_workers"] == MODULE.R7_HASH_WORKERS
    assert [item["layer"] for item in report["r7_shards"]] == list(range(3, 78))
    assert report["manifest_sha256"] == _sha256(b'{"schema":"test"}\n')


def test_validator_rejects_a_corrupt_copied_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _make_checkpoint(source)
    _copy_checkpoint(source, destination)
    path = destination / "config.json"
    original_mtime = path.stat().st_mtime_ns
    path.write_text("[]\n")
    os.utime(path, ns=(original_mtime, original_mtime))

    with pytest.raises(ValueError, match="small copied file failed SHA-256"):
        MODULE.validate_copy(source, destination)


def test_validator_rejects_an_unexpected_destination_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _make_checkpoint(source)
    _copy_checkpoint(source, destination)
    (destination / "untracked-weight.bin").write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="1 unexpected"):
        MODULE.validate_copy(source, destination)


def test_validator_hashes_large_non_r7_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _make_checkpoint(source)
    payload = b"large non-R7 payload"
    (source / "model-layer-003.safetensors").write_bytes(payload)
    _copy_checkpoint(source, destination)
    monkeypatch.setattr(MODULE, "SMALL_FILE_HASH_LIMIT", len(payload) - 1)

    report = MODULE.validate_copy(source, destination)
    assert any(
        item["relative_path"] == "model-layer-003.safetensors"
        and item["size_bytes"] == len(payload)
        and item["sha256"] == _sha256(payload)
        for item in report["large_non_r7_files"]
    )

    destination_file = destination / "model-layer-003.safetensors"
    original_mtime = destination_file.stat().st_mtime_ns
    corrupt_payload = bytearray(payload)
    corrupt_payload[0] ^= 1
    destination_file.write_bytes(corrupt_payload)
    os.utime(destination_file, ns=(original_mtime, original_mtime))
    with pytest.raises(ValueError, match="large non-R7 copied file failed"):
        MODULE.validate_copy(source, destination)

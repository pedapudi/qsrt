from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "index_glm52_experiment_artifacts.py"
SPEC = importlib.util.spec_from_file_location("index_glm52_experiment_artifacts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_index_hashes_every_readable_regular_file_except_its_output(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference" / "logits.bin"
    result = tmp_path / "results" / "report.json"
    reference.parent.mkdir()
    result.parent.mkdir()
    reference.write_bytes(b"reference")
    result.write_bytes(b"result")
    output = tmp_path / "ARTIFACT_INDEX.json"
    output.write_text("superseded index")

    index = MODULE.build_index(tmp_path, output)

    assert index["regular_file_count"] == 2
    assert index["regular_file_bytes"] == len(b"reference") + len(b"result")
    assert index["unhashed_file_count"] == 0
    by_path = {entry["relative_path"]: entry for entry in index["files"]}
    assert set(by_path) == {"reference/logits.bin", "results/report.json"}
    assert by_path["reference/logits.bin"]["sha256"] == hashlib.sha256(
        b"reference"
    ).hexdigest()
    assert by_path["reference/logits.bin"]["cleanup_policy"] == "retain"
    assert by_path["results/report.json"]["category"] == "experiment_result"


def test_build_index_does_not_follow_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source" / "working-tree" / "file.py"
    source.parent.mkdir(parents=True)
    source.write_text("pass\n")
    (tmp_path / "source" / "alias.py").symlink_to(source)

    index = MODULE.build_index(tmp_path, tmp_path / "ARTIFACT_INDEX.json")

    assert [entry["relative_path"] for entry in index["files"]] == [
        "source/working-tree/file.py"
    ]


def test_build_index_records_a_hash_error_without_omitting_the_file(
    tmp_path: Path, monkeypatch
) -> None:
    result = tmp_path / "results" / "root-owned.json"
    result.parent.mkdir()
    result.write_text("result")

    def denied(_path: Path) -> str:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(MODULE, "_sha256", denied)
    index = MODULE.build_index(tmp_path, tmp_path / "ARTIFACT_INDEX.json")

    assert index["regular_file_count"] == 1
    assert index["unhashed_file_count"] == 1
    assert index["files"] == [
        {
            "relative_path": "results/root-owned.json",
            "size_bytes": len("result"),
            "category": "experiment_result",
            "cleanup_policy": "retain",
            "sha256": None,
            "hash_error": "PermissionError: Permission denied",
        }
    ]


def test_model_cache_is_classified_for_post_experiment_cleanup(
    tmp_path: Path,
) -> None:
    cached_shard = tmp_path / "model-cache" / "checkpoint" / "weights.safetensors"
    cached_shard.parent.mkdir(parents=True)
    cached_shard.write_bytes(b"host-local checkpoint copy")

    index = MODULE.build_index(tmp_path, tmp_path / "ARTIFACT_INDEX.json")

    assert index["files"] == [
        {
            "relative_path": "model-cache/checkpoint/weights.safetensors",
            "size_bytes": len(b"host-local checkpoint copy"),
            "category": "local_model_cache",
            "cleanup_policy": "removable_after_experiments",
            "sha256": hashlib.sha256(b"host-local checkpoint copy").hexdigest(),
        }
    ]


@pytest.mark.parametrize(
    ("directory", "category", "cleanup_policy"),
    [
        (
            "captures",
            "routed_activation_capture",
            "retain_until_results_are_consolidated",
        ),
        ("launch-records", "experiment_launch_record", "retain"),
        ("runtime-control", "runtime_control", "removable_after_experiments"),
        ("logs", "experiment_log", "retain"),
    ],
)
def test_operational_artifacts_have_cleanup_categories(
    tmp_path: Path,
    directory: str,
    category: str,
    cleanup_policy: str,
) -> None:
    path = tmp_path / directory / "record.json"
    path.parent.mkdir()
    path.write_text("{}")

    entry = MODULE.build_index(tmp_path, tmp_path / "ARTIFACT_INDEX.json")[
        "files"
    ][0]

    assert entry["category"] == category
    assert entry["cleanup_policy"] == cleanup_policy

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from qsrt.kimi_output_factors import (
    KimiOutputFactorArchive,
    OutputFactorSums,
)


def _sums(layer: int) -> OutputFactorSums:
    left = torch.tensor(
        [[2.0 + layer, 1.0], [0.0, 4.0 + layer]],
        dtype=torch.float32,
    )
    right = torch.tensor(
        [[6.0 + layer, -1.0], [1.0, 2.0 + layer]],
        dtype=torch.float32,
    )
    return OutputFactorSums(
        split_a=left,
        split_a_rows=2,
        split_b=right,
        split_b_rows=3,
    )


def test_segment_commit_seals_document_disjoint_factors(tmp_path: Path) -> None:
    archive = KimiOutputFactorArchive.create(
        tmp_path / "factors",
        num_layers=5,
        dimension=2,
        expected_layers=(1, 2, 3, 4),
        provenance={"weight_revision": "immutable-test"},
    )
    for first, end in ((0, 3), (3, 5)):
        writer = archive.begin_segment(first, end)
        for layer in range(max(first, 1), end):
            writer.add(layer, _sums(layer))
        writer.seal()
        writer.commit()

    archive.seal()
    reopened = KimiOutputFactorArchive(archive.root)
    assert reopened.manifest["complete"] is True
    assert reopened.manifest["provenance"] == {
        "weight_revision": "immutable-test"
    }
    assert len(reopened.manifest["segments"]) == 2

    layer = reopened.manifest["segments"][0]["layers"][0]
    path = reopened.layer_path(1)
    assert path == (
        reopened.root
        / reopened.manifest["segments"][0]["directory"]
        / layer["file"]
    )
    tensors = load_file(path)
    sums = _sums(1)
    expected_a = sums.split_a / sums.split_a_rows
    expected_a = (expected_a + expected_a.T) * 0.5
    expected_b = sums.split_b / sums.split_b_rows
    expected_b = (expected_b + expected_b.T) * 0.5
    expected = (sums.split_a + sums.split_b) / 5
    expected = (expected + expected.T) * 0.5
    assert torch.equal(tensors["output_hessian_split_a"], expected_a)
    assert torch.equal(tensors["output_hessian_split_b"], expected_b)
    assert torch.equal(tensors["output_hessian"], expected)


def test_layer_path_rejects_uncommitted_and_corrupt_factors(tmp_path: Path) -> None:
    archive = KimiOutputFactorArchive.create(
        tmp_path / "factors",
        num_layers=3,
        dimension=2,
        expected_layers=(1, 2),
    )
    writer = archive.begin_segment(0, 2)
    record = writer.add(1, _sums(1))
    writer.seal()
    writer.commit()
    reopened = KimiOutputFactorArchive(archive.root)
    assert reopened.layer_path(1).name == record.file
    with pytest.raises(FileNotFoundError, match="has not committed"):
        reopened.layer_path(2)

    path = reopened.layer_path(1, verify_hash=False)
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 0x01]))
    with pytest.raises(ValueError, match="hash mismatch"):
        reopened.layer_path(1)


def test_pending_segment_requires_matching_cotangent_operation(tmp_path: Path) -> None:
    archive = KimiOutputFactorArchive.create(
        tmp_path / "factors",
        num_layers=3,
        dimension=2,
        expected_layers=(1, 2),
    )
    writer = archive.begin_segment(0, 3)
    writer.add(1, _sums(1))
    writer.add(2, _sums(2))
    writer.seal()

    assert archive.recover_pending(("segment-003-005",)) == ()
    promoted = archive.recover_pending(("segment-000-003",))
    assert len(promoted) == 1
    assert promoted[0].name == "segment-000-003"
    assert KimiOutputFactorArchive(archive.root).manifest["complete"] is True


def test_uncommitted_pending_segment_is_discarded_for_replay(tmp_path: Path) -> None:
    archive = KimiOutputFactorArchive.create(
        tmp_path / "factors",
        num_layers=3,
        dimension=2,
        expected_layers=(1, 2),
    )
    writer = archive.begin_segment(0, 3)
    writer.add(1, _sums(1))
    pending = writer.path

    discarded = archive.discard_uncommitted_pending(())
    assert discarded == (pending,)
    assert not pending.exists()

    replacement = archive.begin_segment(0, 3)
    replacement.add(1, _sums(1))
    replacement.add(2, _sums(2))
    replacement.seal()
    assert archive.discard_uncommitted_pending(("segment-000-003",)) == ()
    assert replacement.path.exists()


def test_corrupt_committed_factor_is_rejected(tmp_path: Path) -> None:
    archive = KimiOutputFactorArchive.create(
        tmp_path / "factors",
        num_layers=2,
        dimension=2,
        expected_layers=(1,),
    )
    writer = archive.begin_segment(0, 2)
    record = writer.add(1, _sums(1))
    writer.seal()
    directory = writer.commit()
    path = directory / record.file
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 0x01]))
    with pytest.raises(ValueError, match="missing or corrupt"):
        archive.seal()


def test_segment_manifest_requires_every_expected_layer(tmp_path: Path) -> None:
    archive = KimiOutputFactorArchive.create(
        tmp_path / "factors",
        num_layers=4,
        dimension=2,
        expected_layers=(1, 2, 3),
    )
    writer = archive.begin_segment(0, 4)
    writer.add(1, _sums(1))
    with pytest.raises(ValueError, match="missing"):
        writer.seal()
    manifest = json.loads(archive.manifest_path.read_text())
    assert manifest["complete"] is False

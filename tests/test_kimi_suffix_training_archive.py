from __future__ import annotations

from pathlib import Path

import pytest
import torch

from qsrt.kimi_boundary_slabs import DocumentIndex
from qsrt.kimi_suffix_training_archive import KimiSuffixTrainingArchive


def _documents() -> DocumentIndex:
    return DocumentIndex(
        input_ids=torch.arange(10, dtype=torch.int32),
        offsets=torch.tensor([0, 3, 7, 10], dtype=torch.int64),
        identifiers=("alpha", "beta", "gamma"),
    )


def _write_student(archive: KimiSuffixTrainingArchive) -> None:
    for boundary in archive.student.retained_boundaries:
        value = (
            torch.arange(40, dtype=torch.float32).reshape(10, 4) + boundary * 100
        ).to(torch.bfloat16)
        archive.student.prepare_boundary(boundary)
        with archive.student.extent_writer(
            boundary,
            writer_id="all",
            first_token=0,
            end_token=10,
            direct=False,
        ) as writer:
            writer.append(value)
            writer.finish()
        archive.student.seal_boundary(boundary)
    archive.student.seal()


def test_suffix_training_archive_roundtrip(tmp_path: Path) -> None:
    archive = KimiSuffixTrainingArchive.create(
        tmp_path / "archive",
        documents=_documents(),
        num_layers=8,
        hidden_dimension=4,
        attn_res_block_size=2,
        cut_layer=6,
        provenance={"population": "test"},
    )
    assert archive.student.retained_boundaries == (0, 2, 4, 6)
    _write_student(archive)

    teacher = torch.arange(40, dtype=torch.float32).reshape(10, 4).to(torch.bfloat16)
    partitions = archive.load_documents().contiguous_partitions(2)
    for partition in partitions:
        with archive.teacher_target_writer(
            writer_id=f"writer-{partition.index}",
            first_token=partition.first_token,
            end_token=partition.end_token,
            direct=False,
        ) as writer:
            writer.append(teacher[partition.first_token : partition.end_token])
            receipt = writer.finish()
        archive.record_teacher_target(receipt)
        assert archive.record_teacher_target(receipt).is_file()
    archive.seal_teacher_target()
    archive.seal()

    reopened = KimiSuffixTrainingArchive(
        archive.root,
        require_complete=True,
    )
    assert reopened.complete
    assert torch.equal(
        reopened.read_teacher_normalized(
            1,
            9,
            direct=False,
            pin_memory=False,
        ),
        teacher[1:9],
    )
    document = reopened.load_document(1, direct=False)
    assert document.identifier == "beta"
    assert tuple(document.student_boundary.hidden.shape) == (4, 4)
    assert tuple(document.student_boundary.residual.shape) == (4, 3, 4)
    assert torch.equal(document.teacher_normalized, teacher[3:7])


def test_suffix_training_archive_requires_complete_teacher_cover(
    tmp_path: Path,
) -> None:
    archive = KimiSuffixTrainingArchive.create(
        tmp_path / "archive",
        documents=_documents(),
        num_layers=8,
        hidden_dimension=4,
        attn_res_block_size=2,
        cut_layer=6,
    )
    with archive.teacher_target_writer(
        writer_id="partial",
        first_token=0,
        end_token=3,
        direct=False,
    ) as writer:
        writer.append(torch.zeros((3, 4), dtype=torch.bfloat16))
        archive.record_teacher_target(writer.finish())
    with pytest.raises(ValueError, match="expected 10"):
        archive.seal_teacher_target()

    assert archive.discard_teacher_target_receipts() == ("partial.json",)
    assert archive._teacher_receipts() == ()


def test_suffix_training_archive_allows_teacher_to_seal_before_student(
    tmp_path: Path,
) -> None:
    archive = KimiSuffixTrainingArchive.create(
        tmp_path / "archive",
        documents=_documents(),
        num_layers=8,
        hidden_dimension=4,
        attn_res_block_size=2,
        cut_layer=6,
    )
    teacher = torch.arange(40, dtype=torch.float32).reshape(10, 4).to(torch.bfloat16)
    with archive.teacher_target_writer(
        writer_id="all",
        first_token=0,
        end_token=10,
        direct=False,
    ) as writer:
        writer.append(teacher)
        archive.record_teacher_target(writer.finish())
    archive.seal_teacher_target()

    assert not archive.complete
    assert bool(archive.manifest["teacher_target"]["sealed"])
    _write_student(archive)
    archive.seal()

    reopened = KimiSuffixTrainingArchive(archive.root, require_complete=True)
    assert reopened.complete
    assert torch.equal(
        reopened.read_teacher_normalized(0, 10, direct=False, pin_memory=False),
        teacher,
    )


def test_suffix_training_archive_requires_residual_boundary_cut(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="geometry"):
        KimiSuffixTrainingArchive.create(
            tmp_path / "invalid",
            documents=_documents(),
            num_layers=8,
            hidden_dimension=4,
            attn_res_block_size=2,
            cut_layer=5,
        )

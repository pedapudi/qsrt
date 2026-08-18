from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from qsrt.kimi_boundary_slabs import (
    DocumentIndex,
    KimiBoundarySlabArchive,
    verify_partition_cover,
)


def _documents() -> DocumentIndex:
    return DocumentIndex(
        input_ids=torch.arange(12, dtype=torch.int32),
        offsets=torch.tensor([0, 2, 7, 8, 12], dtype=torch.int64),
        identifiers=("alpha", "beta", "gamma", "delta"),
    )


def _archive(tmp_path: Path) -> KimiBoundarySlabArchive:
    return KimiBoundarySlabArchive.create(
        tmp_path / "capture",
        documents=_documents(),
        num_layers=2,
        hidden_dimension=4,
        attn_res_block_size=2,
        provenance={"checkpoint_revision": "test-revision"},
    )


def _write_boundary(
    archive: KimiBoundarySlabArchive,
    boundary: int,
    value: torch.Tensor,
) -> None:
    archive.prepare_boundary(boundary)
    partitions = archive.load_documents().contiguous_partitions(2)
    for partition in partitions:
        with archive.extent_writer(
            boundary,
            writer_id=f"worker-{partition.index:03d}",
            first_token=partition.first_token,
            end_token=partition.end_token,
            direct=False,
        ) as writer:
            writer.append(value[partition.first_token : partition.end_token])
            writer.finish()
    archive.seal_boundary(boundary)


def test_document_partitions_preserve_documents_and_balance_tokens() -> None:
    documents = _documents()
    partitions = documents.contiguous_partitions(3)
    verify_partition_cover(
        partitions,
        document_count=documents.document_count,
        token_count=documents.token_count,
    )
    assert [(value.first_document, value.end_document) for value in partitions] == [
        (0, 1),
        (1, 2),
        (2, 4),
    ]
    assert [value.token_count for value in partitions] == [2, 5, 5]


def test_slab_archive_roundtrip_and_completion(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    expected = []
    for boundary in range(3):
        value = (
            torch.arange(48, dtype=torch.float32).reshape(12, 4) + boundary * 100
        ).to(torch.bfloat16)
        expected.append(value)
        _write_boundary(archive, boundary, value)
    manifest = archive.seal()
    assert manifest.is_file()

    reopened = KimiBoundarySlabArchive(archive.root, require_complete=True)
    assert reopened.complete
    assert torch.equal(reopened.load_documents().input_ids, _documents().input_ids)
    for boundary, value in enumerate(expected):
        assert torch.equal(
            reopened.read_cpu(boundary, 1, 11, direct=False, pin_memory=False),
            value[1:11],
        )


def test_selective_archive_seals_only_declared_boundaries(tmp_path: Path) -> None:
    archive = KimiBoundarySlabArchive.create(
        tmp_path / "selective",
        documents=_documents(),
        num_layers=8,
        hidden_dimension=4,
        attn_res_block_size=2,
        retained_boundaries=(0, 2, 4, 6, 8),
    )
    value = torch.arange(48, dtype=torch.float32).reshape(12, 4).to(torch.bfloat16)
    for boundary in archive.retained_boundaries:
        _write_boundary(archive, boundary, value + boundary)
    archive.seal()

    reopened = KimiBoundarySlabArchive(archive.root, require_complete=True)
    assert reopened.retained_boundaries == (0, 2, 4, 6, 8)
    assert int(reopened.manifest["boundary_count"]) == 5
    assert not reopened.boundary_path(1).exists()
    assert torch.equal(
        reopened.read_cpu(8, 0, 12, direct=False, pin_memory=False),
        value + 8,
    )


def test_boundary_requires_complete_nonoverlapping_receipts(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    archive.prepare_boundary(0)
    value = torch.zeros((4, 4), dtype=torch.bfloat16)
    with archive.extent_writer(
        0,
        writer_id="partial",
        first_token=0,
        end_token=4,
        direct=False,
    ) as writer:
        writer.append(value)
        writer.finish()
    with pytest.raises(ValueError, match="expected 12"):
        archive.seal_boundary(0)


def test_writer_refuses_incomplete_extent(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    archive.prepare_boundary(0)
    with archive.extent_writer(
        0,
        writer_id="worker",
        first_token=0,
        end_token=4,
        direct=False,
    ) as writer:
        writer.append(torch.zeros((3, 4), dtype=torch.bfloat16))
        with pytest.raises(ValueError, match="expected 4"):
            writer.finish()
    assert not archive.receipt_path(0, "worker").exists()


def test_attention_residuals_are_boundary_aliases(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    assert archive.residual_boundaries_before(0) == ()
    assert archive.residual_boundaries_before(1) == (0,)

    wider = KimiBoundarySlabArchive.create(
        tmp_path / "wider",
        documents=_documents(),
        num_layers=8,
        hidden_dimension=4,
        attn_res_block_size=3,
    )
    assert wider.residual_boundaries_before(8 - 1) == (0, 3, 6)


def test_manifest_records_durable_geometry(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    document = json.loads((archive.root / ".manifest.partial.json").read_text())
    assert document["kind"] == "Kimi-K3 exact layer-boundary slab archive"
    assert document["boundary_count"] == 3
    assert document["row_bytes"] == 8
    assert document["token_count"] == 12
    assert document["provenance"] == {"checkpoint_revision": "test-revision"}


def test_resume_keeps_sealed_prefix_and_discards_incomplete_receipts(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    value = torch.zeros((12, 4), dtype=torch.bfloat16)
    _write_boundary(archive, 0, value)
    archive.prepare_boundary(1)
    with archive.extent_writer(
        1,
        writer_id="interrupted",
        first_token=0,
        end_token=4,
        direct=False,
    ) as writer:
        writer.append(value[:4])
        writer.finish()

    assert archive.sealed_boundary_prefix() == (0,)
    assert archive.discard_unsealed_receipts() == (1,)
    assert not archive.receipt_path(1, "interrupted").exists()
    assert archive.sealed_boundary_prefix() == (0,)


def test_selective_archive_resume_uses_retained_boundary_prefix(
    tmp_path: Path,
) -> None:
    archive = KimiBoundarySlabArchive.create(
        tmp_path / "selective-resume",
        documents=_documents(),
        num_layers=8,
        hidden_dimension=4,
        attn_res_block_size=2,
        retained_boundaries=(0, 2, 4, 6, 8),
    )
    value = torch.zeros((12, 4), dtype=torch.bfloat16)
    _write_boundary(archive, 0, value)
    _write_boundary(archive, 2, value)
    archive.prepare_boundary(4)
    with archive.extent_writer(
        4,
        writer_id="interrupted",
        first_token=0,
        end_token=4,
        direct=False,
    ) as writer:
        writer.append(value[:4])
        writer.finish()
    assert archive.sealed_boundary_prefix() == (0, 2)
    assert archive.discard_unsealed_receipts() == (4,)
    assert not archive.receipt_path(4, "interrupted").exists()
    assert archive.sealed_boundary_prefix() == (0, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_writer_preserves_order_across_double_buffers(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    archive.prepare_boundary(0)
    value = torch.arange(48, dtype=torch.float32).reshape(12, 4).to(torch.bfloat16)
    with archive.cuda_extent_writer(
        0,
        writer_id="cuda",
        first_token=0,
        end_token=12,
        device="cuda:0",
        buffer_tokens=3,
        direct=False,
    ) as writer:
        writer.append(value[:5].cuda())
        writer.append(value[5:].cuda())
        receipt = writer.finish()
    archive.seal_boundary(0)
    assert receipt.bytes == value.numel() * value.element_size()
    assert torch.equal(
        archive.read_cpu(0, 0, 12, direct=False, pin_memory=False), value
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_writer_waits_for_ephemeral_async_producers(tmp_path: Path) -> None:
    token_count = 8192
    documents = DocumentIndex(
        input_ids=torch.arange(token_count, dtype=torch.int32),
        offsets=torch.tensor([0, token_count], dtype=torch.int64),
        identifiers=("async-producer",),
    )
    archive = KimiBoundarySlabArchive.create(
        tmp_path / "async-producer-capture",
        documents=documents,
        num_layers=1,
        hidden_dimension=4,
        attn_res_block_size=2,
    )
    archive.prepare_boundary(0)
    with archive.cuda_extent_writer(
        0,
        writer_id="cuda",
        first_token=0,
        end_token=token_count,
        device="cuda:0",
        buffer_tokens=8,
        direct=False,
    ) as writer:
        for token in range(token_count):
            writer.append(
                torch.full(
                    (1, 4),
                    float(token),
                    dtype=torch.bfloat16,
                    device="cuda:0",
                )
            )
        writer.finish()
    expected = (
        torch.arange(token_count, dtype=torch.float32)
        .to(torch.bfloat16)
        .unsqueeze(1)
        .expand(-1, 4)
    )
    assert torch.equal(
        archive.read_cpu(
            0,
            0,
            token_count,
            direct=False,
            pin_memory=False,
        ),
        expected,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pinning unavailable")
def test_direct_io_roundtrip_on_data_raid() -> None:
    root = Path("/data/kquant/qsrt-boundary-slab-direct-test")
    if root.exists():
        pytest.skip(f"temporary test path already exists: {root}")
    documents = DocumentIndex(
        input_ids=torch.arange(4, dtype=torch.int32),
        offsets=torch.tensor([0, 4], dtype=torch.int64),
        identifiers=("direct",),
    )
    try:
        archive = KimiBoundarySlabArchive.create(
            root,
            documents=documents,
            num_layers=1,
            hidden_dimension=7168,
            attn_res_block_size=12,
        )
        archive.prepare_boundary(0)
        value = torch.arange(
            4 * 7168, dtype=torch.float32
        ).reshape(4, 7168).to(torch.bfloat16).pin_memory()
        with archive.extent_writer(
            0,
            writer_id="direct",
            first_token=0,
            end_token=4,
            direct=True,
        ) as writer:
            writer.append(value)
            writer.finish()
        archive.seal_boundary(0)
        observed = archive.read_cpu(0, 0, 4, direct=True)
        assert torch.equal(observed, value)
    finally:
        if root.exists():
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()

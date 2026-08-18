from __future__ import annotations

from pathlib import Path

import pytest
import torch

from qsrt.kimi_boundary_slabs import DocumentIndex, KimiBoundarySlabArchive
from qsrt.kimi_cotangent_slabs import KimiCotangentSlabWorkspace


def _sealed_boundary_archive(root: Path) -> KimiBoundarySlabArchive:
    documents = DocumentIndex(
        input_ids=torch.arange(5, dtype=torch.int32),
        offsets=torch.tensor([0, 2, 5], dtype=torch.int64),
        identifiers=("a", "b"),
    )
    archive = KimiBoundarySlabArchive.create(
        root,
        documents=documents,
        num_layers=5,
        hidden_dimension=4,
        attn_res_block_size=2,
    )
    for boundary in range(6):
        archive.prepare_boundary(boundary)
        with archive.extent_writer(
            boundary,
            writer_id="all",
            first_token=0,
            end_token=5,
            direct=False,
        ) as writer:
            writer.append(torch.full((5, 4), boundary, dtype=torch.bfloat16))
            writer.finish()
        archive.seal_boundary(boundary)
    archive.seal()
    return KimiBoundarySlabArchive(root, require_complete=True)


def _write_update(update, values: dict[str, torch.Tensor]) -> None:
    for role, value in values.items():
        writer = update.writer(role, direct=False)
        try:
            writer.append(value)
            update.record(writer.finish())
        finally:
            writer.close()
    update.commit()


def test_double_buffered_segment_commit_and_reopen(tmp_path: Path) -> None:
    boundary = _sealed_boundary_archive(tmp_path / "boundaries")
    workspace = KimiCotangentSlabWorkspace.create(
        tmp_path / "cotangents",
        boundary_archive=boundary,
    )
    suffix = workspace.begin_suffix()
    suffix_values = {
        "chain": torch.full((5, 4), 5, dtype=torch.bfloat16),
        "residual-000": torch.full((5, 4), 10, dtype=torch.bfloat16),
        "residual-002": torch.full((5, 4), 12, dtype=torch.bfloat16),
        "residual-004": torch.full((5, 4), 14, dtype=torch.bfloat16),
    }
    _write_update(suffix, suffix_values)
    assert workspace.manifest["chain_boundary"] == 5

    update = workspace.begin_segment(4)
    old_slot = workspace.active_slots["chain"]
    _write_update(
        update,
        {
            "chain": torch.full((5, 4), 4, dtype=torch.bfloat16),
            "residual-000": torch.full((5, 4), 20, dtype=torch.bfloat16),
            "residual-002": torch.full((5, 4), 22, dtype=torch.bfloat16),
        },
    )
    reopened = KimiCotangentSlabWorkspace(workspace.root)
    assert reopened.manifest["chain_boundary"] == 4
    assert reopened.active_slots["chain"] == 1 - old_slot
    assert torch.equal(
        reopened.read_chain(1, 4, direct=False),
        torch.full((3, 4), 4, dtype=torch.bfloat16),
    )
    assert torch.equal(
        reopened.read_residual(0, 0, 5, direct=False),
        torch.full((5, 4), 20, dtype=torch.bfloat16),
    )
    assert torch.equal(
        reopened.read_residual(4, 0, 5, direct=False),
        torch.full((5, 4), 14, dtype=torch.bfloat16),
    )


def test_uncommitted_slot_does_not_replace_active_state(tmp_path: Path) -> None:
    boundary = _sealed_boundary_archive(tmp_path / "boundaries")
    workspace = KimiCotangentSlabWorkspace.create(
        tmp_path / "cotangents",
        boundary_archive=boundary,
    )
    suffix = workspace.begin_suffix()
    _write_update(
        suffix,
        {
            "chain": torch.ones((5, 4), dtype=torch.bfloat16),
            "residual-000": torch.ones((5, 4), dtype=torch.bfloat16),
            "residual-002": torch.ones((5, 4), dtype=torch.bfloat16),
            "residual-004": torch.ones((5, 4), dtype=torch.bfloat16),
        },
    )
    old_slot = workspace.active_slots["chain"]
    update = workspace.begin_segment(4)
    writer = update.writer("chain", direct=False)
    writer.append(torch.zeros((2, 4), dtype=torch.bfloat16))
    writer.close()

    reopened = KimiCotangentSlabWorkspace(workspace.root)
    assert reopened.active_slots["chain"] == old_slot
    assert reopened.manifest["chain_boundary"] == 5
    assert torch.equal(
        reopened.read_chain(0, 5, direct=False),
        torch.ones((5, 4), dtype=torch.bfloat16),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_writer_preserves_order(tmp_path: Path) -> None:
    boundary = _sealed_boundary_archive(tmp_path / "boundaries")
    workspace = KimiCotangentSlabWorkspace.create(
        tmp_path / "cotangents",
        boundary_archive=boundary,
    )
    update = workspace.begin_suffix()
    for role in update.roles:
        writer = update.writer(role, direct=False)
        cuda_writer = __import__(
            "qsrt.kimi_cotangent_slabs", fromlist=["CudaBf16SlabWriter"]
        ).CudaBf16SlabWriter(
            writer,
            device="cuda:0",
            buffer_tokens=2,
        )
        try:
            value = torch.arange(20, dtype=torch.bfloat16, device="cuda:0").reshape(5, 4)
            cuda_writer.append(value)
            update.record(cuda_writer.finish())
        finally:
            cuda_writer.close()
    update.commit()
    assert torch.equal(
        workspace.read_chain(0, 5, direct=False),
        torch.arange(20, dtype=torch.bfloat16).reshape(5, 4),
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
        tmp_path / "boundaries",
        documents=documents,
        num_layers=1,
        hidden_dimension=4,
        attn_res_block_size=2,
    )
    for boundary in range(2):
        archive.prepare_boundary(boundary)
        with archive.extent_writer(
            boundary,
            writer_id="all",
            first_token=0,
            end_token=token_count,
            direct=False,
        ) as writer:
            writer.append(torch.zeros((token_count, 4), dtype=torch.bfloat16))
            writer.finish()
        archive.seal_boundary(boundary)
    archive.seal()
    workspace = KimiCotangentSlabWorkspace.create(
        tmp_path / "cotangents",
        boundary_archive=KimiBoundarySlabArchive(
            archive.root,
            require_complete=True,
        ),
    )
    update = workspace.begin_suffix()
    for role in update.roles:
        writer = update.writer(role, direct=False)
        cuda_writer = __import__(
            "qsrt.kimi_cotangent_slabs", fromlist=["CudaBf16SlabWriter"]
        ).CudaBf16SlabWriter(
            writer,
            device="cuda:0",
            buffer_tokens=8,
        )
        try:
            for token in range(token_count):
                cuda_writer.append(
                    torch.full(
                        (1, 4),
                        float(token),
                        dtype=torch.bfloat16,
                        device="cuda:0",
                    )
                )
            update.record(cuda_writer.finish())
        finally:
            cuda_writer.close()
    update.commit()
    expected = (
        torch.arange(token_count, dtype=torch.float32)
        .to(torch.bfloat16)
        .unsqueeze(1)
        .expand(-1, 4)
    )
    assert torch.equal(
        workspace.read_chain(0, token_count, direct=False),
        expected,
    )

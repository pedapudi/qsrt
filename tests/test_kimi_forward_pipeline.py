from __future__ import annotations

import threading
from pathlib import Path

import pytest
import torch

from qsrt.kimi_boundary_slabs import DocumentIndex, KimiBoundarySlabArchive
from qsrt.kimi_forward_pipeline import (
    KimiForwardPipeline,
    PipelineActivation,
    stage_layers,
)


def test_cyclic_layer_assignment() -> None:
    assert stage_layers(stage=0, stage_count=3, num_layers=8) == (0, 3, 6)
    assert stage_layers(stage=1, stage_count=3, num_layers=8) == (1, 4, 7)
    assert stage_layers(stage=2, stage_count=3, num_layers=8) == (2, 5)


class _SyntheticAdapter:
    def __init__(self, residual_block_size: int):
        self.residual_block_size = residual_block_size
        self.loads: list[tuple[int, int]] = []
        self._lock = threading.Lock()

    def load_layer(
        self,
        layer: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        with self._lock:
            self.loads.append((device.index, layer))
        return (
            torch.tensor(layer + 1, dtype=torch.bfloat16, device=device),
            {"layer": layer},
        )

    def forward_layer(
        self,
        module: torch.Tensor,
        *,
        layer: int,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_input_blocks = len(range(0, layer, self.residual_block_size))
        assert block_residual.shape[1] == expected_input_blocks
        output_residual = block_residual
        if layer % self.residual_block_size == 0:
            output_residual = torch.cat(
                [output_residual, hidden_states.view(-1, hidden_states.shape[-1]).unsqueeze(1)],
                dim=1,
            )
        return hidden_states + module, output_residual

    def release_layer(self, module: torch.Tensor) -> None:
        del module


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="the cyclic pipeline test requires two CUDA devices",
)
def test_pipeline_segments_across_devices_and_archives_every_boundary(
    tmp_path: Path,
) -> None:
    documents = DocumentIndex(
        input_ids=torch.arange(6, dtype=torch.int32),
        offsets=torch.tensor([0, 2, 5, 6], dtype=torch.int64),
        identifiers=("first", "second", "third"),
    )
    archive = KimiBoundarySlabArchive.create(
        tmp_path / "capture",
        documents=documents,
        num_layers=5,
        hidden_dimension=4,
        attn_res_block_size=2,
    )
    adapter = _SyntheticAdapter(residual_block_size=2)

    def inputs():
        for document in range(documents.document_count):
            first, end = documents.document_extent(document)
            hidden = torch.full(
                (1, end - first, 4),
                float(document),
                dtype=torch.bfloat16,
                device="cuda:0",
            )
            yield PipelineActivation(
                document=document,
                first_token=first,
                end_token=end,
                hidden_states=hidden,
                block_residual=torch.empty(
                    (end - first, 0, 4),
                    dtype=torch.bfloat16,
                    device="cuda:0",
                ),
            )

    result = KimiForwardPipeline(
        adapter=adapter,
        archive=archive,
        devices=("cuda:0", "cuda:1"),
        queue_depth=1,
        slab_buffer_tokens=2,
        direct_io=False,
    ).run(inputs())

    assert [record.layer for record in result.records] == list(range(5))
    assert [layer for device, layer in adapter.loads if device == 0] == [0, 2, 4]
    assert [layer for device, layer in adapter.loads if device == 1] == [1, 3]
    reopened = KimiBoundarySlabArchive(archive.root, require_complete=True)
    base = torch.cat(
        [
            torch.full((end - first, 4), float(document), dtype=torch.bfloat16)
            for document in range(documents.document_count)
            for first, end in [documents.document_extent(document)]
        ]
    )
    running = base
    assert torch.equal(reopened.read_cpu(0, 0, 6, direct=False), running)
    for boundary in range(1, 6):
        running = running + boundary
        assert torch.equal(
            reopened.read_cpu(boundary, 0, 6, direct=False),
            running,
        )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="the segmented pipeline test requires two CUDA devices",
)
def test_segment_handoff_is_bounded_beyond_queue_capacity(tmp_path: Path) -> None:
    document_count = 128
    documents = DocumentIndex(
        input_ids=torch.arange(document_count, dtype=torch.int32),
        offsets=torch.arange(document_count + 1, dtype=torch.int64),
        identifiers=tuple(str(index) for index in range(document_count)),
    )
    archive = KimiBoundarySlabArchive.create(
        tmp_path / "capture",
        documents=documents,
        num_layers=5,
        hidden_dimension=4,
        attn_res_block_size=2,
    )

    def inputs():
        for document in range(document_count):
            hidden = torch.full(
                (1, 1, 4),
                float(document),
                dtype=torch.bfloat16,
                device="cuda:0",
            )
            yield PipelineActivation(
                document=document,
                first_token=document,
                end_token=document + 1,
                hidden_states=hidden,
                block_residual=torch.empty(
                    (1, 0, 4), dtype=torch.bfloat16, device="cuda:0"
                ),
            )

    result = KimiForwardPipeline(
        adapter=_SyntheticAdapter(residual_block_size=2),
        archive=archive,
        devices=("cuda:0", "cuda:1"),
        queue_depth=1,
        slab_buffer_tokens=8,
        direct_io=False,
    ).run(inputs())

    assert len(result.records) == 5
    expected = torch.arange(document_count, dtype=torch.bfloat16).unsqueeze(1)
    expected = expected.expand(-1, 4) + 15
    assert torch.equal(
        archive.read_cpu(5, 0, document_count, direct=False),
        expected,
    )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="the restart pipeline test requires two CUDA devices",
)
def test_pipeline_restarts_from_a_sealed_boundary(tmp_path: Path) -> None:
    documents = DocumentIndex(
        input_ids=torch.arange(6, dtype=torch.int32),
        offsets=torch.tensor([0, 2, 5, 6], dtype=torch.int64),
        identifiers=("first", "second", "third"),
    )
    archive = KimiBoundarySlabArchive.create(
        tmp_path / "capture",
        documents=documents,
        num_layers=5,
        hidden_dimension=4,
        attn_res_block_size=2,
    )
    base = torch.cat(
        [
            torch.full((end - first, 4), float(document), dtype=torch.bfloat16)
            for document in range(documents.document_count)
            for first, end in [documents.document_extent(document)]
        ]
    )
    for boundary, value in ((0, base), (1, base + 1), (2, base + 3)):
        archive.prepare_boundary(boundary)
        with archive.extent_writer(
            boundary,
            writer_id="completed",
            first_token=0,
            end_token=documents.token_count,
            direct=False,
        ) as writer:
            writer.append(value)
            writer.finish()
        archive.seal_boundary(boundary)

    adapter = _SyntheticAdapter(residual_block_size=2)
    result = KimiForwardPipeline(
        adapter=adapter,
        archive=archive,
        devices=("cuda:0", "cuda:1"),
        queue_depth=1,
        slab_buffer_tokens=2,
        direct_io=False,
    ).run(None, start_layer=2)

    assert [record.layer for record in result.records] == [2, 3, 4]
    assert sorted(adapter.loads) == [(0, 2), (0, 4), (1, 3)]
    reopened = KimiBoundarySlabArchive(archive.root, require_complete=True)
    assert torch.equal(
        reopened.read_cpu(5, 0, documents.token_count, direct=False),
        base + 15,
    )

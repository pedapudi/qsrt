from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from instanttensor import Backend
from safetensors.torch import save_file

from qsrt.instanttensor_kimi import InstantTensorLoadConfig
from qsrt.kimi_boundary_slabs import DocumentIndex, KimiBoundarySlabArchive
from qsrt.kimi_cotangent_slabs import KimiCotangentSlabWorkspace
from qsrt.kimi_official_fisher import OfficialKimiFisherSuffix, SUFFIX_TENSORS
from qsrt.kimi_suffix_pipeline import KimiSuffixPipeline, document_fisher_seed


def _checkpoint(root: Path) -> Path:
    root.mkdir()
    tensors = {
        SUFFIX_TENSORS["lm_head"]: torch.tensor(
            [
                [1.0, 0.0, -0.5, 0.25],
                [0.0, 1.0, 0.5, -0.25],
                [-0.5, 0.25, 1.0, 0.0],
                [0.25, -0.5, 0.0, 1.0],
                [0.75, 0.5, -0.25, -1.0],
            ],
            dtype=torch.bfloat16,
        ),
        SUFFIX_TENSORS["final_norm"]: torch.tensor(
            [0.75, 1.0, 1.25, 0.5], dtype=torch.bfloat16
        ),
        SUFFIX_TENSORS["residual_norm"]: torch.tensor(
            [1.0, 0.5, 1.5, 0.75], dtype=torch.bfloat16
        ),
        SUFFIX_TENSORS["residual_projection"]: torch.tensor(
            [[0.5, -0.25, 1.0, 0.75]], dtype=torch.bfloat16
        ),
    }
    shard = "model.safetensors"
    save_file(tensors, root / shard)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}})
    )
    return root


def _archive(root: Path) -> KimiBoundarySlabArchive:
    documents = DocumentIndex(
        input_ids=torch.arange(5, dtype=torch.int32),
        offsets=torch.tensor([0, 2, 5], dtype=torch.int64),
        identifiers=("first", "second"),
    )
    archive = KimiBoundarySlabArchive.create(
        root,
        documents=documents,
        num_layers=5,
        hidden_dimension=4,
        attn_res_block_size=2,
    )
    base = torch.arange(20, dtype=torch.bfloat16).reshape(5, 4) / 8
    for boundary in range(6):
        archive.prepare_boundary(boundary)
        with archive.extent_writer(
            boundary,
            writer_id="all",
            first_token=0,
            end_token=5,
            direct=False,
        ) as writer:
            writer.append(base + boundary / 4)
            writer.finish()
        archive.seal_boundary(boundary)
    archive.seal()
    return KimiBoundarySlabArchive(root, require_complete=True)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="suffix pipeline test requires two CUDA devices",
)
def test_suffix_pipeline_matches_independent_document_vjps(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    archive = _archive(tmp_path / "boundaries")
    workspace = KimiCotangentSlabWorkspace.create(
        tmp_path / "cotangents",
        boundary_archive=archive,
    )
    load_config = InstantTensorLoadConfig(
        buffer_size=1 << 20,
        chunk_size=4096,
        concurrency=1,
        io_depth=3,
        backend=Backend.AIO_BUFFERED,
    )
    result = KimiSuffixPipeline(
        checkpoint=checkpoint,
        boundary_archive=archive,
        workspace=workspace,
        devices=("cuda:0", "cuda:1"),
        epsilon=1e-5,
        vocabulary_size=5,
        base_seed=91,
        lm_head_chunk_tokens=1,
        slab_buffer_tokens=2,
        direct_io=False,
        load_config=load_config,
    ).run()
    assert len(result.workers) == 2
    assert workspace.manifest["chain_boundary"] == 5

    samples = torch.from_file(
        str(result.sample_path),
        dtype=torch.int32,
        size=10,
    ).reshape(5, 2)
    suffix = OfficialKimiFisherSuffix(
        checkpoint=checkpoint,
        device="cuda:0",
        hidden_dimension=4,
        vocabulary_size=5,
        residual_block_count=3,
        epsilon=1e-5,
        load_config=load_config,
    )
    documents = archive.load_documents()
    for document in range(documents.document_count):
        first, end = documents.document_extent(document)
        expected = suffix.vjp(
            final_boundary=archive.read_cpu(5, first, end, direct=False).cuda().unsqueeze(0),
            residual_inputs=tuple(
                archive.read_cpu(boundary, first, end, direct=False).cuda()
                for boundary in (0, 2, 4)
            ),
            seed=document_fisher_seed(
                91,
                document=document,
                identifier=documents.identifiers[document],
            ),
            lm_head_chunk_tokens=2,
        )
        assert torch.equal(
            workspace.read_chain(first, end, direct=False),
            expected.chain_gradient.reshape(-1, 4).cpu(),
        )
        for boundary, gradient in zip(
            (0, 2, 4), expected.residual_gradients, strict=True
        ):
            assert torch.equal(
                workspace.read_residual(boundary, first, end, direct=False),
                gradient.cpu(),
            )
        assert torch.equal(samples[first:end, 0], expected.first_tokens.cpu().int())
        assert torch.equal(samples[first:end, 1], expected.second_tokens.cpu().int())

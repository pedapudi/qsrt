from __future__ import annotations

import json

import pytest
import torch

from qsrt.kimi_fisher_archive import (
    KimiBoundaryArchive,
    KimiBoundaryArchiveWriter,
    PARTIAL_MANIFEST_FILENAME,
)
from qsrt.kimi_stream import StreamState


def _state(
    next_layer: int,
    hidden: torch.Tensor,
    residuals: list[torch.Tensor],
) -> StreamState:
    tokens = hidden.shape[0] * hidden.shape[1]
    block_residual = (
        torch.stack(residuals, dim=1)
        if residuals
        else hidden.new_zeros(tokens, 0, hidden.shape[-1])
    )
    return StreamState(
        next_layer=next_layer,
        input_ids=torch.tensor([[4, 8, 15]], dtype=torch.long),
        hidden_states=hidden,
        block_residual=block_residual,
    )


def test_boundary_archive_round_trip_without_repeated_residuals(tmp_path) -> None:
    generator = torch.Generator().manual_seed(17)
    hiddens = [torch.randn(1, 3, 8, generator=generator).bfloat16() for _ in range(5)]
    residual_a = hiddens[0].reshape(3, 8).clone()
    residual_b = hiddens[2].reshape(3, 8).clone()
    states = [
        _state(0, hiddens[0], []),
        _state(1, hiddens[1], [residual_a]),
        _state(2, hiddens[2], [residual_a]),
        _state(3, hiddens[3], [residual_a, residual_b]),
        _state(4, hiddens[4], [residual_a, residual_b]),
    ]
    writer = KimiBoundaryArchiveWriter(
        tmp_path,
        num_layers=4,
        residual_block_size=2,
        provenance={"checkpoint": "/models/test"},
    )
    for state in states:
        writer.append(state)
    writer.seal()

    archive = KimiBoundaryArchive(tmp_path, verify_hashes=True)
    assert archive.manifest["provenance"] == {"checkpoint": "/models/test"}
    for expected in states:
        actual = archive.state(expected.next_layer)
        assert torch.equal(actual.input_ids, expected.input_ids)
        assert torch.equal(actual.hidden_states, expected.hidden_states)
        assert torch.equal(actual.block_residual, expected.block_residual)

    tensor_keys = []
    from safetensors import safe_open

    for next_layer in range(5):
        with safe_open(
            tmp_path / f"boundary-{next_layer:03d}.safetensors",
            framework="pt",
        ) as reader:
            tensor_keys.append(set(reader.keys()))
    assert sum("appended_residual" in keys for keys in tensor_keys) == 2


def test_boundary_archive_resumes_at_exact_next_boundary(tmp_path) -> None:
    hidden = torch.zeros(1, 3, 8, dtype=torch.bfloat16)
    first = KimiBoundaryArchiveWriter(
        tmp_path,
        num_layers=2,
        residual_block_size=2,
        provenance={"revision": "abc"},
    )
    first.append(_state(0, hidden, []))

    resumed = KimiBoundaryArchiveWriter(
        tmp_path,
        num_layers=2,
        residual_block_size=2,
        provenance={"revision": "abc"},
        resume=True,
    )
    assert resumed.next_layer == 1
    residual = hidden.reshape(3, 8)
    resumed.append(_state(1, hidden + 1, [residual]))
    resumed.append(_state(2, hidden + 2, [residual]))
    resumed.seal()
    assert not (tmp_path / PARTIAL_MANIFEST_FILENAME).exists()
    assert KimiBoundaryArchive(tmp_path).state(2).next_layer == 2


def test_boundary_archive_rejects_wrong_residual_geometry(tmp_path) -> None:
    hidden = torch.zeros(1, 3, 8, dtype=torch.bfloat16)
    writer = KimiBoundaryArchiveWriter(
        tmp_path,
        num_layers=2,
        residual_block_size=2,
    )
    writer.append(_state(0, hidden, []))
    with pytest.raises(ValueError, match="residual count"):
        writer.append(_state(1, hidden, []))


def test_boundary_archive_detects_file_corruption(tmp_path) -> None:
    hidden = torch.zeros(1, 3, 8, dtype=torch.bfloat16)
    writer = KimiBoundaryArchiveWriter(
        tmp_path,
        num_layers=1,
        residual_block_size=1,
    )
    writer.append(_state(0, hidden, []))
    writer.append(_state(1, hidden, [hidden.reshape(3, 8)]))
    writer.seal()

    path = tmp_path / "boundary-001.safetensors"
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="hash mismatch"):
        KimiBoundaryArchive(tmp_path, verify_hashes=True)


def test_boundary_archive_manifest_is_self_describing(tmp_path) -> None:
    hidden = torch.zeros(1, 3, 8, dtype=torch.bfloat16)
    writer = KimiBoundaryArchiveWriter(
        tmp_path,
        num_layers=1,
        residual_block_size=1,
    )
    writer.append(_state(0, hidden, []))
    writer.append(_state(1, hidden, [hidden.reshape(3, 8)]))
    writer.seal()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["batch_shape"] == [1, 3]
    assert manifest["hidden_dimension"] == 8
    assert manifest["hidden_dtype"] == "torch.bfloat16"
    assert [item["residual_count"] for item in manifest["boundaries"]] == [0, 1]

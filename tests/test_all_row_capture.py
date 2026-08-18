from __future__ import annotations

import json

import pytest
import torch

from qsrt.all_row_capture import (
    AllRowCaptureGeometry,
    finalize_all_row_capture,
    initialize_all_row_capture,
    iter_layer_rows,
    load_all_row_capture,
    map_layer_rows,
    materialize_layer_rows,
    write_all_row_chunk,
)


def _rows(begin: int, rows: int) -> dict[str, torch.Tensor]:
    return {
        "input": torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4).bfloat16(),
        "expert_indices": torch.tensor([[1, 3]] * rows, dtype=torch.int32),
        "route_weights": torch.tensor([[0.75, 0.25]] * rows),
        "routed_output": torch.ones(rows, 4, dtype=torch.bfloat16),
        "request_index": torch.arange(begin, begin + rows, dtype=torch.int64) // 3,
        "document_id": torch.full((rows,), 719, dtype=torch.int64),
        "token_offset": torch.arange(begin, begin + rows, dtype=torch.int32),
        "role": torch.zeros(rows, dtype=torch.uint8),
    }


def _initialize(tmp_path, *, expected_rows: int = 5):
    return initialize_all_row_capture(
        tmp_path / "capture.kqrows",
        run_id="unit-test",
        model="example/model",
        revision="f" * 40,
        resident_checkpoint="/models/example",
        corpus_manifest_sha256="a" * 64,
        geometry=AllRowCaptureGeometry(layers=(1, 2), input_size=4, top_k=2),
        chunk_rows=3,
        expected_rows=expected_rows,
        tp_world_size=2,
    )


def test_all_row_capture_closes_contiguous_identical_layers(tmp_path) -> None:
    root = _initialize(tmp_path)
    for layer in (1, 2):
        write_all_row_chunk(root, layer=layer, index=0, row_begin=0, tensors=_rows(0, 3))
        write_all_row_chunk(root, layer=layer, index=1, row_begin=3, tensors=_rows(3, 2))
    manifest = finalize_all_row_capture(
        root, rank_receipts={0: "1" * 64, 1: "2" * 64}
    )
    assert manifest["rows"] == 5
    loaded, geometry, chunks = load_all_row_capture(root)
    assert loaded["canonical_tensor_rank"] == 0
    assert geometry.layers == (1, 2)
    assert [chunk.rows for chunk in chunks[1]] == [3, 2]
    assert sum(int(rows["input"].shape[0]) for rows in iter_layer_rows(root, 2)) == 5


def test_all_row_capture_rejects_layer_row_mismatch(tmp_path) -> None:
    root = _initialize(tmp_path)
    write_all_row_chunk(root, layer=1, index=0, row_begin=0, tensors=_rows(0, 5))
    write_all_row_chunk(root, layer=2, index=0, row_begin=0, tensors=_rows(0, 4))
    with pytest.raises(ValueError, match="identical row counts"):
        finalize_all_row_capture(root, rank_receipts={0: "1" * 64, 1: "2" * 64})


def test_all_row_capture_rejects_chunk_gap(tmp_path) -> None:
    root = _initialize(tmp_path)
    write_all_row_chunk(root, layer=1, index=0, row_begin=1, tensors=_rows(0, 3))
    with pytest.raises(ValueError, match="gap, overlap"):
        finalize_all_row_capture(root, rank_receipts={0: "1" * 64, 1: "2" * 64})


def test_all_row_capture_rejects_tampered_payload(tmp_path) -> None:
    root = _initialize(tmp_path, expected_rows=3)
    for layer in (1, 2):
        chunk = write_all_row_chunk(
            root, layer=layer, index=0, row_begin=0, tensors=_rows(0, 3)
        )
        if layer == 1:
            with chunk.path.open("ab") as stream:
                stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        finalize_all_row_capture(root, rank_receipts={0: "1" * 64, 1: "2" * 64})


def test_all_row_capture_rejects_configuration_drift(tmp_path) -> None:
    root = _initialize(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model"] = "different/model"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match"):
        _initialize(tmp_path)


def test_materialized_layer_indexes_every_routed_occurrence(tmp_path) -> None:
    root = _initialize(tmp_path)
    for layer in (1, 2):
        write_all_row_chunk(root, layer=layer, index=0, row_begin=0, tensors=_rows(0, 3))
        write_all_row_chunk(root, layer=layer, index=1, row_begin=3, tensors=_rows(3, 2))
    finalize_all_row_capture(root, rank_receipts={0: "1" * 64, 1: "2" * 64})

    rows = materialize_layer_rows(
        root,
        1,
        fields=("input", "request_index"),
        num_experts=4,
    )
    assert rows.rows == 5
    assert [index.rows for index in rows.expert_index] == [0, 5, 0, 5]
    batches = list(
        rows.expert_batches(
            3,
            batch_rows=2,
            row_limit=4,
            fields=("input", "request_index"),
        )
    )
    assert [batch["input"].shape[0] for batch in batches] == [2, 2]
    assert torch.cat([batch["row_index"] for batch in batches]).tolist() == [0, 1, 2, 3]
    assert torch.cat([batch["route_slot"] for batch in batches]).tolist() == [1] * 4
    torch.testing.assert_close(
        torch.cat([batch["route_weight"] for batch in batches]),
        torch.full((4,), 0.25),
    )


def test_mapped_layer_matches_materialized_expert_batches(tmp_path) -> None:
    root = _initialize(tmp_path)
    for layer in (1, 2):
        write_all_row_chunk(root, layer=layer, index=0, row_begin=0, tensors=_rows(0, 3))
        write_all_row_chunk(root, layer=layer, index=1, row_begin=3, tensors=_rows(3, 2))
    finalize_all_row_capture(root, rank_receipts={0: "1" * 64, 1: "2" * 64})

    materialized = materialize_layer_rows(
        root,
        1,
        fields=("input",),
        num_experts=4,
    )
    mapped = map_layer_rows(root, 1, num_experts=4)
    assert mapped.rows == materialized.rows
    assert [index.rows for index in mapped.expert_index] == [0, 5, 0, 5]
    expected = list(
        materialized.expert_batches(3, batch_rows=2, fields=("input",))
    )
    actual = list(mapped.expert_batches(3, batch_rows=2, fields=("input",)))
    assert len(actual) == len(expected)
    for expected_batch, actual_batch in zip(expected, actual):
        for name in ("input", "row_index", "route_slot", "route_weight"):
            torch.testing.assert_close(actual_batch[name], expected_batch[name])


def test_capture_chunk_can_load_a_strict_tensor_subset(tmp_path) -> None:
    root = _initialize(tmp_path)
    chunk = write_all_row_chunk(
        root, layer=1, index=0, row_begin=0, tensors=_rows(0, 3)
    )
    tensors = chunk.load(fields=("input", "route_weights"))
    assert set(tensors) == {"input", "route_weights"}
    assert tensors["input"].shape[0] == 3

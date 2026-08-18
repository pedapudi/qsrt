from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from qsrt.kimi_quantized_forward import (
    CompositeCandidateTensorReader,
    QSRTAnchorPayload,
)


def test_composite_candidate_reader_applies_overlays_in_order(tmp_path: Path) -> None:
    base = tmp_path / "base.safetensors"
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    save_file(
        {"a": torch.tensor([1]), "b": torch.tensor([2])},
        base,
    )
    save_file({"a": torch.tensor([3])}, first)
    save_file({"a": torch.tensor([4])}, second)

    with CompositeCandidateTensorReader(base, (first, second)) as reader:
        assert torch.equal(reader.get_tensor("a"), torch.tensor([4]))
        assert torch.equal(reader.get_tensor("b"), torch.tensor([2]))


def test_composite_candidate_reader_rejects_unknown_overlay_tensor(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.safetensors"
    overlay = tmp_path / "overlay.safetensors"
    save_file({"a": torch.tensor([1])}, base)
    save_file({"unknown": torch.tensor([2])}, overlay)

    with pytest.raises(ValueError, match="absent from the sealed layer"):
        with CompositeCandidateTensorReader(base, (overlay,)):
            pass


def test_anchor_payload_resolves_completion_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_pool = tmp_path / "candidates"
    candidate_pool.mkdir()
    base = candidate_pool / "layer.safetensors"
    save_file({"value": torch.tensor([1])}, base)
    overlay_root = tmp_path / "overlay-root"
    completion = overlay_root / "layers" / "layer-001" / "completion.json"
    completion.parent.mkdir(parents=True)
    overlay = completion.parent / "overlay.safetensors"
    save_file({"value": torch.tensor([2])}, overlay)
    completion.write_text(json.dumps({"payload_overlay": str(overlay)}))
    monkeypatch.setattr(
        "qsrt.kimi_quantized_forward.candidate_layer_path",
        lambda _root, _layer: base,
    )

    payload = QSRTAnchorPayload(candidate_pool, overlay_roots=(overlay_root,))
    assert payload.layer_paths(1) == (base, (overlay.resolve(),))

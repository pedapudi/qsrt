from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from qsrt.glm52_engine_kld import (
    ENGINE_KLD_CHUNK_ROWS_ENV,
    ENGINE_KLD_REFERENCE_KEY_ENV,
    ENGINE_KLD_REFERENCE_PATH_ENV,
    ENGINE_KLD_REFERENCE_REPRESENTATION_ENV,
    _engine_kld_configuration,
    _kld_logprobs_tensors,
    engine_kld_from_prompt_logprobs,
    forward_kld_from_model_logprobs,
)
from qsrt.glm52_paired_kld import forward_kld_per_position


def test_engine_kld_matches_existing_per_position_definition(tmp_path: Path) -> None:
    reference = torch.tensor(
        [[1.0, -0.5, 0.25], [-0.75, 0.5, 1.25], [0.0, 0.1, -0.2]],
        dtype=torch.float32,
    )
    model = torch.tensor(
        [[0.8, -0.1, 0.4], [-0.4, 0.2, 1.0], [0.3, -0.2, -0.1]],
        dtype=torch.float32,
    )
    path = tmp_path / "reference.safetensors"
    save_file({"logits": reference}, path)

    actual = forward_kld_from_model_logprobs(
        model.log_softmax(dim=-1),
        reference_path=path,
        reference_key="logits",
        chunk_rows=2,
    )
    expected = forward_kld_per_position(reference, model, chunk_rows=2)

    torch.testing.assert_close(actual.double(), expected, rtol=0.0, atol=2e-7)


def test_engine_kld_rejects_shape_and_key_mismatches(tmp_path: Path) -> None:
    path = tmp_path / "reference.safetensors"
    save_file({"teacher": torch.ones(2, 3)}, path)

    with pytest.raises(KeyError, match="no tensor"):
        forward_kld_from_model_logprobs(
            torch.ones(2, 3).log_softmax(dim=-1),
            reference_path=path,
            reference_key="logits",
            chunk_rows=1,
        )
    with pytest.raises(ValueError, match="shapes differ"):
        forward_kld_from_model_logprobs(
            torch.ones(3, 3).log_softmax(dim=-1),
            reference_path=path,
            reference_key="teacher",
            chunk_rows=1,
        )


def test_engine_kld_accepts_published_singleton_batch_logprobs(
    tmp_path: Path,
) -> None:
    reference_logits = torch.tensor(
        [[1.0, -0.5, 0.25], [-0.75, 0.5, 1.25]], dtype=torch.float32
    )
    reference_logprobs = reference_logits.log_softmax(dim=-1)
    reference_logprobs[0, 1] = float("-inf")
    reference_logprobs[0] -= torch.logsumexp(reference_logprobs[0], dim=-1)
    model_logits = torch.tensor(
        [[0.8, -0.1, 0.4], [-0.4, 0.2, 1.0]], dtype=torch.float32
    )
    model_logprobs = model_logits.log_softmax(dim=-1)
    path = tmp_path / "reference-logprobs.safetensors"
    save_file({"logprobs": reference_logprobs.unsqueeze(0)}, path)

    actual = forward_kld_from_model_logprobs(
        model_logprobs,
        reference_path=path,
        reference_key="logprobs",
        chunk_rows=1,
        reference_representation="logprobs",
    )
    masked_model_logprobs = model_logprobs.clone()
    masked_model_logprobs[0, 1] = float("-inf")
    masked_model_logprobs[0] -= torch.logsumexp(
        masked_model_logprobs[0], dim=-1
    )
    probability = reference_logprobs.exp()
    expected = torch.where(
        torch.isfinite(reference_logprobs),
        probability * (reference_logprobs - masked_model_logprobs),
        torch.zeros_like(reference_logprobs),
    ).sum(dim=-1)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-7)
def test_engine_kld_configuration_requires_one_complete_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "reference.safetensors"
    save_file({"logits": torch.ones(1, 2)}, path)
    monkeypatch.setenv(ENGINE_KLD_REFERENCE_PATH_ENV, str(path))
    with pytest.raises(RuntimeError, match="requires its reference path"):
        _engine_kld_configuration()

    monkeypatch.setenv(ENGINE_KLD_REFERENCE_KEY_ENV, "logits")
    monkeypatch.setenv(ENGINE_KLD_CHUNK_ROWS_ENV, "04")
    with pytest.raises(RuntimeError, match="positive integer"):
        _engine_kld_configuration()

    monkeypatch.setenv(ENGINE_KLD_CHUNK_ROWS_ENV, "4")
    assert _engine_kld_configuration() == (path, "logits", 4, "logits")

    monkeypatch.setenv(ENGINE_KLD_REFERENCE_REPRESENTATION_ENV, "logprobs")
    assert _engine_kld_configuration() == (path, "logits", 4, "logprobs")

    monkeypatch.setenv(ENGINE_KLD_REFERENCE_REPRESENTATION_ENV, "probabilities")
    with pytest.raises(RuntimeError, match="representation"):
        _engine_kld_configuration()


@dataclass
class _FakeLogprobsTensors:
    logprob_token_ids: torch.Tensor
    logprobs: torch.Tensor
    selected_token_ranks: torch.Tensor


def test_kld_values_use_existing_prompt_logprob_container() -> None:
    packed = _kld_logprobs_tensors(
        _FakeLogprobsTensors,
        token_ids=torch.tensor([7, 11], dtype=torch.int64),
        kld_per_position=torch.tensor([0.125, 0.25]),
        output_columns=2,
    )

    assert packed.logprob_token_ids.tolist() == [[7, 7], [11, 11]]
    assert packed.logprobs.tolist() == [[-0.125, -0.125], [-0.25, -0.25]]
    assert packed.selected_token_ranks.tolist() == [0, 0]


def test_engine_kld_decodes_flat_and_mapping_results() -> None:
    flat = SimpleNamespace(
        start_indices=[0, 0, 2],
        end_indices=[0, 2, 4],
        token_ids=[7, 7, 11, 11],
        logprobs=[-0.125, -0.125, -0.25, -0.25],
    )
    mapping = [
        None,
        {7: SimpleNamespace(logprob=-0.125)},
        {11: SimpleNamespace(logprob=-0.25)},
    ]

    expected = torch.tensor([0.125, 0.25], dtype=torch.float64)
    torch.testing.assert_close(
        engine_kld_from_prompt_logprobs(
            flat, positions=2, target_token_ids=[7, 11]
        ),
        expected,
    )
    torch.testing.assert_close(
        engine_kld_from_prompt_logprobs(
            mapping, positions=2, target_token_ids=[7, 11]
        ),
        expected,
    )


def test_engine_kld_rejects_ambiguous_flat_values() -> None:
    flat = SimpleNamespace(
        start_indices=[0, 0],
        end_indices=[0, 2],
        token_ids=[7, 7],
        logprobs=[-0.125, -0.25],
    )

    with pytest.raises(ValueError, match="no unique target-token value"):
        engine_kld_from_prompt_logprobs(
            flat, positions=1, target_token_ids=[7]
        )

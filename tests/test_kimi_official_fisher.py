from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from instanttensor import Backend
from safetensors.torch import save_file

from qsrt.instanttensor_kimi import InstantTensorLoadConfig
from qsrt.kimi_official_fisher import OfficialKimiFisherSuffix, SUFFIX_TENSORS


def _checkpoint(root: Path) -> tuple[Path, dict[str, torch.Tensor]]:
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
    return root, tensors


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_official_suffix_vjp_matches_sampled_reference(tmp_path: Path) -> None:
    checkpoint, tensors = _checkpoint(tmp_path / "checkpoint")
    load_config = InstantTensorLoadConfig(
        buffer_size=1 << 20,
        chunk_size=4096,
        concurrency=1,
        io_depth=3,
        backend=Backend.AIO_BUFFERED,
    )
    suffix = OfficialKimiFisherSuffix(
        checkpoint=checkpoint,
        device="cuda:0",
        hidden_dimension=4,
        vocabulary_size=5,
        residual_block_count=2,
        epsilon=1e-5,
        load_config=load_config,
    )
    hidden = torch.tensor(
        [[[0.5, -0.25, 1.0, 0.75], [1.0, 0.5, -0.5, 0.25]]],
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    residuals = (
        torch.tensor(
            [[0.25, 0.5, -1.0, 0.75], [0.5, -0.25, 0.25, 1.0]],
            dtype=torch.bfloat16,
            device="cuda:0",
        ),
        torch.tensor(
            [[-0.5, 1.0, 0.5, 0.25], [0.75, 0.25, 1.0, -0.5]],
            dtype=torch.bfloat16,
            device="cuda:0",
        ),
    )
    result = suffix.vjp(
        final_boundary=hidden,
        residual_inputs=residuals,
        seed=37,
        lm_head_chunk_tokens=1,
    )
    repeated = suffix.vjp(
        final_boundary=hidden,
        residual_inputs=residuals,
        seed=37,
        lm_head_chunk_tokens=2,
    )
    assert torch.equal(result.first_tokens, repeated.first_tokens)
    assert torch.equal(result.second_tokens, repeated.second_tokens)

    hidden_leaf = hidden.detach().requires_grad_(True)
    residual_leaves = tuple(value.detach().requires_grad_(True) for value in residuals)
    values = torch.cat(
        (
            torch.stack(residual_leaves, dim=1),
            hidden_leaf.reshape(-1, 4).unsqueeze(1),
        ),
        dim=1,
    )
    work = values.float()
    norm = tensors[SUFFIX_TENSORS["residual_norm"]].float().cuda()
    projection = tensors[SUFFIX_TENSORS["residual_projection"]].float().cuda().squeeze(0)
    scores = (
        work
        * torch.rsqrt(work.square().mean(-1, keepdim=True) + 1e-5)
        * norm
        * projection
    ).sum(-1)
    mixed = (torch.softmax(scores, -1).unsqueeze(1) @ work).squeeze(1).to(torch.bfloat16)
    normalized_work = mixed.float() * torch.rsqrt(
        mixed.float().square().mean(-1, keepdim=True) + 1e-5
    )
    normalized = (
        tensors[SUFFIX_TENSORS["final_norm"]].cuda()
        * normalized_work.to(torch.bfloat16)
    ).unsqueeze(0)
    lm_head = tensors[SUFFIX_TENSORS["lm_head"]].float().cuda()
    seed = (
        lm_head.index_select(0, result.first_tokens)
        - lm_head.index_select(0, result.second_tokens)
    ) / math.sqrt(2.0)
    expected = torch.autograd.grad(
        normalized,
        (hidden_leaf, *residual_leaves),
        seed.reshape_as(normalized),
        retain_graph=True,
    )
    assert torch.equal(result.chain_gradient, expected[0])
    assert all(
        torch.equal(observed, wanted)
        for observed, wanted in zip(result.residual_gradients, expected[1:])
    )

    teacher_hidden = (hidden.float() + 0.125).to(torch.bfloat16)
    teacher_residuals = tuple(
        (value.float() - 0.0625).to(torch.bfloat16) for value in residuals
    )
    teacher_normalized = suffix.normalized_hidden(
        final_boundary=teacher_hidden,
        residual_inputs=teacher_residuals,
    )
    channels = suffix.vjp_channels(
        final_boundary=hidden,
        residual_inputs=residuals,
        teacher_normalized=teacher_normalized,
        seed=37,
        lm_head_chunk_tokens=1,
    )
    assert torch.equal(channels.fisher.chain_gradient, result.chain_gradient)
    assert torch.equal(channels.fisher.first_tokens, result.first_tokens)
    anchor_logits = torch.nn.functional.linear(
        normalized.reshape(-1, 4),
        tensors[SUFFIX_TENSORS["lm_head"]].cuda(),
    ).float()
    teacher_logits = torch.nn.functional.linear(
        teacher_normalized.reshape(-1, 4),
        tensors[SUFFIX_TENSORS["lm_head"]].cuda(),
    ).float()
    anchor_log_probabilities = torch.log_softmax(anchor_logits, dim=-1)
    teacher_log_probabilities = torch.log_softmax(teacher_logits, dim=-1)
    kl = torch.sum(
        teacher_log_probabilities.exp()
        * (teacher_log_probabilities - anchor_log_probabilities)
    )
    expected_objective = torch.autograd.grad(
        kl,
        (hidden_leaf, *residual_leaves),
    )
    torch.testing.assert_close(
        channels.objective.chain_gradient,
        expected_objective[0],
        rtol=2e-2,
        atol=2e-3,
    )
    for observed, wanted in zip(
        channels.objective.residual_gradients,
        expected_objective[1:],
        strict=True,
    ):
        torch.testing.assert_close(observed, wanted, rtol=2e-2, atol=2e-3)
    assert channels.objective.token_count == hidden.shape[1]
    assert channels.objective.kl_sum == pytest.approx(float(kl.detach()), rel=1e-6)

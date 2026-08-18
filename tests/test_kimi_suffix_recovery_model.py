from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from qsrt.kimi_suffix_recovery_model import (
    KimiFrozenTeacherLMHead,
    KimiSuffixDecoderStage,
    KimiSuffixStudentOutput,
    enable_shared_experts_and_norms,
)


class _ExpertBlock(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.shared_experts = nn.ModuleDict(
            {
                "gate_proj": nn.Linear(dimension, dimension, bias=False),
                "up_proj": nn.Linear(dimension, dimension, bias=False),
                "down_proj": nn.Linear(dimension, dimension, bias=False),
            }
        )
        self.routed_expert_norm = nn.LayerNorm(
            dimension,
            elementwise_affine=True,
            bias=False,
        )
        self.gate = nn.Linear(dimension, 3, bias=False)


class _Decoder(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.block_sparse_moe = _ExpertBlock(dimension)
        self.input_layernorm = nn.LayerNorm(
            dimension,
            elementwise_affine=True,
            bias=False,
        )
        self.projection = nn.Linear(dimension, dimension, bias=False)


def test_shared_expert_and_norm_allowlist_is_exact() -> None:
    module = _Decoder(5)
    selected = enable_shared_experts_and_norms(module, layer=84)

    expected = {
        "language_model.model.layers.84.block_sparse_moe.shared_experts."
        f"{projection}.weight"
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    expected.update(
        {
            "language_model.model.layers.84.block_sparse_moe."
            "routed_expert_norm.weight",
            "language_model.model.layers.84.input_layernorm.weight",
        }
    )
    assert set(selected) == expected
    for name, parameter in module.named_parameters():
        checkpoint_name = f"language_model.model.layers.84.{name}"
        assert parameter.requires_grad == (checkpoint_name in expected)


class _Adapter:
    def forward_layer(self, module, *, layer, hidden_states, block_residual):
        assert module is decoder
        assert layer == 91
        return hidden_states + 2.0, block_residual + 3.0


decoder = SimpleNamespace()


def test_decoder_stage_adapts_token_major_state() -> None:
    stage = KimiSuffixDecoderStage(
        adapter=_Adapter(),
        layer=91,
        decoder=decoder,
    )
    hidden = torch.randn(7, 4)
    residual = torch.randn(7, 8, 4)
    output, output_residual = stage(hidden, residual)
    torch.testing.assert_close(output, hidden + 2.0)
    torch.testing.assert_close(output_residual, residual + 3.0)


def test_decoder_stage_route_capture_is_one_result_per_forward() -> None:
    class Gate(nn.Module):
        def forward(self, value: torch.Tensor):
            rows = value.shape[-2]
            routes = torch.arange(rows * 2).reshape(rows, 2) % 7
            return routes, torch.ones(rows, 2)

    class RoutedDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.block_sparse_moe = SimpleNamespace(gate=Gate())

    class Adapter:
        def forward_layer(self, module, *, layer, hidden_states, block_residual):
            module.block_sparse_moe.gate(hidden_states.squeeze(0))
            return hidden_states, block_residual

    stage = KimiSuffixDecoderStage(
        adapter=Adapter(),
        layer=84,
        decoder=RoutedDecoder(),
    )
    stage.enable_route_capture()
    hidden = torch.randn(5, 4)
    residual = torch.randn(5, 2, 4)
    stage(hidden, residual)
    expected = torch.arange(10).reshape(5, 2) % 7
    torch.testing.assert_close(stage.take_captured_routes(), expected.to(torch.int16))
    stage.disable_route_capture()


def test_suffix_output_matches_direct_kimi_equations() -> None:
    generator = torch.Generator().manual_seed(19)
    hidden = torch.randn(11, 6, generator=generator, dtype=torch.float64)
    residual = torch.randn(11, 3, 6, generator=generator, dtype=torch.float64)
    final_norm = torch.randn(6, generator=generator, dtype=torch.float64)
    residual_norm = torch.randn(6, generator=generator, dtype=torch.float64)
    residual_projection = torch.randn(1, 6, generator=generator, dtype=torch.float64)
    lm_head = torch.randn(13, 6, generator=generator, dtype=torch.float64)
    epsilon = 1e-6

    module = KimiSuffixStudentOutput(
        lm_head=lm_head,
        final_norm=final_norm,
        residual_norm=residual_norm,
        residual_projection=residual_projection,
        epsilon=epsilon,
    )
    actual = module(hidden, residual)

    values = torch.cat((residual, hidden.unsqueeze(1)), dim=1)
    work = values.float()
    rms = torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + epsilon)
    score_weight = residual_norm.float() * residual_projection.squeeze(0).float()
    scores = (work * rms * score_weight).sum(dim=-1)
    mixed = torch.matmul(torch.softmax(scores, dim=-1).unsqueeze(1), work).squeeze(1)
    final_work = mixed.to(hidden.dtype).float()
    normalized = final_norm * (
        final_work
        * torch.rsqrt(final_work.square().mean(dim=-1, keepdim=True) + epsilon)
    ).to(hidden.dtype)
    expected = F.linear(normalized, lm_head)
    torch.testing.assert_close(actual, expected)

    loss = actual.square().sum()
    loss.backward()
    assert module.final_norm.grad is not None
    assert module.residual_norm.grad is not None
    assert not any(name == "lm_head" for name, _ in module.named_parameters())
    assert not any(name == "residual_projection" for name, _ in module.named_parameters())


def test_frozen_teacher_head_has_no_parameters() -> None:
    weight = torch.randn(9, 4)
    module = KimiFrozenTeacherLMHead(weight)
    value = torch.randn(3, 4)
    torch.testing.assert_close(module(value), F.linear(value, weight))
    assert tuple(module.parameters()) == ()

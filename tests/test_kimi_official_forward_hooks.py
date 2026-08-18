from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from qsrt.kimi_official_forward import (
    OfficialKimiForwardAdapter,
    _architecture_config,
    _pack_grouped_expert_weights,
    install_grouped_low_rank_adapters,
)


class _Expert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(3, 2, bias=False)
        self.w3 = nn.Linear(3, 2, bias=False)
        self.act_fn = nn.Identity()
        self.w2 = nn.Linear(4, 3, bias=False)
        self.forward_count = 0

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        self.forward_count += 1
        gate_up = torch.cat((self.w1(rows), self.w3(rows)), dim=-1)
        return self.w2(self.act_fn(gate_up))


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList([_Expert()])

    @torch.no_grad()
    def moe_infer(
        self,
        rows: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        del topk_ids
        return self.experts[0](rows) * topk_weight[:, :1]


def test_architecture_config_excludes_storage_metadata(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"model_type":"kimi_k3","quantization_config":{"format":"top"},'
        '"text_config":{"hidden_size":8,"quantization_config":{"format":"nested"}}}'
    )
    assert _architecture_config(path) == {
        "model_type": "kimi_k3",
        "text_config": {"hidden_size": 8},
    }


def test_official_hooks_dispatch_multiple_reverse_channels_from_one_forward() -> None:
    block = _Block()
    module = SimpleNamespace(block_sparse_moe=block)
    routed: dict[str, list[torch.Tensor]] = {"fisher": [], "objective": []}
    upstream: dict[str, list[tuple[int, torch.Tensor, torch.Tensor]]] = {
        "fisher": [],
        "objective": [],
    }
    downstream: dict[str, list[tuple[int, torch.Tensor, torch.Tensor]]] = {
        "fisher": [],
        "objective": [],
    }
    channel = "fisher"

    def routed_callback(gradient: torch.Tensor) -> None:
        routed[channel].append(gradient.detach().clone())

    def upstream_callback(
        expert: int,
        rows: torch.Tensor,
        gradient: torch.Tensor,
    ) -> None:
        upstream[channel].append(
            (expert, rows.detach().clone(), gradient.detach().clone())
        )

    def downstream_callback(
        expert: int,
        rows: torch.Tensor,
        gradient: torch.Tensor,
    ) -> None:
        downstream[channel].append(
            (expert, rows.detach().clone(), gradient.detach().clone())
        )

    assert OfficialKimiForwardAdapter.enable_routed_output_gradients(
        module, routed_callback
    )
    assert OfficialKimiForwardAdapter.enable_expert_preactivation_gradients(
        module, upstream_callback
    )
    assert OfficialKimiForwardAdapter.enable_expert_output_gradients(
        module, downstream_callback
    )
    rows = torch.randn((5, 3), generator=torch.Generator().manual_seed(19))
    rows.requires_grad_(True)
    output = block.moe_infer(
        rows,
        torch.zeros((5, 1), dtype=torch.long),
        torch.ones((5, 1)),
    )
    first_seed = torch.ones_like(output)
    torch.autograd.grad(output, rows, first_seed, retain_graph=True)
    channel = "objective"
    second_seed = torch.arange(output.numel()).reshape_as(output).float() / 7
    torch.autograd.grad(output, rows, second_seed)

    assert block.experts[0].forward_count == 1
    assert all(len(values) == 1 for values in routed.values())
    assert all(len(values) == 1 for values in upstream.values())
    assert all(len(values) == 1 for values in downstream.values())
    assert upstream["fisher"][0][0] == 0
    torch.testing.assert_close(upstream["fisher"][0][1], rows.detach())
    assert not torch.equal(
        upstream["fisher"][0][2],
        upstream["objective"][0][2],
    )
    assert downstream["fisher"][0][0] == 0
    assert not torch.equal(
        downstream["fisher"][0][2],
        downstream["objective"][0][2],
    )


class _PairActivation(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        gate, up = values.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up


class _GroupedExpert(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dimension, dimension, bias=False, dtype=torch.bfloat16)
        self.w3 = nn.Linear(dimension, dimension, bias=False, dtype=torch.bfloat16)
        self.act_fn = _PairActivation()
        self.w2 = nn.Linear(dimension, dimension, bias=False, dtype=torch.bfloat16)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        gate_up = torch.cat((self.w1(rows), self.w3(rows)), dim=-1)
        return self.w2(self.act_fn(gate_up))


class _GroupedBlock(nn.Module):
    def __init__(self, experts: int, dimension: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [_GroupedExpert(dimension) for _ in range(experts)]
        )

    def moe_infer(
        self,
        rows: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        outputs = torch.empty(
            (topk_ids.numel(), rows.shape[1]),
            dtype=rows.dtype,
            device=rows.device,
        )
        flat_ids = topk_ids.reshape(-1)
        flat_rows = rows.repeat_interleave(topk_ids.shape[1], dim=0)
        for expert_id, expert in enumerate(self.experts):
            selected = flat_ids == expert_id
            if bool(torch.any(selected)):
                outputs[selected] = expert(flat_rows[selected])
        return (
            outputs.reshape(*topk_ids.shape, -1)
            .mul(topk_weight.unsqueeze(-1))
            .sum(dim=1)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_grouped_expert_dispatch_preserves_output_and_input_vjp() -> None:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    generator = torch.Generator(device="cpu").manual_seed(112)
    block = _GroupedBlock(experts=4, dimension=16).to(device)
    rows = torch.randn((8, 16), generator=generator, dtype=torch.bfloat16).to(device)
    topk_ids = torch.tensor(
        [[0, 2], [1, 3], [2, 0], [3, 1], [0, 1], [2, 3], [1, 2], [3, 0]],
        dtype=torch.long,
        device=device,
    )
    topk_weight = torch.randn(
        (8, 2), generator=generator, dtype=torch.bfloat16
    ).to(device)
    seed = torch.randn((8, 16), generator=generator, dtype=torch.bfloat16).to(device)

    baseline_rows = rows.detach().clone().requires_grad_(True)
    baseline = block.moe_infer(baseline_rows, topk_ids, topk_weight)
    baseline_vjp = torch.autograd.grad(baseline, baseline_rows, seed)[0]

    module = SimpleNamespace(block_sparse_moe=block)
    assert _pack_grouped_expert_weights(module) >= 0.0
    captures: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    assert OfficialKimiForwardAdapter.enable_expert_preactivation_gradients(
        module,
        lambda *values: captures.append(tuple(value.detach() for value in values)),
    )
    grouped_rows = rows.detach().clone().requires_grad_(True)
    grouped = block.moe_infer(grouped_rows, topk_ids, topk_weight)
    grouped_vjp = torch.autograd.grad(grouped, grouped_rows, seed)[0]

    assert torch.equal(grouped, baseline)
    assert torch.equal(grouped_vjp, baseline_vjp)
    assert len(captures) == 1
    sorted_experts, offsets, sorted_rows, gate_up_gradient = captures[0]
    assert sorted_experts.shape == (16,)
    assert offsets.shape == (4,)
    assert sorted_rows.shape == (16, 16)
    assert gate_up_gradient.shape == (16, 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_grouped_expert_dispatch_applies_low_rank_projection_corrections() -> None:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    generator = torch.Generator(device="cpu").manual_seed(209)
    block = _GroupedBlock(experts=4, dimension=16).to(device)
    module = SimpleNamespace(block_sparse_moe=block)
    assert _pack_grouped_expert_weights(module) >= 0.0
    banks = block._qsrt_expert_weight_banks
    adapters: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for matrix, bank in banks.items():
        rank = 3
        a = (
            torch.randn(
                (bank.shape[0], bank.shape[2], rank),
                generator=generator,
                dtype=torch.bfloat16,
            ).to(device)
            / 32
        )
        b = (
            torch.randn(
                (bank.shape[0], bank.shape[1], rank),
                generator=generator,
                dtype=torch.bfloat16,
            ).to(device)
            / 32
        )
        adapters[matrix] = (a, b)
    install_grouped_low_rank_adapters(module, adapters)

    rows = torch.randn(
        (8, 16), generator=generator, dtype=torch.bfloat16
    ).to(device)
    topk_ids = torch.tensor(
        [[0, 2], [1, 3], [2, 0], [3, 1], [0, 1], [2, 3], [1, 2], [3, 0]],
        dtype=torch.long,
        device=device,
    )
    topk_weight = torch.randn(
        (8, 2), generator=generator, dtype=torch.bfloat16
    ).to(device)

    def reference(input_rows: torch.Tensor) -> torch.Tensor:
        routed_rows = input_rows.repeat_interleave(topk_ids.shape[1], dim=0)
        flat_ids = topk_ids.reshape(-1)
        routed_outputs = torch.empty_like(routed_rows)
        for expert_id in range(banks["w1"].shape[0]):
            selected = flat_ids == expert_id
            expert_rows = routed_rows[selected]
            gate = torch.nn.functional.linear(expert_rows, banks["w1"][expert_id])
            gate = gate + expert_rows @ adapters["w1"][0][expert_id] @ adapters["w1"][1][expert_id].T
            up = torch.nn.functional.linear(expert_rows, banks["w3"][expert_id])
            up = up + expert_rows @ adapters["w3"][0][expert_id] @ adapters["w3"][1][expert_id].T
            middle = block.experts[0].act_fn(torch.cat((gate, up), dim=-1))
            output = torch.nn.functional.linear(middle, banks["w2"][expert_id])
            output = output + middle @ adapters["w2"][0][expert_id] @ adapters["w2"][1][expert_id].T
            routed_outputs[selected] = output
        return (
            routed_outputs.reshape(*topk_ids.shape, -1)
            .mul(topk_weight.unsqueeze(-1))
            .sum(dim=1)
        )

    seed = torch.randn(
        (8, 16), generator=generator, dtype=torch.bfloat16
    ).to(device)
    expected_rows = rows.detach().clone().requires_grad_(True)
    expected = reference(expected_rows)
    expected_vjp = torch.autograd.grad(expected, expected_rows, seed)[0]
    actual_rows = rows.detach().clone().requires_grad_(True)
    actual = block.moe_infer(actual_rows, topk_ids, topk_weight)
    actual_vjp = torch.autograd.grad(actual, actual_rows, seed)[0]

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_vjp, expected_vjp, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_grouped_expert_dispatch_zero_adapters_are_bit_exact() -> None:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    generator = torch.Generator(device="cpu").manual_seed(311)
    block = _GroupedBlock(experts=4, dimension=16).to(device)
    module = SimpleNamespace(block_sparse_moe=block)
    assert _pack_grouped_expert_weights(module) >= 0.0
    rows = torch.randn(
        (8, 16), generator=generator, dtype=torch.bfloat16
    ).to(device)
    topk_ids = torch.tensor(
        [[0, 2], [1, 3], [2, 0], [3, 1], [0, 1], [2, 3], [1, 2], [3, 0]],
        dtype=torch.long,
        device=device,
    )
    topk_weight = torch.randn(
        (8, 2), generator=generator, dtype=torch.bfloat16
    ).to(device)
    seed = torch.randn(
        (8, 16), generator=generator, dtype=torch.bfloat16
    ).to(device)
    baseline_rows = rows.detach().clone().requires_grad_(True)
    baseline = block.moe_infer(baseline_rows, topk_ids, topk_weight)
    baseline_vjp = torch.autograd.grad(baseline, baseline_rows, seed)[0]

    zero_adapters = {
        matrix: (
            torch.zeros(
                (bank.shape[0], bank.shape[2], 2),
                dtype=bank.dtype,
                device=device,
            ),
            torch.zeros(
                (bank.shape[0], bank.shape[1], 2),
                dtype=bank.dtype,
                device=device,
            ),
        )
        for matrix, bank in block._qsrt_expert_weight_banks.items()
    }
    install_grouped_low_rank_adapters(module, zero_adapters)
    adapted_rows = rows.detach().clone().requires_grad_(True)
    adapted = block.moe_infer(adapted_rows, topk_ids, topk_weight)
    adapted_vjp = torch.autograd.grad(adapted, adapted_rows, seed)[0]

    assert torch.equal(adapted, baseline)
    assert torch.equal(adapted_vjp, baseline_vjp)

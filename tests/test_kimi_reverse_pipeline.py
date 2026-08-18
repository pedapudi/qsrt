from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from qsrt.kimi_boundary_slabs import DocumentIndex, KimiBoundarySlabArchive
from qsrt.kimi_cotangent_slabs import KimiCotangentSlabWorkspace
from qsrt.kimi_output_factors import (
    KimiOutputFactorArchive,
    document_factor_split,
)
from qsrt.kimi_reverse_pipeline import (
    EmpiricalOutputFactor,
    KimiReversePipeline,
    replay_decoder_segment,
    replay_decoder_segment_channels,
)


class _SyntheticReverseAdapter:
    def __init__(self, residual_block_size: int):
        self.residual_block_size = residual_block_size

    def forward_layer(
        self,
        module: torch.Tensor,
        *,
        layer: int,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer % self.residual_block_size == 0:
            block_residual = torch.cat(
                [
                    block_residual,
                    hidden_states.reshape(-1, hidden_states.shape[-1]).unsqueeze(1),
                ],
                dim=1,
            )
        residual_sum = block_residual.sum(dim=1).unsqueeze(0)
        return hidden_states * module + residual_sum * 0.125, block_residual


def _reference_segment(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    scales: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = hidden.detach().requires_grad_(True)
    residual = residual.detach().requires_grad_(True)
    block_residual = residual.unsqueeze(1)
    output = hidden
    for stage, scale in enumerate(scales):
        layer = 2 + stage
        if layer % 2 == 0:
            block_residual = torch.cat(
                [block_residual, output.reshape(-1, 4).unsqueeze(1)], dim=1
            )
        output = output * scale + block_residual.sum(dim=1).unsqueeze(0) * 0.125
    seed = torch.arange(output.numel(), device=output.device).reshape_as(output) / 17
    gradients = torch.autograd.grad(output, (hidden, residual), seed)
    return output.detach(), gradients[0], gradients[1]


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="segment replay requires two peer-accessible CUDA devices",
)
def test_segment_replay_matches_single_device_vjp() -> None:
    hidden = torch.arange(24, dtype=torch.float32, device="cuda:0").reshape(1, 6, 4) / 13
    residual = torch.arange(24, dtype=torch.float32, device="cuda:0").reshape(6, 4) / 19
    expected_output, expected_hidden, expected_residual = _reference_segment(
        hidden,
        residual,
        (1.25, 0.75),
    )
    seed = torch.arange(24, dtype=torch.float32, device="cuda:1").reshape(1, 6, 4) / 17
    result = replay_decoder_segment(
        adapter=_SyntheticReverseAdapter(residual_block_size=2),
        modules=(
            torch.tensor(1.25, device="cuda:0"),
            torch.tensor(0.75, device="cuda:1"),
        ),
        first_layer=2,
        input_hidden=hidden,
        residual_inputs=(residual,),
        output_gradient=seed,
        devices=("cuda:0", "cuda:1"),
        residual_block_size=2,
    )
    assert torch.allclose(result.output.to("cuda:0"), expected_output)
    assert torch.allclose(result.input_gradient, expected_hidden)
    assert len(result.residual_gradients) == 1
    assert torch.allclose(result.residual_gradients[0], expected_residual)


class _DynamicChannelAdapter(_SyntheticReverseAdapter):
    def __init__(self, residual_block_size: int):
        super().__init__(residual_block_size)
        self.channel: str | None = None
        self.forward_count = 0
        self.hook_gradients: dict[str, list[torch.Tensor]] = {}

    def forward_layer(
        self,
        module: torch.Tensor,
        *,
        layer: int,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.forward_count += 1
        output, residual = super().forward_layer(
            module,
            layer=layer,
            hidden_states=hidden_states,
            block_residual=block_residual,
        )

        def dispatch(gradient: torch.Tensor) -> None:
            if self.channel is not None:
                self.hook_gradients.setdefault(self.channel, []).append(
                    gradient.detach().cpu()
                )

        output.register_hook(dispatch)
        return output, residual


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="multi-channel replay requires two peer-accessible CUDA devices",
)
def test_segment_replay_reuses_forward_graph_for_multiple_vjps() -> None:
    hidden = torch.arange(24, dtype=torch.float32, device="cuda:0").reshape(1, 6, 4) / 13
    residual = torch.arange(24, dtype=torch.float32, device="cuda:0").reshape(6, 4) / 19
    adapter = _DynamicChannelAdapter(residual_block_size=2)
    first_seed = torch.ones((1, 6, 4), device="cuda:1") * 0.25
    second_seed = torch.arange(24, dtype=torch.float32, device="cuda:1").reshape(1, 6, 4) / 11
    result = replay_decoder_segment_channels(
        adapter=adapter,
        modules=(
            torch.tensor(1.25, device="cuda:0"),
            torch.tensor(0.75, device="cuda:1"),
        ),
        first_layer=2,
        input_hidden=hidden,
        residual_inputs=(residual,),
        output_gradients={"fisher": first_seed, "objective": second_seed},
        devices=("cuda:0", "cuda:1"),
        residual_block_size=2,
        select_channel=lambda channel: setattr(adapter, "channel", channel),
    )

    assert adapter.forward_count == 2
    assert set(result.channels) == {"fisher", "objective"}
    assert {name: len(values) for name, values in adapter.hook_gradients.items()} == {
        "fisher": 2,
        "objective": 2,
    }
    assert not torch.equal(
        result.channels["fisher"].input_gradient,
        result.channels["objective"].input_gradient,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_output_factor_accumulates_fp32_outer_products() -> None:
    accumulator = EmpiricalOutputFactor(3, device="cuda:0")
    first = torch.tensor([[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]], device="cuda:0")
    second = torch.tensor([[4.0, 1.0, 0.25]], dtype=torch.bfloat16, device="cuda:0")
    accumulator.add(first)
    accumulator.add(second)
    rows = torch.cat([first.float(), second.float()])
    assert accumulator.rows == 3
    assert torch.allclose(accumulator.mean(), rows.T @ rows / 3)


@dataclass
class _SyntheticLoadedModule:
    layer: int
    scale: torch.Tensor
    callback: object | None = None


class _SyntheticPipelineAdapter:
    def __init__(self, scales: tuple[float, ...], residual_block_size: int):
        self.scales = scales
        self.residual_block_size = residual_block_size

    def load_layer(self, layer: int, device: torch.device):
        return (
            _SyntheticLoadedModule(
                layer=layer,
                scale=torch.tensor(self.scales[layer], device=device),
            ),
            {"layer": layer},
        )

    def forward_layer(
        self,
        module: _SyntheticLoadedModule,
        *,
        layer: int,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer % self.residual_block_size == 0:
            block_residual = torch.cat(
                [
                    block_residual,
                    hidden_states.reshape(-1, hidden_states.shape[-1]).unsqueeze(1),
                ],
                dim=1,
            )
        routed = hidden_states * module.scale
        if routed.requires_grad and module.callback is not None:
            routed.register_hook(module.callback)
        residual = block_residual.sum(dim=1).unsqueeze(0)
        return routed + residual * 0.125, block_residual

    @staticmethod
    def release_layer(module: _SyntheticLoadedModule) -> None:
        module.callback = None

    @staticmethod
    def enable_routed_output_gradients(
        module: _SyntheticLoadedModule,
        callback,
    ) -> bool:
        module.callback = callback
        return True


def _write_boundary(
    archive: KimiBoundarySlabArchive,
    boundary: int,
    value: torch.Tensor,
) -> None:
    archive.prepare_boundary(boundary)
    with archive.extent_writer(
        boundary,
        writer_id="all",
        first_token=0,
        end_token=archive.token_count,
        direct=False,
    ) as writer:
        writer.append(value)
        writer.finish()
    archive.seal_boundary(boundary)


def _write_cotangents(update, values: dict[str, torch.Tensor]) -> None:
    for role, value in values.items():
        writer = update.writer(role, direct=False)
        try:
            writer.append(value)
            update.record(writer.finish())
        finally:
            writer.close()
    update.commit()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="reverse pipeline test requires two CUDA devices",
)
def test_reverse_pipeline_commits_exact_vjp_and_split_factors(
    tmp_path: Path,
) -> None:
    identifiers: list[str] = []
    for index in range(100):
        identifier = f"document-{index}"
        if not identifiers or document_factor_split(identifier) != document_factor_split(identifiers[0]):
            identifiers.append(identifier)
        if len(identifiers) == 2:
            break
    assert len(identifiers) == 2
    documents = DocumentIndex(
        input_ids=torch.arange(4, dtype=torch.int32),
        offsets=torch.tensor([0, 2, 4], dtype=torch.int64),
        identifiers=tuple(identifiers),
    )
    archive = KimiBoundarySlabArchive.create(
        tmp_path / "boundaries",
        documents=documents,
        num_layers=2,
        hidden_dimension=3,
        attn_res_block_size=2,
    )
    boundary0 = (
        torch.arange(12, dtype=torch.float32).reshape(4, 3) / 13
    ).to(torch.bfloat16)
    boundary1 = (boundary0 * 1.5 + boundary0 * 0.125).to(torch.bfloat16)
    boundary2 = (boundary1 * 0.75 + boundary0 * 0.125).to(torch.bfloat16)
    for boundary, value in enumerate((boundary0, boundary1, boundary2)):
        _write_boundary(archive, boundary, value)
    archive.seal()
    archive = KimiBoundarySlabArchive(archive.root, require_complete=True)

    workspace = KimiCotangentSlabWorkspace.create(
        tmp_path / "cotangents",
        boundary_archive=archive,
    )
    chain = (
        torch.arange(12, dtype=torch.float32).reshape(4, 3) / 17 + 0.25
    ).to(torch.bfloat16)
    direct_residual = torch.full((4, 3), 0.125, dtype=torch.bfloat16)
    _write_cotangents(
        workspace.begin_suffix(),
        {"chain": chain, "residual-000": direct_residual},
    )
    factors = KimiOutputFactorArchive.create(
        tmp_path / "factors",
        num_layers=2,
        dimension=3,
        expected_layers=(0, 1),
    )
    result = KimiReversePipeline(
        adapter=_SyntheticPipelineAdapter((1.5, 0.75), 2),
        boundary_archive=archive,
        cotangent_workspace=workspace,
        output_factors=factors,
        devices=("cuda:0", "cuda:1"),
        slab_buffer_tokens=2,
        direct_io=False,
    ).run()
    assert len(result.segments) == 1
    reopened = KimiCotangentSlabWorkspace(workspace.root)
    assert reopened.manifest["chain_boundary"] == 0

    expected_parts: list[torch.Tensor] = []
    for document in range(2):
        first, end = documents.document_extent(document)
        hidden = boundary0[first:end].to("cuda:0").unsqueeze(0).requires_grad_(True)
        residual = hidden.reshape(-1, 3).unsqueeze(1)
        first_output = hidden * 1.5 + residual.sum(dim=1).unsqueeze(0) * 0.125
        second_input = first_output.to("cuda:1")
        second_residual = residual.to("cuda:1")
        output = second_input * 0.75 + second_residual.sum(dim=1).unsqueeze(0) * 0.125
        gradient = torch.autograd.grad(
            output,
            hidden,
            chain[first:end].to("cuda:1").unsqueeze(0),
        )[0]
        expected_parts.append(
            (
                gradient.reshape(-1, 3).float()
                + direct_residual[first:end].to(gradient.device).float()
            )
            .to(torch.bfloat16)
            .cpu()
        )
    assert torch.allclose(
        reopened.read_chain(0, 4, direct=False),
        torch.cat(expected_parts),
        atol=0.004,
        rtol=0.01,
    )

    factor_archive = KimiOutputFactorArchive(factors.root)
    assert factor_archive.manifest["complete"] is True
    segment = factor_archive.manifest["segments"][0]
    stored = {
        int(record["layer"]): load_file(
            factor_archive.root / segment["directory"] / record["file"]
        )["output_hessian"]
        for record in segment["layers"]
    }
    expected_layer1 = chain.float().T @ chain.float() / 4
    expected_layer0_rows = (chain * 0.75).to(torch.bfloat16).float()
    expected_layer0 = expected_layer0_rows.T @ expected_layer0_rows / 4
    assert torch.allclose(stored[1], expected_layer1)
    assert torch.allclose(stored[0], expected_layer0)

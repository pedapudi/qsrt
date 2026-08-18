from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from qsrt.kimi_boundary_slabs import DocumentIndex, KimiBoundarySlabArchive
from qsrt.kimi_cotangent_slabs import KimiCotangentSlabWorkspace
from qsrt.kimi_upstream_factors import KimiUpstreamFactorArchive
from qsrt.kimi_upstream_pipelined_reverse import KimiPipelinedUpstreamReverse
from qsrt.kimi_upstream_reverse_pipeline import KimiUpstreamReversePipeline
from qsrt.qsrt_coupled import CoupledHadamardExecution, CoupledHadamardSpec


@dataclass
class _SyntheticExpertLayer:
    layer: int
    scale: torch.Tensor
    routed: bool = True
    routed_callback: object | None = None
    preactivation_callback: object | None = None


class _SyntheticUpstreamAdapter:
    def __init__(
        self,
        scales: tuple[float, ...],
        residual_block_size: int,
        *,
        routed_layers: tuple[int, ...] | None = None,
    ):
        self.scales = scales
        self.residual_block_size = residual_block_size
        self.routed_layers = routed_layers

    def load_layer(self, layer: int, device: torch.device):
        return (
            _SyntheticExpertLayer(
                layer=layer,
                scale=torch.tensor(self.scales[layer], device=device),
                routed=(
                    self.routed_layers is None or layer in self.routed_layers
                ),
            ),
            {"layer": layer},
        )

    def forward_layer(
        self,
        module: _SyntheticExpertLayer,
        *,
        layer: int,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer % self.residual_block_size == 0:
            block_residual = torch.cat(
                (
                    block_residual,
                    hidden_states.reshape(-1, hidden_states.shape[-1]).unsqueeze(1),
                ),
                dim=1,
            )
        input_rows = hidden_states.reshape(-1, hidden_states.shape[-1])
        gate_up = input_rows * module.scale
        if gate_up.requires_grad and module.preactivation_callback is not None:

            def preactivation_gradient(gradient: torch.Tensor) -> None:
                callback = module.preactivation_callback
                if callback is not None:
                    callback(0, input_rows, gradient)

            gate_up.register_hook(preactivation_gradient)
        routed = gate_up.reshape_as(hidden_states)
        if routed.requires_grad and module.routed_callback is not None:
            routed.register_hook(module.routed_callback)
        residual = block_residual.sum(dim=1).unsqueeze(0)
        return routed + residual * 0.125, block_residual

    @staticmethod
    def release_layer(module: _SyntheticExpertLayer) -> None:
        module.routed_callback = None
        module.preactivation_callback = None

    @staticmethod
    def enable_routed_output_gradients(
        module: _SyntheticExpertLayer,
        callback,
    ) -> bool:
        if not module.routed:
            return False
        module.routed_callback = callback
        return True

    @staticmethod
    def enable_expert_preactivation_gradients(
        module: _SyntheticExpertLayer,
        callback,
    ) -> bool:
        module.preactivation_callback = callback
        return True

    @staticmethod
    def enable_expert_output_gradients(
        module: _SyntheticExpertLayer,
        callback,
    ) -> bool:
        del module, callback
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
    reason="upstream reverse pipeline test requires two CUDA devices",
)
@pytest.mark.parametrize(
    "pipeline_class",
    (KimiUpstreamReversePipeline, KimiPipelinedUpstreamReverse),
)
def test_upstream_reverse_pipeline_commits_coupled_factors_and_vjp(
    tmp_path: Path,
    pipeline_class,
) -> None:
    documents = DocumentIndex(
        input_ids=torch.arange(4, dtype=torch.int32),
        offsets=torch.tensor([0, 2, 4], dtype=torch.int64),
        identifiers=("first", "second"),
    )
    boundaries = KimiBoundarySlabArchive.create(
        tmp_path / "boundaries",
        documents=documents,
        num_layers=2,
        hidden_dimension=4,
        attn_res_block_size=2,
    )
    boundary0 = (
        torch.arange(16, dtype=torch.float32).reshape(4, 4) / 13
    ).to(torch.bfloat16)
    for boundary in range(3):
        _write_boundary(boundaries, boundary, boundary0 + boundary)
    boundaries.seal()
    boundaries = KimiBoundarySlabArchive(boundaries.root, require_complete=True)

    workspace = KimiCotangentSlabWorkspace.create(
        tmp_path / "cotangents",
        boundary_archive=boundaries,
    )
    chain = (
        torch.arange(16, dtype=torch.float32).reshape(4, 4) / 17 + 0.25
    ).to(torch.bfloat16)
    _write_cotangents(
        workspace.begin_suffix(),
        {
            "chain": chain,
            "residual-000": torch.zeros_like(chain),
        },
    )
    factors = KimiUpstreamFactorArchive.create(
        tmp_path / "factors",
        num_layers=2,
        num_experts=1,
        hidden_dimension=4,
        intermediate_dimension=2,
        block_size=2,
        gradient_rank=2,
        expected_layers=(0, 1),
    )
    spec = CoupledHadamardSpec(
        residual_block_size=4,
        preactivation_block_size=2,
        postactivation_block_size=2,
    )
    result = pipeline_class(
        adapter=_SyntheticUpstreamAdapter((1.5, 0.75), 2),
        boundary_archive=boundaries,
        cotangent_workspace=workspace,
        upstream_factors=factors,
        intermediate_draws={0: (0,), 1: (1,)},
        devices=("cuda:0", "cuda:1"),
        slab_buffer_tokens=2,
        direct_io=False,
        coupled_spec=spec,
    ).run()

    assert len(result.segments) == 1
    assert workspace.manifest["chain_boundary"] == 0
    assert factors.manifest["complete"] is True
    assert [record.supported_experts for record in result.segments[0].layers] == [1, 1]
    assert [record.gradient_rows for record in result.segments[0].layers] == [4, 4]

    for layer, source_gradient in ((0, chain * 0.75), (1, chain)):
        execution = CoupledHadamardExecution(
            hidden=4,
            intermediate=2,
            spec=CoupledHadamardSpec(
                residual_block_size=4,
                preactivation_block_size=2,
                postactivation_block_size=2,
                intermediate_draw=layer,
            ),
        )
        transformed = execution.transform_preactivation_gradients(
            source_gradient.to("cuda:0")
        ).cpu()
        blocked = transformed.reshape(4, 2, 2)
        expected = torch.einsum("nbi,nbj->bij", blocked, blocked) / 4
        observed, rows, draw = factors.load_expert_output_blocks(
            layer,
            0,
            matrix="w1",
        )
        assert rows == 4
        assert draw == layer
        assert torch.allclose(observed, expected[:1], atol=5e-4, rtol=5e-4)
        observed, rows, draw = factors.load_expert_output_blocks(
            layer,
            0,
            matrix="w3",
        )
        assert rows == 4
        assert draw == layer
        assert torch.allclose(observed, expected[1:], atol=5e-4, rtol=5e-4)

    expected_chain = (
        chain.float() * (0.75 * (1.5 + 0.125) + 0.125)
    ).to(torch.bfloat16)
    assert torch.allclose(
        workspace.read_chain(0, 4, direct=False),
        expected_chain,
        atol=0.004,
        rtol=0.01,
    )


@pytest.mark.skipif(
    torch.cuda.device_count() < 1,
    reason="upstream reverse pipeline test requires CUDA",
)
def test_pipelined_reverse_separates_fisher_and_objective_graphs(
    tmp_path: Path,
) -> None:
    documents = DocumentIndex(
        input_ids=torch.arange(4, dtype=torch.int32),
        offsets=torch.tensor([0, 4], dtype=torch.int64),
        identifiers=("document",),
    )
    boundaries = KimiBoundarySlabArchive.create(
        tmp_path / "boundaries",
        documents=documents,
        num_layers=1,
        hidden_dimension=4,
        attn_res_block_size=1,
    )
    boundary = (
        torch.arange(16, dtype=torch.float32).reshape(4, 4) / 13
    ).to(torch.bfloat16)
    for index in range(2):
        _write_boundary(boundaries, index, boundary + index)
    boundaries.seal()
    boundaries = KimiBoundarySlabArchive(boundaries.root, require_complete=True)

    fisher = KimiCotangentSlabWorkspace.create(
        tmp_path / "fisher",
        boundary_archive=boundaries,
    )
    objective = KimiCotangentSlabWorkspace.create(
        tmp_path / "objective",
        boundary_archive=boundaries,
    )
    fisher_chain = torch.full((4, 4), 0.25, dtype=torch.bfloat16)
    objective_chain = (
        torch.arange(16, dtype=torch.float32).reshape(4, 4) / 23 + 0.1
    ).to(torch.bfloat16)
    for workspace, chain in (
        (fisher, fisher_chain),
        (objective, objective_chain),
    ):
        _write_cotangents(
            workspace.begin_suffix(),
            {
                "chain": chain,
                "residual-000": torch.zeros_like(chain),
            },
        )

    factors = KimiUpstreamFactorArchive.create(
        tmp_path / "factors",
        num_layers=1,
        num_experts=1,
        hidden_dimension=4,
        intermediate_dimension=2,
        block_size=2,
        gradient_rank=2,
        expected_layers=(0,),
    )
    adapter = _SyntheticUpstreamAdapter((1.5,), 1)
    KimiPipelinedUpstreamReverse(
        adapter=adapter,
        boundary_archive=boundaries,
        cotangent_workspace=fisher,
        objective_workspace=objective,
        upstream_factors=factors,
        intermediate_draws={0: (0,)},
        devices=("cuda:0",),
        slab_buffer_tokens=2,
        direct_io=False,
        coupled_spec=CoupledHadamardSpec(
            residual_block_size=4,
            preactivation_block_size=2,
            postactivation_block_size=2,
        ),
    ).run()

    assert fisher.manifest["chain_boundary"] == 0
    assert objective.manifest["chain_boundary"] == 0
    assert factors.manifest["complete"] is True
    gradient = factors.load_expert_gradient(0, 0, matrix="w1")
    assert gradient.shape == (2, 4)
    assert bool(torch.all(torch.isfinite(gradient)))
    assert float(torch.linalg.vector_norm(gradient)) > 0.0
    expected_objective = (objective_chain.float() * 1.625).to(torch.bfloat16)
    assert torch.allclose(
        objective.read_chain(0, 4, direct=False),
        expected_objective,
        atol=0.004,
        rtol=0.01,
    )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="upstream reverse pipeline test requires two CUDA devices",
)
def test_pipelined_reverse_allows_a_non_routed_dense_layer(
    tmp_path: Path,
) -> None:
    documents = DocumentIndex(
        input_ids=torch.arange(4, dtype=torch.int32),
        offsets=torch.tensor([0, 4], dtype=torch.int64),
        identifiers=("document",),
    )
    boundaries = KimiBoundarySlabArchive.create(
        tmp_path / "boundaries",
        documents=documents,
        num_layers=2,
        hidden_dimension=4,
        attn_res_block_size=2,
    )
    boundary = (
        torch.arange(16, dtype=torch.float32).reshape(4, 4) / 13
    ).to(torch.bfloat16)
    for index in range(3):
        _write_boundary(boundaries, index, boundary + index)
    boundaries.seal()
    boundaries = KimiBoundarySlabArchive(boundaries.root, require_complete=True)

    fisher = KimiCotangentSlabWorkspace.create(
        tmp_path / "fisher",
        boundary_archive=boundaries,
    )
    objective = KimiCotangentSlabWorkspace.create(
        tmp_path / "objective",
        boundary_archive=boundaries,
    )
    for workspace, value in ((fisher, 0.25), (objective, 0.5)):
        chain = torch.full((4, 4), value, dtype=torch.bfloat16)
        _write_cotangents(
            workspace.begin_suffix(),
            {
                "chain": chain,
                "residual-000": torch.zeros_like(chain),
            },
        )

    factors = KimiUpstreamFactorArchive.create(
        tmp_path / "factors",
        num_layers=2,
        num_experts=1,
        hidden_dimension=4,
        intermediate_dimension=2,
        block_size=2,
        gradient_rank=2,
        expected_layers=(1,),
    )
    result = KimiPipelinedUpstreamReverse(
        adapter=_SyntheticUpstreamAdapter(
            (1.5, 0.75),
            2,
            routed_layers=(1,),
        ),
        boundary_archive=boundaries,
        cotangent_workspace=fisher,
        objective_workspace=objective,
        upstream_factors=factors,
        intermediate_draws={1: (1,)},
        devices=("cuda:0", "cuda:1"),
        slab_buffer_tokens=2,
        direct_io=False,
        coupled_spec=CoupledHadamardSpec(
            residual_block_size=4,
            preactivation_block_size=2,
            postactivation_block_size=2,
        ),
    ).run()

    assert fisher.manifest["chain_boundary"] == 0
    assert objective.manifest["chain_boundary"] == 0
    assert factors.manifest["complete"] is True
    assert [record.layer for record in result.segments[0].layers] == [1]
    assert result.segments[0].layers[0].gradient_rows == 4

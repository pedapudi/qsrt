from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from qsrt.kimi_upstream_factors import (
    KimiUpstreamFactorArchive,
    UpstreamFactorSums,
)


def _archive(path: Path) -> KimiUpstreamFactorArchive:
    return KimiUpstreamFactorArchive.create(
        path,
        num_layers=3,
        num_experts=2,
        hidden_dimension=8,
        intermediate_dimension=4,
        block_size=4,
        gradient_rank=2,
        expected_layers=(1, 2),
        provenance={"weight_revision": "immutable-test"},
    )


def _sums(seed: int) -> tuple[UpstreamFactorSums, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    blocks = torch.randn((2, 2, 4, 4), generator=generator)
    blocks = blocks @ blocks.transpose(-1, -2)
    rows = torch.tensor([2, 4], dtype=torch.int64)
    draws = torch.tensor([0, 6], dtype=torch.uint8)

    left_basis = torch.randn((8, 2), generator=generator)
    right_basis = torch.randn((2, 8), generator=generator)
    gradient = left_basis @ right_basis
    omega_input = torch.randn((8, 2), generator=generator)
    omega_output = torch.randn((8, 2), generator=generator)
    normalizer = 7.0
    gradient_sum = gradient * normalizer
    gradient_left = (gradient_sum @ omega_input).unsqueeze(0).repeat(2, 1, 1)
    gradient_right = (omega_output.T @ gradient_sum).unsqueeze(0).repeat(2, 1, 1)
    return (
        UpstreamFactorSums(
            output_hessian_blocks=blocks.clone(),
            output_hessian_rows=rows,
            intermediate_draws=draws,
            gradient_left=gradient_left,
            gradient_right=gradient_right,
            gradient_rows=torch.tensor([11, 13], dtype=torch.int64),
            gradient_output_projection=omega_output,
            objective_normalizer=normalizer,
        ),
        gradient,
    )


def test_archive_stores_expert_blocks_and_reconstructs_gradient(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "upstream")
    sums, gradient = _sums(12)
    writer = archive.begin_segment(0, 2)
    writer.add(1, sums)
    writer.seal()
    writer.commit()

    blocks, rows, draw = archive.load_expert_output_blocks(1, 1, matrix="w3")
    torch.testing.assert_close(blocks, sums.output_hessian_blocks[1, 1:] / 4)
    assert rows == 4
    assert draw == 6
    torch.testing.assert_close(
        archive.load_expert_gradient(1, 0, matrix="w1"),
        gradient[:4],
        rtol=3e-4,
        atol=3e-4,
    )
    torch.testing.assert_close(
        archive.load_expert_gradient(1, 0, matrix="w3"),
        gradient[4:],
        rtol=3e-4,
        atol=3e-4,
    )
    sketch = archive.load_layer_gradient(1)
    w1_left, w1_right = sketch.expert_factors(0, matrix="w1")
    w3_left, w3_right = sketch.expert_factors(0, matrix="w3")
    torch.testing.assert_close(
        w1_left @ w1_right,
        gradient[:4],
        rtol=3e-4,
        atol=3e-4,
    )
    torch.testing.assert_close(
        w3_left @ w3_right,
        gradient[4:],
        rtol=3e-4,
        atol=3e-4,
    )


def test_archive_accepts_pre_normalized_output_blocks(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "upstream")
    sums, _ = _sums(17)
    normalized = UpstreamFactorSums(
        output_hessian_blocks=(
            sums.output_hessian_blocks
            / sums.output_hessian_rows.to(torch.float32).reshape(-1, 1, 1, 1)
        ),
        output_hessian_rows=sums.output_hessian_rows,
        intermediate_draws=sums.intermediate_draws,
        output_hessian_normalized=True,
    )
    writer = archive.begin_segment(0, 2)
    writer.add(1, normalized)
    writer.seal()
    writer.commit()

    blocks, rows, draw = archive.load_expert_output_blocks(1, 1, matrix="w3")
    torch.testing.assert_close(blocks, normalized.output_hessian_blocks[1, 1:])
    assert rows == 4
    assert draw == 6


def test_archive_promotes_only_cotangent_committed_segments(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "upstream")
    writer = archive.begin_segment(0, 2)
    writer.add(1, _sums(13)[0])
    writer.seal()
    assert archive.recover_pending(("segment-002-003",)) == ()
    promoted = archive.recover_pending(("segment-000-002",))
    assert len(promoted) == 1
    assert archive.layer_path(1).is_file()

    replacement = archive.begin_segment(2, 3)
    replacement.add(2, _sums(14)[0])
    assert archive.discard_uncommitted_pending(("segment-000-002",)) == (
        replacement.path,
    )


def test_archive_writes_segment_layers_concurrently(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "upstream")
    writer = archive.begin_segment(0, 3)
    inputs = ((1, _sums(15)[0]), (2, _sums(16)[0]))

    with ThreadPoolExecutor(max_workers=2) as executor:
        records = tuple(executor.map(lambda values: writer.add(*values), inputs))

    assert {record.layer for record in records} == {1, 2}
    writer.seal()
    writer.commit()
    assert archive.layer_path(1).is_file()
    assert archive.layer_path(2).is_file()

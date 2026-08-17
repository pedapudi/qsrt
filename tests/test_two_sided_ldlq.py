from __future__ import annotations

import torch

from qsrt.two_sided_ldlq import (
    kronecker_quadratic_loss,
    two_sided_block_ldlq,
)


def _round_tiles(tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rounded = torch.round(tiles)
    return rounded, rounded.to(torch.int16)


def _official_three_term_reference(
    source: torch.Tensor,
    input_factor: torch.Tensor,
    output_factor: torch.Tensor,
    *,
    block_size: int,
    feedback_multiplier: float = 1.0,
) -> torch.Tensor:
    reconstructed = torch.zeros_like(source)
    output_blocks = source.shape[0] // block_size
    input_blocks = source.shape[1] // block_size
    for diagonal in range(output_blocks + input_blocks - 2, -1, -1):
        coordinates = [
            (output_block, diagonal - output_block)
            for output_block in range(output_blocks)
            if 0 <= diagonal - output_block < input_blocks
        ]
        targets = []
        for output_block, input_block in coordinates:
            output_begin = output_block * block_size
            output_end = output_begin + block_size
            input_begin = input_block * block_size
            input_end = input_begin + block_size
            error_tail = source[output_begin:, input_begin:] - reconstructed[
                output_begin:, input_begin:
            ]
            source_tile = source[
                output_begin:output_end, input_begin:input_end
            ].clone()
            feedback = (
                output_factor[output_begin:, output_begin:output_end].T
                @ error_tail
                @ input_factor[input_begin:, input_begin:input_end]
            )
            feedback.add_(
                output_factor[output_begin:, output_begin:output_end].T
                @ error_tail[:, :block_size]
            )
            feedback.add_(
                error_tail[:block_size]
                @ input_factor[input_begin:, input_begin:input_end]
            )
            target = source_tile + feedback_multiplier * feedback
            targets.append(target)
        quantized, _ = _round_tiles(torch.stack(targets))
        for batch_index, (output_block, input_block) in enumerate(coordinates):
            output_begin = output_block * block_size
            input_begin = input_block * block_size
            reconstructed[
                output_begin : output_begin + block_size,
                input_begin : input_begin + block_size,
            ] = quantized[batch_index]
    return reconstructed


def test_cached_recurrence_matches_the_published_three_term_update() -> None:
    generator = torch.Generator().manual_seed(20260817)
    source = torch.randn(6, 8, generator=generator)
    input_factor = torch.tril(
        0.35 * torch.randn(8, 8, generator=generator), diagonal=-1
    )
    output_factor = torch.tril(
        0.35 * torch.randn(6, 6, generator=generator), diagonal=-1
    )
    for begin in range(0, 8, 2):
        input_factor[begin : begin + 2, begin : begin + 2] = 0
    for begin in range(0, 6, 2):
        output_factor[begin : begin + 2, begin : begin + 2] = 0

    expected = _official_three_term_reference(
        source,
        input_factor,
        output_factor,
        block_size=2,
    )
    actual, codes = two_sided_block_ldlq(
        source,
        input_factor,
        output_factor,
        _round_tiles,
        block_size=2,
    )

    assert torch.equal(actual, expected)
    assert codes.shape == (3, 4, 2, 2)
    assert torch.equal(codes, actual.reshape(3, 2, 4, 2).permute(0, 2, 1, 3))


def test_anti_diagonal_batching_does_not_expose_same_diagonal_errors() -> None:
    source = torch.tensor(
        [
            [0.49, 0.49, 0.49, 0.49],
            [0.49, 0.49, 0.49, 0.49],
            [0.49, 0.49, 0.49, 0.49],
            [0.49, 0.49, 0.49, 0.49],
        ]
    )
    input_factor = torch.zeros(4, 4)
    output_factor = torch.zeros(4, 4)
    input_factor[2:, :2] = 1.0
    output_factor[2:, :2] = 1.0
    batch_sizes: list[int] = []

    def record_batches(tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_sizes.append(tiles.shape[0])
        return _round_tiles(tiles)

    two_sided_block_ldlq(
        source,
        input_factor,
        output_factor,
        record_batches,
        block_size=2,
    )

    assert batch_sizes == [1, 2, 1]


def test_zero_output_factor_reduces_to_input_direction_feedback() -> None:
    source = torch.tensor(
        [
            [0.2, 0.7, 0.1, 0.9],
            [0.8, 0.3, 0.6, 0.4],
            [0.1, 0.2, 0.8, 0.7],
            [0.9, 0.8, 0.3, 0.2],
        ]
    )
    input_factor = torch.zeros(4, 4)
    input_factor[2:, :2] = torch.tensor([[0.4, -0.2], [0.1, 0.3]])
    output_factor = torch.zeros(4, 4)

    reconstructed, _ = two_sided_block_ldlq(
        source,
        input_factor,
        output_factor,
        _round_tiles,
        block_size=2,
    )
    expected = _official_three_term_reference(
        source,
        input_factor,
        output_factor,
        block_size=2,
    )

    assert torch.equal(reconstructed, expected)


def test_feedback_multiplier_scales_the_complete_two_sided_update() -> None:
    generator = torch.Generator().manual_seed(912)
    source = torch.randn(6, 8, generator=generator)
    input_factor = torch.tril(
        0.5 * torch.randn(8, 8, generator=generator), diagonal=-1
    )
    output_factor = torch.tril(
        0.5 * torch.randn(6, 6, generator=generator), diagonal=-1
    )
    for begin in range(0, 8, 2):
        input_factor[begin : begin + 2, begin : begin + 2] = 0
    for begin in range(0, 6, 2):
        output_factor[begin : begin + 2, begin : begin + 2] = 0

    expected = _official_three_term_reference(
        source,
        input_factor,
        output_factor,
        block_size=2,
        feedback_multiplier=0.5,
    )
    actual, _ = two_sided_block_ldlq(
        source,
        input_factor,
        output_factor,
        _round_tiles,
        block_size=2,
        feedback_multiplier=0.5,
    )
    direct, _ = two_sided_block_ldlq(
        source,
        input_factor,
        output_factor,
        _round_tiles,
        block_size=2,
        feedback_multiplier=0.0,
    )

    assert torch.equal(actual, expected)
    assert torch.equal(direct, torch.round(source))


def test_kronecker_loss_matches_vectorized_kron_form() -> None:
    error = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    input_metric = torch.tensor([[2.0, 0.25], [0.25, 1.0]])
    output_metric = torch.tensor([[1.5, -0.1], [-0.1, 0.75]])

    direct = kronecker_quadratic_loss(error, input_metric, output_metric)
    vector = error.reshape(-1)
    expected = vector @ torch.kron(output_metric, input_metric) @ vector

    assert direct == expected


def test_two_sided_recurrence_rejects_nonzero_diagonal_blocks() -> None:
    source = torch.zeros(4, 4)
    input_factor = torch.eye(4)
    output_factor = torch.zeros(4, 4)

    try:
        two_sided_block_ldlq(
            source,
            input_factor,
            output_factor,
            _round_tiles,
            block_size=2,
        )
    except ValueError as error:
        assert "zero 2-coordinate diagonal blocks" in str(error)
    else:
        raise AssertionError("nonzero factor diagonal block was accepted")

"""Two-sided block error feedback for a fixed tile quantizer.

The ordinary BlockLDLQ encoder feeds committed error along a linear layer's
input coordinates. Model-Preserving Adaptive Rounding extends that recurrence
with a second block factor along output coordinates. The resulting update uses
a Kronecker-factored loss metric while leaving the tile quantizer and stored
payload unchanged.

This module contains the quantizer-independent recurrence. Callers supply a
function that maps a batch of source-oriented square tiles to reconstructed
tiles and their codes. The production adapter transposes each tile into QSRT's
encoder orientation before calling the existing Viterbi kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import torch


TileQuantizer: TypeAlias = Callable[
    [torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


def _validate_factor(
    factor: torch.Tensor,
    *,
    dimension: int,
    block_size: int,
    role: str,
) -> None:
    if (
        factor.ndim != 2
        or factor.shape != (dimension, dimension)
        or not torch.is_floating_point(factor)
    ):
        raise ValueError(
            f"{role} factor must be a floating-point {dimension} by "
            f"{dimension} matrix"
        )
    if factor.device.type != "meta" and not bool(torch.isfinite(factor).all()):
        raise ValueError(f"{role} factor must contain only finite values")
    for start in range(0, dimension, block_size):
        diagonal_block = factor[
            start : start + block_size,
            start : start + block_size,
        ]
        if not torch.count_nonzero(diagonal_block).item() == 0:
            raise ValueError(
                f"{role} factor must have zero {block_size}-coordinate "
                "diagonal blocks"
            )


def validate_two_sided_inputs(
    source_weight: torch.Tensor,
    input_factor: torch.Tensor,
    output_factor: torch.Tensor,
    *,
    block_size: int,
) -> None:
    """Validate a source-oriented matrix and its strict block factors."""

    if (
        source_weight.ndim != 2
        or not torch.is_floating_point(source_weight)
        or source_weight.device.type == "meta"
        or not bool(torch.isfinite(source_weight).all())
    ):
        raise ValueError("source weight must be a finite floating-point matrix")
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise TypeError("block_size must be an integer")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    output_dimension, input_dimension = source_weight.shape
    if output_dimension % block_size or input_dimension % block_size:
        raise ValueError("source dimensions must contain complete square blocks")
    if (
        source_weight.device != input_factor.device
        or source_weight.device != output_factor.device
        or source_weight.dtype != input_factor.dtype
        or source_weight.dtype != output_factor.dtype
    ):
        raise ValueError("source weight and factors must share device and dtype")
    _validate_factor(
        input_factor,
        dimension=input_dimension,
        block_size=block_size,
        role="input",
    )
    _validate_factor(
        output_factor,
        dimension=output_dimension,
        block_size=block_size,
        role="output",
    )


def two_sided_block_ldlq(
    source_weight: torch.Tensor,
    input_factor: torch.Tensor,
    output_factor: torch.Tensor,
    quantize_tiles: TileQuantizer,
    *,
    block_size: int = 16,
    feedback_multiplier: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize blocks with input- and output-direction error feedback.

    ``source_weight`` uses ordinary linear-layer orientation
    ``[output, input]``. ``input_factor`` and ``output_factor`` are the strict
    lower block factors produced by block LDL decompositions of the two
    Kronecker metrics. Their diagonal blocks must be zero.

    Blocks on one anti-diagonal have no dependencies on each other. The
    function batches each anti-diagonal into one tile-quantizer call. A running
    right-compensated error matrix avoids evaluating the three-term YAQA
    expression with a dense tail-by-tail product for every block.
    """

    validate_two_sided_inputs(
        source_weight,
        input_factor,
        output_factor,
        block_size=block_size,
    )
    if not callable(quantize_tiles):
        raise TypeError("quantize_tiles must be callable")
    if (
        isinstance(feedback_multiplier, bool)
        or not isinstance(feedback_multiplier, (int, float))
        or not 0.0 <= float(feedback_multiplier) <= 1.0
    ):
        raise ValueError("feedback_multiplier must be a real number in [0, 1]")
    feedback_multiplier = float(feedback_multiplier)

    output_dimension, input_dimension = source_weight.shape
    output_blocks = output_dimension // block_size
    input_blocks = input_dimension // block_size
    reconstructed = torch.zeros_like(source_weight)

    # R = E (I + L_input). Before a block is committed, its own E term is
    # zero, so R at that block contains only feedback from processed input
    # coordinates. L_output.T @ R then supplies output feedback and the
    # input/output cross term together.
    right_compensated_error = torch.zeros_like(source_weight)
    stored_codes: torch.Tensor | None = None

    for diagonal in range(output_blocks + input_blocks - 2, -1, -1):
        coordinates = [
            (output_block, diagonal - output_block)
            for output_block in range(output_blocks)
            if 0 <= diagonal - output_block < input_blocks
        ]
        targets: list[torch.Tensor] = []
        for output_block, input_block in coordinates:
            output_begin = output_block * block_size
            output_end = output_begin + block_size
            input_begin = input_block * block_size
            input_end = input_begin + block_size
            source_tile = source_weight[
                output_begin:output_end, input_begin:input_end
            ]
            feedback = right_compensated_error[
                output_begin:output_end, input_begin:input_end
            ]
            if output_end < output_dimension:
                feedback = feedback + output_factor[
                    output_end:, output_begin:output_end
                ].T @ right_compensated_error[
                    output_end:, input_begin:input_end
                ]
            target = (
                source_tile
                if feedback_multiplier == 0.0
                else source_tile + feedback_multiplier * feedback
            )
            targets.append(target)

        target_batch = torch.stack(targets)
        quantized_batch, code_batch = quantize_tiles(target_batch)
        if quantized_batch.shape != target_batch.shape:
            raise ValueError("tile quantizer returned reconstructions with the wrong shape")
        if (
            quantized_batch.device != source_weight.device
            or quantized_batch.dtype != source_weight.dtype
            or not bool(torch.isfinite(quantized_batch).all())
        ):
            raise ValueError(
                "tile quantizer reconstructions must be finite and match the source"
            )
        if code_batch.ndim < 1 or code_batch.shape[0] != len(coordinates):
            raise ValueError("tile quantizer returned codes with the wrong batch size")
        if stored_codes is None:
            stored_codes = torch.empty(
                (output_blocks, input_blocks, *code_batch.shape[1:]),
                dtype=code_batch.dtype,
                device=code_batch.device,
            )
        elif (
            stored_codes.shape[2:] != code_batch.shape[1:]
            or stored_codes.dtype != code_batch.dtype
            or stored_codes.device != code_batch.device
        ):
            raise ValueError("tile quantizer changed its code shape, dtype, or device")

        # No block on an anti-diagonal depends on another block on that same
        # anti-diagonal. Commit all reconstructions only after the batch has
        # been quantized, then update feedback for later diagonals.
        for batch_index, (output_block, input_block) in enumerate(coordinates):
            output_begin = output_block * block_size
            output_end = output_begin + block_size
            input_begin = input_block * block_size
            input_end = input_begin + block_size
            quantized = quantized_batch[batch_index]
            reconstructed[output_begin:output_end, input_begin:input_end] = quantized
            assert stored_codes is not None
            stored_codes[output_block, input_block] = code_batch[batch_index]
            error = (
                source_weight[output_begin:output_end, input_begin:input_end]
                - quantized
            )
            if input_begin:
                right_compensated_error[
                    output_begin:output_end, :input_begin
                ].addmm_(
                    error,
                    input_factor[input_begin:input_end, :input_begin],
                )
            right_compensated_error[
                output_begin:output_end, input_begin:input_end
            ].add_(error)

    if stored_codes is None:
        raise AssertionError("two-sided traversal did not visit any blocks")
    return reconstructed, stored_codes


def kronecker_quadratic_loss(
    error: torch.Tensor,
    input_metric: torch.Tensor,
    output_metric: torch.Tensor,
) -> torch.Tensor:
    """Return ``tr(H_out E H_in E.T)`` for source-oriented error."""

    if error.ndim != 2 or not torch.is_floating_point(error):
        raise TypeError("error must be a floating-point matrix")
    output_dimension, input_dimension = error.shape
    if input_metric.shape != (input_dimension, input_dimension):
        raise ValueError("input metric does not match the error input dimension")
    if output_metric.shape != (output_dimension, output_dimension):
        raise ValueError("output metric does not match the error output dimension")
    return torch.sum((output_metric @ error @ input_metric) * error)


__all__ = [
    "kronecker_quadratic_loss",
    "two_sided_block_ldlq",
    "validate_two_sided_inputs",
]

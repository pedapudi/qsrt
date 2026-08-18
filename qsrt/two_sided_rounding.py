"""Reference two-sided adaptive rounding for block trellis quantizers.

Weights use the QSRT encoder orientation ``[input, output]``.  The quantizer
callback receives complete ``[block, block]`` tiles in that same orientation,
so a callback may enforce one legal trellis path over the entire tile.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from qsrt.qsrt_coupled import block_hadamard


TileQuantizer = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class TwoSidedRoundingResult:
    """Decoded weights and one payload object per input/output tile."""

    reconstruction: torch.Tensor
    payload: torch.Tensor


def transform_output_hessian_for_regularization(
    hessian: torch.Tensor,
    output_scales: torch.Tensor,
    *,
    block_size: int = 128,
) -> torch.Tensor:
    """Move an output metric through EXL's output scale and Hadamard basis.

    If decoded encoder-oriented error is ``E_work @ H @ D_scale``, the
    returned matrix is ``H @ D_scale @ hessian @ D_scale @ H``.
    """

    _finite_matrix(hessian, "output Hessian")
    if hessian.shape[0] != hessian.shape[1]:
        raise ValueError("output Hessian must be square")
    scales = output_scales.flatten()
    if scales.numel() != hessian.shape[0]:
        raise ValueError("output scales do not match the Hessian dimension")
    if not bool(torch.all(torch.isfinite(scales))):
        raise ValueError("output scales must be finite")
    if scales.device != hessian.device:
        scales = scales.to(hessian.device)
    result = hessian * scales.to(hessian.dtype)[None, :]
    result = result * scales.to(hessian.dtype)[:, None]
    result = block_hadamard(result, block_size=block_size, dim=1)
    result = block_hadamard(result, block_size=block_size, dim=0)
    return ((result + result.T) * 0.5).contiguous()


def transform_output_hessian_blocks_for_regularization(
    hessian_blocks: torch.Tensor,
    output_scales: torch.Tensor,
    *,
    block_size: int = 128,
) -> torch.Tensor:
    """Move independent output-metric blocks through scale and Hadamard bases.

    ``hessian_blocks`` has shape ``[..., blocks, block_size, block_size]`` and
    ``output_scales`` has shape ``[..., blocks * block_size]``.  The leading
    dimensions must agree.  No approximation is introduced when the source
    output metric is block diagonal on the same block boundaries.
    """

    if hessian_blocks.ndim < 3 or hessian_blocks.shape[-2:] != (
        block_size,
        block_size,
    ):
        raise ValueError("output Hessian blocks have incompatible geometry")
    blocks = int(hessian_blocks.shape[-3])
    if output_scales.shape != (*hessian_blocks.shape[:-3], blocks * block_size):
        raise ValueError("output scales do not match the block Hessian geometry")
    if hessian_blocks.device != output_scales.device:
        raise ValueError("output Hessian blocks and scales must share one device")
    if not bool(torch.all(torch.isfinite(hessian_blocks))) or not bool(
        torch.all(torch.isfinite(output_scales))
    ):
        raise ValueError("output Hessian blocks and scales must be finite")

    scales = output_scales.to(hessian_blocks.dtype).reshape(
        *hessian_blocks.shape[:-3], blocks, block_size
    )
    result = hessian_blocks * scales.unsqueeze(-2)
    result = result * scales.unsqueeze(-1)
    result = block_hadamard(result, block_size=block_size, dim=-1)
    result = block_hadamard(result, block_size=block_size, dim=-2)
    return ((result + result.transpose(-1, -2)) * 0.5).contiguous()


def _finite_matrix(value: torch.Tensor, name: str) -> None:
    if value.ndim != 2 or not value.numel():
        raise ValueError(f"{name} must be a nonempty matrix")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must be finite")


def _strict_block_ldl_factor(
    hessian: torch.Tensor,
    block_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return the strict block-lower factor used for adaptive feedback."""

    _finite_matrix(hessian, "Hessian")
    if hessian.shape[0] != hessian.shape[1]:
        raise ValueError("Hessian must be square")
    dimension = int(hessian.shape[0])
    if block_size <= 0 or dimension % block_size:
        raise ValueError("block size must divide the Hessian dimension")
    matrix = hessian.to(device=device, dtype=dtype)
    cholesky = torch.linalg.cholesky((matrix + matrix.T) * 0.5)
    lower = cholesky.clone()
    for begin in range(0, dimension, block_size):
        end = begin + block_size
        diagonal = cholesky[begin:end, begin:end]
        lower[:, begin:end] = torch.linalg.solve_triangular(
            diagonal.T,
            cholesky[:, begin:end].T,
            upper=True,
        ).T
        lower[begin:end, begin:end].zero_()
    return lower.contiguous()


def _reverse_block_indices(
    dimension: int,
    block_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    blocks = dimension // block_size
    return torch.arange(dimension, device=device).reshape(blocks, block_size).flip(0).reshape(-1)


def _unit_block_inverse_cholesky_factor(
    hessian: torch.Tensor,
    block_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the normalized inverse-Cholesky factor in reversed block order."""

    _finite_matrix(hessian, "Hessian")
    if hessian.shape[0] != hessian.shape[1]:
        raise ValueError("Hessian must be square")
    dimension = int(hessian.shape[0])
    if block_size <= 0 or dimension % block_size:
        raise ValueError("block size must divide the Hessian dimension")

    reverse = _reverse_block_indices(dimension, block_size, device=device)
    matrix = hessian.to(device=device, dtype=dtype)
    matrix = matrix.index_select(0, reverse).index_select(1, reverse)
    matrix = (matrix + matrix.T) * 0.5
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(matrix))
    lower = torch.linalg.cholesky((inverse + inverse.T) * 0.5)
    for begin in range(0, dimension, block_size):
        end = begin + block_size
        diagonal = lower[begin:end, begin:end]
        lower[:, begin:end] = torch.linalg.solve_triangular(
            diagonal.T,
            lower[:, begin:end].T,
            upper=True,
        ).T
    return lower.contiguous(), reverse


def factor_bakron_hessian(
    hessian: torch.Tensor,
    *,
    block_size: int = 16,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor one Hessian for reuse by prepared BaKron recurrences."""

    return _unit_block_inverse_cholesky_factor(
        hessian,
        block_size,
        dtype=dtype,
        device=hessian.device,
    )


def _unit_block_inverse_cholesky_factor_batch(
    hessians: torch.Tensor,
    block_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized inverse-Cholesky factors for a matrix batch."""

    if hessians.ndim != 3 or not hessians.shape[0]:
        raise ValueError("Hessians must be a nonempty matrix batch")
    if hessians.shape[1] != hessians.shape[2]:
        raise ValueError("Hessians must be square")
    if not bool(torch.all(torch.isfinite(hessians))):
        raise ValueError("Hessians must be finite")
    dimension = int(hessians.shape[1])
    if block_size <= 0 or dimension % block_size:
        raise ValueError("block size must divide the Hessian dimension")

    reverse = _reverse_block_indices(dimension, block_size, device=device)
    matrices = hessians.to(device=device, dtype=dtype)
    matrices = matrices.index_select(1, reverse).index_select(2, reverse)
    matrices = (matrices + matrices.transpose(1, 2)) * 0.5
    inverses = torch.cholesky_inverse(torch.linalg.cholesky(matrices))
    lower = torch.linalg.cholesky((inverses + inverses.transpose(1, 2)) * 0.5)
    for begin in range(0, dimension, block_size):
        end = begin + block_size
        diagonal = lower[:, begin:end, begin:end]
        lower[:, :, begin:end] = torch.linalg.solve_triangular(
            diagonal.transpose(1, 2),
            lower[:, :, begin:end].transpose(1, 2),
            upper=True,
        ).transpose(1, 2)
    return lower.contiguous(), reverse


def _allocate_payload_grid(
    payload: torch.Tensor,
    input_blocks: int,
    output_blocks: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    if payload.ndim < 1:
        raise ValueError("tile quantizer payload must have one leading item per tile")
    payload_shape = tuple(payload.shape[1:])
    return (
        torch.empty(
            (input_blocks, output_blocks, *payload_shape),
            dtype=payload.dtype,
            device=payload.device,
        ),
        payload_shape,
    )


def _block_matrix_view(
    matrix: torch.Tensor,
    row_blocks: int,
    column_blocks: int,
    block_size: int,
) -> torch.Tensor:
    return matrix.reshape(row_blocks, block_size, column_blocks, block_size).permute(
        0, 2, 1, 3
    )


def _block_matrix_batch_view(
    matrices: torch.Tensor,
    row_blocks: int,
    column_blocks: int,
    block_size: int,
) -> torch.Tensor:
    batch = matrices.shape[0]
    return matrices.reshape(
        batch, row_blocks, block_size, column_blocks, block_size
    ).permute(0, 1, 3, 2, 4)


def _apply_recursive_block_update(
    workspace: torch.Tensor,
    quantized: torch.Tensor,
    output_factor: torch.Tensor,
    input_factor: torch.Tensor,
    *,
    first_diagonal: int,
    middle_diagonal: int,
    last_diagonal: int,
    block_size: int,
    chunk_blocks: int,
) -> None:
    """Propagate one processed block band into the following block band."""

    output_dimension, input_dimension = map(int, workspace.shape)
    output_blocks = output_dimension // block_size
    input_blocks = input_dimension // block_size
    source_width = middle_diagonal - first_diagonal
    full_width = last_diagonal - first_diagonal
    target_width = last_diagonal - middle_diagonal
    device = workspace.device

    workspace_blocks = _block_matrix_view(
        workspace, output_blocks, input_blocks, block_size
    )
    quantized_blocks = _block_matrix_view(
        quantized, output_blocks, input_blocks, block_size
    )
    output_factor_blocks = _block_matrix_view(
        output_factor, output_blocks, output_blocks, block_size
    )
    input_factor_blocks = _block_matrix_view(
        input_factor, input_blocks, input_blocks, block_size
    )

    # First restricted product: output_factor @ error.  For a fixed input
    # block, both the error support and every output block that can reach the
    # requested band are contiguous ranges no wider than this recursion node.
    error_offsets = torch.arange(source_width, device=device)
    propagated_offsets = torch.arange(full_width, device=device)
    intermediate = torch.zeros(
        (output_blocks, input_blocks, block_size, block_size),
        dtype=workspace.dtype,
        device=device,
    )
    for input_begin in range(0, input_blocks, chunk_blocks):
        input_end = min(input_begin + chunk_blocks, input_blocks)
        input_indices = torch.arange(input_begin, input_end, device=device)
        batch = input_indices.numel()
        error_row_start = (first_diagonal - input_indices).clamp(0, output_blocks)
        error_row_end = (middle_diagonal - input_indices).clamp(0, output_blocks)
        propagated_row_end = (last_diagonal - input_indices).clamp(
            0, output_blocks
        )
        error_rows = error_row_start[:, None] + error_offsets
        propagated_rows = error_row_start[:, None] + propagated_offsets
        valid_error_rows = error_rows < error_row_end[:, None]
        valid_propagated_rows = propagated_rows < propagated_row_end[:, None]
        safe_error_rows = error_rows.clamp(max=output_blocks - 1)
        safe_propagated_rows = propagated_rows.clamp(max=output_blocks - 1)

        left = output_factor_blocks[
            safe_propagated_rows[:, :, None],
            safe_error_rows[:, None, :],
        ]
        left = left * (
            valid_propagated_rows[:, :, None, None, None]
            & valid_error_rows[:, None, :, None, None]
        )
        left = left.permute(0, 1, 3, 2, 4).reshape(
            batch,
            full_width * block_size,
            source_width * block_size,
        )
        error = (
            quantized_blocks[safe_error_rows, input_indices[:, None]]
            - workspace_blocks[safe_error_rows, input_indices[:, None]]
        )
        error = error * valid_error_rows[:, :, None, None]
        error = error.reshape(batch, source_width * block_size, block_size)
        first_product = torch.bmm(left, error).reshape(
            batch, full_width, block_size, block_size
        )
        batch_inputs = input_indices[:, None].expand(-1, full_width)
        intermediate[
            propagated_rows[valid_propagated_rows],
            batch_inputs[valid_propagated_rows],
        ] = first_product[valid_propagated_rows]

    # Second restricted product: intermediate @ input_factor.T.  Only target
    # blocks in the second child are materialized.
    intermediate_offsets = torch.arange(full_width, device=device)
    target_offsets = torch.arange(target_width, device=device)
    for output_begin in range(0, output_blocks, chunk_blocks):
        output_end = min(output_begin + chunk_blocks, output_blocks)
        output_indices = torch.arange(output_begin, output_end, device=device)
        batch = output_indices.numel()
        target_column_start = (middle_diagonal - output_indices).clamp(
            0, input_blocks
        )
        target_column_end = (last_diagonal - output_indices).clamp(0, input_blocks)
        intermediate_column_start = (first_diagonal - output_indices).clamp(
            0, input_blocks
        )
        intermediate_column_end = torch.minimum(
            torch.full_like(output_indices, middle_diagonal).clamp(max=input_blocks),
            (last_diagonal - output_indices).clamp(0, input_blocks),
        )
        intermediate_columns = (
            intermediate_column_start[:, None] + intermediate_offsets
        )
        target_columns = target_column_start[:, None] + target_offsets
        valid_intermediate_columns = (
            intermediate_columns < intermediate_column_end[:, None]
        )
        valid_target_columns = target_columns < target_column_end[:, None]
        safe_intermediate_columns = intermediate_columns.clamp(max=input_blocks - 1)
        safe_target_columns = target_columns.clamp(max=input_blocks - 1)

        first_product_rows = intermediate[
            output_indices[:, None], safe_intermediate_columns
        ]
        first_product_rows = (
            first_product_rows * valid_intermediate_columns[:, :, None, None]
        )
        first_product_rows = first_product_rows.permute(0, 2, 1, 3).reshape(
            batch,
            block_size,
            full_width * block_size,
        )
        right = input_factor_blocks[
            safe_target_columns[:, :, None],
            safe_intermediate_columns[:, None, :],
        ]
        right = right * (
            valid_target_columns[:, :, None, None, None]
            & valid_intermediate_columns[:, None, :, None, None]
        )
        right = right.permute(0, 2, 4, 1, 3).reshape(
            batch,
            full_width * block_size,
            target_width * block_size,
        )
        update = torch.bmm(first_product_rows, right).reshape(
            batch, block_size, target_width, block_size
        ).permute(0, 2, 1, 3)
        batch_outputs = output_indices[:, None].expand(-1, target_width)
        workspace_blocks[
            batch_outputs[valid_target_columns],
            target_columns[valid_target_columns],
        ] += update[valid_target_columns]


def yaqa_block_round_encoder(
    weight: torch.Tensor,
    input_hessian: torch.Tensor,
    output_hessian: torch.Tensor,
    quantize_tiles: TileQuantizer,
    *,
    block_size: int = 16,
    work_dtype: torch.dtype = torch.float64,
) -> TwoSidedRoundingResult:
    """Round encoder-oriented weights with two-sided block feedback.

    ``input_hessian`` indexes weight rows and ``output_hessian`` indexes weight
    columns.  Both must already be transformed into the exact coordinates used
    by ``weight``.  The recurrence matches YAQA's reverse anti-diagonal block
    update while preserving the supplied tile quantizer as the legal set.
    """

    _finite_matrix(weight, "weight")
    if work_dtype not in (torch.float32, torch.float64):
        raise ValueError("work_dtype must be float32 or float64")
    input_dimension, output_dimension = map(int, weight.shape)
    if input_dimension % block_size or output_dimension % block_size:
        raise ValueError("block size must divide both weight dimensions")
    if tuple(input_hessian.shape) != (input_dimension, input_dimension):
        raise ValueError("input Hessian has the wrong shape")
    if tuple(output_hessian.shape) != (output_dimension, output_dimension):
        raise ValueError("output Hessian has the wrong shape")

    device = weight.device
    input_factor = _strict_block_ldl_factor(
        input_hessian,
        block_size,
        dtype=work_dtype,
        device=device,
    )
    output_factor = _strict_block_ldl_factor(
        output_hessian,
        block_size,
        dtype=work_dtype,
        device=device,
    )

    # YAQA's published recurrence uses [output, input] weights.  Transposing
    # only around that recurrence keeps the quantizer callback in QSRT's
    # [input, output] tile orientation.
    source = weight.to(dtype=work_dtype).T.contiguous()
    quantized = torch.zeros_like(source)
    output_blocks = output_dimension // block_size
    input_blocks = input_dimension // block_size
    starts = [
        *((output_blocks - index - 1, input_blocks - 1) for index in range(output_blocks)),
        *((0, input_blocks - index - 1) for index in range(input_blocks)),
    ]
    payload_grid: torch.Tensor | None = None
    payload_shape: tuple[int, ...] | None = None

    for output_start_block, input_start_block in starts:
        delta = source - quantized
        targets: list[torch.Tensor] = []
        tile_indices: list[tuple[int, int]] = []
        output_block = output_start_block
        input_block = input_start_block
        while output_block < output_blocks and input_block >= 0:
            output_begin = output_block * block_size
            output_end = output_begin + block_size
            input_begin = input_block * block_size
            input_end = input_begin + block_size
            output_tail = slice(output_begin, output_dimension)
            input_tail = slice(input_begin, input_dimension)
            output_tile = slice(output_begin, output_end)
            input_tile = slice(input_begin, input_end)
            target = source[output_tile, input_tile] + (
                output_factor[output_tail, output_tile].T
                @ delta[output_tail, input_tail]
                @ input_factor[input_tail, input_tile]
                + output_factor[output_tail, output_tile].T
                @ delta[output_tail, input_tile]
                + delta[output_tile, input_tail]
                @ input_factor[input_tail, input_tile]
            )
            targets.append(target.T)
            tile_indices.append((input_block, output_block))
            output_block += 1
            input_block -= 1

        tile_batch = torch.stack(targets)
        reconstructed_tiles, payload = quantize_tiles(tile_batch)
        if reconstructed_tiles.shape != tile_batch.shape:
            raise ValueError("tile quantizer returned the wrong reconstruction shape")
        if payload.ndim < 1 or payload.shape[0] != tile_batch.shape[0]:
            raise ValueError("tile quantizer payload must have one leading item per tile")
        if not bool(torch.all(torch.isfinite(reconstructed_tiles))):
            raise ValueError("tile quantizer returned a nonfinite reconstruction")
        if payload_grid is None:
            payload_shape = tuple(payload.shape[1:])
            payload_grid = torch.empty(
                (input_blocks, output_blocks, *payload_shape),
                dtype=payload.dtype,
                device=payload.device,
            )
        elif tuple(payload.shape[1:]) != payload_shape:
            raise ValueError("tile quantizer payload shape changed between calls")

        reconstructed_tiles = reconstructed_tiles.to(
            device=device, dtype=work_dtype
        )
        for batch_index, (input_index, output_index) in enumerate(tile_indices):
            input_begin = input_index * block_size
            output_begin = output_index * block_size
            quantized[
                output_begin : output_begin + block_size,
                input_begin : input_begin + block_size,
            ] = reconstructed_tiles[batch_index].T
            assert payload_grid is not None
            payload_grid[input_index, output_index] = payload[batch_index]

    assert payload_grid is not None
    return TwoSidedRoundingResult(
        reconstruction=quantized.T.to(dtype=weight.dtype).contiguous(),
        payload=payload_grid,
    )


def bakron_block_antidiagonal_encoder(
    weight: torch.Tensor,
    input_hessian: torch.Tensor,
    output_hessian: torch.Tensor,
    quantize_tiles: TileQuantizer,
    *,
    block_size: int = 16,
    work_dtype: torch.dtype = torch.float64,
) -> TwoSidedRoundingResult:
    """Run the block lift of BaKron's anti-diagonal reference algorithm.

    Block order is reversed before applying BaKron's forward recurrence.  This
    is the inverse-Hessian form of :func:`yaqa_block_round_encoder`; both
    functions produce the same legal tile paths when arithmetic and quantizer
    tie-breaking agree.  This implementation performs full matrix products at
    every anti-diagonal and is therefore a correctness oracle, not a scalable
    matrix encoder.
    """

    _finite_matrix(weight, "weight")
    if work_dtype not in (torch.float32, torch.float64):
        raise ValueError("work_dtype must be float32 or float64")
    input_dimension, output_dimension = map(int, weight.shape)
    if input_dimension % block_size or output_dimension % block_size:
        raise ValueError("block size must divide both weight dimensions")
    if tuple(input_hessian.shape) != (input_dimension, input_dimension):
        raise ValueError("input Hessian has the wrong shape")
    if tuple(output_hessian.shape) != (output_dimension, output_dimension):
        raise ValueError("output Hessian has the wrong shape")

    device = weight.device
    input_factor, input_reverse = _unit_block_inverse_cholesky_factor(
        input_hessian,
        block_size,
        dtype=work_dtype,
        device=device,
    )
    output_factor, output_reverse = _unit_block_inverse_cholesky_factor(
        output_hessian,
        block_size,
        dtype=work_dtype,
        device=device,
    )
    source = weight.to(dtype=work_dtype).T.contiguous()
    workspace = source.index_select(0, output_reverse).index_select(1, input_reverse).clone()
    quantized = torch.empty_like(workspace)
    output_blocks = output_dimension // block_size
    input_blocks = input_dimension // block_size
    payload_grid: torch.Tensor | None = None
    payload_shape: tuple[int, ...] | None = None

    for diagonal in range(output_blocks + input_blocks - 1):
        targets: list[torch.Tensor] = []
        tile_indices: list[tuple[int, int]] = []
        output_begin_block = max(0, diagonal - input_blocks + 1)
        output_end_block = min(output_blocks, diagonal + 1)
        for output_block in range(output_begin_block, output_end_block):
            input_block = diagonal - output_block
            output_begin = output_block * block_size
            input_begin = input_block * block_size
            targets.append(
                workspace[
                    output_begin : output_begin + block_size,
                    input_begin : input_begin + block_size,
                ].T
            )
            tile_indices.append((input_block, output_block))

        tile_batch = torch.stack(targets)
        reconstructed_tiles, payload = quantize_tiles(tile_batch)
        if reconstructed_tiles.shape != tile_batch.shape:
            raise ValueError("tile quantizer returned the wrong reconstruction shape")
        if payload.ndim < 1 or payload.shape[0] != tile_batch.shape[0]:
            raise ValueError("tile quantizer payload must have one leading item per tile")
        if not bool(torch.all(torch.isfinite(reconstructed_tiles))):
            raise ValueError("tile quantizer returned a nonfinite reconstruction")
        if payload_grid is None:
            payload_grid, payload_shape = _allocate_payload_grid(
                payload,
                input_blocks,
                output_blocks,
            )
        elif tuple(payload.shape[1:]) != payload_shape:
            raise ValueError("tile quantizer payload shape changed between calls")

        delta = torch.zeros_like(workspace)
        reconstructed_tiles = reconstructed_tiles.to(device=device, dtype=work_dtype)
        for batch_index, (input_index, output_index) in enumerate(tile_indices):
            input_begin = input_index * block_size
            output_begin = output_index * block_size
            output_tile = slice(output_begin, output_begin + block_size)
            input_tile = slice(input_begin, input_begin + block_size)
            reconstruction = reconstructed_tiles[batch_index].T
            delta[output_tile, input_tile] = reconstruction - workspace[output_tile, input_tile]
            quantized[output_tile, input_tile] = reconstruction
            original_input = input_blocks - input_index - 1
            original_output = output_blocks - output_index - 1
            assert payload_grid is not None
            payload_grid[original_input, original_output] = payload[batch_index]
        workspace.add_(output_factor @ delta @ input_factor.T)

    assert payload_grid is not None
    reconstruction = quantized.index_select(0, output_reverse).index_select(1, input_reverse).T
    return TwoSidedRoundingResult(
        reconstruction=reconstruction.to(dtype=weight.dtype).contiguous(),
        payload=payload_grid,
    )


def bakron_block_round_encoder(
    weight: torch.Tensor,
    input_hessian: torch.Tensor,
    output_hessian: torch.Tensor,
    quantize_tiles: TileQuantizer,
    *,
    block_size: int = 16,
    work_dtype: torch.dtype = torch.float64,
    update_chunk_blocks: int = 16,
    update_backend: str = "auto",
) -> TwoSidedRoundingResult:
    """Run two-sided adaptive rounding with recursive block-band updates.

    The recurrence is the block lift of BaKron's anti-diagonal
    divide-and-conquer solver.  It preserves one quantizer call per block
    anti-diagonal while restricting each matrix product to the source and
    destination bands of its recursion node.
    """

    _finite_matrix(weight, "weight")
    if work_dtype not in (torch.float32, torch.float64):
        raise ValueError("work_dtype must be float32 or float64")
    if update_chunk_blocks <= 0:
        raise ValueError("update_chunk_blocks must be positive")
    if update_backend not in ("auto", "torch", "cuda"):
        raise ValueError("update_backend must be 'auto', 'torch', or 'cuda'")
    input_dimension, output_dimension = map(int, weight.shape)
    if input_dimension % block_size or output_dimension % block_size:
        raise ValueError("block size must divide both weight dimensions")
    if tuple(input_hessian.shape) != (input_dimension, input_dimension):
        raise ValueError("input Hessian has the wrong shape")
    if tuple(output_hessian.shape) != (output_dimension, output_dimension):
        raise ValueError("output Hessian has the wrong shape")

    device = weight.device
    input_factor, input_reverse = _unit_block_inverse_cholesky_factor(
        input_hessian,
        block_size,
        dtype=work_dtype,
        device=device,
    )
    output_factor, output_reverse = _unit_block_inverse_cholesky_factor(
        output_hessian,
        block_size,
        dtype=work_dtype,
        device=device,
    )
    source = weight.to(dtype=work_dtype).T.contiguous()
    workspace = source.index_select(0, output_reverse).index_select(1, input_reverse).clone()
    quantized = torch.zeros_like(workspace)
    output_blocks = output_dimension // block_size
    input_blocks = input_dimension // block_size
    use_cuda_update = update_backend == "cuda" or (
        update_backend == "auto"
        and device.type == "cuda"
        and work_dtype == torch.float32
        and block_size == 16
    )
    if update_backend == "cuda" and not use_cuda_update:
        raise ValueError("CUDA BaKron updates require CUDA FP32 with 16x16 blocks")
    intermediate = torch.empty_like(workspace) if use_cuda_update else None
    payload_grid: torch.Tensor | None = None
    payload_shape: tuple[int, ...] | None = None

    diagonal_offsets = [0]
    flat_output_indices: list[int] = []
    flat_input_indices: list[int] = []
    for diagonal in range(output_blocks + input_blocks - 1):
        output_begin_block = max(0, diagonal - input_blocks + 1)
        output_end_block = min(output_blocks, diagonal + 1)
        for output_block in range(output_begin_block, output_end_block):
            flat_output_indices.append(output_block)
            flat_input_indices.append(diagonal - output_block)
        diagonal_offsets.append(len(flat_output_indices))
    diagonal_output_indices = torch.tensor(
        flat_output_indices, dtype=torch.long, device=device
    )
    diagonal_input_indices = torch.tensor(
        flat_input_indices, dtype=torch.long, device=device
    )
    diagonal_original_output_indices = output_blocks - 1 - diagonal_output_indices
    diagonal_original_input_indices = input_blocks - 1 - diagonal_input_indices
    workspace_blocks = _block_matrix_view(
        workspace, output_blocks, input_blocks, block_size
    )
    quantized_blocks = _block_matrix_view(
        quantized, output_blocks, input_blocks, block_size
    )

    def quantize_diagonal(diagonal: int) -> None:
        nonlocal payload_grid, payload_shape
        begin = diagonal_offsets[diagonal]
        end = diagonal_offsets[diagonal + 1]
        output_indices = diagonal_output_indices[begin:end]
        input_indices = diagonal_input_indices[begin:end]
        tile_batch = workspace_blocks[output_indices, input_indices].transpose(
            1, 2
        ).contiguous()
        reconstructed_tiles, payload = quantize_tiles(tile_batch)
        if reconstructed_tiles.shape != tile_batch.shape:
            raise ValueError("tile quantizer returned the wrong reconstruction shape")
        if payload.ndim < 1 or payload.shape[0] != tile_batch.shape[0]:
            raise ValueError("tile quantizer payload must have one leading item per tile")
        if payload_grid is None:
            payload_grid, payload_shape = _allocate_payload_grid(
                payload,
                input_blocks,
                output_blocks,
            )
        elif tuple(payload.shape[1:]) != payload_shape:
            raise ValueError("tile quantizer payload shape changed between calls")

        reconstructed_tiles = reconstructed_tiles.to(device=device, dtype=work_dtype)
        quantized_blocks[output_indices, input_indices] = reconstructed_tiles.transpose(
            1, 2
        )
        assert payload_grid is not None
        payload_grid[
            diagonal_original_input_indices[begin:end],
            diagonal_original_output_indices[begin:end],
        ] = payload

    def recurse(first_diagonal: int, last_diagonal: int) -> None:
        if last_diagonal == first_diagonal + 1:
            quantize_diagonal(first_diagonal)
            return
        middle_diagonal = (first_diagonal + last_diagonal) // 2
        recurse(first_diagonal, middle_diagonal)
        if use_cuda_update:
            from qsrt.sqg_quantizer import apply_bakron_block_update

            assert intermediate is not None
            apply_bakron_block_update(
                workspace,
                quantized,
                output_factor,
                input_factor,
                intermediate,
                first_diagonal=first_diagonal,
                middle_diagonal=middle_diagonal,
                last_diagonal=last_diagonal,
                block_size=block_size,
            )
        else:
            _apply_recursive_block_update(
                workspace,
                quantized,
                output_factor,
                input_factor,
                first_diagonal=first_diagonal,
                middle_diagonal=middle_diagonal,
                last_diagonal=last_diagonal,
                block_size=block_size,
                chunk_blocks=update_chunk_blocks,
            )
        recurse(middle_diagonal, last_diagonal)

    recurse(0, output_blocks + input_blocks - 1)
    assert payload_grid is not None
    if not bool(torch.all(torch.isfinite(quantized))):
        raise ValueError("tile quantizer returned a nonfinite reconstruction")
    reconstruction = quantized.index_select(0, output_reverse).index_select(1, input_reverse).T
    return TwoSidedRoundingResult(
        reconstruction=reconstruction.to(dtype=weight.dtype).contiguous(),
        payload=payload_grid,
    )


def bakron_block_round_encoder_batch(
    weights: torch.Tensor,
    input_hessians: torch.Tensor,
    output_hessians: torch.Tensor,
    quantize_tiles: TileQuantizer,
    *,
    block_size: int = 16,
    work_dtype: torch.dtype = torch.float32,
) -> TwoSidedRoundingResult:
    """Round independent matrices in lockstep through the CUDA recurrence."""

    if weights.ndim != 3 or not weights.shape[0] or not weights.is_floating_point():
        raise ValueError("weights must be a nonempty floating-point matrix batch")
    if weights.device.type != "cuda":
        raise ValueError("batched BaKron rounding requires CUDA")
    if work_dtype != torch.float32 or block_size != 16:
        raise ValueError("batched BaKron rounding requires FP32 and 16x16 blocks")
    batch, input_dimension, output_dimension = map(int, weights.shape)
    if input_dimension % block_size or output_dimension % block_size:
        raise ValueError("block size must divide both weight dimensions")
    if tuple(input_hessians.shape) != (batch, input_dimension, input_dimension):
        raise ValueError("input Hessians have the wrong shape")
    if tuple(output_hessians.shape) != (batch, output_dimension, output_dimension):
        raise ValueError("output Hessians have the wrong shape")
    if input_hessians.device != weights.device or output_hessians.device != weights.device:
        raise ValueError("weights and Hessians must share one CUDA device")
    if not bool(torch.all(torch.isfinite(weights))):
        raise ValueError("weights must be finite")

    device = weights.device
    # Batched cuSOLVER factorization is numerically close but not bit-identical
    # to the single-matrix path.  Tiny factor differences can select different
    # low-margin trellis paths, so retain the independently factored matrices
    # and batch only the rounding recurrence.
    input_pairs = [
        _unit_block_inverse_cholesky_factor(
            input_hessians[index],
            block_size,
            dtype=work_dtype,
            device=device,
        )
        for index in range(batch)
    ]
    output_pairs = [
        _unit_block_inverse_cholesky_factor(
            output_hessians[index],
            block_size,
            dtype=work_dtype,
            device=device,
        )
        for index in range(batch)
    ]
    input_factors = torch.stack([pair[0] for pair in input_pairs])
    output_factors = torch.stack([pair[0] for pair in output_pairs])
    input_reverse = input_pairs[0][1]
    output_reverse = output_pairs[0][1]
    source = weights.to(dtype=work_dtype).transpose(1, 2).contiguous()
    workspace = source.index_select(1, output_reverse).index_select(
        2, input_reverse
    ).clone()
    quantized = torch.zeros_like(workspace)
    intermediate = torch.empty_like(workspace)
    output_blocks = output_dimension // block_size
    input_blocks = input_dimension // block_size

    diagonal_offsets = [0]
    flat_output_indices: list[int] = []
    flat_input_indices: list[int] = []
    for diagonal in range(output_blocks + input_blocks - 1):
        output_begin_block = max(0, diagonal - input_blocks + 1)
        output_end_block = min(output_blocks, diagonal + 1)
        for output_block in range(output_begin_block, output_end_block):
            flat_output_indices.append(output_block)
            flat_input_indices.append(diagonal - output_block)
        diagonal_offsets.append(len(flat_output_indices))
    diagonal_output_indices = torch.tensor(
        flat_output_indices, dtype=torch.long, device=device
    )
    diagonal_input_indices = torch.tensor(
        flat_input_indices, dtype=torch.long, device=device
    )
    diagonal_original_output_indices = output_blocks - 1 - diagonal_output_indices
    diagonal_original_input_indices = input_blocks - 1 - diagonal_input_indices
    workspace_blocks = _block_matrix_batch_view(
        workspace, output_blocks, input_blocks, block_size
    )
    quantized_blocks = _block_matrix_batch_view(
        quantized, output_blocks, input_blocks, block_size
    )
    payload_grid: torch.Tensor | None = None
    payload_shape: tuple[int, ...] | None = None

    def quantize_diagonal(diagonal: int) -> None:
        nonlocal payload_grid, payload_shape
        begin = diagonal_offsets[diagonal]
        end = diagonal_offsets[diagonal + 1]
        output_indices = diagonal_output_indices[begin:end]
        input_indices = diagonal_input_indices[begin:end]
        diagonal_tiles = workspace_blocks[:, output_indices, input_indices]
        tile_batch = diagonal_tiles.transpose(2, 3).reshape(
            -1, block_size, block_size
        ).contiguous()
        reconstructed_tiles, payload = quantize_tiles(tile_batch)
        if reconstructed_tiles.shape != tile_batch.shape:
            raise ValueError("tile quantizer returned the wrong reconstruction shape")
        if payload.ndim < 1 or payload.shape[0] != tile_batch.shape[0]:
            raise ValueError("tile quantizer payload must have one leading item per tile")
        current_tiles = end - begin
        if payload_grid is None:
            payload_shape = tuple(payload.shape[1:])
            payload_grid = torch.empty(
                (batch, input_blocks, output_blocks, *payload_shape),
                dtype=payload.dtype,
                device=payload.device,
            )
        elif tuple(payload.shape[1:]) != payload_shape:
            raise ValueError("tile quantizer payload shape changed between calls")

        reconstructed = reconstructed_tiles.to(
            device=device, dtype=work_dtype
        ).reshape(batch, current_tiles, block_size, block_size)
        quantized_blocks[:, output_indices, input_indices] = reconstructed.transpose(
            2, 3
        )
        assert payload_grid is not None and payload_shape is not None
        payload_grid[
            :,
            diagonal_original_input_indices[begin:end],
            diagonal_original_output_indices[begin:end],
        ] = payload.reshape(batch, current_tiles, *payload_shape)

    def recurse(first_diagonal: int, last_diagonal: int) -> None:
        if last_diagonal == first_diagonal + 1:
            quantize_diagonal(first_diagonal)
            return
        middle_diagonal = (first_diagonal + last_diagonal) // 2
        recurse(first_diagonal, middle_diagonal)
        from qsrt.sqg_quantizer import apply_bakron_block_update

        apply_bakron_block_update(
            workspace,
            quantized,
            output_factors,
            input_factors,
            intermediate,
            first_diagonal=first_diagonal,
            middle_diagonal=middle_diagonal,
            last_diagonal=last_diagonal,
            block_size=block_size,
        )
        recurse(middle_diagonal, last_diagonal)

    recurse(0, output_blocks + input_blocks - 1)
    assert payload_grid is not None
    if not bool(torch.all(torch.isfinite(quantized))):
        raise ValueError("tile quantizer returned a nonfinite reconstruction")
    reconstruction = quantized.index_select(1, output_reverse).index_select(
        2, input_reverse
    ).transpose(1, 2)
    return TwoSidedRoundingResult(
        reconstruction=reconstruction.to(dtype=weights.dtype).contiguous(),
        payload=payload_grid,
    )


def bakron_block_round_encoder_prepared_batch(
    weights: torch.Tensor,
    input_factors: torch.Tensor,
    output_factors: torch.Tensor,
    input_reverse: torch.Tensor,
    output_reverse: torch.Tensor,
    quantize_tiles: TileQuantizer,
    *,
    block_size: int = 16,
    work_dtype: torch.dtype = torch.float32,
) -> TwoSidedRoundingResult:
    """Round a matrix batch with supplied BaKron factors.

    A rank-two factor is shared by every matrix.  A rank-three factor supplies
    one matrix per batch item.  This is the scalable path for independent
    output stripes that share one input Hessian.
    """

    if weights.ndim != 3 or not weights.shape[0] or not weights.is_floating_point():
        raise ValueError("weights must be a nonempty floating-point matrix batch")
    if weights.device.type != "cuda":
        raise ValueError("prepared batched BaKron rounding requires CUDA")
    if work_dtype != torch.float32 or block_size != 16:
        raise ValueError("prepared batched BaKron requires FP32 and 16x16 blocks")
    batch, input_dimension, output_dimension = map(int, weights.shape)
    if input_dimension % block_size or output_dimension % block_size:
        raise ValueError("block size must divide both weight dimensions")

    def validate_factor(
        factor: torch.Tensor,
        dimension: int,
        name: str,
    ) -> None:
        allowed = ((dimension, dimension), (batch, dimension, dimension))
        if tuple(factor.shape) not in allowed:
            raise ValueError(f"{name} has the wrong shape")
        if (
            factor.device != weights.device
            or factor.dtype != work_dtype
            or not factor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous CUDA FP32")
        if not bool(torch.all(torch.isfinite(factor))):
            raise ValueError(f"{name} must be finite")

    validate_factor(input_factors, input_dimension, "input factors")
    validate_factor(output_factors, output_dimension, "output factors")
    for reverse, dimension, name in (
        (input_reverse, input_dimension, "input reverse order"),
        (output_reverse, output_dimension, "output reverse order"),
    ):
        if (
            tuple(reverse.shape) != (dimension,)
            or reverse.device != weights.device
            or reverse.dtype != torch.long
        ):
            raise ValueError(f"{name} has incompatible geometry")
    if not bool(torch.all(torch.isfinite(weights))):
        raise ValueError("weights must be finite")

    device = weights.device
    source = weights.to(dtype=work_dtype).transpose(1, 2).contiguous()
    workspace = source.index_select(1, output_reverse).index_select(
        2, input_reverse
    ).clone()
    quantized = torch.zeros_like(workspace)
    intermediate = torch.empty_like(workspace)
    output_blocks = output_dimension // block_size
    input_blocks = input_dimension // block_size

    diagonal_offsets = [0]
    flat_output_indices: list[int] = []
    flat_input_indices: list[int] = []
    for diagonal in range(output_blocks + input_blocks - 1):
        output_begin_block = max(0, diagonal - input_blocks + 1)
        output_end_block = min(output_blocks, diagonal + 1)
        for output_block in range(output_begin_block, output_end_block):
            flat_output_indices.append(output_block)
            flat_input_indices.append(diagonal - output_block)
        diagonal_offsets.append(len(flat_output_indices))
    diagonal_output_indices = torch.tensor(
        flat_output_indices, dtype=torch.long, device=device
    )
    diagonal_input_indices = torch.tensor(
        flat_input_indices, dtype=torch.long, device=device
    )
    diagonal_original_output_indices = output_blocks - 1 - diagonal_output_indices
    diagonal_original_input_indices = input_blocks - 1 - diagonal_input_indices
    workspace_blocks = _block_matrix_batch_view(
        workspace, output_blocks, input_blocks, block_size
    )
    quantized_blocks = _block_matrix_batch_view(
        quantized, output_blocks, input_blocks, block_size
    )
    payload_grid: torch.Tensor | None = None
    payload_shape: tuple[int, ...] | None = None

    def quantize_diagonal(diagonal: int) -> None:
        nonlocal payload_grid, payload_shape
        begin = diagonal_offsets[diagonal]
        end = diagonal_offsets[diagonal + 1]
        output_indices = diagonal_output_indices[begin:end]
        input_indices = diagonal_input_indices[begin:end]
        diagonal_tiles = workspace_blocks[:, output_indices, input_indices]
        tile_batch = diagonal_tiles.transpose(2, 3).reshape(
            -1, block_size, block_size
        ).contiguous()
        reconstructed_tiles, payload = quantize_tiles(tile_batch)
        if reconstructed_tiles.shape != tile_batch.shape:
            raise ValueError("tile quantizer returned the wrong reconstruction shape")
        if payload.ndim < 1 or payload.shape[0] != tile_batch.shape[0]:
            raise ValueError("tile quantizer payload must have one leading item per tile")
        current_tiles = end - begin
        if payload_grid is None:
            payload_shape = tuple(payload.shape[1:])
            payload_grid = torch.empty(
                (batch, input_blocks, output_blocks, *payload_shape),
                dtype=payload.dtype,
                device=payload.device,
            )
        elif tuple(payload.shape[1:]) != payload_shape:
            raise ValueError("tile quantizer payload shape changed between calls")

        reconstructed = reconstructed_tiles.to(
            device=device, dtype=work_dtype
        ).reshape(batch, current_tiles, block_size, block_size)
        quantized_blocks[:, output_indices, input_indices] = reconstructed.transpose(
            2, 3
        )
        assert payload_grid is not None and payload_shape is not None
        payload_grid[
            :,
            diagonal_original_input_indices[begin:end],
            diagonal_original_output_indices[begin:end],
        ] = payload.reshape(batch, current_tiles, *payload_shape)

    def recurse(first_diagonal: int, last_diagonal: int) -> None:
        if last_diagonal == first_diagonal + 1:
            quantize_diagonal(first_diagonal)
            return
        middle_diagonal = (first_diagonal + last_diagonal) // 2
        recurse(first_diagonal, middle_diagonal)
        from qsrt.sqg_quantizer import apply_bakron_block_update

        apply_bakron_block_update(
            workspace,
            quantized,
            output_factors,
            input_factors,
            intermediate,
            first_diagonal=first_diagonal,
            middle_diagonal=middle_diagonal,
            last_diagonal=last_diagonal,
            block_size=block_size,
        )
        recurse(middle_diagonal, last_diagonal)

    recurse(0, output_blocks + input_blocks - 1)
    assert payload_grid is not None
    if not bool(torch.all(torch.isfinite(quantized))):
        raise ValueError("tile quantizer returned a nonfinite reconstruction")
    reconstruction = quantized.index_select(1, output_reverse).index_select(
        2, input_reverse
    ).transpose(1, 2)
    return TwoSidedRoundingResult(
        reconstruction=reconstruction.to(dtype=weights.dtype).contiguous(),
        payload=payload_grid,
    )


__all__ = [
    "TileQuantizer",
    "TwoSidedRoundingResult",
    "bakron_block_antidiagonal_encoder",
    "bakron_block_round_encoder",
    "bakron_block_round_encoder_batch",
    "bakron_block_round_encoder_prepared_batch",
    "factor_bakron_hessian",
    "transform_output_hessian_for_regularization",
    "transform_output_hessian_blocks_for_regularization",
    "yaqa_block_round_encoder",
]

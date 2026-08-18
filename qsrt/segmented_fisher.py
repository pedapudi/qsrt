"""CUDA accumulation of block Fisher factors from expert-sorted rows."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _hadamard_stage_128(
    values,
    row_tile: tl.constexpr,
    groups: tl.constexpr,
    width: tl.constexpr,
):
    paired = tl.reshape(values, (row_tile, groups, 2, width))
    paired = tl.permute(paired, (0, 1, 3, 2))
    left, right = tl.split(paired)
    joined = tl.join(left + right, left - right)
    joined = tl.permute(joined, (0, 1, 3, 2))
    return tl.reshape(joined, (row_tile, 128))


@triton.jit
def _coupled_hadamard_128_kernel(
    gate_up_pointer,
    sorted_experts_pointer,
    draws_pointer,
    signs_pointer,
    output_pointer,
    rows,
    intermediate_dimension: tl.constexpr,
    output_dimension: tl.constexpr,
    row_tile: tl.constexpr,
):
    row = tl.program_id(0) * row_tile + tl.arange(0, row_tile)
    block = tl.program_id(1)
    row_mask = row < rows
    pair = tl.arange(0, 64)
    lane = tl.arange(0, 128)
    expert = tl.load(sorted_experts_pointer + row, mask=row_mask, other=0)
    draw = tl.load(draws_pointer + expert, mask=row_mask, other=0)

    coordinate = block * 64 + pair
    source = row[:, None] * output_dimension + coordinate[None, :]
    gate = tl.load(
        gate_up_pointer + source,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        gate_up_pointer + source + intermediate_dimension,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    interleaved = tl.reshape(tl.join(gate, up), (row_tile, 128))
    sign_offset = (
        draw[:, None] * output_dimension + block * 128 + lane[None, :]
    )
    values = interleaved * tl.load(
        signs_pointer + sign_offset,
        mask=row_mask[:, None],
        other=1.0,
    )
    values = _hadamard_stage_128(values, row_tile, 64, 1)
    values = _hadamard_stage_128(values, row_tile, 32, 2)
    values = _hadamard_stage_128(values, row_tile, 16, 4)
    values = _hadamard_stage_128(values, row_tile, 8, 8)
    values = _hadamard_stage_128(values, row_tile, 4, 16)
    values = _hadamard_stage_128(values, row_tile, 2, 32)
    values = _hadamard_stage_128(values, row_tile, 1, 64)
    destination = (
        row[:, None] * output_dimension + block * 128 + lane[None, :]
    )
    tl.store(
        output_pointer + destination,
        values,
        mask=row_mask[:, None],
    )


@triton.jit
def _segmented_fisher_128_kernel(
    rows_pointer,
    offsets_pointer,
    sums_pointer,
    blocks: tl.constexpr,
    block_size: tl.constexpr,
    output_tile: tl.constexpr,
    row_tile: tl.constexpr,
):
    expert = tl.program_id(0)
    block = tl.program_id(1)
    triangle = tl.program_id(2)

    tile_i = tl.where(
        triangle < 4,
        0,
        tl.where(triangle < 7, 1, tl.where(triangle < 9, 2, 3)),
    )
    tile_j = tl.where(
        triangle < 4,
        triangle,
        tl.where(triangle < 7, triangle - 3, tl.where(triangle < 9, triangle - 5, 3)),
    )
    begin = tl.load(offsets_pointer + expert - 1, mask=expert > 0, other=0)
    end = tl.load(offsets_pointer + expert)
    output_i = tile_i * output_tile + tl.arange(0, output_tile)
    output_j = tile_j * output_tile + tl.arange(0, output_tile)
    accumulator = tl.zeros((output_tile, output_tile), dtype=tl.float32)
    cursor = begin
    while cursor < end:
        row = cursor + tl.arange(0, row_tile)
        row_mask = row < end
        base = (row[:, None] * blocks + block) * block_size
        left = tl.load(
            rows_pointer + base + output_i[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        right = tl.load(
            rows_pointer + base + output_j[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        accumulator += tl.dot(tl.trans(left), right, input_precision="ieee")
        cursor += row_tile

    factor_base = (expert * blocks + block) * block_size * block_size
    pointer = (
        sums_pointer
        + factor_base
        + output_i[:, None] * block_size
        + output_j[None, :]
    )
    tl.store(pointer, tl.load(pointer) + accumulator)
    mirror = (
        sums_pointer
        + factor_base
        + output_j[:, None] * block_size
        + output_i[None, :]
    )
    tl.store(
        mirror,
        tl.load(mirror) + tl.trans(accumulator),
        mask=tile_i != tile_j,
    )


def add_segmented_fisher_128_(
    sums: torch.Tensor,
    rows: torch.Tensor,
    offsets: torch.Tensor,
) -> None:
    """Add exact FP32 covariance blocks for expert-sorted transformed rows."""

    if sums.device.type != "cuda" or rows.device != sums.device or offsets.device != sums.device:
        raise ValueError("segmented Fisher tensors must share one CUDA device")
    if sums.dtype != torch.float32 or rows.dtype != torch.float32:
        raise TypeError("segmented Fisher values must be FP32")
    if offsets.dtype != torch.int32:
        raise TypeError("segmented Fisher offsets must be int32")
    if sums.ndim != 4 or rows.ndim != 3 or offsets.ndim != 1:
        raise ValueError("segmented Fisher tensors have incompatible ranks")
    experts, blocks, block_size, trailing = map(int, sums.shape)
    if block_size != 128 or trailing != block_size:
        raise ValueError("segmented Fisher kernel requires 128x128 factors")
    if tuple(rows.shape[1:]) != (blocks, block_size) or offsets.numel() != experts:
        raise ValueError("segmented Fisher geometry does not close")
    if not sums.is_contiguous() or not rows.is_contiguous() or not offsets.is_contiguous():
        raise ValueError("segmented Fisher tensors must be contiguous")
    _segmented_fisher_128_kernel[(experts, blocks, 10)](
        rows,
        offsets,
        sums,
        blocks=blocks,
        block_size=block_size,
        output_tile=32,
        row_tile=16,
        num_warps=8,
    )


def transform_coupled_preactivation_128(
    gate_up_gradient: torch.Tensor,
    sorted_experts: torch.Tensor,
    draws: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    """Apply the stored coupled sign and normalized H128 transform."""

    if gate_up_gradient.device.type != "cuda":
        raise ValueError("coupled preactivation transform requires CUDA")
    device = gate_up_gradient.device
    if (
        sorted_experts.device != device
        or draws.device != device
        or signs.device != device
    ):
        raise ValueError("coupled preactivation tensors must share one CUDA device")
    if gate_up_gradient.ndim != 2 or gate_up_gradient.shape[1] % 256:
        raise ValueError("gate/up width must contain two equal H128-aligned halves")
    rows, output_dimension = map(int, gate_up_gradient.shape)
    intermediate_dimension = output_dimension // 2
    if sorted_experts.shape != (rows,):
        raise ValueError("sorted expert indices do not cover the routed rows")
    if draws.ndim != 1 or signs.shape != (8, output_dimension):
        raise ValueError("draw and sign tables have incompatible geometry")
    if sorted_experts.dtype not in (torch.int32, torch.int64):
        raise TypeError("sorted expert indices must be integral")
    if draws.dtype not in (torch.int32, torch.int64):
        raise TypeError("coupled draws must be integral")
    if signs.dtype != torch.float32:
        raise TypeError("coupled signs must be FP32")
    if not all(
        value.is_contiguous()
        for value in (gate_up_gradient, sorted_experts, draws, signs)
    ):
        raise ValueError("coupled preactivation tensors must be contiguous")

    output = torch.empty_like(gate_up_gradient, dtype=torch.float32)
    row_tile = 8
    _coupled_hadamard_128_kernel[
        (triton.cdiv(rows, row_tile), output_dimension // 128)
    ](
        gate_up_gradient,
        sorted_experts,
        draws,
        signs,
        output,
        rows,
        intermediate_dimension=intermediate_dimension,
        output_dimension=output_dimension,
        row_tile=row_tile,
        num_warps=8,
    )
    output.div_(math.sqrt(128))
    return output


__all__ = ["add_segmented_fisher_128_", "transform_coupled_preactivation_128"]

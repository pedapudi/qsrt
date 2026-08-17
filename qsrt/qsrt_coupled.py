"""Exact coupled activation-boundary coordinates for QSRT expert matrices.

The transform interleaves gate and up rows before a normalized block-Hadamard
rotation. The decoder joins the two stored projections, cancels that rotation,
evaluates the coordinatewise SiTU activation, and rotates the resulting hidden
coordinates into the matching down-projection basis. Residual-side transforms
are layer-shared; the intermediate draw is expert-static.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from qsrt.tp_simulator import situ


Tensor = torch.Tensor
WeightTriplet = tuple[Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class CoupledHadamardSpec:
    """Immutable transform parameters carried by a coupled QSRT profile."""

    residual_block_size: int = 512
    preactivation_block_size: int = 128
    postactivation_block_size: int = 128
    residual_draw: int = 0
    intermediate_draw: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("residual_block_size", self.residual_block_size),
            ("preactivation_block_size", self.preactivation_block_size),
            ("postactivation_block_size", self.postactivation_block_size),
        ):
            if value <= 0 or value & (value - 1):
                raise ValueError(f"{name} must be a positive power of two")
        if self.residual_draw != 0:
            raise ValueError("the coupled profile fixes residual draw zero")
        if not 0 <= self.intermediate_draw < 8:
            raise ValueError("intermediate_draw must lie in 0..7")


def block_hadamard(values: Tensor, *, block_size: int, dim: int = -1) -> Tensor:
    """Apply a normalized self-inverse block Walsh-Hadamard transform."""

    if values.ndim == 0 or not torch.is_floating_point(values):
        raise TypeError("block Hadamard input must be floating point")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    axis = dim % values.ndim
    if values.shape[axis] % block_size:
        raise ValueError("Hadamard axis must be divisible by block_size")
    # Naturally routed experts can have an empty fit or confirmation fold.
    # The transform is linear, so its action on an empty batch is the empty
    # float32 tensor with the same shape.  Returning before the butterfly
    # reshape also avoids an ambiguous ``-1`` inference when numel is zero.
    if values.numel() == 0:
        return values.float().contiguous().clone()
    output = values.float().movedim(axis, -1).contiguous().clone()
    shape = output.shape
    output = output.reshape(*shape[:-1], shape[-1] // block_size, block_size)
    width = 1
    while width < block_size:
        paired = output.reshape(*output.shape[:-1], -1, 2, width)
        left = paired[..., 0, :].clone()
        right = paired[..., 1, :].clone()
        paired[..., 0, :] = left + right
        paired[..., 1, :] = left - right
        output = paired.reshape(*output.shape)
        width *= 2
    return (
        output.div_(math.sqrt(block_size))
        .reshape(shape)
        .movedim(-1, axis)
        .contiguous()
    )


def rotation_signs(
    length: int, *, draw: int, axis: int, device: torch.device
) -> Tensor:
    """Return one deterministic Rademacher vector from the frozen draw family."""

    if length <= 0 or draw < 0:
        raise ValueError("length must be positive and draw nonnegative")
    if draw == 0:
        return torch.ones(length, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        (0x6A09E667F3BCC909 * draw + 0xBB67AE8584CAA73B * axis)
        & ((1 << 63) - 1)
    )
    return (
        torch.randint(0, 2, (length,), generator=generator)
        .mul_(2)
        .sub_(1)
        .float()
        .to(device=device)
    )


def signed_block_hadamard(
    values: Tensor,
    *,
    block_size: int,
    signs: Tensor,
    dim: int = -1,
    inverse: bool = False,
) -> Tensor:
    """Apply ``D H`` to row vectors, or its exact inverse ``H D``."""

    axis = dim % values.ndim
    if signs.ndim != 1 or signs.numel() != values.shape[axis]:
        raise ValueError("rotation signs do not align with the transformed axis")
    shape = [1] * values.ndim
    shape[axis] = signs.numel()
    expanded = signs.to(device=values.device, dtype=torch.float32).reshape(shape)
    if inverse:
        return block_hadamard(values, block_size=block_size, dim=axis) * expanded
    return block_hadamard(
        values.float() * expanded, block_size=block_size, dim=axis
    )


def _validate_triplet(weights: WeightTriplet) -> tuple[int, int]:
    w1, w3, w2 = weights
    if w1.ndim != 2 or w3.ndim != 2 or w2.ndim != 2 or w1.shape != w3.shape:
        raise ValueError("coupled weights require matching two-dimensional w1/w3")
    intermediate, hidden = w1.shape
    if tuple(w2.shape) != (hidden, intermediate):
        raise ValueError("w2 must have shape [hidden, intermediate]")
    if not all(torch.is_floating_point(value) for value in weights):
        raise TypeError("coupled weights must be floating point")
    if len({value.device for value in weights}) != 1:
        raise ValueError("coupled weights must share one device")
    return intermediate, hidden


def encode_coupled_weights(
    weights: WeightTriplet, spec: CoupledHadamardSpec
) -> WeightTriplet:
    """Reparameterize one full-precision expert into coupled coordinates."""

    w1, w3, w2 = weights
    intermediate, hidden = _validate_triplet(weights)
    if hidden % spec.residual_block_size:
        raise ValueError("hidden size does not close residual Hadamard blocks")
    if 2 * intermediate % spec.preactivation_block_size:
        raise ValueError("interleaved size does not close preactivation blocks")
    if intermediate % spec.postactivation_block_size:
        raise ValueError("intermediate size does not close postactivation blocks")
    device = w1.device
    input_signs = rotation_signs(hidden, draw=spec.residual_draw, axis=0, device=device)
    pre_signs = rotation_signs(
        2 * intermediate, draw=spec.intermediate_draw, axis=1, device=device
    )
    post_signs = rotation_signs(
        intermediate, draw=spec.intermediate_draw, axis=2, device=device
    )
    output_signs = rotation_signs(
        hidden, draw=spec.residual_draw, axis=3, device=device
    )
    interleaved = torch.stack((w1, w3), dim=1).reshape(2 * intermediate, hidden)
    upstream = signed_block_hadamard(
        signed_block_hadamard(
            interleaved,
            block_size=spec.residual_block_size,
            signs=input_signs,
            dim=1,
        ),
        block_size=spec.preactivation_block_size,
        signs=pre_signs,
        dim=0,
    )
    down = signed_block_hadamard(
        signed_block_hadamard(
            w2,
            block_size=spec.postactivation_block_size,
            signs=post_signs,
            dim=1,
        ),
        block_size=spec.residual_block_size,
        signs=output_signs,
        dim=0,
    )
    return (
        upstream[:intermediate].contiguous(),
        upstream[intermediate:].contiguous(),
        down.contiguous(),
    )


@dataclass(frozen=True)
class CoupledHadamardExecution:
    """Calibration and scoring operations for one transformed expert basis."""

    hidden: int
    intermediate: int
    spec: CoupledHadamardSpec

    def _signs(self, axis: int, *, device: torch.device) -> Tensor:
        lengths = (self.hidden, 2 * self.intermediate, self.intermediate, self.hidden)
        draw = self.spec.residual_draw if axis in (0, 3) else self.spec.intermediate_draw
        return rotation_signs(lengths[axis], draw=draw, axis=axis, device=device)

    def transform_inputs(self, rows: Tensor) -> Tensor:
        if rows.ndim != 2 or rows.shape[1] != self.hidden:
            raise ValueError("input rows do not match the hidden dimension")
        return signed_block_hadamard(
            rows,
            block_size=self.spec.residual_block_size,
            signs=self._signs(0, device=rows.device),
            dim=1,
        )

    def transform_h13(self, hessian: Tensor) -> Tensor:
        if tuple(hessian.shape) != (self.hidden, self.hidden):
            raise ValueError("H13 does not match the hidden dimension")
        signs = self._signs(0, device=hessian.device)
        return signed_block_hadamard(
            signed_block_hadamard(
                hessian,
                block_size=self.spec.residual_block_size,
                signs=signs,
                dim=0,
            ),
            block_size=self.spec.residual_block_size,
            signs=signs,
            dim=1,
        )

    def transform_h2(self, hessian: Tensor) -> Tensor:
        if tuple(hessian.shape) != (self.intermediate, self.intermediate):
            raise ValueError("H2 does not match the intermediate dimension")
        signs = self._signs(2, device=hessian.device)
        return signed_block_hadamard(
            signed_block_hadamard(
                hessian,
                block_size=self.spec.postactivation_block_size,
                signs=signs,
                dim=0,
            ),
            block_size=self.spec.postactivation_block_size,
            signs=signs,
            dim=1,
        )

    def decode_middle(self, rows: Tensor, w1: Tensor, w3: Tensor) -> Tensor:
        transformed = torch.cat((F.linear(rows, w1), F.linear(rows, w3)), dim=1)
        recovered = signed_block_hadamard(
            transformed,
            block_size=self.spec.preactivation_block_size,
            signs=self._signs(1, device=rows.device),
            dim=1,
            inverse=True,
        )
        middle = situ(recovered[:, 0::2], recovered[:, 1::2])
        return signed_block_hadamard(
            middle,
            block_size=self.spec.postactivation_block_size,
            signs=self._signs(2, device=rows.device),
            dim=1,
        )

    def decode_output(self, transformed: Tensor) -> Tensor:
        if transformed.ndim != 2 or transformed.shape[1] != self.hidden:
            raise ValueError("output rows do not match the hidden dimension")
        return signed_block_hadamard(
            transformed,
            block_size=self.spec.residual_block_size,
            signs=self._signs(3, device=transformed.device),
            dim=1,
            inverse=True,
        )

    def execute(self, rows: Tensor, weights: WeightTriplet) -> Tensor:
        _validate_triplet(weights)
        transformed_rows = self.transform_inputs(rows)
        middle = self.decode_middle(transformed_rows, weights[0], weights[1])
        return self.decode_output(F.linear(middle, weights[2]))


def coupled_execution(
    weights: WeightTriplet, spec: CoupledHadamardSpec
) -> CoupledHadamardExecution:
    intermediate, hidden = _validate_triplet(weights)
    return CoupledHadamardExecution(hidden, intermediate, spec)


__all__ = [
    "CoupledHadamardExecution",
    "CoupledHadamardSpec",
    "WeightTriplet",
    "block_hadamard",
    "coupled_execution",
    "encode_coupled_weights",
    "rotation_signs",
    "signed_block_hadamard",
]

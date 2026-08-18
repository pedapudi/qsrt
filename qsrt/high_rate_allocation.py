"""Equal-byte K2/K3/K4 allocation for Kimi-K3 expert matrices.

Every 16-channel strip contains 24 intermediate-axis records.  A 3.083333-bpw
strip carries 74 trellis bits.  Record-contiguous schedules and tile-local
P33/P24 decisions below preserve that bit count exactly; selector metadata is
reported separately from trellis payload bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from qsrt.qsrt import (
    CONTEXT_GROUP_CHANNELS,
    INTERMEDIATE_CHANNELS,
    RECORDS_PER_EXPERT,
    TILES_PER_RECORD_AXIS,
    expand_group_order,
)
from qsrt.qsrt_candidates import rank_contexts


HIGH_RATE_STRIP_BITS = 74
FIXED_K4_RECORDS = 2
TRANSFERABLE_RECORD_PAIRS = (RECORDS_PER_EXPERT - FIXED_K4_RECORDS) // 2


@dataclass(frozen=True)
class TileRateAllocation:
    """One flattened matrix tile-rate map and its selector cost."""

    tile_bits: tuple[int, ...]
    selected_p24_tiles: int
    candidate_p24_tiles: int
    selector_bytes: int

    @property
    def selected_fraction(self) -> float:
        return self.selected_p24_tiles / self.candidate_p24_tiles


@dataclass(frozen=True)
class RecordRateAllocation:
    """Best record-contiguous 74-bit schedule under tile cost surfaces."""

    donor_records: int
    tile_bits: tuple[int, ...]
    proxy_cost: float
    costs_by_donor_records: tuple[float, ...]


def tile_squared_error(
    reference: torch.Tensor,
    reconstruction: torch.Tensor,
) -> torch.Tensor:
    """Return one squared-error cost per 16-by-16 encoder tile."""

    if reference.ndim != 2 or reconstruction.shape != reference.shape:
        raise ValueError("tile error inputs must be aligned matrices")
    if reference.shape[0] % 16 or reference.shape[1] % 16:
        raise ValueError("tile error inputs must be 16-by-16 aligned")
    if not all(torch.is_floating_point(value) for value in (reference, reconstruction)):
        raise TypeError("tile error inputs must be floating point")
    error = reconstruction.float() - reference.float()
    return (
        error.reshape(reference.shape[0] // 16, 16, reference.shape[1] // 16, 16)
        .square()
        .sum(dim=(1, 3), dtype=torch.float64)
        .contiguous()
    )


def dense_h_tile_error_contributions(
    reference: torch.Tensor,
    reconstruction: torch.Tensor,
    hessian: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
) -> torch.Tensor:
    """Partition the complete dense-H quadratic error over encoder tiles.

    ``reference`` and ``reconstruction`` use the regularized ``[K, N]``
    coordinates consumed by the trellis encoder.  ``hessian`` is the dense
    input covariance before the encoder's scale and block-Hadamard
    conditioning.  The returned tile contributions sum to the exact
    ``trace(error_original.T @ hessian @ error_original)`` objective.

    Dense covariance makes individual contributions signed because cross-tile
    terms are assigned through the quadratic form's gradient.  The complete
    sum is nonnegative for a positive-semidefinite Hessian.
    """

    if reference.ndim != 2 or reconstruction.shape != reference.shape:
        raise ValueError("dense-H tile inputs must be aligned matrices")
    if reference.shape[0] % 128 or reference.shape[1] % 128:
        raise ValueError("dense-H tile inputs must be 128-by-128 aligned")
    if hessian.shape != (reference.shape[0], reference.shape[0]):
        raise ValueError("dense-H input covariance does not match the K dimension")
    if suh.shape != (reference.shape[0],) or svh.shape != (reference.shape[1],):
        raise ValueError("dense-H scale vectors do not match the encoder matrix")
    values = (reference, reconstruction, hessian, suh, svh)
    if not all(torch.is_floating_point(value) for value in values):
        raise TypeError("dense-H tile inputs must be floating point")
    if not all(bool(torch.all(torch.isfinite(value))) for value in values):
        raise ValueError("dense-H tile inputs must be finite")
    if bool(torch.any(suh == 0)) or bool(torch.any(svh == 0)):
        raise ValueError("dense-H scale vectors must be nonzero")

    error = reconstruction.float() - reference.float()
    device = error.device
    h = normalized_hadamard_128(device=device)

    def left_hadamard(value: torch.Tensor) -> torch.Tensor:
        blocks = value.reshape(-1, 128, value.shape[1])
        return torch.matmul(h, blocks).reshape_as(value)

    def right_hadamard(value: torch.Tensor) -> torch.Tensor:
        blocks = value.reshape(value.shape[0], -1, 128)
        return torch.matmul(blocks, h).reshape_as(value)

    # Undo regularization to obtain the physical encoder-coordinate error.
    physical = left_hadamard(error)
    physical = physical * suh.float().unsqueeze(1)
    physical = right_hadamard(physical)
    physical = physical * svh.float().unsqueeze(0)

    # Apply the adjoint of the same transform to H @ error.  The elementwise
    # product with the regularized error is an exact additive decomposition of
    # the full quadratic form.
    gradient = hessian.float() @ physical
    gradient = gradient * svh.float().unsqueeze(0)
    gradient = right_hadamard(gradient)
    gradient = gradient * suh.float().unsqueeze(1)
    gradient = left_hadamard(gradient)
    contributions = error * gradient
    return (
        contributions.reshape(
            reference.shape[0] // 16,
            16,
            reference.shape[1] // 16,
            16,
        )
        .sum(dim=(1, 3), dtype=torch.float64)
        .contiguous()
    )


def normalized_hadamard_128(*, device: torch.device) -> torch.Tensor:
    """Return the normalized 128-point Walsh-Hadamard matrix."""

    result = torch.ones((1, 1), dtype=torch.float32, device=device)
    while result.shape[0] < 128:
        result = torch.cat(
            (
                torch.cat((result, result), dim=1),
                torch.cat((result, -result), dim=1),
            ),
            dim=0,
        )
    return result * (1.0 / (128.0**0.5))


def neuron_permutation_from_scores(
    scores: torch.Tensor,
    *,
    policy: str,
) -> torch.Tensor:
    """Return an exact-gauge neuron order for rate-transfer experiments.

    Scores cover consecutive four-neuron groups.  Every policy keeps these
    groups intact so a common gate/up row and down-column permutation remains
    an exact coordinatewise-activation gauge.
    """

    expected = INTERMEDIATE_CHANNELS // CONTEXT_GROUP_CHANNELS
    if scores.ndim != 1 or scores.numel() != expected:
        raise ValueError(f"neuron scores must contain {expected} four-neuron groups")
    if not torch.is_floating_point(scores) or not bool(torch.all(torch.isfinite(scores))):
        raise ValueError("neuron scores must be finite floating point")
    if policy == "identity":
        group_order = torch.arange(expected, device=scores.device)
    elif policy == "importance":
        group_order = torch.argsort(scores, stable=True)
    elif policy in ("energy_balanced", "stratified_energy_balanced"):
        strata = 1 if policy == "energy_balanced" else 6
        contexts = rank_contexts(scores, strata)
        groups_per_stratum = expected // strata
        records_per_stratum = RECORDS_PER_EXPERT // strata
        groups_per_record = groups_per_stratum // records_per_stratum
        pieces: list[torch.Tensor] = []
        for stratum in range(strata):
            members = torch.nonzero(contexts == stratum).flatten()
            members = members.index_select(
                0, torch.argsort(scores.index_select(0, members), descending=True, stable=True)
            )
            totals = [0.0] * records_per_stratum
            assignments: list[list[int]] = [[] for _ in range(records_per_stratum)]
            for group in members.tolist():
                choices = [
                    record
                    for record in range(records_per_stratum)
                    if len(assignments[record]) < groups_per_record
                ]
                record = min(choices, key=lambda value: (totals[value], value))
                assignments[record].append(group)
                totals[record] += float(scores[group])
            record_order = sorted(
                range(records_per_stratum), key=lambda value: (totals[value], value)
            )
            for record in record_order:
                pieces.append(
                    torch.tensor(
                        sorted(assignments[record], key=lambda value: (float(scores[value]), value)),
                        dtype=torch.long,
                        device=scores.device,
                    )
                )
        group_order = torch.cat(pieces)
    else:
        raise ValueError(f"unsupported neuron permutation policy: {policy}")
    if not torch.equal(torch.sort(group_order).values, torch.arange(expected, device=scores.device)):
        raise AssertionError("neuron permutation policy is not bijective")
    return expand_group_order(group_order)


def high_rate_record_bits(donor_records: int) -> tuple[int, ...]:
    """Return ``N×K2, (22-2N)×K3, (N+2)×K4`` at 74 bits."""

    if not 0 <= donor_records <= TRANSFERABLE_RECORD_PAIRS:
        raise ValueError(
            f"donor records must lie in 0..{TRANSFERABLE_RECORD_PAIRS}"
        )
    result = (
        (2,) * donor_records
        + (3,) * (22 - 2 * donor_records)
        + (4,) * (donor_records + FIXED_K4_RECORDS)
    )
    if len(result) != RECORDS_PER_EXPERT or sum(result) != HIGH_RATE_STRIP_BITS:
        raise AssertionError("high-rate record schedule violated its bit budget")
    return result


def record_rate_map(
    shape: tuple[int, int],
    *,
    rate_axis: str,
    donor_records: int,
) -> tuple[int, ...]:
    """Expand one record-contiguous high-rate schedule over a matrix tile grid."""

    tiles_k, tiles_n = (dimension // 16 for dimension in shape)
    if rate_axis not in ("k", "n"):
        raise ValueError("rate axis must be 'k' or 'n'")
    rate_tiles = tiles_k if rate_axis == "k" else tiles_n
    orthogonal_tiles = tiles_n if rate_axis == "k" else tiles_k
    if rate_tiles != RECORDS_PER_EXPERT * TILES_PER_RECORD_AXIS:
        raise ValueError("matrix does not contain Kimi-K3's 24 rate records")
    rates = torch.tensor(high_rate_record_bits(donor_records), dtype=torch.int8)
    rates = rates.repeat_interleave(TILES_PER_RECORD_AXIS)
    rate_map = (
        rates[None, :].repeat(orthogonal_tiles, 1)
        if rate_axis == "n"
        else rates[:, None].repeat(1, orthogonal_tiles)
    )
    return tuple(int(value) for value in rate_map.flatten())


def rate_map_proxy_cost(
    cost_surfaces: Sequence[Mapping[int, torch.Tensor]],
    tile_bits: Sequence[int],
) -> torch.Tensor:
    """Sum exact-basis tile costs selected by one complete rate map."""

    if not cost_surfaces:
        raise ValueError("rate-map scoring requires at least one cost surface")
    shape = tuple(cost_surfaces[0][3].shape)
    if any(
        set(surface) != {2, 3, 4}
        or any(tuple(surface[bits].shape) != shape for bits in (2, 3, 4))
        for surface in cost_surfaces
    ):
        raise ValueError("K2/K3/K4 cost surfaces must share one shape")
    if len(tile_bits) != shape[0] * shape[1] or any(
        isinstance(bits, bool) or bits not in (2, 3, 4) for bits in tile_bits
    ):
        raise ValueError("rate map does not cover the tile grid with K2/K3/K4")
    combined = {
        bits: sum(
            (surface[bits].detach().double() for surface in cost_surfaces),
            start=torch.zeros_like(cost_surfaces[0][bits], dtype=torch.float64),
        ).flatten()
        for bits in (2, 3, 4)
    }
    selected = torch.tensor(
        tuple(int(bits) for bits in tile_bits),
        dtype=torch.int64,
        device=combined[3].device,
    )
    values = torch.stack(tuple(combined[bits] for bits in (2, 3, 4)), dim=0)
    return values.gather(0, (selected - 2).unsqueeze(0)).sum()


def select_record_rate_allocation(
    cost_surfaces: Sequence[Mapping[int, torch.Tensor]],
    *,
    shape: tuple[int, int],
    rate_axis: str,
) -> RecordRateAllocation:
    """Select among all ``N K2, 22-2N K3, N+2 K4`` schedules."""

    maps = tuple(
        record_rate_map(shape, rate_axis=rate_axis, donor_records=donors)
        for donors in range(TRANSFERABLE_RECORD_PAIRS + 1)
    )
    costs = tuple(
        float(rate_map_proxy_cost(cost_surfaces, tile_bits)) for tile_bits in maps
    )
    winner = min(range(len(costs)), key=lambda index: (costs[index], index))
    return RecordRateAllocation(
        donor_records=winner,
        tile_bits=maps[winner],
        proxy_cost=costs[winner],
        costs_by_donor_records=costs,
    )


def _as_strips(value: torch.Tensor, *, rate_axis: str) -> torch.Tensor:
    if value.ndim != 2:
        raise ValueError("tile cost surfaces must be matrices")
    tiles_k, tiles_n = value.shape
    rate_tiles = tiles_k if rate_axis == "k" else tiles_n
    orthogonal_tiles = tiles_n if rate_axis == "k" else tiles_k
    if rate_tiles != RECORDS_PER_EXPERT * TILES_PER_RECORD_AXIS:
        raise ValueError("tile costs do not contain 24 rate records")
    if rate_axis == "n":
        return value.reshape(
            orthogonal_tiles, RECORDS_PER_EXPERT, TILES_PER_RECORD_AXIS
        ).permute(0, 2, 1).reshape(-1, RECORDS_PER_EXPERT)
    return value.reshape(
        RECORDS_PER_EXPERT, TILES_PER_RECORD_AXIS, orthogonal_tiles
    ).permute(2, 1, 0).reshape(-1, RECORDS_PER_EXPERT)


def _from_strips(
    strips: torch.Tensor,
    *,
    rate_axis: str,
    shape: tuple[int, int],
) -> torch.Tensor:
    tiles_k, tiles_n = shape
    orthogonal_tiles = tiles_n if rate_axis == "k" else tiles_k
    if tuple(strips.shape) != (
        orthogonal_tiles * TILES_PER_RECORD_AXIS,
        RECORDS_PER_EXPERT,
    ):
        raise ValueError("strip rates have the wrong matrix geometry")
    if rate_axis == "n":
        return strips.reshape(
            orthogonal_tiles, TILES_PER_RECORD_AXIS, RECORDS_PER_EXPERT
        ).permute(0, 2, 1).reshape(shape)
    return strips.reshape(
        orthogonal_tiles, TILES_PER_RECORD_AXIS, RECORDS_PER_EXPERT
    ).permute(2, 1, 0).reshape(shape)


def tile_p24_allocation(
    cost_surfaces: Sequence[Mapping[int, torch.Tensor]],
    *,
    rate_axis: str,
    force_fraction: float | None = None,
) -> TileRateAllocation:
    """Choose P33 or P24 independently for every transferable tile pair.

    Records 22 and 23 remain K4.  The other records are paired as ``(0, 21)``,
    ``(1, 20)``, ..., ``(10, 11)``.  Each P33-to-P24 change replaces ``3+3``
    with ``2+4`` and therefore leaves the 74-bit strip payload unchanged.
    Gate and up may supply two cost surfaces while sharing one selected map.
    """

    if rate_axis not in ("k", "n"):
        raise ValueError("rate axis must be 'k' or 'n'")
    if not cost_surfaces:
        raise ValueError("tile allocation requires at least one cost surface")
    if force_fraction is not None and not 0 <= force_fraction <= 1:
        raise ValueError("forced P24 fraction must lie in [0, 1]")
    shape = tuple(cost_surfaces[0][3].shape)
    if any(
        set(surface) != {2, 3, 4}
        or any(tuple(surface[bits].shape) != shape for bits in (2, 3, 4))
        for surface in cost_surfaces
    ):
        raise ValueError("K2/K3/K4 cost surfaces must share one shape")
    costs = {
        bits: sum(
            (surface[bits].detach().double() for surface in cost_surfaces),
            start=torch.zeros_like(cost_surfaces[0][bits], dtype=torch.float64),
        )
        for bits in (2, 3, 4)
    }
    strips = {bits: _as_strips(value, rate_axis=rate_axis) for bits, value in costs.items()}
    strip_count = strips[3].shape[0]
    rates = torch.full_like(strips[3], 3, dtype=torch.int8)
    rates[:, 22:24] = 4

    deltas = []
    for low in range(TRANSFERABLE_RECORD_PAIRS):
        high = 21 - low
        deltas.append(
            strips[2][:, low]
            + strips[4][:, high]
            - strips[3][:, low]
            - strips[3][:, high]
        )
    delta = torch.stack(deltas, dim=1)
    if force_fraction is None:
        selected = delta < 0
    else:
        count = int(round(delta.numel() * force_fraction))
        selected = torch.zeros_like(delta, dtype=torch.bool)
        if count:
            order = torch.argsort(delta.flatten(), stable=True)
            selected.flatten()[order[:count]] = True
    for low in range(TRANSFERABLE_RECORD_PAIRS):
        high = 21 - low
        rates[:, low] = torch.where(selected[:, low], 2, rates[:, low])
        rates[:, high] = torch.where(selected[:, low], 4, rates[:, high])
    if not bool(torch.all(rates.sum(dim=1) == HIGH_RATE_STRIP_BITS)):
        raise AssertionError("tile allocation violated the strip bit budget")
    rate_map = _from_strips(rates, rate_axis=rate_axis, shape=shape)
    candidate_tiles = strip_count * TRANSFERABLE_RECORD_PAIRS
    # One direct bit per transferable pair.  Storage rounds to whole bytes.
    selector_bytes = (candidate_tiles + 7) // 8
    return TileRateAllocation(
        tile_bits=tuple(int(value) for value in rate_map.flatten().cpu()),
        selected_p24_tiles=int(torch.count_nonzero(selected)),
        candidate_p24_tiles=candidate_tiles,
        selector_bytes=selector_bytes,
    )


def triplet_tile_selector_bytes(*, upstream_shared: bool = True) -> int:
    """Return direct bitmap bytes for one Kimi expert's three matrices."""

    strips = 224 * TILES_PER_RECORD_AXIS
    one_map = (strips * TRANSFERABLE_RECORD_PAIRS + 7) // 8
    return one_map * (2 if upstream_shared else 3)


__all__ = [
    "FIXED_K4_RECORDS",
    "HIGH_RATE_STRIP_BITS",
    "TRANSFERABLE_RECORD_PAIRS",
    "TileRateAllocation",
    "RecordRateAllocation",
    "high_rate_record_bits",
    "record_rate_map",
    "rate_map_proxy_cost",
    "select_record_rate_allocation",
    "tile_p24_allocation",
    "dense_h_tile_error_contributions",
    "tile_squared_error",
    "neuron_permutation_from_scores",
    "triplet_tile_selector_bytes",
]

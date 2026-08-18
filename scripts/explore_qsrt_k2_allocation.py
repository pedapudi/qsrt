"""Evaluate fixed-payload K1/K2/K3 rate allocation for QSRT experts.

The allocation screen encodes K1/K2/K3 candidates without cross-tile LDLQ
feedback, then scores their decoded errors under the production dense Hessian.
Every proposed mixed-rate schedule is encoded again from the canonical source
with complete BlockLDLQ feedback.  Routed expert-output measurements are
diagnostic; they do not select or reject a schedule.  This offline numerical
experiment does not define a serialized QSRT format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F

from qsrt.capture import LayerSamples, index_cached_layer_samples, load_layer_hessians
from qsrt.coupled_expert_study import (
    CoupledTriplet,
    apply_permutation_sign_gauge,
    block_hadamard,
    encode_coupled_block_hadamard,
    execute_coupled_block_hadamard,
    hadamard_rotation_signs,
    signed_block_hadamard,
)
from qsrt.exl3_loader import load_qsrt_encoder
from qsrt.exl3_reference import (
    CODEBOOK_SQG_XOR_CHEB_T12,
    _blockwise_hadamard_left,
    _blockwise_hadamard_right,
    decode_regularized_weight,
    decode_qsrt_regularized_weight,
)
from qsrt.high_rate_allocation import dense_h_tile_error_contributions
from qsrt.ldlq import SIGMA_REG, make_shared_h
from qsrt.pack.qsrt_encoder import plan_qsrt_matrix
from qsrt.qsrt import RATE_TRANSFER_MODES, expand_group_order
from qsrt.qsrt_candidates import (
    PERMUTATION_POLICIES,
    PermutationPolicy,
    build_expert_hessians,
    functional_sse_by_request,
    partition_requests,
    permutation_policy_contexts,
    request_documents,
    select_expert_rows,
)
from qsrt.source_weights import OfficialMXFP4Store
from qsrt.sqg_e4m3 import (
    sqg_cheb_normal_rank_e4m3_bytes,
    sqg_xor_cheb_t12_bytes,
    sqg_xor_cheb_t12_rank_lut_bytes,
    sqg_xor_rank_permutation,
)
from qsrt.sqg_quantizer import install_sqg_quantizer
from qsrt.tp_simulator import situ


MATRICES = ("w1", "w3", "w2")
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
TRIPLE_MODES = (
    (3, 3, 3),
    (2, 3, 4),
    (2, 4, 3),
    (3, 2, 4),
    (3, 4, 2),
    (4, 2, 3),
    (4, 3, 2),
)
TWO_BIT_PROJECTION_MODES = (
    (2, 2, 2),
    (1, 2, 3),
    (1, 3, 2),
    (2, 1, 3),
    (2, 3, 1),
    (3, 1, 2),
    (3, 2, 1),
)
OUTER_SLOPES = {
    "q31": 31.0 / 32.0,
    "normal": 1.0,
    "q33": 33.0 / 32.0,
    "q17": 17.0 / 16.0,
}
K2_MENU_LAWS = ("normal", "q31", "q33", "q17")
SCALE_CLOSURE_MULTIPLIERS = (0.94, 0.97, 1.0, 1.03, 1.06)
PermutationChoice = Literal[
    "identity",
    "h2_reverse",
    "energy_balanced",
    "stratified_energy_balanced",
    "h2_exact",
    "functional_exact",
    "h2_tile_balanced",
    "functional_tile_balanced",
    "h2_shape_clustered",
    "h2_global_shape_clustered",
    "functional_global_shape_clustered",
    "h2_priority_shape_clustered",
    "functional_priority_shape_clustered",
    "h2_rate_response_clustered",
    "h2_p24_band_aligned",
    "h2_top2_band_aligned",
]

MiddleDecoder = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]
TripletExecutor = Callable[
    [
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    torch.Tensor,
]
InputTransform = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class CoupledSearchBasis:
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    h13: torch.Tensor
    h2: torch.Tensor
    inputs: torch.Tensor
    permutation: torch.Tensor
    reference_output: torch.Tensor
    transform_inputs: InputTransform
    decode_middle: MiddleDecoder
    execute_triplet: TripletExecutor
    evidence: dict[str, object]


PERMUTATION_CHOICES: tuple[PermutationChoice, ...] = (
    *PERMUTATION_POLICIES,
    "h2_exact",
    "functional_exact",
    "h2_tile_balanced",
    "functional_tile_balanced",
    "h2_shape_clustered",
    "h2_global_shape_clustered",
    "functional_global_shape_clustered",
    "h2_priority_shape_clustered",
    "functional_priority_shape_clustered",
    "h2_rate_response_clustered",
    "h2_p24_band_aligned",
    "h2_top2_band_aligned",
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} does not contain a JSON object")
    return value


def _atomic_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parse_ints(value: str, *, minimum: int, maximum: int) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("expected unique comma-separated integers")
    if any(item < minimum or item > maximum for item in result):
        raise argparse.ArgumentTypeError(
            f"values must lie in {minimum}..{maximum}"
        )
    return result


def _parse_expert_draws(value: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        fields = item.split(":")
        if len(fields) != 2 or not all(field.isdecimal() for field in fields):
            raise argparse.ArgumentTypeError(
                "expected comma-separated expert:draw pairs"
            )
        expert, draw = map(int, fields)
        if expert in result:
            raise argparse.ArgumentTypeError("expert draw keys must be unique")
        result[expert] = draw
    return result


def _request_mask(
    request_steps: torch.Tensor, requests: Mapping[int, str]
) -> torch.Tensor:
    allowed = torch.tensor(
        sorted(map(int, requests)), dtype=request_steps.dtype, device=request_steps.device
    )
    return torch.isin(request_steps, allowed)


def _encoder_coordinates(
    source: torch.Tensor,
    hessian: torch.Tensor,
    contexts: torch.Tensor,
    *,
    matrix: str,
    permutation_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if permutation_override is None:
        plan = plan_qsrt_matrix(
            contexts,
            RATE_TRANSFER_MODES[0],
            matrix=matrix,
            layout="importance_ordered",
        )
        permutation = plan.encoder_permutation
    else:
        permutation = permutation_override.to(device=source.device, dtype=torch.long)
        expected = source.shape[0 if matrix != "w2" else 1]
        if permutation.ndim != 1 or permutation.numel() != expected:
            raise ValueError("permutation override has the wrong shape")
        if not torch.equal(
            torch.sort(permutation).values,
            torch.arange(permutation.numel(), device=permutation.device),
        ):
            raise ValueError("permutation override is not bijective")
    if matrix == "w2":
        weight = source.index_select(1, permutation).T.contiguous()
        hp = permutation.to(device=hessian.device)
        encoder_hessian = hessian.index_select(0, hp).index_select(1, hp)
    else:
        weight = source.index_select(0, permutation).T.contiguous()
        encoder_hessian = hessian
    return weight, encoder_hessian, permutation


def _permutation_sha256(permutation: torch.Tensor) -> str:
    return hashlib.sha256(
        permutation.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _canonical_reconstruction(
    encoder_weight: torch.Tensor,
    source: torch.Tensor,
    permutation: torch.Tensor,
    *,
    matrix: str,
) -> torch.Tensor:
    result = torch.empty_like(source)
    if matrix == "w2":
        result[:, permutation] = encoder_weight.T
    else:
        result[permutation] = encoder_weight.T
    return result.contiguous()


def _tile_view(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or weight.shape[0] % 16 or weight.shape[1] % 16:
        raise ValueError("tile view requires a 16x16-aligned matrix")
    tiles_k = weight.shape[0] // 16
    tiles_n = weight.shape[1] // 16
    return (
        weight.reshape(tiles_k, 16, tiles_n, 16)
        .permute(0, 2, 1, 3)
        .reshape(tiles_k, tiles_n, 256)
    )


def _target_regularized_weight(
    encoder_weight: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor
) -> torch.Tensor:
    target = encoder_weight / svh.float().unsqueeze(0)
    target = _blockwise_hadamard_right(target)
    target /= suh.float().unsqueeze(1)
    return _blockwise_hadamard_left(target).contiguous()


def _uniform_tile_map(shape: tuple[int, int], bits: int) -> tuple[int, ...]:
    return (bits,) * ((shape[0] // 16) * (shape[1] // 16))


def _paired_tile_map(
    shape: tuple[int, int],
    selected: Sequence[int],
    *,
    rate_axis: str,
) -> tuple[int, ...]:
    """Build one equal-byte P33/P24 map over the outer record pair.

    The canonical coordinate is ``orthogonal_tile * 8 + record_tile``.
    This gives the same 1,792 coordinates for gate/up and transposed down
    matrices, allowing one experimental decision to cover the corresponding
    tiles in all three projections.
    """

    if rate_axis not in ("k", "n"):
        raise ValueError("paired tile map needs rate_axis 'k' or 'n'")
    tiles_k, tiles_n = shape[0] // 16, shape[1] // 16
    rate_tiles = tiles_k if rate_axis == "k" else tiles_n
    orthogonal_tiles = tiles_n if rate_axis == "k" else tiles_k
    if rate_tiles != 192 or orthogonal_tiles != 224:
        raise ValueError("Kimi outer tile pair expects a 192x224 tile grid")
    pair_count = orthogonal_tiles * 8
    selected_tensor = torch.as_tensor(tuple(selected), dtype=torch.long)
    if selected_tensor.numel() and (
        int(selected_tensor.min()) < 0 or int(selected_tensor.max()) >= pair_count
    ):
        raise ValueError("paired tile coordinate is out of range")
    selected_mask = torch.zeros((orthogonal_tiles, 8), dtype=torch.bool)
    if selected_tensor.numel():
        selected_mask.flatten()[selected_tensor] = True

    rate_map = torch.full((tiles_k, tiles_n), 3, dtype=torch.int8)
    low = torch.arange(8)
    high = torch.arange(rate_tiles - 8, rate_tiles)
    if rate_axis == "n":
        rate_map[:, low] = torch.where(
            selected_mask, torch.tensor(2, dtype=torch.int8), rate_map[:, low]
        )
        rate_map[:, high] = torch.where(
            selected_mask, torch.tensor(4, dtype=torch.int8), rate_map[:, high]
        )
    else:
        transposed = selected_mask.T
        rate_map[low, :] = torch.where(
            transposed, torch.tensor(2, dtype=torch.int8), rate_map[low, :]
        )
        rate_map[high, :] = torch.where(
            transposed, torch.tensor(4, dtype=torch.int8), rate_map[high, :]
        )
    return tuple(int(value) for value in rate_map.flatten().tolist())


def _qsrt_308_record_map(
    shape: tuple[int, int],
    donor_records: int,
    *,
    rate_axis: str,
) -> tuple[int, ...]:
    """Build a monotone 74-bit schedule over Kimi's 24 records.

    After the common neuron permutation, records are ordered from low to high
    priority.  ``donor_records`` records at the low end use K2, two more than
    that at the high end use K4, and the middle uses K3.  Thus every
    coefficient strip consumes exactly 74 bits across 24 records::

        N * 2 + (22 - 2N) * 3 + (N + 2) * 4 = 74.

    This is equivalent to one P44 pair, ``N`` P24 pairs, and ``11-N`` P33
    pairs.  It is a record-granular baseline for the tile-funded search below.
    """

    if rate_axis not in ("k", "n"):
        raise ValueError("3.08-bpw map needs rate_axis 'k' or 'n'")
    if not 0 <= donor_records <= 11:
        raise ValueError("3.08-bpw donor count must lie in 0..11")
    tiles_k, tiles_n = shape[0] // 16, shape[1] // 16
    rate_tiles = tiles_k if rate_axis == "k" else tiles_n
    if rate_tiles != 192:
        raise ValueError("Kimi 3.08-bpw maps require 24 128-channel records")
    rates = (
        (2,) * (8 * donor_records)
        + (3,) * (8 * (22 - 2 * donor_records))
        + (4,) * (8 * (donor_records + 2))
    )
    if len(rates) != rate_tiles or sum(rates) != 8 * 74:
        raise AssertionError("invalid 3.08-bpw record schedule")
    if rate_axis == "n":
        rate_map = torch.tensor(rates, dtype=torch.int8).repeat(tiles_k, 1)
    else:
        rate_map = torch.tensor(rates, dtype=torch.int8)[:, None].repeat(1, tiles_n)
    return tuple(int(value) for value in rate_map.flatten().tolist())


def _p13_record_map(
    shape: tuple[int, int],
    donor_records: int,
    *,
    rate_axis: str,
) -> tuple[int, ...]:
    """Build an exact-average-two-bit K1/K2/K3 record schedule.

    Records are ordered from low to high priority after the shared neuron
    permutation.  ``donor_records`` records at the low end use K1, the same
    number at the high end use K3, and every remaining record uses K2.  Every
    coefficient strip therefore consumes exactly 48 bits across 24 records.
    """

    if rate_axis not in ("k", "n"):
        raise ValueError("P13 map needs rate_axis 'k' or 'n'")
    if not 0 <= donor_records <= 12:
        raise ValueError("P13 donor count must lie in 0..12")
    tiles_k, tiles_n = shape[0] // 16, shape[1] // 16
    rate_tiles = tiles_k if rate_axis == "k" else tiles_n
    if rate_tiles != 192:
        raise ValueError("Kimi P13 maps require 24 128-channel records")
    rates = (
        (1,) * (8 * donor_records)
        + (2,) * (8 * (24 - 2 * donor_records))
        + (3,) * (8 * donor_records)
    )
    if len(rates) != rate_tiles or sum(rates) != 8 * 48:
        raise AssertionError("invalid P13 record schedule")
    if rate_axis == "n":
        rate_map = torch.tensor(rates, dtype=torch.int8).repeat(tiles_k, 1)
    else:
        rate_map = torch.tensor(rates, dtype=torch.int8)[:, None].repeat(1, tiles_n)
    return tuple(int(value) for value in rate_map.flatten().tolist())


def _p13_record_map_from_rates(
    shape: tuple[int, int],
    record_rates: Sequence[int],
    *,
    rate_axis: str,
) -> tuple[int, ...]:
    """Expand 24 record rates into a K1/K2/K3 tile map."""

    return _p13_channel_group_map_from_rates(
        shape,
        record_rates,
        rate_axis=rate_axis,
        channels_per_group=128,
    )


def _p13_channel_group_map_from_rates(
    shape: tuple[int, int],
    group_rates: Sequence[int],
    *,
    rate_axis: str,
    channels_per_group: int,
) -> tuple[int, ...]:
    """Expand balanced channel-group rates into a K1/K2/K3 tile map."""

    if rate_axis not in ("k", "n"):
        raise ValueError("P13 channel-group map needs rate_axis 'k' or 'n'")
    if channels_per_group <= 0 or channels_per_group % 16:
        raise ValueError("P13 channel groups must contain whole 16-channel tiles")
    rates = tuple(int(rate) for rate in group_rates)
    expected_groups = 3072 // channels_per_group
    if len(rates) != expected_groups or any(rate not in (1, 2, 3) for rate in rates):
        raise ValueError("P13 channel-group rates have the wrong geometry")
    if rates.count(1) != rates.count(3):
        raise ValueError("P13 channel-group rates must balance K1 and K3")
    if sum(rates) != 2 * expected_groups:
        raise ValueError("P13 channel-group rates must retain a two-bit mean")
    tiles_k, tiles_n = shape[0] // 16, shape[1] // 16
    rate_tiles = tiles_k if rate_axis == "k" else tiles_n
    if rate_tiles != 192:
        raise ValueError("Kimi P13 maps require 24 128-channel records")
    group_tiles = channels_per_group // 16
    tile_rates = torch.tensor(rates, dtype=torch.int8).repeat_interleave(
        group_tiles
    )
    if tile_rates.numel() != rate_tiles:
        raise AssertionError("P13 channel-group rates do not cover the rate axis")
    if rate_axis == "n":
        rate_map = tile_rates.repeat(tiles_k, 1)
    else:
        rate_map = tile_rates[:, None].repeat(1, tiles_n)
    return tuple(int(value) for value in rate_map.flatten().tolist())


def _w2_record_traversal_variant(
    prepared: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rate_map: tuple[int, ...],
    *,
    policy: str,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[int, ...],
    dict[str, object],
]:
    """Reorder intact W2 H128 records while preserving channel rate labels.

    BlockLDLQ traverses encoder rows in reverse.  Moving complete 128-channel
    records preserves the H128 conditioning groups and isolates traversal
    order from tile membership.  The tile-rate rows travel with their original
    channels, so every canonical W2 coefficient retains its assigned rate.
    """

    if policy not in ("baseline", "reverse", "donor_first", "recipient_first"):
        raise ValueError(f"unsupported W2 traversal policy: {policy}")
    weight, hessian, permutation = prepared
    if weight.shape[0] != 3072 or hessian.shape != (3072, 3072):
        raise ValueError("W2 traversal variants require 3,072 encoder rows")
    if permutation.shape != (3072,):
        raise ValueError("W2 traversal permutation has the wrong shape")
    rates = torch.tensor(rate_map, dtype=torch.int8).reshape(192, 224)
    record_mean_rates = rates.reshape(24, 8, 224).double().mean(dim=(1, 2))
    if policy == "baseline":
        record_order = list(range(24))
    elif policy == "reverse":
        record_order = list(range(23, -1, -1))
    elif policy == "donor_first":
        # Low-rate donor records occupy the high encoder indices visited first.
        record_order = sorted(
            range(24),
            key=lambda record: (-float(record_mean_rates[record]), record),
        )
    else:
        # High-rate recipient records occupy the high encoder indices visited first.
        record_order = sorted(
            range(24),
            key=lambda record: (float(record_mean_rates[record]), record),
        )

    row_order = torch.cat(
        [
            torch.arange(
                record * 128,
                (record + 1) * 128,
                device=weight.device,
            )
            for record in record_order
        ]
    )
    hessian_order = row_order.to(device=hessian.device)
    permutation_order = row_order.to(device=permutation.device)
    tile_order = torch.cat(
        [
            torch.arange(record * 8, (record + 1) * 8)
            for record in record_order
        ]
    )
    reordered_rates = rates.index_select(0, tile_order).contiguous()
    reordered = (
        weight.index_select(0, row_order).contiguous(),
        hessian.index_select(0, hessian_order)
        .index_select(1, hessian_order)
        .contiguous(),
        permutation.index_select(0, permutation_order).contiguous(),
    )
    return reordered, tuple(int(value) for value in reordered_rates.flatten()), {
        "policy": policy,
        "encoder_record_order": record_order,
        "blockldlq_visit_order": list(reversed(record_order)),
        "record_mean_rates": [float(value) for value in record_mean_rates],
        "preserves_h128_groups": True,
        "preserves_canonical_channel_rates": True,
    }


def _balanced_p13_record_rates(
    donor_delta: torch.Tensor,
    recipient_delta: torch.Tensor,
) -> tuple[dict[str, tuple[int, ...]], dict[str, object]]:
    """Solve every equal-count K1/K3 record allocation exactly.

    ``donor_delta`` is the cost of changing one record from K2 to K1;
    ``recipient_delta`` is the cost of changing one record from K2 to K3.
    The dynamic program enforces disjoint donor and recipient sets.
    """

    donor = donor_delta.detach().double().cpu().flatten()
    recipient = recipient_delta.detach().double().cpu().flatten()
    if donor.shape != (24,) or recipient.shape != (24,):
        raise ValueError("P13 record allocation requires 24 donor and recipient costs")
    if not bool(torch.all(torch.isfinite(donor))) or not bool(
        torch.all(torch.isfinite(recipient))
    ):
        raise ValueError("P13 record costs must be finite")

    # State values are (cost, record-rate prefix).  The state space is only
    # 25 * 13 * 13, so retaining the exact deterministic path is inexpensive.
    states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {
        (0, 0): (0.0, ())
    }
    for index in range(24):
        updated: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
        for (donors, recipients), (cost, path) in states.items():
            choices = ((donors, recipients, 2, 0.0),)
            if donors < 12:
                choices += ((donors + 1, recipients, 1, float(donor[index])),)
            if recipients < 12:
                choices += (
                    (
                        donors,
                        recipients + 1,
                        3,
                        float(recipient[index]),
                    ),
                )
            for next_donors, next_recipients, rate, delta in choices:
                key = (next_donors, next_recipients)
                candidate = (cost + delta, (*path, rate))
                current = updated.get(key)
                if current is None or candidate < current:
                    updated[key] = candidate
        states = updated

    maps: dict[str, tuple[int, ...]] = {}
    allocations: dict[str, object] = {}
    for count in range(13):
        cost, rates = states[(count, count)]
        maps[f"n{count}"] = rates
        allocations[f"n{count}"] = {
            "proxy_delta": cost,
            "k1_records": [index for index, rate in enumerate(rates) if rate == 1],
            "k3_records": [index for index, rate in enumerate(rates) if rate == 3],
        }
    return maps, {
        "allocator": "exact_disjoint_record_dynamic_program",
        "donor_delta": [float(value) for value in donor],
        "recipient_delta": [float(value) for value in recipient],
        "allocations": allocations,
    }


def _balanced_p13_unit_rates(
    donor_delta: torch.Tensor,
    recipient_delta: torch.Tensor,
    *,
    max_count: int | None = None,
) -> dict[int, dict[str, object]]:
    """Solve the minimum-cost disjoint allocation at every requested count."""

    donor = donor_delta.detach().double().cpu().flatten()
    recipient = recipient_delta.detach().double().cpu().flatten()
    if donor.shape != recipient.shape or donor.numel() < 2:
        raise ValueError("P13 unit costs must have equal nontrivial length")
    if not bool(torch.all(torch.isfinite(donor))) or not bool(
        torch.all(torch.isfinite(recipient))
    ):
        raise ValueError("P13 unit costs must be finite")

    limit = donor.numel() // 2
    if max_count is not None:
        if not 0 <= max_count <= limit:
            raise ValueError("P13 maximum unit count is out of range")
        limit = max_count
    states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {
        (0, 0): (0.0, ())
    }
    for index in range(donor.numel()):
        updated: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
        for (donors, recipients), (cost, path) in states.items():
            choices = ((donors, recipients, 2, 0.0),)
            if donors < limit:
                choices += ((donors + 1, recipients, 1, float(donor[index])),)
            if recipients < limit:
                choices += (
                    (
                        donors,
                        recipients + 1,
                        3,
                        float(recipient[index]),
                    ),
                )
            for next_donors, next_recipients, rate, delta in choices:
                key = (next_donors, next_recipients)
                candidate = (cost + delta, (*path, rate))
                current = updated.get(key)
                if current is None or candidate < current:
                    updated[key] = candidate
        states = updated

    return {
        count: {
            "allocator": "exact_disjoint_unit_dynamic_program_at_fixed_count",
            "units": int(donor.numel()),
            "k1_count": count,
            "k3_count": count,
            "delta": states[(count, count)][0],
            "gain": -states[(count, count)][0],
            "k1_units": [
                index
                for index, rate in enumerate(states[(count, count)][1])
                if rate == 1
            ],
            "k3_units": [
                index
                for index, rate in enumerate(states[(count, count)][1])
                if rate == 3
            ],
            "rates": list(states[(count, count)][1]),
        }
        for count in range(limit + 1)
    }


def _best_balanced_p13_units(
    donor_delta: torch.Tensor,
    recipient_delta: torch.Tensor,
) -> dict[str, object]:
    """Solve an unconstrained exact-average-two-bit allocation."""

    by_count = _balanced_p13_unit_rates(donor_delta, recipient_delta)
    count, result = min(
        (
            (count, candidate)
            for count, candidate in by_count.items()
        ),
        key=lambda item: (
            float(item[1]["delta"]),
            item[0],
            tuple(item[1]["rates"]),
        ),
    )
    return {
        "allocator": "exact_unconstrained_disjoint_unit_dynamic_program",
        "units": result["units"],
        "k1_count": count,
        "k3_count": count,
        "delta": result["delta"],
        "gain": result["gain"],
        "k1_units": result["k1_units"],
        "k3_units": result["k3_units"],
        "rates": result["rates"],
    }


def _qsrt_308_boundary_tile_map(
    shape: tuple[int, int],
    donor_records: int,
    selected: Sequence[int],
    *,
    rate_axis: str,
) -> tuple[int, ...]:
    """Partially fund the next P24 pair above a 3.08 record schedule.

    ``donor_records`` complete P24 record pairs are already active.  The next
    pair is ``(donor_records, 21-donor_records)`` because records 22 and 23
    form the fixed P44 tail.  Each selected coordinate changes exactly one
    low/high 16x16 P33 pair into P24, retaining 74 bits in every strip.
    """

    if not 0 <= donor_records < 11:
        raise ValueError("boundary tile funding requires 0..10 donor records")
    if rate_axis not in ("k", "n"):
        raise ValueError("boundary tile map needs rate_axis 'k' or 'n'")
    tiles_k, tiles_n = shape[0] // 16, shape[1] // 16
    rate_tiles = tiles_k if rate_axis == "k" else tiles_n
    orthogonal_tiles = tiles_n if rate_axis == "k" else tiles_k
    if rate_tiles != 192 or orthogonal_tiles != 224:
        raise ValueError("Kimi boundary tile maps require a 192x224 tile grid")
    selected_tensor = torch.as_tensor(tuple(selected), dtype=torch.long)
    pair_count = orthogonal_tiles * 8
    if selected_tensor.numel() and (
        int(selected_tensor.min()) < 0 or int(selected_tensor.max()) >= pair_count
    ):
        raise ValueError("boundary tile coordinate is out of range")
    selected_mask = torch.zeros((orthogonal_tiles, 8), dtype=torch.bool)
    if selected_tensor.numel():
        selected_mask.flatten()[selected_tensor] = True

    rate_map = torch.tensor(
        _qsrt_308_record_map(shape, donor_records, rate_axis=rate_axis),
        dtype=torch.int8,
    ).reshape(tiles_k, tiles_n)
    low = donor_records * 8 + torch.arange(8)
    high = (21 - donor_records) * 8 + torch.arange(8)
    if rate_axis == "n":
        rate_map[:, low] = torch.where(
            selected_mask, torch.tensor(2, dtype=torch.int8), rate_map[:, low]
        )
        rate_map[:, high] = torch.where(
            selected_mask, torch.tensor(4, dtype=torch.int8), rate_map[:, high]
        )
        strip_sums = rate_map.reshape(orthogonal_tiles, 24, 8).sum(dim=1)
    else:
        transposed = selected_mask.T
        rate_map[low, :] = torch.where(
            transposed, torch.tensor(2, dtype=torch.int8), rate_map[low, :]
        )
        rate_map[high, :] = torch.where(
            transposed, torch.tensor(4, dtype=torch.int8), rate_map[high, :]
        )
        strip_sums = rate_map.reshape(24, 8, orthogonal_tiles).sum(dim=0).T
    if not bool(torch.all(strip_sums == 74)):
        raise AssertionError("boundary tile map violated its local 74-bit budget")
    return tuple(int(value) for value in rate_map.flatten().tolist())


def _qsrt_308_boundary_proxy_deltas(
    error_sets: Sequence[Mapping[int, torch.Tensor]],
    donor_records: int,
    *,
    rate_axis: str,
) -> torch.Tensor:
    """Return equal-byte P24-minus-P33 proxy distortion per boundary tile."""

    if not 0 <= donor_records < 11:
        raise ValueError("boundary proxy needs 0..10 donor records")
    if rate_axis not in ("k", "n"):
        raise ValueError("boundary proxy needs rate_axis 'k' or 'n'")
    if not error_sets:
        raise ValueError("boundary proxy needs at least one error surface")
    shape = error_sets[0][3].shape
    if any(errors[3].shape != shape for errors in error_sets):
        raise ValueError("boundary proxy error surfaces do not align")
    low_record = donor_records
    high_record = 21 - donor_records
    by_subrecord: list[torch.Tensor] = []
    for subrecord_tile in range(8):
        low = low_record * 8 + subrecord_tile
        high = high_record * 8 + subrecord_tile
        benefit = None
        for errors in error_sets:
            local = (
                errors[3][:, low]
                + errors[3][:, high]
                - errors[2][:, low]
                - errors[4][:, high]
                if rate_axis == "n"
                else errors[3][low, :]
                + errors[3][high, :]
                - errors[2][low, :]
                - errors[4][high, :]
            )
            benefit = local if benefit is None else benefit + local
        assert benefit is not None
        by_subrecord.append(benefit.detach().double())
    # Boundary bitmap coordinates are orthogonal_tile * 8 + subrecord_tile.
    benefits = torch.stack(by_subrecord, dim=1)
    if benefits.shape != (224, 8):
        raise AssertionError("boundary proxy produced the wrong coordinate grid")
    return -benefits.flatten().cpu()


def _qsrt_308_tile_funded_map(
    error_sets: Sequence[Mapping[int, torch.Tensor]],
    *,
    rate_axis: str,
    fraction: float | None = None,
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Select P33/P24 independently while retaining an exact 74-bit strip.

    Records 22 and 23 are the fixed P44 pair.  The remaining ordered records
    are paired as ``(0, 21), (1, 20), ... (10, 11)``.  Every corresponding
    16x16 tile pair independently chooses P33 or P24 using the summed
    regularized-weight proxy in ``error_sets``.  Each choice is byte-neutral,
    so every orthogonal-tile/subrecord strip remains exactly 74 bits.
    """

    if rate_axis not in ("k", "n"):
        raise ValueError("tile-funded map needs rate_axis 'k' or 'n'")
    if fraction is not None and not 0.0 <= fraction <= 1.0:
        raise ValueError("tile-funded fraction must lie in [0, 1]")
    if not error_sets:
        raise ValueError("tile-funded map needs at least one error surface")
    shape = error_sets[0][3].shape
    if any(errors[3].shape != shape for errors in error_sets):
        raise ValueError("tile-funded error surfaces must align")
    tiles_k, tiles_n = shape
    rate_tiles = tiles_k if rate_axis == "k" else tiles_n
    orthogonal_tiles = tiles_n if rate_axis == "k" else tiles_k
    if rate_tiles != 192 or orthogonal_tiles != 224:
        raise ValueError("Kimi tile-funded maps require a 192x224 tile grid")

    costs = {
        bits: sum(
            (errors[bits] for errors in error_sets),
            start=torch.zeros_like(error_sets[0][bits]),
        )
        for bits in (2, 3, 4)
    }
    map_device = costs[3].device
    rate_map = torch.full(shape, 3, dtype=torch.int8, device=map_device)
    if rate_axis == "n":
        rate_map[:, 22 * 8 : 24 * 8] = 4
    else:
        rate_map[22 * 8 : 24 * 8, :] = 4

    benefits: list[torch.Tensor] = []
    for low_record in range(11):
        high_record = 21 - low_record
        for subrecord_tile in range(8):
            low = low_record * 8 + subrecord_tile
            high = high_record * 8 + subrecord_tile
            if rate_axis == "n":
                benefit = (
                    costs[3][:, low]
                    + costs[3][:, high]
                    - costs[2][:, low]
                    - costs[4][:, high]
                )
            else:
                benefit = (
                    costs[3][low, :]
                    + costs[3][high, :]
                    - costs[2][low, :]
                    - costs[4][high, :]
                )
            benefits.append(benefit.detach().double())

    flat_benefits_device = torch.cat(benefits)
    if fraction is None:
        selected_mask = flat_benefits_device > 0
    else:
        selected_count = int(round(flat_benefits_device.numel() * fraction))
        selected_mask = torch.zeros_like(flat_benefits_device, dtype=torch.bool)
        if selected_count:
            order = torch.argsort(flat_benefits_device, descending=True, stable=True)
            selected_mask[order[:selected_count]] = True

    offset = 0
    for low_record in range(11):
        high_record = 21 - low_record
        for subrecord_tile in range(8):
            low = low_record * 8 + subrecord_tile
            high = high_record * 8 + subrecord_tile
            choose = selected_mask[offset : offset + orthogonal_tiles]
            offset += orthogonal_tiles
            if rate_axis == "n":
                rate_map[:, low] = torch.where(
                    choose,
                    torch.tensor(2, dtype=torch.int8, device=map_device),
                    rate_map[:, low],
                )
                rate_map[:, high] = torch.where(
                    choose,
                    torch.tensor(4, dtype=torch.int8, device=map_device),
                    rate_map[:, high],
                )
            else:
                rate_map[low, :] = torch.where(
                    choose,
                    torch.tensor(2, dtype=torch.int8, device=map_device),
                    rate_map[low, :],
                )
                rate_map[high, :] = torch.where(
                    choose,
                    torch.tensor(4, dtype=torch.int8, device=map_device),
                    rate_map[high, :],
                )

    strip_sums = (
        rate_map.reshape(orthogonal_tiles, 24, 8).sum(dim=1)
        if rate_axis == "n"
        else rate_map.reshape(24, 8, orthogonal_tiles).sum(dim=0).T
    )
    if not bool(torch.all(strip_sums == 74)):
        raise AssertionError("tile-funded map violated its local 74-bit budget")
    flat_benefits = flat_benefits_device.cpu()
    selected = int(torch.count_nonzero(selected_mask))
    return (
        tuple(int(value) for value in rate_map.flatten().cpu().tolist()),
        {
            "fixed_p44_records": [22, 23],
            "p24_candidate_record_pairs": [
                [low, 21 - low] for low in range(11)
            ],
            "tile_pair_decisions": 11 * 8 * orthogonal_tiles,
            "selected_p24_tiles": selected,
            "selected_p24_fraction": selected / (11 * 8 * orthogonal_tiles),
            "selection_fraction": fraction,
            "proxy_benefit_min": float(flat_benefits.min()),
            "proxy_benefit_median": float(flat_benefits.median()),
            "proxy_benefit_max": float(flat_benefits.max()),
        },
    )


def _qsrt_308_top2_k4_map(
    error_sets: Sequence[Mapping[int, torch.Tensor]],
    *,
    rate_axis: str,
    replacement_fraction: float = 1.0,
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Assign exactly two K4 tiles per strip and leave every other tile K3.

    The fixed floor promotes records 22 and 23 in every strip.  For each
    orthogonal-tile and within-record position, compare the 24 corresponding
    record tiles and form the best two-position replacement.  Apply only the
    requested fraction of replacements, ranked by their proxy improvement
    over the fixed floor.  There is no K2 donor and therefore no pairing
    assumption.  The neuron permutation is already frozen before this
    function sees the tile costs.
    """

    if rate_axis not in ("k", "n"):
        raise ValueError("top-two K4 map needs rate_axis 'k' or 'n'")
    if not 0.0 <= replacement_fraction <= 1.0:
        raise ValueError("top-two K4 replacement fraction must lie in [0, 1]")
    if not error_sets:
        raise ValueError("top-two K4 map needs at least one error surface")
    shape = error_sets[0][3].shape
    if any(errors[3].shape != shape for errors in error_sets):
        raise ValueError("top-two K4 error surfaces must align")
    tiles_k, tiles_n = shape
    rate_tiles = tiles_k if rate_axis == "k" else tiles_n
    orthogonal_tiles = tiles_n if rate_axis == "k" else tiles_k
    if rate_tiles != 192 or orthogonal_tiles != 224:
        raise ValueError("Kimi top-two K4 maps require a 192x224 tile grid")

    k3 = sum(
        (errors[3].detach().double() for errors in error_sets),
        start=torch.zeros_like(error_sets[0][3], dtype=torch.float64),
    )
    k4 = sum(
        (errors[4].detach().double() for errors in error_sets),
        start=torch.zeros_like(error_sets[0][4], dtype=torch.float64),
    )

    def as_strips(value: torch.Tensor) -> torch.Tensor:
        if rate_axis == "n":
            return value.reshape(224, 24, 8).permute(0, 2, 1).reshape(-1, 24)
        return value.reshape(24, 8, 224).permute(2, 1, 0).reshape(-1, 24)

    delta = as_strips(k4 - k3)
    proposed = torch.argsort(delta, dim=1, stable=True)[:, :2]
    baseline = torch.tensor((22, 23), dtype=torch.long, device=delta.device)
    baseline_cost = delta.index_select(1, baseline).sum(dim=1)
    proposed_cost = delta.gather(1, proposed).sum(dim=1)
    proposal_gain = baseline_cost - proposed_cost
    replacement_count = int(round(delta.shape[0] * replacement_fraction))
    replaced = torch.zeros(delta.shape[0], dtype=torch.bool, device=delta.device)
    if replacement_count:
        order = torch.argsort(proposal_gain, descending=True, stable=True)
        selected = order[:replacement_count]
        selected = selected[proposal_gain.index_select(0, selected) > 0]
        replaced[selected] = True
    promoted = baseline.expand(delta.shape[0], -1).clone()
    promoted[replaced] = proposed[replaced]
    strip_rates = torch.full_like(delta, 3, dtype=torch.int8)
    strip_rates.scatter_(1, promoted, 4)
    if not bool(torch.all(strip_rates.sum(dim=1) == 74)):
        raise AssertionError("top-two K4 map violated its local 74-bit budget")

    if rate_axis == "n":
        rate_map = (
            strip_rates.reshape(224, 8, 24)
            .permute(0, 2, 1)
            .reshape(224, 192)
        )
    else:
        rate_map = (
            strip_rates.reshape(224, 8, 24)
            .permute(2, 1, 0)
            .reshape(192, 224)
        )
    counts = torch.bincount(promoted.flatten(), minlength=24)
    return (
        tuple(int(value) for value in rate_map.flatten().cpu().tolist()),
        {
            "strips": orthogonal_tiles * 8,
            "k4_tiles_per_strip": 2,
            "k2_tiles_per_strip": 0,
            "compact_selector_bits_per_strip": 9,
            "compact_selector_bits": 9 * orthogonal_tiles * 8,
            "compact_selector_bytes": 9 * orthogonal_tiles,
            "direct_selector": "one_24bit_record_mask_per_strip",
            "direct_selector_storage_bits_per_strip": 32,
            "direct_selector_storage_bytes": 4 * orthogonal_tiles * 8,
            "payload_layout": (
                "rate_grouped_per_strip_22_fixed_k3_slots_then_2_fixed_k4_slots"
            ),
            "serving_access": (
                "load_preparation_expands_record_mask_into_disposable_rate_work_queues"
            ),
            "replacement_fraction": replacement_fraction,
            "replacement_strips": int(torch.count_nonzero(replaced).cpu()),
            "proposal_gain_min": float(proposal_gain.min().cpu()),
            "proposal_gain_median": float(proposal_gain.median().cpu()),
            "proposal_gain_max": float(proposal_gain.max().cpu()),
            "record_selection_counts": [int(value) for value in counts.cpu()],
            "proxy_delta_min": float(delta.min().cpu()),
            "proxy_delta_median": float(delta.median().cpu()),
            "proxy_delta_max": float(delta.max().cpu()),
        },
    )


def _qsrt_308_strip_optimal_map(
    error_sets: Sequence[Mapping[int, torch.Tensor]],
    *,
    rate_axis: str,
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Minimize a tile proxy independently under each strip's 74-bit budget."""

    if rate_axis not in ("k", "n"):
        raise ValueError("strip-optimal map needs rate_axis 'k' or 'n'")
    if not error_sets:
        raise ValueError("strip-optimal map needs at least one error surface")
    shape = error_sets[0][3].shape
    if any(errors[3].shape != shape for errors in error_sets):
        raise ValueError("strip-optimal error surfaces must align")
    tiles_k, tiles_n = shape
    rate_tiles = tiles_k if rate_axis == "k" else tiles_n
    orthogonal_tiles = tiles_n if rate_axis == "k" else tiles_k
    if rate_tiles != 192 or orthogonal_tiles != 224:
        raise ValueError("Kimi strip-optimal maps require a 192x224 tile grid")
    costs = {
        bits: sum(
            (errors[bits].detach().double() for errors in error_sets),
            start=torch.zeros_like(error_sets[0][bits], dtype=torch.float64),
        )
        for bits in (2, 3, 4)
    }

    def as_strips(value: torch.Tensor) -> torch.Tensor:
        if rate_axis == "n":
            return value.reshape(224, 24, 8).permute(0, 2, 1).reshape(-1, 24)
        return value.reshape(24, 8, 224).permute(2, 1, 0).reshape(-1, 24)

    strip_costs = {bits: as_strips(value) for bits, value in costs.items()}
    strip_count = orthogonal_tiles * 8
    # Track the cumulative rate relative to K3.  The target is +2, i.e. state
    # 26 with an offset of 24.  Preference order K3, K4, K2 minimizes needless
    # rate movement under an exact numerical tie.
    state_count = 49
    offset = 24
    infinity = torch.tensor(float("inf"), device=strip_costs[3].device)
    dp = torch.full(
        (strip_count, state_count), infinity, dtype=torch.float64,
        device=strip_costs[3].device,
    )
    dp[:, offset] = 0
    choices = torch.full(
        (24, strip_count, state_count), -1, dtype=torch.int8,
        device=strip_costs[3].device,
    )
    for position in range(24):
        updated = torch.full_like(dp, infinity)
        selected = torch.full_like(choices[position], -1)
        for bits, delta in ((3, 0), (4, 1), (2, -1)):
            candidate = torch.full_like(dp, infinity)
            local = strip_costs[bits][:, position, None]
            if delta == 0:
                candidate = dp + local
            elif delta > 0:
                candidate[:, delta:] = dp[:, :-delta] + local
            else:
                candidate[:, :delta] = dp[:, -delta:] + local
            better = candidate < updated
            updated = torch.where(better, candidate, updated)
            selected = torch.where(
                better, torch.tensor(bits, dtype=torch.int8, device=selected.device), selected
            )
        dp = updated
        choices[position] = selected

    strip_rates = torch.empty(
        (strip_count, 24), dtype=torch.int8, device=dp.device
    )
    states = torch.full(
        (strip_count,), offset + 2, dtype=torch.long, device=dp.device
    )
    rows = torch.arange(strip_count, device=dp.device)
    for position in range(23, -1, -1):
        bits = choices[position, rows, states]
        if bool(torch.any(bits < 2)):
            raise AssertionError("strip-optimal traceback reached an invalid state")
        strip_rates[:, position] = bits
        states -= bits.long() - 3
    if not bool(torch.all(states == offset)):
        raise AssertionError("strip-optimal traceback did not return to K3 origin")
    if not bool(torch.all(strip_rates.sum(dim=1) == 74)):
        raise AssertionError("strip-optimal map violated its 74-bit strip budget")

    if rate_axis == "n":
        rate_map = strip_rates.reshape(224, 8, 24).permute(0, 2, 1).reshape(shape)
    else:
        rate_map = strip_rates.reshape(224, 8, 24).permute(2, 1, 0).reshape(shape)
    baseline_cost = strip_costs[3].sum(dim=1)
    selected_cost = dp[:, offset + 2]
    rate_counts = {
        str(bits): int(torch.count_nonzero(strip_rates == bits))
        for bits in (2, 3, 4)
    }
    donor_counts = torch.count_nonzero(strip_rates == 2, dim=1)
    if not torch.equal(
        torch.count_nonzero(strip_rates == 4, dim=1), donor_counts + 2
    ):
        raise AssertionError("strip-optimal K2/K4 counts do not balance")
    return (
        tuple(int(value) for value in rate_map.flatten().cpu().tolist()),
        {
            "selection": "exact_dynamic_program_over_24_tile_rates",
            "strip_count": strip_count,
            "rate_sum_per_strip": 74,
            "rate_counts": rate_counts,
            "donor_count_per_strip": {
                "minimum": int(donor_counts.min()),
                "median": int(donor_counts.median()),
                "maximum": int(donor_counts.max()),
            },
            "proxy_gain_sum": float((baseline_cost - selected_cost).sum()),
            "proxy_gain_positive_strips": int(
                torch.count_nonzero(selected_cost < baseline_cost)
            ),
            "naive_metadata_bits": 2 * 24 * strip_count,
            "direct_selector": (
                "one_64bit_two_bit_rate_word_per_strip_per_matrix_family"
            ),
            "direct_selector_bytes_per_expert": 16 * strip_count,
        },
    )


def _outer_slope_k2_luts(
    device: torch.device,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, object]]:
    """Build research-only T12 K2 laws on the primary SQG graph."""

    ranks = torch.arange(1 << 16, dtype=torch.float64)
    probability = (ranks + 0.5) / (1 << 16)
    gaussian = torch.special.ndtri(probability)
    q75 = float(torch.special.ndtri(torch.tensor(0.75, dtype=torch.float64)))
    graph_ranks = sqg_xor_rank_permutation(2)
    current_t12 = sqg_xor_cheb_t12_rank_lut_bytes()
    result = {}
    evidence = {}
    current_state = sqg_xor_cheb_t12_bytes(2)
    for name, slope in OUTER_SLOPES.items():
        magnitude = gaussian.abs()
        shaped = torch.where(
            magnitude <= q75,
            magnitude,
            q75 + slope * (magnitude - q75),
        )
        values = 1.5 * gaussian.sign() * shaped
        exact = values.float().to(torch.float8_e4m3fn).view(torch.uint8).clone()
        exact[(exact & 0x7F) == 0] = 0
        blocks = exact.reshape(1 << 12, 16)
        t12 = torch.empty(1 << 12, dtype=torch.uint8)
        for index, block in enumerate(blocks):
            labels, counts = torch.unique(block, return_counts=True)
            t12[index] = labels[counts == counts.max()].min()
        if name == "normal" and not torch.equal(t12, current_t12):
            raise AssertionError("normal switched-law T12 does not match QSRT")
        state_lut = t12.index_select(0, graph_ranks >> 4).contiguous()
        result[name] = {
            2: state_lut.to(device),
            3: sqg_xor_cheb_t12_bytes(3, device=device),
            4: sqg_xor_cheb_t12_bytes(4, device=device),
        }
        evidence[name] = {
            "outer_slope": slope,
            "t12_sha256": hashlib.sha256(t12.numpy().tobytes()).hexdigest(),
            "k2_state_label_differences": int(torch.count_nonzero(state_lut != current_state)),
            "exact_rank_label_differences_from_normal": int(
                torch.count_nonzero(exact != sqg_cheb_normal_rank_e4m3_bytes())
            ),
        }
    return result, evidence


def _quantize_maps(
    source: torch.Tensor,
    hessian: torch.Tensor,
    contexts: torch.Tensor,
    *,
    matrix: str,
    maps: Mapping[str, tuple[int, ...]],
    layer: int,
    expert: int,
    device: torch.device,
    quantizer_module,
    ldlq_tf32: bool,
    g_scale_override: float | None = None,
    luts_by_bits: Mapping[int, torch.Tensor] | None = None,
    tile_codebook_ids: tuple[int, ...] | None = None,
    lut_bank_by_bits: Mapping[int, Sequence[torch.Tensor]] | None = None,
    prepared: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    permutation_override: torch.Tensor | None = None,
    shared_scale_scope: str = "k2-tile-allocation",
    scale_search_bits: int = 3,
    h2_viterbi_refine_sweeps: Mapping[str, int] | None = None,
) -> tuple[dict[str, dict[str, object]], torch.Tensor]:
    if scale_search_bits not in (1, 2, 3, 4):
        raise ValueError("scale-search rate must lie in K1 through K4")
    if prepared is None and permutation_override is None:
        raise ValueError(
            "tile-funded experiments require an explicit frozen neuron permutation"
        )
    if prepared is None:
        weight, encoder_hessian, permutation = _encoder_coordinates(
            source,
            hessian,
            contexts,
            matrix=matrix,
            permutation_override=permutation_override,
        )
    else:
        weight, encoder_hessian, permutation = prepared
        if permutation_override is not None and not torch.equal(
            permutation.to(device=permutation_override.device, dtype=torch.long),
            permutation_override.to(dtype=torch.long),
        ):
            raise ValueError(
                "prepared encoder coordinates do not match the frozen permutation"
            )
        # The extension regularizes its work tensor in place and may retain
        # workspace aliases until the batch completes.  Prepared coordinates
        # are immutable; every encode receives its own materialized weight.
        weight = weight.clone()
    permutation_identity = _permutation_sha256(permutation)
    # EXL3 finalizes and mutates this wrapper while factoring the conditioned
    # Hessian.  Cache the immutable encoder-coordinate tensors above, but give
    # every batched encode a fresh wrapper.
    shared_h = make_shared_h(weight.shape[0], device, encoder_hessian)
    expected = (weight.shape[0] // 16) * (weight.shape[1] // 16)
    if any(len(rate_map) != expected for rate_map in maps.values()):
        raise ValueError("two-dimensional rate map has the wrong tile count")
    transform_seed = layer * 1_000_000 + MATRICES.index(matrix)
    args_group = []
    for name, rate_map in maps.items():
        args: dict[str, object] = {
            "K": scale_search_bits,
            "seed": transform_seed,
            "sv_seed": transform_seed + 499_979,
            "sigma_reg": SIGMA_REG,
            "devices": [str(device)],
            "device_ratios": None,
            "apply_out_scales": False,
            "ldlq_tf32": bool(ldlq_tf32),
            "tailbite_context": 128,
            "mixed_rate_axis": "tile",
            "mixed_tile_bits": rate_map,
            "sqg_e4m3_luts_by_bits": (
                dict(luts_by_bits)
                if luts_by_bits is not None
                else {
                    bits: sqg_xor_cheb_t12_bytes(bits, device=device)
                    for bits in (2, 3, 4)
                }
            ),
        }
        if tile_codebook_ids is not None or lut_bank_by_bits is not None:
            if tile_codebook_ids is None or lut_bank_by_bits is None:
                raise ValueError(
                    "tile codebook IDs and LUT banks must be supplied together"
                )
            if len(maps) != 1:
                raise ValueError(
                    "selector-aware codebook encoding accepts one tile map"
                )
            if len(tile_codebook_ids) != len(rate_map):
                raise ValueError("tile codebook IDs do not align with the rate map")
            args["mixed_tile_codebook_ids"] = tile_codebook_ids
            args["sqg_e4m3_lut_banks_by_bits"] = {
                bits: tuple(bank) for bits, bank in lut_bank_by_bits.items()
            }
        if matrix in ("w1", "w3"):
            args["shared_input_scales_key"] = (
                shared_scale_scope,
                layer,
                expert,
                matrix,
            )
            args["g_scale_into_sv"] = True
        if g_scale_override is not None:
            args["g_scale_override"] = float(g_scale_override)
        if h2_viterbi_refine_sweeps is not None:
            sweeps = int(h2_viterbi_refine_sweeps.get(name, 0))
            if sweeps < 0:
                raise ValueError("H2 Viterbi refinement sweeps must be nonnegative")
            if sweeps:
                args["h2_viterbi_refine_sweeps"] = sweeps
                args["h2_viterbi_refine_dither_scales"] = ()
                args["h2_viterbi_refine_patterns"] = 0
        args_group.append(args)
    raw_group = quantizer_module.quantize_qsrt_batch(
        [weight], [shared_h], [args_group], return_weight_q=True
    )[0]
    if len(raw_group) != len(maps):
        raise ValueError("encoder returned the wrong number of tile-map candidates")
    result: dict[str, dict[str, object]] = {}
    for (name, rate_map), raw in zip(maps.items(), raw_group, strict=True):
        encoder_reconstruction = raw["weight_q"]
        if encoder_reconstruction is None:
            raise ValueError("tile-map experiment requires reconstructed weights")
        if tile_codebook_ids is not None:
            regularized = None
        elif luts_by_bits is not None and len(set(rate_map)) == 1:
            bits = rate_map[0]
            raw_lut = luts_by_bits[bits].detach().contiguous()
            values = raw_lut.view(torch.float8_e4m3fn).float()
            regularized = decode_regularized_weight(
                raw["encoded"], codebook_values=values, bits=bits
            )
        else:
            regularized = decode_qsrt_regularized_weight(
                raw["encoded"],
                rate_axis="tile",
                tile_bits=rate_map,
                codebook=CODEBOOK_SQG_XOR_CHEB_T12,
            )
        result[name] = {
            "reconstruction": _canonical_reconstruction(
                encoder_reconstruction, source, permutation, matrix=matrix
            ),
            "regularized": regularized,
            "states": raw["encoded"],
            "suh": raw["suh"],
            "svh": raw["svh"],
            "proxy": float(raw["proxy"]),
            "g_scale": float(raw["g_scale"]),
            "rate_map": rate_map,
            "permutation_sha256": permutation_identity,
            "h2_viterbi_refine": raw.get(
                "h2_viterbi_refine", {"enabled": False, "sweeps": []}
            ),
        }
    return result, weight


def _prepare_quantize_maps(
    source: torch.Tensor,
    hessian: torch.Tensor,
    contexts: torch.Tensor,
    *,
    matrix: str,
    device: torch.device,
    permutation_override: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight, encoder_hessian, permutation = _encoder_coordinates(
        source,
        hessian,
        contexts,
        matrix=matrix,
        permutation_override=permutation_override,
    )
    return weight, encoder_hessian, permutation


def _functional_neuron_group_scores(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    inputs: torch.Tensor,
    gates: torch.Tensor,
) -> torch.Tensor:
    """Rank four-neuron groups by a first-order complete-expert sensitivity.

    The upstream term measures SiTU derivative energy propagated through the
    corresponding W2 column.  The down term measures the routed post-SiTU
    energy that weights W2 column error.  Each term is normalized before they
    are combined so neither matrix family wins merely through units.
    """

    w1, w3, w2 = source
    gate_pre = F.linear(inputs, w1).float()
    up_pre = F.linear(inputs, w3).float()
    tanh_gate = torch.tanh(gate_pre / 4.0)
    sigmoid = torch.sigmoid(gate_pre)
    gate_value = 4.0 * tanh_gate * sigmoid
    up_tanh = torch.tanh(up_pre / 25.0)
    up_value = 25.0 * up_tanh
    gate_prime = (
        (1.0 - tanh_gate.square()) * sigmoid
        + 4.0 * tanh_gate * sigmoid * (1.0 - sigmoid)
    )
    up_prime = 1.0 - up_tanh.square()
    weights = gates.float().square()
    denominator = weights.double().sum().clamp_min(1e-30)
    derivative_energy = (
        ((up_value * gate_prime).square() + (gate_value * up_prime).square())
        * weights[:, None]
    ).sum(dim=0, dtype=torch.float64) / denominator
    middle_energy = (
        (gate_value * up_value).square() * weights[:, None]
    ).sum(dim=0, dtype=torch.float64) / denominator
    output_energy = w2.float().square().sum(dim=0, dtype=torch.float64)
    upstream = derivative_energy * output_energy

    def normalize(value: torch.Tensor) -> torch.Tensor:
        scale = value.mean().clamp_min(torch.finfo(value.dtype).tiny)
        return value / scale

    neuron_scores = normalize(upstream) + normalize(middle_energy)
    return neuron_scores.reshape(-1, 4).mean(dim=1).float()


def _tile_balanced_group_order(scores: torch.Tensor) -> torch.Tensor:
    """Preserve ordered record populations while balancing physical tiles.

    The 768 four-neuron groups are first partitioned into the same 24
    equal-population low-to-high records as the ordinary exact score order.
    Only the 32 groups inside one record may move.  Greedy scheduling places
    four groups in each of its eight 16-neuron tiles and minimizes tile score
    imbalance.  Consequently this conditioning permutation cannot move a
    neuron across a K2/K3/K4 record boundary; tile funding remains a separate
    decision in the resulting physical basis.
    """

    if scores.ndim != 1 or scores.numel() != 768:
        raise ValueError("tile-balanced ordering needs 768 group scores")
    if not torch.is_floating_point(scores) or not bool(torch.all(torch.isfinite(scores))):
        raise ValueError("tile-balanced group scores must be finite floating point")
    ranked = torch.argsort(scores, stable=True)
    result: list[int] = []
    groups_per_record = 32
    tiles_per_record = 8
    groups_per_tile = 4
    for record in range(24):
        members = ranked[
            record * groups_per_record : (record + 1) * groups_per_record
        ]
        bins: list[list[int]] = [[] for _ in range(tiles_per_record)]
        totals = [0.0] * tiles_per_record
        for group in members.flip(0).tolist():
            choices = [
                tile
                for tile in range(tiles_per_record)
                if len(bins[tile]) < groups_per_tile
            ]
            tile = min(choices, key=lambda index: (totals[index], index))
            bins[tile].append(group)
            totals[tile] += float(scores[group])
        for tile in sorted(range(tiles_per_record), key=lambda index: (totals[index], index)):
            result.extend(sorted(bins[tile], key=lambda group: (float(scores[group]), group)))
    order = torch.tensor(result, dtype=torch.long, device=scores.device)
    if not torch.equal(torch.sort(order).values, torch.arange(768, device=scores.device)):
        raise AssertionError("tile-balanced group order is not bijective")
    return order


def _codec_shape_group_features(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Describe four-neuron groups by scale and tail shape in all matrices."""

    w1, w3, w2 = (matrix.float() for matrix in source)

    def features(values: torch.Tensor, *, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        second = values.square().mean(dim=dim, dtype=torch.float64).clamp_min(1e-30)
        rms = second.sqrt()
        peak = values.abs().amax(dim=dim).double()
        return second.log2(), (peak / rms).clamp_min(1.0).log2()

    w1_scale, w1_tail = features(w1, dim=1)
    w3_scale, w3_tail = features(w3, dim=1)
    w2_scale, w2_tail = features(w2, dim=0)
    per_neuron = torch.stack(
        (w1_scale, w1_tail, w3_scale, w3_tail, w2_scale, w2_tail), dim=1
    )
    grouped = per_neuron.reshape(768, 4, 6).mean(dim=1)
    center = grouped.mean(dim=0)
    scale = grouped.std(dim=0).clamp_min(1e-12)
    return ((grouped - center) / scale).float()


def _record_clustered_group_order(
    priority_scores: torch.Tensor, features: torch.Tensor
) -> torch.Tensor:
    """Keep priority records fixed and form feature-homogeneous neuron tiles."""

    if priority_scores.ndim != 1 or priority_scores.numel() != 768:
        raise ValueError("record clustering needs 768 priority scores")
    if (
        features.ndim != 2
        or features.shape[0] != 768
        or features.shape[1] < 1
        or not bool(torch.all(torch.isfinite(features)))
    ):
        raise ValueError("record clustering needs 768 finite feature vectors")
    ranked = torch.argsort(priority_scores, stable=True)
    result: list[int] = []
    for record in range(24):
        members = ranked[record * 32 : (record + 1) * 32].tolist()
        tiles: list[list[int]] = []
        remaining = list(members)
        while remaining:
            # Anchor tiles in deterministic priority order, then take the three
            # closest source-shape signatures.  Only members of this record are
            # eligible, so conditioning cannot alter rate-region membership.
            seed = remaining.pop(0)
            if remaining:
                candidates = torch.tensor(
                    remaining, dtype=torch.long, device=features.device
                )
                distances = (
                    features.index_select(0, candidates) - features[seed]
                ).square().sum(dim=1)
                neighbor_positions = torch.argsort(distances, stable=True)[:3].tolist()
                neighbors = [remaining[position] for position in neighbor_positions]
                neighbor_set = set(neighbors)
                remaining = [group for group in remaining if group not in neighbor_set]
            else:
                neighbors = []
            tile = [seed, *neighbors]
            if len(tile) != 4:
                raise AssertionError("shape clustering produced an incomplete tile")
            tiles.append(tile)
        tiles.sort(
            key=lambda tile: (
                sum(float(priority_scores[group]) for group in tile),
                min(tile),
            )
        )
        for tile in tiles:
            result.extend(
                sorted(tile, key=lambda group: (float(priority_scores[group]), group))
            )
    order = torch.tensor(result, dtype=torch.long, device=priority_scores.device)
    if not torch.equal(torch.sort(order).values, torch.arange(768, device=order.device)):
        raise AssertionError("record-clustered group order is not bijective")
    return order


def _maximum_weight_assignment(scores: torch.Tensor) -> torch.Tensor:
    """Return the item occupying each column in a small square assignment."""

    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("assignment scores must be square")
    size = scores.shape[0]
    if not 1 <= size <= 12 or not bool(torch.all(torch.isfinite(scores))):
        raise ValueError("assignment scores must be finite with size 1..12")
    values = scores.detach().double().cpu()
    # State values are (score, item-to-column path).  Eight funding stripes
    # need only 8 * 2^8 transitions, so keeping this exact and dependency-free
    # is preferable to introducing a general assignment package.
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for item in range(size):
        updated: dict[int, tuple[float, tuple[int, ...]]] = {}
        for mask, (total, path) in states.items():
            for column in range(size):
                bit = 1 << column
                if mask & bit:
                    continue
                candidate = total + float(values[item, column])
                key = mask | bit
                current = updated.get(key)
                candidate_path = (*path, column)
                if current is None or (candidate, tuple(-v for v in candidate_path)) > (
                    current[0],
                    tuple(-v for v in current[1]),
                ):
                    updated[key] = (candidate, candidate_path)
        states = updated
    item_to_column = states[(1 << size) - 1][1]
    column_to_item = torch.empty(size, dtype=torch.long)
    for item, column in enumerate(item_to_column):
        column_to_item[column] = item
    return column_to_item


def _band_alignment_inputs(
    errors: Mapping[str, Mapping[int, torch.Tensor]],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Validate functional upstream/down rate surfaces for 16-channel bands."""

    if set(errors) != {"w13", "w2"}:
        raise ValueError("band alignment needs coupled upstream and down surfaces")
    upstream = {bits: errors["w13"][bits].detach().double() for bits in (2, 3, 4)}
    down = {bits: errors["w2"][bits].detach().double() for bits in (2, 3, 4)}
    if any(value.shape != (224, 192) for value in upstream.values()):
        raise ValueError("upstream band surfaces must have shape 224x192")
    if any(value.shape != (192, 224) for value in down.values()):
        raise ValueError("down band surfaces must have shape 192x224")
    if any(
        not bool(torch.all(torch.isfinite(value)))
        for family in (upstream, down)
        for value in family.values()
    ):
        raise ValueError("band alignment surfaces must be finite")
    return upstream, down


def _band_permutation(
    base_permutation: torch.Tensor, band_order: torch.Tensor
) -> torch.Tensor:
    """Apply a positional 16-channel-band order to a frozen base permutation."""

    if base_permutation.ndim != 1 or base_permutation.numel() != 3072:
        raise ValueError("band alignment requires a 3072-channel permutation")
    if band_order.shape != (24, 8):
        raise ValueError("band order must have shape 24x8")
    expected = torch.arange(192, device=band_order.device)
    if not torch.equal(torch.sort(band_order.flatten()).values, expected):
        raise ValueError("band order is not bijective")
    base_bands = base_permutation.reshape(192, 16)
    return base_bands.index_select(
        0, band_order.flatten().to(device=base_bands.device)
    ).flatten()


def _p24_band_objective(
    band_order: torch.Tensor,
    upstream: Mapping[int, torch.Tensor],
    down: Mapping[int, torch.Tensor],
) -> float:
    """Fit-proxy gain of independently selected P24/P33 tile pairs."""

    total = torch.zeros((), dtype=torch.float64, device=upstream[3].device)
    order = band_order.to(device=upstream[3].device)
    for low_record in range(11):
        high_record = 21 - low_record
        for column in range(8):
            low = int(order[low_record, column])
            high = int(order[high_record, column])
            upstream_gain = (
                upstream[3][:, low]
                + upstream[3][:, high]
                - upstream[2][:, low]
                - upstream[4][:, high]
            )
            down_gain = (
                down[3][low, :]
                + down[3][high, :]
                - down[2][low, :]
                - down[4][high, :]
            )
            total += upstream_gain.clamp_min(0).sum()
            total += down_gain.clamp_min(0).sum()
    return float(total.cpu())


def _p24_band_aligned_permutation(
    base_permutation: torch.Tensor,
    errors: Mapping[str, Mapping[int, torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, object]]:
    """Pair intact neuron bands to maximize the tile-local P24/P33 frontier.

    Record membership and every existing 16-channel band remain fixed.  Only
    the eight band positions inside each high record are matched to the eight
    positions of its low partner.  This makes the proposal unit agree with the
    neuron-side funding band without pretending that a band is one coefficient
    tile: its full 224-component incident-tile response participates.
    """

    upstream, down = _band_alignment_inputs(errors)
    band_order = torch.arange(192, dtype=torch.long).reshape(24, 8)
    before = _p24_band_objective(band_order, upstream, down)
    changed_pairs = 0
    for low_record in range(11):
        high_record = 21 - low_record
        low_bands = band_order[low_record]
        high_bands = band_order[high_record].clone()
        scores = torch.empty((8, 8), dtype=torch.float64, device=upstream[3].device)
        for high_item, high_value in enumerate(high_bands.tolist()):
            for column, low_value in enumerate(low_bands.tolist()):
                upstream_gain = (
                    upstream[3][:, low_value]
                    + upstream[3][:, high_value]
                    - upstream[2][:, low_value]
                    - upstream[4][:, high_value]
                )
                down_gain = (
                    down[3][low_value, :]
                    + down[3][high_value, :]
                    - down[2][low_value, :]
                    - down[4][high_value, :]
                )
                scores[high_item, column] = (
                    upstream_gain.clamp_min(0).sum()
                    + down_gain.clamp_min(0).sum()
                )
        column_to_item = _maximum_weight_assignment(scores)
        reordered = high_bands.index_select(0, column_to_item)
        changed_pairs += int(torch.count_nonzero(reordered != high_bands))
        band_order[high_record] = reordered
    after = _p24_band_objective(band_order, upstream, down)
    if after + max(abs(before), 1.0) * 1e-12 < before:
        raise AssertionError("P24 band alignment reduced its proposal objective")
    return _band_permutation(base_permutation, band_order), {
        "proposal": "intact_16_channel_band_p24_pair_assignment",
        "record_membership": "fixed_h2_reverse_records",
        "coefficient_tiles_per_band": 224,
        "proxy_objective_before": before,
        "proxy_objective_after": after,
        "proxy_objective_relative_gain": (after - before) / max(abs(before), 1e-30),
        "moved_high_record_band_positions": changed_pairs,
    }


def _top2_band_objective(
    band_order: torch.Tensor, upstream_gain: torch.Tensor, down_gain: torch.Tensor
) -> float:
    """Proxy gain when exactly two K4 records are selected in every strip."""

    order = band_order.to(device=upstream_gain.device)
    total = torch.zeros((), dtype=torch.float64, device=upstream_gain.device)
    for column in range(8):
        bands = order[:, column]
        total += torch.topk(
            upstream_gain.index_select(1, bands), 2, dim=1
        ).values.sum()
        total += torch.topk(
            down_gain.index_select(0, bands), 2, dim=0
        ).values.sum()
    return float(total.cpu())


def _top2_band_aligned_permutation(
    base_permutation: torch.Tensor,
    errors: Mapping[str, Mapping[int, torch.Tensor]],
    *,
    passes: int = 3,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Place intact bands to maximize the exact top-two-K4 tile proxy."""

    if passes <= 0:
        raise ValueError("top-two band alignment needs at least one pass")
    upstream, down = _band_alignment_inputs(errors)
    upstream_gain = upstream[3] - upstream[4]
    down_gain = down[3] - down[4]
    band_order = torch.arange(192, dtype=torch.long).reshape(24, 8)
    before = _top2_band_objective(band_order, upstream_gain, down_gain)
    objectives = [before]
    changed_positions = 0
    for _ in range(passes):
        pass_changes = 0
        for record in range(24):
            items = band_order[record].clone()
            other_records = torch.tensor(
                [index for index in range(24) if index != record], dtype=torch.long
            )
            scores = torch.empty(
                (8, 8), dtype=torch.float64, device=upstream_gain.device
            )
            for item, band in enumerate(items.tolist()):
                for column in range(8):
                    other_bands = band_order.index_select(
                        0, other_records
                    )[:, column].to(device=upstream_gain.device)
                    candidate = torch.tensor(
                        [band], dtype=torch.long, device=upstream_gain.device
                    )
                    bands = torch.cat((other_bands, candidate))
                    scores[item, column] = (
                        torch.topk(
                            upstream_gain.index_select(1, bands), 2, dim=1
                        ).values.sum()
                        + torch.topk(
                            down_gain.index_select(0, bands), 2, dim=0
                        ).values.sum()
                    )
            column_to_item = _maximum_weight_assignment(scores)
            reordered = items.index_select(0, column_to_item)
            pass_changes += int(torch.count_nonzero(reordered != items))
            band_order[record] = reordered
        objective = _top2_band_objective(band_order, upstream_gain, down_gain)
        if objective + max(abs(objectives[-1]), 1.0) * 1e-12 < objectives[-1]:
            raise AssertionError("top-two band alignment reduced its objective")
        objectives.append(objective)
        changed_positions += pass_changes
        if pass_changes == 0:
            break
    after = objectives[-1]
    return _band_permutation(base_permutation, band_order), {
        "proposal": "intact_16_channel_band_top2_k4_assignment",
        "record_membership": "fixed_h2_reverse_records",
        "coefficient_tiles_per_band": 224,
        "passes_completed": len(objectives) - 1,
        "proxy_objectives": objectives,
        "proxy_objective_relative_gain": (after - before) / max(abs(before), 1e-30),
        "moved_band_positions_across_passes": changed_positions,
    }


def _shape_clustered_group_order(
    priority_scores: torch.Tensor, features: torch.Tensor
) -> torch.Tensor:
    """Keep priority records fixed and cluster the six source-shape features."""

    if features.shape != (768, 6):
        raise ValueError("shape clustering needs 768 six-component features")
    return _record_clustered_group_order(priority_scores, features)


def _global_shape_clustered_group_order(
    priority_scores: torch.Tensor, features: torch.Tensor
) -> torch.Tensor:
    """Form globally homogeneous 16-neuron tiles, then order by priority.

    This policy is intended for tile-funded experiments. It may move a
    four-neuron group across an ordinary 128-channel importance-record
    boundary, so all rate-error surfaces must be rebuilt after selecting it.
    """

    if priority_scores.ndim != 1 or priority_scores.numel() != 768:
        raise ValueError("global shape clustering needs 768 priority scores")
    if (
        features.ndim != 2
        or features.shape[0] != 768
        or features.shape[1] < 1
        or not bool(torch.all(torch.isfinite(features)))
    ):
        raise ValueError(
            "global shape clustering needs 768 finite feature vectors"
        )
    remaining = torch.argsort(priority_scores, stable=True).tolist()
    tiles: list[list[int]] = []
    while remaining:
        seed = remaining.pop(0)
        candidates = torch.tensor(
            remaining, dtype=torch.long, device=features.device
        )
        distances = (
            features.index_select(0, candidates) - features[seed]
        ).square().sum(dim=1)
        neighbor_positions = torch.argsort(distances, stable=True)[:3].tolist()
        neighbors = [remaining[position] for position in neighbor_positions]
        neighbor_set = set(neighbors)
        remaining = [group for group in remaining if group not in neighbor_set]
        tile = [seed, *neighbors]
        if len(tile) != 4:
            raise AssertionError("global shape clustering produced an incomplete tile")
        tiles.append(tile)
    tiles.sort(
        key=lambda tile: (
            sum(float(priority_scores[group]) for group in tile),
            min(tile),
        )
    )
    order = torch.tensor(
        [
            group
            for tile in tiles
            for group in sorted(
                tile, key=lambda group: (float(priority_scores[group]), group)
            )
        ],
        dtype=torch.long,
        device=priority_scores.device,
    )
    if not torch.equal(
        torch.sort(order).values, torch.arange(768, device=order.device)
    ):
        raise AssertionError("global shape-clustered group order is not bijective")
    return order


def _priority_shape_clustered_group_order(
    priority_scores: torch.Tensor, features: torch.Tensor
) -> torch.Tensor:
    """Form tiles that jointly condition sensitivity and codec source shape.

    The six source-shape coordinates are standardized by their constructor.
    Normalize the priority score and give it squared weight equal to the sum
    of the six shape coordinates.  This makes priority and source shape equal
    aggregate axes in the nearest-neighbor tile construction, rather than
    silently allowing either one to dominate by dimensionality alone.
    """

    if priority_scores.ndim != 1 or priority_scores.numel() != 768:
        raise ValueError("priority-shape clustering needs 768 priority scores")
    if features.shape != (768, 6) or not bool(torch.all(torch.isfinite(features))):
        raise ValueError(
            "priority-shape clustering needs 768 finite six-component features"
        )
    priority = priority_scores.float()
    priority = (priority - priority.mean()) / priority.std().clamp_min(1e-12)
    joint_features = torch.cat(
        (features, priority[:, None] * (features.shape[1] ** 0.5)), dim=1
    )
    return _global_shape_clustered_group_order(priority_scores, joint_features)


def _permutation_tile_geometry(
    permutation: torch.Tensor,
    scores: torch.Tensor,
    *,
    codec_features: torch.Tensor | None = None,
) -> dict[str, object]:
    """Describe rate sensitivity and source-shape geometry after permutation."""

    grouped = permutation.reshape(-1, 4)
    if not bool(torch.all(grouped == grouped[:, :1] + torch.arange(4, device=grouped.device))):
        raise ValueError("neuron permutation split a four-neuron score group")
    group_order = torch.div(grouped[:, 0], 4, rounding_mode="floor")
    ordered = scores.index_select(0, group_order)
    records = ordered.reshape(24, 32)
    tile_groups = ordered.reshape(192, 4)
    tile_totals = tile_groups.sum(dim=1).reshape(24, 8)
    record_totals = records.sum(dim=1)
    record_tile_spread = tile_totals.max(dim=1).values - tile_totals.min(dim=1).values
    within_tile_priority_spread = tile_groups.max(dim=1).values - tile_groups.min(
        dim=1
    ).values
    exact_ranked = torch.argsort(scores, stable=True).reshape(24, 32)
    membership_matches = []
    for record in range(24):
        membership_matches.append(
            bool(
                torch.equal(
                    torch.sort(group_order.reshape(24, 32)[record]).values,
                    torch.sort(exact_ranked[record]).values,
                )
            )
        )
    result = {
        "record_membership_matches_exact_score_quantiles": membership_matches,
        "all_record_memberships_match": all(membership_matches),
        "record_totals_monotone": bool(torch.all(record_totals[1:] >= record_totals[:-1])),
        "record_total_range": [float(record_totals.min()), float(record_totals.max())],
        "within_record_tile_spread": {
            "minimum": float(record_tile_spread.min()),
            "median": float(record_tile_spread.median()),
            "maximum": float(record_tile_spread.max()),
        },
        "within_tile_priority_spread": {
            "minimum": float(within_tile_priority_spread.min()),
            "median": float(within_tile_priority_spread.median()),
            "maximum": float(within_tile_priority_spread.max()),
        },
    }
    if codec_features is not None:
        if codec_features.shape != (768, 6):
            raise ValueError("codec feature geometry requires a 768x6 tensor")
        ordered_features = codec_features.index_select(0, group_order).reshape(
            192, 4, 6
        )
        feature_variance = (
            ordered_features - ordered_features.mean(dim=1, keepdim=True)
        ).square().mean(dim=(1, 2))
        result["within_tile_codec_feature_variance"] = {
            "minimum": float(feature_variance.min()),
            "median": float(feature_variance.median()),
            "maximum": float(feature_variance.max()),
            "mean": float(feature_variance.mean()),
        }
    return result


def _tile_errors(
    target: torch.Tensor, candidates: Mapping[int, Mapping[str, object]]
) -> dict[int, torch.Tensor]:
    target_tiles = _tile_view(target)
    result = {}
    for bits, candidate in candidates.items():
        regularized = candidate["regularized"]
        if not isinstance(regularized, torch.Tensor):
            raise TypeError("candidate regularized weight is not a tensor")
        result[bits] = (_tile_view(regularized) - target_tiles).square().sum(dim=2)
    return result


def _dense_h_tile_errors(
    encoder_weight: torch.Tensor,
    encoder_hessian: torch.Tensor,
    candidates: Mapping[int | str, Mapping[str, object]],
) -> dict[int | str, torch.Tensor]:
    """Return additive dense-H costs for independently decoded candidates."""

    result: dict[int | str, torch.Tensor] = {}
    for name, candidate in candidates.items():
        regularized = candidate.get("regularized")
        suh = candidate.get("suh")
        svh = candidate.get("svh")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (regularized, suh, svh)
        ):
            raise TypeError("dense-H candidates require decoded weights and scales")
        target = _target_regularized_weight(encoder_weight, suh, svh)
        contributions = dense_h_tile_error_contributions(
            target,
            regularized,
            encoder_hessian,
            suh,
            svh,
        )
        total = float(contributions.sum())
        if not torch.isfinite(torch.tensor(total)) or total < -1e-4:
            raise ValueError("dense-H candidate has an invalid total distortion")
        result[name] = contributions
    return result


def _four_channel_group_errors(
    target: torch.Tensor,
    candidates: Mapping[int, Mapping[str, object]],
    *,
    rate_axis: str,
    permutation: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Attribute uniform-rate error to each movable group and incident tile.

    The result is expressed in original four-channel group identity and has
    shape ``[768, 224]``.  A row therefore describes one channel group's
    response across every orthogonal 16-channel band affected by moving it.
    """

    if rate_axis not in ("k", "n"):
        raise ValueError("group error attribution needs rate_axis 'k' or 'n'")
    if target.ndim != 2:
        raise ValueError("group error target must be a matrix")
    neuron_channels = target.shape[0] if rate_axis == "k" else target.shape[1]
    orthogonal_channels = target.shape[1] if rate_axis == "k" else target.shape[0]
    if neuron_channels % 4 or orthogonal_channels % 16:
        raise ValueError("group error target is not 4x16 aligned")
    groups = neuron_channels // 4
    orthogonal_bands = orthogonal_channels // 16
    if permutation.numel() != neuron_channels:
        raise ValueError("group error permutation has the wrong length")
    grouped_permutation = permutation.to(dtype=torch.long).reshape(groups, 4)
    if not bool(
        torch.all(
            grouped_permutation
            == grouped_permutation[:, :1]
            + torch.arange(4, device=grouped_permutation.device)
        )
    ):
        raise ValueError("group error attribution requires intact four-channel groups")
    group_at_position = torch.div(
        grouped_permutation[:, 0], 4, rounding_mode="floor"
    )
    result: dict[int, torch.Tensor] = {}
    for bits, candidate in candidates.items():
        regularized = candidate.get("regularized")
        if not isinstance(regularized, torch.Tensor):
            raise TypeError("group error candidate lacks a regularized tensor")
        squared = (regularized.float() - target.float()).square()
        if rate_axis == "n":
            positional = squared.reshape(
                orthogonal_bands, 16, groups, 4
            ).sum(dim=(1, 3)).T
        else:
            positional = squared.reshape(
                groups, 4, orthogonal_bands, 16
            ).sum(dim=(1, 3))
        original = torch.empty_like(positional)
        original.index_copy_(0, group_at_position.to(positional.device), positional)
        result[bits] = original
    if set(result) != {2, 3, 4}:
        raise ValueError("group error attribution requires K2, K3, and K4")
    return result


def _rate_response_group_features(
    targets: Mapping[str, torch.Tensor],
    uniform: Mapping[str, Mapping[int, Mapping[str, object]]],
    *,
    permutation: torch.Tensor,
) -> torch.Tensor:
    """Describe each channel group by all incident K2/K3/K4 responses.

    Six equally normalized feature blocks encode K3 error, K2 donor cost, and
    K4 recipient gain for the coupled upstream pair and for down separately.
    Clustering these vectors inside fixed importance records aligns groups
    that want the same rate at the same orthogonal tile coordinates.
    """

    required = {"w1", "w3", "w2"}
    if set(targets) != required or set(uniform) != required:
        raise ValueError("rate response features require w1, w3, and w2")
    errors = {
        matrix: _four_channel_group_errors(
            targets[matrix],
            uniform[matrix],
            rate_axis="k" if matrix == "w2" else "n",
            permutation=permutation,
        )
        for matrix in ("w1", "w3", "w2")
    }
    families = (
        {
            bits: errors["w1"][bits] + errors["w3"][bits]
            for bits in ((1, 2, 3, 4) if args.p13_search else (2, 3, 4))
        },
        errors["w2"],
    )

    def standardized(value: torch.Tensor) -> torch.Tensor:
        value = value.double()
        center = value.mean(dim=0, keepdim=True)
        scale = value.std(dim=0, keepdim=True).clamp_min(1e-12)
        return ((value - center) / scale).clamp(-6.0, 6.0).float() / (
            value.shape[1] ** 0.5
        )

    blocks: list[torch.Tensor] = []
    for family in families:
        reference = family[3].double()
        epsilon = (
            reference.median(dim=0, keepdim=True).values * 1e-6
        ).clamp_min(torch.finfo(torch.float64).tiny)
        log2 = torch.log(family[2].double() + epsilon)
        log3 = torch.log(reference + epsilon)
        log4 = torch.log(family[4].double() + epsilon)
        blocks.extend(
            (
                standardized(log3),
                standardized(log2 - log3),
                standardized(log3 - log4),
            )
        )
    features = torch.cat(blocks, dim=1)
    if features.shape != (768, 6 * 224) or not bool(torch.all(torch.isfinite(features))):
        raise AssertionError("rate response features have the wrong shape or values")
    return features


def _functional_proxy_tile_errors(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    uniform: Mapping[str, Mapping[int, Mapping[str, object]]],
    *,
    h13: torch.Tensor,
    h2: torch.Tensor,
    inputs: torch.Tensor,
    gates: torch.Tensor,
    permutation: torch.Tensor,
    rates: Sequence[int] = (2, 3, 4),
) -> dict[str, dict[int, torch.Tensor]]:
    """Build local activation/Hessian tile costs in the permuted neuron basis."""

    rates = tuple(int(bits) for bits in rates)
    if not rates or len(set(rates)) != len(rates):
        raise ValueError("functional proxy rates must be unique and nonempty")
    if any(bits not in (1, 2, 3, 4) for bits in rates):
        raise ValueError("functional proxy rates must lie in K1 through K4")

    w1, w3, w2 = source
    gate_pre = F.linear(inputs, w1).float()
    up_pre = F.linear(inputs, w3).float()
    tanh_gate = torch.tanh(gate_pre / 4.0)
    sigmoid = torch.sigmoid(gate_pre)
    gate_value = 4.0 * tanh_gate * sigmoid
    up_tanh = torch.tanh(up_pre / 25.0)
    up_value = 25.0 * up_tanh
    gate_prime = (
        (1.0 - tanh_gate.square()) * sigmoid
        + 4.0 * tanh_gate * sigmoid * (1.0 - sigmoid)
    )
    up_prime = 1.0 - up_tanh.square()
    derivative1 = up_value * gate_prime
    derivative3 = gate_value * up_prime
    route_weights = gates.float().square()
    denominator = route_weights.double().sum().clamp_min(1e-30)
    output_energy = w2.float().square().sum(dim=0, dtype=torch.float64)
    m11 = (
        derivative1.square() * route_weights[:, None]
    ).sum(dim=0, dtype=torch.float64) / denominator * output_energy
    m33 = (
        derivative3.square() * route_weights[:, None]
    ).sum(dim=0, dtype=torch.float64) / denominator * output_energy
    m13 = (
        derivative1 * derivative3 * route_weights[:, None]
    ).sum(dim=0, dtype=torch.float64) / denominator * output_energy
    del gate_pre, up_pre, gate_value, up_value, derivative1, derivative3

    permutation = permutation.to(device=w1.device, dtype=torch.long)
    m11 = m11.index_select(0, permutation).float().reshape(192, 16)
    m33 = m33.index_select(0, permutation).float().reshape(192, 16)
    m13 = m13.index_select(0, permutation).float().reshape(192, 16)
    h13 = h13.to(device=w1.device, dtype=torch.float32)
    input_blocks = h13.reshape(224, 16, 224, 16)
    input_index = torch.arange(224, device=w1.device)
    input_blocks = input_blocks[input_index, :, input_index, :].contiguous()
    hp = permutation.to(device=h2.device)
    h2_permuted = h2.index_select(0, hp).index_select(1, hp).float()
    middle_blocks = h2_permuted.reshape(192, 16, 192, 16)
    middle_index = torch.arange(192, device=w1.device)
    middle_blocks = middle_blocks[
        middle_index, :, middle_index, :
    ].contiguous()

    upstream: dict[int, torch.Tensor] = {}
    downstream: dict[int, torch.Tensor] = {}
    for bits in rates:
        r1 = uniform["w1"][bits]["reconstruction"]
        r3 = uniform["w3"][bits]["reconstruction"]
        r2 = uniform["w2"][bits]["reconstruction"]
        if not all(isinstance(value, torch.Tensor) for value in (r1, r3, r2)):
            raise TypeError("functional proxy needs tensor reconstructions")
        e1 = (r1 - w1).index_select(0, permutation).reshape(192, 16, 224, 16)
        e3 = (r3 - w3).index_select(0, permutation).reshape(192, 16, 224, 16)
        q11 = torch.einsum("bnti,tij,bntj->bnt", e1, input_blocks, e1)
        q33 = torch.einsum("bnti,tij,bntj->bnt", e3, input_blocks, e3)
        q13 = torch.einsum("bnti,tij,bntj->bnt", e1, input_blocks, e3)
        upstream[bits] = (
            m11[:, :, None] * q11
            + m33[:, :, None] * q33
            + 2.0 * m13[:, :, None] * q13
        ).sum(dim=1).T.contiguous()
        e2 = (
            (r2 - w2)
            .index_select(1, permutation)
            .T.reshape(192, 16, 224, 16)
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        downstream[bits] = torch.einsum(
            "toai,tij,toaj->to", e2, middle_blocks, e2
        ).contiguous()
    if any(value.shape != (224, 192) for value in upstream.values()):
        raise AssertionError("upstream functional proxy has the wrong tile grid")
    if any(value.shape != (192, 224) for value in downstream.values()):
        raise AssertionError("down functional proxy has the wrong tile grid")
    if any(
        not bool(torch.all(torch.isfinite(value))) or bool(torch.any(value < -1e-5))
        for errors in (upstream, downstream)
        for value in errors.values()
    ):
        raise ValueError("functional proxy produced invalid tile costs")
    return {"w13": upstream, "w2": downstream}


def _maximum_weight_pairing(scores: torch.Tensor) -> tuple[int, ...]:
    """Return the exact deterministic column assignment for a square matrix."""

    values = scores.detach().double().cpu()
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("pairing scores must form a square matrix")
    if values.shape[0] > 12:
        raise ValueError("pairing search supports at most 12 records")
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError("pairing scores must be finite")

    # State values are (score, assigned-column prefix).  At 12 records the
    # exact bitmask dynamic program contains only 4096 states.
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for row in range(values.shape[0]):
        updated: dict[int, tuple[float, tuple[int, ...]]] = {}
        for mask, (score, path) in states.items():
            for column in range(values.shape[1]):
                bit = 1 << column
                if mask & bit:
                    continue
                candidate = (score + float(values[row, column]), (*path, column))
                next_mask = mask | bit
                current = updated.get(next_mask)
                if current is None or candidate[0] > current[0] or (
                    candidate[0] == current[0] and candidate[1] < current[1]
                ):
                    updated[next_mask] = candidate
        states = updated
    return states[(1 << values.shape[0]) - 1][1]


def _p13_fractional_tile_map(
    errors: Mapping[int, torch.Tensor],
    record_rates: Sequence[int],
    *,
    rate_axis: str,
    pairing_policy: str = "maximum_weight",
    positive_fraction: float = 1.0,
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Choose P13 or P22 independently for paired 16x16 tiles.

    The supplied record allocation determines the K1 donor and K3 recipient
    records.  Donors and recipients are paired to maximize the sum of positive
    tile-level P13 benefits.  Every selected tile pair consumes the same 1024
    bits as P22, so the resulting map remains exactly two trellis bits per
    weight.
    """

    if rate_axis not in ("k", "n"):
        raise ValueError("fractional P13 map needs rate_axis 'k' or 'n'")
    if not 0.0 <= positive_fraction <= 1.0:
        raise ValueError("positive P13 tile fraction must lie in [0, 1]")
    if set(errors) != {1, 2, 3}:
        raise ValueError("fractional P13 needs K1, K2, and K3 tile costs")
    shape = errors[2].shape
    if any(value.shape != shape for value in errors.values()):
        raise ValueError("fractional P13 tile cost grids do not align")
    if any(not bool(torch.all(torch.isfinite(value))) for value in errors.values()):
        raise ValueError("fractional P13 tile costs must be finite")
    rates = tuple(int(rate) for rate in record_rates)
    if len(rates) != 24 or rates.count(1) != rates.count(3):
        raise ValueError("fractional P13 needs a balanced 24-record allocation")
    if any(rate not in (1, 2, 3) for rate in rates):
        raise ValueError("fractional P13 record rates must be K1, K2, or K3")

    tiles_k, tiles_n = shape
    rate_tiles = tiles_k if rate_axis == "k" else tiles_n
    if rate_tiles != 192:
        raise ValueError("Kimi fractional P13 needs 24 128-channel records")
    donors = tuple(index for index, rate in enumerate(rates) if rate == 1)
    recipients = tuple(index for index, rate in enumerate(rates) if rate == 3)
    rate_map = torch.full(shape, 2, dtype=torch.int8)
    if not donors:
        return tuple(int(value) for value in rate_map.flatten().tolist()), {
            "record_pairs": [],
            "pair_tiles": 0,
            "selected_pair_tiles": 0,
            "selected_fraction": 0.0,
            "selector_bits": 0,
            "proxy_benefit": 0.0,
        }

    pair_benefits: list[list[torch.Tensor]] = []
    pairing_scores = torch.empty((len(donors), len(recipients)), dtype=torch.float64)
    for donor_index, donor in enumerate(donors):
        row: list[torch.Tensor] = []
        donor_slice = slice(8 * donor, 8 * (donor + 1))
        for recipient_index, recipient in enumerate(recipients):
            recipient_slice = slice(8 * recipient, 8 * (recipient + 1))
            if rate_axis == "n":
                benefit = (
                    errors[2][:, donor_slice]
                    + errors[2][:, recipient_slice]
                    - errors[1][:, donor_slice]
                    - errors[3][:, recipient_slice]
                )
            else:
                benefit = (
                    errors[2][donor_slice, :]
                    + errors[2][recipient_slice, :]
                    - errors[1][donor_slice, :]
                    - errors[3][recipient_slice, :]
                )
            row.append(benefit)
            pairing_scores[donor_index, recipient_index] = benefit.clamp_min(0).sum(
                dtype=torch.float64
            )
        pair_benefits.append(row)

    if pairing_policy == "maximum_weight":
        assignment = _maximum_weight_pairing(pairing_scores)
    elif pairing_policy == "mirror":
        assignment = tuple(reversed(range(len(recipients))))
    else:
        raise ValueError(
            "fractional P13 pairing policy must be maximum_weight or mirror"
        )
    assigned_benefits = [
        pair_benefits[donor_index][recipient_index]
        for donor_index, recipient_index in enumerate(assignment)
    ]
    flat_benefits = torch.cat(
        [benefit.flatten() for benefit in assigned_benefits]
    )
    positive_count = int(torch.count_nonzero(flat_benefits > 0))
    selected_count = int(round(positive_count * positive_fraction))
    selected_flat = torch.zeros_like(flat_benefits, dtype=torch.bool)
    if selected_count:
        positive_indices = torch.nonzero(
            flat_benefits > 0,
            as_tuple=False,
        ).flatten()
        order = torch.argsort(
            flat_benefits.index_select(0, positive_indices),
            descending=True,
            stable=True,
        )
        selected_flat[positive_indices.index_select(0, order[:selected_count])] = True
    selected_total = 0
    tile_total = 0
    proxy_benefit = 0.0
    record_pairs = []
    selected_offset = 0
    for donor_index, recipient_index in enumerate(assignment):
        donor = donors[donor_index]
        recipient = recipients[recipient_index]
        benefit = pair_benefits[donor_index][recipient_index]
        selected = selected_flat[
            selected_offset : selected_offset + benefit.numel()
        ].reshape_as(benefit)
        selected_offset += benefit.numel()
        selected_total += int(selected.sum())
        tile_total += selected.numel()
        proxy_benefit += float(benefit[selected].sum(dtype=torch.float64))
        donor_slice = slice(8 * donor, 8 * (donor + 1))
        recipient_slice = slice(8 * recipient, 8 * (recipient + 1))
        if rate_axis == "n":
            selected_cpu = selected.cpu()
            donor_values = rate_map[:, donor_slice]
            recipient_values = rate_map[:, recipient_slice]
            donor_values[selected_cpu] = 1
            recipient_values[selected_cpu] = 3
        else:
            selected_cpu = selected.cpu()
            donor_values = rate_map[donor_slice, :]
            recipient_values = rate_map[recipient_slice, :]
            donor_values[selected_cpu] = 1
            recipient_values[selected_cpu] = 3
        record_pairs.append(
            {
                "donor": donor,
                "recipient": recipient,
                "pair_tiles": selected.numel(),
                "selected_pair_tiles": int(selected.sum()),
                "proxy_benefit": float(
                    benefit[selected].sum(dtype=torch.float64)
                ),
            }
        )

    flat = tuple(int(value) for value in rate_map.flatten().tolist())
    if sum(flat) != 2 * len(flat):
        raise AssertionError("fractional P13 map does not retain two-bit mean")
    return flat, {
        "pairing_policy": pairing_policy,
        "record_pairs": record_pairs,
        "pair_tiles": tile_total,
        "selected_pair_tiles": selected_total,
        "selected_fraction": selected_total / tile_total,
        "positive_pair_tiles": positive_count,
        "selected_positive_fraction": positive_fraction,
        "selector_bits": tile_total,
        "proxy_benefit": proxy_benefit,
    }


def _p13_record_deltas_from_tile_errors(
    errors: Mapping[int, torch.Tensor],
    *,
    rate_axis: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate dense-H tile costs into K1 donor and K3 recipient deltas."""

    return _p13_channel_group_deltas_from_tile_errors(
        errors,
        rate_axis=rate_axis,
        channels_per_group=128,
    )


def _p13_channel_group_deltas_from_tile_errors(
    errors: Mapping[int, torch.Tensor],
    *,
    rate_axis: str,
    channels_per_group: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate dense-H tile costs over fixed channel groups."""

    if rate_axis not in ("k", "n"):
        raise ValueError("P13 channel deltas need rate_axis 'k' or 'n'")
    if set(errors) != {1, 2, 3}:
        raise ValueError("P13 channel deltas need K1, K2, and K3 tile costs")
    shape = errors[2].shape
    if any(value.shape != shape for value in errors.values()):
        raise ValueError("P13 channel tile grids do not align")
    if channels_per_group <= 0 or channels_per_group % 16:
        raise ValueError("P13 channel groups must contain whole 16-channel tiles")
    tiles_k, tiles_n = shape
    rate_tiles = tiles_k if rate_axis == "k" else tiles_n
    group_tiles = channels_per_group // 16
    if rate_tiles != 192 or rate_tiles % group_tiles:
        raise ValueError("Kimi P13 channel groups must partition 3072 channels")

    donor_tiles = errors[1].double() - errors[2].double()
    recipient_tiles = errors[3].double() - errors[2].double()
    if rate_axis == "n":
        donor = donor_tiles.reshape(
            tiles_k, rate_tiles // group_tiles, group_tiles
        ).sum(dim=(0, 2))
        recipient = recipient_tiles.reshape(
            tiles_k, rate_tiles // group_tiles, group_tiles
        ).sum(dim=(0, 2))
    else:
        donor = donor_tiles.reshape(
            rate_tiles // group_tiles, group_tiles, tiles_n
        ).sum(dim=(1, 2))
        recipient = recipient_tiles.reshape(
            rate_tiles // group_tiles, group_tiles, tiles_n
        ).sum(dim=(1, 2))
    return donor, recipient


def _p13_tile_pair_diagnostics(
    errors: Mapping[int, torch.Tensor],
) -> dict[str, object]:
    """Report exact profitable P13 pair statistics for 16x16 tiles."""

    if set(errors) != {1, 2, 3}:
        raise ValueError("P13 tile diagnostics need K1, K2, and K3 tile costs")
    shape = errors[2].shape
    if any(value.shape != shape for value in errors.values()):
        raise ValueError("P13 tile cost grids do not align")
    donor = (errors[1].double() - errors[2].double()).cpu().flatten()
    gain = (errors[2].double() - errors[3].double()).cpu().flatten()
    if donor.numel() < 2 or not bool(torch.all(torch.isfinite(donor))) or not bool(
        torch.all(torch.isfinite(gain))
    ):
        raise ValueError("P13 tile costs must be finite and nontrivial")

    donor_order = torch.argsort(donor, stable=True)
    least = int(donor_order[0])
    second = int(donor_order[1])
    unit_ids = torch.arange(donor.numel())
    best_other_donor = torch.where(
        unit_ids == least,
        donor[second],
        donor[least],
    )
    margins = gain - best_other_donor
    recipient = int(torch.argmax(margins))
    selected_donor = second if recipient == least else least

    sorted_donor = torch.sort(donor, stable=True).values
    profitable_by_recipient = torch.searchsorted(
        sorted_donor,
        gain,
        right=False,
    ).to(torch.int64)
    profitable_by_recipient -= (donor < gain).to(torch.int64)
    profitable_pairs = int(profitable_by_recipient.sum())
    return {
        "units": int(donor.numel()),
        "maximum_pair_margin": float(margins[recipient]),
        "unconstrained_profitable_pairs": profitable_pairs,
        "best_pair": {
            "k1_tile": selected_donor,
            "k3_tile": recipient,
            "donor_cost": float(donor[selected_donor]),
            "recipient_gain": float(gain[recipient]),
        },
        "balanced_total": {
            "status": "not_computed",
            "reason": (
                "the exact disjoint balanced allocator is reserved for tile "
                "families with at least one profitable pair"
            ),
        },
    }


def _p13_marginal_diagnostics(
    donor_delta: torch.Tensor,
    recipient_delta: torch.Tensor,
) -> dict[str, object]:
    """Summarize additive K1 donor costs and K3 recipient gains."""

    donor = donor_delta.detach().double().cpu().flatten()
    gain = -recipient_delta.detach().double().cpu().flatten()
    if donor.shape != gain.shape or donor.numel() < 2:
        raise ValueError("P13 marginal tables must have equal nontrivial length")
    if not bool(torch.all(torch.isfinite(donor))) or not bool(
        torch.all(torch.isfinite(gain))
    ):
        raise ValueError("P13 marginal tables must be finite")

    def quantiles(values: torch.Tensor, points: Sequence[float]) -> dict[str, float]:
        result = torch.quantile(
            values,
            torch.tensor(tuple(points), dtype=torch.float64),
            interpolation="linear",
        )
        return {
            f"p{int(round(point * 100))}": float(value)
            for point, value in zip(points, result, strict=True)
        }

    pair_margin = gain[None, :] - donor[:, None]
    pair_margin.fill_diagonal_(float("-inf"))
    profitable = pair_margin > 0
    best = _best_balanced_p13_units(donor, -gain)
    best_delta = float(best["delta"])
    return {
        "units": int(donor.numel()),
        "donor_cost": {
            "minimum": float(donor.min()),
            **quantiles(donor, (0.01, 0.05, 0.10, 0.50)),
        },
        "recipient_gain": {
            **quantiles(gain, (0.90, 0.95, 0.99)),
            "maximum": float(gain.max()),
        },
        "maximum_pair_margin": float(pair_margin.max()),
        "unconstrained_profitable_pairs": int(profitable.sum()),
        "best_balanced_allocation": f"n{best['k1_count']}",
        "best_balanced_delta": best_delta,
        "best_balanced_gain": -best_delta,
        "best_balanced_units": best,
    }


def _fractional_pair_maps(
    errors_by_matrix: Sequence[Mapping[int, torch.Tensor]],
    *,
    rate_axis: str,
) -> tuple[dict[str, tuple[int, ...]], dict[str, object]]:
    if rate_axis not in ("k", "n"):
        raise ValueError("fractional pair map needs a matrix rate axis")
    shape = next(iter(errors_by_matrix[0].values())).shape
    if any(value.shape != shape for errors in errors_by_matrix for value in errors.values()):
        raise ValueError("fractional pair error grids do not align")
    tiles_k, tiles_n = shape
    rate_tiles = tiles_k if rate_axis == "k" else tiles_n
    if rate_tiles != 192:
        raise ValueError("Kimi fractional P24 expects a 3072-channel rate axis")
    low = torch.arange(8, device=next(iter(errors_by_matrix[0].values())).device)
    high = torch.arange(rate_tiles - 8, rate_tiles, device=low.device)
    benefits = None
    for errors in errors_by_matrix:
        if rate_axis == "n":
            local = (
                errors[3][:, low]
                + errors[3][:, high]
                - errors[2][:, low]
                - errors[4][:, high]
            )
        else:
            local = (
                errors[3][low, :]
                + errors[3][high, :]
                - errors[2][low, :]
                - errors[4][high, :]
            )
        benefits = local if benefits is None else benefits + local
    assert benefits is not None
    flat_benefits = benefits.flatten()
    order = torch.argsort(flat_benefits, descending=True, stable=True)
    pair_count = flat_benefits.numel()

    def build(selected: torch.Tensor) -> tuple[int, ...]:
        rate_map = torch.full((tiles_k, tiles_n), 3, dtype=torch.int8)
        selected_mask = torch.zeros(pair_count, dtype=torch.bool, device=selected.device)
        selected_mask[selected] = True
        selected_mask = selected_mask.reshape_as(benefits).cpu()
        if rate_axis == "n":
            rate_map[:, low.cpu()] = torch.where(
                selected_mask, torch.tensor(2, dtype=torch.int8), rate_map[:, low.cpu()]
            )
            rate_map[:, high.cpu()] = torch.where(
                selected_mask, torch.tensor(4, dtype=torch.int8), rate_map[:, high.cpu()]
            )
        else:
            rate_map[low.cpu(), :] = torch.where(
                selected_mask, torch.tensor(2, dtype=torch.int8), rate_map[low.cpu(), :]
            )
            rate_map[high.cpu(), :] = torch.where(
                selected_mask, torch.tensor(4, dtype=torch.int8), rate_map[high.cpu(), :]
            )
        return tuple(int(value) for value in rate_map.flatten().tolist())

    maps = {}
    for fraction in FRACTIONS:
        count = int(round(pair_count * fraction))
        maps[f"p{int(round(fraction * 100)):03d}"] = build(order[:count])
    positive = torch.nonzero(flat_benefits > 0).flatten()
    maps["positive"] = build(positive)
    evidence = {
        "pair_tiles": pair_count,
        "positive_pair_tiles": int(positive.numel()),
        "positive_fraction": float(positive.numel() / pair_count),
        "benefit_min": float(flat_benefits.min()),
        "benefit_median": float(flat_benefits.median()),
        "benefit_max": float(flat_benefits.max()),
    }
    return maps, evidence


def _tile_triplet_map(
    errors: Mapping[str, Mapping[int, torch.Tensor]],
) -> tuple[dict[str, tuple[int, ...]], dict[str, int]]:
    w1_shape = errors["w1"][3].shape
    if errors["w3"][3].shape != w1_shape or errors["w2"][3].shape != w1_shape[::-1]:
        raise ValueError("gate/up/down tile grids do not align by transpose")
    costs = []
    for w1_bits, w3_bits, w2_bits in TRIPLE_MODES:
        costs.append(
            errors["w1"][w1_bits]
            + errors["w3"][w3_bits]
            + errors["w2"][w2_bits].T
        )
    choices = torch.stack(costs).argmin(dim=0).cpu()
    w1_map = torch.empty(w1_shape, dtype=torch.int8)
    w3_map = torch.empty_like(w1_map)
    w2_map = torch.empty(w1_shape[::-1], dtype=torch.int8)
    counts = {}
    for index, mode in enumerate(TRIPLE_MODES):
        selected = choices == index
        counts["".join(map(str, mode))] = int(selected.sum())
        w1_map[selected] = mode[0]
        w3_map[selected] = mode[1]
        w2_map.T[selected] = mode[2]
    return (
        {
            "w1": tuple(int(value) for value in w1_map.flatten().tolist()),
            "w3": tuple(int(value) for value in w3_map.flatten().tolist()),
            "w2": tuple(int(value) for value in w2_map.flatten().tolist()),
        },
        counts,
    )


def _record_triplet_map(
    errors: Mapping[str, Mapping[int, torch.Tensor]],
) -> tuple[dict[str, tuple[int, ...]], dict[str, int]]:
    """Choose one 333/234-permutation mode per 128-channel record."""

    w1_shape = errors["w1"][3].shape
    if errors["w3"][3].shape != w1_shape or errors["w2"][3].shape != w1_shape[::-1]:
        raise ValueError("gate/up/down tile grids do not align by transpose")
    tiles_k, intermediate_tiles = w1_shape
    if intermediate_tiles != 192 or intermediate_tiles % 8:
        raise ValueError("Kimi record triples require 24 aligned 128-channel records")
    record_count = intermediate_tiles // 8
    mode_costs = torch.empty(
        (len(TRIPLE_MODES), record_count),
        dtype=torch.float64,
        device=errors["w1"][3].device,
    )
    for mode_index, (w1_bits, w3_bits, w2_bits) in enumerate(TRIPLE_MODES):
        tile_cost = (
            errors["w1"][w1_bits]
            + errors["w3"][w3_bits]
            + errors["w2"][w2_bits].T
        )
        mode_costs[mode_index] = tile_cost.reshape(
            tiles_k, record_count, 8
        ).sum(dim=(0, 2))
    choices = mode_costs.argmin(dim=0).cpu()
    w1_map = torch.empty(w1_shape, dtype=torch.int8)
    w3_map = torch.empty_like(w1_map)
    w2_map = torch.empty(w1_shape[::-1], dtype=torch.int8)
    counts = {}
    for index, mode in enumerate(TRIPLE_MODES):
        selected_records = torch.nonzero(choices == index).flatten()
        counts["".join(map(str, mode))] = int(selected_records.numel())
        for record in selected_records.tolist():
            tile_slice = slice(record * 8, (record + 1) * 8)
            w1_map[:, tile_slice] = mode[0]
            w3_map[:, tile_slice] = mode[1]
            w2_map[tile_slice, :] = mode[2]
    return (
        {
            "w1": tuple(int(value) for value in w1_map.flatten().tolist()),
            "w3": tuple(int(value) for value in w3_map.flatten().tolist()),
            "w2": tuple(int(value) for value in w2_map.flatten().tolist()),
        },
        counts,
    )


def _functional_record_triplet_map(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    uniform: Mapping[str, Mapping[int, Mapping[str, object]]],
    permutation: torch.Tensor,
    *,
    inputs: torch.Tensor,
    gates: torch.Tensor,
) -> tuple[dict[str, tuple[int, ...]], dict[str, int], dict[str, float]]:
    """Screen 24 record modes with exact spliced whole-expert fit loss.

    Gate/up output records are independent under the dense-H traversal.  The
    down-projection splice is an approximation because its input-axis LDLQ
    feedback is rerun only after the complete record map has been selected.
    """

    reconstructions: dict[str, dict[int, torch.Tensor]] = {}
    for matrix in MATRICES:
        reconstructions[matrix] = {}
        for bits in (2, 3, 4):
            value = uniform[matrix][bits]["reconstruction"]
            if not isinstance(value, torch.Tensor):
                raise TypeError("uniform candidate reconstruction is not a tensor")
            reconstructions[matrix][bits] = value

    gate_outputs = {
        bits: F.linear(inputs, reconstructions["w1"][bits])
        for bits in (2, 3, 4)
    }
    up_outputs = {
        bits: F.linear(inputs, reconstructions["w3"][bits])
        for bits in (2, 3, 4)
    }
    base_middle = situ(gate_outputs[3], up_outputs[3])
    base_w2 = reconstructions["w2"][3]
    base_output = F.linear(base_middle, base_w2)
    reference = F.linear(
        situ(F.linear(inputs, source[0]), F.linear(inputs, source[1])), source[2]
    )
    route_weights = gates.float().square().unsqueeze(1)

    choices = []
    individual_costs = torch.empty(
        (24, len(TRIPLE_MODES)), dtype=torch.float64, device=inputs.device
    )
    for record in range(24):
        indices = permutation[record * 128 : (record + 1) * 128]
        base_middle_slice = base_middle.index_select(1, indices)
        base_w2_slice = base_w2.index_select(1, indices)
        for mode_index, (w1_bits, w3_bits, w2_bits) in enumerate(TRIPLE_MODES):
            middle_slice = situ(
                gate_outputs[w1_bits].index_select(1, indices),
                up_outputs[w3_bits].index_select(1, indices),
            )
            candidate = base_output + F.linear(
                middle_slice - base_middle_slice, base_w2_slice
            )
            candidate += F.linear(
                middle_slice,
                reconstructions["w2"][w2_bits].index_select(1, indices)
                - base_w2_slice,
            )
            individual_costs[record, mode_index] = (
                (candidate - reference).square() * route_weights
            ).sum(dtype=torch.float64)
        choices.append(int(individual_costs[record].argmin()))

    w1_map = torch.empty((224, 192), dtype=torch.int8)
    w3_map = torch.empty_like(w1_map)
    w2_map = torch.empty((192, 224), dtype=torch.int8)
    combined = {
        matrix: reconstructions[matrix][3].clone() for matrix in MATRICES
    }
    counts = {"".join(map(str, mode)): 0 for mode in TRIPLE_MODES}
    for record, mode_index in enumerate(choices):
        mode = TRIPLE_MODES[mode_index]
        counts["".join(map(str, mode))] += 1
        tile_slice = slice(record * 8, (record + 1) * 8)
        w1_map[:, tile_slice] = mode[0]
        w3_map[:, tile_slice] = mode[1]
        w2_map[tile_slice, :] = mode[2]
        indices = permutation[record * 128 : (record + 1) * 128]
        combined["w1"].index_copy_(
            0, indices, reconstructions["w1"][mode[0]].index_select(0, indices)
        )
        combined["w3"].index_copy_(
            0, indices, reconstructions["w3"][mode[1]].index_select(0, indices)
        )
        combined["w2"].index_copy_(
            1, indices, reconstructions["w2"][mode[2]].index_select(1, indices)
        )

    baseline_sse = float(
        ((base_output - reference).square() * route_weights).sum(dtype=torch.float64)
    )
    combined_output = F.linear(
        situ(F.linear(inputs, combined["w1"]), F.linear(inputs, combined["w3"])),
        combined["w2"],
    )
    combined_sse = float(
        ((combined_output - reference).square() * route_weights).sum(
            dtype=torch.float64
        )
    )
    return (
        {
            "w1": tuple(int(value) for value in w1_map.flatten().tolist()),
            "w3": tuple(int(value) for value in w3_map.flatten().tolist()),
            "w2": tuple(int(value) for value in w2_map.flatten().tolist()),
        },
        counts,
        {
            "baseline_fit_sse": baseline_sse,
            "combined_splice_fit_sse": combined_sse,
            "combined_splice_relative_to_p33": _relative(
                combined_sse, baseline_sse
            ),
        },
    )


def _encode_triplet_map(
    rate_maps: Mapping[str, tuple[int, ...]],
    *,
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    uniform: Mapping[str, Mapping[int, Mapping[str, object]]],
    h13: torch.Tensor,
    h2: torch.Tensor,
    contexts: torch.Tensor,
    layer: int,
    expert: int,
    device: torch.device,
    quantizer_module,
    ldlq_tf32: bool,
    permutation_override: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstructions = []
    for matrix in MATRICES:
        shape = (3584, 3072) if matrix in ("w1", "w3") else (3072, 3584)
        candidates, _ = _quantize_maps(
            source[MATRICES.index(matrix)],
            h2 if matrix == "w2" else h13,
            contexts,
            matrix=matrix,
            maps={
                "p000": _uniform_tile_map(shape, 3),
                "triplet": rate_maps[matrix],
            },
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            g_scale_override=float(uniform[matrix][3]["g_scale"]),
            permutation_override=permutation_override,
        )
        reconstruction = candidates["triplet"]["reconstruction"]
        if not isinstance(reconstruction, torch.Tensor):
            raise TypeError("triplet candidate reconstruction is not a tensor")
        reconstructions.append(reconstruction)
    return tuple(reconstructions)


def _functional_totals(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    reconstruction: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    inputs: torch.Tensor,
    gates: torch.Tensor,
    request_steps: torch.Tensor,
    requests: Mapping[int, str],
) -> dict[str, float | int]:
    reference_middle = situ(F.linear(inputs, source[0]), F.linear(inputs, source[1]))
    reference = F.linear(reference_middle, source[2])
    candidate_middle = situ(
        F.linear(inputs, reconstruction[0]), F.linear(inputs, reconstruction[1])
    )
    candidate = F.linear(candidate_middle, reconstruction[2])
    sse, energy, counts = functional_sse_by_request(
        reference, candidate, gates, request_steps, requests
    )
    total_sse = float(sse.sum())
    total_energy = float(energy.sum())
    return {
        "sse": total_sse,
        "reference_energy": total_energy,
        "nmse": total_sse / total_energy,
        "rows": int(counts.sum()),
        "documents": int(torch.count_nonzero(counts)),
    }


def _score_candidate(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    reconstruction: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    inputs: torch.Tensor,
    gates: torch.Tensor,
    request_steps: torch.Tensor,
    fit_mask: torch.Tensor,
    confirmation_mask: torch.Tensor,
    fit_requests: Mapping[int, str],
    confirmation_requests: Mapping[int, str],
) -> dict[str, dict[str, float | int]]:
    return {
        "fit": _functional_totals(
            source,
            reconstruction,
            inputs=inputs[fit_mask],
            gates=gates[fit_mask],
            request_steps=request_steps[fit_mask],
            requests=fit_requests,
        ),
        "confirmation": _functional_totals(
            source,
            reconstruction,
            inputs=inputs[confirmation_mask],
            gates=gates[confirmation_mask],
            request_steps=request_steps[confirmation_mask],
            requests=confirmation_requests,
        ),
    }


def _functional_output_totals(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    gates: torch.Tensor,
    request_steps: torch.Tensor,
    requests: Mapping[int, str],
) -> dict[str, float | int]:
    sse, energy, counts = functional_sse_by_request(
        reference, candidate, gates, request_steps, requests
    )
    total_sse = float(sse.sum())
    total_energy = float(energy.sum())
    return {
        "sse": total_sse,
        "reference_energy": total_energy,
        "nmse": total_sse / total_energy,
        "rows": int(counts.sum()),
        "documents": int(torch.count_nonzero(counts)),
    }


def _encode_uniform_k2_triplet(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    h13: torch.Tensor,
    source_h2: torch.Tensor,
    contexts: torch.Tensor,
    layer: int,
    expert: int,
    device: torch.device,
    quantizer_module,
    ldlq_tf32: bool,
    fit_inputs: torch.Tensor,
    fit_gates: torch.Tensor,
    permutation_override: torch.Tensor,
    shared_scale_scope: str,
    middle_from_upstream,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, object]
]:
    maps = {
        "w1": _uniform_tile_map((3584, 3072), 2),
        "w3": _uniform_tile_map((3584, 3072), 2),
        "w2": _uniform_tile_map((3072, 3584), 2),
    }
    reconstructions: list[torch.Tensor] = []
    matrix_evidence: dict[str, object] = {}
    for matrix in ("w1", "w3"):
        candidates, _ = _quantize_maps(
            source[MATRICES.index(matrix)],
            h13,
            contexts,
            matrix=matrix,
            maps={"k2": maps[matrix]},
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            permutation_override=permutation_override,
            shared_scale_scope=shared_scale_scope,
        )
        candidate = candidates["k2"]
        reconstruction = candidate["reconstruction"]
        if not isinstance(reconstruction, torch.Tensor):
            raise TypeError("uniform K2 upstream reconstruction is missing")
        reconstructions.append(reconstruction)
        matrix_evidence[matrix] = {
            "g_scale": float(candidate["g_scale"]),
            "proxy": float(candidate["proxy"]),
            "permutation_sha256": candidate["permutation_sha256"],
        }

    candidate_middle = middle_from_upstream(
        reconstructions[0], reconstructions[1]
    )
    _, candidate_h2, h2_evidence = build_expert_hessians(
        fit_inputs,
        fit_gates,
        candidate_middle,
        global_h13=h13,
        global_h2=source_h2,
        device=device,
        h13_alpha=0.0,
    )
    down_candidates, _ = _quantize_maps(
        source[2],
        candidate_h2,
        contexts,
        matrix="w2",
        maps={"k2": maps["w2"]},
        layer=layer,
        expert=expert,
        device=device,
        quantizer_module=quantizer_module,
        ldlq_tf32=ldlq_tf32,
        permutation_override=permutation_override,
        shared_scale_scope=shared_scale_scope,
    )
    down = down_candidates["k2"]
    down_reconstruction = down["reconstruction"]
    if not isinstance(down_reconstruction, torch.Tensor):
        raise TypeError("uniform K2 down reconstruction is missing")
    reconstructions.append(down_reconstruction)
    matrix_evidence["w2"] = {
        "g_scale": float(down["g_scale"]),
        "proxy": float(down["proxy"]),
        "permutation_sha256": down["permutation_sha256"],
    }
    return (
        (reconstructions[0], reconstructions[1], reconstructions[2]),
        {
            "matrices": matrix_evidence,
            "conditional_h2": h2_evidence,
        },
    )


def _coupled_hadamard_k2(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    h13: torch.Tensor,
    source_h2: torch.Tensor,
    contexts: torch.Tensor,
    layer: int,
    expert: int,
    device: torch.device,
    quantizer_module,
    ldlq_tf32: bool,
    inputs: torch.Tensor,
    gates: torch.Tensor,
    request_steps: torch.Tensor,
    fit_mask: torch.Tensor,
    confirmation_mask: torch.Tensor,
    fit_requests: Mapping[int, str],
    confirmation_requests: Mapping[int, str],
    baseline_permutation: torch.Tensor,
    block_size: int,
    preactivation_block_size: int,
    postactivation_block_size: int,
    pre_permutation: str,
    residual_rotation_draw: int,
    intermediate_rotation_draw: int,
) -> dict[str, object]:
    """Compare production-basis and coupled-boundary uniform K2 encodes."""

    fit_inputs = inputs[fit_mask]
    fit_gates = gates[fit_mask]
    baseline, baseline_evidence = _encode_uniform_k2_triplet(
        source,
        h13=h13,
        source_h2=source_h2,
        contexts=contexts,
        layer=layer,
        expert=expert,
        device=device,
        quantizer_module=quantizer_module,
        ldlq_tf32=ldlq_tf32,
        fit_inputs=fit_inputs,
        fit_gates=fit_gates,
        permutation_override=baseline_permutation,
        shared_scale_scope="k2-coupled-control",
        middle_from_upstream=lambda w1, w3: situ(
            F.linear(fit_inputs, w1), F.linear(fit_inputs, w3)
        ),
    )

    basis = _prepare_coupled_search_basis(
        source,
        h13=h13,
        h2=source_h2,
        inputs=inputs,
        selected_permutation=baseline_permutation,
        block_size=block_size,
        preactivation_block_size=preactivation_block_size,
        postactivation_block_size=postactivation_block_size,
        pre_permutation=pre_permutation,
        residual_rotation_draw=residual_rotation_draw,
        intermediate_rotation_draw=intermediate_rotation_draw,
    )

    coupled, coupled_evidence = _encode_uniform_k2_triplet(
        basis.source,
        h13=basis.h13,
        source_h2=basis.h2,
        contexts=contexts,
        layer=layer,
        expert=expert,
        device=device,
        quantizer_module=quantizer_module,
        ldlq_tf32=ldlq_tf32,
        fit_inputs=basis.inputs[fit_mask],
        fit_gates=fit_gates,
        permutation_override=basis.permutation,
        shared_scale_scope=(
            f"k2-coupled-hadamard-b{block_size}-"
            f"pre{preactivation_block_size}-post{postactivation_block_size}-"
            f"{pre_permutation}"
        ),
        middle_from_upstream=lambda w1, w3: basis.decode_middle(
            basis.inputs[fit_mask], w1, w3
        ),
    )

    reference_middle = situ(
        F.linear(inputs, source[0]), F.linear(inputs, source[1])
    )
    reference = F.linear(reference_middle, source[2])
    baseline_middle = situ(
        F.linear(inputs, baseline[0]), F.linear(inputs, baseline[1])
    )
    baseline_output = F.linear(baseline_middle, baseline[2])
    coupled_output = basis.execute_triplet(basis.inputs, coupled)
    closure_output = basis.execute_triplet(basis.inputs, basis.source)

    def score(output: torch.Tensor) -> dict[str, dict[str, float | int]]:
        return {
            "fit": _functional_output_totals(
                reference[fit_mask],
                output[fit_mask],
                gates=gates[fit_mask],
                request_steps=request_steps[fit_mask],
                requests=fit_requests,
            ),
            "confirmation": _functional_output_totals(
                reference[confirmation_mask],
                output[confirmation_mask],
                gates=gates[confirmation_mask],
                request_steps=request_steps[confirmation_mask],
                requests=confirmation_requests,
            ),
        }

    baseline_score = score(baseline_output)
    coupled_score = score(coupled_output)
    closure_error = float(
        (closure_output.double() - reference.double()).square().sum()
        / reference.double().square().sum().clamp_min(1e-30)
    )
    return {
        "rate": "uniform_k2",
        "block_size": block_size,
        "preactivation_block_size": preactivation_block_size,
        "postactivation_block_size": postactivation_block_size,
        "baseline_basis": "production_h2_reverse_neuron_permutation",
        "pre_hadamard_neuron_permutation": pre_permutation,
        "pre_hadamard_permutation_sha256": basis.evidence[
            "pre_hadamard_permutation_sha256"
        ],
        "residual_rotation_draw": residual_rotation_draw,
        "intermediate_rotation_draw": intermediate_rotation_draw,
        "coupled_basis": (
            "interleaved_gate_up_two_sided_block_hadamard_with_identity_"
            "transformed_coordinate_order"
        ),
        "runtime_boundary_transforms": {
            "foldable_shared": ["expert_input", "expert_output"],
            "per_selected_expert": ["pre_situ_inverse", "post_situ_forward"],
        },
        "conditional_h2": (
            "candidate_local_decoded_upstream_in_each_representation_basis_"
            "with_adaptive_scaled_identity_shrinkage"
        ),
        "full_precision_closure_relative_sse": closure_error,
        "scores": {
            "baseline": baseline_score,
            "coupled_hadamard": coupled_score,
        },
        "evidence": {
            "baseline": baseline_evidence,
            "coupled_hadamard": coupled_evidence,
        },
        "fit_relative_to_baseline": _relative(
            float(coupled_score["fit"]["sse"]),
            float(baseline_score["fit"]["sse"]),
        ),
        "confirmation_relative_to_baseline": _relative(
            float(coupled_score["confirmation"]["sse"]),
            float(baseline_score["confirmation"]["sse"]),
        ),
    }


def _h2_viterbi_refinement(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    global_h13: torch.Tensor,
    global_h2: torch.Tensor,
    contexts: torch.Tensor,
    layer: int,
    expert: int,
    device: torch.device,
    quantizer_module,
    ldlq_tf32: bool,
    inputs: torch.Tensor,
    gates: torch.Tensor,
    request_steps: torch.Tensor,
    baseline_permutation: torch.Tensor,
    block_size: int,
    preactivation_block_size: int,
    postactivation_block_size: int,
    pre_permutation: str,
    residual_rotation_draw: int,
    intermediate_rotation_draw: int,
    refine_sweeps: int,
) -> dict[str, object]:
    """Refine canonical W2 K2 paths under its complete routed-row H2.

    The baseline and refined W2 candidates share source weights, decoded
    upstream activations, dense H2, scales, transforms, and initial BlockLDLQ
    path.  Only the refined candidate receives conditional-target sweeps.
    Every routed row available for the expert contributes to H2 and to the
    mapped-output measurement; no document partition selects either payload.
    """

    basis = _prepare_coupled_search_basis(
        source,
        h13=global_h13,
        h2=global_h2,
        inputs=inputs,
        selected_permutation=baseline_permutation,
        block_size=block_size,
        preactivation_block_size=preactivation_block_size,
        postactivation_block_size=postactivation_block_size,
        pre_permutation=pre_permutation,
        residual_rotation_draw=residual_rotation_draw,
        intermediate_rotation_draw=intermediate_rotation_draw,
    )
    uniform_maps = {
        "w1": _uniform_tile_map((3584, 3072), 2),
        "w3": _uniform_tile_map((3584, 3072), 2),
        "w2": _uniform_tile_map((3072, 3584), 2),
    }
    upstream = []
    upstream_evidence: dict[str, object] = {}
    for matrix in ("w1", "w3"):
        candidates, _ = _quantize_maps(
            basis.source[MATRICES.index(matrix)],
            basis.h13,
            contexts,
            matrix=matrix,
            maps={"k2": uniform_maps[matrix]},
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            permutation_override=basis.permutation,
            shared_scale_scope="h2-viterbi-refinement",
        )
        candidate = candidates["k2"]
        reconstruction = candidate["reconstruction"]
        if not isinstance(reconstruction, torch.Tensor):
            raise TypeError("uniform K2 upstream reconstruction is missing")
        upstream.append(reconstruction)
        upstream_evidence[matrix] = {
            "g_scale": float(candidate["g_scale"]),
            "proxy": float(candidate["proxy"]),
            "permutation_sha256": candidate["permutation_sha256"],
        }

    decoded_middle = basis.decode_middle(basis.inputs, upstream[0], upstream[1])
    _, candidate_h2, h2_evidence = build_expert_hessians(
        basis.inputs,
        gates,
        decoded_middle,
        global_h13=basis.h13,
        global_h2=basis.h2,
        device=device,
        h13_alpha=0.0,
    )
    down_candidates, _ = _quantize_maps(
        basis.source[2],
        candidate_h2,
        contexts,
        matrix="w2",
        maps={
            "baseline": uniform_maps["w2"],
            "refined": uniform_maps["w2"],
        },
        layer=layer,
        expert=expert,
        device=device,
        quantizer_module=quantizer_module,
        ldlq_tf32=ldlq_tf32,
        permutation_override=basis.permutation,
        shared_scale_scope="h2-viterbi-refinement",
        h2_viterbi_refine_sweeps={"refined": refine_sweeps},
    )
    baseline = down_candidates["baseline"]
    refined = down_candidates["refined"]
    baseline_down = baseline["reconstruction"]
    refined_down = refined["reconstruction"]
    baseline_states = baseline["states"]
    refined_states = refined["states"]
    if not all(
        isinstance(value, torch.Tensor)
        for value in (
            baseline_down,
            refined_down,
            baseline_states,
            refined_states,
        )
    ):
        raise TypeError("W2 refinement candidates lack decoded tensors")
    if float(baseline["g_scale"]) != float(refined["g_scale"]):
        raise AssertionError("controlled W2 candidates chose different scales")
    if not torch.equal(baseline["suh"], refined["suh"]) or not torch.equal(
        baseline["svh"], refined["svh"]
    ):
        raise AssertionError("controlled W2 candidates chose different transforms")

    baseline_triplet = (upstream[0], upstream[1], baseline_down)
    refined_triplet = (upstream[0], upstream[1], refined_down)
    baseline_output = basis.execute_triplet(basis.inputs, baseline_triplet)
    refined_output = basis.execute_triplet(basis.inputs, refined_triplet)
    reference = basis.reference_output
    route_weights = gates.double().square()

    def output_score(output: torch.Tensor) -> dict[str, float]:
        error = output.double() - reference.double()
        row_sse = error.square().sum(dim=1)
        reference_energy = reference.double().square().sum(dim=1)
        routed_sse = float((row_sse * route_weights).sum())
        routed_energy = float((reference_energy * route_weights).sum())
        return {
            "routed_sse": routed_sse,
            "routed_reference_energy": routed_energy,
            "routed_nmse": routed_sse / max(routed_energy, 1.0e-30),
            "unweighted_sse": float(row_sse.sum()),
        }

    baseline_score = output_score(baseline_output)
    refined_score = output_score(refined_output)
    receipt = refined["h2_viterbi_refine"]
    if not isinstance(receipt, Mapping) or not receipt.get("enabled"):
        raise AssertionError("refined W2 candidate lacks its refinement receipt")
    sweeps = receipt.get("sweeps")
    if not isinstance(sweeps, list) or len(sweeps) != refine_sweeps:
        raise AssertionError(
            "W2 refinement receipt does not match the requested sweep count"
        )
    proxy_relative = _relative(
        float(refined["proxy"]), float(baseline["proxy"])
    )
    objective_relative = _relative(
        float(sweeps[-1]["objective_after"]),
        float(sweeps[0]["objective_before"]),
    )
    changed = baseline_states != refined_states
    changed_tiles = changed.reshape(*changed.shape[:2], -1).any(dim=2)
    return {
        "rate": "uniform_k2",
        "scope": "canonical_w2_target_only",
        "row_population": "all_available_routed_rows",
        "rows": int(inputs.shape[0]),
        "documents": int(torch.unique(request_steps).numel()),
        "basis": basis.evidence,
        "upstream": upstream_evidence,
        "conditional_h2": h2_evidence,
        "baseline": {
            "dense_h_proxy": float(baseline["proxy"]),
            "mapped_output": baseline_score,
        },
        "refined": {
            "dense_h_proxy": float(refined["proxy"]),
            "mapped_output": refined_score,
            "receipt": receipt,
        },
        "dense_h_relative_to_baseline": proxy_relative,
        "refiner_objective_relative_to_baseline": objective_relative,
        "proxy_objective_ratio_closure_absolute": abs(
            proxy_relative - objective_relative
        ),
        "mapped_output_relative_to_baseline": _relative(
            refined_score["routed_sse"], baseline_score["routed_sse"]
        ),
        "path_changes": {
            "tiles": int(torch.count_nonzero(changed_tiles)),
            "tile_fraction": float(changed_tiles.float().mean()),
            "scalar_codes": int(torch.count_nonzero(changed)),
            "scalar_code_fraction": float(changed.float().mean()),
        },
        "controlled_candidate_closure": {
            "same_global_scale": True,
            "same_input_transform": True,
            "same_output_transform": True,
        },
    }


def _prepare_coupled_search_basis(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    h13: torch.Tensor,
    h2: torch.Tensor,
    inputs: torch.Tensor,
    selected_permutation: torch.Tensor,
    block_size: int,
    preactivation_block_size: int,
    postactivation_block_size: int,
    pre_permutation: str,
    residual_rotation_draw: int = 0,
    intermediate_rotation_draw: int = 0,
) -> CoupledSearchBasis:
    """Build exact coupled-Hadamard coordinates for mixed-rate research.

    Quantization and candidate-local H2 fitting happen entirely in the
    transformed execution basis.  Candidate outputs are mapped back to the
    ordinary expert basis before fit or confirmation SSE is accumulated.
    """

    source_triplet = CoupledTriplet(*source)
    if pre_permutation == "identity":
        permutation = torch.arange(
            source_triplet.intermediate,
            dtype=torch.long,
            device=inputs.device,
        )
        permuted_triplet = source_triplet
    elif pre_permutation == "selected":
        permutation = selected_permutation.to(
            device=inputs.device, dtype=torch.long
        )
        permuted_triplet = apply_permutation_sign_gauge(
            source_triplet,
            permutation,
            torch.ones(
                source_triplet.intermediate,
                dtype=source_triplet.up.dtype,
                device=inputs.device,
            ),
        )
    else:
        raise ValueError(f"unsupported coupled pre-permutation: {pre_permutation}")

    encoded = encode_coupled_block_hadamard(
        permuted_triplet,
        block_size=block_size,
        preactivation_block_size=preactivation_block_size,
        postactivation_block_size=postactivation_block_size,
        residual_rotation_draw=residual_rotation_draw,
        intermediate_rotation_draw=intermediate_rotation_draw,
    )
    input_signs = hadamard_rotation_signs(
        source_triplet.hidden,
        draw=residual_rotation_draw,
        axis=0,
        device=inputs.device,
    )
    preactivation_signs = hadamard_rotation_signs(
        2 * source_triplet.intermediate,
        draw=intermediate_rotation_draw,
        axis=1,
        device=inputs.device,
    )
    postactivation_signs = hadamard_rotation_signs(
        source_triplet.intermediate,
        draw=intermediate_rotation_draw,
        axis=2,
        device=inputs.device,
    )
    output_signs = hadamard_rotation_signs(
        source_triplet.hidden,
        draw=residual_rotation_draw,
        axis=3,
        device=inputs.device,
    )
    def transform_inputs(rows: torch.Tensor) -> torch.Tensor:
        return signed_block_hadamard(
            rows,
            block_size=block_size,
            signs=input_signs,
            dim=1,
        )

    transformed_inputs = transform_inputs(inputs)
    transformed_h13 = signed_block_hadamard(
        signed_block_hadamard(
            h13,
            block_size=block_size,
            signs=input_signs,
            dim=0,
        ),
        block_size=block_size,
        signs=input_signs,
        dim=1,
    )
    h2_permutation = permutation.to(device=h2.device)
    permuted_h2 = h2.index_select(0, h2_permutation).index_select(
        1, h2_permutation
    )
    transformed_h2 = signed_block_hadamard(
        signed_block_hadamard(
            permuted_h2,
            block_size=postactivation_block_size,
            signs=postactivation_signs,
            dim=0,
        ),
        block_size=postactivation_block_size,
        signs=postactivation_signs,
        dim=1,
    )
    transformed_identity = torch.arange(
        source_triplet.intermediate,
        dtype=torch.long,
        device=inputs.device,
    )

    def decode_middle(
        transformed_rows: torch.Tensor,
        w1: torch.Tensor,
        w3: torch.Tensor,
    ) -> torch.Tensor:
        transformed_pre = torch.cat(
            (
                F.linear(transformed_rows, w1),
                F.linear(transformed_rows, w3),
            ),
            dim=1,
        )
        recovered = signed_block_hadamard(
            transformed_pre,
            block_size=preactivation_block_size,
            signs=preactivation_signs,
            dim=1,
            inverse=True,
        )
        middle = situ(recovered[:, 0::2], recovered[:, 1::2])
        return signed_block_hadamard(
            middle,
            block_size=postactivation_block_size,
            signs=postactivation_signs,
            dim=1,
        )

    def execute_triplet(
        transformed_rows: torch.Tensor,
        reconstruction: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        transformed_middle = decode_middle(
            transformed_rows, reconstruction[0], reconstruction[1]
        )
        transformed_output = F.linear(
            transformed_middle, reconstruction[2]
        )
        return signed_block_hadamard(
            transformed_output,
            block_size=block_size,
            signs=output_signs,
            dim=1,
            inverse=True,
        )

    reference_output = _execute_standard_triplet(inputs, source)
    closure_output = execute_triplet(transformed_inputs, encoded.tensors())
    closure_relative_sse = float(
        (closure_output.double() - reference_output.double()).square().sum()
        / reference_output.double().square().sum().clamp_min(1e-30)
    )
    return CoupledSearchBasis(
        source=encoded.tensors(),
        h13=transformed_h13,
        h2=transformed_h2,
        inputs=transformed_inputs,
        permutation=transformed_identity,
        reference_output=reference_output,
        transform_inputs=transform_inputs,
        decode_middle=decode_middle,
        execute_triplet=execute_triplet,
        evidence={
            "basis": "coupled_interleaved_block_hadamard",
            "block_size": block_size,
            "preactivation_block_size": preactivation_block_size,
            "postactivation_block_size": postactivation_block_size,
            "residual_rotation_draw": residual_rotation_draw,
            "intermediate_rotation_draw": intermediate_rotation_draw,
            "pre_hadamard_neuron_permutation": pre_permutation,
            "pre_hadamard_permutation_sha256": _permutation_sha256(
                permutation
            ),
            "encoder_permutation": "identity_in_transformed_coordinates",
            "full_precision_closure_relative_sse": closure_relative_sse,
            "conditional_h2": (
                "decoded_candidate_post_situ_in_transformed_coordinates_"
                "with_adaptive_scaled_identity_shrinkage"
            ),
            "runtime_boundary_transforms": {
                "foldable_shared": ["expert_input", "expert_output"],
                "per_selected_expert": [
                    "pre_situ_inverse",
                    "post_situ_forward",
                ],
            },
        },
    )


def _k2_menu_selector_stats(
    errors: Mapping[str, torch.Tensor],
    *,
    laws: Sequence[str],
) -> tuple[torch.Tensor, dict[str, object]]:
    """Select a shared codebook ID for each corresponding matrix tile triplet."""

    if tuple(laws)[0] != "normal" or set(laws) - set(K2_MENU_LAWS):
        raise ValueError("K2 tile menus must begin with the normal law")
    shape = errors["normal"].shape
    if shape != (224, 192) or any(errors[law].shape != shape for law in laws):
        raise ValueError("K2 tile-menu costs must use the 224x192 triplet grid")
    stacked = torch.stack(tuple(errors[law] for law in laws), dim=0)
    selected = stacked.argmin(dim=0)
    selected_cost = stacked.gather(0, selected.unsqueeze(0)).squeeze(0)
    baseline_cost = stacked[0]
    all_equal = stacked.max(dim=0).values == stacked.min(dim=0).values
    counts = {
        law: int(torch.count_nonzero(selected == index))
        for index, law in enumerate(laws)
    }
    total = selected.numel()
    changed = selected != 0
    return selected, {
        "laws": list(laws),
        "selection": "minimum_combined_regularized_weight_tile_sse",
        "tile_count": total,
        "mode_counts": counts,
        "mode_fractions": {law: count / total for law, count in counts.items()},
        "non_normal_tiles": int(torch.count_nonzero(changed)),
        "non_normal_fraction": float(changed.float().mean()),
        "proxy_relative_to_normal": _relative(
            float(selected_cost.sum()), float(baseline_cost.sum())
        ),
        "proxy_gain_tiles": int(torch.count_nonzero(selected_cost < baseline_cost)),
        "proxy_unchanged_tiles": int(
            torch.count_nonzero(selected_cost == baseline_cost)
        ),
        "all_laws_tie_tiles": int(torch.count_nonzero(all_equal)),
    }


def _k2_codebook_menu(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    switched_luts: Mapping[str, Mapping[int, torch.Tensor]],
    *,
    h13: torch.Tensor,
    source_h2: torch.Tensor,
    contexts: torch.Tensor,
    layer: int,
    expert: int,
    device: torch.device,
    quantizer_module,
    ldlq_tf32: bool,
    inputs: torch.Tensor,
    gates: torch.Tensor,
    request_steps: torch.Tensor,
    fit_mask: torch.Tensor,
    confirmation_mask: torch.Tensor,
    fit_requests: Mapping[int, str],
    confirmation_requests: Mapping[int, str],
    permutation_override: torch.Tensor,
    decode_middle: MiddleDecoder | None = None,
    execute_triplet: TripletExecutor | None = None,
    reference_output: torch.Tensor | None = None,
    representation: Mapping[str, object] | None = None,
    external_inputs: torch.Tensor | None = None,
    external_gates: torch.Tensor | None = None,
    external_request_steps: torch.Tensor | None = None,
    external_requests: Mapping[int, str] | None = None,
    external_reference_output: torch.Tensor | None = None,
) -> dict[str, object]:
    """Measure a two-bit K2 staircase menu with an exact selector-aware encode.

    Proposal costs use one independently fitted normal-K2 scale per matrix and
    hold that scale fixed across laws.  The chosen map is then encoded again
    from the source weights, with its own fitted scale.  W2 always uses an
    expert-local H2 constructed from the decoded upstream candidate it follows.
    """

    missing = set(K2_MENU_LAWS) - set(switched_luts)
    if missing:
        raise ValueError(f"missing K2 menu laws: {sorted(missing)}")
    middle_decoder = (
        _decode_standard_middle if decode_middle is None else decode_middle
    )
    triplet_executor = (
        _execute_standard_triplet if execute_triplet is None else execute_triplet
    )
    if reference_output is None:
        reference_output = triplet_executor(inputs, source)
    if reference_output.shape != (inputs.shape[0], source[2].shape[0]):
        raise ValueError("K2 menu reference output does not align with routed rows")
    external_values = (
        external_inputs,
        external_gates,
        external_request_steps,
        external_requests,
        external_reference_output,
    )
    if any(value is not None for value in external_values) and not all(
        value is not None for value in external_values
    ):
        raise ValueError("external K2 menu scoring requires complete row metadata")
    if (
        external_inputs is not None
        and external_reference_output is not None
        and external_reference_output.shape
        != (external_inputs.shape[0], source[2].shape[0])
    ):
        raise ValueError("external K2 menu reference output does not align")

    def score_candidate(
        reconstruction: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, dict[str, float | int]]:
        output = triplet_executor(inputs, reconstruction)
        result = {
            "fit": _functional_output_totals(
                reference_output[fit_mask],
                output[fit_mask],
                gates=gates[fit_mask],
                request_steps=request_steps[fit_mask],
                requests=fit_requests,
            ),
            "confirmation": _functional_output_totals(
                reference_output[confirmation_mask],
                output[confirmation_mask],
                gates=gates[confirmation_mask],
                request_steps=request_steps[confirmation_mask],
                requests=confirmation_requests,
            ),
        }
        if external_inputs is not None:
            assert external_gates is not None
            assert external_request_steps is not None
            assert external_requests is not None
            assert external_reference_output is not None
            external_output = triplet_executor(external_inputs, reconstruction)
            result["external"] = _functional_output_totals(
                external_reference_output,
                external_output,
                gates=external_gates,
                request_steps=external_request_steps,
                requests=external_requests,
            )
        return result

    fit_inputs = inputs[fit_mask]
    fit_gates = gates[fit_mask]
    uniform_maps = {
        "w1": _uniform_tile_map((3584, 3072), 2),
        "w3": _uniform_tile_map((3584, 3072), 2),
        "w2": _uniform_tile_map((3072, 3584), 2),
    }
    prepared = {
        matrix: _prepare_quantize_maps(
            source[MATRICES.index(matrix)],
            h13,
            contexts,
            matrix=matrix,
            device=device,
            permutation_override=permutation_override,
        )
        for matrix in ("w1", "w3")
    }
    encoded: dict[str, dict[str, dict[str, object]]] = {
        matrix: {} for matrix in MATRICES
    }
    targets: dict[str, torch.Tensor] = {}
    tile_errors: dict[str, dict[str, torch.Tensor]] = {
        matrix: {} for matrix in MATRICES
    }

    for matrix in ("w1", "w3"):
        normal, encoder_weight = _quantize_maps(
            source[MATRICES.index(matrix)],
            h13,
            contexts,
            matrix=matrix,
            maps={"normal": uniform_maps[matrix]},
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            luts_by_bits=switched_luts["normal"],
            prepared=prepared[matrix],
        )
        encoded[matrix]["normal"] = normal["normal"]
        targets[matrix] = _target_regularized_weight(
            encoder_weight,
            normal["normal"]["suh"],
            normal["normal"]["svh"],
        )
        normal_scale = float(normal["normal"]["g_scale"])
        for law in K2_MENU_LAWS[1:]:
            candidate, _ = _quantize_maps(
                source[MATRICES.index(matrix)],
                h13,
                contexts,
                matrix=matrix,
                maps={law: uniform_maps[matrix]},
                layer=layer,
                expert=expert,
                device=device,
                quantizer_module=quantizer_module,
                ldlq_tf32=ldlq_tf32,
                g_scale_override=normal_scale,
                luts_by_bits=switched_luts[law],
                prepared=prepared[matrix],
            )
            encoded[matrix][law] = candidate[law]
        target_tiles = _tile_view(targets[matrix])
        for law in K2_MENU_LAWS:
            regularized = encoded[matrix][law]["regularized"]
            if not isinstance(regularized, torch.Tensor):
                raise TypeError("uniform K2 menu candidate lacks regularized weight")
            tile_errors[matrix][law] = (
                (_tile_view(regularized) - target_tiles).square().sum(dim=2)
            )

    normal_w1 = encoded["w1"]["normal"]["reconstruction"]
    normal_w3 = encoded["w3"]["normal"]["reconstruction"]
    if not isinstance(normal_w1, torch.Tensor) or not isinstance(
        normal_w3, torch.Tensor
    ):
        raise TypeError("normal K2 upstream reconstruction is missing")
    normal_middle = middle_decoder(fit_inputs, normal_w1, normal_w3)
    _, normal_h2, normal_h2_evidence = build_expert_hessians(
        fit_inputs,
        fit_gates,
        normal_middle,
        global_h13=h13,
        global_h2=source_h2,
        device=device,
    )
    prepared_w2 = _prepare_quantize_maps(
        source[2],
        normal_h2,
        contexts,
        matrix="w2",
        device=device,
        permutation_override=permutation_override,
    )
    normal, encoder_weight = _quantize_maps(
        source[2],
        normal_h2,
        contexts,
        matrix="w2",
        maps={"normal": uniform_maps["w2"]},
        layer=layer,
        expert=expert,
        device=device,
        quantizer_module=quantizer_module,
        ldlq_tf32=ldlq_tf32,
        luts_by_bits=switched_luts["normal"],
        prepared=prepared_w2,
    )
    encoded["w2"]["normal"] = normal["normal"]
    targets["w2"] = _target_regularized_weight(
        encoder_weight,
        normal["normal"]["suh"],
        normal["normal"]["svh"],
    )
    normal_scale = float(normal["normal"]["g_scale"])
    for law in K2_MENU_LAWS[1:]:
        candidate, _ = _quantize_maps(
            source[2],
            normal_h2,
            contexts,
            matrix="w2",
            maps={law: uniform_maps["w2"]},
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            g_scale_override=normal_scale,
            luts_by_bits=switched_luts[law],
            prepared=prepared_w2,
        )
        encoded["w2"][law] = candidate[law]
    target_tiles = _tile_view(targets["w2"])
    for law in K2_MENU_LAWS:
        regularized = encoded["w2"][law]["regularized"]
        if not isinstance(regularized, torch.Tensor):
            raise TypeError("uniform K2 down candidate lacks regularized weight")
        tile_errors["w2"][law] = (
            (_tile_view(regularized) - target_tiles).square().sum(dim=2)
        )

    combined_errors = {
        law: tile_errors["w1"][law]
        + tile_errors["w3"][law]
        + tile_errors["w2"][law].T
        for law in K2_MENU_LAWS
    }
    menu_specs: dict[str, tuple[tuple[str, ...], torch.Tensor, dict[str, object]]] = {}
    for law in K2_MENU_LAWS[1:]:
        ids, stats = _k2_menu_selector_stats(
            combined_errors, laws=("normal", law)
        )
        menu_specs[f"normal_{law}"] = (("normal", law), ids, stats)
    ids, stats = _k2_menu_selector_stats(combined_errors, laws=K2_MENU_LAWS)
    menu_specs["best_of_four"] = (K2_MENU_LAWS, ids, stats)

    normal_w2 = encoded["w2"]["normal"]["reconstruction"]
    if not isinstance(normal_w2, torch.Tensor):
        raise TypeError("normal K2 down reconstruction is missing")
    scores = {
        "normal": score_candidate((normal_w1, normal_w3, normal_w2))
    }
    h2_evidence: dict[str, object] = {"normal": normal_h2_evidence}
    for name, (laws, ids, _) in menu_specs.items():
        upstream_ids = tuple(int(value) for value in ids.flatten().cpu().tolist())
        down_ids = tuple(int(value) for value in ids.T.flatten().cpu().tolist())
        bank = {2: tuple(switched_luts[law][2] for law in laws)}
        mixed_upstream = []
        for matrix in ("w1", "w3"):
            candidate, _ = _quantize_maps(
                source[MATRICES.index(matrix)],
                h13,
                contexts,
                matrix=matrix,
                maps={name: uniform_maps[matrix]},
                layer=layer,
                expert=expert,
                device=device,
                quantizer_module=quantizer_module,
                ldlq_tf32=ldlq_tf32,
                tile_codebook_ids=upstream_ids,
                lut_bank_by_bits=bank,
                prepared=prepared[matrix],
            )
            reconstruction = candidate[name]["reconstruction"]
            if not isinstance(reconstruction, torch.Tensor):
                raise TypeError("mixed K2 upstream reconstruction is missing")
            mixed_upstream.append(reconstruction)
        mixed_middle = middle_decoder(
            fit_inputs,
            mixed_upstream[0],
            mixed_upstream[1],
        )
        _, mixed_h2, evidence = build_expert_hessians(
            fit_inputs,
            fit_gates,
            mixed_middle,
            global_h13=h13,
            global_h2=source_h2,
            device=device,
        )
        mixed_down, _ = _quantize_maps(
            source[2],
            mixed_h2,
            contexts,
            matrix="w2",
            maps={name: uniform_maps["w2"]},
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            tile_codebook_ids=down_ids,
            lut_bank_by_bits=bank,
            permutation_override=permutation_override,
        )
        mixed_w2 = mixed_down[name]["reconstruction"]
        if not isinstance(mixed_w2, torch.Tensor):
            raise TypeError("mixed K2 down reconstruction is missing")
        scores[name] = score_candidate(
            (mixed_upstream[0], mixed_upstream[1], mixed_w2)
        )
        h2_evidence[name] = evidence

    selected = min(scores, key=lambda name: float(scores[name]["fit"]["sse"]))
    baseline_confirmation = float(scores["normal"]["confirmation"]["sse"])
    baseline_fit = float(scores["normal"]["fit"]["sse"])
    eligible = [
        name
        for name in menu_specs
        if float(scores[name]["fit"]["sse"]) < baseline_fit
        and float(scores[name]["confirmation"]["sse"]) < baseline_confirmation
    ]
    confirmation_selected = (
        min(
            eligible,
            key=lambda name: float(scores[name]["confirmation"]["sse"]),
        )
        if eligible
        else "normal"
    )
    external_relative: dict[str, float] | None = None
    if "external" in scores["normal"]:
        baseline_external = float(scores["normal"]["external"]["sse"])
        external_relative = {
            name: _relative(float(score["external"]["sse"]), baseline_external)
            for name, score in scores.items()
        }
    return {
        "selection_granularity": (
            "one_shared_two_bit_codebook_id_per_corresponding_"
            "w1_w3_w2_16x16_tile_triplet"
        ),
        "proposal_scale": "independently_fitted_normal_k2_then_held_fixed",
        "exact_encode_scale": "independently_refit_per_matrix_and_menu",
        "w2_hessian": "expert_local_conditioned_on_decoded_upstream_candidate",
        "representation": (
            {"basis": "production"}
            if representation is None
            else dict(representation)
        ),
        "metadata": {
            "tile_triplets": 224 * 192,
            "bits_per_tile_triplet": 2,
            "bytes_per_expert": (224 * 192 * 2) // 8,
            "bits_per_weight": (224 * 192 * 2) / (3 * 3072 * 3584),
        },
        "uniform_normal_h2": normal_h2_evidence,
        "selector_stats": {
            name: stats for name, (_, _, stats) in menu_specs.items()
        },
        "conditional_h2": h2_evidence,
        "scores": scores,
        "selected_on_fit": selected,
        "selected_after_confirmation_gate": confirmation_selected,
        "confirmation_gate": {
            "rule": "strictly_lower_fit_and_confirmation_sse_than_normal",
            "eligible": eligible,
        },
        "confirmation_relative_to_normal": _relative(
            float(scores[selected]["confirmation"]["sse"]),
            baseline_confirmation,
        ),
        "external_relative_to_normal": external_relative,
        "confirmation_gated_external_relative_to_normal": (
            None
            if external_relative is None
            else external_relative[confirmation_selected]
        ),
        "best_of_four_confirmation_relative_to_normal": _relative(
            float(scores["best_of_four"]["confirmation"]["sse"]),
            baseline_confirmation,
        ),
    }


def _prepare_weighted_functional_target(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    inputs: torch.Tensor,
    gates: torch.Tensor,
    execute_triplet: TripletExecutor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if execute_triplet is None:
        reference = _execute_standard_triplet(inputs, source)
    else:
        reference = execute_triplet(inputs, source)
    return reference, gates.float().square().unsqueeze(1)


def _decode_standard_middle(
    inputs: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
) -> torch.Tensor:
    return situ(F.linear(inputs, w1), F.linear(inputs, w3))


def _execute_standard_triplet(
    inputs: torch.Tensor,
    reconstruction: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    middle = _decode_standard_middle(
        inputs, reconstruction[0], reconstruction[1]
    )
    return F.linear(middle, reconstruction[2])


def _two_bit_projection_search(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    h13: torch.Tensor,
    h2: torch.Tensor,
    contexts: torch.Tensor,
    layer: int,
    expert: int,
    device: torch.device,
    quantizer_module,
    ldlq_tf32: bool,
    inputs: torch.Tensor,
    gates: torch.Tensor,
    fit_mask: torch.Tensor,
    confirmation_mask: torch.Tensor,
    permutation_override: torch.Tensor,
    decode_middle: MiddleDecoder | None = None,
    execute_triplet: TripletExecutor | None = None,
    reference_output: torch.Tensor | None = None,
    representation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Compare exact-average-two-bit projection allocations.

    Gate and up are encoded once at each of K1, K2, and K3.  Every projection
    allocation then rebuilds the expert-local H2 from its decoded upstream
    activations and independently encodes the canonical W2 target at the
    requested rate.  No fitted replacement target enters the comparison.
    """

    middle_decoder = (
        _decode_standard_middle if decode_middle is None else decode_middle
    )
    triplet_executor = (
        _execute_standard_triplet if execute_triplet is None else execute_triplet
    )
    luts_by_bits = {
        bits: sqg_xor_cheb_t12_bytes(bits, device=device)
        for bits in (1, 2, 3)
    }
    fit_inputs = inputs[fit_mask]
    fit_gates = gates[fit_mask]
    fit_weights = fit_gates.float().square().unsqueeze(1)
    confirmation_weights = gates[confirmation_mask].float().square().unsqueeze(1)
    if reference_output is None:
        reference_output = triplet_executor(inputs, source)
    if reference_output.shape != (inputs.shape[0], source[2].shape[0]):
        raise ValueError("reference output does not align with the expert rows")
    fit_target = reference_output[fit_mask]
    confirmation_target = reference_output[confirmation_mask]
    fit_reference_energy = float(
        (fit_target.square() * fit_weights).sum(dtype=torch.float64)
    )
    confirmation_reference_energy = float(
        (confirmation_target.square() * confirmation_weights).sum(
            dtype=torch.float64
        )
    )

    prepared_upstream = {
        matrix: _prepare_quantize_maps(
            source[MATRICES.index(matrix)],
            h13,
            contexts,
            matrix=matrix,
            device=device,
            permutation_override=permutation_override,
        )
        for matrix in ("w1", "w3")
    }
    upstream: dict[str, dict[int, dict[str, object]]] = {
        "w1": {},
        "w3": {},
    }
    for matrix in ("w1", "w3"):
        for bits in (2, 1, 3):
            candidate, _ = _quantize_maps(
                source[MATRICES.index(matrix)],
                h13,
                contexts,
                matrix=matrix,
                maps={f"k{bits}": _uniform_tile_map((3584, 3072), bits)},
                layer=layer,
                expert=expert,
                device=device,
                quantizer_module=quantizer_module,
                ldlq_tf32=ldlq_tf32,
                luts_by_bits=luts_by_bits,
                prepared=prepared_upstream[matrix],
                shared_scale_scope="two-bit-projection-search",
                scale_search_bits=bits,
            )
            upstream[matrix][bits] = candidate[f"k{bits}"]

    scores: dict[str, dict[str, dict[str, float]]] = {}
    matrix_evidence: dict[str, dict[str, object]] = {}
    conditional_h2: dict[str, object] = {}
    modes = ((1, 1, 1), *TWO_BIT_PROJECTION_MODES, (3, 3, 3))
    for w1_bits, w3_bits, w2_bits in modes:
        name = f"{w1_bits}{w3_bits}{w2_bits}"
        w1 = upstream["w1"][w1_bits]["reconstruction"]
        w3 = upstream["w3"][w3_bits]["reconstruction"]
        if not isinstance(w1, torch.Tensor) or not isinstance(w3, torch.Tensor):
            raise TypeError("projection-rate upstream reconstruction is missing")
        decoded_middle = middle_decoder(fit_inputs, w1, w3)
        _, candidate_h2, h2_evidence = build_expert_hessians(
            fit_inputs,
            fit_gates,
            decoded_middle,
            global_h13=h13,
            global_h2=h2,
            device=device,
        )
        down, _ = _quantize_maps(
            source[2],
            candidate_h2,
            contexts,
            matrix="w2",
            maps={f"k{w2_bits}": _uniform_tile_map((3072, 3584), w2_bits)},
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            luts_by_bits=luts_by_bits,
            permutation_override=permutation_override,
            shared_scale_scope="two-bit-projection-search",
            scale_search_bits=w2_bits,
        )
        w2_candidate = down[f"k{w2_bits}"]
        w2 = w2_candidate["reconstruction"]
        if not isinstance(w2, torch.Tensor):
            raise TypeError("projection-rate down reconstruction is missing")
        fit_sse = _weighted_functional_sse(
            (w1, w3, w2),
            inputs=fit_inputs,
            reference=fit_target,
            route_weights=fit_weights,
            execute_triplet=triplet_executor,
        )
        confirmation_sse = _weighted_functional_sse(
            (w1, w3, w2),
            inputs=inputs[confirmation_mask],
            reference=confirmation_target,
            route_weights=confirmation_weights,
            execute_triplet=triplet_executor,
        )
        scores[name] = {
            "fit": {
                "sse": fit_sse,
                "reference_energy": fit_reference_energy,
                "nmse": fit_sse / fit_reference_energy,
            },
            "confirmation": {
                "sse": confirmation_sse,
                "reference_energy": confirmation_reference_energy,
                "nmse": confirmation_sse / confirmation_reference_energy,
            },
        }
        matrix_evidence[name] = {
            "w1_g_scale": float(upstream["w1"][w1_bits]["g_scale"]),
            "w3_g_scale": float(upstream["w3"][w3_bits]["g_scale"]),
            "w2_g_scale": float(w2_candidate["g_scale"]),
            "w1_proxy": float(upstream["w1"][w1_bits]["proxy"]),
            "w3_proxy": float(upstream["w3"][w3_bits]["proxy"]),
            "w2_proxy": float(w2_candidate["proxy"]),
        }
        conditional_h2[name] = h2_evidence

    equal_rate_names = tuple("".join(str(value) for value in mode) for mode in TWO_BIT_PROJECTION_MODES)
    selected = min(
        equal_rate_names,
        key=lambda name: float(scores[name]["fit"]["sse"]),
    )
    confirmation_oracle = min(
        equal_rate_names,
        key=lambda name: float(scores[name]["confirmation"]["sse"]),
    )
    baseline_confirmation = float(scores["222"]["confirmation"]["sse"])
    return {
        "rates": {
            "controls": ["111", "222", "333"],
            "exact_average_two_bit_modes": list(equal_rate_names),
            "trellis_bits_per_weight": 2.0,
            "expert_static_mode_bits": 3,
            "mode_metadata_bits_per_weight": 3 / (3 * 3072 * 3584),
        },
        "targets": "canonical_w1_w3_w2",
        "scale_selection": "independent_per_matrix_and_rate",
        "w2_hessian": "expert_local_conditioned_on_decoded_upstream_candidate",
        "representation": (
            {"basis": "production"}
            if representation is None
            else dict(representation)
        ),
        "scores": scores,
        "matrix_evidence": matrix_evidence,
        "conditional_h2": conditional_h2,
        "selected_on_fit": selected,
        "confirmation_oracle": confirmation_oracle,
        "confirmation_relative_to_222": _relative(
            float(scores[selected]["confirmation"]["sse"]),
            baseline_confirmation,
        ),
        "confirmation_oracle_relative_to_222": _relative(
            float(scores[confirmation_oracle]["confirmation"]["sse"]),
            baseline_confirmation,
        ),
    }


def _p13_functional_record_deltas(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    uniform: Mapping[str, Mapping[int, Mapping[str, object]]],
    *,
    inputs: torch.Tensor,
    reference: torch.Tensor,
    route_weights: torch.Tensor,
    permutation: torch.Tensor,
    execute_triplet: TripletExecutor,
) -> dict[str, object]:
    """Measure single-record K1 donor costs and K3 recipient gains.

    Each hybrid changes one physical 128-channel encoder record from the
    uniform-K2 reconstruction.  Gate and up change together; down changes
    independently.  The complete decoded expert function supplies the fit
    loss, including the coupled activation-boundary transform when active.
    """

    reconstructions: dict[str, dict[int, torch.Tensor]] = {}
    for matrix in MATRICES:
        by_rate: dict[int, torch.Tensor] = {}
        for bits in (1, 2, 3):
            value = uniform[matrix][bits].get("reconstruction")
            if not isinstance(value, torch.Tensor):
                raise TypeError("P13 uniform candidate lacks a reconstruction")
            by_rate[bits] = value
        reconstructions[matrix] = by_rate
    permutation = permutation.to(device=inputs.device, dtype=torch.long)
    if permutation.shape != (3072,) or not torch.equal(
        torch.sort(permutation).values,
        torch.arange(3072, device=permutation.device),
    ):
        raise ValueError("P13 record attribution requires a 3072-channel bijection")

    baseline = tuple(reconstructions[matrix][2] for matrix in MATRICES)
    baseline_sse = _weighted_functional_sse(
        baseline,
        inputs=inputs,
        reference=reference,
        route_weights=route_weights,
        execute_triplet=execute_triplet,
    )
    hybrid_sse = {
        "w13": {1: [], 3: []},
        "w2": {1: [], 3: []},
    }
    for record in range(24):
        indices = permutation[record * 128 : (record + 1) * 128]
        for bits in (1, 3):
            w1 = baseline[0].clone()
            w3 = baseline[1].clone()
            w1.index_copy_(
                0, indices, reconstructions["w1"][bits].index_select(0, indices)
            )
            w3.index_copy_(
                0, indices, reconstructions["w3"][bits].index_select(0, indices)
            )
            hybrid_sse["w13"][bits].append(
                _weighted_functional_sse(
                    (w1, w3, baseline[2]),
                    inputs=inputs,
                    reference=reference,
                    route_weights=route_weights,
                    execute_triplet=execute_triplet,
                )
            )

            w2 = baseline[2].clone()
            w2.index_copy_(
                1, indices, reconstructions["w2"][bits].index_select(1, indices)
            )
            hybrid_sse["w2"][bits].append(
                _weighted_functional_sse(
                    (baseline[0], baseline[1], w2),
                    inputs=inputs,
                    reference=reference,
                    route_weights=route_weights,
                    execute_triplet=execute_triplet,
                )
            )

    result: dict[str, object] = {"baseline_sse": baseline_sse}
    for family in ("w13", "w2"):
        donor = torch.tensor(hybrid_sse[family][1], dtype=torch.float64) - baseline_sse
        recipient = (
            torch.tensor(hybrid_sse[family][3], dtype=torch.float64) - baseline_sse
        )
        result[family] = {
            "k1_hybrid_sse": hybrid_sse[family][1],
            "k3_hybrid_sse": hybrid_sse[family][3],
            "donor_delta": donor,
            "recipient_delta": recipient,
        }
    return result


def _p13_record_search(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    h13: torch.Tensor,
    h2: torch.Tensor,
    contexts: torch.Tensor,
    layer: int,
    expert: int,
    device: torch.device,
    quantizer_module,
    ldlq_tf32: bool,
    inputs: torch.Tensor,
    gates: torch.Tensor,
    fit_mask: torch.Tensor,
    confirmation_mask: torch.Tensor,
    permutation_override: torch.Tensor,
    decode_middle: MiddleDecoder | None = None,
    execute_triplet: TripletExecutor | None = None,
    reference_output: torch.Tensor | None = None,
    representation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Search channel-record P13 schedules with absolute dense-H loss."""

    middle_decoder = (
        _decode_standard_middle if decode_middle is None else decode_middle
    )
    triplet_executor = (
        _execute_standard_triplet if execute_triplet is None else execute_triplet
    )
    luts_by_bits = {
        bits: sqg_xor_cheb_t12_bytes(bits, device=device)
        for bits in (1, 2, 3)
    }
    if reference_output is None:
        reference_output = triplet_executor(inputs, source)
    fit_inputs = inputs[fit_mask]
    fit_weights = gates[fit_mask].float().square().unsqueeze(1)
    confirmation_inputs = inputs[confirmation_mask]
    confirmation_weights = (
        gates[confirmation_mask].float().square().unsqueeze(1)
    )
    fit_target = reference_output[fit_mask]
    confirmation_target = reference_output[confirmation_mask]

    def score(
        reconstruction: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for name, local_inputs, local_target, weights in (
            ("fit", fit_inputs, fit_target, fit_weights),
            (
                "confirmation",
                confirmation_inputs,
                confirmation_target,
                confirmation_weights,
            ),
        ):
            reference_energy = float(
                (local_target.square() * weights).sum(dtype=torch.float64)
            )
            sse = _weighted_functional_sse(
                reconstruction,
                inputs=local_inputs,
                reference=local_target,
                route_weights=weights,
                execute_triplet=triplet_executor,
            )
            result[name] = {
                "sse": sse,
                "reference_energy": reference_energy,
                "nmse": sse / reference_energy,
            }
        return result

    def encode_rate_curves(
        matrix: str,
        hessian: torch.Tensor,
        prepared: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        scope: str,
    ) -> tuple[
        dict[int, dict[str, object]],
        dict[int, torch.Tensor],
        dict[int, dict[str, object]],
        dict[int, torch.Tensor],
        dict[str, object],
    ]:
        shape = (3584, 3072) if matrix in ("w1", "w3") else (3072, 3584)
        source_matrix = source[MATRICES.index(matrix)]
        native: dict[int, dict[str, object]] = {}
        for bits in (1, 2, 3):
            candidates, _ = _quantize_maps(
                source_matrix,
                hessian,
                contexts,
                matrix=matrix,
                maps={f"k{bits}": _uniform_tile_map(shape, bits)},
                layer=layer,
                expert=expert,
                device=device,
                quantizer_module=quantizer_module,
                ldlq_tf32=ldlq_tf32,
                luts_by_bits=luts_by_bits,
                prepared=prepared,
                shared_scale_scope=f"{scope}-independent",
                scale_search_bits=bits,
            )
            native[bits] = candidates[f"k{bits}"]
        shared_group, _ = _quantize_maps(
            source_matrix,
            hessian,
            contexts,
            matrix=matrix,
            maps={
                f"k{bits}": _uniform_tile_map(shape, bits)
                for bits in (1, 2, 3)
            },
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            luts_by_bits=luts_by_bits,
            prepared=prepared,
            shared_scale_scope=f"{scope}-shared",
            scale_search_bits=2,
        )
        shared = {bits: shared_group[f"k{bits}"] for bits in (1, 2, 3)}
        encoder_weight, encoder_hessian, _ = prepared
        native_errors = _dense_h_tile_errors(
            encoder_weight, encoder_hessian, native
        )
        shared_errors = _dense_h_tile_errors(
            encoder_weight, encoder_hessian, shared
        )
        optimized: dict[int, dict[str, object]] = {}
        optimized_errors: dict[int, torch.Tensor] = {}
        scale_search: dict[str, object] = {}
        grid_specs = {
            1: (33, 0.75, float(shared[2]["g_scale"])),
            2: (9, 0.25, float(native[2]["g_scale"])),
            3: (9, 0.25, float(native[3]["g_scale"])),
        }
        for bits in (1, 2, 3):
            count, half_width_octaves, center = grid_specs[bits]
            scales = [
                center
                * 2.0
                ** (
                    -half_width_octaves
                    + 2.0 * half_width_octaves * index / (count - 1)
                )
                for index in range(count)
            ]
            scales.extend(
                (
                    float(native[bits]["g_scale"]),
                    float(shared[bits]["g_scale"]),
                )
            )
            unique_scales = sorted({round(value, 12) for value in scales})
            best_candidate = native[bits]
            best_error = native_errors[bits]
            best_total = float(best_error.sum())
            best_scale = float(best_candidate["g_scale"])
            path_hashes: set[str] = set()
            evaluated = 0
            for scale_value in unique_scales:
                if abs(scale_value - float(native[bits]["g_scale"])) <= 1e-11:
                    candidate = native[bits]
                    candidate_error = native_errors[bits]
                elif abs(scale_value - float(shared[bits]["g_scale"])) <= 1e-11:
                    candidate = shared[bits]
                    candidate_error = shared_errors[bits]
                else:
                    group, _ = _quantize_maps(
                        source_matrix,
                        hessian,
                        contexts,
                        matrix=matrix,
                        maps={f"k{bits}": _uniform_tile_map(shape, bits)},
                        layer=layer,
                        expert=expert,
                        device=device,
                        quantizer_module=quantizer_module,
                        ldlq_tf32=ldlq_tf32,
                        g_scale_override=scale_value,
                        luts_by_bits=luts_by_bits,
                        prepared=prepared,
                        shared_scale_scope=f"{scope}-dense-h-grid-k{bits}",
                        scale_search_bits=bits,
                    )
                    candidate = group[f"k{bits}"]
                    candidate_error = _dense_h_tile_errors(
                        encoder_weight,
                        encoder_hessian,
                        {bits: candidate},
                    )[bits]
                states = candidate.get("states")
                if not isinstance(states, torch.Tensor):
                    raise TypeError("scale-search candidate lacks trellis states")
                path_hashes.add(
                    hashlib.sha256(
                        states.detach().cpu().contiguous().numpy().tobytes()
                    ).hexdigest()
                )
                evaluated += 1
                total = float(candidate_error.sum())
                candidate_scale = float(candidate["g_scale"])
                if (total, candidate_scale) < (best_total, best_scale):
                    best_candidate = candidate
                    best_error = candidate_error
                    best_total = total
                    best_scale = candidate_scale
            optimized[bits] = best_candidate
            optimized_errors[bits] = best_error
            scale_search[str(bits)] = {
                "selection_metric": "absolute_dense_h_blockldlq_distortion",
                "evaluated_scales": evaluated,
                "distinct_trellis_paths": len(path_hashes),
                "minimum_scale": min(unique_scales),
                "maximum_scale": max(unique_scales),
                "native_scale": float(native[bits]["g_scale"]),
                "native_dense_h_loss": float(native_errors[bits].sum()),
                "shared_k2_oriented_scale": float(shared[bits]["g_scale"]),
                "shared_k2_oriented_dense_h_loss": float(
                    shared_errors[bits].sum()
                ),
                "selected_scale": best_scale,
                "selected_dense_h_loss": best_total,
            }
        return (
            optimized,
            optimized_errors,
            shared,
            shared_errors,
            scale_search,
        )

    def encode_zero_feedback_rate_curves(
        matrix: str,
        hessian: torch.Tensor,
        prepared: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        scope: str,
    ) -> tuple[
        dict[int, dict[str, object]],
        dict[int, torch.Tensor],
        dict[str, object],
    ]:
        """Construct an order-neutral rate surface for schedule proposals.

        An identity encoder Hessian makes the off-diagonal BlockLDLQ feedback
        exactly zero while retaining the production transforms, scalar law,
        common K2-oriented scale search, tail-biting Viterbi, and decode path.
        The resulting decoded candidates are scored with the actual dense
        Hessian.  They are never accepted as final payloads.
        """

        encoder_weight, encoder_hessian, permutation = prepared
        identity_hessian = torch.eye(
            encoder_weight.shape[0],
            dtype=encoder_hessian.dtype,
            device=encoder_hessian.device,
        )
        group, _ = _quantize_maps(
            source[MATRICES.index(matrix)],
            hessian,
            contexts,
            matrix=matrix,
            maps={
                f"k{bits}": _uniform_tile_map(
                    (3584, 3072) if matrix in ("w1", "w3") else (3072, 3584),
                    bits,
                )
                for bits in (1, 2, 3)
            },
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            luts_by_bits=luts_by_bits,
            prepared=(encoder_weight, identity_hessian, permutation),
            shared_scale_scope=f"{scope}-zero-feedback",
            scale_search_bits=2,
        )
        candidates = {bits: group[f"k{bits}"] for bits in (1, 2, 3)}
        errors = _dense_h_tile_errors(
            encoder_weight,
            encoder_hessian,
            candidates,
        )
        return candidates, errors, {
            "candidate_construction": (
                "identity_hessian_with_zero_cross_tile_ldlq_feedback"
            ),
            "candidate_scoring": "decoded_absolute_dense_h_distortion",
            "scale_policy": "one_zero_feedback_k2_oriented_scale_per_matrix",
            "g_scale": float(candidates[2]["g_scale"]),
            "rates": curve_evidence(candidates, errors),
        }

    def curve_evidence(
        candidates: Mapping[int, Mapping[str, object]],
        errors: Mapping[int, torch.Tensor],
    ) -> dict[str, object]:
        return {
            str(bits): {
                "g_scale": float(candidates[bits]["g_scale"]),
                "absolute_dense_h_loss": float(errors[bits].sum()),
                "normalized_encoder_proxy": float(candidates[bits]["proxy"]),
            }
            for bits in (1, 2, 3)
        }

    def family_allocation(
        errors: Mapping[int, torch.Tensor],
        *,
        rate_axis: str,
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, object]]:
        donor, recipient = _p13_record_deltas_from_tile_errors(
            errors, rate_axis=rate_axis
        )
        maps, allocation = _balanced_p13_record_rates(donor, recipient)
        return maps, {
            "marginals": _p13_marginal_diagnostics(donor, recipient),
            **allocation,
        }

    def granularity_oracles(
        errors: Mapping[int, torch.Tensor],
        *,
        rate_axis: str,
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for channels in (128, 64, 16):
            donor, recipient = _p13_channel_group_deltas_from_tile_errors(
                errors,
                rate_axis=rate_axis,
                channels_per_group=channels,
            )
            result[f"channel_group_{channels}"] = _p13_marginal_diagnostics(
                donor,
                recipient,
            )
        result["tile_16x16"] = _p13_tile_pair_diagnostics(errors)
        return result

    prepared_upstream = {
        matrix: _prepare_quantize_maps(
            source[MATRICES.index(matrix)],
            h13,
            contexts,
            matrix=matrix,
            device=device,
            permutation_override=permutation_override,
        )
        for matrix in ("w1", "w3")
    }
    independent: dict[str, dict[int, dict[str, object]]] = {}
    independent_errors: dict[str, dict[int, torch.Tensor]] = {}
    shared: dict[str, dict[int, dict[str, object]]] = {}
    shared_errors: dict[str, dict[int, torch.Tensor]] = {}
    zero_feedback: dict[str, dict[int, dict[str, object]]] = {}
    zero_feedback_errors: dict[str, dict[int, torch.Tensor]] = {}
    zero_feedback_evidence: dict[str, object] = {}
    dense_h_scale_search: dict[str, object] = {}
    for matrix in ("w1", "w3"):
        (
            independent[matrix],
            independent_errors[matrix],
            shared[matrix],
            shared_errors[matrix],
            dense_h_scale_search[matrix],
        ) = encode_rate_curves(
            matrix,
            h13,
            prepared_upstream[matrix],
            scope="p13-upstream-rate-curves",
        )
        (
            zero_feedback[matrix],
            zero_feedback_errors[matrix],
            zero_feedback_evidence[matrix],
        ) = encode_zero_feedback_rate_curves(
            matrix,
            h13,
            prepared_upstream[matrix],
            scope="p13-upstream-proposal-surface",
        )

    baseline_w1 = shared["w1"][2]["reconstruction"]
    baseline_w3 = shared["w3"][2]["reconstruction"]
    if not isinstance(baseline_w1, torch.Tensor) or not isinstance(
        baseline_w3, torch.Tensor
    ):
        raise TypeError("P13 uniform K2 upstream reconstruction is missing")
    baseline_middle = middle_decoder(inputs, baseline_w1, baseline_w3)
    _, baseline_h2, baseline_h2_evidence = build_expert_hessians(
        inputs,
        gates,
        baseline_middle,
        global_h13=h13,
        global_h2=h2,
        device=device,
    )
    prepared_down = _prepare_quantize_maps(
        source[2],
        baseline_h2,
        contexts,
        matrix="w2",
        device=device,
        permutation_override=permutation_override,
    )
    (
        independent["w2"],
        independent_errors["w2"],
        shared["w2"],
        shared_errors["w2"],
        dense_h_scale_search["w2"],
    ) = encode_rate_curves(
        "w2",
        baseline_h2,
        prepared_down,
        scope="p13-down-rate-curves",
    )
    (
        zero_feedback["w2"],
        zero_feedback_errors["w2"],
        zero_feedback_evidence["w2"],
    ) = encode_zero_feedback_rate_curves(
        "w2",
        baseline_h2,
        prepared_down,
        scope="p13-down-proposal-surface-uniform-upstream",
    )

    def allocation_oracles(
        errors: Mapping[str, Mapping[int, torch.Tensor]],
    ) -> dict[str, object]:
        matrix_marginals: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for matrix, rate_axis in (("w1", "n"), ("w3", "n"), ("w2", "k")):
            matrix_marginals[matrix] = _p13_record_deltas_from_tile_errors(
                errors[matrix], rate_axis=rate_axis
            )
        return {
            "cross_projection_oracle": {
                "status": "not_computed",
                "required_metric": (
                    "one additive mapped-expert-output objective shared by "
                    "W1, W3, and W2"
                ),
                "reason": (
                    "matrix-native dense-H losses are additive within one "
                    "projection but are not cross-projection functional units"
                ),
            },
            "per_matrix": {
                matrix: {
                    "marginals": _p13_marginal_diagnostics(*matrix_marginals[matrix]),
                    "allocation": _best_balanced_p13_units(
                        *matrix_marginals[matrix]
                    ),
                }
                for matrix in MATRICES
            },
        }

    independent_oracles = allocation_oracles(independent_errors)
    shared_oracles = allocation_oracles(shared_errors)
    independent_w13 = {
        bits: independent_errors["w1"][bits]
        + independent_errors["w3"][bits]
        for bits in (1, 2, 3)
    }
    shared_w13 = {
        bits: shared_errors["w1"][bits] + shared_errors["w3"][bits]
        for bits in (1, 2, 3)
    }
    zero_feedback_w13 = {
        bits: (
            zero_feedback_errors["w1"][bits]
            + zero_feedback_errors["w3"][bits]
        )
        for bits in (1, 2, 3)
    }
    _, independent_w13_allocation = family_allocation(
        independent_w13, rate_axis="n"
    )
    _, independent_w2_allocation = family_allocation(
        independent_errors["w2"], rate_axis="k"
    )
    _, shared_w13_allocation = family_allocation(shared_w13, rate_axis="n")
    upstream_record_rates, zero_feedback_w13_allocation = family_allocation(
        zero_feedback_w13, rate_axis="n"
    )
    _, shared_w2_baseline_allocation = family_allocation(
        shared_errors["w2"], rate_axis="k"
    )

    upstream_maps = {
        name: _p13_record_map_from_rates(
            (3584, 3072), rates, rate_axis="n"
        )
        for name, rates in upstream_record_rates.items()
    }
    encoded_upstream: dict[str, dict[str, dict[str, object]]] = {}
    encoded_upstream_errors: dict[str, dict[str, torch.Tensor]] = {}
    for matrix in ("w1", "w3"):
        encoded_upstream[matrix], _ = _quantize_maps(
            source[MATRICES.index(matrix)],
            h13,
            contexts,
            matrix=matrix,
            maps=upstream_maps,
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            g_scale_override=float(shared[matrix][2]["g_scale"]),
            luts_by_bits=luts_by_bits,
            prepared=prepared_upstream[matrix],
            shared_scale_scope="p13-upstream-mixed-schedules",
            scale_search_bits=2,
        )
        encoded_upstream_errors[matrix] = _dense_h_tile_errors(
            prepared_upstream[matrix][0],
            prepared_upstream[matrix][1],
            encoded_upstream[matrix],
        )
    upstream_dense_h = {
        name: float(
            encoded_upstream_errors["w1"][name].sum()
            + encoded_upstream_errors["w3"][name].sum()
        )
        for name in upstream_maps
    }
    selected_upstream = min(
        upstream_dense_h,
        key=lambda name: (upstream_dense_h[name], name),
    )
    selected_w1 = encoded_upstream["w1"][selected_upstream]["reconstruction"]
    selected_w3 = encoded_upstream["w3"][selected_upstream]["reconstruction"]
    if not isinstance(selected_w1, torch.Tensor) or not isinstance(
        selected_w3, torch.Tensor
    ):
        raise TypeError("selected P13 upstream reconstruction is missing")
    selected_middle = middle_decoder(inputs, selected_w1, selected_w3)
    _, selected_h2, selected_h2_evidence = build_expert_hessians(
        inputs,
        gates,
        selected_middle,
        global_h13=h13,
        global_h2=h2,
        device=device,
    )

    prepared_selected_down = _prepare_quantize_maps(
        source[2],
        selected_h2,
        contexts,
        matrix="w2",
        device=device,
        permutation_override=permutation_override,
    )
    selected_down_group, _ = _quantize_maps(
        source[2],
        selected_h2,
        contexts,
        matrix="w2",
        maps={
            f"k{bits}": _uniform_tile_map((3072, 3584), bits)
            for bits in (1, 2, 3)
        },
        layer=layer,
        expert=expert,
        device=device,
        quantizer_module=quantizer_module,
        ldlq_tf32=ldlq_tf32,
        luts_by_bits=luts_by_bits,
        prepared=prepared_selected_down,
        shared_scale_scope="p13-selected-down-rate-curves",
        scale_search_bits=2,
    )
    selected_down_uniform = {
        bits: selected_down_group[f"k{bits}"] for bits in (1, 2, 3)
    }
    selected_down_errors = _dense_h_tile_errors(
        prepared_selected_down[0],
        prepared_selected_down[1],
        selected_down_uniform,
    )
    (
        selected_down_zero_feedback,
        selected_down_zero_feedback_errors,
        selected_down_zero_feedback_evidence,
    ) = encode_zero_feedback_rate_curves(
        "w2",
        selected_h2,
        prepared_selected_down,
        scope="p13-down-proposal-surface-selected-upstream",
    )
    fine_down_plans: dict[str, dict[str, object]] = {}
    fine_down_maps: dict[str, tuple[int, ...]] = {}
    for channels in (64, 16):
        donor, recipient = _p13_channel_group_deltas_from_tile_errors(
            selected_down_zero_feedback_errors,
            rate_axis="k",
            channels_per_group=channels,
        )
        maximum = 4 if channels == 64 else 8
        plans_by_count = _balanced_p13_unit_rates(
            donor,
            recipient,
            max_count=maximum,
        )
        best_count = min(
            plans_by_count,
            key=lambda count: (
                float(plans_by_count[count]["delta"]),
                count,
            ),
        )
        counts = sorted({1, 2, 4, maximum, best_count})
        for count in counts:
            plan = plans_by_count[count]
            name = f"channel{channels}_n{count}"
            rates = plan["rates"]
            if not isinstance(rates, list):
                raise TypeError("P13 channel-group allocation is missing rates")
            fine_down_plans[name] = plan
            fine_down_maps[name] = _p13_channel_group_map_from_rates(
                (3072, 3584),
                rates,
                rate_axis="k",
                channels_per_group=channels,
            )

    tile_down_plans: dict[str, dict[str, object]] = {}
    tile_down_maps: dict[str, tuple[int, ...]] = {}
    tile_record_rates = (*((1,) * 12), *((3,) * 12))
    for pairing_policy in ("mirror", "maximum_weight"):
        for numerator, denominator in ((1, 16), (1, 8), (1, 4), (1, 2), (1, 1)):
            fraction = numerator / denominator
            tile_map, plan = _p13_fractional_tile_map(
                selected_down_zero_feedback_errors,
                tile_record_rates,
                rate_axis="k",
                pairing_policy=pairing_policy,
                positive_fraction=fraction,
            )
            if int(plan["selected_pair_tiles"]) == 0:
                continue
            name = f"tile_{pairing_policy}_p{numerator}of{denominator}"
            tile_down_plans[name] = plan
            tile_down_maps[name] = tile_map

    fine_encoded_down: dict[str, dict[str, object]] = {}
    fine_down_dense_h: dict[str, float] = {}
    fine_down_scores: dict[str, dict[str, object]] = {}
    all_fine_down_maps = {**fine_down_maps, **tile_down_maps}
    if all_fine_down_maps:
        fine_encoded_down, _ = _quantize_maps(
            source[2],
            selected_h2,
            contexts,
            matrix="w2",
            maps=all_fine_down_maps,
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            g_scale_override=float(selected_down_uniform[2]["g_scale"]),
            luts_by_bits=luts_by_bits,
            prepared=prepared_selected_down,
            shared_scale_scope="p13-down-channel-group-schedules",
            scale_search_bits=2,
        )
        fine_down_errors = _dense_h_tile_errors(
            prepared_selected_down[0],
            prepared_selected_down[1],
            fine_encoded_down,
        )
        fine_down_dense_h = {
            name: float(value.sum()) for name, value in fine_down_errors.items()
        }
    down_record_rates, selected_down_allocation = family_allocation(
        selected_down_zero_feedback_errors, rate_axis="k"
    )
    down_maps = {
        name: _p13_record_map_from_rates(
            (3072, 3584), rates, rate_axis="k"
        )
        for name, rates in down_record_rates.items()
    }
    encoded_down, _ = _quantize_maps(
        source[2],
        selected_h2,
        contexts,
        matrix="w2",
        maps=down_maps,
        layer=layer,
        expert=expert,
        device=device,
        quantizer_module=quantizer_module,
        ldlq_tf32=ldlq_tf32,
        g_scale_override=float(selected_down_uniform[2]["g_scale"]),
        luts_by_bits=luts_by_bits,
        prepared=prepared_selected_down,
        shared_scale_scope="p13-down-mixed-schedules",
        scale_search_bits=2,
    )
    encoded_down_errors = _dense_h_tile_errors(
        prepared_selected_down[0],
        prepared_selected_down[1],
        encoded_down,
    )
    down_dense_h = {
        name: float(value.sum()) for name, value in encoded_down_errors.items()
    }

    def encode_w2_traversal_variant(
        rate_map: tuple[int, ...],
        *,
        policy: str,
        scope: str,
    ) -> tuple[dict[str, object], float, dict[str, object]]:
        ordered_prepared, ordered_map, evidence = _w2_record_traversal_variant(
            prepared_selected_down,
            rate_map,
            policy=policy,
        )
        group, _ = _quantize_maps(
            source[2],
            selected_h2,
            contexts,
            matrix="w2",
            maps={"candidate": ordered_map},
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            g_scale_override=float(selected_down_uniform[2]["g_scale"]),
            luts_by_bits=luts_by_bits,
            prepared=ordered_prepared,
            shared_scale_scope=scope,
            scale_search_bits=2,
        )
        candidate = group["candidate"]
        dense_h = float(
            _dense_h_tile_errors(
                ordered_prepared[0],
                ordered_prepared[1],
                {"candidate": candidate},
            )["candidate"].sum()
        )
        reconstruction = candidate.get("reconstruction")
        if not isinstance(reconstruction, torch.Tensor):
            raise TypeError("W2 traversal candidate reconstruction is missing")
        evidence["dense_h"] = dense_h
        evidence["routed_expert_output_diagnostic"] = score(
            (selected_w1, selected_w3, reconstruction)
        )
        return candidate, dense_h, evidence

    uniform_reverse_candidate, uniform_reverse_dense_h, uniform_reverse_evidence = (
        encode_w2_traversal_variant(
            down_maps["n0"],
            policy="reverse",
            scope="p13-w2-uniform-k2-reverse-record-traversal",
        )
    )
    del uniform_reverse_candidate
    mixed_order_sources = {
        **{
            f"record_{name}": (rate_map, down_dense_h[name])
            for name, rate_map in down_maps.items()
            if name != "n0"
        },
        **{
            f"fine_{name}": (rate_map, fine_down_dense_h[name])
            for name, rate_map in all_fine_down_maps.items()
        },
    }
    mixed_order_study: dict[str, object]
    if mixed_order_sources:
        mixed_source = min(
            mixed_order_sources,
            key=lambda name: (mixed_order_sources[name][1], name),
        )
        mixed_rate_map, mixed_base_dense_h = mixed_order_sources[mixed_source]
        mixed_variants: dict[str, object] = {
            "baseline": {
                "dense_h": mixed_base_dense_h,
                "relative_to_uniform_k2": _relative(
                    mixed_base_dense_h,
                    down_dense_h["n0"],
                ),
            }
        }
        for policy in ("donor_first", "recipient_first"):
            _, dense_h, evidence = encode_w2_traversal_variant(
                mixed_rate_map,
                policy=policy,
                scope=f"p13-w2-{mixed_source}-{policy}-record-traversal",
            )
            mixed_variants[policy] = {
                **evidence,
                "relative_to_uniform_k2": _relative(
                    dense_h,
                    down_dense_h["n0"],
                ),
                "relative_to_baseline_order": _relative(
                    dense_h,
                    mixed_base_dense_h,
                ),
            }
        mixed_order_study = {
            "status": "evaluated",
            "schedule": mixed_source,
            "variants": mixed_variants,
        }
    else:
        mixed_order_study = {"status": "no_nonuniform_schedule_proposed"}

    selected_down = min(
        down_dense_h,
        key=lambda name: (down_dense_h[name], name),
    )
    selected_w2 = encoded_down[selected_down]["reconstruction"]
    baseline_w2 = shared["w2"][2]["reconstruction"]
    if not isinstance(selected_w2, torch.Tensor) or not isinstance(
        baseline_w2, torch.Tensor
    ):
        raise TypeError("P13 down reconstruction is missing")
    baseline = score((baseline_w1, baseline_w3, baseline_w2))
    selected_score = score((selected_w1, selected_w3, selected_w2))
    for name, candidate in fine_encoded_down.items():
        reconstruction = candidate.get("reconstruction")
        if not isinstance(reconstruction, torch.Tensor):
            raise TypeError("P13 channel-group W2 reconstruction is missing")
        fine_down_scores[name] = score(
            (selected_w1, selected_w3, reconstruction)
        )
    selected_name = f"u{selected_upstream}/d{selected_down}"
    baseline_dense_h = (
        upstream_dense_h["n0"] + float(shared_errors["w2"][2].sum())
    )
    selected_dense_h = (
        upstream_dense_h[selected_upstream] + down_dense_h[selected_down]
    )

    return {
        "allocation_unit": "128_channel_record",
        "schedule": (
            "K1 donors and K3 recipients are proposed from decoded "
            "zero-feedback rate surfaces, then every proposed mixed schedule "
            "is encoded from the canonical source with complete BlockLDLQ."
        ),
        "trellis_bits_per_weight": 2.0,
        "w13_rate_map": "shared",
        "w2_rate_map": "independent",
        "selector": {
            "grammar": "disjoint K1 and K3 record masks for W13 and W2",
            "bits_per_expert": 96,
            "bytes_per_expert": 12,
            "bits_per_weight": 96 / (3 * 3072 * 3584),
        },
        "targets": "canonical_w1_w3_w2",
        "selection_metric": (
            "complete_mixed_rate_blockldlq_absolute_dense_h_distortion"
        ),
        "proposal_metric": (
            "zero_feedback_decoded_candidates_scored_by_absolute_dense_h"
        ),
        "document_partition_used_for_selection": False,
        "zero_feedback_schedule_screen": {
            "role": "proposal_only",
            "cross_tile_feedback": False,
            "matrices_at_uniform_k2_upstream": zero_feedback_evidence,
            "w2_at_selected_upstream": selected_down_zero_feedback_evidence,
            "deployed_family_constraints": {
                "w13": zero_feedback_w13_allocation,
                "w2_at_selected_upstream": selected_down_allocation,
            },
            "granularity_oracles": {
                "w13": granularity_oracles(
                    zero_feedback_w13,
                    rate_axis="n",
                ),
                "w2_at_uniform_k2_upstream": granularity_oracles(
                    zero_feedback_errors["w2"],
                    rate_axis="k",
                ),
                "w2_at_selected_upstream": granularity_oracles(
                    selected_down_zero_feedback_errors,
                    rate_axis="k",
                ),
            },
        },
        "scale_oracles": {
            "independent_rate_scales": {
                "dense_h_scale_search": dense_h_scale_search,
                "matrices": {
                    matrix: curve_evidence(
                        independent[matrix], independent_errors[matrix]
                    )
                    for matrix in MATRICES
                },
                "allocation_oracles": independent_oracles,
                "deployed_family_constraints": {
                    "w13": independent_w13_allocation,
                    "w2": independent_w2_allocation,
                },
                "granularity_oracles": {
                    "w13": granularity_oracles(
                        independent_w13,
                        rate_axis="n",
                    ),
                    "w2": granularity_oracles(
                        independent_errors["w2"],
                        rate_axis="k",
                    ),
                },
            },
            "one_shared_deployable_scale_per_matrix": {
                "matrices": {
                    matrix: curve_evidence(shared[matrix], shared_errors[matrix])
                    for matrix in MATRICES
                },
                "allocation_oracles": shared_oracles,
                "deployed_family_constraints": {
                    "w13": shared_w13_allocation,
                    "w2_at_uniform_k2_upstream": shared_w2_baseline_allocation,
                },
                "granularity_oracles": {
                    "w13": granularity_oracles(
                        shared_w13,
                        rate_axis="n",
                    ),
                    "w2_at_uniform_k2_upstream": granularity_oracles(
                        shared_errors["w2"],
                        rate_axis="k",
                    ),
                    "w2_at_selected_upstream": granularity_oracles(
                        selected_down_errors,
                        rate_axis="k",
                    ),
                },
            },
        },
        "selected_on_dense_h": selected_name,
        "selected_rate_records": {
            "w13": zero_feedback_w13_allocation["allocations"][selected_upstream],
            "w2": selected_down_allocation["allocations"][selected_down],
        },
        "mixed_schedule_dense_h": {
            "baseline_p22": baseline_dense_h,
            "selected": selected_dense_h,
            "relative_to_p22": _relative(selected_dense_h, baseline_dense_h),
            "upstream_by_schedule": upstream_dense_h,
            "down_by_schedule": down_dense_h,
        },
        "w2_record_traversal_order": {
            "status": "research_only",
            "invariant": (
                "complete_128_channel_hadamard_records_and_canonical_channel_"
                "rate_assignments_are_preserved"
            ),
            "uniform_k2": {
                "baseline_dense_h": down_dense_h["n0"],
                "reverse_record_order": {
                    **uniform_reverse_evidence,
                    "relative_to_baseline": _relative(
                        uniform_reverse_dense_h,
                        down_dense_h["n0"],
                    ),
                },
            },
            "mixed_rate": mixed_order_study,
        },
        "channel_group_reencode": {
            "status": "evaluated" if fine_down_maps else "no_profitable_marginal",
            "proposal_metric": (
                "zero_feedback_decoded_candidates_scored_by_absolute_dense_h"
            ),
            "evaluation_metric": (
                "complete_mixed_rate_blockldlq_absolute_dense_h_distortion"
            ),
            "baseline_w2_dense_h": down_dense_h["n0"],
            "candidates": {
                name: {
                    "allocation": fine_down_plans[name],
                    "dense_h": fine_down_dense_h[name],
                    "dense_h_relative_to_k2": _relative(
                        fine_down_dense_h[name],
                        down_dense_h["n0"],
                    ),
                    "routed_expert_output_diagnostic": fine_down_scores[name],
                }
                for name in fine_down_maps
            },
        },
        "tile_pair_reencode": {
            "status": "evaluated" if tile_down_maps else "no_profitable_pair",
            "proposal_metric": (
                "zero_feedback_decoded_candidates_scored_by_absolute_dense_h"
            ),
            "evaluation_metric": (
                "complete_mixed_rate_blockldlq_absolute_dense_h_distortion"
            ),
            "baseline_w2_dense_h": down_dense_h["n0"],
            "selector": {
                "grammar": (
                    "one P22-or-P13 bit for each corresponding 16x16 tile "
                    "pair in twelve low/high 128-channel record pairs"
                ),
                "bits_per_expert": 12 * 8 * 224,
                "bytes_per_expert": (12 * 8 * 224) // 8,
                "bits_per_weight": (12 * 8 * 224) / (3 * 3072 * 3584),
            },
            "candidates": {
                name: {
                    "allocation": tile_down_plans[name],
                    "dense_h": fine_down_dense_h[name],
                    "dense_h_relative_to_k2": _relative(
                        fine_down_dense_h[name],
                        down_dense_h["n0"],
                    ),
                    "routed_expert_output_diagnostic": fine_down_scores[name],
                }
                for name in tile_down_maps
            },
        },
        "w2_hessian": "expert_local_conditioned_on_decoded_upstream_candidate",
        "h2_evidence": {
            "uniform_k2_upstream": baseline_h2_evidence,
            "selected_upstream": selected_h2_evidence,
        },
        "representation": (
            {"basis": "production"}
            if representation is None
            else dict(representation)
        ),
        "routed_expert_output_diagnostic": {
            "baseline_p22": baseline,
            "selected": selected_score,
            "confirmation_relative_to_p22": _relative(
                float(selected_score["confirmation"]["sse"]),
                float(baseline["confirmation"]["sse"]),
            ),
        },
    }


def _weighted_functional_sse(
    reconstruction: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    inputs: torch.Tensor,
    reference: torch.Tensor,
    route_weights: torch.Tensor,
    execute_triplet: TripletExecutor | None = None,
) -> float:
    candidate = (
        _execute_standard_triplet(inputs, reconstruction)
        if execute_triplet is None
        else execute_triplet(inputs, reconstruction)
    )
    return float(
        ((candidate - reference).square() * route_weights).sum(dtype=torch.float64)
    )


def _qsrt_308_search(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    uniform: Mapping[str, Mapping[int, Mapping[str, object]]],
    errors: Mapping[str, Mapping[int, torch.Tensor]],
    *,
    h13: torch.Tensor,
    h2: torch.Tensor,
    contexts: torch.Tensor,
    layer: int,
    expert: int,
    device: torch.device,
    quantizer_module,
    ldlq_tf32: bool,
    inputs: torch.Tensor,
    gates: torch.Tensor,
    fit_mask: torch.Tensor,
    confirmation_mask: torch.Tensor,
    permutation_override: torch.Tensor,
    include_tile_fractions: bool,
    max_donors: int,
    scale_closure: bool,
    decode_middle: MiddleDecoder | None = None,
    execute_triplet: TripletExecutor | None = None,
    reference_output: torch.Tensor | None = None,
    representation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Compare exact 74-bit record schedules and optional tile-funded schedules."""

    middle_decoder = (
        _decode_standard_middle if decode_middle is None else decode_middle
    )
    triplet_executor = (
        _execute_standard_triplet if execute_triplet is None else execute_triplet
    )

    expert_weight_count = 3 * 3072 * 3584
    strip_count = 8 * 224
    payload_bpw = 74 / 24
    paired_selector_bytes = 4 * strip_count
    # Each matrix family selects an unordered pair from 24 records.  Its
    # combinadic rank needs nine bits because C(24, 2) = 276.  Pack the W13
    # and W2 pair ranks into one canonical u32 per strip; load preparation may
    # expand that word into two independent 24-bit masks for the kernel.
    top2_selector_bytes = 4 * strip_count
    top2_prepared_selector_bytes = 8 * strip_count
    arbitrary_selector_bytes = 16 * strip_count

    tile_fractions: dict[str, float | None] = {
        "tile_p000": 0.0,
        "tile_p025": 0.25,
        "tile_p050": 0.5,
        "tile_p075": 0.75,
        "tile_p100": 1.0,
        "tile_positive": None,
    }
    upstream_tile_maps: dict[str, torch.Tensor] = {}
    downstream_tile_maps: dict[str, torch.Tensor] = {}
    upstream_tile_evidence: dict[str, object] = {}
    down_tile_evidence: dict[str, object] = {}
    if include_tile_fractions:
        for name, fraction in tile_fractions.items():
            upstream_tile_maps[name], upstream_tile_evidence[name] = (
                _qsrt_308_tile_funded_map(
                    (errors["w1"], errors["w3"]),
                    rate_axis="n",
                    fraction=fraction,
                )
            )
            downstream_tile_maps[name], down_tile_evidence[name] = (
                _qsrt_308_tile_funded_map(
                    (errors["w2"],), rate_axis="k", fraction=fraction
                )
            )
        for name, replacement_fraction in (
            ("tile_k4_top2_p010", 0.01),
            ("tile_k4_top2_p050", 0.05),
            ("tile_k4_top2_p250", 0.25),
            ("tile_k4_top2", 1.0),
        ):
            upstream_tile_maps[name], upstream_tile_evidence[name] = (
                _qsrt_308_top2_k4_map(
                    (errors["w1"], errors["w3"]),
                    rate_axis="n",
                    replacement_fraction=replacement_fraction,
                )
            )
            downstream_tile_maps[name], down_tile_evidence[name] = (
                _qsrt_308_top2_k4_map(
                    (errors["w2"],),
                    rate_axis="k",
                    replacement_fraction=replacement_fraction,
                )
            )
        upstream_tile_maps["tile_strip_optimal"], upstream_tile_evidence[
            "tile_strip_optimal"
        ] = _qsrt_308_strip_optimal_map(
            (errors["w1"], errors["w3"]), rate_axis="n"
        )
        downstream_tile_maps["tile_strip_optimal"], down_tile_evidence[
            "tile_strip_optimal"
        ] = _qsrt_308_strip_optimal_map((errors["w2"],), rate_axis="k")
    upstream_maps = {
        f"n{donors}": _qsrt_308_record_map(
            (3584, 3072), donors, rate_axis="n"
        )
        for donors in range(max_donors + 1)
    }
    down_maps = {
        f"n{donors}": _qsrt_308_record_map(
            (3072, 3584), donors, rate_axis="k"
        )
        for donors in range(max_donors + 1)
    }
    upstream_maps.update(upstream_tile_maps)
    down_maps.update(downstream_tile_maps)

    prepared_upstream = {
        matrix: _prepare_quantize_maps(
            source[MATRICES.index(matrix)],
            h13,
            contexts,
            matrix=matrix,
            device=device,
            permutation_override=permutation_override,
        )
        for matrix in ("w1", "w3")
    }
    encoded: dict[str, dict[str, dict[str, object]]] = {}
    for matrix in ("w1", "w3"):
        encoded[matrix], _ = _quantize_maps(
            source[MATRICES.index(matrix)],
            h13,
            contexts,
            matrix=matrix,
            maps=upstream_maps,
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            prepared=prepared_upstream[matrix],
        )

    fit_inputs = inputs[fit_mask]
    fit_gates = gates[fit_mask]
    conditional_down: dict[str, dict[str, dict[str, object]]] = {}
    conditional_h2_evidence: dict[str, object] = {}
    for upstream_name in upstream_maps:
        w1 = encoded["w1"][upstream_name]["reconstruction"]
        w3 = encoded["w3"][upstream_name]["reconstruction"]
        if not isinstance(w1, torch.Tensor) or not isinstance(w3, torch.Tensor):
            raise TypeError("3.08-bpw upstream reconstruction is not a tensor")
        decoded_middle = middle_decoder(fit_inputs, w1, w3)
        _, candidate_h2, evidence = build_expert_hessians(
            fit_inputs,
            fit_gates,
            decoded_middle,
            global_h13=h13,
            global_h2=h2,
            device=device,
        )
        conditional_down[upstream_name], _ = _quantize_maps(
            source[2],
            candidate_h2,
            contexts,
            matrix="w2",
            maps=down_maps,
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            permutation_override=permutation_override,
        )
        conditional_h2_evidence[upstream_name] = evidence

    baseline_w1 = uniform["w1"][3]["reconstruction"]
    baseline_w3 = uniform["w3"][3]["reconstruction"]
    if not isinstance(baseline_w1, torch.Tensor) or not isinstance(
        baseline_w3, torch.Tensor
    ):
        raise TypeError("uniform K3 upstream reconstruction is not a tensor")
    baseline_middle = middle_decoder(fit_inputs, baseline_w1, baseline_w3)
    _, baseline_h2, baseline_h2_evidence = build_expert_hessians(
        fit_inputs,
        fit_gates,
        baseline_middle,
        global_h13=h13,
        global_h2=h2,
        device=device,
    )
    prepared_baseline_down = _prepare_quantize_maps(
        source[2],
        baseline_h2,
        contexts,
        matrix="w2",
        device=device,
        permutation_override=permutation_override,
    )
    baseline_down, _ = _quantize_maps(
        source[2],
        baseline_h2,
        contexts,
        matrix="w2",
        maps={"k3": _uniform_tile_map((3072, 3584), 3)},
        layer=layer,
        expert=expert,
        device=device,
        quantizer_module=quantizer_module,
        ldlq_tf32=ldlq_tf32,
        prepared=prepared_baseline_down,
    )

    if reference_output is None:
        reference_output = triplet_executor(inputs, source)
    if reference_output.shape != (inputs.shape[0], source[2].shape[0]):
        raise ValueError("reference output does not align with the expert rows")
    fit_target = reference_output[fit_mask]
    confirmation_target = reference_output[confirmation_mask]
    fit_weights = gates[fit_mask].float().square().unsqueeze(1)
    confirmation_weights = gates[confirmation_mask].float().square().unsqueeze(1)
    fit_reference_energy = float(
        (fit_target.square() * fit_weights).sum(dtype=torch.float64)
    )
    confirmation_reference_energy = float(
        (confirmation_target.square() * confirmation_weights).sum(dtype=torch.float64)
    )

    baseline_reconstruction = (
        baseline_w1,
        baseline_w3,
        baseline_down["k3"]["reconstruction"],
    )
    if not all(isinstance(value, torch.Tensor) for value in baseline_reconstruction):
        raise TypeError("uniform K3 reconstruction is not a tensor")
    baseline_fit_sse = _weighted_functional_sse(
        baseline_reconstruction,  # type: ignore[arg-type]
        inputs=inputs[fit_mask],
        reference=fit_target,
        route_weights=fit_weights,
        execute_triplet=triplet_executor,
    )
    baseline_confirmation_sse = _weighted_functional_sse(
        baseline_reconstruction,  # type: ignore[arg-type]
        inputs=inputs[confirmation_mask],
        reference=confirmation_target,
        route_weights=confirmation_weights,
        execute_triplet=triplet_executor,
    )
    scores: dict[str, dict[str, dict[str, float]]] = {}
    for upstream_name in upstream_maps:
        w1 = encoded["w1"][upstream_name]["reconstruction"]
        w3 = encoded["w3"][upstream_name]["reconstruction"]
        if not isinstance(w1, torch.Tensor) or not isinstance(w3, torch.Tensor):
            raise TypeError("3.08-bpw upstream reconstruction is not a tensor")
        for down_name in down_maps:
            w2 = conditional_down[upstream_name][down_name]["reconstruction"]
            if not isinstance(w2, torch.Tensor):
                raise TypeError("3.08-bpw down reconstruction is not a tensor")
            key = f"u{upstream_name}/d{down_name}"
            fit_sse = _weighted_functional_sse(
                (w1, w3, w2),
                inputs=inputs[fit_mask],
                reference=fit_target,
                route_weights=fit_weights,
                execute_triplet=triplet_executor,
            )
            confirmation_sse = _weighted_functional_sse(
                (w1, w3, w2),
                inputs=inputs[confirmation_mask],
                reference=confirmation_target,
                route_weights=confirmation_weights,
                execute_triplet=triplet_executor,
            )
            scores[key] = {
                "fit": {
                    "sse": fit_sse,
                    "reference_energy": fit_reference_energy,
                    "nmse": fit_sse / fit_reference_energy,
                },
                "confirmation": {
                    "sse": confirmation_sse,
                    "reference_energy": confirmation_reference_energy,
                    "nmse": confirmation_sse / confirmation_reference_energy,
                },
            }
    selected = min(scores, key=lambda name: float(scores[name]["fit"]["sse"]))
    confirmation_oracle = min(
        scores, key=lambda name: float(scores[name]["confirmation"]["sse"])
    )
    selected_confirmation = float(scores[selected]["confirmation"]["sse"])

    serial_cache: dict[
        tuple[str, str], tuple[dict[str, dict[str, float]], dict[str, object]]
    ] = {}

    def serial_reencode(
        upstream_name: str, down_name: str
    ) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
        """Re-encode one complete schedule independently of candidate batching.

        Hard trellis paths can expose implementation defects or numerical
        sensitivity when otherwise identical maps are embedded in differently
        sized candidate batches.  Candidate batches remain useful for proposal
        selection, but the scientific comparison is closed with one isolated
        encode of the chosen map and one isolated encode of its baseline.
        """

        cache_key = (upstream_name, down_name)
        cached = serial_cache.get(cache_key)
        if cached is not None:
            return cached

        upstream_rate_map = (
            _uniform_tile_map((3584, 3072), 3)
            if upstream_name == "p33"
            else upstream_maps[upstream_name]
        )
        down_rate_map = (
            _uniform_tile_map((3072, 3584), 3)
            if down_name == "p33"
            else down_maps[down_name]
        )
        serial_upstream: dict[str, dict[str, object]] = {}
        for matrix in ("w1", "w3"):
            serial_group, _ = _quantize_maps(
                source[MATRICES.index(matrix)],
                h13,
                contexts,
                matrix=matrix,
                maps={"candidate": upstream_rate_map},
                layer=layer,
                expert=expert,
                device=device,
                quantizer_module=quantizer_module,
                ldlq_tf32=ldlq_tf32,
                prepared=prepared_upstream[matrix],
            )
            serial_upstream[matrix] = serial_group["candidate"]

        w1 = serial_upstream["w1"]["reconstruction"]
        w3 = serial_upstream["w3"]["reconstruction"]
        if not isinstance(w1, torch.Tensor) or not isinstance(w3, torch.Tensor):
            raise TypeError("serial upstream reconstruction is not a tensor")
        decoded_middle = middle_decoder(fit_inputs, w1, w3)
        _, serial_h2, serial_h2_evidence = build_expert_hessians(
            fit_inputs,
            fit_gates,
            decoded_middle,
            global_h13=h13,
            global_h2=h2,
            device=device,
        )
        serial_down_group, _ = _quantize_maps(
            source[2],
            serial_h2,
            contexts,
            matrix="w2",
            maps={"candidate": down_rate_map},
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            permutation_override=permutation_override,
        )
        serial_down = serial_down_group["candidate"]
        w2 = serial_down["reconstruction"]
        if not isinstance(w2, torch.Tensor):
            raise TypeError("serial down reconstruction is not a tensor")

        fit_sse = _weighted_functional_sse(
            (w1, w3, w2),
            inputs=inputs[fit_mask],
            reference=fit_target,
            route_weights=fit_weights,
            execute_triplet=triplet_executor,
        )
        confirmation_sse = _weighted_functional_sse(
            (w1, w3, w2),
            inputs=inputs[confirmation_mask],
            reference=confirmation_target,
            route_weights=confirmation_weights,
            execute_triplet=triplet_executor,
        )
        batch_upstream = {
            matrix: (
                uniform[matrix][3]
                if upstream_name == "p33"
                else encoded[matrix][upstream_name]
            )
            for matrix in ("w1", "w3")
        }
        batch_down = (
            baseline_down["k3"]
            if upstream_name == "p33" and down_name == "p33"
            else conditional_down[upstream_name][down_name]
        )
        evidence = {
            "upstream": {
                matrix: {
                    "states_equal_to_batch": torch.equal(
                        serial_upstream[matrix]["states"],
                        batch_upstream[matrix]["states"],
                    ),
                    "state_disagreement": int(
                        torch.count_nonzero(
                            serial_upstream[matrix]["states"]
                            != batch_upstream[matrix]["states"]
                        )
                    ),
                    "g_scale_serial": float(serial_upstream[matrix]["g_scale"]),
                    "g_scale_batch": float(batch_upstream[matrix]["g_scale"]),
                }
                for matrix in ("w1", "w3")
            },
            "down": {
                "states_equal_to_batch": torch.equal(
                    serial_down["states"], batch_down["states"]
                ),
                "state_disagreement": int(
                    torch.count_nonzero(
                        serial_down["states"] != batch_down["states"]
                    )
                ),
                "g_scale_serial": float(serial_down["g_scale"]),
                "g_scale_batch": float(batch_down["g_scale"]),
            },
            "h2": serial_h2_evidence,
        }
        result = (
            {
                "fit": {
                    "sse": fit_sse,
                    "reference_energy": fit_reference_energy,
                    "nmse": fit_sse / fit_reference_energy,
                },
                "confirmation": {
                    "sse": confirmation_sse,
                    "reference_energy": confirmation_reference_energy,
                    "nmse": confirmation_sse / confirmation_reference_energy,
                },
            },
            evidence,
        )
        serial_cache[cache_key] = result
        return result

    selected_upstream, selected_down = selected.split("/")
    selected_upstream = selected_upstream.removeprefix("u")
    selected_down = selected_down.removeprefix("d")
    serial_p33_score, serial_p33_evidence = serial_reencode("p33", "p33")
    serial_floor_score, serial_floor_evidence = serial_reencode("n0", "n0")
    if selected_upstream == "n0" and selected_down == "n0":
        serial_selected_score = serial_floor_score
        serial_selected_evidence = serial_floor_evidence
    else:
        serial_selected_score, serial_selected_evidence = serial_reencode(
            selected_upstream, selected_down
        )
    batch_shortlist_names = sorted(
        scores,
        key=lambda name: float(scores[name]["fit"]["sse"]),
    )[:5]
    serial_batch_shortlist: dict[str, dict[str, object]] = {}
    for name in batch_shortlist_names:
        upstream_name, down_name = name.split("/")
        upstream_name = upstream_name.removeprefix("u")
        down_name = down_name.removeprefix("d")
        score, evidence = serial_reencode(upstream_name, down_name)
        serial_batch_shortlist[name] = {
            "score": score,
            "batch_closure": evidence,
        }
    serial_tile_controls: dict[str, dict[str, object]] = {}
    if include_tile_fractions:
        for top2_name in (
            "tile_k4_top2_p010",
            "tile_k4_top2_p050",
            "tile_k4_top2_p250",
            "tile_k4_top2",
        ):
            for upstream_name, down_name in (
                (top2_name, "n0"),
                ("n0", top2_name),
                (top2_name, top2_name),
            ):
                score, evidence = serial_reencode(upstream_name, down_name)
                serial_tile_controls[f"u{upstream_name}/d{down_name}"] = {
                    "score": score,
                    "batch_closure": evidence,
                }
    serial_fit_candidates = {
        "un0/dn0": serial_floor_score,
        f"u{selected_upstream}/d{selected_down}": serial_selected_score,
        **{
            name: value["score"]
            for name, value in serial_batch_shortlist.items()
        },
        **{
            name: value["score"]
            for name, value in serial_tile_controls.items()
        },
    }
    serial_fit_selected = min(
        serial_fit_candidates,
        key=lambda name: float(serial_fit_candidates[name]["fit"]["sse"]),
    )
    serial_fit_selected_score = serial_fit_candidates[serial_fit_selected]
    serial_confirmation_relative_to_p33 = _relative(
        float(serial_selected_score["confirmation"]["sse"]),
        float(serial_p33_score["confirmation"]["sse"]),
    )

    def scale_close_schedule(
        upstream_name: str, down_name: str
    ) -> dict[str, object]:
        """Fit mixed-schedule scales without using confirmation documents.

        Every multiplier triggers a complete hard-trellis re-encode of the
        requested map.  Gate and up scales are selected jointly by complete
        expert-function SSE with exact source down weights.  Their decoded
        activations then define the candidate-local H2 used to select the down
        scale by complete expert-function SSE.  Confirmation is evaluated only
        for the resulting fit-selected triplet.
        """

        upstream_rate_map = (
            _uniform_tile_map((3584, 3072), 3)
            if upstream_name == "p33"
            else upstream_maps[upstream_name]
        )
        down_rate_map = (
            _uniform_tile_map((3072, 3584), 3)
            if down_name == "p33"
            else down_maps[down_name]
        )
        upstream_candidates: dict[str, list[dict[str, object]]] = {}
        for matrix in ("w1", "w3"):
            baseline_scale = float(uniform[matrix][3]["g_scale"])
            variants = []
            for multiplier in SCALE_CLOSURE_MULTIPLIERS:
                group, _ = _quantize_maps(
                    source[MATRICES.index(matrix)],
                    h13,
                    contexts,
                    matrix=matrix,
                    maps={"candidate": upstream_rate_map},
                    layer=layer,
                    expert=expert,
                    device=device,
                    quantizer_module=quantizer_module,
                    ldlq_tf32=ldlq_tf32,
                    g_scale_override=baseline_scale * multiplier,
                    prepared=prepared_upstream[matrix],
                )
                variants.append(group["candidate"])
            upstream_candidates[matrix] = variants

        upstream_fit = torch.empty(
            (
                len(SCALE_CLOSURE_MULTIPLIERS),
                len(SCALE_CLOSURE_MULTIPLIERS),
            ),
            dtype=torch.float64,
        )
        for w1_index, w1_candidate in enumerate(upstream_candidates["w1"]):
            w1 = w1_candidate["reconstruction"]
            if not isinstance(w1, torch.Tensor):
                raise TypeError("scale-closure w1 reconstruction is not a tensor")
            for w3_index, w3_candidate in enumerate(upstream_candidates["w3"]):
                w3 = w3_candidate["reconstruction"]
                if not isinstance(w3, torch.Tensor):
                    raise TypeError("scale-closure w3 reconstruction is not a tensor")
                upstream_fit[w1_index, w3_index] = _weighted_functional_sse(
                    (w1, w3, source[2]),
                    inputs=fit_inputs,
                    reference=fit_target,
                    route_weights=fit_weights,
                    execute_triplet=triplet_executor,
                )
        upstream_flat = int(torch.argmin(upstream_fit))
        width = len(SCALE_CLOSURE_MULTIPLIERS)
        w1_index, w3_index = divmod(upstream_flat, width)
        selected_w1 = upstream_candidates["w1"][w1_index]
        selected_w3 = upstream_candidates["w3"][w3_index]
        w1 = selected_w1["reconstruction"]
        w3 = selected_w3["reconstruction"]
        if not isinstance(w1, torch.Tensor) or not isinstance(w3, torch.Tensor):
            raise TypeError("scale-closure upstream reconstruction is not a tensor")

        decoded_middle = middle_decoder(fit_inputs, w1, w3)
        _, closed_h2, closed_h2_evidence = build_expert_hessians(
            fit_inputs,
            fit_gates,
            decoded_middle,
            global_h13=h13,
            global_h2=h2,
            device=device,
        )
        down_variants = []
        baseline_down_scale = float(uniform["w2"][3]["g_scale"])
        for multiplier in SCALE_CLOSURE_MULTIPLIERS:
            group, _ = _quantize_maps(
                source[2],
                closed_h2,
                contexts,
                matrix="w2",
                maps={"candidate": down_rate_map},
                layer=layer,
                expert=expert,
                device=device,
                quantizer_module=quantizer_module,
                ldlq_tf32=ldlq_tf32,
                g_scale_override=baseline_down_scale * multiplier,
                permutation_override=permutation_override,
            )
            down_variants.append(group["candidate"])
        down_fit = torch.empty(width, dtype=torch.float64)
        for index, candidate in enumerate(down_variants):
            w2 = candidate["reconstruction"]
            if not isinstance(w2, torch.Tensor):
                raise TypeError("scale-closure down reconstruction is not a tensor")
            down_fit[index] = _weighted_functional_sse(
                (w1, w3, w2),
                inputs=fit_inputs,
                reference=fit_target,
                route_weights=fit_weights,
                execute_triplet=triplet_executor,
            )
        w2_index = int(torch.argmin(down_fit))
        selected_w2 = down_variants[w2_index]
        w2 = selected_w2["reconstruction"]
        if not isinstance(w2, torch.Tensor):
            raise TypeError("scale-closure selected down reconstruction is not a tensor")
        confirmation_sse = _weighted_functional_sse(
            (w1, w3, w2),
            inputs=inputs[confirmation_mask],
            reference=confirmation_target,
            route_weights=confirmation_weights,
            execute_triplet=triplet_executor,
        )
        fit_sse = float(down_fit[w2_index])
        return {
            "schedule": f"u{upstream_name}/d{down_name}",
            "multipliers": list(SCALE_CLOSURE_MULTIPLIERS),
            "selected_multiplier": {
                "w1": SCALE_CLOSURE_MULTIPLIERS[w1_index],
                "w3": SCALE_CLOSURE_MULTIPLIERS[w3_index],
                "w2": SCALE_CLOSURE_MULTIPLIERS[w2_index],
            },
            "selected_g_scale": {
                "w1": float(selected_w1["g_scale"]),
                "w3": float(selected_w3["g_scale"]),
                "w2": float(selected_w2["g_scale"]),
            },
            "upstream_fit_sse": upstream_fit.tolist(),
            "down_fit_sse": down_fit.tolist(),
            "h2": closed_h2_evidence,
            "score": {
                "fit": {
                    "sse": fit_sse,
                    "reference_energy": fit_reference_energy,
                    "nmse": fit_sse / fit_reference_energy,
                },
                "confirmation": {
                    "sse": confirmation_sse,
                    "reference_energy": confirmation_reference_energy,
                    "nmse": confirmation_sse / confirmation_reference_energy,
                },
            },
        }

    scale_closure_result = None
    if scale_closure:
        closed_upstream, closed_down = serial_fit_selected.split("/")
        closed_upstream = closed_upstream.removeprefix("u")
        closed_down = closed_down.removeprefix("d")
        closed_high_rate = scale_close_schedule(closed_upstream, closed_down)
        closed_p33 = scale_close_schedule("p33", "p33")
        closed_candidates = {
            "high_rate": closed_high_rate,
            "p33": closed_p33,
        }
        closed_selected = min(
            closed_candidates,
            key=lambda name: float(
                closed_candidates[name]["score"]["fit"]["sse"]  # type: ignore[index]
            ),
        )
        scale_closure_result = {
            "selection_partition": "fit_only",
            "selection_order": (
                "joint_w1_w3_with_exact_source_w2_then_conditional_h2_then_w2"
            ),
            "candidates": closed_candidates,
            "selected_on_fit": closed_selected,
            "confirmation_relative_to_closed_p33": _relative(
                float(
                    closed_candidates[closed_selected]["score"]["confirmation"]["sse"]  # type: ignore[index]
                ),
                float(closed_p33["score"]["confirmation"]["sse"]),  # type: ignore[index]
            ),
        }
    return {
        "representation": (
            dict(representation)
            if representation is not None
            else {
                "basis": "ordinary_expert_coordinates",
                "execution": "SiTU(w1*x,w3*x)_then_w2",
            }
        ),
        "payload_trellis_bits_per_weight": payload_bpw,
        "storage_summary": {
            "expert_weight_count": expert_weight_count,
            "strip_count": strip_count,
            "paired_p33_p24": {
                "direct_selector_bytes_per_expert": paired_selector_bytes,
                "payload_plus_selector_bpw": (
                    payload_bpw
                    + paired_selector_bytes * 8 / expert_weight_count
                ),
            },
            "top2_k4": {
                "canonical_selector_bytes_per_expert": top2_selector_bytes,
                "canonical_selector": (
                    "one_u32_per_strip_with_two_9bit_combinadic_record_pairs"
                ),
                "prepared_selector_bytes_per_expert": (
                    top2_prepared_selector_bytes
                ),
                "prepared_selector": (
                    "two_disposable_u32_24bit_record_masks_per_strip"
                ),
                "payload_plus_selector_bpw": (
                    payload_bpw + top2_selector_bytes * 8 / expert_weight_count
                ),
            },
            "arbitrary_strip_optimal": {
                "direct_selector_bytes_per_expert": arbitrary_selector_bytes,
                "payload_plus_selector_bpw": (
                    payload_bpw
                    + arbitrary_selector_bytes * 8 / expert_weight_count
                ),
            },
        },
        "coordinate_contract": {
            "order": (
                "common_four_channel_group_permutation_then_"
                "16_channel_tiling_then_rate_funding"
            ),
            "shared_permutation": ["w1_rows", "w3_rows", "w2_columns"],
            "permutation_search_unit_channels": 4,
            "groups_per_funding_tile": 4,
            "funding_coordinates": "post_permutation_encoder_tiles",
            "upstream_rate_axis": "encoder_n",
            "down_rate_axis": "encoder_k",
            "budget_unit": "corresponding_low_high_16x16_tile_pair",
        },
        "record_schedule_identity": "N*K2 + (22-2N)*K3 + (N+2)*K4",
        "record_donor_counts_tested": list(range(max_donors + 1)),
        "record_storage": {
            "fixed_wide_pair": "records_22_23_are_P44",
            "fixed_six_bit_pairs": 11,
            "variable_pair_kinds": ["P33", "P24"],
            "mode_bits_per_expert": 8,
            "mode_encoding": "four_bit_upstream_N_plus_four_bit_down_N",
        },
        "tile_funding": {
            "grammar": (
                "paired controls use one fixed P44 pair plus eleven P33/P24 "
                "slots; strip_optimal permits any K2/K3/K4 assignment whose "
                "24 rates sum to 74"
            ),
            "evaluated": include_tile_fractions,
            "fractions_tested": tile_fractions if include_tile_fractions else {},
            "upstream": upstream_tile_evidence,
            "down": down_tile_evidence,
            "logical_metadata_bits": 2 * 11 * 8 * 224,
            "logical_metadata_bytes": (2 * 11 * 8 * 224) // 8,
            "logical_metadata_bpw_all_expert_weights": (
                2 * 11 * 8 * 224 / (3 * 3072 * 3584)
            ),
            "direct_selector": (
                "one_u32_per_strip_with_11_w13_bits_and_11_w2_bits"
            ),
            "payload_layout": (
                "eleven_fixed_192_byte_low_high_slots_plus_one_fixed_p44_slot"
            ),
            "serving_access": "direct_pair_stride_without_prefix_offsets",
            "direct_selector_bytes_per_expert": paired_selector_bytes,
            "direct_selector_bpw_all_expert_weights": (
                paired_selector_bytes * 8 / expert_weight_count
            ),
            "strip_optimal_naive_metadata": {
                "bits_per_expert": 2 * 2 * 24 * 8 * 224,
                "bytes_per_expert": (2 * 2 * 24 * 8 * 224) // 8,
                "bits_per_all_expert_weight": (
                    2 * 2 * 24 * 8 * 224 / (3 * 3072 * 3584)
                ),
                "note": "one 2-bit rate code per tile for shared upstream and down maps",
                "direct_selector": (
                    "two_64bit_rate_words_per_strip_one_for_w13_one_for_w2"
                ),
                "direct_selector_bytes_per_expert": arbitrary_selector_bytes,
                "serving_status": (
                    "research_oracle_variable_rate_counts_need_a_qualified_"
                    "prepared_offset_or_work_queue_layout"
                ),
            },
        },
        "scale": {
            "production_policy": (
                "one_source_local_uniform_k3_scale_shared_across_rate_schedules"
            ),
            "storage": "folded_into_existing_suh_svh_metadata",
            "qualification_control": (
                "path_aware_schedule_specific_scale_search_on_fit_then_"
                "confirmation_score_once"
            ),
            "closure": scale_closure_result,
        },
        "down_hessian": {
            "policy": "decoded_upstream_candidate_post_situ_adaptive_identity_shrinkage",
            "baseline_p33": baseline_h2_evidence,
            "by_upstream_schedule": conditional_h2_evidence,
        },
        "scores": scores,
        "selected_on_fit": selected,
        "confirmation_oracle": confirmation_oracle,
        "serial_validation": {
            "selected": serial_selected_score,
            "p33": serial_p33_score,
            "fixed_p44_floor": serial_floor_score,
            "confirmation_relative_to_p33": serial_confirmation_relative_to_p33,
            "confirmation_relative_to_fixed_p44_floor": _relative(
                float(serial_selected_score["confirmation"]["sse"]),
                float(serial_floor_score["confirmation"]["sse"]),
            ),
            "selected_batch_closure": serial_selected_evidence,
            "p33_batch_closure": serial_p33_evidence,
            "fixed_p44_floor_batch_closure": serial_floor_evidence,
            "batch_top5_shortlist": serial_batch_shortlist,
            "tile_k4_top2_controls": serial_tile_controls,
            "selected_on_serial_fit": serial_fit_selected,
            "serial_fit_selected": serial_fit_selected_score,
            "serial_fit_selected_confirmation_relative_to_p33": _relative(
                float(serial_fit_selected_score["confirmation"]["sse"]),
                float(serial_p33_score["confirmation"]["sse"]),
            ),
            "serial_fit_selected_confirmation_relative_to_fixed_p44_floor": _relative(
                float(serial_fit_selected_score["confirmation"]["sse"]),
                float(serial_floor_score["confirmation"]["sse"]),
            ),
            "selection_source": (
                "isolated_serial_fit_over_batch_top5_plus_predeclared_controls"
            ),
            "validation_source": "independent_single_candidate_reencode",
        },
        "p33": {
            "fit_sse": baseline_fit_sse,
            "confirmation_sse": baseline_confirmation_sse,
        },
        "confirmation_relative_to_p33": _relative(
            selected_confirmation, baseline_confirmation_sse
        ),
    }


def _tile_prefix_counts(deltas: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    order = torch.argsort(deltas, stable=True)
    total = deltas.numel()
    negative = int(torch.count_nonzero(deltas < 0))
    counts = {
        0,
        total,
        negative,
        *(
            min(total, value)
            for value in (
                1,
                2,
                4,
                8,
                16,
                32,
                64,
                128,
                256,
                384,
                512,
                768,
                1024,
                1280,
                1536,
            )
        ),
    }
    return order, tuple(sorted(counts))


def _functional_tile_pair_search_independent(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    uniform: Mapping[str, Mapping[int, Mapping[str, object]]],
    *,
    h13: torch.Tensor,
    h2: torch.Tensor,
    contexts: torch.Tensor,
    layer: int,
    expert: int,
    device: torch.device,
    quantizer_module,
    ldlq_tf32: bool,
    inputs: torch.Tensor,
    gates: torch.Tensor,
    fit_mask: torch.Tensor,
    confirmation_mask: torch.Tensor,
    chunk_size: int,
    pair_limit: int,
    permutation_override: torch.Tensor,
    base_upstream_donors: int | None = None,
    base_down_donors: int | None = None,
    proposal_deltas: tuple[torch.Tensor, torch.Tensor] | None = None,
    decode_middle: MiddleDecoder | None = None,
    execute_triplet: TripletExecutor | None = None,
    reference_output: torch.Tensor | None = None,
    representation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Search shared gate/up and independent down P33/P24 tile maps."""

    middle_decoder = (
        _decode_standard_middle if decode_middle is None else decode_middle
    )
    triplet_executor = (
        _execute_standard_triplet if execute_triplet is None else execute_triplet
    )

    pair_count = 224 * 8
    evaluated_count = pair_count if pair_limit <= 0 else min(pair_limit, pair_count)
    if chunk_size <= 0:
        raise ValueError("functional tile search chunk size must be positive")
    boundary_search = base_upstream_donors is not None
    if boundary_search != (base_down_donors is not None):
        raise ValueError("boundary tile search needs both donor counts")
    if boundary_search and (
        not 0 <= int(base_upstream_donors) < 11
        or not 0 <= int(base_down_donors) < 11
    ):
        raise ValueError("boundary donor counts must lie in 0..10")
    if proposal_deltas is not None:
        if not boundary_search:
            raise ValueError("proxy proposals are defined only at a 3.08 boundary")
        if any(delta.ndim != 1 or delta.numel() != pair_count for delta in proposal_deltas):
            raise ValueError("proxy proposal deltas must cover all boundary tiles")

    def rate_map(matrix: str, selected: Sequence[int]) -> tuple[int, ...]:
        shape = (3584, 3072) if matrix in ("w1", "w3") else (3072, 3584)
        rate_axis = "n" if matrix in ("w1", "w3") else "k"
        if boundary_search:
            donors = (
                int(base_upstream_donors)
                if matrix in ("w1", "w3")
                else int(base_down_donors)
            )
            return _qsrt_308_boundary_tile_map(
                shape, donors, selected, rate_axis=rate_axis
            )
        return _paired_tile_map(shape, selected, rate_axis=rate_axis)

    prepared_upstream = {
        matrix: _prepare_quantize_maps(
            source[MATRICES.index(matrix)],
            h13,
            contexts,
            matrix=matrix,
            device=device,
            permutation_override=permutation_override,
        )
        for matrix in ("w1", "w3")
    }
    baseline_candidates: dict[str, Mapping[str, object]] = {}
    if boundary_search:
        for matrix in ("w1", "w3"):
            candidates, _ = _quantize_maps(
                source[MATRICES.index(matrix)],
                h13,
                contexts,
                matrix=matrix,
                maps={"baseline": rate_map(matrix, ())},
                layer=layer,
                expert=expert,
                device=device,
                quantizer_module=quantizer_module,
                ldlq_tf32=ldlq_tf32,
                prepared=prepared_upstream[matrix],
            )
            baseline_candidates[matrix] = candidates["baseline"]
    else:
        baseline_candidates = {
            matrix: uniform[matrix][3] for matrix in ("w1", "w3")
        }

    fit_inputs = inputs[fit_mask]
    fit_gates = gates[fit_mask]
    baseline_w1 = baseline_candidates["w1"]["reconstruction"]
    baseline_w3 = baseline_candidates["w3"]["reconstruction"]
    if not isinstance(baseline_w1, torch.Tensor) or not isinstance(
        baseline_w3, torch.Tensor
    ):
        raise TypeError("baseline upstream reconstruction is not a tensor")
    baseline_middle = middle_decoder(fit_inputs, baseline_w1, baseline_w3)
    _, baseline_h2, baseline_h2_evidence = build_expert_hessians(
        fit_inputs,
        fit_gates,
        baseline_middle,
        global_h13=h13,
        global_h2=h2,
        device=device,
    )
    prepared_baseline_down = _prepare_quantize_maps(
        source[2],
        baseline_h2,
        contexts,
        matrix="w2",
        device=device,
        permutation_override=permutation_override,
    )
    baseline_down, _ = _quantize_maps(
        source[2],
        baseline_h2,
        contexts,
        matrix="w2",
        maps={"baseline": rate_map("w2", ())},
        layer=layer,
        expert=expert,
        device=device,
        quantizer_module=quantizer_module,
        ldlq_tf32=ldlq_tf32,
        prepared=prepared_baseline_down,
    )
    baseline_candidates["w2"] = baseline_down["baseline"]
    baseline = {
        matrix: baseline_candidates[matrix]["reconstruction"] for matrix in MATRICES
    }
    if not all(isinstance(value, torch.Tensor) for value in baseline.values()):
        raise TypeError("baseline reconstruction is not a tensor")
    baseline_reconstruction = tuple(baseline[matrix] for matrix in MATRICES)
    if reference_output is None:
        reference_output = triplet_executor(inputs, source)
    if reference_output.shape != (inputs.shape[0], source[2].shape[0]):
        raise ValueError("reference output does not align with expert rows")
    fit_reference = reference_output[fit_mask]
    fit_route_weights = gates[fit_mask].float().square().unsqueeze(1)
    baseline_fit_sse = _weighted_functional_sse(
        baseline_reconstruction,  # type: ignore[arg-type]
        inputs=fit_inputs,
        reference=fit_reference,
        route_weights=fit_route_weights,
        execute_triplet=triplet_executor,
    )
    if proposal_deltas is None:
        marginal_upstream = torch.empty(evaluated_count, dtype=torch.float64)
        marginal_down = torch.empty(evaluated_count, dtype=torch.float64)
        screen_started = time.perf_counter()
        for begin in range(0, evaluated_count, chunk_size):
            end = min(begin + chunk_size, evaluated_count)
            names = tuple(f"t{coordinate:04d}" for coordinate in range(begin, end))
            selected_by_name = {
                name: (coordinate,)
                for name, coordinate in zip(names, range(begin, end), strict=True)
            }
            upstream: dict[str, dict[str, dict[str, object]]] = {}
            for matrix in ("w1", "w3"):
                maps = {
                    name: rate_map(matrix, selected)
                    for name, selected in selected_by_name.items()
                }
                upstream[matrix], _ = _quantize_maps(
                    source[MATRICES.index(matrix)],
                    h13,
                    contexts,
                    matrix=matrix,
                    maps=maps,
                    layer=layer,
                    expert=expert,
                    device=device,
                    quantizer_module=quantizer_module,
                    ldlq_tf32=ldlq_tf32,
                    g_scale_override=float(baseline_candidates[matrix]["g_scale"]),
                    prepared=prepared_upstream[matrix],
                )
            for offset, name in enumerate(names):
                reconstruction = (
                    upstream["w1"][name]["reconstruction"],
                    upstream["w3"][name]["reconstruction"],
                    baseline["w2"],
                )
                if not all(isinstance(value, torch.Tensor) for value in reconstruction):
                    raise TypeError("upstream toggle reconstruction is not a tensor")
                marginal_upstream[begin + offset] = _weighted_functional_sse(
                    reconstruction,  # type: ignore[arg-type]
                    inputs=fit_inputs,
                    reference=fit_reference,
                    route_weights=fit_route_weights,
                    execute_triplet=triplet_executor,
                )
            del upstream

            down_maps = {
                name: rate_map("w2", selected)
                for name, selected in selected_by_name.items()
            }
            down, _ = _quantize_maps(
                source[2],
                baseline_h2,
                contexts,
                matrix="w2",
                maps=down_maps,
                layer=layer,
                expert=expert,
                device=device,
                quantizer_module=quantizer_module,
                ldlq_tf32=ldlq_tf32,
                g_scale_override=float(baseline_candidates["w2"]["g_scale"]),
                prepared=prepared_baseline_down,
            )
            for offset, name in enumerate(names):
                reconstruction = (
                    baseline["w1"],
                    baseline["w3"],
                    down[name]["reconstruction"],
                )
                if not all(isinstance(value, torch.Tensor) for value in reconstruction):
                    raise TypeError("down toggle reconstruction is not a tensor")
                marginal_down[begin + offset] = _weighted_functional_sse(
                    reconstruction,  # type: ignore[arg-type]
                    inputs=fit_inputs,
                    reference=fit_reference,
                    route_weights=fit_route_weights,
                    execute_triplet=triplet_executor,
                )
            del down
            if end == evaluated_count or end % max(128, chunk_size) == 0:
                print(
                    f"layer {layer} expert {expert}: independent functional tile "
                    f"toggles {end}/{evaluated_count} in "
                    f"{time.perf_counter() - screen_started:.1f}s",
                    flush=True,
                )
            torch.cuda.empty_cache()
        deltas_upstream = marginal_upstream - baseline_fit_sse
        deltas_down = marginal_down - baseline_fit_sse
        proposal_metric = "single_toggle_complete_expert_functional_sse"
    else:
        deltas_upstream = proposal_deltas[0][:evaluated_count].double().cpu()
        deltas_down = proposal_deltas[1][:evaluated_count].double().cpu()
        proposal_metric = "regularized_weight_p24_minus_p33_proxy"
    del fit_reference, fit_route_weights
    order_upstream, counts_upstream = _tile_prefix_counts(deltas_upstream)
    order_down, counts_down = _tile_prefix_counts(deltas_down)

    upstream_prefixes: dict[str, dict[str, dict[str, object]]] = {}
    baseline_reencode_evidence: dict[str, object] = {}
    for matrix in ("w1", "w3"):
        maps = {
            f"n{count:04d}": rate_map(
                matrix, order_upstream[:count].tolist()
            )
            for count in counts_upstream
        }
        upstream_prefixes[matrix], _ = _quantize_maps(
            source[MATRICES.index(matrix)],
            h13,
            contexts,
            matrix=matrix,
            maps=maps,
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            prepared=prepared_upstream[matrix],
        )
        all_p33 = upstream_prefixes[matrix]["n0000"]
        reference_baseline = baseline_candidates[matrix]
        states_equal = torch.equal(all_p33["states"], reference_baseline["states"])
        reconstruction_equal = torch.equal(
            all_p33["reconstruction"], reference_baseline["reconstruction"]
        )
        baseline_reencode_evidence[matrix] = {
            "states_equal": states_equal,
            "state_disagreement": int(
                torch.count_nonzero(all_p33["states"] != reference_baseline["states"])
            ),
            "reconstruction_equal": reconstruction_equal,
            "g_scale_single_map": float(reference_baseline["g_scale"]),
            "g_scale_prefix_batch": float(all_p33["g_scale"]),
        }

    fit_target = reference_output[fit_mask]
    fit_weights = gates[fit_mask].float().square().unsqueeze(1)
    confirmation_target = reference_output[confirmation_mask]
    confirmation_weights = (
        gates[confirmation_mask].float().square().unsqueeze(1)
    )
    fit_reference_energy = float(
        (fit_target.square() * fit_weights).sum(dtype=torch.float64)
    )
    confirmation_reference_energy = float(
        (confirmation_target.square() * confirmation_weights).sum(dtype=torch.float64)
    )
    scores: dict[str, dict[str, dict[str, float | int]]] = {}
    conditional_h2_evidence: dict[str, object] = {}
    conditional_down_baseline_evidence: dict[str, object] = {}
    down_maps = {
        f"n{count:04d}": rate_map("w2", order_down[:count].tolist())
        for count in counts_down
    }
    for upstream_count in counts_upstream:
        upstream_name = f"n{upstream_count:04d}"
        w1 = upstream_prefixes["w1"][upstream_name]["reconstruction"]
        w3 = upstream_prefixes["w3"][upstream_name]["reconstruction"]
        if not isinstance(w1, torch.Tensor) or not isinstance(w3, torch.Tensor):
            raise TypeError("upstream prefix reconstruction is not a tensor")
        decoded_middle = middle_decoder(fit_inputs, w1, w3)
        _, candidate_h2, h2_evidence = build_expert_hessians(
            fit_inputs,
            fit_gates,
            decoded_middle,
            global_h13=h13,
            global_h2=h2,
            device=device,
        )
        down_prefixes, _ = _quantize_maps(
            source[2],
            candidate_h2,
            contexts,
            matrix="w2",
            maps=down_maps,
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            permutation_override=permutation_override,
        )
        conditional_h2_evidence[upstream_name] = h2_evidence
        if upstream_count == 0:
            final_baseline = down_prefixes["n0000"]
            screen_baseline = baseline_candidates["w2"]
            conditional_down_baseline_evidence = {
                "states_equal": torch.equal(
                    final_baseline["states"], screen_baseline["states"]
                ),
                "state_disagreement": int(
                    torch.count_nonzero(
                        final_baseline["states"] != screen_baseline["states"]
                    )
                ),
                "reconstruction_equal": torch.equal(
                    final_baseline["reconstruction"],
                    screen_baseline["reconstruction"],
                ),
                "g_scale_screen_single_map": float(screen_baseline["g_scale"]),
                "g_scale_final_prefix_batch": float(final_baseline["g_scale"]),
            }
        for down_count in counts_down:
            down_name = f"n{down_count:04d}"
            w2 = down_prefixes[down_name]["reconstruction"]
            if not isinstance(w2, torch.Tensor):
                raise TypeError("down prefix reconstruction is not a tensor")
            fit_sse = _weighted_functional_sse(
                (w1, w3, w2),
                inputs=inputs[fit_mask],
                reference=fit_target,
                route_weights=fit_weights,
                execute_triplet=triplet_executor,
            )
            confirmation_sse = _weighted_functional_sse(
                (w1, w3, w2),
                inputs=inputs[confirmation_mask],
                reference=confirmation_target,
                route_weights=confirmation_weights,
                execute_triplet=triplet_executor,
            )
            key = f"u{upstream_count:04d}/d{down_count:04d}"
            scores[key] = {
                "fit": {
                    "sse": fit_sse,
                    "reference_energy": fit_reference_energy,
                    "nmse": fit_sse / fit_reference_energy,
                },
                "confirmation": {
                    "sse": confirmation_sse,
                    "reference_energy": confirmation_reference_energy,
                    "nmse": confirmation_sse / confirmation_reference_energy,
                },
            }
    batch_selected_name = min(
        scores, key=lambda name: float(scores[name]["fit"]["sse"])
    )
    baseline_name = "u0000/d0000"
    full_endpoint_name = f"u{evaluated_count:04d}/d{evaluated_count:04d}"
    batch_shortlist_names = sorted(
        scores,
        key=lambda name: float(scores[name]["fit"]["sse"]),
    )[:5]
    serial_candidate_names = {
        baseline_name,
        full_endpoint_name,
        batch_selected_name,
        *batch_shortlist_names,
    }
    # Small prefixes are the actual scientific question.  Include them even
    # when batch-dependent hard paths rank them outside the initial shortlist.
    fine_counts = (1, 2, 4, 8, 16, 32, 64)
    for count in fine_counts:
        if count in counts_upstream:
            serial_candidate_names.add(f"u{count:04d}/d0000")
        if count in counts_down:
            serial_candidate_names.add(f"u0000/d{count:04d}")
        if count in counts_upstream and count in counts_down:
            serial_candidate_names.add(f"u{count:04d}/d{count:04d}")

    def parse_candidate_name(name: str) -> tuple[int, int]:
        upstream_text, down_text = name.split("/")
        return (
            int(upstream_text.removeprefix("u")),
            int(down_text.removeprefix("d")),
        )

    serial_groups: dict[int, set[int]] = {}
    for name in serial_candidate_names:
        upstream_count, down_count = parse_candidate_name(name)
        serial_groups.setdefault(upstream_count, set()).add(down_count)

    serial_scores: dict[str, dict[str, dict[str, float | int]]] = {}
    serial_evidence: dict[str, dict[str, object]] = {}
    for upstream_count in sorted(serial_groups):
        upstream_name = f"n{upstream_count:04d}"
        serial_upstream: dict[str, dict[str, object]] = {}
        for matrix in ("w1", "w3"):
            group, _ = _quantize_maps(
                source[MATRICES.index(matrix)],
                h13,
                contexts,
                matrix=matrix,
                maps={
                    "candidate": rate_map(
                        matrix, order_upstream[:upstream_count].tolist()
                    )
                },
                layer=layer,
                expert=expert,
                device=device,
                quantizer_module=quantizer_module,
                ldlq_tf32=ldlq_tf32,
                prepared=prepared_upstream[matrix],
            )
            serial_upstream[matrix] = group["candidate"]
        w1 = serial_upstream["w1"]["reconstruction"]
        w3 = serial_upstream["w3"]["reconstruction"]
        if not isinstance(w1, torch.Tensor) or not isinstance(w3, torch.Tensor):
            raise TypeError("serial tile-prefix reconstruction is not a tensor")
        serial_middle = middle_decoder(fit_inputs, w1, w3)
        _, serial_h2, serial_h2_evidence = build_expert_hessians(
            fit_inputs,
            fit_gates,
            serial_middle,
            global_h13=h13,
            global_h2=h2,
            device=device,
        )
        upstream_batch_evidence = {
            matrix: {
                "states_equal_to_batch": torch.equal(
                    serial_upstream[matrix]["states"],
                    upstream_prefixes[matrix][upstream_name]["states"],
                ),
                "state_disagreement": int(
                    torch.count_nonzero(
                        serial_upstream[matrix]["states"]
                        != upstream_prefixes[matrix][upstream_name]["states"]
                    )
                ),
                "g_scale_serial": float(serial_upstream[matrix]["g_scale"]),
                "g_scale_batch": float(
                    upstream_prefixes[matrix][upstream_name]["g_scale"]
                ),
            }
            for matrix in ("w1", "w3")
        }
        for down_count in sorted(serial_groups[upstream_count]):
            candidate_name = f"u{upstream_count:04d}/d{down_count:04d}"
            down_group, _ = _quantize_maps(
                source[2],
                serial_h2,
                contexts,
                matrix="w2",
                maps={
                    "candidate": rate_map(
                        "w2", order_down[:down_count].tolist()
                    )
                },
                layer=layer,
                expert=expert,
                device=device,
                quantizer_module=quantizer_module,
                ldlq_tf32=ldlq_tf32,
                permutation_override=permutation_override,
            )
            serial_down = down_group["candidate"]
            w2 = serial_down["reconstruction"]
            if not isinstance(w2, torch.Tensor):
                raise TypeError("serial down tile-prefix reconstruction is not a tensor")
            fit_sse = _weighted_functional_sse(
                (w1, w3, w2),
                inputs=inputs[fit_mask],
                reference=fit_target,
                route_weights=fit_weights,
                execute_triplet=triplet_executor,
            )
            confirmation_sse = _weighted_functional_sse(
                (w1, w3, w2),
                inputs=inputs[confirmation_mask],
                reference=confirmation_target,
                route_weights=confirmation_weights,
                execute_triplet=triplet_executor,
            )
            serial_scores[candidate_name] = {
                "fit": {
                    "sse": fit_sse,
                    "reference_energy": fit_reference_energy,
                    "nmse": fit_sse / fit_reference_energy,
                },
                "confirmation": {
                    "sse": confirmation_sse,
                    "reference_energy": confirmation_reference_energy,
                    "nmse": confirmation_sse / confirmation_reference_energy,
                },
            }
            down_evidence: dict[str, object] = {
                "g_scale_serial": float(serial_down["g_scale"]),
            }
            if upstream_count == 0 and down_count == 0:
                baseline_down_candidate = baseline_candidates["w2"]
                down_evidence.update(
                    {
                        "states_equal_to_single_map_baseline": torch.equal(
                            serial_down["states"],
                            baseline_down_candidate["states"],
                        ),
                        "state_disagreement": int(
                            torch.count_nonzero(
                                serial_down["states"]
                                != baseline_down_candidate["states"]
                            )
                        ),
                        "g_scale_single_map_baseline": float(
                            baseline_down_candidate["g_scale"]
                        ),
                    }
                )
            serial_evidence[candidate_name] = {
                "upstream": upstream_batch_evidence,
                "down": down_evidence,
                "h2": serial_h2_evidence,
            }
        del serial_upstream, serial_middle, serial_h2
        torch.cuda.empty_cache()

    selected_name = min(
        serial_scores,
        key=lambda name: float(serial_scores[name]["fit"]["sse"]),
    )
    selected_confirmation = float(
        serial_scores[selected_name]["confirmation"]["sse"]
    )
    endpoint_confirmation = min(
        float(serial_scores[name]["confirmation"]["sse"])
        for name in (baseline_name, full_endpoint_name)
    )
    return {
        "representation": (
            dict(representation)
            if representation is not None
            else {
                "basis": "ordinary_expert_coordinates",
                "execution": "SiTU(w1*x,w3*x)_then_w2",
            }
        ),
        "baseline": (
            {
                "kind": "qsrt_3p08_record_schedule",
                "upstream_donor_records": base_upstream_donors,
                "down_donor_records": base_down_donors,
                "upstream_boundary_pair": [
                    base_upstream_donors,
                    21 - int(base_upstream_donors),
                ],
                "down_boundary_pair": [
                    base_down_donors,
                    21 - int(base_down_donors),
                ],
            }
            if boundary_search
            else {"kind": "uniform_k3_outer_pair"}
        ),
        "selection_units": {
            "upstream": "one shared gate/up boundary 16x16 tile pair",
            "down": "one independent down boundary 16x16 tile pair",
        },
        "coordinate_contract": {
            "basis": "after_the_one_common_expert_neuron_permutation",
            "coordinate": "orthogonal_tile_times_8_plus_subrecord_tile",
            "upstream_bitmap_shared_by": ["w1", "w3"],
            "down_bitmap_applies_to": "w2",
            "budget_invariant": "each_bit_selects_equal_size_P33_or_P24",
        },
        "pair_tiles_total_per_axis": pair_count,
        "pair_tiles_evaluated_per_axis": evaluated_count,
        "screen_h2": (
            "baseline_decoded_upstream_post_situ_adaptive_identity_shrinkage"
        ),
        "final_h2": (
            "decoded_upstream_prefix_post_situ_adaptive_identity_shrinkage"
        ),
        "h2_evidence": {
            "baseline": baseline_h2_evidence,
            "by_upstream_prefix": conditional_h2_evidence,
        },
        "baseline_reencode_evidence": {
            "upstream": baseline_reencode_evidence,
            "down": conditional_down_baseline_evidence,
            "interpretation": (
                "batched prefix paths are proposal evidence only; isolated "
                "single-candidate encodes determine the scientific result"
            ),
        },
        "screen_approximation": (
            "single_upstream_tile proposals hold the baseline decoded w2 fixed; "
            "every scored prefix rebuilds H2 and re-encodes w2"
            if proposal_deltas is None
            else "regularized-weight proxy orders tiles; every scored prefix "
            "uses complete expert-function loss, rebuilds H2, and re-encodes w2"
        ),
        "proposal_metric": proposal_metric,
        "screen_scale": "baseline_fixed",
        "final_scale": "independently_refit_per_prefix",
        "metadata": {
            "record_prefix_bits": 8,
            "boundary_bitmap_bits": 2 * pair_count,
            "total_bytes_unaligned": (8 + 2 * pair_count) // 8,
            "bits_per_all_expert_weight": (8 + 2 * pair_count)
            / (3 * 3072 * 3584),
        },
        "baseline_fit_sse": baseline_fit_sse,
        "proposal_delta": {
            "upstream_negative_count": int(torch.count_nonzero(deltas_upstream < 0)),
            "upstream_delta_min": float(deltas_upstream.min()),
            "upstream_delta_median": float(deltas_upstream.median()),
            "upstream_delta_max": float(deltas_upstream.max()),
            "down_negative_count": int(torch.count_nonzero(deltas_down < 0)),
            "down_delta_min": float(deltas_down.min()),
            "down_delta_median": float(deltas_down.median()),
            "down_delta_max": float(deltas_down.max()),
        },
        "upstream_order": [int(value) for value in order_upstream.tolist()],
        "down_order": [int(value) for value in order_down.tolist()],
        "prefix_scores": scores,
        "batch_selected_on_fit": batch_selected_name,
        "selected_on_fit": selected_name,
        "serial_validation": {
            "scores": serial_scores,
            "evidence": serial_evidence,
            "batch_top5_shortlist": batch_shortlist_names,
            "predeclared_fine_prefix_counts": list(fine_counts),
            "selection_source": (
                "isolated_serial_fit_over_batch_top5_plus_predeclared_controls"
            ),
            "validation_source": "independent_single_candidate_reencode",
        },
        "confirmation_relative_to_p33": _relative(
            selected_confirmation,
            float(serial_scores[baseline_name]["confirmation"]["sse"]),
        ),
        "confirmation_relative_to_better_endpoint": _relative(
            selected_confirmation, endpoint_confirmation
        ),
    }


def _relative(candidate: float, baseline: float) -> float:
    return candidate / baseline - 1.0


def _run_expert(
    *,
    layer: int,
    expert: int,
    samples: LayerSamples,
    partition,
    external_samples: LayerSamples | None,
    external_requests: Mapping[int, str] | None,
    global_h13: torch.Tensor,
    global_h2: torch.Tensor,
    store: OfficialMXFP4Store,
    device: torch.device,
    quantizer_module,
    ldlq_tf32: bool,
    include_tile_triplet_oracle: bool,
    switched_luts: Mapping[str, Mapping[int, torch.Tensor]],
    k2_codebook_menu: bool,
    coupled_hadamard_k2: bool,
    h2_viterbi_refine: bool,
    h2_viterbi_refine_sweeps: int,
    coupled_hadamard_block_size: int,
    coupled_hadamard_preactivation_block_size: int,
    coupled_hadamard_postactivation_block_size: int,
    coupled_hadamard_pre_permutation: str,
    coupled_residual_draw: int,
    coupled_intermediate_draws: Mapping[int, int],
    functional_tile_search: bool,
    p13_search: bool,
    qsrt_308_search: bool,
    qsrt_308_tile_fractions: bool,
    qsrt_308_boundary_tile_search: bool,
    qsrt_308_boundary_proxy_search: bool,
    qsrt_308_max_donors: int,
    qsrt_308_scale_closure: bool,
    tile_search_chunk: int,
    tile_search_limit: int,
    permutation_policy: PermutationChoice,
) -> dict[str, object]:
    all_rows = select_expert_rows(samples, expert, partition.all)
    fit_mask_cpu = _request_mask(all_rows.request_steps, partition.fit)
    confirmation_mask_cpu = _request_mask(
        all_rows.request_steps, partition.confirmation
    )
    fit_documents = int(torch.unique(all_rows.request_steps[fit_mask_cpu]).numel())
    confirmation_documents = int(
        torch.unique(all_rows.request_steps[confirmation_mask_cpu]).numel()
    )
    if (
        not h2_viterbi_refine
        and (fit_documents < 6 or confirmation_documents < 4)
    ):
        return {
            "skipped": True,
            "reason": "insufficient document support",
            "fit_documents": fit_documents,
            "confirmation_documents": confirmation_documents,
        }
    inputs = all_rows.inputs.float().to(device)
    gates = all_rows.gates.float().to(device)
    request_steps = all_rows.request_steps.to(device)
    fit_mask = fit_mask_cpu.to(device)
    confirmation_mask = confirmation_mask_cpu.to(device)
    source = tuple(
        store.load_matrix(layer, expert, matrix, device=device).float().contiguous()
        for matrix in MATRICES
    )
    external_inputs: torch.Tensor | None = None
    external_gates: torch.Tensor | None = None
    external_request_steps: torch.Tensor | None = None
    external_support: dict[str, int] | None = None
    if external_samples is not None:
        if external_requests is None:
            raise ValueError("external samples require external request metadata")
        external_rows = select_expert_rows(
            external_samples, expert, external_requests
        )
        if external_rows.rows:
            external_inputs = external_rows.inputs.float().to(device)
            external_gates = external_rows.gates.float().to(device)
            external_request_steps = external_rows.request_steps.to(device)
            external_support = {
                "rows": external_rows.rows,
                "documents": external_rows.documents,
            }
    with torch.inference_mode():
        source_middle = situ(F.linear(inputs, source[0]), F.linear(inputs, source[1]))
        context_policy: PermutationPolicy = (
            "h2_reverse"
            if permutation_policy
            in (
                "h2_exact",
                "functional_exact",
                "h2_tile_balanced",
                "functional_tile_balanced",
                "h2_shape_clustered",
                "h2_global_shape_clustered",
                "functional_global_shape_clustered",
                "h2_priority_shape_clustered",
                "functional_priority_shape_clustered",
                "h2_rate_response_clustered",
                "h2_p24_band_aligned",
                "h2_top2_band_aligned",
            )
            else permutation_policy
        )
        contexts, block_scores = permutation_policy_contexts(
            source_middle[fit_mask], gates[fit_mask], policy=context_policy
        )
        h13, h2, covariance = build_expert_hessians(
            inputs[fit_mask],
            gates[fit_mask],
            source_middle[fit_mask],
            global_h13=global_h13,
            global_h2=global_h2,
            device=device,
        )
        codec_features = _codec_shape_group_features(source)
        if permutation_policy in (
            "h2_exact",
            "h2_tile_balanced",
            "h2_shape_clustered",
            "h2_global_shape_clustered",
            "h2_priority_shape_clustered",
            "h2_rate_response_clustered",
            "h2_p24_band_aligned",
            "h2_top2_band_aligned",
        ):
            permutation_scores = block_scores
        elif permutation_policy in (
            "functional_exact",
            "functional_tile_balanced",
            "functional_global_shape_clustered",
            "functional_priority_shape_clustered",
        ):
            permutation_scores = _functional_neuron_group_scores(
                source,
                inputs=inputs[fit_mask],
                gates=gates[fit_mask],
            )
        else:
            permutation_scores = None
        rate_response_evidence: dict[str, object] | None = None
        tile_funding_alignment_evidence: dict[str, object] | None = None
        if permutation_policy in (
            "h2_rate_response_clustered",
            "h2_p24_band_aligned",
            "h2_top2_band_aligned",
        ):
            initial_permutation = plan_qsrt_matrix(
                contexts,
                RATE_TRANSFER_MODES[0],
                matrix="w1",
                layout="importance_ordered",
            ).encoder_permutation.to(device)
            preliminary_uniform: dict[
                str, dict[int, dict[str, object]]
            ] = {}
            preliminary_targets: dict[str, torch.Tensor] = {}
            for matrix, source_matrix in zip(MATRICES, source, strict=True):
                weight_shape = (
                    (3584, 3072) if matrix in ("w1", "w3") else (3072, 3584)
                )
                maps = {
                    f"k{bits}": _uniform_tile_map(weight_shape, bits)
                    for bits in (3, 2, 4)
                }
                candidates, encoder_weight = _quantize_maps(
                    source_matrix,
                    h2 if matrix == "w2" else h13,
                    contexts,
                    matrix=matrix,
                    maps=maps,
                    layer=layer,
                    expert=expert,
                    device=device,
                    quantizer_module=quantizer_module,
                    ldlq_tf32=ldlq_tf32,
                    permutation_override=initial_permutation,
                )
                by_bits = {bits: candidates[f"k{bits}"] for bits in (2, 3, 4)}
                preliminary_uniform[matrix] = by_bits
                preliminary_targets[matrix] = _target_regularized_weight(
                    encoder_weight,
                    by_bits[3]["suh"],
                    by_bits[3]["svh"],
                )
            if permutation_policy == "h2_rate_response_clustered":
                response_features = _rate_response_group_features(
                    preliminary_targets,
                    preliminary_uniform,
                    permutation=initial_permutation,
                )
                response_order = _record_clustered_group_order(
                    block_scores, response_features
                )
                permutation_override = expand_group_order(response_order).to(device)
                rate_response_evidence = {
                    "teacher_permutation_sha256": _permutation_sha256(
                        initial_permutation
                    ),
                    "feature_shape": list(response_features.shape),
                    "feature_blocks": [
                        "upstream_k3_error",
                        "upstream_k2_over_k3",
                        "upstream_k3_over_k4",
                        "down_k3_error",
                        "down_k2_over_k3",
                        "down_k3_over_k4",
                    ],
                    "orthogonal_bands_per_block": 224,
                    "clustering_scope": "within_fixed_h2_importance_record",
                    "selection_partition": "fit_only",
                }
            else:
                functional_proxy = _functional_proxy_tile_errors(
                    source,
                    preliminary_uniform,
                    h13=h13,
                    h2=h2,
                    inputs=inputs[fit_mask],
                    gates=gates[fit_mask],
                    permutation=initial_permutation,
                )
                if permutation_policy == "h2_p24_band_aligned":
                    permutation_override, tile_funding_alignment_evidence = (
                        _p24_band_aligned_permutation(
                            initial_permutation, functional_proxy
                        )
                    )
                else:
                    permutation_override, tile_funding_alignment_evidence = (
                        _top2_band_aligned_permutation(
                            initial_permutation, functional_proxy
                        )
                    )
                tile_funding_alignment_evidence.update(
                    {
                        "teacher_permutation_sha256": _permutation_sha256(
                            initial_permutation
                        ),
                        "selection_partition": "fit_only",
                        "proposal_metric": (
                            "coupled_upstream_plus_down_functional_tile_proxy"
                        ),
                    }
                )
        elif permutation_scores is None:
            permutation_override = None
        elif permutation_policy.endswith("_global_shape_clustered"):
            permutation_override = expand_group_order(
                _global_shape_clustered_group_order(
                    permutation_scores, codec_features
                )
            ).to(device)
        elif permutation_policy.endswith("_priority_shape_clustered"):
            permutation_override = expand_group_order(
                _priority_shape_clustered_group_order(
                    permutation_scores, codec_features
                )
            ).to(device)
        elif permutation_policy == "h2_shape_clustered":
            permutation_override = expand_group_order(
                _shape_clustered_group_order(
                    permutation_scores, codec_features
                )
            ).to(device)
        else:
            permutation_override = expand_group_order(
                _tile_balanced_group_order(permutation_scores)
                if permutation_policy.endswith("_tile_balanced")
                else torch.argsort(permutation_scores, stable=True)
            ).to(device)
    contexts = contexts.to(device=device, dtype=torch.long)
    selected_permutation = (
        permutation_override
        if permutation_override is not None
        else plan_qsrt_matrix(
            contexts,
            RATE_TRANSFER_MODES[0],
            matrix="w1",
            layout="importance_ordered",
        ).encoder_permutation
    )
    permutation_sha256 = _permutation_sha256(selected_permutation)
    geometry_scores = (
        block_scores if permutation_scores is None else permutation_scores
    )
    permutation_geometry = _permutation_tile_geometry(
        selected_permutation.to(device=geometry_scores.device),
        geometry_scores,
        codec_features=codec_features.to(device=geometry_scores.device),
    )

    if h2_viterbi_refine:
        return {
            "skipped": False,
            "support": {
                "rows": int(inputs.shape[0]),
                "documents": int(torch.unique(request_steps).numel()),
                "fit_rows_descriptive": int(fit_mask.sum()),
                "confirmation_rows_descriptive": int(confirmation_mask.sum()),
            },
            "permutation_policy": permutation_policy,
            "permutation_sha256": permutation_sha256,
            "h2_viterbi_refinement": _h2_viterbi_refinement(
                source,
                global_h13=global_h13,
                global_h2=global_h2,
                contexts=contexts,
                layer=layer,
                expert=expert,
                device=device,
                quantizer_module=quantizer_module,
                ldlq_tf32=ldlq_tf32,
                inputs=inputs,
                gates=gates,
                request_steps=request_steps,
                baseline_permutation=selected_permutation,
                block_size=coupled_hadamard_block_size,
                preactivation_block_size=(
                    coupled_hadamard_preactivation_block_size
                ),
                postactivation_block_size=(
                    coupled_hadamard_postactivation_block_size
                ),
                pre_permutation=coupled_hadamard_pre_permutation,
                residual_rotation_draw=coupled_residual_draw,
                intermediate_rotation_draw=coupled_intermediate_draws.get(
                    expert, 0
                ),
                refine_sweeps=h2_viterbi_refine_sweeps,
            ),
        }

    if (
        coupled_hadamard_k2
        and not (p13_search or qsrt_308_search or k2_codebook_menu)
    ):
        return {
            "skipped": False,
            "support": {
                "fit_rows": int(fit_mask.sum()),
                "fit_documents": fit_documents,
                "confirmation_rows": int(confirmation_mask.sum()),
                "confirmation_documents": confirmation_documents,
            },
            "covariance": covariance,
            "permutation_policy": permutation_policy,
            "permutation_sha256": permutation_sha256,
            "permutation_tile_geometry": permutation_geometry,
            "coupled_hadamard_k2": _coupled_hadamard_k2(
                source,
                h13=h13,
                source_h2=h2,
                contexts=contexts,
                layer=layer,
                expert=expert,
                device=device,
                quantizer_module=quantizer_module,
                ldlq_tf32=ldlq_tf32,
                inputs=inputs,
                gates=gates,
                request_steps=request_steps,
                fit_mask=fit_mask,
                confirmation_mask=confirmation_mask,
                fit_requests=partition.fit,
                confirmation_requests=partition.confirmation,
                baseline_permutation=selected_permutation,
                block_size=coupled_hadamard_block_size,
                preactivation_block_size=(
                    coupled_hadamard_preactivation_block_size
                ),
                postactivation_block_size=(
                    coupled_hadamard_postactivation_block_size
                ),
                pre_permutation=coupled_hadamard_pre_permutation,
                residual_rotation_draw=coupled_residual_draw,
                intermediate_rotation_draw=coupled_intermediate_draws.get(
                    expert, 0
                ),
            ),
        }

    if k2_codebook_menu:
        menu_source = source
        menu_h13 = h13
        menu_h2 = h2
        menu_inputs = inputs
        menu_permutation = selected_permutation
        menu_reference_output = None
        menu_decode_middle = None
        menu_execute_triplet = None
        menu_representation = None
        menu_external_inputs = external_inputs
        external_reference_output = (
            None
            if external_inputs is None
            else _execute_standard_triplet(external_inputs, source)
        )
        if coupled_hadamard_k2:
            menu_basis = _prepare_coupled_search_basis(
                source,
                h13=h13,
                h2=h2,
                inputs=inputs,
                selected_permutation=selected_permutation,
                block_size=coupled_hadamard_block_size,
                preactivation_block_size=(
                    coupled_hadamard_preactivation_block_size
                ),
                postactivation_block_size=(
                    coupled_hadamard_postactivation_block_size
                ),
                pre_permutation=coupled_hadamard_pre_permutation,
                residual_rotation_draw=coupled_residual_draw,
                intermediate_rotation_draw=(
                    coupled_intermediate_draws.get(expert, 0)
                ),
            )
            menu_source = menu_basis.source
            menu_h13 = menu_basis.h13
            menu_h2 = menu_basis.h2
            menu_inputs = menu_basis.inputs
            menu_permutation = menu_basis.permutation
            menu_reference_output = menu_basis.reference_output
            menu_decode_middle = menu_basis.decode_middle
            menu_execute_triplet = menu_basis.execute_triplet
            menu_representation = menu_basis.evidence
            if external_inputs is not None:
                menu_external_inputs = menu_basis.transform_inputs(
                    external_inputs
                )
        return {
            "skipped": False,
            "support": {
                "fit_rows": int(fit_mask.sum()),
                "fit_documents": fit_documents,
                "confirmation_rows": int(confirmation_mask.sum()),
                "confirmation_documents": confirmation_documents,
                "external": external_support,
            },
            "covariance": covariance,
            "permutation_policy": permutation_policy,
            "permutation_sha256": permutation_sha256,
            "permutation_tile_geometry": permutation_geometry,
            "k2_codebook_menu": _k2_codebook_menu(
                menu_source,
                switched_luts,
                h13=menu_h13,
                source_h2=menu_h2,
                contexts=contexts,
                layer=layer,
                expert=expert,
                device=device,
                quantizer_module=quantizer_module,
                ldlq_tf32=ldlq_tf32,
                inputs=menu_inputs,
                gates=gates,
                request_steps=request_steps,
                fit_mask=fit_mask,
                confirmation_mask=confirmation_mask,
                fit_requests=partition.fit,
                confirmation_requests=partition.confirmation,
                permutation_override=menu_permutation,
                decode_middle=menu_decode_middle,
                execute_triplet=menu_execute_triplet,
                reference_output=menu_reference_output,
                representation=menu_representation,
                external_inputs=menu_external_inputs,
                external_gates=external_gates,
                external_request_steps=external_request_steps,
                external_requests=external_requests,
                external_reference_output=external_reference_output,
            ),
        }

    coupled_search_basis: CoupledSearchBasis | None = None
    research_source = source
    research_h13 = h13
    research_h2 = h2
    research_inputs = inputs
    research_permutation = selected_permutation
    research_reference_output: torch.Tensor | None = None
    research_decode_middle: MiddleDecoder | None = None
    research_execute_triplet: TripletExecutor | None = None
    if coupled_hadamard_k2:
        coupled_search_basis = _prepare_coupled_search_basis(
            source,
            h13=h13,
            h2=h2,
            inputs=inputs,
            selected_permutation=selected_permutation,
            block_size=coupled_hadamard_block_size,
            preactivation_block_size=(
                coupled_hadamard_preactivation_block_size
            ),
            postactivation_block_size=(
                coupled_hadamard_postactivation_block_size
            ),
            pre_permutation=coupled_hadamard_pre_permutation,
            residual_rotation_draw=coupled_residual_draw,
            intermediate_rotation_draw=(
                coupled_intermediate_draws.get(expert, 0)
            ),
        )
        research_source = coupled_search_basis.source
        research_h13 = coupled_search_basis.h13
        research_h2 = coupled_search_basis.h2
        research_inputs = coupled_search_basis.inputs
        research_permutation = coupled_search_basis.permutation
        research_reference_output = coupled_search_basis.reference_output
        research_decode_middle = coupled_search_basis.decode_middle
        research_execute_triplet = coupled_search_basis.execute_triplet

    if p13_search:
        shared_search_args = {
            "h13": research_h13,
            "h2": research_h2,
            "contexts": contexts,
            "layer": layer,
            "expert": expert,
            "device": device,
            "quantizer_module": quantizer_module,
            "ldlq_tf32": ldlq_tf32,
            "inputs": research_inputs,
            "gates": gates,
            "fit_mask": fit_mask,
            "confirmation_mask": confirmation_mask,
            "permutation_override": research_permutation,
            "decode_middle": research_decode_middle,
            "execute_triplet": research_execute_triplet,
            "reference_output": research_reference_output,
            "representation": (
                coupled_search_basis.evidence
                if coupled_search_basis is not None
                else None
            ),
        }
        return {
            "skipped": False,
            "support": {
                "fit_rows": int(fit_mask.sum()),
                "fit_documents": fit_documents,
                "confirmation_rows": int(confirmation_mask.sum()),
                "confirmation_documents": confirmation_documents,
            },
            "covariance": covariance,
            "permutation_policy": permutation_policy,
            "permutation_sha256": permutation_sha256,
            "permutation_tile_geometry": permutation_geometry,
            "p13_record_rates": _p13_record_search(
                research_source,
                **shared_search_args,
            ),
        }

    uniform: dict[str, dict[int, dict[str, object]]] = {}
    targets: dict[str, torch.Tensor] = {}
    errors: dict[str, dict[int, torch.Tensor]] = {}
    for matrix, source_matrix in zip(MATRICES, research_source, strict=True):
        weight_shape = (3584, 3072) if matrix in ("w1", "w3") else (3072, 3584)
        uniform_maps = {
            f"k{bits}": _uniform_tile_map(weight_shape, bits)
            for bits in (3, 2, 4)
        }
        candidates, encoder_weight = _quantize_maps(
            source_matrix,
            research_h2 if matrix == "w2" else research_h13,
            contexts,
            matrix=matrix,
            maps=uniform_maps,
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            permutation_override=research_permutation,
        )
        by_bits = {bits: candidates[f"k{bits}"] for bits in (2, 3, 4)}
        uniform[matrix] = by_bits
        baseline = by_bits[3]
        targets[matrix] = _target_regularized_weight(
            encoder_weight,
            baseline["suh"],
            baseline["svh"],
        )
        errors[matrix] = _tile_errors(targets[matrix], by_bits)

    allocation_errors: Mapping[str, Mapping[int, torch.Tensor]] = errors
    if (
        qsrt_308_tile_fractions or qsrt_308_boundary_proxy_search
    ) and coupled_search_basis is None:
        functional_proxy = _functional_proxy_tile_errors(
            source,
            uniform,
            h13=h13,
            h2=h2,
            inputs=inputs[fit_mask],
            gates=gates[fit_mask],
            permutation=selected_permutation,
        )
        allocation_errors = {
            "w1": functional_proxy["w13"],
            "w3": {
                bits: torch.zeros_like(value)
                for bits, value in functional_proxy["w13"].items()
            },
            "w2": functional_proxy["w2"],
        }

    if qsrt_308_search:
        qsrt_308 = _qsrt_308_search(
            research_source,
            uniform,
            allocation_errors,
            h13=research_h13,
            h2=research_h2,
            contexts=contexts,
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            inputs=research_inputs,
            gates=gates,
            fit_mask=fit_mask,
            confirmation_mask=confirmation_mask,
            permutation_override=research_permutation,
            include_tile_fractions=qsrt_308_tile_fractions,
            max_donors=qsrt_308_max_donors,
            scale_closure=qsrt_308_scale_closure,
            decode_middle=research_decode_middle,
            execute_triplet=research_execute_triplet,
            reference_output=research_reference_output,
            representation=(
                coupled_search_basis.evidence
                if coupled_search_basis is not None
                else None
            ),
        )
        payload = {
            "skipped": False,
            "support": {
                "fit_rows": int(fit_mask.sum()),
                "fit_documents": fit_documents,
                "confirmation_rows": int(confirmation_mask.sum()),
                "confirmation_documents": confirmation_documents,
            },
            "covariance": covariance,
            "permutation_policy": permutation_policy,
            "permutation_sha256": permutation_sha256,
            "permutation_tile_geometry": permutation_geometry,
            "block_score_range": [
                float(block_scores.min()),
                float(block_scores.median()),
                float(block_scores.max()),
            ],
            "permutation_score_range": [
                float(geometry_scores.min()),
                float(geometry_scores.median()),
                float(geometry_scores.max()),
            ],
            "rate_response_clustering": rate_response_evidence,
            "tile_funding_alignment": tile_funding_alignment_evidence,
            "tile_funding_proposal_metric": (
                "regularized_weight_tile_sse_in_coupled_basis"
                if coupled_search_basis is not None
                else (
                    "fit_only_complete_expert_linearized_functional_proxy"
                    if qsrt_308_tile_fractions
                    else "regularized_weight_tile_sse"
                )
            ),
            "qsrt_308": qsrt_308,
        }
        if qsrt_308_boundary_tile_search or qsrt_308_boundary_proxy_search:
            upstream_text, down_text = str(qsrt_308["selected_on_fit"]).split("/")
            upstream_donors = int(upstream_text.removeprefix("un"))
            down_donors = int(down_text.removeprefix("dn"))
            if upstream_donors == 11 or down_donors == 11:
                payload["qsrt_308_boundary_tile_p24"] = {
                    "skipped": True,
                    "reason": (
                        "fit-selected record schedule has no remaining boundary pair"
                    ),
                    "upstream_donor_records": upstream_donors,
                    "down_donor_records": down_donors,
                }
            else:
                payload["qsrt_308_boundary_tile_p24"] = (
                    _functional_tile_pair_search_independent(
                        research_source,
                        uniform,
                        h13=research_h13,
                        h2=research_h2,
                        contexts=contexts,
                        layer=layer,
                        expert=expert,
                        device=device,
                        quantizer_module=quantizer_module,
                        ldlq_tf32=ldlq_tf32,
                        inputs=research_inputs,
                        gates=gates,
                        fit_mask=fit_mask,
                        confirmation_mask=confirmation_mask,
                        chunk_size=tile_search_chunk,
                        pair_limit=tile_search_limit,
                        permutation_override=research_permutation,
                        base_upstream_donors=upstream_donors,
                        base_down_donors=down_donors,
                        proposal_deltas=(
                            (
                                _qsrt_308_boundary_proxy_deltas(
                                    (
                                        allocation_errors["w1"],
                                        allocation_errors["w3"],
                                    ),
                                    upstream_donors,
                                    rate_axis="n",
                                ),
                                _qsrt_308_boundary_proxy_deltas(
                                    (allocation_errors["w2"],),
                                    down_donors,
                                    rate_axis="k",
                                ),
                            )
                            if qsrt_308_boundary_proxy_search
                            else None
                        ),
                        decode_middle=research_decode_middle,
                        execute_triplet=research_execute_triplet,
                        reference_output=research_reference_output,
                        representation=(
                            coupled_search_basis.evidence
                            if coupled_search_basis is not None
                            else None
                        ),
                    )
                )
        return payload

    if functional_tile_search:
        return {
            "skipped": False,
            "support": {
                "fit_rows": int(fit_mask.sum()),
                "fit_documents": fit_documents,
                "confirmation_rows": int(confirmation_mask.sum()),
                "confirmation_documents": confirmation_documents,
            },
            "covariance": covariance,
            "permutation_policy": permutation_policy,
            "permutation_sha256": permutation_sha256,
            "permutation_tile_geometry": permutation_geometry,
            "block_score_range": [
                float(block_scores.min()),
                float(block_scores.median()),
                float(block_scores.max()),
            ],
            "permutation_score_range": [
                float(geometry_scores.min()),
                float(geometry_scores.median()),
                float(geometry_scores.max()),
            ],
            "rate_response_clustering": rate_response_evidence,
            "tile_funding_alignment": tile_funding_alignment_evidence,
            "functional_tile_p24": _functional_tile_pair_search_independent(
                source,
                uniform,
                h13=h13,
                h2=h2,
                contexts=contexts,
                layer=layer,
                expert=expert,
                device=device,
                quantizer_module=quantizer_module,
                ldlq_tf32=ldlq_tf32,
                inputs=inputs,
                gates=gates,
                fit_mask=fit_mask,
                confirmation_mask=confirmation_mask,
                chunk_size=tile_search_chunk,
                pair_limit=tile_search_limit,
                permutation_override=selected_permutation,
            ),
        }

    upstream_maps, upstream_map_evidence = _fractional_pair_maps(
        (errors["w1"], errors["w3"]), rate_axis="n"
    )
    down_maps, down_map_evidence = _fractional_pair_maps(
        (errors["w2"],), rate_axis="k"
    )
    mapped: dict[str, dict[str, dict[str, object]]] = {}
    for matrix, maps in (("w1", upstream_maps), ("w3", upstream_maps), ("w2", down_maps)):
        baseline_scale = uniform[matrix][3]["g_scale"]
        mapped[matrix], _ = _quantize_maps(
            source[MATRICES.index(matrix)],
            h2 if matrix == "w2" else h13,
            contexts,
            matrix=matrix,
            maps=maps,
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            g_scale_override=float(baseline_scale),
            permutation_override=selected_permutation,
        )

    fractional_scores = {}
    for name in upstream_maps:
        reconstruction = (
            mapped["w1"][name]["reconstruction"],
            mapped["w3"][name]["reconstruction"],
            mapped["w2"][name]["reconstruction"],
        )
        fractional_scores[name] = _score_candidate(
            source,
            reconstruction,
            inputs=inputs,
            gates=gates,
            request_steps=request_steps,
            fit_mask=fit_mask,
            confirmation_mask=confirmation_mask,
            fit_requests=partition.fit,
            confirmation_requests=partition.confirmation,
        )
    selected_fraction = min(
        fractional_scores,
        key=lambda name: float(fractional_scores[name]["fit"]["sse"]),
    )

    law_reconstructions: dict[str, dict[str, torch.Tensor]] = {"normal": {}}
    for matrix in MATRICES:
        value = mapped[matrix]["p100"]["reconstruction"]
        if not isinstance(value, torch.Tensor):
            raise TypeError("fractional candidate reconstruction is not a tensor")
        law_reconstructions["normal"][matrix] = value
    for law, luts_by_bits in switched_luts.items():
        if law == "normal":
            continue
        law_reconstructions[law] = {}
        for matrix, rate_map in (
            ("w1", upstream_maps["p100"]),
            ("w3", upstream_maps["p100"]),
            ("w2", down_maps["p100"]),
        ):
            candidates, _ = _quantize_maps(
                source[MATRICES.index(matrix)],
                h2 if matrix == "w2" else h13,
                contexts,
                matrix=matrix,
                maps={"p100": rate_map},
                layer=layer,
                expert=expert,
                device=device,
                quantizer_module=quantizer_module,
                ldlq_tf32=ldlq_tf32,
                g_scale_override=float(uniform[matrix][3]["g_scale"]),
                luts_by_bits=luts_by_bits,
                permutation_override=selected_permutation,
            )
            reconstruction = candidates["p100"]["reconstruction"]
            if not isinstance(reconstruction, torch.Tensor):
                raise TypeError("switched-law reconstruction is not a tensor")
            law_reconstructions[law][matrix] = reconstruction

    switched_scores = {}
    for upstream_law in OUTER_SLOPES:
        for down_law in OUTER_SLOPES:
            name = f"{upstream_law}/{down_law}"
            switched_scores[name] = _score_candidate(
                source,
                (
                    law_reconstructions[upstream_law]["w1"],
                    law_reconstructions[upstream_law]["w3"],
                    law_reconstructions[down_law]["w2"],
                ),
                inputs=inputs,
                gates=gates,
                request_steps=request_steps,
                fit_mask=fit_mask,
                confirmation_mask=confirmation_mask,
                fit_requests=partition.fit,
                confirmation_requests=partition.confirmation,
            )
    selected_laws = min(
        switched_scores,
        key=lambda name: float(switched_scores[name]["fit"]["sse"]),
    )

    _, proxy_record_triplet_counts = _record_triplet_map(errors)
    # Every tile map and its complete-expert score must refer to the same
    # frozen neuron basis used to build the K2/K3/K4 error surfaces.  The
    # channel permutation is not part of the tile allocator's search space.
    permutation = selected_permutation
    record_triplet_maps, record_triplet_counts, record_triplet_screen = (
        _functional_record_triplet_map(
            source,
            uniform,
            permutation,
            inputs=inputs[fit_mask],
            gates=gates[fit_mask],
        )
    )
    record_triplet_score = _score_candidate(
        source,
        _encode_triplet_map(
            record_triplet_maps,
            source=source,
            uniform=uniform,
            h13=h13,
            h2=h2,
            contexts=contexts,
            layer=layer,
            expert=expert,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=ldlq_tf32,
            permutation_override=selected_permutation,
        ),
        inputs=inputs,
        gates=gates,
        request_steps=request_steps,
        fit_mask=fit_mask,
        confirmation_mask=confirmation_mask,
        fit_requests=partition.fit,
        confirmation_requests=partition.confirmation,
    )
    tile_triplet_result = None
    if include_tile_triplet_oracle:
        tile_triplet_maps, tile_triplet_counts = _tile_triplet_map(errors)
        tile_triplet_score = _score_candidate(
            source,
            _encode_triplet_map(
                tile_triplet_maps,
                source=source,
                uniform=uniform,
                h13=h13,
                h2=h2,
                contexts=contexts,
                layer=layer,
                expert=expert,
                device=device,
                quantizer_module=quantizer_module,
                ldlq_tf32=ldlq_tf32,
                permutation_override=selected_permutation,
            ),
            inputs=inputs,
            gates=gates,
            request_steps=request_steps,
            fit_mask=fit_mask,
            confirmation_mask=confirmation_mask,
            fit_requests=partition.fit,
            confirmation_requests=partition.confirmation,
        )
        tile_triplet_result = {
            "tile_mode_counts": tile_triplet_counts,
            "metadata_bits": 3 * 43_008,
            "score": tile_triplet_score,
            "confirmation_relative_to_p33": _relative(
                float(tile_triplet_score["confirmation"]["sse"]),
                float(fractional_scores["p000"]["confirmation"]["sse"]),
            ),
        }
    baseline = fractional_scores["p000"]
    full_p24 = fractional_scores["p100"]
    selected = fractional_scores[selected_fraction]
    return {
        "skipped": False,
        "support": {
            "fit_rows": int(fit_mask.sum()),
            "fit_documents": fit_documents,
            "confirmation_rows": int(confirmation_mask.sum()),
            "confirmation_documents": confirmation_documents,
        },
        "covariance": covariance,
        "block_score_range": [
            float(block_scores.min()),
            float(block_scores.median()),
            float(block_scores.max()),
        ],
        "fractional_p24": {
            "upstream_map": upstream_map_evidence,
            "down_map": down_map_evidence,
            "selected_on_fit": selected_fraction,
            "scores": fractional_scores,
            "confirmation_relative_to_p33": _relative(
                float(selected["confirmation"]["sse"]),
                float(baseline["confirmation"]["sse"]),
            ),
            "confirmation_relative_to_better_endpoint": _relative(
                float(selected["confirmation"]["sse"]),
                min(
                    float(baseline["confirmation"]["sse"]),
                    float(full_p24["confirmation"]["sse"]),
                ),
            ),
        },
        "switched_k2_laws_full_p24": {
            "selection_granularity": "one shared w1/w3 law and one w2 law per expert",
            "selected_on_fit": selected_laws,
            "scores": switched_scores,
            "confirmation_relative_to_normal": _relative(
                float(switched_scores[selected_laws]["confirmation"]["sse"]),
                float(switched_scores["normal/normal"]["confirmation"]["sse"]),
            ),
            "q33_confirmation_relative_to_normal": _relative(
                float(switched_scores["q33/q33"]["confirmation"]["sse"]),
                float(switched_scores["normal/normal"]["confirmation"]["sse"]),
            ),
        },
        "projection_triples_record": {
            "record_mode_counts": record_triplet_counts,
            "proxy_record_mode_counts": proxy_record_triplet_counts,
            "screen": record_triplet_screen,
            "metadata_bits": 3 * 24,
            "score": record_triplet_score,
            "confirmation_relative_to_p33": _relative(
                float(record_triplet_score["confirmation"]["sse"]),
                float(baseline["confirmation"]["sse"]),
            ),
        },
        "projection_triples_tile_oracle": tile_triplet_result,
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("K2 allocation exploration requires CUDA")
    torch.cuda.set_device(device)
    report = _read_json(args.training_report)
    if Path(str(report.get("capture_dir", ""))).resolve() != args.capture.resolve():
        raise ValueError("training report does not describe the requested capture")
    requests = request_documents(report)
    partition = partition_requests(requests)
    samples = index_cached_layer_samples(args.sample_cache, [args.layer - 1]).pop(
        args.layer - 1
    )
    external_samples: LayerSamples | None = None
    external_requests: dict[int, str] | None = None
    external_contract: dict[str, object] | None = None
    if args.external_sample_cache is not None:
        assert args.external_report is not None
        external_report = _read_json(args.external_report)
        external_requests = request_documents(
            external_report, deduplicate=True
        )
        overlap = set(requests.values()) & set(external_requests.values())
        if overlap:
            raise ValueError(
                f"external corpus overlaps training by {len(overlap)} documents"
            )
        external_manifest_path = args.external_sample_cache / "manifest.json"
        external_manifest = _read_json(external_manifest_path)
        external_capture = Path(
            str(external_report.get("capture_dir", ""))
        ).resolve()
        if (
            Path(str(external_manifest.get("source_capture", ""))).resolve()
            != external_capture
        ):
            raise ValueError(
                "external sample cache does not describe the external report"
            )
        external_samples = index_cached_layer_samples(
            args.external_sample_cache, [args.layer - 1]
        ).pop(args.layer - 1)
        external_contract = {
            "sample_cache": str(args.external_sample_cache.resolve()),
            "report": str(args.external_report.resolve()),
            "capture": str(external_capture),
            "documents": len(external_requests),
            "duplicate_document_epochs": int(
                external_report.get("completed_requests", len(external_requests))
            )
            - len(external_requests),
            "document_overlap_with_training": 0,
            "selection_role": "untouched_frozen_candidate_evaluation_only",
        }
    global_h13, global_h2 = load_layer_hessians(args.hessians, args.layer)
    store_kwargs: dict[str, object] = {"repo_dir": args.official_repo_dir}
    if args.official_revision is not None:
        store_kwargs["revision"] = args.official_revision
    store = OfficialMXFP4Store(**store_kwargs)
    quantizer_module = load_qsrt_encoder(args.exllamav3_root)
    install_sqg_quantizer(quantizer_module)
    switched_luts, switched_lut_evidence = _outer_slope_k2_luts(device)
    codebook_sha256 = {
        "rank_t12": hashlib.sha256(
            sqg_xor_cheb_t12_rank_lut_bytes().cpu().numpy().tobytes()
        ).hexdigest(),
        **{
            f"k{bits}_direct_labels": hashlib.sha256(
                sqg_xor_cheb_t12_bytes(bits).cpu().numpy().tobytes()
            ).hexdigest()
            for bits in (1, 2, 3, 4)
        },
    }
    payload: dict[str, object] = {
        "kind": "qsrt_k2_tile_allocation_experiment",
        "schema_version": 2,
        "complete": False,
        "contract": {
            "experiment_implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "capture": str(args.capture.resolve()),
            "sample_cache": str(args.sample_cache.resolve()),
            "training_report": str(args.training_report.resolve()),
            "hessians": str(args.hessians.resolve()),
            "layer": args.layer,
            "experts": list(args.experts),
            "codebook": CODEBOOK_SQG_XOR_CHEB_T12,
            "codebook_sha256": codebook_sha256,
            "official_revision": args.official_revision,
            "ldlq_tf32": args.ldlq_tf32,
            "selection": (
                "strict exact dense-H improvement; no document partition selection"
                if args.h2_viterbi_refine
                else (
                    "zero-feedback schedule proposal followed by complete "
                    "mixed-rate BlockLDLQ dense-H selection"
                )
                if args.p13_search
                else "fit-document dense-H tile geometry"
            ),
            "evaluation": (
                "all available routed rows; same-population mechanism measurement"
                if args.h2_viterbi_refine
                else (
                    "routed document partitions are diagnostic only"
                    if args.p13_search
                    else "document-disjoint confirmation partition"
                )
            ),
            "external_validation": external_contract,
            "w2_hessian": (
                "mixed-rate candidates rebuild expert-local H2 from their "
                "decoded upstream reconstruction and shrink only toward "
                "scaled identity"
            ),
            "permutation_policy": args.permutation_policy,
            "allocation_search": {
                "qsrt_308_search": args.qsrt_308_search,
                "qsrt_308_tile_fractions": args.qsrt_308_tile_fractions,
                "qsrt_308_boundary_tile_search": args.qsrt_308_boundary_tile_search,
                "qsrt_308_boundary_proxy_search": args.qsrt_308_boundary_proxy_search,
                "qsrt_308_max_donors": args.qsrt_308_max_donors,
                "qsrt_308_scale_closure": args.qsrt_308_scale_closure,
                "functional_tile_search": args.functional_tile_search,
                "p13_search": args.p13_search,
                "k2_codebook_menu": args.k2_codebook_menu,
                "coupled_hadamard_k2": args.coupled_hadamard_k2,
                "h2_viterbi_refine": args.h2_viterbi_refine,
                "h2_viterbi_refine_sweeps": args.h2_viterbi_refine_sweeps,
                "coupled_hadamard_block_size": (
                    args.coupled_hadamard_block_size
                ),
                "coupled_hadamard_preactivation_block_size": (
                    args.coupled_hadamard_preactivation_block_size
                ),
                "coupled_hadamard_postactivation_block_size": (
                    args.coupled_hadamard_postactivation_block_size
                ),
                "coupled_hadamard_pre_permutation": (
                    args.coupled_hadamard_pre_permutation
                ),
                "coupled_residual_draw": args.coupled_residual_draw,
                "coupled_intermediate_draws": {
                    str(expert): draw
                    for expert, draw in sorted(
                        args.coupled_intermediate_draws.items()
                    )
                },
                "tile_search_limit": args.tile_search_limit,
            },
            "allocation_coordinates": {
                "optimization_order": (
                    "freeze_shared_neuron_permutation_then_build_rate_surfaces_"
                    "then_select_tile_map_then_full_reencode"
                ),
                "mathematical_permutation": (
                    "one_bijection_over_all_3072_intermediate_channels"
                ),
                "permutation_search_unit": (
                    "indivisible_contiguous_four_channel_group"
                ),
                "tile_aligned_policy_unit": (
                    "intact_sixteen_channel_band_with_fixed_record_membership"
                ),
                "incident_coefficient_tiles_per_neuron_band": 224,
                "funding_granularity": "16x16_codec_tile",
                "funding_basis": "post_permutation_encoder_coordinates",
                "w13_rate_map": "shared",
                "w2_rate_map": "independent",
                "policy_comparison": (
                    "absolute_document_disjoint_confirmation_sse_after_"
                    "fit_only_allocation"
                ),
                "local_tile_cost_role": "proposal_only",
            },
            "switched_k2_laws": switched_lut_evidence,
        },
        "results": {},
    }
    started = time.perf_counter()
    for expert in args.experts:
        print(f"layer {args.layer} expert {expert}: starting", flush=True)
        expert_started = time.perf_counter()
        result = _run_expert(
            layer=args.layer,
            expert=expert,
            samples=samples,
            partition=partition,
            external_samples=external_samples,
            external_requests=external_requests,
            global_h13=global_h13,
            global_h2=global_h2,
            store=store,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=args.ldlq_tf32,
            include_tile_triplet_oracle=args.include_tile_triplet_oracle,
            switched_luts=switched_luts,
            k2_codebook_menu=args.k2_codebook_menu,
            coupled_hadamard_k2=args.coupled_hadamard_k2,
            h2_viterbi_refine=args.h2_viterbi_refine,
            h2_viterbi_refine_sweeps=args.h2_viterbi_refine_sweeps,
            coupled_hadamard_block_size=args.coupled_hadamard_block_size,
            coupled_hadamard_preactivation_block_size=(
                args.coupled_hadamard_preactivation_block_size
            ),
            coupled_hadamard_postactivation_block_size=(
                args.coupled_hadamard_postactivation_block_size
            ),
            coupled_hadamard_pre_permutation=(
                args.coupled_hadamard_pre_permutation
            ),
            coupled_residual_draw=args.coupled_residual_draw,
            coupled_intermediate_draws=args.coupled_intermediate_draws,
            functional_tile_search=args.functional_tile_search,
            p13_search=args.p13_search,
            qsrt_308_search=args.qsrt_308_search,
            qsrt_308_tile_fractions=args.qsrt_308_tile_fractions,
            qsrt_308_boundary_tile_search=args.qsrt_308_boundary_tile_search,
            qsrt_308_boundary_proxy_search=args.qsrt_308_boundary_proxy_search,
            qsrt_308_max_donors=args.qsrt_308_max_donors,
            qsrt_308_scale_closure=args.qsrt_308_scale_closure,
            tile_search_chunk=args.tile_search_chunk,
            tile_search_limit=args.tile_search_limit,
            permutation_policy=args.permutation_policy,
        )
        result["seconds"] = time.perf_counter() - expert_started
        payload["results"][str(expert)] = result
        _atomic_write(args.output, payload)
        if not result["skipped"]:
            if args.h2_viterbi_refine:
                refinement = result["h2_viterbi_refinement"]
                print(
                    f"layer {args.layer} expert {expert}: "
                    f"h2-viterbi dense-h="
                    f"{100 * refinement['dense_h_relative_to_baseline']:+.3f}% "
                    f"mapped="
                    f"{100 * refinement['mapped_output_relative_to_baseline']:+.3f}% "
                    f"changed={refinement['path_changes']['tiles']} tiles",
                    flush=True,
                )
                torch.cuda.empty_cache()
                continue
            if (
                args.coupled_hadamard_k2
                and not (
                    args.p13_search
                    or args.qsrt_308_search
                    or args.k2_codebook_menu
                )
            ):
                coupled = result["coupled_hadamard_k2"]
                print(
                    f"layer {args.layer} expert {expert}: "
                    f"coupled-hadamard-k2 fit="
                    f"{100 * coupled['fit_relative_to_baseline']:+.3f}% "
                    f"confirm="
                    f"{100 * coupled['confirmation_relative_to_baseline']:+.3f}%",
                    flush=True,
                )
                torch.cuda.empty_cache()
                continue
            if args.k2_codebook_menu:
                menu = result["k2_codebook_menu"]
                selected_stats = menu["selector_stats"]["best_of_four"]
                external = menu[
                    "confirmation_gated_external_relative_to_normal"
                ]
                external_text = (
                    ""
                    if external is None
                    else f" external={100 * external:+.3f}%"
                )
                print(
                    f"layer {args.layer} expert {expert}: "
                    f"k2-menu={menu['selected_on_fit']} "
                    f"gated={menu['selected_after_confirmation_gate']} "
                    f"non-normal={100 * selected_stats['non_normal_fraction']:.2f}% "
                    f"confirm="
                    f"{100 * menu['confirmation_relative_to_normal']:+.3f}%"
                    f"{external_text}",
                    flush=True,
                )
                torch.cuda.empty_cache()
                continue
            if args.qsrt_308_search:
                qsrt_308 = result["qsrt_308"]
                print(
                    f"layer {args.layer} expert {expert}: "
                    f"qsrt-3.08={qsrt_308['selected_on_fit']} "
                    f"confirm="
                    f"{100 * qsrt_308['confirmation_relative_to_p33']:+.3f}%",
                    flush=True,
                )
                torch.cuda.empty_cache()
                continue
            if args.p13_search:
                records = result["p13_record_rates"]
                dense_h = records["mixed_schedule_dense_h"]
                routed = records["routed_expert_output_diagnostic"]
                print(
                    f"layer {args.layer} expert {expert}: "
                    f"records={records['selected_on_dense_h']} "
                    f"dense-h={100 * dense_h['relative_to_p22']:+.3f}% "
                    f"confirmation-diagnostic="
                    f"{100 * routed['confirmation_relative_to_p22']:+.3f}%",
                    flush=True,
                )
                torch.cuda.empty_cache()
                continue
            if args.functional_tile_search:
                functional = result["functional_tile_p24"]
                print(
                    f"layer {args.layer} expert {expert}: "
                    f"functional-tile={functional['selected_on_fit']} "
                    f"confirm="
                    f"{100 * functional['confirmation_relative_to_p33']:+.3f}% "
                    f"vs-endpoint="
                    f"{100 * functional['confirmation_relative_to_better_endpoint']:+.3f}%",
                    flush=True,
                )
                torch.cuda.empty_cache()
                continue
            fractional = result["fractional_p24"]
            record_triples = result["projection_triples_record"]
            tile_triples = result["projection_triples_tile_oracle"]
            switched = result["switched_k2_laws_full_p24"]
            tile_text = (
                ""
                if tile_triples is None
                else (
                    f" tile-triples="
                    f"{100 * tile_triples['confirmation_relative_to_p33']:+.3f}%"
                )
            )
            print(
                f"layer {args.layer} expert {expert}: "
                f"fraction={fractional['selected_on_fit']} "
                f"confirm={100 * fractional['confirmation_relative_to_p33']:+.3f}% "
                f"record-triples="
                f"{100 * record_triples['confirmation_relative_to_p33']:+.3f}%"
                f" switched={switched['selected_on_fit']} "
                f"{100 * switched['confirmation_relative_to_normal']:+.3f}%"
                f"{tile_text}",
                flush=True,
            )
        torch.cuda.empty_cache()
    payload["seconds"] = time.perf_counter() - started
    payload["complete"] = True
    _atomic_write(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--sample-cache", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument(
        "--external-sample-cache",
        type=Path,
        help=(
            "optional document-disjoint cache used only to score frozen "
            "K2 codebook-menu candidates"
        ),
    )
    parser.add_argument(
        "--external-report",
        type=Path,
        help="finalized corpus report for --external-sample-cache",
    )
    parser.add_argument("--hessians", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--experts",
        type=lambda value: _parse_ints(value, minimum=0, maximum=895),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument("--official-revision", default=None)
    parser.add_argument(
        "--exllamav3-root", type=Path, default=Path("/home/luke/projects/exllamav3")
    )
    parser.add_argument("--ldlq-tf32", action="store_true")
    parser.add_argument("--include-tile-triplet-oracle", action="store_true")
    parser.add_argument("--functional-tile-search", action="store_true")
    parser.add_argument(
        "--p13-search",
        action="store_true",
        help=(
            "compare uniform K2 with every gate/up/down permutation of "
            "K1/K2/K3 at exact average two-bit trellis rate"
        ),
    )
    parser.add_argument(
        "--k2-codebook-menu",
        action="store_true",
        help=(
            "test normal/q31/q33/q17 as a shared two-bit codebook menu "
            "over corresponding pure-K2 matrix tile triplets"
        ),
    )
    parser.add_argument(
        "--coupled-hadamard-k2",
        action="store_true",
        help=(
            "compare production-basis and coupled-boundary uniform K2 with "
            "candidate-local dense H2 and held-out functional scoring; with "
            "--qsrt-308-search, run that allocator entirely in the coupled "
            "basis instead"
        ),
    )
    parser.add_argument(
        "--h2-viterbi-refine",
        action="store_true",
        help=(
            "compare the qualified coupled-Hadamard uniform-K2 W2 encode "
            "with exact dense-H conditional-target path refinement"
        ),
    )
    parser.add_argument(
        "--h2-viterbi-refine-sweeps",
        type=int,
        default=1,
        help="number of exact conditional-target coordinate-descent sweeps",
    )
    parser.add_argument(
        "--coupled-hadamard-block-size",
        type=int,
        choices=(64, 128, 256, 512),
        default=512,
        help="block width of the exact coupled-boundary Hadamard gauge",
    )
    parser.add_argument(
        "--coupled-hadamard-preactivation-block-size",
        type=int,
        choices=(64, 128, 256, 512, 1024, 2048),
        default=512,
        help="block width over the 6,144 interleaved pre-SiTU coordinates",
    )
    parser.add_argument(
        "--coupled-hadamard-postactivation-block-size",
        type=int,
        choices=(64, 128, 256, 512, 1024),
        default=512,
        help="block width over the 3,072 post-SiTU/W2 coordinates",
    )
    parser.add_argument(
        "--coupled-hadamard-pre-permutation",
        choices=("identity", "selected"),
        default="identity",
        help=(
            "optionally apply the selected exact neuron permutation before "
            "interleaving gate/up and applying the coupled Hadamard"
        ),
    )
    parser.add_argument(
        "--coupled-residual-draw",
        type=int,
        default=0,
        help="layer-shared residual-boundary rotation draw",
    )
    parser.add_argument(
        "--coupled-intermediate-draws",
        type=_parse_expert_draws,
        default={},
        help="expert:draw overrides for expert-private intermediate rotations",
    )
    parser.add_argument("--qsrt-308-search", action="store_true")
    parser.add_argument(
        "--qsrt-308-tile-fractions",
        action="store_true",
        help="also evaluate the research-only tile-funding proxy schedules",
    )
    parser.add_argument(
        "--qsrt-308-boundary-tile-search",
        action="store_true",
        help="search partial P24 funding at the fit-selected record boundary",
    )
    parser.add_argument(
        "--qsrt-308-boundary-proxy-search",
        action="store_true",
        help=(
            "order boundary tiles by a cheap equal-byte weight proxy, then "
            "fully re-encode and functionally score the prefix candidates"
        ),
    )
    parser.add_argument(
        "--qsrt-308-max-donors",
        type=int,
        default=11,
        help="largest N in the exact N*K2+(22-2N)*K3+(N+2)*K4 schedule",
    )
    parser.add_argument(
        "--qsrt-308-scale-closure",
        action="store_true",
        help=(
            "path-aware fit-only scale closure for the isolated high-rate "
            "winner and P33"
        ),
    )
    parser.add_argument(
        "--permutation-policy",
        choices=PERMUTATION_CHOICES,
        default="h2_reverse",
    )
    parser.add_argument("--tile-search-chunk", type=int, default=8)
    parser.add_argument(
        "--tile-search-limit",
        type=int,
        default=0,
        help="smoke-test only; zero evaluates all 1,792 paired tile coordinates",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.layer <= 92:
        parser.error("--layer must lie in 1..92")
    if args.output.exists():
        parser.error("--output already exists")
    if (args.external_sample_cache is None) != (args.external_report is None):
        parser.error(
            "--external-sample-cache and --external-report must be supplied together"
        )
    if args.external_sample_cache is not None and not args.k2_codebook_menu:
        parser.error("external frozen scoring currently requires --k2-codebook-menu")
    if args.tile_search_chunk <= 0:
        parser.error("--tile-search-chunk must be positive")
    if not 0 <= args.tile_search_limit <= 1792:
        parser.error("--tile-search-limit must lie in 0..1792")
    if not 0 <= args.qsrt_308_max_donors <= 11:
        parser.error("--qsrt-308-max-donors must lie in 0..11")
    if args.coupled_residual_draw < 0:
        parser.error("--coupled-residual-draw must be nonnegative")
    if args.h2_viterbi_refine and not args.coupled_hadamard_k2:
        parser.error("--h2-viterbi-refine requires --coupled-hadamard-k2")
    if args.h2_viterbi_refine_sweeps <= 0:
        parser.error("--h2-viterbi-refine-sweeps must be positive")
    if args.h2_viterbi_refine and (
        args.coupled_hadamard_block_size != 512
        or args.coupled_hadamard_preactivation_block_size != 128
        or args.coupled_hadamard_postactivation_block_size != 128
        or args.coupled_hadamard_pre_permutation != "identity"
        or args.coupled_residual_draw != 0
    ):
        parser.error(
            "--h2-viterbi-refine requires the qualified H512/H128/H128 "
            "coupled basis, identity pre-permutation, and residual draw zero"
        )
    if any(
        expert not in args.experts or draw < 0
        for expert, draw in args.coupled_intermediate_draws.items()
    ):
        parser.error("coupled intermediate draws must name selected experts")
    if args.qsrt_308_tile_fractions and not args.qsrt_308_search:
        parser.error("--qsrt-308-tile-fractions requires --qsrt-308-search")
    if args.qsrt_308_boundary_tile_search and not args.qsrt_308_search:
        parser.error("--qsrt-308-boundary-tile-search requires --qsrt-308-search")
    if args.qsrt_308_boundary_proxy_search and not args.qsrt_308_search:
        parser.error("--qsrt-308-boundary-proxy-search requires --qsrt-308-search")
    if args.qsrt_308_scale_closure and not args.qsrt_308_search:
        parser.error("--qsrt-308-scale-closure requires --qsrt-308-search")
    if args.qsrt_308_boundary_tile_search and args.qsrt_308_boundary_proxy_search:
        parser.error("choose exact or proxy boundary proposals, not both")
    experiment_count = sum(
        (
            args.functional_tile_search,
            args.p13_search,
            args.qsrt_308_search,
            args.k2_codebook_menu,
            args.h2_viterbi_refine,
            args.coupled_hadamard_k2
            and not args.h2_viterbi_refine
            and not (
                args.p13_search
                or args.qsrt_308_search
                or args.k2_codebook_menu
            ),
        )
    )
    if experiment_count > 1:
        parser.error(
            "functional tile, exact-average-two-bit, 3.08-bpw, K2 "
            "codebook-menu, and H2 path-refinement searches are separate "
            "experiments; coupled Hadamard may qualify one search or run as "
            "its own uniform-K2 control"
        )
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

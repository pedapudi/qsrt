"""Metadata-free periodic K2/K3/K4 schedules for QSRT research.

A schedule assigns one branch width to each scalar position in a 256-value
trellis tile.  Three schedule classes cover one 768-position super-period so
the Kimi-K3 matrix tile count receives exactly 74/24 trellis bits per value
without a selector bitmap.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from typing import Iterable


POSITIONS_PER_TILE = 256
SCHEDULE_CLASSES = 3
RECORD_PERIOD = 24
BITS_PER_RECORD_PERIOD = 74


def high_rate_period(donor_records: int) -> tuple[int, ...]:
    """Return one 24-position period with exactly 74 branch bits.

    ``donor_records`` K2 positions finance the same number of additional K4
    positions beyond the two K4 positions in the 3.083333-bpw baseline.
    """

    if not 0 <= donor_records <= 11:
        raise ValueError("donor_records must lie in 0..11")
    return (
        (2,) * donor_records
        + (3,) * (22 - 2 * donor_records)
        + (4,) * (donor_records + 2)
    )


def _spread(values: Iterable[int]) -> tuple[int, ...]:
    """Distribute each rate count as evenly as possible over 24 positions."""

    counts = Counter(int(value) for value in values)
    result = [3] * RECORD_PERIOD
    occupied: set[int] = set()
    for bits in (2, 4):
        count = counts[bits]
        for index in range(count):
            position = ((2 * index + 1) * RECORD_PERIOD) // (2 * count)
            while position in occupied:
                position = (position + 1) % RECORD_PERIOD
            result[position] = bits
            occupied.add(position)
    return tuple(result)


def rate_period(
    donor_records: int,
    *,
    ordering: str,
    seed: int = 0,
) -> tuple[int, ...]:
    """Construct a deterministic 24-position rate period."""

    values = high_rate_period(donor_records)
    if ordering == "clustered":
        result = values
    elif ordering == "interleaved":
        result = _spread(values)
    elif ordering == "random":
        shuffled = list(values)
        random.Random(seed).shuffle(shuffled)
        result = tuple(shuffled)
    else:
        raise ValueError(f"unsupported periodic-rate ordering: {ordering}")
    if len(result) != RECORD_PERIOD or sum(result) != BITS_PER_RECORD_PERIOD:
        raise AssertionError("periodic-rate construction violated the 74-bit budget")
    return result


def tile_schedules(period: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Split 32 repeated 24-position periods into three 256-position classes."""

    if len(period) != RECORD_PERIOD or any(bits not in (2, 3, 4) for bits in period):
        raise ValueError("rate period must contain 24 K2/K3/K4 entries")
    if sum(period) != BITS_PER_RECORD_PERIOD:
        raise ValueError("rate period must contain exactly 74 bits")
    stream = period * 32
    schedules = tuple(
        stream[index * POSITIONS_PER_TILE : (index + 1) * POSITIONS_PER_TILE]
        for index in range(SCHEDULE_CLASSES)
    )
    if tuple(len(schedule) for schedule in schedules) != (256, 256, 256):
        raise AssertionError("periodic-rate classes do not cover 768 positions")
    if sum(map(sum, schedules)) != 32 * BITS_PER_RECORD_PERIOD:
        raise AssertionError("periodic-rate classes changed the total bit budget")
    return schedules


def schedule_sha256(schedules: tuple[tuple[int, ...], ...]) -> str:
    """Return the stable identity of a schedule-class bank."""

    return hashlib.sha256(bytes(bits for schedule in schedules for bits in schedule)).hexdigest()


__all__ = [
    "BITS_PER_RECORD_PERIOD",
    "POSITIONS_PER_TILE",
    "RECORD_PERIOD",
    "SCHEDULE_CLASSES",
    "high_rate_period",
    "rate_period",
    "schedule_sha256",
    "tile_schedules",
]

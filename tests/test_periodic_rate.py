from collections import Counter

import pytest

from qsrt.periodic_rate import (
    high_rate_period,
    rate_period,
    schedule_sha256,
    tile_schedules,
)


@pytest.mark.parametrize("donors", (0, 1, 2, 5, 11))
def test_high_rate_period_has_exact_budget(donors: int) -> None:
    period = high_rate_period(donors)
    assert len(period) == 24
    assert sum(period) == 74
    assert Counter(period) == {
        bits: count
        for bits, count in ((2, donors), (3, 22 - 2 * donors), (4, donors + 2))
        if count
    }


@pytest.mark.parametrize("ordering", ("clustered", "interleaved", "random"))
def test_tile_schedules_have_exact_superperiod_budget(ordering: str) -> None:
    period = rate_period(1, ordering=ordering, seed=17)
    schedules = tile_schedules(period)
    assert tuple(map(len, schedules)) == (256, 256, 256)
    assert sum(map(sum, schedules)) == 32 * 74
    assert sum(map(sum, schedules)) == 32 * 74
    assert len(schedule_sha256(schedules)) == 64


def test_random_period_is_seeded() -> None:
    first = rate_period(1, ordering="random", seed=17)
    assert first == rate_period(1, ordering="random", seed=17)
    assert first != rate_period(1, ordering="random", seed=18)

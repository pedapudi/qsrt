from __future__ import annotations

import pytest

from qsrt.glm52_blockldlq_feedback import _validate_feedback_multiplier


@pytest.mark.parametrize("value", (0.0, 0.25, 0.5, 0.999))
def test_feedback_ablation_accepts_bounded_multipliers(value: float) -> None:
    assert _validate_feedback_multiplier(value) == value


@pytest.mark.parametrize("value", (-0.1, 1.0, 1.5, float("nan"), True))
def test_feedback_ablation_rejects_non_ablation_values(value: object) -> None:
    with pytest.raises(ValueError, match="feedback multiplier"):
        _validate_feedback_multiplier(value)  # type: ignore[arg-type]

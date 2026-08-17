from __future__ import annotations

import json
from pathlib import Path

from qsrt.glm52_k3_k4_allocation import frozen_k4_rate_map


ROOT = Path(__file__).resolve().parents[1]
PRE_REGISTRATION = ROOT / "experiments/glm52_layer3_k3_k4_allocation_pre_registration.json"


def test_frozen_allocation_spends_exactly_twelve_projection_promotions() -> None:
    pre_registration = json.loads(PRE_REGISTRATION.read_text())
    rate_map = frozen_k4_rate_map(pre_registration)
    cells = {(expert, projection) for expert, projections in rate_map.items() for projection in projections}

    assert len(cells) == 12
    assert (204, "down_proj") in cells
    assert (89, "down_proj") in cells
    assert pre_registration["logical_byte_gate"]["mixed_qsrt_bytes_at_twelve_k4_projections"] < pre_registration["logical_byte_gate"]["comparison_exl3_bytes"]

from __future__ import annotations

import json
from pathlib import Path

from qsrt.glm52_k3_k4_allocation import frozen_k4_rate_map


ROOT = Path(__file__).resolve().parents[1]
PRE_REGISTRATION = ROOT / "experiments/glm52_layer3_k3_k4_allocation_pre_registration.json"
RATE_PRESERVING_PRE_REGISTRATION = (
    ROOT
    / "experiments/glm52_layer3_rate_preserving_down_refit_k3_k4_pre_registration.json"
)


def test_frozen_allocation_spends_exactly_twelve_projection_promotions() -> None:
    pre_registration = json.loads(PRE_REGISTRATION.read_text())
    rate_map = frozen_k4_rate_map(pre_registration)
    cells = {(expert, projection) for expert, projections in rate_map.items() for projection in projections}

    assert len(cells) == 12
    assert (204, "down_proj") in cells
    assert (89, "down_proj") in cells
    assert pre_registration["logical_byte_gate"]["mixed_qsrt_bytes_at_twelve_k4_projections"] < pre_registration["logical_byte_gate"]["comparison_exl3_bytes"]


def test_rate_preserving_pre_registration_separates_source_and_teacher_revisions() -> None:
    pre_registration = json.loads(RATE_PRESERVING_PRE_REGISTRATION.read_text())
    rate_map = frozen_k4_rate_map(pre_registration)

    assert sum(len(projections) for projections in rate_map.values()) == 12
    assert pre_registration["source_weights"]["revision"] == (
        "b4734de4facf877f85769a911abafc5283eab3d9"
    )
    assert pre_registration["reporting_reference"][
        "teacher_revision_recorded_by_manifest"
    ] == "4d67f66cc64d3219133b767c253b2ad1425c6c88"
    assert pre_registration["candidate_construction"]["reporting_context_used"] is False

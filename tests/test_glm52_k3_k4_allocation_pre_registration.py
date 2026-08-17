from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
PRE_REGISTRATION_PATH = (
    ROOT
    / "experiments"
    / "glm52_layer3_k3_k4_allocation_pre_registration.json"
)


def test_k3_k4_allocation_is_frozen_and_strictly_below_logical_exl3_bytes() -> None:
    registration = json.loads(PRE_REGISTRATION_PATH.read_text())
    ledger = registration["logical_byte_gate"]
    rate = registration["rate_contract"]
    base = registration["base_representation"]

    assert registration["status"] == "frozen_before_k4_candidate_measurement"
    assert rate["maximum_k4_projection_count"] == 12
    assert ledger["mixed_qsrt_bytes_at_twelve_k4_projections"] == (
        base["logical_bytes"]
        + rate["maximum_k4_projection_count"]
        * rate["k4_projection_increment_bytes"]
    )
    assert ledger["logical_margin_bytes"] == (
        ledger["comparison_exl3_bytes"]
        - ledger["mixed_qsrt_bytes_at_twelve_k4_projections"]
    )
    assert ledger["logical_margin_bytes"] > 0
    assert ledger["thirteen_k4_projection_bytes"] > ledger[
        "comparison_exl3_bytes"
    ]
    assert "not acceptable" in ledger["serialized_gate"]


def test_fixed_allocation_uses_twelve_unique_panel_projections() -> None:
    registration = json.loads(PRE_REGISTRATION_PATH.read_text())
    panel = set(registration["expert_panel"]["expert_order"])
    projections = registration["rate_contract"]["projection_order"]
    fixed = registration["fixed_exl3_rate_stratified_allocation"]
    promoted = [
        (record["expert"], record["projection"])
        for record in fixed["k4_projections"]
    ]

    assert len(promoted) == 12
    assert len(set(promoted)) == len(promoted)
    assert all(expert in panel for expert, _ in promoted)
    assert all(projection in projections for _, projection in promoted)
    assert fixed["candidate_measurements_used"] is False


def test_data_dependent_allocation_cannot_read_reporting_context() -> None:
    registration = json.loads(PRE_REGISTRATION_PATH.read_text())
    selection = registration["complete_expert_selection_allocation"]
    reporting = registration["reporting_context"]

    assert registration["selection_inputs"]["candidate_selection_window_count"] == 8
    assert selection["reporting_context_used"] is False
    assert "select an allocation" in reporting["prohibited_uses"]

from __future__ import annotations

import json
from pathlib import Path

from scripts.materialize_glm52_multi_layer_dense_intervention import (
    _validate_registration,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = (
    ROOT / "experiments/glm52_layers3_52_cross_layer_recovery_composition.json"
)
LAUNCHER = (
    ROOT
    / "experiments/build_and_screen_glm52_layers3_52_cross_layer_recovery_composition_on_kossel.sh"
)


def test_cross_layer_recovery_registration_freezes_two_favorable_components() -> None:
    value = json.loads(REGISTRATION.read_text())

    components = _validate_registration(value)

    assert [component["model_layer"] for component in components] == [3, 52]
    assert [component["expert_ids"] for component in components] == [[103], [36]]
    assert [component["logical_adapter_bytes"] for component in components] == [
        65536,
        0,
    ]
    assert all(
        component["public_document_mean_forward_kld_change"] < 0
        for component in components
    )
    assert all(
        component["single_reference_candidate_mean_forward_kld"]
        < component["single_reference_baseline_mean_forward_kld"]
        for component in components
    )


def test_cross_layer_recovery_registration_declares_reused_reference_boundary() -> None:
    value = json.loads(REGISTRATION.read_text())
    selection = value["selection_rule"]

    assert selection["absolute_target_mean_forward_kld"] == 0.059
    assert "same published 2,048-token" in selection["component_measurement_reference"]
    assert all(
        len(value) == 64
        for key, value in selection.items()
        if key.endswith("_sha256")
    )
    assert "exploratory interaction screen" in value["evidence_boundary"]
    assert "Independent document-level references" in value["evidence_boundary"]


def test_cross_layer_recovery_launcher_requires_manual_post_capture_start() -> None:
    source = LAUNCHER.read_text()

    assert "glm52-layers55-58-capture-stopping-point.json" in source
    assert 'record["status"] == "safe_for_host_shutdown"' in source
    assert "a GPU process is already active" in source
    assert "glm52_layers3_52_cross_layer_recovery_composition.json" in source
    assert "run_glm52_frozen_expert_subset_single_reference_on_kossel.sh" in source
    assert "predecessor" not in source
    assert "nohup" not in source

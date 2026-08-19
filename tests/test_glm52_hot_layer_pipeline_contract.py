from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bounded_hot_layer_download_names_only_registered_shards() -> None:
    manifest = json.loads(
        (ROOT / "experiments/glm52_layers_55_56_57_58_source_shards.json").read_text()
    )
    launcher = (
        ROOT
        / "experiments/download_glm52_layers_55_56_57_58_source_shards_after_gpu_queue_on_kossel.sh"
    ).read_text()

    assert manifest["layers"] == [55, 56, 57, 58]
    assert len(manifest["shards"]) == 16
    assert manifest["total_bytes"] == sum(row["size"] for row in manifest["shards"])
    assert len({row["file"] for row in manifest["shards"]}) == 16
    assert "parallel_downloads=16" in launcher
    assert "expected_shards=16" in launcher
    assert "expected_bytes=85783011360" in launcher
    assert "download-manifest.json" in launcher
    assert "receipt.json" in launcher


def test_hot_layer_capture_uses_arm_invariant_inputs_and_no_model_downloads() -> None:
    launcher = (
        ROOT
        / "experiments/run_glm52_layers55_56_57_58_input_capture_after_source_download_on_kossel.sh"
    ).read_text()

    assert "QSRT_GLM52_ACTIVATION_CAPTURE_LAYERS=55,56,57,58" in launcher
    assert launcher.count("--activation-capture-panel-manifest") == 4
    assert "--activation-capture-only" in launcher
    assert "--network none" in launcher
    assert "qsrt.model-downloads-performed=false" in launcher
    assert "source-windows/glm52-b4734de-layers-55-56-57-58" in launcher
    assert "receipt.json" in launcher
    assert "expected_capture_bytes=5000000000" in launcher


def test_hot_layer_recovery_pipeline_rebuilds_each_dependent_candidate() -> None:
    launcher = (
        ROOT
        / "experiments/build_and_screen_glm52_hot_layer_down_recovery_on_kossel.sh"
    ).read_text()

    assert "build_uniform_k3" in launcher
    assert "build_down_refit \"${uniform_name}\"" in launcher
    assert "build_uniform_k4" in launcher
    assert 'build_k4_down_refit "${uniform_name}" "${refit_name}"' in launcher
    assert "build_glm52_all_panel_k4_down_registration.py" in launcher
    assert "build_glm52_down_refit_rate_pool.py" in launcher
    assert "materialize_glm52_registered_partial_rate_map.py" in launcher
    assert '"k4-down-refit"' in launcher
    assert "build_rank4_down_recovery \"${refit_name}\"" in launcher
    assert "--base-construction reconstructed_activation_down_refit" in launcher
    assert "--include-complete-panel" in launcher
    assert "candidate-subset-public-reference-selection" in launcher
    assert "--network none" in launcher
    assert "qsrt.model-downloads-performed=false" in launcher
    assert "source-windows/glm52-b4734de-layers-55-56-57-58" in launcher
    assert "GLM-5.2-b4734de-layer003-source-window" not in launcher


def test_hot_layer_recovery_waits_for_a_complete_capture() -> None:
    launcher = (
        ROOT
        / "experiments/continue_glm52_hot_layer_recovery_after_input_capture_on_kossel.sh"
    ).read_text()

    assert "docker inspect \"${capture_container}\"" in launcher
    assert "State.ExitCode" in launcher
    assert "test -f \"${capture_index}\"" in launcher
    assert "for layer in 55 56 57 58" in launcher
    assert "bash \"${pipeline}\" \"${layer}\"" in launcher


def test_hot_layer_panels_are_frozen_before_candidate_measurement() -> None:
    expected = {
        55: [130, 74, 172, 83, 249, 84, 214, 232],
        56: [186, 215, 221, 197, 2, 173, 209, 201],
        57: [111, 164, 4, 194, 168, 90, 179, 58],
        58: [156, 55, 133, 165, 86, 76, 96, 21],
    }
    for layer, experts in expected.items():
        panel = json.loads(
            (ROOT / f"experiments/glm52_layer{layer}_rate_pattern_panel.json").read_text()
        )
        assert panel["selection_status"] == "frozen_before_candidate_measurement"
        assert panel["layer"] == layer
        assert [row["expert"] for row in panel["experts"]] == experts
        assert "error-blind" in panel["evidence_role"]

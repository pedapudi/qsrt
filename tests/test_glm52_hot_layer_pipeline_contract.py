from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _first_quoted_python_heredoc(path: Path) -> str:
    lines = path.read_text().splitlines()
    start = next(
        index + 1 for index, line in enumerate(lines) if "<<'PY'" in line
    )
    end = next(
        index for index in range(start, len(lines)) if lines[index] == "PY"
    )
    return "\n".join(lines[start:end]) + "\n"


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


def test_hot_layer_long_screen_freezes_a_cross_layer_candidate_first() -> None:
    launcher = (
        ROOT
        / "experiments/continue_glm52_hot_layer_retained_arms_to_single_reference_on_kossel.sh"
    ).read_text()

    assert "best_direct_record_by_layer" in launcher
    assert "selected_cross_layer_records" in launcher
    assert "logical_additional_bytes" in launcher
    assert "1_572_864 * expert_count" in launcher
    assert "65_536 * expert_count" in launcher
    assert (
        "glm52_layers55_56_57_58_public_document_selected_cross_layer_composition.json"
        in launcher
    )
    assert "materialize_glm52_multi_layer_dense_intervention.py" in launcher
    assert 'if test "${artifact_kind}" = "multi_layer"' in launcher
    queue_write = launcher.index("os.replace(temporary, queue_path)")
    long_screen_loop = launcher.index("while IFS=$'\\t' read -r artifact_kind")
    assert queue_write < long_screen_loop

    single_reference_launcher = (
        ROOT
        / "experiments/run_glm52_frozen_expert_subset_single_reference_on_kossel.sh"
    ).read_text()
    assert "8 * len(model_layers)" in single_reference_launcher


def test_hot_layer_queue_selects_one_public_document_arm_per_layer(
    tmp_path: Path,
) -> None:
    launcher_path = (
        ROOT
        / "experiments/continue_glm52_hot_layer_retained_arms_to_single_reference_on_kossel.sh"
    )
    experiment_root = tmp_path / "experiment"
    constructions = {
        "down-refit": "hot-band-frozen8-reconstructed-activation-down-refit-merged",
        "k4-down-refit": "hot-band-frozen8-k3-gate-k3-up-k4-down-refit-merged",
        "rank4-down-recovery": (
            "hot-band-frozen8-low-rank-down-refit-bf16-rank4-merged"
        ),
        "uniform-k3": "hot-band-frozen8-uniform-k3-merged",
    }
    priorities = {
        "down-refit": -0.001,
        "k4-down-refit": -0.002,
        "rank4-down-recovery": -0.003,
        "uniform-k3": -0.0005,
    }
    for layer in (55, 56, 57, 58):
        for construction, artifact_suffix in constructions.items():
            selection_result = (
                f"glm52-layer{layer}-{construction}-candidate-subset-"
                "public-reference-selection"
            )
            report_path = experiment_root / "results" / selection_result / "report.json"
            report_path.parent.mkdir(parents=True)
            report_path.write_text(json.dumps({"status": "complete"}))
            decision_path = (
                experiment_root
                / "launch-records"
                / selection_result
                / "selection-decision.json"
            )
            decision_path.parent.mkdir(parents=True)
            priority = priorities[construction]
            decision_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "report_sha256": hashlib.sha256(
                            report_path.read_bytes()
                        ).hexdigest(),
                        "arms": [
                            {
                                "retained": True,
                                "selected_experts": [layer - 50],
                                "name": "expert-alone",
                                "screening_document_mean_delta": priority,
                                "selection_check_document_mean_delta": priority,
                                "all_document_mean_delta": priority,
                            }
                        ],
                    }
                )
            )
            artifact_report_path = (
                experiment_root
                / "results"
                / f"glm52-layer{layer}-{artifact_suffix}"
                / "report.json"
            )
            artifact_report_path.parent.mkdir(parents=True)
            artifact_report_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "manifest_sha256": f"{layer:02x}" * 32,
                    }
                )
            )

    queue_path = experiment_root / "launch-records/queue.json"
    registration_path = experiment_root / "registrations/cross-layer.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-",
            str(experiment_root),
            str(queue_path),
            str(registration_path),
            "cross-layer-artifact",
        ],
        input=_first_quoted_python_heredoc(launcher_path),
        text=True,
        capture_output=True,
        check=True,
    )
    assert "retained_arm_count" in completed.stdout
    registration = json.loads(registration_path.read_text())
    queue = json.loads(queue_path.read_text())

    assert [record["model_layer"] for record in registration["components"]] == [
        55,
        56,
        57,
        58,
    ]
    assert all(
        record["construction"] == "rank4-down-recovery"
        for record in registration["components"]
    )
    assert all(
        record["logical_additional_bytes"] == 65_536
        for record in registration["components"]
    )
    cross_layer = next(
        record for record in queue["records"] if record["artifact_kind"] == "multi_layer"
    )
    assert cross_layer["model_layers"] == [55, 56, 57, 58]
    assert cross_layer["logical_additional_bytes"] == 4 * 65_536
    assert queue["status"] == "frozen_before_single_reference_measurement"


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

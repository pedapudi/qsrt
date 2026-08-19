import json
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPOSITORY_ROOT
    / "experiments/continue_glm52_hot_layer_retained_arms_to_single_reference_on_kossel.sh"
)


def test_hot_layer_single_reference_queue_is_frozen_before_measurement() -> None:
    source = SCRIPT.read_text()
    freeze = source.index("os.replace(temporary, queue_path)")
    measurement = source.index('"${single_reference_launcher}" "${artifact_name}"')
    assert freeze < measurement
    assert "all_document_mean_delta" in source
    assert "absolute_development_target_mean_kld" in source
    assert "One 2,048-token reference is a development screen" in source


def test_hot_layer_single_reference_queue_uses_only_retained_arms() -> None:
    source = SCRIPT.read_text()
    assert 'if not arm.get("retained"):' in source
    assert 'decision.get("report_sha256")' in source
    assert 'artifact_report["manifest_sha256"]' in source
    assert 'subset = report.get("candidate_expert_subset_paired", {}).get(' in source
    assert '"frozen_expert_subset"' in source


def test_hot_layer_single_reference_queue_freezes_singleton_unions() -> None:
    source = SCRIPT.read_text()
    assert 'if len(retained_singletons) >= 2:' in source
    assert 'arm_name = "union-of-retained-singletons"' in source
    assert '"component_singleton_arm_names"' in source
    assert '"selection_priority_score": sum(' in source
    assert (
        "deterministic union of every singleton that passed both "
        in source
    )


def test_hot_layer_single_reference_script_has_valid_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_hot_layer_single_reference_queue_schema_is_descriptive() -> None:
    source = SCRIPT.read_text()
    schemas = [
        "qsrt_glm52_hot_layer_retained_arm_single_reference_queue",
        "qsrt_glm52_hot_layer_retained_arm_single_reference_results",
    ]
    for schema in schemas:
        assert schema in source
    assert json.loads('{"target": 0.059}')["target"] == 0.059

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = (
    ROOT / "scripts/run_glm52_document_disjoint_reference_confirmation.py"
)
SEMANTIC_ENTRY_POINT = (
    ROOT / "scripts/run_glm52_document_disjoint_candidate_evaluation.py"
)
LAUNCHER = (
    ROOT
    / "experiments/evaluate_glm52_layer3_expert103_rank4_on_terminal_screening_references.sh"
)
FREEZE_LAUNCHER = (
    ROOT
    / "experiments/freeze_glm52_layer3_expert103_rank4_after_terminal_screening.sh"
)
CONFIRMATION_GENERATOR = (
    ROOT / "experiments/generate_glm52_confirmation_teacher_logits_on_kossel.sh"
)
SCREENING_GENERATOR = (
    ROOT / "experiments/generate_glm52_screening_teacher_logits_on_kossel.sh"
)
CONFIRMATION_LAUNCHER = (
    ROOT
    / "experiments/evaluate_glm52_layer3_expert103_rank4_on_terminal_confirmation_references.sh"
)


def _runner_module():
    spec = importlib.util.spec_from_file_location(
        "glm52_document_disjoint_candidate_evaluation", IMPLEMENTATION
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_kinds_are_mutually_exclusive() -> None:
    parser = _runner_module().build_parser()
    common = [
        "--model",
        "/model",
        "--reference-directory",
        "/reference",
        "--reference-link",
        "/control/reference.safetensors",
        "--intervention-artifact",
        "/artifact",
        "--control",
        "/control/control.json",
        "--dest",
        "/results/evaluation",
    ]
    parsed = parser.parse_args(
        common
        + [
            "--terminal-reference-plan",
            "/reference-plan.json",
            "--evaluation-tier",
            "screening",
        ]
    )
    assert parsed.evaluation_tier == "screening"
    with pytest.raises(SystemExit):
        parser.parse_args(
            common
            + [
                "--reference-plan",
                "/public-plan.json",
                "--terminal-reference-plan",
                "/terminal-plan.json",
            ]
        )


def test_terminal_screening_launcher_is_offline_and_explicit() -> None:
    source = LAUNCHER.read_text()
    entry_point = SEMANTIC_ENTRY_POINT.read_text()

    assert "--network none" in source
    assert "--pull never" in source
    assert "QSRT_GLM52_ENGINE_KLD_REFERENCE_KEY=logits" in source
    assert "QSRT_GLM52_ENGINE_KLD_REFERENCE_REPRESENTATION=logits" in source
    assert "--evaluation-tier screening" in source
    assert "--max-model-len 2049" in source
    assert "--max-num-batched-tokens 2048" in source
    assert "test ! -e \"${RESULT_ROOT}\"" in source
    assert "terminal-confirmation-freeze" not in source
    assert "run_glm52_document_disjoint_reference_confirmation" in entry_point


def test_confirmation_pipeline_is_sealed_and_offline() -> None:
    freeze = FREEZE_LAUNCHER.read_text()
    generator = CONFIRMATION_GENERATOR.read_text()
    screening_generator = SCREENING_GENERATOR.read_text()
    confirmation = CONFIRMATION_LAUNCHER.read_text()

    assert "--network none" in freeze
    assert "--screening-report /screening-report.json" in freeze
    assert "--teacher-reference-plan /reference-plan.json" in freeze
    assert "--pull never" in generator
    assert "--network none" in generator
    assert "--evaluation-tier confirmation" in generator
    assert "--confirmation-freeze /confirmation-freeze.json" in generator
    assert "--screening-report /screening-report.json" in generator
    assert "test ! -e \"${DESTINATION}\"" in generator
    for source in (screening_generator, generator):
        assert 'TEACHER_LOGIT_DEVICES="${TEACHER_LOGIT_DEVICES:-0,1,2,3}"' in source
        assert '--devices "${TEACHER_LOGIT_DEVICES}"' in source
        assert 'nvidia-smi --id="${device}"' in source
    assert "--network none" in confirmation
    assert "--pull never" in confirmation
    assert "--evaluation-tier confirmation" in confirmation
    assert "--terminal-confirmation-freeze /confirmation-freeze.json" in confirmation
    assert "--terminal-screening-report /screening-report.json" in confirmation
    assert "report[\"confirmation_decision\"]" in confirmation

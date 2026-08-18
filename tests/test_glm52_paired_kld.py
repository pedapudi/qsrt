from __future__ import annotations

import hashlib
import importlib.util
import json
import numpy as np
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

from qsrt.glm52_expert_intervention_runtime import _capture_layer_input

from qsrt.glm52_paired_kld import (
    forward_kld_per_position,
    paired_kld_summary,
    route_support_summary,
    target_layer_routes,
)


RUNNER_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_glm52_paired_expert_intervention_kld.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("glm52_paired_kld_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_defaults_match_the_qualified_r7_fused_runtime() -> None:
    runner = _load_runner()
    parser = runner.build_parser()
    args = parser.parse_args(
        [
            "--model",
            "/model",
            "--reference-logits",
            "/reference",
            "--intervention-artifact",
            "/artifact",
            "--control",
            "/control.json",
            "--dest",
            "/results",
        ]
    )

    assert args.kv_cache_dtype == "nvfp4_ds_mla"
    assert args.load_format == "safetensors"
    assert args.quantization == "exl3"
    assert args.attention_backend == "B12X_MLA_SPARSE"
    assert args.max_model_len == 2049
    assert args.gpu_memory_utilization == 0.95
    assert args.kld_device == "cpu"
    assert args.source_sparse_index_topk is None
    assert args.reporting_activation_capture_dir is None
    assert args.activation_capture_only is False
    assert args.omit_individual_expert_arms is False
    assert args.candidate_expert_subsets_json == "{}"
    assert args.candidate_runtime_mode == "candidate"
    assert args.measurement_controls_only is False
    assert "does not provide document-level replication" in (
        runner.PAIRED_KLD_EVIDENCE_BOUNDARY
    )
    assert "not a serialized QSRT decoder output" in (
        runner.PAIRED_KLD_EVIDENCE_BOUNDARY
    )
    assert "has not been established separately" in (
        runner.PAIRED_KLD_EVIDENCE_BOUNDARY
    )


def test_runner_accepts_capture_only_mode() -> None:
    runner = _load_runner()
    args = runner.build_parser().parse_args(
        [
            "--model",
            "/model",
            "--reference-logits",
            "/reference",
            "--intervention-artifact",
            "/artifact",
            "--control",
            "/control.json",
            "--dest",
            "/results",
            "--corpus-plan",
            "/corpus.json",
            "--activation-capture-dir",
            "/capture",
            "--activation-capture-only",
        ]
    )

    assert args.activation_capture_only is True
    assert args.activation_capture_dir == Path("/capture")


def test_one_prompt_run_captures_multiple_frozen_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    corpus_plan_path = tmp_path / "corpus-plan.json"
    windows = {
        collection: {
            "window_count": 1,
            "windows": [
                {
                    "window_id": f"{collection}-document",
                    "token_count": 128,
                    "token_ids": list(range(128)),
                }
            ],
        }
        for collection in ("activation_fit", "candidate_selection")
    }
    corpus_plan_path.write_text(
        json.dumps(
            {
                "schema": "qsrt_glm52_document_disjoint_corpus_plan",
                "schema_version": 1,
                **windows,
                "separation": {
                    "fit_selection_row_overlap": 0,
                    "reference_fit_row_overlap": 0,
                    "reference_selection_row_overlap": 0,
                    "unit": "WikiText article delimited by a top-level heading",
                },
            }
        )
    )
    capture_dir = tmp_path / "captures"
    control_path = tmp_path / "control.json"
    plan_sha256 = hashlib.sha256(corpus_plan_path.read_bytes()).hexdigest()
    monkeypatch.setenv("QSRT_GLM52_ACTIVATION_CAPTURE_DIR", str(capture_dir))
    monkeypatch.setenv("QSRT_GLM52_ACTIVATION_CAPTURE_PLAN_SHA256", plan_sha256)
    monkeypatch.setenv("QSRT_GLM52_ACTIVATION_CAPTURE_LAYERS", "52,60")
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(SamplingParams=lambda **kwargs: kwargs),
    )

    class FakeLlm:
        def generate(self, prompts: object, sampling_params: object) -> list[object]:
            del prompts, sampling_params
            generation = json.loads(control_path.read_text())["generation"]
            routes = np.zeros((128, 78, 8), dtype=np.int64)
            for model_layer, expert in ((52, 5), (60, 7)):
                routes[:, model_layer, 0] = expert
                layer_root = capture_dir / f"layer-{model_layer:03d}"
                _capture_layer_input(
                    root=layer_root,
                    model_layer=model_layer,
                    x=torch.zeros((128, 6144), dtype=torch.bfloat16),
                    topk_weights=torch.full((128, 8), 0.125),
                    topk_ids=torch.tensor(routes[:, model_layer, :]),
                    generation=generation,
                    plan_sha256=plan_sha256,
                )
            return [SimpleNamespace(outputs=[SimpleNamespace(routed_experts=routes)])]

    index = runner._capture_planned_layer_inputs(
        FakeLlm(),
        corpus_plan_path=corpus_plan_path,
        capture_dir=capture_dir,
        control_path=control_path,
        artifact_manifest_sha256="a" * 64,
        selected_experts_by_layer={52: (5,), 60: (7,)},
    )

    assert index["schema"] == "qsrt_glm52_multi_layer_input_capture_index"
    assert index["model_layers"] == [52, 60]
    for model_layer in (52, 60):
        manifest = json.loads(
            (capture_dir / f"layer-{model_layer:03d}" / "manifest.json").read_text()
        )
        assert manifest["model_layer"] == model_layer
        assert manifest["collections"] == {
            "activation_fit": 1,
            "candidate_selection": 1,
        }
        assert len(manifest["records"]) == 2


def test_reporting_capture_is_hash_bound_and_selection_forbidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    token_ids = [11, 12, 13, 14]
    reference_manifest_path = tmp_path / "reference-manifest.json"
    reference_manifest_path.write_text(
        json.dumps({"context_length": len(token_ids), "token_first16": token_ids})
    )
    reference_sha256 = hashlib.sha256(
        reference_manifest_path.read_bytes()
    ).hexdigest()
    corpus_plan_path = tmp_path / "corpus-plan.json"
    corpus_plan_path.write_text(
        json.dumps(
            {
                "schema": "qsrt_glm52_document_disjoint_corpus_plan",
                "schema_version": 1,
                "context_length": len(token_ids),
                "published_bf16_reference": {
                    "manifest_sha256": reference_sha256,
                    "token_first16": token_ids,
                    "role": "untouched BF16-reference reporting context",
                },
                "separation": {
                    "fit_selection_row_overlap": 0,
                    "reference_fit_row_overlap": 0,
                    "reference_selection_row_overlap": 0,
                    "unit": "WikiText article delimited by a top-level heading",
                },
            }
        )
    )
    capture_dir = tmp_path / "reporting-capture"
    control_path = tmp_path / "control.json"
    plan_sha256 = hashlib.sha256(corpus_plan_path.read_bytes()).hexdigest()
    monkeypatch.setenv("QSRT_GLM52_ACTIVATION_CAPTURE_DIR", str(capture_dir))
    monkeypatch.setenv(
        "QSRT_GLM52_ACTIVATION_CAPTURE_PLAN_SHA256", plan_sha256
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(SamplingParams=lambda **kwargs: kwargs),
    )

    routes = np.zeros((len(token_ids), 78, 8), dtype=np.int64)
    routes[:, 3, :] = np.array([64, 208, 1, 2, 3, 4, 5, 6])

    class FakeLlm:
        def generate(self, prompts: object, sampling_params: object) -> list[object]:
            del prompts, sampling_params
            generation = json.loads(control_path.read_text())["generation"]
            _capture_layer_input(
                root=capture_dir,
                model_layer=3,
                x=torch.zeros((len(token_ids), 6144), dtype=torch.bfloat16),
                topk_weights=torch.full((len(token_ids), 8), 0.125),
                topk_ids=torch.tensor(routes[:, 3, :]),
                generation=generation,
                plan_sha256=plan_sha256,
            )
            return [
                SimpleNamespace(
                    outputs=[SimpleNamespace(routed_experts=routes)]
                )
            ]

    manifest = runner._capture_reporting_layer_input(
        FakeLlm(),
        token_ids=token_ids,
        corpus_plan_path=corpus_plan_path,
        reference_manifest_path=reference_manifest_path,
        capture_dir=capture_dir,
        control_path=control_path,
        artifact_manifest_sha256="a" * 64,
        selected_experts=[64, 208],
    )

    assert manifest["status"] == "complete"
    assert manifest["collections"] == {"untouched_reporting_context": 1}
    assert manifest["reference_manifest_sha256"] == reference_sha256
    assert "must not select paths" in manifest["reuse_policy"]
    assert manifest["records"][0]["route_support"]["selected_route_count"] == 8
    assert json.loads(control_path.read_text())["capture_enabled"] is False
    assert (capture_dir / "manifest.json").is_file()


def test_runner_defines_individual_and_complete_panel_arms() -> None:
    runner = _load_runner()
    expert_ids = [64, 208, 106, 204, 89, 212, 96, 103]

    arms = runner.intervention_arm_definitions(
        expert_ids, omit_individual_expert_arms=False
    )
    assert len(arms) == 12
    assert len({name for name, _, _ in arms}) == len(arms)
    assert arms[:3] == [
        ("resident_exl3", "off", None),
        ("resident_exl3_repeat", "off", None),
        ("dense_resident_identity", "dense_resident_identity", None),
    ]
    assert arms[3:-1] == [
        (f"selected_candidate_expert_{expert:03d}", "candidate", [expert])
        for expert in expert_ids
    ]
    assert arms[-1] == ("selected_candidate", "candidate", None)

    aggregate_only = runner.intervention_arm_definitions(
        expert_ids, omit_individual_expert_arms=True
    )
    assert aggregate_only == [arms[0], arms[1], arms[2], arms[-1]]

    controls_only = runner.intervention_arm_definitions(
        expert_ids,
        omit_individual_expert_arms=False,
        measurement_controls_only=True,
    )
    assert controls_only == arms[:3]


def test_runner_rejects_invalid_intervention_expert_ids() -> None:
    runner = _load_runner()
    for invalid in ([], [1, 1], [-1], [256], [True]):
        with pytest.raises(ValueError, match="unique values from 0 to 255"):
            runner.intervention_arm_definitions(
                invalid, omit_individual_expert_arms=False
            )


def test_runner_validates_and_defines_named_candidate_subsets() -> None:
    runner = _load_runner()
    expert_ids = [64, 208, 106]
    subsets = runner.parse_candidate_expert_subsets(
        '{"locally_helpful_experts":[64,106],"strongest_expert":[64]}',
        artifact_expert_ids=expert_ids,
    )
    arms = runner.intervention_arm_definitions(
        expert_ids,
        omit_individual_expert_arms=True,
        candidate_expert_subsets=subsets,
    )

    assert arms[-3:] == [
        (
            "selected_candidate_subset_locally_helpful_experts",
            "candidate",
            [64, 106],
        ),
        ("selected_candidate_subset_strongest_expert", "candidate", [64]),
        ("selected_candidate", "candidate", None),
    ]

    invalid_values = (
        "[]",
        '{"opaque-label":[64]}',
        '{"missing_expert":[89]}',
        '{"repeated":[64,64]}',
        '{"empty":[]}',
    )
    for value in invalid_values:
        with pytest.raises((TypeError, ValueError)):
            runner.parse_candidate_expert_subsets(
                value, artifact_expert_ids=expert_ids
            )


def test_runner_can_execute_named_subsets_through_factorized_down() -> None:
    runner = _load_runner()
    arms = runner.intervention_arm_definitions(
        [89, 103],
        omit_individual_expert_arms=True,
        candidate_expert_subsets={"frozen_expert_103": [103]},
        candidate_runtime_mode="factorized_low_rank_candidate",
    )

    assert arms[-2:] == [
        (
            "selected_candidate_subset_frozen_expert_103",
            "factorized_low_rank_candidate",
            [103],
        ),
        ("selected_candidate", "factorized_low_rank_candidate", None),
    ]


def test_runner_can_execute_named_subsets_through_load_time_materialization() -> None:
    runner = _load_runner()
    mode = "stored_low_rank_factors_materialized_at_load_candidate"
    arms = runner.intervention_arm_definitions(
        [89, 103],
        omit_individual_expert_arms=True,
        candidate_expert_subsets={"frozen_expert_103": [103]},
        candidate_runtime_mode=mode,
    )

    assert arms[-2:] == [
        ("selected_candidate_subset_frozen_expert_103", mode, [103]),
        ("selected_candidate", mode, None),
    ]


def test_measurement_controls_require_kld_and_route_bitwise_equality() -> None:
    runner = _load_runner()
    baseline_kld = torch.tensor([0.1, 0.2], dtype=torch.float64)
    baseline_routes = np.arange(12).reshape(2, 3, 2)
    klds = {
        "resident_exl3": baseline_kld,
        "resident_exl3_repeat": baseline_kld.clone(),
        "dense_resident_identity": baseline_kld.clone(),
    }
    routes = {
        "resident_exl3": baseline_routes,
        "resident_exl3_repeat": baseline_routes.copy(),
        "dense_resident_identity": baseline_routes.copy(),
    }

    passed = runner.measurement_control_summary(klds, routes)
    assert passed["passed"] is True
    assert passed["resident_repeatability_control"][
        "forward_kld_bitwise_equal"
    ]
    assert passed["dense_resident_identity_control"][
        "all_layer_route_array_equal"
    ]

    klds["resident_exl3_repeat"][1] += 1e-12
    failed_kld = runner.measurement_control_summary(klds, routes)
    assert failed_kld["passed"] is False

    klds["resident_exl3_repeat"] = baseline_kld.clone()
    routes["dense_resident_identity"][1, 2, 1] += 1
    failed_route = runner.measurement_control_summary(klds, routes)
    assert failed_route["passed"] is False


def test_forward_kld_has_teacher_to_candidate_direction() -> None:
    reference = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    identical = forward_kld_per_position(reference, reference, chunk_rows=1)
    changed = forward_kld_per_position(
        reference, torch.tensor([[0.0, 3.0], [3.0, 0.0]]), chunk_rows=1
    )
    assert torch.allclose(identical, torch.zeros_like(identical))
    assert torch.all(changed > 0.0)


def test_forward_kld_explicit_cpu_device_matches_default() -> None:
    reference = torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.0, -0.5]])
    model = torch.tensor([[0.75, 0.0, -0.75], [0.25, 0.0, -0.25]])
    expected = forward_kld_per_position(reference, model, chunk_rows=1)
    explicit = forward_kld_per_position(
        reference,
        model,
        chunk_rows=2,
        compute_device="cpu",
    )
    assert torch.equal(expected, explicit)


def test_target_layer_routes_accepts_both_vllm_axis_conventions() -> None:
    dense_inclusive = np.arange(4 * 78 * 2).reshape(4, 78, 2)
    moe_only = dense_inclusive[:, 3:, :]
    assert np.array_equal(
        target_layer_routes(
            dense_inclusive,
            model_layer=3,
            total_decoder_layers=78,
            first_moe_layer=3,
        ),
        dense_inclusive[:, 3, :],
    )
    assert np.array_equal(
        target_layer_routes(
            moe_only,
            model_layer=3,
            total_decoder_layers=78,
            first_moe_layer=3,
        ),
        dense_inclusive[:, 3, :],
    )


def test_route_support_and_paired_summary_are_explicit() -> None:
    routes = np.array([[1, 2], [3, 1], [4, 5]])
    support = route_support_summary(routes, selected_experts=[1, 5])
    assert support["selected_route_count"] == 3
    assert support["selected_token_count"] == 3
    assert support["route_count_by_expert"] == {"1": 2, "5": 1}

    summary = paired_kld_summary(
        torch.tensor([0.4, 0.2, 0.3]), torch.tensor([0.3, 0.2, 0.4])
    )
    assert summary["position_count"] == 3
    assert summary["candidate_better_position_count"] == 1
    assert summary["candidate_equal_position_count"] == 1
    assert summary["candidate_worse_position_count"] == 1
    assert "not constitute document-level replicates" in summary["evidence_boundary"]


def test_kld_and_route_helpers_reject_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="same rank-two shape"):
        forward_kld_per_position(torch.ones(2, 3), torch.ones(2, 2))
    with pytest.raises(ValueError, match="layer axis"):
        target_layer_routes(
            np.zeros((2, 12, 8)),
            model_layer=3,
            total_decoder_layers=78,
            first_moe_layer=3,
        )
    with pytest.raises(ValueError, match="nonempty unique"):
        route_support_summary(np.zeros((2, 8)), selected_experts=[])
    with pytest.raises(ValueError, match="CPU or CUDA"):
        forward_kld_per_position(
            torch.ones(2, 3),
            torch.ones(2, 3),
            compute_device="meta",
        )

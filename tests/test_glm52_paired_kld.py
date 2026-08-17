from __future__ import annotations

import importlib.util
import numpy as np
from pathlib import Path
import pytest
import torch

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
    assert args.omit_individual_expert_arms is False
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
        (f"selected_qsrt_k3_expert_{expert:03d}", "qsrt_k3", [expert])
        for expert in expert_ids
    ]
    assert arms[-1] == ("selected_qsrt_k3", "qsrt_k3", None)

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

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import qsrt.glm52_expert_intervention_runtime as intervention_runtime
from qsrt.glm52_expert_intervention_runtime import (
    CONTROL_SCHEMA,
    DenseExpertSlice,
    FACTORIZED_LOW_RANK_CANDIDATE_MODE,
    MATERIALIZED_LOW_RANK_CANDIDATE_MODE,
    _apply_with_per_expert_exl3_moe,
    _r7_fused_kernel_selection_disabled,
    _capture_layer_input,
    _parse_capture_layers,
    atomic_write_control,
    evaluate_expert_with_factorized_down,
    read_control,
    routed_candidate_delta,
    validate_dense_intervention_artifact,
    validate_intervention_artifact,
    validate_multi_layer_intervention_artifact,
)
from scripts.materialize_glm52_multi_layer_dense_intervention import (
    materialize_multi_layer_intervention,
)
from scripts.materialize_glm52_single_layer_dense_intervention_composition import (
    materialize_single_layer_composition,
)


def _write_bounded_dense_artifact(
    root: Path, *, model_layer: int, experts: list[int]
) -> dict[str, object]:
    expert_root = root / "experts"
    expert_root.mkdir(parents=True)
    records = []
    for expert in experts:
        endpoint = expert_root / f"layer-{model_layer:03d}-expert-{expert:03d}.bin"
        endpoint.write_bytes(f"layer={model_layer};expert={expert}".encode())
        records.append(
            {
                "layer": model_layer,
                "expert": expert,
                "dense_endpoint_file": endpoint.name,
                "dense_endpoint_file_bytes": endpoint.stat().st_size,
                "dense_endpoint_file_sha256": hashlib.sha256(
                    endpoint.read_bytes()
                ).hexdigest(),
            }
        )
    manifest = {
        "kind": "qsrt_glm52_dense_expert_intervention_v1_manifest",
        "candidate": {"tensor_prefix": "candidate"},
        "exl3_endpoint_identity": {"layer": model_layer},
        "panel": {str(model_layer): experts},
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    report = {
        "kind": "qsrt_glm52_dense_expert_intervention_v1",
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "layer": model_layer,
        "expert_count": len(experts),
        "dense_endpoint_bytes": sum(
            int(record["dense_endpoint_file_bytes"]) for record in records
        ),
        "experts": records,
    }
    (root / "report.json").write_text(json.dumps(report))
    return report


def test_capture_layer_list_is_explicit_and_unique() -> None:
    assert _parse_capture_layers(None, default_layer=3) == (3,)
    assert _parse_capture_layers("52,60,63,64", default_layer=3) == (
        52,
        60,
        63,
        64,
    )
    with pytest.raises(ValueError, match="repeats"):
        _parse_capture_layers("60,60", default_layer=3)
    with pytest.raises(ValueError, match="comma-separated"):
        _parse_capture_layers("60, 64", default_layer=3)


@pytest.mark.parametrize("derived_root", [None, False, True, "complete_slices"])
def test_dense_intervention_artifact_declares_its_model_layer(
    tmp_path: Path, derived_root: bool | str | None,
) -> None:
    expert_root = tmp_path / "experts"
    expert_root.mkdir()
    endpoint = expert_root / "layer-060-expert-007.safetensors"
    endpoint.write_bytes(b"bounded endpoint fixture")
    endpoint_sha256 = hashlib.sha256(endpoint.read_bytes()).hexdigest()
    manifest = {
        "kind": "qsrt_glm52_dense_expert_intervention_v1_manifest",
        "candidate": {"tensor_prefix": "qsrt_k3"},
    }
    if derived_root is not None:
        input_identity = {
            "manifest_sha256": "a" * 64,
        }
        if derived_root == "complete_slices":
            manifest["input_artifacts"] = [
                {
                    "experts": [7],
                    "manifest_sha256": "b" * 64,
                    "report_sha256": "c" * 64,
                    "root": "/results/derived-slice",
                }
            ]
        else:
            input_identity["report_sha256"] = "b" * 64
        if derived_root is True or derived_root == "complete_slices":
            input_identity["root"] = "/artifact"
        manifest.update(
            {
                "input_intervention_artifact": input_identity,
                "panel": {"60": [7]},
            }
        )
    else:
        manifest["exl3_endpoint_identity"] = {"layer": 60}
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    record = {
        "layer": 60,
        "expert": 7,
        "dense_endpoint_file": endpoint.name,
        "dense_endpoint_file_bytes": endpoint.stat().st_size,
        "dense_endpoint_file_sha256": endpoint_sha256,
    }
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "kind": "qsrt_glm52_dense_expert_intervention_v1",
                "status": "complete",
                "manifest_sha256": manifest_sha256,
                "layer": 60,
                "expert_count": 1,
                "dense_endpoint_bytes": endpoint.stat().st_size,
                "experts": [record],
            }
        )
    )

    validated = validate_dense_intervention_artifact(tmp_path)

    assert validated["model_layer"] == 60
    assert validated["expert_ids"] == (7,)


def test_dense_intervention_rejects_slice_ancestry_for_other_experts(
    tmp_path: Path,
) -> None:
    expert_root = tmp_path / "experts"
    expert_root.mkdir()
    endpoint = expert_root / "layer-060-expert-007.safetensors"
    endpoint.write_bytes(b"bounded endpoint fixture")
    manifest = {
        "kind": "qsrt_glm52_dense_expert_intervention_v1_manifest",
        "candidate": {"tensor_prefix": "qsrt_k3"},
        "input_intervention_artifact": {
            "manifest_sha256": "a" * 64,
            "root": "/artifact",
        },
        "input_artifacts": [
            {
                "experts": [8],
                "manifest_sha256": "b" * 64,
                "report_sha256": "c" * 64,
                "root": "/results/wrong-derived-slice",
            }
        ],
        "panel": {"60": [7]},
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "kind": "qsrt_glm52_dense_expert_intervention_v1",
                "status": "complete",
                "manifest_sha256": manifest_sha256,
                "layer": 60,
                "expert_count": 1,
                "dense_endpoint_bytes": endpoint.stat().st_size,
                "experts": [
                    {
                        "layer": 60,
                        "expert": 7,
                        "dense_endpoint_file": endpoint.name,
                        "dense_endpoint_file_bytes": endpoint.stat().st_size,
                        "dense_endpoint_file_sha256": hashlib.sha256(
                            endpoint.read_bytes()
                        ).hexdigest(),
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="does not cover the merged experts"):
        validate_dense_intervention_artifact(tmp_path)


def test_multi_layer_intervention_materializes_only_registered_experts(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    layer_60 = results_root / "layer-60-candidates"
    layer_63 = results_root / "layer-63-candidates"
    _write_bounded_dense_artifact(layer_60, model_layer=60, experts=[1, 7])
    _write_bounded_dense_artifact(layer_63, model_layer=63, experts=[2, 8, 9])
    registration = {
        "schema": "qsrt_glm52_multi_layer_intervention_registration",
        "schema_version": 1,
        "frozen_at_utc": "2026-08-19T00:00:00Z",
        "components": [
            {
                "model_layer": 60,
                "source_artifact_name": layer_60.name,
                "expert_ids": [7],
            },
            {
                "model_layer": 63,
                "source_artifact_name": layer_63.name,
                "expert_ids": [8, 2],
            },
        ],
    }
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps(registration))
    destination = tmp_path / "multi-layer-candidate"

    report = materialize_multi_layer_intervention(
        registration_path=registration_path,
        results_root=results_root,
        destination=destination,
    )
    validated = validate_multi_layer_intervention_artifact(destination)
    dispatched = validate_intervention_artifact(destination)

    assert report["model_layers"] == [60, 63]
    assert report["expert_ids_by_layer"] == {"60": [7], "63": [2, 8]}
    assert report["logical_additional_bytes"] == 0
    assert validated["model_layers"] == (60, 63)
    assert validated["expert_ids_by_layer"] == {60: (7,), 63: (2, 8)}
    assert dispatched["artifact_kind"] == "multi_layer"
    assert sorted((destination / "components/layer-060/experts").iterdir()) == [
        destination / "components/layer-060/experts/layer-060-expert-007.bin"
    ]
    assert sorted((destination / "components/layer-063/experts").iterdir()) == [
        destination / "components/layer-063/experts/layer-063-expert-002.bin",
        destination / "components/layer-063/experts/layer-063-expert-008.bin",
    ]


def test_multi_layer_intervention_rejects_component_identity_drift(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    for layer in (60, 63):
        _write_bounded_dense_artifact(
            results_root / f"layer-{layer}", model_layer=layer, experts=[layer]
        )
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(
        json.dumps(
            {
                "schema": "qsrt_glm52_multi_layer_intervention_registration",
                "schema_version": 1,
                "frozen_at_utc": "2026-08-19T00:00:00Z",
                "components": [
                    {
                        "model_layer": layer,
                        "source_artifact_name": f"layer-{layer}",
                        "expert_ids": [layer],
                    }
                    for layer in (60, 63)
                ],
            }
        )
    )
    destination = tmp_path / "multi-layer-candidate"
    materialize_multi_layer_intervention(
        registration_path=registration_path,
        results_root=results_root,
        destination=destination,
    )
    endpoint = next((destination / "components/layer-063/experts").iterdir())
    endpoint.write_bytes(b"changed")

    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        validate_multi_layer_intervention_artifact(destination)


def test_single_layer_composition_copies_disjoint_registered_experts(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    down_refits = results_root / "down-refits"
    low_rank = results_root / "low-rank"
    down_report = _write_bounded_dense_artifact(
        down_refits, model_layer=52, experts=[29, 36]
    )
    _write_bounded_dense_artifact(
        low_rank, model_layer=52, experts=[186, 235]
    )
    selection_result_name = "down-refit-singleton-selection"
    selection_result = results_root / selection_result_name
    selection_result.mkdir()
    selection_report = {
        "status": "complete",
        "measurement_controls": {"passed": True},
        "intervention_artifact": {
            "manifest_sha256": down_report["manifest_sha256"]
        },
        "documents": [
            {
                "candidate_arms": {
                    "expert-36-alone": {
                        "candidate_minus_resident_mean_forward_kld": -0.001
                    }
                }
            }
            for _ in range(16)
        ],
    }
    selection_report_path = selection_result / "report.json"
    selection_report_path.write_text(json.dumps(selection_report))
    selection_report_sha256 = hashlib.sha256(
        selection_report_path.read_bytes()
    ).hexdigest()
    registration = {
        "schema": (
            "qsrt_glm52_single_layer_dense_intervention_"
            "composition_registration"
        ),
        "schema_version": 1,
        "frozen_at_utc": "2026-08-19T02:00:00Z",
        "model_layer": 52,
        "components": [
            {
                "source_artifact_name": down_refits.name,
                "expert_ids": [36],
                "selection_evidence": {
                    "selection_result_name": selection_result_name,
                    "selection_report_sha256": selection_report_sha256,
                    "candidate_arm": "expert-36-alone",
                    "first_eight_mean_candidate_minus_resident_kld": -0.001,
                    "second_eight_mean_candidate_minus_resident_kld": -0.001,
                    "all_document_mean_candidate_minus_resident_kld": -0.001,
                },
            },
            {
                "source_artifact_name": low_rank.name,
                "expert_ids": [186],
                "logical_adapter_bytes": 65_536,
            },
        ],
    }
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps(registration))
    destination = tmp_path / "composed-layer-52-candidate"

    report = materialize_single_layer_composition(
        registration_path=registration_path,
        results_root=results_root,
        destination=destination,
    )
    validated = validate_dense_intervention_artifact(destination)

    assert report["layer"] == 52
    assert report["logical_adapter_bytes"] == 65_536
    assert validated["expert_ids"] == (36, 186)
    assert sorted((destination / "experts").iterdir()) == [
        destination / "experts/layer-052-expert-036.bin",
        destination / "experts/layer-052-expert-186.bin",
    ]


def test_single_layer_composition_rejects_repeated_expert(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    for name in ("first", "second"):
        _write_bounded_dense_artifact(
            results_root / name, model_layer=52, experts=[36]
        )
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(
        json.dumps(
            {
                "schema": (
                    "qsrt_glm52_single_layer_dense_intervention_"
                    "composition_registration"
                ),
                "schema_version": 1,
                "frozen_at_utc": "2026-08-19T02:00:00Z",
                "model_layer": 52,
                "components": [
                    {"source_artifact_name": "first", "expert_ids": [36]},
                    {"source_artifact_name": "second", "expert_ids": [36]},
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="disjoint"):
        materialize_single_layer_composition(
            registration_path=registration_path,
            results_root=results_root,
            destination=tmp_path / "candidate",
        )


def test_per_expert_execution_temporarily_disables_r7_kernel_flags() -> None:
    layer = SimpleNamespace(
        exl3_r7_fused=True,
        exl3_r7_graph=True,
    )
    expected = torch.tensor([3.0])

    def original_apply(
        method: object,
        received_layer: SimpleNamespace,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: object,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del method, x, topk_weights, topk_ids, shared_experts, shared_experts_input
        assert received_layer.exl3_r7_fused is False
        assert received_layer.exl3_r7_graph is False
        return expected

    actual = _apply_with_per_expert_exl3_moe(
        original_apply,
        object(),
        layer,
        torch.ones(1),
        torch.ones(1),
        torch.zeros(1, dtype=torch.int64),
        None,
        None,
    )

    assert actual is expected
    assert layer.exl3_r7_fused is True
    assert layer.exl3_r7_graph is True


def test_per_expert_execution_restores_missing_flags_after_failure() -> None:
    layer = SimpleNamespace()

    def failing_apply(*args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        assert layer.exl3_r7_fused is False
        assert layer.exl3_r7_graph is False
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        _apply_with_per_expert_exl3_moe(
            failing_apply,
            object(),
            layer,
            torch.ones(1),
            torch.ones(1),
            torch.zeros(1, dtype=torch.int64),
            None,
            None,
        )

    assert not hasattr(layer, "exl3_r7_fused")
    assert not hasattr(layer, "exl3_r7_graph")


def test_non_fused_weight_preparation_restores_r7_selection_flags() -> None:
    layer = SimpleNamespace(exl3_r7_graph=True)

    with _r7_fused_kernel_selection_disabled(layer):
        assert layer.exl3_r7_graph is False
        assert layer.exl3_r7_fused is False
        layer.raw_trellis_tensors_retained = True

    assert layer.exl3_r7_graph is True
    assert not hasattr(layer, "exl3_r7_fused")
    assert layer.raw_trellis_tensors_retained is True


def _endpoint(*, base: float, candidate: float) -> DenseExpertSlice:
    gate_base = torch.tensor([[base, 0.0]], dtype=torch.float16)
    up_base = torch.tensor([[0.0, base]], dtype=torch.float16)
    down_base = torch.zeros((6144, 1), dtype=torch.float16)
    down_base[0, 0] = base
    gate_candidate = torch.tensor([[candidate, 0.0]], dtype=torch.float16)
    up_candidate = torch.tensor([[0.0, candidate]], dtype=torch.float16)
    down_candidate = torch.zeros((6144, 1), dtype=torch.float16)
    down_candidate[0, 0] = candidate
    return DenseExpertSlice(
        exl3_gate=gate_base,
        exl3_up=up_base,
        exl3_down=down_base,
        candidate_gate=gate_candidate,
        candidate_up=up_candidate,
        candidate_down=down_candidate,
    )


def test_control_is_atomic_and_identity_bound(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    digest = "a" * 64
    atomic_write_control(
        path,
        mode="candidate",
        artifact_manifest_sha256=digest,
        generation=2,
        capture_enabled=True,
    )

    assert read_control(path, expected_manifest_sha256=digest) == {
        "schema": CONTROL_SCHEMA,
        "mode": "candidate",
        "artifact_manifest_sha256": digest,
        "generation": 2,
        "capture_enabled": True,
        "selected_experts": None,
    }
    with pytest.raises(ValueError, match="identity mismatch"):
        read_control(path, expected_manifest_sha256="b" * 64)

    value = json.loads(path.read_text())
    value["mode"] = "unreviewed_candidate"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="unsupported intervention mode"):
        read_control(path, expected_manifest_sha256=digest)


def test_routed_delta_changes_only_selected_routes() -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)
    topk_ids = torch.tensor([[7, 9], [9, 10]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]], dtype=torch.float32)

    delta = routed_candidate_delta(
        x,
        topk_weights,
        topk_ids,
        expert_slices={7: _endpoint(base=1.0, candidate=2.0)},
    )

    assert tuple(delta.shape) == (2, 6144)
    assert delta[0, 0] != 0.0
    assert torch.count_nonzero(delta[0, 1:]) == 0
    assert torch.count_nonzero(delta[1]) == 0


def test_identical_endpoints_produce_exact_zero_delta() -> None:
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    delta = routed_candidate_delta(
        x,
        torch.tensor([[1.0]], dtype=torch.float32),
        torch.tensor([[7]], dtype=torch.int64),
        expert_slices={7: _endpoint(base=1.0, candidate=1.0)},
    )
    assert torch.count_nonzero(delta) == 0


def test_dense_resident_identity_mode_ignores_the_candidate_endpoint() -> None:
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    delta = routed_candidate_delta(
        x,
        torch.tensor([[1.0]], dtype=torch.float32),
        torch.tensor([[7]], dtype=torch.int64),
        expert_slices={7: _endpoint(base=1.0, candidate=3.0)},
        use_resident_endpoint=True,
    )

    assert torch.count_nonzero(delta) == 0


def test_dense_resident_identity_reuses_one_resident_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def changing_evaluation(
        expert_input: torch.Tensor,
        *,
        gate: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
    ) -> torch.Tensor:
        del gate, up, down
        nonlocal calls
        calls += 1
        return torch.full(
            (expert_input.shape[0], 6144),
            float(calls),
            dtype=torch.float16,
        )

    monkeypatch.setattr(
        intervention_runtime, "evaluate_expert", changing_evaluation
    )
    delta = routed_candidate_delta(
        torch.tensor([[1.0, 2.0]], dtype=torch.float16),
        torch.tensor([[1.0]], dtype=torch.float32),
        torch.tensor([[7]], dtype=torch.int64),
        expert_slices={7: _endpoint(base=1.0, candidate=3.0)},
        use_resident_endpoint=True,
    )

    assert calls == 1
    assert torch.count_nonzero(delta) == 0


def test_factorized_down_path_executes_the_stored_two_gemm_correction() -> None:
    endpoint = _endpoint(base=1.0, candidate=1.5)
    down_base = torch.zeros((6144, 1), dtype=torch.float16)
    factor_a = torch.tensor([[0.5]], dtype=torch.bfloat16)
    factor_b = torch.zeros((6144, 1), dtype=torch.bfloat16)
    factor_b[0, 0] = 0.25
    endpoint = DenseExpertSlice(
        **{
            **endpoint.__dict__,
            "candidate_down_base": down_base,
            "candidate_down_factor_a": factor_a,
            "candidate_down_factor_b": factor_b,
        }
    )
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float16)

    expected_candidate = evaluate_expert_with_factorized_down(
        x,
        gate=endpoint.candidate_gate,
        up=endpoint.candidate_up,
        down_base=down_base,
        factor_a=factor_a,
        factor_b=factor_b,
    )
    resident = intervention_runtime.evaluate_expert(
        x,
        gate=endpoint.exl3_gate,
        up=endpoint.exl3_up,
        down=endpoint.exl3_down,
    )
    actual = routed_candidate_delta(
        x,
        torch.tensor([[1.0]], dtype=torch.float32),
        torch.tensor([[7]], dtype=torch.int64),
        expert_slices={7: endpoint},
        use_factorized_low_rank_down=True,
    )

    torch.testing.assert_close(
        actual,
        expected_candidate.float() - resident.float(),
        rtol=0,
        atol=0,
    )


def test_factorized_down_mode_requires_stored_factors() -> None:
    with pytest.raises(ValueError, match="no factorized low-rank down endpoint"):
        routed_candidate_delta(
            torch.tensor([[1.0, 2.0]], dtype=torch.float16),
            torch.tensor([[1.0]], dtype=torch.float32),
            torch.tensor([[7]], dtype=torch.int64),
            expert_slices={7: _endpoint(base=1.0, candidate=1.5)},
            use_factorized_low_rank_down=True,
        )


def test_load_time_materialized_down_path_uses_the_reconstructed_endpoint() -> None:
    endpoint = _endpoint(base=1.0, candidate=1.5)
    materialized_down = torch.zeros((6144, 1), dtype=torch.float16)
    materialized_down[0, 0] = 0.125
    endpoint = DenseExpertSlice(
        **{
            **endpoint.__dict__,
            "candidate_down_materialized_from_factors": materialized_down,
        }
    )
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    expected_candidate = intervention_runtime.evaluate_expert(
        x,
        gate=endpoint.candidate_gate,
        up=endpoint.candidate_up,
        down=materialized_down,
    )
    resident = intervention_runtime.evaluate_expert(
        x,
        gate=endpoint.exl3_gate,
        up=endpoint.exl3_up,
        down=endpoint.exl3_down,
    )

    actual = routed_candidate_delta(
        x,
        torch.tensor([[1.0]], dtype=torch.float32),
        torch.tensor([[7]], dtype=torch.int64),
        expert_slices={7: endpoint},
        use_materialized_low_rank_down=True,
    )

    torch.testing.assert_close(
        actual,
        expected_candidate.float() - resident.float(),
        rtol=0,
        atol=0,
    )


def test_load_time_materialized_mode_requires_reconstructed_endpoint() -> None:
    with pytest.raises(ValueError, match="no load-time-materialized low-rank"):
        routed_candidate_delta(
            torch.tensor([[1.0, 2.0]], dtype=torch.float16),
            torch.tensor([[1.0]], dtype=torch.float32),
            torch.tensor([[7]], dtype=torch.int64),
            expert_slices={7: _endpoint(base=1.0, candidate=1.5)},
            use_materialized_low_rank_down=True,
        )


def test_control_accepts_the_dense_resident_identity_mode(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    digest = "c" * 64
    atomic_write_control(
        path,
        mode="dense_resident_identity",
        artifact_manifest_sha256=digest,
        generation=4,
        selected_experts=[64, 208],
    )

    value = read_control(path, expected_manifest_sha256=digest)
    assert value["mode"] == "dense_resident_identity"
    assert value["selected_experts"] == [64, 208]


def test_control_accepts_factorized_low_rank_candidate_mode(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    digest = "e" * 64
    atomic_write_control(
        path,
        mode=FACTORIZED_LOW_RANK_CANDIDATE_MODE,
        artifact_manifest_sha256=digest,
        generation=5,
        selected_experts=[103],
    )

    value = read_control(path, expected_manifest_sha256=digest)
    assert value["mode"] == FACTORIZED_LOW_RANK_CANDIDATE_MODE
    assert value["selected_experts"] == [103]


def test_control_accepts_load_time_materialized_low_rank_mode(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    digest = "f" * 64
    atomic_write_control(
        path,
        mode=MATERIALIZED_LOW_RANK_CANDIDATE_MODE,
        artifact_manifest_sha256=digest,
        generation=6,
        selected_experts=[103],
    )

    value = read_control(path, expected_manifest_sha256=digest)
    assert value["mode"] == MATERIALIZED_LOW_RANK_CANDIDATE_MODE
    assert value["selected_experts"] == [103]


@pytest.mark.parametrize(
    "selected_experts",
    [[4, 4], [-1], [256], [True], "4"],
)
def test_control_rejects_invalid_selected_expert_subsets(
    tmp_path: Path, selected_experts: object
) -> None:
    with pytest.raises((TypeError, ValueError), match="selected_experts"):
        atomic_write_control(
            tmp_path / "control.json",
            mode="qsrt_k3",
            artifact_manifest_sha256="d" * 64,
            generation=1,
            selected_experts=selected_experts,  # type: ignore[arg-type]
        )


def test_layer_input_capture_preserves_hidden_routes_and_applied_weights(
    tmp_path: Path,
) -> None:
    path = _capture_layer_input(
        root=tmp_path,
        model_layer=60,
        x=torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4),
        topk_weights=torch.tensor(
            [[[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]], [[0.1, 0.9], [0.5, 0.5], [0.6, 0.4]]]
        ),
        topk_ids=torch.tensor(
            [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
        ),
        generation=3,
        plan_sha256="a" * 64,
    )

    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        assert handle.metadata() == {
            "schema": "qsrt_glm52_layer_input_capture_v1",
            "model_layer": "60",
            "control_generation": "3",
            "corpus_plan_sha256": "a" * 64,
        }
        assert tuple(handle.get_tensor("hidden_states").shape) == (6, 4)
        assert tuple(handle.get_tensor("topk_ids").shape) == (6, 2)
        assert tuple(handle.get_tensor("topk_weights").shape) == (6, 2)
        assert handle.get_tensor("hidden_states").dtype == torch.bfloat16
        assert handle.get_tensor("topk_ids").dtype == torch.int32

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _capture_layer_input(
            root=tmp_path,
            model_layer=60,
            x=torch.ones(1, 4),
            topk_weights=torch.ones(1, 1),
            topk_ids=torch.zeros(1, 1, dtype=torch.int64),
            generation=0,
            plan_sha256="invalid",
        )

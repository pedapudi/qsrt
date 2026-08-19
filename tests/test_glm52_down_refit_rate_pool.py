from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import qsrt.glm52_down_refit_rate_pool as rate_pool
from qsrt.glm52_down_refit_rate_pool import (
    RATE_PRESERVING_PRE_REGISTRATION_SHA256,
    RATE_TUPLES,
    materialize_registered_partial_rate_map,
    select_pooled_rate_allocation,
    validate_down_refit_rate_pool,
    weighted_error_sums,
)
from qsrt.glm52_pilot import _expert_path, atomic_write_json


def _flat_metrics(value: float) -> dict[tuple[int, int, int], float]:
    return {rates: value for rates in RATE_TUPLES}


def test_rate_preserving_pre_registration_hash_is_frozen() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/glm52_layer3_rate_preserving_down_refit_k3_k4_pre_registration.json"
    )

    assert rate_pool.sha256_file(path) == RATE_PRESERVING_PRE_REGISTRATION_SHA256


def test_weighted_error_sums_apply_route_weights_before_squaring() -> None:
    teacher = torch.tensor([[2.0, 1.0], [1.0, -1.0]])
    candidate = torch.tensor([[1.0, 3.0], [3.0, -1.0]])
    route_weights = torch.tensor([0.5, 0.25])

    error, reference = weighted_error_sums(teacher, candidate, route_weights)

    assert error == pytest.approx(1.5)
    assert reference == pytest.approx(1.375)


def test_pooled_allocator_spends_budget_on_the_larger_complete_expert_gain() -> None:
    expert_10 = _flat_metrics(10.0)
    expert_20 = _flat_metrics(10.0)
    expert_10[(4, 3, 3)] = 3.0
    expert_20[(3, 4, 4)] = 1.0

    selected = select_pooled_rate_allocation(
        [10, 20],
        {10: expert_10, 20: expert_20},
        maximum_k4_projection_count=2,
    )

    assert selected["k4_projection_count"] == 2
    assert selected["candidate_selection_weighted_error_sum"] == 11.0
    assert selected["rates_by_expert"] == {
        "10": {"gate_proj": 3, "up_proj": 3, "down_proj": 3},
        "20": {"gate_proj": 3, "up_proj": 4, "down_proj": 4},
    }


def test_pooled_allocator_combines_independent_gains_when_budget_allows() -> None:
    expert_10 = _flat_metrics(10.0)
    expert_20 = _flat_metrics(10.0)
    expert_10[(4, 3, 3)] = 3.0
    expert_20[(3, 4, 4)] = 1.0

    selected = select_pooled_rate_allocation(
        [10, 20],
        {10: expert_10, 20: expert_20},
        maximum_k4_projection_count=3,
    )

    assert selected["k4_projection_count"] == 3
    assert selected["candidate_selection_weighted_error_sum"] == 4.0
    assert selected["rates_by_expert"]["10"]["gate_proj"] == 4
    assert selected["rates_by_expert"]["20"] == {
        "gate_proj": 3,
        "up_proj": 4,
        "down_proj": 4,
    }


def test_pooled_allocator_uses_lexicographic_k3_tie_break() -> None:
    selected = select_pooled_rate_allocation(
        [10, 20],
        {10: _flat_metrics(1.0), 20: _flat_metrics(1.0)},
        maximum_k4_projection_count=6,
    )

    assert selected["k4_projection_count"] == 0
    assert selected["rates_by_expert"] == {
        "10": {"gate_proj": 3, "up_proj": 3, "down_proj": 3},
        "20": {"gate_proj": 3, "up_proj": 3, "down_proj": 3},
    }


def test_pooled_allocator_rejects_incomplete_or_invalid_metrics() -> None:
    complete = _flat_metrics(1.0)
    incomplete = dict(complete)
    incomplete.pop((4, 4, 4))
    with pytest.raises(ValueError, match="all eight rate tuples"):
        select_pooled_rate_allocation(
            [10], {10: incomplete}, maximum_k4_projection_count=3
        )

    invalid = dict(complete)
    invalid[(4, 4, 4)] = math.nan
    with pytest.raises(ValueError, match="finite and nonnegative"):
        select_pooled_rate_allocation(
            [10], {10: invalid}, maximum_k4_projection_count=3
        )


def test_rate_pool_slice_merge_requires_shared_input_artifacts(
    tmp_path, monkeypatch
) -> None:
    identity = {"down_refit": {"root": "/refit", "manifest_sha256": "a" * 64}}
    slices = [
        {
            "manifest": {
                "common": {"frozen_panel_sha256": "b" * 64},
                "input_artifacts": identity,
                "evidence_boundary": "test",
            },
            "root": str(tmp_path / f"slice-{index}"),
            "manifest_sha256": chr(ord("c") + index) * 64,
            "expert_ids": (10 + index,),
            "report": {"experts": []},
        }
        for index in range(2)
    ]
    slices[1]["manifest"]["input_artifacts"] = {
        "down_refit": {"root": "/different", "manifest_sha256": "d" * 64}
    }
    monkeypatch.setattr(
        rate_pool,
        "validate_down_refit_rate_pool",
        lambda path: slices[int(path.name.rsplit("-", 1)[1])],
    )

    with pytest.raises(ValueError, match="disagree on their input artifacts"):
        rate_pool.merge_down_refit_rate_pool_slices(
            inputs=[tmp_path / "slice-0", tmp_path / "slice-1"],
            dest=tmp_path / "merged",
            panel_manifest_path=tmp_path / "panel.json",
            layer=63,
        )


@pytest.mark.parametrize("layer", [3, 63])
def test_rate_pool_validator_closes_tensor_schema_and_candidate_metrics(
    tmp_path, monkeypatch, layer: int
) -> None:
    specs = tuple(
        SimpleNamespace(name=name, source_shape=(2, 2))
        for name in rate_pool.PROJECTION_NAMES
    )
    monkeypatch.setattr(rate_pool, "PROJECTIONS", specs)
    root = tmp_path / "pool"
    (root / "experts").mkdir(parents=True)
    manifest = {
        "kind": f"{rate_pool.DOWN_REFIT_RATE_POOL_KIND}_manifest",
        "common": {"test": True},
        "panel": {str(layer): [10]},
    }
    atomic_write_json(root / "manifest.json", manifest)
    manifest_sha256 = rate_pool._canonical_json_sha256(manifest)
    tensors = {
        f"{prefix}.{spec.name}": torch.full(
            spec.source_shape,
            float(index + 1),
            dtype=torch.float16,
        )
        for index, (prefix, spec) in enumerate(
            (prefix, spec)
            for prefix in ("exl3", "qsrt_k3", "qsrt_k4")
            for spec in specs
        )
    }
    tensor_path = rate_pool._pool_tensor_path(root, layer, 10)
    rate_pool._atomic_save_pool_tensors(tensor_path, tensors)
    candidates = []
    for index, rates in enumerate(RATE_TUPLES, start=1):
        error_sum = float(index)
        candidates.append(
            {
                "key": rate_pool._rate_key(rates),
                "rates": dict(zip(rate_pool.PROJECTION_NAMES, rates, strict=True)),
                "k4_projection_count": sum(rate == 4 for rate in rates),
                "candidate_selection_weighted_error_sum": error_sum,
                "candidate_selection_weighted_reference_sum": 10.0,
                "candidate_selection_weighted_relative_sse": error_sum / 10.0,
            }
        )
    record = {
        "kind": f"{rate_pool.DOWN_REFIT_RATE_POOL_KIND}_expert",
        "complete": True,
        "manifest_sha256": manifest_sha256,
        "layer": layer,
        "expert": 10,
        "tensor_file": tensor_path.name,
        "tensor_file_bytes": tensor_path.stat().st_size,
        "tensor_file_sha256": rate_pool.sha256_file(tensor_path),
        "down_refit_accepted": True,
        "down_target": {"kind": "reconstructed_activation_refit"},
        "tensor_sha256": {
            prefix: {
                spec.name: rate_pool.tensor_sha256(tensors[f"{prefix}.{spec.name}"])
                for spec in specs
            }
            for prefix in ("exl3", "qsrt_k3", "qsrt_k4")
        },
        "candidate_selection_rows": 4,
        "rate_candidates": candidates,
    }
    receipt_path = _expert_path(root, layer, 10)
    atomic_write_json(receipt_path, record)
    record["receipt_sha256"] = rate_pool.sha256_file(receipt_path)
    report = {
        "kind": rate_pool.DOWN_REFIT_RATE_POOL_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "layer": layer,
        "expert_count": 1,
        "panel": {str(layer): [10]},
        "tensor_file_bytes": tensor_path.stat().st_size,
        "accepted_down_refit_count": 1,
        "experts": [record],
    }
    atomic_write_json(root / "report.json", report)

    validated = validate_down_refit_rate_pool(root)

    assert validated["expert_ids"] == (10,)
    assert validated["model_layer"] == layer
    invalid_report = copy.deepcopy(report)
    invalid_report["experts"][0]["rate_candidates"][0][
        "candidate_selection_weighted_relative_sse"
    ] = 9.0
    atomic_write_json(root / "report.json", invalid_report)
    with pytest.raises(ValueError, match="candidate metric mismatch"):
        validate_down_refit_rate_pool(root)


def test_registered_partial_rate_map_charges_unchanged_exl3_experts(
    tmp_path, monkeypatch
) -> None:
    specs = tuple(
        SimpleNamespace(name=name, source_shape=(2, 2))
        for name in rate_pool.PROJECTION_NAMES
    )
    monkeypatch.setattr(rate_pool, "PROJECTIONS", specs)
    root = tmp_path / "pool"
    (root / "experts").mkdir(parents=True)
    identity_a = "a" * 64
    identity_b = "b" * 64
    panel_identity = "c" * 64
    manifest = {
        "kind": f"{rate_pool.DOWN_REFIT_RATE_POOL_KIND}_manifest",
        "common": {
            "frozen_panel_sha256": panel_identity,
            "down_refit_manifest_file_sha256": identity_a,
            "down_refit_report_file_sha256": identity_b,
        },
        "input_artifacts": {
            "down_refit": {
                "root": "/not-required-for-materialization",
                "manifest_sha256": identity_a,
            }
        },
        "panel": {"63": [10, 20]},
    }
    atomic_write_json(root / "manifest.json", manifest)
    manifest_sha256 = rate_pool._canonical_json_sha256(manifest)
    records = []
    for expert in (10, 20):
        tensors = {
            f"{prefix}.{spec.name}": torch.full(
                spec.source_shape,
                float(expert + prefix_index + spec_index),
                dtype=torch.float16,
            )
            for prefix_index, prefix in enumerate(
                ("exl3", "qsrt_k3", "qsrt_k4")
            )
            for spec_index, spec in enumerate(specs)
        }
        tensor_path = rate_pool._pool_tensor_path(root, 63, expert)
        rate_pool._atomic_save_pool_tensors(tensor_path, tensors)
        candidates = []
        for index, rates in enumerate(RATE_TUPLES, start=1):
            candidates.append(
                {
                    "key": rate_pool._rate_key(rates),
                    "rates": dict(zip(rate_pool.PROJECTION_NAMES, rates, strict=True)),
                    "k4_projection_count": sum(rate == 4 for rate in rates),
                    "candidate_selection_weighted_error_sum": float(index),
                    "candidate_selection_weighted_reference_sum": 10.0,
                    "candidate_selection_weighted_relative_sse": float(index) / 10.0,
                }
            )
        record = {
            "kind": f"{rate_pool.DOWN_REFIT_RATE_POOL_KIND}_expert",
            "complete": True,
            "manifest_sha256": manifest_sha256,
            "layer": 63,
            "expert": expert,
            "tensor_file": tensor_path.name,
            "tensor_file_bytes": tensor_path.stat().st_size,
            "tensor_file_sha256": rate_pool.sha256_file(tensor_path),
            "down_refit_accepted": True,
            "down_target": {"kind": "reconstructed_activation_refit"},
            "tensor_sha256": {
                prefix: {
                    spec.name: rate_pool.tensor_sha256(
                        tensors[f"{prefix}.{spec.name}"]
                    )
                    for spec in specs
                }
                for prefix in ("exl3", "qsrt_k3", "qsrt_k4")
            },
            "candidate_selection_rows": 4,
            "rate_candidates": candidates,
        }
        receipt_path = _expert_path(root, 63, expert)
        atomic_write_json(receipt_path, record)
        record["receipt_sha256"] = rate_pool.sha256_file(receipt_path)
        records.append(record)
    report = {
        "kind": rate_pool.DOWN_REFIT_RATE_POOL_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "layer": 63,
        "expert_count": 2,
        "panel": {"63": [10, 20]},
        "tensor_file_bytes": sum(record["tensor_file_bytes"] for record in records),
        "accepted_down_refit_count": 2,
        "experts": records,
    }
    atomic_write_json(root / "report.json", report)
    registration = {
        "schema": rate_pool.REGISTERED_PARTIAL_RATE_MAP_SCHEMA,
        "schema_version": 1,
        "status": "frozen_before_candidate_k4_measurement",
        "model_layer": 63,
        "comparison_panel": {
            "manifest_sha256": panel_identity,
            "expert_order": [10, 20],
        },
        "down_refit_base": {
            "manifest_file_sha256": identity_a,
            "report_file_sha256": identity_b,
        },
        "comparison_exl3_rates": {
            "10": {"gate_proj": 4, "up_proj": 3, "down_proj": 4},
            "20": {"gate_proj": 4, "up_proj": 4, "down_proj": 4},
        },
        "registered_replacements": [
            {
                "expert": 20,
                "candidate_rates": {
                    "gate_proj": 3,
                    "up_proj": 3,
                    "down_proj": 4,
                },
            }
        ],
        "logical_byte_contract": {
            "uniform_k3_panel_bytes": 100,
            "one_projection_bit_increment_bytes": 10,
            "comparison_exl3_panel_bytes": 150,
            "registered_candidate_panel_bytes": 130,
        },
    }
    registration_path = tmp_path / "registration.json"
    atomic_write_json(registration_path, registration)

    materialized = materialize_registered_partial_rate_map(
        rate_pool_root=root,
        registration_path=registration_path,
        dest=tmp_path / "candidate",
    )

    assert materialized["panel"] == {"63": [20]}
    assert materialized["logical_byte_accounting"] == {
        "scope": "complete registered comparison panel",
        "comparison_exl3_rate_sum": 23,
        "registered_candidate_rate_sum": 21,
        "uniform_k3_panel_bytes": 100,
        "one_projection_bit_increment_bytes": 10,
        "comparison_exl3_panel_bytes": 150,
        "registered_candidate_panel_bytes": 130,
        "logical_margin_bytes": 20,
        "serialized_container_gate_passed": False,
    }
    assert materialized["experts"][0]["rates"] == {
        "gate_proj": 3,
        "up_proj": 3,
        "down_proj": 4,
    }

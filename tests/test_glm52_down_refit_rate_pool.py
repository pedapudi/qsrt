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


def test_rate_pool_validator_closes_tensor_schema_and_candidate_metrics(
    tmp_path, monkeypatch
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
        "panel": {"3": [10]},
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
    tensor_path = rate_pool._pool_tensor_path(root, 3, 10)
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
        "layer": 3,
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
    receipt_path = _expert_path(root, 3, 10)
    atomic_write_json(receipt_path, record)
    record["receipt_sha256"] = rate_pool.sha256_file(receipt_path)
    report = {
        "kind": rate_pool.DOWN_REFIT_RATE_POOL_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "layer": 3,
        "expert_count": 1,
        "panel": {"3": [10]},
        "tensor_file_bytes": tensor_path.stat().st_size,
        "accepted_down_refit_count": 1,
        "experts": [record],
    }
    atomic_write_json(root / "report.json", report)

    validated = validate_down_refit_rate_pool(root)

    assert validated["expert_ids"] == (10,)
    invalid_report = copy.deepcopy(report)
    invalid_report["experts"][0]["rate_candidates"][0][
        "candidate_selection_weighted_relative_sse"
    ] = 9.0
    atomic_write_json(root / "report.json", invalid_report)
    with pytest.raises(ValueError, match="candidate metric mismatch"):
        validate_down_refit_rate_pool(root)

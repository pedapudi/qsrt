from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize_qsrt_permutation_sweep import summarize


def _write_result(
    path: Path,
    *,
    policy: str,
    expert: int,
    fit_sse: float,
    confirmation_sse: float,
    capture: str = "/capture",
    tile_fractions: bool = True,
) -> None:
    payload = {
        "complete": True,
        "contract": {
            "capture": capture,
            "sample_cache": "/samples",
            "training_report": "/report",
            "hessians": "/hessians",
            "codebook": "sqg_xor_cheb_t12",
            "codebook_sha256": {
                "rank_t12": "a" * 64,
                "k2_direct_labels": "b" * 64,
                "k3_direct_labels": "c" * 64,
                "k4_direct_labels": "d" * 64,
            },
            "official_revision": "revision",
            "ldlq_tf32": True,
            "w2_hessian": "conditional expert-local identity shrinkage",
            "layer": 24,
            "permutation_policy": policy,
            "allocation_search": {"qsrt_308_tile_fractions": tile_fractions},
            "allocation_coordinates": {
                "optimization_order": "permutation_then_tiles",
                "funding_basis": "post_permutation_encoder_coordinates",
            },
        },
        "results": {
            str(expert): {
                "skipped": False,
                "permutation_sha256": f"{expert:064x}",
                "qsrt_308": {
                    "selected_on_fit": "candidate",
                    "scores": {
                        "candidate": {
                            "fit": {"sse": fit_sse},
                            "confirmation": {"sse": confirmation_sse},
                        }
                    },
                    "serial_validation": {
                        "selected_on_serial_fit": "candidate",
                        "serial_fit_selected": {
                            "fit": {"sse": fit_sse},
                            "confirmation": {"sse": confirmation_sse},
                        },
                    },
                },
            }
        },
    }
    path.write_text(json.dumps(payload))


def test_sweep_selects_on_absolute_fit_and_reports_confirmation(tmp_path: Path) -> None:
    paths = []
    values = {
        (1, "h2_reverse"): (10.0, 11.0),
        (1, "shape"): (9.0, 10.0),
        (2, "h2_reverse"): (20.0, 22.0),
        (2, "shape"): (19.0, 25.0),
    }
    for (expert, policy), (fit, confirmation) in values.items():
        path = tmp_path / f"{expert}-{policy}.json"
        _write_result(
            path,
            policy=policy,
            expert=expert,
            fit_sse=fit,
            confirmation_sse=confirmation,
        )
        paths.append(path)

    result = summarize(paths)
    assert result["pooled_selected_on_fit"] == "shape"
    assert result["pooled"]["shape"]["confirmation_sse"] == 35.0
    assert result["pooled_selected_confirmation_relative_to_baseline"] == pytest.approx(
        35.0 / 33.0 - 1.0
    )

    held_out = summarize(paths, selection_experts=frozenset({(24, 1)}))
    assert held_out["expert_partition"] == "expert_and_document_disjoint"
    assert held_out["pooled_selected_on_fit"] == "shape"
    assert held_out["pooled_selected_confirmation_relative_to_baseline"] == pytest.approx(
        25.0 / 22.0 - 1.0
    )


def test_sweep_uses_isolated_serial_scores_not_batched_proposals(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_result(
        baseline,
        policy="h2_reverse",
        expert=1,
        fit_sse=10.0,
        confirmation_sse=11.0,
    )
    _write_result(
        candidate,
        policy="shape",
        expert=1,
        fit_sse=12.0,
        confirmation_sse=13.0,
    )
    payload = json.loads(candidate.read_text())
    payload["results"]["1"]["qsrt_308"]["scores"]["candidate"] = {
        "fit": {"sse": 1.0},
        "confirmation": {"sse": 1.0},
    }
    candidate.write_text(json.dumps(payload))

    result = summarize([baseline, candidate])
    assert result["pooled_selected_on_fit"] == "h2_reverse"
    assert result["comparison"] == "absolute_fit_selected_serial_reencode_sse"


def test_sweep_rejects_contract_mismatch(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_result(
        first,
        policy="h2_reverse",
        expert=1,
        fit_sse=1.0,
        confirmation_sse=1.0,
    )
    _write_result(
        second,
        policy="shape",
        expert=1,
        fit_sse=1.0,
        confirmation_sse=1.0,
        capture="/different-capture",
    )
    with pytest.raises(ValueError, match="does not match the sweep contract"):
        summarize([first, second])


def test_sweep_rejects_different_tile_search_spaces(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_result(
        first,
        policy="h2_reverse",
        expert=1,
        fit_sse=1.0,
        confirmation_sse=1.0,
    )
    _write_result(
        second,
        policy="shape",
        expert=1,
        fit_sse=1.0,
        confirmation_sse=1.0,
        tile_fractions=False,
    )
    with pytest.raises(ValueError, match="does not match the sweep contract"):
        summarize([first, second])

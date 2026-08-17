from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from qsrt.glm52_pilot import (
    BASELINE_STORED_ERROR_ATOL,
    EXPERTS_PER_LAYER,
    K4_PANEL,
    PANEL,
    PILOT_KIND,
    PROJECTIONS,
    _canonical_json_sha256,
    _load_completed_expert,
    _validate_baseline_header,
    aggregate_records,
    aggregate_uniform_rate_records,
    cluster_bootstrap_ratio,
    metric_terms,
    panel_cells,
    prepare_destination,
    production_codec_microbenchmark_panel,
    select_hashed_experts,
    summary_markdown,
    validate_and_select_panel,
    validate_and_select_rate_panel,
    validate_architecture_configs,
    UNIFORM_RATE_PILOT_SEED,
    UNIFORM_HIGH_RATE_BITS,
    UNIFORM_HIGH_RATE_PILOT_KIND,
    run_fresh_uniform_rate_expert,
    uniform_high_rate_summary_markdown,
)


def _tier_bitmap() -> dict[str, object]:
    result: dict[str, object] = {}
    for layer, selected in PANEL.items():
        rates = [4] * EXPERTS_PER_LAYER
        errors = [0.01 + expert * 1e-6 for expert in range(EXPERTS_PER_LAYER)]
        for expert in selected:
            rates[expert] = 3
        result[str(layer)] = {
            "k": rates,
            "expert_rel_rt_mse": errors,
        }
    return result


def _source_config() -> dict[str, object]:
    return {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "hidden_size": 6144,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 78,
        "first_k_dense_replace": 3,
        "num_nextn_predict_layers": 1,
        "n_shared_experts": 1,
        "hidden_act": "silu",
    }


def _k4_tier_bitmap() -> dict[str, object]:
    result: dict[str, object] = {}
    for layer, selected in K4_PANEL.items():
        rates = [3] * EXPERTS_PER_LAYER
        errors = [0.01 + expert * 1e-6 for expert in range(EXPERTS_PER_LAYER)]
        for expert in selected:
            rates[expert] = 4
        result[str(layer)] = {"k": rates, "expert_rel_rt_mse": errors}
    return result


def _baseline_config() -> dict[str, object]:
    config = _source_config()
    config["hybrid_tr3_tail"] = {
        "format": "exl3-trellis",
        "codebook": "mcg",
        "source_format": "BF16",
        "experts_per_layer": 256,
        "tp": 4,
        "tier_bitmap": "tier_bitmap.json",
        "source_config_sha256": (
            "185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a"
        ),
        "source_index_sha256": (
            "5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e"
        ),
        "k_values": [3, 4],
        "moe_layers": [3, 78],
        "mcg_multiplier": 0xCBAC1FED,
    }
    return config


def _panel_records(*, ratio: float = 0.75) -> list[dict[str, object]]:
    records = []
    for layer, expert in panel_cells():
        projections = {}
        for spec in PROJECTIONS:
            projections[spec.name] = {
                "source_energy": 10.0,
                "baseline_sse": 1.0,
                "sqg_sse": ratio,
            }
        records.append(
            {
                "layer": layer,
                "expert": expert,
                "source_energy": 30.0,
                "baseline_sse": 3.0,
                "sqg_sse": 3.0 * ratio,
                "sqg_over_baseline": ratio,
                "projections": projections,
            }
        )
    return records


def _baseline_header(layer: int, expert: int) -> dict[str, object]:
    header: dict[str, object] = {}
    for spec in PROJECTIONS:
        for rank in range(4):
            prefix = (
                f"model.layers.{layer}.mlp.experts.{expert}."
                f"{spec.name}.rank{rank}"
            )
            shared = (
                f"model.layers.{layer}.mlp.experts.shared_h."
                f"{spec.name}.rank{rank}"
            )
            expert_length = (
                spec.rank_encoder_shape[0]
                if spec.expert_scale == "suh"
                else spec.rank_encoder_shape[1]
            )
            shared_length = (
                spec.rank_encoder_shape[0]
                if spec.shared_scale == "suh"
                else spec.rank_encoder_shape[1]
            )
            header[f"{prefix}.trellis"] = {
                "dtype": "I16",
                "shape": list(spec.rank_trellis_shape),
            }
            header[f"{prefix}.{spec.expert_scale}"] = {
                "dtype": "F16",
                "shape": [expert_length],
            }
            header[f"{prefix}.mcg"] = {"dtype": "I32", "shape": []}
            header[f"{shared}.{spec.shared_scale}"] = {
                "dtype": "F16",
                "shape": [shared_length],
            }
    return header


def test_frozen_panel_is_48_error_blind_k3_experts() -> None:
    tiers = _tier_bitmap()
    assert validate_and_select_panel(tiers) == PANEL
    assert len(panel_cells()) == 48
    assert len({cell for cell in panel_cells()}) == 48

    tiers["3"]["expert_rel_rt_mse"] = list(reversed(tiers["3"]["expert_rel_rt_mse"]))
    assert validate_and_select_panel(tiers) == PANEL


def test_production_microbenchmark_panel_is_depth_spread_and_error_blind() -> None:
    panel = production_codec_microbenchmark_panel()

    assert len(panel_cells(panel)) == 8
    assert min(panel) == min(PANEL)
    assert max(panel) == max(PANEL)
    for layer, experts in panel.items():
        assert experts == PANEL[layer][: len(experts)]

    assert production_codec_microbenchmark_panel(48) == PANEL
    for invalid in (0, 49, True, 2.5):
        with pytest.raises(ValueError, match="integer between 1 and 48"):
            production_codec_microbenchmark_panel(invalid)  # type: ignore[arg-type]


def test_panel_rejects_rate_drift() -> None:
    tiers = _tier_bitmap()
    tiers["3"]["k"][PANEL[3][0]] = 4
    with pytest.raises(ValueError, match="not enough eligible|panel drifted"):
        validate_and_select_panel(tiers)


def test_frozen_k4_panel_is_48_error_blind_real_k4_experts() -> None:
    tiers = _k4_tier_bitmap()
    assert validate_and_select_rate_panel(
        tiers,
        panel=K4_PANEL,
        eligible_rate=4,
        seed=UNIFORM_RATE_PILOT_SEED,
    ) == K4_PANEL
    assert len(panel_cells(K4_PANEL)) == 48


def test_hashed_selector_validates_inputs() -> None:
    assert len(select_hashed_experts(range(10), 3, 4)) == 4
    with pytest.raises(ValueError, match="unique"):
        select_hashed_experts([1, 1], 3, 1)
    with pytest.raises(ValueError, match="not enough"):
        select_hashed_experts([1], 3, 2)
    with pytest.raises(ValueError, match="outside"):
        select_hashed_experts([256], 3, 1)


def test_architecture_inventory_is_model_native_and_pinned() -> None:
    source = _source_config()
    baseline = _baseline_config()
    validate_architecture_configs(source, baseline)

    source["moe_intermediate_size"] = 3072
    with pytest.raises(ValueError, match="moe_intermediate_size"):
        validate_architecture_configs(source, baseline)

    source = _source_config()
    baseline["hybrid_tr3_tail"]["source_index_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source_index_sha256"):
        validate_architecture_configs(source, baseline)


def test_projection_inventory_closes_orientations_and_k3_bytes() -> None:
    for spec in PROJECTIONS:
        assert spec.source_shape == tuple(reversed(spec.encoder_shape))
        rank_weights = spec.rank_encoder_shape[0] * spec.rank_encoder_shape[1]
        trellis_bytes = (
            spec.rank_trellis_shape[0]
            * spec.rank_trellis_shape[1]
            * spec.rank_trellis_shape[2]
            * 2
        )
        assert trellis_bytes * 8 / rank_weights == 3.0
    assert PROJECTIONS[0].concat_dim == 0
    assert PROJECTIONS[1].concat_dim == 0
    assert PROJECTIONS[2].concat_dim == 1


def test_baseline_header_rejects_wrong_trellis_width() -> None:
    header = _baseline_header(3, 239)
    _validate_baseline_header(header, 3, [239])
    key = "model.layers.3.mlp.experts.239.gate_proj.rank0.trellis"
    header[key]["shape"][-1] = 64
    with pytest.raises(ValueError, match="shape"):
        _validate_baseline_header(header, 3, [239])


def test_baseline_header_accepts_model_artifact_k4_width() -> None:
    header = _baseline_header(3, 51)
    for name, entry in header.items():
        if name.endswith(".trellis"):
            entry["shape"][-1] = 64
    _validate_baseline_header(header, 3, [51], bits=4)


def test_metric_terms_use_source_relative_sse() -> None:
    source = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.bfloat16)
    reconstruction = torch.tensor(
        [[0.0, -1.0], [2.0, -5.0]], dtype=torch.float16
    )
    energy, sse = metric_terms(source, reconstruction)
    assert energy == 30.0
    assert sse == 4.0
    assert sse / energy == pytest.approx(4.0 / 30.0)

    with pytest.raises(ValueError, match="shape mismatch"):
        metric_terms(source, reconstruction[:, :1])


def test_layer_cluster_bootstrap_is_reproducible() -> None:
    records = [
        {"layer": 3, "baseline_sse": 10.0, "sqg_sse": 5.0},
        {"layer": 3, "baseline_sse": 10.0, "sqg_sse": 6.0},
        {"layer": 10, "baseline_sse": 10.0, "sqg_sse": 9.0},
        {"layer": 10, "baseline_sse": 10.0, "sqg_sse": 8.0},
    ]
    first = cluster_bootstrap_ratio(records, replicates=100, seed=17)
    second = cluster_bootstrap_ratio(records, replicates=100, seed=17)
    assert first == second
    assert first["layer_clusters"] == 2
    assert first["ratio_ci95"][1] < 1.0


def test_aggregate_report_classifies_clear_lower_distortion() -> None:
    aggregate = aggregate_records(_panel_records(ratio=0.75))
    assert aggregate["classification"] == "clear_lower_distortion"
    assert aggregate["overall"]["expert_wins"] == 48
    assert aggregate["overall"]["sqg_over_baseline"] == pytest.approx(0.75)
    for item in aggregate["by_projection"].values():
        assert item["sqg_over_baseline"] == pytest.approx(0.75)


def test_destination_resume_is_manifest_exact(tmp_path: Path) -> None:
    dest = tmp_path / "pilot"
    manifest = {"kind": f"{PILOT_KIND}_manifest", "value": 1}
    digest = prepare_destination(dest, manifest, resume=False)
    assert digest == _canonical_json_sha256(manifest)
    assert json.loads((dest / "manifest.json").read_text()) == manifest
    assert prepare_destination(dest, manifest, resume=True) == digest
    with pytest.raises(FileExistsError):
        prepare_destination(dest, manifest, resume=False)
    with pytest.raises(ValueError, match="does not match"):
        prepare_destination(dest, {**manifest, "value": 2}, resume=True)


def test_completed_expert_receipt_is_bound_to_manifest(tmp_path: Path) -> None:
    path = tmp_path / "expert.json"
    path.write_text(
        json.dumps(
            {
                "kind": f"{PILOT_KIND}_expert",
                "complete": True,
                "manifest_sha256": "abc",
                "layer": 3,
                "expert": 239,
            }
        )
    )
    assert _load_completed_expert(
        path, layer=3, expert=239, manifest_sha256="abc"
    )["complete"]
    with pytest.raises(ValueError, match="foreign"):
        _load_completed_expert(path, layer=3, expert=239, manifest_sha256="def")


def test_summary_discloses_pilot_limitations() -> None:
    report = {"aggregate": aggregate_records(_panel_records(ratio=0.75))}
    summary = summary_markdown(report)
    assert "clear_lower_distortion" in summary
    assert "does not isolate the codebook" in summary
    assert "end-to-end" in summary
    assert BASELINE_STORED_ERROR_ATOL == 1e-6


def test_uniform_rate_aggregation_uses_the_matched_k4_panel() -> None:
    records = []
    for layer, expert in panel_cells(K4_PANEL):
        rates = {}
        for label, ratio in (("K2", 0.8), ("K4", 0.6)):
            projections = {
                spec.name: {
                    "source_energy": 10.0,
                    "baseline_sse": 1.0,
                    "sqg_sse": ratio,
                }
                for spec in PROJECTIONS
            }
            rates[label] = {
                "source_energy": 30.0,
                "baseline_sse": 3.0,
                "sqg_sse": 3.0 * ratio,
                "projections": projections,
            }
        records.append({"layer": layer, "expert": expert, "rates": rates})
    aggregate = aggregate_uniform_rate_records(records)
    assert aggregate["K2"]["overall"]["sqg_over_baseline"] == pytest.approx(0.8)
    assert aggregate["K4"]["overall"]["sqg_over_baseline"] == pytest.approx(0.6)


def test_uniform_rate_aggregation_accepts_a_bounded_real_weight_panel() -> None:
    panel = {3: (64,)}
    records = [
        {
            "layer": 3,
            "expert": 64,
            "rates": {
                "K3": {
                    "source_energy": 30.0,
                    "baseline_sse": 3.0,
                    "sqg_sse": 2.7,
                    "projections": {
                        spec.name: {
                            "source_energy": 10.0,
                            "baseline_sse": 1.0,
                            "sqg_sse": 0.9,
                        }
                        for spec in PROJECTIONS
                    },
                }
            },
        }
    ]

    aggregate = aggregate_uniform_rate_records(
        records, rate_labels=("K3",), panel=panel
    )

    assert aggregate["K3"]["overall"]["expert_count"] == 1
    assert aggregate["K3"]["overall"]["sqg_over_baseline"] == pytest.approx(0.9)


def test_uniform_high_rate_aggregation_reuses_the_matched_panel() -> None:
    records = []
    for layer, expert in panel_cells(K4_PANEL):
        rates = {}
        for label, ratio in (("K5", 0.7), ("K6", 0.9)):
            projections = {
                spec.name: {
                    "source_energy": 10.0,
                    "baseline_sse": 1.0,
                    "sqg_sse": ratio,
                }
                for spec in PROJECTIONS
            }
            rates[label] = {
                "source_energy": 30.0,
                "baseline_sse": 3.0,
                "sqg_sse": 3.0 * ratio,
                "projections": projections,
            }
        records.append({"layer": layer, "expert": expert, "rates": rates})
    aggregate = aggregate_uniform_rate_records(
        records, rate_labels=("K5", "K6")
    )
    assert aggregate["K5"]["overall"]["sqg_over_baseline"] == pytest.approx(0.7)
    assert aggregate["K6"]["overall"]["sqg_over_baseline"] == pytest.approx(0.9)
    report = {"aggregate": aggregate}
    summary = uniform_high_rate_summary_markdown(report)
    assert "uniform-K5/K6" in summary
    assert "freshly encoded" in summary
    assert UNIFORM_HIGH_RATE_BITS == (5, 6)
    assert UNIFORM_HIGH_RATE_PILOT_KIND.endswith("uniform_k5_k6_codec_pilot_v1")


def test_fresh_uniform_expert_rejects_invalid_rate_ladder_before_io() -> None:
    with pytest.raises(ValueError, match="K2 through K6"):
        run_fresh_uniform_rate_expert(
            source=None,
            layer=3,
            expert=51,
            bits=(6, 5),
            device=torch.device("cpu"),
            quantizer_module=None,
            manifest_sha256="test",
        )

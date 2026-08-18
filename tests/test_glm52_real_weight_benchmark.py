from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from qsrt.glm52_real_weight_benchmark import (
    R7_MARKER,
    R7_RECIPE_VERSION,
    R7_SCHEMA_VERSION,
    _normalized_panel,
    _r7_projection_key,
    load_frozen_real_weight_panel,
    select_frozen_panel_slice,
    select_rate_pattern_stratified_experts,
    validate_r7_rate_map,
)
from qsrt.glm52_pilot import EXPERTS_PER_LAYER, PROJECTIONS


def _rate_patterns() -> dict[int, tuple[int, int, int]]:
    patterns = (
        [(3, 3, 4)] * 98
        + [(3, 3, 3)] * 71
        + [(4, 4, 4)] * 58
        + [(4, 4, 5)] * 27
        + [(3, 3, 5)]
        + [(3, 4, 4)]
    )
    assert len(patterns) == EXPERTS_PER_LAYER
    return dict(enumerate(patterns))


def _sidecar() -> dict[str, object]:
    layer = 3
    patterns = _rate_patterns()
    bit_map = {
        _r7_projection_key(layer, expert, spec.name): patterns[expert][index]
        for expert in range(EXPERTS_PER_LAYER)
        for index, spec in enumerate(PROJECTIONS)
    }
    return {
        "marker": R7_MARKER,
        "schema_version": R7_SCHEMA_VERSION,
        "recipe_version": R7_RECIPE_VERSION,
        "layer": layer,
        "allocation_bit_units": 2688,
        "allocation_target_bpw": 3.5,
        "shard": "r7-experts-layer-003.safetensors",
        "bit_map": bit_map,
    }


def test_r7_rate_map_closes_complete_layer_and_allocation() -> None:
    sidecar = _sidecar()
    sidecar["allocation_target_bpw"] = "3.5"
    patterns = validate_r7_rate_map(sidecar, layer=3)

    assert patterns == _rate_patterns()
    assert Counter(patterns.values()) == Counter(
        {
            (3, 3, 4): 98,
            (3, 3, 3): 71,
            (4, 4, 4): 58,
            (4, 4, 5): 27,
            (3, 3, 5): 1,
            (3, 4, 4): 1,
        }
    )
    assert sum(sum(pattern) for pattern in patterns.values()) == 2688


def test_r7_rate_map_rejects_missing_projection_and_budget_drift() -> None:
    sidecar = _sidecar()
    bit_map = sidecar["bit_map"]
    assert isinstance(bit_map, dict)
    bit_map.pop(_r7_projection_key(3, 0, "gate_proj"))
    with pytest.raises(ValueError, match="missing"):
        validate_r7_rate_map(sidecar, layer=3)

    sidecar = _sidecar()
    bit_map = sidecar["bit_map"]
    assert isinstance(bit_map, dict)
    bit_map[_r7_projection_key(3, 0, "gate_proj")] = 4
    with pytest.raises(ValueError, match="3.5 bpw"):
        validate_r7_rate_map(sidecar, layer=3)


def test_rate_pattern_selection_covers_all_six_layer_patterns_at_eight_experts() -> None:
    patterns = _rate_patterns()
    selected = select_rate_pattern_stratified_experts(
        patterns, layer=3, expert_count=8
    )

    assert len(selected) == len(set(selected)) == 8
    assert set(patterns[expert] for expert in selected) == set(patterns.values())
    assert selected == select_rate_pattern_stratified_experts(
        patterns, layer=3, expert_count=8
    )


def test_rate_pattern_selection_validates_count_and_complete_population() -> None:
    patterns = _rate_patterns()
    for invalid in (0, 257, True, 1.5):
        with pytest.raises(ValueError, match="integer between 1 and 256"):
            select_rate_pattern_stratified_experts(
                patterns, layer=3, expert_count=invalid  # type: ignore[arg-type]
            )
    patterns.pop(255)
    with pytest.raises(ValueError, match="cover expert IDs"):
        select_rate_pattern_stratified_experts(
            patterns, layer=3, expert_count=8
        )


def test_panel_normalization_rejects_non_routed_layers_and_duplicate_experts() -> None:
    assert _normalized_panel({3: [7, 9]}) == {3: (7, 9)}
    with pytest.raises(ValueError, match="outside 3..77"):
        _normalized_panel({2: [0]})
    with pytest.raises(ValueError, match="repeats"):
        _normalized_panel({3: [7, 7]})


def test_committed_real_weight_panel_is_frozen_before_candidate_measurement() -> None:
    path = Path("experiments/glm52_layer3_rate_pattern_panel.json")
    panel = load_frozen_real_weight_panel(path, layer=3)

    assert panel["experts"] == (64, 208, 106, 204, 89, 212, 96, 103)
    assert panel["rate_patterns"] == {
        64: (3, 3, 4),
        208: (3, 3, 3),
        106: (4, 4, 4),
        204: (4, 4, 5),
        89: (3, 3, 5),
        212: (3, 4, 4),
        96: (3, 3, 4),
        103: (3, 3, 3),
    }
    assert set(panel["rate_patterns"].values()) == set(_rate_patterns().values())


def test_frozen_panel_slices_are_disjoint_and_ordered() -> None:
    panel = load_frozen_real_weight_panel(
        Path("experiments/glm52_layer3_rate_pattern_panel.json"), layer=3
    )
    slices = [
        select_frozen_panel_slice(panel, offset=offset, expert_count=2)
        for offset in (0, 2, 4, 6)
    ]

    assert slices == [
        (64, 208),
        (106, 204),
        (89, 212),
        (96, 103),
    ]
    assert tuple(expert for group in slices for expert in group) == panel["experts"]


@pytest.mark.parametrize(
    "layer,expected_experts",
    [
        (52, (254, 186, 96, 116, 68, 29, 36, 235)),
        (60, (78, 125, 186, 28, 230, 136, 180, 142)),
        (63, (215, 32, 123, 164, 199, 118, 149, 29)),
        (64, (241, 253, 76, 90, 85, 155, 106, 210)),
    ],
)
def test_high_impact_and_nearby_control_panels_are_error_blind(
    layer: int, expected_experts: tuple[int, ...]
) -> None:
    panel = load_frozen_real_weight_panel(
        Path(f"experiments/glm52_layer{layer}_rate_pattern_panel.json"),
        layer=layer,
    )

    assert panel["experts"] == expected_experts
    assert len(set(panel["rate_patterns"].values())) >= 4


@pytest.mark.parametrize("offset,count", [(-1, 1), (0, 0), (7, 2), (8, 1)])
def test_frozen_panel_slice_rejects_invalid_bounds(offset: int, count: int) -> None:
    panel = load_frozen_real_weight_panel(
        Path("experiments/glm52_layer3_rate_pattern_panel.json"), layer=3
    )

    with pytest.raises(ValueError):
        select_frozen_panel_slice(panel, offset=offset, expert_count=count)

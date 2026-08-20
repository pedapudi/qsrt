import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_PATH = (
    ROOT
    / "experiments"
    / "glm52_layer_shared_routed_aggregate_corrector_pre_registration.json"
)
STRATEGY_PATH = ROOT / "docs" / "qsrt-improvement-strategy.md"
DESIGN_PATH = (
    ROOT / "docs" / "glm52-layer-shared-routed-aggregate-corrector-design.md"
)


def _registration() -> dict:
    return json.loads(REGISTRATION_PATH.read_text())


def test_corrector_parameter_and_byte_counts_close() -> None:
    registration = _registration()
    architecture = registration["architecture"]
    rank = architecture["rank"]
    hidden_width = registration["student_model"]["hidden_width"]
    experts = registration["student_model"]["routed_experts_per_layer"]

    values_per_layer = 2 * hidden_width * rank + experts * rank
    assert architecture["trainable_values_per_layer"] == values_per_layer
    assert architecture["logical_bf16_bytes_per_layer"] == 2 * values_per_layer
    assert architecture["logical_bf16_bytes_for_23_layers"] == (
        23 * 2 * values_per_layer
    )


def test_corrector_gates_and_route_strata_are_frozen() -> None:
    registration = _registration()
    basis_gate = registration["shared_basis_gate"]
    assert basis_gate["shared_basis_rank"] == 16
    assert basis_gate["required_shared_recovery_fraction"] == 0.8
    assert basis_gate["warm_start_required_ceiling_fraction"] == 0.8

    route_strata = registration["route_report"]["mutually_exclusive_strata"]
    assert len(route_strata) == 4
    assert len(set(route_strata)) == 4
    assert all("current corrected-layer route" in value for value in route_strata)


def test_corrector_registration_preserves_data_and_serving_boundaries() -> None:
    registration = _registration()
    assert registration["source_model"]["complete_bf16_checkpoint_required"] is False
    assert registration["correction_boundary"]["shared_expert_changed"] is False
    assert registration["correction_boundary"]["current_layer_route_changed_by_this_boundary"] is False
    assert "bypass" in registration["serving_contract"]["disabled_path"] or (
        "bit-identical" in registration["serving_contract"]["disabled_path"]
    )
    assert len(registration["execution_order"]) == 7
    assert len(registration["escalation_rules"]) == 5


def test_strategy_links_the_corrector_registration() -> None:
    strategy = STRATEGY_PATH.read_text()
    assert REGISTRATION_PATH.name in strategy
    assert DESIGN_PATH.name in strategy
    assert "80%" in strategy
    assert "shared-basis" in strategy
    assert (
        "A small group's point estimate cannot support a headline conclusion."
        in strategy
        or "A sparse stratum remains descriptive." in strategy
    )


def test_design_document_records_academic_precedents_and_limits() -> None:
    registration = _registration()
    assert registration["design_document"] == str(DESIGN_PATH.relative_to(ROOT))

    design = DESIGN_PATH.read_text()
    for expected_source in (
        "Side-Tuning",
        "Ladder Side-Tuning",
        "Parameter-Efficient Transfer Learning for NLP",
        "ControlNet",
        "CALDERA",
        "LQER",
        "QERA",
        "LoftQ",
    ):
        assert expected_source in design
    assert "None validates\nthe complete GLM-5.2 construction." in design

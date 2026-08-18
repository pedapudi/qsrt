"""Model-wide expert-static rotation plan for coupled QSRT experts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from qsrt import constants as C
from qsrt.qsrt_coupled import CoupledHadamardSpec


KIND = "qsrt_coupled_rotation_plan"
LEGACY_KIND = "qsrt_k2_coupled_rotation_plan"
SCHEMA_VERSION = 1
PRODUCTION_DRAW_CANDIDATES = (0, 6)
PRODUCTION_SELECTION = "c128_fit_propose_confirmation_total_sse_accept"
POOLED_ALL_ROW_SELECTION = "pooled_all_routed_rows_total_post_projection_sse"


@dataclass(frozen=True)
class CoupledDrawSelection:
    """One expert's coupled-rotation decision and its support evidence."""

    evaluated_draws: tuple[int, ...]
    proposed_draw: int
    selected_draw: int
    fit_documents: int
    confirmation_documents: int
    confirmation_relative_improvement: float | None
    accepted: bool
    reason: str

    def to_json(self) -> dict[str, object]:
        return {
            "evaluated_draws": list(self.evaluated_draws),
            "proposed_draw": self.proposed_draw,
            "selected_draw": self.selected_draw,
            "fit_documents": self.fit_documents,
            "confirmation_documents": self.confirmation_documents,
            "confirmation_relative_improvement": (
                self.confirmation_relative_improvement
            ),
            "accepted": self.accepted,
            "reason": self.reason,
        }


def select_coupled_draw(
    draw_candidates: Sequence[int],
    fit_sse: Mapping[int, float],
    confirmation_sse: Mapping[int, float],
    *,
    fit_documents: int,
    confirmation_documents: int,
    min_fit_documents: int,
    min_confirmation_documents: int,
    minimum_improvement: float,
) -> CoupledDrawSelection:
    """Select one draw without using confirmation to search alternatives.

    The fit fold proposes one draw from the frozen portfolio.  The confirmation
    fold can only accept that proposal against draw zero; it never chooses among
    alternatives.  This prevents the confirmation fold from becoming a draw
    oracle while still rejecting expert-local regressions.
    """

    draws = tuple(int(draw) for draw in draw_candidates)
    if not draws or draws[0] != 0 or len(set(draws)) != len(draws):
        raise ValueError("coupled draw candidates must be unique and begin with zero")
    if any(not 0 <= draw < 8 for draw in draws):
        raise ValueError("coupled draw candidates must lie in 0..7")
    if set(fit_sse) != set(draws) or set(confirmation_sse) != set(draws):
        raise ValueError("coupled draw SSE maps must cover exactly the candidates")
    if min(fit_documents, confirmation_documents) < 0:
        raise ValueError("coupled draw document counts must be nonnegative")
    if min(min_fit_documents, min_confirmation_documents) < 1:
        raise ValueError("coupled draw support thresholds must be positive")
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0:
        raise ValueError("coupled draw minimum improvement must be finite and nonnegative")
    for name, values in (("fit", fit_sse), ("confirmation", confirmation_sse)):
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in values.values()):
            raise ValueError(f"coupled draw {name} SSE must be finite and nonnegative")

    proposed = 0
    selected = 0
    improvement: float | None = None
    accepted = False
    if fit_documents < min_fit_documents:
        reason = "insufficient_fit_document_support"
    else:
        proposed = min(draws, key=lambda draw: (float(fit_sse[draw]), draw))
        if proposed == 0:
            reason = "draw_zero_best_on_fit"
        elif confirmation_documents < min_confirmation_documents:
            reason = "insufficient_confirmation_document_support"
        else:
            baseline = float(confirmation_sse[0])
            candidate = float(confirmation_sse[proposed])
            improvement = 0.0 if baseline == 0.0 else (baseline - candidate) / baseline
            if candidate < baseline and improvement > minimum_improvement:
                selected = proposed
                accepted = True
                reason = "confirmation_total_sse_improved"
            else:
                reason = "confirmation_total_sse_did_not_improve"
    return CoupledDrawSelection(
        evaluated_draws=draws,
        proposed_draw=proposed,
        selected_draw=selected,
        fit_documents=fit_documents,
        confirmation_documents=confirmation_documents,
        confirmation_relative_improvement=improvement,
        accepted=accepted,
        reason=reason,
    )


@dataclass(frozen=True)
class CoupledRotationPlan:
    """One three-bit intermediate rotation draw per routed expert."""

    draws_by_layer: Mapping[int, tuple[int, ...]]
    selection: str

    def __post_init__(self) -> None:
        if not self.selection or not self.selection.isascii():
            raise ValueError("coupled rotation selection must be nonempty ASCII")
        expected_layers = set(C.MOE_LAYERS)
        if set(self.draws_by_layer) != expected_layers:
            raise ValueError("coupled rotation plan must cover every MoE layer")
        for layer, draws in self.draws_by_layer.items():
            if len(draws) != C.NUM_EXPERTS or any(not 0 <= draw < 8 for draw in draws):
                raise ValueError(
                    f"layer {layer} must contain {C.NUM_EXPERTS} draws in 0..7"
                )

    def for_layer(self, layer: int) -> tuple[int, ...]:
        try:
            return self.draws_by_layer[layer]
        except KeyError as exc:
            raise ValueError(f"rotation plan has no layer {layer}") from exc

    def for_experts(self, layer: int, experts: list[int]) -> dict[int, int]:
        draws = self.for_layer(layer)
        if len(set(experts)) != len(experts) or any(
            not 0 <= expert < C.NUM_EXPERTS for expert in experts
        ):
            raise ValueError("expert list is invalid")
        return {expert: draws[expert] for expert in experts}

    def to_json(self) -> dict[str, object]:
        profile = CoupledHadamardSpec()
        return {
            "kind": KIND,
            "schema_version": SCHEMA_VERSION,
            "residual_draw": 0,
            "draw_family_size": 8,
            "residual_block_size": profile.residual_block_size,
            "preactivation_block_size": profile.preactivation_block_size,
            "postactivation_block_size": profile.postactivation_block_size,
            "selection": self.selection,
            "layers": {
                str(layer): list(self.draws_by_layer[layer])
                for layer in C.MOE_LAYERS
            },
        }

    @classmethod
    def from_json(cls, value: object) -> "CoupledRotationPlan":
        if not isinstance(value, dict):
            raise TypeError("coupled rotation plan must be a JSON object")
        if value.get("kind") not in (KIND, LEGACY_KIND) or value.get(
            "schema_version"
        ) != SCHEMA_VERSION:
            raise ValueError("unsupported coupled rotation plan schema")
        profile = CoupledHadamardSpec()
        expected_profile = {
            "residual_draw": profile.residual_draw,
            "draw_family_size": 8,
            "residual_block_size": profile.residual_block_size,
            "preactivation_block_size": profile.preactivation_block_size,
            "postactivation_block_size": profile.postactivation_block_size,
        }
        if any(value.get(name) != expected for name, expected in expected_profile.items()):
            raise ValueError("coupled rotation plan changes the frozen draw family")
        selection = value.get("selection")
        if not isinstance(selection, str):
            raise ValueError("coupled rotation plan is missing its selection policy")
        layers = value.get("layers")
        if not isinstance(layers, dict):
            raise ValueError("coupled rotation plan is missing layers")
        parsed: dict[int, tuple[int, ...]] = {}
        for key, draws in layers.items():
            try:
                layer = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError("coupled rotation layer IDs must be integers") from exc
            if not isinstance(draws, list) or any(
                isinstance(draw, bool) or not isinstance(draw, int) for draw in draws
            ):
                raise ValueError(f"layer {key} draws must be integer lists")
            parsed[layer] = tuple(draws)
        return cls(parsed, selection)


def load_coupled_rotation_plan(path: str | Path) -> CoupledRotationPlan:
    return CoupledRotationPlan.from_json(json.loads(Path(path).read_text()))


__all__ = [
    "CoupledDrawSelection",
    "CoupledRotationPlan",
    "KIND",
    "PRODUCTION_DRAW_CANDIDATES",
    "POOLED_ALL_ROW_SELECTION",
    "PRODUCTION_SELECTION",
    "SCHEMA_VERSION",
    "load_coupled_rotation_plan",
    "select_coupled_draw",
]

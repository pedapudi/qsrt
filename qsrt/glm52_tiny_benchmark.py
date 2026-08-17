"""CPU-only mechanism benchmark for two-bit GLM-5.2 expert quantization.

The synthetic expert retains GLM-5.2's complete SwiGLU dataflow while reducing
its dimensions to one input, two hidden coordinates, and two outputs.  Its
eight weights consume sixteen K2 branch bits at two bits per weight.  The
fourteen-bit K2 history therefore puts at most thirty logical bits in play.
Experts run sequentially, so an expert-count sweep does not increase that
bound.

Fit rows may select trellis paths, reciprocal up/down balance factors, and a
down-matrix refit.  Independently generated held-out rows only report transfer.
This benchmark can falsify a quantization mechanism cheaply.  It cannot prove
full-model KLD, comparison-checkpoint dominance, or serialized model size.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Iterable

import numpy as np

from qsrt.tiny_improvement_benchmark import (
    SQG_XOR_CHEB_T12_SHA256,
    _decode_e4m3,
    _round_bf16,
    _sqg_edge_rank_for_rate,
    sqg_xor_cheb_t12_bytes,
)


DEFAULT_EXPERT_COUNT = 8
MAXIMUM_EXPERT_COUNT = 256
DEFAULT_FIT_ROWS = 48
DEFAULT_HELDOUT_ROWS = 48
INPUT_DIMENSION = 1
HIDDEN_COORDINATES = 2
OUTPUT_DIMENSION = 2
SOURCE_WEIGHT_COUNT = 8
PAYLOAD_BITS = SOURCE_WEIGHT_COUNT * 2
SCALAR_K2_HISTORY_BITS = 14
BITS_AT_PLAY = PAYLOAD_BITS + SCALAR_K2_HISTORY_BITS
BIT_LIMIT = 32
RIDGE_STRENGTH = 0.05
DEFAULT_GAUGE_VALUES = (0.5, 2.0**-0.5, 1.0, 2.0**0.5, 2.0)
SOURCE_FAMILIES = ("mixed", "gaussian", "heavy_tail", "saturated")
STAGE_NAMES = (
    "scalar_k2_weight_error_control",
    "reciprocal_up_down_balance",
    "fit_kld_selected_scalar_paths",
    "reciprocal_balance_with_fit_kld_paths",
    "frozen_upstream_down_refit",
)

# The four rows sum to zero and span both expert-output directions.  A common
# shift cannot hide an error, as it could if the two outputs were used directly
# as two logits.
PSEUDO_VOCABULARY_READOUT = np.asarray(
    ((1.0, 0.25), (-0.25, 1.0), (-1.0, -0.25), (0.25, -1.0)),
    dtype=np.float64,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Synthetic source and selection settings shared by an expert sweep."""

    fit_rows: int = DEFAULT_FIT_ROWS
    heldout_rows: int = DEFAULT_HELDOUT_ROWS
    source_family: str = "mixed"
    gate_up_correlation: float = 0.6
    tail_degrees_of_freedom: float = 3.5
    weight_scale: float = 1.0
    input_scale: float = 1.0
    gauge_values: tuple[float, ...] = DEFAULT_GAUGE_VALUES

    def validate(self) -> None:
        if self.fit_rows <= 0 or self.heldout_rows <= 0:
            raise ValueError("fit_rows and heldout_rows must be positive")
        if self.source_family not in SOURCE_FAMILIES:
            raise ValueError(
                f"source_family must be one of {', '.join(SOURCE_FAMILIES)}"
            )
        if not -0.99 < self.gate_up_correlation < 0.99:
            raise ValueError("gate_up_correlation must be between -0.99 and 0.99")
        if self.tail_degrees_of_freedom <= 2.0:
            raise ValueError("tail_degrees_of_freedom must exceed 2")
        if self.weight_scale <= 0.0 or self.input_scale <= 0.0:
            raise ValueError("weight_scale and input_scale must be positive")
        if not self.gauge_values or any(value <= 0.0 for value in self.gauge_values):
            raise ValueError("gauge_values must contain positive values")
        if not any(np.isclose(value, 1.0) for value in self.gauge_values):
            raise ValueError("gauge_values must contain 1.0 as the identity control")


@dataclass(frozen=True)
class TinyGlmExpert:
    """One full-precision SwiGLU expert and its disjoint input rows."""

    source_seed: int
    source_family: str
    source_gate: np.ndarray
    source_up: np.ndarray
    source_down: np.ndarray
    fit_inputs: np.ndarray
    heldout_inputs: np.ndarray
    fit_source_output: np.ndarray
    heldout_source_output: np.ndarray


@dataclass(frozen=True)
class ScalarPathSet:
    """All sixteen closed, two-step K2 reconstructions for one scalar stream."""

    reconstructions: np.ndarray
    branches: tuple[tuple[int, int], ...]
    states: tuple[tuple[int, int], ...]
    ranks: tuple[tuple[int, int], ...]
    table_indices: tuple[tuple[int, int], ...]
    scale: float


@dataclass(frozen=True)
class EncodedExpert:
    """One candidate expert and the fit-only decisions that produced it."""

    gauge: np.ndarray
    transformed_source_gate: np.ndarray
    transformed_source_up: np.ndarray
    transformed_source_down: np.ndarray
    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray
    gate_path: dict[str, Any]
    up_path: dict[str, Any]
    down_paths: tuple[dict[str, Any], ...]
    selection_objective: str
    fit_hidden: np.ndarray
    heldout_hidden: np.ndarray
    metrics: dict[str, float]


def silu(values: np.ndarray) -> np.ndarray:
    """Evaluate the coordinatewise SiLU used by a GLM-5.2 SwiGLU expert."""

    values = np.asarray(values, dtype=np.float64)
    return values / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def expert_hidden(inputs: np.ndarray, gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Return ``SiLU(W_gate x) * (W_up x)`` for the two hidden coordinates."""

    return silu(np.asarray(inputs) @ np.asarray(gate).reshape(1, 2)) * (
        np.asarray(inputs) @ np.asarray(up).reshape(1, 2)
    )


def expert_output(
    inputs: np.ndarray, gate: np.ndarray, up: np.ndarray, down: np.ndarray
) -> np.ndarray:
    """Evaluate the complete one-input, two-hidden, two-output SwiGLU expert."""

    return expert_hidden(inputs, gate, up) @ np.asarray(down).T


def apply_reciprocal_balance(
    gate: np.ndarray, up: np.ndarray, down: np.ndarray, gauge: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply an exact up/down reciprocal balance without decoder metadata."""

    gauge = np.asarray(gauge, dtype=np.float64)
    if gauge.shape != (HIDDEN_COORDINATES,) or np.any(gauge <= 0.0):
        raise ValueError("gauge must contain two positive hidden-coordinate values")
    return (
        np.asarray(gate, dtype=np.float64).copy(),
        np.asarray(up, dtype=np.float64) * gauge,
        np.asarray(down, dtype=np.float64) / gauge[None, :],
    )


def pseudo_vocabulary_logits(outputs: np.ndarray) -> np.ndarray:
    """Project both expert-output directions into four centered logits."""

    return np.asarray(outputs, dtype=np.float64) @ PSEUDO_VOCABULARY_READOUT.T


def forward_kld(teacher_outputs: np.ndarray, candidate_outputs: np.ndarray) -> float:
    """Return teacher-to-candidate KLD after the fixed four-logit readout."""

    teacher_logits = pseudo_vocabulary_logits(teacher_outputs)
    candidate_logits = pseudo_vocabulary_logits(candidate_outputs)
    teacher_logits -= teacher_logits.max(axis=-1, keepdims=True)
    candidate_logits -= candidate_logits.max(axis=-1, keepdims=True)
    teacher_log_probability = teacher_logits - np.log(
        np.exp(teacher_logits).sum(axis=-1, keepdims=True)
    )
    candidate_log_probability = candidate_logits - np.log(
        np.exp(candidate_logits).sum(axis=-1, keepdims=True)
    )
    teacher_probability = np.exp(teacher_log_probability)
    divergence = np.sum(
        teacher_probability * (teacher_log_probability - candidate_log_probability),
        axis=-1,
    ).mean()
    return max(0.0, float(divergence))


def _candidate_forward_klds(
    teacher_outputs: np.ndarray, candidate_outputs: np.ndarray
) -> np.ndarray:
    """Vectorize fit KLD over candidate axes following the row axis."""

    teacher_logits = pseudo_vocabulary_logits(teacher_outputs)
    candidate_logits = np.einsum(
        "r...o,vo->r...v", candidate_outputs, PSEUDO_VOCABULARY_READOUT
    )
    teacher_logits -= teacher_logits.max(axis=-1, keepdims=True)
    candidate_logits -= candidate_logits.max(axis=-1, keepdims=True)
    teacher_log_probability = teacher_logits - np.log(
        np.exp(teacher_logits).sum(axis=-1, keepdims=True)
    )
    candidate_log_probability = candidate_logits - np.log(
        np.exp(candidate_logits).sum(axis=-1, keepdims=True)
    )
    extra_axes = (None,) * (candidate_outputs.ndim - teacher_outputs.ndim)
    expanded_teacher_log_probability = teacher_log_probability[
        (slice(None),) + extra_axes + (slice(None),)
    ]
    expanded_teacher_probability = np.exp(expanded_teacher_log_probability)
    divergence = np.sum(
        expanded_teacher_probability
        * (expanded_teacher_log_probability - candidate_log_probability),
        axis=-1,
    ).mean(axis=0)
    return np.maximum(0.0, divergence)


def _mean_squared_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.square(np.asarray(candidate) - np.asarray(reference)).mean())


def _bf16_expert_output(
    inputs: np.ndarray, gate: np.ndarray, up: np.ndarray, down: np.ndarray
) -> np.ndarray:
    """Replay BF16 operand boundaries with FP32 matrix multiplication."""

    bf16_inputs = _round_bf16(inputs)
    gate_projection = _round_bf16(bf16_inputs @ _round_bf16(gate).reshape(1, 2))
    up_projection = _round_bf16(bf16_inputs @ _round_bf16(up).reshape(1, 2))
    hidden = _round_bf16(silu(gate_projection) * up_projection)
    return _round_bf16(hidden @ _round_bf16(down).T)


@lru_cache(maxsize=1)
def _normalized_scalar_k2_paths() -> ScalarPathSet:
    """Build the exact production T12 reconstructions for all K2 closed paths."""

    levels = _decode_e4m3(sqg_xor_cheb_t12_bytes())
    branches_out: list[tuple[int, int]] = []
    states_out: list[tuple[int, int]] = []
    ranks_out: list[tuple[int, int]] = []
    indices_out: list[tuple[int, int]] = []
    reconstructions: list[np.ndarray] = []
    mask = (1 << SCALAR_K2_HISTORY_BITS) - 1
    for branch_pair in itertools.product(range(4), repeat=2):
        history_symbols = [branch_pair[index % 2] for index in range(1, 8)]
        state = 0
        for symbol in history_symbols:
            state = ((state << 2) | symbol) & mask
        initial_state = state
        path_states: list[int] = []
        path_ranks: list[int] = []
        for branch in branch_pair:
            path_states.append(state)
            path_ranks.append(_sqg_edge_rank_for_rate(state, branch, 2))
            state = ((state << 2) | branch) & mask
        if state != initial_state:
            raise AssertionError("periodic scalar K2 path did not close")
        indices = tuple(rank >> 4 for rank in path_ranks)
        branches_out.append(branch_pair)
        states_out.append(tuple(path_states))
        ranks_out.append(tuple(path_ranks))
        indices_out.append(indices)
        reconstructions.append(levels[np.asarray(indices)])
    return ScalarPathSet(
        reconstructions=np.stack(reconstructions),
        branches=tuple(branches_out),
        states=tuple(states_out),
        ranks=tuple(ranks_out),
        table_indices=tuple(indices_out),
        scale=1.0,
    )


def scalar_k2_paths(source: np.ndarray) -> ScalarPathSet:
    """Scale all exact T12 K2 paths to one two-weight source stream."""

    source = np.asarray(source, dtype=np.float64)
    if source.shape != (2,):
        raise ValueError("a tiny scalar K2 stream must contain exactly two weights")
    scale = max(float(np.max(np.abs(source))) / 5.5, 1e-12)
    normalized = _normalized_scalar_k2_paths()
    return replace(
        normalized,
        reconstructions=normalized.reconstructions * scale,
        scale=scale,
    )


def _coefficient_path_index(source: np.ndarray, paths: ScalarPathSet) -> int:
    errors = np.square(paths.reconstructions - np.asarray(source)[None, :]).sum(axis=1)
    return int(np.argmin(errors))


def _path_report(paths: ScalarPathSet, index: int, source: np.ndarray) -> dict[str, Any]:
    return {
        "branches": list(paths.branches[index]),
        "states": list(paths.states[index]),
        "edge_ranks": list(paths.ranks[index]),
        "table_indices": list(paths.table_indices[index]),
        "scale": paths.scale,
        "coefficient_squared_error": float(
            np.square(paths.reconstructions[index] - np.asarray(source)).sum()
        ),
    }


def _draw_unit_variance(
    generator: np.random.Generator,
    shape: tuple[int, ...],
    family: str,
    tail_degrees_of_freedom: float,
) -> np.ndarray:
    if family == "heavy_tail":
        correction = np.sqrt((tail_degrees_of_freedom - 2.0) / tail_degrees_of_freedom)
        return generator.standard_t(tail_degrees_of_freedom, shape) * correction
    return generator.normal(0.0, 1.0, shape)


def _resolved_source_family(requested: str, expert_index: int) -> str:
    if requested != "mixed":
        return requested
    return ("gaussian", "heavy_tail", "saturated")[expert_index % 3]


def make_problem(
    source_seed: int,
    *,
    expert_index: int = 0,
    config: BenchmarkConfig | None = None,
) -> TinyGlmExpert:
    """Create one synthetic GLM expert and independent fit and held-out rows."""

    config = config or BenchmarkConfig()
    config.validate()
    family = _resolved_source_family(config.source_family, expert_index)
    source_generator = np.random.default_rng(10_000_019 + source_seed)
    gate = _draw_unit_variance(
        source_generator, (2,), family, config.tail_degrees_of_freedom
    )
    innovation = _draw_unit_variance(
        source_generator, (2,), family, config.tail_degrees_of_freedom
    )
    up = (
        config.gate_up_correlation * gate
        + np.sqrt(1.0 - config.gate_up_correlation**2) * innovation
    )
    down = _draw_unit_variance(
        source_generator, (2, 2), family, config.tail_degrees_of_freedom
    )
    family_weight_scale = 2.75 if family == "saturated" else 1.0
    expert_weight_scale = (0.65, 1.0, 1.55)[expert_index % 3]
    source_gate = gate * config.weight_scale * family_weight_scale * expert_weight_scale
    source_up = up * config.weight_scale * expert_weight_scale
    source_down = down * config.weight_scale / np.sqrt(2.0)

    fit_generator = np.random.default_rng(20_000_033 + 2 * source_seed)
    heldout_generator = np.random.default_rng(30_000_047 + 2 * source_seed)
    input_family = "heavy_tail" if family == "heavy_tail" else "gaussian"
    family_input_scale = 1.8 if family == "saturated" else 1.0
    fit_inputs = _draw_unit_variance(
        fit_generator,
        (config.fit_rows, 1),
        input_family,
        config.tail_degrees_of_freedom,
    )
    heldout_inputs = _draw_unit_variance(
        heldout_generator,
        (config.heldout_rows, 1),
        input_family,
        config.tail_degrees_of_freedom,
    )
    fit_inputs *= config.input_scale * family_input_scale
    heldout_inputs *= config.input_scale * family_input_scale
    return TinyGlmExpert(
        source_seed=source_seed,
        source_family=family,
        source_gate=source_gate,
        source_up=source_up,
        source_down=source_down,
        fit_inputs=fit_inputs,
        heldout_inputs=heldout_inputs,
        fit_source_output=expert_output(
            fit_inputs, source_gate, source_up, source_down
        ),
        heldout_source_output=expert_output(
            heldout_inputs, source_gate, source_up, source_down
        ),
    )


def _evaluate_encoded(
    problem: TinyGlmExpert,
    *,
    gauge: np.ndarray,
    transformed_gate: np.ndarray,
    transformed_up: np.ndarray,
    transformed_down: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    gate_path: dict[str, Any],
    up_path: dict[str, Any],
    down_paths: tuple[dict[str, Any], ...],
    selection_objective: str,
) -> EncodedExpert:
    fit_hidden = expert_hidden(problem.fit_inputs, gate, up)
    heldout_hidden = expert_hidden(problem.heldout_inputs, gate, up)
    fit_output = fit_hidden @ down.T
    heldout_output = heldout_hidden @ down.T
    source_bf16 = _bf16_expert_output(
        problem.heldout_inputs,
        problem.source_gate,
        problem.source_up,
        problem.source_down,
    )
    candidate_bf16 = _bf16_expert_output(
        problem.heldout_inputs, gate, up, down
    )
    metrics = {
        "transformed_coefficient_squared_error": float(
            np.square(gate - transformed_gate).sum()
            + np.square(up - transformed_up).sum()
            + np.square(down - transformed_down).sum()
        ),
        "fit_output_mean_squared_error": _mean_squared_error(
            fit_output, problem.fit_source_output
        ),
        "heldout_output_mean_squared_error": _mean_squared_error(
            heldout_output, problem.heldout_source_output
        ),
        "fit_forward_kld": forward_kld(problem.fit_source_output, fit_output),
        "heldout_forward_kld": forward_kld(
            problem.heldout_source_output, heldout_output
        ),
        "heldout_bf16_output_mean_squared_error": _mean_squared_error(
            candidate_bf16, source_bf16
        ),
    }
    return EncodedExpert(
        gauge=np.asarray(gauge),
        transformed_source_gate=transformed_gate,
        transformed_source_up=transformed_up,
        transformed_source_down=transformed_down,
        gate=gate,
        up=up,
        down=down,
        gate_path=gate_path,
        up_path=up_path,
        down_paths=down_paths,
        selection_objective=selection_objective,
        fit_hidden=fit_hidden,
        heldout_hidden=heldout_hidden,
        metrics=metrics,
    )


def _encode_at_gauge(
    problem: TinyGlmExpert,
    gauge: np.ndarray,
    *,
    upstream_objective: str,
) -> EncodedExpert:
    transformed_gate, transformed_up, transformed_down = apply_reciprocal_balance(
        problem.source_gate, problem.source_up, problem.source_down, gauge
    )
    gate_paths = scalar_k2_paths(transformed_gate)
    up_paths = scalar_k2_paths(transformed_up)
    down_path_sets = tuple(scalar_k2_paths(row) for row in transformed_down)
    down_indices = tuple(
        _coefficient_path_index(row, paths)
        for row, paths in zip(transformed_down, down_path_sets)
    )
    down = np.stack(
        [paths.reconstructions[index] for paths, index in zip(down_path_sets, down_indices)]
    )
    if upstream_objective == "weight_error":
        gate_index = _coefficient_path_index(transformed_gate, gate_paths)
        up_index = _coefficient_path_index(transformed_up, up_paths)
    elif upstream_objective == "fit_forward_kld":
        fit_gate = (
            problem.fit_inputs[:, 0, None, None]
            * gate_paths.reconstructions[None, :, :]
        )
        fit_up = (
            problem.fit_inputs[:, 0, None, None]
            * up_paths.reconstructions[None, :, :]
        )
        hidden = silu(fit_gate[:, :, None, :]) * fit_up[:, None, :, :]
        outputs = np.einsum("rguh,oh->rguo", hidden, down)
        klds = _candidate_forward_klds(problem.fit_source_output, outputs)
        flat_index = int(np.argmin(klds))
        gate_index, up_index = np.unravel_index(flat_index, klds.shape)
    else:
        raise ValueError(f"unsupported upstream objective: {upstream_objective}")
    return _evaluate_encoded(
        problem,
        gauge=gauge,
        transformed_gate=transformed_gate,
        transformed_up=transformed_up,
        transformed_down=transformed_down,
        gate=gate_paths.reconstructions[gate_index],
        up=up_paths.reconstructions[up_index],
        down=down,
        gate_path=_path_report(gate_paths, gate_index, transformed_gate),
        up_path=_path_report(up_paths, up_index, transformed_up),
        down_paths=tuple(
            _path_report(paths, index, source)
            for paths, index, source in zip(
                down_path_sets, down_indices, transformed_down
            )
        ),
        selection_objective=upstream_objective,
    )


def _gauge_grid(values: Iterable[float]) -> tuple[np.ndarray, ...]:
    unique = tuple(dict.fromkeys(float(value) for value in values))
    return tuple(
        np.asarray(pair, dtype=np.float64)
        for pair in itertools.product(unique, repeat=HIDDEN_COORDINATES)
    )


def _best_fit_candidate(candidates: Iterable[EncodedExpert]) -> EncodedExpert:
    return min(
        candidates,
        key=lambda candidate: (
            candidate.metrics["fit_forward_kld"],
            candidate.metrics["fit_output_mean_squared_error"],
            tuple(candidate.gauge.tolist()),
            tuple(candidate.gate_path["branches"]),
            tuple(candidate.up_path["branches"]),
        ),
    )


def _refit_down_after_upstream_freezes(
    problem: TinyGlmExpert, upstream: EncodedExpert
) -> tuple[EncodedExpert, dict[str, Any]]:
    hidden = upstream.fit_hidden
    gram = hidden.T @ hidden + RIDGE_STRENGTH * np.eye(2, dtype=np.float64)
    rhs = (
        hidden.T @ problem.fit_source_output
        + RIDGE_STRENGTH * upstream.transformed_source_down.T
    )
    continuous_target = np.linalg.solve(gram, rhs).T
    options: list[EncodedExpert] = []
    for multiplier in np.linspace(0.8, 1.2, 9):
        path_sets: list[ScalarPathSet] = []
        indices: list[int] = []
        rows: list[np.ndarray] = []
        for row in continuous_target:
            nominal = scalar_k2_paths(row)
            paths = replace(
                nominal,
                reconstructions=nominal.reconstructions * float(multiplier),
                scale=nominal.scale * float(multiplier),
            )
            index = _coefficient_path_index(row, paths)
            path_sets.append(paths)
            indices.append(index)
            rows.append(paths.reconstructions[index])
        down = np.stack(rows)
        options.append(
            _evaluate_encoded(
                problem,
                gauge=upstream.gauge,
                transformed_gate=upstream.transformed_source_gate,
                transformed_up=upstream.transformed_source_up,
                transformed_down=continuous_target,
                gate=upstream.gate,
                up=upstream.up,
                down=down,
                gate_path=upstream.gate_path,
                up_path=upstream.up_path,
                down_paths=tuple(
                    _path_report(paths, index, row)
                    for paths, index, row in zip(path_sets, indices, continuous_target)
                ),
                selection_objective="fit_only_down_refit_after_upstream_freeze",
            )
        )
    candidate = _best_fit_candidate(options)
    tolerance = max(
        1e-15, abs(upstream.metrics["fit_forward_kld"]) * 1e-12
    )
    accepted = (
        candidate.metrics["fit_forward_kld"]
        < upstream.metrics["fit_forward_kld"] - tolerance
    )
    return (candidate if accepted else upstream), {
        "accepted_on_fit": accepted,
        "upstream_gate_path_frozen": candidate.gate_path == upstream.gate_path,
        "upstream_up_path_frozen": candidate.up_path == upstream.up_path,
        "gauge_frozen": bool(np.array_equal(candidate.gauge, upstream.gauge)),
        "continuous_down_target": continuous_target.tolist(),
        "candidate_metrics": candidate.metrics,
    }


def _relative_reduction(before: float, after: float) -> float:
    return 0.0 if before == 0.0 else float(1.0 - after / before)


def _change_status(before: float, after: float) -> str:
    tolerance = max(abs(before), abs(after)) * 1e-12 + 1e-18
    if after < before - tolerance:
        return "improved"
    if after > before + tolerance:
        return "regressed"
    return "unchanged"


def _encoded_report(
    encoded: EncodedExpert, comparison: EncodedExpert | None
) -> dict[str, Any]:
    report: dict[str, Any] = {
        **encoded.metrics,
        "selection_objective": encoded.selection_objective,
        "reciprocal_gauge": encoded.gauge.tolist(),
        "identity_gauge_selected": bool(np.allclose(encoded.gauge, 1.0)),
        "runtime_metadata_bits_for_gauge": 0,
        "decoded_gate": encoded.gate.tolist(),
        "decoded_up": encoded.up.tolist(),
        "decoded_down": encoded.down.tolist(),
        "gate_k2_path": encoded.gate_path,
        "up_k2_path": encoded.up_path,
        "down_k2_paths": list(encoded.down_paths),
    }
    if comparison is not None:
        for metric in (
            "heldout_output_mean_squared_error",
            "heldout_forward_kld",
            "heldout_bf16_output_mean_squared_error",
        ):
            before = comparison.metrics[metric]
            after = encoded.metrics[metric]
            report[f"{metric}_reduction_from_comparison"] = _relative_reduction(
                before, after
            )
            report[f"{metric}_status"] = _change_status(before, after)
    return report


def run_problem(
    problem: TinyGlmExpert, *, config: BenchmarkConfig | None = None
) -> dict[str, Any]:
    """Evaluate all fit-selected mechanisms on one supplied expert problem."""

    config = config or BenchmarkConfig(
        fit_rows=problem.fit_inputs.shape[0],
        heldout_rows=problem.heldout_inputs.shape[0],
    )
    config.validate()
    identity = np.ones(2, dtype=np.float64)
    control = _encode_at_gauge(problem, identity, upstream_objective="weight_error")
    reciprocal = _best_fit_candidate(
        _encode_at_gauge(problem, gauge, upstream_objective="weight_error")
        for gauge in _gauge_grid(config.gauge_values)
    )
    fit_kld_paths = _encode_at_gauge(
        problem, identity, upstream_objective="fit_forward_kld"
    )
    composed = _best_fit_candidate(
        _encode_at_gauge(problem, gauge, upstream_objective="fit_forward_kld")
        for gauge in _gauge_grid(config.gauge_values)
    )
    refitted, refit_details = _refit_down_after_upstream_freezes(problem, composed)
    candidates = {
        STAGE_NAMES[0]: control,
        STAGE_NAMES[1]: reciprocal,
        STAGE_NAMES[2]: fit_kld_paths,
        STAGE_NAMES[3]: composed,
        STAGE_NAMES[4]: refitted,
    }
    comparison_names = {
        STAGE_NAMES[0]: None,
        STAGE_NAMES[1]: STAGE_NAMES[0],
        STAGE_NAMES[2]: STAGE_NAMES[0],
        STAGE_NAMES[3]: STAGE_NAMES[0],
        STAGE_NAMES[4]: STAGE_NAMES[3],
    }
    stage_reports: dict[str, Any] = {}
    for name, encoded in candidates.items():
        comparison_name = comparison_names[name]
        stage_reports[name] = _encoded_report(
            encoded, candidates[comparison_name] if comparison_name else None
        )
        stage_reports[name]["comparison_stage"] = comparison_name
    stage_reports[STAGE_NAMES[4]]["down_refit"] = refit_details
    return {
        "source_seed": problem.source_seed,
        "source_family": problem.source_family,
        "fit_and_heldout_rows_share_memory": bool(
            np.shares_memory(problem.fit_inputs, problem.heldout_inputs)
        ),
        "source": {
            "gate": problem.source_gate.tolist(),
            "up": problem.source_up.tolist(),
            "down": problem.source_down.tolist(),
        },
        "stages": stage_reports,
    }


def _transition_summary(
    expert_reports: list[dict[str, Any]], stage: str, metric: str
) -> dict[str, Any]:
    comparison = expert_reports[0]["stages"][stage]["comparison_stage"]
    if comparison is None:
        raise ValueError("the control stage has no transition")
    before = np.asarray(
        [report["stages"][comparison][metric] for report in expert_reports]
    )
    after = np.asarray([report["stages"][stage][metric] for report in expert_reports])
    statuses = [_change_status(float(a), float(b)) for a, b in zip(before, after)]
    return {
        "comparison_stage": comparison,
        "pooled_error_reduction": _relative_reduction(float(before.sum()), float(after.sum())),
        "median_expert_error_reduction": float(
            np.median(
                np.divide(
                    before - after,
                    before,
                    out=np.zeros_like(before),
                    where=before != 0,
                )
            )
        ),
        "improved_experts": statuses.count("improved"),
        "unchanged_experts": statuses.count("unchanged"),
        "regressed_experts": statuses.count("regressed"),
    }


def run_sweep(
    expert_count: int = DEFAULT_EXPERT_COUNT,
    *,
    start_seed: int = 0,
    config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Run one to 256 synthetic experts sequentially and preserve regressions."""

    if isinstance(expert_count, bool) or not isinstance(expert_count, int):
        raise ValueError("expert_count must be an integer")
    if not 1 <= expert_count <= MAXIMUM_EXPERT_COUNT:
        raise ValueError(
            f"expert_count must be between 1 and {MAXIMUM_EXPERT_COUNT}"
        )
    if BITS_AT_PLAY > BIT_LIMIT:
        raise AssertionError(
            f"tiny GLM benchmark uses {BITS_AT_PLAY} bits, limit is {BIT_LIMIT}"
        )
    config = config or BenchmarkConfig()
    config.validate()
    expert_reports = [
        run_problem(
            make_problem(start_seed + index, expert_index=index, config=config),
            config=config,
        )
        for index in range(expert_count)
    ]
    transition_stages = STAGE_NAMES[1:]
    metrics = (
        "fit_output_mean_squared_error",
        "heldout_output_mean_squared_error",
        "fit_forward_kld",
        "heldout_forward_kld",
        "heldout_bf16_output_mean_squared_error",
    )
    metric_sums = {
        stage: {
            metric: float(
                sum(report["stages"][stage][metric] for report in expert_reports)
            )
            for metric in metrics
        }
        for stage in STAGE_NAMES
    }
    return {
        "status": "research-only synthetic GLM-5.2 mechanism benchmark",
        "kld_evidence_boundary": (
            "KLD is synthetic teacher-to-candidate forward KLD after a fixed "
            "centered full-rank readout. It is not full-model or EXL3 KLD and "
            "cannot establish checkpoint dominance."
        ),
        "size_evidence_boundary": (
            "The bit bound counts payload plus live K2 history. It excludes "
            "tables, scales, metadata, padding, and non-expert tensors, so it "
            "cannot establish serialized model size."
        ),
        "expert_count": expert_count,
        "start_seed": start_seed,
        "sequential_execution": True,
        "selection_rows": "fit rows only",
        "reporting_rows": "independently generated held-out rows",
        "stage_names": list(STAGE_NAMES),
        "stage_explanations": {
            STAGE_NAMES[0]: (
                "Choose every scalar K2 path only by its two reconstructed "
                "weights' squared error."
            ),
            STAGE_NAMES[1]: (
                "Search exact reciprocal up/down balance factors on fit rows, "
                "then retain weight-error-selected K2 paths."
            ),
            STAGE_NAMES[2]: (
                "Keep the original full-precision parameterization and choose "
                "the gate and up K2 paths by fit-row forward KLD."
            ),
            STAGE_NAMES[3]: (
                "Jointly select reciprocal balance factors and gate/up K2 paths "
                "by fit-row forward KLD."
            ),
            STAGE_NAMES[4]: (
                "Freeze the selected gauge and upstream paths, fit a continuous "
                "down matrix on fit rows, and encode each down row with K2."
            ),
        },
        "configuration": {
            "fit_rows": config.fit_rows,
            "heldout_rows": config.heldout_rows,
            "source_family": config.source_family,
            "gate_up_correlation": config.gate_up_correlation,
            "tail_degrees_of_freedom": config.tail_degrees_of_freedom,
            "weight_scale": config.weight_scale,
            "input_scale": config.input_scale,
            "gauge_values": list(config.gauge_values),
        },
        "bit_budget": {
            "source_weight_count": SOURCE_WEIGHT_COUNT,
            "payload_bits": PAYLOAD_BITS,
            "payload_bits_per_weight": 2.0,
            "scalar_k2_history_bits": SCALAR_K2_HISTORY_BITS,
            "bits_at_play": BITS_AT_PLAY,
            "limit": BIT_LIMIT,
            "aggregate_payload_bits": expert_count * PAYLOAD_BITS,
        },
        "problem": {
            "expert_equation": "down(SiLU(gate(input)) * up(input))",
            "input_dimensions": INPUT_DIMENSION,
            "hidden_coordinates": HIDDEN_COORDINATES,
            "output_dimensions": OUTPUT_DIMENSION,
            "gate_weights": 2,
            "up_weights": 2,
            "down_weights": 4,
            "k2_closed_paths_per_two_weight_stream": 16,
            "scalar_reconstruction": "exact frozen sqg_xor_cheb_t12 K2 table",
            "scalar_reconstruction_sha256": SQG_XOR_CHEB_T12_SHA256,
            "pseudo_vocabulary_readout": PSEUDO_VOCABULARY_READOUT.tolist(),
            "reciprocal_balance": (
                "W_up' = D W_up and W_down' = W_down D^-1; D is folded into "
                "decoded weights and requires no runtime metadata"
            ),
            "omitted_real_system_components": [
                "the 2,048-coordinate hidden axis and 128-channel coding records",
                "BlockLDLQ feedback and captured activation covariances",
                "official BF16 or FP8 source decoding and FP8 block scales",
                "top-8 routing, the shared expert, and downstream transformer layers",
                "complete-checkpoint byte accounting",
            ],
        },
        "stage_metric_sums": metric_sums,
        "heldout_transitions": {
            stage: _transition_summary(
                expert_reports, stage, "heldout_output_mean_squared_error"
            )
            for stage in transition_stages
        },
        "heldout_forward_kld_transitions": {
            stage: _transition_summary(
                expert_reports, stage, "heldout_forward_kld"
            )
            for stage in transition_stages
        },
        "experts": expert_reports,
    }


def run_benchmark(
    source_seed: int = 0, *, config: BenchmarkConfig | None = None
) -> dict[str, Any]:
    """Run one expert while retaining the sweep report schema."""

    return run_sweep(1, start_seed=source_seed, config=config)


__all__ = [
    "BITS_AT_PLAY",
    "BIT_LIMIT",
    "BenchmarkConfig",
    "DEFAULT_EXPERT_COUNT",
    "MAXIMUM_EXPERT_COUNT",
    "PAYLOAD_BITS",
    "PSEUDO_VOCABULARY_READOUT",
    "SOURCE_FAMILIES",
    "SOURCE_WEIGHT_COUNT",
    "STAGE_NAMES",
    "TinyGlmExpert",
    "apply_reciprocal_balance",
    "expert_hidden",
    "expert_output",
    "forward_kld",
    "make_problem",
    "run_benchmark",
    "run_problem",
    "run_sweep",
    "scalar_k2_paths",
    "silu",
]

"""A bounded CPU falsification benchmark for three two-bit QSRT proposals.

Each synthetic expert has one input coordinate, two SiTU neurons, and two
output coordinates.  Its two aligned gate/up symbols use four branch bits
apiece, and its 2-by-2 down matrix uses two bits per coefficient.  Together
with the pair trellis's twelve live history bits, the pair proposal has
twenty-eight logical bits at play.  The matched-payload scalar K2 control has
fourteen live history bits and therefore sets the overall peak at thirty.
Experts run sequentially, so increasing ``expert_count`` does not increase it.

The benchmark separates fit rows from held-out rows.  Fit
rows choose the output-aware path and decide whether to accept a quantized
down-matrix refit.  Held-out rows only report transfer.  A stage may regress;
the report preserves that result instead of forcing monotonic error.

This NumPy-only benchmark is small enough to run under CPython or Pyodide.  It
tests whether the proposed mechanisms can survive basic transfer and geometry
checks.  It does not predict Kimi-K3 model quality.
"""

from __future__ import annotations

import base64
import hashlib
import itertools
import zlib
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


CALIBRATION_SEED = 20260815
SOURCE_SEED = 1
CALIBRATION_PAIR_COUNT = 65_536
FIT_ROW_COUNT = 48
HELDOUT_ROW_COUNT = 48
INPUT_DIMENSION = 1
INTERMEDIATE_NEURONS = 2
OUTPUT_DIMENSION = 2
PAIR_STEPS = INPUT_DIMENSION * INTERMEDIATE_NEURONS
PAIR_BRANCH_BITS = 4
PAIR_BRANCHES = 1 << PAIR_BRANCH_BITS
DOWN_WEIGHT_COUNT = INTERMEDIATE_NEURONS * OUTPUT_DIMENSION
DOWN_WEIGHT_BITS = 2
TRELLIS_STATE_BITS = 12
PAYLOAD_BITS = PAIR_STEPS * PAIR_BRANCH_BITS + DOWN_WEIGHT_COUNT * DOWN_WEIGHT_BITS
PAIR_BITS_AT_PLAY = PAYLOAD_BITS + TRELLIS_STATE_BITS
SCALAR_K2_STATE_BITS = 14
SCALAR_K2_BITS_AT_PLAY = PAYLOAD_BITS + SCALAR_K2_STATE_BITS
BITS_AT_PLAY = max(PAIR_BITS_AT_PLAY, SCALAR_K2_BITS_AT_PLAY)
BIT_LIMIT = 32

RIDGE_STRENGTH = 0.05
_HISTORY_MASK = (1 << TRELLIS_STATE_BITS) - 1
_WORD_MASK = (1 << (TRELLIS_STATE_BITS + PAIR_BRANCH_BITS)) - 1
SQG_XOR_CHEB_T12_SHA256 = "cca11fe5744c9c93a34f4217f342fbc0f74ecc8a007c076582424a505fc9da5e"
_SQG_XOR_CHEB_T12_B85 = (
    "c-p<eg;D}R002-B5Gld#?(XhZ{3;NY2D`hvyIb+6++*&Bn{}9l+dba<b-8}rzCFHJKL7df>fIM_{(B|hg)h&fJ>hvI;enic^6vO@%eR|=xF+t3qf4$XxI4#ihT{~^37%s=9ibc|9H8u@?4j%;?VxOXvjy9PZNS!HYp_+=3S^mWiLppqV9e9z7_+n)#<bfMFiDvJ#(^<llrjQ1olb|tZnxR2R;$HgHk(XF<M8m%(BPoKU>F$C>vcMTKqw3f5{blOu|yIaER}|Y$YgT4LZMVDRjSZXwOXUmgaP4{2p|%O0-`A~K&)FFBc7JPNTek(l4&W7RJJroIxGX03Cn_I!*XD`-sGX=BNd<&q7<PNBb1<&`l$?0Ii3m}l^CkHtLCbPqgvwX0-~O84SZ=NuZf&y5?b)IlGetTb^<#5*XfHcue$%~Vd?egbJJ@3{{ZxHPbL"
)
PSEUDO_VOCABULARY_READOUT = np.array(
    ((1.0, 0.25), (-0.25, 1.0), (-1.0, -0.25), (0.25, -1.0)),
    dtype=np.float64,
)


@dataclass(frozen=True)
class TinyProblem:
    """Full-precision expert, disjoint rows, and reconstruction tables."""

    source_pairs: np.ndarray
    source_down: np.ndarray
    fit_inputs: np.ndarray
    heldout_inputs: np.ndarray
    fit_source_output: np.ndarray
    heldout_source_output: np.ndarray
    trained_pair_table: np.ndarray
    upstream_scales: np.ndarray
    down_scales: np.ndarray
    source_down_k2_paths: tuple[dict[str, Any], ...]
    quantized_source_down: np.ndarray


@dataclass(frozen=True)
class PathChoice:
    """One reconstruction choice and its fit and held-out losses."""

    branches: tuple[int, ...]
    states: tuple[int, ...]
    edge_words: tuple[int, ...]
    edge_ranks: tuple[int, ...]
    table_indices: tuple[int, ...]
    reconstructed_pairs: np.ndarray
    fit_hidden: np.ndarray
    heldout_hidden: np.ndarray
    coefficient_squared_error: float
    fit_output_mean_squared_error: float
    heldout_output_mean_squared_error: float
    fit_forward_kld: float
    heldout_forward_kld: float


@dataclass(frozen=True)
class DownRefit:
    """Fit-only down-matrix proposal and the selected matrix."""

    continuous_target: np.ndarray
    quantized_candidate: np.ndarray
    selected_down: np.ndarray
    selected_scales: np.ndarray
    candidate_k2_paths: tuple[dict[str, Any], ...]
    accepted_on_fit: bool
    candidate_fit_mse: float
    candidate_heldout_mse: float
    selected_fit_mse: float
    selected_heldout_mse: float
    candidate_fit_kld: float
    candidate_heldout_kld: float
    selected_fit_kld: float
    selected_heldout_kld: float
    candidate_direction_cosine: float
    candidate_direction_changed: bool


def situ(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Evaluate Kimi-K3's coordinatewise SiTU activation."""

    return (4.0 * np.tanh(gate / 4.0) / (1.0 + np.exp(-gate))) * (
        25.0 * np.tanh(up / 25.0)
    )


def sqg_xor_cheb_t12_bytes() -> np.ndarray:
    """Return the frozen production T12 rank table as 4,096 raw E4M3 bytes."""

    raw = zlib.decompress(base64.b85decode(_SQG_XOR_CHEB_T12_B85))
    if len(raw) != 4096 or hashlib.sha256(raw).hexdigest() != SQG_XOR_CHEB_T12_SHA256:
        raise RuntimeError("embedded sqg_xor_cheb_t12 rank table failed identity check")
    return np.frombuffer(raw, dtype=np.uint8).copy()


def _decode_e4m3(raw: np.ndarray) -> np.ndarray:
    """Decode finite E4M3FN bytes without a framework-specific float8 dtype."""

    raw = np.asarray(raw, dtype=np.uint8)
    sign = np.where((raw & 0x80) != 0, -1.0, 1.0)
    exponent = (raw >> 3) & 0x0F
    mantissa = raw & 0x07
    if np.any((exponent == 0x0F) & (mantissa == 0x07)):
        raise ValueError("E4M3 table contains NaN labels")
    subnormal = (mantissa.astype(np.float64) / 8.0) * (2.0**-6)
    normal = (1.0 + mantissa.astype(np.float64) / 8.0) * np.exp2(
        exponent.astype(np.int16) - 7
    )
    return sign * np.where(exponent == 0, subnormal, normal)


def _quantize_finite_e4m3(values: np.ndarray) -> np.ndarray:
    raw = np.arange(256, dtype=np.uint8)
    finite = ~((((raw >> 3) & 0x0F) == 0x0F) & ((raw & 0x07) == 0x07))
    raw = raw[finite]
    decoded = _decode_e4m3(raw)
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    nearest = np.abs(flat[:, None] - decoded[None, :]).argmin(axis=1)
    return _decode_e4m3(raw[nearest]).reshape(np.asarray(values).shape)


def _reverse_four_bits(value: int) -> int:
    return (
        ((value & 0x1) << 3)
        | ((value & 0x2) << 1)
        | ((value & 0x4) >> 1)
        | ((value & 0x8) >> 3)
    )


def _next_state(state: int, branch: int) -> int:
    """Advance the twelve-history-bit state of the K4 pair graph."""

    return ((state << PAIR_BRANCH_BITS) | branch) & _HISTORY_MASK


def _sqg_edge_rank(state: int, branch: int) -> int:
    """Reproduce the production K4 carry-mixed SQG rank map for one edge."""

    mixed = state ^ (state >> 11)
    mixed ^= (mixed << 11) & _HISTORY_MASK
    product = (0x3FA7D929 * mixed + 0xC928FD8E) & 0xFFFFFFFF
    phase = product & _HISTORY_MASK
    syndrome = product >> 28
    stratum = _reverse_four_bits(branch) ^ syndrome
    return (stratum << TRELLIS_STATE_BITS) | phase


def _sqg_edge_rank_for_rate(state: int, branch: int, bits: int) -> int:
    """Reproduce the carry-mixed SQG rank map for K2 or K4."""

    width = 16 - bits
    history_mask = (1 << width) - 1
    mixed = state ^ (state >> 11)
    mixed ^= (mixed << 11) & history_mask
    product = (0x3FA7D929 * mixed + 0xC928FD8E) & 0xFFFFFFFF
    phase = product & history_mask
    syndrome = product >> (32 - bits)
    reversed_branch = int(f"{branch:0{bits}b}"[::-1], 2)
    return ((reversed_branch ^ syndrome) << width) | phase


def _table_index(state: int, branch: int) -> int:
    """Retain four stratum and eight phase bits from the production SQG rank."""

    return _sqg_edge_rank(state, branch) >> 4


def _tail_biting_initial_state(branches: tuple[int, ...]) -> int:
    """Build the three-nibble history preceding a periodic pair path."""

    history = [branches[(-offset) % len(branches)] for offset in range(3, 0, -1)]
    state = 0
    for branch in history:
        state = (state << PAIR_BRANCH_BITS) | branch
    return state


def closed_paths(
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    """Yield every two-step periodic path with its states and table indices."""

    for branches in itertools.product(range(PAIR_BRANCHES), repeat=PAIR_STEPS):
        state = _tail_biting_initial_state(branches)
        initial_state = state
        states: list[int] = []
        indices: list[int] = []
        for branch in branches:
            states.append(state)
            indices.append(_table_index(state, branch))
            state = _next_state(state, branch)
        if state != initial_state:
            raise AssertionError("periodic pair path did not close")
        yield branches, tuple(states), tuple(indices)


def _expert_hidden(inputs: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    gate_weight = pairs[:, 0].reshape(INTERMEDIATE_NEURONS, INPUT_DIMENSION)
    up_weight = pairs[:, 1].reshape(INTERMEDIATE_NEURONS, INPUT_DIMENSION)
    return situ(inputs @ gate_weight.T, inputs @ up_weight.T)


def _expert_output(hidden: np.ndarray, down: np.ndarray) -> np.ndarray:
    return hidden @ down.T


def _round_bf16(values: np.ndarray) -> np.ndarray:
    """Round float values to BF16 with round-to-nearest-even semantics."""

    fp32 = np.asarray(values, dtype=np.float32)
    bits = fp32.view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = (bits + rounding) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def _bf16_expert_output(
    inputs: np.ndarray, pairs: np.ndarray, down: np.ndarray
) -> np.ndarray:
    """Replay BF16 operands, FP32 dot accumulation, and BF16 boundaries."""

    bf16_inputs = _round_bf16(inputs)
    bf16_pairs = _round_bf16(pairs)
    gate_weight = bf16_pairs[:, 0].reshape(INTERMEDIATE_NEURONS, INPUT_DIMENSION)
    up_weight = bf16_pairs[:, 1].reshape(INTERMEDIATE_NEURONS, INPUT_DIMENSION)
    gate = _round_bf16(bf16_inputs @ gate_weight.T)
    up = _round_bf16(bf16_inputs @ up_weight.T)
    hidden = _round_bf16(situ(gate, up))
    return _round_bf16(hidden @ _round_bf16(down).T)


def make_normalized_pair_table(correlation: float = 0.7) -> np.ndarray:
    """Synthesize a 4,096-entry joint-quartile/phase finite-E4M3 table."""

    if not -0.99 < correlation < 0.99:
        raise ValueError("pair-table calibration correlation must be between -0.99 and 0.99")
    generator = np.random.default_rng(CALIBRATION_SEED)
    gate_samples = generator.normal(0.0, 1.0, CALIBRATION_PAIR_COUNT)
    residual_scale = np.sqrt(max(0.0, 1.0 - correlation * correlation))
    pair_samples = np.column_stack(
        (
            gate_samples,
            correlation * gate_samples
            + generator.normal(0.0, residual_scale, CALIBRATION_PAIR_COUNT),
        )
    ) * 1.5
    boundaries = np.quantile(pair_samples, (0.25, 0.5, 0.75), axis=0)
    quartiles = np.column_stack(
        [np.searchsorted(boundaries[:, component], pair_samples[:, component])
         for component in range(2)]
    )
    strata = (quartiles[:, 0] << 2) | quartiles[:, 1]
    table = np.empty((4096, 2), dtype=np.float64)
    quantiles = (np.arange(256, dtype=np.float64) + 0.5) / 256.0
    for stratum in range(16):
        samples = pair_samples[strata == stratum]
        order = np.argsort(samples[:, 0] + 0.61803398875 * samples[:, 1])
        positions = np.minimum((quantiles * len(order)).astype(np.int64), len(order) - 1)
        table[stratum * 256 : (stratum + 1) * 256] = samples[order[positions]]
    return _quantize_finite_e4m3(table)


def _make_rows(source_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create disjoint fit and held-out rows from independent random streams."""

    fit_generator = np.random.default_rng(100_003 + 2 * source_seed)
    heldout_generator = np.random.default_rng(200_003 + 2 * source_seed)
    fit = fit_generator.normal(0.0, 0.9, (FIT_ROW_COUNT, INPUT_DIMENSION))
    heldout = heldout_generator.normal(0.15, 1.15, (HELDOUT_ROW_COUNT, INPUT_DIMENSION))
    return fit.astype(np.float64), heldout.astype(np.float64)


def make_problem(
    source_seed: int = SOURCE_SEED,
    *,
    normalized_pair_table: np.ndarray | None = None,
    pair_table_correlation: float = 0.7,
) -> TinyProblem:
    """Create one multi-neuron expert and its disjoint evaluation rows."""

    if normalized_pair_table is None:
        normalized_pair_table = make_normalized_pair_table(pair_table_correlation)
    if normalized_pair_table.shape != (4096, 2):
        raise ValueError("normalized pair table must have shape [4096, 2]")
    generator = np.random.default_rng(source_seed)
    source_pairs = generator.normal(0.0, 1.0, (PAIR_STEPS, 2))
    source_pairs[:, 1] = 0.7 * source_pairs[:, 0] + generator.normal(
        0.0, 0.4, PAIR_STEPS
    )
    upstream_scales = np.maximum(
        np.abs(source_pairs).max(axis=0) / 5.5, 1e-8
    )
    trained_pair_table = normalized_pair_table * upstream_scales

    source_down = generator.uniform(
        -0.45, 0.45, (OUTPUT_DIMENSION, INTERMEDIATE_NEURONS)
    )
    down_scales = np.maximum(np.abs(source_down).max(axis=1) / 5.5, 1e-8)
    source_down_rows = [
        _scalar_k2_stream(source_down[row], float(down_scales[row]))
        for row in range(OUTPUT_DIMENSION)
    ]
    quantized_source_down = np.stack([result[0] for result in source_down_rows])
    fit_inputs, heldout_inputs = _make_rows(source_seed)
    return TinyProblem(
        source_pairs=source_pairs,
        source_down=source_down,
        fit_inputs=fit_inputs,
        heldout_inputs=heldout_inputs,
        fit_source_output=_expert_output(
            _expert_hidden(fit_inputs, source_pairs), source_down
        ),
        heldout_source_output=_expert_output(
            _expert_hidden(heldout_inputs, source_pairs), source_down
        ),
        trained_pair_table=trained_pair_table,
        upstream_scales=upstream_scales,
        down_scales=down_scales,
        source_down_k2_paths=tuple(result[1] for result in source_down_rows),
        quantized_source_down=quantized_source_down,
    )


def _mean_squared_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.square(candidate - reference).mean())


def _pseudo_vocabulary_logits(expert_outputs: np.ndarray) -> np.ndarray:
    """Map both expert-output directions into four centered synthetic logits."""

    return np.asarray(expert_outputs, dtype=np.float64) @ PSEUDO_VOCABULARY_READOUT.T


def _forward_kld(teacher_outputs: np.ndarray, candidate_outputs: np.ndarray) -> float:
    """Return forward KLD after the fixed rank-two pseudo-vocabulary readout."""

    teacher_logits = _pseudo_vocabulary_logits(teacher_outputs)
    candidate_logits = _pseudo_vocabulary_logits(candidate_outputs)
    teacher_shifted = teacher_logits - teacher_logits.max(axis=1, keepdims=True)
    candidate_shifted = candidate_logits - candidate_logits.max(axis=1, keepdims=True)
    teacher_log_probability = teacher_shifted - np.log(
        np.exp(teacher_shifted).sum(axis=1, keepdims=True)
    )
    candidate_log_probability = candidate_shifted - np.log(
        np.exp(candidate_shifted).sum(axis=1, keepdims=True)
    )
    teacher_probability = np.exp(teacher_log_probability)
    divergence = np.sum(
        teacher_probability * (teacher_log_probability - candidate_log_probability), axis=1
    ).mean()
    return max(0.0, float(divergence))


def choose_path(
    problem: TinyProblem,
    table: np.ndarray,
    *,
    objective: str,
) -> PathChoice:
    """Choose a closed path by coefficient error or fit-row forward KLD."""

    if objective not in {"coefficient", "fit_forward_kld"}:
        raise ValueError(f"unsupported path objective: {objective}")
    best: PathChoice | None = None
    best_key: tuple[float, tuple[int, ...]] | None = None
    for branches, states, indices in closed_paths():
        reconstructed = table[np.asarray(indices)]
        coefficient_error = float(np.square(reconstructed - problem.source_pairs).sum())
        fit_hidden = _expert_hidden(problem.fit_inputs, reconstructed)
        heldout_hidden = _expert_hidden(problem.heldout_inputs, reconstructed)
        fit_output = _expert_output(fit_hidden, problem.quantized_source_down)
        heldout_output = _expert_output(heldout_hidden, problem.quantized_source_down)
        fit_error = _mean_squared_error(fit_output, problem.fit_source_output)
        heldout_error = _mean_squared_error(heldout_output, problem.heldout_source_output)
        fit_kld = _forward_kld(problem.fit_source_output, fit_output)
        heldout_kld = _forward_kld(problem.heldout_source_output, heldout_output)
        cost = coefficient_error if objective == "coefficient" else fit_kld
        key = (cost, branches)
        if best_key is None or key < best_key:
            edge_words = tuple(
                ((state << PAIR_BRANCH_BITS) | branch) & _WORD_MASK
                for state, branch in zip(states, branches)
            )
            best_key = key
            best = PathChoice(
                branches=branches,
                states=states,
                edge_words=edge_words,
                edge_ranks=tuple(
                    _sqg_edge_rank(state, branch)
                    for state, branch in zip(states, branches)
                ),
                table_indices=indices,
                reconstructed_pairs=reconstructed,
                fit_hidden=fit_hidden,
                heldout_hidden=heldout_hidden,
                coefficient_squared_error=coefficient_error,
                fit_output_mean_squared_error=fit_error,
                heldout_output_mean_squared_error=heldout_error,
                fit_forward_kld=fit_kld,
                heldout_forward_kld=heldout_kld,
            )
    if best is None:
        raise RuntimeError("the pair trellis has no closed path")
    return best


def _scalar_k2_stream(source: np.ndarray, scale: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Quantize two weights through a periodic production-form K2 control."""

    width = SCALAR_K2_STATE_BITS
    mask = (1 << width) - 1
    levels = _decode_e4m3(sqg_xor_cheb_t12_bytes()) * scale
    best: tuple[float, tuple[int, ...], tuple[int, ...], tuple[int, ...], np.ndarray] | None = None
    for branches in itertools.product(range(4), repeat=2):
        history_symbols = [branches[index % 2] for index in range(1, 8)]
        state = 0
        for symbol in history_symbols:
            state = ((state << 2) | symbol) & mask
        initial_state = state
        states: list[int] = []
        ranks: list[int] = []
        indices: list[int] = []
        for branch in branches:
            states.append(state)
            rank = _sqg_edge_rank_for_rate(state, branch, 2)
            ranks.append(rank)
            indices.append(rank >> 4)
            state = ((state << 2) | branch) & mask
        if state != initial_state:
            raise AssertionError("periodic scalar K2 path did not close")
        reconstructed = levels[np.asarray(indices)]
        candidate = (float(np.square(reconstructed - source).sum()), branches,
                     tuple(states), tuple(ranks), reconstructed)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("the scalar K2 trellis has no closed path")
    error, branches, states, ranks, reconstructed = best
    return reconstructed, {
        "branches": list(branches), "states": list(states), "edge_ranks": list(ranks),
        "table_indices": [rank >> 4 for rank in ranks],
        "coefficient_squared_error": error,
    }


def choose_scalar_k2_control(problem: TinyProblem) -> tuple[PathChoice, dict[str, Any]]:
    """Build the matched-payload independent scalar K2 comparator."""

    gate, gate_path = _scalar_k2_stream(
        problem.source_pairs[:, 0], float(problem.upstream_scales[0])
    )
    up, up_path = _scalar_k2_stream(
        problem.source_pairs[:, 1], float(problem.upstream_scales[1])
    )
    reconstructed = np.column_stack((gate, up))
    fit_hidden = _expert_hidden(problem.fit_inputs, reconstructed)
    heldout_hidden = _expert_hidden(problem.heldout_inputs, reconstructed)
    fit_output = _expert_output(fit_hidden, problem.quantized_source_down)
    heldout_output = _expert_output(heldout_hidden, problem.quantized_source_down)
    combined_branches = tuple(
        gate_branch | (up_branch << 2)
        for gate_branch, up_branch in zip(gate_path["branches"], up_path["branches"])
    )
    choice = PathChoice(
        branches=combined_branches,
        states=tuple(gate_path["states"]), edge_words=(),
        edge_ranks=tuple(gate_path["edge_ranks"]),
        table_indices=tuple(gate_path["table_indices"]),
        reconstructed_pairs=reconstructed, fit_hidden=fit_hidden,
        heldout_hidden=heldout_hidden,
        coefficient_squared_error=float(np.square(reconstructed - problem.source_pairs).sum()),
        fit_output_mean_squared_error=_mean_squared_error(fit_output, problem.fit_source_output),
        heldout_output_mean_squared_error=_mean_squared_error(
            heldout_output, problem.heldout_source_output
        ),
        fit_forward_kld=_forward_kld(problem.fit_source_output, fit_output),
        heldout_forward_kld=_forward_kld(problem.heldout_source_output, heldout_output),
    )
    return choice, {"gate": gate_path, "up": up_path}


def _direction_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.sum(left * right) / denominator)


def refit_down_target(problem: TinyProblem, upstream: PathChoice) -> DownRefit:
    """Fit a 2-by-2 down matrix on fit rows and report held-out transfer."""

    hidden = upstream.fit_hidden
    regularized_gram = hidden.T @ hidden + RIDGE_STRENGTH * np.eye(
        INTERMEDIATE_NEURONS, dtype=np.float64
    )
    regularized_rhs = (
        hidden.T @ problem.fit_source_output
        + RIDGE_STRENGTH * problem.source_down.T
    )
    continuous_target = np.linalg.solve(regularized_gram, regularized_rhs).T

    nominal_scales = np.maximum(np.abs(continuous_target).max(axis=1) / 5.5, 1e-8)
    candidate_options: list[
        tuple[float, float, float, np.ndarray, np.ndarray, tuple[dict[str, Any], ...]]
    ] = []
    for scale_multiplier in np.linspace(0.8, 1.2, 9):
        scales = nominal_scales * scale_multiplier
        encoded_rows = [
            _scalar_k2_stream(continuous_target[row], float(scales[row]))
            for row in range(OUTPUT_DIMENSION)
        ]
        quantized = np.stack([result[0] for result in encoded_rows])
        fit_output = _expert_output(upstream.fit_hidden, quantized)
        fit_kld = _forward_kld(problem.fit_source_output, fit_output)
        fit_mse = _mean_squared_error(fit_output, problem.fit_source_output)
        candidate_options.append(
            (fit_kld, fit_mse, float(scale_multiplier), quantized, scales,
             tuple(result[1] for result in encoded_rows))
        )
    (candidate_fit_kld, candidate_fit_mse, _candidate_multiplier,
     quantized_candidate, candidate_scales, candidate_paths) = min(
        candidate_options, key=lambda value: (value[0], value[1])
    )
    candidate_heldout_output = _expert_output(
        upstream.heldout_hidden, quantized_candidate
    )
    candidate_heldout_mse = _mean_squared_error(
        candidate_heldout_output, problem.heldout_source_output
    )
    candidate_heldout_kld = _forward_kld(
        problem.heldout_source_output, candidate_heldout_output
    )
    baseline_fit_kld = upstream.fit_forward_kld
    baseline_heldout_kld = upstream.heldout_forward_kld
    accepted = candidate_fit_kld < baseline_fit_kld - max(
        1e-15, abs(baseline_fit_kld) * 1e-12
    )
    selected_down = quantized_candidate if accepted else problem.quantized_source_down
    selected_scales = candidate_scales if accepted else problem.down_scales
    cosine = _direction_cosine(quantized_candidate, problem.quantized_source_down)
    return DownRefit(
        continuous_target=continuous_target,
        quantized_candidate=quantized_candidate,
        selected_down=selected_down,
        selected_scales=selected_scales,
        candidate_k2_paths=candidate_paths,
        accepted_on_fit=accepted,
        candidate_fit_mse=candidate_fit_mse,
        candidate_heldout_mse=candidate_heldout_mse,
        selected_fit_mse=(
            candidate_fit_mse
            if accepted
            else upstream.fit_output_mean_squared_error
        ),
        selected_heldout_mse=(
            candidate_heldout_mse
            if accepted
            else upstream.heldout_output_mean_squared_error
        ),
        candidate_fit_kld=candidate_fit_kld,
        candidate_heldout_kld=candidate_heldout_kld,
        selected_fit_kld=candidate_fit_kld if accepted else baseline_fit_kld,
        selected_heldout_kld=(candidate_heldout_kld if accepted else baseline_heldout_kld),
        candidate_direction_cosine=cosine,
        candidate_direction_changed=not np.isclose(abs(cosine), 1.0, atol=1e-10),
    )


def _path_json(choice: PathChoice) -> dict[str, Any]:
    return {
        "branches": list(choice.branches),
        "branch_hex": [format(value, "x") for value in choice.branches],
        "states": list(choice.states),
        "state_hex": [format(value, "03x") for value in choice.states],
        "edge_words": list(choice.edge_words),
        "edge_hex": [format(value, "04x") for value in choice.edge_words],
        "edge_ranks": list(choice.edge_ranks),
        "rank_hex": [format(value, "04x") for value in choice.edge_ranks],
        "table_indices": list(choice.table_indices),
        "reconstructed_pairs": choice.reconstructed_pairs.tolist(),
        "coefficient_squared_error": choice.coefficient_squared_error,
        "fit_output_mean_squared_error": choice.fit_output_mean_squared_error,
        "heldout_output_mean_squared_error": choice.heldout_output_mean_squared_error,
        "fit_forward_kld": choice.fit_forward_kld,
        "heldout_forward_kld": choice.heldout_forward_kld,
    }


def _relative_change(before: float, after: float) -> float:
    return 0.0 if before == 0.0 else 1.0 - after / before


def _change_status(before: float, after: float) -> str:
    tolerance = max(abs(before), abs(after)) * 1e-12 + 1e-18
    if after < before - tolerance:
        return "improved"
    if after > before + tolerance:
        return "regressed"
    return "unchanged"


def _stage_changes(
    previous_fit: float, previous_heldout: float, fit: float, heldout: float
) -> dict[str, Any]:
    return {
        "fit_reduction_from_previous": _relative_change(previous_fit, fit),
        "fit_status": _change_status(previous_fit, fit),
        "heldout_reduction_from_previous": _relative_change(previous_heldout, heldout),
        "heldout_status": _change_status(previous_heldout, heldout),
        # Compatibility alias. Held-out error is the generalization result.
        "reduction_from_previous": _relative_change(previous_heldout, heldout),
    }


def _kld_stage_changes(
    previous_fit: float, previous_heldout: float, fit: float, heldout: float
) -> dict[str, Any]:
    return {
        "fit_forward_kld_reduction_from_previous": _relative_change(previous_fit, fit),
        "fit_forward_kld_status": _change_status(previous_fit, fit),
        "heldout_forward_kld_reduction_from_previous": _relative_change(
            previous_heldout, heldout
        ),
        "heldout_forward_kld_status": _change_status(previous_heldout, heldout),
    }


def run_benchmark(
    source_seed: int = SOURCE_SEED,
    *,
    normalized_pair_table: np.ndarray | None = None,
    pair_table_correlation: float = 0.7,
) -> dict[str, Any]:
    """Run all three falsification checks for one synthetic expert."""

    if BITS_AT_PLAY > BIT_LIMIT:
        raise AssertionError(f"tiny benchmark uses {BITS_AT_PLAY} bits, limit is {BIT_LIMIT}")
    problem = make_problem(
        source_seed,
        normalized_pair_table=normalized_pair_table,
        pair_table_correlation=pair_table_correlation,
    )
    scalar_k2, scalar_k2_paths = choose_scalar_k2_control(problem)
    trained = choose_path(problem, problem.trained_pair_table, objective="coefficient")
    output_aware = choose_path(
        problem, problem.trained_pair_table, objective="fit_forward_kld"
    )
    refit = refit_down_target(problem, output_aware)

    choices = (scalar_k2, trained, output_aware)
    path_names = ("scalar_k2_control", "trained_pair_table", "output_aware_path")
    unique_paths = {choice.branches for choice in choices}
    pairwise_path_differences = {
        f"{path_names[left]}_vs_{path_names[right]}": sum(
            a != b for a, b in zip(choices[left].branches, choices[right].branches)
        )
        for left in range(len(choices))
        for right in range(left + 1, len(choices))
    }

    bf16_fit_source = _bf16_expert_output(
        problem.fit_inputs, problem.source_pairs, problem.source_down
    )
    bf16_heldout_source = _bf16_expert_output(
        problem.heldout_inputs, problem.source_pairs, problem.source_down
    )

    def bf16_errors(choice: PathChoice, down: np.ndarray) -> tuple[float, float]:
        return (
            _mean_squared_error(
                _bf16_expert_output(problem.fit_inputs, choice.reconstructed_pairs, down),
                bf16_fit_source,
            ),
            _mean_squared_error(
                _bf16_expert_output(
                    problem.heldout_inputs, choice.reconstructed_pairs, down
                ),
                bf16_heldout_source,
            ),
        )

    bf16 = (
        bf16_errors(scalar_k2, problem.quantized_source_down),
        bf16_errors(trained, problem.quantized_source_down),
        bf16_errors(output_aware, problem.quantized_source_down),
        bf16_errors(output_aware, refit.selected_down),
    )
    fit_errors = (
        scalar_k2.fit_output_mean_squared_error,
        trained.fit_output_mean_squared_error,
        output_aware.fit_output_mean_squared_error,
        refit.selected_fit_mse,
    )
    heldout_errors = (
        scalar_k2.heldout_output_mean_squared_error,
        trained.heldout_output_mean_squared_error,
        output_aware.heldout_output_mean_squared_error,
        refit.selected_heldout_mse,
    )
    fit_klds = (
        scalar_k2.fit_forward_kld,
        trained.fit_forward_kld,
        output_aware.fit_forward_kld,
        refit.selected_fit_kld,
    )
    heldout_klds = (
        scalar_k2.heldout_forward_kld,
        trained.heldout_forward_kld,
        output_aware.heldout_forward_kld,
        refit.selected_heldout_kld,
    )

    stages: dict[str, dict[str, Any]] = {
        "scalar_k2_control": {
            "explanation": (
                "matched-payload independent scalar K2 control using the exact "
                "4,096-entry sqg_xor_cheb_t12 reconstruction table"
            ),
            "comparator": "matched_payload_scalar_K2_control",
            "not_exl3": True,
            "selection_split": "coefficients_only",
            "path": _path_json(scalar_k2),
            "scalar_k2_paths": scalar_k2_paths,
            "fit_output_mean_squared_error": fit_errors[0],
            "heldout_output_mean_squared_error": heldout_errors[0],
            "output_mean_squared_error": heldout_errors[0],
            "fit_forward_kld": fit_klds[0],
            "heldout_forward_kld": heldout_klds[0],
            "fit_bf16_output_mean_squared_error": bf16[0][0],
            "heldout_bf16_output_mean_squared_error": bf16[0][1],
            "bf16_output_mean_squared_error": bf16[0][1],
        },
        "trained_pair_table": {
            "explanation": (
                "shared trained gate/up pair table and coefficient-error "
                "path selection"
            ),
            "selection_split": "coefficients_only",
            "path": _path_json(trained),
            "fit_output_mean_squared_error": fit_errors[1],
            "heldout_output_mean_squared_error": heldout_errors[1],
            "output_mean_squared_error": heldout_errors[1],
            "fit_forward_kld": fit_klds[1],
            "heldout_forward_kld": heldout_klds[1],
            "fit_bf16_output_mean_squared_error": bf16[1][0],
            "heldout_bf16_output_mean_squared_error": bf16[1][1],
            "bf16_output_mean_squared_error": bf16[1][1],
            **_stage_changes(
                fit_errors[0], heldout_errors[0], fit_errors[1], heldout_errors[1]
            ),
            **_kld_stage_changes(
                fit_klds[0], heldout_klds[0], fit_klds[1], heldout_klds[1]
            ),
        },
        "output_aware_path": {
            "explanation": (
                "trained pair table with path selection on fit-row synthetic "
                "forward KLD"
            ),
            "selection_split": "fit_rows_only",
            "path": _path_json(output_aware),
            "fit_output_mean_squared_error": fit_errors[2],
            "heldout_output_mean_squared_error": heldout_errors[2],
            "output_mean_squared_error": heldout_errors[2],
            "fit_forward_kld": fit_klds[2],
            "heldout_forward_kld": heldout_klds[2],
            "fit_bf16_output_mean_squared_error": bf16[2][0],
            "heldout_bf16_output_mean_squared_error": bf16[2][1],
            "bf16_output_mean_squared_error": bf16[2][1],
            **_stage_changes(fit_errors[1], heldout_errors[1], fit_errors[2], heldout_errors[2]),
            **_kld_stage_changes(
                fit_klds[1], heldout_klds[1], fit_klds[2], heldout_klds[2]
            ),
        },
        "refit_down_target": {
            "explanation": "2-by-2 down matrix fitted and accepted on fit rows only",
            "selection_split": "fit_rows_only",
            "continuous_down_target": refit.continuous_target.tolist(),
            "quantized_down_candidate": refit.quantized_candidate.tolist(),
            "quantized_down_target": refit.selected_down.tolist(),
            "down_target_scales": refit.selected_scales.tolist(),
            "down_candidate_k2_paths": list(refit.candidate_k2_paths),
            "down_target_accepted": refit.accepted_on_fit,
            "candidate_direction_cosine": refit.candidate_direction_cosine,
            "candidate_direction_changed": refit.candidate_direction_changed,
            "candidate_fit_output_mean_squared_error": refit.candidate_fit_mse,
            "candidate_heldout_output_mean_squared_error": refit.candidate_heldout_mse,
            "candidate_fit_forward_kld": refit.candidate_fit_kld,
            "candidate_heldout_forward_kld": refit.candidate_heldout_kld,
            "fit_output_mean_squared_error": fit_errors[3],
            "heldout_output_mean_squared_error": heldout_errors[3],
            "output_mean_squared_error": heldout_errors[3],
            "fit_forward_kld": fit_klds[3],
            "heldout_forward_kld": heldout_klds[3],
            "fit_bf16_output_mean_squared_error": bf16[3][0],
            "heldout_bf16_output_mean_squared_error": bf16[3][1],
            "bf16_output_mean_squared_error": bf16[3][1],
            **_stage_changes(fit_errors[2], heldout_errors[2], fit_errors[3], heldout_errors[3]),
            **_kld_stage_changes(
                fit_klds[2], heldout_klds[2], fit_klds[3], heldout_klds[3]
            ),
        },
    }
    return {
        "status": "research-only falsification benchmark",
        "kld_evidence_boundary": (
            "Synthetic teacher-to-candidate forward KLD after a fixed centered "
            "rank-two readout to four pseudo-vocabulary logits; "
            "this is not EXL3 or full-model KLD and cannot establish EXL3 dominance."
        ),
        "size_evidence_boundary": (
            "The <=32-bit bound counts one expert's payload plus live encoder state, "
            "not stored exact bytes. Tables, scales, metadata, and padding are outside "
            "the count, so this benchmark cannot establish smaller total model size."
        ),
        "calibration_seed": CALIBRATION_SEED,
        "source_seed": source_seed,
        "pair_table_calibration_correlation": pair_table_correlation,
        "bit_budget": {
            "source_weight_count": PAIR_STEPS * 2 + DOWN_WEIGHT_COUNT,
            "gate_up_stream_bits": PAIR_STEPS * PAIR_BRANCH_BITS,
            "down_stream_bits": DOWN_WEIGHT_COUNT * DOWN_WEIGHT_BITS,
            "payload_bits": PAYLOAD_BITS,
            "payload_bits_per_weight": PAYLOAD_BITS
            / (PAIR_STEPS * 2 + DOWN_WEIGHT_COUNT),
            "pair_working_state_bits": TRELLIS_STATE_BITS,
            "scalar_k2_working_state_bits": SCALAR_K2_STATE_BITS,
            "pair_bits_at_play": PAIR_BITS_AT_PLAY,
            "scalar_k2_bits_at_play": SCALAR_K2_BITS_AT_PLAY,
            "bits_at_play": BITS_AT_PLAY,
            "limit": BIT_LIMIT,
        },
        "problem": {
            "input_dimensions": INPUT_DIMENSION,
            "intermediate_neurons": INTERMEDIATE_NEURONS,
            "output_dimensions": OUTPUT_DIMENSION,
            "calibration_pairs": CALIBRATION_PAIR_COUNT,
            "fit_rows": FIT_ROW_COUNT,
            "heldout_rows": HELDOUT_ROW_COUNT,
            "fit_and_heldout_rows_disjoint": not np.any(
                np.all(
                    problem.fit_inputs[:, None, :]
                    == problem.heldout_inputs[None, :, :],
                    axis=2,
                )
            ),
            "closed_paths": sum(1 for _ in closed_paths()),
            "trellis_graph": {
                "history_bits": TRELLIS_STATE_BITS,
                "branch_bits": PAIR_BRANCH_BITS,
                "branches_per_state": PAIR_BRANCHES,
                "rank_map": "sqg_xor_rank_permutation_k4",
                "pair_table": (
                    "4,096 finite-E4M3 pair entries selected by rank >> 4; "
                    "upper four bits select joint quartiles and eight phase bits "
                    "select deterministic within-stratum quantiles"
                ),
                "pair_table_training": (
                    "deterministic 65,536-sample correlated-normal calibration "
                    "with empirical joint quartiles and 256 within-stratum quantiles; "
                    "this is a tiny synthetic training approximation"
                ),
            },
            "pseudo_vocabulary_readout": {
                "matrix": PSEUDO_VOCABULARY_READOUT.tolist(),
                "centered": bool(
                    np.allclose(PSEUDO_VOCABULARY_READOUT.sum(axis=0), 0.0)
                ),
                "rank": int(np.linalg.matrix_rank(PSEUDO_VOCABULARY_READOUT)),
                "pseudo_vocabulary_size": int(PSEUDO_VOCABULARY_READOUT.shape[0]),
            },
            "comparator": {
                "name": "matched_payload_scalar_K2_control",
                "not_exl3": True,
                "history_bits": SCALAR_K2_STATE_BITS,
                "branch_bits": 2,
                "payload_bits": PAYLOAD_BITS,
                "scalar_table": "exact frozen 4,096-entry sqg_xor_cheb_t12 rank table",
                "scalar_table_sha256": SQG_XOR_CHEB_T12_SHA256,
            },
            "source_pairs": problem.source_pairs.tolist(),
            "source_down": problem.source_down.tolist(),
            "upstream_scales": problem.upstream_scales.tolist(),
            "down_scales": problem.down_scales.tolist(),
            "source_down_k2_paths": list(problem.source_down_k2_paths),
            "trained_pair_table": problem.trained_pair_table.tolist(),
            "quantized_source_down": problem.quantized_source_down.tolist(),
            "blockldlq": "omitted from this bounded synthetic benchmark",
            "fit_source_output_energy": float(np.square(problem.fit_source_output).mean()),
            "heldout_source_output_energy": float(
                np.square(problem.heldout_source_output).mean()
            ),
        },
        "path_comparison": {
            "unique_reconstruction_paths": len(unique_paths),
            "all_three_paths_distinct": len(unique_paths) == len(choices),
            "pairwise_branch_hamming_distance": pairwise_path_differences,
        },
        "stages": stages,
        "heldout_regression_stages": [
            name
            for name in path_names[1:] + ("refit_down_target",)
            if stages[name]["heldout_status"] == "regressed"
        ],
        "heldout_forward_kld_regression_stages": [
            name
            for name in path_names[1:] + ("refit_down_target",)
            if stages[name]["heldout_forward_kld_status"] == "regressed"
        ],
        "total_output_error_reduction": _relative_change(
            heldout_errors[0], heldout_errors[-1]
        ),
        "total_bf16_output_error_reduction": _relative_change(
            bf16[0][1], bf16[-1][1]
        ),
        "total_heldout_forward_kld_reduction": _relative_change(
            heldout_klds[0], heldout_klds[-1]
        ),
    }


def _transition_summary(errors: np.ndarray, stage_names: tuple[str, ...]) -> dict[str, Any]:
    transitions: dict[str, Any] = {}
    for index, name in enumerate(stage_names[1:]):
        before = errors[:, index]
        after = errors[:, index + 1]
        tolerance = np.maximum(np.abs(before), np.abs(after)) * 1e-12 + 1e-18
        reductions = np.divide(
            before - after,
            before,
            out=np.zeros_like(before),
            where=before != 0,
        )
        transitions[name] = {
            "pooled_error_reduction": float(1.0 - after.sum() / before.sum()),
            "median_expert_error_reduction": float(np.median(reductions)),
            "improved_experts": int(np.count_nonzero(after < before - tolerance)),
            "unchanged_experts": int(np.count_nonzero(np.abs(after - before) <= tolerance)),
            "regressed_experts": int(np.count_nonzero(after > before + tolerance)),
        }
    return transitions


def run_sweep(
    expert_count: int = 8, *, start_seed: int = 0,
    pair_table_correlation: float = 0.7,
) -> dict[str, Any]:
    """Evaluate configurable experts sequentially with one shared pair table."""

    if isinstance(expert_count, bool) or not isinstance(expert_count, int) or expert_count <= 0:
        raise ValueError("expert_count must be a positive integer")
    normalized_pair_table = make_normalized_pair_table(pair_table_correlation)
    expert_reports = [
        run_benchmark(
            start_seed + expert,
            normalized_pair_table=normalized_pair_table,
            pair_table_correlation=pair_table_correlation,
        )
        for expert in range(expert_count)
    ]
    stage_names = (
        "scalar_k2_control",
        "trained_pair_table",
        "output_aware_path",
        "refit_down_target",
    )

    def error_matrix(prefix: str) -> np.ndarray:
        return np.asarray(
            [
                [
                    report["stages"][stage][f"{prefix}_output_mean_squared_error"]
                    for stage in stage_names
                ]
                for report in expert_reports
            ],
            dtype=np.float64,
        )

    fit_errors = error_matrix("fit")
    heldout_errors = error_matrix("heldout")
    fit_klds = np.asarray(
        [[report["stages"][stage]["fit_forward_kld"] for stage in stage_names]
         for report in expert_reports], dtype=np.float64
    )
    heldout_klds = np.asarray(
        [[report["stages"][stage]["heldout_forward_kld"] for stage in stage_names]
         for report in expert_reports], dtype=np.float64
    )
    heldout_bf16_errors = np.asarray(
        [
            [
                report["stages"][stage]["heldout_bf16_output_mean_squared_error"]
                for stage in stage_names
            ]
            for report in expert_reports
        ],
        dtype=np.float64,
    )
    return {
        "status": "research-only sequential falsification benchmark",
        "calibration_seed": CALIBRATION_SEED,
        "start_seed": start_seed,
        "pair_table_calibration_correlation": pair_table_correlation,
        "expert_count": expert_count,
        "shared_pair_table": True,
        "sequential_execution": True,
        "selection_rows": "fit",
        "reporting_rows": "heldout",
        "kld_evidence_boundary": (
            "Synthetic teacher-to-candidate forward KLD after a fixed centered "
            "rank-two readout to four pseudo-vocabulary logits; "
            "this is not EXL3 or full-model KLD and cannot establish EXL3 dominance."
        ),
        "size_evidence_boundary": (
            "The <=32-bit bound counts payload plus live encoder state, not stored "
            "exact bytes; tables, scales, metadata, and padding are excluded."
        ),
        "peak_bits_at_play": BITS_AT_PLAY,
        "aggregate_payload_bits": expert_count * PAYLOAD_BITS,
        "stage_fit_error_sums": {
            stage: float(fit_errors[:, index].sum())
            for index, stage in enumerate(stage_names)
        },
        "stage_output_error_sums": {
            stage: float(heldout_errors[:, index].sum())
            for index, stage in enumerate(stage_names)
        },
        "fit_transitions": _transition_summary(fit_errors, stage_names),
        "transitions": _transition_summary(heldout_errors, stage_names),
        "fit_forward_kld_transitions": _transition_summary(fit_klds, stage_names),
        "heldout_forward_kld_transitions": _transition_summary(
            heldout_klds, stage_names
        ),
        "bf16_transitions": _transition_summary(heldout_bf16_errors, stage_names),
        "experts_with_multiple_reconstruction_paths": sum(
            report["path_comparison"]["unique_reconstruction_paths"] > 1
            for report in expert_reports
        ),
        "experts_with_output_aware_path_change": sum(
            report["stages"]["trained_pair_table"]["path"]["branches"]
            != report["stages"]["output_aware_path"]["path"]["branches"]
            for report in expert_reports
        ),
        "down_refit_fit_acceptances": sum(
            report["stages"]["refit_down_target"]["down_target_accepted"]
            for report in expert_reports
        ),
        "down_refit_direction_changes": sum(
            report["stages"]["refit_down_target"]["candidate_direction_changed"]
            for report in expert_reports
        ),
        "heldout_regressions_by_stage": {
            stage: summary["regressed_experts"]
            for stage, summary in _transition_summary(heldout_errors, stage_names).items()
        },
        "heldout_forward_kld_regressions_by_stage": {
            stage: summary["regressed_experts"]
            for stage, summary in _transition_summary(heldout_klds, stage_names).items()
        },
        "total_pooled_output_error_reduction": float(
            1.0 - heldout_errors[:, -1].sum() / heldout_errors[:, 0].sum()
        ),
        "total_pooled_bf16_output_error_reduction": float(
            1.0 - heldout_bf16_errors[:, -1].sum() / heldout_bf16_errors[:, 0].sum()
        ),
        "total_pooled_heldout_forward_kld_reduction": float(
            1.0 - heldout_klds[:, -1].sum() / heldout_klds[:, 0].sum()
        ),
        "experts": [
            {
                "source_seed": report["source_seed"],
                "unique_reconstruction_paths": report["path_comparison"][
                    "unique_reconstruction_paths"
                ],
                "heldout_regression_stages": report["heldout_regression_stages"],
                "stage_fit_errors": {
                    stage: report["stages"][stage]["fit_output_mean_squared_error"]
                    for stage in stage_names
                },
                "stage_output_errors": {
                    stage: report["stages"][stage]["heldout_output_mean_squared_error"]
                    for stage in stage_names
                },
                "stage_fit_forward_klds": {
                    stage: report["stages"][stage]["fit_forward_kld"]
                    for stage in stage_names
                },
                "stage_heldout_forward_klds": {
                    stage: report["stages"][stage]["heldout_forward_kld"]
                    for stage in stage_names
                },
                "stage_bf16_output_errors": {
                    stage: report["stages"][stage][
                        "heldout_bf16_output_mean_squared_error"
                    ]
                    for stage in stage_names
                },
            }
            for report in expert_reports
        ],
    }


__all__ = [
    "BIT_LIMIT",
    "BITS_AT_PLAY",
    "DOWN_WEIGHT_COUNT",
    "HELDOUT_ROW_COUNT",
    "FIT_ROW_COUNT",
    "INTERMEDIATE_NEURONS",
    "OUTPUT_DIMENSION",
    "PAIR_BRANCH_BITS",
    "PAIR_STEPS",
    "PAYLOAD_BITS",
    "PSEUDO_VOCABULARY_READOUT",
    "SQG_XOR_CHEB_T12_SHA256",
    "DownRefit",
    "PathChoice",
    "TinyProblem",
    "_sqg_edge_rank",
    "_sqg_edge_rank_for_rate",
    "_forward_kld",
    "_pseudo_vocabulary_logits",
    "choose_path",
    "closed_paths",
    "make_normalized_pair_table",
    "make_problem",
    "refit_down_target",
    "run_benchmark",
    "run_sweep",
    "sqg_xor_cheb_t12_bytes",
    "situ",
]

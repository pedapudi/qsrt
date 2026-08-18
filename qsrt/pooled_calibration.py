"""Pooled sufficient statistics for routed expert calibration.

The statistics in this module preserve the exact weighted quadratic objective
for a decoded upstream candidate without retaining its reconstructed hidden
rows.  Route weights enter once as their square, matching the squared error of
the routed expert contribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F

from qsrt.coupled_expert_study import CoupledTriplet
from qsrt.coupled_expert_study import situ_derivatives
from qsrt.qsrt_coupled import CoupledHadamardExecution, encode_coupled_weights
from qsrt.tp_simulator import situ


@dataclass
class CandidateHiddenStatistics:
    """Weighted source/candidate post-activation cross-products."""

    candidate_gram: torch.Tensor
    candidate_source_cross: torch.Tensor
    source_gram: torch.Tensor | None
    candidate_residual_cross: torch.Tensor
    hidden_residual_gram: torch.Tensor | None
    weight_sum: float
    weight_square_sum: float
    rows: int

    @classmethod
    def zeros(
        cls,
        width: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
        retain_source_gram: bool = True,
    ) -> "CandidateHiddenStatistics":
        if width <= 0:
            raise ValueError("hidden width must be positive")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("sufficient statistics require FP32 or FP64")
        shape = (width, width)
        return cls(
            candidate_gram=torch.zeros(shape, device=device, dtype=dtype),
            candidate_source_cross=torch.zeros(shape, device=device, dtype=dtype),
            source_gram=(
                torch.zeros(shape, device=device, dtype=dtype)
                if retain_source_gram
                else None
            ),
            candidate_residual_cross=torch.zeros(
                shape, device=device, dtype=dtype
            ),
            hidden_residual_gram=(
                torch.zeros(shape, device=device, dtype=dtype)
                if retain_source_gram
                else None
            ),
            weight_sum=0.0,
            weight_square_sum=0.0,
            rows=0,
        )

    @property
    def width(self) -> int:
        return int(self.candidate_gram.shape[0])

    @property
    def effective_sample_size(self) -> float:
        if self.weight_sum <= 0 or self.weight_square_sum <= 0:
            return 0.0
        return self.weight_sum**2 / self.weight_square_sum

    def _validate(self) -> None:
        shape = self.candidate_gram.shape
        if (
            self.candidate_gram.ndim != 2
            or shape[0] != shape[1]
            or self.candidate_source_cross.shape != shape
            or (
                self.source_gram is not None
                and self.source_gram.shape != shape
            )
            or (
                self.candidate_residual_cross is not None
                and self.candidate_residual_cross.shape != shape
            )
            or (
                self.hidden_residual_gram is not None
                and self.hidden_residual_gram.shape != shape
            )
            or (self.source_gram is None) != (self.hidden_residual_gram is None)
        ):
            raise ValueError("candidate statistics must contain equal-size squares")
        if not all(
            torch.all(torch.isfinite(value))
            for value in (
                self.candidate_gram,
                self.candidate_source_cross,
                *((self.source_gram,) if self.source_gram is not None else ()),
                self.candidate_residual_cross,
                *(
                    (self.hidden_residual_gram,)
                    if self.hidden_residual_gram is not None
                    else ()
                ),
            )
        ):
            raise ValueError("candidate statistics must be finite")
        if (
            not math.isfinite(self.weight_sum)
            or not math.isfinite(self.weight_square_sum)
            or self.weight_sum < 0
            or self.weight_square_sum < 0
            or self.rows < 0
        ):
            raise ValueError("candidate statistics have invalid support")

    def update(
        self,
        candidate_hidden: torch.Tensor,
        source_hidden: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> None:
        """Accumulate one row chunk using ``route_weights.square()``."""

        self._validate()
        if (
            candidate_hidden.ndim != 2
            or source_hidden.shape != candidate_hidden.shape
            or candidate_hidden.shape[1] != self.width
        ):
            raise ValueError("source and candidate hidden rows do not align")
        if route_weights.ndim != 1 or route_weights.numel() != candidate_hidden.shape[0]:
            raise ValueError("route weights must contain one value per hidden row")
        if not all(
            torch.all(torch.isfinite(value))
            for value in (candidate_hidden, source_hidden, route_weights)
        ):
            raise ValueError("candidate rows and route weights must be finite")

        device = self.candidate_gram.device
        dtype = self.candidate_gram.dtype
        candidate = candidate_hidden.to(device=device, dtype=dtype)
        source = source_hidden.to(device=device, dtype=dtype)
        weights = route_weights.to(device=device, dtype=dtype).square()
        if bool(torch.any(weights < 0)):
            raise ValueError("squared route weights must be nonnegative")
        weighted_candidate = candidate * weights[:, None]
        self.candidate_gram.add_(candidate.T @ weighted_candidate)
        self.candidate_source_cross.add_(candidate.T @ (source * weights[:, None]))
        residual = candidate - source
        weighted_residual = residual * weights[:, None]
        self.candidate_residual_cross.add_(candidate.T @ weighted_residual)
        if self.source_gram is not None:
            self.source_gram.add_(source.T @ (source * weights[:, None]))
            if self.hidden_residual_gram is None:
                raise AssertionError("retained source statistics are incomplete")
            self.hidden_residual_gram.add_(residual.T @ weighted_residual)
        weights64 = weights.double()
        self.weight_sum += float(weights64.sum())
        self.weight_square_sum += float(weights64.square().sum())
        self.rows += int(candidate.shape[0])

    def merge_(self, other: "CandidateHiddenStatistics") -> None:
        """Merge a disjoint row partition."""

        self._validate()
        other._validate()
        if other.width != self.width:
            raise ValueError("candidate statistics widths differ")
        self.candidate_gram.add_(other.candidate_gram.to(self.candidate_gram))
        self.candidate_source_cross.add_(
            other.candidate_source_cross.to(self.candidate_source_cross)
        )
        self.candidate_residual_cross.add_(
            other.candidate_residual_cross.to(self.candidate_residual_cross)
        )
        if (self.source_gram is None) != (other.source_gram is None):
            raise ValueError("candidate statistics disagree on source-Gram retention")
        if self.source_gram is not None and other.source_gram is not None:
            self.source_gram.add_(other.source_gram.to(self.source_gram))
            if (
                self.hidden_residual_gram is None
                or other.hidden_residual_gram is None
            ):
                raise AssertionError("retained source statistics are incomplete")
            self.hidden_residual_gram.add_(
                other.hidden_residual_gram.to(self.hidden_residual_gram)
            )
        self.weight_sum += other.weight_sum
        self.weight_square_sum += other.weight_square_sum
        self.rows += other.rows

    def to(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "CandidateHiddenStatistics":
        """Copy tensor statistics while preserving scalar support evidence."""

        target_dtype = self.candidate_gram.dtype if dtype is None else dtype
        if target_dtype not in (torch.float32, torch.float64):
            raise TypeError("sufficient statistics require FP32 or FP64")
        return CandidateHiddenStatistics(
            candidate_gram=self.candidate_gram.to(
                device=device, dtype=target_dtype
            ),
            candidate_source_cross=self.candidate_source_cross.to(
                device=device, dtype=target_dtype
            ),
            source_gram=(
                None
                if self.source_gram is None
                else self.source_gram.to(device=device, dtype=target_dtype)
            ),
            candidate_residual_cross=self.candidate_residual_cross.to(
                device=device, dtype=target_dtype
            ),
            hidden_residual_gram=(
                None
                if self.hidden_residual_gram is None
                else self.hidden_residual_gram.to(
                    device=device, dtype=target_dtype
                )
            ),
            weight_sum=self.weight_sum,
            weight_square_sum=self.weight_square_sum,
            rows=self.rows,
        )


@dataclass(frozen=True)
class PooledExpertEvaluation:
    """Exact routed-function score and W2 statistics for one candidate."""

    statistics: CandidateHiddenStatistics
    sse: float
    source_energy: float
    routed_occurrences: int
    prefix_scores: Mapping[int, tuple[float, float, int]]

    @property
    def nmse(self) -> float:
        if self.source_energy <= 0:
            raise ValueError("pooled expert evaluation has no source energy")
        return self.sse / self.source_energy


@dataclass(frozen=True)
class PooledPortfolioEvaluation:
    """Exact routed-function scores for candidates sharing one source expert."""

    candidate_sse: Mapping[str, float]
    source_energy: float
    routed_occurrences: int

    def __post_init__(self) -> None:
        if (
            not self.candidate_sse
            or any(not math.isfinite(value) or value < 0 for value in self.candidate_sse.values())
            or not math.isfinite(self.source_energy)
            or self.source_energy <= 0
            or self.routed_occurrences <= 0
        ):
            raise ValueError("pooled portfolio evaluation is invalid")


@dataclass
class UpstreamFunctionalStatistics:
    """Per-neuron SiTU and downstream sensitivity over routed rows."""

    hidden_energy: torch.Tensor
    derivative_metric: torch.Tensor
    weight_sum: float
    weight_square_sum: float
    rows: int

    @property
    def effective_sample_size(self) -> float:
        if self.weight_sum <= 0 or self.weight_square_sum <= 0:
            return 0.0
        return self.weight_sum**2 / self.weight_square_sum


def blockwise_upstream_conditioning_coefficients(
    derivative_metric: torch.Tensor,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return conditional gate/up target coefficients from a PSD metric.

    The input contains one 2-by-2 gate/up metric per ordinary intermediate
    neuron.  Metrics are summed over contiguous neuron blocks before forming
    ``m13 / m11`` and ``m13 / m33``.  The returned vectors contain one
    coefficient per neuron and can therefore adjust ordinary-basis targets
    before the coupled activation-boundary transform is applied.
    """

    if (
        derivative_metric.ndim != 3
        or derivative_metric.shape[1:] != (2, 2)
        or not torch.is_floating_point(derivative_metric)
    ):
        raise ValueError("derivative metric must have shape [neurons, 2, 2]")
    neurons = int(derivative_metric.shape[0])
    if block_size <= 0 or neurons % block_size:
        raise ValueError("block size must divide the intermediate width")
    if not bool(torch.all(torch.isfinite(derivative_metric))):
        raise ValueError("derivative metric must be finite")
    metric = derivative_metric.double()
    metric = (metric + metric.transpose(1, 2)) * 0.5
    tolerance = 256 * torch.finfo(metric.dtype).eps * max(
        float(metric.abs().max()), 1.0
    )
    if float(torch.linalg.eigvalsh(metric).min()) < -tolerance:
        raise ValueError("derivative metric must be positive semidefinite")
    blocks = metric.reshape(neurons // block_size, block_size, 2, 2).sum(dim=1)
    maximum_diagonal = float(
        blocks.diagonal(dim1=1, dim2=2).max().clamp_min(0)
    )
    floor = max(maximum_diagonal * 1e-12, torch.finfo(metric.dtype).tiny)
    gate_from_up_blocks = blocks[:, 0, 1] / blocks[:, 0, 0].clamp_min(floor)
    up_from_gate_blocks = blocks[:, 0, 1] / blocks[:, 1, 1].clamp_min(floor)
    gate_from_up = gate_from_up_blocks.repeat_interleave(block_size)
    up_from_gate = up_from_gate_blocks.repeat_interleave(block_size)

    def evidence(prefix: str, values: torch.Tensor) -> dict[str, float]:
        quantiles = torch.quantile(
            values,
            torch.tensor((0.05, 0.5, 0.95), device=values.device, dtype=values.dtype),
        )
        return {
            f"{prefix}_minimum": float(values.min()),
            f"{prefix}_p05": float(quantiles[0]),
            f"{prefix}_median": float(quantiles[1]),
            f"{prefix}_p95": float(quantiles[2]),
            f"{prefix}_maximum": float(values.max()),
        }

    return (
        gate_from_up,
        up_from_gate,
        {
            "block_size": float(block_size),
            "blocks": float(blocks.shape[0]),
            "diagonal_floor": floor,
            **evidence("gate_from_up", gate_from_up_blocks),
            **evidence("up_from_gate", up_from_gate_blocks),
        },
    )


def collect_upstream_functional_statistics(
    batches: Iterable[Mapping[str, torch.Tensor]],
    *,
    source: CoupledTriplet,
    device: torch.device | str,
) -> UpstreamFunctionalStatistics:
    """Accumulate a PSD gate/up metric and post-SiTU neuron energy.

    The 2-by-2 matrix for each neuron is weighted by the squared applied route
    weight and by the squared norm of the corresponding W2 column.  The route
    weight therefore enters exactly once in the routed-output quadratic loss.
    """

    target = torch.device(device)
    source = source.to(device=target, dtype=torch.float32)
    hidden_energy = torch.zeros(source.intermediate, device=target, dtype=torch.float64)
    metric = torch.zeros(source.intermediate, 2, 2, device=target, dtype=torch.float64)
    down_energy = source.down.double().square().sum(dim=0)
    weight_sum = 0.0
    weight_square_sum = 0.0
    rows = 0
    previous_row = -1
    for batch in batches:
        required = {"input", "route_weight", "row_index"}
        if not required.issubset(batch):
            raise ValueError("functional-statistics batch lacks required tensors")
        inputs = batch["input"]
        route_weights = batch["route_weight"]
        row_indices = batch["row_index"]
        if (
            inputs.ndim != 2
            or inputs.shape[1] != source.hidden
            or route_weights.ndim != 1
            or row_indices.ndim != 1
            or inputs.shape[0] != route_weights.numel()
            or inputs.shape[0] != row_indices.numel()
        ):
            raise ValueError("functional-statistics batch tensors do not align")
        if inputs.shape[0] == 0:
            continue
        if row_indices.dtype != torch.int64:
            raise TypeError("global row indices must be int64")
        if int(row_indices[0]) <= previous_row or bool(torch.any(row_indices[1:] <= row_indices[:-1])):
            raise ValueError("functional-statistics rows must be globally increasing")
        previous_row = int(row_indices[-1])
        inputs = inputs.to(device=target, dtype=torch.float32)
        weights = route_weights.to(device=target, dtype=torch.float64).square()
        gate = F.linear(inputs, source.gate)
        up = F.linear(inputs, source.up)
        hidden = situ(gate, up).double()
        d_gate, d_up = situ_derivatives(gate, up)
        derivatives = torch.stack((d_gate, d_up), dim=2).double()
        hidden_energy.add_((hidden.square() * weights[:, None]).sum(dim=0))
        metric.add_(
            torch.einsum("r,rni,rnj,n->nij", weights, derivatives, derivatives, down_energy)
        )
        weight_sum += float(weights.sum())
        weight_square_sum += float(weights.square().sum())
        rows += int(inputs.shape[0])
    if rows == 0 or weight_sum <= 0:
        raise ValueError("functional statistics require routed support")
    return UpstreamFunctionalStatistics(
        hidden_energy=hidden_energy,
        derivative_metric=metric,
        weight_sum=weight_sum,
        weight_square_sum=weight_square_sum,
        rows=rows,
    )


def collect_coupled_hidden_statistics(
    batches: Iterable[Mapping[str, torch.Tensor]],
    *,
    source: CoupledTriplet,
    candidate_coordinates: CoupledTriplet,
    execution: CoupledHadamardExecution,
    statistics_dtype: torch.dtype = torch.float32,
    retain_source_gram: bool = False,
) -> CandidateHiddenStatistics:
    """Accumulate decoded-upstream statistics without evaluating W2.

    This is the reusable expensive part of conditional W2 construction.  One
    decoded gate/up candidate supplies an expert-local H2, the functional W2
    refit target, and the exact quadratic score for every subsequently encoded
    down-projection candidate.
    """

    if source.hidden != candidate_coordinates.hidden or source.intermediate != (
        candidate_coordinates.intermediate
    ):
        raise ValueError("source and candidate expert geometry differ")
    if execution.hidden != source.hidden or execution.intermediate != source.intermediate:
        raise ValueError("coupled execution geometry differs from the expert")
    if statistics_dtype not in (torch.float32, torch.float64):
        raise TypeError("statistics require FP32 or FP64")
    devices = {value.device for value in candidate_coordinates.tensors()}
    if len(devices) != 1:
        raise ValueError("candidate coordinate tensors must share one device")
    device = devices.pop()
    source = source.to(device=device, dtype=torch.float32)
    candidate_coordinates = candidate_coordinates.to(device=device, dtype=torch.float32)
    source_coordinates = CoupledTriplet(
        *encode_coupled_weights(source.tensors(), execution.spec)
    )
    statistics = CandidateHiddenStatistics.zeros(
        source.intermediate,
        device=device,
        dtype=statistics_dtype,
        retain_source_gram=retain_source_gram,
    )
    previous_row = -1
    for batch in batches:
        required = {"input", "route_weight", "row_index"}
        if not required.issubset(batch):
            raise ValueError("pooled expert batch lacks input, route weight, or row index")
        rows = batch["input"]
        route_weights = batch["route_weight"]
        row_indices = batch["row_index"]
        if (
            rows.ndim != 2
            or rows.shape[1] != source.hidden
            or route_weights.ndim != 1
            or row_indices.ndim != 1
            or route_weights.numel() != rows.shape[0]
            or row_indices.numel() != rows.shape[0]
        ):
            raise ValueError("pooled expert batch tensors do not align")
        if rows.shape[0] == 0:
            continue
        if row_indices.dtype != torch.int64:
            raise TypeError("global row indices must be int64")
        if int(row_indices[0]) <= previous_row or bool(
            torch.any(row_indices[1:] <= row_indices[:-1])
        ):
            raise ValueError("pooled expert rows must be globally increasing")
        previous_row = int(row_indices[-1])
        rows = rows.to(device=device, dtype=torch.float32)
        route_weights = route_weights.to(device=device, dtype=torch.float32)
        transformed_rows = execution.transform_inputs(rows)
        source_hidden = execution.decode_middle(
            transformed_rows,
            source_coordinates.gate,
            source_coordinates.up,
        )
        candidate_hidden = execution.decode_middle(
            transformed_rows,
            candidate_coordinates.gate,
            candidate_coordinates.up,
        )
        statistics.update(candidate_hidden, source_hidden, route_weights)
    if statistics.rows == 0:
        raise ValueError("pooled hidden statistics require routed support")
    return statistics


def evaluate_coupled_expert_batches(
    batches: Iterable[Mapping[str, torch.Tensor]],
    *,
    source: CoupledTriplet,
    teacher_source: CoupledTriplet | None = None,
    candidate_coordinates: CoupledTriplet,
    execution: CoupledHadamardExecution,
    prefix_row_limits: Iterable[int] = (),
    statistics_dtype: torch.dtype = torch.float32,
    retain_source_gram: bool = False,
) -> PooledExpertEvaluation:
    """Evaluate a coupled candidate over naturally routed row batches.

    ``source`` defines the exact coupled coordinate system and the W2
    sufficient-statistic target.  ``teacher_source`` optionally supplies an
    equivalent checkpoint-coordinate expert for the scoring reference; this
    is required when ``source`` has an exact neuron permutation or sign gauge.
    ``candidate_coordinates`` contains decoded weights in the coupled
    transform basis.  Each batch must
    contain ``input``, ``route_weight``, and monotonically increasing global
    ``row_index`` tensors.  Prefix limits refer to the all-row capture, not to
    the number of occurrences routed to this expert.
    """

    if source.hidden != candidate_coordinates.hidden or source.intermediate != (
        candidate_coordinates.intermediate
    ):
        raise ValueError("source and candidate expert geometry differ")
    reference = source if teacher_source is None else teacher_source
    if reference.hidden != source.hidden or reference.intermediate != source.intermediate:
        raise ValueError("teacher and coordinate-source expert geometry differ")
    if execution.hidden != source.hidden or execution.intermediate != source.intermediate:
        raise ValueError("coupled execution geometry differs from the expert")
    if statistics_dtype not in (torch.float32, torch.float64):
        raise TypeError("statistics require FP32 or FP64")
    limits = tuple(sorted(set(int(value) for value in prefix_row_limits)))
    if any(value <= 0 for value in limits):
        raise ValueError("prefix row limits must be positive")

    devices = {value.device for value in candidate_coordinates.tensors()}
    if len(devices) != 1:
        raise ValueError("candidate coordinate tensors must share one device")
    device = devices.pop()
    source = source.to(device=device, dtype=torch.float32)
    reference = reference.to(device=device, dtype=torch.float32)
    candidate_coordinates = candidate_coordinates.to(device=device, dtype=torch.float32)
    source_coordinates = CoupledTriplet(
        *encode_coupled_weights(source.tensors(), execution.spec)
    )
    statistics = CandidateHiddenStatistics.zeros(
        source.intermediate,
        device=device,
        dtype=statistics_dtype,
        retain_source_gram=retain_source_gram,
    )
    sse = 0.0
    source_energy = 0.0
    occurrences = 0
    prefix_accumulators = {
        limit: [0.0, 0.0, 0] for limit in limits
    }
    previous_row = -1

    for batch in batches:
        required = {"input", "route_weight", "row_index"}
        if not required.issubset(batch):
            raise ValueError("pooled expert batch lacks input, route weight, or row index")
        rows = batch["input"]
        route_weights = batch["route_weight"]
        row_indices = batch["row_index"]
        if (
            rows.ndim != 2
            or rows.shape[1] != source.hidden
            or route_weights.ndim != 1
            or row_indices.ndim != 1
            or route_weights.numel() != rows.shape[0]
            or row_indices.numel() != rows.shape[0]
        ):
            raise ValueError("pooled expert batch tensors do not align")
        if rows.shape[0] == 0:
            continue
        if row_indices.dtype != torch.int64:
            raise TypeError("global row indices must be int64")
        if int(row_indices[0]) <= previous_row or bool(
            torch.any(row_indices[1:] <= row_indices[:-1])
        ):
            raise ValueError("pooled expert rows must be globally increasing")
        previous_row = int(row_indices[-1])
        rows = rows.to(device=device, dtype=torch.float32)
        route_weights = route_weights.to(device=device, dtype=torch.float32)
        transformed_rows = execution.transform_inputs(rows)
        source_hidden = execution.decode_middle(
            transformed_rows,
            source_coordinates.gate,
            source_coordinates.up,
        )
        candidate_hidden = execution.decode_middle(
            transformed_rows,
            candidate_coordinates.gate,
            candidate_coordinates.up,
        )
        # Use the ordinary checkpoint-coordinate expert as the scoring
        # reference.  The transformed source hidden rows remain necessary for
        # the candidate-specific W2 sufficient statistics, but roundoff in the
        # exact coupled change of basis must not redefine the teacher output.
        ordinary_source_hidden = situ(
            F.linear(rows, reference.gate), F.linear(rows, reference.up)
        )
        source_output = F.linear(ordinary_source_hidden, reference.down)
        candidate_output = execution.decode_output(
            candidate_hidden @ candidate_coordinates.down.T
        )
        statistics.update(candidate_hidden, source_hidden, route_weights)

        routed_source = source_output * route_weights[:, None]
        routed_error = (candidate_output - source_output) * route_weights[:, None]
        row_sse = routed_error.square().sum(dim=1, dtype=torch.float64).cpu()
        row_energy = routed_source.square().sum(dim=1, dtype=torch.float64).cpu()
        sse += float(row_sse.sum())
        source_energy += float(row_energy.sum())
        occurrences += int(rows.shape[0])
        cpu_indices = row_indices.cpu()
        for limit, accumulator in prefix_accumulators.items():
            count = int(torch.searchsorted(cpu_indices, limit, right=False))
            if count:
                accumulator[0] += float(row_sse[:count].sum())
                accumulator[1] += float(row_energy[:count].sum())
                accumulator[2] += count

    if occurrences == 0 or source_energy <= 0:
        raise ValueError("pooled expert evaluation requires routed support")
    return PooledExpertEvaluation(
        statistics=statistics,
        sse=sse,
        source_energy=source_energy,
        routed_occurrences=occurrences,
        prefix_scores={
            limit: (float(values[0]), float(values[1]), int(values[2]))
            for limit, values in prefix_accumulators.items()
        },
    )


def evaluate_coupled_candidate_portfolio_batches(
    batches: Iterable[Mapping[str, torch.Tensor]],
    *,
    source: CoupledTriplet,
    candidate_coordinates: Mapping[str, CoupledTriplet],
    executions: Mapping[str, CoupledHadamardExecution],
) -> PooledPortfolioEvaluation:
    """Score a coupled candidate portfolio with one shared source forward.

    Every candidate may use a different intermediate transform draw.  The
    coupled profile keeps the residual-side transform fixed, so transformed
    inputs and the output inverse are shared without changing any numerical
    result.
    """

    names = tuple(candidate_coordinates)
    if not names or set(executions) != set(names):
        raise ValueError("candidate coordinates and executions must share identities")
    if len(set(names)) != len(names):
        raise ValueError("pooled portfolio candidate names must be unique")
    devices = {
        tensor.device
        for candidate in candidate_coordinates.values()
        for tensor in candidate.tensors()
    }
    if len(devices) != 1:
        raise ValueError("pooled portfolio candidates must share one device")
    device = devices.pop()
    source = source.to(device=device, dtype=torch.float32)
    candidates = {
        name: candidate.to(device=device, dtype=torch.float32)
        for name, candidate in candidate_coordinates.items()
    }
    first_execution = executions[names[0]]
    if first_execution.hidden != source.hidden or first_execution.intermediate != source.intermediate:
        raise ValueError("pooled portfolio execution geometry differs from the source")
    residual_contract = (
        first_execution.spec.residual_block_size,
        first_execution.spec.residual_draw,
    )
    for name in names:
        candidate = candidates[name]
        execution = executions[name]
        if (
            candidate.hidden != source.hidden
            or candidate.intermediate != source.intermediate
            or execution.hidden != source.hidden
            or execution.intermediate != source.intermediate
            or (
                execution.spec.residual_block_size,
                execution.spec.residual_draw,
            )
            != residual_contract
        ):
            raise ValueError("pooled portfolio candidates do not share residual geometry")

    sse = {name: 0.0 for name in names}
    source_energy = 0.0
    occurrences = 0
    previous_row = -1
    for batch in batches:
        required = {"input", "route_weight", "row_index"}
        if not required.issubset(batch):
            raise ValueError("pooled expert batch lacks input, route weight, or row index")
        rows = batch["input"]
        route_weights = batch["route_weight"]
        row_indices = batch["row_index"]
        if (
            rows.ndim != 2
            or rows.shape[1] != source.hidden
            or route_weights.ndim != 1
            or row_indices.ndim != 1
            or route_weights.numel() != rows.shape[0]
            or row_indices.numel() != rows.shape[0]
        ):
            raise ValueError("pooled portfolio batch tensors do not align")
        if rows.shape[0] == 0:
            continue
        if row_indices.dtype != torch.int64:
            raise TypeError("global row indices must be int64")
        if int(row_indices[0]) <= previous_row or bool(
            torch.any(row_indices[1:] <= row_indices[:-1])
        ):
            raise ValueError("pooled expert rows must be globally increasing")
        previous_row = int(row_indices[-1])
        rows = rows.to(device=device, dtype=torch.float32)
        route_weights = route_weights.to(device=device, dtype=torch.float32)
        source_hidden = situ(F.linear(rows, source.gate), F.linear(rows, source.up))
        source_output = F.linear(source_hidden, source.down)
        transformed_rows = first_execution.transform_inputs(rows)
        routed_source = source_output * route_weights[:, None]
        source_energy += float(
            routed_source.square().sum(dtype=torch.float64).cpu()
        )
        for name in names:
            candidate = candidates[name]
            execution = executions[name]
            candidate_hidden = execution.decode_middle(
                transformed_rows, candidate.gate, candidate.up
            )
            candidate_output = execution.decode_output(
                candidate_hidden @ candidate.down.T
            )
            routed_error = (candidate_output - source_output) * route_weights[:, None]
            sse[name] += float(routed_error.square().sum(dtype=torch.float64).cpu())
        occurrences += int(rows.shape[0])
    return PooledPortfolioEvaluation(
        candidate_sse=sse,
        source_energy=source_energy,
        routed_occurrences=occurrences,
    )


def decoded_down_sse(
    statistics: CandidateHiddenStatistics,
    candidate_down_t: torch.Tensor,
    source_down_t: torch.Tensor,
) -> torch.Tensor:
    """Evaluate exact pooled SSE from cross-products.

    ``candidate_down_t`` and ``source_down_t`` use ``[intermediate, output]``
    orientation.  The result equals the explicit routed-output SSE for the rows
    used to construct ``statistics`` up to floating-point accumulation error.
    """

    statistics._validate()
    if statistics.source_gram is None:
        raise ValueError("decoded down SSE requires retained source Gram statistics")
    if candidate_down_t.ndim != 2:
        raise ValueError("candidate down projection must be two-dimensional")
    expected = (statistics.width, candidate_down_t.shape[1])
    if tuple(candidate_down_t.shape) != expected:
        raise ValueError("candidate down projection has the wrong shape")
    if source_down_t.shape != candidate_down_t.shape:
        raise ValueError("source and candidate down projections do not align")
    dtype = statistics.candidate_gram.dtype
    device = statistics.candidate_gram.device
    candidate = candidate_down_t.to(device=device, dtype=dtype)
    source = source_down_t.to(device=device, dtype=dtype)
    # Evaluate around the source down projection.  Expanding the three
    # absolute output-energy terms directly loses precision when the decoded
    # candidate is close to the source.  This residual-coordinate form is
    # algebraically identical but every contraction is at the error scale.
    delta = candidate - source
    if (
        statistics.hidden_residual_gram is None
    ):
        raise AssertionError("decoded down SSE lost residual statistics")
    delta_term = torch.sum(delta * (statistics.candidate_gram @ delta))
    cross_term = torch.sum(
        delta * (statistics.candidate_residual_cross @ source)
    )
    source_residual_term = torch.sum(
        source * (statistics.hidden_residual_gram @ source)
    )
    value = delta_term + 2.0 * cross_term + source_residual_term
    tolerance = 64 * torch.finfo(dtype).eps * max(
        float(delta_term.abs()),
        float((2.0 * cross_term).abs()),
        float(source_residual_term.abs()),
        1.0,
    )
    if float(value) < -tolerance:
        raise ArithmeticError(
            f"pooled SSE is negative beyond roundoff: {float(value):.9g}"
        )
    return value.clamp_min(0)


def decoded_down_sse_difference(
    statistics: CandidateHiddenStatistics,
    candidate_down_t: torch.Tensor,
    baseline_down_t: torch.Tensor,
    source_down_t: torch.Tensor,
) -> torch.Tensor:
    """Evaluate candidate-minus-baseline SSE without a source Gram matrix."""

    statistics._validate()
    if candidate_down_t.ndim != 2:
        raise ValueError("candidate down projection must be two-dimensional")
    expected = (statistics.width, candidate_down_t.shape[1])
    if (
        tuple(candidate_down_t.shape) != expected
        or baseline_down_t.shape != candidate_down_t.shape
        or source_down_t.shape != candidate_down_t.shape
    ):
        raise ValueError("down projections do not share the expected shape")
    dtype = statistics.candidate_gram.dtype
    device = statistics.candidate_gram.device
    candidate = candidate_down_t.to(device=device, dtype=dtype)
    baseline = baseline_down_t.to(device=device, dtype=dtype)
    source = source_down_t.to(device=device, dtype=dtype)

    delta = candidate - baseline
    baseline_gradient = (
        statistics.candidate_gram @ (baseline - source)
        + statistics.candidate_residual_cross @ source
    )
    return 2.0 * torch.sum(delta * baseline_gradient) + torch.sum(
        delta * (statistics.candidate_gram @ delta)
    )


def candidate_h2(
    statistics: CandidateHiddenStatistics,
    *,
    max_local_alpha: float = 0.75,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build an OAS-shrunk candidate-local covariance.

    The prior is ``trace(H) / dimension * I``.  No pooled post-activation
    covariance participates because intermediate coordinates are expert-local.
    """

    statistics._validate()
    max_local_alpha = float(max_local_alpha)
    if not math.isfinite(max_local_alpha) or not 0 <= max_local_alpha <= 1:
        raise ValueError("max_local_alpha must be finite and in [0, 1]")
    if statistics.weight_sum <= 0 or statistics.effective_sample_size <= 0:
        raise ValueError("candidate H2 requires positive weighted support")
    covariance = statistics.candidate_gram / statistics.weight_sum
    covariance = (covariance + covariance.T) * 0.5
    dimension = statistics.width
    trace = torch.trace(covariance)
    if float(trace) <= 0:
        raise ValueError("candidate covariance must have positive trace")
    identity_scale = trace / dimension
    trace_square = trace.square()
    frobenius_square = covariance.square().sum()
    denominator = (
        statistics.effective_sample_size + 1.0 - 2.0 / dimension
    ) * (frobenius_square - trace_square / dimension)
    if float(denominator) <= 0:
        shrinkage = 1.0
    else:
        shrinkage = min(
            1.0,
            max(
                0.0,
                float(
                    (
                        (1.0 - 2.0 / dimension) * frobenius_square
                        + trace_square
                    )
                    / denominator
                ),
            ),
        )
    local_alpha = min(max_local_alpha, 1.0 - shrinkage)
    identity = torch.eye(
        dimension,
        dtype=covariance.dtype,
        device=covariance.device,
    ).mul_(identity_scale)
    result = torch.lerp(identity, covariance, local_alpha)
    return result.contiguous(), {
        "rows": float(statistics.rows),
        "weight_sum": statistics.weight_sum,
        "effective_sample_size": statistics.effective_sample_size,
        "oas_shrinkage": shrinkage,
        "local_alpha": local_alpha,
        "max_local_alpha": max_local_alpha,
        "identity_scale": float(identity_scale),
    }


def ridge_refit_down_from_statistics(
    statistics: CandidateHiddenStatistics,
    source_down_t: torch.Tensor,
    *,
    regularization_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Solve the regularized functional W2 target from pooled statistics."""

    statistics._validate()
    ratio = float(regularization_ratio)
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("regularization_ratio must be finite and positive")
    if source_down_t.ndim != 2 or source_down_t.shape[0] != statistics.width:
        raise ValueError("source down projection does not match candidate width")
    dtype = statistics.candidate_gram.dtype
    device = statistics.candidate_gram.device
    source = source_down_t.to(device=device, dtype=dtype)
    scale = float(torch.trace(statistics.candidate_gram)) / statistics.width
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("candidate Gram matrix must have positive trace")
    regularization = ratio * scale
    system = statistics.candidate_gram.clone()
    system.diagonal().add_(regularization)
    rhs = statistics.candidate_source_cross @ source + regularization * source
    refit = torch.linalg.solve(system, rhs)
    return refit.contiguous(), {
        "regularization_ratio": ratio,
        "regularization": regularization,
        "gram_trace_per_dimension": scale,
    }


__all__ = [
    "CandidateHiddenStatistics",
    "PooledExpertEvaluation",
    "PooledPortfolioEvaluation",
    "UpstreamFunctionalStatistics",
    "candidate_h2",
    "blockwise_upstream_conditioning_coefficients",
    "collect_coupled_hidden_statistics",
    "collect_upstream_functional_statistics",
    "decoded_down_sse",
    "decoded_down_sse_difference",
    "evaluate_coupled_expert_batches",
    "evaluate_coupled_candidate_portfolio_batches",
    "ridge_refit_down_from_statistics",
]

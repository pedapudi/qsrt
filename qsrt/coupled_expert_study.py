"""Generic PTQ research helpers for coupled gated-MoE expert triplets.

The production QSRT codec deliberately treats model architecture and storage
as adapter-owned concerns.  This module follows the same boundary: it knows
only that one expert is a coupled ``(gate, up, down)`` triplet, that the two
upstream projections feed a coordinatewise activation, and that routed expert
errors are aggregated before an optional output metric.

The routines here are research oracles and diagnostics.  They do not define a
checkpoint schema or a serving profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class CoupledTriplet:
    """One gated expert in ordinary framework ``[out, in]`` orientation."""

    gate: Tensor
    up: Tensor
    down: Tensor

    def __post_init__(self) -> None:
        if any(value.ndim != 2 for value in (self.gate, self.up, self.down)):
            raise ValueError("coupled expert weights must be rank-two tensors")
        if self.gate.shape != self.up.shape:
            raise ValueError("gate and up projections must have identical shapes")
        intermediate, hidden = self.gate.shape
        if tuple(self.down.shape) != (hidden, intermediate):
            raise ValueError("down must have shape [hidden, intermediate]")
        if not all(torch.is_floating_point(value) for value in self.tensors()):
            raise TypeError("coupled expert weights must be floating point")

    @property
    def intermediate(self) -> int:
        return int(self.gate.shape[0])

    @property
    def hidden(self) -> int:
        return int(self.gate.shape[1])

    @property
    def numel(self) -> int:
        return sum(value.numel() for value in self.tensors())

    def tensors(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.gate, self.up, self.down

    def to(self, *args: object, **kwargs: object) -> "CoupledTriplet":
        return CoupledTriplet(
            *(value.to(*args, **kwargs) for value in self.tensors())
        )


class CoupledTripletSource(Protocol):
    """Minimal stream interface used by cross-expert experiments."""

    def load_triplet(self, expert: int) -> CoupledTriplet: ...


@dataclass(frozen=True)
class ActivationLaw:
    """Coordinatewise two-input activation and its first derivatives."""

    value: Callable[[Tensor, Tensor], Tensor]
    derivatives: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]


@dataclass(frozen=True)
class SiTUComponentGeometry:
    """Dimensionless SiTU inputs, factors, and component derivatives."""

    normalized_gate: Tensor
    normalized_up: Tensor
    gate_factor: Tensor
    up_factor: Tensor
    gate_derivative: Tensor
    up_derivative: Tensor


@dataclass(frozen=True)
class RoutedOutputMetric:
    """RMS normalization followed by a gain and output projection."""

    gain: Tensor
    projection: Tensor
    epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if self.gain.ndim != 1 or self.projection.ndim != 2:
            raise ValueError("output metric expects a gain vector and matrix")
        if self.projection.shape[1] != self.gain.numel():
            raise ValueError("output projection input does not match gain")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("RMS epsilon must be finite and positive")

    def output(self, aggregate: Tensor) -> Tensor:
        if aggregate.ndim != 2 or aggregate.shape[1] != self.gain.numel():
            raise ValueError("aggregate has the wrong output-metric shape")
        rms = torch.sqrt(aggregate.float().square().mean(dim=1, keepdim=True) + self.epsilon)
        normalized = aggregate.float() / rms
        return (normalized * self.gain.float()) @ self.projection.float().T

    def jacobian_vectors(self, aggregate: Tensor, vectors: Tensor) -> Tensor:
        """Apply the exact RMSNorm Jacobian and the fixed output projection.

        ``vectors`` may be ``[rows, hidden]`` or ``[rows, items, hidden]``.
        """

        if aggregate.ndim != 2 or aggregate.shape[1] != self.gain.numel():
            raise ValueError("aggregate has the wrong output-metric shape")
        if vectors.ndim not in (2, 3):
            raise ValueError("metric vectors must have rank two or three")
        if vectors.shape[0] != aggregate.shape[0] or vectors.shape[-1] != aggregate.shape[1]:
            raise ValueError("metric vectors do not align with the aggregate")
        u = aggregate.float()
        v = vectors.float()
        d = u.shape[1]
        rms = torch.sqrt(u.square().mean(dim=1) + self.epsilon)
        if v.ndim == 2:
            dot = (u * v).sum(dim=1)
            tangent = v / rms[:, None] - u * (dot / (d * rms.pow(3)))[:, None]
            gained = tangent * self.gain.float()[None, :]
            return gained @ self.projection.float().T
        dot = (u[:, None, :] * v).sum(dim=2)
        tangent = v / rms[:, None, None] - u[:, None, :] * (
            dot / (d * rms.pow(3)[:, None])
        )[:, :, None]
        gained = tangent * self.gain.float()[None, None, :]
        return torch.einsum("rqh,oh->rqo", gained, self.projection.float())

    def exact_delta(self, aggregate: Tensor, error: Tensor) -> Tensor:
        if aggregate.shape != error.shape:
            raise ValueError("aggregate and error must have identical shapes")
        return self.output(aggregate + error) - self.output(aggregate)


def situ_value(gate: Tensor, up: Tensor, *, beta: float = 4.0, linear_beta: float = 25.0) -> Tensor:
    """Kimi's SiTU activation in a model-independent helper."""

    gate_f = gate.float()
    up_f = up.float()
    return (
        beta
        * torch.tanh(gate_f / beta)
        * torch.sigmoid(gate_f)
        * linear_beta
        * torch.tanh(up_f / linear_beta)
    )


def situ_derivatives(
    gate: Tensor,
    up: Tensor,
    *,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> tuple[Tensor, Tensor]:
    """Return derivatives of SiTU with respect to gate and up inputs."""

    a = gate.float()
    b = up.float()
    tanh_gate = torch.tanh(a / beta)
    sigmoid = torch.sigmoid(a)
    f = beta * tanh_gate * sigmoid
    g = linear_beta * torch.tanh(b / linear_beta)
    f_prime = (1.0 - tanh_gate.square()) * sigmoid + beta * tanh_gate * sigmoid * (1.0 - sigmoid)
    g_prime = 1.0 - torch.tanh(b / linear_beta).square()
    return g * f_prime, f * g_prime


def situ_component_geometry(
    gate: Tensor,
    up: Tensor,
    *,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> SiTUComponentGeometry:
    """Expose the two SiTU factors before their product.

    ``situ_derivatives`` returns the derivatives of the complete product,
    which intentionally fold the opposite factor into each result.  This
    diagnostic instead returns ``f``, ``g``, ``f'``, and ``g'`` separately so
    activation-regime studies can determine whether the nominally linear up
    branch is actually operating near ``g'(u) = 1``.
    """

    if gate.shape != up.shape:
        raise ValueError("gate and up projections must have identical shapes")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("gate temperature must be finite and positive")
    if not math.isfinite(linear_beta) or linear_beta <= 0:
        raise ValueError("up temperature must be finite and positive")
    gate_f = gate.float()
    up_f = up.float()
    normalized_gate = gate_f / beta
    normalized_up = up_f / linear_beta
    tanh_gate = torch.tanh(normalized_gate)
    tanh_up = torch.tanh(normalized_up)
    sigmoid_gate = torch.sigmoid(gate_f)
    gate_factor = beta * tanh_gate * sigmoid_gate
    up_factor = linear_beta * tanh_up
    gate_derivative = (
        (1.0 - tanh_gate.square()) * sigmoid_gate
        + beta * tanh_gate * sigmoid_gate * (1.0 - sigmoid_gate)
    )
    up_derivative = 1.0 - tanh_up.square()
    return SiTUComponentGeometry(
        normalized_gate=normalized_gate,
        normalized_up=normalized_up,
        gate_factor=gate_factor,
        up_factor=up_factor,
        gate_derivative=gate_derivative,
        up_derivative=up_derivative,
    )


SITU = ActivationLaw(value=situ_value, derivatives=situ_derivatives)


def expert_hidden(inputs: Tensor, triplet: CoupledTriplet, activation: ActivationLaw = SITU) -> Tensor:
    gate = inputs.float() @ triplet.gate.float().T
    up = inputs.float() @ triplet.up.float().T
    return activation.value(gate, up)


def expert_output(inputs: Tensor, triplet: CoupledTriplet, activation: ActivationLaw = SITU) -> Tensor:
    return expert_hidden(inputs, triplet, activation) @ triplet.down.float().T


@dataclass(frozen=True)
class PairMetricSummary:
    metric: Tensor
    small_eigenvalue_fraction: Tensor
    condition_number: Tensor


def pair_activation_metric(
    gate_projection: Tensor,
    up_projection: Tensor,
    *,
    row_weights: Tensor | None = None,
    activation: ActivationLaw = SITU,
) -> PairMetricSummary:
    """Build the route-weighted 2x2 local W1/W3 activation metric."""

    if gate_projection.shape != up_projection.shape or gate_projection.ndim != 2:
        raise ValueError("gate/up projections must be aligned rank-two tensors")
    rows, _ = gate_projection.shape
    weights = (
        torch.ones(rows, dtype=torch.float64, device=gate_projection.device)
        if row_weights is None
        else row_weights.to(device=gate_projection.device, dtype=torch.float64)
    )
    if weights.ndim != 1 or weights.numel() != rows or bool(torch.any(weights < 0)):
        raise ValueError("pair metric weights must be one non-negative value per row")
    a, b = activation.derivatives(gate_projection, up_projection)
    a = a.double()
    b = b.double()
    normalizer = weights.sum().clamp_min(torch.finfo(torch.float64).tiny)
    m11 = (weights[:, None] * a.square()).sum(dim=0) / normalizer
    m12 = (weights[:, None] * a * b).sum(dim=0) / normalizer
    m22 = (weights[:, None] * b.square()).sum(dim=0) / normalizer
    metric = torch.stack(
        (torch.stack((m11, m12), dim=1), torch.stack((m12, m22), dim=1)),
        dim=1,
    )
    eigenvalues = torch.linalg.eigvalsh(metric).clamp_min(0)
    trace = eigenvalues.sum(dim=1)
    small_fraction = torch.where(trace > 0, eigenvalues[:, 0] / trace, torch.zeros_like(trace))
    condition = torch.where(
        eigenvalues[:, 0] > 0,
        eigenvalues[:, 1] / eigenvalues[:, 0],
        torch.full_like(trace, float("inf")),
    )
    return PairMetricSummary(metric=metric, small_eigenvalue_fraction=small_fraction, condition_number=condition)


def pair_residual_decomposition(
    inputs: Tensor,
    source: CoupledTriplet,
    candidate: CoupledTriplet,
    *,
    route_gates: Tensor | None = None,
    activation: ActivationLaw = SITU,
) -> dict[str, float]:
    """Decompose actual W1/W3 residuals into linear terms and their cross term."""

    if source.gate.shape != candidate.gate.shape or source.up.shape != candidate.up.shape:
        raise ValueError("source and candidate upstream weights do not align")
    x = inputs.float()
    gate = x @ source.gate.float().T
    up = x @ source.up.float().T
    delta_gate = x @ (candidate.gate.float() - source.gate.float()).T
    delta_up = x @ (candidate.up.float() - source.up.float()).T
    d_gate, d_up = activation.derivatives(gate, up)
    term_gate = d_gate * delta_gate
    term_up = d_up * delta_up
    exact = activation.value(gate + delta_gate, up + delta_up) - activation.value(gate, up)
    weights = (
        torch.ones(x.shape[0], device=x.device)
        if route_gates is None
        else route_gates.float().to(x.device).square()
    )[:, None]
    gate_energy = (weights * term_gate.square()).double().sum()
    up_energy = (weights * term_up.square()).double().sum()
    cross = (weights * (2.0 * term_gate * term_up)).double().sum()
    linear = term_gate + term_up
    linear_energy = (weights * linear.square()).double().sum()
    exact_energy = (weights * exact.square()).double().sum()
    mismatch = (weights * (linear - exact).square()).double().sum()
    separate = gate_energy + up_energy
    return {
        "gate_linear_sse": float(gate_energy),
        "up_linear_sse": float(up_energy),
        "separate_linear_sse": float(separate),
        "cross_term": float(cross),
        "joint_linear_sse": float(linear_energy),
        "exact_hidden_sse": float(exact_energy),
        "linearization_mismatch_sse": float(mismatch),
        "joint_over_separate": float(linear_energy / separate) if separate > 0 else 0.0,
        "linear_over_exact": float(linear_energy / exact_energy) if exact_energy > 0 else 0.0,
    }


def local_triplet_metrics(
    inputs: Tensor,
    source: CoupledTriplet,
    *,
    route_gates: Tensor,
    aggregate: Tensor,
    output_metric: RoutedOutputMetric,
    neuron_indices: Tensor,
    coordinate_indices: Tensor,
    activation: ActivationLaw = SITU,
) -> Tensor:
    """Return one exact first-order 3x3 functional metric per ``(j, k)``.

    The three perturbations are ``W1[j,k]``, ``W3[j,k]``, and ``W2[k,j]``.
    This intentionally samples local coordinates rather than constructing the
    impossible dense Hessian of an entire expert.
    """

    if neuron_indices.ndim != 1 or coordinate_indices.ndim != 1:
        raise ValueError("triplet metric indices must be vectors")
    if neuron_indices.numel() != coordinate_indices.numel():
        raise ValueError("triplet metric index vectors must have equal length")
    if route_gates.ndim != 1 or route_gates.numel() != inputs.shape[0]:
        raise ValueError("route gates must align with inputs")
    if aggregate.shape != (inputs.shape[0], source.hidden):
        raise ValueError("aggregate does not align with expert inputs")
    j_values = neuron_indices.to(dtype=torch.long, device=inputs.device)
    k_values = coordinate_indices.to(dtype=torch.long, device=inputs.device)
    if bool(torch.any((j_values < 0) | (j_values >= source.intermediate))):
        raise ValueError("neuron index is out of range")
    if bool(torch.any((k_values < 0) | (k_values >= source.hidden))):
        raise ValueError("coordinate index is out of range")

    x = inputs.float()
    gate_weight = source.gate.float().index_select(0, j_values)
    up_weight = source.up.float().index_select(0, j_values)
    gate = x @ gate_weight.T
    up = x @ up_weight.T
    hidden = activation.value(gate, up)
    d_gate, d_up = activation.derivatives(gate, up)
    xk = x.index_select(1, k_values)
    down_columns = source.down.float().index_select(1, j_values).T
    routed = route_gates.float().to(x.device)
    u = aggregate.float()
    width = u.shape[1]
    rms = torch.sqrt(u.square().mean(dim=1) + output_metric.epsilon)
    projection = output_metric.projection.float().to(x.device)
    gain = output_metric.gain.float().to(x.device)
    projected_u = (u * gain[None, :]) @ projection.T
    projected_columns = (down_columns * gain[None, :]) @ projection.T
    projected_coordinates = (
        projection.index_select(1, k_values).T * gain.index_select(0, k_values)[:, None]
    )
    result = torch.empty((j_values.numel(), 3, 3), dtype=torch.float64)
    for q in range(j_values.numel()):
        column = down_columns[q]
        factors = torch.stack(
            (d_gate[:, q] * xk[:, q], d_up[:, q] * xk[:, q]), dim=1
        )
        column_dot = u @ column
        column_mapped = (
            factors[:, :, None] * projected_columns[q][None, None, :] / rms[:, None, None]
            - projected_u[:, None, :]
            * (
                factors
                * column_dot[:, None]
                / (width * rms.pow(3))[:, None]
            )[:, :, None]
        )
        hidden_factor = hidden[:, q]
        coordinate = int(k_values[q])
        coordinate_mapped = (
            hidden_factor[:, None] * projected_coordinates[q][None, :] / rms[:, None]
            - projected_u
            * (
                hidden_factor
                * u[:, coordinate]
                / (width * rms.pow(3))
            )[:, None]
        )
        mapped = torch.cat((column_mapped, coordinate_mapped[:, None, :]), dim=1)
        mapped = mapped.double() * routed[:, None, None].double()
        result[q] = torch.einsum("rqi,rsi->qs", mapped, mapped).cpu() / max(inputs.shape[0], 1)
    return result


def radial_tangent_decomposition(aggregate: Tensor, error: Tensor) -> dict[str, Tensor]:
    """Split errors into components parallel and orthogonal to the aggregate."""

    if aggregate.shape != error.shape or aggregate.ndim != 2:
        raise ValueError("aggregate and error must be aligned matrices")
    u = aggregate.float()
    e = error.float()
    denominator = u.square().sum(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    radial = u * ((u * e).sum(dim=1, keepdim=True) / denominator)
    tangent = e - radial
    total_energy = e.square().sum(dim=1)
    radial_energy = radial.square().sum(dim=1)
    tangent_energy = tangent.square().sum(dim=1)
    return {
        "radial": radial,
        "tangent": tangent,
        "total_energy": total_energy,
        "radial_energy": radial_energy,
        "tangent_energy": tangent_energy,
        "radial_fraction": torch.where(total_energy > 0, radial_energy / total_energy, torch.zeros_like(total_energy)),
    }


def route_error_covariance(
    aggregate: Tensor,
    expert_errors: Tensor,
    route_gates: Tensor,
    output_metric: RoutedOutputMetric,
) -> dict[str, float]:
    """Decompose top-k tangent error into diagonal and cross-expert terms."""

    if expert_errors.ndim != 3:
        raise ValueError("expert errors must have shape [rows, topk, hidden]")
    if route_gates.shape != expert_errors.shape[:2]:
        raise ValueError("route gates do not align with expert errors")
    if aggregate.shape != (expert_errors.shape[0], expert_errors.shape[2]):
        raise ValueError("aggregate does not align with expert errors")
    routed = expert_errors.float() * route_gates.float()[:, :, None]
    mapped = output_metric.jacobian_vectors(aggregate, routed).double()
    diagonal = mapped.square().sum()
    summed = mapped.sum(dim=1)
    total = summed.square().sum()
    cross = total - diagonal
    mean = summed.mean(dim=0)
    bias = mean.square().sum() * summed.shape[0]
    return {
        "diagonal_sse": float(diagonal),
        "cross_term": float(cross),
        "total_sse": float(total),
        "bias_sse": float(bias),
        "total_over_diagonal": float(total / diagonal) if diagonal > 0 else 0.0,
        "independent_averaging_factor": float(diagonal / total) if total > 0 else float("inf"),
    }


def select_corouted_candidate_modes(
    expert_ids: Tensor,
    candidate_errors: Tensor,
    *,
    valid_modes: Tensor | None = None,
    unary_relative_slack: float | None = None,
    maximum_sweeps: int = 20,
) -> dict[str, Tensor | float | int]:
    """Select one candidate per expert against the routed mixture error.

    ``candidate_errors`` has shape ``[rows, slots, modes, features]`` and is
    expected to contain route-gate weighting and any desired output-metric
    projection already.  The optimized objective is therefore

    ``sum_rows ||sum_slots error[row, slot, mode[expert]])||^2``.

    This deterministic coordinate solver is an analysis tool.  It exposes the
    headroom from choosing among several valid, nearly equal payloads without
    prescribing how those payloads are generated or stored.
    """

    if expert_ids.ndim != 2 or candidate_errors.ndim != 4:
        raise ValueError("co-routing inputs must have rank two and four")
    if tuple(candidate_errors.shape[:2]) != tuple(expert_ids.shape):
        raise ValueError("candidate errors do not align with routed expert slots")
    if candidate_errors.shape[2] < 2 or candidate_errors.shape[3] == 0:
        raise ValueError("co-routing selection requires multiple nonempty candidates")
    if expert_ids.numel() == 0 or bool(torch.any(expert_ids < 0)):
        raise ValueError("routed expert identifiers must be nonempty and non-negative")
    if not torch.is_floating_point(candidate_errors) or not bool(
        torch.all(torch.isfinite(candidate_errors))
    ):
        raise ValueError("candidate errors must be finite floating-point values")
    if maximum_sweeps <= 0:
        raise ValueError("maximum sweeps must be positive")
    if unary_relative_slack is not None and (
        not math.isfinite(unary_relative_slack) or unary_relative_slack < 0
    ):
        raise ValueError("unary relative slack must be finite and non-negative")

    ids = expert_ids.detach().long().cpu()
    errors = candidate_errors.detach().double().cpu()
    modes = errors.shape[2]
    routed_experts = int(ids.max()) + 1
    experts = int(valid_modes.shape[0]) if valid_modes is not None else routed_experts
    if experts < routed_experts:
        raise ValueError("valid mode mask omits a routed expert")
    present = torch.zeros(experts, dtype=torch.bool)
    present[torch.unique(ids)] = True
    if valid_modes is None:
        allowed = torch.ones((experts, modes), dtype=torch.bool)
    else:
        if tuple(valid_modes.shape) != (experts, modes):
            raise ValueError("valid mode mask does not match expert and mode counts")
        allowed = valid_modes.detach().bool().cpu().clone()
    if bool(torch.any(present & ~allowed.any(dim=1))):
        raise ValueError("every routed expert must retain at least one valid candidate")

    unary = torch.zeros((experts, modes), dtype=torch.float64)
    contributions: dict[int, tuple[Tensor, Tensor]] = {}
    for expert in torch.nonzero(present, as_tuple=False).flatten().tolist():
        locations = torch.nonzero(ids == expert, as_tuple=False)
        rows = torch.unique(locations[:, 0], sorted=True)
        by_row = torch.zeros(
            (rows.numel(), modes, errors.shape[3]), dtype=torch.float64
        )
        row_lookup = torch.searchsorted(rows, locations[:, 0])
        for location, row_index in zip(locations, row_lookup):
            by_row[row_index] += errors[location[0], location[1]]
        unary[expert] = by_row.square().sum(dim=(0, 2))
        contributions[expert] = rows, by_row

    if unary_relative_slack is not None:
        masked = unary.masked_fill(~allowed, float("inf"))
        best = masked.min(dim=1).values
        threshold = best * (1.0 + unary_relative_slack) + 1e-30
        allowed &= unary <= threshold[:, None]
        if bool(torch.any(present & ~allowed.any(dim=1))):
            raise RuntimeError("unary filtering removed every candidate for an expert")

    unary_selection = torch.full((experts,), -1, dtype=torch.long)
    unary_selection[present] = unary.masked_fill(~allowed, float("inf")).argmin(
        dim=1
    )[present]

    def optimize(initial: Tensor) -> tuple[Tensor, Tensor, int, int]:
        selection = initial.clone()
        aggregate = torch.zeros(
            (ids.shape[0], errors.shape[3]), dtype=torch.float64
        )
        for expert, (rows, by_row) in contributions.items():
            aggregate[rows] += by_row[:, selection[expert]]
        changes = 0
        completed_sweeps = 0
        for completed_sweeps in range(1, maximum_sweeps + 1):
            sweep_changes = 0
            for expert, (rows, by_row) in contributions.items():
                current = int(selection[expert])
                base = aggregate[rows] - by_row[:, current]
                costs = (base[:, None, :] + by_row).square().sum(dim=(0, 2))
                costs.masked_fill_(~allowed[expert], float("inf"))
                winner = int(costs.argmin())
                if winner == current:
                    continue
                aggregate[rows] = base + by_row[:, winner]
                selection[expert] = winner
                sweep_changes += 1
            changes += sweep_changes
            if sweep_changes == 0:
                break
        return selection, aggregate, completed_sweeps, changes

    starts = [unary_selection]
    for mode in range(modes):
        uniform = unary_selection.clone()
        uniform[present & allowed[:, mode]] = mode
        starts.append(uniform)
    candidates = [optimize(start) for start in starts]
    selection, aggregate, completed_sweeps, changes = min(
        candidates, key=lambda item: float(item[1].square().sum())
    )

    selected_unary = float(
        unary.gather(1, selection.clamp_min(0)[:, None]).squeeze(1)[present].sum()
    )
    objective = float(aggregate.square().sum())
    return {
        "selection": selection,
        "objective": objective,
        "selected_unary": selected_unary,
        "cross_term": objective - selected_unary,
        "sweeps": completed_sweeps,
        "changes": changes,
        "experts": int(present.sum()),
        "modes": modes,
    }


def apply_permutation_sign_gauge(
    triplet: CoupledTriplet,
    permutation: Tensor,
    signs: Tensor,
) -> CoupledTriplet:
    """Apply the exact intermediate permutation and W3/W2 sign gauge."""

    p = permutation.to(dtype=torch.long, device=triplet.gate.device)
    s = signs.to(dtype=triplet.up.dtype, device=triplet.up.device)
    if p.ndim != 1 or p.numel() != triplet.intermediate:
        raise ValueError("permutation has the wrong length")
    if not torch.equal(torch.sort(p).values, torch.arange(triplet.intermediate, device=p.device)):
        raise ValueError("intermediate permutation must be bijective")
    if s.ndim != 1 or s.numel() != triplet.intermediate or not bool(torch.all(torch.abs(s) == 1)):
        raise ValueError("sign gauge must contain one +/-1 value per neuron")
    return CoupledTriplet(
        triplet.gate.index_select(0, p).contiguous(),
        (triplet.up.index_select(0, p) * s[:, None]).contiguous(),
        (triplet.down.index_select(1, p) * s[None, :]).contiguous(),
    )


def apply_common_input_gauge(triplet: CoupledTriplet, transform: Tensor) -> CoupledTriplet:
    """Return weights for inputs transformed as ``z' = z A.T``."""

    if transform.shape != (triplet.hidden, triplet.hidden):
        raise ValueError("input gauge has the wrong shape")
    inverse = torch.linalg.inv(transform.float())
    return CoupledTriplet(
        triplet.gate.float() @ inverse,
        triplet.up.float() @ inverse,
        triplet.down.float(),
    )


def apply_output_rotation(triplet: CoupledTriplet, rotation: Tensor) -> CoupledTriplet:
    if rotation.shape != (triplet.hidden, triplet.hidden):
        raise ValueError("output rotation has the wrong shape")
    closure = rotation.float() @ rotation.float().T
    identity = torch.eye(triplet.hidden, device=rotation.device)
    if not torch.allclose(closure, identity, rtol=2e-5, atol=2e-5):
        raise ValueError("output rotation must be orthogonal")
    return CoupledTriplet(triplet.gate, triplet.up, rotation.float() @ triplet.down.float())


def apply_postactivation_scale(triplet: CoupledTriplet, scale: Tensor) -> CoupledTriplet:
    if scale.ndim != 1 or scale.numel() != triplet.intermediate:
        raise ValueError("postactivation scale has the wrong shape")
    if not bool(torch.all(torch.isfinite(scale))) or bool(torch.any(scale == 0)):
        raise ValueError("postactivation scale must be finite and nonzero")
    return CoupledTriplet(
        triplet.gate,
        triplet.up,
        triplet.down / scale.to(triplet.down)[None, :],
    )


def encode_two_sided_linear(weight: Tensor, left: Tensor, right: Tensor) -> Tensor:
    """Encode ``weight`` with explicit orthogonal boundary transforms.

    The ordinary linear is ``y = x @ weight.T``.  With ``x_t = x @ right.T``
    and ``weight_t = left @ weight @ right.T``, the transformed result is
    ``y_t = y @ left.T`` and is recovered as ``y = y_t @ left``.

    This dense helper is an exact small-matrix oracle.  Production-width
    experiments should supply structured Hadamard/butterfly operators instead
    of materializing the transform matrices.
    """

    if weight.ndim != 2:
        raise ValueError("two-sided transform expects a rank-two weight")
    if left.shape != (weight.shape[0], weight.shape[0]):
        raise ValueError("left transform has the wrong shape")
    if right.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("right transform has the wrong shape")
    for name, value in (("left", left), ("right", right)):
        identity = torch.eye(value.shape[0], device=value.device, dtype=torch.float32)
        closure = value.float() @ value.float().T
        if not torch.allclose(closure, identity, rtol=2e-5, atol=2e-5):
            raise ValueError(f"{name} transform must be orthogonal")
    return (left.float() @ weight.float() @ right.float().T).contiguous()


def execute_two_sided_linear(
    inputs: Tensor,
    encoded_weight: Tensor,
    left: Tensor,
    right: Tensor,
) -> Tensor:
    """Execute and cancel an explicit two-sided linear transform."""

    if inputs.ndim != 2 or inputs.shape[1] != encoded_weight.shape[1]:
        raise ValueError("inputs do not align with the transformed weight")
    if left.shape != (encoded_weight.shape[0], encoded_weight.shape[0]):
        raise ValueError("left transform has the wrong shape")
    if right.shape != (encoded_weight.shape[1], encoded_weight.shape[1]):
        raise ValueError("right transform has the wrong shape")
    transformed_inputs = inputs.float() @ right.float().T
    transformed_outputs = transformed_inputs @ encoded_weight.float().T
    return transformed_outputs @ left.float()


def block_hadamard(values: Tensor, *, block_size: int, dim: int = -1) -> Tensor:
    """Apply a normalized block-diagonal Walsh-Hadamard transform.

    The operation is self-inverse and intentionally carries no model-specific
    geometry.  It is the CPU oracle for cheap explicit expert-boundary
    transforms; serving implementations would use their fused transform path.
    """

    if values.ndim == 0 or not torch.is_floating_point(values):
        raise TypeError("block Hadamard input must be a floating-point tensor")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("Hadamard block size must be a positive power of two")
    axis = dim % values.ndim
    if values.shape[axis] % block_size:
        raise ValueError("Hadamard axis must be divisible by the block size")
    # The butterfly below is intentionally in-place on its private work
    # buffer.  ``float().contiguous()`` may alias an already-contiguous FP32
    # caller tensor, so an explicit clone is required to keep this research
    # transform observationally pure.
    output = values.float().movedim(axis, -1).contiguous().clone()
    shape = output.shape
    output = output.reshape(*shape[:-1], shape[-1] // block_size, block_size)
    width = 1
    while width < block_size:
        paired = output.reshape(*output.shape[:-1], -1, 2, width)
        left = paired[..., 0, :].clone()
        right = paired[..., 1, :].clone()
        paired[..., 0, :] = left + right
        paired[..., 1, :] = left - right
        output = paired.reshape(*output.shape)
        width *= 2
    output = output.div_(math.sqrt(block_size)).reshape(shape).movedim(-1, axis)
    return output.contiguous()


def _rotation_signs(
    length: int,
    *,
    draw: int,
    axis: int,
    device: torch.device,
) -> Tensor:
    """Return deterministic Rademacher signs for one Hadamard boundary."""

    if draw < 0:
        raise ValueError("Hadamard rotation draw must be nonnegative")
    if draw == 0:
        return torch.ones(length, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        (0x6A09E667F3BCC909 * draw + 0xBB67AE8584CAA73B * axis)
        & ((1 << 63) - 1)
    )
    signs = torch.randint(0, 2, (length,), generator=generator)
    return signs.mul_(2).sub_(1).float().to(device=device)


def hadamard_rotation_signs(
    length: int,
    *,
    draw: int,
    axis: int,
    device: torch.device,
) -> Tensor:
    """Return the deterministic signs that identify one boundary rotation."""

    return _rotation_signs(
        length,
        draw=draw,
        axis=axis,
        device=device,
    )


def apply_w3_w2_sign_draw(
    triplet: CoupledTriplet,
    *,
    draw: int,
) -> CoupledTriplet:
    """Apply one deterministic exact W3/W2 sign-gauge representative.

    The Kimi up activation is odd, so multiplying one up row and the matching
    down column by the same sign preserves the full-precision expert exactly.
    Draw zero is identity. Nonzero draws require neither stored metadata nor
    runtime work once the signs are baked into the encoded weights.
    """

    signs = _rotation_signs(
        triplet.intermediate,
        draw=draw,
        axis=4,
        device=triplet.up.device,
    ).to(dtype=triplet.up.dtype)
    return CoupledTriplet(
        triplet.gate,
        (triplet.up * signs[:, None]).contiguous(),
        (triplet.down * signs[None, :]).contiguous(),
    )


def apply_w3_w2_scale_gauge(
    triplet: CoupledTriplet,
    *,
    policy: str,
    strength: float,
) -> CoupledTriplet:
    """Apply one bounded positive W3/W2 scale-gauge proposal.

    This is only approximately function preserving because Kimi's saturated
    up activation is odd but not homogeneous. The proposal is nevertheless
    useful when routed up preactivations remain in its nearly linear region.
    """

    if policy == "identity":
        if strength != 0.0:
            raise ValueError("the identity scale gauge requires zero strength")
        return triplet
    if policy not in {"up_down_rms", "down_rms", "down_absmax"}:
        raise ValueError(f"unknown W3/W2 scale-gauge policy {policy!r}")
    if not math.isfinite(strength) or not 0.0 < strength <= 1.0:
        raise ValueError("scale-gauge strength must be in (0, 1]")
    up_rms = triplet.up.float().square().mean(dim=1).sqrt().clamp_min(1e-8)
    down_rms = triplet.down.float().square().mean(dim=0).sqrt().clamp_min(1e-8)
    if policy == "up_down_rms":
        proposal = (down_rms / up_rms).sqrt()
    elif policy == "down_rms":
        proposal = down_rms
    else:
        proposal = triplet.down.float().abs().amax(dim=0).clamp_min(1e-8)
    proposal = proposal / proposal.log().mean().exp()
    scale = proposal.pow(strength).clamp(0.5, 2.0)
    return CoupledTriplet(
        triplet.gate,
        (triplet.up * scale[:, None]).contiguous(),
        (triplet.down / scale[None, :]).contiguous(),
    )


def signed_block_hadamard(
    values: Tensor,
    *,
    block_size: int,
    signs: Tensor,
    dim: int = -1,
    inverse: bool = False,
) -> Tensor:
    """Apply a signed block-Hadamard rotation or its exact inverse.

    The forward row-vector operation is ``v D H`` and the inverse is
    ``v H D``, where ``D`` contains the supplied Rademacher signs and ``H``
    is the normalized block Hadamard.  Draw zero therefore reduces to the
    existing self-inverse transform.
    """

    axis = dim % values.ndim
    if signs.ndim != 1 or signs.shape[0] != values.shape[axis]:
        raise ValueError("Hadamard rotation signs do not align with the axis")
    shape = [1] * values.ndim
    shape[axis] = signs.shape[0]
    expanded = signs.to(device=values.device, dtype=torch.float32).reshape(shape)
    if inverse:
        return block_hadamard(values, block_size=block_size, dim=axis) * expanded
    return block_hadamard(values.float() * expanded, block_size=block_size, dim=axis)


def encode_coupled_block_hadamard(
    triplet: CoupledTriplet,
    *,
    block_size: int,
    preactivation_block_size: int | None = None,
    postactivation_block_size: int | None = None,
    residual_rotation_draw: int = 0,
    intermediate_rotation_draw: int = 0,
) -> CoupledTriplet:
    """Apply an exact two-sided block-Hadamard expert reparameterization.

    Gate and up rows are interleaved before the output-side transform.  The
    transformed upstream rows are split into two ordinary matrix streams only
    for storage; :func:`execute_coupled_block_hadamard` joins them before
    cancelling the transform and evaluating the activation.  The down matrix
    receives the corresponding intermediate- and output-boundary transforms.
    Residual draws control the shared input/output boundaries; intermediate
    draws control the coupled preactivation/postactivation boundaries.
    """

    pre_block = (
        block_size
        if preactivation_block_size is None
        else preactivation_block_size
    )
    post_block = (
        block_size
        if postactivation_block_size is None
        else postactivation_block_size
    )
    if triplet.hidden % block_size:
        raise ValueError("hidden dimension must be divisible by the boundary block size")
    if (2 * triplet.intermediate) % pre_block:
        raise ValueError(
            "interleaved preactivation dimension must be divisible by its block size"
        )
    if triplet.intermediate % post_block:
        raise ValueError(
            "postactivation dimension must be divisible by its block size"
        )
    interleaved = torch.stack((triplet.gate, triplet.up), dim=1).reshape(
        2 * triplet.intermediate, triplet.hidden
    )
    input_signs = _rotation_signs(
        triplet.hidden,
        draw=residual_rotation_draw,
        axis=0,
        device=interleaved.device,
    )
    preactivation_signs = _rotation_signs(
        2 * triplet.intermediate,
        draw=intermediate_rotation_draw,
        axis=1,
        device=interleaved.device,
    )
    postactivation_signs = _rotation_signs(
        triplet.intermediate,
        draw=intermediate_rotation_draw,
        axis=2,
        device=interleaved.device,
    )
    output_signs = _rotation_signs(
        triplet.hidden,
        draw=residual_rotation_draw,
        axis=3,
        device=interleaved.device,
    )
    upstream = signed_block_hadamard(
        signed_block_hadamard(
            interleaved,
            block_size=block_size,
            signs=input_signs,
            dim=1,
        ),
        block_size=pre_block,
        signs=preactivation_signs,
        dim=0,
    )
    down = signed_block_hadamard(
        signed_block_hadamard(
            triplet.down,
            block_size=post_block,
            signs=postactivation_signs,
            dim=1,
        ),
        block_size=block_size,
        signs=output_signs,
        dim=0,
    )
    return CoupledTriplet(
        upstream[: triplet.intermediate].contiguous(),
        upstream[triplet.intermediate :].contiguous(),
        down,
    )


def execute_coupled_block_hadamard(
    inputs: Tensor,
    encoded: CoupledTriplet,
    *,
    block_size: int,
    preactivation_block_size: int | None = None,
    postactivation_block_size: int | None = None,
    residual_rotation_draw: int = 0,
    intermediate_rotation_draw: int = 0,
    activation: ActivationLaw = SITU,
) -> Tensor:
    """Execute an expert encoded by :func:`encode_coupled_block_hadamard`."""

    if inputs.ndim != 2 or inputs.shape[1] != encoded.hidden:
        raise ValueError("inputs do not align with the encoded expert")
    pre_block = (
        block_size
        if preactivation_block_size is None
        else preactivation_block_size
    )
    post_block = (
        block_size
        if postactivation_block_size is None
        else postactivation_block_size
    )
    input_signs = _rotation_signs(
        encoded.hidden,
        draw=residual_rotation_draw,
        axis=0,
        device=inputs.device,
    )
    preactivation_signs = _rotation_signs(
        2 * encoded.intermediate,
        draw=intermediate_rotation_draw,
        axis=1,
        device=inputs.device,
    )
    postactivation_signs = _rotation_signs(
        encoded.intermediate,
        draw=intermediate_rotation_draw,
        axis=2,
        device=inputs.device,
    )
    output_signs = _rotation_signs(
        encoded.hidden,
        draw=residual_rotation_draw,
        axis=3,
        device=inputs.device,
    )
    transformed_inputs = signed_block_hadamard(
        inputs,
        block_size=block_size,
        signs=input_signs,
        dim=1,
    )
    transformed_pre = torch.cat(
        (
            transformed_inputs @ encoded.gate.float().T,
            transformed_inputs @ encoded.up.float().T,
        ),
        dim=1,
    )
    recovered = signed_block_hadamard(
        transformed_pre,
        block_size=pre_block,
        signs=preactivation_signs,
        dim=1,
        inverse=True,
    )
    hidden = activation.value(recovered[:, 0::2], recovered[:, 1::2])
    transformed_hidden = signed_block_hadamard(
        hidden,
        block_size=post_block,
        signs=postactivation_signs,
        dim=1,
    )
    transformed_output = transformed_hidden @ encoded.down.float().T
    return signed_block_hadamard(
        transformed_output,
        block_size=block_size,
        signs=output_signs,
        dim=1,
        inverse=True,
    )


def blockwise_codebook_quantize(
    values: Tensor,
    *,
    codebook: Sequence[float] = (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0),
    block_size: int = 32,
    dim: int = -1,
    iterations: int = 4,
) -> Tensor:
    """Quantize fixed-size blocks to one externally scaled scalar codebook."""

    if values.ndim == 0 or not torch.is_floating_point(values):
        raise TypeError("block quantizer input must be a floating-point tensor")
    if block_size <= 0 or iterations <= 0:
        raise ValueError("block size and iterations must be positive")
    levels = torch.tensor(tuple(codebook), dtype=torch.float32, device=values.device)
    if levels.ndim != 1 or levels.numel() < 2 or not bool(torch.all(torch.isfinite(levels))):
        raise ValueError("quantizer codebook must contain finite scalar levels")
    axis = dim % values.ndim
    if values.shape[axis] % block_size:
        raise ValueError("quantized axis must be divisible by the block size")
    moved = values.float().movedim(axis, -1).contiguous()
    shape = moved.shape
    blocks = moved.reshape(-1, block_size)
    maximum = levels.abs().max().clamp_min(torch.finfo(torch.float32).tiny)
    scale = blocks.abs().amax(dim=1, keepdim=True) / maximum
    indices = torch.zeros_like(blocks, dtype=torch.long)
    for _ in range(iterations):
        normalized = blocks / scale.clamp_min(torch.finfo(torch.float32).tiny)
        indices = (normalized[:, :, None] - levels[None, None, :]).square().argmin(dim=2)
        selected = levels.index_select(0, indices.reshape(-1)).reshape_as(blocks)
        denominator = selected.square().sum(dim=1, keepdim=True)
        fitted = (blocks * selected).sum(dim=1, keepdim=True) / denominator.clamp_min(1e-30)
        scale = fitted.abs().clamp_min(torch.finfo(torch.float32).tiny)
    reconstructed = levels.index_select(0, indices.reshape(-1)).reshape_as(blocks) * scale
    return reconstructed.reshape(shape).movedim(-1, axis).to(values.dtype).contiguous()


def temperature_scaled_situ(
    gate: Tensor,
    up: Tensor,
    gate_scale: Tensor,
    up_scale: Tensor,
    *,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> Tensor:
    """Evaluate the exact per-neuron temperature reparameterization."""

    s = gate_scale.to(gate).reshape(1, -1)
    t = up_scale.to(up).reshape(1, -1)
    if bool(torch.any(s <= 0)) or bool(torch.any(t <= 0)):
        raise ValueError("temperature scales must be positive")
    gate_term = (s * beta) * torch.tanh(gate.float() / (s * beta)) * torch.sigmoid(gate.float() / s)
    up_term = (t * linear_beta) * torch.tanh(up.float() / (t * linear_beta))
    return gate_term * up_term


def canonical_w3_w2_signs(triplet: CoupledTriplet, projection: Tensor) -> Tensor:
    """Choose an exact W3/W2 sign representative from fixed fingerprints."""

    if projection.ndim != 2 or projection.shape[0] != triplet.hidden:
        raise ValueError("fingerprint projection has the wrong shape")
    fingerprints = triplet.up.float() @ projection.float()
    pivot = fingerprints.abs().argmax(dim=1)
    values = fingerprints.gather(1, pivot[:, None]).squeeze(1)
    return torch.where(values < 0, -torch.ones_like(values), torch.ones_like(values))


def rademacher_projection(rows: int, columns: int, *, seed: int, dtype: torch.dtype = torch.float32) -> Tensor:
    if rows <= 0 or columns <= 0:
        raise ValueError("projection dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    bits = torch.randint(0, 2, (rows, columns), generator=generator, dtype=torch.int8)
    return (bits.float().mul_(2).sub_(1) / math.sqrt(columns)).to(dtype)


def micro_neuron_fingerprints(
    triplet: CoupledTriplet,
    projection: Tensor,
    *,
    canonicalize_sign: bool = True,
) -> tuple[Tensor, Tensor]:
    """Build normalized fingerprints for permutation alignment."""

    if projection.ndim != 2 or projection.shape[0] != triplet.hidden:
        raise ValueError("fingerprint projection has the wrong shape")
    signs = (
        canonical_w3_w2_signs(triplet, projection)
        if canonicalize_sign
        else torch.ones(triplet.intermediate)
    )
    gate = triplet.gate.float() @ projection.float()
    up = (triplet.up.float() * signs[:, None]) @ projection.float()
    down = (triplet.down.float() * signs[None, :]).T @ projection.float()
    norms = torch.stack(
        (
            triplet.gate.float().norm(dim=1),
            triplet.up.float().norm(dim=1),
            triplet.down.float().norm(dim=0),
        ),
        dim=1,
    )
    norms = torch.log(norms.clamp_min(torch.finfo(torch.float32).tiny))
    features = torch.cat((gate, up, down, norms), dim=1)
    features = (features - features.mean(dim=0)) / features.std(dim=0).clamp_min(1e-5)
    features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return features.contiguous(), signs.contiguous()


def _auction_assignment(candidate_items: Tensor, candidate_costs: Tensor, *, epsilon: float) -> Tensor | None:
    """Build a deterministic sparse greedy bijection.

    A literal Python auction can require millions of displacement rounds at
    the 3,072-neuron production width.  For screening, sorting the sparse edge
    set once captures the strong nearest-neighbour matches, then pairs the few
    unmatched rows and columns deterministically.  This is an alignment
    heuristic, not a claim of globally optimal assignment cost.
    """

    if candidate_items.shape != candidate_costs.shape or candidate_items.ndim != 2:
        raise ValueError("auction candidates must be aligned matrices")
    buyers, choices = candidate_items.shape
    if choices < 1:
        return None
    items = candidate_items.cpu().to(torch.long)
    costs = candidate_costs.cpu().double()
    if not bool(torch.all(torch.isfinite(costs))):
        raise ValueError("auction candidate costs must be finite")
    assignment = torch.full((buyers,), -1, dtype=torch.long)
    used = torch.zeros(buyers, dtype=torch.bool)
    edge_order = torch.argsort(costs.reshape(-1), stable=True)
    for edge in edge_order.tolist():
        buyer = edge // choices
        item = int(items.reshape(-1)[edge])
        if assignment[buyer] < 0 and not used[item]:
            assignment[buyer] = item
            used[item] = True
    missing_buyers = torch.nonzero(assignment < 0, as_tuple=False).squeeze(1)
    missing_items = torch.nonzero(~used, as_tuple=False).squeeze(1)
    if missing_buyers.numel() != missing_items.numel():
        return None
    assignment[missing_buyers] = missing_items
    return assignment


def sparse_fingerprint_alignment(
    source: Tensor,
    template: Tensor,
    *,
    candidate_counts: Sequence[int] = (32, 64, 128),
    epsilon: float = 1e-6,
    chunk_rows: int = 256,
) -> Tensor:
    """Return ``template_index -> source_index`` alignment permutation."""

    if source.shape != template.shape or source.ndim != 2:
        raise ValueError("alignment fingerprints must have identical shapes")
    neurons = source.shape[0]
    if neurons == 0:
        raise ValueError("alignment requires at least one neuron")
    max_candidates = min(max(int(value) for value in candidate_counts), neurons)
    all_items: list[Tensor] = []
    all_costs: list[Tensor] = []
    template_f = template.float()
    template_norm = template_f.square().sum(dim=1)
    for start in range(0, neurons, max(int(chunk_rows), 1)):
        block = source[start : start + chunk_rows].float()
        distances = block.square().sum(dim=1, keepdim=True) + template_norm[None, :] - 2.0 * block @ template_f.T
        values, indices = torch.topk(distances, max_candidates, dim=1, largest=False, sorted=True)
        all_items.append(indices.cpu())
        all_costs.append(values.cpu())
    items = torch.cat(all_items, dim=0)
    costs = torch.cat(all_costs, dim=0)
    source_to_template: Tensor | None = None
    for count in candidate_counts:
        width = min(int(count), neurons)
        source_to_template = _auction_assignment(items[:, :width], costs[:, :width], epsilon=epsilon)
        if source_to_template is not None:
            break
    if source_to_template is None:
        # A full graph is guaranteed to contain a bijection.  This fallback is
        # intentionally explicit because it is expensive at production width.
        full_items = torch.arange(neurons).repeat(neurons, 1)
        full_costs = torch.empty((neurons, neurons), dtype=torch.float32)
        for start in range(0, neurons, max(int(chunk_rows), 1)):
            block = source[start : start + chunk_rows].float()
            full_costs[start : start + block.shape[0]] = (
                block.square().sum(dim=1, keepdim=True)
                + template_norm[None, :]
                - 2.0 * block @ template_f.T
            )
        source_to_template = _auction_assignment(full_items, full_costs, epsilon=epsilon)
    if source_to_template is None:
        raise RuntimeError("fingerprint auction failed to produce a bijection")
    template_to_source = torch.empty_like(source_to_template)
    template_to_source[source_to_template] = torch.arange(neurons)
    return template_to_source.contiguous()


def fit_low_rank_basis(matrix: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    """Fit an ordinary low-rank basis to ``[experts, features]`` data."""

    if matrix.ndim != 2 or not torch.is_floating_point(matrix):
        raise TypeError("basis input must be a floating-point matrix")
    if not 1 <= rank <= min(matrix.shape):
        raise ValueError("basis rank is out of range")
    u, singular, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    coefficients = u[:, :rank] * singular[:rank]
    basis = vh[:rank]
    return coefficients.contiguous(), basis.contiguous()


def fit_cross_matrix_predictor(
    triplet: CoupledTriplet,
    *,
    per: str,
    regularization: float = 1e-6,
) -> dict[str, Tensor | float]:
    """Predict ``W2.T`` from ``W1`` and ``W3`` along one natural axis.

    ``per='neuron'`` fits two coefficients for each intermediate row;
    ``per='hidden'`` fits two coefficients for each latent coordinate.  The
    result tests conditional structure without assuming a serving format.
    """

    if per not in {"neuron", "hidden"}:
        raise ValueError("cross-matrix predictor axis must be 'neuron' or 'hidden'")
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("predictor regularization must be finite and positive")
    predictors = torch.stack((triplet.gate.float(), triplet.up.float()), dim=-1)
    target = triplet.down.float().T
    reduce_dim = 1 if per == "neuron" else 0
    normal = torch.einsum(
        "...i,...j->...ij",
        predictors,
        predictors,
    ).sum(dim=reduce_dim)
    rhs = (predictors * target[..., None]).sum(dim=reduce_dim)
    identity = torch.eye(2, device=normal.device, dtype=normal.dtype)
    scale = normal.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)
    coefficients = torch.linalg.solve(
        normal + identity * (scale[..., None] * regularization + 1e-12),
        rhs[..., None],
    ).squeeze(-1)
    if per == "neuron":
        reconstruction = (predictors * coefficients[:, None, :]).sum(dim=-1)
    else:
        reconstruction = (predictors * coefficients[None, :, :]).sum(dim=-1)
    residual = target - reconstruction
    target_sse = target.double().square().sum()
    residual_sse = residual.double().square().sum()
    return {
        "coefficients": coefficients.contiguous(),
        "reconstruction": reconstruction.contiguous(),
        "residual": residual.contiguous(),
        "residual_fraction": float(residual_sse / target_sse) if target_sse > 0 else 0.0,
    }


def fit_metric_codebook(
    values: Tensor,
    codebook_size: int,
    *,
    metrics: Tensor | None = None,
    iterations: int = 20,
    seed: int = 0,
) -> Tensor:
    """Fit a small Euclidean or per-sample quadratic-metric codebook."""

    if values.ndim != 2 or not torch.is_floating_point(values) or values.shape[0] < codebook_size:
        raise ValueError("codebook training values have an invalid shape")
    rows, dimension = values.shape
    if codebook_size <= 0 or iterations <= 0:
        raise ValueError("codebook size and iterations must be positive")
    x = values.float().cpu()
    if metrics is None:
        m = torch.eye(dimension).repeat(rows, 1, 1)
    else:
        if metrics.shape != (rows, dimension, dimension):
            raise ValueError("quadratic metrics do not align with values")
        m = metrics.float().cpu()
    generator = torch.Generator().manual_seed(int(seed))
    first = int(torch.randint(0, rows, (), generator=generator))
    centers = [x[first]]
    nearest = torch.full((rows,), float("inf"))
    for _ in range(1, codebook_size):
        delta = x - centers[-1]
        distance = torch.einsum("ri,rij,rj->r", delta, m, delta).clamp_min(0)
        nearest = torch.minimum(nearest, distance)
        total = nearest.sum()
        if total <= 0:
            index = len(centers) % rows
        else:
            index = int(torch.multinomial(nearest / total, 1, generator=generator))
        centers.append(x[index])
    codebook = torch.stack(centers)
    for _ in range(iterations):
        delta = x[:, None, :] - codebook[None, :, :]
        distances = torch.einsum("rki,rij,rkj->rk", delta, m, delta)
        assignment = distances.argmin(dim=1)
        updated = codebook.clone()
        for index in range(codebook_size):
            selected = assignment == index
            if not bool(torch.any(selected)):
                continue
            normal = m[selected].sum(dim=0)
            rhs = torch.einsum("rij,rj->i", m[selected], x[selected])
            regularizer = torch.eye(dimension) * (torch.trace(normal) * 1e-8 + 1e-12)
            updated[index] = torch.linalg.solve(normal + regularizer, rhs)
        if torch.allclose(updated, codebook, rtol=1e-6, atol=1e-7):
            codebook = updated
            break
        codebook = updated
    return codebook.contiguous()


def quantize_metric_codebook(values: Tensor, codebook: Tensor, *, metrics: Tensor | None = None) -> tuple[Tensor, Tensor]:
    if values.ndim != 2 or codebook.ndim != 2 or values.shape[1] != codebook.shape[1]:
        raise ValueError("values and codebook do not align")
    x = values.float()
    c = codebook.float().to(x.device)
    delta = x[:, None, :] - c[None, :, :]
    if metrics is None:
        distances = delta.square().sum(dim=2)
    else:
        if metrics.shape != (x.shape[0], x.shape[1], x.shape[1]):
            raise ValueError("quadratic metrics do not align with values")
        distances = torch.einsum("rki,rij,rkj->rk", delta, metrics.float().to(x.device), delta)
    indices = distances.argmin(dim=1)
    return c.index_select(0, indices).to(values.dtype), indices


def ridge_refit_down(
    original_down: Tensor,
    source_hidden: Tensor,
    quantized_hidden: Tensor,
    *,
    target_down: Tensor | None = None,
    regularization: float,
) -> Tensor:
    """Refit W2 toward its source value using the sample-space ridge solve."""

    if source_hidden.shape != quantized_hidden.shape or source_hidden.ndim != 2:
        raise ValueError("source and quantized hidden rows must align")
    if original_down.ndim != 2 or original_down.shape[1] != source_hidden.shape[1]:
        raise ValueError("down projection does not align with hidden rows")
    target_weight = original_down if target_down is None else target_down
    if target_weight.shape != original_down.shape:
        raise ValueError("target down projection does not align with the original")
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("ridge regularization must be finite and positive")
    h = quantized_hidden.float()
    x0 = original_down.float().T
    target = source_hidden.float() @ target_weight.float().T
    residual = target - h @ x0
    kernel = h @ h.T
    kernel.diagonal().add_(regularization)
    dual = torch.linalg.solve(kernel, residual)
    return (x0 + h.T @ dual).T.contiguous()


def fit_function_space_correction(
    features: Tensor,
    target_error: Tensor,
    *,
    rank: int,
    regularization: float,
) -> tuple[Tensor, Tensor]:
    """Fit a reduced-rank linear correction ``features @ left @ right``.

    The ridge solve happens in sample space, so it remains useful when the
    feature and output widths are thousands but the calibration row count is
    deliberately small.
    """

    if features.ndim != 2 or target_error.ndim != 2:
        raise ValueError("correction features and targets must be matrices")
    if features.shape[0] != target_error.shape[0]:
        raise ValueError("correction features and targets must share rows")
    if not 1 <= rank <= min(features.shape[0], features.shape[1], target_error.shape[1]):
        raise ValueError("correction rank is out of range")
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("correction regularization must be finite and positive")
    x = features.float()
    y = target_error.float()
    kernel = x @ x.T
    kernel.diagonal().add_(regularization)
    full = x.T @ torch.linalg.solve(kernel, y)
    u, singular, vh = torch.linalg.svd(full, full_matrices=False)
    left = (u[:, :rank] * singular[:rank]).contiguous()
    right = vh[:rank].contiguous()
    return left, right


def search_expert_output_gain(
    aggregate: Tensor,
    source_output: Tensor,
    candidate_output: Tensor,
    route_gates: Tensor,
    output_metric: RoutedOutputMetric,
    gains: Tensor,
) -> dict[str, float]:
    """Search the exact post-RMS objective for one expert-local output gain."""

    if source_output.shape != candidate_output.shape or source_output.shape != aggregate.shape:
        raise ValueError("gain-search outputs and aggregate must align")
    if route_gates.ndim != 1 or route_gates.numel() != aggregate.shape[0]:
        raise ValueError("gain-search route gates must align with rows")
    if gains.ndim != 1 or gains.numel() == 0 or not bool(torch.all(torch.isfinite(gains))):
        raise ValueError("gain-search candidates must be a finite vector")
    routed_source = source_output.float() * route_gates.float()[:, None]
    base = aggregate.float() - routed_source
    losses = []
    for gain in gains:
        candidate_aggregate = base + route_gates.float()[:, None] * (
            candidate_output.float() * gain.float()
        )
        delta = output_metric.output(candidate_aggregate) - output_metric.output(aggregate)
        losses.append(delta.double().square().sum())
    stacked = torch.stack(losses)
    winner = int(stacked.argmin())
    return {
        "gain": float(gains[winner]),
        "sse": float(stacked[winner]),
        "unit_gain_sse": float(
            output_metric.exact_delta(
                aggregate,
                route_gates.float()[:, None] * (candidate_output.float() - source_output.float()),
            ).double().square().sum()
        ),
    }


def entropy_bits(symbols: Tensor) -> float:
    if symbols.numel() == 0:
        raise ValueError("entropy requires at least one symbol")
    _, counts = torch.unique(symbols.detach().cpu(), return_counts=True)
    probabilities = counts.double() / counts.sum()
    return float(-(probabilities * probabilities.log2()).sum())


def conditional_entropy_bits(symbols: Tensor, contexts: Tensor) -> float:
    """Return empirical ``H(symbols | contexts)`` in bits per symbol."""

    if symbols.shape != contexts.shape or symbols.numel() == 0:
        raise ValueError("conditional entropy inputs must be nonempty and aligned")
    symbol_values, symbol_inverse = torch.unique(
        symbols.detach().cpu().reshape(-1), return_inverse=True
    )
    _, context_inverse = torch.unique(
        contexts.detach().cpu().reshape(-1), return_inverse=True
    )
    joint = context_inverse.to(torch.int64) * symbol_values.numel() + symbol_inverse
    _, joint_counts = torch.unique(joint, return_counts=True)
    _, context_counts = torch.unique(context_inverse, return_counts=True)
    total = float(symbols.numel())
    joint_probabilities = joint_counts.double() / total
    context_probabilities = context_counts.double() / total
    joint_entropy = -(joint_probabilities * joint_probabilities.log2()).sum()
    context_entropy = -(context_probabilities * context_probabilities.log2()).sum()
    return float(joint_entropy - context_entropy)


def fixed_tile_entropy_bound(symbols: Tensor, *, tile_symbols: int, restart_bytes: int = 0) -> dict[str, float]:
    if tile_symbols <= 0 or restart_bytes < 0:
        raise ValueError("tile size must be positive and restart bytes non-negative")
    flat = symbols.detach().cpu().reshape(-1)
    total_bits = 0.0
    tiles = 0
    for start in range(0, flat.numel(), tile_symbols):
        tile = flat[start : start + tile_symbols]
        total_bits += entropy_bits(tile) * tile.numel() + restart_bytes * 8
        tiles += 1
    return {
        "symbols": float(flat.numel()),
        "tiles": float(tiles),
        "total_bits": total_bits,
        "bits_per_symbol": total_bits / flat.numel(),
    }


@dataclass(frozen=True)
class RateComponent:
    name: str
    bits: int


def effective_bpw(weight_count: int, components: Iterable[RateComponent]) -> dict[str, float | int]:
    if weight_count <= 0:
        raise ValueError("weight count must be positive")
    values = tuple(components)
    if any(component.bits < 0 for component in values):
        raise ValueError("rate components cannot contain negative bits")
    total = sum(component.bits for component in values)
    return {
        "weights": int(weight_count),
        "bits": int(total),
        "bytes": int((total + 7) // 8),
        "effective_bpw": float(total / weight_count),
    }


def allocate_rate_options(
    distortion: Tensor,
    bits: Tensor,
    *,
    target_bits: int,
    iterations: int = 80,
) -> dict[str, Tensor | int | float]:
    """Choose one option per item with a deterministic Lagrangian frontier.

    This is an analysis allocator rather than a checkpoint allocator.  It
    exposes whether unequal rates or progressive refinements resonate before
    a format-specific exact-byte solver is justified.
    """

    if distortion.ndim != 2 or bits.shape != distortion.shape:
        raise ValueError("rate-option distortion and bits must be aligned matrices")
    if distortion.shape[0] == 0 or distortion.shape[1] < 2:
        raise ValueError("rate allocation requires items with at least two options")
    if bits.dtype.is_floating_point or bool(torch.any(bits < 0)):
        raise ValueError("rate-option bits must be non-negative integers")
    if target_bits < 0 or iterations <= 0:
        raise ValueError("target bits and iterations are invalid")
    d = distortion.double().cpu()
    b = bits.double().cpu()

    def choose(penalty: float) -> Tensor:
        return (d + penalty * b).argmin(dim=1)

    minimum = bits.min(dim=1).values.sum().item()
    maximum = bits.max(dim=1).values.sum().item()
    if not minimum <= target_bits <= maximum:
        raise ValueError("target bits fall outside the available option envelope")
    low, high = 0.0, 1.0
    while float(b.gather(1, choose(high)[:, None]).sum()) > target_bits:
        high *= 2.0
    best = choose(high)
    for _ in range(iterations):
        midpoint = (low + high) / 2.0
        selected = choose(midpoint)
        used = float(b.gather(1, selected[:, None]).sum())
        if used > target_bits:
            low = midpoint
        else:
            high = midpoint
            best = selected
    selected_bits = bits.cpu().gather(1, best[:, None]).squeeze(1)
    selected_distortion = distortion.cpu().gather(1, best[:, None]).squeeze(1)
    return {
        "selection": best,
        "bits": int(selected_bits.sum()),
        "distortion": float(selected_distortion.double().sum()),
        "unused_bits": int(target_bits - selected_bits.sum()),
        "penalty": float(high),
    }


def micro_neuron_energy_saliency(hidden: Tensor, down: Tensor, route_gates: Tensor | None = None) -> Tensor:
    if hidden.ndim != 2 or down.ndim != 2 or hidden.shape[1] != down.shape[1]:
        raise ValueError("hidden rows and down columns do not align")
    weights = (
        torch.ones(hidden.shape[0], device=hidden.device)
        if route_gates is None
        else route_gates.float().to(hidden.device).square()
    )
    if weights.ndim != 1 or weights.numel() != hidden.shape[0]:
        raise ValueError("route gates do not align with hidden rows")
    hidden_energy = (hidden.float().square() * weights[:, None]).sum(dim=0)
    return hidden_energy * down.float().square().sum(dim=0)


def prune_micro_neurons(triplet: CoupledTriplet, keep: Tensor) -> CoupledTriplet:
    if keep.dtype != torch.bool or keep.ndim != 1 or keep.numel() != triplet.intermediate:
        raise ValueError("keep mask must contain one boolean per neuron")
    return CoupledTriplet(
        triplet.gate[keep].contiguous(),
        triplet.up[keep].contiguous(),
        triplet.down[:, keep].contiguous(),
    )


__all__ = [
    "ActivationLaw",
    "CoupledTriplet",
    "CoupledTripletSource",
    "PairMetricSummary",
    "RateComponent",
    "RoutedOutputMetric",
    "SITU",
    "SiTUComponentGeometry",
    "apply_common_input_gauge",
    "apply_output_rotation",
    "apply_permutation_sign_gauge",
    "apply_postactivation_scale",
    "apply_w3_w2_sign_draw",
    "apply_w3_w2_scale_gauge",
    "allocate_rate_options",
    "block_hadamard",
    "blockwise_codebook_quantize",
    "canonical_w3_w2_signs",
    "conditional_entropy_bits",
    "encode_two_sided_linear",
    "effective_bpw",
    "entropy_bits",
    "execute_two_sided_linear",
    "expert_hidden",
    "expert_output",
    "fit_cross_matrix_predictor",
    "fit_function_space_correction",
    "fit_low_rank_basis",
    "fit_metric_codebook",
    "fixed_tile_entropy_bound",
    "hadamard_rotation_signs",
    "local_triplet_metrics",
    "micro_neuron_energy_saliency",
    "micro_neuron_fingerprints",
    "pair_activation_metric",
    "pair_residual_decomposition",
    "prune_micro_neurons",
    "quantize_metric_codebook",
    "rademacher_projection",
    "radial_tangent_decomposition",
    "ridge_refit_down",
    "route_error_covariance",
    "search_expert_output_gain",
    "select_corouted_candidate_modes",
    "situ_derivatives",
    "situ_component_geometry",
    "situ_value",
    "sparse_fingerprint_alignment",
    "temperature_scaled_situ",
]

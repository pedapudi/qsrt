"""Training-free low-rank corrections for frozen expert matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import torch
from safetensors import safe_open


Tensor = torch.Tensor

_FACTOR_NAME = re.compile(
    r"^experts\.(\d+)\.(w1|w2|w3)\.(plain|weighted)\.rank_(\d+)\.(a|b)$"
)


@dataclass(frozen=True)
class LowRankAdapterFit:
    """One balanced ``B @ A.T`` approximation to a matrix error."""

    a: Tensor
    b: Tensor
    singular_values: Tensor
    objective_total: float
    objective_captured: Tensor

    def __post_init__(self) -> None:
        if self.a.ndim != 2 or self.b.ndim != 2:
            raise ValueError("adapter factors must be matrices")
        if self.a.shape[1] != self.b.shape[1]:
            raise ValueError("adapter factors must have the same rank")
        rank = self.a.shape[1]
        if self.singular_values.shape != (rank,):
            raise ValueError("singular values do not match adapter rank")
        if self.objective_captured.shape != (rank,):
            raise ValueError("captured-objective curve does not match adapter rank")
        if not self.objective_total > 0:
            raise ValueError("adapter objective must be positive")


def _balanced_factors(left: Tensor, right: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Factor ``left @ right`` with equal singular-value scaling."""

    core_left, singular, right_vectors = torch.linalg.svd(
        right.float(), full_matrices=False
    )
    root = singular.clamp_min(0).sqrt()
    b = (left.float() @ core_left) * root[None, :]
    a = right_vectors.T * root[None, :]
    return a.contiguous(), b.contiguous(), singular.contiguous()


def fit_plain_error_adapter(
    error: Tensor,
    *,
    rank: int,
    oversampling: int = 8,
    power_iterations: int = 2,
) -> LowRankAdapterFit:
    """Fit a truncated Frobenius SVD to ``error = source - anchor``."""

    if error.ndim != 2 or not torch.is_floating_point(error):
        raise TypeError("matrix error must be a floating-point matrix")
    if not 1 <= rank <= min(error.shape):
        raise ValueError("adapter rank is outside the matrix geometry")
    q = min(min(error.shape), rank + max(int(oversampling), 0))
    u, singular, v = torch.svd_lowrank(
        error.float(), q=q, niter=max(int(power_iterations), 0)
    )
    order = torch.argsort(singular, descending=True)[:rank]
    u = u.index_select(1, order)
    singular = singular.index_select(0, order)
    v = v.index_select(1, order)
    root = singular.clamp_min(0).sqrt()
    a = v * root[None, :]
    b = u * root[None, :]
    total = float(error.float().square().double().sum().item())
    captured = singular.double().square().cumsum(0)
    return LowRankAdapterFit(
        a=a.contiguous(),
        b=b.contiguous(),
        singular_values=singular.contiguous(),
        objective_total=total,
        objective_captured=captured.contiguous(),
    )


def _weighted_operator(
    error: Tensor,
    vectors: Tensor,
    row_batches: Iterable[tuple[Tensor, Tensor]],
    *,
    weight_sum: float,
) -> Tensor:
    """Apply ``E H E.T`` without materializing the input covariance."""

    if weight_sum <= 0:
        raise ValueError("weighted row mass must be positive")
    projected = error.float().T @ vectors.float()
    covariance_product = torch.zeros_like(projected)
    for rows, weights in row_batches:
        x = rows.to(device=error.device, dtype=torch.float32, non_blocking=True)
        w = weights.to(device=error.device, dtype=torch.float32, non_blocking=True)
        covariance_product.addmm_(
            x.T,
            (x @ projected) * w.square()[:, None],
            beta=1.0,
            alpha=1.0 / weight_sum,
        )
    return (error.float() @ covariance_product).contiguous()


def fit_weighted_error_adapter(
    error: Tensor,
    rows: Tensor,
    weights: Tensor,
    *,
    rank: int,
    oversampling: int = 8,
    power_iterations: int = 2,
    batch_rows: int = 2048,
    seed: int = 0,
) -> LowRankAdapterFit:
    """Fit the best rank-limited correction under routed input samples.

    The minimized objective is ``sum_i w_i**2 ||(E - B A.T) x_i||**2``.
    The returned factors are balanced by the singular values of the resulting
    correction so either factor is suitable as a trainable initialization.
    """

    if error.ndim != 2 or not torch.is_floating_point(error):
        raise TypeError("matrix error must be a floating-point matrix")
    if rows.ndim != 2 or rows.shape[1] != error.shape[1]:
        raise ValueError("weighted rows do not match the matrix input dimension")
    if weights.ndim != 1 or weights.shape[0] != rows.shape[0]:
        raise ValueError("weighted rows and route weights are not aligned")
    if not 1 <= rank <= min(error.shape):
        raise ValueError("adapter rank is outside the matrix geometry")
    if batch_rows <= 0 or rows.shape[0] == 0:
        raise ValueError("weighted fit requires nonempty positive-size batches")
    if not torch.all(torch.isfinite(weights)) or torch.any(weights < 0):
        raise ValueError("route weights must be finite and nonnegative")
    weight_sum = float(weights.double().square().sum().item())
    if not weight_sum > 0:
        raise ValueError("weighted fit has zero route mass")

    def batches() -> Iterable[tuple[Tensor, Tensor]]:
        for begin in range(0, rows.shape[0], batch_rows):
            end = min(begin + batch_rows, rows.shape[0])
            yield rows[begin:end], weights[begin:end]

    q = min(error.shape[0], rank + max(int(oversampling), 0))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    omega = torch.randn(
        (error.shape[0], q), dtype=torch.float32, generator=generator
    ).to(error.device)
    basis, _ = torch.linalg.qr(
        _weighted_operator(error, omega, batches(), weight_sum=weight_sum),
        mode="reduced",
    )
    for _ in range(max(int(power_iterations), 0)):
        basis, _ = torch.linalg.qr(
            _weighted_operator(error, basis, batches(), weight_sum=weight_sum),
            mode="reduced",
        )
    reduced = basis.T @ _weighted_operator(
        error, basis, batches(), weight_sum=weight_sum
    )
    reduced = (reduced + reduced.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(reduced)
    order = torch.argsort(eigenvalues, descending=True)[:rank]
    eigenvalues = eigenvalues.index_select(0, order).clamp_min(0)
    left = (basis @ eigenvectors.index_select(1, order)).contiguous()

    # For a fixed orthonormal output subspace, the weighted least-squares
    # optimum is ``left @ left.T @ error``. Rebalance that correction before
    # serialization so both trainable factors have comparable scale.
    a, b, singular = _balanced_factors(left, left.T @ error.float())

    total = 0.0
    for batch, route_weight in batches():
        x = batch.to(device=error.device, dtype=torch.float32, non_blocking=True)
        w = route_weight.to(
            device=error.device, dtype=torch.float32, non_blocking=True
        )
        output = x @ error.float().T
        total += float(
            (output.square().sum(dim=1) * w.square()).double().sum().item()
        )
    total /= weight_sum
    captured = eigenvalues.double().cumsum(0)
    return LowRankAdapterFit(
        a=a,
        b=b,
        singular_values=singular,
        objective_total=total,
        objective_captured=captured.contiguous(),
    )


def load_sparse_expert_adapter_banks(
    path: str | Path,
    *,
    variant: str,
    rank: int,
    matrix_shapes: dict[str, tuple[int, int]],
    num_experts: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, tuple[Tensor, Tensor]], tuple[int, ...], dict[str, str]]:
    """Expand sparse serialized factors into zero-filled grouped banks."""

    if variant not in {"plain", "weighted"}:
        raise ValueError("adapter variant must be plain or weighted")
    if rank <= 0 or num_experts <= 0:
        raise ValueError("adapter rank and expert count must be positive")
    if set(matrix_shapes) != {"w1", "w2", "w3"}:
        raise ValueError("adapter matrix shapes do not cover all projections")
    factor_path = Path(path).expanduser().resolve()
    if not factor_path.is_file():
        raise FileNotFoundError(factor_path)

    banks = {
        matrix: (
            torch.zeros(
                (num_experts, input_dimension, rank),
                device=device,
                dtype=dtype,
            ),
            torch.zeros(
                (num_experts, output_dimension, rank),
                device=device,
                dtype=dtype,
            ),
        )
        for matrix, (output_dimension, input_dimension) in matrix_shapes.items()
    }
    loaded: dict[int, set[tuple[str, str]]] = {}
    with safe_open(str(factor_path), framework="pt", device="cpu") as reader:
        metadata = dict(reader.metadata() or {})
        for name in reader.keys():
            match = _FACTOR_NAME.fullmatch(name)
            if match is None:
                raise ValueError(f"unrecognized low-rank factor tensor {name}")
            expert_text, matrix, stored_variant, stored_rank, side = match.groups()
            if stored_variant != variant or int(stored_rank) != rank:
                continue
            expert = int(expert_text)
            if not 0 <= expert < num_experts:
                raise ValueError(f"adapter expert index {expert} is out of range")
            destination = banks[matrix][0 if side == "a" else 1][expert]
            value = reader.get_tensor(name)
            if tuple(value.shape) != tuple(destination.shape):
                raise ValueError(f"adapter tensor {name} has incompatible geometry")
            destination.copy_(value.to(device=device, dtype=dtype))
            loaded.setdefault(expert, set()).add((matrix, side))
    expected = {(matrix, side) for matrix in matrix_shapes for side in ("a", "b")}
    incomplete = {
        expert: sorted(expected - names)
        for expert, names in loaded.items()
        if names != expected
    }
    if incomplete:
        expert = min(incomplete)
        raise ValueError(
            f"adapter expert {expert} lacks factors {incomplete[expert]}"
        )
    if not loaded:
        raise ValueError(
            f"factor file has no {variant} rank-{rank} adapter tensors"
        )
    return banks, tuple(sorted(loaded)), metadata


__all__ = [
    "LowRankAdapterFit",
    "fit_plain_error_adapter",
    "fit_weighted_error_adapter",
    "load_sparse_expert_adapter_banks",
]

"""Dense-H coordinate refinement for legal QSRT trellis paths.

The ordinary BlockLDLQ pass commits to each 16-row block once.  This module
revisits those blocks after all quantization errors are known.  For each block
it derives the exact conditional target under the supplied dense input
Hessian, asks the ordinary tile Viterbi encoder for legal replacement paths,
and accepts only paths that strictly reduce the same dense-H objective.

All tensors must already use the regularized encoder coordinates.  The
quantizer callback consumes and returns flattened 16x16 tiles in the EXL
tensor-core order and may return any payload shape after its batch dimension.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import torch

from qsrt.gradient_guided_viterbi import (
    LowRankViterbiGradientGuidance,
    ViterbiGradientGuidance,
)


QuantizeTiles = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
GradientGuidance = ViterbiGradientGuidance | LowRankViterbiGradientGuidance


@dataclass(frozen=True)
class H2ViterbiRefineConfig:
    """Search controls for dense-H block-coordinate path refinement."""

    sweeps: int = 1
    alternate_sweep_order: bool = True
    dither_scales: tuple[float, ...] = ()
    num_dither_patterns: int = 0
    seed: int = 0x48325643
    max_quantizer_batch: int = 256
    acceptance_abs_tol: float = 1.0e-8
    acceptance_rel_tol: float = 1.0e-8
    score_dtype: torch.dtype = torch.float32
    stop_on_no_change: bool = True


@dataclass(frozen=True)
class H2ViterbiSweepStats:
    sweep: int
    objective_before: float
    objective_after: float
    quadratic_before: float
    quadratic_after: float
    linear_before: float
    linear_after: float
    relative_improvement: float
    accepted_tiles: int
    proposed_tiles: int
    unique_candidate_paths: int


@dataclass
class H2ViterbiRefineResult:
    weight_q: torch.Tensor
    encoded: torch.Tensor
    sweep_stats: list[H2ViterbiSweepStats]

    def stats_dict(self) -> dict[str, object]:
        return {"sweeps": [asdict(stats) for stats in self.sweep_stats]}


def dense_h_objective(
    weight: torch.Tensor,
    weight_q: torch.Tensor,
    hessian: torch.Tensor,
    *,
    column_weights: torch.Tensor | None = None,
    chunk_columns: int = 512,
    dtype: torch.dtype = torch.float64,
) -> float:
    """Return ``sum_c a[c] * (W-Q)[:,c].T @ H @ (W-Q)[:,c]``."""

    if weight.shape != weight_q.shape or weight.ndim != 2:
        raise ValueError("weight and reconstruction must have one rank-2 shape")
    if hessian.shape != (weight.shape[0], weight.shape[0]):
        raise ValueError("Hessian shape does not match the input dimension")
    if chunk_columns <= 0:
        raise ValueError("chunk_columns must be positive")
    if column_weights is None:
        weights = torch.ones(weight.shape[1], dtype=dtype, device=weight.device)
    else:
        if column_weights.numel() != weight.shape[1]:
            raise ValueError("column weights do not match the output dimension")
        weights = column_weights.reshape(-1).to(device=weight.device, dtype=dtype)
        if bool(torch.any(weights < 0)):
            raise ValueError("column weights must be nonnegative")
    hessian_work = hessian.to(device=weight.device, dtype=dtype)
    total = torch.zeros((), dtype=dtype, device=weight.device)
    for start in range(0, weight.shape[1], chunk_columns):
        stop = min(start + chunk_columns, weight.shape[1])
        error = (weight[:, start:stop] - weight_q[:, start:stop]).to(dtype)
        total += torch.sum(
            error * (hessian_work @ error) * weights[start:stop].unsqueeze(0)
        )
    return float(total.item())


def dense_h_guided_objective(
    weight: torch.Tensor,
    weight_q: torch.Tensor,
    hessian: torch.Tensor,
    guidance: GradientGuidance,
    *,
    column_weights: torch.Tensor | None = None,
    chunk_columns: int = 512,
    dtype: torch.dtype = torch.float64,
) -> tuple[float, float, float]:
    """Return quadratic, anchor-relative linear, and combined objectives."""

    guidance.validate(weight.shape)
    quadratic = dense_h_objective(
        weight,
        weight_q,
        hessian,
        column_weights=column_weights,
        chunk_columns=chunk_columns,
        dtype=dtype,
    )
    if isinstance(guidance, LowRankViterbiGradientGuidance):
        linear = guidance.linear_term(weight_q, dtype=dtype)
    else:
        gradient = guidance.gradient.to(device=weight.device, dtype=dtype)
        anchor = guidance.anchor.to(device=weight.device, dtype=dtype)
        candidate = weight_q.to(dtype=dtype)
        linear = float(
            float(guidance.strength)
            * torch.sum(gradient * (candidate - anchor)).item()
        )
    return quadratic, linear, quadratic + linear


def _tensor_core_permutation(device: torch.device) -> torch.Tensor:
    permutation = [0] * 256
    for tile in range(32):
        row0 = (tile % 4) * 2
        row1 = row0 + 1
        row2 = row0 + 8
        row3 = row0 + 9
        column0 = tile // 4
        column1 = column0 + 8
        permutation[tile * 8 + 0] = row0 * 16 + column0
        permutation[tile * 8 + 1] = row1 * 16 + column0
        permutation[tile * 8 + 2] = row2 * 16 + column0
        permutation[tile * 8 + 3] = row3 * 16 + column0
        permutation[tile * 8 + 4] = row0 * 16 + column1
        permutation[tile * 8 + 5] = row1 * 16 + column1
        permutation[tile * 8 + 6] = row2 * 16 + column1
        permutation[tile * 8 + 7] = row3 * 16 + column1
    return torch.tensor(permutation, dtype=torch.long, device=device)


def _quantize_in_chunks(
    inputs: torch.Tensor,
    quantize_tiles: QuantizeTiles,
    max_batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_batch <= 0:
        raise ValueError("max_quantizer_batch must be positive")
    reconstructions = []
    payloads = []
    for start in range(0, inputs.shape[0], max_batch):
        source = inputs[start : start + max_batch].contiguous()
        reconstruction, payload = quantize_tiles(source)
        if reconstruction.shape != source.shape:
            raise ValueError("tile quantizer returned the wrong reconstruction shape")
        if payload.ndim < 1 or payload.shape[0] != source.shape[0]:
            raise ValueError("tile quantizer returned the wrong payload batch")
        reconstructions.append(reconstruction)
        payloads.append(payload)
    return torch.cat(reconstructions), torch.cat(payloads)


def _h_whitened_patterns(
    hessian_block: torch.Tensor,
    *,
    count: int,
    seed: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    block = hessian_block.shape[0]
    if count <= 0:
        return torch.empty(
            (0, block, block), dtype=dtype, device=hessian_block.device
        )
    generator = torch.Generator(device=hessian_block.device)
    generator.manual_seed(seed)
    probes = torch.randint(
        0,
        2,
        (count, block, block),
        generator=generator,
        device=hessian_block.device,
        dtype=torch.int64,
    )
    probes = (probes * 2 - 1).to(dtype)
    hessian_work = hessian_block.to(dtype)
    cholesky = torch.linalg.cholesky(hessian_work)
    columns = probes.permute(1, 0, 2).reshape(block, count * block)
    columns = torch.linalg.solve_triangular(
        cholesky.transpose(0, 1), columns, upper=True
    )
    patterns = columns.reshape(block, count, block).permute(1, 0, 2)
    hessian_patterns = torch.einsum("ij,pjc->pic", hessian_work, patterns)
    energy = torch.sum(patterns * hessian_patterns, dim=(1, 2), keepdim=True)
    energy = energy.clamp_min(1.0e-20)
    target_energy = torch.tensor(
        float(block * block), dtype=dtype, device=hessian_block.device
    )
    return patterns * torch.sqrt(target_energy / energy)


def _conditional_target(
    weight_block: torch.Tensor,
    error_block: torch.Tensor,
    hessian_error_block: torch.Tensor,
    hessian_block: torch.Tensor,
    *,
    gradient_block: torch.Tensor | None = None,
    gradient_strength: float = 0.0,
    column_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    cross = hessian_error_block - hessian_block @ error_block
    target = weight_block + torch.linalg.solve(hessian_block, cross)
    if gradient_block is None or gradient_strength == 0.0:
        return target
    if column_weights is None:
        scaled_gradient = gradient_block
    else:
        if bool(torch.any(column_weights <= 0)):
            raise ValueError(
                "gradient-guided refinement requires positive column weights"
            )
        scaled_gradient = gradient_block / column_weights.unsqueeze(0)
    correction = torch.linalg.solve(hessian_block, scaled_gradient)
    return target - 0.5 * float(gradient_strength) * correction


def _conditional_scores(
    target_tiles: torch.Tensor,
    candidates: torch.Tensor,
    hessian_block: torch.Tensor,
    column_weight_tiles: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    difference = (target_tiles[:, None] - candidates).to(dtype)
    weighted = torch.einsum(
        "ij,tmjc->tmic", hessian_block.to(dtype), difference
    )
    return torch.sum(
        difference
        * weighted
        * column_weight_tiles[:, None, None, :].to(dtype),
        dim=(-2, -1),
    )


def _candidate_paths(
    target_tiles: torch.Tensor,
    current_tiles: torch.Tensor,
    current_payload: torch.Tensor,
    hessian_block: torch.Tensor,
    permutation: torch.Tensor,
    inverse_permutation: torch.Tensor,
    quantize_tiles: QuantizeTiles,
    config: H2ViterbiRefineConfig,
    *,
    sweep: int,
    row_block: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    patterns = _h_whitened_patterns(
        hessian_block,
        count=config.num_dither_patterns,
        seed=config.seed + 1009 * sweep + 9176 * row_block,
        dtype=target_tiles.dtype,
    )
    variants = [target_tiles]
    if patterns.numel() and config.dither_scales:
        tile_rms = target_tiles.square().mean(dim=(-2, -1), keepdim=True).sqrt()
        tile_rms = tile_rms.clamp_min(1.0e-8)
        for scale in config.dither_scales:
            for pattern in patterns:
                shift = float(scale) * tile_rms * pattern.unsqueeze(0)
                variants.extend((target_tiles + shift, target_tiles - shift))

    stacked = torch.stack(variants)
    variant_count, tile_count = stacked.shape[:2]
    inputs = stacked.reshape(variant_count * tile_count, 256)[:, permutation]
    reconstruction, payload = _quantize_in_chunks(
        inputs, quantize_tiles, config.max_quantizer_batch
    )
    reconstruction = reconstruction[:, inverse_permutation].reshape(
        variant_count, tile_count, 16, 16
    )
    reconstruction = reconstruction.permute(1, 0, 2, 3).contiguous()
    payload_shape = payload.shape[1:]
    payload = payload.reshape(variant_count, tile_count, *payload_shape)
    payload = payload.permute(1, 0, *range(2, payload.ndim)).contiguous()
    if current_payload.shape[1:] != payload_shape:
        raise ValueError("stored and proposed trellis payload shapes differ")

    candidates = torch.cat((current_tiles[:, None], reconstruction), dim=1)
    payloads = torch.cat((current_payload[:, None], payload), dim=1)
    flattened = payloads.reshape(tile_count, payloads.shape[1], -1)
    unique_per_tile = torch.ones(
        tile_count,
        dtype=torch.int64,
        device=payloads.device,
    )
    for candidate in range(1, flattened.shape[1]):
        duplicates = torch.any(
            torch.all(
                flattened[:, candidate : candidate + 1]
                == flattened[:, :candidate],
                dim=-1,
            ),
            dim=1,
        )
        unique_per_tile.add_(~duplicates)
    unique = int(unique_per_tile.sum().item())
    return candidates, payloads, unique


@torch.no_grad()
def refine_h2_viterbi_paths(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    initial_weight_q: torch.Tensor,
    initial_encoded: torch.Tensor,
    quantize_tiles: QuantizeTiles,
    config: H2ViterbiRefineConfig = H2ViterbiRefineConfig(),
    *,
    column_weights: torch.Tensor | None = None,
    guidance: GradientGuidance | None = None,
) -> H2ViterbiRefineResult:
    """Refine legal paths under dense-H plus an optional linear KL objective."""

    if weight.ndim != 2 or initial_weight_q.shape != weight.shape:
        raise ValueError("weight and reconstruction must have one rank-2 shape")
    rows, columns = weight.shape
    if rows % 16 or columns % 16:
        raise ValueError("weight dimensions must be multiples of 16")
    if hessian.shape != (rows, rows):
        raise ValueError("Hessian shape does not match the input dimension")
    expected_grid = (rows // 16, columns // 16)
    if initial_encoded.ndim < 3 or initial_encoded.shape[:2] != expected_grid:
        raise ValueError("stored trellis payload has the wrong tile grid")
    if config.sweeps <= 0:
        raise ValueError("sweeps must be positive")
    if guidance is not None:
        guidance.validate(weight.shape)

    device = weight.device
    hessian_work = hessian.to(device=device, dtype=config.score_dtype)
    if column_weights is None:
        column_weights_work = torch.ones(
            columns, dtype=config.score_dtype, device=device
        )
    else:
        if column_weights.numel() != columns:
            raise ValueError("column weights do not match the output dimension")
        column_weights_work = column_weights.reshape(-1).to(
            device=device, dtype=config.score_dtype
        )
        if bool(torch.any(column_weights_work < 0)):
            raise ValueError("column weights must be nonnegative")

    weight_work = weight.to(device)
    reconstruction = initial_weight_q.to(device).clone()
    encoded = initial_encoded.to(device).clone()
    dense_gradient_work = (
        None
        if guidance is None
        or isinstance(guidance, LowRankViterbiGradientGuidance)
        else guidance.gradient.to(device=device, dtype=config.score_dtype)
    )
    anchor_work = (
        None
        if guidance is None
        else guidance.anchor.to(device=device, dtype=config.score_dtype)
    )
    gradient_strength = 0.0 if guidance is None else float(guidance.strength)
    error = (weight_work - reconstruction).to(config.score_dtype)
    hessian_error = hessian_work @ error
    quadratic = float(
        torch.sum(error * hessian_error * column_weights_work.unsqueeze(0)).item()
    )
    linear = (
        0.0
        if guidance is None or anchor_work is None
        else (
            guidance.linear_term(reconstruction, dtype=config.score_dtype)
            if isinstance(guidance, LowRankViterbiGradientGuidance)
            else float(
                gradient_strength
                * torch.sum(
                    dense_gradient_work
                    * (reconstruction.to(config.score_dtype) - anchor_work)
                ).item()
            )
        )
    )
    objective = quadratic + linear
    sweep_stats = []
    row_blocks = rows // 16
    column_blocks = columns // 16
    permutation = _tensor_core_permutation(device)
    inverse_permutation = torch.argsort(permutation)

    for sweep in range(config.sweeps):
        before = objective
        quadratic_before = quadratic
        linear_before = linear
        accepted_tiles = 0
        unique_paths = 0
        reverse = not config.alternate_sweep_order or sweep % 2 == 0
        order: Iterable[int] = (
            range(row_blocks - 1, -1, -1) if reverse else range(row_blocks)
        )
        for row_block in order:
            start = row_block * 16
            stop = start + 16
            row_slice = slice(start, stop)
            hessian_block = hessian_work[row_slice, row_slice]
            target = _conditional_target(
                weight_work[row_slice].to(config.score_dtype),
                error[row_slice],
                hessian_error[row_slice],
                hessian_block,
                gradient_block=(
                    None
                    if guidance is None
                    else (
                        guidance.rows(
                            row_slice,
                            device=device,
                            dtype=config.score_dtype,
                        )
                        if isinstance(guidance, LowRankViterbiGradientGuidance)
                        else dense_gradient_work[row_slice]
                    )
                ),
                gradient_strength=gradient_strength,
                column_weights=column_weights_work,
            ).to(weight_work.dtype)
            target_tiles = target.reshape(16, column_blocks, 16).permute(1, 0, 2)
            current_tiles = (
                reconstruction[row_slice]
                .reshape(16, column_blocks, 16)
                .permute(1, 0, 2)
                .contiguous()
            )
            current_payload = encoded[row_block]
            candidates, payloads, unique = _candidate_paths(
                target_tiles.contiguous(),
                current_tiles,
                current_payload,
                hessian_block,
                permutation,
                inverse_permutation,
                quantize_tiles,
                config,
                sweep=sweep,
                row_block=row_block,
            )
            unique_paths += unique
            scores = _conditional_scores(
                target_tiles,
                candidates,
                hessian_block,
                column_weights_work.reshape(column_blocks, 16),
                config.score_dtype,
            )
            current_score = scores[:, 0]
            best_score, best_index = torch.min(scores, dim=1)
            tolerance = config.acceptance_abs_tol + (
                config.acceptance_rel_tol * current_score.abs()
            )
            accept = best_score + tolerance < current_score
            accepted_tiles += int(torch.count_nonzero(accept))
            if not bool(torch.any(accept)):
                continue

            tile_indices = torch.arange(column_blocks, device=device)
            selected_tiles = candidates[tile_indices, best_index]
            selected_payload = payloads[tile_indices, best_index]
            selected_tiles = torch.where(
                accept[:, None, None], selected_tiles, current_tiles
            )
            payload_mask = accept.reshape(
                accept.shape[0], *([1] * (selected_payload.ndim - 1))
            )
            selected_payload = torch.where(
                payload_mask, selected_payload, current_payload
            )
            new_block = selected_tiles.permute(1, 0, 2).reshape(16, columns)
            old_error = error[row_slice].clone()
            new_error = (weight_work[row_slice] - new_block).to(config.score_dtype)
            delta_error = new_error - old_error
            reconstruction[row_slice] = new_block
            encoded[row_block] = selected_payload
            error[row_slice] = new_error
            hessian_error.addmm_(hessian_work[:, row_slice], delta_error)

        hessian_error = hessian_work @ error
        quadratic = float(
            torch.sum(
                error * hessian_error * column_weights_work.unsqueeze(0)
            ).item()
        )
        linear = (
            0.0
            if guidance is None or anchor_work is None
            else (
                guidance.linear_term(reconstruction, dtype=config.score_dtype)
                if isinstance(guidance, LowRankViterbiGradientGuidance)
                else float(
                    gradient_strength
                    * torch.sum(
                        dense_gradient_work
                        * (reconstruction.to(config.score_dtype) - anchor_work)
                    ).item()
                )
            )
        )
        objective = quadratic + linear
        allowed_drift = max(1.0e-5, 1.0e-6 * abs(before))
        if objective > before + allowed_drift:
            raise RuntimeError(
                "dense-H objective increased; quantizer coordinates are inconsistent"
            )
        sweep_stats.append(
            H2ViterbiSweepStats(
                sweep=sweep,
                objective_before=before,
                objective_after=objective,
                quadratic_before=quadratic_before,
                quadratic_after=quadratic,
                linear_before=linear_before,
                linear_after=linear,
                relative_improvement=(before - objective)
                / max(abs(before), 1.0e-30),
                accepted_tiles=accepted_tiles,
                proposed_tiles=row_blocks * column_blocks,
                unique_candidate_paths=unique_paths,
            )
        )
        if config.stop_on_no_change and accepted_tiles == 0:
            break

    return H2ViterbiRefineResult(reconstruction, encoded, sweep_stats)


def config_from_quant_args(quant_args: dict[str, object]) -> H2ViterbiRefineConfig:
    """Build a refinement configuration from private encoder arguments."""

    return H2ViterbiRefineConfig(
        sweeps=int(quant_args.get("h2_viterbi_refine_sweeps", 1)),
        alternate_sweep_order=bool(
            quant_args.get("h2_viterbi_refine_alternate_order", True)
        ),
        dither_scales=tuple(
            float(value)
            for value in quant_args.get("h2_viterbi_refine_dither_scales", ())
        ),
        num_dither_patterns=int(
            quant_args.get("h2_viterbi_refine_patterns", 0)
        ),
        seed=int(quant_args.get("h2_viterbi_refine_seed", 0x48325643)),
        max_quantizer_batch=int(
            quant_args.get("h2_viterbi_refine_max_batch", 256)
        ),
        score_dtype=(
            torch.float64
            if quant_args.get("h2_viterbi_refine_fp64_score", False)
            else torch.float32
        ),
    )

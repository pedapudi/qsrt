#!/usr/bin/env python3
"""Run bounded, resumable CPU studies of coupled Kimi-K3 expert structure.

The numerical machinery lives in :mod:`qsrt.coupled_expert_study` and is
model independent.  This file is deliberately the thin adapter that knows the
Kimi tensor names, source MXFP4 container, routed capture, and selected QSRT
candidate layout.  It writes research reports only; it cannot materialize a
checkpoint or alter a runtime format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from qsrt import constants as C
from qsrt.capture import LayerSamples, index_cached_layer_samples
from qsrt.coupled_expert_study import (
    CoupledTriplet,
    RoutedOutputMetric,
    conditional_entropy_bits,
    encode_coupled_block_hadamard,
    entropy_bits,
    execute_coupled_block_hadamard,
    expert_hidden,
    apply_permutation_sign_gauge,
    blockwise_codebook_quantize,
    fit_cross_matrix_predictor,
    fit_function_space_correction,
    fit_metric_codebook,
    local_triplet_metrics,
    micro_neuron_energy_saliency,
    micro_neuron_fingerprints,
    pair_activation_metric,
    pair_residual_decomposition,
    quantize_metric_codebook,
    rademacher_projection,
    radial_tangent_decomposition,
    ridge_refit_down,
    route_error_covariance,
    search_expert_output_gain,
    situ_component_geometry,
    situ_value,
    sparse_fingerprint_alignment,
    temperature_scaled_situ,
)
from qsrt.io.mxfp4 import scale_factors, unpack_codes
from qsrt.io.stream import load_tensor
from qsrt.pack.qsrt_validation import decode_candidate_matrix
from qsrt.pack.qsrt_encoder import plan_qsrt_matrix
from qsrt.source_weights import OfficialMXFP4Store


KIND = "qsrt_k3_coupled_expert_cpu_study"
SCHEMA_VERSION = 1
DEFAULT_POOL = Path("/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-CANDIDATES-v1")
DEFAULT_FIT_CACHE = Path(
    "/data/datasets/kquant/captures/k3-denseh-broad-v6-1m-train-input-v1.kqsamples"
)
DEFAULT_VALIDATION_CACHE = Path(
    "/data/datasets/kquant/captures/k3-codec-diverse-validation-v3-128k-input-v1.kqsamples"
)
DEFAULT_DEST = Path("/data/kquant/research/k3-coupled-ptq-v1")
DEFAULT_LAYERS = (1, 8, 24, 40, 64, 80, 92)
CROSS_LAYERS = (1, 24, 64, 92)


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("expected a nonempty list of unique integers")
    return result


def _parse_stages(value: str) -> tuple[str, ...]:
    stages = tuple(item.strip() for item in value.split(",") if item.strip())
    allowed = {"native", "functional", "cross", "routed", "summary"}
    if not stages or any(stage not in allowed for stage in stages):
        raise argparse.ArgumentTypeError(f"stages must be drawn from {sorted(allowed)}")
    return stages


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _candidate_paths(pool: Path, layer: int) -> tuple[Path, Path]:
    stem = f"qsrt-layer-{layer:05d}"
    root = pool / "candidates"
    payload = root / f"{stem}.safetensors"
    metrics = root / f"{stem}.metrics.safetensors"
    if not payload.is_file() or not metrics.is_file():
        raise FileNotFoundError(f"candidate layer {layer} is incomplete in {pool}")
    return payload, metrics


def _load_output_metric(store: OfficialMXFP4Store, layer: int) -> RoutedOutputMetric:
    prefix = f"{C.LM_PREFIX}layers.{layer}.block_sparse_moe"
    gain = load_tensor(store.cache, f"{prefix}.routed_expert_norm.weight")
    projection = load_tensor(store.cache, C.latent_up_proj_tensor(layer))
    return RoutedOutputMetric(gain=gain.float(), projection=projection.float())


def _source_triplet(layer_store: Any, layer: int, expert: int) -> CoupledTriplet:
    return CoupledTriplet(
        layer_store.load_matrix(layer, expert, "w1"),
        layer_store.load_matrix(layer, expert, "w3"),
        layer_store.load_matrix(layer, expert, "w2"),
    )


def _candidate_triplet(
    reader: Any,
    *,
    layer: int,
    expert: int,
    r13: int,
    r2: int,
    schema: str,
    codebook: str,
) -> CoupledTriplet:
    device = torch.device("cpu")
    # Stored QSRT tensors decode in EXL physical [input, output] order.
    return CoupledTriplet(
        decode_candidate_matrix(
            reader,
            layer=layer,
            expert=expert,
            matrix="w1",
            mode_id=r13,
            device=device,
            logical_trellis_schema=schema,
            codebook=codebook,
        ).T.contiguous(),
        decode_candidate_matrix(
            reader,
            layer=layer,
            expert=expert,
            matrix="w3",
            mode_id=r13,
            device=device,
            logical_trellis_schema=schema,
            codebook=codebook,
        ).T.contiguous(),
        decode_candidate_matrix(
            reader,
            layer=layer,
            expert=expert,
            matrix="w2",
            mode_id=r2,
            device=device,
            logical_trellis_schema=schema,
            codebook=codebook,
        ).T.contiguous(),
    )


def _expert_occurrences(
    samples: LayerSamples,
    expert: int,
    *,
    maximum: int,
    balance_splits: bool = False,
) -> dict[str, torch.Tensor]:
    locations = torch.nonzero(samples.input_experts == expert, as_tuple=False)
    if locations.numel() == 0:
        raise ValueError(f"expert {expert} has no routed rows in the selected cache")
    if locations.shape[0] > maximum:
        if balance_splits:
            location_splits = samples.input_split.index_select(
                0, locations[:, 0]
            )

            def evenly(group: torch.Tensor, count: int) -> torch.Tensor:
                if count <= 0:
                    return group[:0]
                if group.shape[0] <= count:
                    return group
                positions = (
                    torch.linspace(0, group.shape[0] - 1, count)
                    .round()
                    .long()
                )
                return group.index_select(0, positions)

            fit = locations[location_splits == 0]
            confirmation = locations[location_splits == 1]
            fit_count = min(fit.shape[0], maximum // 2)
            confirmation_count = min(
                confirmation.shape[0], maximum - fit_count
            )
            remaining = maximum - fit_count - confirmation_count
            if remaining:
                fit_count += min(fit.shape[0] - fit_count, remaining)
                remaining = maximum - fit_count - confirmation_count
            if remaining:
                confirmation_count += min(
                    confirmation.shape[0] - confirmation_count, remaining
                )
            locations = torch.cat(
                (evenly(fit, fit_count), evenly(confirmation, confirmation_count))
            )
            locations = locations.index_select(
                0, torch.argsort(locations[:, 0])
            )
        else:
            positions = (
                torch.linspace(0, locations.shape[0] - 1, maximum)
                .round()
                .long()
            )
            locations = locations.index_select(0, positions)
    rows = locations[:, 0]
    slots = locations[:, 1]
    observations = samples.input_observations.index_select(0, rows)
    return {
        "inputs": samples.input_values.index_select(0, rows).float(),
        "gates": samples.input_gates[rows, slots].float(),
        "aggregate": samples.routed_latent.index_select(0, rows).float(),
        "rows": rows,
        "documents": torch.bitwise_right_shift(observations, 32),
        "split": samples.input_split.index_select(0, rows),
    }


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().double().reshape(-1).cpu()
    if flat.numel() == 0:
        return {}
    points = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], dtype=torch.float64)
    measured = torch.quantile(flat, points)
    return {
        name: float(value)
        for name, value in zip(("min", "p10", "p25", "median", "p75", "p90", "max"), measured)
    }


def _weighted_moments(values: torch.Tensor, row_weights: torch.Tensor) -> dict[str, float]:
    if values.ndim != 2 or row_weights.ndim != 1 or values.shape[0] != row_weights.numel():
        raise ValueError("weighted activation values and routed-row weights do not align")
    weights = row_weights.detach().double().clamp_min(0)
    denominator = weights.sum() * values.shape[1]
    if denominator <= 0:
        return {"mean": 0.0, "mean_absolute": 0.0, "rms": 0.0}
    measured = values.detach().double()
    return {
        "mean": float((weights[:, None] * measured).sum() / denominator),
        "mean_absolute": float((weights[:, None] * measured.abs()).sum() / denominator),
        "rms": float(torch.sqrt((weights[:, None] * measured.square()).sum() / denominator)),
    }


def _weighted_fraction(
    mask: torch.Tensor,
    row_weights: torch.Tensor,
) -> float:
    if mask.ndim != 2 or row_weights.ndim != 1 or mask.shape[0] != row_weights.numel():
        raise ValueError("weighted activation mask and routed-row weights do not align")
    weights = row_weights.detach().double().clamp_min(0)
    denominator = weights.sum() * mask.shape[1]
    if denominator <= 0:
        return 0.0
    return float((weights[:, None] * mask.detach().double()).sum() / denominator)


def _document_bootstrap(values: torch.Tensor, documents: torch.Tensor, *, seed: int) -> dict[str, float | int]:
    unique, inverse = torch.unique(documents.cpu(), return_inverse=True)
    totals = torch.zeros(unique.numel(), dtype=torch.float64)
    totals.scatter_add_(0, inverse, values.detach().double().cpu())
    if totals.numel() == 1:
        estimate = float(totals[0])
        return {"documents": 1, "mean_document_sse": estimate, "ci95_low": estimate, "ci95_high": estimate}
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(0, totals.numel(), (2000, totals.numel()), generator=generator)
    estimates = totals.index_select(0, draws.reshape(-1)).reshape(draws.shape).mean(dim=1)
    interval = torch.quantile(estimates, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "documents": int(totals.numel()),
        "mean_document_sse": float(totals.mean()),
        "ci95_low": float(interval[0]),
        "ci95_high": float(interval[1]),
    }


def _weighted_four_level_palette(codes: torch.Tensor, scales: torch.Tensor) -> dict[str, Any]:
    values = torch.tensor(C.E2M1_LUT, dtype=torch.float64)
    blocked = codes.reshape(*scales.shape, C.MXFP4_BLOCK)
    weights = torch.zeros(16, dtype=torch.float64)
    scale_weights = scale_factors(scales).double().square().unsqueeze(-1).expand_as(blocked)
    weights.scatter_add_(0, blocked.reshape(-1).long(), scale_weights.reshape(-1))
    centers = torch.tensor([-3.0, -0.5, 0.5, 3.0], dtype=torch.float64)
    for _ in range(100):
        assignment = (values[:, None] - centers[None, :]).square().argmin(dim=1)
        updated = centers.clone()
        for index in range(4):
            selected = assignment == index
            denominator = weights[selected].sum()
            if denominator > 0:
                updated[index] = (weights[selected] * values[selected]).sum() / denominator
        updated = updated.sort().values
        if torch.allclose(updated, centers, rtol=0, atol=1e-12):
            break
        centers = updated
    reconstruction = centers.index_select(0, assignment)
    error = (weights * (values - reconstruction).square()).sum()
    energy = (weights * values.square()).sum()
    return {
        "levels": centers.tolist(),
        "weighted_nmse": float(error / energy) if energy > 0 else 0.0,
        "code_assignment": assignment.tolist(),
    }


def native_source_report(store: OfficialMXFP4Store, layer: int, expert: int) -> dict[str, Any]:
    started = time.time()
    matrix_reports: dict[str, Any] = {}
    codes_by_matrix: dict[str, torch.Tensor] = {}
    with store.open_layer(layer, experts=(expert,)) as layer_store:
        for matrix in C.EXPERT_MATRICES:
            packed = layer_store.load_packed_matrix(layer, expert, matrix)
            codes = unpack_codes(packed.packed)
            codes_by_matrix[matrix] = codes
            scales = packed.scale
            delta = scales[:, 1:].to(torch.int16) - scales[:, :-1].to(torch.int16)
            scale_context = scales.repeat_interleave(C.MXFP4_BLOCK, dim=1)
            counts = torch.bincount(codes.reshape(-1).long(), minlength=16)
            matrix_reports[matrix] = {
                "weights": int(codes.numel()),
                "blocks": int(scales.numel()),
                "symbol_entropy_bits": entropy_bits(codes),
                "symbol_entropy_given_scale_bits": conditional_entropy_bits(codes, scale_context),
                "scale_entropy_bits": entropy_bits(scales),
                "within_row_scale_delta_entropy_bits": entropy_bits(delta),
                "positive_zero_fraction": float(counts[0] / counts.sum()),
                "negative_zero_fraction": float(counts[8] / counts.sum()),
                "zero_fraction": float((counts[0] + counts[8]) / counts.sum()),
                "code_counts": counts.tolist(),
                "four_level_native_palette": _weighted_four_level_palette(codes, scales),
            }
    w1 = codes_by_matrix["w1"]
    w3 = codes_by_matrix["w3"]
    w2_t = codes_by_matrix["w2"].T.contiguous()
    context_13 = w1.to(torch.int16) * 16 + w3.to(torch.int16)
    joint = {
        "h_w3_given_w1_bits": conditional_entropy_bits(w3, w1),
        "h_w2_given_w1_bits": conditional_entropy_bits(w2_t, w1),
        "h_w2_given_w3_bits": conditional_entropy_bits(w2_t, w3),
        "h_w2_given_w1_w3_bits": conditional_entropy_bits(w2_t, context_13),
        "signed_zero_side_channel_capacity_bpw": float(
            ((w1 == 0) | (w1 == 8)).float().mean()
            + ((w3 == 0) | (w3 == 8)).float().mean()
            + ((w2_t == 0) | (w2_t == 8)).float().mean()
        )
        / 3.0,
    }
    return {
        "kind": KIND,
        "stage": "native",
        "layer": layer,
        "expert": expert,
        "matrices": matrix_reports,
        "coupled_symbols": joint,
        "seconds": time.time() - started,
    }


def _pair_codebook_report(
    source: CoupledTriplet,
    metrics: torch.Tensor,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    neurons = torch.randint(0, source.intermediate, (samples,), generator=generator)
    coordinates = torch.randint(0, source.hidden, (samples,), generator=generator)
    values = torch.stack(
        (source.gate[neurons, coordinates], source.up[neurons, coordinates]), dim=1
    ).float()
    sample_metrics = metrics.index_select(0, neurons).float()
    split = max(samples * 3 // 4, 16)
    split = min(split, samples - 1)
    train_values, validation_values = values[:split], values[split:]
    train_metrics, validation_metrics = sample_metrics[:split], sample_metrics[split:]
    metric_codebook = fit_metric_codebook(
        train_values, 16, metrics=train_metrics, iterations=25, seed=seed
    )
    euclidean_codebook = fit_metric_codebook(train_values, 16, iterations=25, seed=seed)

    def score(codebook: torch.Tensor) -> tuple[float, float]:
        reconstructed, _ = quantize_metric_codebook(
            validation_values, codebook, metrics=validation_metrics
        )
        residual = validation_values - reconstructed
        functional = torch.einsum(
            "ri,rij,rj->r", residual, validation_metrics, residual
        ).double().mean()
        euclidean = residual.double().square().sum(dim=1).mean()
        return float(functional), float(euclidean)

    metric_functional, metric_euclidean = score(metric_codebook)
    euclidean_functional, euclidean_euclidean = score(euclidean_codebook)
    return {
        "training_pairs": split,
        "validation_pairs": samples - split,
        "metric_codebook": metric_codebook.tolist(),
        "metric_functional_mse": metric_functional,
        "metric_euclidean_mse": metric_euclidean,
        "euclidean_functional_mse": euclidean_functional,
        "euclidean_euclidean_mse": euclidean_euclidean,
        "functional_improvement": (
            1.0 - metric_functional / euclidean_functional
            if euclidean_functional > 0
            else 0.0
        ),
        "payload_bpw_if_pair_nibble_plus_w2_2bit": 2.0,
    }


def _relative_improvement(before: torch.Tensor, after: torch.Tensor) -> float:
    denominator = before.double().square().sum()
    numerator = after.double().square().sum()
    return float(1.0 - numerator / denominator) if denominator > 0 else 0.0


def _two_bit_transform_proxy(
    source: CoupledTriplet,
    inputs: torch.Tensor,
    source_hidden: torch.Tensor,
    source_output: torch.Tensor,
    gates: torch.Tensor,
    aggregate: torch.Tensor,
    output_metric: RoutedOutputMetric,
    row_split: torch.Tensor,
) -> dict[str, Any]:
    """Compare exact reparameterizations under one generic 2-bit quantizer."""

    if row_split.shape != (inputs.shape[0],):
        raise ValueError("2-bit proxy split does not align with routed rows")
    fit_mask = row_split == 0
    confirmation_mask = row_split == 1
    if not bool(fit_mask.any()) or not bool(confirmation_mask.any()):
        indices = torch.arange(inputs.shape[0])
        fit_mask = indices % 2 == 0
        confirmation_mask = ~fit_mask

    def score_rows(output: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
        error = output[mask] - source_output[mask]
        exact = output_metric.exact_delta(
            aggregate[mask], gates[mask, None] * error
        )
        expert_sse = error.double().square().sum()
        return {
            "rows": int(mask.sum()),
            "expert_sse": float(expert_sse),
            "post_projection_sse": float(exact.double().square().sum()),
        }

    def score(output: torch.Tensor) -> dict[str, Any]:
        return {
            **score_rows(output, torch.ones(inputs.shape[0], dtype=torch.bool)),
            "fit": score_rows(output, fit_mask),
            "confirmation": score_rows(output, confirmation_mask),
        }

    q_gate = blockwise_codebook_quantize(source.gate)
    q_up = blockwise_codebook_quantize(source.up)
    q_down = blockwise_codebook_quantize(source.down)
    baseline_hidden = situ_value(inputs @ q_gate.T, inputs @ q_up.T)
    baseline_output = baseline_hidden @ q_down.T
    baseline = score(baseline_output)

    input_scale = inputs.square().mean(dim=0).sqrt().clamp_min(1e-6)
    input_scale /= input_scale.log().mean().exp()
    scaled_gate = blockwise_codebook_quantize(source.gate / input_scale[None, :])
    scaled_up = blockwise_codebook_quantize(source.up / input_scale[None, :])
    scaled_inputs = inputs * input_scale[None, :]
    input_gauge_hidden = situ_value(
        scaled_inputs @ scaled_gate.T, scaled_inputs @ scaled_up.T
    )
    input_gauge = score(input_gauge_hidden @ q_down.T)
    input_gauge["fp16_metadata_bpw"] = float(
        16 * source.hidden / (C.NUM_EXPERTS * source.numel)
    )

    column_scale = source.down.square().mean(dim=0).sqrt().clamp_min(1e-6)
    column_scale /= column_scale.log().mean().exp()
    balanced_down = blockwise_codebook_quantize(
        source.down / column_scale[None, :]
    )
    postactivation = score((source_hidden * column_scale[None, :]) @ balanced_down.T)
    down_only_baseline = score(source_hidden @ q_down.T)
    postactivation["down_only_expert_sse_improvement"] = float(
        1.0
        - postactivation["expert_sse"]
        / max(down_only_baseline["expert_sse"], 1e-30)
    )
    postactivation["fp16_metadata_bpw"] = float(
        16 * source.intermediate / source.numel
    )

    temperature_scale = column_scale.sqrt()
    temperature_gate = blockwise_codebook_quantize(
        source.gate * temperature_scale[:, None]
    )
    temperature_up = blockwise_codebook_quantize(
        source.up * temperature_scale[:, None]
    )
    temperature_hidden = temperature_scaled_situ(
        inputs @ temperature_gate.T,
        inputs @ temperature_up.T,
        temperature_scale,
        temperature_scale,
    )
    temperature = score(temperature_hidden @ balanced_down.T)
    temperature["fp16_metadata_bpw"] = float(
        16 * 2 * source.intermediate / source.numel
    )
    temperature["requires_activation_temperature"] = True

    # q(u) = 25*tanh(u/25) is almost linear on the routed Kimi rows.  In that
    # regime, scaling each W3 row by c and the matching W2 column by 1/c is an
    # approximately function-preserving gauge that can be baked into stored
    # weights.  Scalar quantization of a complete W3 row is scale-homogeneous,
    # while W2 quantization couples groups of adjacent columns.  Compare the
    # symmetric up/down RMS rule with W2-oriented RMS and absmax rules instead
    # of treating one arbitrary equilibration policy as the gauge itself.
    up_rms = source.up.float().square().mean(dim=1).sqrt().clamp_min(1e-8)
    down_rms = source.down.float().square().mean(dim=0).sqrt().clamp_min(1e-8)
    down_absmax = source.down.float().abs().amax(dim=0).clamp_min(1e-8)

    def normalized_scale(value: torch.Tensor) -> torch.Tensor:
        return value / value.log().mean().exp()

    scale_gauge_policies = {
        "up_down_rms": normalized_scale((down_rms / up_rms).sqrt()),
        "down_rms": normalized_scale(down_rms),
        "down_absmax": normalized_scale(down_absmax),
    }
    scale_gauge_arms: dict[str, dict[str, float]] = {}
    for policy, proposal in scale_gauge_policies.items():
        for strength in (0.25, 0.5, 0.75, 1.0):
            scale = proposal.pow(strength).clamp(0.5, 2.0)
            full_precision_hidden = situ_value(
                inputs @ source.gate.T,
                inputs @ (source.up * scale[:, None]).T,
            )
            full_precision_output = full_precision_hidden @ (
                source.down / scale[None, :]
            ).T
            quantized_up = blockwise_codebook_quantize(
                source.up * scale[:, None]
            )
            quantized_down = blockwise_codebook_quantize(
                source.down / scale[None, :]
            )
            quantized_hidden = situ_value(
                inputs @ q_gate.T, inputs @ quantized_up.T
            )
            arm = score(quantized_hidden @ quantized_down.T)
            full_precision_error = full_precision_output - source_output
            arm.update(
                {
                    "policy": policy,
                    "strength": strength,
                    "scale_minimum": float(scale.min()),
                    "scale_maximum": float(scale.max()),
                    "full_precision_expert_relative_sse": float(
                        full_precision_error.double().square().sum()
                        / source_output.double().square().sum().clamp_min(1e-30)
                    ),
                    "expert_sse_improvement_vs_baseline": float(
                        1.0
                        - arm["expert_sse"]
                        / max(baseline["expert_sse"], 1e-30)
                    ),
                    "post_projection_sse_improvement_vs_baseline": float(
                        1.0
                        - arm["post_projection_sse"]
                        / max(baseline["post_projection_sse"], 1e-30)
                    ),
                    "fit_post_projection_sse_improvement_vs_baseline": float(
                        1.0
                        - arm["fit"]["post_projection_sse"]
                        / max(baseline["fit"]["post_projection_sse"], 1e-30)
                    ),
                    "confirmation_post_projection_sse_improvement_vs_baseline": float(
                        1.0
                        - arm["confirmation"]["post_projection_sse"]
                        / max(
                            baseline["confirmation"]["post_projection_sse"],
                            1e-30,
                        )
                    ),
                }
            )
            scale_gauge_arms[f"{policy}:{strength:.2f}"] = arm
    scale_gauge_arms["identity:0.00"] = {
        "policy": "identity",
        "strength": 0.0,
        "scale_minimum": 1.0,
        "scale_maximum": 1.0,
        "full_precision_expert_relative_sse": 0.0,
        "expert_sse": baseline["expert_sse"],
        "post_projection_sse": baseline["post_projection_sse"],
        "fit": baseline["fit"],
        "confirmation": baseline["confirmation"],
        "expert_sse_improvement_vs_baseline": 0.0,
        "post_projection_sse_improvement_vs_baseline": 0.0,
        "fit_post_projection_sse_improvement_vs_baseline": 0.0,
        "confirmation_post_projection_sse_improvement_vs_baseline": 0.0,
    }
    scale_gauge_best = max(
        scale_gauge_arms.values(),
        key=lambda arm: arm["post_projection_sse_improvement_vs_baseline"],
    )
    scale_gauge_fit_key, scale_gauge_fit_selected = max(
        scale_gauge_arms.items(),
        key=lambda item: item[1][
            "fit_post_projection_sse_improvement_vs_baseline"
        ],
    )
    scale_gauge_confirmation_accepted = (
        scale_gauge_fit_selected[
            "confirmation_post_projection_sse_improvement_vs_baseline"
        ]
        > 0.0
    )

    # Interleave the two branches before the left transform so each block can
    # exploit their coupled functional geometry, then invert before SiTU.
    # Draw zero is the plain Walsh-Hadamard transform.  The other draws retain
    # exactly the same butterfly topology while changing only the two coupled
    # intermediate-side Rademacher sign streams.  Residual-side signs remain
    # fixed at layer-shared draw zero in this expert-local screen.
    hadamard_arms: dict[str, dict[str, Any]] = {}
    for draw in range(8):
        transformed = encode_coupled_block_hadamard(
            source,
            block_size=512,
            residual_rotation_draw=0,
            intermediate_rotation_draw=draw,
        )
        transformed = CoupledTriplet(
            *(
                blockwise_codebook_quantize(value)
                for value in transformed.tensors()
            )
        )
        explicit_output = execute_coupled_block_hadamard(
            inputs,
            transformed,
            block_size=512,
            residual_rotation_draw=0,
            intermediate_rotation_draw=draw,
        )
        arm = score(explicit_output)
        arm.update(
            {
                "residual_rotation_draw": 0,
                "intermediate_rotation_draw": draw,
                "metadata_bpw": 0.0,
                "runtime": "four signed block-512 boundary Hadamards per selected expert",
            }
        )
        hadamard_arms[str(draw)] = arm

    for arm in (
        input_gauge,
        postactivation,
        temperature,
        *hadamard_arms.values(),
    ):
        arm["expert_sse_improvement_vs_baseline"] = float(
            1.0 - arm["expert_sse"] / max(baseline["expert_sse"], 1e-30)
        )
        arm["post_projection_sse_improvement_vs_baseline"] = float(
            1.0
            - arm["post_projection_sse"]
            / max(baseline["post_projection_sse"], 1e-30)
        )
        arm["fit_post_projection_sse_improvement_vs_baseline"] = float(
            1.0
            - arm["fit"]["post_projection_sse"]
            / max(baseline["fit"]["post_projection_sse"], 1e-30)
        )
        arm["confirmation_post_projection_sse_improvement_vs_baseline"] = float(
            1.0
            - arm["confirmation"]["post_projection_sse"]
            / max(baseline["confirmation"]["post_projection_sse"], 1e-30)
        )
    hadamard_best = max(
        hadamard_arms.values(),
        key=lambda arm: arm["post_projection_sse_improvement_vs_baseline"],
    )
    hadamard_fit_key, hadamard_fit_selected = max(
        hadamard_arms.items(),
        key=lambda item: item[1][
            "fit_post_projection_sse_improvement_vs_baseline"
        ],
    )
    hadamard_draw_zero = hadamard_arms["0"]
    hadamard_confirmation_accepted = (
        hadamard_fit_selected[
            "confirmation_post_projection_sse_improvement_vs_baseline"
        ]
        >= hadamard_draw_zero[
            "confirmation_post_projection_sse_improvement_vs_baseline"
        ]
    )
    hadamard_applied = (
        hadamard_fit_selected
        if hadamard_confirmation_accepted
        else hadamard_draw_zero
    )
    return {
        "quantizer": "four-level symmetric block-32 scalar proxy",
        "payload_bpw": 2.0,
        "baseline": baseline,
        "common_input_diagonal_gauge": input_gauge,
        "exact_postactivation_equilibration": postactivation,
        "temperature_equilibration": temperature,
        "approximate_w3_w2_scale_gauge": {
            "selection_metric": "post_projection_sse_improvement_vs_baseline",
            "best": scale_gauge_best,
            "fit_selected": {
                "arm": scale_gauge_fit_key,
                "confirmation_accepted": scale_gauge_confirmation_accepted,
                "applied_arm": (
                    scale_gauge_fit_key
                    if scale_gauge_confirmation_accepted
                    else "identity:0.00"
                ),
                "applied_confirmation_post_projection_sse_improvement_vs_baseline": (
                    scale_gauge_fit_selected[
                        "confirmation_post_projection_sse_improvement_vs_baseline"
                    ]
                    if scale_gauge_confirmation_accepted
                    else 0.0
                ),
                **scale_gauge_fit_selected,
            },
            "arms": scale_gauge_arms,
            "metadata_bpw": 0.0,
            "runtime": "none; transformed weights are baked into the checkpoint",
            "warning": "approximately, not exactly, function preserving because tanh is not homogeneous",
        },
        "expert_local_two_sided_block_hadamard": {
            "selection_metric": "fit post-projection SSE",
            "rotation_draws": len(hadamard_arms),
            "best": hadamard_best,
            "fit_selected": {
                "arm": hadamard_fit_key,
                "confirmation_accepted_over_draw_zero": (
                    hadamard_confirmation_accepted
                ),
                "applied_arm": (
                    hadamard_fit_key
                    if hadamard_confirmation_accepted
                    else "0"
                ),
                "applied_confirmation_post_projection_sse_improvement_vs_baseline": (
                    hadamard_applied[
                        "confirmation_post_projection_sse_improvement_vs_baseline"
                    ]
                ),
                **hadamard_fit_selected,
            },
            "arms": hadamard_arms,
            "metadata_bpw": 0.0,
            "runtime": "four signed block-512 boundary Hadamards per selected expert",
        },
        "warning": "proxy ranking only; this is not SQG, MCG, MXFP4, or a serving format",
    }


def functional_report(
    store: OfficialMXFP4Store,
    reader: Any,
    samples: LayerSamples,
    output_metric: RoutedOutputMetric,
    *,
    layer: int,
    expert: int,
    r13: int,
    r2: int,
    block_contexts: torch.Tensor,
    schema: str,
    codebook: str,
    maximum_rows: int,
    triplet_samples: int,
    pair_samples: int,
    two_bit_proxy: bool,
    seed: int,
) -> dict[str, Any]:
    started = time.time()
    rows = _expert_occurrences(
        samples,
        expert,
        maximum=maximum_rows,
        balance_splits=two_bit_proxy,
    )
    inputs = rows["inputs"]
    gates = rows["gates"]
    aggregate = rows["aggregate"]
    with store.open_layer(layer, experts=(expert,)) as layer_store:
        canonical_source = _source_triplet(layer_store, layer, expert)
    upstream_plan = plan_qsrt_matrix(
        block_contexts,
        r13,
        matrix="w1",
        layout="importance_ordered",
    )
    down_plan = plan_qsrt_matrix(
        block_contexts,
        r2,
        matrix="w2",
        layout="importance_ordered",
    )
    if not torch.equal(upstream_plan.physical_permutation, down_plan.physical_permutation):
        raise ValueError("selected r13/r2 payloads do not share a physical permutation")
    source = apply_permutation_sign_gauge(
        canonical_source,
        upstream_plan.physical_permutation,
        torch.ones(canonical_source.intermediate),
    )
    candidate = _candidate_triplet(
        reader,
        layer=layer,
        expert=expert,
        r13=r13,
        r2=r2,
        schema=schema,
        codebook=codebook,
    )

    source_gate = inputs @ source.gate.T
    source_up = inputs @ source.up.T
    source_hidden = situ_value(source_gate, source_up)
    source_output = source_hidden @ source.down.T
    candidate_gate = inputs @ candidate.gate.T
    candidate_up = inputs @ candidate.up.T
    candidate_hidden = situ_value(candidate_gate, candidate_up)
    candidate_output = candidate_hidden @ candidate.down.T
    expert_error = candidate_output - source_output
    routed_error = gates[:, None] * expert_error
    exact_delta = output_metric.exact_delta(aggregate, routed_error)
    linear_delta = output_metric.jacobian_vectors(aggregate, routed_error)
    radial = radial_tangent_decomposition(aggregate, routed_error)
    candidate_aggregate = aggregate + routed_error
    cosine = torch.nn.functional.cosine_similarity(aggregate, candidate_aggregate, dim=1)

    pair_summary = pair_activation_metric(
        source_gate,
        source_up,
        row_weights=gates.square(),
    )
    activation = situ_component_geometry(source_gate, source_up)
    activation_weights = gates.square()
    residual_decomposition = pair_residual_decomposition(
        inputs, source, candidate, route_gates=gates
    )
    generator = torch.Generator().manual_seed(seed + layer * 1009 + expert)
    local_count = min(triplet_samples, source.intermediate, source.hidden)
    neuron_indices = torch.randint(0, source.intermediate, (local_count,), generator=generator)
    coordinate_indices = torch.randint(0, source.hidden, (local_count,), generator=generator)
    triplet_metrics = local_triplet_metrics(
        inputs,
        source,
        route_gates=gates,
        aggregate=aggregate,
        output_metric=output_metric,
        neuron_indices=neuron_indices,
        coordinate_indices=coordinate_indices,
    )
    triplet_eigenvalues = torch.linalg.eigvalsh(triplet_metrics).clamp_min(0)

    gain_grid = torch.linspace(0.8, 1.2, 81)
    fitted_gain = search_expert_output_gain(
        aggregate, source_output, candidate_output, gates, output_metric, gain_grid
    )
    gained_error = gates[:, None] * (
        candidate_output * fitted_gain["gain"] - source_output
    )
    gain_exact = output_metric.exact_delta(aggregate, gained_error)

    indices = torch.arange(inputs.shape[0])
    fit = indices % 2 == 0
    validation = ~fit
    interventions: dict[str, Any] = {}
    if int(validation.sum()) > 0 and int(fit.sum()) > 0:
        refit = ridge_refit_down(
            candidate.down,
            source_hidden[fit],
            candidate_hidden[fit],
            target_down=source.down,
            regularization=1e-2,
        )
        refit_output = candidate_hidden[validation] @ refit.T
        interventions["w2_ridge_refit"] = {
            "validation_expert_sse_improvement": _relative_improvement(
                expert_error[validation], refit_output - source_output[validation]
            ),
            "fp16_delta_bpw_if_stored_dense": 16.0 / 3.0,
            "oracle_only": True,
        }

        target_error = source_output - candidate_output
        bias = target_error[fit].mean(dim=0)
        bias_error = candidate_output[validation] + bias - source_output[validation]
        interventions["output_bias"] = {
            "validation_expert_sse_improvement": _relative_improvement(
                expert_error[validation], bias_error
            ),
            "fp16_metadata_bpw": float(16 * source.hidden / source.numel),
        }
        correction_rank = min(4, int(fit.sum()), inputs.shape[1], source.hidden)
        correction_left, correction_right = fit_function_space_correction(
            inputs[fit],
            target_error[fit],
            rank=correction_rank,
            regularization=1e-2,
        )
        corrected_error = (
            candidate_output[validation]
            + inputs[validation] @ correction_left @ correction_right
            - source_output[validation]
        )
        interventions["rank4_function_correction"] = {
            "rank": correction_rank,
            "validation_expert_sse_improvement": _relative_improvement(
                expert_error[validation], corrected_error
            ),
            "fp16_metadata_bpw": float(
                16 * correction_rank * (source.hidden + source.hidden) / source.numel
            ),
        }

    saliency = micro_neuron_energy_saliency(source_hidden, source.down, gates)
    saliency_order = torch.argsort(saliency)
    pruning: dict[str, Any] = {}
    for fraction in (0.01, 0.05, 0.10, 0.20):
        count = max(1, round(source.intermediate * fraction))
        removed = saliency_order[:count]
        pruned_error = -(source_hidden[:, removed] @ source.down[:, removed].T)
        pruning[f"{fraction:.2f}"] = {
            "neurons_removed": count,
            "saliency_fraction": float(saliency[removed].sum() / saliency.sum()),
            "expert_output_energy_fraction": float(
                pruned_error.double().square().sum()
                / source_output.double().square().sum().clamp_min(1e-30)
            ),
            "survivor_rate_at_two_bpw_budget": float(2.0 / (1.0 - count / source.intermediate)),
        }

    weight_metrics = {}
    for name, source_weight, candidate_weight in zip(
        C.EXPERT_MATRICES, source.tensors(), candidate.tensors()
    ):
        difference = candidate_weight.double() - source_weight.double()
        weight_metrics[name] = {
            "sse": float(difference.square().sum()),
            "nmse": float(difference.square().sum() / source_weight.double().square().sum()),
            "maximum_absolute_error": float(difference.abs().max()),
        }
    predictor = {
        axis: {
            "residual_fraction": fit_cross_matrix_predictor(source, per=axis)[
                "residual_fraction"
            ],
            "fp16_coefficient_bpw": float(
                16 * 2 * (source.intermediate if axis == "neuron" else source.hidden)
                / source.numel
            ),
        }
        for axis in ("neuron", "hidden")
    }
    row_exact_sse = exact_delta.double().square().sum(dim=1)
    transform_proxy = (
        _two_bit_transform_proxy(
            source,
            inputs,
            source_hidden,
            source_output,
            gates,
            aggregate,
            output_metric,
            rows["split"],
        )
        if two_bit_proxy
        else None
    )
    return {
        "kind": KIND,
        "stage": "functional",
        "layer": layer,
        "expert": expert,
        "selected_modes": {"r13": r13, "r2": r2},
        "rows": int(inputs.shape[0]),
        "documents": int(torch.unique(rows["documents"]).numel()),
        "weight_distortion": weight_metrics,
        "section_5_0_activation_geometry": {
            "normalized_gate": {
                "quantiles": _quantiles(activation.normalized_gate),
                "route_weighted_moments": _weighted_moments(
                    activation.normalized_gate, activation_weights
                ),
            },
            "normalized_up": {
                "quantiles": _quantiles(activation.normalized_up),
                "route_weighted_moments": _weighted_moments(
                    activation.normalized_up, activation_weights
                ),
            },
            "gate_factor_derivative": {
                "quantiles": _quantiles(activation.gate_derivative),
                "route_weighted_moments": _weighted_moments(
                    activation.gate_derivative, activation_weights
                ),
            },
            "up_factor_derivative": {
                "quantiles": _quantiles(activation.up_derivative),
                "route_weighted_moments": _weighted_moments(
                    activation.up_derivative, activation_weights
                ),
                "route_weighted_fraction_at_least": {
                    "0.90": _weighted_fraction(
                        activation.up_derivative >= 0.90, activation_weights
                    ),
                    "0.95": _weighted_fraction(
                        activation.up_derivative >= 0.95, activation_weights
                    ),
                    "0.99": _weighted_fraction(
                        activation.up_derivative >= 0.99, activation_weights
                    ),
                },
            },
        },
        "section_5_1_pair_geometry": {
            "small_eigenvalue_fraction": _quantiles(pair_summary.small_eigenvalue_fraction),
            "condition_number": _quantiles(pair_summary.condition_number),
            "candidate_residual": residual_decomposition,
            "pair_codebook": _pair_codebook_report(
                source,
                pair_summary.metric,
                samples=pair_samples,
                seed=seed + layer * 1009 + expert,
            ),
        },
        "section_5_2_triplet_geometry": {
            "samples": local_count,
            "eigenvalues": {
                "small": _quantiles(triplet_eigenvalues[:, 0]),
                "middle": _quantiles(triplet_eigenvalues[:, 1]),
                "large": _quantiles(triplet_eigenvalues[:, 2]),
            },
            "small_over_trace": _quantiles(
                triplet_eigenvalues[:, 0]
                / triplet_eigenvalues.sum(dim=1).clamp_min(1e-30)
            ),
            "off_diagonal_energy_fraction": _quantiles(
                (
                    triplet_metrics
                    - torch.diag_embed(torch.diagonal(triplet_metrics, dim1=1, dim2=2))
                ).square().sum(dim=(1, 2))
                / triplet_metrics.square().sum(dim=(1, 2)).clamp_min(1e-30)
            ),
        },
        "section_5_3_post_rms_geometry": {
            "routed_latent_sse": float(routed_error.double().square().sum()),
            "radial_fraction": float(radial["radial_energy"].sum() / radial["total_energy"].sum()),
            "exact_post_projection_sse": float(exact_delta.double().square().sum()),
            "linear_post_projection_sse": float(linear_delta.double().square().sum()),
            "linear_over_exact": float(
                linear_delta.double().square().sum()
                / exact_delta.double().square().sum().clamp_min(1e-30)
            ),
            "aggregate_cosine_error_mean": float((1.0 - cosine).double().mean()),
            "document_bootstrap": _document_bootstrap(
                row_exact_sse, rows["documents"], seed=seed + 31 * layer + expert
            ),
            "expert_gain": {
                **fitted_gain,
                "exact_sse_at_fitted_gain": float(gain_exact.double().square().sum()),
                "improvement": float(
                    1.0
                    - gain_exact.double().square().sum()
                    / exact_delta.double().square().sum().clamp_min(1e-30)
                ),
                "metadata_bpw_fp16": float(16 / source.numel),
            },
        },
        "cross_matrix_prediction": predictor,
        "exact_transform_2bit_proxy": transform_proxy,
        "micro_neuron_pruning": pruning,
        "tiny_corrections": interventions,
        "seconds": time.time() - started,
    }


def _basis_curve(vectors: torch.Tensor, maximum_rank: int = 8) -> dict[str, float]:
    measured = vectors.float()
    gram = measured @ measured.T
    energy = torch.linalg.eigvalsh(gram).double().clamp_min_(0).flip(0)
    total = energy.sum().clamp_min(1e-30)
    return {
        str(rank): float(1.0 - energy[:rank].sum() / total)
        for rank in range(1, min(maximum_rank, vectors.shape[0]) + 1)
    }


def _batched_basis_curve(
    vectors: torch.Tensor, maximum_rank: int = 8
) -> dict[str, float]:
    """Return aggregate residual energy for one basis fitted per first axis."""

    if vectors.ndim != 3 or vectors.shape[1] < 2:
        raise ValueError("batched basis vectors must be [groups, samples, features]")
    measured = vectors.float()
    gram = measured @ measured.transpose(1, 2)
    energy = torch.linalg.eigvalsh(gram).double().clamp_min_(0).flip(1)
    total = energy.sum().clamp_min(1e-30)
    return {
        str(rank): float(1.0 - energy[:, :rank].sum() / total)
        for rank in range(1, min(maximum_rank, vectors.shape[1]) + 1)
    }


def cross_expert_report(
    store: OfficialMXFP4Store,
    *,
    layer: int,
    experts: tuple[int, ...],
    projection_width: int,
    coordinate_samples: int,
    seed: int,
) -> dict[str, Any]:
    if len(experts) < 2:
        raise ValueError("cross-expert study needs at least two experts")
    started = time.time()
    projection = rademacher_projection(C.LATENT, projection_width, seed=seed + layer)
    generator = torch.Generator().manual_seed(seed + 101 * layer)
    coordinates = torch.randint(0, C.LATENT, (coordinate_samples,), generator=generator)
    fingerprints: list[torch.Tensor] = []
    signs: list[torch.Tensor] = []
    triplets: list[CoupledTriplet] = []
    with store.open_layer(layer, experts=experts) as layer_store:
        for expert in experts:
            triplet = _source_triplet(layer_store, layer, expert)
            fingerprint, sign = micro_neuron_fingerprints(triplet, projection)
            triplets.append(triplet)
            fingerprints.append(fingerprint)
            signs.append(sign)
    template = fingerprints[0]
    alignments = [torch.arange(template.shape[0])]
    raw_distance = [0.0]
    aligned_distance = [0.0]
    for fingerprint in fingerprints[1:]:
        alignment = sparse_fingerprint_alignment(fingerprint, template)
        alignments.append(alignment)
        raw_distance.append(float((fingerprint - template).square().sum(dim=1).mean()))
        aligned_distance.append(
            float((fingerprint.index_select(0, alignment) - template).square().sum(dim=1).mean())
        )

    raw_vectors = []
    aligned_vectors = []
    raw_neuron_vectors = []
    aligned_neuron_vectors = []
    for triplet, sign, alignment in zip(triplets, signs, alignments):
        raw = torch.cat(
            (
                triplet.gate.index_select(1, coordinates),
                triplet.up.index_select(1, coordinates),
                triplet.down.index_select(0, coordinates).T,
            ),
            dim=1,
        )
        aligned = torch.cat(
            (
                triplet.gate.index_select(0, alignment).index_select(1, coordinates),
                (
                    triplet.up * sign[:, None]
                ).index_select(0, alignment).index_select(1, coordinates),
                (
                    triplet.down * sign[None, :]
                ).index_select(1, alignment).index_select(0, coordinates).T,
            ),
            dim=1,
        )
        raw_vectors.append(raw.reshape(-1))
        aligned_vectors.append(aligned.reshape(-1))
        raw_neuron_vectors.append(raw)
        aligned_neuron_vectors.append(aligned)
    raw_matrix = torch.stack(raw_vectors)
    aligned_matrix = torch.stack(aligned_vectors)
    raw_by_neuron = torch.stack(raw_neuron_vectors, dim=1)
    aligned_by_neuron = torch.stack(aligned_neuron_vectors, dim=1)
    null_generator = torch.Generator().manual_seed(seed + 65537 * layer)
    isotropic_null = torch.randn(
        aligned_by_neuron.shape, generator=null_generator
    )
    null_energy = isotropic_null.square().sum(dim=(1, 2)).clamp_min_(1e-30)
    real_energy = aligned_by_neuron.square().sum(dim=(1, 2))
    isotropic_null *= (real_energy / null_energy).sqrt()[:, None, None]
    null_curve = _batched_basis_curve(isotropic_null)
    aligned_neuron_curve = _batched_basis_curve(aligned_by_neuron)
    maximum_rank = min(8, len(experts))
    return {
        "kind": KIND,
        "stage": "cross",
        "layer": layer,
        "experts": list(experts),
        "projection_width": projection_width,
        "coordinate_samples_per_neuron": coordinate_samples,
        "alignment": {
            "raw_fingerprint_mse": raw_distance,
            "aligned_fingerprint_mse": aligned_distance,
            "mean_improvement": float(
                1.0 - sum(aligned_distance[1:]) / max(sum(raw_distance[1:]), 1e-30)
            ),
        },
        "sampled_shared_basis": {
            "raw_residual_fraction_by_rank": _basis_curve(raw_matrix),
            "aligned_residual_fraction_by_rank": _basis_curve(aligned_matrix),
            "raw_per_neuron_residual_fraction_by_rank": _batched_basis_curve(
                raw_by_neuron
            ),
            "aligned_per_neuron_residual_fraction_by_rank": aligned_neuron_curve,
            "isotropic_null_per_neuron_residual_fraction_by_rank": null_curve,
            "aligned_per_neuron_excess_captured_fraction_over_null": {
                rank: float(null_curve[rank] - aligned_neuron_curve[rank])
                for rank in aligned_neuron_curve
            },
            "fp8_basis_rate_bpw": {
                str(rank): float(8 * rank / C.NUM_EXPERTS)
                for rank in range(1, maximum_rank + 1)
            },
            "int8_per_neuron_coefficient_rate_bpw": {
                str(rank): float(
                    8
                    * rank
                    * triplets[0].intermediate
                    / sum(value.numel() for value in triplets[0].tensors())
                )
                for rank in range(1, maximum_rank + 1)
            },
            "sampling_warning": "basis curve is a deterministic coordinate sketch, not a full-weight factorization",
        },
        "seconds": time.time() - started,
    }


def routed_covariance_report(
    store: OfficialMXFP4Store,
    reader: Any,
    samples: LayerSamples,
    output_metric: RoutedOutputMetric,
    selected_r13: torch.Tensor,
    selected_r2: torch.Tensor,
    *,
    layer: int,
    rows: int,
    schema: str,
    codebook: str,
) -> dict[str, Any]:
    started = time.time()
    row_ids = torch.linspace(0, samples.input_values.shape[0] - 1, rows).round().long()
    route_experts = samples.input_experts.index_select(0, row_ids).long()
    route_gates = samples.input_gates.index_select(0, row_ids).float()
    inputs = samples.input_values.index_select(0, row_ids).float()
    aggregate = samples.routed_latent.index_select(0, row_ids).float()
    errors = torch.zeros(rows, route_experts.shape[1], aggregate.shape[1])
    unique_experts = tuple(sorted(set(route_experts.reshape(-1).tolist())))
    with store.open_layer(layer, experts=unique_experts) as layer_store:
        for expert in unique_experts:
            locations = torch.nonzero(route_experts == expert, as_tuple=False)
            selected_rows = torch.unique(locations[:, 0])
            source = _source_triplet(layer_store, layer, expert)
            candidate = _candidate_triplet(
                reader,
                layer=layer,
                expert=expert,
                r13=int(selected_r13[expert]),
                r2=int(selected_r2[expert]),
                schema=schema,
                codebook=codebook,
            )
            expert_inputs = inputs.index_select(0, selected_rows)
            delta = (
                expert_hidden(expert_inputs, candidate) @ candidate.down.T
                - expert_hidden(expert_inputs, source) @ source.down.T
            )
            row_lookup = {int(row): index for index, row in enumerate(selected_rows)}
            for row, slot in locations.tolist():
                errors[row, slot] = delta[row_lookup[row]]
    covariance = route_error_covariance(aggregate, errors, route_gates, output_metric)
    routed_error = (errors * route_gates[:, :, None]).sum(dim=1)
    exact = output_metric.exact_delta(aggregate, routed_error)
    return {
        "kind": KIND,
        "stage": "routed",
        "layer": layer,
        "rows": rows,
        "unique_experts": len(unique_experts),
        "section_5_4_topk_covariance": covariance,
        "exact_post_projection_sse": float(exact.double().square().sum()),
        "linear_over_exact": float(
            covariance["total_sse"] / exact.double().square().sum().clamp_min(1e-30)
        ),
        "seconds": time.time() - started,
    }


def _stratified_experts(
    metrics: dict[str, torch.Tensor],
    samples: LayerSamples,
    count: int,
) -> tuple[int, ...]:
    r13 = metrics["selected_r13"].long()
    r2 = metrics["selected_r2"].long()
    damage = metrics["official_source_excess_sse"].double()
    support = torch.bincount(samples.input_experts.reshape(-1).long(), minlength=C.NUM_EXPERTS)
    selected: list[int] = []
    present = set(zip(r13.tolist(), r2.tolist()))
    priority = ((0, 0), (2, 2), (1, 2), (2, 1), (1, 1), (0, 2), (2, 0), (0, 1), (1, 0))
    pairs = [pair for pair in priority if pair in present]
    pairs.extend(sorted(present.difference(pairs)))
    for pair in pairs:
        members = torch.nonzero((r13 == pair[0]) & (r2 == pair[1]), as_tuple=False).squeeze(1)
        if members.numel():
            winner = int(members[support.index_select(0, members).argmax()])
            selected.append(winner)
            if len(selected) >= count:
                return tuple(selected)
    order = torch.argsort(damage / support.clamp_min(1), descending=True)
    for expert in order.tolist():
        if expert not in selected and support[expert] > 0:
            selected.append(expert)
        if len(selected) >= count:
            break
    return tuple(selected)


def _manifest(args: argparse.Namespace, pool_manifest: dict[str, Any]) -> dict[str, Any]:
    identities = {}
    for name, root in (("fit_cache", args.fit_cache), ("validation_cache", args.validation_cache)):
        path = root / "manifest.json"
        identities[name] = {"path": str(root.resolve()), "manifest_sha256": _sha256(path)}
    pool_path = args.candidate_pool / "qsrt-candidate-manifest.json"
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "source_model": C.MODEL_ID,
        "source_revision": args.official_revision,
        "candidate_pool": str(args.candidate_pool.resolve()),
        "candidate_manifest_sha256": _sha256(pool_path),
        "candidate_codebook": pool_manifest["codebook"],
        "candidate_schema": pool_manifest["logical_trellis_schema"],
        "inputs": identities,
        "configuration": {
            "layers": list(args.layers),
            "explicit_experts": None if args.experts is None else list(args.experts),
            "experts_per_layer": args.experts_per_layer,
            "stages": list(args.stages),
            "maximum_rows": args.maximum_rows,
            "triplet_samples": args.triplet_samples,
            "pair_samples": args.pair_samples,
            "two_bit_proxy": args.two_bit_proxy,
            "cross_projection_width": args.cross_projection_width,
            "cross_coordinate_samples": args.cross_coordinate_samples,
            "routed_rows": args.routed_rows,
            "seed": args.seed,
        },
        "no_qat": True,
        "writes_checkpoint_payloads": False,
    }


def _result_path(dest: Path, stage: str, layer: int, expert: int | None = None) -> Path:
    suffix = f"-expert-{expert:04d}" if expert is not None else ""
    return dest / "results" / f"layer-{layer:05d}{suffix}.{stage}.json"


def _run_or_resume(path: Path, function: Any) -> dict[str, Any]:
    if path.is_file():
        print(f"resume {path}", flush=True)
        return _read_json(path)
    result = function()
    _atomic_json(path, result)
    print(f"wrote {path} ({result.get('seconds', 0):.1f}s)", flush=True)
    return result


def summarize(dest: Path) -> dict[str, Any]:
    reports = [_read_json(path) for path in sorted((dest / "results").glob("*.json"))]
    functional = [report for report in reports if report.get("stage") == "functional"]
    cross = [report for report in reports if report.get("stage") == "cross"]

    def values(path: tuple[str, ...]) -> list[float]:
        result = []
        for report in functional:
            value: Any = report
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                result.append(float(value))
        return result

    signals = {
        "normalized_gate_rms": values(("section_5_0_activation_geometry", "normalized_gate", "route_weighted_moments", "rms")),
        "normalized_up_rms": values(("section_5_0_activation_geometry", "normalized_up", "route_weighted_moments", "rms")),
        "up_derivative_fraction_ge_099": values(("section_5_0_activation_geometry", "up_factor_derivative", "route_weighted_fraction_at_least", "0.99")),
        "w1_w3_residual_cancellation": values(("section_5_1_pair_geometry", "candidate_residual", "joint_over_separate")),
        "metric_pair_codebook_improvement": values(("section_5_1_pair_geometry", "pair_codebook", "functional_improvement")),
        "post_rms_radial_fraction": values(("section_5_3_post_rms_geometry", "radial_fraction")),
        "expert_gain_improvement": values(("section_5_3_post_rms_geometry", "expert_gain", "improvement")),
        "output_bias_improvement": values(("tiny_corrections", "output_bias", "validation_expert_sse_improvement")),
        "function_correction_improvement": values(("tiny_corrections", "rank4_function_correction", "validation_expert_sse_improvement")),
        "w2_refit_improvement": values(("tiny_corrections", "w2_ridge_refit", "validation_expert_sse_improvement")),
        "input_diagonal_2bit_improvement": values(("exact_transform_2bit_proxy", "common_input_diagonal_gauge", "post_projection_sse_improvement_vs_baseline")),
        "postactivation_down_2bit_improvement": values(("exact_transform_2bit_proxy", "exact_postactivation_equilibration", "down_only_expert_sse_improvement")),
        "temperature_2bit_improvement": values(("exact_transform_2bit_proxy", "temperature_equilibration", "post_projection_sse_improvement_vs_baseline")),
        "approximate_w3_w2_scale_gauge_2bit_improvement": values(("exact_transform_2bit_proxy", "approximate_w3_w2_scale_gauge", "best", "post_projection_sse_improvement_vs_baseline")),
        "approximate_w3_w2_scale_gauge_full_precision_relative_sse": values(("exact_transform_2bit_proxy", "approximate_w3_w2_scale_gauge", "best", "full_precision_expert_relative_sse")),
        "approximate_w3_w2_scale_gauge_fit_selected_confirmation_improvement": values(("exact_transform_2bit_proxy", "approximate_w3_w2_scale_gauge", "fit_selected", "applied_confirmation_post_projection_sse_improvement_vs_baseline")),
        "expert_local_hadamard_2bit_improvement": values(("exact_transform_2bit_proxy", "expert_local_two_sided_block_hadamard", "fit_selected", "applied_confirmation_post_projection_sse_improvement_vs_baseline")),
        "expert_local_hadamard_2bit_oracle_improvement": values(("exact_transform_2bit_proxy", "expert_local_two_sided_block_hadamard", "best", "post_projection_sse_improvement_vs_baseline")),
    }
    medians = {
        key: float(torch.tensor(value).median()) if value else None
        for key, value in signals.items()
    }
    promotions = []
    for name in (
        "metric_pair_codebook_improvement",
        "expert_gain_improvement",
        "output_bias_improvement",
        "function_correction_improvement",
        "input_diagonal_2bit_improvement",
        "postactivation_down_2bit_improvement",
        "temperature_2bit_improvement",
        "approximate_w3_w2_scale_gauge_2bit_improvement",
        "expert_local_hadamard_2bit_improvement",
    ):
        median = medians[name]
        if median is not None and median >= 0.05:
            promotions.append({"idea": name, "median_improvement": median, "gate": "runtime_invasive_5pct"})
        elif median is not None and median >= 0.02:
            promotions.append({"idea": name, "median_improvement": median, "gate": "cheap_exact_2pct"})
    return {
        "kind": KIND,
        "stage": "summary",
        "reports": len(reports),
        "functional_experts": len(functional),
        "cross_layers": len(cross),
        "signal_medians": medians,
        "promotions": promotions,
        "promotion_policy": {
            "cheap_exact_transform_relative_improvement": 0.02,
            "runtime_invasive_relative_improvement": 0.05,
            "minimum_net_rate_gain_bpw": 0.01,
            "effective_rate_targets_bpw": [1.90, 1.934, 1.95, 1.98, 2.0045, 2.05],
        },
        "interpretation": "screening results only; promoted codec ideas still require fresh K2 SQG-T12 and MCG confirmation",
        "oracle_only": {
            "w2_refit_improvement": medians["w2_refit_improvement"],
            "reason": "the fitted dense down matrix costs more than the target rate; use only as a ceiling for a structured correction",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--fit-cache", type=Path, default=DEFAULT_FIT_CACHE)
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_VALIDATION_CACHE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--official-revision", default=C.REVISION)
    parser.add_argument("--layers", type=_parse_ints, default=DEFAULT_LAYERS)
    parser.add_argument("--experts", type=_parse_ints)
    parser.add_argument("--experts-per-layer", type=int, default=4)
    parser.add_argument("--stages", type=_parse_stages, default=("native", "functional", "cross", "summary"))
    parser.add_argument("--maximum-rows", type=int, default=64)
    parser.add_argument("--triplet-samples", type=int, default=8)
    parser.add_argument("--pair-samples", type=int, default=32768)
    parser.add_argument(
        "--two-bit-proxy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the generic block-32 2-bit exact-transform proxy",
    )
    parser.add_argument("--cross-projection-width", type=int, default=8)
    parser.add_argument("--cross-coordinate-samples", type=int, default=16)
    parser.add_argument("--routed-rows", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 32))
    args = parser.parse_args()
    if any(layer not in C.MOE_LAYERS for layer in args.layers):
        parser.error(f"layers must be Kimi MoE decoder layers {C.MOE_LAYERS.start}..{C.MOE_LAYERS.stop - 1}")
    if args.experts is not None and any(not 0 <= expert < C.NUM_EXPERTS for expert in args.experts):
        parser.error(f"experts must be in 0..{C.NUM_EXPERTS - 1}")
    for name in ("experts_per_layer", "maximum_rows", "triplet_samples", "pair_samples", "cross_projection_width", "cross_coordinate_samples", "routed_rows", "threads"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    pool_manifest = _read_json(args.candidate_pool / "qsrt-candidate-manifest.json")
    if pool_manifest.get("source_model") != C.MODEL_ID or pool_manifest.get("source_revision") != args.official_revision:
        raise ValueError("candidate pool does not match the pinned official source")
    manifest = _manifest(args, pool_manifest)
    manifest_path = args.dest / "manifest.json"
    if manifest_path.is_file():
        if _read_json(manifest_path) != manifest:
            raise ValueError(f"existing study manifest differs; use a fresh destination: {args.dest}")
    else:
        _atomic_json(manifest_path, manifest)
    store = OfficialMXFP4Store(revision=args.official_revision)
    fit_index = index_cached_layer_samples(args.fit_cache, (layer - 1 for layer in args.layers))
    validation_index = index_cached_layer_samples(
        args.validation_cache, (layer - 1 for layer in args.layers)
    )
    schema = str(pool_manifest["logical_trellis_schema"])
    codebook = str(pool_manifest["codebook"])

    for layer in args.layers:
        fit_samples = fit_index.pop(layer - 1)
        validation_samples = validation_index.pop(layer - 1)
        payload_path, metrics_path = _candidate_paths(args.candidate_pool, layer)
        metrics = load_file(str(metrics_path), device="cpu")
        experts = (
            args.experts
            if args.experts is not None
            else _stratified_experts(metrics, fit_samples, args.experts_per_layer)
        )
        print(f"layer {layer}: experts {experts}", flush=True)
        if "native" in args.stages:
            for expert in experts:
                path = _result_path(args.dest, "native", layer, expert)
                _run_or_resume(path, lambda expert=expert: native_source_report(store, layer, expert))
        output_metric = None
        if any(stage in args.stages for stage in ("functional", "routed")):
            output_metric = _load_output_metric(store, layer)
        with safe_open(str(payload_path), framework="pt", device="cpu") as reader:
            if "functional" in args.stages:
                assert output_metric is not None
                for expert in experts:
                    path = _result_path(args.dest, "functional", layer, expert)
                    _run_or_resume(
                        path,
                        lambda expert=expert: functional_report(
                            store,
                            reader,
                            validation_samples,
                            output_metric,
                            layer=layer,
                            expert=expert,
                            r13=int(metrics["selected_r13"][expert]),
                            r2=int(metrics["selected_r2"][expert]),
                            block_contexts=metrics["block_contexts"][expert],
                            schema=schema,
                            codebook=codebook,
                            maximum_rows=args.maximum_rows,
                            triplet_samples=args.triplet_samples,
                            pair_samples=args.pair_samples,
                            two_bit_proxy=args.two_bit_proxy,
                            seed=args.seed,
                        ),
                    )
            if "routed" in args.stages:
                assert output_metric is not None
                path = _result_path(args.dest, "routed", layer)
                _run_or_resume(
                    path,
                    lambda: routed_covariance_report(
                        store,
                        reader,
                        validation_samples,
                        output_metric,
                        metrics["selected_r13"],
                        metrics["selected_r2"],
                        layer=layer,
                        rows=args.routed_rows,
                        schema=schema,
                        codebook=codebook,
                    ),
                )
        if "cross" in args.stages and layer in CROSS_LAYERS:
            cross_experts = experts if len(experts) >= 2 else _stratified_experts(metrics, fit_samples, 4)
            path = _result_path(args.dest, "cross", layer)
            _run_or_resume(
                path,
                lambda: cross_expert_report(
                    store,
                    layer=layer,
                    experts=tuple(cross_experts),
                    projection_width=args.cross_projection_width,
                    coordinate_samples=args.cross_coordinate_samples,
                    seed=args.seed,
                ),
            )
    if "summary" in args.stages:
        report = summarize(args.dest)
        _atomic_json(args.dest / "summary.json", report)
        print(json.dumps(_json_safe(report), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

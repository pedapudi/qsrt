#!/usr/bin/env python3
"""Confirm promoted coupled transforms with fresh uniform SQG/MCG encodes.

This is a bounded research comparison at uniform K2 by default.  It compares
the official decoded Kimi expert against independently encoded MCG and the
production SQG-T12 scalar law, with and without an explicit coupled two-sided
block-Hadamard reparameterization.  It writes no checkpoint payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

from qsrt import constants as C
from qsrt.capture import index_cached_layer_samples
from qsrt.coupled_expert_study import (
    CoupledTriplet,
    RoutedOutputMetric,
    apply_w3_w2_scale_gauge,
    apply_w3_w2_sign_draw,
    encode_coupled_block_hadamard,
    execute_coupled_block_hadamard,
    expert_hidden,
)
from qsrt.exl3_loader import load_qsrt_encoder
from qsrt.exl3_reference import CODEBOOK_SQG_XOR_CHEB_T12
from qsrt.io.stream import load_tensor
from qsrt.qsrt import matrix_rate_axis
from qsrt.qsrt_codec_pilot import CODEBOOK_MCG, encode_uniform_candidate
from qsrt.source_weights import OfficialMXFP4Store
from qsrt.sqg_quantizer import install_sqg_quantizer


KIND = "qsrt_k3_coupled_uniform_codec_confirmation"
SCHEMA_VERSION = 2
DEFAULT_CACHE = Path(
    "/data/datasets/kquant/captures/k3-codec-diverse-validation-v3-128k-input-v1.kqsamples"
)


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("expected a nonempty list of unique integers")
    return result


def _parse_names(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("expected a nonempty list of unique names")
    return result


def _parse_expert_draws(value: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        expert_text, separator, draw_text = item.partition(":")
        if not separator or not expert_text.isdecimal() or not draw_text.isdecimal():
            raise argparse.ArgumentTypeError(
                "expected comma-separated expert:draw pairs"
            )
        expert = int(expert_text)
        draw = int(draw_text)
        if expert in result:
            raise argparse.ArgumentTypeError("expert draw keys must be unique")
        result[expert] = draw
    return result


def _parse_expert_scale_gauges(value: str) -> dict[int, tuple[str, float]]:
    result: dict[int, tuple[str, float]] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        fields = item.split(":")
        if len(fields) != 3 or not fields[0].isdecimal():
            raise argparse.ArgumentTypeError(
                "expected comma-separated expert:policy:strength triples"
            )
        expert = int(fields[0])
        try:
            strength = float(fields[2])
        except ValueError as error:
            raise argparse.ArgumentTypeError("gauge strength must be numeric") from error
        if expert in result:
            raise argparse.ArgumentTypeError("expert gauge keys must be unique")
        result[expert] = (fields[1], strength)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _output_metric(store: OfficialMXFP4Store, layer: int) -> RoutedOutputMetric:
    prefix = f"{C.LM_PREFIX}layers.{layer}.block_sparse_moe"
    return RoutedOutputMetric(
        load_tensor(store.cache, f"{prefix}.routed_expert_norm.weight").float(),
        load_tensor(store.cache, C.latent_up_proj_tensor(layer)).float(),
    )


def _rows(samples: Any, expert: int, maximum: int) -> dict[str, torch.Tensor]:
    locations = torch.nonzero(samples.input_experts == expert, as_tuple=False)
    if locations.numel() == 0:
        raise ValueError(f"expert {expert} has no validation rows")
    if locations.shape[0] > maximum:
        location_splits = samples.input_split.index_select(0, locations[:, 0])

        def evenly(group: torch.Tensor, count: int) -> torch.Tensor:
            if count <= 0:
                return group[:0]
            if group.shape[0] <= count:
                return group
            indices = torch.linspace(0, group.shape[0] - 1, count).round().long()
            return group.index_select(0, indices)

        fit = locations[location_splits == 0]
        confirmation = locations[location_splits == 1]
        fit_count = min(fit.shape[0], maximum // 2)
        confirmation_count = min(confirmation.shape[0], maximum - fit_count)
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
        locations = locations.index_select(0, torch.argsort(locations[:, 0]))
    rows, slots = locations[:, 0], locations[:, 1]
    return {
        "inputs": samples.input_values.index_select(0, rows).float(),
        "gates": samples.input_gates[rows, slots].float(),
        "aggregate": samples.routed_latent.index_select(0, rows).float(),
        "split": samples.input_split.index_select(0, rows),
        "documents": torch.bitwise_right_shift(
            samples.input_observations.index_select(0, rows), 32
        ),
    }


def _external_transform(
    source: CoupledTriplet,
    *,
    residual_draw: int,
    intermediate_draw: int,
    sign_draw: int,
    scale_gauge: tuple[str, float],
    block_size: int,
    preactivation_block_size: int,
    postactivation_block_size: int,
) -> CoupledTriplet:
    gauged = apply_w3_w2_scale_gauge(
        apply_w3_w2_sign_draw(source, draw=sign_draw),
        policy=scale_gauge[0],
        strength=scale_gauge[1],
    )
    return encode_coupled_block_hadamard(
        gauged,
        block_size=block_size,
        preactivation_block_size=preactivation_block_size,
        postactivation_block_size=postactivation_block_size,
        residual_rotation_draw=residual_draw,
        intermediate_rotation_draw=intermediate_draw,
    )


def _execute_arm(
    inputs: torch.Tensor,
    reconstruction: CoupledTriplet,
    arm: str,
    *,
    residual_draw: int,
    intermediate_draw: int,
    block_size: int,
    preactivation_block_size: int,
    postactivation_block_size: int,
) -> torch.Tensor:
    if arm == "baseline":
        return expert_hidden(inputs, reconstruction) @ reconstruction.down.T
    if arm != "coupled_hadamard":
        raise ValueError(f"unknown confirmation arm {arm!r}")
    return execute_coupled_block_hadamard(
        inputs,
        reconstruction,
        block_size=block_size,
        preactivation_block_size=preactivation_block_size,
        postactivation_block_size=postactivation_block_size,
        residual_rotation_draw=residual_draw,
        intermediate_rotation_draw=intermediate_draw,
    )


def _encode_triplet(
    source: CoupledTriplet,
    *,
    layer: int,
    expert: int,
    arm: str,
    bits: int,
    codebook: str,
    device: torch.device,
    quantizer_module: Any,
    ldlq_tf32: bool,
    tailbite_context: int,
) -> tuple[CoupledTriplet, list[dict[str, Any]]]:
    reconstructions = []
    evidence = []
    for matrix, weight in zip(C.EXPERT_MATRICES, source.tensors(), strict=True):
        seed = layer * 1_000_000 + C.EXPERT_MATRICES.index(matrix)
        result = encode_uniform_candidate(
            weight,
            bits=bits,
            codebook=codebook,
            device=device,
            quantizer_module=quantizer_module,
            input_sign_seed=seed,
            output_sign_seed=seed + 499_979,
            rate_axis=matrix_rate_axis(matrix),
            scale_scope_key=(KIND, layer, expert, arm, codebook, matrix, bits),
            g_scale_into_sv=matrix in ("w1", "w3"),
            ldlq_tf32=ldlq_tf32,
            tailbite_context=tailbite_context,
        )
        reconstructions.append(result["reconstruction"].float())
        evidence.append(result["payload"])
        torch.cuda.empty_cache()
    return CoupledTriplet(*reconstructions), evidence


def _score(
    source: CoupledTriplet,
    encoded_source: CoupledTriplet,
    reconstruction: CoupledTriplet,
    rows: dict[str, torch.Tensor],
    output_metric: RoutedOutputMetric,
    arm: str,
    evidence: list[dict[str, Any]],
    residual_draw: int,
    intermediate_draw: int,
    block_size: int,
    preactivation_block_size: int,
    postactivation_block_size: int,
) -> dict[str, Any]:
    source_output = expert_hidden(rows["inputs"], source) @ source.down.T
    encoded_source_output = _execute_arm(
        rows["inputs"],
        encoded_source,
        arm,
        residual_draw=residual_draw,
        intermediate_draw=intermediate_draw,
        block_size=block_size,
        preactivation_block_size=preactivation_block_size,
        postactivation_block_size=postactivation_block_size,
    )
    output = _execute_arm(
        rows["inputs"],
        reconstruction,
        arm,
        residual_draw=residual_draw,
        intermediate_draw=intermediate_draw,
        block_size=block_size,
        preactivation_block_size=preactivation_block_size,
        postactivation_block_size=postactivation_block_size,
    )
    error = output - source_output

    def score_rows(mask: torch.Tensor) -> dict[str, Any]:
        selected_error = error[mask]
        selected_source = source_output[mask]
        routed_error = rows["gates"][mask, None] * selected_error
        exact = output_metric.exact_delta(rows["aggregate"][mask], routed_error)
        return {
            "expert_output_nmse": float(
                selected_error.double().square().sum()
                / selected_source.double().square().sum().clamp_min(1e-30)
            ),
            "post_projection_sse": float(exact.double().square().sum()),
            "rows": int(mask.sum()),
            "documents": int(torch.unique(rows["documents"][mask]).numel()),
        }

    all_rows = torch.ones(error.shape[0], dtype=torch.bool)
    fit_rows = rows["split"] == 0
    confirmation_rows = rows["split"] == 1
    if not bool(fit_rows.any()) or not bool(confirmation_rows.any()):
        fit_rows = torch.arange(error.shape[0]) % 2 == 0
        confirmation_rows = ~fit_rows
    routed_scores = score_rows(all_rows)
    weight_sse = sum(
        float((candidate.double() - target.double()).square().sum())
        for candidate, target in zip(
            reconstruction.tensors(), encoded_source.tensors(), strict=True
        )
    )
    weight_energy = sum(
        float(target.double().square().sum()) for target in encoded_source.tensors()
    )
    payload_bits = sum(int(item["trellis_bytes"]) * 8 for item in evidence)
    scale_bits = sum(int(item["scale_bytes"]) * 8 for item in evidence)
    return {
        "weight_nmse": weight_sse / weight_energy,
        "encoded_source_relative_sse": float(
            (encoded_source_output - source_output).double().square().sum()
            / source_output.double().square().sum().clamp_min(1e-30)
        ),
        **routed_scores,
        "fit": score_rows(fit_rows),
        "confirmation": score_rows(confirmation_rows),
        "trellis_bpw": payload_bits / source.numel,
        "scale_bpw": scale_bits / source.numel,
        "all_in_bpw": (payload_bits + scale_bits) / source.numel,
        "matrices": evidence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--experts", type=_parse_ints, required=True)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument(
        "--codebooks",
        type=_parse_names,
        default=(CODEBOOK_MCG, CODEBOOK_SQG_XOR_CHEB_T12),
    )
    parser.add_argument(
        "--arms", type=_parse_names, default=("baseline", "coupled_hadamard")
    )
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--maximum-rows", type=int, default=64)
    parser.add_argument(
        "--coupled-residual-draw",
        type=int,
        default=0,
        help="one residual-side signed-Hadamard draw shared by the layer",
    )
    parser.add_argument(
        "--coupled-hadamard-block-size",
        type=int,
        choices=(64, 128, 256, 512),
        default=512,
    )
    parser.add_argument(
        "--coupled-hadamard-preactivation-block-size",
        type=int,
        choices=(64, 128, 256, 512, 1024, 2048),
        default=512,
    )
    parser.add_argument(
        "--coupled-hadamard-postactivation-block-size",
        type=int,
        choices=(64, 128, 256, 512, 1024),
        default=512,
    )
    parser.add_argument(
        "--coupled-intermediate-draws",
        type=_parse_expert_draws,
        default={},
        help="expert:draw overrides for coupled intermediate rotations",
    )
    parser.add_argument(
        "--w3-w2-sign-draws",
        type=_parse_expert_draws,
        default={},
        help="expert:draw overrides for exact baked W3/W2 sign gauges",
    )
    parser.add_argument(
        "--w3-w2-scale-gauges",
        type=_parse_expert_scale_gauges,
        default={},
        help="expert:policy:strength overrides for bounded positive W3/W2 gauges",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exllamav3-root", type=Path, default=Path("/home/luke/projects/exllamav3"))
    parser.add_argument("--official-revision", default=C.REVISION)
    parser.add_argument("--ldlq-tf32", action="store_true")
    parser.add_argument(
        "--tailbite-context",
        type=int,
        default=128,
        help="cyclic Viterbi screening context (1..128)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.layer not in C.MOE_LAYERS:
        parser.error("--layer must be a Kimi MoE layer 1..92")
    if any(not 0 <= expert < C.NUM_EXPERTS for expert in args.experts):
        parser.error(f"--experts must be in 0..{C.NUM_EXPERTS - 1}")
    if args.bits not in range(2, 7):
        parser.error("--bits must be K2..K6")
    if any(item not in (CODEBOOK_MCG, CODEBOOK_SQG_XOR_CHEB_T12) for item in args.codebooks):
        parser.error("--codebooks supports only mcg and sqg_xor_cheb_t12")
    if any(item not in ("baseline", "coupled_hadamard") for item in args.arms):
        parser.error("--arms supports baseline and coupled_hadamard")
    if args.maximum_rows <= 0:
        parser.error("--maximum-rows must be positive")
    if not 1 <= args.tailbite_context <= 128:
        parser.error("--tailbite-context must lie in 1..128")
    if args.coupled_residual_draw < 0:
        parser.error("--coupled-residual-draw must be nonnegative")
    if any(
        expert not in args.experts or not 0 <= draw
        for expert, draw in args.coupled_intermediate_draws.items()
    ):
        parser.error(
            "coupled intermediate draws must be nonnegative and name selected experts"
        )
    if any(
        expert not in args.experts or not 0 <= draw
        for expert, draw in args.w3_w2_sign_draws.items()
    ):
        parser.error("W3/W2 sign draws must be nonnegative and name selected experts")
    allowed_scale_policies = {"identity", "up_down_rms", "down_rms", "down_absmax"}
    if any(
        expert not in args.experts
        or policy not in allowed_scale_policies
        or not math.isfinite(strength)
        or (policy == "identity" and strength != 0.0)
        or (policy != "identity" and not 0.0 < strength <= 1.0)
        for expert, (policy, strength) in args.w3_w2_scale_gauges.items()
    ):
        parser.error("W3/W2 scale gauges contain an invalid expert, policy, or strength")
    return args


def main() -> None:
    args = parse_args()
    signature = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "source_model": C.MODEL_ID,
        "source_revision": args.official_revision,
        "layer": args.layer,
        "experts": list(args.experts),
        "bits": args.bits,
        "codebooks": list(args.codebooks),
        "arms": list(args.arms),
        "validation_cache": str(args.validation_cache.resolve()),
        "validation_manifest_sha256": _sha256(args.validation_cache / "manifest.json"),
        "maximum_rows": args.maximum_rows,
        "coupled_rotations": {
            "residual_block_size": args.coupled_hadamard_block_size,
            "preactivation_block_size": (
                args.coupled_hadamard_preactivation_block_size
            ),
            "postactivation_block_size": (
                args.coupled_hadamard_postactivation_block_size
            ),
            "residual_layer_shared": args.coupled_residual_draw,
            "intermediate_by_expert": {
                str(expert): draw
                for expert, draw in sorted(
                    args.coupled_intermediate_draws.items()
                )
            },
        },
        "w3_w2_sign_gauge_by_expert": {
            str(expert): draw
            for expert, draw in sorted(args.w3_w2_sign_draws.items())
        },
        "w3_w2_scale_gauge_by_expert": {
            str(expert): {"policy": policy, "strength": strength}
            for expert, (policy, strength) in sorted(args.w3_w2_scale_gauges.items())
        },
        "ldlq_tf32": args.ldlq_tf32,
        "tailbite_context": args.tailbite_context,
        "no_qat": True,
        "writes_checkpoint_payloads": False,
    }
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        payload = json.loads(args.output.read_text())
        if payload.get("signature") != signature:
            raise ValueError("resume output signature mismatch")
    else:
        payload = {"signature": signature, "results": {}, "complete": False}
        _atomic_json(args.output, payload)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("codec confirmation requires an available CUDA device")
    store = OfficialMXFP4Store(revision=args.official_revision)
    samples = index_cached_layer_samples(args.validation_cache, [args.layer - 1]).pop(
        args.layer - 1
    )
    output_metric = _output_metric(store, args.layer)
    quantizer_module = load_qsrt_encoder(args.exllamav3_root)
    install_sqg_quantizer(quantizer_module)
    for expert in args.experts:
        key = str(expert)
        if key in payload["results"]:
            continue
        started = time.time()
        selected_rows = _rows(samples, expert, args.maximum_rows)
        with store.open_layer(args.layer, experts=(expert,)) as layer_store:
            source = CoupledTriplet(
                layer_store.load_matrix(args.layer, expert, "w1"),
                layer_store.load_matrix(args.layer, expert, "w3"),
                layer_store.load_matrix(args.layer, expert, "w2"),
            )
        intermediate_draw = args.coupled_intermediate_draws.get(expert, 0)
        sign_draw = args.w3_w2_sign_draws.get(expert, 0)
        scale_gauge = args.w3_w2_scale_gauges.get(expert, ("identity", 0.0))
        arm_sources = {
            "baseline": source,
            "coupled_hadamard": _external_transform(
                source,
                residual_draw=args.coupled_residual_draw,
                intermediate_draw=intermediate_draw,
                sign_draw=sign_draw,
                scale_gauge=scale_gauge,
                block_size=args.coupled_hadamard_block_size,
                preactivation_block_size=(
                    args.coupled_hadamard_preactivation_block_size
                ),
                postactivation_block_size=(
                    args.coupled_hadamard_postactivation_block_size
                ),
            ),
        }
        results: dict[str, Any] = {}
        for codebook in args.codebooks:
            results[codebook] = {}
            for arm in args.arms:
                reconstruction, evidence = _encode_triplet(
                    arm_sources[arm],
                    layer=args.layer,
                    expert=expert,
                    arm=arm,
                    bits=args.bits,
                    codebook=codebook,
                    device=device,
                    quantizer_module=quantizer_module,
                    ldlq_tf32=args.ldlq_tf32,
                    tailbite_context=args.tailbite_context,
                )
                results[codebook][arm] = _score(
                    source,
                    arm_sources[arm],
                    reconstruction,
                    selected_rows,
                    output_metric,
                    arm,
                    evidence,
                    args.coupled_residual_draw,
                    intermediate_draw,
                    args.coupled_hadamard_block_size,
                    args.coupled_hadamard_preactivation_block_size,
                    args.coupled_hadamard_postactivation_block_size,
                )
                print(
                    f"layer {args.layer} expert {expert} {codebook} {arm}: "
                    f"post SSE {results[codebook][arm]['post_projection_sse']:.6g}",
                    flush=True,
                )
                del reconstruction
                torch.cuda.empty_cache()
        comparisons = {}
        for codebook in args.codebooks:
            baseline = results[codebook].get("baseline")
            transformed = results[codebook].get("coupled_hadamard")
            if baseline and transformed:
                comparisons[f"{codebook}:coupled_hadamard_vs_baseline"] = {
                    metric: 1.0 - transformed[metric] / baseline[metric]
                    for metric in ("weight_nmse", "expert_output_nmse", "post_projection_sse")
                }
        if CODEBOOK_MCG in results and CODEBOOK_SQG_XOR_CHEB_T12 in results:
            for arm in args.arms:
                comparisons[f"sqg_vs_mcg:{arm}"] = {
                    metric: 1.0
                    - results[CODEBOOK_SQG_XOR_CHEB_T12][arm][metric]
                    / results[CODEBOOK_MCG][arm][metric]
                    for metric in ("weight_nmse", "expert_output_nmse", "post_projection_sse")
                }
        payload["results"][key] = {
            "expert": expert,
            "coupled_rotations": {
                "residual_draw": args.coupled_residual_draw,
                "intermediate_draw": intermediate_draw,
            },
            "w3_w2_sign_gauge_draw": sign_draw,
            "w3_w2_scale_gauge": {
                "policy": scale_gauge[0],
                "strength": scale_gauge[1],
            },
            "candidates": results,
            "comparisons": comparisons,
            "seconds": time.time() - started,
        }
        _atomic_json(args.output, payload)
    payload["complete"] = True
    _atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output.resolve()), "complete": True}, indent=2))


if __name__ == "__main__":
    main()

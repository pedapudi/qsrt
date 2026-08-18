#!/usr/bin/env python3
"""Encode one layer's canonical expert targets into uniform-K2 overlays.

The supported objectives are independent tile distortion, two-sided Kronecker
curvature, and anchor-relative final-logit gradient refinement. Every mode
preserves the serving format, reconstruction law, scales, and rate.

Independent tile encoding can retain all three expert matrices in one source
bank and emit W1, W3, and W2 together. Curvature-guided W2 encoding remains a
dependent step because its input Hessian is constructed from decoded W1/W3.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from qsrt import constants as C
from qsrt.capture import load_layer_hessians
from qsrt.exl3_loader import load_qsrt_encoder
from qsrt.instanttensor_kimi import InstantTensorKimiLayerLoader
from qsrt.kimi_upstream_factors import KimiUpstreamFactorArchive
from qsrt.pack.qsrt_atoms import candidate_layer_path
from qsrt.pack.qsrt_candidates import candidate_tensor_name
from qsrt.pack.qsrt_encoder import (
    QSRTMatrixCandidate,
    finalize_qsrt_matrix_candidate,
    plan_qsrt_matrix,
    qsrt_transform_seed_draw,
)
from qsrt.qsrt import K2, SCHEMA as QSRT_SCHEMA, matrix_rate_axis
from qsrt.qsrt import (
    PackedQSRTTrellis,
    QSRTTrellisDescriptor,
    unpack_qsrt_trellis_states,
)
from qsrt.qsrt_atoms_v2 import unpack_atoms_v2_format_section
from qsrt.qsrt_coupled import (
    CoupledHadamardExecution,
    CoupledHadamardSpec,
    encode_coupled_upstream_weights,
    encode_coupled_weights,
)
from qsrt.source_weights import OfficialMXFP4Store
from qsrt.sqg_quantizer import install_sqg_quantizer
from qsrt.two_sided_qsrt import (
    UniformSQGSharedInputConditioning,
    encode_uniform_sqg_direct_batch,
    encode_uniform_sqg_two_sided_output_blocks_batch,
    prepare_uniform_sqg_anchor_input_hessian,
    refine_uniform_sqg_anchor_gradient,
)


DEFAULT_PROFILE = Path(
    "/data/releases/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-v1"
)
DEFAULT_CANDIDATE_POOL = Path(
    "/data/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-CANDIDATES-v1"
)
DEFAULT_HESSIANS = Path(
    "/data/datasets/kquant/hessians/"
    "k3-denseh-broad-v7-4m-train-h13-identity-qsrt-v1.kqhess"
)
DEFAULT_FACTORS = Path(
    "/data/datasets/kquant/hessians/"
    "k3-official-mxfp4-final-logit-fisher-100k-v1-upstream-factors"
)


def _parse_experts(raw: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in raw.split(",") if value)
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("experts must be unique comma-separated IDs")
    if any(not 0 <= expert < C.NUM_EXPERTS for expert in values):
        raise argparse.ArgumentTypeError("expert lies outside Kimi-K3 geometry")
    return values


def _profile_draws(profile: Path, layer: int) -> tuple[int, ...]:
    path = profile / f"qsrt-layer-{layer:05d}.safetensors"
    with safe_open(path, framework="pt", device="cpu") as reader:
        metadata = reader.metadata()
        if metadata is None or "profile" not in metadata:
            raise ValueError("served profile layer lacks its format identity")
        formats, draws = unpack_atoms_v2_format_section(
            str(metadata["profile"]),
            reader.get_tensor("_qsrt_format_section"),
        )
    if draws is None or len(draws) != C.NUM_EXPERTS:
        raise ValueError("served profile layer lacks complete rotation draws")
    if any(value != "K2" for value in formats):
        raise ValueError("served profile layer is not uniformly K2")
    return tuple(int(value) for value in draws)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experts", type=_parse_experts)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--candidate-pool", type=Path, default=DEFAULT_CANDIDATE_POOL
    )
    parser.add_argument("--hessians", type=Path, default=DEFAULT_HESSIANS)
    parser.add_argument("--factor-archive", type=Path, default=DEFAULT_FACTORS)
    parser.add_argument(
        "--exllamav3-root", type=Path, default=Path("/home/luke/projects/exllamav3")
    )
    parser.add_argument("--expert-batch-size", type=int, default=16)
    parser.add_argument("--output-damping-ratio", type=float, default=3.0)
    parser.add_argument("--tailbite-context", type=int, default=128)
    parser.add_argument("--verify-factor-hash", action="store_true")
    parser.add_argument(
        "--direct-viterbi",
        action="store_true",
        help="encode independent SQG tiles without dense-H or Fisher feedback",
    )
    parser.add_argument(
        "--include-w2",
        action="store_true",
        help="retain and encode W1, W3, and W2 in one direct-Viterbi layer pass",
    )
    parser.add_argument(
        "--gradient-strength",
        type=float,
        default=0.0,
        help="Scale applied to the deterministic final-logit KL gradient.",
    )
    parser.add_argument(
        "--gradient-strength-normalization",
        choices=("none", "h13_mean_diagonal"),
        default="none",
        help=(
            "Interpret the gradient strength directly or multiply it by the "
            "mean diagonal of the layer H13 factor."
        ),
    )
    parser.add_argument(
        "--gradient-core-rcond",
        type=float,
        default=1.0e-3,
        help="Relative singular-value cutoff for the gradient sketch core.",
    )
    parser.add_argument(
        "--gradient-anchor-layer",
        type=Path,
        help=(
            "Optional layer safetensors containing the anchor W1/W3 payloads. "
            "The sealed uniform-K2 candidate layer is used when omitted."
        ),
    )
    parser.add_argument(
        "--gradient-anchor-id",
        help="Durable identity of the checkpoint at which the gradient was captured.",
    )
    parser.add_argument(
        "--gradient-objective-id",
        help="Durable identity of the deterministic final-logit objective.",
    )
    parser.add_argument("--gradient-refinement-sweeps", type=int, default=1)
    return parser.parse_args()


def _load_anchor_matrix(
    reader,
    *,
    layer: int,
    expert: int,
    matrix: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    names = {
        part: candidate_tensor_name(layer, expert, matrix, part)
        for part in ("trellis", "suh", "svh")
    }
    payload = reader.get_tensor(names["trellis"])
    suh = reader.get_tensor(names["suh"])
    svh = reader.get_tensor(names["svh"])
    descriptor = QSRTTrellisDescriptor(
        mode_id=K2.mode_id,
        rate_axis=matrix_rate_axis(matrix),
        k_tiles=suh.numel() // 16,
        n_tiles=svh.numel() // 16,
        schema=QSRT_SCHEMA,
    )
    if payload.numel() * payload.element_size() != descriptor.payload_bytes:
        raise ValueError("gradient anchor payload has the wrong byte count")
    packed = PackedQSRTTrellis(
        descriptor,
        payload.to(device=device, dtype=torch.int16).contiguous(),
    )
    return (
        unpack_qsrt_trellis_states(packed),
        suh.to(device=device, dtype=torch.float16).contiguous(),
        svh.to(device=device, dtype=torch.float16).contiguous(),
    )


def main() -> None:
    args = _parse_args()
    if args.layer not in C.MOE_LAYERS:
        raise ValueError("layer must lie in Kimi-K3's routed-MoE layer set")
    if args.output.exists() or (args.result is not None and args.result.exists()):
        raise FileExistsError("output path already exists")
    if args.expert_batch_size < 1:
        raise ValueError("expert batch size must be positive")
    if args.gradient_refinement_sweeps < 1:
        raise ValueError("gradient refinement sweep count must be positive")
    if (
        not torch.isfinite(torch.tensor(args.gradient_strength))
        or args.gradient_strength < 0
    ):
        raise ValueError("gradient strength must be finite and nonnegative")
    if not 0.0 < args.gradient_core_rcond <= 1.0:
        raise ValueError("gradient core rcond must lie in (0, 1]")
    gradient_enabled = args.gradient_strength > 0.0
    if args.direct_viterbi and gradient_enabled:
        raise ValueError("direct Viterbi and gradient refinement are exclusive")
    if args.include_w2 and not args.direct_viterbi:
        raise ValueError(
            "combined W1/W3/W2 encoding currently requires direct Viterbi"
        )
    if gradient_enabled and (
        not args.gradient_anchor_id or not args.gradient_objective_id
    ):
        raise ValueError(
            "gradient guidance requires explicit anchor and objective identities"
        )
    if not gradient_enabled and any(
        value is not None
        for value in (
            args.gradient_anchor_layer,
            args.gradient_anchor_id,
            args.gradient_objective_id,
        )
    ):
        raise ValueError("gradient anchor arguments require positive gradient strength")
    experts = (
        tuple(range(C.NUM_EXPERTS)) if args.experts is None else args.experts
    )
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("upstream Fisher encoding requires an indexed CUDA device")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    timings: dict[str, float] = {}

    stage = time.monotonic()
    profile_draws = _profile_draws(args.profile.resolve(), args.layer)
    archive = (
        None
        if args.direct_viterbi
        else KimiUpstreamFactorArchive(args.factor_archive)
    )
    gradient_layer = None
    factor_blocks = None
    if args.direct_viterbi:
        factor_rows_cpu = torch.ones(C.NUM_EXPERTS, dtype=torch.long)
        factor_strength_cpu = torch.ones((C.NUM_EXPERTS, 2))
        factor_draws_cpu = torch.tensor(profile_draws, dtype=torch.long)
    elif gradient_enabled:
        assert archive is not None
        provenance = archive.manifest.get("provenance", {})
        if not isinstance(provenance, dict) or (
            provenance.get("gradient_anchor_id") != args.gradient_anchor_id
            or provenance.get("gradient_objective_id")
            != args.gradient_objective_id
        ):
            raise ValueError(
                "gradient identities differ from the reverse-factor archive"
            )
        gradient_layer = archive.load_layer_gradient(
            args.layer,
            device=device,
            verify_hash=args.verify_factor_hash,
        )
        factor_rows_cpu = gradient_layer.rows.cpu()
        factor_draws_cpu = gradient_layer.intermediate_draws.cpu()
        factor_strength_cpu = torch.ones((C.NUM_EXPERTS, 2))
    else:
        assert archive is not None
        factor_blocks, factor_rows, factor_draws = archive.load_layer_output_blocks(
            args.layer,
            device=device,
            verify_hash=args.verify_factor_hash,
        )
        factor_rows_cpu = factor_rows.cpu()
        factor_block_strength_cpu = (
            torch.diagonal(factor_blocks, dim1=-2, dim2=-1)
            .double()
            .mean(dim=-1)
            .cpu()
        )
        factor_strength_cpu = torch.stack(
            (
                factor_block_strength_cpu[:, :24].mean(dim=-1),
                factor_block_strength_cpu[:, 24:].mean(dim=-1),
            ),
            dim=-1,
        )
        factor_draws_cpu = factor_draws.cpu()
    if tuple(int(value) for value in factor_draws_cpu.tolist()) != profile_draws:
        raise ValueError("Fisher factors and served profile use different rotations")
    timings["factor_load"] = time.monotonic() - stage

    stage = time.monotonic()
    h13 = None
    h13_mean_diagonal = 1.0
    if not args.direct_viterbi:
        h13, _ = load_layer_hessians(args.hessians, args.layer)
        h13 = CoupledHadamardExecution(
            C.LATENT,
            C.EXPERT_INTER,
            CoupledHadamardSpec(intermediate_draw=0),
        ).transform_h13(h13.to(device=device))
        h13_mean_diagonal = float(torch.diagonal(h13).double().mean().item())
        if (
            not torch.isfinite(torch.tensor(h13_mean_diagonal))
            or h13_mean_diagonal <= 0
        ):
            raise ValueError("H13 must have a finite positive mean diagonal")
    gradient_normalization_scale = (
        h13_mean_diagonal
        if args.gradient_strength_normalization == "h13_mean_diagonal"
        else 1.0
    )
    effective_gradient_strength = (
        float(args.gradient_strength) * gradient_normalization_scale
    )
    timings["input_factor_load"] = time.monotonic() - stage

    stage = time.monotonic()
    source_store = OfficialMXFP4Store()
    matrices = ("w1", "w3", "w2") if args.include_w2 else ("w1", "w3")
    if len(experts) == C.NUM_EXPERTS:
        loader = InstantTensorKimiLayerLoader(source_store.root, device=device)
        banks, load_stats = loader.load_expert_banks(
            layer=args.layer,
            matrices=matrices,
        )
        source_load_result = {
            "serialized_bytes": load_stats.serialized_bytes,
            "dense_bytes": load_stats.dense_bytes,
            "seconds": load_stats.elapsed_seconds,
            "strategy": "whole-layer grouped decode",
        }
    else:
        banks = {matrix: {} for matrix in matrices}
        serialized_bytes = 0
        dense_bytes = 0
        with source_store.open_layer(
            args.layer,
            experts=experts,
            matrices=matrices,
        ) as layer_store:
            for expert in experts:
                for matrix in matrices:
                    value = layer_store.load_matrix(
                        args.layer,
                        expert,
                        matrix,
                        device=device,
                    )
                    banks[matrix][expert] = value
                    dense_bytes += value.numel() * value.element_size()
                    output_features, input_features = C.EXPERT_SHAPES[matrix]
                    serialized_bytes += output_features * (
                        input_features // 2 + input_features // C.MXFP4_BLOCK
                    )
        torch.cuda.synchronize(device)
        source_load_result = {
            "serialized_bytes": serialized_bytes,
            "dense_bytes": dense_bytes,
            "seconds": time.monotonic() - stage,
            "strategy": "selected-expert decode",
        }
    timings["source_load"] = time.monotonic() - stage

    backend = load_qsrt_encoder(args.exllamav3_root)
    install_sqg_quantizer(backend)
    plans = {
        matrix: plan_qsrt_matrix(
            torch.zeros(768, dtype=torch.long, device=device),
            K2,
            matrix=matrix,
            layout="importance_ordered",
        )
        for matrix in matrices
    }
    conditioning: dict[str, UniformSQGSharedInputConditioning | None] = {
        "w1": None,
        "w3": None,
    }
    overlay: dict[str, torch.Tensor] = {}
    refinement_summary = {
        matrix: {
            "supported_experts": 0,
            "changed_experts": 0,
            "changed_tiles": 0,
            "accepted_tiles": 0,
            "proposed_tiles": 0,
            "objective_before": 0.0,
            "objective_after": 0.0,
            "quadratic_before": 0.0,
            "quadratic_after": 0.0,
            "linear_before": 0.0,
            "linear_after": 0.0,
        }
        for matrix in ("w1", "w3")
    }
    supported = tuple(
        expert
        for expert in experts
        if int(factor_rows_cpu[expert]) > 0
        and bool(torch.all(factor_strength_cpu[expert] > 0.0))
    )
    unsupported = tuple(
        expert
        for expert in experts
        if int(factor_rows_cpu[expert]) == 0
        or not bool(torch.all(factor_strength_cpu[expert] > 0.0))
    )
    encode_started = time.monotonic()

    baseline_anchor_path = candidate_layer_path(args.candidate_pool, args.layer)
    anchor_path = (
        baseline_anchor_path
        if args.gradient_anchor_layer is None
        else args.gradient_anchor_layer.expanduser().resolve()
    )
    anchor_reader_context = (
        safe_open(anchor_path, framework="pt", device="cpu")
        if gradient_enabled
        else None
    )
    anchor_reader = (
        None if anchor_reader_context is None else anchor_reader_context.__enter__()
    )
    prepared_input_hessian: dict[str, torch.Tensor] = {}
    anchor_input_scales: dict[str, torch.Tensor] = {}
    if gradient_enabled:
        assert anchor_reader is not None
        for matrix in ("w1", "w3"):
            _, suh, _ = _load_anchor_matrix(
                anchor_reader,
                layer=args.layer,
                expert=experts[0],
                matrix=matrix,
                device=device,
            )
            seeds = qsrt_transform_seed_draw(args.layer, matrix)
            prepared_input_hessian[matrix] = (
                prepare_uniform_sqg_anchor_input_hessian(
                    h13,
                    suh,
                    bits=2,
                    device=device,
                    quantizer_module=backend,
                    input_sign_seed=seeds.input_sign,
                    output_sign_seed=seeds.output_sign,
                    rate_axis=matrix_rate_axis(matrix),
                    tailbite_context=args.tailbite_context,
                    ldlq_tf32=False,
                )
            )
            anchor_input_scales[matrix] = suh

    try:
        for begin in range(0, len(supported), args.expert_batch_size):
            batch_experts = supported[begin : begin + args.expert_batch_size]
            transformed = {matrix: [] for matrix in matrices}
            output_seeds = {matrix: [] for matrix in matrices}
            batch_gradient_factors: dict[
                int, tuple[torch.Tensor, torch.Tensor]
            ] = {}
            for expert in batch_experts:
                draw = profile_draws[expert]
                spec = CoupledHadamardSpec(intermediate_draw=draw)
                if args.include_w2:
                    encoded_triplet = encode_coupled_weights(
                        (
                            banks["w1"][expert],
                            banks["w3"][expert],
                            banks["w2"][expert],
                        ),
                        spec,
                    )
                    for matrix, value in zip(
                        ("w1", "w3", "w2"), encoded_triplet, strict=True
                    ):
                        transformed[matrix].append(value)
                else:
                    w1, w3 = encode_coupled_upstream_weights(
                        banks["w1"][expert], banks["w3"][expert], spec
                    )
                    transformed["w1"].append(w1)
                    transformed["w3"].append(w3)
                for matrix in matrices:
                    transform_draw = 0 if args.direct_viterbi else draw
                    output_seeds[matrix].append(
                        qsrt_transform_seed_draw(
                            args.layer,
                            matrix,
                            intermediate_draw=transform_draw,
                            expert=expert if transform_draw else None,
                        ).output_sign
                    )
                if gradient_enabled:
                    assert h13 is not None
                    assert gradient_layer is not None
                    batch_gradient_factors[expert] = (
                        gradient_layer.expert_joint_factors(
                            expert,
                            core_rcond=args.gradient_core_rcond,
                        )
                    )

            matrix_blocks = (("w1", 0), ("w3", 24), ("w2", -1))
            for matrix, block_begin in matrix_blocks[: len(matrices)]:
                sources = torch.stack(transformed[matrix])
                if args.direct_viterbi:
                    input_seeds = qsrt_transform_seed_draw(args.layer, matrix)
                    results = encode_uniform_sqg_direct_batch(
                        sources,
                        bits=2,
                        device=device,
                        quantizer_module=backend,
                        input_sign_seed=input_seeds.input_sign,
                        output_sign_seed=output_seeds[matrix],
                        rate_axis=matrix_rate_axis(matrix),
                        scale_scope_key=(
                            ("direct-viterbi", args.layer, matrix)
                            if matrix in ("w1", "w3")
                            else None
                        ),
                        shared_scale_axis=(
                            "input" if matrix in ("w1", "w3") else None
                        ),
                        tailbite_context=args.tailbite_context,
                    )
                    result_scales = []
                elif gradient_enabled:
                    assert anchor_reader is not None
                    assert gradient_layer is not None
                    results = []
                    result_scales = []
                    for source, expert, output_seed in zip(
                        sources,
                        batch_experts,
                        output_seeds[matrix],
                        strict=True,
                    ):
                        states, suh, svh = _load_anchor_matrix(
                            anchor_reader,
                            layer=args.layer,
                            expert=expert,
                            matrix=matrix,
                            device=device,
                        )
                        if not torch.equal(suh, anchor_input_scales[matrix]):
                            raise ValueError(
                                "uniform-K2 gradient anchors do not share input scales"
                            )
                        joint_left, source_right = batch_gradient_factors[expert]
                        offset = 0 if matrix == "w1" else C.EXPERT_INTER
                        source_factors = (
                            joint_left[offset : offset + C.EXPERT_INTER],
                            source_right,
                        )
                        input_seeds = qsrt_transform_seed_draw(args.layer, matrix)
                        results.append(
                            refine_uniform_sqg_anchor_gradient(
                                source,
                                h13,
                                states,
                                suh,
                                svh,
                                source_factors,
                                bits=2,
                                device=device,
                                quantizer_module=backend,
                                input_sign_seed=input_seeds.input_sign,
                                output_sign_seed=output_seed,
                                rate_axis=matrix_rate_axis(matrix),
                                gradient_strength=effective_gradient_strength,
                                anchor_id=args.gradient_anchor_id,
                                objective_id=args.gradient_objective_id,
                                tailbite_context=args.tailbite_context,
                                sweeps=args.gradient_refinement_sweeps,
                                input_hessian_work=prepared_input_hessian[matrix],
                            )
                        )
                        result_scales.append((suh, svh))
                        result = results[-1]
                        summary = refinement_summary[matrix]
                        summary["supported_experts"] += 1
                        changed_tiles = int(
                            torch.count_nonzero(
                                torch.any(result.states != states, dim=-1)
                            ).item()
                        )
                        summary["changed_tiles"] += changed_tiles
                        summary["changed_experts"] += int(changed_tiles > 0)
                        sweeps = (result.refinement or {}).get("sweeps", [])
                        if sweeps:
                            first_sweep = sweeps[0]
                            final_sweep = sweeps[-1]
                            summary["accepted_tiles"] += sum(
                                int(value["accepted_tiles"]) for value in sweeps
                            )
                            summary["proposed_tiles"] += sum(
                                int(value["proposed_tiles"]) for value in sweeps
                            )
                            for key in (
                                "objective_before",
                                "quadratic_before",
                                "linear_before",
                            ):
                                summary[key] += float(first_sweep[key])
                            for key in (
                                "objective_after",
                                "quadratic_after",
                                "linear_after",
                            ):
                                summary[key] += float(final_sweep[key])
                else:
                    assert factor_blocks is not None
                    assert h13 is not None
                    input_seeds = qsrt_transform_seed_draw(args.layer, matrix)
                    results, conditioning[matrix] = (
                        encode_uniform_sqg_two_sided_output_blocks_batch(
                            sources,
                            h13 if conditioning[matrix] is None else None,
                            factor_blocks[
                                list(batch_experts), block_begin : block_begin + 24
                            ],
                            bits=2,
                            device=device,
                            quantizer_module=backend,
                            input_sign_seed=input_seeds.input_sign,
                            output_sign_seed=output_seeds[matrix],
                            rate_axis=matrix_rate_axis(matrix),
                            scale_scope_key=("final-logit-fisher", args.layer, matrix),
                            shared_input=conditioning[matrix],
                            tailbite_context=args.tailbite_context,
                            output_damping_ratio=args.output_damping_ratio,
                            compute_objective=False,
                        )
                    )
                    result_scales = []
                for result_index, (expert, result) in enumerate(
                    zip(batch_experts, results, strict=True)
                ):
                    transform_draw = (
                        0 if args.direct_viterbi else profile_draws[expert]
                    )
                    seeds = qsrt_transform_seed_draw(
                        args.layer,
                        matrix,
                        intermediate_draw=transform_draw,
                        expert=expert if transform_draw else None,
                    )
                    direct_result = result if args.direct_viterbi else None
                    packed = finalize_qsrt_matrix_candidate(
                        QSRTMatrixCandidate(
                            reconstruction=(
                                direct_result.candidate.reconstruction
                                if direct_result is not None
                                else result.reconstruction
                                if gradient_enabled
                                else result.two_sided.reconstruction
                            ),
                            encoded=(
                                direct_result.candidate.states
                                if direct_result is not None
                                else result.states
                                if gradient_enabled
                                else result.two_sided.states
                            ),
                            tensors=(
                                {
                                    "suh": direct_result.suh,
                                    "svh": direct_result.svh,
                                }
                                if direct_result is not None
                                else
                                {
                                    "suh": result_scales[result_index][0],
                                    "svh": result_scales[result_index][1],
                                }
                                if gradient_enabled
                                else {"suh": result.suh, "svh": result.svh}
                            ),
                            plan=plans[matrix],
                            proxy=0.0,
                            transform_seeds=seeds,
                            global_scale=(
                                direct_result.global_scale
                                if direct_result is not None
                                else 1.0
                                if gradient_enabled
                                else result.global_scale
                            ),
                        ),
                        layer=args.layer,
                        logical_trellis_schema=QSRT_SCHEMA,
                        tailbite_context=args.tailbite_context,
                    )
                    for part, tensor in packed.tensors.items():
                        overlay[
                            candidate_tensor_name(
                                args.layer, expert, matrix, part
                            )
                        ] = tensor.detach().cpu().contiguous()
                del sources, results
            print(
                json.dumps(
                    {
                        "layer": args.layer,
                        "encoded_experts": min(
                            begin + len(batch_experts), len(supported)
                        ),
                        "supported_experts": len(supported),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if anchor_reader_context is not None:
            anchor_reader_context.__exit__(None, None, None)

    if unsupported:
        with safe_open(
            anchor_path
            if gradient_enabled
            else candidate_layer_path(args.candidate_pool, args.layer),
            framework="pt",
            device="cpu",
        ) as reader:
            for expert in unsupported:
                for matrix in ("w1", "w3"):
                    for part in ("trellis", "suh", "svh"):
                        name = candidate_tensor_name(
                            args.layer, expert, matrix, part
                        )
                        overlay[name] = reader.get_tensor(name).contiguous()

    expected_tensors = 3 * len(matrices) * len(experts)
    if len(overlay) != expected_tensors:
        raise RuntimeError(
            f"upstream overlay has {len(overlay)} tensors, expected {expected_tensors}"
        )
    timings["encode_and_pack"] = time.monotonic() - encode_started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    output_kind = (
        "qsrt_uniform_k2_direct_viterbi_all_linear_overlay"
        if args.include_w2
        else "qsrt_uniform_k2_direct_viterbi_upstream_overlay"
        if args.direct_viterbi
        else "qsrt_uniform_k2_gradient_refined_upstream_overlay"
        if gradient_enabled
        else "qsrt_uniform_k2_final_logit_fisher_upstream_overlay"
    )
    save_file(
        overlay,
        str(temporary),
        metadata={
            "kind": output_kind,
            "layer": str(args.layer),
            "experts": str(len(experts)),
        },
    )
    os.replace(temporary, args.output)
    timings["total"] = time.monotonic() - started
    result = {
        "kind": (
            "qsrt_uniform_k2_direct_viterbi_all_linear_layer"
            if args.include_w2
            else "qsrt_uniform_k2_direct_viterbi_upstream_layer"
            if args.direct_viterbi
            else "qsrt_uniform_k2_gradient_refined_upstream_layer"
            if gradient_enabled
            else "qsrt_uniform_k2_final_logit_fisher_upstream_layer"
        ),
        "schema_version": 1,
        "complete": True,
        "layer": args.layer,
        "experts": len(experts),
        "matrices": list(matrices),
        "supported_experts": len(supported),
        "anchor_fallback_experts": list(unsupported),
        "payload_overlay": str(args.output.resolve()),
        "payload_bytes": args.output.stat().st_size,
        "expert_batch_size": args.expert_batch_size,
        "output_damping_ratio": args.output_damping_ratio,
        "gradient_guidance": (
            None
            if not gradient_enabled
            else {
                "coefficient": args.gradient_strength,
                "normalization": args.gradient_strength_normalization,
                "normalization_scale": gradient_normalization_scale,
                "strength": effective_gradient_strength,
                "core_rcond": args.gradient_core_rcond,
                "anchor_layer": str(anchor_path),
                "anchor_id": args.gradient_anchor_id,
                "objective_id": args.gradient_objective_id,
                "sweeps": args.gradient_refinement_sweeps,
                "refinement": refinement_summary,
            }
        ),
        "timings": timings,
        "source_load": source_load_result,
    }
    if args.result is not None:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(args.result, result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

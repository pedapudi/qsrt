#!/usr/bin/env python
"""Direct-Viterbi encode FP32 source targets shifted by final-KL gradients."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

_REPOSITORY = Path(__file__).resolve().parents[1]
if str(_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY))

from qsrt import constants as C
from qsrt.exl3_loader import load_qsrt_encoder
from qsrt.instanttensor_kimi import InstantTensorKimiLayerLoader
from qsrt.kimi_dense_objective_gradients import DenseObjectiveGradientArchive
from qsrt.kimi_quantized_forward import (
    CompositeCandidateTensorReader,
    QSRTAnchorPayload,
)
from qsrt.pack.qsrt_candidates import candidate_tensor_name
from qsrt.pack.qsrt_encoder import (
    QSRTMatrixCandidate,
    finalize_qsrt_matrix_candidate,
    plan_qsrt_matrix,
    qsrt_transform_seed_draw,
)
from qsrt.pack.qsrt_validation import decode_candidate_matrix
from qsrt.qsrt import K2, SCHEMA as QSRT_SCHEMA, matrix_rate_axis
from qsrt.qsrt_coupled import (
    CoupledHadamardSpec,
    encode_coupled_down_weight,
    encode_coupled_upstream_weights,
)
from qsrt.sqg_quantizer import install_sqg_quantizer
from qsrt.qsrt_atoms_v2 import unpack_atoms_v2_format_section
from qsrt.source_weights import OfficialMXFP4Store
from qsrt.two_sided_qsrt import encode_uniform_sqg_direct_batch


DEFAULT_GRADIENTS = Path(
    "/data/kquant/research/qsrt-fp32-kl-refinement-layers89-92/gradients"
)
DEFAULT_DEST = Path(
    "/data/kquant/research/qsrt-fp32-kl-refinement-direct-k2-anchor-v1/overlays"
)
DEFAULT_PROFILE = Path(
    "/data/releases/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-v1"
)
DEFAULT_CANDIDATES = Path(
    "/data/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-CANDIDATES-v1"
)
DEFAULT_UPSTREAM_OVERLAYS = Path(
    "/data/kquant/research/"
    "k3-uniform-k2-direct-viterbi-all-linears-v1/upstream-overlays"
)
DEFAULT_W2_OVERLAYS = Path(
    "/data/kquant/research/k3-uniform-k2-direct-viterbi-w2-fixed-v1"
)
DEFAULT_OFFICIAL_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"


def _parse_alphas(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError("alphas must be comma-separated numbers") from error
    if not result or any(not math.isfinite(alpha) for alpha in result):
        raise argparse.ArgumentTypeError("alphas must be finite and nonempty")
    return result


def _alpha_name(alpha: float) -> str:
    if alpha == 0.0:
        return "zero"
    prefix = "p" if alpha > 0 else "m"
    digits = format(abs(alpha), ".8g").replace(".", "p")
    return prefix + digits


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _profile_draws(profile: Path, layer: int) -> tuple[int, ...]:
    path = profile / f"qsrt-layer-{layer:05d}.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        metadata = reader.metadata()
        if metadata is None or "profile" not in metadata:
            raise ValueError(f"QSRT layer {layer} lacks its atoms profile")
        formats, draws = unpack_atoms_v2_format_section(
            str(metadata["profile"]),
            reader.get_tensor("_qsrt_format_section"),
        )
    if draws is None or len(draws) != C.NUM_EXPERTS:
        raise ValueError(f"QSRT layer {layer} lacks coupled-Hadamard draws")
    if len(formats) != C.NUM_EXPERTS or any(value != "K2" for value in formats):
        raise ValueError(f"QSRT layer {layer} is not uniform K2")
    return tuple(int(value) for value in draws)


class _GradientLayer:
    def __init__(self, archive: DenseObjectiveGradientArchive, layer: int):
        if archive.manifest.get("complete") is not True:
            raise ValueError("dense-gradient archive is not sealed")
        if layer not in tuple(int(value) for value in archive.manifest["layers"]):
            raise ValueError(f"gradient archive does not contain layer {layer}")
        self.archive = archive
        self.layer = int(layer)
        self.width = int(archive.manifest["shard_experts"])
        self.hidden = int(archive.manifest["hidden_dimension"])
        self.intermediate = int(archive.manifest["intermediate_dimension"])
        self._mappings: dict[tuple[int, str], np.memmap] = {}

    def _record(self, expert: int) -> tuple[dict[str, object], int]:
        begin = (expert // self.width) * self.width
        key = self.archive.shard_key(self.layer, begin, begin + self.width)
        record = dict(self.archive.manifest["shards"])[key]
        return dict(record), expert - begin

    def tensor(self, expert: int, matrix: str) -> torch.Tensor:
        record, local = self._record(expert)
        role = "upstream" if matrix in ("w1", "w3") else "down"
        begin = int(record["expert_begin"])
        cache_key = (begin, role)
        mapping = self._mappings.get(cache_key)
        if mapping is None:
            file_record = dict(record["files"])[role]
            shape = tuple(int(value) for value in file_record["shape"])
            mapping = np.memmap(
                self.archive.root / str(file_record["file"]),
                mode="r",
                dtype=np.float32,
                shape=shape,
            )
            self._mappings[cache_key] = mapping
        value = mapping[local]
        if matrix == "w1":
            value = value[: self.intermediate]
        elif matrix == "w3":
            value = value[self.intermediate :]
        return torch.from_numpy(np.array(value, copy=True))

    def sum_squares(self) -> tuple[float, float]:
        upstream = 0.0
        down = 0.0
        for key, value in dict(self.archive.manifest["shards"]).items():
            record = dict(value)
            if int(record["layer"]) != self.layer:
                continue
            sums = dict(record["sum_squares"])
            upstream += float(sums["upstream"])
            down += float(sums["down"])
        return upstream, down


def _batches(experts: tuple[int, ...], width: int) -> Iterator[tuple[int, ...]]:
    for begin in range(0, len(experts), width):
        yield experts[begin : begin + width]


def _pack_result(
    *,
    result,
    plan,
    layer: int,
    matrix: str,
):
    seeds = qsrt_transform_seed_draw(layer, matrix)
    return finalize_qsrt_matrix_candidate(
        QSRTMatrixCandidate(
            reconstruction=result.candidate.reconstruction,
            encoded=result.candidate.states,
            tensors={"suh": result.suh, "svh": result.svh},
            plan=plan,
            proxy=0.0,
            transform_seeds=seeds,
            global_scale=result.global_scale,
        ),
        layer=layer,
        logical_trellis_schema=QSRT_SCHEMA,
        tailbite_context=128,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--gradients", type=Path, default=DEFAULT_GRADIENTS)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--anchor-overlay-root",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--official-revision", default=DEFAULT_OFFICIAL_REVISION)
    parser.add_argument(
        "--alphas",
        type=_parse_alphas,
        default=(0.0, 0.125, 0.25, 0.5, -0.25),
    )
    parser.add_argument("--upstream-batch-size", type=int, default=32)
    parser.add_argument("--down-batch-size", type=int, default=16)
    parser.add_argument("--expert-begin", type=int, default=0)
    parser.add_argument("--expert-end", type=int, default=C.NUM_EXPERTS)
    parser.add_argument(
        "--refinement-objective",
        choices=("source-target", "direct-gradient"),
        default="source-target",
        help=(
            "source-target shifts source weights before scale fitting; "
            "direct-gradient fixes anchor scales and adds the final-KL linear "
            "term in the Viterbi work basis"
        ),
    )
    parser.add_argument("--exllamav3-root", type=Path, default=Path("/home/luke/projects/exllamav3"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.layer not in C.MOE_LAYERS:
        raise ValueError("target layer is not a routed expert layer")
    if args.upstream_batch_size <= 0 or args.down_batch_size <= 0:
        raise ValueError("expert batch sizes must be positive")
    if not 0 <= args.expert_begin < args.expert_end <= C.NUM_EXPERTS:
        raise ValueError("expert range must lie within the complete layer")
    assigned_experts = tuple(range(args.expert_begin, args.expert_end))
    partial_layer = len(assigned_experts) != C.NUM_EXPERTS
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    archive = DenseObjectiveGradientArchive(args.gradients)
    gradients = _GradientLayer(archive, args.layer)
    draws = _profile_draws(args.profile.expanduser().resolve(), args.layer)
    overlay_roots = tuple(
        path.expanduser().resolve()
        for path in (
            args.anchor_overlay_root
            if args.anchor_overlay_root is not None
            else (DEFAULT_UPSTREAM_OVERLAYS, DEFAULT_W2_OVERLAYS)
        )
    )
    anchor_payload = QSRTAnchorPayload(
        args.candidate_pool.expanduser().resolve(),
        overlay_roots=overlay_roots,
    )
    anchor_base, anchor_overlays = anchor_payload.layer_paths(args.layer)
    source_store = OfficialMXFP4Store(revision=args.official_revision)
    loader = InstantTensorKimiLayerLoader(source_store.root, device=device)
    banks, load_stats = loader.load_expert_banks(
        layer=args.layer,
        matrices=C.EXPERT_MATRICES,
    )
    backend = load_qsrt_encoder(args.exllamav3_root)
    install_sqg_quantizer(backend)
    plans = {
        matrix: plan_qsrt_matrix(
            torch.zeros(768, dtype=torch.long, device=device),
            K2,
            matrix=matrix,
            layout="importance_ordered",
        )
        for matrix in C.EXPERT_MATRICES
    }

    destination = args.dest.expanduser().resolve()
    normalization_path = (
        destination
        / "zero"
        / "layers"
        / f"layer-{args.layer:03d}"
        / "completion.json"
    )
    if 0.0 not in args.alphas and normalization_path.is_file():
        normalization = json.loads(normalization_path.read_text())
        if (
            normalization.get("alpha_zero_bit_exact") is not True
            or int(normalization.get("layer", -1)) != args.layer
            or normalization.get("refinement_objective")
            != args.refinement_objective
        ):
            raise ValueError("alpha-zero normalization record is invalid")
        scales = {
            key: float(value)
            for key, value in normalization["gradient_scales"].items()
        }
        residual_upstream = float(
            normalization["anchor_residual_sum_squares"]["upstream"]
        )
        residual_down = float(
            normalization["anchor_residual_sum_squares"]["down"]
        )
        gradient_upstream = float(
            normalization["gradient_sum_squares"]["upstream"]
        )
        gradient_down = float(normalization["gradient_sum_squares"]["down"])
    else:
        residual_upstream = 0.0
        residual_down = 0.0
        with CompositeCandidateTensorReader(anchor_base, anchor_overlays) as reader:
            for expert in range(C.NUM_EXPERTS):
                spec = CoupledHadamardSpec(intermediate_draw=draws[expert])
                w1, w3 = encode_coupled_upstream_weights(
                    banks["w1"][expert],
                    banks["w3"][expert],
                    spec,
                )
                w2 = encode_coupled_down_weight(
                    banks["w2"][expert].float(),
                    spec,
                )
                for matrix, value in zip(
                    C.EXPERT_MATRICES,
                    (w1, w3, w2),
                    strict=True,
                ):
                    anchor = decode_candidate_matrix(
                        reader,
                        layer=args.layer,
                        expert=expert,
                        matrix=matrix,
                        mode_id=K2.mode_id,
                        device=device,
                    ).T
                    error = anchor.float() - value.float()
                    square_sum = float(error.square().sum(dtype=torch.float64).item())
                    if matrix in ("w1", "w3"):
                        residual_upstream += square_sum
                    else:
                        residual_down += square_sum
        gradient_upstream, gradient_down = gradients.sum_squares()

    if min(
        residual_upstream,
        residual_down,
        gradient_upstream,
        gradient_down,
    ) <= 0:
        raise ValueError("gradient or anchor residual norm is zero")
    scales = {
        "upstream": math.sqrt(residual_upstream / gradient_upstream),
        "down": math.sqrt(residual_down / gradient_down),
    }

    layer_results: dict[str, object] = {}
    with CompositeCandidateTensorReader(anchor_base, anchor_overlays) as anchor_reader:
        for alpha in args.alphas:
            alpha_started = time.monotonic()
            name = _alpha_name(alpha)
            layer_directory = destination / name / "layers" / f"layer-{args.layer:03d}"
            if partial_layer:
                layer_directory = (
                    layer_directory
                    / "shards"
                    / f"experts-{args.expert_begin:04d}-{args.expert_end:04d}"
                )
            completion = layer_directory / "completion.json"
            if completion.is_file():
                layer_results[name] = json.loads(completion.read_text())
                continue
            overlay: dict[str, torch.Tensor] = {}
            for matrices, batch_size, scale_role in (
                (("w1", "w3"), args.upstream_batch_size, "upstream"),
                (("w2",), args.down_batch_size, "down"),
            ):
                primer_pending = matrices == ("w1", "w3") and args.expert_begin != 0
                for batch_experts in _batches(assigned_experts, batch_size):
                    has_primer = primer_pending
                    quantization_experts = (
                        (0,) + batch_experts if has_primer else batch_experts
                    )
                    transformed = {
                        matrix: [] for matrix in matrices
                    }
                    gradient_by_matrix = {matrix: [] for matrix in matrices}
                    output_seeds = {matrix: [] for matrix in matrices}
                    for expert in quantization_experts:
                        spec = CoupledHadamardSpec(
                            intermediate_draw=draws[expert]
                        )
                        if matrices == ("w1", "w3"):
                            source_by_matrix = dict(
                                zip(
                                    matrices,
                                    encode_coupled_upstream_weights(
                                        banks["w1"][expert],
                                        banks["w3"][expert],
                                        spec,
                                    ),
                                    strict=True,
                                )
                            )
                        else:
                            source_by_matrix = {
                                "w2": encode_coupled_down_weight(
                                    banks["w2"][expert].float(),
                                    spec,
                                )
                            }
                        for matrix in matrices:
                            gradient = gradients.tensor(expert, matrix).to(
                                device=device,
                                dtype=torch.float32,
                                non_blocking=False,
                            )
                            if args.refinement_objective == "direct-gradient":
                                transformed[matrix].append(
                                    source_by_matrix[matrix].float()
                                )
                                gradient_by_matrix[matrix].append(gradient)
                            else:
                                transformed[matrix].append(
                                    source_by_matrix[matrix].float()
                                    - float(alpha)
                                    * scales[scale_role]
                                    * gradient
                                )
                            output_seeds[matrix].append(
                                qsrt_transform_seed_draw(
                                    args.layer,
                                    matrix,
                                ).output_sign
                            )
                    for matrix in matrices:
                        scope = None
                        shared_scale_axis = None
                        if matrix in ("w1", "w3"):
                            if (
                                alpha == 0.0
                                or args.refinement_objective == "direct-gradient"
                            ):
                                scope = ("direct-viterbi", args.layer, matrix)
                            else:
                                scope = (
                                    "fp32-kl-refinement",
                                    name,
                                    args.layer,
                                    matrix,
                                )
                            shared_scale_axis = "input"
                        encoded = encode_uniform_sqg_direct_batch(
                            torch.stack(transformed[matrix]),
                            bits=2,
                            device=device,
                            quantizer_module=backend,
                            input_sign_seed=qsrt_transform_seed_draw(
                                args.layer,
                                matrix,
                            ).input_sign,
                            output_sign_seed=output_seeds[matrix],
                            rate_axis=matrix_rate_axis(matrix),
                            scale_scope_key=scope,
                            shared_scale_axis=shared_scale_axis,
                            tailbite_context=128,
                            source_gradients=(
                                torch.stack(gradient_by_matrix[matrix])
                                if args.refinement_objective == "direct-gradient"
                                else None
                            ),
                            gradient_strength=(
                                float(alpha) * scales[scale_role]
                                if args.refinement_objective == "direct-gradient"
                                else 0.0
                            ),
                        )
                        if has_primer:
                            encoded = encoded[1:]
                        for expert, result in zip(
                            batch_experts,
                            encoded,
                            strict=True,
                        ):
                            packed = _pack_result(
                                result=result,
                                plan=plans[matrix],
                                layer=args.layer,
                                matrix=matrix,
                            )
                            for part, tensor in packed.tensors.items():
                                tensor_name = candidate_tensor_name(
                                    args.layer,
                                    expert,
                                    matrix,
                                    part,
                                )
                                value = tensor.detach().cpu().contiguous()
                                if alpha == 0.0:
                                    reference = anchor_reader.get_tensor(tensor_name)
                                    if not torch.equal(value, reference):
                                        raise ValueError(
                                            "alpha-zero direct encode differs from the "
                                            f"anchor at {tensor_name}"
                                        )
                                else:
                                    overlay[tensor_name] = value
                    primer_pending = False
            record: dict[str, object] = {
                "kind": "qsrt_fp32_final_kl_gradient_refinement_layer",
                "schema_version": 1,
                "complete": True,
                "layer": args.layer,
                "expert_begin": args.expert_begin,
                "expert_end": args.expert_end,
                "experts": len(assigned_experts),
                "alpha": alpha,
                "refinement_objective": args.refinement_objective,
                "gradient_scales": scales,
                "anchor_residual_sum_squares": {
                    "upstream": residual_upstream,
                    "down": residual_down,
                },
                "gradient_sum_squares": {
                    "upstream": gradient_upstream,
                    "down": gradient_down,
                },
                "elapsed_seconds": time.monotonic() - alpha_started,
            }
            layer_directory.mkdir(parents=True, exist_ok=True)
            if alpha != 0.0:
                output = layer_directory / "payload-overlay.safetensors"
                temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
                save_file(
                    overlay,
                    str(temporary),
                    metadata={
                        "kind": "qsrt_fp32_final_kl_gradient_refinement_overlay",
                        "layer": str(args.layer),
                        "expert_begin": str(args.expert_begin),
                        "expert_end": str(args.expert_end),
                        "alpha": str(alpha),
                        "refinement_objective": args.refinement_objective,
                    },
                )
                os.replace(temporary, output)
                record["payload_overlay"] = str(output.resolve())
                record["payload_bytes"] = output.stat().st_size
            else:
                record["alpha_zero_bit_exact"] = True
            _atomic_json(completion, record)
            layer_results[name] = record
            print(json.dumps(record, sort_keys=True), flush=True)

    summary = {
        "complete": True,
        "layer": args.layer,
        "source_load_seconds": load_stats.elapsed_seconds,
        "alphas": layer_results,
    }
    if partial_layer:
        summary_root = (
            destination
            / _alpha_name(args.alphas[0])
            / "layers"
            / f"layer-{args.layer:03d}"
            / "shards"
            / f"experts-{args.expert_begin:04d}-{args.expert_end:04d}"
        )
        _atomic_json(summary_root / "summary.json", summary)
    else:
        summary_root = (
            destination / _alpha_name(args.alphas[0])
            if len(args.alphas) == 1
            else destination
        )
        _atomic_json(summary_root / f"layer-{args.layer:03d}.json", summary)


if __name__ == "__main__":
    main()

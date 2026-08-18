#!/usr/bin/env python3
"""Measure how uniform-K2 SQG and MCG errors propagate through one expert.

Both codecs use the same coupled H512/H128 coordinate system, layer-global
input covariance, reconstruction-specific expert-local post-activation
covariance, dense-H BlockLDLQ encoder, and independent scale fit. The output
contains all eight gate/up/down codec hybrids and the exact immediate mapping
through routed aggregation, RMS normalization, and the routed output
projection. It does not alter or materialize a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file as load_torch_file

from qsrt import constants as C
from qsrt.all_row_capture import MaterializedExpertRows, materialize_expert_rows
from qsrt.capture import load_layer_hessians
from qsrt.coupled_expert_study import CoupledTriplet, RoutedOutputMetric, expert_hidden
from qsrt.distortion_transfer import quadratic_matrix_sse, routed_error_geometry
from qsrt.exl3_loader import load_qsrt_encoder
from qsrt.exl3_reference import CODEBOOK_SQG_XOR_CHEB_T12
from qsrt.io.stream import load_tensor
from qsrt.pack.qsrt_encoder import default_qsrt_transform_seeds
from qsrt.pooled_calibration import candidate_h2, collect_coupled_hidden_statistics
from qsrt.qsrt import matrix_rate_axis
from qsrt.qsrt_atoms_v2 import unpack_atoms_v2_format_section
from qsrt.qsrt_codec_pilot import CODEBOOK_MCG, encode_uniform_candidate
from qsrt.qsrt_coupled import (
    CoupledHadamardExecution,
    CoupledHadamardSpec,
    decode_coupled_weights,
    encode_coupled_weights,
)
from qsrt.source_weights import OfficialMXFP4Store
from qsrt.sqg_quantizer import install_sqg_quantizer


KIND = "qsrt_uniform_k2_distortion_transfer_screen"
SCHEMA_VERSION = 1
CODEBOOKS = (CODEBOOK_MCG, CODEBOOK_SQG_XOR_CHEB_T12)
MATRICES = ("w1", "w3", "w2")
DEFAULT_CAPTURE = Path(
    "/data/datasets/kquant/captures/k3-all-routed-4m-v1.kqrows"
)
DEFAULT_HESSIANS = Path(
    "/data/datasets/kquant/hessians/"
    "k3-denseh-broad-v7-4m-train-h13-identity-qsrt-v1.kqhess"
)
DEFAULT_EVALUATION = Path(
    "/data/datasets/kquant/captures/"
    "k3-codec-diverse-validation-v3-128k-input-v1.kqsamples"
)
DEFAULT_PROFILE = Path(
    "/data/releases/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _source_triplet(
    layer_store: object,
    *,
    layer: int,
    expert: int,
    device: torch.device,
) -> CoupledTriplet:
    return CoupledTriplet(
        *(
            layer_store.load_matrix(layer, expert, matrix, device=device).float()
            for matrix in MATRICES
        )
    )


def _profile_draw(profile: Path, layer: int, expert: int) -> int:
    layer_path = profile / f"qsrt-layer-{layer:05d}.safetensors"
    with safe_open(layer_path, framework="pt", device="cpu") as reader:
        metadata = reader.metadata()
        if metadata is None or "profile" not in metadata:
            raise ValueError("profile layer lacks the atoms-v2 format identity")
        formats, draws = unpack_atoms_v2_format_section(
            str(metadata["profile"]), reader.get_tensor("_qsrt_format_section")
        )
    if formats[expert] != "K2" or draws is None:
        raise ValueError("profile expert is not a coupled uniform-K2 expert")
    return int(draws[expert])


def _fit_batches(
    rows: MaterializedExpertRows,
    *,
    maximum_rows: int,
    batch_rows: int,
) -> Iterable[dict[str, torch.Tensor]]:
    count = int(rows.row_indices.numel())
    if count == 0:
        return
    if count <= maximum_rows:
        selected = torch.arange(count)
    else:
        selected = torch.linspace(0, count - 1, maximum_rows).round().long()
    for begin in range(0, selected.numel(), batch_rows):
        index = selected[begin : begin + batch_rows]
        yield {
            "input": rows.tensors["input"].index_select(0, index),
            "row_index": rows.row_indices.index_select(0, index),
            "route_slot": rows.route_slots.index_select(0, index),
            "route_weight": rows.route_weights.index_select(0, index),
        }


def _evaluation_rows(samples: Any, expert: int, maximum_rows: int) -> dict[str, torch.Tensor]:
    locations = torch.nonzero(samples.input_experts == expert, as_tuple=False)
    if locations.numel() == 0:
        raise ValueError(f"expert {expert} has no evaluation rows")
    if locations.shape[0] > maximum_rows:
        index = torch.linspace(0, locations.shape[0] - 1, maximum_rows).round().long()
        locations = locations.index_select(0, index)
    rows = locations[:, 0]
    slots = locations[:, 1]
    return {
        "inputs": samples.input_values.index_select(0, rows).float(),
        "gates": samples.input_gates[rows, slots].float(),
        "aggregate": samples.routed_latent.index_select(0, rows).float(),
        "documents": torch.bitwise_right_shift(
            samples.input_observations.index_select(0, rows), 32
        ),
    }


def _load_evaluation_samples(cache: Path, layer: int) -> Any:
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest.get("kind") not in (
        "qsrt_layer_sample_cache",
        "kquant_layer_sample_cache",
    ) or int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("evaluation input is not a supported layer-sample cache")
    entry = manifest.get("layers", {}).get(str(layer))
    if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
        raise ValueError(f"evaluation cache has no decoder layer {layer}")
    tensors = load_torch_file(str(cache / entry["file"]), device="cpu")
    required = (
        "input.values",
        "input.observation",
        "input.experts",
        "input.gates",
        "input.routed_latent",
    )
    missing = [name for name in required if name not in tensors]
    if missing:
        raise ValueError(f"evaluation layer lacks tensors {missing}")
    return SimpleNamespace(
        input_values=tensors["input.values"],
        input_observations=tensors["input.observation"],
        input_experts=tensors["input.experts"],
        input_gates=tensors["input.gates"],
        routed_latent=tensors["input.routed_latent"],
    )


def _output_metric(store: OfficialMXFP4Store, layer: int, device: torch.device) -> RoutedOutputMetric:
    prefix = f"{C.LM_PREFIX}layers.{layer}.block_sparse_moe"
    return RoutedOutputMetric(
        load_tensor(store.cache, f"{prefix}.routed_expert_norm.weight")
        .float()
        .to(device),
        load_tensor(store.cache, C.latent_up_proj_tensor(layer)).float().to(device),
    )


def _encode_matrix(
    source: torch.Tensor,
    hessian: torch.Tensor,
    *,
    layer: int,
    expert: int,
    matrix: str,
    codebook: str,
    device: torch.device,
    quantizer_module: Any,
    ldlq_tf32: bool,
    tailbite_context: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    seeds = default_qsrt_transform_seeds(layer, matrix)
    result = encode_uniform_candidate(
        source,
        bits=2,
        codebook=codebook,
        device=device,
        quantizer_module=quantizer_module,
        input_sign_seed=seeds.input_sign,
        output_sign_seed=seeds.output_sign,
        rate_axis=matrix_rate_axis(matrix),
        scale_scope_key=(KIND, layer, expert, codebook, matrix),
        g_scale_into_sv=matrix in ("w1", "w3"),
        ldlq_tf32=ldlq_tf32,
        tailbite_context=tailbite_context,
        hessian=hessian,
    )
    return result["reconstruction"].float().to(device), result["payload"]


def _matrix_metrics(
    source_coordinates: CoupledTriplet,
    candidate_coordinates: CoupledTriplet,
    source_ordinary: CoupledTriplet,
    candidate_ordinary: CoupledTriplet,
    *,
    h13: torch.Tensor,
    candidate_h2_matrix: torch.Tensor,
    source_h2: torch.Tensor,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for index, matrix in enumerate(MATRICES):
        source = source_coordinates.tensors()[index]
        candidate = candidate_coordinates.tensors()[index]
        ordinary_source = source_ordinary.tensors()[index]
        ordinary_candidate = candidate_ordinary.tensors()[index]
        raw_sse = float((ordinary_candidate.double() - ordinary_source.double()).square().sum())
        raw_energy = float(ordinary_source.double().square().sum())
        covariance = h13 if matrix != "w2" else candidate_h2_matrix
        dense_sse, dense_energy = quadratic_matrix_sse(source, candidate, covariance)
        entry: dict[str, object] = {
            "raw_weight_sse": raw_sse,
            "raw_weight_energy": raw_energy,
            "raw_weight_nmse": raw_sse / raw_energy,
            "encoding_covariance_sse": dense_sse,
            "encoding_covariance_energy": dense_energy,
            "encoding_covariance_nmse": dense_sse / dense_energy,
        }
        if matrix == "w2":
            common_sse, common_energy = quadratic_matrix_sse(
                source, candidate, source_h2
            )
            entry.update(
                {
                    "source_hidden_covariance_sse": common_sse,
                    "source_hidden_covariance_energy": common_energy,
                    "source_hidden_covariance_nmse": common_sse / common_energy,
                }
            )
        result[matrix] = entry
    return result


def _hybrid_metrics(
    source: CoupledTriplet,
    ordinary: Mapping[str, CoupledTriplet],
    rows: Mapping[str, torch.Tensor],
    output_metric: RoutedOutputMetric,
) -> dict[str, object]:
    inputs = rows["inputs"]
    source_gate = F.linear(inputs, source.gate)
    source_up = F.linear(inputs, source.up)
    source_hidden = expert_hidden(inputs, source)
    source_output = F.linear(source_hidden, source.down)
    result: dict[str, object] = {}
    for gate_codebook in CODEBOOKS:
        for up_codebook in CODEBOOKS:
            gate = ordinary[gate_codebook].gate
            up = ordinary[up_codebook].up
            gate_projection = F.linear(inputs, gate)
            up_projection = F.linear(inputs, up)
            hidden = expert_hidden(inputs, CoupledTriplet(gate, up, source.down))
            gate_error = gate_projection - source_gate
            up_error = up_projection - source_up
            hidden_error = hidden - source_hidden
            for down_codebook in CODEBOOKS:
                down = ordinary[down_codebook].down
                output = F.linear(hidden, down)
                output_error = output - source_output
                routed_error = rows["gates"][:, None] * output_error
                geometry = routed_error_geometry(
                    rows["aggregate"], routed_error, output_metric
                )

                def normalized(error: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
                    sse = float(error.double().square().sum())
                    energy = float(reference.double().square().sum())
                    return {"sse": sse, "energy": energy, "nmse": sse / energy}

                key = "".join(
                    "s" if value == CODEBOOK_SQG_XOR_CHEB_T12 else "m"
                    for value in (gate_codebook, up_codebook, down_codebook)
                )
                result[key] = {
                    "gate": normalized(gate_error, source_gate),
                    "up": normalized(up_error, source_up),
                    "post_situ": normalized(hidden_error, source_hidden),
                    "expert_output": normalized(output_error, source_output),
                    "route_weighted_expert_sse": geometry.residual_sse,
                    "residual_geometry": asdict(geometry),
                    "codebooks": {
                        "w1": gate_codebook,
                        "w3": up_codebook,
                        "w2": down_codebook,
                    },
                }
    return result


def _codec_comparison(
    candidates: Mapping[str, Mapping[str, object]],
    hybrids: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    stages = {
        "raw_weight": (
            sum(
                float(candidates[CODEBOOK_MCG]["matrices"][matrix]["raw_weight_sse"])
                for matrix in MATRICES
            ),
            sum(
                float(candidates[CODEBOOK_SQG_XOR_CHEB_T12]["matrices"][matrix]["raw_weight_sse"])
                for matrix in MATRICES
            ),
        ),
        "gate_preactivation": (
            float(hybrids["mmm"]["gate"]["sse"]),
            float(hybrids["sss"]["gate"]["sse"]),
        ),
        "up_preactivation": (
            float(hybrids["mmm"]["up"]["sse"]),
            float(hybrids["sss"]["up"]["sse"]),
        ),
        "post_situ": (
            float(hybrids["mmm"]["post_situ"]["sse"]),
            float(hybrids["sss"]["post_situ"]["sse"]),
        ),
        "expert_output": (
            float(hybrids["mmm"]["expert_output"]["sse"]),
            float(hybrids["sss"]["expert_output"]["sse"]),
        ),
        "routed_expert": (
            float(hybrids["mmm"]["route_weighted_expert_sse"]),
            float(hybrids["sss"]["route_weighted_expert_sse"]),
        ),
        "mapped_linear": (
            float(hybrids["mmm"]["residual_geometry"]["mapped_linear_sse"]),
            float(hybrids["sss"]["residual_geometry"]["mapped_linear_sse"]),
        ),
        "mapped_exact": (
            float(hybrids["mmm"]["residual_geometry"]["mapped_exact_sse"]),
            float(hybrids["sss"]["residual_geometry"]["mapped_exact_sse"]),
        ),
    }
    return {
        stage: {
            "mcg_sse": mcg,
            "sqg_sse": sqg,
            "sqg_relative_reduction": 1.0 - sqg / mcg,
            "log_mcg_over_sqg": math.log(mcg / sqg),
        }
        for stage, (mcg, sqg) in stages.items()
        if mcg > 0 and sqg > 0
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--hessians", type=Path, default=DEFAULT_HESSIANS)
    parser.add_argument("--evaluation-cache", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--maximum-fit-rows", type=int, default=8192)
    parser.add_argument("--maximum-evaluation-rows", type=int, default=1024)
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exllamav3-root", type=Path, default=Path("/home/luke/projects/exllamav3"))
    parser.add_argument("--official-revision", default=C.REVISION)
    parser.add_argument("--ldlq-tf32", action="store_true")
    parser.add_argument("--tailbite-context", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.layer not in C.MOE_LAYERS or not 0 <= args.expert < C.NUM_EXPERTS:
        parser.error("layer or expert lies outside the Kimi-K3 routed-expert geometry")
    if args.maximum_fit_rows <= 0 or args.maximum_evaluation_rows <= 0 or args.batch_rows <= 0:
        parser.error("row limits and batch size must be positive")
    if not 1 <= args.tailbite_context <= 128:
        parser.error("tail-biting context must lie in 1..128")
    return args


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("distortion-transfer encoding requires CUDA")
    draw = _profile_draw(args.profile, args.layer, args.expert)
    signature = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "source_model": C.MODEL_ID,
        "source_revision": args.official_revision,
        "layer": args.layer,
        "expert": args.expert,
        "codebooks": list(CODEBOOKS),
        "rate": 2,
        "capture": str(args.capture.resolve()),
        "capture_manifest_sha256": _sha256(args.capture / "manifest.json"),
        "hessians": str(args.hessians.resolve()),
        "hessian_manifest_sha256": _sha256(args.hessians / "manifest.json"),
        "evaluation_cache": str(args.evaluation_cache.resolve()),
        "evaluation_manifest_sha256": _sha256(args.evaluation_cache / "manifest.json"),
        "profile": str(args.profile.resolve()),
        "profile_completion_sha256": _sha256(args.profile / "qsrt-completion.json"),
        "coupled_transform": {
            "residual_block_size": 512,
            "preactivation_block_size": 128,
            "postactivation_block_size": 128,
            "residual_draw": 0,
            "intermediate_draw": draw,
        },
        "maximum_fit_occurrences": args.maximum_fit_rows,
        "maximum_evaluation_occurrences": args.maximum_evaluation_rows,
        "ldlq_tf32": args.ldlq_tf32,
        "tailbite_context": args.tailbite_context,
        "writes_checkpoint_payloads": False,
    }
    receipt: dict[str, object] = {"signature": signature, "complete": False}
    _atomic_json(args.output, receipt)
    started = time.time()

    fit_rows = materialize_expert_rows(
        args.capture, args.layer, args.expert, fields=("input",), verify_hashes=False
    )
    samples = _load_evaluation_samples(args.evaluation_cache, args.layer)
    evaluation = _evaluation_rows(samples, args.expert, args.maximum_evaluation_rows)
    evaluation = {name: value.to(device) for name, value in evaluation.items()}
    h13, _ = load_layer_hessians(args.hessians, args.layer)
    h13 = h13.float().to(device)

    store = OfficialMXFP4Store(revision=args.official_revision)
    with store.open_layer(args.layer, experts=(args.expert,)) as layer_store:
        source = _source_triplet(
            layer_store, layer=args.layer, expert=args.expert, device=device
        )
    output_metric = _output_metric(store, args.layer, device)
    spec = CoupledHadamardSpec(intermediate_draw=draw)
    execution = CoupledHadamardExecution(source.hidden, source.intermediate, spec)
    source_coordinates = CoupledTriplet(*encode_coupled_weights(source.tensors(), spec))
    transformed_h13 = execution.transform_h13(h13)
    source_statistics = collect_coupled_hidden_statistics(
        _fit_batches(
            fit_rows, maximum_rows=args.maximum_fit_rows, batch_rows=args.batch_rows
        ),
        source=source,
        candidate_coordinates=source_coordinates,
        execution=execution,
    )
    source_h2, source_h2_evidence = candidate_h2(source_statistics)

    quantizer_module = load_qsrt_encoder(args.exllamav3_root)
    install_sqg_quantizer(quantizer_module)
    candidates: dict[str, dict[str, object]] = {}
    ordinary: dict[str, CoupledTriplet] = {}
    for codebook in CODEBOOKS:
        gate, gate_evidence = _encode_matrix(
            source_coordinates.gate,
            transformed_h13,
            layer=args.layer,
            expert=args.expert,
            matrix="w1",
            codebook=codebook,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=args.ldlq_tf32,
            tailbite_context=args.tailbite_context,
        )
        up, up_evidence = _encode_matrix(
            source_coordinates.up,
            transformed_h13,
            layer=args.layer,
            expert=args.expert,
            matrix="w3",
            codebook=codebook,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=args.ldlq_tf32,
            tailbite_context=args.tailbite_context,
        )
        upstream = CoupledTriplet(gate, up, source_coordinates.down)
        statistics = collect_coupled_hidden_statistics(
            _fit_batches(
                fit_rows,
                maximum_rows=args.maximum_fit_rows,
                batch_rows=args.batch_rows,
            ),
            source=source,
            candidate_coordinates=upstream,
            execution=execution,
        )
        h2, h2_evidence = candidate_h2(statistics)
        down, down_evidence = _encode_matrix(
            source_coordinates.down,
            h2,
            layer=args.layer,
            expert=args.expert,
            matrix="w2",
            codebook=codebook,
            device=device,
            quantizer_module=quantizer_module,
            ldlq_tf32=args.ldlq_tf32,
            tailbite_context=args.tailbite_context,
        )
        coordinates = CoupledTriplet(gate, up, down)
        decoded = CoupledTriplet(*decode_coupled_weights(coordinates.tensors(), spec))
        ordinary[codebook] = decoded
        candidates[codebook] = {
            "matrices": _matrix_metrics(
                source_coordinates,
                coordinates,
                source,
                decoded,
                h13=transformed_h13,
                candidate_h2_matrix=h2,
                source_h2=source_h2,
            ),
            "candidate_h2": h2_evidence,
            "payloads": {
                "w1": gate_evidence,
                "w3": up_evidence,
                "w2": down_evidence,
            },
        }
        print(
            f"layer {args.layer} expert {args.expert}: encoded {codebook}",
            flush=True,
        )

    hybrids = _hybrid_metrics(source, ordinary, evaluation, output_metric)
    receipt.update(
        {
            "fit_support": {
                "captured_occurrences": int(fit_rows.row_indices.numel()),
                "sampled_occurrences": min(
                    int(fit_rows.row_indices.numel()), args.maximum_fit_rows
                ),
                "source_h2": source_h2_evidence,
            },
            "evaluation_support": {
                "occurrences": int(evaluation["inputs"].shape[0]),
                "documents": int(torch.unique(evaluation["documents"]).numel()),
            },
            "candidates": candidates,
            "hybrids": hybrids,
            "sqg_vs_mcg": _codec_comparison(candidates, hybrids),
            "seconds": time.time() - started,
            "complete": True,
        }
    )
    _atomic_json(args.output, receipt)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sqg_vs_mcg": receipt["sqg_vs_mcg"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

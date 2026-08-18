#!/usr/bin/env python3
"""Fit frozen low-rank corrections to a bounded set of K2 expert errors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Mapping

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from qsrt import constants as C
from qsrt.all_row_capture import layer_chunks
from qsrt.coupled_expert_study import CoupledTriplet, expert_hidden
from qsrt.kimi_quantized_forward import (
    CompositeCandidateTensorReader,
    QSRTAnchorPayload,
)
from qsrt.low_rank_adapters import (
    LowRankAdapterFit,
    fit_plain_error_adapter,
    fit_weighted_error_adapter,
)
from qsrt.pack.qsrt_validation import decode_candidate_matrix
from qsrt.qsrt import K2
from qsrt.qsrt_atoms_v2 import unpack_atoms_v2_format_section
from qsrt.qsrt_coupled import CoupledHadamardSpec, decode_coupled_weights
from qsrt.source_weights import OfficialMXFP4Store


DEFAULT_MODEL = Path(
    "/data/models/Kimi-K3-QSRT-K2-DIRECT-VITERBI-ALL-LINEARS-v1-model"
)
DEFAULT_CANDIDATE_POOL = Path(
    "/data/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-CANDIDATES-v1"
)
DEFAULT_W2_OVERLAY = Path(
    "/data/kquant/research/k3-uniform-k2-direct-viterbi-w2-fixed-v1"
)
DEFAULT_UPSTREAM_OVERLAY = Path(
    "/data/kquant/research/k3-uniform-k2-direct-viterbi-all-linears-v1/"
    "upstream-overlays"
)
DEFAULT_ROUTED_ROWS = Path(
    "/data/datasets/kquant/captures/k3-all-routed-4m-v1.kqrows"
)
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=84)
    parser.add_argument(
        "--experts",
        default=",".join(str(value) for value in range(0, 896, 56)),
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--ranks", default="2,4,8,16")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--candidate-pool", type=Path, default=DEFAULT_CANDIDATE_POOL
    )
    parser.add_argument(
        "--overlay-root",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--routed-rows", type=Path, default=DEFAULT_ROUTED_ROWS)
    parser.add_argument("--validation-document-modulus", type=int, default=5)
    parser.add_argument("--validation-document-residue", type=int, default=0)
    parser.add_argument("--batch-rows", type=int, default=2048)
    parser.add_argument("--oversampling", type=int, default=8)
    parser.add_argument("--power-iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--dest", type=Path, required=True)
    return parser


def _parse_unique_ints(value: str, *, name: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique integer values")
    return result


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _profile_draws(model: Path, layer: int) -> tuple[int, ...]:
    path = model / f"qsrt-layer-{layer:05d}.safetensors"
    with safe_open(path, framework="pt", device="cpu") as reader:
        metadata = reader.metadata()
        if metadata is None or "profile" not in metadata:
            raise ValueError(f"{path} lacks its atom profile")
        _formats, draws = unpack_atoms_v2_format_section(
            str(metadata["profile"]), reader.get_tensor("_qsrt_format_section")
        )
    if draws is None or len(draws) != C.NUM_EXPERTS:
        raise ValueError(f"{path} lacks coupled-Hadamard draws")
    return tuple(int(value) for value in draws)


def _collect_upstream_rows(
    root: Path,
    *,
    layer: int,
    experts: tuple[int, ...],
    validation_document_modulus: int,
    validation_document_residue: int,
) -> dict[int, dict[str, tuple[torch.Tensor, torch.Tensor, int]]]:
    """Read the 4M-token layer once and retain only selected occurrences."""

    populations = ("fit", "validation")
    row_parts = {
        expert: {population: [] for population in populations}
        for expert in experts
    }
    weight_parts = {
        expert: {population: [] for population in populations}
        for expert in experts
    }
    document_parts = {
        expert: {population: [] for population in populations}
        for expert in experts
    }
    for chunk in layer_chunks(root, layer, verify_hashes=False):
        tensors = chunk.load(
            verify_checksum=False,
            fields=("input", "expert_indices", "route_weights", "document_id"),
        )
        routed = tensors["expert_indices"]
        for expert in experts:
            positions = torch.nonzero(routed == expert, as_tuple=False)
            if not positions.numel():
                continue
            rows = positions[:, 0]
            slots = positions[:, 1]
            documents = tensors["document_id"].index_select(0, rows)
            validation = (
                documents.remainder(validation_document_modulus)
                == validation_document_residue
            )
            for population, mask in (
                ("fit", ~validation),
                ("validation", validation),
            ):
                if not torch.any(mask):
                    continue
                selected_rows = rows[mask]
                selected_slots = slots[mask]
                row_parts[expert][population].append(
                    tensors["input"].index_select(0, selected_rows)
                )
                weight_parts[expert][population].append(
                    tensors["route_weights"][selected_rows, selected_slots]
                )
                document_parts[expert][population].append(documents[mask])
    result: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor, int]]] = {}
    for expert in experts:
        result[expert] = {}
        for population in populations:
            if not row_parts[expert][population]:
                raise ValueError(
                    f"layer {layer} expert {expert} has no {population} rows"
                )
            documents = torch.cat(
                document_parts[expert][population], dim=0
            ).contiguous()
            result[expert][population] = (
                torch.cat(row_parts[expert][population], dim=0).contiguous(),
                torch.cat(weight_parts[expert][population], dim=0)
                .float()
                .contiguous(),
                int(torch.unique(documents).numel()),
            )
    return result


def _down_rows(
    inputs: torch.Tensor,
    anchor: CoupledTriplet,
    *,
    device: torch.device,
    batch_rows: int,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for begin in range(0, inputs.shape[0], batch_rows):
        x = inputs[begin : begin + batch_rows].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        parts.append(expert_hidden(x, anchor).to(torch.bfloat16).cpu())
    return torch.cat(parts, dim=0).contiguous()


def _factor_tensors(
    output: dict[str, torch.Tensor],
    *,
    expert: int,
    matrix: str,
    variant: str,
    rank: int,
    fit: LowRankAdapterFit,
) -> None:
    prefix = f"experts.{expert}.{matrix}.{variant}.rank_{rank}"
    output[f"{prefix}.a"] = fit.a.to(torch.bfloat16).cpu().contiguous()
    output[f"{prefix}.b"] = fit.b.to(torch.bfloat16).cpu().contiguous()


def _score_fits(
    error: torch.Tensor,
    rows: torch.Tensor,
    weights: torch.Tensor,
    fits: Mapping[tuple[str, int], LowRankAdapterFit],
    *,
    batch_rows: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Cross-score every factor under coefficient and routed-input loss."""

    frobenius_total = error.double().square().sum()
    frobenius_residual: dict[tuple[str, int], float] = {}
    for key, fit in fits.items():
        correction = fit.b.float() @ fit.a.float().T
        frobenius_residual[key] = float(
            (error.float() - correction).double().square().sum().item()
        )

    weight_square_sum = float(weights.double().square().sum().item())
    weighted_total = 0.0
    weighted_residual = {key: 0.0 for key in fits}
    for begin in range(0, rows.shape[0], batch_rows):
        end = min(begin + batch_rows, rows.shape[0])
        x = rows[begin:end].to(
            device=error.device, dtype=torch.float32, non_blocking=True
        )
        w2 = weights[begin:end].to(
            device=error.device, dtype=torch.float32, non_blocking=True
        ).square()
        target = x @ error.float().T
        weighted_total += float(
            (target.square().sum(dim=1) * w2).double().sum().item()
        )
        for key, fit in fits.items():
            predicted = (x @ fit.a.float()) @ fit.b.float().T
            weighted_residual[key] += float(
                ((target - predicted).square().sum(dim=1) * w2)
                .double()
                .sum()
                .item()
            )
    weighted_total /= weight_square_sum
    for key in weighted_residual:
        weighted_residual[key] /= weight_square_sum

    result: dict[str, dict[str, dict[str, float]]] = {
        "plain": {},
        "weighted": {},
    }
    for (variant, rank), _fit in fits.items():
        frobenius_fraction = 1.0 - (
            frobenius_residual[(variant, rank)] / float(frobenius_total.item())
        )
        weighted_fraction = 1.0 - (
            weighted_residual[(variant, rank)] / weighted_total
        )
        result[variant][str(rank)] = {
            "frobenius_captured_fraction": frobenius_fraction,
            "weighted_captured_fraction": weighted_fraction,
        }
    return result


def main() -> None:
    args = _parser().parse_args()
    experts = _parse_unique_ints(args.experts, name="--experts")
    ranks = _parse_unique_ints(args.ranks, name="--ranks")
    if args.layer not in C.MOE_LAYERS:
        raise ValueError("--layer must identify a routed decoder layer")
    if any(not 0 <= expert < C.NUM_EXPERTS for expert in experts):
        raise ValueError("--experts contains an out-of-range expert")
    if args.rank != max(ranks) or min(ranks) < 1:
        raise ValueError("--rank must equal the largest reported rank")
    if args.validation_document_modulus < 2:
        raise ValueError("--validation-document-modulus must be at least two")
    if not 0 <= args.validation_document_residue < args.validation_document_modulus:
        raise ValueError(
            "--validation-document-residue must be inside the document modulus"
        )
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("--device must be an indexed CUDA device")
    if args.dest.exists():
        raise FileExistsError(args.dest)
    args.dest.mkdir(parents=True)
    overlay_roots = tuple(
        args.overlay_root
        or (DEFAULT_W2_OVERLAY, DEFAULT_UPSTREAM_OVERLAY)
    )
    started = time.monotonic()
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    print("collecting routed gate/up inputs", flush=True)
    upstream_rows = _collect_upstream_rows(
        args.routed_rows,
        layer=args.layer,
        experts=experts,
        validation_document_modulus=args.validation_document_modulus,
        validation_document_residue=args.validation_document_residue,
    )
    draws = _profile_draws(args.model, args.layer)
    payload = QSRTAnchorPayload(
        args.candidate_pool,
        overlay_roots=overlay_roots,
    )
    base, overlays = payload.layer_paths(args.layer)
    source_store = OfficialMXFP4Store()
    factor_tensors: dict[str, torch.Tensor] = {}
    records: list[dict[str, object]] = []

    with (
        source_store.open_layer(args.layer, experts=experts) as source_layer,
        CompositeCandidateTensorReader(base, overlays) as reader,
    ):
        for ordinal, expert in enumerate(experts):
            expert_started = time.monotonic()
            source = CoupledTriplet(
                *(
                    source_layer.load_matrix(
                        args.layer, expert, matrix, device=device
                    ).float()
                    for matrix in C.EXPERT_MATRICES
                )
            )
            stored = tuple(
                decode_candidate_matrix(
                    reader,
                    layer=args.layer,
                    expert=expert,
                    matrix=matrix,
                    mode_id=K2.mode_id,
                    device=device,
                )
                .T.contiguous()
                for matrix in C.EXPERT_MATRICES
            )
            anchor = CoupledTriplet(
                *decode_coupled_weights(
                    stored,
                    CoupledHadamardSpec(intermediate_draw=draws[expert]),
                )
            ).to(dtype=torch.float32)
            fit_upstream_rows, fit_weights, fit_documents = upstream_rows[expert][
                "fit"
            ]
            validation_upstream_rows, validation_weights, validation_documents = (
                upstream_rows[expert]["validation"]
            )
            fit_down_rows = _down_rows(
                fit_upstream_rows,
                anchor,
                device=device,
                batch_rows=args.batch_rows,
            )
            validation_down_rows = _down_rows(
                validation_upstream_rows,
                anchor,
                device=device,
                batch_rows=args.batch_rows,
            )
            weighted_populations = {
                "w1": {
                    "fit": (fit_upstream_rows, fit_weights, fit_documents),
                    "validation": (
                        validation_upstream_rows,
                        validation_weights,
                        validation_documents,
                    ),
                },
                "w3": {
                    "fit": (fit_upstream_rows, fit_weights, fit_documents),
                    "validation": (
                        validation_upstream_rows,
                        validation_weights,
                        validation_documents,
                    ),
                },
                "w2": {
                    "fit": (fit_down_rows, fit_weights, fit_documents),
                    "validation": (
                        validation_down_rows,
                        validation_weights,
                        validation_documents,
                    ),
                },
            }
            matrix_records: dict[str, object] = {}
            for matrix, source_matrix, anchor_matrix in zip(
                C.EXPERT_MATRICES,
                source.tensors(),
                anchor.tensors(),
                strict=True,
            ):
                error = (source_matrix - anchor_matrix).contiguous()
                rows, weights, fit_document_count = weighted_populations[matrix][
                    "fit"
                ]
                validation_rows, validation_weights, validation_document_count = (
                    weighted_populations[matrix]["validation"]
                )
                fits: dict[tuple[str, int], LowRankAdapterFit] = {}
                for rank in ranks:
                    fits[("plain", rank)] = fit_plain_error_adapter(
                        error,
                        rank=rank,
                        oversampling=args.oversampling,
                        power_iterations=args.power_iterations,
                    )
                    fits[("weighted", rank)] = fit_weighted_error_adapter(
                        error,
                        rows,
                        weights,
                        rank=rank,
                        oversampling=args.oversampling,
                        power_iterations=args.power_iterations,
                        batch_rows=args.batch_rows,
                        seed=(
                            args.seed
                            + expert * 17
                            + C.EXPERT_MATRICES.index(matrix) * 5
                            + rank
                        ),
                    )
                for (variant, rank), fit in fits.items():
                    _factor_tensors(
                        factor_tensors,
                        expert=expert,
                        matrix=matrix,
                        variant=variant,
                        rank=rank,
                        fit=fit,
                    )
                matrix_records[matrix] = {
                    "fit_input_rows": int(rows.shape[0]),
                    "fit_documents": fit_document_count,
                    "fit_route_square_mass": float(
                        weights.double().square().sum()
                    ),
                    "validation_input_rows": int(validation_rows.shape[0]),
                    "validation_documents": validation_document_count,
                    "validation_route_square_mass": float(
                        validation_weights.double().square().sum()
                    ),
                    "fit_curves": _score_fits(
                        error,
                        rows,
                        weights,
                        fits,
                        batch_rows=args.batch_rows,
                    ),
                    "validation_curves": _score_fits(
                        error,
                        validation_rows,
                        validation_weights,
                        fits,
                        batch_rows=args.batch_rows,
                    ),
                }
                del error, fits
            records.append(
                {
                    "expert": expert,
                    "coupled_hadamard_draw": draws[expert],
                    "seconds": time.monotonic() - expert_started,
                    "matrices": matrix_records,
                }
            )
            print(
                json.dumps(
                    {
                        "layer": args.layer,
                        "expert": expert,
                        "completed": ordinal + 1,
                        "experts": len(experts),
                        "seconds": records[-1]["seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del source, stored, anchor, fit_down_rows, validation_down_rows
            torch.cuda.empty_cache()

    factor_path = args.dest / "factors.safetensors"
    temporary = factor_path.with_name(f".{factor_path.name}.{os.getpid()}.tmp")
    save_file(
        factor_tensors,
        temporary,
        metadata={
            "kind": "qsrt_low_rank_error_probe_factors",
            "schema_version": "3",
            "coordinate_system": "canonical_decoded_expert_gemm",
            "layer": str(args.layer),
            "maximum_rank": str(args.rank),
        },
    )
    os.replace(temporary, factor_path)
    report = {
        "kind": "qsrt_low_rank_error_probe",
        "schema_version": 3,
        "status": "research_only",
        "purpose": "bounded mechanism and throughput probe for M2 error truncation",
        "layer": args.layer,
        "experts": list(experts),
        "ranks": list(ranks),
        "coordinate_system": "canonical decoded expert GEMM [out, in]",
        "source_checkpoint": str(source_store.root),
        "anchor_model": str(args.model.resolve()),
        "candidate_pool": str(args.candidate_pool.resolve()),
        "overlay_roots": [str(path.resolve()) for path in overlay_roots],
        "routed_rows": str(args.routed_rows.resolve()),
        "routed_rows_role": "document-partitioned expert input weighting proxy captured from the 3.08-bpw checkpoint",
        "down_rows_role": "post-SiTU inputs reconstructed through decoded K2 gate/up for every retained routed occurrence",
        "validation_document_modulus": args.validation_document_modulus,
        "validation_document_residue": args.validation_document_residue,
        "maximum_rank": args.rank,
        "oversampling": args.oversampling,
        "power_iterations": args.power_iterations,
        "seed": args.seed,
        "factor_file": str(factor_path.resolve()),
        "elapsed_seconds": time.monotonic() - started,
        "records": records,
    }
    _atomic_json(args.dest / "result.json", report)
    print(
        json.dumps(
            {
                "dest": str(args.dest),
                "layer": args.layer,
                "experts": len(experts),
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

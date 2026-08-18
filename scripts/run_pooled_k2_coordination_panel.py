#!/usr/bin/env python3
"""Evaluate format-preserving gate/up coordination for uniform-K2 QSRT.

Every candidate retains the production two-bit payload, coupled activation-
boundary transform, shared T12 reconstruction staircase, and dense-H encoder.
Gate/up target adjustments are formed in ordinary expert coordinates, mapped
through the exact coupled transform, independently encoded, decoded, and then
scored through a freshly encoded candidate-specific down projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import torch
from safetensors import safe_open

from qsrt.all_row_capture import MaterializedExpertRows, materialize_expert_rows
from qsrt.capture import load_layer_hessians
from qsrt.coupled_expert_study import CoupledTriplet
from qsrt.pack.qsrt_encoder import QSRTTileMapCandidate, quantize_qsrt_tile_map_candidates
from qsrt.pooled_calibration import (
    CandidateHiddenStatistics,
    blockwise_upstream_conditioning_coefficients,
    candidate_h2,
    collect_coupled_hidden_statistics,
    collect_upstream_functional_statistics,
    decoded_down_sse,
    evaluate_coupled_expert_batches,
    ridge_refit_down_from_statistics,
)
from qsrt.qsrt_coupled import (
    CoupledHadamardExecution,
    CoupledHadamardSpec,
    decode_coupled_weights,
    encode_coupled_weights,
)
from qsrt.qsrt_atoms_v2 import unpack_atoms_v2_format_section
from qsrt.source_weights import OfficialMXFP4Store


MATRICES = ("w1", "w3", "w2")
DEFAULT_REFIT_RATIOS = (1e-5, 1e-4, 1e-3, 1e-2)


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(dict.fromkeys(int(item) for item in value.split(",") if item))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("integer list must be nonempty")
    return result


def _parse_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(dict.fromkeys(float(item) for item in value.split(",") if item))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated real numbers") from exc
    if not result or any(not math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("real-number list must be nonempty and finite")
    return result


def _parse_strings(value: str) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(item for item in value.split(",") if item))
    if not result:
        raise argparse.ArgumentTypeError("string list must be nonempty")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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


def _batches(
    rows: MaterializedExpertRows,
    expert: int,
    *,
    batch_rows: int,
    row_limit: int | None,
):
    return rows.expert_batches(
        expert,
        batch_rows=batch_rows,
        row_limit=row_limit,
        fields=("input",),
    )


def _uniform_k2_map(shape: tuple[int, int]) -> dict[str, tuple[int, ...]]:
    tiles = (shape[0] // 16) * (shape[1] // 16)
    return {"uniform_k2": (2,) * tiles}


def _encode_matrix(
    source: torch.Tensor,
    hessian: torch.Tensor,
    *,
    matrix: str,
    layer: int,
    device: torch.device,
    shared_scale_scope: object,
) -> QSRTTileMapCandidate:
    return quantize_qsrt_tile_map_candidates(
        source,
        hessian,
        torch.arange(
            source.shape[0] if matrix != "w2" else source.shape[1],
            device=device,
        ),
        matrix=matrix,
        maps=_uniform_k2_map(
            (source.shape[1], source.shape[0])
            if matrix != "w2"
            else (source.shape[1], source.shape[0])
        ),
        layer=layer,
        device=device,
        shared_scale_scope=shared_scale_scope,
        ldlq_tf32=True,
    )["uniform_k2"]


def _encode_upstream(
    ordinary_target: CoupledTriplet,
    *,
    spec: CoupledHadamardSpec,
    transformed_h13: torch.Tensor,
    layer: int,
    device: torch.device,
    shared_scale_scope: object,
) -> tuple[CoupledTriplet, tuple[QSRTTileMapCandidate, QSRTTileMapCandidate]]:
    coordinates = CoupledTriplet(*encode_coupled_weights(ordinary_target.tensors(), spec))
    gate = _encode_matrix(
        coordinates.gate,
        transformed_h13,
        matrix="w1",
        layer=layer,
        device=device,
        shared_scale_scope=shared_scale_scope,
    )
    up = _encode_matrix(
        coordinates.up,
        transformed_h13,
        matrix="w3",
        layer=layer,
        device=device,
        shared_scale_scope=shared_scale_scope,
    )
    return coordinates, (gate, up)


def _prime_shared_scales(
    source: CoupledTriplet,
    global_h13: torch.Tensor,
    *,
    layer: int,
    device: torch.device,
    shared_scale_scope: object,
) -> dict[str, object]:
    spec = CoupledHadamardSpec(intermediate_draw=0)
    execution = CoupledHadamardExecution(source.hidden, source.intermediate, spec)
    transformed_h13 = execution.transform_h13(
        global_h13.to(device=device, dtype=torch.float32)
    )
    _coordinates, candidates = _encode_upstream(
        source,
        spec=spec,
        transformed_h13=transformed_h13,
        layer=layer,
        device=device,
        shared_scale_scope=shared_scale_scope,
    )
    return {
        "source_expert": 0,
        "intermediate_draw": 0,
        "rate": 2,
        "w1_suh_sha256": hashlib.sha256(
            candidates[0].tensors["suh"].detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
        "w3_suh_sha256": hashlib.sha256(
            candidates[1].tensors["suh"].detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
    }


def _ordinary_upstream(
    pair: tuple[QSRTTileMapCandidate, QSRTTileMapCandidate],
    coordinate_down: torch.Tensor,
    spec: CoupledHadamardSpec,
) -> CoupledTriplet:
    return CoupledTriplet(
        *decode_coupled_weights(
            (pair[0].reconstruction, pair[1].reconstruction, coordinate_down),
            spec,
        )
    )


def _conditional_target(
    source: CoupledTriplet,
    reconstruction: CoupledTriplet,
    gate_from_up: torch.Tensor,
    up_from_gate: torch.Tensor,
    *,
    damping: float,
    adjust_gate: bool,
    adjust_up: bool,
    retained_gate: torch.Tensor | None = None,
    retained_up: torch.Tensor | None = None,
) -> CoupledTriplet:
    gate = source.gate if retained_gate is None else retained_gate
    up = source.up if retained_up is None else retained_up
    if adjust_gate:
        gate = source.gate + (
            damping
            * gate_from_up.to(device=source.gate.device, dtype=source.gate.dtype)[:, None]
            * (source.up - reconstruction.up)
        )
    if adjust_up:
        up = source.up + (
            damping
            * up_from_gate.to(device=source.up.device, dtype=source.up.dtype)[:, None]
            * (source.gate - reconstruction.gate)
        )
    return CoupledTriplet(gate.contiguous(), up.contiguous(), source.down)


def _upstream_statistics(
    rows: MaterializedExpertRows,
    expert: int,
    *,
    source: CoupledTriplet,
    coordinate_down: torch.Tensor,
    pair: tuple[QSRTTileMapCandidate, QSRTTileMapCandidate],
    execution: CoupledHadamardExecution,
    batch_rows: int,
    row_limit: int | None,
) -> CandidateHiddenStatistics:
    return collect_coupled_hidden_statistics(
        _batches(rows, expert, batch_rows=batch_rows, row_limit=row_limit),
        source=source,
        candidate_coordinates=CoupledTriplet(
            pair[0].reconstruction,
            pair[1].reconstruction,
            coordinate_down,
        ),
        execution=execution,
        retain_source_gram=True,
    )


def _source_energy(
    statistics: CandidateHiddenStatistics,
    source_down_t: torch.Tensor,
) -> float:
    if statistics.source_gram is None:
        raise ValueError("source energy requires retained source Gram")
    source = source_down_t.to(
        device=statistics.source_gram.device,
        dtype=statistics.source_gram.dtype,
    )
    return float(torch.sum(source * (statistics.source_gram @ source)))


def _score_down(
    statistics: CandidateHiddenStatistics,
    candidate: QSRTTileMapCandidate,
    source_down_t: torch.Tensor,
) -> float:
    return float(
        decoded_down_sse(
            statistics,
            candidate.reconstruction.T,
            source_down_t,
        )
    )


def _encode_expert(
    *,
    rows: MaterializedExpertRows,
    source: CoupledTriplet,
    global_h13: torch.Tensor,
    layer: int,
    expert: int,
    draw: int,
    device: torch.device,
    batch_rows: int,
    row_limit: int | None,
    shared_scale_scope: object,
    block_sizes: tuple[int, ...],
    dampings: tuple[float, ...],
    strategies: tuple[str, ...],
    refit_ratios: tuple[float, ...],
    maximum_upstream_candidates: int,
) -> dict[str, object]:
    spec = CoupledHadamardSpec(intermediate_draw=draw)
    execution = CoupledHadamardExecution(source.hidden, source.intermediate, spec)
    transformed_h13 = execution.transform_h13(
        global_h13.to(device=device, dtype=torch.float32)
    )
    coordinate_source, baseline_pair = _encode_upstream(
        source,
        spec=spec,
        transformed_h13=transformed_h13,
        layer=layer,
        device=device,
        shared_scale_scope=shared_scale_scope,
    )
    baseline_ordinary = _ordinary_upstream(baseline_pair, coordinate_source.down, spec)
    coordinate_strategies = set(strategies) - {"baseline"}
    functional = (
        collect_upstream_functional_statistics(
            _batches(rows, expert, batch_rows=batch_rows, row_limit=row_limit),
            source=source,
            device=device,
        )
        if coordinate_strategies
        else None
    )

    upstream: dict[str, tuple[QSRTTileMapCandidate, QSRTTileMapCandidate]] = {
        "baseline": baseline_pair
    }
    coefficient_evidence: dict[str, object] = {}
    for block_size in block_sizes if functional is not None else ():
        gate_from_up, up_from_gate, evidence = (
            blockwise_upstream_conditioning_coefficients(
                functional.derivative_metric,
                block_size=block_size,
            )
        )
        coefficient_evidence[str(block_size)] = evidence
        for damping in dampings:
            suffix = f"b{block_size}-d{damping:g}"
            if "simultaneous" in coordinate_strategies:
                target = _conditional_target(
                    source,
                    baseline_ordinary,
                    gate_from_up,
                    up_from_gate,
                    damping=damping,
                    adjust_gate=True,
                    adjust_up=True,
                )
                _coordinates, upstream[f"simultaneous-{suffix}"] = _encode_upstream(
                    target,
                    spec=spec,
                    transformed_h13=transformed_h13,
                    layer=layer,
                    device=device,
                    shared_scale_scope=shared_scale_scope,
                )
            if "gate_then_up" in coordinate_strategies:
                gate_target = _conditional_target(
                    source,
                    baseline_ordinary,
                    gate_from_up,
                    up_from_gate,
                    damping=damping,
                    adjust_gate=True,
                    adjust_up=False,
                )
                _coordinates, interim_pair = _encode_upstream(
                    gate_target,
                    spec=spec,
                    transformed_h13=transformed_h13,
                    layer=layer,
                    device=device,
                    shared_scale_scope=shared_scale_scope,
                )
                interim = _ordinary_upstream(interim_pair, coordinate_source.down, spec)
                final_target = _conditional_target(
                    source,
                    interim,
                    gate_from_up,
                    up_from_gate,
                    damping=damping,
                    adjust_gate=False,
                    adjust_up=True,
                    retained_gate=gate_target.gate,
                )
                _coordinates, upstream[f"gate_then_up-{suffix}"] = _encode_upstream(
                    final_target,
                    spec=spec,
                    transformed_h13=transformed_h13,
                    layer=layer,
                    device=device,
                    shared_scale_scope=shared_scale_scope,
                )
            if "up_then_gate" in coordinate_strategies:
                up_target = _conditional_target(
                    source,
                    baseline_ordinary,
                    gate_from_up,
                    up_from_gate,
                    damping=damping,
                    adjust_gate=False,
                    adjust_up=True,
                )
                _coordinates, interim_pair = _encode_upstream(
                    up_target,
                    spec=spec,
                    transformed_h13=transformed_h13,
                    layer=layer,
                    device=device,
                    shared_scale_scope=shared_scale_scope,
                )
                interim = _ordinary_upstream(interim_pair, coordinate_source.down, spec)
                final_target = _conditional_target(
                    source,
                    interim,
                    gate_from_up,
                    up_from_gate,
                    damping=damping,
                    adjust_gate=True,
                    adjust_up=False,
                    retained_up=up_target.up,
                )
                _coordinates, upstream[f"up_then_gate-{suffix}"] = _encode_upstream(
                    final_target,
                    spec=spec,
                    transformed_h13=transformed_h13,
                    layer=layer,
                    device=device,
                    shared_scale_scope=shared_scale_scope,
                )

    source_down_t = coordinate_source.down.T
    upstream_statistics: dict[str, CandidateHiddenStatistics] = {}
    upstream_source_w2_sse: dict[str, float] = {}
    for name, pair in upstream.items():
        statistics = _upstream_statistics(
            rows,
            expert,
            source=source,
            coordinate_down=coordinate_source.down,
            pair=pair,
            execution=execution,
            batch_rows=batch_rows,
            row_limit=row_limit,
        )
        upstream_statistics[name] = statistics
        upstream_source_w2_sse[name] = float(
            decoded_down_sse(statistics, source_down_t, source_down_t)
        )
    selected = sorted(
        upstream,
        key=lambda name: (upstream_source_w2_sse[name], name),
    )[:maximum_upstream_candidates]
    if "baseline" not in selected:
        selected.append("baseline")

    combinations: list[dict[str, object]] = []
    retained: dict[tuple[str, str], QSRTTileMapCandidate] = {}
    for upstream_name in selected:
        statistics = upstream_statistics[upstream_name]
        h2, h2_evidence = candidate_h2(statistics)
        energy = _source_energy(statistics, source_down_t)
        source_w2 = _encode_matrix(
            coordinate_source.down,
            h2,
            matrix="w2",
            layer=layer,
            device=device,
            shared_scale_scope=shared_scale_scope,
        )
        retained[(upstream_name, "source_w2")] = source_w2
        source_sse = _score_down(statistics, source_w2, source_down_t)
        combinations.append(
            {
                "upstream": upstream_name,
                "down_target": "source_w2",
                "sse": source_sse,
                "source_energy": energy,
                "nmse": source_sse / energy,
                "candidate_h2": h2_evidence,
            }
        )
        for ratio in refit_ratios:
            refit_t, evidence = ridge_refit_down_from_statistics(
                statistics,
                source_down_t,
                regularization_ratio=ratio,
            )
            name = f"w2_refit_{ratio:.0e}"
            encoded = _encode_matrix(
                refit_t.T.float().contiguous(),
                h2,
                matrix="w2",
                layer=layer,
                device=device,
                shared_scale_scope=shared_scale_scope,
            )
            retained[(upstream_name, name)] = encoded
            sse = _score_down(statistics, encoded, source_down_t)
            combinations.append(
                {
                    "upstream": upstream_name,
                    "down_target": name,
                    "sse": sse,
                    "source_energy": energy,
                    "nmse": sse / energy,
                    "refit": evidence,
                    "candidate_h2": h2_evidence,
                }
            )

    winner = min(combinations, key=lambda value: (float(value["sse"]), str(value)))
    winner_upstream = str(winner["upstream"])
    winner_down_target = str(winner["down_target"])
    winner_down = retained[(winner_upstream, winner_down_target)]
    explicit = evaluate_coupled_expert_batches(
        _batches(rows, expert, batch_rows=batch_rows, row_limit=row_limit),
        source=source,
        candidate_coordinates=CoupledTriplet(
            upstream[winner_upstream][0].reconstruction,
            upstream[winner_upstream][1].reconstruction,
            winner_down.reconstruction,
        ),
        execution=execution,
    )
    quadratic_sse = float(winner["sse"])
    relative_closure = abs(explicit.sse - quadratic_sse) / max(explicit.sse, 1e-30)
    if relative_closure > 2e-4:
        raise ArithmeticError(
            "explicit and sufficient-statistic whole-expert SSE do not close: "
            f"{explicit.sse} versus {quadratic_sse}"
        )
    return {
        "draw": draw,
        "routed_occurrences": explicit.routed_occurrences,
        "functional_metric": (
            {
                "rows": functional.rows,
                "effective_sample_size": functional.effective_sample_size,
                "coefficient_evidence": coefficient_evidence,
            }
            if functional is not None
            else {"status": "not required for baseline-upstream W2 refit"}
        ),
        "upstream_source_w2_sse": upstream_source_w2_sse,
        "upstream_candidates_carried_to_w2": selected,
        "combinations": combinations,
        "winner": winner,
        "winner_explicit_score": {
            "sse": explicit.sse,
            "source_energy": explicit.source_energy,
            "nmse": explicit.nmse,
            "quadratic_sse": quadratic_sse,
            "relative_sse_closure": relative_closure,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--hessians", type=Path, required=True)
    parser.add_argument("--production-model", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--block-sizes", type=_parse_ints, default=(16, 128))
    parser.add_argument("--dampings", type=_parse_floats, default=(0.5, 1.0))
    parser.add_argument(
        "--strategies",
        type=_parse_strings,
        default=("simultaneous", "gate_then_up", "up_then_gate"),
    )
    parser.add_argument(
        "--w2-refit-ratios",
        type=_parse_floats,
        default=DEFAULT_REFIT_RATIOS,
    )
    parser.add_argument("--maximum-upstream-candidates", type=int, default=3)
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument("--official-revision")
    parser.add_argument("--verify-capture-hashes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.layer <= 92 or not 0 <= args.expert < 896:
        parser.error("layer must lie in 1..92 and expert in 0..895")
    if args.batch_rows <= 0 or args.maximum_upstream_candidates <= 0:
        parser.error("batch sizes and candidate count must be positive")
    if any(size <= 0 or 3072 % size for size in args.block_sizes):
        parser.error("every block size must divide 3072")
    if any(value <= 0 or value > 1 for value in args.dampings):
        parser.error("damping values must lie in (0, 1]")
    allowed_strategies = {"baseline", "simultaneous", "gate_then_up", "up_then_gate"}
    if set(args.strategies) - allowed_strategies:
        parser.error("unsupported coordination strategy")
    if any(value <= 0 for value in args.w2_refit_ratios):
        parser.error("W2 refit ratios must be positive")

    capture_manifest_path = args.capture / "manifest.json"
    hessian_manifest_path = args.hessians / "manifest.json"
    model_completion_path = args.production_model / "qsrt-completion.json"
    capture_manifest = json.loads(capture_manifest_path.read_text())
    if not bool(capture_manifest.get("complete", False)):
        parser.error("capture must be finalized")
    if not model_completion_path.is_file():
        parser.error("production model lacks qsrt-completion.json")
    layer_path = args.production_model / f"qsrt-layer-{args.layer:05d}.safetensors"
    with safe_open(layer_path, framework="pt", device="cpu") as reader:
        metadata = reader.metadata()
        if metadata is None or "profile" not in metadata:
            parser.error("production layer lacks the atoms-v2 profile identity")
        formats, draws = unpack_atoms_v2_format_section(
            str(metadata["profile"]),
            reader.get_tensor("_qsrt_format_section"),
        )
        if formats[args.expert] != "K2" or draws is None:
            parser.error("production expert is not a coupled uniform-K2 profile")
        draw = int(draws[args.expert])

    signature = {
        "kind": "qsrt_pooled_uniform_k2_coordination_panel",
        "schema_version": 1,
        "capture": str(args.capture.resolve()),
        "capture_manifest_sha256": _sha256(capture_manifest_path),
        "hessians": str(args.hessians.resolve()),
        "hessian_manifest_sha256": _sha256(hessian_manifest_path),
        "production_model": str(args.production_model.resolve()),
        "production_completion_sha256": _sha256(model_completion_path),
        "layer": args.layer,
        "expert": args.expert,
        "intermediate_draw": draw,
        "row_limit": args.row_limit,
        "batch_rows": args.batch_rows,
        "block_sizes": list(args.block_sizes),
        "dampings": list(args.dampings),
        "strategies": list(args.strategies),
        "w2_refit_ratios": list(args.w2_refit_ratios),
        "maximum_upstream_candidates": args.maximum_upstream_candidates,
        "profile": "uniform K2 coupled H512/H128 with SQG XOR T12",
        "selection_population": "all naturally routed rows; no document partition",
        "acceptance_metric": "decoded route-weighted whole-expert SSE",
        "capture_hashes_verified_during_run": args.verify_capture_hashes,
    }
    if args.output.exists():
        receipt = json.loads(args.output.read_text())
        if receipt.get("signature") != signature:
            parser.error("output receipt belongs to another experiment")
        if bool(receipt.get("complete", False)):
            print(json.dumps({"complete": str(args.output)}, sort_keys=True))
            return
    receipt: dict[str, object] = {"signature": signature, "complete": False}
    _atomic_json(args.output, receipt)

    rows = materialize_expert_rows(
        args.capture,
        args.layer,
        args.expert,
        fields=("input",),
        verify_hashes=args.verify_capture_hashes,
    )
    if args.row_limit is not None and not 0 < args.row_limit <= rows.rows:
        parser.error("row limit is outside the capture")
    global_h13, _ = load_layer_hessians(args.hessians, args.layer)
    device = torch.device(args.device)
    store_kwargs: dict[str, object] = {}
    if args.official_repo_dir is not None:
        store_kwargs["repo_dir"] = args.official_repo_dir
    if args.official_revision is not None:
        store_kwargs["revision"] = args.official_revision
    store = OfficialMXFP4Store(**store_kwargs)
    shared_scale_scope = (
        "pooled_uniform_k2_coordination",
        signature["capture_manifest_sha256"],
        args.layer,
    )
    source_experts = tuple(sorted({0, args.expert}))
    with store.open_layer(args.layer, experts=source_experts) as layer_store:
        primer_source = _source_triplet(
            layer_store,
            layer=args.layer,
            expert=0,
            device=device,
        )
        receipt["shared_scale_evidence"] = _prime_shared_scales(
            primer_source,
            global_h13,
            layer=args.layer,
            device=device,
            shared_scale_scope=shared_scale_scope,
        )
        source = _source_triplet(
            layer_store,
            layer=args.layer,
            expert=args.expert,
            device=device,
        )
        receipt["result"] = _encode_expert(
            rows=rows,
            source=source,
            global_h13=global_h13,
            layer=args.layer,
            expert=args.expert,
            draw=draw,
            device=device,
            batch_rows=args.batch_rows,
            row_limit=args.row_limit,
            shared_scale_scope=shared_scale_scope,
            block_sizes=args.block_sizes,
            dampings=args.dampings,
            strategies=args.strategies,
            refit_ratios=args.w2_refit_ratios,
            maximum_upstream_candidates=args.maximum_upstream_candidates,
        )
    receipt["complete"] = True
    _atomic_json(args.output, receipt)
    print(
        json.dumps(
            {
                "complete": str(args.output),
                "winner": receipt["result"]["winner"],
                "explicit": receipt["result"]["winner_explicit_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

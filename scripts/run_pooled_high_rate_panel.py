#!/usr/bin/env python3
"""Compare equal-byte 3.083-bpw QSRT allocations on all routed rows.

The script treats regularized-basis tile error only as a proposal mechanism.
Every reported candidate is re-encoded with its complete rate map and scored
after decoding the coupled expert function over the pooled routed population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import torch

from qsrt.all_row_capture import (
    MaterializedExpertRows,
    MaterializedLayerRows,
    materialize_expert_rows,
    materialize_layer_rows,
)
from qsrt.capture import load_layer_hessians
from qsrt.coupled_expert_study import (
    CoupledTriplet,
    apply_permutation_sign_gauge,
)
from qsrt.exl3_reference import qsrt_regularized_target
from qsrt.high_rate_allocation import (
    dense_h_tile_error_contributions,
    neuron_permutation_from_scores,
    record_rate_map,
    select_record_rate_allocation,
    tile_p24_allocation,
)
from qsrt.pack.qsrt_encoder import (
    QSRTTileMapCandidate,
    quantize_qsrt_tile_map_candidates,
)
from qsrt.pooled_calibration import (
    CandidateHiddenStatistics,
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
    encode_coupled_weights,
)
from qsrt.source_weights import OfficialMXFP4Store


RATES = (2, 3, 4)
DEFAULT_PREFIX_ROWS = (1_000_000, 2_000_000, 4_000_000)


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(dict.fromkeys(int(item) for item in value.split(",") if item))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("integer list must be nonempty")
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


def _rate_map_sha256(bits: Iterable[int]) -> str:
    return hashlib.sha256(bytes(int(value) for value in bits)).hexdigest()


def _integer_vector_sha256(values: Iterable[int]) -> str:
    tensor = torch.tensor(tuple(int(value) for value in values), dtype=torch.int32)
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


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
            for matrix in ("w1", "w3", "w2")
        )
    )


def _batches(
    rows: MaterializedLayerRows | MaterializedExpertRows,
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


def _combined_group_scores(
    rows: MaterializedLayerRows | MaterializedExpertRows,
    expert: int,
    *,
    source: CoupledTriplet,
    device: torch.device,
    batch_rows: int,
    row_limit: int | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    statistics = collect_upstream_functional_statistics(
        _batches(
            rows,
            expert,
            batch_rows=batch_rows,
            row_limit=row_limit,
        ),
        source=source,
        device=device,
    )
    hidden = statistics.hidden_energy.reshape(-1, 4).sum(dim=1)
    derivative = (
        statistics.derivative_metric.diagonal(dim1=1, dim2=2)
        .sum(dim=1)
        .reshape(-1, 4)
        .sum(dim=1)
    )
    hidden = hidden / hidden.mean().clamp_min(1e-30)
    derivative = derivative / derivative.mean().clamp_min(1e-30)
    score = (hidden + derivative) * 0.5
    return score.float().cpu(), {
        "rows": statistics.rows,
        "effective_sample_size": statistics.effective_sample_size,
        "minimum_score": float(score.min()),
        "maximum_score": float(score.max()),
    }


def _uniform_maps(shape: tuple[int, int]) -> dict[str, tuple[int, ...]]:
    tiles = (shape[0] // 16) * (shape[1] // 16)
    return {f"k{bits}": (bits,) * tiles for bits in RATES}


def _regularized_cost_surfaces(
    encoder_source: torch.Tensor,
    candidates: Mapping[str, QSRTTileMapCandidate],
    hessian: torch.Tensor,
) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    for bits in RATES:
        candidate = candidates[f"k{bits}"]
        target = qsrt_regularized_target(
            encoder_source,
            candidate.tensors["suh"],
            candidate.tensors["svh"],
        )
        result[bits] = dense_h_tile_error_contributions(
            target,
            candidate.regularized,
            hessian,
            candidate.tensors["suh"],
            candidate.tensors["svh"],
        )
    return result


def _proposal_maps(
    cost_surfaces: tuple[Mapping[int, torch.Tensor], ...],
    *,
    encoder_shape: tuple[int, int],
    rate_axis: str,
) -> tuple[dict[str, tuple[int, ...]], dict[str, object]]:
    fixed = record_rate_map(encoder_shape, rate_axis=rate_axis, donor_records=0)
    record = select_record_rate_allocation(
        cost_surfaces,
        shape=encoder_shape,
        rate_axis=rate_axis,
    )
    tile = tile_p24_allocation(cost_surfaces, rate_axis=rate_axis)
    proposed = {
        "fixed_k3x22_k4x2": fixed,
        f"record_donors_{record.donor_records}": record.tile_bits,
        "tile_p33_p24": tile.tile_bits,
    }
    deduplicated: dict[str, tuple[int, ...]] = {}
    aliases: dict[str, str] = {}
    by_bits: dict[tuple[int, ...], str] = {}
    for name, bits in proposed.items():
        canonical = by_bits.setdefault(bits, name)
        aliases[name] = canonical
        deduplicated.setdefault(canonical, bits)
    return deduplicated, {
        "record": {
            "donor_records": record.donor_records,
            "costs_by_donor_records": list(record.costs_by_donor_records),
            "rate_map_sha256": _rate_map_sha256(record.tile_bits),
        },
        "tile": {
            "selected_p24_tiles": tile.selected_p24_tiles,
            "candidate_p24_tiles": tile.candidate_p24_tiles,
            "selected_fraction": tile.selected_fraction,
            "selector_bytes": tile.selector_bytes,
            "rate_map_sha256": _rate_map_sha256(tile.tile_bits),
        },
        "aliases": aliases,
    }


def _source_energy(
    statistics: CandidateHiddenStatistics,
    source_down_t: torch.Tensor,
) -> float:
    if statistics.source_gram is None:
        raise ValueError("source energy requires the retained source Gram matrix")
    down = source_down_t.to(
        device=statistics.source_gram.device,
        dtype=statistics.source_gram.dtype,
    )
    return float(torch.sum(down * (statistics.source_gram @ down)))


def _score_down_candidates(
    statistics: CandidateHiddenStatistics,
    candidates: Mapping[str, QSRTTileMapCandidate],
    source_down_t: torch.Tensor,
) -> dict[str, float]:
    return {
        name: float(
            decoded_down_sse(
                statistics,
                candidate.reconstruction.T,
                source_down_t,
            )
        )
        for name, candidate in candidates.items()
    }


def _candidate_coordinates(
    upstream: tuple[QSRTTileMapCandidate, QSRTTileMapCandidate],
    down: QSRTTileMapCandidate,
) -> CoupledTriplet:
    return CoupledTriplet(
        upstream[0].reconstruction,
        upstream[1].reconstruction,
        down.reconstruction,
    )


def _encode_matrix_maps(
    source: torch.Tensor,
    hessian: torch.Tensor,
    *,
    matrix: str,
    maps: Mapping[str, tuple[int, ...]],
    layer: int,
    device: torch.device,
    shared_scale_scope: object,
) -> dict[str, QSRTTileMapCandidate]:
    return quantize_qsrt_tile_map_candidates(
        source,
        hessian,
        torch.arange(source.shape[0] if matrix != "w2" else source.shape[1], device=device),
        matrix=matrix,
        maps=maps,
        layer=layer,
        device=device,
        shared_scale_scope=shared_scale_scope,
        ldlq_tf32=True,
    )


def _seed_layer_shared_scales(
    source: CoupledTriplet,
    global_h13: torch.Tensor,
    *,
    layer: int,
    device: torch.device,
    shared_scale_scope: object,
) -> dict[str, object]:
    """Derive the representable layer/matrix-shared upstream scale profile."""

    execution = CoupledHadamardExecution(
        source.hidden,
        source.intermediate,
        CoupledHadamardSpec(intermediate_draw=0),
    )
    coordinates = CoupledTriplet(*encode_coupled_weights(source.tensors(), execution.spec))
    h13 = execution.transform_h13(global_h13.to(device=device, dtype=torch.float32))
    maps = {
        "uniform_k3": (3,)
        * ((source.hidden // 16) * (source.intermediate // 16))
    }
    evidence: dict[str, object] = {
        "source_expert": 0,
        "intermediate_draw": 0,
        "rate": 3,
        "matrices": {},
    }
    matrices = evidence["matrices"]
    assert isinstance(matrices, dict)
    for matrix, matrix_source in (("w1", coordinates.gate), ("w3", coordinates.up)):
        encoded = _encode_matrix_maps(
            matrix_source,
            h13,
            matrix=matrix,
            maps=maps,
            layer=layer,
            device=device,
            shared_scale_scope=shared_scale_scope,
        )["uniform_k3"]
        matrices[matrix] = {
            "suh_sha256": _tensor_sha256(encoded.tensors["suh"]),
            "suh_shape": list(encoded.tensors["suh"].shape),
        }
    return evidence


def _encode_arm(
    *,
    layer_rows: MaterializedLayerRows | MaterializedExpertRows,
    source: CoupledTriplet,
    layer: int,
    expert: int,
    draw: int,
    policy: str,
    group_scores: torch.Tensor,
    global_h13: torch.Tensor,
    device: torch.device,
    batch_rows: int,
    row_limit: int | None,
    evaluation_row_limit: int | None,
    shared_scale_scope: object,
    maximum_upstream_candidates: int,
    refit_ratios: tuple[float, ...],
) -> dict[str, object]:
    permutation = neuron_permutation_from_scores(group_scores, policy=policy).to(device)
    gauged = apply_permutation_sign_gauge(
        source,
        permutation,
        torch.ones(source.intermediate, device=device),
    )
    execution = CoupledHadamardExecution(
        source.hidden,
        source.intermediate,
        CoupledHadamardSpec(intermediate_draw=draw),
    )
    coordinate_source = CoupledTriplet(
        *encode_coupled_weights(gauged.tensors(), execution.spec)
    )
    h13 = execution.transform_h13(global_h13.to(device=device, dtype=torch.float32))
    upstream_uniform: dict[str, dict[str, QSRTTileMapCandidate]] = {}
    upstream_costs: dict[str, dict[int, torch.Tensor]] = {}
    for matrix, matrix_source in (("w1", coordinate_source.gate), ("w3", coordinate_source.up)):
        encoder_shape = (matrix_source.shape[1], matrix_source.shape[0])
        uniform = _encode_matrix_maps(
            matrix_source,
            h13,
            matrix=matrix,
            maps=_uniform_maps(encoder_shape),
            layer=layer,
            device=device,
            shared_scale_scope=shared_scale_scope,
        )
        upstream_uniform[matrix] = uniform
        upstream_costs[matrix] = _regularized_cost_surfaces(
            matrix_source.T,
            uniform,
            h13,
        )
    upstream_maps, upstream_proposal = _proposal_maps(
        (upstream_costs["w1"], upstream_costs["w3"]),
        encoder_shape=(source.hidden, source.intermediate),
        rate_axis="n",
    )
    del upstream_uniform
    upstream_encoded = {
        matrix: _encode_matrix_maps(
            matrix_source,
            h13,
            matrix=matrix,
            maps=upstream_maps,
            layer=layer,
            device=device,
            shared_scale_scope=shared_scale_scope,
        )
        for matrix, matrix_source in (("w1", coordinate_source.gate), ("w3", coordinate_source.up))
    }

    upstream_statistics: dict[str, CandidateHiddenStatistics] = {}
    upstream_only_sse: dict[str, float] = {}
    source_down_t = coordinate_source.down.T
    for name in upstream_maps:
        statistics = collect_coupled_hidden_statistics(
            _batches(
                layer_rows,
                expert,
                batch_rows=batch_rows,
                row_limit=row_limit,
            ),
            source=gauged,
            candidate_coordinates=CoupledTriplet(
                upstream_encoded["w1"][name].reconstruction,
                upstream_encoded["w3"][name].reconstruction,
                coordinate_source.down,
            ),
            execution=execution,
            retain_source_gram=True,
        )
        upstream_statistics[name] = statistics
        upstream_only_sse[name] = float(
            decoded_down_sse(statistics, source_down_t, source_down_t)
        )
    upstream_order = sorted(
        upstream_maps,
        key=lambda name: (upstream_only_sse[name], name),
    )[:maximum_upstream_candidates]

    combinations: list[dict[str, object]] = []
    retained_candidates: dict[
        tuple[str, str, str],
        tuple[
            tuple[QSRTTileMapCandidate, QSRTTileMapCandidate],
            QSRTTileMapCandidate,
            CandidateHiddenStatistics,
        ],
    ] = {}
    for upstream_name in upstream_order:
        statistics = upstream_statistics[upstream_name]
        h2, h2_evidence = candidate_h2(statistics)
        uniform = _encode_matrix_maps(
            coordinate_source.down,
            h2,
            matrix="w2",
            maps=_uniform_maps((source.intermediate, source.hidden)),
            layer=layer,
            device=device,
            shared_scale_scope=shared_scale_scope,
        )
        costs = _regularized_cost_surfaces(
            coordinate_source.down.T,
            uniform,
            h2,
        )
        down_maps, down_proposal = _proposal_maps(
            (costs,),
            encoder_shape=(source.intermediate, source.hidden),
            rate_axis="k",
        )
        del uniform
        down_encoded = _encode_matrix_maps(
            coordinate_source.down,
            h2,
            matrix="w2",
            maps=down_maps,
            layer=layer,
            device=device,
            shared_scale_scope=shared_scale_scope,
        )
        scores = _score_down_candidates(statistics, down_encoded, source_down_t)
        energy = _source_energy(statistics, source_down_t)
        for down_name, sse in scores.items():
            key = (upstream_name, "source_w2", down_name)
            retained_candidates[key] = (
                (
                    upstream_encoded["w1"][upstream_name],
                    upstream_encoded["w3"][upstream_name],
                ),
                down_encoded[down_name],
                statistics,
            )
            combinations.append(
                {
                    "upstream": upstream_name,
                    "down_target": "source_w2",
                    "down": down_name,
                    "sse": sse,
                    "source_energy": energy,
                    "nmse": sse / energy,
                    "candidate_h2": h2_evidence,
                    "down_proposal": down_proposal,
                }
            )

    best = min(combinations, key=lambda value: (float(value["sse"]), str(value)))
    best_upstream = str(best["upstream"])
    best_statistics = upstream_statistics[best_upstream]
    best_h2, _ = candidate_h2(best_statistics)
    refit_receipts: list[dict[str, object]] = []
    fixed_map = {
        "fixed_k3x22_k4x2": record_rate_map(
            (source.intermediate, source.hidden),
            rate_axis="k",
            donor_records=0,
        )
    }
    for ratio in refit_ratios:
        refit_t, evidence = ridge_refit_down_from_statistics(
            best_statistics,
            source_down_t,
            regularization_ratio=ratio,
        )
        refit_source = refit_t.T.float().contiguous()
        encoded = _encode_matrix_maps(
            refit_source,
            best_h2,
            matrix="w2",
            maps=fixed_map,
            layer=layer,
            device=device,
            shared_scale_scope=shared_scale_scope,
        )
        name = "fixed_k3x22_k4x2"
        sse = float(
            decoded_down_sse(
                best_statistics,
                encoded[name].reconstruction.T,
                source_down_t,
            )
        )
        energy = _source_energy(best_statistics, source_down_t)
        key = (best_upstream, f"w2_refit_{ratio:.0e}", name)
        retained_candidates[key] = (
            (
                upstream_encoded["w1"][best_upstream],
                upstream_encoded["w3"][best_upstream],
            ),
            encoded[name],
            best_statistics,
        )
        combinations.append(
            {
                "upstream": best_upstream,
                "down_target": f"w2_refit_{ratio:.0e}",
                "down": name,
                "sse": sse,
                "source_energy": energy,
                "nmse": sse / energy,
                "refit": evidence,
            }
        )
        refit_receipts.append({**evidence, "encoded_sse": sse, "encoded_nmse": sse / energy})

    winner = min(combinations, key=lambda value: (float(value["sse"]), str(value)))
    winner_key = (
        str(winner["upstream"]),
        str(winner["down_target"]),
        str(winner["down"]),
    )
    winner_upstream, winner_down, _ = retained_candidates[winner_key]
    evaluated = evaluate_coupled_expert_batches(
        _batches(
            layer_rows,
            expert,
            batch_rows=batch_rows,
            row_limit=evaluation_row_limit,
        ),
        source=gauged,
        teacher_source=source,
        candidate_coordinates=_candidate_coordinates(winner_upstream, winner_down),
        execution=execution,
        prefix_row_limits=(
            limit
            for limit in DEFAULT_PREFIX_ROWS
            if limit
            <= (
                layer_rows.rows
                if evaluation_row_limit is None
                else evaluation_row_limit
            )
        ),
    )
    return {
        "draw": draw,
        "permutation_policy": policy,
        "permutation_sha256": _integer_vector_sha256(permutation.cpu().tolist()),
        "upstream_proposal": upstream_proposal,
        "upstream_only_sse": upstream_only_sse,
        "upstream_candidates_carried_to_w2": upstream_order,
        "combinations": combinations,
        "w2_refit_candidates": refit_receipts,
        "winner": winner,
        "winner_all_row_score": {
            "sse": evaluated.sse,
            "source_energy": evaluated.source_energy,
            "nmse": evaluated.nmse,
            "routed_occurrences": evaluated.routed_occurrences,
            "prefix_scores": {
                str(limit): {
                    "sse": values[0],
                    "source_energy": values[1],
                    "routed_occurrences": values[2],
                    "nmse": values[0] / values[1] if values[1] > 0 else None,
                }
                for limit, values in evaluated.prefix_scores.items()
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--hessians", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--experts", type=_parse_ints, required=True)
    parser.add_argument("--draws", type=_parse_ints, default=(0,))
    parser.add_argument(
        "--policies",
        type=_parse_strings,
        default=("identity", "importance", "energy_balanced", "stratified_energy_balanced"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--evaluation-row-limit", type=int)
    parser.add_argument("--maximum-upstream-candidates", type=int, default=2)
    parser.add_argument("--w2-refit-ratios", type=str, default="1e-5,1e-4,1e-3,1e-2")
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument("--official-revision")
    parser.add_argument("--verify-capture-hashes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.layer <= 92:
        parser.error("layer must lie in 1..92")
    if any(not 0 <= expert < 896 for expert in args.experts):
        parser.error("experts must lie in 0..895")
    if any(not 0 <= draw < 8 for draw in args.draws):
        parser.error("draws must lie in 0..7")
    allowed_policies = {
        "identity",
        "importance",
        "energy_balanced",
        "stratified_energy_balanced",
    }
    if set(args.policies) - allowed_policies:
        parser.error("unsupported neuron permutation policy")
    if args.batch_rows <= 0 or args.maximum_upstream_candidates <= 0:
        parser.error("batch sizes and candidate counts must be positive")
    try:
        refit_ratios = tuple(float(value) for value in args.w2_refit_ratios.split(",") if value)
    except ValueError as exc:
        parser.error(f"invalid W2 refit ratio: {exc}")
    if any(not math.isfinite(value) or value <= 0 for value in refit_ratios):
        parser.error("W2 refit ratios must be positive and finite")

    capture_manifest_path = args.capture / "manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text())
    if not bool(capture_manifest.get("complete", False)):
        parser.error("capture must be finalized")
    hessian_manifest_path = args.hessians / "manifest.json"
    signature = {
        "kind": "qsrt_pooled_high_rate_panel",
        "schema_version": 1,
        "capture": str(args.capture.resolve()),
        "capture_manifest_sha256": _sha256(capture_manifest_path),
        "hessians": str(args.hessians.resolve()),
        "hessian_manifest_sha256": _sha256(hessian_manifest_path),
        "layer": args.layer,
        "experts": list(args.experts),
        "draws": list(args.draws),
        "permutation_policies": list(args.policies),
        "row_limit": args.row_limit,
        "evaluation_row_limit": args.evaluation_row_limit,
        "batch_rows": args.batch_rows,
        "maximum_upstream_candidates": args.maximum_upstream_candidates,
        "w2_refit_ratios": list(refit_ratios),
        "trellis_bits_per_strip": 74,
        "selection_population": "all naturally routed rows; no document partition",
        "proposal_metric": (
            "candidate-scale dense-H quadratic contributions partitioned over "
            "16x16 encoder tiles"
        ),
        "acceptance_metric": "decoded route-weighted whole-expert SSE",
        "capture_hashes_verified_during_run": args.verify_capture_hashes,
    }
    receipt: dict[str, object] = {"signature": signature, "complete": False, "results": {}}
    if args.output.exists():
        receipt = json.loads(args.output.read_text())
        if receipt.get("signature") != signature:
            parser.error("output receipt belongs to another experiment")

    if len(args.experts) == 1:
        rows = materialize_expert_rows(
            args.capture,
            args.layer,
            args.experts[0],
            fields=("input",),
            verify_hashes=args.verify_capture_hashes,
        )
    else:
        rows = materialize_layer_rows(
            args.capture,
            args.layer,
            fields=("input",),
            verify_hashes=args.verify_capture_hashes,
        )
    if args.row_limit is not None and not 0 < args.row_limit <= rows.rows:
        parser.error("row limit is outside the capture")
    if (
        args.evaluation_row_limit is not None
        and not 0 < args.evaluation_row_limit <= rows.rows
    ):
        parser.error("evaluation row limit is outside the capture")
    global_h13, _ = load_layer_hessians(args.hessians, args.layer)
    device = torch.device(args.device)
    store_kwargs: dict[str, object] = {}
    if args.official_repo_dir is not None:
        store_kwargs["repo_dir"] = args.official_repo_dir
    if args.official_revision is not None:
        store_kwargs["revision"] = args.official_revision
    store = OfficialMXFP4Store(**store_kwargs)
    results = receipt["results"]
    assert isinstance(results, dict)
    shared_scale_scope = (
        "pooled_high_rate_panel",
        signature["capture_manifest_sha256"],
        args.layer,
    )
    source_experts = tuple(sorted(set((*args.experts, 0))))
    with store.open_layer(args.layer, experts=source_experts) as layer_store:
        scale_source = _source_triplet(
            layer_store,
            layer=args.layer,
            expert=0,
            device=device,
        )
        shared_scale_evidence = _seed_layer_shared_scales(
            scale_source,
            global_h13,
            layer=args.layer,
            device=device,
            shared_scale_scope=shared_scale_scope,
        )
        prior_scale_evidence = receipt.get("shared_scale_evidence")
        if prior_scale_evidence is not None and prior_scale_evidence != shared_scale_evidence:
            parser.error("layer-shared scale derivation changed while resuming")
        receipt["shared_scale_evidence"] = shared_scale_evidence
        _atomic_json(args.output, receipt)
        for expert in args.experts:
            source = _source_triplet(
                layer_store,
                layer=args.layer,
                expert=expert,
                device=device,
            )
            group_scores, score_evidence = _combined_group_scores(
                rows,
                expert,
                source=source,
                device=device,
                batch_rows=args.batch_rows,
                row_limit=args.row_limit,
            )
            for draw in args.draws:
                for policy in args.policies:
                    key = f"layer-{args.layer:05d}-expert-{expert:04d}-draw-{draw}-{policy}"
                    if key in results:
                        continue
                    results[key] = {
                        "layer": args.layer,
                        "expert": expert,
                        "functional_score_evidence": score_evidence,
                        **_encode_arm(
                            layer_rows=rows,
                            source=source,
                            layer=args.layer,
                            expert=expert,
                            draw=draw,
                            policy=policy,
                            group_scores=group_scores,
                            global_h13=global_h13,
                            device=device,
                            batch_rows=args.batch_rows,
                            row_limit=args.row_limit,
                            evaluation_row_limit=args.evaluation_row_limit,
                            shared_scale_scope=shared_scale_scope,
                            maximum_upstream_candidates=args.maximum_upstream_candidates,
                            refit_ratios=refit_ratios,
                        ),
                    }
                    _atomic_json(args.output, receipt)
                    print(
                        json.dumps(
                            {
                                "completed": key,
                                "winner": results[key]["winner"],
                                "all_row": results[key]["winner_all_row_score"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    receipt["complete"] = True
    _atomic_json(args.output, receipt)


if __name__ == "__main__":
    main()

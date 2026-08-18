#!/usr/bin/env python3
"""Compare fixed-record and metadata-free periodic 3.083-bpw QSRT experts.

Every arm uses the coupled activation-boundary transform, dense-H BlockLDLQ,
decoded-upstream H2, a regularized functional W2 refit, and decoded
route-weighted whole-expert SSE over all selected naturally routed rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import torch

from qsrt.all_row_capture import materialize_expert_rows
from qsrt.capture import load_layer_hessians
from qsrt.coupled_expert_study import CoupledTriplet, apply_permutation_sign_gauge
from qsrt.high_rate_allocation import neuron_permutation_from_scores, record_rate_map
from qsrt.pack.qsrt_encoder import (
    QSRTTileMapCandidate,
    quantize_qsrt_periodic_candidate,
    quantize_qsrt_tile_map_candidates,
)
from qsrt.periodic_rate import rate_period, schedule_sha256, tile_schedules
from qsrt.pooled_calibration import (
    candidate_h2,
    collect_coupled_hidden_statistics,
    collect_upstream_functional_statistics,
    evaluate_coupled_expert_batches,
    ridge_refit_down_from_statistics,
)
from qsrt.qsrt import CONTEXT_GROUP_CHANNELS, INTERMEDIATE_CHANNELS, expand_group_order
from qsrt.qsrt_coupled import CoupledHadamardExecution, CoupledHadamardSpec, encode_coupled_weights
from qsrt.qsrt_coupled_plan import CoupledRotationPlan
from qsrt.source_weights import OfficialMXFP4Store


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


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _source_triplet(layer_store: object, layer: int, expert: int, device: torch.device) -> CoupledTriplet:
    return CoupledTriplet(
        *(
            layer_store.load_matrix(layer, expert, matrix, device=device).float()
            for matrix in ("w1", "w3", "w2")
        )
    )


def _batches(rows: object, expert: int, *, batch_rows: int, row_limit: int | None):
    return rows.expert_batches(
        expert,
        batch_rows=batch_rows,
        row_limit=row_limit,
        fields=("input",),
    )


def _group_scores(
    rows: object,
    expert: int,
    source: CoupledTriplet,
    *,
    device: torch.device,
    batch_rows: int,
    row_limit: int | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    statistics = collect_upstream_functional_statistics(
        _batches(rows, expert, batch_rows=batch_rows, row_limit=row_limit),
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
    scores = ((hidden + derivative) * 0.5).float().cpu()
    return scores, {
        "rows": float(statistics.rows),
        "effective_sample_size": statistics.effective_sample_size,
        "minimum": float(scores.min()),
        "maximum": float(scores.max()),
    }


def _permutation(scores: torch.Tensor, policy: str, *, layer: int, expert: int) -> torch.Tensor:
    if policy in ("identity", "importance", "energy_balanced", "stratified_energy_balanced"):
        return neuron_permutation_from_scores(scores, policy=policy)
    if not policy.startswith("random-"):
        raise ValueError(f"unsupported neuron permutation: {policy}")
    seed = int(policy.removeprefix("random-"))
    groups = INTERMEDIATE_CHANNELS // CONTEXT_GROUP_CHANNELS
    generator = torch.Generator().manual_seed(
        (0x51A7E3 + 1_000_003 * layer + 10_007 * expert + seed) & 0x7FFF_FFFF_FFFF_FFFF
    )
    return expand_group_order(torch.randperm(groups, generator=generator))


def _fixed_tile_map(matrix: str, matrix_source: torch.Tensor) -> tuple[int, ...]:
    encoder_shape = (matrix_source.shape[1], matrix_source.shape[0])
    return record_rate_map(
        encoder_shape,
        rate_axis="n" if matrix in ("w1", "w3") else "k",
        donor_records=0,
    )


def _encode_candidate_matrix(
    matrix_source: torch.Tensor,
    hessian: torch.Tensor,
    *,
    matrix: str,
    allocation: str,
    schedules: tuple[tuple[int, ...], ...] | None,
    layer: int,
    device: torch.device,
    shared_scale_scope: object,
) -> QSRTTileMapCandidate:
    identity = torch.arange(INTERMEDIATE_CHANNELS, device=device)
    if allocation == "fixed":
        return quantize_qsrt_tile_map_candidates(
            matrix_source,
            hessian,
            identity,
            matrix=matrix,
            maps={"candidate": _fixed_tile_map(matrix, matrix_source)},
            layer=layer,
            device=device,
            shared_scale_scope=shared_scale_scope,
            ldlq_tf32=True,
        )["candidate"]
    if schedules is None:
        raise ValueError("periodic allocation requires schedules")
    return quantize_qsrt_periodic_candidate(
        matrix_source,
        hessian,
        identity,
        matrix=matrix,
        schedules=schedules,
        layer=layer,
        device=device,
        shared_scale_scope=shared_scale_scope,
        ldlq_tf32=True,
    )


def _seed_shared_scales(
    source: CoupledTriplet,
    h13: torch.Tensor,
    *,
    layer: int,
    device: torch.device,
    shared_scale_scope: object,
) -> dict[str, object]:
    execution = CoupledHadamardExecution(source.hidden, source.intermediate, CoupledHadamardSpec())
    coordinates = CoupledTriplet(*encode_coupled_weights(source.tensors(), execution.spec))
    transformed_h13 = execution.transform_h13(h13.to(device=device, dtype=torch.float32))
    result: dict[str, object] = {}
    for matrix, matrix_source in (("w1", coordinates.gate), ("w3", coordinates.up)):
        candidate = quantize_qsrt_tile_map_candidates(
            matrix_source,
            transformed_h13,
            torch.arange(INTERMEDIATE_CHANNELS, device=device),
            matrix=matrix,
            maps={"uniform_k3": (3,) * ((matrix_source.shape[0] // 16) * (matrix_source.shape[1] // 16))},
            layer=layer,
            device=device,
            shared_scale_scope=shared_scale_scope,
            ldlq_tf32=True,
        )["uniform_k3"]
        result[matrix] = {
            "suh_sha256": _tensor_sha256(candidate.tensors["suh"]),
            "suh_shape": list(candidate.tensors["suh"].shape),
        }
    return result


def _arm(
    *,
    rows: object,
    source: CoupledTriplet,
    layer: int,
    expert: int,
    draw: int,
    permutation_policy: str,
    group_scores: torch.Tensor,
    allocation: str,
    schedules: tuple[tuple[int, ...], ...] | None,
    global_h13: torch.Tensor,
    device: torch.device,
    batch_rows: int,
    row_limit: int | None,
    shared_scale_scope: object,
    w2_refit_ratio: float,
) -> dict[str, object]:
    permutation = _permutation(group_scores, permutation_policy, layer=layer, expert=expert).to(device)
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
    coordinates = CoupledTriplet(*encode_coupled_weights(gauged.tensors(), execution.spec))
    h13 = execution.transform_h13(global_h13.to(device=device, dtype=torch.float32))
    upstream = tuple(
        _encode_candidate_matrix(
            matrix_source,
            h13,
            matrix=matrix,
            allocation=allocation,
            schedules=schedules,
            layer=layer,
            device=device,
            shared_scale_scope=shared_scale_scope,
        )
        for matrix, matrix_source in (("w1", coordinates.gate), ("w3", coordinates.up))
    )
    statistics = collect_coupled_hidden_statistics(
        _batches(rows, expert, batch_rows=batch_rows, row_limit=row_limit),
        source=gauged,
        candidate_coordinates=CoupledTriplet(
            upstream[0].reconstruction,
            upstream[1].reconstruction,
            coordinates.down,
        ),
        execution=execution,
        retain_source_gram=True,
    )
    h2, h2_evidence = candidate_h2(statistics)
    refit_t, refit_evidence = ridge_refit_down_from_statistics(
        statistics,
        coordinates.down.T,
        regularization_ratio=w2_refit_ratio,
    )
    down = _encode_candidate_matrix(
        refit_t.T.float().contiguous(),
        h2,
        matrix="w2",
        allocation=allocation,
        schedules=schedules,
        layer=layer,
        device=device,
        shared_scale_scope=shared_scale_scope,
    )
    evaluation = evaluate_coupled_expert_batches(
        _batches(rows, expert, batch_rows=batch_rows, row_limit=row_limit),
        source=gauged,
        teacher_source=source,
        candidate_coordinates=CoupledTriplet(
            upstream[0].reconstruction,
            upstream[1].reconstruction,
            down.reconstruction,
        ),
        execution=execution,
    )
    return {
        "allocation": allocation,
        "draw": draw,
        "permutation_policy": permutation_policy,
        "permutation_sha256": _tensor_sha256(permutation.cpu().to(torch.int32)),
        "schedule_sha256": None if schedules is None else schedule_sha256(schedules),
        "schedule_class_sums": None if schedules is None else [sum(value) for value in schedules],
        "candidate_h2": h2_evidence,
        "w2_refit": refit_evidence,
        "sse": evaluation.sse,
        "source_energy": evaluation.source_energy,
        "nmse": evaluation.nmse,
        "routed_occurrences": evaluation.routed_occurrences,
        "matrix_proxy": {
            "w1": upstream[0].proxy,
            "w3": upstream[1].proxy,
            "w2": down.proxy,
        },
        "matrix_global_scale": {
            "w1": upstream[0].global_scale,
            "w3": upstream[1].global_scale,
            "w2": down.global_scale,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--hessians", type=Path, required=True)
    parser.add_argument("--rotation-plan", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--experts", type=_parse_ints, required=True)
    parser.add_argument("--permutations", type=_parse_strings, default=("identity",))
    parser.add_argument(
        "--periodic-schedules",
        type=_parse_strings,
        default=("clustered", "interleaved", "random-0", "random-1", "random-2"),
    )
    parser.add_argument("--donor-records", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--w2-refit-ratio", type=float, default=1e-2)
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument("--official-revision")
    parser.add_argument("--verify-capture-hashes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.layer <= 92:
        parser.error("layer must lie in 1..92")
    if any(not 0 <= expert < 896 for expert in args.experts):
        parser.error("experts must lie in 0..895")
    if not 0 <= args.donor_records <= 11:
        parser.error("donor records must lie in 0..11")
    if args.batch_rows <= 0:
        parser.error("batch rows must be positive")
    if not math.isfinite(args.w2_refit_ratio) or args.w2_refit_ratio <= 0:
        parser.error("W2 refit ratio must be positive and finite")

    schedule_banks: dict[str, tuple[tuple[int, ...], ...]] = {}
    for name in args.periodic_schedules:
        if name in ("clustered", "interleaved"):
            period = rate_period(args.donor_records, ordering=name)
        elif name.startswith("random-"):
            period = rate_period(
                args.donor_records,
                ordering="random",
                seed=int(name.removeprefix("random-")),
            )
        else:
            parser.error(f"unsupported periodic schedule {name!r}")
        schedule_banks[name] = tile_schedules(period)

    capture_manifest = args.capture / "manifest.json"
    hessian_manifest = args.hessians / "manifest.json"
    raw_plan = json.loads(args.rotation_plan.read_text())
    if "coupled_rotation_plan" in raw_plan:
        raw_plan = raw_plan["coupled_rotation_plan"]
    plan = CoupledRotationPlan.from_json(raw_plan)
    signature = {
        "kind": "qsrt_periodic_rate_expert_panel",
        "schema_version": 1,
        "capture": str(args.capture.resolve()),
        "capture_manifest_sha256": _sha256(capture_manifest),
        "hessians": str(args.hessians.resolve()),
        "hessian_manifest_sha256": _sha256(hessian_manifest),
        "rotation_plan": str(args.rotation_plan.resolve()),
        "rotation_plan_sha256": _sha256(args.rotation_plan),
        "layer": args.layer,
        "experts": list(args.experts),
        "permutations": list(args.permutations),
        "periodic_schedules": {
            name: schedule_sha256(schedules) for name, schedules in schedule_banks.items()
        },
        "donor_records_per_24_positions": args.donor_records,
        "row_limit": args.row_limit,
        "batch_rows": args.batch_rows,
        "w2_refit_ratio": args.w2_refit_ratio,
        "selection_population": "all selected naturally routed rows; no document partition",
        "metric": "decoded route-weighted whole-expert SSE",
        "capture_hashes_verified_during_run": args.verify_capture_hashes,
    }
    receipt: dict[str, object] = {"signature": signature, "complete": False, "results": {}}
    if args.output.exists():
        receipt = json.loads(args.output.read_text())
        if receipt.get("signature") != signature:
            parser.error("output receipt belongs to another experiment")
    results = receipt["results"]
    assert isinstance(results, dict)

    global_h13, _ = load_layer_hessians(args.hessians, args.layer)
    device = torch.device(args.device)
    store_kwargs: dict[str, object] = {}
    if args.official_repo_dir is not None:
        store_kwargs["repo_dir"] = args.official_repo_dir
    if args.official_revision is not None:
        store_kwargs["revision"] = args.official_revision
    store = OfficialMXFP4Store(**store_kwargs)
    shared_scale_scope = (
        "periodic_rate_panel",
        signature["capture_manifest_sha256"],
        args.layer,
    )
    source_experts = tuple(sorted(set((*args.experts, 0))))
    with store.open_layer(args.layer, experts=source_experts) as layer_store:
        source_zero = _source_triplet(layer_store, args.layer, 0, device)
        scale_evidence = _seed_shared_scales(
            source_zero,
            global_h13,
            layer=args.layer,
            device=device,
            shared_scale_scope=shared_scale_scope,
        )
        if receipt.get("shared_scale_evidence", scale_evidence) != scale_evidence:
            parser.error("shared scale derivation changed while resuming")
        receipt["shared_scale_evidence"] = scale_evidence
        _atomic_json(args.output, receipt)

        layer_draws = plan.for_layer(args.layer)
        for expert in args.experts:
            rows = materialize_expert_rows(
                args.capture,
                args.layer,
                expert,
                fields=("input",),
                verify_hashes=args.verify_capture_hashes,
            )
            if args.row_limit is not None and not 0 < args.row_limit <= rows.rows:
                parser.error(f"row limit is outside expert {expert}'s capture")
            source = _source_triplet(layer_store, args.layer, expert, device)
            scores, score_evidence = _group_scores(
                rows,
                expert,
                source,
                device=device,
                batch_rows=args.batch_rows,
                row_limit=args.row_limit,
            )
            draw = layer_draws[expert]
            arms = [("fixed", "fixed", None)] + [
                (f"periodic-{name}", "periodic", schedules)
                for name, schedules in schedule_banks.items()
            ]
            for permutation_policy in args.permutations:
                for arm_name, allocation, schedules in arms:
                    key = f"expert-{expert:04d}-{permutation_policy}-{arm_name}"
                    if key in results:
                        continue
                    result = _arm(
                        rows=rows,
                        source=source,
                        layer=args.layer,
                        expert=expert,
                        draw=draw,
                        permutation_policy=permutation_policy,
                        group_scores=scores,
                        allocation=allocation,
                        schedules=schedules,
                        global_h13=global_h13,
                        device=device,
                        batch_rows=args.batch_rows,
                        row_limit=args.row_limit,
                        shared_scale_scope=shared_scale_scope,
                        w2_refit_ratio=args.w2_refit_ratio,
                    )
                    results[key] = {
                        "layer": args.layer,
                        "expert": expert,
                        "functional_score_evidence": score_evidence,
                        **result,
                    }
                    _atomic_json(args.output, receipt)
                    print(json.dumps({"completed": key, **result}, sort_keys=True), flush=True)
                    torch.cuda.empty_cache()
    receipt["complete"] = True
    _atomic_json(args.output, receipt)


if __name__ == "__main__":
    main()

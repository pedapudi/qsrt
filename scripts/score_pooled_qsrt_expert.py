#!/usr/bin/env python3
"""Score one decoded QSRT expert over every naturally routed captured row.

The score compares the serialized candidate with the official MXFP4 source
expert.  It uses the applied route weight exactly once before squared error,
constructs candidate-specific post-SiTU sufficient statistics, and reports
selection stability for requested prefixes of the capture population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from qsrt.all_row_capture import materialize_layer_rows
from qsrt.coupled_expert_study import CoupledTriplet
from qsrt.pack.qsrt_validation import decode_candidate_matrix
from qsrt.pooled_calibration import (
    candidate_h2,
    decoded_down_sse_difference,
    evaluate_coupled_expert_batches,
    ridge_refit_down_from_statistics,
)
from qsrt.qsrt_coupled import (
    CoupledHadamardExecution,
    CoupledHadamardSpec,
    encode_coupled_weights,
)
from qsrt.source_weights import OfficialMXFP4Store


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _parse_prefixes(value: str) -> tuple[int, ...]:
    try:
        result = tuple(sorted(set(int(item) for item in value.split(",") if item)))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("prefix rows must be comma-separated integers") from exc
    if any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("prefix rows must be positive")
    return result


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--prefix-rows", type=_parse_prefixes, default=())
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument("--official-revision")
    parser.add_argument("--statistics-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-local-alpha", type=float, default=0.75)
    parser.add_argument(
        "--w2-refit-regularization-ratios",
        type=str,
        default="1e-5,1e-4,1e-3,1e-2",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.layer <= 0 or args.expert < 0 or args.batch_rows <= 0:
        parser.error("layer and batch rows must be positive; expert must be nonnegative")
    try:
        refit_ratios = tuple(
            float(item) for item in args.w2_refit_regularization_ratios.split(",") if item
        )
    except ValueError as exc:
        parser.error(f"invalid W2 refit regularization ratio: {exc}")
    if not refit_ratios or any(value <= 0 for value in refit_ratios):
        parser.error("W2 refit regularization ratios must be positive")

    capture_manifest_path = args.capture / "manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text())
    if not bool(capture_manifest.get("complete", False)):
        parser.error("capture must be finalized before pooled scoring")
    rows = int(capture_manifest.get("rows", 0))
    if any(limit > rows for limit in args.prefix_rows):
        parser.error("a prefix row limit exceeds the finalized capture")

    pool_manifest_path = args.candidate_pool / "qsrt-candidate-manifest.json"
    pool_manifest = json.loads(pool_manifest_path.read_text())
    candidate_path = (
        args.candidate_pool
        / "candidates"
        / f"qsrt-layer-{args.layer:05d}.safetensors"
    )
    metrics_path = candidate_path.with_name(
        f"qsrt-layer-{args.layer:05d}.metrics.safetensors"
    )
    if not candidate_path.is_file() or not metrics_path.is_file():
        parser.error("candidate pool lacks the canonical layer payload or metrics")
    metrics = load_file(str(metrics_path), device="cpu")
    required_metrics = {"coupled_draw_selected", "selected_r13", "selected_r2"}
    if not required_metrics.issubset(metrics):
        parser.error("candidate metrics lack coupled draw or rate selections")
    if args.expert >= metrics["selected_r13"].numel():
        parser.error("expert is outside the candidate layer")

    layer_rows = materialize_layer_rows(
        args.capture,
        args.layer,
        fields=("input",),
        verify_hashes=True,
    )
    device = torch.device(args.device)
    store_kwargs: dict[str, object] = {}
    if args.official_repo_dir is not None:
        store_kwargs["repo_dir"] = args.official_repo_dir
    if args.official_revision is not None:
        store_kwargs["revision"] = args.official_revision
    source_store = OfficialMXFP4Store(**store_kwargs)
    with source_store.open_layer(args.layer, experts=(args.expert,)) as layer_store:
        source = CoupledTriplet(
            *(
                layer_store.load_matrix(
                    args.layer,
                    args.expert,
                    matrix,
                    device=device,
                ).float()
                for matrix in ("w1", "w3", "w2")
            )
        )

    r13 = int(metrics["selected_r13"][args.expert])
    r2 = int(metrics["selected_r2"][args.expert])
    draw = int(metrics["coupled_draw_selected"][args.expert])
    with safe_open(candidate_path, framework="pt", device="cpu") as reader:
        candidate = CoupledTriplet(
            *(
                decode_candidate_matrix(
                    reader,
                    layer=args.layer,
                    expert=args.expert,
                    matrix=matrix,
                    mode_id=r13 if matrix != "w2" else r2,
                    device=device,
                    logical_trellis_schema=str(pool_manifest["logical_trellis_schema"]),
                    codebook=str(pool_manifest["codebook"]),
                ).T.float().contiguous()
                for matrix in ("w1", "w3", "w2")
            )
        )
    execution = CoupledHadamardExecution(
        hidden=source.hidden,
        intermediate=source.intermediate,
        spec=CoupledHadamardSpec(intermediate_draw=draw),
    )
    evaluation = evaluate_coupled_expert_batches(
        layer_rows.expert_batches(
            args.expert,
            batch_rows=args.batch_rows,
            fields=("input",),
        ),
        source=source,
        candidate_coordinates=candidate,
        execution=execution,
        prefix_row_limits=args.prefix_rows,
        statistics_dtype=(
            torch.float64 if args.statistics_dtype == "float64" else torch.float32
        ),
    )
    h2, h2_evidence = candidate_h2(
        evaluation.statistics,
        max_local_alpha=args.max_local_alpha,
    )
    source_coordinates = CoupledTriplet(
        *encode_coupled_weights(source.tensors(), execution.spec)
    )
    refits = []
    statistics = evaluation.statistics
    for ratio in refit_ratios:
        target, evidence = ridge_refit_down_from_statistics(
            statistics,
            source_coordinates.down.T,
            regularization_ratio=ratio,
        )
        refits.append(
            {
                **evidence,
                "dense_target_sse_change_from_serialized_w2": float(
                    decoded_down_sse_difference(
                        statistics,
                        target,
                        candidate.down.T,
                        source_coordinates.down.T,
                    )
                ),
            }
        )
    receipt: dict[str, object] = {
        "kind": "qsrt_pooled_expert_score",
        "schema_version": 1,
        "capture": str(args.capture.resolve()),
        "capture_manifest_sha256": _sha256(capture_manifest_path),
        "capture_rows": rows,
        "candidate_pool": str(args.candidate_pool.resolve()),
        "candidate_manifest_sha256": _sha256(pool_manifest_path),
        "candidate_layer_payload_sha256": _sha256(candidate_path),
        "candidate_layer_metrics_sha256": _sha256(metrics_path),
        "official_source_revision": source_store.revision,
        "layer": args.layer,
        "expert": args.expert,
        "intermediate_draw": draw,
        "r13": r13,
        "r2": r2,
        "routed_occurrences": evaluation.routed_occurrences,
        "sse": evaluation.sse,
        "source_energy": evaluation.source_energy,
        "nmse": evaluation.nmse,
        "route_weight_convention": "applied_gate; squared_once_in_sse",
        "prefix_scores": {
            str(limit): {
                "sse": score[0],
                "source_energy": score[1],
                "routed_occurrences": score[2],
                "nmse": score[0] / score[1] if score[1] > 0 else None,
            }
            for limit, score in evaluation.prefix_scores.items()
        },
        "candidate_h2": {
            **h2_evidence,
            "trace": float(torch.trace(h2)),
        },
        "dense_w2_refit_oracles": refits,
        "dense_w2_refit_status": "research-only target; requires K2/K3/K4 re-encode",
    }
    _write_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

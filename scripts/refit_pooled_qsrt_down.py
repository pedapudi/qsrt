#!/usr/bin/env python3
"""Refit and encode one uniform-K2 down projection from pooled routed rows.

The stored gate and up payloads are decoded from an existing candidate pool.
Their post-activation rows define both the regularized functional target for
the down projection and its expert-local dense covariance.  The replacement
down projection is encoded through the production fixed-K2 path.  The command
fails if the replacement changes the layer-shared down-projection scale plane.
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
from safetensors.torch import save_file

from qsrt.all_row_capture import MaterializedExpertRows, materialize_expert_rows
from qsrt.coupled_expert_study import CoupledTriplet
from qsrt.exl3_reference import CODEBOOK_SQG_XOR_CHEB_T12
from qsrt.pack.qsrt_atoms import candidate_layer_path
from qsrt.pack.qsrt_candidates import candidate_tensor_name
from qsrt.pack.qsrt_encoder import quantize_qsrt_matrix
from qsrt.pack.qsrt_validation import decode_candidate_matrix
from qsrt.pooled_calibration import (
    candidate_h2,
    collect_coupled_hidden_statistics,
    decoded_down_sse,
    evaluate_coupled_expert_batches,
    ridge_refit_down_from_statistics,
)
from qsrt.qsrt import K2, SCHEMA
from qsrt.qsrt_coupled import (
    CoupledHadamardExecution,
    CoupledHadamardSpec,
    encode_coupled_weights,
)
from qsrt.qsrt_coupled_plan import CoupledRotationPlan
from qsrt.source_weights import OfficialMXFP4Store


MATRICES = ("w1", "w3", "w2")
CONTEXT_GROUPS = 3072 // 4


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _batches(
    rows: MaterializedExpertRows,
    expert: int,
    *,
    batch_rows: int,
) -> Iterable[dict[str, torch.Tensor]]:
    return rows.expert_batches(
        expert,
        batch_rows=batch_rows,
        fields=("input",),
    )


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


def _rotation_plan(manifest: Mapping[str, object]) -> CoupledRotationPlan:
    value = manifest.get("coupled_k2_rotation_plan")
    if value is None:
        raise ValueError("source model does not define a coupled rotation plan")
    if isinstance(value, dict) and value.get("kind") == (
        "kquant_qsrt_k2_coupled_rotation_plan"
    ):
        value = {**value, "kind": "qsrt_k2_coupled_rotation_plan"}
    return CoupledRotationPlan.from_json(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--regularization-ratio", type=float, default=1e-2)
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument("--official-revision")
    parser.add_argument("--verify-capture-hashes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.layer <= 92 or not 0 <= args.expert < 896:
        parser.error("layer must lie in 1..92 and expert in 0..895")
    if args.batch_rows <= 0:
        parser.error("batch rows must be positive")
    if not math.isfinite(args.regularization_ratio) or args.regularization_ratio <= 0:
        parser.error("regularization ratio must be finite and positive")
    if args.device == "cpu" or not args.device.startswith("cuda"):
        parser.error("the production QSRT encoder requires a CUDA device")

    capture_manifest_path = args.capture / "manifest.json"
    pool_manifest_path = args.candidate_pool / "qsrt-candidate-manifest.json"
    source_manifest_path = args.source_model / "qsrt-manifest.json"
    capture_manifest = _read_json(capture_manifest_path)
    pool_manifest = _read_json(pool_manifest_path)
    source_manifest = _read_json(source_manifest_path)
    if not bool(capture_manifest.get("complete", False)):
        parser.error("capture is not finalized")
    if pool_manifest.get("format_grid") != "fixed_k2":
        parser.error("candidate pool is not the uniform-K2 profile")
    codebook = str(pool_manifest.get("codebook", ""))
    if codebook != CODEBOOK_SQG_XOR_CHEB_T12:
        parser.error("candidate pool uses an unsupported reconstruction law")
    source_logical_schema = str(pool_manifest.get("logical_trellis_schema", ""))
    if not source_logical_schema:
        parser.error("candidate pool omits its logical trellis schema")
    logical_schema = {
        "kquant_kimi_k3_qsrt_candidate_v1": SCHEMA,
        SCHEMA: SCHEMA,
    }.get(source_logical_schema)
    if logical_schema is None:
        parser.error(f"candidate pool uses an unsupported schema: {source_logical_schema}")

    rotations = _rotation_plan(source_manifest)
    draw = rotations.for_layer(args.layer)[args.expert]
    candidate_path = candidate_layer_path(args.candidate_pool, args.layer)
    if not candidate_path.is_file():
        parser.error(f"candidate layer does not exist: {candidate_path}")

    signature = {
        "kind": "qsrt_pooled_functional_down_refit",
        "schema_version": 1,
        "capture": str(args.capture.resolve()),
        "capture_manifest_sha256": _sha256(capture_manifest_path),
        "candidate_pool": str(args.candidate_pool.resolve()),
        "candidate_pool_manifest_sha256": _sha256(pool_manifest_path),
        "source_model": str(args.source_model.resolve()),
        "source_model_manifest_sha256": _sha256(source_manifest_path),
        "layer": args.layer,
        "expert": args.expert,
        "intermediate_rotation_draw": draw,
        "regularization_ratio": args.regularization_ratio,
        "batch_rows": args.batch_rows,
        "codebook": codebook,
        "source_logical_trellis_schema": source_logical_schema,
        "logical_trellis_schema": logical_schema,
        "capture_hashes_verified_during_run": args.verify_capture_hashes,
        "objective": (
            "route-weight-squared expert-output SSE against the official source "
            "expert on all naturally routed captured rows"
        ),
    }
    receipt_path = args.output.with_suffix(args.output.suffix + ".json")
    if receipt_path.is_file():
        prior = _read_json(receipt_path)
        if prior.get("signature") != signature:
            parser.error("output receipt belongs to another refit")
        if bool(prior.get("complete", False)) and args.output.is_file():
            print(json.dumps({"complete": str(args.output)}, sort_keys=True))
            return
    _atomic_json(receipt_path, {"complete": False, "signature": signature})

    rows = materialize_expert_rows(
        args.capture,
        args.layer,
        args.expert,
        fields=("input",),
        verify_hashes=args.verify_capture_hashes,
    )
    device = torch.device(args.device)
    store_kwargs: dict[str, object] = {}
    if args.official_repo_dir is not None:
        store_kwargs["repo_dir"] = args.official_repo_dir
    if args.official_revision is not None:
        store_kwargs["revision"] = args.official_revision
    store = OfficialMXFP4Store(**store_kwargs)

    with store.open_layer(args.layer, experts=(args.expert,)) as layer_store:
        source = _source_triplet(
            layer_store,
            layer=args.layer,
            expert=args.expert,
            device=device,
        )
    spec = CoupledHadamardSpec(intermediate_draw=draw)
    execution = CoupledHadamardExecution(source.hidden, source.intermediate, spec)
    source_coordinates = CoupledTriplet(*encode_coupled_weights(source.tensors(), spec))

    with safe_open(candidate_path, framework="pt", device="cpu") as reader:
        decoded = {
            matrix: decode_candidate_matrix(
                reader,
                layer=args.layer,
                expert=args.expert,
                matrix=matrix,
                mode_id=K2.mode_id,
                device=device,
                logical_trellis_schema=logical_schema,
                codebook=codebook,
            ).T.contiguous()
            for matrix in MATRICES
        }
        baseline_shared_svh = reader.get_tensor(
            candidate_tensor_name(args.layer, args.expert, "w2", "svh")
        ).contiguous()
    baseline = CoupledTriplet(decoded["w1"], decoded["w3"], decoded["w2"])
    statistics = collect_coupled_hidden_statistics(
        _batches(rows, args.expert, batch_rows=args.batch_rows),
        source=source,
        candidate_coordinates=baseline,
        execution=execution,
        retain_source_gram=True,
    )
    source_down_t = source_coordinates.down.T
    baseline_sse = float(decoded_down_sse(statistics, baseline.down.T, source_down_t))
    h2, h2_evidence = candidate_h2(statistics)
    refit_t, refit_evidence = ridge_refit_down_from_statistics(
        statistics,
        source_down_t,
        regularization_ratio=args.regularization_ratio,
    )
    replacement = quantize_qsrt_matrix(
        refit_t.T.float().contiguous(),
        torch.zeros(CONTEXT_GROUPS, dtype=torch.long, device=device),
        matrix="w2",
        mode=K2,
        hessian=h2,
        layer=args.layer,
        device=device,
        layout="importance_ordered",
        logical_trellis_schema=logical_schema,
        codebook=codebook,
        ldlq_tf32=True,
        tailbite_context=int(pool_manifest.get("tailbite_context", 128)),
    )
    replacement_sse = float(
        decoded_down_sse(statistics, replacement.reconstruction.T, source_down_t)
    )
    if replacement_sse >= baseline_sse:
        raise ArithmeticError(
            "encoded functional down refit did not improve pooled expert SSE: "
            f"{replacement_sse:.9g} >= {baseline_sse:.9g}"
        )
    if not torch.equal(replacement.tensors["svh"].cpu(), baseline_shared_svh):
        maximum = float(
            (
                replacement.tensors["svh"].detach().cpu().float()
                - baseline_shared_svh.float()
            )
            .abs()
            .max()
        )
        raise ArithmeticError(
            "encoded down refit changed the layer-shared svh plane; "
            f"maximum absolute difference {maximum:.9g}"
        )

    explicit = evaluate_coupled_expert_batches(
        _batches(rows, args.expert, batch_rows=args.batch_rows),
        source=source,
        candidate_coordinates=CoupledTriplet(
            baseline.gate,
            baseline.up,
            replacement.reconstruction,
        ),
        execution=execution,
    )
    relative_closure = abs(explicit.sse - replacement_sse) / max(explicit.sse, 1e-30)
    if relative_closure > 2e-4:
        raise ArithmeticError(
            "explicit and sufficient-statistic SSE do not close: "
            f"{explicit.sse:.9g} versus {replacement_sse:.9g}"
        )

    output_tensors = {
        candidate_tensor_name(args.layer, args.expert, "w2", part): (
            replacement.tensors[part].detach().cpu().contiguous()
        )
        for part in ("trellis", "suh", "svh")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    save_file(
        output_tensors,
        temporary,
        metadata={
            "kind": "qsrt_pooled_functional_down_refit",
            "layer": str(args.layer),
            "expert": str(args.expert),
            "codebook": codebook,
        },
    )
    temporary.replace(args.output)
    improvement = (baseline_sse - replacement_sse) / baseline_sse
    receipt = {
        "complete": True,
        "signature": signature,
        "output_sha256": _sha256(args.output),
        "result": {
            "routed_occurrences": statistics.rows,
            "effective_sample_size": statistics.effective_sample_size,
            "baseline_sse": baseline_sse,
            "replacement_sse": replacement_sse,
            "relative_improvement": improvement,
            "explicit_sse": explicit.sse,
            "explicit_source_energy": explicit.source_energy,
            "relative_sse_closure": relative_closure,
            "shared_svh_sha256": _tensor_sha256(baseline_shared_svh),
            "candidate_h2": h2_evidence,
            "refit": refit_evidence,
            "coding": replacement.coding,
        },
    }
    _atomic_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "complete": str(args.output),
                "routed_occurrences": statistics.rows,
                "relative_improvement": improvement,
                "relative_sse_closure": relative_closure,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

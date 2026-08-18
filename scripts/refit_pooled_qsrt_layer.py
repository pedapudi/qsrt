#!/usr/bin/env python3
"""Build one uniform-K2 candidate layer with pooled functional W2 refits.

The command materializes one all-row capture layer once and shares those CPU
rows between one worker per requested GPU.  Gate and up payloads are copied
byte-for-byte from the source candidate pool.  Each worker decodes the stored
upstream matrices, fits and encodes a candidate-specific down projection, and
retains the source-pool down payload whenever the encoded refit does not reduce
the exact pooled quadratic expert-output loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from qsrt.all_row_capture import MaterializedLayerRows, materialize_layer_rows
from qsrt.coupled_expert_study import CoupledTriplet
from qsrt.exl3_loader import load_qsrt_encoder
from qsrt.exl3_reference import CODEBOOK_SQG_XOR_CHEB_T12
from qsrt.io.safetensors_stream import AtomicSafetensorsWriter, TensorSpec
from qsrt.pack.qsrt_atoms import candidate_layer_path
from qsrt.pack.qsrt_candidates import candidate_tensor_name
from qsrt.pack.qsrt_encoder import quantize_qsrt_matrix
from qsrt.pack.qsrt_pool import (
    POOLED_FIXED_PROFILE_METRICS_KIND,
    POOLED_FIXED_PROFILE_METRICS_SCHEMA_VERSION,
    pooled_fixed_profile_selection_contract,
    validate_candidate_layer_payload,
    validate_pooled_fixed_profile_ledger_evidence,
    validate_pooled_fixed_profile_metrics,
)
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
from qsrt.sqg_quantizer import install_sqg_quantizer


MATRICES = ("w1", "w3", "w2")
PARTS = ("trellis", "suh", "svh")
CONTEXT_GROUPS = 3072 // 4


def _parse_ints(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            begin, end = map(int, item.split("-", 1))
            result.extend(range(begin, end + 1))
        else:
            result.append(int(item))
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("expected unique comma-separated integers/ranges")
    return tuple(result)


def _parse_devices(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("device list must be nonempty and unique")
    if any(not item.startswith("cuda:") for item in result):
        raise argparse.ArgumentTypeError("every worker device must be an explicit CUDA device")
    return result


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


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_safetensors(
    path: Path,
    tensors: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        save_file(dict(tensors), temporary, metadata=dict(metadata))
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sidecar_paths(payload: Path) -> tuple[Path, Path, Path]:
    if payload.suffix != ".safetensors":
        raise ValueError("candidate layer output must use a .safetensors suffix")
    stem = payload.with_suffix("")
    return (
        stem.with_suffix(".metrics.safetensors"),
        stem.with_suffix(".selection.json"),
        stem.with_suffix(".build.json"),
    )


def _translated_rotation_plan(manifest: Mapping[str, object]) -> CoupledRotationPlan:
    value = manifest.get("coupled_k2_rotation_plan")
    if not isinstance(value, dict):
        raise ValueError("source model does not define a coupled rotation plan")
    if value.get("kind") == "kquant_qsrt_k2_coupled_rotation_plan":
        value = {**value, "kind": "qsrt_k2_coupled_rotation_plan"}
    return CoupledRotationPlan.from_json(value)


def _translated_logical_schema(manifest: Mapping[str, object]) -> tuple[str, str]:
    source = str(manifest.get("logical_trellis_schema", ""))
    translated = {
        "kquant_kimi_k3_qsrt_candidate_v1": SCHEMA,
        SCHEMA: SCHEMA,
    }.get(source)
    if translated is None:
        raise ValueError(f"source pool uses an unsupported logical schema: {source}")
    return source, translated


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
    rows: MaterializedLayerRows,
    expert: int,
    *,
    batch_rows: int,
) -> Iterable[dict[str, torch.Tensor]]:
    return rows.expert_batches(
        expert,
        batch_rows=batch_rows,
        fields=("input",),
    )


def _payload_specs(
    layer: int,
    experts: Sequence[int],
    source_path: Path,
) -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = []
    with safe_open(source_path, framework="pt", device="cpu") as reader:
        for expert in experts:
            for matrix in MATRICES:
                for part in PARTS:
                    name = candidate_tensor_name(layer, expert, matrix, part)
                    tensor = reader.get_tensor(name)
                    specs.append(TensorSpec(name, tensor.dtype, tuple(tensor.shape)))
    return tuple(specs)


def _refit_expert(
    *,
    rows: MaterializedLayerRows,
    source_reader: object,
    layer_store: object,
    quantizer_module: object,
    layer: int,
    expert: int,
    draw: int,
    device: torch.device,
    batch_rows: int,
    regularization_ratio: float,
    logical_schema: str,
    codebook: str,
    tailbite_context: int,
    explicit_validation: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    started = time.time()
    source = _source_triplet(
        layer_store,
        layer=layer,
        expert=expert,
        device=device,
    )
    spec = CoupledHadamardSpec(intermediate_draw=draw)
    execution = CoupledHadamardExecution(source.hidden, source.intermediate, spec)
    source_coordinates = CoupledTriplet(*encode_coupled_weights(source.tensors(), spec))
    decoded = {
        matrix: decode_candidate_matrix(
            source_reader,
            layer=layer,
            expert=expert,
            matrix=matrix,
            mode_id=K2.mode_id,
            device=device,
            logical_trellis_schema=logical_schema,
            codebook=codebook,
        ).T.contiguous()
        for matrix in MATRICES
    }
    baseline = CoupledTriplet(decoded["w1"], decoded["w3"], decoded["w2"])
    statistics = collect_coupled_hidden_statistics(
        _batches(rows, expert, batch_rows=batch_rows),
        source=source,
        candidate_coordinates=baseline,
        execution=execution,
        retain_source_gram=True,
    )
    source_down_t = source_coordinates.down.T
    if statistics.source_gram is None:
        raise AssertionError("pooled down refit did not retain source statistics")
    source_down64 = source_down_t.to(
        device=statistics.source_gram.device,
        dtype=statistics.source_gram.dtype,
    )
    source_energy = float(
        torch.sum(source_down64 * (statistics.source_gram @ source_down64))
    )
    baseline_sse = float(decoded_down_sse(statistics, baseline.down.T, source_down_t))
    h2, h2_evidence = candidate_h2(statistics)
    refit_t, refit_evidence = ridge_refit_down_from_statistics(
        statistics,
        source_down_t,
        regularization_ratio=regularization_ratio,
    )
    replacement = quantize_qsrt_matrix(
        refit_t.T.float().contiguous(),
        torch.zeros(CONTEXT_GROUPS, dtype=torch.long, device=device),
        matrix="w2",
        mode=K2,
        hessian=h2,
        layer=layer,
        device=device,
        layout="importance_ordered",
        quantizer_module=quantizer_module,
        logical_trellis_schema=logical_schema,
        codebook=codebook,
        ldlq_tf32=True,
        tailbite_context=tailbite_context,
    )
    replacement_sse = float(
        decoded_down_sse(statistics, replacement.reconstruction.T, source_down_t)
    )
    selected = "functional_refit" if replacement_sse < baseline_sse else "source_pool"
    selected_sse = min(baseline_sse, replacement_sse)
    baseline_tensors = {
        part: source_reader.get_tensor(candidate_tensor_name(layer, expert, "w2", part))
        for part in PARTS
    }
    replacement_shared = replacement.tensors["svh"].detach().cpu().contiguous()
    if not torch.equal(replacement_shared, baseline_tensors["svh"]):
        maximum = float(
            (replacement_shared.float() - baseline_tensors["svh"].float()).abs().max()
        )
        raise ArithmeticError(
            f"expert {expert} replacement changed layer-shared w2 svh; "
            f"maximum absolute difference {maximum:.9g}"
        )
    tensors = (
        {
            part: replacement.tensors[part].detach().cpu().contiguous()
            for part in PARTS
        }
        if selected == "functional_refit"
        else {part: value.contiguous() for part, value in baseline_tensors.items()}
    )
    explicit: dict[str, float] | None = None
    if explicit_validation:
        evaluated = evaluate_coupled_expert_batches(
            _batches(rows, expert, batch_rows=batch_rows),
            source=source,
            candidate_coordinates=CoupledTriplet(
                baseline.gate,
                baseline.up,
                replacement.reconstruction if selected == "functional_refit" else baseline.down,
            ),
            execution=execution,
        )
        closure = abs(evaluated.sse - selected_sse) / max(evaluated.sse, 1e-30)
        if closure > 2e-4:
            raise ArithmeticError(
                f"expert {expert} explicit and quadratic SSE do not close: "
                f"{evaluated.sse:.9g} versus {selected_sse:.9g}"
            )
        explicit = {
            "sse": evaluated.sse,
            "source_energy": evaluated.source_energy,
            "relative_sse_closure": closure,
        }
    return tensors, {
        "expert": expert,
        "draw": draw,
        "selected": selected,
        "routed_occurrences": statistics.rows,
        "effective_sample_size": statistics.effective_sample_size,
        "source_energy": source_energy,
        "baseline_sse": baseline_sse,
        "replacement_sse": replacement_sse,
        "selected_sse": selected_sse,
        "relative_improvement": (baseline_sse - selected_sse) / baseline_sse,
        "candidate_h2": h2_evidence,
        "refit": refit_evidence,
        "explicit_validation": explicit,
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--experts", type=_parse_ints, default=tuple(range(896)))
    parser.add_argument("--devices", type=_parse_devices, required=True)
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--regularization-ratio", type=float, default=1e-2)
    parser.add_argument("--explicit-validation-stride", type=int, default=128)
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument("--official-revision")
    parser.add_argument("--exllamav3-root", type=Path, default=Path("/home/luke/projects/exllamav3"))
    parser.add_argument("--verify-capture-hashes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experts = tuple(int(value) for value in args.experts)
    if not 1 <= args.layer <= 92 or any(not 0 <= expert < 896 for expert in experts):
        parser.error("layer must lie in 1..92 and experts in 0..895")
    if args.batch_rows <= 0 or args.explicit_validation_stride <= 0:
        parser.error("batch rows and validation stride must be positive")
    if not math.isfinite(args.regularization_ratio) or args.regularization_ratio <= 0:
        parser.error("regularization ratio must be finite and positive")
    if len(args.devices) > len(experts):
        parser.error("worker device count cannot exceed expert count")

    capture_manifest_path = args.capture / "manifest.json"
    pool_manifest_path = args.candidate_pool / "qsrt-candidate-manifest.json"
    source_manifest_path = args.source_model / "qsrt-manifest.json"
    capture_manifest = _read_json(capture_manifest_path)
    pool_manifest = _read_json(pool_manifest_path)
    source_manifest = _read_json(source_manifest_path)
    if not bool(capture_manifest.get("complete", False)):
        parser.error("capture is not finalized")
    if pool_manifest.get("format_grid") != "fixed_k2":
        parser.error("source candidate pool is not uniform K2")
    codebook = str(pool_manifest.get("codebook", ""))
    if codebook != CODEBOOK_SQG_XOR_CHEB_T12:
        parser.error("source candidate pool uses an unsupported reconstruction law")
    source_schema, logical_schema = _translated_logical_schema(pool_manifest)
    rotations = _translated_rotation_plan(source_manifest)
    draws = rotations.for_layer(args.layer)
    source_path = candidate_layer_path(args.candidate_pool, args.layer)
    if not source_path.is_file():
        parser.error(f"source candidate layer does not exist: {source_path}")

    metrics_path, ledger_path, receipt_path = _sidecar_paths(args.output)
    signature = {
        "kind": "qsrt_pooled_functional_down_refit_layer",
        "schema_version": 1,
        "capture": str(args.capture.resolve()),
        "capture_manifest_sha256": _sha256(capture_manifest_path),
        "source_candidate_pool": str(args.candidate_pool.resolve()),
        "source_candidate_pool_manifest_sha256": _sha256(pool_manifest_path),
        "source_model": str(args.source_model.resolve()),
        "source_model_manifest_sha256": _sha256(source_manifest_path),
        "layer": args.layer,
        "experts": list(experts),
        "batch_rows": args.batch_rows,
        "regularization_ratio": args.regularization_ratio,
        "explicit_validation_stride": args.explicit_validation_stride,
        "source_logical_trellis_schema": source_schema,
        "logical_trellis_schema": logical_schema,
        "codebook": codebook,
        "capture_hashes_verified_during_run": args.verify_capture_hashes,
    }
    if receipt_path.is_file():
        prior = _read_json(receipt_path)
        if prior.get("signature") != signature:
            parser.error("output receipt belongs to another layer build")
        if bool(prior.get("complete", False)) and all(
            path.is_file() for path in (args.output, metrics_path, ledger_path)
        ):
            print(json.dumps({"complete": str(args.output)}, sort_keys=True))
            return
        for path in (args.output, metrics_path, ledger_path):
            path.unlink(missing_ok=True)
    AtomicSafetensorsWriter.discard_stale_partial(args.output)
    existing_outputs = [
        path for path in (args.output, metrics_path, ledger_path) if path.exists()
    ]
    if existing_outputs:
        parser.error(
            "output exists without a complete matching receipt: "
            + ", ".join(str(path) for path in existing_outputs)
        )
    _atomic_json(receipt_path, {"complete": False, "signature": signature})

    materialized_started = time.time()
    rows = materialize_layer_rows(
        args.capture,
        args.layer,
        fields=("input",),
        verify_hashes=args.verify_capture_hashes,
    )
    materialized_seconds = time.time() - materialized_started
    quantizer_module = load_qsrt_encoder(args.exllamav3_root)
    install_sqg_quantizer(quantizer_module)
    result_queue: queue.Queue[object] = queue.Queue(maxsize=2 * len(args.devices))
    stop = threading.Event()
    assignments = [experts[index :: len(args.devices)] for index in range(len(args.devices))]
    store_kwargs: dict[str, object] = {}
    if args.official_repo_dir is not None:
        store_kwargs["repo_dir"] = args.official_repo_dir
    if args.official_revision is not None:
        store_kwargs["revision"] = args.official_revision

    def worker(device_name: str, assigned: tuple[int, ...]) -> None:
        try:
            device = torch.device(device_name)
            store = OfficialMXFP4Store(**store_kwargs)
            with (
                store.open_layer(args.layer, experts=assigned) as layer_store,
                safe_open(source_path, framework="pt", device="cpu") as source_reader,
            ):
                for expert in assigned:
                    if stop.is_set():
                        break
                    tensors, result = _refit_expert(
                        rows=rows,
                        source_reader=source_reader,
                        layer_store=layer_store,
                        quantizer_module=quantizer_module,
                        layer=args.layer,
                        expert=expert,
                        draw=draws[expert],
                        device=device,
                        batch_rows=args.batch_rows,
                        regularization_ratio=args.regularization_ratio,
                        logical_schema=logical_schema,
                        codebook=codebook,
                        tailbite_context=int(pool_manifest.get("tailbite_context", 128)),
                        explicit_validation=(
                            expert == experts[0]
                            or expert == experts[-1]
                            or expert % args.explicit_validation_stride == 0
                        ),
                    )
                    result_queue.put(("result", expert, tensors, result))
        except BaseException as exc:
            stop.set()
            result_queue.put(("error", device_name, exc))
        finally:
            result_queue.put(("done", device_name))

    results: dict[int, dict[str, object]] = {}
    with AtomicSafetensorsWriter(
        args.output,
        _payload_specs(args.layer, experts, source_path),
        metadata={
            "kind": "qsrt_kimi_k3_qsrt_candidate_pool",
            "schema_version": "8",
            "layer": str(args.layer),
            "codebook": codebook,
            "tailbite_context": str(pool_manifest.get("tailbite_context", 128)),
        },
    ) as writer:
        with safe_open(source_path, framework="pt", device="cpu") as source_reader:
            for expert in experts:
                for matrix in ("w1", "w3"):
                    for part in PARTS:
                        name = candidate_tensor_name(args.layer, expert, matrix, part)
                        writer.write(name, source_reader.get_tensor(name).contiguous())
        with ThreadPoolExecutor(max_workers=len(args.devices)) as executor:
            futures = [
                executor.submit(worker, device, assigned)
                for device, assigned in zip(args.devices, assignments, strict=True)
            ]
            done = 0
            failure: BaseException | None = None
            while done < len(futures):
                item = result_queue.get()
                if item[0] == "done":
                    done += 1
                    continue
                if item[0] == "error":
                    failure = RuntimeError(f"worker {item[1]} failed: {item[2]}")
                    stop.set()
                    continue
                _kind, expert, tensors, result = item
                for part in PARTS:
                    writer.write(
                        candidate_tensor_name(args.layer, expert, "w2", part),
                        tensors[part],
                    )
                results[expert] = result
                print(
                    json.dumps(
                        {
                            "layer": args.layer,
                            "expert": expert,
                            "complete_experts": len(results),
                            "selected": result["selected"],
                            "relative_improvement": result["relative_improvement"],
                            "elapsed_seconds": result["elapsed_seconds"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            for future in futures:
                future.result()
            if failure is not None:
                raise failure
        if set(results) != set(experts):
            raise RuntimeError("layer workers did not produce every requested expert")

    selected_refits = sum(value["selected"] == "functional_refit" for value in results.values())
    baseline_sse = sum(float(value["baseline_sse"]) for value in results.values())
    selected_sse = sum(float(value["selected_sse"]) for value in results.values())
    ordered = [results[expert] for expert in experts]
    fixed_modes = torch.full((len(experts),), K2.mode_id, dtype=torch.uint8)
    pooled_metrics = {
        "expert_ids": torch.tensor(experts, dtype=torch.int16),
        "mode_ids": torch.tensor([K2.mode_id], dtype=torch.uint8),
        "selected_r13": fixed_modes.clone(),
        "selected_r2": fixed_modes.clone(),
        "proposed_r13": fixed_modes.clone(),
        "proposed_r2": fixed_modes.clone(),
        "format_evaluated": torch.ones((len(experts), 1, 1), dtype=torch.bool),
        "pooled_baseline_sse": torch.tensor(
            [value["baseline_sse"] for value in ordered], dtype=torch.float64
        ),
        "pooled_replacement_sse": torch.tensor(
            [value["replacement_sse"] for value in ordered], dtype=torch.float64
        ),
        "pooled_selected_sse": torch.tensor(
            [value["selected_sse"] for value in ordered], dtype=torch.float64
        ),
        "pooled_source_energy": torch.tensor(
            [value["source_energy"] for value in ordered], dtype=torch.float64
        ),
        "official_source_excess_sse": torch.tensor(
            [value["selected_sse"] for value in ordered], dtype=torch.float64
        ),
        "routed_occurrences": torch.tensor(
            [value["routed_occurrences"] for value in ordered], dtype=torch.int64
        ),
        "effective_sample_size": torch.tensor(
            [value["effective_sample_size"] for value in ordered], dtype=torch.float64
        ),
        "selected_functional_refit": torch.tensor(
            [value["selected"] == "functional_refit" for value in ordered],
            dtype=torch.bool,
        ),
        "coupled_draw_selected": torch.tensor(
            [value["draw"] for value in ordered], dtype=torch.uint8
        ),
    }
    _atomic_safetensors(
        metrics_path,
        pooled_metrics,
        metadata={
            "kind": POOLED_FIXED_PROFILE_METRICS_KIND,
            "schema_version": str(POOLED_FIXED_PROFILE_METRICS_SCHEMA_VERSION),
            "layer": str(args.layer),
        },
    )
    draw_histogram = {
        str(draw): sum(int(value["draw"]) == draw for value in ordered)
        for draw in sorted({int(value["draw"]) for value in ordered})
    }
    selection_contract = {
        "mode_ids": [K2.mode_id],
        **pooled_fixed_profile_selection_contract((K2.mode_id,)),
    }
    rotation_plan = rotations.to_json()
    ledger = {
        "kind": "qsrt_kimi_k3_qsrt_candidate_pool",
        "schema_version": 8,
        "complete": True,
        "layer": args.layer,
        "all_experts": experts == tuple(range(896)),
        "experts": list(experts),
        "payload": args.output.name,
        "metrics": metrics_path.name,
        "source_revision": pool_manifest["source_revision"],
        "logical_trellis_schema": logical_schema,
        "codebook": codebook,
        "tailbite_context": int(pool_manifest.get("tailbite_context", 128)),
        "coupled_rotation": {
            "source": "model_rotation_plan",
            "plan_sha256": _json_sha256(rotation_plan),
        },
        "selection_contract": selection_contract,
        "damage_contract": {
            "metric": "official_source_excess_sse",
            "population": "all_captured_natural_routes",
            "route_weighting": "applied_gate_squared_once",
        },
        "selected_format_histogram": {K2.name: len(experts)},
        "selected_coupled_draw_histogram": draw_histogram,
        "selections": {
            str(expert): {
                "selected_down_target": (
                    "functional_ridge"
                    if results[expert]["selected"] == "functional_refit"
                    else "source_pool"
                ),
                "coupled_draw": int(results[expert]["draw"]),
                "routed_occurrences": int(results[expert]["routed_occurrences"]),
                "effective_sample_size": float(
                    results[expert]["effective_sample_size"]
                ),
                "baseline_sse": float(results[expert]["baseline_sse"]),
                "replacement_sse": float(results[expert]["replacement_sse"]),
                "selected_sse": float(results[expert]["selected_sse"]),
                "source_energy": float(results[expert]["source_energy"]),
            }
            for expert in experts
        },
    }
    _atomic_json(ledger_path, ledger)
    if experts == tuple(range(896)):
        validate_candidate_layer_payload(
            args.output,
            layer=args.layer,
            codebook=codebook,
            tailbite_context=int(pool_manifest.get("tailbite_context", 128)),
            mode_ids=(K2.mode_id,),
        )
    validate_pooled_fixed_profile_metrics(
        pooled_metrics,
        mode_ids=(K2.mode_id,),
        expert_ids=experts,
        expected_coupled_draws=tuple(draws[expert] for expert in experts),
    )
    validate_pooled_fixed_profile_ledger_evidence(
        ledger,
        pooled_metrics,
        expert_ids=experts,
        logical_trellis_schema=logical_schema,
    )
    receipt = {
        "complete": True,
        "signature": signature,
        "devices": list(args.devices),
        "output_sha256": _sha256(args.output),
        "metrics_sha256": _sha256(metrics_path),
        "selection_sha256": _sha256(ledger_path),
        "materialized_layer_seconds": materialized_seconds,
        "result": {
            "experts": len(experts),
            "selected_functional_refits": selected_refits,
            "selected_source_pool": len(experts) - selected_refits,
            "baseline_sse": baseline_sse,
            "selected_sse": selected_sse,
            "pooled_relative_improvement": (baseline_sse - selected_sse) / baseline_sse,
            "per_expert": {str(expert): results[expert] for expert in sorted(results)},
        },
    }
    _atomic_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "complete": str(args.output),
                "experts": len(experts),
                "selected_functional_refits": selected_refits,
                "pooled_relative_improvement": receipt["result"]["pooled_relative_improvement"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

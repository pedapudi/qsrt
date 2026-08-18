#!/usr/bin/env python3
"""Build every fixed-K2 candidate layer with pooled functional W2 refits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Mapping

from qsrt import constants as C
from qsrt.pack.qsrt_candidates import (
    CANDIDATE_POOL_KIND,
    CANDIDATE_POOL_SCHEMA_VERSION,
    OFFICIAL_SOURCE_DAMAGE_METRIC,
)
from qsrt.pack.qsrt_pool import pooled_fixed_profile_selection_contract
from qsrt.qsrt import K2, SCHEMA
from qsrt.qsrt_coupled_plan import CoupledRotationPlan


DEFAULT_DEVICE_GROUPS = (
    "cuda:0,cuda:1,cuda:2",
    "cuda:3,cuda:4,cuda:5",
    "cuda:6,cuda:7,cuda:8",
    "cuda:9,cuda:10,cuda:11",
)


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


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _manifest(args: argparse.Namespace) -> dict[str, object]:
    source_pool = _read_json(args.candidate_pool / "qsrt-candidate-manifest.json")
    source_model = _read_json(args.source_model / "qsrt-manifest.json")
    capture_manifest = args.capture / "manifest.json"
    raw_rotation = source_model.get("coupled_k2_rotation_plan")
    if isinstance(raw_rotation, dict) and raw_rotation.get("kind") == (
        "kquant_qsrt_k2_coupled_rotation_plan"
    ):
        raw_rotation = {**raw_rotation, "kind": "qsrt_k2_coupled_rotation_plan"}
    rotation_plan = CoupledRotationPlan.from_json(raw_rotation).to_json()
    contract = pooled_fixed_profile_selection_contract((K2.mode_id,))
    return {
        "kind": CANDIDATE_POOL_KIND,
        "schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
        "source_model": C.MODEL_ID,
        "source_revision": C.REVISION,
        "source_candidate_pool": str(args.candidate_pool.resolve()),
        "source_candidate_pool_manifest_sha256": _sha256(
            args.candidate_pool / "qsrt-candidate-manifest.json"
        ),
        "source_checkpoint": str(args.source_model.resolve()),
        "source_checkpoint_manifest_sha256": _sha256(
            args.source_model / "qsrt-manifest.json"
        ),
        "capture": str(args.capture.resolve()),
        "capture_manifest_sha256": _sha256(capture_manifest),
        "capture_population": "all_captured_natural_routes",
        "damage_metric": OFFICIAL_SOURCE_DAMAGE_METRIC,
        "damage_weighting": (
            "all captured natural-routing occurrences with applied gate squared once"
        ),
        "damage_provenance": {
            "capture": str(args.capture.resolve()),
            "capture_manifest_sha256": _sha256(capture_manifest),
        },
        "damage_already_natural_route_and_gate_weighted": True,
        "mode_ids": [K2.mode_id],
        "format_grid": contract["format_grid"],
        "shared_r": contract["shared_r"],
        "logical_trellis_schema": SCHEMA,
        "codebook": source_pool["codebook"],
        "tailbite_context": int(source_pool["tailbite_context"]),
        "layout": source_pool["layout"],
        "ldlq_tf32": bool(source_pool["ldlq_tf32"]),
        "coupled_rotation": {
            "source": "model_rotation_plan",
            "plan": rotation_plan,
        },
        "down_target": {
            "candidate": "regularized_functional_refit",
            "fallback": "source_pool_payload",
            "selection_metric": OFFICIAL_SOURCE_DAMAGE_METRIC,
            "regularization_ratio": args.regularization_ratio,
        },
        "encoder": {
            "batch_rows": args.batch_rows,
            "explicit_validation_stride": args.explicit_validation_stride,
            "layout": source_pool["layout"],
            "ldlq_tf32": bool(source_pool["ldlq_tf32"]),
        },
        "h2_contract": {
            "basis": "decoded_candidate_post_situ",
            "population": "all_captured_natural_routes",
            "route_weighting": "applied_gate_squared_once",
            "shrinkage_policy": "weighted_oas_scaled_identity",
            "prior": "expert_local_trace_scaled_identity",
        },
        "selection_contract": contract,
    }


def _complete_layer(destination: Path, layer: int) -> bool:
    stem = destination / "candidates" / f"qsrt-layer-{layer:05d}"
    paths = (
        stem.with_suffix(".safetensors"),
        stem.with_suffix(".metrics.safetensors"),
        stem.with_suffix(".selection.json"),
        stem.with_suffix(".build.json"),
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        return False
    try:
        return bool(_read_json(paths[-1]).get("complete", False))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument(
        "--device-groups", default=";".join(DEFAULT_DEVICE_GROUPS)
    )
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--regularization-ratio", type=float, default=1e-2)
    parser.add_argument("--explicit-validation-stride", type=int, default=128)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    args = parser.parse_args()
    groups = tuple(group for group in args.device_groups.split(";") if group)
    if not groups or args.maximum_attempts < 1:
        parser.error("device groups and maximum attempts must be nonempty and positive")
    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "candidates").mkdir(exist_ok=True)
    (args.dest / "logs").mkdir(exist_ok=True)
    manifest_path = args.dest / "qsrt-candidate-manifest.json"
    expected_manifest = _manifest(args)
    if manifest_path.is_file():
        if _read_json(manifest_path) != expected_manifest:
            parser.error("destination manifest belongs to another build")
    else:
        _atomic_json(manifest_path, expected_manifest)

    pending: queue.Queue[tuple[int, int]] = queue.Queue()
    completed = {layer for layer in C.MOE_LAYERS if _complete_layer(args.dest, layer)}
    for layer in C.MOE_LAYERS:
        if layer not in completed:
            pending.put((layer, 1))
    lock = threading.Lock()
    failures: dict[int, str] = {}
    started = time.time()
    run_id = uuid.uuid4().hex[:12]

    def write_status(running: Mapping[str, int]) -> None:
        with lock:
            document = {
                "complete": len(completed) == C.NUM_MOE_LAYERS and not failures,
                "completed_layers": sorted(completed),
                "failed_layers": failures,
                "running": dict(running),
                "remaining_layers": pending.qsize(),
                "elapsed_seconds": time.time() - started,
            }
            # Serialize the replace with the snapshot.  Otherwise an older
            # snapshot can win the final rename when several lanes report at
            # nearly the same time.
            _atomic_json(args.dest / "build-status.json", document)

    running: dict[str, int] = {}

    def lane(initial_devices: str) -> None:
        devices = initial_devices
        while True:
            try:
                layer, attempt = pending.get_nowait()
            except queue.Empty:
                return
            with lock:
                running[initial_devices] = layer
            write_status(running)
            log_path = args.dest / "logs" / (
                f"qsrt-layer-{layer:05d}-attempt-{attempt}-run-{run_id}.log"
            )
            output = args.dest / "candidates" / f"qsrt-layer-{layer:05d}.safetensors"
            command = [
                sys.executable,
                str(Path(__file__).with_name("refit_pooled_qsrt_layer.py")),
                "--capture",
                str(args.capture),
                "--candidate-pool",
                str(args.candidate_pool),
                "--source-model",
                str(args.source_model),
                "--layer",
                str(layer),
                "--devices",
                devices,
                "--batch-rows",
                str(args.batch_rows),
                "--regularization-ratio",
                str(args.regularization_ratio),
                "--explicit-validation-stride",
                str(args.explicit_validation_stride),
                "--output",
                str(output),
            ]
            layer_started = time.time()
            with log_path.open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if result.returncode == 0 and _complete_layer(args.dest, layer):
                with lock:
                    completed.add(layer)
                print(
                    json.dumps(
                        {
                            "layer": layer,
                            "devices": devices,
                            "elapsed_seconds": time.time() - layer_started,
                            "completed_layers": len(completed),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            else:
                tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
                if "worker cuda:10 failed" in tail:
                    devices = ",".join(
                        device for device in devices.split(",") if device != "cuda:10"
                    )
                if attempt < args.maximum_attempts:
                    pending.put((layer, attempt + 1))
                else:
                    with lock:
                        failures[layer] = tail
                    print(
                        json.dumps(
                            {"layer": layer, "attempts": attempt, "failed": str(log_path)},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            with lock:
                running.pop(initial_devices, None)
            pending.task_done()
            write_status(running)

    threads = [threading.Thread(target=lane, args=(group,)) for group in groups]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    write_status(running)
    if failures or len(completed) != C.NUM_MOE_LAYERS:
        raise SystemExit(
            f"pool incomplete: {len(completed)}/{C.NUM_MOE_LAYERS} layers; "
            f"failed={sorted(failures)}"
        )
    print(json.dumps({"complete": str(args.dest), "layers": len(completed)}))


if __name__ == "__main__":
    main()

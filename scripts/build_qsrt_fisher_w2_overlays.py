#!/usr/bin/env python3
"""Build resumable uniform-K2 W2 payload overlays.

Each GPU processes complete MoE layers.  Every layer produces a safetensors
overlay containing format-preserving uniform-K2 W2 payloads; W1 and W3 remain
unchanged.  Layer outputs are atomic and an identical build can resume without
re-encoding completed layers.
"""

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


DEFAULT_FACTOR_ARCHIVE = Path(
    "/data/datasets/kquant/hessians/"
    "k3-official-mxfp4-final-logit-fisher-100k-v1-output-factors"
)
DEFAULT_FIT_CACHE = Path(
    "/data/datasets/kquant/captures/"
    "k3-denseh-broad-v7-4m-train-input-v1.kqsamples"
)
DEFAULT_FIT_REPORT = Path(
    "/home/luke/projects/qsrt/out/"
    "k3-denseh-broad-v7-4m-train-corpus.json"
)
DEFAULT_EVALUATION_CACHE = Path(
    "/data/datasets/kquant/captures/"
    "k3-codec-diverse-validation-v3-128k-input-v1.kqsamples"
)
DEFAULT_EVALUATION_REPORT = Path(
    "/home/luke/projects/qsrt/out/"
    "k3-codec-diverse-validation-v3-128k-corpus.json"
)
DEFAULT_PROFILE = Path(
    "/data/releases/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-v1"
)
DEFAULT_CANDIDATE_POOL = Path(
    "/data/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-CANDIDATES-v1"
)


def _parse_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in raw.split(",") if value)
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must be unique comma-separated integers")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _validate_cuda_devices(devices: tuple[int, ...]) -> None:
    script = """
import sys
import torch

for raw in sys.argv[1:]:
    device = int(raw)
    torch.cuda.get_device_properties(device)
    torch.empty(1, device=f"cuda:{device}")
    torch.cuda.synchronize(device)
"""
    process = subprocess.run(
        [sys.executable, "-c", script, *(str(device) for device in devices)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if process.returncode:
        detail = process.stdout.strip()
        raise RuntimeError(
            f"one or more requested CUDA devices failed allocation preflight: {detail}"
        )


def _layer_completion(root: Path, layer: int) -> Path:
    return root / "layers" / f"layer-{layer:03d}" / "completion.json"


def _complete_layer(root: Path, layer: int, *, verify_hash: bool) -> bool:
    completion_path = _layer_completion(root, layer)
    if not completion_path.is_file():
        return False
    try:
        completion = _read_json(completion_path)
        result_path = Path(str(completion["result"]))
        overlay_path = Path(str(completion["payload_overlay"]))
        result = _read_json(result_path)
        expected_hash = str(completion["payload_overlay_sha256"])
        if not bool(result.get("complete")):
            return False
        if str(result.get("payload_overlay_sha256")) != expected_hash:
            return False
        if not overlay_path.is_file() or overlay_path.stat().st_size == 0:
            return False
        return not verify_hash or _sha256(overlay_path) == expected_hash
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _build_config(args: argparse.Namespace) -> dict[str, object]:
    uses_decoded_h2 = args.input_factor_mode == "decoded_h2"
    return {
        "kind": (
            "qsrt_uniform_k2_direct_viterbi_w2_overlays"
            if args.direct_viterbi
            else "qsrt_uniform_k2_identity_input_final_logit_fisher_w2_overlays"
            if not uses_decoded_h2
            else "qsrt_uniform_k2_final_logit_fisher_w2_overlays"
        ),
        "schema_version": 1,
        "layers": list(args.layers),
        "experts_per_layer": C.NUM_EXPERTS,
        "output_factor_archive": (
            None if args.direct_viterbi else str(args.output_factor_archive.resolve())
        ),
        "output_factor_archive_manifest_sha256": (
            None
            if args.direct_viterbi
            else _sha256(args.output_factor_archive / "manifest.json")
        ),
        "fit_cache": (
            str(args.fit_cache.resolve())
            if not args.direct_viterbi and uses_decoded_h2
            else None
        ),
        "fit_cache_manifest_sha256": (
            None
            if args.direct_viterbi or not uses_decoded_h2
            else _sha256(args.fit_cache / "manifest.json")
        ),
        "fit_report": (
            str(args.fit_report.resolve())
            if not args.direct_viterbi and uses_decoded_h2
            else None
        ),
        "fit_report_sha256": (
            None
            if args.direct_viterbi or not uses_decoded_h2
            else _sha256(args.fit_report)
        ),
        "evaluation_cache": (
            None if args.direct_viterbi else str(args.evaluation_cache.resolve())
        ),
        "evaluation_cache_manifest_sha256": (
            None
            if args.direct_viterbi
            else _sha256(args.evaluation_cache / "manifest.json")
        ),
        "evaluation_report": (
            None if args.direct_viterbi else str(args.evaluation_report.resolve())
        ),
        "evaluation_report_sha256": (
            None if args.direct_viterbi else _sha256(args.evaluation_report)
        ),
        "profile": str(args.profile.resolve()),
        "profile_completion_sha256": _sha256(
            args.profile / "qsrt-completion.json"
        ),
        "candidate_pool": str(args.candidate_pool.resolve()),
        "candidate_pool_manifest_sha256": _sha256(
            args.candidate_pool / "qsrt-candidate-manifest.json"
        ),
        "output_factor_mode": args.output_factor_mode,
        "input_factor_mode": args.input_factor_mode,
        "direct_viterbi": args.direct_viterbi,
        "output_damping_ratio": args.output_damping_ratio,
        "expert_batch_size": args.expert_batch_size,
        "tailbite_context": args.tailbite_context,
        "encoder": str(
            Path(__file__).with_name("run_qsrt_two_sided_w2_pilot.py").resolve()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument(
        "--output-factor-archive", type=Path, default=DEFAULT_FACTOR_ARCHIVE
    )
    parser.add_argument("--fit-cache", type=Path, default=DEFAULT_FIT_CACHE)
    parser.add_argument("--fit-report", type=Path, default=DEFAULT_FIT_REPORT)
    parser.add_argument(
        "--evaluation-cache", type=Path, default=DEFAULT_EVALUATION_CACHE
    )
    parser.add_argument(
        "--evaluation-report", type=Path, default=DEFAULT_EVALUATION_REPORT
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--candidate-pool", type=Path, default=DEFAULT_CANDIDATE_POOL
    )
    parser.add_argument(
        "--devices", type=_parse_ints, default=tuple(range(12))
    )
    parser.add_argument(
        "--layers", type=_parse_ints, default=tuple(C.MOE_LAYERS)
    )
    parser.add_argument("--output-damping-ratio", type=float, default=3.0)
    parser.add_argument(
        "--output-factor-mode",
        choices=("fisher", "inverse_fisher"),
        default="fisher",
    )
    parser.add_argument(
        "--input-factor-mode",
        choices=("decoded_h2", "identity"),
        default="decoded_h2",
    )
    parser.add_argument("--expert-batch-size", type=int, default=16)
    parser.add_argument("--tailbite-context", type=int, default=128)
    parser.add_argument("--maximum-attempts", type=int, default=2)
    parser.add_argument(
        "--direct-viterbi",
        action="store_true",
        help="encode independent SQG tiles without decoded-H2 or Fisher feedback",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if any(layer not in C.MOE_LAYERS for layer in args.layers):
        parser.error("layers must lie in the routed-MoE layer set")
    if any(device < 0 for device in args.devices):
        parser.error("devices must be nonnegative")
    if args.expert_batch_size < 1 or args.maximum_attempts < 1:
        parser.error("batch size and maximum attempts must be positive")
    if args.direct_viterbi and args.input_factor_mode != "decoded_h2":
        parser.error("direct Viterbi does not consume an input factor")
    if not args.direct_viterbi and not args.output_factor_archive.is_dir():
        parser.error("output-factor archive does not exist")
    required = [
        args.profile / "qsrt-completion.json",
        args.candidate_pool / "qsrt-candidate-manifest.json",
    ]
    if not args.direct_viterbi:
        required.extend(
            (
                args.evaluation_cache / "manifest.json",
                args.evaluation_report,
            )
        )
    if not args.direct_viterbi and args.input_factor_mode == "decoded_h2":
        required.extend((args.fit_cache / "manifest.json", args.fit_report))
    if any(not path.exists() for path in required):
        parser.error("one or more frozen encoder inputs do not exist")
    if args.dest.exists() and not args.resume:
        parser.error("destination exists; pass --resume for an identical build")
    try:
        _validate_cuda_devices(args.devices)
    except RuntimeError as error:
        parser.error(str(error))

    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "layers").mkdir(exist_ok=True)
    (args.dest / "logs").mkdir(exist_ok=True)
    config_path = args.dest / "build-config.json"
    expected_config = _build_config(args)
    if config_path.exists():
        if _read_json(config_path) != expected_config:
            parser.error("destination belongs to a different W2 overlay build")
    else:
        _atomic_json(config_path, expected_config)

    completed = {
        layer
        for layer in args.layers
        if _complete_layer(args.dest, layer, verify_hash=args.resume)
    }
    pending: queue.Queue[tuple[int, int]] = queue.Queue()
    for layer in args.layers:
        if layer not in completed:
            pending.put((layer, 1))

    failures: dict[int, str] = {}
    running: dict[int, int] = {}
    unavailable_devices: set[int] = set()
    lock = threading.RLock()
    started = time.time()
    run_id = uuid.uuid4().hex[:12]
    experts = ",".join(str(expert) for expert in range(C.NUM_EXPERTS))

    def write_status() -> None:
        with lock:
            status = {
                "complete": len(completed) == len(args.layers) and not failures,
                "completed_layers": sorted(completed),
                "failed_layers": dict(failures),
                "running": {str(device): layer for device, layer in running.items()},
                "unavailable_devices": sorted(unavailable_devices),
                "remaining_layers": pending.qsize(),
                "elapsed_seconds": time.time() - started,
            }
            _atomic_json(args.dest / "build-status.json", status)

    def lane(device: int) -> None:
        while True:
            try:
                layer, attempt = pending.get_nowait()
            except queue.Empty:
                return
            with lock:
                running[device] = layer
            write_status()

            attempt_root = (
                args.dest
                / "layers"
                / f"layer-{layer:03d}"
                / f"attempt-{attempt}-{run_id}"
            )
            attempt_root.mkdir(parents=True, exist_ok=False)
            result_path = attempt_root / "result.json"
            overlay_path = attempt_root / "w2-overlay.safetensors"
            log_path = args.dest / "logs" / (
                f"layer-{layer:03d}-attempt-{attempt}-{run_id}.log"
            )
            command = [
                sys.executable,
                str(Path(__file__).with_name("run_qsrt_two_sided_w2_pilot.py")),
                "--layer",
                str(layer),
                "--experts",
                experts,
                "--output-factor-archive",
                str(args.output_factor_archive),
                "--fit-cache",
                str(args.fit_cache),
                "--fit-report",
                str(args.fit_report),
                "--evaluation-cache",
                str(args.evaluation_cache),
                "--evaluation-report",
                str(args.evaluation_report),
                "--profile",
                str(args.profile),
                "--candidate-pool",
                str(args.candidate_pool),
                "--output-damping-ratio",
                str(args.output_damping_ratio),
                "--output-factor-mode",
                args.output_factor_mode,
                "--input-factor-mode",
                args.input_factor_mode,
                "--expert-batch-size",
                str(args.expert_batch_size),
                "--tailbite-context",
                str(args.tailbite_context),
                "--maximum-evaluation-rows",
                "1",
                "--device",
                f"cuda:{device}",
                "--payload-only",
                "--payload-overlay",
                str(overlay_path),
                "--output",
                str(result_path),
            ]
            if args.direct_viterbi:
                command.append("--direct-viterbi")
            layer_started = time.time()
            with log_path.open("w") as log:
                process = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            failure = None
            if process.returncode:
                failure = f"encoder exited with status {process.returncode}"
            elif not result_path.is_file() or not overlay_path.is_file():
                failure = "encoder did not produce both layer outputs"
            else:
                result = _read_json(result_path)
                expected_hash = str(result.get("payload_overlay_sha256"))
                if not bool(result.get("complete")) or len(result.get("experts", {})) != (
                    C.NUM_EXPERTS
                ):
                    failure = "encoder result is incomplete"
                elif not expected_hash or expected_hash == "None":
                    failure = "encoder result lacks the overlay identity"
                elif _sha256(overlay_path) != expected_hash:
                    failure = "overlay identity does not match the encoder result"
                else:
                    completion = {
                        "kind": (
                            "qsrt_uniform_k2_direct_viterbi_w2_layer"
                            if args.direct_viterbi
                            else "qsrt_uniform_k2_identity_input_final_logit_fisher_w2_layer"
                            if args.input_factor_mode == "identity"
                            else "qsrt_uniform_k2_final_logit_fisher_w2_layer"
                        ),
                        "schema_version": 1,
                        "layer": layer,
                        "experts": C.NUM_EXPERTS,
                        "device": device,
                        "result": str(result_path.resolve()),
                        "payload_overlay": str(overlay_path.resolve()),
                        "payload_overlay_sha256": expected_hash,
                        "seconds": time.time() - layer_started,
                    }
                    _atomic_json(_layer_completion(args.dest, layer), completion)
                    with lock:
                        completed.add(layer)
                    print(
                        json.dumps(
                            {
                                "layer": layer,
                                "device": device,
                                "seconds": completion["seconds"],
                                "completed_layers": len(completed),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

            if failure is not None:
                tail = "\n".join(
                    log_path.read_text(errors="replace").splitlines()[-30:]
                )
                if "invalid device ordinal" in tail:
                    pending.put((layer, attempt))
                    with lock:
                        unavailable_devices.add(device)
                    print(
                        json.dumps(
                            {
                                "device": device,
                                "failure": "CUDA device became unavailable",
                                "layer_requeued": layer,
                                "log": str(log_path),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                elif attempt < args.maximum_attempts:
                    pending.put((layer, attempt + 1))
                else:
                    with lock:
                        failures[layer] = f"{failure}\n{tail}"
                    print(
                        json.dumps(
                            {
                                "layer": layer,
                                "device": device,
                                "failure": failure,
                                "log": str(log_path),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            with lock:
                running.pop(device, None)
            pending.task_done()
            write_status()
            if device in unavailable_devices:
                return

    write_status()
    threads = [threading.Thread(target=lane, args=(device,)) for device in args.devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    write_status()

    if failures or len(completed) != len(args.layers):
        raise SystemExit(
            f"W2 overlay build incomplete: {len(completed)}/{len(args.layers)}; "
            f"failed={sorted(failures)}"
        )
    print(json.dumps({"complete": str(args.dest), "layers": len(completed)}))


if __name__ == "__main__":
    main()

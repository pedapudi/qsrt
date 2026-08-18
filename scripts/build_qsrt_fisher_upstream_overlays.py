#!/usr/bin/env python3
"""Build resumable uniform-K2 expert-matrix payload overlays.

Direct-Viterbi builds may encode W1, W3, and W2 in one layer worker. Curvature
and gradient modes retain the W1/W3-only interface because W2 requires its own
decoded-upstream input factor.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import torch

from qsrt import constants as C


DEFAULT_PROFILE = Path(
    "/data/releases/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-v1"
)
DEFAULT_CANDIDATE_POOL = Path(
    "/data/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-CANDIDATES-v1"
)
DEFAULT_HESSIANS = Path(
    "/data/datasets/kquant/hessians/"
    "k3-denseh-broad-v7-4m-train-h13-identity-qsrt-v1.kqhess"
)
DEFAULT_FACTORS = Path(
    "/data/datasets/kquant/hessians/"
    "k3-official-mxfp4-final-logit-fisher-100k-v1-upstream-factors"
)


def _parse_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in raw.split(",") if value)
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must be unique comma-separated integers")
    return values


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _completion_path(root: Path, layer: int) -> Path:
    return root / "layers" / f"layer-{layer:03d}" / "completion.json"


def _completed_overlay(root: Path, layer: int) -> Path:
    path = _completion_path(root, layer)
    if not path.is_file():
        raise FileNotFoundError(
            f"gradient anchor has no completed layer {layer}: {path}"
        )
    completion = _read_json(path)
    overlay = Path(str(completion["payload_overlay"])).expanduser().resolve()
    if not overlay.is_file():
        raise FileNotFoundError(
            f"gradient anchor layer {layer} payload is missing: {overlay}"
        )
    return overlay


def _layer_is_complete(root: Path, layer: int) -> bool:
    path = _completion_path(root, layer)
    if not path.is_file():
        return False
    try:
        completion = _read_json(path)
        overlay = Path(str(completion["payload_overlay"]))
        result = _read_json(Path(str(completion["result"])))
        return (
            bool(result.get("complete"))
            and int(result.get("experts", 0)) == C.NUM_EXPERTS
            and overlay.is_file()
            and overlay.stat().st_size == int(result.get("payload_bytes", -1))
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--devices", type=_parse_ints, default=tuple(range(12)))
    parser.add_argument("--layers", type=_parse_ints, default=tuple(C.MOE_LAYERS))
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--candidate-pool", type=Path, default=DEFAULT_CANDIDATE_POOL
    )
    parser.add_argument("--hessians", type=Path, default=DEFAULT_HESSIANS)
    parser.add_argument("--factor-archive", type=Path, default=DEFAULT_FACTORS)
    parser.add_argument(
        "--exllamav3-root", type=Path, default=Path("/home/luke/projects/exllamav3")
    )
    parser.add_argument("--expert-batch-size", type=int, default=32)
    parser.add_argument("--output-damping-ratio", type=float, default=3.0)
    parser.add_argument("--tailbite-context", type=int, default=128)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    parser.add_argument(
        "--direct-viterbi",
        action="store_true",
        help="encode independent SQG tiles without dense-H or Fisher feedback",
    )
    parser.add_argument(
        "--include-w2",
        action="store_true",
        help="encode W1, W3, and W2 from one resident source bank",
    )
    parser.add_argument(
        "--gradient-strength",
        type=float,
        default=0.0,
        help="Scale applied to the deterministic final-logit KL gradient.",
    )
    parser.add_argument(
        "--gradient-strength-normalization",
        choices=("none", "h13_mean_diagonal"),
        default="none",
        help=(
            "Interpret the gradient strength directly or multiply it by each "
            "layer H13 factor's mean diagonal."
        ),
    )
    parser.add_argument(
        "--gradient-core-rcond",
        type=float,
        default=1.0e-3,
        help="Relative singular-value cutoff for the gradient sketch core.",
    )
    parser.add_argument(
        "--gradient-anchor-id",
        help="Durable identity of the checkpoint at which the gradient was captured.",
    )
    parser.add_argument(
        "--gradient-objective-id",
        help="Durable identity of the deterministic final-logit objective.",
    )
    parser.add_argument(
        "--gradient-anchor-overlay-root",
        type=Path,
        help=(
            "Completed upstream-overlay build supplying each layer's W1/W3 "
            "anchor payload. The sealed candidate pool is used when omitted."
        ),
    )
    parser.add_argument("--gradient-refinement-sweeps", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if any(layer not in C.MOE_LAYERS for layer in args.layers):
        raise ValueError("layers must lie in Kimi-K3's routed-MoE layer set")
    if any(device < 0 or device >= torch.cuda.device_count() for device in args.devices):
        raise ValueError("device lies outside the visible CUDA inventory")
    if (
        args.expert_batch_size < 1
        or args.maximum_attempts < 1
        or args.gradient_refinement_sweeps < 1
    ):
        raise ValueError("batch size and retry count must be positive")
    if not torch.isfinite(torch.tensor(args.gradient_strength)) or (
        args.gradient_strength < 0.0
    ):
        raise ValueError("gradient strength must be finite and nonnegative")
    if not 0.0 < args.gradient_core_rcond <= 1.0:
        raise ValueError("gradient core rcond must lie in (0, 1]")
    gradient_enabled = args.gradient_strength > 0.0
    if args.direct_viterbi and gradient_enabled:
        raise ValueError("direct Viterbi and gradient refinement are exclusive")
    if args.include_w2 and not args.direct_viterbi:
        raise ValueError(
            "combined W1/W3/W2 encoding currently requires direct Viterbi"
        )
    if gradient_enabled and (
        not args.gradient_anchor_id or not args.gradient_objective_id
    ):
        raise ValueError(
            "gradient guidance requires explicit anchor and objective identities"
        )
    if not gradient_enabled and any(
        value is not None
        for value in (
            args.gradient_anchor_id,
            args.gradient_objective_id,
            args.gradient_anchor_overlay_root,
        )
    ):
        raise ValueError("gradient anchor arguments require positive gradient strength")
    required_paths = [args.profile, args.candidate_pool, args.exllamav3_root]
    if not args.direct_viterbi:
        required_paths.extend((args.hessians, args.factor_archive))
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.gradient_anchor_overlay_root is not None:
        args.gradient_anchor_overlay_root = (
            args.gradient_anchor_overlay_root.expanduser().resolve()
        )
        if not args.gradient_anchor_overlay_root.is_dir():
            raise FileNotFoundError(args.gradient_anchor_overlay_root)
        for layer in args.layers:
            _completed_overlay(args.gradient_anchor_overlay_root, layer)
    if args.dest.exists() and not args.resume:
        raise FileExistsError("destination exists; pass --resume for the same build")

    build_kind = (
        "qsrt_uniform_k2_direct_viterbi_all_linear_overlays"
        if args.include_w2
        else "qsrt_uniform_k2_direct_viterbi_upstream_overlays"
        if args.direct_viterbi
        else "qsrt_uniform_k2_gradient_refined_upstream_overlays"
        if gradient_enabled
        else "qsrt_uniform_k2_final_logit_fisher_upstream_overlays"
    )
    layer_kind = (
        "qsrt_uniform_k2_direct_viterbi_all_linear_layer"
        if args.include_w2
        else "qsrt_uniform_k2_direct_viterbi_upstream_layer"
        if args.direct_viterbi
        else "qsrt_uniform_k2_gradient_refined_upstream_layer"
        if gradient_enabled
        else "qsrt_uniform_k2_final_logit_fisher_upstream_layer"
    )
    config = {
        "kind": build_kind,
        "schema_version": 1,
        "layers": list(args.layers),
        "experts_per_layer": C.NUM_EXPERTS,
        "profile": str(args.profile.resolve()),
        "candidate_pool": str(args.candidate_pool.resolve()),
        "hessians": None if args.direct_viterbi else str(args.hessians.resolve()),
        "factor_archive": (
            None if args.direct_viterbi else str(args.factor_archive.resolve())
        ),
        "exllamav3_root": str(args.exllamav3_root.resolve()),
        "expert_batch_size": args.expert_batch_size,
        "output_damping_ratio": args.output_damping_ratio,
        "tailbite_context": args.tailbite_context,
        "direct_viterbi": args.direct_viterbi,
        "matrices": ["w1", "w3", "w2"] if args.include_w2 else ["w1", "w3"],
        "gradient_guidance": (
            None
            if not gradient_enabled
            else {
                "strength": args.gradient_strength,
                "normalization": args.gradient_strength_normalization,
                "core_rcond": args.gradient_core_rcond,
                "anchor_id": args.gradient_anchor_id,
                "objective_id": args.gradient_objective_id,
                "anchor_overlay_root": (
                    None
                    if args.gradient_anchor_overlay_root is None
                    else str(args.gradient_anchor_overlay_root)
                ),
                "sweeps": args.gradient_refinement_sweeps,
            }
        ),
    }
    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "layers").mkdir(exist_ok=True)
    (args.dest / "logs").mkdir(exist_ok=True)
    config_path = args.dest / "build-config.json"
    if config_path.exists():
        if _read_json(config_path) != config:
            raise ValueError("destination belongs to a different overlay build")
    else:
        _atomic_json(config_path, config)

    completed = {
        layer for layer in args.layers if _layer_is_complete(args.dest, layer)
    }
    pending: queue.Queue[tuple[int, int]] = queue.Queue()
    for layer in args.layers:
        if layer not in completed:
            pending.put((layer, 1))
    failures: dict[int, str] = {}
    running: dict[int, int] = {}
    lock = threading.RLock()
    started = time.monotonic()
    run_id = uuid.uuid4().hex[:12]

    def write_status() -> None:
        with lock:
            _atomic_json(
                args.dest / "build-status.json",
                {
                    "complete": len(completed) == len(args.layers) and not failures,
                    "completed_layers": sorted(completed),
                    "failed_layers": failures,
                    "running": {str(device): layer for device, layer in running.items()},
                    "remaining_layers": pending.qsize(),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )

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
            overlay = attempt_root / (
                "all-linear-overlay.safetensors"
                if args.include_w2
                else "upstream-overlay.safetensors"
            )
            result = attempt_root / "result.json"
            log = args.dest / "logs" / (
                f"layer-{layer:03d}-attempt-{attempt}-{run_id}.log"
            )
            command = [
                sys.executable,
                str(Path(__file__).with_name("encode_qsrt_fisher_upstream_layer.py")),
                "--layer",
                str(layer),
                "--device",
                f"cuda:{device}",
                "--output",
                str(overlay),
                "--result",
                str(result),
                "--profile",
                str(args.profile),
                "--candidate-pool",
                str(args.candidate_pool),
                "--hessians",
                str(args.hessians),
                "--factor-archive",
                str(args.factor_archive),
                "--exllamav3-root",
                str(args.exllamav3_root),
                "--expert-batch-size",
                str(args.expert_batch_size),
                "--output-damping-ratio",
                str(args.output_damping_ratio),
                "--tailbite-context",
                str(args.tailbite_context),
            ]
            if args.direct_viterbi:
                command.append("--direct-viterbi")
            if args.include_w2:
                command.append("--include-w2")
            if gradient_enabled:
                command.extend(
                    [
                        "--gradient-strength",
                        str(args.gradient_strength),
                        "--gradient-strength-normalization",
                        args.gradient_strength_normalization,
                        "--gradient-core-rcond",
                        str(args.gradient_core_rcond),
                        "--gradient-anchor-id",
                        args.gradient_anchor_id,
                        "--gradient-objective-id",
                        args.gradient_objective_id,
                        "--gradient-refinement-sweeps",
                        str(args.gradient_refinement_sweeps),
                    ]
                )
                if args.gradient_anchor_overlay_root is not None:
                    command.extend(
                        [
                            "--gradient-anchor-layer",
                            str(
                                _completed_overlay(
                                    args.gradient_anchor_overlay_root,
                                    layer,
                                )
                            ),
                        ]
                    )
            layer_started = time.monotonic()
            with log.open("w") as stream:
                process = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            failure = None
            if process.returncode:
                failure = f"encoder exited with status {process.returncode}"
            elif not result.is_file() or not overlay.is_file():
                failure = "encoder did not produce both layer outputs"
            else:
                document = _read_json(result)
                if not bool(document.get("complete")) or int(
                    document.get("experts", 0)
                ) != C.NUM_EXPERTS:
                    failure = "encoder result is incomplete"
                elif overlay.stat().st_size != int(document.get("payload_bytes", -1)):
                    failure = "overlay byte count differs from the encoder result"

            if failure is None:
                _atomic_json(
                    _completion_path(args.dest, layer),
                    {
                        "kind": layer_kind,
                        "schema_version": 1,
                        "layer": layer,
                        "device": device,
                        "seconds": time.monotonic() - layer_started,
                        "result": str(result.resolve()),
                        "payload_overlay": str(overlay.resolve()),
                    },
                )
                with lock:
                    completed.add(layer)
                print(
                    json.dumps(
                        {
                            "layer": layer,
                            "device": device,
                            "seconds": time.monotonic() - layer_started,
                            "completed_layers": len(completed),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            elif attempt < args.maximum_attempts:
                pending.put((layer, attempt + 1))
            else:
                tail = "\n".join(log.read_text(errors="replace").splitlines()[-30:])
                with lock:
                    failures[layer] = f"{failure}\n{tail}"
                print(
                    json.dumps(
                        {"layer": layer, "device": device, "failure": failure},
                        sort_keys=True,
                    ),
                    flush=True,
                )
            with lock:
                running.pop(device, None)
            pending.task_done()
            write_status()

    write_status()
    threads = [threading.Thread(target=lane, args=(device,)) for device in args.devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    write_status()
    if failures or len(completed) != len(args.layers):
        raise SystemExit(
            f"upstream overlay build incomplete: {len(completed)}/{len(args.layers)}; "
            f"failed={sorted(failures)}"
        )
    print(json.dumps({"complete": str(args.dest), "layers": len(completed)}))


if __name__ == "__main__":
    main()

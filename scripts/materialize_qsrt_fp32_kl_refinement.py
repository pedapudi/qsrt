#!/usr/bin/env python3
"""Materialize selected FP32 final-KL refinement layers over a QSRT anchor."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import torch

from qsrt import constants as C


DEFAULT_ANCHOR = Path(
    "/data/models/Kimi-K3-QSRT-K2-DIRECT-VITERBI-ALL-LINEARS-v1-model"
)
DEFAULT_CANDIDATES = Path(
    "/data/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-CANDIDATES-v1"
)
DEFAULT_PROFILE = Path(
    "/data/releases/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-v1"
)
ATOM_BYTES = 7_415_300_096


def _parse_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(sorted({int(item) for item in value.split(",") if item}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from error
    if not layers or any(layer not in C.MOE_LAYERS for layer in layers):
        raise argparse.ArgumentTypeError("layers must be routed Kimi-K3 layers")
    return layers


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _overlay_path(root: Path, layer: int) -> Path:
    completion = _read_json(
        root / "layers" / f"layer-{layer:03d}" / "completion.json"
    )
    if completion.get("complete") is not True or int(completion["layer"]) != layer:
        raise ValueError(f"incomplete refinement overlay for layer {layer}")
    path = Path(str(completion["payload_overlay"]))
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def _layer_complete(root: Path, layer: int) -> bool:
    atom = root / "atoms" / f"qsrt-layer-{layer:05d}.safetensors"
    receipt = root / "receipts" / f"layer-{layer:03d}.json"
    if not atom.is_file() or atom.stat().st_size != ATOM_BYTES or not receipt.is_file():
        return False
    document = _read_json(receipt)
    return (
        int(document.get("layer", -1)) == layer
        and document.get("replacement_matrices") == ["w1", "w3", "w2"]
    )


def _publish_model(
    *,
    anchor: Path,
    artifact: Path,
    destination: Path,
    provenance: dict[str, object],
) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    staging = destination.with_name(f".{destination.name}.partial")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-al", "--", str(anchor), str(staging)], check=True)
    try:
        for layer in C.MOE_LAYERS:
            target = staging / f"qsrt-layer-{layer:05d}.safetensors"
            target.unlink()
            os.link(artifact / f"qsrt-layer-{layer:05d}.safetensors", target)
        for name in ("qsrt-manifest.json", "qsrt-completion.json"):
            target = staging / name
            target.unlink(missing_ok=True)
            os.link(artifact / name, target)
        view_path = staging / "qsrt-model-view.json"
        view = _read_json(view_path)
        view_path.unlink()
        view["artifact"] = str(artifact.resolve())
        validation = view.get("artifact_validation")
        if isinstance(validation, dict):
            validation["artifact"] = str(artifact.resolve())
        view["research_payload_overlays"] = provenance
        _atomic_json(view_path, view)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--anchor-model", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--layers", type=_parse_layers, default=(89, 90, 91, 92))
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.jobs <= 0:
        raise ValueError("jobs must be positive")
    overlay_root = args.overlay_root.expanduser().resolve()
    anchor = args.anchor_model.expanduser().resolve()
    candidate_pool = args.candidate_pool.expanduser().resolve()
    profile = args.profile.expanduser().resolve()
    destination = args.dest.expanduser().resolve()
    model_output = args.model_output.expanduser().resolve()
    for path in (overlay_root, anchor, candidate_pool, profile):
        if not path.exists():
            raise FileNotFoundError(path)
    overlays = {layer: _overlay_path(overlay_root, layer) for layer in args.layers}
    if destination.exists() and not args.resume:
        raise FileExistsError("destination exists; pass --resume for the same build")
    if model_output.exists() or model_output.is_symlink():
        raise FileExistsError(model_output)
    if torch.cuda.device_count() < min(args.jobs, len(args.layers)):
        raise RuntimeError("insufficient CUDA devices for requested jobs")

    provenance = {
        "kind": "fp32_final_kl_gradient_refinement",
        "anchor": str(anchor),
        "overlay_root": str(overlay_root),
        "layers": list(args.layers),
    }
    config = {
        "kind": "qsrt_fp32_final_kl_gradient_refinement_materialization",
        "schema_version": 1,
        "anchor_model": str(anchor),
        "candidate_pool": str(candidate_pool),
        "profile": str(profile),
        "payload_overlays": provenance,
        "model_output": str(model_output),
    }
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("atoms", "receipts", "work", "logs"):
        (destination / name).mkdir(exist_ok=True)
    config_path = destination / "build-config.json"
    if config_path.exists():
        if _read_json(config_path) != config:
            raise ValueError("destination belongs to another materialization")
    else:
        _atomic_json(config_path, config)

    pending: queue.Queue[int] = queue.Queue()
    completed = {layer for layer in args.layers if _layer_complete(destination, layer)}
    for layer in args.layers:
        if layer not in completed:
            pending.put(layer)
    failures: dict[int, str] = {}
    lock = threading.Lock()
    run_id = uuid.uuid4().hex[:12]

    def lane(worker: int) -> None:
        while True:
            try:
                layer = pending.get_nowait()
            except queue.Empty:
                return
            started = time.monotonic()
            work = destination / "work" / f"layer-{layer:03d}-{run_id}"
            work.mkdir(parents=True, exist_ok=False)
            atom = work / f"qsrt-layer-{layer:05d}.safetensors"
            receipt = work / "receipt.json"
            log_path = destination / "logs" / f"layer-{layer:03d}-{run_id}.log"
            command = [
                sys.executable,
                str(Path(__file__).with_name("materialize_qsrt_w2_overlay_layer.py")),
                "--candidate-pool",
                str(candidate_pool),
                "--profile",
                str(profile),
                "--layer",
                str(layer),
                "--payload-overlay",
                str(overlays[layer]),
                "--replace-matrices",
                "w1,w3,w2",
                "--work-root",
                str(work / "candidate-work"),
                "--atom-output",
                str(atom),
                "--receipt",
                str(receipt),
                "--batch-size",
                "64",
                "--batched-pure-k2",
                "--materialize-device",
                f"cuda:{worker}",
                "--skip-atom-sync",
                "--skip-content-verification",
            ]
            environment = os.environ.copy()
            environment["OMP_NUM_THREADS"] = "1"
            environment["MKL_NUM_THREADS"] = "1"
            with log_path.open("w") as stream:
                process = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if (
                process.returncode == 0
                and atom.is_file()
                and atom.stat().st_size == ATOM_BYTES
                and receipt.is_file()
            ):
                final_atom = destination / "atoms" / atom.name
                final_receipt = destination / "receipts" / f"layer-{layer:03d}.json"
                os.replace(atom, final_atom)
                os.replace(receipt, final_receipt)
                shutil.rmtree(work, ignore_errors=True)
                with lock:
                    completed.add(layer)
                print(
                    json.dumps(
                        {
                            "layer": layer,
                            "seconds": time.monotonic() - started,
                            "completed_layers": len(completed),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            else:
                tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
                with lock:
                    failures[layer] = (
                        f"materializer status={process.returncode}; {tail}"
                    )
            pending.task_done()

    workers = [
        threading.Thread(target=lane, args=(worker,), daemon=True)
        for worker in range(min(args.jobs, len(args.layers)))
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if failures or completed != set(args.layers):
        raise RuntimeError(
            f"materialization failed; completed={sorted(completed)} failures={failures}"
        )

    for layer in C.MOE_LAYERS:
        target = destination / "atoms" / f"qsrt-layer-{layer:05d}.safetensors"
        if target.exists():
            continue
        os.link(anchor / target.name, target)
    manifest = _read_json(profile / "qsrt-manifest.json")
    manifest["research_payload_overlays"] = provenance
    _atomic_json(destination / "atoms" / "qsrt-manifest.json", manifest)
    _atomic_json(
        destination / "atoms" / "qsrt-completion.json",
        _read_json(profile / "qsrt-completion.json"),
    )
    _publish_model(
        anchor=anchor,
        artifact=destination / "atoms",
        destination=model_output,
        provenance=provenance,
    )
    _atomic_json(
        destination / "completion.json",
        {
            "complete": True,
            "layers": list(args.layers),
            "model": str(model_output),
        },
    )
    print(
        json.dumps(
            {
                "complete": str(destination),
                "model": str(model_output),
                "replaced_layers": list(args.layers),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

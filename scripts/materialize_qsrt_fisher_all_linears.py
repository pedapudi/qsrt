#!/usr/bin/env python3
"""Materialize uniform-K2 W1/W3/W2 payload overlays as a serving model."""

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


DEFAULT_CANDIDATE_POOL = Path(
    "/data/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-CANDIDATES-v1"
)
DEFAULT_PROFILE = Path(
    "/data/releases/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-v1"
)
DEFAULT_W2_OVERLAYS = Path(
    "/data/kquant/research/"
    "k3-uniform-k2-w2-final-logit-fisher-full-d3-batched-v1"
)
DEFAULT_TEMPLATE_MODEL = Path(
    "/data/models/"
    "Kimi-K3-QSRT-K2-W2-FINAL-LOGIT-FISHER-FULL-D3-v1-model"
)


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


def _overlay_from_completion(root: Path, layer: int) -> Path:
    completion = _read_json(
        root / "layers" / f"layer-{layer:03d}" / "completion.json"
    )
    path = Path(str(completion["payload_overlay"]))
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def _layer_complete(root: Path, layer: int) -> bool:
    atom = root / "atoms" / f"qsrt-layer-{layer:05d}.safetensors"
    receipt = root / "receipts" / f"layer-{layer:03d}.json"
    if not atom.is_file() or not receipt.is_file():
        return False
    try:
        document = _read_json(receipt)
        return (
            int(document.get("layer", -1)) == layer
            and document.get("replacement_matrices") == ["w1", "w3", "w2"]
            and atom.stat().st_size == 7_415_300_096
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _publish_model(
    artifact: Path,
    template: Path,
    destination: Path,
    *,
    overlay_provenance: dict[str, object],
) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    staging = destination.with_name(f".{destination.name}.partial")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cp", "-al", "--", str(template), str(staging)],
        check=True,
    )
    try:
        for layer in C.MOE_LAYERS:
            target = staging / f"qsrt-layer-{layer:05d}.safetensors"
            target.unlink()
            os.link(
                artifact / f"qsrt-layer-{layer:05d}.safetensors",
                target,
            )
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
        view["research_payload_overlays"] = overlay_provenance
        _atomic_json(view_path, view)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-linear-overlays", type=Path)
    parser.add_argument("--upstream-overlays", type=Path)
    parser.add_argument("--w2-overlays", type=Path, default=DEFAULT_W2_OVERLAYS)
    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_CANDIDATE_POOL)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--template-model", type=Path, default=DEFAULT_TEMPLATE_MODEL)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.jobs < 1 or args.maximum_attempts < 1:
        raise ValueError("job and retry counts must be positive")
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("batched pure-K2 materialization requires CUDA")
    combined = args.all_linear_overlays is not None
    if combined == (args.upstream_overlays is not None):
        raise ValueError(
            "specify either --all-linear-overlays or --upstream-overlays"
        )
    overlay_roots = (
        (args.all_linear_overlays,)
        if combined
        else (args.upstream_overlays, args.w2_overlays)
    )
    assert all(root is not None for root in overlay_roots)
    resolved_overlay_roots = tuple(root for root in overlay_roots if root is not None)
    for path in (
        *resolved_overlay_roots,
        args.candidate_pool,
        args.profile,
        args.template_model,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    for layer in C.MOE_LAYERS:
        for root in resolved_overlay_roots:
            _overlay_from_completion(root, layer)
    if args.dest.exists() and not args.resume:
        raise FileExistsError("destination exists; pass --resume for the same build")

    overlay_kinds = tuple(
        str(_read_json(root / "build-config.json").get("kind"))
        for root in resolved_overlay_roots
    )
    if combined:
        if overlay_kinds != (
            "qsrt_uniform_k2_direct_viterbi_all_linear_overlays",
        ):
            raise ValueError("combined overlay does not encode all three matrices")
        materialization_kind = (
            "qsrt_uniform_k2_direct_viterbi_all_linear_materialization"
        )
        overlay_provenance = {
            "matrices": ["w1", "w3", "w2"],
            "combined": str(resolved_overlay_roots[0].resolve()),
        }
    else:
        if overlay_kinds == (
            "qsrt_uniform_k2_direct_viterbi_upstream_overlays",
            "qsrt_uniform_k2_direct_viterbi_w2_overlays",
        ):
            materialization_kind = (
                "qsrt_uniform_k2_direct_viterbi_all_linear_materialization"
            )
        elif overlay_kinds == (
            "qsrt_uniform_k2_direct_viterbi_upstream_overlays",
            "qsrt_uniform_k2_identity_input_final_logit_fisher_w2_overlays",
        ):
            materialization_kind = (
                "qsrt_uniform_k2_direct_upstream_identity_input_fisher_w2_materialization"
            )
        elif overlay_kinds == (
            "qsrt_uniform_k2_final_logit_fisher_upstream_overlays",
            "qsrt_uniform_k2_final_logit_fisher_w2_overlays",
        ):
            materialization_kind = (
                "qsrt_uniform_k2_final_logit_fisher_all_linear_materialization"
            )
        else:
            raise ValueError(
                "unsupported upstream/W2 overlay objective combination: "
                f"{overlay_kinds}"
            )
        overlay_provenance = {
            "matrices": ["w1", "w3", "w2"],
            "upstream": str(resolved_overlay_roots[0].resolve()),
            "downstream": str(resolved_overlay_roots[1].resolve()),
        }

    config = {
        "kind": materialization_kind,
        "schema_version": 1,
        "payload_overlays": overlay_provenance,
        "candidate_pool": str(args.candidate_pool.resolve()),
        "profile": str(args.profile.resolve()),
        "template_model": str(args.template_model.resolve()),
        "model_output": str(args.model_output.resolve()),
    }
    args.dest.mkdir(parents=True, exist_ok=True)
    for name in ("atoms", "receipts", "work", "logs"):
        (args.dest / name).mkdir(exist_ok=True)
    config_path = args.dest / "build-config.json"
    if config_path.exists():
        if _read_json(config_path) != config:
            raise ValueError("destination belongs to a different materialization")
    else:
        _atomic_json(config_path, config)

    completed = {layer for layer in C.MOE_LAYERS if _layer_complete(args.dest, layer)}
    pending: queue.Queue[tuple[int, int]] = queue.Queue()
    for layer in C.MOE_LAYERS:
        if layer not in completed:
            pending.put((layer, 1))
    failures: dict[int, str] = {}
    running: dict[int, int] = {}
    lock = threading.RLock()
    run_id = uuid.uuid4().hex[:12]
    started = time.monotonic()

    def write_status() -> None:
        with lock:
            _atomic_json(
                args.dest / "build-status.json",
                {
                    "complete": len(completed) == C.NUM_MOE_LAYERS and not failures,
                    "completed_layers": sorted(completed),
                    "failed_layers": failures,
                    "running": {str(worker): layer for worker, layer in running.items()},
                    "remaining_layers": pending.qsize(),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )

    def lane(worker: int) -> None:
        while True:
            try:
                layer, attempt = pending.get_nowait()
            except queue.Empty:
                return
            with lock:
                running[worker] = layer
            write_status()
            attempt_root = (
                args.dest
                / "work"
                / f"layer-{layer:03d}"
                / f"attempt-{attempt}-{run_id}"
            )
            attempt_root.mkdir(parents=True, exist_ok=False)
            atom = attempt_root / f"qsrt-layer-{layer:05d}.safetensors"
            receipt = attempt_root / "receipt.json"
            log = args.dest / "logs" / (
                f"layer-{layer:03d}-attempt-{attempt}-{run_id}.log"
            )
            command = [
                sys.executable,
                str(Path(__file__).with_name("materialize_qsrt_w2_overlay_layer.py")),
                "--candidate-pool",
                str(args.candidate_pool),
                "--profile",
                str(args.profile),
                "--layer",
                str(layer),
            ]
            for root in resolved_overlay_roots:
                command.extend(
                    ["--payload-overlay", str(_overlay_from_completion(root, layer))]
                )
            command.extend(
                [
                    "--replace-matrices",
                    "w1,w3,w2",
                    "--work-root",
                    str(attempt_root / "candidate-work"),
                    "--atom-output",
                    str(atom),
                    "--receipt",
                    str(receipt),
                    "--batch-size",
                    "64",
                    "--batched-pure-k2",
                    "--materialize-device",
                    f"cuda:{worker % device_count}",
                    "--skip-atom-sync",
                    "--skip-content-verification",
                ]
            )
            layer_started = time.monotonic()
            with log.open("w") as stream:
                environment = os.environ.copy()
                environment["OMP_NUM_THREADS"] = "1"
                environment["MKL_NUM_THREADS"] = "1"
                process = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=environment,
                )
            failure = None
            if process.returncode:
                failure = f"materializer exited with status {process.returncode}"
            elif not atom.is_file() or atom.stat().st_size != 7_415_300_096:
                failure = "materializer did not produce a complete atom layer"
            elif not receipt.is_file():
                failure = "materializer did not produce a receipt"
            if failure is None:
                final_atom = args.dest / "atoms" / atom.name
                final_receipt = args.dest / "receipts" / f"layer-{layer:03d}.json"
                os.replace(atom, final_atom)
                document = _read_json(receipt)
                document["atom_layer"] = str(final_atom.resolve())
                _atomic_json(final_receipt, document)
                shutil.rmtree(attempt_root, ignore_errors=True)
                with lock:
                    completed.add(layer)
                print(
                    json.dumps(
                        {
                            "layer": layer,
                            "worker": worker,
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
            with lock:
                running.pop(worker, None)
            pending.task_done()
            write_status()

    write_status()
    threads = [
        threading.Thread(target=lane, args=(worker,))
        for worker in range(args.jobs)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    write_status()
    if failures or len(completed) != C.NUM_MOE_LAYERS:
        raise SystemExit(
            f"materialization incomplete: {len(completed)}/{C.NUM_MOE_LAYERS}; "
            f"failed={sorted(failures)}"
        )

    source_manifest = _read_json(args.profile / "qsrt-manifest.json")
    source_manifest["research_payload_overlays"] = overlay_provenance
    _atomic_json(args.dest / "atoms" / "qsrt-manifest.json", source_manifest)
    _atomic_json(
        args.dest / "atoms" / "qsrt-completion.json",
        _read_json(args.profile / "qsrt-completion.json"),
    )
    _publish_model(
        args.dest / "atoms",
        args.template_model,
        args.model_output,
        overlay_provenance=overlay_provenance,
    )
    print(
        json.dumps(
            {
                "complete": str(args.dest.resolve()),
                "model": str(args.model_output.resolve()),
                "layers": C.NUM_MOE_LAYERS,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

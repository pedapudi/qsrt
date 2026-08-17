#!/usr/bin/env python3
"""Materialize a sealed fixed-profile candidate pool as QSRT atoms-v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from qsrt import constants as C
from qsrt.exl3_reference import CODEBOOK_SQG_XOR_CHEB_T12
from qsrt.pack.qsrt_atoms_v2 import (
    QSRTAtomsV2Reader,
    layer_filename,
    materialize_atoms_v2_layer,
)
from qsrt.pack.qsrt_pool import (
    CANDIDATE_POOL_COMPLETION_FILENAME,
    load_qsrt_candidate_pool,
)
from qsrt.qsrt import H308, K2, record_bits
from qsrt.qsrt_atoms_v2 import (
    COUPLED_H308_PROFILE,
    PROFILE,
    PURE_K2_PROFILE,
    QSRTAtomsV2Layout,
    SCHEMA,
)
from qsrt.qsrt_coupled_plan import CoupledRotationPlan


MANIFEST_FILENAME = "qsrt-manifest.json"
COMPLETION_FILENAME = "qsrt-completion.json"


def _atomic_json(path: Path, document: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(document, handle, indent=1, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(64 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_layers(value: str) -> tuple[int, ...]:
    if value == "all":
        return tuple(C.MOE_LAYERS)
    result = tuple(int(part) for part in value.split(",") if part)
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("layers must be unique or 'all'")
    if any(layer not in C.MOE_LAYERS for layer in result):
        raise argparse.ArgumentTypeError("layers must lie in 1..92")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--layers", type=_parse_layers, default=tuple(C.MOE_LAYERS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--discard-partials", action="store_true")
    parser.add_argument("--hash-layers", action="store_true")
    args = parser.parse_args()

    pool = load_qsrt_candidate_pool(args.candidate_pool, require_completion=True)
    if pool.mode_ids == (H308.mode_id,):
        mode = H308
        if pool.coupled_rotation_draws is None:
            profile = PROFILE
            rotation_plan = None
        else:
            profile = COUPLED_H308_PROFILE
            coupled_contract = pool.manifest.get("coupled_rotation")
            if not isinstance(coupled_contract, dict):
                raise ValueError(
                    "coupled H308 candidate pool has no rotation contract"
                )
            rotation_plan = CoupledRotationPlan(
                {
                    layer: tuple(
                        int(draw)
                        for draw in pool.coupled_rotation_draws[layer - 1]
                    )
                    for layer in C.MOE_LAYERS
                },
                str(
                    coupled_contract.get("selection")
                    or coupled_contract.get("source")
                ),
            )
    elif pool.mode_ids == (K2.mode_id,):
        mode = K2
        profile = PURE_K2_PROFILE
        if pool.coupled_rotation_draws is None:
            raise ValueError("pure K2 candidate pool has no coupled rotation draws")
        coupled_contract = pool.manifest.get("coupled_rotation")
        if not isinstance(coupled_contract, dict):
            raise ValueError("pure K2 candidate pool has no rotation contract")
        rotation_plan = CoupledRotationPlan(
            {
                layer: tuple(
                    int(draw)
                    for draw in pool.coupled_rotation_draws[layer - 1]
                )
                for layer in C.MOE_LAYERS
            },
            str(
                coupled_contract.get("selection")
                or coupled_contract.get("source")
            ),
        )
    else:
        raise ValueError("atoms-v2 materialization requires fixed H308 or K2")
    if pool.codebook != CODEBOOK_SQG_XOR_CHEB_T12:
        raise ValueError("atoms-v2 materialization requires sqg_xor_cheb_t12")
    if pool.completion is None or pool.content_sha256 is None:
        raise AssertionError("sealed candidate pool lost its completion identity")

    destination = args.dest.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / MANIFEST_FILENAME
    manifest = {
        "kind": "qsrt_kimi_k3_qsrt_artifact",
        "schema_version": 2,
        "codec": "QSRT",
        "complete": False,
        "storage_schema": SCHEMA,
        "storage_format": "qsrt_atoms_v2",
        "profile": profile,
        "record_bits": list(record_bits(mode)),
        "trellis_bits_per_weight": sum(record_bits(mode)) / len(record_bits(mode)),
        "tensor_parallel_independent": True,
        "all_experts_qsrt": True,
        "candidate_pool": str(pool.root),
        "candidate_pool_content_sha256": pool.content_sha256,
        "candidate_codebook": pool.codebook,
        "candidate_mode_ids": list(pool.mode_ids),
        "source_revision": pool.manifest.get("source_revision"),
        "coupled_rotation_plan": (
            None if rotation_plan is None else rotation_plan.to_json()
        ),
        "layer_layout": QSRTAtomsV2Layout(1, profile=profile).to_manifest(),
        "layers": {},
    }
    identity_fields = tuple(name for name in manifest if name not in {"complete", "layers"})
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        for field in identity_fields:
            if existing.get(field) != manifest[field]:
                raise ValueError(
                    f"atoms-v2 destination manifest {field} disagrees with this build"
                )
    else:
        _atomic_json(manifest_path, manifest)

    for index, layer in enumerate(args.layers, 1):
        path = destination / layer_filename(layer)
        if path.exists():
            if not args.resume:
                raise FileExistsError(path)
            with QSRTAtomsV2Reader(path) as reader:
                if reader.header.layer != layer:
                    raise ValueError(f"layer {layer} file has the wrong identity")
            print(f"layer {layer}: already complete ({index}/{len(args.layers)})", flush=True)
            continue
        result = materialize_atoms_v2_layer(
            pool.root,
            path,
            layer,
            batch_size=args.batch_size,
            discard_partial=args.discard_partials,
            profile=profile,
            rotation_draws=(
                None if rotation_plan is None else rotation_plan.for_layer(layer)
            ),
        )
        print(
            f"layer {layer}: {result['disk_bytes']} bytes ({index}/{len(args.layers)})",
            flush=True,
        )

    if set(args.layers) != set(C.MOE_LAYERS):
        return
    layers: dict[str, dict[str, object]] = {}
    completion_layers: dict[str, dict[str, object]] = {}
    total_bytes = 0
    for layer in C.MOE_LAYERS:
        path = destination / layer_filename(layer)
        with QSRTAtomsV2Reader(path) as reader:
            layout = reader.header.layout
        entry: dict[str, object] = {
            "qsrt_atoms": path.name,
            "atom_file": path.name,
            "atom_disk_bytes": path.stat().st_size,
            "compressed_experts": C.NUM_EXPERTS,
            "x4t_experts": 0,
            "payload_closure": "bit_exact",
            "layout": layout.to_manifest(),
        }
        completion_entry: dict[str, object] = {
            "file": path.name,
            "bytes": path.stat().st_size,
        }
        if args.hash_layers:
            digest = _sha256(path)
            entry["sha256"] = digest
            completion_entry["sha256"] = digest
        layers[str(layer)] = entry
        completion_layers[str(layer)] = completion_entry
        total_bytes += path.stat().st_size

    manifest.update(
        {
            "complete": True,
            "layer_count": len(C.MOE_LAYERS),
            "compressed_experts": len(C.MOE_LAYERS) * C.NUM_EXPERTS,
            "x4t_experts": 0,
            "container_bytes": total_bytes,
            "layers": layers,
        }
    )
    _atomic_json(manifest_path, manifest)
    completion = {
        "kind": "qsrt_kimi_k3_qsrt_completion",
        "schema_version": 2,
        "complete": True,
        "manifest": MANIFEST_FILENAME,
        "candidate_completion": CANDIDATE_POOL_COMPLETION_FILENAME,
        "candidate_pool_content_sha256": pool.content_sha256,
        "storage_schema": SCHEMA,
        "profile": profile,
        "layer_count": len(C.MOE_LAYERS),
        "layer_bytes": total_bytes,
        "layers": completion_layers,
    }
    completion_path = destination / COMPLETION_FILENAME
    if completion_path.exists() and json.loads(completion_path.read_text()) != completion:
        raise ValueError("existing atoms-v2 completion index drifted")
    if not completion_path.exists():
        _atomic_json(completion_path, completion)
    print(f"sealed {destination}: {total_bytes} layer bytes", flush=True)


if __name__ == "__main__":
    main()

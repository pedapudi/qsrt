#!/usr/bin/env python3
"""Materialize one uniform-K2 atom layer with replacement matrix payloads.

The replacement file set must contain exactly one packed trellis and its two
scale vectors for every expert and selected matrix. The script clones the
sealed candidate layer with a filesystem reflink, replaces only those tensor
byte ranges, verifies the replacement bytes, and invokes the ordinary atoms-v2
materializer. The sealed candidate pool is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from qsrt import constants as C
from qsrt.exl3_reference import CODEBOOK_SQG_XOR_CHEB_T12
from qsrt.io.hf_cache import read_safetensors_header
from qsrt.pack.qsrt_atoms import candidate_layer_path
from qsrt.pack.qsrt_atoms_v2 import materialize_atoms_v2_layer
from qsrt.pack.qsrt_atoms_v2_fast import materialize_pure_k2_atoms_v2_layer
from qsrt.pack.qsrt_candidates import candidate_tensor_name
from qsrt.qsrt import K2
from qsrt.qsrt_atoms_v2 import (
    PURE_K2_PROFILE,
    unpack_atoms_v2_format_section,
)


_HEADER_LENGTH = struct.Struct("<Q")
_COPY_BLOCK_BYTES = 64 << 20


@dataclass(frozen=True)
class TensorSlice:
    path: Path
    offset: int
    length: int


@dataclass(frozen=True)
class UniformK2Source:
    root: Path
    manifest: dict
    completion: dict
    content_sha256: str
    source: Path
    source_entry: dict
    rotation_draws: tuple[int, ...]


def _descriptor_for(descriptors: dict[Path, int], path: Path) -> int:
    descriptor = descriptors.get(path)
    if descriptor is None:
        descriptor = os.open(path, os.O_RDONLY)
        descriptors[path] = descriptor
    return descriptor


def _data_start(path: Path) -> int:
    with path.open("rb") as handle:
        raw = handle.read(_HEADER_LENGTH.size)
    if len(raw) != _HEADER_LENGTH.size:
        raise ValueError(f"short safetensors header length in {path}")
    return _HEADER_LENGTH.size + _HEADER_LENGTH.unpack(raw)[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_COPY_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _profile_rotation_draws(profile: Path, layer: int) -> tuple[int, ...]:
    path = profile / f"qsrt-layer-{layer:05d}.safetensors"
    with safe_open(path, framework="pt", device="cpu") as reader:
        metadata = reader.metadata()
        if metadata is None or "profile" not in metadata:
            raise ValueError("served profile layer lacks its atoms-v2 identity")
        formats, draws = unpack_atoms_v2_format_section(
            str(metadata["profile"]), reader.get_tensor("_qsrt_format_section")
        )
    if draws is None or len(draws) != C.NUM_EXPERTS:
        raise ValueError("served profile layer lacks complete rotation draws")
    if any(value != "K2" for value in formats):
        raise ValueError("served profile layer is not uniformly K2")
    return tuple(int(value) for value in draws)


def _load_uniform_k2_source(
    root: Path,
    *,
    profile: Path,
    layer: int,
    verify_source_hash: bool = True,
) -> UniformK2Source:
    root = root.resolve()
    manifest_path = root / "qsrt-candidate-manifest.json"
    completion_path = root / "qsrt-candidate-completion.json"
    manifest = _read_json(manifest_path)
    completion = _read_json(completion_path)
    accepted_manifest_kinds = {
        "kquant_kimi_k3_qsrt_candidate_pool",
        "qsrt_kimi_k3_qsrt_candidate_pool",
    }
    accepted_completion_kinds = {
        "kquant_kimi_k3_qsrt_candidate_completion",
        "qsrt_kimi_k3_qsrt_candidate_completion",
    }
    if manifest.get("kind") not in accepted_manifest_kinds:
        raise ValueError("candidate pool has an unsupported manifest identity")
    if completion.get("kind") not in accepted_completion_kinds:
        raise ValueError("candidate pool has an unsupported completion identity")
    if manifest.get("source_model") != C.MODEL_ID:
        raise ValueError("candidate pool source model differs from Kimi-K3")
    if manifest.get("source_revision") != C.REVISION:
        raise ValueError(
            "candidate pool source revision differs from the frozen source"
        )
    if tuple(int(value) for value in manifest.get("mode_ids", ())) != (
        K2.mode_id,
    ):
        raise ValueError("replacement materialization requires a uniform-K2 pool")
    if manifest.get("codebook") != CODEBOOK_SQG_XOR_CHEB_T12:
        raise ValueError("candidate pool does not use the qualified SQG codebook")
    if completion.get("manifest") != manifest_path.name:
        raise ValueError("candidate completion references a different manifest")
    manifest_sha256 = _sha256(manifest_path)
    if completion.get("manifest_sha256") != manifest_sha256:
        raise ValueError("candidate completion manifest hash does not close")
    content_sha256 = completion.get("content_sha256")
    if not isinstance(content_sha256, str) or len(content_sha256) != 64:
        raise ValueError("candidate completion lacks its content identity")
    layers = completion.get("layers")
    if not isinstance(layers, dict) or not isinstance(layers.get(str(layer)), dict):
        raise ValueError(f"candidate completion lacks layer {layer}")
    source_entry = layers[str(layer)].get("payload")
    if not isinstance(source_entry, dict) or not isinstance(
        source_entry.get("sha256"), str
    ):
        raise ValueError("candidate completion lacks its layer payload identity")
    source = candidate_layer_path(root, layer)
    if verify_source_hash and _sha256(source) != source_entry["sha256"]:
        raise ValueError("sealed candidate layer hash does not close")

    metrics_path = source.with_name(f"qsrt-layer-{layer:05d}.metrics.safetensors")
    with safe_open(metrics_path, framework="pt", device="cpu") as reader:
        required = {"selected_r13", "selected_r2", "coupled_draw_selected"}
        if not required.issubset(reader.keys()):
            raise ValueError("candidate metrics lack the uniform-K2 selections")
        selected_r13 = reader.get_tensor("selected_r13")
        selected_r2 = reader.get_tensor("selected_r2")
        metric_draws = reader.get_tensor("coupled_draw_selected")
    expected_shape = (C.NUM_EXPERTS,)
    if (
        tuple(selected_r13.shape) != expected_shape
        or tuple(selected_r2.shape) != expected_shape
    ):
        raise ValueError("candidate rate selections have the wrong expert extent")
    if not bool(torch.all(selected_r13 == K2.mode_id)) or not bool(
        torch.all(selected_r2 == K2.mode_id)
    ):
        raise ValueError("candidate layer contains a non-K2 expert")
    draws = _profile_rotation_draws(profile.resolve(), layer)
    if tuple(int(value) for value in metric_draws.tolist()) != draws:
        raise ValueError("served profile and candidate pool rotation draws differ")
    return UniformK2Source(
        root=root,
        manifest=manifest,
        completion=completion,
        content_sha256=content_sha256,
        source=source,
        source_entry=source_entry,
        rotation_draws=draws,
    )


def _hash_slices(slices: dict[str, TensorSlice]) -> str:
    digest = hashlib.sha256()
    descriptors: dict[Path, int] = {}
    try:
        for name in sorted(slices):
            item = slices[name]
            descriptor = _descriptor_for(descriptors, item.path)
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            cursor = 0
            while cursor < item.length:
                size = min(_COPY_BLOCK_BYTES, item.length - cursor)
                block = os.pread(descriptor, size, item.offset + cursor)
                if len(block) != size:
                    raise ValueError(
                        f"short tensor payload for {name} in {item.path}"
                    )
                digest.update(block)
                cursor += size
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)
    return digest.hexdigest()


def _overlay_slices(
    overlays: tuple[Path, ...],
    *,
    target_header: dict,
    layer: int,
    matrices: tuple[str, ...],
) -> dict[str, TensorSlice]:
    expected = {
        candidate_tensor_name(layer, expert, matrix, part)
        for expert in range(C.NUM_EXPERTS)
        for matrix in matrices
        for part in ("trellis", "suh", "svh")
    }
    slices: dict[str, TensorSlice] = {}
    for path in overlays:
        if not path.is_file():
            raise FileNotFoundError(path)
        header = read_safetensors_header(path)
        start = _data_start(path)
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if name not in expected:
                raise ValueError(
                    f"replacement tensor is outside layer {layer} "
                    f"matrices {matrices}: {name}"
                )
            if name in slices:
                raise ValueError(f"replacement tensor appears more than once: {name}")
            target_spec = target_header.get(name)
            if not isinstance(target_spec, dict):
                raise ValueError(f"sealed candidate layer lacks {name}")
            if spec.get("dtype") != target_spec.get("dtype") or spec.get(
                "shape"
            ) != target_spec.get("shape"):
                raise ValueError(f"replacement tensor metadata differs for {name}")
            begin, end = (int(value) for value in spec["data_offsets"])
            target_begin, target_end = (
                int(value) for value in target_spec["data_offsets"]
            )
            if begin < 0 or end < begin or end - begin != target_end - target_begin:
                raise ValueError(f"replacement tensor byte count differs for {name}")
            slices[name] = TensorSlice(path, start + begin, end - begin)
    missing = expected - slices.keys()
    if missing:
        raise ValueError(
            f"replacement set is missing {len(missing)} tensors; "
            f"first={sorted(missing)[0]}"
        )
    return slices


def _target_slices(
    target: Path,
    header: dict,
    names: set[str],
) -> dict[str, TensorSlice]:
    start = _data_start(target)
    result: dict[str, TensorSlice] = {}
    for name in names:
        begin, end = (int(value) for value in header[name]["data_offsets"])
        result[name] = TensorSlice(target, start + begin, end - begin)
    return result


def _write_slices(
    target: Path,
    sources: dict[str, TensorSlice],
    destinations: dict[str, TensorSlice],
    *,
    sync: bool = True,
) -> None:
    source_descriptors: dict[Path, int] = {}
    target_descriptor = os.open(target, os.O_RDWR)
    try:
        for name in sorted(sources):
            source = sources[name]
            destination = destinations[name]
            source_descriptor = _descriptor_for(source_descriptors, source.path)
            cursor = 0
            while cursor < source.length:
                size = min(_COPY_BLOCK_BYTES, source.length - cursor)
                block = os.pread(source_descriptor, size, source.offset + cursor)
                if len(block) != size:
                    raise ValueError(f"short replacement tensor payload for {name}")
                written = 0
                while written < size:
                    count = os.pwrite(
                        target_descriptor,
                        block[written:],
                        destination.offset + cursor + written,
                    )
                    if count <= 0:
                        raise OSError(f"short candidate-layer write for {name}")
                    written += count
                cursor += size
        if sync:
            os.fsync(target_descriptor)
    finally:
        os.close(target_descriptor)
        for descriptor in source_descriptors.values():
            os.close(descriptor)


def _atomic_json(path: Path, document: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--payload-overlay", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--replace-matrices",
        default="w2",
        help="comma-separated complete matrix set drawn from w1,w3,w2",
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--atom-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batched-pure-k2", action="store_true")
    parser.add_argument("--materialize-device", default="cpu")
    parser.add_argument("--skip-atom-sync", action="store_true")
    parser.add_argument(
        "--skip-content-verification",
        action="store_true",
        help=(
            "skip full-file and replacement-payload hashes for transient "
            "research materializations"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrices = tuple(value for value in args.replace_matrices.split(",") if value)
    if not matrices or len(matrices) != len(set(matrices)):
        raise ValueError("replacement matrices must be unique and nonempty")
    if any(matrix not in {"w1", "w3", "w2"} for matrix in matrices):
        raise ValueError("replacement matrices must be drawn from w1,w3,w2")
    if args.layer not in C.MOE_LAYERS:
        raise ValueError("layer must lie in 1..92")
    if args.work_root.exists():
        raise FileExistsError(args.work_root)
    if args.atom_output.exists():
        raise FileExistsError(args.atom_output)
    if args.receipt.exists():
        raise FileExistsError(args.receipt)

    pool = _load_uniform_k2_source(
        args.candidate_pool,
        profile=args.profile,
        layer=args.layer,
        verify_source_hash=not args.skip_content_verification,
    )
    source = pool.source
    source_entry = pool.source_entry

    target_root = args.work_root / "candidate"
    target = candidate_layer_path(target_root, args.layer)
    target.parent.mkdir(parents=True)
    subprocess.run(
        ["cp", "--reflink=always", "--", str(source), str(target)],
        check=True,
    )
    target_header = read_safetensors_header(target)
    source_slices = _overlay_slices(
        tuple(path.resolve() for path in args.payload_overlay),
        target_header=target_header,
        layer=args.layer,
        matrices=matrices,
    )
    target_slices = _target_slices(target, target_header, set(source_slices))
    source_payload_sha256 = (
        None if args.skip_content_verification else _hash_slices(source_slices)
    )
    _write_slices(
        target,
        source_slices,
        target_slices,
        sync=not args.skip_content_verification,
    )
    if not args.skip_content_verification:
        target_payload_sha256 = _hash_slices(target_slices)
        if target_payload_sha256 != source_payload_sha256:
            raise ValueError("candidate-layer replacement bytes failed exact closure")

    rotation_draws = tuple(int(value) for value in pool.rotation_draws)
    if args.batched_pure_k2:
        atom_result = materialize_pure_k2_atoms_v2_layer(
            target_root,
            args.atom_output,
            args.layer,
            batch_size=args.batch_size,
            rotation_draws=rotation_draws,
            device=args.materialize_device,
            sync=not args.skip_atom_sync,
        )
    else:
        atom_result = materialize_atoms_v2_layer(
            target_root,
            args.atom_output,
            args.layer,
            batch_size=args.batch_size,
            profile=PURE_K2_PROFILE,
            rotation_draws=rotation_draws,
        )
    receipt = {
        "kind": "qsrt_uniform_k2_payload_overlay_layer",
        "schema_version": 1,
        "layer": args.layer,
        "candidate_pool": str(pool.root),
        "candidate_pool_content_sha256": pool.content_sha256,
        "served_profile": str(args.profile.resolve()),
        "source_candidate_layer": str(source),
        "source_candidate_layer_sha256": source_entry["sha256"],
        "replacement_files": [
            str(path.resolve()) for path in args.payload_overlay
        ],
        "replacement_matrices": list(matrices),
        "replacement_tensors": len(source_slices),
        "replacement_payload_sha256": source_payload_sha256,
        "patched_candidate_layer": str(target),
        "patched_candidate_layer_sha256": (
            None if args.skip_content_verification else _sha256(target)
        ),
        "atom_layer": str(args.atom_output.resolve()),
        "atom_layer_sha256": (
            None if args.skip_content_verification else _sha256(args.atom_output)
        ),
        "atom_materialization": atom_result,
        "atom_materializer": (
            "batched_pure_k2" if args.batched_pure_k2 else "generic"
        ),
        "content_verification": (
            "skipped" if args.skip_content_verification else "complete"
        ),
        "payload_closure": (
            "not_checked" if args.skip_content_verification else "bit_exact"
        ),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

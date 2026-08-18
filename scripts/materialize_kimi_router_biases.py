#!/usr/bin/env python3
"""Publish a Kimi checkpoint view with replacement router-selection biases."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file


DEFAULT_ANCHOR = Path(
    "/data/models/Kimi-K3-QSRT-K2-DIRECT-VITERBI-ALL-LINEARS-v1-model"
)
BIAS_SUFFIX = ".block_sparse_moe.gate.e_score_correction_bias"
EXPECTED_LAYERS = 93
EXPECTED_EXPERTS = 896


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _layer(name: str) -> int:
    prefix = "language_model.model.layers."
    if not name.startswith(prefix) or not name.endswith(BIAS_SUFFIX):
        raise ValueError(f"not a Kimi router-bias tensor: {name}")
    value = name[len(prefix) : -len(BIAS_SUFFIX)]
    if not value.isdigit():
        raise ValueError(f"cannot parse layer from {name}")
    return int(value)


def _header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"{path}: truncated safetensors prefix")
        header_bytes = struct.unpack("<Q", size_bytes)[0]
        document = json.loads(stream.read(header_bytes))
    if not isinstance(document, dict):
        raise TypeError(f"{path}: safetensors header must be a JSON object")
    return 8 + header_bytes, document


def _reflink_and_patch(
    *,
    source: Path,
    target: Path,
    replacements: dict[str, torch.Tensor],
) -> None:
    temporary = target.with_name(f".{target.name}.router-bias.partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    subprocess.run(
        [
            "cp",
            "--reflink=always",
            "--preserve=mode,timestamps",
            "--",
            str(source),
            str(temporary),
        ],
        check=True,
    )
    try:
        data_offset, header = _header(temporary)
        descriptor = os.open(temporary, os.O_WRONLY)
        try:
            for name, tensor in replacements.items():
                entry = header.get(name)
                if not isinstance(entry, dict):
                    raise KeyError(f"{temporary}: missing tensor {name}")
                if entry.get("dtype") != "F32" or entry.get("shape") != [896]:
                    raise ValueError(
                        f"{temporary}: {name} is not the expected F32[896] tensor"
                    )
                offsets = entry.get("data_offsets")
                if (
                    not isinstance(offsets, list)
                    or len(offsets) != 2
                    or int(offsets[1]) - int(offsets[0]) != EXPECTED_EXPERTS * 4
                ):
                    raise ValueError(f"{temporary}: invalid byte range for {name}")
                payload = tensor.to(dtype=torch.float32, device="cpu").contiguous()
                view = memoryview(payload.numpy()).cast("B")
                cursor = 0
                absolute = data_offset + int(offsets[0])
                while cursor < len(view):
                    written = os.pwrite(descriptor, view[cursor:], absolute + cursor)
                    if written <= 0:
                        raise OSError("router-bias pwrite made no forward progress")
                    cursor += written
            os.fdatasync(descriptor)
        finally:
            os.close(descriptor)
        target.unlink(missing_ok=True)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _verify(destination: Path, weight_map: dict[str, str], biases: torch.Tensor) -> None:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, filename in weight_map.items():
        if name.endswith(BIAS_SUFFIX):
            by_shard[filename].append(name)
    observed: set[int] = set()
    for filename, names in by_shard.items():
        with safe_open(destination / filename, framework="pt", device="cpu") as reader:
            for name in names:
                layer = _layer(name)
                actual = reader.get_tensor(name).to(torch.float32)
                if not torch.equal(actual, biases[layer]):
                    raise ValueError(f"materialized router bias differs at layer {layer}")
                observed.add(layer)
    expected = set(range(1, EXPECTED_LAYERS))
    if observed != expected:
        raise ValueError(
            f"router-bias inventory mismatch; missing={sorted(expected - observed)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biases", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    args = parser.parse_args()

    anchor = args.anchor.expanduser().resolve()
    bias_path = args.biases.expanduser().resolve()
    destination = args.dest.expanduser().resolve()
    if not anchor.is_dir():
        raise FileNotFoundError(anchor)
    if not bias_path.is_file():
        raise FileNotFoundError(bias_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    tensors = load_file(bias_path, device="cpu")
    biases = tensors["biases"].to(torch.float32).contiguous()
    active = tensors["active_layers"].to(torch.bool)
    if biases.shape != (EXPECTED_LAYERS, EXPECTED_EXPERTS):
        raise ValueError(f"biases have shape {tuple(biases.shape)}")
    expected_active = torch.zeros(EXPECTED_LAYERS, dtype=torch.bool)
    expected_active[1:] = True
    if not torch.equal(active, expected_active):
        raise ValueError("bias file does not cover exactly the 92 routed layers")

    index_path = anchor / "model.safetensors.index.json"
    index = _read_json(index_path)
    raw_weight_map = index.get("weight_map")
    if not isinstance(raw_weight_map, dict):
        raise ValueError("anchor model index has no weight_map")
    weight_map = {str(name): str(filename) for name, filename in raw_weight_map.items()}
    replacements: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    layers: set[int] = set()
    for name, filename in weight_map.items():
        if not name.endswith(BIAS_SUFFIX):
            continue
        layer = _layer(name)
        replacements[filename][name] = biases[layer]
        layers.add(layer)
    if layers != set(range(1, EXPECTED_LAYERS)):
        raise ValueError("anchor model does not contain all routed-layer biases")

    staging = destination.with_name(f".{destination.name}.partial")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-al", "--", str(anchor), str(staging)], check=True)
    try:
        for filename, shard_replacements in sorted(replacements.items()):
            _reflink_and_patch(
                source=anchor / filename,
                target=staging / filename,
                replacements=shard_replacements,
            )
        _atomic_json(
            staging / "qsrt-router-bias-overlay.json",
            {
                "kind": "Kimi-K3 router-selection bias overlay",
                "schema_version": 1,
                "anchor": str(anchor),
                "biases": str(bias_path),
                "replaced_layers": list(range(1, EXPECTED_LAYERS)),
                "replaced_shards": sorted(replacements),
            },
        )
        _verify(staging, weight_map, biases)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "destination": str(destination),
                "anchor": str(anchor),
                "biases": str(bias_path),
                "replaced_shards": len(replacements),
                "replaced_layers": len(layers),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

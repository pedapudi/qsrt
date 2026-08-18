#!/usr/bin/env python3
"""Publish a Kimi checkpoint view with a BF16 suffix-training overlay."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from qsrt.suffix_recovery_training import is_shared_expert_or_norm_tensor


FIRST_LAYER = 84
END_LAYER = 93
INDEX_FILENAME = "model.safetensors.index.json"
CONFIG_FILENAME = "config.json"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _layer(name: str) -> int | None:
    prefix = "language_model.model.layers."
    if not name.startswith(prefix):
        return None
    value = name[len(prefix) :].split(".", 1)[0]
    return int(value) if value.isdigit() else None


def _allowed(name: str) -> bool:
    layer = _layer(name)
    if layer is not None:
        return FIRST_LAYER <= layer < END_LAYER and is_shared_expert_or_norm_tensor(name)
    return name in {
        "language_model.model.norm.weight",
        "language_model.model.output_attn_res_norm.weight",
    }


def _tensor_bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def _bf16_dense_modules(replacements: Mapping[str, torch.Tensor]) -> tuple[str, ...]:
    marker = ".block_sparse_moe.shared_experts."
    return tuple(
        sorted(
            name.removesuffix(".weight")
            for name in replacements
            if marker in name and name.endswith(".weight")
        )
    )


def _configure_bf16_dense_modules(
    config: dict[str, Any],
    modules: tuple[str, ...],
) -> None:
    if not modules:
        return
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("config.json has no text_config object")
    quantization = text_config.get("quantization_config")
    if not isinstance(quantization, dict) or quantization.get("dense_format") != "mxfp8":
        raise ValueError(
            "BF16 shared-expert replacements require the serialized MXFP8 "
            "dense-format configuration"
        )
    ignored = quantization.get("ignored_layers")
    if not isinstance(ignored, list) or any(not isinstance(value, str) for value in ignored):
        raise ValueError("quantization_config.ignored_layers must be a string list")
    quantization["ignored_layers"] = sorted(set(ignored).union(modules))


def _rewrite_shard(
    *,
    source: Path,
    target: Path,
    replacements: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    with safe_open(str(source), framework="pt", device="cpu") as reader:
        metadata = reader.metadata()
        tensors = {name: reader.get_tensor(name) for name in reader.keys()}
    removed_scales = []
    for name, replacement in replacements.items():
        if name not in tensors:
            raise KeyError(f"{source}: missing overlay tensor {name}")
        if tuple(tensors[name].shape) != tuple(replacement.shape):
            raise ValueError(f"{name}: overlay shape does not match the checkpoint")
        if replacement.dtype != torch.bfloat16:
            raise TypeError(f"{name}: suffix overlay must contain BF16 tensors")
        tensors[name] = replacement.contiguous()
        scale_name = f"{name}_scale"
        if scale_name in tensors:
            del tensors[scale_name]
            removed_scales.append(scale_name)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    save_file(tensors, temporary, metadata=metadata)
    target.unlink()
    os.replace(temporary, target)
    return tuple(removed_scales)


def _verify(
    destination: Path,
    *,
    weight_map: Mapping[str, str],
    replacements: Mapping[str, torch.Tensor],
    removed_scales: set[str],
) -> None:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name in replacements:
        by_shard[weight_map[name]].append(name)
    for filename, names in by_shard.items():
        with safe_open(
            str(destination / filename),
            framework="pt",
            device="cpu",
        ) as reader:
            present = set(reader.keys())
            for name in names:
                actual = reader.get_tensor(name)
                if actual.dtype != torch.bfloat16 or not torch.equal(
                    actual,
                    replacements[name],
                ):
                    raise ValueError(f"materialized overlay differs for {name}")
            unexpected = present & removed_scales
            if unexpected:
                raise ValueError(
                    f"materialized shard retained scale tensors {sorted(unexpected)}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()

    anchor = args.anchor.expanduser().resolve()
    overlay_path = args.overlay.expanduser().resolve()
    destination = args.dest.expanduser().resolve()
    if not anchor.is_dir():
        raise FileNotFoundError(anchor)
    if not overlay_path.is_file():
        raise FileNotFoundError(overlay_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    index_path = anchor / INDEX_FILENAME
    index = _read_json(index_path)
    config_path = anchor / CONFIG_FILENAME
    config = _read_json(config_path)
    raw_weight_map = index.get("weight_map")
    if not isinstance(raw_weight_map, dict):
        raise ValueError("anchor model index has no weight map")
    weight_map = {str(name): str(filename) for name, filename in raw_weight_map.items()}
    replacements = load_file(overlay_path, device="cpu")
    if not replacements or any(not _allowed(name) for name in replacements):
        invalid = sorted(name for name in replacements if not _allowed(name))
        raise ValueError(f"overlay contains tensors outside the first arm: {invalid[:3]}")
    missing = set(replacements) - weight_map.keys()
    if missing:
        raise KeyError(f"anchor lacks overlay tensors {sorted(missing)[:3]}")
    bf16_dense_modules = _bf16_dense_modules(replacements)
    _configure_bf16_dense_modules(config, bf16_dense_modules)

    by_shard: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for name, value in replacements.items():
        by_shard[weight_map[name]][name] = value

    staging = destination.with_name(f".{destination.name}.partial")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-al", "--", str(anchor), str(staging)], check=True)
    removed_scales: set[str] = set()
    try:
        for filename, shard_replacements in sorted(by_shard.items()):
            removed_scales.update(
                _rewrite_shard(
                    source=anchor / filename,
                    target=staging / filename,
                    replacements=shard_replacements,
                )
            )
        for name in removed_scales:
            weight_map.pop(name, None)
        index["weight_map"] = weight_map
        metadata = index.get("metadata")
        if isinstance(metadata, dict) and "total_size" in metadata:
            total_size = 0
            for filename in sorted(set(weight_map.values())):
                with safe_open(
                    str(staging / filename),
                    framework="pt",
                    device="cpu",
                ) as reader:
                    total_size += sum(
                        _tensor_bytes(reader.get_tensor(name)) for name in reader.keys()
                    )
            metadata["total_size"] = total_size
        index_target = staging / INDEX_FILENAME
        index_target.unlink()
        _atomic_json(index_target, index)
        _atomic_json(staging / CONFIG_FILENAME, config)
        _atomic_json(
            staging / "qsrt-continuous-recovery-overlay.json",
            {
                "kind": "Kimi-K3 BF16 continuous-parameter recovery overlay",
                "schema_version": 1,
                "anchor": str(anchor),
                "overlay": str(overlay_path),
                "replaced_tensors": sorted(replacements),
                "removed_scale_tensors": sorted(removed_scales),
                "bf16_dense_modules": list(bf16_dense_modules),
                "rewritten_shards": sorted(by_shard),
            },
        )
        _verify(
            staging,
            weight_map=weight_map,
            replacements=replacements,
            removed_scales=removed_scales,
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "destination": str(destination),
                "replaced_tensors": len(replacements),
                "removed_scale_tensors": len(removed_scales),
                "rewritten_shards": len(by_shard),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

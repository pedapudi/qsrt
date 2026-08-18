"""Differentiable Kimi-K3 replay from a materialized QSRT checkpoint.

Decoder layers use the served checkpoint's non-expert tensors and reconstruct
the routed experts from the sealed uniform-K2 candidate pool plus optional
payload overlays.  The resulting layer has the same grouped differentiable
expert dispatch used by the Fisher capture pipeline.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open

from qsrt import constants as C
from qsrt.instanttensor_kimi import (
    InstantTensorLoadConfig,
    KimiLayerLoadStats,
    OfficialKimiLayerShards,
    load_checkpoint_tensors_cuda,
)
from qsrt.kimi_official_forward import (
    OfficialKimiForwardAdapter,
    OfficialKimiRuntime,
    _allocate_grouped_expert_weights,
    _install_grouped_expert_dispatch,
    new_meta_decoder_layer,
)
from qsrt.kimi_stream import MODEL_TENSOR_PREFIX, assign_parameter, fit_checkpoint_parameter
from qsrt.pack.qsrt_atoms import candidate_layer_path
from qsrt.pack.qsrt_validation import decode_candidate_matrix
from qsrt.qsrt import K2
from qsrt.qsrt_atoms_v2 import unpack_atoms_v2_format_section
from qsrt.qsrt_coupled import (
    CoupledHadamardSpec,
    decode_coupled_weights,
)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _overlay_path(root: Path, layer: int) -> Path:
    completion = _read_json(
        root / "layers" / f"layer-{layer:03d}" / "completion.json"
    )
    value = completion.get("payload_overlay")
    if not isinstance(value, str):
        raise ValueError(f"{root}: layer {layer} has no payload overlay")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class CompositeCandidateTensorReader:
    """Resolve candidate tensors through ordered full-layer overlays."""

    def __init__(self, base: str | Path, overlays: Iterable[str | Path] = ()):
        self.base = Path(base).expanduser().resolve()
        self.overlays = tuple(Path(path).expanduser().resolve() for path in overlays)
        self._stack = contextlib.ExitStack()
        self._owners: dict[str, Any] = {}

    def __enter__(self) -> "CompositeCandidateTensorReader":
        paths = (self.base, *self.overlays)
        for index, path in enumerate(paths):
            if not path.is_file():
                raise FileNotFoundError(path)
            handle = self._stack.enter_context(
                safe_open(str(path), framework="pt", device="cpu")
            )
            names = tuple(handle.keys())
            if index == 0:
                self._owners.update((name, handle) for name in names)
                continue
            unknown = set(names) - self._owners.keys()
            if unknown:
                raise ValueError(
                    f"overlay {path} contains tensors absent from the sealed layer; "
                    f"first={sorted(unknown)[0]}"
                )
            self._owners.update((name, handle) for name in names)
        return self

    def __exit__(self, *args: object) -> None:
        self._owners.clear()
        self._stack.close()

    def get_tensor(self, name: str) -> torch.Tensor:
        try:
            owner = self._owners[name]
        except KeyError as error:
            raise KeyError(f"candidate layer has no tensor {name}") from error
        return owner.get_tensor(name)


class QSRTAnchorPayload:
    """Resolve exact per-layer expert payloads for one quantized anchor."""

    def __init__(
        self,
        candidate_pool: str | Path,
        *,
        overlay_roots: Iterable[str | Path] = (),
    ):
        self.candidate_pool = Path(candidate_pool).expanduser().resolve()
        self.overlay_roots = tuple(
            Path(root).expanduser().resolve() for root in overlay_roots
        )
        if not self.candidate_pool.is_dir():
            raise FileNotFoundError(self.candidate_pool)
        for root in self.overlay_roots:
            if not root.is_dir():
                raise FileNotFoundError(root)

    def layer_paths(self, layer: int) -> tuple[Path, tuple[Path, ...]]:
        if layer not in C.MOE_LAYERS:
            raise ValueError(f"layer {layer} is not a routed Kimi-K3 layer")
        base = candidate_layer_path(self.candidate_pool, layer)
        if not base.is_file():
            raise FileNotFoundError(base)
        return base, tuple(_overlay_path(root, layer) for root in self.overlay_roots)


def _tensor_bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def _dequantize_mxfp8_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        raise TypeError("MXFP8 weight must be a two-dimensional E4M3 tensor")
    if weight.shape[1] % 32:
        raise ValueError("MXFP8 input dimension must be divisible by 32")
    expected = (int(weight.shape[0]), int(weight.shape[1]) // 32)
    if tuple(scale.shape) != expected:
        raise ValueError(f"MXFP8 scale shape {tuple(scale.shape)} != {expected}")
    if scale.dtype == torch.uint8:
        scale = scale.view(torch.float8_e8m0fnu)
    elif scale.dtype != torch.float8_e8m0fnu:
        raise TypeError("MXFP8 scale must be serialized as U8 or E8M0")
    return (
        weight.float()
        .view(weight.shape[0], weight.shape[1] // 32, 32)
        .mul_(scale.float().unsqueeze(-1))
        .view(weight.shape)
        .to(dtype=output_dtype)
    )


def _assign_buffer(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    parts = name.split(".")
    owner: torch.nn.Module = module
    for part in parts[:-1]:
        owner = getattr(owner, part)
    leaf = parts[-1]
    if leaf not in owner._buffers:
        raise KeyError(f"{name} is not a registered buffer")
    owner._buffers[leaf] = value


def _load_nonexpert_layer(
    module: torch.nn.Module,
    *,
    layer: int,
    checkpoint: Path,
    index: OfficialKimiLayerShards,
    device: torch.device,
    config: InstantTensorLoadConfig,
) -> KimiLayerLoadStats:
    started = time.monotonic()
    prefix = f"{MODEL_TENSOR_PREFIX}.layers.{layer}."
    parameters = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if not name.startswith("block_sparse_moe.experts.")
    ]
    buffers = list(module.named_buffers())
    requested: list[str] = []
    for name, _parameter in parameters:
        checkpoint_name = prefix + name
        requested.append(checkpoint_name)
        scale_name = f"{checkpoint_name}_scale"
        if scale_name in index.weight_map:
            requested.append(scale_name)
    for name, _buffer in buffers:
        checkpoint_name = prefix + name
        if checkpoint_name in index.weight_map:
            requested.append(checkpoint_name)

    loaded = load_checkpoint_tensors_cuda(
        checkpoint,
        requested,
        device=device,
        config=config,
    )
    shards = sorted({str(index.tensor_shard(name)) for name in requested})
    stats = KimiLayerLoadStats(layer=layer, shard=",".join(shards))
    stats.serialized_bytes = sum(_tensor_bytes(value) for value in loaded.values())
    for name, parameter in parameters:
        checkpoint_name = prefix + name
        value, fix = fit_checkpoint_parameter(
            checkpoint_name,
            loaded[checkpoint_name],
            tuple(parameter.shape),
        )
        scale_name = f"{checkpoint_name}_scale"
        if value.dtype == torch.float8_e4m3fn:
            value = _dequantize_mxfp8_weight(
                value,
                loaded[scale_name],
                output_dtype=parameter.dtype,
            )
        else:
            value = value.to(device=device, dtype=parameter.dtype)
        owned = value.contiguous()
        assign_parameter(module, name, owned)
        stats.dense_bytes += _tensor_bytes(owned)
        stats.nonexpert_parameters += 1
        if fix is not None:
            stats.compatibility_fixes.append(f"{checkpoint_name}: {fix}")

    for name, buffer in buffers:
        checkpoint_name = prefix + name
        value = loaded.get(checkpoint_name)
        if value is not None:
            _assign_buffer(
                module,
                name,
                value.to(device=device, dtype=buffer.dtype).contiguous(),
            )

    meta_parameters = [name for name, value in module.named_parameters() if value.is_meta]
    meta_parameters = [
        name
        for name in meta_parameters
        if name.startswith("block_sparse_moe.experts.")
    ]
    unexpected_meta = [
        name
        for name, value in module.named_parameters()
        if value.is_meta and name not in meta_parameters
    ]
    meta_buffers = [name for name, value in module.named_buffers() if value.is_meta]
    if unexpected_meta or meta_buffers:
        raise RuntimeError(
            f"layer {layer} retains non-expert meta state; "
            f"parameters={unexpected_meta[:3]}, buffers={meta_buffers[:3]}"
        )
    stats.elapsed_seconds = time.monotonic() - started
    return stats


class QSRTKimiForwardAdapter(OfficialKimiForwardAdapter):
    """Replay an exact QSRT anchor through differentiable official layers."""

    def __init__(
        self,
        runtime: OfficialKimiRuntime,
        *,
        model_checkpoint: str | Path,
        expert_payload: QSRTAnchorPayload,
        load_config: InstantTensorLoadConfig | None = None,
        validate_outputs: bool = True,
    ):
        self.runtime = runtime
        self.model_checkpoint = Path(model_checkpoint).expanduser().resolve()
        self.expert_payload = expert_payload
        self.load_config = load_config or InstantTensorLoadConfig()
        self.validate_outputs = bool(validate_outputs)
        self.grouped_expert_dispatch = True
        self._index = OfficialKimiLayerShards(self.model_checkpoint)

    def _coupled_draws(self, layer: int) -> tuple[int, ...]:
        path = self.model_checkpoint / f"qsrt-layer-{layer:05d}.safetensors"
        if not path.is_file():
            raise FileNotFoundError(path)
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            metadata = reader.metadata()
            if metadata is None or "profile" not in metadata:
                raise ValueError(f"QSRT layer {layer} lacks its atoms profile")
            _formats, draws = unpack_atoms_v2_format_section(
                str(metadata["profile"]),
                reader.get_tensor("_qsrt_format_section"),
            )
        if draws is None or len(draws) != C.NUM_EXPERTS:
            raise ValueError(f"QSRT layer {layer} lacks coupled-Hadamard draws")
        return tuple(int(value) for value in draws)

    def load_layer(
        self,
        layer: int,
        device: torch.device,
    ) -> tuple[Any, KimiLayerLoadStats]:
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        module = new_meta_decoder_layer(
            self.runtime,
            layer,
            grouped_expert_dispatch=True,
        )
        stats = _load_nonexpert_layer(
            module,
            layer=layer,
            checkpoint=self.model_checkpoint,
            index=self._index,
            device=device,
            config=self.load_config,
        )
        block = getattr(module, "block_sparse_moe", None)
        if block is not None:
            decode_started = time.monotonic()
            banks = _allocate_grouped_expert_weights(module, device)
            base, overlays = self.expert_payload.layer_paths(layer)
            draws = self._coupled_draws(layer)
            with CompositeCandidateTensorReader(base, overlays) as reader:
                for expert in range(C.NUM_EXPERTS):
                    stored = tuple(
                        decode_candidate_matrix(
                            reader,
                            layer=layer,
                            expert=expert,
                            matrix=matrix,
                            mode_id=K2.mode_id,
                            device=device,
                        ).T.contiguous()
                        for matrix in C.EXPERT_MATRICES
                    )
                    source = decode_coupled_weights(
                        stored,
                        CoupledHadamardSpec(intermediate_draw=draws[expert]),
                    )
                    for matrix, decoded in zip(
                        C.EXPERT_MATRICES,
                        source,
                        strict=True,
                    ):
                        banks[matrix][expert].copy_(
                            decoded.to(dtype=torch.bfloat16)
                        )
                        stats.expert_matrices += 1
                    del stored, source
            _install_grouped_expert_dispatch(module, banks)
            stats.grouped_expert_pack_seconds = time.monotonic() - decode_started
            stats.dense_bytes += sum(_tensor_bytes(bank) for bank in banks.values())
        module.requires_grad_(False)
        stats.elapsed_seconds = time.monotonic() - started
        stats.peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
        return module, stats


__all__ = [
    "CompositeCandidateTensorReader",
    "QSRTAnchorPayload",
    "QSRTKimiForwardAdapter",
]

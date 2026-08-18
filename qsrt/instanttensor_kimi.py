"""Bounded GPU loading of one official Kimi-K3 decoder layer.

The official checkpoint stores each decoder layer in one safetensors shard.
Routed-expert matrices are MXFP4 code and scale pairs ordered consecutively in
that shard.  This module streams the shard directly to a CUDA device, retains
non-expert parameters, and expands one expert matrix at a time into its final
BF16 parameter.  Peak memory is therefore the materialized layer plus the
InstantTensor ring and one MXFP4 matrix under reconstruction.
"""

from __future__ import annotations

import ctypes
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import torch
import triton
import triton.language as tl
import instanttensor._C as _instanttensor_c
from instanttensor import Backend, safe_open
from instanttensor._impl import safetensors_to_torch_dtype

from qsrt import constants as C
from qsrt.kimi_stream import (
    MODEL_TENSOR_PREFIX,
    assign_parameter,
    fit_checkpoint_parameter,
)
from qsrt.io.hf_cache import resolve


_EXPERT_PART = re.compile(
    r"^block_sparse_moe\.experts\.(?P<expert>[0-9]+)\."
    r"(?P<matrix>w1|w2|w3)\.weight_(?P<part>packed|scale)$"
)


class _DLDevice(ctypes.Structure):
    _fields_ = (("device_type", ctypes.c_int), ("device_id", ctypes.c_int))


class _DLTensorPrefix(ctypes.Structure):
    _fields_ = (("data", ctypes.c_void_p), ("device", _DLDevice))


class _DLManagedTensorPrefix(ctypes.Structure):
    _fields_ = (("dl_tensor", _DLTensorPrefix),)


_PYCAPSULE_GET_POINTER = ctypes.pythonapi.PyCapsule_GetPointer
_PYCAPSULE_GET_POINTER.argtypes = (ctypes.py_object, ctypes.c_char_p)
_PYCAPSULE_GET_POINTER.restype = ctypes.c_void_p

_GET_DL_TENSOR_HAS_CONSUMER_STREAM = "consumer_stream" in (
    _instanttensor_c.get_dl_tensor.__doc__ or ""
)
_GET_DL_TENSOR_SPAN = getattr(_instanttensor_c, "get_dl_tensor_span", None)


def _set_dlpack_device(capsule: object, device_index: int) -> None:
    """Set the destination recorded in an InstantTensor DLPack capsule."""

    pointer = _PYCAPSULE_GET_POINTER(capsule, b"dltensor")
    if not pointer:
        raise RuntimeError("InstantTensor returned an invalid DLPack capsule")
    managed = ctypes.cast(pointer, ctypes.POINTER(_DLManagedTensorPrefix))
    device = managed.contents.dl_tensor.device
    if device.device_type != 2:
        raise RuntimeError(
            f"InstantTensor returned DLPack device type {device.device_type}, expected CUDA"
        )
    managed.contents.dl_tensor.device.device_id = int(device_index)


def _take_instanttensor_span(handle: Any) -> torch.Tensor:
    """Consume a whole-buffer handle and return its byte-addressable CUDA span."""

    if _GET_DL_TENSOR_SPAN is None:
        raise RuntimeError("InstantTensor does not provide whole-span loading")
    if handle._invalidated:
        raise RuntimeError("InstantTensor handle was already closed")
    if handle.iterated:
        raise RuntimeError("InstantTensor handle can only be iterated once")
    if handle.buffer_size < handle.total_tensor_size:
        raise RuntimeError("InstantTensor buffer does not cover the complete shard")
    handle.iterated = True
    stream = torch.cuda.current_stream(handle.device)
    capsule = _GET_DL_TENSOR_SPAN(
        handle.loader_handle,
        0,
        len(handle.tensor_sizes) - 1,
        handle.total_tensor_size,
        stream.cuda_stream,
    )
    _set_dlpack_device(capsule, handle.device_idx)
    return torch.from_dlpack(capsule)


def _span_tensor(span: torch.Tensor, metadata: Mapping[str, Any]) -> torch.Tensor:
    """Create one typed tensor view from safetensors-relative data offsets."""

    dtype_name = metadata["dtype"]
    dtype = safetensors_to_torch_dtype.get(dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported safetensors dtype: {dtype_name}")
    start, end = map(int, metadata["data_offsets"])
    value = span.narrow(0, start, end - start).view(dtype)
    return value.view(metadata["shape"])


def _iter_instanttensor_tensors(handle: Any) -> Iterator[tuple[str, torch.Tensor]]:
    """Iterate one handle using its own device identity under concurrent loads."""

    if handle._invalidated:
        raise RuntimeError("InstantTensor handle was already closed")
    if handle.iterated:
        raise RuntimeError("InstantTensor handle can only be iterated once")
    if (
        _GET_DL_TENSOR_SPAN is not None
        and handle.buffer_size >= handle.total_tensor_size
    ):
        span = _take_instanttensor_span(handle)
        offset = 0
        for (name, metadata), size in zip(
            handle.ordered_tensor_metadatas,
            handle.tensor_sizes,
            strict=True,
        ):
            tensor = _span_tensor(span, metadata)
            if handle.copy:
                tensor = tensor.clone()
            yield name, tensor
            offset += size
        if offset != handle.total_tensor_size:
            raise RuntimeError("InstantTensor tensor sizes do not cover the loaded span")
        return
    handle.iterated = True
    for tensor_index, (name, metadata) in enumerate(handle.ordered_tensor_metadatas):
        dtype = safetensors_to_torch_dtype.get(metadata["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported safetensors dtype: {metadata['dtype']}")
        stream = torch.cuda.current_stream(handle.device)
        arguments = (
            handle.loader_handle,
            tensor_index,
            handle.tensor_sizes[tensor_index],
        )
        if _GET_DL_TENSOR_HAS_CONSUMER_STREAM:
            capsule = _instanttensor_c.get_dl_tensor(
                *arguments,
                stream.cuda_stream,
            )
        else:
            capsule = _instanttensor_c.get_dl_tensor(*arguments)
        _set_dlpack_device(capsule, handle.device_idx)
        tensor = torch.from_dlpack(capsule).view(dtype).view(metadata["shape"])
        if tensor.device != handle.device:
            raise RuntimeError(
                f"InstantTensor yielded {name} on {tensor.device}, expected {handle.device}"
            )
        if tensor.data_ptr() % tensor.element_size() != 0:
            raise ValueError(f"InstantTensor yielded misaligned tensor {name}")
        if handle.copy:
            tensor = tensor.clone()
        yield name, tensor


@dataclass(frozen=True)
class InstantTensorLoadConfig:
    """Direct-I/O geometry measured on the local NVMe RAID."""

    buffer_size: int = 4 << 30
    chunk_size: int = 64 << 20
    concurrency: int = 2
    io_depth: int = 32
    backend: Backend = Backend.AIO
    whole_shard: bool = True

    def open_kwargs(self, *, total_tensor_size: int | None = None) -> dict[str, object]:
        buffer_size = self.buffer_size
        if (
            self.whole_shard
            and _GET_DL_TENSOR_SPAN is not None
            and total_tensor_size is not None
        ):
            buffer_size = int(total_tensor_size)
        return {
            "buffer_size": buffer_size,
            "chunk_size": self.chunk_size,
            "concurrency": self.concurrency,
            "io_depth": self.io_depth,
            "backend": self.backend,
            "copy": False,
        }


def _safetensors_data_size(path: Path) -> int:
    """Return the contiguous tensor-data extent without parsing its JSON header."""

    with path.open("rb", buffering=0) as handle:
        encoded_header_size = handle.read(8)
    if len(encoded_header_size) != 8:
        raise ValueError(f"safetensors header is truncated: {path}")
    header_size = int.from_bytes(encoded_header_size, byteorder="little")
    data_size = path.stat().st_size - 8 - header_size
    if data_size <= 0:
        raise ValueError(f"safetensors data section is empty: {path}")
    return data_size


@dataclass
class KimiLayerLoadStats:
    """Measured storage and materialization work for one decoder layer."""

    layer: int
    shard: str
    serialized_bytes: int = 0
    dense_bytes: int = 0
    nonexpert_parameters: int = 0
    expert_matrices: int = 0
    compatibility_fixes: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    peak_allocated_bytes: int = 0
    grouped_expert_pack_seconds: float = 0.0


@triton.jit
def _mxfp4_to_bf16_kernel(
    packed_pointer,
    scale_pointer,
    output_pointer,
    element_count: tl.constexpr,
    columns: tl.constexpr,
    packed_columns: tl.constexpr,
    scale_columns: tl.constexpr,
    block_size: tl.constexpr,
):
    positions = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = positions < element_count
    rows = positions // columns
    columns_in_row = positions - rows * columns
    packed_offsets = rows * packed_columns + columns_in_row // 2
    packed = tl.load(packed_pointer + packed_offsets, mask=mask, other=0).to(
        tl.int32
    )
    codes = (packed >> ((columns_in_row & 1) * 4)) & 15
    magnitudes = codes & 7
    values = tl.where(
        magnitudes == 0,
        0.0,
        tl.where(
            magnitudes == 1,
            0.5,
            tl.where(
                magnitudes == 2,
                1.0,
                tl.where(
                    magnitudes == 3,
                    1.5,
                    tl.where(
                        magnitudes == 4,
                        2.0,
                        tl.where(magnitudes == 5, 3.0, tl.where(magnitudes == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    scale_offsets = rows * scale_columns + columns_in_row // 32
    exponents = tl.load(scale_pointer + scale_offsets, mask=mask, other=127).to(
        tl.float32
    )
    values *= tl.exp2(exponents - 127.0)
    bits = values.to(tl.bfloat16).to(tl.uint16, bitcast=True)
    bits |= (codes & 8) << 12
    tl.store(output_pointer + positions, bits, mask=mask)


@triton.jit
def _grouped_mxfp4_to_bf16_kernel(
    packed_pointer,
    scale_pointer,
    output_pointer,
    matrix_elements: tl.constexpr,
    columns: tl.constexpr,
    packed_columns: tl.constexpr,
    scale_columns: tl.constexpr,
    packed_matrix_elements: tl.constexpr,
    scale_matrix_elements: tl.constexpr,
    block_size: tl.constexpr,
):
    positions = tl.program_id(0) * block_size + tl.arange(0, block_size)
    expert = tl.program_id(1).to(tl.int64)
    mask = positions < matrix_elements
    rows = positions // columns
    columns_in_row = positions - rows * columns
    packed_offsets = (
        expert * packed_matrix_elements
        + rows * packed_columns
        + columns_in_row // 2
    )
    packed = tl.load(packed_pointer + packed_offsets, mask=mask, other=0).to(
        tl.int32
    )
    codes = (packed >> ((columns_in_row & 1) * 4)) & 15
    magnitudes = codes & 7
    values = tl.where(
        magnitudes == 0,
        0.0,
        tl.where(
            magnitudes == 1,
            0.5,
            tl.where(
                magnitudes == 2,
                1.0,
                tl.where(
                    magnitudes == 3,
                    1.5,
                    tl.where(
                        magnitudes == 4,
                        2.0,
                        tl.where(
                            magnitudes == 5,
                            3.0,
                            tl.where(magnitudes == 6, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    scale_offsets = (
        expert * scale_matrix_elements
        + rows * scale_columns
        + columns_in_row // 32
    )
    exponents = tl.load(
        scale_pointer + scale_offsets, mask=mask, other=127
    ).to(tl.float32)
    values *= tl.exp2(exponents - 127.0)
    bits = values.to(tl.bfloat16).to(tl.uint16, bitcast=True)
    bits |= (codes & 8) << 12
    output_offsets = expert * matrix_elements + positions.to(tl.int64)
    tl.store(output_pointer + output_offsets, bits, mask=mask)


@triton.jit
def _strided_grouped_mxfp4_to_bf16_kernel(
    packed_pointer,
    scale_pointer,
    expert_order_pointer,
    output_pointer,
    matrix_elements: tl.constexpr,
    columns: tl.constexpr,
    packed_columns: tl.constexpr,
    scale_columns: tl.constexpr,
    source_expert_stride: tl.constexpr,
    block_size: tl.constexpr,
):
    positions = tl.program_id(0) * block_size + tl.arange(0, block_size)
    source_expert = tl.program_id(1).to(tl.int64)
    destination_expert = tl.load(
        expert_order_pointer + source_expert
    ).to(tl.int64)
    mask = positions < matrix_elements
    rows = positions // columns
    columns_in_row = positions - rows * columns
    source_base = source_expert * source_expert_stride
    packed_offsets = (
        source_base + rows * packed_columns + columns_in_row // 2
    )
    packed = tl.load(
        packed_pointer + packed_offsets, mask=mask, other=0
    ).to(tl.int32)
    codes = (packed >> ((columns_in_row & 1) * 4)) & 15
    magnitudes = codes & 7
    values = tl.where(
        magnitudes == 0,
        0.0,
        tl.where(
            magnitudes == 1,
            0.5,
            tl.where(
                magnitudes == 2,
                1.0,
                tl.where(
                    magnitudes == 3,
                    1.5,
                    tl.where(
                        magnitudes == 4,
                        2.0,
                        tl.where(
                            magnitudes == 5,
                            3.0,
                            tl.where(magnitudes == 6, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    scale_offsets = (
        source_base + rows * scale_columns + columns_in_row // 32
    )
    exponents = tl.load(
        scale_pointer + scale_offsets, mask=mask, other=127
    ).to(tl.float32)
    values *= tl.exp2(exponents - 127.0)
    bits = values.to(tl.bfloat16).to(tl.uint16, bitcast=True)
    bits |= (codes & 8) << 12
    output_offsets = destination_expert * matrix_elements + positions.to(tl.int64)
    tl.store(output_pointer + output_offsets, bits, mask=mask)


def decode_mxfp4_bf16_into(
    packed: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Expand an official MXFP4 matrix into an existing BF16 CUDA tensor."""

    if packed.device.type != "cuda" or scale.device != packed.device:
        raise ValueError("MXFP4 codes and scales must share one CUDA device")
    if packed.dtype != torch.uint8 or scale.dtype != torch.uint8:
        raise TypeError("MXFP4 codes and scales must be uint8")
    if packed.ndim != 2 or scale.ndim != 2:
        raise ValueError("MXFP4 codes and scales must be matrices")
    rows, packed_columns = map(int, packed.shape)
    columns = packed_columns * 2
    if tuple(scale.shape) != (rows, columns // 32):
        raise ValueError(
            f"MXFP4 scale shape {tuple(scale.shape)} does not cover "
            f"{(rows, columns)} at K32"
        )
    if not packed.is_contiguous() or not scale.is_contiguous():
        raise ValueError("MXFP4 inputs must be contiguous")

    if (
        output.device != packed.device
        or output.dtype != torch.bfloat16
        or tuple(output.shape) != (rows, columns)
        or not output.is_contiguous()
    ):
        raise ValueError("MXFP4 destination must be a matching contiguous BF16 matrix")
    element_count = output.numel()
    block_size = 256
    _mxfp4_to_bf16_kernel[(triton.cdiv(element_count, block_size),)](
        packed,
        scale,
        output.view(torch.uint16),
        element_count=element_count,
        columns=columns,
        packed_columns=packed_columns,
        scale_columns=int(scale.shape[1]),
        block_size=block_size,
        num_warps=4,
    )


def decode_grouped_mxfp4_bf16_into(
    packed: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Expand equal-shaped MXFP4 matrices into one grouped BF16 bank."""

    if packed.device.type != "cuda" or scale.device != packed.device:
        raise ValueError("grouped MXFP4 codes and scales must share one CUDA device")
    if packed.dtype != torch.uint8 or scale.dtype != torch.uint8:
        raise TypeError("grouped MXFP4 codes and scales must be uint8")
    if packed.ndim != 3 or scale.ndim != 3:
        raise ValueError("grouped MXFP4 codes and scales must be rank-three")
    experts, rows, packed_columns = map(int, packed.shape)
    columns = packed_columns * 2
    if tuple(scale.shape) != (experts, rows, columns // 32):
        raise ValueError("grouped MXFP4 scales do not cover the decoded matrices")
    if not packed.is_contiguous() or not scale.is_contiguous():
        raise ValueError("grouped MXFP4 inputs must be contiguous")
    if (
        output.device != packed.device
        or output.dtype != torch.bfloat16
        or tuple(output.shape) != (experts, rows, columns)
        or not output.is_contiguous()
    ):
        raise ValueError("grouped MXFP4 destination has incompatible geometry")
    matrix_elements = rows * columns
    block_size = 1024
    _grouped_mxfp4_to_bf16_kernel[
        (triton.cdiv(matrix_elements, block_size), experts)
    ](
        packed,
        scale,
        output.view(torch.uint16),
        matrix_elements=matrix_elements,
        columns=columns,
        packed_columns=packed_columns,
        scale_columns=int(scale.shape[2]),
        packed_matrix_elements=rows * packed_columns,
        scale_matrix_elements=rows * int(scale.shape[2]),
        block_size=block_size,
        num_warps=8,
    )


def decode_strided_grouped_mxfp4_bf16_into(
    packed: torch.Tensor,
    scale: torch.Tensor,
    expert_order: torch.Tensor,
    output: torch.Tensor,
    *,
    source_expert_stride: int,
) -> None:
    """Decode interleaved expert records directly from a stable shard span."""

    if packed.device.type != "cuda" or scale.device != packed.device:
        raise ValueError("strided MXFP4 codes and scales must share one CUDA device")
    if packed.dtype != torch.uint8 or scale.dtype != torch.uint8:
        raise TypeError("strided MXFP4 codes and scales must be uint8")
    if packed.ndim != 2 or scale.ndim != 2:
        raise ValueError("strided MXFP4 codes and scales must be matrices")
    experts = int(expert_order.numel())
    rows, packed_columns = map(int, packed.shape)
    columns = packed_columns * 2
    if tuple(scale.shape) != (rows, columns // 32):
        raise ValueError("strided MXFP4 scales do not cover the decoded matrix")
    if (
        expert_order.device != packed.device
        or expert_order.dtype != torch.int32
        or expert_order.ndim != 1
        or not expert_order.is_contiguous()
    ):
        raise ValueError("expert order must be a contiguous CUDA int32 vector")
    if (
        output.device != packed.device
        or output.dtype != torch.bfloat16
        or tuple(output.shape) != (experts, rows, columns)
        or not output.is_contiguous()
    ):
        raise ValueError("strided MXFP4 destination has incompatible geometry")
    if source_expert_stride <= packed.numel() + scale.numel():
        raise ValueError("source expert stride does not cover one MXFP4 pair")
    matrix_elements = rows * columns
    block_size = 1024
    _strided_grouped_mxfp4_to_bf16_kernel[
        (triton.cdiv(matrix_elements, block_size), experts)
    ](
        packed,
        scale,
        expert_order,
        output.view(torch.uint16),
        matrix_elements=matrix_elements,
        columns=columns,
        packed_columns=packed_columns,
        scale_columns=int(scale.shape[1]),
        source_expert_stride=int(source_expert_stride),
        block_size=block_size,
        num_warps=8,
    )


@dataclass
class _GroupedMXFP4Source:
    """One projection's regularly interleaved views in a loaded shard span."""

    experts: list[int] = field(default_factory=list)
    first_packed: torch.Tensor | None = None
    first_scale: torch.Tensor | None = None
    source_expert_stride: int | None = None

    def append(
        self,
        expert: int,
        packed: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        ordinal = len(self.experts)
        if ordinal == 0:
            self.first_packed = packed
            self.first_scale = scale
        else:
            assert self.first_packed is not None
            assert self.first_scale is not None
            packed_delta = packed.data_ptr() - self.first_packed.data_ptr()
            scale_delta = scale.data_ptr() - self.first_scale.data_ptr()
            if ordinal == 1:
                if packed_delta <= 0 or packed_delta != scale_delta:
                    raise ValueError("MXFP4 expert records are not regularly interleaved")
                self.source_expert_stride = packed_delta
            expected_delta = ordinal * int(self.source_expert_stride or 0)
            if packed_delta != expected_delta or scale_delta != expected_delta:
                raise ValueError("MXFP4 expert-record stride changed within the shard")
        self.experts.append(expert)


def decode_mxfp4_bf16(
    packed: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Expand an official MXFP4 matrix directly into a BF16 CUDA tensor."""

    if packed.ndim != 2:
        raise ValueError("MXFP4 codes must be a matrix")
    output = torch.empty(
        (int(packed.shape[0]), int(packed.shape[1]) * 2),
        dtype=torch.bfloat16,
        device=packed.device,
    )
    decode_mxfp4_bf16_into(packed, scale, output)
    return output


class OfficialKimiLayerShards:
    """Validated mapping from decoder layers to exclusive source shards."""

    def __init__(self, checkpoint: str | Path):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        index_path = self.checkpoint / "model.safetensors.index.json"
        try:
            document = json.loads(index_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read checkpoint index {index_path}: {error}") from error
        weight_map = document.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{index_path}: missing or empty weight_map")
        self.weight_map = {
            str(name): str(filename) for name, filename in weight_map.items()
        }
        self._names_by_file: dict[str, set[str]] = {}
        for name, filename in self.weight_map.items():
            self._names_by_file.setdefault(filename, set()).add(name)
        self._paths_by_file = {
            filename: self.checkpoint / filename
            for filename in self._names_by_file
            if (self.checkpoint / filename).is_file()
        }
        if (
            len(self._paths_by_file) != len(self._names_by_file)
            and self.checkpoint.parent.name == "snapshots"
        ):
            cache = resolve(
                repo_dir=self.checkpoint.parent.parent,
                revision=self.checkpoint.name,
            )
            self._paths_by_file.update(cache.shard_paths)

    @staticmethod
    def prefix(layer: int) -> str:
        if layer < 0:
            raise ValueError("decoder layer must be nonnegative")
        return f"{MODEL_TENSOR_PREFIX}.layers.{layer}."

    def layer_names(self, layer: int) -> frozenset[str]:
        prefix = self.prefix(layer)
        names = frozenset(name for name in self.weight_map if name.startswith(prefix))
        if not names:
            raise KeyError(f"checkpoint has no decoder layer {layer}")
        return names

    def tensor_shard(self, name: str) -> Path:
        """Resolve one indexed tensor to its content-addressed shard."""

        filename = self.weight_map.get(name)
        if filename is None:
            raise KeyError(name)
        path = self._paths_by_file.get(filename)
        if path is None or not path.is_file():
            raise FileNotFoundError(self.checkpoint / filename)
        return path

    def layer_shard(self, layer: int) -> Path:
        names = self.layer_names(layer)
        filenames = {self.weight_map[name] for name in names}
        if len(filenames) != 1:
            raise ValueError(
                f"decoder layer {layer} spans {len(filenames)} safetensors shards"
            )
        filename = next(iter(filenames))
        if self._names_by_file[filename] != set(names):
            foreign = sorted(self._names_by_file[filename] - set(names))
            raise ValueError(
                f"decoder layer {layer} does not exclusively own {filename}; "
                f"foreign tensors include {foreign[:3]}"
            )
        path = self._paths_by_file.get(filename)
        if path is None or not path.is_file():
            raise FileNotFoundError(self.checkpoint / filename)
        return path


def load_checkpoint_tensor_cuda(
    checkpoint: str | Path,
    name: str,
    *,
    device: torch.device | str,
    config: InstantTensorLoadConfig | None = None,
) -> torch.Tensor:
    """Load one indexed checkpoint tensor through the direct-I/O ring."""

    target = torch.device(device)
    if target.type != "cuda" or target.index is None:
        raise ValueError("InstantTensor loading requires an indexed CUDA device")
    load_config = config or InstantTensorLoadConfig()
    index = OfficialKimiLayerShards(checkpoint)
    shard = index.tensor_shard(name)
    result: torch.Tensor | None = None
    with torch.cuda.device(target):
        with safe_open(
            str(shard),
            framework="pt",
            device=target,
            **load_config.open_kwargs(),
        ) as handle:
            if name not in handle.keys():
                raise ValueError(f"checkpoint index and shard disagree for {name}")
            for tensor_name, value in _iter_instanttensor_tensors(handle):
                if tensor_name == name:
                    result = value.clone()
                    torch.cuda.current_stream(target).synchronize()
                    break
    if result is None:
        raise RuntimeError(f"InstantTensor stream did not yield {name}")
    return result


def load_checkpoint_tensors_cuda(
    checkpoint: str | Path,
    names: Iterable[str],
    *,
    device: torch.device | str,
    config: InstantTensorLoadConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Load indexed tensors while streaming each containing shard once."""

    requested = tuple(str(name) for name in names)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("checkpoint tensor names must be nonempty and unique")
    target = torch.device(device)
    if target.type != "cuda" or target.index is None:
        raise ValueError("InstantTensor loading requires an indexed CUDA device")
    load_config = config or InstantTensorLoadConfig()
    index = OfficialKimiLayerShards(checkpoint)
    by_shard: dict[Path, set[str]] = {}
    for name in requested:
        by_shard.setdefault(index.tensor_shard(name), set()).add(name)
    result: dict[str, torch.Tensor] = {}
    for shard, shard_names in by_shard.items():
        with torch.cuda.device(target):
            with safe_open(
                str(shard),
                framework="pt",
                device=target,
                **load_config.open_kwargs(),
            ) as handle:
                missing = shard_names - set(handle.keys())
                if missing:
                    raise ValueError(
                        f"checkpoint index and shard disagree for {sorted(missing)[:3]}"
                    )
                for tensor_name, value in _iter_instanttensor_tensors(handle):
                    if tensor_name in shard_names:
                        result[tensor_name] = value.clone()
            torch.cuda.current_stream(target).synchronize()
    missing = set(requested) - set(result)
    if missing:
        raise RuntimeError(f"InstantTensor stream did not yield {sorted(missing)[:3]}")
    return {name: result[name] for name in requested}


class InstantTensorKimiLayerLoader:
    """Materialize an official decoder layer with bounded temporary memory."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: torch.device | str,
        config: InstantTensorLoadConfig | None = None,
    ):
        self.shards = OfficialKimiLayerShards(checkpoint)
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("InstantTensor loading requires an indexed CUDA device")
        self.config = config or InstantTensorLoadConfig()

    def _stream(self, layer: int) -> Iterator[tuple[str, torch.Tensor]]:
        shard = self.shards.layer_shard(layer)
        expected = self.shards.layer_names(layer)
        with torch.cuda.device(self.device):
            with safe_open(
                str(shard),
                framework="pt",
                device=self.device,
                **self.config.open_kwargs(
                    total_tensor_size=_safetensors_data_size(shard)
                ),
            ) as handle:
                actual = frozenset(handle.keys())
                if actual != expected:
                    raise ValueError(
                        f"decoder layer {layer} shard inventory differs from checkpoint index"
                    )
                yield from _iter_instanttensor_tensors(handle)

    def load_expert_banks(
        self,
        *,
        layer: int,
        matrices: tuple[str, ...] = ("w1", "w3"),
    ) -> tuple[dict[str, torch.Tensor], KimiLayerLoadStats]:
        """Decode selected routed-expert matrices from one whole-shard read."""

        if (
            not matrices
            or len(set(matrices)) != len(matrices)
            or any(matrix not in C.EXPERT_MATRICES for matrix in matrices)
        ):
            raise ValueError(f"matrices must be unique members of {C.EXPERT_MATRICES}")
        if not self.config.whole_shard or _GET_DL_TENSOR_SPAN is None:
            raise RuntimeError("grouped expert-bank loading requires whole-shard support")

        started = time.monotonic()
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(self.device)
        shard = self.shards.layer_shard(layer)
        prefix = self.shards.prefix(layer)
        expected_names = self.shards.layer_names(layer)
        banks = {
            matrix: torch.empty(
                (C.NUM_EXPERTS, *C.EXPERT_SHAPES[matrix]),
                dtype=torch.bfloat16,
                device=self.device,
            )
            for matrix in matrices
        }
        stats = KimiLayerLoadStats(layer=layer, shard=str(shard))

        def metadata_for(handle: Any, name: str) -> Mapping[str, Any]:
            try:
                index = handle.tensor_name_to_index[name]
            except KeyError as error:
                raise KeyError(f"checkpoint shard has no tensor {name}") from error
            actual_name, metadata = handle.ordered_tensor_metadatas[index]
            if actual_name != name:
                raise RuntimeError("InstantTensor name index is inconsistent")
            return metadata

        source_order = sorted(range(C.NUM_EXPERTS), key=lambda value: str(value))
        expert_order = torch.tensor(
            source_order,
            dtype=torch.int32,
            device=self.device,
        )
        with torch.cuda.device(self.device):
            with safe_open(
                str(shard),
                framework="pt",
                device=self.device,
                **self.config.open_kwargs(
                    total_tensor_size=_safetensors_data_size(shard)
                ),
            ) as handle:
                if frozenset(handle.keys()) != expected_names:
                    raise ValueError(
                        f"decoder layer {layer} shard inventory differs from checkpoint index"
                    )
                span = _take_instanttensor_span(handle)
                stats.serialized_bytes = int(handle.total_tensor_size)
                decode_started = time.monotonic()
                for matrix, bank in banks.items():
                    output_features, input_features = C.EXPERT_SHAPES[matrix]
                    expected_packed = (output_features, input_features // 2)
                    expected_scale = (
                        output_features,
                        input_features // C.MXFP4_BLOCK,
                    )
                    first_expert, second_expert = source_order[:2]
                    first_base = (
                        f"{prefix}block_sparse_moe.experts.{first_expert}."
                        f"{matrix}.weight"
                    )
                    second_base = (
                        f"{prefix}block_sparse_moe.experts.{second_expert}."
                        f"{matrix}.weight"
                    )
                    first_packed = metadata_for(handle, f"{first_base}_packed")
                    first_scale = metadata_for(handle, f"{first_base}_scale")
                    second_packed = metadata_for(handle, f"{second_base}_packed")
                    second_scale = metadata_for(handle, f"{second_base}_scale")
                    packed_start = int(first_packed["data_offsets"][0])
                    scale_start = int(first_scale["data_offsets"][0])
                    source_expert_stride = (
                        int(second_packed["data_offsets"][0]) - packed_start
                    )
                    if (
                        int(second_scale["data_offsets"][0]) - scale_start
                        != source_expert_stride
                    ):
                        raise ValueError(
                            f"{matrix} packed and scale expert strides differ"
                        )

                    for ordinal, expert in enumerate(source_order):
                        base = (
                            f"{prefix}block_sparse_moe.experts.{expert}."
                            f"{matrix}.weight"
                        )
                        packed = metadata_for(handle, f"{base}_packed")
                        scale = metadata_for(handle, f"{base}_scale")
                        if (
                            packed["dtype"] != "U8"
                            or tuple(packed["shape"]) != expected_packed
                            or scale["dtype"] != "U8"
                            or tuple(scale["shape"]) != expected_scale
                            or int(packed["data_offsets"][0])
                            != packed_start + ordinal * source_expert_stride
                            or int(scale["data_offsets"][0])
                            != scale_start + ordinal * source_expert_stride
                        ):
                            raise ValueError(
                                f"expert {expert} {matrix} violates the regular shard layout"
                            )

                    decode_strided_grouped_mxfp4_bf16_into(
                        _span_tensor(span, first_packed),
                        _span_tensor(span, first_scale),
                        expert_order,
                        bank,
                        source_expert_stride=source_expert_stride,
                    )
                    stats.dense_bytes += bank.numel() * bank.element_size()
                    stats.expert_matrices += C.NUM_EXPERTS
                torch.cuda.current_stream(self.device).synchronize()
                stats.grouped_expert_pack_seconds = (
                    time.monotonic() - decode_started
                )

        stats.elapsed_seconds = time.monotonic() - started
        stats.peak_allocated_bytes = torch.cuda.max_memory_allocated(self.device)
        return banks, stats

    def _load_grouped_whole_span(
        self,
        module: torch.nn.Module,
        *,
        layer: int,
        expert_weight_banks: Mapping[str, torch.Tensor],
    ) -> KimiLayerLoadStats:
        """Materialize grouped expert banks without per-tensor Python views."""

        started = time.monotonic()
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(self.device)
        shard = self.shards.layer_shard(layer)
        prefix = self.shards.prefix(layer)
        expected_names = self.shards.layer_names(layer)
        parameters = dict(module.named_parameters())
        nonexpert = {
            f"{prefix}{name}": (name, parameter)
            for name, parameter in parameters.items()
            if not name.startswith("block_sparse_moe.experts.")
        }
        expert_prototypes = {
            matrix: parameters.get(
                f"block_sparse_moe.experts.0.{matrix}.weight"
            )
            for matrix in ("w1", "w2", "w3")
        }
        stats = KimiLayerLoadStats(layer=layer, shard=str(shard))

        def metadata_for(handle: Any, name: str) -> Mapping[str, Any]:
            try:
                index = handle.tensor_name_to_index[name]
            except KeyError as error:
                raise KeyError(f"checkpoint shard has no tensor {name}") from error
            actual_name, metadata = handle.ordered_tensor_metadatas[index]
            if actual_name != name:
                raise RuntimeError("InstantTensor name index is inconsistent")
            return metadata

        with torch.cuda.device(self.device):
            with safe_open(
                str(shard),
                framework="pt",
                device=self.device,
                **self.config.open_kwargs(
                    total_tensor_size=_safetensors_data_size(shard)
                ),
            ) as handle:
                if frozenset(handle.keys()) != expected_names:
                    raise ValueError(
                        f"decoder layer {layer} shard inventory differs from checkpoint index"
                    )
                span = _take_instanttensor_span(handle)
                stats.serialized_bytes = int(handle.total_tensor_size)

                for checkpoint_name, (parameter_name, parameter) in nonexpert.items():
                    metadata = metadata_for(handle, checkpoint_name)
                    value = _span_tensor(span, metadata)
                    fitted, fix = fit_checkpoint_parameter(
                        checkpoint_name,
                        value,
                        tuple(parameter.shape),
                    )
                    owned = fitted.clone()
                    assign_parameter(module, parameter_name, owned)
                    stats.dense_bytes += owned.numel() * owned.element_size()
                    stats.nonexpert_parameters += 1
                    if fix is not None:
                        stats.compatibility_fixes.append(
                            f"{checkpoint_name}: {fix}"
                        )

                expert_counts = {
                    int(bank.shape[0]) for bank in expert_weight_banks.values()
                }
                if len(expert_counts) != 1:
                    raise ValueError("grouped expert banks have different expert counts")
                experts = next(iter(expert_counts))
                source_order = sorted(range(experts), key=lambda value: str(value))
                if experts < 2:
                    raise ValueError("whole-span grouped decoding requires multiple experts")
                expert_order = torch.tensor(
                    source_order,
                    dtype=torch.int32,
                    device=self.device,
                )
                decode_started = time.monotonic()
                for matrix in ("w1", "w2", "w3"):
                    parameter = expert_prototypes[matrix]
                    bank = expert_weight_banks.get(matrix)
                    if parameter is None or bank is None:
                        raise ValueError(f"grouped expert bank is missing {matrix}")
                    if parameter.dtype != torch.bfloat16:
                        raise TypeError(
                            f"expert prototype {matrix} must be BF16, got {parameter.dtype}"
                        )
                    expected_shape = tuple(map(int, parameter.shape))
                    if (
                        bank.device != self.device
                        or bank.dtype != parameter.dtype
                        or tuple(bank.shape) != (experts, *expected_shape)
                        or not bank.is_contiguous()
                    ):
                        raise ValueError(
                            f"grouped expert bank has incompatible {matrix} geometry"
                        )
                    expected_packed = (expected_shape[0], expected_shape[1] // 2)
                    expected_scale = (expected_shape[0], expected_shape[1] // 32)

                    first_expert = source_order[0]
                    second_expert = source_order[1]
                    first_base = (
                        f"{prefix}block_sparse_moe.experts.{first_expert}."
                        f"{matrix}.weight"
                    )
                    second_base = (
                        f"{prefix}block_sparse_moe.experts.{second_expert}."
                        f"{matrix}.weight"
                    )
                    first_packed_metadata = metadata_for(
                        handle, f"{first_base}_packed"
                    )
                    first_scale_metadata = metadata_for(
                        handle, f"{first_base}_scale"
                    )
                    second_packed_metadata = metadata_for(
                        handle, f"{second_base}_packed"
                    )
                    second_scale_metadata = metadata_for(
                        handle, f"{second_base}_scale"
                    )
                    packed_start = int(first_packed_metadata["data_offsets"][0])
                    scale_start = int(first_scale_metadata["data_offsets"][0])
                    source_expert_stride = (
                        int(second_packed_metadata["data_offsets"][0])
                        - packed_start
                    )
                    if (
                        int(second_scale_metadata["data_offsets"][0])
                        - scale_start
                        != source_expert_stride
                    ):
                        raise ValueError(
                            f"{matrix} packed and scale expert strides differ"
                        )

                    for ordinal, expert in enumerate(source_order):
                        base = (
                            f"{prefix}block_sparse_moe.experts.{expert}."
                            f"{matrix}.weight"
                        )
                        packed_metadata = metadata_for(handle, f"{base}_packed")
                        scale_metadata = metadata_for(handle, f"{base}_scale")
                        if (
                            packed_metadata["dtype"] != "U8"
                            or tuple(packed_metadata["shape"]) != expected_packed
                            or scale_metadata["dtype"] != "U8"
                            or tuple(scale_metadata["shape"]) != expected_scale
                        ):
                            raise ValueError(
                                f"expert {expert} {matrix} has incompatible MXFP4 geometry"
                            )
                        if (
                            int(packed_metadata["data_offsets"][0])
                            != packed_start + ordinal * source_expert_stride
                            or int(scale_metadata["data_offsets"][0])
                            != scale_start + ordinal * source_expert_stride
                        ):
                            raise ValueError(
                                f"expert {expert} {matrix} violates the regular shard layout"
                            )

                    packed = _span_tensor(span, first_packed_metadata)
                    scale = _span_tensor(span, first_scale_metadata)
                    decode_strided_grouped_mxfp4_bf16_into(
                        packed,
                        scale,
                        expert_order,
                        bank,
                        source_expert_stride=source_expert_stride,
                    )
                    stats.dense_bytes += bank.numel() * bank.element_size()
                    stats.expert_matrices += experts

                torch.cuda.current_stream(self.device).synchronize()
                stats.grouped_expert_pack_seconds = (
                    time.monotonic() - decode_started
                )

        meta_parameters = [
            name
            for name, parameter in module.named_parameters()
            if parameter.is_meta
            and not name.startswith("block_sparse_moe.experts.")
        ]
        meta_buffers = [
            name for name, buffer in module.named_buffers() if buffer.is_meta
        ]
        if meta_parameters or meta_buffers:
            raise RuntimeError(
                f"decoder layer {layer} retains meta state; "
                f"parameters={meta_parameters[:3]}, buffers={meta_buffers[:3]}"
            )
        stats.elapsed_seconds = time.monotonic() - started
        stats.peak_allocated_bytes = torch.cuda.max_memory_allocated(self.device)
        return stats

    def load(
        self,
        module: torch.nn.Module,
        *,
        layer: int,
        expert_weight_banks: Mapping[str, torch.Tensor] | None = None,
    ) -> KimiLayerLoadStats:
        """Populate a meta-initialized official layer and return load statistics."""

        if (
            expert_weight_banks
            and self.config.whole_shard
            and _GET_DL_TENSOR_SPAN is not None
        ):
            return self._load_grouped_whole_span(
                module,
                layer=layer,
                expert_weight_banks=expert_weight_banks,
            )

        started = time.monotonic()
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(self.device)
        shard = self.shards.layer_shard(layer)
        prefix = self.shards.prefix(layer)
        parameters = dict(module.named_parameters())
        nonexpert = {
            f"{prefix}{name}": (name, parameter)
            for name, parameter in parameters.items()
            if not name.startswith("block_sparse_moe.experts.")
        }
        expected_expert_parameters = {
            name
            for name in parameters
            if name.startswith("block_sparse_moe.experts.")
        }
        expert_prototypes = {
            matrix: parameters.get(
                f"block_sparse_moe.experts.0.{matrix}.weight"
            )
            for matrix in ("w1", "w2", "w3")
        }
        assigned_nonexpert: set[str] = set()
        assigned_expert: set[str] = set()
        pending: tuple[int, str, torch.Tensor, str] | None = None
        source_views_are_stable = (
            self.config.whole_shard and _GET_DL_TENSOR_SPAN is not None
        )
        grouped_sources = {
            matrix: _GroupedMXFP4Source() for matrix in ("w1", "w2", "w3")
        }
        completed_groups: set[str] = set()
        stats = KimiLayerLoadStats(layer=layer, shard=str(shard))

        for checkpoint_name, value in self._stream(layer):
            stats.serialized_bytes += value.numel() * value.element_size()
            if not checkpoint_name.startswith(prefix):
                raise ValueError(f"unexpected tensor outside layer prefix: {checkpoint_name}")
            local_name = checkpoint_name[len(prefix) :]
            match = _EXPERT_PART.fullmatch(local_name)
            if match is None:
                if pending is not None:
                    raise ValueError(
                        f"MXFP4 packed tensor {pending[3]} is not followed by its scale"
                    )
                item = nonexpert.get(checkpoint_name)
                if item is None:
                    raise KeyError(f"layer module has no parameter for {checkpoint_name}")
                parameter_name, parameter = item
                fitted, fix = fit_checkpoint_parameter(
                    checkpoint_name, value, tuple(parameter.shape)
                )
                owned = fitted.clone()
                assign_parameter(module, parameter_name, owned)
                assigned_nonexpert.add(checkpoint_name)
                stats.dense_bytes += owned.numel() * owned.element_size()
                stats.nonexpert_parameters += 1
                if fix is not None:
                    stats.compatibility_fixes.append(f"{checkpoint_name}: {fix}")
                continue

            expert = int(match.group("expert"))
            matrix = match.group("matrix")
            part = match.group("part")
            dense_name = f"block_sparse_moe.experts.{expert}.{matrix}.weight"
            parameter = parameters.get(dense_name)
            if parameter is None and expert_weight_banks is not None:
                parameter = expert_prototypes[matrix]
            if parameter is None:
                raise KeyError(f"layer module has no parameter {dense_name}")
            if parameter.dtype != torch.bfloat16:
                raise TypeError(f"{dense_name} must be BF16, got {parameter.dtype}")

            if part == "packed":
                if pending is not None:
                    raise ValueError(
                        f"MXFP4 packed tensor {pending[3]} has no adjacent scale"
                    )
                packed = value if source_views_are_stable else value.clone()
                pending = (expert, matrix, packed, checkpoint_name)
                continue

            if pending is None:
                raise ValueError(f"MXFP4 scale has no preceding packed tensor: {checkpoint_name}")
            packed_expert, packed_matrix, packed, packed_name = pending
            if (expert, matrix) != (packed_expert, packed_matrix):
                raise ValueError(
                    f"MXFP4 pair mismatch: {packed_name} followed by {checkpoint_name}"
                )
            expected_packed = (int(parameter.shape[0]), int(parameter.shape[1]) // 2)
            expected_scale = (int(parameter.shape[0]), int(parameter.shape[1]) // 32)
            if tuple(packed.shape) != expected_packed or tuple(value.shape) != expected_scale:
                raise ValueError(
                    f"{dense_name}: packed/scale shapes {tuple(packed.shape)}, "
                    f"{tuple(value.shape)} do not cover {tuple(parameter.shape)}"
                )
            if expert_weight_banks is None:
                dense = decode_mxfp4_bf16(packed, value)
            else:
                bank = expert_weight_banks.get(matrix)
                if bank is None or expert >= bank.shape[0]:
                    raise ValueError(
                        f"grouped expert bank does not cover {dense_name}"
                    )
                dense = bank[expert]
                if (
                    dense.device != self.device
                    or dense.dtype != parameter.dtype
                    or tuple(dense.shape) != tuple(parameter.shape)
                ):
                    raise ValueError(
                        f"grouped expert bank has incompatible {matrix} geometry"
                    )
                if source_views_are_stable:
                    source = grouped_sources[matrix]
                    source.append(expert, packed, value)
                    if len(source.experts) == int(bank.shape[0]):
                        pack_started = time.monotonic()
                        expected_experts = list(range(int(bank.shape[0])))
                        if sorted(source.experts) != expected_experts:
                            raise ValueError(
                                f"grouped {matrix} sources do not cover every expert"
                            )
                        if (
                            source.first_packed is None
                            or source.first_scale is None
                            or source.source_expert_stride is None
                        ):
                            raise RuntimeError(
                                f"grouped {matrix} source geometry is incomplete"
                            )
                        expert_order = torch.tensor(
                            source.experts,
                            dtype=torch.int32,
                            device=self.device,
                        )
                        decode_strided_grouped_mxfp4_bf16_into(
                            source.first_packed,
                            source.first_scale,
                            expert_order,
                            bank,
                            source_expert_stride=source.source_expert_stride,
                        )
                        stats.grouped_expert_pack_seconds += (
                            time.monotonic() - pack_started
                        )
                        completed_groups.add(matrix)
                        if len(completed_groups) == len(grouped_sources):
                            torch.cuda.current_stream(self.device).synchronize()
                else:
                    decode_mxfp4_bf16_into(packed, value, dense)
            if expert_weight_banks is None:
                assign_parameter(module, dense_name, dense)
            assigned_expert.add(dense_name)
            stats.dense_bytes += dense.numel() * dense.element_size()
            stats.expert_matrices += 1
            pending = None

        if pending is not None:
            raise ValueError(f"MXFP4 packed tensor {pending[3]} has no scale")
        incomplete_groups = {
            matrix: len(source.experts)
            for matrix, source in grouped_sources.items()
            if source.experts and matrix not in completed_groups
        }
        if incomplete_groups:
            raise RuntimeError(
                f"incomplete grouped expert sources: {incomplete_groups}"
            )
        missing_nonexpert = sorted(set(nonexpert) - assigned_nonexpert)
        missing_expert = sorted(expected_expert_parameters - assigned_expert)
        if missing_nonexpert or missing_expert:
            raise RuntimeError(
                f"decoder layer {layer} was not fully materialized; "
                f"missing non-expert={missing_nonexpert[:3]}, "
                f"missing expert={missing_expert[:3]}"
            )
        bank_backed_parameters = (
            assigned_expert if expert_weight_banks is not None else set()
        )
        meta_parameters = [
            name
            for name, parameter in module.named_parameters()
            if parameter.is_meta and name not in bank_backed_parameters
        ]
        meta_buffers = [name for name, buffer in module.named_buffers() if buffer.is_meta]
        if meta_parameters or meta_buffers:
            raise RuntimeError(
                f"decoder layer {layer} retains meta state; "
                f"parameters={meta_parameters[:3]}, buffers={meta_buffers[:3]}"
            )
        stats.elapsed_seconds = time.monotonic() - started
        stats.peak_allocated_bytes = torch.cuda.max_memory_allocated(self.device)
        return stats


def release_layer(module: torch.nn.Module) -> None:
    """Release every parameter in a materialized layer back to meta storage."""

    block = getattr(module, "block_sparse_moe", None)
    if block is not None:
        grouped = bool(getattr(block, "_qsrt_grouped_expert_dispatch", False))
        for name in (
            "_qsrt_expert_weight_banks",
            "_qsrt_grouped_preactivation_gradient_callback",
            "_qsrt_grouped_output_gradient_callback",
            "_qsrt_routed_output_gradient_callback",
            "_qsrt_grouped_expert_dispatch",
        ):
            if hasattr(block, name):
                delattr(block, name)
        if grouped and "moe_infer" in block.__dict__:
            delattr(block, "moe_infer")
    for name, parameter in list(module.named_parameters()):
        if parameter.is_meta:
            continue
        assign_parameter(
            module,
            name,
            torch.empty(tuple(parameter.shape), dtype=parameter.dtype, device="meta"),
        )

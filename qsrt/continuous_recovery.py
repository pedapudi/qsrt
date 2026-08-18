"""Static contracts for continuous recovery of a frozen QSRT checkpoint.

Continuous recovery keeps every routed-expert payload fixed and optimizes the
ordinary parameters in a decoder suffix.  This module derives the exact
trainable tensor inventory from a materialized checkpoint without loading the
tensor contents.  It also accounts for every hidden vector required to replay
a suffix boundary under Kimi's attention-residual construction.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from safetensors import safe_open

from qsrt.kimi_official_fisher import SUFFIX_TENSORS


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class TrainableTensor:
    """One parameter in the suffix recovery overlay."""

    name: str
    group: str
    owner: str
    source_dtype: str
    runtime_dtype: str
    source_shape: tuple[int, ...]
    runtime_shape: tuple[int, ...]
    parameters: int
    source_bytes: int
    parameter_bytes: int
    gradient_bytes: int
    fp32_master_bytes: int
    adam_moment_bytes: int

    @property
    def training_bytes(self) -> int:
        return (
            self.parameter_bytes
            + self.gradient_bytes
            + self.fp32_master_bytes
            + self.adam_moment_bytes
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["training_bytes"] = self.training_bytes
        return value


@dataclass(frozen=True)
class ExcludedTensor:
    """One serialized suffix tensor that is not optimized by gradient descent."""

    name: str
    reason: str
    dtype: str
    shape: tuple[int, ...]
    parameters: int
    serialized_bytes: int


@dataclass(frozen=True)
class SuffixCaptureStorage:
    """Exact uncompressed BF16 storage required for suffix replay."""

    token_count: int
    hidden_dimension: int
    first_layer: int
    attention_residual_block_size: int
    student_hidden_vectors_per_token: int
    student_bytes: int
    teacher_hidden_vectors_per_token: int
    teacher_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.student_bytes + self.teacher_bytes

    def to_dict(self) -> dict[str, int]:
        value = asdict(self)
        value["total_bytes"] = self.total_bytes
        return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _shape_size(shape: Iterable[int]) -> int:
    return math.prod(int(value) for value in shape)


def _dtype_bytes(dtype: str) -> int:
    try:
        return _DTYPE_BYTES[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported safetensors dtype {dtype!r}") from error


def _layer_number(name: str) -> int | None:
    prefix = "language_model.model.layers."
    if not name.startswith(prefix):
        return None
    tail = name[len(prefix) :]
    number, separator, _rest = tail.partition(".")
    if not separator or not number.isdigit():
        return None
    return int(number)


def _group(name: str) -> str:
    if name in SUFFIX_TENSORS.values():
        return "output_suffix"
    if ".block_sparse_moe.shared_experts." in name:
        return "shared_experts"
    if ".block_sparse_moe.gate.weight" in name:
        return "router_gate"
    if ".block_sparse_moe.routed_expert_" in name:
        return "routed_expert_interface"
    if ".self_attn." in name:
        return "attention"
    return "residual_and_norm"


def _runtime_dtype(source_dtype: str) -> str:
    # The QSRT forward adapter expands serialized MXFP8 linears to BF16 before
    # installing them as ordinary parameters. Other parameter dtypes survive.
    return "BF16" if source_dtype in {"F8_E4M3", "F8_E5M2"} else source_dtype


def _runtime_shape(
    name: str,
    source_shape: tuple[int, ...],
    *,
    kda_heads: int,
) -> tuple[int, ...]:
    # The official checkpoint pads KDA A_log to 128 entries. The model owns
    # only num_heads entries and the loader rejects a nonzero padded tail.
    if name.endswith(".self_attn.A_log") and source_shape == (128,):
        if not 0 < kda_heads <= 128:
            raise ValueError(f"invalid KDA head count {kda_heads}")
        return (kda_heads,)
    return source_shape


def _exclusion_reason(name: str) -> str | None:
    if name.endswith(".weight_scale"):
        return "serialized MXFP8 scale auxiliary"
    if name.endswith(".block_sparse_moe.gate.e_score_correction_bias"):
        return "selection-only router correction; optimized by frequency matching"
    if ".block_sparse_moe.experts." in name:
        return "frozen routed-expert payload"
    return None


def _tensor_headers(
    checkpoint: Path,
    names: Iterable[str],
    weight_map: dict[str, str],
) -> dict[str, tuple[str, tuple[int, ...]]]:
    requested = tuple(sorted(set(names)))
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name in requested:
        try:
            by_shard[weight_map[name]].append(name)
        except KeyError as error:
            raise KeyError(f"checkpoint index has no tensor {name}") from error

    result: dict[str, tuple[str, tuple[int, ...]]] = {}
    for filename, shard_names in by_shard.items():
        shard = checkpoint / filename
        if not shard.is_file():
            raise FileNotFoundError(shard)
        with safe_open(str(shard), framework="pt") as handle:
            keys = set(handle.keys())
            missing = set(shard_names) - keys
            if missing:
                raise ValueError(
                    f"checkpoint index and {shard} disagree; missing={sorted(missing)[:3]}"
                )
            for name in shard_names:
                tensor = handle.get_slice(name)
                result[name] = (
                    str(tensor.get_dtype()),
                    tuple(int(value) for value in tensor.get_shape()),
                )
    return result


def suffix_capture_storage(
    token_count: int,
    *,
    hidden_dimension: int,
    first_layer: int,
    attention_residual_block_size: int,
) -> SuffixCaptureStorage:
    """Return exact BF16 bytes for student suffix state and teacher targets.

    At a segment boundary, Kimi carries one chain hidden plus every preceding
    residual-segment prefix. The residual prefixes occur at boundaries
    ``0, block_size, ..., first_layer - block_size``.
    """

    if token_count <= 0 or hidden_dimension <= 0:
        raise ValueError("token count and hidden dimension must be positive")
    if first_layer < 0 or attention_residual_block_size <= 0:
        raise ValueError("suffix layer and residual block size are invalid")
    if first_layer % attention_residual_block_size:
        raise ValueError("suffix cut must be an attention-residual segment boundary")
    residual_vectors = first_layer // attention_residual_block_size
    student_vectors = 1 + residual_vectors
    vector_bytes = token_count * hidden_dimension * 2
    return SuffixCaptureStorage(
        token_count=token_count,
        hidden_dimension=hidden_dimension,
        first_layer=first_layer,
        attention_residual_block_size=attention_residual_block_size,
        student_hidden_vectors_per_token=student_vectors,
        student_bytes=student_vectors * vector_bytes,
        teacher_hidden_vectors_per_token=1,
        teacher_bytes=vector_bytes,
    )


def audit_continuous_recovery(
    checkpoint: str | Path,
    *,
    first_layer: int = 84,
    end_layer: int = 93,
    capture_token_count: int = 50_000_000,
) -> dict[str, Any]:
    """Derive the suffix overlay, optimizer, and capture-storage contracts."""

    root = Path(checkpoint).expanduser().resolve()
    index = _read_json(root / "model.safetensors.index.json")
    raw_weight_map = index.get("weight_map")
    if not isinstance(raw_weight_map, dict) or not raw_weight_map:
        raise ValueError("checkpoint index has no nonempty weight_map")
    weight_map = {str(name): str(filename) for name, filename in raw_weight_map.items()}
    config = _read_json(root / "config.json")
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("checkpoint config has no text_config object")
    num_layers = int(text_config["num_hidden_layers"])
    hidden_dimension = int(text_config["hidden_size"])
    block_size = int(text_config["attn_res_block_size"])
    linear_attention = text_config.get("linear_attn_config")
    if not isinstance(linear_attention, dict):
        raise ValueError("checkpoint config has no linear_attn_config object")
    kda_heads = int(linear_attention["num_heads"])
    if not 0 <= first_layer < end_layer <= num_layers:
        raise ValueError(
            f"suffix range [{first_layer}, {end_layer}) exceeds {num_layers} layers"
        )
    if first_layer % block_size:
        raise ValueError(
            f"suffix layer {first_layer} is not aligned to residual block size {block_size}"
        )

    terminal = set(SUFFIX_TENSORS.values())
    missing_terminal = terminal - weight_map.keys()
    if missing_terminal:
        raise ValueError(f"checkpoint omits suffix tensors {sorted(missing_terminal)}")
    suffix_names = [
        name
        for name in weight_map
        if (
            (layer := _layer_number(name)) is not None
            and first_layer <= layer < end_layer
        )
        or name in terminal
    ]
    present_layers = {_layer_number(name) for name in suffix_names}
    missing_layers = set(range(first_layer, end_layer)) - present_layers
    if missing_layers:
        raise ValueError(f"checkpoint omits suffix decoder layers {sorted(missing_layers)}")
    headers = _tensor_headers(root, suffix_names, weight_map)

    trainable: list[TrainableTensor] = []
    excluded: list[ExcludedTensor] = []
    for name in sorted(suffix_names):
        source_dtype, source_shape = headers[name]
        source_parameters = _shape_size(source_shape)
        source_bytes = source_parameters * _dtype_bytes(source_dtype)
        reason = _exclusion_reason(name)
        if reason is not None:
            excluded.append(
                ExcludedTensor(
                    name=name,
                    reason=reason,
                    dtype=source_dtype,
                    shape=source_shape,
                    parameters=source_parameters,
                    serialized_bytes=source_bytes,
                )
            )
            continue
        runtime_dtype = _runtime_dtype(source_dtype)
        runtime_shape = _runtime_shape(name, source_shape, kda_heads=kda_heads)
        parameters = _shape_size(runtime_shape)
        runtime_bytes = parameters * _dtype_bytes(runtime_dtype)
        layer = _layer_number(name)
        trainable.append(
            TrainableTensor(
                name=name,
                group=_group(name),
                owner=f"layer_{layer:03d}" if layer is not None else "output_suffix",
                source_dtype=source_dtype,
                runtime_dtype=runtime_dtype,
                source_shape=source_shape,
                runtime_shape=runtime_shape,
                parameters=parameters,
                source_bytes=source_bytes,
                parameter_bytes=runtime_bytes,
                gradient_bytes=runtime_bytes,
                fp32_master_bytes=parameters * 4,
                adam_moment_bytes=parameters * 8,
            )
        )

    def summarize(items: Iterable[TrainableTensor]) -> dict[str, int]:
        values = tuple(items)
        return {
            "tensor_count": len(values),
            "parameters": sum(item.parameters for item in values),
            "source_bytes": sum(item.source_bytes for item in values),
            "parameter_bytes": sum(item.parameter_bytes for item in values),
            "gradient_bytes": sum(item.gradient_bytes for item in values),
            "fp32_master_bytes": sum(item.fp32_master_bytes for item in values),
            "adam_moment_bytes": sum(item.adam_moment_bytes for item in values),
            "training_bytes": sum(item.training_bytes for item in values),
        }

    groups = {
        group: summarize(item for item in trainable if item.group == group)
        for group in sorted({item.group for item in trainable})
    }
    owners = {
        owner: summarize(item for item in trainable if item.owner == owner)
        for owner in sorted({item.owner for item in trainable})
    }
    capture = suffix_capture_storage(
        capture_token_count,
        hidden_dimension=hidden_dimension,
        first_layer=first_layer,
        attention_residual_block_size=block_size,
    )
    excluded_by_reason: dict[str, dict[str, int]] = {}
    for reason in sorted({item.reason for item in excluded}):
        values = [item for item in excluded if item.reason == reason]
        excluded_by_reason[reason] = {
            "tensor_count": len(values),
            "serialized_bytes": sum(item.serialized_bytes for item in values),
        }
    return {
        "schema_version": 1,
        "checkpoint": str(root),
        "suffix": {
            "first_layer": first_layer,
            "end_layer": end_layer,
            "decoder_layers": end_layer - first_layer,
            "attention_residual_block_size": block_size,
            "hidden_dimension": hidden_dimension,
        },
        "trainable": {
            "total": summarize(trainable),
            "by_group": groups,
            "by_owner": owners,
            "tensors": [item.to_dict() for item in trainable],
        },
        "excluded": {
            "total_tensor_count": len(excluded),
            "by_reason": excluded_by_reason,
            "tensors": [asdict(item) for item in excluded],
        },
        "capture_storage": capture.to_dict(),
    }


__all__ = [
    "ExcludedTensor",
    "SuffixCaptureStorage",
    "TrainableTensor",
    "audit_continuous_recovery",
    "suffix_capture_storage",
]

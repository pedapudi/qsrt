"""Exact final-objective gradients for coupled QSRT expert weights."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from qsrt.qsrt_coupled import CoupledHadamardExecution, CoupledHadamardSpec


SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
PARTIAL_MANIFEST_FILENAME = ".manifest.partial.json"


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class DenseObjectiveGradientConfig:
    """Geometry of one bounded expert shard."""

    num_experts: int
    expert_begin: int
    expert_end: int
    hidden_dimension: int
    intermediate_dimension: int
    intermediate_draws: tuple[int, ...]
    spec: CoupledHadamardSpec = CoupledHadamardSpec()

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError("expert count must be positive")
        if not 0 <= self.expert_begin < self.expert_end <= self.num_experts:
            raise ValueError("expert shard is outside the routed-expert range")
        if self.hidden_dimension <= 0 or self.intermediate_dimension <= 0:
            raise ValueError("expert dimensions must be positive")
        if len(self.intermediate_draws) != self.num_experts:
            raise ValueError("intermediate draws must cover every expert")
        if any(not 0 <= int(value) < 8 for value in self.intermediate_draws):
            raise ValueError("intermediate draws must lie in 0..7")

    @property
    def shard_experts(self) -> int:
        return self.expert_end - self.expert_begin

    def spec_for_expert(self, expert: int) -> CoupledHadamardSpec:
        if not self.expert_begin <= expert < self.expert_end:
            raise IndexError("expert does not belong to this gradient shard")
        return CoupledHadamardSpec(
            residual_block_size=self.spec.residual_block_size,
            preactivation_block_size=self.spec.preactivation_block_size,
            postactivation_block_size=self.spec.postactivation_block_size,
            residual_draw=self.spec.residual_draw,
            intermediate_draw=int(self.intermediate_draws[expert]),
        )


class DenseObjectiveGradientAccumulator:
    """Accumulate exact W1/W3/W2 gradients in stored coupled coordinates."""

    def __init__(
        self,
        config: DenseObjectiveGradientConfig,
        *,
        device: torch.device | str,
    ):
        target = torch.device(device)
        if target.type not in ("cpu", "cuda"):
            raise ValueError("gradient accumulation requires CPU or CUDA")
        self.config = config
        self.device = target
        self._executions: dict[int, CoupledHadamardExecution] = {}
        self.upstream = torch.zeros(
            (
                config.shard_experts,
                2 * config.intermediate_dimension,
                config.hidden_dimension,
            ),
            dtype=torch.float32,
            device=target,
        )
        self.down = torch.zeros(
            (
                config.shard_experts,
                config.hidden_dimension,
                config.intermediate_dimension,
            ),
            dtype=torch.float32,
            device=target,
        )
        self.upstream_rows = torch.zeros(
            config.shard_experts,
            dtype=torch.int64,
            device=target,
        )
        self.down_rows = torch.zeros_like(self.upstream_rows)

    def execution_for_expert(self, expert: int) -> CoupledHadamardExecution:
        draw = int(self.config.intermediate_draws[expert])
        execution = self._executions.get(draw)
        if execution is None:
            execution = CoupledHadamardExecution(
                self.config.hidden_dimension,
                self.config.intermediate_dimension,
                self.config.spec_for_expert(expert),
            )
            self._executions[draw] = execution
        return execution

    @torch.no_grad()
    def add_upstream(
        self,
        expert: int,
        input_rows: torch.Tensor,
        gate_up_gradient: torch.Tensor,
    ) -> None:
        if not self.config.expert_begin <= expert < self.config.expert_end:
            return
        if input_rows.device != self.device or gate_up_gradient.device != self.device:
            raise ValueError("upstream callback tensors are on the wrong device")
        if tuple(input_rows.shape[1:]) != (self.config.hidden_dimension,):
            raise ValueError("expert inputs have incompatible geometry")
        if tuple(gate_up_gradient.shape) != (
            input_rows.shape[0],
            2 * self.config.intermediate_dimension,
        ):
            raise ValueError("gate/up gradients have incompatible geometry")
        execution = self.execution_for_expert(expert)
        inputs = execution.transform_inputs(input_rows.detach()).float()
        gradients = execution.transform_preactivation_gradients(
            gate_up_gradient.detach()
        ).float()
        local = expert - self.config.expert_begin
        self.upstream[local].addmm_(gradients.T, inputs)
        self.upstream_rows[local] += inputs.shape[0]

    @torch.no_grad()
    def add_down(
        self,
        expert: int,
        postactivation_rows: torch.Tensor,
        output_gradient: torch.Tensor,
    ) -> None:
        if not self.config.expert_begin <= expert < self.config.expert_end:
            return
        if (
            postactivation_rows.device != self.device
            or output_gradient.device != self.device
        ):
            raise ValueError("down callback tensors are on the wrong device")
        if tuple(postactivation_rows.shape[1:]) != (
            self.config.intermediate_dimension,
        ):
            raise ValueError("postactivation rows have incompatible geometry")
        if tuple(output_gradient.shape) != (
            postactivation_rows.shape[0],
            self.config.hidden_dimension,
        ):
            raise ValueError("expert-output gradients have incompatible geometry")
        execution = self.execution_for_expert(expert)
        inputs = execution.transform_postactivation_rows(
            postactivation_rows.detach()
        ).float()
        gradients = execution.transform_expert_output_gradients(
            output_gradient.detach()
        ).float()
        local = expert - self.config.expert_begin
        self.down[local].addmm_(gradients.T, inputs)
        self.down_rows[local] += inputs.shape[0]

    @staticmethod
    def _boundaries(offsets: torch.Tensor) -> list[int]:
        if offsets.ndim != 1 or offsets.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "grouped expert offsets must be one-dimensional integer values"
            )
        return [int(value) for value in offsets.detach().cpu().tolist()]

    @torch.no_grad()
    def add_grouped_upstream(
        self,
        sorted_experts: torch.Tensor,
        offsets: torch.Tensor,
        input_rows: torch.Tensor,
        gate_up_gradient: torch.Tensor,
    ) -> None:
        if sorted_experts.shape != (input_rows.shape[0],):
            raise ValueError("grouped expert indices do not cover the routed rows")
        boundaries = self._boundaries(offsets)
        if len(boundaries) != self.config.num_experts:
            raise ValueError("grouped offsets do not cover every expert")
        begin = 0 if self.config.expert_begin == 0 else boundaries[self.config.expert_begin - 1]
        for expert in range(self.config.expert_begin, self.config.expert_end):
            end = boundaries[expert]
            if end > begin:
                self.add_upstream(
                    expert,
                    input_rows[begin:end],
                    gate_up_gradient[begin:end],
                )
            begin = end

    @torch.no_grad()
    def add_grouped_down(
        self,
        sorted_experts: torch.Tensor,
        offsets: torch.Tensor,
        postactivation_rows: torch.Tensor,
        output_gradient: torch.Tensor,
    ) -> None:
        if sorted_experts.shape != (postactivation_rows.shape[0],):
            raise ValueError("grouped expert indices do not cover the routed rows")
        boundaries = self._boundaries(offsets)
        if len(boundaries) != self.config.num_experts:
            raise ValueError("grouped offsets do not cover every expert")
        begin = 0 if self.config.expert_begin == 0 else boundaries[self.config.expert_begin - 1]
        for expert in range(self.config.expert_begin, self.config.expert_end):
            end = boundaries[expert]
            if end > begin:
                self.add_down(
                    expert,
                    postactivation_rows[begin:end],
                    output_gradient[begin:end],
                )
            begin = end

    def allocated_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (
                self.upstream,
                self.down,
                self.upstream_rows,
                self.down_rows,
            )
        )


class DenseObjectiveGradientRouter:
    """Attach exact objective-gradient accumulation to a routed expert layer."""

    def __init__(self, accumulator: DenseObjectiveGradientAccumulator):
        self.accumulator = accumulator
        self.channel: str | None = None

    def select_channel(self, channel: str | None) -> None:
        self.channel = channel

    def expert_preactivation(
        self,
        expert: int,
        input_rows: torch.Tensor,
        gate_up_gradient: torch.Tensor,
    ) -> None:
        if self.channel is not None:
            self.accumulator.add_upstream(expert, input_rows, gate_up_gradient)

    def grouped_expert_preactivation(
        self,
        sorted_experts: torch.Tensor,
        offsets: torch.Tensor,
        input_rows: torch.Tensor,
        gate_up_gradient: torch.Tensor,
    ) -> None:
        if self.channel is not None:
            self.accumulator.add_grouped_upstream(
                sorted_experts,
                offsets,
                input_rows,
                gate_up_gradient,
            )

    def expert_output(
        self,
        expert: int,
        postactivation_rows: torch.Tensor,
        output_gradient: torch.Tensor,
    ) -> None:
        if self.channel is not None:
            self.accumulator.add_down(expert, postactivation_rows, output_gradient)

    def grouped_expert_output(
        self,
        sorted_experts: torch.Tensor,
        offsets: torch.Tensor,
        postactivation_rows: torch.Tensor,
        output_gradient: torch.Tensor,
    ) -> None:
        if self.channel is not None:
            self.accumulator.add_grouped_down(
                sorted_experts,
                offsets,
                postactivation_rows,
                output_gradient,
            )

    def install(self, adapter: Any, module: Any) -> None:
        block = getattr(module, "block_sparse_moe", None)
        grouped = bool(getattr(block, "_qsrt_grouped_expert_dispatch", False))
        if not adapter.enable_expert_preactivation_gradients(
            module,
            self.grouped_expert_preactivation if grouped else self.expert_preactivation,
        ):
            raise ValueError("decoder layer has no gate/up preactivation tap")
        if not adapter.enable_expert_output_gradients(
            module,
            self.grouped_expert_output if grouped else self.expert_output,
        ):
            raise ValueError("decoder layer has no expert W2 output tap")

    def uninstall(self, adapter: Any, module: Any) -> None:
        adapter.enable_expert_preactivation_gradients(module, None)
        adapter.enable_expert_output_gradients(module, None)
        self.channel = None


def _write_tensor_raw(
    path: Path,
    tensor: torch.Tensor,
    *,
    chunk_experts: int = 4,
) -> tuple[int, float | None]:
    if tensor.dtype not in (torch.float32, torch.int64) or tensor.ndim < 1:
        raise TypeError("raw gradient tensors must be FP32 or int64")
    total_bytes = tensor.numel() * tensor.element_size()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    sum_squares = 0.0 if tensor.dtype == torch.float32 else None
    try:
        os.posix_fallocate(fd, 0, total_bytes)
        offset = 0
        for begin in range(0, tensor.shape[0], chunk_experts):
            value = tensor[begin : begin + chunk_experts].detach().cpu().contiguous()
            if sum_squares is not None:
                sum_squares += float(value.square().sum(dtype=torch.float64).item())
            byte_view = memoryview(value.view(torch.uint8).numpy())
            written = 0
            while written < byte_view.nbytes:
                count = os.pwrite(fd, byte_view[written:], offset + written)
                if count <= 0:
                    raise OSError("raw gradient write made no progress")
                written += count
            offset += byte_view.nbytes
        if offset != total_bytes:
            raise RuntimeError("raw gradient write has the wrong byte count")
        os.fsync(fd)
    finally:
        os.close(fd)
    return total_bytes, sum_squares


class DenseObjectiveGradientArchive:
    """Fixed-layout FP32 gradient shards for direct-Viterbi refinement."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        path = self.root / (
            MANIFEST_FILENAME
            if (self.root / MANIFEST_FILENAME).is_file()
            else PARTIAL_MANIFEST_FILENAME
        )
        document = json.loads(path.read_text())
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("gradient archive schema is incompatible")
        self.manifest_path = path
        self.manifest: dict[str, object] = document
        self._lock = threading.Lock()

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        layers: Sequence[int],
        num_experts: int,
        shard_experts: int,
        hidden_dimension: int,
        intermediate_dimension: int,
        objective_normalizer: float,
        provenance: Mapping[str, object],
    ) -> "DenseObjectiveGradientArchive":
        destination = Path(root).expanduser().resolve()
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(destination)
        destination.mkdir(parents=True, exist_ok=True)
        if num_experts % shard_experts:
            raise ValueError("gradient shard width must divide the expert count")
        document: dict[str, object] = {
            "kind": "Kimi-K3 exact coupled expert objective gradients",
            "schema_version": SCHEMA_VERSION,
            "complete": False,
            "layers": [int(value) for value in layers],
            "num_experts": int(num_experts),
            "shard_experts": int(shard_experts),
            "hidden_dimension": int(hidden_dimension),
            "intermediate_dimension": int(intermediate_dimension),
            "dtype": "float32",
            "objective_normalizer": float(objective_normalizer),
            "shards": {},
            "provenance": dict(provenance),
        }
        _atomic_json(destination / PARTIAL_MANIFEST_FILENAME, document)
        return cls(destination)

    @staticmethod
    def shard_key(layer: int, expert_begin: int, expert_end: int) -> str:
        return f"layer-{layer:03d}/experts-{expert_begin:04d}-{expert_end:04d}"

    def write_shard(
        self,
        *,
        layer: int,
        accumulator: DenseObjectiveGradientAccumulator,
    ) -> Mapping[str, object]:
        config = accumulator.config
        key = self.shard_key(layer, config.expert_begin, config.expert_end)
        with self._lock:
            if key in dict(self.manifest.get("shards", {})):
                raise FileExistsError(f"gradient shard already exists: {key}")
        directory = self.root / key
        directory.mkdir(parents=True, exist_ok=False)
        files = {
            "upstream": ("upstream.f32", accumulator.upstream),
            "down": ("down.f32", accumulator.down),
            "upstream_rows": ("upstream-rows.i64", accumulator.upstream_rows),
            "down_rows": ("down-rows.i64", accumulator.down_rows),
        }
        record_files: dict[str, object] = {}
        sum_squares: dict[str, float] = {}
        for name, (filename, tensor) in files.items():
            path = directory / filename
            written_bytes, tensor_sum_squares = _write_tensor_raw(path, tensor)
            record_files[name] = {
                "file": str(path.relative_to(self.root)),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "bytes": written_bytes,
            }
            if tensor_sum_squares is not None:
                sum_squares[name] = tensor_sum_squares
        record: dict[str, object] = {
            "layer": int(layer),
            "expert_begin": config.expert_begin,
            "expert_end": config.expert_end,
            "files": record_files,
            "sum_squares": sum_squares,
        }
        _atomic_json(directory / "receipt.json", record)
        with self._lock:
            shards = dict(self.manifest.get("shards", {}))
            if key in shards:
                raise FileExistsError(f"gradient shard already exists: {key}")
            shards[key] = record
            self.manifest["shards"] = shards
            _atomic_json(self.root / PARTIAL_MANIFEST_FILENAME, self.manifest)
        return record

    def seal(self) -> Path:
        if self.manifest.get("complete") is True:
            if self.manifest_path.name != MANIFEST_FILENAME:
                raise ValueError("complete gradient archive has a partial manifest")
            return self.manifest_path
        layers = tuple(int(value) for value in self.manifest["layers"])
        num_experts = int(self.manifest["num_experts"])
        width = int(self.manifest["shard_experts"])
        expected = {
            self.shard_key(layer, begin, begin + width)
            for layer in layers
            for begin in range(0, num_experts, width)
        }
        actual = set(dict(self.manifest.get("shards", {})))
        if actual != expected:
            missing = sorted(expected - actual)
            raise ValueError(f"gradient archive is missing shards: {missing[:4]}")
        self.manifest["complete"] = True
        destination = self.root / MANIFEST_FILENAME
        _atomic_json(destination, self.manifest)
        (self.root / PARTIAL_MANIFEST_FILENAME).unlink()
        self.manifest_path = destination
        return destination


__all__ = [
    "DenseObjectiveGradientAccumulator",
    "DenseObjectiveGradientArchive",
    "DenseObjectiveGradientConfig",
    "DenseObjectiveGradientRouter",
]

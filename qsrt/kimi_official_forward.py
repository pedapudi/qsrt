"""Official Kimi-K3 operations for out-of-core boundary capture.

The immutable MXFP4 checkpoint supplies every tensor.  A separate local
snapshot may supply the matching remote Python model implementation when the
weight snapshot does not contain those files.  Decoder layers are constructed
on the meta device and materialized directly from their exclusive checkpoint
shards.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import importlib
import json
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from qsrt.instanttensor_kimi import (
    InstantTensorKimiLayerLoader,
    InstantTensorLoadConfig,
    KimiLayerLoadStats,
    load_checkpoint_tensor_cuda,
    release_layer,
)
from qsrt.kimi_boundary_slabs import DocumentIndex
from qsrt.kimi_forward_pipeline import PipelineActivation
from qsrt.kimi_stream import MODEL_TENSOR_PREFIX, assign_parameter


_META_CONSTRUCTION_LOCK = threading.Lock()
_FLA_AUTOTUNER_PATCH_LOCK = threading.Lock()


def _grouped_low_rank_delta(
    rows: torch.Tensor,
    factors: tuple[torch.Tensor, torch.Tensor],
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one expert-indexed ``B @ A.T`` correction."""

    a, b = factors
    latent = torch._grouped_mm(rows, a, offs=offsets)
    return torch._grouped_mm(latent, b.transpose(1, 2), offs=offsets)


def _grouped_expert_dispatch(
    block: Any,
    inputs: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
) -> torch.Tensor:
    """Evaluate all routed experts with three grouped BF16 matrix products."""

    flat_experts = topk_ids.reshape(-1)
    order = flat_experts.argsort()
    sorted_experts = flat_experts[order]
    sorted_inputs = inputs[order // topk_ids.shape[1]]
    banks = block._qsrt_expert_weight_banks
    counts = torch.bincount(
        flat_experts,
        minlength=int(banks["w1"].shape[0]),
    ).to(torch.int32)
    offsets = counts.cumsum(0, dtype=torch.int32)
    adapters = block._qsrt_grouped_low_rank_adapters
    gate = torch._grouped_mm(
        sorted_inputs,
        banks["w1"].transpose(1, 2),
        offs=offsets,
    )
    up = torch._grouped_mm(
        sorted_inputs,
        banks["w3"].transpose(1, 2),
        offs=offsets,
    )
    if adapters is not None:
        gate = gate + _grouped_low_rank_delta(
            sorted_inputs,
            adapters["w1"],
            offsets,
        )
        up = up + _grouped_low_rank_delta(
            sorted_inputs,
            adapters["w3"],
            offsets,
        )
    gate_up = torch.cat((gate, up), dim=-1)
    if gate_up.requires_grad:

        def preactivation_gradient(gradient: torch.Tensor) -> None:
            callback = block._qsrt_grouped_preactivation_gradient_callback
            if callback is not None:
                callback(
                    sorted_experts,
                    offsets,
                    sorted_inputs,
                    gradient,
                )

        gate_up.register_hook(preactivation_gradient)
    middle = block.experts[0].act_fn(gate_up)
    outputs = torch._grouped_mm(
        middle,
        banks["w2"].transpose(1, 2),
        offs=offsets,
    )
    if adapters is not None:
        outputs = outputs + _grouped_low_rank_delta(
            middle,
            adapters["w2"],
            offsets,
        )
    if outputs.requires_grad:

        def output_gradient(gradient: torch.Tensor) -> None:
            callback = block._qsrt_grouped_output_gradient_callback
            if callback is not None:
                callback(sorted_experts, offsets, middle, gradient)

        outputs.register_hook(output_gradient)

    scattered = torch.empty_like(outputs)
    scattered[order] = outputs
    final = (
        scattered.view(*topk_ids.shape, -1)
        .type(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(scattered.dtype)
    )
    if final.requires_grad:

        def routed_gradient(gradient: torch.Tensor) -> None:
            callback = block._qsrt_routed_output_gradient_callback
            if callback is not None:
                callback(gradient)

        final.register_hook(routed_gradient)
    return final


def _pack_grouped_expert_weights(module: Any) -> float:
    """Coalesce each expert projection into one grouped-matrix bank."""

    block = getattr(module, "block_sparse_moe", None)
    if block is None:
        return 0.0
    if hasattr(block, "_qsrt_expert_weight_banks"):
        return 0.0
    experts = tuple(block.experts)
    if not experts:
        raise ValueError("routed expert block has no experts")
    started = time.monotonic()
    banks: dict[str, torch.Tensor] = {}
    for matrix in ("w1", "w3", "w2"):
        bank = torch.stack(
            [getattr(expert, matrix).weight for expert in experts],
            dim=0,
        )
        for expert, view in zip(experts, bank.unbind(0), strict=True):
            assign_parameter(getattr(expert, matrix), "weight", view)
        banks[matrix] = bank
        gc.collect()
        torch.cuda.empty_cache()
    _install_grouped_expert_dispatch(module, banks)
    return time.monotonic() - started


def _allocate_grouped_expert_weights(
    module: Any,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Allocate the final grouped banks before streamed MXFP4 expansion."""

    block = getattr(module, "block_sparse_moe", None)
    if block is None:
        return {}
    experts = tuple(block.experts)
    if not experts:
        raise ValueError("routed expert block has no experts")
    num_experts = int(block.num_experts)
    if num_experts < len(experts):
        raise ValueError("routed expert prototype count exceeds configured experts")
    banks: dict[str, torch.Tensor] = {}
    for matrix in ("w1", "w3", "w2"):
        parameter = getattr(experts[0], matrix).weight
        if parameter.dtype != torch.bfloat16 or parameter.ndim != 2:
            raise TypeError(f"grouped {matrix} weights must be BF16 matrices")
        shape = tuple(int(value) for value in parameter.shape)
        if any(tuple(getattr(expert, matrix).weight.shape) != shape for expert in experts):
            raise ValueError(f"grouped {matrix} weight geometry changes by expert")
        banks[matrix] = torch.empty(
            (num_experts, *shape),
            dtype=torch.bfloat16,
            device=device,
        )
    return banks


def _install_grouped_expert_dispatch(
    module: Any,
    banks: dict[str, torch.Tensor],
) -> None:
    """Bind populated grouped weight banks to the differentiable dispatch."""

    block = getattr(module, "block_sparse_moe", None)
    if block is None:
        if banks:
            raise ValueError("non-routed layer received grouped expert banks")
        return
    if set(banks) != {"w1", "w2", "w3"}:
        raise ValueError("grouped expert banks do not cover all projections")
    block._qsrt_expert_weight_banks = banks
    block._qsrt_grouped_low_rank_adapters = None
    block._qsrt_grouped_preactivation_gradient_callback = None
    block._qsrt_grouped_output_gradient_callback = None
    block._qsrt_routed_output_gradient_callback = None
    block.moe_infer = types.MethodType(_grouped_expert_dispatch, block)
    block._qsrt_grouped_expert_dispatch = True


def install_grouped_low_rank_adapters(
    module: Any,
    adapters: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
) -> None:
    """Install frozen expert-indexed ``B @ A.T`` projection corrections."""

    block = getattr(module, "block_sparse_moe", None)
    if block is None or not getattr(block, "_qsrt_grouped_expert_dispatch", False):
        raise ValueError("low-rank adapters require grouped expert dispatch")
    if adapters is None:
        block._qsrt_grouped_low_rank_adapters = None
        return
    if set(adapters) != {"w1", "w2", "w3"}:
        raise ValueError("low-rank adapters do not cover all expert projections")
    banks = block._qsrt_expert_weight_banks
    validated: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for matrix, (a, b) in adapters.items():
        bank = banks[matrix]
        expected_a = (bank.shape[0], bank.shape[2])
        expected_b = (bank.shape[0], bank.shape[1])
        if a.ndim != 3 or a.shape[:2] != expected_a:
            raise ValueError(f"{matrix} adapter A has incompatible geometry")
        if b.ndim != 3 or b.shape[:2] != expected_b:
            raise ValueError(f"{matrix} adapter B has incompatible geometry")
        if a.shape[2] != b.shape[2] or a.shape[2] == 0:
            raise ValueError(f"{matrix} adapter rank is invalid")
        if a.dtype != bank.dtype or b.dtype != bank.dtype:
            raise TypeError(f"{matrix} adapter factors must match the weight dtype")
        if a.device != bank.device or b.device != bank.device:
            raise ValueError(f"{matrix} adapter factors must match the weight device")
        padded_rank = (int(a.shape[2]) + 7) // 8 * 8
        if padded_rank != a.shape[2]:
            padding = (0, padded_rank - int(a.shape[2]))
            a = F.pad(a, padding)
            b = F.pad(b, padding)
        validated[matrix] = (a.contiguous(), b.contiguous())
    block._qsrt_grouped_low_rank_adapters = validated


def _make_fla_autotuners_thread_safe() -> None:
    """Serialize concurrent calls to each mutable FLA autotuner instance."""

    from fla.ops.utils.cache import CachedAutotuner
    from triton.runtime.autotuner import Autotuner

    with _FLA_AUTOTUNER_PATCH_LOCK:
        for autotuner_class in (Autotuner, CachedAutotuner):
            if autotuner_class.__dict__.get("_qsrt_thread_safe_run", False):
                continue
            original_run = autotuner_class.run

            def locked_run(
                instance: Any,
                *args: Any,
                _original_run: Any = original_run,
                **kwargs: Any,
            ) -> Any:
                lock = getattr(instance, "_qsrt_run_lock", None)
                if lock is None:
                    with _FLA_AUTOTUNER_PATCH_LOCK:
                        lock = getattr(instance, "_qsrt_run_lock", None)
                        if lock is None:
                            lock = threading.RLock()
                            instance._qsrt_run_lock = lock
                with lock:
                    return _original_run(instance, *args, **kwargs)

            autotuner_class.run = locked_run
            autotuner_class._qsrt_thread_safe_run = True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _architecture_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        text_config.pop("quantization_config", None)
    config.pop("quantization_config", None)
    return config


@dataclass(frozen=True)
class OfficialKimiRuntime:
    """Loaded official configuration and model implementation."""

    config: Any
    text_config: Any
    linear_module: Any
    create_causal_mask: Any
    weight_checkpoint: Path
    code_checkpoint: Path


def load_official_kimi_runtime(
    *,
    weight_checkpoint: str | Path,
    code_checkpoint: str | Path,
) -> OfficialKimiRuntime:
    """Load matching official model code without materializing model weights."""

    weights = Path(weight_checkpoint).expanduser().resolve()
    code = Path(code_checkpoint).expanduser().resolve()
    for root in (weights, code):
        if not (root / "config.json").is_file():
            raise FileNotFoundError(root / "config.json")
    weight_config_path = weights / "config.json"
    code_config_path = code / "config.json"
    configs_match = _sha256(weight_config_path) == _sha256(code_config_path)
    if not configs_match and _architecture_config(
        weight_config_path
    ) != _architecture_config(code_config_path):
        raise ValueError(
            "weight and model-code snapshots have different architecture configs"
        )

    try:
        import transformers.utils.generic as generic
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from transformers.masking_utils import create_causal_mask
        from transformers.utils.output_capturing import OutputRecorder
    except ImportError as error:
        raise RuntimeError(
            "official Kimi execution requires the vLLM environment's Transformers "
            "and FLA dependencies"
        ) from error

    if not hasattr(generic, "OutputRecorder"):
        generic.OutputRecorder = OutputRecorder
    _make_fla_autotuners_thread_safe()
    config = AutoConfig.from_pretrained(str(code), trust_remote_code=True)
    text_config = config.text_config
    model_class = get_class_from_dynamic_module(
        "modeling_kimi_k3.KimiK3ForConditionalGeneration",
        str(code),
    )
    package = model_class.__module__.rsplit(".", 1)[0]
    linear_module = importlib.import_module(f"{package}.modeling_kimi_linear")
    text_config._attn_implementation = "eager"
    text_config.use_cache = False
    return OfficialKimiRuntime(
        config=config,
        text_config=text_config,
        linear_module=linear_module,
        create_causal_mask=create_causal_mask,
        weight_checkpoint=weights,
        code_checkpoint=code,
    )


def new_meta_decoder_layer(
    runtime: OfficialKimiRuntime,
    layer: int,
    *,
    grouped_expert_dispatch: bool = False,
) -> Any:
    """Construct one official decoder layer without allocating its weights."""

    # torch.set_default_dtype is process-global. Pipeline workers construct
    # layers concurrently, so protect the complete construction interval.
    with _META_CONSTRUCTION_LOCK:
        config = runtime.text_config
        configured_experts = int(config.num_experts or 0)
        if grouped_expert_dispatch and configured_experts:
            config = copy.copy(config)
            config.num_experts = 1
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("meta"):
                module = runtime.linear_module.KimiDecoderLayer(
                    config,
                    layer,
                )
                if grouped_expert_dispatch and configured_experts:
                    config.num_experts = configured_experts
                    block = getattr(module, "block_sparse_moe", None)
                    if block is not None:
                        block.gate = runtime.linear_module.KimiMoEGate(config)
                        block.num_experts = configured_experts
                        block.experts_per_rank = configured_experts
        finally:
            torch.set_default_dtype(previous_dtype)
    module.eval()
    return module


class OfficialKimiForwardAdapter:
    """Bind official decoder layers to the cyclic forward pipeline."""

    def __init__(
        self,
        runtime: OfficialKimiRuntime,
        *,
        load_config: InstantTensorLoadConfig | None = None,
        validate_outputs: bool = True,
        grouped_expert_dispatch: bool = False,
    ):
        self.runtime = runtime
        self.load_config = load_config or InstantTensorLoadConfig()
        self.validate_outputs = bool(validate_outputs)
        self.grouped_expert_dispatch = bool(grouped_expert_dispatch)
        self._loaders = {
            index: InstantTensorKimiLayerLoader(
                runtime.weight_checkpoint,
                device=torch.device("cuda", index),
                config=self.load_config,
            )
            for index in range(torch.cuda.device_count())
        }

    def load_layer(
        self,
        layer: int,
        device: torch.device,
    ) -> tuple[Any, KimiLayerLoadStats]:
        if device.index not in self._loaders:
            raise ValueError(f"CUDA device {device.index} was not initialized")
        torch.cuda.set_device(device)
        module = new_meta_decoder_layer(
            self.runtime,
            layer,
            grouped_expert_dispatch=self.grouped_expert_dispatch,
        )
        grouped_started = time.monotonic()
        grouped_banks = (
            _allocate_grouped_expert_weights(module, device)
            if self.grouped_expert_dispatch
            else None
        )
        grouped_prepare_seconds = time.monotonic() - grouped_started
        stats = self._loaders[device.index].load(
            module,
            layer=layer,
            expert_weight_banks=grouped_banks,
        )
        module.requires_grad_(False)
        if self.grouped_expert_dispatch:
            assert grouped_banks is not None
            _install_grouped_expert_dispatch(module, grouped_banks)
            stats.grouped_expert_pack_seconds += grouped_prepare_seconds
            stats.elapsed_seconds += grouped_prepare_seconds
            stats.peak_allocated_bytes = max(
                stats.peak_allocated_bytes,
                torch.cuda.max_memory_allocated(device),
            )
        return module, stats

    def forward_layer(
        self,
        module: Any,
        *,
        layer: int,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(module.layer_idx) != layer:
            raise ValueError("materialized decoder layer has the wrong index")
        sequence = int(hidden_states.shape[1])
        device = hidden_states.device
        cache_position = torch.arange(sequence, dtype=torch.long, device=device)
        positions = cache_position.unsqueeze(0)
        attention_mask = None
        if not module.is_linear_attn:
            attention_mask = self.runtime.create_causal_mask(
                config=self.runtime.text_config,
                inputs_embeds=hidden_states,
                attention_mask=None,
                past_key_values=None,
                position_ids=positions,
            )
        output = module(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=positions,
            past_key_values=None,
            use_cache=False,
            block_residual=block_residual,
            cache_position=cache_position,
        )
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError("Kimi decoder layer did not return its residual prefix")
        hidden, residual = output
        if self.validate_outputs and not bool(torch.all(torch.isfinite(hidden))):
            raise FloatingPointError(f"decoder layer {layer} produced non-finite values")
        return hidden, residual

    def release_layer(self, module: Any) -> None:
        release_layer(module)

    @staticmethod
    def enable_routed_output_gradients(
        module: Any,
        callback: Any,
    ) -> bool:
        """Make routed expert dispatch differentiable and tap its latent output.

        The official evaluation path wraps only ``moe_infer`` in
        ``torch.no_grad``. Decoder attention, routing, experts, and residual
        operations otherwise support input differentiation. The tapped tensor
        is the route-weighted expert sum before the routed RMSNorm and output
        projection; it is therefore the output coordinate of every expert W2.
        """

        block = getattr(module, "block_sparse_moe", None)
        if block is None:
            return False
        if getattr(block, "_qsrt_grouped_expert_dispatch", False):
            block._qsrt_routed_output_gradient_callback = callback
            return True
        if getattr(block, "_qsrt_differentiable_moe", False):
            block._qsrt_routed_output_gradient_callback = callback
            return True
        wrapped = block.moe_infer
        implementation = getattr(wrapped, "__wrapped__", None)
        if implementation is None:
            raise RuntimeError("official routed expert dispatch lacks its raw implementation")
        differentiable = implementation.__get__(block, type(block))

        def dispatch(
            instance: Any,
            inputs: torch.Tensor,
            topk_ids: torch.Tensor,
            topk_weight: torch.Tensor,
        ) -> torch.Tensor:
            output = differentiable(inputs, topk_ids, topk_weight)
            if output.requires_grad:
                def route_gradient(gradient: torch.Tensor) -> None:
                    tap = instance._qsrt_routed_output_gradient_callback
                    if tap is not None:
                        tap(gradient)

                output.register_hook(route_gradient)
            return output

        block._qsrt_routed_output_gradient_callback = callback
        block.moe_infer = types.MethodType(dispatch, block)
        block._qsrt_differentiable_moe = True
        return True

    @staticmethod
    def enable_expert_preactivation_gradients(
        module: Any,
        callback: Any,
    ) -> bool:
        """Tap routed expert inputs and gate/up preactivation cotangents.

        The callback receives ``(expert_id, input_rows, gate_up_gradient)``.
        Its lookup occurs during backward rather than forward, allowing one
        retained decoder graph to serve independent Fisher and deterministic
        objective-gradient VJPs.
        """

        block = getattr(module, "block_sparse_moe", None)
        if block is None:
            return False
        if getattr(block, "_qsrt_grouped_expert_dispatch", False):
            block._qsrt_grouped_preactivation_gradient_callback = callback
            return True
        if getattr(block, "_qsrt_expert_preactivation_hooks", None) is not None:
            block._qsrt_expert_preactivation_gradient_callback = callback
            return True

        handles = []
        for expert_id, expert in enumerate(block.experts):
            activation = getattr(expert, "act_fn", None)
            if activation is None:
                raise RuntimeError("routed expert lacks its gate/up activation module")

            def capture_input(
                instance: Any,
                inputs: tuple[torch.Tensor, ...],
            ) -> None:
                if len(inputs) != 1 or inputs[0].ndim != 2:
                    raise ValueError("routed expert input has incompatible geometry")
                instance._qsrt_expert_input_rows = inputs[0]

            def capture_preactivation(
                _activation: Any,
                inputs: tuple[torch.Tensor, ...],
                *,
                owner: Any = expert,
                routed_expert: int = expert_id,
            ) -> None:
                if len(inputs) != 1 or inputs[0].ndim != 2:
                    raise ValueError("gate/up preactivation has incompatible geometry")
                gate_up = inputs[0]
                input_rows = getattr(owner, "_qsrt_expert_input_rows", None)
                if input_rows is None or input_rows.shape[0] != gate_up.shape[0]:
                    raise RuntimeError("expert input and preactivation rows are not aligned")
                if gate_up.requires_grad:
                    def preactivation_gradient(gradient: torch.Tensor) -> None:
                        tap = block._qsrt_expert_preactivation_gradient_callback
                        if tap is not None:
                            tap(routed_expert, input_rows, gradient)

                    gate_up.register_hook(preactivation_gradient)

            handles.append(expert.register_forward_pre_hook(capture_input))
            handles.append(activation.register_forward_pre_hook(capture_preactivation))

        block._qsrt_expert_preactivation_gradient_callback = callback
        block._qsrt_expert_preactivation_hooks = tuple(handles)
        return True

    @staticmethod
    def enable_expert_output_gradients(
        module: Any,
        callback: Any,
    ) -> bool:
        """Tap post-SiTU inputs and route-weighted W2 output cotangents.

        The callback receives ``(expert_id, postactivation_rows,
        output_gradient)``. The gradient at an expert's W2 output already
        includes its applied route weight because the hook lies before the
        routed gather and weighted sum.
        """

        block = getattr(module, "block_sparse_moe", None)
        if block is None:
            return False
        if getattr(block, "_qsrt_grouped_expert_dispatch", False):
            block._qsrt_grouped_output_gradient_callback = callback
            return True
        if getattr(block, "_qsrt_expert_output_hooks", None) is not None:
            block._qsrt_expert_output_gradient_callback = callback
            return True

        handles = []
        for expert_id, expert in enumerate(block.experts):
            down = getattr(expert, "w2", None)
            if down is None:
                raise RuntimeError("routed expert lacks its down projection")

            def capture_output(
                _down: Any,
                inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                routed_expert: int = expert_id,
            ) -> None:
                if len(inputs) != 1 or inputs[0].ndim != 2 or output.ndim != 2:
                    raise ValueError("expert down projection has incompatible geometry")
                postactivation_rows = inputs[0]
                if postactivation_rows.shape[0] != output.shape[0]:
                    raise RuntimeError("expert down input and output rows are not aligned")
                if output.requires_grad:
                    def output_gradient(gradient: torch.Tensor) -> None:
                        tap = block._qsrt_expert_output_gradient_callback
                        if tap is not None:
                            tap(routed_expert, postactivation_rows, gradient)

                    output.register_hook(output_gradient)

            handles.append(down.register_forward_hook(capture_output))

        block._qsrt_expert_output_gradient_callback = callback
        block._qsrt_expert_output_hooks = tuple(handles)
        return True


@dataclass
class OfficialKimiEmbeddingInputs:
    """Generate document boundary-zero activations from the official embedding."""

    runtime: OfficialKimiRuntime
    documents: DocumentIndex
    device: torch.device
    load_config: InstantTensorLoadConfig
    load_seconds: float = 0.0

    def __iter__(self) -> Iterator[PipelineActivation]:
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("embedding generation requires an indexed CUDA device")
        torch.cuda.set_device(self.device)
        name = f"{MODEL_TENSOR_PREFIX}.embed_tokens.weight"
        started = time.monotonic()
        weight = load_checkpoint_tensor_cuda(
            self.runtime.weight_checkpoint,
            name,
            device=self.device,
            config=self.load_config,
        )
        self.load_seconds = time.monotonic() - started
        hidden_dimension = int(self.runtime.text_config.hidden_size)
        if weight.dtype != torch.bfloat16 or weight.shape[1] != hidden_dimension:
            raise ValueError("official token embedding has incompatible geometry")

        stream = torch.cuda.current_stream(self.device)
        try:
            for document in range(self.documents.document_count):
                first, end = self.documents.document_extent(document)
                token_ids = self.documents.input_ids[first:end].to(
                    device=self.device,
                    dtype=torch.long,
                    non_blocking=True,
                )
                hidden = F.embedding(token_ids, weight).unsqueeze(0)
                ready = torch.cuda.Event()
                ready.record(stream)
                yield PipelineActivation(
                    document=document,
                    first_token=first,
                    end_token=end,
                    hidden_states=hidden,
                    block_residual=torch.empty(
                        (end - first, 0, hidden_dimension),
                        dtype=torch.bfloat16,
                        device=self.device,
                    ),
                    ready=ready,
                )
        finally:
            stream.synchronize()
            del weight


__all__ = [
    "OfficialKimiEmbeddingInputs",
    "OfficialKimiForwardAdapter",
    "OfficialKimiRuntime",
    "install_grouped_low_rank_adapters",
    "load_official_kimi_runtime",
    "new_meta_decoder_layer",
]

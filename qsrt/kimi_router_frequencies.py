"""Forward-only Kimi router-frequency and score-margin measurement."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from qsrt.kimi_forward_pipeline import KimiForwardPipelineAdapter


@dataclass(frozen=True)
class RouterFrequencyLayer:
    """Exact selection counts and threshold-margin statistics for one layer."""

    layer: int
    tokens: int
    selections: int
    margin_mean: float
    margin_min: float
    margin_p01: float
    margin_p05: float
    margin_p50: float
    margin_p95: float
    margin_max: float
    margin_le_1e6: float
    margin_le_1e5: float
    margin_le_1e4: float
    margin_le_1e3: float


class RouterFrequencyCollector:
    """Own the completed per-layer frequency measurements."""

    def __init__(self, *, num_layers: int, num_experts: int, top_k: int):
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.counts = torch.zeros(
            (self.num_layers, self.num_experts), dtype=torch.int64
        )
        self.biases = torch.zeros(
            (self.num_layers, self.num_experts), dtype=torch.float32
        )
        self.active = torch.zeros(self.num_layers, dtype=torch.bool)
        self.layers: dict[int, RouterFrequencyLayer] = {}
        self._lock = threading.Lock()

    def record(
        self,
        *,
        layer: int,
        counts: torch.Tensor,
        bias: torch.Tensor,
        margins: torch.Tensor,
    ) -> None:
        counts = counts.to(device="cpu", dtype=torch.int64).contiguous()
        bias = bias.to(device="cpu", dtype=torch.float32).contiguous()
        margins = margins.to(device="cpu", dtype=torch.float32).contiguous()
        if counts.shape != (self.num_experts,) or bias.shape != (self.num_experts,):
            raise ValueError("router frequency tensors have incompatible expert geometry")
        if margins.ndim != 1 or not margins.numel():
            raise ValueError("router threshold margins must be a nonempty vector")
        quantiles = torch.quantile(
            margins,
            torch.tensor((0.01, 0.05, 0.5, 0.95), dtype=torch.float32),
        )
        tokens = int(margins.numel())
        record = RouterFrequencyLayer(
            layer=int(layer),
            tokens=tokens,
            selections=int(counts.sum()),
            margin_mean=float(margins.mean()),
            margin_min=float(margins.min()),
            margin_p01=float(quantiles[0]),
            margin_p05=float(quantiles[1]),
            margin_p50=float(quantiles[2]),
            margin_p95=float(quantiles[3]),
            margin_max=float(margins.max()),
            margin_le_1e6=float((margins <= 1e-6).to(torch.float32).mean()),
            margin_le_1e5=float((margins <= 1e-5).to(torch.float32).mean()),
            margin_le_1e4=float((margins <= 1e-4).to(torch.float32).mean()),
            margin_le_1e3=float((margins <= 1e-3).to(torch.float32).mean()),
        )
        if record.selections != tokens * self.top_k:
            raise ValueError("router selection counts do not close")
        with self._lock:
            if layer in self.layers:
                raise ValueError(f"router frequency layer {layer} was recorded twice")
            self.counts[layer].copy_(counts)
            self.biases[layer].copy_(bias)
            self.active[layer] = True
            self.layers[layer] = record

    def validate_complete(self, *, first_layer: int = 1) -> None:
        expected = set(range(first_layer, self.num_layers))
        observed = set(self.layers)
        if observed != expected:
            raise ValueError(
                "router frequency capture is incomplete; "
                f"missing={sorted(expected - observed)[:8]}"
            )

    def report(self) -> list[dict[str, object]]:
        return [asdict(self.layers[layer]) for layer in sorted(self.layers)]


class _LayerAccumulator:
    def __init__(self, *, num_experts: int, top_k: int):
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.counts: torch.Tensor | None = None
        self.margins: list[torch.Tensor] = []

    def append(self, indices: torch.Tensor, margins: torch.Tensor) -> None:
        if indices.ndim != 2 or indices.shape[1] != self.top_k:
            raise ValueError("router indices have incompatible top-k geometry")
        if margins.shape != indices.shape[:1]:
            raise ValueError("router margins do not match the token count")
        device = indices.device
        if self.counts is None:
            self.counts = torch.zeros(
                self.num_experts,
                dtype=torch.int64,
                device=device,
            )
        elif self.counts.device != device:
            raise ValueError("router accumulator received multiple devices")
        self.counts.add_(torch.bincount(
            indices.detach().reshape(-1).to(torch.int64),
            minlength=self.num_experts,
        ))
        self.margins.append(margins.detach().to(torch.float32).contiguous())


class FrequencyCapturingAdapter:
    """Measure router marginals without persisting per-token route matrices."""

    def __init__(
        self,
        adapter: KimiForwardPipelineAdapter,
        collector: RouterFrequencyCollector,
        *,
        bias_overrides: Mapping[int, torch.Tensor] | None = None,
    ):
        self.adapter = adapter
        self.collector = collector
        self.bias_overrides = {
            int(layer): value.detach().to(device="cpu", dtype=torch.float32)
            for layer, value in (bias_overrides or {}).items()
        }
        self._state: dict[int, tuple[Any, _LayerAccumulator, torch.Tensor]] = {}

    def load_layer(self, layer: int, device: torch.device) -> tuple[Any, object | None]:
        module, receipt = self.adapter.load_layer(layer, device)
        block = getattr(module, "block_sparse_moe", None)
        gate = None if block is None else getattr(block, "gate", None)
        if gate is None:
            return module, receipt
        if gate.moe_router_activation_func != "sigmoid":
            self.adapter.release_layer(module)
            raise ValueError("router frequency capture currently requires sigmoid routing")
        if int(gate.num_expert_group) != 1:
            self.adapter.release_layer(module)
            raise ValueError("router frequency capture requires ungrouped expert selection")
        override = self.bias_overrides.get(layer)
        if override is not None:
            if override.shape != gate.e_score_correction_bias.shape:
                self.adapter.release_layer(module)
                raise ValueError(f"layer {layer} bias override has the wrong shape")
            gate.e_score_correction_bias.data.copy_(
                override.to(device=device, dtype=gate.e_score_correction_bias.dtype)
            )
        bias = gate.e_score_correction_bias.detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        accumulator = _LayerAccumulator(
            num_experts=self.collector.num_experts,
            top_k=self.collector.top_k,
        )

        def capture(gate_module: Any, inputs: Any, output: Any) -> None:
            if not isinstance(output, tuple) or len(output) != 2:
                raise TypeError("Kimi gate did not return route indices and weights")
            if not isinstance(inputs, tuple) or len(inputs) != 1:
                raise TypeError("Kimi gate received unexpected inputs")
            hidden = inputs[0]
            flat = hidden.reshape(-1, hidden.shape[-1])
            scores = torch.sigmoid(
                F.linear(flat.to(torch.float32), gate_module.weight.to(torch.float32))
            )
            choice = scores + gate_module.e_score_correction_bias.unsqueeze(0)
            threshold = torch.topk(
                choice,
                k=self.collector.top_k + 1,
                dim=1,
                sorted=True,
            ).values
            margins = threshold[:, self.collector.top_k - 1] - threshold[:, self.collector.top_k]
            accumulator.append(output[0], margins)

        handle = gate.register_forward_hook(capture)
        self._state[id(module)] = (handle, accumulator, bias)
        return module, receipt

    def forward_layer(
        self,
        module: Any,
        *,
        layer: int,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.adapter.forward_layer(
            module,
            layer=layer,
            hidden_states=hidden_states,
            block_residual=block_residual,
        )

    def release_layer(self, module: Any) -> None:
        state = self._state.pop(id(module), None)
        if state is not None:
            handle, accumulator, bias = state
            handle.remove()
            if accumulator.counts is None or not accumulator.margins:
                raise ValueError(f"layer {module.layer_idx} produced no router samples")
            self.collector.record(
                layer=int(module.layer_idx),
                counts=accumulator.counts,
                bias=bias,
                margins=torch.cat(accumulator.margins),
            )
        self.adapter.release_layer(module)


__all__ = [
    "FrequencyCapturingAdapter",
    "RouterFrequencyCollector",
    "RouterFrequencyLayer",
]

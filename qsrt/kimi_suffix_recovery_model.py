"""Real Kimi-K3 modules for exact suffix recovery training.

The decoder wrappers retain the production checkpoint's frozen QSRT routed
experts while exposing an explicit continuous-parameter allowlist.  The
output modules reproduce Kimi-K3's residual-boundary mixer, final RMSNorm,
and frozen LM head from the stored suffix state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from qsrt.instanttensor_kimi import (
    InstantTensorLoadConfig,
    load_checkpoint_tensors_cuda,
)
from qsrt.kimi_official_fisher import SUFFIX_TENSORS
from qsrt.kimi_quantized_forward import QSRTKimiForwardAdapter
from qsrt.suffix_recovery_training import is_shared_expert_or_norm_tensor


def _stage_checkpoint_name(layer: int, relative_name: str) -> str:
    return f"language_model.model.layers.{layer}.{relative_name}"


def enable_shared_experts_and_norms(
    module: nn.Module,
    *,
    layer: int,
) -> tuple[str, ...]:
    """Freeze a decoder layer except for the first suffix-training arm."""

    selected: list[str] = []
    module.requires_grad_(False)
    for relative_name, parameter in module.named_parameters():
        checkpoint_name = _stage_checkpoint_name(layer, relative_name)
        enabled = is_shared_expert_or_norm_tensor(checkpoint_name)
        parameter.requires_grad_(enabled)
        if enabled:
            selected.append(checkpoint_name)
    if not selected:
        raise ValueError(f"layer {layer} exposes no shared-expert or norm tensors")
    return tuple(selected)


class KimiSuffixDecoderStage(nn.Module):
    """One resident Kimi-K3 decoder layer with frozen routed experts."""

    def __init__(
        self,
        *,
        adapter: QSRTKimiForwardAdapter,
        layer: int,
        decoder: nn.Module,
    ):
        super().__init__()
        self.adapter = adapter
        self.layer = int(layer)
        self.decoder = decoder
        self._route_capture_handle: object | None = None
        self._captured_routes: torch.Tensor | None = None

    def enable_route_capture(self) -> None:
        """Record one gate result for each subsequent stage invocation."""

        if self._route_capture_handle is not None:
            raise RuntimeError("suffix route capture is already enabled")
        gate = getattr(getattr(self.decoder, "block_sparse_moe", None), "gate", None)
        if gate is None:
            raise TypeError(f"decoder layer {self.layer} has no routed MoE gate")

        def capture(_module: nn.Module, _inputs: object, output: object) -> None:
            if not isinstance(output, tuple) or len(output) != 2:
                raise TypeError("Kimi gate did not return route indices and weights")
            if self._captured_routes is not None:
                raise RuntimeError("suffix route result was not consumed")
            self._captured_routes = output[0].detach().to(
                device="cpu",
                dtype=torch.int16,
            )

        self._route_capture_handle = gate.register_forward_hook(capture)

    def take_captured_routes(self) -> torch.Tensor:
        routes = self._captured_routes
        if routes is None:
            raise RuntimeError("suffix stage did not capture routed expert indices")
        self._captured_routes = None
        return routes

    def disable_route_capture(self) -> None:
        if self._captured_routes is not None:
            raise RuntimeError("suffix route result remained unconsumed")
        handle = self._route_capture_handle
        if handle is None:
            return
        handle.remove()
        self._route_capture_handle = None

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.ndim != 2:
            raise ValueError("suffix decoder hidden state must have shape [tokens, hidden]")
        output, output_residual = self.adapter.forward_layer(
            self.decoder,
            layer=self.layer,
            hidden_states=hidden.unsqueeze(0),
            block_residual=residual,
        )
        if output.ndim != 3 or output.shape[0] != 1:
            raise ValueError("Kimi decoder returned an incompatible hidden state")
        return output.squeeze(0), output_residual


def load_kimi_suffix_decoder_stage(
    adapter: QSRTKimiForwardAdapter,
    *,
    layer: int,
    device: torch.device,
) -> tuple[KimiSuffixDecoderStage, object, tuple[str, ...]]:
    """Load one QSRT decoder stage and expose only shared experts and norms."""

    decoder, stats = adapter.load_layer(layer, device)
    selected = enable_shared_experts_and_norms(decoder, layer=layer)
    stage = KimiSuffixDecoderStage(
        adapter=adapter,
        layer=layer,
        decoder=decoder,
    )
    stage.eval()
    return stage, stats, selected


class KimiSuffixStudentOutput(nn.Module):
    """Trainable output normalizers with frozen residual scores and LM head."""

    CHECKPOINT_PARAMETER_NAMES: Mapping[str, str] = {
        "final_norm": SUFFIX_TENSORS["final_norm"],
        "residual_norm": SUFFIX_TENSORS["residual_norm"],
    }

    def __init__(
        self,
        *,
        lm_head: torch.Tensor,
        final_norm: torch.Tensor,
        residual_norm: torch.Tensor,
        residual_projection: torch.Tensor,
        epsilon: float,
    ):
        super().__init__()
        if lm_head.ndim != 2:
            raise ValueError("Kimi LM head must be a matrix")
        hidden_dimension = int(lm_head.shape[1])
        expected = {
            "final_norm": (hidden_dimension,),
            "residual_norm": (hidden_dimension,),
            "residual_projection": (1, hidden_dimension),
        }
        values = {
            "final_norm": final_norm,
            "residual_norm": residual_norm,
            "residual_projection": residual_projection,
        }
        for name, shape in expected.items():
            value = values[name]
            if tuple(value.shape) != shape or not value.is_floating_point():
                raise ValueError(f"Kimi output tensor {name} has incompatible geometry")
        if epsilon <= 0.0:
            raise ValueError("Kimi RMSNorm epsilon must be positive")
        self.hidden_dimension = hidden_dimension
        self.epsilon = float(epsilon)
        self.final_norm = nn.Parameter(final_norm.detach().clone())
        self.residual_norm = nn.Parameter(residual_norm.detach().clone())
        self.register_buffer("residual_projection", residual_projection.detach().clone())
        self.register_buffer("lm_head", lm_head.detach().clone())

    def _normalize(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        work = value.float()
        normalized = work * torch.rsqrt(
            work.square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        return weight * normalized.to(value.dtype)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.ndim != 2 or hidden.shape[1] != self.hidden_dimension:
            raise ValueError("suffix output hidden state has incompatible geometry")
        if (
            residual.ndim != 3
            or residual.shape[0] != hidden.shape[0]
            or residual.shape[2] != self.hidden_dimension
        ):
            raise ValueError("suffix output residual state has incompatible geometry")
        values = torch.cat((residual, hidden.unsqueeze(1)), dim=1)
        work = values.float()
        reciprocal_rms = torch.rsqrt(
            work.square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        score_weight = (
            self.residual_norm.float()
            * self.residual_projection.squeeze(0).float()
        )
        scores = (work * reciprocal_rms * score_weight).sum(dim=-1)
        probabilities = torch.softmax(scores, dim=-1).unsqueeze(1)
        mixed = torch.matmul(probabilities, work).squeeze(1).to(hidden.dtype)
        final = self._normalize(mixed, self.final_norm)
        return F.linear(final, self.lm_head)


class KimiFrozenTeacherLMHead(nn.Module):
    """Frozen LM head applied to archived normalized teacher states."""

    def __init__(self, lm_head: torch.Tensor):
        super().__init__()
        if lm_head.ndim != 2 or not lm_head.is_floating_point():
            raise ValueError("Kimi teacher LM head must be a floating-point matrix")
        self.register_buffer("lm_head", lm_head.detach().clone())

    def forward(self, normalized: torch.Tensor) -> torch.Tensor:
        return F.linear(normalized, self.lm_head)


def load_kimi_suffix_output_modules(
    *,
    checkpoint: str | Path,
    student_device: torch.device,
    teacher_device: torch.device,
    epsilon: float,
    load_config: InstantTensorLoadConfig | None = None,
) -> tuple[KimiSuffixStudentOutput, KimiFrozenTeacherLMHead]:
    """Load the student output mixer and an independent frozen teacher head."""

    student_values = load_checkpoint_tensors_cuda(
        checkpoint,
        SUFFIX_TENSORS.values(),
        device=student_device,
        config=load_config,
    )
    teacher_values = load_checkpoint_tensors_cuda(
        checkpoint,
        (SUFFIX_TENSORS["lm_head"],),
        device=teacher_device,
        config=load_config,
    )
    student = KimiSuffixStudentOutput(
        lm_head=student_values[SUFFIX_TENSORS["lm_head"]],
        final_norm=student_values[SUFFIX_TENSORS["final_norm"]],
        residual_norm=student_values[SUFFIX_TENSORS["residual_norm"]],
        residual_projection=student_values[SUFFIX_TENSORS["residual_projection"]],
        epsilon=epsilon,
    ).to(student_device)
    teacher = KimiFrozenTeacherLMHead(
        teacher_values[SUFFIX_TENSORS["lm_head"]]
    ).to(teacher_device)
    student.eval()
    teacher.eval()
    return student, teacher


__all__ = [
    "KimiFrozenTeacherLMHead",
    "KimiSuffixDecoderStage",
    "KimiSuffixStudentOutput",
    "enable_shared_experts_and_norms",
    "load_kimi_suffix_decoder_stage",
    "load_kimi_suffix_output_modules",
]

"""Small, format-generic FlashInfer FP4 round-trip helpers.

This module deliberately knows nothing about checkpoint tensor names, model
dimensions, expert panels, or QSRT.  Callers provide one ordinary 2-D weight
tensor at a time and receive its decoded FP32 reconstruction plus exact payload
byte accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FP4RoundTrip:
    """Decoded weight and serialized payload sizes for one FP4 tensor."""

    reconstruction: torch.Tensor
    value_bytes: int
    block_scale_bytes: int
    tensor_scale_bytes: int

    @property
    def payload_bytes(self) -> int:
        return self.value_bytes + self.block_scale_bytes + self.tensor_scale_bytes

    def effective_bpw(self, weight_count: int) -> float:
        if weight_count <= 0:
            raise ValueError("weight_count must be positive")
        return 8.0 * self.payload_bytes / weight_count


def _flashinfer() -> Any:
    try:
        import flashinfer
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "FlashInfer is required for FP4 round trips; run this through the "
            "vLLM environment or install flashinfer-python"
        ) from exc
    return flashinfer


def _validate_weight(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    if not weight.is_cuda:
        raise ValueError("FlashInfer FP4 quantization requires a CUDA tensor")
    if weight.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"expected FP16 or BF16 weight, got {weight.dtype}")
    if weight.shape[1] % 32:
        raise ValueError("the innermost weight dimension must be divisible by 32")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("weight contains non-finite values")
    return weight.contiguous()


def mxfp4_roundtrip(weight: torch.Tensor) -> FP4RoundTrip:
    """Quantize and decode one tensor with FlashInfer's MXFP4 kernels."""

    flashinfer = _flashinfer()
    weight = _validate_weight(weight)
    packed, block_scales = flashinfer.mxfp4_quantize(weight)
    reconstruction = flashinfer.mxfp4_dequantize(packed, block_scales)
    return FP4RoundTrip(
        reconstruction=reconstruction,
        value_bytes=packed.numel() * packed.element_size(),
        block_scale_bytes=block_scales.numel() * block_scales.element_size(),
        tensor_scale_bytes=0,
    )


def nvfp4_roundtrip(weight: torch.Tensor) -> FP4RoundTrip:
    """Quantize and decode one tensor with FlashInfer's NVFP4 kernels.

    FlashInfer's quantizer takes the reciprocal global scale.  The serialized
    tensor scale used by reconstruction is its reciprocal, matching the NVFP4
    representation consumed by ``e2m1_and_ufp8sf_scale_to_float``.
    """

    flashinfer = _flashinfer()
    weight = _validate_weight(weight)
    amax = float(weight.float().abs().max().item())
    reciprocal_scale = torch.tensor(
        [(448.0 * 6.0) / amax if amax else 1.0],
        device=weight.device,
        dtype=torch.float32,
    )
    packed, block_scales = flashinfer.nvfp4_quantize(weight, reciprocal_scale)
    tensor_scale = reciprocal_scale.reciprocal()
    reconstruction = flashinfer.e2m1_and_ufp8sf_scale_to_float(
        packed.cpu().view(torch.uint8),
        block_scales.cpu().view(torch.uint8).reshape(-1),
        tensor_scale,
        16,
        1,
        True,
    )
    return FP4RoundTrip(
        reconstruction=reconstruction,
        value_bytes=packed.numel() * packed.element_size(),
        block_scale_bytes=block_scales.numel() * block_scales.element_size(),
        tensor_scale_bytes=tensor_scale.numel() * tensor_scale.element_size(),
    )

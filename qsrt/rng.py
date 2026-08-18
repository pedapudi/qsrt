"""Random-number helpers for deterministic transform construction."""

from __future__ import annotations

import torch


def seeded_normal(
    length: int,
    *,
    device: torch.device,
    seed: int | None,
) -> torch.Tensor:
    """Draw from a device-local stream without changing process RNG state."""

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    return torch.randn(length, device=device, generator=generator)

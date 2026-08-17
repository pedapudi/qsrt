#!/usr/bin/env python3
"""Exercise the experiment-only KLD hook against an installed vLLM runtime."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from safetensors import safe_open

from qsrt.glm52_engine_kld import (
    ENGINE_KLD_CHUNK_ROWS_ENV,
    ENGINE_KLD_REFERENCE_KEY_ENV,
    ENGINE_KLD_REFERENCE_PATH_ENV,
    install_vllm_engine_kld_patch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--key", default="logits")
    parser.add_argument("--chunk-rows", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with safe_open(args.reference, framework="pt", device="cpu") as handle:
        shape = tuple(handle.get_slice(args.key).get_shape())
    if len(shape) != 2:
        raise ValueError("reference logits must be rank two")

    os.environ[ENGINE_KLD_REFERENCE_PATH_ENV] = str(args.reference.resolve())
    os.environ[ENGINE_KLD_REFERENCE_KEY_ENV] = args.key
    os.environ[ENGINE_KLD_CHUNK_ROWS_ENV] = str(args.chunk_rows)
    install_vllm_engine_kld_patch()

    from vllm.v1.sample.sampler import Sampler

    device = torch.device(args.device)
    model_logprobs = torch.full(
        shape,
        -math.log(shape[1]),
        dtype=torch.float32,
        device=device,
    )
    token_ids = torch.zeros(shape[0], dtype=torch.int64, device=device)
    packed = Sampler.gather_logprobs(model_logprobs, 1, token_ids)
    values = -packed.logprobs[:, 0]
    if (
        tuple(packed.logprobs.shape) != (shape[0], 2)
        or not bool(torch.isfinite(values).all())
        or not bool(torch.all(values >= 0.0))
    ):
        raise RuntimeError("installed vLLM returned an invalid engine KLD result")
    print(
        json.dumps(
            {
                "status": "passed",
                "reference_shape": list(shape),
                "packed_shape": list(packed.logprobs.shape),
                "mean_forward_kld_against_uniform": float(values.mean().item()),
                "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

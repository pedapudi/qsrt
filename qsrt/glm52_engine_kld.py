"""Compute GLM-5.2 reference KLD inside vLLM without vocabulary sorting.

vLLM 0.17's prompt-logprob path computes a complete log-softmax tensor and
then calls ``torch.topk``.  Requesting every vocabulary entry makes that sort
allocate more than two additional GiB for one 2,048-token GLM-5.2 prompt.  The
experiment hook in this module retains the complete model log-probabilities on
the GPU, streams small reference-logit slices from a safetensors file, and
returns one negated forward-KLD value per prompt position through the existing
prompt-logprob result channel. Negation preserves the channel's nonpositive
numeric contract; the host runner restores the KLD sign before measurement.

This is an evaluation hook, not a serving feature.  It requires one complete
prefill chunk so the reference row alignment is explicit and rejects any
vocabulary or position mismatch instead of truncating either tensor.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open


ENGINE_KLD_REFERENCE_PATH_ENV = "QSRT_GLM52_ENGINE_KLD_REFERENCE_PATH"
ENGINE_KLD_REFERENCE_KEY_ENV = "QSRT_GLM52_ENGINE_KLD_REFERENCE_KEY"
ENGINE_KLD_CHUNK_ROWS_ENV = "QSRT_GLM52_ENGINE_KLD_CHUNK_ROWS"

_PATCH_LOCK = Lock()
_PATCHED = False


def _engine_kld_configuration() -> tuple[Path, str, int] | None:
    path_text = os.environ.get(ENGINE_KLD_REFERENCE_PATH_ENV)
    key = os.environ.get(ENGINE_KLD_REFERENCE_KEY_ENV)
    chunk_rows_text = os.environ.get(ENGINE_KLD_CHUNK_ROWS_ENV)
    if not path_text and not key and not chunk_rows_text:
        return None
    if not path_text or not key or not chunk_rows_text:
        raise RuntimeError(
            "engine KLD requires its reference path, tensor key, and chunk-row "
            "environment variables together"
        )
    try:
        chunk_rows = int(chunk_rows_text)
    except ValueError as error:
        raise RuntimeError("engine KLD chunk rows must be a positive integer") from error
    if chunk_rows < 1 or str(chunk_rows) != chunk_rows_text:
        raise RuntimeError("engine KLD chunk rows must be a positive integer")
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, key, chunk_rows


def forward_kld_from_model_logprobs(
    model_logprobs: torch.Tensor,
    *,
    reference_path: Path,
    reference_key: str,
    chunk_rows: int,
) -> torch.Tensor:
    """Return per-position ``KL(reference || model)`` on the model device."""

    if model_logprobs.ndim != 2 or not model_logprobs.is_floating_point():
        raise ValueError("model log-probabilities must be one floating rank-two tensor")
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise ValueError("engine KLD chunk rows must be a positive integer")
    result = torch.empty(
        model_logprobs.shape[0],
        dtype=torch.float32,
        device=model_logprobs.device,
    )
    with torch.inference_mode(), safe_open(
        reference_path, framework="pt", device="cpu"
    ) as handle:
        if reference_key not in handle.keys():
            raise KeyError(
                f"reference safetensors file has no tensor {reference_key!r}"
            )
        reference_slice = handle.get_slice(reference_key)
        reference_shape = tuple(reference_slice.get_shape())
        if reference_shape != tuple(model_logprobs.shape):
            raise ValueError(
                "reference and model log-probability shapes differ: "
                f"{reference_shape} versus {tuple(model_logprobs.shape)}"
            )
        for start in range(0, model_logprobs.shape[0], chunk_rows):
            stop = min(model_logprobs.shape[0], start + chunk_rows)
            reference = reference_slice[start:stop].to(
                device=model_logprobs.device,
                dtype=torch.float32,
            )
            log_reference = F.log_softmax(reference, dim=-1)
            values = F.kl_div(
                model_logprobs[start:stop],
                log_reference,
                reduction="none",
                log_target=True,
            ).sum(dim=-1)
            result[start:stop] = values
    if not bool(torch.isfinite(result).all()):
        raise ValueError("engine KLD produced a non-finite per-position value")
    return result


def _kld_logprobs_tensors(
    logprobs_tensors_type: Any,
    *,
    token_ids: torch.Tensor,
    kld_per_position: torch.Tensor,
    output_columns: int,
) -> Any:
    """Encode one KLD value per row in vLLM's existing result container."""

    if token_ids.ndim != 1 or token_ids.shape[0] != kld_per_position.shape[0]:
        raise ValueError("engine KLD token and position counts differ")
    if output_columns < 1:
        raise ValueError("engine KLD output must contain at least one column")
    indices = token_ids.to(torch.int32).unsqueeze(-1).expand(-1, output_columns)
    values = (-kld_per_position).unsqueeze(-1).expand(-1, output_columns)
    ranks = torch.zeros(
        token_ids.shape[0], dtype=torch.int64, device=token_ids.device
    )
    return logprobs_tensors_type(indices, values, ranks)


def engine_kld_from_prompt_logprobs(
    prompt_logprobs: Any,
    *,
    positions: int,
    target_token_ids: list[int],
) -> torch.Tensor:
    """Recover per-position KLD values carried by the evaluation hook."""

    if len(target_token_ids) != positions:
        raise ValueError("engine KLD target-token count mismatch")
    values = torch.empty(positions, dtype=torch.float64)
    if hasattr(prompt_logprobs, "start_indices"):
        for position, target_token_id in enumerate(target_token_ids):
            source_position = position + 1
            start = prompt_logprobs.start_indices[source_position]
            stop = prompt_logprobs.end_indices[source_position]
            matches = [
                -float(prompt_logprobs.logprobs[index])
                for index in range(start, stop)
                if int(prompt_logprobs.token_ids[index]) == target_token_id
            ]
            if not matches or any(value != matches[0] for value in matches[1:]):
                raise ValueError(
                    f"engine KLD row {position} has no unique target-token value"
                )
            values[position] = matches[0]
    else:
        for position, target_token_id in enumerate(target_token_ids):
            entry = prompt_logprobs[position + 1]
            if target_token_id not in entry:
                raise ValueError(
                    f"engine KLD row {position} has no target-token value"
                )
            values[position] = -float(entry[target_token_id].logprob)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("engine KLD result contains a non-finite value")
    return values


def install_vllm_engine_kld_patch() -> None:
    """Replace full-vocabulary top-k with KLD when the hook is configured."""

    global _PATCHED
    with _PATCH_LOCK:
        if _PATCHED:
            return
        configuration = _engine_kld_configuration()
        if configuration is None:
            return
        reference_path, reference_key, chunk_rows = configuration

        from vllm.v1.sample.sampler import (  # type: ignore[import-not-found]
            LogprobsTensors,
            Sampler,
        )

        original_gather_logprobs = Sampler.gather_logprobs

        def patched_gather_logprobs(
            model_logprobs: torch.Tensor,
            num_logprobs: int,
            token_ids: torch.Tensor,
        ) -> Any:
            if model_logprobs.shape[0] == 0:
                return original_gather_logprobs(
                    model_logprobs, num_logprobs, token_ids
                )
            kld = forward_kld_from_model_logprobs(
                model_logprobs,
                reference_path=reference_path,
                reference_key=reference_key,
                chunk_rows=chunk_rows,
            )
            return _kld_logprobs_tensors(
                LogprobsTensors,
                token_ids=token_ids,
                kld_per_position=kld,
                output_columns=num_logprobs + 1,
            )

        Sampler.gather_logprobs = staticmethod(patched_gather_logprobs)
        _PATCHED = True


__all__ = [
    "ENGINE_KLD_CHUNK_ROWS_ENV",
    "ENGINE_KLD_REFERENCE_KEY_ENV",
    "ENGINE_KLD_REFERENCE_PATH_ENV",
    "engine_kld_from_prompt_logprobs",
    "forward_kld_from_model_logprobs",
    "install_vllm_engine_kld_patch",
]

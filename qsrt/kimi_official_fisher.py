"""Final-output real-Fisher seeds for official Kimi-K3 replay.

The final decoder boundary is mixed with the eight 12-layer attention-residual
boundaries, normalized, and projected through the official LM head. Two
independent vocabulary samples provide an unbiased softmax-Fisher seed:

``(W[v1] - W[v2]) / sqrt(2)``.

Autograd pulls that seed into the decoder-chain boundary and every residual
boundary. No label loss, proxy output, or quantized resident model enters the
construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from qsrt.instanttensor_kimi import (
    InstantTensorLoadConfig,
    load_checkpoint_tensors_cuda,
)


SUFFIX_TENSORS: Mapping[str, str] = {
    "lm_head": "language_model.lm_head.weight",
    "final_norm": "language_model.model.norm.weight",
    "residual_norm": "language_model.model.output_attn_res_norm.weight",
    "residual_projection": "language_model.model.output_attn_res_proj.weight",
}


@dataclass(frozen=True)
class FisherSuffixResult:
    """One document's final-output cotangents and sampled vocabulary pairs."""

    chain_gradient: torch.Tensor
    residual_gradients: tuple[torch.Tensor, ...]
    first_tokens: torch.Tensor
    second_tokens: torch.Tensor


@dataclass(frozen=True)
class ObjectiveSuffixResult:
    """Deterministic teacher-to-anchor KL cotangents for one document."""

    chain_gradient: torch.Tensor
    residual_gradients: tuple[torch.Tensor, ...]
    kl_sum: float
    token_count: int


@dataclass(frozen=True)
class FisherObjectiveSuffixResult:
    """Fisher and KL VJPs evaluated against one retained suffix graph."""

    fisher: FisherSuffixResult
    objective: ObjectiveSuffixResult


class OfficialKimiFisherSuffix:
    """Official final normalization and LM-head Fisher VJP on one GPU."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        device: torch.device | str,
        hidden_dimension: int,
        vocabulary_size: int,
        residual_block_count: int,
        epsilon: float,
        logit_scale: float = 1.0,
        load_config: InstantTensorLoadConfig | None = None,
    ):
        target = torch.device(device)
        if target.type != "cuda" or target.index is None:
            raise ValueError("official Fisher suffix requires an indexed CUDA device")
        if hidden_dimension <= 0 or vocabulary_size <= 0 or residual_block_count <= 0:
            raise ValueError("official Fisher suffix geometry must be positive")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("RMSNorm epsilon must be finite and positive")
        if not math.isfinite(logit_scale) or logit_scale <= 0.0:
            raise ValueError("logit scale must be finite and positive")
        torch.cuda.set_device(target)
        loaded = load_checkpoint_tensors_cuda(
            checkpoint,
            SUFFIX_TENSORS.values(),
            device=target,
            config=load_config,
        )
        self.device = target
        self.hidden_dimension = int(hidden_dimension)
        self.vocabulary_size = int(vocabulary_size)
        self.residual_block_count = int(residual_block_count)
        self.epsilon = float(epsilon)
        self.logit_scale = float(logit_scale)
        self.lm_head = loaded[SUFFIX_TENSORS["lm_head"]]
        self.final_norm = loaded[SUFFIX_TENSORS["final_norm"]]
        self.residual_norm = loaded[SUFFIX_TENSORS["residual_norm"]]
        self.residual_projection = loaded[SUFFIX_TENSORS["residual_projection"]]
        expected = {
            "lm_head": (self.vocabulary_size, self.hidden_dimension),
            "final_norm": (self.hidden_dimension,),
            "residual_norm": (self.hidden_dimension,),
            "residual_projection": (1, self.hidden_dimension),
        }
        for role, shape in expected.items():
            value = getattr(self, role)
            if value.dtype != torch.bfloat16 or tuple(value.shape) != shape:
                raise ValueError(f"official suffix tensor {role} has incompatible geometry")

    def _attention_residual(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.cat((residual, hidden.reshape(-1, self.hidden_dimension).unsqueeze(1)), dim=1)
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
        return torch.matmul(probabilities, work).squeeze(1).to(hidden.dtype).unsqueeze(0)

    def _normalize(self, hidden: torch.Tensor) -> torch.Tensor:
        work = hidden.float()
        normalized = work * torch.rsqrt(
            work.square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        return self.final_norm * normalized.to(hidden.dtype)

    def _sample_seed(
        self,
        normalized: torch.Tensor,
        *,
        seed: int,
        chunk_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if chunk_tokens <= 0:
            raise ValueError("LM-head chunk size must be positive")
        flat = normalized.detach().reshape(-1, self.hidden_dimension)
        gradient = torch.empty_like(flat, dtype=torch.float32)
        first_parts: list[torch.Tensor] = []
        second_parts: list[torch.Tensor] = []
        # Draw once on CPU so vocabulary samples depend only on the document
        # seed and token index, not GPU assignment or LM-head batching.
        uniforms = torch.rand(
            (flat.shape[0], 2),
            dtype=torch.float64,
            generator=torch.Generator(device="cpu").manual_seed(int(seed)),
        )
        for first in range(0, flat.shape[0], chunk_tokens):
            end = min(first + chunk_tokens, int(flat.shape[0]))
            logits = F.linear(flat[first:end], self.lm_head).float()
            logits.mul_(self.logit_scale)
            probabilities = torch.softmax(logits, dim=-1)
            cumulative = torch.cumsum(probabilities, dim=-1)
            cumulative[:, -1] = 1.0
            sampled = torch.searchsorted(
                cumulative,
                uniforms[first:end].to(
                    device=self.device,
                    dtype=cumulative.dtype,
                    non_blocking=True,
                ),
                right=False,
            )
            del logits, probabilities, cumulative
            first_tokens, second_tokens = sampled.unbind(dim=1)
            first_weight = self.lm_head.index_select(0, first_tokens).float()
            second_weight = self.lm_head.index_select(0, second_tokens).float()
            gradient[first:end] = (
                first_weight - second_weight
            ) * (self.logit_scale / math.sqrt(2.0))
            first_parts.append(first_tokens)
            second_parts.append(second_tokens)
        return (
            gradient.reshape_as(normalized),
            torch.cat(first_parts),
            torch.cat(second_parts),
        )

    def _objective_seed(
        self,
        anchor_normalized: torch.Tensor,
        teacher_normalized: torch.Tensor,
        *,
        chunk_tokens: int,
    ) -> tuple[torch.Tensor, float]:
        """Return the exact BF16-LM-head gradient of summed teacher KL."""

        if chunk_tokens <= 0:
            raise ValueError("LM-head chunk size must be positive")
        if (
            teacher_normalized.device != self.device
            or teacher_normalized.dtype != torch.bfloat16
            or teacher_normalized.shape != anchor_normalized.shape
        ):
            raise ValueError(
                "teacher LM-head input must match the anchor suffix output"
            )
        anchor = anchor_normalized.detach().reshape(-1, self.hidden_dimension)
        teacher = teacher_normalized.detach().reshape(-1, self.hidden_dimension)
        gradient = torch.empty_like(anchor, dtype=torch.float32)
        kl_sum = 0.0
        for first in range(0, anchor.shape[0], chunk_tokens):
            end = min(first + chunk_tokens, int(anchor.shape[0]))
            anchor_logits = F.linear(anchor[first:end], self.lm_head).float()
            teacher_logits = F.linear(teacher[first:end], self.lm_head).float()
            anchor_logits.mul_(self.logit_scale)
            teacher_logits.mul_(self.logit_scale)
            anchor_log_probabilities = torch.log_softmax(anchor_logits, dim=-1)
            teacher_log_probabilities = torch.log_softmax(teacher_logits, dim=-1)
            teacher_probabilities = teacher_log_probabilities.exp()
            anchor_probabilities = anchor_log_probabilities.exp()
            kl_sum += float(
                torch.sum(
                    teacher_probabilities
                    * (teacher_log_probabilities - anchor_log_probabilities)
                )
            )
            logit_gradient = anchor_probabilities - teacher_probabilities
            gradient[first:end] = (
                logit_gradient.to(self.lm_head.dtype) @ self.lm_head
            ).float().mul_(self.logit_scale)
        return gradient.reshape_as(anchor_normalized), kl_sum

    def _validate_inputs(
        self,
        final_boundary: torch.Tensor,
        residual_inputs: Sequence[torch.Tensor],
    ) -> int:
        if final_boundary.device != self.device or final_boundary.dtype != torch.bfloat16:
            raise ValueError("final boundary must be BF16 on the suffix device")
        if final_boundary.ndim != 3 or final_boundary.shape[0] != 1:
            raise ValueError("final boundary must have shape [1, tokens, hidden]")
        tokens = int(final_boundary.shape[1])
        if final_boundary.shape[2] != self.hidden_dimension:
            raise ValueError("final boundary has the wrong hidden dimension")
        if len(residual_inputs) != self.residual_block_count:
            raise ValueError("suffix received the wrong residual-boundary count")
        for value in residual_inputs:
            if (
                value.device != self.device
                or value.dtype != torch.bfloat16
                or tuple(value.shape) != (tokens, self.hidden_dimension)
            ):
                raise ValueError("suffix residual input has incompatible geometry")
        return tokens

    def _forward_graph(
        self,
        final_boundary: torch.Tensor,
        residual_inputs: Sequence[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, ...],
        torch.Tensor,
    ]:
        self._validate_inputs(final_boundary, residual_inputs)
        hidden_leaf = final_boundary.detach().requires_grad_(True)
        residual_leaves = tuple(
            value.detach().requires_grad_(True) for value in residual_inputs
        )
        residual = torch.stack(residual_leaves, dim=1)
        mixed = self._attention_residual(hidden_leaf, residual)
        normalized = self._normalize(mixed)
        return hidden_leaf, residual_leaves, normalized

    @torch.no_grad()
    def normalized_hidden(
        self,
        *,
        final_boundary: torch.Tensor,
        residual_inputs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate the exact LM-head input for a teacher boundary archive."""

        self._validate_inputs(final_boundary, residual_inputs)
        residual = torch.stack(tuple(residual_inputs), dim=1)
        return self._normalize(
            self._attention_residual(final_boundary, residual)
        ).detach()

    def vjp(
        self,
        *,
        final_boundary: torch.Tensor,
        residual_inputs: Sequence[torch.Tensor],
        seed: int,
        lm_head_chunk_tokens: int = 128,
    ) -> FisherSuffixResult:
        """Sample the exact final distribution and pull it into decoder inputs."""

        hidden_leaf, residual_leaves, normalized = self._forward_graph(
            final_boundary,
            residual_inputs,
        )
        gradient, first_tokens, second_tokens = self._sample_seed(
            normalized,
            seed=seed,
            chunk_tokens=lm_head_chunk_tokens,
        )
        gradients = torch.autograd.grad(
            normalized,
            (hidden_leaf, *residual_leaves),
            grad_outputs=gradient,
            allow_unused=False,
        )
        return FisherSuffixResult(
            chain_gradient=gradients[0],
            residual_gradients=tuple(gradients[1:]),
            first_tokens=first_tokens,
            second_tokens=second_tokens,
        )

    def vjp_channels(
        self,
        *,
        final_boundary: torch.Tensor,
        residual_inputs: Sequence[torch.Tensor],
        teacher_normalized: torch.Tensor,
        seed: int,
        lm_head_chunk_tokens: int = 128,
    ) -> FisherObjectiveSuffixResult:
        """Pull Fisher and teacher-KL seeds through one suffix forward graph.

        ``final_boundary`` and ``residual_inputs`` belong to the anchor model.
        ``teacher_normalized`` is the teacher's exact BF16 input to the same LM
        head for the same token positions. The objective is the token-summed
        ``KL(teacher || anchor)``; normalization is deferred until the complete
        capture support is known.
        """

        tokens = self._validate_inputs(final_boundary, residual_inputs)
        hidden_leaf, residual_leaves, normalized = self._forward_graph(
            final_boundary,
            residual_inputs,
        )
        fisher_seed, first_tokens, second_tokens = self._sample_seed(
            normalized,
            seed=seed,
            chunk_tokens=lm_head_chunk_tokens,
        )
        objective_seed, kl_sum = self._objective_seed(
            normalized,
            teacher_normalized,
            chunk_tokens=lm_head_chunk_tokens,
        )
        leaves = (hidden_leaf, *residual_leaves)
        fisher_gradients = torch.autograd.grad(
            normalized,
            leaves,
            grad_outputs=fisher_seed,
            allow_unused=False,
            retain_graph=True,
        )
        objective_gradients = torch.autograd.grad(
            normalized,
            leaves,
            grad_outputs=objective_seed,
            allow_unused=False,
        )
        return FisherObjectiveSuffixResult(
            fisher=FisherSuffixResult(
                chain_gradient=fisher_gradients[0],
                residual_gradients=tuple(fisher_gradients[1:]),
                first_tokens=first_tokens,
                second_tokens=second_tokens,
            ),
            objective=ObjectiveSuffixResult(
                chain_gradient=objective_gradients[0],
                residual_gradients=tuple(objective_gradients[1:]),
                kl_sum=kl_sum,
                token_count=tokens,
            ),
        )


__all__ = [
    "FisherObjectiveSuffixResult",
    "FisherSuffixResult",
    "ObjectiveSuffixResult",
    "OfficialKimiFisherSuffix",
    "SUFFIX_TENSORS",
]

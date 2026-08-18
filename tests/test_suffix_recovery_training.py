from __future__ import annotations

import copy
from collections.abc import Iterable

import pytest
import torch
from torch import nn

from qsrt.suffix_recovery_training import (
    DenseDistributionKLLossHead,
    FP32AdamWConfig,
    FP32MasterAdamW,
    SuffixReplayTrainer,
    SuffixState,
    SuffixTrainingDocument,
    _module_device,
    combined_gradient_norm,
    is_shared_expert_or_norm_tensor,
)


def test_module_device_ignores_inactive_meta_parameters() -> None:
    module = nn.Module()
    module.live = nn.Parameter(torch.ones(1))
    module.placeholder = nn.Parameter(torch.empty(1, device="meta"), requires_grad=False)
    assert _module_device(module) == torch.device("cpu")


class _ToyRoutedSharedStage(nn.Module):
    """Small eval-only MoE stage with frozen routed and trainable shared paths."""

    def __init__(self, hidden: int, intermediate: int, experts: int, *, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)

        def value(*shape: int) -> torch.Tensor:
            return torch.randn(shape, generator=generator) * 0.12

        self.register_buffer("router_weight", value(experts, hidden))
        self.register_buffer("routed_gate", value(experts, intermediate, hidden))
        self.register_buffer("routed_up", value(experts, intermediate, hidden))
        self.register_buffer("routed_down", value(experts, hidden, intermediate))
        self.shared_gate = nn.Parameter(value(intermediate, hidden))
        self.shared_up = nn.Parameter(value(intermediate, hidden))
        self.shared_down = nn.Parameter(value(hidden, intermediate))
        self.input_norm = nn.Parameter(torch.ones(hidden))
        self.output_norm = nn.Parameter(torch.ones(hidden))
        self.top_k = 2

    @staticmethod
    def _rms_norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        work = value.float()
        normalized = work * torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + 1e-6)
        return normalized.to(value.dtype) * weight

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = torch.cat((residual, hidden.unsqueeze(1)), dim=1)
        normalized = self._rms_norm(hidden, self.input_norm)
        scores = torch.sigmoid(normalized @ self.router_weight.T)
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        routed_values = []
        for expert in range(self.router_weight.shape[0]):
            gate = torch.nn.functional.silu(normalized @ self.routed_gate[expert].T)
            up = normalized @ self.routed_up[expert].T
            routed_values.append((gate * up) @ self.routed_down[expert].T)
        routed_bank = torch.stack(routed_values, dim=1)
        selected = torch.gather(
            routed_bank,
            1,
            indices.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]),
        )
        routed = torch.sum(selected * weights.unsqueeze(-1), dim=1)
        shared_gate = torch.nn.functional.silu(normalized @ self.shared_gate.T)
        shared_up = normalized @ self.shared_up.T
        shared = (shared_gate * shared_up) @ self.shared_down.T
        mixed = hidden + routed * 0.2 + shared + residual.mean(dim=1) * 0.07
        return self._rms_norm(mixed, self.output_norm), residual


class _ToyDenseOutputHead(nn.Module):
    def __init__(self, hidden: int, vocabulary: int, *, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.residual_norm = nn.Parameter(torch.ones(hidden))
        self.final_norm = nn.Parameter(torch.ones(hidden))
        self.lm_head = nn.Parameter(
            torch.randn((vocabulary, hidden), generator=generator) * 0.16
        )
        self.register_buffer(
            "residual_projection",
            torch.randn(hidden, generator=generator) * 0.1,
        )

    def normalized_hidden(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.cat((residual, hidden.unsqueeze(1)), dim=1)
        work = values.float()
        reciprocal_rms = torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + 1e-6)
        scores = (
            work
            * reciprocal_rms
            * self.residual_norm.float()
            * self.residual_projection.float()
        ).sum(dim=-1)
        mixed = torch.sum(torch.softmax(scores, dim=1).unsqueeze(-1) * work, dim=1)
        normalized = mixed * torch.rsqrt(mixed.square().mean(dim=-1, keepdim=True) + 1e-6)
        return normalized.to(hidden.dtype) * self.final_norm

    def forward(self, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.normalized_hidden(hidden, residual) @ self.lm_head.T


class _ToyFrozenLMHead(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight.detach().clone())

    def forward(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized @ self.weight.T


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _state(seed: int, tokens: int, hidden: int) -> SuffixState:
    generator = torch.Generator().manual_seed(seed)
    return SuffixState(
        torch.randn((tokens, hidden), generator=generator) * 0.3,
        torch.randn((tokens, 2, hidden), generator=generator) * 0.2,
    )


@torch.no_grad()
def _teacher_target(
    stages: Iterable[nn.Module],
    boundary: SuffixState,
    output_head: _ToyDenseOutputHead,
) -> torch.Tensor:
    state = boundary
    for stage in stages:
        hidden, residual = stage(state.hidden, state.residual)
        state = SuffixState(hidden, residual)
    return output_head.normalized_hidden(state.hidden, state.residual).detach()


def _reference_gradients(
    stages: tuple[nn.Module, ...],
    head: nn.Module,
    teacher_head: nn.Module,
    documents: tuple[SuffixTrainingDocument, ...],
) -> tuple[float, tuple[dict[str, torch.Tensor], ...], dict[str, torch.Tensor]]:
    loss_sum = torch.zeros((), dtype=torch.float32)
    token_count = 0
    for document in documents:
        state = document.student_boundary
        for stage in stages:
            hidden, residual = stage(state.hidden, state.residual)
            state = SuffixState(hidden, residual)
        student_logits = head(state.hidden, state.residual).float()
        with torch.no_grad():
            teacher_logits = teacher_head(document.teacher_normalized).float()
            teacher_log_probabilities = torch.log_softmax(teacher_logits, dim=-1)
            teacher_probabilities = teacher_log_probabilities.exp()
        student_log_probabilities = torch.log_softmax(student_logits, dim=-1)
        loss_sum = loss_sum + torch.sum(
            teacher_probabilities
            * (teacher_log_probabilities - student_log_probabilities)
        )
        token_count += state.tokens
    loss = loss_sum / token_count
    stage_parameters = tuple(
        tuple(
            (name, parameter)
            for name, parameter in stage.named_parameters()
            if parameter.requires_grad
        )
        for stage in stages
    )
    head_parameters = tuple(
        (name, parameter)
        for name, parameter in head.named_parameters()
        if parameter.requires_grad
    )
    flat_parameters = tuple(
        parameter
        for values in (*stage_parameters, head_parameters)
        for _name, parameter in values
    )
    flat_gradients = torch.autograd.grad(loss, flat_parameters)
    cursor = 0
    stage_gradients = []
    for values in stage_parameters:
        stage_gradients.append(
            {
                name: flat_gradients[cursor + index].detach().float()
                for index, (name, _parameter) in enumerate(values)
            }
        )
        cursor += len(values)
    head_gradients = {
        name: flat_gradients[cursor + index].detach().float()
        for index, (name, _parameter) in enumerate(head_parameters)
    }
    return float(loss.detach()), tuple(stage_gradients), head_gradients


def _named_trainables(module: nn.Module) -> dict[str, nn.Parameter]:
    return {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def test_queue_chained_suffix_gradients_and_optimizer_match_monolithic_autograd() -> None:
    torch.manual_seed(0)
    hidden = 6
    pipeline_stages = tuple(
        _ToyRoutedSharedStage(hidden, 5, 8, seed=100 + index).eval()
        for index in range(3)
    )
    reference_stages = tuple(copy.deepcopy(stage) for stage in pipeline_stages)
    pipeline_head = _ToyDenseOutputHead(hidden, 11, seed=200).eval()
    reference_head = copy.deepcopy(pipeline_head)
    teacher_stages = tuple(
        _freeze(_ToyRoutedSharedStage(hidden, 5, 8, seed=300 + index))
        for index in range(3)
    )
    teacher_output_head = _freeze(_ToyDenseOutputHead(hidden, 11, seed=400))
    teacher_head = _freeze(_ToyFrozenLMHead(teacher_output_head.lm_head))

    boundaries = tuple(
        _state(500 + index, 2 + index % 4, hidden)
        for index in range(8)
    )
    documents = tuple(
        SuffixTrainingDocument(
            identifier=f"document-{index}",
            student_boundary=boundary,
            teacher_normalized=_teacher_target(
                teacher_stages,
                boundary,
                teacher_output_head,
            ),
        )
        for index, boundary in enumerate(boundaries)
    )
    reference_loss, reference_stage_gradients, reference_head_gradients = (
        _reference_gradients(
            reference_stages,
            reference_head,
            teacher_head,
            documents,
        )
    )

    trainer = SuffixReplayTrainer(
        stages=pipeline_stages,
        loss_head=DenseDistributionKLLossHead(
            student=pipeline_head,
            teacher=teacher_head,
            chunk_tokens=2,
            loss_dtype=torch.float32,
        ),
        queue_depth=1,
    )
    result = trainer.gradients(documents)
    assert result.mean_kl == pytest.approx(reference_loss, abs=2e-7)
    pipeline_stage_gradients = result.normalized_stage_gradients()
    pipeline_head_gradients = result.normalized_output_gradients()
    for actual, expected in zip(
        pipeline_stage_gradients,
        reference_stage_gradients,
        strict=True,
    ):
        assert actual.keys() == expected.keys()
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=2e-6, atol=2e-7)
    assert pipeline_head_gradients.keys() == reference_head_gradients.keys()
    for name in pipeline_head_gradients:
        torch.testing.assert_close(
            pipeline_head_gradients[name],
            reference_head_gradients[name],
            rtol=2e-6,
            atol=2e-7,
        )
    assert result.input_gradients.keys() == {
        f"document-{index}"
        for index in range(len(boundaries))
    }

    evaluation = SuffixReplayTrainer(
        stages=pipeline_stages,
        loss_head=DenseDistributionKLLossHead(
            student=pipeline_head,
            teacher=teacher_head,
            chunk_tokens=2,
            loss_dtype=torch.float32,
        ),
        queue_depth=1,
    ).evaluate(documents)
    assert evaluation.mean_kl == pytest.approx(reference_loss, abs=2e-7)

    microbatch_results = (
        trainer.gradients(documents[:3], retain_input_gradients=False),
        trainer.gradients(documents[3:], retain_input_gradients=False),
    )
    assert all(not value.input_gradients for value in microbatch_results)
    assert sum(value.token_count for value in microbatch_results) == result.token_count
    assert sum(value.kl_sum for value in microbatch_results) == pytest.approx(
        result.kl_sum,
        abs=2e-6,
    )
    for stage_index, expected in enumerate(result.stage_parameter_gradients):
        for name, value in expected.items():
            torch.testing.assert_close(
                sum(
                    microbatch.stage_parameter_gradients[stage_index][name]
                    for microbatch in microbatch_results
                ),
                value,
                rtol=2e-6,
                atol=2e-7,
            )
    for name, value in result.output_parameter_gradients.items():
        torch.testing.assert_close(
            sum(
                microbatch.output_parameter_gradients[name]
                for microbatch in microbatch_results
            ),
            value,
            rtol=2e-6,
            atol=2e-7,
        )

    config = FP32AdamWConfig(
        learning_rate=2e-3,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
    )
    pipeline_optimizers = [
        FP32MasterAdamW(_named_trainables(stage), config)
        for stage in pipeline_stages
    ]
    pipeline_optimizers.append(FP32MasterAdamW(_named_trainables(pipeline_head), config))
    for microbatch in microbatch_results:
        for optimizer, gradients in zip(
            pipeline_optimizers,
            (
                *microbatch.stage_parameter_gradients,
                microbatch.output_parameter_gradients,
            ),
            strict=True,
        ):
            optimizer.accumulate(gradients)
    gradient_scale = 1.0 / sum(
        microbatch.token_count for microbatch in microbatch_results
    )
    pipeline_norm = combined_gradient_norm(
        pipeline_optimizers,
        gradient_scale=gradient_scale,
    )
    reference_parameters = [
        parameter
        for module in (*reference_stages, reference_head)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    reference_optimizer = torch.optim.AdamW(
        reference_parameters,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    for parameter, gradient in zip(
        reference_parameters,
        (
            gradient
            for values in (*reference_stage_gradients, reference_head_gradients)
            for gradient in values.values()
        ),
        strict=True,
    ):
        parameter.grad = gradient.clone()
    reference_norm = float(
        torch.sqrt(
            sum(
                torch.sum(parameter.grad.double().square())
                for parameter in reference_parameters
                if parameter.grad is not None
            )
        )
    )
    assert pipeline_norm == pytest.approx(reference_norm, rel=2e-7)
    reference_optimizer.step()
    for optimizer in pipeline_optimizers:
        optimizer.step(gradient_scale=gradient_scale)

    for pipeline_module, reference_module in zip(
        (*pipeline_stages, pipeline_head),
        (*reference_stages, reference_head),
        strict=True,
    ):
        pipeline_parameters = _named_trainables(pipeline_module)
        reference_parameters_by_name = _named_trainables(reference_module)
        assert pipeline_parameters.keys() == reference_parameters_by_name.keys()
        for name in pipeline_parameters:
            torch.testing.assert_close(
                pipeline_parameters[name],
                reference_parameters_by_name[name],
                rtol=2e-6,
                atol=2e-7,
            )


def test_shared_expert_and_norm_allowlist_is_narrow() -> None:
    assert is_shared_expert_or_norm_tensor(
        "language_model.model.layers.84.block_sparse_moe.shared_experts.gate_proj.weight"
    )
    assert is_shared_expert_or_norm_tensor(
        "language_model.model.layers.84.input_layernorm.weight"
    )
    assert is_shared_expert_or_norm_tensor("language_model.model.norm.weight")
    assert not is_shared_expert_or_norm_tensor(
        "language_model.model.layers.84.block_sparse_moe.shared_experts.gate_proj.weight_scale"
    )
    assert not is_shared_expert_or_norm_tensor(
        "language_model.model.layers.84.block_sparse_moe.gate.weight"
    )
    assert not is_shared_expert_or_norm_tensor("language_model.lm_head.weight")


def test_suffix_replay_reports_worker_failure() -> None:
    class BrokenStage(_ToyRoutedSharedStage):
        def forward(
            self,
            hidden: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            raise ValueError("intentional stage failure")

    hidden = 6
    stage = BrokenStage(hidden, 5, 8, seed=100).eval()
    student_head = _ToyDenseOutputHead(hidden, 11, seed=200).eval()
    teacher_head = _freeze(_ToyFrozenLMHead(student_head.lm_head))
    boundary = _state(500, 5, hidden)
    document = SuffixTrainingDocument(
        identifier="document",
        student_boundary=boundary,
        teacher_normalized=boundary.hidden.detach(),
    )

    trainer = SuffixReplayTrainer(
        stages=(stage,),
        loss_head=DenseDistributionKLLossHead(
            student=student_head,
            teacher=teacher_head,
            chunk_tokens=2,
            loss_dtype=torch.float32,
        ),
        queue_depth=1,
    )
    with pytest.raises(
        RuntimeError,
        match="suffix forward stage 0 failed: ValueError: intentional stage failure",
    ):
        trainer.gradients((document,))


def test_suffix_training_document_excludes_final_causal_position() -> None:
    document = SuffixTrainingDocument(
        identifier="document",
        student_boundary=SuffixState(
            hidden=torch.arange(15, dtype=torch.float32).reshape(5, 3),
            residual=torch.arange(30, dtype=torch.float32).reshape(5, 2, 3),
        ),
        teacher_normalized=torch.arange(15, dtype=torch.float32).reshape(5, 3),
    )

    scored = document.causal_positions()

    assert scored.identifier == document.identifier
    assert scored.student_boundary.tokens == 4
    assert torch.equal(
        scored.student_boundary.hidden,
        document.student_boundary.hidden[:4],
    )
    assert torch.equal(scored.teacher_normalized, document.teacher_normalized[:4])

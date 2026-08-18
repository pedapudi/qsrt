"""Archive-independent execution primitives for suffix recovery training.

The routed-expert payload remains frozen.  Decoder stages receive stored
boundary states, retain one local autograd graph per in-flight document, and
exchange detached activations and explicit cotangents through bounded queues.
The output head evaluates dense teacher-to-student KL in vocabulary chunks so
the full token-by-vocabulary tensor is never retained.
"""

from __future__ import annotations

import math
import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class SuffixState:
    """One token-major decoder state at a residual-segment boundary."""

    hidden: torch.Tensor
    residual: torch.Tensor

    @property
    def tokens(self) -> int:
        return int(self.hidden.shape[0])

    def validate(self) -> None:
        if self.hidden.ndim != 2:
            raise ValueError("suffix hidden state must have shape [tokens, hidden]")
        if self.residual.ndim != 3:
            raise ValueError(
                "suffix residual state must have shape [tokens, prefixes, hidden]"
            )
        if (
            self.residual.shape[0] != self.hidden.shape[0]
            or self.residual.shape[2] != self.hidden.shape[1]
        ):
            raise ValueError("suffix hidden and residual states disagree")
        if not self.hidden.is_floating_point() or not self.residual.is_floating_point():
            raise ValueError("suffix states must be floating-point tensors")

    def to(self, device: torch.device) -> SuffixState:
        return SuffixState(
            hidden=self.hidden.to(device=device, non_blocking=True),
            residual=self.residual.to(device=device, non_blocking=True),
        )

    def detached_leaf(self, device: torch.device) -> SuffixState:
        value = self.to(device)
        return SuffixState(
            hidden=value.hidden.detach().requires_grad_(True),
            residual=value.residual.detach().requires_grad_(True),
        )

    def detached(self) -> SuffixState:
        return SuffixState(self.hidden.detach(), self.residual.detach())

    def token_slice(self, first: int, end: int) -> SuffixState:
        return SuffixState(self.hidden[first:end], self.residual[first:end])


@dataclass(frozen=True)
class SuffixTrainingDocument:
    """Stored student boundary input and frozen teacher output target."""

    identifier: str
    student_boundary: SuffixState
    teacher_normalized: torch.Tensor

    def validate(self) -> None:
        self.student_boundary.validate()
        if (
            self.teacher_normalized.ndim != 2
            or not self.teacher_normalized.is_floating_point()
        ):
            raise ValueError(
                "teacher normalized targets must have shape [tokens, hidden]"
            )
        if tuple(self.teacher_normalized.shape) != tuple(
            self.student_boundary.hidden.shape
        ):
            raise ValueError("student boundary and teacher target geometry disagree")

    def causal_positions(self) -> "SuffixTrainingDocument":
        """Exclude the final position, which has no in-document next token."""

        self.validate()
        if self.student_boundary.tokens < 2:
            raise ValueError("suffix training documents require at least two tokens")
        end = self.student_boundary.tokens - 1
        return SuffixTrainingDocument(
            identifier=self.identifier,
            student_boundary=self.student_boundary.token_slice(0, end),
            teacher_normalized=self.teacher_normalized[:end],
        )


@dataclass(frozen=True)
class DenseKLLossBackward:
    """One document's summed KL and output-head cotangents."""

    kl_sum: float
    token_count: int
    state_gradients: SuffixState
    parameter_gradients: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class SuffixTrainingEvaluation:
    """Summed dense KL for a complete set of replayed documents."""

    kl_sum: float
    token_count: int

    @property
    def mean_kl(self) -> float:
        if self.token_count <= 0:
            raise RuntimeError("suffix evaluation contains no tokens")
        return self.kl_sum / self.token_count


@dataclass(frozen=True)
class SuffixTrainingGradients:
    """Unnormalized gradients accumulated over complete documents."""

    kl_sum: float
    token_count: int
    stage_parameter_gradients: tuple[Mapping[str, torch.Tensor], ...]
    output_parameter_gradients: Mapping[str, torch.Tensor]
    input_gradients: Mapping[str, SuffixState]

    @property
    def mean_kl(self) -> float:
        if self.token_count <= 0:
            raise RuntimeError("suffix training result contains no tokens")
        return self.kl_sum / self.token_count

    def normalized_stage_gradients(self) -> tuple[dict[str, torch.Tensor], ...]:
        scale = 1.0 / self.token_count
        return tuple(
            {name: value * scale for name, value in gradients.items()}
            for gradients in self.stage_parameter_gradients
        )

    def normalized_output_gradients(self) -> dict[str, torch.Tensor]:
        scale = 1.0 / self.token_count
        return {
            name: value * scale
            for name, value in self.output_parameter_gradients.items()
        }


def _named_trainable_parameters(module: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    return tuple(
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    )


def _module_device(module: nn.Module) -> torch.device:
    values = tuple(
        value
        for value in (*module.parameters(), *module.buffers())
        if value.device.type != "meta"
    )
    if not values:
        return torch.device("cpu")
    devices = {value.device for value in values}
    if len(devices) != 1:
        raise ValueError("one suffix stage cannot span multiple devices")
    return next(iter(devices))


def _accumulate(
    destination: dict[str, torch.Tensor],
    names: Sequence[str],
    gradients: Sequence[torch.Tensor | None],
) -> None:
    for name, gradient in zip(names, gradients, strict=True):
        if gradient is None:
            continue
        value = gradient.detach().float()
        previous = destination.get(name)
        if previous is None:
            destination[name] = value.clone()
        else:
            previous.add_(value)


def _require_parameter_gradients(
    owner: str,
    names: Sequence[str],
    gradients: Sequence[torch.Tensor | None],
) -> None:
    missing = [name for name, gradient in zip(names, gradients, strict=True) if gradient is None]
    if missing:
        raise RuntimeError(
            f"{owner} dropped gradients for trainable parameters {missing}"
        )


class DenseDistributionKLLossHead:
    """Chunked dense forward-KL evaluator with explicit output cotangents."""

    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module,
        chunk_tokens: int,
        logit_scale: float = 1.0,
        loss_dtype: torch.dtype = torch.float32,
    ):
        if chunk_tokens <= 0:
            raise ValueError("dense KL chunk size must be positive")
        if not math.isfinite(logit_scale) or logit_scale <= 0.0:
            raise ValueError("logit scale must be finite and positive")
        if loss_dtype not in {torch.float32, torch.float64}:
            raise ValueError("dense KL accumulation must use FP32 or FP64")
        if student.training or teacher.training:
            raise ValueError("suffix output modules must remain in eval mode")
        if any(parameter.requires_grad for parameter in teacher.parameters()):
            raise ValueError("teacher output parameters must remain frozen")
        self.student = student
        self.teacher = teacher
        self.chunk_tokens = int(chunk_tokens)
        self.logit_scale = float(logit_scale)
        self.loss_dtype = loss_dtype
        self.student_device = _module_device(student)
        self.teacher_device = _module_device(teacher)
        self.parameters = _named_trainable_parameters(student)

    @torch.no_grad()
    def evaluate_document(
        self,
        student_state: SuffixState,
        teacher_normalized: torch.Tensor,
    ) -> SuffixTrainingEvaluation:
        student_state.validate()
        if (
            teacher_normalized.ndim != 2
            or not teacher_normalized.is_floating_point()
            or tuple(teacher_normalized.shape) != tuple(student_state.hidden.shape)
        ):
            raise ValueError("teacher normalized target has incompatible geometry")
        student_state = student_state.to(self.student_device)
        teacher_normalized = teacher_normalized.to(
            device=self.teacher_device,
            non_blocking=True,
        )
        kl_sum = 0.0
        for first in range(0, student_state.tokens, self.chunk_tokens):
            end = min(first + self.chunk_tokens, student_state.tokens)
            student_chunk = student_state.token_slice(first, end)
            teacher_logits = self.teacher(
                teacher_normalized[first:end]
            ).to(dtype=self.loss_dtype)
            teacher_logits = teacher_logits * self.logit_scale
            teacher_log_probabilities = torch.log_softmax(
                teacher_logits,
                dim=-1,
            ).to(device=self.student_device)
            teacher_probabilities = teacher_log_probabilities.exp()
            student_logits = self.student(
                student_chunk.hidden,
                student_chunk.residual,
            ).to(dtype=self.loss_dtype)
            student_logits = student_logits * self.logit_scale
            student_log_probabilities = torch.log_softmax(student_logits, dim=-1)
            loss = torch.sum(
                teacher_probabilities
                * (teacher_log_probabilities - student_log_probabilities)
            )
            kl_sum += float(loss)
        return SuffixTrainingEvaluation(kl_sum, student_state.tokens)

    def backward_document(
        self,
        student_state: SuffixState,
        teacher_normalized: torch.Tensor,
    ) -> DenseKLLossBackward:
        student_state.validate()
        if (
            teacher_normalized.ndim != 2
            or not teacher_normalized.is_floating_point()
            or tuple(teacher_normalized.shape) != tuple(student_state.hidden.shape)
        ):
            raise ValueError("teacher normalized target has incompatible geometry")
        student_state = student_state.detached_leaf(self.student_device)
        teacher_normalized = teacher_normalized.to(
            device=self.teacher_device,
            non_blocking=True,
        ).detach()
        hidden_gradient = torch.empty_like(student_state.hidden)
        residual_gradient = torch.empty_like(student_state.residual)
        parameter_sums: dict[str, torch.Tensor] = {}
        names = tuple(name for name, _parameter in self.parameters)
        parameters = tuple(parameter for _name, parameter in self.parameters)
        kl_sum = 0.0

        for first in range(0, student_state.tokens, self.chunk_tokens):
            end = min(first + self.chunk_tokens, student_state.tokens)
            student_chunk = student_state.token_slice(first, end)
            with torch.no_grad():
                teacher_logits = self.teacher(
                    teacher_normalized[first:end]
                ).to(dtype=self.loss_dtype)
                teacher_logits = teacher_logits * self.logit_scale
                teacher_log_probabilities = torch.log_softmax(
                    teacher_logits,
                    dim=-1,
                ).to(device=self.student_device)
                teacher_probabilities = teacher_log_probabilities.exp()
            student_logits = self.student(
                student_chunk.hidden,
                student_chunk.residual,
            ).to(dtype=self.loss_dtype)
            student_logits = student_logits * self.logit_scale
            student_log_probabilities = torch.log_softmax(student_logits, dim=-1)
            loss = torch.sum(
                teacher_probabilities
                * (teacher_log_probabilities - student_log_probabilities)
            )
            values = torch.autograd.grad(
                loss,
                (student_chunk.hidden, student_chunk.residual, *parameters),
                allow_unused=True,
            )
            hidden_value, residual_value = values[:2]
            if hidden_value is None or residual_value is None:
                raise RuntimeError("output head dropped a suffix-state gradient")
            _require_parameter_gradients("output head", names, values[2:])
            hidden_gradient[first:end].copy_(hidden_value)
            residual_gradient[first:end].copy_(residual_value)
            _accumulate(parameter_sums, names, values[2:])
            kl_sum += float(loss.detach())

        return DenseKLLossBackward(
            kl_sum=kl_sum,
            token_count=student_state.tokens,
            state_gradients=SuffixState(hidden_gradient, residual_gradient),
            parameter_gradients=parameter_sums,
        )


@dataclass
class _StageTape:
    input_state: SuffixState
    output_state: SuffixState


@dataclass(frozen=True)
class _ForwardDocument:
    index: int
    document: SuffixTrainingDocument
    state: SuffixState


@dataclass(frozen=True)
class _ReverseDocument:
    index: int
    document: SuffixTrainingDocument
    gradients: SuffixState


@dataclass(frozen=True)
class _WorkerFailure:
    stage: int
    operation: str
    message: str
    traceback: str


_STOP = object()


class SuffixReplayTrainer:
    """Bounded document pipeline for exact suffix parameter gradients."""

    def __init__(
        self,
        *,
        stages: Sequence[nn.Module],
        loss_head: DenseDistributionKLLossHead,
        queue_depth: int = 1,
        checkpoint_stages: bool = True,
    ):
        if not stages:
            raise ValueError("suffix replay requires at least one decoder stage")
        if queue_depth <= 0:
            raise ValueError("suffix replay queue depth must be positive")
        if any(stage.training for stage in stages):
            raise ValueError("suffix decoder stages must remain in eval mode")
        identities: set[int] = set()
        for module in (*stages, loss_head.student):
            for parameter in module.parameters():
                identity = id(parameter)
                if identity in identities:
                    raise ValueError("suffix trainable parameters cannot span stage owners")
                identities.add(identity)
        self.stages = tuple(stages)
        self.loss_head = loss_head
        self.queue_depth = int(queue_depth)
        self.checkpoint_stages = bool(checkpoint_stages)
        self.stage_devices = tuple(_module_device(stage) for stage in self.stages)
        self.stage_parameters = tuple(
            _named_trainable_parameters(stage) for stage in self.stages
        )

    @staticmethod
    def _stage_forward(module: nn.Module, state: SuffixState) -> SuffixState:
        output = module(state.hidden, state.residual)
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError("suffix stage must return hidden and residual tensors")
        result = SuffixState(output[0], output[1])
        result.validate()
        if result.tokens != state.tokens:
            raise ValueError("suffix stage changed the document token count")
        return result

    def gradients(
        self,
        documents: Sequence[SuffixTrainingDocument],
        *,
        retain_input_gradients: bool = True,
    ) -> SuffixTrainingGradients:
        values = tuple(documents)
        if not values:
            raise ValueError("suffix replay requires at least one document")
        identifiers = [value.identifier for value in values]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("suffix replay document identifiers must be unique")
        for value in values:
            value.validate()

        stage_count = len(self.stages)
        forward_queues = [queue.Queue(maxsize=self.queue_depth) for _ in self.stages]
        reverse_queues = [queue.Queue(maxsize=self.queue_depth) for _ in self.stages]
        output_queue: queue.Queue[object] = queue.Queue(maxsize=self.queue_depth)
        completion_queue: queue.Queue[object] = queue.Queue()
        tapes: list[dict[int, _StageTape]] = [dict() for _ in self.stages]
        stage_sums: list[dict[str, torch.Tensor]] = [dict() for _ in self.stages]
        head_sums: dict[str, torch.Tensor] = {}
        input_gradients: dict[str, SuffixState] = {}
        failures: queue.Queue[_WorkerFailure] = queue.Queue()
        abort = threading.Event()

        def publish(target: queue.Queue[object], item: object) -> None:
            while not abort.is_set():
                try:
                    target.put(item, timeout=0.1)
                    return
                except queue.Full:
                    continue
            raise RuntimeError("suffix replay stopped while publishing work")

        def receive(source: queue.Queue[object]) -> object:
            while not abort.is_set():
                try:
                    return source.get(timeout=0.1)
                except queue.Empty:
                    continue
            raise RuntimeError("suffix replay stopped while awaiting work")

        def fail(stage: int, operation: str, error: BaseException) -> None:
            failures.put(
                _WorkerFailure(
                    stage=stage,
                    operation=operation,
                    message=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                )
            )
            abort.set()

        def forward_worker(stage: int) -> None:
            source = forward_queues[stage]
            target = output_queue if stage + 1 == stage_count else forward_queues[stage + 1]
            try:
                while True:
                    item = receive(source)
                    if item is _STOP:
                        publish(target, _STOP)
                        return
                    assert isinstance(item, _ForwardDocument)
                    leaf = item.state.detached_leaf(self.stage_devices[stage])
                    with torch.enable_grad():
                        if self.checkpoint_stages:
                            hidden, residual = checkpoint(
                                self.stages[stage],
                                leaf.hidden,
                                leaf.residual,
                                use_reentrant=False,
                            )
                            output = SuffixState(hidden, residual)
                            output.validate()
                            if output.tokens != leaf.tokens:
                                raise ValueError(
                                    "suffix stage changed the document token count"
                                )
                        else:
                            output = self._stage_forward(self.stages[stage], leaf)
                    if item.index in tapes[stage]:
                        raise RuntimeError("suffix stage received a duplicate document")
                    tapes[stage][item.index] = _StageTape(leaf, output)
                    publish(
                        target,
                        _ForwardDocument(
                            index=item.index,
                            document=item.document,
                            state=output.detached(),
                        ),
                    )
            except BaseException as error:
                fail(stage, "forward", error)

        def reverse_worker(stage: int) -> None:
            source = reverse_queues[stage]
            target = completion_queue if stage == 0 else reverse_queues[stage - 1]
            names = tuple(name for name, _parameter in self.stage_parameters[stage])
            parameters = tuple(parameter for _name, parameter in self.stage_parameters[stage])
            try:
                while True:
                    item = receive(source)
                    if item is _STOP:
                        publish(target, _STOP)
                        return
                    assert isinstance(item, _ReverseDocument)
                    try:
                        tape = tapes[stage].pop(item.index)
                    except KeyError as error:
                        raise RuntimeError(
                            "suffix reverse stage has no matching forward graph"
                        ) from error
                    gradients = item.gradients.to(self.stage_devices[stage])
                    values = torch.autograd.grad(
                        (tape.output_state.hidden, tape.output_state.residual),
                        (
                            tape.input_state.hidden,
                            tape.input_state.residual,
                            *parameters,
                        ),
                        grad_outputs=(gradients.hidden, gradients.residual),
                        allow_unused=True,
                    )
                    hidden_value, residual_value = values[:2]
                    if hidden_value is None or residual_value is None:
                        raise RuntimeError("suffix stage dropped a boundary gradient")
                    _require_parameter_gradients(
                        f"suffix stage {stage}",
                        names,
                        values[2:],
                    )
                    _accumulate(stage_sums[stage], names, values[2:])
                    publish(
                        target,
                        _ReverseDocument(
                            index=item.index,
                            document=item.document,
                            gradients=SuffixState(hidden_value, residual_value).detached(),
                        ),
                    )
            except BaseException as error:
                fail(stage, "reverse", error)

        def feed() -> None:
            try:
                for index, document in enumerate(values):
                    publish(
                        forward_queues[0],
                        _ForwardDocument(index, document, document.student_boundary),
                    )
                publish(forward_queues[0], _STOP)
            except BaseException as error:
                fail(-1, "feed", error)

        workers = [
            threading.Thread(
                target=forward_worker,
                args=(stage,),
                name=f"suffix-forward-{stage:03d}",
                daemon=True,
            )
            for stage in range(stage_count)
        ] + [
            threading.Thread(
                target=reverse_worker,
                args=(stage,),
                name=f"suffix-reverse-{stage:03d}",
                daemon=True,
            )
            for stage in range(stage_count)
        ]
        feeder = threading.Thread(target=feed, name="suffix-feed", daemon=True)
        kl_sum = 0.0
        token_count = 0
        pipeline_error: BaseException | None = None
        try:
            for worker in workers:
                worker.start()
            feeder.start()
            completed_forward = 0
            completed_reverse = 0
            while completed_forward < len(values):
                item = receive(output_queue)
                if item is _STOP:
                    raise RuntimeError("suffix forward pipeline stopped before all documents")
                assert isinstance(item, _ForwardDocument)
                loss = self.loss_head.backward_document(
                    item.state,
                    item.document.teacher_normalized,
                )
                kl_sum += loss.kl_sum
                token_count += loss.token_count
                _accumulate(
                    head_sums,
                    tuple(loss.parameter_gradients),
                    tuple(loss.parameter_gradients.values()),
                )
                publish(
                    reverse_queues[-1],
                    _ReverseDocument(
                        index=item.index,
                        document=item.document,
                        gradients=loss.state_gradients,
                    ),
                )
                completed_forward += 1
                while True:
                    try:
                        completed = completion_queue.get_nowait()
                    except queue.Empty:
                        break
                    if completed is _STOP:
                        raise RuntimeError(
                            "suffix reverse pipeline stopped before all documents"
                        )
                    assert isinstance(completed, _ReverseDocument)
                    if retain_input_gradients:
                        input_gradients[completed.document.identifier] = (
                            completed.gradients
                        )
                    completed_reverse += 1
            if receive(output_queue) is not _STOP:
                raise RuntimeError("suffix forward pipeline omitted its completion marker")
            publish(reverse_queues[-1], _STOP)

            while completed_reverse < len(values):
                item = receive(completion_queue)
                if item is _STOP:
                    raise RuntimeError("suffix reverse pipeline stopped before all documents")
                assert isinstance(item, _ReverseDocument)
                if retain_input_gradients:
                    input_gradients[item.document.identifier] = item.gradients
                completed_reverse += 1
            if receive(completion_queue) is not _STOP:
                raise RuntimeError("suffix reverse pipeline omitted its completion marker")
        except BaseException as error:
            pipeline_error = error
        finally:
            abort.set()
            feeder.join(timeout=5.0)
            for worker in workers:
                worker.join(timeout=5.0)
        if not failures.empty():
            failure = failures.get()
            raise RuntimeError(
                f"suffix {failure.operation} stage {failure.stage} failed: "
                f"{failure.message}\n{failure.traceback}"
            ) from pipeline_error
        if pipeline_error is not None:
            raise pipeline_error
        if any(tape for tape in tapes):
            raise RuntimeError("suffix replay retained unconsumed autograd graphs")
        return SuffixTrainingGradients(
            kl_sum=kl_sum,
            token_count=token_count,
            stage_parameter_gradients=tuple(stage_sums),
            output_parameter_gradients=head_sums,
            input_gradients=input_gradients,
        )

    def evaluate(
        self,
        documents: Sequence[SuffixTrainingDocument],
        *,
        stage_observer: Callable[
            [int, SuffixTrainingDocument, nn.Module], None
        ]
        | None = None,
    ) -> SuffixTrainingEvaluation:
        """Evaluate dense KL with a bounded forward-only document pipeline."""

        values = tuple(documents)
        if not values:
            raise ValueError("suffix replay requires at least one document")
        identifiers = [value.identifier for value in values]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("suffix replay document identifiers must be unique")
        for value in values:
            value.validate()

        stage_count = len(self.stages)
        forward_queues = [queue.Queue(maxsize=self.queue_depth) for _ in self.stages]
        output_queue: queue.Queue[object] = queue.Queue(maxsize=self.queue_depth)
        failures: queue.Queue[_WorkerFailure] = queue.Queue()
        abort = threading.Event()

        def publish(target: queue.Queue[object], item: object) -> None:
            while not abort.is_set():
                try:
                    target.put(item, timeout=0.1)
                    return
                except queue.Full:
                    continue
            raise RuntimeError("suffix evaluation stopped while publishing work")

        def receive(source: queue.Queue[object]) -> object:
            while not abort.is_set():
                try:
                    return source.get(timeout=0.1)
                except queue.Empty:
                    continue
            raise RuntimeError("suffix evaluation stopped while awaiting work")

        def fail(stage: int, operation: str, error: BaseException) -> None:
            failures.put(
                _WorkerFailure(
                    stage=stage,
                    operation=operation,
                    message=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                )
            )
            abort.set()

        def forward_worker(stage: int) -> None:
            source = forward_queues[stage]
            target = output_queue if stage + 1 == stage_count else forward_queues[stage + 1]
            try:
                while True:
                    item = receive(source)
                    if item is _STOP:
                        publish(target, _STOP)
                        return
                    assert isinstance(item, _ForwardDocument)
                    state = item.state.to(self.stage_devices[stage])
                    with torch.no_grad():
                        output = self._stage_forward(self.stages[stage], state)
                    if stage_observer is not None:
                        stage_observer(stage, item.document, self.stages[stage])
                    publish(
                        target,
                        _ForwardDocument(
                            index=item.index,
                            document=item.document,
                            state=output.detached(),
                        ),
                    )
            except BaseException as error:
                fail(stage, "evaluation", error)

        def feed() -> None:
            try:
                for index, document in enumerate(values):
                    publish(
                        forward_queues[0],
                        _ForwardDocument(index, document, document.student_boundary),
                    )
                publish(forward_queues[0], _STOP)
            except BaseException as error:
                fail(-1, "evaluation feed", error)

        workers = [
            threading.Thread(
                target=forward_worker,
                args=(stage,),
                name=f"suffix-evaluate-{stage:03d}",
                daemon=True,
            )
            for stage in range(stage_count)
        ]
        feeder = threading.Thread(target=feed, name="suffix-evaluate-feed", daemon=True)
        kl_sum = 0.0
        token_count = 0
        try:
            for worker in workers:
                worker.start()
            feeder.start()
            completed = 0
            while completed < len(values):
                item = receive(output_queue)
                if item is _STOP:
                    raise RuntimeError(
                        "suffix evaluation stopped before all documents"
                    )
                assert isinstance(item, _ForwardDocument)
                result = self.loss_head.evaluate_document(
                    item.state,
                    item.document.teacher_normalized,
                )
                kl_sum += result.kl_sum
                token_count += result.token_count
                completed += 1
            if receive(output_queue) is not _STOP:
                raise RuntimeError("suffix evaluation omitted its completion marker")
        finally:
            abort.set()
            feeder.join(timeout=5.0)
            for worker in workers:
                worker.join(timeout=5.0)
        if not failures.empty():
            failure = failures.get()
            raise RuntimeError(
                f"suffix {failure.operation} stage {failure.stage} failed: "
                f"{failure.message}\n{failure.traceback}"
            )
        return SuffixTrainingEvaluation(kl_sum, token_count)


@dataclass(frozen=True)
class FP32AdamWConfig:
    """Stage-local AdamW configuration for FP32 master parameters."""

    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay_toward_initial: float = 0.0

    def validate(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate < 0.0:
            raise ValueError("learning rate must be finite and nonnegative")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("Adam beta values must lie in [0, 1)")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("Adam epsilon must be finite and positive")
        if (
            not math.isfinite(self.weight_decay_toward_initial)
            or self.weight_decay_toward_initial < 0.0
        ):
            raise ValueError("decay toward initial parameters must be nonnegative")


@dataclass(frozen=True)
class FP32AdamWStep:
    """Norm accounting for one completed local optimizer step."""

    step: int
    learning_rate: float
    gradient_norm: float
    clipping_scale: float
    parameter_update_norms: Mapping[str, float]


class FP32MasterAdamW:
    """AdamW with FP32 gradients, masters, and moments for one stage owner."""

    def __init__(
        self,
        parameters: Mapping[str, nn.Parameter],
        config: FP32AdamWConfig,
    ):
        config.validate()
        if not parameters:
            raise ValueError("FP32 AdamW requires at least one parameter")
        self.parameters = dict(parameters)
        if len(self.parameters) != len(parameters):
            raise ValueError("FP32 AdamW parameter names must be unique")
        if any(not value.requires_grad for value in self.parameters.values()):
            raise ValueError("FP32 AdamW received a frozen parameter")
        if any(not value.is_floating_point() for value in self.parameters.values()):
            raise ValueError("FP32 AdamW parameters must be floating point")
        self.config = config
        self.master = {
            name: value.detach().float().clone()
            for name, value in self.parameters.items()
        }
        self.initial = {name: value.clone() for name, value in self.master.items()}
        self.first_moment = {
            name: torch.zeros_like(value) for name, value in self.master.items()
        }
        self.second_moment = {
            name: torch.zeros_like(value) for name, value in self.master.items()
        }
        self.gradient_sums = {
            name: torch.zeros_like(value) for name, value in self.master.items()
        }
        self.present = {name: False for name in self.parameters}
        self.step_count = 0

    def accumulate(self, gradients: Mapping[str, torch.Tensor]) -> None:
        unknown = set(gradients) - self.parameters.keys()
        if unknown:
            raise KeyError(f"optimizer received unknown gradients {sorted(unknown)}")
        for name, gradient in gradients.items():
            if gradient.shape != self.parameters[name].shape:
                raise ValueError(f"gradient for {name} has incompatible shape")
            self.gradient_sums[name].add_(
                gradient.detach().to(
                    device=self.gradient_sums[name].device,
                    dtype=torch.float32,
                )
            )
            self.present[name] = True

    def scaled_gradient_norm_squared(self, gradient_scale: float) -> torch.Tensor:
        if not math.isfinite(gradient_scale) or gradient_scale <= 0.0:
            raise ValueError("gradient scale must be finite and positive")
        values = [
            torch.sum((self.gradient_sums[name] * gradient_scale).square())
            for name in self.parameters
            if self.present[name]
        ]
        if not values:
            return torch.tensor(0.0, dtype=torch.float64)
        result = values[0].double()
        for value in values[1:]:
            result = result + value.double()
        return result

    def step(
        self,
        *,
        gradient_scale: float,
        max_gradient_norm: float | None = None,
        global_gradient_norm: float | None = None,
        learning_rate: float | None = None,
    ) -> FP32AdamWStep:
        if max_gradient_norm is not None and (
            not math.isfinite(max_gradient_norm) or max_gradient_norm <= 0.0
        ):
            raise ValueError("maximum gradient norm must be finite and positive")
        local_norm = float(torch.sqrt(self.scaled_gradient_norm_squared(gradient_scale)))
        norm = local_norm if global_gradient_norm is None else float(global_gradient_norm)
        if not math.isfinite(norm) or norm < 0.0:
            raise ValueError("global gradient norm must be finite and nonnegative")
        rate = self.config.learning_rate if learning_rate is None else learning_rate
        if not math.isfinite(rate) or rate < 0.0:
            raise ValueError("learning rate must be finite and nonnegative")
        clipping_scale = (
            1.0
            if max_gradient_norm is None or norm <= max_gradient_norm
            else max_gradient_norm / norm
        )
        self.step_count += 1
        beta1 = self.config.beta1
        beta2 = self.config.beta2
        bias1 = 1.0 - beta1**self.step_count
        bias2_sqrt = math.sqrt(1.0 - beta2**self.step_count)
        step_size = rate / bias1
        updates: dict[str, float] = {}
        for name, parameter in self.parameters.items():
            if not self.present[name]:
                continue
            gradient = self.gradient_sums[name]
            gradient.mul_(gradient_scale * clipping_scale)
            first = self.first_moment[name]
            second = self.second_moment[name]
            first.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            second.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            before = self.master[name].clone()
            denominator = second.sqrt().div_(bias2_sqrt).add_(self.config.epsilon)
            self.master[name].addcdiv_(first, denominator, value=-step_size)
            if self.config.weight_decay_toward_initial:
                self.master[name].add_(
                    self.master[name] - self.initial[name],
                    alpha=(
                        -rate
                        * self.config.weight_decay_toward_initial
                    ),
                )
            parameter.data.copy_(self.master[name].to(dtype=parameter.dtype))
            updates[name] = float(torch.linalg.vector_norm(self.master[name] - before))
        self.zero_gradients()
        return FP32AdamWStep(
            step=self.step_count,
            learning_rate=rate,
            gradient_norm=norm,
            clipping_scale=clipping_scale,
            parameter_update_norms=updates,
        )

    def zero_gradients(self) -> None:
        for name, value in self.gradient_sums.items():
            value.zero_()
            self.present[name] = False


def combined_gradient_norm(
    optimizers: Sequence[FP32MasterAdamW],
    *,
    gradient_scale: float,
) -> float:
    """Return one global norm without moving optimizer state between stages."""

    values = [
        optimizer.scaled_gradient_norm_squared(gradient_scale).detach().cpu()
        for optimizer in optimizers
    ]
    return math.sqrt(sum(float(value) for value in values))


def is_shared_expert_or_norm_tensor(name: str) -> bool:
    """Return whether a checkpoint tensor belongs to the first training arm."""

    return (
        ".block_sparse_moe.shared_experts." in name and name.endswith(".weight")
    ) or name.endswith("norm.weight")


__all__ = [
    "DenseDistributionKLLossHead",
    "DenseKLLossBackward",
    "FP32AdamWConfig",
    "FP32AdamWStep",
    "FP32MasterAdamW",
    "SuffixReplayTrainer",
    "SuffixState",
    "SuffixTrainingDocument",
    "SuffixTrainingGradients",
    "combined_gradient_norm",
    "is_shared_expert_or_norm_tensor",
]

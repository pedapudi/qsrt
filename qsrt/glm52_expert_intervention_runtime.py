"""Apply bounded dense-expert substitutions inside a resident GLM-5.2 model.

The runtime first executes the unchanged EXL3 mixture-of-experts layer.  For
selected routes it then evaluates the decoded resident expert and a candidate
expert on the same input and adds their output difference.  Tensor-parallel
ranks own disjoint 512-neuron gate, up, and down slices, so the existing vLLM
all-reduce reconstructs the complete substituted expert output.

This module is an experiment hook, not a serving format. It accepts only the
hash-bound dense endpoint artifact emitted by
``qsrt.glm52_expert_intervention`` at tensor parallelism four. The artifact
declares the one layer that may be changed. Input capture may observe one or
more explicitly configured mixture-of-experts layers without changing them.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from qsrt.glm52_expert_intervention import INTERVENTION_ARTIFACT_KIND
from qsrt.glm52_engine_kld import install_vllm_engine_kld_patch
from qsrt.glm52_pilot import HIDDEN_SIZE, INTERMEDIATE_SIZE, TP_RANKS


CONTROL_SCHEMA = "qsrt_glm52_expert_intervention_control_v2"
FIRST_MOE_LAYER = 3
LAST_MOE_LAYER = 77
CANDIDATE_MODE = "candidate"
FACTORIZED_LOW_RANK_CANDIDATE_MODE = "factorized_low_rank_candidate"
MATERIALIZED_LOW_RANK_CANDIDATE_MODE = (
    "stored_low_rank_factors_materialized_at_load_candidate"
)
LEGACY_K3_CANDIDATE_MODE = "qsrt_k3"
IDENTITY_CONTROL_MODE = "dense_resident_identity"
SUPPORTED_MODES = (
    "off",
    IDENTITY_CONTROL_MODE,
    CANDIDATE_MODE,
    FACTORIZED_LOW_RANK_CANDIDATE_MODE,
    MATERIALIZED_LOW_RANK_CANDIDATE_MODE,
    LEGACY_K3_CANDIDATE_MODE,
)
RANK_INTERMEDIATE_SIZE = INTERMEDIATE_SIZE // TP_RANKS
FORCE_PER_EXPERT_EXL3_MOE_ENV = "QSRT_GLM52_FORCE_PER_EXPERT_EXL3_MOE"
ACTIVATION_CAPTURE_LAYERS_ENV = "QSRT_GLM52_ACTIVATION_CAPTURE_LAYERS"


def _validate_model_layer(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not FIRST_MOE_LAYER <= value <= LAST_MOE_LAYER
    ):
        raise ValueError(
            f"model layer must be an integer from {FIRST_MOE_LAYER} through "
            f"{LAST_MOE_LAYER}"
        )
    return value


def _layer_name(model_layer: int) -> str:
    return f"model.layers.{_validate_model_layer(model_layer)}.mlp.experts"


def _parse_capture_layers(value: str | None, *, default_layer: int) -> tuple[int, ...]:
    """Parse the mixture-of-experts layers observed by one capture run."""

    if value is None:
        return (_validate_model_layer(default_layer),)
    pieces = value.split(",")
    if not pieces or any(not piece or not piece.isdecimal() for piece in pieces):
        raise ValueError(
            f"{ACTIVATION_CAPTURE_LAYERS_ENV} must be a comma-separated layer list"
        )
    layers = tuple(_validate_model_layer(int(piece)) for piece in pieces)
    if len(set(layers)) != len(layers):
        raise ValueError(f"{ACTIVATION_CAPTURE_LAYERS_ENV} repeats a layer")
    return layers


def _canonical_json_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_control(
    path: Path,
    *,
    mode: str,
    artifact_manifest_sha256: str,
    generation: int,
    capture_enabled: bool = False,
    selected_experts: tuple[int, ...] | list[int] | None = None,
) -> None:
    """Switch one resident experiment between the unmodified and candidate arms."""

    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported intervention mode {mode!r}")
    normalized_experts = _validate_selected_experts(selected_experts)
    value = {
        "schema": CONTROL_SCHEMA,
        "mode": mode,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "generation": int(generation),
        "capture_enabled": bool(capture_enabled),
        "selected_experts": normalized_experts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_control(path: Path, *, expected_manifest_sha256: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("intervention control must contain one JSON object")
    if value.get("schema") != CONTROL_SCHEMA:
        raise ValueError("intervention control schema mismatch")
    if value.get("artifact_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("intervention control artifact identity mismatch")
    mode = value.get("mode")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported intervention mode {mode!r}")
    generation = value.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("intervention generation must be a nonnegative integer")
    if not isinstance(value.get("capture_enabled"), bool):
        raise ValueError("intervention capture_enabled must be a boolean")
    value["selected_experts"] = _validate_selected_experts(
        value.get("selected_experts")
    )
    return value


def _validate_selected_experts(
    selected_experts: tuple[int, ...] | list[int] | None,
) -> list[int] | None:
    if selected_experts is None:
        return None
    if not isinstance(selected_experts, (tuple, list)):
        raise TypeError("selected_experts must be a sequence or null")
    normalized = list(selected_experts)
    if any(
        isinstance(expert, bool)
        or not isinstance(expert, int)
        or not 0 <= expert < 256
        for expert in normalized
    ):
        raise ValueError("selected_experts must contain expert IDs from 0 through 255")
    if len(set(normalized)) != len(normalized):
        raise ValueError("selected_experts must not repeat an expert ID")
    return normalized


def validate_dense_intervention_artifact(root: Path) -> dict[str, Any]:
    """Hash every selected dense endpoint before a model process loads it."""

    root = root.resolve(strict=True)
    manifest_path = root / "manifest.json"
    report_path = root / "report.json"
    manifest = json.loads(manifest_path.read_text())
    report = json.loads(report_path.read_text())
    if manifest.get("kind") != f"{INTERVENTION_ARTIFACT_KIND}_manifest":
        raise ValueError("dense intervention manifest kind mismatch")
    manifest_sha256 = _canonical_json_sha256(manifest)
    model_layer = _validate_model_layer(report.get("layer"))
    required_report = {
        "kind": INTERVENTION_ARTIFACT_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
    }
    for field, expected in required_report.items():
        if report.get(field) != expected:
            raise ValueError(
                f"dense intervention report field {field!r} is "
                f"{report.get(field)!r}, expected {expected!r}"
            )
    endpoint = manifest.get("exl3_endpoint")
    if endpoint is None:
        endpoint = manifest.get("exl3_endpoint_identity")
    input_artifact = manifest.get("input_intervention_artifact")
    if endpoint is not None:
        if not isinstance(endpoint, dict) or endpoint.get("layer") != model_layer:
            raise ValueError("dense intervention manifest and report layer mismatch")
    elif input_artifact is not None:
        input_manifest_sha256 = None
        input_report_sha256 = None
        input_root = None
        if isinstance(input_artifact, dict):
            input_manifest_sha256 = input_artifact.get("manifest_sha256")
            input_report_sha256 = input_artifact.get("report_sha256")
            input_root = input_artifact.get("root")

        def is_sha256(value: object) -> bool:
            return (
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
            )

        if (
            not isinstance(input_artifact, dict)
            or not is_sha256(input_manifest_sha256)
            or not is_sha256(input_report_sha256)
            or (input_root is not None and not isinstance(input_root, str))
        ):
            raise ValueError("dense intervention input-artifact identity is invalid")
    else:
        raise ValueError("dense intervention manifest has no endpoint identity")
    candidate = manifest.get("candidate")
    if not isinstance(candidate, dict):
        raise TypeError("dense intervention manifest candidate must be an object")
    tensor_prefix = candidate.get("tensor_prefix", LEGACY_K3_CANDIDATE_MODE)
    if (
        not isinstance(tensor_prefix, str)
        or not tensor_prefix
        or any(not (character.isalnum() or character == "_") for character in tensor_prefix)
    ):
        raise ValueError("dense intervention candidate tensor prefix is unsafe")
    experts = report.get("experts")
    if not isinstance(experts, list) or not experts:
        raise ValueError("dense intervention report has no experts")
    seen: set[int] = set()
    total_bytes = 0
    for record in experts:
        if not isinstance(record, dict):
            raise TypeError("dense intervention expert receipt must be an object")
        expert = record.get("expert")
        if (
            isinstance(expert, bool)
            or not isinstance(expert, int)
            or not 0 <= expert < 256
            or expert in seen
        ):
            raise ValueError("dense intervention expert IDs must be unique 0..255")
        if record.get("layer") != model_layer:
            raise ValueError(f"expert {expert} receipt layer mismatch")
        seen.add(expert)
        filename = record.get("dense_endpoint_file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"expert {expert} has an unsafe dense endpoint filename")
        path = root / "experts" / filename
        expected_bytes = int(record["dense_endpoint_file_bytes"])
        expected_sha256 = record.get("dense_endpoint_file_sha256")
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ValueError(f"expert {expert} dense endpoint size mismatch")
        if not isinstance(expected_sha256, str) or _sha256(path) != expected_sha256:
            raise ValueError(f"expert {expert} dense endpoint SHA-256 mismatch")
        total_bytes += expected_bytes
    panel = manifest.get("panel")
    if panel is not None:
        panel_experts = panel.get(str(model_layer)) if isinstance(panel, dict) else None
        if (
            not isinstance(panel_experts, list)
            or len(panel) != 1
            or any(
                isinstance(expert, bool) or not isinstance(expert, int)
                for expert in panel_experts
            )
            or set(panel_experts) != seen
        ):
            raise ValueError("dense intervention manifest panel disagrees with report")
    if report.get("expert_count") != len(experts):
        raise ValueError("dense intervention expert count mismatch")
    if report.get("dense_endpoint_bytes") != total_bytes:
        raise ValueError("dense intervention byte total mismatch")
    return {
        "root": str(root),
        "manifest_sha256": manifest_sha256,
        "expert_ids": tuple(sorted(seen)),
        "expert_count": len(seen),
        "dense_endpoint_bytes": total_bytes,
        "candidate_tensor_prefix": tensor_prefix,
        "model_layer": model_layer,
        "report": report,
    }


@dataclass(frozen=True)
class DenseExpertSlice:
    """One tensor-parallel rank's gate, up, and down endpoint slices."""

    exl3_gate: torch.Tensor
    exl3_up: torch.Tensor
    exl3_down: torch.Tensor
    candidate_gate: torch.Tensor
    candidate_up: torch.Tensor
    candidate_down: torch.Tensor
    candidate_down_base: torch.Tensor | None = None
    candidate_down_factor_a: torch.Tensor | None = None
    candidate_down_factor_b: torch.Tensor | None = None
    candidate_down_materialized_from_factors: torch.Tensor | None = None


def evaluate_expert(
    expert_input: torch.Tensor,
    *,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one tensor-parallel SwiGLU expert slice in FP16."""

    input_fp16 = expert_input.to(torch.float16)
    gate_values = F.linear(input_fp16, gate)
    up_values = F.linear(input_fp16, up)
    hidden = F.silu(gate_values) * up_values
    return F.linear(hidden, down)


def evaluate_expert_with_factorized_down(
    expert_input: torch.Tensor,
    *,
    gate: torch.Tensor,
    up: torch.Tensor,
    down_base: torch.Tensor,
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one expert slice with a stored low-rank down correction."""

    input_fp16 = expert_input.to(torch.float16)
    gate_values = F.linear(input_fp16, gate)
    up_values = F.linear(input_fp16, up)
    hidden_fp16 = F.silu(gate_values) * up_values
    base_output = F.linear(hidden_fp16, down_base)
    hidden_bf16 = hidden_fp16.to(torch.bfloat16)
    rank_values = F.linear(hidden_bf16, factor_a.T.contiguous())
    correction = F.linear(rank_values, factor_b)
    return (base_output.float() + correction.float()).to(torch.float16)


def routed_candidate_delta(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    expert_slices: Mapping[int, DenseExpertSlice],
    use_resident_endpoint: bool = False,
    use_factorized_low_rank_down: bool = False,
    use_materialized_low_rank_down: bool = False,
) -> torch.Tensor:
    """Return the route-weighted candidate-minus-EXL3 local expert output."""

    if use_factorized_low_rank_down and use_materialized_low_rank_down:
        raise ValueError("low-rank down execution modes are mutually exclusive")

    original_shape = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1])
    ids = topk_ids.reshape(x_2d.shape[0], -1)
    weights = topk_weights.reshape_as(ids).float()
    delta = torch.zeros(
        (x_2d.shape[0], HIDDEN_SIZE), dtype=torch.float32, device=x.device
    )
    for expert_id, endpoint in expert_slices.items():
        positions = (ids == expert_id).nonzero(as_tuple=False)
        if positions.shape[0] == 0:
            continue
        token_ids = positions[:, 0]
        route_ids = positions[:, 1]
        expert_input = x_2d.index_select(0, token_ids)
        exl3_output = evaluate_expert(
            expert_input,
            gate=endpoint.exl3_gate,
            up=endpoint.exl3_up,
            down=endpoint.exl3_down,
        )
        if use_resident_endpoint:
            # Reusing the same result makes the wiring control exactly zero.
            # Evaluating an identical GEMM twice is not a numerical identity on
            # every GPU kernel: different accumulation schedules can create a
            # small nonzero difference that later changes expert routing.
            candidate_output = exl3_output
        elif use_factorized_low_rank_down:
            if (
                endpoint.candidate_down_base is None
                or endpoint.candidate_down_factor_a is None
                or endpoint.candidate_down_factor_b is None
            ):
                raise ValueError(
                    f"expert {expert_id} has no factorized low-rank down endpoint"
                )
            candidate_output = evaluate_expert_with_factorized_down(
                expert_input,
                gate=endpoint.candidate_gate,
                up=endpoint.candidate_up,
                down_base=endpoint.candidate_down_base,
                factor_a=endpoint.candidate_down_factor_a,
                factor_b=endpoint.candidate_down_factor_b,
            )
        elif use_materialized_low_rank_down:
            if endpoint.candidate_down_materialized_from_factors is None:
                raise ValueError(
                    f"expert {expert_id} has no load-time-materialized low-rank "
                    "down endpoint"
                )
            candidate_output = evaluate_expert(
                expert_input,
                gate=endpoint.candidate_gate,
                up=endpoint.candidate_up,
                down=endpoint.candidate_down_materialized_from_factors,
            )
        else:
            candidate_output = evaluate_expert(
                expert_input,
                gate=endpoint.candidate_gate,
                up=endpoint.candidate_up,
                down=endpoint.candidate_down,
            )
        route_weight = weights[token_ids, route_ids].unsqueeze(-1)
        delta.index_add_(
            0,
            token_ids,
            (candidate_output.float() - exl3_output.float()) * route_weight,
        )
    return delta.reshape(*original_shape, HIDDEN_SIZE)


class DenseEndpointStore:
    """Lazily load only this process's tensor-parallel endpoint slices."""

    def __init__(
        self,
        root: Path,
        *,
        device: torch.device,
        tensor_parallel_rank: int,
        expected_manifest_sha256: str,
    ):
        if not 0 <= tensor_parallel_rank < TP_RANKS:
            raise ValueError("tensor-parallel rank must be 0, 1, 2, or 3")
        self.root = root.resolve(strict=True)
        self.device = device
        self.rank = tensor_parallel_rank
        manifest = json.loads((self.root / "manifest.json").read_text())
        if _canonical_json_sha256(manifest) != expected_manifest_sha256:
            raise ValueError("runtime intervention manifest identity mismatch")
        candidate = manifest.get("candidate")
        if not isinstance(candidate, dict):
            raise TypeError("runtime intervention candidate must be an object")
        self.candidate_tensor_prefix = candidate.get(
            "tensor_prefix", LEGACY_K3_CANDIDATE_MODE
        )
        report = json.loads((self.root / "report.json").read_text())
        self.model_layer = _validate_model_layer(report.get("layer"))
        if (
            report.get("kind") != INTERVENTION_ARTIFACT_KIND
            or report.get("status") != "complete"
            or report.get("manifest_sha256") != expected_manifest_sha256
        ):
            raise ValueError("runtime intervention report identity mismatch")
        records = report.get("experts")
        if not isinstance(records, list) or not records:
            raise ValueError("runtime intervention report has no selected experts")
        self.records = {int(record["expert"]): record for record in records}
        if len(self.records) != len(records):
            raise ValueError("runtime intervention report repeats an expert")
        self._loaded: dict[int, DenseExpertSlice] = {}

    @property
    def expert_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.records))

    def _load(self, expert: int) -> DenseExpertSlice:
        record = self.records[expert]
        filename = record["dense_endpoint_file"]
        path = self.root / "experts" / filename
        if path.stat().st_size != int(record["dense_endpoint_file_bytes"]):
            raise ValueError(f"expert {expert} dense endpoint size changed")
        start = self.rank * RANK_INTERMEDIATE_SIZE
        stop = start + RANK_INTERMEDIATE_SIZE

        def load_slice(handle: Any, endpoint: str, projection: str) -> torch.Tensor:
            key = f"{endpoint}.{projection}"
            sliced = handle.get_slice(key)
            shape = tuple(sliced.get_shape())
            if projection in ("gate_proj", "up_proj"):
                if shape != (INTERMEDIATE_SIZE, HIDDEN_SIZE):
                    raise ValueError(f"{key} has invalid shape {shape}")
                value = sliced[start:stop, :]
                expected = (RANK_INTERMEDIATE_SIZE, HIDDEN_SIZE)
            else:
                if shape != (HIDDEN_SIZE, INTERMEDIATE_SIZE):
                    raise ValueError(f"{key} has invalid shape {shape}")
                value = sliced[:, start:stop]
                expected = (HIDDEN_SIZE, RANK_INTERMEDIATE_SIZE)
            value = value.to(device=self.device, dtype=torch.float16).contiguous()
            if tuple(value.shape) != expected or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{key} produced an invalid tensor-parallel slice")
            return value

        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            low_rank_keys = {
                "adapter.down.base",
                "adapter.down.a",
                "adapter.down.b",
            }
            present_low_rank_keys = low_rank_keys & keys
            if present_low_rank_keys and present_low_rank_keys != low_rank_keys:
                raise ValueError(
                    f"expert {expert} has an incomplete factorized down endpoint"
                )

            candidate_down_base = None
            candidate_down_factor_a = None
            candidate_down_factor_b = None
            if present_low_rank_keys:
                base_slice = handle.get_slice("adapter.down.base")
                if tuple(base_slice.get_shape()) != (HIDDEN_SIZE, INTERMEDIATE_SIZE):
                    raise ValueError("adapter.down.base has invalid shape")
                candidate_down_base = (
                    base_slice[:, start:stop]
                    .to(device=self.device, dtype=torch.float16)
                    .contiguous()
                )
                factor_a_slice = handle.get_slice("adapter.down.a")
                factor_a_shape = tuple(factor_a_slice.get_shape())
                if len(factor_a_shape) != 2 or factor_a_shape[0] != INTERMEDIATE_SIZE:
                    raise ValueError("adapter.down.a has invalid shape")
                rank = factor_a_shape[1]
                candidate_down_factor_a = (
                    factor_a_slice[start:stop, :]
                    .to(device=self.device, dtype=torch.bfloat16)
                    .contiguous()
                )
                factor_b_value = handle.get_tensor("adapter.down.b")
                if tuple(factor_b_value.shape) != (HIDDEN_SIZE, rank):
                    raise ValueError("adapter.down.b has invalid shape")
                candidate_down_factor_b = factor_b_value.to(
                    device=self.device, dtype=torch.bfloat16
                ).contiguous()
            candidate_down = load_slice(
                handle, self.candidate_tensor_prefix, "down_proj"
            )
            candidate_down_materialized_from_factors = None
            if present_low_rank_keys:
                assert candidate_down_base is not None
                assert candidate_down_factor_a is not None
                assert candidate_down_factor_b is not None
                candidate_down_materialized_from_factors = (
                    candidate_down_base.float()
                    + candidate_down_factor_b.float()
                    @ candidate_down_factor_a.float().T
                ).to(torch.float16)
                if not torch.equal(
                    candidate_down_materialized_from_factors, candidate_down
                ):
                    raise ValueError(
                        f"expert {expert} stored low-rank factors do not reconstruct "
                        "the materialized FP16 down endpoint"
                    )
            return DenseExpertSlice(
                exl3_gate=load_slice(handle, "exl3", "gate_proj"),
                exl3_up=load_slice(handle, "exl3", "up_proj"),
                exl3_down=load_slice(handle, "exl3", "down_proj"),
                candidate_gate=load_slice(
                    handle, self.candidate_tensor_prefix, "gate_proj"
                ),
                candidate_up=load_slice(
                    handle, self.candidate_tensor_prefix, "up_proj"
                ),
                candidate_down=candidate_down,
                candidate_down_base=candidate_down_base,
                candidate_down_factor_a=candidate_down_factor_a,
                candidate_down_factor_b=candidate_down_factor_b,
                candidate_down_materialized_from_factors=(
                    candidate_down_materialized_from_factors
                ),
            )

    def selected_slices(self) -> Mapping[int, DenseExpertSlice]:
        for expert in self.expert_ids:
            self.expert_slice(expert)
        return self._loaded

    def expert_slice(self, expert: int) -> DenseExpertSlice:
        """Load one selected expert without materializing the rest of the panel."""

        if expert not in self.records:
            raise KeyError(f"expert {expert} is not present in the intervention artifact")
        if expert not in self._loaded:
            self._loaded[expert] = self._load(expert)
        return self._loaded[expert]


_PATCH_LOCK = Lock()
_PATCHED = False
_STORES: dict[tuple[str, int, str], DenseEndpointStore] = {}
_CAPTURE_CHUNK = 0
_MISSING_ATTRIBUTE = object()


@contextmanager
def _r7_fused_kernel_selection_disabled(layer: Any):
    """Temporarily expose vLLM's non-fused EXL3 preparation and execution."""

    attributes = ("exl3_r7_fused", "exl3_r7_graph")
    previous = {
        attribute: getattr(layer, attribute, _MISSING_ATTRIBUTE)
        for attribute in attributes
    }
    for attribute in attributes:
        setattr(layer, attribute, False)
    try:
        yield
    finally:
        for attribute, value in previous.items():
            if value is _MISSING_ATTRIBUTE:
                delattr(layer, attribute)
            else:
                setattr(layer, attribute, value)


def _apply_with_per_expert_exl3_moe(
    original_apply: Any,
    method: Any,
    layer: Any,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts: Any,
    shared_experts_input: torch.Tensor | None,
) -> torch.Tensor:
    """Run vLLM's existing three-GEMM-per-expert EXL3 correctness path.

    The GLM-5.2 R7 loader retains the raw per-expert tensors after it prepares
    its fused execution structures.  The installed EXL3 method selects those
    fused structures through two layer attributes.  Temporarily clearing both
    attributes reaches the method's existing correctness implementation while
    preserving the checkpoint and prepared structures for later calls.
    """

    with _r7_fused_kernel_selection_disabled(layer):
        return original_apply(
            method,
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts,
            shared_experts_input,
        )


def _capture_layer_input(
    *,
    root: Path,
    model_layer: int,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    generation: int,
    plan_sha256: str,
) -> Path:
    """Atomically save one rank-zero layer-input chunk on the host filesystem."""

    global _CAPTURE_CHUNK
    model_layer = _validate_model_layer(model_layer)
    if len(plan_sha256) != 64 or any(character not in "0123456789abcdef" for character in plan_sha256):
        raise ValueError("activation-capture plan identity must be a lowercase SHA-256")
    root.mkdir(parents=True, exist_ok=True)
    chunk = _CAPTURE_CHUNK
    _CAPTURE_CHUNK += 1
    destination = root / (
        f"layer-{model_layer:03d}-input-chunk-{chunk:06d}.safetensors"
    )
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    x_2d = x.detach().reshape(-1, x.shape[-1]).to("cpu", copy=True).contiguous()
    ids = (
        topk_ids.detach()
        .reshape(x_2d.shape[0], -1)
        .to("cpu", dtype=torch.int32, copy=True)
        .contiguous()
    )
    weights = (
        topk_weights.detach()
        .reshape_as(ids)
        .to("cpu", dtype=torch.float32, copy=True)
        .contiguous()
    )
    try:
        save_file(
            {
                "hidden_states": x_2d,
                "topk_ids": ids,
                "topk_weights": weights,
            },
            temporary,
            metadata={
                "schema": "qsrt_glm52_layer_input_capture_v1",
                "model_layer": str(model_layer),
                "control_generation": str(generation),
                "corpus_plan_sha256": plan_sha256,
            },
        )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def install_vllm_patch() -> None:
    """Patch the installed EXL3 method when the experiment environment is set."""

    global _PATCHED
    with _PATCH_LOCK:
        if _PATCHED:
            return
        root_text = os.environ.get("QSRT_GLM52_INTERVENTION_ROOT")
        control_text = os.environ.get("QSRT_GLM52_INTERVENTION_CONTROL")
        manifest_sha256 = os.environ.get("QSRT_GLM52_INTERVENTION_MANIFEST_SHA256")
        if not root_text and not control_text and not manifest_sha256:
            return
        if not root_text or not control_text or not manifest_sha256:
            raise RuntimeError(
                "QSRT GLM-5.2 intervention requires root, control, and manifest "
                "SHA-256 environment variables together"
            )
        force_per_expert_text = os.environ.get(FORCE_PER_EXPERT_EXL3_MOE_ENV, "0")
        if force_per_expert_text not in ("0", "1"):
            raise RuntimeError(
                f"{FORCE_PER_EXPERT_EXL3_MOE_ENV} must be 0 or 1, got "
                f"{force_per_expert_text!r}"
            )
        force_per_expert_moe = force_per_expert_text == "1"
        install_vllm_engine_kld_patch()
        root = Path(root_text)
        control_path = Path(control_text)
        manifest = json.loads((root / "manifest.json").read_text())
        if _canonical_json_sha256(manifest) != manifest_sha256:
            raise RuntimeError("intervention manifest identity mismatch")
        artifact_report = json.loads((root / "report.json").read_text())
        target_model_layer = _validate_model_layer(artifact_report.get("layer"))
        target_layer_name = _layer_name(target_model_layer)
        capture_root_text = os.environ.get("QSRT_GLM52_ACTIVATION_CAPTURE_DIR")
        capture_plan_sha256 = os.environ.get(
            "QSRT_GLM52_ACTIVATION_CAPTURE_PLAN_SHA256"
        )
        if bool(capture_root_text) != bool(capture_plan_sha256):
            raise RuntimeError(
                "activation capture requires its directory and corpus-plan "
                "SHA-256 environment variables together"
            )
        capture_model_layers = (
            _parse_capture_layers(
                os.environ.get(ACTIVATION_CAPTURE_LAYERS_ENV),
                default_layer=target_model_layer,
            )
            if capture_root_text
            else ()
        )
        capture_layer_names = {
            _layer_name(model_layer): model_layer
            for model_layer in capture_model_layers
        }

        from vllm.distributed import (  # type: ignore[import-not-found]
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.model_executor.layers.quantization.exl3 import (  # type: ignore[import-not-found]
            Exl3MoEMethod,
        )

        original_apply = Exl3MoEMethod.apply
        original_process_weights = Exl3MoEMethod.process_weights_after_loading

        def patched_process_weights(self: Any, layer: Any) -> None:
            if force_per_expert_moe and not getattr(
                layer, "exl3_rank_sliced", False
            ):
                # The correctness path consumes the raw per-expert tensors and
                # does not need either R7 fused staging arena.  Skip their
                # construction so a large scheduler prefill contract does not
                # reserve several unused GiB on every tensor-parallel rank.
                with _r7_fused_kernel_selection_disabled(layer):
                    original_process_weights(self, layer)
                return
            original_process_weights(self, layer)

        def patched_apply(
            self: Any,
            layer: Any,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts: Any,
            shared_experts_input: torch.Tensor | None,
        ) -> torch.Tensor:
            if force_per_expert_moe and not getattr(
                layer, "exl3_rank_sliced", False
            ):
                output = _apply_with_per_expert_exl3_moe(
                    original_apply,
                    self,
                    layer,
                    x,
                    topk_weights,
                    topk_ids,
                    shared_experts,
                    shared_experts_input,
                )
            else:
                output = original_apply(
                    self,
                    layer,
                    x,
                    topk_weights,
                    topk_ids,
                    shared_experts,
                    shared_experts_input,
                )
            layer_name = layer.layer_name
            capture_model_layer = capture_layer_names.get(layer_name)
            if capture_model_layer is None and layer_name != target_layer_name:
                return output
            control = read_control(
                control_path, expected_manifest_sha256=manifest_sha256
            )
            world_size = get_tensor_model_parallel_world_size()
            rank = get_tensor_model_parallel_rank()
            if (
                capture_model_layer is not None
                and capture_root_text
                and control["mode"] == "off"
                and control["capture_enabled"]
                and rank == 0
                and x.numel() // x.shape[-1] >= 128
            ):
                capture_root = Path(capture_root_text)
                if len(capture_model_layers) > 1:
                    capture_root = capture_root / f"layer-{capture_model_layer:03d}"
                _capture_layer_input(
                    root=capture_root,
                    model_layer=capture_model_layer,
                    x=x,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    generation=control["generation"],
                    plan_sha256=str(capture_plan_sha256),
                )
            if layer_name != target_layer_name:
                return output
            if control["mode"] == "off":
                return output
            if world_size != TP_RANKS:
                raise RuntimeError(
                    f"dense intervention requires tensor parallelism {TP_RANKS}, "
                    f"got {world_size}"
                )
            key = (str(x.device), rank, manifest_sha256)
            store = _STORES.get(key)
            if store is None:
                store = DenseEndpointStore(
                    root,
                    device=x.device,
                    tensor_parallel_rank=rank,
                    expected_manifest_sha256=manifest_sha256,
                )
                _STORES[key] = store
            if control["mode"] == IDENTITY_CONTROL_MODE:
                # Load and validate the requested dense endpoints, but return
                # the resident kernel result without arithmetic.  Even adding
                # an exactly zero FP32 delta introduces a dtype round trip and
                # is therefore not a valid no-op control for later routing.
                if control["selected_experts"] is None:
                    store.selected_slices()
                else:
                    for expert in control["selected_experts"]:
                        store.expert_slice(expert)
                return output
            delta = routed_candidate_delta(
                x,
                topk_weights,
                topk_ids,
                expert_slices=(
                    store.selected_slices()
                    if control["selected_experts"] is None
                    else {
                        expert: store.expert_slice(expert)
                        for expert in control["selected_experts"]
                    }
                ),
                use_factorized_low_rank_down=(
                    control["mode"] == FACTORIZED_LOW_RANK_CANDIDATE_MODE
                ),
                use_materialized_low_rank_down=(
                    control["mode"] == MATERIALIZED_LOW_RANK_CANDIDATE_MODE
                ),
            )
            return (output.float() + delta).to(output.dtype)

        Exl3MoEMethod.process_weights_after_loading = patched_process_weights
        Exl3MoEMethod.apply = patched_apply
        _PATCHED = True


__all__ = [
    "CONTROL_SCHEMA",
    "ACTIVATION_CAPTURE_LAYERS_ENV",
    "FORCE_PER_EXPERT_EXL3_MOE_ENV",
    "DenseEndpointStore",
    "DenseExpertSlice",
    "CANDIDATE_MODE",
    "FACTORIZED_LOW_RANK_CANDIDATE_MODE",
    "MATERIALIZED_LOW_RANK_CANDIDATE_MODE",
    "IDENTITY_CONTROL_MODE",
    "LEGACY_K3_CANDIDATE_MODE",
    "SUPPORTED_MODES",
    "atomic_write_control",
    "evaluate_expert",
    "evaluate_expert_with_factorized_down",
    "_capture_layer_input",
    "_parse_capture_layers",
    "_apply_with_per_expert_exl3_moe",
    "install_vllm_patch",
    "read_control",
    "routed_candidate_delta",
    "validate_dense_intervention_artifact",
]

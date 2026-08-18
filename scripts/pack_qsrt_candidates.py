#!/usr/bin/env python3
"""Build the all-expert QSRT candidate pool.

The resident teacher is represented only by its finalized calibration capture.
Official MXFP4 expert matrices are streamed from the source checkpoint; the
official model is never instantiated.  Each layer is written atomically, and
a canonical all-expert layer is considered resumable only when its payload,
metrics, and selection ledger are all present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from qsrt import constants as C
from qsrt.all_row_capture import load_all_row_capture, map_layer_rows
from qsrt.capture import (
    LayerSampleIndex,
    LayerSampleCacheIndex,
    index_cached_layer_samples,
    index_layer_samples,
    load_capture,
    load_layer_hessians,
    load_layer_samples,
)
from qsrt.io.safetensors_stream import AtomicSafetensorsWriter, TensorSpec
from qsrt.exl3_reference import (
    CODEBOOK_SQG_XOR_CHEB_T12,
    QSRT_CODEBOOKS,
)
from qsrt.sqg_quantizer import install_sqg_quantizer
from qsrt.qsrt_candidates import (
    PERMUTATION_POLICIES,
    PHASE1_BOOTSTRAP_REPLICATES,
    PHASE1_MIN_CONFIRMATION_DOCUMENTS,
    PHASE1_MIN_FIT_DOCUMENTS,
    RequestPartition,
    layer_fallback_contexts,
    partition_requests,
    request_documents,
)
from qsrt.qsrt_rotations import (
    QSRTRotationPlan,
    QSRTLayerRotationPlan,
    load_qsrt_rotation_plan,
)
from qsrt.qsrt_coupled_plan import (
    CoupledDrawSelection,
    CoupledRotationPlan,
    POOLED_ALL_ROW_SELECTION,
    PRODUCTION_SELECTION,
    load_coupled_rotation_plan,
    select_coupled_draw,
)
from qsrt.coupled_expert_study import CoupledTriplet
from qsrt.pooled_calibration import evaluate_coupled_candidate_portfolio_batches
from qsrt.pooled_selection import PooledCandidateScore, select_pooled_candidate
from qsrt.qsrt_coupled import CoupledHadamardExecution, CoupledHadamardSpec
from qsrt.qsrt import (
    FIXED_HIGH_RATE_TRELLIS_BYTES,
    H308,
    K2,
    INTERMEDIATE_CHANNELS,
    LATENT_CHANNELS,
    LOGICAL_CANDIDATE_SCHEMAS,
    MATRIX_TRELLIS_BYTES,
    PURE_K2_MATRIX_TRELLIS_BYTES,
    PHASE1_H2_EXPERT_LOCAL_ALPHA,
    PHASE1_H2_LOCAL_BASIS,
    PHASE1_H2_SHRINKAGE_POLICY,
    PHASE1_MODE_IDS,
    R0,
    SCHEMA as QSRT_SCHEMA,
    resolve_mode,
)
from qsrt.pack.qsrt_candidates import (
    CANDIDATE_POOL_KIND,
    CANDIDATE_POOL_SCHEMA_VERSION,
    DOWN_TARGET_POLICIES,
    HESSIAN_POLICIES,
    OFFICIAL_SOURCE_DAMAGE_METRIC,
    candidate_tensor_name,
    encode_phase1_expert_batch,
    selected_candidate_tensors,
)
from qsrt.pack.qsrt_encoder import (
    MATRICES,
    MIXED_SEARCH_LAYOUTS,
    qsrt_transform_seed_draw,
)
from qsrt.source_weights import OfficialMXFP4Store


DEFAULT_PARALLEL_WORKERS = 12
SCHEMA_VERSION = CANDIDATE_POOL_SCHEMA_VERSION
KIND = CANDIDATE_POOL_KIND


@dataclass
class _WorkerResources:
    report: dict
    partition: RequestPartition
    sample_index: LayerSampleIndex | LayerSampleCacheIndex
    store: OfficialMXFP4Store
    quantizer_module: object


@dataclass(frozen=True)
class _ScheduleJob:
    layer: int
    experts: tuple[int, ...]
    estimated_work: int

    def __post_init__(self) -> None:
        if self.layer not in C.MOE_LAYERS:
            raise ValueError("scheduled job has an invalid layer")
        if (
            not self.experts
            or tuple(sorted(self.experts)) != self.experts
            or len(set(self.experts)) != len(self.experts)
            or any(not 0 <= expert < C.NUM_EXPERTS for expert in self.experts)
        ):
            raise ValueError("scheduled job has invalid expert IDs")
        if self.estimated_work <= 0:
            raise ValueError("scheduled job must have positive work")

    def to_json(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "experts": list(self.experts),
            "estimated_work": self.estimated_work,
        }

    @classmethod
    def from_json(cls, value: object) -> "_ScheduleJob":
        if not isinstance(value, dict):
            raise ValueError("scheduled job must be an object")
        experts = value.get("experts")
        if not isinstance(experts, list) or any(
            isinstance(expert, bool) or not isinstance(expert, int)
            for expert in experts
        ):
            raise ValueError("scheduled job experts must be integers")
        layer = value.get("layer")
        work = value.get("estimated_work")
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or isinstance(work, bool)
            or not isinstance(work, int)
        ):
            raise ValueError("scheduled job layer/work must be integers")
        return cls(layer, tuple(experts), work)


def _reset_cuda_after_failure(device: torch.device | None = None) -> None:
    try:
        torch.cuda.synchronize(device)
    except Exception:
        traceback.print_exc()
    try:
        torch.cuda.empty_cache()
    except Exception:
        traceback.print_exc()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _configured_rotation_plan(args: argparse.Namespace) -> QSRTRotationPlan:
    cached = getattr(args, "_rotation_plan_data", None)
    if cached is not None:
        return cached
    path = getattr(args, "rotation_plan", None)
    plan = QSRTRotationPlan() if path is None else load_qsrt_rotation_plan(path)
    setattr(args, "_rotation_plan_data", plan)
    return plan


def _layer_rotation_plan(
    args: argparse.Namespace, layer: int
) -> QSRTLayerRotationPlan:
    if getattr(args, "rotation_plan", None) is not None:
        return _configured_rotation_plan(args).for_layer(layer)
    return QSRTLayerRotationPlan(
        residual_draw=int(getattr(args, "residual_rotation_draw", 0)),
        intermediate_default=int(
            getattr(args, "intermediate_rotation_draw", 0)
        ),
    )


def _rotation_manifest_contract(args: argparse.Namespace) -> dict[str, object]:
    if getattr(args, "rotation_plan", None) is None:
        return {
            "source": "fixed_cli_draws",
            "fixed": _layer_rotation_plan(args, C.MOE_LAYERS[0]).to_json(),
        }
    return {
        "source": "model_rotation_plan",
        "plan": _configured_rotation_plan(args).to_json(),
    }


def _configured_coupled_plan(
    args: argparse.Namespace,
) -> CoupledRotationPlan | None:
    path = getattr(args, "coupled_rotation_plan", None)
    if path is None:
        return None
    cached = getattr(args, "_coupled_rotation_plan_data", None)
    if cached is None:
        cached = load_coupled_rotation_plan(path)
        setattr(args, "_coupled_rotation_plan_data", cached)
    return cached


def _configured_coupled_draw_candidates(
    args: argparse.Namespace,
) -> tuple[int, ...] | None:
    draws = getattr(args, "coupled_draw_candidates", None)
    if draws is None:
        return None
    return tuple(int(draw) for draw in draws)


def _coupled_manifest_contract(
    args: argparse.Namespace,
) -> dict[str, object] | None:
    plan = _configured_coupled_plan(args)
    draws = _configured_coupled_draw_candidates(args)
    if plan is not None:
        return {
            "source": "model_rotation_plan",
            "plan": plan.to_json(),
        }
    if draws is not None:
        pooled_capture = getattr(args, "pooled_selection_capture", None)
        if pooled_capture is not None:
            manifest_path = pooled_capture / "manifest.json"
            manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            return {
                "source": "candidate_pool_selection",
                "selection": POOLED_ALL_ROW_SELECTION,
                "draw_candidates": list(draws),
                "capture": str(pooled_capture.resolve()),
                "capture_manifest_sha256": manifest_sha256,
                "population": "every_naturally_routed_captured_occurrence",
                "metric": "whole_expert_post_projection_sse",
                "router_weighting": "applied_gate_squared_once",
            }
        return {
            "source": "candidate_pool_selection",
            "selection": PRODUCTION_SELECTION,
            "draw_candidates": list(draws),
            "proposal_fold": "fit",
            "proposal_metric": "total_post_projection_sse",
            "acceptance_fold": "confirmation",
            "acceptance_metric": "total_post_projection_sse_vs_draw_zero",
        }
    return None


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=1, sort_keys=True))
    temporary.replace(path)


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file(tensors, str(temporary))
    temporary.replace(path)


def _candidate_payload_specs(layer: int, experts: list[int]) -> tuple[TensorSpec, ...]:
    return _candidate_payload_specs_with_trellis_bytes(
        layer, experts, MATRIX_TRELLIS_BYTES
    )


def _candidate_payload_specs_for_modes(
    layer: int, experts: list[int], mode_ids: tuple[int, ...]
) -> tuple[TensorSpec, ...]:
    if mode_ids == (K2.mode_id,):
        return _candidate_payload_specs_with_trellis_bytes(
            layer, experts, PURE_K2_MATRIX_TRELLIS_BYTES
        )
    if mode_ids != (H308.mode_id,):
        return _candidate_payload_specs(layer, experts)
    return _candidate_payload_specs_with_trellis_bytes(
        layer, experts, FIXED_HIGH_RATE_TRELLIS_BYTES
    )


def _candidate_payload_specs_with_trellis_bytes(
    layer: int, experts: list[int], trellis_bytes: int
) -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = []
    for expert in experts:
        for matrix in C.EXPERT_MATRICES:
            specs.append(
                TensorSpec(
                    candidate_tensor_name(layer, expert, matrix, "trellis"),
                    torch.int16,
                    (trellis_bytes // torch.int16.itemsize,),
                )
            )
            suh = INTERMEDIATE_CHANNELS if matrix == "w2" else LATENT_CHANNELS
            svh = LATENT_CHANNELS if matrix == "w2" else INTERMEDIATE_CHANNELS
            specs.extend(
                (
                    TensorSpec(
                        candidate_tensor_name(layer, expert, matrix, "suh"),
                        torch.float16,
                        (suh,),
                    ),
                    TensorSpec(
                        candidate_tensor_name(layer, expert, matrix, "svh"),
                        torch.float16,
                        (svh,),
                    ),
                )
            )
    return tuple(specs)


def _parse_ints(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            begin, end = map(int, item.split("-", 1))
            result.extend(range(begin, end + 1))
        else:
            result.append(int(item))
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("expected unique comma-separated integers/ranges")
    return tuple(result)


def _parse_modes(value: str) -> tuple[int, ...]:
    modes = _parse_ints(value)
    if modes in ((H308.mode_id,), (K2.mode_id,)):
        return modes
    if modes[0] != 0 or any(mode not in (0, 1, 2) for mode in modes):
        raise argparse.ArgumentTypeError(
            "mode IDs must be H308 (3) or K2 (4) alone, or begin with R0"
        )
    return modes


def _format_label(r13: int, r2: int) -> str:
    if r13 == r2 and r13 in (H308.mode_id, K2.mode_id):
        return resolve_mode(r13).name
    return f"R{r13}/R{r2}"


def _selection_contract(args: argparse.Namespace) -> dict[str, object]:
    fixed_k3 = args.mode_ids == (R0.mode_id,)
    fixed_high_rate = args.mode_ids == (H308.mode_id,)
    fixed_k2 = args.mode_ids == (K2.mode_id,)
    if fixed_k3 or fixed_high_rate or fixed_k2:
        fixed_mode = H308 if fixed_high_rate else (K2 if fixed_k2 else R0)
        return {
            "mode_ids": [fixed_mode.mode_id],
            "format_grid": f"fixed_{fixed_mode.name.lower()}",
            "shared_r": True,
            "candidate_construction_fold": "fit",
            "mode_selection_fold": "none",
            "mode_proposal_metric": "none",
            "mode_acceptance": (
                "fixed_22k3_2k4_high_rate_profile"
                if fixed_high_rate
                else (
                    "fixed_uniform_k2_profile"
                    if fixed_k2
                    else "fixed_uniform_k3_profile"
                )
            ),
            "external_validation_used": False,
        }
    return {
        "mode_ids": list(args.mode_ids),
        "format_grid": "cartesian_r13_r2",
        "shared_r": False,
        "candidate_construction_fold": "fit",
        "mode_selection_fold": "confirmation",
        "mode_proposal_metric": "confirmation_routed_functional_sse",
        "mode_acceptance": "paired_document_bootstrap_lower_bound_vs_r0",
        "external_validation_used": False,
    }


def _h2_contract(args: argparse.Namespace) -> dict[str, object]:
    fixed_k3 = args.mode_ids == (R0.mode_id,)
    fixed_high_rate = args.mode_ids == (H308.mode_id,)
    fixed_k2 = args.mode_ids == (K2.mode_id,)
    fixed_profile = fixed_k3 or fixed_high_rate or fixed_k2
    return {
        "basis": PHASE1_H2_LOCAL_BASIS,
        "indexed_by": (
            "expert_upstream_profile" if fixed_profile else "expert_r13"
        ),
        "shrinkage_policy": PHASE1_H2_SHRINKAGE_POLICY,
        "maximum_local_alpha": PHASE1_H2_EXPERT_LOCAL_ALPHA,
        "prior": "expert_local_trace_scaled_identity",
        "unsupported_expert_fallback": "identity",
        "router_weighting": "applied_gate_squared",
        "down_candidate_grid": (
            "one_decoded_upstream_conditioned_h308"
            if fixed_high_rate
            else (
                "one_decoded_upstream_conditioned_k2"
                if fixed_k2
                else (
                    "one_decoded_upstream_conditioned_k3"
                    if fixed_k3
                    else "w2_r13_r2"
                )
            )
        ),
    }


def _down_target_contract(args: argparse.Namespace) -> dict[str, object]:
    policy = getattr(args, "down_target_policy", "source")
    if policy == "source":
        return {"policy": "source_down_projection"}
    return {
        "policy": "candidate_hidden_regularized_functional_refit",
        "indexed_by": "expert_upstream_profile",
        "objective": "applied_gate_squared_routed_expert_output_sse",
        "prior": "source_down_projection",
        "regularization_ratio": float(args.w2_refit_regularization_ratio),
        "candidate_construction_population": (
            "all_available_routed_rows"
            if args.mode_ids in (
                (R0.mode_id,),
                (H308.mode_id,),
                (K2.mode_id,),
            )
            else "fit_partition"
        ),
        "serialized_format_change": False,
    }


def _layer_stem(layer: int, experts: list[int]) -> str:
    if experts == list(range(C.NUM_EXPERTS)):
        return f"qsrt-layer-{layer:05d}"
    digest = __import__("hashlib").blake2b(
        ",".join(map(str, experts)).encode(), digest_size=6
    ).hexdigest()
    return f"qsrt-layer-{layer:05d}-subset-{digest}"


def _job_complete(candidate_dir: Path, job: _ScheduleJob) -> bool:
    stem = _layer_stem(job.layer, list(job.experts))
    required = (
        candidate_dir / f"{stem}.safetensors",
        candidate_dir / f"{stem}.metrics.safetensors",
        candidate_dir / f"{stem}.selection.json",
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def _layer_complete(candidate_dir: Path, layer: int) -> bool:
    return _job_complete(
        candidate_dir,
        _ScheduleJob(layer, tuple(range(C.NUM_EXPERTS)), 1),
    )


def _full_grid_eligibility_by_layer(
    capture: Path,
    partition: RequestPartition,
    *,
    min_fit_documents: int,
    min_confirmation_documents: int,
) -> dict[int, tuple[bool, ...]]:
    """Identify experts eligible for the full grid without loading activations.

    Scheduling needs only document/expert coverage.  Read the three compact
    route tensors from rank zero instead of materializing the captured latent
    and intermediate values in the parent process.
    """

    _root, geometry, ranks = load_capture(capture, load_stats=False)
    if geometry.num_layers != len(C.MOE_LAYERS):
        raise ValueError("capture layer count does not match K3")
    if geometry.num_experts != C.NUM_EXPERTS:
        raise ValueError("capture expert count does not match K3")
    rank_zero = next((rank for rank in ranks if rank.rank == 0), None)
    if rank_zero is None:
        raise ValueError("capture has no rank-zero routing samples")

    fit_steps = tuple(sorted(map(int, partition.fit)))
    confirmation_steps = tuple(sorted(map(int, partition.confirmation)))
    all_steps = (*fit_steps, *confirmation_steps)
    step_to_slot = torch.full(
        (max(all_steps) + 1,), -1, dtype=torch.int64
    )
    for slot, step in enumerate(all_steps):
        step_to_slot[step] = slot
    seen = torch.zeros(
        (geometry.num_layers, len(all_steps), geometry.num_experts),
        dtype=torch.bool,
    )
    seen_flat = seen.view(-1)

    required = ("input.layer", "input.observation", "input.experts")
    for part in rank_zero.sample_parts():
        with safe_open(str(part), framework="pt", device="cpu") as handle:
            missing = [key for key in required if key not in handle.keys()]
            if missing:
                raise ValueError(f"sample part {part} is missing {missing}")
            layers = handle.get_tensor("input.layer").to(torch.int64)
            observations = handle.get_tensor("input.observation").to(torch.int64)
            experts = handle.get_tensor("input.experts").to(torch.int64)
        if (
            layers.ndim != 1
            or observations.ndim != 1
            or experts.ndim != 2
            or layers.numel() != observations.numel()
            or experts.shape[0] != layers.numel()
            or experts.shape[1] != geometry.top_k
        ):
            raise ValueError(f"sample part {part} has invalid route metadata")
        steps = torch.bitwise_right_shift(observations, 32)
        valid = (
            (layers >= 0)
            & (layers < geometry.num_layers)
            & (steps >= 0)
            & (steps < step_to_slot.numel())
        )
        slots = torch.full_like(steps, -1)
        slots[valid] = step_to_slot[steps[valid]]
        valid &= slots >= 0
        if not bool(valid.any()):
            continue
        selected_layers = layers[valid]
        selected_slots = slots[valid]
        selected_experts = experts[valid]
        if selected_experts.numel() and (
            int(selected_experts.min()) < 0
            or int(selected_experts.max()) >= geometry.num_experts
        ):
            raise ValueError(f"sample part {part} has invalid expert IDs")
        bases = (
            (selected_layers * len(all_steps) + selected_slots)
            * geometry.num_experts
        ).unsqueeze(1)
        seen_flat[(bases + selected_experts).reshape(-1)] = True

    fit_count = seen[:, : len(fit_steps)].sum(dim=1)
    confirmation_count = seen[:, len(fit_steps) :].sum(dim=1)
    eligible = (fit_count >= min_fit_documents) & (
        confirmation_count >= min_confirmation_documents
    )
    return {
        layer: tuple(bool(value) for value in eligible[layer - 1].tolist())
        for layer in C.MOE_LAYERS
    }


def _full_grid_experts_by_layer(
    capture: Path,
    partition: RequestPartition,
    *,
    min_fit_documents: int,
    min_confirmation_documents: int,
) -> dict[int, int]:
    """Compatibility wrapper returning only per-layer eligibility counts."""

    eligibility = _full_grid_eligibility_by_layer(
        capture,
        partition,
        min_fit_documents=min_fit_documents,
        min_confirmation_documents=min_confirmation_documents,
    )
    return {layer: sum(flags) for layer, flags in eligibility.items()}


def _balanced_layer_schedule(
    layer_costs: dict[int, int],
    layers: list[int],
    *,
    workers: int = DEFAULT_PARALLEL_WORKERS,
) -> tuple[tuple[int, ...], ...]:
    """Greedily LPT-pack indivisible layer outputs across GPU workers."""

    if workers < 1:
        raise ValueError("workers must be positive")
    if len(set(layers)) != len(layers) or any(layer not in layer_costs for layer in layers):
        raise ValueError("scheduled layers must be unique and have costs")
    assignments: list[list[int]] = [[] for _ in range(workers)]
    loads = [0] * workers
    for layer in sorted(layers, key=lambda value: (-layer_costs[value], value)):
        worker = min(
            range(workers),
            key=lambda value: (loads[value], len(assignments[value]), value),
        )
        assignments[worker].append(layer)
        loads[worker] += int(layer_costs[layer])
    return tuple(tuple(sorted(assignment)) for assignment in assignments)


def _balanced_expert_schedule(
    expert_costs: dict[int, tuple[int, ...]],
    layers: list[int],
    *,
    workers: int = DEFAULT_PARALLEL_WORKERS,
    expert_batch_size: int = 20,
    max_splits: int | None = None,
) -> tuple[tuple[_ScheduleJob, ...], ...]:
    """LPT-pack layers, then split a few boundary-aligned expert ranges.

    Ninety-two indivisible layers leave eight workers with eight layers and
    four workers with seven.  The resulting ~4% tail is enough to break the
    six-hour search budget.  Moving a contiguous, batch-aligned prefix or
    suffix from a heavy worker to a light one retains large encoder batches
    while requiring only a handful of layers to be merged after encoding.
    """

    if workers < 1 or expert_batch_size < 1:
        raise ValueError("workers and expert batch size must be positive")
    if len(set(layers)) != len(layers):
        raise ValueError("scheduled layers must be unique")
    expected_experts = tuple(range(C.NUM_EXPERTS))
    for layer in layers:
        costs = expert_costs.get(layer)
        if (
            costs is None
            or len(costs) != C.NUM_EXPERTS
            or any(cost <= 0 for cost in costs)
        ):
            raise ValueError("every scheduled layer needs positive per-expert costs")

    def job_work(layer: int, experts: tuple[int, ...], source_work: int) -> int:
        # A subset not containing expert zero performs one extra expert-zero
        # encode to reproduce the canonical layer-shared su seed.
        return source_work + (
            expert_costs[layer][0] if 0 not in experts else 0
        )

    layer_costs = {layer: sum(expert_costs[layer]) for layer in layers}
    layer_schedule = _balanced_layer_schedule(
        layer_costs, layers, workers=workers
    )
    assignments: list[list[_ScheduleJob]] = [
        [
            _ScheduleJob(layer, expected_experts, layer_costs[layer])
            for layer in worker_layers
        ]
        for worker_layers in layer_schedule
    ]
    loads = [sum(job.estimated_work for job in jobs) for jobs in assignments]
    # One 20-expert move is the useful granularity of the GPU batching.  Once
    # every worker is within one such worst-case unit of the mean, another
    # split buys less wall time than its extra layer setup and merge I/O.
    tolerance = expert_batch_size * max(
        cost for layer in layers for cost in expert_costs[layer]
    )
    split_limit = 2 * workers if max_splits is None else max_splits
    if split_limit < 0:
        raise ValueError("max_splits must be non-negative")

    for _ in range(split_limit):
        target = math.ceil(sum(loads) / workers)
        if max(loads) <= target + tolerance:
            break
        best = None
        current_mean = sum(loads) / workers
        current_square = sum((load - current_mean) ** 2 for load in loads)
        for donor in range(workers):
            if loads[donor] <= target:
                continue
            for receiver in range(workers):
                if donor == receiver or loads[receiver] >= target:
                    continue
                receiver_layers = {job.layer for job in assignments[receiver]}
                gap = loads[donor] - loads[receiver]
                for job_index, job in enumerate(assignments[donor]):
                    if job.layer in receiver_layers:
                        continue
                    count = len(job.experts)
                    # Split an original layer at most once.  Recursively
                    # fragmenting an already moved subset adds another
                    # expert-zero scale primer and creates avoidable merge
                    # shards for very little extra balance.
                    if count != C.NUM_EXPERTS or count < 2 * expert_batch_size:
                        continue
                    source_job_cost = sum(
                        expert_costs[job.layer][expert]
                        for expert in job.experts
                    )
                    prefix_cost = 0
                    for cut in range(1, count):
                        expert = job.experts[cut - 1]
                        prefix_cost += expert_costs[job.layer][expert]
                        if cut % expert_batch_size or count - cut < expert_batch_size:
                            continue
                        suffix_source_cost = source_job_cost - prefix_cost
                        prefix_job_cost = job_work(
                            job.layer, job.experts[:cut], prefix_cost
                        )
                        suffix_job_cost = job_work(
                            job.layer, job.experts[cut:], suffix_source_cost
                        )
                        for move_prefix, move_cost, retain_cost in (
                            (True, prefix_job_cost, suffix_job_cost),
                            (False, suffix_job_cost, prefix_job_cost),
                        ):
                            candidate_loads = list(loads)
                            candidate_loads[donor] += retain_cost - job.estimated_work
                            candidate_loads[receiver] += move_cost
                            candidate_mean = sum(candidate_loads) / workers
                            square = sum(
                                (load - candidate_mean) ** 2
                                for load in candidate_loads
                            )
                            if square >= current_square:
                                continue
                            score = (
                                square,
                                max(candidate_loads),
                                max(candidate_loads) - min(candidate_loads),
                                abs(move_cost * 2 - gap),
                                job.layer,
                                cut,
                                not move_prefix,
                                donor,
                                receiver,
                            )
                            if best is None or score < best[0]:
                                best = (
                                    score,
                                    donor,
                                    receiver,
                                    job_index,
                                    cut,
                                    move_prefix,
                                    prefix_job_cost,
                                    suffix_job_cost,
                                )
        if best is None:
            break
        (
            _score,
            donor,
            receiver,
            job_index,
            cut,
            move_prefix,
            prefix_job_cost,
            suffix_job_cost,
        ) = best
        job = assignments[donor][job_index]
        prefix = job.experts[:cut]
        suffix = job.experts[cut:]
        if move_prefix:
            moved = _ScheduleJob(job.layer, prefix, prefix_job_cost)
            retained = _ScheduleJob(job.layer, suffix, suffix_job_cost)
        else:
            moved = _ScheduleJob(job.layer, suffix, suffix_job_cost)
            retained = _ScheduleJob(job.layer, prefix, prefix_job_cost)
        assignments[donor][job_index] = retained
        assignments[receiver].append(moved)
        loads[donor] += retained.estimated_work - job.estimated_work
        loads[receiver] += moved.estimated_work

    schedule = tuple(
        tuple(sorted(jobs, key=lambda job: (job.layer, job.experts[0])))
        for jobs in assignments
    )
    coverage = {
        (job.layer, expert)
        for jobs in schedule
        for job in jobs
        for expert in job.experts
    }
    expected = {
        (layer, expert) for layer in layers for expert in expected_experts
    }
    if coverage != expected or sum(
        len(job.experts) for jobs in schedule for job in jobs
    ) != len(expected):
        raise AssertionError("expert-balanced schedule lost or duplicated work")
    for jobs in schedule:
        if len({job.layer for job in jobs}) != len(jobs):
            raise AssertionError("one worker received two jobs from the same layer")
    return schedule


def _schedule_report(
    schedule: tuple[tuple[_ScheduleJob, ...], ...],
    *,
    expert_batch_size: int,
    r0_work: int,
    full_grid_work: int,
    full_grid_experts_by_layer: dict[int, int],
) -> dict[str, object]:
    loads = [sum(job.estimated_work for job in jobs) for jobs in schedule]
    total = sum(loads)
    return {
        "schema_version": 2,
        "cost_unit": "equivalent_trellis_records",
        "expert_batch_size": expert_batch_size,
        "r0_expert_cost": r0_work,
        "full_grid_expert_cost": full_grid_work,
        "full_grid_experts_by_layer": {
            str(layer): full_grid_experts_by_layer[layer]
            for layer in C.MOE_LAYERS
        },
        "total_estimated_work": total,
        "mean_estimated_work": total / len(schedule),
        "max_estimated_work": max(loads),
        "max_over_mean": max(loads) / (total / len(schedule)),
        "split_layer_count": sum(
            count > 1
            for count in __import__("collections").Counter(
                job.layer for jobs in schedule for job in jobs
            ).values()
        ),
        "workers": [
            {
                "worker": worker,
                "layers": sorted({job.layer for job in jobs}),
                "estimated_work": loads[worker],
                "jobs": [job.to_json() for job in jobs],
            }
            for worker, jobs in enumerate(schedule)
        ],
    }


def _schedule_from_report(
    report: object,
    *,
    workers: int | None = None,
) -> tuple[tuple[_ScheduleJob, ...], ...]:
    if not isinstance(report, dict) or report.get("schema_version") != 2:
        raise ValueError("worker schedule must use schema version 2")
    worker_values = report.get("workers")
    if not isinstance(worker_values, list) or not worker_values:
        raise ValueError("worker schedule has no workers")
    if workers is not None and len(worker_values) != workers:
        raise ValueError("worker schedule has the wrong worker count")
    schedule: list[tuple[_ScheduleJob, ...]] = []
    for expected_worker, worker_value in enumerate(worker_values):
        if (
            not isinstance(worker_value, dict)
            or worker_value.get("worker") != expected_worker
            or not isinstance(worker_value.get("jobs"), list)
        ):
            raise ValueError("worker schedule has malformed worker entries")
        jobs = tuple(
            _ScheduleJob.from_json(value) for value in worker_value["jobs"]
        )
        if len({job.layer for job in jobs}) != len(jobs):
            raise ValueError("one worker has multiple jobs for one layer")
        if worker_value.get("estimated_work") != sum(
            job.estimated_work for job in jobs
        ):
            raise ValueError("worker schedule work accounting drifted")
        schedule.append(jobs)
    ownership: dict[tuple[int, int], int] = {}
    for worker, jobs in enumerate(schedule):
        for job in jobs:
            for expert in job.experts:
                key = (job.layer, expert)
                if key in ownership:
                    raise ValueError("worker schedule duplicates an expert assignment")
                ownership[key] = worker
    expected = {
        (layer, expert)
        for layer in C.MOE_LAYERS
        for expert in range(C.NUM_EXPERTS)
    }
    if set(ownership) != expected:
        raise ValueError("worker schedule does not cover the complete model")
    return tuple(schedule)


def _validate_training_contract(
    report_path: Path,
    *,
    capture: Path,
    teacher_checkpoint: Path,
) -> tuple[dict, object]:
    report = _read_json(report_path)
    reported_capture = Path(str(report.get("capture_dir", ""))).resolve()
    if reported_capture != capture.resolve():
        raise ValueError(f"training report capture {reported_capture} != {capture.resolve()}")
    reported_teacher = Path(str(report.get("model_dir", ""))).resolve()
    if reported_teacher != teacher_checkpoint.resolve():
        raise ValueError(
            f"training report teacher {reported_teacher} != {teacher_checkpoint.resolve()}"
        )
    partition = partition_requests(request_documents(report))
    return report, partition


def _validate_sample_cache_contract(sample_cache: Path, capture: Path) -> None:
    manifest = _read_json(sample_cache / "manifest.json")
    if manifest.get("kind") != "qsrt_layer_sample_cache":
        raise ValueError(f"{sample_cache} is not a layer sample cache")
    source = Path(str(manifest.get("source_capture", ""))).resolve()
    if source != capture.resolve():
        raise ValueError(f"sample cache capture {source} != {capture.resolve()}")


def _load_quantizer_module(root: Path, codebook: str = CODEBOOK_SQG_XOR_CHEB_T12):
    # Import only the encoder namespace.  The full exllamav3 package pulls in
    # serving/tokenizer dependencies that are intentionally absent here.
    from qsrt.exl3_loader import load_qsrt_encoder

    module = load_qsrt_encoder(root)
    if codebook in QSRT_CODEBOOKS:
        install_sqg_quantizer(module)
    return module


def _allocate_metrics(
    experts: list[int],
    modes: tuple[int, ...],
    fit_documents: int,
    confirmation_documents: int,
    *,
    coupled: bool = False,
    pooled_coupled: bool = False,
) -> dict[str, torch.Tensor]:
    count = len(experts)
    mode_count = len(modes)
    metrics = {
        "expert_ids": torch.tensor(experts, dtype=torch.int16),
        "mode_ids": torch.tensor(modes, dtype=torch.uint8),
        "selected_r13": torch.zeros(count, dtype=torch.uint8),
        "selected_r2": torch.zeros(count, dtype=torch.uint8),
        "proposed_r13": torch.zeros(count, dtype=torch.uint8),
        "proposed_r2": torch.zeros(count, dtype=torch.uint8),
        "format_evaluated": torch.zeros(
            (count, mode_count, mode_count), dtype=torch.bool
        ),
        "fit_sse": torch.zeros(
            (count, mode_count, mode_count, fit_documents), dtype=torch.float64
        ),
        "confirmation_sse": torch.zeros(
            (count, mode_count, mode_count, confirmation_documents),
            dtype=torch.float64,
        ),
        "fit_reference_energy": torch.zeros((count, fit_documents), dtype=torch.float64),
        "confirmation_reference_energy": torch.zeros(
            (count, confirmation_documents), dtype=torch.float64
        ),
        "fit_counts": torch.zeros((count, fit_documents), dtype=torch.int32),
        "confirmation_counts": torch.zeros(
            (count, confirmation_documents), dtype=torch.int32
        ),
        "block_contexts": torch.zeros((count, 768), dtype=torch.uint8),
        "block_scores": torch.zeros((count, 768), dtype=torch.float32),
        "confirmation_improvement": torch.full((count,), float("nan"), dtype=torch.float64),
        "confirmation_ci95": torch.full((count, 2), float("nan"), dtype=torch.float64),
        # This is already integrated over the sampled natural-routing rows and
        # includes the applied router gate squared.  It is therefore the
        # direct keep benefit for the codec objective; allocators must not
        # multiply it by route or gate mass a second time.
        OFFICIAL_SOURCE_DAMAGE_METRIC: torch.zeros(count, dtype=torch.float64),
    }
    if coupled:
        metrics.update(
            {
                "coupled_draw_evaluated": torch.zeros(
                    (count, 8), dtype=torch.bool
                ),
                "coupled_draw_fit_sse": torch.full(
                    (count, 8), float("nan"), dtype=torch.float64
                ),
                "coupled_draw_confirmation_sse": torch.full(
                    (count, 8), float("nan"), dtype=torch.float64
                ),
                "coupled_draw_proposed": torch.zeros(count, dtype=torch.uint8),
                "coupled_draw_selected": torch.zeros(count, dtype=torch.uint8),
                "coupled_draw_confirmation_improvement": torch.full(
                    (count,), float("nan"), dtype=torch.float64
                ),
            }
        )
    if pooled_coupled:
        metrics.update(
            {
                "coupled_draw_pooled_sse": torch.full(
                    (count, 8), float("nan"), dtype=torch.float64
                ),
                "coupled_draw_pooled_source_energy": torch.zeros(
                    count, dtype=torch.float64
                ),
                "coupled_draw_pooled_occurrences": torch.zeros(
                    count, dtype=torch.int64
                ),
            }
        )
    return metrics


def _store_metrics(
    metrics: dict[str, torch.Tensor],
    row: int,
    modes: tuple[int, ...],
    candidate,
    *,
    coupled_selection: CoupledDrawSelection | None = None,
    coupled_fit_sse: dict[int, float] | None = None,
    coupled_confirmation_sse: dict[int, float] | None = None,
    coupled_pooled_sse: dict[int, float] | None = None,
    coupled_pooled_source_energy: float | None = None,
    coupled_pooled_occurrences: int | None = None,
) -> None:
    selection = candidate.selection
    metrics["selected_r13"][row] = selection.selected_r13
    metrics["selected_r2"][row] = selection.selected_r2
    metrics["proposed_r13"][row] = selection.proposed_r13
    metrics["proposed_r2"][row] = selection.proposed_r2
    metrics["fit_reference_energy"][row].copy_(candidate.fit_reference_energy)
    metrics["confirmation_reference_energy"][row].copy_(
        candidate.confirmation_reference_energy
    )
    metrics["fit_counts"][row].copy_(candidate.fit_counts.to(torch.int32))
    metrics["confirmation_counts"][row].copy_(
        candidate.confirmation_counts.to(torch.int32)
    )
    metrics["block_contexts"][row].copy_(candidate.block_contexts)
    metrics["block_scores"][row].copy_(candidate.block_scores)
    mode_index = {mode: index for index, mode in enumerate(modes)}
    for rate_pair in candidate.evaluated_formats:
        r13_column = mode_index[rate_pair[0]]
        r2_column = mode_index[rate_pair[1]]
        metrics["format_evaluated"][row, r13_column, r2_column] = True
        metrics["fit_sse"][row, r13_column, r2_column].copy_(
            candidate.fit_sse[rate_pair]
        )
        metrics["confirmation_sse"][row, r13_column, r2_column].copy_(
            candidate.confirmation_sse[rate_pair]
        )
    if selection.confirmation_relative_improvement is not None:
        metrics["confirmation_improvement"][row] = (
            selection.confirmation_relative_improvement
        )
    for column, value in enumerate(selection.confirmation_ci95):
        if value is not None:
            metrics["confirmation_ci95"][row, column] = value
    selected_r13_column = mode_index[selection.selected_r13]
    selected_r2_column = mode_index[selection.selected_r2]
    official_source_excess_sse = (
        metrics["fit_sse"][row, selected_r13_column, selected_r2_column].sum()
        + metrics[
            "confirmation_sse"
        ][row, selected_r13_column, selected_r2_column].sum()
    )
    metrics[OFFICIAL_SOURCE_DAMAGE_METRIC][row] = official_source_excess_sse
    if coupled_selection is None:
        if any(
            value is not None
            for value in (
                coupled_fit_sse,
                coupled_confirmation_sse,
                coupled_pooled_sse,
                coupled_pooled_source_energy,
                coupled_pooled_occurrences,
            )
        ):
            raise ValueError("coupled draw SSE was provided without a selection")
        return
    if coupled_fit_sse is None or coupled_confirmation_sse is None:
        raise ValueError("coupled draw selection is missing its SSE evidence")
    for draw in coupled_selection.evaluated_draws:
        metrics["coupled_draw_evaluated"][row, draw] = True
        metrics["coupled_draw_fit_sse"][row, draw] = coupled_fit_sse[draw]
        metrics["coupled_draw_confirmation_sse"][row, draw] = (
            coupled_confirmation_sse[draw]
        )
    metrics["coupled_draw_proposed"][row] = coupled_selection.proposed_draw
    metrics["coupled_draw_selected"][row] = coupled_selection.selected_draw
    if coupled_selection.confirmation_relative_improvement is not None:
        metrics["coupled_draw_confirmation_improvement"][row] = (
            coupled_selection.confirmation_relative_improvement
        )
    if coupled_pooled_sse is not None:
        if coupled_pooled_source_energy is None or coupled_pooled_occurrences is None:
            raise ValueError("pooled coupled scores are missing source support")
        for draw in coupled_selection.evaluated_draws:
            metrics["coupled_draw_pooled_sse"][row, draw] = coupled_pooled_sse[draw]
        metrics["coupled_draw_pooled_source_energy"][row] = coupled_pooled_source_energy
        metrics["coupled_draw_pooled_occurrences"][row] = coupled_pooled_occurrences


def _load_source_triplet(
    store: OfficialMXFP4Store,
    *,
    layer: int,
    expert: int,
    device: torch.device,
) -> CoupledTriplet:
    return CoupledTriplet(
        *(
            store.load_matrix(layer, expert, matrix, device=device).float()
            for matrix in ("w1", "w3", "w2")
        )
    )


def _pooled_coupled_draw_scores(
    layer_rows,
    store: OfficialMXFP4Store,
    candidate_by_draw: dict[int, object],
    *,
    layer: int,
    expert: int,
    device: torch.device,
    batch_rows: int,
) -> tuple[dict[int, float], float, int]:
    source = _load_source_triplet(
        store, layer=layer, expert=expert, device=device
    )
    coordinates = {
        str(draw): CoupledTriplet(
            candidate.expert.matrices["w1"].reconstruction,
            candidate.expert.matrices["w3"].reconstruction,
            candidate.expert.matrices["w2"].reconstruction,
        )
        for draw, candidate in candidate_by_draw.items()
    }
    executions = {
        str(draw): CoupledHadamardExecution(
            hidden=source.hidden,
            intermediate=source.intermediate,
            spec=CoupledHadamardSpec(intermediate_draw=draw),
        )
        for draw in candidate_by_draw
    }
    evaluation = evaluate_coupled_candidate_portfolio_batches(
        layer_rows.expert_batches(
            expert, batch_rows=batch_rows, fields=("input",)
        ),
        source=source,
        candidate_coordinates=coordinates,
        executions=executions,
    )
    return (
        {int(draw): value for draw, value in evaluation.candidate_sse.items()},
        evaluation.source_energy,
        evaluation.routed_occurrences,
    )


def encode_layer(
    args: argparse.Namespace,
    layer: int,
    experts: list[int],
    *,
    resources: _WorkerResources | None = None,
) -> dict:
    candidate_dir = args.dest / "candidates"
    stem = _layer_stem(layer, experts)
    payload_path = candidate_dir / f"{stem}.safetensors"
    metrics_path = candidate_dir / f"{stem}.metrics.safetensors"
    ledger_path = candidate_dir / f"{stem}.selection.json"
    partial_path = AtomicSafetensorsWriter.partial_path(payload_path)
    if partial_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"abandoned or active partial output exists for layer {layer}; "
                "use --resume only after the original worker has stopped"
            )
        AtomicSafetensorsWriter.discard_stale_partial(payload_path)
    if any(path.exists() for path in (payload_path, metrics_path, ledger_path)):
        if experts == list(range(C.NUM_EXPERTS)) and _layer_complete(candidate_dir, layer):
            print(f"layer {layer}: complete, skip", flush=True)
            return {"layer": layer, "skipped": True}
        if not args.resume:
            raise FileExistsError(
                f"partial or subset output already exists for layer {layer}; "
                "use a fresh destination"
            )
        print(
            f"layer {layer} experts {experts[0]}-{experts[-1]}: "
            "discard incomplete job outputs",
            flush=True,
        )
        for path in (payload_path, metrics_path, ledger_path):
            path.unlink(missing_ok=True)

    if resources is None:
        report, partition = _validate_training_contract(
            args.training_report,
            capture=args.capture,
            teacher_checkpoint=args.teacher_checkpoint,
        )
        samples = load_layer_samples(args.capture, layer - 1)
        store = OfficialMXFP4Store(
            repo_dir=args.official_repo_dir,
            revision=args.official_revision,
        )
        quantizer_module = _load_quantizer_module(
            args.exllamav3_root, args.codebook
        )
    else:
        report = resources.report
        partition = resources.partition
        samples = resources.sample_index.pop(layer - 1)
        store = resources.store
        quantizer_module = resources.quantizer_module
    if args.hessian_policy == "identity":
        global_h13 = torch.eye(LATENT_CHANNELS, dtype=torch.float32)
        global_h2 = torch.eye(INTERMEDIATE_CHANNELS, dtype=torch.float32)
    else:
        global_h13, global_h2 = load_layer_hessians(args.hessians, layer)
    fallback_contexts, fallback_scores = layer_fallback_contexts(
        samples, partition.fit
    )
    device = torch.device("cuda", 0)
    global_h13 = global_h13.to(device=device, dtype=torch.float32)
    global_h2 = global_h2.to(device=device, dtype=torch.float32)
    pooled_coupled = (
        _coupled_manifest_contract(args) is not None
        and _coupled_manifest_contract(args).get("selection")
        == POOLED_ALL_ROW_SELECTION
    )
    pooled_layer_rows = None
    if pooled_coupled:
        mapped_started = time.time()
        pooled_layer_rows = map_layer_rows(
            args.pooled_selection_capture,
            layer,
            verify_hashes=False,
        )
        print(
            f"layer {layer}: mapped {pooled_layer_rows.rows:,} pooled-selection "
            f"rows in {time.time() - mapped_started:.1f}s",
            flush=True,
        )
    metric_options = (
        {
            "coupled": True,
            **({"pooled_coupled": True} if pooled_coupled else {}),
        }
        if _coupled_manifest_contract(args) is not None
        else {}
    )
    metrics = _allocate_metrics(
        experts,
        args.mode_ids,
        len(partition.fit),
        len(partition.confirmation),
        **metric_options,
    )
    selections: dict[str, dict] = {}
    shared_hidden: dict[str, torch.Tensor] = {}
    shared_scale_primer = experts[0] != 0
    primer_candidates = 0
    rotation_plan = _layer_rotation_plan(args, layer)
    coupled_plan = _configured_coupled_plan(args)
    coupled_draw_candidates = _configured_coupled_draw_candidates(args)
    started = time.time()
    with AtomicSafetensorsWriter(
        payload_path,
        _candidate_payload_specs_for_modes(layer, experts, args.mode_ids),
        metadata={
            "kind": KIND,
            "schema_version": str(SCHEMA_VERSION),
            "layer": str(layer),
            "codebook": args.codebook,
            "tailbite_context": str(args.tailbite_context),
        },
    ) as payload_writer:
        completed = 0
        batch_begin = 0
        primer_pending = shared_scale_primer
        while batch_begin < len(experts):
            if primer_pending and args.expert_batch_size == 1:
                batch_experts = [0]
                actual_experts: list[int] = []
            else:
                capacity = args.expert_batch_size - int(primer_pending)
                actual_experts = experts[batch_begin : batch_begin + capacity]
                batch_experts = (
                    [0, *actual_experts] if primer_pending else actual_experts
                )
            intermediate_draws = {
                expert: rotation_plan.intermediate_draw(expert)
                for expert in batch_experts
            }
            if coupled_plan is not None:
                coupled_draw_maps: list[dict[int, int] | None] = [
                    coupled_plan.for_experts(layer, batch_experts)
                ]
            elif coupled_draw_candidates is not None:
                coupled_draw_maps = [
                    {expert: draw for expert in batch_experts}
                    for draw in coupled_draw_candidates
                ]
            else:
                coupled_draw_maps = [None]
            candidate_sets = []
            for coupled_draw_map in coupled_draw_maps:
                draw = (
                    None
                    if coupled_draw_map is None
                    else next(iter(coupled_draw_map.values()))
                )
                for attempt in range(args.batch_retries + 1):
                    try:
                        draw_candidates = encode_phase1_expert_batch(
                            store,
                            samples,
                            layer=layer,
                            experts=batch_experts,
                            partition=partition,
                            global_h13=global_h13,
                            global_h2=global_h2,
                            fallback_block_contexts=fallback_contexts,
                            fallback_block_scores=fallback_scores,
                            device=device,
                            shared_scale_scope=(
                                KIND,
                                str(args.dest.resolve()),
                                layer,
                            ),
                            mode_ids=args.mode_ids,
                            min_fit_documents=args.min_fit_documents,
                            min_confirmation_documents=args.min_confirmation_documents,
                            bootstrap_replicates=args.bootstrap_replicates,
                            minimum_improvement=args.minimum_improvement,
                            quantizer_module=quantizer_module,
                            logical_trellis_schema=args.logical_trellis_schema,
                            codebook=args.codebook,
                            layout=args.layout,
                            ldlq_tf32=args.ldlq_tf32,
                            tailbite_context=args.tailbite_context,
                            hessian_policy=args.hessian_policy,
                            permutation_policy=args.permutation_policy,
                            folded_scale_power=args.folded_scale_power,
                            down_target_policy=args.down_target_policy,
                            w2_refit_regularization_ratio=(
                                args.w2_refit_regularization_ratio
                            ),
                            transform_seeds_by_expert=(
                                None
                                if not (
                                    rotation_plan.residual_draw
                                    or any(intermediate_draws.values())
                                )
                                else {
                                    expert: {
                                        matrix: qsrt_transform_seed_draw(
                                            layer,
                                            matrix,
                                            residual_draw=rotation_plan.residual_draw,
                                            intermediate_draw=intermediate_draws[expert],
                                            expert=expert,
                                        )
                                        for matrix in MATRICES
                                    }
                                    for expert in batch_experts
                                }
                            ),
                            coupled_intermediate_draws_by_expert=coupled_draw_map,
                        )
                        break
                    except Exception as exc:
                        if attempt >= args.batch_retries:
                            raise RuntimeError(
                                f"layer {layer}, experts {batch_experts[0]}-"
                                f"{batch_experts[-1]}, coupled draw {draw} failed "
                                f"after {attempt + 1} attempts"
                            ) from exc
                        print(
                            f"layer {layer}, experts {batch_experts[0]}-"
                            f"{batch_experts[-1]}, coupled draw {draw}: "
                            f"attempt {attempt + 1} failed; retrying",
                            flush=True,
                        )
                        traceback.print_exc()
                        _reset_cuda_after_failure(device)
                if len(draw_candidates) != len(batch_experts):
                    raise ValueError(
                        "expert batch returned the wrong candidate count"
                    )
                candidate_sets.append((coupled_draw_map, draw_candidates))
            if primer_pending:
                # Shared-su is historically defined by expert zero, the first
                # source in a canonical full-layer encode.  A split worker
                # repeats that one encode only to seed the process-local EXL
                # scale cache, then discards its payload.  Keeping the primer
                # in the same first batch also preserves the normal batch-20
                # launch geometry.
                candidate_sets = [
                    (draw_map, candidates[1:])
                    for draw_map, candidates in candidate_sets
                ]
                primer_candidates += len(candidate_sets)
                primer_pending = False
            if any(
                len(candidates) != len(actual_experts)
                for _draw_map, candidates in candidate_sets
            ):
                raise ValueError("shared-scale primer candidate accounting drifted")
            for batch_row, expert in enumerate(actual_experts):
                row = batch_begin + batch_row
                coupled_selection: CoupledDrawSelection | None = None
                coupled_fit_sse: dict[int, float] | None = None
                coupled_confirmation_sse: dict[int, float] | None = None
                coupled_pooled_sse: dict[int, float] | None = None
                coupled_pooled_source_energy: float | None = None
                coupled_pooled_occurrences: int | None = None
                if coupled_plan is not None or coupled_draw_candidates is not None:
                    candidate_by_draw = {
                        int(draw_map[expert]): candidates[batch_row]
                        for draw_map, candidates in candidate_sets
                        if draw_map is not None
                    }
                    uniform_pair = (args.mode_ids[0], args.mode_ids[0])
                    coupled_fit_sse = {
                        draw: float(candidate.fit_sse[uniform_pair].sum())
                        for draw, candidate in candidate_by_draw.items()
                    }
                    coupled_confirmation_sse = {
                        draw: float(
                            candidate.confirmation_sse[uniform_pair].sum()
                        )
                        for draw, candidate in candidate_by_draw.items()
                    }
                    reference_candidate = next(iter(candidate_by_draw.values()))
                    fit_documents = int(
                        torch.count_nonzero(reference_candidate.fit_counts)
                    )
                    confirmation_documents = int(
                        torch.count_nonzero(reference_candidate.confirmation_counts)
                    )
                    if coupled_draw_candidates is not None:
                        if pooled_coupled:
                            if pooled_layer_rows is None:
                                raise AssertionError(
                                    "pooled selection lacks mapped layer rows"
                                )
                            pooled_started = time.monotonic()
                            (
                                coupled_pooled_sse,
                                coupled_pooled_source_energy,
                                coupled_pooled_occurrences,
                            ) = _pooled_coupled_draw_scores(
                                pooled_layer_rows,
                                store,
                                candidate_by_draw,
                                layer=layer,
                                expert=expert,
                                device=device,
                                batch_rows=args.pooled_selection_batch_rows,
                            )
                            print(
                                f"layer {layer}, expert {expert}: scored "
                                f"{len(coupled_draw_candidates)} coupled draws over "
                                f"{coupled_pooled_occurrences} routed occurrences in "
                                f"{time.monotonic() - pooled_started:.1f}s",
                                flush=True,
                            )
                            pooled_selection = select_pooled_candidate(
                                PooledCandidateScore(
                                    name=str(draw),
                                    sse=coupled_pooled_sse[draw],
                                    source_energy=coupled_pooled_source_energy,
                                    metadata_bytes=0,
                                )
                                for draw in coupled_draw_candidates
                            )
                            selected_draw = int(pooled_selection.selected.name)
                            coupled_selection = CoupledDrawSelection(
                                evaluated_draws=coupled_draw_candidates,
                                proposed_draw=selected_draw,
                                selected_draw=selected_draw,
                                fit_documents=fit_documents,
                                confirmation_documents=confirmation_documents,
                                confirmation_relative_improvement=None,
                                accepted=selected_draw != 0,
                                reason="pooled_all_routed_rows_minimum_sse",
                            )
                        else:
                            coupled_selection = select_coupled_draw(
                                coupled_draw_candidates,
                                coupled_fit_sse,
                                coupled_confirmation_sse,
                                fit_documents=fit_documents,
                                confirmation_documents=confirmation_documents,
                                min_fit_documents=args.min_fit_documents,
                                min_confirmation_documents=(
                                    args.min_confirmation_documents
                                ),
                                minimum_improvement=args.minimum_improvement,
                            )
                    else:
                        selected_draw = next(iter(candidate_by_draw))
                        coupled_selection = CoupledDrawSelection(
                            evaluated_draws=(selected_draw,),
                            proposed_draw=selected_draw,
                            selected_draw=selected_draw,
                            fit_documents=fit_documents,
                            confirmation_documents=confirmation_documents,
                            confirmation_relative_improvement=None,
                            accepted=False,
                            reason="fixed_model_rotation_plan",
                        )
                    candidate = candidate_by_draw[coupled_selection.selected_draw]
                else:
                    candidate = candidate_sets[0][1][batch_row]
                try:
                    selected_payload = selected_candidate_tensors(
                        layer, expert, candidate
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"layer {layer} expert {expert} payload closure failed"
                    ) from exc
                payload_writer.write_many(selected_payload)
                del selected_payload
                _store_metrics(
                    metrics,
                    row,
                    args.mode_ids,
                    candidate,
                    coupled_selection=coupled_selection,
                    coupled_fit_sse=coupled_fit_sse,
                    coupled_confirmation_sse=coupled_confirmation_sse,
                    coupled_pooled_sse=coupled_pooled_sse,
                    coupled_pooled_source_energy=coupled_pooled_source_energy,
                    coupled_pooled_occurrences=coupled_pooled_occurrences,
                )
                for matrix, part in (
                    ("w1", "suh"),
                    ("w3", "suh"),
                    ("w2", "svh"),
                ):
                    value = (
                        candidate.expert.matrices[matrix]
                        .tensors[part]
                        .detach()
                        .cpu()
                    )
                    key = f"{matrix}.{part}"
                    if key not in shared_hidden:
                        shared_hidden[key] = value.clone()
                    elif not torch.equal(value, shared_hidden[key]):
                        reference = shared_hidden[key]
                        difference = (value.float() - reference.float()).abs()
                        raise ValueError(
                            f"layer {layer} expert {expert} {key} is not shared across experts; "
                            f"mismatched={int(torch.count_nonzero(value != reference))}/"
                            f"{value.numel()}, max_abs={float(difference.max())}, "
                            f"reference_mean={float(reference.float().mean())}, "
                            f"actual_mean={float(value.float().mean())}"
                        )
                selections[str(expert)] = candidate.metadata()
                if coupled_selection is not None:
                    selections[str(expert)]["coupled_rotation"] = {
                        **coupled_selection.to_json(),
                        "fit_post_projection_sse": {
                            str(draw): coupled_fit_sse[draw]
                            for draw in coupled_selection.evaluated_draws
                        },
                        "confirmation_post_projection_sse": {
                            str(draw): coupled_confirmation_sse[draw]
                            for draw in coupled_selection.evaluated_draws
                        },
                        **(
                            {
                                "pooled_post_projection_sse": {
                                    str(draw): coupled_pooled_sse[draw]
                                    for draw in coupled_selection.evaluated_draws
                                },
                                "pooled_source_energy": (
                                    coupled_pooled_source_energy
                                ),
                                "pooled_routed_occurrences": (
                                    coupled_pooled_occurrences
                                ),
                            }
                            if coupled_pooled_sse is not None
                            else {}
                        ),
                    }
                completed += 1
                elapsed = time.time() - started
                print(
                    f"layer {layer} expert {expert}: {completed}/{len(experts)} "
                    f"({elapsed / completed:.1f}s/expert)",
                    flush=True,
                )
            del candidate_sets
            torch.cuda.empty_cache()
            batch_begin += len(actual_experts)

    _atomic_safetensors(metrics_path, metrics)
    selected_histogram = {
        _format_label(r13, r2): int(
            (
                (metrics["selected_r13"] == r13)
                & (metrics["selected_r2"] == r2)
            ).sum()
        )
        for r13 in args.mode_ids
        for r2 in args.mode_ids
    }
    ledger = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "layer": layer,
        "experts": experts,
        "all_experts": experts == list(range(C.NUM_EXPERTS)),
        "source_checkpoint": str(store.root),
        "source_revision": store.revision,
        "resident_teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "capture": str(args.capture.resolve()),
        "training_report": str(args.training_report.resolve()),
        "hessians": str(args.hessians.resolve()),
        "hessian_policy": args.hessian_policy,
        "permutation_policy": getattr(args, "permutation_policy", "h2_reverse"),
        "folded_scale_power": float(getattr(args, "folded_scale_power", 0.0)),
        "h2_contract": _h2_contract(args),
        "down_target_contract": _down_target_contract(args),
        "selection_contract": {
            **_selection_contract(args),
            "fit_documents": len(partition.fit),
            "confirmation_documents": len(partition.confirmation),
            "min_fit_documents": args.min_fit_documents,
            "min_confirmation_documents": args.min_confirmation_documents,
            "minimum_improvement": args.minimum_improvement,
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        "damage_contract": {
            "metric": OFFICIAL_SOURCE_DAMAGE_METRIC,
            "reference": "decoded official MXFP4 source expert output",
            "candidate": "selected coupled w1/w3/w2 QSRT reconstruction",
            "population": "natural-routing token-uniform captured input rows",
            "router_weighting": "applied gate squared inside functional SSE",
            "keep_benefit": "equal to this value; do not apply traffic weighting again",
        },
        "source_residency": (
            "complete_expert_reencode_per_coupled_draw"
            if _configured_coupled_draw_candidates(args)
            else (
                "w1_w3_once_then_candidate_conditioned_w2_target_per_r13"
                if args.down_target_policy == "functional_ridge"
                else "w1_w3_once_then_w2_reused_across_conditional_r13_grid"
            )
        ),
        "payload_write": "atomic_bounded_memory_safetensors_stream",
        "logical_trellis_schema": args.logical_trellis_schema,
        "codebook": args.codebook,
        "layout": args.layout,
        "ldlq_tf32": args.ldlq_tf32,
        "tailbite_context": args.tailbite_context,
        "permutation_policy": args.permutation_policy,
        "folded_scale_power": args.folded_scale_power,
        "rotation_draws": {
            "residual_layer_shared": rotation_plan.residual_draw,
            "intermediate_expert_private_default": (
                rotation_plan.intermediate_default
            ),
            "intermediate_expert_private_overrides": {
                str(expert): draw_id
                for expert, draw_id in rotation_plan.intermediate_overrides
            },
        },
        "coupled_rotation": _coupled_manifest_contract(args),
        "selected_coupled_draw_histogram": (
            None
            if "coupled_draw_selected" not in metrics
            else {
                str(draw): int((metrics["coupled_draw_selected"] == draw).sum())
                for draw in range(8)
                if bool((metrics["coupled_draw_selected"] == draw).any())
            }
        ),
        "selected_format_histogram": selected_histogram,
        "shared_hidden_scale_closure": sorted(shared_hidden),
        "shared_scale_primer_expert": 0 if shared_scale_primer else None,
        "shared_scale_primer_candidates": primer_candidates,
        "payload": payload_path.name,
        "metrics": metrics_path.name,
        "seconds": time.time() - started,
        "selections": selections,
        "training_corpus_documents": len(report["documents"]),
    }
    _atomic_json(ledger_path, ledger)
    print(
        f"layer {layer}: wrote {len(experts)} candidates in {ledger['seconds']:.0f}s; "
        f"modes {selected_histogram}",
        flush=True,
    )
    return ledger


def _validate_canonical_payload(
    path: Path,
    *,
    layer: int,
    codebook: str,
    tailbite_context: int,
    mode_ids: tuple[int, ...],
) -> None:
    expected = {
        spec.name: (spec.dtype, spec.shape)
        for spec in _candidate_payload_specs_for_modes(
            layer, list(range(C.NUM_EXPERTS)), mode_ids
        )
    }
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected):
            raise ValueError(f"merged payload tensor inventory drifted: {path}")
        metadata = handle.metadata() or {}
        if metadata.get("codebook") != codebook:
            raise ValueError(f"merged payload codebook drifted: {path}")
        if metadata.get("tailbite_context") != str(tailbite_context):
            raise ValueError(f"merged payload tail-biting context drifted: {path}")
        for name, (dtype, shape) in expected.items():
            tensor = handle.get_slice(name)
            if tensor.get_dtype() != {
                torch.int16: "I16",
                torch.float16: "F16",
            }[dtype] or tuple(tensor.get_shape()) != shape:
                raise ValueError(f"merged payload tensor geometry drifted: {name}")


def _merge_scheduled_layer(
    args: argparse.Namespace,
    layer: int,
    jobs: list[_ScheduleJob],
    partition: RequestPartition,
) -> None:
    """Merge independently encoded expert ranges into one canonical layer."""

    candidate_dir = args.dest / "candidates"
    canonical_stem = _layer_stem(layer, list(range(C.NUM_EXPERTS)))
    payload_path = candidate_dir / f"{canonical_stem}.safetensors"
    metrics_path = candidate_dir / f"{canonical_stem}.metrics.safetensors"
    ledger_path = candidate_dir / f"{canonical_stem}.selection.json"
    if _layer_complete(candidate_dir, layer):
        return
    if ledger_path.exists() and not (
        payload_path.is_file() and metrics_path.is_file()
    ):
        raise ValueError(f"canonical layer {layer} ledger precedes its tensors")

    jobs = sorted(jobs, key=lambda job: job.experts[0])
    flattened = [expert for job in jobs for expert in job.experts]
    if sorted(flattened) != list(range(C.NUM_EXPERTS)) or len(flattened) != C.NUM_EXPERTS:
        raise ValueError(f"merge jobs do not cover layer {layer}")
    for job in jobs:
        if not _job_complete(candidate_dir, job):
            raise ValueError(
                f"cannot merge incomplete layer {layer} expert subset"
            )

    started = time.time()
    if not payload_path.exists():
        partial = AtomicSafetensorsWriter.partial_path(payload_path)
        if partial.exists():
            if not args.resume:
                raise FileExistsError(partial)
            AtomicSafetensorsWriter.discard_stale_partial(payload_path)
        with AtomicSafetensorsWriter(
            payload_path,
            _candidate_payload_specs_for_modes(
                layer, list(range(C.NUM_EXPERTS)), args.mode_ids
            ),
            metadata={
                "kind": KIND,
                "schema_version": str(SCHEMA_VERSION),
                "layer": str(layer),
                "codebook": args.codebook,
                "tailbite_context": str(args.tailbite_context),
            },
        ) as writer:
            for job in jobs:
                stem = _layer_stem(layer, list(job.experts))
                source_path = candidate_dir / f"{stem}.safetensors"
                expected_names = {
                    spec.name
                    for spec in _candidate_payload_specs_for_modes(
                        layer, list(job.experts), args.mode_ids
                    )
                }
                with safe_open(
                    str(source_path), framework="pt", device="cpu"
                ) as handle:
                    if set(handle.keys()) != expected_names:
                        raise ValueError(
                            f"subset payload tensor inventory drifted: {source_path}"
                        )
                    metadata = handle.metadata() or {}
                    if metadata.get("codebook") != args.codebook:
                        raise ValueError(
                            f"subset payload codebook drifted: {source_path}"
                        )
                    if metadata.get("tailbite_context") != str(args.tailbite_context):
                        raise ValueError(
                            f"subset payload tail-biting context drifted: {source_path}"
                        )
                    for spec in _candidate_payload_specs_for_modes(
                        layer, list(job.experts), args.mode_ids
                    ):
                        writer.write(spec.name, handle.get_tensor(spec.name))
    _validate_canonical_payload(
        payload_path,
        layer=layer,
        codebook=args.codebook,
        tailbite_context=args.tailbite_context,
        mode_ids=args.mode_ids,
    )

    if not metrics_path.exists():
        coupled_contract = _coupled_manifest_contract(args)
        metric_options = (
            {
                "coupled": True,
                **(
                    {"pooled_coupled": True}
                    if coupled_contract.get("selection")
                    == POOLED_ALL_ROW_SELECTION
                    else {}
                ),
            }
            if coupled_contract is not None
            else {}
        )
        merged_metrics = _allocate_metrics(
            list(range(C.NUM_EXPERTS)),
            args.mode_ids,
            len(partition.fit),
            len(partition.confirmation),
            **metric_options,
        )
        expected_metric_keys = set(merged_metrics)
        for job in jobs:
            stem = _layer_stem(layer, list(job.experts))
            source_path = candidate_dir / f"{stem}.metrics.safetensors"
            source = load_file(str(source_path), device="cpu")
            if set(source) != expected_metric_keys:
                raise ValueError(
                    f"subset metric inventory drifted: {source_path}"
                )
            if not torch.equal(
                source["expert_ids"],
                torch.tensor(job.experts, dtype=torch.int16),
            ):
                raise ValueError(f"subset metric expert order drifted: {source_path}")
            if not torch.equal(
                source["mode_ids"], torch.tensor(args.mode_ids, dtype=torch.uint8)
            ):
                raise ValueError(f"subset metric mode IDs drifted: {source_path}")
            destination = torch.tensor(job.experts, dtype=torch.long)
            for name, tensor in source.items():
                if name in ("expert_ids", "mode_ids"):
                    continue
                if tensor.ndim < 1 or tensor.shape[0] != len(job.experts):
                    raise ValueError(
                        f"subset metric {name} has no expert leading axis"
                    )
                merged_metrics[name].index_copy_(0, destination, tensor)
        _atomic_safetensors(metrics_path, merged_metrics)

    if not ledger_path.exists():
        source_ledgers = []
        for job in jobs:
            stem = _layer_stem(layer, list(job.experts))
            source_ledgers.append(
                _read_json(candidate_dir / f"{stem}.selection.json")
            )
        invariant_keys = (
            "kind",
            "schema_version",
            "complete",
            "layer",
            "source_checkpoint",
            "source_revision",
            "resident_teacher_checkpoint",
            "capture",
            "training_report",
            "hessians",
            "hessian_policy",
            "h2_contract",
            "selection_contract",
            "damage_contract",
            "source_residency",
            "logical_trellis_schema",
            "codebook",
            "layout",
            "ldlq_tf32",
            "tailbite_context",
            "rotation_draws",
            "coupled_rotation",
            "shared_hidden_scale_closure",
            "training_corpus_documents",
        )
        reference = source_ledgers[0]
        for source in source_ledgers:
            for key in invariant_keys:
                if source.get(key) != reference.get(key):
                    raise ValueError(
                        f"layer {layer} subset ledger disagrees on {key}"
                    )
        selections: dict[str, object] = {}
        for job, source in zip(jobs, source_ledgers, strict=True):
            if source.get("experts") != list(job.experts):
                raise ValueError(f"layer {layer} subset ledger expert order drifted")
            source_selections = source.get("selections")
            if not isinstance(source_selections, dict):
                raise ValueError(f"layer {layer} subset ledger has no selections")
            for expert in job.experts:
                key = str(expert)
                if key not in source_selections or key in selections:
                    raise ValueError(f"layer {layer} subset selection inventory drifted")
                selections[key] = source_selections[key]
        if set(selections) != {str(expert) for expert in range(C.NUM_EXPERTS)}:
            raise ValueError(f"layer {layer} merged selections are incomplete")
        metrics = load_file(str(metrics_path), device="cpu")
        selected_histogram = {
            _format_label(r13, r2): int(
                (
                    (metrics["selected_r13"] == r13)
                    & (metrics["selected_r2"] == r2)
                ).sum()
            )
            for r13 in args.mode_ids
            for r2 in args.mode_ids
        }
        ledger = dict(reference)
        ledger.update(
            {
                "experts": list(range(C.NUM_EXPERTS)),
                "all_experts": True,
                "payload_write": "atomic_merged_subset_safetensors_stream",
                "payload": payload_path.name,
                "metrics": metrics_path.name,
                "seconds": sum(float(source["seconds"]) for source in source_ledgers),
                "merge_seconds": time.time() - started,
                "merged_from_subsets": [
                    _layer_stem(layer, list(job.experts)) for job in jobs
                ],
                "shared_scale_primer_expert": 0,
                "shared_scale_primer_candidates": sum(
                    int(source.get("shared_scale_primer_candidates", 0))
                    for source in source_ledgers
                ),
                "selected_format_histogram": selected_histogram,
                "selected_coupled_draw_histogram": (
                    None
                    if "coupled_draw_selected" not in metrics
                    else {
                        str(draw): int(
                            (metrics["coupled_draw_selected"] == draw).sum()
                        )
                        for draw in range(8)
                        if bool(
                            (metrics["coupled_draw_selected"] == draw).any()
                        )
                    }
                ),
                "selections": selections,
            }
        )
        _atomic_json(ledger_path, ledger)
    if not _layer_complete(candidate_dir, layer):
        raise AssertionError(f"merged layer {layer} did not close")
    print(
        f"layer {layer}: merged {len(jobs)} expert subsets in "
        f"{time.time() - started:.1f}s",
        flush=True,
    )


def _merge_scheduled_layers(
    args: argparse.Namespace,
    schedule: tuple[tuple[_ScheduleJob, ...], ...],
    partition: RequestPartition,
) -> None:
    by_layer: dict[int, list[_ScheduleJob]] = {
        layer: [] for layer in C.MOE_LAYERS
    }
    for jobs in schedule:
        for job in jobs:
            by_layer[job.layer].append(job)
    for layer in C.MOE_LAYERS:
        jobs = by_layer[layer]
        if len(jobs) == 1 and jobs[0].experts == tuple(range(C.NUM_EXPERTS)):
            if not _job_complete(args.dest / "candidates", jobs[0]):
                raise ValueError(f"worker did not complete canonical layer {layer}")
            continue
        _merge_scheduled_layer(args, layer, jobs, partition)


def worker(args: argparse.Namespace) -> None:
    # Independent GPU encoders share one host. PyTorch otherwise gives every
    # process a full-host intra-op pool. The dense numerical work is CUDA, so
    # reserve an explicit, small CPU budget per worker instead of
    # oversubscribing the host.
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    if args.work_plan is not None:
        if args.layers is not None or args.experts is not None:
            raise ValueError("a scheduled worker cannot override layers or experts")
        schedule = _schedule_from_report(
            _read_json(args.work_plan), workers=args.worker_count
        )
        jobs = list(schedule[args.worker])
    else:
        layers = (
            list(args.layers)
            if args.layers is not None
            else [
                layer
                for index, layer in enumerate(C.MOE_LAYERS)
                if index % args.worker_count == args.worker
            ]
        )
        experts = (
            tuple(range(C.NUM_EXPERTS))
            if args.experts is None
            else tuple(args.experts)
        )
        jobs = [_ScheduleJob(layer, experts, 1) for layer in layers]
    pending_jobs = [
        job
        for job in jobs
        if not _job_complete(args.dest / "candidates", job)
    ]
    for job in jobs:
        if job not in pending_jobs:
            print(
                f"layer {job.layer} experts {job.experts[0]}-{job.experts[-1]}: "
                "complete, skip",
                flush=True,
            )
    if not pending_jobs:
        return
    pending_layers = [job.layer for job in pending_jobs]
    if len(set(pending_layers)) != len(pending_layers):
        raise ValueError("one worker cannot encode two subsets of the same layer")
    report, partition = _validate_training_contract(
        args.training_report,
        capture=args.capture,
        teacher_checkpoint=args.teacher_checkpoint,
    )
    sample_layers = [layer - 1 for layer in pending_layers]
    sample_index = (
        index_layer_samples(args.capture, sample_layers)
        if args.sample_cache is None
        else index_cached_layer_samples(args.sample_cache, sample_layers)
    )
    resources = _WorkerResources(
        report=report,
        partition=partition,
        sample_index=sample_index,
        store=OfficialMXFP4Store(
            repo_dir=args.official_repo_dir,
            revision=args.official_revision,
        ),
        quantizer_module=_load_quantizer_module(
            args.exllamav3_root, args.codebook
        ),
    )
    failures: list[tuple[_ScheduleJob, BaseException]] = []
    for job in pending_jobs:
        try:
            encode_layer(
                args,
                job.layer,
                list(job.experts),
                resources=resources,
            )
        except Exception as exc:
            failures.append((job, exc))
            print(
                f"worker {args.worker}: layer {job.layer} experts "
                f"{job.experts[0]}-{job.experts[-1]} failed; "
                "continuing with remaining jobs",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
            _reset_cuda_after_failure()
    if failures:
        failed_ranges = ", ".join(
            f"L{job.layer}:{job.experts[0]}-{job.experts[-1]}"
            for job, _exc in failures
        )
        raise RuntimeError(
            f"worker {args.worker} left {len(failures)} incomplete jobs: "
            f"{failed_ranges}"
        )


def _manifest(args: argparse.Namespace) -> dict:
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "source_model": C.MODEL_ID,
        "source_revision": args.official_revision,
        "exllamav3_root": str(args.exllamav3_root.resolve()),
        "resident_teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "capture": str(args.capture.resolve()),
        "sample_cache": (
            None
            if getattr(args, "sample_cache", None) is None
            else str(args.sample_cache.resolve())
        ),
        "training_report": str(args.training_report.resolve()),
        "hessians": str(args.hessians.resolve()),
        "hessian_policy": args.hessian_policy,
        "permutation_policy": getattr(args, "permutation_policy", "h2_reverse"),
        "folded_scale_power": float(getattr(args, "folded_scale_power", 0.0)),
        "h2_contract": _h2_contract(args),
        "down_target_contract": _down_target_contract(args),
        **_selection_contract(args),
        "min_fit_documents": args.min_fit_documents,
        "min_confirmation_documents": args.min_confirmation_documents,
        "minimum_improvement": args.minimum_improvement,
        "bootstrap_replicates": args.bootstrap_replicates,
        "damage_metric": OFFICIAL_SOURCE_DAMAGE_METRIC,
        "damage_already_natural_route_and_gate_weighted": True,
        "random_transform_contract": (
            "layer_shared_residual_draw_expert_private_intermediate_draw"
        ),
        "rotation_draws": _rotation_manifest_contract(args),
        "coupled_rotation": _coupled_manifest_contract(args),
        "logical_trellis_schema": args.logical_trellis_schema,
        "codebook": args.codebook,
        "layout": getattr(args, "layout", "qsrt_guarded_reuse"),
        "ldlq_tf32": bool(getattr(args, "ldlq_tf32", False)),
        "tailbite_context": int(getattr(args, "tailbite_context", 128)),
    }


def _bind_candidate_manifest(
    args: argparse.Namespace,
    manifest_path: Path,
    *,
    create: bool,
) -> dict:
    """Freeze the complete QSRT encoding contract for safe resume."""

    expected = _manifest(args)
    if not manifest_path.exists():
        if not create:
            raise ValueError("resume destination is missing its candidate manifest")
        _atomic_json(manifest_path, expected)
        return expected

    actual = _read_json(manifest_path)
    if actual == expected:
        return actual

    raise ValueError("destination manifest does not match this candidate run")


def _stop_worker_processes(processes: list[subprocess.Popen]) -> None:
    live = [process for process in processes if process.poll() is None]
    for process in live:
        process.terminate()
    deadline = time.monotonic() + 10.0
    for process in live:
        timeout = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
    for process in live:
        process.wait()


def _wait_for_worker_processes(processes: list[subprocess.Popen]) -> int:
    pending = list(processes)
    first_failure = 0
    try:
        while pending:
            for process in tuple(pending):
                return_code = process.poll()
                if return_code is None:
                    continue
                pending.remove(process)
                if return_code and not first_failure:
                    first_failure = return_code
            if pending:
                time.sleep(0.1)
    except BaseException:
        _stop_worker_processes(pending)
        raise
    return first_failure


def _candidate_worker_command(
    args: argparse.Namespace,
    *,
    worker_id: int,
    worker_count: int,
    schedule_path: Path,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        __file__,
        "--dest",
        str(args.dest),
        "--capture",
        str(args.capture),
        "--training-report",
        str(args.training_report),
        "--hessians",
        str(args.hessians),
        "--hessian-policy",
        args.hessian_policy,
        "--permutation-policy",
        args.permutation_policy,
        "--folded-scale-power",
        str(args.folded_scale_power),
        "--down-target-policy",
        args.down_target_policy,
        "--w2-refit-regularization-ratio",
        str(args.w2_refit_regularization_ratio),
        "--teacher-checkpoint",
        str(args.teacher_checkpoint),
        "--official-revision",
        args.official_revision,
        "--exllamav3-root",
        str(args.exllamav3_root),
        "--mode-ids",
        ",".join(map(str, args.mode_ids)),
        "--codebook",
        args.codebook,
        "--layout",
        args.layout,
        "--tailbite-context",
        str(args.tailbite_context),
        "--residual-rotation-draw",
        str(args.residual_rotation_draw),
        "--intermediate-rotation-draw",
        str(args.intermediate_rotation_draw),
        "--min-fit-documents",
        str(args.min_fit_documents),
        "--min-confirmation-documents",
        str(args.min_confirmation_documents),
        "--minimum-improvement",
        str(args.minimum_improvement),
        "--bootstrap-replicates",
        str(args.bootstrap_replicates),
        "--expert-batch-size",
        str(args.expert_batch_size),
        "--batch-retries",
        str(args.batch_retries),
        "--cpu-threads",
        str(args.cpu_threads),
        "--logical-trellis-schema",
        args.logical_trellis_schema,
        "--worker",
        str(worker_id),
        "--worker-count",
        str(worker_count),
        "--work-plan",
        str(schedule_path),
    ]
    if args.sample_cache is not None:
        command.extend(("--sample-cache", str(args.sample_cache)))
    if args.rotation_plan is not None:
        command.extend(("--rotation-plan", str(args.rotation_plan)))
    if args.coupled_rotation_plan is not None:
        command.extend(
            (
                "--coupled-rotation-plan",
                str(args.coupled_rotation_plan),
            )
        )
    if args.coupled_draw_candidates is not None:
        command.extend(
            (
                "--coupled-draw-candidates",
                ",".join(str(draw) for draw in args.coupled_draw_candidates),
            )
        )
    if args.pooled_selection_capture is not None:
        command.extend(
            (
                "--pooled-selection-capture",
                str(args.pooled_selection_capture),
                "--pooled-selection-batch-rows",
                str(args.pooled_selection_batch_rows),
            )
        )
    if args.official_repo_dir is not None:
        command.extend(("--official-repo-dir", str(args.official_repo_dir)))
    if resume:
        command.append("--resume")
    if args.ldlq_tf32:
        command.append("--ldlq-tf32")
    return command


def _launch_candidate_worker(
    args: argparse.Namespace,
    *,
    worker_id: int,
    worker_count: int,
    device: int,
    schedule_path: Path,
    resume: bool,
) -> subprocess.Popen:
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device))
    command = _candidate_worker_command(
        args,
        worker_id=worker_id,
        worker_count=worker_count,
        schedule_path=schedule_path,
        resume=resume,
    )
    return subprocess.Popen(command, env=environment)


def _run_candidate_worker_queue(
    args: argparse.Namespace,
    *,
    worker_ids: list[int],
    devices: list[int],
    worker_count: int,
    schedule_path: Path,
    resume: bool,
) -> dict[int, tuple[int, int]]:
    pending = list(worker_ids)
    available_devices = list(devices)
    active: dict[int, tuple[int, subprocess.Popen]] = {}
    results: dict[int, tuple[int, int]] = {}

    def launch_available() -> None:
        while pending and available_devices:
            worker_id = pending.pop(0)
            device = available_devices.pop(0)
            active[device] = (
                worker_id,
                _launch_candidate_worker(
                    args,
                    worker_id=worker_id,
                    worker_count=worker_count,
                    device=device,
                    schedule_path=schedule_path,
                    resume=resume,
                ),
            )

    launch_available()
    try:
        while active:
            for device, (worker_id, process) in tuple(active.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                del active[device]
                results[worker_id] = (return_code, device)
                if return_code == 0:
                    available_devices.append(device)
                else:
                    print(
                        f"worker {worker_id} failed on device {device}; "
                        "device quarantined for this run",
                        file=sys.stderr,
                        flush=True,
                    )
            launch_available()
            if active:
                time.sleep(0.1)
    except BaseException:
        _stop_worker_processes(
            [process for _worker_id, process in active.values()]
        )
        raise
    return results


def _repair_candidate_workers(
    args: argparse.Namespace,
    *,
    failed_worker_ids: list[int],
    devices: list[int],
    worker_count: int,
    schedule_path: Path,
) -> dict[int, tuple[int, int]]:
    return _run_candidate_worker_queue(
        args,
        worker_ids=failed_worker_ids,
        devices=devices,
        worker_count=worker_count,
        schedule_path=schedule_path,
        resume=True,
    )


def parent(args: argparse.Namespace) -> None:
    if args.experts is not None or args.layers is not None or args.work_plan is not None:
        raise ValueError("parent mode always builds all experts and all 92 MoE layers")
    manifest_path = args.dest / "qsrt-candidate-manifest.json"
    schedule_path = args.dest / "worker-schedule.json"
    if args.dest.exists():
        if not args.resume:
            raise FileExistsError(
                f"destination {args.dest} exists; use --resume only for the identical run"
            )
        if not (args.dest / "candidates").is_dir():
            raise ValueError("resume destination is missing its candidates directory")
        _bind_candidate_manifest(args, manifest_path, create=False)
    else:
        args.dest.mkdir(parents=True)
        (args.dest / "candidates").mkdir()
        _bind_candidate_manifest(args, manifest_path, create=True)
    _report, partition = _validate_training_contract(
        args.training_report,
        capture=args.capture,
        teacher_checkpoint=args.teacher_checkpoint,
    )
    fixed_k3 = args.mode_ids == (R0.mode_id,)
    fixed_high_rate = args.mode_ids == (H308.mode_id,)
    fixed_k2 = args.mode_ids == (K2.mode_id,)
    fixed_profile = fixed_k3 or fixed_high_rate or fixed_k2
    max_mode = max(args.mode_ids)
    coupled_draws = _configured_coupled_draw_candidates(args)
    base_work = 74 if fixed_high_rate else (48 if fixed_k2 else 72)
    r0_work = base_work * (1 if coupled_draws is None else len(coupled_draws))
    full_work = r0_work if fixed_profile or max_mode == 0 else 96 + 8 * max_mode
    if schedule_path.exists():
        if not args.resume:
            raise FileExistsError(schedule_path)
        schedule_report = _read_json(schedule_path)
        if schedule_report.get("expert_batch_size") != args.expert_batch_size:
            raise ValueError("resume worker schedule uses another expert batch size")
        if (
            schedule_report.get("r0_expert_cost") != r0_work
            or schedule_report.get("full_grid_expert_cost") != full_work
        ):
            raise ValueError("resume worker schedule uses another mode-grid cost")
        schedule = _schedule_from_report(schedule_report)
    else:
        if args.resume:
            raise ValueError("resume destination is missing its worker schedule")
        eligibility = _full_grid_eligibility_by_layer(
            args.capture,
            partition,
            min_fit_documents=args.min_fit_documents,
            min_confirmation_documents=args.min_confirmation_documents,
        )
        expert_costs = {
            layer: tuple(
                full_work if eligible else r0_work
                for eligible in eligibility[layer]
            )
            for layer in C.MOE_LAYERS
        }
        schedule = _balanced_expert_schedule(
            expert_costs,
            list(C.MOE_LAYERS),
            workers=len(args.devices),
            expert_batch_size=args.expert_batch_size,
        )
        full_grid = {
            layer: sum(eligibility[layer]) for layer in C.MOE_LAYERS
        }
        schedule_report = _schedule_report(
            schedule,
            expert_batch_size=args.expert_batch_size,
            r0_work=r0_work,
            full_grid_work=full_work,
            full_grid_experts_by_layer=full_grid,
        )
        _atomic_json(schedule_path, schedule_report)
    for worker in schedule_report["workers"]:
        print(
            f"worker {worker['worker']}: layers {worker['layers']}; "
            f"jobs {len(worker['jobs'])}; estimated work {worker['estimated_work']}",
            flush=True,
        )
    print(
        f"schedule: {schedule_report['split_layer_count']} split layers; "
        f"max/mean={schedule_report['max_over_mean']:.5f}",
        flush=True,
    )

    worker_ids = [
        worker_id for worker_id, jobs in enumerate(schedule) if jobs
    ]
    worker_count = len(schedule)
    results = _run_candidate_worker_queue(
        args,
        worker_ids=worker_ids,
        devices=list(args.devices),
        worker_count=worker_count,
        schedule_path=schedule_path,
        resume=args.resume,
    )
    failed_worker_ids = [
        worker_id
        for worker_id in worker_ids
        if worker_id not in results or results[worker_id][0]
    ]
    failed_devices = {
        device for return_code, device in results.values() if return_code
    }
    healthy_devices = [
        device for device in args.devices if device not in failed_devices
    ]
    for attempt in range(args.worker_retries):
        if not failed_worker_ids or not healthy_devices:
            break
        print(
            f"repair attempt {attempt + 1}/{args.worker_retries}: workers "
            f"{failed_worker_ids} on healthy devices {healthy_devices}",
            flush=True,
        )
        results = _repair_candidate_workers(
            args,
            failed_worker_ids=failed_worker_ids,
            devices=healthy_devices[attempt % len(healthy_devices) :]
            + healthy_devices[: attempt % len(healthy_devices)],
            worker_count=worker_count,
            schedule_path=schedule_path,
        )
        failed_devices.update(
            device for return_code, device in results.values() if return_code
        )
        healthy_devices = [
            device for device in healthy_devices if device not in failed_devices
        ]
        failed_worker_ids = [
            worker_id
            for worker_id in failed_worker_ids
            if worker_id not in results or results[worker_id][0]
        ]
    if failed_worker_ids:
        raise RuntimeError(
            "candidate workers left incomplete jobs after repair: "
            + ", ".join(map(str, failed_worker_ids))
        )
    _merge_scheduled_layers(args, schedule, partition)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().with_name("finalize_qsrt_candidate_pool.py")),
            str(args.dest),
            "--jobs",
            str(args.finalize_jobs),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument(
        "--sample-cache",
        type=Path,
        help="layer-indexed rank-0 input cache built from the same capture",
    )
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--hessians", type=Path, required=True)
    parser.add_argument(
        "--hessian-policy",
        choices=HESSIAN_POLICIES,
        default="captured_blend",
        help=(
            "encoder geometry: captured_blend uses candidate-specific decoded-"
            "upstream H2 with adaptive expert-local identity shrinkage; identity "
            "is the robust SQG baseline"
        ),
    )
    parser.add_argument(
        "--permutation-policy",
        choices=PERMUTATION_POLICIES,
        default="h2_reverse",
        help=(
            "common intermediate-neuron record assignment; h2_reverse is "
            "the established post-SiTU importance ordering"
        ),
    )
    parser.add_argument(
        "--folded-scale-power",
        type=float,
        default=0.0,
        help=(
            "metadata-free w1/w3 per-128 conditioning strength; zero disables"
        ),
    )
    parser.add_argument(
        "--down-target-policy",
        choices=DOWN_TARGET_POLICIES,
        default="source",
        help=(
            "source encodes the checkpoint W2; functional_ridge first fits one "
            "W2 target to each decoded upstream candidate"
        ),
    )
    parser.add_argument(
        "--w2-refit-regularization-ratio",
        type=float,
        default=1e-2,
        help=(
            "ridge strength relative to the candidate hidden Gram trace per "
            "dimension when --down-target-policy=functional_ridge"
        ),
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=Path("/models/Kimi-K3-EXL3-3p09-serve"),
    )
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument("--official-revision", default=C.REVISION)
    parser.add_argument(
        "--exllamav3-root",
        type=Path,
        default=Path("/home/luke/projects/exllamav3"),
    )
    parser.add_argument("--mode-ids", type=_parse_modes, default=PHASE1_MODE_IDS)
    parser.add_argument(
        "--codebook",
        choices=QSRT_CODEBOOKS,
        default=CODEBOOK_SQG_XOR_CHEB_T12,
        help="procedural reconstruction family used by the Viterbi encoder",
    )
    parser.add_argument(
        "--layout",
        choices=MIXED_SEARCH_LAYOUTS,
        default="qsrt_guarded_reuse",
        help="offline LDLQ traversal order; canonical QSRT pair storage is unchanged",
    )
    parser.add_argument(
        "--ldlq-tf32",
        action="store_true",
        help="use TF32 Tensor Cores for offline dense-H LDLQ feedback products",
    )
    parser.add_argument(
        "--tailbite-context",
        type=int,
        default=128,
        help=(
            "symmetric Viterbi primer symbols on each side of the cyclic "
            "tile boundary (1..128; 128 is the full cyclic search)"
        ),
    )
    parser.add_argument(
        "--residual-rotation-draw",
        type=int,
        default=0,
        help=(
            "layer-shared residual-side sign draw; zero reproduces the "
            "established production transform"
        ),
    )
    parser.add_argument(
        "--intermediate-rotation-draw",
        type=int,
        default=0,
        help=(
            "expert-private intermediate-side sign draw; zero reproduces "
            "the established production transform"
        ),
    )
    parser.add_argument(
        "--rotation-plan",
        type=Path,
        help=(
            "validated model-wide JSON plan with one residual draw per layer "
            "and optional expert-private intermediate draw overrides"
        ),
    )
    parser.add_argument(
        "--coupled-rotation-plan",
        type=Path,
        help=(
            "complete model-wide expert-static coupled-transform draw plan"
        ),
    )
    parser.add_argument(
        "--coupled-draw-candidates",
        type=_parse_ints,
        help=(
            "expert-static coupled-transform draw portfolio; without a pooled "
            "selection capture, fit proposes one draw and the disjoint "
            "confirmation fold may only accept it against draw zero"
        ),
    )
    parser.add_argument(
        "--pooled-selection-capture",
        type=Path,
        help=(
            "finalized all-routed-row capture used to select the minimum "
            "whole-expert post-projection SSE across coupled draws"
        ),
    )
    parser.add_argument(
        "--pooled-selection-batch-rows",
        type=int,
        default=512,
        help="naturally routed expert occurrences scored per GPU batch",
    )
    parser.add_argument("--min-fit-documents", type=int, default=PHASE1_MIN_FIT_DOCUMENTS)
    parser.add_argument(
        "--min-confirmation-documents",
        type=int,
        default=PHASE1_MIN_CONFIRMATION_DOCUMENTS,
    )
    parser.add_argument("--minimum-improvement", type=float, default=0.0)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=PHASE1_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--expert-batch-size",
        type=int,
        default=8,
        help="experts jointly scheduled on one GPU to fill trellis launches",
    )
    parser.add_argument(
        "--batch-retries",
        type=int,
        default=1,
        help="retries after a failed GPU expert-batch encode (default: 1)",
    )
    parser.add_argument(
        "--worker-retries",
        type=int,
        default=1,
        help=(
            "repair attempts for incomplete worker schedules on devices whose "
            "initial workers completed successfully (default: 1)"
        ),
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=2,
        help="PyTorch intra-op CPU threads per GPU worker (default: 2)",
    )
    parser.add_argument(
        "--finalize-jobs",
        type=int,
        default=12,
        help="parallel SHA-256 jobs used to seal a completed parent pool",
    )
    parser.add_argument(
        "--logical-trellis-schema",
        choices=LOGICAL_CANDIDATE_SCHEMAS,
        default=QSRT_SCHEMA,
        help="logical candidate descriptor schema written into every manifest",
    )
    parser.add_argument("--worker", type=int)
    parser.add_argument("--worker-count", type=int)
    parser.add_argument(
        "--devices",
        type=lambda value: tuple(
            int(item) for item in value.split(",") if item.strip()
        ),
        help=(
            "comma-separated GPU indices for parent scheduling; defaults to "
            "all CUDA devices visible to this process"
        ),
    )
    parser.add_argument(
        "--work-plan",
        type=Path,
        help="frozen schema-v2 worker schedule generated by parent mode",
    )
    parser.add_argument("--layers", type=_parse_ints)
    parser.add_argument("--experts", type=_parse_ints)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an identical all-layer parent run from completed atomic layers",
    )
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if not math.isfinite(args.folded_scale_power):
        parser.error("--folded-scale-power must be finite")
    if (
        not math.isfinite(args.w2_refit_regularization_ratio)
        or args.w2_refit_regularization_ratio <= 0
    ):
        parser.error("--w2-refit-regularization-ratio must be finite and positive")
    if min(args.residual_rotation_draw, args.intermediate_rotation_draw) < 0:
        parser.error("rotation draw indices must be nonnegative")
    if args.rotation_plan is not None and (
        args.residual_rotation_draw or args.intermediate_rotation_draw
    ):
        parser.error(
            "--rotation-plan cannot be combined with nonzero fixed rotation draws"
        )
    coupled_sources = sum(
        value is not None
        for value in (
            args.coupled_rotation_plan,
            args.coupled_draw_candidates,
        )
    )
    if coupled_sources > 1:
        parser.error(
            "--coupled-rotation-plan and --coupled-draw-candidates "
            "are mutually exclusive"
        )
    if args.mode_ids == (K2.mode_id,) and coupled_sources != 1:
        parser.error(
            "fixed K2 requires exactly one coupled-transform rotation source"
        )
    if coupled_sources and args.mode_ids not in (
        (R0.mode_id,),
        (H308.mode_id,),
        (K2.mode_id,),
    ):
        parser.error(
            "coupled transforms require a fixed K2, K3, or "
            "22-K3/2-K4 schedule"
        )
    if coupled_sources and (
        args.rotation_plan is not None
        or args.residual_rotation_draw
        or args.intermediate_rotation_draw
    ):
        parser.error(
            "coupled transforms cannot be combined with ordinary internal "
            "rotation controls"
        )
    if args.coupled_draw_candidates is not None and (
        args.coupled_draw_candidates[0] != 0
        or any(not 0 <= draw < 8 for draw in args.coupled_draw_candidates)
    ):
        parser.error(
            "--coupled-draw-candidates must begin with zero and lie in 0..7"
        )
    if args.pooled_selection_capture is not None:
        if args.coupled_draw_candidates is None:
            parser.error(
                "--pooled-selection-capture requires --coupled-draw-candidates"
            )
        try:
            pooled_manifest, pooled_geometry, _pooled_chunks = load_all_row_capture(
                args.pooled_selection_capture,
                verify_hashes=False,
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(f"invalid pooled selection capture: {exc}")
        if (
            tuple(pooled_geometry.layers) != tuple(C.MOE_LAYERS)
            or pooled_geometry.input_size != C.LATENT
            or pooled_geometry.top_k != C.TOP_K
            or int(pooled_manifest.get("rows", 0)) <= 0
        ):
            parser.error("pooled selection capture has incompatible Kimi geometry")
    if args.pooled_selection_batch_rows < 1:
        parser.error("--pooled-selection-batch-rows must be positive")
    if not math.isfinite(args.minimum_improvement) or args.minimum_improvement < 0:
        parser.error("--minimum-improvement must be finite and nonnegative")
    if args.worker is None:
        if args.worker_count is not None:
            parser.error("--worker-count is internal to worker mode")
        if args.devices is None:
            args.devices = tuple(range(torch.cuda.device_count()))
        if (
            not args.devices
            or len(set(args.devices)) != len(args.devices)
            or any(device < 0 for device in args.devices)
        ):
            parser.error("--devices must contain unique nonnegative GPU indices")
    else:
        if args.worker_count is None or args.worker_count < 1:
            parser.error("worker mode requires a positive --worker-count")
        if not 0 <= args.worker < args.worker_count:
            parser.error("--worker must lie in 0..worker-count-1")
    if args.work_plan is not None and args.worker is None:
        parser.error("--work-plan requires --worker")
    if args.layers is not None and any(layer not in C.MOE_LAYERS for layer in args.layers):
        parser.error("--layers must contain decoder layers 1..92")
    if args.experts is not None and any(not 0 <= expert < C.NUM_EXPERTS for expert in args.experts):
        parser.error("--experts must contain IDs 0..895")
    if min(args.min_fit_documents, args.min_confirmation_documents) < 1:
        parser.error("document support thresholds must be positive")
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    if args.expert_batch_size < 1:
        parser.error("--expert-batch-size must be positive")
    if min(args.batch_retries, args.worker_retries) < 0:
        parser.error("retry counts must be nonnegative")
    if not 1 <= args.tailbite_context <= 128:
        parser.error("--tailbite-context must be in 1..128")
    if args.finalize_jobs < 1:
        parser.error("--finalize-jobs must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.sample_cache is not None:
        _validate_sample_cache_contract(args.sample_cache, args.capture)
    if args.worker is None:
        parent(args)
    else:
        args.dest.mkdir(parents=True, exist_ok=True)
        (args.dest / "candidates").mkdir(exist_ok=True)
        manifest_path = args.dest / "qsrt-candidate-manifest.json"
        _bind_candidate_manifest(args, manifest_path, create=True)
        worker(args)


if __name__ == "__main__":
    main()

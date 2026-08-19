"""Build and allocate GLM-5.2 K3/K4 candidates with one down-refit target.

The reconstructed-activation down refit first derives one continuous down
matrix from the frozen K3 gate and up reconstructions.  This module encodes
that same fitted matrix at K3 and K4.  It fails unless the repeated K3 encode
is byte-identical to the accepted stored refit.  Gate and up K4 tensors come
from the reusable source-target K4 pool.

Rate allocation uses only the frozen candidate-selection documents.  The
published BF16 reporting context cannot enter target fitting, path selection,
or rate selection.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from qsrt.correctness import sha256_file
from qsrt.exl3_loader import load_qsrt_encoder
from qsrt.exl3_reference import CODEBOOK_SQG_XOR_CHEB_T12
from qsrt.glm52_down_refit import (
    _expert_hidden,
    _read_capture_rows,
    _teacher_output,
    solve_down_correction,
)
from qsrt.glm52_expert_intervention import (
    INTERVENTION_ARTIFACT_KIND,
    _atomic_save_tensors,
    _dense_expert_path,
)
from qsrt.glm52_expert_intervention_runtime import (
    validate_dense_intervention_artifact,
)
from qsrt.glm52_k3_k4_allocation import frozen_k4_rate_map
from qsrt.glm52_pilot import (
    PROJECTIONS,
    IndexedTensorStore,
    _expert_path,
    _transform_seeds,
    atomic_write_json,
    prepare_destination,
    source_tensor_name,
)
from qsrt.glm52_real_weight_benchmark import (
    load_frozen_real_weight_panel,
    select_frozen_panel_slice,
    validate_bounded_source_window,
)
from qsrt.ldlq import SIGMA_REG
from qsrt.qsrt_codec_pilot import encode_uniform_candidate, tensor_sha256
from qsrt.sqg_quantizer import install_sqg_quantizer


DOWN_REFIT_RATE_POOL_KIND = "qsrt_glm52_down_refit_k3_k4_rate_pool_v1"
DOWN_REFIT_RATE_POOL_TENSOR_PREFIXES = ("qsrt_k3", "qsrt_k4")
RATE_PRESERVING_PRE_REGISTRATION_SHA256 = (
    "d9f34c83e1d152018ae9305cc6f9835c3efd0e380ed43e99d81a7c6490d0aa3b"
)
RATE_TUPLES = tuple(itertools.product((3, 4), repeat=3))
PROJECTION_NAMES = tuple(spec.name for spec in PROJECTIONS)
REGISTERED_PARTIAL_RATE_MAP_SCHEMA = (
    "qsrt_glm52_registered_partial_down_refit_k3_k4_intervention"
)


def _canonical_json_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _sha256_identity(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 identity")
    return value


def _model_layer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 3 <= value <= 77:
        raise ValueError("GLM mixture-of-experts layer must be an integer from 3 through 77")
    return value


def _pool_tensor_path(root: Path, layer: int, expert: int) -> Path:
    return root / "experts" / f"layer-{layer:03d}-expert-{expert:03d}.safetensors"


def _atomic_save_pool_tensors(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
            temporary,
            metadata={"format": DOWN_REFIT_RATE_POOL_KIND},
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _records_by_expert(artifact: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {
        int(record["expert"]): record
        for record in artifact["report"]["experts"]
    }


def _load_intervention_tensors(
    root: Path,
    record: Mapping[str, Any],
    *,
    candidate_prefix: str,
) -> dict[str, torch.Tensor]:
    path = root / "experts" / str(record["dense_endpoint_file"])
    if (
        path.stat().st_size != int(record["dense_endpoint_file_bytes"])
        or sha256_file(path) != record["dense_endpoint_file_sha256"]
    ):
        raise ValueError(f"dense endpoint {path} failed byte closure")
    required = {
        f"{prefix}.{projection}"
        for prefix in ("exl3", candidate_prefix)
        for projection in PROJECTION_NAMES
    }
    with safe_open(path, framework="pt", device="cpu") as handle:
        if not required.issubset(handle.keys()):
            raise ValueError(f"dense endpoint {path} lacks required tensors")
        return {name: handle.get_tensor(name) for name in required}


def weighted_error_sums(
    teacher: torch.Tensor,
    candidate: torch.Tensor,
    route_weights: torch.Tensor,
) -> tuple[float, float]:
    """Return route-weighted squared error and reference-energy sums."""

    if teacher.shape != candidate.shape or teacher.ndim != 2:
        raise ValueError("teacher and candidate outputs must share one matrix shape")
    if route_weights.ndim != 1 or route_weights.numel() != teacher.shape[0]:
        raise ValueError("route weights must contain one value per output row")
    weights = route_weights.double().unsqueeze(1)
    error_sum = float((((teacher.double() - candidate.double()) * weights).square().sum()).item())
    reference_sum = float(((teacher.double() * weights).square().sum()).item())
    if not math.isfinite(error_sum) or not math.isfinite(reference_sum) or reference_sum <= 0.0:
        raise ValueError("weighted output metric is non-finite or degenerate")
    return error_sum, reference_sum


def _rate_key(rates: Sequence[int]) -> str:
    values = tuple(int(rate) for rate in rates)
    if values not in RATE_TUPLES:
        raise ValueError("rate tuple must contain one K3/K4 value per projection")
    return "_".join(f"k{rate}" for rate in values)


def select_pooled_rate_allocation(
    expert_order: Sequence[int],
    candidate_metrics: Mapping[int, Mapping[tuple[int, int, int], float]],
    *,
    maximum_k4_projection_count: int,
) -> dict[str, Any]:
    """Minimize pooled selection error under a global K4 projection budget."""

    experts = tuple(int(expert) for expert in expert_order)
    if not experts or len(set(experts)) != len(experts):
        raise ValueError("expert order must contain unique expert IDs")
    if maximum_k4_projection_count < 0:
        raise ValueError("K4 projection budget must be nonnegative")
    normalized: dict[int, dict[tuple[int, int, int], float]] = {}
    for expert in experts:
        metrics = candidate_metrics.get(expert)
        if metrics is None or set(metrics) != set(RATE_TUPLES):
            raise ValueError(f"expert {expert} must contain all eight rate tuples")
        normalized[expert] = {}
        for rates, value in metrics.items():
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("candidate error sums must be finite and nonnegative")
            normalized[expert][tuple(rates)] = value

    # Each state stores total error and the rate sequence.  Sequence comparison
    # implements the pre-registered lexicographic tie break.
    states: dict[int, tuple[float, tuple[tuple[int, int, int], ...]]] = {0: (0.0, ())}
    for expert in experts:
        next_states: dict[int, tuple[float, tuple[tuple[int, int, int], ...]]] = {}
        for spent, (total, sequence) in states.items():
            for rates in RATE_TUPLES:
                promoted = sum(rate == 4 for rate in rates)
                new_spent = spent + promoted
                if new_spent > maximum_k4_projection_count:
                    continue
                proposal = (total + normalized[expert][rates], sequence + (rates,))
                incumbent = next_states.get(new_spent)
                if incumbent is None or proposal < incumbent:
                    next_states[new_spent] = proposal
        states = next_states
    if not states:
        raise RuntimeError("rate allocation has no feasible state")
    spent, (total_error, sequence) = min(
        states.items(), key=lambda item: (item[1][0], item[1][1], item[0])
    )
    return {
        "expert_order": list(experts),
        "rates_by_expert": {
            str(expert): dict(zip(PROJECTION_NAMES, rates, strict=True))
            for expert, rates in zip(experts, sequence, strict=True)
        },
        "k4_projection_count": spent,
        "candidate_selection_weighted_error_sum": total_error,
        "tie_break": "lexicographically earliest expert rate-tuple sequence",
    }


def validate_down_refit_rate_pool(root: Path) -> dict[str, Any]:
    """Hash and validate every receipt and tensor file in one rate pool."""

    root = root.resolve(strict=True)
    manifest = _read_json(root / "manifest.json")
    report = _read_json(root / "report.json")
    if manifest.get("kind") != f"{DOWN_REFIT_RATE_POOL_KIND}_manifest":
        raise ValueError("down-refit rate-pool manifest identity mismatch")
    manifest_sha256 = _canonical_json_sha256(manifest)
    layer = _model_layer(report.get("layer"))
    expected = {
        "kind": DOWN_REFIT_RATE_POOL_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "layer": layer,
    }
    if any(report.get(field) != value for field, value in expected.items()):
        raise ValueError("down-refit rate-pool report identity mismatch")
    records = report.get("experts")
    if not isinstance(records, list) or not records:
        raise ValueError("down-refit rate pool has no expert records")
    seen: set[int] = set()
    total_bytes = 0
    expected_tensor_keys = {
        f"{prefix}.{spec.name}"
        for prefix in ("exl3", *DOWN_REFIT_RATE_POOL_TENSOR_PREFIXES)
        for spec in PROJECTIONS
    }
    for record in records:
        expert = record.get("expert")
        if (
            isinstance(expert, bool)
            or not isinstance(expert, int)
            or not 0 <= expert < 256
            or expert in seen
        ):
            raise ValueError("down-refit rate-pool expert IDs must be unique")
        seen.add(expert)
        path = _pool_tensor_path(root, layer, expert)
        if (
            path.stat().st_size != int(record["tensor_file_bytes"])
            or sha256_file(path) != record["tensor_file_sha256"]
        ):
            raise ValueError(f"rate-pool tensor file for expert {expert} failed closure")
        tensor_hashes = record.get("tensor_sha256")
        if not isinstance(tensor_hashes, dict):
            raise TypeError(f"rate-pool expert {expert} lacks tensor identities")
        with safe_open(path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != expected_tensor_keys:
                raise ValueError(
                    f"rate-pool tensor file for expert {expert} has an unexpected schema"
                )
            if (handle.metadata() or {}) != {"format": DOWN_REFIT_RATE_POOL_KIND}:
                raise ValueError(
                    f"rate-pool tensor file for expert {expert} has unexpected metadata"
                )
            for prefix in ("exl3", *DOWN_REFIT_RATE_POOL_TENSOR_PREFIXES):
                prefix_hashes = tensor_hashes.get(prefix)
                if not isinstance(prefix_hashes, dict):
                    raise TypeError(
                        f"rate-pool expert {expert} lacks {prefix} tensor identities"
                    )
                for spec in PROJECTIONS:
                    tensor = handle.get_tensor(f"{prefix}.{spec.name}")
                    if tensor.dtype != torch.float16 or tuple(tensor.shape) != spec.source_shape:
                        raise ValueError(
                            f"rate-pool expert {expert} {prefix}.{spec.name} "
                            "has an unexpected dtype or shape"
                        )
                    if tensor_sha256(tensor) != prefix_hashes.get(spec.name):
                        raise ValueError(
                            f"rate-pool expert {expert} {prefix}.{spec.name} "
                            "failed tensor closure"
                        )
        candidates = record.get("rate_candidates")
        if not isinstance(candidates, list) or len(candidates) != len(RATE_TUPLES):
            raise ValueError(f"rate-pool expert {expert} lacks all rate candidates")
        seen_rates: set[tuple[int, int, int]] = set()
        reference_sum: float | None = None
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise TypeError("rate candidate must be an object")
            rates = tuple(
                int(candidate.get("rates", {}).get(projection, -1))
                for projection in PROJECTION_NAMES
            )
            if rates not in RATE_TUPLES or rates in seen_rates:
                raise ValueError(f"rate-pool expert {expert} has invalid rate candidates")
            seen_rates.add(rates)
            if candidate.get("key") != _rate_key(rates):
                raise ValueError(f"rate-pool expert {expert} candidate key mismatch")
            error_sum = float(candidate["candidate_selection_weighted_error_sum"])
            candidate_reference_sum = float(
                candidate["candidate_selection_weighted_reference_sum"]
            )
            relative_sse = float(candidate["candidate_selection_weighted_relative_sse"])
            if (
                not math.isfinite(error_sum)
                or error_sum < 0.0
                or not math.isfinite(candidate_reference_sum)
                or candidate_reference_sum <= 0.0
                or not math.isclose(
                    relative_sse,
                    error_sum / candidate_reference_sum,
                    rel_tol=1e-12,
                    abs_tol=0.0,
                )
            ):
                raise ValueError(f"rate-pool expert {expert} candidate metric mismatch")
            if reference_sum is None:
                reference_sum = candidate_reference_sum
            elif candidate_reference_sum != reference_sum:
                raise ValueError(
                    f"rate-pool expert {expert} reference energy changes by candidate"
                )
        if seen_rates != set(RATE_TUPLES):
            raise ValueError(f"rate-pool expert {expert} candidate coverage mismatch")
        receipt = _expert_path(root, layer, expert)
        if sha256_file(receipt) != record["receipt_sha256"]:
            raise ValueError(f"rate-pool receipt for expert {expert} failed closure")
        total_bytes += path.stat().st_size
    if (
        report.get("expert_count") != len(records)
        or report.get("tensor_file_bytes") != total_bytes
        or report.get("panel")
        != {str(layer): [int(record["expert"]) for record in records]}
        or manifest.get("panel") != report.get("panel")
    ):
        raise ValueError("down-refit rate-pool aggregate counts differ")
    return {
        "root": str(root),
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "model_layer": layer,
        "report": report,
        "expert_ids": tuple(sorted(seen)),
        "expert_count": len(seen),
        "tensor_file_bytes": total_bytes,
    }


@torch.no_grad()
def build_down_refit_rate_pool_slice(
    *,
    source_root: Path,
    source_inventory_path: Path,
    uniform_k3_root: Path,
    down_refit_root: Path,
    uniform_k4_root: Path,
    capture_root: Path,
    panel_manifest_path: Path,
    dest: Path,
    layer: int,
    expert_count: int,
    panel_offset: int,
    device: torch.device,
    exllamav3_root: Path,
    verify_source_shard_hashes: bool = True,
) -> dict[str, Any]:
    """Build one disjoint slice of K3/K4 candidates and selection metrics."""

    layer = _model_layer(layer)
    frozen = load_frozen_real_weight_panel(panel_manifest_path, layer=layer)
    experts = select_frozen_panel_slice(
        frozen, offset=panel_offset, expert_count=expert_count
    )
    uniform = validate_dense_intervention_artifact(uniform_k3_root)
    refit = validate_dense_intervention_artifact(down_refit_root)
    k4 = validate_dense_intervention_artifact(uniform_k4_root)
    if uniform["candidate_tensor_prefix"] != "qsrt_k3":
        raise ValueError("uniform input artifact must expose qsrt_k3 tensors")
    if refit["candidate_tensor_prefix"] != "qsrt_k3":
        raise ValueError("down-refit artifact must expose qsrt_k3 tensors")
    if k4["candidate_tensor_prefix"] != "qsrt_k4":
        raise ValueError("uniform-K4 artifact must expose qsrt_k4 tensors")
    for name, artifact in (("uniform K3", uniform), ("down refit", refit), ("uniform K4", k4)):
        if not set(experts).issubset(artifact["expert_ids"]):
            raise ValueError(f"{name} artifact does not cover the selected slice")
    refit_manifest = _read_json(Path(refit["root"]) / "manifest.json")
    if refit_manifest.get("input_intervention_artifact", {}).get("manifest_sha256") != uniform["manifest_sha256"]:
        raise ValueError("down-refit artifact was not derived from the supplied uniform K3 artifact")
    source_inventory = validate_bounded_source_window(
        source_root,
        source_inventory_path,
        panel={layer: experts},
        verify_shard_hashes=verify_source_shard_hashes,
    )
    capture_manifest_path = capture_root / "manifest.json"
    capture_manifest_sha256 = sha256_file(capture_manifest_path)
    routed = _read_capture_rows(capture_root, experts=experts, model_layer=layer)
    common = {
        "profile": "qsrt_sqg_e4m3",
        "codebook": CODEBOOK_SQG_XOR_CHEB_T12,
        "rates": [3, 4],
        "uniform_k3_manifest_sha256": uniform["manifest_sha256"],
        "down_refit_manifest_sha256": refit["manifest_sha256"],
        "down_refit_manifest_file_sha256": sha256_file(
            Path(refit["root"]) / "manifest.json"
        ),
        "down_refit_report_file_sha256": sha256_file(
            Path(refit["root"]) / "report.json"
        ),
        "uniform_k4_manifest_sha256": k4["manifest_sha256"],
        "activation_capture_manifest_sha256": capture_manifest_sha256,
        "frozen_panel_sha256": frozen["sha256"],
        "source_model_id": source_inventory["model_id"],
        "source_revision": source_inventory["revision"],
        "source_config_sha256": source_inventory["config_sha256"],
        "source_index_sha256": source_inventory["index_sha256"],
        "source_inventory_sha256": source_inventory["source_inventory_sha256"],
        "selection_collection": "candidate_selection",
        "reporting_context_used": False,
        "accepted_down_target_contract": (
            "recompute one continuous refit target, prove its K3 re-encode "
            "matches the stored refit, then encode the same target at K4"
        ),
    }
    manifest = {
        "kind": f"{DOWN_REFIT_RATE_POOL_KIND}_manifest",
        "common": common,
        "source": source_inventory,
        "input_artifacts": {
            "uniform_k3": {"root": uniform["root"], "manifest_sha256": uniform["manifest_sha256"]},
            "down_refit": {"root": refit["root"], "manifest_sha256": refit["manifest_sha256"]},
            "uniform_k4": {"root": k4["root"], "manifest_sha256": k4["manifest_sha256"]},
        },
        "activation_capture": {"root": str(capture_root.resolve()), "manifest_sha256": capture_manifest_sha256},
        "frozen_panel": {
            "path": frozen["path"],
            "sha256": frozen["sha256"],
            "selected_offset": panel_offset,
            "selected_count": expert_count,
        },
        "panel": {str(layer): list(experts)},
        "device": str(device),
        "coordinate_bases": {
            "exl3": "sealed_R7_permuted_middle_coordinates",
            "qsrt_k3": "official_source_middle_coordinates",
            "qsrt_k4": "official_source_middle_coordinates",
        },
        "numeric_policy": {
            "float32_matmul_precision": "highest",
            "route_weight_dtype": "FP32",
            "teacher_weight_dtype": "BF16",
            "candidate_weight_dtype": "FP16",
            "ldlq_tf32": True,
        },
        "model_downloads_required": False,
        "complete_bf16_checkpoint_required": False,
        "evidence_boundary": (
            "candidate-selection routed output SSE can choose a panel candidate; "
            "only untouched multi-document BF16-reference KLD can accept it"
        ),
    }
    manifest_sha256 = prepare_destination(dest, manifest, resume=False)
    uniform_records = _records_by_expert(uniform)
    refit_records = _records_by_expert(refit)
    k4_records = _records_by_expert(k4)
    source = IndexedTensorStore(source_root)
    quantizer_module = load_qsrt_encoder(exllamav3_root)
    install_sqg_quantizer(quantizer_module)
    records: list[dict[str, Any]] = []
    for ordinal, expert in enumerate(experts, start=1):
        started = time.monotonic()
        uniform_tensors = _load_intervention_tensors(
            Path(uniform["root"]), uniform_records[expert], candidate_prefix="qsrt_k3"
        )
        refit_tensors = _load_intervention_tensors(
            Path(refit["root"]), refit_records[expert], candidate_prefix="qsrt_k3"
        )
        k4_tensors = _load_intervention_tensors(
            Path(k4["root"]), k4_records[expert], candidate_prefix="qsrt_k4"
        )
        for projection in PROJECTION_NAMES:
            exl3_key = f"exl3.{projection}"
            if not torch.equal(uniform_tensors[exl3_key], refit_tensors[exl3_key]) or not torch.equal(uniform_tensors[exl3_key], k4_tensors[exl3_key]):
                raise ValueError(f"expert {expert} {projection} EXL3 endpoints differ")
        source_weights = {
            spec.name: source.get(source_tensor_name(layer, expert, spec.name))
            for spec in PROJECTIONS
        }
        source_gpu = {
            name: value.to(device=device, dtype=torch.bfloat16).contiguous()
            for name, value in source_weights.items()
        }
        uniform_gpu = {
            projection: uniform_tensors[f"qsrt_k3.{projection}"].to(device).half().contiguous()
            for projection in PROJECTION_NAMES
        }
        accepted = bool(refit_records[expert]["accepted"])
        down_target_sha256: str | None = None
        repeated_k3_payload: Mapping[str, Any] | None = None
        k4_down_payload: Mapping[str, Any] = k4_records[expert]["projections"]["down_proj"]["qsrt_k4"]["payload"]
        if accepted:
            fit_x_cpu, fit_route_weight_cpu = routed["activation_fit"][expert]
            fit_x = fit_x_cpu.to(device)
            teacher_fit = _teacher_output(
                fit_x,
                source_gpu["gate_proj"],
                source_gpu["up_proj"],
                source_gpu["down_proj"],
            )
            hidden_fit = _expert_hidden(
                fit_x, uniform_gpu["gate_proj"], uniform_gpu["up_proj"]
            )
            baseline_fit = F.linear(hidden_fit, uniform_gpu["down_proj"]).float()
            correction, _ = solve_down_correction(
                hidden_fit,
                teacher_fit - baseline_fit,
                fit_route_weight_cpu.to(device).float(),
                ridge_factor=float(refit_records[expert]["selected_ridge_factor"]),
            )
            down_target = uniform_gpu["down_proj"].float() + correction
            down_target_sha256 = tensor_sha256(down_target)
            down_spec = next(spec for spec in PROJECTIONS if spec.name == "down_proj")
            input_seed, output_seed = _transform_seeds(layer, down_spec)
            encodes: dict[int, dict[str, Any]] = {}
            for bits in (3, 4):
                encoded = encode_uniform_candidate(
                    down_target.cpu(),
                    bits=bits,
                    codebook=CODEBOOK_SQG_XOR_CHEB_T12,
                    device=device,
                    quantizer_module=quantizer_module,
                    input_sign_seed=input_seed,
                    output_sign_seed=output_seed,
                    sigma_reg=SIGMA_REG,
                    tailbite_context=128,
                    ldlq_tf32=True,
                )
                encodes[bits] = encoded
            repeated_k3 = encodes[3].pop("reconstruction").half().cpu()
            if not torch.equal(repeated_k3, refit_tensors["qsrt_k3.down_proj"]):
                raise RuntimeError(
                    f"expert {expert} repeated K3 down-refit encode differs from the stored refit"
                )
            repeated_k3_payload = encodes[3]["payload"]
            k4_down = encodes[4].pop("reconstruction").half().cpu()
            k4_down_payload = encodes[4]["payload"]
        else:
            if not torch.equal(
                refit_tensors["qsrt_k3.down_proj"],
                uniform_tensors["qsrt_k3.down_proj"],
            ):
                raise RuntimeError(f"expert {expert} rejected refit changed its K3 down tensor")
            k4_down = k4_tensors["qsrt_k4.down_proj"]

        pool_tensors: dict[str, torch.Tensor] = {}
        for projection in PROJECTION_NAMES:
            pool_tensors[f"exl3.{projection}"] = refit_tensors[f"exl3.{projection}"]
            pool_tensors[f"qsrt_k3.{projection}"] = refit_tensors[f"qsrt_k3.{projection}"]
            pool_tensors[f"qsrt_k4.{projection}"] = (
                k4_down if projection == "down_proj" else k4_tensors[f"qsrt_k4.{projection}"]
            )

        selection_x_cpu, selection_route_weight_cpu = routed["candidate_selection"][expert]
        selection_x = selection_x_cpu.to(device)
        selection_route_weights = selection_route_weight_cpu.to(device).float()
        teacher_selection = _teacher_output(
            selection_x,
            source_gpu["gate_proj"],
            source_gpu["up_proj"],
            source_gpu["down_proj"],
        )
        rate_candidates: list[dict[str, Any]] = []
        reference_sum: float | None = None
        for rates in RATE_TUPLES:
            weights = {
                projection: pool_tensors[f"qsrt_k{rate}.{projection}"].to(device).half().contiguous()
                for projection, rate in zip(PROJECTION_NAMES, rates, strict=True)
            }
            candidate_output = _teacher_output(
                selection_x,
                weights["gate_proj"],
                weights["up_proj"],
                weights["down_proj"],
            )
            error_sum, candidate_reference_sum = weighted_error_sums(
                teacher_selection, candidate_output, selection_route_weights
            )
            if reference_sum is None:
                reference_sum = candidate_reference_sum
            elif candidate_reference_sum != reference_sum:
                raise RuntimeError("candidate reference energy changed across rate tuples")
            rate_candidates.append(
                {
                    "key": _rate_key(rates),
                    "rates": dict(zip(PROJECTION_NAMES, rates, strict=True)),
                    "k4_projection_count": sum(rate == 4 for rate in rates),
                    "candidate_selection_weighted_error_sum": error_sum,
                    "candidate_selection_weighted_reference_sum": candidate_reference_sum,
                    "candidate_selection_weighted_relative_sse": error_sum / candidate_reference_sum,
                }
            )

        tensor_path = _pool_tensor_path(dest, layer, expert)
        _atomic_save_pool_tensors(tensor_path, pool_tensors)
        record = {
            "kind": f"{DOWN_REFIT_RATE_POOL_KIND}_expert",
            "complete": True,
            "manifest_sha256": manifest_sha256,
            "layer": layer,
            "expert": expert,
            "tensor_file": tensor_path.name,
            "tensor_file_bytes": tensor_path.stat().st_size,
            "tensor_file_sha256": sha256_file(tensor_path),
            "down_refit_accepted": accepted,
            "down_target": (
                {
                    "kind": "reconstructed_activation_refit",
                    "continuous_fp32_sha256": down_target_sha256,
                    "selected_ridge_factor": refit_records[expert]["selected_ridge_factor"],
                    "repeated_k3_matches_stored_refit": True,
                    "repeated_k3_payload": repeated_k3_payload,
                    "k4_payload": k4_down_payload,
                }
                if accepted
                else {
                    "kind": "source_target_fallback_after_refit_rejection",
                    "repeated_k3_matches_stored_refit": True,
                    "k4_payload": k4_down_payload,
                }
            ),
            "tensor_sha256": {
                prefix: {
                    projection: tensor_sha256(pool_tensors[f"{prefix}.{projection}"])
                    for projection in PROJECTION_NAMES
                }
                for prefix in ("exl3", *DOWN_REFIT_RATE_POOL_TENSOR_PREFIXES)
            },
            "candidate_selection_rows": int(selection_x.shape[0]),
            "rate_candidates": rate_candidates,
            "wall_seconds": time.monotonic() - started,
        }
        receipt_path = _expert_path(dest, layer, expert)
        atomic_write_json(receipt_path, record)
        record["receipt_sha256"] = sha256_file(receipt_path)
        # The report carries the receipt hash. The receipt itself cannot carry
        # its own hash without becoming self-referential.
        records.append(record)
        print(
            f"[{ordinal:02d}/{len(experts)}] layer {layer} expert {expert}: "
            f"refit={accepted} K3={rate_candidates[0]['candidate_selection_weighted_relative_sse']:.6g} "
            f"K4={rate_candidates[-1]['candidate_selection_weighted_relative_sse']:.6g}",
            flush=True,
        )
        torch.cuda.empty_cache()

    report = {
        "kind": DOWN_REFIT_RATE_POOL_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "layer": layer,
        "expert_count": len(records),
        "panel": manifest["panel"],
        "tensor_file_bytes": sum(record["tensor_file_bytes"] for record in records),
        "accepted_down_refit_count": sum(record["down_refit_accepted"] for record in records),
        "experts": records,
        "evidence_boundary": manifest["evidence_boundary"],
    }
    atomic_write_json(dest / "report.json", report)
    validate_down_refit_rate_pool(dest)
    return report


def merge_down_refit_rate_pool_slices(
    *,
    inputs: Sequence[Path],
    dest: Path,
    panel_manifest_path: Path,
    layer: int,
) -> dict[str, Any]:
    """Merge disjoint rate-pool slices without changing tensor bytes."""

    slices = [validate_down_refit_rate_pool(path) for path in inputs]
    if len(slices) < 2:
        raise ValueError("rate-pool merge requires at least two slices")
    common = slices[0]["manifest"]["common"]
    if any(item["manifest"]["common"] != common for item in slices):
        raise ValueError("rate-pool slices disagree on the common experiment contract")
    input_artifacts = slices[0]["manifest"].get("input_artifacts")
    if not isinstance(input_artifacts, dict):
        raise TypeError("rate-pool slice lacks its input-artifact identities")
    if any(
        item["manifest"].get("input_artifacts") != input_artifacts
        for item in slices
    ):
        raise ValueError("rate-pool slices disagree on their input artifacts")
    frozen = load_frozen_real_weight_panel(panel_manifest_path, layer=layer)
    records: dict[int, dict[str, Any]] = {}
    for item in slices:
        for record in item["report"]["experts"]:
            expert = int(record["expert"])
            if expert in records:
                raise ValueError(f"rate-pool slices repeat expert {expert}")
            records[expert] = dict(record)
    expected = tuple(frozen["experts"][: len(records)])
    if set(records) != set(expected):
        raise ValueError("merged rate-pool experts do not equal the frozen-panel prefix")
    manifest = {
        "kind": f"{DOWN_REFIT_RATE_POOL_KIND}_manifest",
        "composition": "verified_disjoint_rate_pool_slices",
        "common": common,
        "input_artifacts": input_artifacts,
        "input_slices": [
            {
                "root": item["root"],
                "manifest_sha256": item["manifest_sha256"],
                "experts": list(item["expert_ids"]),
            }
            for item in slices
        ],
        "frozen_panel": {
            "path": frozen["path"],
            "sha256": frozen["sha256"],
            "selected_offset": 0,
            "selected_count": len(records),
        },
        "panel": {str(layer): list(expected)},
        "model_downloads_required": False,
        "complete_bf16_checkpoint_required": False,
        "evidence_boundary": slices[0]["manifest"]["evidence_boundary"],
    }
    manifest_sha256 = prepare_destination(dest, manifest, resume=False)
    root_by_expert: dict[int, Path] = {}
    for item in slices:
        for expert in item["expert_ids"]:
            root_by_expert[expert] = Path(item["root"])
    merged_records: list[dict[str, Any]] = []
    for expert in expected:
        record = records[expert]
        source_root = root_by_expert[expert]
        source_tensor = _pool_tensor_path(source_root, layer, expert)
        destination_tensor = _pool_tensor_path(dest, layer, expert)
        try:
            os.link(source_tensor, destination_tensor)
        except OSError:
            shutil.copyfile(source_tensor, destination_tensor)
        record["manifest_sha256"] = manifest_sha256
        receipt = _expert_path(dest, layer, expert)
        receipt_record = {key: value for key, value in record.items() if key != "receipt_sha256"}
        atomic_write_json(receipt, receipt_record)
        record["receipt_sha256"] = sha256_file(receipt)
        merged_records.append(record)
    report = {
        "kind": DOWN_REFIT_RATE_POOL_KIND,
        "status": "complete",
        "composition": manifest["composition"],
        "manifest_sha256": manifest_sha256,
        "layer": layer,
        "expert_count": len(merged_records),
        "panel": manifest["panel"],
        "tensor_file_bytes": sum(record["tensor_file_bytes"] for record in merged_records),
        "accepted_down_refit_count": sum(record["down_refit_accepted"] for record in merged_records),
        "experts": merged_records,
        "evidence_boundary": manifest["evidence_boundary"],
    }
    atomic_write_json(dest / "report.json", report)
    validate_down_refit_rate_pool(dest)
    return report


def _rates_from_pool_selection(pool: Mapping[str, Any], expert_order: Sequence[int], maximum: int) -> dict[str, Any]:
    metrics: dict[int, dict[tuple[int, int, int], float]] = {}
    records = {int(record["expert"]): record for record in pool["report"]["experts"]}
    for expert in expert_order:
        metrics[expert] = {}
        for candidate in records[expert]["rate_candidates"]:
            rates = tuple(int(candidate["rates"][projection]) for projection in PROJECTION_NAMES)
            metrics[expert][rates] = float(candidate["candidate_selection_weighted_error_sum"])
    return select_pooled_rate_allocation(
        expert_order, metrics, maximum_k4_projection_count=maximum
    )


def _add_selection_metric_summary(
    allocation: dict[str, Any],
    pool: Mapping[str, Any],
    expert_order: Sequence[int],
) -> None:
    """Attach the frozen selection evidence for every materialized rate tuple."""

    records = {int(record["expert"]): record for record in pool["report"]["experts"]}
    selected_error_sum = 0.0
    uniform_k3_error_sum = 0.0
    reference_sum = 0.0
    selected_candidates: dict[str, dict[str, Any]] = {}
    for expert in expert_order:
        record = records[expert]
        by_rates = {
            tuple(
                int(candidate["rates"][projection])
                for projection in PROJECTION_NAMES
            ): candidate
            for candidate in record["rate_candidates"]
        }
        rates = tuple(
            int(allocation["rates_by_expert"][str(expert)][projection])
            for projection in PROJECTION_NAMES
        )
        selected = by_rates[rates]
        uniform = by_rates[(3, 3, 3)]
        selected_error_sum += float(selected["candidate_selection_weighted_error_sum"])
        uniform_k3_error_sum += float(uniform["candidate_selection_weighted_error_sum"])
        reference_sum += float(selected["candidate_selection_weighted_reference_sum"])
        selected_candidates[str(expert)] = {
            "key": selected["key"],
            "candidate_selection_weighted_error_sum": selected[
                "candidate_selection_weighted_error_sum"
            ],
            "candidate_selection_weighted_relative_sse": selected[
                "candidate_selection_weighted_relative_sse"
            ],
        }
    if reference_sum <= 0.0:
        raise ValueError("pooled candidate-selection reference energy is degenerate")
    allocation.update(
        {
            "candidate_selection_weighted_error_sum": selected_error_sum,
            "uniform_k3_candidate_selection_weighted_error_sum": uniform_k3_error_sum,
            "candidate_selection_weighted_reference_sum": reference_sum,
            "candidate_selection_weighted_relative_sse": selected_error_sum
            / reference_sum,
            "candidate_selection_error_reduction_from_uniform_k3": (
                1.0 - selected_error_sum / uniform_k3_error_sum
                if uniform_k3_error_sum > 0.0
                else None
            ),
            "selected_candidate_metrics_by_expert": selected_candidates,
        }
    )


def materialize_down_refit_rate_pool_allocation(
    *,
    rate_pool_root: Path,
    pre_registration_path: Path,
    allocation_kind: str,
    dest: Path,
) -> dict[str, Any]:
    """Materialize either the frozen fixed map or selection-data optimum."""

    if allocation_kind not in ("fixed_rate_stratified", "selection_data_complete_expert"):
        raise ValueError("allocation kind must name the fixed or selection-data rule")
    pool = validate_down_refit_rate_pool(rate_pool_root)
    if pool["model_layer"] != 3:
        raise ValueError(
            "the frozen layer-3 allocation contract cannot materialize another layer"
        )
    pre_registration = _read_json(pre_registration_path)
    if sha256_file(pre_registration_path) != RATE_PRESERVING_PRE_REGISTRATION_SHA256:
        raise ValueError("rate-preserving down-refit pre-registration SHA-256 mismatch")
    expert_order = tuple(int(expert) for expert in pre_registration["expert_panel"]["expert_order"])
    if set(expert_order) != set(pool["expert_ids"]):
        raise ValueError("rate pool does not cover the pre-registered panel")
    expected_base = pre_registration["base_representation"]
    common = pool["manifest"].get("common")
    if not isinstance(common, dict):
        raise TypeError("rate-pool manifest lacks its common experiment contract")
    if (
        common.get("down_refit_manifest_file_sha256")
        != expected_base["artifact_manifest_file_sha256"]
        or common.get("down_refit_report_file_sha256")
        != expected_base["artifact_report_file_sha256"]
    ):
        raise ValueError(
            "rate pool was not derived from the pre-registered K3 down-refit base"
        )
    maximum = int(pre_registration["rate_contract"]["maximum_k4_projection_count"])
    if allocation_kind == "fixed_rate_stratified":
        fixed = frozen_k4_rate_map(pre_registration)
        rates_by_expert = {
            str(expert): {
                projection: 4 if projection in fixed.get(expert, set()) else 3
                for projection in PROJECTION_NAMES
            }
            for expert in expert_order
        }
        allocation = {
            "expert_order": list(expert_order),
            "rates_by_expert": rates_by_expert,
            "k4_projection_count": sum(
                rate == 4
                for rates in rates_by_expert.values()
                for rate in rates.values()
            ),
            "candidate_measurements_used": False,
            "source": "pre_registered_EXL3_rate_stratification",
        }
    else:
        allocation = _rates_from_pool_selection(pool, expert_order, maximum)
        allocation["candidate_measurements_used"] = True
        allocation["source"] = "frozen_candidate_selection_complete_expert_output_sse"
    _add_selection_metric_summary(allocation, pool, expert_order)
    promoted = int(allocation["k4_projection_count"])
    if promoted > maximum:
        raise ValueError("materialized allocation exceeds the pre-registered K4 budget")
    base_bytes = int(pre_registration["base_representation"]["logical_bytes"])
    increment = int(pre_registration["rate_contract"]["k4_projection_increment_bytes"])
    logical_bytes = base_bytes + promoted * increment
    comparison_bytes = int(pre_registration["logical_byte_gate"]["comparison_exl3_bytes"])
    if logical_bytes >= comparison_bytes:
        raise ValueError("materialized allocation is not logically smaller than EXL3")
    manifest = {
        "kind": f"{INTERVENTION_ARTIFACT_KIND}_manifest",
        "candidate": {
            "profile": "qsrt_sqg_e4m3",
            "tensor_prefix": "candidate",
            "variant": "mixed_k3_k4_with_rate_preserving_reconstructed_activation_down_refit",
            "allocation_kind": allocation_kind,
            "candidate_measurements_used": allocation["candidate_measurements_used"],
        },
        "rate_pool": {"root": pool["root"], "manifest_sha256": pool["manifest_sha256"]},
        "pre_registration": {
            "path": str(pre_registration_path.resolve()),
            "sha256": sha256_file(pre_registration_path),
        },
        "allocation": allocation,
        "panel": {"3": list(expert_order)},
        "logical_byte_accounting": {
            "base_k3_bytes": base_bytes,
            "k4_projection_increment_bytes": increment,
            "k4_projection_count": promoted,
            "mixed_qsrt_bytes": logical_bytes,
            "comparison_exl3_bytes": comparison_bytes,
            "logical_margin_bytes": comparison_bytes - logical_bytes,
            "serialized_container_gate_passed": False,
        },
        "resident_endpoint_dtype": "FP16",
        "coordinate_bases": {
            "exl3": "sealed_R7_permuted_middle_coordinates",
            "candidate": "official_source_middle_coordinates",
        },
        "model_downloads_required": False,
        "complete_bf16_checkpoint_required": False,
        "evidence_boundary": (
            "the allocation is frozen from candidate-selection documents; only "
            "untouched multi-document BF16-reference KLD can accept it, and "
            "logical bytes are not complete serialized container bytes"
        ),
    }
    manifest_sha256 = prepare_destination(dest, manifest, resume=False)
    pool_records = {int(record["expert"]): record for record in pool["report"]["experts"]}
    records: list[dict[str, Any]] = []
    for expert in expert_order:
        pool_record = pool_records[expert]
        pool_path = _pool_tensor_path(Path(pool["root"]), 3, expert)
        rates = allocation["rates_by_expert"][str(expert)]
        with safe_open(pool_path, framework="pt", device="cpu") as handle:
            output = {
                f"exl3.{projection}": handle.get_tensor(f"exl3.{projection}")
                for projection in PROJECTION_NAMES
            }
            output.update(
                {
                    f"candidate.{projection}": handle.get_tensor(
                        f"qsrt_k{int(rates[projection])}.{projection}"
                    )
                    for projection in PROJECTION_NAMES
                }
            )
        output_path = _dense_expert_path(dest, 3, expert)
        _atomic_save_tensors(output_path, output)
        record = {
            "kind": f"{INTERVENTION_ARTIFACT_KIND}_expert",
            "complete": True,
            "manifest_sha256": manifest_sha256,
            "layer": 3,
            "expert": expert,
            "dense_endpoint_file": output_path.name,
            "dense_endpoint_file_bytes": output_path.stat().st_size,
            "dense_endpoint_file_sha256": sha256_file(output_path),
            "rates": rates,
            "down_target_kind": pool_record["down_target"]["kind"],
            "rate_pool_tensor_sha256": pool_record["tensor_file_sha256"],
            "candidate_tensor_sha256": {
                projection: tensor_sha256(output[f"candidate.{projection}"])
                for projection in PROJECTION_NAMES
            },
        }
        atomic_write_json(_expert_path(dest, 3, expert), record)
        records.append(record)
    report = {
        "kind": INTERVENTION_ARTIFACT_KIND,
        "experiment": "qsrt_glm52_mixed_k3_k4_rate_preserving_down_refit_v1",
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "layer": 3,
        "expert_count": len(records),
        "panel": manifest["panel"],
        "allocation_kind": allocation_kind,
        "k4_projection_count": promoted,
        "logical_byte_accounting": manifest["logical_byte_accounting"],
        "dense_endpoint_bytes": sum(record["dense_endpoint_file_bytes"] for record in records),
        "experts": records,
        "evidence_boundary": manifest["evidence_boundary"],
    }
    atomic_write_json(dest / "report.json", report)
    validate_dense_intervention_artifact(dest)
    return report


def materialize_registered_partial_rate_map(
    *,
    rate_pool_root: Path,
    registration_path: Path,
    dest: Path,
) -> dict[str, Any]:
    """Materialize a pre-registered partial replacement inside an EXL3 panel.

    Experts absent from the registration remain resident EXL3 experts. The
    logical-byte ledger therefore starts from the complete comparison panel,
    removes the selected experts' EXL3 rates, and inserts their registered
    QSRT rates. This differs from an all-panel uniform-K3 accounting shortcut.
    """

    pool = validate_down_refit_rate_pool(rate_pool_root)
    registration_path = registration_path.resolve(strict=True)
    registration = _read_json(registration_path)
    if registration.get("schema") != REGISTERED_PARTIAL_RATE_MAP_SCHEMA:
        raise ValueError("registered partial rate-map schema mismatch")
    if registration.get("schema_version") != 1:
        raise ValueError("registered partial rate-map version mismatch")
    if registration.get("status") != "frozen_before_candidate_k4_measurement":
        raise ValueError("partial rate map was not frozen before K4 measurement")
    layer = _model_layer(registration.get("model_layer"))
    if pool["model_layer"] != layer:
        raise ValueError("rate pool and registered partial rate map use different layers")

    common = pool["manifest"].get("common")
    if not isinstance(common, dict):
        raise TypeError("rate-pool manifest lacks its common experiment contract")
    panel = registration.get("comparison_panel")
    if not isinstance(panel, dict):
        raise TypeError("registration comparison_panel must be an object")
    expert_order = panel.get("expert_order")
    if (
        not isinstance(expert_order, list)
        or any(type(expert) is not int for expert in expert_order)
        or len(expert_order) != len(set(expert_order))
        or set(expert_order) != set(pool["expert_ids"])
    ):
        raise ValueError("registration must cover the complete rate-pool panel")
    if (
        _sha256_identity(panel.get("manifest_sha256"), name="panel manifest")
        != common.get("frozen_panel_sha256")
    ):
        raise ValueError("registration and rate pool use different frozen panels")

    base = registration.get("down_refit_base")
    if not isinstance(base, dict):
        raise TypeError("registration down_refit_base must be an object")
    base_manifest_sha256 = _sha256_identity(
        base.get("manifest_file_sha256"), name="down-refit manifest file"
    )
    base_report_sha256 = _sha256_identity(
        base.get("report_file_sha256"), name="down-refit report file"
    )
    if (
        base_manifest_sha256 != common.get("down_refit_manifest_file_sha256")
        or base_report_sha256 != common.get("down_refit_report_file_sha256")
    ):
        raise ValueError("rate pool was not derived from the registered down-refit base")

    comparison_rates = registration.get("comparison_exl3_rates")
    if not isinstance(comparison_rates, dict) or set(comparison_rates) != {
        str(expert) for expert in expert_order
    }:
        raise ValueError("registration must provide EXL3 rates for every panel expert")

    replacements = registration.get("registered_replacements")
    if not isinstance(replacements, list) or not replacements:
        raise ValueError("registration has no partial replacements")
    rates_by_expert: dict[int, dict[str, int]] = {}
    for replacement in replacements:
        if not isinstance(replacement, dict):
            raise TypeError("registered replacement must be an object")
        expert = replacement.get("expert")
        if type(expert) is not int or expert not in pool["expert_ids"]:
            raise ValueError("registered replacement expert is outside the rate pool")
        if expert in rates_by_expert:
            raise ValueError("registered replacement repeats an expert")
        rates = replacement.get("candidate_rates")
        if not isinstance(rates, dict) or set(rates) != set(PROJECTION_NAMES):
            raise ValueError("registered replacement must assign every projection rate")
        normalized_rates = {name: int(rates[name]) for name in PROJECTION_NAMES}
        if tuple(normalized_rates[name] for name in PROJECTION_NAMES) not in RATE_TUPLES:
            raise ValueError("registered replacement rates must be K3 or K4")
        rates_by_expert[expert] = normalized_rates

    def normalized_comparison_rates(expert: int) -> dict[str, int]:
        value = comparison_rates[str(expert)]
        if not isinstance(value, dict) or set(value) != set(PROJECTION_NAMES):
            raise ValueError("comparison EXL3 rate entry has an invalid schema")
        normalized = {name: int(value[name]) for name in PROJECTION_NAMES}
        if any(rate not in (3, 4, 5) for rate in normalized.values()):
            raise ValueError("comparison EXL3 rates must be K3, K4, or K5")
        return normalized

    comparison_rate_sum = sum(
        sum(normalized_comparison_rates(expert).values()) for expert in expert_order
    )
    candidate_rate_sum = comparison_rate_sum
    for expert, rates in rates_by_expert.items():
        candidate_rate_sum -= sum(normalized_comparison_rates(expert).values())
        candidate_rate_sum += sum(rates.values())

    byte_contract = registration.get("logical_byte_contract")
    if not isinstance(byte_contract, dict):
        raise TypeError("registration logical_byte_contract must be an object")
    uniform_k3_bytes = int(byte_contract["uniform_k3_panel_bytes"])
    rate_increment_bytes = int(byte_contract["one_projection_bit_increment_bytes"])
    projection_count = len(expert_order) * len(PROJECTION_NAMES)
    comparison_bytes = uniform_k3_bytes + (
        comparison_rate_sum - 3 * projection_count
    ) * rate_increment_bytes
    candidate_bytes = uniform_k3_bytes + (
        candidate_rate_sum - 3 * projection_count
    ) * rate_increment_bytes
    if (
        comparison_bytes != int(byte_contract["comparison_exl3_panel_bytes"])
        or candidate_bytes != int(byte_contract["registered_candidate_panel_bytes"])
    ):
        raise ValueError("registered logical-byte ledger does not close")
    if candidate_bytes >= comparison_bytes:
        raise ValueError("registered partial candidate is not smaller than EXL3")

    pool_records = {
        int(record["expert"]): record for record in pool["report"]["experts"]
    }
    selected_experts = tuple(
        expert for expert in expert_order if expert in rates_by_expert
    )
    candidate_construction = registration.get("candidate_construction")
    allow_source_target_fallback = bool(
        isinstance(candidate_construction, dict)
        and candidate_construction.get("allow_source_target_fallback") is True
    )
    for expert in selected_experts:
        if (
            not bool(pool_records[expert]["down_refit_accepted"])
            and not allow_source_target_fallback
        ):
            raise ValueError(
                f"registered expert {expert} did not retain its down-refit target"
            )

    input_artifacts = pool["manifest"].get("input_artifacts")
    down_refit_input = (
        input_artifacts.get("down_refit")
        if isinstance(input_artifacts, dict)
        else None
    )
    if not isinstance(down_refit_input, dict):
        raise TypeError("rate-pool manifest lacks its down-refit input artifact")
    manifest = {
        "kind": f"{INTERVENTION_ARTIFACT_KIND}_manifest",
        "candidate": {
            "profile": "qsrt_sqg_e4m3",
            "tensor_prefix": "candidate",
            "variant": "registered_partial_k3_k4_reconstructed_activation_down_refit",
            "candidate_k4_measurements_used": False,
        },
        "input_intervention_artifact": {
            "root": down_refit_input.get("root"),
            "manifest_sha256": base_manifest_sha256,
            "report_sha256": base_report_sha256,
        },
        "rate_pool": {
            "root": pool["root"],
            "manifest_sha256": pool["manifest_sha256"],
        },
        "registration": {
            "path": str(registration_path),
            "sha256": sha256_file(registration_path),
        },
        "panel": {str(layer): list(selected_experts)},
        "rates_by_expert": {
            str(expert): rates_by_expert[expert] for expert in selected_experts
        },
        "logical_byte_accounting": {
            "scope": "complete registered comparison panel",
            "comparison_exl3_rate_sum": comparison_rate_sum,
            "registered_candidate_rate_sum": candidate_rate_sum,
            "uniform_k3_panel_bytes": uniform_k3_bytes,
            "one_projection_bit_increment_bytes": rate_increment_bytes,
            "comparison_exl3_panel_bytes": comparison_bytes,
            "registered_candidate_panel_bytes": candidate_bytes,
            "logical_margin_bytes": comparison_bytes - candidate_bytes,
            "serialized_container_gate_passed": False,
        },
        "resident_endpoint_dtype": "FP16",
        "coordinate_bases": {
            "exl3": "sealed_R7_permuted_middle_coordinates",
            "candidate": "official_source_middle_coordinates",
        },
        "model_downloads_required": False,
        "complete_bf16_checkpoint_required": False,
        "evidence_boundary": (
            "the rate map was frozen before K4 candidate measurement; only "
            "document-disjoint model KLD can screen it, and complete serialized "
            "container bytes remain unmeasured"
        ),
    }
    manifest_sha256 = prepare_destination(dest, manifest, resume=False)
    records: list[dict[str, Any]] = []
    for expert in selected_experts:
        pool_record = pool_records[expert]
        pool_path = _pool_tensor_path(Path(pool["root"]), layer, expert)
        rates = rates_by_expert[expert]
        with safe_open(pool_path, framework="pt", device="cpu") as handle:
            output = {
                f"exl3.{projection}": handle.get_tensor(f"exl3.{projection}")
                for projection in PROJECTION_NAMES
            }
            output.update(
                {
                    f"candidate.{projection}": handle.get_tensor(
                        f"qsrt_k{rates[projection]}.{projection}"
                    )
                    for projection in PROJECTION_NAMES
                }
            )
        output_path = _dense_expert_path(dest, layer, expert)
        _atomic_save_tensors(output_path, output)
        selected_candidate = next(
            candidate
            for candidate in pool_record["rate_candidates"]
            if candidate["key"]
            == _rate_key(tuple(rates[name] for name in PROJECTION_NAMES))
        )
        record = {
            "kind": f"{INTERVENTION_ARTIFACT_KIND}_expert",
            "complete": True,
            "manifest_sha256": manifest_sha256,
            "layer": layer,
            "expert": expert,
            "dense_endpoint_file": output_path.name,
            "dense_endpoint_file_bytes": output_path.stat().st_size,
            "dense_endpoint_file_sha256": sha256_file(output_path),
            "rates": rates,
            "down_target_kind": pool_record["down_target"]["kind"],
            "rate_pool_tensor_sha256": pool_record["tensor_file_sha256"],
            "candidate_selection_metric": selected_candidate,
            "candidate_tensor_sha256": {
                projection: tensor_sha256(output[f"candidate.{projection}"])
                for projection in PROJECTION_NAMES
            },
        }
        atomic_write_json(_expert_path(dest, layer, expert), record)
        records.append(record)
    report = {
        "kind": INTERVENTION_ARTIFACT_KIND,
        "experiment": "qsrt_glm52_registered_partial_k3_k4_down_refit_v1",
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "layer": layer,
        "expert_count": len(records),
        "panel": manifest["panel"],
        "logical_byte_accounting": manifest["logical_byte_accounting"],
        "dense_endpoint_bytes": sum(
            record["dense_endpoint_file_bytes"] for record in records
        ),
        "experts": records,
        "evidence_boundary": manifest["evidence_boundary"],
    }
    atomic_write_json(dest / "report.json", report)
    validate_dense_intervention_artifact(dest)
    return report


__all__ = [
    "DOWN_REFIT_RATE_POOL_KIND",
    "RATE_PRESERVING_PRE_REGISTRATION_SHA256",
    "RATE_TUPLES",
    "build_down_refit_rate_pool_slice",
    "materialize_registered_partial_rate_map",
    "materialize_down_refit_rate_pool_allocation",
    "merge_down_refit_rate_pool_slices",
    "select_pooled_rate_allocation",
    "validate_down_refit_rate_pool",
    "weighted_error_sums",
]

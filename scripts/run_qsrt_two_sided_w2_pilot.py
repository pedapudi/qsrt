#!/usr/bin/env python3
"""Encode or compare canonical uniform-K2 W2 payloads.

The gate and up projections are decoded from the sealed uniform-K2 candidate
pool.  Their post-SiTU activations define the expert-local input Hessian for
W2.  The output Hessian is either the exact local
RMSNorm/output-projection Jacobian Gram or a supplied empirical Fisher factor
at the routed W2 output.  The canonical W2 target is then re-encoded with
one-sided BlockLDLQ and two-sided BaKron under identical preparation.  Both
are scored against the exact served W2 reconstruction.

The direct payload mode minimizes independent tile distortion. Curvature modes
compare one-sided and two-sided rounding while keeping the decoded-upstream
input Hessian frozen. The script emits an overlay and does not materialize a
checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file as load_torch_file, save_file

from qsrt import constants as C
from qsrt.coupled_expert_study import (
    CoupledTriplet,
    RoutedOutputMetric,
    expert_hidden,
)
from qsrt.distortion_transfer import (
    mapped_output_hessian,
    quadratic_matrix_sse,
    routed_error_geometry,
    two_sided_encoder_sse,
)
from qsrt.exl3_loader import load_qsrt_encoder
from qsrt.io.stream import load_tensor
from qsrt.kimi_layer92_fisher import (
    PairedLayer92FisherSamples,
    load_paired_layer92_fisher_samples,
)
from qsrt.kimi_output_factors import (
    KIND as OUTPUT_FACTOR_ARCHIVE_KIND,
    KimiOutputFactorArchive,
)
from qsrt.pack.qsrt_validation import decode_candidate_matrix
from qsrt.pack.qsrt_encoder import default_qsrt_transform_seeds
from qsrt.pack.qsrt_encoder import (
    QSRTMatrixCandidate,
    finalize_qsrt_matrix_candidate,
    plan_qsrt_matrix,
)
from qsrt.pack.qsrt_candidates import candidate_tensor_name
from qsrt.candidate_hessian import adaptive_identity_shrinkage
from qsrt.pooled_calibration import collect_coupled_hidden_statistics
from qsrt.qsrt_candidates import partition_requests
from qsrt.qsrt import K2, SCHEMA as QSRT_SCHEMA, matrix_rate_axis
from qsrt.qsrt_atoms_v2 import unpack_atoms_v2_format_section
from qsrt.qsrt_coupled import (
    CoupledHadamardExecution,
    CoupledHadamardSpec,
    encode_coupled_down_weight,
    encode_coupled_weights,
)
from qsrt.source_weights import OfficialMXFP4Store
from qsrt.sqg_quantizer import install_sqg_quantizer
from qsrt.suffix_fisher import (
    sketch_a_input_factor_update,
    sketch_a_output_factor_update,
)
from qsrt.two_sided_qsrt import (
    UniformSQGCandidate,
    encode_uniform_sqg_baseline,
    encode_uniform_sqg_direct_batch,
    encode_uniform_sqg_two_sided_batch,
    encode_uniform_sqg_two_sided_pair,
)


KIND = "qsrt_uniform_k2_two_sided_w2_pilot"
SCHEMA_VERSION = 1
DEFAULT_FIT_CACHE = Path(
    "/data/datasets/kquant/captures/"
    "k3-denseh-broad-v7-4m-train-input-v1.kqsamples"
)
DEFAULT_FIT_REPORT = Path(
    "/home/luke/projects/qsrt/out/"
    "k3-denseh-broad-v7-4m-train-corpus.json"
)
DEFAULT_EVALUATION = Path(
    "/data/datasets/kquant/captures/"
    "k3-codec-diverse-validation-v3-128k-input-v1.kqsamples"
)
DEFAULT_EVALUATION_REPORT = Path(
    "/home/luke/projects/qsrt/out/"
    "k3-codec-diverse-validation-v3-128k-corpus.json"
)
DEFAULT_PROFILE = Path(
    "/data/releases/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-v1"
)
DEFAULT_CANDIDATE_POOL = Path(
    "/data/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-CANDIDATES-v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _report_hashes(
    report_path: Path,
) -> tuple[set[str], set[str], dict[str, object]]:
    report = json.loads(report_path.read_text())
    documents = report.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"corpus report has no document inventory: {report_path}")
    document_hashes: list[str] = []
    prompt_hashes: list[str] = []
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"corpus report document {index} is not an object")
        for key, destination in (
            ("document_hash", document_hashes),
            ("prompt_hash", prompt_hashes),
        ):
            value = document.get(key)
            if not isinstance(value, str) or len(value) != 32:
                raise ValueError(
                    f"corpus report document {index} has an invalid {key}"
                )
            try:
                bytes.fromhex(value)
            except ValueError as exc:
                raise ValueError(
                    f"corpus report document {index} has an invalid {key}"
                ) from exc
            destination.append(value)
    return (
        set(document_hashes),
        set(prompt_hashes),
        {
            "report": str(report_path.resolve()),
            "report_sha256": _sha256(report_path),
            "planned_entries": len(documents),
            "unique_documents": len(set(document_hashes)),
            "unique_prompts": len(set(prompt_hashes)),
        },
    )


def _corpus_hashes(
    report_path: Path,
    cache_path: Path,
) -> tuple[set[str], set[str], dict[str, object]]:
    document_hashes, prompt_hashes, identity = _report_hashes(report_path)
    cache_manifest = json.loads((cache_path / "manifest.json").read_text())
    source_capture = Path(str(cache_manifest["source_capture"]))
    if not source_capture.exists():
        moved_capture = cache_path.parent / source_capture.name
        if not moved_capture.exists():
            raise FileNotFoundError(source_capture)
        source_capture = moved_capture
    capture_manifest = json.loads((source_capture / "manifest.json").read_text())
    report = json.loads(report_path.read_text())
    report_capture = report.get("capture_manifest_after_run")
    if not isinstance(report_capture, dict):
        raise ValueError(f"corpus report lacks its completed capture: {report_path}")
    run_id = capture_manifest.get("run_id")
    if report_capture.get("run_id") != run_id:
        raise ValueError(
            f"corpus report and sample-cache capture disagree: {report_path}"
        )
    identity.update(
        {
            "sample_cache": str(cache_path.resolve()),
            "source_capture": str(source_capture.resolve()),
            "capture_run_id": run_id,
        }
    )
    return document_hashes, prompt_hashes, identity


def _verify_corpus_disjointness(
    *,
    fit_report: Path,
    fit_cache: Path,
    evaluation_report: Path,
    evaluation_cache: Path,
) -> dict[str, object]:
    fit_documents, fit_prompts, fit = _corpus_hashes(fit_report, fit_cache)
    evaluation_documents, evaluation_prompts, evaluation = _corpus_hashes(
        evaluation_report, evaluation_cache
    )
    shared_documents = fit_documents & evaluation_documents
    shared_prompts = fit_prompts & evaluation_prompts
    if shared_documents:
        raise ValueError(
            "construction and evaluation corpora share document content hashes"
        )
    if shared_prompts:
        raise ValueError(
            "construction and evaluation corpora share tokenized prompt hashes"
        )
    return {
        "fit": fit,
        "evaluation": evaluation,
        "shared_document_hashes": 0,
        "shared_prompt_hashes": 0,
    }


def _verify_output_factor_evaluation_disjointness(
    *,
    archive: KimiOutputFactorArchive,
    evaluation_report: Path,
) -> dict[str, object]:
    provenance = archive.manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("output-factor archive has no provenance")
    cotangent_manifest_path = Path(str(provenance["cotangent_manifest"]))
    cotangent_manifest = json.loads(cotangent_manifest_path.read_text())
    boundary_manifest_path = Path(str(cotangent_manifest["boundary_manifest"]))
    boundary_manifest_sha256 = _sha256(boundary_manifest_path)
    if boundary_manifest_sha256 != provenance.get("boundary_manifest_sha256"):
        raise ValueError("output-factor and boundary manifests disagree")
    boundary_manifest = json.loads(boundary_manifest_path.read_text())
    boundary_provenance = boundary_manifest.get("provenance")
    if not isinstance(boundary_provenance, Mapping):
        raise ValueError("boundary archive has no provenance")
    factor_report_path = Path(str(boundary_provenance["corpus_report"]))
    if _sha256(factor_report_path) != boundary_provenance.get(
        "corpus_report_sha256"
    ):
        raise ValueError("boundary archive and Fisher corpus report disagree")
    factor_documents, factor_prompts, factor_identity = _report_hashes(
        factor_report_path
    )
    evaluation_documents, evaluation_prompts, evaluation_identity = (
        _report_hashes(evaluation_report)
    )
    shared_documents = factor_documents & evaluation_documents
    shared_prompts = factor_prompts & evaluation_prompts
    if shared_documents:
        raise ValueError(
            "output-factor and evaluation corpora share document content hashes"
        )
    if shared_prompts:
        raise ValueError(
            "output-factor and evaluation corpora share tokenized prompt hashes"
        )
    return {
        "factor": factor_identity,
        "evaluation": evaluation_identity,
        "boundary_manifest": str(boundary_manifest_path.resolve()),
        "boundary_manifest_sha256": boundary_manifest_sha256,
        "cotangent_manifest": str(cotangent_manifest_path.resolve()),
        "cotangent_manifest_sha256": _sha256(cotangent_manifest_path),
        "shared_document_hashes": 0,
        "shared_prompt_hashes": 0,
    }


def _request_partition(report_path: Path):
    report = json.loads(report_path.read_text())
    documents = report.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"corpus report has no document inventory: {report_path}")
    requests: dict[int, str] = {}
    for request_step, document in enumerate(documents, start=1):
        if not isinstance(document, dict):
            raise ValueError(f"corpus report document {request_step} is not an object")
        document_hash = document.get("document_hash")
        if not isinstance(document_hash, str) or not document_hash:
            raise ValueError(
                f"corpus report document {request_step} has no content hash"
            )
        requests[request_step] = document_hash
    if len(set(requests.values())) != len(requests):
        raise ValueError("construction corpus contains duplicate document hashes")
    return partition_requests(requests)


def _parse_experts(value: str) -> tuple[int, ...]:
    try:
        experts = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "experts must be comma-separated integers"
        ) from exc
    if not experts or len(set(experts)) != len(experts):
        raise argparse.ArgumentTypeError("experts must be nonempty and unique")
    if any(not 0 <= expert < C.NUM_EXPERTS for expert in experts):
        raise argparse.ArgumentTypeError("expert lies outside Kimi-K3 geometry")
    return experts


def _resolved_candidate_schema(manifest: Mapping[str, object]) -> tuple[str, str]:
    source = str(manifest.get("logical_trellis_schema", ""))
    resolved = {
        "kquant_kimi_k3_qsrt_candidate_v1": QSRT_SCHEMA,
        QSRT_SCHEMA: QSRT_SCHEMA,
    }.get(source)
    if resolved is None:
        raise ValueError(f"candidate pool uses an unsupported schema: {source}")
    return source, resolved


def _profile_draws(
    profile: Path,
    layer: int,
    experts: tuple[int, ...],
) -> dict[int, int]:
    layer_path = profile / f"qsrt-layer-{layer:05d}.safetensors"
    with safe_open(layer_path, framework="pt", device="cpu") as reader:
        metadata = reader.metadata()
        if metadata is None or "profile" not in metadata:
            raise ValueError("profile layer lacks the atoms-v2 format identity")
        formats, draws = unpack_atoms_v2_format_section(
            str(metadata["profile"]), reader.get_tensor("_qsrt_format_section")
        )
    if draws is None:
        raise ValueError("profile layer lacks coupled-Hadamard draws")
    result: dict[int, int] = {}
    for expert in experts:
        if formats[expert] != "K2":
            raise ValueError(f"profile expert {expert} is not uniform K2")
        result[expert] = int(draws[expert])
    return result


def _fit_batches(
    rows: Mapping[str, torch.Tensor],
    *,
    maximum_rows: int,
    batch_rows: int,
) -> Iterable[dict[str, torch.Tensor]]:
    count = int(rows["row_index"].numel())
    if maximum_rows == 0 or count <= maximum_rows:
        selected = torch.arange(count)
    else:
        selected = torch.linspace(0, count - 1, maximum_rows).round().long()
    for begin in range(0, selected.numel(), batch_rows):
        index = selected[begin : begin + batch_rows]
        yield {
            name: value.index_select(0, index)
            for name, value in rows.items()
            if name in {"input", "row_index", "route_slot", "route_weight"}
        }


def _fit_rows(
    samples: Any,
    expert: int,
    maximum_rows: int,
    fit_requests: Mapping[int, str],
) -> dict[str, torch.Tensor]:
    steps = torch.bitwise_right_shift(samples.input_observations, 32)
    allowed = torch.tensor(sorted(fit_requests), dtype=steps.dtype)
    matches = samples.input_experts == expert
    selected = matches.any(dim=1) & torch.isin(steps, allowed)
    rows = torch.nonzero(selected, as_tuple=False).flatten()
    route_weights = (samples.input_gates[selected] * matches[selected]).sum(dim=1)
    if maximum_rows and rows.numel() > maximum_rows:
        retained = torch.linspace(0, rows.numel() - 1, maximum_rows).round().long()
        rows = rows.index_select(0, retained)
        route_weights = route_weights.index_select(0, retained)
    return {
        "input": samples.input_values.index_select(0, rows).float(),
        "row_index": rows.long(),
        "route_weight": route_weights.float(),
        "document": steps.index_select(0, rows),
    }


def _index_fit_rows(
    samples: Any,
    maximum_rows: int,
    fit_requests: Mapping[int, str],
) -> dict[int, dict[str, torch.Tensor]]:
    """Index routed rows once without copying the dense expert inputs."""
    steps = torch.bitwise_right_shift(samples.input_observations, 32)
    allowed = torch.tensor(sorted(fit_requests), dtype=steps.dtype)
    eligible_rows = torch.nonzero(
        torch.isin(steps, allowed), as_tuple=False
    ).flatten()
    experts = samples.input_experts.index_select(0, eligible_rows)
    route_slots = int(experts.shape[1])
    flat_experts = experts.reshape(-1).long()
    if flat_experts.numel() == 0:
        return {
            expert: {
                "row_index": torch.empty(0, dtype=torch.long),
                "route_weight": torch.empty(0, dtype=torch.float32),
                "document": torch.empty(0, dtype=steps.dtype),
            }
            for expert in range(C.NUM_EXPERTS)
        }
    if int(flat_experts.min()) < 0 or int(flat_experts.max()) >= C.NUM_EXPERTS:
        raise ValueError("fit cache contains an out-of-range routed expert")

    order = torch.argsort(flat_experts, stable=True)
    counts = torch.bincount(flat_experts, minlength=C.NUM_EXPERTS)
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), counts.cumsum(dim=0))
    )
    indexed: dict[int, dict[str, torch.Tensor]] = {}
    for expert in range(C.NUM_EXPERTS):
        positions = order[offsets[expert] : offsets[expert + 1]]
        local_rows = torch.div(positions, route_slots, rounding_mode="floor")
        slots = torch.remainder(positions, route_slots)
        rows = eligible_rows.index_select(0, local_rows)
        route_weights = samples.input_gates[rows, slots].float()

        # Top-k routing normally contains each expert at most once. Preserve the
        # previous sum-over-slots behavior if malformed or synthetic data does
        # contain a duplicate expert in one row.
        if rows.numel() > 1 and bool(torch.any(rows[1:] == rows[:-1])):
            unique_rows, inverse = torch.unique_consecutive(
                rows, return_inverse=True
            )
            summed_weights = torch.zeros(
                unique_rows.numel(), dtype=route_weights.dtype
            )
            summed_weights.scatter_add_(0, inverse, route_weights)
            rows = unique_rows
            route_weights = summed_weights

        if maximum_rows and rows.numel() > maximum_rows:
            retained = (
                torch.linspace(0, rows.numel() - 1, maximum_rows)
                .round()
                .long()
            )
            rows = rows.index_select(0, retained)
            route_weights = route_weights.index_select(0, retained)
        indexed[expert] = {
            "row_index": rows.long(),
            "route_weight": route_weights,
            "document": steps.index_select(0, rows),
        }
    return indexed


def _materialize_fit_rows(
    samples: Any,
    indexed: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    rows = indexed["row_index"]
    return {
        "input": samples.input_values.index_select(0, rows).float(),
        "row_index": rows,
        "route_weight": indexed["route_weight"],
        "document": indexed["document"],
    }


def _sample_layer_outputs(
    samples: Any, maximum_rows: int, fit_requests: Mapping[int, str]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    steps = torch.bitwise_right_shift(samples.input_observations, 32)
    allowed = torch.tensor(sorted(fit_requests), dtype=steps.dtype)
    eligible = torch.nonzero(torch.isin(steps, allowed), as_tuple=False).flatten()
    if eligible.numel() > maximum_rows:
        selected = torch.linspace(0, eligible.numel() - 1, maximum_rows).round().long()
        eligible = eligible.index_select(0, selected)
    return (
        samples.routed_latent.index_select(0, eligible).contiguous(),
        steps.index_select(0, eligible).contiguous(),
        eligible.contiguous(),
    )


def _production_h2(statistics, route_weights: torch.Tensor):
    covariance = statistics.candidate_gram / statistics.weight_sum
    covariance = ((covariance + covariance.T) * 0.5).contiguous()
    h2, evidence = adaptive_identity_shrinkage(
        covariance,
        route_weights.float().square().to(covariance.device),
        max_local_alpha=0.75,
    )
    return h2, {
        "rows": statistics.rows,
        "gate_square_sum": statistics.weight_sum,
        "h2_effective_sample_size": evidence["effective_sample_size"],
        "h2_oas_shrinkage": evidence["oas_shrinkage"],
        "h2_local_alpha": evidence["local_alpha"],
        "h2_local_alpha_cap": evidence["max_local_alpha"],
        "h2_identity_scale": evidence["identity_scale"],
        "h2_shrinkage_policy": "weighted_oas_scaled_identity",
    }


def _identity_input_factor(
    dimension: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, object]]:
    return torch.eye(dimension, dtype=torch.float32, device=device), {
        "rows": 0,
        "gate_square_sum": 0.0,
        "h2_effective_sample_size": None,
        "h2_oas_shrinkage": None,
        "h2_local_alpha": None,
        "h2_local_alpha_cap": None,
        "h2_identity_scale": 1.0,
        "h2_shrinkage_policy": "identity_input_factor",
    }


def _load_evaluation_samples(cache: Path, layer: int) -> Any:
    manifest = json.loads((cache / "manifest.json").read_text())
    entry = manifest.get("layers", {}).get(str(layer))
    if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
        raise ValueError(f"evaluation cache has no decoder layer {layer}")
    tensors = load_torch_file(str(cache / entry["file"]), device="cpu")
    return SimpleNamespace(
        input_values=tensors["input.values"],
        input_observations=tensors["input.observation"],
        input_experts=tensors["input.experts"],
        input_gates=tensors["input.gates"],
        routed_latent=tensors["input.routed_latent"],
        input_split=tensors["input.split"],
    )


def _evaluation_rows(
    samples: Any,
    expert: int,
    maximum_rows: int,
) -> dict[str, torch.Tensor]:
    locations = torch.nonzero(samples.input_experts == expert, as_tuple=False)
    if locations.numel() == 0:
        raise ValueError(f"expert {expert} has no evaluation rows")
    if locations.shape[0] > maximum_rows:
        index = torch.linspace(0, locations.shape[0] - 1, maximum_rows).round().long()
        locations = locations.index_select(0, index)
    rows = locations[:, 0]
    slots = locations[:, 1]
    return {
        "inputs": samples.input_values.index_select(0, rows).float(),
        "gates": samples.input_gates[rows, slots].float(),
        "aggregate": samples.routed_latent.index_select(0, rows).float(),
        "documents": torch.bitwise_right_shift(
            samples.input_observations.index_select(0, rows), 32
        ),
    }


def _output_metric(
    store: OfficialMXFP4Store,
    layer: int,
    device: torch.device,
) -> RoutedOutputMetric:
    prefix = f"{C.LM_PREFIX}layers.{layer}.block_sparse_moe"
    return RoutedOutputMetric(
        load_tensor(store.cache, f"{prefix}.routed_expert_norm.weight")
        .float()
        .to(device),
        load_tensor(store.cache, C.latent_up_proj_tensor(layer)).float().to(device),
    )


def _source_triplet(
    layer_store: object,
    *,
    layer: int,
    expert: int,
    device: torch.device,
) -> CoupledTriplet:
    return CoupledTriplet(
        *(
            layer_store.load_matrix(layer, expert, matrix, device=device).float()
            for matrix in ("w1", "w3", "w2")
        )
    )


def _candidate_summary(candidate: UniformSQGCandidate) -> dict[str, object]:
    return {
        "one_sided_work_sse": candidate.one_sided_sse,
        "two_sided_work_sse": candidate.two_sided_sse,
        "proxy_relative_error": candidate.proxy_relative_error,
        "encode_seconds": candidate.seconds,
        "refinement": candidate.refinement,
    }


def _reconstruction_comparison(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float | bool]:
    difference = candidate.double() - reference.double()
    reference_energy = reference.double().square().sum()
    return {
        "bit_exact": bool(torch.equal(reference, candidate)),
        "relative_sse": float(difference.square().sum() / reference_energy),
        "maximum_absolute_error": float(difference.abs().max()),
        "cosine": float(
            F.cosine_similarity(reference.flatten(), candidate.flatten(), dim=0)
        ),
    }


def _factor_summary(factor: torch.Tensor) -> dict[str, float]:
    work = factor.double()
    diagonal = torch.diagonal(work)
    total_energy = work.square().sum()
    diagonal_energy = diagonal.square().sum()
    return {
        "diagonal_max": float(diagonal.max()),
        "diagonal_mean": float(diagonal.mean()),
        "diagonal_min": float(diagonal.min()),
        "frobenius_norm": float(torch.sqrt(total_energy)),
        "off_diagonal_energy_fraction": float(
            (total_energy - diagonal_energy) / total_energy
        ),
        "trace": float(diagonal.sum()),
    }


def _invert_output_metric(
    factor: torch.Tensor,
    *,
    damping_ratio: float,
) -> torch.Tensor:
    """Return the trace-matched inverse of the damped PSD metric."""

    if factor.ndim != 2 or factor.shape[0] != factor.shape[1]:
        raise ValueError("output metric must be square")
    if not math.isfinite(damping_ratio) or damping_ratio < 0.0:
        raise ValueError("output damping ratio must be finite and nonnegative")
    symmetric = (factor.float() + factor.float().T) * 0.5
    mean_diagonal = torch.diagonal(symmetric).mean()
    if not bool(torch.isfinite(mean_diagonal)) or float(mean_diagonal) <= 0.0:
        raise ValueError("output metric must have a finite positive trace")
    damped = symmetric.clone()
    damped.diagonal().add_(float(damping_ratio) * mean_diagonal)
    eigenvalues, eigenvectors = torch.linalg.eigh(damped)
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError("damped output metric is not positive definite")
    inverse = (
        eigenvectors * eigenvalues.reciprocal().unsqueeze(0)
    ) @ eigenvectors.T
    inverse.mul_(torch.diagonal(damped).mean() / torch.diagonal(inverse).mean())
    return ((inverse + inverse.T) * 0.5).contiguous()


def _factor_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    numerator = torch.sum(first.double() * second.double())
    denominator = (
        torch.linalg.vector_norm(first.double())
        * torch.linalg.vector_norm(second.double())
    )
    if float(denominator) == 0.0:
        raise ValueError("factor cosine is undefined for a zero factor")
    return float(numerator / denominator)


def _sketch_a_factors(
    *,
    samples: PairedLayer92FisherSamples,
    expert: int,
    production: CoupledTriplet,
    execution: CoupledHadamardExecution,
    initial_input_factor: torch.Tensor,
    factor_updates: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    occurrences = samples.expert_occurrences(expert)
    transformed = execution.transform_inputs(occurrences["expert_input"].float())
    decoded_input = execution.decode_middle(
        transformed,
        production.gate,
        production.up,
    )
    weighted_input = decoded_input * occurrences["route_weight"][:, None]
    gradients = occurrences["output_gradient"].float()

    def iterate(selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_factor = initial_input_factor
        output_factor = None
        selected_inputs = weighted_input[selected]
        selected_gradients = gradients[selected]
        for update in range(factor_updates):
            if update % 2 == 0:
                output_factor = sketch_a_output_factor_update(
                    selected_inputs,
                    selected_gradients,
                    input_factor,
                )
            else:
                if output_factor is None:
                    raise AssertionError("Sketch-A output update did not run first")
                input_factor = sketch_a_input_factor_update(
                    selected_inputs,
                    selected_gradients,
                    output_factor,
                )
        if output_factor is None:
            raise AssertionError("Sketch-A produced no output factor")
        return input_factor.contiguous(), output_factor.contiguous()

    all_rows = torch.arange(weighted_input.shape[0], device=weighted_input.device)
    input_factor, output_factor = iterate(all_rows)
    split_factors: list[tuple[torch.Tensor, torch.Tensor]] = []
    split_counts: list[int] = []
    for split in (0, 1):
        selected = torch.nonzero(
            occurrences["split"] == split, as_tuple=False
        ).flatten()
        if selected.numel() == 0:
            raise ValueError(f"expert {expert} has no Sketch-A split {split} rows")
        split_counts.append(int(selected.numel()))
        split_factors.append(iterate(selected))
    input_cosine = _factor_cosine(split_factors[0][0], split_factors[1][0])
    output_cosine = _factor_cosine(split_factors[0][1], split_factors[1][1])
    paired_rows = torch.stack(
        (occurrences["context_index"], occurrences["row_index"]), dim=1
    )
    route_weights = occurrences["route_weight"].double()
    evidence: dict[str, object] = {
        "factor_updates": factor_updates,
        "update_sequence": [
            "output" if update % 2 == 0 else "input"
            for update in range(factor_updates)
        ],
        "fisher_samples": int(weighted_input.shape[0]),
        "captured_rows": int(torch.unique(paired_rows, dim=0).shape[0]),
        "contexts": int(torch.unique(occurrences["context_index"]).numel()),
        "split_fisher_samples": split_counts,
        "route_weight_min": float(route_weights.min()),
        "route_weight_max": float(route_weights.max()),
        "route_weight_square_sum": float(route_weights.square().sum()),
        "input_factor_reweighted": factor_updates >= 2,
        "input_factor": _factor_summary(input_factor),
        "output_factor": _factor_summary(output_factor),
        "split_input_factor_cosine": input_cosine,
        "split_output_factor_cosine": output_cosine,
        "split_kronecker_factor_cosine": input_cosine * output_cosine,
    }
    return input_factor, output_factor, evidence


def _score_down_candidate(
    *,
    source_coordinates: CoupledTriplet,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    production_h2: torch.Tensor,
    optimization_input_hessian: torch.Tensor,
    output_hessian_coupled: torch.Tensor,
    scoring_output_hessian_coupled: torch.Tensor,
    execution: CoupledHadamardExecution,
    evaluation: Mapping[str, torch.Tensor],
    output_metric: RoutedOutputMetric,
) -> dict[str, object]:
    candidate_coordinates = CoupledTriplet(gate, up, down)
    source_output = execution.execute(
        evaluation["inputs"], source_coordinates.tensors()
    )
    canonical_down_output = execution.execute(
        evaluation["inputs"],
        CoupledTriplet(
            source_coordinates.gate,
            source_coordinates.up,
            down,
        ).tensors(),
    )
    candidate_output = execution.execute(
        evaluation["inputs"], candidate_coordinates.tensors()
    )
    canonical_down_error = canonical_down_output - source_output
    output_error = candidate_output - source_output
    canonical_down_routed_error = (
        evaluation["gates"][:, None] * canonical_down_error
    )
    routed_error = evaluation["gates"][:, None] * output_error
    canonical_down_geometry = routed_error_geometry(
        evaluation["aggregate"], canonical_down_routed_error, output_metric
    )
    geometry = routed_error_geometry(
        evaluation["aggregate"], routed_error, output_metric
    )
    dense_sse, dense_energy = quadratic_matrix_sse(
        source_coordinates.down,
        down,
        production_h2,
    )
    optimization_sse, optimization_energy = quadratic_matrix_sse(
        source_coordinates.down,
        down,
        optimization_input_hessian,
    )
    two_sided_sse, two_sided_energy = two_sided_encoder_sse(
        source_coordinates.down.T,
        down.T,
        optimization_input_hessian,
        output_hessian_coupled,
    )
    scoring_two_sided_sse, scoring_two_sided_energy = two_sided_encoder_sse(
        source_coordinates.down.T,
        down.T,
        optimization_input_hessian,
        scoring_output_hessian_coupled,
    )
    return {
        "input_hessian_sse": dense_sse,
        "input_hessian_nmse": dense_sse / dense_energy,
        "optimization_input_hessian_sse": optimization_sse,
        "optimization_input_hessian_nmse": (
            optimization_sse / optimization_energy
        ),
        "two_sided_source_sse": two_sided_sse,
        "two_sided_source_nmse": two_sided_sse / two_sided_energy,
        "scoring_two_sided_source_sse": scoring_two_sided_sse,
        "scoring_two_sided_source_nmse": (
            scoring_two_sided_sse / scoring_two_sided_energy
        ),
        "canonical_down_expert_output_sse": float(
            canonical_down_error.double().square().sum()
        ),
        "canonical_down_routed_output": asdict(canonical_down_geometry),
        "expert_output_sse": float(output_error.double().square().sum()),
        "routed_output": asdict(geometry),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--experts", type=_parse_experts, required=True)
    parser.add_argument("--fit-cache", type=Path, default=DEFAULT_FIT_CACHE)
    parser.add_argument("--fit-report", type=Path, default=DEFAULT_FIT_REPORT)
    parser.add_argument("--evaluation-cache", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument(
        "--evaluation-report", type=Path, default=DEFAULT_EVALUATION_REPORT
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--candidate-pool", type=Path, default=DEFAULT_CANDIDATE_POOL
    )
    parser.add_argument(
        "--maximum-fit-rows",
        type=int,
        default=0,
        help="maximum construction rows per expert; zero preserves all production rows",
    )
    parser.add_argument("--maximum-output-factor-rows", type=int, default=4096)
    factor_group = parser.add_mutually_exclusive_group()
    factor_group.add_argument(
        "--output-hessian",
        type=Path,
        help=(
            "one routed-output Fisher safetensors file; when omitted, build "
            "the local mapped-output factor from the construction cache"
        ),
    )
    factor_group.add_argument(
        "--output-factor-archive",
        type=Path,
        help=(
            "sealed Kimi routed-output Fisher archive; select the factor for "
            "--layer from its committed inventory"
        ),
    )
    factor_group.add_argument(
        "--sketch-a-capture-dir",
        type=Path,
        help=(
            "sealed format-2 layer-92 capture containing paired expert inputs, "
            "routes, and common-suffix states"
        ),
    )
    parser.add_argument(
        "--output-hessian-key",
        choices=(
            "output_hessian",
            "output_hessian_split_a",
            "output_hessian_split_b",
        ),
        default="output_hessian",
        help="tensor selected from --output-hessian",
    )
    parser.add_argument(
        "--scoring-output-hessian-key",
        choices=(
            "output_hessian",
            "output_hessian_split_a",
            "output_hessian_split_b",
        ),
        help=(
            "independent tensor from the same output-Hessian file used only "
            "to score the encoded path; defaults to --output-hessian-key"
        ),
    )
    parser.add_argument(
        "--sketch-a-suite-dir",
        type=Path,
        help="distribution-fidelity suite supplying reference hidden states and LM head",
    )
    parser.add_argument(
        "--sketch-a-checkpoint",
        type=Path,
        help="exact checkpoint that produced the paired suffix capture",
    )
    parser.add_argument("--sketch-a-pairs-per-row", type=int, default=1)
    parser.add_argument("--sketch-a-seed", type=int, default=20260814)
    parser.add_argument(
        "--sketch-a-factor-updates",
        type=int,
        default=1,
        help=(
            "alternating factor updates starting with output; one keeps the "
            "production H2 fixed, two also reweights H2, and three refreshes "
            "the output factor from the reweighted H2"
        ),
    )
    parser.add_argument("--maximum-evaluation-rows", type=int, default=1024)
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument(
        "--expert-batch-size",
        type=int,
        default=16,
        help=(
            "number of independent W2 matrices encoded in one lockstep "
            "BaKron traversal when --payload-only is set"
        ),
    )
    parser.add_argument("--output-damping-ratio", type=float, default=1e-4)
    parser.add_argument(
        "--output-factor-mode",
        choices=("fisher", "inverse_fisher"),
        default="fisher",
        help=(
            "use the captured Fisher metric or its damped, trace-matched "
            "spectral inverse"
        ),
    )
    parser.add_argument(
        "--input-factor-mode",
        choices=("decoded_h2", "identity"),
        default="decoded_h2",
        help=(
            "use the decoded-upstream activation covariance or isolate the "
            "captured output Fisher with an identity input factor"
        ),
    )
    parser.add_argument("--tailbite-context", type=int, default=128)
    parser.add_argument(
        "--h2-vcd-sweeps",
        type=int,
        default=0,
        help=(
            "run this many plain conditional-target H2-aware Viterbi "
            "coordinate-descent sweeps from the one-sided candidate"
        ),
    )
    parser.add_argument("--ldlq-tf32", action="store_true")
    parser.add_argument(
        "--direct-viterbi",
        action="store_true",
        help="encode independent SQG tiles without decoded-H2 or Fisher feedback",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--exllamav3-root",
        type=Path,
        default=Path("/home/luke/projects/exllamav3"),
    )
    parser.add_argument("--official-revision", default=C.REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--payload-overlay",
        type=Path,
        help=(
            "write the packed two-sided W2 tensors for the requested experts "
            "to a safetensors overlay"
        ),
    )
    parser.add_argument(
        "--payload-only",
        action="store_true",
        help=(
            "encode and validate the packed W2 overlay without loading or "
            "scoring local evaluation rows"
        ),
    )
    args = parser.parse_args()
    if args.layer not in C.MOE_LAYERS:
        parser.error("layer lies outside Kimi-K3 routed-MoE geometry")
    positive = (
        args.maximum_output_factor_rows,
        args.maximum_evaluation_rows,
        args.batch_rows,
        args.expert_batch_size,
    )
    if any(value <= 0 for value in positive):
        parser.error("row limits and batch size must be positive")
    if args.maximum_fit_rows < 0:
        parser.error("maximum fit rows must be nonnegative")
    if (
        args.output_hessian is None
        and args.output_factor_archive is None
        and args.output_hessian_key != "output_hessian"
    ):
        parser.error("split output-Hessian keys require --output-hessian")
    if (
        args.scoring_output_hessian_key is not None
        and args.output_hessian is None
        and args.output_factor_archive is None
    ):
        parser.error("independent output-Hessian scoring requires a factor file")
    if (
        args.scoring_output_hessian_key is not None
        and args.sketch_a_capture_dir is not None
    ):
        parser.error("paired Sketch A does not support a separate scoring factor")
    sketch_paths = (args.sketch_a_suite_dir, args.sketch_a_checkpoint)
    if args.sketch_a_capture_dir is not None and any(
        path is None for path in sketch_paths
    ):
        parser.error(
            "--sketch-a-capture-dir requires --sketch-a-suite-dir and "
            "--sketch-a-checkpoint"
        )
    if args.sketch_a_capture_dir is None and any(
        path is not None for path in sketch_paths
    ):
        parser.error(
            "Sketch-A suite and checkpoint require --sketch-a-capture-dir"
        )
    if args.sketch_a_capture_dir is not None and args.layer != 92:
        parser.error("paired common-suffix Sketch A applies only to layer 92")
    if args.sketch_a_pairs_per_row <= 0:
        parser.error("Sketch-A pairs per row must be positive")
    if args.sketch_a_factor_updates <= 0:
        parser.error("Sketch-A factor updates must be positive")
    if args.sketch_a_capture_dir is not None and args.h2_vcd_sweeps:
        parser.error("paired Sketch A and H2-VCD must be measured separately")
    if not 1 <= args.tailbite_context <= 128:
        parser.error("tail-biting context must lie in 1..128")
    if args.h2_vcd_sweeps < 0:
        parser.error("H2-VCD sweep count must be nonnegative")
    if args.output_factor_mode == "inverse_fisher" and (
        args.output_hessian is None and args.output_factor_archive is None
    ):
        parser.error("inverse Fisher requires a captured output factor")
    if args.direct_viterbi and not args.payload_only:
        parser.error("direct Viterbi is a payload-only control")
    if args.direct_viterbi and args.input_factor_mode != "decoded_h2":
        parser.error("direct Viterbi does not consume an input factor")
    if args.input_factor_mode == "identity" and (
        args.output_hessian is None and args.output_factor_archive is None
    ):
        parser.error("identity input mode requires a captured output factor")
    if args.input_factor_mode == "identity" and args.sketch_a_capture_dir is not None:
        parser.error("identity input mode is distinct from paired Sketch A")
    return args


def _run_direct_viterbi_payload(
    args: argparse.Namespace,
    *,
    candidate_manifest: dict[str, object],
    resolved_schema: str,
    started: float,
) -> None:
    if args.payload_overlay is None:
        raise ValueError("direct Viterbi requires a payload overlay")
    device = torch.device(args.device)
    draws = _profile_draws(args.profile, args.layer, args.experts)
    candidate_path = (
        args.candidate_pool
        / "candidates"
        / f"qsrt-layer-{args.layer:05d}.safetensors"
    )
    candidate_metrics = load_torch_file(
        str(
            candidate_path.with_name(
                f"qsrt-layer-{args.layer:05d}.metrics.safetensors"
            )
        ),
        device="cpu",
    )
    required_metrics = {"selected_r13", "selected_r2", "coupled_draw_selected"}
    if not required_metrics.issubset(candidate_metrics):
        raise ValueError("candidate-pool metrics lack rate or rotation selections")
    for expert in args.experts:
        if int(candidate_metrics["selected_r13"][expert]) != K2.mode_id or int(
            candidate_metrics["selected_r2"][expert]
        ) != K2.mode_id:
            raise ValueError(f"candidate-pool expert {expert} is not uniform K2")
        if int(candidate_metrics["coupled_draw_selected"][expert]) != draws[expert]:
            raise ValueError(
                f"profile and candidate-pool draws differ for expert {expert}"
            )

    quantizer_module = load_qsrt_encoder(args.exllamav3_root)
    install_sqg_quantizer(quantizer_module)
    store = OfficialMXFP4Store(revision=args.official_revision)
    down_seeds = default_qsrt_transform_seeds(args.layer, "w2")
    plan = plan_qsrt_matrix(
        torch.zeros(768, dtype=torch.long, device=device),
        K2,
        matrix="w2",
        layout="importance_ordered",
    )
    overlay: dict[str, torch.Tensor] = {}
    results: dict[str, object] = {}
    with store.open_layer(args.layer, experts=args.experts) as layer_store:
        for batch_start in range(0, len(args.experts), args.expert_batch_size):
            batch_experts = args.experts[
                batch_start : batch_start + args.expert_batch_size
            ]
            sources = []
            for expert in batch_experts:
                source = _source_triplet(
                    layer_store,
                    layer=args.layer,
                    expert=expert,
                    device=device,
                )
                coordinates = CoupledTriplet(
                    *encode_coupled_weights(
                        source.tensors(),
                        CoupledHadamardSpec(intermediate_draw=draws[expert]),
                    )
                )
                sources.append(coordinates.down)
            encoded = encode_uniform_sqg_direct_batch(
                torch.stack(sources),
                bits=2,
                device=device,
                quantizer_module=quantizer_module,
                input_sign_seed=down_seeds.input_sign,
                output_sign_seed=down_seeds.output_sign,
                rate_axis=matrix_rate_axis("w2"),
                scale_scope_key=None,
                shared_scale_axis=None,
                tailbite_context=args.tailbite_context,
            )
            for expert, result in zip(batch_experts, encoded, strict=True):
                packed = finalize_qsrt_matrix_candidate(
                    QSRTMatrixCandidate(
                        reconstruction=result.candidate.reconstruction,
                        encoded=result.candidate.states,
                        tensors={"suh": result.suh, "svh": result.svh},
                        plan=plan,
                        proxy=0.0,
                        transform_seeds=down_seeds,
                        global_scale=result.global_scale,
                    ),
                    layer=args.layer,
                    logical_trellis_schema=resolved_schema,
                    codebook=str(candidate_manifest["codebook"]),
                    tailbite_context=args.tailbite_context,
                )
                for part, value in packed.tensors.items():
                    overlay[
                        candidate_tensor_name(args.layer, expert, "w2", part)
                    ] = value.detach().cpu().contiguous()
                results[str(expert)] = {
                    "intermediate_draw": draws[expert],
                    "global_scale": result.global_scale,
                    "seconds": result.candidate.seconds,
                    "stored_decode_vs_encoder": packed.coding[
                        "stored_decode_vs_encoder"
                    ],
                }
            print(
                json.dumps(
                    {
                        "layer": args.layer,
                        "encoded_experts": min(
                            batch_start + len(batch_experts), len(args.experts)
                        ),
                        "experts": len(args.experts),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if len(overlay) != 3 * len(args.experts):
        raise RuntimeError("direct W2 overlay has the wrong tensor count")
    args.payload_overlay.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.payload_overlay.with_name(
        f".{args.payload_overlay.name}.{os.getpid()}.tmp"
    )
    save_file(
        overlay,
        str(temporary),
        metadata={
            "kind": "qsrt_uniform_k2_direct_viterbi_w2_overlay",
            "layer": str(args.layer),
            "experts": str(len(args.experts)),
        },
    )
    os.replace(temporary, args.payload_overlay)
    receipt = {
        "kind": "qsrt_uniform_k2_direct_viterbi_w2_layer",
        "schema_version": 1,
        "complete": True,
        "layer": args.layer,
        "experts": results,
        "payload_overlay": str(args.payload_overlay.resolve()),
        "payload_overlay_sha256": _sha256(args.payload_overlay),
        "seconds": time.time() - started,
        "rotation_source": str(args.profile.resolve()),
        "objective": "independent tail-biting SQG tile distortion",
    }
    _atomic_json(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.payload_overlay is not None and args.payload_overlay.exists():
        raise FileExistsError(args.payload_overlay)
    if args.payload_only and args.payload_overlay is None:
        raise ValueError("--payload-only requires --payload-overlay")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("two-sided W2 encoding requires CUDA")
    uses_decoded_h2 = args.input_factor_mode == "decoded_h2"
    started = time.time()
    output_factor_archive = None
    output_factor_corpus_identity = None
    if args.output_factor_archive is not None and not args.direct_viterbi:
        output_factor_archive = KimiOutputFactorArchive(args.output_factor_archive)
        if output_factor_archive.manifest.get("complete") is not True:
            raise ValueError("output-factor archive is not complete")
        output_factor_corpus_identity = (
            _verify_output_factor_evaluation_disjointness(
                archive=output_factor_archive,
                evaluation_report=args.evaluation_report,
            )
        )
        args.output_hessian = output_factor_archive.layer_path(args.layer)
    candidate_manifest = json.loads(
        (args.candidate_pool / "qsrt-candidate-manifest.json").read_text()
    )
    source_schema, resolved_schema = _resolved_candidate_schema(candidate_manifest)
    if args.direct_viterbi:
        _run_direct_viterbi_payload(
            args,
            candidate_manifest=candidate_manifest,
            resolved_schema=resolved_schema,
            started=started,
        )
        return
    signature = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "source_model": C.MODEL_ID,
        "source_revision": args.official_revision,
        "layer": args.layer,
        "experts": list(args.experts),
        "rate": 2,
        "codebook": "sqg_xor_cheb_t12",
        "fit_cache": (
            str(args.fit_cache.resolve()) if uses_decoded_h2 else None
        ),
        "fit_cache_manifest_sha256": (
            _sha256(args.fit_cache / "manifest.json") if uses_decoded_h2 else None
        ),
        "fit_report": str(args.fit_report.resolve()) if uses_decoded_h2 else None,
        "fit_report_sha256": _sha256(args.fit_report) if uses_decoded_h2 else None,
        "evaluation_cache": str(args.evaluation_cache.resolve()),
        "evaluation_manifest_sha256": _sha256(
            args.evaluation_cache / "manifest.json"
        ),
        "evaluation_report": str(args.evaluation_report.resolve()),
        "evaluation_report_sha256": _sha256(args.evaluation_report),
        "profile": str(args.profile.resolve()),
        "profile_completion_sha256": _sha256(
            args.profile / "qsrt-completion.json"
        ),
        "candidate_pool": str(args.candidate_pool.resolve()),
        "candidate_pool_manifest_sha256": _sha256(
            args.candidate_pool / "qsrt-candidate-manifest.json"
        ),
        "candidate_pool_schema": source_schema,
        "decoder_schema": resolved_schema,
        "maximum_fit_occurrences": args.maximum_fit_rows,
        "fit_sampling": (
            "all production construction-fold rows"
            if uses_decoded_h2 and args.maximum_fit_rows == 0
            else "evenly sampled production construction-fold rows"
            if uses_decoded_h2
            else None
        ),
        "maximum_output_factor_rows": args.maximum_output_factor_rows,
        "maximum_evaluation_occurrences": args.maximum_evaluation_rows,
        "expert_batch_size": (
            args.expert_batch_size if args.payload_only else 1
        ),
        "input_factor_mode": args.input_factor_mode,
        "input_factor": (
            "expert-local decoded-upstream post-SiTU covariance"
            if uses_decoded_h2
            else "identity in coupled W2 input coordinates"
        ),
        "output_factor_mode": args.output_factor_mode,
        "output_damping_ratio": args.output_damping_ratio,
        "tailbite_context": args.tailbite_context,
        "h2_vcd_sweeps": args.h2_vcd_sweeps,
        "h2_vcd_candidate_generation": (
            "plain conditional target with no dither probes"
            if args.h2_vcd_sweeps
            else None
        ),
        "ldlq_tf32": args.ldlq_tf32,
        "output_factor": (
            "expert-conditional paired YAQA Sketch A"
            if args.sketch_a_capture_dir is not None
            else (
                "official final-logit empirical Fisher at the routed W2 output"
                if output_factor_archive is not None
                else "layer-92 common-suffix empirical Fisher"
                if args.output_hessian is not None
                else (
                    "layer-shared exact local RMSNorm/output-projection "
                    "Jacobian Gram"
                )
            )
        ),
        "output_hessian": (
            None
            if args.output_hessian is None
            else str(args.output_hessian.resolve())
        ),
        "output_hessian_sha256": (
            None
            if args.output_hessian is None
            else _sha256(args.output_hessian)
        ),
        "output_hessian_key": (
            None if args.output_hessian is None else args.output_hessian_key
        ),
        "scoring_output_hessian_key": (
            None
            if args.output_hessian is None
            else args.scoring_output_hessian_key or args.output_hessian_key
        ),
        "output_factor_archive": (
            None
            if output_factor_archive is None
            else str(output_factor_archive.root)
        ),
        "output_factor_archive_manifest_sha256": (
            None
            if output_factor_archive is None
            else _sha256(output_factor_archive.manifest_path)
        ),
        "output_factor_corpus_identity": output_factor_corpus_identity,
        "sketch_a_capture": (
            None
            if args.sketch_a_capture_dir is None
            else str(args.sketch_a_capture_dir.resolve())
        ),
        "sketch_a_suite": (
            None
            if args.sketch_a_suite_dir is None
            else str(args.sketch_a_suite_dir.resolve())
        ),
        "sketch_a_checkpoint": (
            None
            if args.sketch_a_checkpoint is None
            else str(args.sketch_a_checkpoint.resolve())
        ),
        "sketch_a_pairs_per_row": (
            args.sketch_a_pairs_per_row
            if args.sketch_a_capture_dir is not None
            else None
        ),
        "sketch_a_seed": (
            args.sketch_a_seed if args.sketch_a_capture_dir is not None else None
        ),
        "sketch_a_factor_updates": (
            args.sketch_a_factor_updates
            if args.sketch_a_capture_dir is not None
            else None
        ),
        "output_factor_route_weighting": (
            "captured applied route weight multiplies the decoded W2 input once"
            if args.sketch_a_capture_dir is not None
            else "none; expert H2 contains route_weight_squared"
            if uses_decoded_h2
            else (
                "none; identity input curvature isolates output-error direction "
                "and per-expert scalar route mass cannot change the minimizer"
            )
        ),
        "solver": "block_bakron_recursive",
        "scale_policy": (
            "one BlockLDLQ scale search shared by the one-sided and two-sided "
            "W2 re-encodes; the served payload retains its stored scales"
        ),
        "writes_checkpoint_payloads": False,
        "payload_overlay": (
            None
            if args.payload_overlay is None
            else str(args.payload_overlay.resolve())
        ),
    }
    receipt: dict[str, object] = {"signature": signature, "complete": False}
    _atomic_json(args.output, receipt)

    corpus_identity = (
        _verify_corpus_disjointness(
            fit_report=args.fit_report,
            fit_cache=args.fit_cache,
            evaluation_report=args.evaluation_report,
            evaluation_cache=args.evaluation_cache,
        )
        if uses_decoded_h2
        else {
            "input_factor": "identity; no activation capture consumed",
            "output_factor": output_factor_corpus_identity,
        }
    )

    store = OfficialMXFP4Store(revision=args.official_revision)
    output_metric = (
        None if args.payload_only else _output_metric(store, args.layer, device)
    )
    fit_samples = (
        _load_evaluation_samples(args.fit_cache, args.layer)
        if uses_decoded_h2
        else None
    )
    fit_partition = _request_partition(args.fit_report) if uses_decoded_h2 else None
    paired_fisher_samples = None
    output_hessian_ordinary = None
    scoring_output_hessian_ordinary = None
    if args.sketch_a_capture_dir is not None:
        if args.sketch_a_suite_dir is None or args.sketch_a_checkpoint is None:
            raise AssertionError("validated Sketch-A paths are absent")
        paired_fisher_samples, output_factor_support = (
            load_paired_layer92_fisher_samples(
                capture_dir=args.sketch_a_capture_dir,
                suite_dir=args.sketch_a_suite_dir,
                checkpoint=args.sketch_a_checkpoint,
                device=device,
                pairs_per_row=args.sketch_a_pairs_per_row,
                seed=args.sketch_a_seed,
            )
        )
        output_factor_support["source"] = (
            "paired expert inputs and common-suffix real-Fisher gradients"
        )
    elif args.output_hessian is None:
        if fit_samples is None or fit_partition is None:
            raise AssertionError("mapped-output curvature requires fit samples")
        output_rows, output_documents, output_row_indices = _sample_layer_outputs(
            fit_samples, args.maximum_output_factor_rows, fit_partition.fit
        )
        output_rows = output_rows.float().to(device)
        output_hessian_ordinary = mapped_output_hessian(
            output_rows,
            output_metric,
            accumulation_dtype=torch.float32,
        )
        del output_rows
        output_factor_support: dict[str, object] = {
            "rows": int(output_row_indices.numel()),
            "documents": int(torch.unique(output_documents).numel()),
            "row_first": int(output_row_indices[0]),
            "row_last": int(output_row_indices[-1]),
            "source": "construction-cache mapped-output states",
        }
    else:
        with safe_open(
            args.output_hessian, framework="pt", device="cpu"
        ) as factor_reader:
            metadata = factor_reader.metadata() or {}
            if metadata.get("kind") == OUTPUT_FACTOR_ARCHIVE_KIND:
                if int(metadata.get("layer", -1)) != args.layer:
                    raise ValueError("output Fisher belongs to a different decoder layer")
                if metadata.get("semantic_point") != (
                    "route-weighted expert W2 sum before routed latent RMSNorm "
                    "and output projection"
                ):
                    raise ValueError("output Fisher has the wrong semantic point")
            else:
                if args.layer != 92:
                    raise ValueError(
                        "the legacy common-suffix Fisher factor applies only to layer 92"
                    )
                if metadata.get("partition") != "analysis":
                    raise ValueError("output Fisher must identify the analysis partition")
                if metadata.get("semantic_point") != (
                    "aggregated_routed_w2_output_before_routed_rmsnorm"
                ):
                    raise ValueError("output Fisher has the wrong semantic point")
            if args.output_hessian_key not in factor_reader.keys():
                raise ValueError(
                    f"output Fisher lacks tensor {args.output_hessian_key}"
                )
            output_hessian_ordinary = factor_reader.get_tensor(
                args.output_hessian_key
            )
            scoring_key = (
                args.scoring_output_hessian_key or args.output_hessian_key
            )
            if scoring_key not in factor_reader.keys():
                raise ValueError(f"output Fisher lacks tensor {scoring_key}")
            scoring_output_hessian_ordinary = factor_reader.get_tensor(scoring_key)
        if output_hessian_ordinary.shape != (C.LATENT, C.LATENT):
            raise ValueError("output Fisher has the wrong Kimi routed dimension")
        output_hessian_ordinary = output_hessian_ordinary.float().to(device)
        if scoring_output_hessian_ordinary.shape != (C.LATENT, C.LATENT):
            raise ValueError("scoring output Fisher has the wrong Kimi routed dimension")
        scoring_output_hessian_ordinary = (
            scoring_output_hessian_ordinary.float().to(device)
        )
        factor_receipt_path = args.output_hessian.with_suffix(".json")
        factor_receipt = (
            json.loads(factor_receipt_path.read_text())
            if factor_receipt_path.exists()
            else None
        )
        output_factor_support = {
            "source": (
                "official final-logit empirical Fisher at the routed W2 output"
                if metadata.get("kind") == OUTPUT_FACTOR_ARCHIVE_KIND
                else "layer-92 common-suffix empirical Fisher"
            ),
            "factor": str(args.output_hessian.resolve()),
            "factor_sha256": _sha256(args.output_hessian),
            "factor_tensor": args.output_hessian_key,
            "receipt": (
                None
                if factor_receipt is None
                else str(factor_receipt_path.resolve())
            ),
            "fisher_samples": (
                None
                if factor_receipt is None
                else factor_receipt.get("fisher_samples")
            ),
            "split_cosine": (
                None
                if factor_receipt is None
                else factor_receipt.get("split_cosine")
            ),
            "rows": (
                None if metadata.get("rows") is None else int(metadata["rows"])
            ),
            "split_a_rows": (
                None
                if metadata.get("split_a_rows") is None
                else int(metadata["split_a_rows"])
            ),
            "split_b_rows": (
                None
                if metadata.get("split_b_rows") is None
                else int(metadata["split_b_rows"])
            ),
        }

    encoder_output_damping_ratio = args.output_damping_ratio
    if args.output_factor_mode == "inverse_fisher":
        if output_hessian_ordinary is None:
            raise AssertionError("inverse Fisher mode has no output factor")
        output_hessian_ordinary = _invert_output_metric(
            output_hessian_ordinary,
            damping_ratio=args.output_damping_ratio,
        ).to(device)
        if scoring_output_hessian_ordinary is not None:
            scoring_output_hessian_ordinary = _invert_output_metric(
                scoring_output_hessian_ordinary,
                damping_ratio=args.output_damping_ratio,
            ).to(device)
        encoder_output_damping_ratio = 0.0
        output_factor_support["metric_transform"] = {
            "kind": "trace_matched_spectral_inverse",
            "damping_ratio_before_inverse": args.output_damping_ratio,
            "encoder_damping_ratio": encoder_output_damping_ratio,
        }
    samples = (
        None
        if args.payload_only
        else _load_evaluation_samples(args.evaluation_cache, args.layer)
    )

    draws = _profile_draws(args.profile, args.layer, args.experts)
    fit_manifest = (
        json.loads((args.fit_cache / "manifest.json").read_text())
        if uses_decoded_h2
        else {}
    )
    if uses_decoded_h2:
        expected_fit_cache = Path(str(candidate_manifest["sample_cache"])).name
        if args.fit_cache.name != expected_fit_cache:
            raise ValueError(
                "fit cache does not match the candidate-pool construction cache"
            )
    candidate_path = (
        args.candidate_pool
        / "candidates"
        / f"qsrt-layer-{args.layer:05d}.safetensors"
    )
    candidate_metrics_path = candidate_path.with_name(
        f"qsrt-layer-{args.layer:05d}.metrics.safetensors"
    )
    candidate_metrics = load_torch_file(
        str(candidate_metrics_path), device="cpu"
    )
    candidate_ledger = (
        json.loads(
            candidate_path.with_name(
                f"qsrt-layer-{args.layer:05d}.selection.json"
            ).read_text()
        )
        if uses_decoded_h2
        else None
    )
    required_metrics = {"selected_r13", "selected_r2", "coupled_draw_selected"}
    if not required_metrics.issubset(candidate_metrics):
        raise ValueError("candidate-pool metrics lack rate or coupled-draw selections")
    for expert in args.experts:
        r13 = int(candidate_metrics["selected_r13"][expert])
        r2 = int(candidate_metrics["selected_r2"][expert])
        draw = int(candidate_metrics["coupled_draw_selected"][expert])
        if r13 != 4 or r2 != 4:
            raise ValueError(f"candidate-pool expert {expert} is not uniform K2")
        if draw != draws[expert]:
            raise ValueError(
                f"profile and candidate-pool draws differ for expert {expert}"
            )
    quantizer_module = load_qsrt_encoder(args.exllamav3_root)
    install_sqg_quantizer(quantizer_module)
    fit_row_index = (
        _index_fit_rows(fit_samples, args.maximum_fit_rows, fit_partition.fit)
        if fit_samples is not None and fit_partition is not None
        else None
    )
    results: dict[str, object] = {}
    payload_overlay: dict[str, torch.Tensor] = {}
    needs_candidate_payload = uses_decoded_h2 or not args.payload_only
    candidate_context = (
        safe_open(candidate_path, framework="pt", device="cpu")
        if needs_candidate_payload
        else nullcontext(None)
    )
    with (
        store.open_layer(args.layer, experts=args.experts) as layer_store,
        candidate_context as candidate_reader,
    ):
        use_batched_payload_encoder = (
            args.payload_only
            and args.expert_batch_size > 1
            and paired_fisher_samples is None
            and args.h2_vcd_sweeps == 0
        )
        if use_batched_payload_encoder:
            if output_hessian_ordinary is None:
                raise AssertionError("batched W2 encoding has no output Hessian")
            down_seeds = default_qsrt_transform_seeds(args.layer, "w2")
            plan = plan_qsrt_matrix(
                torch.zeros(768, dtype=torch.long, device=device),
                K2,
                matrix="w2",
                layout="importance_ordered",
            )
            for batch_start in range(0, len(args.experts), args.expert_batch_size):
                batch_started = time.time()
                batch_experts = args.experts[
                    batch_start : batch_start + args.expert_batch_size
                ]
                contexts: list[SimpleNamespace] = []
                for expert in batch_experts:
                    preparation_started = time.time()
                    spec = CoupledHadamardSpec(intermediate_draw=draws[expert])
                    if uses_decoded_h2:
                        if (
                            candidate_reader is None
                            or fit_samples is None
                            or fit_row_index is None
                            or candidate_ledger is None
                        ):
                            raise AssertionError("decoded H2 inputs are absent")
                        source = _source_triplet(
                            layer_store,
                            layer=args.layer,
                            expert=expert,
                            device=device,
                        )
                        execution = CoupledHadamardExecution(
                            source.hidden, source.intermediate, spec
                        )
                        source_coordinates = CoupledTriplet(
                            *encode_coupled_weights(source.tensors(), spec)
                        )
                        production = CoupledTriplet(
                            *(
                                decode_candidate_matrix(
                                    candidate_reader,
                                    layer=args.layer,
                                    expert=expert,
                                    matrix=matrix,
                                    mode_id=4,
                                    device=device,
                                    logical_trellis_schema=resolved_schema,
                                    codebook=str(candidate_manifest["codebook"]),
                                )
                                .T.float()
                                .contiguous()
                                for matrix in ("w1", "w3", "w2")
                            )
                        )
                        fit_rows = _materialize_fit_rows(
                            fit_samples, fit_row_index[expert]
                        )
                        upstream = CoupledTriplet(
                            production.gate,
                            production.up,
                            source_coordinates.down,
                        )
                        if fit_rows["input"].shape[0] == 0:
                            h2, h2_evidence = _identity_input_factor(
                                source_coordinates.down.shape[1], device=device
                            )
                            h2_evidence["h2_shrinkage_policy"] = "identity_fallback"
                        else:
                            statistics = collect_coupled_hidden_statistics(
                                _fit_batches(
                                    fit_rows,
                                    maximum_rows=args.maximum_fit_rows,
                                    batch_rows=args.batch_rows,
                                ),
                                source=source,
                                candidate_coordinates=upstream,
                                execution=execution,
                            )
                            h2, h2_evidence = _production_h2(
                                statistics, fit_rows["route_weight"]
                            )
                        sealed_covariance = candidate_ledger["selections"][
                            str(expert)
                        ]["covariance"]
                        expected_rows = int(sealed_covariance["fit_rows"])
                        if h2_evidence["rows"] != expected_rows:
                            raise ValueError(
                                f"expert {expert} construction support differs "
                                f"from the sealed pool: {h2_evidence['rows']} != "
                                f"{expected_rows}"
                            )
                    else:
                        source_down = layer_store.load_matrix(
                            args.layer,
                            expert,
                            "w2",
                            device=device,
                        ).float()
                        execution = CoupledHadamardExecution(
                            int(source_down.shape[0]),
                            int(source_down.shape[1]),
                            spec,
                        )
                        source_coordinates = SimpleNamespace(
                            down=encode_coupled_down_weight(source_down, spec)
                        )
                        h2, h2_evidence = _identity_input_factor(
                            source_coordinates.down.shape[1], device=device
                        )
                    contexts.append(
                        SimpleNamespace(
                            expert=expert,
                            source_down=source_coordinates.down,
                            input_hessian=h2,
                            output_hessian=execution.transform_output_hessian(
                                output_hessian_ordinary
                            ),
                            h2_evidence=h2_evidence,
                            preparation_seconds=time.time()
                            - preparation_started,
                        )
                    )

                downs = encode_uniform_sqg_two_sided_batch(
                    torch.stack([context.source_down for context in contexts]),
                    torch.stack(
                        [context.input_hessian for context in contexts]
                    ),
                    torch.stack(
                        [context.output_hessian for context in contexts]
                    ),
                    bits=2,
                    device=device,
                    quantizer_module=quantizer_module,
                    input_sign_seed=down_seeds.input_sign,
                    output_sign_seed=down_seeds.output_sign,
                    rate_axis=matrix_rate_axis("w2"),
                    ldlq_tf32=args.ldlq_tf32,
                    tailbite_context=args.tailbite_context,
                    output_damping_ratio=encoder_output_damping_ratio,
                    compute_objective=False,
                )
                batch_seconds = time.time() - batch_started
                for context, down in zip(contexts, downs, strict=True):
                    packed = finalize_qsrt_matrix_candidate(
                        QSRTMatrixCandidate(
                            reconstruction=down.two_sided.reconstruction,
                            encoded=down.two_sided.states,
                            tensors={"suh": down.suh, "svh": down.svh},
                            plan=plan,
                            proxy=0.0,
                            transform_seeds=down_seeds,
                            global_scale=down.global_scale,
                        ),
                        layer=args.layer,
                        logical_trellis_schema=resolved_schema,
                        codebook=str(candidate_manifest["codebook"]),
                        tailbite_context=args.tailbite_context,
                    )
                    for part, value in packed.tensors.items():
                        payload_overlay[
                            candidate_tensor_name(
                                args.layer, context.expert, "w2", part
                            )
                        ] = value.detach().cpu().contiguous()
                    payload_closure = {
                        "stored_decode_vs_encoder": packed.coding[
                            "stored_decode_vs_encoder"
                        ],
                        "scoring_reconstruction_vs_stored": packed.coding[
                            "scoring_reconstruction_vs_stored"
                        ],
                        "trellis_bytes": int(
                            packed.tensors["trellis"].numel()
                            * packed.tensors["trellis"].element_size()
                        ),
                        "scale_bytes": int(
                            sum(
                                packed.tensors[name].numel()
                                * packed.tensors[name].element_size()
                                for name in ("suh", "svh")
                            )
                        ),
                    }
                    results[str(context.expert)] = {
                        "intermediate_draw": draws[context.expert],
                        "fit_support": context.h2_evidence,
                        "sketch_a_support": None,
                        "evaluation_support": None,
                        "upstream": (
                            "decoded sealed uniform-K2 candidate payload"
                            if uses_decoded_h2
                            else "not consumed by identity input curvature"
                        ),
                        "w2": {
                            "production": {},
                            "baseline": None,
                            "production_vs_reencoded_baseline": None,
                            "factor_input_one_sided": None,
                            "production_vs_two_sided": None,
                            "two_sided": _candidate_summary(down.two_sided),
                            "h2_vcd": None,
                            "path_state_disagreement_fraction": None,
                            "h2_vcd_path_state_disagreement_fraction": None,
                            "global_scale": down.global_scale,
                            "payload_closure": payload_closure,
                        },
                        "seconds": (
                            context.preparation_seconds
                            + down.two_sided.seconds
                        ),
                    }
                    print(
                        json.dumps(
                            {
                                "layer": args.layer,
                                "expert": context.expert,
                                "batch_size": len(contexts),
                                "batch_seconds": batch_seconds,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                receipt["experts"] = results
                _atomic_json(args.output, receipt)

        loop_experts = () if use_batched_payload_encoder else args.experts
        for expert in loop_experts:
            expert_started = time.time()
            spec = CoupledHadamardSpec(intermediate_draw=draws[expert])
            if uses_decoded_h2 or not args.payload_only:
                if candidate_reader is None:
                    raise AssertionError("candidate payload reader is absent")
                source = _source_triplet(
                    layer_store,
                    layer=args.layer,
                    expert=expert,
                    device=device,
                )
                execution = CoupledHadamardExecution(
                    source.hidden, source.intermediate, spec
                )
                source_coordinates = CoupledTriplet(
                    *encode_coupled_weights(source.tensors(), spec)
                )
                production = CoupledTriplet(
                    *(
                        decode_candidate_matrix(
                            candidate_reader,
                            layer=args.layer,
                            expert=expert,
                            matrix=matrix,
                            mode_id=4,
                            device=device,
                            logical_trellis_schema=resolved_schema,
                            codebook=str(candidate_manifest["codebook"]),
                        )
                        .T.float()
                        .contiguous()
                        for matrix in ("w1", "w3", "w2")
                    )
                )
            else:
                source_down = layer_store.load_matrix(
                    args.layer,
                    expert,
                    "w2",
                    device=device,
                ).float()
                execution = CoupledHadamardExecution(
                    int(source_down.shape[0]), int(source_down.shape[1]), spec
                )
                source_coordinates = SimpleNamespace(
                    down=encode_coupled_down_weight(source_down, spec)
                )
                source = None
                production = None
            down_seeds = default_qsrt_transform_seeds(args.layer, "w2")
            if not uses_decoded_h2:
                h2, h2_evidence = _identity_input_factor(
                    source_coordinates.down.shape[1], device=device
                )
            else:
                if (
                    source is None
                    or production is None
                    or fit_samples is None
                    or fit_row_index is None
                    or candidate_ledger is None
                ):
                    raise AssertionError("decoded H2 inputs are absent")
                fit_rows = _materialize_fit_rows(
                    fit_samples, fit_row_index[expert]
                )
                upstream = CoupledTriplet(
                    production.gate,
                    production.up,
                    source_coordinates.down,
                )
                if fit_rows["input"].shape[0] == 0:
                    h2, h2_evidence = _identity_input_factor(
                        source_coordinates.down.shape[1], device=device
                    )
                    h2_evidence["h2_shrinkage_policy"] = "identity_fallback"
                else:
                    statistics = collect_coupled_hidden_statistics(
                        _fit_batches(
                            fit_rows,
                            maximum_rows=args.maximum_fit_rows,
                            batch_rows=args.batch_rows,
                        ),
                        source=source,
                        candidate_coordinates=upstream,
                        execution=execution,
                    )
                    h2, h2_evidence = _production_h2(
                        statistics, fit_rows["route_weight"]
                    )
                sealed_covariance = candidate_ledger["selections"][str(expert)][
                    "covariance"
                ]
                expected_rows = int(sealed_covariance["fit_rows"])
                if h2_evidence["rows"] != expected_rows:
                    raise ValueError(
                        f"expert {expert} construction support differs from the "
                        f"sealed pool: {h2_evidence['rows']} != {expected_rows}"
                    )
            sketch_a_evidence = None
            optimization_input_hessian = h2
            expert_output_hessian = output_hessian_ordinary
            expert_scoring_output_hessian = scoring_output_hessian_ordinary
            if paired_fisher_samples is not None:
                if production is None:
                    raise AssertionError("paired Sketch A requires decoded upstream")
                (
                    optimization_input_hessian,
                    expert_output_hessian,
                    sketch_a_evidence,
                ) = _sketch_a_factors(
                    samples=paired_fisher_samples,
                    expert=expert,
                    production=production,
                    execution=execution,
                    initial_input_factor=h2,
                    factor_updates=args.sketch_a_factor_updates,
                )
                expert_scoring_output_hessian = expert_output_hessian
            if expert_output_hessian is None:
                raise AssertionError("W2 pilot has no output Hessian")
            if expert_scoring_output_hessian is None:
                expert_scoring_output_hessian = expert_output_hessian
            output_hessian_coupled = execution.transform_output_hessian(
                expert_output_hessian
            )
            scoring_output_hessian_coupled = execution.transform_output_hessian(
                expert_scoring_output_hessian
            )
            down = encode_uniform_sqg_two_sided_pair(
                source_coordinates.down,
                optimization_input_hessian,
                output_hessian_coupled,
                bits=2,
                device=device,
                quantizer_module=quantizer_module,
                input_sign_seed=down_seeds.input_sign,
                output_sign_seed=down_seeds.output_sign,
                rate_axis=matrix_rate_axis("w2"),
                ldlq_tf32=args.ldlq_tf32,
                tailbite_context=args.tailbite_context,
                output_damping_ratio=encoder_output_damping_ratio,
                work_dtype=torch.float32,
                h2_vcd_sweeps=args.h2_vcd_sweeps,
                include_baseline=not args.payload_only,
            )
            if args.payload_only:
                baseline_candidate = None
                factor_input_baseline = None
            elif paired_fisher_samples is None or args.sketch_a_factor_updates == 1:
                if down.baseline is None:
                    raise AssertionError("one-sided control was not encoded")
                baseline_candidate = down.baseline
                factor_input_baseline = None
            else:
                baseline_candidate = encode_uniform_sqg_baseline(
                    source_coordinates.down,
                    h2,
                    bits=2,
                    device=device,
                    quantizer_module=quantizer_module,
                    input_sign_seed=down_seeds.input_sign,
                    output_sign_seed=down_seeds.output_sign,
                    rate_axis=matrix_rate_axis("w2"),
                    ldlq_tf32=args.ldlq_tf32,
                    tailbite_context=args.tailbite_context,
                )
                if down.baseline is None:
                    raise AssertionError("factor-input control was not encoded")
                factor_input_baseline = down.baseline
            payload_closure = None
            if args.payload_overlay is not None:
                plan = plan_qsrt_matrix(
                    torch.zeros(768, dtype=torch.long, device=device),
                    K2,
                    matrix="w2",
                    layout="importance_ordered",
                )
                packed = finalize_qsrt_matrix_candidate(
                    QSRTMatrixCandidate(
                        reconstruction=down.two_sided.reconstruction,
                        encoded=down.two_sided.states,
                        tensors={"suh": down.suh, "svh": down.svh},
                        plan=plan,
                        proxy=float(down.two_sided.two_sided_sse),
                        transform_seeds=down_seeds,
                        global_scale=down.global_scale,
                    ),
                    layer=args.layer,
                    logical_trellis_schema=resolved_schema,
                    codebook=str(candidate_manifest["codebook"]),
                    tailbite_context=args.tailbite_context,
                )
                for part, value in packed.tensors.items():
                    payload_overlay[
                        candidate_tensor_name(
                            args.layer, expert, "w2", part
                        )
                    ] = value.detach().cpu().contiguous()
                payload_closure = {
                    "stored_decode_vs_encoder": packed.coding[
                        "stored_decode_vs_encoder"
                    ],
                    "scoring_reconstruction_vs_stored": packed.coding[
                        "scoring_reconstruction_vs_stored"
                    ],
                    "trellis_bytes": int(
                        packed.tensors["trellis"].numel()
                        * packed.tensors["trellis"].element_size()
                    ),
                    "scale_bytes": int(
                        sum(
                            packed.tensors[name].numel()
                            * packed.tensors[name].element_size()
                            for name in ("suh", "svh")
                        )
                    ),
                }
            evaluation = None
            if not args.payload_only:
                if samples is None or output_metric is None:
                    raise AssertionError("local scoring inputs are absent")
                evaluation = _evaluation_rows(
                    samples, expert, args.maximum_evaluation_rows
                )
                evaluation = {
                    name: value.to(device) for name, value in evaluation.items()
                }
            if args.payload_only:
                baseline_score = {}
                factor_input_baseline_score = None
                two_sided_score = {}
                h2_vcd_score = None
                production_score = {}
            else:
                if evaluation is None or output_metric is None:
                    raise AssertionError("local scoring inputs are absent")
                baseline_score = _score_down_candidate(
                    source_coordinates=source_coordinates,
                    gate=production.gate,
                    up=production.up,
                    down=baseline_candidate.reconstruction,
                    production_h2=h2,
                    optimization_input_hessian=optimization_input_hessian,
                    output_hessian_coupled=output_hessian_coupled,
                    scoring_output_hessian_coupled=scoring_output_hessian_coupled,
                    execution=execution,
                    evaluation=evaluation,
                    output_metric=output_metric,
                )
                factor_input_baseline_score = None
                if factor_input_baseline is not None:
                    factor_input_baseline_score = _score_down_candidate(
                        source_coordinates=source_coordinates,
                        gate=production.gate,
                        up=production.up,
                        down=factor_input_baseline.reconstruction,
                        production_h2=h2,
                        optimization_input_hessian=optimization_input_hessian,
                        output_hessian_coupled=output_hessian_coupled,
                        scoring_output_hessian_coupled=(
                            scoring_output_hessian_coupled
                        ),
                        execution=execution,
                        evaluation=evaluation,
                        output_metric=output_metric,
                    )
                two_sided_score = _score_down_candidate(
                    source_coordinates=source_coordinates,
                    gate=production.gate,
                    up=production.up,
                    down=down.two_sided.reconstruction,
                    production_h2=h2,
                    optimization_input_hessian=optimization_input_hessian,
                    output_hessian_coupled=output_hessian_coupled,
                    scoring_output_hessian_coupled=scoring_output_hessian_coupled,
                    execution=execution,
                    evaluation=evaluation,
                    output_metric=output_metric,
                )
                h2_vcd_score = None
                if down.h2_vcd is not None:
                    h2_vcd_score = _score_down_candidate(
                        source_coordinates=source_coordinates,
                        gate=production.gate,
                        up=production.up,
                        down=down.h2_vcd.reconstruction,
                        production_h2=h2,
                        optimization_input_hessian=optimization_input_hessian,
                        output_hessian_coupled=output_hessian_coupled,
                        scoring_output_hessian_coupled=(
                            scoring_output_hessian_coupled
                        ),
                        execution=execution,
                        evaluation=evaluation,
                        output_metric=output_metric,
                    )
                production_score = _score_down_candidate(
                    source_coordinates=source_coordinates,
                    gate=production.gate,
                    up=production.up,
                    down=production.down,
                    production_h2=h2,
                    optimization_input_hessian=optimization_input_hessian,
                    output_hessian_coupled=output_hessian_coupled,
                    scoring_output_hessian_coupled=scoring_output_hessian_coupled,
                    execution=execution,
                    evaluation=evaluation,
                    output_metric=output_metric,
                )
            path_disagreement = (
                None
                if baseline_candidate is None
                else float(
                    torch.ne(baseline_candidate.states, down.two_sided.states)
                    .float()
                    .mean()
                )
            )
            h2_vcd_path_disagreement = (
                None
                if down.h2_vcd is None
                else float(
                    torch.ne(down.baseline.states, down.h2_vcd.states)
                    .float()
                    .mean()
                )
            )
            results[str(expert)] = {
                "intermediate_draw": draws[expert],
                "fit_support": h2_evidence,
                "sketch_a_support": sketch_a_evidence,
                "evaluation_support": {
                    "occurrences": int(evaluation["inputs"].shape[0]),
                    "documents": int(torch.unique(evaluation["documents"]).numel()),
                }
                if evaluation is not None
                else None,
                "upstream": (
                    "decoded sealed uniform-K2 candidate payload"
                    if production is not None
                    else "not consumed by identity input curvature"
                ),
                "w2": {
                    "production": production_score,
                    "baseline": (
                        None
                        if baseline_candidate is None
                        else {
                            **_candidate_summary(baseline_candidate),
                            **baseline_score,
                        }
                    ),
                    "production_vs_reencoded_baseline": (
                        None
                        if baseline_candidate is None
                        else _reconstruction_comparison(
                            production.down, baseline_candidate.reconstruction
                        )
                    ),
                    "factor_input_one_sided": (
                        None
                        if factor_input_baseline is None
                        or factor_input_baseline_score is None
                        else {
                            **_candidate_summary(factor_input_baseline),
                            **factor_input_baseline_score,
                        }
                    ),
                    "production_vs_two_sided": (
                        None
                        if production is None
                        else _reconstruction_comparison(
                            production.down, down.two_sided.reconstruction
                        )
                    ),
                    "two_sided": {
                        **_candidate_summary(down.two_sided),
                        **two_sided_score,
                    },
                    "h2_vcd": (
                        None
                        if down.h2_vcd is None or h2_vcd_score is None
                        else {
                            **_candidate_summary(down.h2_vcd),
                            **h2_vcd_score,
                        }
                    ),
                    "path_state_disagreement_fraction": path_disagreement,
                    "h2_vcd_path_state_disagreement_fraction": (
                        h2_vcd_path_disagreement
                    ),
                    "global_scale": down.global_scale,
                    "payload_closure": payload_closure,
                },
                "seconds": time.time() - expert_started,
            }
            receipt["experts"] = results
            _atomic_json(args.output, receipt)
            print(
                json.dumps(
                    {
                        "layer": args.layer,
                        "expert": expert,
                        "path_disagreement": path_disagreement,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if args.payload_overlay is not None:
        args.payload_overlay.parent.mkdir(parents=True, exist_ok=True)
        save_file(payload_overlay, str(args.payload_overlay))

    output_factor_receipt = {
        **output_factor_support,
        "fit_cache_run_id": fit_manifest.get("run_id"),
    }
    if output_hessian_ordinary is not None:
        output_factor_receipt.update(_factor_summary(output_hessian_ordinary))
    else:
        output_factor_receipt["factor_scope"] = "expert-conditional"

    receipt.update(
        {
            "corpus_identity": corpus_identity,
            "output_factor_support": output_factor_receipt,
            "experts": results,
            "payload_overlay_sha256": (
                None
                if args.payload_overlay is None
                else _sha256(args.payload_overlay)
            ),
            "seconds": time.time() - started,
            "complete": True,
        }
    )
    _atomic_json(args.output, receipt)


if __name__ == "__main__":
    main()

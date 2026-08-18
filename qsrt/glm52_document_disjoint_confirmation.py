"""Validate and summarize document-disjoint GLM-5.2 KLD measurements."""

from __future__ import annotations

import hashlib
import math
import os
import re
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from qsrt.correctness import sha256_file


_DIGEST = re.compile(r"[0-9a-f]{64}")


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash token IDs using the public auxiliary plan's fixed byte encoding."""

    digest = hashlib.sha256()
    for token_id in token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("token IDs must be integers")
        if not 0 <= token_id <= 0xFFFFFFFF:
            raise ValueError("token IDs must fit in an unsigned 32-bit integer")
        digest.update(struct.pack("<I", token_id))
    return digest.hexdigest()


def validate_public_reference_auxiliary_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable public-reference subset before reading its logits."""

    if (
        plan.get("schema") != "qsrt_glm52_public_reference_auxiliary_plan"
        or plan.get("schema_version") != 1
        or plan.get("status") != "frozen_before_reference_file_download"
    ):
        raise ValueError("public-reference auxiliary plan identity is invalid")
    tokenization = plan.get("tokenization")
    if not isinstance(tokenization, Mapping) or tokenization.get(
        "context_tokens"
    ) != 512:
        raise ValueError("public-reference auxiliary plan must use 512-token contexts")
    rows = plan.get("selected_chunks")
    if (
        not isinstance(rows, list)
        or plan.get("selected_document_count") != len(rows)
        or len(rows) < 2
    ):
        raise ValueError("public-reference auxiliary document count is invalid")

    document_hashes: set[str] = set()
    chunk_indices: set[int] = set()
    reference_files: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("public-reference auxiliary rows must be objects")
        document_hash = row.get("document_sha256")
        prompt_hash = row.get("prompt_token_ids_sha256")
        reference_hash = row.get("reference_file_sha256")
        if any(
            not isinstance(value, str) or _DIGEST.fullmatch(value) is None
            for value in (document_hash, prompt_hash, reference_hash)
        ):
            raise ValueError("public-reference auxiliary row has an invalid SHA-256")
        chunk = row.get("selected_chunk")
        if isinstance(chunk, bool) or not isinstance(chunk, int) or not 0 <= chunk < 565:
            raise ValueError("public-reference auxiliary chunk index is invalid")
        expected_name = f"batch_{chunk:06d}_001.safetensors"
        if row.get("reference_file") != expected_name:
            raise ValueError("public-reference auxiliary filename disagrees with its chunk")
        byte_count = row.get("reference_file_bytes")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
        ):
            raise ValueError("public-reference auxiliary file size is invalid")
        if (
            document_hash in document_hashes
            or chunk in chunk_indices
            or expected_name in reference_files
        ):
            raise ValueError("public-reference auxiliary rows must be unique")
        document_hashes.add(document_hash)
        chunk_indices.add(chunk)
        reference_files.add(expected_name)
        normalized.append(dict(row))
    return {
        "document_count": len(normalized),
        "context_tokens": 512,
        "rows": normalized,
    }


def validate_public_reference_files(
    plan: Mapping[str, Any], reference_directory: Path
) -> dict[str, Any]:
    """Verify every selected public reference by size and content hash."""

    validated = validate_public_reference_auxiliary_plan(plan)
    total_bytes = 0
    for row in validated["rows"]:
        path = reference_directory / row["reference_file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != row["reference_file_bytes"]:
            raise ValueError(f"public reference file size mismatch: {path}")
        if sha256_file(path) != row["reference_file_sha256"]:
            raise ValueError(f"public reference file SHA-256 mismatch: {path}")
        total_bytes += size
    return {
        **validated,
        "reference_directory": str(reference_directory.resolve()),
        "total_reference_bytes": total_bytes,
    }


def validate_frozen_low_rank_candidate(
    registration: Mapping[str, Any], artifact: Mapping[str, Any]
) -> dict[str, Any]:
    """Match the runtime artifact to the candidate frozen before confirmation."""

    if (
        registration.get("schema")
        != "qsrt_glm52_low_rank_down_confirmation_registration"
        or registration.get("schema_version") != 1
        or registration.get("status")
        != "frozen_before_document_disjoint_confirmation"
    ):
        raise ValueError("low-rank confirmation registration identity is invalid")
    frozen = registration.get("frozen_correction")
    report = artifact.get("report")
    if not isinstance(frozen, Mapping) or not isinstance(report, Mapping):
        raise TypeError("low-rank registration and artifact report are required")
    experts = report.get("experts")
    if not isinstance(experts, list):
        raise TypeError("low-rank artifact report has no expert records")
    matches = [row for row in experts if row.get("expert") == frozen.get("expert")]
    if len(matches) != 1:
        raise ValueError("low-rank artifact does not contain the frozen expert once")
    record = matches[0]
    required = {
        "layer": frozen.get("layer"),
        "expert": frozen.get("expert"),
        "rank": frozen.get("rank"),
        "factor_dtype": frozen.get("factor_dtype"),
        "selected_ridge_factor": frozen.get("selected_ridge_factor"),
        "factor_a_sha256": frozen.get("factor_a_sha256"),
        "factor_b_sha256": frozen.get("factor_b_sha256"),
        "logical_adapter_bytes": frozen.get("logical_factor_bytes"),
        "materialized_down_sha256": frozen.get("materialized_down_sha256"),
    }
    disagreements = {
        key: {"registered": value, "artifact": record.get(key)}
        for key, value in required.items()
        if record.get(key) != value
    }
    if disagreements:
        raise ValueError(f"runtime artifact differs from frozen correction: {disagreements}")
    return {
        "layer": int(record["layer"]),
        "expert": int(record["expert"]),
        "rank": int(record["rank"]),
        "factor_dtype": str(record["factor_dtype"]),
        "factor_a_sha256": str(record["factor_a_sha256"]),
        "factor_b_sha256": str(record["factor_b_sha256"]),
        "materialized_down_sha256": str(record["materialized_down_sha256"]),
        "logical_adapter_bytes": int(record["logical_adapter_bytes"]),
    }


def retarget_reference_symlink(link: Path, reference_file: Path) -> None:
    """Atomically point the worker-visible KLD input at one reference file."""

    if not reference_file.is_file():
        raise FileNotFoundError(reference_file)
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f".{link.name}.partial")
    temporary.unlink(missing_ok=True)
    os.symlink(reference_file, temporary)
    os.replace(temporary, link)


def _tail_metrics(values: torch.Tensor) -> dict[str, float]:
    values = values.double().flatten()
    count = max(1, math.ceil(values.numel() * 0.01))
    return {
        "p99": float(torch.quantile(values, 0.99).item()),
        "cvar1": float(torch.topk(values, count).values.mean().item()),
        "maximum": float(values.max().item()),
    }


def summarize_document_paired_kld(
    baseline_by_document: Mapping[str, torch.Tensor],
    candidate_by_document: Mapping[str, torch.Tensor],
    *,
    bootstrap_resamples: int = 20_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Summarize paired KLD with documents as the independent sampling units."""

    if set(baseline_by_document) != set(candidate_by_document) or len(
        baseline_by_document
    ) < 2:
        raise ValueError("paired KLD requires the same two or more documents")
    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples < 1
    ):
        raise ValueError("bootstrap resamples must be a positive integer")

    per_document: list[dict[str, Any]] = []
    baseline_parts: list[torch.Tensor] = []
    candidate_parts: list[torch.Tensor] = []
    for document in sorted(baseline_by_document):
        baseline = baseline_by_document[document].double().flatten()
        candidate = candidate_by_document[document].double().flatten()
        if baseline.numel() == 0 or candidate.shape != baseline.shape:
            raise ValueError("each paired document must contain equal nonempty vectors")
        if not bool(torch.isfinite(baseline).all() and torch.isfinite(candidate).all()):
            raise ValueError("paired document KLD values must be finite")
        baseline_mean = float(baseline.mean().item())
        candidate_mean = float(candidate.mean().item())
        per_document.append(
            {
                "document": document,
                "position_count": baseline.numel(),
                "baseline_mean_forward_kld": baseline_mean,
                "candidate_mean_forward_kld": candidate_mean,
                "candidate_minus_baseline_mean_forward_kld": (
                    candidate_mean - baseline_mean
                ),
            }
        )
        baseline_parts.append(baseline)
        candidate_parts.append(candidate)

    document_differences = torch.tensor(
        [row["candidate_minus_baseline_mean_forward_kld"] for row in per_document],
        dtype=torch.float64,
    )
    document_baselines = torch.tensor(
        [row["baseline_mean_forward_kld"] for row in per_document],
        dtype=torch.float64,
    )
    document_candidates = torch.tensor(
        [row["candidate_mean_forward_kld"] for row in per_document],
        dtype=torch.float64,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(bootstrap_seed)
    samples = torch.randint(
        len(per_document),
        (bootstrap_resamples, len(per_document)),
        generator=generator,
    )
    bootstrap_means = document_differences[samples].mean(dim=1)
    pooled_baseline = torch.cat(baseline_parts)
    pooled_candidate = torch.cat(candidate_parts)
    pooled_difference = pooled_candidate - pooled_baseline

    return {
        "document_count": len(per_document),
        "position_count": pooled_baseline.numel(),
        "equal_document_weight": {
            "baseline_mean_forward_kld": float(document_baselines.mean().item()),
            "candidate_mean_forward_kld": float(document_candidates.mean().item()),
            "candidate_minus_baseline_mean_forward_kld": float(
                document_differences.mean().item()
            ),
        },
        "pooled_position_weight": {
            "baseline_mean_forward_kld": float(pooled_baseline.mean().item()),
            "candidate_mean_forward_kld": float(pooled_candidate.mean().item()),
            "candidate_minus_baseline_mean_forward_kld": float(
                pooled_difference.mean().item()
            ),
            "relative_forward_kld_reduction": float(
                1.0 - pooled_candidate.mean().item() / pooled_baseline.mean().item()
            ),
            "candidate_better_position_count": int(
                torch.count_nonzero(pooled_difference < 0).item()
            ),
            "candidate_equal_position_count": int(
                torch.count_nonzero(pooled_difference == 0).item()
            ),
            "candidate_worse_position_count": int(
                torch.count_nonzero(pooled_difference > 0).item()
            ),
        },
        "document_outcomes": {
            "candidate_better": int(torch.count_nonzero(document_differences < 0)),
            "candidate_equal": int(torch.count_nonzero(document_differences == 0)),
            "candidate_worse": int(torch.count_nonzero(document_differences > 0)),
        },
        "paired_document_bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "difference_lower_95_percentile": float(
                torch.quantile(bootstrap_means, 0.025).item()
            ),
            "difference_upper_95_percentile": float(
                torch.quantile(bootstrap_means, 0.975).item()
            ),
            "difference_upper_one_sided_95_percentile": float(
                torch.quantile(bootstrap_means, 0.95).item()
            ),
        },
        "tail_metrics": {
            "baseline": _tail_metrics(pooled_baseline),
            "candidate": _tail_metrics(pooled_candidate),
        },
        "per_document": per_document,
    }


__all__ = [
    "retarget_reference_symlink",
    "summarize_document_paired_kld",
    "token_ids_sha256",
    "validate_frozen_low_rank_candidate",
    "validate_public_reference_auxiliary_plan",
    "validate_public_reference_files",
]

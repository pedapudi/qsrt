"""Generate GLM-5.2 teacher logits from captured terminal hidden states.

The official causal-language-model endpoint applies the final RMS
normalization to decoder layer 77 and multiplies by the untied language-model
head.  The preceding decoder layers are unnecessary when their exact BF16
output has already been captured.  This module vocabulary-shards that endpoint
across available GPUs and writes one independently hashable reference file per
frozen document.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from qsrt.glm52_pilot import atomic_write_json
from qsrt.glm52_terminal_teacher_assets import (
    build_terminal_teacher_asset_download_contract,
    validate_downloaded_terminal_teacher_assets,
)
from qsrt.glm52_terminal_teacher_reference import (
    HIDDEN_SIZE,
    VOCABULARY_SIZE,
    sha256_file,
    validate_document_tokenization,
    validate_terminal_teacher_reference_plan,
)


GENERATION_CONTRACT_SCHEMA = "qsrt_glm52_terminal_teacher_logit_generation"
DOCUMENT_RECEIPT_SCHEMA = "qsrt_glm52_terminal_teacher_logit_document"
REFERENCE_MANIFEST_SCHEMA = "qsrt_glm52_terminal_teacher_logit_manifest"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
)


def exact_glm52_final_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Apply the official GLM-5.2 final normalization arithmetic exactly."""

    if hidden_states.ndim != 2 or hidden_states.shape[1] != weight.numel():
        raise ValueError("terminal hidden states and normalization weight differ")
    if not hidden_states.is_floating_point() or not weight.is_floating_point():
        raise TypeError("terminal endpoint tensors must be floating point")
    input_dtype = hidden_states.dtype
    float_hidden = hidden_states.to(torch.float32)
    variance = float_hidden.pow(2).mean(-1, keepdim=True)
    normalized = float_hidden * torch.rsqrt(variance + epsilon)
    return weight * normalized.to(input_dtype)


def balanced_vocabulary_slices(
    vocabulary_size: int, shard_count: int
) -> tuple[tuple[int, int], ...]:
    """Partition a vocabulary into contiguous slices differing by at most one."""

    if vocabulary_size < 1 or shard_count < 1 or shard_count > vocabulary_size:
        raise ValueError("vocabulary and shard counts must be positive")
    quotient, remainder = divmod(vocabulary_size, shard_count)
    result = []
    start = 0
    for index in range(shard_count):
        stop = start + quotient + (1 if index < remainder else 0)
        result.append((start, stop))
        start = stop
    if start != vocabulary_size:
        raise AssertionError("vocabulary partition does not close")
    return tuple(result)


def tokenizer_file_identity(tokenizer_root: Path) -> dict[str, Any]:
    """Hash every local tokenizer file that can affect token identities."""

    tokenizer_root = tokenizer_root.resolve()
    files = {}
    for filename in TOKENIZER_FILES:
        path = tokenizer_root / filename
        if path.is_file():
            files[filename] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    if not files:
        raise ValueError("tokenizer directory contains no recognized tokenizer files")
    return {"root": str(tokenizer_root), "files": files}


def selected_document_token_receipts(
    *,
    plan: Mapping[str, Any],
    corpus_path: Path,
    tokenizer: Any,
    evaluation_tier: str,
) -> dict[str, dict[str, Any]]:
    """Retokenize selected documents and verify their captured token hashes."""

    validate_terminal_teacher_reference_plan(plan)
    if evaluation_tier not in {"screening", "confirmation"}:
        raise ValueError("evaluation tier must be screening or confirmation")
    corpus_lines = corpus_path.read_text(encoding="utf-8").splitlines()
    expected_corpus = plan["sources"]["calibration_corpus"]
    if (
        len(corpus_lines) != expected_corpus["rows"]
        or corpus_path.stat().st_size != expected_corpus["bytes"]
        or sha256_file(corpus_path) != expected_corpus["sha256"]
    ):
        raise ValueError("teacher-reference calibration corpus differs")

    receipts: dict[str, dict[str, Any]] = {}
    for row in plan["documents"]:
        if row["evaluation_tier"] != evaluation_tier:
            continue
        record = json.loads(corpus_lines[row["corpus_line"]])
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError("teacher-reference corpus row has no text")
        token_ids = list(tokenizer.encode(text))[
            : int(row["source_document_tokens"])
        ]
        receipt = validate_document_tokenization(
            row=row, text=text, token_ids=token_ids
        )
        receipts[row["reference_file"]] = receipt
    expected_count = 8 if evaluation_tier == "screening" else 32
    if len(receipts) != expected_count:
        raise ValueError("teacher-reference token receipt count differs")
    return receipts


def _raw_bf16_tensor(path: Path, shape: Sequence[int]) -> torch.Tensor:
    elements = 1
    for dimension in shape:
        elements *= int(dimension)
    expected_bytes = elements * 2
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"raw BF16 tensor size differs at {path}")
    return torch.from_file(
        str(path), shared=False, size=elements, dtype=torch.bfloat16
    ).reshape(*shape)


def _asset_by_name(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result = {str(asset["name"]): asset for asset in contract["assets"]}
    if len(result) != contract["asset_count"]:
        raise ValueError("teacher-reference asset names are not unique")
    return result


def _runtime_identity(devices: Sequence[torch.device]) -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "devices": [
            {
                "device": str(device),
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
            }
            for device in devices
        ],
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "allow_bf16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def build_terminal_teacher_logit_generation_contract(
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    asset_complete_sha256: str,
    tokenizer_identity: Mapping[str, Any],
    evaluation_tier: str,
    devices: Sequence[torch.device],
    closure_rows: int,
) -> dict[str, Any]:
    """Bind reference outputs to inputs, endpoint arithmetic, and GPU runtime."""

    validated = validate_terminal_teacher_reference_plan(plan)
    if evaluation_tier not in {"screening", "confirmation"}:
        raise ValueError("evaluation tier must be screening or confirmation")
    if closure_rows < 1:
        raise ValueError("numerical closure needs at least one row")
    tier_count = (
        validated["screening_document_count"]
        if evaluation_tier == "screening"
        else validated["confirmation_document_count"]
    )
    vocabulary_slices = balanced_vocabulary_slices(
        VOCABULARY_SIZE, len(devices)
    )
    return {
        "schema": GENERATION_CONTRACT_SCHEMA,
        "schema_version": 1,
        "teacher_reference_plan_sha256": plan_sha256,
        "asset_complete_sha256": asset_complete_sha256,
        "evaluation_tier": evaluation_tier,
        "document_count": tier_count,
        "tokenizer": dict(tokenizer_identity),
        "endpoint": {
            "equation": "lm_head(final_rms_norm(layer_77_output))",
            "normalization_variance_dtype": "float32",
            "normalization_output_dtype": "bfloat16",
            "head_weight_dtype": "bfloat16",
            "logit_dtype": "bfloat16",
            "rms_epsilon": plan["reference_assets"]["final_normalization"][
                "epsilon"
            ],
            "hidden_size": HIDDEN_SIZE,
            "vocabulary_size": VOCABULARY_SIZE,
        },
        "vocabulary_slices": [list(value) for value in vocabulary_slices],
        "numerical_closure_rows": closure_rows,
        "runtime": _runtime_identity(devices),
    }


def _prepare_generation_destination(
    *, destination: Path, contract: Mapping[str, Any]
) -> Path:
    destination = destination.resolve()
    contract_path = destination / "generation_contract.json"
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if contract_path.is_file():
        if json.loads(contract_path.read_text()) != dict(contract):
            raise ValueError("reference destination belongs to another generation contract")
    else:
        if any(destination.iterdir()):
            raise ValueError("unbound reference destination is not empty")
        atomic_write_json(contract_path, contract)
    return destination


def _load_endpoint_shards(
    *,
    assets_root: Path,
    asset_contract: Mapping[str, Any],
    devices: Sequence[torch.device],
) -> list[dict[str, Any]]:
    assets = _asset_by_name(asset_contract)
    head_asset = assets["language_model_head"]
    norm_asset = assets["final_normalization"]
    head_path = assets_root / head_asset["destination"]
    norm_path = assets_root / norm_asset["destination"]
    head_cpu = _raw_bf16_tensor(head_path, (VOCABULARY_SIZE, HIDDEN_SIZE))
    norm_cpu = _raw_bf16_tensor(norm_path, (HIDDEN_SIZE,))
    slices = balanced_vocabulary_slices(VOCABULARY_SIZE, len(devices))

    def load_one(item: tuple[torch.device, tuple[int, int]]) -> dict[str, Any]:
        device, (start, stop) = item
        with torch.cuda.device(device):
            return {
                "device": device,
                "start": start,
                "stop": stop,
                "head": head_cpu[start:stop].to(device=device, non_blocking=False),
                "normalization": norm_cpu.to(device=device, non_blocking=False),
            }

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(devices)
    ) as executor:
        shards = list(executor.map(load_one, zip(devices, slices, strict=True)))
    return shards


def _compute_one_endpoint_shard(
    *,
    shard: Mapping[str, Any],
    hidden_cpu: torch.Tensor,
    epsilon: float,
    closure_rows: int,
    capture_closure: bool,
) -> dict[str, Any]:
    device = shard["device"]
    with torch.inference_mode(), torch.cuda.device(device):
        hidden = hidden_cpu.to(device=device, non_blocking=False)
        normalized = exact_glm52_final_rms_norm(
            hidden,
            shard["normalization"],
            epsilon=epsilon,
        )
        logits = F.linear(normalized, shard["head"])
        result = logits.cpu()
        if not capture_closure:
            return {"logits": result}
        rows = min(closure_rows, normalized.shape[0])
        repeated = F.linear(normalized[:rows], shard["head"])
        if not torch.equal(logits[:rows], repeated):
            raise ValueError("BF16 endpoint GEMM is not bit-repeatable")
        fp32 = F.linear(
            normalized[:rows].to(torch.float32),
            shard["head"].to(torch.float32),
        ).cpu()
        return {
            "logits": result,
            "closure_bf16": result[:rows].to(torch.float32),
            "closure_fp32": fp32,
        }


def compute_terminal_teacher_logits(
    *,
    hidden_cpu: torch.Tensor,
    endpoint_shards: Sequence[Mapping[str, Any]],
    epsilon: float,
    closure_rows: int,
    capture_closure: bool,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    """Compute full-vocabulary logits and an optional FP32 endpoint comparison."""

    if hidden_cpu.shape[1:] != (HIDDEN_SIZE,) or hidden_cpu.dtype != torch.bfloat16:
        raise ValueError("terminal hidden tensor must be BF16 with width 6144")

    def compute(shard: Mapping[str, Any]) -> dict[str, Any]:
        return _compute_one_endpoint_shard(
            shard=shard,
            hidden_cpu=hidden_cpu,
            epsilon=epsilon,
            closure_rows=closure_rows,
            capture_closure=capture_closure,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(endpoint_shards)
    ) as executor:
        results = list(executor.map(compute, endpoint_shards))
    logits = torch.cat([result["logits"] for result in results], dim=-1)
    if logits.shape != (hidden_cpu.shape[0], VOCABULARY_SIZE):
        raise ValueError("assembled teacher-logit shape differs")
    if not capture_closure:
        return logits, None
    bf16 = torch.cat([result["closure_bf16"] for result in results], dim=-1)
    fp32 = torch.cat([result["closure_fp32"] for result in results], dim=-1)
    difference = bf16 - fp32
    log_fp32 = F.log_softmax(fp32, dim=-1)
    log_bf16 = F.log_softmax(bf16, dim=-1)
    forward_kld = (log_fp32.exp() * (log_fp32 - log_bf16)).sum(dim=-1)
    closure = {
        "rows": int(fp32.shape[0]),
        "bf16_repeat_bit_exact": True,
        "maximum_absolute_logit_difference_from_fp32_head": float(
            difference.abs().max().item()
        ),
        "mean_absolute_logit_difference_from_fp32_head": float(
            difference.abs().mean().item()
        ),
        "mean_forward_kld_fp32_head_to_bf16_head": float(
            forward_kld.mean().item()
        ),
        "maximum_forward_kld_fp32_head_to_bf16_head": float(
            forward_kld.max().item()
        ),
    }
    return logits, closure


def _existing_document_receipt(
    *, output_path: Path, receipt_path: Path, expected: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not output_path.exists() and not receipt_path.exists():
        return None
    if not output_path.is_file() or not receipt_path.is_file():
        raise ValueError(f"incomplete teacher-reference output at {output_path}")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema") != DOCUMENT_RECEIPT_SCHEMA
        or receipt.get("document") != dict(expected)
        or receipt.get("bytes") != output_path.stat().st_size
        or receipt.get("sha256") != sha256_file(output_path)
    ):
        raise ValueError(f"teacher-reference receipt differs at {output_path}")
    return receipt


def _write_reference_file(
    *,
    output_path: Path,
    logits: torch.Tensor,
    prompt_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
    metadata: Mapping[str, str],
) -> None:
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    tensors = {
        "logits": logits.contiguous(),
        "prompt_token_ids": torch.tensor(prompt_token_ids, dtype=torch.int32),
        "target_token_ids": torch.tensor(target_token_ids, dtype=torch.int32),
    }
    try:
        save_file(tensors, str(temporary), metadata=dict(metadata))
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def generate_terminal_teacher_references(
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    asset_contract: Mapping[str, Any],
    assets_root: Path,
    asset_complete_sha256: str,
    token_receipts: Mapping[str, Mapping[str, Any]],
    tokenizer_identity: Mapping[str, Any],
    evaluation_tier: str,
    devices: Sequence[torch.device],
    closure_rows: int,
    destination: Path,
) -> dict[str, Any]:
    """Generate or resume one tier of document-level teacher references."""

    validate_downloaded_terminal_teacher_assets(
        contract=asset_contract, destination=assets_root
    )
    contract = build_terminal_teacher_logit_generation_contract(
        plan=plan,
        plan_sha256=plan_sha256,
        asset_complete_sha256=asset_complete_sha256,
        tokenizer_identity=tokenizer_identity,
        evaluation_tier=evaluation_tier,
        devices=devices,
        closure_rows=closure_rows,
    )
    destination = _prepare_generation_destination(
        destination=destination, contract=contract
    )
    endpoint_shards = _load_endpoint_shards(
        assets_root=assets_root,
        asset_contract=asset_contract,
        devices=devices,
    )
    assets = _asset_by_name(asset_contract)
    epsilon = float(
        plan["reference_assets"]["final_normalization"]["epsilon"]
    )
    document_receipts = []
    closure_path = destination / "numerical_closure.json"
    numerical_closure = (
        json.loads(closure_path.read_text()) if closure_path.is_file() else None
    )
    tier_documents = [
        row
        for row in plan["documents"]
        if row["evaluation_tier"] == evaluation_tier
    ]
    for row in tier_documents:
        token_receipt = token_receipts[row["reference_file"]]
        expected = {
            "reference_plan_row": dict(row),
            "token_receipt": dict(token_receipt),
        }
        output_path = destination / row["reference_file"]
        receipt_path = output_path.with_name(output_path.name + ".receipt.json")
        existing = _existing_document_receipt(
            output_path=output_path,
            receipt_path=receipt_path,
            expected=expected,
        )
        if existing is not None:
            document_receipts.append(existing)
            continue
        hidden_asset = assets[f"terminal_hidden_{row['document_sha256']}"]
        hidden_path = assets_root / hidden_asset["destination"]
        hidden = _raw_bf16_tensor(hidden_path, (row["logit_rows"], HIDDEN_SIZE))
        logits, closure = compute_terminal_teacher_logits(
            hidden_cpu=hidden,
            endpoint_shards=endpoint_shards,
            epsilon=epsilon,
            closure_rows=closure_rows,
            capture_closure=numerical_closure is None,
        )
        if closure is not None:
            numerical_closure = {
                "document_sha256": row["document_sha256"],
                **closure,
            }
            atomic_write_json(closure_path, numerical_closure)
        _write_reference_file(
            output_path=output_path,
            logits=logits,
            prompt_token_ids=token_receipt["prompt_token_ids"],
            target_token_ids=token_receipt["target_token_ids"],
            metadata={
                "schema": DOCUMENT_RECEIPT_SCHEMA,
                "teacher_model": plan["teacher"]["model_id"],
                "teacher_revision": plan["teacher"]["source_revision"],
                "document_sha256": row["document_sha256"],
                "evaluation_tier": evaluation_tier,
                "logit_equation": contract["endpoint"]["equation"],
            },
        )
        receipt = {
            "schema": DOCUMENT_RECEIPT_SCHEMA,
            "schema_version": 1,
            "document": expected,
            "file": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "logits_key": "logits",
            "logits_shape": list(logits.shape),
            "logits_dtype": str(logits.dtype).removeprefix("torch."),
            "completed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        }
        atomic_write_json(receipt_path, receipt)
        document_receipts.append(receipt)
    if numerical_closure is None:
        raise ValueError("reference generation lacks a numerical closure record")

    for receipt in document_receipts:
        path = destination / receipt["file"]
        with safe_open(path, framework="pt", device="cpu") as handle:
            if handle.get_slice("logits").get_shape() != receipt["logits_shape"]:
                raise ValueError(f"teacher-reference tensor shape differs at {path}")
    manifest = {
        "schema": REFERENCE_MANIFEST_SCHEMA,
        "schema_version": 1,
        "status": (
            "available_for_candidate_screening"
            if evaluation_tier == "screening"
            else "sealed_until_candidate_construction_and_bytes_are_frozen"
        ),
        "generation_contract": contract,
        "numerical_closure": numerical_closure,
        "documents": document_receipts,
        "document_count": len(document_receipts),
        "total_logit_rows": sum(
            receipt["logits_shape"][0] for receipt in document_receipts
        ),
        "total_payload_bytes": sum(
            receipt["bytes"] for receipt in document_receipts
        ),
        "completed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    atomic_write_json(destination / "manifest.json", manifest)
    return manifest


__all__ = [
    "DOCUMENT_RECEIPT_SCHEMA",
    "GENERATION_CONTRACT_SCHEMA",
    "REFERENCE_MANIFEST_SCHEMA",
    "balanced_vocabulary_slices",
    "build_terminal_teacher_logit_generation_contract",
    "compute_terminal_teacher_logits",
    "exact_glm52_final_rms_norm",
    "generate_terminal_teacher_references",
    "selected_document_token_receipts",
    "tokenizer_file_identity",
]

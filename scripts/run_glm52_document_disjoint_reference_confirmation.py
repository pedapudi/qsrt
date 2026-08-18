#!/usr/bin/env python3
"""Measure a frozen GLM-5.2 correction on public document-disjoint references.

The runner loads the resident EXL3 checkpoint once. It then evaluates the
unchanged resident and the frozen layer-3 expert-103 correction on one
published 512-token reference chunk from each untouched WikiText document.
The public reference files contain BF16-teacher log-probabilities from the
same source weights as the bounded QSRT source shards.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

from qsrt.correctness import sha256_file
from qsrt.glm52_document_disjoint_confirmation import (
    retarget_reference_symlink,
    summarize_document_paired_kld,
    token_ids_sha256,
    validate_frozen_low_rank_candidate,
    validate_public_reference_files,
)
from qsrt.glm52_engine_kld import (
    ENGINE_KLD_CHUNK_ROWS_ENV,
    ENGINE_KLD_REFERENCE_KEY_ENV,
    ENGINE_KLD_REFERENCE_PATH_ENV,
    ENGINE_KLD_REFERENCE_REPRESENTATION_ENV,
    engine_kld_from_prompt_logprobs,
)
from qsrt.glm52_expert_intervention_runtime import (
    FORCE_PER_EXPERT_EXL3_MOE_ENV,
    IDENTITY_CONTROL_MODE,
    MATERIALIZED_LOW_RANK_CANDIDATE_MODE,
    atomic_write_control,
    validate_dense_intervention_artifact,
)
from qsrt.glm52_paired_kld import route_support_summary, target_layer_routes
from qsrt.glm52_pilot import atomic_write_json


def _public_reference_token_ids(
    *, model: Path, plan: dict[str, Any]
) -> tuple[list[int], dict[str, Any]]:
    dependency_path = os.getenv("KLD_PYDEPS")
    if dependency_path:
        sys.path.append(dependency_path)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset_contract = plan["dataset"]
    dataset = load_dataset(
        dataset_contract["id"],
        dataset_contract["configuration"],
        split=dataset_contract["split"],
    )
    rows = [str(row.get("text", "")) for row in dataset]
    if len(rows) != dataset_contract["row_count"]:
        raise ValueError("public-reference dataset row count changed")
    joined_text = "\n".join(rows)
    import hashlib

    if hashlib.sha256(joined_text.encode()).hexdigest() != dataset_contract[
        "joined_text_sha256"
    ]:
        raise ValueError("public-reference joined text changed")
    tokenizer = AutoTokenizer.from_pretrained(
        model, trust_remote_code=True, local_files_only=True
    )
    encoded = tokenizer(joined_text, add_special_tokens=False)
    token_ids = encoded["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    token_ids = [int(value) for value in token_ids]
    tokenization = plan["tokenization"]
    if (
        len(token_ids) != tokenization["full_token_count"]
        or token_ids[:16] != tokenization["first16"]
    ):
        raise ValueError("public-reference tokenization changed")
    return token_ids, {
        "dataset_rows": len(rows),
        "full_token_count": len(token_ids),
        "first16": token_ids[:16],
    }


def _validate_reference_tensor(
    path: Path, *, prompt_token_ids: list[int]
) -> list[int]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if keys != {"logprobs", "target_token_ids"}:
            raise ValueError(f"public reference has unexpected tensors: {path}")
        logprob_shape = list(handle.get_slice("logprobs").get_shape())
        target_shape = list(handle.get_slice("target_token_ids").get_shape())
        if logprob_shape != [1, 511, 154880] or target_shape != [1, 511]:
            raise ValueError(f"public reference tensor shape changed: {path}")
        targets = [
            int(value)
            for value in handle.get_tensor("target_token_ids")[0].tolist()
        ]
    if targets != prompt_token_ids[1:]:
        raise ValueError(f"public reference target tokens changed: {path}")
    return targets


def _measure_kld_and_routes(
    llm: Any, *, token_ids: list[int]
) -> tuple[torch.Tensor, np.ndarray]:
    from vllm import SamplingParams

    parameters = inspect.signature(SamplingParams).parameters
    kwargs: dict[str, Any] = {
        "prompt_logprobs": 1,
        "max_tokens": 1,
        "routed_experts_prompt_start": 0,
        "detokenize": False,
    }
    if "flat_logprobs" in parameters:
        kwargs["flat_logprobs"] = True
    output = llm.generate(
        [
            {
                "prompt_token_ids": token_ids,
                "target_token_ids": token_ids[1:],
            }
        ],
        sampling_params=SamplingParams(**kwargs),
    )[0]
    if output.prompt_logprobs is None:
        raise RuntimeError("vLLM returned no engine KLD values")
    kld = engine_kld_from_prompt_logprobs(
        output.prompt_logprobs,
        positions=len(token_ids) - 1,
        target_token_ids=token_ids[1:],
    )
    routed_experts = output.outputs[0].routed_experts
    if routed_experts is None:
        raise RuntimeError("vLLM returned no routed-expert capture")
    routes = np.asarray(routed_experts)
    if routes.ndim != 3 or routes.shape[0] != len(token_ids):
        raise RuntimeError("vLLM routed-expert capture shape changed")
    return kld, routes


def _changed_route_summary(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, int]:
    if baseline.shape != candidate.shape:
        raise ValueError("baseline and candidate route arrays have different shapes")
    changed = baseline != candidate
    return {
        "changed_route_identifier_count": int(np.count_nonzero(changed)),
        "changed_token_layer_row_count": int(np.count_nonzero(np.any(changed, axis=2))),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--reference-plan", type=Path, required=True)
    parser.add_argument("--reference-link", type=Path, required=True)
    parser.add_argument("--intervention-artifact", type=Path, required=True)
    parser.add_argument("--confirmation-registration", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.89)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--kv-cache-dtype", default="bfloat16")
    parser.add_argument("--load-format", default="safetensors")
    parser.add_argument("--quantization", default="exl3")
    parser.add_argument("--attention-backend", default="TRITON_MLA")
    parser.add_argument("--max-model-len", type=int, default=513)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--kld-chunk-rows", type=int, default=4)
    parser.add_argument("--hf-overrides", default='{"index_topk":0,"use_index_cache":false}')
    parser.add_argument(
        "--llm-extra-json",
        default=(
            '{"decode_context_parallel_size":1,"moe_backend":"b12x",'
            '"enforce_eager":true,"disable_custom_all_reduce":true,'
            '"async_scheduling":false}'
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.dest.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.reference_plan.read_text())
    references = validate_public_reference_files(plan, args.reference_directory)
    artifact = validate_dense_intervention_artifact(args.intervention_artifact)
    registration = json.loads(args.confirmation_registration.read_text())
    frozen = validate_frozen_low_rank_candidate(registration, artifact)
    if frozen["expert"] != 103:
        raise ValueError("public-reference confirmation requires frozen expert 103")

    configured_link = Path(os.environ[ENGINE_KLD_REFERENCE_PATH_ENV])
    if configured_link.absolute() != args.reference_link.absolute():
        raise ValueError("engine KLD reference link differs from the runner input")
    if os.environ.get(ENGINE_KLD_REFERENCE_KEY_ENV) != "logprobs":
        raise ValueError("engine KLD reference tensor key must be 'logprobs'")
    if os.environ.get(ENGINE_KLD_REFERENCE_REPRESENTATION_ENV) != "logprobs":
        raise ValueError("engine KLD reference representation must be 'logprobs'")
    if os.environ.get(ENGINE_KLD_CHUNK_ROWS_ENV) != str(args.kld_chunk_rows):
        raise ValueError("engine and runner KLD chunk-row settings differ")

    all_token_ids, tokenization_receipt = _public_reference_token_ids(
        model=args.model, plan=plan
    )
    document_inputs: list[dict[str, Any]] = []
    for row in references["rows"]:
        chunk = row["selected_chunk"]
        prompt = all_token_ids[chunk * 512 : (chunk + 1) * 512]
        if len(prompt) != 512 or token_ids_sha256(prompt) != row[
            "prompt_token_ids_sha256"
        ]:
            raise ValueError("public-reference prompt token hash changed")
        reference_path = args.reference_directory / row["reference_file"]
        _validate_reference_tensor(reference_path, prompt_token_ids=prompt)
        document_inputs.append(
            {**row, "prompt_token_ids": prompt, "reference_path": reference_path}
        )

    first = document_inputs[0]
    retarget_reference_symlink(args.reference_link, first["reference_path"])
    atomic_write_control(
        args.control,
        mode="off",
        artifact_manifest_sha256=artifact["manifest_sha256"],
        generation=0,
    )
    from vllm import LLM

    llm_kwargs: dict[str, Any] = {
        "model": str(args.model),
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "load_format": args.load_format,
        "quantization": args.quantization,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": 1,
        "attention_backend": args.attention_backend,
        "hf_overrides": json.loads(args.hf_overrides),
        "enable_prefix_caching": False,
        "disable_log_stats": True,
        "max_logprobs": -1,
        "enable_return_routed_experts": True,
        "enforce_eager": True,
    }
    llm_kwargs.update(json.loads(args.llm_extra_json))
    started = time.monotonic()
    llm = LLM(**llm_kwargs)
    model_load_seconds = time.monotonic() - started

    generation = 0

    def measure(
        row: dict[str, Any], *, mode: str, selected_experts: list[int] | None
    ) -> tuple[torch.Tensor, np.ndarray, float]:
        nonlocal generation
        generation += 1
        retarget_reference_symlink(args.reference_link, row["reference_path"])
        atomic_write_control(
            args.control,
            mode=mode,
            artifact_manifest_sha256=artifact["manifest_sha256"],
            generation=generation,
            selected_experts=selected_experts,
        )
        arm_started = time.monotonic()
        kld, routes = _measure_kld_and_routes(
            llm, token_ids=row["prompt_token_ids"]
        )
        return kld, routes, time.monotonic() - arm_started

    first_baseline, first_routes, first_seconds = measure(
        first, mode="off", selected_experts=None
    )
    first_repeat, first_repeat_routes, _ = measure(
        first, mode="off", selected_experts=None
    )
    first_identity, first_identity_routes, _ = measure(
        first, mode=IDENTITY_CONTROL_MODE, selected_experts=[103]
    )
    controls_passed = bool(
        torch.equal(first_baseline, first_repeat)
        and torch.equal(first_baseline, first_identity)
        and np.array_equal(first_routes, first_repeat_routes)
        and np.array_equal(first_routes, first_identity_routes)
    )
    if not controls_passed:
        raise RuntimeError("public-reference measurement controls failed")

    baseline_by_document: dict[str, torch.Tensor] = {}
    candidate_by_document: dict[str, torch.Tensor] = {}
    document_reports: list[dict[str, Any]] = []
    final_baseline: torch.Tensor | None = None
    final_routes: np.ndarray | None = None
    for index, row in enumerate(document_inputs):
        if index == 0:
            baseline, baseline_routes, baseline_seconds = (
                first_baseline,
                first_routes,
                first_seconds,
            )
        else:
            baseline, baseline_routes, baseline_seconds = measure(
                row, mode="off", selected_experts=None
            )
        candidate, candidate_routes, candidate_seconds = measure(
            row,
            mode=MATERIALIZED_LOW_RANK_CANDIDATE_MODE,
            selected_experts=[103],
        )
        layer_baseline_routes = target_layer_routes(
            baseline_routes,
            model_layer=3,
            total_decoder_layers=78,
            first_moe_layer=3,
        )
        layer_candidate_routes = target_layer_routes(
            candidate_routes,
            model_layer=3,
            total_decoder_layers=78,
            first_moe_layer=3,
        )
        if not np.array_equal(layer_baseline_routes, layer_candidate_routes):
            raise RuntimeError("layer-3 routes changed before the frozen intervention")
        document_hash = row["document_sha256"]
        baseline_by_document[document_hash] = baseline
        candidate_by_document[document_hash] = candidate
        stem = f"chunk-{row['selected_chunk']:06d}"
        torch.save(baseline, args.dest / f"{stem}-resident-forward-kld.pt")
        torch.save(candidate, args.dest / f"{stem}-candidate-forward-kld.pt")
        np.savez_compressed(
            args.dest / f"{stem}-routes.npz",
            resident=baseline_routes,
            candidate=candidate_routes,
        )
        document_reports.append(
            {
                "document_sha256": document_hash,
                "document_title": row["document_title"],
                "selected_chunk": row["selected_chunk"],
                "reference_file": row["reference_file"],
                "reference_file_sha256": row["reference_file_sha256"],
                "resident_mean_forward_kld": float(baseline.mean().item()),
                "candidate_mean_forward_kld": float(candidate.mean().item()),
                "candidate_minus_resident_mean_forward_kld": float(
                    (candidate - baseline).mean().item()
                ),
                "resident_seconds": baseline_seconds,
                "candidate_seconds": candidate_seconds,
                "layer_3_expert_103_route_support": route_support_summary(
                    layer_baseline_routes, selected_experts=[103]
                ),
                "all_layer_routes": _changed_route_summary(
                    baseline_routes, candidate_routes
                ),
            }
        )
        final_baseline, final_routes = baseline, baseline_routes

    assert final_baseline is not None and final_routes is not None
    final = document_inputs[-1]
    final_repeat, final_repeat_routes, _ = measure(
        final, mode="off", selected_experts=None
    )
    final_bracketing_control_passed = bool(
        torch.equal(final_baseline, final_repeat)
        and np.array_equal(final_routes, final_repeat_routes)
    )
    if not final_bracketing_control_passed:
        raise RuntimeError("final resident bracketing control failed")

    summary = summarize_document_paired_kld(
        baseline_by_document,
        candidate_by_document,
        bootstrap_resamples=20_000,
        bootstrap_seed=0,
    )
    report = {
        "schema": "qsrt_glm52_document_disjoint_public_reference_confirmation",
        "schema_version": 1,
        "status": "complete",
        "candidate": frozen,
        "intervention_artifact": {
            "root": artifact["root"],
            "manifest_sha256": artifact["manifest_sha256"],
            "expert_ids": list(artifact["expert_ids"]),
        },
        "confirmation_registration": {
            "path": str(args.confirmation_registration.resolve()),
            "sha256": sha256_file(args.confirmation_registration),
        },
        "public_reference_plan": {
            "path": str(args.reference_plan.resolve()),
            "sha256": sha256_file(args.reference_plan),
            "repository": plan["reference_repository"],
            "revision": plan["reference_revision"],
            "teacher_model": plan["teacher_model"],
            "teacher_revision": plan["teacher_revision"],
            "total_reference_bytes": references["total_reference_bytes"],
        },
        "tokenization": tokenization_receipt,
        "measurement_controls": {
            "initial_resident_repeat_bitwise_equal": bool(
                torch.equal(first_baseline, first_repeat)
            ),
            "initial_identity_kld_bitwise_equal": bool(
                torch.equal(first_baseline, first_identity)
            ),
            "initial_resident_repeat_routes_equal": bool(
                np.array_equal(first_routes, first_repeat_routes)
            ),
            "initial_identity_routes_equal": bool(
                np.array_equal(first_routes, first_identity_routes)
            ),
            "final_resident_bracketing_kld_bitwise_equal": bool(
                torch.equal(final_baseline, final_repeat)
            ),
            "final_resident_bracketing_routes_equal": bool(
                np.array_equal(final_routes, final_repeat_routes)
            ),
            "passed": bool(controls_passed and final_bracketing_control_passed),
        },
        "summary": summary,
        "documents": document_reports,
        "runtime": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "kv_cache_dtype": args.kv_cache_dtype,
            "load_format": args.load_format,
            "attention_backend": args.attention_backend,
            "context_tokens": 512,
            "model_load_seconds": model_load_seconds,
            "engine_reference_representation": "logprobs",
            "kld_chunk_rows": args.kld_chunk_rows,
            "exl3_moe_execution": (
                "three_gemm_per_expert_correctness"
                if os.environ.get(FORCE_PER_EXPERT_EXL3_MOE_ENV, "0") == "1"
                else "checkpoint_selected_fused_kernel"
            ),
            "llm_kwargs": llm_kwargs,
        },
        "numerical_target": {
            "target_mean_forward_kld": 0.059,
            "pooled_candidate_below_target": bool(
                summary["pooled_position_weight"]["candidate_mean_forward_kld"]
                < 0.059
            ),
            "equal_document_candidate_below_target": bool(
                summary["equal_document_weight"]["candidate_mean_forward_kld"]
                < 0.059
            ),
        },
        "evidence_boundary": (
            "The candidate was frozen before these reference files were accessed, "
            "and all 16 documents are disjoint from its fit, ridge-selection, "
            "attribution, and candidate-selection documents. The public cache uses "
            "512-token contexts and contains only 16 eligible untouched documents. "
            "This result therefore supplies independent auxiliary evidence but does "
            "not satisfy the registered requirement for 32 documents with 2,048 "
            "tokens each."
        ),
        "model_downloads_performed": False,
    }
    atomic_write_json(args.dest / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

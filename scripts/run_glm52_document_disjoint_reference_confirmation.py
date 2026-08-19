#!/usr/bin/env python3
"""Evaluate a frozen GLM-5.2 intervention on document-disjoint references.

The runner loads the resident EXL3 checkpoint once and evaluates the unchanged
resident and one prebuilt intervention on every document in one reference
tier. It supports the published 512-token WikiText auxiliary references and
the bounded terminal-hidden-state references generated from the exact GLM-5.2
source endpoint. The thirty-two-document terminal confirmation tier cannot be
opened without a content-addressed candidate-freeze record.
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
    evaluate_terminal_reference_confirmation_decision,
    retarget_reference_symlink,
    summarize_document_paired_kld,
    token_ids_sha256,
    validate_frozen_low_rank_candidate,
    validate_public_reference_files,
    validate_terminal_reference_confirmation_freeze,
)
from qsrt.glm52_engine_kld import (
    ENGINE_KLD_CHUNK_ROWS_ENV,
    ENGINE_KLD_REFERENCE_KEY_ENV,
    ENGINE_KLD_REFERENCE_PATH_ENV,
    ENGINE_KLD_REFERENCE_REPRESENTATION_ENV,
    engine_kld_from_prompt_logprobs,
)
from qsrt.glm52_expert_intervention_runtime import (
    CANDIDATE_MODE,
    FORCE_PER_EXPERT_EXL3_MOE_ENV,
    IDENTITY_CONTROL_MODE,
    MATERIALIZED_LOW_RANK_CANDIDATE_MODE,
    atomic_write_control,
    validate_intervention_artifact,
)
from qsrt.glm52_paired_kld import route_support_summary, target_layer_routes
from qsrt.glm52_pilot import atomic_write_json
from qsrt.glm52_terminal_teacher_logits import (
    load_validated_terminal_teacher_reference_documents,
)


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
    reference_kind = parser.add_mutually_exclusive_group(required=True)
    reference_kind.add_argument("--reference-plan", type=Path)
    reference_kind.add_argument("--terminal-reference-plan", type=Path)
    parser.add_argument(
        "--evaluation-tier", choices=("screening", "confirmation")
    )
    parser.add_argument("--reference-link", type=Path, required=True)
    parser.add_argument("--intervention-artifact", type=Path, required=True)
    parser.add_argument("--confirmation-registration", type=Path)
    parser.add_argument("--terminal-confirmation-freeze", type=Path)
    parser.add_argument("--terminal-screening-report", type=Path)
    parser.add_argument(
        "--candidate-runtime-mode",
        choices=(CANDIDATE_MODE, MATERIALIZED_LOW_RANK_CANDIDATE_MODE),
        default=MATERIALIZED_LOW_RANK_CANDIDATE_MODE,
    )
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
    artifact = validate_intervention_artifact(args.intervention_artifact)
    if args.confirmation_registration is not None:
        if artifact["artifact_kind"] != "single_layer":
            raise ValueError(
                "a low-rank confirmation registration cannot select from a "
                "multi-layer intervention"
            )
        registration = json.loads(args.confirmation_registration.read_text())
        frozen = validate_frozen_low_rank_candidate(registration, artifact)
        selected_experts = [frozen["expert"]]
    elif artifact["artifact_kind"] == "multi_layer":
        frozen = {
            "role": (
                "complete multi-layer intervention frozen before "
                "document-disjoint reference scoring"
            ),
            "model_layers": list(artifact["model_layers"]),
            "expert_ids_by_layer": {
                str(layer): list(experts)
                for layer, experts in artifact["expert_ids_by_layer"].items()
            },
            "artifact_manifest_sha256": artifact["manifest_sha256"],
            "candidate_runtime_mode": args.candidate_runtime_mode,
        }
        selected_experts = None
    else:
        frozen = {
            "role": (
                "complete intervention artifact frozen before document-disjoint "
                "reference scoring"
            ),
            "model_layer": artifact["model_layer"],
            "expert_ids": list(artifact["expert_ids"]),
            "artifact_manifest_sha256": artifact["manifest_sha256"],
            "candidate_runtime_mode": args.candidate_runtime_mode,
        }
        selected_experts = list(artifact["expert_ids"])

    terminal_confirmation_freeze: dict[str, Any] | None = None
    if args.reference_plan is not None:
        if (
            args.evaluation_tier is not None
            or args.terminal_confirmation_freeze is not None
            or args.terminal_screening_report is not None
        ):
            raise ValueError(
                "terminal-reference options cannot accompany a public-reference plan"
            )
        plan = json.loads(args.reference_plan.read_text())
        references = validate_public_reference_files(plan, args.reference_directory)
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
                {
                    **row,
                    "prompt_token_ids": prompt,
                    "target_token_ids": prompt[1:],
                    "reference_path": reference_path,
                }
            )
        reference_key = "logprobs"
        reference_representation = "logprobs"
        evaluation_tier = "public_auxiliary_confirmation"
        reference_report = {
            "kind": "published_public_auxiliary",
            "plan_path": str(args.reference_plan.resolve()),
            "plan_sha256": sha256_file(args.reference_plan),
            "repository": plan["reference_repository"],
            "revision": plan["reference_revision"],
            "teacher_model": plan["teacher_model"],
            "teacher_revision": plan["teacher_revision"],
            "total_reference_bytes": references["total_reference_bytes"],
        }
        evidence_boundary = (
            "The candidate was frozen before these reference files were accessed, "
            "and all sixteen documents are disjoint from its fit, ridge-selection, "
            "attribution, and candidate-selection documents. The public cache uses "
            "512-token contexts and supplies independent auxiliary evidence; it does "
            "not satisfy the registered thirty-two-document confirmation requirement."
        )
    else:
        if args.evaluation_tier is None:
            raise ValueError("terminal references require an evaluation tier")
        assert args.terminal_reference_plan is not None
        plan_sha256 = sha256_file(args.terminal_reference_plan)
        plan = json.loads(args.terminal_reference_plan.read_text())
        if args.evaluation_tier == "confirmation":
            if (
                args.terminal_confirmation_freeze is None
                or args.terminal_screening_report is None
            ):
                raise PermissionError(
                    "terminal confirmation requires a freeze record and screening report"
                )
            terminal_confirmation_freeze = (
                validate_terminal_reference_confirmation_freeze(
                    json.loads(args.terminal_confirmation_freeze.read_text()),
                    artifact=artifact,
                    candidate_runtime_mode=args.candidate_runtime_mode,
                    teacher_reference_plan_sha256=plan_sha256,
                    screening_report_path=args.terminal_screening_report,
                )
            )
        elif (
            args.terminal_confirmation_freeze is not None
            or args.terminal_screening_report is not None
        ):
            raise ValueError(
                "terminal screening cannot consume a confirmation freeze record"
            )
        references = load_validated_terminal_teacher_reference_documents(
            plan=plan,
            plan_sha256=plan_sha256,
            reference_directory=args.reference_directory,
            evaluation_tier=args.evaluation_tier,
            confirmation_authorization=terminal_confirmation_freeze,
        )
        document_inputs = list(references["documents"])
        tokenization_receipt = {
            "source": "prompt and target token IDs embedded in each reference file",
            "document_count": references["document_count"],
            "total_logit_rows": references["total_logit_rows"],
        }
        reference_key = "logits"
        reference_representation = "logits"
        evaluation_tier = args.evaluation_tier
        reference_report = {
            "kind": "bounded_terminal_hidden_teacher_endpoint",
            "plan_path": str(args.terminal_reference_plan.resolve()),
            "plan_sha256": plan_sha256,
            "manifest_path": str(references["manifest_path"]),
            "manifest_sha256": references["manifest_sha256"],
            "teacher_model": plan["teacher"]["model_id"],
            "teacher_revision": plan["teacher"]["source_revision"],
            "evaluation_tier": evaluation_tier,
            "document_count": references["document_count"],
            "total_logit_rows": references["total_logit_rows"],
            "total_reference_bytes": references["total_payload_bytes"],
        }
        evidence_boundary = (
            "The bounded references reproduce the exact source-model endpoint from "
            "captured BF16 decoder-layer-77 outputs, the official final RMS "
            "normalization, and the official language-model head. They do not require "
            "the preceding BF16 decoder weights. Screening documents may select a "
            "candidate. Confirmation documents are opened only after the candidate, "
            "runtime mode, and exact charged bytes are frozen in a hash-bound record."
        )

    configured_link = Path(os.environ[ENGINE_KLD_REFERENCE_PATH_ENV])
    if configured_link.absolute() != args.reference_link.absolute():
        raise ValueError("engine KLD reference link differs from the runner input")
    if os.environ.get(ENGINE_KLD_REFERENCE_KEY_ENV) != reference_key:
        raise ValueError("engine KLD reference tensor key differs from the reference")
    if (
        os.environ.get(ENGINE_KLD_REFERENCE_REPRESENTATION_ENV)
        != reference_representation
    ):
        raise ValueError(
            "engine KLD reference representation differs from the reference"
        )
    if os.environ.get(ENGINE_KLD_CHUNK_ROWS_ENV) != str(args.kld_chunk_rows):
        raise ValueError("engine and runner KLD chunk-row settings differ")

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
        first, mode=IDENTITY_CONTROL_MODE, selected_experts=selected_experts
    )
    controls_passed = bool(
        torch.equal(first_baseline, first_repeat)
        and torch.equal(first_baseline, first_identity)
        and np.array_equal(first_routes, first_repeat_routes)
        and np.array_equal(first_routes, first_identity_routes)
    )
    if not controls_passed:
        raise RuntimeError("document-reference measurement controls failed")

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
            mode=args.candidate_runtime_mode,
            selected_experts=selected_experts,
        )
        earliest_model_layer = artifact["model_layers"][0]
        layer_baseline_routes = target_layer_routes(
            baseline_routes,
            model_layer=earliest_model_layer,
            total_decoder_layers=78,
            first_moe_layer=3,
        )
        layer_candidate_routes = target_layer_routes(
            candidate_routes,
            model_layer=earliest_model_layer,
            total_decoder_layers=78,
            first_moe_layer=3,
        )
        if not np.array_equal(layer_baseline_routes, layer_candidate_routes):
            raise RuntimeError(
                f"layer-{earliest_model_layer} routes changed before the "
                "frozen intervention"
            )
        document_hash = row["document_sha256"]
        baseline_by_document[document_hash] = baseline
        candidate_by_document[document_hash] = candidate
        stem = (
            f"chunk-{row['selected_chunk']:06d}"
            if "selected_chunk" in row
            else f"document-{document_hash[:16]}"
        )
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
                "document_title": row.get("document_title"),
                "source": row.get("source"),
                "axis": row.get("axis"),
                "selected_chunk": row.get("selected_chunk"),
                "context_tokens": len(row["prompt_token_ids"]),
                "reference_file": row["reference_file"],
                "reference_file_sha256": row["reference_file_sha256"],
                "resident_mean_forward_kld": float(baseline.mean().item()),
                "candidate_mean_forward_kld": float(candidate.mean().item()),
                "candidate_minus_resident_mean_forward_kld": float(
                    (candidate - baseline).mean().item()
                ),
                "resident_seconds": baseline_seconds,
                "candidate_seconds": candidate_seconds,
                "intervention_layer_route_support": (
                    route_support_summary(
                        layer_baseline_routes, selected_experts=selected_experts
                    )
                    if artifact["artifact_kind"] == "single_layer"
                    else {
                        str(model_layer): route_support_summary(
                            target_layer_routes(
                                baseline_routes,
                                model_layer=model_layer,
                                total_decoder_layers=78,
                                first_moe_layer=3,
                            ),
                            selected_experts=list(
                                artifact["expert_ids_by_layer"][model_layer]
                            ),
                        )
                        for model_layer in artifact["model_layers"]
                    }
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
        "schema": "qsrt_glm52_document_disjoint_candidate_evaluation",
        "schema_version": 1,
        "status": "complete",
        "evaluation_tier": evaluation_tier,
        "candidate": frozen,
        "intervention_artifact": {
            "root": artifact["root"],
            "manifest_sha256": artifact["manifest_sha256"],
            "artifact_kind": artifact["artifact_kind"],
            "model_layers": list(artifact["model_layers"]),
            "expert_ids_by_layer": {
                str(layer): list(experts)
                for layer, experts in artifact["expert_ids_by_layer"].items()
            },
            **(
                {
                    "model_layer": artifact["model_layer"],
                    "expert_ids": list(artifact["expert_ids"]),
                }
                if artifact["artifact_kind"] == "single_layer"
                else {}
            ),
        },
        "confirmation_registration": (
            {
                "path": str(args.confirmation_registration.resolve()),
                "sha256": sha256_file(args.confirmation_registration),
            }
            if args.confirmation_registration is not None
            else None
        ),
        "terminal_confirmation_freeze": (
            {
                "path": str(args.terminal_confirmation_freeze.resolve()),
                "sha256": sha256_file(args.terminal_confirmation_freeze),
                **terminal_confirmation_freeze,
            }
            if terminal_confirmation_freeze is not None
            else None
        ),
        "teacher_references": reference_report,
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
            "minimum_context_tokens": min(
                len(row["prompt_token_ids"]) for row in document_inputs
            ),
            "maximum_context_tokens": max(
                len(row["prompt_token_ids"]) for row in document_inputs
            ),
            "model_load_seconds": model_load_seconds,
            "engine_reference_key": reference_key,
            "engine_reference_representation": reference_representation,
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
            "comparison_status": "not_comparable_across_reference_suites",
            "reason": (
                "the 0.059 target belongs to the registered 2,048-token suite; "
                "a different document set can establish paired candidate-minus-"
                "resident change but cannot reuse that absolute threshold"
            ),
        },
        "evidence_boundary": evidence_boundary,
        "model_downloads_performed": False,
    }
    if evaluation_tier == "confirmation":
        assert terminal_confirmation_freeze is not None
        report["confirmation_decision"] = (
            evaluate_terminal_reference_confirmation_decision(
                report, terminal_confirmation_freeze
            )
        )
    atomic_write_json(args.dest / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

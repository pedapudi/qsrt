#!/usr/bin/env python3
"""Select GLM-5.2 expert subsets by paired model KLD in one model load.

This runner evaluates several frozen subsets from one prebuilt intervention
artifact.  It measures the resident once per document and changes only the
runtime expert subset between candidate arms.  Its documents become candidate-
selection data and cannot later qualify the chosen subset.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qsrt.correctness import sha256_file
from qsrt.glm52_document_disjoint_confirmation import (
    retarget_reference_symlink,
    summarize_document_paired_kld,
    token_ids_sha256,
    validate_public_reference_files,
)
from qsrt.glm52_engine_kld import (
    ENGINE_KLD_CHUNK_ROWS_ENV,
    ENGINE_KLD_REFERENCE_KEY_ENV,
    ENGINE_KLD_REFERENCE_PATH_ENV,
    ENGINE_KLD_REFERENCE_REPRESENTATION_ENV,
)
from qsrt.glm52_expert_intervention_runtime import (
    CANDIDATE_MODE,
    FORCE_PER_EXPERT_EXL3_MOE_ENV,
    IDENTITY_CONTROL_MODE,
    MATERIALIZED_LOW_RANK_CANDIDATE_MODE,
    atomic_write_control,
    validate_dense_intervention_artifact,
)
from qsrt.glm52_model_kld_candidate_selection import (
    validate_candidate_subset_selection_plan,
)
from qsrt.glm52_paired_kld import route_support_summary, target_layer_routes
from qsrt.glm52_pilot import atomic_write_json

# The confirmation runner owns the bit-checked public-reference tokenization,
# tensor validation, engine measurement, and route-difference helpers.  Import
# them so selection and confirmation cannot drift into different KLD protocols.
from run_glm52_document_disjoint_reference_confirmation import (  # noqa: E402
    _changed_route_summary,
    _measure_kld_and_routes,
    _public_reference_token_ids,
    _validate_reference_tensor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--reference-plan", type=Path, required=True)
    parser.add_argument("--reference-link", type=Path, required=True)
    parser.add_argument("--intervention-artifact", type=Path, required=True)
    parser.add_argument("--candidate-selection-plan", type=Path, required=True)
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
    reference_plan = json.loads(args.reference_plan.read_text())
    references = validate_public_reference_files(
        reference_plan, args.reference_directory
    )
    artifact = validate_dense_intervention_artifact(args.intervention_artifact)
    selection_plan = json.loads(args.candidate_selection_plan.read_text())
    candidate_arms = validate_candidate_subset_selection_plan(
        selection_plan, artifact=artifact
    )

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
        model=args.model, plan=reference_plan
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

    identity_experts = sorted(
        {expert for arm in candidate_arms for expert in arm["selected_experts"]}
    )
    first_baseline, first_routes, first_seconds = measure(
        first, mode="off", selected_experts=None
    )
    first_repeat, first_repeat_routes, _ = measure(
        first, mode="off", selected_experts=None
    )
    first_identity, first_identity_routes, _ = measure(
        first, mode=IDENTITY_CONTROL_MODE, selected_experts=identity_experts
    )
    controls_passed = bool(
        torch.equal(first_baseline, first_repeat)
        and torch.equal(first_baseline, first_identity)
        and np.array_equal(first_routes, first_repeat_routes)
        and np.array_equal(first_routes, first_identity_routes)
    )
    if not controls_passed:
        raise RuntimeError("candidate-selection measurement controls failed")

    baseline_by_document: dict[str, torch.Tensor] = {}
    candidate_by_arm: dict[str, dict[str, torch.Tensor]] = {
        arm["name"]: {} for arm in candidate_arms
    }
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
        layer_baseline_routes = target_layer_routes(
            baseline_routes,
            model_layer=artifact["model_layer"],
            total_decoder_layers=78,
            first_moe_layer=3,
        )
        document_hash = row["document_sha256"]
        baseline_by_document[document_hash] = baseline
        stem = f"chunk-{row['selected_chunk']:06d}"
        torch.save(baseline, args.dest / f"{stem}-resident-forward-kld.pt")
        arm_reports: dict[str, Any] = {}
        for arm in candidate_arms:
            name = arm["name"]
            selected_experts = arm["selected_experts"]
            candidate, candidate_routes, candidate_seconds = measure(
                row,
                mode=args.candidate_runtime_mode,
                selected_experts=selected_experts,
            )
            layer_candidate_routes = target_layer_routes(
                candidate_routes,
                model_layer=artifact["model_layer"],
                total_decoder_layers=78,
                first_moe_layer=3,
            )
            if not np.array_equal(layer_baseline_routes, layer_candidate_routes):
                raise RuntimeError(
                    f"layer-{artifact['model_layer']} routes changed before "
                    f"candidate arm {name}"
                )
            candidate_by_arm[name][document_hash] = candidate
            torch.save(candidate, args.dest / f"{stem}-{name}-forward-kld.pt")
            np.savez_compressed(
                args.dest / f"{stem}-{name}-routes.npz",
                resident=baseline_routes,
                candidate=candidate_routes,
            )
            arm_reports[name] = {
                "candidate_mean_forward_kld": float(candidate.mean().item()),
                "candidate_minus_resident_mean_forward_kld": float(
                    (candidate - baseline).mean().item()
                ),
                "candidate_seconds": candidate_seconds,
                "intervention_layer_route_support": route_support_summary(
                    layer_baseline_routes, selected_experts=selected_experts
                ),
                "all_layer_routes": _changed_route_summary(
                    baseline_routes, candidate_routes
                ),
            }
        document_reports.append(
            {
                "document_sha256": document_hash,
                "document_title": row["document_title"],
                "selected_chunk": row["selected_chunk"],
                "reference_file": row["reference_file"],
                "reference_file_sha256": row["reference_file_sha256"],
                "resident_mean_forward_kld": float(baseline.mean().item()),
                "resident_seconds": baseline_seconds,
                "candidate_arms": arm_reports,
            }
        )
        final_baseline, final_routes = baseline, baseline_routes

    assert final_baseline is not None and final_routes is not None
    final_repeat, final_repeat_routes, _ = measure(
        document_inputs[-1], mode="off", selected_experts=None
    )
    final_control_passed = bool(
        torch.equal(final_baseline, final_repeat)
        and np.array_equal(final_routes, final_repeat_routes)
    )
    if not final_control_passed:
        raise RuntimeError("final resident bracketing control failed")

    arm_summaries = {
        arm["name"]: {
            "selected_experts": arm["selected_experts"],
            "reason": arm["reason"],
            "summary": summarize_document_paired_kld(
                baseline_by_document,
                candidate_by_arm[arm["name"]],
                bootstrap_resamples=20_000,
                bootstrap_seed=0,
            ),
        }
        for arm in candidate_arms
    }
    report = {
        "schema": "qsrt_glm52_document_disjoint_model_kld_candidate_selection",
        "schema_version": 1,
        "status": "complete",
        "intervention_artifact": {
            "root": artifact["root"],
            "manifest_sha256": artifact["manifest_sha256"],
            "expert_ids": list(artifact["expert_ids"]),
            "model_layer": artifact["model_layer"],
        },
        "candidate_selection_plan": {
            "path": str(args.candidate_selection_plan.resolve()),
            "sha256": sha256_file(args.candidate_selection_plan),
            "frozen_at_utc": selection_plan["frozen_at_utc"],
        },
        "public_reference_plan": {
            "path": str(args.reference_plan.resolve()),
            "sha256": sha256_file(args.reference_plan),
            "repository": reference_plan["reference_repository"],
            "revision": reference_plan["reference_revision"],
            "teacher_model": reference_plan["teacher_model"],
            "teacher_revision": reference_plan["teacher_revision"],
            "total_reference_bytes": references["total_reference_bytes"],
        },
        "tokenization": tokenization_receipt,
        "measurement_controls": {
            "initial_resident_repeat_kld_bitwise_equal": bool(
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
            "passed": bool(controls_passed and final_control_passed),
        },
        "candidate_arms": arm_summaries,
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
        "evidence_boundary": (
            "These 16 public 512-token documents have already been opened for "
            "mechanism screens. This report uses them to select among frozen "
            "expert subsets and supplies no confirmation evidence. Any chosen "
            "subset requires a new document-disjoint reference tier generated "
            "from the same immutable BF16 teacher revision."
        ),
        "model_downloads_performed": False,
    }
    atomic_write_json(args.dest / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

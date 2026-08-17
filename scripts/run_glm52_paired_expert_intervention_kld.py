#!/usr/bin/env python3
"""Measure paired BF16-reference KLD with one resident EXL3 model load.

Every run begins with the resident checkpoint, an unchanged repeat, and an
intervention hook that returns the resident output directly.  Candidate arms
run only after both controls reproduce bitwise KLD and complete route arrays.
The remaining arms substitute individual or complete dense QSRT-K3 expert
endpoints through the same runtime hook.
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

from qsrt.glm52_engine_kld import (
    ENGINE_KLD_CHUNK_ROWS_ENV,
    ENGINE_KLD_REFERENCE_KEY_ENV,
    ENGINE_KLD_REFERENCE_PATH_ENV,
    engine_kld_from_prompt_logprobs,
)
from qsrt.glm52_expert_intervention_runtime import (
    FORCE_PER_EXPERT_EXL3_MOE_ENV,
    atomic_write_control,
    validate_dense_intervention_artifact,
)
from qsrt.glm52_paired_kld import (
    forward_kld_per_position,
    paired_kld_summary,
    route_support_summary,
    target_layer_routes,
)
from qsrt.glm52_pilot import atomic_write_json
from qsrt.correctness import sha256_file


PAIRED_KLD_EVIDENCE_BOUNDARY = (
    "paired BF16-reference forward KLD on one published 2,048-token WikiText "
    "context; the candidate was chosen without this context, but the context "
    "does not provide document-level replication; the intervention adds a dense "
    "candidate-minus-dense-EXL3 endpoint correction to the live EXL3 layer, not "
    "a serialized QSRT decoder output; dense EXL3 endpoint closure to the fused "
    "runtime kernel has not been established separately"
)


def _dense_from_flat_prompt_logprobs(
    prompt_logprobs: Any, *, positions: int, vocabulary_size: int
) -> torch.Tensor:
    dense = torch.empty((positions, vocabulary_size), dtype=torch.float32)
    if hasattr(prompt_logprobs, "start_indices"):
        for position in range(positions):
            source_position = position + 1
            start = prompt_logprobs.start_indices[source_position]
            stop = prompt_logprobs.end_indices[source_position]
            token_ids = torch.tensor(
                prompt_logprobs.token_ids[start:stop], dtype=torch.long
            )
            values = torch.tensor(
                prompt_logprobs.logprobs[start:stop], dtype=torch.float32
            )
            row = torch.full((vocabulary_size,), float("-inf"), dtype=torch.float32)
            valid = (token_ids >= 0) & (token_ids < vocabulary_size)
            row[token_ids[valid]] = values[valid]
            dense[position] = row
        return dense
    for position in range(positions):
        row = torch.full((vocabulary_size,), float("-inf"), dtype=torch.float32)
        for token_id, logprob in prompt_logprobs[position + 1].items():
            index = int(token_id)
            if 0 <= index < vocabulary_size:
                row[index] = float(logprob.logprob)
        dense[position] = row
    return dense


def _reference_tokens(
    *,
    model: Path,
    manifest: dict[str, Any],
    context_length: int,
) -> list[int]:
    dependency_path = os.getenv("KLD_PYDEPS")
    if dependency_path:
        sys.path.append(dependency_path)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="test"
    )
    texts = [
        row["text"]
        for row in dataset
        if row.get("text") and str(row["text"]).strip()
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        model, trust_remote_code=True, local_files_only=True
    )
    encoded = tokenizer(
        "\n\n".join(texts)[: context_length * 5],
        add_special_tokens=False,
        truncation=True,
        max_length=context_length,
    )
    token_ids = encoded["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    token_ids = [int(value) for value in token_ids]
    if len(token_ids) != context_length:
        raise RuntimeError(
            f"reference context has {len(token_ids)} tokens, expected {context_length}"
        )
    if token_ids[:16] != manifest["token_first16"]:
        raise RuntimeError("reference token fingerprint mismatch")
    return token_ids


def _model_logits_and_routes(
    llm: Any,
    *,
    token_ids: list[int],
    expected_shape: list[int],
) -> tuple[torch.Tensor, np.ndarray]:
    from vllm import SamplingParams

    engine_kld_enabled = bool(os.environ.get(ENGINE_KLD_REFERENCE_PATH_ENV))
    supports_prompt_logits = "return_prompt_logits" in inspect.signature(
        SamplingParams
    ).parameters
    if engine_kld_enabled:
        parameters = SamplingParams(
            prompt_logprobs=1,
            flat_logprobs=True,
            max_tokens=1,
            routed_experts_prompt_start=0,
            detokenize=False,
        )
    elif supports_prompt_logits:
        parameters = SamplingParams(
            prompt_logprobs=1,
            max_tokens=1,
            return_prompt_logits=True,
            routed_experts_prompt_start=0,
            detokenize=False,
        )
    else:
        parameters = SamplingParams(
            prompt_logprobs=-1,
            flat_logprobs=True,
            max_tokens=1,
            routed_experts_prompt_start=0,
            detokenize=False,
        )
    output = llm.generate(
        [
            {
                "prompt_token_ids": token_ids,
                "target_token_ids": token_ids[1:],
            }
        ],
        sampling_params=parameters,
    )[0]
    if engine_kld_enabled:
        if output.prompt_logprobs is None:
            raise RuntimeError("vLLM returned no engine KLD values")
        logits = engine_kld_from_prompt_logprobs(
            output.prompt_logprobs,
            positions=expected_shape[0],
            target_token_ids=token_ids[1:],
        )
    elif supports_prompt_logits:
        raw = output.prompt_logits
        if raw is None:
            raise RuntimeError("vLLM returned no prompt logits")
        logits = raw.detach().to("cpu", copy=True)
    else:
        if output.prompt_logprobs is None:
            raise RuntimeError("vLLM returned no prompt log probabilities")
        logits = _dense_from_flat_prompt_logprobs(
            output.prompt_logprobs,
            positions=expected_shape[0],
            vocabulary_size=expected_shape[1],
        )
    actual_shape = list(logits.shape)
    required_shape = [expected_shape[0]] if engine_kld_enabled else expected_shape
    if actual_shape != required_shape:
        raise RuntimeError(
            f"model result has shape {actual_shape}, expected {required_shape}"
        )
    routes = output.outputs[0].routed_experts
    if routes is None:
        raise RuntimeError("vLLM returned no routed-expert capture")
    return logits, np.asarray(routes)


def _capture_planned_layer_inputs(
    llm: Any,
    *,
    corpus_plan_path: Path,
    capture_dir: Path,
    control_path: Path,
    artifact_manifest_sha256: str,
    selected_experts: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    """Run fit and selection prompts while rank zero stores layer-3 inputs."""

    from safetensors import safe_open
    from vllm import SamplingParams

    plan_sha256 = sha256_file(corpus_plan_path)
    configured_dir = os.getenv("QSRT_GLM52_ACTIVATION_CAPTURE_DIR")
    configured_plan = os.getenv("QSRT_GLM52_ACTIVATION_CAPTURE_PLAN_SHA256")
    if configured_dir != str(capture_dir) or configured_plan != plan_sha256:
        raise RuntimeError(
            "activation-capture environment does not match the requested directory "
            "and corpus-plan SHA-256"
        )
    if capture_dir.exists():
        raise FileExistsError(capture_dir)
    plan = json.loads(corpus_plan_path.read_text())
    if (
        plan.get("schema") != "qsrt_glm52_document_disjoint_corpus_plan"
        or plan.get("schema_version") != 1
        or plan.get("separation")
        != {
            "fit_selection_row_overlap": 0,
            "reference_fit_row_overlap": 0,
            "reference_selection_row_overlap": 0,
            "unit": "WikiText article delimited by a top-level heading",
        }
    ):
        raise ValueError("activation corpus plan identity or separation is invalid")

    parameters = SamplingParams(
        max_tokens=1,
        routed_experts_prompt_start=0,
        detokenize=False,
    )
    records: list[dict[str, Any]] = []
    generation = 100
    for collection in ("activation_fit", "candidate_selection"):
        raw_collection = plan.get(collection)
        if not isinstance(raw_collection, dict):
            raise ValueError(f"corpus plan has no {collection!r} collection")
        windows = raw_collection.get("windows")
        if (
            not isinstance(windows, list)
            or raw_collection.get("window_count") != len(windows)
        ):
            raise ValueError(f"corpus plan {collection!r} window count mismatch")
        for window in windows:
            generation += 1
            token_ids = [int(value) for value in window["token_ids"]]
            if len(token_ids) != int(window["token_count"]) or len(token_ids) < 128:
                raise ValueError(f"corpus window {window['window_id']} is malformed")
            atomic_write_control(
                control_path,
                mode="off",
                artifact_manifest_sha256=artifact_manifest_sha256,
                generation=generation,
                capture_enabled=True,
            )
            output = llm.generate(
                [{"prompt_token_ids": token_ids}], sampling_params=parameters
            )[0]
            routed_experts = output.outputs[0].routed_experts
            if routed_experts is None:
                raise RuntimeError("vLLM returned no routed experts during capture")
            layer_routes = target_layer_routes(
                np.asarray(routed_experts),
                model_layer=3,
                total_decoder_layers=78,
                first_moe_layer=3,
            )
            matches: list[Path] = []
            for path in capture_dir.glob("layer-003-input-chunk-*.safetensors"):
                with safe_open(path, framework="pt", device="cpu") as handle:
                    metadata = handle.metadata() or {}
                    if metadata.get("control_generation") == str(generation):
                        hidden_shape = tuple(handle.get_slice("hidden_states").get_shape())
                        if hidden_shape != (len(token_ids), 6144):
                            raise ValueError(
                                f"capture {path.name} has hidden shape {hidden_shape}"
                            )
                        matches.append(path)
            if len(matches) != 1:
                raise RuntimeError(
                    f"control generation {generation} produced {len(matches)} captures"
                )
            records.append(
                {
                    "collection": collection,
                    "window_id": window["window_id"],
                    "token_count": len(token_ids),
                    "control_generation": generation,
                    "capture_file": matches[0].name,
                    "capture_file_bytes": matches[0].stat().st_size,
                    "capture_file_sha256": sha256_file(matches[0]),
                    "route_support": route_support_summary(
                        layer_routes, selected_experts=selected_experts
                    ),
                }
            )
    atomic_write_control(
        control_path,
        mode="off",
        artifact_manifest_sha256=artifact_manifest_sha256,
        generation=generation + 1,
        capture_enabled=False,
    )
    capture_manifest = {
        "schema": "qsrt_glm52_layer_input_capture_manifest",
        "schema_version": 1,
        "status": "complete",
        "model_layer": 3,
        "corpus_plan_path": str(corpus_plan_path),
        "corpus_plan_sha256": plan_sha256,
        "intervention_artifact_manifest_sha256": artifact_manifest_sha256,
        "records": records,
        "collections": {
            collection: sum(record["collection"] == collection for record in records)
            for collection in ("activation_fit", "candidate_selection")
        },
        "evidence_boundary": (
            "exact resident-EXL3 layer-3 inputs, route IDs, and applied route "
            "weights from document-disjoint fit and candidate-selection articles"
        ),
    }
    atomic_write_json(capture_dir / "manifest.json", capture_manifest)
    return capture_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-logits", type=Path, required=True)
    parser.add_argument("--intervention-artifact", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--corpus-plan", type=Path)
    parser.add_argument("--activation-capture-dir", type=Path)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument(
        "--source-sparse-index-topk",
        type=int,
        help=(
            "source model sparse-attention key count; when dense attention is "
            "used, the scored context may not exceed this count"
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--kv-cache-dtype", default="nvfp4_ds_mla")
    parser.add_argument("--load-format", default="safetensors")
    parser.add_argument("--quantization", default="exl3")
    parser.add_argument("--attention-backend", default="B12X_MLA_SPARSE")
    parser.add_argument("--max-model-len", type=int, default=2049)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--kld-chunk-rows", type=int, default=16)
    parser.add_argument(
        "--kld-device",
        default="cpu",
        help="CPU or CUDA device used for bounded full-vocabulary KLD chunks",
    )
    parser.add_argument("--hf-overrides", default="{}")
    parser.add_argument("--llm-extra-json", default="{}")
    parser.add_argument(
        "--omit-individual-expert-arms",
        action="store_true",
        help=(
            "measure the repeatability and wiring controls plus the complete "
            "selected panel"
        ),
    )
    parser.add_argument(
        "--measurement-controls-only",
        action="store_true",
        help=(
            "measure only repeated resident inference and the direct-return "
            "identity hook; do not evaluate a candidate"
        ),
    )
    return parser


def intervention_arm_definitions(
    expert_ids: list[int] | tuple[int, ...],
    *,
    omit_individual_expert_arms: bool,
    measurement_controls_only: bool = False,
) -> list[tuple[str, str, list[int] | None]]:
    """Describe repeatability, wiring, individual, and complete-panel arms."""

    experts = list(expert_ids)
    if (
        not experts
        or len(set(experts)) != len(experts)
        or any(
            isinstance(expert, bool)
            or not isinstance(expert, int)
            or not 0 <= expert < 256
            for expert in experts
        )
    ):
        raise ValueError("intervention expert IDs must be unique values from 0 to 255")
    definitions: list[tuple[str, str, list[int] | None]] = [
        ("resident_exl3", "off", None),
        ("resident_exl3_repeat", "off", None),
        ("dense_resident_identity", "dense_resident_identity", None),
    ]
    if measurement_controls_only:
        return definitions
    if not omit_individual_expert_arms:
        definitions.extend(
            (f"selected_qsrt_k3_expert_{expert:03d}", "qsrt_k3", [expert])
            for expert in experts
        )
    definitions.append(("selected_qsrt_k3", "qsrt_k3", None))
    return definitions


def measurement_control_summary(
    klds: dict[str, torch.Tensor],
    routes: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compare an unchanged repeat and a direct-return hook with the baseline."""

    baseline_kld = klds["resident_exl3"]
    repeat_kld = klds["resident_exl3_repeat"]
    identity_kld = klds["dense_resident_identity"]
    baseline_routes = routes["resident_exl3"]
    repeat_routes = routes["resident_exl3_repeat"]
    identity_routes = routes["dense_resident_identity"]

    repeat_kld_equal = bool(torch.equal(baseline_kld, repeat_kld))
    repeat_routes_equal = bool(np.array_equal(baseline_routes, repeat_routes))
    identity_kld_equal = bool(torch.equal(baseline_kld, identity_kld))
    identity_routes_equal = bool(np.array_equal(baseline_routes, identity_routes))
    return {
        "passed": bool(
            repeat_kld_equal
            and repeat_routes_equal
            and identity_kld_equal
            and identity_routes_equal
        ),
        "resident_repeatability_control": {
            "paired": paired_kld_summary(baseline_kld, repeat_kld),
            "forward_kld_bitwise_equal": repeat_kld_equal,
            "maximum_absolute_forward_kld_difference": float(
                (repeat_kld - baseline_kld).abs().max().item()
            ),
            "all_layer_route_array_equal": repeat_routes_equal,
            "expected_result": (
                "bitwise-equal forward KLD and route arrays because both arms "
                "run the unchanged resident model in the same process"
            ),
        },
        "dense_resident_identity_control": {
            "paired": paired_kld_summary(baseline_kld, identity_kld),
            "forward_kld_bitwise_equal": identity_kld_equal,
            "maximum_absolute_forward_kld_difference": float(
                (identity_kld - baseline_kld).abs().max().item()
            ),
            "all_layer_route_array_equal": identity_routes_equal,
            "expected_result": (
                "bitwise-equal forward KLD and route arrays because the hook "
                "loads and validates the dense endpoint artifact but returns "
                "the resident layer output without arithmetic"
            ),
            "limitation": (
                "this direct-return control proves intervention wiring and "
                "artifact loading; it does not compare the decoded dense EXL3 "
                "endpoint with the fused runtime kernel"
            ),
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.source_sparse_index_topk is not None:
        if args.source_sparse_index_topk < 1:
            raise ValueError("source sparse-index top-k must be positive")
        if args.context_length > args.source_sparse_index_topk:
            raise ValueError(
                "dense-attention scoring would admit more keys than the source "
                "sparse-attention contract"
            )
    args.dest.mkdir(parents=True, exist_ok=False)
    artifact = validate_dense_intervention_artifact(args.intervention_artifact)
    manifest_path = args.reference_logits / "manifest.json"
    reference_manifest = json.loads(manifest_path.read_text())
    if args.context_length != int(reference_manifest["context_length"]):
        raise ValueError("context length does not match the reference manifest")
    if len(reference_manifest["windows"]) != 1:
        raise ValueError("this bounded runner requires exactly one reference window")
    expected_shape = [int(value) for value in reference_manifest["windows"][0]["shape"]]
    token_ids = _reference_tokens(
        model=args.model,
        manifest=reference_manifest,
        context_length=args.context_length,
    )
    reference_path = args.reference_logits / "logits_0.safetensors"
    from safetensors import safe_open

    engine_kld_enabled = bool(os.environ.get(ENGINE_KLD_REFERENCE_PATH_ENV))
    if engine_kld_enabled:
        configured_reference = Path(
            os.environ[ENGINE_KLD_REFERENCE_PATH_ENV]
        ).resolve(strict=True)
        if configured_reference != reference_path.resolve(strict=True):
            raise ValueError("engine KLD reference path differs from the runner input")
        if os.environ.get(ENGINE_KLD_REFERENCE_KEY_ENV) != "logits":
            raise ValueError("engine KLD reference tensor key must be 'logits'")
        if os.environ.get(ENGINE_KLD_CHUNK_ROWS_ENV) != str(args.kld_chunk_rows):
            raise ValueError("engine and runner KLD chunk-row settings differ")
    with safe_open(reference_path, framework="pt", device="cpu") as handle:
        reference_shape = list(handle.get_slice("logits").get_shape())
        reference_logits = (
            None if engine_kld_enabled else handle.get_tensor("logits")
        )
    if reference_shape != expected_shape:
        raise ValueError("reference logits shape does not match its manifest")

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
    load_seconds = time.monotonic() - started

    activation_capture = None
    if bool(args.corpus_plan) != bool(args.activation_capture_dir):
        raise ValueError(
            "--corpus-plan and --activation-capture-dir must be supplied together"
        )
    if args.corpus_plan is not None and args.activation_capture_dir is not None:
        activation_capture = _capture_planned_layer_inputs(
            llm,
            corpus_plan_path=args.corpus_plan,
            capture_dir=args.activation_capture_dir,
            control_path=args.control,
            artifact_manifest_sha256=artifact["manifest_sha256"],
            selected_experts=artifact["expert_ids"],
        )

    arm_results: dict[str, Any] = {}
    klds: dict[str, torch.Tensor] = {}
    routes: dict[str, np.ndarray] = {}
    arm_definitions = intervention_arm_definitions(
        artifact["expert_ids"],
        omit_individual_expert_arms=args.omit_individual_expert_arms,
        measurement_controls_only=args.measurement_controls_only,
    )
    for generation, (arm, mode, selected_experts) in enumerate(
        arm_definitions, start=1
    ):
        atomic_write_control(
            args.control,
            mode=mode,
            artifact_manifest_sha256=artifact["manifest_sha256"],
            generation=generation,
            selected_experts=selected_experts,
        )
        arm_started = time.monotonic()
        logits, routed_experts = _model_logits_and_routes(
            llm, token_ids=token_ids, expected_shape=expected_shape
        )
        arm_seconds = time.monotonic() - arm_started
        if engine_kld_enabled:
            klds[arm] = logits.double()
        else:
            assert reference_logits is not None
            klds[arm] = forward_kld_per_position(
                reference_logits,
                logits,
                chunk_rows=args.kld_chunk_rows,
                compute_device=args.kld_device,
            )
        routes[arm] = routed_experts
        np.savez_compressed(
            args.dest / f"{arm}-routed-experts.npz", routed_experts=routed_experts
        )
        torch.save(klds[arm], args.dest / f"{arm}-forward-kld-per-position.pt")
        arm_results[arm] = {
            "mode": mode,
            "selected_experts": selected_experts,
            "elapsed_seconds": arm_seconds,
            "mean_forward_kld": float(klds[arm].mean().item()),
            "routed_experts_shape": list(routed_experts.shape),
        }
        del logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if arm == "dense_resident_identity" and not args.measurement_controls_only:
            controls = measurement_control_summary(klds, routes)
            atomic_write_json(
                args.dest / "measurement-controls.json",
                {
                    "schema": "qsrt_glm52_paired_kld_measurement_controls",
                    "schema_version": 1,
                    **controls,
                },
            )
            if not controls["passed"]:
                raise RuntimeError(
                    "paired KLD measurement controls failed; candidate arms were "
                    "not evaluated"
                )

    baseline_layer_routes = target_layer_routes(
        routes["resident_exl3"],
        model_layer=3,
        total_decoder_layers=78,
        first_moe_layer=3,
    )
    for arm, _, _ in arm_definitions[1:]:
        arm_layer_routes = target_layer_routes(
            routes[arm],
            model_layer=3,
            total_decoder_layers=78,
            first_moe_layer=3,
        )
        if not np.array_equal(baseline_layer_routes, arm_layer_routes):
            raise RuntimeError(
                f"layer-3 routing changed before the {arm!r} intervention"
            )
    controls = measurement_control_summary(klds, routes)
    candidate_measured = "selected_qsrt_k3" in klds
    report = {
        "schema": "qsrt_glm52_paired_expert_intervention_kld",
        "schema_version": 2,
        "status": "complete",
        "model": str(args.model.resolve()),
        "reference_logits": str(args.reference_logits.resolve()),
        "reference_manifest": reference_manifest,
        "intervention_artifact": {
            key: artifact[key]
            for key in (
                "root",
                "manifest_sha256",
                "expert_ids",
                "expert_count",
                "dense_endpoint_bytes",
            )
        },
        "runtime": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "kv_cache_dtype": args.kv_cache_dtype,
            "load_format": args.load_format,
            "attention_backend": args.attention_backend,
            "kld_device": args.kld_device,
            "kld_chunk_rows": args.kld_chunk_rows,
            "engine_kld_enabled": engine_kld_enabled,
            "model_load_seconds": load_seconds,
            "exl3_moe_execution": (
                "three_gemm_per_expert_correctness"
                if os.environ.get(FORCE_PER_EXPERT_EXL3_MOE_ENV, "0") == "1"
                else "checkpoint_selected_fused_kernel"
            ),
            "llm_kwargs": llm_kwargs,
        },
        "attention_contract": {
            "backend": args.attention_backend,
            "scored_context_tokens": args.context_length,
            "source_sparse_index_topk": args.source_sparse_index_topk,
            "complete_causal_key_set_preserved": bool(
                args.source_sparse_index_topk is not None
                and args.context_length <= args.source_sparse_index_topk
            ),
            "evidence_boundary": (
                "dense attention removes sparse-index ordering and reuse; it "
                "admits the same complete causal key set only because the "
                "scored context does not exceed the source model's sparse "
                "top-k count"
                if args.source_sparse_index_topk is not None
                else "the runner did not receive the source sparse-index count"
            ),
        },
        "arms": arm_results,
        "activation_capture": (
            {
                "manifest_path": str(args.activation_capture_dir / "manifest.json"),
                "corpus_plan_sha256": activation_capture["corpus_plan_sha256"],
                "collections": activation_capture["collections"],
            }
            if activation_capture is not None
            else None
        ),
        "paired": (
            paired_kld_summary(
                klds["resident_exl3"], klds["selected_qsrt_k3"]
            )
            if candidate_measured
            else None
        ),
        "measurement_controls_passed": controls["passed"],
        "resident_repeatability_control": controls[
            "resident_repeatability_control"
        ],
        "dense_resident_identity_control": controls[
            "dense_resident_identity_control"
        ],
        "individual_expert_paired": {
            str(expert): {
                "arm": f"selected_qsrt_k3_expert_{expert:03d}",
                "paired": paired_kld_summary(
                    klds["resident_exl3"],
                    klds[f"selected_qsrt_k3_expert_{expert:03d}"],
                ),
                "layer_3_route_support": route_support_summary(
                    baseline_layer_routes, selected_experts=[expert]
                ),
            }
            for expert in artifact["expert_ids"]
        }
        if candidate_measured and not args.omit_individual_expert_arms
        else None,
        "layer_3_route_support": route_support_summary(
            baseline_layer_routes, selected_experts=artifact["expert_ids"]
        ),
        "all_layer_route_array_equal": {
            arm: bool(np.array_equal(routes["resident_exl3"], routes[arm]))
            for arm, _, _ in arm_definitions[1:]
        },
        "evidence_boundary": PAIRED_KLD_EVIDENCE_BOUNDARY,
        "model_downloads_performed": False,
    }
    atomic_write_json(args.dest / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

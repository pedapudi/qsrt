#!/usr/bin/env python3
"""Build a layer-92 routed-output Fisher factor from captured suffix states.

The factor uses the official reference distribution at the LM head and pulls
paired softmax-Fisher samples through the exact Kimi-K3 layer-92 common suffix.
Captured states must come exclusively from the fidelity suite's analysis
partition.  The resulting matrix weights output directions for canonical W2
quantization; it does not refit or otherwise replace the W2 target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from qsrt.suffix_fisher import (
    kimi_layer92_routed_fisher_pullback,
    paired_lm_head_fisher_gradients,
)


TENSOR_NAMES = {
    "final_norm_weight": "language_model.model.norm.weight",
    "attention_norm_weight": "language_model.model.output_attn_res_norm.weight",
    "attention_projection_weight": (
        "language_model.model.output_attn_res_proj.weight"
    ),
    "routed_norm_weight": (
        "language_model.model.layers.92.block_sparse_moe."
        "routed_expert_norm.weight"
    ),
    "routed_projection_weight": (
        "language_model.model.layers.92.block_sparse_moe."
        "routed_expert_up_proj.weight"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_checkpoint_tensor(
    checkpoint: Path,
    index: Mapping[str, Any],
    name: str,
) -> torch.Tensor:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or name not in weight_map:
        raise KeyError(f"checkpoint does not contain {name}")
    shard = checkpoint / str(weight_map[name])
    with safe_open(shard, framework="pt", device="cpu") as reader:
        return reader.get_tensor(name)


def _text_config(checkpoint: Path) -> Mapping[str, Any]:
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    nested = config.get("text_config", config)
    if not isinstance(nested, dict):
        raise TypeError("checkpoint text configuration must be an object")
    return nested


def _split(context_index: int) -> int:
    digest = hashlib.sha256(
        f"kimi-k3-layer92-suffix-fisher-v1:{context_index}".encode()
    ).digest()
    return digest[0] & 1


def _factor_from_sum(
    outer_sum: torch.Tensor,
    count: int,
) -> torch.Tensor:
    if count <= 0:
        raise ValueError("Fisher split has no samples")
    factor = outer_sum / float(count)
    factor = (factor + factor.T) * 0.5
    return factor.contiguous()


def _damped(factor: torch.Tensor, ratio: float) -> torch.Tensor:
    result = factor.clone()
    result.diagonal().add_(ratio * torch.diagonal(result).mean())
    return result


def _factor_summary(factor: torch.Tensor) -> dict[str, float]:
    diagonal = torch.diagonal(factor)
    return {
        "diagonal_max": float(diagonal.max()),
        "diagonal_mean": float(diagonal.mean()),
        "diagonal_min": float(diagonal.min()),
        "frobenius_norm": float(torch.linalg.vector_norm(factor)),
        "trace": float(diagonal.sum()),
    }


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    numerator = torch.sum(first * second)
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    return float(numerator / denominator)


def _validate_reference_hidden(
    path: Path,
    *,
    context_index: int,
    token_hash: str,
    hidden_dimension: int,
) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as reader:
        if list(reader.keys()) != ["hidden_states"]:
            raise RuntimeError(f"unexpected reference-hidden keys in {path}")
        metadata = reader.metadata() or {}
        if metadata.get("context_index") != str(context_index):
            raise RuntimeError(f"reference-hidden context mismatch in {path}")
        if metadata.get("token_ids_json_sha256") != token_hash:
            raise RuntimeError(f"reference-hidden token mismatch in {path}")
        hidden = reader.get_tensor("hidden_states")
    if hidden.dtype != torch.bfloat16 or hidden.shape[1] != hidden_dimension:
        raise RuntimeError(f"invalid reference-hidden tensor in {path}")
    return hidden


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-contexts", type=int, default=8)
    parser.add_argument("--pairs-per-row", type=int, default=4)
    parser.add_argument("--validation-damping-ratio", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("output and receipt paths must be fresh")
    if args.batch_contexts <= 0 or args.pairs_per_row <= 0:
        parser.error("batch contexts and pairs per row must be positive")
    if (
        not math.isfinite(args.validation_damping_ratio)
        or args.validation_damping_ratio <= 0.0
    ):
        parser.error("validation damping ratio must be finite and positive")

    capture_manifest_path = args.capture_dir / "manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    suite_manifest_path = args.suite_dir / "suite-manifest.json"
    suite = json.loads(suite_manifest_path.read_text(encoding="utf-8"))
    if capture_manifest.get("partition") != "analysis":
        raise RuntimeError("suffix Fisher capture must contain only analysis contexts")
    if capture_manifest.get("complete") is not True:
        raise RuntimeError("suffix Fisher capture is not sealed as complete")
    if capture_manifest.get("suite_token_hash_sha256") != suite.get(
        "suite_token_hash_sha256"
    ):
        raise RuntimeError("suffix capture and fidelity suite disagree")
    suite_contexts = {int(item["context_index"]): item for item in suite["contexts"]}
    expected_indices = sorted(
        index
        for index, item in suite_contexts.items()
        if item.get("partition") == "analysis"
    )
    expected_indices_sha256 = hashlib.sha256(
        json.dumps(expected_indices, separators=(",", ":")).encode()
    ).hexdigest()
    if capture_manifest.get("expected_context_count") != len(expected_indices):
        raise RuntimeError("suffix capture has the wrong expected context count")
    if (
        capture_manifest.get("expected_context_indices_sha256")
        != expected_indices_sha256
    ):
        raise RuntimeError("suffix capture has the wrong expected context identity")
    records = capture_manifest.get("contexts")
    if not isinstance(records, list) or not records:
        raise RuntimeError("suffix capture manifest contains no contexts")
    observed_indices = sorted(int(record["context_index"]) for record in records)
    if observed_indices != expected_indices:
        raise RuntimeError("suffix capture does not contain every analysis context exactly once")
    for record in records:
        index = int(record["context_index"])
        if suite_contexts[index].get("partition") != "analysis":
            raise RuntimeError(f"qualification context {index} entered factor capture")

    index_path = args.checkpoint / "model.safetensors.index.json"
    checkpoint_index = json.loads(index_path.read_text(encoding="utf-8"))
    config = _text_config(args.checkpoint)
    hidden_dimension = int(config["hidden_size"])
    latent_dimension = 3584
    epsilon = float(config["rms_norm_eps"])
    logit_scale = float(config.get("logit_scale", 1.0))
    static = {
        role: _load_checkpoint_tensor(args.checkpoint, checkpoint_index, name)
        for role, name in TENSOR_NAMES.items()
    }
    expected_shapes = {
        "final_norm_weight": (hidden_dimension,),
        "attention_norm_weight": (hidden_dimension,),
        "attention_projection_weight": (1, hidden_dimension),
        "routed_norm_weight": (latent_dimension,),
        "routed_projection_weight": (hidden_dimension, latent_dimension),
    }
    for role, shape in expected_shapes.items():
        if tuple(static[role].shape) != shape:
            observed = tuple(static[role].shape)
            raise RuntimeError(f"unexpected {role} shape: {observed}")

    lm_head_path = args.suite_dir / "lm-head" / "weight.safetensors"
    lm_head = load_file(lm_head_path, device="cpu")["weight"]
    if lm_head.dtype != torch.bfloat16 or lm_head.shape[1] != hidden_dimension:
        raise RuntimeError("fidelity LM head has an incompatible tensor")

    device = torch.device(args.device)
    lm_head = lm_head.to(device=device)
    static = {key: value.float().to(device=device) for key, value in static.items()}
    attention_query = (
        static["attention_norm_weight"]
        * static["attention_projection_weight"].squeeze(0)
    )
    sums = [
        torch.zeros(
            (latent_dimension, latent_dimension),
            dtype=torch.float32,
            device=device,
        )
        for _ in range(3)
    ]
    counts = [0, 0, 0]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    processed: list[dict[str, int]] = []

    for begin in range(0, len(records), args.batch_contexts):
        batch_gradients: list[torch.Tensor] = []
        batch_states: dict[str, list[torch.Tensor]] = {
            key: []
            for key in (
                "routed_latent",
                "updated_prefix",
                "final_mixed",
                "prefix_weight",
            )
        }
        batch_splits: list[torch.Tensor] = []
        batch_records = records[begin : begin + args.batch_contexts]
        for record in batch_records:
            index = int(record["context_index"])
            context = suite_contexts[index]
            token_hash = str(context["token_ids_json_sha256"])
            capture_path = args.capture_dir / str(record["file"])
            if _sha256(capture_path) != record["sha256"]:
                raise RuntimeError(f"suffix capture hash mismatch for context {index}")
            captured = load_file(capture_path, device="cpu")
            rows = captured["row_index"]
            reference = _validate_reference_hidden(
                args.suite_dir
                / "reference-hidden"
                / f"hidden_{index:04d}.safetensors",
                context_index=index,
                token_hash=token_hash,
                hidden_dimension=hidden_dimension,
            ).index_select(0, rows)
            logits = F.linear(reference.to(device=device), lm_head).float()
            logits.mul_(logit_scale)
            probabilities = torch.softmax(logits, dim=-1)
            sampled = torch.multinomial(
                probabilities,
                2 * args.pairs_per_row,
                replacement=True,
                generator=generator,
            )
            del logits, probabilities
            first, second = sampled.chunk(2, dim=-1)
            hidden_gradient = paired_lm_head_fisher_gradients(
                lm_head,
                first,
                second,
                logit_scale=logit_scale,
            ).reshape(-1, hidden_dimension)
            repeat = args.pairs_per_row
            split = _split(index)
            sample_count = int(hidden_gradient.shape[0])
            batch_gradients.append(hidden_gradient)
            batch_splits.append(
                torch.full(
                    (sample_count,), split, dtype=torch.int64, device=device
                )
            )
            for key in batch_states:
                batch_states[key].append(
                    captured[key]
                    .to(device=device)
                    .repeat_interleave(repeat, dim=0)
                )
            processed.append(
                {
                    "context_index": index,
                    "rows": int(rows.numel()),
                    "fisher_samples": sample_count,
                    "split": split,
                }
            )
        hidden_gradient = torch.cat(batch_gradients, dim=0)
        captured_device = {
            key: torch.cat(value, dim=0) for key, value in batch_states.items()
        }
        routed_gradient = kimi_layer92_routed_fisher_pullback(
            captured_device["routed_latent"],
            captured_device["updated_prefix"],
            captured_device["final_mixed"],
            captured_device["prefix_weight"],
            hidden_gradient,
            final_norm_weight=static["final_norm_weight"],
            final_norm_epsilon=epsilon,
            attention_score_query=attention_query,
            attention_epsilon=epsilon,
            routed_norm_weight=static["routed_norm_weight"],
            routed_norm_epsilon=epsilon,
            routed_projection_weight=static["routed_projection_weight"],
        )
        split_indices = torch.cat(batch_splits, dim=0)
        sums[0].add_(routed_gradient.T @ routed_gradient)
        counts[0] += int(routed_gradient.shape[0])
        for split in (0, 1):
            selected = routed_gradient[split_indices == split]
            if selected.numel():
                sums[split + 1].add_(selected.T @ selected)
                counts[split + 1] += int(selected.shape[0])
        first_index = int(batch_records[0]["context_index"])
        last_index = int(batch_records[-1]["context_index"])
        print(
            f"contexts {first_index:04d}-{last_index:04d}: "
            f"{routed_gradient.shape[0]} Fisher samples; total {counts[0]}",
            flush=True,
        )

    factors = [
        _factor_from_sum(total, count)
        for total, count in zip(sums, counts, strict=True)
    ]
    cholesky_info = []
    for factor in factors:
        _, info = torch.linalg.cholesky_ex(
            _damped(factor, args.validation_damping_ratio)
        )
        cholesky_info.append(int(info.max()))
    if any(cholesky_info):
        raise RuntimeError(
            f"damped Fisher factor is not positive definite: {cholesky_info}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    save_file(
        {
            "output_hessian": factors[0].cpu(),
            "output_hessian_split_a": factors[1].cpu(),
            "output_hessian_split_b": factors[2].cpu(),
        },
        str(temporary),
        metadata={
            "kind": "Kimi-K3 layer-92 routed-output empirical Fisher",
            "semantic_point": "aggregated_routed_w2_output_before_routed_rmsnorm",
            "partition": "analysis",
            "damping": "none",
            "validation_damping_ratio": str(args.validation_damping_ratio),
            "samples": str(counts[0]),
        },
    )
    os.replace(temporary, args.output)
    receipt = {
        "capture_manifest": str(capture_manifest_path.resolve()),
        "capture_manifest_sha256": _sha256(capture_manifest_path),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_index_sha256": _sha256(index_path),
        "cholesky_info": cholesky_info,
        "contexts": processed,
        "stored_factor_damping": "none",
        "validation_damping_ratio": args.validation_damping_ratio,
        "factor": _factor_summary(factors[0]),
        "factor_sha256": _sha256(args.output),
        "fisher_samples": counts,
        "format_version": 1,
        "kind": "Kimi-K3 layer-92 routed-output empirical Fisher receipt",
        "lm_head_sha256": _sha256(lm_head_path),
        "pairs_per_row": args.pairs_per_row,
        "seed": args.seed,
        "split_a": _factor_summary(factors[1]),
        "split_b": _factor_summary(factors[2]),
        "split_cosine": _cosine(factors[1], factors[2]),
        "suite_manifest_sha256": _sha256(suite_manifest_path),
        "tensor_names": TENSOR_NAMES,
    }
    _write_json(args.output.with_suffix(".json"), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

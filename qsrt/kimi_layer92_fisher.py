"""Paired Kimi-K3 layer-92 samples for two-sided W2 curvature."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from qsrt.suffix_fisher import (
    kimi_layer92_routed_fisher_pullback,
    paired_lm_head_fisher_gradients,
)


CAPTURE_TENSORS = (
    "expert_indices",
    "expert_input",
    "final_mixed",
    "prefix_weight",
    "route_weights",
    "routed_latent",
    "row_index",
    "updated_prefix",
)

CHECKPOINT_TENSORS = {
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


def _context_split(context_index: int) -> int:
    digest = hashlib.sha256(
        f"kimi-k3-layer92-suffix-fisher-v1:{context_index}".encode()
    ).digest()
    return digest[0] & 1


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


@dataclass(frozen=True)
class PairedLayer92FisherSamples:
    """Aligned expert inputs, routing data, and common-suffix gradients."""

    expert_input: torch.Tensor
    expert_indices: torch.Tensor
    route_weights: torch.Tensor
    output_gradients: torch.Tensor
    context_index: torch.Tensor
    row_index: torch.Tensor
    split: torch.Tensor

    def validate(self) -> None:
        rows = int(self.expert_input.shape[0])
        if self.expert_input.ndim != 2 or not self.expert_input.is_floating_point():
            raise TypeError("paired expert inputs must be a floating-point matrix")
        if self.output_gradients.ndim != 2 or not (
            self.output_gradients.is_floating_point()
        ):
            raise TypeError("paired output gradients must be a floating-point matrix")
        if self.expert_indices.ndim != 2 or self.expert_indices.dtype != torch.int32:
            raise TypeError("paired expert indices must be an int32 matrix")
        if self.route_weights.shape != self.expert_indices.shape or not (
            self.route_weights.is_floating_point()
        ):
            raise TypeError("paired route weights must match the expert-index matrix")
        vectors = (self.context_index, self.row_index, self.split)
        if any(value.ndim != 1 or value.dtype != torch.int64 for value in vectors):
            raise TypeError("paired row identities and splits must be int64 vectors")
        if any(value.shape[0] != rows for value in (*vectors, self.output_gradients)):
            raise ValueError("paired Fisher tensors contain different row counts")
        if self.expert_indices.shape[0] != rows:
            raise ValueError("paired routing tensors contain the wrong row count")
        floating = (self.expert_input, self.route_weights, self.output_gradients)
        if any(value.device != self.expert_input.device for value in self.__dict__.values()):
            raise ValueError("paired Fisher tensors must share one device")
        if not all(bool(torch.all(torch.isfinite(value))) for value in floating):
            raise ValueError("paired Fisher tensors must be finite")
        if bool(torch.any(self.route_weights < 0)):
            raise ValueError("paired route weights must be nonnegative")
        if not bool(torch.all((self.split == 0) | (self.split == 1))):
            raise ValueError("paired context splits must be zero or one")

    def expert_occurrences(
        self,
        expert: int,
    ) -> dict[str, torch.Tensor]:
        """Select aligned rows for one routed expert."""

        self.validate()
        locations = torch.nonzero(self.expert_indices == expert, as_tuple=False)
        if locations.shape[0] == 0:
            raise ValueError(f"paired suffix capture contains no expert {expert} rows")
        rows = locations[:, 0]
        slots = locations[:, 1]
        return {
            "expert_input": self.expert_input.index_select(0, rows),
            "output_gradient": self.output_gradients.index_select(0, rows),
            "route_weight": self.route_weights[rows, slots],
            "context_index": self.context_index.index_select(0, rows),
            "row_index": self.row_index.index_select(0, rows),
            "split": self.split.index_select(0, rows),
        }


def load_paired_layer92_fisher_samples(
    *,
    capture_dir: Path,
    suite_dir: Path,
    checkpoint: Path,
    device: torch.device,
    pairs_per_row: int,
    seed: int,
) -> tuple[PairedLayer92FisherSamples, dict[str, object]]:
    """Load a sealed paired capture and construct real-Fisher suffix gradients."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("layer-92 Fisher construction requires CUDA")
    if pairs_per_row <= 0:
        raise ValueError("pairs per captured row must be positive")
    capture_manifest_path = capture_dir / "manifest.json"
    suite_manifest_path = suite_dir / "suite-manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    suite = json.loads(suite_manifest_path.read_text(encoding="utf-8"))
    required_capture = {
        "complete": True,
        "format_version": 2,
        "kind": "Kimi-K3 layer-92 paired expert-input and suffix capture",
        "partition": "analysis",
        "suite_token_hash_sha256": suite.get("suite_token_hash_sha256"),
    }
    for key, expected in required_capture.items():
        if capture_manifest.get(key) != expected:
            raise RuntimeError(f"paired suffix capture has invalid {key}")
    hidden_dimension = int(capture_manifest["hidden_dimension"])
    latent_dimension = int(capture_manifest["latent_dimension"])
    input_dimension = int(capture_manifest["expert_input_dimension"])
    experts_per_token = int(capture_manifest["experts_per_token"])
    num_experts = int(capture_manifest["num_experts"])

    suite_contexts = {int(item["context_index"]): item for item in suite["contexts"]}
    expected_indices = sorted(
        index
        for index, item in suite_contexts.items()
        if item.get("partition") == "analysis"
    )
    records = capture_manifest.get("contexts")
    if not isinstance(records, list) or not records:
        raise RuntimeError("paired suffix capture contains no contexts")
    observed_indices = sorted(int(record["context_index"]) for record in records)
    if observed_indices != expected_indices:
        raise RuntimeError("paired suffix capture does not exactly cover analysis contexts")

    checkpoint_root = None
    server_models = capture_manifest.get("server_models")
    if isinstance(server_models, dict):
        models = server_models.get("data")
        if isinstance(models, list):
            matching = [item for item in models if item.get("id") == capture_manifest.get("model")]
            if len(matching) == 1 and isinstance(matching[0].get("root"), str):
                checkpoint_root = Path(matching[0]["root"])
    if checkpoint_root is None:
        raise RuntimeError("paired suffix capture lacks its served checkpoint identity")
    if checkpoint_root.resolve() != checkpoint.resolve():
        raise RuntimeError(
            "paired suffix capture and suffix-gradient checkpoint paths differ"
        )

    index_path = checkpoint / "model.safetensors.index.json"
    checkpoint_index = json.loads(index_path.read_text(encoding="utf-8"))
    config = _text_config(checkpoint)
    if int(config["hidden_size"]) != hidden_dimension:
        raise RuntimeError("paired suffix capture and checkpoint hidden dimensions differ")
    epsilon = float(config["rms_norm_eps"])
    logit_scale = float(config.get("logit_scale", 1.0))
    static = {
        role: _load_checkpoint_tensor(checkpoint, checkpoint_index, name)
        for role, name in CHECKPOINT_TENSORS.items()
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
            raise RuntimeError(f"suffix checkpoint has invalid {role} shape")
    static = {key: value.float().to(device=device) for key, value in static.items()}
    attention_query = (
        static["attention_norm_weight"]
        * static["attention_projection_weight"].squeeze(0)
    )

    lm_head_path = suite_dir / "lm-head" / "weight.safetensors"
    lm_head = load_file(lm_head_path, device="cpu")["weight"]
    if lm_head.dtype != torch.bfloat16 or tuple(lm_head.shape[1:]) != (
        hidden_dimension,
    ):
        raise RuntimeError("fidelity LM head has an incompatible tensor")
    lm_head = lm_head.to(device=device)
    generator = torch.Generator(device=device).manual_seed(seed)

    collected: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "expert_input",
            "expert_indices",
            "route_weights",
            "output_gradients",
            "context_index",
            "row_index",
            "split",
        )
    }
    context_receipts: list[dict[str, int]] = []
    for record in records:
        context_index = int(record["context_index"])
        context = suite_contexts[context_index]
        if context.get("partition") != "analysis":
            raise RuntimeError("qualification context entered paired Fisher capture")
        token_hash = str(context["token_ids_json_sha256"])
        capture_path = capture_dir / str(record["file"])
        if _sha256(capture_path) != record["sha256"]:
            raise RuntimeError(f"paired capture hash mismatch for context {context_index}")
        with safe_open(capture_path, framework="pt", device="cpu") as reader:
            if tuple(reader.keys()) != CAPTURE_TENSORS:
                raise RuntimeError(f"paired capture has invalid keys in {capture_path}")
            metadata = reader.metadata() or {}
            expected_metadata = {
                "context_index": str(context_index),
                "format_version": "2",
                "partition": "analysis",
                "route_weight_semantics": "applied_moe_weight",
                "semantic_point": "kimi_k3_layer_92_common_suffix",
                "token_ids_json_sha256": token_hash,
            }
            for key, expected in expected_metadata.items():
                if metadata.get(key) != expected:
                    raise RuntimeError(f"paired capture has invalid {key} in {capture_path}")
        captured = load_file(capture_path, device="cpu")
        rows = captured["row_index"]
        row_count = int(rows.numel())
        expected_shapes = {
            "expert_input": (row_count, input_dimension),
            "expert_indices": (row_count, experts_per_token),
            "route_weights": (row_count, experts_per_token),
            "routed_latent": (row_count, latent_dimension),
            "updated_prefix": (row_count, hidden_dimension),
            "final_mixed": (row_count, hidden_dimension),
            "prefix_weight": (row_count,),
        }
        for key, shape in expected_shapes.items():
            if tuple(captured[key].shape) != shape:
                raise RuntimeError(f"paired capture has invalid {key} shape in {capture_path}")
        if captured["expert_indices"].dtype != torch.int32:
            raise RuntimeError("paired capture expert indices must be int32")
        if captured["route_weights"].dtype != torch.float32:
            raise RuntimeError("paired capture route weights must be FP32")
        indices = captured["expert_indices"]
        if not bool(torch.all((indices >= 0) & (indices < num_experts))):
            raise RuntimeError("paired capture contains an invalid expert index")

        reference = _validate_reference_hidden(
            suite_dir
            / "reference-hidden"
            / f"hidden_{context_index:04d}.safetensors",
            context_index=context_index,
            token_hash=token_hash,
            hidden_dimension=hidden_dimension,
        ).index_select(0, rows)
        logits = F.linear(reference.to(device=device), lm_head).float()
        logits.mul_(logit_scale)
        probabilities = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(
            probabilities,
            2 * pairs_per_row,
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
        repeated_states = {
            key: captured[key].to(device=device).repeat_interleave(
                pairs_per_row, dim=0
            )
            for key in (
                "routed_latent",
                "updated_prefix",
                "final_mixed",
                "prefix_weight",
            )
        }
        output_gradient = kimi_layer92_routed_fisher_pullback(
            repeated_states["routed_latent"],
            repeated_states["updated_prefix"],
            repeated_states["final_mixed"],
            repeated_states["prefix_weight"],
            hidden_gradient,
            final_norm_weight=static["final_norm_weight"],
            final_norm_epsilon=epsilon,
            attention_score_query=attention_query,
            attention_epsilon=epsilon,
            routed_norm_weight=static["routed_norm_weight"],
            routed_norm_epsilon=epsilon,
            routed_projection_weight=static["routed_projection_weight"],
        )
        repeated_rows = rows.to(device=device).repeat_interleave(pairs_per_row)
        sample_count = int(output_gradient.shape[0])
        collected["expert_input"].append(
            captured["expert_input"].to(device=device).repeat_interleave(
                pairs_per_row, dim=0
            )
        )
        collected["expert_indices"].append(
            captured["expert_indices"].to(device=device).repeat_interleave(
                pairs_per_row, dim=0
            )
        )
        collected["route_weights"].append(
            captured["route_weights"].to(device=device).repeat_interleave(
                pairs_per_row, dim=0
            )
        )
        collected["output_gradients"].append(output_gradient)
        collected["context_index"].append(
            torch.full(
                (sample_count,),
                context_index,
                dtype=torch.int64,
                device=device,
            )
        )
        collected["row_index"].append(repeated_rows)
        collected["split"].append(
            torch.full(
                (sample_count,),
                _context_split(context_index),
                dtype=torch.int64,
                device=device,
            )
        )
        context_receipts.append(
            {
                "context_index": context_index,
                "rows": row_count,
                "fisher_samples": sample_count,
                "split": _context_split(context_index),
            }
        )

    samples = PairedLayer92FisherSamples(
        **{key: torch.cat(value, dim=0).contiguous() for key, value in collected.items()}
    )
    samples.validate()
    support: dict[str, object] = {
        "capture_manifest": str(capture_manifest_path.resolve()),
        "capture_manifest_sha256": _sha256(capture_manifest_path),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_index_sha256": _sha256(index_path),
        "contexts": len(records),
        "captured_rows": sum(int(item["rows"]) for item in context_receipts),
        "fisher_samples": int(samples.expert_input.shape[0]),
        "pairs_per_row": pairs_per_row,
        "seed": seed,
        "suite_manifest": str(suite_manifest_path.resolve()),
        "suite_manifest_sha256": _sha256(suite_manifest_path),
        "lm_head_sha256": _sha256(lm_head_path),
        "route_weight_semantics": "applied once on the decoded W2 input row",
    }
    return samples, support


__all__ = [
    "CAPTURE_TENSORS",
    "CHECKPOINT_TENSORS",
    "PairedLayer92FisherSamples",
    "load_paired_layer92_fisher_samples",
]

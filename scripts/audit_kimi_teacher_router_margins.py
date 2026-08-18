#!/usr/bin/env python3
"""Recompute Kimi teacher router score margins from stored boundary states."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

_REPOSITORY = Path(__file__).resolve().parents[1]
if str(_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY))

from qsrt.instanttensor_kimi import InstantTensorLoadConfig, OfficialKimiLayerShards
from qsrt.kimi_boundary_slabs import KimiBoundarySlabArchive
from qsrt.kimi_official_forward import (
    OfficialKimiForwardAdapter,
    load_official_kimi_runtime,
    new_meta_decoder_layer,
)
from qsrt.kimi_routes import KimiRouteArchive
from qsrt.kimi_stream import assign_parameter, fit_checkpoint_parameter
from qsrt.kimi_quantized_forward import _load_nonexpert_layer


DEFAULT_BOUNDARIES = Path(
    "/data/kquant/research/qsrt-continuous-recovery-m0/teacher-legacy32-boundaries"
)
DEFAULT_ROUTES = Path(
    "/data/kquant/research/qsrt-continuous-recovery-m0/teacher-legacy32-routes"
)
DEFAULT_WEIGHT_CHECKPOINT = Path(
    "/data/cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
)
DEFAULT_CODE_CHECKPOINT = Path(
    "/data/cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/"
    "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721"
)


class _GateReached(RuntimeError):
    pass


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    probabilities = torch.tensor(
        [0.0, 0.001, 0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99, 0.999, 1.0],
        dtype=torch.float64,
    )
    quantiles = torch.quantile(values.to(torch.float64), probabilities)
    labels = (
        "min",
        "p001",
        "p01",
        "p05",
        "p10",
        "median",
        "p90",
        "p95",
        "p99",
        "p999",
        "max",
    )
    return {
        label: float(value)
        for label, value in zip(labels, quantiles.tolist(), strict=True)
    }


def _load_gate_prefix_parameters(
    module: torch.nn.Module,
    *,
    checkpoint: Path,
    layer: int,
    device: torch.device,
) -> dict[str, object]:
    """Load only parameters executed before routed-expert dispatch."""

    index = OfficialKimiLayerShards(checkpoint)
    prefix = index.prefix(layer)
    shard = index.layer_shard(layer)
    parameters = dict(module.named_parameters())
    local_names = tuple(
        name
        for name in parameters
        if not name.startswith("block_sparse_moe.")
        or name.startswith("block_sparse_moe.gate.")
    )
    checkpoint_names = tuple(f"{prefix}{name}" for name in local_names)
    loaded_bytes = 0
    fixes: list[str] = []
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        missing = set(checkpoint_names) - keys
        if missing:
            raise KeyError(f"checkpoint is missing {sorted(missing)[:3]}")
        for local_name, checkpoint_name in zip(
            local_names, checkpoint_names, strict=True
        ):
            value = handle.get_tensor(checkpoint_name)
            fitted, fix = fit_checkpoint_parameter(
                checkpoint_name,
                value,
                tuple(parameters[local_name].shape),
            )
            owned = fitted.to(device=device, non_blocking=False)
            assign_parameter(module, local_name, owned)
            loaded_bytes += owned.numel() * owned.element_size()
            if fix is not None:
                fixes.append(f"{checkpoint_name}: {fix}")
    remaining_used_meta = [
        name
        for name, parameter in module.named_parameters()
        if parameter.is_meta and name in local_names
    ]
    if remaining_used_meta:
        raise RuntimeError(f"gate-prefix parameters remain meta: {remaining_used_meta[:3]}")
    return {
        "shard": str(shard),
        "loaded_parameters": len(local_names),
        "loaded_bytes": loaded_bytes,
        "compatibility_fixes": fixes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARIES)
    parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
    parser.add_argument("--weight-checkpoint", type=Path, default=DEFAULT_WEIGHT_CHECKPOINT)
    parser.add_argument("--code-checkpoint", type=Path, default=DEFAULT_CODE_CHECKPOINT)
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        help="quantized checkpoint supplying the executed non-expert layer",
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-scores", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("device must be an indexed CUDA device")
    torch.cuda.set_device(device)
    boundaries = KimiBoundarySlabArchive(args.boundaries)
    routes = KimiRouteArchive(args.routes)
    if boundaries.token_count != routes.token_count:
        raise ValueError("boundary and route archives have different token counts")
    documents = boundaries.load_documents()
    archived_ids = routes.read_layer(args.layer)

    runtime = load_official_kimi_runtime(
        weight_checkpoint=args.weight_checkpoint,
        code_checkpoint=args.code_checkpoint,
    )
    module = new_meta_decoder_layer(runtime, args.layer)
    load_started = time.monotonic()
    if args.model_checkpoint is None:
        load_receipt = _load_gate_prefix_parameters(
            module,
            checkpoint=args.weight_checkpoint,
            layer=args.layer,
            device=device,
        )
    else:
        model_checkpoint = args.model_checkpoint.expanduser().resolve()
        stats = _load_nonexpert_layer(
            module,
            layer=args.layer,
            checkpoint=model_checkpoint,
            index=OfficialKimiLayerShards(model_checkpoint),
            device=device,
            config=InstantTensorLoadConfig(),
        )
        load_receipt = {
            "shard": stats.shard,
            "loaded_parameters": stats.nonexpert_parameters,
            "loaded_bytes": stats.dense_bytes,
            "compatibility_fixes": list(stats.compatibility_fixes),
        }
    load_receipt["elapsed_seconds"] = time.monotonic() - load_started
    module.eval()
    adapter = OfficialKimiForwardAdapter(runtime, validate_outputs=False)
    gate = module.block_sparse_moe.gate
    if gate.num_expert_group != 1 or gate.topk_group != 1:
        raise NotImplementedError("group-constrained router margins require group masking")
    correction_bias_cpu = gate.e_score_correction_bias.detach().cpu().contiguous()

    captured: dict[str, torch.Tensor] = {}

    def gate_hook(_gate: Any, inputs: tuple[torch.Tensor, ...], output: Any) -> None:
        if len(inputs) != 1 or not isinstance(output, tuple) or len(output) != 2:
            raise TypeError("unexpected gate hook contract")
        hidden = inputs[0].reshape(-1, inputs[0].shape[-1])
        logits = F.linear(hidden.float(), gate.weight.float())
        if gate.moe_router_activation_func == "sigmoid":
            scores = logits.sigmoid()
        elif gate.moe_router_activation_func == "softmax":
            scores = logits.softmax(dim=1)
        else:
            raise NotImplementedError(gate.moe_router_activation_func)
        choice = scores + gate.e_score_correction_bias.unsqueeze(0)
        top17_values, top17_ids = torch.topk(choice, 17, dim=1, sorted=True)
        captured["gate_ids"] = output[0].detach().cpu()
        captured["recomputed_ids"] = top17_ids[:, :16].detach().cpu()
        captured["margins"] = (top17_values[:, 15] - top17_values[:, 16]).detach().cpu()
        if args.save_scores is not None:
            captured["scores"] = scores.detach().cpu()
        raise _GateReached

    hook = gate.register_forward_hook(gate_hook)
    margin_parts: list[torch.Tensor] = []
    score_parts: list[torch.Tensor] = []
    archived_set_mismatch = 0
    recomputed_set_mismatch = 0
    recomputed_mismatch_margin_max = 0.0
    started = time.monotonic()
    try:
        for document in range(documents.document_count):
            first, end = documents.document_extent(document)
            hidden = boundaries.read_cpu(
                args.layer, first, end, direct=True, pin_memory=True
            ).to(device=device, non_blocking=True).unsqueeze(0)
            residual = boundaries.reconstruct_block_residual(
                layer=args.layer,
                first_token=first,
                end_token=end,
                device=device,
                direct=True,
            )
            captured.clear()
            try:
                with torch.inference_mode():
                    adapter.forward_layer(
                        module,
                        layer=args.layer,
                        hidden_states=hidden,
                        block_residual=residual,
                    )
            except _GateReached:
                pass
            else:
                raise RuntimeError("decoder layer completed without reaching the router gate")
            gate_ids = captured["gate_ids"].to(torch.int16)
            recomputed_ids = captured["recomputed_ids"].to(torch.int16)
            archived = archived_ids[first:end]
            archived_set_mismatch += int(
                (gate_ids.sort(dim=1).values != archived.sort(dim=1).values)
                .any(dim=1)
                .sum()
            )
            recomputed_mismatch = (
                gate_ids.sort(dim=1).values != recomputed_ids.sort(dim=1).values
            ).any(dim=1)
            recomputed_set_mismatch += int(recomputed_mismatch.sum())
            if bool(recomputed_mismatch.any()):
                recomputed_mismatch_margin_max = max(
                    recomputed_mismatch_margin_max,
                    float(captured["margins"][recomputed_mismatch].max()),
                )
            margin_parts.append(captured["margins"])
            if args.save_scores is not None:
                score_parts.append(captured["scores"])
    finally:
        hook.remove()
        adapter.release_layer(module)

    margins = torch.cat(margin_parts).to(torch.float64)
    thresholds = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
    report = {
        "kind": "Kimi-K3 teacher biased router top-16 boundary margin",
        "layer": args.layer,
        "boundary_archive": str(args.boundaries.resolve()),
        "route_archive": str(args.routes.resolve()),
        "weight_checkpoint": str(args.weight_checkpoint.resolve()),
        "model_checkpoint": (
            None if args.model_checkpoint is None else str(args.model_checkpoint.resolve())
        ),
        "code_checkpoint": str(args.code_checkpoint.resolve()),
        "token_count": int(margins.numel()),
        "score_definition": "router activation plus e_score_correction_bias",
        "margin_definition": "16th-largest biased score minus 17th-largest biased score",
        "margin_quantiles": _quantiles(margins),
        "mean_margin": float(margins.mean()),
        "root_mean_square_margin": float(margins.square().mean().sqrt()),
        "zero_margin_tokens": int((margins == 0).sum()),
        "fraction_at_or_below": {
            f"{threshold:.0e}": float((margins <= threshold).to(torch.float64).mean())
            for threshold in thresholds
        },
        "route_validation": {
            "archived_top16_set_mismatch_tokens": archived_set_mismatch,
            "recomputed_vs_gate_top16_set_mismatch_tokens": recomputed_set_mismatch,
            "recomputed_mismatch_max_margin": recomputed_mismatch_margin_max,
        },
        "gate_prefix_load": load_receipt,
        "replay_elapsed_seconds": time.monotonic() - started,
    }
    if archived_set_mismatch:
        raise RuntimeError(f"router replay validation failed: {report['route_validation']}")
    if args.save_scores is not None:
        args.save_scores.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.save_scores.with_name(
            f".{args.save_scores.name}.{os.getpid()}.tmp"
        )
        save_file(
            {
                "scores": torch.cat(score_parts).contiguous(),
                "correction_bias": correction_bias_cpu,
            },
            str(temporary),
            metadata={
                "kind": "Kimi-K3 unbiased router scores for correction-bias fitting",
                "layer": str(args.layer),
                "boundary_archive": str(args.boundaries.resolve()),
                "route_archive": str(args.routes.resolve()),
                "model_checkpoint": (
                    str(args.weight_checkpoint.resolve())
                    if args.model_checkpoint is None
                    else str(args.model_checkpoint.resolve())
                ),
            },
        )
        temporary.replace(args.save_scores)
        report["score_archive"] = str(args.save_scores.resolve())
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

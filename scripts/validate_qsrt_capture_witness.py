#!/usr/bin/env python3
"""Validate a tiny QSRT capture against eager Kimi correctness traces.

The witness run must use input sampling rate one and save complete token rows
for the selected calls.  Matching is content-based: capture epochs are paired
with trace calls by their MoE input, then routes, applied gates, and the
post-all-reduce/pre-RMSNorm routed latent are checked on the same token rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from qsrt.blockldlq_proof import (
    capture_sample_selected,
    capture_validation_split,
)


_TRACE_RE = re.compile(
    r"layer-(?P<layer>\d+)\.call-(?P<call>\d+)\.(?P<stage>[^.]+)\.pt$"
)
_STAGES = (
    "routed_latent_input",
    "canonical_topk_ids",
    "canonical_topk_weights",
    "routed_latent_reduced",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _parse_layers(value: str) -> list[int]:
    layers = sorted({int(item) for item in value.split(",") if item.strip()})
    if not layers or any(layer <= 0 for layer in layers):
        raise argparse.ArgumentTypeError("layers must be positive, comma-separated")
    return layers


def _load_trace(path: Path) -> tuple[dict[str, Any], torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    tensor = payload["tensor"].contiguous()
    digest = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError(f"trace digest mismatch: {path}")
    if list(tensor.shape) != metadata["saved_shape"]:
        raise ValueError(f"trace shape metadata mismatch: {path}")
    return metadata, tensor


def _index_traces(trace_dir: Path, layer: int) -> dict[int, dict[str, torch.Tensor]]:
    result: dict[int, dict[str, torch.Tensor]] = {}
    rank_dir = trace_dir / "tp-rank-000"
    for path in sorted(rank_dir.glob(f"layer-{layer:03d}.call-*.pt")):
        match = _TRACE_RE.match(path.name)
        if match is None or int(match.group("layer")) != layer:
            continue
        stage = match.group("stage")
        if stage not in _STAGES:
            continue
        call = int(match.group("call"))
        metadata, tensor = _load_trace(path)
        if int(metadata["tp_rank"]) != 0 or int(metadata["layer"]) != layer:
            raise ValueError(f"trace identity mismatch: {path}")
        result.setdefault(call, {})[stage] = tensor
    return {
        call: stages
        for call, stages in result.items()
        if all(stage in stages for stage in _STAGES)
    }


def _match_epoch(
    *,
    indices: torch.Tensor,
    token_indices: torch.Tensor,
    values: torch.Tensor,
    traces: dict[int, dict[str, torch.Tensor]],
    used_calls: set[int],
) -> int:
    maximum = int(token_indices.max()) if token_indices.numel() else -1
    for call, stages in sorted(traces.items()):
        if call in used_calls:
            continue
        trace = stages["routed_latent_input"]
        if trace.ndim != 2 or maximum >= trace.shape[0]:
            continue
        candidate = trace.index_select(0, token_indices).to(values.dtype)
        if torch.equal(values.index_select(0, indices), candidate):
            return call
    raise AssertionError("no eager trace call matches the captured MoE input epoch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--sample-cache", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--layers", type=_parse_layers, default=[1, 24, 92])
    parser.add_argument(
        "--output", type=Path, default=Path("out/qsrt-capture-witness.json")
    )
    args = parser.parse_args()

    capture_manifest = _json(args.capture / "manifest.json")
    cache_manifest = _json(args.sample_cache / "manifest.json")
    if int(capture_manifest["sampling"]["input_hessian"]) != 1:
        raise ValueError("capture witness requires input_hessian sample rate one")
    if Path(cache_manifest["source_capture"]).resolve() != args.capture.resolve():
        raise ValueError("sample cache does not name the witness capture")
    modulus = int(capture_manifest["sampling"]["validation_modulus"])

    layer_results: list[dict[str, Any]] = []
    for layer in args.layers:
        # Capture/cache layer IDs are decoder-layer IDs.  Layer zero is the
        # dense warmup layer; routed experts occupy decoder layers 1..92.
        trace_layer = layer
        entry = cache_manifest["layers"][str(layer)]
        tensors = load_file(str(args.sample_cache / entry["file"]), device="cpu")
        values = tensors["input.values"]
        observations = tensors["input.observation"].to(torch.int64)
        experts = tensors["input.experts"]
        gates = tensors["input.gates"].float()
        weights = tensors["input.weight"].float()
        splits = tensors["input.split"].to(torch.int8)
        routed_latent = tensors["input.routed_latent"]
        traces = _index_traces(args.trace_dir, trace_layer)
        if not traces:
            raise ValueError(
                f"layer {layer} has no complete rank-zero trace calls "
                f"for decoder layer {trace_layer}"
            )

        expected_weight = gates.square().sum(dim=1)
        torch.testing.assert_close(weights, expected_weight, rtol=2e-6, atol=2e-7)
        expected_split = torch.tensor(
            [capture_validation_split(int(value), modulus) for value in observations],
            dtype=torch.int8,
        )
        if not torch.equal(splits, expected_split):
            raise AssertionError(f"layer {layer} split replay differs")
        if not all(capture_sample_selected(int(value), 1) for value in observations):
            raise AssertionError(f"layer {layer} contains a nonselected row")

        epochs = torch.bitwise_right_shift(observations, 32)
        token_ids = torch.bitwise_and(observations, 0xFFFFFFFF)
        used_calls: set[int] = set()
        epoch_results: list[dict[str, Any]] = []
        for epoch in sorted(int(value) for value in torch.unique(epochs).tolist()):
            indices = torch.nonzero(epochs == epoch, as_tuple=False).flatten()
            tokens = token_ids.index_select(0, indices).to(torch.long)
            call = _match_epoch(
                indices=indices,
                token_indices=tokens,
                values=values,
                traces=traces,
                used_calls=used_calls,
            )
            used_calls.add(call)
            trace = traces[call]
            actual_experts = trace["canonical_topk_ids"].index_select(0, tokens)
            actual_gates = trace["canonical_topk_weights"].index_select(0, tokens)
            actual_latent = trace["routed_latent_reduced"].index_select(0, tokens)
            if not torch.equal(experts.index_select(0, indices), actual_experts.to(experts.dtype)):
                raise AssertionError(f"layer {layer} epoch {epoch} routes differ")
            torch.testing.assert_close(
                gates.index_select(0, indices),
                actual_gates.float(),
                # The trace independently reruns grouped-topk while capture
                # observes the router's applied result.  Their FP32 reductions
                # can differ by a few ulps even when routes are identical.
                rtol=5e-6,
                atol=1e-7,
            )
            gate_error = (
                gates.index_select(0, indices) - actual_gates.float()
            ).abs()
            if not torch.equal(
                routed_latent.index_select(0, indices),
                actual_latent.to(routed_latent.dtype),
            ):
                raise AssertionError(
                    f"layer {layer} epoch {epoch} routed latent differs"
                )
            epoch_results.append(
                {
                    "epoch": epoch,
                    "trace_call": call,
                    "rows": len(indices),
                    "maximum_gate_absolute_error": float(gate_error.max()),
                }
            )
        layer_results.append(
            {
                "layer": layer,
                "trace_layer": trace_layer,
                "rows": int(values.shape[0]),
                "epochs": epoch_results,
                "trace_calls_available": len(traces),
            }
        )

    result = {
        "schema_version": 1,
        "kind": "qsrt_capture_eager_witness",
        "passed": True,
        "capture": str(args.capture.resolve()),
        "sample_cache": str(args.sample_cache.resolve()),
        "trace_dir": str(args.trace_dir.resolve()),
        "proved": [
            "captured input rows equal eager routed-latent expert input rows",
            "captured expert IDs equal canonical post-router top-k IDs",
            "captured gates match an independent FP32 grouped-topk replay within 5e-6 relative tolerance",
            "captured routed latent equals post-TP-all-reduce/pre-RMSNorm trace",
            "observation sampling, split, and gate-square weights replay exactly",
        ],
        "layers": layer_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

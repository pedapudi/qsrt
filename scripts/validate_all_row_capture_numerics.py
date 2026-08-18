#!/usr/bin/env python3
"""Compare captured routed outputs with independently decoded QSRT experts.

The validator reconstructs every expert selected by the sampled rows directly
from the candidate-pool trellis and scale tensors.  It applies the selected
coupled activation-boundary transform, evaluates each expert in PyTorch, and
sums the outputs using the applied route weights stored in the capture.  The
result is independent of the serving kernel and its atom layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from qsrt.all_row_capture import load_all_row_capture
from qsrt.pack.qsrt_validation import decode_candidate_matrix
from qsrt.qsrt_coupled import CoupledHadamardExecution, CoupledHadamardSpec


def _parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(item) for item in value.split(",") if item)
    if not layers or len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("layers must be a unique comma-separated list")
    return layers


def _sample_rows(
    chunks: tuple[Any, ...], count: int
) -> dict[str, torch.Tensor]:
    parts: dict[str, list[torch.Tensor]] = {
        "input": [],
        "expert_indices": [],
        "route_weights": [],
        "routed_output": [],
        "request_index": [],
        "token_offset": [],
    }
    remaining = count
    for chunk in chunks:
        if remaining == 0:
            break
        tensors = chunk.load()
        take = min(remaining, chunk.rows)
        for name in parts:
            parts[name].append(tensors[name][:take])
        remaining -= take
    if remaining:
        raise ValueError(f"capture contains fewer than {count} requested sample rows")
    return {name: torch.cat(values) for name, values in parts.items()}


def _evaluate_layer(
    *,
    candidate_pool: Path,
    layer: int,
    rows: dict[str, torch.Tensor],
    device: torch.device,
    codebook: str,
    logical_trellis_schema: str,
) -> dict[str, Any]:
    candidate_path = candidate_pool / "candidates" / f"qsrt-layer-{layer:05d}.safetensors"
    metrics_path = (
        candidate_pool / "candidates" / f"qsrt-layer-{layer:05d}.metrics.safetensors"
    )
    if not candidate_path.is_file() or not metrics_path.is_file():
        raise ValueError(f"candidate pool lacks canonical layer {layer} artifacts")
    metrics = load_file(str(metrics_path), device="cpu")
    required = {"coupled_draw_selected", "selected_r13", "selected_r2"}
    if not required.issubset(metrics):
        raise ValueError(f"layer {layer} metrics lack coupled candidate selections")

    inputs = rows["input"].to(device=device, dtype=torch.float32)
    expert_indices = rows["expert_indices"].to(device=device, dtype=torch.int64)
    route_weights = rows["route_weights"].to(device=device, dtype=torch.float32)
    target = rows["routed_output"].to(device=device, dtype=torch.float32)
    reconstructed = torch.zeros_like(target)
    experts = torch.unique(expert_indices).cpu().tolist()

    with safe_open(candidate_path, framework="pt", device="cpu") as reader:
        for expert in experts:
            r13_mode = int(metrics["selected_r13"][expert])
            r2_mode = int(metrics["selected_r2"][expert])
            draw = int(metrics["coupled_draw_selected"][expert])
            weights = []
            for matrix, mode in (("w1", r13_mode), ("w3", r13_mode), ("w2", r2_mode)):
                decoded = decode_candidate_matrix(
                    reader,
                    layer=layer,
                    expert=expert,
                    matrix=matrix,
                    mode_id=mode,
                    device=device,
                    logical_trellis_schema=logical_trellis_schema,
                    codebook=codebook,
                )
                weights.append(decoded.T.float().contiguous())
            execution = CoupledHadamardExecution(
                hidden=inputs.shape[1],
                intermediate=weights[0].shape[0],
                spec=CoupledHadamardSpec(intermediate_draw=draw),
            )
            output = execution.execute(inputs, tuple(weights))
            applied = torch.where(
                expert_indices == expert,
                route_weights,
                torch.zeros_like(route_weights),
            ).sum(dim=1)
            reconstructed.add_(output * applied[:, None])
            del output, weights

    row_cosine = torch.nn.functional.cosine_similarity(reconstructed, target, dim=1)
    error = reconstructed - target
    target_energy = target.square().sum(dtype=torch.float64)
    relative_sse = error.square().sum(dtype=torch.float64) / target_energy
    return {
        "layer": layer,
        "rows": int(inputs.shape[0]),
        "unique_experts": len(experts),
        "request_indices": rows["request_index"].tolist(),
        "token_offsets": rows["token_offset"].tolist(),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                reconstructed.flatten(), target.flatten(), dim=0
            )
        ),
        "minimum_row_cosine": float(row_cosine.min()),
        "relative_sse": float(relative_sse),
        "maximum_absolute_error": float(error.abs().max()),
        "reference_norm": float(torch.linalg.vector_norm(reconstructed)),
        "captured_norm": float(torch.linalg.vector_norm(target)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--layers", type=_parse_layers, default=(1, 24, 92))
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-cosine", type=float, default=0.9999)
    parser.add_argument(
        "--verify-capture-hashes",
        action="store_true",
        help="rehash every capture chunk before sampling rows",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rows <= 0 or not -1.0 <= args.minimum_cosine <= 1.0:
        parser.error("rows must be positive and minimum cosine must lie in [-1, 1]")

    manifest = json.loads(
        (args.candidate_pool / "qsrt-candidate-manifest.json").read_text()
    )
    codebook = str(manifest["codebook"])
    logical_trellis_schema = str(manifest["logical_trellis_schema"])
    capture_manifest, geometry, chunks = load_all_row_capture(
        args.capture, verify_hashes=args.verify_capture_hashes
    )
    absent = sorted(set(args.layers) - set(geometry.layers))
    if absent:
        parser.error(f"capture lacks requested layers: {absent}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA device requested but CUDA is unavailable")

    results = []
    for layer in args.layers:
        rows = _sample_rows(chunks[layer], args.rows)
        results.append(
            _evaluate_layer(
                candidate_pool=args.candidate_pool,
                layer=layer,
                rows=rows,
                device=device,
                codebook=codebook,
                logical_trellis_schema=logical_trellis_schema,
            )
        )
    passed = all(item["minimum_row_cosine"] >= args.minimum_cosine for item in results)
    receipt = {
        "kind": "qsrt_all_routed_rows_numerical_validation",
        "capture": str(args.capture.resolve()),
        "capture_rows": int(capture_manifest["rows"]),
        "candidate_pool": str(args.candidate_pool.resolve()),
        "candidate_pool_content_sha256": json.loads(
            (args.candidate_pool / "qsrt-candidate-completion.json").read_text()
        ).get("content_sha256"),
        "codebook": codebook,
        "logical_trellis_schema": logical_trellis_schema,
        "minimum_cosine": args.minimum_cosine,
        "passed": passed,
        "layers": results,
    }
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(encoded)
        temporary.replace(args.output)
    print(encoded, end="")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

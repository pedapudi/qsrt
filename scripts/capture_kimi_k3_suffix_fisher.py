#!/usr/bin/env python3
"""Capture sparse Kimi-K3 layer-92 suffix states on analysis contexts.

The running vLLM server must set ``VLLM_QSRT_SUFFIX_FISHER_CAPTURE_DIR`` to
the same raw capture directory passed here.  Row stride and offset are runtime
properties recorded by every chunk.  Only contexts assigned to the analysis
partition are requested; the qualification partition is never read or served.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


CHUNK_RE = re.compile(r"suffix\.rows-(\d+)-(\d+)\.safetensors$")
TENSOR_KEYS = (
    "expert_indices",
    "expert_input",
    "final_mixed",
    "prefix_weight",
    "route_weights",
    "routed_latent",
    "row_index",
    "updated_prefix",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_tokens(path: Path) -> tuple[list[int], str]:
    tokens = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(tokens, list) or not all(
        isinstance(item, int) for item in tokens
    ):
        raise TypeError(f"token file must contain an integer array: {path}")
    canonical = json.dumps(tokens, separators=(",", ":"))
    return tokens, hashlib.sha256(canonical.encode()).hexdigest()


def _http_json(
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    if not isinstance(result, dict):
        raise TypeError(f"HTTP response from {url} must be an object")
    return result


def _load_chunks(
    request_dir: Path,
    *,
    scored_rows: int,
    row_stride: int,
    row_offset: int,
    expert_input_dimension: int,
    experts_per_token: int,
    latent_dimension: int,
    hidden_dimension: int,
    num_experts: int,
) -> dict[str, torch.Tensor]:
    paths: list[tuple[int, int, Path]] = []
    for path in request_dir.glob("suffix.rows-*.safetensors"):
        match = CHUNK_RE.fullmatch(path.name)
        if match is not None:
            paths.append((int(match[1]), int(match[2]), path))
    paths.sort()
    if not paths:
        raise RuntimeError(f"no suffix capture chunks in {request_dir}")

    pieces: dict[str, list[torch.Tensor]] = {key: [] for key in TENSOR_KEYS}
    for start, end, path in paths:
        with safe_open(path, framework="pt", device="cpu") as reader:
            if tuple(reader.keys()) != TENSOR_KEYS:
                raise RuntimeError(f"unexpected tensor keys in {path}")
            metadata = reader.metadata() or {}
            expected_metadata = {
                "format_version": "2",
                "row_start": str(start),
                "row_end": str(end),
                "row_stride": str(row_stride),
                "row_offset": str(row_offset),
                "expert_input_dimension": str(expert_input_dimension),
                "experts_per_token": str(experts_per_token),
                "latent_dimension": str(latent_dimension),
                "hidden_dimension": str(hidden_dimension),
                "semantic_point": "kimi_k3_layer_92_common_suffix",
                "route_weight_semantics": "applied_moe_weight",
            }
            for key, expected in expected_metadata.items():
                if metadata.get(key) != expected:
                    raise RuntimeError(f"invalid {key} metadata in {path}")
        chunk = load_file(path, device="cpu")
        row_index = chunk["row_index"]
        if row_index.dtype != torch.int64 or row_index.ndim != 1:
            raise RuntimeError(f"invalid row indices in {path}")
        if row_index.numel() == 0:
            raise RuntimeError(f"empty suffix capture chunk: {path}")
        if not bool(torch.all((row_index >= start) & (row_index < end))):
            raise RuntimeError(f"row index lies outside its chunk in {path}")
        if not bool(torch.all(row_index.remainder(row_stride) == row_offset)):
            raise RuntimeError(f"row index violates capture stride in {path}")
        count = int(row_index.numel())
        expected_shapes = {
            "expert_indices": (count, experts_per_token),
            "expert_input": (count, expert_input_dimension),
            "final_mixed": (count, hidden_dimension),
            "prefix_weight": (count,),
            "route_weights": (count, experts_per_token),
            "routed_latent": (count, latent_dimension),
            "updated_prefix": (count, hidden_dimension),
        }
        for key, shape in expected_shapes.items():
            value = chunk[key]
            if tuple(value.shape) != shape:
                raise RuntimeError(f"invalid {key} shape in {path}")
        if chunk["expert_indices"].dtype != torch.int32:
            raise RuntimeError(f"expert indices must be int32 in {path}")
        for key in ("prefix_weight", "route_weights"):
            if chunk[key].dtype != torch.float32:
                raise RuntimeError(f"{key} must be FP32 in {path}")
        for key in (
            "expert_input",
            "final_mixed",
            "routed_latent",
            "updated_prefix",
        ):
            if chunk[key].dtype != torch.bfloat16:
                raise RuntimeError(f"{key} must be BF16 in {path}")
        floating_fields = (
            "expert_input",
            "final_mixed",
            "prefix_weight",
            "route_weights",
            "routed_latent",
            "updated_prefix",
        )
        for key in floating_fields:
            if not bool(torch.all(torch.isfinite(chunk[key]))):
                raise RuntimeError(f"{key} is not finite in {path}")
        if not bool(torch.all(chunk["route_weights"] >= 0)):
            raise RuntimeError(f"route weights are negative in {path}")
        if not bool(
            torch.all(
                (chunk["prefix_weight"] >= 0)
                & (chunk["prefix_weight"] <= 1)
            )
        ):
            raise RuntimeError(f"prefix weights lie outside [0, 1] in {path}")
        indices = chunk["expert_indices"]
        if not bool(torch.all((indices >= 0) & (indices < num_experts))):
            raise RuntimeError(f"expert index lies outside [0, {num_experts}) in {path}")
        sorted_indices = torch.sort(indices, dim=-1).values
        if bool(torch.any(torch.diff(sorted_indices, dim=-1) == 0)):
            raise RuntimeError(f"duplicate routed expert in {path}")
        retained = row_index < scored_rows
        for key in TENSOR_KEYS:
            pieces[key].append(chunk[key][retained])

    result = {key: torch.cat(value, dim=0) for key, value in pieces.items()}
    expected_rows = torch.arange(row_offset, scored_rows, row_stride, dtype=torch.int64)
    if not torch.equal(result["row_index"], expected_rows):
        raise RuntimeError(
            f"suffix rows are incomplete: got {result['row_index'].tolist()}, "
            f"expected {expected_rows.tolist()}"
        )
    return {key: value.contiguous() for key, value in result.items()}


def _validate_output(
    path: Path,
    *,
    token_hash: str,
    expected_rows: torch.Tensor,
    expert_input_dimension: int,
    experts_per_token: int,
    latent_dimension: int,
    hidden_dimension: int,
    num_experts: int,
) -> None:
    with safe_open(path, framework="pt", device="cpu") as reader:
        if tuple(reader.keys()) != TENSOR_KEYS:
            raise RuntimeError(f"unexpected tensor keys in {path}")
        metadata = reader.metadata() or {}
        expected_metadata = {
            "format_version": "2",
            "token_ids_json_sha256": token_hash,
            "semantic_point": "kimi_k3_layer_92_common_suffix",
            "route_weight_semantics": "applied_moe_weight",
            "expert_input_dimension": str(expert_input_dimension),
            "experts_per_token": str(experts_per_token),
            "latent_dimension": str(latent_dimension),
            "hidden_dimension": str(hidden_dimension),
            "num_experts": str(num_experts),
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise RuntimeError(f"invalid {key} metadata in {path}")
        rows = reader.get_tensor("row_index")
    if not torch.equal(rows, expected_rows):
        raise RuntimeError(f"row identity mismatch in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", default="Kimi-K3")
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--raw-capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--row-stride", type=int, default=256)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--expert-input-dimension", type=int, default=3584)
    parser.add_argument("--experts-per-token", type=int, default=16)
    parser.add_argument("--latent-dimension", type=int, default=3584)
    parser.add_argument("--hidden-dimension", type=int, default=7168)
    parser.add_argument("--num-experts", type=int, default=896)
    parser.add_argument("--start-context", type=int, default=0)
    parser.add_argument("--stop-context", type=int)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--delete-raw-chunks-after-finalize", action="store_true")
    args = parser.parse_args()
    if args.row_stride <= 0 or not 0 <= args.row_offset < args.row_stride:
        parser.error("row offset must lie within a positive row stride")

    suite_path = args.suite_dir / "suite-manifest.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    analysis_contexts = [
        item for item in suite["contexts"] if item.get("partition") == "analysis"
    ]
    analysis_indices = [int(item["context_index"]) for item in analysis_contexts]
    analysis_indices_sha256 = hashlib.sha256(
        json.dumps(analysis_indices, separators=(",", ":")).encode()
    ).hexdigest()
    stop = (
        len(analysis_contexts) if args.stop_context is None else args.stop_context
    )
    contexts = analysis_contexts[args.start_context : stop]
    scored_rows = int(suite["scored_positions_per_context"])
    expected_rows = torch.arange(
        args.row_offset, scored_rows, args.row_stride, dtype=torch.int64
    )

    args.raw_capture_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        models_response = _http_json(
            args.url.rsplit("/v1/", 1)[0] + "/v1/models",
            timeout=args.timeout,
        )
        manifest = {
            "created_utc": datetime.now(UTC).isoformat(),
            "format_version": 2,
            "kind": "Kimi-K3 layer-92 paired expert-input and suffix capture",
            "model": args.model,
            "partition": "analysis",
            "row_offset": args.row_offset,
            "row_stride": args.row_stride,
            "expert_input_dimension": args.expert_input_dimension,
            "experts_per_token": args.experts_per_token,
            "latent_dimension": args.latent_dimension,
            "hidden_dimension": args.hidden_dimension,
            "num_experts": args.num_experts,
            "rows_per_context": int(expected_rows.numel()),
            "scored_rows_per_context": scored_rows,
            "suite_token_hash_sha256": suite["suite_token_hash_sha256"],
            "expected_context_count": len(analysis_contexts),
            "expected_context_indices_sha256": analysis_indices_sha256,
            "complete": False,
            "server_models": models_response,
            "contexts": [],
        }
    identity = {
        "format_version": 2,
        "kind": "Kimi-K3 layer-92 paired expert-input and suffix capture",
        "partition": "analysis",
        "row_offset": args.row_offset,
        "row_stride": args.row_stride,
        "expert_input_dimension": args.expert_input_dimension,
        "experts_per_token": args.experts_per_token,
        "latent_dimension": args.latent_dimension,
        "hidden_dimension": args.hidden_dimension,
        "num_experts": args.num_experts,
        "suite_token_hash_sha256": suite["suite_token_hash_sha256"],
        "expected_context_count": len(analysis_contexts),
        "expected_context_indices_sha256": analysis_indices_sha256,
    }
    for key, expected in identity.items():
        if key not in manifest:
            manifest[key] = expected
        elif manifest[key] != expected:
            raise RuntimeError(f"capture manifest disagrees on {key}")
    completed = {int(item["context_index"]): item for item in manifest["contexts"]}

    for context in contexts:
        index = int(context["context_index"])
        token_path = args.suite_dir / context["token_file"]
        tokens, token_hash = _load_tokens(token_path)
        if token_hash != context["token_ids_json_sha256"]:
            raise RuntimeError(f"token hash mismatch for context {index}")
        output_path = args.output_dir / f"suffix_{index:04d}.safetensors"
        if index in completed:
            _validate_output(
                output_path,
                token_hash=token_hash,
                expected_rows=expected_rows,
                expert_input_dimension=args.expert_input_dimension,
                experts_per_token=args.experts_per_token,
                latent_dimension=args.latent_dimension,
                hidden_dimension=args.hidden_dimension,
                num_experts=args.num_experts,
            )
            if _sha256(output_path) != completed[index]["sha256"]:
                raise RuntimeError(f"capture hash mismatch for context {index}")
            print(f"context {index:04d}: already captured", flush=True)
            continue

        before = {item.name for item in args.raw_capture_dir.iterdir() if item.is_dir()}
        started = time.monotonic()
        response = _http_json(
            args.url,
            payload={
                "ignore_eos": True,
                "max_tokens": 1,
                "model": args.model,
                "prompt": tokens,
                "seed": 1,
                "temperature": 0,
            },
            timeout=args.timeout,
        )
        elapsed = time.monotonic() - started
        after = {item.name for item in args.raw_capture_dir.iterdir() if item.is_dir()}
        created = sorted(after - before)
        if len(created) != 1:
            raise RuntimeError(
                f"expected one suffix capture directory for context {index}; "
                f"got {created}"
            )
        request_dir = args.raw_capture_dir / created[0]
        tensors = _load_chunks(
            request_dir,
            scored_rows=scored_rows,
            row_stride=args.row_stride,
            row_offset=args.row_offset,
            expert_input_dimension=args.expert_input_dimension,
            experts_per_token=args.experts_per_token,
            latent_dimension=args.latent_dimension,
            hidden_dimension=args.hidden_dimension,
            num_experts=args.num_experts,
        )
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        save_file(
            tensors,
            temporary,
            metadata={
                "context_index": str(index),
                "partition": "analysis",
                "semantic_point": "kimi_k3_layer_92_common_suffix",
                "format_version": "2",
                "route_weight_semantics": "applied_moe_weight",
                "token_ids_json_sha256": token_hash,
                "row_stride": str(args.row_stride),
                "row_offset": str(args.row_offset),
                "expert_input_dimension": str(args.expert_input_dimension),
                "experts_per_token": str(args.experts_per_token),
                "latent_dimension": str(args.latent_dimension),
                "hidden_dimension": str(args.hidden_dimension),
                "num_experts": str(args.num_experts),
            },
        )
        os.replace(temporary, output_path)
        record = {
            "context_index": index,
            "elapsed_seconds": elapsed,
            "file": output_path.name,
            "request_id": response.get("id"),
            "rows": int(tensors["row_index"].numel()),
            "sha256": _sha256(output_path),
            "size_bytes": output_path.stat().st_size,
            "token_ids_json_sha256": token_hash,
        }
        manifest["contexts"].append(record)
        manifest["contexts"].sort(key=lambda item: int(item["context_index"]))
        manifest["total_size_bytes"] = sum(
            int(item["size_bytes"]) for item in manifest["contexts"]
        )
        _write_json(manifest_path, manifest)
        completed[index] = record
        if args.delete_raw_chunks_after_finalize:
            shutil.rmtree(request_dir)
        print(json.dumps(record, sort_keys=True), flush=True)

    completed_indices = sorted(
        int(item["context_index"]) for item in manifest["contexts"]
    )
    manifest["complete"] = completed_indices == analysis_indices
    if manifest["complete"]:
        manifest.setdefault("completed_utc", datetime.now(UTC).isoformat())
    else:
        manifest.pop("completed_utc", None)
    _write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()

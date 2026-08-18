#!/usr/bin/env python3
"""Validate a finalized all-row routed-MoE calibration capture."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any

import torch

from qsrt.all_row_capture import load_all_row_capture


_PLAN_FIELDS = (
    "kind",
    "schema_version",
    "expected_capture_source",
    "model_dir",
    "capture_dir",
    "sources",
    "fold",
    "excluded_corpus_reports",
    "excluded_token_files",
    "excluded_document_hashes",
    "excluded_prompt_hashes",
    "seed",
    "target_tokens",
    "planned_tokens",
    "planned_requests",
    "documents",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} does not contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _plan_sha256(report: dict[str, Any]) -> str:
    immutable = {key: report[key] for key in _PLAN_FIELDS}
    return hashlib.sha256(
        json.dumps(immutable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _document_id(document_hash: str) -> int:
    return int.from_bytes(bytes.fromhex(document_hash)[:8], "little", signed=True)


def _expected_identity(report: dict[str, Any]) -> dict[str, torch.Tensor]:
    request: list[torch.Tensor] = []
    document: list[torch.Tensor] = []
    offset: list[torch.Tensor] = []
    for index, item in enumerate(report["documents"]):
        rows = int(item["tokens"])
        request.append(torch.full((rows,), index, dtype=torch.int64))
        document.append(
            torch.full(
                (rows,),
                _document_id(str(item["document_hash"])),
                dtype=torch.int64,
            )
        )
        offset.append(torch.arange(rows, dtype=torch.int32))
    return {
        "request_index": torch.cat(request),
        "document_id": torch.cat(document),
        "token_offset": torch.cat(offset),
        "role": torch.zeros(sum(value.numel() for value in request), dtype=torch.uint8),
    }


class Proof:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, condition: bool, name: str, **evidence: Any) -> None:
        passed = bool(condition)
        self.checks.append({"name": name, "passed": passed, **evidence})
        if not passed:
            raise AssertionError(f"all-row capture proof failed: {name}: {evidence}")


_WORKER_EXPECTED: dict[str, torch.Tensor] | None = None


def _verify_chunk_worker(chunk: Any) -> dict[str, Any]:
    if _WORKER_EXPECTED is None:
        raise RuntimeError("capture-proof worker lacks expected row identities")
    return _verify_chunk(chunk, _WORKER_EXPECTED)


def _verify_chunk(
    chunk: Any,
    expected: dict[str, torch.Tensor],
) -> dict[str, Any]:
    tensors = chunk.load(verify_checksum=True)
    begin = int(chunk.row_begin)
    end = int(chunk.row_end)
    for key, expected_value in expected.items():
        if not torch.equal(tensors[key], expected_value[begin:end]):
            raise AssertionError(
                f"layer {chunk.layer} chunk {chunk.index} has incorrect {key} rows"
            )
    inputs = tensors["input"].float()
    outputs = tensors["routed_output"].float()
    if not bool(torch.isfinite(inputs).all()) or not bool(torch.isfinite(outputs).all()):
        raise AssertionError(
            f"layer {chunk.layer} chunk {chunk.index} contains non-finite activations"
        )
    ids = tensors["expert_indices"]
    if not bool(torch.all((ids >= 0) & (ids < 896))):
        raise AssertionError(
            f"layer {chunk.layer} chunk {chunk.index} has invalid expert indices"
        )
    if not bool(torch.all(torch.sort(ids, dim=1).values.diff(dim=1) != 0)):
        raise AssertionError(
            f"layer {chunk.layer} chunk {chunk.index} repeats a routed expert"
        )
    weights = tensors["route_weights"]
    sums = weights.sum(dim=1)
    if not (
        bool(torch.isfinite(weights).all())
        and bool(torch.all(weights >= 0))
        and bool(torch.all(sums > 0))
    ):
        raise AssertionError(
            f"layer {chunk.layer} chunk {chunk.index} has invalid applied gates"
        )
    return {
        "layer": int(chunk.layer),
        "input_squared_norm": float(inputs.square().sum()),
        "routed_output_squared_norm": float(outputs.square().sum()),
        "route_weight_minimum": float(weights.min()),
        "route_weight_maximum": float(weights.max()),
        "route_sum_minimum": float(sums.min()),
        "route_sum_maximum": float(sums.max()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    capture = args.capture.resolve()
    report_path = args.corpus_report.resolve()
    report = _read_json(report_path)
    proof = Proof()
    plan_sha256 = _plan_sha256(report)
    proof.check(plan_sha256 == report.get("plan_sha256"), "immutable request plan hash")
    proof.check(
        report.get("finalized") is True
        and report.get("completed_requests") == report.get("planned_requests"),
        "corpus requests finalized",
        completed=report.get("completed_requests"),
        planned=report.get("planned_requests"),
    )
    proof.check(
        report.get("planned_tokens")
        == report.get("reported_prompt_tokens")
        == report.get("target_tokens"),
        "corpus token accounting",
        tokens=report.get("planned_tokens"),
    )
    documents = report.get("documents")
    proof.check(isinstance(documents, list) and bool(documents), "document inventory")
    assert isinstance(documents, list)
    document_hashes = [str(item["document_hash"]) for item in documents]
    prompt_hashes = [str(item["prompt_hash"]) for item in documents]
    proof.check(
        len(document_hashes) == len(set(document_hashes)),
        "unique source documents",
        documents=len(documents),
    )
    proof.check(
        len(prompt_hashes) == len(set(prompt_hashes)),
        "unique tokenized prompts",
        prompts=len(prompt_hashes),
    )

    manifest, geometry, chunks = load_all_row_capture(capture, verify_hashes=False)
    rows = int(manifest["rows"])
    proof.check(
        manifest.get("corpus_manifest_sha256") == plan_sha256,
        "capture binds immutable request plan",
    )
    proof.check(
        Path(str(manifest.get("corpus"))).resolve() == report_path
        and Path(str(report.get("capture_dir"))).resolve() == capture,
        "capture and corpus paths are reciprocal",
    )
    proof.check(
        rows == int(report["planned_tokens"]),
        "all prompt rows persisted",
        rows=rows,
    )
    proof.check(
        manifest.get("route_weight_convention")
        == "applied_gate; squared_once_in_sse",
        "route-weight convention",
    )
    world = int(manifest["tp_world_size"])
    rank_receipts = manifest.get("rank_receipts", {})
    proof.check(
        sorted(int(rank) for rank in rank_receipts) == list(range(world)),
        "TP rank receipt coverage",
        world=world,
    )
    for rank in range(world):
        path = capture / f"rank-{rank:05d}.json"
        proof.check(
            path.is_file() and _sha256(path) == rank_receipts[str(rank)],
            f"TP rank {rank} receipt hash",
        )

    expected = _expected_identity(report)
    encoded_identity = b"".join(
        expected[key].contiguous().view(torch.uint8).numpy().tobytes()
        for key in expected
    )
    identity_digest = hashlib.sha256(encoded_identity).hexdigest()
    accumulators = {
        layer: {
            "input_squared_norm": 0.0,
            "routed_output_squared_norm": 0.0,
            "route_weight_minimum": float("inf"),
            "route_weight_maximum": float("-inf"),
            "route_sum_minimum": float("inf"),
            "route_sum_maximum": float("-inf"),
        }
        for layer in geometry.layers
    }
    flat_chunks = [chunk for layer in geometry.layers for chunk in chunks[layer]]
    torch.set_num_threads(1)
    global _WORKER_EXPECTED
    _WORKER_EXPECTED = expected
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        futures = [
            executor.submit(_verify_chunk_worker, chunk) for chunk in flat_chunks
        ]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            result = future.result()
            accumulator = accumulators[int(result["layer"])]
            accumulator["input_squared_norm"] += result["input_squared_norm"]
            accumulator["routed_output_squared_norm"] += result[
                "routed_output_squared_norm"
            ]
            for field in ("route_weight_minimum", "route_sum_minimum"):
                accumulator[field] = min(accumulator[field], result[field])
            for field in ("route_weight_maximum", "route_sum_maximum"):
                accumulator[field] = max(accumulator[field], result[field])
            if completed % 256 == 0 or completed == len(flat_chunks):
                print(
                    json.dumps(
                        {
                            "verified_chunks": completed,
                            "total_chunks": len(flat_chunks),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    summaries: list[dict[str, Any]] = []
    for layer in geometry.layers:
        accumulator = accumulators[layer]
        proof.check(True, f"layer {layer} exhaustive chunk checks")
        proof.check(
            True,
            f"layer {layer} row identity alignment",
            sha256=identity_digest,
        )
        proof.check(
            accumulator["input_squared_norm"] > 0
            and accumulator["routed_output_squared_norm"] > 0,
            f"layer {layer} nonzero activations",
        )
        summaries.append(
            {
                "layer": layer,
                "rows": rows,
                "chunks": len(chunks[layer]),
                **accumulator,
            }
        )

    result = {
        "kind": "qsrt_all_routed_rows_proof",
        "schema_version": 1,
        "capture": str(capture),
        "corpus_report": str(report_path),
        "plan_sha256": plan_sha256,
        "rows": rows,
        "layers": list(geometry.layers),
        "checks": proof.checks,
        "layer_summaries": summaries,
        "passed": all(item["passed"] for item in proof.checks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--corpus-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("workers must be positive")
    return args


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"passed": result["passed"], "rows": result["rows"]}))


if __name__ == "__main__":
    main()

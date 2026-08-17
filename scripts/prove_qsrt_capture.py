#!/usr/bin/env python3
"""Produce a read-only proof receipt for a finalized QSRT capture.

This program independently replays the saved-data contracts that feed dense-H
BlockLDLQ.  It deliberately does not claim that hook placement is correct;
that claim requires the trace-linked GPU witness validated by
``validate_qsrt_capture_witness.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from qsrt.blockldlq_proof import (
    capture_sample_selected,
    capture_validation_split,
    explicit_weighted_output_error,
    quadratic_error,
    relative_error,
)
from qsrt.capture import load_capture


DEFAULT_CAPTURE = Path(
    "/data/kquant/captures/k3-denseh-broad-v6-1m-train.kqcapture"
)
DEFAULT_CACHE = Path(
    "/data/kquant/captures/k3-denseh-broad-v6-1m-train-input-v1.kqsamples"
)
DEFAULT_HESSIANS = Path(
    "/data/kquant/hessians/k3-denseh-broad-v6-1m-train-h13-identity-v1.kqhess"
)
DEFAULT_REPORT = Path("out/k3-denseh-broad-v6-1m-train-corpus.json")


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


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


class Receipt:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, condition: bool, name: str, **evidence: Any) -> None:
        passed = bool(condition)
        self.checks.append({"name": name, "passed": passed, **evidence})
        if not passed:
            raise AssertionError(f"capture proof failed: {name}: {evidence}")


def _validate_ranges(
    receipt: Receipt,
    ranks: list,
    *,
    attribute: str,
    expected_end: int,
) -> None:
    intervals = [getattr(rank, attribute) for rank in ranks]
    cursor = 0
    for begin, end in intervals:
        receipt.check(
            begin == cursor and end >= begin,
            f"{attribute} contiguous at {cursor}",
            interval=[begin, end],
        )
        cursor = end
    receipt.check(cursor == expected_end, f"{attribute} complete", end=cursor)


def _validate_corpus(
    receipt: Receipt,
    report: dict[str, Any],
    root: dict[str, Any],
    report_path: Path,
    capture_path: Path,
) -> None:
    documents = report.get("documents")
    receipt.check(isinstance(documents, list) and bool(documents), "corpus documents")
    assert isinstance(documents, list)
    document_hashes = [row["document_hash"] for row in documents]
    prompt_hashes = [row["prompt_hash"] for row in documents]
    document_tokens = sum(int(row["tokens"]) for row in documents)
    receipt.check(
        len(document_hashes) == len(set(document_hashes)),
        "unique source documents",
        documents=len(document_hashes),
    )
    receipt.check(
        len(prompt_hashes) == len(set(prompt_hashes)),
        "unique tokenized prompts",
        prompts=len(prompt_hashes),
    )
    receipt.check(
        report.get("finalized") is True
        and report.get("completed_requests") == report.get("planned_requests"),
        "corpus finalized after all requests",
        completed=report.get("completed_requests"),
        planned=report.get("planned_requests"),
    )
    receipt.check(
        document_tokens
        == report.get("planned_tokens")
        == report.get("reported_prompt_tokens")
        == report.get("target_tokens"),
        "corpus token accounting",
        document_tokens=document_tokens,
        executed_tokens=root.get("executed_tokens"),
    )
    receipt.check(
        Path(root["corpus"]).resolve() == report_path.resolve()
        and Path(report["capture_dir"]).resolve() == capture_path.resolve(),
        "capture/corpus path binding",
        corpus=root.get("corpus"),
        capture=report.get("capture_dir"),
    )
    receipt.check(
        report.get("capture_manifest_after_run") == root,
        "final report embeds exact capture manifest",
    )
    executed = int(root.get("executed_tokens", -1))
    prompt = int(report["reported_prompt_tokens"])
    completion = int(report.get("reported_completion_tokens", 0))
    # The capture is finalized by a sentinel observed at a model-execution
    # boundary.  Depending on scheduler batching, zero through all generated
    # one-token completions can run before that boundary; all prompt tokens
    # must be present and no unreported model token may be present.
    receipt.check(
        prompt <= executed <= prompt + completion,
        "executed token accounting",
        executed=executed,
        prompt=prompt,
        completion=completion,
    )


def _selected_mask(
    observations: torch.Tensor,
    splits: torch.Tensor,
    hessian_manifest: dict[str, Any],
) -> torch.Tensor:
    split_name = hessian_manifest["sample_split"]
    split_value = {"train": 0, "validation": 1, "all": None}[split_name]
    steps = torch.bitwise_right_shift(observations.to(torch.int64), 32)
    request_range = hessian_manifest["request_step_range"]
    mask = steps >= int(request_range["minimum_inclusive"])
    maximum = request_range.get("maximum_inclusive")
    if maximum is not None:
        mask &= steps <= int(maximum)
    request_filter = hessian_manifest.get("request_step_filter")
    if request_filter is not None:
        mask &= torch.isin(steps, torch.tensor(request_filter, dtype=steps.dtype))
    if split_value is not None:
        mask &= splits.to(torch.int8) == split_value
    return mask


def _prove_layer(
    receipt: Receipt,
    *,
    layer: int,
    cache_path: Path,
    cache_manifest: dict[str, Any],
    hessian_path: Path,
    hessian_manifest: dict[str, Any],
    sampling: dict[str, Any],
    subspace_size: int,
    seed: int,
) -> dict[str, Any]:
    cache_entry = cache_manifest["layers"][str(layer)]
    tensors = load_file(str(cache_path / cache_entry["file"]), device="cpu")
    values = tensors["input.values"]
    weights = tensors["input.weight"].float()
    observations = tensors["input.observation"].to(torch.int64)
    experts = tensors["input.experts"]
    gates = tensors["input.gates"].float()
    splits = tensors["input.split"].to(torch.int8)
    routed_latent = tensors["input.routed_latent"]

    rows, width = values.shape
    receipt.check(rows == int(cache_entry["rows"]), f"layer {layer} cache rows")
    receipt.check(
        rows == observations.numel() == weights.numel() == splits.numel(),
        f"layer {layer} aligned scalar metadata",
    )
    receipt.check(
        experts.shape == gates.shape and experts.shape[0] == rows,
        f"layer {layer} aligned routes",
        route_shape=list(experts.shape),
    )
    receipt.check(
        tuple(routed_latent.shape) == (rows, width),
        f"layer {layer} aligned routed latent",
    )
    receipt.check(
        torch.isfinite(values).all()
        and torch.isfinite(gates).all()
        and torch.isfinite(routed_latent).all(),
        f"layer {layer} finite tensors",
    )
    expected_weight = gates.square().sum(dim=1)
    gate_weight_error = float((weights - expected_weight).abs().max())
    receipt.check(
        torch.allclose(weights, expected_weight, rtol=2e-6, atol=2e-7),
        f"layer {layer} gate-square weights",
        max_abs=gate_weight_error,
    )
    receipt.check(
        torch.unique(observations).numel() == rows,
        f"layer {layer} unique observations",
    )

    sample_rate = int(sampling["input_hessian"])
    modulus = int(sampling["validation_modulus"])
    expected_splits = torch.tensor(
        [capture_validation_split(int(value), modulus) for value in observations],
        dtype=torch.int8,
    )
    receipt.check(
        torch.equal(splits, expected_splits),
        f"layer {layer} deterministic split replay",
        validation_rows=int((splits == 1).sum()),
    )
    receipt.check(
        all(capture_sample_selected(int(value), sample_rate) for value in observations),
        f"layer {layer} deterministic sample replay",
        sample_rate=sample_rate,
    )

    hessian_entry = hessian_manifest["layers"][str(layer)]
    hessian = load_file(
        str(hessian_path / hessian_entry["file"]), device="cpu"
    )["w13"].float()
    mask = _selected_mask(observations, splits, hessian_manifest)
    selected_values = values[mask].float()
    selected_weights = weights[mask]
    receipt.check(
        int(mask.sum()) == int(hessian_entry["w13_rows"]),
        f"layer {layer} Hessian row selection",
        selected=int(mask.sum()),
    )
    weight_sum = float(selected_weights.double().sum())
    receipt.check(
        math.isclose(
            weight_sum,
            float(hessian_entry["w13_weight_sum"]),
            rel_tol=2e-12,
            abs_tol=2e-10,
        ),
        f"layer {layer} Hessian weight sum",
        actual=weight_sum,
        stored=hessian_entry["w13_weight_sum"],
    )
    receipt.check(
        hessian.shape == (width, width)
        and torch.equal(hessian, hessian.T)
        and torch.isfinite(hessian).all(),
        f"layer {layer} stored Hessian structure",
        shape=list(hessian.shape),
    )

    generator = torch.Generator().manual_seed(seed + layer)
    error = torch.randn((4, width), generator=generator, dtype=torch.float32)
    proxy = quadratic_error(error, hessian)
    direct = explicit_weighted_output_error(
        selected_values, selected_weights, error
    )
    sse_relative = relative_error(proxy, direct)
    receipt.check(
        sse_relative < 3e-6,
        f"layer {layer} Hessian equals direct routed-input SSE",
        relative_error=sse_relative,
    )

    subspace_size = min(int(subspace_size), width)
    columns = torch.randperm(width, generator=generator)[:subspace_size].sort().values
    sub_rows = selected_values.index_select(1, columns).double()
    sub_weights = selected_weights.double()
    recomputed = sub_rows.T @ (sub_rows * sub_weights[:, None]) / sub_weights.sum()
    stored_subspace = hessian.index_select(0, columns).index_select(1, columns).double()
    covariance_relative = relative_error(stored_subspace, recomputed)
    receipt.check(
        covariance_relative < 3e-6,
        f"layer {layer} Hessian subspace reproduction",
        relative_error=covariance_relative,
        subspace_size=subspace_size,
    )
    probes = torch.randn((8, width), generator=generator, dtype=torch.float64)
    rayleigh = torch.einsum("bi,ij,bj->b", probes, hessian.double(), probes)
    receipt.check(
        bool(torch.all(rayleigh >= -1e-8 * torch.diagonal(hessian.double()).sum())),
        f"layer {layer} Hessian sampled PSD",
        minimum=float(rayleigh.min()),
    )
    return {
        "layer": layer,
        "rows": rows,
        "training_rows": int(mask.sum()),
        "validation_rows": int((splits == 1).sum()),
        "weight_sum": weight_sum,
        "direct_sse_relative_error": sse_relative,
        "subspace_covariance_relative_error": covariance_relative,
        "hessian_sha256": _tensor_sha256(hessian),
    }


def _prove_raw_cache(
    receipt: Receipt,
    *,
    ranks: list,
    layers: list[int],
    cache_path: Path,
    cache_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Stream rank-zero parts and prove that the cache is a lossless repack."""

    rank_zero = next(rank for rank in ranks if rank.rank == 0)
    keys = (
        "input.values",
        "input.weight",
        "input.observation",
        "input.experts",
        "input.gates",
        "input.split",
        "input.routed_latent",
    )
    cached = {
        layer: load_file(
            str(cache_path / cache_manifest["layers"][str(layer)]["file"]),
            device="cpu",
        )
        for layer in layers
    }
    seen = {
        layer: torch.zeros(cached[layer]["input.observation"].numel(), dtype=torch.bool)
        for layer in layers
    }
    selected_rows = {layer - 1: layer for layer in layers}
    for part in rank_zero.sample_parts():
        with safe_open(str(part), framework="pt", device="cpu") as handle:
            part_layers = handle.get_tensor("input.layer").to(torch.int64)
            present = set(int(value) for value in torch.unique(part_layers).tolist())
            wanted = present.intersection(selected_rows)
            if not wanted:
                continue
            part_tensors = {key: handle.get_tensor(key) for key in keys}
        for raw_layer in sorted(wanted):
            layer = selected_rows[raw_layer]
            indices = torch.nonzero(part_layers == raw_layer, as_tuple=False).flatten()
            actual_observations = part_tensors["input.observation"].index_select(
                0, indices
            )
            cached_observations = cached[layer]["input.observation"]
            positions = torch.searchsorted(cached_observations, actual_observations)
            in_bounds = positions < cached_observations.numel()
            if not bool(torch.all(in_bounds)):
                receipt.check(False, f"layer {layer} raw/cache observation bounds")
            matched = cached_observations.index_select(0, positions)
            if not torch.equal(matched, actual_observations):
                receipt.check(False, f"layer {layer} raw/cache observation identity")
            if bool(torch.any(seen[layer].index_select(0, positions))):
                receipt.check(False, f"layer {layer} raw/cache duplicate observation")
            for key in keys:
                expected = cached[layer][key].index_select(0, positions)
                actual = part_tensors[key].index_select(0, indices)
                if not torch.equal(actual, expected):
                    receipt.check(
                        False,
                        f"layer {layer} raw/cache {key}",
                        part=part.name,
                    )
            seen[layer].index_fill_(0, positions, True)
    for layer in layers:
        receipt.check(
            bool(torch.all(seen[layer])),
            f"layer {layer} raw/cache complete",
            rows=int(seen[layer].sum()),
        )
    return {
        "rank_zero_parts": len(rank_zero.sample_parts()),
        "rows": {layer: int(mask.sum()) for layer, mask in seen.items()},
        "matching": "observation-keyed byte equality for every tensor row",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--sample-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--hessians", type=Path, default=DEFAULT_HESSIANS)
    parser.add_argument("--corpus-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--layers", type=_parse_layers, default=[1, 24, 92])
    parser.add_argument("--subspace-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--verify-raw-cache", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("out/qsrt-capture-proof.json")
    )
    args = parser.parse_args()

    receipt = Receipt()
    root, geometry, ranks = load_capture(args.capture, load_stats=False)
    cache_manifest = _json(args.sample_cache / "manifest.json")
    hessian_manifest = _json(args.hessians / "manifest.json")
    corpus_report = _json(args.corpus_report)

    receipt.check(root.get("complete") is True, "capture finalized")
    receipt.check(
        all(int(rank.manifest.get("input_samples_dropped", -1)) == 0 for rank in ranks),
        "no input sample drops",
    )
    receipt.check(
        all(int(rank.manifest.get("mid_samples_dropped", -1)) == 0 for rank in ranks),
        "no middle sample drops",
    )
    receipt.check(
        all(
            len(rank.sample_parts()) == int(rank.manifest["sample_parts"])
            for rank in ranks
        ),
        "sample part counts",
    )
    _validate_ranges(
        receipt,
        ranks,
        attribute="input_expert_range",
        expected_end=geometry.num_experts,
    )
    _validate_ranges(
        receipt,
        ranks,
        attribute="intermediate_channel_range",
        expected_end=geometry.intermediate_size,
    )
    for manifest, name in (
        (cache_manifest, "sample cache"),
        (hessian_manifest, "Hessian bundle"),
    ):
        receipt.check(manifest.get("run_id") == root.get("run_id"), f"{name} run ID")
        receipt.check(
            Path(manifest["source_capture"]).resolve() == args.capture.resolve(),
            f"{name} source capture",
        )
        receipt.check(
            manifest.get("model") == root.get("model")
            and manifest.get("revision") == root.get("revision"),
            f"{name} model identity",
        )
    receipt.check(
        hessian_manifest.get("h2_policy") == "identity_prior_expert_local_jit",
        "H2 is expert-local JIT with identity fallback",
    )
    _validate_corpus(
        receipt, corpus_report, root, args.corpus_report, args.capture
    )

    layer_results = [
        _prove_layer(
            receipt,
            layer=layer,
            cache_path=args.sample_cache,
            cache_manifest=cache_manifest,
            hessian_path=args.hessians,
            hessian_manifest=hessian_manifest,
            sampling=root["sampling"],
            subspace_size=args.subspace_size,
            seed=args.seed,
        )
        for layer in args.layers
    ]
    raw_cache = None
    if args.verify_raw_cache:
        raw_cache = _prove_raw_cache(
            receipt,
            ranks=ranks,
            layers=args.layers,
            cache_path=args.sample_cache,
            cache_manifest=cache_manifest,
        )

    payload = {
        "schema_version": 1,
        "kind": "qsrt_capture_dense_h_proof",
        "passed": all(check["passed"] for check in receipt.checks),
        "scope": {
            "proved": [
                "finalized capture structure and zero drops",
                "TP ownership intervals",
                "corpus request/token/document accounting",
                "deterministic row sampling and split labels",
                "gate-square weighting",
                "sample-cache/Hessian provenance",
                "stored H13 equals direct weighted activation SSE",
                "raw capture to sample-cache equality when requested",
            ],
            "requires_gpu_witness": [
                "hook tensor equals model MoE input",
                "captured routes and gates equal the actual router output",
                "captured routed latent is post-TP-all-reduce and pre-RMSNorm",
                "TP middle-channel join if teacher-middle capture is reused",
            ],
            "statistical_note": (
                "the in-capture 15/16 versus 1/16 split is token-observation "
                "disjoint, not document-disjoint; final selection and validation "
                "must use separate corpus reports/captures"
            ),
        },
        "inputs": {
            "capture": str(args.capture.resolve()),
            "sample_cache": str(args.sample_cache.resolve()),
            "hessians": str(args.hessians.resolve()),
            "corpus_report": str(args.corpus_report.resolve()),
        },
        "layers": layer_results,
        "raw_cache": raw_cache,
        "checks": receipt.checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "passed": payload["passed"],
        "checks": len(receipt.checks),
        "layers": layer_results,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()

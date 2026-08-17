#!/usr/bin/env python3
"""Build a controlled fixed-count X4T allocation from the L0 router proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from qsrt import constants as C
from qsrt.pack.qsrt_allocation import (
    make_qsrt_fixed_allocation,
    qsrt_allocation_document,
    write_qsrt_allocation,
)
from qsrt.pack.qsrt_pool import load_qsrt_candidate_pool
from qsrt.pack.qsrt_validation import (
    VALIDATION_DAMAGE_METRIC,
    VALIDATION_DAMAGE_WEIGHTING,
    load_qsrt_validation_scores,
)
from qsrt.pack.x4t_index import load_x4t_cost_index


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--x4t-cost-index", type=Path, required=True)
    parser.add_argument("--static-stats", type=Path, required=True)
    parser.add_argument("--x4t-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-scores", type=Path)
    parser.add_argument("--skip-payload-header-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    total = C.NUM_MOE_LAYERS * C.NUM_EXPERTS
    if not 0 <= args.x4t_count <= total:
        raise ValueError(f"X4T count must lie in 0..{total}")
    pool = load_qsrt_candidate_pool(
        args.candidate_pool,
        validate_payload_headers=not args.skip_payload_header_validation,
    )
    x4t_index = load_x4t_cost_index(args.x4t_cost_index)
    if args.validation_scores is not None:
        validation = load_qsrt_validation_scores(args.validation_scores, pool)
        pool = replace(
            pool,
            damage=validation.damage,
            damage_metric=VALIDATION_DAMAGE_METRIC,
            damage_weighting=VALIDATION_DAMAGE_WEIGHTING,
            damage_provenance={
                "validation_scores": str(validation.root),
                "validation_score_set_sha256": validation.content_sha256,
                "validation_capture": validation.manifest["validation_capture"],
                "validation_report": validation.manifest["validation_report"],
                "validation_documents": validation.manifest[
                    "validation_documents"
                ],
                "selection_data_used": False,
            },
        )

    stats_root = args.static_stats.resolve()
    manifest_path = stats_root / "manifest.json"
    arrays_path = stats_root / "arrays.safetensors"
    manifest = json.loads(manifest_path.read_text())
    arrays = load_file(str(arrays_path))
    bias = np.asarray(arrays["router_bias"], dtype=np.float64)
    expected_shape = (C.NUM_MOE_LAYERS, C.NUM_EXPERTS)
    if bias.shape != expected_shape:
        raise ValueError(f"router bias has shape {bias.shape}, expected {expected_shape}")
    mean = np.nanmean(bias, axis=1, keepdims=True)
    std = np.nanstd(bias, axis=1, keepdims=True)
    zscore = (bias - mean) / np.where(std > 0, std, 1.0)
    proxy = np.exp(-zscore)
    proxy /= np.nanmean(proxy, axis=1, keepdims=True)
    proxy = np.nan_to_num(proxy, nan=1.0)
    mask = np.zeros(total, dtype=np.bool_)
    if args.x4t_count:
        selected = np.argpartition(proxy.reshape(-1), -args.x4t_count)[
            -args.x4t_count:
        ]
        mask[selected] = True
    mask = mask.reshape(expected_shape)

    allocation = make_qsrt_fixed_allocation(
        pool.damage,
        x4t_index.expert_storage_bytes,
        mask,
    )
    provenance = {
        "policy": "l0_router_bias_fixed_count",
        "requested_x4t_experts": args.x4t_count,
        "ranking": (
            "global top count of per-layer exp(-zscore(router_bias)), "
            "normalized to layer mean one; numpy.argpartition"
        ),
        "static_stats": str(stats_root),
        "static_stats_manifest_sha256": _sha256(manifest_path),
        "static_stats_arrays_sha256": _sha256(arrays_path),
        "static_stats_model": manifest.get("model"),
        "static_stats_revision": manifest.get("revision"),
    }
    document = qsrt_allocation_document(
        pool,
        x4t_index,
        allocation,
        fixed_selection_provenance=provenance,
    )
    write_qsrt_allocation(args.output, document)
    print(json.dumps(document["meta"], indent=1, sort_keys=True))


if __name__ == "__main__":
    main()

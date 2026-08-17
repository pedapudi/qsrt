#!/usr/bin/env python3
"""Merge disjoint GLM-5.2 codec worker results after identity validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from qsrt.correctness import sha256_file
from qsrt.glm52_pilot import (
    PROJECTIONS,
    _canonical_json_sha256,
    aggregate_uniform_rate_records,
    atomic_write_json,
)
from qsrt.glm52_real_weight_benchmark import (
    REAL_WEIGHT_CODEC_BENCHMARK_KIND,
    load_frozen_real_weight_panel,
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _selected_fields(value: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return {field: value.get(field) for field in fields}


def merge_results(
    *,
    input_roots: list[Path],
    frozen_panel_path: Path,
    preflight_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate and merge a complete frozen panel from disjoint worker roots."""

    frozen = load_frozen_real_weight_panel(frozen_panel_path, layer=3)
    preflight = _read_object(preflight_path)
    expected_preflight_panel = {
        "path": frozen["path"],
        "sha256": frozen["sha256"],
        "experts": list(frozen["experts"]),
    }
    if preflight.get("frozen_panel") != expected_preflight_panel:
        raise ValueError("preflight receipt does not bind the frozen panel")
    source_preflight = preflight.get("source")
    endpoint_preflight = preflight.get("exl3_endpoint")
    if not isinstance(source_preflight, dict) or not isinstance(
        endpoint_preflight, dict
    ):
        raise TypeError("preflight receipt is missing source or endpoint metadata")
    source_shards = source_preflight.get("selected_shards")
    if not isinstance(source_shards, list) or not source_shards:
        raise ValueError("preflight receipt contains no selected source shards")
    if any(
        not isinstance(shard, dict) or shard.get("sha256_verified") is not True
        for shard in source_shards
    ):
        raise ValueError("preflight receipt has an unverified source shard")
    if endpoint_preflight.get("shard_sha256_verified") is not True:
        raise ValueError("preflight receipt has an unverified EXL3 layer payload")

    source_fields = (
        "model_id",
        "revision",
        "config_sha256",
        "index_sha256",
        "source_inventory_sha256",
    )
    endpoint_fields = (
        "model_id",
        "revision",
        "manifest_sha256",
        "manifest_json_sha256",
        "layer",
        "sidecar_sha256",
        "shard",
        "shard_sha256",
        "allocation_bpw",
    )
    expected_source = _selected_fields(source_preflight, source_fields)
    expected_endpoint = _selected_fields(endpoint_preflight, endpoint_fields)
    expected_kind = f"{REAL_WEIGHT_CODEC_BENCHMARK_KIND}_manifest"

    worker_receipts: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    rates: tuple[int, ...] | None = None
    expected_codec_arms: dict[str, Any] | None = None
    for root in input_roots:
        root = root.resolve()
        manifest_path = root / "manifest.json"
        report_path = root / "report.json"
        manifest = _read_object(manifest_path)
        report = _read_object(report_path)
        if manifest.get("kind") != expected_kind:
            raise ValueError(f"unexpected worker manifest kind in {manifest_path}")
        if report.get("kind") != REAL_WEIGHT_CODEC_BENCHMARK_KIND:
            raise ValueError(f"unexpected worker report kind in {report_path}")
        if report.get("status") != "complete":
            raise ValueError(f"worker report is incomplete: {report_path}")
        manifest_sha256 = _canonical_json_sha256(manifest)
        if report.get("manifest_sha256") != manifest_sha256:
            raise ValueError(f"worker report does not bind {manifest_path}")
        if _selected_fields(manifest.get("source", {}), source_fields) != expected_source:
            raise ValueError(f"worker source identity mismatch in {manifest_path}")
        if (
            _selected_fields(manifest.get("exl3_endpoint", {}), endpoint_fields)
            != expected_endpoint
        ):
            raise ValueError(f"worker endpoint identity mismatch in {manifest_path}")
        frozen_receipt = manifest.get("frozen_panel")
        if not isinstance(frozen_receipt, dict):
            raise TypeError(f"worker has no frozen-panel receipt: {manifest_path}")
        if frozen_receipt.get("sha256") != frozen["sha256"]:
            raise ValueError(f"worker frozen-panel hash mismatch in {manifest_path}")
        offset = frozen_receipt.get("selected_offset")
        count = frozen_receipt.get("selected_count")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or offset < 0
            or count < 1
        ):
            raise ValueError(f"worker has an invalid panel slice in {manifest_path}")
        worker_rates = tuple(int(rate) for rate in manifest.get("rates", []))
        if rates is None:
            rates = worker_rates
        elif worker_rates != rates:
            raise ValueError("workers used different codec rates")
        codec_arms = manifest.get("codec_arms")
        if not isinstance(codec_arms, dict):
            raise TypeError(f"worker has no codec-arm metadata: {manifest_path}")
        if expected_codec_arms is None:
            expected_codec_arms = codec_arms
        elif codec_arms != expected_codec_arms:
            raise ValueError("workers used different codec arms")
        raw_records = report.get("experts")
        if not isinstance(raw_records, list) or len(raw_records) != count:
            raise ValueError(f"worker record count mismatch in {report_path}")
        selected = tuple(
            int(expert)
            for experts in manifest.get("panel", {}).values()
            for expert in experts
        )
        if selected != frozen["experts"][offset : offset + count]:
            raise ValueError(f"worker panel slice mismatch in {manifest_path}")
        if tuple(int(record.get("expert", -1)) for record in raw_records) != selected:
            raise ValueError(f"worker expert records are out of order in {report_path}")
        records.extend(raw_records)
        worker_receipts.append(
            {
                "root": str(root),
                "offset": offset,
                "count": count,
                "experts": list(selected),
                "manifest_sha256": sha256_file(manifest_path),
                "report_sha256": sha256_file(report_path),
            }
        )

    worker_receipts.sort(key=lambda receipt: receipt["offset"])
    expected_offset = 0
    ordered_experts: list[int] = []
    for receipt in worker_receipts:
        if receipt["offset"] != expected_offset:
            raise ValueError("worker panel slices contain a gap or overlap")
        expected_offset += receipt["count"]
        ordered_experts.extend(receipt["experts"])
    if tuple(ordered_experts) != frozen["experts"]:
        raise ValueError("worker panel slices do not cover the frozen panel")
    if not rates:
        raise ValueError("worker results contain no codec rate")

    records.sort(key=lambda record: frozen["experts"].index(int(record["expert"])))
    panel = {3: frozen["experts"]}
    report = {
        "kind": "qsrt_glm52_real_weight_codec_benchmark_merged",
        "status": "complete",
        "frozen_panel": {
            "path": frozen["path"],
            "sha256": frozen["sha256"],
            "experts": list(frozen["experts"]),
        },
        "preflight": {
            "path": str(preflight_path.resolve()),
            "sha256": sha256_file(preflight_path),
        },
        "workers": worker_receipts,
        "rates": list(rates),
        "panel": {"3": list(frozen["experts"])},
        "matrix_count": len(records) * len(PROJECTIONS),
        "aggregate": aggregate_uniform_rate_records(
            records,
            rate_labels=tuple(f"K{rate}" for rate in rates),
            panel=panel,
        ),
        "experts": records,
        "evidence_boundary": (
            "raw weight distortion on one frozen real-weight layer; this is "
            "not full-model KLD, task quality, or a population estimate"
        ),
    }
    atomic_write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--frozen-panel", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = merge_results(
        input_roots=args.inputs,
        frozen_panel_path=args.frozen_panel,
        preflight_path=args.preflight_receipt,
        output_path=args.output,
    )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

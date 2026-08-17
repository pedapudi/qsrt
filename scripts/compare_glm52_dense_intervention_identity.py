#!/usr/bin/env python3
"""Prove whether two GLM-5.2 dense intervention artifacts are byte-identical."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qsrt.glm52_expert_intervention_runtime import (
    validate_dense_intervention_artifact,
)
from qsrt.glm52_pilot import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    args = parser.parse_args()
    if args.dest.exists():
        raise FileExistsError(args.dest)
    baseline = validate_dense_intervention_artifact(args.baseline)
    candidate = validate_dense_intervention_artifact(args.candidate)
    if baseline["expert_ids"] != candidate["expert_ids"]:
        raise ValueError("dense intervention expert panels differ")
    baseline_records = {
        int(record["expert"]): record
        for record in baseline["report"]["experts"]
    }
    candidate_records = {
        int(record["expert"]): record
        for record in candidate["report"]["experts"]
    }
    comparisons = []
    for expert in baseline["expert_ids"]:
        baseline_record = baseline_records[expert]
        candidate_record = candidate_records[expert]
        same_file = (
            baseline_record["dense_endpoint_file_bytes"]
            == candidate_record["dense_endpoint_file_bytes"]
            and baseline_record["dense_endpoint_file_sha256"]
            == candidate_record["dense_endpoint_file_sha256"]
        )
        comparisons.append(
            {
                "expert": expert,
                "byte_identical": same_file,
                "baseline_dense_endpoint_sha256": baseline_record[
                    "dense_endpoint_file_sha256"
                ],
                "candidate_dense_endpoint_sha256": candidate_record[
                    "dense_endpoint_file_sha256"
                ],
                "dense_endpoint_bytes": candidate_record[
                    "dense_endpoint_file_bytes"
                ],
            }
        )
    all_identical = all(item["byte_identical"] for item in comparisons)
    report = {
        "schema": "qsrt_glm52_dense_intervention_identity_comparison",
        "schema_version": 1,
        "status": "identical" if all_identical else "different",
        "baseline_root": str(args.baseline.resolve()),
        "baseline_manifest_sha256": baseline["manifest_sha256"],
        "candidate_root": str(args.candidate.resolve()),
        "candidate_manifest_sha256": candidate["manifest_sha256"],
        "expert_count": len(comparisons),
        "dense_endpoint_bytes": candidate["dense_endpoint_bytes"],
        "all_dense_endpoint_files_byte_identical": all_identical,
        "experts": comparisons,
        "inference_consequence": (
            "The runtime reads only the dense endpoint tensors. Byte-identical "
            "files therefore produce the same expert outputs and inherit the "
            "same full-model KLD measurement under the same runtime controls."
        ),
    }
    atomic_write_json(args.dest, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not all_identical:
        raise RuntimeError("dense intervention artifacts differ")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail

# Freeze every rank-four singleton that improves both ordered halves of the
# public selection set, then screen those candidates on the existing frozen
# 2,048-token BF16 reference after the supplied predecessor releases the GPUs.

if test "$#" != 1; then
  echo "usage: $0 <predecessor-process-id>" >&2
  exit 2
fi
predecessor_pid="$1"
if [[ ! "${predecessor_pid}" =~ ^[1-9][0-9]*$ ]]; then
  echo "predecessor process ID must be a positive integer" >&2
  exit 2
fi
while kill -0 "${predecessor_pid}" 2>/dev/null; do
  sleep 30
done

root="/home/sunil/qsrt-glm52-experiments"
decision="${root}/launch-records/rank4-low-rank-down-singleton-selection-outcome.json"
test ! -e "${decision}"

mapfile -t retained < <(python3 - "${root}" "${decision}" <<'PY'
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
decision_path = Path(sys.argv[2])
records = []
retained = []
for layer in (52, 60, 64):
    result_name = f"glm52-layer{layer}-rank4-low-rank-down-singleton-model-kld-selection-public-reference"
    report_path = root / "results" / result_name / "report.json"
    plan_path = root / "registrations" / f"glm52_layer{layer}_rank4_low_rank_down_singleton_model_kld_selection_plan.json"
    artifact_name = (
        f"glm52-layer{layer}-frozen8-low-rank-down-reconstructed-activation-refit-"
        "derived-input-identity-checked-selection-fallback-bf16-rank-4-merged"
    )
    artifact_report_path = root / "results" / artifact_name / "report.json"
    report = json.loads(report_path.read_text())
    plan = json.loads(plan_path.read_text())
    artifact_report = json.loads(artifact_report_path.read_text())
    assert report["status"] == "complete"
    assert report["measurement_controls"]["passed"] is True
    assert artifact_report["manifest_sha256"] == plan["artifact_manifest_sha256"]
    assert report["intervention_artifact"]["manifest_sha256"] == plan["artifact_manifest_sha256"]
    documents = report["documents"]
    assert len(documents) == 16
    planned_arms = [arm["name"] for arm in plan["candidate_arms"]]
    assert set(planned_arms) == set(report["candidate_arms"])
    for arm in plan["candidate_arms"]:
        name = arm["name"]
        experts = arm["selected_experts"]
        assert len(experts) == 1
        deltas = [
            float(document["candidate_arms"][name]["candidate_minus_resident_mean_forward_kld"])
            for document in documents
        ]
        first_mean = statistics.fmean(deltas[:8])
        second_mean = statistics.fmean(deltas[8:])
        all_mean = statistics.fmean(deltas)
        accepted = first_mean < 0.0 and second_mean < 0.0
        record = {
            "model_layer": layer,
            "expert": experts[0],
            "candidate_arm": name,
            "artifact_directory_name": artifact_name,
            "artifact_manifest_identity": plan["artifact_manifest_sha256"],
            "first_eight_mean_candidate_minus_resident_kld": first_mean,
            "second_eight_mean_candidate_minus_resident_kld": second_mean,
            "all_document_mean_candidate_minus_resident_kld": all_mean,
            "retained": accepted,
        }
        records.append(record)
        if accepted:
            retained.append(record)
    records.append(
        {
            "model_layer": layer,
            "selection_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "selection_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "selection_report_controls_passed": True,
        }
    )

retained.sort(
    key=lambda item: (
        item["all_document_mean_candidate_minus_resident_kld"],
        item["model_layer"],
        item["expert"],
    )
)
decision = {
    "schema": "qsrt_glm52_rank4_low_rank_down_singleton_selection_outcome",
    "schema_version": 1,
    "status": "frozen_before_single_reference_measurement",
    "selection_documents": {
        "document_count": 16,
        "ordered_group_sizes": [8, 8],
        "retention_rule": "Both ordered group means must be negative.",
    },
    "candidate_records": records,
    "retained_candidates": retained,
    "evidence_boundary": (
        "The 2,048-token reference is one development screen. It cannot qualify "
        "a checkpoint or estimate document-level uncertainty."
    ),
}
temporary = decision_path.with_suffix(".partial")
temporary.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
os.replace(temporary, decision_path)
for item in retained:
    print(item["model_layer"], item["expert"])
PY
)

for item in "${retained[@]}"; do
  read -r layer expert <<<"${item}"
  artifact="glm52-layer${layer}-frozen8-low-rank-down-reconstructed-activation-refit-derived-input-identity-checked-selection-fallback-bf16-rank-4-merged"
  result="glm52-layer${layer}-expert${expert}-rank4-low-rank-down-single-reference-absolute-target-screen"
  "${root}/launch-scripts/run_glm52_frozen_expert_subset_single_reference_on_kossel.sh" \
    "${artifact}" "${result}" "${expert}"
done

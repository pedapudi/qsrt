#!/usr/bin/env bash
set -euo pipefail

# Open the independent 2,048-token reference only when the pre-registered
# expert-164 K4-down arm improves both ordered halves of the public selection
# set. The decision record is written before the independent launcher starts.

if test "$#" -ne 1; then
  echo "usage: $0 <active-process-id>" >&2
  exit 2
fi
active_pid="$1"
if [[ ! "${active_pid}" =~ ^[1-9][0-9]*$ ]]; then
  echo "active process ID must be a positive integer" >&2
  exit 2
fi
while kill -0 "${active_pid}" 2>/dev/null; do
  sleep 15
done

root="/home/sunil/qsrt-glm52-experiments"
report_name="glm52-layer63-expert164-k4-down-model-kld-selection-public-reference"
report="${root}/results/${report_name}/report.json"
decision="${root}/launch-records/layer63-expert164-k4-down-selection-gate.json"
test -f "${report}"
test ! -e "${decision}"

accepted="$({ python3 - "${report}" "${decision}" <<'PY'
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
decision_path = Path(sys.argv[2])
report = json.loads(report_path.read_text())
arm_name = "expert-164-k4-down-alone"
documents = report["documents"]
assert len(documents) == 16
deltas = [
    float(document["candidate_arms"][arm_name]["candidate_minus_resident_mean_forward_kld"])
    for document in documents
]
first_mean = statistics.fmean(deltas[:8])
second_mean = statistics.fmean(deltas[8:])
all_mean = statistics.fmean(deltas)
accepted = first_mean < 0.0 and second_mean < 0.0
digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
record = {
    "schema": "qsrt_glm52_two_group_model_kld_selection_decision",
    "schema_version": 1,
    "status": "accepted" if accepted else "rejected",
    "candidate_arm": arm_name,
    "selection_report_sha256": digest,
    "document_count": 16,
    "first_eight_mean_candidate_minus_resident_kld": first_mean,
    "second_eight_mean_candidate_minus_resident_kld": second_mean,
    "all_document_mean_candidate_minus_resident_kld": all_mean,
    "acceptance_rule": "both ordered eight-document means must be negative",
}
temporary = decision_path.with_suffix(".partial")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
os.replace(temporary, decision_path)
print("yes" if accepted else "no")
PY
} )"

if test "${accepted}" != yes; then
  exit 0
fi

"${root}/launch-scripts/run_glm52_frozen_expert_subset_single_reference_on_kossel.sh" \
  glm52-layer63-experts149-164-registered-k3-k3-k4-down-refit-v2-input-ancestry \
  glm52-layer63-expert164-k4-down-single-reference-absolute-target-screen \
  164

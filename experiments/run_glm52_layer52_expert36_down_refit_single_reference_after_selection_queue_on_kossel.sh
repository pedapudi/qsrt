#!/usr/bin/env bash
set -euo pipefail

# Verify the frozen plan-order selection evidence for layer 52 expert 36, then
# open the independent 2,048-token reference after the active selection queue
# releases all four GPUs.

if test "$#" != 1; then
  echo "usage: $0 <selection-queue-process-id>" >&2
  exit 2
fi
predecessor_pid="$1"
if [[ ! "${predecessor_pid}" =~ ^[1-9][0-9]*$ ]]; then
  echo "selection queue process ID must be a positive integer" >&2
  exit 2
fi
while kill -0 "${predecessor_pid}" 2>/dev/null; do
  sleep 30
done

root="/home/sunil/qsrt-glm52-experiments"
report="${root}/results/glm52-layer52-frozen8-reconstructed-activation-down-refit-merged-singleton-model-kld-selection-public-reference-v2-complete-slice-ancestry/report.json"
registration="${root}/registrations/glm52_layer52_expert36_down_refit_single_reference_registration.json"
decision="${root}/launch-records/layer52-expert36-down-refit-selection-gate.json"
test -f "${report}"
test -f "${registration}"
test ! -e "${decision}"

python3 - "${report}" "${registration}" "${decision}" <<'PY'
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
registration_path = Path(sys.argv[2])
decision_path = Path(sys.argv[3])
report = json.loads(report_path.read_text())
registration = json.loads(registration_path.read_text())
evidence = registration["selection_evidence"]
arm_name = evidence["candidate_arm"]
assert hashlib.sha256(report_path.read_bytes()).hexdigest() == evidence["selection_report_sha256"]
assert report["status"] == "complete"
assert report["measurement_controls"]["passed"] is True
documents = report["documents"]
assert len(documents) == evidence["document_count"] == 16
deltas = [
    float(document["candidate_arms"][arm_name]["candidate_minus_resident_mean_forward_kld"])
    for document in documents
]
first_mean = statistics.fmean(deltas[:8])
second_mean = statistics.fmean(deltas[8:])
all_mean = statistics.fmean(deltas)
for actual, key in (
    (first_mean, "first_eight_mean_candidate_minus_resident_kld"),
    (second_mean, "second_eight_mean_candidate_minus_resident_kld"),
    (all_mean, "all_document_mean_candidate_minus_resident_kld"),
):
    assert math.isclose(actual, float(evidence[key]), rel_tol=0.0, abs_tol=1e-18)
assert first_mean < 0.0 and second_mean < 0.0
record = {
    "schema": "qsrt_glm52_two_group_model_kld_selection_decision",
    "schema_version": 1,
    "status": "accepted",
    "candidate_arm": arm_name,
    "selection_report_sha256": evidence["selection_report_sha256"],
    "registration_sha256": hashlib.sha256(registration_path.read_bytes()).hexdigest(),
    "document_count": len(documents),
    "first_eight_mean_candidate_minus_resident_kld": first_mean,
    "second_eight_mean_candidate_minus_resident_kld": second_mean,
    "all_document_mean_candidate_minus_resident_kld": all_mean,
    "acceptance_rule": "both ordered eight-document means must be negative",
}
temporary = decision_path.with_suffix(".partial")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
os.replace(temporary, decision_path)
PY

"${root}/launch-scripts/run_glm52_frozen_expert_subset_single_reference_on_kossel.sh" \
  glm52-layer52-frozen8-reconstructed-activation-down-refit-merged \
  glm52-layer52-expert36-down-refit-single-reference-absolute-target-screen \
  36

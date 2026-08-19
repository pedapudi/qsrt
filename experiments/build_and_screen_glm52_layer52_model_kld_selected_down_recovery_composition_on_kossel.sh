#!/usr/bin/env bash
set -euo pipefail

# Build the registered layer-52 expert-36 down-refit plus expert-186 rank-four
# correction. Measure the composition on the sixteen selection documents, and
# open the separate 2,048-token development reference only if both ordered
# selection groups improve against resident EXL3.

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
results_root="${root}/results"
source_snapshot="${root}/source/qsrt-multi-layer-intervention-runtime-20260819"
registration_name="glm52_layer52_model_kld_selected_down_recovery_composition.json"
registration="${root}/registrations/${registration_name}"
artifact_name="glm52-layer52-model-kld-selected-down-recovery-experts36-186"
artifact_root="${results_root}/${artifact_name}"
selection_result="${artifact_name}-public-reference-selection-screen"
selection_report="${results_root}/${selection_result}/report.json"
independent_result="${artifact_name}-single-reference-absolute-target-screen"
materialization_record="${root}/launch-records/${artifact_name}-materialization"
decision="${root}/launch-records/${artifact_name}-selection-decision.json"
container="qsrt-${artifact_name}-materialization"
image="verdictai/glm52-exl3-sparkinfer:v39-r28-r7fused-broadcast-cu132-sm120a"

for required in \
  "${registration}" \
  "${source_snapshot}/scripts/materialize_glm52_single_layer_dense_intervention_composition.py" \
  "${source_snapshot}/qsrt/glm52_expert_intervention_runtime.py"; do
  test -f "${required}"
done
test ! -e "${artifact_root}"
test ! -e "${materialization_record}"
test ! -e "${decision}"
test -z "$(docker ps -a --filter "name=^/${container}$" -q)"
docker image inspect "${image}" >/dev/null
mkdir -p "${materialization_record}"
sha256sum \
  "${registration}" \
  "${source_snapshot}/scripts/materialize_glm52_single_layer_dense_intervention_composition.py" \
  "${source_snapshot}/qsrt/glm52_expert_intervention_runtime.py" \
  > "${materialization_record}/materialization-inputs.sha256"

docker create \
  --name "${container}" \
  --label qsrt.experiment=glm52-single-layer-model-kld-selected-composition-materialization \
  --label qsrt.model-downloads-performed=false \
  --network none \
  --entrypoint /opt/venv/bin/python \
  -e PYTHONPATH=/workspace/qsrt \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "${source_snapshot}:/workspace/qsrt:ro" \
  -v "${registration}:/registration.json:ro" \
  -v "${results_root}:/results:rw" \
  "${image}" \
  /workspace/qsrt/scripts/materialize_glm52_single_layer_dense_intervention_composition.py \
  --registration /registration.json \
  --results-root /results \
  --dest "/results/${artifact_name}"

docker inspect "${container}" > "${materialization_record}/container-created-inspect.json"
set +e
docker start -a "${container}" > "${materialization_record}/run.log" 2>&1
status=$?
set -e
docker inspect "${container}" > "${materialization_record}/container-completed-inspect.json"
if test "${status}" -ne 0; then
  exit "${status}"
fi
sha256sum "${artifact_root}/manifest.json" "${artifact_root}/report.json" \
  > "${materialization_record}/materialized-artifact.sha256"

"${root}/launch-scripts/run_glm52_complete_panel_public_reference_screen_on_kossel.sh" \
  "${artifact_name}" \
  "${selection_result}"

python3 - "${selection_report}" "${registration}" "${decision}" <<'PY'
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
registration_path = Path(sys.argv[2])
decision_path = Path(sys.argv[3])
report = json.loads(report_path.read_text())
registration = json.loads(registration_path.read_text())
assert report["status"] == "complete"
assert report["measurement_controls"]["passed"] is True
documents = report["documents"]
assert len(documents) == registration["selection_rule"]["document_count"] == 16
deltas = [
    float(document["candidate_minus_resident_mean_forward_kld"])
    for document in documents
]
first_mean = statistics.fmean(deltas[:8])
second_mean = statistics.fmean(deltas[8:])
all_mean = statistics.fmean(deltas)
accepted = first_mean < 0.0 and second_mean < 0.0
record = {
    "schema": "qsrt_glm52_composed_intervention_selection_decision",
    "schema_version": 1,
    "status": "accepted" if accepted else "rejected",
    "registration_sha256": hashlib.sha256(registration_path.read_bytes()).hexdigest(),
    "selection_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    "document_count": len(documents),
    "first_eight_mean_candidate_minus_resident_kld": first_mean,
    "second_eight_mean_candidate_minus_resident_kld": second_mean,
    "all_document_mean_candidate_minus_resident_kld": all_mean,
    "acceptance_rule": "both ordered eight-document means must be negative",
}
temporary = decision_path.with_suffix(".partial")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
os.replace(temporary, decision_path)
if not accepted:
    raise SystemExit(3)
PY

"${root}/launch-scripts/run_glm52_frozen_expert_subset_single_reference_on_kossel.sh" \
  "${artifact_name}" \
  "${independent_result}"

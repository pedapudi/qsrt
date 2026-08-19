#!/usr/bin/env bash
set -euo pipefail

# After the late-middle-layer construction queue releases the GPUs, freeze all
# arms that passed both ordered public-document groups and measure each arm on
# the separate published 2,048-token BF16 reference.  Public-document KLD alone
# determines the measurement order.  The longer reference cannot enter arm
# selection because every selected expert set is written to the queue manifest
# before the first long-reference process starts.

if test "$#" -ne 1 || [[ ! "$1" =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 <hot-layer-construction-process-id>" >&2
  exit 2
fi
predecessor_pid="$1"
while kill -0 "${predecessor_pid}" 2>/dev/null; do
  sleep 30
done

experiment_root="/home/sunil/qsrt-glm52-experiments"
queue_path="${experiment_root}/launch-records/glm52-hot-layer-retained-arm-single-reference-queue.json"
summary_path="${experiment_root}/launch-records/glm52-hot-layer-retained-arm-single-reference-results.json"
single_reference_launcher="${experiment_root}/launch-scripts/run_glm52_frozen_expert_subset_single_reference_on_kossel.sh"
test -x "${single_reference_launcher}"
test ! -e "${queue_path}"
test ! -e "${summary_path}"

python3 - "${experiment_root}" "${queue_path}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
queue_path = Path(sys.argv[2])
constructions = {
    "down-refit": "hot-band-frozen8-reconstructed-activation-down-refit-merged",
    "rank4-down-recovery": (
        "hot-band-frozen8-low-rank-down-refit-bf16-rank4-merged"
    ),
    "uniform-k3": "hot-band-frozen8-uniform-k3-merged",
}
records = []
for layer in (55, 56, 57, 58):
    for construction, artifact_suffix in constructions.items():
        selection_result = (
            f"glm52-layer{layer}-{construction}-candidate-subset-"
            "public-reference-selection"
        )
        decision_path = (
            root / "launch-records" / selection_result / "selection-decision.json"
        )
        report_path = root / "results" / selection_result / "report.json"
        artifact_name = f"glm52-layer{layer}-{artifact_suffix}"
        artifact_report_path = root / "results" / artifact_name / "report.json"
        for path in (decision_path, report_path, artifact_report_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        decision = json.loads(decision_path.read_text())
        artifact_report = json.loads(artifact_report_path.read_text())
        if decision.get("status") != "complete":
            raise ValueError(f"selection decision is incomplete: {decision_path}")
        if artifact_report.get("status") != "complete":
            raise ValueError(f"candidate artifact is incomplete: {artifact_report_path}")
        if decision.get("report_sha256") != hashlib.sha256(report_path.read_bytes()).hexdigest():
            raise ValueError(f"selection report identity changed: {report_path}")
        for arm in decision.get("arms", []):
            if not arm.get("retained"):
                continue
            experts = arm.get("selected_experts")
            if (
                not isinstance(experts, list)
                or not experts
                or any(type(expert) is not int for expert in experts)
            ):
                raise ValueError(f"retained arm has malformed experts: {decision_path}")
            arm_name = arm.get("name")
            if not isinstance(arm_name, str) or not arm_name:
                raise ValueError(f"retained arm has no name: {decision_path}")
            result_name = (
                f"glm52-layer{layer}-{construction}-{arm_name}-"
                "single-reference-absolute-target-screen"
            )
            records.append(
                {
                    "model_layer": layer,
                    "construction": construction,
                    "arm_name": arm_name,
                    "selected_experts": experts,
                    "artifact_directory_name": artifact_name,
                    "artifact_manifest_identity": artifact_report["manifest_sha256"],
                    "selection_decision_sha256": hashlib.sha256(
                        decision_path.read_bytes()
                    ).hexdigest(),
                    "screening_document_mean_delta": arm[
                        "screening_document_mean_delta"
                    ],
                    "selection_check_document_mean_delta": arm[
                        "selection_check_document_mean_delta"
                    ],
                    "all_document_mean_delta": arm["all_document_mean_delta"],
                    "result_directory_name": result_name,
                }
            )
records.sort(
    key=lambda item: (
        item["all_document_mean_delta"],
        item["model_layer"],
        item["construction"],
        item["arm_name"],
    )
)
queue = {
    "schema": "qsrt_glm52_hot_layer_retained_arm_single_reference_queue",
    "schema_version": 1,
    "status": "frozen_before_single_reference_measurement",
    "selection_source": (
        "two ordered groups of eight public 512-token BF16-reference documents"
    ),
    "ordering_rule": (
        "ascending all-document candidate-minus-resident mean KLD; the separate "
        "2,048-token reference does not affect selection or ordering"
    ),
    "absolute_development_target_mean_kld": 0.059,
    "evidence_boundary": (
        "One 2,048-token reference is a development screen, not document-replicated "
        "checkpoint qualification."
    ),
    "records": records,
}
queue_path.parent.mkdir(parents=True, exist_ok=True)
temporary = queue_path.with_name(f".{queue_path.name}.partial-{os.getpid()}")
temporary.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n")
os.replace(temporary, queue_path)
print(json.dumps({"retained_arm_count": len(records), "queue": str(queue_path)}))
PY

while IFS=$'\t' read -r artifact_name result_name experts; do
  if test -f "${experiment_root}/results/${result_name}/report.json"; then
    continue
  fi
  "${single_reference_launcher}" "${artifact_name}" "${result_name}" "${experts}"
done < <(
  python3 - "${queue_path}" <<'PY'
import json
import sys

queue = json.load(open(sys.argv[1]))
for record in queue["records"]:
    experts = ",".join(str(value) for value in record["selected_experts"])
    print(record["artifact_directory_name"], record["result_directory_name"], experts, sep="\t")
PY
)

python3 - "${experiment_root}" "${queue_path}" "${summary_path}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
queue_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
queue = json.loads(queue_path.read_text())
results = []
for queued in queue["records"]:
    report_path = root / "results" / queued["result_directory_name"] / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("status") != "complete" or report.get("measurement_controls_passed") is not True:
        raise ValueError(f"single-reference controls failed: {report_path}")
    subset = report.get("candidate_expert_subset_paired", {}).get(
        "frozen_expert_subset"
    )
    if not isinstance(subset, dict):
        raise ValueError(f"single-reference report lacks frozen subset: {report_path}")
    paired = subset["paired"]
    result = dict(queued)
    result.update(
        {
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "resident_mean_forward_kld": paired["baseline_mean_forward_kld"],
            "candidate_mean_forward_kld": paired["candidate_mean_forward_kld"],
            "candidate_minus_resident_mean_forward_kld": paired[
                "candidate_minus_baseline_mean_forward_kld"
            ],
            "candidate_below_absolute_development_target": (
                paired["candidate_mean_forward_kld"]
                < queue["absolute_development_target_mean_kld"]
            ),
        }
    )
    results.append(result)
summary = {
    "schema": "qsrt_glm52_hot_layer_retained_arm_single_reference_results",
    "schema_version": 1,
    "status": "complete",
    "queue_sha256": hashlib.sha256(queue_path.read_bytes()).hexdigest(),
    "absolute_development_target_mean_kld": queue[
        "absolute_development_target_mean_kld"
    ],
    "target_reached": any(
        record["candidate_below_absolute_development_target"] for record in results
    ),
    "results": results,
    "evidence_boundary": queue["evidence_boundary"],
}
temporary = summary_path.with_name(f".{summary_path.name}.partial-{os.getpid()}")
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
os.replace(temporary, summary_path)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

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
cross_layer_registration_path="${experiment_root}/registrations/glm52_layers55_56_57_58_public_document_selected_cross_layer_composition.json"
cross_layer_artifact_name="glm52-layers55-58-public-document-selected-cross-layer-recovery"
cross_layer_artifact_root="${experiment_root}/results/${cross_layer_artifact_name}"
source_snapshot="${experiment_root}/source/qsrt-multi-layer-intervention-general-bytes-20260819"
image="verdictai/glm52-exl3-sparkinfer:v39-r28-r7fused-broadcast-cu132-sm120a"
test -x "${single_reference_launcher}"
test -f "${source_snapshot}/scripts/materialize_glm52_multi_layer_dense_intervention.py"
test ! -e "${queue_path}"
test ! -e "${summary_path}"
test ! -e "${cross_layer_registration_path}"

python3 - \
  "${experiment_root}" \
  "${queue_path}" \
  "${cross_layer_registration_path}" \
  "${cross_layer_artifact_name}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
queue_path = Path(sys.argv[2])
cross_layer_registration_path = Path(sys.argv[3])
cross_layer_artifact_name = sys.argv[4]
constructions = {
    "down-refit": "hot-band-frozen8-reconstructed-activation-down-refit-merged",
    "k4-down-refit": (
        "hot-band-frozen8-k3-gate-k3-up-k4-down-refit-merged"
    ),
    "rank4-down-recovery": (
        "hot-band-frozen8-low-rank-down-refit-bf16-rank4-merged"
    ),
    "uniform-k3": "hot-band-frozen8-uniform-k3-merged",
}
records = []
direct_records = []
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
        retained_singletons = []
        retained_subsets = set()
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
            expert_subset = tuple(sorted(experts))
            retained_subsets.add(expert_subset)
            if len(expert_subset) == 1:
                retained_singletons.append(arm)
            result_name = (
                f"glm52-layer{layer}-{construction}-{arm_name}-"
                "single-reference-absolute-target-screen"
            )
            record = {
                "artifact_kind": "single_layer",
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
                "selection_priority_score": arm["all_document_mean_delta"],
                "selection_basis": "direct KLD for this predeclared arm",
                "result_directory_name": result_name,
            }
            records.append(record)
            direct_records.append(record)
        if len(retained_singletons) >= 2:
            union_experts = tuple(
                sorted(
                    arm["selected_experts"][0]
                    for arm in retained_singletons
                )
            )
            if union_experts not in retained_subsets:
                arm_name = "union-of-retained-singletons"
                result_name = (
                    f"glm52-layer{layer}-{construction}-{arm_name}-"
                    "single-reference-absolute-target-screen"
                )
                records.append(
                    {
                        "artifact_kind": "single_layer",
                        "model_layer": layer,
                        "construction": construction,
                        "arm_name": arm_name,
                        "selected_experts": list(union_experts),
                        "component_singleton_arm_names": sorted(
                            arm["name"] for arm in retained_singletons
                        ),
                        "artifact_directory_name": artifact_name,
                        "artifact_manifest_identity": artifact_report[
                            "manifest_sha256"
                        ],
                        "selection_decision_sha256": hashlib.sha256(
                            decision_path.read_bytes()
                        ).hexdigest(),
                        "screening_document_mean_delta": None,
                        "selection_check_document_mean_delta": None,
                        "all_document_mean_delta": None,
                        "selection_priority_score": sum(
                            arm["all_document_mean_delta"]
                            for arm in retained_singletons
                        ),
                        "selection_basis": (
                            "deterministic union of every singleton that passed both "
                            "ordered document groups; the union had no direct KLD "
                            "measurement before this long-reference screen"
                        ),
                        "result_directory_name": result_name,
                    }
                )

best_direct_record_by_layer = {}
for record in direct_records:
    layer = record["model_layer"]
    incumbent = best_direct_record_by_layer.get(layer)
    ordering = (
        record["selection_priority_score"],
        record["construction"],
        record["arm_name"],
    )
    if incumbent is None or ordering < (
        incumbent["selection_priority_score"],
        incumbent["construction"],
        incumbent["arm_name"],
    ):
        best_direct_record_by_layer[layer] = record

selected_cross_layer_records = [
    best_direct_record_by_layer[layer]
    for layer in sorted(best_direct_record_by_layer)
]
cross_layer_registration = None
if len(selected_cross_layer_records) >= 2:
    components = []
    for selected in selected_cross_layer_records:
        expert_count = len(selected["selected_experts"])
        if selected["construction"] == "rank4-down-recovery":
            logical_additional_bytes = 65_536 * expert_count
        elif selected["construction"] == "k4-down-refit":
            logical_additional_bytes = 1_572_864 * expert_count
        else:
            logical_additional_bytes = 0
        components.append(
            {
                "model_layer": selected["model_layer"],
                "source_artifact_name": selected["artifact_directory_name"],
                "expert_ids": selected["selected_experts"],
                "construction": selected["construction"],
                "logical_additional_bytes": logical_additional_bytes,
                "selection_document_mean_forward_kld_change": selected[
                    "all_document_mean_delta"
                ],
                "screening_document_mean_forward_kld_change": selected[
                    "screening_document_mean_delta"
                ],
                "selection_check_document_mean_forward_kld_change": selected[
                    "selection_check_document_mean_delta"
                ],
                "selection_decision_sha256": selected[
                    "selection_decision_sha256"
                ],
            }
        )
    cross_layer_registration = {
        "schema": "qsrt_glm52_multi_layer_intervention_registration",
        "schema_version": 1,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": {
            "description": (
                "For each measured layer, choose the directly measured arm with "
                "the lowest candidate-minus-resident mean KLD among arms that "
                "improved both ordered public-document groups. Combine the "
                "chosen layer arms before opening the 2,048-token reference."
            ),
            "document_source": (
                "two ordered groups of eight public 512-token BF16-reference "
                "documents"
            ),
            "model_layers": [
                record["model_layer"] for record in selected_cross_layer_records
            ],
        },
        "components": components,
        "evidence_boundary": (
            "The selected arms and their cross-layer composition are frozen "
            "before the 2,048-token reference is opened. That reference can "
            "screen the interaction and the absolute 0.059 target. Independent "
            "document-level references must qualify any reported reduction."
        ),
    }
    cross_layer_registration_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_registration = cross_layer_registration_path.with_name(
        f".{cross_layer_registration_path.name}.partial-{os.getpid()}"
    )
    temporary_registration.write_text(
        json.dumps(cross_layer_registration, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary_registration, cross_layer_registration_path)
    cross_layer_result_name = (
        f"{cross_layer_artifact_name}-single-reference-absolute-target-screen"
    )
    records.append(
        {
            "artifact_kind": "multi_layer",
            "model_layers": cross_layer_registration["selection_rule"][
                "model_layers"
            ],
            "construction": "public-document-selected cross-layer composition",
            "arm_name": "best-direct-arm-per-layer",
            "selected_experts": [
                expert
                for component in components
                for expert in component["expert_ids"]
            ],
            "experts_by_layer": {
                str(component["model_layer"]): component["expert_ids"]
                for component in components
            },
            "artifact_directory_name": cross_layer_artifact_name,
            "artifact_manifest_identity": None,
            "selection_decision_sha256": None,
            "cross_layer_registration_sha256": hashlib.sha256(
                cross_layer_registration_path.read_bytes()
            ).hexdigest(),
            "screening_document_mean_delta": None,
            "selection_check_document_mean_delta": None,
            "all_document_mean_delta": None,
            "selection_priority_score": sum(
                record["all_document_mean_delta"]
                for record in selected_cross_layer_records
            ),
            "selection_basis": (
                "deterministic composition of the best directly measured arm "
                "from each layer that retained at least one arm"
            ),
            "logical_additional_bytes": sum(
                component["logical_additional_bytes"] for component in components
            ),
            "result_directory_name": cross_layer_result_name,
        }
    )
records.sort(
    key=lambda item: (
        item["selection_priority_score"],
        item.get("model_layer", min(item.get("model_layers", [999]))),
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
        "ascending selection priority score; a directly measured arm uses its "
        "all-document candidate-minus-resident mean KLD, while a deterministic "
        "union uses the sum of its directly measured component values. The "
        "separate 2,048-token reference does not affect selection or ordering"
    ),
    "absolute_development_target_mean_kld": 0.059,
    "cross_layer_registration": (
        None
        if cross_layer_registration is None
        else {
            "path": str(cross_layer_registration_path),
            "sha256": hashlib.sha256(
                cross_layer_registration_path.read_bytes()
            ).hexdigest(),
            "artifact_directory_name": cross_layer_artifact_name,
        }
    ),
    "evidence_boundary": (
        "One 2,048-token reference supplies an interaction and absolute-target "
        "screen. Document-replicated checkpoint qualification requires independent "
        "contexts."
    ),
    "records": records,
}
queue_path.parent.mkdir(parents=True, exist_ok=True)
temporary = queue_path.with_name(f".{queue_path.name}.partial-{os.getpid()}")
temporary.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n")
os.replace(temporary, queue_path)
print(json.dumps({"retained_arm_count": len(records), "queue": str(queue_path)}))
PY

if test -f "${cross_layer_registration_path}"; then
  cross_layer_materialization_record="${experiment_root}/launch-records/${cross_layer_artifact_name}-materialization"
  cross_layer_container="qsrt-${cross_layer_artifact_name}-materialization"
  test ! -e "${cross_layer_artifact_root}"
  test ! -e "${cross_layer_materialization_record}"
  test -z "$(docker ps -a --filter "name=^/${cross_layer_container}$" -q)"
  mkdir -p "${cross_layer_materialization_record}"
  sha256sum \
    "${cross_layer_registration_path}" \
    "${source_snapshot}/qsrt/glm52_expert_intervention_runtime.py" \
    "${source_snapshot}/scripts/materialize_glm52_multi_layer_dense_intervention.py" \
    > "${cross_layer_materialization_record}/materialization-inputs.sha256"
  docker create \
    --name "${cross_layer_container}" \
    --label qsrt.experiment="glm52-public-document-selected-cross-layer-materialization" \
    --label qsrt.model-downloads-performed=false \
    --network none \
    --entrypoint /opt/venv/bin/python \
    -e PYTHONPATH=/workspace/qsrt \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "${source_snapshot}:/workspace/qsrt:ro" \
    -v "${cross_layer_registration_path}:/registration.json:ro" \
    -v "${experiment_root}/results:/results:rw" \
    "${image}" \
    /workspace/qsrt/scripts/materialize_glm52_multi_layer_dense_intervention.py \
    --registration /registration.json \
    --results-root /results \
    --dest "/results/${cross_layer_artifact_name}" >/dev/null
  if ! docker start --attach "${cross_layer_container}" \
    >"${cross_layer_materialization_record}/run.log" 2>&1; then
    cat "${cross_layer_materialization_record}/run.log" >&2
    exit 1
  fi
  test -f "${cross_layer_artifact_root}/report.json"
  sha256sum \
    "${cross_layer_artifact_root}/manifest.json" \
    "${cross_layer_artifact_root}/report.json" \
    > "${cross_layer_materialization_record}/materialized-artifact.sha256"
fi

while IFS=$'\t' read -r artifact_kind artifact_name result_name experts; do
  if test -f "${experiment_root}/results/${result_name}/report.json"; then
    continue
  fi
  if test "${artifact_kind}" = "multi_layer"; then
    "${single_reference_launcher}" "${artifact_name}" "${result_name}"
  else
    "${single_reference_launcher}" "${artifact_name}" "${result_name}" "${experts}"
  fi
done < <(
  python3 - "${queue_path}" <<'PY'
import json
import sys

queue = json.load(open(sys.argv[1]))
for record in queue["records"]:
    experts = ",".join(str(value) for value in record["selected_experts"])
    print(
        record["artifact_kind"],
        record["artifact_directory_name"],
        record["result_directory_name"],
        experts,
        sep="\t",
    )
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
    if queued["artifact_kind"] == "multi_layer":
        paired = report["paired"]
    else:
        subset = report.get("candidate_expert_subset_paired", {}).get(
            "frozen_expert_subset"
        )
        if not isinstance(subset, dict):
            raise ValueError(
                f"single-reference report lacks frozen subset: {report_path}"
            )
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

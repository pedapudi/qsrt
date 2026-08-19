#!/usr/bin/env bash
set -euo pipefail

# Materialize the model-KLD-selected layer-60, layer-63, and layer-64 recovery
# composition, then measure it on the sixteen candidate-selection documents.
# The optional PID keeps this work behind an existing GPU queue.

if test "$#" -gt 1; then
  echo "usage: $0 [predecessor-pid]" >&2
  exit 2
fi
predecessor_pid="${1:-}"
if test -n "${predecessor_pid}"; then
  while kill -0 "${predecessor_pid}" 2>/dev/null; do
    sleep 15
  done
fi

experiment_root="/home/sunil/qsrt-glm52-experiments"
results_root="${experiment_root}/results"
registration_name="glm52_layers60_63_64_model_kld_selected_recovery_composition.json"
registration_path="${experiment_root}/registrations/${registration_name}"
source_snapshot="${experiment_root}/source/qsrt-multi-layer-intervention-runtime-20260819"
artifact_name="glm52-layers60-63-64-model-kld-selected-recovery-experts136-164-253"
result_name="${artifact_name}-public-reference-selection-screen"
artifact_root="${results_root}/${artifact_name}"
record_root="${experiment_root}/launch-records/${artifact_name}-materialization"
container_name="qsrt-${artifact_name}-materialization"
image="verdictai/glm52-exl3-sparkinfer:v39-r28-r7fused-broadcast-cu132-sm120a"

test -f "${registration_path}"
test -f "${source_snapshot}/scripts/materialize_glm52_multi_layer_dense_intervention.py"
test ! -e "${artifact_root}"
test ! -e "${record_root}"
test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"
docker image inspect "${image}" >/dev/null
mkdir -p "${record_root}"
sha256sum \
  "${registration_path}" \
  "${source_snapshot}/qsrt/glm52_expert_intervention_runtime.py" \
  "${source_snapshot}/scripts/materialize_glm52_multi_layer_dense_intervention.py" \
  > "${record_root}/materialization-inputs.sha256"

docker create \
  --name "${container_name}" \
  --label qsrt.experiment="glm52-model-kld-selected-multi-layer-recovery-materialization" \
  --label qsrt.model-downloads-performed=false \
  --network none \
  --entrypoint /opt/venv/bin/python \
  -e PYTHONPATH=/workspace/qsrt \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "${source_snapshot}:/workspace/qsrt:ro" \
  -v "${registration_path}:/registration.json:ro" \
  -v "${results_root}:/results:rw" \
  "${image}" \
  /workspace/qsrt/scripts/materialize_glm52_multi_layer_dense_intervention.py \
  --registration /registration.json \
  --results-root /results \
  --dest "/results/${artifact_name}"

docker inspect "${container_name}" > "${record_root}/container-created-inspect.json"
set +e
docker start -a "${container_name}" > "${record_root}/run.log" 2>&1
status=$?
set -e
docker inspect "${container_name}" > "${record_root}/container-completed-inspect.json"
if test "${status}" -ne 0; then
  exit "${status}"
fi
test -f "${artifact_root}/report.json"
sha256sum "${artifact_root}/manifest.json" "${artifact_root}/report.json" \
  > "${record_root}/materialized-artifact.sha256"

"${experiment_root}/launch-scripts/run_glm52_complete_panel_public_reference_screen_on_kossel.sh" \
  "${artifact_name}" \
  "${result_name}"

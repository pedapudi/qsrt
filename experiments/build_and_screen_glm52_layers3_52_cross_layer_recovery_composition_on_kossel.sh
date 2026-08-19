#!/usr/bin/env bash
set -euo pipefail

# Materialize the frozen layer-3 and layer-52 recovery composition, then
# measure its combined effect on the reused 2,048-token BF16 reference. An
# operator must start this script explicitly after the layer-55 through
# layer-58 capture has reached its recorded stopping point.

if test "$#" -ne 0; then
  echo "usage: $0" >&2
  exit 2
fi

experiment_root="/home/sunil/qsrt-glm52-experiments"
results_root="${experiment_root}/results"
registration_name="glm52_layers3_52_cross_layer_recovery_composition.json"
registration_path="${experiment_root}/registrations/${registration_name}"
source_snapshot="${experiment_root}/source/qsrt-multi-layer-intervention-runtime-20260819"
stopping_point="${experiment_root}/launch-records/glm52-layers55-58-capture-stopping-point.json"
artifact_name="glm52-layers3-52-rank4-down-correction-expert103-and-k3-down-refit-expert36"
result_name="${artifact_name}-reused-2048-reference-screen"
artifact_root="${results_root}/${artifact_name}"
record_root="${experiment_root}/launch-records/${artifact_name}-materialization"
container_name="qsrt-${artifact_name}-materialization"
image="verdictai/glm52-exl3-sparkinfer:v39-r28-r7fused-broadcast-cu132-sm120a"

test -f "${stopping_point}"
python3 - "${stopping_point}" <<'PY'
import json
import sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text())
assert record["status"] == "safe_for_host_shutdown"
PY

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "a GPU process is already active" >&2
  exit 1
fi

for required in \
  "${registration_path}" \
  "${source_snapshot}/scripts/materialize_glm52_multi_layer_dense_intervention.py" \
  "${results_root}/glm52-layer3-frozen8-low-rank-down-reconstructed_activation_down_refit-bf16-rank-4-merged/manifest.json" \
  "${results_root}/glm52-layer52-frozen8-reconstructed-activation-down-refit-merged/manifest.json"; do
  test -f "${required}"
done
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
  --label qsrt.experiment="glm52-cross-layer-recovery-composition-materialization" \
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

"${experiment_root}/launch-scripts/run_glm52_frozen_expert_subset_single_reference_on_kossel.sh" \
  "${artifact_name}" \
  "${result_name}"

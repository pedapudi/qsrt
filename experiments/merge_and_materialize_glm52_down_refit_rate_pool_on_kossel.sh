#!/usr/bin/env bash
set -euo pipefail

# Merge the four hash-closed rate-pool slices, then materialize both the frozen
# EXL3-stratified control and the complete-expert selection-data allocation.

experiment_root="/home/sunil/qsrt-glm52-experiments"
source_copy="${experiment_root}/source/qsrt-working-tree"
results_root="${experiment_root}/results"
record_root="${experiment_root}/launch-records/glm52-layer3-down-refit-k3-k4-rate-pool-materialization"
image="sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"
pool_name="glm52-layer3-frozen8-down-refit-k3-k4-rate-pool-merged"
pool_destination="${results_root}/${pool_name}"
pre_registration="/workspace/qsrt/experiments/glm52_layer3_k3_k4_allocation_pre_registration.json"
inputs=(
  /results/glm52-layer3-frozen8-down-refit-k3-k4-rate-pool-slice-00-01
  /results/glm52-layer3-frozen8-down-refit-k3-k4-rate-pool-slice-02-03
  /results/glm52-layer3-frozen8-down-refit-k3-k4-rate-pool-slice-04-05
  /results/glm52-layer3-frozen8-down-refit-k3-k4-rate-pool-slice-06-07
)

mkdir -p "${record_root}"
test ! -e "${pool_destination}"
for input in "${inputs[@]}"; do
  test -f "${results_root}/${input#/results/}/report.json"
done

merge_container="qsrt-${pool_name}"
test -z "$(docker ps -a --filter "name=^/${merge_container}$" -q)"
docker create \
  --name "${merge_container}" \
  --label qsrt.experiment=glm52-layer3-down-refit-k3-k4-rate-pool-merge \
  --label qsrt.model-downloads-performed=false \
  --network none \
  --entrypoint /opt/venv/bin/python \
  -e PYTHONPATH=/workspace/qsrt \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "${source_copy}:/workspace/qsrt:ro" \
  -v "${results_root}:/results:rw" \
  "${image}" \
  /workspace/qsrt/scripts/merge_glm52_down_refit_rate_pool.py \
  "${inputs[@]}" \
  --dest "/results/${pool_name}" \
  --panel-manifest /workspace/qsrt/experiments/glm52_layer3_rate_pattern_panel.json \
  --layer 3
docker inspect "${merge_container}" > "${record_root}/${pool_name}-created-inspect.json"
docker start --attach "${merge_container}" > "${record_root}/${pool_name}.log" 2>&1
docker inspect "${merge_container}" > "${record_root}/${pool_name}-completed-inspect.json"

for allocation_kind in fixed_rate_stratified selection_data_complete_expert; do
  case "${allocation_kind}" in
    fixed_rate_stratified)
      artifact_name="glm52-layer3-frozen8-fixed-rate-preserving-down-refit-k3-k4"
      ;;
    selection_data_complete_expert)
      artifact_name="glm52-layer3-frozen8-selection-data-rate-preserving-down-refit-k3-k4"
      ;;
  esac
  destination="${results_root}/${artifact_name}"
  container_name="qsrt-${artifact_name}"
  test ! -e "${destination}"
  test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"
  docker create \
    --name "${container_name}" \
    --label qsrt.experiment="glm52-layer3-${allocation_kind}-rate-preserving-down-refit-k3-k4" \
    --label qsrt.model-downloads-performed=false \
    --network none \
    --entrypoint /opt/venv/bin/python \
    -e PYTHONPATH=/workspace/qsrt \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "${source_copy}:/workspace/qsrt:ro" \
    -v "${results_root}:/results:rw" \
    "${image}" \
    /workspace/qsrt/scripts/materialize_glm52_down_refit_rate_pool_allocation.py \
    --rate-pool "/results/${pool_name}" \
    --pre-registration "${pre_registration}" \
    --allocation-kind "${allocation_kind}" \
    --dest "/results/${artifact_name}"
  docker inspect "${container_name}" > "${record_root}/${artifact_name}-created-inspect.json"
  docker start --attach "${container_name}" > "${record_root}/${artifact_name}.log" 2>&1
  docker inspect "${container_name}" > "${record_root}/${artifact_name}-completed-inspect.json"
done

sha256sum \
  "${pool_destination}/manifest.json" \
  "${pool_destination}/report.json" \
  "${results_root}/glm52-layer3-frozen8-fixed-rate-preserving-down-refit-k3-k4/manifest.json" \
  "${results_root}/glm52-layer3-frozen8-fixed-rate-preserving-down-refit-k3-k4/report.json" \
  "${results_root}/glm52-layer3-frozen8-selection-data-rate-preserving-down-refit-k3-k4/manifest.json" \
  "${results_root}/glm52-layer3-frozen8-selection-data-rate-preserving-down-refit-k3-k4/report.json" \
  "${record_root}"/*.json \
  "${record_root}"/*.log

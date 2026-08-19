#!/usr/bin/env bash
set -euo pipefail

# Build one coherent layer-63 candidate without consulting its K4 measurements.
# Every accepted continuous down-refit target is re-encoded at K3 and K4; the
# registered artifact then uses K4 down projections only for experts 149 and
# 164. Four disjoint two-expert workers keep the four GPUs occupied.

experiment_root="/home/sunil/qsrt-glm52-experiments"
source_root="${experiment_root}/source-windows/glm52-b4734de-layers-52-60-63-64"
endpoint_root="${experiment_root}/model-cache/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78-nvme"
source_snapshot="${experiment_root}/source/qsrt-layer63-registered-partial-k3-k4-down-refit-20260819"
dependency_root="${experiment_root}/dependencies/exllamav3-v0.0.43-source"
uniform_k3="${experiment_root}/results/glm52-layer63-frozen8-uniform-k3-source-target-capture-sequenced-merged"
down_refit="${experiment_root}/results/glm52-layer63-frozen8-reconstructed-activation-down-refit-merged"
capture_root="${experiment_root}/captures/glm52-layers52-60-63-64-wikitext-document-disjoint-routed-inputs-artifact-layer-compatible/layer-063"
results_root="${experiment_root}/results"
registration="${experiment_root}/registrations/glm52_layer63_registered_partial_k3_k4_down_refit.json"
panel="glm52_layer63_rate_pattern_panel.json"
uniform_k4_name="glm52-layer63-frozen8-uniform-k4-source-target-merged"
rate_pool_name="glm52-layer63-frozen8-down-refit-k3-k4-rate-pool-merged"
candidate_name="glm52-layer63-experts149-164-registered-k3-k3-k4-down-refit"
record_root="${experiment_root}/launch-records/${candidate_name}-build"
image="sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"

for required in \
  "${source_root}/receipt.json" \
  "${endpoint_root}/reproducibility/r10/inventories/source_inventory.json" \
  "${source_snapshot}/scripts/build_glm52_dense_expert_interventions.py" \
  "${source_snapshot}/scripts/build_glm52_down_refit_rate_pool.py" \
  "${source_snapshot}/scripts/materialize_glm52_registered_partial_rate_map.py" \
  "${uniform_k3}/report.json" \
  "${down_refit}/report.json" \
  "${capture_root}/manifest.json" \
  "${registration}"; do
  test -f "${required}"
done
docker image inspect "${image}" >/dev/null
mkdir -p "${record_root}"

for output in "${uniform_k4_name}" "${rate_pool_name}" "${candidate_name}"; do
  test ! -e "${results_root}/${output}"
done

k4_workers=()
for gpu in 0 1 2 3; do
  offset=$((gpu * 2))
  end=$((offset + 1))
  slice_name="glm52-layer63-frozen8-uniform-k4-source-target-slice-$(printf '%02d' "${offset}")-$(printf '%02d' "${end}")"
  container_name="qsrt-${slice_name}"
  k4_workers+=("${container_name}")
  test ! -e "${results_root}/${slice_name}"
  test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"
  docker create \
    --name "${container_name}" \
    --label qsrt.experiment=glm52-layer63-uniform-k4-source-target \
    --label qsrt.model-downloads-performed=false \
    --label qsrt.output-device=internal-nvme \
    --label qsrt.panel-offset="${offset}" \
    --network none \
    --gpus "device=${gpu}" \
    --ipc host \
    --shm-size 32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --entrypoint /opt/venv/bin/python \
    -e PYTHONPATH=/workspace/qsrt:/exllamav3-python:/opt/exllamav3 \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONUNBUFFERED=1 \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -v "${source_root}:/source:ro" \
    -v "${endpoint_root}:/endpoint:ro" \
    -v "${source_snapshot}:/workspace/qsrt:ro" \
    -v "${dependency_root}:/exllamav3-python:ro" \
    -v "${results_root}:/results:rw" \
    "${image}" \
    /workspace/qsrt/scripts/build_glm52_dense_expert_interventions.py \
    --source /source \
    --source-inventory /endpoint/reproducibility/r10/inventories/source_inventory.json \
    --exl3-endpoint /endpoint \
    --panel-manifest "/workspace/qsrt/experiments/${panel}" \
    --dest "/results/${slice_name}" \
    --layer 63 \
    --candidate-bits 4 \
    --experts 2 \
    --panel-offset "${offset}" \
    --device cuda:0 \
    --exllamav3-root /exllamav3-python \
    --skip-source-shard-hashes \
    --skip-exl3-shard-hash
  docker inspect "${container_name}" > "${record_root}/${slice_name}-created.json"
  docker start "${container_name}"
done

for container_name in "${k4_workers[@]}"; do
  exit_code="$(docker wait "${container_name}")"
  if test "${exit_code}" != 0; then
    docker logs "${container_name}" >&2
    exit "${exit_code}"
  fi
done

k4_inputs=()
for offset in 0 2 4 6; do
  end=$((offset + 1))
  slice_name="glm52-layer63-frozen8-uniform-k4-source-target-slice-$(printf '%02d' "${offset}")-$(printf '%02d' "${end}")"
  k4_inputs+=("/results/${slice_name}")
done
k4_merge_container="qsrt-${uniform_k4_name}"
docker create \
  --name "${k4_merge_container}" \
  --label qsrt.experiment=glm52-layer63-uniform-k4-source-target-merge \
  --label qsrt.model-downloads-performed=false \
  --network none \
  --entrypoint /opt/venv/bin/python \
  -e PYTHONPATH=/workspace/qsrt \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "${source_snapshot}:/workspace/qsrt:ro" \
  -v "${results_root}:/results:rw" \
  "${image}" \
  /workspace/qsrt/scripts/merge_glm52_dense_expert_interventions.py \
  "${k4_inputs[@]}" \
  --dest "/results/${uniform_k4_name}" \
  --panel-manifest "/workspace/qsrt/experiments/${panel}" \
  --layer 63
docker start --attach "${k4_merge_container}"
test "$(docker inspect "${k4_merge_container}" --format '{{.State.ExitCode}}')" = 0

rate_workers=()
for gpu in 0 1 2 3; do
  offset=$((gpu * 2))
  end=$((offset + 1))
  slice_name="glm52-layer63-frozen8-down-refit-k3-k4-rate-pool-slice-$(printf '%02d' "${offset}")-$(printf '%02d' "${end}")"
  container_name="qsrt-${slice_name}"
  rate_workers+=("${container_name}")
  test ! -e "${results_root}/${slice_name}"
  test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"
  docker create \
    --name "${container_name}" \
    --label qsrt.experiment=glm52-layer63-down-refit-k3-k4-rate-pool \
    --label qsrt.model-downloads-performed=false \
    --label qsrt.output-device=internal-nvme \
    --label qsrt.panel-offset="${offset}" \
    --network none \
    --gpus "device=${gpu}" \
    --ipc host \
    --shm-size 32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --entrypoint /opt/venv/bin/python \
    -e PYTHONPATH=/workspace/qsrt:/exllamav3-python:/opt/exllamav3 \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONUNBUFFERED=1 \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -v "${source_root}:/source:ro" \
    -v "${endpoint_root}:/endpoint:ro" \
    -v "${source_snapshot}:/workspace/qsrt:ro" \
    -v "${uniform_k3}:/uniform-k3:ro" \
    -v "${down_refit}:/down-refit:ro" \
    -v "${results_root}/${uniform_k4_name}:/uniform-k4:ro" \
    -v "${capture_root}:/capture:ro" \
    -v "${dependency_root}:/exllamav3-python:ro" \
    -v "${results_root}:/results:rw" \
    "${image}" \
    /workspace/qsrt/scripts/build_glm52_down_refit_rate_pool.py \
    --source /source \
    --source-inventory /endpoint/reproducibility/r10/inventories/source_inventory.json \
    --uniform-k3 /uniform-k3 \
    --down-refit /down-refit \
    --uniform-k4 /uniform-k4 \
    --capture /capture \
    --panel-manifest "/workspace/qsrt/experiments/${panel}" \
    --dest "/results/${slice_name}" \
    --layer 63 \
    --experts 2 \
    --panel-offset "${offset}" \
    --device cuda:0 \
    --exllamav3-root /exllamav3-python \
    --skip-source-shard-hashes
  docker inspect "${container_name}" > "${record_root}/${slice_name}-created.json"
  docker start "${container_name}"
done

for container_name in "${rate_workers[@]}"; do
  exit_code="$(docker wait "${container_name}")"
  if test "${exit_code}" != 0; then
    docker logs "${container_name}" >&2
    exit "${exit_code}"
  fi
done

rate_inputs=()
for offset in 0 2 4 6; do
  end=$((offset + 1))
  slice_name="glm52-layer63-frozen8-down-refit-k3-k4-rate-pool-slice-$(printf '%02d' "${offset}")-$(printf '%02d' "${end}")"
  rate_inputs+=("/results/${slice_name}")
done
rate_merge_container="qsrt-${rate_pool_name}"
docker create \
  --name "${rate_merge_container}" \
  --label qsrt.experiment=glm52-layer63-down-refit-k3-k4-rate-pool-merge \
  --label qsrt.model-downloads-performed=false \
  --network none \
  --entrypoint /opt/venv/bin/python \
  -e PYTHONPATH=/workspace/qsrt \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "${source_snapshot}:/workspace/qsrt:ro" \
  -v "${results_root}:/results:rw" \
  "${image}" \
  /workspace/qsrt/scripts/merge_glm52_down_refit_rate_pool.py \
  "${rate_inputs[@]}" \
  --dest "/results/${rate_pool_name}" \
  --panel-manifest "/workspace/qsrt/experiments/${panel}" \
  --layer 63
docker start --attach "${rate_merge_container}"
test "$(docker inspect "${rate_merge_container}" --format '{{.State.ExitCode}}')" = 0

materialize_container="qsrt-${candidate_name}-materialize"
docker create \
  --name "${materialize_container}" \
  --label qsrt.experiment=glm52-layer63-registered-partial-k3-k4-down-refit-materialization \
  --label qsrt.model-downloads-performed=false \
  --network none \
  --entrypoint /opt/venv/bin/python \
  -e PYTHONPATH=/workspace/qsrt \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "${source_snapshot}:/workspace/qsrt:ro" \
  -v "${results_root}/${rate_pool_name}:/rate-pool:ro" \
  -v "${registration}:/registration.json:ro" \
  -v "${results_root}:/results:rw" \
  "${image}" \
  /workspace/qsrt/scripts/materialize_glm52_registered_partial_rate_map.py \
  --rate-pool /rate-pool \
  --registration /registration.json \
  --dest "/results/${candidate_name}"
docker start --attach "${materialize_container}"
test "$(docker inspect "${materialize_container}" --format '{{.State.ExitCode}}')" = 0

sha256sum \
  "${registration}" \
  "${results_root}/${uniform_k4_name}/manifest.json" \
  "${results_root}/${uniform_k4_name}/report.json" \
  "${results_root}/${rate_pool_name}/manifest.json" \
  "${results_root}/${rate_pool_name}/report.json" \
  "${results_root}/${candidate_name}/manifest.json" \
  "${results_root}/${candidate_name}/report.json" \
  > "${record_root}/outputs.sha256"

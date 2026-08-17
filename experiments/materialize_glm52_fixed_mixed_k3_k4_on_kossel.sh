#!/usr/bin/env bash
set -euo pipefail

# Materialize the allocation frozen before K4 measurement.  The operation is
# CPU-only and cannot select rates from the reporting KLD context.

experiment_root="/home/sunil/qsrt-glm52-experiments"
source_copy="${experiment_root}/source/qsrt-working-tree"
results_root="${experiment_root}/results"
base_name="glm52-layer3-frozen8-reconstructed-activation-down-refit-merged"
k4_name="glm52-layer3-frozen8-uniform-k4-source-target-merged"
destination_name="glm52-layer3-frozen8-fixed-mixed-k3-k4-down-refit"
destination="${results_root}/${destination_name}"
record_root="${experiment_root}/launch-records/${destination_name}"
container_name="qsrt-${destination_name}"
image="sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"

test -f "${results_root}/${base_name}/report.json"
test -f "${results_root}/${k4_name}/report.json"
test ! -e "${destination}"
test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"
mkdir -p "${record_root}"

docker create \
  --name "${container_name}" \
  --label qsrt.experiment=glm52-layer3-fixed-mixed-k3-k4-down-refit-materialization \
  --label qsrt.model-downloads-performed=false \
  --network none \
  --entrypoint /opt/venv/bin/python \
  -e PYTHONPATH=/workspace/qsrt \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 \
  -v "${source_copy}:/workspace/qsrt:ro" \
  -v "${results_root}:/results:rw" \
  "${image}" \
  /workspace/qsrt/scripts/materialize_glm52_fixed_mixed_k3_k4.py \
  --base "/results/${base_name}" \
  --uniform-k4 "/results/${k4_name}" \
  --pre-registration /workspace/qsrt/experiments/glm52_layer3_k3_k4_allocation_pre_registration.json \
  --dest "/results/${destination_name}"

docker inspect "${container_name}" > "${record_root}/container-created-inspect.json"
docker start --attach "${container_name}"
docker inspect "${container_name}" > "${record_root}/container-completed-inspect.json"
sha256sum \
  "${destination}/manifest.json" \
  "${destination}/report.json" \
  "${record_root}/container-created-inspect.json" \
  "${record_root}/container-completed-inspect.json"

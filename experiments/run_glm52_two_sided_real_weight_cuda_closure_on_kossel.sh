#!/usr/bin/env bash
set -euo pipefail

# Exercise the two-sided K3 encoder on one complete real GLM-5.2 gate matrix.
# The identity output metric checks dimensions, transformations, memory use,
# frozen-scale replay, and CUDA traversal. It does not measure downstream KLD.

experiment_root="/home/sunil/qsrt-glm52-experiments"
source_root="/home/sunil/usb-mnt/zai-org/GLM-5.2-b4734de-layer003-source-window"
endpoint_root="/home/sunil/usb-mnt/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78"
source_copy="${experiment_root}/source/qsrt-working-tree"
input_artifact="${experiment_root}/results/glm52-layer3-frozen8-dense-endpoints-r7-closure-merged-v2"
dependency_root="${experiment_root}/dependencies/exllamav3-v0.0.43-source"
results_root="${experiment_root}/results"
result_name="glm52-layer3-expert64-gate-two-sided-identity-curvature-complete-matrix-canonical-loss-cuda-closure"
destination="${results_root}/${result_name}.json"
record_root="${experiment_root}/launch-records/${result_name}"
container_name="qsrt-${result_name}"
image="sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"

test ! -e "${destination}"
test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"
mkdir -p "${record_root}"

docker create \
  --name "${container_name}" \
  --label qsrt.experiment=glm52-two-sided-real-dimension-cuda-closure \
  --label qsrt.model-downloads-performed=false \
  --label qsrt.evidence-role=dimensional-and-memory-closure \
  --network none \
  --gpus device=0 \
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
  -v "${source_copy}:/workspace/qsrt:ro" \
  -v "${input_artifact}:/artifact:ro" \
  -v "${dependency_root}:/exllamav3-python:ro" \
  -v "${results_root}:/results:rw" \
  "${image}" \
  /workspace/qsrt/scripts/validate_glm52_two_sided_real_weight_cuda.py \
  --source-root /source \
  --source-inventory /endpoint/reproducibility/r10/inventories/source_inventory.json \
  --input-artifact /artifact \
  --dest "/results/${result_name}.json" \
  --exllamav3-root /exllamav3-python \
  --layer 3 \
  --expert 64 \
  --device cuda:0 \
  --skip-source-shard-hashes

docker inspect "${container_name}" > "${record_root}/container-created-inspect.json"
docker start --attach "${container_name}"
docker logs "${container_name}" > "${record_root}/container.log" 2>&1
docker inspect "${container_name}" > "${record_root}/container-completed-inspect.json"
sha256sum \
  "${destination}" \
  "${record_root}/container-created-inspect.json" \
  "${record_root}/container-completed-inspect.json" \
  "${record_root}/container.log"

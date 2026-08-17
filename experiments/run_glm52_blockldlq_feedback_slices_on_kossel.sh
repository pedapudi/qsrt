#!/usr/bin/env bash
set -euo pipefail

# Encode the frozen panel with identical K3 scales and no BlockLDLQ feedback.
# Each network-disabled container receives two experts and one physical GPU.

experiment_root="/home/sunil/qsrt-glm52-experiments"
source_root="/home/sunil/usb-mnt/zai-org/GLM-5.2-b4734de-layer003-source-window"
endpoint_root="/home/sunil/usb-mnt/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78"
source_copy="${experiment_root}/source/qsrt-working-tree"
input_artifact="${experiment_root}/results/glm52-layer3-frozen8-dense-endpoints-r7-closure-merged-v2"
results_root="${experiment_root}/results"
record_root="${experiment_root}/launch-records/glm52-layer3-blockldlq-no-feedback-frozen-k3-scale"
dependency_root="${experiment_root}/dependencies/exllamav3-v0.0.43-source"
image="sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"

mkdir -p "${record_root}"
for gpu in 0 1 2 3; do
  offset=$((gpu * 2))
  end=$((offset + 1))
  slice_name="glm52-layer3-frozen8-blockldlq-no-feedback-frozen-k3-scale-slice-0${offset}-0${end}"
  container_name="qsrt-${slice_name}"
  destination="${results_root}/${slice_name}"
  test ! -e "${destination}"
  test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"
  docker create \
    --name "${container_name}" \
    --label qsrt.experiment=glm52-layer3-blockldlq-no-feedback-frozen-k3-scale \
    --label qsrt.model-downloads-performed=false \
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
    -v "${source_copy}:/workspace/qsrt:ro" \
    -v "${input_artifact}:/artifact:ro" \
    -v "${dependency_root}:/exllamav3-python:ro" \
    -v "${results_root}:/results:rw" \
    "${image}" \
    /workspace/qsrt/scripts/run_glm52_blockldlq_feedback_ablation.py \
    --source-root /source \
    --source-inventory /endpoint/reproducibility/r10/inventories/source_inventory.json \
    --input-artifact /artifact \
    --panel-manifest /workspace/qsrt/experiments/glm52_layer3_rate_pattern_panel.json \
    --dest "/results/${slice_name}" \
    --layer 3 \
    --experts 2 \
    --panel-offset "${offset}" \
    --feedback-multiplier 0 \
    --device cuda:0 \
    --exllamav3-root /exllamav3-python \
    --skip-source-shard-hashes
  docker inspect "${container_name}" > "${record_root}/${slice_name}-created-inspect.json"
  docker start "${container_name}"
  docker inspect "${container_name}" > "${record_root}/${slice_name}-started-inspect.json"
done

sha256sum "${record_root}"/*-inspect.json

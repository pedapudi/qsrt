#!/usr/bin/env bash
set -euo pipefail

# Fit one BF16 rank-four correction to each frozen panel expert's refitted down
# projection. Each fit uses the candidate's own reconstructed gate/up
# activations. Four independent two-expert workers are merged without changing
# their dense screening endpoints or stored factors.

experiment_root="/home/sunil/qsrt-glm52-experiments"
source_root="${experiment_root}/source-windows/glm52-b4734de-layers-52-60-63-64"
endpoint_root="${experiment_root}/model-cache/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78-nvme"
source_snapshot="${experiment_root}/source/qsrt-rank4-selection-fallback-and-public-kld-20260818"
capture_parent="${experiment_root}/captures/glm52-layers52-60-63-64-wikitext-document-disjoint-routed-inputs-artifact-layer-compatible"
results_root="${experiment_root}/results"
image="sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"

test -f "${capture_parent}/manifest.json"
test -f "${source_root}/receipt.json"
test -f "${endpoint_root}/reproducibility/r10/inventories/source_inventory.json"
test -f "${source_snapshot}/scripts/build_glm52_low_rank_down_adapter.py"
docker image inspect "${image}" >/dev/null

for layer in 60 64 52 63; do
  panel="glm52_layer${layer}_rate_pattern_panel.json"
  input_name="glm52-layer${layer}-frozen8-reconstructed-activation-down-refit-merged"
  capture_root="${capture_parent}/layer-$(printf '%03d' "${layer}")"
  slice_stem="glm52-layer${layer}-frozen8-low-rank-down-reconstructed-activation-refit-derived-input-identity-checked-selection-fallback-bf16-rank-4-slice"
  merged_name="glm52-layer${layer}-frozen8-low-rank-down-reconstructed-activation-refit-derived-input-identity-checked-selection-fallback-bf16-rank-4-merged"
  record_root="${experiment_root}/launch-records/${merged_name}"
  test -f "${results_root}/${input_name}/report.json"
  test -f "${capture_root}/manifest.json"
  test ! -e "${results_root}/${merged_name}"
  mkdir -p "${record_root}"

  worker_names=()
  for gpu in 0 1 2 3; do
    offset=$((gpu * 2))
    end=$((offset + 1))
    slice_name="${slice_stem}-$(printf '%02d' "${offset}")-$(printf '%02d' "${end}")"
    container_name="qsrt-${slice_name}"
    worker_names+=("${container_name}")
    test ! -e "${results_root}/${slice_name}"
    test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"
    docker create \
      --name "${container_name}" \
      --label qsrt.experiment="glm52-layer${layer}-activation-weighted-rank4-down-correction" \
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
      -e PYTHONPATH=/workspace/qsrt \
      -e PYTHONDONTWRITEBYTECODE=1 \
      -e PYTHONUNBUFFERED=1 \
      -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
      -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      -v "${source_root}:/source:ro" \
      -v "${endpoint_root}:/endpoint:ro" \
      -v "${source_snapshot}:/workspace/qsrt:ro" \
      -v "${results_root}/${input_name}:/artifact:ro" \
      -v "${capture_root}:/capture:ro" \
      -v "${results_root}:/results:rw" \
      "${image}" \
      /workspace/qsrt/scripts/build_glm52_low_rank_down_adapter.py \
      --source /source \
      --source-inventory /endpoint/reproducibility/r10/inventories/source_inventory.json \
      --input-artifact /artifact \
      --capture /capture \
      --panel-manifest "/workspace/qsrt/experiments/${panel}" \
      --dest "/results/${slice_name}" \
      --base-construction reconstructed_activation_down_refit \
      --rank 4 \
      --layer "${layer}" \
      --experts 2 \
      --panel-offset "${offset}" \
      --ridge-factors 0.001,0.01,0.1,1.0 \
      --oversampling 8 \
      --power-iterations 2 \
      --batch-rows 2048 \
      --seed 20260818 \
      --device cuda:0 \
      --skip-source-shard-hashes
    docker inspect "${container_name}" > "${record_root}/${slice_name}-created.json"
    docker start "${container_name}"
  done

  for container_name in "${worker_names[@]}"; do
    exit_code="$(docker wait "${container_name}")"
    if test "${exit_code}" != "0"; then
      docker logs "${container_name}" >&2
      exit "${exit_code}"
    fi
  done

  inputs=()
  for offset in 0 2 4 6; do
    end=$((offset + 1))
    slice_name="${slice_stem}-$(printf '%02d' "${offset}")-$(printf '%02d' "${end}")"
    test -f "${results_root}/${slice_name}/report.json"
    inputs+=("/results/${slice_name}")
  done
  merge_container="qsrt-${merged_name}"
  test -z "$(docker ps -a --filter "name=^/${merge_container}$" -q)"
  docker create \
    --name "${merge_container}" \
    --label qsrt.experiment="glm52-layer${layer}-activation-weighted-rank4-down-correction-merge" \
    --label qsrt.model-downloads-performed=false \
    --network none \
    --entrypoint /opt/venv/bin/python \
    -e PYTHONPATH=/workspace/qsrt \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "${source_snapshot}:/workspace/qsrt:ro" \
    -v "${results_root}:/results:rw" \
    "${image}" \
    /workspace/qsrt/scripts/merge_glm52_dense_expert_interventions.py \
    "${inputs[@]}" \
    --dest "/results/${merged_name}" \
    --panel-manifest "/workspace/qsrt/experiments/${panel}" \
    --layer "${layer}"
  docker start --attach "${merge_container}"
  test "$(docker inspect "${merge_container}" --format '{{.State.ExitCode}}')" = "0"
  sha256sum \
    "${results_root}/${merged_name}/manifest.json" \
    "${results_root}/${merged_name}/report.json" \
    > "${record_root}/merged-artifact.sha256"
done

#!/usr/bin/env bash
set -euo pipefail

# Build and screen one frozen late-middle GLM-5.2 expert panel. The pipeline
# constructs uniform QSRT K3, reconstructed-activation down refits, and BF16
# rank-four down corrections. Every model-KLD screen measures all eight
# singleton experts and the predeclared complete panel in one resident-model
# load.

if test "$#" -ne 1; then
  echo "usage: $0 <55|56|57|58>" >&2
  exit 2
fi
layer="$1"
case "${layer}" in
  55|56|57|58) ;;
  *) echo "layer must be 55, 56, 57, or 58" >&2; exit 2 ;;
esac

experiment_root="/home/sunil/qsrt-glm52-experiments"
source_root="${experiment_root}/source-windows/glm52-b4734de-layers-55-56-57-58"
endpoint_root="${experiment_root}/model-cache/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78-nvme"
source_snapshot="${experiment_root}/source/qsrt-layers55-58-capture-artifact-layer-compatible-20260819"
dependency_root="${experiment_root}/dependencies/exllamav3-v0.0.43-source"
capture_parent="${experiment_root}/captures/glm52-layers55-56-57-58-wikitext-document-disjoint-routed-inputs"
capture_root="${capture_parent}/layer-$(printf '%03d' "${layer}")"
results_root="${experiment_root}/results"
registration_root="${experiment_root}/registrations"
screen_script="${experiment_root}/launch-scripts/run_glm52_complete_panel_public_reference_screen_on_kossel.sh"
image="sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"
panel="glm52_layer${layer}_rate_pattern_panel.json"

test -f "${source_root}/receipt.json"
test -f "${endpoint_root}/reproducibility/r10/inventories/source_inventory.json"
test -f "${source_snapshot}/experiments/${panel}"
test -f "${source_snapshot}/scripts/build_glm52_error_blind_candidate_subset_plan.py"
test -f "${source_snapshot}/scripts/build_glm52_all_panel_k4_down_registration.py"
test -f "${source_snapshot}/scripts/build_glm52_down_refit_rate_pool.py"
test -f "${source_snapshot}/scripts/merge_glm52_down_refit_rate_pool.py"
test -f "${source_snapshot}/scripts/materialize_glm52_registered_partial_rate_map.py"
test -f "${source_snapshot}/scripts/summarize_glm52_candidate_subset_selection.py"
test -f "${capture_parent}/manifest.json"
test -f "${capture_root}/manifest.json"
test -x "${screen_script}"
docker image inspect "${image}" >/dev/null
mkdir -p "${registration_root}"

validate_capture() {
  python3 - "${capture_root}/manifest.json" "${layer}" <<'PY'
import json
import sys

capture = json.load(open(sys.argv[1]))
layer = int(sys.argv[2])
assert capture["schema"] == "qsrt_glm52_layer_input_capture_manifest"
assert capture["schema_version"] == 1
assert capture["status"] == "complete"
assert capture["model_layer"] == layer
assert capture["collections"] == {"activation_fit": 32, "candidate_selection": 8}
PY
}
validate_capture

wait_for_workers() {
  local container_name exit_code
  for container_name in "$@"; do
    exit_code="$(docker wait "${container_name}")"
    if test "${exit_code}" != "0"; then
      docker logs "${container_name}" >&2
      return "${exit_code}"
    fi
  done
}

merge_slices() {
  local merged_name="$1"
  local slice_stem="$2"
  local label="$3"
  local merged_root="${results_root}/${merged_name}"
  if test -f "${merged_root}/report.json"; then
    return
  fi
  test ! -e "${merged_root}"
  local inputs=() offset end slice_name
  for offset in 0 2 4 6; do
    end=$((offset + 1))
    slice_name="${slice_stem}-$(printf '%02d' "${offset}")-$(printf '%02d' "${end}")"
    test -f "${results_root}/${slice_name}/report.json"
    inputs+=("/results/${slice_name}")
  done
  local container_name="qsrt-${merged_name}"
  local record_root="${experiment_root}/launch-records/${merged_name}"
  mkdir -p "${record_root}"
  test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"
  docker create \
    --name "${container_name}" \
    --label qsrt.experiment="${label}" \
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
    --layer "${layer}" >/dev/null
  if ! docker start --attach "${container_name}" >"${record_root}/merge.log" 2>&1; then
    cat "${record_root}/merge.log" >&2
    return 1
  fi
  test "$(docker inspect "${container_name}" --format '{{.State.ExitCode}}')" = "0"
  test -f "${merged_root}/report.json"
}

build_uniform_k3() {
  local slice_stem="glm52-layer${layer}-hot-band-frozen8-uniform-k3-slice"
  local merged_name="glm52-layer${layer}-hot-band-frozen8-uniform-k3-merged"
  if test -f "${results_root}/${merged_name}/report.json"; then
    printf '%s\n' "${merged_name}"
    return
  fi
  local workers=() gpu offset end slice_name destination container_name
  for gpu in 0 1 2 3; do
    offset=$((gpu * 2))
    end=$((offset + 1))
    slice_name="${slice_stem}-$(printf '%02d' "${offset}")-$(printf '%02d' "${end}")"
    destination="${results_root}/${slice_name}"
    if test -f "${destination}/report.json"; then
      continue
    fi
    test ! -e "${destination}"
    container_name="qsrt-${slice_name}"
    test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"
    docker create \
      --name "${container_name}" \
      --label qsrt.experiment="glm52-layer${layer}-hot-band-uniform-k3" \
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
      --layer "${layer}" \
      --candidate-bits 3 \
      --experts 2 \
      --panel-offset "${offset}" \
      --device cuda:0 \
      --exllamav3-root /exllamav3-python \
      --skip-source-shard-hashes \
      --skip-exl3-shard-hash >/dev/null
    docker start "${container_name}" >/dev/null
    workers+=("${container_name}")
  done
  if test "${#workers[@]}" -gt 0; then
    wait_for_workers "${workers[@]}"
  fi
  merge_slices "${merged_name}" "${slice_stem}" "glm52-layer${layer}-hot-band-uniform-k3-merge"
  printf '%s\n' "${merged_name}"
}

build_uniform_k4() {
  local slice_stem="glm52-layer${layer}-hot-band-frozen8-uniform-k4-slice"
  local merged_name="glm52-layer${layer}-hot-band-frozen8-uniform-k4-merged"
  if test -f "${results_root}/${merged_name}/report.json"; then
    printf '%s\n' "${merged_name}"
    return
  fi
  local workers=() gpu offset end slice_name destination container_name
  for gpu in 0 1 2 3; do
    offset=$((gpu * 2))
    end=$((offset + 1))
    slice_name="${slice_stem}-$(printf '%02d' "${offset}")-$(printf '%02d' "${end}")"
    destination="${results_root}/${slice_name}"
    if test -f "${destination}/report.json"; then
      continue
    fi
    test ! -e "${destination}"
    container_name="qsrt-${slice_name}"
    test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"
    docker create \
      --name "${container_name}" \
      --label qsrt.experiment="glm52-layer${layer}-hot-band-uniform-k4" \
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
      --layer "${layer}" \
      --candidate-bits 4 \
      --experts 2 \
      --panel-offset "${offset}" \
      --device cuda:0 \
      --exllamav3-root /exllamav3-python \
      --skip-source-shard-hashes \
      --skip-exl3-shard-hash >/dev/null
    docker start "${container_name}" >/dev/null
    workers+=("${container_name}")
  done
  if test "${#workers[@]}" -gt 0; then
    wait_for_workers "${workers[@]}"
  fi
  merge_slices "${merged_name}" "${slice_stem}" "glm52-layer${layer}-hot-band-uniform-k4-merge"
  printf '%s\n' "${merged_name}"
}

build_down_refit() {
  local input_name="$1"
  local slice_stem="glm52-layer${layer}-hot-band-frozen8-reconstructed-activation-down-refit-slice"
  local merged_name="glm52-layer${layer}-hot-band-frozen8-reconstructed-activation-down-refit-merged"
  if test -f "${results_root}/${merged_name}/report.json"; then
    printf '%s\n' "${merged_name}"
    return
  fi
  local workers=() gpu offset end slice_name destination container_name
  for gpu in 0 1 2 3; do
    offset=$((gpu * 2))
    end=$((offset + 1))
    slice_name="${slice_stem}-$(printf '%02d' "${offset}")-$(printf '%02d' "${end}")"
    destination="${results_root}/${slice_name}"
    if test -f "${destination}/report.json"; then
      continue
    fi
    test ! -e "${destination}"
    container_name="qsrt-${slice_name}"
    test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"
    docker create \
      --name "${container_name}" \
      --label qsrt.experiment="glm52-layer${layer}-hot-band-down-refit" \
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
      -v "${source_snapshot}:/workspace/qsrt:ro" \
      -v "${results_root}/${input_name}:/artifact:ro" \
      -v "${capture_root}:/capture:ro" \
      -v "${dependency_root}:/exllamav3-python:ro" \
      -v "${results_root}:/results:rw" \
      "${image}" \
      /workspace/qsrt/scripts/refit_glm52_reconstructed_activation_down.py \
      --source /source \
      --source-inventory /endpoint/reproducibility/r10/inventories/source_inventory.json \
      --input-artifact /artifact \
      --capture /capture \
      --panel-manifest "/workspace/qsrt/experiments/${panel}" \
      --dest "/results/${slice_name}" \
      --layer "${layer}" \
      --experts 2 \
      --panel-offset "${offset}" \
      --ridge-factors 0.001,0.01,0.1,1.0 \
      --device cuda:0 \
      --exllamav3-root /exllamav3-python \
      --skip-source-shard-hashes >/dev/null
    docker start "${container_name}" >/dev/null
    workers+=("${container_name}")
  done
  if test "${#workers[@]}" -gt 0; then
    wait_for_workers "${workers[@]}"
  fi
  merge_slices "${merged_name}" "${slice_stem}" "glm52-layer${layer}-hot-band-down-refit-merge"
  printf '%s\n' "${merged_name}"
}

build_rank4_down_recovery() {
  local input_name="$1"
  local slice_stem="glm52-layer${layer}-hot-band-frozen8-low-rank-down-refit-bf16-rank4-slice"
  local merged_name="glm52-layer${layer}-hot-band-frozen8-low-rank-down-refit-bf16-rank4-merged"
  if test -f "${results_root}/${merged_name}/report.json"; then
    printf '%s\n' "${merged_name}"
    return
  fi
  local workers=() gpu offset end slice_name destination container_name
  for gpu in 0 1 2 3; do
    offset=$((gpu * 2))
    end=$((offset + 1))
    slice_name="${slice_stem}-$(printf '%02d' "${offset}")-$(printf '%02d' "${end}")"
    destination="${results_root}/${slice_name}"
    if test -f "${destination}/report.json"; then
      continue
    fi
    test ! -e "${destination}"
    container_name="qsrt-${slice_name}"
    test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"
    docker create \
      --name "${container_name}" \
      --label qsrt.experiment="glm52-layer${layer}-hot-band-rank4-down-recovery" \
      --label qsrt.model-downloads-performed=false \
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
      --skip-source-shard-hashes >/dev/null
    docker start "${container_name}" >/dev/null
    workers+=("${container_name}")
  done
  if test "${#workers[@]}" -gt 0; then
    wait_for_workers "${workers[@]}"
  fi
  merge_slices "${merged_name}" "${slice_stem}" "glm52-layer${layer}-hot-band-rank4-down-recovery-merge"
  printf '%s\n' "${merged_name}"
}

build_k4_down_refit() {
  local uniform_k3_name="$1"
  local refit_name="$2"
  local registration_name="glm52-layer${layer}-hot-band-all-panel-k4-down-refit-registration.json"
  local registration_path="${registration_root}/${registration_name}"
  if test ! -f "${registration_path}"; then
    python3 "${source_snapshot}/scripts/build_glm52_all_panel_k4_down_registration.py" \
      --panel-manifest "${source_snapshot}/experiments/${panel}" \
      --down-refit "${results_root}/${refit_name}" \
      --dest "${registration_path}" \
      >"${registration_path%.json}.build.log"
  fi

  local uniform_k4_name
  uniform_k4_name="$(build_uniform_k4)"
  local slice_stem="glm52-layer${layer}-hot-band-frozen8-k3-k4-down-refit-rate-pool-slice"
  local rate_pool_name="glm52-layer${layer}-hot-band-frozen8-k3-k4-down-refit-rate-pool-merged"
  if test ! -f "${results_root}/${rate_pool_name}/report.json"; then
    local workers=() gpu offset end slice_name destination container_name
    for gpu in 0 1 2 3; do
      offset=$((gpu * 2))
      end=$((offset + 1))
      slice_name="${slice_stem}-$(printf '%02d' "${offset}")-$(printf '%02d' "${end}")"
      destination="${results_root}/${slice_name}"
      if test -f "${destination}/report.json"; then
        continue
      fi
      test ! -e "${destination}"
      container_name="qsrt-${slice_name}"
      test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"
      docker create \
        --name "${container_name}" \
        --label qsrt.experiment="glm52-layer${layer}-hot-band-k4-down-refit-rate-pool" \
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
        -v "${source_snapshot}:/workspace/qsrt:ro" \
        -v "${results_root}/${uniform_k3_name}:/uniform-k3:ro" \
        -v "${results_root}/${refit_name}:/down-refit:ro" \
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
        --layer "${layer}" \
        --experts 2 \
        --panel-offset "${offset}" \
        --device cuda:0 \
        --exllamav3-root /exllamav3-python \
        --skip-source-shard-hashes >/dev/null
      docker start "${container_name}" >/dev/null
      workers+=("${container_name}")
    done
    if test "${#workers[@]}" -gt 0; then
      wait_for_workers "${workers[@]}"
    fi
    local inputs=() offset end slice_name
    for offset in 0 2 4 6; do
      end=$((offset + 1))
      slice_name="${slice_stem}-$(printf '%02d' "${offset}")-$(printf '%02d' "${end}")"
      test -f "${results_root}/${slice_name}/report.json"
      inputs+=("/results/${slice_name}")
    done
    local merge_container="qsrt-${rate_pool_name}"
    local merge_record="${experiment_root}/launch-records/${rate_pool_name}"
    mkdir -p "${merge_record}"
    test -z "$(docker ps -a --filter "name=^/${merge_container}$" -q)"
    docker create \
      --name "${merge_container}" \
      --label qsrt.experiment="glm52-layer${layer}-hot-band-k4-down-refit-rate-pool-merge" \
      --label qsrt.model-downloads-performed=false \
      --network none \
      --entrypoint /opt/venv/bin/python \
      -e PYTHONPATH=/workspace/qsrt \
      -e PYTHONDONTWRITEBYTECODE=1 \
      -v "${source_snapshot}:/workspace/qsrt:ro" \
      -v "${results_root}:/results:rw" \
      "${image}" \
      /workspace/qsrt/scripts/merge_glm52_down_refit_rate_pool.py \
      "${inputs[@]}" \
      --dest "/results/${rate_pool_name}" \
      --panel-manifest "/workspace/qsrt/experiments/${panel}" \
      --layer "${layer}" >/dev/null
    if ! docker start --attach "${merge_container}" >"${merge_record}/merge.log" 2>&1; then
      cat "${merge_record}/merge.log" >&2
      return 1
    fi
    test -f "${results_root}/${rate_pool_name}/report.json"
  fi

  local artifact_name="glm52-layer${layer}-hot-band-frozen8-k3-gate-k3-up-k4-down-refit-merged"
  if test ! -f "${results_root}/${artifact_name}/report.json"; then
    local materialize_container="qsrt-${artifact_name}"
    local materialize_record="${experiment_root}/launch-records/${artifact_name}"
    mkdir -p "${materialize_record}"
    test -z "$(docker ps -a --filter "name=^/${materialize_container}$" -q)"
    docker create \
      --name "${materialize_container}" \
      --label qsrt.experiment="glm52-layer${layer}-hot-band-k4-down-refit-materialization" \
      --label qsrt.model-downloads-performed=false \
      --network none \
      --entrypoint /opt/venv/bin/python \
      -e PYTHONPATH=/workspace/qsrt \
      -e PYTHONDONTWRITEBYTECODE=1 \
      -v "${source_snapshot}:/workspace/qsrt:ro" \
      -v "${results_root}/${rate_pool_name}:/rate-pool:ro" \
      -v "${registration_path}:/registration.json:ro" \
      -v "${results_root}:/results:rw" \
      "${image}" \
      /workspace/qsrt/scripts/materialize_glm52_registered_partial_rate_map.py \
      --rate-pool /rate-pool \
      --registration /registration.json \
      --dest "/results/${artifact_name}" >/dev/null
    if ! docker start --attach "${materialize_container}" >"${materialize_record}/materialize.log" 2>&1; then
      cat "${materialize_record}/materialize.log" >&2
      return 1
    fi
    test -f "${results_root}/${artifact_name}/report.json"
  fi
  printf '%s\n' "${artifact_name}"
}

screen_artifact() {
  local artifact_name="$1"
  local construction_name="$2"
  local description="$3"
  local plan_name="glm52-layer${layer}-${construction_name}-error-blind-candidate-subsets.json"
  local plan_path="${registration_root}/${plan_name}"
  local result_name="glm52-layer${layer}-${construction_name}-candidate-subset-public-reference-selection"
  local decision="${experiment_root}/launch-records/${result_name}/selection-decision.json"
  if test ! -f "${plan_path}"; then
    python3 "${source_snapshot}/scripts/build_glm52_error_blind_candidate_subset_plan.py" \
      --artifact "${results_root}/${artifact_name}" \
      --dest "${plan_path}" \
      --candidate-description "${description}" \
      --include-complete-panel
  fi
  if test ! -f "${results_root}/${result_name}/report.json"; then
    bash "${screen_script}" "${artifact_name}" "${result_name}" "${plan_name}"
  fi
  if test ! -f "${decision}"; then
    PYTHONPATH="${source_snapshot}" python3 \
      "${source_snapshot}/scripts/summarize_glm52_candidate_subset_selection.py" \
      --plan "${plan_path}" \
      --report "${results_root}/${result_name}/report.json" \
      --dest "${decision}"
  fi
  jq '{retained_arm_names, arms: [.arms[] | select(.retained)]}' "${decision}"
}

uniform_name="$(build_uniform_k3)"
refit_name="$(build_down_refit "${uniform_name}")"
screen_artifact \
  "${refit_name}" \
  "down-refit" \
  "uniform K3 with a reconstructed-activation down refit"

k4_down_refit_name="$(build_k4_down_refit "${uniform_name}" "${refit_name}")"
screen_artifact \
  "${k4_down_refit_name}" \
  "k4-down-refit" \
  "K3 gate and up with a K4 reconstructed-activation down target or source-target fallback"

rank4_name="$(build_rank4_down_recovery "${refit_name}")"
screen_artifact \
  "${rank4_name}" \
  "rank4-down-recovery" \
  "uniform K3 with a reconstructed-activation down refit and a BF16 rank-four down correction"

screen_artifact \
  "${uniform_name}" \
  "uniform-k3" \
  "uniform K3"

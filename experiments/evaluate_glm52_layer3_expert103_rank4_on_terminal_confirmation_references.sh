#!/usr/bin/env bash
set -euo pipefail

# Qualify the correction frozen after the eight-document screen on thirty-two
# untouched documents. A failed quality gate is a valid experimental result.

SOURCE_ROOT="${1:-/home/sunil/qsrt-glm52-experiments/source/qsrt-terminal-teacher-reference-b4734de}"
REFERENCE_ROOT="${2:-/home/sunil/usb-mnt/qsrt-experiments/glm52-terminal-teacher-reference-b4734de}"
EXPERIMENT_ROOT="${3:-/home/sunil/qsrt-glm52-experiments}"
MODEL_ROOT="${EXPERIMENT_ROOT}/model-cache/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78-nvme"
ARTIFACT_NAME="glm52-layer3-frozen8-low-rank-down-reconstructed_activation_down_refit-bf16-rank-4-factorized-runtime-v1-merged"
SCREEN_NAME="glm52-layer3-expert103-rank4-terminal-endpoint-eight-document-screen"
FREEZE_NAME="glm52-layer3-expert103-rank4-terminal-confirmation-freeze"
RESULT_NAME="glm52-layer3-expert103-rank4-terminal-endpoint-thirty-two-document-confirmation"
ARTIFACT_ROOT="${EXPERIMENT_ROOT}/results/${ARTIFACT_NAME}"
SCREENING_REPORT="${EXPERIMENT_ROOT}/results/${SCREEN_NAME}/report.json"
CONFIRMATION_FREEZE="${EXPERIMENT_ROOT}/results/${FREEZE_NAME}/confirmation-freeze.json"
RESULT_ROOT="${EXPERIMENT_ROOT}/results/${RESULT_NAME}"
REFERENCE_DIRECTORY="${REFERENCE_ROOT}/confirmation-logits"
REFERENCE_PLAN="${SOURCE_ROOT}/experiments/glm52_terminal_hidden_teacher_reference_plan.json"
REGISTRATION="${SOURCE_ROOT}/experiments/glm52_layer3_rank4_expert103_low_rank_down_confirmation_registration.json"
CONTROL_ROOT="${EXPERIMENT_ROOT}/runtime-control/${RESULT_NAME}"
RECORD_ROOT="${EXPERIMENT_ROOT}/launch-records/${RESULT_NAME}"
RUNTIME_CACHE="${EXPERIMENT_ROOT}/runtime-cache/glm52-per-expert-exl3-without-fused-staging-terminal-reference-kld"
RUNNER="${SOURCE_ROOT}/scripts/run_glm52_document_disjoint_candidate_evaluation.py"
IMAGE="verdictai/glm52-exl3-sparkinfer:v39-r28-r7fused-broadcast-cu132-sm120a"
IMAGE_ID="sha256:12f86065d7fe64d30dad678585e68c91f47f1f2a32bed45ccaf108382f3928ac"
CONTAINER_NAME="qsrt-${RESULT_NAME}"

test -f "${ARTIFACT_ROOT}/manifest.json"
test -f "${ARTIFACT_ROOT}/report.json"
test -f "${SCREENING_REPORT}"
test -f "${CONFIRMATION_FREEZE}"
test -f "${REFERENCE_PLAN}"
test -f "${REGISTRATION}"
test -f "${REFERENCE_DIRECTORY}/manifest.json"
test -f "${RUNNER}"
test -f "${SOURCE_ROOT}/SOURCE_SNAPSHOT_MANIFEST.json"
test -f "${SOURCE_ROOT}/SOURCE_SNAPSHOT_MANIFEST.sha256"
test -f "${SOURCE_ROOT}/tools/verify_source_snapshot.py"
test -d "${MODEL_ROOT}"
test ! -e "${RESULT_ROOT}"
test ! -e "${CONTROL_ROOT}"
test -z "$(docker ps -a --filter "name=^/${CONTAINER_NAME}$" -q)"
test "$(docker image inspect --format '{{.Id}}' "${IMAGE}")" = "${IMAGE_ID}"
gpu_processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "${gpu_processes}" ]]; then
  echo "terminal-reference confirmation requires four idle GPUs" >&2
  exit 1
fi

read -r artifact_manifest_sha256 first_reference < <(
  python3 - "${ARTIFACT_ROOT}/report.json" "${REFERENCE_DIRECTORY}/manifest.json" "${CONFIRMATION_FREEZE}" <<'PY'
import json
import sys

artifact = json.load(open(sys.argv[1]))
references = json.load(open(sys.argv[2]))
freeze = json.load(open(sys.argv[3]))
assert artifact["status"] == "complete"
assert artifact["experiment"] == "qsrt_glm52_activation_weighted_down_adapter_v1"
assert artifact["layer"] == 3
assert artifact["expert_count"] == 8
assert freeze["status"] == "frozen_before_confirmation_reference_access"
assert freeze["artifact_manifest_sha256"] == artifact["manifest_sha256"]
assert references["status"] == "available_only_for_frozen_candidate_confirmation"
assert references["generation_contract"]["evaluation_tier"] == "confirmation"
assert references["document_count"] == 32
assert references["total_logit_rows"] == 65482
print(artifact["manifest_sha256"], references["documents"][0]["file"])
PY
)
test "${#artifact_manifest_sha256}" -eq 64
test -f "${REFERENCE_DIRECTORY}/${first_reference}"

mkdir -p "${CONTROL_ROOT}" "${RECORD_ROOT}" "${RUNTIME_CACHE}"
ln -s "/reference/${first_reference}" "${CONTROL_ROOT}/current-reference.safetensors"
python3 "${SOURCE_ROOT}/tools/verify_source_snapshot.py" \
  >"${RECORD_ROOT}/source-snapshot-verification.json"
sha256sum \
  "${SOURCE_ROOT}/SOURCE_SNAPSHOT_MANIFEST.json" \
  "${SOURCE_ROOT}/SOURCE_SNAPSHOT_MANIFEST.sha256" \
  "${REFERENCE_PLAN}" \
  "${REGISTRATION}" \
  "${SCREENING_REPORT}" \
  "${CONFIRMATION_FREEZE}" \
  "${REFERENCE_DIRECTORY}/generation_contract.json" \
  "${REFERENCE_DIRECTORY}/numerical_closure.json" \
  "${REFERENCE_DIRECTORY}/manifest.json" \
  "${SOURCE_ROOT}/qsrt/glm52_document_disjoint_confirmation.py" \
  "${SOURCE_ROOT}/qsrt/glm52_engine_kld.py" \
  "${SOURCE_ROOT}/qsrt/glm52_terminal_teacher_logits.py" \
  "${RUNNER}" \
  >"${RECORD_ROOT}/bound-inputs-and-source.sha256"

docker create \
  --name "${CONTAINER_NAME}" \
  --label qsrt.experiment="glm52-terminal-endpoint-thirty-two-document-candidate-confirmation" \
  --label qsrt.model-downloads-performed=false \
  --label qsrt.reference-document-count=32 \
  --label qsrt.reference-representation=bf16-logits \
  --label qsrt.selected-candidate="layer-3-expert-103-rank-4-down-correction" \
  --label qsrt.exl3-moe-execution=three-gemm-per-expert-correctness \
  --network none \
  --gpus all \
  --ipc host \
  --shm-size 64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --entrypoint /opt/venv/bin/python \
  -e PYTHONPATH=/workspace/runtime/glm52_expert_intervention:/workspace \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 \
  -e QSRT_GLM52_INTERVENTION_ROOT=/artifact \
  -e QSRT_GLM52_INTERVENTION_CONTROL=/control/control.json \
  -e QSRT_GLM52_INTERVENTION_MANIFEST_SHA256="${artifact_manifest_sha256}" \
  -e QSRT_GLM52_FORCE_PER_EXPERT_EXL3_MOE=1 \
  -e QSRT_GLM52_ENGINE_KLD_REFERENCE_PATH=/control/current-reference.safetensors \
  -e QSRT_GLM52_ENGINE_KLD_REFERENCE_KEY=logits \
  -e QSRT_GLM52_ENGINE_KLD_REFERENCE_REPRESENTATION=logits \
  -e QSRT_GLM52_ENGINE_KLD_CHUNK_ROWS=4 \
  -e HF_HOME=/tmp/huggingface \
  -e HF_DATASETS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=0 \
  -e B12X_MOE_FORCE_A16=1 \
  -e B12X_W4A16_TC_DECODE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_EXL3_R7_FUSED=0 \
  -e KV_FP8_ROPE=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -v "${MODEL_ROOT}:/model:ro" \
  -v "${REFERENCE_DIRECTORY}:/reference:ro" \
  -v "${REFERENCE_PLAN}:/reference-plan.json:ro" \
  -v "${REGISTRATION}:/candidate-registration.json:ro" \
  -v "${SCREENING_REPORT}:/screening-report.json:ro" \
  -v "${CONFIRMATION_FREEZE}:/confirmation-freeze.json:ro" \
  -v "${ARTIFACT_ROOT}:/artifact:ro" \
  -v "${SOURCE_ROOT}:/workspace:ro" \
  -v "${CONTROL_ROOT}:/control:rw" \
  -v "${EXPERIMENT_ROOT}/results:/results:rw" \
  -v "${RUNTIME_CACHE}:/cache:rw" \
  --pull never \
  "${IMAGE}" \
  /workspace/scripts/run_glm52_document_disjoint_candidate_evaluation.py \
  --model /model \
  --reference-directory /reference \
  --terminal-reference-plan /reference-plan.json \
  --evaluation-tier confirmation \
  --terminal-confirmation-freeze /confirmation-freeze.json \
  --terminal-screening-report /screening-report.json \
  --reference-link /control/current-reference.safetensors \
  --intervention-artifact /artifact \
  --confirmation-registration /candidate-registration.json \
  --candidate-runtime-mode stored_low_rank_factors_materialized_at_load_candidate \
  --control /control/control.json \
  --dest "/results/${RESULT_NAME}" \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.89 \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --load-format safetensors \
  --quantization exl3 \
  --attention-backend TRITON_MLA \
  --max-model-len 2049 \
  --max-num-batched-tokens 2048 \
  --kld-chunk-rows 4

docker inspect "${CONTAINER_NAME}" >"${RECORD_ROOT}/container-created-inspect.json"
set +e
docker start -a "${CONTAINER_NAME}" >"${RECORD_ROOT}/run.log" 2>&1
status=$?
set -e
docker inspect "${CONTAINER_NAME}" >"${RECORD_ROOT}/container-completed-inspect.json"
if test "${status}" -ne 0; then
  exit "${status}"
fi

python3 - "${RESULT_ROOT}/report.json" "${RECORD_ROOT}/completion.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
report = json.loads(report_path.read_text())
assert report["schema"] == "qsrt_glm52_document_disjoint_candidate_evaluation"
assert report["status"] == "complete"
assert report["evaluation_tier"] == "confirmation"
assert report["measurement_controls"]["passed"] is True
assert report["summary"]["document_count"] == 32
assert report["summary"]["position_count"] == 65482
assert report["model_downloads_performed"] is False
completion = {
    "schema": "qsrt_glm52_terminal_endpoint_candidate_confirmation_completion",
    "schema_version": 1,
    "status": "complete",
    "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    "candidate": report["candidate"],
    "confirmation_decision": report["confirmation_decision"],
    "summary": report["summary"],
}
temporary = Path(sys.argv[2] + ".partial")
temporary.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
os.replace(temporary, sys.argv[2])
print(json.dumps(completion, indent=2, sort_keys=True))
PY

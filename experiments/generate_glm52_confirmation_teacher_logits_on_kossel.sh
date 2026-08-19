#!/usr/bin/env bash
set -euo pipefail

# Generate the thirty-two confirmation references only after the screened
# candidate, runtime construction, and serialized correction bytes are frozen.

SOURCE_ROOT="${1:-/home/sunil/qsrt-glm52-experiments/source/qsrt-terminal-teacher-reference-b4734de}"
REFERENCE_ROOT="${2:-/home/sunil/usb-mnt/qsrt-experiments/glm52-terminal-teacher-reference-b4734de}"
EXPERIMENT_ROOT="${3:-/home/sunil/qsrt-glm52-experiments}"
MODEL_ROOT="${4:-/home/sunil/usb-mnt/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78}"
SCREEN_NAME="glm52-layer3-expert103-rank4-terminal-endpoint-eight-document-screen"
FREEZE_NAME="glm52-layer3-expert103-rank4-terminal-confirmation-freeze"
SCREENING_REPORT="${EXPERIMENT_ROOT}/results/${SCREEN_NAME}/report.json"
CONFIRMATION_FREEZE="${EXPERIMENT_ROOT}/results/${FREEZE_NAME}/confirmation-freeze.json"
REFERENCE_PLAN="${SOURCE_ROOT}/experiments/glm52_terminal_hidden_teacher_reference_plan.json"
DESTINATION="${REFERENCE_ROOT}/confirmation-logits"
RECORD_ROOT="${EXPERIMENT_ROOT}/launch-records/glm52-terminal-confirmation-reference-generation"
IMAGE="verdictai/glm52-exl3-sparkinfer:v39-r28-r7fused-broadcast-cu132-sm120a"
IMAGE_ID="sha256:12f86065d7fe64d30dad678585e68c91f47f1f2a32bed45ccaf108382f3928ac"
REQUIRED_FREE_BYTES=26000000000

test -f "${REFERENCE_PLAN}"
test -f "${REFERENCE_ROOT}/assets/complete.json"
test -f "${SCREENING_REPORT}"
test -f "${CONFIRMATION_FREEZE}"
test -f "${MODEL_ROOT}/tokenizer.json"
test -f "${MODEL_ROOT}/tokenizer_config.json"
test ! -e "${DESTINATION}"
test "$(docker image inspect --format '{{.Id}}' "${IMAGE}")" = "${IMAGE_ID}"
available_bytes=$(df -PB1 "${REFERENCE_ROOT}" | awk 'NR == 2 {print $4}')
if [[ ! "${available_bytes}" =~ ^[0-9]+$ ]] || (( available_bytes < REQUIRED_FREE_BYTES )); then
  echo "confirmation teacher-reference generation needs at least ${REQUIRED_FREE_BYTES} free bytes" >&2
  exit 1
fi
gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ -n "${gpu_processes}" ]]; then
  echo "confirmation teacher-reference generation requires four idle GPUs" >&2
  exit 1
fi

mkdir -p "${RECORD_ROOT}"
python3 "${SOURCE_ROOT}/tools/verify_source_snapshot.py" \
  >"${RECORD_ROOT}/source-snapshot-verification.json"
sha256sum \
  "${REFERENCE_PLAN}" \
  "${REFERENCE_ROOT}/assets/complete.json" \
  "${SCREENING_REPORT}" \
  "${CONFIRMATION_FREEZE}" \
  "${SOURCE_ROOT}/scripts/generate_glm52_terminal_teacher_logits.py" \
  >"${RECORD_ROOT}/bound-inputs-and-source.sha256"

docker run --rm --pull never --network none --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONPATH=/workspace \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -v "${SOURCE_ROOT}:/workspace:ro" \
  -v "${REFERENCE_ROOT}:/reference:rw" \
  -v "${MODEL_ROOT}:/model:ro" \
  -v "${SCREENING_REPORT}:/screening-report.json:ro" \
  -v "${CONFIRMATION_FREEZE}:/confirmation-freeze.json:ro" \
  --entrypoint python3 \
  "${IMAGE}" \
  /workspace/scripts/generate_glm52_terminal_teacher_logits.py \
  --plan /workspace/experiments/glm52_terminal_hidden_teacher_reference_plan.json \
  --assets /reference/assets \
  --tokenizer /model \
  --evaluation-tier confirmation \
  --devices 0,1,2,3 \
  --closure-rows 8 \
  --confirmation-freeze /confirmation-freeze.json \
  --screening-report /screening-report.json \
  --dest /reference/confirmation-logits \
  >"${RECORD_ROOT}/generation-report.json"

test -f "${DESTINATION}/manifest.json"
sha256sum \
  "${DESTINATION}/generation_contract.json" \
  "${DESTINATION}/numerical_closure.json" \
  "${DESTINATION}/manifest.json" \
  >"${RECORD_ROOT}/confirmation-reference.sha256"

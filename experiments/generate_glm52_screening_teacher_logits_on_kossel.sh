#!/usr/bin/env bash
set -euo pipefail

# Generate only the eight screening references.  The thirty-two confirmation
# documents remain outside candidate scoring until a candidate construction,
# factor dtype, and exact byte ledger have been frozen.

SOURCE_ROOT="${1:-/home/sunil/qsrt-glm52-experiments/source/qsrt-terminal-teacher-reference-b4734de}"
REFERENCE_ROOT="${2:-/home/sunil/usb-mnt/qsrt-experiments/glm52-terminal-teacher-reference-b4734de}"
MODEL_ROOT="${3:-/home/sunil/usb-mnt/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78}"
IMAGE="verdictai/glm52-exl3-sparkinfer:v39-r28-r7fused-broadcast-cu132-sm120a"
IMAGE_ID="sha256:12f86065d7fe64d30dad678585e68c91f47f1f2a32bed45ccaf108382f3928ac"
PLAN_RELATIVE="experiments/glm52_terminal_hidden_teacher_reference_plan.json"
REQUIRED_FREE_BYTES=12000000000
TEACHER_LOGIT_DEVICES="${TEACHER_LOGIT_DEVICES:-0,1,2,3}"

test -f "${SOURCE_ROOT}/${PLAN_RELATIVE}"
test -f "${REFERENCE_ROOT}/assets/complete.json"
test -f "${MODEL_ROOT}/tokenizer.json"
test -f "${MODEL_ROOT}/tokenizer_config.json"
test "$(docker image inspect --format '{{.Id}}' "${IMAGE}")" = "${IMAGE_ID}"
available_bytes=$(df -PB1 "${REFERENCE_ROOT}" | awk 'NR == 2 {print $4}')
if [[ ! "${available_bytes}" =~ ^[0-9]+$ ]] || (( available_bytes < REQUIRED_FREE_BYTES )); then
  echo "screening teacher-reference generation needs at least ${REQUIRED_FREE_BYTES} free bytes" >&2
  exit 1
fi
if [[ ! "${TEACHER_LOGIT_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "TEACHER_LOGIT_DEVICES must be a comma-separated GPU index list" >&2
  exit 1
fi
IFS=',' read -r -a teacher_logit_devices <<<"${TEACHER_LOGIT_DEVICES}"
declare -A seen_devices=()
for device in "${teacher_logit_devices[@]}"; do
  if [[ -n "${seen_devices[${device}]:-}" ]]; then
    echo "TEACHER_LOGIT_DEVICES repeats GPU ${device}" >&2
    exit 1
  fi
  seen_devices["${device}"]=1
  gpu_processes=$(nvidia-smi --id="${device}" --query-compute-apps=pid --format=csv,noheader,nounits)
  if [[ -n "${gpu_processes}" ]]; then
    echo "screening teacher-reference generation requires idle GPU ${device}" >&2
    exit 1
  fi
done

docker run --rm --pull never --network none --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONPATH=/workspace \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -v "${SOURCE_ROOT}:/workspace:ro" \
  -v "${REFERENCE_ROOT}:/reference" \
  -v "${MODEL_ROOT}:/model:ro" \
  --entrypoint python3 \
  "${IMAGE}" \
  /workspace/scripts/generate_glm52_terminal_teacher_logits.py \
  --plan "/workspace/${PLAN_RELATIVE}" \
  --assets /reference/assets \
  --tokenizer /model \
  --evaluation-tier screening \
  --devices "${TEACHER_LOGIT_DEVICES}" \
  --closure-rows 8 \
  --dest /reference/screening-logits \
  >"${REFERENCE_ROOT}/screening-logit-generation-report.json"

test -f "${REFERENCE_ROOT}/screening-logits/manifest.json"
sha256sum \
  "${REFERENCE_ROOT}/screening-logits/generation_contract.json" \
  "${REFERENCE_ROOT}/screening-logits/numerical_closure.json" \
  "${REFERENCE_ROOT}/screening-logits/manifest.json" \
  >"${REFERENCE_ROOT}/screening-reference-sha256.txt"

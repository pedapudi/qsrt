#!/usr/bin/env bash
set -euo pipefail

# Freeze the registered correction only after it passes the eight-document
# screen. This CPU-only command does not read confirmation references.

SOURCE_ROOT="${1:-/home/sunil/qsrt-glm52-experiments/source/qsrt-terminal-teacher-reference-b4734de}"
EXPERIMENT_ROOT="${2:-/home/sunil/qsrt-glm52-experiments}"
REFERENCE_ROOT="${3:-/home/sunil/usb-mnt/qsrt-experiments/glm52-terminal-teacher-reference-b4734de}"
ARTIFACT_NAME="glm52-layer3-frozen8-low-rank-down-reconstructed_activation_down_refit-bf16-rank-4-factorized-runtime-v1-merged"
SCREEN_NAME="glm52-layer3-expert103-rank4-terminal-endpoint-eight-document-screen"
FREEZE_NAME="glm52-layer3-expert103-rank4-terminal-confirmation-freeze"
ARTIFACT_ROOT="${EXPERIMENT_ROOT}/results/${ARTIFACT_NAME}"
SCREENING_REPORT="${EXPERIMENT_ROOT}/results/${SCREEN_NAME}/report.json"
FREEZE_ROOT="${EXPERIMENT_ROOT}/results/${FREEZE_NAME}"
REFERENCE_PLAN="${SOURCE_ROOT}/experiments/glm52_terminal_hidden_teacher_reference_plan.json"
REGISTRATION="${SOURCE_ROOT}/experiments/glm52_layer3_rank4_expert103_low_rank_down_confirmation_registration.json"
RECORD_ROOT="${EXPERIMENT_ROOT}/launch-records/${FREEZE_NAME}"
IMAGE="verdictai/glm52-exl3-sparkinfer:v39-r28-r7fused-broadcast-cu132-sm120a"
IMAGE_ID="sha256:12f86065d7fe64d30dad678585e68c91f47f1f2a32bed45ccaf108382f3928ac"

test -f "${ARTIFACT_ROOT}/manifest.json"
test -f "${ARTIFACT_ROOT}/report.json"
test -f "${SCREENING_REPORT}"
test -f "${REFERENCE_PLAN}"
test -f "${REGISTRATION}"
test -f "${SOURCE_ROOT}/scripts/freeze_glm52_terminal_confirmation_candidate.py"
test -f "${REFERENCE_ROOT}/screening-logits/manifest.json"
test ! -e "${FREEZE_ROOT}"
test "$(docker image inspect --format '{{.Id}}' "${IMAGE}")" = "${IMAGE_ID}"

mkdir -p "${RECORD_ROOT}"
python3 "${SOURCE_ROOT}/tools/verify_source_snapshot.py" \
  >"${RECORD_ROOT}/source-snapshot-verification.json"
sha256sum \
  "${ARTIFACT_ROOT}/manifest.json" \
  "${ARTIFACT_ROOT}/report.json" \
  "${SCREENING_REPORT}" \
  "${REFERENCE_PLAN}" \
  "${REGISTRATION}" \
  "${REFERENCE_ROOT}/screening-logits/manifest.json" \
  "${SOURCE_ROOT}/scripts/freeze_glm52_terminal_confirmation_candidate.py" \
  >"${RECORD_ROOT}/bound-inputs-and-source.sha256"

frozen_at_utc="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
docker run --rm --pull never --network none \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONPATH=/workspace \
  -v "${SOURCE_ROOT}:/workspace:ro" \
  -v "${ARTIFACT_ROOT}:/artifact:ro" \
  -v "${SCREENING_REPORT}:/screening-report.json:ro" \
  -v "${REFERENCE_PLAN}:/reference-plan.json:ro" \
  -v "${REGISTRATION}:/candidate-registration.json:ro" \
  -v "${EXPERIMENT_ROOT}/results:/results:rw" \
  --entrypoint python3 \
  "${IMAGE}" \
  /workspace/scripts/freeze_glm52_terminal_confirmation_candidate.py \
  --intervention-artifact /artifact \
  --candidate-registration /candidate-registration.json \
  --screening-report /screening-report.json \
  --teacher-reference-plan /reference-plan.json \
  --frozen-at-utc "${frozen_at_utc}" \
  --dest "/results/${FREEZE_NAME}" \
  >"${RECORD_ROOT}/freeze-report.json"

test -f "${FREEZE_ROOT}/manifest.json"
test -f "${FREEZE_ROOT}/confirmation-freeze.json"
sha256sum \
  "${FREEZE_ROOT}/manifest.json" \
  "${FREEZE_ROOT}/confirmation-freeze.json" \
  "${FREEZE_ROOT}/layer-003-expert-103-down-rank-4-bf16.safetensors" \
  >"${RECORD_ROOT}/frozen-candidate.sha256"

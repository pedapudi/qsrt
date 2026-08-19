#!/usr/bin/env bash
set -euo pipefail

# Download only the selected terminal-hidden rows and two official endpoint
# tensors.  This launcher never requests the complete BF16 checkpoint or the
# complete 12.9 GB terminal-hidden-state file.

SOURCE_ROOT="${1:-/home/sunil/qsrt-glm52-experiments/source/qsrt-terminal-teacher-reference-b4734de}"
REFERENCE_ROOT="${2:-/home/sunil/usb-mnt/qsrt-experiments/glm52-terminal-teacher-reference-b4734de}"
IMAGE="verdictai/glm52-exl3-sparkinfer:v39-r28-r7fused-broadcast-cu132-sm120a"
IMAGE_ID="sha256:12f86065d7fe64d30dad678585e68c91f47f1f2a32bed45ccaf108382f3928ac"
PLAN_RELATIVE="experiments/glm52_terminal_hidden_teacher_reference_plan.json"
REQUIRED_FREE_BYTES=8000000000

test -f "${SOURCE_ROOT}/${PLAN_RELATIVE}"
test "$(docker image inspect --format '{{.Id}}' "${IMAGE}")" = "${IMAGE_ID}"
mkdir -p "${REFERENCE_ROOT}"
available_bytes=$(df -PB1 "${REFERENCE_ROOT}" | awk 'NR == 2 {print $4}')
if [[ ! "${available_bytes}" =~ ^[0-9]+$ ]] || (( available_bytes < REQUIRED_FREE_BYTES )); then
  echo "terminal teacher-reference download needs at least ${REQUIRED_FREE_BYTES} free bytes" >&2
  exit 1
fi

docker run --rm --pull never --network host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONPATH=/workspace \
  -v "${SOURCE_ROOT}:/workspace:ro" \
  -v "${REFERENCE_ROOT}:/reference" \
  --entrypoint python3 \
  "${IMAGE}" \
  /workspace/scripts/download_glm52_terminal_teacher_assets.py \
  --plan "/workspace/${PLAN_RELATIVE}" \
  --dest /reference/assets \
  --jobs 8 \
  --timeout-seconds 120 \
  --retries 8 \
  >"${REFERENCE_ROOT}/asset-download-report.json"

test -f "${REFERENCE_ROOT}/assets/complete.json"
sha256sum \
  "${REFERENCE_ROOT}/assets/download_contract.json" \
  "${REFERENCE_ROOT}/assets/complete.json" \
  >"${REFERENCE_ROOT}/asset-receipt-sha256.txt"

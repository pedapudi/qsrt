#!/usr/bin/env bash
# Capture every routed row from a fixed corpus.
set -euo pipefail

QSRT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_ROOT="${VLLM_ROOT:-/home/luke/projects/vllm}"
PYTHON="${QSRT_CAPTURE_PYTHON:-${VLLM_ROOT}/.venv/bin/python}"
RESIDENT="${QSRT_CAPTURE_RESIDENT:-/data/releases/Kimi-K3-QSRT-3p08-COUPLED-HADAMARD-DRAWS-0-7-v1}"
SOURCE_ID="coupled_qsrt_k3x22_k4x2"
RUN_ID="${QSRT_CAPTURE_RUN_ID:-k3-all-routed-4m-v1}"
TARGET_TOKENS="${QSRT_CAPTURE_TARGET_TOKENS:-4000000}"
CAPTURE="${QSRT_CAPTURE_ROOT:-/data/datasets/kquant/captures/${RUN_ID}.kqrows}"
REPORT="${QSRT_CAPTURE_REPORT:-/data/datasets/kquant/captures/${RUN_ID}-corpus.json}"
FINALIZE="${QSRT_CAPTURE_FINALIZE:-/data/datasets/kquant/captures/${RUN_ID}.finalize}"
SERVER_LOG="${QSRT_CAPTURE_SERVER_LOG:-/data/datasets/kquant/captures/${RUN_ID}.server.log}"
CORPUS_LOG="${QSRT_CAPTURE_CORPUS_LOG:-/data/datasets/kquant/captures/${RUN_ID}.corpus.log}"
SERVER_PID_FILE="${QSRT_CAPTURE_SERVER_PID_FILE:-/data/datasets/kquant/captures/${RUN_ID}.server.pid}"
CORPUS_PID_FILE="${QSRT_CAPTURE_CORPUS_PID_FILE:-/data/datasets/kquant/captures/${RUN_ID}.corpus.pid}"

corpus_args=(
  --source /data/datasets/kquant/corpora/k3-broad-external-v1/fineweb-edu-sample10bt.jsonl=0.35@3072
  --source /data/datasets/kquant/corpora/k3-hybrid-v2-train-by-source-v1/local-deep-calib.jsonl=0.05@4096
  --source /data/datasets/kquant/corpora/k3-hybrid-v2-train-by-source-v1/local-diverse-calib.jsonl=0.15@2048
  --source /data/datasets/kquant/corpora/k3-hybrid-v2-train-by-source-v1/ultrachat.jsonl=0.15@2048
  --source /data/datasets/kquant/corpora/k3-broad-external-v1/open-web-math.jsonl=0.125@3072
  --source /data/datasets/kquant/corpora/k3-hybrid-v2-train-by-source-v1/swe-agent.jsonl=0.04@4096
  --source /data/datasets/kquant/corpora/k3-hybrid-v2-train-by-source-v1/swe-openhands.jsonl=0.03@4096
  --source /data/datasets/kquant/corpora/k3-hybrid-v2-train-by-source-v1/apigen-mt.jsonl=0.02@2048
  --source /data/datasets/kquant/corpora/k3-hybrid-v2-train-by-source-v1/toolace.jsonl=0.01@4096
  --source /data/datasets/kquant/corpora/k3-4m-v1/reap_recall_calib.jsonl=0.05@4096
  --source /data/datasets/kquant/corpora/k3-broad-external-v1/fineweb2-cmn_Hani.jsonl=0.025@2048
  --target-tokens "${TARGET_TOKENS}"
  --max-prompt-tokens 4096
  --min-prompt-tokens 64
  --fold-modulus 1
  --fold-index 0
  --fold-mode include
  --seed 20260812
  --exclude-report "${QSRT_ROOT}/out/k3-denseh-broad-v6-1m-train-corpus.json"
  --exclude-report "${QSRT_ROOT}/out/k3-denseh-broad-v5-selection-corpus.json"
  --exclude-report "${QSRT_ROOT}/out/k3-denseh-broad-v5-final-validation-corpus.json"
  --exclude-report "${QSRT_ROOT}/out/k3-denseh-broad-v3-selection-corpus.json"
  --exclude-report "${QSRT_ROOT}/out/k3-denseh-broad-v3-final-validation-corpus.json"
  --model-dir "${RESIDENT}"
  --expected-capture-source "${SOURCE_ID}"
  --capture-dir "${CAPTURE}"
  --report "${REPORT}"
  --finalize-file "${FINALIZE}"
  --allow-other-teacher-layout
)

for token_file in /data/datasets/kld/k3/tokens/window-*.json; do
  corpus_args+=(--exclude-token-file "${token_file}")
done

pid_alive() {
  local path="$1"
  [[ -f "${path}" ]] && kill -0 "$(<"${path}")" 2>/dev/null
}

clear_stale_pid_file() {
  local path="$1"
  if [[ -f "${path}" ]] && ! pid_alive "${path}"; then
    rm -- "${path}"
  fi
}

plan() {
  [[ -d "${RESIDENT}" ]] || { echo "missing resident checkpoint: ${RESIDENT}" >&2; return 1; }
  [[ -x "${PYTHON}" ]] || { echo "missing Python: ${PYTHON}" >&2; return 1; }
  [[ ! -e "${REPORT}" ]] || { echo "corpus report already exists: ${REPORT}" >&2; return 1; }
  mkdir -p "$(dirname -- "${REPORT}")"
  "${PYTHON}" "${QSRT_ROOT}/scripts/run_interim_calibration_corpus.py" \
    "${corpus_args[@]}" --model Kimi-K3 --dry-run >/dev/null
  jq '{plan_sha256,planned_tokens,planned_requests,sources,excluded_token_files}' "${REPORT}"
}

preflight() {
  [[ -d "${RESIDENT}" ]] || { echo "missing resident checkpoint: ${RESIDENT}" >&2; return 1; }
  [[ -x "${PYTHON}" ]] || { echo "missing Python: ${PYTHON}" >&2; return 1; }
  [[ -x "${VLLM_ROOT}/serve-kimi-k3-qsrt.sh" ]] || { echo "missing vLLM launcher" >&2; return 1; }
  [[ -f "${REPORT}" ]] || { echo "missing immutable corpus plan: ${REPORT}" >&2; return 1; }
  jq -e --arg source "${SOURCE_ID}" --argjson target "${TARGET_TOKENS}" '
    .kind == "qsrt_interim_calibration_corpus_run" and
    .expected_capture_source == $source and
    .target_tokens == $target and .planned_tokens == $target and
    (.completed_requests | type == "number") and
    .completed_requests >= 0 and
    .completed_requests <= .planned_requests and
    .finalized == false and
    (.plan_sha256 | type == "string" and length == 64) and
    (.excluded_token_files | length == 32)
  ' "${REPORT}" >/dev/null
  if [[ -e "${CAPTURE}" ]]; then
    jq -e --arg plan "$(jq -r .plan_sha256 "${REPORT}")" '
      .kind == "qsrt_all_routed_rows" and .complete == false and
      .corpus_manifest_sha256 == $plan
    ' "${CAPTURE}/manifest.json" >/dev/null
  else
    jq -e '.completed_requests == 0 and .reported_prompt_tokens == 0' \
      "${REPORT}" >/dev/null
  fi
  [[ ! -e "${FINALIZE}" ]] || { echo "finalize sentinel exists: ${FINALIZE}" >&2; return 1; }
  clear_stale_pid_file "${SERVER_PID_FILE}"
  clear_stale_pid_file "${CORPUS_PID_FILE}"
  [[ ! -e "${SERVER_PID_FILE}" ]] || { echo "server PID file exists: ${SERVER_PID_FILE}" >&2; return 1; }
  [[ ! -e "${CORPUS_PID_FILE}" ]] || { echo "corpus PID file exists: ${CORPUS_PID_FILE}" >&2; return 1; }
  if ss -ltn '( sport = :8000 )' | tail -n +2 | grep -q .; then
    echo "port 8000 is already in use" >&2
    return 1
  fi
  [[ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -eq 12 ]] || {
    echo "the TP12 capture requires twelve visible GPUs" >&2
    return 1
  }
  nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader
  df -h /data
}

start() {
  preflight
  mkdir -p "$(dirname -- "${CAPTURE}")"
  setsid env \
    K3_MODEL_DIR="${RESIDENT}" \
    K3_SERVED_MODEL_NAME=Kimi-K3 \
    K3_ENABLE_DSPARK=0 \
    K3_LANGUAGE_MODEL_ONLY=1 \
    K3_ENFORCE_EAGER=1 \
    K3_ENABLE_PREFIX_CACHE=0 \
    K3_TP_SIZE=12 \
    K3_MAX_NUM_SEQS=1 \
    K3_MAX_NUM_BATCHED_TOKENS=4096 \
    K3_MAX_MODEL_LEN=8192 \
    K3_KV_CACHE_MEMORY_BYTES=1073741824 \
    VLLM_KQUANT_CAPTURE_DIR="${CAPTURE}" \
    VLLM_KQUANT_CAPTURE_PROFILE=all_routed_rows \
    VLLM_KQUANT_CORPUS="${REPORT}" \
    VLLM_KQUANT_TEACHER_CHECKPOINT="${RESIDENT}" \
    VLLM_KQUANT_SOURCE="${SOURCE_ID}" \
    VLLM_KQUANT_CAPTURE_RUN_ID="${RUN_ID}" \
    VLLM_KQUANT_EXPECTED_ROWS="${TARGET_TOKENS}" \
    VLLM_KQUANT_CHUNK_ROWS=16384 \
    VLLM_KQUANT_WRITER_PROCESSES=4 \
    VLLM_KQUANT_WRITER_QUEUE_DEPTH=2 \
    VLLM_KQUANT_FINALIZE_FILE="${FINALIZE}" \
    "${VLLM_ROOT}/serve-kimi-k3-qsrt.sh" >"${SERVER_LOG}" 2>&1 &
  echo "$!" >"${SERVER_PID_FILE}"

  setsid "${PYTHON}" "${QSRT_ROOT}/scripts/run_interim_calibration_corpus.py" \
    "${corpus_args[@]}" --model Kimi-K3 --resume >"${CORPUS_LOG}" 2>&1 &
  echo "$!" >"${CORPUS_PID_FILE}"
  echo "server PID $(<"${SERVER_PID_FILE}"); corpus PID $(<"${CORPUS_PID_FILE}")"
}

status() {
  if pid_alive "${SERVER_PID_FILE}"; then
    echo "server: running (PID $(<"${SERVER_PID_FILE}"))"
  else
    echo "server: stopped"
  fi
  if pid_alive "${CORPUS_PID_FILE}"; then
    echo "corpus: running (PID $(<"${CORPUS_PID_FILE}"))"
  else
    echo "corpus: stopped"
  fi
  [[ ! -f "${REPORT}" ]] || jq \
    '{plan_sha256,completed_requests,planned_requests,reported_prompt_tokens,planned_tokens,finalized}' \
    "${REPORT}"
  if [[ -f "${CAPTURE}/manifest.json" ]]; then
    jq '{source,resident_checkpoint,complete,sealed_request_index,rows}' \
      "${CAPTURE}/manifest.json"
    du -sh "${CAPTURE}"
  fi
  [[ ! -f "${SERVER_LOG}" ]] || tail -n 12 "${SERVER_LOG}"
  [[ ! -f "${CORPUS_LOG}" ]] || tail -n 12 "${CORPUS_LOG}"
}

stop_one() {
  local path="$1"
  local label="$2"
  if pid_alive "${path}"; then
    local pid
    pid="$(<"${path}")"
    echo "stopping ${label} process group ${pid}"
    kill -TERM -- "-${pid}"
    for _ in $(seq 1 120); do
      if ! kill -0 -- "-${pid}" 2>/dev/null; then
        rm -- "${path}"
        echo "${label}: stopped"
        return 0
      fi
      sleep 0.25
    done
    echo "${label} process group ${pid} did not stop after 30 seconds" >&2
    return 1
  fi
  rm -f -- "${path}"
}

case "${1:-status}" in
  plan) plan ;;
  preflight) preflight ;;
  start) start ;;
  status) status ;;
  stop)
    stop_one "${CORPUS_PID_FILE}" corpus
    stop_one "${SERVER_PID_FILE}" server
    ;;
  *)
    echo "usage: $0 {plan|preflight|start|status|stop}" >&2
    exit 2
    ;;
esac

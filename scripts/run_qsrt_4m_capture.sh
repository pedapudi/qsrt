#!/usr/bin/env bash
# Plan, launch, inspect, or stop the source-pinned 4M pure-QSRT capture.
set -euo pipefail

QSRT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_ROOT="${VLLM_ROOT:-/home/luke/projects/vllm}"
CORPUS_PYTHON="${QSRT_4M_CORPUS_PYTHON:-${VLLM_ROOT}/.venv/bin/python}"
TEACHER="${QSRT_4M_TEACHER:-/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-PURE-v1-model}"
CAPTURE="${QSRT_4M_CAPTURE:-/data/kquant/captures/k3-denseh-broad-v7-4m-train.kqcapture}"
REPORT="${QSRT_4M_REPORT:-${QSRT_ROOT}/out/k3-denseh-broad-v7-4m-train-corpus.json}"
INTEGRITY="${QSRT_4M_INTEGRITY:-${QSRT_ROOT}/out/k3-denseh-broad-v7-4m-integrity.json}"
FINALIZE="${QSRT_4M_FINALIZE:-/data/kquant/captures/k3-denseh-broad-v7-4m-train.finalize}"
SERVER_LOG="${QSRT_4M_SERVER_LOG:-/data/kquant/captures/k3-denseh-broad-v7-4m-train.server.log}"
CORPUS_LOG="${QSRT_4M_CORPUS_LOG:-/data/kquant/captures/k3-denseh-broad-v7-4m-train.corpus.log}"
SERVER_PID_FILE="${QSRT_4M_SERVER_PID_FILE:-/data/kquant/captures/k3-denseh-broad-v7-4m-train.server.pid}"
CORPUS_PID_FILE="${QSRT_4M_CORPUS_PID_FILE:-/data/kquant/captures/k3-denseh-broad-v7-4m-train.corpus.pid}"

corpus_args=(
  --source /data/kquant/corpora/k3-4m-v1/reap_recall_calib.jsonl=0.25@4096
  --source /data/kquant/corpora/k3-broad-external-v1/fineweb-edu-sample10bt.jsonl=0.27@2048
  --source /data/kquant/corpora/k3-broad-external-v1/open-web-math.jsonl=0.125@3072
  --source /data/kquant/corpora/k3-hybrid-v2-train-by-source-v1/ultrachat.jsonl=0.10@2048
  --source /home/luke/projects/quantization/data/text/diverse_calib.jsonl=0.10@1024
  --source /home/luke/projects/quantization/data/text/deep_calib.jsonl=0.05@4096
  --source /data/kquant/corpora/k3-broad-external-v1/fineweb2-cmn_Hani.jsonl=0.025@1024
  --source /data/kquant/corpora/k3-hybrid-v2-train-by-source-v1/swe-agent.jsonl=0.04@4096
  --source /data/kquant/corpora/k3-hybrid-v2-train-by-source-v1/swe-openhands.jsonl=0.02@4096
  --source /data/kquant/corpora/k3-hybrid-v2-train-by-source-v1/apigen-mt.jsonl=0.015@2048
  --source /data/kquant/corpora/k3-hybrid-v2-train-by-source-v1/toolace.jsonl=0.005@4096
  --target-tokens 4000000
  --max-prompt-tokens 2048
  --min-prompt-tokens 64
  --fold-modulus 4
  --fold-index 0
  --fold-mode exclude
  --seed 20260808
  --exclude-report "${QSRT_ROOT}/out/k3-denseh-broad-v6-1m-train-corpus.json"
  --exclude-report "${QSRT_ROOT}/out/k3-denseh-broad-v5-selection-corpus.json"
  --exclude-report "${QSRT_ROOT}/out/k3-denseh-broad-v5-final-validation-corpus.json"
  --exclude-report "${QSRT_ROOT}/out/k3-denseh-broad-v3-selection-corpus.json"
  --exclude-report "${QSRT_ROOT}/out/k3-denseh-broad-v3-final-validation-corpus.json"
  --model-dir "${TEACHER}"
  --expected-capture-source pure_qsrt_sqg_xor_cheb_t12
  --capture-dir "${CAPTURE}"
  --report "${REPORT}"
  --finalize-file "${FINALIZE}"
)

pid_alive() {
  local path="$1"
  [[ -f "${path}" ]] && kill -0 "$(<"${path}")" 2>/dev/null
}

preflight() {
  [[ -d "${TEACHER}" ]] || { echo "missing teacher: ${TEACHER}" >&2; return 1; }
  [[ -x "${CORPUS_PYTHON}" ]] || { echo "missing Python: ${CORPUS_PYTHON}" >&2; return 1; }
  [[ -x "${VLLM_ROOT}/serve-kimi-k3-exl3-3p09-tp12.sh" ]] || {
    echo "missing vLLM launcher" >&2
    return 1
  }
  [[ -f "${REPORT}" ]] || { echo "missing frozen plan: ${REPORT}" >&2; return 1; }
  [[ -f "${INTEGRITY}" ]] || { echo "missing integrity receipt: ${INTEGRITY}" >&2; return 1; }
  jq -e '
    .expected_capture_source == "pure_qsrt_sqg_xor_cheb_t12" and
    .target_tokens == 4000000 and .planned_tokens == 4000000 and
    .completed_requests == 0 and .finalized == false
  ' "${REPORT}" >/dev/null
  jq -e '.status == "pass" and (.issues | length) == 0' "${INTEGRITY}" >/dev/null
  [[ ! -e "${CAPTURE}" ]] || { echo "capture already exists: ${CAPTURE}" >&2; return 1; }
  [[ ! -e "${FINALIZE}" ]] || { echo "finalize sentinel already exists: ${FINALIZE}" >&2; return 1; }
  [[ ! -e "${SERVER_LOG}" ]] || { echo "server log already exists: ${SERVER_LOG}" >&2; return 1; }
  [[ ! -e "${CORPUS_LOG}" ]] || { echo "corpus log already exists: ${CORPUS_LOG}" >&2; return 1; }
  [[ ! -e "${SERVER_PID_FILE}" ]] || { echo "server PID file exists: ${SERVER_PID_FILE}" >&2; return 1; }
  [[ ! -e "${CORPUS_PID_FILE}" ]] || { echo "corpus PID file exists: ${CORPUS_PID_FILE}" >&2; return 1; }
  if ss -ltn '( sport = :8000 )' | tail -n +2 | grep -q .; then
    echo "port 8000 is already in use" >&2
    return 1
  fi
  local gpu_count
  gpu_count="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)"
  [[ "${gpu_count}" -eq 12 ]] || { echo "expected 12 GPUs, found ${gpu_count}" >&2; return 1; }
  nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader
  df -h /data
}

start() {
  preflight
  mkdir -p "$(dirname -- "${CAPTURE}")"
  setsid env \
    K3_MODEL_DIR="${TEACHER}" \
    K3_SERVED_MODEL_NAME=kimi-k3-qsrt \
    K3_QSRT_CAPTURE_DIR="${CAPTURE}" \
    K3_QSRT_CORPUS="${REPORT}" \
    VLLM_QSRT_SOURCE=pure_qsrt_sqg_xor_cheb_t12 \
    VLLM_QSRT_CAPTURE_RUN_ID=k3-denseh-broad-v7-4m-train \
    VLLM_QSRT_MOMENT_SAMPLE_RATE=16 \
    VLLM_QSRT_INPUT_HESSIAN_SAMPLE_RATE=64 \
    VLLM_QSRT_MID_HESSIAN_SAMPLE_RATE=32 \
    VLLM_QSRT_SAMPLE_CAPACITY=1024 \
    VLLM_QSRT_SAMPLE_SAVE_EVERY=32 \
    VLLM_QSRT_SAMPLE_FLUSH_BYTES=268435456 \
    VLLM_QSRT_FINALIZE_FILE="${FINALIZE}" \
    "${VLLM_ROOT}/serve-kimi-k3-exl3-3p09-tp12.sh" \
    >"${SERVER_LOG}" 2>&1 &
  echo "$!" >"${SERVER_PID_FILE}"

  setsid "${CORPUS_PYTHON}" \
    "${QSRT_ROOT}/scripts/run_interim_calibration_corpus.py" \
    "${corpus_args[@]}" --model kimi-k3-qsrt --resume \
    >"${CORPUS_LOG}" 2>&1 &
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
  if [[ -f "${REPORT}" ]]; then
    jq '{completed_requests,planned_requests,reported_prompt_tokens,planned_tokens,finalized}' "${REPORT}"
  fi
  if [[ -f "${CAPTURE}/manifest.json" ]]; then
    jq '{source,teacher_checkpoint,run_id,complete,dropped_sample_rows,sampling}' \
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
  fi
}

case "${1:-status}" in
  preflight) preflight ;;
  start) start ;;
  status) status ;;
  stop)
    stop_one "${CORPUS_PID_FILE}" corpus
    stop_one "${SERVER_PID_FILE}" server
    ;;
  *)
    echo "usage: $0 {preflight|start|status|stop}" >&2
    exit 2
    ;;
esac

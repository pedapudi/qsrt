#!/usr/bin/env bash
# Run the small eager-only witness that authenticates QSRT capture hook values.
set -euo pipefail

QSRT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_ROOT="${VLLM_ROOT:-/home/luke/projects/vllm}"
CORPUS_PYTHON="${QSRT_CAPTURE_WITNESS_CORPUS_PYTHON:-${VLLM_ROOT}/.venv/bin/python}"
PROOF_ROOT="${QSRT_CAPTURE_WITNESS_ROOT:-/data/kquant/proofs/k3-capture-eager-witness-v3}"
CAPTURE="${PROOF_ROOT}.kqcapture"
CACHE="${PROOF_ROOT}.kqsamples"
TRACE="${PROOF_ROOT}.trace"
FINALIZE="${PROOF_ROOT}.finalize"
SERVER_LOG="${PROOF_ROOT}.server.log"
REPORT="${QSRT_CAPTURE_WITNESS_REPORT:-${QSRT_ROOT}/out/k3-capture-eager-witness-v3-corpus.json}"
RECEIPT="${QSRT_CAPTURE_WITNESS_RECEIPT:-${QSRT_ROOT}/out/k3-capture-eager-witness-v3.json}"
SOURCE="${QSRT_CAPTURE_WITNESS_SOURCE:-/home/luke/projects/quantization/data/text/diverse_calib.jsonl}"
TEACHER="${QSRT_CAPTURE_WITNESS_TEACHER:-/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-PURE-v1-model}"

for path in "${CAPTURE}" "${CACHE}" "${TRACE}" "${FINALIZE}" "${REPORT}" "${RECEIPT}"; do
  if [[ -e "${path}" ]]; then
    echo "ERROR: witness output already exists: ${path}" >&2
    exit 1
  fi
done
if ss -ltn '( sport = :8000 )' | tail -n +2 | grep -q .; then
  echo "ERROR: port 8000 is already in use" >&2
  exit 1
fi
if [[ ! -f "${SOURCE}" ]]; then
  echo "ERROR: witness source does not exist: ${SOURCE}" >&2
  exit 1
fi
mkdir -p "$(dirname -- "${PROOF_ROOT}")" "$(dirname -- "${REPORT}")"

SERVER_PID=""
stop_server() {
  if [[ -z "${SERVER_PID}" ]] || ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    return
  fi
  kill -TERM -- "-${SERVER_PID}"
  for _ in $(seq 1 120); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      wait "${SERVER_PID}" 2>/dev/null || true
      return
    fi
    sleep 1
  done
  echo "ERROR: witness vLLM process group did not stop after 120 seconds" >&2
  return 1
}
trap stop_server EXIT INT TERM

echo "Starting eager pure-QSRT teacher for the capture witness; log: ${SERVER_LOG}" >&2
setsid env \
  K3_MODEL_DIR="${TEACHER}" \
  K3_QSRT_CAPTURE_DIR="${CAPTURE}" \
  K3_QSRT_CORPUS="${REPORT}" \
  VLLM_QSRT_SOURCE="pure_qsrt_sqg_xor_cheb_t12" \
  VLLM_QSRT_CAPTURE_RUN_ID="k3-capture-eager-witness-v3" \
  VLLM_QSRT_INPUT_HESSIAN_SAMPLE_RATE=1 \
  VLLM_QSRT_MID_HESSIAN_SAMPLE_RATE=1073741824 \
  VLLM_QSRT_MOMENT_SAMPLE_RATE=1073741824 \
  VLLM_QSRT_SAMPLE_CAPACITY=512 \
  VLLM_QSRT_SAMPLE_SAVE_EVERY=1 \
  VLLM_QSRT_FINALIZE_FILE="${FINALIZE}" \
  KIMI_CORRECTNESS_TRACE_DIR="${TRACE}" \
  KIMI_CORRECTNESS_TRACE_LAYERS="1,24,92" \
  KIMI_CORRECTNESS_TRACE_RANKS=0 \
  KIMI_CORRECTNESS_TRACE_START_CALL=0 \
  KIMI_CORRECTNESS_TRACE_MAX_CALLS=16 \
  KIMI_CORRECTNESS_TRACE_MAX_TOKENS=128 \
  KIMI_CORRECTNESS_TRACE_TOKEN_WINDOW=head \
  "${VLLM_ROOT}/serve-kimi-k3-exl3-3p09-tp12.sh" \
  --enforce-eager >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cd "${QSRT_ROOT}"
"${CORPUS_PYTHON}" scripts/run_interim_calibration_corpus.py \
  --source "${SOURCE}" \
  --target-tokens 256 \
  --max-prompt-tokens 128 \
  --min-prompt-tokens 64 \
  --fold-modulus 8 \
  --fold-index 0 \
  --fold-mode exclude \
  --seed 20260808 \
  --model-dir "${TEACHER}" \
  --expected-capture-source pure_qsrt_sqg_xor_cheb_t12 \
  --allow-other-teacher-layout \
  --capture-dir "${CAPTURE}" \
  --report "${REPORT}" \
  --finalize-file "${FINALIZE}"

stop_server
SERVER_PID=""

"${QSRT_ROOT}/.venv/bin/python" scripts/build_qsrt_sample_cache.py \
  "${CAPTURE}" "${CACHE}"
"${QSRT_ROOT}/.venv/bin/python" scripts/validate_qsrt_capture_witness.py \
  --capture "${CAPTURE}" \
  --sample-cache "${CACHE}" \
  --trace-dir "${TRACE}" \
  --layers 1,24,92 \
  --output "${RECEIPT}"

echo "Capture witness passed: ${RECEIPT}" >&2

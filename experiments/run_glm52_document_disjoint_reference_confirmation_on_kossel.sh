#!/usr/bin/env bash
set -euo pipefail

# Evaluate the frozen expert-103 correction on public BF16 references from
# documents that did not contribute to fitting, attribution, or selection.

experiment_root="/home/sunil/qsrt-glm52-experiments"
results_root="${experiment_root}/results"
artifact_name="glm52-layer3-frozen8-low-rank-down-reconstructed_activation_down_refit-bf16-rank-4-factorized-runtime-v1-merged"
result_name="${artifact_name}-frozen-expert103-document-disjoint-public-reference-auxiliary"
artifact_root="${results_root}/${artifact_name}"
result_root="${results_root}/${result_name}"
reference_root="${experiment_root}/reference/glm52-unsloth-document-disjoint-auxiliary-v1"
reference_directory="${reference_root}/reference-logprobs"
reference_plan="${reference_root}/selection-plan.json"
reference_receipt="${reference_root}/receipt.json"
control_root="${experiment_root}/runtime-control/${result_name}"
record_root="${experiment_root}/launch-records/${result_name}"
source_root="${experiment_root}/source/qsrt-working-tree"
registration="${source_root}/experiments/glm52_layer3_rank4_expert103_low_rank_down_confirmation_registration.json"
model="${experiment_root}/model-cache/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78-nvme"
runtime_cache="${experiment_root}/runtime-cache/glm52-per-expert-exl3-without-fused-staging-dense-triton-bf16-reference-kld"
container_name="qsrt-${result_name}"
image="verdictai/glm52-exl3-sparkinfer:v39-r28-r7fused-broadcast-cu132-sm120a"

test -f "${artifact_root}/manifest.json"
test -f "${artifact_root}/report.json"
test -f "${reference_plan}"
test -f "${reference_receipt}"
test -f "${registration}"
test -d "${model}"
test ! -e "${result_root}"
test ! -e "${control_root}"
test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"

python3 - "${reference_plan}" "${reference_receipt}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
receipt = json.loads(Path(sys.argv[2]).read_text())
plan = json.loads(plan_path.read_text())
assert plan["status"] == "frozen_before_reference_file_download"
assert plan["selected_document_count"] == 16
assert receipt["status"] == "complete"
assert receipt["file_count"] == 16
assert receipt["model_weights_downloaded"] is False
assert receipt["selection_plan_sha256"] == hashlib.sha256(plan_path.read_bytes()).hexdigest()
PY

artifact_manifest_sha256="$(
  python3 - "${artifact_root}/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["status"] == "complete"
assert report["experiment"] == "qsrt_glm52_activation_weighted_down_adapter_v1"
assert report["layer"] == 3
assert report["expert_count"] == 8
print(report["manifest_sha256"])
PY
)"
test "${#artifact_manifest_sha256}" -eq 64

first_reference="$(jq -r '.selected_chunks[0].reference_file' "${reference_plan}")"
test -f "${reference_directory}/${first_reference}"
mkdir -p "${control_root}" "${record_root}/source-snapshot/qsrt" "${record_root}/source-snapshot/scripts" "${record_root}/source-snapshot/experiments" "${runtime_cache}"
ln -s "/reference/${first_reference}" "${control_root}/current-reference.safetensors"

cp \
  "${source_root}/qsrt/glm52_document_disjoint_confirmation.py" \
  "${source_root}/qsrt/glm52_engine_kld.py" \
  "${source_root}/qsrt/glm52_expert_intervention_runtime.py" \
  "${source_root}/qsrt/glm52_paired_kld.py" \
  "${record_root}/source-snapshot/qsrt/"
cp \
  "${source_root}/scripts/run_glm52_document_disjoint_reference_confirmation.py" \
  "${record_root}/source-snapshot/scripts/"
cp \
  "${source_root}/experiments/glm52_layer3_rank4_expert103_low_rank_down_confirmation_registration.json" \
  "${source_root}/experiments/glm52_public_reference_auxiliary_plan.json" \
  "${source_root}/experiments/glm52_source_teacher_weight_identity.json" \
  "${source_root}/experiments/run_glm52_document_disjoint_reference_confirmation_on_kossel.sh" \
  "${record_root}/source-snapshot/experiments/"
cp "${reference_receipt}" "${record_root}/reference-download-receipt.json"

docker create \
  --name "${container_name}" \
  --label qsrt.experiment="glm52-document-disjoint-public-reference-auxiliary" \
  --label qsrt.model-downloads-performed=false \
  --label qsrt.reference-document-count=16 \
  --label qsrt.reference-context-tokens=512 \
  --label qsrt.exl3-moe-execution=three-gemm-per-expert-correctness \
  --label qsrt.candidate-runtime=stored-low-rank-factors-materialized-at-load \
  --network none \
  --gpus all \
  --ipc host \
  --shm-size 64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --entrypoint /opt/venv/bin/python \
  -e PYTHONPATH=/workspace/qsrt/runtime/glm52_expert_intervention:/workspace/qsrt \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 \
  -e QSRT_GLM52_INTERVENTION_ROOT=/artifact \
  -e QSRT_GLM52_INTERVENTION_CONTROL=/control/control.json \
  -e QSRT_GLM52_INTERVENTION_MANIFEST_SHA256="${artifact_manifest_sha256}" \
  -e QSRT_GLM52_FORCE_PER_EXPERT_EXL3_MOE=1 \
  -e QSRT_GLM52_ENGINE_KLD_REFERENCE_PATH=/control/current-reference.safetensors \
  -e QSRT_GLM52_ENGINE_KLD_REFERENCE_KEY=logprobs \
  -e QSRT_GLM52_ENGINE_KLD_REFERENCE_REPRESENTATION=logprobs \
  -e QSRT_GLM52_ENGINE_KLD_CHUNK_ROWS=4 \
  -e HF_HOME=/hf-corpus \
  -e HF_HUB_CACHE=/hf-corpus/hub \
  -e HF_DATASETS_CACHE=/hf-corpus/datasets \
  -e KLD_PYDEPS=/kld-pydeps \
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
  -e VLLM_EXL3_R7_FUSED_LAYERS=48 \
  -e VLLM_EXL3_R7_A1_MIN_ROWS=0 \
  -e VLLM_EXL3_PREFILL_CAPACITY=256 \
  -e VLLM_EXL3_PREFILL_BLOCK_M=64 \
  -e KV_FP8_ROPE=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -v "${model}:/model:ro" \
  -v "${reference_directory}:/reference:ro" \
  -v "${reference_plan}:/reference-plan.json:ro" \
  -v "${artifact_root}:/artifact:ro" \
  -v "${registration}:/confirmation-registration.json:ro" \
  -v "${source_root}:/workspace/qsrt:ro" \
  -v "${control_root}:/control:rw" \
  -v "${results_root}:/results:rw" \
  -v "${experiment_root}/corpus-cache/huggingface:/hf-corpus:rw" \
  -v "${experiment_root}/dependencies/kld-datasets-pydeps:/kld-pydeps:ro" \
  -v "${runtime_cache}:/cache:rw" \
  "${image}" \
  /workspace/qsrt/scripts/run_glm52_document_disjoint_reference_confirmation.py \
  --model /model \
  --reference-directory /reference \
  --reference-plan /reference-plan.json \
  --reference-link /control/current-reference.safetensors \
  --intervention-artifact /artifact \
  --confirmation-registration /confirmation-registration.json \
  --control /control/control.json \
  --dest "/results/${result_name}" \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.89 \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --load-format safetensors \
  --quantization exl3 \
  --attention-backend TRITON_MLA \
  --max-model-len 513 \
  --max-num-batched-tokens 512 \
  --kld-chunk-rows 4

docker inspect "${container_name}" > "${record_root}/container-created-inspect.json"
set +e
docker start -a "${container_name}" > "${record_root}/run.log" 2>&1
status=$?
set -e
docker inspect "${container_name}" > "${record_root}/container-completed-inspect.json"
docker rm "${container_name}" > "${record_root}/removed-container-id.txt"
if test "${status}" -ne 0; then
  exit "${status}"
fi

test -f "${result_root}/report.json"
python3 - "${result_root}/report.json" "${record_root}/completion.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
report = json.loads(report_path.read_text())
assert report["status"] == "complete"
assert report["measurement_controls"]["passed"] is True
assert report["model_downloads_performed"] is False
completion = {
    "schema": "qsrt_glm52_document_disjoint_public_reference_launch",
    "schema_version": 1,
    "status": "complete",
    "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    "document_count": report["summary"]["document_count"],
    "candidate_mean_forward_kld": report["summary"]["pooled_position_weight"]["candidate_mean_forward_kld"],
    "candidate_below_0_059": report["numerical_target"]["pooled_candidate_below_target"],
}
temporary = Path(sys.argv[2] + ".partial")
temporary.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
os.replace(temporary, sys.argv[2])
print(json.dumps(completion, indent=2, sort_keys=True))
PY

#!/usr/bin/env bash
set -euo pipefail

# Measure one frozen intervention against the unchanged resident EXL3 model on
# sixteen public, document-disjoint BF16 reference chunks. With no plan file,
# all experts in the artifact are active. A low-rank confirmation registration
# selects one frozen expert. A candidate-subset selection plan measures several
# frozen expert subsets in one model load. Resident controls bracket every run.

if test "$#" -lt 2 || test "$#" -gt 3; then
  echo "usage: $0 <artifact-directory-name> <result-directory-name> [registration-or-selection-plan-file-name]" >&2
  exit 2
fi

artifact_name="$1"
result_name="$2"
candidate_input_name="${3:-}"
for value in "${artifact_name}" "${result_name}"; do
  if [[ ! "${value}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "artifact, result, and registration names must be path-safe basenames" >&2
    exit 2
  fi
done
if test -n "${candidate_input_name}" && [[ ! "${candidate_input_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "artifact, result, and registration names must be path-safe basenames" >&2
  exit 2
fi

experiment_root="/home/sunil/qsrt-glm52-experiments"
results_root="${experiment_root}/results"
artifact_root="${results_root}/${artifact_name}"
result_root="${results_root}/${result_name}"
reference_root="${experiment_root}/reference/glm52-unsloth-document-disjoint-auxiliary-v1"
reference_directory="${reference_root}/reference-logprobs"
reference_plan="${reference_root}/selection-plan.json"
reference_receipt="${reference_root}/receipt.json"
control_root="${experiment_root}/runtime-control/${result_name}"
record_root="${experiment_root}/launch-records/${result_name}"
registration_root="${experiment_root}/registrations"
model="${experiment_root}/model-cache/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78-nvme"
runtime_cache="${experiment_root}/runtime-cache/glm52-per-expert-exl3-without-fused-staging-dense-triton-bf16-reference-kld"
container_name="qsrt-${result_name}"
image="verdictai/glm52-exl3-sparkinfer:v39-r28-r7fused-broadcast-cu132-sm120a"

test -f "${artifact_root}/manifest.json"
test -f "${artifact_root}/report.json"
test -f "${reference_plan}"
test -f "${reference_receipt}"
test -d "${model}"
test ! -e "${result_root}"
test ! -e "${control_root}"
test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"
docker image inspect "${image}" >/dev/null

candidate_input_mount=()
candidate_input_arguments=()
selected_expert_policy="complete-artifact"
runner_relative="scripts/run_glm52_document_disjoint_reference_confirmation.py"
source_snapshot="${experiment_root}/source/qsrt-multi-layer-intervention-runtime-20260819"
candidate_input_path=""
candidate_input_schema=""
if test -n "${candidate_input_name}"; then
  candidate_input_path="${registration_root}/${candidate_input_name}"
  test -f "${candidate_input_path}"
  candidate_input_schema="$(jq -r .schema "${candidate_input_path}")"
  if test "${candidate_input_schema}" = "qsrt_glm52_low_rank_down_confirmation_registration"; then
    candidate_input_mount=(-v "${candidate_input_path}:/confirmation-registration.json:ro")
    candidate_input_arguments=(--confirmation-registration /confirmation-registration.json)
    selected_expert_policy="registered-singleton"
  elif test "${candidate_input_schema}" = "qsrt_glm52_model_kld_candidate_subset_selection"; then
    candidate_input_mount=(-v "${candidate_input_path}:/candidate-selection-plan.json:ro")
    candidate_input_arguments=(--candidate-selection-plan /candidate-selection-plan.json)
    selected_expert_policy="frozen-candidate-subsets"
    runner_relative="scripts/run_glm52_document_disjoint_candidate_selection.py"
  else
    echo "candidate input has an unsupported schema: ${candidate_input_schema}" >&2
    exit 2
  fi
fi
test -f "${source_snapshot}/${runner_relative}"

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

read -r artifact_manifest_sha256 model_layers expert_count < <(
  python3 - "${artifact_root}/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["status"] == "complete"
assert 1 <= report["expert_count"] <= 8
if "layer" in report:
    layers = [report["layer"]]
else:
    layers = report["model_layers"]
assert layers and all(3 <= layer <= 77 for layer in layers)
assert len(layers) == len(set(layers))
print(report["manifest_sha256"], ",".join(map(str, layers)), report["expert_count"])
PY
)
test "${#artifact_manifest_sha256}" -eq 64

first_reference="$(jq -r '.selected_chunks[0].reference_file' "${reference_plan}")"
test -f "${reference_directory}/${first_reference}"
mkdir -p "${control_root}" "${record_root}" "${runtime_cache}"
ln -s "/reference/${first_reference}" "${control_root}/current-reference.safetensors"
source_files=(
  "${source_snapshot}/qsrt/glm52_document_disjoint_confirmation.py"
  "${source_snapshot}/qsrt/glm52_engine_kld.py"
  "${source_snapshot}/qsrt/glm52_expert_intervention_runtime.py"
  "${source_snapshot}/qsrt/glm52_paired_kld.py"
  "${source_snapshot}/scripts/run_glm52_document_disjoint_reference_confirmation.py"
)
if test "${candidate_input_schema}" = "qsrt_glm52_model_kld_candidate_subset_selection"; then
  source_files+=(
    "${source_snapshot}/qsrt/glm52_model_kld_candidate_selection.py"
    "${source_snapshot}/scripts/run_glm52_document_disjoint_candidate_selection.py"
  )
fi
sha256sum "${source_files[@]}" > "${record_root}/executable-source.sha256"
cp "${reference_receipt}" "${record_root}/reference-download-receipt.json"
if test -n "${candidate_input_path}"; then
  sha256sum "${candidate_input_path}" > "${record_root}/candidate-input.sha256"
fi

docker create \
  --name "${container_name}" \
  --label qsrt.experiment="glm52-frozen-intervention-document-disjoint-public-reference-screen" \
  --label qsrt.model-downloads-performed=false \
  --label qsrt.reference-document-count=16 \
  --label qsrt.reference-context-tokens=512 \
  --label qsrt.model-layers="${model_layers}" \
  --label qsrt.artifact-expert-count="${expert_count}" \
  --label qsrt.selected-expert-policy="${selected_expert_policy}" \
  --label qsrt.exl3-moe-execution=three-gemm-per-expert-correctness \
  --label qsrt.candidate-runtime=stored-dense-endpoint \
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
  -v "${source_snapshot}:/workspace/qsrt:ro" \
  -v "${control_root}:/control:rw" \
  -v "${results_root}:/results:rw" \
  -v "${experiment_root}/corpus-cache/huggingface:/hf-corpus:rw" \
  -v "${experiment_root}/dependencies/kld-datasets-pydeps:/kld-pydeps:ro" \
  -v "${runtime_cache}:/cache:rw" \
  "${candidate_input_mount[@]}" \
  "${image}" \
  "/workspace/qsrt/${runner_relative}" \
  --model /model \
  --reference-directory /reference \
  --reference-plan /reference-plan.json \
  --reference-link /control/current-reference.safetensors \
  --intervention-artifact /artifact \
  "${candidate_input_arguments[@]}" \
  --candidate-runtime-mode candidate \
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
if report["schema"] == "qsrt_glm52_document_disjoint_model_kld_candidate_selection":
    completion = {
        "schema": "qsrt_glm52_model_kld_candidate_subset_selection_completion",
        "schema_version": 1,
        "status": "complete",
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "candidate_arms": {
            name: value["summary"] for name, value in report["candidate_arms"].items()
        },
    }
else:
    summary = report["summary"]
    completion = {
        "schema": "qsrt_glm52_complete_panel_public_reference_screen",
        "schema_version": 1,
        "status": "complete",
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "document_count": summary["document_count"],
        "equal_document_weight": summary["equal_document_weight"],
        "paired_document_bootstrap": summary["paired_document_bootstrap"],
        "tail_metrics": summary["tail_metrics"],
    }
temporary = Path(sys.argv[2] + ".partial")
temporary.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
os.replace(temporary, sys.argv[2])
print(json.dumps(completion, indent=2, sort_keys=True))
PY

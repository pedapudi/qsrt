#!/usr/bin/env bash
set -euo pipefail

# Screen one frozen expert subset on the independent published 2,048-token
# BF16 reference. The script writes and hashes a complete measurement
# registration before the model process can open the reference.

if test "$#" -lt 2 || test "$#" -gt 3; then
  echo "usage: $0 <artifact-directory-name> <result-directory-name> [comma-separated-expert-ids]" >&2
  exit 2
fi
artifact_name="$1"
result_name="$2"
selected_expert_text="${3:-}"
for value in "${artifact_name}" "${result_name}"; do
  if [[ ! "${value}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "artifact and result names must be path-safe basenames" >&2
    exit 2
  fi
done
if test -n "${selected_expert_text}" && [[ ! "${selected_expert_text}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "selected experts must be a comma-separated list of integers" >&2
  exit 2
fi

experiment_root="/home/sunil/qsrt-glm52-experiments"
source_root="${experiment_root}/source/qsrt-multi-layer-intervention-runtime-20260819"
artifact_root="${experiment_root}/results/${artifact_name}"
reference_root="${experiment_root}/reference/glm52-bf16-kld-20260708/reference-logits"
model_root="${experiment_root}/model-cache/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78-nvme"
result_root="${experiment_root}/results/${result_name}"
record_root="${experiment_root}/launch-records/${result_name}"
control_root="${experiment_root}/runtime-control/${result_name}"
# Reuse the cache that defines the established per-expert EXL3 correctness
# runtime. Fresh autotuning changed resident routes and KLD even though the
# model, reference, image, and explicit runtime options were unchanged.
runtime_cache="${experiment_root}/runtime-cache/glm52-per-expert-exl3-without-fused-staging-dense-triton-bf16-reference-kld"
container_name="qsrt-${result_name}"
image="verdictai/glm52-exl3-sparkinfer:v39-r28-r7fused-broadcast-cu132-sm120a"

for required in \
  "${artifact_root}/manifest.json" \
  "${artifact_root}/report.json" \
  "${reference_root}/manifest.json" \
  "${reference_root}/logits_0.safetensors" \
  "${source_root}/qsrt/glm52_expert_intervention_runtime.py" \
  "${source_root}/qsrt/glm52_engine_kld.py" \
  "${source_root}/scripts/run_glm52_paired_expert_intervention_kld.py"; do
  test -f "${required}"
done
test -d "${model_root}"
test ! -e "${result_root}"
test ! -e "${record_root}"
test ! -e "${control_root}"
test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"
mkdir -p "${record_root}" "${control_root}" "${runtime_cache}"

read -r artifact_manifest_sha256 candidate_subset_json < <(
  python3 - \
    "${artifact_root}/manifest.json" \
    "${artifact_root}/report.json" \
    "${reference_root}/manifest.json" \
    "${reference_root}/logits_0.safetensors" \
    "${record_root}/measurement-registration.json" \
    "${artifact_name}" \
    "${result_name}" \
    "${selected_expert_text}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

manifest_path, report_path, reference_manifest_path, logits_path, output_path = map(Path, sys.argv[1:6])
artifact_name, result_name, selected_expert_text = sys.argv[6:9]

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

manifest = json.loads(manifest_path.read_text())
report = json.loads(report_path.read_text())
assert report["status"] == "complete"
is_multi_layer = "model_layers" in report
if is_multi_layer:
    assert not selected_expert_text
    experts_by_layer = report["expert_ids_by_layer"]
    available_experts = [
        expert for experts in experts_by_layer.values() for expert in experts
    ]
    experts = available_experts
    model_layers = report["model_layers"]
else:
    available_experts = [record["expert"] for record in report["experts"]]
    if selected_expert_text:
        experts = [int(value) for value in selected_expert_text.split(",")]
    else:
        experts = available_experts
    model_layers = [report["layer"]]
    experts_by_layer = {str(report["layer"]): experts}
if is_multi_layer:
    assert 2 <= len(model_layers) <= 75
    assert 1 <= len(available_experts) <= 8 * len(model_layers)
else:
    assert 1 <= len(available_experts) <= 8
assert experts
if not is_multi_layer:
    assert len(experts) == len(set(experts))
    assert set(experts) <= set(available_experts)
registration = {
    "schema": "qsrt_glm52_registered_partial_single_reference_measurement",
    "schema_version": 1,
    "status": "frozen_before_reference_open",
    "artifact": {
        "name": artifact_name,
        "manifest_identity": report["manifest_sha256"],
        "manifest_file_sha256": digest(manifest_path),
        "report_file_sha256": digest(report_path),
        "model_layers": model_layers,
        "experts_by_layer": experts_by_layer,
    },
    "reference": {
        "manifest_sha256": digest(reference_manifest_path),
        "logits_sha256": digest(logits_path),
        "logits_bytes": logits_path.stat().st_size,
        "context_count": 1,
        "context_length": 2048,
    },
    "result_name": result_name,
    "selection_provenance": {
        "available_artifact_experts": available_experts,
        "selected_experts": experts,
        "selection_was_explicit": bool(selected_expert_text),
    },
    "acceptance": {
        "absolute_target_mean_kld": 0.059,
        "evidence_boundary": (
            "One independent context supplies an absolute-target screen. "
            "Document-replicated evidence is required for qualification."
        ),
    },
}
temporary = output_path.with_suffix(".partial")
temporary.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output_path)
subset = {} if is_multi_layer else {"frozen_expert_subset": experts}
print(report["manifest_sha256"], json.dumps(subset, separators=(",", ":"), sort_keys=True))
PY
)
test "${#artifact_manifest_sha256}" = 64
sha256sum "${record_root}/measurement-registration.json" > "${record_root}/measurement-registration.sha256"

docker create \
  --name "${container_name}" \
  --label qsrt.experiment=glm52-registered-partial-single-reference-screen \
  --label qsrt.model-downloads-performed=false \
  --label qsrt.evidence-boundary=one-independent-context \
  --label qsrt.prompt-logprob-chunk-tokens=2048 \
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
  -e QSRT_GLM52_ENGINE_KLD_REFERENCE_PATH=/reference/logits_0.safetensors \
  -e QSRT_GLM52_ENGINE_KLD_REFERENCE_KEY=logits \
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
  -v "${model_root}:/model:ro" \
  -v "${reference_root}:/reference:ro" \
  -v "${artifact_root}:/artifact:ro" \
  -v "${source_root}:/workspace/qsrt:ro" \
  -v "${control_root}:/control:rw" \
  -v "${experiment_root}/results:/results:rw" \
  -v "${experiment_root}/corpus-cache/huggingface:/hf-corpus:rw" \
  -v "${experiment_root}/dependencies/kld-datasets-pydeps:/kld-pydeps:ro" \
  -v "${runtime_cache}:/cache:rw" \
  "${image}" \
  /workspace/qsrt/scripts/run_glm52_paired_expert_intervention_kld.py \
  --model /model \
  --reference-logits /reference \
  --intervention-artifact /artifact \
  --control /control/control.json \
  --dest "/results/${result_name}" \
  --context-length 2048 \
  --source-sparse-index-topk 2048 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.89 \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --load-format safetensors \
  --quantization exl3 \
  --attention-backend TRITON_MLA \
  --max-model-len 2049 \
  --max-num-batched-tokens 2048 \
  --kld-chunk-rows 4 \
  --kld-device cuda:0 \
  --candidate-expert-subsets-json "${candidate_subset_json}" \
  --omit-individual-expert-arms \
  --hf-overrides '{"index_topk":0,"use_index_cache":false}' \
  --llm-extra-json '{"decode_context_parallel_size":1,"moe_backend":"b12x","enforce_eager":true,"disable_custom_all_reduce":true,"async_scheduling":false}'

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
sha256sum "${result_root}/report.json" > "${record_root}/result.sha256"

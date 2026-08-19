#!/usr/bin/env bash
set -euo pipefail

# Measure one already-frozen layer-63 composition on the independent published
# 2,048-token BF16 reference. The registration fixes the artifact, expert set,
# reference, and result name before the candidate process opens the reference.
# One context supplies a directional screen, not checkpoint evidence.

experiment_root="/home/sunil/qsrt-glm52-experiments"
source_root="${experiment_root}/source/qsrt-model-kld-candidate-selection-complete-slice-ancestry-20260818"
artifact_name="glm52-layer63-frozen8-reconstructed-activation-down-refit-merged"
registration_name="glm52_layer63_down_refit_model_kld_retained_composition_single_reference_registration.json"
registration_path="${experiment_root}/registrations/${registration_name}"
subset_plan_path="${experiment_root}/registrations/glm52_layer63_down_refit_model_kld_retained_composition_plan.json"
reference_root="${experiment_root}/reference/glm52-bf16-kld-20260708/reference-logits"
model_root="${experiment_root}/model-cache/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78-nvme"
artifact_root="${experiment_root}/results/${artifact_name}"
result_name="glm52-layer63-down-refit-experts149-164-single-reference-absolute-target-screen"
result_root="${experiment_root}/results/${result_name}"
record_root="${experiment_root}/launch-records/${result_name}"
control_root="${experiment_root}/runtime-control/${result_name}"
# Reuse the cache that defines the established per-expert EXL3 correctness
# runtime. A fresh autotuning cache is a different numerical runtime.
runtime_cache="${experiment_root}/runtime-cache/glm52-per-expert-exl3-without-fused-staging-dense-triton-bf16-reference-kld"
container_name="qsrt-${result_name}"
image="verdictai/glm52-exl3-sparkinfer:v39-r28-r7fused-broadcast-cu132-sm120a"

test -f "${registration_path}"
test -f "${subset_plan_path}"
test -f "${artifact_root}/manifest.json"
test -f "${artifact_root}/report.json"
test -f "${reference_root}/manifest.json"
test -f "${reference_root}/logits_0.safetensors"
test -f "${source_root}/qsrt/glm52_expert_intervention_runtime.py"
test -f "${source_root}/qsrt/glm52_engine_kld.py"
test -f "${source_root}/scripts/run_glm52_paired_expert_intervention_kld.py"
test -f "${source_root}/experiments/glm52_wikitext_document_disjoint_corpus_plan.json"
test -d "${model_root}"
test ! -e "${result_root}"
test ! -e "${record_root}"
test ! -e "${control_root}"
test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"

candidate_subset_json="$(python3 -c '
import hashlib
import json
import sys
from pathlib import Path

registration_path, subset_plan_path, artifact_manifest, artifact_report, reference_manifest, reference_logits, separation_plan_path = map(Path, sys.argv[1:])
registration = json.loads(registration_path.read_text())
plan = json.loads(subset_plan_path.read_text())
report = json.loads(artifact_report.read_text())

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

assert registration["schema"] == "qsrt_glm52_single_reference_candidate_registration"
assert registration["status"] == "frozen_before_single_reference_measurement"
candidate = registration["candidate"]
reference = registration["reference"]
assert registration["result_name"] == "glm52-layer63-down-refit-experts149-164-single-reference-absolute-target-screen"
assert digest(artifact_manifest) == candidate["artifact_manifest_file_sha256"]
assert digest(artifact_report) == candidate["artifact_report_file_sha256"]
assert digest(subset_plan_path) == candidate["candidate_subset_plan_sha256"]
assert digest(reference_manifest) == reference["manifest_sha256"]
assert digest(reference_logits) == reference["logits_sha256"]
assert digest(separation_plan_path) == reference["corpus_separation_plan_sha256"]
assert reference_logits.stat().st_size == reference["logits_bytes"]
separation_plan = json.loads(separation_plan_path.read_text())
assert separation_plan["published_bf16_reference"]["manifest_sha256"] == reference["manifest_sha256"]
assert separation_plan["separation"]["reference_fit_row_overlap"] == reference["reference_fit_document_overlap"] == 0
assert separation_plan["separation"]["reference_selection_row_overlap"] == reference["reference_selection_document_overlap"] == 0
assert report["status"] == "complete"
assert report["layer"] == candidate["model_layer"]
assert report["manifest_sha256"] == candidate["artifact_manifest_identity"]
assert plan["status"] == "frozen_before_candidate_subset_kld_measurement"
assert plan["artifact_manifest_sha256"] == candidate["artifact_manifest_identity"]
arms = {arm["name"]: arm["selected_experts"] for arm in plan["candidate_arms"]}
assert arms == {candidate["arm_name"]: candidate["selected_experts"]}
assert set(candidate["selected_experts"]) <= set(report["panel"][str(candidate["model_layer"])])
print(json.dumps(arms, separators=(",", ":"), sort_keys=True))
' "${registration_path}" "${subset_plan_path}" "${artifact_root}/manifest.json" "${artifact_root}/report.json" "${reference_root}/manifest.json" "${reference_root}/logits_0.safetensors" "${source_root}/experiments/glm52_wikitext_document_disjoint_corpus_plan.json")"

mkdir -p "${record_root}" "${control_root}" "${runtime_cache}"
cp "${registration_path}" "${record_root}/"
cp "${subset_plan_path}" "${record_root}/"

docker create \
  --name "${container_name}" \
  --label qsrt.experiment="glm52-layer63-down-refit-retained-composition-single-reference" \
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
  -e QSRT_GLM52_INTERVENTION_MANIFEST_SHA256=0ced0fdc2898e5091ce5afe0c3c744dbea1b57240d76169a283293a4c18b6d2e \
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
sha256sum \
  "${registration_path}" \
  "${subset_plan_path}" \
  "${artifact_root}/manifest.json" \
  "${artifact_root}/report.json" \
  "${reference_root}/manifest.json" \
  "${reference_root}/logits_0.safetensors" \
  "${source_root}/qsrt/glm52_expert_intervention_runtime.py" \
  "${source_root}/qsrt/glm52_engine_kld.py" \
  "${source_root}/scripts/run_glm52_paired_expert_intervention_kld.py" \
  "${source_root}/experiments/glm52_wikitext_document_disjoint_corpus_plan.json" \
  "${record_root}/container-created-inspect.json" \
  > "${record_root}/inputs.sha256"

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

#!/usr/bin/env bash
set -euo pipefail

# Capture the same resident-EXL3 prompt rows at four frozen GLM-5.2 layers in
# one network-isolated model load. The layer panels were selected from EXL3
# rate metadata before any QSRT candidate error or KLD was measured.

experiment_root="/home/sunil/qsrt-glm52-experiments"
source_snapshot="${experiment_root}/source/qsrt-layers52-60-63-64-capture-artifact-layer-compatible-20260818"
model_root="${experiment_root}/model-cache/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78-nvme"
artifact_root="${experiment_root}/results/glm52-layer3-frozen8-dense-endpoints-r7-closure-merged-v2"
reference_root="${experiment_root}/reference/glm52-bf16-kld-20260708/reference-logits"
capture_name="glm52-layers52-60-63-64-wikitext-document-disjoint-routed-inputs-artifact-layer-compatible"
capture_root="${experiment_root}/captures/${capture_name}"
result_name="glm52-layers52-60-63-64-input-capture-artifact-layer-compatible-measurement-controls"
result_root="${experiment_root}/results/${result_name}"
record_root="${experiment_root}/launch-records/${result_name}"
control_root="${experiment_root}/runtime-control/${result_name}"
runtime_cache="${experiment_root}/runtime-cache/glm52-layers52-60-63-64-input-capture"
validation_report="${experiment_root}/preflight/glm52-exl3-host-local-copy/validation.json"
container_name="qsrt-glm52-layers52-60-63-64-input-capture-artifact-layer-compatible"
artifact_manifest_sha256="11e26125921be272992ef07c7430e234309e4b2f6b20146a224598a59c7a7af9"
plan_sha256="b694ac0a1aeb09f7c61a20b5f72289f3e791d616ff7471b5894d857f8c363b55"

test -f "${source_snapshot}/scripts/run_glm52_paired_expert_intervention_kld.py"
test -f "${validation_report}"
test -f "${artifact_root}/report.json"
test -f "${reference_root}/logits_0.safetensors"
test ! -e "${capture_root}"
test ! -e "${result_root}"
test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"
mkdir -p "${record_root}" "${control_root}" "${runtime_cache}"

df -B1 "${experiment_root}" > "${record_root}/internal-nvme-space-before.txt"
nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader > "${record_root}/gpu-state-before.csv"
sha256sum \
  "${source_snapshot}/experiments/glm52_layer52_rate_pattern_panel.json" \
  "${source_snapshot}/experiments/glm52_layer60_rate_pattern_panel.json" \
  "${source_snapshot}/experiments/glm52_layer63_rate_pattern_panel.json" \
  "${source_snapshot}/experiments/glm52_layer64_rate_pattern_panel.json" \
  "${source_snapshot}/experiments/glm52_wikitext_document_disjoint_corpus_plan.json" \
  "${artifact_root}/manifest.json" \
  "${artifact_root}/report.json" \
  "${reference_root}/manifest.json" \
  "${reference_root}/logits_0.safetensors" \
  > "${record_root}/bound-inputs.sha256"

docker create \
  --name "${container_name}" \
  --label qsrt.experiment=glm52-layers52-60-63-64-input-capture \
  --label qsrt.model-downloads-performed=false \
  --label qsrt.capture-output-device=internal-nvme \
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
  -e QSRT_GLM52_ACTIVATION_CAPTURE_DIR="/captures/${capture_name}" \
  -e QSRT_GLM52_ACTIVATION_CAPTURE_PLAN_SHA256="${plan_sha256}" \
  -e QSRT_GLM52_ACTIVATION_CAPTURE_LAYERS=52,60,63,64 \
  -e HF_HOME=/hf-corpus \
  -e HF_HUB_CACHE=/hf-corpus/hub \
  -e HF_DATASETS_CACHE=/hf-corpus/datasets \
  -e KLD_PYDEPS=/kld-pydeps \
  -e HF_DATASETS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e B12X_MOE_FORCE_A16=1 \
  -e B12X_W4A16_TC_DECODE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_EXL3_R7_FUSED=1 \
  -e VLLM_EXL3_R7_FUSED_LAYERS=48 \
  -e VLLM_EXL3_R7_A1_MIN_ROWS=0 \
  -e VLLM_EXL3_PREFILL_CAPACITY=1024 \
  -e VLLM_EXL3_PREFILL_BLOCK_M=64 \
  -e KV_FP8_ROPE=0 \
  -e VLLM_NVFP4_MLA_SCALES_FILE=/model/nvfp4_mla_outer_scales.json \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -v "${model_root}:/model:ro" \
  -v "${reference_root}:/reference:ro" \
  -v "${artifact_root}:/artifact:ro" \
  -v "${source_snapshot}:/workspace/qsrt:ro" \
  -v "${control_root}:/control:rw" \
  -v "${experiment_root}/captures:/captures:rw" \
  -v "${experiment_root}/results:/results:rw" \
  -v "${experiment_root}/corpus-cache/huggingface:/hf-corpus:rw" \
  -v "${experiment_root}/dependencies/kld-datasets-pydeps:/kld-pydeps:ro" \
  -v "${runtime_cache}:/cache:rw" \
  verdictai/glm52-exl3-sparkinfer:v39-r28-r7fused-broadcast-cu132-sm120a \
  /workspace/qsrt/scripts/run_glm52_paired_expert_intervention_kld.py \
  --model /model \
  --reference-logits /reference \
  --intervention-artifact /artifact \
  --control /control/control.json \
  --dest "/results/${result_name}" \
  --corpus-plan /workspace/qsrt/experiments/glm52_wikitext_document_disjoint_corpus_plan.json \
  --activation-capture-dir "/captures/${capture_name}" \
  --activation-capture-panel-manifest /workspace/qsrt/experiments/glm52_layer52_rate_pattern_panel.json \
  --activation-capture-panel-manifest /workspace/qsrt/experiments/glm52_layer60_rate_pattern_panel.json \
  --activation-capture-panel-manifest /workspace/qsrt/experiments/glm52_layer63_rate_pattern_panel.json \
  --activation-capture-panel-manifest /workspace/qsrt/experiments/glm52_layer64_rate_pattern_panel.json \
  --context-length 2048 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --dtype bfloat16 \
  --kv-cache-dtype nvfp4_ds_mla \
  --load-format safetensors \
  --quantization exl3 \
  --attention-backend B12X_MLA_SPARSE \
  --max-model-len 2049 \
  --max-num-batched-tokens 2048 \
  --kld-chunk-rows 16 \
  --measurement-controls-only \
  --hf-overrides '{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}' \
  --llm-extra-json '{"decode_context_parallel_size":1,"moe_backend":"b12x","enforce_eager":true,"disable_custom_all_reduce":true,"async_scheduling":false}'

docker inspect "${container_name}" > "${record_root}/container-created-inspect.json"
python3 - "${record_root}/container-created-inspect.json" <<'PY'
import json
import sys
from pathlib import Path

container = json.loads(Path(sys.argv[1]).read_text())[0]
assert container["State"]["Status"] == "created"
assert container["HostConfig"]["NetworkMode"] == "none"
assert container["Image"] == "sha256:12f86065d7fe64d30dad678585e68c91f47f1f2a32bed45ccaf108382f3928ac"
environment = set(container["Config"]["Env"])
assert "QSRT_GLM52_ACTIVATION_CAPTURE_LAYERS=52,60,63,64" in environment
mount_modes = {item["Destination"]: item["Mode"] for item in container["Mounts"]}
for destination in ("/model", "/reference", "/artifact", "/workspace/qsrt", "/kld-pydeps"):
    assert mount_modes[destination] == "ro"
assert mount_modes["/captures"] == "rw"
assert container["Config"]["Labels"]["qsrt.model-downloads-performed"] == "false"
PY

docker start "${container_name}"
docker inspect "${container_name}" > "${record_root}/container-started-inspect.json"

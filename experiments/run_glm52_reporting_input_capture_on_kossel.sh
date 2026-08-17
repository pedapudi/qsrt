#!/usr/bin/env bash
set -euo pipefail

# Capture the resident layer-3 input, routes, and applied route weights for the
# already-published BF16-reference context. The run executes only repeatability
# and direct-return identity controls after capture. The reporting context is
# prohibited from selecting any later quantization candidate.

experiment_root="/home/sunil/qsrt-glm52-experiments"
source_root="${experiment_root}/source/qsrt-working-tree"
artifact_root="${experiment_root}/results/glm52-layer3-frozen8-dense-endpoints-r7-closure-merged-v2"
capture_name="glm52-layer3-published-reference-reporting-inputs"
capture_root="${experiment_root}/captures/${capture_name}"
result_name="glm52-layer3-reporting-input-capture-measurement-controls"
result_root="${experiment_root}/results/${result_name}"
record_root="${experiment_root}/launch-records/${result_name}"
snapshot_root="${record_root}/source-snapshot"
control_root="${experiment_root}/runtime-control/${result_name}"
runtime_cache="${experiment_root}/runtime-cache/glm52-per-expert-exl3-without-fused-staging-dense-triton-bf16-reference-kld"
validation_report="${experiment_root}/preflight/glm52-exl3-host-local-copy/validation.json"
reference_root="${experiment_root}/reference/glm52-bf16-kld-20260708/reference-logits"
measurement_control_report="${experiment_root}/results/glm52-layer3-per-expert-exl3-engine-kld-paired-bf16-reference-kld-repeatability-control/report.json"
container_name="qsrt-${result_name}"
plan_relative="experiments/glm52_wikitext_document_disjoint_corpus_plan.json"
expected_plan_sha256="b694ac0a1aeb09f7c61a20b5f72289f3e791d616ff7471b5894d857f8c363b55"

test -f "${source_root}/${plan_relative}"
test -f "${artifact_root}/manifest.json"
test -f "${artifact_root}/report.json"
test -f "${validation_report}"
test -f "${reference_root}/manifest.json"
test -f "${reference_root}/logits_0.safetensors"
test -f "${measurement_control_report}"
test ! -e "${capture_root}"
test ! -e "${result_root}"
test ! -e "${record_root}"
test ! -e "${control_root}"
test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"
test "$(sha256sum "${source_root}/${plan_relative}" | cut -d ' ' -f 1)" = "${expected_plan_sha256}"

python3 -c '
import json
import sys
from pathlib import Path

validation = json.loads(Path(sys.argv[1]).read_text())
reference = json.loads(Path(sys.argv[2]).read_text())
controls = json.loads(Path(sys.argv[3]).read_text())
assert validation["status"] == "passed"
assert validation["network_transfer"] is False
assert reference["context_length"] == 2048
assert len(reference["windows"]) == 1
assert controls["status"] == "complete"
assert controls["measurement_controls_passed"] is True
for name in ("resident_repeatability_control", "dense_resident_identity_control"):
    assert controls[name]["forward_kld_bitwise_equal"] is True
    assert controls[name]["all_layer_route_array_equal"] is True
' "${validation_report}" "${reference_root}/manifest.json" "${measurement_control_report}"

mkdir -p "${snapshot_root}" "${control_root}" "${runtime_cache}"
rsync -a \
  --exclude .git \
  --exclude .venv \
  --exclude out \
  --exclude __pycache__ \
  --exclude '*.pyc' \
  --exclude .pytest_cache \
  "${source_root}/" "${snapshot_root}/"
(
  cd "${snapshot_root}"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > "${record_root}/source-files.sha256"

artifact_manifest_sha256="$(
  python3 -c '
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
assert report["status"] == "complete"
assert report["expert_count"] == 8
assert report["panel"] == {"3": [64, 208, 106, 204, 89, 212, 96, 103]}
assert report["manifest_sha256"] == "11e26125921be272992ef07c7430e234309e4b2f6b20146a224598a59c7a7af9"
print(report["manifest_sha256"])
' "${artifact_root}/report.json"
)"

docker create \
  --name "${container_name}" \
  --label qsrt.experiment=glm52-layer3-reporting-input-capture-measurement-controls \
  --label qsrt.reporting-context-use=diagnosis-only-candidate-selection-prohibited \
  --label qsrt.model-downloads-performed=false \
  --label qsrt.model-runner=v1 \
  --label qsrt.attention-contract=dense-triton-mla \
  --label qsrt.exl3-moe-execution=three-gemm-per-expert-correctness \
  --label qsrt.exl3-weight-preparation=raw-per-expert-without-fused-staging \
  --label qsrt.kld-result-transport=prompt-logprob-scalar-channel \
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
  -e QSRT_GLM52_ACTIVATION_CAPTURE_PLAN_SHA256="${expected_plan_sha256}" \
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
  -v "${experiment_root}/model-cache/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78-nvme:/model:ro" \
  -v "${reference_root}:/reference:ro" \
  -v "${artifact_root}:/artifact:ro" \
  -v "${snapshot_root}:/workspace/qsrt:ro" \
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
  --corpus-plan "/workspace/qsrt/${plan_relative}" \
  --reporting-activation-capture-dir "/captures/${capture_name}" \
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
  --measurement-controls-only \
  --hf-overrides '{"index_topk":0,"use_index_cache":false}' \
  --llm-extra-json '{"decode_context_parallel_size":1,"moe_backend":"b12x","enforce_eager":true,"disable_custom_all_reduce":true,"async_scheduling":false}'

docker inspect "${container_name}" > "${record_root}/container-created-inspect.json"
python3 -c '
import json
import sys
from pathlib import Path

container = json.loads(Path(sys.argv[1]).read_text())[0]
assert container["State"]["Status"] == "created"
assert container["HostConfig"]["NetworkMode"] == "none"
assert container["Image"] == "sha256:12f86065d7fe64d30dad678585e68c91f47f1f2a32bed45ccaf108382f3928ac"
environment = set(container["Config"]["Env"])
assert "QSRT_GLM52_ACTIVATION_CAPTURE_DIR=/captures/glm52-layer3-published-reference-reporting-inputs" in environment
assert "QSRT_GLM52_ACTIVATION_CAPTURE_PLAN_SHA256=b694ac0a1aeb09f7c61a20b5f72289f3e791d616ff7471b5894d857f8c363b55" in environment
mount_modes = {item["Destination"]: item["Mode"] for item in container["Mounts"]}
for destination in ("/model", "/reference", "/artifact", "/workspace/qsrt", "/kld-pydeps"):
    assert mount_modes[destination] == "ro"
assert mount_modes["/captures"] == "rw"
assert container["Config"]["Labels"]["qsrt.reporting-context-use"] == "diagnosis-only-candidate-selection-prohibited"
' "${record_root}/container-created-inspect.json"

docker start "${container_name}"
docker inspect "${container_name}" > "${record_root}/container-started-inspect.json"
sha256sum \
  "${validation_report}" \
  "${reference_root}/manifest.json" \
  "${reference_root}/logits_0.safetensors" \
  "${measurement_control_report}" \
  "${artifact_root}/manifest.json" \
  "${artifact_root}/report.json" \
  "${record_root}/source-files.sha256" \
  "${record_root}/container-created-inspect.json" \
  "${record_root}/container-started-inspect.json"

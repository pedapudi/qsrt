#!/usr/bin/env bash
set -euo pipefail

# Measure one complete candidate panel against the resident EXL3 checkpoint and
# published BF16 reference logits. Dense attention admits the same complete
# causal key set as GLM-5.2's 2,048-entry sparse index at this 2,048-token bound.
# Four-row KLD chunks run on GPU 0 and return only per-position values to CPU.
# v39's deliberately slow three-GEMM-per-expert path avoids both fused EXL3
# MoE implementations. The measurement controls must prove that this path is
# bitwise repeatable before the complete selected panel is evaluated.

if test "$#" -ne 1; then
  echo "usage: $0 {uniform-k3|routed-input-curvature|reconstructed-activation-down-refit|down-covariance-source-target|down-identity-refit-target|down-covariance-refit-target|fixed-mixed-k3-k4-down-refit|fixed-rate-preserving-down-refit-k3-k4|selection-data-rate-preserving-down-refit-k3-k4}" >&2
  exit 2
fi

method="$1"
experiment_root="/home/sunil/qsrt-glm52-experiments"
results_root="${experiment_root}/results"

case "${method}" in
  uniform-k3)
    artifact_name="glm52-layer3-frozen8-dense-endpoints-r7-closure-merged-v2"
    expected_experiment="uniform-k3-base-artifact"
    ;;
  routed-input-curvature)
    artifact_name="glm52-layer3-frozen8-routed-input-curvature-merged"
    expected_experiment="qsrt_glm52_routed_input_curvature_control_v1"
    ;;
  reconstructed-activation-down-refit)
    artifact_name="glm52-layer3-frozen8-reconstructed-activation-down-refit-merged"
    expected_experiment="qsrt_glm52_reconstructed_activation_down_refit_v1"
    ;;
  down-covariance-source-target)
    artifact_name="glm52-layer3-frozen8-down-construction-reconstructed_input_covariance__source_weights-merged"
    expected_experiment="qsrt_glm52_down_construction_comparison_v1"
    ;;
  down-identity-refit-target)
    artifact_name="glm52-layer3-frozen8-down-construction-identity__reconstructed_activation_refit-merged"
    expected_experiment="qsrt_glm52_down_construction_comparison_v1"
    ;;
  down-covariance-refit-target)
    artifact_name="glm52-layer3-frozen8-down-construction-reconstructed_input_covariance__reconstructed_activation_refit-merged"
    expected_experiment="qsrt_glm52_down_construction_comparison_v1"
    ;;
  fixed-mixed-k3-k4-down-refit)
    artifact_name="glm52-layer3-frozen8-fixed-mixed-k3-k4-down-refit"
    expected_experiment="qsrt_glm52_fixed_mixed_k3_k4_down_refit_v1"
    expected_allocation_kind=""
    ;;
  fixed-rate-preserving-down-refit-k3-k4)
    artifact_name="glm52-layer3-frozen8-fixed-rate-preserving-down-refit-k3-k4"
    expected_experiment="qsrt_glm52_mixed_k3_k4_rate_preserving_down_refit_v1"
    expected_allocation_kind="fixed_rate_stratified"
    ;;
  selection-data-rate-preserving-down-refit-k3-k4)
    artifact_name="glm52-layer3-frozen8-selection-data-rate-preserving-down-refit-k3-k4"
    expected_experiment="qsrt_glm52_mixed_k3_k4_rate_preserving_down_refit_v1"
    expected_allocation_kind="selection_data_complete_expert"
    ;;
  *)
    echo "unknown candidate method: ${method}" >&2
    exit 2
    ;;
esac

artifact_root="${results_root}/${artifact_name}"
result_name="${artifact_name}-paired-bf16-reference-kld-engine-per-expert-correctness"
result_path="${results_root}/${result_name}"
control_root="${experiment_root}/runtime-control/${artifact_name}-per-expert-correctness"
record_root="${experiment_root}/launch-records/${result_name}"
container_name="qsrt-${result_name}"
validation_report="${experiment_root}/preflight/glm52-exl3-host-local-copy/validation.json"
capture_manifest="${experiment_root}/captures/glm52-layer3-wikitext-document-disjoint-routed-inputs/manifest.json"
measurement_control_report="${results_root}/glm52-layer3-per-expert-exl3-engine-kld-paired-bf16-reference-kld-repeatability-control/report.json"
runtime_cache="${experiment_root}/runtime-cache/glm52-per-expert-exl3-without-fused-staging-dense-triton-bf16-reference-kld"

test -f "${artifact_root}/manifest.json"
test -f "${artifact_root}/report.json"
test -f "${validation_report}"
test -f "${capture_manifest}"
test -f "${measurement_control_report}"
test ! -e "${result_path}"
test ! -e "${control_root}"
test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"

python3 -c '
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
assert report["schema"] == "qsrt_glm52_paired_expert_intervention_kld"
assert report["schema_version"] == 2
assert report["status"] == "complete"
assert report["measurement_controls_passed"] is True
for name in ("resident_repeatability_control", "dense_resident_identity_control"):
    assert report[name]["forward_kld_bitwise_equal"] is True
    assert report[name]["all_layer_route_array_equal"] is True
' "${measurement_control_report}"

artifact_manifest_sha256="$(
  python3 -c '
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
assert report["status"] == "complete"
assert report["expert_count"] == 8
assert report["panel"] == {"3": [64, 208, 106, 204, 89, 212, 96, 103]}
if sys.argv[2] == "uniform-k3-base-artifact":
    assert report["manifest_sha256"] == "11e26125921be272992ef07c7430e234309e4b2f6b20146a224598a59c7a7af9"
else:
    assert report["experiment"] == sys.argv[2]
if sys.argv[3]:
    assert report["allocation_kind"] == sys.argv[3]
print(report["manifest_sha256"])
' "${artifact_root}/report.json" "${expected_experiment}" "${expected_allocation_kind:-}"
)"
test "${#artifact_manifest_sha256}" -eq 64

mkdir -p "${control_root}" "${record_root}" "${runtime_cache}"
docker create \
  --name "${container_name}" \
  --label qsrt.experiment="glm52-layer3-${method}-paired-bf16-reference-kld" \
  --label qsrt.model-downloads-performed=false \
  --label qsrt.model-runner=v1 \
  --label qsrt.gpu-memory-utilization=0.89 \
  --label qsrt.prompt-logprob-chunk-tokens=2048 \
  --label qsrt.exl3-prefill-capacity=256 \
  --label qsrt.attention-contract=dense-triton-mla \
  --label qsrt.exl3-moe-execution=three-gemm-per-expert-correctness \
  --label qsrt.exl3-weight-preparation=raw-per-expert-without-fused-staging \
  --label qsrt.kld-device=cuda-0-four-row-chunks \
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
  -v "${experiment_root}/reference/glm52-bf16-kld-20260708/reference-logits:/reference:ro" \
  -v "${artifact_root}:/artifact:ro" \
  -v "${experiment_root}/source/qsrt-working-tree:/workspace/qsrt:ro" \
  -v "${control_root}:/control:rw" \
  -v "${results_root}:/results:rw" \
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
  --omit-individual-expert-arms \
  --hf-overrides '{"index_topk":0,"use_index_cache":false}' \
  --llm-extra-json '{"decode_context_parallel_size":1,"moe_backend":"b12x","enforce_eager":true,"disable_custom_all_reduce":true,"async_scheduling":false}'

docker inspect "${container_name}" > "${record_root}/container-created-inspect.json"
mkdir -p \
  "${record_root}/source-snapshot/qsrt" \
  "${record_root}/source-snapshot/scripts" \
  "${record_root}/source-snapshot/experiments"
cp \
  "${experiment_root}/source/qsrt-working-tree/qsrt/glm52_engine_kld.py" \
  "${experiment_root}/source/qsrt-working-tree/qsrt/glm52_expert_intervention_runtime.py" \
  "${record_root}/source-snapshot/qsrt/"
cp \
  "${experiment_root}/source/qsrt-working-tree/scripts/run_glm52_paired_expert_intervention_kld.py" \
  "${record_root}/source-snapshot/scripts/"
cp \
  "${experiment_root}/source/qsrt-working-tree/experiments/run_glm52_candidate_kld_chunked_full_vocabulary_on_kossel.sh" \
  "${record_root}/source-snapshot/experiments/"
python3 -c '
import json
import sys
from pathlib import Path

validation = json.loads(Path(sys.argv[1]).read_text())
capture = json.loads(Path(sys.argv[2]).read_text())
container = json.loads(Path(sys.argv[3]).read_text())[0]
assert validation["status"] == "passed"
assert validation["network_transfer"] is False
assert capture["status"] == "complete"
assert capture["collections"] == {"activation_fit": 32, "candidate_selection": 8}
assert container["State"]["Status"] == "created"
assert container["HostConfig"]["NetworkMode"] == "none"
assert container["Image"] == "sha256:12f86065d7fe64d30dad678585e68c91f47f1f2a32bed45ccaf108382f3928ac"
environment = set(container["Config"]["Env"])
assert "VLLM_EXL3_PREFILL_CAPACITY=256" in environment
assert "VLLM_USE_B12X_SPARSE_INDEXER=0" in environment
assert "VLLM_EXL3_R7_FUSED=0" in environment
assert "QSRT_GLM52_FORCE_PER_EXPERT_EXL3_MOE=1" in environment
assert "QSRT_GLM52_ENGINE_KLD_REFERENCE_PATH=/reference/logits_0.safetensors" in environment
assert "QSRT_GLM52_ENGINE_KLD_REFERENCE_KEY=logits" in environment
assert "QSRT_GLM52_ENGINE_KLD_CHUNK_ROWS=4" in environment
mount_modes = {item["Destination"]: item["Mode"] for item in container["Mounts"]}
for destination in ("/model", "/reference", "/artifact", "/workspace/qsrt", "/kld-pydeps"):
    assert mount_modes[destination] == "ro"
assert container["Config"]["Labels"]["qsrt.attention-contract"] == "dense-triton-mla"
assert container["Config"]["Labels"]["qsrt.exl3-moe-execution"] == "three-gemm-per-expert-correctness"
assert container["Config"]["Labels"]["qsrt.exl3-weight-preparation"] == "raw-per-expert-without-fused-staging"
assert container["Config"]["Labels"]["qsrt.kld-device"] == "cuda-0-four-row-chunks"
assert container["Config"]["Labels"]["qsrt.kld-result-transport"] == "prompt-logprob-scalar-channel"
assert container["Config"]["Labels"]["qsrt.gpu-memory-utilization"] == "0.89"
assert container["Config"]["Labels"]["qsrt.prompt-logprob-chunk-tokens"] == "2048"
' "${validation_report}" "${capture_manifest}" "${record_root}/container-created-inspect.json"

docker start "${container_name}"
docker inspect "${container_name}" > "${record_root}/container-started-inspect.json"
sha256sum \
  "${validation_report}" \
  "${capture_manifest}" \
  "${measurement_control_report}" \
  "${artifact_root}/manifest.json" \
  "${artifact_root}/report.json" \
  "${record_root}/container-created-inspect.json" \
  "${record_root}/container-started-inspect.json" \
  "${record_root}/source-snapshot/qsrt/glm52_engine_kld.py" \
  "${record_root}/source-snapshot/qsrt/glm52_expert_intervention_runtime.py" \
  "${record_root}/source-snapshot/scripts/run_glm52_paired_expert_intervention_kld.py" \
  "${record_root}/source-snapshot/experiments/run_glm52_candidate_kld_chunked_full_vocabulary_on_kossel.sh"

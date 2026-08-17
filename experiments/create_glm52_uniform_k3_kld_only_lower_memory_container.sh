#!/usr/bin/env bash
set -euo pipefail

# Score the frozen uniform-K3 panel after routed capture has completed. The
# lower vLLM memory reservation leaves room for the 1.18 GiB full-vocabulary
# FP32 log-softmax buffer required by this v39 image.

container_name="qsrt-glm52-layer3-uniform-k3-kld-model-runner-v1-lower-memory"
experiment_root="/home/sunil/qsrt-glm52-experiments"
record_dir="${experiment_root}/launch-records/glm52-layer3-frozen8-uniform-k3-kld-lower-memory"
result_path="${experiment_root}/results/glm52-layer3-frozen8-uniform-k3-paired-bf16-reference-kld-lower-memory"

mkdir -p "${record_dir}" "${experiment_root}/results"
test ! -e "${result_path}"
test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"

container_id="$(
  docker create \
    --name "${container_name}" \
    --label qsrt.experiment=glm52-layer3-uniform-k3-paired-bf16-reference-kld \
    --label qsrt.model-downloads-performed=false \
    --label qsrt.model-runner=v1 \
    --label qsrt.gpu-memory-utilization=0.92 \
    --label qsrt.routed-capture-reused=true \
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
    -e QSRT_GLM52_INTERVENTION_MANIFEST_SHA256=11e26125921be272992ef07c7430e234309e4b2f6b20146a224598a59c7a7af9 \
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
    -v "${experiment_root}/model-cache/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78-nvme:/model:ro" \
    -v "${experiment_root}/reference/glm52-bf16-kld-20260708/reference-logits:/reference:ro" \
    -v "${experiment_root}/results/glm52-layer3-frozen8-dense-endpoints-r7-closure-merged-v2:/artifact:ro" \
    -v "${experiment_root}/source/qsrt-working-tree:/workspace/qsrt:ro" \
    -v "${experiment_root}/runtime-control/glm52-layer3-frozen8-uniform-k3:/control:rw" \
    -v "${experiment_root}/results:/results:rw" \
    -v "${experiment_root}/corpus-cache/huggingface:/hf-corpus:rw" \
    -v "${experiment_root}/dependencies/kld-datasets-pydeps:/kld-pydeps:ro" \
    -v "${experiment_root}/runtime-cache/glm52-exl3-r7-fused-bf16-reference-kld:/cache:rw" \
    verdictai/glm52-exl3-sparkinfer:v39-r28-r7fused-broadcast-cu132-sm120a \
    /workspace/qsrt/scripts/run_glm52_paired_expert_intervention_kld.py \
    --model /model \
    --reference-logits /reference \
    --intervention-artifact /artifact \
    --control /control/control.json \
    --dest /results/glm52-layer3-frozen8-uniform-k3-paired-bf16-reference-kld-lower-memory \
    --context-length 2048 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.92 \
    --dtype bfloat16 \
    --kv-cache-dtype nvfp4_ds_mla \
    --load-format safetensors \
    --quantization exl3 \
    --attention-backend B12X_MLA_SPARSE \
    --max-model-len 2049 \
    --max-num-batched-tokens 2048 \
    --kld-chunk-rows 16 \
    --hf-overrides '{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}' \
    --llm-extra-json '{"decode_context_parallel_size":1,"moe_backend":"b12x","enforce_eager":true,"disable_custom_all_reduce":true,"async_scheduling":false}'
)"

docker inspect "${container_name}" > "${record_dir}/container-created-inspect.json"
sha256sum "${record_dir}/container-created-inspect.json"
printf '%s\n' "${container_id}"

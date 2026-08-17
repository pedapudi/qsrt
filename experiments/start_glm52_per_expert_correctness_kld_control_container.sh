#!/usr/bin/env bash
set -euo pipefail

container_name="qsrt-glm52-layer3-per-expert-exl3-engine-kld-repeatability-control"
experiment_root="/home/sunil/qsrt-glm52-experiments"
record_dir="${experiment_root}/launch-records/glm52-layer3-per-expert-exl3-engine-kld-repeatability-control"
validation_report="${experiment_root}/preflight/glm52-exl3-host-local-copy/validation.json"
capture_manifest="${experiment_root}/captures/glm52-layer3-wikitext-document-disjoint-routed-inputs/manifest.json"
result_path="${experiment_root}/results/glm52-layer3-per-expert-exl3-engine-kld-paired-bf16-reference-kld-repeatability-control"

test -f "${validation_report}"
test -f "${capture_manifest}"
test ! -e "${result_path}"
test "$(docker inspect "${container_name}" --format '{{.State.Status}}')" = "created"
test "$(docker inspect "${container_name}" --format '{{.HostConfig.NetworkMode}}')" = "none"
mkdir -p \
  "${record_dir}/source-snapshot/qsrt" \
  "${record_dir}/source-snapshot/scripts" \
  "${record_dir}/source-snapshot/experiments"
cp \
  "${experiment_root}/source/qsrt-working-tree/qsrt/glm52_engine_kld.py" \
  "${experiment_root}/source/qsrt-working-tree/qsrt/glm52_expert_intervention_runtime.py" \
  "${record_dir}/source-snapshot/qsrt/"
cp \
  "${experiment_root}/source/qsrt-working-tree/scripts/run_glm52_paired_expert_intervention_kld.py" \
  "${record_dir}/source-snapshot/scripts/"
cp \
  "${experiment_root}/source/qsrt-working-tree/experiments/create_glm52_per_expert_correctness_kld_control_container.sh" \
  "${experiment_root}/source/qsrt-working-tree/experiments/start_glm52_per_expert_correctness_kld_control_container.sh" \
  "${record_dir}/source-snapshot/experiments/"
docker inspect "${container_name}" > "${record_dir}/container-before-start-inspect.json"

python3 -c '
import json
from pathlib import Path

root = Path("/home/sunil/qsrt-glm52-experiments")
validation = json.loads((root / "preflight/glm52-exl3-host-local-copy/validation.json").read_text())
capture = json.loads((root / "captures/glm52-layer3-wikitext-document-disjoint-routed-inputs/manifest.json").read_text())
container = json.loads((root / "launch-records/glm52-layer3-per-expert-exl3-engine-kld-repeatability-control/container-before-start-inspect.json").read_text())[0]
assert validation["status"] == "passed"
assert validation["network_transfer"] is False
assert capture["schema"] == "qsrt_glm52_layer_input_capture_manifest"
assert capture["schema_version"] == 1
assert capture["status"] == "complete"
assert capture["collections"] == {"activation_fit": 32, "candidate_selection": 8}
assert container["State"]["Status"] == "created"
assert container["HostConfig"]["NetworkMode"] == "none"
assert container["Image"] == "sha256:12f86065d7fe64d30dad678585e68c91f47f1f2a32bed45ccaf108382f3928ac"
environment = set(container["Config"]["Env"])
for required in (
    "VLLM_USE_V2_MODEL_RUNNER=0",
    "VLLM_USE_B12X_SPARSE_INDEXER=0",
    "VLLM_EXL3_R7_FUSED=0",
    "QSRT_GLM52_FORCE_PER_EXPERT_EXL3_MOE=1",
    "QSRT_GLM52_ENGINE_KLD_REFERENCE_PATH=/reference/logits_0.safetensors",
    "QSRT_GLM52_ENGINE_KLD_REFERENCE_KEY=logits",
    "QSRT_GLM52_ENGINE_KLD_CHUNK_ROWS=4",
):
    assert required in environment
mount_modes = {item["Destination"]: item["Mode"] for item in container["Mounts"]}
for destination in ("/model", "/reference", "/artifact", "/workspace/qsrt", "/kld-pydeps"):
    assert mount_modes[destination] == "ro"
labels = container["Config"]["Labels"]
assert labels["qsrt.exl3-moe-execution"] == "three-gemm-per-expert-correctness"
assert labels["qsrt.exl3-weight-preparation"] == "raw-per-expert-without-fused-staging"
assert labels["qsrt.gpu-memory-utilization"] == "0.89"
assert labels["qsrt.prompt-logprob-chunk-tokens"] == "2048"
assert labels["qsrt.kld-reduction-device"] == "cuda-0"
assert labels["qsrt.kld-result-transport"] == "prompt-logprob-scalar-channel"
assert labels["qsrt.attention-contract"] == "dense-triton-mla"
assert labels["qsrt.source-sparse-index-topk"] == "2048"
assert labels["qsrt.scored-context-tokens"] == "2048"
assert labels["qsrt.model-downloads-performed"] == "false"
'

docker start "${container_name}"
docker inspect "${container_name}" > "${record_dir}/container-started-inspect.json"
sha256sum \
  "${validation_report}" \
  "${capture_manifest}" \
  "${record_dir}/container-before-start-inspect.json" \
  "${record_dir}/container-started-inspect.json" \
  "${record_dir}/source-snapshot/qsrt/glm52_engine_kld.py" \
  "${record_dir}/source-snapshot/qsrt/glm52_expert_intervention_runtime.py" \
  "${record_dir}/source-snapshot/scripts/run_glm52_paired_expert_intervention_kld.py" \
  "${record_dir}/source-snapshot/experiments/create_glm52_per_expert_correctness_kld_control_container.sh" \
  "${record_dir}/source-snapshot/experiments/start_glm52_per_expert_correctness_kld_control_container.sh"

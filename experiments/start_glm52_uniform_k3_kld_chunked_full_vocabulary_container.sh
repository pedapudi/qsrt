#!/usr/bin/env bash
set -euo pipefail

container_name="qsrt-glm52-layer3-uniform-k3-kld-dense-triton-r7-control-repeat-direct-identity"
experiment_root="/home/sunil/qsrt-glm52-experiments"
record_dir="${experiment_root}/launch-records/glm52-layer3-frozen8-uniform-k3-kld-dense-triton-r7-control-repeat-direct-identity"
validation_report="${experiment_root}/preflight/glm52-exl3-host-local-copy/validation.json"
capture_manifest="${experiment_root}/captures/glm52-layer3-wikitext-document-disjoint-routed-inputs/manifest.json"
result_path="${experiment_root}/results/glm52-layer3-frozen8-uniform-k3-paired-bf16-reference-kld-dense-triton-r7-control-repeat-direct-identity"

test -f "${validation_report}"
test -f "${capture_manifest}"
test ! -e "${result_path}"
test "$(docker inspect "${container_name}" --format '{{.State.Status}}')" = "created"
test "$(docker inspect "${container_name}" --format '{{.HostConfig.NetworkMode}}')" = "none"
docker inspect "${container_name}" > "${record_dir}/container-before-start-inspect.json"

python3 -c '
import json
from pathlib import Path

validation = json.loads(Path("/home/sunil/qsrt-glm52-experiments/preflight/glm52-exl3-host-local-copy/validation.json").read_text())
capture = json.loads(Path("/home/sunil/qsrt-glm52-experiments/captures/glm52-layer3-wikitext-document-disjoint-routed-inputs/manifest.json").read_text())
container = json.loads(Path("/home/sunil/qsrt-glm52-experiments/launch-records/glm52-layer3-frozen8-uniform-k3-kld-dense-triton-r7-control-repeat-direct-identity/container-before-start-inspect.json").read_text())[0]
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
assert "VLLM_USE_V2_MODEL_RUNNER=0" in environment
assert "VLLM_EXL3_PREFILL_CAPACITY=256" in environment
assert "VLLM_USE_B12X_SPARSE_INDEXER=0" in environment
assert "VLLM_EXL3_R7_FUSED=0" in environment
assert "QSRT_GLM52_ACTIVATION_CAPTURE_DIR=/captures/glm52-layer3-wikitext-document-disjoint-routed-inputs" not in environment
mount_modes = {item["Destination"]: item["Mode"] for item in container["Mounts"]}
for destination in ("/model", "/reference", "/artifact", "/workspace/qsrt", "/kld-pydeps"):
    assert mount_modes[destination] == "ro"
assert container["Config"]["Labels"]["qsrt.gpu-memory-utilization"] == "0.92"
assert container["Config"]["Labels"]["qsrt.prompt-logprob-chunk-tokens"] == "256"
assert container["Config"]["Labels"]["qsrt.exl3-prefill-capacity"] == "256"
assert container["Config"]["Labels"]["qsrt.attention-contract"] == "dense-triton-mla"
assert container["Config"]["Labels"]["qsrt.exl3-r7-path"] == "exl3-moe-r7-fused-control"
assert container["Config"]["Labels"]["qsrt.source-sparse-index-topk"] == "2048"
assert container["Config"]["Labels"]["qsrt.scored-context-tokens"] == "2048"
assert container["Config"]["Labels"]["qsrt.repeatability-control"] == "resident-off-repeat"
assert container["Config"]["Labels"]["qsrt.identity-control"] == "direct-return-resident-output"
assert container["Config"]["Labels"]["qsrt.model-downloads-performed"] == "false"
'

docker start "${container_name}"
docker inspect "${container_name}" > "${record_dir}/container-started-inspect.json"
sha256sum \
  "${validation_report}" \
  "${capture_manifest}" \
  "${record_dir}/container-before-start-inspect.json" \
  "${record_dir}/container-started-inspect.json"

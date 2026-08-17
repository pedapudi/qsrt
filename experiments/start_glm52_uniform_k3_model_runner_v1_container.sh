#!/usr/bin/env bash
set -euo pipefail

container_name="qsrt-glm52-layer3-uniform-k3-capture-kld-model-runner-v1"
experiment_root="/home/sunil/qsrt-glm52-experiments"
record_dir="${experiment_root}/launch-records/glm52-layer3-frozen8-uniform-k3-model-runner-v1"
validation_report="${experiment_root}/preflight/glm52-exl3-host-local-copy/validation.json"
capture_path="${experiment_root}/captures/glm52-layer3-wikitext-document-disjoint-routed-inputs"
result_path="${experiment_root}/results/glm52-layer3-frozen8-uniform-k3-paired-bf16-reference-kld-model-runner-v1"

test -f "${validation_report}"
test ! -e "${capture_path}"
test ! -e "${result_path}"
test "$(docker inspect "${container_name}" --format '{{.State.Status}}')" = "created"
test "$(docker inspect "${container_name}" --format '{{.HostConfig.NetworkMode}}')" = "none"
docker inspect "${container_name}" > "${record_dir}/container-before-start-inspect.json"

python3 -c '
import json
from pathlib import Path

report = json.loads(Path("/home/sunil/qsrt-glm52-experiments/preflight/glm52-exl3-host-local-copy/validation.json").read_text())
container = json.loads(Path("/home/sunil/qsrt-glm52-experiments/launch-records/glm52-layer3-frozen8-uniform-k3-model-runner-v1/container-before-start-inspect.json").read_text())[0]
assert report["status"] == "passed"
assert report["network_transfer"] is False
assert report["source"] == "/home/sunil/usb-mnt/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78"
assert report["destination"] == "/home/sunil/qsrt-glm52-experiments/model-cache/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78-nvme"
assert report["r7_shard_count"] == 75
assert report["large_non_r7_sha256_count"] >= 1
assert container["State"]["Status"] == "created"
assert container["HostConfig"]["NetworkMode"] == "none"
assert container["Image"] == "sha256:12f86065d7fe64d30dad678585e68c91f47f1f2a32bed45ccaf108382f3928ac"
environment = set(container["Config"]["Env"])
assert "VLLM_USE_V2_MODEL_RUNNER=0" in environment
mount_modes = {item["Destination"]: item["Mode"] for item in container["Mounts"]}
for destination in ("/model", "/reference", "/artifact", "/workspace/qsrt", "/kld-pydeps"):
    assert mount_modes[destination] == "ro"
assert container["Config"]["Labels"]["qsrt.model-downloads-performed"] == "false"
'

docker start "${container_name}"
docker inspect "${container_name}" > "${record_dir}/container-started-inspect.json"
sha256sum \
  "${validation_report}" \
  "${record_dir}/container-before-start-inspect.json" \
  "${record_dir}/container-started-inspect.json"

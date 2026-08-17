#!/usr/bin/env bash
set -euo pipefail

# Compare the already-frozen uniform-K3 and routed-input-curvature artifacts
# against official BF16 expert functions on the reporting-only layer input.
# This diagnosis asks whether complete-expert output error and full-model KLD
# rank the two artifacts in the same order. It must not select a later codec
# candidate.

experiment_root="/home/sunil/qsrt-glm52-experiments"
source_copy="${experiment_root}/source/qsrt-working-tree"
source_window="/home/sunil/usb-mnt/zai-org/GLM-5.2-b4734de-layer003-source-window"
source_inventory="${experiment_root}/model-cache/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78-nvme/reproducibility/r10/inventories/source_inventory.json"
uniform_artifact="${experiment_root}/results/glm52-layer3-frozen8-dense-endpoints-r7-closure-merged-v2"
comparison_artifact="${experiment_root}/results/glm52-layer3-frozen8-routed-input-curvature-merged"
capture_root="${experiment_root}/captures/glm52-layer3-published-reference-reporting-inputs"
uniform_kld="${experiment_root}/results/glm52-layer3-frozen8-dense-endpoints-r7-closure-merged-v2-paired-bf16-reference-kld-engine-per-expert-correctness/report.json"
comparison_kld="${experiment_root}/results/glm52-layer3-frozen8-routed-input-curvature-merged-paired-bf16-reference-kld-engine-per-expert-correctness/report.json"
result_name="glm52-layer3-frozen8-uniform-versus-routed-input-curvature-reporting-output-comparison"
result_root="${experiment_root}/results/${result_name}"
record_root="${experiment_root}/launch-records/${result_name}"
snapshot_root="${record_root}/source-snapshot"
container_name="qsrt-${result_name}"

for path in \
  "${source_window}/config.json" \
  "${source_window}/model.safetensors.index.json" \
  "${source_inventory}" \
  "${uniform_artifact}/manifest.json" \
  "${uniform_artifact}/report.json" \
  "${comparison_artifact}/manifest.json" \
  "${comparison_artifact}/report.json" \
  "${capture_root}/manifest.json" \
  "${uniform_kld}" \
  "${comparison_kld}"
do
  test -f "${path}"
done
test ! -e "${result_root}"
test ! -e "${record_root}"
test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"

mkdir -p "${snapshot_root}"
rsync -a \
  --exclude .git \
  --exclude .venv \
  --exclude out \
  --exclude __pycache__ \
  --exclude '*.pyc' \
  --exclude .pytest_cache \
  "${source_copy}/" "${snapshot_root}/"
(
  cd "${snapshot_root}"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > "${record_root}/source-files.sha256"

docker create \
  --name "${container_name}" \
  --label qsrt.experiment=glm52-layer3-reporting-complete-expert-output-comparison \
  --label qsrt.reporting-context-use=diagnosis-only-candidate-selection-prohibited \
  --label qsrt.model-downloads-performed=false \
  --network none \
  --gpus device=0 \
  --ipc host \
  --shm-size 16g \
  --entrypoint /opt/venv/bin/python \
  -e PYTHONPATH=/workspace/qsrt \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 \
  -v "${source_window}:/source:ro" \
  -v "${source_inventory}:/source-inventory.json:ro" \
  -v "${uniform_artifact}:/uniform-artifact:ro" \
  -v "${comparison_artifact}:/comparison-artifact:ro" \
  -v "${capture_root}:/reporting-capture:ro" \
  -v "${uniform_kld}:/uniform-kld-report.json:ro" \
  -v "${comparison_kld}:/comparison-kld-report.json:ro" \
  -v "${snapshot_root}:/workspace/qsrt:ro" \
  -v "${experiment_root}/results:/results:rw" \
  voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34 \
  /workspace/qsrt/scripts/score_glm52_reporting_expert_outputs.py \
  --source-root /source \
  --source-inventory /source-inventory.json \
  --uniform-artifact /uniform-artifact \
  --comparison-artifact /comparison-artifact \
  --reporting-capture /reporting-capture \
  --uniform-kld-report /uniform-kld-report.json \
  --comparison-kld-report /comparison-kld-report.json \
  --dest "/results/${result_name}" \
  --device cuda:0

docker inspect "${container_name}" > "${record_root}/container-created-inspect.json"
python3 -c '
import json
import sys
from pathlib import Path

container = json.loads(Path(sys.argv[1]).read_text())[0]
assert container["State"]["Status"] == "created"
assert container["HostConfig"]["NetworkMode"] == "none"
assert container["Image"] == "sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"
assert container["Config"]["Labels"]["qsrt.reporting-context-use"] == "diagnosis-only-candidate-selection-prohibited"
mount_modes = {item["Destination"]: item["Mode"] for item in container["Mounts"]}
for destination in (
    "/source",
    "/source-inventory.json",
    "/uniform-artifact",
    "/comparison-artifact",
    "/reporting-capture",
    "/uniform-kld-report.json",
    "/comparison-kld-report.json",
    "/workspace/qsrt",
):
    assert mount_modes[destination] == "ro"
assert mount_modes["/results"] == "rw"
' "${record_root}/container-created-inspect.json"

docker start "${container_name}"
docker inspect "${container_name}" > "${record_root}/container-started-inspect.json"
sha256sum \
  "${source_window}/config.json" \
  "${source_window}/model.safetensors.index.json" \
  "${source_inventory}" \
  "${uniform_artifact}/manifest.json" \
  "${uniform_artifact}/report.json" \
  "${comparison_artifact}/manifest.json" \
  "${comparison_artifact}/report.json" \
  "${capture_root}/manifest.json" \
  "${uniform_kld}" \
  "${comparison_kld}" \
  "${record_root}/source-files.sha256" \
  "${record_root}/container-created-inspect.json" \
  "${record_root}/container-started-inspect.json"

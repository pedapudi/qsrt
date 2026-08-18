#!/usr/bin/env bash
set -euo pipefail

# Build one disjoint GLM-5.2 down-construction slice on one physical GPU.

if test "$#" -ne 5; then
  echo "usage: $0 <identity|reconstructed_input_covariance> <source_weights|reconstructed_activation_refit> <physical-gpu> <panel-offset> <expert-count>" >&2
  exit 2
fi

input_metric="$1"
target="$2"
physical_gpu="$3"
panel_offset="$4"
expert_count="$5"

case "${input_metric}" in
  identity|reconstructed_input_covariance) ;;
  *) echo "unsupported input metric: ${input_metric}" >&2; exit 2 ;;
esac
case "${target}" in
  source_weights|reconstructed_activation_refit) ;;
  *) echo "unsupported target: ${target}" >&2; exit 2 ;;
esac
case "${physical_gpu}" in 0|1|2|3) ;; *) echo "physical GPU must be 0 through 3" >&2; exit 2 ;; esac
case "${panel_offset}" in 0|1|2|3|4|5|6|7) ;; *) echo "panel offset must be 0 through 7" >&2; exit 2 ;; esac
case "${expert_count}" in 1|2) ;; *) echo "expert count must be one or two" >&2; exit 2 ;; esac
if test $((panel_offset + expert_count)) -gt 8; then
  echo "panel slice exceeds the frozen eight-expert panel" >&2
  exit 2
fi

experiment_root="/home/sunil/qsrt-glm52-experiments"
source_root="/home/sunil/usb-mnt/zai-org/GLM-5.2-b4734de-layer003-source-window"
endpoint_root="/home/sunil/usb-mnt/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78"
source_copy="${experiment_root}/source/qsrt-working-tree"
input_artifact="${experiment_root}/results/glm52-layer3-frozen8-dense-endpoints-r7-closure-merged-v2"
capture_root="${experiment_root}/captures/glm52-layer3-wikitext-document-disjoint-routed-inputs"
results_root="${experiment_root}/results"
dependency_root="${experiment_root}/dependencies/exllamav3-v0.0.43-source"
image="sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"

end=$((panel_offset + expert_count - 1))
printf -v offset_text "%02d" "${panel_offset}"
printf -v end_text "%02d" "${end}"
construction="${input_metric}__${target}"
slice_name="glm52-layer3-frozen8-down-construction-${construction}-slice-${offset_text}-${end_text}"
container_name="qsrt-${slice_name}"
destination="${results_root}/${slice_name}"
record_root="${experiment_root}/launch-records/${slice_name}"

test ! -e "${destination}"
test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"
test "$(docker image inspect "${image}" --format '{{.Id}}')" = "${image}"
test -f "${source_root}/model.safetensors.index.json"
test -f "${endpoint_root}/reproducibility/r10/inventories/source_inventory.json"
test -f "${input_artifact}/report.json"
test -f "${capture_root}/manifest.json"
test -d "${dependency_root}/exllamav3"
mkdir -p "${record_root}"

python3 -c '
import json
from pathlib import Path

capture = json.loads(Path("/home/sunil/qsrt-glm52-experiments/captures/glm52-layer3-wikitext-document-disjoint-routed-inputs/manifest.json").read_text())
assert capture["schema"] == "qsrt_glm52_layer_input_capture_manifest"
assert capture["schema_version"] == 1
assert capture["status"] == "complete"
assert capture["model_layer"] == 3
assert capture["collections"] == {"activation_fit": 32, "candidate_selection": 8}
'

docker create \
  --name "${container_name}" \
  --label qsrt.experiment=glm52-layer3-down-construction-comparison \
  --label qsrt.model-downloads-performed=false \
  --label qsrt.input-metric="${input_metric}" \
  --label qsrt.target="${target}" \
  --label qsrt.panel-offset="${panel_offset}" \
  --network none \
  --gpus "device=${physical_gpu}" \
  --ipc host \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --entrypoint /opt/venv/bin/python \
  -e PYTHONPATH=/workspace/qsrt:/exllamav3-python:/opt/exllamav3 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -v "${source_root}:/source:ro" \
  -v "${endpoint_root}:/endpoint:ro" \
  -v "${source_copy}:/workspace/qsrt:ro" \
  -v "${input_artifact}:/artifact:ro" \
  -v "${capture_root}:/capture:ro" \
  -v "${dependency_root}:/exllamav3-python:ro" \
  -v "${results_root}:/results:rw" \
  "${image}" \
  /workspace/qsrt/scripts/build_glm52_down_construction_comparison.py \
  --source /source \
  --source-inventory /endpoint/reproducibility/r10/inventories/source_inventory.json \
  --input-artifact /artifact \
  --capture /capture \
  --panel-manifest /workspace/qsrt/experiments/glm52_layer3_rate_pattern_panel.json \
  --dest "/results/${slice_name}" \
  --layer 3 \
  --experts "${expert_count}" \
  --panel-offset "${panel_offset}" \
  --input-metric "${input_metric}" \
  --target "${target}" \
  --ridge-factors 0.001,0.01,0.1,1.0 \
  --covariance-identity-shrinkage 0.01 \
  --local-tail-relative-tolerance 0.0 \
  --device cuda:0 \
  --exllamav3-root /exllamav3-python \
  --skip-source-shard-hashes

docker inspect "${container_name}" > "${record_root}/container-created-inspect.json"
docker start --attach "${container_name}" 2>&1 | tee "${record_root}/container.log"
docker inspect "${container_name}" > "${record_root}/container-completed-inspect.json"
python3 - "${record_root}/container-completed-inspect.json" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1]))[0]
assert record["State"]["ExitCode"] == 0, record["State"]
assert not record["State"]["OOMKilled"], record["State"]
assert record["HostConfig"]["NetworkMode"] == "none"
PY
sha256sum \
  "${destination}/manifest.json" \
  "${destination}/report.json" \
  "${record_root}/container-created-inspect.json" \
  "${record_root}/container-completed-inspect.json" \
  "${record_root}/container.log"

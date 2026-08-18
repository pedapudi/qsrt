#!/usr/bin/env bash
set -euo pipefail

# Fit one disjoint GLM-5.2 down-only low-rank adapter slice on one GPU.

if test "$#" -lt 5 || test "$#" -gt 6; then
  echo "usage: $0 <uniform_k3|reconstructed_activation_down_refit> <2|4> <physical-gpu> <panel-offset> <expert-count> [dense_screen|factorized_runtime]" >&2
  exit 2
fi

base_construction="$1"
rank="$2"
physical_gpu="$3"
panel_offset="$4"
expert_count="$5"
artifact_role="${6:-dense_screen}"

case "${base_construction}" in
  uniform_k3)
    input_artifact_name="glm52-layer3-frozen8-dense-endpoints-r7-closure-merged-v2"
    ;;
  reconstructed_activation_down_refit)
    input_artifact_name="glm52-layer3-frozen8-reconstructed-activation-down-refit-merged"
    ;;
  *) echo "unsupported base construction: ${base_construction}" >&2; exit 2 ;;
esac
case "${rank}" in 2|4) ;; *) echo "adapter rank must be two or four" >&2; exit 2 ;; esac
case "${physical_gpu}" in 0|1|2|3) ;; *) echo "physical GPU must be 0 through 3" >&2; exit 2 ;; esac
case "${panel_offset}" in 0|1|2|3|4|5|6|7) ;; *) echo "panel offset must be 0 through 7" >&2; exit 2 ;; esac
case "${expert_count}" in 1|2) ;; *) echo "expert count must be one or two" >&2; exit 2 ;; esac
if test $((panel_offset + expert_count)) -gt 8; then
  echo "panel slice exceeds the frozen eight-expert panel" >&2
  exit 2
fi
case "${artifact_role}" in
  dense_screen) artifact_suffix="" ;;
  factorized_runtime) artifact_suffix="-factorized-runtime-v1" ;;
  *) echo "artifact role must be dense_screen or factorized_runtime" >&2; exit 2 ;;
esac

experiment_root="/home/sunil/qsrt-glm52-experiments"
source_root="/home/sunil/usb-mnt/zai-org/GLM-5.2-b4734de-layer003-source-window"
endpoint_root="/home/sunil/usb-mnt/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78"
source_copy="${experiment_root}/source/qsrt-working-tree"
input_artifact="${experiment_root}/results/${input_artifact_name}"
capture_root="${experiment_root}/captures/glm52-layer3-wikitext-document-disjoint-routed-inputs"
results_root="${experiment_root}/results"
image="sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"

end=$((panel_offset + expert_count - 1))
printf -v offset_text "%02d" "${panel_offset}"
printf -v end_text "%02d" "${end}"
slice_name="glm52-layer3-frozen8-low-rank-down-${base_construction}-bf16-rank-${rank}${artifact_suffix}-slice-${offset_text}-${end_text}"
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
  --label qsrt.experiment=glm52-layer3-activation-weighted-low-rank-down-adapter \
  --label qsrt.model-downloads-performed=false \
  --label qsrt.base-construction="${base_construction}" \
  --label qsrt.adapter-rank="${rank}" \
  --label qsrt.artifact-role="${artifact_role}" \
  --label qsrt.panel-offset="${panel_offset}" \
  --network none \
  --gpus "device=${physical_gpu}" \
  --ipc host \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --entrypoint /opt/venv/bin/python \
  -e PYTHONPATH=/workspace/qsrt \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -v "${source_root}:/source:ro" \
  -v "${endpoint_root}:/endpoint:ro" \
  -v "${source_copy}:/workspace/qsrt:ro" \
  -v "${input_artifact}:/artifact:ro" \
  -v "${capture_root}:/capture:ro" \
  -v "${results_root}:/results:rw" \
  "${image}" \
  /workspace/qsrt/scripts/build_glm52_low_rank_down_adapter.py \
  --source /source \
  --source-inventory /endpoint/reproducibility/r10/inventories/source_inventory.json \
  --input-artifact /artifact \
  --capture /capture \
  --panel-manifest /workspace/qsrt/experiments/glm52_layer3_rate_pattern_panel.json \
  --dest "/results/${slice_name}" \
  --base-construction "${base_construction}" \
  --rank "${rank}" \
  --layer 3 \
  --experts "${expert_count}" \
  --panel-offset "${panel_offset}" \
  --ridge-factors 0.001,0.01,0.1,1.0 \
  --oversampling 8 \
  --power-iterations 2 \
  --batch-rows 2048 \
  --seed 20260818 \
  --device cuda:0 \
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

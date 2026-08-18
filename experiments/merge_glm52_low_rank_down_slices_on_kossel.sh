#!/usr/bin/env bash
set -euo pipefail

# Merge four checked two-expert low-rank adapter slices without changing bytes.

if test "$#" -lt 2 || test "$#" -gt 3; then
  echo "usage: $0 <uniform_k3|reconstructed_activation_down_refit> <2|4> [dense_screen|factorized_runtime]" >&2
  exit 2
fi

base_construction="$1"
rank="$2"
artifact_role="${3:-dense_screen}"
case "${base_construction}" in uniform_k3|reconstructed_activation_down_refit) ;; *) exit 2 ;; esac
case "${rank}" in 2|4) ;; *) exit 2 ;; esac
case "${artifact_role}" in
  dense_screen) artifact_suffix="" ;;
  factorized_runtime) artifact_suffix="-factorized-runtime-v1" ;;
  *) echo "artifact role must be dense_screen or factorized_runtime" >&2; exit 2 ;;
esac

experiment_root="/home/sunil/qsrt-glm52-experiments"
source_copy="${experiment_root}/source/qsrt-working-tree"
results_root="${experiment_root}/results"
image="sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"
slice_stem="glm52-layer3-frozen8-low-rank-down-${base_construction}-bf16-rank-${rank}${artifact_suffix}-slice"
merged_name="glm52-layer3-frozen8-low-rank-down-${base_construction}-bf16-rank-${rank}${artifact_suffix}-merged"
destination="${results_root}/${merged_name}"
record_root="${experiment_root}/launch-records/${merged_name}"
container_name="qsrt-${merged_name}"
inputs=(
  "/results/${slice_stem}-00-01"
  "/results/${slice_stem}-02-03"
  "/results/${slice_stem}-04-05"
  "/results/${slice_stem}-06-07"
)

test ! -e "${destination}"
test -z "$(docker ps -a --filter "name=^/${container_name}$" -q)"
for input in "${inputs[@]}"; do
  test -f "${results_root}/${input#/results/}/report.json"
done
mkdir -p "${record_root}"

docker create \
  --name "${container_name}" \
  --label qsrt.experiment=glm52-layer3-activation-weighted-low-rank-down-adapter-merge \
  --label qsrt.model-downloads-performed=false \
  --label qsrt.base-construction="${base_construction}" \
  --label qsrt.adapter-rank="${rank}" \
  --label qsrt.artifact-role="${artifact_role}" \
  --network none \
  --entrypoint /opt/venv/bin/python \
  -e PYTHONPATH=/workspace/qsrt \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "${source_copy}:/workspace/qsrt:ro" \
  -v "${results_root}:/results:rw" \
  "${image}" \
  /workspace/qsrt/scripts/merge_glm52_dense_expert_interventions.py \
  "${inputs[@]}" \
  --dest "/results/${merged_name}" \
  --panel-manifest /workspace/qsrt/experiments/glm52_layer3_rate_pattern_panel.json \
  --layer 3

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
assert record["Config"]["Labels"]["qsrt.model-downloads-performed"] == "false"
PY
sha256sum \
  "${destination}/manifest.json" \
  "${destination}/report.json" \
  "${record_root}/container-created-inspect.json" \
  "${record_root}/container-completed-inspect.json" \
  "${record_root}/container.log"

#!/usr/bin/env bash
set -euo pipefail

# Merge four checked two-expert down-construction slices without changing bytes.

if test "$#" -ne 2; then
  echo "usage: $0 <identity|reconstructed_input_covariance> <source_weights|reconstructed_activation_refit>" >&2
  exit 2
fi

input_metric="$1"
target="$2"
case "${input_metric}" in identity|reconstructed_input_covariance) ;; *) exit 2 ;; esac
case "${target}" in source_weights|reconstructed_activation_refit) ;; *) exit 2 ;; esac

experiment_root="/home/sunil/qsrt-glm52-experiments"
source_copy="${experiment_root}/source/qsrt-working-tree"
results_root="${experiment_root}/results"
image="sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"
construction="${input_metric}__${target}"
slice_stem="glm52-layer3-frozen8-down-construction-${construction}-slice"
merged_name="glm52-layer3-frozen8-down-construction-${construction}-merged"
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
  --label qsrt.experiment=glm52-layer3-down-construction-merge \
  --label qsrt.model-downloads-performed=false \
  --label qsrt.input-metric="${input_metric}" \
  --label qsrt.target="${target}" \
  --network none \
  --entrypoint /opt/venv/bin/python \
  -e PYTHONPATH=/workspace/qsrt \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 \
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
PY
sha256sum \
  "${destination}/manifest.json" \
  "${destination}/report.json" \
  "${record_root}/container-created-inspect.json" \
  "${record_root}/container-completed-inspect.json" \
  "${record_root}/container.log"

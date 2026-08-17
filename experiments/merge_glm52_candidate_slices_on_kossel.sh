#!/usr/bin/env bash
set -euo pipefail

# Merge one complete four-slice candidate panel without changing dense tensor
# bytes. Accepted method names map to fixed, descriptive artifact paths.

if test "$#" -ne 1; then
  echo "usage: $0 {routed-input-curvature|reconstructed-activation-down-refit|blockldlq-no-feedback-frozen-k3-scale}" >&2
  exit 2
fi

method="$1"
experiment_root="/home/sunil/qsrt-glm52-experiments"
source_copy="${experiment_root}/source/qsrt-working-tree"
results_root="${experiment_root}/results"
image="sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b"

case "${method}" in
  routed-input-curvature)
    slice_stem="glm52-layer3-frozen8-routed-input-curvature-exllamav3-package-slice"
    merged_name="glm52-layer3-frozen8-routed-input-curvature-merged"
    ;;
  reconstructed-activation-down-refit)
    slice_stem="glm52-layer3-frozen8-reconstructed-activation-down-refit-slice"
    merged_name="glm52-layer3-frozen8-reconstructed-activation-down-refit-merged"
    ;;
  blockldlq-no-feedback-frozen-k3-scale)
    slice_stem="glm52-layer3-frozen8-blockldlq-no-feedback-frozen-k3-scale-slice"
    merged_name="glm52-layer3-frozen8-blockldlq-no-feedback-frozen-k3-scale-merged"
    ;;
  *)
    echo "unknown candidate method: ${method}" >&2
    exit 2
    ;;
esac

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
  --label qsrt.experiment="glm52-layer3-${method}-merge" \
  --label qsrt.model-downloads-performed=false \
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
docker start --attach "${container_name}"
docker inspect "${container_name}" > "${record_root}/container-completed-inspect.json"
sha256sum \
  "${destination}/manifest.json" \
  "${destination}/report.json" \
  "${record_root}/container-created-inspect.json" \
  "${record_root}/container-completed-inspect.json"

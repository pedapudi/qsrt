#!/usr/bin/env bash
set -euo pipefail

# Wait for the guarded uniform-K3 build process, require all four merged
# artifacts, then run down refits followed by rank-four down corrections.

if test "$#" != 1; then
  echo "usage: $0 <uniform-k3-process-id>" >&2
  exit 2
fi

uniform_k3_pid="$1"
case "${uniform_k3_pid}" in
  ''|*[!0-9]*) echo "uniform-K3 process ID must be numeric" >&2; exit 2 ;;
esac

experiment_root="/home/sunil/qsrt-glm52-experiments"
results_root="${experiment_root}/results"
down_refit_script="${experiment_root}/launch-scripts/run_glm52_layers60_64_52_63_down_refit_and_merge_on_kossel.sh"
rank4_script="${experiment_root}/launch-scripts/run_glm52_layers60_64_52_63_rank4_down_adapter_and_merge_on_kossel.sh"

while kill -0 "${uniform_k3_pid}" 2>/dev/null; do
  sleep 30
done

for layer in 60 64 52 63; do
  test -f "${results_root}/glm52-layer${layer}-frozen8-uniform-k3-source-target-capture-sequenced-merged/report.json"
done

bash "${down_refit_script}"
for layer in 60 64 52 63; do
  test -f "${results_root}/glm52-layer${layer}-frozen8-reconstructed-activation-down-refit-merged/report.json"
done

exec bash "${rank4_script}"

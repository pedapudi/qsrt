#!/usr/bin/env bash
set -euo pipefail

# Screen every previously unmeasured, locally accepted rank-four down
# correction from layers 52, 60, and 64 after the active K4-down experiment
# releases all four GPUs. The public references are selection data.

if test "$#" != 1; then
  echo "usage: $0 <predecessor-process-id>" >&2
  exit 2
fi

predecessor_pid="$1"
case "${predecessor_pid}" in
  ''|*[!0-9]*) echo "predecessor process ID must be numeric" >&2; exit 2 ;;
esac

experiment_root="/home/sunil/qsrt-glm52-experiments"
screen_script="${experiment_root}/launch-scripts/run_glm52_complete_panel_public_reference_screen_on_kossel.sh"

while kill -0 "${predecessor_pid}" 2>/dev/null; do
  sleep 30
done

for layer in 52 60 64; do
  artifact="glm52-layer${layer}-frozen8-low-rank-down-reconstructed-activation-refit-derived-input-identity-checked-selection-fallback-bf16-rank-4-merged"
  result="glm52-layer${layer}-rank4-low-rank-down-singleton-model-kld-selection-public-reference"
  plan="glm52_layer${layer}_rank4_low_rank_down_singleton_model_kld_selection_plan.json"
  bash "${screen_script}" "${artifact}" "${result}" "${plan}"
done

#!/usr/bin/env bash
set -euo pipefail

# Wait for all four down-refit singleton screens, then measure the frozen
# layer-63 composition containing the two experts that passed both document
# groups. The public documents remain candidate-selection data.

if test "$#" -ne 1 || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 <singleton-screen-sequence-pid>" >&2
  exit 2
fi

singleton_screen_sequence_pid="$1"
while kill -0 "${singleton_screen_sequence_pid}" 2>/dev/null; do
  sleep 5
done

experiment_root="/home/sunil/qsrt-glm52-experiments"
results_root="${experiment_root}/results"
for layer in 63 52 60 64; do
  singleton_result="glm52-layer${layer}-frozen8-reconstructed-activation-down-refit-merged-singleton-model-kld-selection-public-reference-v2-complete-slice-ancestry"
  test -f "${results_root}/${singleton_result}/report.json"
done
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d')"

artifact="glm52-layer63-frozen8-reconstructed-activation-down-refit-merged"
result="${artifact}-experts149-164-model-kld-selected-composition-public-reference-v1"
plan="glm52_layer63_down_refit_model_kld_retained_composition_plan.json"
bash "${experiment_root}/launch-scripts/run_glm52_complete_panel_public_reference_screen_on_kossel.sh" \
  "${artifact}" "${result}" "${plan}"

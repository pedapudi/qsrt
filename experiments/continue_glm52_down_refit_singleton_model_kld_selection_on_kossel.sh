#!/usr/bin/env bash
set -euo pipefail

# Wait for the complete-panel down-refit screens, then measure every frozen
# candidate expert as a singleton. Each layer uses one resident-model load for
# eight arms. A rejected refit remains the uniform-K3 fallback stored in the
# candidate artifact.

if test "$#" -ne 1 || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 <complete-panel-sequence-pid>" >&2
  exit 2
fi

complete_panel_sequence_pid="$1"
while kill -0 "${complete_panel_sequence_pid}" 2>/dev/null; do
  sleep 5
done

experiment_root="/home/sunil/qsrt-glm52-experiments"
results_root="${experiment_root}/results"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d')"

# Layer 63 runs first because its complete refit panel is the only measured
# high-layer panel with a favorable mean-KLD point estimate. The remaining
# layers retain their error-blind numerical order.
for layer in 63 52 60 64; do
  complete_panel_result="glm52-layer${layer}-frozen8-reconstructed-activation-down-refit-merged-complete-panel-document-disjoint-public-reference-screen-complete-slice-ancestry-validated"
  test -f "${results_root}/${complete_panel_result}/report.json"

  artifact="glm52-layer${layer}-frozen8-reconstructed-activation-down-refit-merged"
  result="${artifact}-singleton-model-kld-selection-public-reference-v2-complete-slice-ancestry"
  plan="glm52_layer${layer}_down_refit_singleton_model_kld_selection_plan.json"
  bash "${experiment_root}/launch-scripts/run_glm52_complete_panel_public_reference_screen_on_kossel.sh" \
    "${artifact}" "${result}" "${plan}"
done

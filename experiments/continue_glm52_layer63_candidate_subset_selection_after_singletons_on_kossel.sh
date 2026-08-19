#!/usr/bin/env bash
set -euo pipefail

# Wait for the two pre-registered singleton screens, verify their atomic
# reports, then evaluate six frozen layer-63 expert subsets in one model load.

if test "$#" -ne 1 || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 <singleton-sequence-pid>" >&2
  exit 2
fi

singleton_sequence_pid="$1"
while kill -0 "${singleton_sequence_pid}" 2>/dev/null; do
  sleep 5
done

experiment_root="/home/sunil/qsrt-glm52-experiments"
results_root="${experiment_root}/results"
for result in \
  glm52-layer63-frozen8-low-rank-down-reconstructed-activation-refit-derived-input-identity-checked-selection-fallback-bf16-rank-4-merged-registered-expert164-document-disjoint-public-reference-screen-ancestry-hash-validated \
  glm52-layer64-frozen8-low-rank-down-reconstructed-activation-refit-derived-input-identity-checked-selection-fallback-bf16-rank-4-merged-registered-expert253-document-disjoint-public-reference-screen-ancestry-hash-validated
do
  test -f "${results_root}/${result}/report.json"
done

test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d')"

artifact="glm52-layer63-frozen8-low-rank-down-reconstructed-activation-refit-derived-input-identity-checked-selection-fallback-bf16-rank-4-merged"
result="${artifact}-six-subset-model-kld-selection-public-reference-v1"
plan="glm52_layer63_low_rank_down_candidate_subset_selection_plan.json"
bash "${experiment_root}/launch-scripts/run_glm52_complete_panel_public_reference_screen_on_kossel.sh" \
  "${artifact}" "${result}" "${plan}"

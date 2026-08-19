#!/usr/bin/env bash
set -euo pipefail

# Run the two singleton corrections frozen from local selection data after the
# complete-panel sequence releases all four GPUs. Each registration is checked
# by the model runner against the stored factor and materialized-down hashes.

if test "$#" != 1; then
  echo "usage: $0 <complete-panel-sequence-process-id>" >&2
  exit 2
fi

predecessor_pid="$1"
case "${predecessor_pid}" in
  ''|*[!0-9]*) echo "measurement process ID must be numeric" >&2; exit 2 ;;
esac

experiment_root="/home/sunil/qsrt-glm52-experiments"
screen_script="${experiment_root}/launch-scripts/run_glm52_complete_panel_public_reference_screen_on_kossel.sh"

while kill -0 "${predecessor_pid}" 2>/dev/null; do
  sleep 30
done

for specification in \
  "63 164 glm52_layer63_rank4_expert164_low_rank_down_public_reference_registration.json" \
  "64 253 glm52_layer64_rank4_expert253_low_rank_down_public_reference_registration.json"; do
  read -r layer expert registration <<<"${specification}"
  artifact="glm52-layer${layer}-frozen8-low-rank-down-reconstructed-activation-refit-derived-input-identity-checked-selection-fallback-bf16-rank-4-merged"
  result="${artifact}-registered-expert${expert}-document-disjoint-public-reference-screen-ancestry-hash-validated"
  bash "${screen_script}" "${artifact}" "${result}" "${registration}"
done

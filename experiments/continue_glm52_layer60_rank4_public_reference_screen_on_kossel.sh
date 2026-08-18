#!/usr/bin/env bash
set -euo pipefail

# Wait for the down-refit and rank-four construction chain, then screen the
# complete frozen layer-60 rank-four panel on public document-disjoint BF16
# references. Missing construction receipts stop the screen.

if test "$#" != 1; then
  echo "usage: $0 <construction-process-id>" >&2
  exit 2
fi

construction_pid="$1"
case "${construction_pid}" in
  ''|*[!0-9]*) echo "construction process ID must be numeric" >&2; exit 2 ;;
esac

experiment_root="/home/sunil/qsrt-glm52-experiments"
artifact_name="glm52-layer60-frozen8-low-rank-down-reconstructed-activation-refit-derived-input-identity-checked-selection-fallback-bf16-rank-4-merged"
result_name="${artifact_name}-complete-panel-document-disjoint-public-reference-screen-ancestry-hash-validated"
screen_script="${experiment_root}/launch-scripts/run_glm52_complete_panel_public_reference_screen_on_kossel.sh"

while kill -0 "${construction_pid}" 2>/dev/null; do
  sleep 30
done

test -f "${experiment_root}/results/${artifact_name}/report.json"
exec bash "${screen_script}" "${artifact_name}" "${result_name}"

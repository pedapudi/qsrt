#!/usr/bin/env bash
set -euo pipefail

# After the layer-60 measurement closes successfully, measure the other three
# frozen eight-expert panels one at a time. Layer 63 runs first because its
# held-out pooled correction is largest. Sequential model loads avoid GPU,
# host-memory, and NVMe contention while preserving independent result roots.

if test "$#" != 1; then
  echo "usage: $0 <layer-60-measurement-process-id>" >&2
  exit 2
fi

predecessor_pid="$1"
case "${predecessor_pid}" in
  ''|*[!0-9]*) echo "measurement process ID must be numeric" >&2; exit 2 ;;
esac

experiment_root="/home/sunil/qsrt-glm52-experiments"
results_root="${experiment_root}/results"
screen_script="${experiment_root}/launch-scripts/run_glm52_complete_panel_public_reference_screen_on_kossel.sh"
suffix="complete-panel-document-disjoint-public-reference-screen-ancestry-hash-validated"
layer60_artifact="glm52-layer60-frozen8-low-rank-down-reconstructed-activation-refit-derived-input-identity-checked-selection-fallback-bf16-rank-4-merged"
layer60_result="${layer60_artifact}-${suffix}"

while kill -0 "${predecessor_pid}" 2>/dev/null; do
  sleep 30
done

python3 - "${results_root}/${layer60_result}/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["status"] == "complete"
assert report["measurement_controls"]["passed"] is True
PY

for layer in 63 52 64; do
  artifact="glm52-layer${layer}-frozen8-low-rank-down-reconstructed-activation-refit-derived-input-identity-checked-selection-fallback-bf16-rank-4-merged"
  result="${artifact}-${suffix}"
  bash "${screen_script}" "${artifact}" "${result}"
done

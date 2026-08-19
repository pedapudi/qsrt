#!/usr/bin/env bash
set -euo pipefail

# Wait for the active multi-layer public-reference run, then measure the
# independently registered layer-63 K4-down candidate. The two runs share the
# established EXL3 numerical-runtime cache and never overlap GPU ownership.

if test "$#" -ne 1; then
  echo "usage: $0 <active-process-id>" >&2
  exit 2
fi
active_pid="$1"
if [[ ! "${active_pid}" =~ ^[1-9][0-9]*$ ]]; then
  echo "active process ID must be a positive integer" >&2
  exit 2
fi

while kill -0 "${active_pid}" 2>/dev/null; do
  sleep 15
done

launch_root="/home/sunil/qsrt-glm52-experiments/launch-scripts"
"${launch_root}/run_glm52_complete_panel_public_reference_screen_on_kossel.sh" \
  glm52-layer63-experts149-164-registered-k3-k3-k4-down-refit-v2-input-ancestry \
  glm52-layer63-experts149-164-registered-k3-k3-k4-down-refit-v2-input-ancestry-public-reference-selection-screen

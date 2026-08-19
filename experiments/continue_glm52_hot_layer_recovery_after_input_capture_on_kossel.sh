#!/usr/bin/env bash
set -euo pipefail

# Wait for the queued four-layer input capture, require its complete manifest,
# and then build and screen layers 55 through 58 in numerical order.

if test "$#" -ne 1 || [[ ! "$1" =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 <input-capture-launcher-process-id>" >&2
  exit 2
fi
capture_launcher_pid="$1"
experiment_root="/home/sunil/qsrt-glm52-experiments"
capture_container="qsrt-glm52-layers55-56-57-58-input-capture"
capture_index="${experiment_root}/captures/glm52-layers55-56-57-58-wikitext-document-disjoint-routed-inputs/manifest.json"
pipeline="${experiment_root}/launch-scripts/build_and_screen_glm52_hot_layer_down_recovery_on_kossel.sh"

while ! docker inspect "${capture_container}" >/dev/null 2>&1; do
  if ! kill -0 "${capture_launcher_pid}" 2>/dev/null; then
    echo "input-capture launcher exited before creating its container" >&2
    exit 3
  fi
  sleep 15
done
while test "$(docker inspect "${capture_container}" --format '{{.State.Running}}')" = "true"; do
  sleep 15
done
test "$(docker inspect "${capture_container}" --format '{{.State.ExitCode}}')" = "0"
test -f "${capture_index}"
test -x "${pipeline}"

for layer in 55 56 57 58; do
  bash "${pipeline}" "${layer}"
done

#!/usr/bin/env bash
set -euo pipefail

if test "$#" -ne 1 || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 <selection-and-composition-process-pid>" >&2
  exit 2
fi

selection_process_pid="$1"
while kill -0 "${selection_process_pid}" 2>/dev/null; do
  sleep 5
done

test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d')"
exec /home/sunil/qsrt-glm52-experiments/launch-scripts/run_glm52_layer63_retained_down_refit_single_reference_on_kossel.sh

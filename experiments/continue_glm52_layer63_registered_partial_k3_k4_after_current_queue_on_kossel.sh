#!/usr/bin/env bash
set -euo pipefail

# Wait for the already-registered high-layer screens, then build and measure
# the registered coherent layer-63 K3/K4 candidate without sharing GPU work.

if test "$#" != 1 || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 <predecessor-pid>" >&2
  exit 2
fi
predecessor_pid="$1"
while kill -0 "${predecessor_pid}" 2>/dev/null; do
  sleep 15
done

experiment_root="/home/sunil/qsrt-glm52-experiments"
launch_root="${experiment_root}/launch-scripts"
"${launch_root}/run_glm52_frozen_expert_subset_single_reference_on_kossel.sh" \
  glm52-layer60-frozen8-reconstructed-activation-down-refit-merged \
  glm52-layer60-down-refit-expert136-single-reference-absolute-target-screen-v1 \
  136
"${launch_root}/build_glm52_layer63_registered_partial_k3_k4_down_refit_on_kossel.sh"
"${launch_root}/run_glm52_complete_panel_public_reference_screen_on_kossel.sh" \
  glm52-layer63-experts149-164-registered-k3-k3-k4-down-refit \
  glm52-layer63-experts149-164-registered-k3-k3-k4-down-refit-public-reference-screen-v1
"${launch_root}/run_glm52_frozen_expert_subset_single_reference_on_kossel.sh" \
  glm52-layer63-experts149-164-registered-k3-k3-k4-down-refit \
  glm52-layer63-experts149-164-registered-k3-k3-k4-down-refit-single-reference-absolute-target-screen-v1

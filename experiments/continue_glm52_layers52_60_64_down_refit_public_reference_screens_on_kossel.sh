#!/usr/bin/env bash
set -euo pipefail

# Test the frozen reconstructed-activation down refit across three additional
# high-layer panels. Each merged artifact passed complete slice-ancestry and
# expert-payload validation before this sequence was launched.

experiment_root="/home/sunil/qsrt-glm52-experiments"
screen_script="${experiment_root}/launch-scripts/run_glm52_complete_panel_public_reference_screen_on_kossel.sh"
test -f "${screen_script}"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d')"

for layer in 52 60 64
do
  artifact="glm52-layer${layer}-frozen8-reconstructed-activation-down-refit-merged"
  result="${artifact}-complete-panel-document-disjoint-public-reference-screen-complete-slice-ancestry-validated"
  bash "${screen_script}" "${artifact}" "${result}"
done

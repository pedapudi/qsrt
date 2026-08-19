#!/usr/bin/env bash
set -euo pipefail

# Decompose the layer-63 low-rank result into the uniform-K3 base, the
# reconstructed-activation down refit, and the already measured low-rank arm.
# Each arm uses the same resident EXL3 model and public-reference protocol.

experiment_root="/home/sunil/qsrt-glm52-experiments"
screen_script="${experiment_root}/launch-scripts/run_glm52_complete_panel_public_reference_screen_on_kossel.sh"
test -f "${screen_script}"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d')"

uniform_artifact="glm52-layer63-frozen8-uniform-k3-source-target-capture-sequenced-merged"
uniform_result="${uniform_artifact}-complete-panel-document-disjoint-public-reference-screen-ancestry-hash-validated"
bash "${screen_script}" "${uniform_artifact}" "${uniform_result}"

refit_artifact="glm52-layer63-frozen8-reconstructed-activation-down-refit-merged"
refit_result="${refit_artifact}-complete-panel-document-disjoint-public-reference-screen-ancestry-hash-validated"
bash "${screen_script}" "${refit_artifact}" "${refit_result}"

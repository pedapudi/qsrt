#!/usr/bin/env bash
set -euo pipefail

# Hold the GPU encoder launch until the multi-layer input-capture container has
# exited successfully and its complete capture index is durable on NVMe.

capture_container="qsrt-glm52-layers52-60-63-64-input-capture-artifact-layer-compatible"
capture_index="/home/sunil/qsrt-glm52-experiments/captures/glm52-layers52-60-63-64-wikitext-document-disjoint-routed-inputs-artifact-layer-compatible/manifest.json"
encoder_script="/home/sunil/qsrt-glm52-experiments/launch-scripts/run_glm52_layers60_64_52_63_uniform_k3_and_merge_on_kossel.sh"

while test "$(docker inspect "${capture_container}" --format '{{.State.Running}}')" = "true"; do
  sleep 15
done
test "$(docker inspect "${capture_container}" --format '{{.State.ExitCode}}')" = "0"
test -f "${capture_index}"
exec bash "${encoder_script}"

#!/usr/bin/env bash
set -euo pipefail

# Copy only the validated rate-preserving down-refit implementation and its
# documentation to kossel's indexed source tree. The transfer never deletes a
# remote file and verifies every destination byte against the local source.

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote_host="kossel.lan"
remote_root="/home/sunil/qsrt-glm52-experiments/source/qsrt-working-tree"
files=(
  docs/glm52-experiment-journal.md
  docs/glm52-layer3-kld-results.md
  experiments/glm52_layer3_rate_preserving_down_refit_k3_k4_pre_registration.json
  experiments/merge_and_materialize_glm52_down_refit_rate_pool_on_kossel.sh
  experiments/run_glm52_candidate_kld_chunked_full_vocabulary_on_kossel.sh
  experiments/run_glm52_down_refit_rate_pool_slices_on_kossel.sh
  experiments/sync_glm52_rate_preserving_down_refit_source_to_kossel.sh
  qsrt/glm52_down_refit_rate_pool.py
  qsrt/glm52_k3_k4_allocation.py
  scripts/build_glm52_down_refit_rate_pool.py
  scripts/materialize_glm52_down_refit_rate_pool_allocation.py
  scripts/merge_glm52_down_refit_rate_pool.py
  tests/test_glm52_down_refit_rate_pool.py
  tests/test_glm52_k3_k4_allocation.py
)

cd "${repository_root}"
for path in "${files[@]}"; do
  test -f "${path}"
done
ssh -o BatchMode=yes "${remote_host}" "test -d '${remote_root}'"
rsync -a --relative "${files[@]}" "${remote_host}:${remote_root}/"
sha256sum "${files[@]}" | ssh -o BatchMode=yes "${remote_host}" \
  "cd '${remote_root}' && sha256sum --check --strict"
ssh -o BatchMode=yes "${remote_host}" \
  "cd '${remote_root}' && bash -n \
    experiments/merge_and_materialize_glm52_down_refit_rate_pool_on_kossel.sh \
    experiments/run_glm52_candidate_kld_chunked_full_vocabulary_on_kossel.sh \
    experiments/run_glm52_down_refit_rate_pool_slices_on_kossel.sh \
    experiments/sync_glm52_rate_preserving_down_refit_source_to_kossel.sh"

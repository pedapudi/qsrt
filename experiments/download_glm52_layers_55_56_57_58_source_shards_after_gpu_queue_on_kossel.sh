#!/usr/bin/env bash
set -euo pipefail

# Download only the official BF16 shards that contain routed-expert tensors
# for GLM-5.2 layers 55 through 58. The supplied predecessor keeps model loads
# and the NVMe writes from competing for the same device.

if test "$#" != 1; then
  echo "usage: $0 <predecessor-process-id>" >&2
  exit 2
fi
predecessor_pid="$1"
if [[ ! "${predecessor_pid}" =~ ^[1-9][0-9]*$ ]]; then
  echo "predecessor process ID must be a positive integer" >&2
  exit 2
fi
while kill -0 "${predecessor_pid}" 2>/dev/null; do
  sleep 30
done

experiment_root="/home/sunil/qsrt-glm52-experiments"
metadata_root="${experiment_root}/metadata/glm52-source-and-reference-inventory-20260818"
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manifest="${script_root}/glm52_layers_55_56_57_58_source_shards.json"
destination="${experiment_root}/source-windows/glm52-b4734de-layers-55-56-57-58"
revision="b4734de4facf877f85769a911abafc5283eab3d9"
expected_shards=16
expected_bytes=85783011360
parallel_downloads=16
lock_file="${experiment_root}/locks/glm52-layers-55-56-57-58-source-download.lock"

test -f "${manifest}"
test "$(jq -r '.source_model' "${manifest}")" = "zai-org/GLM-5.2"
test "$(jq -r '.source_revision' "${manifest}")" = "${revision}"
jq -e '.layers == [55, 56, 57, 58]' "${manifest}" >/dev/null
test "$(jq '.shards | length' "${manifest}")" -eq "${expected_shards}"
test "$(jq '.total_bytes' "${manifest}")" -eq "${expected_bytes}"

mkdir -p "${destination}" "$(dirname "${lock_file}")"
exec 9>"${lock_file}"
flock -n 9

available_bytes="$(df -B1 --output=avail "${destination}" | tail -n 1 | tr -d ' ')"
required_bytes=$((expected_bytes + 32 * 1024 * 1024 * 1024))
if (( available_bytes < required_bytes )); then
  printf 'insufficient space: available=%s required=%s\n' \
    "${available_bytes}" "${required_bytes}" >&2
  exit 1
fi

cp "${metadata_root}/source-b4734de-config.json" "${destination}/config.json"
cp "${metadata_root}/source-b4734de-model.safetensors.index.json" \
  "${destination}/model.safetensors.index.json"
cp "${manifest}" "${destination}/download-manifest.json"

download_one() {
  local filename="$1"
  local size="$2"
  local sha256="$3"
  local final_path="${destination}/${filename}"
  local partial_path="${final_path}.partial"

  if test -f "${final_path}"; then
    test "$(stat -c %s "${final_path}")" -eq "${size}"
    printf '%s  %s\n' "${sha256}" "${final_path}" | sha256sum --check --strict
    return
  fi

  printf 'downloading %s\n' "${filename}"
  curl \
    --fail \
    --location \
    --retry 20 \
    --retry-all-errors \
    --silent \
    --show-error \
    --continue-at - \
    --output "${partial_path}" \
    "https://huggingface.co/zai-org/GLM-5.2/resolve/${revision}/${filename}"

  test "$(stat -c %s "${partial_path}")" -eq "${size}"
  printf '%s  %s\n' "${sha256}" "${partial_path}" | sha256sum --check --strict
  mv "${partial_path}" "${final_path}"
  printf 'completed %s\n' "${filename}"
}

batch_pids=()
while IFS=$'\t' read -r filename size sha256; do
  download_one "${filename}" "${size}" "${sha256}" &
  batch_pids+=("$!")
  if test "${#batch_pids[@]}" -eq "${parallel_downloads}"; then
    batch_failed=0
    for pid in "${batch_pids[@]}"; do
      if ! wait "${pid}"; then
        batch_failed=1
      fi
    done
    test "${batch_failed}" -eq 0
    batch_pids=()
  fi
done < <(jq -r '.shards[] | [.file, (.size | tostring), .sha256] | @tsv' "${manifest}")

batch_failed=0
for pid in "${batch_pids[@]}"; do
  if ! wait "${pid}"; then
    batch_failed=1
  fi
done
test "${batch_failed}" -eq 0

actual_bytes=0
while IFS=$'\t' read -r filename size sha256; do
  path="${destination}/${filename}"
  test -f "${path}"
  test "$(stat -c %s "${path}")" -eq "${size}"
  printf '%s  %s\n' "${sha256}" "${path}" | sha256sum --check --strict
  actual_bytes=$((actual_bytes + size))
done < <(jq -r '.shards[] | [.file, (.size | tostring), .sha256] | @tsv' "${manifest}")
test "${actual_bytes}" -eq "${expected_bytes}"

jq -n \
  --arg revision "${revision}" \
  --arg manifest_sha256 "$(sha256sum "${manifest}" | cut -d ' ' -f 1)" \
  --argjson shard_count "${expected_shards}" \
  --argjson total_bytes "${actual_bytes}" \
  '{
    schema: "qsrt_glm52_bounded_source_download_receipt",
    source_model: "zai-org/GLM-5.2",
    source_revision: $revision,
    layers: [55, 56, 57, 58],
    shard_count: $shard_count,
    total_bytes: $total_bytes,
    download_manifest_sha256: $manifest_sha256,
    complete: true
  }' > "${destination}/receipt.json"

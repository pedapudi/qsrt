#!/usr/bin/env bash
set -euo pipefail

# Download only the public BF16-reference chunks frozen in the auxiliary plan.
# The destination must reside on kossel's internal NVMe. Existing checkpoint
# files below /home/sunil/usb-mnt are outside this operation.

root="${1:-/home/sunil/qsrt-glm52-experiments/reference/glm52-unsloth-document-disjoint-auxiliary-v1}"
plan="${root}/selection-plan.json"
destination="${root}/reference-logprobs"
receipt="${root}/receipt.json"

test -f "${plan}"
mount_source="$(findmnt -n -o SOURCE -T "${root}")"
if test "${mount_source}" = "/dev/sda" || [[ "${root}" == /home/sunil/usb-mnt/* ]]; then
  echo "refusing to place new reference files on the external disk" >&2
  exit 1
fi

repository="$(jq -r .reference_repository "${plan}")"
revision="$(jq -r .reference_revision "${plan}")"
test -n "${repository}"
test "${#revision}" -eq 40
mkdir -p "${destination}"

download_one() {
  local record="$1"
  local filename expected_sha256 expected_bytes final partial actual_sha256
  IFS=$'\t' read -r filename expected_sha256 expected_bytes <<< "${record}"
  final="${destination}/${filename}"
  partial="${final}.partial"
  if test -f "${final}"; then
    test "$(stat -c %s "${final}")" = "${expected_bytes}"
    test "$(sha256sum "${final}" | cut -d ' ' -f 1)" = "${expected_sha256}"
    return
  fi
  curl \
    --fail \
    --location \
    --retry 8 \
    --retry-all-errors \
    --continue-at - \
    --output "${partial}" \
    "https://huggingface.co/datasets/${repository}/resolve/${revision}/reference-logprobs/${filename}"
  test "$(stat -c %s "${partial}")" = "${expected_bytes}"
  actual_sha256="$(sha256sum "${partial}" | cut -d ' ' -f 1)"
  test "${actual_sha256}" = "${expected_sha256}"
  mv "${partial}" "${final}"
}
export -f download_one
export destination repository revision

jq -r '.selected_chunks[] | [.reference_file, .reference_file_sha256, .reference_file_bytes] | @tsv' "${plan}" \
  | xargs -d '\n' -P 4 -I '{}' bash -c 'download_one "$1"' _ '{}'

python3 - "${plan}" "${destination}" "${receipt}" "${mount_source}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
directory = Path(sys.argv[2])
receipt_path = Path(sys.argv[3])
plan = json.loads(plan_path.read_text())
files = []
for row in plan["selected_chunks"]:
    path = directory / row["reference_file"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    size = path.stat().st_size
    if digest != row["reference_file_sha256"] or size != row["reference_file_bytes"]:
        raise SystemExit(f"reference validation failed: {path}")
    files.append({"path": path.name, "bytes": size, "sha256": digest})
receipt = {
    "schema": "qsrt_glm52_public_reference_auxiliary_download",
    "schema_version": 1,
    "status": "complete",
    "selection_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    "reference_repository": plan["reference_repository"],
    "reference_revision": plan["reference_revision"],
    "mount_source": sys.argv[4],
    "file_count": len(files),
    "total_bytes": sum(row["bytes"] for row in files),
    "files": files,
    "model_weights_downloaded": False,
}
temporary = receipt_path.with_suffix(".json.partial")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
os.replace(temporary, receipt_path)
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

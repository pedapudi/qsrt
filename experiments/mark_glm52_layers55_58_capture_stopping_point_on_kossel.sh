#!/usr/bin/env bash
set -euo pipefail

# Wait for the bounded four-layer input capture to finish, verify every capture
# manifest, and write a durable shutdown marker.  This script starts no GPU
# job.  Candidate construction resumes only through a separate explicit launch.

if test "$#" -ne 1 || [[ ! "$1" =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 <input-capture-launcher-process-id>" >&2
  exit 2
fi
capture_launcher_pid="$1"
experiment_root="/home/sunil/qsrt-glm52-experiments"
capture_container="qsrt-glm52-layers55-56-57-58-input-capture"
capture_root="${experiment_root}/captures/glm52-layers55-56-57-58-wikitext-document-disjoint-routed-inputs"
source_root="${experiment_root}/source-windows/glm52-b4734de-layers-55-56-57-58"
marker="${experiment_root}/launch-records/glm52-layers55-58-capture-stopping-point.json"
test ! -e "${marker}"

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

python3 - "${capture_root}" "${source_root}" "${marker}" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

capture_root = Path(sys.argv[1])
source_root = Path(sys.argv[2])
marker = Path(sys.argv[3])

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

capture_manifest_path = capture_root / "manifest.json"
capture_manifest = json.loads(capture_manifest_path.read_text())
if capture_manifest.get("status") != "complete":
    raise ValueError("four-layer capture manifest is incomplete")
layer_manifests = []
for layer in (55, 56, 57, 58):
    path = capture_root / f"layer-{layer:03d}" / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "qsrt_glm52_layer_input_capture_manifest":
        raise ValueError(f"layer {layer} capture schema mismatch")
    if manifest.get("status") != "complete" or manifest.get("model_layer") != layer:
        raise ValueError(f"layer {layer} capture is incomplete")
    if manifest.get("collections") != {"activation_fit": 32, "candidate_selection": 8}:
        raise ValueError(f"layer {layer} capture collection counts differ")
    layer_manifests.append(
        {
            "model_layer": layer,
            "path": str(path),
            "sha256": digest(path),
        }
    )
receipt_path = source_root / "receipt.json"
receipt = json.loads(receipt_path.read_text())
if (
    receipt.get("complete") is not True
    or receipt.get("layers") != [55, 56, 57, 58]
    or receipt.get("shard_count") != 16
    or receipt.get("total_bytes") != 85_783_011_360
):
    raise ValueError("bounded source receipt is incomplete")

gpu_processes = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if gpu_processes:
    raise RuntimeError("a GPU process remains active after capture completion")
record = {
    "schema": "qsrt_glm52_layers55_58_capture_stopping_point",
    "schema_version": 1,
    "status": "safe_for_host_shutdown",
    "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    "bounded_source": {
        "receipt_path": str(receipt_path),
        "receipt_sha256": digest(receipt_path),
        "layers": [55, 56, 57, 58],
        "shard_count": 16,
        "total_bytes": 85_783_011_360,
    },
    "capture": {
        "manifest_path": str(capture_manifest_path),
        "manifest_sha256": digest(capture_manifest_path),
        "layer_manifests": layer_manifests,
    },
    "gpu_compute_processes": [],
    "automatic_candidate_queue_active": False,
    "resume_commands": [
        (
            "for layer in 55 56 57 58; do bash "
            "/home/sunil/qsrt-glm52-experiments/launch-scripts/"
            "build_and_screen_glm52_hot_layer_down_recovery_on_kossel.sh "
            "${layer}; done"
        ),
        (
            "bash /home/sunil/qsrt-glm52-experiments/launch-scripts/"
            "continue_glm52_hot_layer_retained_arms_to_single_reference_on_kossel.sh "
            "<candidate-construction-process-id>"
        ),
    ],
}
marker.parent.mkdir(parents=True, exist_ok=True)
temporary = marker.with_name(f".{marker.name}.partial-{os.getpid()}")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
os.replace(temporary, marker)
print(json.dumps(record, indent=2, sort_keys=True))
PY

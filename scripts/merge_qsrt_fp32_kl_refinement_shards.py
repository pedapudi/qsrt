#!/usr/bin/env python
"""Merge disjoint expert payload overlays for one QSRT refinement layer."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from qsrt import constants as C


_EXPERT = re.compile(r"\.experts\.(\d+)\.")


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layer_directory", type=Path)
    args = parser.parse_args()

    root = args.layer_directory.expanduser().resolve()
    final_completion = root / "completion.json"
    final_payload = root / "payload-overlay.safetensors"
    if final_completion.exists() or final_payload.exists():
        raise ValueError("final layer overlay already exists")

    records = []
    for path in sorted((root / "shards").glob("experts-*/completion.json")):
        record = json.loads(path.read_text())
        if record.get("complete") is not True:
            raise ValueError(f"incomplete shard: {path}")
        records.append((path, record))
    if not records:
        raise ValueError("no completed expert shards found")

    expected_begin = 0
    tensors: dict[str, torch.Tensor] = {}
    reference = records[0][1]
    worker_seconds = []
    for completion_path, record in records:
        begin = int(record["expert_begin"])
        end = int(record["expert_end"])
        if begin != expected_begin or not begin < end:
            raise ValueError("expert shards are not a contiguous partition")
        expected_begin = end
        for key in (
            "layer",
            "alpha",
            "refinement_objective",
            "gradient_scales",
            "anchor_residual_sum_squares",
            "gradient_sum_squares",
        ):
            if record[key] != reference[key]:
                raise ValueError(f"shard metadata differs for {key}")
        payload = Path(record["payload_overlay"])
        with safe_open(str(payload), framework="pt", device="cpu") as reader:
            for name in reader.keys():
                match = _EXPERT.search(name)
                if match is None or not begin <= int(match.group(1)) < end:
                    raise ValueError(f"tensor lies outside shard range: {name}")
                if name in tensors:
                    raise ValueError(f"duplicate tensor in expert shards: {name}")
                tensors[name] = reader.get_tensor(name)
        worker_seconds.append(float(record["elapsed_seconds"]))
    if expected_begin != C.NUM_EXPERTS:
        raise ValueError("expert shards do not cover the complete layer")
    expected_tensors = C.NUM_EXPERTS * len(C.EXPERT_MATRICES) * 3
    if len(tensors) != expected_tensors:
        raise ValueError(
            f"merged overlay has {len(tensors)} tensors, expected {expected_tensors}"
        )

    temporary = final_payload.with_name(f".{final_payload.name}.{os.getpid()}.tmp")
    save_file(
        tensors,
        str(temporary),
        metadata={
            "kind": "qsrt_fp32_final_kl_gradient_refinement_overlay",
            "layer": str(reference["layer"]),
            "alpha": str(reference["alpha"]),
            "refinement_objective": str(reference["refinement_objective"]),
        },
    )
    os.replace(temporary, final_payload)
    result = {
        key: reference[key]
        for key in (
            "kind",
            "schema_version",
            "layer",
            "alpha",
            "refinement_objective",
            "gradient_scales",
            "anchor_residual_sum_squares",
            "gradient_sum_squares",
        )
    }
    result.update(
        {
            "complete": True,
            "expert_begin": 0,
            "expert_end": C.NUM_EXPERTS,
            "experts": C.NUM_EXPERTS,
            "payload_overlay": str(final_payload),
            "payload_bytes": final_payload.stat().st_size,
            "worker_count": len(records),
            "worker_elapsed_seconds_max": max(worker_seconds),
            "worker_elapsed_seconds_sum": sum(worker_seconds),
        }
    )
    _atomic_json(final_completion, result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

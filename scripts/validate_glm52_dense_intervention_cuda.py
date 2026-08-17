#!/usr/bin/env python3
"""Validate full versus tensor-parallel dense expert intervention arithmetic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from qsrt.glm52_expert_intervention_runtime import (
    DenseEndpointStore,
    evaluate_expert,
    validate_dense_intervention_artifact,
)
from qsrt.glm52_pilot import atomic_write_json


def _relative_mse(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    error = (reference.float() - candidate.float()).double()
    denominator = reference.float().double().square().sum().clamp_min(1e-30)
    return float(error.square().sum().div(denominator).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=8)
    args = parser.parse_args()
    if args.dest.exists():
        raise FileExistsError(args.dest)
    artifact = validate_dense_intervention_artifact(args.artifact)
    if args.expert not in artifact["expert_ids"]:
        raise ValueError("requested expert is absent from the artifact")
    device = torch.device(args.device)
    torch.manual_seed(20_260_816)
    x = torch.randn((args.rows, 6144), dtype=torch.bfloat16, device=device)
    record = next(
        value
        for value in artifact["report"]["experts"]
        if int(value["expert"]) == args.expert
    )
    path = args.artifact / "experts" / record["dense_endpoint_file"]
    full: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for endpoint in ("exl3", "qsrt_k3"):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                key = f"{endpoint}.{projection}"
                full[key] = handle.get_tensor(key).to(device).contiguous()

    full_outputs = {
        endpoint: evaluate_expert(
            x,
            gate=full[f"{endpoint}.gate_proj"],
            up=full[f"{endpoint}.up_proj"],
            down=full[f"{endpoint}.down_proj"],
        ).float()
        for endpoint in ("exl3", "qsrt_k3")
    }
    sliced_outputs = {
        "exl3": torch.zeros_like(full_outputs["exl3"]),
        "qsrt_k3": torch.zeros_like(full_outputs["qsrt_k3"]),
    }
    for rank in range(4):
        store = DenseEndpointStore(
            args.artifact,
            device=device,
            tensor_parallel_rank=rank,
            expected_manifest_sha256=artifact["manifest_sha256"],
        )
        item = store.expert_slice(args.expert)
        sliced_outputs["exl3"].add_(
            evaluate_expert(
                x, gate=item.exl3_gate, up=item.exl3_up, down=item.exl3_down
            ).float()
        )
        sliced_outputs["qsrt_k3"].add_(
            evaluate_expert(
                x,
                gate=item.candidate_gate,
                up=item.candidate_up,
                down=item.candidate_down,
            ).float()
        )
    relative_mse = {
        endpoint: _relative_mse(full_outputs[endpoint], sliced_outputs[endpoint])
        for endpoint in ("exl3", "qsrt_k3")
    }
    delta_reference = full_outputs["qsrt_k3"] - full_outputs["exl3"]
    delta_sliced = sliced_outputs["qsrt_k3"] - sliced_outputs["exl3"]
    report = {
        "schema": "qsrt_glm52_dense_intervention_cuda_closure",
        "schema_version": 1,
        "status": "passed",
        "artifact_manifest_sha256": artifact["manifest_sha256"],
        "expert": args.expert,
        "rows": args.rows,
        "device": str(device),
        "full_vs_four_slice_relative_mse": relative_mse,
        "delta_full_vs_four_slice_relative_mse": _relative_mse(
            delta_reference, delta_sliced
        ),
        "maximum_allowed_relative_mse": 1e-5,
    }
    if max(
        *relative_mse.values(), report["delta_full_vs_four_slice_relative_mse"]
    ) > report["maximum_allowed_relative_mse"]:
        report["status"] = "failed"
        atomic_write_json(args.dest, report)
        raise RuntimeError("dense intervention tensor-parallel arithmetic failed closure")
    atomic_write_json(args.dest, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

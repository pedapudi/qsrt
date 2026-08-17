from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from safetensors.torch import save_file


def _save_trace(
    root: Path,
    *,
    layer: int,
    call: int,
    stage: str,
    tensor: torch.Tensor,
) -> None:
    tensor = tensor.contiguous()
    digest = hashlib.sha256(
        tensor.view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    payload = {
        "metadata": {
            "tp_rank": 0,
            "layer": layer,
            "saved_shape": list(tensor.shape),
            "sha256": digest,
        },
        "tensor": tensor,
    }
    rank = root / "tp-rank-000"
    rank.mkdir(parents=True, exist_ok=True)
    torch.save(
        payload,
        rank / f"layer-{layer:03d}.call-{call:06d}.{stage}.pt",
    )


def test_capture_witness_matches_epochs_by_content(tmp_path: Path) -> None:
    capture = tmp_path / "witness.kqcapture"
    cache = tmp_path / "witness.kqsamples"
    trace = tmp_path / "trace"
    capture.mkdir()
    cache.mkdir()
    (capture / "manifest.json").write_text(
        json.dumps(
            {
                "sampling": {
                    "input_hessian": 1,
                    "validation_modulus": 16,
                }
            }
        )
    )
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "source_capture": str(capture.resolve()),
                "layers": {"1": {"file": "layer-00001.safetensors"}},
            }
        )
    )

    calls = {
        3: {
            "routed_latent_input": torch.arange(20, dtype=torch.bfloat16).reshape(4, 5),
            "canonical_topk_ids": torch.tensor(
                [[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int32
            ),
            "canonical_topk_weights": torch.tensor(
                [[0.2, 0.3], [0.4, 0.1], [0.6, 0.2], [0.7, 0.1]],
                dtype=torch.float32,
            ),
            "routed_latent_reduced": torch.arange(
                20, 40, dtype=torch.bfloat16
            ).reshape(4, 5),
        },
        5: {
            "routed_latent_input": torch.arange(40, 60, dtype=torch.bfloat16).reshape(4, 5),
            "canonical_topk_ids": torch.tensor(
                [[9, 10], [11, 12], [13, 14], [15, 16]], dtype=torch.int32
            ),
            "canonical_topk_weights": torch.tensor(
                [[0.15, 0.25], [0.35, 0.45], [0.55, 0.05], [0.65, 0.15]],
                dtype=torch.float32,
            ),
            "routed_latent_reduced": torch.arange(
                60, 80, dtype=torch.bfloat16
            ).reshape(4, 5),
        },
    }
    for call, stages in calls.items():
        for stage, tensor in stages.items():
            _save_trace(trace, layer=1, call=call, stage=stage, tensor=tensor)

    # Cache order is observation order, not trace-call order.  Select rows 1
    # and 3 from each call so content matching and token indexing are tested.
    rows = [(7, 1, 3), (7, 3, 3), (11, 1, 5), (11, 3, 5)]
    observations = torch.tensor(
        [(epoch << 32) | token for epoch, token, _ in rows], dtype=torch.int64
    )
    values = torch.stack(
        [calls[call]["routed_latent_input"][token] for _, token, call in rows]
    )
    experts = torch.stack(
        [calls[call]["canonical_topk_ids"][token] for _, token, call in rows]
    )
    gates = torch.stack(
        [calls[call]["canonical_topk_weights"][token] for _, token, call in rows]
    )
    routed = torch.stack(
        [calls[call]["routed_latent_reduced"][token] for _, token, call in rows]
    )
    split = torch.zeros(len(rows), dtype=torch.int8)
    from qsrt.blockldlq_proof import capture_validation_split

    split[:] = torch.tensor(
        [capture_validation_split(int(value), 16) for value in observations],
        dtype=torch.int8,
    )
    save_file(
        {
            "input.values": values,
            "input.observation": observations,
            "input.experts": experts,
            "input.gates": gates,
            "input.weight": gates.square().sum(dim=1),
            "input.split": split,
            "input.routed_latent": routed,
        },
        str(cache / "layer-00001.safetensors"),
    )

    output = tmp_path / "receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_qsrt_capture_witness.py",
            "--capture",
            str(capture),
            "--sample-cache",
            str(cache),
            "--trace-dir",
            str(trace),
            "--layers",
            "1",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    receipt = json.loads(output.read_text())
    assert receipt["passed"] is True
    assert receipt["layers"][0]["trace_layer"] == 1
    assert {row["trace_call"] for row in receipt["layers"][0]["epochs"]} == {3, 5}

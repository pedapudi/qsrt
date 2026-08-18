#!/usr/bin/env python3
"""Compare one stored uniform-K2 expert with same-draw fresh encodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open
import torch

from qsrt import constants as C
from qsrt.exl3_loader import load_qsrt_encoder
from qsrt.pack.qsrt_candidates import candidate_tensor_name
from qsrt.pack.qsrt_encoder import (
    QSRTMatrixCandidate,
    finalize_qsrt_matrix_candidate,
    plan_qsrt_matrix,
    qsrt_transform_seed_draw,
)
from qsrt.pack.qsrt_validation import decode_candidate_matrix
from qsrt.qsrt import K2, matrix_rate_axis
from qsrt.qsrt_coupled import CoupledHadamardSpec, encode_coupled_weights
from qsrt.source_weights import OfficialMXFP4Store
from qsrt.sqg_quantizer import install_sqg_quantizer
from qsrt.two_sided_qsrt import encode_uniform_sqg_direct_batch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--draw", type=int, required=True)
    parser.add_argument("--upstream-overlay", type=Path, required=True)
    parser.add_argument("--down-overlay", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exllamav3-root",
        type=Path,
        default=Path("/home/luke/projects/exllamav3"),
    )
    return parser


def _tensor_comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, object]:
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        return {
            "identical": False,
            "left_dtype": str(left.dtype),
            "right_dtype": str(right.dtype),
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    mismatch = left != right
    result: dict[str, object] = {
        "identical": not bool(torch.any(mismatch)),
        "elements": left.numel(),
        "different_elements": int(torch.count_nonzero(mismatch)),
    }
    if left.is_floating_point():
        delta = left.float() - right.float()
        result["max_abs_delta"] = float(delta.abs().max())
        result["sse_delta"] = float(delta.double().square().sum())
    elif left.dtype == torch.int16:
        xor = torch.bitwise_xor(left.to(torch.int32), right.to(torch.int32)) & 0xFFFF
        byte_view = xor.contiguous().cpu().numpy().astype("<u2", copy=False).view("u1")
        result["different_bits"] = int(
            sum(int(value).bit_count() for value in byte_view.tolist())
        )
    return result


def _distortion(target: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, float]:
    error = reconstruction.float() - target.float()
    sse = float(error.double().square().sum())
    energy = float(target.float().double().square().sum())
    return {
        "sse": sse,
        "source_energy": energy,
        "relative_sse": sse / energy,
        "relative_l2": (sse / energy) ** 0.5,
        "max_abs": float(error.abs().max()),
    }


def main() -> None:
    args = _parser().parse_args()
    if args.layer not in C.MOE_LAYERS:
        raise ValueError("--layer must identify a routed decoder layer")
    if not 0 <= args.expert < C.NUM_EXPERTS:
        raise ValueError("--expert must lie in 0..895")
    if not 0 <= args.draw < 8:
        raise ValueError("--draw must lie in 0..7")
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("--device must be an indexed CUDA device")

    store = OfficialMXFP4Store()
    with store.open_layer(
        args.layer,
        experts=(args.expert,),
        matrices=C.EXPERT_MATRICES,
    ) as layer_store:
        source_triplet = tuple(
            layer_store.load_matrix(
                args.layer, args.expert, matrix, device=device
            )
            for matrix in C.EXPERT_MATRICES
        )
    transformed = encode_coupled_weights(
        source_triplet,
        CoupledHadamardSpec(intermediate_draw=args.draw),
    )

    backend = load_qsrt_encoder(args.exllamav3_root)
    install_sqg_quantizer(backend)
    plans = {
        matrix: plan_qsrt_matrix(
            torch.zeros(768, dtype=torch.long, device=device),
            K2,
            matrix=matrix,
            layout="importance_ordered",
        )
        for matrix in C.EXPERT_MATRICES
    }
    overlay_paths = {"w1": args.upstream_overlay, "w3": args.upstream_overlay, "w2": args.down_overlay}
    report: dict[str, object] = {
        "kind": "qsrt_uniform_k2_fresh_payload_identity",
        "layer": args.layer,
        "expert": args.expert,
        "intermediate_draw": args.draw,
        "source_checkpoint": str(store.root),
        "fresh_encoder_revision": os.popen("git rev-parse HEAD").read().strip(),
        "stored_encoder_revision": None,
        "stored_encoder_revision_available": False,
        "matrices": {},
    }

    for matrix, source in zip(C.EXPERT_MATRICES, transformed, strict=True):
        seeds = qsrt_transform_seed_draw(args.layer, matrix)
        fresh_encodings = []
        for repeat in range(2):
            direct = encode_uniform_sqg_direct_batch(
                source.unsqueeze(0),
                bits=2,
                device=device,
                quantizer_module=backend,
                input_sign_seed=seeds.input_sign,
                output_sign_seed=seeds.output_sign,
                rate_axis=matrix_rate_axis(matrix),
                scale_scope_key=(
                    ("fresh-payload-identity", repeat, args.layer, matrix)
                    if matrix in ("w1", "w3")
                    else None
                ),
                shared_scale_axis="input" if matrix in ("w1", "w3") else None,
                tailbite_context=128,
            )[0]
            fresh_encodings.append(
                finalize_qsrt_matrix_candidate(
                    QSRTMatrixCandidate(
                        reconstruction=direct.candidate.reconstruction,
                        encoded=direct.candidate.states,
                        tensors={"suh": direct.suh, "svh": direct.svh},
                        plan=plans[matrix],
                        proxy=0.0,
                        transform_seeds=seeds,
                        global_scale=direct.global_scale,
                    ),
                    layer=args.layer,
                    tailbite_context=128,
                )
            )

        with safe_open(overlay_paths[matrix], framework="pt", device="cpu") as reader:
            stored_tensors = {
                part: reader.get_tensor(
                    candidate_tensor_name(args.layer, args.expert, matrix, part)
                )
                for part in ("trellis", "suh", "svh")
            }
            stored_decode = decode_candidate_matrix(
                reader,
                layer=args.layer,
                expert=args.expert,
                matrix=matrix,
                mode_id=K2.mode_id,
                device=device,
            )

        target_physical = source.T.contiguous()
        fresh_physical = fresh_encodings[0].reconstruction.T.contiguous()
        matrix_report = {
            "stored_vs_fresh": {
                part: _tensor_comparison(
                    stored_tensors[part],
                    fresh_encodings[0].tensors[part].detach().cpu(),
                )
                for part in ("trellis", "suh", "svh")
            },
            "fresh_repeat": {
                part: _tensor_comparison(
                    fresh_encodings[0].tensors[part].detach().cpu(),
                    fresh_encodings[1].tensors[part].detach().cpu(),
                )
                for part in ("trellis", "suh", "svh")
            },
            "stored_distortion": _distortion(target_physical, stored_decode),
            "fresh_distortion": _distortion(target_physical, fresh_physical),
            "stored_vs_fresh_decode": _distortion(stored_decode, fresh_physical),
            "fresh_global_scale": fresh_encodings[0].coding.get("global_scale", fresh_encodings[0].coding.get("proxy")),
        }
        matrix_report["fresh_minus_stored_sse"] = (
            matrix_report["fresh_distortion"]["sse"]
            - matrix_report["stored_distortion"]["sse"]
        )
        report["matrices"][matrix] = matrix_report

    all_stored_equal = all(
        all(matrix["stored_vs_fresh"][part]["identical"] for part in ("trellis", "suh", "svh"))
        for matrix in report["matrices"].values()
    )
    all_fresh_repeat_equal = all(
        all(matrix["fresh_repeat"][part]["identical"] for part in ("trellis", "suh", "svh"))
        for matrix in report["matrices"].values()
    )
    scales_equal = all(
        all(matrix["stored_vs_fresh"][part]["identical"] for part in ("suh", "svh"))
        for matrix in report["matrices"].values()
    )
    if all_stored_equal:
        attribution = "bit-identical; no encoder-provenance difference observed"
    elif not scales_equal:
        attribution = "scale construction differs; payload comparison does not isolate Viterbi tie-breaking"
    elif all_fresh_repeat_equal:
        attribution = "scales match and fresh repeats are deterministic; difference is deterministic encoder-version or tie-breaking provenance"
    else:
        attribution = "fresh encoder is nondeterministic under an identical target and draw"
    report["summary"] = {
        "stored_payload_bit_identical_to_fresh": all_stored_equal,
        "fresh_repeat_bit_identical": all_fresh_repeat_equal,
        "stored_scales_bit_identical_to_fresh": scales_equal,
        "attribution": attribution,
        "fresh_minus_stored_sse": sum(
            float(matrix["fresh_minus_stored_sse"])
            for matrix in report["matrices"].values()
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

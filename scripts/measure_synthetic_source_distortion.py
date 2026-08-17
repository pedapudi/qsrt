#!/usr/bin/env python3
"""Measure per-symbol synthetic-source distortion for L16 trellis codes.

Runs the exact CPU Viterbi from ``qsrt.synthetic_source_distortion`` over an
independent standard-normal source and reports steady-state per-symbol MSE
for each requested reconstruction code, with a fitted global scale per code.
Use it to compare QSRT's SQG labels against ExLlamaV3's native MCG and MUL1
codebooks at one trellis rate, and to reproduce the endpoint and rank-map
ablations.

Reference points at the default settings (256 sequences, 65,536 measured
symbols): the Gaussian rate-distortion bound is 0.0625 at K2 and 0.015625 at
K3; scalar Lloyd-Max at two bits is 0.1175; the production ``sqg_t12_e4m3``
labels measure about 2.6% (K2) and 3.0% (K3) lower MSE than ``exl3_mcg``.

Example:

    .venv/bin/python scripts/measure_synthetic_source_distortion.py \\
        --bits 3 --codes exl3_mcg exl3_mul1 sqg_t12_e4m3 \\
        --out /tmp/k3_distortion.json

The default eight-code, 256-sequence run takes tens of minutes on a
12-thread CPU; pass ``--sequences 32`` for a quick, noisier pass.  Output is
JSON only; benchmark output is not a source artifact and must not be
committed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from qsrt import synthetic_source_distortion as ssd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bits", type=int, required=True, choices=(2, 3, 4))
    parser.add_argument(
        "--codes",
        nargs="+",
        default=list(ssd.CODE_NAMES),
        choices=list(ssd.CODE_NAMES),
        help="reconstruction codes to measure (default: all)",
    )
    parser.add_argument("--sequences", type=int, default=256)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument(
        "--window",
        type=int,
        nargs=2,
        default=(128, 384),
        metavar=("START", "STOP"),
        help="interior scored span; keep both edges >=128 steps away",
    )
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--menu-statistics",
        action="store_true",
        help="also report per-menu value and stratum diversity",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    results: dict[str, dict] = {}
    for name in args.codes:
        start = time.time()
        entry = ssd.measure_code(
            name,
            args.bits,
            sequences=args.sequences,
            steps=args.steps,
            window=tuple(args.window),
            seed=args.seed,
        ).as_dict()
        entry["seconds"] = round(time.time() - start, 1)
        if args.menu_statistics:
            entry["menu_statistics"] = ssd.menu_statistics(name, args.bits)
        results[name] = entry
        edge = "  SCALE AT GRID EDGE" if entry["scale_at_grid_edge"] else ""
        print(
            f"K{args.bits} {name:24s} scale={entry['fitted_scale']:.5f} "
            f"MSE={entry['mse']:.6f} +- {entry['stderr']:.6f}"
            f"{edge} [{entry['seconds']:.0f}s]",
            flush=True,
        )
        args.out.write_text(json.dumps(results, indent=2))

    if "exl3_mcg" in results:
        reference = results["exl3_mcg"]["mse"]
        for name, entry in results.items():
            if name == "exl3_mcg":
                continue
            delta = (entry["mse"] / reference - 1.0) * 100.0
            print(f"  {name:24s} vs exl3_mcg: {delta:+.2f}% MSE")


if __name__ == "__main__":
    main()

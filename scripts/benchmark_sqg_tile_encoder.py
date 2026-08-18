"""Benchmark and quality-compare two builds of the SQG tile trellis quantizer.

The offline QSRT encoder quantizes 256-coefficient tiles with the tail-biting
trellis CUDA kernel in ``qsrt/csrc``. This harness compiles that extension
twice, from the working tree and from a pinned reference source directory,
runs both on identical tile batches, and reports:

- kernel wall time per batch (CUDA events, alternating A/B iterations); and
- reconstruction quality: exact FP64 per-tile squared error of the returned
  reconstruction against the input, compared tile-by-tile between builds.

Every output is validated structurally regardless of timing: encoded states
must reconstruct a closed trellis path, and the returned reconstruction must
equal the codebook decode of the returned states.

The kernel is a search heuristic evaluated in FP16, so the two builds may
legally select different paths. The quality gate for an encoder change is
therefore distributional: mean squared error must not regress, and per-tile
regressions must be balanced by at least equivalent improvements.

Example:

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python \\
        scripts/benchmark_sqg_tile_encoder.py \\
        --reference-src /path/to/pinned/csrc \\
        --bits 2,3,4 --contexts 128 --families normal,scaled,heavy,spike \\
        --json out/sqg-tile-encoder-benchmark.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import qsrt  # noqa: E402

assert Path(qsrt.__file__).resolve().parents[1] == REPO_ROOT, (
    "qsrt resolved outside this checkout: " + str(qsrt.__file__)
)

from qsrt.sqg_e4m3 import SQG_XOR_CHEB_T12, sqg_codebook_bytes  # noqa: E402


def build_extension(name: str, source_dir: Path):
    from torch.utils.cpp_extension import load

    return load(
        name=name,
        sources=[
            str(source_dir / "sqg_quantize.cpp"),
            str(source_dir / "sqg_quantize.cu"),
        ],
        extra_include_paths=[str(source_dir)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-Xcudafe",
            "--diag_suppress=177",
            "-Xcudafe",
            "--diag_suppress=20012",
        ],
        verbose=False,
    )


def transpose_lut_predecessor_major(lut: torch.Tensor, bits: int) -> torch.Tensor:
    """Reorder state-indexed labels to the kernel's per-edge-pair layout.

    Mirrors the layout produced in :mod:`qsrt.sqg_quantizer`: labels move from
    ``[predecessor, out_edge_pair, pair_byte]`` order to
    ``[out_edge_pair, predecessor, pair_byte]`` so one vector load fetches all
    predecessor labels of an aligned out-edge pair.
    """

    if bits not in (2, 3, 4):
        return lut.contiguous()
    predecessors = 1 << bits
    out_edge_pairs = (65536 >> bits) // 2
    return (
        lut.reshape(predecessors, out_edge_pairs, 2)
        .permute(1, 0, 2)
        .contiguous()
        .reshape(-1)
    )


def make_tiles(family: str, tiles: int, seed: int, device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn((tiles, 256), generator=generator, dtype=torch.float32)
    if family == "normal":
        pass
    elif family == "scaled":
        scales = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0])
        values *= scales.repeat((tiles + 4) // 5)[:tiles, None]
    elif family == "heavy":
        draws = (
            torch.randn((tiles, 256, 3), generator=generator).square().sum(dim=-1)
        )
        values *= torch.rsqrt(draws / 3.0).clamp(max=1e3)
    elif family == "spike":
        mask = torch.rand((tiles, 256), generator=generator) < 0.01
        values = torch.where(mask, values * 100.0, values)
    else:
        raise ValueError(f"unknown tile family {family!r}")
    return values.to(device)


def check_closure(indices: torch.Tensor, bits: int) -> None:
    edges = indices.to(torch.int64) & ((1 << bits) - 1)
    expected = torch.zeros_like(edges)
    for lag in range(math.ceil(16 / bits)):
        expected |= torch.roll(edges, shifts=lag, dims=-1) << (lag * bits)
    expected = (expected & 0xFFFF).to(torch.int16)
    mismatch = int((indices != expected).sum())
    if mismatch:
        raise RuntimeError(f"K{bits}: {mismatch} non-closing trellis states")


def check_decode(
    indices: torch.Tensor, output: torch.Tensor, lut: torch.Tensor, bits: int
) -> None:
    values = (
        lut.view(torch.float8_e4m3fn).to(torch.float32).to(indices.device)
    )
    decoded = values[indices.to(torch.int64) & 0xFFFF]
    mismatch = int((decoded != output).sum())
    if mismatch:
        raise RuntimeError(
            f"K{bits}: {mismatch} reconstructions differ from state decode"
        )


def tile_sse(tiles: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    delta = output.double() - tiles.double()
    return (delta * delta).sum(dim=1).cpu()


class Encoder:
    def __init__(self, label: str, ext, device: torch.device):
        self.label = label
        self.ext = ext
        self.device = device
        self._buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._luts: dict[int, torch.Tensor] = {}

    def buffers(self, bits: int, tiles: int):
        cached = self._buffers.get(bits)
        if cached is not None and cached[0].shape[0] >= tiles:
            return cached
        edges = 65536 >> bits
        decisions = edges // 4 if bits == 2 else edges // 2 if bits <= 4 else edges
        costs = torch.zeros(
            (tiles, 2, edges), dtype=torch.float16, device=self.device
        )
        traceback = torch.empty(
            (tiles, 256, decisions), dtype=torch.uint8, device=self.device
        )
        self._buffers[bits] = (costs, traceback)
        return costs, traceback

    def lut(self, bits: int) -> tuple[torch.Tensor, float]:
        cached = self._luts.get(bits)
        if cached is None:
            base = sqg_codebook_bytes(bits, SQG_XOR_CHEB_T12)
            abs_max = float(
                base.view(torch.float8_e4m3fn).to(torch.float32).abs().amax()
            )
            table = transpose_lut_predecessor_major(base, bits).to(self.device)
            cached = (table, abs_max)
            self._luts[bits] = cached
        return cached

    def encode(self, tiles: torch.Tensor, bits: int, context: int):
        costs, traceback = self.buffers(bits, tiles.shape[0])
        output = torch.empty_like(tiles)
        indices = torch.empty_like(tiles, dtype=torch.int16)
        table, abs_max = self.lut(bits)
        try:
            self.ext.quantize_tiles_sqg(
                tiles, output, indices, costs, traceback, table, bits, context,
                abs_max,
            )
        except TypeError:
            self.ext.quantize_tiles_sqg(
                tiles, output, indices, costs, traceback, table, bits, context
            )
        return output, indices


def timed_encode(encoder: Encoder, tiles, bits, context) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    encoder.encode(tiles, bits, context)
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference-src", type=Path, default=None)
    parser.add_argument("--candidate-src", type=Path, default=REPO_ROOT / "qsrt/csrc")
    parser.add_argument("--tiles", type=int, default=512)
    parser.add_argument("--bits", type=str, default="2,3,4")
    parser.add_argument("--contexts", type=str, default="128")
    parser.add_argument(
        "--families", type=str, default="normal,scaled,heavy,spike"
    )
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    bits_list = [int(b) for b in args.bits.split(",")]
    contexts = [int(c) for c in args.contexts.split(",")]
    families = args.families.split(",")

    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"candidate src: {args.candidate_src}")
    t0 = time.monotonic()
    candidate = Encoder(
        "candidate", build_extension("qsrt_sqg_bench_candidate", args.candidate_src), device
    )
    print(f"candidate build: {time.monotonic() - t0:.1f}s")
    encoders = [candidate]
    if args.reference_src is not None:
        print(f"reference src: {args.reference_src}")
        t0 = time.monotonic()
        encoders.insert(
            0,
            Encoder(
                "reference",
                build_extension("qsrt_sqg_bench_reference", args.reference_src),
                device,
            ),
        )
        print(f"reference build: {time.monotonic() - t0:.1f}s")

    report = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "tiles": args.tiles,
        "configs": [],
    }

    for bits in bits_list:
        base_lut = sqg_codebook_bytes(bits, SQG_XOR_CHEB_T12)
        for context in contexts:
            for family in families:
                tiles = make_tiles(family, args.tiles, args.seed, device)
                entry = {
                    "bits": bits,
                    "context": context,
                    "family": family,
                    "encoders": {},
                }
                results = {}
                for encoder in encoders:
                    output, indices = encoder.encode(tiles, bits, context)
                    check_closure(indices, bits)
                    check_decode(indices, output, base_lut, bits)
                    results[encoder.label] = tile_sse(tiles, output)
                times = {encoder.label: [] for encoder in encoders}
                for _ in range(args.warmup):
                    for encoder in encoders:
                        timed_encode(encoder, tiles, bits, context)
                for _ in range(args.iters):
                    for encoder in encoders:
                        times[encoder.label].append(
                            timed_encode(encoder, tiles, bits, context)
                        )
                for encoder in encoders:
                    sse = results[encoder.label]
                    entry["encoders"][encoder.label] = {
                        "median_ms": statistics.median(times[encoder.label]),
                        "min_ms": min(times[encoder.label]),
                        "times_ms": times[encoder.label],
                        "mean_tile_sse": float(sse.mean()),
                    }
                line = (
                    f"K{bits} C{context} {family:>7}: "
                    + "  ".join(
                        f"{label} {vals['median_ms']:7.3f}ms"
                        f" sse {vals['mean_tile_sse']:.6f}"
                        for label, vals in entry["encoders"].items()
                    )
                )
                if len(encoders) == 2:
                    ref = results["reference"]
                    new = results["candidate"]
                    delta = new - ref
                    entry["sse_delta"] = {
                        "mean": float(delta.mean()),
                        "mean_relative": float(delta.mean() / ref.mean()),
                        "tiles_better": int((delta < 0).sum()),
                        "tiles_worse": int((delta > 0).sum()),
                        "worst_tile_delta": float(delta.max()),
                        "best_tile_delta": float(delta.min()),
                    }
                    speedup = (
                        entry["encoders"]["reference"]["median_ms"]
                        / entry["encoders"]["candidate"]["median_ms"]
                    )
                    line += (
                        f"  Δsse {entry['sse_delta']['mean_relative']:+.4%}"
                        f" (worse {entry['sse_delta']['tiles_worse']}"
                        f"/better {entry['sse_delta']['tiles_better']})"
                        f"  speedup {speedup:.3f}x"
                    )
                print(line)
                report["configs"].append(entry)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()

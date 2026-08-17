"""Research-only bridge from EXL's production encoder to SQG E4M3 tables."""

from __future__ import annotations

import hashlib
import math
import os
from functools import lru_cache
from pathlib import Path
from collections.abc import Mapping

import torch
from torch.utils.cpp_extension import load

from qsrt.sqg_e4m3 import (
    sqg_e4m3_bytes,
    sqg_xor_cheb_t12_rank_lut_bytes,
    sqg_xor_rank_permutation,
)


_DIAGNOSTIC_HISTOGRAM_BINS = 96
_DIAGNOSTIC_HISTOGRAM_MIN = -6.0
_DIAGNOSTIC_HISTOGRAM_MAX = 6.0
_DIAGNOSTIC_TAIL_THRESHOLDS = (3.0, 4.0, 5.0)


def _empty_distribution_accumulator(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "moments": torch.zeros(4, dtype=torch.float64, device=device),
        "histogram": torch.zeros(
            _DIAGNOSTIC_HISTOGRAM_BINS, dtype=torch.int64, device=device
        ),
        "outside_histogram": torch.zeros(2, dtype=torch.int64, device=device),
        "tail_count": torch.zeros(
            len(_DIAGNOSTIC_TAIL_THRESHOLDS), dtype=torch.int64, device=device
        ),
        "tail_energy": torch.zeros(
            len(_DIAGNOSTIC_TAIL_THRESHOLDS), dtype=torch.float64, device=device
        ),
        "adjacent": torch.zeros(6, dtype=torch.float64, device=device),
        "position_sum": torch.zeros(256, dtype=torch.float64, device=device),
        "position_square_sum": torch.zeros(
            256, dtype=torch.float64, device=device
        ),
    }


def _accumulate_distribution(
    accumulator: dict[str, torch.Tensor], values: torch.Tensor
) -> None:
    values64 = values.to(torch.float64)
    flat = values64.reshape(-1)
    for order in range(1, 5):
        accumulator["moments"][order - 1] += flat.pow(order).sum()
    accumulator["histogram"] += torch.histc(
        values.float(),
        bins=_DIAGNOSTIC_HISTOGRAM_BINS,
        min=_DIAGNOSTIC_HISTOGRAM_MIN,
        max=_DIAGNOSTIC_HISTOGRAM_MAX,
    ).to(torch.int64)
    accumulator["outside_histogram"][0] += torch.count_nonzero(
        values < _DIAGNOSTIC_HISTOGRAM_MIN
    )
    accumulator["outside_histogram"][1] += torch.count_nonzero(
        values > _DIAGNOSTIC_HISTOGRAM_MAX
    )
    square = values64.square()
    absolute = values64.abs()
    for index, threshold in enumerate(_DIAGNOSTIC_TAIL_THRESHOLDS):
        selected = absolute > threshold
        accumulator["tail_count"][index] += torch.count_nonzero(selected)
        accumulator["tail_energy"][index] += square[selected].sum()
    left = values64[:, :-1]
    right = values64[:, 1:]
    accumulator["adjacent"] += torch.stack(
        (
            left.sum(),
            right.sum(),
            left.square().sum(),
            right.square().sum(),
            (left * right).sum(),
            torch.tensor(left.numel(), dtype=torch.float64, device=values.device),
        )
    )
    accumulator["position_sum"] += values64.sum(dim=0)
    accumulator["position_square_sum"] += values64.square().sum(dim=0)


def _accumulate_trellis_diagnostics(
    tiles: torch.Tensor,
    reconstruction: torch.Tensor,
    indices: torch.Tensor,
    bits: int,
    quant_args: dict,
) -> None:
    """Accumulate exact post-feedback Viterbi-input diagnostics on the GPU."""

    if not quant_args.get("return_trellis_diagnostics") or not quant_args.get(
        "trellis_diagnostics_ldlq_active"
    ):
        return
    accumulator = quant_args.get("trellis_diagnostics_accumulator")
    if accumulator is None:
        accumulator = {
            "bits": bits,
            "tile_count": 0,
            "target": _empty_distribution_accumulator(tiles.device),
            "residual": _empty_distribution_accumulator(tiles.device),
            "table_occupancy": torch.zeros(
                4096, dtype=torch.int64, device=tiles.device
            ),
            "table_target_sum": torch.zeros(
                4096, dtype=torch.float64, device=tiles.device
            ),
            "table_target_square_sum": torch.zeros(
                4096, dtype=torch.float64, device=tiles.device
            ),
        }
        quant_args["trellis_diagnostics_accumulator"] = accumulator
    elif accumulator["bits"] != bits:
        raise ValueError("one trellis diagnostic accumulator cannot mix rates")
    accumulator["tile_count"] += int(tiles.shape[0])
    residual = tiles - reconstruction
    _accumulate_distribution(accumulator["target"], tiles)
    _accumulate_distribution(accumulator["residual"], residual)
    states = indices.to(torch.int64) & 0xFFFF
    ranks = sqg_xor_rank_permutation(bits).to(indices.device)
    table_indices = ranks.index_select(0, states.reshape(-1)) >> 4
    accumulator["table_occupancy"] += torch.bincount(
        table_indices, minlength=4096
    )
    target_values = tiles.reshape(-1).to(torch.float64)
    accumulator["table_target_sum"] += torch.bincount(
        table_indices, weights=target_values, minlength=4096
    )
    accumulator["table_target_square_sum"] += torch.bincount(
        table_indices, weights=target_values.square(), minlength=4096
    )


def _fixed_path_table_oracle(accumulator: Mapping) -> dict:
    """Measure table-update headroom while holding selected paths fixed."""

    occupancy = accumulator["table_occupancy"].cpu()
    target_sum = accumulator["table_target_sum"].cpu()
    target_square_sum = accumulator["table_target_square_sum"].cpu()
    used = occupancy > 0
    centroids = torch.zeros(4096, dtype=torch.float64)
    centroids[used] = target_sum[used] / occupancy[used]

    unconstrained_sse = torch.clamp(
        target_square_sum[used]
        - target_sum[used].square() / occupancy[used],
        min=0.0,
    ).sum()
    learned_bytes = sqg_xor_cheb_t12_rank_lut_bytes().clone()
    rounded_centroids = (
        centroids[used]
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
    )
    learned_bytes[used] = rounded_centroids.view(torch.uint8)
    finite_centroids = learned_bytes.view(torch.float8_e4m3fn).to(torch.float64)
    finite_e4m3_sse = torch.clamp(
        target_square_sum[used]
        - 2.0 * finite_centroids[used] * target_sum[used]
        + occupancy[used] * finite_centroids[used].square(),
        min=0.0,
    ).sum()
    selected_path_sse = accumulator["residual"]["moments"][1].cpu()
    selected_path_sse_value = float(selected_path_sse)

    def relative_reduction(candidate_sse: torch.Tensor) -> float:
        if selected_path_sse_value <= 0.0:
            return 0.0
        return 1.0 - float(candidate_sse) / selected_path_sse_value

    learned_raw = bytes(learned_bytes.tolist())
    return {
        "selected_path_frozen_table_sse": selected_path_sse_value,
        "selected_path_unconstrained_centroid_sse": float(unconstrained_sse),
        "selected_path_unconstrained_relative_reduction": relative_reduction(
            unconstrained_sse
        ),
        "selected_path_finite_e4m3_centroid_sse": float(finite_e4m3_sse),
        "selected_path_finite_e4m3_relative_reduction": relative_reduction(
            finite_e4m3_sse
        ),
        "finite_e4m3_table_sha256": hashlib.sha256(learned_raw).hexdigest(),
        "finite_e4m3_table_bytes": learned_bytes.tolist(),
        "evidence_boundary": (
            "fit-set normalized-domain oracle with selected trellis paths held "
            "fixed; it does not measure path reassignment, a disjoint document "
            "set, curvature-weighted loss, or full-model KLD"
        ),
    }


def _finalize_distribution(
    accumulator: Mapping[str, torch.Tensor], *, count: int
) -> dict:
    moments = accumulator["moments"].cpu().tolist()
    mean = moments[0] / count
    second = moments[1] / count
    third = moments[2] / count
    fourth = moments[3] / count
    variance = max(0.0, second - mean * mean)
    if variance > 0.0:
        skewness = (third - 3.0 * mean * second + 2.0 * mean**3) / variance**1.5
        excess_kurtosis = (
            fourth
            - 4.0 * mean * third
            + 6.0 * mean * mean * second
            - 3.0 * mean**4
        ) / variance**2 - 3.0
    else:
        skewness = 0.0
        excess_kurtosis = 0.0
    adjacent = accumulator["adjacent"].cpu().tolist()
    pair_count = int(adjacent[5])
    left_mean = adjacent[0] / pair_count
    right_mean = adjacent[1] / pair_count
    left_variance = max(0.0, adjacent[2] / pair_count - left_mean**2)
    right_variance = max(0.0, adjacent[3] / pair_count - right_mean**2)
    denominator = math.sqrt(left_variance * right_variance)
    adjacent_correlation = (
        (adjacent[4] / pair_count - left_mean * right_mean) / denominator
        if denominator > 0.0
        else 0.0
    )
    total_energy = moments[1]
    tail_count = accumulator["tail_count"].cpu().tolist()
    tail_energy = accumulator["tail_energy"].cpu().tolist()
    position_sum = accumulator["position_sum"].cpu()
    position_square_sum = accumulator["position_square_sum"].cpu()
    rows = count // 256
    position_mean = position_sum / rows
    position_variance = torch.clamp(
        position_square_sum / rows - position_mean.square(), min=0.0
    )
    return {
        "count": count,
        "mean": mean,
        "variance": variance,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "adjacent_correlation_within_path": adjacent_correlation,
        "histogram": {
            "bins": _DIAGNOSTIC_HISTOGRAM_BINS,
            "minimum": _DIAGNOSTIC_HISTOGRAM_MIN,
            "maximum": _DIAGNOSTIC_HISTOGRAM_MAX,
            "counts": accumulator["histogram"].cpu().tolist(),
            "below_minimum": int(accumulator["outside_histogram"][0].cpu()),
            "above_maximum": int(accumulator["outside_histogram"][1].cpu()),
        },
        "tails": {
            str(threshold): {
                "count_fraction": int(tail_count[index]) / count,
                "energy_fraction": (
                    float(tail_energy[index]) / total_energy
                    if total_energy > 0.0
                    else 0.0
                ),
            }
            for index, threshold in enumerate(_DIAGNOSTIC_TAIL_THRESHOLDS)
        },
        "position_mean": position_mean.tolist(),
        "position_variance": position_variance.tolist(),
    }


def finalize_trellis_diagnostics(accumulator: Mapping) -> dict:
    """Move one optional encoder diagnostic accumulator into JSON-safe data."""

    tile_count = int(accumulator["tile_count"])
    count = tile_count * 256
    if tile_count <= 0:
        raise ValueError("trellis diagnostics contain no tiles")
    occupancy = accumulator["table_occupancy"].cpu()
    probability = occupancy.double() / count
    nonzero = probability > 0
    entropy = float(-(probability[nonzero] * probability[nonzero].log2()).sum())
    return {
        "bits": int(accumulator["bits"]),
        "tile_count": tile_count,
        "target": _finalize_distribution(accumulator["target"], count=count),
        "residual": _finalize_distribution(accumulator["residual"], count=count),
        "table": {
            "entry_count": 4096,
            "used_entries": int(torch.count_nonzero(occupancy)),
            "maximum_entry_fraction": float(occupancy.max().double() / count),
            "occupancy_entropy_bits": entropy,
            "occupancy": occupancy.tolist(),
        },
        "fixed_path_table_oracle": _fixed_path_table_oracle(accumulator),
        "measurement_domain": (
            "exact normalized 256-value tile presented to Viterbi after "
            "BlockLDLQ feedback; residual is tile minus selected reconstruction"
        ),
    }


def _debug_check_trellis_closure(indices: torch.Tensor, bits: int) -> None:
    """Fail at the CUDA boundary if a tile contains an inconsistent state.

    This is intentionally opt-in because it adds a device synchronization to
    every tile-quantizer call.  It distinguishes an invalid state written by
    the traceback kernel from corruption introduced later while mixed-rate
    candidates are assembled and scored.
    """

    edges = indices.to(torch.int64) & ((1 << bits) - 1)
    expected = torch.zeros_like(edges)
    for lag in range(math.ceil(16 / bits)):
        expected |= torch.roll(edges, shifts=lag, dims=-1) << (lag * bits)
    expected = (expected & 0xFFFF).to(torch.int16)
    mismatch = indices != expected
    if bool(torch.any(mismatch)):
        count = int(torch.count_nonzero(mismatch))
        first = tuple(
            int(value)
            for value in torch.nonzero(mismatch, as_tuple=False)[0].cpu().tolist()
        )
        encoded = int(indices[first].to(torch.int32).cpu())
        reconstructed = int(expected[first].to(torch.int32).cpu())
        raise RuntimeError(
            "SQG tile kernel emitted a non-closing trellis state: "
            f"K{bits}, {count} states differ; first at {first}, "
            f"encoded={encoded}, reconstructed={reconstructed}"
        )


@lru_cache(maxsize=1)
def _extension():
    project = Path(__file__).resolve().parents[1]
    return load(
        name="qsrt_sqg_quantize_ext_v25",
        sources=[
            str(project / "qsrt/csrc/sqg_quantize.cpp"),
            str(project / "qsrt/csrc/sqg_quantize.cu"),
        ],
        extra_include_paths=[str(project / "qsrt/csrc")],
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


@lru_cache(maxsize=None)
def _sqg_temp_buffers(device: torch.device, bits: int):
    """Allocate the packed traceback buffers consumed by QSRT's kernel."""

    multiprocessors = torch.cuda.get_device_properties(device).multi_processor_count
    edges = 65536 >> bits
    decisions = edges // 4 if bits == 2 else edges // 2 if bits <= 4 else edges
    free_bytes, _ = torch.cuda.mem_get_info(device)
    decision_bytes_per_tile = 256 * decisions
    affordable = max(256, int(free_bytes * 0.5) // decision_bytes_per_tile)
    max_batch = min(max(256, 3 * multiprocessors), affordable)
    costs = torch.zeros(
        (max_batch, 2, edges), dtype=torch.float16, device=device
    )
    traceback = torch.empty(
        (max_batch, 256, decisions), dtype=torch.uint8, device=device
    )
    return costs, traceback


def install_sqg_quantizer(quantizer_module) -> None:
    """Teach a loaded EXL encoder module to consume ``sqg_e4m3_lut``.

    The patch is process-local. SQG and explicit MCG/MUL1 controls use
    QSRT's CUDA extension with one Viterbi/tail-biting implementation. A
    ``None`` entry in a rate-specific mapping explicitly selects MCG,
    permitting controlled hybrid rate-curve studies. Calls without any
    QSRT codebook argument retain the unmodified upstream EXL behavior.
    """

    if getattr(quantizer_module, "_qsrt_sqg_installed", False):
        return
    original = quantizer_module.quantize_tiles

    device_luts: dict[tuple[str, int, str], torch.Tensor] = {}
    transposed_sqg_luts: dict[
        tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]
    ] = {}

    def quantize_tiles(tiles: torch.Tensor, quant_args: dict):
        codebook = quant_args.get("sqg_e4m3_lut")
        fp16_codebook = quant_args.get("sqg_fp16_lut")
        rate_codebooks = quant_args.get("sqg_e4m3_luts_by_bits")
        mode = quant_args.get("sqg_e4m3_mode")
        if (
            codebook is None
            and fp16_codebook is None
            and rate_codebooks is None
            and mode is None
        ):
            return original(tiles, quant_args)
        if fp16_codebook is not None and (
            codebook is not None or rate_codebooks is not None or mode is not None
        ):
            raise ValueError("an FP16 SQG table cannot be combined with another codebook")
        if len(quant_args["devices"]) != 1:
            raise ValueError("the SQG validation hook currently requires one CUDA device")
        tiles = tiles.contiguous()
        if tiles.dtype != torch.float32 or tiles.ndim != 2 or tiles.shape[1] != 256:
            raise ValueError("SQG tiles must be contiguous FP32 [N, 256]")
        bits = int(quant_args["K"])

        def finish(output: torch.Tensor, indices: torch.Tensor):
            _accumulate_trellis_diagnostics(
                tiles, output, indices, bits, quant_args
            )
            return output, indices

        if fp16_codebook is not None:
            output = torch.empty_like(tiles)
            indices = torch.empty_like(tiles, dtype=torch.int16)
            costs, edges = _sqg_temp_buffers(tiles.device, bits)
            lut = fp16_codebook.to(
                device=tiles.device, dtype=torch.float16
            ).contiguous()
            if lut.ndim != 1 or lut.numel() != 65536 or not bool(
                torch.isfinite(lut).all()
            ):
                raise ValueError(
                    "an experimental FP16 SQG table must contain 65,536 finite values"
                )
            _extension().quantize_tiles_fp16(
                tiles,
                output,
                indices,
                costs,
                edges,
                lut,
                bits,
                int(quant_args.get("tailbite_context", 128)),
            )
            if os.environ.get("QSRT_DEBUG_TILE_CLOSURE") == "1":
                _debug_check_trellis_closure(indices, bits)
            return finish(output, indices)
        if rate_codebooks is not None:
            if codebook is not None or mode is not None:
                raise ValueError(
                    "rate-specific SQG LUTs cannot be combined with another SQG law"
                )
            if not isinstance(rate_codebooks, Mapping):
                raise TypeError("sqg_e4m3_luts_by_bits must be a mapping")
            if set(rate_codebooks) - {2, 3, 4}:
                raise ValueError("rate-specific SQG LUT keys must be K2, K3, or K4")
            try:
                codebook = rate_codebooks[bits]
            except KeyError as exc:
                raise ValueError(f"missing rate-specific SQG K{bits} LUT") from exc
            if codebook is None:
                output = torch.empty_like(tiles)
                indices = torch.empty_like(tiles, dtype=torch.int16)
                costs, edges = _sqg_temp_buffers(tiles.device, bits)
                _extension().quantize_tiles_procedural(
                    tiles,
                    output,
                    indices,
                    costs,
                    edges,
                    bits,
                    1,
                    int(quant_args.get("tailbite_context", 128)),
                )
                return finish(output, indices)
        elif codebook is None:
            if mode != "normal":
                raise ValueError("the supported R44 mode is 'normal'")
            key = (str(tiles.device), bits, mode)
            codebook = device_luts.get(key)
            if codebook is None:
                codebook = sqg_e4m3_bytes(bits, mode, device=tiles.device)
                device_luts[key] = codebook
        output = torch.empty_like(tiles)
        indices = torch.empty_like(tiles, dtype=torch.int16)
        costs, edges = _sqg_temp_buffers(tiles.device, bits)
        lut = codebook.to(device=tiles.device, dtype=torch.uint8).contiguous()
        if bits in (2, 3, 4):
            source_key = codebook.data_ptr() if codebook.is_cuda else id(codebook)
            cache_key = (str(tiles.device), bits, source_key)
            cached = transposed_sqg_luts.get(cache_key)
            if cached is None or cached[0] is not codebook:
                # [predecessor, out-edge-pair, pair-byte] ->
                # [out-edge-pair, predecessor, pair-byte].  Each CUDA thread
                # can then fetch all predecessor labels in one uint2 (K2),
                # one uint4 (K3), or two uint4s (K4), rather than issuing one
                # gather per predecessor.
                predecessors = 1 << bits
                out_edge_pairs = (65536 >> bits) // 2
                lut = (
                    lut.reshape(predecessors, out_edge_pairs, 2)
                    .permute(1, 0, 2)
                    .contiguous()
                    .reshape(-1)
                )
                transposed_sqg_luts[cache_key] = (codebook, lut)
            else:
                lut = cached[1]
        _extension().quantize_tiles_sqg(
            tiles,
            output,
            indices,
            costs,
            edges,
            lut,
            bits,
            int(quant_args.get("tailbite_context", 128)),
        )
        if os.environ.get("QSRT_DEBUG_TILE_CLOSURE") == "1":
            _debug_check_trellis_closure(indices, bits)
        return finish(output, indices)

    quantizer_module.quantize_tiles = quantize_tiles
    quantizer_module._qsrt_sqg_installed = True

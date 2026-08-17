"""Per-symbol synthetic-source distortion for L16 trellis reconstruction codes.

This module measures the steady-state mean squared error that a 16-bit
edge-window (L16) trellis code achieves on an independent standard-normal
source, using an exact full Viterbi search on CPU.  It exists to compare
QSRT's Stratified Quantile Graph (SQG) labels against ExLlamaV3's native
computed codebooks (MCG and MUL1) at the same trellis rate, per symbol,
before any Hessian weighting, error feedback, or model context.

The source model matches the marginal a production tile presents to the
trellis: after the encoder's blockwise Hadamard transforms and per-channel
normalization, tile coefficients are close to independent N(0, 1).  Each code
receives its own fitted global scale, mirroring the production encoder, which
fits one scalar per matrix by golden-section search (``g_scale_gss`` in
``qsrt/exl3_encoder_backend.py``) after dividing by per-channel RMS and a
fixed codebook constant.  Fitting the single scale on the scored sequences is
therefore not a leak; it reproduces the production contract and removes the
selection noise a separate sweep set would add.

Trellis convention, shared with ``qsrt/csrc/qsrt_quantize_tiles_kernel.cuh``
and ``qsrt.exl3_reference.reconstruct_trellis_states``: the 16-bit codeword at
one step is ``(previous_window << K) | edge`` masked to 16 bits, so a
codeword's low K bits are its newest edge and its low ``16 - K`` bits are the
persistent Viterbi state.  The search here keeps all ``2**(16 - K)`` states
and evaluates all ``2**K`` in-edges per state at every step with float32 path
costs; the production CUDA kernel uses float16 costs, so this search is if
anything slightly stronger than production, equally for every code.

Reported distortion is windowed: sequences are longer than the scored span
and only interior symbols are scored, so neither the free initial state nor
the free final state (each worth about 16 unpaid bits) inflates the figure.
In this shift-register trellis every symbol string closes a tail-biting tile,
so the windowed figure is the right per-symbol distortion for production's
cyclic 256-symbol tiles.

These are coding measurements on a synthetic source.  They rank
reconstruction labels on the shared graph; they do not predict held-out model
quality, which additionally depends on the Hessian-weighted objective,
BlockLDLQ feedback, and allocation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

N_CODEWORDS = 1 << 16

#: Optimal four-level scalar quantizer for N(0, 1) (Lloyd-Max).  Its nearest-
#: neighbour MSE is 0.1175, used to validate the measurement harness.
LLOYD_MAX_2BIT_LEVELS = (-1.5104, -0.4528, 0.4528, 1.5104)
LLOYD_MAX_2BIT_MSE = 0.1175

#: Transcription anchors.  The encoder hard-codes codebook 0's standard
#: deviation (``codebook_scale`` in ``qsrt/exl3_encoder_backend.py``); the
#: MCG distinct-value count is a stable fingerprint of that construction.
EXL3_CB0_SIGMA = 1.24371088
EXL3_MCG_DISTINCT_VALUES = 10746


def exl3_mcg_values() -> np.ndarray:
    """ExLlamaV3 codebook 1 ("mcg"): multiply, mask/xor halves, add.

    Transcribed from ``decode_3inst<1>`` in
    ``qsrt/csrc/exl3_compat/quant/codebook.cuh``.  The ``lop3.b32`` with
    immediate LUT 0x6a computes ``(a & b) ^ c``.
    """

    c = np.arange(N_CODEWORDS, dtype=np.uint32)
    x = c * np.uint32(0xCBAC1FED)
    x = (x & np.uint32(0x8FFF8FFF)) ^ np.uint32(0x3B603B60)
    lo = (x & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
    hi = (x >> np.uint32(16)).astype(np.uint16).view(np.float16)
    return (lo + hi).astype(np.float64)


def exl3_mul1_values() -> np.ndarray:
    """ExLlamaV3 codebook 2 ("mul1"): multiply, byte-sum, affine map in half.

    Transcribed from ``decode_3inst<2>`` in
    ``qsrt/csrc/exl3_compat/quant/codebook.cuh``, including the ``__dp4a``
    byte sum and the 0x1EEE / 0xC931 half-precision affine constants.
    """

    c = np.arange(N_CODEWORDS, dtype=np.uint32)
    x = c * np.uint32(0x83DCD12D)
    byte_sum = np.zeros(N_CODEWORDS, dtype=np.uint32)
    for shift in (0, 8, 16, 24):
        byte_sum += (x >> np.uint32(shift)) & np.uint32(0xFF)
    bits = (byte_sum + np.uint32(0x6400)).astype(np.uint16)
    h = bits.view(np.float16).astype(np.float32)
    k_inv = np.uint16(0x1EEE).reshape(1).view(np.float16)[0].astype(np.float32)
    k_bias = np.uint16(0xC931).reshape(1).view(np.float16)[0].astype(np.float32)
    return (h * k_inv + k_bias).astype(np.float16).astype(np.float64)


def exl3_cb0_values() -> np.ndarray:
    """ExLlamaV3 codebook 0: multiply-add, then the two-half construction."""

    c = np.arange(N_CODEWORDS, dtype=np.uint32)
    x = c * np.uint32(89226354) + np.uint32(64248484)
    x = (x & np.uint32(0x8FFF8FFF)) ^ np.uint32(0x3B603B60)
    lo = (x & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
    hi = (x >> np.uint32(16)).astype(np.uint16).view(np.float16)
    return (lo + hi).astype(np.float64)


def _reverse_low_bits(values: np.ndarray, bits: int) -> np.ndarray:
    out = np.zeros_like(values)
    for i in range(bits):
        out |= ((values >> i) & 1) << (bits - 1 - i)
    return out


def menu_oriented_rank(bits: int) -> np.ndarray:
    """SQG's rank construction re-expressed along the Viterbi menu axis.

    The frozen construction derives its coarse stratum from a codeword's low
    K branch bits and its phase from the high ``16 - K`` history bits.  Along
    a real Viterbi menu (the ``2**K`` codewords that can arrive at one
    outgoing state) the low ``16 - K`` bits are pinned and the top K bits are
    the free choice, so the frozen graph exposes only about 2.04 distinct
    strata per menu at K2 rather than the nominal 4.  This control keeps the
    identical mixer, multiplier, bit reversal, and XOR but feeds them the
    menu-free coordinate ``codeword >> (16 - K)`` and the pinned coordinate
    ``codeword & (2**(16 - K) - 1)``.  It is still a bijection over all
    65,536 (stratum, phase) coordinates and exposes all ``2**K`` strata in
    every menu by construction.

    Measured on this harness, the reoriented graph is worse than the frozen
    one (about 3.5% at K2 and 5.4% at K3) despite far better memoryless
    menus: sequence-space coverage beats per-step menu coverage.  The control
    is retained so that conclusion stays reproducible.
    """

    width = 16 - bits
    mask = np.uint64((1 << width) - 1)
    c = np.arange(N_CODEWORDS, dtype=np.uint64)
    free = c >> np.uint64(width)
    pinned = c & mask
    mixed = pinned ^ (pinned >> np.uint64(11))
    mixed = mixed ^ ((mixed << np.uint64(11)) & mask)
    product = (np.uint64(0x3FA7D929) * mixed + np.uint64(0xC928FD8E)) & np.uint64(
        0xFFFFFFFF
    )
    phase = product & mask
    syndrome = product >> np.uint64(32 - bits)
    stratum = _reverse_low_bits(free, bits) ^ syndrome
    return ((stratum << np.uint64(width)) | phase).astype(np.int64)


def _exact_normal_rank_law() -> torch.Tensor:
    ranks = torch.arange(N_CODEWORDS, dtype=torch.float64)
    return 1.5 * torch.special.ndtri((ranks + 0.5) / N_CODEWORDS)


def sqg_values(bits: int, variant: str) -> np.ndarray:
    """SQG reconstruction table for one trellis rate and one label variant.

    ``t12_e4m3`` is the production table from ``qsrt.sqg_e4m3``.  The
    ``exact_*`` variants replace the modal T12 staircase with the exact
    65,536-rank normal law under three endpoint alphabets, isolating what the
    T12 reduction and the E4M3 endpoint each cost.  The ``menu_oriented_*``
    variants swap in :func:`menu_oriented_rank`.
    """

    from qsrt import sqg_e4m3 as sqg

    if variant == "t12_e4m3":
        raw = sqg.sqg_xor_cheb_t12_bytes(bits)
        return raw.view(torch.float8_e4m3fn).float().numpy().astype(np.float64)
    if variant.startswith("menu_oriented_"):
        ranks = menu_oriented_rank(bits)
        endpoint = variant[len("menu_oriented_") :]
    elif variant.startswith("exact_"):
        ranks = sqg.sqg_xor_rank_permutation(bits).numpy()
        endpoint = variant[len("exact_") :]
    else:
        raise ValueError(f"unknown SQG variant {variant!r}")
    law = _exact_normal_rank_law()
    if endpoint == "fp16":
        table = law.numpy().astype(np.float16).astype(np.float64)
    elif endpoint == "e4m3":
        table = law.float().to(torch.float8_e4m3fn).float().numpy().astype(np.float64)
    elif endpoint == "fp32":
        table = law.numpy()
    else:
        raise ValueError(f"unknown SQG endpoint {endpoint!r}")
    return table[ranks]


CODE_NAMES = (
    "exl3_mcg",
    "exl3_mul1",
    "exl3_cb0",
    "sqg_t12_e4m3",
    "sqg_exact_e4m3",
    "sqg_exact_fp16",
    "sqg_exact_fp32",
    "sqg_menu_oriented_e4m3",
    "sqg_menu_oriented_fp16",
)


def reconstruction_table(name: str, bits: int) -> np.ndarray:
    """Return the 65,536-entry reconstruction table for a named code."""

    if name == "exl3_mcg":
        return exl3_mcg_values()
    if name == "exl3_mul1":
        return exl3_mul1_values()
    if name == "exl3_cb0":
        return exl3_cb0_values()
    if name.startswith("sqg_"):
        return sqg_values(bits, name[len("sqg_") :])
    raise ValueError(f"unknown code {name!r}; known codes: {CODE_NAMES}")


def predecessor_index(bits: int) -> torch.Tensor:
    """Map each 16-bit codeword to the state index its path arrived from.

    States are the low ``16 - K`` bits of the codeword.  Enumerating a
    codeword as ``(free << (16 - K)) | state`` makes ``free`` the K
    predecessor bits the menu ranges over; the predecessor state is then
    ``(free << (16 - 2K)) | (state >> K)``.
    """

    n_states = 1 << (16 - bits)
    c = torch.arange(N_CODEWORDS, dtype=torch.int64)
    free = c >> (16 - bits)
    state = c & (n_states - 1)
    return ((free << (16 - 2 * bits)) | (state >> bits)).contiguous()


def viterbi_windowed_sse(
    source: torch.Tensor,
    values: torch.Tensor,
    bits: int,
    *,
    window: tuple[int, int],
) -> torch.Tensor:
    """Exact-search optimal squared error inside a measurement window.

    ``source`` is ``[batch, steps]`` and ``values`` holds one reconstruction
    per codeword.  Every state starts and ends free, so both boundaries hand
    the encoder about 16 unpaid bits; ``window`` selects an interior span far
    from both, making the result the code's steady-state distortion.  Returns
    the per-sequence sum of squared error over the window, on the paths that
    are globally optimal for the full sequences.
    """

    batch, steps = source.shape
    start, stop = window
    if not 0 <= start < stop <= steps:
        raise ValueError("window must select a nonempty span inside the sequence")
    n_states = 1 << (16 - bits)
    n_branch = 1 << bits
    pred = predecessor_index(bits)
    table = values.to(torch.float32).contiguous()

    cost = torch.zeros(batch, n_states, dtype=torch.float32)
    windowed = torch.zeros(batch, n_states, dtype=torch.float32)

    for step in range(steps):
        err = (table.unsqueeze(0) - source[:, step : step + 1]).square_()
        cand = cost.index_select(1, pred)
        cand += err
        cand = cand.view(batch, n_branch, n_states)
        best, choice = cand.min(dim=1)
        carried = windowed.index_select(1, pred).view(batch, n_branch, n_states)
        if start <= step < stop:
            carried = carried + err.view(batch, n_branch, n_states)
        windowed = carried.gather(1, choice.unsqueeze(1)).squeeze(1).contiguous()
        cost = best.contiguous()

    final = cost.argmin(dim=1)
    return windowed.gather(1, final.unsqueeze(1)).squeeze(1)


def gaussian_sequences(count: int, steps: int, seed: int) -> torch.Tensor:
    """Independent N(0, 1) sequences from a fixed torch generator seed."""

    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, steps, generator=generator, dtype=torch.float32)


def scalar_nearest_mse(levels: np.ndarray, samples: torch.Tensor) -> float:
    """Nearest-neighbour scalar quantizer MSE, for harness validation."""

    table = torch.from_numpy(np.asarray(levels, dtype=np.float32))
    flat = samples.reshape(-1, 1)
    return float((flat - table.unsqueeze(0)).square().min(dim=1).values.mean())


def _per_scale_mse(
    table: torch.Tensor,
    bits: int,
    source: torch.Tensor,
    scales: list[float],
    window: tuple[int, int],
) -> torch.Tensor:
    """Per-scale, per-sequence windowed MSE for one reconstruction table.

    Scaling the code by ``s`` equals quantizing ``source / s`` and rescaling
    the squared error by ``s**2``, so every scale shares one batched search.
    """

    stacked = torch.cat([source / float(s) for s in scales], dim=0)
    scored = viterbi_windowed_sse(stacked, table, bits, window=window)
    factor = torch.tensor([float(s) ** 2 for s in scales]).unsqueeze(1)
    span = window[1] - window[0]
    return scored.view(len(scales), source.shape[0]) / span * factor


@dataclass(frozen=True)
class DistortionResult:
    """Steady-state per-symbol measurement for one code at one rate."""

    code: str
    bits: int
    mse: float
    stderr: float
    fitted_scale: float
    scale_at_grid_edge: bool
    distinct_values: int
    codebook_sigma: float
    n_sequences: int
    measured_symbols: int
    per_sequence_mse: tuple[float, ...]

    def as_dict(self) -> dict:
        payload = dict(self.__dict__)
        payload["per_sequence_mse"] = list(self.per_sequence_mse)
        payload["gaussian_rd_bound"] = 2.0 ** (-2 * self.bits)
        return payload


def measure_code(
    name: str,
    bits: int,
    *,
    sequences: int = 256,
    steps: int = 512,
    window: tuple[int, int] = (128, 384),
    seed: int = 4242,
    scale_span: float = 0.10,
    scale_points: int = 9,
) -> DistortionResult:
    """Measure one code's steady-state per-symbol Gaussian MSE.

    The global scale is fitted in two grid stages around the unit-variance
    normalization ``1 / sigma`` of the code's own values, on the scored
    sequences, mirroring the production one-scalar-per-matrix fit.
    """

    raw = reconstruction_table(name, bits)
    table = torch.from_numpy(raw).float()
    sigma = float(raw.std())
    source = gaussian_sequences(sequences, steps, seed)

    centre = 1.0 / sigma
    coarse = [
        round(float(v), 6)
        for v in np.linspace(1.0 - scale_span, 1.0 + scale_span, scale_points) * centre
    ]
    coarse_mse = _per_scale_mse(table, bits, source, coarse, window).mean(dim=1)
    best = int(coarse_mse.argmin())
    lo = coarse[max(best - 1, 0)]
    hi = coarse[min(best + 1, len(coarse) - 1)]
    fine = [round(float(v), 6) for v in np.linspace(lo, hi, scale_points)]
    fine_mse = _per_scale_mse(table, bits, source, fine, window)
    means = fine_mse.mean(dim=1)
    index = int(means.argmin())
    per_seq = fine_mse[index]

    return DistortionResult(
        code=name,
        bits=bits,
        mse=float(per_seq.mean()),
        stderr=float(per_seq.std(unbiased=True) / math.sqrt(len(per_seq))),
        fitted_scale=fine[index],
        scale_at_grid_edge=bool(
            best in (0, len(coarse) - 1) or index in (0, len(fine) - 1)
        ),
        distinct_values=int(len(np.unique(raw))),
        codebook_sigma=sigma,
        n_sequences=int(len(per_seq)),
        measured_symbols=int(len(per_seq) * (window[1] - window[0])),
        per_sequence_mse=tuple(float(v) for v in per_seq),
    )


def menu_statistics(name: str, bits: int) -> dict:
    """How much choice the Viterbi has at each step, independent of search.

    A menu is the set of ``2**K`` codewords that can arrive at one outgoing
    state.  For SQG codes the statistics include the mean count of distinct
    coarse strata actually exposed per menu; the frozen production graph
    exposes about 2.04 of 4 at K2 and 7.91 of 8 at K3.
    """

    values = reconstruction_table(name, bits)
    n_states = 1 << (16 - bits)
    n_branch = 1 << bits
    # Codeword = (free << (16 - K)) | state, so reshaping to [free, state] and
    # transposing puts one menu per row.
    menu = values.reshape(n_branch, n_states).T

    result = {
        "code": name,
        "bits": bits,
        "mean_distinct_values_per_menu": float(
            np.array([len(np.unique(row)) for row in menu]).mean()
        ),
    }
    if name.startswith("sqg_"):
        from qsrt import sqg_e4m3 as sqg

        if name.startswith("sqg_menu_oriented_"):
            ranks = menu_oriented_rank(bits)
        else:
            ranks = sqg.sqg_xor_rank_permutation(bits).numpy()
        strata = ranks.reshape(n_branch, n_states).T >> (16 - bits)
        result["mean_distinct_strata_per_menu"] = float(
            np.array([len(np.unique(row)) for row in strata]).mean()
        )
        result["nominal_strata"] = n_branch
    return result

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from qsrt.periodic_rate import rate_period, tile_schedules
from qsrt.sqg_e4m3 import sqg_xor_cheb_t12_bytes
from qsrt.sqg_quantizer import install_sqg_quantizer


def _reference_closed_path(
    target: np.ndarray,
    rates: np.ndarray,
    codebooks: np.ndarray,
    initial_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the variable-width recurrence for one fixed closing state."""

    previous: np.ndarray | None = None
    traces: list[np.ndarray] = []
    for position in range(256):
        bits = int(rates[position])
        next_bits = int(rates[(position + 1) & 255])
        next_states = 1 << (16 - next_bits)
        decisions = np.arange(1 << next_bits, dtype=np.int64)[:, None]
        states = np.arange(next_states, dtype=np.int64)[None, :]
        words = (decisions << (16 - next_bits)) | states
        predecessors = words >> bits
        residual = codebooks[bits - 2, words] - float(target[position])
        local = residual * residual
        if previous is None:
            candidate = np.where(predecessors == initial_state, local, np.inf)
        else:
            candidate = local + previous[predecessors]
        trace = np.argmin(candidate, axis=0).astype(np.uint8)
        previous = np.take_along_axis(candidate, trace[None, :], axis=0)[0]
        traces.append(trace)

    assert previous is not None and np.isfinite(previous[initial_state])
    words = np.empty(256, dtype=np.uint16)
    state = initial_state
    for position in range(255, -1, -1):
        bits = int(rates[position])
        next_bits = int(rates[(position + 1) & 255])
        decision = int(traces[position][state])
        word = (decision << (16 - next_bits)) | state
        words[position] = word
        state = word >> bits
    assert state == initial_state
    values = codebooks[rates.astype(np.int64) - 2, words.astype(np.int64)]
    return words, values


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_periodic_quantizer_matches_independent_closed_recurrence() -> None:
    module = SimpleNamespace(quantize_tiles=lambda *_args: "upstream")
    install_sqg_quantizer(module)
    schedules = tile_schedules(rate_period(1, ordering="random", seed=17))
    generator = torch.Generator(device="cuda").manual_seed(1771)
    target = torch.randn((1, 256), generator=generator, device="cuda")
    output, words = module.quantize_tiles(
        target,
        {
            "K": 3,
            "devices": ["cuda:0"],
            "periodic_rate_schedules": schedules,
            "sqg_e4m3_luts_by_bits": {
                bits: sqg_xor_cheb_t12_bytes(bits) for bits in (2, 3, 4)
            },
            "tailbite_context": 128,
        },
    )

    rates = np.asarray(schedules[0], dtype=np.int64)
    actual_words = (words[0].cpu().numpy().astype(np.int64) & 0xFFFF)
    initial_state = int(actual_words[0] >> rates[0])
    codebooks = np.stack(
        [
            sqg_xor_cheb_t12_bytes(bits)
            .view(torch.float8_e4m3fn)
            .float()
            .numpy()
            for bits in (2, 3, 4)
        ]
    )
    reference_words, reference = _reference_closed_path(
        target[0].cpu().numpy(), rates, codebooks, initial_state
    )
    actual = output[0].cpu().numpy()
    actual_sse = float(np.square(actual - target[0].cpu().numpy()).sum())
    reference_sse = float(np.square(reference - target[0].cpu().numpy()).sum())

    # FP16 accumulated path costs can choose a different near-tied path than
    # the float64 reference.  It must nevertheless remain very close to the
    # exact optimum for the identical closing state.
    assert actual_sse <= reference_sse * 1.002
    next_words = np.roll(actual_words, -1)
    assert np.array_equal(
        actual_words & ((1 << (16 - np.roll(rates, -1))) - 1),
        next_words >> np.roll(rates, -1),
    )
    decoded = codebooks[rates - 2, actual_words]
    assert np.array_equal(decoded, actual)
    assert reference_words.shape == actual_words.shape

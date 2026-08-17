from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import qsrt.sqg_quantizer as sqg_quantizer
from qsrt.exl3_reference import reconstruct_trellis_states
from qsrt.sqg_e4m3 import sqg_xor_cheb_t12_bytes
from qsrt.sqg_quantizer import install_sqg_quantizer


def test_trellis_diagnostics_accumulate_post_feedback_tiles() -> None:
    arguments = {
        "return_trellis_diagnostics": True,
        "trellis_diagnostics_ldlq_active": True,
    }
    tiles = torch.ones((2, 256), dtype=torch.float32)
    reconstruction = torch.full_like(tiles, 0.25)
    states = torch.zeros_like(tiles, dtype=torch.int16)

    sqg_quantizer._accumulate_trellis_diagnostics(
        tiles, reconstruction, states, 3, arguments
    )
    report = sqg_quantizer.finalize_trellis_diagnostics(
        arguments["trellis_diagnostics_accumulator"]
    )

    assert report["bits"] == 3
    assert report["tile_count"] == 2
    assert report["target"]["count"] == 512
    assert report["target"]["mean"] == pytest.approx(1.0)
    assert report["residual"]["mean"] == pytest.approx(0.75)
    assert report["target"]["adjacent_correlation_within_path"] == 0.0
    assert report["table"]["used_entries"] == 1
    assert sum(report["table"]["occupancy"]) == 512
    oracle = report["fixed_path_table_oracle"]
    assert oracle["selected_path_frozen_table_sse"] == pytest.approx(288.0)
    assert oracle["selected_path_unconstrained_centroid_sse"] == pytest.approx(0.0)
    assert oracle["selected_path_finite_e4m3_centroid_sse"] == pytest.approx(0.0)
    assert len(oracle["finite_e4m3_table_bytes"]) == 4096
    assert len(oracle["finite_e4m3_table_sha256"]) == 64
    assert "selected trellis paths held fixed" in oracle["evidence_boundary"]
    assert report["measurement_domain"].startswith("exact normalized")


def test_fixed_path_table_oracle_uses_per_entry_target_centroids() -> None:
    arguments = {
        "return_trellis_diagnostics": True,
        "trellis_diagnostics_ldlq_active": True,
    }
    tiles = torch.tensor([[0.875, 1.125] * 128], dtype=torch.float32)
    reconstruction = torch.zeros_like(tiles)
    states = torch.zeros_like(tiles, dtype=torch.int16)

    sqg_quantizer._accumulate_trellis_diagnostics(
        tiles, reconstruction, states, 3, arguments
    )
    report = sqg_quantizer.finalize_trellis_diagnostics(
        arguments["trellis_diagnostics_accumulator"]
    )
    oracle = report["fixed_path_table_oracle"]

    assert oracle["selected_path_frozen_table_sse"] == pytest.approx(260.0)
    assert oracle["selected_path_unconstrained_centroid_sse"] == pytest.approx(4.0)
    assert oracle["selected_path_finite_e4m3_centroid_sse"] == pytest.approx(4.0)
    assert oracle["selected_path_finite_e4m3_relative_reduction"] == pytest.approx(
        1.0 - 4.0 / 260.0
    )


def test_trellis_diagnostics_are_inert_without_explicit_request() -> None:
    arguments: dict = {"trellis_diagnostics_ldlq_active": True}
    tiles = torch.ones((1, 256), dtype=torch.float32)
    sqg_quantizer._accumulate_trellis_diagnostics(
        tiles, tiles, torch.zeros_like(tiles, dtype=torch.int16), 3, arguments
    )
    assert "trellis_diagnostics_accumulator" not in arguments


def test_trellis_diagnostics_ignore_scale_search_probe_tiles() -> None:
    arguments = {"return_trellis_diagnostics": True}
    tiles = torch.ones((1, 256), dtype=torch.float32)
    sqg_quantizer._accumulate_trellis_diagnostics(
        tiles, tiles, torch.zeros_like(tiles, dtype=torch.int16), 3, arguments
    )
    assert "trellis_diagnostics_accumulator" not in arguments


def test_rate_specific_none_dispatches_to_qsrt_mcg(monkeypatch) -> None:
    calls: list[tuple[torch.Tensor, dict]] = []
    procedural_calls: list[tuple] = []

    def original(tiles: torch.Tensor, args: dict):
        calls.append((tiles, args))
        return "upstream"

    def procedural(*args):
        procedural_calls.append(args)

    monkeypatch.setattr(
        sqg_quantizer,
        "_sqg_temp_buffers",
        lambda _device, _bits: (torch.empty(0), torch.empty(0)),
    )
    monkeypatch.setattr(
        sqg_quantizer,
        "_extension",
        lambda: SimpleNamespace(quantize_tiles_procedural=procedural),
    )

    module = SimpleNamespace(quantize_tiles=original)
    install_sqg_quantizer(module)
    tiles = torch.zeros((1, 256), dtype=torch.float32)
    result = module.quantize_tiles(
        tiles,
        {
            "K": 2,
            "devices": ["cuda:0"],
            "sqg_e4m3_luts_by_bits": {2: None, 3: None, 4: None},
        },
    )

    assert all(isinstance(tensor, torch.Tensor) for tensor in result)
    assert not calls
    assert len(procedural_calls) == 1
    assert procedural_calls[0][5:] == (2, 1, 128)


def test_sqg_luts_are_transposed_by_predecessor_for_vector_loads(
    monkeypatch,
) -> None:
    calls: list[tuple] = []

    monkeypatch.setattr(
        sqg_quantizer,
        "_sqg_temp_buffers",
        lambda _device, _bits: (torch.empty(0), torch.empty(0)),
    )
    monkeypatch.setattr(
        sqg_quantizer,
        "_extension",
        lambda: SimpleNamespace(quantize_tiles_sqg=lambda *args: calls.append(args)),
    )

    module = SimpleNamespace(quantize_tiles=lambda *_args: "upstream")
    install_sqg_quantizer(module)
    tiles = torch.zeros((1, 256), dtype=torch.float32)
    for bits in (2, 3, 4):
        raw = torch.arange(65536, dtype=torch.int64).to(torch.uint8)
        module.quantize_tiles(
            tiles,
            {
                "K": bits,
                "devices": ["cuda:0"],
                "sqg_e4m3_lut": raw,
            },
        )
        actual = calls[-1][5]
        predecessors = 1 << bits
        out_edge_pairs = (65536 >> bits) // 2
        expected = (
            raw.reshape(predecessors, out_edge_pairs, 2)
            .permute(1, 0, 2)
            .contiguous()
            .reshape(-1)
        )
        assert torch.equal(actual, expected)
        assert calls[-1][6:] == (bits, 128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_sqg_quantizer_keeps_saturated_cost_paths_closed(bits: int) -> None:
    module = SimpleNamespace(quantize_tiles=lambda *_args: "upstream")
    install_sqg_quantizer(module)
    tiles = torch.full((2, 256), 1000.0, dtype=torch.float32, device="cuda")

    _output, states = module.quantize_tiles(
        tiles,
        {
            "K": bits,
            "devices": ["cuda:0"],
            "sqg_e4m3_lut": sqg_xor_cheb_t12_bytes(bits),
            "tailbite_context": 128,
        },
    )

    edges = states.to(torch.int64) & ((1 << bits) - 1)
    expected = reconstruct_trellis_states(edges, bits)
    assert torch.equal(states, expected)

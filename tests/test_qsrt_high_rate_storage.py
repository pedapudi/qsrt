from __future__ import annotations

import torch

from qsrt.pack.qsrt_atoms_v2 import (
    K3_RECORD_BUNDLE_BYTES,
    K4_RECORD_BUNDLE_BYTES,
    assemble_candidate_records,
    assemble_coupled_h308_pair_atoms,
    assemble_coupled_k2_atoms,
    assemble_record_pair_atoms,
    disassemble_candidate_records,
    pack_local_scale_rate_records,
    pack_matrix_rate_records,
    unpack_local_scale_rate_records,
    unpack_matrix_rate_records,
)
from qsrt.qsrt import (
    FIXED_HIGH_RATE_RECORD_BITS,
    H308,
    K2,
    RECORDS_PER_EXPERT,
    PackedQSRTTrellis,
    QSRTTrellisDescriptor,
)
from qsrt.qsrt_atoms_v2 import (
    BASE_PHYSICAL_TO_LOGICAL_RECORD,
    COUPLED_H308_PROFILE,
    K3_RECORDS,
    K4_RECORDS,
    P22_ATOM_BUNDLE_BYTES,
    logical_rate_record_index,
    physical_to_logical_records,
    P33_ATOM_BUNDLE_BYTES,
    P43_ATOM_BUNDLE_BYTES,
    P33_MATRIX_TRELLIS_BYTES,
    P43_MATRIX_TRELLIS_BYTES,
    P44_MATRIX_TRELLIS_BYTES,
    PURE_K2_PROFILE,
    QSRTAtomsV2Header,
    QSRTAtomsV2Layout,
    expert_pair_is_p43,
    coupled_h308_atom_bundle_bytes,
    coupled_h308_pair_kinds,
    pack_atoms_v2_format_section,
    unpack_atoms_v2_format_section,
)


def _descriptor(rate_axis: str) -> QSRTTrellisDescriptor:
    return QSRTTrellisDescriptor(
        mode_id=H308.mode_id,
        rate_axis=rate_axis,  # type: ignore[arg-type]
        k_tiles=192 if rate_axis == "k" else 224,
        n_tiles=224 if rate_axis == "k" else 192,
    )


def _k2_descriptor(rate_axis: str) -> QSRTTrellisDescriptor:
    return QSRTTrellisDescriptor(
        mode_id=K2.mode_id,
        rate_axis=rate_axis,  # type: ignore[arg-type]
        k_tiles=192 if rate_axis == "k" else 224,
        n_tiles=224 if rate_axis == "k" else 192,
    )


def test_high_rate_record_placement_preserves_funding_and_balances_tp12() -> None:
    assert sorted(BASE_PHYSICAL_TO_LOGICAL_RECORD) == list(range(RECORDS_PER_EXPERT))
    k4_per_rank = [0] * 12
    for expert in range(896):
        mapping = physical_to_logical_records(24, expert)
        rates = tuple(FIXED_HIGH_RATE_RECORD_BITS[logical] for logical in mapping)
        assert sorted(mapping) == list(range(RECORDS_PER_EXPERT))
        assert rates.count(3) == K3_RECORDS
        assert rates.count(4) == K4_RECORDS
        high = [index for index, bits in enumerate(rates) if bits == 4]
        assert all(index % 2 == 0 for index in high)
        assert high[1] - high[0] in (12, -12)
        for record in high:
            k4_per_rank[record // 2] += 1
    assert max(k4_per_rank) - min(k4_per_rank) <= 2

def test_high_rate_physical_placement_preserves_each_funded_record() -> None:
    """Physical balancing must never substitute another same-rate record."""

    logical_bundles = {
        3: tuple((3, index) for index in range(K3_RECORDS)),
        4: tuple((4, index) for index in range(K4_RECORDS)),
    }
    for layer, expert in ((1, 0), (24, 37), (92, 895)):
        physical = []
        for logical_record in physical_to_logical_records(layer, expert):
            bits, rate_index = logical_rate_record_index(logical_record)
            physical.append(logical_bundles[bits][rate_index])
        recovered = [None] * RECORDS_PER_EXPERT
        for physical_record, logical_record in enumerate(
            physical_to_logical_records(layer, expert)
        ):
            recovered[logical_record] = physical[physical_record]
        assert recovered == [
            logical_bundles[3][index] for index in range(K3_RECORDS)
        ] + [logical_bundles[4][index] for index in range(K4_RECORDS)]


def test_high_rate_matrix_records_round_trip_without_decoding() -> None:
    generator = torch.Generator().manual_seed(9127)
    for rate_axis in ("k", "n"):
        descriptor = _descriptor(rate_axis)
        payload = torch.randint(
            -32768,
            32768,
            (descriptor.payload_words,),
            dtype=torch.int16,
            generator=generator,
        )
        packed = PackedQSRTTrellis(descriptor, payload)
        records = pack_matrix_rate_records(packed)
        actual = unpack_matrix_rate_records(records, descriptor)
        assert torch.equal(actual.payload, payload)


def test_high_rate_scale_records_round_trip() -> None:
    scale = torch.arange(3072, dtype=torch.float32).to(torch.float16)
    records = pack_local_scale_rate_records(scale)
    actual = unpack_local_scale_rate_records(records)
    assert records[3].shape == (22, 128)
    assert records[4].shape == (2, 128)
    assert torch.equal(actual, scale)


def test_high_rate_candidate_bundle_round_trip_preserves_funding_order() -> None:
    generator = torch.Generator().manual_seed(10931)
    tensors = {}
    for matrix, rate_axis in (("w1", "k"), ("w3", "k"), ("w2", "n")):
        descriptor = _descriptor(rate_axis)
        shared_part = "svh" if matrix == "w2" else "suh"
        local_part = "suh" if matrix == "w2" else "svh"
        tensors[matrix] = {
            "trellis": torch.randint(
                -32768,
                32768,
                (descriptor.payload_words,),
                dtype=torch.int16,
                generator=generator,
            ),
            shared_part: torch.randn(3584, generator=generator).to(torch.float16),
            local_part: torch.randn(3072, generator=generator).to(torch.float16),
        }
    bundles, shared = assemble_candidate_records(tensors=tensors)
    actual = disassemble_candidate_records(bundles=bundles, shared=shared)
    for matrix in tensors:
        for part in tensors[matrix]:
            assert torch.equal(actual[matrix][part], tensors[matrix][part])


def test_high_rate_physical_record_permutation_preserves_expert_function() -> None:
    generator = torch.Generator().manual_seed(3107)
    channels = 24
    inputs = 7
    outputs = 5
    w1 = torch.randn(channels, inputs, generator=generator)
    w3 = torch.randn(channels, inputs, generator=generator)
    w2 = torch.randn(outputs, channels, generator=generator)
    x = torch.randn(inputs, generator=generator)
    logical = torch.tensor(physical_to_logical_records(24, 37), dtype=torch.long)
    w1_physical = w1.index_select(0, logical)
    w3_physical = w3.index_select(0, logical)
    w2_physical = w2.index_select(1, logical)

    def activation(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(gate) * torch.tanh(up)

    expected = w2 @ activation(w1 @ x, w3 @ x)
    actual = w2_physical @ activation(w1_physical @ x, w3_physical @ x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2.0e-6)


def test_high_rate_record_permutation_commutes_with_block_transform() -> None:
    """The storage permutation moves transform blocks, never lanes within one."""

    generator = torch.Generator().manual_seed(7021)
    values = torch.randn(24, 4, generator=generator)
    hadamard4 = torch.tensor(
        [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
        dtype=torch.float32,
    ) / 2
    logical = torch.tensor(physical_to_logical_records(24, 37), dtype=torch.long)
    transform_then_place = (values @ hadamard4).index_select(0, logical)
    place_then_transform = values.index_select(0, logical) @ hadamard4
    assert torch.equal(place_then_transform, transform_then_place)


def test_atoms_v2_preserves_atom_major_balanced_high_rate_layout() -> None:
    layout = QSRTAtomsV2Layout(24)
    assert P33_ATOM_BUNDLE_BYTES == 129216
    assert P43_ATOM_BUNDLE_BYTES == 150720
    assert layout.disk_bytes == 11_424_530_432
    for pair in range(12):
        p33 = layout.group_experts(8 * pair, p43=False)
        p43 = layout.group_experts(8 * pair, p43=True)
        assert sorted(p33 + p43) == list(range(896))
        assert not set(p33).intersection(p43)
        assert len(p43) in (149, 150)
        for expert in p43:
            assert expert_pair_is_p43(24, expert, pair)
        for stripe in range(8):
            slot = 8 * pair + stripe
            assert layout.atom_slot_payload_bytes(slot) <= (
                layout.atom_slot_stride_bytes
            )


def test_atoms_v2_header_is_canonical_safetensors_metadata() -> None:
    layout = QSRTAtomsV2Layout(24)
    header = QSRTAtomsV2Header(24, layout)
    assert QSRTAtomsV2Header.from_bytes(header.to_bytes()) == header
    assert len(header.to_bytes()) == 4096


def test_pure_k2_records_and_candidate_bundle_round_trip() -> None:
    generator = torch.Generator().manual_seed(7713)
    tensors = {}
    for matrix, rate_axis in (("w1", "k"), ("w3", "k"), ("w2", "n")):
        descriptor = _k2_descriptor(rate_axis)
        shared_part = "svh" if matrix == "w2" else "suh"
        local_part = "suh" if matrix == "w2" else "svh"
        tensors[matrix] = {
            "trellis": torch.randint(
                -32768,
                32768,
                (descriptor.payload_words,),
                dtype=torch.int16,
                generator=generator,
            ),
            shared_part: torch.randn(3584, generator=generator).to(torch.float16),
            local_part: torch.randn(3072, generator=generator).to(torch.float16),
        }
    bundles, shared = assemble_candidate_records(tensors=tensors, mode=K2)
    assert tuple(bundles) == (2,)
    assert bundles[2].shape[0] == RECORDS_PER_EXPERT
    actual = disassemble_candidate_records(
        bundles=bundles, shared=shared, mode=K2
    )
    for matrix in tensors:
        for part in tensors[matrix]:
            assert torch.equal(actual[matrix][part], tensors[matrix][part])

    atoms = assemble_record_pair_atoms(
        bundles[2][0:1],
        bundles[2][1:2],
        p43=False,
        pair_bits=(2, 2),
    )
    assert atoms.shape == (8, 1, P22_ATOM_BUNDLE_BYTES)


def test_coupled_k2_atoms_colocate_preactivation_and_down_coordinates() -> None:
    generator = torch.Generator().manual_seed(9191)
    tensors = {}
    local_scales = {
        "w1": torch.arange(3072, dtype=torch.float32).remainder(997).to(torch.float16),
        "w3": (
            torch.arange(3072, dtype=torch.float32).remainder(991) + 1000
        ).to(torch.float16),
        "w2": (
            torch.arange(3072, dtype=torch.float32).remainder(983) + 2000
        ).to(torch.float16),
    }
    for matrix, rate_axis in (("w1", "k"), ("w3", "k"), ("w2", "n")):
        descriptor = _k2_descriptor(rate_axis)
        shared_part = "svh" if matrix == "w2" else "suh"
        local_part = "suh" if matrix == "w2" else "svh"
        tensors[matrix] = {
            "trellis": torch.randint(
                -32768,
                32768,
                (descriptor.payload_words,),
                dtype=torch.int16,
                generator=generator,
            ),
            shared_part: torch.ones(3584, dtype=torch.float16),
            local_part: local_scales[matrix],
        }
    records, _ = assemble_candidate_records(tensors=tensors, mode=K2)
    atoms = assemble_coupled_k2_atoms(records[2].unsqueeze(0))
    assert atoms.shape == (96, 1, P22_ATOM_BUNDLE_BYTES)

    scale_base = 3 * (P22_ATOM_BUNDLE_BYTES - 192) // 3
    scales = (
        atoms[:, 0, scale_base:]
        .contiguous()
        .view(torch.float16)
        .reshape(96, 3, 32)
    )
    pre = torch.cat((local_scales["w1"], local_scales["w3"]))
    for atom in (0, 1, 47, 48, 95):
        assert torch.equal(scales[atom, 0], pre[64 * atom : 64 * atom + 32])
        assert torch.equal(
            scales[atom, 1], pre[64 * atom + 32 : 64 * atom + 64]
        )
        assert torch.equal(
            scales[atom, 2], local_scales["w2"][32 * atom : 32 * atom + 32]
        )


def test_pure_k2_atoms_v2_header_and_draw_section_are_canonical() -> None:
    layout = QSRTAtomsV2Layout(24, profile=PURE_K2_PROFILE)
    assert P22_ATOM_BUNDLE_BYTES == 86208
    assert layout.disk_bytes == 7_415_300_096
    assert layout.group_experts(0, p43=False) == tuple(range(896))
    assert layout.group_experts(0, p43=True) == ()
    header = QSRTAtomsV2Header(24, layout)
    assert QSRTAtomsV2Header.from_bytes(header.to_bytes()) == header

    draws = tuple(expert % 8 for expert in range(896))
    section = pack_atoms_v2_format_section(PURE_K2_PROFILE, draws)
    formats, actual_draws = unpack_atoms_v2_format_section(
        PURE_K2_PROFILE, section
    )
    assert formats == (K2.name,) * 896
    assert actual_draws == draws


def test_coupled_h308_layout_has_static_tp_independent_pair_extents() -> None:
    layout = QSRTAtomsV2Layout(24, profile=COUPLED_H308_PROFILE)
    expected_kinds = [("P33", "P33")] * 12
    expected_kinds[5] = ("P43", "P33")
    expected_kinds[11] = ("P43", "P44")
    assert [coupled_h308_pair_kinds(pair) for pair in range(12)] == expected_kinds
    assert [
        coupled_h308_atom_bundle_bytes(8 * pair) for pair in range(12)
    ] == [129216] * 5 + [143552] + [129216] * 5 + [157888]
    assert layout.compressed_payload_bytes == 11_422_924_800
    assert layout.disk_bytes == 11_422_957_568

    previous_end = 32768
    for rank in range(12):
        extent = layout.shard_extent(12, rank, require_equal=True)
        assert extent.first_atom_slot == 8 * rank
        assert extent.atom_slots == 8
        assert extent.offset_bytes == previous_end
        assert extent.extent_bytes == extent.payload_bytes
        assert extent.padding_bytes == 0
        previous_end += extent.extent_bytes
    assert previous_end == layout.disk_bytes

    draws = tuple(expert % 8 for expert in range(896))
    section = pack_atoms_v2_format_section(COUPLED_H308_PROFILE, draws)
    formats, actual_draws = unpack_atoms_v2_format_section(
        COUPLED_H308_PROFILE, section
    )
    assert formats == (H308.name,) * 896
    assert actual_draws == draws
    header = QSRTAtomsV2Header(24, layout)
    assert QSRTAtomsV2Header.from_bytes(header.to_bytes()) == header


def test_coupled_h308_pair_assembly_preserves_rate_boundaries() -> None:
    # Every logical record receives a distinct byte.  The packed pair can then
    # be checked segment-by-segment without depending on the trellis values.
    records = {
        3: torch.stack(
            [
                torch.full((K3_RECORD_BUNDLE_BYTES,), index + 1, dtype=torch.uint8)
                for index in range(22)
            ]
        ).unsqueeze(0),
        4: torch.stack(
            [
                torch.full((K4_RECORD_BUNDLE_BYTES,), 101 + index, dtype=torch.uint8)
                for index in range(2)
            ]
        ).unsqueeze(0),
    }

    def assert_segment(
        value: torch.Tensor, begin: int, width: int, expected: int
    ) -> int:
        assert bool(torch.all(value[begin : begin + width] == expected))
        return begin + width

    # Pair 5 carries W1 logical records 20/21 at K3 and 22/23 at K4.
    mixed = assemble_coupled_h308_pair_atoms(records, 5)[0, 0]
    cursor = 0
    for expected, width in (
        (101, P43_MATRIX_TRELLIS_BYTES * 4 // 7),
        (21, P43_MATRIX_TRELLIS_BYTES * 3 // 7),
        (102, P43_MATRIX_TRELLIS_BYTES * 4 // 7),
        (22, P43_MATRIX_TRELLIS_BYTES * 3 // 7),
        (11, P33_MATRIX_TRELLIS_BYTES // 2),
        (12, P33_MATRIX_TRELLIS_BYTES // 2),
    ):
        cursor = assert_segment(mixed, cursor, width, expected)
    assert cursor == 2 * P43_MATRIX_TRELLIS_BYTES + P33_MATRIX_TRELLIS_BYTES

    # Pair 11 repeats the upstream boundary for W3 and funds both W2 records.
    high = assemble_coupled_h308_pair_atoms(records, 11)[0, 0]
    cursor = 0
    for expected, width in (
        (101, P43_MATRIX_TRELLIS_BYTES * 4 // 7),
        (21, P43_MATRIX_TRELLIS_BYTES * 3 // 7),
        (102, P43_MATRIX_TRELLIS_BYTES * 4 // 7),
        (22, P43_MATRIX_TRELLIS_BYTES * 3 // 7),
        (101, P44_MATRIX_TRELLIS_BYTES // 2),
        (102, P44_MATRIX_TRELLIS_BYTES // 2),
    ):
        cursor = assert_segment(high, cursor, width, expected)
    assert cursor == 2 * P43_MATRIX_TRELLIS_BYTES + P44_MATRIX_TRELLIS_BYTES

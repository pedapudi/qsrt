"""Pack and read TP-independent QSRT atoms-v2 profiles."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

import torch
from safetensors import safe_open

from qsrt import constants as C
from qsrt.pack.qsrt_atoms import candidate_layer_path
from qsrt.pack.qsrt_candidates import candidate_tensor_name
from qsrt.qsrt import (
    EXPERTS_PER_LAYER,
    FORMAT_SECTION_BYTES,
    H308,
    INTERMEDIATE_CHANNELS,
    LATENT_CHANNELS,
    LAYER_HEADER_BYTES,
    MATRIX_RATE_AXIS,
    K2,
    RECORDS_PER_EXPERT,
    SHARED_SCALE_SECTION_BYTES,
    TILE_CHANNELS,
    ExpertFormatSpec,
    PackedQSRTTrellis,
    QSRTTrellisDescriptor,
    pack_qsrt_shared_scale_section,
    record_bits,
    unpack_qsrt_shared_scale_section,
)
from qsrt.qsrt_atoms_v2 import (
    COUPLED_H308_PROFILE,
    K3_RECORDS,
    K4_RECORDS,
    P22_ATOM_BUNDLE_BYTES,
    P22_MATRIX_TRELLIS_BYTES,
    P33_ATOM_BUNDLE_BYTES,
    P33_MATRIX_TRELLIS_BYTES,
    P43_ATOM_BUNDLE_BYTES,
    P43_MATRIX_TRELLIS_BYTES,
    P44_MATRIX_TRELLIS_BYTES,
    PROFILE,
    PURE_K2_PROFILE,
    QSRTAtomsV2Header,
    QSRTAtomsV2Layout,
    SCHEMA,
    coupled_h308_atom_bundle_bytes,
    coupled_h308_pair_kinds,
    logical_rate_record_index,
    pack_atoms_v2_format_section,
    physical_to_logical_records,
    unpack_atoms_v2_format_section,
)
from qsrt.qsrt_storage import (
    ATOMS_PER_EXPERT,
    ATOMS_PER_RECORD_PAIR,
    MATRIX_ATOM_SCALE_BYTES,
)


LAYER_PREFIX = "qsrt-layer-"
LAYER_SUFFIX = ".safetensors"
RECORD_CHANNELS = INTERMEDIATE_CHANNELS // RECORDS_PER_EXPERT
MATRIX_RECORD_SCALE_BYTES = RECORD_CHANNELS * torch.float16.itemsize
MATRIX_K3_RECORD_TRELLIS_BYTES = RECORD_CHANNELS * LATENT_CHANNELS * 3 // 8
MATRIX_K4_RECORD_TRELLIS_BYTES = RECORD_CHANNELS * LATENT_CHANNELS * 4 // 8
MATRIX_K2_RECORD_TRELLIS_BYTES = RECORD_CHANNELS * LATENT_CHANNELS * 2 // 8
K2_RECORDS = RECORDS_PER_EXPERT
K3_RECORD_TRELLIS_BYTES = 3 * MATRIX_K3_RECORD_TRELLIS_BYTES
K4_RECORD_TRELLIS_BYTES = 3 * MATRIX_K4_RECORD_TRELLIS_BYTES
RECORD_SCALE_BYTES = 3 * MATRIX_RECORD_SCALE_BYTES
K3_RECORD_BUNDLE_BYTES = K3_RECORD_TRELLIS_BYTES + RECORD_SCALE_BYTES
K4_RECORD_BUNDLE_BYTES = K4_RECORD_TRELLIS_BYTES + RECORD_SCALE_BYTES
K2_RECORD_TRELLIS_BYTES = 3 * MATRIX_K2_RECORD_TRELLIS_BYTES
K2_RECORD_BUNDLE_BYTES = K2_RECORD_TRELLIS_BYTES + RECORD_SCALE_BYTES

_RECORD_COUNTS = {2: K2_RECORDS, 3: K3_RECORDS, 4: K4_RECORDS}
_MATRIX_RECORD_TRELLIS_BYTES = {
    2: MATRIX_K2_RECORD_TRELLIS_BYTES,
    3: MATRIX_K3_RECORD_TRELLIS_BYTES,
    4: MATRIX_K4_RECORD_TRELLIS_BYTES,
}
_RECORD_TRELLIS_BYTES = {
    2: K2_RECORD_TRELLIS_BYTES,
    3: K3_RECORD_TRELLIS_BYTES,
    4: K4_RECORD_TRELLIS_BYTES,
}
_RECORD_BUNDLE_BYTES = {
    2: K2_RECORD_BUNDLE_BYTES,
    3: K3_RECORD_BUNDLE_BYTES,
    4: K4_RECORD_BUNDLE_BYTES,
}

MATRIX_K3_TRELLIS_OFFSETS = {
    matrix: index * MATRIX_K3_RECORD_TRELLIS_BYTES
    for index, matrix in enumerate(C.EXPERT_MATRICES)
}
MATRIX_K4_TRELLIS_OFFSETS = {
    matrix: index * MATRIX_K4_RECORD_TRELLIS_BYTES
    for index, matrix in enumerate(C.EXPERT_MATRICES)
}
MATRIX_K3_SCALE_OFFSETS = {
    matrix: K3_RECORD_TRELLIS_BYTES + index * MATRIX_RECORD_SCALE_BYTES
    for index, matrix in enumerate(C.EXPERT_MATRICES)
}
MATRIX_K4_SCALE_OFFSETS = {
    matrix: K4_RECORD_TRELLIS_BYTES + index * MATRIX_RECORD_SCALE_BYTES
    for index, matrix in enumerate(C.EXPERT_MATRICES)
}
MATRIX_K2_TRELLIS_OFFSETS = {
    matrix: index * MATRIX_K2_RECORD_TRELLIS_BYTES
    for index, matrix in enumerate(C.EXPERT_MATRICES)
}
MATRIX_K2_SCALE_OFFSETS = {
    matrix: K2_RECORD_TRELLIS_BYTES + index * MATRIX_RECORD_SCALE_BYTES
    for index, matrix in enumerate(C.EXPERT_MATRICES)
}
_MATRIX_TRELLIS_OFFSETS = {
    2: MATRIX_K2_TRELLIS_OFFSETS,
    3: MATRIX_K3_TRELLIS_OFFSETS,
    4: MATRIX_K4_TRELLIS_OFFSETS,
}
_MATRIX_SCALE_OFFSETS = {
    2: MATRIX_K2_SCALE_OFFSETS,
    3: MATRIX_K3_SCALE_OFFSETS,
    4: MATRIX_K4_SCALE_OFFSETS,
}


def layer_filename(layer: int) -> str:
    if not 1 <= layer <= 92:
        raise ValueError("Kimi-K3 MoE layer must lie in 1..92")
    return f"{LAYER_PREFIX}{layer:05d}{LAYER_SUFFIX}"


def _flat_candidate_tensor(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    values: int,
    name: str,
) -> torch.Tensor:
    result = tensor.detach().cpu().contiguous().reshape(-1)
    if result.dtype != dtype or result.numel() != values:
        raise ValueError(
            f"{name} must contain {values} contiguous {dtype} values, got "
            f"{result.numel()} {result.dtype} values"
        )
    if dtype != torch.int16 and not bool(torch.all(torch.isfinite(result))):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _matrix_descriptor(matrix: str, mode=H308) -> QSRTTrellisDescriptor:
    rate_axis = MATRIX_RATE_AXIS[matrix]
    return QSRTTrellisDescriptor(
        mode_id=mode.mode_id,
        rate_axis=rate_axis,
        k_tiles=(INTERMEDIATE_CHANNELS if rate_axis == "k" else LATENT_CHANNELS)
        // TILE_CHANNELS,
        n_tiles=(LATENT_CHANNELS if rate_axis == "k" else INTERMEDIATE_CHANNELS)
        // TILE_CHANNELS,
    )


def _record_word_ranges(
    packed: PackedQSRTTrellis,
) -> tuple[tuple[int, int, int], ...]:
    packed.validate()
    cursor = 0
    result = []
    for bits in record_bits(packed.descriptor.mode):
        words = packed.descriptor.tiles_per_record * TILE_CHANNELS * bits
        result.append((cursor, words, bits))
        cursor += words
    if cursor != packed.payload.numel():
        raise AssertionError("atoms-v2 record accounting drifted")
    return tuple(result)


def pack_matrix_rate_records(packed: PackedQSRTTrellis) -> dict[int, torch.Tensor]:
    """Split one logical matrix into records grouped by rate."""

    rates = tuple(sorted(set(record_bits(packed.descriptor.mode))))
    result: dict[int, list[torch.Tensor]] = {bits: [] for bits in rates}
    for offset, words, bits in _record_word_ranges(packed):
        result[bits].append(packed.payload.narrow(0, offset, words).contiguous())
    output = {bits: torch.stack(records).contiguous() for bits, records in result.items()}
    expected = {
        bits: (_RECORD_COUNTS[bits], _MATRIX_RECORD_TRELLIS_BYTES[bits] // 2)
        for bits in rates
    }
    for bits in rates:
        if tuple(output[bits].shape) != expected[bits]:
            raise AssertionError(f"K{bits} record packing geometry drifted")
    return output


def unpack_matrix_rate_records(
    records: dict[int, torch.Tensor], descriptor: QSRTTrellisDescriptor
) -> PackedQSRTTrellis:
    """Invert :func:`pack_matrix_rate_records` without decoding symbols."""

    rates = tuple(sorted(set(record_bits(descriptor.mode))))
    expected = {
        bits: (_RECORD_COUNTS[bits], _MATRIX_RECORD_TRELLIS_BYTES[bits] // 2)
        for bits in rates
    }
    for bits in rates:
        value = records.get(bits)
        if (
            value is None
            or value.dtype != torch.int16
            or tuple(value.shape) != expected[bits]
            or not value.is_contiguous()
        ):
            raise ValueError(f"K{bits} records must be contiguous int16 {expected[bits]}")
    packed = PackedQSRTTrellis(
        descriptor, torch.empty(descriptor.payload_words, dtype=torch.int16)
    )
    cursors = {bits: 0 for bits in rates}
    for offset, words, bits in _record_word_ranges(packed):
        packed.payload.narrow(0, offset, words).copy_(records[bits][cursors[bits]])
        cursors[bits] += 1
    packed.validate()
    return packed


def pack_local_scale_rate_records(scale: torch.Tensor, mode=H308) -> dict[int, torch.Tensor]:
    if (
        scale.dtype != torch.float16
        or tuple(scale.shape) != (INTERMEDIATE_CHANNELS,)
        or not scale.is_contiguous()
    ):
        raise ValueError(
            f"local scale must be contiguous float16 [{INTERMEDIATE_CHANNELS}]"
        )
    logical = scale.reshape(RECORDS_PER_EXPERT, RECORD_CHANNELS)
    result: dict[int, list[torch.Tensor]] = {
        bits: [] for bits in sorted(set(record_bits(mode)))
    }
    for index, bits in enumerate(record_bits(mode)):
        result[bits].append(logical[index])
    return {
        bits: torch.stack(records).contiguous() for bits, records in result.items()
    }


def unpack_local_scale_rate_records(
    records: dict[int, torch.Tensor], mode=H308
) -> torch.Tensor:
    rates = tuple(sorted(set(record_bits(mode))))
    expected = {
        bits: (_RECORD_COUNTS[bits], RECORD_CHANNELS) for bits in rates
    }
    for bits in rates:
        value = records.get(bits)
        if (
            value is None
            or value.dtype != torch.float16
            or tuple(value.shape) != expected[bits]
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"K{bits} scale records must be contiguous float16 {expected[bits]}"
            )
    cursors = {bits: 0 for bits in rates}
    logical = []
    for bits in record_bits(mode):
        logical.append(records[bits][cursors[bits]])
        cursors[bits] += 1
    return torch.stack(logical).reshape(-1).contiguous()


def assemble_candidate_records(
    *, tensors: dict[str, dict[str, torch.Tensor]], mode=H308
) -> tuple[dict[int, torch.Tensor], dict[str, torch.Tensor]]:
    """Split one fixed-profile candidate into logical record bundles."""

    if set(tensors) != set(C.EXPERT_MATRICES):
        raise ValueError("candidate must contain w1, w3, and w2")
    rates = tuple(sorted(set(record_bits(mode))))
    bundles = {
        bits: torch.empty(
            (_RECORD_COUNTS[bits], _RECORD_BUNDLE_BYTES[bits]), dtype=torch.uint8
        )
        for bits in rates
    }
    shared: dict[str, torch.Tensor] = {}
    for matrix in C.EXPERT_MATRICES:
        parts = tensors[matrix]
        if set(parts) != {"trellis", "suh", "svh"}:
            raise ValueError(f"{matrix} candidate components are incomplete")
        descriptor = _matrix_descriptor(matrix, mode)
        trellis = _flat_candidate_tensor(
            parts["trellis"],
            dtype=torch.int16,
            values=descriptor.payload_words,
            name=f"{matrix}.trellis",
        )
        rate_records = pack_matrix_rate_records(PackedQSRTTrellis(descriptor, trellis))
        for bits, value in rate_records.items():
            raw = value.view(torch.uint8).reshape(value.shape[0], -1)
            begin = _MATRIX_TRELLIS_OFFSETS[bits][matrix]
            bundles[bits][:, begin : begin + raw.shape[1]].copy_(raw)

        shared_part = "svh" if matrix == "w2" else "suh"
        local_part = "suh" if matrix == "w2" else "svh"
        shared[f"{matrix}.{shared_part}"] = _flat_candidate_tensor(
            parts[shared_part],
            dtype=torch.float16,
            values=LATENT_CHANNELS,
            name=f"{matrix}.{shared_part}",
        )
        scale_records = pack_local_scale_rate_records(
            _flat_candidate_tensor(
                parts[local_part],
                dtype=torch.float16,
                values=INTERMEDIATE_CHANNELS,
                name=f"{matrix}.{local_part}",
            ),
            mode,
        )
        for bits, value in scale_records.items():
            raw = value.view(torch.uint8).reshape(value.shape[0], -1)
            begin = _MATRIX_SCALE_OFFSETS[bits][matrix]
            bundles[bits][:, begin : begin + MATRIX_RECORD_SCALE_BYTES].copy_(raw)
    return bundles, shared


def disassemble_candidate_records(
    *, bundles: dict[int, torch.Tensor], shared: dict[str, torch.Tensor], mode=H308
) -> dict[str, dict[str, torch.Tensor]]:
    """Invert :func:`assemble_candidate_records` in logical importance order."""

    rates = tuple(sorted(set(record_bits(mode))))
    expected = {
        bits: (_RECORD_COUNTS[bits], _RECORD_BUNDLE_BYTES[bits]) for bits in rates
    }
    for bits, shape in expected.items():
        value = bundles.get(bits)
        if value is None or value.dtype != torch.uint8 or tuple(value.shape) != shape:
            raise ValueError(f"K{bits} bundles must be uint8 {shape}")
    if set(shared) != {"w1.suh", "w3.suh", "w2.svh"}:
        raise ValueError("shared transform inventory is incomplete")
    result: dict[str, dict[str, torch.Tensor]] = {}
    for matrix in C.EXPERT_MATRICES:
        descriptor = _matrix_descriptor(matrix, mode)
        rate_records: dict[int, torch.Tensor] = {}
        scale_records: dict[int, torch.Tensor] = {}
        for bits in rates:
            trellis_begin = _MATRIX_TRELLIS_OFFSETS[bits][matrix]
            trellis_width = _MATRIX_RECORD_TRELLIS_BYTES[bits]
            scale_begin = _MATRIX_SCALE_OFFSETS[bits][matrix]
            rate_records[bits] = (
                bundles[bits][:, trellis_begin : trellis_begin + trellis_width]
                .contiguous()
                .view(torch.int16)
            )
            scale_records[bits] = (
                bundles[bits][
                    :, scale_begin : scale_begin + MATRIX_RECORD_SCALE_BYTES
                ]
                .contiguous()
                .view(torch.float16)
            )
        packed = unpack_matrix_rate_records(rate_records, descriptor)
        shared_part = "svh" if matrix == "w2" else "suh"
        local_part = "suh" if matrix == "w2" else "svh"
        result[matrix] = {
            "trellis": packed.payload,
            shared_part: _flat_candidate_tensor(
                shared[f"{matrix}.{shared_part}"],
                dtype=torch.float16,
                values=LATENT_CHANNELS,
                name=f"{matrix}.{shared_part}",
            ),
            local_part: unpack_local_scale_rate_records(scale_records, mode),
        }
    return result


def _trellis_stripes(
    records: torch.Tensor, *, matrix: str, bits: int
) -> torch.Tensor:
    record_bytes = _RECORD_BUNDLE_BYTES[bits]
    begin = _MATRIX_TRELLIS_OFFSETS[bits][matrix]
    width = _MATRIX_RECORD_TRELLIS_BYTES[bits]
    if records.dtype != torch.uint8 or records.ndim != 2 or records.shape[1] != record_bytes:
        raise ValueError(f"K{bits} record bundles are malformed")
    count = records.shape[0]
    hidden_tiles = LATENT_CHANNELS // TILE_CHANNELS
    raw = records[:, begin : begin + width].contiguous().view(torch.int16)
    if matrix == "w2":
        shaped = raw.reshape(count, ATOMS_PER_RECORD_PAIR, hidden_tiles, 16 * bits)
        stripes = shaped.permute(1, 0, 2, 3)
    else:
        shaped = raw.reshape(count, hidden_tiles, ATOMS_PER_RECORD_PAIR, 16 * bits)
        stripes = shaped.permute(2, 0, 1, 3)
    return stripes.contiguous().view(torch.uint8).reshape(
        ATOMS_PER_RECORD_PAIR, count, -1
    )


def _scale_stripes(records: torch.Tensor, *, matrix: str, bits: int) -> torch.Tensor:
    record_bytes = _RECORD_BUNDLE_BYTES[bits]
    begin = _MATRIX_SCALE_OFFSETS[bits][matrix]
    if records.dtype != torch.uint8 or records.ndim != 2 or records.shape[1] != record_bytes:
        raise ValueError(f"K{bits} record bundles are malformed")
    count = records.shape[0]
    raw = (
        records[:, begin : begin + MATRIX_RECORD_SCALE_BYTES]
        .contiguous()
        .view(torch.float16)
        .reshape(count, ATOMS_PER_RECORD_PAIR, 16)
        .permute(1, 0, 2)
        .contiguous()
    )
    return raw.view(torch.uint8).reshape(ATOMS_PER_RECORD_PAIR, count, -1)


def assemble_record_pair_atoms(
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    p43: bool,
    pair_bits: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Return compact ``[8, experts, bundle]`` P33 or P43 atoms."""

    if low.shape[0] != high.shape[0]:
        raise ValueError("paired record batches must contain the same experts")
    low_bits, high_bits = pair_bits or ((4, 3) if p43 else (3, 3))
    if (low_bits, high_bits) == (2, 2):
        matrix_bytes = P22_MATRIX_TRELLIS_BYTES
        bundle_bytes = P22_ATOM_BUNDLE_BYTES
    elif (low_bits, high_bits) == (3, 3):
        matrix_bytes = P33_MATRIX_TRELLIS_BYTES
        bundle_bytes = P33_ATOM_BUNDLE_BYTES
    elif (low_bits, high_bits) == (4, 3):
        matrix_bytes = P43_MATRIX_TRELLIS_BYTES
        bundle_bytes = P43_ATOM_BUNDLE_BYTES
    else:
        raise ValueError("atoms-v2 supports only P22, P33, and P43")
    count = low.shape[0]
    output = torch.empty(
        (ATOMS_PER_RECORD_PAIR, count, bundle_bytes), dtype=torch.uint8
    )
    for matrix_index, matrix in enumerate(C.EXPERT_MATRICES):
        joined = torch.cat(
            (
                _trellis_stripes(low, matrix=matrix, bits=low_bits),
                _trellis_stripes(high, matrix=matrix, bits=high_bits),
            ),
            dim=2,
        )
        if joined.shape[2] != matrix_bytes:
            raise AssertionError("atoms-v2 matrix stripe accounting drifted")
        begin = matrix_index * matrix_bytes
        output[:, :, begin : begin + matrix_bytes].copy_(joined)
    scale_base = 3 * matrix_bytes
    for matrix_index, matrix in enumerate(C.EXPERT_MATRICES):
        joined = torch.cat(
            (
                _scale_stripes(low, matrix=matrix, bits=low_bits),
                _scale_stripes(high, matrix=matrix, bits=high_bits),
            ),
            dim=2,
        )
        if joined.shape[2] != MATRIX_ATOM_SCALE_BYTES:
            raise AssertionError("atoms-v2 scale stripe accounting drifted")
        begin = scale_base + matrix_index * MATRIX_ATOM_SCALE_BYTES
        output[:, :, begin : begin + MATRIX_ATOM_SCALE_BYTES].copy_(joined)
    return output


def assemble_coupled_k2_atoms(records: torch.Tensor) -> torch.Tensor:
    """Return TP-localizable pure-K2 atoms for the coupled H128 boundary.

    ``records`` is ``[experts, 24, record_bytes]`` in logical intermediate
    order.  A pure-K2 atom owns 32 consecutive post-SiTU coordinates.  Its two
    upstream matrix slots instead carry the corresponding 64 consecutive
    coordinates of the interleaved gate/up preactivation vector.  Consequently
    two adjacent atoms close one complete 128-point preactivation Hadamard and
    four adjacent atoms close one complete 128-point postactivation Hadamard.

    This profile-specific placement is what makes a balanced atom extent
    directly executable at arbitrary supported TP without an intermediate-axis
    collective.  The three matrix slots retain their existing byte sizes; only
    their coupled-coordinate interpretation changes.
    """

    if (
        records.dtype != torch.uint8
        or records.ndim != 3
        or tuple(records.shape[1:])
        != (_RECORD_COUNTS[2], _RECORD_BUNDLE_BYTES[2])
    ):
        raise ValueError(
            "pure-K2 records must be uint8 "
            f"[experts, {_RECORD_COUNTS[2]}, {_RECORD_BUNDLE_BYTES[2]}]"
        )
    experts = int(records.shape[0])
    if experts <= 0:
        raise ValueError("pure-K2 atom assembly requires at least one expert")

    flattened = records.reshape(experts * _RECORD_COUNTS[2], -1)

    def chunks(matrix: str, *, scales: bool) -> torch.Tensor:
        stripes = (
            _scale_stripes(flattened, matrix=matrix, bits=2)
            if scales
            else _trellis_stripes(flattened, matrix=matrix, bits=2)
        )
        # [8 stripes, E*24 records, stripe_bytes] ->
        # [E, 24 records, 4 contiguous 32-channel chunks, chunk_bytes].
        stripe_bytes = int(stripes.shape[-1])
        shaped = stripes.reshape(
            ATOMS_PER_RECORD_PAIR,
            experts,
            _RECORD_COUNTS[2],
            stripe_bytes,
        ).permute(1, 2, 0, 3)
        return (
            shaped.reshape(experts, _RECORD_COUNTS[2], 4, 2, stripe_bytes)
            .reshape(experts, ATOMS_PER_EXPERT, 2 * stripe_bytes)
            .contiguous()
        )

    w1 = chunks("w1", scales=False)
    w3 = chunks("w3", scales=False)
    w2 = chunks("w2", scales=False)
    w1_scale = chunks("w1", scales=True)
    w3_scale = chunks("w3", scales=True)
    w2_scale = chunks("w2", scales=True)

    if tuple(w1.shape[1:]) != (ATOMS_PER_EXPERT, P22_MATRIX_TRELLIS_BYTES):
        raise AssertionError("pure-K2 trellis chunk accounting drifted")
    if tuple(w1_scale.shape[1:]) != (
        ATOMS_PER_EXPERT,
        MATRIX_ATOM_SCALE_BYTES,
    ):
        raise AssertionError("pure-K2 scale chunk accounting drifted")

    # The encoded upstream matrices are the two halves of one length-2I
    # transformed vector, not semantic gate/up matrices.  Consecutive pairs of
    # 32-coordinate chunks become the two FC1 slots of the owning 32-neuron
    # atom.  W2 remains in consecutive postactivation order.
    pre = torch.cat((w1, w3), dim=1)
    pre_scale = torch.cat((w1_scale, w3_scale), dim=1)
    output = torch.empty(
        (ATOMS_PER_EXPERT, experts, P22_ATOM_BUNDLE_BYTES), dtype=torch.uint8
    )
    for atom in range(ATOMS_PER_EXPERT):
        output[atom, :, 0:P22_MATRIX_TRELLIS_BYTES].copy_(pre[:, 2 * atom])
        output[
            atom,
            :,
            P22_MATRIX_TRELLIS_BYTES : 2 * P22_MATRIX_TRELLIS_BYTES,
        ].copy_(pre[:, 2 * atom + 1])
        output[
            atom,
            :,
            2 * P22_MATRIX_TRELLIS_BYTES : 3 * P22_MATRIX_TRELLIS_BYTES,
        ].copy_(w2[:, atom])
        scale_base = 3 * P22_MATRIX_TRELLIS_BYTES
        output[
            atom,
            :,
            scale_base : scale_base + MATRIX_ATOM_SCALE_BYTES,
        ].copy_(pre_scale[:, 2 * atom])
        output[
            atom,
            :,
            scale_base
            + MATRIX_ATOM_SCALE_BYTES : scale_base
            + 2 * MATRIX_ATOM_SCALE_BYTES,
        ].copy_(pre_scale[:, 2 * atom + 1])
        output[
            atom,
            :,
            scale_base
            + 2 * MATRIX_ATOM_SCALE_BYTES : scale_base
            + 3 * MATRIX_ATOM_SCALE_BYTES,
        ].copy_(w2_scale[:, atom])
    return output


def assemble_coupled_h308_pair_atoms(
    records: dict[int, torch.Tensor], physical_pair: int
) -> torch.Tensor:
    """Return one physical coupled H308 record-pair extent.

    The transformed upstream coordinate is the concatenation of the stored
    ``w1`` and ``w3`` matrices.  A 256-neuron postactivation extent consumes
    four consecutive 128-coordinate upstream records.  The two physical FC1
    slots carry alternating records so each slot remains a conventional
    two-record decoder input.  At the two K4 boundaries the K4 record is the
    low plane of P43 and the activation loader swaps the logical half order.
    """

    if not 0 <= physical_pair < 12:
        raise ValueError("coupled H308 physical pair must lie in 0..11")
    expected = {
        3: (K3_RECORDS, K3_RECORD_BUNDLE_BYTES),
        4: (K4_RECORDS, K4_RECORD_BUNDLE_BYTES),
    }
    experts: int | None = None
    for bits, shape in expected.items():
        value = records.get(bits)
        if (
            value is None
            or value.dtype != torch.uint8
            or value.ndim != 3
            or tuple(value.shape[1:]) != shape
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"coupled H308 K{bits} records must be contiguous uint8 "
                f"[experts, {shape[0]}, {shape[1]}]"
            )
        if experts is None:
            experts = int(value.shape[0])
        elif int(value.shape[0]) != experts:
            raise ValueError("coupled H308 record batches disagree on experts")
    if experts is None or experts <= 0:
        raise ValueError("coupled H308 assembly requires at least one expert")

    def record(matrix: str, logical_record: int) -> tuple[torch.Tensor, str, int]:
        bits, index = logical_rate_record_index(logical_record)
        return records[bits][:, index], matrix, bits

    first_pre_record = 4 * physical_pair
    pre_records = []
    for pre_record in range(first_pre_record, first_pre_record + 4):
        matrix = "w1" if pre_record < RECORDS_PER_EXPERT else "w3"
        pre_records.append(record(matrix, pre_record % RECORDS_PER_EXPERT))
    post_records = [
        record("w2", 2 * physical_pair),
        record("w2", 2 * physical_pair + 1),
    ]

    fc1_kind, fc2_kind = coupled_h308_pair_kinds(physical_pair)
    if fc1_kind == "P33":
        slot_pairs = (
            (pre_records[0], pre_records[2]),
            (pre_records[1], pre_records[3]),
        )
    elif fc1_kind == "P43":
        # P43 stores K4 first.  The runtime maps its high K3 half to the first
        # logical preactivation block and its low K4 half to the second.
        slot_pairs = (
            (pre_records[2], pre_records[0]),
            (pre_records[3], pre_records[1]),
        )
    else:
        raise AssertionError("coupled H308 FC1 pair kind drifted")

    matrix_bytes = {
        "P33": P33_MATRIX_TRELLIS_BYTES,
        "P43": P43_MATRIX_TRELLIS_BYTES,
        "P44": P44_MATRIX_TRELLIS_BYTES,
    }
    pair_bits = {"P33": (3, 3), "P43": (4, 3), "P44": (4, 4)}

    def packed_matrix(
        pair: tuple[
            tuple[torch.Tensor, str, int],
            tuple[torch.Tensor, str, int],
        ],
        kind: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        low, high = pair
        if (low[2], high[2]) != pair_bits[kind]:
            raise AssertionError(
                f"coupled H308 {kind} records have K{low[2]}/K{high[2]}"
            )
        trellis = torch.cat(
            (
                _trellis_stripes(low[0], matrix=low[1], bits=low[2]),
                _trellis_stripes(high[0], matrix=high[1], bits=high[2]),
            ),
            dim=2,
        )
        scales = torch.cat(
            (
                _scale_stripes(low[0], matrix=low[1], bits=low[2]),
                _scale_stripes(high[0], matrix=high[1], bits=high[2]),
            ),
            dim=2,
        )
        if trellis.shape != (
            ATOMS_PER_RECORD_PAIR,
            experts,
            matrix_bytes[kind],
        ):
            raise AssertionError("coupled H308 trellis stripe accounting drifted")
        if scales.shape != (
            ATOMS_PER_RECORD_PAIR,
            experts,
            MATRIX_ATOM_SCALE_BYTES,
        ):
            raise AssertionError("coupled H308 scale stripe accounting drifted")
        return trellis, scales

    matrices = [
        packed_matrix(slot_pairs[0], fc1_kind),
        packed_matrix(slot_pairs[1], fc1_kind),
        packed_matrix((post_records[0], post_records[1]), fc2_kind),
    ]
    bundle_bytes = coupled_h308_atom_bundle_bytes(
        physical_pair * ATOMS_PER_RECORD_PAIR
    )
    output = torch.empty(
        (ATOMS_PER_RECORD_PAIR, experts, bundle_bytes), dtype=torch.uint8
    )
    cursor = 0
    for trellis, _scales in matrices:
        width = int(trellis.shape[2])
        output[:, :, cursor : cursor + width].copy_(trellis)
        cursor += width
    for _trellis, scales in matrices:
        output[:, :, cursor : cursor + MATRIX_ATOM_SCALE_BYTES].copy_(scales)
        cursor += MATRIX_ATOM_SCALE_BYTES
    if cursor != bundle_bytes:
        raise AssertionError("coupled H308 atom bundle accounting drifted")
    return output


def _pwrite_exact(descriptor: int, payload: torch.Tensor | bytes, offset: int) -> None:
    data = payload if isinstance(payload, bytes) else payload.numpy().tobytes()
    cursor = 0
    while cursor < len(data):
        written = os.pwrite(descriptor, data[cursor:], offset + cursor)
        if written <= 0:
            raise OSError("short QSRT atoms-v2 write")
        cursor += written


def materialize_atoms_v2_layer(
    candidate_root: str | Path,
    destination: str | Path,
    layer: int,
    *,
    batch_size: int = 8,
    discard_partial: bool = False,
    profile: str = PROFILE,
    rotation_draws: tuple[int, ...] | None = None,
) -> dict[str, int | str]:
    """Write one atoms-v2 layer from a sealed fixed-profile candidate pool."""

    if not 1 <= layer <= 92:
        raise ValueError("Kimi-K3 MoE layer must lie in 1..92")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    candidate_path = candidate_layer_path(candidate_root, layer)
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    partial = destination.with_name(f".{destination.name}.partial")
    if partial.exists():
        if not discard_partial:
            raise FileExistsError(partial)
        partial.unlink()

    layout = QSRTAtomsV2Layout(layer, profile=profile)
    mode = K2 if layout.pure_k2 else H308
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(descriptor, 0, layout.disk_bytes)
        else:
            os.ftruncate(descriptor, layout.disk_bytes)
        _pwrite_exact(descriptor, QSRTAtomsV2Header(layer, layout).to_bytes(), 0)
        _pwrite_exact(
            descriptor,
            pack_atoms_v2_format_section(profile, rotation_draws),
            LAYER_HEADER_BYTES,
        )

        group_experts = (
            {}
            if layout.coupled_h308
            else {
                (pair, p43): layout.group_experts(
                    pair * ATOMS_PER_RECORD_PAIR, p43=p43
                )
                for pair in range(12)
                for p43 in (False, True)
            }
        )
        group_slots = {
            key: {expert: slot for slot, expert in enumerate(experts)}
            for key, experts in group_experts.items()
        }
        shared_reference: dict[str, torch.Tensor] = {}
        with safe_open(candidate_path, framework="pt", device="cpu") as handle:
            for first in range(0, EXPERTS_PER_LAYER, batch_size):
                stop = min(first + batch_size, EXPERTS_PER_LAYER)
                pending: dict[
                    tuple[int, bool],
                    list[tuple[int, torch.Tensor, torch.Tensor]],
                ] = {}
                pure_records: list[tuple[int, torch.Tensor]] = []
                coupled_h308_records: list[
                    tuple[int, dict[int, torch.Tensor]]
                ] = []
                for expert in range(first, stop):
                    tensors = {
                        matrix: {
                            part: handle.get_tensor(
                                candidate_tensor_name(layer, expert, matrix, part)
                            )
                            for part in ("trellis", "suh", "svh")
                        }
                        for matrix in C.EXPERT_MATRICES
                    }
                    records, shared = assemble_candidate_records(
                        tensors=tensors, mode=mode
                    )
                    for name, value in shared.items():
                        reference = shared_reference.setdefault(name, value.clone())
                        if not torch.equal(reference, value):
                            raise ValueError(f"layer-shared transform {name} drifted")
                    if layout.pure_k2:
                        pure_records.append((expert, records[2]))
                        continue
                    if layout.coupled_h308:
                        coupled_h308_records.append((expert, records))
                        continue
                    mapping = (
                        physical_to_logical_records(layer, expert)
                    )
                    for pair in range(12):
                        low_bits, low_index = logical_rate_record_index(
                            mapping[2 * pair]
                        )
                        high_bits, high_index = logical_rate_record_index(
                            mapping[2 * pair + 1]
                        )
                        p43 = low_bits == 4
                        if high_bits != 3 or low_bits not in (3, 4):
                            raise AssertionError(
                                "atoms-v2 pair placement is malformed"
                            )
                        pending.setdefault((pair, p43), []).append(
                            (
                                expert,
                                records[low_bits][low_index],
                                records[high_bits][high_index],
                            )
                        )

                if layout.pure_k2:
                    if [expert for expert, _ in pure_records] != list(
                        range(first, stop)
                    ):
                        raise AssertionError("pure-K2 expert batch is not contiguous")
                    atoms = assemble_coupled_k2_atoms(
                        torch.stack(
                            [records for _, records in pure_records]
                        ).contiguous()
                    )
                    for physical_atom in range(ATOMS_PER_EXPERT):
                        _pwrite_exact(
                            descriptor,
                            atoms[physical_atom],
                            layout.group_offset(physical_atom, p43=False)
                            + first * P22_ATOM_BUNDLE_BYTES,
                        )
                    continue

                if layout.coupled_h308:
                    if [expert for expert, _ in coupled_h308_records] != list(
                        range(first, stop)
                    ):
                        raise AssertionError(
                            "coupled H308 expert batch is not contiguous"
                        )
                    stacked = {
                        bits: torch.stack(
                            [records[bits] for _, records in coupled_h308_records]
                        ).contiguous()
                        for bits in (3, 4)
                    }
                    for pair in range(12):
                        atoms = assemble_coupled_h308_pair_atoms(stacked, pair)
                        bundle_bytes = coupled_h308_atom_bundle_bytes(
                            pair * ATOMS_PER_RECORD_PAIR
                        )
                        for stripe in range(ATOMS_PER_RECORD_PAIR):
                            physical_atom = pair * ATOMS_PER_RECORD_PAIR + stripe
                            _pwrite_exact(
                                descriptor,
                                atoms[stripe],
                                layout.atom_offset(physical_atom)
                                + first * bundle_bytes,
                            )
                    continue

                for (pair, p43), values in pending.items():
                    slots = [group_slots[(pair, p43)][expert] for expert, _, _ in values]
                    if slots != list(range(slots[0], slots[0] + len(slots))):
                        raise AssertionError("batched atom group slots are not contiguous")
                    atoms = assemble_record_pair_atoms(
                        torch.stack([low for _, low, _ in values]).contiguous(),
                        torch.stack([high for _, _, high in values]).contiguous(),
                        p43=p43,
                    )
                    bundle_bytes = layout.group_bundle_bytes(p43=p43)
                    for stripe in range(ATOMS_PER_RECORD_PAIR):
                        physical_atom = pair * ATOMS_PER_RECORD_PAIR + stripe
                        _pwrite_exact(
                            descriptor,
                            atoms[stripe],
                            layout.group_offset(physical_atom, p43=p43)
                            + slots[0] * bundle_bytes,
                        )

        if set(shared_reference) != {"w1.suh", "w3.suh", "w2.svh"}:
            raise AssertionError("shared transform inventory did not close")
        _pwrite_exact(
            descriptor,
            pack_qsrt_shared_scale_section(
                shared_reference["w1.suh"],
                shared_reference["w3.suh"],
                shared_reference["w2.svh"],
            ),
            LAYER_HEADER_BYTES + FORMAT_SECTION_BYTES,
        )
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        partial.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    partial.replace(destination)
    return {"schema": SCHEMA, "layer": layer, "disk_bytes": layout.disk_bytes}


class QSRTAtomsV2Reader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._descriptor = os.open(self.path, os.O_RDONLY)
        try:
            self.header = QSRTAtomsV2Header.from_bytes(
                os.pread(self._descriptor, LAYER_HEADER_BYTES, 0)
            )
            if self.path.stat().st_size != self.header.layout.disk_bytes:
                raise ValueError("QSRT atoms-v2 layer size disagrees with its header")
            format_payload = os.pread(
                self._descriptor, FORMAT_SECTION_BYTES, LAYER_HEADER_BYTES
            )
            if len(format_payload) != FORMAT_SECTION_BYTES:
                raise ValueError("short QSRT atoms-v2 format section")
            formats, self.rotation_draws = unpack_atoms_v2_format_section(
                self.header.layout.profile,
                torch.frombuffer(bytearray(format_payload), dtype=torch.uint8)
            )
            expected_format = (
                K2.name if self.header.layout.pure_k2 else H308.name
            )
            if formats != (expected_format,) * EXPERTS_PER_LAYER:
                raise ValueError(
                    f"QSRT atoms-v2 contains a non-{expected_format} expert"
                )
            shared_offset = LAYER_HEADER_BYTES + FORMAT_SECTION_BYTES
            shared_payload = os.pread(
                self._descriptor, SHARED_SCALE_SECTION_BYTES, shared_offset
            )
            if len(shared_payload) != SHARED_SCALE_SECTION_BYTES:
                raise ValueError("short QSRT atoms-v2 shared-scale section")
            self.shared_scales = unpack_qsrt_shared_scale_section(
                torch.frombuffer(bytearray(shared_payload), dtype=torch.uint8)
            )
        except BaseException:
            os.close(self._descriptor)
            raise

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1

    def __enter__(self) -> "QSRTAtomsV2Reader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def read_group(self, physical_atom: int, *, p43: bool) -> torch.Tensor:
        experts = self.header.layout.group_experts(physical_atom, p43=p43)
        bundle = (
            coupled_h308_atom_bundle_bytes(physical_atom)
            if self.header.layout.coupled_h308 and not p43
            else self.header.layout.group_bundle_bytes(p43=p43)
        )
        size = len(experts) * bundle
        raw = os.pread(
            self._descriptor,
            size,
            self.header.layout.group_offset(physical_atom, p43=p43),
        )
        if len(raw) != size:
            raise ValueError("short QSRT atoms-v2 group")
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(
            len(experts), bundle
        )


__all__ = [
    "QSRTAtomsV2Reader",
    "assemble_candidate_records",
    "assemble_coupled_h308_pair_atoms",
    "assemble_coupled_k2_atoms",
    "assemble_record_pair_atoms",
    "disassemble_candidate_records",
    "layer_filename",
    "materialize_atoms_v2_layer",
    "pack_local_scale_rate_records",
    "pack_matrix_rate_records",
    "unpack_local_scale_rate_records",
    "unpack_matrix_rate_records",
]

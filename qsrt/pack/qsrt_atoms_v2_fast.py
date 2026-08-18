"""Batched pure-K2 atoms-v2 materialization."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

import torch
from safetensors import safe_open

from qsrt import constants as C
from qsrt.pack.qsrt_atoms import candidate_layer_path
from qsrt.pack.qsrt_atoms_v2 import _matrix_descriptor
from qsrt.pack.qsrt_candidates import candidate_tensor_name
from qsrt.qsrt import (
    EXPERTS_PER_LAYER,
    FORMAT_SECTION_BYTES,
    INTERMEDIATE_CHANNELS,
    LATENT_CHANNELS,
    LAYER_HEADER_BYTES,
    RECORDS_PER_EXPERT,
    SHARED_SCALE_SECTION_BYTES,
    K2,
    pack_qsrt_shared_scale_section,
)
from qsrt.qsrt_atoms_v2 import (
    P22_ATOM_BUNDLE_BYTES,
    P22_MATRIX_TRELLIS_BYTES,
    PURE_K2_PROFILE,
    QSRTAtomsV2Header,
    QSRTAtomsV2Layout,
    SCHEMA,
    pack_atoms_v2_format_section,
)
from qsrt.qsrt_storage import ATOMS_PER_EXPERT, MATRIX_ATOM_SCALE_BYTES


_STRIPES_PER_RECORD = ATOMS_PER_EXPERT // RECORDS_PER_EXPERT * 2
_STRIPES_PER_ATOM = 2
_PACKED_WORDS_PER_TILE_ROW = 16 * K2.context_bits[0]
_ORTHOGONAL_TILES = LATENT_CHANNELS // 16


def _validate_batch_tensor(
    value: torch.Tensor,
    *,
    count: int,
    values: int,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if value.dtype != dtype or tuple(value.shape) != (count, values):
        raise ValueError(
            f"{name} must be {dtype} [{count}, {values}], got "
            f"{value.dtype} {tuple(value.shape)}"
        )
    return value.contiguous()


def _trellis_chunks(matrix: str, payload: torch.Tensor) -> torch.Tensor:
    """Map one matrix batch to ``[expert, atom-like chunk, bytes]``."""

    count = int(payload.shape[0])
    descriptor = _matrix_descriptor(matrix, K2)
    payload = _validate_batch_tensor(
        payload,
        count=count,
        values=descriptor.payload_words,
        dtype=torch.int16,
        name=f"{matrix}.trellis",
    )
    if matrix == "w2":
        shaped = payload.reshape(
            count,
            RECORDS_PER_EXPERT,
            _STRIPES_PER_RECORD,
            _ORTHOGONAL_TILES,
            _PACKED_WORDS_PER_TILE_ROW,
        )
    else:
        shaped = payload.reshape(
            count,
            RECORDS_PER_EXPERT,
            _ORTHOGONAL_TILES,
            _STRIPES_PER_RECORD,
            _PACKED_WORDS_PER_TILE_ROW,
        ).permute(0, 1, 3, 2, 4)
    chunks = shaped.reshape(count, ATOMS_PER_EXPERT, -1).contiguous()
    chunks = chunks.view(torch.uint8).reshape(count, ATOMS_PER_EXPERT, -1)
    if chunks.shape[2] != P22_MATRIX_TRELLIS_BYTES:
        raise AssertionError("pure-K2 trellis chunk accounting drifted")
    return chunks


def assemble_coupled_k2_candidate_batch(
    tensors: dict[str, dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pack a batch of complete pure-K2 candidates without record bundles."""

    if set(tensors) != set(C.EXPERT_MATRICES):
        raise ValueError("candidate batch must contain w1, w3, and w2")
    count = int(tensors["w1"]["trellis"].shape[0])
    if count <= 0:
        raise ValueError("candidate batch must contain at least one expert")

    trellis: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    shared: dict[str, torch.Tensor] = {}
    for matrix in C.EXPERT_MATRICES:
        parts = tensors[matrix]
        if set(parts) != {"trellis", "suh", "svh"}:
            raise ValueError(f"{matrix} candidate components are incomplete")
        trellis[matrix] = _trellis_chunks(matrix, parts["trellis"])
        shared_part = "svh" if matrix == "w2" else "suh"
        local_part = "suh" if matrix == "w2" else "svh"
        local = _validate_batch_tensor(
            parts[local_part],
            count=count,
            values=INTERMEDIATE_CHANNELS,
            dtype=torch.float16,
            name=f"{matrix}.{local_part}",
        )
        scales[matrix] = local.reshape(count, ATOMS_PER_EXPERT, 32)
        common = _validate_batch_tensor(
            parts[shared_part],
            count=count,
            values=LATENT_CHANNELS,
            dtype=torch.float16,
            name=f"{matrix}.{shared_part}",
        )
        if not torch.equal(common, common[0].expand_as(common)):
            raise ValueError(f"layer-shared transform {matrix}.{shared_part} drifted")
        shared[f"{matrix}.{shared_part}"] = common[0].detach().cpu().clone()

    output = torch.empty(
        (ATOMS_PER_EXPERT, count, P22_ATOM_BUNDLE_BYTES),
        dtype=torch.uint8,
        device=trellis["w1"].device,
    )
    pre = torch.cat((trellis["w1"], trellis["w3"]), dim=1)
    output[:, :, 0:P22_MATRIX_TRELLIS_BYTES].copy_(
        pre[:, 0::2].permute(1, 0, 2)
    )
    output[
        :,
        :,
        P22_MATRIX_TRELLIS_BYTES : 2 * P22_MATRIX_TRELLIS_BYTES,
    ].copy_(pre[:, 1::2].permute(1, 0, 2))
    output[
        :,
        :,
        2 * P22_MATRIX_TRELLIS_BYTES : 3 * P22_MATRIX_TRELLIS_BYTES,
    ].copy_(trellis["w2"].permute(1, 0, 2))

    scale_base = 3 * P22_MATRIX_TRELLIS_BYTES
    pre_scale = torch.cat((scales["w1"], scales["w3"]), dim=1)
    scale_bytes = MATRIX_ATOM_SCALE_BYTES
    output[:, :, scale_base : scale_base + scale_bytes].copy_(
        pre_scale[:, 0::2].permute(1, 0, 2).contiguous().view(torch.uint8)
    )
    output[:, :, scale_base + scale_bytes : scale_base + 2 * scale_bytes].copy_(
        pre_scale[:, 1::2].permute(1, 0, 2).contiguous().view(torch.uint8)
    )
    output[:, :, scale_base + 2 * scale_bytes :].copy_(
        scales["w2"].permute(1, 0, 2).contiguous().view(torch.uint8)
    )
    return output, shared


def _write_exact(descriptor: int, payload: torch.Tensor | bytes, offset: int) -> None:
    if isinstance(payload, bytes):
        view = memoryview(payload)
    else:
        if payload.device.type != "cpu" or not payload.is_contiguous():
            raise ValueError("atoms-v2 writes require a contiguous CPU tensor")
        view = memoryview(payload.numpy()).cast("B")
    cursor = 0
    while cursor < len(view):
        written = os.pwrite(descriptor, view[cursor:], offset + cursor)
        if written <= 0:
            raise OSError("short QSRT atoms-v2 write")
        cursor += written


def materialize_pure_k2_atoms_v2_layer(
    candidate_root: str | Path,
    destination: str | Path,
    layer: int,
    *,
    batch_size: int = 64,
    rotation_draws: tuple[int, ...],
    device: str | torch.device = "cpu",
    sync: bool = True,
) -> dict[str, int | str]:
    """Write one pure-K2 layer using batched candidate-to-atom transforms."""

    if not 1 <= layer <= 92:
        raise ValueError("Kimi-K3 MoE layer must lie in 1..92")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device(device)
    candidate_path = candidate_layer_path(candidate_root, layer)
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    partial = destination.with_name(f".{destination.name}.partial")
    if partial.exists():
        raise FileExistsError(partial)

    layout = QSRTAtomsV2Layout(layer, profile=PURE_K2_PROFILE)
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    shared_reference: dict[str, torch.Tensor] = {}
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(descriptor, 0, layout.disk_bytes)
        else:
            os.ftruncate(descriptor, layout.disk_bytes)
        _write_exact(descriptor, QSRTAtomsV2Header(layer, layout).to_bytes(), 0)
        _write_exact(
            descriptor,
            pack_atoms_v2_format_section(PURE_K2_PROFILE, rotation_draws),
            LAYER_HEADER_BYTES,
        )

        with safe_open(candidate_path, framework="pt", device="cpu") as handle:
            for first in range(0, EXPERTS_PER_LAYER, batch_size):
                stop = min(first + batch_size, EXPERTS_PER_LAYER)
                tensors: dict[str, dict[str, torch.Tensor]] = {}
                for matrix in C.EXPERT_MATRICES:
                    tensors[matrix] = {}
                    for part in ("trellis", "suh", "svh"):
                        tensors[matrix][part] = torch.stack(
                            [
                                handle.get_tensor(
                                    candidate_tensor_name(
                                        layer, expert, matrix, part
                                    )
                                )
                                for expert in range(first, stop)
                            ]
                        ).to(device)
                atoms, shared = assemble_coupled_k2_candidate_batch(tensors)
                for name, value in shared.items():
                    reference = shared_reference.setdefault(name, value)
                    if not torch.equal(reference, value):
                        raise ValueError(f"layer-shared transform {name} drifted")
                atoms_cpu = atoms.cpu()
                for physical_atom in range(ATOMS_PER_EXPERT):
                    _write_exact(
                        descriptor,
                        atoms_cpu[physical_atom],
                        layout.group_offset(physical_atom, p43=False)
                        + first * P22_ATOM_BUNDLE_BYTES,
                    )

        if set(shared_reference) != {"w1.suh", "w3.suh", "w2.svh"}:
            raise AssertionError("shared transform inventory did not close")
        _write_exact(
            descriptor,
            pack_qsrt_shared_scale_section(
                shared_reference["w1.suh"],
                shared_reference["w3.suh"],
                shared_reference["w2.svh"],
            ),
            LAYER_HEADER_BYTES + FORMAT_SECTION_BYTES,
        )
        if sync:
            os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        partial.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    partial.replace(destination)
    return {"schema": SCHEMA, "layer": layer, "disk_bytes": layout.disk_bytes}

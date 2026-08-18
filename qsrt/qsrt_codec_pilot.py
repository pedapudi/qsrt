"""Model-agnostic helpers for uniform and fixed-average-rate codec pilots.

The production checkpoint adapters own tensor discovery, architecture identity,
and sample selection.  This module owns the reusable numerical experiment:

* divide one shared nonlinear coordinate axis into fixed-width records;
* order that axis once for every coupled expert matrix;
* exchange equal numbers of K2 donor and K4 recipient records around K3;
* encode every R mode with one reconstruction law and exact mean bit rate; and
* select one mode jointly for each caller-defined matrix family.

This is intentionally a weight-distortion pilot utility, not a checkpoint
container or serving layout.  A production port still needs model-native
calibration, storage, and runtime contracts.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from qsrt.exl3_reference import (
    CODEBOOK_SQG_CHEB_NORMAL_E4M3,
    CODEBOOK_SQG_XOR_CHEB_T12,
    decode_exl3_weight,
    decode_qsrt_weight,
    reconstruct_trellis_states,
)
from qsrt.ldlq import SIGMA_REG, make_shared_h
from qsrt.sqg_e4m3 import sqg_codebook_bytes, sqg_xor_cheb_t12_bytes
from qsrt.sqg_high_rate import SQG_FP16_D3L, sqg_fp16_d3l_codebook
from qsrt.sqg_quantizer import finalize_trellis_diagnostics


CODEBOOK_MCG = "mcg"
UNIFORM_CODEBOOKS = (
    CODEBOOK_MCG,
    CODEBOOK_SQG_XOR_CHEB_T12,
    CODEBOOK_SQG_CHEB_NORMAL_E4M3,
    SQG_FP16_D3L,
)


@dataclass(frozen=True)
class FixedAverageRateGeometry:
    """A model-independent K2/K3/K4 record-exchange geometry."""

    axis_channels: int
    record_channels: int = 128
    tile_channels: int = 16
    baseline_bits: int = 3
    donor_bits: int = 2
    recipient_bits: int = 4
    mode_ids: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        integer_fields = (
            self.axis_channels,
            self.record_channels,
            self.tile_channels,
            self.baseline_bits,
            self.donor_bits,
            self.recipient_bits,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields):
            raise TypeError("rate geometry fields must be integers")
        if self.axis_channels <= 0 or self.record_channels <= 0 or self.tile_channels <= 0:
            raise ValueError("rate geometry channel counts must be positive")
        if self.axis_channels % self.record_channels:
            raise ValueError("shared axis must contain whole coding records")
        if self.record_channels % self.tile_channels:
            raise ValueError("coding records must contain whole coding tiles")
        if (self.donor_bits, self.baseline_bits, self.recipient_bits) != (2, 3, 4):
            raise ValueError("the SQG E4M3 pilot supports only K2/K3/K4")
        if not self.mode_ids or len(set(self.mode_ids)) != len(self.mode_ids):
            raise ValueError("mode IDs must be nonempty and unique")
        if tuple(sorted(self.mode_ids)) != self.mode_ids:
            raise ValueError("mode IDs must be ordered")
        if any(
            isinstance(mode, bool)
            or not isinstance(mode, int)
            or mode < 0
            or 2 * mode > self.record_count
            for mode in self.mode_ids
        ):
            raise ValueError("a mode requests too many rate transfers")

    @property
    def record_count(self) -> int:
        return self.axis_channels // self.record_channels

    @property
    def tiles_per_record(self) -> int:
        return self.record_channels // self.tile_channels

    def record_bits(self, mode_id: int) -> tuple[int, ...]:
        if mode_id not in self.mode_ids:
            raise ValueError(f"unsupported rate mode R{mode_id}")
        bits = (
            (self.donor_bits,) * mode_id
            + (self.baseline_bits,) * (self.record_count - 2 * mode_id)
            + (self.recipient_bits,) * mode_id
        )
        if len(bits) != self.record_count:
            raise AssertionError("rate schedule has the wrong record count")
        if sum(bits) != self.baseline_bits * self.record_count:
            raise AssertionError("rate schedule does not preserve mean bit rate")
        return bits

    def tile_bits(self, mode_id: int) -> tuple[int, ...]:
        return tuple(
            bits
            for bits in self.record_bits(mode_id)
            for _ in range(self.tiles_per_record)
        )

    def logical_trellis_bytes(self, matrix_shape: Sequence[int]) -> int:
        if len(matrix_shape) != 2:
            raise ValueError("a codec matrix must be rank two")
        values = math.prod(int(size) for size in matrix_shape)
        total_bits = values * self.baseline_bits
        if total_bits % 8:
            raise ValueError("logical trellis payload is not byte aligned")
        return total_bits // 8


def shared_axis_weight_energy_order(
    matrices: Sequence[tuple[torch.Tensor, int]],
    *,
    group_channels: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one low-to-high weight-energy order for a coupled axis.

    Each input is ``(source_weight, shared_axis)`` in ordinary framework
    orientation.  Scores are summed across every coupled matrix, so the same
    permutation can be applied to gate/up rows and down columns without
    violating the expert's hidden-coordinate symmetry.  Small channel groups
    keep the order deterministic without splitting the caller's grouping unit.
    """

    if not matrices:
        raise ValueError("shared-axis ordering requires at least one matrix")
    if isinstance(group_channels, bool) or not isinstance(group_channels, int):
        raise TypeError("group_channels must be an integer")
    if group_channels <= 0:
        raise ValueError("group_channels must be positive")
    axis_channels: int | None = None
    scores: torch.Tensor | None = None
    for source, axis in matrices:
        if source.ndim != 2 or not torch.is_floating_point(source):
            raise TypeError("source weights must be floating-point matrices")
        if axis not in (0, 1):
            raise ValueError("shared source axis must be 0 or 1")
        current_channels = int(source.shape[axis])
        if axis_channels is None:
            axis_channels = current_channels
            if axis_channels % group_channels:
                raise ValueError("shared axis must contain whole ordering groups")
            scores = torch.zeros(axis_channels, dtype=torch.float64)
        elif current_channels != axis_channels:
            raise ValueError("coupled matrices do not share one axis length")
        reduce_axis = 1 - axis
        contribution = source.detach().cpu().double().square().sum(dim=reduce_axis)
        assert scores is not None
        scores.add_(contribution)
    assert axis_channels is not None and scores is not None
    group_scores = scores.reshape(-1, group_channels).sum(dim=1)
    group_order = torch.argsort(group_scores, stable=True)
    offsets = torch.arange(group_channels, dtype=torch.long)
    permutation = (
        group_order[:, None] * group_channels + offsets[None, :]
    ).reshape(-1)
    if not torch.equal(torch.sort(permutation).values, torch.arange(axis_channels)):
        raise AssertionError("weight-energy ordering is not a permutation")
    return permutation.contiguous(), group_scores.contiguous()


def permute_shared_axis(
    source: torch.Tensor, permutation: torch.Tensor, *, axis: int
) -> torch.Tensor:
    if source.ndim != 2 or axis not in (0, 1):
        raise ValueError("shared-axis permutation expects a matrix axis")
    if permutation.dtype != torch.long or permutation.ndim != 1:
        raise TypeError("shared-axis permutation must be a one-dimensional int64 tensor")
    if permutation.numel() != source.shape[axis]:
        raise ValueError("shared-axis permutation length does not match the matrix")
    return source.index_select(axis, permutation.to(source.device)).contiguous()


def restore_shared_axis(
    ordered: torch.Tensor, permutation: torch.Tensor, *, axis: int
) -> torch.Tensor:
    """Undo :func:`permute_shared_axis` into canonical source coordinates."""

    if ordered.ndim != 2 or axis not in (0, 1):
        raise ValueError("shared-axis restoration expects a matrix axis")
    if permutation.dtype != torch.long or permutation.ndim != 1:
        raise TypeError("shared-axis permutation must be a one-dimensional int64 tensor")
    if permutation.numel() != ordered.shape[axis]:
        raise ValueError("shared-axis permutation length does not match the matrix")
    restored = torch.empty_like(ordered)
    index = permutation.to(ordered.device)
    if axis == 0:
        restored[index] = ordered
    else:
        restored[:, index] = ordered
    return restored.contiguous()


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def unpack_uniform_trellis_states(
    packed: torch.Tensor, bits: int
) -> torch.Tensor:
    """Unpack native EXL words without depending on a model container."""

    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in range(2, 7):
        raise ValueError("uniform trellis rate must be K2 through K6")
    if packed.dtype != torch.int16 or packed.ndim != 3:
        raise TypeError("packed uniform trellis must be a rank-three int16 tensor")
    if packed.shape[-1] != 16 * bits:
        raise ValueError("packed uniform trellis has the wrong word width")
    words = packed.to(dtype=torch.int64) & 0xFFFF
    words = words.reshape(*words.shape[:-1], -1, 2).flip(-1).reshape(words.shape)
    words = words.reshape(*words.shape[:-1], 16, bits)
    word_shifts = torch.arange(15, -1, -1, device=words.device)
    bitstream = ((words[..., None] >> word_shifts) & 1).reshape(
        *words.shape[:-2], 16, bits * 16
    )
    symbol_bits = bitstream.reshape(
        *words.shape[:-2], 16, 16, bits
    )
    symbol_shifts = torch.arange(bits - 1, -1, -1, device=words.device)
    edges = (symbol_bits << symbol_shifts).sum(dim=-1)
    edges = edges.reshape(*words.shape[:-2], 256).to(torch.int16).contiguous()
    return reconstruct_trellis_states(edges, bits)


def pack_uniform_trellis_edges(
    edges: torch.Tensor, bits: int
) -> torch.Tensor:
    """Pack K2--K6 edge symbols in native EXL word ordering.

    This experimental helper intentionally lives outside the frozen QSRT
    container implementation, whose materialized mixed-rate contract remains
    K2/K3/K4.
    """

    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in range(2, 7):
        raise ValueError("uniform trellis rate must be K2 through K6")
    if edges.ndim < 1 or edges.shape[-1] != 256:
        raise ValueError("edges must have a final dimension of 256")
    if edges.dtype == torch.bool or edges.is_floating_point():
        raise TypeError("edges must use an integer dtype")
    values = edges.to(dtype=torch.int64) & ((1 << bits) - 1)
    spans = values.reshape(*values.shape[:-1], 16, 16)
    symbol_shifts = torch.arange(bits - 1, -1, -1, device=values.device)
    bitstream = ((spans[..., None] >> symbol_shifts) & 1).reshape(
        *values.shape[:-1], 16, bits * 16
    )
    word_shifts = torch.arange(15, -1, -1, device=values.device)
    words = (
        bitstream.reshape(*values.shape[:-1], 16, bits, 16) << word_shifts
    ).sum(dim=-1)
    flat = words.reshape(*values.shape[:-1], 16 * bits)
    return (
        flat.reshape(*flat.shape[:-1], -1, 2)
        .flip(-1)
        .reshape(flat.shape)
        .to(dtype=torch.int16)
        .contiguous()
    )


@torch.no_grad()
def encode_uniform_candidate(
    source: torch.Tensor,
    *,
    bits: int,
    codebook: str,
    device: torch.device,
    quantizer_module: Any,
    input_sign_seed: int,
    output_sign_seed: int,
    rate_axis: str | None = None,
    scale_scope_key: object | None = None,
    g_scale_into_sv: bool = False,
    sigma_reg: float = SIGMA_REG,
    tailbite_context: int = 128,
    ldlq_tf32: bool = True,
    g_scale_override: float | None = None,
    ldlq_feedback_multiplier: float = 1.0,
    codebook_values: torch.Tensor | None = None,
    hessian: torch.Tensor | None = None,
    input_hessian: torch.Tensor | None = None,
    output_hessian: torch.Tensor | None = None,
    use_two_sided_traversal_without_output_feedback: bool = False,
    return_replay_state: bool = False,
    return_trellis_diagnostics: bool = False,
) -> dict[str, Any]:
    """Encode one uniform K2--K6 MCG or SQG matrix and decode its payload."""

    if source.ndim != 2 or not torch.is_floating_point(source):
        raise TypeError("source must be a floating-point matrix")
    if (
        isinstance(ldlq_feedback_multiplier, bool)
        or not isinstance(ldlq_feedback_multiplier, (int, float))
        or not math.isfinite(float(ldlq_feedback_multiplier))
        or not 0.0 <= float(ldlq_feedback_multiplier) <= 1.0
    ):
        raise ValueError("ldlq_feedback_multiplier must be a real number in [0, 1]")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in range(2, 7):
        raise ValueError("uniform codec rate must be K2 through K6")
    if not isinstance(use_two_sided_traversal_without_output_feedback, bool):
        raise TypeError(
            "use_two_sided_traversal_without_output_feedback must be a boolean"
        )
    if (
        use_two_sided_traversal_without_output_feedback
        and output_hessian is not None
    ):
        raise ValueError(
            "the zero-output-feedback traversal control cannot be combined "
            "with an output Hessian"
        )
    if codebook_values is None and codebook not in UNIFORM_CODEBOOKS:
        raise ValueError(f"unsupported uniform codebook {codebook!r}")
    if codebook == SQG_FP16_D3L:
        if bits not in (5, 6):
            raise ValueError("sqg_fp16_d3l is defined only for uniform K5/K6")
        if codebook_values is not None:
            raise ValueError(
                "sqg_fp16_d3l uses its frozen descriptor law; "
                "do not supply codebook_values"
            )
        codebook_values = sqg_fp16_d3l_codebook(bits)
    if codebook_values is not None:
        if (
            codebook_values.dtype != torch.float16
            or codebook_values.ndim != 1
            or codebook_values.numel() != 65536
            or not bool(torch.isfinite(codebook_values).all())
        ):
            raise ValueError(
                "an experimental SQG codebook must contain 65,536 finite FP16 values"
            )
    if output_hessian is None:
        canonical_output_hessian = None
        output_hessian_role = "identity_implicit"
        output_hessian_sha256 = None
    else:
        if (
            output_hessian.ndim != 2
            or tuple(output_hessian.shape) != (source.shape[0], source.shape[0])
            or not torch.is_floating_point(output_hessian)
            or not bool(torch.isfinite(output_hessian).all())
        ):
            raise ValueError(
                "output_hessian must be a finite floating-point square matrix "
                "matching the source output dimension"
            )
        canonical_output_hessian = output_hessian.float()
        if not torch.allclose(
            canonical_output_hessian,
            canonical_output_hessian.T,
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("output_hessian must be symmetric")
        if float(canonical_output_hessian.diagonal().mean().item()) <= 0.0:
            raise ValueError("output_hessian must have a positive mean diagonal")
        output_hessian_role = "caller_supplied_downstream_curvature"
        output_hessian_sha256 = tensor_sha256(canonical_output_hessian)
    if device.type != "cuda":
        raise ValueError("uniform codec encoding requires a CUDA device")
    encoder_weight = source.T.float().contiguous().to(device)
    if hessian is not None and input_hessian is not None:
        raise ValueError("supply either hessian or input_hessian, not both")
    selected_input_hessian = (
        input_hessian if input_hessian is not None else hessian
    )
    if selected_input_hessian is None:
        shared_h = make_shared_h(encoder_weight.shape[0], device)
        input_hessian_role = "identity_control"
        input_hessian_sha256 = None
    else:
        if (
            selected_input_hessian.ndim != 2
            or tuple(selected_input_hessian.shape)
            != (encoder_weight.shape[0], encoder_weight.shape[0])
            or not torch.is_floating_point(selected_input_hessian)
            or not bool(torch.isfinite(selected_input_hessian).all())
        ):
            raise ValueError(
                "the input Hessian must be a finite floating-point square matrix "
                "matching the encoder input dimension"
            )
        canonical_hessian = selected_input_hessian.float()
        if not torch.allclose(
            canonical_hessian,
            canonical_hessian.T,
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("input_hessian must be symmetric")
        if float(canonical_hessian.diagonal().mean().item()) <= 0.0:
            raise ValueError("input_hessian must have a positive mean diagonal")
        shared_h = make_shared_h(
            encoder_weight.shape[0], device, canonical_hessian
        )
        input_hessian_role = "caller_supplied_curvature"
        input_hessian_sha256 = tensor_sha256(canonical_hessian)
    quant_args: dict[str, Any] = {
        "K": bits,
        "seed": int(input_sign_seed),
        "sv_seed": int(output_sign_seed),
        "sigma_reg": float(sigma_reg),
        "devices": [str(device)],
        "device_ratios": None,
        "apply_out_scales": False,
        "ldlq_tf32": bool(ldlq_tf32),
        "tailbite_context": int(tailbite_context),
        "ldlq_feedback_multiplier": float(ldlq_feedback_multiplier),
    }
    if codebook_values is not None:
        quant_args["sqg_fp16_lut"] = codebook_values
    elif codebook == CODEBOOK_MCG:
        quant_args["mcg"] = True
    else:
        quant_args["sqg_e4m3_lut"] = sqg_codebook_bytes(
            bits, codebook, rate_axis=rate_axis
        )
    if scale_scope_key is not None:
        quant_args["shared_input_scales_key"] = scale_scope_key
    if g_scale_into_sv:
        quant_args["g_scale_into_sv"] = True
    if g_scale_override is not None:
        quant_args["g_scale_override"] = float(g_scale_override)
    if return_trellis_diagnostics:
        quant_args["return_trellis_diagnostics"] = True
    if canonical_output_hessian is not None:
        quant_args["output_hessian"] = canonical_output_hessian
    if use_two_sided_traversal_without_output_feedback:
        quant_args["use_two_sided_traversal_without_output_feedback"] = True

    torch.cuda.synchronize(device)
    started = time.monotonic()
    _, proxy, tensors = quantizer_module.quantize_qsrt(
        encoder_weight,
        shared_h,
        quant_args,
        False,
        progress_str="",
    )
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    trellis = tensors["trellis"].contiguous()
    suh = tensors["suh"].contiguous()
    svh = tensors["svh"].contiguous()
    expected_trellis = (
        encoder_weight.shape[0] // 16,
        encoder_weight.shape[1] // 16,
        16 * bits,
    )
    if trellis.dtype != torch.int16 or tuple(trellis.shape) != expected_trellis:
        raise ValueError("uniform encoder returned an invalid trellis payload")
    if suh.dtype != torch.float16 or tuple(suh.shape) != (encoder_weight.shape[0],):
        raise ValueError("uniform encoder returned an invalid suh vector")
    if svh.dtype != torch.float16 or tuple(svh.shape) != (encoder_weight.shape[1],):
        raise ValueError("uniform encoder returned an invalid svh vector")

    if codebook == CODEBOOK_MCG and codebook_values is None:
        marker = tensors.get("mcg")
        if (
            not isinstance(marker, torch.Tensor)
            or marker.dtype != torch.int32
            or marker.ndim != 0
            or (int(marker.item()) & 0xFFFF_FFFF) != 0xCBAC1FED
        ):
            raise ValueError("uniform MCG encode omitted its codebook marker")
        decoded = torch.empty_like(encoder_weight, dtype=torch.float16)
        quantizer_module.ext.reconstruct(decoded, trellis, bits, True, False)
        decoded = quantizer_module.preapply_had_l(decoded, 128)
        decoded *= suh.unsqueeze(1)
        decoded = quantizer_module.preapply_had_r(decoded, 128)
        decoded *= svh.unsqueeze(0)
        marker_value: int | None = int(marker.item()) & 0xFFFF_FFFF
    else:
        states = unpack_uniform_trellis_states(trellis, bits)
        if codebook_values is None:
            decoded = decode_exl3_weight(
                states,
                suh,
                svh,
                codebook=codebook,
                bits=bits,
            ).half()
        else:
            decoded = decode_exl3_weight(
                states,
                suh,
                svh,
                codebook_values=codebook_values,
            ).half()
        marker_value = None
    reconstruction = decoded.T.contiguous().cpu()
    logical_bytes = trellis.numel() * trellis.element_size()
    trellis_bpw = logical_bytes * 8.0 / source.numel()
    if trellis_bpw != float(bits):
        raise ValueError("uniform trellis payload does not match its declared rate")
    proxy_value = float(proxy)
    if not math.isfinite(proxy_value):
        raise ValueError("uniform encoder returned a non-finite proxy")
    result = {
        "reconstruction": reconstruction,
        "payload": {
            "profile": (
                "exl3_mcg"
                if codebook == CODEBOOK_MCG
                else SQG_FP16_D3L
                if codebook == SQG_FP16_D3L
                else "offline_sqg_fp16_control"
                if codebook_values is not None
                else "offline_sqg_control"
                if codebook != CODEBOOK_SQG_XOR_CHEB_T12
                else "qsrt_sqg_e4m3"
            ),
            "codebook": codebook,
            "rate": bits,
            "trellis_bytes": logical_bytes,
            "trellis_bpw": trellis_bpw,
            "scale_bytes": suh.numel() * suh.element_size()
            + svh.numel() * svh.element_size(),
            "trellis_sha256": tensor_sha256(trellis),
            "suh_sha256": tensor_sha256(suh),
            "svh_sha256": tensor_sha256(svh),
            "mcg_multiplier": marker_value,
            "proxy_relative_error": proxy_value,
            "encode_seconds": elapsed,
            "input_sign_seed": int(input_sign_seed),
            "output_sign_seed": int(output_sign_seed),
            "rate_axis": rate_axis,
            "g_scale": float(quant_args["g_scale"]),
            "codebook_values_sha256": (
                tensor_sha256(codebook_values)
                if codebook_values is not None
                else None
            ),
            "input_hessian_role": input_hessian_role,
            "input_hessian_sha256": input_hessian_sha256,
            "output_hessian_role": output_hessian_role,
            "output_hessian_sha256": output_hessian_sha256,
            "two_sided_feedback": canonical_output_hessian is not None,
            "output_hessian_factorization": quant_args.get(
                "output_hessian_record"
            ),
            "two_sided_traversal_without_output_feedback": (
                use_two_sided_traversal_without_output_feedback
            ),
            "ldlq_feedback_multiplier": float(ldlq_feedback_multiplier),
        },
    }
    if return_trellis_diagnostics:
        accumulator = quant_args.get("trellis_diagnostics_accumulator")
        if accumulator is None:
            raise RuntimeError("SQG encoder omitted requested trellis diagnostics")
        result["trellis_diagnostics"] = finalize_trellis_diagnostics(accumulator)
    if return_replay_state:
        result["replay_state"] = {
            "bits": bits,
            "trellis": trellis.cpu(),
            "suh": suh.cpu(),
            "svh": svh.cpu(),
        }
    return result


@torch.no_grad()
def replay_uniform_candidate(
    replay_state: Mapping[str, Any],
    *,
    codebook_values: torch.Tensor,
) -> torch.Tensor:
    """Decode one stored uniform trellis path through another scalar table.

    This is an offline codec diagnostic: it holds path and persisted scales
    fixed while replacing only the state-indexed reconstruction labels.
    The returned matrix has the caller/source orientation used by
    :func:`encode_uniform_candidate`.
    """

    bits = replay_state.get("bits")
    trellis = replay_state.get("trellis")
    suh = replay_state.get("suh")
    svh = replay_state.get("svh")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in range(2, 7):
        raise ValueError("replay state rate must be K2 through K6")
    if not all(isinstance(value, torch.Tensor) for value in (trellis, suh, svh)):
        raise TypeError("replay state must contain trellis, suh, and svh tensors")
    if (
        codebook_values.dtype != torch.float16
        or codebook_values.ndim != 1
        or codebook_values.numel() != 65536
        or not bool(torch.isfinite(codebook_values).all())
    ):
        raise ValueError("replay codebook must contain 65,536 finite FP16 values")
    states = unpack_uniform_trellis_states(trellis, bits)
    return (
        decode_exl3_weight(
            states,
            suh,
            svh,
            codebook_values=codebook_values,
        )
        .half()
        .T.contiguous()
        .cpu()
    )


@torch.no_grad()
def encode_fixed_rate_candidates(
    source: torch.Tensor,
    *,
    shared_axis: int,
    permutation: torch.Tensor,
    geometry: FixedAverageRateGeometry,
    device: torch.device,
    quantizer_module: Any,
    input_sign_seed: int,
    output_sign_seed: int,
    scale_scope_key: object | None = None,
    g_scale_into_sv: bool = False,
    sigma_reg: float = SIGMA_REG,
    tailbite_context: int = 128,
    ldlq_tf32: bool = True,
) -> list[dict[str, Any]]:
    """Encode and independently decode every fixed-average-rate candidate."""

    if source.ndim != 2 or not torch.is_floating_point(source):
        raise TypeError("source must be a floating-point matrix")
    if shared_axis not in (0, 1):
        raise ValueError("shared source axis must be 0 or 1")
    if source.shape[shared_axis] != geometry.axis_channels:
        raise ValueError("source shared axis does not match rate geometry")
    if device.type != "cuda":
        raise ValueError("SQG candidate encoding requires a CUDA device")

    ordered_source = permute_shared_axis(source, permutation, axis=shared_axis)
    encoder_weight = ordered_source.T.float().contiguous().to(device)
    rate_axis = "n" if shared_axis == 0 else "k"
    rate_tiles = encoder_weight.shape[1 if rate_axis == "n" else 0] // 16
    if rate_tiles != len(geometry.tile_bits(geometry.mode_ids[0])):
        raise ValueError("rate geometry does not cover the encoded tile axis")
    shared_h = make_shared_h(encoder_weight.shape[0], device)
    rate_luts = {bits: sqg_xor_cheb_t12_bytes(bits) for bits in (2, 3, 4)}
    argument_group: list[dict[str, Any]] = []
    for mode_id in geometry.mode_ids:
        args: dict[str, Any] = {
            "K": geometry.baseline_bits,
            "seed": int(input_sign_seed),
            "sv_seed": int(output_sign_seed),
            "sigma_reg": float(sigma_reg),
            "devices": [str(device)],
            "device_ratios": None,
            "apply_out_scales": False,
            "ldlq_tf32": bool(ldlq_tf32),
            "tailbite_context": int(tailbite_context),
            "mixed_rate_axis": rate_axis,
            "mixed_tile_bits": geometry.tile_bits(mode_id),
            "sqg_e4m3_luts_by_bits": rate_luts,
        }
        if scale_scope_key is not None:
            args["shared_input_scales_key"] = scale_scope_key
        if g_scale_into_sv:
            args["g_scale_into_sv"] = True
        argument_group.append(args)

    torch.cuda.synchronize(device)
    started = time.monotonic()
    raw_groups = quantizer_module.quantize_qsrt_batch(
        [encoder_weight],
        [shared_h],
        [argument_group],
        return_weight_q=False,
    )
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    if len(raw_groups) != 1 or len(raw_groups[0]) != len(geometry.mode_ids):
        raise ValueError("mixed-rate encoder returned the wrong candidate count")

    logical_bytes = geometry.logical_trellis_bytes(source.shape)
    results: list[dict[str, Any]] = []
    for mode_id, raw in zip(geometry.mode_ids, raw_groups[0]):
        tile_bits = geometry.tile_bits(mode_id)
        states = raw["encoded"].contiguous()
        suh = raw["suh"].contiguous()
        svh = raw["svh"].contiguous()
        expected_states = (
            encoder_weight.shape[0] // 16,
            encoder_weight.shape[1] // 16,
            256,
        )
        if states.dtype != torch.int16 or tuple(states.shape) != expected_states:
            raise ValueError("mixed-rate encoder returned invalid trellis states")
        if suh.dtype != torch.float16 or tuple(suh.shape) != (encoder_weight.shape[0],):
            raise ValueError("mixed-rate encoder returned an invalid suh vector")
        if svh.dtype != torch.float16 or tuple(svh.shape) != (encoder_weight.shape[1],):
            raise ValueError("mixed-rate encoder returned an invalid svh vector")
        decoded = decode_qsrt_weight(
            states,
            suh,
            svh,
            rate_axis=rate_axis,
            tile_bits=tile_bits,
            codebook=CODEBOOK_SQG_XOR_CHEB_T12,
        )
        ordered_reconstruction = decoded.half().T.contiguous().cpu()
        reconstruction = restore_shared_axis(
            ordered_reconstruction, permutation, axis=shared_axis
        )
        proxy = float(raw["proxy"])
        if not math.isfinite(proxy):
            raise ValueError("mixed-rate encoder returned a non-finite proxy")
        results.append(
            {
                "mode_id": mode_id,
                "reconstruction": reconstruction,
                "payload": {
                    "profile": "qsrt_sqg_e4m3",
                    "codebook": CODEBOOK_SQG_XOR_CHEB_T12,
                    "record_bits": list(geometry.record_bits(mode_id)),
                    "tile_bits_sha256": tensor_sha256(
                        torch.tensor(tile_bits, dtype=torch.uint8)
                    ),
                    "logical_trellis_bytes": logical_bytes,
                    "trellis_bpw": float(geometry.baseline_bits),
                    "scale_bytes": suh.numel() * suh.element_size()
                    + svh.numel() * svh.element_size(),
                    "proxy_relative_error": proxy,
                    "g_scale": float(raw["g_scale"]),
                    "batch_encode_seconds": elapsed,
                    "rate_axis": rate_axis,
                    "input_sign_seed": int(input_sign_seed),
                    "output_sign_seed": int(output_sign_seed),
                },
            }
        )
    return results


def select_coupled_modes(
    matrix_sse: Mapping[str, Mapping[int, float]],
    families: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, Any]]:
    """Choose the minimum-SSE rate mode once per coupled matrix family."""

    if not matrix_sse or not families:
        raise ValueError("mode selection requires matrices and families")
    covered: list[str] = []
    result: dict[str, dict[str, Any]] = {}
    for family, matrices in families.items():
        members = tuple(matrices)
        if not members:
            raise ValueError(f"rate family {family!r} is empty")
        covered.extend(members)
        try:
            common_modes = set(matrix_sse[members[0]])
        except KeyError as exc:
            raise KeyError(f"unknown matrix in rate family {family!r}") from exc
        for matrix in members[1:]:
            if matrix not in matrix_sse:
                raise KeyError(f"unknown matrix {matrix!r}")
            if set(matrix_sse[matrix]) != common_modes:
                raise ValueError("coupled matrices do not share one mode ladder")
        totals = {
            int(mode): sum(float(matrix_sse[matrix][mode]) for matrix in members)
            for mode in common_modes
        }
        if not totals or any(not math.isfinite(value) or value < 0.0 for value in totals.values()):
            raise ValueError("mode SSE must be finite and nonnegative")
        selected = min(totals, key=lambda mode: (totals[mode], mode))
        result[str(family)] = {
            "matrices": list(members),
            "selected_mode": int(selected),
            "sse_by_mode": {f"R{mode}": totals[mode] for mode in sorted(totals)},
        }
    if len(covered) != len(set(covered)) or set(covered) != set(matrix_sse):
        raise ValueError("rate families must partition the codec matrices exactly")
    return result


def summarize_mode_selections(
    records: Sequence[Mapping[str, Any]], family_names: Sequence[str]
) -> dict[str, Any]:
    """Count per-family and any-family R1+ selections."""

    names = tuple(family_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("family names must be nonempty and unique")
    histograms: dict[str, dict[str, int]] = {name: {} for name in names}
    any_shift = 0
    all_shift = 0
    for record in records:
        selection = record.get("rate_selection")
        if not isinstance(selection, Mapping):
            raise ValueError("expert record has no rate selection")
        modes: list[int] = []
        for name in names:
            family = selection.get(name)
            if not isinstance(family, Mapping):
                raise ValueError(f"expert record has no {name!r} rate family")
            mode = int(family["selected_mode"])
            label = f"R{mode}"
            histograms[name][label] = histograms[name].get(label, 0) + 1
            modes.append(mode)
        any_shift += int(any(mode >= 1 for mode in modes))
        all_shift += int(all(mode >= 1 for mode in modes))
    return {
        "expert_count": len(records),
        "any_family_r1_plus": any_shift,
        "all_families_r1_plus": all_shift,
        "all_r0": len(records) - any_shift,
        "family_histograms": {
            name: dict(sorted(histogram.items()))
            for name, histogram in histograms.items()
        },
    }

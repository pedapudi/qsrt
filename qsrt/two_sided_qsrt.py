"""Format-preserving two-sided rounding for uniform SQG matrices.

The source matrix uses ordinary linear-layer orientation ``[output, input]``.
The input and output Hessians must already be expressed in those source
coordinates.  The helper applies the same signs, scales, Hadamards, global
scale, SQG tile quantizer, and persisted FP16 scales to the one-sided
BlockLDLQ control and the two-sided candidate.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch

from qsrt.distortion_transfer import two_sided_encoder_sse
from qsrt.exl3_reference import (
    CODEBOOK_SQG_XOR_CHEB_T12,
    decode_qsrt_regularized_weight,
    qsrt_regularized_target,
)
from qsrt.gradient_guided_viterbi import (
    LowRankViterbiGradientGuidance,
    ViterbiGradientGuidance,
    shift_two_sided_target,
)
from qsrt.ldlq import SIGMA_REG, make_shared_h
from qsrt.rng import seeded_normal
from qsrt.sqg_e4m3 import sqg_codebook_bytes
from qsrt.two_sided_rounding import (
    TwoSidedRoundingResult,
    bakron_block_round_encoder,
    bakron_block_round_encoder_batch,
    bakron_block_round_encoder_prepared_batch,
    factor_bakron_hessian,
    transform_output_hessian_for_regularization,
    transform_output_hessian_blocks_for_regularization,
)


@dataclass(frozen=True)
class UniformSQGCandidate:
    """One decoded candidate and its unpacked per-tile trellis states."""

    reconstruction: torch.Tensor
    work_reconstruction: torch.Tensor
    states: torch.Tensor
    one_sided_sse: float | None
    two_sided_sse: float | None
    proxy_relative_error: float | None
    seconds: float
    refinement: dict[str, object] | None = None
    gradient_linear_term: float | None = None
    guided_objective: float | None = None


@dataclass(frozen=True)
class UniformSQGTwoSidedResult:
    """A one-sided control and a two-sided candidate with shared preparation."""

    baseline: UniformSQGCandidate | None
    two_sided: UniformSQGCandidate
    input_hessian_work: torch.Tensor
    output_hessian_work: torch.Tensor
    suh: torch.Tensor
    svh: torch.Tensor
    global_scale: float
    h2_vcd: UniformSQGCandidate | None = None
    guided_one_sided: UniformSQGCandidate | None = None


@dataclass(frozen=True)
class UniformSQGBlockTwoSidedResult:
    """Two-sided SQG result under a block-diagonal output Fisher factor."""

    two_sided: UniformSQGCandidate
    input_hessian_work: torch.Tensor
    output_hessian_blocks_work: torch.Tensor
    suh: torch.Tensor
    svh: torch.Tensor
    global_scale: float


@dataclass(frozen=True)
class UniformSQGDirectResult:
    """Uniform SQG reconstruction produced without Hessian feedback."""

    candidate: UniformSQGCandidate
    suh: torch.Tensor
    svh: torch.Tensor
    global_scale: float


@dataclass(frozen=True)
class UniformSQGSharedInputConditioning:
    """Reusable input preparation for matrices with one shared Hessian."""

    input_signs: torch.Tensor
    input_hessian_work: torch.Tensor
    input_factor: torch.Tensor
    bakron_input_factor: torch.Tensor
    bakron_input_reverse: torch.Tensor
    suh: torch.Tensor
    input_dimension: int
    bits: int
    input_sign_seed: int
    rate_axis: str
    scale_scope_key: object


@dataclass(frozen=True)
class _UniformSQGPreparation:
    work_source: torch.Tensor
    input_hessian_work: torch.Tensor
    input_factor: torch.Tensor
    output_hessian_work: torch.Tensor | None
    input_signs: torch.Tensor
    suh: torch.Tensor
    svh: torch.Tensor
    global_scale: float
    quant_args: dict[str, Any]
    tile_quant_args: dict[str, Any]
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor


def _uniform_quant_args(
    *,
    bits: int,
    device: torch.device,
    input_sign_seed: int,
    output_sign_seed: int,
    rate_axis: str | None,
    scale_scope_key: object | None,
    sigma_reg: float,
    tailbite_context: int,
    ldlq_tf32: bool,
    g_scale_into_sv: bool = False,
) -> dict[str, Any]:
    quant_args: dict[str, Any] = {
        # Production QSRT preserves K3 regularization and scale search.  The
        # tile-rate map selects the actual trellis rate.
        "K": 3,
        "seed": int(input_sign_seed),
        "sv_seed": int(output_sign_seed),
        "sigma_reg": float(sigma_reg),
        "devices": [str(device)],
        "device_ratios": None,
        "apply_out_scales": False,
        "ldlq_tf32": bool(ldlq_tf32),
        "tailbite_context": int(tailbite_context),
        "sqg_e4m3_luts_by_bits": {
            rate: sqg_codebook_bytes(
                rate,
                CODEBOOK_SQG_XOR_CHEB_T12,
                rate_axis=rate_axis,
            )
            for rate in (2, 3, 4)
        },
    }
    if scale_scope_key is not None:
        quant_args["shared_input_scales_key"] = scale_scope_key
    if g_scale_into_sv:
        quant_args["g_scale_into_sv"] = True
    return quant_args


def _validate_hessian(
    hessian: torch.Tensor,
    dimension: int,
    name: str,
) -> None:
    if hessian.ndim != 2 or tuple(hessian.shape) != (dimension, dimension):
        raise ValueError(f"{name} has the wrong shape")
    if not bool(torch.all(torch.isfinite(hessian))):
        raise ValueError(f"{name} must be finite")


def _damp_output_hessian(
    hessian: torch.Tensor,
    ratio: float,
) -> tuple[torch.Tensor, float]:
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("output Hessian damping ratio must be finite and nonnegative")
    result = ((hessian + hessian.T) * 0.5).float().contiguous()
    diagonal_mean = float(torch.diagonal(result).double().mean())
    if not math.isfinite(diagonal_mean) or diagonal_mean <= 0.0:
        raise ValueError("output Hessian must have positive diagonal mean")
    damping = ratio * diagonal_mean
    result.diagonal().add_(damping)
    return result, damping


def _damp_output_hessian_blocks(
    hessian_blocks: torch.Tensor,
    ratio: float,
) -> tuple[torch.Tensor, float]:
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("output Hessian damping ratio must be finite and nonnegative")
    result = (
        (hessian_blocks + hessian_blocks.transpose(-1, -2)) * 0.5
    ).float().contiguous()
    diagonal_mean = float(torch.diagonal(result, dim1=-2, dim2=-1).double().mean())
    if not math.isfinite(diagonal_mean) or diagonal_mean <= 0.0:
        raise ValueError("output Hessian blocks must have positive diagonal mean")
    damping = ratio * diagonal_mean
    result.diagonal(dim1=-2, dim2=-1).add_(damping)
    return result, damping


def _decode_work_reconstruction(
    backend: Any,
    work_reconstruction: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
) -> torch.Tensor:
    decoded = backend.preapply_had_l(work_reconstruction, 128)
    decoded *= suh.float().unsqueeze(1)
    decoded = backend.preapply_had_r(decoded, 128)
    decoded *= svh.float().unsqueeze(0)
    return decoded.T.contiguous()


def transform_source_gradient_to_work(
    backend: Any,
    source_gradient: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
) -> torch.Tensor:
    """Apply the adjoint of source reconstruction to an objective gradient."""

    if source_gradient.ndim != 2 or not source_gradient.is_floating_point():
        raise TypeError("source gradient must be a floating-point matrix")
    if source_gradient.shape != (svh.numel(), suh.numel()):
        raise ValueError("source gradient does not match reconstruction scales")
    device = suh.device
    work = source_gradient.to(device=device, dtype=torch.float32).T.contiguous()
    work *= svh.to(device=device, dtype=torch.float32).unsqueeze(0)
    work = backend.preapply_had_r(work, 128)
    work *= suh.to(device=device, dtype=torch.float32).unsqueeze(1)
    return backend.preapply_had_l(work, 128).contiguous()


def transform_source_gradient_factors_to_work(
    backend: Any,
    source_left: torch.Tensor,
    source_right: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform ``source_left @ source_right`` through the decoder adjoint."""

    if source_left.ndim != 2 or source_right.ndim != 2:
        raise ValueError("source gradient factors must be matrices")
    output_dimension, rank = source_left.shape
    if source_right.shape != (rank, suh.numel()) or output_dimension != svh.numel():
        raise ValueError("source gradient factors do not match reconstruction scales")
    if not all(
        value.is_floating_point()
        for value in (source_left, source_right, suh, svh)
    ):
        raise TypeError("source gradient factors and scales must be floating point")
    device = suh.device
    left = source_right.to(device=device, dtype=torch.float32).T.contiguous()
    left *= suh.to(device=device, dtype=torch.float32).unsqueeze(1)
    left = backend.preapply_had_l(left, 128).contiguous()
    right = source_left.to(device=device, dtype=torch.float32).T.contiguous()
    right *= svh.to(device=device, dtype=torch.float32).unsqueeze(0)
    right = backend.preapply_had_r(right, 128).contiguous()
    return left, right


def prepare_uniform_sqg_anchor_input_hessian(
    input_hessian: torch.Tensor,
    anchor_suh: torch.Tensor,
    *,
    bits: int,
    device: torch.device,
    quantizer_module: Any,
    input_sign_seed: int,
    output_sign_seed: int,
    rate_axis: str,
    tailbite_context: int = 128,
    ldlq_tf32: bool = False,
) -> torch.Tensor:
    """Transform a source input Hessian using an exact stored input scale."""

    input_dimension = int(anchor_suh.numel())
    _validate_hessian(input_hessian, input_dimension, "input Hessian")
    quant_args = _uniform_quant_args(
        bits=bits,
        device=device,
        input_sign_seed=input_sign_seed,
        output_sign_seed=output_sign_seed,
        rate_axis=rate_axis,
        scale_scope_key=None,
        sigma_reg=SIGMA_REG,
        tailbite_context=tailbite_context,
        ldlq_tf32=ldlq_tf32,
    )
    hessian_state = make_shared_h(
        input_dimension,
        device,
        input_hessian.to(device=device, dtype=torch.float32),
    )
    q_fallback, _, _, _ = quantizer_module.prepare_capture_H_for_conditioning(
        hessian_state,
        quant_args,
        False,
    )
    if q_fallback:
        raise ValueError("gradient refinement requires a materialized input Hessian")
    q_fallback, result, _, _, _ = (
        quantizer_module.finalize_capture_H_with_conditioning(
            hessian_state,
            anchor_suh.to(device=device, dtype=torch.float32).unsqueeze(1),
            quant_args,
            False,
        )
    )
    if q_fallback or result is None:
        raise ValueError("gradient refinement could not transform the input Hessian")
    return result.to(device=device, dtype=torch.float32)


@torch.no_grad()
def refine_uniform_sqg_anchor_gradient(
    source: torch.Tensor,
    input_hessian: torch.Tensor,
    anchor_states: torch.Tensor,
    anchor_suh: torch.Tensor,
    anchor_svh: torch.Tensor,
    source_gradient_factors: tuple[torch.Tensor, torch.Tensor],
    *,
    bits: int,
    device: torch.device,
    quantizer_module: Any,
    input_sign_seed: int,
    output_sign_seed: int,
    rate_axis: str,
    gradient_strength: float,
    anchor_id: str,
    objective_id: str,
    tailbite_context: int = 128,
    ldlq_tf32: bool = False,
    sweeps: int = 1,
    input_hessian_work: torch.Tensor | None = None,
) -> UniformSQGCandidate:
    """Refine an exact stored payload under dense H plus final-KL gradient.

    The stored reconstruction is candidate zero for every tile.  Scales,
    trellis rate, reconstruction law, and decoder bytes remain fixed.
    """

    if source.ndim != 2 or not source.is_floating_point():
        raise TypeError("source must be a floating-point matrix")
    if bits not in range(2, 7):
        raise ValueError("uniform SQG rate must be K2 through K6")
    if rate_axis not in ("k", "n"):
        raise ValueError("uniform SQG encoding requires a rate axis")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("uniform SQG refinement requires CUDA")
    if sweeps <= 0:
        raise ValueError("gradient refinement requires at least one sweep")
    output_dimension, input_dimension = map(int, source.shape)
    _validate_hessian(input_hessian, input_dimension, "input Hessian")
    expected_states = (input_dimension // 16, output_dimension // 16, 256)
    if tuple(anchor_states.shape) != expected_states:
        raise ValueError("anchor trellis states have incompatible geometry")
    if anchor_suh.numel() != input_dimension or anchor_svh.numel() != output_dimension:
        raise ValueError("anchor scales have incompatible geometry")

    source = source.to(device=device, dtype=torch.float32)
    suh = anchor_suh.to(device=device, dtype=torch.float16).contiguous()
    svh = anchor_svh.to(device=device, dtype=torch.float16).contiguous()
    states = anchor_states.to(device=device, dtype=torch.int16).contiguous()
    work_source = qsrt_regularized_target(source.T.contiguous(), suh, svh)
    axis_tiles = expected_states[0 if rate_axis == "k" else 1]
    work_anchor = decode_qsrt_regularized_weight(
        states,
        rate_axis=rate_axis,
        tile_bits=(bits,) * axis_tiles,
        codebook=CODEBOOK_SQG_XOR_CHEB_T12,
    )

    source_left, source_right = source_gradient_factors
    work_left, work_right = transform_source_gradient_factors_to_work(
        quantizer_module,
        source_left,
        source_right,
        suh,
        svh,
    )
    guidance = LowRankViterbiGradientGuidance(
        left=work_left,
        right=work_right,
        anchor=work_anchor,
        anchor_id=anchor_id,
        objective_id=objective_id,
        strength=float(gradient_strength),
    )
    guidance.validate(work_source.shape)

    quant_args = _uniform_quant_args(
        bits=bits,
        device=device,
        input_sign_seed=input_sign_seed,
        output_sign_seed=output_sign_seed,
        rate_axis=rate_axis,
        scale_scope_key=None,
        sigma_reg=SIGMA_REG,
        tailbite_context=tailbite_context,
        ldlq_tf32=ldlq_tf32,
    )
    if input_hessian_work is None:
        input_hessian_work = prepare_uniform_sqg_anchor_input_hessian(
            input_hessian,
            suh,
            bits=bits,
            device=device,
            quantizer_module=quantizer_module,
            input_sign_seed=input_sign_seed,
            output_sign_seed=output_sign_seed,
            rate_axis=rate_axis,
            tailbite_context=tailbite_context,
            ldlq_tf32=ldlq_tf32,
        )
    elif tuple(input_hessian_work.shape) != (input_dimension, input_dimension):
        raise ValueError("prepared input Hessian has incompatible geometry")
    refinement_args = dict(quant_args)
    refinement_args["K"] = bits
    refinement_args["sqg_e4m3_lut"] = refinement_args[
        "sqg_e4m3_luts_by_bits"
    ][bits]
    refinement_args.pop("sqg_e4m3_luts_by_bits")
    refinement_args.update(
        {
            "h2_viterbi_refine_sweeps": int(sweeps),
            "h2_viterbi_refine_dither_scales": (),
            "h2_viterbi_refine_patterns": 0,
        }
    )

    torch.cuda.synchronize(device)
    started = time.monotonic()
    if gradient_strength == 0.0:
        refined_work = work_anchor
        refined_states = states
        receipt: dict[str, object] = {
            "enabled": False,
            "reason": "zero gradient strength preserves the anchor payload",
            "sweeps": [],
        }
    else:
        refined_work, refined_states, receipt = (
            quantizer_module.refine_uniform_h2_candidate(
                work_source,
                input_hessian_work.to(device=device),
                work_anchor,
                states,
                refinement_args,
                guidance=guidance,
            )
        )
    torch.cuda.synchronize(device)
    seconds = time.monotonic() - started
    reconstruction = _decode_work_reconstruction(
        quantizer_module,
        refined_work,
        suh,
        svh,
    )
    sweeps_receipt = receipt.get("sweeps", [])
    final = sweeps_receipt[-1] if sweeps_receipt else None
    return UniformSQGCandidate(
        reconstruction=reconstruction,
        work_reconstruction=refined_work,
        states=refined_states,
        one_sided_sse=None,
        two_sided_sse=None,
        proxy_relative_error=None,
        seconds=seconds,
        refinement=receipt,
        gradient_linear_term=(
            None if final is None else float(final["linear_after"])
        ),
        guided_objective=(
            None if final is None else float(final["objective_after"])
        ),
    )


def _prepare_uniform_sqg_two_sided(
    source: torch.Tensor,
    input_hessian: torch.Tensor | None,
    output_hessian: torch.Tensor | None,
    *,
    bits: int,
    device: torch.device,
    quantizer_module: Any,
    input_sign_seed: int,
    output_sign_seed: int,
    rate_axis: str,
    scale_scope_key: object | None,
    sigma_reg: float,
    tailbite_context: int,
    ldlq_tf32: bool,
    output_damping_ratio: float,
    global_scale_override: float | None,
    g_scale_into_sv: bool = False,
    shared_input: UniformSQGSharedInputConditioning | None = None,
) -> _UniformSQGPreparation:
    output_dimension, input_dimension = map(int, source.shape)
    encoder_weight = source.T.float().contiguous().to(device)
    if shared_input is None:
        if input_hessian is None:
            raise ValueError("input Hessian is required without shared conditioning")
        input_hessian = input_hessian.float().to(device)
    if output_hessian is not None:
        output_hessian = output_hessian.float().to(device)
        output_hessian, _ = _damp_output_hessian(
            output_hessian, output_damping_ratio
        )
    quant_args = _uniform_quant_args(
        bits=bits,
        device=device,
        input_sign_seed=input_sign_seed,
        output_sign_seed=output_sign_seed,
        rate_axis=rate_axis,
        scale_scope_key=scale_scope_key,
        sigma_reg=sigma_reg,
        tailbite_context=tailbite_context,
        ldlq_tf32=ldlq_tf32,
        g_scale_into_sv=g_scale_into_sv,
    )
    tile_dimension = encoder_weight.shape[1 if rate_axis == "n" else 0] // 16
    quant_args.update(
        {
            "mixed_rate_axis": rate_axis,
            "mixed_tile_bits": (bits,) * tile_dimension,
        }
    )

    hessian_state = None
    if shared_input is None:
        hessian_state = make_shared_h(input_dimension, device, input_hessian)
        q_fallback, _, su, input_diagonal = (
            quantizer_module.prepare_capture_H_for_conditioning(
                hessian_state, quant_args, False
            )
        )
        if q_fallback:
            raise ValueError("two-sided rounding requires a non-fallback input Hessian")
        su = su.to(device)
        if input_diagonal is not None:
            input_diagonal = input_diagonal.to(device)
    else:
        expected = (
            shared_input.input_dimension,
            shared_input.bits,
            shared_input.input_sign_seed,
            shared_input.rate_axis,
            shared_input.scale_scope_key,
        )
        observed = (
            input_dimension,
            bits,
            input_sign_seed,
            rate_axis,
            scale_scope_key,
        )
        if observed != expected:
            raise ValueError("shared input conditioning does not match the matrix")
        su = shared_input.input_signs.to(device=device, dtype=torch.float32).clone()
        input_diagonal = None
        q_fallback = False
    input_signs = su.clone()
    sv = quantizer_module.output_signs(output_dimension, device, quant_args)
    _, work_source, _, su, sv = quantizer_module.regularize(
        encoder_weight,
        su,
        sv,
        quant_args,
        False,
        input_diagonal,
        None,
        skip_g_scale=True,
        q_fallback=False,
    )
    if shared_input is None:
        assert hessian_state is not None
        q_fallback, input_hessian_work, input_factor, su, _ = (
            quantizer_module.finalize_capture_H_with_conditioning(
                hessian_state, su, quant_args, False
            )
        )
        if q_fallback or input_hessian_work is None or input_factor is None:
            raise ValueError("two-sided rounding requires a factored input Hessian")
        input_hessian_work = input_hessian_work.to(device)
        input_factor = input_factor.to(device)
        su = su.to(device)
    else:
        input_hessian_work = shared_input.input_hessian_work
        input_factor = shared_input.input_factor

    if global_scale_override is None:
        global_scale, _ = quantizer_module.g_scale_gss(
            work_source, False, quant_args, pb=None
        )
    else:
        global_scale = float(global_scale_override)
        if not math.isfinite(global_scale) or global_scale <= 0.0:
            raise ValueError("global scale override must be positive and finite")
    work_source, su, sv = quantizer_module.apply_g_scale(
        work_source,
        su,
        sv,
        global_scale,
        into_sv=g_scale_into_sv,
    )
    suh = su.flatten().contiguous().to(dtype=torch.float16, copy=True)
    svh = sv.flatten().contiguous().to(dtype=torch.float16, copy=True)
    if shared_input is not None and not torch.equal(suh, shared_input.suh):
        raise ValueError("shared input scale preparation changed between matrices")
    output_hessian_work = None
    if output_hessian is not None:
        output_hessian_work = transform_output_hessian_for_regularization(
            output_hessian,
            sv.flatten(),
            block_size=128,
        ).to(device=device, dtype=torch.float32)

    tile_quant_args = dict(quant_args)
    tile_quant_args["K"] = bits
    tile_quant_args["sqg_e4m3_lut"] = tile_quant_args[
        "sqg_e4m3_luts_by_bits"
    ][bits]
    tile_quant_args.pop("sqg_e4m3_luts_by_bits")
    tile_quant_args.pop("mixed_rate_axis")
    tile_quant_args.pop("mixed_tile_bits")
    return _UniformSQGPreparation(
        work_source=work_source,
        input_hessian_work=input_hessian_work,
        input_factor=input_factor,
        output_hessian_work=output_hessian_work,
        input_signs=input_signs,
        suh=suh,
        svh=svh,
        global_scale=float(global_scale),
        quant_args=quant_args,
        tile_quant_args=tile_quant_args,
        permutation=quantizer_module.tensor_core_perm(device),
        inverse_permutation=quantizer_module.tensor_core_perm_i(device),
    )


def _shared_input_conditioning(
    preparation: _UniformSQGPreparation,
    *,
    input_dimension: int,
    bits: int,
    input_sign_seed: int,
    rate_axis: str,
    scale_scope_key: object,
) -> UniformSQGSharedInputConditioning:
    bakron_input_factor, bakron_input_reverse = factor_bakron_hessian(
        preparation.input_hessian_work,
        block_size=16,
        dtype=torch.float32,
    )
    return UniformSQGSharedInputConditioning(
        input_signs=preparation.input_signs,
        input_hessian_work=preparation.input_hessian_work,
        input_factor=preparation.input_factor,
        bakron_input_factor=bakron_input_factor,
        bakron_input_reverse=bakron_input_reverse,
        suh=preparation.suh,
        input_dimension=input_dimension,
        bits=bits,
        input_sign_seed=input_sign_seed,
        rate_axis=rate_axis,
        scale_scope_key=scale_scope_key,
    )


def _prepared_tile_quantizer(
    preparation: _UniformSQGPreparation,
    quantizer_module: Any,
) -> Any:
    def quantize_tiles(tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = tiles.reshape(tiles.shape[0], 256).to(
            dtype=preparation.work_source.dtype
        )
        flat = flat[:, preparation.permutation]
        reconstruction, states = quantizer_module.quantize_tiles_multigpu(
            flat, preparation.tile_quant_args
        )
        reconstruction = reconstruction[:, preparation.inverse_permutation]
        return reconstruction.reshape(-1, 16, 16), states

    return quantize_tiles


@torch.no_grad()
def encode_uniform_sqg_direct_batch(
    sources: torch.Tensor,
    *,
    bits: int,
    device: torch.device,
    quantizer_module: Any,
    input_sign_seed: int,
    output_sign_seed: int | Sequence[int],
    rate_axis: str,
    scale_scope_key: object | None,
    shared_scale_axis: Literal["input"] | None = None,
    tailbite_context: int = 128,
    source_gradients: torch.Tensor | None = None,
    gradient_strength: float = 0.0,
) -> list[UniformSQGDirectResult]:
    """Encode uniform SQG matrices with independent tile Viterbi searches.

    When ``source_gradients`` is supplied, the decoder adjoint maps each
    source-coordinate gradient into the trellis basis and adds its linear term
    directly to the Viterbi objective.  Scale fitting remains anchored to the
    unmodified source matrices.
    """

    if sources.ndim != 3 or not sources.shape[0] or not sources.is_floating_point():
        raise ValueError("sources must be a nonempty floating-point matrix batch")
    if sources.device != device or device.type != "cuda":
        raise ValueError("direct SQG encoding requires one CUDA device")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in range(2, 7):
        raise ValueError("uniform SQG rate must be K2 through K6")
    if rate_axis not in ("k", "n"):
        raise ValueError("uniform SQG encoding requires a rate axis")
    if shared_scale_axis not in (None, "input"):
        raise ValueError("shared scale axis must be input or None")
    if (shared_scale_axis == "input") != (scale_scope_key is not None):
        raise ValueError("shared input scales require exactly one scale scope key")
    if not 1 <= tailbite_context <= 128:
        raise ValueError("tail-biting context must lie in 1..128")
    if not math.isfinite(gradient_strength):
        raise ValueError("gradient strength must be finite")
    if source_gradients is not None:
        if source_gradients.shape != sources.shape:
            raise ValueError("source gradients must match the source batch")
        if source_gradients.device != device or not source_gradients.is_floating_point():
            raise ValueError("source gradients must be floating point on the target device")
        if not bool(torch.all(torch.isfinite(source_gradients))):
            raise ValueError("source gradients must be finite")
    elif gradient_strength != 0.0:
        raise ValueError("nonzero gradient strength requires source gradients")

    batch, output_dimension, input_dimension = map(int, sources.shape)
    if output_dimension % 16 or input_dimension % 16:
        raise ValueError("direct SQG matrix dimensions must be divisible by 16")
    if isinstance(output_sign_seed, int) and not isinstance(output_sign_seed, bool):
        output_sign_seeds = (output_sign_seed,) * batch
    else:
        try:
            output_sign_seeds = tuple(output_sign_seed)
        except TypeError as error:
            raise TypeError("output sign seeds must be an integer or sequence") from error
        if len(output_sign_seeds) != batch:
            raise ValueError("output sign seeds must match the matrix batch")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in output_sign_seeds
    ):
        raise ValueError("output sign seeds must be nonnegative integers")

    results: list[UniformSQGDirectResult] = []
    shared_scale_reference: torch.Tensor | None = None
    permutation = quantizer_module.tensor_core_perm(device)
    inverse_permutation = quantizer_module.tensor_core_perm_i(device)
    gradient_batch: Sequence[torch.Tensor | None]
    if source_gradients is None:
        gradient_batch = (None,) * batch
    else:
        gradient_batch = source_gradients
    for source, source_gradient, local_output_seed in zip(
        sources, gradient_batch, output_sign_seeds, strict=True
    ):
        started = time.monotonic()
        quant_args = _uniform_quant_args(
            bits=bits,
            device=device,
            input_sign_seed=input_sign_seed,
            output_sign_seed=local_output_seed,
            rate_axis=rate_axis,
            scale_scope_key=scale_scope_key,
            sigma_reg=SIGMA_REG,
            tailbite_context=tailbite_context,
            ldlq_tf32=False,
            g_scale_into_sv=shared_scale_axis == "input",
        )
        input_signs = (
            (seeded_normal(input_dimension, device=device, seed=input_sign_seed).sign() + 1e-5)
            .sign()
            .float()
            .unsqueeze(1)
        )
        output_signs = quantizer_module.output_signs(
            output_dimension, device, quant_args
        )
        _, work_source, global_scale, input_scales, output_scales = (
            quantizer_module.regularize(
                source.T.float().contiguous(),
                input_signs,
                output_signs,
                quant_args,
                False,
                None,
                None,
                q_fallback=False,
            )
        )
        if source_gradient is not None and gradient_strength != 0.0:
            work_source = work_source - float(gradient_strength) * (
                transform_source_gradient_to_work(
                    quantizer_module,
                    source_gradient,
                    input_scales.flatten(),
                    output_scales.flatten(),
                )
            )
        tile_args = dict(quant_args)
        tile_args["K"] = bits
        tile_args["sqg_e4m3_lut"] = tile_args["sqg_e4m3_luts_by_bits"][bits]
        tile_args.pop("sqg_e4m3_luts_by_bits")

        input_tiles = input_dimension // 16
        output_tiles = output_dimension // 16
        tiles = (
            work_source.reshape(input_tiles, 16, output_tiles, 16)
            .permute(0, 2, 1, 3)
            .reshape(input_tiles * output_tiles, 256)
        )
        quantized, states = quantizer_module.quantize_tiles_multigpu(
            tiles[:, permutation].contiguous(), tile_args
        )
        if states.dtype != torch.int16 or tuple(states.shape) != (
            input_tiles * output_tiles,
            256,
        ):
            raise ValueError("direct SQG quantizer returned invalid states")
        work_reconstruction = (
            quantized[:, inverse_permutation]
            .reshape(input_tiles, output_tiles, 16, 16)
            .permute(0, 2, 1, 3)
            .reshape(input_dimension, output_dimension)
            .contiguous()
        )
        states = states.reshape(input_tiles, output_tiles, 256).contiguous()
        suh = input_scales.flatten().contiguous().to(
            dtype=torch.float16, copy=True
        )
        svh = output_scales.flatten().contiguous().to(
            dtype=torch.float16, copy=True
        )
        shared_scale = suh if shared_scale_axis == "input" else None
        if shared_scale is not None:
            if shared_scale_reference is None:
                shared_scale_reference = shared_scale.clone()
            elif not torch.equal(shared_scale_reference, shared_scale):
                raise ValueError(
                    f"direct SQG {shared_scale_axis} scale drifted within the batch"
                )
        reconstruction = _decode_work_reconstruction(
            quantizer_module, work_reconstruction, suh, svh
        )
        torch.cuda.synchronize(device)
        results.append(
            UniformSQGDirectResult(
                candidate=UniformSQGCandidate(
                    reconstruction=reconstruction,
                    work_reconstruction=work_reconstruction,
                    states=states,
                    one_sided_sse=None,
                    two_sided_sse=None,
                    proxy_relative_error=None,
                    seconds=time.monotonic() - started,
                ),
                suh=suh,
                svh=svh,
                global_scale=float(global_scale),
            )
        )
    return results


@torch.no_grad()
def encode_uniform_sqg_direct_work_batch(
    work_targets: torch.Tensor,
    *,
    suhs: torch.Tensor,
    svhs: torch.Tensor,
    bits: int,
    device: torch.device,
    quantizer_module: Any,
    input_sign_seed: int,
    output_sign_seed: int | Sequence[int],
    rate_axis: str,
    tailbite_context: int = 128,
) -> list[UniformSQGDirectResult]:
    """Encode trellis-basis targets while preserving supplied decoder scales."""

    if (
        work_targets.ndim != 3
        or not work_targets.shape[0]
        or not work_targets.is_floating_point()
    ):
        raise ValueError("work targets must be a nonempty floating-point matrix batch")
    if work_targets.device != device or device.type != "cuda":
        raise ValueError("direct SQG encoding requires one CUDA device")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in range(2, 7):
        raise ValueError("uniform SQG rate must be K2 through K6")
    if rate_axis not in ("k", "n"):
        raise ValueError("uniform SQG encoding requires a rate axis")
    if not 1 <= tailbite_context <= 128:
        raise ValueError("tail-biting context must lie in 1..128")

    batch, input_dimension, output_dimension = map(int, work_targets.shape)
    if output_dimension % 16 or input_dimension % 16:
        raise ValueError("direct SQG matrix dimensions must be divisible by 16")
    if tuple(suhs.shape) != (batch, input_dimension):
        raise ValueError("input scales do not match the work-target batch")
    if tuple(svhs.shape) != (batch, output_dimension):
        raise ValueError("output scales do not match the work-target batch")
    if not all(value.is_floating_point() for value in (suhs, svhs)):
        raise TypeError("decoder scales must be floating point")
    if not all(bool(torch.all(torch.isfinite(value))) for value in (suhs, svhs)):
        raise ValueError("decoder scales must be finite")
    if any(bool(torch.any(value == 0)) for value in (suhs, svhs)):
        raise ValueError("decoder scales must be nonzero")
    if isinstance(output_sign_seed, int) and not isinstance(output_sign_seed, bool):
        output_sign_seeds = (output_sign_seed,) * batch
    else:
        try:
            output_sign_seeds = tuple(output_sign_seed)
        except TypeError as error:
            raise TypeError("output sign seeds must be an integer or sequence") from error
        if len(output_sign_seeds) != batch:
            raise ValueError("output sign seeds must match the matrix batch")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in output_sign_seeds
    ):
        raise ValueError("output sign seeds must be nonnegative integers")

    input_tiles = input_dimension // 16
    output_tiles = output_dimension // 16
    permutation = quantizer_module.tensor_core_perm(device)
    inverse_permutation = quantizer_module.tensor_core_perm_i(device)
    results: list[UniformSQGDirectResult] = []
    for work_target, suh, svh, local_output_seed in zip(
        work_targets, suhs, svhs, output_sign_seeds, strict=True
    ):
        started = time.monotonic()
        quant_args = _uniform_quant_args(
            bits=bits,
            device=device,
            input_sign_seed=input_sign_seed,
            output_sign_seed=local_output_seed,
            rate_axis=rate_axis,
            scale_scope_key=None,
            sigma_reg=SIGMA_REG,
            tailbite_context=tailbite_context,
            ldlq_tf32=False,
        )
        tile_args = dict(quant_args)
        tile_args["K"] = bits
        tile_args["sqg_e4m3_lut"] = tile_args["sqg_e4m3_luts_by_bits"][bits]
        tile_args.pop("sqg_e4m3_luts_by_bits")
        tiles = (
            work_target.reshape(input_tiles, 16, output_tiles, 16)
            .permute(0, 2, 1, 3)
            .reshape(input_tiles * output_tiles, 256)
        )
        quantized, states = quantizer_module.quantize_tiles_multigpu(
            tiles[:, permutation].contiguous(), tile_args
        )
        if states.dtype != torch.int16 or tuple(states.shape) != (
            input_tiles * output_tiles,
            256,
        ):
            raise ValueError("direct SQG quantizer returned invalid states")
        work_reconstruction = (
            quantized[:, inverse_permutation]
            .reshape(input_tiles, output_tiles, 16, 16)
            .permute(0, 2, 1, 3)
            .reshape(input_dimension, output_dimension)
            .contiguous()
        )
        states = states.reshape(input_tiles, output_tiles, 256).contiguous()
        suh = suh.to(device=device, dtype=torch.float16).contiguous()
        svh = svh.to(device=device, dtype=torch.float16).contiguous()
        reconstruction = _decode_work_reconstruction(
            quantizer_module, work_reconstruction, suh, svh
        )
        torch.cuda.synchronize(device)
        results.append(
            UniformSQGDirectResult(
                candidate=UniformSQGCandidate(
                    reconstruction=reconstruction,
                    work_reconstruction=work_reconstruction,
                    states=states,
                    one_sided_sse=None,
                    two_sided_sse=None,
                    proxy_relative_error=None,
                    seconds=time.monotonic() - started,
                ),
                suh=suh,
                svh=svh,
                global_scale=1.0,
            )
        )
    return results


@torch.no_grad()
def encode_uniform_sqg_baseline(
    source: torch.Tensor,
    input_hessian: torch.Tensor,
    *,
    bits: int,
    device: torch.device,
    quantizer_module: Any,
    input_sign_seed: int,
    output_sign_seed: int,
    rate_axis: str,
    scale_scope_key: object | None = None,
    sigma_reg: float = SIGMA_REG,
    tailbite_context: int = 128,
    ldlq_tf32: bool = False,
) -> UniformSQGCandidate:
    """Encode one uniform SQG matrix through the conditioned batch path."""

    if source.ndim != 2 or not source.is_floating_point():
        raise TypeError("source must be a floating-point matrix")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in range(2, 7):
        raise ValueError("uniform SQG rate must be K2 through K6")
    if rate_axis not in ("k", "n"):
        raise ValueError("rate axis must be 'k' or 'n'")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("uniform SQG encoding requires CUDA")
    output_dimension, input_dimension = map(int, source.shape)
    _validate_hessian(input_hessian, input_dimension, "input Hessian")
    encoder_weight = source.T.float().contiguous().to(device)
    hessian_state = make_shared_h(input_dimension, device, input_hessian)
    quant_args = _uniform_quant_args(
        bits=bits,
        device=device,
        input_sign_seed=input_sign_seed,
        output_sign_seed=output_sign_seed,
        rate_axis=rate_axis,
        scale_scope_key=scale_scope_key,
        sigma_reg=sigma_reg,
        tailbite_context=tailbite_context,
        ldlq_tf32=ldlq_tf32,
    )
    if rate_axis is None:
        raise ValueError("uniform SQG encoding requires a rate axis")
    tile_dimension = encoder_weight.shape[1 if rate_axis == "n" else 0] // 16
    quant_args.update(
        {
            "mixed_rate_axis": rate_axis,
            "mixed_tile_bits": (bits,) * tile_dimension,
        }
    )
    torch.cuda.synchronize(device)
    started = time.monotonic()
    raw_groups = quantizer_module.quantize_qsrt_batch(
        [encoder_weight],
        [hessian_state],
        [[quant_args]],
        return_weight_q=True,
    )
    torch.cuda.synchronize(device)
    seconds = time.monotonic() - started
    if len(raw_groups) != 1 or len(raw_groups[0]) != 1:
        raise ValueError("uniform SQG batch encoder returned the wrong result count")
    raw = raw_groups[0][0]
    reconstruction = raw["weight_q"]
    states = raw["encoded"]
    if tuple(reconstruction.shape) != (input_dimension, output_dimension):
        raise ValueError("uniform SQG batch encoder returned the wrong reconstruction")
    expected_states = (input_dimension // 16, output_dimension // 16, 256)
    if states.dtype != torch.int16 or tuple(states.shape) != expected_states:
        raise ValueError("uniform SQG batch encoder returned invalid states")
    proxy = float(raw["proxy"])
    return UniformSQGCandidate(
        reconstruction=reconstruction.T.contiguous(),
        work_reconstruction=torch.empty(0, device=device),
        states=states,
        one_sided_sse=None,
        two_sided_sse=None,
        proxy_relative_error=proxy,
        seconds=seconds,
    )


@torch.no_grad()
def encode_uniform_sqg_two_sided_pair(
    source: torch.Tensor,
    input_hessian: torch.Tensor,
    output_hessian: torch.Tensor,
    *,
    bits: int,
    device: torch.device,
    quantizer_module: Any,
    input_sign_seed: int,
    output_sign_seed: int,
    rate_axis: str | None = None,
    scale_scope_key: object | None = None,
    sigma_reg: float = SIGMA_REG,
    tailbite_context: int = 128,
    ldlq_tf32: bool = False,
    output_damping_ratio: float = 1e-4,
    global_scale_override: float | None = None,
    work_dtype: torch.dtype = torch.float32,
    update_chunk_blocks: int = 16,
    h2_vcd_sweeps: int = 0,
    include_baseline: bool = True,
    gradient_guidance: ViterbiGradientGuidance | None = None,
) -> UniformSQGTwoSidedResult:
    """Encode one matrix with BlockLDLQ and two-sided block BaKron.

    Reconstruction scales are searched once under the ordinary SQG scale
    objective and then frozen for both candidates.  Router weights therefore
    belong in ``input_hessian`` or ``output_hessian``, but not both.
    """

    if source.ndim != 2 or not source.is_floating_point():
        raise TypeError("source must be a floating-point matrix")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in range(2, 7):
        raise ValueError("uniform SQG rate must be K2 through K6")
    if rate_axis not in ("k", "n"):
        raise ValueError("uniform SQG encoding requires a rate axis")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("uniform SQG encoding requires CUDA")
    if work_dtype == torch.float32 and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("two-sided FP32 rounding requires TF32 matmul disabled")
    output_dimension, input_dimension = map(int, source.shape)
    _validate_hessian(input_hessian, input_dimension, "input Hessian")
    _validate_hessian(output_hessian, output_dimension, "output Hessian")
    if not 1 <= tailbite_context <= 128:
        raise ValueError("tail-biting context must lie in 1..128")
    if (
        isinstance(h2_vcd_sweeps, bool)
        or not isinstance(h2_vcd_sweeps, int)
        or h2_vcd_sweeps < 0
    ):
        raise ValueError("H2-VCD sweep count must be a nonnegative integer")
    if h2_vcd_sweeps and not include_baseline:
        raise ValueError("H2-VCD requires the one-sided baseline candidate")

    preparation = _prepare_uniform_sqg_two_sided(
        source,
        input_hessian,
        output_hessian,
        bits=bits,
        device=device,
        quantizer_module=quantizer_module,
        input_sign_seed=input_sign_seed,
        output_sign_seed=output_sign_seed,
        rate_axis=rate_axis,
        scale_scope_key=scale_scope_key,
        sigma_reg=sigma_reg,
        tailbite_context=tailbite_context,
        ldlq_tf32=ldlq_tf32,
        output_damping_ratio=output_damping_ratio,
        global_scale_override=global_scale_override,
    )
    work_source = preparation.work_source
    input_hessian_work = preparation.input_hessian_work
    input_factor = preparation.input_factor
    output_hessian_work = preparation.output_hessian_work
    suh = preparation.suh
    svh = preparation.svh
    global_scale = preparation.global_scale
    quant_args = preparation.quant_args
    quantize_tiles = _prepared_tile_quantizer(preparation, quantizer_module)
    rounding_target = work_source
    work_gradient = None
    if gradient_guidance is not None:
        gradient_guidance.validate(source.shape)
        work_gradient = transform_source_gradient_to_work(
            quantizer_module,
            gradient_guidance.gradient,
            suh,
            svh,
        )
        rounding_target = shift_two_sided_target(
            work_source,
            work_gradient,
            input_hessian_work,
            output_hessian_work,
            strength=gradient_guidance.strength,
        ).to(dtype=work_source.dtype)

    baseline_work = None
    baseline_states = None
    baseline_seconds = 0.0
    if include_baseline:
        torch.cuda.synchronize(device)
        baseline_started = time.monotonic()
        baseline_work, baseline_states = quantizer_module.ldlq_mixed(
            work_source, input_factor, quant_args
        )
        torch.cuda.synchronize(device)
        baseline_seconds = time.monotonic() - baseline_started

    h2_vcd_work = None
    h2_vcd_states = None
    h2_vcd_receipt = None
    h2_vcd_seconds = 0.0
    if h2_vcd_sweeps:
        assert baseline_work is not None and baseline_states is not None
        refinement_args = dict(quant_args)
        refinement_args.update(
            {
                "h2_viterbi_refine_sweeps": h2_vcd_sweeps,
                "h2_viterbi_refine_dither_scales": (),
                "h2_viterbi_refine_patterns": 0,
            }
        )
        torch.cuda.synchronize(device)
        h2_vcd_started = time.monotonic()
        h2_vcd_work, h2_vcd_states, h2_vcd_receipt = (
            quantizer_module.refine_uniform_h2_candidate(
                work_source,
                input_hessian_work,
                baseline_work,
                baseline_states,
                refinement_args,
            )
        )
        torch.cuda.synchronize(device)
        h2_vcd_seconds = time.monotonic() - h2_vcd_started

    guided_one_sided_work = None
    guided_one_sided_states = None
    guided_one_sided_seconds = 0.0
    if gradient_guidance is not None:
        assert work_gradient is not None
        one_sided_target = shift_two_sided_target(
            work_source,
            work_gradient,
            input_hessian_work,
            torch.eye(output_dimension, device=device),
            strength=gradient_guidance.strength,
        ).to(dtype=work_source.dtype)
        torch.cuda.synchronize(device)
        guided_one_sided_started = time.monotonic()
        guided_one_sided_work, guided_one_sided_states = (
            quantizer_module.ldlq_mixed(
                one_sided_target,
                input_factor,
                quant_args,
            )
        )
        torch.cuda.synchronize(device)
        guided_one_sided_seconds = time.monotonic() - guided_one_sided_started

    torch.cuda.synchronize(device)
    two_sided_started = time.monotonic()
    rounded: TwoSidedRoundingResult = bakron_block_round_encoder(
        rounding_target,
        input_hessian_work,
        output_hessian_work,
        quantize_tiles,
        block_size=16,
        work_dtype=work_dtype,
        update_chunk_blocks=update_chunk_blocks,
    )
    torch.cuda.synchronize(device)
    two_sided_seconds = time.monotonic() - two_sided_started

    baseline_one_sided = None
    baseline_two_sided = None
    candidate_one_sided = None
    if include_baseline:
        assert baseline_work is not None and baseline_states is not None
        identity_output_hessian = torch.eye(output_dimension, device=device)
        baseline_one_sided, _ = two_sided_encoder_sse(
            work_source,
            baseline_work,
            input_hessian_work,
            identity_output_hessian,
        )
        baseline_two_sided, _ = two_sided_encoder_sse(
            work_source,
            baseline_work,
            input_hessian_work,
            output_hessian_work,
        )
        candidate_one_sided, _ = two_sided_encoder_sse(
            work_source,
            rounded.reconstruction,
            input_hessian_work,
            identity_output_hessian,
        )
    candidate_two_sided, _ = two_sided_encoder_sse(
        work_source,
        rounded.reconstruction,
        input_hessian_work,
        output_hessian_work,
    )
    candidate_reconstruction = _decode_work_reconstruction(
        quantizer_module, rounded.reconstruction, suh, svh
    )
    gradient_linear_term = None
    guided_objective = None
    if gradient_guidance is not None:
        gradient = gradient_guidance.gradient.to(
            device=candidate_reconstruction.device,
            dtype=torch.float32,
        )
        anchor = gradient_guidance.anchor.to(
            device=candidate_reconstruction.device,
            dtype=torch.float32,
        )
        gradient_linear_term = float(
            gradient_guidance.strength
            * torch.sum(gradient * (candidate_reconstruction.float() - anchor))
        )
        guided_objective = candidate_two_sided + gradient_linear_term

    h2_vcd = None
    if h2_vcd_work is not None and h2_vcd_states is not None:
        h2_vcd_one_sided, _ = two_sided_encoder_sse(
            work_source,
            h2_vcd_work,
            input_hessian_work,
            torch.eye(output_dimension, device=device),
        )
        h2_vcd_two_sided, _ = two_sided_encoder_sse(
            work_source,
            h2_vcd_work,
            input_hessian_work,
            output_hessian_work,
        )
        h2_vcd = UniformSQGCandidate(
            reconstruction=_decode_work_reconstruction(
                quantizer_module, h2_vcd_work, suh, svh
            ),
            work_reconstruction=h2_vcd_work,
            states=h2_vcd_states,
            one_sided_sse=h2_vcd_one_sided,
            two_sided_sse=h2_vcd_two_sided,
            proxy_relative_error=None,
            seconds=h2_vcd_seconds,
            refinement=h2_vcd_receipt,
        )

    baseline = None
    if baseline_work is not None and baseline_states is not None:
        baseline = UniformSQGCandidate(
            reconstruction=_decode_work_reconstruction(
                quantizer_module, baseline_work, suh, svh
            ),
            work_reconstruction=baseline_work,
            states=baseline_states,
            one_sided_sse=baseline_one_sided,
            two_sided_sse=baseline_two_sided,
            proxy_relative_error=None,
            seconds=baseline_seconds,
        )

    guided_one_sided = None
    if guided_one_sided_work is not None and guided_one_sided_states is not None:
        assert gradient_guidance is not None
        guided_reconstruction = _decode_work_reconstruction(
            quantizer_module,
            guided_one_sided_work,
            suh,
            svh,
        )
        guided_one_sided_sse, _ = two_sided_encoder_sse(
            work_source,
            guided_one_sided_work,
            input_hessian_work,
            torch.eye(output_dimension, device=device),
        )
        guided_two_sided_sse, _ = two_sided_encoder_sse(
            work_source,
            guided_one_sided_work,
            input_hessian_work,
            output_hessian_work,
        )
        guided_linear = float(
            gradient_guidance.strength
            * torch.sum(
                gradient_guidance.gradient.to(device=device, dtype=torch.float32)
                * (
                    guided_reconstruction.float()
                    - gradient_guidance.anchor.to(
                        device=device,
                        dtype=torch.float32,
                    )
                )
            )
        )
        guided_one_sided = UniformSQGCandidate(
            reconstruction=guided_reconstruction,
            work_reconstruction=guided_one_sided_work,
            states=guided_one_sided_states,
            one_sided_sse=guided_one_sided_sse,
            two_sided_sse=guided_two_sided_sse,
            proxy_relative_error=None,
            seconds=guided_one_sided_seconds,
            gradient_linear_term=guided_linear,
            guided_objective=guided_one_sided_sse + guided_linear,
        )

    return UniformSQGTwoSidedResult(
        baseline=baseline,
        two_sided=UniformSQGCandidate(
            reconstruction=candidate_reconstruction,
            work_reconstruction=rounded.reconstruction,
            states=rounded.payload,
            one_sided_sse=candidate_one_sided,
            two_sided_sse=candidate_two_sided,
            proxy_relative_error=None,
            seconds=two_sided_seconds,
            gradient_linear_term=gradient_linear_term,
            guided_objective=guided_objective,
        ),
        input_hessian_work=input_hessian_work,
        output_hessian_work=output_hessian_work,
        suh=suh,
        svh=svh,
        global_scale=float(global_scale),
        h2_vcd=h2_vcd,
        guided_one_sided=guided_one_sided,
    )


@torch.no_grad()
def encode_uniform_sqg_two_sided_batch(
    sources: torch.Tensor,
    input_hessians: torch.Tensor,
    output_hessians: torch.Tensor,
    *,
    bits: int,
    device: torch.device,
    quantizer_module: Any,
    input_sign_seed: int,
    output_sign_seed: int,
    rate_axis: str,
    sigma_reg: float = SIGMA_REG,
    tailbite_context: int = 128,
    ldlq_tf32: bool = False,
    output_damping_ratio: float = 1e-4,
    global_scale_overrides: tuple[float | None, ...] | None = None,
    compute_objective: bool = False,
    gradient_guidance: Sequence[ViterbiGradientGuidance | None] | None = None,
) -> list[UniformSQGTwoSidedResult]:
    """Encode equal-shaped matrices with one lockstep BaKron traversal.

    Each matrix retains independent conditioning, scale search, Hessian
    factors, recurrence state, and payload.  Batching only combines the legal
    tile quantizer calls at each dependency-ordered anti-diagonal.
    """

    if sources.ndim != 3 or not sources.shape[0] or not sources.is_floating_point():
        raise ValueError("sources must be a nonempty floating-point matrix batch")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in range(2, 7):
        raise ValueError("uniform SQG rate must be K2 through K6")
    if rate_axis not in ("k", "n"):
        raise ValueError("uniform SQG encoding requires a rate axis")
    if device.type != "cuda" or sources.device != device:
        raise ValueError("batched uniform SQG encoding requires one CUDA device")
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("two-sided FP32 rounding requires TF32 matmul disabled")
    if not 1 <= tailbite_context <= 128:
        raise ValueError("tail-biting context must lie in 1..128")
    batch, output_dimension, input_dimension = map(int, sources.shape)
    if tuple(input_hessians.shape) != (batch, input_dimension, input_dimension):
        raise ValueError("input Hessians have the wrong shape")
    if tuple(output_hessians.shape) != (batch, output_dimension, output_dimension):
        raise ValueError("output Hessians have the wrong shape")
    if input_hessians.device != device or output_hessians.device != device:
        raise ValueError("sources and Hessians must share one CUDA device")
    if global_scale_overrides is None:
        global_scale_overrides = (None,) * batch
    elif len(global_scale_overrides) != batch:
        raise ValueError("global scale overrides must match the batch size")
    if gradient_guidance is None:
        gradient_guidance = (None,) * batch
    elif len(gradient_guidance) != batch:
        raise ValueError("gradient guidance must match the batch size")

    preparations: list[_UniformSQGPreparation] = []
    for index in range(batch):
        _validate_hessian(
            input_hessians[index], input_dimension, "input Hessian"
        )
        _validate_hessian(
            output_hessians[index], output_dimension, "output Hessian"
        )
        preparations.append(
            _prepare_uniform_sqg_two_sided(
                sources[index],
                input_hessians[index],
                output_hessians[index],
                bits=bits,
                device=device,
                quantizer_module=quantizer_module,
                input_sign_seed=input_sign_seed,
                output_sign_seed=output_sign_seed,
                rate_axis=rate_axis,
                scale_scope_key=None,
                sigma_reg=sigma_reg,
                tailbite_context=tailbite_context,
                ldlq_tf32=ldlq_tf32,
                output_damping_ratio=output_damping_ratio,
                global_scale_override=global_scale_overrides[index],
            )
        )

    rounding_targets: list[torch.Tensor] = []
    for index, (preparation, guidance) in enumerate(
        zip(preparations, gradient_guidance, strict=True)
    ):
        if guidance is None:
            rounding_targets.append(preparation.work_source)
            continue
        guidance.validate(sources[index].shape)
        work_gradient = transform_source_gradient_to_work(
            quantizer_module,
            guidance.gradient,
            preparation.suh,
            preparation.svh,
        )
        rounding_targets.append(
            shift_two_sided_target(
                preparation.work_source,
                work_gradient,
                preparation.input_hessian_work,
                preparation.output_hessian_work,
                strength=guidance.strength,
            ).to(dtype=preparation.work_source.dtype)
        )

    quantize_tiles = _prepared_tile_quantizer(preparations[0], quantizer_module)
    torch.cuda.synchronize(device)
    started = time.monotonic()
    rounded = bakron_block_round_encoder_batch(
        torch.stack(rounding_targets),
        torch.stack([item.input_hessian_work for item in preparations]),
        torch.stack([item.output_hessian_work for item in preparations]),
        quantize_tiles,
        block_size=16,
        work_dtype=torch.float32,
    )
    torch.cuda.synchronize(device)
    seconds_per_matrix = (time.monotonic() - started) / batch

    results: list[UniformSQGTwoSidedResult] = []
    for index, preparation in enumerate(preparations):
        objective = None
        guidance = gradient_guidance[index]
        if compute_objective:
            objective, _ = two_sided_encoder_sse(
                preparation.work_source,
                rounded.reconstruction[index],
                preparation.input_hessian_work,
                preparation.output_hessian_work,
            )
        reconstruction = _decode_work_reconstruction(
            quantizer_module,
            rounded.reconstruction[index],
            preparation.suh,
            preparation.svh,
        )
        gradient_linear_term = None
        guided_objective = None
        if guidance is not None:
            gradient_linear_term = float(
                guidance.strength
                * torch.sum(
                    guidance.gradient.to(device=device, dtype=torch.float32)
                    * (
                        reconstruction.float()
                        - guidance.anchor.to(device=device, dtype=torch.float32)
                    )
                )
            )
            if objective is not None:
                guided_objective = objective + gradient_linear_term
        results.append(
            UniformSQGTwoSidedResult(
                baseline=None,
                two_sided=UniformSQGCandidate(
                    reconstruction=reconstruction,
                    work_reconstruction=rounded.reconstruction[index],
                    states=rounded.payload[index],
                    one_sided_sse=None,
                    two_sided_sse=objective,
                    proxy_relative_error=None,
                    seconds=seconds_per_matrix,
                    gradient_linear_term=gradient_linear_term,
                    guided_objective=guided_objective,
                ),
                input_hessian_work=preparation.input_hessian_work,
                output_hessian_work=preparation.output_hessian_work,
                suh=preparation.suh,
                svh=preparation.svh,
                global_scale=preparation.global_scale,
            )
        )
    return results


@torch.no_grad()
def encode_uniform_sqg_two_sided_output_blocks_batch(
    sources: torch.Tensor,
    input_hessian: torch.Tensor | None,
    output_hessian_blocks: torch.Tensor,
    *,
    bits: int,
    device: torch.device,
    quantizer_module: Any,
    input_sign_seed: int,
    output_sign_seed: int | Sequence[int],
    rate_axis: str,
    scale_scope_key: object,
    shared_input: UniformSQGSharedInputConditioning | None = None,
    output_factor_block_size: int = 128,
    sigma_reg: float = SIGMA_REG,
    tailbite_context: int = 128,
    ldlq_tf32: bool = False,
    output_damping_ratio: float = 1e-4,
    global_scale_overrides: tuple[float | None, ...] | None = None,
    compute_objective: bool = False,
    gradient_guidance: Sequence[ViterbiGradientGuidance | None] | None = None,
) -> tuple[
    list[UniformSQGBlockTwoSidedResult],
    UniformSQGSharedInputConditioning,
]:
    """Encode matrices under an exact block-diagonal output Fisher metric.

    Each output-factor block becomes an independent matrix stripe.  All
    stripes share the complete input Hessian and its BaKron factor, so the
    implementation preserves every within-block Fisher interaction without
    materializing a dense output Hessian or duplicating the input factor.
    """

    if sources.ndim != 3 or not sources.shape[0] or not sources.is_floating_point():
        raise ValueError("sources must be a nonempty floating-point matrix batch")
    if sources.device != device or device.type != "cuda":
        raise ValueError("block two-sided SQG encoding requires one CUDA device")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in range(2, 7):
        raise ValueError("uniform SQG rate must be K2 through K6")
    if rate_axis not in ("k", "n"):
        raise ValueError("uniform SQG encoding requires a rate axis")
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("two-sided FP32 rounding requires TF32 matmul disabled")
    if output_factor_block_size != 128:
        raise ValueError("coupled output Fisher blocks must match the H128 basis")
    if not 1 <= tailbite_context <= 128:
        raise ValueError("tail-biting context must lie in 1..128")

    batch, output_dimension, input_dimension = map(int, sources.shape)
    if isinstance(output_sign_seed, int) and not isinstance(output_sign_seed, bool):
        if output_sign_seed < 0:
            raise ValueError("output sign seeds must be nonnegative")
        output_sign_seeds = (output_sign_seed,) * batch
    else:
        try:
            output_sign_seeds = tuple(output_sign_seed)
        except TypeError as error:
            raise TypeError("output sign seeds must be an integer or sequence") from error
        if len(output_sign_seeds) != batch or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in output_sign_seeds
        ):
            raise ValueError("output sign seeds must match the matrix batch")
    if output_dimension % output_factor_block_size:
        raise ValueError("output factor block size must divide the output dimension")
    output_blocks = output_dimension // output_factor_block_size
    expected_output_shape = (
        batch,
        output_blocks,
        output_factor_block_size,
        output_factor_block_size,
    )
    if tuple(output_hessian_blocks.shape) != expected_output_shape:
        raise ValueError("output Hessian blocks have the wrong shape")
    if not bool(torch.all(torch.isfinite(output_hessian_blocks))):
        raise ValueError("output Hessian blocks must be finite")
    if shared_input is None:
        if input_hessian is None:
            raise ValueError("input Hessian is required for initial conditioning")
        _validate_hessian(input_hessian, input_dimension, "input Hessian")
    if global_scale_overrides is None:
        global_scale_overrides = (None,) * batch
    elif len(global_scale_overrides) != batch:
        raise ValueError("global scale overrides must match the batch size")
    if gradient_guidance is None:
        gradient_guidance = (None,) * batch
    elif len(gradient_guidance) != batch:
        raise ValueError("gradient guidance must match the batch size")

    preparations: list[_UniformSQGPreparation] = []
    conditioning = shared_input
    for index in range(batch):
        preparation = _prepare_uniform_sqg_two_sided(
            sources[index],
            input_hessian if conditioning is None else None,
            None,
            bits=bits,
            device=device,
            quantizer_module=quantizer_module,
            input_sign_seed=input_sign_seed,
            output_sign_seed=output_sign_seeds[index],
            rate_axis=rate_axis,
            scale_scope_key=scale_scope_key,
            sigma_reg=sigma_reg,
            tailbite_context=tailbite_context,
            ldlq_tf32=ldlq_tf32,
            output_damping_ratio=output_damping_ratio,
            global_scale_override=global_scale_overrides[index],
            g_scale_into_sv=True,
            shared_input=conditioning,
        )
        if conditioning is None:
            conditioning = _shared_input_conditioning(
                preparation,
                input_dimension=input_dimension,
                bits=bits,
                input_sign_seed=input_sign_seed,
                rate_axis=rate_axis,
                scale_scope_key=scale_scope_key,
            )
        preparations.append(preparation)
    assert conditioning is not None

    blocks_device = output_hessian_blocks.to(
        device=device,
        dtype=torch.float32,
    )
    transformed_blocks: list[torch.Tensor] = []
    for index, preparation in enumerate(preparations):
        damped, _ = _damp_output_hessian_blocks(
            blocks_device[index],
            output_damping_ratio,
        )
        transformed_blocks.append(
            transform_output_hessian_blocks_for_regularization(
                damped,
                preparation.svh.float(),
                block_size=output_factor_block_size,
            )
        )
    output_hessian_blocks_work = torch.stack(transformed_blocks)

    work_gradients: list[torch.Tensor | None] = []
    for index, (preparation, guidance) in enumerate(
        zip(preparations, gradient_guidance, strict=True)
    ):
        if guidance is None:
            work_gradients.append(None)
            continue
        guidance.validate(sources[index].shape)
        work_gradients.append(
            transform_source_gradient_to_work(
                quantizer_module,
                guidance.gradient,
                preparation.suh,
                preparation.svh,
            )
        )

    output_factor_pairs = [
        factor_bakron_hessian(
            block,
            block_size=16,
            dtype=torch.float32,
        )
        for block in output_hessian_blocks_work.flatten(0, 1)
    ]
    output_factors = torch.stack([pair[0] for pair in output_factor_pairs])
    output_reverse = output_factor_pairs[0][1]

    stripe_sources = (
        torch.stack([item.work_source for item in preparations])
        .reshape(
            batch,
            input_dimension,
            output_blocks,
            output_factor_block_size,
        )
        .permute(0, 2, 1, 3)
        .reshape(
            batch * output_blocks,
            input_dimension,
            output_factor_block_size,
        )
        .contiguous()
    )
    stripe_targets = stripe_sources.clone()
    for index, guidance in enumerate(gradient_guidance):
        if guidance is None:
            continue
        work_gradient = work_gradients[index]
        assert work_gradient is not None
        gradient_stripes = (
            work_gradient.reshape(
                input_dimension,
                output_blocks,
                output_factor_block_size,
            )
            .contiguous()
        )
        for block_index in range(output_blocks):
            stripe_index = index * output_blocks + block_index
            stripe_targets[stripe_index].copy_(
                shift_two_sided_target(
                    stripe_sources[stripe_index],
                    gradient_stripes[:, block_index, :],
                    preparations[index].input_hessian_work,
                    output_hessian_blocks_work[index, block_index],
                    strength=guidance.strength,
                ).to(dtype=stripe_sources.dtype)
            )
    quantize_tiles = _prepared_tile_quantizer(preparations[0], quantizer_module)
    torch.cuda.synchronize(device)
    started = time.monotonic()
    rounded = bakron_block_round_encoder_prepared_batch(
        stripe_targets,
        conditioning.bakron_input_factor,
        output_factors,
        conditioning.bakron_input_reverse,
        output_reverse,
        quantize_tiles,
        block_size=16,
        work_dtype=torch.float32,
    )
    torch.cuda.synchronize(device)
    seconds_per_matrix = (time.monotonic() - started) / batch

    work_reconstructions = (
        rounded.reconstruction.reshape(
            batch,
            output_blocks,
            input_dimension,
            output_factor_block_size,
        )
        .permute(0, 2, 1, 3)
        .reshape(batch, input_dimension, output_dimension)
        .contiguous()
    )
    payload_shape = tuple(rounded.payload.shape[3:])
    states = (
        rounded.payload.reshape(
            batch,
            output_blocks,
            input_dimension // 16,
            output_factor_block_size // 16,
            *payload_shape,
        )
        .permute(0, 2, 1, 3, *range(4, 4 + len(payload_shape)))
        .reshape(
            batch,
            input_dimension // 16,
            output_dimension // 16,
            *payload_shape,
        )
        .contiguous()
    )

    results: list[UniformSQGBlockTwoSidedResult] = []
    for index, preparation in enumerate(preparations):
        objective = None
        guidance = gradient_guidance[index]
        if compute_objective or guidance is not None:
            objective = 0.0
            for block_index in range(output_blocks):
                begin = block_index * output_factor_block_size
                end = begin + output_factor_block_size
                value, _ = two_sided_encoder_sse(
                    preparation.work_source[:, begin:end],
                    work_reconstructions[index, :, begin:end],
                    preparation.input_hessian_work,
                    output_hessian_blocks_work[index, block_index],
                )
                objective += value
        reconstruction = _decode_work_reconstruction(
            quantizer_module,
            work_reconstructions[index],
            preparation.suh,
            preparation.svh,
        )
        gradient_linear_term = None
        guided_objective = None
        if guidance is not None:
            assert objective is not None
            gradient_linear_term = float(
                guidance.strength
                * torch.sum(
                    guidance.gradient.to(device=device, dtype=torch.float32)
                    * (
                        reconstruction.float()
                        - guidance.anchor.to(device=device, dtype=torch.float32)
                    )
                )
            )
            guided_objective = objective + gradient_linear_term
        results.append(
            UniformSQGBlockTwoSidedResult(
                two_sided=UniformSQGCandidate(
                    reconstruction=reconstruction,
                    work_reconstruction=work_reconstructions[index],
                    states=states[index],
                    one_sided_sse=None,
                    two_sided_sse=objective,
                    proxy_relative_error=None,
                    seconds=seconds_per_matrix,
                    gradient_linear_term=gradient_linear_term,
                    guided_objective=guided_objective,
                ),
                input_hessian_work=preparation.input_hessian_work,
                output_hessian_blocks_work=output_hessian_blocks_work[index],
                suh=preparation.suh,
                svh=preparation.svh,
                global_scale=preparation.global_scale,
            )
        )
    return results, conditioning


__all__ = [
    "UniformSQGBlockTwoSidedResult",
    "UniformSQGCandidate",
    "UniformSQGDirectResult",
    "UniformSQGSharedInputConditioning",
    "UniformSQGTwoSidedResult",
    "encode_uniform_sqg_baseline",
    "encode_uniform_sqg_direct_batch",
    "encode_uniform_sqg_direct_work_batch",
    "encode_uniform_sqg_two_sided_output_blocks_batch",
    "encode_uniform_sqg_two_sided_batch",
    "encode_uniform_sqg_two_sided_pair",
    "transform_source_gradient_to_work",
]

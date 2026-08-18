from __future__ import annotations

import torch

from qsrt.gradient_guided_viterbi import (
    ViterbiGradientGuidance,
    shift_elementwise_target,
    shift_two_sided_target,
    two_sided_guided_objective,
)
from qsrt.qsrt_coupled import block_hadamard
from qsrt.two_sided_qsrt import (
    transform_source_gradient_factors_to_work,
    transform_source_gradient_to_work,
)


def _spd(dimension: int, generator: torch.Generator) -> torch.Tensor:
    root = torch.randn((dimension + 2, dimension), generator=generator)
    return root.T @ root + torch.eye(dimension) * 0.25


def test_elementwise_target_shift_preserves_candidate_order() -> None:
    generator = torch.Generator().manual_seed(414)
    target = torch.randn((5, 7), generator=generator)
    gradient = torch.randn((5, 7), generator=generator)
    weights = torch.rand((5, 7), generator=generator) + 0.2
    shifted = shift_elementwise_target(
        target,
        gradient,
        strength=0.37,
        quadratic_weight=weights,
    )

    offsets = []
    for _ in range(4):
        candidate = torch.randn((5, 7), generator=generator)
        original = torch.sum(weights * (candidate - target).square())
        original += 0.37 * torch.sum(gradient * candidate)
        shifted_quadratic = torch.sum(weights * (candidate - shifted).square())
        offsets.append(original - shifted_quadratic)
    torch.testing.assert_close(
        torch.stack(offsets),
        offsets[0].expand(4),
        rtol=1e-5,
        atol=1e-6,
    )


def test_two_sided_target_shift_preserves_candidate_order() -> None:
    generator = torch.Generator().manual_seed(991)
    target = torch.randn((4, 6), generator=generator, dtype=torch.float64)
    gradient = torch.randn((4, 6), generator=generator, dtype=torch.float64)
    anchor = torch.randn((4, 6), generator=generator, dtype=torch.float64)
    row_hessian = _spd(4, generator).double()
    column_hessian = _spd(6, generator).double()
    shifted = shift_two_sided_target(
        target,
        gradient,
        row_hessian,
        column_hessian,
        strength=0.23,
    )

    offsets = []
    for _ in range(4):
        candidate = torch.randn((4, 6), generator=generator, dtype=torch.float64)
        _, _, original = two_sided_guided_objective(
            target,
            candidate,
            anchor,
            gradient,
            row_hessian,
            column_hessian,
            strength=0.23,
        )
        error = candidate - shifted
        shifted_quadratic = torch.sum(
            error * (row_hessian @ error @ column_hessian)
        )
        offsets.append(original - shifted_quadratic)
    torch.testing.assert_close(
        torch.stack(offsets),
        offsets[0].expand(4),
        rtol=1e-5,
        atol=1e-6,
    )


def test_gradient_guidance_requires_explicit_anchor_identity() -> None:
    guidance = ViterbiGradientGuidance(
        gradient=torch.ones((2, 3)),
        anchor=torch.zeros((2, 3)),
        anchor_id="uniform-k2-anchor",
        objective_id="teacher-kl",
        strength=0.5,
    )
    guidance.validate((2, 3))


class _HadamardBackend:
    @staticmethod
    def preapply_had_l(values: torch.Tensor, block_size: int) -> torch.Tensor:
        return block_hadamard(values, block_size=block_size, dim=0)

    @staticmethod
    def preapply_had_r(values: torch.Tensor, block_size: int) -> torch.Tensor:
        return block_hadamard(values, block_size=block_size, dim=1)


def test_source_gradient_adjoint_closes_work_reconstruction() -> None:
    generator = torch.Generator().manual_seed(820)
    work_displacement = torch.randn((128, 128), generator=generator)
    source_gradient = torch.randn((128, 128), generator=generator)
    suh = torch.rand((128,), generator=generator) + 0.25
    svh = torch.rand((128,), generator=generator) + 0.25
    backend = _HadamardBackend()

    decoded = backend.preapply_had_l(work_displacement, 128)
    decoded *= suh[:, None]
    decoded = backend.preapply_had_r(decoded, 128)
    decoded *= svh[None, :]
    decoded = decoded.T.contiguous()
    work_gradient = transform_source_gradient_to_work(
        backend,
        source_gradient,
        suh,
        svh,
    )
    source_derivative = torch.sum(source_gradient * decoded)
    work_derivative = torch.sum(work_gradient * work_displacement)
    torch.testing.assert_close(
        source_derivative,
        work_derivative,
        rtol=2e-5,
        atol=2e-5,
    )


def test_factorized_source_gradient_transform_matches_dense_transform() -> None:
    generator = torch.Generator().manual_seed(821)
    source_left = torch.randn((128, 7), generator=generator)
    source_right = torch.randn((7, 128), generator=generator)
    suh = torch.rand((128,), generator=generator) + 0.25
    svh = torch.rand((128,), generator=generator) + 0.25
    backend = _HadamardBackend()
    dense = transform_source_gradient_to_work(
        backend,
        source_left @ source_right,
        suh,
        svh,
    )
    left, right = transform_source_gradient_factors_to_work(
        backend,
        source_left,
        source_right,
        suh,
        svh,
    )
    torch.testing.assert_close(left @ right, dense, rtol=3e-5, atol=3e-5)

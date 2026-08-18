from __future__ import annotations

import torch

from qsrt.qsrt_coupled import block_hadamard
from qsrt.two_sided_rounding import (
    bakron_block_antidiagonal_encoder,
    bakron_block_round_encoder,
    bakron_block_round_encoder_batch,
    bakron_block_round_encoder_prepared_batch,
    factor_bakron_hessian,
    transform_output_hessian_for_regularization,
    transform_output_hessian_blocks_for_regularization,
    yaqa_block_round_encoder,
)


def _positive_definite(dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    root = torch.randn((3 * dimension, dimension), generator=generator)
    return root.T @ root + 0.3 * torch.eye(dimension)


def _round_tiles(tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.round(tiles * 4).to(torch.int16)
    return payload.to(tiles.dtype) / 4, payload


def _one_sided_reference(
    weight: torch.Tensor,
    input_hessian: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dimension = weight.shape[0]
    cholesky = torch.linalg.cholesky(input_hessian.double())
    factor = cholesky.clone()
    for begin in range(0, dimension, block_size):
        end = begin + block_size
        diagonal = cholesky[begin:end, begin:end]
        factor[:, begin:end] = torch.linalg.solve_triangular(
            diagonal.T, cholesky[:, begin:end].T, upper=True
        ).T
        factor[begin:end, begin:end].zero_()

    quantized = torch.zeros_like(weight, dtype=torch.float64)
    payload = torch.empty_like(weight, dtype=torch.int16)
    output_blocks = weight.shape[1] // block_size
    for end in range(dimension, 0, -block_size):
        begin = end - block_size
        delta = weight.double() - quantized
        target = weight[begin:end].double() + factor[begin:, begin:end].T @ delta[begin:]
        target_tiles = target.reshape(block_size, output_blocks, block_size).permute(1, 0, 2)
        reconstruction, indices = _round_tiles(target_tiles)
        quantized[begin:end] = reconstruction.permute(1, 0, 2).reshape(block_size, -1)
        payload[begin:end] = indices.permute(1, 0, 2).reshape(block_size, -1)
    return quantized.to(weight.dtype), payload


def test_identity_factors_reduce_to_independent_tile_quantization() -> None:
    generator = torch.Generator().manual_seed(1)
    weight = torch.randn((8, 12), generator=generator)
    result = yaqa_block_round_encoder(
        weight,
        torch.eye(8),
        torch.eye(12),
        _round_tiles,
        block_size=4,
    )
    expected_payload = torch.round(weight * 4).to(torch.int16)
    expected = expected_payload.float() / 4

    torch.testing.assert_close(result.reconstruction, expected, rtol=0, atol=0)
    reconstructed_payload = result.payload.permute(0, 2, 1, 3).reshape_as(weight)
    torch.testing.assert_close(reconstructed_payload, expected_payload, rtol=0, atol=0)


def test_identity_output_factor_matches_one_sided_blockldlq() -> None:
    generator = torch.Generator().manual_seed(2)
    weight = torch.randn((12, 8), generator=generator)
    input_hessian = _positive_definite(12, seed=3)
    expected, expected_payload = _one_sided_reference(weight, input_hessian, 4)
    result = yaqa_block_round_encoder(
        weight,
        input_hessian,
        torch.eye(8),
        _round_tiles,
        block_size=4,
    )

    torch.testing.assert_close(result.reconstruction, expected, rtol=0, atol=0)
    reconstructed_payload = result.payload.permute(0, 2, 1, 3).reshape_as(weight)
    torch.testing.assert_close(reconstructed_payload, expected_payload, rtol=0, atol=0)


def test_two_sided_rounding_accepts_non_scalar_tile_payload() -> None:
    generator = torch.Generator().manual_seed(4)
    weight = torch.randn((8, 8), generator=generator)

    def quantizer(tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reconstruction = torch.where(tiles.mean(dim=(1, 2))[:, None, None] >= 0, 0.5, -0.5)
        reconstruction = reconstruction.expand_as(tiles).contiguous()
        payload = torch.stack(
            (reconstruction[:, 0, 0] > 0, tiles.square().mean(dim=(1, 2)) > 1),
            dim=1,
        ).to(torch.uint8)
        return reconstruction, payload

    result = yaqa_block_round_encoder(
        weight,
        _positive_definite(8, seed=5),
        _positive_definite(8, seed=6),
        quantizer,
        block_size=4,
    )

    assert result.reconstruction.shape == weight.shape
    assert result.payload.shape == (2, 2, 2)
    assert set(result.reconstruction.unique().tolist()) <= {-0.5, 0.5}


def test_block_bakron_antidiagonal_matches_yaqa_reference() -> None:
    for block_size, shape in ((1, (4, 4)), (2, (8, 12)), (4, (12, 8))):
        generator = torch.Generator().manual_seed(1000 + block_size)
        weight = torch.randn(shape, generator=generator)
        input_hessian = _positive_definite(shape[0], seed=2000 + block_size)
        output_hessian = _positive_definite(shape[1], seed=3000 + block_size)
        expected = yaqa_block_round_encoder(
            weight,
            input_hessian,
            output_hessian,
            _round_tiles,
            block_size=block_size,
        )
        actual = bakron_block_antidiagonal_encoder(
            weight,
            input_hessian,
            output_hessian,
            _round_tiles,
            block_size=block_size,
        )

        torch.testing.assert_close(actual.reconstruction, expected.reconstruction, rtol=0, atol=0)
        torch.testing.assert_close(actual.payload, expected.payload, rtol=0, atol=0)


def test_recursive_block_bakron_matches_antidiagonal_reference() -> None:
    for block_size, shape in ((1, (4, 4)), (2, (8, 12)), (4, (12, 8))):
        generator = torch.Generator().manual_seed(4000 + block_size)
        weight = torch.randn(shape, generator=generator)
        input_hessian = _positive_definite(shape[0], seed=5000 + block_size)
        output_hessian = _positive_definite(shape[1], seed=6000 + block_size)
        expected = bakron_block_antidiagonal_encoder(
            weight,
            input_hessian,
            output_hessian,
            _round_tiles,
            block_size=block_size,
        )
        actual = bakron_block_round_encoder(
            weight,
            input_hessian,
            output_hessian,
            _round_tiles,
            block_size=block_size,
        )

        torch.testing.assert_close(actual.reconstruction, expected.reconstruction, rtol=0, atol=0)
        torch.testing.assert_close(actual.payload, expected.payload, rtol=0, atol=0)


def test_recursive_block_bakron_chunked_updates_match_reference() -> None:
    block_size = 4
    shape = (20, 28)
    generator = torch.Generator().manual_seed(6500)
    weight = torch.randn(shape, generator=generator)
    input_hessian = _positive_definite(shape[0], seed=6501)
    output_hessian = _positive_definite(shape[1], seed=6502)
    expected = bakron_block_antidiagonal_encoder(
        weight,
        input_hessian,
        output_hessian,
        _round_tiles,
        block_size=block_size,
    )
    actual = bakron_block_round_encoder(
        weight,
        input_hessian,
        output_hessian,
        _round_tiles,
        block_size=block_size,
        update_chunk_blocks=2,
    )

    torch.testing.assert_close(actual.reconstruction, expected.reconstruction, rtol=0, atol=0)
    torch.testing.assert_close(actual.payload, expected.payload, rtol=0, atol=0)


def test_output_hessian_regularization_preserves_quadratic_form() -> None:
    generator = torch.Generator().manual_seed(7001)
    work_error = torch.randn((12, 8), generator=generator)
    output_root = torch.randn((13, 8), generator=generator)
    output_hessian = output_root.T @ output_root + 0.2 * torch.eye(8)
    output_scales = torch.rand(8, generator=generator) + 0.5
    regularized_hessian = transform_output_hessian_for_regularization(
        output_hessian,
        output_scales,
        block_size=4,
    )
    decoded_error = block_hadamard(work_error, block_size=4, dim=1)
    decoded_error = decoded_error * output_scales
    expected = torch.sum(decoded_error * (decoded_error @ output_hessian))
    actual = torch.sum(work_error * (work_error @ regularized_hessian))

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_block_output_hessian_regularization_matches_dense_block_diagonal() -> None:
    generator = torch.Generator().manual_seed(7002)
    roots = torch.randn((3, 8, 8), generator=generator)
    blocks = roots.transpose(1, 2) @ roots + 0.2 * torch.eye(8)
    scales = torch.rand(24, generator=generator) + 0.5
    actual = transform_output_hessian_blocks_for_regularization(
        blocks,
        scales,
        block_size=8,
    )
    expected = transform_output_hessian_for_regularization(
        torch.block_diag(*blocks),
        scales,
        block_size=8,
    ).reshape(3, 8, 3, 8)

    torch.testing.assert_close(
        actual,
        expected[torch.arange(3), :, torch.arange(3)],
        rtol=0,
        atol=0,
    )
    off_diagonal = expected.clone()
    off_diagonal[torch.arange(3), :, torch.arange(3)] = 0
    assert torch.count_nonzero(off_diagonal) == 0


def test_cuda_batched_bakron_matches_independent_encodes() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:0")
    generator = torch.Generator().manual_seed(8100)
    weights = torch.randn((2, 32, 48), generator=generator).to(device)
    input_hessians = torch.stack(
        (_positive_definite(32, seed=8101), _positive_definite(32, seed=8102))
    ).to(device)
    output_hessians = torch.stack(
        (_positive_definite(48, seed=8103), _positive_definite(48, seed=8104))
    ).to(device)

    batched = bakron_block_round_encoder_batch(
        weights,
        input_hessians,
        output_hessians,
        _round_tiles,
    )
    for index in range(2):
        independent = bakron_block_round_encoder(
            weights[index],
            input_hessians[index],
            output_hessians[index],
            _round_tiles,
            block_size=16,
            work_dtype=torch.float32,
            update_backend="cuda",
        )
        torch.testing.assert_close(
            batched.reconstruction[index],
            independent.reconstruction,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            batched.payload[index], independent.payload, rtol=0, atol=0
        )


def test_cuda_prepared_bakron_shares_one_input_factor() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:0")
    generator = torch.Generator().manual_seed(8200)
    weights = torch.randn((3, 32, 48), generator=generator).to(device)
    input_hessian = _positive_definite(32, seed=8201).to(device)
    output_hessians = torch.stack(
        tuple(_positive_definite(48, seed=8202 + index) for index in range(3))
    ).to(device)
    input_factor, input_reverse = factor_bakron_hessian(input_hessian)
    output_pairs = tuple(
        factor_bakron_hessian(output_hessians[index]) for index in range(3)
    )

    prepared = bakron_block_round_encoder_prepared_batch(
        weights,
        input_factor,
        torch.stack(tuple(pair[0] for pair in output_pairs)),
        input_reverse,
        output_pairs[0][1],
        _round_tiles,
    )
    expected = bakron_block_round_encoder_batch(
        weights,
        input_hessian.expand(3, -1, -1).contiguous(),
        output_hessians,
        _round_tiles,
    )

    torch.testing.assert_close(prepared.reconstruction, expected.reconstruction)
    torch.testing.assert_close(prepared.payload, expected.payload)

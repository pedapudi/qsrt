import torch

from qsrt.high_rate_allocation import (
    dense_h_tile_error_contributions,
    high_rate_record_bits,
    neuron_permutation_from_scores,
    record_rate_map,
    rate_map_proxy_cost,
    select_record_rate_allocation,
    tile_p24_allocation,
    tile_squared_error,
    triplet_tile_selector_bytes,
)
from qsrt.exl3_reference import qsrt_regularized_target


def test_record_schedules_preserve_3p083_bit_budget() -> None:
    for donors in range(12):
        rates = high_rate_record_bits(donors)
        assert len(rates) == 24
        assert sum(rates) == 74
        assert rates.count(2) == donors
        assert rates.count(4) == donors + 2
        assert len(record_rate_map((3584, 3072), rate_axis="n", donor_records=donors)) == 224 * 192
        assert len(record_rate_map((3072, 3584), rate_axis="k", donor_records=donors)) == 224 * 192


def test_record_schedule_selection_uses_complete_equal_byte_maps() -> None:
    shape = (2, 192)
    costs = {bits: torch.zeros(shape, dtype=torch.float64) for bits in (2, 3, 4)}
    costs[2][:, : 3 * 8] = -2
    costs[4][:, (22 - 3) * 8 : 22 * 8] = 1
    allocation = select_record_rate_allocation(
        (costs,), shape=(32, 3072), rate_axis="n"
    )
    assert allocation.donor_records == 3
    assert len(allocation.costs_by_donor_records) == 12
    assert allocation.proxy_cost == float(
        rate_map_proxy_cost((costs,), allocation.tile_bits)
    )
    rates = torch.tensor(allocation.tile_bits).reshape(shape)
    strips = rates.reshape(2, 24, 8).permute(0, 2, 1).reshape(-1, 24)
    assert torch.all(strips.sum(dim=1) == 74)


def test_tile_allocation_chooses_only_profitable_equal_byte_pairs() -> None:
    shape = (4, 192)
    costs = {bits: torch.zeros(shape, dtype=torch.float64) for bits in (2, 3, 4)}
    # In every strip, pair (record 0, record 21) is profitable and pair 1 is not.
    costs[2][:, 0:8] = -2
    costs[4][:, 21 * 8 : 22 * 8] = 1
    costs[2][:, 8:16] = 2
    allocation = tile_p24_allocation((costs,), rate_axis="n")
    rates = torch.tensor(allocation.tile_bits).reshape(shape)
    assert allocation.candidate_p24_tiles == 4 * 8 * 11
    assert allocation.selected_p24_tiles == 4 * 8
    assert torch.all(rates[:, 0:8] == 2)
    assert torch.all(rates[:, 21 * 8 : 22 * 8] == 4)
    assert torch.all(rates[:, 22 * 8 : 24 * 8] == 4)
    strips = rates.reshape(4, 24, 8).permute(0, 2, 1).reshape(-1, 24)
    assert torch.all(strips.sum(dim=1) == 74)


def test_direct_tile_selector_cost_is_4928_bytes_per_expert() -> None:
    assert triplet_tile_selector_bytes(upstream_shared=True) == 4928
    assert triplet_tile_selector_bytes(upstream_shared=False) == 7392


def test_tile_squared_error_uses_physical_16_by_16_tiles() -> None:
    reference = torch.zeros(32, 48)
    reconstruction = reference.clone()
    reconstruction[16:32, 32:48] = 2
    costs = tile_squared_error(reference, reconstruction)
    expected = torch.zeros(2, 3, dtype=torch.float64)
    expected[1, 2] = 16 * 16 * 4
    assert torch.equal(costs, expected)


def test_dense_h_tile_contributions_sum_to_complete_quadratic_error() -> None:
    generator = torch.Generator().manual_seed(17)
    physical_reference = torch.randn(128, 256, generator=generator)
    physical_reconstruction = physical_reference + 0.05 * torch.randn(
        128, 256, generator=generator
    )
    factor = torch.randn(128, 29, generator=generator)
    hessian = factor @ factor.T + 0.25 * torch.eye(128)
    suh = torch.rand(128, generator=generator) + 0.5
    svh = torch.rand(256, generator=generator) + 0.5
    reference = qsrt_regularized_target(physical_reference, suh, svh)
    reconstruction = qsrt_regularized_target(physical_reconstruction, suh, svh)

    contributions = dense_h_tile_error_contributions(
        reference,
        reconstruction,
        hessian,
        suh,
        svh,
    )
    error = physical_reconstruction - physical_reference
    expected = torch.sum(error * (hessian @ error), dtype=torch.float64)
    torch.testing.assert_close(contributions.sum(), expected, rtol=2e-5, atol=2e-5)


def test_neuron_permutation_policies_preserve_four_neuron_groups() -> None:
    scores = torch.linspace(0.25, 3.0, 768)
    scores[::7] *= 11
    for policy in (
        "identity",
        "importance",
        "energy_balanced",
        "stratified_energy_balanced",
    ):
        permutation = neuron_permutation_from_scores(scores, policy=policy)
        assert torch.equal(torch.sort(permutation).values, torch.arange(3072))
        groups = permutation.reshape(-1, 4)
        assert torch.all(groups == groups[:, :1] + torch.arange(4))

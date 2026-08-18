from __future__ import annotations

import pytest
import torch

from scripts.explore_qsrt_k2_allocation import (
    _balanced_p13_unit_rates,
    _balanced_p13_record_rates,
    _best_balanced_p13_units,
    _canonical_reconstruction,
    _encoder_coordinates,
    _four_channel_group_errors,
    _global_shape_clustered_group_order,
    _k2_menu_selector_stats,
    _maximum_weight_pairing,
    _permutation_tile_geometry,
    _p24_band_aligned_permutation,
    _p13_channel_group_deltas_from_tile_errors,
    _p13_channel_group_map_from_rates,
    _p13_record_map,
    _p13_record_map_from_rates,
    _p13_marginal_diagnostics,
    _p13_fractional_tile_map,
    _p13_record_deltas_from_tile_errors,
    _p13_tile_pair_diagnostics,
    _prepare_coupled_search_basis,
    _priority_shape_clustered_group_order,
    _quantize_maps,
    _record_clustered_group_order,
    _qsrt_308_strip_optimal_map,
    _qsrt_308_tile_funded_map,
    _qsrt_308_top2_k4_map,
    _shape_clustered_group_order,
    _tile_balanced_group_order,
    _top2_band_aligned_permutation,
    _weighted_functional_sse,
    _w2_record_traversal_variant,
)


def _tile_error_surfaces(rate_axis: str) -> dict[int, torch.Tensor]:
    shape = (224, 192) if rate_axis == "n" else (192, 224)
    generator = torch.Generator().manual_seed(9182)
    return {
        bits: torch.rand(shape, generator=generator) + (4 - bits) * 0.1
        for bits in (2, 3, 4)
    }


def _as_rate_strips(rate_map: tuple[int, ...], rate_axis: str) -> torch.Tensor:
    shape = (224, 192) if rate_axis == "n" else (192, 224)
    rates = torch.tensor(rate_map, dtype=torch.int8).reshape(shape)
    if rate_axis == "n":
        return rates.reshape(224, 24, 8).permute(0, 2, 1).reshape(-1, 24)
    return rates.reshape(24, 8, 224).permute(2, 1, 0).reshape(-1, 24)


@pytest.mark.parametrize("rate_axis", ("n", "k"))
@pytest.mark.parametrize("donor_records", (0, 1, 6, 12))
def test_p13_record_map_preserves_exact_two_bit_strip_budget(
    rate_axis: str, donor_records: int
) -> None:
    shape = (3584, 3072) if rate_axis == "n" else (3072, 3584)
    rate_map = _p13_record_map(
        shape, donor_records, rate_axis=rate_axis
    )
    strips = _as_rate_strips(rate_map, rate_axis)

    assert torch.equal(strips.sum(dim=1), torch.full((1792,), 48))
    assert torch.all((strips == 1).sum(dim=1) == donor_records)
    assert torch.all((strips == 3).sum(dim=1) == donor_records)
    assert torch.all((strips == 2).sum(dim=1) == 24 - 2 * donor_records)


def test_balanced_p13_record_rates_select_disjoint_best_exchanges() -> None:
    donor = torch.arange(24, dtype=torch.float64)
    recipient = -torch.arange(24, dtype=torch.float64)
    rates_by_count, evidence = _balanced_p13_record_rates(donor, recipient)

    assert rates_by_count["n0"] == (2,) * 24
    for count in (1, 6, 12):
        rates = rates_by_count[f"n{count}"]
        assert rates.count(1) == count
        assert rates.count(3) == count
        assert set(index for index, rate in enumerate(rates) if rate == 1) == set(
            range(count)
        )
        assert set(index for index, rate in enumerate(rates) if rate == 3) == set(
            range(24 - count, 24)
        )
    assert evidence["allocator"] == "exact_disjoint_record_dynamic_program"


def test_unconstrained_p13_allocator_uses_marginal_slopes() -> None:
    donor = torch.tensor((7.0, 1.0, 5.0, 8.0), dtype=torch.float64)
    recipient = torch.tensor((-2.0, -3.0, -9.0, -4.0), dtype=torch.float64)
    result = _best_balanced_p13_units(donor, recipient)

    assert result["k1_count"] == result["k3_count"] == 1
    assert result["k1_units"] == [1]
    assert result["k3_units"] == [2]
    assert result["delta"] == pytest.approx(-8.0)


def test_balanced_p13_unit_rates_retains_fixed_count_shortlist() -> None:
    donor = torch.tensor((7.0, 1.0, 5.0, 8.0), dtype=torch.float64)
    recipient = torch.tensor((-2.0, -3.0, -9.0, -4.0), dtype=torch.float64)
    result = _balanced_p13_unit_rates(donor, recipient, max_count=2)

    assert set(result) == {0, 1, 2}
    assert result[1]["k1_units"] == [1]
    assert result[1]["k3_units"] == [2]
    assert result[1]["delta"] == pytest.approx(-8.0)
    assert result[2]["k1_count"] == result[2]["k3_count"] == 2


def test_p13_marginals_report_crossover_and_exact_gain() -> None:
    donor = torch.full((24,), 4.0, dtype=torch.float64)
    recipient = torch.full((24,), 1.0, dtype=torch.float64)
    donor[3] = 0.25
    recipient[17] = -3.0

    result = _p13_marginal_diagnostics(donor, recipient)

    assert result["maximum_pair_margin"] == pytest.approx(2.75)
    assert result["unconstrained_profitable_pairs"] == 1
    assert result["best_balanced_allocation"] == "n1"
    assert result["best_balanced_gain"] == pytest.approx(2.75)


@pytest.mark.parametrize("rate_axis", ("n", "k"))
def test_p13_record_map_from_rates_preserves_record_identity(
    rate_axis: str,
) -> None:
    shape = (3584, 3072) if rate_axis == "n" else (3072, 3584)
    rates = (1, 3, 1, 3, *(2 for _ in range(20)))
    rate_map = _p13_record_map_from_rates(shape, rates, rate_axis=rate_axis)
    strips = _as_rate_strips(rate_map, rate_axis)

    assert torch.all(strips == torch.tensor(rates, dtype=torch.int8))


@pytest.mark.parametrize("rate_axis", ("n", "k"))
@pytest.mark.parametrize("channels", (16, 64, 128))
def test_p13_channel_group_map_preserves_group_identity_and_budget(
    rate_axis: str,
    channels: int,
) -> None:
    shape = (3584, 3072) if rate_axis == "n" else (3072, 3584)
    groups = 3072 // channels
    rates = (1, 3, *(2 for _ in range(groups - 2)))
    rate_map = _p13_channel_group_map_from_rates(
        shape,
        rates,
        rate_axis=rate_axis,
        channels_per_group=channels,
    )
    values = torch.tensor(rate_map, dtype=torch.int8).reshape(
        (224, 192) if rate_axis == "n" else (192, 224)
    )
    axis_values = values[0] if rate_axis == "n" else values[:, 0]

    expected = torch.tensor(rates, dtype=torch.int8).repeat_interleave(
        channels // 16
    )
    assert torch.equal(axis_values, expected)
    assert int(values.sum()) == 2 * values.numel()


@pytest.mark.parametrize(
    ("policy", "first_visited_record"),
    (("donor_first", 3), ("recipient_first", 20)),
)
def test_w2_record_traversal_preserves_rates_and_h128_groups(
    policy: str,
    first_visited_record: int,
) -> None:
    weight = torch.arange(3072, dtype=torch.float32)[:, None].repeat(1, 3584)
    hessian = torch.diag(torch.arange(1, 3073, dtype=torch.float32))
    permutation = torch.arange(3072)
    record_rates = [2] * 24
    record_rates[3] = 1
    record_rates[20] = 3
    rate_map = _p13_record_map_from_rates(
        (3072, 3584), record_rates, rate_axis="k"
    )

    reordered, reordered_map, evidence = _w2_record_traversal_variant(
        (weight, hessian, permutation),
        rate_map,
        policy=policy,
    )

    reordered_weight, reordered_hessian, reordered_permutation = reordered
    assert evidence["blockldlq_visit_order"][0] == first_visited_record
    for encoder_record, source_record in enumerate(evidence["encoder_record_order"]):
        source_rows = slice(source_record * 128, (source_record + 1) * 128)
        encoder_rows = slice(encoder_record * 128, (encoder_record + 1) * 128)
        assert torch.equal(
            reordered_permutation[encoder_rows], permutation[source_rows]
        )
        assert torch.equal(reordered_weight[encoder_rows], weight[source_rows])
        assert torch.equal(
            torch.diag(reordered_hessian)[encoder_rows],
            torch.diag(hessian)[source_rows],
        )
        tile_rows = torch.tensor(reordered_map, dtype=torch.int8).reshape(192, 224)
        assert torch.all(
            tile_rows[encoder_record * 8 : (encoder_record + 1) * 8]
            == record_rates[source_record]
        )


def test_maximum_weight_pairing_is_exact() -> None:
    assert _maximum_weight_pairing(
        torch.tensor([[1.0, 4.0], [3.0, 2.0]])
    ) == (1, 0)


@pytest.mark.parametrize("rate_axis", ("n", "k"))
def test_p13_fractional_tile_map_selects_only_profitable_pairs(
    rate_axis: str,
) -> None:
    shape = (224, 192) if rate_axis == "n" else (192, 224)
    errors = {
        1: torch.full(shape, 12.0),
        2: torch.full(shape, 10.0),
        3: torch.full(shape, 9.0),
    }
    # One donor and one recipient record.  Half of the matching recipient
    # tiles repay the donor cost; the other half remain P22.
    if rate_axis == "n":
        errors[3][:112, 8:16] = 5.0
    else:
        errors[3][8:16, :112] = 5.0
    record_rates = (1, 3, *(2 for _ in range(22)))
    rate_map, evidence = _p13_fractional_tile_map(
        errors, record_rates, rate_axis=rate_axis
    )
    values = torch.tensor(rate_map, dtype=torch.int8).reshape(shape)

    assert int(values.sum()) == 2 * values.numel()
    assert int((values == 1).sum()) == int((values == 3).sum()) == 8 * 112
    assert evidence["pair_tiles"] == 8 * 224
    assert evidence["selected_pair_tiles"] == 8 * 112
    assert evidence["selected_fraction"] == 0.5


def test_p13_fractional_tile_map_supports_fixed_mirror_record_pairs() -> None:
    shape = (192, 224)
    errors = {
        1: torch.full(shape, 12.0),
        2: torch.full(shape, 10.0),
        3: torch.full(shape, 9.0),
    }
    errors[3][184:192, :] = 5.0
    rates = (*((1,) * 12), *((3,) * 12))

    rate_map, evidence = _p13_fractional_tile_map(
        errors,
        rates,
        rate_axis="k",
        pairing_policy="mirror",
    )
    values = torch.tensor(rate_map, dtype=torch.int8).reshape(shape)

    assert evidence["pairing_policy"] == "mirror"
    assert evidence["record_pairs"][0]["donor"] == 0
    assert evidence["record_pairs"][0]["recipient"] == 23
    assert torch.all(values[0:8] == 1)
    assert torch.all(values[184:192] == 3)
    assert int(values.sum()) == 2 * values.numel()


def test_p13_fractional_tile_map_can_shortlist_best_positive_prefix() -> None:
    shape = (192, 224)
    errors = {
        1: torch.full(shape, 12.0),
        2: torch.full(shape, 10.0),
        3: torch.full(shape, 9.0),
    }
    errors[3][184:192, :112] = 5.0
    rates = (*((1,) * 12), *((3,) * 12))

    rate_map, evidence = _p13_fractional_tile_map(
        errors,
        rates,
        rate_axis="k",
        pairing_policy="mirror",
        positive_fraction=0.25,
    )
    values = torch.tensor(rate_map, dtype=torch.int8).reshape(shape)

    assert evidence["positive_pair_tiles"] == 8 * 112
    assert evidence["selected_pair_tiles"] == 2 * 112
    assert int((values == 1).sum()) == int((values == 3).sum()) == 2 * 112
    assert int(values.sum()) == 2 * values.numel()


@pytest.mark.parametrize("rate_axis", ("n", "k"))
def test_p13_record_deltas_sum_dense_h_tile_costs(rate_axis: str) -> None:
    shape = (224, 192) if rate_axis == "n" else (192, 224)
    errors = {
        1: torch.full(shape, 3.0),
        2: torch.full(shape, 2.0),
        3: torch.full(shape, 1.75),
    }
    donor, recipient = _p13_record_deltas_from_tile_errors(
        errors, rate_axis=rate_axis
    )

    assert donor.shape == recipient.shape == (24,)
    assert torch.all(donor == 8 * 224)
    assert torch.all(recipient == -0.25 * 8 * 224)


@pytest.mark.parametrize("rate_axis", ("n", "k"))
@pytest.mark.parametrize("channels", (16, 64, 128))
def test_p13_channel_group_deltas_preserve_additive_loss(
    rate_axis: str,
    channels: int,
) -> None:
    shape = (224, 192) if rate_axis == "n" else (192, 224)
    errors = {
        1: torch.full(shape, 3.0),
        2: torch.full(shape, 2.0),
        3: torch.full(shape, 1.75),
    }
    donor, recipient = _p13_channel_group_deltas_from_tile_errors(
        errors,
        rate_axis=rate_axis,
        channels_per_group=channels,
    )

    groups = 3072 // channels
    orthogonal_tiles = 224
    group_tiles = channels // 16
    assert donor.shape == recipient.shape == (groups,)
    assert torch.all(donor == group_tiles * orthogonal_tiles)
    assert torch.all(recipient == -0.25 * group_tiles * orthogonal_tiles)


def test_p13_tile_diagnostics_exclude_same_tile_pairing() -> None:
    errors = {
        1: torch.tensor([[2.0, 3.0], [5.0, 7.0]]),
        2: torch.ones((2, 2)),
        3: torch.tensor([[0.0, -1.0], [0.5, 0.75]]),
    }
    result = _p13_tile_pair_diagnostics(errors)

    assert result["maximum_pair_margin"] == pytest.approx(1.0)
    assert result["unconstrained_profitable_pairs"] == 1
    assert result["best_pair"] == {
        "k1_tile": 0,
        "k3_tile": 1,
        "donor_cost": 1.0,
        "recipient_gain": 2.0,
    }


def _functional_band_surfaces() -> dict[str, dict[int, torch.Tensor]]:
    generator = torch.Generator().manual_seed(11703)
    upstream_k3 = torch.rand((224, 192), generator=generator) + 1.0
    down_k3 = torch.rand((192, 224), generator=generator) + 1.0
    return {
        "w13": {
            2: upstream_k3 + 0.25 * torch.rand(
                upstream_k3.shape, generator=generator
            ),
            3: upstream_k3,
            4: upstream_k3 - 0.25 * torch.rand(
                upstream_k3.shape, generator=generator
            ),
        },
        "w2": {
            2: down_k3 + 0.25 * torch.rand(down_k3.shape, generator=generator),
            3: down_k3,
            4: down_k3 - 0.25 * torch.rand(down_k3.shape, generator=generator),
        },
    }


def test_coupled_search_basis_closes_and_scores_in_original_output_basis() -> None:
    generator = torch.Generator().manual_seed(8704)
    hidden = 8
    intermediate = 8
    rows = 11
    source = (
        torch.randn(intermediate, hidden, generator=generator),
        torch.randn(intermediate, hidden, generator=generator),
        torch.randn(hidden, intermediate, generator=generator),
    )
    inputs = torch.randn(rows, hidden, generator=generator)
    h13_samples = torch.randn(17, hidden, generator=generator)
    h2_samples = torch.randn(17, intermediate, generator=generator)
    h13 = h13_samples.T @ h13_samples
    h2 = h2_samples.T @ h2_samples
    permutation = torch.tensor((3, 5, 0, 7, 1, 6, 2, 4))

    basis = _prepare_coupled_search_basis(
        source,
        h13=h13,
        h2=h2,
        inputs=inputs,
        selected_permutation=permutation,
        block_size=4,
        preactivation_block_size=4,
        postactivation_block_size=4,
        pre_permutation="selected",
    )

    assert basis.evidence["full_precision_closure_relative_sse"] < 1e-11
    assert torch.equal(basis.permutation, torch.arange(intermediate))
    assert torch.equal(basis.transform_inputs(inputs), basis.inputs)
    assert torch.allclose(basis.h13, basis.h13.T, rtol=1e-5, atol=1e-5)
    assert torch.allclose(basis.h2, basis.h2.T, rtol=1e-5, atol=1e-5)
    route_weights = torch.ones(rows, 1)
    assert _weighted_functional_sse(
        basis.source,
        inputs=basis.inputs,
        reference=basis.reference_output,
        route_weights=route_weights,
        execute_triplet=basis.execute_triplet,
    ) == pytest.approx(0.0, abs=2e-8)

    external_inputs = torch.randn(7, hidden, generator=generator)
    external_gate = external_inputs @ source[0].T
    external_up = external_inputs @ source[1].T
    external_reference = (
        4.0
        * torch.tanh(external_gate / 4.0)
        * torch.sigmoid(external_gate)
        * 25.0
        * torch.tanh(external_up / 25.0)
    ) @ source[2].T
    external_output = basis.execute_triplet(
        basis.transform_inputs(external_inputs), basis.source
    )
    assert torch.allclose(
        external_output,
        external_reference,
        rtol=2e-5,
        atol=2e-5,
    )


def test_k2_menu_selector_counts_shared_triplet_winners() -> None:
    shape = (224, 192)
    normal = torch.ones(shape)
    challenger = normal.clone()
    challenger[:, :96] = 0.5
    challenger[:, 96:144] = 1.0
    challenger[:, 144:] = 1.5

    selected, evidence = _k2_menu_selector_stats(
        {"normal": normal, "q33": challenger},
        laws=("normal", "q33"),
    )

    assert selected.shape == shape
    assert evidence["mode_counts"] == {
        "normal": 224 * 96,
        "q33": 224 * 96,
    }
    assert evidence["non_normal_fraction"] == pytest.approx(0.5)
    assert evidence["proxy_gain_tiles"] == 224 * 96
    assert evidence["proxy_unchanged_tiles"] == 224 * 96
    assert evidence["all_laws_tie_tiles"] == 224 * 48


def test_conditioning_orders_preserve_rate_record_membership() -> None:
    generator = torch.Generator().manual_seed(1234)
    scores = torch.rand(768, generator=generator)
    features = torch.randn(768, 6, generator=generator)

    for order in (
        _tile_balanced_group_order(scores),
        _shape_clustered_group_order(scores, features),
    ):
        permutation = (order[:, None] * 4 + torch.arange(4)).flatten()
        geometry = _permutation_tile_geometry(permutation, scores)
        assert geometry["all_record_memberships_match"] is True
        assert geometry["record_totals_monotone"] is True


def test_record_clustering_matches_full_incident_tile_response_vectors() -> None:
    scores = torch.arange(768, dtype=torch.float32)
    classes = torch.arange(768) % 4
    features = torch.nn.functional.one_hot(classes, num_classes=4).float()
    order = _record_clustered_group_order(scores, features)

    ordered_classes = classes.index_select(0, order).reshape(192, 4)
    assert torch.all(ordered_classes == ordered_classes[:, :1])
    permutation = (order[:, None] * 4 + torch.arange(4)).flatten()
    geometry = _permutation_tile_geometry(permutation, scores)
    assert geometry["all_record_memberships_match"] is True


@pytest.mark.parametrize(
    "align",
    (_p24_band_aligned_permutation, _top2_band_aligned_permutation),
)
def test_funding_aligned_permutations_move_only_intact_bands_within_records(
    align,
) -> None:
    base = torch.arange(3072)
    permutation, evidence = align(base, _functional_band_surfaces())

    assert torch.equal(torch.sort(permutation).values, base)
    base_bands = base.reshape(24, 8, 16)
    actual_bands = permutation.reshape(24, 8, 16)
    for record in range(24):
        expected = {tuple(row.tolist()) for row in base_bands[record]}
        actual = {tuple(row.tolist()) for row in actual_bands[record]}
        assert actual == expected
    assert evidence["record_membership"] == "fixed_h2_reverse_records"
    assert evidence["coefficient_tiles_per_band"] == 224
    assert evidence["proxy_objective_relative_gain"] >= -1e-12


def test_group_error_attribution_is_transpose_consistent() -> None:
    group_order = torch.tensor((3, 0, 6, 1, 7, 2, 5, 4))
    permutation = (group_order[:, None] * 4 + torch.arange(4)).flatten()
    upstream = torch.zeros(32, 32)
    for position, original in enumerate(group_order.tolist()):
        upstream[:, position * 4 : (position + 1) * 4] = float(original + 1)
    candidates_up = {
        bits: {"regularized": upstream}
        for bits in (2, 3, 4)
    }
    candidates_down = {
        bits: {"regularized": upstream.T.contiguous()}
        for bits in (2, 3, 4)
    }
    errors_up = _four_channel_group_errors(
        torch.zeros_like(upstream),
        candidates_up,
        rate_axis="n",
        permutation=permutation,
    )
    errors_down = _four_channel_group_errors(
        torch.zeros_like(upstream.T),
        candidates_down,
        rate_axis="k",
        permutation=permutation,
    )
    expected = 64.0 * torch.arange(1, 9, dtype=torch.float32).square()
    for bits in (2, 3, 4):
        assert errors_up[bits].shape == (8, 2)
        assert torch.equal(errors_up[bits], errors_down[bits])
        assert torch.equal(errors_up[bits][:, 0], expected)


def test_global_shape_clustering_is_a_valid_tile_conditioning_permutation() -> None:
    generator = torch.Generator().manual_seed(5678)
    scores = torch.rand(768, generator=generator)
    features = torch.randn(768, 6, generator=generator)
    order = _global_shape_clustered_group_order(scores, features)
    permutation = (order[:, None] * 4 + torch.arange(4)).flatten()
    geometry = _permutation_tile_geometry(
        permutation, scores, codec_features=features
    )

    assert torch.equal(torch.sort(order).values, torch.arange(768))
    assert geometry["record_totals_monotone"] is True
    assert geometry["all_record_memberships_match"] is False
    assert "within_tile_priority_spread" in geometry
    assert "within_tile_codec_feature_variance" in geometry
    clustered = features.index_select(0, order).reshape(-1, 4, 6)
    exact = features.index_select(
        0, torch.argsort(scores, stable=True)
    ).reshape(-1, 4, 6)
    clustered_variance = (
        clustered - clustered.mean(dim=1, keepdim=True)
    ).square().mean()
    exact_variance = (exact - exact.mean(dim=1, keepdim=True)).square().mean()
    assert clustered_variance < exact_variance


def test_conditioning_permutation_forms_tiles_from_complete_four_channel_groups() -> None:
    generator = torch.Generator().manual_seed(9012)
    scores = torch.rand(768, generator=generator)
    features = torch.randn(768, 6, generator=generator)
    order = _global_shape_clustered_group_order(scores, features)
    permutation = (order[:, None] * 4 + torch.arange(4)).flatten()

    grouped = permutation.reshape(192, 4, 4)
    assert torch.equal(
        grouped,
        grouped[:, :, :1] + torch.arange(4),
    )
    assert torch.equal(
        torch.sort(grouped[:, :, 0].flatten() // 4).values,
        torch.arange(768),
    )


def test_priority_shape_clustering_retains_a_monotone_record_axis() -> None:
    generator = torch.Generator().manual_seed(3456)
    scores = torch.rand(768, generator=generator)
    features = torch.randn(768, 6, generator=generator)
    order = _priority_shape_clustered_group_order(scores, features)
    permutation = (order[:, None] * 4 + torch.arange(4)).flatten()
    geometry = _permutation_tile_geometry(
        permutation, scores, codec_features=features
    )

    assert torch.equal(torch.sort(order).values, torch.arange(768))
    assert geometry["record_totals_monotone"] is True
    assert geometry["all_record_memberships_match"] is False


def test_coupled_neuron_permutation_is_identical_across_matrix_roles() -> None:
    permutation = torch.tensor([2, 0, 3, 1])
    contexts = torch.zeros(4, dtype=torch.long)
    upstream_hessian = torch.eye(3)
    down_hessian = torch.eye(4)
    w1 = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    w3 = w1 + 100
    w2 = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 200

    for matrix, source, hessian in (
        ("w1", w1, upstream_hessian),
        ("w3", w3, upstream_hessian),
        ("w2", w2, down_hessian),
    ):
        encoder_weight, _, actual = _encoder_coordinates(
            source,
            hessian,
            contexts,
            matrix=matrix,
            permutation_override=permutation,
        )
        assert torch.equal(actual, permutation)
        assert torch.equal(
            _canonical_reconstruction(
                encoder_weight,
                source,
                permutation,
                matrix=matrix,
            ),
            source,
        )


def test_permuted_funding_tiles_align_neurons_across_matrix_roles() -> None:
    """Upstream columns and down rows must name the same neuron tile."""

    permutation = torch.tensor([*range(16, 32), *range(0, 16)])
    contexts = torch.zeros(32, dtype=torch.long)
    h13 = torch.eye(16)
    h2 = torch.eye(32)
    w1 = torch.arange(32 * 16, dtype=torch.float32).reshape(32, 16)
    w3 = w1 + 10_000
    w2 = torch.arange(16 * 32, dtype=torch.float32).reshape(16, 32) + 20_000

    encoded_w1, _, _ = _encoder_coordinates(
        w1,
        h13,
        contexts,
        matrix="w1",
        permutation_override=permutation,
    )
    encoded_w3, _, _ = _encoder_coordinates(
        w3,
        h13,
        contexts,
        matrix="w3",
        permutation_override=permutation,
    )
    encoded_w2, _, _ = _encoder_coordinates(
        w2,
        h2,
        contexts,
        matrix="w2",
        permutation_override=permutation,
    )

    # The neuron coordinate is N for the transposed upstream matrices and K
    # for the transposed down matrix. Tile zero must contain the same first
    # 16 entries of P in all three matrix roles.
    assert torch.equal(encoded_w1[:, :16], w1.index_select(0, permutation[:16]).T)
    assert torch.equal(encoded_w3[:, :16], w3.index_select(0, permutation[:16]).T)
    assert torch.equal(encoded_w2[:16, :], w2.index_select(1, permutation[:16]).T)


def test_transposed_funding_surfaces_produce_transposed_rate_maps() -> None:
    """The same tile problem must agree across upstream-N and down-K axes."""

    generator = torch.Generator().manual_seed(7719)
    upstream_errors = {
        bits: torch.rand((224, 192), generator=generator) + bits * 0.01
        for bits in (2, 3, 4)
    }
    down_errors = {
        bits: values.T.contiguous() for bits, values in upstream_errors.items()
    }

    upstream_paired, _ = _qsrt_308_tile_funded_map(
        (upstream_errors,), rate_axis="n", fraction=None
    )
    down_paired, _ = _qsrt_308_tile_funded_map(
        (down_errors,), rate_axis="k", fraction=None
    )
    assert torch.equal(
        torch.tensor(upstream_paired).reshape(224, 192).T,
        torch.tensor(down_paired).reshape(192, 224),
    )

    upstream_top2, _ = _qsrt_308_top2_k4_map(
        (upstream_errors,), rate_axis="n"
    )
    down_top2, _ = _qsrt_308_top2_k4_map(
        (down_errors,), rate_axis="k"
    )
    assert torch.equal(
        torch.tensor(upstream_top2).reshape(224, 192).T,
        torch.tensor(down_top2).reshape(192, 224),
    )

    upstream_optimal, _ = _qsrt_308_strip_optimal_map(
        (upstream_errors,), rate_axis="n"
    )
    down_optimal, _ = _qsrt_308_strip_optimal_map(
        (down_errors,), rate_axis="k"
    )
    assert torch.equal(
        torch.tensor(upstream_optimal).reshape(224, 192).T,
        torch.tensor(down_optimal).reshape(192, 224),
    )


def test_tile_quantizer_rejects_an_implicit_permutation() -> None:
    with pytest.raises(ValueError, match="explicit frozen neuron permutation"):
        _quantize_maps(
            torch.empty(16, 16),
            torch.eye(16),
            torch.zeros(16, dtype=torch.long),
            matrix="w1",
            maps={"k3": (3,)},
            layer=0,
            expert=0,
            device=torch.device("cpu"),
            quantizer_module=None,
            ldlq_tf32=False,
        )


def test_prepared_coordinates_must_match_the_frozen_permutation() -> None:
    prepared = (
        torch.empty(16, 16),
        torch.eye(16),
        torch.arange(16),
    )
    with pytest.raises(ValueError, match="do not match the frozen permutation"):
        _quantize_maps(
            torch.empty(16, 16),
            torch.eye(16),
            torch.zeros(16, dtype=torch.long),
            matrix="w1",
            maps={"k3": (3,)},
            layer=0,
            expert=0,
            device=torch.device("cpu"),
            quantizer_module=None,
            ldlq_tf32=False,
            prepared=prepared,
            permutation_override=torch.arange(15, -1, -1),
        )


@pytest.mark.parametrize("rate_axis", ("n", "k"))
def test_tile_funded_h308_has_an_exact_local_74_bit_budget(
    rate_axis: str,
) -> None:
    rate_map, evidence = _qsrt_308_tile_funded_map(
        (_tile_error_surfaces(rate_axis),),
        rate_axis=rate_axis,
        fraction=0.5,
    )
    strips = _as_rate_strips(rate_map, rate_axis)
    assert torch.equal(strips.sum(dim=1), torch.full((1792,), 74))
    assert evidence["tile_pair_decisions"] == 11 * 8 * 224


@pytest.mark.parametrize("rate_axis", ("n", "k"))
def test_top2_h308_assigns_exactly_two_k4_tiles_per_strip(
    rate_axis: str,
) -> None:
    rate_map, _ = _qsrt_308_top2_k4_map(
        (_tile_error_surfaces(rate_axis),),
        rate_axis=rate_axis,
    )
    strips = _as_rate_strips(rate_map, rate_axis)
    assert torch.equal((strips == 4).sum(dim=1), torch.full((1792,), 2))
    assert int(torch.count_nonzero(strips == 2)) == 0
    assert torch.equal(strips.sum(dim=1), torch.full((1792,), 74))


@pytest.mark.parametrize("rate_axis", ("n", "k"))
def test_strip_optimal_h308_balances_each_tile_strip_independently(
    rate_axis: str,
) -> None:
    rate_map, _ = _qsrt_308_strip_optimal_map(
        (_tile_error_surfaces(rate_axis),),
        rate_axis=rate_axis,
    )
    strips = _as_rate_strips(rate_map, rate_axis)
    k2 = (strips == 2).sum(dim=1)
    k4 = (strips == 4).sum(dim=1)
    assert torch.equal(k4, k2 + 2)
    assert torch.equal(strips.sum(dim=1), torch.full((1792,), 74))


@pytest.mark.parametrize("rate_axis", ("n", "k"))
def test_strip_optimal_h308_finds_a_coupled_k2_k4_optimum(
    rate_axis: str,
) -> None:
    shape = (224, 192) if rate_axis == "n" else (192, 224)
    errors = {
        bits: torch.full(shape, 100.0)
        for bits in (2, 3, 4)
    }

    def records(value: torch.Tensor) -> torch.Tensor:
        if rate_axis == "n":
            return value.reshape(224, 24, 8)
        return value.reshape(24, 8, 224)

    if rate_axis == "k":
        records(errors[2])[0] = 0
        records(errors[4])[1:4] = 0
        records(errors[3])[4:] = 0
    else:
        records(errors[2])[:, 0] = 0
        records(errors[4])[:, 1:4] = 0
        records(errors[3])[:, 4:] = 0

    rate_map, _ = _qsrt_308_strip_optimal_map((errors,), rate_axis=rate_axis)
    strips = _as_rate_strips(rate_map, rate_axis)
    assert torch.all(strips[:, 0] == 2)
    assert torch.all(strips[:, 1:4] == 4)
    assert torch.all(strips[:, 4:] == 3)

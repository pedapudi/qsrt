import pytest
import torch

from qsrt.kimi_layer92_fisher import PairedLayer92FisherSamples


def _samples() -> PairedLayer92FisherSamples:
    return PairedLayer92FisherSamples(
        expert_input=torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        ),
        expert_indices=torch.tensor([[1, 4], [2, 3], [4, 0]], dtype=torch.int32),
        route_weights=torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.7, 0.3]]),
        output_gradients=torch.tensor(
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]
        ),
        context_index=torch.tensor([10, 10, 11], dtype=torch.int64),
        row_index=torch.tensor([0, 1, 0], dtype=torch.int64),
        split=torch.tensor([0, 0, 1], dtype=torch.int64),
    )


def test_paired_layer92_fisher_selects_aligned_expert_occurrences() -> None:
    rows = _samples().expert_occurrences(4)
    torch.testing.assert_close(
        rows["expert_input"], torch.tensor([[1.0, 2.0], [5.0, 6.0]])
    )
    torch.testing.assert_close(rows["route_weight"], torch.tensor([0.8, 0.7]))
    torch.testing.assert_close(
        rows["output_gradient"],
        torch.tensor([[7.0, 8.0, 9.0], [13.0, 14.0, 15.0]]),
    )
    assert rows["context_index"].tolist() == [10, 11]
    assert rows["row_index"].tolist() == [0, 0]
    assert rows["split"].tolist() == [0, 1]


def test_paired_layer92_fisher_rejects_negative_route_weight() -> None:
    samples = _samples()
    samples.route_weights[0, 0] = -0.1
    with pytest.raises(ValueError, match="nonnegative"):
        samples.validate()


def test_paired_layer92_fisher_requires_requested_expert_support() -> None:
    with pytest.raises(ValueError, match="no expert 5 rows"):
        _samples().expert_occurrences(5)

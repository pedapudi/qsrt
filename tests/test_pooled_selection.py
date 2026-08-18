import pytest

from qsrt.pooled_selection import (
    PooledCandidateScore,
    select_pooled_candidate,
    selection_convergence,
)


def _score(
    name: str,
    sse: float,
    *,
    metadata: int = 1,
    n13: int = 0,
    n2: int = 0,
) -> PooledCandidateScore:
    return PooledCandidateScore(
        name=name,
        sse=sse,
        source_energy=1000.0,
        metadata_bytes=metadata,
        n13=n13,
        n2=n2,
    )


def test_pooled_selector_uses_all_rows_without_document_fields() -> None:
    selected = select_pooled_candidate(
        (_score("R0", 10.0), _score("R1", 9.0, n13=1))
    )
    assert selected.selected.name == "R1"
    assert selected.minimum_sse == 9.0


def test_pooled_selector_tie_breaks_by_metadata_and_rate_movement() -> None:
    tolerance = 1e-7
    candidates = (
        _score("tile", 4.0, metadata=449),
        _score("record-high", 4.0 + tolerance / 2, metadata=1, n13=1, n2=1),
        _score("record-low", 4.0 + tolerance / 2, metadata=1, n13=0, n2=1),
    )
    selected = select_pooled_candidate(candidates)
    assert selected.tie_tolerance == pytest.approx(tolerance)
    assert selected.selected.name == "record-low"
    assert selected.tied_names == ("record-high", "record-low", "tile")


def test_pooled_selector_rejects_mismatched_source_reference() -> None:
    candidates = (
        _score("a", 1.0),
        PooledCandidateScore("b", 1.0, 999.0, 1),
    )
    with pytest.raises(ValueError, match="source-energy"):
        select_pooled_candidate(candidates)


def test_selection_convergence_compares_every_prefix_to_full_population() -> None:
    report = selection_convergence(
        {
            250_000: (_score("R0", 8.0), _score("R1", 8.1, n13=1)),
            500_000: (_score("R0", 8.0), _score("R1", 7.9, n13=1)),
            4_000_000: (_score("R0", 8.0), _score("R1", 7.0, n13=1)),
        }
    )
    assert report["authoritative_rows"] == 4_000_000
    assert report["authoritative_candidate"] == "R1"
    assert [row["matches_authoritative"] for row in report["prefixes"]] == [
        False,
        True,
        True,
    ]

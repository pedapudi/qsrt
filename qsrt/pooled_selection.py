"""Deterministic selection over pooled routed-expert distortion.

Candidate distortion is accumulated over every captured row.  Document
partitions are deliberately absent: the selector consumes sufficient
statistics for the complete calibration population and leaves model-level
generalization checks to independent KLD and task corpora.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class PooledCandidateScore:
    """One serialized expert candidate and its pooled functional error."""

    name: str
    sse: float
    source_energy: float
    metadata_bytes: int
    n13: int = 0
    n2: int = 0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("candidate name must be nonempty")
        if not math.isfinite(self.sse) or self.sse < 0:
            raise ValueError(f"candidate {self.name!r} has invalid SSE")
        if not math.isfinite(self.source_energy) or self.source_energy <= 0:
            raise ValueError(f"candidate {self.name!r} has invalid source energy")
        if self.metadata_bytes < 0:
            raise ValueError(f"candidate {self.name!r} has invalid metadata size")
        if self.n13 < 0 or self.n2 < 0:
            raise ValueError(f"candidate {self.name!r} has invalid transfer count")

    @property
    def transfer_count(self) -> int:
        return self.n13 + self.n2


@dataclass(frozen=True)
class PooledSelection:
    """Selected candidate and the numerical tie set used to choose it."""

    selected: PooledCandidateScore
    minimum_sse: float
    tie_tolerance: float
    tied_names: tuple[str, ...]


def select_pooled_candidate(
    candidates: Iterable[PooledCandidateScore],
    *,
    tie_relative_to_source_energy: float = 1e-10,
) -> PooledSelection:
    """Select minimum SSE with storage- and rate-conservative tie breaking.

    Scores within ``tie_relative_to_source_energy * source_energy`` of the
    minimum are treated as numerically indistinguishable.  Ties prefer fewer
    metadata bytes, then fewer total K2/K4 transfers, then smaller upstream and
    down transfer counts, and finally the stable candidate name.
    """

    scores = tuple(candidates)
    if not scores:
        raise ValueError("pooled selection requires at least one candidate")
    for score in scores:
        score.validate()
    if len({score.name for score in scores}) != len(scores):
        raise ValueError("pooled candidate names must be unique")
    source_energy = scores[0].source_energy
    energy_tolerance = max(64 * math.ulp(source_energy), 1e-12 * source_energy)
    if any(abs(score.source_energy - source_energy) > energy_tolerance for score in scores):
        raise ValueError("pooled candidates do not share one source-energy reference")
    relative_tolerance = float(tie_relative_to_source_energy)
    if not math.isfinite(relative_tolerance) or relative_tolerance < 0:
        raise ValueError("tie tolerance must be finite and nonnegative")
    minimum = min(score.sse for score in scores)
    tolerance = relative_tolerance * source_energy
    tied = tuple(score for score in scores if score.sse <= minimum + tolerance)
    selected = min(
        tied,
        key=lambda score: (
            score.metadata_bytes,
            score.transfer_count,
            score.n13,
            score.n2,
            score.name,
        ),
    )
    return PooledSelection(
        selected=selected,
        minimum_sse=minimum,
        tie_tolerance=tolerance,
        tied_names=tuple(sorted(score.name for score in tied)),
    )


def selection_convergence(
    prefix_scores: Mapping[int, Iterable[PooledCandidateScore]],
    *,
    authoritative_rows: int | None = None,
    tie_relative_to_source_energy: float = 1e-10,
) -> dict[str, object]:
    """Report candidate stability as progressively more rows are accumulated."""

    if not prefix_scores:
        raise ValueError("selection convergence requires prefix scores")
    rows = tuple(sorted(int(value) for value in prefix_scores))
    if rows[0] <= 0 or len(set(rows)) != len(rows):
        raise ValueError("prefix row counts must be unique and positive")
    if authoritative_rows is None:
        authoritative_rows = rows[-1]
    if authoritative_rows not in prefix_scores:
        raise ValueError("authoritative prefix is absent")
    selections = {
        row_count: select_pooled_candidate(
            prefix_scores[row_count],
            tie_relative_to_source_energy=tie_relative_to_source_energy,
        )
        for row_count in rows
    }
    authoritative = selections[authoritative_rows].selected
    report = []
    for row_count in rows:
        result = selections[row_count]
        selected = result.selected
        report.append(
            {
                "rows": row_count,
                "selected": selected.name,
                "matches_authoritative": selected.name == authoritative.name,
                "selected_nmse": selected.sse / selected.source_energy,
                "tie_tolerance": result.tie_tolerance,
                "tied_names": list(result.tied_names),
            }
        )
    return {
        "authoritative_rows": authoritative_rows,
        "authoritative_candidate": authoritative.name,
        "prefixes": report,
    }


__all__ = [
    "PooledCandidateScore",
    "PooledSelection",
    "select_pooled_candidate",
    "selection_convergence",
]

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from qsrt.glm52_document_disjoint_confirmation import (
    retarget_reference_symlink,
    summarize_document_paired_kld,
    token_ids_sha256,
    validate_frozen_low_rank_candidate,
    validate_public_reference_auxiliary_plan,
    validate_public_reference_files,
)


def _plan(*, digest: str, byte_count: int) -> dict:
    return {
        "schema": "qsrt_glm52_public_reference_auxiliary_plan",
        "schema_version": 1,
        "status": "frozen_before_reference_file_download",
        "selected_document_count": 2,
        "tokenization": {"context_tokens": 512},
        "selected_chunks": [
            {
                "document_sha256": "1" * 64,
                "prompt_token_ids_sha256": "2" * 64,
                "reference_file_sha256": digest,
                "reference_file_bytes": byte_count,
                "selected_chunk": 7,
                "reference_file": "batch_000007_001.safetensors",
            },
            {
                "document_sha256": "3" * 64,
                "prompt_token_ids_sha256": "4" * 64,
                "reference_file_sha256": "5" * 64,
                "reference_file_bytes": 9,
                "selected_chunk": 11,
                "reference_file": "batch_000011_001.safetensors",
            },
        ],
    }


def test_auxiliary_plan_and_files_are_content_addressed(tmp_path: Path) -> None:
    first = b"reference"
    second = b"123456789"
    (tmp_path / "batch_000007_001.safetensors").write_bytes(first)
    (tmp_path / "batch_000011_001.safetensors").write_bytes(second)
    plan = _plan(digest=hashlib.sha256(first).hexdigest(), byte_count=len(first))
    plan["selected_chunks"][1]["reference_file_sha256"] = hashlib.sha256(
        second
    ).hexdigest()

    validated = validate_public_reference_files(plan, tmp_path)

    assert validated["document_count"] == 2
    assert validated["total_reference_bytes"] == len(first) + len(second)
    assert validate_public_reference_auxiliary_plan(plan)["context_tokens"] == 512


def test_auxiliary_plan_rejects_duplicate_documents() -> None:
    plan = _plan(digest="0" * 64, byte_count=1)
    plan["selected_chunks"][1]["document_sha256"] = "1" * 64

    with pytest.raises(ValueError, match="unique"):
        validate_public_reference_auxiliary_plan(plan)


def test_committed_auxiliary_plan_is_disjoint_from_every_candidate_input() -> None:
    repository = Path(__file__).parents[1]
    auxiliary_path = (
        repository / "experiments/glm52_public_reference_auxiliary_plan.json"
    )
    source_plan_path = (
        repository / "experiments/glm52_wikitext_document_disjoint_corpus_plan.json"
    )
    auxiliary = json.loads(auxiliary_path.read_text())
    source_plan = json.loads(source_plan_path.read_text())
    assert hashlib.sha256(auxiliary_path.read_bytes()).hexdigest() == (
        "91484042ccd375993129d7d41d15dfeb457d48d69bef2cd5077882c418c9f24c"
    )
    assert auxiliary["source_corpus_plan_sha256"] == hashlib.sha256(
        source_plan_path.read_bytes()
    ).hexdigest()
    validated = validate_public_reference_auxiliary_plan(auxiliary)
    assert validated["document_count"] == 16
    assert [row["selected_chunk"] for row in validated["rows"]] == [
        462,
        465,
        474,
        484,
        487,
        492,
        498,
        503,
        505,
        507,
        510,
        519,
        531,
        542,
        553,
        561,
    ]

    used_rows: set[int] = set()
    for start, stop in source_plan["published_bf16_reference"][
        "overlapping_row_ranges"
    ]:
        used_rows.update(range(start, stop))
    for collection in ("activation_fit", "candidate_selection"):
        for window in source_plan[collection]["windows"]:
            document = window["document"]
            used_rows.update(
                range(
                    document["row_start_inclusive"],
                    document["row_stop_exclusive"],
                )
            )
    for row in validated["rows"]:
        assert not used_rows.intersection(
            range(row["row_start_inclusive"], row["row_stop_exclusive"])
        )


def test_reference_symlink_retargets_atomically(tmp_path: Path) -> None:
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    link = tmp_path / "current.safetensors"

    retarget_reference_symlink(link, first)
    assert link.read_bytes() == b"first"
    retarget_reference_symlink(link, second)
    assert link.read_bytes() == b"second"


def test_document_summary_uses_documents_as_sampling_units() -> None:
    baseline = {
        "a": torch.tensor([0.10, 0.20]),
        "b": torch.tensor([0.30, 0.40]),
    }
    candidate = {
        "a": torch.tensor([0.08, 0.18]),
        "b": baseline["b"].clone(),
    }

    summary = summarize_document_paired_kld(
        baseline, candidate, bootstrap_resamples=1000, bootstrap_seed=17
    )

    assert summary["document_count"] == 2
    assert summary["position_count"] == 4
    assert summary["document_outcomes"] == {
        "candidate_better": 1,
        "candidate_equal": 1,
        "candidate_worse": 0,
    }
    assert summary["pooled_position_weight"][
        "candidate_mean_forward_kld"
    ] == pytest.approx(0.24)
    assert summary["paired_document_bootstrap"]["resamples"] == 1000


def test_token_hash_has_a_fixed_unsigned_little_endian_encoding() -> None:
    assert token_ids_sha256([1, 256]) == hashlib.sha256(
        b"\x01\x00\x00\x00\x00\x01\x00\x00"
    ).hexdigest()
    with pytest.raises(ValueError, match="unsigned"):
        token_ids_sha256([-1])


def test_runtime_artifact_must_match_the_frozen_correction() -> None:
    frozen = {
        "layer": 3,
        "expert": 103,
        "rank": 4,
        "factor_dtype": "BF16",
        "selected_ridge_factor": 0.001,
        "factor_a_sha256": "a" * 64,
        "factor_b_sha256": "b" * 64,
        "logical_factor_bytes": 65536,
        "materialized_down_sha256": "c" * 64,
    }
    registration = {
        "schema": "qsrt_glm52_low_rank_down_confirmation_registration",
        "schema_version": 1,
        "status": "frozen_before_document_disjoint_confirmation",
        "frozen_correction": frozen,
    }
    record = {
        **frozen,
        "logical_adapter_bytes": frozen["logical_factor_bytes"],
    }
    del record["logical_factor_bytes"]
    artifact = {"report": {"experts": [record]}}

    validated = validate_frozen_low_rank_candidate(registration, artifact)
    assert validated["expert"] == 103

    artifact["report"]["experts"][0]["factor_a_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="differs"):
        validate_frozen_low_rank_candidate(registration, artifact)

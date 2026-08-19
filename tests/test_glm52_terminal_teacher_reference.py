from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import qsrt.glm52_terminal_teacher_reference as terminal_reference
from qsrt.glm52_document_disjoint_confirmation import token_ids_sha256
from qsrt.glm52_terminal_teacher_reference import (
    AXES,
    build_terminal_teacher_reference_plan,
    validate_document_tokenization,
    validate_terminal_teacher_reference_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    ROOT / "experiments/glm52_terminal_hidden_teacher_reference_plan.json"
)


def test_committed_terminal_reference_plan_is_frozen_and_maximizes_confirmation_rows() -> None:
    plan = json.loads(PLAN_PATH.read_text())
    validated = validate_terminal_teacher_reference_plan(plan)

    assert hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest() == (
        "b73690cb3507e64b51c45312d7817ccb1d9d8a0372d05ee3b68c28a3ff1e9519"
    )
    assert validated == {
        "document_count": 40,
        "screening_document_count": 8,
        "confirmation_document_count": 32,
        "maximum_context_tokens": 2048,
        "total_logit_rows": 74816,
    }
    for axis in AXES:
        assert sum(
            row["axis"] == axis and row["evaluation_tier"] == "screening"
            for row in plan["documents"]
        ) == 2
    assert plan["selection_rule"]["confirmation_axis_counts"] == {
        "axis1_general": 22,
        "axis2_legal": 1,
        "axis3_code_agentic": 9,
        "axis4_reasoning_termination": 0,
    }
    confirmation = [
        row
        for row in plan["documents"]
        if row["evaluation_tier"] == "confirmation"
    ]
    assert sum(row["logit_rows"] for row in confirmation) == 65_482
    assert [row["context_tokens"] for row in confirmation].count(2_048) == 31
    assert min(row["context_tokens"] for row in confirmation) == 2_026


def test_terminal_reference_plan_rejects_confirmation_document_reuse() -> None:
    plan = json.loads(PLAN_PATH.read_text())
    plan["documents"][8]["document_sha256"] = plan["documents"][0][
        "document_sha256"
    ]

    with pytest.raises(ValueError, match="document identity"):
        validate_terminal_teacher_reference_plan(plan)


def test_terminal_reference_asset_ranges_match_tensor_shapes() -> None:
    plan = json.loads(PLAN_PATH.read_text())
    assets = plan["reference_assets"]
    head = assets["language_model_head"]
    norm = assets["final_normalization"]
    hidden = assets["terminal_hidden"]

    assert head["source_byte_stop_exclusive"] - head["source_byte_start"] == (
        154_880 * 6_144 * 2
    )
    assert norm["source_byte_stop_exclusive"] - norm["source_byte_start"] == (
        6_144 * 2
    )
    assert hidden["bytes"] == 1_049_589 * 6_144 * 2


def test_document_tokenization_requires_the_published_full_token_hash() -> None:
    token_ids = [0, 7, 11, 13]
    text = "one frozen document"
    row = {
        "document_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_document_tokens": len(token_ids),
        "full_token_ids_sha256_u32le": token_ids_sha256(token_ids),
        "context_tokens": 3,
    }

    receipt = validate_document_tokenization(
        row=row, text=text, token_ids=token_ids
    )

    assert receipt["prompt_token_ids"] == [0, 7, 11]
    assert receipt["target_token_ids"] == [7, 11]
    with pytest.raises(ValueError, match="tokenization differs"):
        validate_document_tokenization(
            row=row, text=text, token_ids=[0, 7, 11, 14]
        )


def test_plan_builder_selects_by_document_identity_before_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = [
        {"axis": "axis1_general", "source": "unused", "text": f"unused-{index}"}
        for index in range(12_228)
    ]
    documents = []
    row_start = 0
    for axis_index, axis in enumerate(AXES):
        for within_axis in range(10):
            corpus_line = axis_index * 10 + within_axis
            text = f"{axis}-document-{within_axis}"
            corpus[corpus_line] = {"axis": axis, "source": axis, "text": text}
            tokens = 64 + within_axis
            documents.append(
                {
                    "role": "holdout",
                    "tokens": tokens,
                    "corpus_line": corpus_line,
                    "document_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "epoch": len(documents),
                    "global_row_start": row_start,
                    "token_ids_sha256_u32le": "a" * 64,
                }
            )
            row_start += tokens
    documents.extend(
        {"role": "fit", "tokens": 1, "epoch": index}
        for index in range(40, 1_773)
    )
    capture = {
        "schema": "glm52-r10-sqg-document-plan-v1",
        "documents_total": 1_773,
        "tokens_total": terminal_reference.TERMINAL_HIDDEN_ROWS,
        "documents": documents,
    }
    capture_bytes = json.dumps(capture, sort_keys=True).encode()
    corpus_bytes = (
        "\n".join(json.dumps(row, sort_keys=True) for row in corpus) + "\n"
    ).encode()
    monkeypatch.setattr(
        terminal_reference,
        "CANONICAL_DOCUMENT_PLAN_SHA256",
        hashlib.sha256(capture_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        terminal_reference,
        "CALIBRATION_CORPUS_SHA256",
        hashlib.sha256(corpus_bytes).hexdigest(),
    )

    plan = build_terminal_teacher_reference_plan(
        capture_plan_bytes=capture_bytes,
        corpus_bytes=corpus_bytes,
        frozen_at_utc="2026-08-19T00:00:00Z",
    )

    assert len(plan["documents"]) == 40
    assert plan["selection_rule"]["selection_uses_model_output"] is False
    assert {
        row["evaluation_tier"] for row in plan["documents"]
    } == {"screening", "confirmation"}

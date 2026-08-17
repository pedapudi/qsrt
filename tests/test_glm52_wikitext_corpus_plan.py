from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path("scripts/build_glm52_wikitext_corpus_plan.py")
SPEC = importlib.util.spec_from_file_location("glm52_wikitext_corpus_plan", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _WhitespaceTokenizer:
    def __call__(self, text: str, **kwargs):
        values = list(range(len(text.split())))
        maximum = kwargs.get("max_length")
        if maximum is not None and kwargs.get("truncation"):
            values = values[:maximum]
        return {"input_ids": values}


def test_document_grouping_uses_only_top_level_headings() -> None:
    documents = MODULE.group_wikitext_documents(
        [
            "",
            " = First article = \n",
            "alpha beta gamma",
            " == Subheading == \n",
            "delta epsilon",
            " = Second article = \n",
            "zeta eta theta iota",
        ]
    )
    assert [(item.row_start, item.row_stop) for item in documents] == [(1, 5), (5, 7)]
    assert [item.title for item in documents] == ["First article", "Second article"]


def test_windows_keep_documents_disjoint_and_store_exact_tokens() -> None:
    documents = tuple(
        MODULE.WikiDocument(index, index + 1, f"doc {index}", "one two three four")
        for index in range(4)
    )
    windows, consumed = MODULE.document_windows(
        documents,
        tokenizer=_WhitespaceTokenizer(),
        collection="activation_fit",
        count=2,
        context_length=3,
        minimum_tokens=3,
    )
    assert consumed == 2
    assert [window["token_ids"] for window in windows] == [[0, 1, 2], [0, 1, 2]]
    assert windows[0]["document"]["row_stop_exclusive"] <= windows[1]["document"][
        "row_start_inclusive"
    ]


def test_reference_document_count_stops_at_partial_article() -> None:
    documents = (
        MODULE.WikiDocument(0, 1, "first", "one two"),
        MODULE.WikiDocument(1, 2, "second", "three four five"),
    )
    assert MODULE.reference_document_count(
        documents, tokenizer=_WhitespaceTokenizer(), reference_tokens=4
    ) == 2
    with pytest.raises(ValueError, match="do not contain"):
        MODULE.reference_document_count(
            documents, tokenizer=_WhitespaceTokenizer(), reference_tokens=99
        )

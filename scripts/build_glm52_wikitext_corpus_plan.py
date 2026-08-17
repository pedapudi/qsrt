#!/usr/bin/env python3
"""Freeze document-disjoint GLM-5.2 activation-fit and selection token windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


TOP_LEVEL_HEADING = re.compile(r"^\s*=\s+([^=].*?)\s+=\s*$")


@dataclass(frozen=True)
class WikiDocument:
    row_start: int
    row_stop: int
    title: str
    text: str


def group_wikitext_documents(rows: Sequence[str]) -> tuple[WikiDocument, ...]:
    """Group WikiText rows at single-equals article headings."""

    documents: list[WikiDocument] = []
    current_start: int | None = None
    current_title = ""
    current_rows: list[str] = []

    def finish(stop: int) -> None:
        nonlocal current_start, current_title, current_rows
        if current_start is not None and current_rows:
            text = "\n\n".join(value for value in current_rows if value.strip())
            if text:
                documents.append(
                    WikiDocument(current_start, stop, current_title, text)
                )
        current_start = None
        current_title = ""
        current_rows = []

    for row_index, raw_text in enumerate(rows):
        text = str(raw_text)
        heading = TOP_LEVEL_HEADING.match(text)
        if heading:
            finish(row_index)
            current_start = row_index
            current_title = heading.group(1).strip()
            current_rows = [text]
        elif current_start is not None and text.strip():
            current_rows.append(text)
    finish(len(rows))
    if not documents:
        raise ValueError("WikiText rows contained no top-level article headings")
    return tuple(documents)


def reference_document_count(
    documents: Sequence[WikiDocument],
    *,
    tokenizer: Any,
    reference_tokens: int,
) -> int:
    """Count complete or partial articles consumed by the published prefix."""

    if reference_tokens < 1:
        raise ValueError("reference_tokens must be positive")
    prefix: list[str] = []
    for count, document in enumerate(documents, start=1):
        prefix.append(document.text)
        token_ids = tokenizer(
            "\n\n".join(prefix), add_special_tokens=False
        )["input_ids"]
        if len(token_ids) >= reference_tokens:
            return count
    raise ValueError("WikiText documents do not contain the published reference prefix")


def document_windows(
    documents: Sequence[WikiDocument],
    *,
    tokenizer: Any,
    collection: str,
    count: int,
    context_length: int,
    minimum_tokens: int,
) -> tuple[list[dict[str, Any]], int]:
    """Take at most one bounded token window from each successive article."""

    if collection not in ("activation_fit", "candidate_selection"):
        raise ValueError("unsupported corpus-plan collection")
    if count < 1 or minimum_tokens < 1 or context_length < minimum_tokens:
        raise ValueError("window counts and token bounds must be positive and ordered")
    windows: list[dict[str, Any]] = []
    consumed = 0
    for document in documents:
        consumed += 1
        token_ids = tokenizer(
            document.text,
            add_special_tokens=False,
            truncation=True,
            max_length=context_length,
        )["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        token_ids = [int(value) for value in token_ids]
        if len(token_ids) < minimum_tokens:
            continue
        windows.append(
            {
                "window_id": (
                    f"{collection}-wikitext-rows-"
                    f"{document.row_start:05d}-{document.row_stop:05d}"
                ),
                "document": {
                    "title": document.title,
                    "row_start_inclusive": document.row_start,
                    "row_stop_exclusive": document.row_stop,
                    "text_sha256": hashlib.sha256(
                        document.text.encode("utf-8")
                    ).hexdigest(),
                },
                "token_count": len(token_ids),
                "token_first16": token_ids[:16],
                "token_ids": token_ids,
            }
        )
        if len(windows) == count:
            return windows, consumed
    raise ValueError(
        f"only {len(windows)} documents met the {collection} token requirement; "
        f"need {count}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--fit-windows", type=int, default=32)
    parser.add_argument("--selection-windows", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--minimum-tokens", type=int, default=1024)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dependency_path = os.getenv("KLD_PYDEPS")
    if dependency_path:
        import sys

        sys.path.append(dependency_path)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    reference_manifest = json.loads(args.reference_manifest.read_text())
    if int(reference_manifest["context_length"]) != args.context_length:
        raise ValueError("reference and corpus-plan context lengths differ")
    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="test"
    )
    rows = [str(row["text"]) for row in dataset]
    documents = group_wikitext_documents(rows)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    reference_text = "\n\n".join(text for text in rows if text.strip())
    reference_ids = tokenizer(
        reference_text,
        add_special_tokens=False,
        truncation=True,
        max_length=args.context_length,
    )["input_ids"]
    if reference_ids[:16] != reference_manifest["token_first16"]:
        raise ValueError("cached dataset and tokenizer do not reproduce the reference")
    overlap_count = reference_document_count(
        documents,
        tokenizer=tokenizer,
        reference_tokens=args.context_length,
    )
    remaining = documents[overlap_count:]
    fit, fit_consumed = document_windows(
        remaining,
        tokenizer=tokenizer,
        collection="activation_fit",
        count=args.fit_windows,
        context_length=args.context_length,
        minimum_tokens=args.minimum_tokens,
    )
    selection, selection_consumed = document_windows(
        remaining[fit_consumed:],
        tokenizer=tokenizer,
        collection="candidate_selection",
        count=args.selection_windows,
        context_length=args.context_length,
        minimum_tokens=args.minimum_tokens,
    )
    fit_rows = {
        row
        for window in fit
        for row in range(
            window["document"]["row_start_inclusive"],
            window["document"]["row_stop_exclusive"],
        )
    }
    selection_rows = {
        row
        for window in selection
        for row in range(
            window["document"]["row_start_inclusive"],
            window["document"]["row_stop_exclusive"],
        )
    }
    reference_rows = set()
    for document in documents[:overlap_count]:
        reference_rows.update(range(document.row_start, document.row_stop))
    if fit_rows & selection_rows or fit_rows & reference_rows or selection_rows & reference_rows:
        raise AssertionError("corpus-plan document separation failed")
    plan = {
        "schema": "qsrt_glm52_document_disjoint_corpus_plan",
        "schema_version": 1,
        "dataset": {
            "id": "Salesforce/wikitext",
            "configuration": "wikitext-2-raw-v1",
            "split": "test",
            "row_count": len(rows),
            "offline_cache_required": True,
        },
        "tokenizer": {
            "path": str(args.model.resolve()),
            "local_files_only": True,
        },
        "context_length": args.context_length,
        "minimum_tokens": args.minimum_tokens,
        "published_bf16_reference": {
            "manifest_path": str(args.reference_manifest.resolve()),
            "manifest_sha256": hashlib.sha256(
                args.reference_manifest.read_bytes()
            ).hexdigest(),
            "token_first16": reference_manifest["token_first16"],
            "overlapping_document_count": overlap_count,
            "overlapping_row_ranges": [
                [document.row_start, document.row_stop]
                for document in documents[:overlap_count]
            ],
            "role": "untouched BF16-reference reporting context",
        },
        "activation_fit": {
            "role": "fit curvature estimates and reconstructed-activation refits",
            "window_count": len(fit),
            "windows": fit,
        },
        "candidate_selection": {
            "role": "select candidates without consulting the reporting context",
            "window_count": len(selection),
            "windows": selection,
        },
        "separation": {
            "unit": "WikiText article delimited by a top-level heading",
            "reference_fit_row_overlap": 0,
            "reference_selection_row_overlap": 0,
            "fit_selection_row_overlap": 0,
        },
        "model_downloads_required": False,
    }
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.dest.with_name(f".{args.dest.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.dest)
    print(
        json.dumps(
            {
                "dest": str(args.dest),
                "document_count": len(documents),
                "reference_overlap_documents": overlap_count,
                "fit_windows": len(fit),
                "selection_windows": len(selection),
                "sha256": hashlib.sha256(args.dest.read_bytes()).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

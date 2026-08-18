from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qsrt.kimi_capture_documents import load_corpus_document_index


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize and not add_generation_prompt
        return [len(str(message["content"])) for message in messages]


def _hash_content(raw: str) -> str:
    return hashlib.blake2b(raw.encode(), digest_size=16).hexdigest()


def _hash_prompt(tokens: list[int]) -> str:
    value = b"".join(token.to_bytes(4, "little") for token in tokens)
    return hashlib.blake2b(value, digest_size=16).hexdigest()


def _report(tmp_path: Path) -> Path:
    source = tmp_path / "source.jsonl"
    rows = [
        json.dumps({"prompt": "abc"}, separators=(",", ":")),
        json.dumps(
            {"messages": [{"role": "user", "content": "hello"}]},
            separators=(",", ":"),
        ),
    ]
    source.write_text("\n".join(rows) + "\n")
    prompts = [[97, 98, 99], [5]]
    documents = [
        {
            "source": str(source.resolve()),
            "line": index + 1,
            "document_hash": _hash_content(row),
            "prompt_hash": _hash_prompt(tokens),
            "tokens": len(tokens),
        }
        for index, (row, tokens) in enumerate(zip(rows, prompts, strict=True))
    ]
    report = {
        "kind": "qsrt_interim_calibration_corpus_run",
        "schema_version": 1,
        "expected_capture_source": "test",
        "model_dir": "/model",
        "capture_dir": "/capture",
        "sources": [
            {
                "path": str(source.resolve()),
                "weight": 1.0,
                "max_prompt_tokens": 10,
                "record_source": None,
            }
        ],
        "fold": {"modulus": 2, "index": 0, "mode": "include"},
        "excluded_corpus_reports": [],
        "excluded_token_files": [],
        "excluded_document_hashes": [],
        "excluded_prompt_hashes": [],
        "seed": 7,
        "target_tokens": 4,
        "planned_tokens": 4,
        "planned_requests": 2,
        "documents": documents,
        "completed_requests": 2,
        "reported_prompt_tokens": 4,
        "finalized": True,
    }
    fields = (
        "kind",
        "schema_version",
        "expected_capture_source",
        "model_dir",
        "capture_dir",
        "sources",
        "fold",
        "excluded_corpus_reports",
        "excluded_token_files",
        "excluded_document_hashes",
        "excluded_prompt_hashes",
        "seed",
        "target_tokens",
        "planned_tokens",
        "planned_requests",
        "documents",
    )
    immutable = {field: report[field] for field in fields}
    report["plan_sha256"] = hashlib.sha256(
        json.dumps(immutable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    return path


def test_load_corpus_document_index_reproduces_report(tmp_path: Path) -> None:
    documents = load_corpus_document_index(_report(tmp_path), _Tokenizer())
    assert documents.input_ids.tolist() == [97, 98, 99, 5]
    assert documents.offsets.tolist() == [0, 3, 4]
    assert len(documents.identifiers) == 2


def test_load_corpus_document_index_rejects_changed_source(tmp_path: Path) -> None:
    report = _report(tmp_path)
    source = tmp_path / "source.jsonl"
    source.write_text(source.read_text().replace("abc", "abd"))
    with pytest.raises(ValueError, match="content hash differs"):
        load_corpus_document_index(report, _Tokenizer())

"""Authenticate and retokenize a QSRT corpus report for Kimi replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Protocol

import torch

from qsrt.kimi_boundary_slabs import DocumentIndex


SUPPORTED_REPORT_KINDS = {
    "qsrt_interim_calibration_corpus_run",
    "kquant_interim_calibration_corpus_run",
}
_PLAN_FIELDS = (
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


class TokenizerLike(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]: ...


def _content_hash(raw: str) -> str:
    return hashlib.blake2b(raw.encode(), digest_size=16).hexdigest()


def _prompt_hash(tokens: list[int]) -> str:
    encoded = b"".join(int(token).to_bytes(4, "little") for token in tokens)
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def _record_tokens(row: dict[str, Any], tokenizer: TokenizerLike) -> list[int]:
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        return list(tokenizer.encode(prompt, add_special_tokens=False))
    messages = row.get("messages")
    text = row.get("text")
    if not isinstance(messages, list) and isinstance(text, str):
        try:
            nested = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError("record text is not a valid serialized chat") from error
        if isinstance(nested, dict):
            messages = nested.get("messages")
    if not isinstance(messages, list):
        messages = row.get("conversations")
    if not isinstance(messages, list) or not messages:
        raise ValueError("record has no prompt or message sequence")
    if any(not isinstance(message, dict) for message in messages):
        raise ValueError("record messages must be JSON objects")
    return list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )


def _validate_plan_hash(report: dict[str, Any], path: Path) -> None:
    if "plan_sha256" not in report:
        raise ValueError(f"{path}: corpus report has no plan_sha256")
    immutable = {field: report[field] for field in _PLAN_FIELDS}
    observed = hashlib.sha256(
        json.dumps(immutable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if observed != report["plan_sha256"]:
        raise ValueError(f"{path}: corpus plan SHA-256 does not close")


def load_corpus_document_index(
    report_path: str | Path,
    tokenizer: TokenizerLike,
) -> DocumentIndex:
    """Reconstruct the report's exact token sequences from authenticated lines."""

    path = Path(report_path).expanduser().resolve()
    report = json.loads(path.read_text())
    if not isinstance(report, dict) or report.get("kind") not in SUPPORTED_REPORT_KINDS:
        raise ValueError(f"{path}: unsupported corpus report")
    if report.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported corpus report schema")
    if report.get("finalized") is not True:
        raise ValueError(f"{path}: corpus report is not finalized")
    _validate_plan_hash(report, path)

    documents = report.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"{path}: corpus report has no documents")
    planned_requests = int(report["planned_requests"])
    planned_tokens = int(report["planned_tokens"])
    if (
        len(documents) != planned_requests
        or int(report["completed_requests"]) != planned_requests
        or int(report["reported_prompt_tokens"]) != planned_tokens
    ):
        raise ValueError(f"{path}: completed corpus counts do not close")

    source_specs = report.get("sources")
    if not isinstance(source_specs, list):
        raise ValueError(f"{path}: corpus report has no source inventory")
    source_filters = {
        Path(str(source["path"])).expanduser().resolve(): source.get("record_source")
        for source in source_specs
        if isinstance(source, dict)
    }
    requested: dict[Path, dict[int, int]] = {}
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise TypeError(f"{path}: document {index} is not a JSON object")
        source = Path(str(document["source"])).expanduser().resolve()
        if source not in source_filters:
            raise ValueError(f"{path}: document source is absent from source inventory")
        line = int(document["line"])
        if line <= 0 or line in requested.setdefault(source, {}):
            raise ValueError(f"{path}: duplicate or invalid source line {source}:{line}")
        requested[source][line] = index

    raw_records: list[str | None] = [None] * len(documents)
    for source, lines in requested.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        remaining = set(lines)
        with source.open(encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, 1):
                index = lines.get(line_number)
                if index is not None:
                    raw_records[index] = raw.strip()
                    remaining.remove(line_number)
                    if not remaining:
                        break
        if remaining:
            raise ValueError(
                f"{source}: missing requested source lines {sorted(remaining)[:8]}"
            )

    token_sequences: list[torch.Tensor] = []
    offsets = [0]
    identifiers: list[str] = []
    for index, (document, raw) in enumerate(zip(documents, raw_records, strict=True)):
        assert isinstance(document, dict) and raw is not None
        if _content_hash(raw) != document["document_hash"]:
            raise ValueError(f"{path}: document {index} content hash differs")
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise TypeError(f"{path}: document {index} source row is not an object")
        source = Path(str(document["source"])).expanduser().resolve()
        record_source = source_filters[source]
        if record_source is not None and row.get("source") != record_source:
            raise ValueError(f"{path}: document {index} source filter differs")
        expected_tokens = int(document["tokens"])
        tokens = _record_tokens(row, tokenizer)
        if expected_tokens <= 0 or len(tokens) < expected_tokens:
            raise ValueError(f"{path}: document {index} token count cannot be reproduced")
        tokens = tokens[:expected_tokens]
        if _prompt_hash(tokens) != document["prompt_hash"]:
            raise ValueError(f"{path}: document {index} prompt hash differs")
        tensor = torch.tensor(tokens, dtype=torch.int32)
        token_sequences.append(tensor)
        offsets.append(offsets[-1] + expected_tokens)
        identifiers.append(f"{document['document_hash']}:{document['prompt_hash']}")

    if offsets[-1] != planned_tokens:
        raise ValueError(f"{path}: reconstructed token total does not close")
    return DocumentIndex(
        input_ids=torch.cat(token_sequences),
        offsets=torch.tensor(offsets, dtype=torch.int64),
        identifiers=tuple(identifiers),
    )


def load_token_suite_document_index(
    suite_dir: str | Path,
    context_indices: Iterable[int],
) -> DocumentIndex:
    """Load stored token IDs from a distribution-fidelity suite.

    The suite already defines exact model input IDs. Reading them directly
    avoids retokenization and preserves the screening population byte for
    byte. Every token file is authenticated against ``suite-manifest.json``.
    """

    root = Path(suite_dir).expanduser().resolve()
    manifest_path = root / "suite-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read token-suite manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object")
    contexts = manifest.get("contexts")
    index_field = "context_index"
    if contexts is None:
        contexts = manifest.get("windows")
        index_field = "index"
    if not isinstance(contexts, list) or not contexts:
        raise ValueError(f"{manifest_path} has no contexts or windows")
    by_index: dict[int, dict[str, Any]] = {}
    for item in contexts:
        if not isinstance(item, dict) or index_field not in item:
            raise ValueError(f"{manifest_path} contains an invalid context")
        index = int(item[index_field])
        if index in by_index:
            raise ValueError(f"{manifest_path} repeats context {index}")
        by_index[index] = item

    requested = tuple(int(value) for value in context_indices)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("token-suite context indices must be nonempty and unique")
    token_sequences: list[torch.Tensor] = []
    identifiers: list[str] = []
    offsets = [0]
    for index in requested:
        try:
            item = by_index[index]
        except KeyError as error:
            raise IndexError(f"token suite has no context {index}") from error
        token_file = item.get("token_file")
        if not isinstance(token_file, str):
            raise ValueError(f"token-suite context {index} has no token file")
        path = root / token_file
        raw = path.read_bytes()
        values = json.loads(raw)
        if not isinstance(values, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values
        ):
            raise ValueError(f"token-suite context {index} has invalid token IDs")
        canonical = json.dumps(values, separators=(",", ":")).encode("utf-8")
        expected_hash = item.get("token_ids_json_sha256")
        if (
            not isinstance(expected_hash, str)
            or hashlib.sha256(canonical).hexdigest() != expected_hash
        ):
            raise ValueError(f"token-suite context {index} failed token-file authentication")
        expected_tokens = int(item["num_tokens"])
        if len(values) != expected_tokens:
            raise ValueError(
                f"token-suite context {index} has {len(values)} tokens, "
                f"expected {expected_tokens}"
            )
        token_sequences.append(torch.tensor(values, dtype=torch.int32))
        offsets.append(offsets[-1] + expected_tokens)
        identifiers.append(
            f"distribution-fidelity:{index}:{expected_hash}"
        )
    return DocumentIndex(
        input_ids=torch.cat(token_sequences),
        offsets=torch.tensor(offsets, dtype=torch.int64),
        identifiers=tuple(identifiers),
    )


__all__ = [
    "TokenizerLike",
    "load_corpus_document_index",
    "load_token_suite_document_index",
]

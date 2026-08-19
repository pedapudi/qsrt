"""Define a document-disjoint GLM-5.2 teacher-reference plan.

The canonical Hessian archive contains the BF16 hidden state after decoder
layer 77.  Applying the official final normalization and language-model head
to selected rows produces source-model logits without loading the preceding
decoder layers.  This module freezes which documents may enter screening and
confirmation before those logits are generated.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from qsrt.glm52_document_disjoint_confirmation import token_ids_sha256


PLAN_SCHEMA = "qsrt_glm52_terminal_hidden_teacher_reference_plan"
MODEL_ID = "zai-org/GLM-5.2"
SOURCE_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
EARLIER_TEACHER_REVISION = "4d67f66cc64d3219133b767c253b2ad1425c6c88"
CANONICAL_DATASET_ID = (
    "brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians-Canonical"
)
CANONICAL_DATASET_REVISION = "1be5d12221bf885a5568a431562041ce2e073352"
CANONICAL_DATASET_PUBLISHED_TAG = "canonical-v1"
CANONICAL_DOCUMENT_PLAN_SHA256 = (
    "f1d3bae9c1ab6cc52fc3d98cd76ef94a85391efb698d86efaf39beeab052b2a8"
)
CALIBRATION_CORPUS_SHA256 = (
    "cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4"
)
CALIBRATION_CORPUS_BYTES = 34_002_059
CANONICAL_DOCUMENT_PLAN_BYTES = 811_558
TERMINAL_HIDDEN_SHA256 = (
    "68548c3f76feb4e568c6c282ddb60779e7ef2574b828697f8b75407078186105"
)
TERMINAL_HIDDEN_ROWS = 1_049_589
HIDDEN_SIZE = 6_144
VOCABULARY_SIZE = 154_880
TERMINAL_HIDDEN_BYTES = TERMINAL_HIDDEN_ROWS * HIDDEN_SIZE * 2
MINIMUM_CONTEXT_TOKENS = 64
MAXIMUM_CONTEXT_TOKENS = 2_048
SCREENING_DOCUMENTS_PER_AXIS = 2
CONFIRMATION_DOCUMENT_COUNT = 32
AXES = (
    "axis1_general",
    "axis2_legal",
    "axis3_code_agentic",
    "axis4_reasoning_termination",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_assets() -> dict[str, Any]:
    return {
        "terminal_hidden": {
            "repository": CANONICAL_DATASET_ID,
            "revision": CANONICAL_DATASET_REVISION,
            "published_tag": CANONICAL_DATASET_PUBLISHED_TAG,
            "path": "raw_capture/terminal_layer_077/hidden.bf16.bin",
            "sha256": TERMINAL_HIDDEN_SHA256,
            "bytes": TERMINAL_HIDDEN_BYTES,
            "dtype": "bfloat16-little-endian",
            "shape": [TERMINAL_HIDDEN_ROWS, HIDDEN_SIZE],
            "semantics": "decoder layer-77 output before the final RMS normalization",
        },
        "language_model_head": {
            "repository": MODEL_ID,
            "revision": SOURCE_REVISION,
            "source_shard": "model-00001-of-00282.safetensors",
            "source_shard_sha256": (
                "004bf9404964da8ea71ea2d3ebf02148fa766b956bd4fca3f54b093e58a6a74c"
            ),
            "safetensors_header_bytes": 4_128,
            "tensor": "lm_head.weight",
            "source_byte_start": 4_136,
            "source_byte_stop_exclusive": 1_903_169_576,
            "extracted_bytes": 1_903_165_440,
            "dtype": "BF16",
            "shape": [VOCABULARY_SIZE, HIDDEN_SIZE],
            "extracted_file": "lm_head.weight.bf16.bin",
        },
        "final_normalization": {
            "repository": MODEL_ID,
            "revision": SOURCE_REVISION,
            "source_shard": "model-00282-of-00282.safetensors",
            "source_shard_sha256": (
                "46c23d3a25db83ab8b7e47d0a5efec49e00f57a815358323f1ab45136de327d7"
            ),
            "safetensors_header_bytes": 552,
            "tensor": "model.norm.weight",
            "source_byte_start": 293_605_936,
            "source_byte_stop_exclusive": 293_618_224,
            "extracted_bytes": 12_288,
            "dtype": "BF16",
            "shape": [HIDDEN_SIZE],
            "epsilon": 1e-5,
            "extracted_file": "model.norm.weight.bf16.bin",
        },
    }


def _load_corpus_lines(corpus: bytes) -> list[dict[str, Any]]:
    if sha256_bytes(corpus) != CALIBRATION_CORPUS_SHA256:
        raise ValueError("calibration corpus SHA-256 differs")
    rows = []
    for line_number, line in enumerate(corpus.decode("utf-8").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"calibration corpus line {line_number} is invalid") from error
        if not isinstance(value, dict) or not isinstance(value.get("text"), str):
            raise ValueError(f"calibration corpus line {line_number} has no text")
        rows.append(value)
    if len(rows) != 12_228:
        raise ValueError("calibration corpus row count differs")
    return rows


def _validate_capture_plan(capture_plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if (
        capture_plan.get("schema") != "glm52-r10-sqg-document-plan-v1"
        or capture_plan.get("documents_total") != 1_773
        or capture_plan.get("tokens_total") != TERMINAL_HIDDEN_ROWS
    ):
        raise ValueError("canonical capture document plan identity differs")
    documents = capture_plan.get("documents")
    if not isinstance(documents, list) or len(documents) != 1_773:
        raise ValueError("canonical capture document list differs")
    return documents


def build_terminal_teacher_reference_plan(
    *,
    capture_plan_bytes: bytes,
    corpus_bytes: bytes,
    frozen_at_utc: str,
) -> dict[str, Any]:
    """Select forty untouched documents using identities that precede logits."""

    if sha256_bytes(capture_plan_bytes) != CANONICAL_DOCUMENT_PLAN_SHA256:
        raise ValueError("canonical capture document plan SHA-256 differs")
    capture_plan = json.loads(capture_plan_bytes)
    documents = _validate_capture_plan(capture_plan)
    corpus = _load_corpus_lines(corpus_bytes)
    if not isinstance(frozen_at_utc, str) or not frozen_at_utc:
        raise ValueError("reference plan needs a freeze time")

    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        if document.get("role") != "holdout":
            continue
        source_tokens = document.get("tokens")
        if (
            isinstance(source_tokens, bool)
            or not isinstance(source_tokens, int)
            or source_tokens < MINIMUM_CONTEXT_TOKENS
        ):
            continue
        corpus_line = document.get("corpus_line")
        if (
            isinstance(corpus_line, bool)
            or not isinstance(corpus_line, int)
            or not 0 <= corpus_line < len(corpus)
        ):
            raise ValueError("capture document has an invalid corpus line")
        source = corpus[corpus_line]
        text = source["text"]
        document_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if document_sha256 != document.get("document_sha256"):
            raise ValueError("capture document text SHA-256 differs")
        axis = source.get("axis")
        if axis not in AXES:
            raise ValueError("capture document has an unsupported corpus axis")
        context_tokens = min(source_tokens, MAXIMUM_CONTEXT_TOKENS)
        hidden_row_start = int(document["global_row_start"])
        eligible[axis].append(
            {
                "axis": axis,
                "source": str(source.get("source")),
                "document_sha256": document_sha256,
                "document_epoch": int(document["epoch"]),
                "corpus_line": corpus_line,
                "source_document_tokens": source_tokens,
                "context_tokens": context_tokens,
                "logit_rows": context_tokens - 1,
                "full_token_ids_sha256_u32le": document[
                    "token_ids_sha256_u32le"
                ],
                "terminal_hidden_row_start": hidden_row_start,
                "terminal_hidden_row_stop_exclusive": (
                    hidden_row_start + context_tokens - 1
                ),
                "expected_logits_shape": [context_tokens - 1, VOCABULARY_SIZE],
            }
        )

    all_eligible = [row for axis in AXES for row in eligible[axis]]
    confirmation = sorted(
        all_eligible,
        key=lambda row: (
            -row["context_tokens"],
            row["document_sha256"],
            row["document_epoch"],
        ),
    )[:CONFIRMATION_DOCUMENT_COUNT]
    if len(confirmation) != CONFIRMATION_DOCUMENT_COUNT:
        raise ValueError("canonical capture lacks thirty-two holdout documents")
    confirmation_hashes = {row["document_sha256"] for row in confirmation}

    screening: list[dict[str, Any]] = []
    for axis in AXES:
        remaining = sorted(
            (
                row
                for row in eligible[axis]
                if row["document_sha256"] not in confirmation_hashes
            ),
            key=lambda row: (
                -row["context_tokens"],
                row["document_sha256"],
                row["document_epoch"],
            ),
        )
        if len(remaining) < SCREENING_DOCUMENTS_PER_AXIS:
            raise ValueError(f"axis {axis} lacks two unused screening documents")
        screening.extend(remaining[:SCREENING_DOCUMENTS_PER_AXIS])

    selected: list[dict[str, Any]] = []
    index_within_axis: dict[tuple[str, str], int] = defaultdict(int)
    for tier, rows in (("screening", screening), ("confirmation", confirmation)):
        for row in rows:
            axis = row["axis"]
            tier_index = index_within_axis[(tier, axis)]
            index_within_axis[(tier, axis)] += 1
            selected.append(
                {
                    **row,
                    "evaluation_tier": tier,
                    "tier_index_within_axis": tier_index,
                    "reference_file": (
                        f"{tier}-{axis.replace('_', '-')}-{tier_index:02d}.safetensors"
                    ),
                }
            )

    confirmation_axis_counts = {
        axis: sum(row["axis"] == axis for row in confirmation) for axis in AXES
    }

    return {
        "schema": PLAN_SCHEMA,
        "schema_version": 1,
        "status": "frozen_before_teacher_reference_generation",
        "frozen_at_utc": frozen_at_utc,
        "teacher": {
            "model_id": MODEL_ID,
            "source_revision": SOURCE_REVISION,
            "earlier_reference_revision": EARLIER_TEACHER_REVISION,
            "weight_identity": (
                "all 282 safetensors objects and the tensor index are byte-identical"
            ),
            "logit_equation": "lm_head(final_rms_norm(layer_77_output))",
        },
        "sources": {
            "canonical_dataset": {
                "id": CANONICAL_DATASET_ID,
                "revision": CANONICAL_DATASET_REVISION,
                "published_tag": CANONICAL_DATASET_PUBLISHED_TAG,
                "document_plan_path": "capture_view/document_plan.json",
                "document_plan_sha256": CANONICAL_DOCUMENT_PLAN_SHA256,
                "document_plan_bytes": CANONICAL_DOCUMENT_PLAN_BYTES,
            },
            "calibration_corpus": {
                "repository": "brandonmusic/GLM-5.2-NVFP4-REAP-Recall-N172",
                "revision": "52d000a10666178fc648e806cb240a8106c061a9",
                "path": "calibration data/reap_recall_calib.jsonl",
                "sha256": CALIBRATION_CORPUS_SHA256,
                "bytes": CALIBRATION_CORPUS_BYTES,
                "rows": 12_228,
            },
        },
        "reference_assets": _reference_assets(),
        "selection_rule": {
            "eligible_role": "holdout",
            "minimum_context_tokens": MINIMUM_CONTEXT_TOKENS,
            "maximum_context_tokens": MAXIMUM_CONTEXT_TOKENS,
            "confirmation_ordering": (
                "descending available context length across holdout documents, "
                "then ascending document SHA-256"
            ),
            "screening_ordering": (
                "after confirmation documents are excluded, descending available "
                "context length and then ascending document SHA-256 within each axis"
            ),
            "screening_documents_per_axis": SCREENING_DOCUMENTS_PER_AXIS,
            "confirmation_document_count": CONFIRMATION_DOCUMENT_COUNT,
            "confirmation_axis_counts": confirmation_axis_counts,
            "axes": list(AXES),
            "selection_uses_model_output": False,
        },
        "screening_document_count": SCREENING_DOCUMENTS_PER_AXIS * len(AXES),
        "confirmation_document_count": CONFIRMATION_DOCUMENT_COUNT,
        "documents": selected,
        "evidence_boundary": (
            "The eight screening documents may select among already-built candidates. "
            "The thirty-two confirmation documents remain sealed until candidate "
            "construction, selection, factor dtype, and exact charged bytes are frozen."
        ),
    }


def validate_terminal_teacher_reference_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate plan structure without opening its source corpus or logits."""

    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("schema_version") != 1
        or plan.get("status") != "frozen_before_teacher_reference_generation"
    ):
        raise ValueError("terminal teacher-reference plan identity differs")
    if plan.get("reference_assets") != _reference_assets():
        raise ValueError("terminal teacher-reference asset contract differs")
    rule = plan.get("selection_rule")
    if not isinstance(rule, Mapping) or rule.get("axes") != list(AXES):
        raise ValueError("terminal teacher-reference selection rule differs")
    rows = plan.get("documents")
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError("terminal teacher-reference plan must contain forty documents")

    seen_documents: set[str] = set()
    seen_files: set[str] = set()
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("terminal teacher-reference document must be an object")
        axis = row.get("axis")
        tier = row.get("evaluation_tier")
        document_sha256 = row.get("document_sha256")
        reference_file = row.get("reference_file")
        context_tokens = row.get("context_tokens")
        start = row.get("terminal_hidden_row_start")
        stop = row.get("terminal_hidden_row_stop_exclusive")
        if axis not in AXES or tier not in {"screening", "confirmation"}:
            raise ValueError("terminal teacher-reference tier or axis differs")
        if (
            not isinstance(document_sha256, str)
            or len(document_sha256) != 64
            or document_sha256 in seen_documents
        ):
            raise ValueError("terminal teacher-reference document identity differs")
        if (
            not isinstance(reference_file, str)
            or Path(reference_file).name != reference_file
            or reference_file in seen_files
        ):
            raise ValueError("terminal teacher-reference filename differs")
        if (
            isinstance(context_tokens, bool)
            or not isinstance(context_tokens, int)
            or not MINIMUM_CONTEXT_TOKENS <= context_tokens <= MAXIMUM_CONTEXT_TOKENS
        ):
            raise ValueError("terminal teacher-reference context length differs")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(stop, bool)
            or not isinstance(stop, int)
            or stop - start != context_tokens - 1
            or not 0 <= start < stop <= TERMINAL_HIDDEN_ROWS
        ):
            raise ValueError("terminal teacher-reference hidden-row range differs")
        if row.get("expected_logits_shape") != [
            context_tokens - 1,
            VOCABULARY_SIZE,
        ]:
            raise ValueError("terminal teacher-reference logit shape differs")
        seen_documents.add(document_sha256)
        seen_files.add(reference_file)
        counts[(tier, axis)] += 1

    for axis in AXES:
        if counts[("screening", axis)] != SCREENING_DOCUMENTS_PER_AXIS:
            raise ValueError("screening document count differs")
        expected_confirmation = rule["confirmation_axis_counts"].get(axis)
        if counts[("confirmation", axis)] != expected_confirmation:
            raise ValueError("confirmation axis count differs")
    if sum(counts[("confirmation", axis)] for axis in AXES) != 32:
        raise ValueError("confirmation document count differs")
    return {
        "document_count": len(rows),
        "screening_document_count": 8,
        "confirmation_document_count": 32,
        "maximum_context_tokens": max(row["context_tokens"] for row in rows),
        "total_logit_rows": sum(row["logit_rows"] for row in rows),
    }


def validate_terminal_teacher_reference_sources(
    *, plan: Mapping[str, Any], capture_plan_bytes: bytes, corpus_bytes: bytes
) -> dict[str, Any]:
    """Rebuild the frozen plan from immutable source metadata and compare it."""

    validate_terminal_teacher_reference_plan(plan)
    rebuilt = build_terminal_teacher_reference_plan(
        capture_plan_bytes=capture_plan_bytes,
        corpus_bytes=corpus_bytes,
        frozen_at_utc=str(plan["frozen_at_utc"]),
    )
    if rebuilt != dict(plan):
        raise ValueError("terminal teacher-reference plan differs from its sources")
    return validate_terminal_teacher_reference_plan(plan)


def validate_document_tokenization(
    *, row: Mapping[str, Any], text: str, token_ids: Sequence[int]
) -> dict[str, Any]:
    """Verify one source document and return its frozen prompt-token receipt."""

    if hashlib.sha256(text.encode("utf-8")).hexdigest() != row.get(
        "document_sha256"
    ):
        raise ValueError("terminal teacher-reference document text differs")
    normalized = [int(value) for value in token_ids]
    if (
        len(normalized) != row.get("source_document_tokens")
        or token_ids_sha256(normalized)
        != row.get("full_token_ids_sha256_u32le")
    ):
        raise ValueError("terminal teacher-reference tokenization differs")
    context_tokens = int(row["context_tokens"])
    prompt = normalized[:context_tokens]
    return {
        "prompt_token_ids": prompt,
        "prompt_token_ids_sha256_u32le": token_ids_sha256(prompt),
        "target_token_ids": prompt[1:],
        "context_tokens": context_tokens,
        "logit_rows": context_tokens - 1,
    }


__all__ = [
    "AXES",
    "HIDDEN_SIZE",
    "PLAN_SCHEMA",
    "TERMINAL_HIDDEN_ROWS",
    "VOCABULARY_SIZE",
    "build_terminal_teacher_reference_plan",
    "sha256_file",
    "validate_document_tokenization",
    "validate_terminal_teacher_reference_plan",
    "validate_terminal_teacher_reference_sources",
]

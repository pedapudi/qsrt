from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import pytest
import torch

import qsrt.glm52_terminal_teacher_logits as terminal_logits
from qsrt.glm52_document_disjoint_confirmation import token_ids_sha256
from qsrt.glm52_terminal_teacher_logits import (
    balanced_vocabulary_slices,
    build_terminal_teacher_logit_generation_contract,
    compute_terminal_teacher_logits,
    exact_glm52_final_rms_norm,
    load_validated_terminal_teacher_reference_documents,
    selected_document_token_receipts,
    tokenizer_file_identity,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    ROOT / "experiments/glm52_terminal_hidden_teacher_reference_plan.json"
)


def test_exact_final_rms_norm_matches_official_glm_arithmetic() -> None:
    hidden = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0], [0.5, 0.25, -0.75, 1.5]],
        dtype=torch.bfloat16,
    )
    weight = torch.tensor([1.0, 0.5, 2.0, -1.0], dtype=torch.bfloat16)

    actual = exact_glm52_final_rms_norm(hidden, weight, epsilon=1e-5)
    float_hidden = hidden.float()
    expected = weight * (
        float_hidden
        * torch.rsqrt(float_hidden.pow(2).mean(-1, keepdim=True) + 1e-5)
    ).to(torch.bfloat16)

    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)


def test_vocabulary_shards_are_contiguous_and_balanced() -> None:
    assert balanced_vocabulary_slices(154_880, 4) == (
        (0, 38_720),
        (38_720, 77_440),
        (77_440, 116_160),
        (116_160, 154_880),
    )
    uneven = balanced_vocabulary_slices(11, 3)
    assert uneven == ((0, 4), (4, 8), (8, 11))
    assert max(stop - start for start, stop in uneven) - min(
        stop - start for start, stop in uneven
    ) == 1


def test_sharded_endpoint_matches_one_full_bf16_matrix_multiply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden = torch.tensor(
        [[0.5, -1.0, 1.5, 2.0], [-0.75, 0.25, 1.0, -1.5]],
        dtype=torch.bfloat16,
    )
    normalization = torch.tensor(
        [1.0, 0.5, 1.5, -0.75], dtype=torch.bfloat16
    )
    head = torch.tensor(
        [
            [0.25, 0.5, -0.75, 1.0],
            [-1.0, 0.25, 0.5, -0.5],
            [0.75, -0.25, 1.0, 0.125],
            [0.5, 0.5, 0.5, 0.5],
            [-0.125, 0.25, -0.5, 1.0],
            [1.0, -1.0, 0.5, -0.25],
        ],
        dtype=torch.bfloat16,
    )
    monkeypatch.setattr(terminal_logits, "HIDDEN_SIZE", 4)
    monkeypatch.setattr(terminal_logits, "VOCABULARY_SIZE", 6)
    monkeypatch.setattr(torch.cuda, "device", lambda unused: nullcontext())
    endpoint_shards = [
        {
            "device": torch.device("cpu"),
            "head": head[:3],
            "normalization": normalization,
        },
        {
            "device": torch.device("cpu"),
            "head": head[3:],
            "normalization": normalization,
        },
    ]

    actual, closure = compute_terminal_teacher_logits(
        hidden_cpu=hidden,
        endpoint_shards=endpoint_shards,
        epsilon=1e-5,
        closure_rows=2,
        capture_closure=True,
    )
    normalized = exact_glm52_final_rms_norm(
        hidden, normalization, epsilon=1e-5
    )
    expected = torch.nn.functional.linear(normalized, head)

    assert torch.equal(actual, expected)
    assert closure is not None
    assert closure["rows"] == 2
    assert closure["bf16_repeat_bit_exact"] is True
    assert closure["mean_forward_kld_fp32_head_to_bf16_head"] >= 0.0


def test_generation_contract_separates_screening_from_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = PLAN_PATH.read_bytes()
    plan = json.loads(raw)
    monkeypatch.setattr(
        terminal_logits,
        "_runtime_identity",
        lambda devices: {"devices": [str(device) for device in devices]},
    )
    devices = [torch.device(f"cuda:{index}") for index in range(4)]

    screening = build_terminal_teacher_logit_generation_contract(
        plan=plan,
        plan_sha256=hashlib.sha256(raw).hexdigest(),
        asset_complete_sha256="a" * 64,
        tokenizer_identity={"files": {"tokenizer.json": {"sha256": "b" * 64}}},
        evaluation_tier="screening",
        devices=devices,
        closure_rows=8,
    )
    confirmation = build_terminal_teacher_logit_generation_contract(
        plan=plan,
        plan_sha256=hashlib.sha256(raw).hexdigest(),
        asset_complete_sha256="a" * 64,
        tokenizer_identity={"files": {"tokenizer.json": {"sha256": "b" * 64}}},
        evaluation_tier="confirmation",
        devices=devices,
        closure_rows=8,
        confirmation_authorization={
            "artifact_manifest_sha256": "c" * 64,
            "candidate_runtime_mode": "candidate",
        },
    )

    assert screening["document_count"] == 8
    assert confirmation["document_count"] == 32
    assert confirmation["confirmation_authorization"] == {
        "artifact_manifest_sha256": "c" * 64,
        "candidate_runtime_mode": "candidate",
    }
    assert screening["vocabulary_slices"] == [
        [0, 38_720],
        [38_720, 77_440],
        [77_440, 116_160],
        [116_160, 154_880],
    ]
    with pytest.raises(PermissionError, match="frozen authorization"):
        build_terminal_teacher_logit_generation_contract(
            plan=plan,
            plan_sha256=hashlib.sha256(raw).hexdigest(),
            asset_complete_sha256="a" * 64,
            tokenizer_identity={"files": {}},
            evaluation_tier="confirmation",
            devices=devices,
            closure_rows=8,
        )


def test_selected_document_tokens_use_captured_default_encode_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    texts = [f"document {index}" for index in range(8)]
    encoded = {
        text: [1, 10 + index, 20 + index, 2]
        for index, text in enumerate(texts)
    }
    rows = []
    for index, text in enumerate(texts):
        token_ids = encoded[text]
        rows.append(
            {
                "evaluation_tier": "screening",
                "reference_file": f"screening-{index}.safetensors",
                "corpus_line": index,
                "document_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "source_document_tokens": len(token_ids),
                "full_token_ids_sha256_u32le": token_ids_sha256(token_ids),
                "context_tokens": len(token_ids),
            }
        )
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(json.dumps({"text": text}) + "\n" for text in texts)
    )
    plan = {
        "sources": {
            "calibration_corpus": {
                "rows": len(texts),
                "bytes": corpus.stat().st_size,
                "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
            }
        },
        "documents": rows,
    }
    monkeypatch.setattr(
        terminal_logits,
        "validate_terminal_teacher_reference_plan",
        lambda value: {"document_count": len(value["documents"])},
    )

    class Tokenizer:
        def encode(self, text: str) -> list[int]:
            values = encoded[text]
            return values + [999] if text == texts[0] else values

    receipts = selected_document_token_receipts(
        plan=plan,
        corpus_path=corpus,
        tokenizer=Tokenizer(),
        evaluation_tier="screening",
    )

    assert len(receipts) == 8
    assert receipts["screening-0.safetensors"]["prompt_token_ids"] == encoded[
        texts[0]
    ]


def test_tokenizer_identity_hashes_only_local_material(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.json").write_text('{"version":"1"}\n')
    (tmp_path / "unrelated.bin").write_bytes(b"not tokenizer material")

    identity = tokenizer_file_identity(tmp_path)

    assert identity["root"] == str(tmp_path.resolve())
    assert set(identity["files"]) == {"tokenizer.json"}
    assert identity["files"]["tokenizer.json"]["sha256"] == hashlib.sha256(
        (tmp_path / "tokenizer.json").read_bytes()
    ).hexdigest()


def test_generated_screening_references_are_hash_and_token_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_bytes = PLAN_PATH.read_bytes()
    plan = json.loads(plan_bytes)
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    rows = [
        row for row in plan["documents"] if row["evaluation_tier"] == "screening"
    ]
    tensors_by_path: dict[Path, dict[str, torch.Tensor]] = {}
    receipts = []
    total_rows = 0
    for index, row in enumerate(rows):
        path = tmp_path / row["reference_file"]
        path.write_bytes(bytes([index]))
        prompt = list(range(row["context_tokens"]))
        target = prompt[1:]
        tensors_by_path[path] = {
            "prompt_token_ids": torch.tensor(prompt, dtype=torch.int32),
            "target_token_ids": torch.tensor(target, dtype=torch.int32),
        }
        receipts.append(
            {
                "schema": terminal_logits.DOCUMENT_RECEIPT_SCHEMA,
                "schema_version": 1,
                "document": {
                    "reference_plan_row": row,
                    "token_receipt": {
                        "prompt_token_ids": prompt,
                        "prompt_token_ids_sha256_u32le": token_ids_sha256(prompt),
                        "target_token_ids": target,
                        "context_tokens": row["context_tokens"],
                        "logit_rows": row["logit_rows"],
                    },
                },
                "file": row["reference_file"],
                "bytes": 1,
                "sha256": hashlib.sha256(bytes([index])).hexdigest(),
                "logits_key": "logits",
                "logits_shape": row["expected_logits_shape"],
                "logits_dtype": "bfloat16",
            }
        )
        total_rows += row["logit_rows"]

    class FakeSlice:
        def __init__(self, shape: list[int], dtype: str) -> None:
            self.shape = shape
            self.dtype = dtype

        def get_shape(self) -> list[int]:
            return self.shape

        def get_dtype(self) -> str:
            return self.dtype

    class FakeHandle:
        def __init__(self, path: Path) -> None:
            self.path = path

        def __enter__(self) -> FakeHandle:
            return self

        def __exit__(self, *unused: object) -> None:
            return None

        def keys(self) -> list[str]:
            return ["logits", "prompt_token_ids", "target_token_ids"]

        def metadata(self) -> dict[str, str]:
            row = next(item for item in rows if item["reference_file"] == self.path.name)
            return {
                "schema": terminal_logits.DOCUMENT_RECEIPT_SCHEMA,
                "teacher_model": plan["teacher"]["model_id"],
                "teacher_revision": plan["teacher"]["source_revision"],
                "document_sha256": row["document_sha256"],
                "evaluation_tier": "screening",
                "logit_equation": "lm_head(final_rms_norm(layer_77_output))",
            }

        def get_slice(self, key: str) -> FakeSlice:
            row = next(item for item in rows if item["reference_file"] == self.path.name)
            if key == "logits":
                return FakeSlice(row["expected_logits_shape"], "BF16")
            if key == "prompt_token_ids":
                return FakeSlice([row["context_tokens"]], "I32")
            return FakeSlice([row["logit_rows"]], "I32")

        def get_tensor(self, key: str) -> torch.Tensor:
            return tensors_by_path[self.path][key]

    monkeypatch.setattr(
        terminal_logits,
        "safe_open",
        lambda path, **unused: FakeHandle(Path(path)),
    )
    manifest = {
        "schema": terminal_logits.REFERENCE_MANIFEST_SCHEMA,
        "schema_version": 1,
        "status": "available_for_candidate_screening",
        "generation_contract": {
            "schema": terminal_logits.GENERATION_CONTRACT_SCHEMA,
            "schema_version": 1,
            "teacher_reference_plan_sha256": plan_sha256,
            "evaluation_tier": "screening",
            "document_count": 8,
            "endpoint": {
                "equation": "lm_head(final_rms_norm(layer_77_output))",
                "hidden_size": 6144,
                "vocabulary_size": 154880,
                "normalization_variance_dtype": "float32",
                "normalization_output_dtype": "bfloat16",
                "head_weight_dtype": "bfloat16",
                "logit_dtype": "bfloat16",
            },
        },
        "numerical_closure": {"bf16_repeat_bit_exact": True},
        "documents": receipts,
        "document_count": 8,
        "total_logit_rows": total_rows,
        "total_payload_bytes": 8,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    validated = load_validated_terminal_teacher_reference_documents(
        plan=plan,
        plan_sha256=plan_sha256,
        reference_directory=tmp_path,
        evaluation_tier="screening",
    )

    assert validated["document_count"] == 8
    assert validated["total_logit_rows"] == 9334
    assert validated["documents"][0]["prompt_token_ids"][0] == 0
    (tmp_path / rows[0]["reference_file"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="file differs"):
        load_validated_terminal_teacher_reference_documents(
            plan=plan,
            plan_sha256=plan_sha256,
            reference_directory=tmp_path,
            evaluation_tier="screening",
        )


def test_confirmation_reference_validation_denies_access_before_freeze(
    tmp_path: Path,
) -> None:
    plan_bytes = PLAN_PATH.read_bytes()
    with pytest.raises(PermissionError, match="frozen-candidate"):
        load_validated_terminal_teacher_reference_documents(
            plan=json.loads(plan_bytes),
            plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
            reference_directory=tmp_path,
            evaluation_tier="confirmation",
        )

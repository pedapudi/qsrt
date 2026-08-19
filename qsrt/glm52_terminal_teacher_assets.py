"""Download the minimal GLM-5.2 assets needed to reconstruct teacher logits.

The canonical Hessian archive stores the BF16 output of decoder layer 77 for
every captured token.  Teacher logits need only selected rows from that tensor,
the official final RMS-normalization vector, and the official language-model
head.  This module downloads exact HTTP byte ranges and records a hash receipt
for every extracted file; it never downloads the complete source checkpoint or
the complete terminal-hidden-state tensor.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request

from qsrt.glm52_pilot import atomic_write_json
from qsrt.glm52_terminal_teacher_reference import (
    HIDDEN_SIZE,
    sha256_file,
    validate_terminal_teacher_reference_plan,
)


DOWNLOAD_CONTRACT_SCHEMA = "qsrt_glm52_terminal_teacher_asset_download"
DOWNLOAD_RECEIPT_SCHEMA = "qsrt_glm52_terminal_teacher_asset_receipt"
COMPLETE_RECEIPT_SCHEMA = "qsrt_glm52_terminal_teacher_assets_complete"
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


def _repository_url(
    *, repository_kind: str, repository: str, revision: str, path: str
) -> str:
    if repository_kind not in {"dataset", "model"}:
        raise ValueError("repository kind must be dataset or model")
    prefix = "datasets/" if repository_kind == "dataset" else ""
    encoded_path = urllib.parse.quote(path, safe="/")
    return (
        f"https://huggingface.co/{prefix}{repository}/resolve/"
        f"{revision}/{encoded_path}"
    )


def _asset(
    *,
    name: str,
    destination: str,
    semantics: str,
    repository_kind: str,
    repository: str,
    revision: str,
    path: str,
    source_byte_start: int,
    extracted_bytes: int,
    expected_source_etag: str | None = None,
    expected_extracted_sha256: str | None = None,
) -> dict[str, Any]:
    if source_byte_start < 0 or extracted_bytes <= 0:
        raise ValueError("asset byte range must be positive")
    source_byte_stop_exclusive = source_byte_start + extracted_bytes
    return {
        "name": name,
        "destination": destination,
        "semantics": semantics,
        "source": {
            "repository_kind": repository_kind,
            "repository": repository,
            "revision": revision,
            "path": path,
            "url": _repository_url(
                repository_kind=repository_kind,
                repository=repository,
                revision=revision,
                path=path,
            ),
            "byte_start": source_byte_start,
            "byte_stop_exclusive": source_byte_stop_exclusive,
            "expected_etag": expected_source_etag,
        },
        "extracted_bytes": extracted_bytes,
        "expected_extracted_sha256": expected_extracted_sha256,
    }


def build_terminal_teacher_asset_download_contract(
    *, plan: Mapping[str, Any], plan_sha256: str
) -> dict[str, Any]:
    """Build the complete immutable range-download contract for one plan."""

    validated = validate_terminal_teacher_reference_plan(plan)
    if len(plan_sha256) != 64:
        raise ValueError("teacher-reference plan SHA-256 must have 64 digits")
    sources = plan["sources"]
    reference_assets = plan["reference_assets"]
    canonical = sources["canonical_dataset"]
    corpus = sources["calibration_corpus"]
    terminal = reference_assets["terminal_hidden"]
    head = reference_assets["language_model_head"]
    normalization = reference_assets["final_normalization"]

    assets = [
        _asset(
            name="canonical_document_plan",
            destination="metadata/canonical_document_plan.json",
            semantics=(
                "document identities, token hashes, roles, and terminal row offsets"
            ),
            repository_kind="dataset",
            repository=canonical["id"],
            revision=canonical["revision"],
            path=canonical["document_plan_path"],
            source_byte_start=0,
            extracted_bytes=canonical["document_plan_bytes"],
            expected_extracted_sha256=canonical["document_plan_sha256"],
        ),
        _asset(
            name="calibration_corpus",
            destination="metadata/reap_recall_calib.jsonl",
            semantics="source text used to reproduce and verify tokenization",
            repository_kind="model",
            repository=corpus["repository"],
            revision=corpus["revision"],
            path=corpus["path"],
            source_byte_start=0,
            extracted_bytes=corpus["bytes"],
            expected_extracted_sha256=corpus["sha256"],
        ),
        _asset(
            name="final_normalization",
            destination=f"weights/{normalization['extracted_file']}",
            semantics="official BF16 final RMS-normalization weight",
            repository_kind="model",
            repository=normalization["repository"],
            revision=normalization["revision"],
            path=normalization["source_shard"],
            source_byte_start=normalization["source_byte_start"],
            extracted_bytes=normalization["extracted_bytes"],
            expected_source_etag=normalization["source_shard_sha256"],
        ),
        _asset(
            name="language_model_head",
            destination=f"weights/{head['extracted_file']}",
            semantics="official BF16 untied language-model head",
            repository_kind="model",
            repository=head["repository"],
            revision=head["revision"],
            path=head["source_shard"],
            source_byte_start=head["source_byte_start"],
            extracted_bytes=head["extracted_bytes"],
            expected_source_etag=head["source_shard_sha256"],
        ),
    ]

    bytes_per_hidden_row = HIDDEN_SIZE * 2
    for document in plan["documents"]:
        start_row = document["terminal_hidden_row_start"]
        rows = document["logit_rows"]
        destination = document["reference_file"].removesuffix(".safetensors")
        assets.append(
            _asset(
                name=f"terminal_hidden_{document['document_sha256']}",
                destination=f"terminal_hidden/{destination}.bf16.bin",
                semantics=(
                    "BF16 decoder layer-77 outputs for next-token teacher logits"
                ),
                repository_kind="dataset",
                repository=terminal["repository"],
                revision=terminal["revision"],
                path=terminal["path"],
                source_byte_start=start_row * bytes_per_hidden_row,
                extracted_bytes=rows * bytes_per_hidden_row,
                expected_source_etag=terminal["sha256"],
            )
        )

    destinations = [asset["destination"] for asset in assets]
    if len(destinations) != len(set(destinations)):
        raise ValueError("teacher-reference asset destinations are not unique")
    return {
        "schema": DOWNLOAD_CONTRACT_SCHEMA,
        "schema_version": 1,
        "teacher_reference_plan_sha256": plan_sha256,
        "document_count": validated["document_count"],
        "assets": assets,
        "asset_count": len(assets),
        "total_download_bytes": sum(asset["extracted_bytes"] for asset in assets),
        "excludes": [
            "all decoder-layer weights",
            "unselected terminal-hidden rows",
            "the MTP layer",
            "the remainder of both source safetensors shards",
        ],
    }


def validate_terminal_teacher_asset_download_contract(
    *, contract: Mapping[str, Any], plan: Mapping[str, Any], plan_sha256: str
) -> dict[str, int]:
    """Rebuild an asset contract and reject any byte-range drift."""

    expected = build_terminal_teacher_asset_download_contract(
        plan=plan, plan_sha256=plan_sha256
    )
    if dict(contract) != expected:
        raise ValueError("teacher-reference asset download contract differs")
    return {
        "asset_count": expected["asset_count"],
        "total_download_bytes": expected["total_download_bytes"],
    }


def _authorization_headers(token: str | None) -> dict[str, str]:
    headers = {"Accept-Encoding": "identity", "User-Agent": "qsrt/teacher-assets"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _normalized_etag(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().strip('"')
    return normalized.removeprefix("W/").strip('"')


def validate_remote_asset_identity(
    asset: Mapping[str, Any], *, token: str | None, timeout_seconds: float
) -> dict[str, str | None]:
    """Validate an immutable Hugging Face object before requesting a range."""

    source = asset["source"]
    request = urllib.request.Request(
        source["url"],
        method="HEAD",
        headers=_authorization_headers(token),
    )
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        response = opener.open(request, timeout=timeout_seconds)
    except urllib.error.HTTPError as error:
        if error.code not in {301, 302, 303, 307, 308}:
            raise
        headers = error.headers
    else:
        with response:
            headers = response.headers
    expected_etag = source.get("expected_etag")
    observed_etag = _normalized_etag(
        headers.get("x-linked-etag") or headers.get("etag")
    )
    if expected_etag is not None and observed_etag != expected_etag:
        raise ValueError(
            f"remote source identity differs for {asset['name']}: "
            f"expected {expected_etag}, observed {observed_etag}"
        )
    return {
        "expected_etag": expected_etag,
        "observed_etag": observed_etag,
        "xet_hash": _normalized_etag(headers.get("x-xet-hash")),
    }


def _safe_destination(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("asset destination escapes its output directory")
    return candidate


def _load_valid_receipt(
    *, target: Path, receipt_path: Path, asset: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not target.exists() and not receipt_path.exists():
        return None
    if not target.is_file() or not receipt_path.is_file():
        raise ValueError(f"incomplete asset/receipt pair for {target}")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema") != DOWNLOAD_RECEIPT_SCHEMA
        or receipt.get("asset") != dict(asset)
        or receipt.get("bytes") != target.stat().st_size
        or receipt.get("sha256") != sha256_file(target)
    ):
        raise ValueError(f"asset receipt differs for {target}")
    return receipt


def _stream_one_range(
    *,
    url: str,
    request_start: int,
    request_stop_inclusive: int,
    output: Any,
    token: str | None,
    timeout_seconds: float,
) -> dict[str, str | int | None]:
    headers = _authorization_headers(token)
    headers["Range"] = f"bytes={request_start}-{request_stop_inclusive}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        if response.status != 206:
            raise ValueError(f"range request returned HTTP {response.status}")
        match = _CONTENT_RANGE.fullmatch(response.headers.get("content-range", ""))
        if match is None:
            raise ValueError("range response lacks a valid Content-Range")
        observed_start, observed_stop, source_bytes = map(int, match.groups())
        if (observed_start, observed_stop) != (
            request_start,
            request_stop_inclusive,
        ):
            raise ValueError("range response boundaries differ")
        expected_bytes = request_stop_inclusive - request_start + 1
        content_length = int(response.headers.get("content-length", "-1"))
        if content_length != expected_bytes:
            raise ValueError("range response length differs")
        copied = 0
        while copied < expected_bytes:
            block = response.read(min(8 * 1024 * 1024, expected_bytes - copied))
            if not block:
                raise OSError("range response ended before all bytes arrived")
            output.write(block)
            copied += len(block)
        if response.read(1):
            raise ValueError("range response contains trailing bytes")
        return {
            "response_etag": _normalized_etag(response.headers.get("etag")),
            "content_range": response.headers.get("content-range"),
            "source_bytes": source_bytes,
        }


def download_one_terminal_teacher_asset(
    *,
    asset: Mapping[str, Any],
    destination_root: Path,
    token: str | None,
    timeout_seconds: float,
    retries: int,
) -> dict[str, Any]:
    """Download or verify one asset, preserving a resumable partial file."""

    target = _safe_destination(destination_root, str(asset["destination"]))
    receipt_path = target.with_name(target.name + ".receipt.json")
    existing = _load_valid_receipt(
        target=target, receipt_path=receipt_path, asset=asset
    )
    if existing is not None:
        return existing
    if target.exists() or receipt_path.exists():
        raise ValueError(f"unreceipted completed asset exists at {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    expected_bytes = int(asset["extracted_bytes"])
    partial_bytes = partial.stat().st_size if partial.exists() else 0
    if partial_bytes > expected_bytes:
        raise ValueError(f"partial asset is larger than expected at {partial}")

    source = asset["source"]
    response_record: dict[str, Any] | None = None
    for attempt in range(retries + 1):
        partial_bytes = partial.stat().st_size if partial.exists() else 0
        if partial_bytes == expected_bytes:
            break
        request_start = int(source["byte_start"]) + partial_bytes
        request_stop = int(source["byte_stop_exclusive"]) - 1
        try:
            with partial.open("ab") as output:
                response_record = _stream_one_range(
                    url=str(source["url"]),
                    request_start=request_start,
                    request_stop_inclusive=request_stop,
                    output=output,
                    token=token,
                    timeout_seconds=timeout_seconds,
                )
                output.flush()
                os.fsync(output.fileno())
        except (OSError, TimeoutError, urllib.error.URLError):
            if attempt >= retries:
                raise
            time.sleep(min(2**attempt, 30))

    if not partial.is_file() or partial.stat().st_size != expected_bytes:
        raise OSError(f"asset download remains incomplete at {partial}")
    digest = sha256_file(partial)
    expected_digest = asset.get("expected_extracted_sha256")
    if expected_digest is not None and digest != expected_digest:
        raise ValueError(f"downloaded asset SHA-256 differs at {partial}")
    partial.replace(target)
    receipt = {
        "schema": DOWNLOAD_RECEIPT_SCHEMA,
        "schema_version": 1,
        "asset": dict(asset),
        "bytes": expected_bytes,
        "sha256": digest,
        "completed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "last_range_response": response_record,
    }
    atomic_write_json(receipt_path, receipt)
    return receipt


def prepare_terminal_teacher_asset_destination(
    *, destination: Path, contract: Mapping[str, Any]
) -> Path:
    """Bind an empty or resumed destination to exactly one download contract."""

    destination = destination.resolve()
    contract_path = destination / "download_contract.json"
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != dict(contract):
            raise ValueError("destination belongs to another asset contract")
    else:
        occupants = [path for path in destination.iterdir()]
        if occupants:
            raise ValueError("unbound asset destination is not empty")
        atomic_write_json(contract_path, contract)
    return destination


def download_terminal_teacher_assets(
    *,
    contract: Mapping[str, Any],
    destination: Path,
    token: str | None,
    jobs: int,
    timeout_seconds: float,
    retries: int,
) -> dict[str, Any]:
    """Download every contracted range and write a complete hash manifest."""

    if not 1 <= jobs <= 32:
        raise ValueError("download jobs must be between one and thirty-two")
    destination = prepare_terminal_teacher_asset_destination(
        destination=destination, contract=contract
    )
    if (destination / "complete.json").is_file():
        return validate_downloaded_terminal_teacher_assets(
            contract=contract, destination=destination
        )
    assets: Sequence[Mapping[str, Any]] = contract["assets"]
    identity_by_url: dict[str, dict[str, str | None]] = {}
    for asset in assets:
        url = str(asset["source"]["url"])
        if url not in identity_by_url:
            identity_by_url[url] = validate_remote_asset_identity(
                asset, token=token, timeout_seconds=timeout_seconds
            )

    def download(asset: Mapping[str, Any]) -> dict[str, Any]:
        return download_one_terminal_teacher_asset(
            asset=asset,
            destination_root=destination,
            token=token,
            timeout_seconds=timeout_seconds,
            retries=retries,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        receipts = list(executor.map(download, assets))
    complete = {
        "schema": COMPLETE_RECEIPT_SCHEMA,
        "schema_version": 1,
        "teacher_reference_plan_sha256": contract[
            "teacher_reference_plan_sha256"
        ],
        "download_contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "asset_count": len(receipts),
        "total_bytes": sum(receipt["bytes"] for receipt in receipts),
        "remote_identities": identity_by_url,
        "files": [
            {
                "path": asset["destination"],
                "bytes": receipt["bytes"],
                "sha256": receipt["sha256"],
            }
            for asset, receipt in zip(assets, receipts, strict=True)
        ],
        "completed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    atomic_write_json(destination / "complete.json", complete)
    return complete


def validate_downloaded_terminal_teacher_assets(
    *, contract: Mapping[str, Any], destination: Path
) -> dict[str, Any]:
    """Rehash every extracted file and close the complete download receipt."""

    destination = destination.resolve()
    stored_contract = json.loads((destination / "download_contract.json").read_text())
    if stored_contract != dict(contract):
        raise ValueError("downloaded teacher-asset contract differs")
    complete = json.loads((destination / "complete.json").read_text())
    if (
        complete.get("schema") != COMPLETE_RECEIPT_SCHEMA
        or complete.get("teacher_reference_plan_sha256")
        != contract["teacher_reference_plan_sha256"]
        or complete.get("asset_count") != contract["asset_count"]
        or complete.get("total_bytes") != contract["total_download_bytes"]
    ):
        raise ValueError("teacher-asset complete receipt differs")
    expected_files = []
    for asset in contract["assets"]:
        target = _safe_destination(destination, str(asset["destination"]))
        receipt = _load_valid_receipt(
            target=target,
            receipt_path=target.with_name(target.name + ".receipt.json"),
            asset=asset,
        )
        if receipt is None:
            raise FileNotFoundError(target)
        expected_files.append(
            {
                "path": asset["destination"],
                "bytes": receipt["bytes"],
                "sha256": receipt["sha256"],
            }
        )
    if complete.get("files") != expected_files:
        raise ValueError("teacher-asset file manifest differs")
    return complete


__all__ = [
    "COMPLETE_RECEIPT_SCHEMA",
    "DOWNLOAD_CONTRACT_SCHEMA",
    "DOWNLOAD_RECEIPT_SCHEMA",
    "build_terminal_teacher_asset_download_contract",
    "download_one_terminal_teacher_asset",
    "download_terminal_teacher_assets",
    "prepare_terminal_teacher_asset_destination",
    "validate_downloaded_terminal_teacher_assets",
    "validate_remote_asset_identity",
    "validate_terminal_teacher_asset_download_contract",
]

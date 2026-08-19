from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import threading
from typing import Iterator

import pytest

from qsrt.glm52_terminal_teacher_assets import (
    DOWNLOAD_RECEIPT_SCHEMA,
    build_terminal_teacher_asset_download_contract,
    download_one_terminal_teacher_asset,
    download_terminal_teacher_assets,
    prepare_terminal_teacher_asset_destination,
    validate_downloaded_terminal_teacher_assets,
    validate_remote_asset_identity,
    validate_terminal_teacher_asset_download_contract,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    ROOT / "experiments/glm52_terminal_hidden_teacher_reference_plan.json"
)


def _plan_and_hash() -> tuple[dict[str, object], str]:
    raw = PLAN_PATH.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def test_asset_contract_fetches_only_frozen_terminal_rows_and_endpoint() -> None:
    plan, plan_sha256 = _plan_and_hash()
    contract = build_terminal_teacher_asset_download_contract(
        plan=plan, plan_sha256=plan_sha256
    )

    assert validate_terminal_teacher_asset_download_contract(
        contract=contract, plan=plan, plan_sha256=plan_sha256
    ) == {
        "asset_count": 44,
        "total_download_bytes": 2_857_330_353,
    }
    assert contract["total_download_bytes"] < 2_900_000_000
    hidden = [
        asset
        for asset in contract["assets"]
        if asset["name"].startswith("terminal_hidden_")
    ]
    assert len(hidden) == 40
    assert sum(asset["extracted_bytes"] for asset in hidden) == 919_339_008
    assert all(asset["source"]["byte_start"] % (6_144 * 2) == 0 for asset in hidden)
    assert all(
        asset["source"]["byte_stop_exclusive"] % (6_144 * 2) == 0
        for asset in hidden
    )
    names = {asset["name"] for asset in contract["assets"]}
    assert {"language_model_head", "final_normalization"} <= names


def test_asset_destination_requires_an_immutable_contract(tmp_path: Path) -> None:
    plan, plan_sha256 = _plan_and_hash()
    contract = build_terminal_teacher_asset_download_contract(
        plan=plan, plan_sha256=plan_sha256
    )
    destination = tmp_path / "assets"

    assert prepare_terminal_teacher_asset_destination(
        destination=destination, contract=contract
    ) == destination
    assert prepare_terminal_teacher_asset_destination(
        destination=destination, contract=contract
    ) == destination
    changed = dict(contract)
    changed["teacher_reference_plan_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="another asset contract"):
        prepare_terminal_teacher_asset_destination(
            destination=destination, contract=changed
        )


@contextmanager
def _range_server(payload: bytes, linked_etag: str) -> Iterator[tuple[str, list[str]]]:
    observed_ranges: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/object")
            self.send_header("x-linked-etag", f'"{linked_etag}"')
            self.end_headers()

        def do_GET(self) -> None:
            value = self.headers.get("Range")
            if value is None or not value.startswith("bytes="):
                self.send_error(400)
                return
            observed_ranges.append(value)
            start_text, stop_text = value.removeprefix("bytes=").split("-", 1)
            start, stop = int(start_text), int(stop_text)
            body = payload[start : stop + 1]
            self.send_response(206)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Range", f"bytes {start}-{stop}/{len(payload)}")
            self.send_header("ETag", '"local-object"')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/source", observed_ranges
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_range_download_resumes_and_receipts_exact_extracted_bytes(
    tmp_path: Path,
) -> None:
    payload = bytes(range(256))
    linked_etag = hashlib.sha256(payload).hexdigest()
    with _range_server(payload, linked_etag) as (url, observed_ranges):
        asset = {
            "name": "test_range",
            "destination": "weights/range.bin",
            "semantics": "test bytes",
            "source": {
                "repository_kind": "model",
                "repository": "example/model",
                "revision": "a" * 40,
                "path": "model.safetensors",
                "url": url,
                "byte_start": 10,
                "byte_stop_exclusive": 100,
                "expected_etag": linked_etag,
            },
            "extracted_bytes": 90,
            "expected_extracted_sha256": hashlib.sha256(payload[10:100]).hexdigest(),
        }
        identity = validate_remote_asset_identity(
            asset, token=None, timeout_seconds=5
        )
        assert identity["observed_etag"] == linked_etag
        partial = tmp_path / "weights/range.bin.partial"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(payload[10:37])

        receipt = download_one_terminal_teacher_asset(
            asset=asset,
            destination_root=tmp_path,
            token=None,
            timeout_seconds=5,
            retries=0,
        )

    assert observed_ranges == ["bytes=37-99"]
    assert (tmp_path / "weights/range.bin").read_bytes() == payload[10:100]
    assert receipt["schema"] == DOWNLOAD_RECEIPT_SCHEMA
    assert receipt["bytes"] == 90
    assert receipt["sha256"] == hashlib.sha256(payload[10:100]).hexdigest()
    assert json.loads(
        (tmp_path / "weights/range.bin.receipt.json").read_text()
    ) == receipt


def test_complete_download_receipt_rehashes_every_file(tmp_path: Path) -> None:
    payload = bytes(range(128))
    linked_etag = hashlib.sha256(payload).hexdigest()
    with _range_server(payload, linked_etag) as (url, observed_ranges):
        asset = {
            "name": "complete_test",
            "destination": "source.bin",
            "semantics": "complete receipt test",
            "source": {
                "repository_kind": "model",
                "repository": "example/model",
                "revision": "a" * 40,
                "path": "model.safetensors",
                "url": url,
                "byte_start": 16,
                "byte_stop_exclusive": 80,
                "expected_etag": linked_etag,
            },
            "extracted_bytes": 64,
            "expected_extracted_sha256": hashlib.sha256(payload[16:80]).hexdigest(),
        }
        contract = {
            "schema": "qsrt_glm52_terminal_teacher_asset_download",
            "schema_version": 1,
            "teacher_reference_plan_sha256": "f" * 64,
            "assets": [asset],
            "asset_count": 1,
            "total_download_bytes": 64,
            "excludes": [],
        }
        complete = download_terminal_teacher_assets(
            contract=contract,
            destination=tmp_path / "assets",
            token=None,
            jobs=1,
            timeout_seconds=5,
            retries=0,
        )
        validated = validate_downloaded_terminal_teacher_assets(
            contract=contract, destination=tmp_path / "assets"
        )

    assert observed_ranges == ["bytes=16-79"]
    assert validated == complete
    assert validated["total_bytes"] == 64

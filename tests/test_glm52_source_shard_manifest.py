from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/glm52_layers_52_60_63_64_source_shards.json"
DOWNLOAD_SCRIPT = (
    ROOT
    / "experiments/download_glm52_layers_52_60_63_64_source_shards_on_kossel.sh"
)


def test_bounded_glm52_source_manifest_has_frozen_scope_and_byte_total() -> None:
    manifest = json.loads(MANIFEST.read_text())
    shards = manifest["shards"]

    assert manifest["source_model"] == "zai-org/GLM-5.2"
    assert manifest["source_revision"] == (
        "b4734de4facf877f85769a911abafc5283eab3d9"
    )
    assert manifest["layers"] == [52, 60, 63, 64]
    assert len(shards) == 17
    assert len({shard["file"] for shard in shards}) == len(shards)
    assert sum(shard["size"] for shard in shards) == 91_142_336_944
    assert manifest["total_bytes"] == 91_142_336_944
    for shard in shards:
        assert shard["file"].startswith("model-")
        assert shard["file"].endswith("-of-00282.safetensors")
        assert shard["size"] > 0
        assert len(shard["sha256"]) == 64
        assert set(shard["sha256"]) <= set("0123456789abcdef")


def test_downloader_uses_the_committed_bounded_manifest() -> None:
    source = DOWNLOAD_SCRIPT.read_text()

    assert 'parallel_downloads=4' in source
    assert 'glm52_layers_52_60_63_64_source_shards.json' in source
    assert 'expected_shards=17' in source
    assert 'expected_bytes=91142336944' in source
    assert '--continue-at -' in source
    assert 'sha256sum --check --strict' in source

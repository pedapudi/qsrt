from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/build_glm52_all_panel_k4_down_registration.py"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_all_panel_k4_down_registration_is_error_blind_and_smaller(
    tmp_path: Path,
) -> None:
    panel_path = tmp_path / "panel.json"
    refit_root = tmp_path / "refit"
    destination = tmp_path / "registration.json"
    expert_rates = (
        (130, [4, 3, 4]),
        (74, [3, 3, 4]),
        (172, [3, 3, 3]),
        (83, [4, 4, 4]),
        (249, [4, 3, 4]),
        (84, [3, 3, 4]),
        (214, [3, 3, 3]),
        (232, [4, 4, 4]),
    )
    panel = {
        "schema": "qsrt_glm52_real_weight_panel",
        "layer": 55,
        "source": {
            "model_id": "zai-org/GLM-5.2",
            "revision": "source-revision",
            "config_sha256": "a" * 64,
            "index_sha256": "b" * 64,
        },
        "comparison_checkpoint": {
            "model_id": "comparison/model",
            "revision": "comparison-revision",
            "manifest_sha256": "c" * 64,
        },
        "experts": [
            {"expert": expert, "exl3_rates": rates}
            for expert, rates in expert_rates
        ],
    }
    _write_json(panel_path, panel)
    refit_manifest = {"kind": "test-refit", "panel": {"55": [x[0] for x in expert_rates]}}
    _write_json(refit_root / "manifest.json", refit_manifest)
    _write_json(
        refit_root / "report.json",
        {
            "status": "complete",
            "layer": 55,
            "manifest_sha256": _canonical_sha256(refit_manifest),
            "experts": [
                {"expert": expert, "local_error": 10_000 + expert}
                for expert, _ in expert_rates
            ],
        },
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)

    completed = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--panel-manifest",
            str(panel_path),
            "--down-refit",
            str(refit_root),
            "--dest",
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    registration = json.loads(destination.read_text())
    assert registration["status"] == "frozen_before_candidate_k4_measurement"
    assert registration["candidate_construction"] == {
        "allow_source_target_fallback": True,
        "down_rate": 4,
        "down_target_policy": (
            "use the reconstructed-activation refit when the K3 refit artifact "
            "accepted it; otherwise use that artifact's source-target fallback"
        ),
        "gate_rate": 3,
        "k4_candidate_measurements_used": False,
        "up_rate": 3,
    }
    assert len(registration["registered_replacements"]) == 8
    assert all(
        replacement["candidate_rates"]
        == {"gate_proj": 3, "up_proj": 3, "down_proj": 4}
        for replacement in registration["registered_replacements"]
    )
    byte_contract = registration["logical_byte_contract"]
    assert byte_contract["comparison_exl3_rate_sum"] == 84
    assert byte_contract["registered_candidate_rate_sum"] == 80
    assert byte_contract["logical_margin_bytes"] == 4 * 1_572_864


def test_all_panel_k4_down_registration_rejects_a_non_smaller_map(
    tmp_path: Path,
) -> None:
    panel_path = tmp_path / "panel.json"
    refit_root = tmp_path / "refit"
    destination = tmp_path / "registration.json"
    experts = list(range(8))
    panel = {
        "schema": "qsrt_glm52_real_weight_panel",
        "layer": 55,
        "source": {
            "model_id": "source",
            "revision": "revision",
            "config_sha256": "a" * 64,
            "index_sha256": "b" * 64,
        },
        "comparison_checkpoint": {
            "model_id": "comparison",
            "revision": "revision",
            "manifest_sha256": "c" * 64,
        },
        "experts": [
            {"expert": expert, "exl3_rates": [3, 3, 3]} for expert in experts
        ],
    }
    _write_json(panel_path, panel)
    refit_manifest = {"kind": "test-refit"}
    _write_json(refit_root / "manifest.json", refit_manifest)
    _write_json(
        refit_root / "report.json",
        {
            "status": "complete",
            "layer": 55,
            "manifest_sha256": _canonical_sha256(refit_manifest),
            "experts": [{"expert": expert} for expert in experts],
        },
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)

    completed = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--panel-manifest",
            str(panel_path),
            "--down-refit",
            str(refit_root),
            "--dest",
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert "not smaller than EXL3" in completed.stderr
    assert not destination.exists()

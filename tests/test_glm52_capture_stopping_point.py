import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPOSITORY_ROOT
    / "experiments/mark_glm52_layers55_58_capture_stopping_point_on_kossel.sh"
)


def test_capture_stopping_point_starts_no_candidate_job() -> None:
    source = SCRIPT.read_text()
    assert "docker create" not in source
    assert "docker start" not in source
    assert '"automatic_candidate_queue_active": False' in source
    assert '"status": "safe_for_host_shutdown"' in source
    assert '"gpu_compute_processes": []' in source


def test_capture_stopping_point_validates_all_four_layers() -> None:
    source = SCRIPT.read_text()
    assert "for layer in (55, 56, 57, 58):" in source
    assert '{"activation_fit": 32, "candidate_selection": 8}' in source
    assert 'receipt.get("shard_count") != 16' in source
    assert 'receipt.get("total_bytes") != 85_783_011_360' in source


def test_capture_stopping_point_has_valid_shell_syntax() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

#!/usr/bin/env python3
"""Embed the browser-executable GLM-5.2 benchmark in the QSRT explainer."""

from __future__ import annotations

import json
import re
from pathlib import Path

from qsrt.glm52_tiny_benchmark import run_benchmark, run_sweep


REPOSITORY = Path(__file__).parents[1]
INFOGRAPHIC = REPOSITORY / "docs" / "qsrt-three-improvements-infographic.html"
BENCHMARK = REPOSITORY / "qsrt" / "glm52_tiny_benchmark.py"
BENCHMARK_DEPENDENCY = REPOSITORY / "qsrt" / "tiny_improvement_benchmark.py"
SOURCE_PATTERN = re.compile(
    r'(<script type="text/plain" id="python-benchmark-source">\n)'
    r".*?"
    r"(\n</script>)",
    flags=re.DOTALL,
)
DEPENDENCY_PATTERN = re.compile(
    r'(<script type="text/plain" id="python-benchmark-dependency-source">\n)'
    r".*?"
    r"(\n</script>)",
    flags=re.DOTALL,
)
DATA_PATTERN = re.compile(
    r'(<script type="application/json" id="benchmark-data">\n)'
    r".*?"
    r"(\n</script>)",
    flags=re.DOTALL,
)


def _compact_stress_report(report: dict) -> dict:
    """Keep aggregate evidence while omitting 256 per-expert records."""

    return {
        "expert_count": report["expert_count"],
        "configuration": report["configuration"],
        "peak_bits_at_play": report["bit_budget"]["bits_at_play"],
        "heldout_transitions": report["heldout_transitions"],
        "heldout_forward_kld_transitions": report[
            "heldout_forward_kld_transitions"
        ],
        "kld_evidence_boundary": report["kld_evidence_boundary"],
        "size_evidence_boundary": report["size_evidence_boundary"],
    }


def main() -> None:
    source = BENCHMARK.read_text().rstrip()
    dependency_source = BENCHMARK_DEPENDENCY.read_text().rstrip()
    if "</script>" in source.lower() or "</script>" in dependency_source.lower():
        raise ValueError("embedded Python source contains an HTML script closing tag")
    reports = {
        "representative": run_benchmark(source_seed=0),
        "sweep": run_sweep(8, start_seed=0),
        "stress": _compact_stress_report(run_sweep(256, start_seed=0)),
    }
    compact_data = json.dumps(reports, separators=(",", ":"), sort_keys=True)
    infographic = INFOGRAPHIC.read_text()
    updated, data_replacements = DATA_PATTERN.subn(
        lambda match: match.group(1) + compact_data + match.group(2),
        infographic,
        count=1,
    )
    if data_replacements != 1:
        raise ValueError("explainer must contain one benchmark data block")
    updated, dependency_replacements = DEPENDENCY_PATTERN.subn(
        lambda match: match.group(1) + dependency_source + match.group(2),
        updated,
        count=1,
    )
    if dependency_replacements != 1:
        raise ValueError("explainer must contain one Python dependency block")
    updated, source_replacements = SOURCE_PATTERN.subn(
        lambda match: match.group(1) + source + match.group(2),
        updated,
        count=1,
    )
    if source_replacements != 1:
        raise ValueError("explainer must contain one Python benchmark source block")
    INFOGRAPHIC.write_text(updated)


if __name__ == "__main__":
    main()

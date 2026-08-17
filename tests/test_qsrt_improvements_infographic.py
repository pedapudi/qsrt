from __future__ import annotations

import json
import math
import re
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

import pytest

from qsrt.glm52_tiny_benchmark import STAGE_NAMES, run_benchmark, run_sweep


REPOSITORY = Path(__file__).parents[1]
INFOGRAPHIC = REPOSITORY / "docs" / "qsrt-three-improvements-infographic.html"
BENCHMARK_SOURCE = REPOSITORY / "qsrt" / "glm52_tiny_benchmark.py"
BENCHMARK_DEPENDENCY = REPOSITORY / "qsrt" / "tiny_improvement_benchmark.py"


class _StructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.inline_handlers: list[str] = []
        self.scene_links = 0
        self.scene_panels = 0
        self.overview_links = 0
        self.comparison_links = 0
        self.navigation_hrefs: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if identifier := attributes.get("id"):
            self.ids.append(identifier)
        self.inline_handlers.extend(name for name in attributes if name.startswith("on"))
        if tag == "a" and "data-scene" in attributes:
            self.scene_links += 1
            self.navigation_hrefs.append(attributes.get("href", ""))
        if tag == "section" and "data-scene-panel" in attributes:
            self.scene_panels += 1
        if tag == "a" and attributes.get("data-nav-target") == "overview":
            self.overview_links += 1
            self.navigation_hrefs.append(attributes.get("href", ""))
        if tag == "a" and attributes.get("data-nav-target") == "comparison":
            self.comparison_links += 1
            self.navigation_hrefs.append(attributes.get("href", ""))


def _script_text(source: str, identifier: str) -> str:
    match = re.search(
        rf'<script type="[^"]+" id="{identifier}">\n(.*?)\n</script>',
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def _embedded_json(source: str) -> dict:
    return json.loads(_script_text(source, "benchmark-data"))


def _visible_source(source: str) -> str:
    return source.split('<script type="application/json" id="benchmark-data">', 1)[0]


def _relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92
        if value <= 0.04045
        else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(left: str, right: str) -> float:
    high, low = sorted(
        (_relative_luminance(left), _relative_luminance(right)), reverse=True
    )
    return (high + 0.05) / (low + 0.05)


def test_explainer_starts_with_qsrt_foundations_before_two_active_experiments() -> None:
    source = INFOGRAPHIC.read_text()
    visible = _visible_source(source)
    parser = _StructureParser()
    parser.feed(source)

    assert parser.overview_links == 1
    assert parser.comparison_links == 1
    assert parser.scene_links == 2
    assert parser.scene_panels == 3
    assert parser.navigation_hrefs == [
        "#overview",
        "#comparison",
        "#loss",
        "#down",
    ]
    assert len(parser.ids) == len(set(parser.ids))
    assert not parser.inline_handlers
    assert visible.index("Start with one number") < visible.index(
        "Two changes remain"
    )
    assert 'data-rejected-mechanism hidden aria-hidden="true"' in source
    assert 'const scenes = ["loss", "down"]' in source
    assert visible.count('class="concept-number"') == 4
    assert visible.count('class="baby-steps"') == 3
    assert visible.count('class="animation"') == 3
    assert visible.count('class="mechanism-svg"') == 3
    assert visible.count('class="worked"') == 3
    assert visible.count('class="code-explainer"') == 3
    assert 'class="journey-svg"' in visible
    assert "A two-step K2 tail-biting trellis" in visible
    assert '<noscript>' in source
    assert '@media (prefers-reduced-motion:reduce)' in source
    assert 'id="motion-button"' in source
    assert 'id="theme-button"' in source


def test_foundation_explains_qsrt_without_assuming_quantization_context() -> None:
    visible = _visible_source(INFOGRAPHIC.read_text())

    required_in_order = (
        "A weight changes a signal.",
        "Two bits select four choices.",
        "A trellis links choices across a list of target numbers.",
        "One search chooses the whole path.",
        "one adjusted target through the trellis",
        "What the letters in QSRT mean",
    )
    positions = [visible.index(phrase) for phrase in required_in_order]
    assert positions == sorted(positions)
    assert "path decisions" not in visible
    assert "fixed directed graph called a trellis" in visible
    assert "current history state to the next history state" in visible
    assert "two stored bits choose one outgoing arrow" in visible
    assert "selected edge are called a branch" not in visible
    assert "decoder follows the stored route" in visible
    assert visible.index("fixed directed graph called a trellis") < visible.index(
        "A two-step K2 tail-biting trellis"
    )
    assert "QSRT is the Quantile-Stratified Rate-shifted Trellis codec" in visible
    assert "14 history bits" in visible
    assert "+ branch 10" in visible
    assert "T12 shelf returns +0.323418" in visible
    assert "stored model contains the branch bits" in visible
    assert "They are not fourteen stored bits per weight" in visible
    assert "coupled Hadamard transform" in visible
    assert "BlockLDLQ uses earlier numerical errors" in visible
    assert "four-bit representation called X4T" in visible
    assert "eight-bit representation called MXFP8" in visible


def test_opening_balances_short_orientation_text_with_a_storage_drawing() -> None:
    visible = _visible_source(INFOGRAPHIC.read_text())
    copy_match = re.search(r'<p class="hero-copy">(.*?)</p>', visible, flags=re.DOTALL)

    assert copy_match is not None
    copy = re.sub(r"<[^>]+>", "", copy_match.group(1))
    assert len(copy.split()) <= 65
    assert "16-by-16" not in copy
    assert "BlockLDLQ" not in copy
    assert 'class="hero-summary"' in visible
    assert "ORIGINAL MATRIX" in visible
    assert "STORED BRANCHES" in visible
    assert "REBUILT MATRIX" in visible
    assert "The original matrix is the object being compressed" in visible


def test_storage_summary_separates_payload_from_decoder_working_memory() -> None:
    visible = _visible_source(INFOGRAPHIC.read_text())

    assert "Before compression · eight learned numbers" in visible
    assert "In the checkpoint · eight two-bit branches" in visible
    assert "Working memory · not stored per slice" in visible
    assert "the file stores sixteen payload bits for eight weights" in visible
    assert "The fourteen-bit register only helps interpret the path" in visible
    assert "30 bits present at peak" not in visible


def test_navigation_uses_real_links_and_offsets_targets_below_the_topbar() -> None:
    source = INFOGRAPHIC.read_text()

    assert 'href="#overview" data-nav-target="overview"' in source
    assert 'href="#comparison" data-nav-target="comparison"' in source
    assert 'id="comparison" aria-labelledby="comparison-title"' in source
    for name in ("loss", "down"):
        assert f'href="#{name}" data-nav-target="{name}" data-scene="{name}"' in source
        assert f'id="{name}" data-scene-panel="{name}"' in source
    assert 'href="#table" data-nav-target="table"' not in source
    assert 'id="table" data-scene-panel="table"' in source
    assert 'data-rejected-mechanism hidden aria-hidden="true"' in source
    assert "scroll-padding-top:72px" in source
    assert source.count("scroll-margin-top:72px") >= 4
    assert '$(".topbar").getBoundingClientRect().height' in source
    assert "elementTop - topbarHeight - 18" in source
    assert "requestAnimationFrame" in source
    assert "window.scrollTo" in source
    assert "scrollIntoView" not in source
    assert "event.preventDefault()" in source
    assert 'scrollToElement($("#" + name))' in source
    assert 'scrollToElement($("#comparison"))' in source
    assert 'window.addEventListener("popstate"' in source
    assert 'aria-current="location"' in source


def test_two_target_k2_illustration_shows_every_closed_path_and_its_limits() -> None:
    visible = _visible_source(INFOGRAPHIC.read_text())

    assert visible.count('class="path-candidate" role="listitem"') == 15
    assert visible.count('class="path-candidate selected" role="listitem"') == 1
    assert visible.count('class="candidate-route"') == 16
    assert "all sixteen legal tail-biting paths" in visible
    assert "has 16,384 history states" in visible
    assert "uses 128 targets of cyclic context on either side" in visible
    assert "two-target drawing removes that context" in visible
    for first in ("00", "01", "10", "11"):
        for second in ("00", "01", "10", "11"):
            assert f"branches {first} · {second}" in visible
    assert "least-bad path in the shortened graph" in visible
    assert "TWO GATE WEIGHTS" in visible
    assert "ORIGINAL" in visible
    assert "REBUILT" in visible
    assert "second target is +0.730844" in visible
    assert "selected path rebuilds only +0.298982" in visible
    assert "This reconstruction is poor" in visible
    for state in ("0x0445", "0x0446", "0x0447", "0x1110", "0x1112", "0x1113"):
        assert state in visible
    assert "faint edges are the alternatives at the two visited states" in visible
    assert visible.count("index 1 · state 0x0444") == 4
    assert "yes; closes" in visible


def test_two_target_trellis_arrowheads_follow_paths_without_svg_transforms() -> None:
    source = _visible_source(INFOGRAPHIC.read_text())
    illustration = re.search(
        r'(two-target trellis illustration.*?<div class="choice-table-wrap">)',
        source,
        flags=re.DOTALL,
    )

    assert illustration is not None
    drawing = illustration.group(1)
    assert drawing.count('class="route-arrow"') == 16
    assert drawing.count(
        "M109.6 12.1L117.2 9.4L111 4.2M51.7 32.7L43.8 34.3L49.1 40.3"
    ) == 16
    assert ".candidate-route .route-arrow { fill:none; stroke:var(--faint)" in source
    assert ".path-candidate.selected .route-arrow { stroke:var(--accent)" in source
    assert drawing.count('class="trellis-arrowhead"') == 6
    assert drawing.count('class="trellis-arrowhead-accent"') == 3
    assert "marker-end=" not in drawing
    assert "orient=" not in drawing
    assert 'd="M329 80H414"' in drawing
    assert 'd="M406 75L414 80L406 85"' in drawing
    assert 'd="M538 80H620"' in drawing
    assert 'd="M612 75L620 80L612 85"' in drawing
    assert 'd="M681 106C681 306 267 306 267 117"' in drawing
    assert 'd="M262 125L267 117L272 125"' in drawing
    assert "rotate(" not in drawing

    tangent_and_heads = (
        ((132.24, 24.0), ((109.6, 12.1), (117.2, 9.4), (111.0, 4.2))),
        ((-123.0, -43.0), ((51.7, 32.7), (43.8, 34.3), (49.1, 40.3))),
        ((1.0, 0.0), ((406.0, 75.0), (414.0, 80.0), (406.0, 85.0))),
        ((35.0, 7.0), ((405.4, 142.4), (414.0, 140.0), (406.9, 134.5))),
        ((32.0, 9.0), ((405.2, 201.7), (414.0, 200.0), (407.4, 194.0))),
        ((30.0, 11.0), ((405.1, 261.0), (414.0, 260.0), (407.9, 253.5))),
        ((33.0, 7.0), ((611.3, 142.3), (620.0, 140.0), (613.0, 134.4))),
        ((31.0, 9.0), ((611.2, 201.6), (620.0, 200.0), (613.4, 193.9))),
        ((30.0, 11.0), ((611.1, 261.0), (620.0, 260.0), (613.9, 253.5))),
        ((0.0, -1.0), ((262.0, 125.0), (267.0, 117.0), (272.0, 125.0))),
    )
    for tangent, (first_wing, tip, second_wing) in tangent_and_heads:
        base = (
            (first_wing[0] + second_wing[0]) / 2,
            (first_wing[1] + second_wing[1]) / 2,
        )
        direction = (tip[0] - base[0], tip[1] - base[1])
        cosine = (tangent[0] * direction[0] + tangent[1] * direction[1]) / (
            math.hypot(*tangent) * math.hypot(*direction)
        )
        assert cosine > 0.98


def test_adjusted_target_journey_has_no_decorative_progress_animation() -> None:
    source = _visible_source(INFOGRAPHIC.read_text())
    journey = re.search(
        r'(<div class="weight-journey".*?</div>)', source, flags=re.DOTALL
    )

    assert journey is not None
    drawing = journey.group(1)
    assert "Animated journey" not in drawing
    assert 'class="active-wire"' not in drawing
    assert 'class="cursor"' not in drawing
    assert "@keyframes journey" not in source
    assert ".journey-svg .cursor" not in source


def test_exl3_comparison_explains_both_codecs_and_the_required_evidence() -> None:
    visible = _visible_source(INFOGRAPHIC.read_text())
    section = re.search(
        r'(<section class="comparison".*?</section>)', visible, flags=re.DOTALL
    )

    assert section is not None
    comparison = section.group(1)
    required_in_order = (
        "A language model is software trained to predict the next piece of text",
        "billions of learned numbers called weights",
        "Lossy compression stores less data by allowing controlled numerical differences",
        "Replacing detailed values with compact codes and approximate values is called quantization",
        "A trellis is a directed graph repeated across the ordered weights",
        "the encoder searches complete paths",
        "How EXL3 compresses one weight matrix",
        "Spread unusually large values.",
        "Search linked reconstruction choices.",
        "Account for earlier errors.",
        "Store the winning route.",
        "What QSRT keeps and what it changes",
        "Shared foundation",
        "Reconstruction graph and law",
        "Optimization scope",
        "Allocation and storage",
        "The comparison requires two measurements",
    )
    positions = [comparison.index(phrase) for phrase in required_in_order]
    assert positions == sorted(positions)
    assert "detailed weight +0.314 as code 010" in comparison
    assert "three zero-or-one digits called bits" in comparison
    assert "A decoder reads that code and rebuilds +0.301" in comparison
    assert "0.0125 is much smaller than 0.6725" in comparison
    assert "The decoder does not search; it follows the stored path" in comparison
    assert "ExLlamaV3 is software for running compressed language models" in comparison
    assert "EXL3 weight format specifies the stored codes, supporting data, and decoder rules" in comparison
    assert "A fixed calculation that converts an edge number" in comparison
    assert "called a procedural codebook" in comparison
    assert "EXL3 supplies codebooks called MCG and MUL1" in comparison
    assert "history state is temporary memory" in comparison
    assert "A closed route ends in the same history state where it began" in comparison
    assert "condition called tail-biting" in comparison
    assert "Stratified Quantile Graph, or SQG" in comparison
    assert "K2, K3, and K4 mean two, three, and four stored branch bits per weight" in comparison
    assert "X4T reproduces the official four-bit expert values" in comparison
    assert "Forward KLD measures how far each candidate's output probabilities" in comparison
    assert "third high-quality reference" in comparison
    assert "complete serialized checkpoint is smaller" in comparison
    assert "paired document-disjoint teacher-to-candidate forward KLD is lower" in comparison
    assert "No complete EXL3-versus-QSRT quality result exists yet" in comparison
    for pictorial_title in (
        "Detailed weights become compact codes, and linked choices make a complete-path search useful",
        "The EXL3 matrix encoding pipeline",
        "The QSRT expert encoding and packaging pipeline",
        "A third high-quality reference supplies probabilities",
    ):
        assert pictorial_title in comparison


def test_exl3_comparison_and_trellis_have_narrow_layout_fallbacks() -> None:
    source = INFOGRAPHIC.read_text()

    assert ".map { position:static; width:100%; height:auto; display:grid; grid-template-columns:repeat(5,1fr)" in source
    assert ".qsrt-parts,.acceptance,.baby-steps,.worked-grid,.code-explainer,.expert-primer,.codec-lanes { grid-template-columns:1fr; }" in source
    assert ".comparison-grid { grid-template-columns:1fr; }" in source
    assert ".actual-trellis > svg,.journey-svg { min-width:680px; }" in source
    assert ".path-candidate-grid { grid-template-columns:1fr; }" in source


def test_trellis_contract_defines_the_encoded_object_and_every_graph_role() -> None:
    visible = _visible_source(INFOGRAPHIC.read_text())

    assert "stores learned numbers, called weights, in rectangular tables called matrices" in visible
    assert "The original matrix is the object being compressed" in visible
    assert "rebuild 256 detailed numbers from a few stored bits per number" in visible
    assert "Independent two-bit rounding always chooses from the same four" in visible
    assert "its history state changes which four approximate values are available" in visible
    assert "searches the legal linked choices and keeps the closed path" in visible
    assert visible.index("QSRT must replace every detailed number") < visible.index(
        "The production trellis does not receive"
    )
    assert "does not receive an untouched source-matrix entry directly" in visible
    assert "reversible Hadamard mixing transform" in visible
    assert "BlockLDLQ then adjusts later targets" in visible
    assert "The encoder begins with a list of numbers" in visible
    assert "Each number is called a target" in visible
    assert "A sequence index is its position in the list" in visible
    assert visible.index("A sequence index is its position in the list") < visible.index(
        "QSRT cuts the adjusted matrix into 16-by-16 tiles"
    )
    assert "visits all 256 cells in the fixed order required by the decoder" in visible
    assert "That 256-target list is the trellis input" in visible
    assert "same mapping from each sequence index to one tile cell" in visible
    assert "ordered tile" not in visible.lower()
    assert "weight position" not in visible.lower()
    assert "rather than a model neuron or activation" in visible
    assert "With two stored choice bits, called K2, a node offers four edges" in visible
    assert "K3 offers eight edges, and K4 offers sixteen" in visible
    assert "The output is one packed branch per target" in visible
    assert "lowest sum of squared differences" in visible
    assert "It explains graph mechanics; it does not predict production reconstruction quality" in visible
    for value in (
        "−0.107966",
        "+0.166101",
        "−0.332202",
        "+0.024915",
    ):
        assert value in visible


def test_diagrams_use_pictorial_line_art_for_each_major_mechanism() -> None:
    visible = _visible_source(INFOGRAPHIC.read_text())

    assert visible.count('class="pipeline-icon"') == 5
    assert visible.count('class="candidate-route"') == 16
    for pictorial_element in (
        "weight dial",
        "railway switch",
        "edge cassette",
        "T12 shelf",
        "weight ruler",
        "output-probability bars",
        "down-matrix mixer",
        "bell-shaped reference distribution",
        "weighted finite-E4M3 center",
        "Complete production tiles",
    ):
        assert pictorial_element in visible
    for drawing_class in (
        "draw-paper",
        "draw-panel",
        "draw-fill",
        "draw-soft",
        "draw-accent",
        "draw-accent-dot",
    ):
        assert visible.count(f'class="{drawing_class}"') >= 5
    step_four = re.search(
        r'(<svg[^>]+aria-label="A highlighted train route.*?</svg>)',
        visible,
        flags=re.DOTALL,
    )
    assert step_four is not None
    assert 'orient="auto"' in step_four.group(1)
    assert step_four.group(1).count('marker-end="url(#step4-route-arrow)"') == 2


def test_glm_expert_primer_defines_up_and_down_weights_before_proposals() -> None:
    source = INFOGRAPHIC.read_text()
    visible = _visible_source(source)

    primer = visible.index("What “up weight” and “down weight” mean")
    proposal = visible.index("Train the <em>reconstruction values</em>")
    assert primer < proposal
    assert "rectangular tables called matrices" in visible
    assert "An <strong>up weight</strong> is one learned number in the up matrix" in visible
    assert "A <strong>down weight</strong> is one learned number in the down matrix" in visible
    assert "A <strong>quantization grid</strong> is a set of approximate values" in visible
    assert "It is not a matrix" in visible
    assert "up-grid weight" not in visible
    assert "down-grid weight" not in visible
    for term in ("up weight", "down weight", "quantization grid", "up matrix"):
        assert f'"{term}":' in source


def test_glossary_covers_the_page_with_plain_self_contained_definitions() -> None:
    source = INFOGRAPHIC.read_text()

    for term in (
        "weight",
        "matrix",
        "quantization",
        "lossy compression",
        "weight format",
        "trellis",
        "directed graph",
        "edge",
        "path",
        "state",
        "branch",
        "sequence index",
        "target",
        "coefficient",
        "tile scale",
        "encoder",
        "decoder",
        "Viterbi search",
        "tail-biting",
        "expert",
        "gate",
        "up projection",
        "down projection",
        "SwiGLU",
        "K2",
        "K3",
        "K4",
        "K5",
        "T12",
        "E4M3",
        "FP16",
        "residual",
        "curvature-weighted",
        "coupled transform",
        "coding domain",
        "forward KLD",
        "fit rows",
        "held-out rows",
        "serialized bytes",
        "EXL3",
        "procedural codebook",
        "MCG",
        "MUL1",
        "SQG",
        "tensor-parallel-independent",
        "high-quality reference",
        "checkpoint",
    ):
        assert f'"{term}":' in source
    assert "A weighted calculation followed by a smooth formula" in source
    assert "A scalar representation that stores two choice bits per number" in source
    assert "The evaluation text comes from different source documents" in source
    assert "installGlossary();" in source
    assert 'button.type = "button"' in source
    assert 'button.className = "term"' in source


def test_only_surviving_format_preserving_directions_are_proposals() -> None:
    visible = _visible_source(INFOGRAPHIC.read_text())

    assert "Train the <em>reconstruction values</em>" in visible
    assert "Score scalar paths by their <em>downstream damage</em>" in visible
    assert "Fit the down matrix to the <em>reconstructed hidden values</em>" in visible
    assert "per-coordinate reciprocal up/down balancing increased pooled error" in visible
    assert "Exhaustive KLD search across every gate/up path has no feasible production computation" in visible
    assert "Pair trellises, residual streams, and entropy-coded paths remain deferred" in visible


def test_explainer_states_the_complete_checkpoint_acceptance_condition() -> None:
    visible = _visible_source(INFOGRAPHIC.read_text())

    assert "occupy fewer bytes in its finished files" in visible
    assert "output probabilities must also stay closer to the reference" in visible
    assert "evaluation text from different documents" in visible
    assert "teacher-to-candidate forward KLD" in visible
    assert "named fixed GLM EXL3 checkpoint" in visible
    assert "It does not run a production trellis and must not be used as GLM quality evidence" in visible
    assert "Only complete serialized bytes and paired document-disjoint full-model forward KLD" in visible
    assert "complete serialized bytes" in visible
    assert "equal bits per weight" not in visible.lower()


def test_embedded_default_reports_match_the_eight_weight_illustration() -> None:
    embedded = _embedded_json(INFOGRAPHIC.read_text())

    assert embedded["representative"] == run_benchmark(source_seed=0)
    assert embedded["sweep"] == run_sweep(8, start_seed=0)


def test_embedded_illustration_stress_summary_preserves_all_regressions() -> None:
    stress = _embedded_json(INFOGRAPHIC.read_text())["stress"]
    transitions = stress["heldout_forward_kld_transitions"]

    assert stress["expert_count"] == 256
    assert stress["peak_bits_at_play"] == 30
    expected = {
        "reciprocal_up_down_balance": (0.4675957309560702, 204, 38, 14),
        "fit_kld_selected_scalar_paths": (0.5600457006348085, 163, 79, 14),
        "reciprocal_balance_with_fit_kld_paths": (
            0.6920675785433164,
            241,
            7,
            8,
        ),
        "frozen_upstream_down_refit": (0.2192442707145451, 162, 87, 7),
    }
    for stage, (reduction, improved, unchanged, regressed) in expected.items():
        summary = transitions[stage]
        assert summary["pooled_error_reduction"] == pytest.approx(reduction)
        assert summary["improved_experts"] == improved
        assert summary["unchanged_experts"] == unchanged
        assert summary["regressed_experts"] == regressed
        assert improved + unchanged + regressed == 256


def test_embedded_sources_match_both_repository_modules() -> None:
    source = INFOGRAPHIC.read_text()

    assert _script_text(source, "python-benchmark-source") == (
        BENCHMARK_SOURCE.read_text().rstrip()
    )
    assert _script_text(source, "python-benchmark-dependency-source") == (
        BENCHMARK_DEPENDENCY.read_text().rstrip()
    )


def test_embedded_illustration_is_glm_shaped_and_under_the_drawing_bit_limit() -> None:
    report = _embedded_json(INFOGRAPHIC.read_text())["sweep"]
    problem = report["problem"]

    assert report["stage_names"] == list(STAGE_NAMES)
    assert problem["expert_equation"] == "down(SiLU(gate(input)) * up(input))"
    assert problem["input_dimensions"] == 1
    assert problem["hidden_coordinates"] == 2
    assert problem["output_dimensions"] == 2
    assert problem["gate_weights"] == 2
    assert problem["up_weights"] == 2
    assert problem["down_weights"] == 4
    assert problem["k2_closed_paths_per_two_weight_stream"] == 16
    assert report["bit_budget"] == {
        "source_weight_count": 8,
        "payload_bits": 16,
        "payload_bits_per_weight": 2.0,
        "scalar_k2_history_bits": 14,
        "bits_at_play": 30,
        "limit": 32,
        "aggregate_payload_bits": 128,
    }
    assert "pair" not in " ".join(report["stage_names"])


def test_reconstruction_table_scene_uses_production_paths_and_finite_values() -> None:
    visible = _visible_source(INFOGRAPHIC.read_text())

    assert "The table contains 4,096 finite E4M3 reconstruction values" in visible
    assert "Encode complete 256-position tiles with the production path and scale search" in visible
    assert "curvature-weighted center" in visible
    assert "(−0.50×1 −0.25×2 +0.25×5) ÷ 8" in visible
    assert "+0.03125" in visible
    assert "production_viterbi(real_tiles, table, scales, curvature)" in visible
    assert "it does not create a second payload stream" in visible


def test_model_loss_scene_uses_a_scalable_additive_curvature_cost() -> None:
    visible = _visible_source(INFOGRAPHIC.read_text())

    assert "raw squared error</b> 0.012" in visible
    assert "downstream sensitivity</b> 8.0" in visible
    assert "curvature cost</b> 0.096" in visible
    assert "raw squared error</b> 0.018" in visible
    assert "curvature cost</b> 0.018 · selected" in visible
    assert "two_sided_quadratic_error" in visible
    assert "same Viterbi algorithm, model-aware additive edge cost" in visible
    assert "production-sized CUDA traversal are implemented" in visible
    assert "Source-identity and explicit zero-output-feedback controls now reproduce" in visible
    assert "does not estimate downstream quality" in visible


def test_experiment_order_and_measurement_gate_match_the_research_charter() -> None:
    visible = _visible_source(INFOGRAPHIC.read_text())

    loss_link = visible.index("Score downstream damage")
    down_link = visible.index("Retune the output matrix")
    assert loss_link < down_link
    assert 'data-scene="table"' not in visible
    assert "Repeatable KLD and feedback controls" in visible
    assert "One-sided curvature rejected; two-sided identity audit passes" in visible
    assert "Down refit retained" in visible
    assert "output-gradient capture remains missing" in visible
    assert "changed none of 24 trellis paths" in visible
    assert "all eight expert files remained byte-identical" in visible
    assert "full-matrix CUDA closure passed" in visible
    assert "2.3453% relative to uniform K3" in visible
    assert "87.3960% of K3's excess KLD above EXL3" in visible
    assert "same validated score then prices mixed K3/K4 and residual candidates" in visible


def test_added_byte_follow_ups_state_coupling_and_kill_thresholds() -> None:
    visible = _visible_source(INFOGRAPHIC.read_text())

    assert "These allocation follow-ups begin only after one of the two active codec proposals passes" in visible
    assert "Kimi's coupled transform first mixes gate and up rows" in visible
    assert "both must share one rate" in visible
    assert "largest curvature-weighted residual holds more than 6%" in visible
    assert "two largest hold more than 10%" in visible
    assert "save more error per added serialized byte than K4" in visible
    assert "production quantizer supports K2 through K4" in visible
    assert "research-only FP16 reconstruction endpoint" in visible
    assert "28,638 + 2 × 81 = 28,800" in visible
    assert "75 × 384 = 28,800" in visible


def test_rejected_table_training_retains_its_diagnostic_record_off_navigation() -> None:
    source = INFOGRAPHIC.read_text()

    assert "nearly Gaussian values, negligible adjacent correlation" in source
    assert "cheap gate first" in source
    assert "decoder requirement" in source
    assert "whether B12X can select a table per layer" in source


def test_down_scene_states_the_production_transfer_requirement() -> None:
    visible = _visible_source(INFOGRAPHIC.read_text())

    assert "1.8269% lower KLD than uniform K3" in visible
    assert "EXL3 still lower" in visible
    assert "source hidden rows</b> H" in visible
    assert "rebuilt hidden rows</b> Ĥ" in visible
    assert "minimize ‖Ĥ V<sup>T</sup> − Y‖²" in visible
    assert "production K2 encode of V" in visible
    assert "same production scalar trellis, scale search, and BlockLDLQ path" in visible
    assert "Use complete BF16 GLM matrices, real routed rows" in visible


def test_browser_python_runner_is_bounded_and_bootstraps_the_dependency() -> None:
    source = INFOGRAPHIC.read_text()

    assert '<input id="python-experts" type="number" min="1" max="256"' in source
    assert "clamp(Math.trunc(Number($(\"#python-experts\").value) || 8), 1, 256)" in source
    assert "run_sweep(configured_expert_count)" in source
    assert 'types.ModuleType("qsrt.tiny_improvement_benchmark")' in source
    assert "exec(dependency_source, dependency.__dict__)" in source
    assert "exec(benchmark_source, globals())" in source
    assert "synthetic transitions (not production evidence)" in source
    assert "evidence boundary: the two-position graph" in source
    assert "production requirement: complete 256-position tiles" in source
    assert "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/" in source


def test_explainer_defines_the_cpu_tile_screen_and_gpu_evidence_boundary() -> None:
    visible = _visible_source(INFOGRAPHIC.read_text())

    assert "CPU tile screen" in visible
    assert "complete 256-value fixtures captured after production transforms and BlockLDLQ feedback" in visible
    assert "GPU expert confirmation" in visible
    assert "complete BF16 gate, up, and down matrices" in visible
    assert "The faithful codec microbenchmark requires the pinned BF16 checkpoint" in visible
    assert "cannot execute in a browser Python runtime" in visible


def test_text_colors_meet_wcag_aa_on_their_surfaces() -> None:
    paper = {
        "backgrounds": ("#f2eede", "#e6e2d3"),
        "foregrounds": (
            "#1a1a1a",
            "#45433c",
            "#625f57",
            "#155fae",
            "#216609",
            "#a93423",
            "#6d5b00",
        ),
    }
    dark = {
        "backgrounds": ("#1e1f1c", "#272822"),
        "foregrounds": (
            "#f8f8f2",
            "#c9cabf",
            "#a7a89f",
            "#66d9ef",
            "#a6e22e",
            "#ff5c8d",
            "#e6db74",
        ),
    }
    source = INFOGRAPHIC.read_text().lower()

    for palette in (paper, dark):
        for background in palette["backgrounds"]:
            for foreground in palette["foregrounds"]:
                assert background in source
                assert foreground in source
                assert _contrast(background, foreground) >= 4.5, (
                    background,
                    foreground,
                )


def test_static_svg_marks_and_labels_stay_inside_viewboxes() -> None:
    visible = _visible_source(INFOGRAPHIC.read_text())
    drawings = re.findall(r"(<svg\b.*?</svg>)", visible, flags=re.DOTALL)

    assert len(drawings) >= 14
    for drawing in drawings:
        root = ElementTree.fromstring(drawing)
        if "viewBox" not in root.attrib:
            continue
        _left, _top, width, height = map(float, root.attrib["viewBox"].split())
        text_boxes: list[tuple[float, float, float, float, str]] = []
        for mark in root.iter():
            tag = mark.tag.rsplit("}", 1)[-1]
            if tag == "circle":
                x, y, radius = map(
                    float, (mark.attrib["cx"], mark.attrib["cy"], mark.attrib["r"])
                )
                assert 0 <= x - radius <= x + radius <= width
                assert 0 <= y - radius <= y + radius <= height
            elif tag == "rect" and "transform" not in mark.attrib:
                x = float(mark.attrib.get("x", 0))
                y = float(mark.attrib.get("y", 0))
                assert 0 <= x <= x + float(mark.attrib["width"]) <= width
                assert 0 <= y <= y + float(mark.attrib["height"]) <= height
            elif tag == "text":
                label = " ".join("".join(mark.itertext()).split())
                css_class = mark.attrib.get("class", "")
                if "small" in css_class:
                    font_size, width_factor = 9.0, 0.60
                elif "diagram-" in css_class:
                    font_size, width_factor = 10.0, 0.60
                elif "candidate" in root.attrib.get("class", ""):
                    font_size, width_factor = 8.0, 0.60
                else:
                    font_size, width_factor = 11.0, 0.56
                estimated_width = max(
                    font_size * 0.5,
                    len(label) * font_size * width_factor,
                )
                x = float(mark.attrib["x"])
                y = float(mark.attrib["y"])
                anchor = mark.attrib.get("text-anchor", "start")
                if anchor == "middle":
                    left = x - estimated_width / 2
                elif anchor == "end":
                    left = x - estimated_width
                else:
                    left = x
                right = left + estimated_width
                top = y - font_size
                bottom = y + font_size * 0.22
                assert 0 <= left <= right <= width, label
                assert 0 <= top <= bottom <= height, label
                text_boxes.append((left, top, right, bottom, label))

        for index, first in enumerate(text_boxes):
            for second in text_boxes[index + 1 :]:
                horizontal_overlap = min(first[2], second[2]) - max(
                    first[0], second[0]
                )
                vertical_overlap = min(first[3], second[3]) - max(
                    first[1], second[1]
                )
                assert horizontal_overlap <= 2 or vertical_overlap <= 1, (
                    first[4],
                    second[4],
                )


def test_artifact_is_self_contained_except_for_pinned_browser_python() -> None:
    source = INFOGRAPHIC.read_text()

    assert INFOGRAPHIC.stat().st_size < 1_000_000
    assert "fetch(" not in source
    assert "<iframe" not in source
    external_urls = set(re.findall(r'https://[^"\s]+', source))
    runtime_urls = {
        "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/",
        "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js",
    }
    citation_urls = external_urls - runtime_urls
    assert runtime_urls <= external_urls
    assert len(citation_urls) == 29
    assert all(
        url.startswith(
            (
                "https://doi.org/",
                "https://arxiv.org/abs/",
                "https://aclanthology.org/",
                "https://proceedings.mlr.press/",
                "https://proceedings.mlsys.org/",
                "https://proceedings.neurips.cc/",
            )
        )
        for url in citation_urls
    )
    assert ".animation,.actual-trellis,.weight-journey { overflow-x:auto; }" in source
    assert ".animation-head,.mechanism-svg { min-width:640px; }" in source
    assert ".actual-trellis > svg,.journey-svg { min-width:680px; }" in source


def test_research_corpus_is_a_separate_switchable_view() -> None:
    source = INFOGRAPHIC.read_text()
    visible = _visible_source(source)

    assert 'data-view-panel="mechanism"' in visible
    assert 'id="research-view" data-view-panel="research" hidden' in visible
    assert 'data-view-choice="mechanism" aria-pressed="true"' in visible
    assert 'data-view-choice="research" aria-pressed="false"' in visible
    assert visible.count('data-family="') == 21
    assert "21 techniques shown" in visible
    for family in ("signal", "trellis", "objective", "recovery", "allocation"):
        assert f'data-research-filter="{family}"' in visible
        assert f'data-family="{family}"' in visible
    for status in ("adopted", "measure", "deferred", "rejected"):
        assert f'data-status="{status}"' in visible
    assert "function setView(name)" in source
    assert "function showResearch(" in source
    assert "function filterResearch(family)" in source
    assert "researchSections.includes(initial)" in source


def test_research_corpus_separates_sources_transfers_costs_and_evidence() -> None:
    visible = _visible_source(INFOGRAPHIC.read_text())
    research = visible.split('id="research-view"', 1)[1]

    assert "Published evidence" in research
    assert "Repository evidence" in research
    assert "Proposed transfer" in research
    assert research.count("Source demonstrated") >= 15
    assert research.count("QSRT translation") >= 15
    assert research.count("Cost or risk") >= 7
    assert "Failure mode" in research
    assert "No complete GLM-5.2 candidate has passed this gate" in research
    assert "reproduced unchanged and direct-return identity controls bit for bit" in research
    assert "increased mean full-model KLD by 2.3453% relative to uniform K3" in research
    assert "recovered 87.3960% of K3's excess above EXL3" in research
    assert "No source in this library establishes complete-checkpoint superiority over EXL3" in research


def test_research_corpus_states_the_exact_checkpoint_acceptance_gate() -> None:
    visible = _visible_source(INFOGRAPHIC.read_text())

    assert "QSRT serialized bytes &lt; EXL3 serialized bytes" in visible
    assert "QSRT forward KLD &lt; EXL3 forward KLD" in visible
    assert "payloads, tables, scales, indices, metadata, alignment, padding" in visible
    assert "paired document-disjoint prompts" in visible
    assert "one separate high-quality reference" in visible
    assert "identical non-expert weights" in visible
    assert "repeatability control" in visible


def test_research_cards_use_full_primary_source_citations() -> None:
    visible = _visible_source(INFOGRAPHIC.read_text())
    cards = re.findall(
        r'(<article class="paper-card".*?</article>)',
        visible,
        flags=re.DOTALL,
    )

    assert len(cards) == 21
    for card in cards:
        assert 'class="citation"' in card
        assert 'target="_blank" rel="noreferrer"' in card
        assert re.search(r'href="https://(?:doi\.org|arxiv\.org|aclanthology\.org|proceedings\.)', card)
    assert visible.count('class="research-bibliography"') == 1
    bibliography = visible.split('class="research-bibliography"', 1)[1]
    assert bibliography.count("<li>") == 29


def test_research_view_has_narrow_layout_fallbacks() -> None:
    source = INFOGRAPHIC.read_text()

    assert ".research-shell { grid-template-columns:1fr; }" in source
    assert ".research-hero,.research-section-head { grid-template-columns:1fr; gap:20px; }" in source
    assert ".research-orientation,.evidence-ladder,.paper-grid { grid-template-columns:1fr; }" in source
    assert ".research-gate { grid-template-columns:1fr; gap:10px; }" in source
    assert ".paper-body { grid-template-columns:1fr; }" in source
    assert "#motion-button,#theme-button { display:none; }" in source

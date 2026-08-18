# QSRT

QSRT is a fixed-payload weight codec for gated mixture-of-experts models. The
name means **Quantile-Stratified Rate-shifted Trellis**.

QSRT encodes each weight sequence as a path through a finite-state graph. The
stored bits select both the current reconstruction value and the state that
determines future choices. Each state exposes values from different probability
regions, giving a low-rate path broad sign and magnitude coverage. Reconstructed
weights use a finite eight-bit floating-point alphabet.

The Kimi-K3 implementation supports:

- a uniform two-bit routed-expert representation that occupies 2.004464 bits
  per weight including scales;
- a 3.083333-trellis-bpw all-QSRT representation with two four-bit records and
  twenty-two three-bit records per expert matrix;
- equal-size exchanges between two- and four-bit records around a three-bit
  baseline; and
- an exact high-quality endpoint that preserves the source model's four-bit
  microscaled (MXFP4) weights.

The two-bit representation uses an exact coupled Hadamard change of basis
across the gate, up, and down matrices. The encoder reconstructs the quantized
gate and up projections before deriving the down-projection covariance, then
selects complete expert candidates using naturally routed activation error.

This repository contains the offline encoder, calibration and covariance
tools, candidate selection, canonical checkpoint storage, and correctness
validation. Production model integration lives in vLLM, and GPU kernels live
in B12X.

## Setup

```bash
uv sync --dev
.venv/bin/pytest -q
```

## Documentation

- [Two-bit codec](docs/qsrt-2bpw-codec.md) — authoritative specification for
  the uniform two-bit representation.
- [Complete technical specification](docs/qsrt-technical-brief.md) — full
  system, storage, allocation, evidence, and implementation status.
- [Two-bit quality research corpus](docs/qsrt-two-bit-research-corpus.md) —
  evidence boundaries, literature, proposals, and decisive experiments.
- [Comparative assessment of QSRT and EXL3](docs/qsrt-exl3-comparative-assessment.md) —
  measured per-symbol results, byte accounting, evidence audit, and judged
  probabilities.
- [GLM-5.2 experiment journal and artifact map](docs/glm52-experiment-journal.md) —
  chronological operations, immutable inputs, generated files, container
  identities, measurements, and cleanup policies.
- [GLM-5.2 layer-3 KLD results](docs/glm52-layer3-kld-results.md) — repeatable
  one-context mechanism comparisons and an independent 16-document auxiliary
  replication of the frozen low-rank correction against EXL3.
- [QSRT improvement strategy](docs/qsrt-improvement-strategy.md) — model-loss
  curvature implementation status, measured mechanism decisions, and exact-byte
  promotion gates.
- [GLM-5.2 mixed-K3/K4 allocation pre-registration](experiments/glm52_layer3_k3_k4_allocation_pre_registration.json) —
  frozen twelve-projection byte budget, fixed EXL3-rate control, deterministic
  complete-expert selection rule, and reporting-data prohibition.
- [GLM-5.2 low-rank auxiliary replication receipt](experiments/glm52_layer3_rank4_expert103_public_reference_auxiliary_result.json) —
  frozen candidate identity, public-reference provenance, paired
  document-bootstrap result, controls, and the unmet qualification boundary.
- [Audit of the K2 no-feedback claim](docs/qsrt-blockldlq-no-feedback-audit.md) —
  scale-selection confound, greedy-feedback counterexample, and the frozen-scale
  K3 measurement that produced byte-identical feedback and no-feedback experts.
- [Interactive scalar-trellis walkthrough](docs/viterbi-trellis-explainer.html)
- [Interactive QSRT explanation and trellis-quantization research corpus](docs/qsrt-three-improvements-infographic.html)

Run the CPU-only tiny benchmark with its default eight-expert sweep:

```bash
.venv/bin/python scripts/benchmark_qsrt_tiny_improvements.py
```

Pass `--experts 4` or another positive count to change the sequential sweep.
Pass `--experts 1` to emit the detailed report for one expert.
Pass `--pair-table-correlation -0.7` to stress a pair table trained on a
weight distribution that does not match the default synthetic experts.
The benchmark compares a matched-payload scalar K2 control with the proposed
gate/up pair trellis on disjoint synthetic fit and held-out rows. Its two-output
expert result passes through a fixed four-logit readout before forward
Kullback–Leibler divergence (KLD) is measured. KLD compares the teacher's and
candidate's output probability distributions. The result is a falsification
proxy. It does not measure EXL3 or full-model quality.

Generated checkpoints, calibration captures, traces, and benchmark output are
not source artifacts and must not be committed.

# Comparative assessment of QSRT and EXL3

Status: technical review of the repository state at 2026-08-15. This is an
explicitly evaluative document. It records findings, measurements, and
judgments from one review; it does not define the codec or its roadmap. The
implemented format is defined by
[the two-bit codec specification](qsrt-2bpw-codec.md) and
[the technical brief](qsrt-technical-brief.md); the research program is
defined by [the research corpus](qsrt-two-bit-research-corpus.md).

## Question and criterion

QSRT is the repository's fixed-payload trellis weight codec for the routed
experts of Kimi-K3. EXL3 is the deployed comparison checkpoint at
`/models/Kimi-K3-EXL3-3p09`, produced by the ExLlamaV3 trellis quantizer.
The governing objective, stated in the research corpus, is compression
dominance: a complete QSRT artifact that occupies fewer exact serialized
bytes than the complete EXL3 checkpoint and has lower paired held-out forward
Kullback-Leibler divergence (KL). Exact bits-per-weight parity is
irrelevant to the criterion; both inequalities are required.

Two scenarios are assessed separately:

- **The 3-bit-class contest.** The sealed all-QSRT artifact with the
  22-record-K3 plus 2-record-K4 schedule, where K2, K3, and K4 name the
  two-, three-, and four-bit trellis rates (3.088232 container bits per
  weight, 1,051,056,799,744 bytes), against the EXL3 checkpoint under its
  inherited byte cap of 1,058,586,247,168 bytes. The QSRT artifact is 0.71%
  smaller, so the byte inequality currently holds and the contest is over
  KL.
- **The two-bit endpoint.** The uniform two-bit artifact (2.004473 container
  bits per weight, 682,207,608,832 bytes, 35.6% smaller than the cap)
  against the same checkpoint.

The review combined five analyses: a per-symbol rate-distortion measurement
with new CPU experiments, a system-design assessment of the machinery above
the per-symbol code, a byte-accounting audit of both artifacts, an inventory
of the repository's quantitative evidence, and an adversarial pass whose job
was to break the thesis. Code citations give the file and line that were
verified during the review; line numbers refer to the working tree at the
review date.

## Summary

The per-symbol code is measured to be slightly better than EXL3's, the
system-level machinery is sound but almost entirely absent from the sealed
3-bit artifact, and the decisive comparison has never been run on either
side. Judged probabilities: roughly 0.35 that the 3-bit-class contest ends
in QSRT's favor if the comparison is run fairly, and roughly 0.02 that the
two-bit artifact wins without training-scale recovery.

Four facts carry most of that judgment:

1. **Per-symbol labels are a small measured advantage.** On an independent
   standard-normal source with an exact full Viterbi search, the
   production Stratified Quantile Graph (SQG) labels achieve 2.62% (K2)
   and 3.00% (K3) lower mean squared error than MCG, the computed
   codebook EXL3 deploys. The advantage is worth about 0.022 bits per
   weight, or 0.7% of trellis payload.
2. **The sealed 3-bit artifact carries almost none of the measured
   system-level gains.** The coupled transform and draw selection were
   measured only on the uniform two-bit profile and are not part of the
   sealed `k3x22_k4x2` artifact. The artifact also contains zero exact
   high-quality (X4T) promotions, while the EXL3 cap reconstructs as a
   hybrid that stores roughly 8% of experts losslessly.
3. **No QSRT-versus-EXL3 quality measurement exists.** The repository
   contains no EXL3 KL number in any branch, and the sealed 3-bit artifact
   has no full-model KL of its own. The paired comparison machinery in
   `qsrt/kld_gate.py` is complete and has never been executed.
4. **The measurement instrument cannot certify the expected effect.** The
   gate's hard-coded 0.004 absolute noise floor is applied to paired
   differences, where correlated runtime noise cancels. The expected paired
   effect in the 3-bit contest is on the order of 0.0002 KL. As configured,
   a true win of that size returns `repeat_required` indefinitely.

## Per-symbol code, measured

The measurement harness is committed at
`qsrt/synthetic_source_distortion.py` with the driver
`scripts/measure_synthetic_source_distortion.py`; the numbers below
reproduce from it deterministically (seed 4242, 256 sequences, 65,536
scored interior symbols per configuration, one fitted global scale per code,
mirroring the production one-scalar-per-matrix fit in
`qsrt/exl3_encoder_backend.py`). MCG and MUL1 are EXL3's two native
computed codebooks, transcribed from the kernel sources; T12 is the frozen
4,096-entry modal staircase over the top twelve rank bits; E4M3 is the
finite eight-bit floating-point alphabet with four exponent and three
mantissa bits.

| Code | K2 MSE | K3 MSE | Relative to MCG (K2 / K3) | Distinct values |
| --- | --- | --- | --- | --- |
| EXL3 MCG | 0.068223 | 0.017334 | reference | 10,746 |
| EXL3 MUL1 | 0.068170 | 0.017322 | -0.08% / -0.07% | 913 |
| SQG production (T12, E4M3) | 0.066435 | 0.016814 | **-2.62% / -3.00%** | 151 |
| SQG exact rank, E4M3 endpoint | 0.066427 | 0.016812 | -2.63% / -3.01% | 155 |
| SQG exact rank, FP16 endpoint | 0.066430 | 0.016791 | -2.63% / -3.13% | 14,888 |
| SQG menu-oriented control | 0.068750 | 0.017729 | +0.77% / +2.28% | 155 |

The Gaussian rate-distortion bound is 0.0625 at two bits and 0.015625 at
three. SQG sits 6.3% and 7.6% above the bound; MCG sits 9.2% and 10.9%
above it. The paired standard error on the SQG-to-MCG ratio is 0.0025 at K2
and 0.0030 at K3, so both margins are about ten standard errors. The
harness passed five independent validations, recorded in
`tests/test_synthetic_source_distortion.py`: a Lloyd-Max scalar reference at
0.1177 against the literature value 0.1175, an exact collapse of a
degenerate trellis to scalar quantization, reproduction of the encoder's
hard-coded codebook-0 standard deviation to eight significant figures,
reproduction of MCG's 10,746 distinct values, and an MCG operating point
inside 0.068 to 0.069, the two-bit Gaussian trellis result published by
QTIP, the trellis-coded quantization method EXL3 implements.

Three prior concerns are resolved by the ablations:

- **The E4M3 endpoint does not bind at K2 or K3.** A deleted investigation
  (recoverable with `git show
  dddfc89:docs/glm52-k5-k6-sqg-investigator-brief.md`) found the shipping
  E4M3 endpoint losing to MCG at K5 and K6, with the 151-value alphabet as
  the diagnosed cause. An FP16-endpoint variant won on all seven panel
  experts in the same study. The ablation shows the endpoint costs 0.005% at K2 and
  0.14% at K3: the alphabet ceiling is purely a high-rate phenomenon, and
  changing the frozen K3 endpoint would buy about 0.001 bits per weight.
- **The T12 modal reduction is free at these rates** (at most 0.01% at
  both).
- **The low menu-stratum diversity of the frozen graph is benign.** The
  frozen rank construction exposes on average 2.0437 of 4 nominal coarse
  strata per Viterbi menu at K2 and 7.9062 of 8 at K3; both constants
  reproduce in `qsrt.synthetic_source_distortion.menu_statistics`. The
  minimal reorientation that restores full per-menu stratum coverage
  measures 3.5% worse at K2 and 5.4% worse at K3 and falls below MCG at
  both rates, despite far better memoryless menus. Sequence-space coverage
  is what matters; per-step menu coverage is the wrong figure of merit, and
  orientation repair should not be pursued as a quality lever.

Scope: this ranks reconstruction labels on the shared graph under an
independent Gaussian source with identity curvature and no error feedback.
It establishes that the label choice is a small credit rather than a risk.
It does not predict held-out KL. The real-weight K3 head-to-head with the
production encoder and a fresh MCG control has never been run anywhere in
the repository and is the highest-value follow-up on this dimension.

## Byte accounting

Recomputed from `qsrt/qsrt_atoms_v2.py` by executing the layout code:

- **Uniform two-bit profile:** 2.000000 trellis + 0.004464 local scales +
  0.000009 shared sections and headers = **2.004473** container bits per
  weight, 682,207,608,832 bytes across 92 layers.
- **Sealed 3-bit-class profile (`k3x22_k4x2`):** 3.083333 trellis +
  0.004464 scales + 0.000425 atom-stride padding + 0.000009 sections =
  **3.088232**, 1,051,056,799,744 bytes. The padding is structural: the K4
  record placement leaves 64 of 96 atom slots 21,504 bytes short of the
  stride, about 145 MB model-wide. The coupled storage variant uses
  pair-variable strides and has zero padding at 3.087806.

The EXL3 side cannot be recounted from this repository. The inherited cap
is a bare command-line literal (`AGENTS.md:370`) that no code has ever
verified against the artifact. Dividing it by the 2,722,740,830,208 routed
expert weights gives 3.110355 bits per weight. That figure reconstructs
almost exactly as a hybrid: roughly 8.1% of experts retained in the source
model's own microscaled four-bit floating-point format (MXFP4) at 4.25
bits per weight and the remainder at EXL3 K3 with
per-expert scale vectors at about 3.010. The reconstruction is an
inference; a plain uniform mix reaching the same total would erase the
conclusions that depend on it. If a recount lands at the directory token's
nominal 3.09 rather than 3.110, QSRT's byte margin shrinks from 0.71% to
0.06%, roughly 602 MB in 1.05 TB, and any discovered index or alignment
plane could flip its sign.

Two asymmetries sit outside both rate ledgers:

- **Non-expert overlay.** QSRT model views hard-code a microscaled
  eight-bit floating-point (MXFP8) non-expert overlay
  (`qsrt/pack/qsrt_model_view.py`), while the EXL3 checkpoint carries
  16-bit brain-float (BF16) non-experts. A paired measurement on the unmerged branch
  `838b00e` puts the BF16 overlay 0.00164 KL ahead (confidence interval
  -0.00177 to -0.00151, 1,024 contexts). That is roughly eight times the
  expected codec-level effect and points against QSRT. The corpus requires
  hash-identical non-experts only for attribution experiments; the release
  comparison needs the same requirement.
- **Scale planes.** QSRT stores 18,432 scale bytes per expert (0.004464
  bits per weight) against an estimated 39,936 for EXL3 (0.009673). The
  EXL3 figure is QSRT's own accounting of undeduplicated per-expert sign
  vectors rather than a measurement of the artifact, and the difference is
  worth about 1% of squared error at fixed bytes.

## System-level mechanisms

Assessed individually against what per-matrix EXL3 encoding already does
(randomized sign-vector incoherence processing plus Hessian-driven
blockwise error feedback):

- **Coupled gate/up rotation.** Sound and exactly closed around SiTU,
  Kimi-K3's coordinatewise gated activation
  (`qsrt/qsrt_coupled.py:145-196`, decode at `:254-269`).
  Measured at 3.052% pooled routed squared-error reduction on 24 experts —
  but under identity curvature (`qsrt/qsrt_codec_pilot.py:332` leaves the
  covariance unset, and `qsrt/ldlq.py:17-19` substitutes the identity), so
  dense-covariance error feedback had nothing to adapt to in either arm.
  An estimated half to three quarters of the gain overlaps what dense
  feedback provides on its own. The control arm did include EXL3-style
  random sign vectors, so the gain is on top of EXL3-style conditioning.
  The mechanism is absent from the sealed 3-bit artifact.
- **Draw selection.** The eight-member deterministic sign family with the
  fit-proposes, confirm-against-zero protocol
  (`qsrt/qsrt_coupled_plan.py:61-67`) is well guarded against the
  confirmation fold becoming an oracle. Measured at 1.308% pooled on a
  disjoint corpus with a bootstrap interval of 0.364% to 1.864%. Also
  absent from the sealed 3-bit artifact. EXL3's randomized incoherence
  runs unchanged underneath it (`qsrt/qsrt_codec_pilot.py:335-336` passes
  the sign seeds through to the backend), so nothing was given up to add
  it.
- **Down-projection conditioning on reconstructed activations.** The
  encoder rebuilds each expert's down-projection covariance from the
  decoded gate and up activations before encoding the down matrix. This is
  the one mechanism per-matrix quantization structurally cannot express,
  it is present in the sealed artifact, and it composes with blockwise
  feedback rather than overlapping it. No isolated ablation against
  source-statistics conditioning exists; the related dense-refit oracle
  improved 20 of 28 experts with a 1.55% median.
- **Covariance policy.** Layer-global input covariance where the basis is
  shared, expert-local post-SiTU covariance with
  support-dependent shrinkage where it is not, and rejection of the pooled
  post-SiTU covariance as invalid. Statistically careful. One untuned
  constant deserves attention: `PHASE1_H2_EXPERT_LOCAL_ALPHA = 0.75`
  (`qsrt/qsrt.py:231`) blends at least 25% identity into every expert's
  local covariance, including fully supported experts, with no recorded
  sweep.
- **Mode selection and the fixed rate schedule.** The headline 11.907%
  improvement of the 22-K3 plus 2-K4 schedule over uniform K3 is a rate
  purchase. The schedule spends 0.0833 more bits per weight, and on the
  repository's own measured distortion slope of about six-to-one per bit
  (`out/report.md`), a uniform rate increase of that size predicts a
  13.8% reduction. The remaining allocation content is the 0.35% or less
  by which the fixed schedule beat the tile-selector alternatives. The
  byte-neutral per-expert mode pool has no measured benefit against a
  matched all-baseline-mode control. Its acceptance test takes an argmin
  over nine cells and tests it on the same confirmation documents, with a
  zero minimum-improvement threshold and no multiplicity correction
  (`qsrt/qsrt_candidates.py:856-870`, `:793`). The observed 11.4%
  acceptance rate is statistically indistinguishable from selection under
  the null.
- **Exact-tier allocation.** The mechanism with enough leverage to decide
  the contest. X4T stores a lossless expert at about 4.03 bits per weight
  against the 4.25 the raw MXFP4 source costs (`qsrt/x4t.py:133-138`,
  corroborated by the measured 4.00 effective entropy bound in
  `out/report.md`). Every exact promotion therefore costs QSRT about 5%
  fewer bytes than the EXL3 hybrid pays for the same zero error. The Lagrangian
  allocator exists (`qsrt/pack/qsrt_allocation.py:121-158`), but the
  all-expert cost index, the damage scores, and a materialized mixed
  artifact do not. Under the hybrid reconstruction, EXL3's coarse
  allocation is probably ahead today: removing all error from the
  most-damaged 8% of experts beats a uniform schedule whenever those
  experts carry more than about 16% of total damage, which heavy-tailed
  routed sensitivity makes likely.

Two transfer risks apply to every squared-error figure above. First, no
measurement anywhere in the repository relates routed squared error to
held-out KL. The review's central estimate for the transfer coefficient is
about 0.5 with substantial mass at zero. Two recorded proxy inversions
inform that estimate: a proxy-selected scale gauge worsened real error by
2.250%, and a synthetic pair table improved its training metric while
regressing held-out KL by 0.72%. Second, gains compose sublinearly: the
one measured stack recovered 0.448% from mechanisms individually worth
1.308%.

## Evidence base

The complete set of full-model KL numbers in the repository, all branches:

| Measurement | Value | Where |
| --- | --- | --- |
| Uniform two-bit artifact, 32 windows, eight-way tensor parallel (TP8), MXFP8 overlay | 0.0851995464 | `docs/qsrt-technical-brief.md` |
| Uniform two-bit artifact, 1,024 contexts, sixteen-way tensor parallel (TP16), BF16 overlay | 0.06538554 (CI 0.06102-0.07075) | unmerged branch `838b00e` |
| Paired overlay swap, BF16 minus MXFP8 | -0.00163534 (CI -0.00177 to -0.00151) | same branch |

The sealed 3-bit artifact has no full-model KL. No EXL3 checkpoint has ever
been measured for KL. The two two-bit numbers differ by 23% because suite,
tensor-parallel degree, and overlay all differ; absolute KL is not portable
across configurations, and only a strictly paired comparison on one suite
means anything at these effect sizes. The stronger of the two measurements,
and the paired overlay study, are visible only on an unmerged branch while
the documentation cites the weaker one.

One head-to-head against EXL3's codebook family exists on real weights: a
24-expert, three-rate study in which SQG beat both external controls on all
216 comparisons (`docs/qsrt-technical-brief.md`). The repository itself
limits its weight: the winning variant is an offline control rather than
the shipping profile, the quoted margin is against MUL1, the down-matrix
metric used the covariance the project later ruled invalid, and the
candidate shards predate the current graph. The deleted GLM-5.2
investigation is the only study with matched fresh controls, and its
shipping-configuration result was a loss at K5 and K6 that the endpoint
ablations above now bound away from K2 and K3.

## The two-bit endpoint

The deficit is a factor, and the levers are percentages. The gap from
2.004 to 3.088 bits per weight implies a distortion ratio between 4.5
(theoretical slope) and 7.3 (the repository's measured scalar slope).
Closing it at fixed rate requires roughly a 78% to 86% error reduction.
The full measured-plus-oracle stack, credited generously and ignoring
sublinear composition, is worth 10% to 15%. The source model's own
measured code entropy is about 4.00 bits per weight, so a two-bit code
discards about 47% of the information in the weights it stores. The
published finding that two-bit representations require training-scale
recovery (ParetoQ, indexed in the corpus) points the same direction. The
uniform two-bit artifact is best understood as the codec-and-runtime
integration proof the technical brief describes. A two-bit quality program
that does not include behavioral recovery as a planned phase is unlikely to
close this gap.

## Judged probabilities

These are review judgments, stated with their main sensitivities.

- **3-bit-class contest, run fairly: about 0.35.** Two sensitivities point
  up. The EXL3 checkpoint may have been calibrated on a small generic
  corpus, in which case QSRT's million-token natural-routing calibration
  and MoE-aware covariances could be worth far more than every codec-level
  figure in this review. And if the hybrid reconstruction of the cap is
  correct, QSRT's direct fractional rate plus cheaper exact tier are a
  structural advantage at equal bytes. Three point down: an adverse byte
  recount, an unequalized non-expert overlay, and the possibility that the
  clean-source label advantage does not survive the production encoder.
- **Two-bit endpoint without recovery training: about 0.02**, for the
  reasons in the preceding section. The estimate would change materially
  only if the EXL3 checkpoint's own KL turned out to be several times
  worse than the extrapolation used here.

## What would decide it

In order, with each step's reason:

1. **Recount the EXL3 artifact's routed-expert bytes from its shard
   headers.** The entire byte margin, and the hybrid reconstruction that
   drives the allocation analysis, rest on one unverified number.
2. **Merge branch `838b00e` and restore the deleted GLM-5.2 investigation
   brief.** The strongest KL measurement, the paired overlay result, and
   the only matched-control codec head-to-head should be visible in the
   tree the project reasons from.
3. **Fix the KL gate's paired noise floor.** Replace the absolute 0.004
   documented-runtime constant with a paired-difference noise model
   estimated from repeat captures of the same artifact pair; the present
   configuration cannot certify effects of the size at stake.
4. **Equalize the non-expert overlay by policy for the release
   comparison**, hash-identical on both sides.
5. **Run the paired KL gate** on the 1,024-context suite: EXL3 against the
   sealed 3-bit artifact, matched tensor parallelism and overlays. The
   machinery is complete; this is days of compute and no new code, and it
   is the first direct answer to the governing question.
6. **Build the all-expert X4T cost index and damage scores, run the
   Lagrangian allocator to the recounted cap, and materialize the mixed
   artifact.** Allocation against the heavy tail of expert damage is the
   only intervention in the repository with plausible leverage above the
   measurement floor.
7. **Run the real-weight K3 label head-to-head** with the production
   encoder and a fresh MCG control, converting the synthetic-source label
   advantage into an artifact-level claim or retiring it.
8. **Deprioritize** the pair-trellis representation until items 1 through
   7 resolve (it targets the K2 rate class, which does not appear in the
   contest artifact, and its one held-out test regressed), endpoint
   changes at K3 (measured headroom 0.14%), and graph-orientation repair
   (measured to lose at both rates).

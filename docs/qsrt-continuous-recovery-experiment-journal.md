# QSRT Continuous-Recovery Experiment Journal

This journal records measurements and decisions for continuous recovery of the
Kimi-K3 uniform two-bit QSRT checkpoint. It complements
`docs/qsrt-continuous-recovery-tuning.md`, which defines the resulting design
and acceptance contract. Entries here retain chronology so that rejected
hypotheses, artifact provenance, and experimental baselines remain auditable.

## 2026-08-17: Anchor routing and boundary divergence

### Artifacts

- Anchor checkpoint:
  `/data/models/Kimi-K3-QSRT-K2-DIRECT-VITERBI-ALL-LINEARS-v1-model`
- Teacher boundary archive:
  `/data/kquant/research/qsrt-continuous-recovery-m0/teacher-legacy32-boundaries`
- Teacher route archive:
  `/data/kquant/research/qsrt-continuous-recovery-m0/teacher-legacy32-routes`
- Student boundary archive:
  `/data/kquant/research/qsrt-continuous-recovery-m0/student-legacy32-boundaries`
- Student route archive:
  `/data/kquant/research/qsrt-continuous-recovery-m0/student-legacy32-routes`

### Findings

The comparison covers 65,536 exact KLD-suite tokens in 32 complete documents
and all 92 routed layers. Mean top-16 route overlap is 0.898409, exact route-set
agreement is 0.185439, and mean marginal expert-frequency total variation is
0.0184905.

Layer 12 is anomalous: route overlap is 0.700719, exact route-set agreement is
0.001099, and marginal total variation is 0.284148. Its incoming boundary has
6.7371% relative L2 error, compared with 6.7293% at boundary 11 and 8.7577% at
boundary 13. This makes layer 12's abrupt routing divergence inconsistent with
a simple monotone growth in hidden-state error. Per-expert payload distortion
and transform-draw attribution remain open.

Router weights and correction-bias tensors are identical between the teacher
and anchor checkpoint. The Kimi gate applies the correction bias only while
selecting the top-16 expert set; route weights come from the unbiased sigmoid
scores. Consequently, the ordinary KL gradient with respect to this bias is
zero almost everywhere. Bias optimization must use expert-frequency feedback,
and its directly controllable metric is marginal frequency total variation.
Screening KLD non-regression is the acceptance gate; routing agreement is a
reported secondary metric.

## 2026-08-17: Packed serving numerics versus decoded-BF16 replay

### Implementation and artifacts

- Parity driver: `scripts/verify_qsrt_packed_numerics.py`
- Combined result:
  `/data/kquant/research/qsrt-continuous-recovery-m0/packed-parity-summary.json`
- Layer cases:
  `packed-parity-layer12-rank0.json`,
  `packed-parity-layer24-rank5.json`, and
  `packed-parity-layer84-rank11.json` in the same directory.

The driver loads real checkpoint payloads through the B12X production loader,
executes the production packed W4A16 MoE kernel, and compares it with B12X's
independent decoded reconstruction and coupled-boundary reference. Inputs are
real routed rows from
`/data/datasets/kquant/captures/k3-all-routed-4m-v1.kqrows`; they provide real
activation magnitudes while the comparison isolates one selected expert per
row with unit route weight.

The B12X benchmark loader was corrected to accept both production identities:
`kquant_kimi_k3_qsrt_atoms_v2` in completion records and
`qsrt_kimi_k3_qsrt_atoms_v2` in layer metadata. Rejecting either identity was a
stale benchmark constraint, not a checkpoint defect.

### Results

The measurement covers 48 rows, 48 distinct routed experts, three layers,
three TP extents, and 10,752 output tiles.

| Layer / TP extent | Full-output relative L2 | Full-output max abs |
| --- | ---: | ---: |
| 12 / 0 | 0.0026725% | 2.8819e-5 |
| 24 / 5 | 0.0319470% | 9.7863e-6 |
| 84 / 11 | 0.0361047% | 4.0546e-5 |

Across all output tiles, relative L2 error has a 0.04130% median, 0.26355%
p99, and 0.43739% maximum. Max-absolute error has a 1.026e-6 median, 2.641e-5
p99, and 4.055e-5 maximum.

### Decision

Train against decoded-BF16 expert weights. The measured difference is bounded
fused-arithmetic rounding, not a reconstruction mismatch, and does not justify
emulating the packed kernel in the recovery trainer.

## 2026-08-17: Boundary-archive storage contract

The `/data` filesystem has 30 TB nominal capacity and 6.22 TiB available. The
4.9 TB routed-row capture at
`/data/datasets/kquant/captures/k3-all-routed-4m-v1.kqrows` remains resident and
is required for routed-row-weighted low-rank error fitting.

A 50-million-token archive containing the exact student state at layer 84 and
teacher normalized LM-head targets requires approximately 5.87 TiB and leaves
insufficient operational headroom. The student state is eight slabs: hidden
boundary 84 plus residual-prefix boundaries 0, 12, ..., 72. The architecture
uses every residual-prefix tensor when executing layers 84-92; they cannot be
deferred. The teacher target adds a ninth slab. The disjoint screening
partition must contain the same state and target because periodic suffix-replay
evaluation reads them directly.

### Decision

Capture the exact eight-slab student state and teacher normalized LM-head targets
for both training and screening partitions. The 4M research-training archive
requires 0.469 TiB and fits. Before writing the 5.867 TiB 50M archive, complete
the low-rank fits and relocate or remove the routed-row capture to restore
operational headroom.

## 2026-08-17: Layer-12 payload-distortion attribution

### Artifacts

- Audit driver: `scripts/audit_qsrt_layer_payload_distortion.py`
- Layer 11 result:
  `/data/kquant/research/qsrt-continuous-recovery-m0/layer-011-direct-payload-distortion.json`
- Layer 12 result:
  `/data/kquant/research/qsrt-continuous-recovery-m0/layer-012-direct-payload-distortion.json`
- Layer 13 result:
  `/data/kquant/research/qsrt-continuous-recovery-m0/layer-013-direct-payload-distortion.json`

The audit decodes the stored uniform-K2 payloads, applies each expert's stored
coupled-Hadamard draw to the official MXFP4 source weights, and measures
additive source-to-decoded squared error in the same transformed coordinates.

| Layer | Relative SSE | Relative L2 |
| --- | ---: | ---: |
| 11 | 0.0673018 | 0.259426 |
| 12 | 0.0672839 | 0.259391 |
| 13 | 0.0673097 | 0.259441 |

Layer 12 has no aggregate payload-distortion anomaly. Its per-expert relative
SSE ranges from 0.0672253 to 0.0677412, with median 0.0672807 and p99
0.0673710. Matrix-relative SSE is 0.0673003 for gate, 0.0672997 for up, and
0.0672508 for down.

Layer 12 selects coupled-Hadamard draw 6 for 186 of 896 experts, the smallest
draw-6 count among the 92 routed layers. Applying draw 6 to every layer-12
expert increases confirmation SSE by 0.1002% relative to draw 0. Within the
186 experts that select draw 6, however, it reduces confirmation SSE by 3.318%
collectively. The low count is therefore consistent with draw 6 being useful
for a minority rather than evidence of a broken selector.

### Decision

Do not attribute layer 12's route divergence to gross K2 reconstruction error
or to the existing draw-0/draw-6 selection rule. Search all eight defined
coupled-Hadamard draws using the optimized C128 K2 encoder, then qualify any
meaningful per-expert distortion reduction through a materialized layer-12
checkpoint and full-model KLD.

### Eight-draw search result

- Search driver: `scripts/search_qsrt_layer_coupled_draw.py`
- Combined selection:
  `/data/kquant/research/qsrt-continuous-recovery-m0/layer-012-k2-eight-draw-selection.json`
- Fixed-draw results:
  `layer-012-k2-draw-0-full.json` through
  `layer-012-k2-draw-7-full.json` in the same directory.

The search used the dedicated K2 CUDA encoder, complete C128 tail-biting, the
same transform seeds and upstream scale-sharing contract as the direct-Viterbi
checkpoint builder, and the canonical K2 reconstruction target. Each expert
selected one draw from the additive W1, W3, and W2 source-to-decoded SSE; the
three matrices did not select draws independently.

| Draw | Winning experts | Share |
| ---: | ---: | ---: |
| 0 | 98 | 10.94% |
| 1 | 104 | 11.61% |
| 2 | 119 | 13.28% |
| 3 | 101 | 11.27% |
| 4 | 128 | 14.29% |
| 5 | 126 | 14.06% |
| 6 | 113 | 12.61% |
| 7 | 107 | 11.94% |

The stored draw-0/draw-6 payload has aggregate SSE 1,374,412.0283. Draw 5 is
the best layer-global fixed draw at 1,374,382.3614, a 0.00216% reduction. The
per-expert eight-draw selection, retaining the stored payload whenever it is
better than every fresh encode, has aggregate SSE 1,373,858.1373, a 0.04030%
reduction. Its matrix reductions are 0.04369% for W1, 0.04453% for W3, and
0.03247% for W2.

The winner distribution is nearly uniform and the aggregate reduction is too
small to explain the layer-12 routing anomaly. A materialized KLD experiment is
not justified as an anomaly diagnosis from this result alone.

### Stored-payload reproduction

- Audit driver: `scripts/audit_qsrt_fresh_payload_identity.py`
- Result:
  `/data/kquant/research/qsrt-continuous-recovery-m0/layer-012-expert-000-draw-6-fresh-payload-identity.json`

Layer 12 expert 0 uses coupled-Hadamard draw 6 in the stored checkpoint. A
fresh draw-6 encode at repository revision
`df3ccfd61db37be63afccaf60575daf4a5fabd97` reproduced the stored W1, W3,
and W2 payloads exactly. All 4,128,768 packed trellis words and all 20,480
FP16 scale values were bit-identical. The independently decoded matrices were
also bit-identical, so the fresh-minus-stored SSE was exactly zero. Repeating
the fresh encode with independent scale-cache scopes produced the same payload
again.

This tested anchor has no observable difference from scale sharing, current
kernel behavior, or deterministic tie-breaking. The stored build did not
record its source revision, so the result establishes numerical identity for
this expert rather than a repository-wide encoder-provenance guarantee. The
eight-draw selection combines 60 retained stored payloads with 836 fresh
candidate choices. Any materialization from that selection must retain each
expert's payload source and encoder identity even though the tested same-draw
case closes exactly.

### Layer-12 router sensitivity

- Margin audit driver: `scripts/audit_kimi_teacher_router_margins.py`
- Layer-11 result:
  `/data/kquant/research/qsrt-continuous-recovery-m0/layer-011-teacher-router-margins.json`
- Layer-12 result:
  `/data/kquant/research/qsrt-continuous-recovery-m0/layer-012-teacher-router-margins.json`
- Layer-13 result:
  `/data/kquant/research/qsrt-continuous-recovery-m0/layer-013-teacher-router-margins.json`
- Error-alignment audit driver:
  `scripts/audit_qsrt_router_error_alignment.py`
- Error-alignment result:
  `/data/kquant/research/qsrt-continuous-recovery-m0/layer-012-router-error-alignment.json`

The route archive stores top-16 expert IDs but not router scores. The margin
audit therefore replayed each complete document through the official source
layer up to the routed-expert gate. It computed the difference between the
16th- and 17th-largest sigmoid-plus-correction-bias scores. Replayed top-16
sets matched the archived teacher route sets on all 65,536 tokens in each
layer.

| Layer | Mean margin | Median margin | Fraction at or below `1e-4` |
| ---: | ---: | ---: | ---: |
| 11 | 0.00156405 | 0.000938542 | 7.7499% |
| 12 | 0.000132817 | 0.0000883937 | 54.2786% |
| 13 | 0.00112601 | 0.000738822 | 9.3521% |

Layer 12 is structurally selection-sensitive. Its median score margin is
9.42% of layer 11's and 11.96% of layer 13's. Its mean margin is 8.49% and
11.80% of the corresponding neighboring-layer means. Thirteen layer-12 tokens
have exact zero threshold margins; differing top-k tie choices in an independent
17-way recomputation did not change the exact archived-route closure of the
official gate output.

The boundary-12 student-minus-teacher error places 16.7906% of its energy in
the row-space of the unchanged 896-by-7,168 layer-12 router matrix. A random
isotropic direction has expected projection fraction 12.5%. The measured
fraction is 1.3433 times that expectation. The router matrix has full numerical
row rank under the recorded QR threshold. The per-token projection fraction
has median 16.6621%, p05 14.9271%, and p95 18.8491%.

### Decision

Layer 12 has both diagnosed causes of route instability: unusually tight
selection margins and boundary error disproportionately aligned with router
directions. Its 28.4148% marginal expert-frequency total variation does not
require an anomalous expert payload distortion to explain it. Router correction
bias optimization can address systematic frequency displacement created by
the tight margins, but it cannot remove the router-aligned hidden-state error.
Evaluate the bias loop per layer, use layer-12 marginal total variation as its
primary owned metric, and retain screening-KLD non-regression as the promotion
gate.

## 2026-08-17: Router-frequency fit population and execution witness

### Fit population

The frequency-feedback fit uses the finalized 4,000,000-token corpus report at
`/data/datasets/kquant/captures/k3-all-routed-4m-v1-corpus.json`. The report
contains 4,013 whole documents, has SHA-256
`065c38b8bae6a1feb828105fae6535858aa79ddf2e05e1fd4e7713772b9526c7`,
and identifies corpus plan SHA-256
`bc67d3e28067ef27a18e66a0b10471bb72f67459095bcc0b32ee7c0e93ee09db`.
Its exclusion inventory contains every token file used by the 32-context and
768-context distribution-fidelity suites.

The 1,000,000-token corpus report at
`out/k3-denseh-broad-v6-1m-train-corpus.json` was rejected for this purpose
because it does not contain a corpus-plan hash. Frequency updates are not fit
from either screening suite.

### Forward execution witness

`scripts/capture_kimi_router_frequencies.py` executes exact Kimi layer modules
through the weight-stationary 12-GPU pipeline and retains only the eight
attention-residual handoff boundaries required by the pipeline schedule. The
4,096-token official-teacher witness is stored at
`/data/kquant/research/qsrt-continuous-recovery-m0/router-frequency-teacher-witness-4096.safetensors`.
It completed all 92 routed layers in 175.96 seconds and has SHA-256
`1c9f754e10b833d6fdcda84818bd899b6357376f894293506c04e07b243dc534`.

The witness independently reproduced layer 12's tight router threshold:
median 16th-to-17th biased-score margin `9.989738e-5`, with 50.0244% of token
margins no larger than `1e-4`. Selection counts closed at exactly 16 experts
per token in every routed layer.

### Bias-only checkpoint publication

`scripts/materialize_kimi_router_biases.py` publishes a checkpoint view by
hard-linking the frozen anchor, reflinking only the 13 non-expert shards that
own router biases, and replacing the 92 FP32 `[896]` bias tensors in those
copy-on-write files. A complete zero-change witness published and reread all
92 tensors in 0.746 seconds. A direct shard test confirmed that the patched
view changed while the anchor hardlink remained byte-identical.

### Frequency-feedback qualification contract

Router-frequency feedback uses the authenticated 4,000,000-token population
for every measurement. For expert `e` in layer `l`, the update suppresses a
frequency difference unless its magnitude exceeds 2.5 times the conservative
independent-binomial standard error of the teacher/student difference. The two
captures replay identical documents, so this threshold overestimates rather
than underestimates independent sampling noise. The remaining update is scaled
per layer so its largest expert-bias change equals that layer's median
16th-to-17th router-score margin, subject to the declared eta bounds.

The feedback loop permits at most four remeasured student passes. It stops
earlier when no layer has noise-resolved total variation above the declared
floor, when the noise-resolved variation ceases to decrease, or when successive
bias updates oscillate. Failure to settle within four passes is evidence that
the margin-scaled update is unstable; it is not grounds for adding iterations
or changing the corpus after observing the result.

The update implementation initially zeroed sub-threshold expert-frequency
differences and then centered the complete layer gradient. Centering restored a
nonzero update on every suppressed expert whenever the retained differences did
not sum to zero. The centering was removed before the first update was
constructed. A four-expert control confirmed that the two noise-resolved
experts received opposite updates while the two unresolved experts remained
exactly zero.

The exact 32-window screening suite at `/data/datasets/kld/k3` has manifest
SHA-256 `2112611fa037cb3266c115b26f3759d92f5d7b3c24892ca7f058fec05514acf0`
and token-suite SHA-256
`a6856e1d0504fd00d13c67a5515c081f349088664d7ea0894dc4d15db2c7d209`.
It is distinct from the 1,024-context final distribution-fidelity suite whose
manifest identity is
`f3a79f7f28365d406a19a82cf210c25adf18974c4b9b607ab3754e9939f941cf`.
Layer 12's marginal
expert-frequency total variation is the bias loop's owned screening metric.
Mean reference-to-candidate KLD must not regress from
`0.07834965130622809` on the same 65,504 scored positions.

### Waterfall throughput measurement

The 4,096-token execution witness is a correctness result, not a throughput
estimate. The 4,000,000-token teacher and student frequency captures measure
the weight-stationary waterfall at a representative megabatch. Capture reports
record total wall time, embedding-load time, per-layer weight-load time, and
per-layer compute-lane occupancy. Planning for the 50,000,000-token boundary
archive must use these measurements. Compute-lane occupancy includes pipeline
waits between a layer's first and last CUDA work and must not be reported as
pure kernel time.

The first 4M teacher attempt exposed a per-document synchronization in the
router-statistics collector: every layer copied both a `bincount` result and
its score-margin vector to CPU after every document. It wrote 44.68 GB in
16 minutes, corresponding to approximately 39% of the first 12-layer segment,
and was stopped before producing an artifact. The collector now accumulates
exact `int64` counts and FP32 margins on the layer's GPU and transfers them
once when that layer completes.

The optimized 4,096-token witness is stored at
`/data/kquant/research/qsrt-continuous-recovery-m0/router-frequency-teacher-witness-4096-gpu-accumulator-v1.safetensors`.
Against the preceding witness, all 92 layers' counts, biases, active mask, and
every recorded margin statistic are exactly equal. Wall time changed from
175.96 to 172.68 seconds because layer loading dominates this small witness.
The 4M capture measures the amortized effect.

The official-teacher adapter also supports grouped expert execution: gate and
up projections are evaluated by two grouped BF16 matrix multiplications, and
the down projection by a third. The grouped 4,096-token witness is stored at
`/data/kquant/research/qsrt-continuous-recovery-m0/router-frequency-teacher-witness-4096-grouped-v1.safetensors`.
It completed in 80.32 seconds. Its complete frequency payload has SHA-256
`aea26d10ec20a1531be85bcfb25e92a488e796eae60c61c55dd6e4b20a7aa8d4`;
the non-grouped GPU-accumulator witness has the same payload identity. Every
selection count, router bias, active-layer bit, and margin statistic is exactly
equal. Grouped dispatch is therefore the qualified official-teacher execution
path for the 4,000,000-token throughput measurement.

### Measured 4,000,000-token teacher execution

The grouped official-teacher capture completed all 92 routed layers and wrote
`/data/kquant/research/qsrt-continuous-recovery-m0/router-frequency-teacher-fit4m-v1.safetensors`.
It processed 4,000,000 tokens from 4,013 complete documents in 2,015.04 seconds,
or 1,985.07 tokens per second. Every routed layer recorded exactly 16 expert
selections per token. The result retained the authenticated corpus-plan and
corpus-report identities stated above and has SHA-256
`74503981ffca460c14c1ecb254ee3234de6201e273d08865b034732a1d0ed346`.

Embedding loading took 5.06 seconds. Summed layer-weight load time was 563.67
seconds. Per-layer load time had minimum, median, and maximum values of 1.16,
7.57, and 9.97 seconds. Summed compute-lane occupancy was 22,587.68 seconds;
its per-layer minimum, median, and maximum were 224.02, 245.50, and 253.37
seconds. Compute-lane occupancy includes pipeline waits and is not a sum of
kernel execution time. The measured wall throughput is 0.75% below the
2,000-token-per-second design target and is the scheduling input for subsequent
waterfall captures.

Live RAID telemetry during the second band measured approximately 357 MB/s of
reads and 248 MB/s of writes at 5% array utilization while the GPUs remained
compute-active. The grouped 4M teacher execution is therefore compute-bound at
the observed operating point rather than limited by the NVMe array.

The 4M execution preallocates eight 57,344,000,000-byte transient boundary
slabs. A 50M run would require approximately 5.73 TB for the same simultaneous
scratch layout, before permanent training and screening outputs. The 50M
capture must therefore be scheduled as deterministic, document-aligned shards
of one authenticated corpus plan, with transient slabs removed after each
shard. The completed 4M receipt determines the shard size; the 4,096-token
witness is not used for that decision.

The suffix replay input at layer 84 also requires those eight student slabs as
permanent state: hidden boundary 84 and residual-prefix boundaries 0, 12, ...,
72. The teacher normalized LM-head target is a ninth slab. It is derived from
the teacher final decoder boundary, every residual prefix required by the
residual mixer, and the frozen teacher final RMSNorm. A raw final boundary
alone cannot reproduce teacher logits. Exact archive sizes are
516,096,000,000 bytes (0.469 TiB) for 4M training tokens and
6,451,200,000,000 bytes (5.867 TiB) for 50M training tokens. A 65,536-token
screening archive is 7.875 GiB. Sharding limits transient peak space but does
not reduce the completed archive size. The M2 low-rank fits must finish before
the 4.9 TB routed-row capture is relocated or removed to provide operational
headroom for the 50M archive.

The first 50M corpus-plan construction attempt preserved the 4M source weights,
excluded the complete 4M corpus manifest, and excluded every token file from
both distribution-fidelity suites. It failed before writing a report because
the local FineWeb-Edu source supplied only 2,213,369 usable tokens after those
exclusions, below the 17,500,000 tokens required by its 35% allocation. The 50M
plan therefore requires additional prose data or an explicitly changed source
mixture; the existing recipe cannot be scaled by changing only the target token
count.

### Active 4,000,000-token student measurement

The quantized-anchor round-zero frequency capture runs as PID `760498` and
writes
`/data/kquant/research/qsrt-continuous-recovery-m0/router-frequency-student-fit4m-round0-v1.safetensors`.
It executes the direct-Viterbi uniform-K2 anchor with its W1, W3, and W2 overlay
payloads and the same authenticated 4,000,000-token document population as the
teacher capture. Its result will supply the uncorrected teacher/student
frequency difference used to construct the first noise-thresholded,
margin-scaled router-bias update.

## Active investigations

The remaining work executes in this order because a router-bias decision
changes the prefix activations consumed by every boundary archive:

1. Complete the quantized-anchor round-zero frequency capture and construct
   the first 2.5-sigma-thresholded, margin-scaled correction-bias update.
2. Remeasure the complete authenticated 4,000,000-token population after each
   accepted update. Permit at most four damped student rounds. Accept the
   converged bias view only when layer 12's marginal selection-frequency total
   variation improves and the 32-window screening KLD does not regress from
   `0.07834965130622809`; otherwise reject the bias view and retain the anchor.
3. With the prefix frozen, capture hidden boundary 84 and residual-prefix
   boundaries 0, 12, ..., 72 for the student plus the teacher's normalized
   LM-head target. Produce this nine-slab contract for both the authenticated
   4,000,000-token training population and the 65,536-token screening
   partition.
4. Execute the archive-dependent zero-step gates: anchor KL equality,
   bit-identical zero-step overlay, and screening-evaluator execution.
5. Train shared experts and normalization tensors on the 4,000,000-token
   archive. Report held-out screening KL as the primary result, with per-layer
   route agreement and per-tensor-group update norms. Materialize and run the
   complete distribution-fidelity suite only for the best held-out state.

The archive-independent trainer gradient and optimizer parity gate is already
closed. Low-rank error-factor fitting and the broader 50,000,000-token corpus
remain parallel research tracks and do not delay this sequence.

## 2026-08-17: Archive-independent suffix trainer gate

`qsrt/suffix_recovery_training.py` implements the segment-resident gradient
path independently of the boundary archive reader. Stored student boundary
states are queue-chained through eval-mode decoder stages. Each stage owns its
trainable parameters and FP32 gradient accumulator, retains or recomputes its
local autograd graph, and sends explicit state cotangents to the preceding
stage. The loss head applies a frozen teacher head to stored teacher targets
and computes token-mean forward KL in bounded token chunks. Stage-local AdamW
uses FP32 masters, moments, and accumulated gradients and supports one global
gradient-norm clipping decision.

`tests/test_suffix_recovery_training.py` compares this machinery with one
monolithic `torch.autograd.grad` graph on a three-stage miniature model. Each
stage contains frozen top-2 routed experts, trainable shared experts, and
trainable norms; the output path contains a trainable final norm and LM head.
The test uses two stored documents of different lengths, a frozen teacher,
two-token KL chunks, bounded queues, layer checkpoint recomputation, and the
same AdamW configuration in both executions. Every trainable parameter
gradient and every parameter after one optimizer step agree in FP32.

Focused verification:

```text
.venv/bin/pytest -q tests/test_suffix_recovery_training.py
2 passed in 0.92s
```

The archive-dependent zero-step KL, zero-step overlay identity, screening
evaluation, and nonzero 4M-token optimization gates remain unexecuted. They
require the layer-84 student-state and teacher-target archives.

## 2026-08-17: Selective suffix-training archive contract

`KimiBoundarySlabArchive` now declares its retained decoder boundaries in the
manifest and seals only that declared set. The forward waterfall validates the
same retained-boundary contract, persists only those boundaries, and still
retains every 12-layer handoff required for exact continuation. The permanent
student state at the layer-84 cut is therefore exactly boundaries 0, 12, ...,
72 and 84 rather than all 94 decoder boundaries.

`KimiSuffixTrainingArchive` pairs that eight-slab student archive with one BF16
teacher slab containing the normalized input to the frozen teacher LM head.
The teacher target is derived from the teacher final decoder boundary, teacher
residual prefixes 0, 12, ..., 84, the frozen residual mixer, and the frozen
final RMSNorm. A raw final decoder boundary is not a valid single-slab target
because it cannot reproduce the residual mix. The archive validates document
identity, exact token coverage, target-writer receipts, student-boundary
geometry, and complete sealing before replay.

The suffix trainer consumes the normalized teacher target directly through a
frozen LM head. The student continues to execute its residual mixer, final
RMSNorm, and LM head, so those student tensors remain eligible for training
without changing the frozen teacher distribution.

CPU-only focused verification:

```text
.venv/bin/pytest -q tests/test_kimi_suffix_training_archive.py \
  tests/test_suffix_recovery_training.py tests/test_kimi_cotangent_slabs.py
9 passed in 9.52s
```

The teacher-target reducer reads the final boundary and all eight residual
prefixes in bounded 2,048-token extents. It never stages nine full-document
slabs on a GPU, so target construction has a fixed activation-memory bound
independent of document length. The capture command exposes this bound as
`--target-replay-chunk-tokens`; the default requires approximately 252 MiB of
BF16 state per worker before model weights and writer buffers.

The archive writer, selective forward interval, target reducer, and suffix
trainer pass Python compilation and `git diff --check`. The archive-dependent
real-model gates remain blocked on a frozen router-bias decision and the two
boundary archives.

Read-only dry runs closed the exact storage and population contracts. The
4,000,000-token archive contains 4,013 complete documents and requires
516,096,000,000 permanent bytes plus the same transient teacher-state bytes.
The 32-window screening archive contains 65,536 tokens and requires
8,455,716,864 permanent bytes plus the same transient bytes. Both preflights
resolve their filesystem from the nearest existing ancestor, so a fresh nested
destination does not have to be created before validation.

## 2026-08-17: Real suffix-module integration

`qsrt/kimi_suffix_recovery_model.py` binds the archive-independent trainer to
the actual Kimi-K3 execution modules. Each of layers 84-92 owns one decoded
uniform-K2 routed-expert bank and exposes only the first optimization arm:
the three shared-expert matrices and parameters whose checkpoint names end in
`norm.weight`. The student output module reproduces the residual-boundary
mixer, final RMSNorm, and LM head. Only the final RMSNorm gain and output
residual-normalization gain are trainable there; the residual score projection
and LM head remain frozen.

The explicit allowlist contains 86 tensors and 1,189,396,736 parameters:
27 shared-expert matrices, 57 normalization tensors across layers 84-92, and
two output normalization tensors. Its BF16 parameters occupy 2.215 GiB and its
BF16-gradient plus FP32-master, accumulator, and Adam state occupies 17.723
GiB distributed across the nine layer owners and output owner. Router weights,
attention and latent projections, residual score projections, routed-expert
interfaces, routed-expert payloads, and the LM head are excluded.

`scripts/train_kimi_suffix_recovery.py` implements the real archive consumer,
nine-GPU layer loading, independent student and teacher output heads, complete-
document batching, stage-local FP32 Adam state, global gradient clipping,
screening replay, and BF16 overlay emission. The script has not executed on
real archives because router-bias qualification must freeze the prefix first.
`scripts/materialize_kimi_suffix_recovery.py` publishes such an overlay as a
hardlinked checkpoint view, rewrites only affected safetensors shards, and
removes the MXFP8 scale auxiliaries for shared-expert matrices replaced by BF16
weights. Unchanged shards remain hardlinked to the anchor.

Focused verification:

```text
.venv/bin/pytest -q tests/test_kimi_suffix_recovery_model.py \
  tests/test_suffix_recovery_training.py \
  tests/test_kimi_suffix_training_archive.py
9 passed in 0.71s
```

The materializer's isolated schema test additionally verifies exact BF16
replacement, scale-auxiliary removal, index byte accounting, preservation of
the anchor shard, and inode identity for an untouched shard.

The zero-step overlay identity is defined in the representation consumed by
the trainer. Normalization tensors already stored as BF16 must remain
bit-identical to their checkpoint values. MXFP8 shared-expert matrices must
equal the anchor loader's decoded BF16 runtime tensors bit-for-bit; their
materialized shards replace the MXFP8 tensor and scale pair with that BF16
tensor, so raw serialized bytes are intentionally different. Archive-backed
zero-step KL and materialized execution parity remain separate required gates.

Archive reads now prefetch one bounded complete-document batch while the GPUs
execute the preceding batch. The queue does not change batch order or corpus
membership and holds at most one additional batch in host memory.

## 2026-08-17: Four-million-token router-frequency feedback

The direct-Viterbi uniform-K2 anchor frequency capture completed over exactly
4,000,000 tokens from 4,013 authenticated documents and all 92 routed layers.
Wall time was 3,070.085 seconds, or 1,303 tokens per second. Its corpus-plan
SHA-256 is
`bc67d3e28067ef27a18e66a0b10471bb72f67459095bcc0b32ee7c0e93ee09db`.
The capture report retains load and execution time for every decoder layer.
Across the 93 pipeline stages, weight loading and expert reconstruction used
12,313.071 lane-seconds and layer execution used 23,271.443 lane-seconds.
Those lane totals overlap across the 12 devices and therefore do not sum to
wall time; they attribute 34.60% of measured lane service time to loading and
65.40% to execution. The observed end-to-end rate predicts approximately
10.66 hours for one 50,000,000-token waterfall pass at unchanged geometry.

The mean marginal frequency total variation from the teacher was 1.51638%.
Layer 12 was the maximum at 27.10772%; 27.10298% remained after the fixed
2.5-standard-error sampling-noise threshold, so the anomaly is not sampling
noise. The layer's median 16th-to-17th router-score margin was
`1.0144710540771484e-4`. Margin scaling therefore selected
`eta_b=3.770364008486199e-4` and limited the largest layer-12 bias update to
exactly that median margin. The other 91 layers reached the predeclared
`eta_b=0.05` cap. The cumulative round-1 bias tensor is
`/data/kquant/research/qsrt-continuous-recovery-m0/router-bias-fit4m-round1-v1.safetensors`.

The first detached round-1 launch exited before creating scratch state or a
Python traceback. No kernel OOM event was recorded and host memory and storage
were healthy. The identical command remains active through the managed command
session as PID 782349; its log is
`/data/kquant/research/qsrt-continuous-recovery-m0/router-frequency-student-fit4m-round1-v1.log`.

The round-1 capture subsequently completed over the same 4,000,000-token,
4,013-document population in 3,084.052 seconds. Its output SHA-256 is
`863005d249b7a4de48a414c7be119ca458d7d0ed347aa9e775ab0683555f780e`.
Relative to the teacher frequencies, mean marginal total variation decreased
from 1.51638% to 1.48748%. Layer 12 decreased from 27.10772% to 26.88369%; its
noise-resolved value decreased from 27.10298% to 26.87891%. The direction is
correct and the movement exceeds the sampling-noise threshold, but the
layer-12 reduction is only 0.22403 percentage points (0.83% relative), so the
loop has not converged.

The cumulative round-2 bias was generated with the unchanged rule
(`step_fraction=1`, `noise_sigma=2.5`, `eta_b` clamped to
`[1e-4, 0.05]`, and `tv_floor=2e-4`). Layer 12 used
`eta_b=3.874978470756582e-4`; its largest update remained equal to its measured
median selection margin, `1.0117888450622559e-4`. The round-2 frequency capture
uses `/data/kquant/research/qsrt-continuous-recovery-m0/router-bias-fit4m-round2-v1.safetensors`
and writes
`/data/kquant/research/qsrt-continuous-recovery-m0/router-frequency-student-fit4m-round2-v1.safetensors`.

The round-2 capture completed over the same 4,000,000-token population in
3,073 seconds. Its output SHA-256 is
`70d8fe7d413cdf4e3ca3c6114cb501df9ff7eb6ad570fcbb73b35a9f9b1c56b3`.
Mean marginal total variation decreased from 1.48748% to 1.46880%; the median
across routed layers decreased from 1.22469% to 1.21194%. Layer 12 remained the
maximum and decreased from 26.88369% to 26.66244%, a reduction of 0.22125
percentage points or 0.82% relative. The first two updates are therefore
stable and correctly signed, but their nearly constant small improvement
confirms that median-margin scaling is over-damped for the layer-level
frequency response.

The cumulative round-3 bias used the same registered update rule. Its
4,000,000-token student capture completed in 3,068.395 seconds with output
SHA-256
`50ddf871156390054ac0b37b8b974bbb41e474f18d201062436203df3eb94222`.
Mean marginal total variation decreased from 1.46880% to 1.45484%; the median
decreased from 1.21194% to 1.20325%. Layer 12 remained the maximum and
decreased from 26.66244% to 26.44399%, a reduction of 0.21845 percentage
points or 0.82% relative. The layer-12 reductions over the first three updates
were 0.22403, 0.22125, and 0.21845 percentage points. This smooth decay
confirms over-damping rather than instability.

The first round-3 process invocation used the repository virtual environment
and exited before creating scratch state because that environment does not
contain `transformers`. The capture was run without changing its inputs or
semantics using `/home/luke/projects/vllm/.venv/bin/python`, matching the
environment used by the preceding frequency captures.

The cumulative round-4 bias completed the registered four-update contract.
Its 4,000,000-token capture completed in 3,075.894 seconds with output SHA-256
`91277a8d7ffc77c27c8ae751a4702bcad1205ab9a1033670d73bc54cb9583ad9`.
Mean marginal total variation decreased from 1.45484% to 1.44223%; the median
decreased from 1.20325% to 1.18654%. All 92 routed layers improved. Layer 12
remained the maximum and decreased from 26.44399% to 26.22674%, a reduction of
0.21725 percentage points or 0.82% relative. Across four updates, mean total
variation decreased by 5.0% relative and layer-12 total variation decreased by
3.25% relative. The registered loop is stable but does not approach frequency
closure at the permitted step size.

The four-pass margin-scaled contract will complete without modification and
will retain every layer's marginal and noise-resolved TV trajectory. The
round-0-to-round-1 result is diagnosed as over-damping rather than instability:
the update was stable and correctly signed, but a token-level median-margin
step is not an empirical layer-level stability bound.

A separate layer-12 secant calibration experiment is pre-registered in
`docs/qsrt-continuous-recovery-tuning.md`. It uses the resolved positive
frequency-versus-bias slopes from rounds 0 and 1, applies them to the residual
fourth-pass layer-12 frequency error, clamps each update to 64 times the
fourth-pass median selection margin, and leaves every other layer unchanged.
It receives exactly one 4,000,000-token frequency measurement and the fixed
32-context screening KLD evaluation. No suffix boundary archive will be
captured until the four-pass contract and this probe determine the frozen
prefix. A frozen bias view that differs from the anchor also requires the full
768-context distribution-fidelity suite before it becomes the suffix pilot's
baseline.

The observed aggregate response predicts that a 32-margin intervention would
remove only about seven percentage points of the remaining 26.23% layer-12
total variation. The 64-margin clamp therefore provides a more informative
nonlinear response measurement while remaining below the approximately
120-margin linear extrapolation required for complete closure. All slope
sources, noise filtering, affected layers, capture population, and screening
gates remain unchanged.

The resulting layer-12 probe payload is
`/data/kquant/research/qsrt-continuous-recovery-m0/router-bias-layer12-secant64-fit4m-v1.safetensors`.
Round-0-to-round-1 movement supplied a noise-resolved positive slope for 29 of
896 experts. The fourth-pass median margin was `1.0076165199279785e-4`, giving
an absolute 64-margin clamp of `0.0064487457275390625`. After removing the
selection-null common shift, the maximum applied update was
`0.006447030231356621`; all layers other than layer 12 are byte-identical to
the cumulative round-4 bias view.

The 64-margin probe frequency capture completed over the same 4,000,000-token,
4,013-document population in 3,091.099 seconds. Its artifact is
`/data/kquant/research/qsrt-continuous-recovery-m0/router-frequency-student-fit4m-layer12-secant64-v1.safetensors`
and its output SHA-256 is
`0348e18c6cc56dfb58963155caf640fce1dcac034293cc3897558fa3b7ccdf0d`.
Layer-12 marginal total variation decreased from 26.22674% under the fourth
margin-scaled update to 21.05291%, a reduction of 5.17382 percentage points or
19.73% relative. Mean layer TV decreased from 1.44223% to 1.38519%; median TV
changed from 1.18654% to 1.18456%. The response is correctly signed but
strongly sublinear relative to the 64-margin scale, confirming that the
round-0-to-round-1 secant cannot be extrapolated to frequency closure.

The fixed 65,504-position screening suite evaluated the materialized bias view
at
`/data/models/Kimi-K3-QSRT-K2-DIRECT-VITERBI-ALL-LINEARS-ROUTER-L12-SECANT64-v1-model`.
Its KL divergence from the full-precision MXFP4 reference was `0.0787172394`
with 94.36370% top-1 agreement. The preregistered direct-Viterbi anchor was
`0.0783496513`, so the probe did not pass the non-regression gate. A control
captured with the same runtime revision measured `0.0787126153`; relative to
that control, the probe changed mean KL by only `+0.0000046241`. The paired
32-window bootstrap interval for this same-runtime difference was
`[-0.00057977, +0.00058367]`; 15 windows improved and 17 regressed. Top-1
agreement changed by -0.05496 percentage points.

The layer-12 router-frequency discrepancy is therefore measurable and
correctable, but correcting 19.73% of it produced no detectable distributional
quality gain. The 64-margin layer-local probe is rejected.

The probe did not start from the unchanged anchor. Its baseline was the bias
tensor recorded by the round-four student frequency capture, which contains
the four cumulative all-layer frequency-feedback updates after runtime BF16
conversion. The other 91 layer rows are bit-identical between that capture and
the probe; only layer 12 received the additional secant update. The screening
null therefore covers the small four-round all-layer correction together with
the large layer-12 correction. It does not cover large secant-calibrated
updates at the other layers.

### Registered all-layer secant closure probe

One final router-bias experiment is registered before suffix-archive capture.
It starts from the round-four captured biases and applies one update to all 92
routed layers simultaneously. For each expert, an ordinary least-squares
frequency-versus-centered-bias slope is fitted over all five measured
round-zero through round-four pairs. A slope is eligible only when it is
finite, positive, and its fitted response exceeds 2.5 times the conservative
independent-binomial frequency uncertainty. Round-four residuals below the
same 2.5-standard-error threshold receive no update. Each remaining proposal
is `-(f_round4 - f_teacher) / slope`, clamped to plus or minus 64 times that
layer's round-four median 16th-to-17th selection margin.

The budget is exactly one 4,000,000-token frequency capture and one
65,504-position screening KLD evaluation, with no iterative extension. The
capture reports marginal and noise-resolved TV changes, linear predictions,
resolved slopes, updated-expert counts, and raw and noise-resolved residual
sign flips per layer. The linear prediction is a mean marginal-TV landing of
approximately 0.7-1.0%, down from 1.442% after round four. Wider-margin layers
carry a live regression risk because their changed selections may be more
decisive than layer 12's near-tied selections.

The screening gate is a same-runtime paired comparison. The historical pinned
anchor KLD `0.07834965130622809` is retired as a gate reference. The absolute
mean-KLD detection threshold is `0.000584`, derived from the preceding paired
32-window null. A detected gain requires a candidate-minus-anchor delta below
`-0.000584` and a paired window-bootstrap interval wholly below zero. A null or
regression closes the router-bias channel and freezes the unchanged anchor. A
detected improvement permits only a separately registered bounded convergence
experiment before the prefix freeze.

The pinned and rebuilt anchors used different B12X, vLLM, and QSRT tree
identities, so the observed 0.46%-relative evaluator shift cannot be assigned
to a single runtime change. Every subsequent screening gate must pair anchor
and candidate measurements from identical serving and evaluation revisions.

The registered payload was constructed at
`/data/kquant/research/qsrt-continuous-recovery-m0/router-bias-all-layers-secant64-fit4m-v1.safetensors`
before launching its frequency capture. Across 82,432 active layer/expert
coordinates, 2,776 slopes passed the positive-response resolution test and
2,685 also had a noise-resolved round-four residual. The resulting linear
prediction is mean marginal TV `1.44223% -> 1.14632%`. This is above the broad
0.7-1.0% preregistered expectation because only 3.26% of coordinates receive
an update under the fixed resolution rules. Layer 12 has 189 updated experts
and a predicted TV landing of 8.01664%; layer 84 has 43 updated experts and a
predicted landing of 0.83914%. These are construction-time predictions, not
capture results.

The corresponding 4,000,000-token frequency replay completed in 3,045.211
seconds and wrote
`/data/kquant/research/qsrt-continuous-recovery-m0/router-frequency-student-fit4m-all-layers-secant64-v1.safetensors`.
Mean marginal total variation across the 92 routed layers decreased from
1.442234% to 1.301780%, a reduction of 0.140454 percentage points or 9.739%
relative. The median decreased from 1.186539% to 1.087841%. Ninety-one layers
improved; layer 59 increased by 0.000925 percentage points. Of the 2,685
updated layer/expert coordinates, 2,437 reduced their absolute residual, 247
increased it, and 1,138 crossed the teacher frequency. Layer 12 decreased from
26.226737% to 19.910008%, a 24.085% relative reduction. The measured response
is smaller than the linear prediction but is broad and correctly signed.
Per-layer measurements are stored in
`/data/kquant/research/qsrt-continuous-recovery-m0/router-frequency-all-layers-secant64-response-v1.json`.

The same-runtime KLD pair used the direct-Viterbi uniform-K2 anchor at
`/data/models/Kimi-K3-QSRT-K2-DIRECT-VITERBI-ALL-LINEARS-v1-model` and the
bias-only view at
`/data/models/Kimi-K3-QSRT-K2-DIRECT-VITERBI-ALL-LINEARS-ROUTER-ALL-LAYERS-SECANT64-v1-model`.
Both production-kernel audits passed with 1,104 QSRT layer loads and 12 W4A16
repeat checks. The anchor exactly reproduced mean KL `0.07871261528544193`.
The bias view measured `0.07768437211427645`, an absolute improvement of
`0.001028243171165489` or 1.3063% relative. The paired 32-window bootstrap
interval was `[0.00034931691850491494, 0.0017563141610074]` in the
anchor-minus-candidate direction; 21 windows improved and 11 regressed. Code
and prose improved with positive paired intervals. Instruction changed by
`-0.000033661021687594796`, with an interval spanning zero. Top-1 agreement
changed from 94.41866% to 94.39271%, a reduction of 0.02595 percentage points.
The aggregate improvement exceeds the registered `0.000584` runtime-noise
threshold and is statistically supported. The gate report is
`/data/kquant/research/qsrt-continuous-recovery-m0/router-bias-all-layers-secant64-kld-gate-v1.json`.

The all-layer bias channel therefore remains active for one separately
registered bounded convergence experiment. The small unresolved instruction
point regression requires repetition before the prefix can be frozen; it is
not evidence of a supported instruction regression.

The bounded convergence experiment is deferred in favor of training priority.
This is a scope decision rather than a change to the completed all-layer probe
contract. The all-layer secant view is the frozen suffix-training prefix,
pending full-suite confirmation. Its materialized model is
`/data/models/Kimi-K3-QSRT-K2-DIRECT-VITERBI-ALL-LINEARS-ROUTER-ALL-LAYERS-SECANT64-v1-model`.
The unchanged direct-Viterbi model remains the counterfactual anchor. Further
frequency matching remains a post-training option and requires fresh student
boundary slabs if it changes any layer below the suffix cut.

Teacher normalized LM-head targets are independent of router-bias selection and
student-prefix weights. Their 4,000,000-token training and 65,536-token
screening waterfalls therefore precede student-boundary capture. Student
boundaries 0, 12, ..., 84 are captured afterward from the frozen all-layer
secant view. The archive-backed zero-step gate targets the measured screening
KL `0.07768437211427645` before any nonzero optimization step.

## 2026-08-17: Resumable suffix-archive construction

`scripts/capture_kimi_suffix_training_archive.py --resume` validates the exact
document index, archive geometry, source checkpoints, student payload roots,
and population provenance before continuing an interrupted capture. Student
and transient teacher waterfalls restart from the highest sealed retained
boundary. Incomplete extent receipts are discarded before recomputation;
sealed slabs remain immutable. A completed teacher-boundary archive can be
reused to regenerate an interrupted normalized-target reduction.

Selective archives use their declared retained-boundary sequence when
validating a sealed prefix. For the layer-84 suffix cut, the legal restart
points are therefore boundaries 0, 12, ..., 84 rather than numerically
contiguous decoder boundaries. Normalized-target receipts are replayed as a
complete set after an interruption, avoiding a mixture of partial data-parallel
reductions. Storage preflight accounts only for slab allocation that remains
at resume time.

A sealed archive is authoritative even if interruption occurs after manifest
sealing but before `capture-run.json` is written. Resume now validates the
requested population and checkpoint provenance against that sealed manifest,
and reconstructs the missing completion record. It leaves any transient teacher
directory untouched because the early recovery path cannot prove that
directory's identity. The suffix trainer also reloads its zero-step safetensors
overlay and requires exact equality to every BF16 runtime tensor before
screening or optimization begins.

Focused verification:

```text
.venv/bin/pytest -q tests/test_kimi_boundary_slabs.py \
  tests/test_kimi_suffix_training_archive.py
15 passed in 1.47s
```

## 2026-08-17: Bounded low-rank expert-error factor fit

A research-only factor fit is registered for decoder layer 84 and experts
`0,56,112,168,224,280,336,392,448,504,560,616,672,728,784,840`. It tests
the factor construction and measures its memory and throughput while the
router-frequency capture remains resident. It is not an adapter qualification
result and cannot replace suffix-replay KL evaluation.

For each gate, up, and down matrix, the target correction is the canonical
decoded-GEMM error `W_source - W_K2` from the direct-Viterbi uniform-K2 anchor.
The fit stores rank-16 factors and reports the rank-2, rank-4, rank-8, and
rank-16 truncation curves. The two registered objectives are ordinary
Frobenius error and input-second-moment-weighted error. Gate and up weighting
uses every naturally routed occurrence in layer 84 of
`/data/datasets/kquant/captures/k3-all-routed-4m-v1.kqrows`. Down weighting
uses post-SiTU inputs reconstructed through the decoded K2 gate/up matrices
from
`/data/datasets/kquant/captures/k3-denseh-broad-v7-4m-train-input-v1.kqsamples`.
Route weights are squared exactly once in both weighted objectives.

Both input populations were captured from the 3.08-bpw checkpoint and are
therefore labeled weighting proxies rather than evidence from the uniform-K2
student distribution. The factors are fitted in the ordinary expert matrix
coordinates used by decoded-BF16 execution. No payload, model artifact, or
serving format is changed. The result root is
`/data/kquant/research/qsrt-continuous-recovery-m2/layer84-experts16-error-svd-v1`.

The bounded fit completed in 21.213 seconds and wrote 20 MiB of factors. The
table reports the median fraction of the fitting objective captured across the
16 experts:

| Matrix | Objective | Rank 2 | Rank 4 | Rank 8 | Rank 16 |
| --- | --- | ---: | ---: | ---: | ---: |
| Gate | Frobenius | 0.202% | 0.401% | 0.791% | 1.544% |
| Gate | Routed input weighted | 11.081% | 14.080% | 17.978% | 23.221% |
| Up | Frobenius | 0.203% | 0.402% | 0.793% | 1.547% |
| Up | Routed input weighted | 11.121% | 13.992% | 17.841% | 22.949% |
| Down | Frobenius | 0.202% | 0.402% | 0.792% | 1.546% |
| Down | Reconstructed input weighted | 43.524% | 47.012% | 52.327% | 58.929% |

Gate and up fits used 47,330 to 122,963 routed occurrences per expert, with a
median of 60,989.5. Down fits used 829 to 2,048 reconstructed post-SiTU rows,
with a median of 1,244.5. The down result is consequently a high-overfitting-risk
estimate of concentration in the proxy population. The result establishes that
the source-minus-anchor error is much more compressible in activation-weighted
directions than under uniform coefficient weighting. It does not establish
held-out KL recovery, the preferred rank, or sufficient support for down
factors. Those decisions require suffix-replay evaluation on the screening
archive.

### Document-disjoint cross-scoring with complete routed support

The stronger bounded factor fit is stored at
`/data/kquant/research/qsrt-continuous-recovery-m2/layer84-experts16-error-svd-v3`.
It partitions the 4,000,000-token routed-row capture by complete document:
documents whose integer identity is zero modulo five form the validation
population, and every other document forms the fit population. The fit uses
38,064 to 101,293 routed occurrences per expert; validation uses 9,266 to
21,670. Each expert has 1,396 to 3,002 fit documents and 299 to 710 validation
documents. Gate, up, and down therefore use the same supported population. The
down inputs are reconstructed through the decoded K2 gate and up matrices for
every occurrence rather than taken from the smaller expert sample cache.

Every requested rank is fitted independently. Plain and routed-input-weighted
factors are then cross-scored under the same Frobenius and routed objectives on
both populations. The table reports median validation weighted-error capture
across 16 experts:

| Matrix | Factor fit | Rank 2 | Rank 4 | Rank 8 | Rank 16 |
| --- | --- | ---: | ---: | ---: | ---: |
| Gate | Plain Frobenius | 0.186% | 0.399% | 0.778% | 1.539% |
| Gate | Routed input weighted | 7.814% | 9.963% | 11.687% | 14.120% |
| Up | Plain Frobenius | 0.190% | 0.390% | 0.771% | 1.554% |
| Up | Routed input weighted | 7.880% | 9.891% | 11.702% | 14.181% |
| Down | Plain Frobenius | 0.194% | 0.374% | 0.851% | 1.551% |
| Down | Routed input weighted | 36.005% | 37.752% | 38.668% | 39.790% |

The weighted factor wins the weighted validation objective for all 16 experts
at every matrix and rank. The down correction has a pronounced rank-2 knee;
gate and up continue to gain through rank 16. This is strong evidence that the
functionally exercised source-minus-anchor error is low-rank even though the
coefficient-space error is not. It remains a weighting-proxy result from a
3.08-bpw resident capture. The frozen factors require screening suffix replay
before they can establish KL recovery or justify an adapter serving format.

## 2026-08-17: Frozen all-layer router-bias prefix and archive capture

The bounded router-bias convergence experiment is deferred in favor of
training priority. The frozen suffix-training prefix is
`/data/models/Kimi-K3-QSRT-K2-DIRECT-VITERBI-ALL-LINEARS-ROUTER-ALL-LAYERS-SECANT64-v1-model`,
pending the required full-suite confirmation. Its paired screening mean KL is
`0.07768437211427645`. The unchanged direct-Viterbi checkpoint remains the
counterfactual anchor.

Teacher normalized LM-head targets are independent of the student checkpoint
and its router biases. `scripts/capture_kimi_suffix_training_archive.py` now
supports separate `teacher` and `student` capture components without changing
the archive schema. A teacher-only run seals the teacher target while leaving
the student boundary archive incomplete. A subsequent student-only resume
requires that sealed target, captures boundaries 0, 12, ..., 84 from the
declared frozen prefix, and seals the combined archive.

The authenticated training population contains 4,000,000 tokens in 4,013
documents. Its destination is
`/data/kquant/research/kimi-k3-k2-all-layer-secant64-suffix-training-fit4m-v1`.
Nine permanent BF16 slabs require 516,096,000,000 bytes; nine transient teacher
boundary slabs require the same amount, for a 1,032,192,000,000-byte peak.
Teacher-target capture began on all 12 GPUs before student-boundary capture.

The first teacher-target launch used the official checkpoint's per-expert
execution path. After approximately ten minutes, no 12-layer boundary segment
had sealed, slab writes were approximately 57 MB/s, and GPU utilization was
approximately 10--20%. A matching 1 GiB synchronous direct-I/O probe sustained
approximately 12.8 GB/s, excluding the storage array as the bottleneck. The
capture adapter had omitted the official forward pipeline's grouped-expert
dispatch option. That omission forced the slow per-expert execution path even
though the existing grouped-dispatch parity test establishes identical forward
outputs and input cotangents. The incomplete run was stopped without sealing
an archive, grouped-expert dispatch was enabled explicitly, and the capture was
resumed into the same unsealed scratch state.

At 76 seconds after restart, GPUs 1--11 sustained 86--96% compute utilization,
the process had written 21.6 GB, and all twelve devices held the expected
teacher state. The first twelve-layer segment sealed boundary 0 and boundary
12 after 294 seconds, including 114,688,000,000 bytes of durable slab output.
This establishes that grouped-expert dispatch is required for the production
waterfall rate. Final elapsed time and component throughput are recorded when
the teacher target seals.

The teacher-target reducer originally sealed both its own target and the
combined suffix archive. Separate teacher-first capture makes that ownership
invalid because the student boundary component is deliberately incomplete.
Combined-archive sealing now belongs only to the capture coordinator after it
observes both sealed components. The running forward waterfall loaded the
preceding implementation; it can still seal the teacher target without losing
data, after which a fixed-code resume finalizes the teacher-only component and
removes its transient boundary archive.

The screening population contains 65,536 tokens in the 32 complete contexts
of `/data/datasets/kld/k3`. Its destination is
`/data/kquant/research/kimi-k3-k2-all-layer-secant64-suffix-screening-kld32-v1`.
The permanent archive requires 8,455,716,864 bytes and has a
16,911,433,728-byte construction peak.

Suffix replay scores causal next-token positions. The distribution-fidelity
screening contract stores 2,048 inputs per document but scores the first 2,047
hidden positions, matching the vLLM prompt-logit capture. The trainer therefore
excludes each document's final archived position during both optimization and
screening. The screening zero-step gate covers exactly 65,504 positions, not
all 65,536 stored positions.

### Shared-expert and normalization suffix-training registration

The first nonzero suffix-training run is registered before archive-backed
screening results are available. It trains only the 86 allowlisted tensors in
decoder layers 84 through 92: the shared-expert gate, up, and down matrices;
the decoder-layer normalization weights; and the two output normalization
weights. Routed-expert payloads, attention tensors, router tensors, residual
projection, LM head, embeddings, and every layer below 84 remain frozen.

The training population is the complete 4,000,000-token archive described
above. Document order is independently shuffled for each epoch with seed
`20260817`; documents are never split between optimizer batches. The fixed
optimization contract is:

```text
optimizer                       AdamW
beta1, beta2                    0.9, 0.95
epsilon                         1e-8
weight decay                    0
peak learning rate              2e-5
schedule                        100-step linear warmup, then constant
global gradient clipping        1.0
gradient accumulation           FP32
target optimizer batch          32,768 tokens
training epochs                 12
screening cadence               zero step and every completed epoch
```

Training runs for all 12 epochs; screening measurements do not terminate or
extend the run. The selected state is the epoch checkpoint with the lowest
mean forward KL on the fixed 65,504-position screening archive. Held-out
screening KL is the primary selection metric. Per-layer top-16 routing
agreement against the teacher and L2 update norms for shared experts, decoder
normalizations, and output normalizations are secondary measurements. Only the
selected state is eligible for materialization and the 768-context analysis
suite. The 256-context qualification partition remains untouched.

The existing route archives named `teacher-legacy32-routes` and
`student-legacy32-routes` do not match this screening population. They contain
contexts 0 through 31 from
`/data/datasets/kld/kimi-k3-distribution-fidelity-1024x2048-v1`, whose suite
manifest SHA-256 is
`f3a79f7f28365d406a19a82cf210c25adf18974c4b9b607ab3754e9939f941cf`.
The archive-backed screening gate instead uses `/data/datasets/kld/k3`, whose
suite manifest SHA-256 is
`2112611fa037cb3266c115b26f3759d92f5d7b3c24892ca7f058fec05514acf0`.
Those token populations differ. The existing route archives are therefore not
valid for the pilot's route-agreement secondary; matching teacher routes must
be captured on `/data/datasets/kld/k3`. This does not affect the KL gate.

The archive-independent trainer gates pass under
`tests/test_suffix_recovery_training.py`: queue-chained distributed parameter
gradients and one FP32 AdamW update match a monolithic autograd graph, the
trainable tensor allowlist remains restricted to 86 shared-expert and
normalization tensors, and replay excludes each document's final position from
the causal next-token objective. The run manifest records the fixed AdamW
coefficients, epsilon, zero weight decay, and FP32 accumulation explicitly.
Archive-backed zero-step KL and execution parity remain pending the completed
training and screening archives.
The screening teacher waterfall will capture those matching top-k routes in
the same forward pass, avoiding a second 93-layer official-weight stream.
Suffix screening now records mean top-16 overlap, exact top-16 set agreement,
and marginal total variation for layers 84 through 92 from those routes. The
trainer rejects a route archive unless its stored population descriptor exactly
matches the screening archive, preventing equal-sized but token-distinct suites
from being compared.

### Four-million-token teacher target capture

The grouped official-checkpoint waterfall over the 4,000,000-token training
population completed in 2,033.14 seconds, or approximately 1,968 tokens per
second wall-clock. It sealed all nine transient boundaries and the
57,344,000,000-byte normalized LM-head-input target. The stale in-process
coordinator then attempted to seal the combined archive before the student
component existed and raised `student boundary archive is not sealed`. The
teacher target was already durable and complete. A resume under the corrected
coordinator reused the sealed target, wrote `teacher-target-run.json`, removed
the obsolete failure marker, and deleted the 516,096,000,000-byte transient
teacher archive without repeating model execution.

The durable training target is
`/data/kquant/research/kimi-k3-k2-all-layer-secant64-suffix-training-fit4m-v1/teacher-normalized-lm-head-input.bf16`.
Its archive manifest records 4,013 documents, 4,000,000 tokens, and twelve
non-overlapping writer extents covering the complete target.

The matched teacher waterfall over the 32-document screening population
completed in 115.19 seconds. It sealed the 939,524,096-byte normalized target
and a 93-layer top-16 route archive on the exact 65,536 stored tokens from
`/data/datasets/kld/k3`. The normalized target is stored under
`/data/kquant/research/kimi-k3-k2-all-layer-secant64-suffix-screening-kld32-v1`,
and the matching route archive is stored at
`/data/kquant/research/kimi-k3-k2-all-layer-secant64-suffix-screening-kld32-v1-teacher-routes`.
The screening waterfall spent 95.46 seconds in the layer pipeline and 14.93
seconds reducing the normalized target; its transient boundary archive was
removed after sealing.

The frozen-prefix checkpoint and official MXFP4 checkpoint both store
`language_model.lm_head.weight` as BF16 with shape 163,840 by 7,168. A direct
byte comparison of the complete 2,348,810,240-byte tensor payload was
identical. The replay loss head therefore applies the same teacher LM-head
matrix used to construct the archived normalized teacher targets; it is not a
student-specific approximation to that matrix.

### Four-million-token frozen-prefix boundary capture

The frozen all-layer router-bias checkpoint boundary waterfall completed over
the 4,013-document, 4,000,000-token training population in 2,707.34 seconds,
or approximately 1,477 tokens per second wall-clock. It sealed decoder
boundaries 0, 12, 24, 36, 48, 60, 72, and 84 as eight 57,344,000,000-byte
BF16 slabs. The combined suffix-training archive at
`/data/kquant/research/kimi-k3-k2-all-layer-secant64-suffix-training-fit4m-v1`
is complete.

The 84 per-layer measurements report 21,163.48 summed compute seconds and
11,133.62 summed load seconds. Grouped routed-expert preparation accounts for
10,779.66 seconds of the load total; only 353.96 summed seconds remain outside
that preparation. Median per-layer compute and load times were 252.99 and
134.76 seconds respectively. With twelve layers resident concurrently, these
measurements account for the observed 45.1-minute wall time and show that the
waterfall is dominated by model compute and expert preparation rather than
storage bandwidth.

The matching frozen-prefix screening waterfall completed over 32 documents
and 65,536 stored tokens in 967.29 seconds. The completed screening archive at
`/data/kquant/research/kimi-k3-k2-all-layer-secant64-suffix-screening-kld32-v1`
contains the eight student decoder-boundary slabs, the normalized teacher
LM-head-input target, and the matching teacher route population. Its replay
objective covers 65,504 causal next-token positions.

### Archive-backed zero-step qualification

The archive-backed suffix replay loaded the 86 registered BF16 tensors without
an optimizer update and reloaded all 86 values bit-for-bit from
`/data/kquant/research/kimi-k3-k2-all-layer-secant64-suffix-zero-step-kld32-v1/overlay-step-000000.safetensors`.
Decoded-BF16 replay produced mean forward KL `0.07716259803690438` over 65,504
positions. This value defines the zero-step baseline for the decoded-BF16
training operator; it is not the production packed-kernel gate.

The materialized production view is
`/data/models/Kimi-K3-QSRT-K2-DIRECT-VITERBI-ALL-LINEARS-ROUTER-ALL-LAYERS-SECANT64-SUFFIX-ZEROSTEP-v2-model`.
The materializer preserves the routed QSRT payload and every unmodified MXFP8
dense tensor. It stores the 18 replaced shared-expert linears as BF16, removes
their 27 obsolete split-projection scale tensors, and adds their exact module
paths to the serialized-MXFP8 exclusion list. This allows the production
loader to instantiate only those linears as unquantized BF16 modules while
retaining missing-scale validation for every remaining MXFP8 tensor.

The production TP12 QSRT/B12X execution produced mean forward KL
`0.07766050207252373` over the same 65,504 positions. The kernel audit passed:
all 92 QSRT layers loaded on all 12 ranks, and every rank completed the W4A16
repeat check. Relative to the frozen bias view's same-population mean KL
`0.07768437211427645`, the zero-step materialization changed KL by
`-0.00002387004175272`, or `-0.03073%`. The production zero-step gate therefore
passes. The larger difference between decoded-BF16 replay and packed execution
is retained as an operator distinction rather than being used to retarget the
registered training objective.

### Frozen-prefix distribution-fidelity baseline

The frozen all-layer secant-bias view was captured on the 768-context analysis
partition of
`/data/datasets/kld/kimi-k3-distribution-fidelity-1024x2048-v1`. The capture
contains 768 BF16 normalized LM-head-input tensors covering 1,572,096 causal
positions. Deterministic two-pass replay used the official BF16 LM head,
disabled TF32 and BF16-reduced accumulation, and verified all source file
hashes.

The result at
`/data/kld/kimi-k3-qsrt-k2-all-layer-secant64-distribution-fidelity-1024x2048-analysis-v1/analysis-kld.json`
is mean forward KL `0.06307220013704327` and top-1 agreement
`0.9357272074987787`. The context-bootstrap 95% interval is
`[0.059526951623562814, 0.06674021214696624]`; the source-cluster-bootstrap
interval is `[0.0581711275971169, 0.06846612220788545]`.

The separately captured direct-Viterbi analysis artifact reports mean KL
`0.06299320815521789` and top-1 agreement `0.9357100329750855`. The frozen
bias view differs by `+0.00007899198182538` mean KL, or `+0.12540%` relative,
and `+0.0000171745236932` absolute top-1 agreement. The two captures used
different serving-runtime revisions, so this comparison establishes the
frozen view's analysis-suite ledger baseline but does not isolate the router
bias intervention. The same-runtime 65,504-position production comparison
remains the qualified intervention result.

### Suffix-training launch qualification

The registered 12-epoch shared-expert-and-normalization run uses the vLLM
environment because the production Kimi loader requires its Transformers and
FLA dependencies. An initial invocation under the repository virtual
environment failed before loading weights or creating training state; no
experimental input or parameter changed.

The first complete launch reproduced the bit-identical zero-step overlay and
the decoded-BF16 screening mean KL `0.07716259803690438`, then exhausted GPU 7
memory in reverse autograd at suffix stage 7. The failed allocation was
4.80 GiB; PyTorch reported 9.72 GiB reserved but unallocated. The retry
therefore enabled expandable CUDA allocator segments without changing the
registered corpus, batch size, optimizer, schedule, model, or objective.

That retry exposed a separate bounded-queue deadlock. The main thread blocked
publishing a reverse item while all reverse workers blocked behind the full
completion queue; the main thread had been written to drain completions only
after every forward document finished. The trainer now uses a completion sink
that cannot backpressure the reverse pipeline and drains completed boundary
gradients during forward progress. Worker exceptions are also surfaced with
their stage, operation, and original traceback instead of the generic pipeline
abort message. The autograd parity gate now covers eight documents, enough to
exercise the former queue cycle; all five focused suffix-training tests pass.

The qualified launch destination is
`/data/kquant/research/kimi-k3-k2-all-layer-secant64-suffix-shared-experts-norms-lr2e5-12epoch-v1`.
The first six optimizer steps completed in 9.2 to 10.6 seconds each with the
registered 32,768-token target batch. The seventh batch then exhausted GPU 3
memory in reverse autograd at suffix stage 3. The failed 6.00-GiB allocation
occurred with 90.46 GiB allocated and only 3.26 GiB free, so this failure was
not allocator fragmentation. The failing optimizer batch contained 28 complete
documents, including a 4,096-token document after seven earlier documents had
entered the replay pipeline.

The registered optimizer batch remains 32,768 tokens. Execution now partitions
each optimizer batch into whole-document replay microbatches capped at 8,192
stored tokens, accumulates every unnormalized parameter gradient in the
existing FP32 optimizer buffers, and performs one normalization, clipping, and
AdamW update after the complete optimizer batch. No document is split and the
optimizer step count, learning-rate schedule, corpus order, and objective are
unchanged. The failed batch becomes five replay microbatches; its 4,096-token
document enters the second microbatch without the seven preceding autograd
tapes. A toy-model parity test confirms that split replay reproduces the
unsplit KL sum, every accumulated parameter gradient, global gradient norm,
and AdamW update. Five focused suffix-training tests pass.

The corrected production run reproduced the preceding launch's first six
optimizer-step measurements to floating-point accumulation tolerance. It then
completed the formerly fatal seventh batch in 19.31 seconds without exhausting
GPU memory. The first seven microbatched steps took 16.82 to 19.54 seconds each,
corresponding to approximately 39 minutes per 4,000,000-token epoch before the
screening evaluation.

### Shared-expert and normalization training: epoch 1

The first epoch completed 126 optimizer steps over all 4,000,000 archived
tokens. The warmup reached the registered constant learning rate of
`2e-5` at step 100. No step activated the gradient-norm clip, and the replay
pipeline completed without a CUDA allocation failure.

The decoded-BF16 held-out screening mean KL fell from the zero-step value
`0.07716259803690438` to `0.07622576831303911` at step 126. The change is
`-0.0009368297238652734` absolute and `-1.2140982129933198%` relative. The
serialized epoch overlay is
`/data/kquant/research/kimi-k3-k2-all-layer-secant64-suffix-shared-experts-norms-lr2e5-12epoch-v1/overlay-step-000126.safetensors`.
It is the minimum-screening-KL state among steps 0 and 126, so the registered
selection rule retains it unless a later epoch improves the same held-out
objective.

The suffix routing secondary moved adversely. Mean marginal total variation
over layers 84 through 92 rose from `0.0329590764724546` to
`0.04859149849249257`, while mean top-16 overlap fell from
`0.7772025372496336` to `0.772886006961407`. Layer 84 marginal total variation
improved by `0.0002738349139690399` absolute; every layer from 85 through 92
regressed, with the largest absolute increases at layers 90 (`0.024627309292554855`)
and 91 (`0.02449660375714302`). Routing agreement remains a secondary readout;
the run continues under the pre-registered 12-epoch budget and primary
held-out-KL selection rule.

### Shared-expert and normalization training: epoch 2

The second epoch completed at optimizer step 253. Its decoded-BF16 held-out
screening mean KL was `0.07663228957657099`. This remains
`0.0005303084603333896` below the zero-step value, a `0.6872610226003051%`
relative improvement, but it is `0.0004065212635318838` above the epoch-1
minimum, a `0.533312123352303%` relative regression from that state. The
registered selection rule therefore continues to retain
`overlay-step-000126.safetensors`.

The routing secondary also moved farther from the teacher. Mean marginal total
variation over layers 84 through 92 rose from `0.04859149849249257` after
epoch 1 to `0.053422309251295194`, and mean top-16 overlap fell from
`0.772886006961407` to `0.7690147421022635`. The run continues through its
registered 12-epoch budget; later overlays replace the step-126 selection only
if they achieve a lower held-out screening KL.

### Shared-expert and normalization training: epoch 3

The third epoch completed at optimizer step 379. Its decoded-BF16 held-out
screening mean KL was `0.07962287806237825`. This is
`0.0024602800254738683` above the zero-step value, a
`3.188435962585401%` relative regression, and `0.0033971097493391417` above
the epoch-1 minimum, a `4.456642188751858%` relative regression from that
state. The registered selection rule therefore continues to retain
`overlay-step-000126.safetensors`.

The routing secondary continued moving away from the teacher. Mean marginal
total variation over layers 84 through 92 rose to `0.05519997887313366`, an
increase of `0.0017776696218384652` from epoch 2, while mean top-16 overlap
fell to `0.7681795513759976`, a decrease of `0.0008351907262659219`. The run
continues through its registered 12-epoch budget without changing its learning
rate, stopping rule, or selection objective.

### Shared-expert and normalization training: epoch 4

The fourth epoch completed at optimizer step 505. Its decoded-BF16 held-out
screening mean KL was `0.0814997439552181`. This is
`0.004337145918313717` above the zero-step value, a `5.620787828112528%`
relative regression, and `0.005273975642178991` above the epoch-1 minimum, a
`6.918888138352597%` relative regression from that state. It also regressed by
`0.001876865892839849` or `2.357194236773852%` from epoch 3. The registered
selection rule therefore continues to retain `overlay-step-000126.safetensors`.

Mean marginal total variation over layers 84 through 92 rose to
`0.05919507104489538`, while mean top-16 overlap fell to
`0.7634500904100852`. Both secondary routing measurements continue moving
away from the teacher. The run continues through its registered 12-epoch
budget without changing its learning rate, stopping rule, or selection
objective.

### Shared-expert and normalization training: epoch 5

The fifth epoch completed at optimizer step 631. Its decoded-BF16 held-out
screening mean KL was `0.08250901908811938`. This is
`0.0053464210512149946` above the zero-step value, a
`6.9287727308740665%` relative regression, and `0.006283250775080268` above
the epoch-1 minimum, an `8.242948433496423%` relative regression from that
state. It also regressed by `0.0010092751329012772` or
`1.238378286753683%` from epoch 4. The registered selection rule therefore
continues to retain `overlay-step-000126.safetensors`.

Mean marginal total variation over layers 84 through 92 rose to
`0.06102881787551774`, while mean top-16 overlap fell to
`0.7619362939667806`. Both secondary routing measurements continue moving
away from the teacher. The run continues through its registered 12-epoch
budget without changing its learning rate, stopping rule, or selection
objective.

### Shared-expert and normalization training: epoch 6

The sixth epoch completed at optimizer step 757. Its decoded-BF16 held-out
screening mean KL was `0.08330476108079019`. This is
`0.006142163043885808` above the zero-step value, a `7.96002623051677%`
relative regression, and `0.007078992767751078` above the epoch-1 minimum, a
`9.286876242007192%` relative regression from that state. It also regressed by
`0.0007957419926708108` or `0.9644303149707323%` from epoch 5. The registered
selection rule therefore continues to retain `overlay-step-000126.safetensors`.

Mean marginal total variation over layers 84 through 92 rose to
`0.06454609706997871`, while mean top-16 overlap fell to
`0.758101604482169`. Both secondary routing measurements continue moving away
from the teacher. The run continues through its registered 12-epoch budget
without changing its learning rate, stopping rule, or selection objective.

## 2026-08-18: Frozen low-rank adapter screening path

The grouped decoded-BF16 expert path now accepts an optional expert-indexed
`B @ A.T` correction at each gate, up, and down projection. Gate and up
corrections are added before SiTU; the down correction is added to the expert
output. This is the execution point defined by the factor coordinate system,
so the suffix-replay probe does not reinterpret or fold the correction into a
quantized payload.

The grouped BF16 matrix product requires the adapter rank stride to be a
multiple of 16 bytes. Rank-2 and rank-4 factors are therefore zero-padded to
rank 8 in memory. The padded values are mathematically identical to the
serialized factors. CUDA comparison against explicit per-expert projection
corrections closed both forward output and input-vector-Jacobian products. A
zero-factor rank-2 path is bit-identical to the adapter-free output and input
vector-Jacobian product after its internal rank-8 padding. The grouped baseline
parity and factor-fitting tests also pass:

```text
6 passed in 1.70s
```

`scripts/evaluate_qsrt_low_rank_error_probe.py` provides the missing
qualification measurement. It loads decoder layers 84 through 92 once,
measures a same-runtime adapter-free anchor, requires a zero-factor adapter
arm to reproduce its KL sum exactly, and evaluates the plain and
routed-input-weighted rank-2, rank-4, rank-8, and rank-16 layer-84 factors on
the 65,504-position screening archive. Only the 16 experts represented in
`layer84-experts16-error-svd-v3/factors.safetensors` receive nonzero factors.
The layer-84 QSRT payload in the factor-fit anchor and the frozen router-bias
checkpoint is the same hard-linked 7,415,300,096-byte file (device/inode
`2304:34361366486`), so the factors apply to exactly the decoded expert weights
against which they were fitted.
The evaluation is queued after the suffix checkpoint's packed production KLD
and optional 768-context suite so it cannot contend with the primary training
and qualification path.

### Shared-expert and normalization training: epoch 7

The seventh epoch completed at optimizer step 883. Its decoded-BF16 held-out
screening mean KL was `0.08288577999297767`. This is
`0.005723181956073284` above the zero-step value, a
`7.4170415482071705%` relative regression, and `0.006660011679938557` above
the epoch-1 minimum, an `8.737218170878446%` relative regression from that
state. It improved by `0.00041898108781251997` or `0.5029497502623936%` from
epoch 6, but remains well outside the retained state.

Mean marginal total variation over layers 84 through 92 was
`0.06426590122282505`, while mean top-16 overlap was `0.7577478304972046`.
The small epoch-over-epoch KL recovery did not restore routing agreement. The
registered selection rule continues to retain
`overlay-step-000126.safetensors`, and the run continues through its fixed
12-epoch budget.

The token-weighted training KL fell monotonically from `0.06170908496449796`
in epoch 1 to `0.0260858136553304` in epoch 7. The opposing held-out trajectory
therefore establishes optimization with generalization loss rather than a
stalled optimizer. The minimum held-out state remains the epoch-1 overlay.

## 2026-08-18: Fifty-million-token corpus capacity audit

The revision-pinned `HuggingFaceFW/fineweb-edu` `sample-10BT` source was
expanded to 50,000 documents at
`/data/datasets/kquant/corpora/k3-broad-external-v2/fineweb-edu-sample10bt-50k.jsonl`.
It contains 239,297,052 selected characters and has SHA-256
`ee54ce5d30aeb145e92e933b0b987d92ceebf41f4a39244d7508bbbd3cd7d40f`.
The source uses immutable upstream revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` and the same quality thresholds as
the 5,000-document shard.

A 50,000,000-token dry plan preserved the 4,000,000-token source weights and
excluded the complete 4,000,000-token training manifest, all preceding
calibration manifests, and every distribution-fidelity token file. The
expanded FineWeb-Edu source removed the preceding prose-capacity failure. The
plan then stopped before writing a report because
`local-deep-calib.jsonl` supplied 695,105 usable tokens after exclusions while
its unchanged 5% allocation required 2,500,000 tokens. A valid plan therefore
requires either more data for that semantic source or an explicitly revised
source mixture; no allocation was changed implicitly.

### Shared-expert and normalization training: epoch 8

The eighth epoch completed at optimizer step 1009. Its decoded-BF16 held-out
screening mean KL was `0.08271855622633403`. This is
`0.005555958189429652` above the zero-step value, a `7.200325456605827%`
relative regression, and `0.006492787913294926` above the epoch-1 minimum, an
`8.517838595776105%` relative regression from that state. It improved by
`0.0001672237666436316`, or `0.20175205766030935%`, from epoch 7 but remains
well outside the retained state.

The token-weighted training KL for epoch 8 was `0.024544292136275105`. Mean
marginal total variation over layers 84 through 92 was
`0.06459634792473581`, while mean top-16 overlap was
`0.7574835336603702`. The registered selection rule continues to retain
`overlay-step-000126.safetensors`.

### Shared-expert and normalization training: epoch 9

The ninth epoch completed at optimizer step 1135. Its decoded-BF16 held-out
screening mean KL was `0.08417647299246428`. This is
`0.007013874955559898` above the zero-step value, a `9.089734060283172%`
relative regression, and `0.007950704679425172` above the epoch-1 minimum, a
`10.430468403773553%` relative regression from that state. It regressed by
`0.0014579167661302461`, or `1.762502674902966%`, from epoch 8.

The token-weighted training KL for epoch 9 was `0.023070438106575476`. Mean
marginal total variation over layers 84 through 92 rose to
`0.06658244650397036`, while mean top-16 overlap fell to
`0.7556649424293003`. The registered selection rule continues to retain
`overlay-step-000126.safetensors`.

### Shared-expert and normalization training: manual stop

The 12-epoch training run was stopped on request during epoch 10 at optimizer
step 1195, after batch 59 of that epoch. Epoch 10 did not reach its held-out
screening boundary at step 1261 and therefore has no screening result or
eligible overlay. The most recent completed screening remains epoch 9 at step
1135. The minimum held-out state remains the epoch-1 overlay at
`overlay-step-000126.safetensors`, with mean KL `0.07622576831303911`.

The trainer processes and the dependent materialization, packed-KLD,
distribution-fidelity, and low-rank screening services were stopped. No
`complete.json` was synthesized for the interrupted run, no checkpoint was
materialized, and all GPUs were released.

### Shared-expert and normalization training: discarded legacy-suite evaluation

The retained step-126 overlay was materialized against the frozen all-layer
Secant64 router-bias checkpoint as:

```text
/data/models/Kimi-K3-QSRT-K2-DIRECT-VITERBI-ALL-LINEARS-ROUTER-ALL-LAYERS-SECANT64-SUFFIX-SHARED-EXPERTS-NORMS-v1-model
```

Materialization replaced 86 BF16 shared-expert and normalization tensors,
removed 27 obsolete MXFP8 scale tensors, and rewrote two checkpoint shards.
The router-bias correction remains supplied by the Secant64 anchor. A TP12
production-vLLM run mistakenly used the retired 32-context, 65,504-position
reference at `/data/datasets/kld/k3` instead of the distribution-fidelity
suite used by the uniform-K2 baseline.

The materialized model achieved mean reference-to-candidate KL
`0.07654699886302353` and top-1 agreement `0.9447209330727895`. Relative to
the same-path zero-step materialization at mean KL `0.07766050207252373`, the
absolute KL change was `-0.0011135032095001968`, a `1.4338089244650298%`
relative improvement. Top-1 agreement increased by
`0.04427210552028216` percentage points. Twenty-four of 32 windows improved;
a 100,000-resample window-cluster bootstrap placed the paired mean-KL change
between `-0.0017715391851696066` and `-0.0005205432727042181` at 95%.

Relative to the directly served Secant64 anchor at mean KL
`0.07768437211427645`, the improvement was `0.0011373732512529183`, or
`1.4640953132501355%`, and top-1 agreement increased by
`0.07938446507084196` percentage points. Mean KL improved in prose, code, and
instruction windows. These measurements are not comparable with the
768-context uniform-K2 result and do not qualify the step-126 overlay. The
legacy suite and this derived run were deleted. The materialized model must be
evaluated on
`/data/datasets/kld/kimi-k3-distribution-fidelity-1024x2048-v1` before any
model-quality conclusion is made.

The deleted result directory was:

```text
/data/kld/kimi-k3-qsrt-k2-all-layer-secant64-suffix-shared-experts-norms-epoch1-v1
```

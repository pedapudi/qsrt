# QSRT improvement strategy and evidence contract

## Objective

The Quantile-Stratified Rate-shifted Trellis codec (QSRT) compresses model
weights by storing paths through a labelled reconstruction graph. EXL3 is the
comparison checkpoint produced by ExLlamaV3's trellis quantizer. The research
target is a complete QSRT checkpoint that occupies fewer bytes than the
immutable EXL3 checkpoint and has lower teacher-to-candidate forward
Kullback–Leibler divergence (KLD) on the same document-disjoint model inputs.
The QSRT and EXL3 evaluations must use identical non-expert weights.
Tables, scales, metadata, alignment, and padding count toward checkpoint size.

No coefficient-error experiment, isolated expert calculation, or encoder timing
result establishes that target. Those measurements can reject weak ideas before
an expensive model build. Only paired full-model evaluation can establish the
quality comparison.

## Available measurement and remaining evidence gap

The repository contains a repeatable one-context GLM-5.2 KLD measurement for
an eager, dense-attention, per-expert EXL3 runtime. The unchanged-model repeat
and a direct-return identity intervention reproduced all 2,047 position-level
KLD values bit for bit. They also reproduced every routed expert identifier at
all `2,048 × 78 × 8` route positions. Engine-side KLD streams four reference
rows at a time and avoids the full-vocabulary sort that exceeded GPU memory.

The published BF16 reference contains one 2,048-token context. Its correlated
token positions cannot provide document-level replication or a clustered
confidence interval. The measurement can reject a large regression and compare
mechanisms within that context. It cannot qualify a complete checkpoint. A
paired forward-KLD gate still needs multiple document-disjoint reference
contexts, identical non-expert weights, exact artifact identities, and
repeatability controls in every candidate process.

The calibration capture stack records forward activations. The repository now
implements the two-sided error-feedback recurrence, output-metric
factorization, bounded factor-artifact contract, and frozen-scale GLM candidate
encoder. The capture stack still lacks the output gradients needed to build a
real downstream-loss metric. It must add a bounded backward pass before the
GLM candidate can be selected or scored. An absolute 0.004 KLD threshold
derived from separate runs cannot estimate paired noise because shared runtime
effects cancel in a paired comparison.

Natural routing frequency defines the primary deployment estimator. A separate
support-balanced diagnostic may oversample rare experts, but its samples must
carry documented importance weights when they contribute to the primary
estimate. Both estimators must freeze their expert reservoirs, shrinkage, and
identity fallbacks before candidate errors are available.

The frozen GLM-5.2 panel includes a fresh matched encode with EXL3's computed
scalar reconstruction law, identified by its implementation name MCG.
The control uses the same BF16 source tensors, transforms, covariance, scale
search, and fit rows as QSRT. The existing EXL3 bit maps show final three-bit
(K3), four-bit (K4), and five-bit (K5) assignments. They do not replace a
matched encode or expose the candidate losses that produced those assignments.

## Measured GLM-5.2 mechanism results

The following measurements replace the earlier assumption that three
format-preserving ideas deserved equal implementation priority.

- **Reconstruction-table training is rejected on the measured residual
  domain.** Production post-feedback values were nearly Gaussian, adjacent
  correlation was negligible, and every matrix used all 4,096 table entries.
  A per-matrix fixed-path oracle with finite E4M3 centroids reduced pooled SSE
  by 0.00175%. A shared production table has less freedom, so the observed
  headroom does not justify an alternating table-training implementation.
- **One-sided input-covariance path selection is rejected.** It produced large
  held-out local expert-output SSE reductions, then increased one-context mean
  full-model KLD by 2.3453% relative to uniform K3. The inversion demonstrates
  that the local proxy is unsafe for candidate promotion.
- **Reconstructed-activation down refitting remains active.** Seven of eight
  experts accepted re-encoded K3 down targets. On the scored context, the panel
  reduced mean KLD from `0.0623782807651` for uniform K3 to
  `0.0612386895257`. It recovered 87.3960% of uniform K3's excess KLD above
  EXL3, whose mean was `0.0610743407031`. The refit still regressed 0.2691%
  relative to EXL3 and needs document-replicated confirmation.
- **Two-sided downstream-loss curvature is implemented but lacks a real GLM
  loss metric.** A synthetic 128-by-128 CUDA closure changed the trellis path
  and reduced its supplied Kronecker proxy. A complete `2,048 × 6,144` GLM
  gate-matrix closure then reproduced the ordinary K3 endpoint, held every
  stored scale fixed, and changed the path with about 1.57 GiB of peak allocated
  GPU memory. Identity output curvature reduced source-space relative SSE by
  only 0.000039%. These closures validate the implementation and dimensions.
  They do not test downstream-loss prediction because the required GLM output
  gradients have not been captured.
- **Removing BlockLDLQ feedback was neutral in the frozen-scale K3 control.**
  Setting the feedback multiplier to zero changed the numerical targets that
  reached Viterbi but changed none of the 24 gate, up, or down paths. All eight
  complete expert files were byte-identical to uniform K3 and therefore inherit
  its KLD. This result rules out a hidden feedback-removal gain for the tested
  layer-3 identity-metric panel. It does not resolve K2 or dense captured
  curvature.

All three KLD processes reproduced their resident baseline vector bit for bit.
The logical eight-expert byte ledger charges 133,791,744 bytes for EXL3 and
113,643,520 bytes for uniform QSRT K3, a 20,148,224-byte reduction. The GLM
QSRT container, headers, alignment, padding, and serving directory do not exist,
so this arithmetic is a logical-rate result rather than complete serialized
model-size evidence.

## Why the eight-weight calculation is not a quantization benchmark

The interactive explainer contains an eight-weight SwiGLU calculation because a
reader can follow every multiplication. It does not reproduce the production
encoder:

- A production scalar QSRT path covers all 256 positions in a transformed
  16-by-16 tile. The encoder uses 128 positions of cyclic context on either side
  to infer the closed boundary state.
- The eight-weight calculation gives each scalar stream only two positions. Its
  sixteen closed paths expose a small and biased subset of the reconstruction
  table.
- The calculation chooses a scale from the largest of two values. The
  production encoder searches a global scale on sampled full tiles.
- The calculation omits the model's transforms, BlockLDLQ error feedback,
  captured input covariance, native BF16 source values, real routed inputs, the
  shared expert, and downstream layers.
- Enumerating every gate-path and up-path combination under an exact output KLD
  objective is cheap with sixteen paths. The same computation is intractable at
  production dimensions and is therefore not a deployable encoder method.

The synthetic calculation can explain how a mechanism changes an expert. Its
reported gains are not evidence that the mechanism helps GLM-5.2 or Kimi-K3.
The browser page must label it as an illustration and must not promote its
expert counts or synthetic KLD reductions as results.

## What a faithful microbenchmark preserves

Here, *microbenchmark* means that few real experts or layers are processed. It
does not mean that the trellis, matrix, transform, or loss is made smaller. A
production-shaped GLM-5.2 expert measurement preserves all of the following:

- the pinned BF16 source revision and tensor hashes;
- the 6,144-dimensional model state, 2,048 intermediate coordinates, 256 routed
  experts per mixture layer, top-eight routing, one shared expert, and SiLU
  gated expert equation;
- complete gate, up, and down matrices in their native orientations;
- the same 16-by-16 transform tiles and 256-position tensor-core ordering;
- the frozen SQG rank map, reconstruction table bytes, rate, 128-position cyclic
  context, Viterbi implementation, traceback, and decoder;
- production scale search, persisted scale precision, matrix transforms,
  BlockLDLQ feedback, and the exact covariance policy under test;
- fit, selection, and reporting rows from disjoint source documents;
- applied router coefficients, natural expert co-routing, and the complete
  routed expert output; and
- the same candidate implementation that a complete checkpoint build would
  call.

Changing any item creates a mechanism illustration or an ablation. It cannot be
used as transfer evidence for the complete encoder.

The repository's production codec benchmark already streams complete pinned GLM
experts through the production CUDA encoder. The configurable driver is
[`scripts/benchmark_glm52_production_codec.py`](../scripts/benchmark_glm52_production_codec.py).
Its default panel contains eight error-blind experts spread across early,
middle, and late mixture layers. Increasing `--experts` extends the same frozen
panel to at most 48 experts. This host has neither the source checkpoint nor a
CUDA device, so only its selection and manifest contracts can run here.

The production codec driver measures source-relative weight error. A separate
intervention runtime now measures full-model forward KLD for selected layer-3
experts under the eager per-expert correctness path. The remote experiment
tree also contains a pinned routed-input capture with separate fit and
selection documents. The two-sided encoder has passed synthetic and complete
real-matrix CUDA closure. A GLM candidate still requires an output-gradient
capture adapter and the resulting expert-local factor artifact.

## Claim-specific evidence ladder

A single small experiment cannot make improvement at small and large scale
logically equivalent. Routing, rare experts, layer depth, and downstream error
propagation introduce effects that do not exist in a single expert. The
research program instead uses the smallest unchanged system that can answer
each question.

| Decision | Smallest valid measurement | What a pass permits |
|---|---|---|
| Determine whether the graph, table, scale, and traceback lower tile distortion | Complete 256-position production tiles drawn from real transformed and BlockLDLQ-feedback values | Run complete real experts |
| Determine whether an expert-level target improves the gated calculation | Complete gate, up, and down matrices for a depth- and route-support-spread real-expert panel | Run complete layers |
| Determine whether the change survives routing and co-routed error cancellation | Complete early, middle, and late mixture layers on natural routed traffic | Build a complete checkpoint candidate |
| Determine whether the checkpoint beats EXL3 | Exact-byte complete checkpoints with paired document-disjoint forward KLD and task evaluation | Support the stated research claim |

An idea is rejected when a smaller valid measurement shows a clear regression.
A pass only authorizes the next measurement. It never skips the larger-scale
gate. The reverse direction is also checked: after each complete-layer or model
run, compare its direction and ranking with the smaller panel. If the panel did
not predict the larger result, redesign the panel before using it to screen
another idea.

## Real-expert panel design

The default eight-expert run is a wiring and severe-regression screen. It is not
large enough for a quality estimate. Panel construction is fixed before any
candidate errors are observed and covers:

- early, middle, and late mixture layers;
- low, median, and high routed-row support;
- low, median, and high source and transformed weight scale;
- common and heavy-tail tile distributions;
- experts that frequently and rarely co-route with the shared expert; and
- every gate, up, and down projection.

The configurable codec driver's larger panel is depth-spread and error-blind.
The bounded official BF16 source window limits the completed intervention panel
to eight layer-3 experts. That panel was frozen before candidate errors were
available, and every expert appeared in all 32 fit documents and eight
selection documents. A broader confirmation panel must add depth and
route-support strata while preserving error-blind selection.

The named GLM-5.2 EXL3 checkpoint supplies one additional stratification axis.
Its per-layer manifests record the final bit width of each gate, up, and down
projection after router-mass-weighted MCG allocation. Include experts from the
common all-three-bit, down-upgraded, and all-four-bit patterns, plus rare
five-bit down projections. Treat those patterns as allocation outcomes rather
than an expert ranking. Preserve independently selected experts in every route
support stratum so the panel does not inherit EXL3's selection policy.

The parsed rate map has two independent arithmetic checks. Its 75 mixture
layers contain 19,200 routed experts and therefore 57,600 gate, up, and down
projections. Relative to an all-K3 assignment, 28,638 K4 projections and 81 K5
projections spend
`28,638 × (4 − 3) + 81 × (5 − 3) = 28,800` added projection-bit units. The
model card specifies 384 added units per layer, and `75 × 384` is also 28,800.
This equality checks both the manifest parse and the per-layer budget. The
listed common rate patterns cover 18,989 of 19,200 experts; the remaining 211
experts include 12 of the 81 K5 projections. The truncated pattern list must
not be mistaken for the complete rate map.

Freeze the confirmation panel as a source-controlled list of layer and expert
identifiers before encoding any candidate. Store the selection inputs and
their hashes with the list so another run can reproduce the route-support,
depth, scale, tail, and EXL3-rate strata.

Eight experts should complete quickly enough for iteration on the intended GPU
host. A 32- or 48-expert confirmation panel supplies clustered uncertainty
estimates. The full production expert encoder rejects CPU execution and cannot
meet a one-minute CPU iteration budget: one expert contains three 2,048-by-6,144
matrices, or 37,748,736 weights, and each 256-value tile requires production
scale search plus Viterbi passes over 16,384 K2 states.

A useful CPU screen remains possible if its claim is narrower. Export complete
256-value fixtures after the production transform and BlockLDLQ feedback, then
replay the exact SQG graph, T12 bytes, cyclic context, scale candidates, edge
cost, traceback, and decoder with a CPU reference. A small fixed fixture panel
can test reconstruction-table updates and additive curvature costs in less than
one minute because it processes tiles rather than full expert matrices. It
cannot test the reconstructed-activation down fit, natural routing, downstream
KLD, checkpoint bytes, or full-expert runtime. Every CPU winner must reproduce
its direction under the unchanged CUDA encoder on complete experts before it
advances.

## Running the production panel on one to four GPUs

The present production driver accepts one explicit CUDA device and processes
its selected experts serially. It already writes one atomic result per expert,
so the natural multi-GPU unit is a complete expert rather than a fraction of an
expert. A one-to-four-GPU harness should start one process per GPU, assign each
process a disjoint error-blind subset of the frozen panel, stream one expert at
a time from CPU storage, and merge the results only after their source, codec,
and experiment identities agree. Worker device details belong in run metadata;
they must not change the codec identity used to compare results.

With one GPU, encode the selected experts sequentially. With two through four
GPUs, distribute complete experts evenly and preserve the same aggregate order
when results are reduced. This exposes independent work and avoids copying one
expert's transform and BlockLDLQ state across devices. Splitting the tiles of a
single expert across GPUs is a secondary latency optimization; use it only if a
profile shows that one expert is the iteration bottleneck and its cross-device
copies do not erase the gain.

Measure one complete expert first. Estimate the panel time as the slowest
worker's assigned expert times plus source-loading and aggregation overhead.
Do not promise a one-minute eight-expert iteration before that measurement. A
useful operating pattern is one complete expert for implementation closure,
the fixed eight-expert depth-spread panel for repeated screening, and 32 or 48
experts for confirmation. Every run records projection-level encode time,
expert wall time, GPU identity, peak memory, source hashes, and resumable expert
paths.

The production codec driver reports source-relative weight error for complete
BF16 gate, up, and down matrices. Separate scripts report routed expert-output
loss and one-context full-model KLD for dense endpoint interventions. Adding
more GPUs reduces wall time; it does not broaden the claim supported by any
metric.

## Mechanisms and decisions

### Two-sided model-loss curvature — implement after gradient capture

The existing scalar SQG format, QSRT's scalar trellis reconstruction law, is
retained. Fit data supplies both an input metric
and a downstream output metric, producing a two-sided local curvature estimate
for the error that each matrix sends into the rest of the model. The Viterbi
cost remains additive, so the production dynamic program and stored branch
format remain usable. The first experiment compares raw scalar error, routed
expert output error, and the two-sided curvature objective by how well they
predict held-out forward KLD.

This is the first priority because it changes the offline objective without
changing checkpoint bytes or runtime decoding. It must use expert-local output
coordinates and documented shrinkage. A layer-global post-activation covariance
is invalid when intermediate coordinates belong to different experts.

Model-Preserving Adaptive Rounding reports about 30% lower KLD than LDLQ on its
evaluated models and quantizers, but it applies two-sided feedback rather than
scoring paths alone. A QSRT scoring-only result should be expected to recover
less. The first deliverable is therefore predictive: the curvature score must
rank held-out forward KLD better than routed squared error across experts and
layers.

The rate name K3 means three stored trellis branch bits per weight; K4 means
four. The same validated predictor becomes the damage score for exact-byte
allocation. It must score K3, mixed K3/K4, and any later residual candidate
under one common metric. If curvature does not improve held-out KLD prediction,
reject it as an allocation score. Down refitting retains its separate measured
KLD rationale and does not depend on curvature for its fitted target.

### Reconstruction values trained on production residuals — rejected

The frozen T12 table approximates a reference scalar distribution. The encoder
actually sees transformed targets after BlockLDLQ feedback, scale selection, and
model-specific conditioning. Collect those real targets and their curvature
weights, assign paths with the production Viterbi encoder, update table entries
with weighted finite eight-bit E4M3 centroids, and repeat. Training and reporting
documents remain disjoint. The table shape, finite E4M3 decoder, and branch
payload stay unchanged; table bytes and hashes are charged to the artifact.

The production residual diagnosis and fixed-path oracle fail the implementation
gate. Table updates can overfit frequent layers or route-heavy experts and can
damage tail coverage. Reopen this mechanism only if another model, rate, or
loss-weighted residual diagnosis shows material headroom before reassignment.

Run a residual diagnosis before implementing the alternating update. Measure
the post-transform, post-BlockLDLQ histogram, reconstruction-table occupancy,
conditional moments, adjacent-window correlation, and curvature-weighted tail
mass. On an independent Gaussian source, the current K3 table is 7.6% above the
Gaussian rate-distortion bound and has 3.0% lower error than MCG. Table training
has little headroom if production residuals follow the same distribution.

No decoder change or alternating training loop is justified for the measured
GLM domain. If later evidence reopens the mechanism, first confirm whether the
B12X GPU trellis decoder can index a distinct table for each layer without
adding an indirect lookup to its hot path. If the decoder supports only one
shared 4,096-byte table, train and report one global table.

### Down target fitted to reconstructed upstream activations — confirm next

Freeze the encoded gate and up matrices, execute them on real routed fit rows,
and fit the down target against the source expert output. Encode that fitted
target with the same production scalar trellis, scale search, and BlockLDLQ
path. Store only the ordinary quantized down matrix. The continuous fit is an
offline target and adds no checkpoint bytes.

A 28-expert Kimi oracle found headroom for this operation, but the result used
more freedom than the stored codec. The GLM eight-expert candidate retained the
ordinary K3 representation and improved mean KLD relative to uniform K3 on the
available context. The next test compares the quantized fitted target against a
freshly encoded source target on multiple disjoint documents and complete
layers. Reject the fit if live routing or held-out KLD regresses.

Retain the ordinary source-target encode as a per-expert fallback. The
one-context result recovered most of the uniform-K3 penalty in the mean while
1,043 of 2,047 token positions regressed. That heavy-tailed pattern makes
document-replicated evaluation essential before broader encoding.

## Additional-byte candidates for GLM-5.2

GLM-5.2 uses BF16 source weights, so Kimi-K3's exact MXFP4 endpoint, X4T, does
not transfer. The first GLM-5.2 high-quality experiment uses per-matrix scalar
rates. The GLM codec driver encodes gate, up, and down separately. It
does not apply the interleaved gate/up Hadamard transform. The per-matrix panel
can therefore construct seven gate, up, and down rate tuples:

- all three matrices at K3: `(K3, K3, K3)`;
- K3 gate and up with K4 down: `(K3, K3, K4)`;
- K4 gate with K3 up and down: `(K4, K3, K3)`;
- K4 gate and down with K3 up: `(K4, K3, K4)`;
- K4 gate and up with K3 down: `(K4, K4, K3)`;
- all three matrices at K4: `(K4, K4, K4)`; and
- K4 gate and up with the rare K5 down diagnostic: `(K4, K4, K5)`.

The implemented GLM selection policy shares one gate/up mode, so this experiment
must explicitly add independent per-matrix candidate selection. Score each
tuple through the complete expert because gate, up, and down errors interact.

Per-matrix rates conflict with the coupled gate/up transform used by Kimi-K3.
That transform interleaves gate and up rows before applying the Hadamard basis,
so each coded row contains a mixture of both projections. A coupled candidate
can express only `(K3, K3, K3)`, `(K3, K3, K4)`, `(K4, K4, K3)`, and
`(K4, K4, K4)`. Unequal gate/up tuples require encoding that expert without
the coupled transform and charging the measured 3.052% pooled conditioning
gain that the uncoupled candidate forfeits. A two-dimensional gate/up
symbol basis imposes the same joint-rate restriction. On the gate/up axis,
independent rates, the interleaved transform, and joint pair symbols cannot all
be retained at once.

K5 is unsupported by the production SQG quantizer. The table generator supports
K5, but the rate-specific production quantizer accepts only K2, K3, and K4. The
high-rate evidence also found that the finite E4M3 reconstruction alphabet
became the limiting error at K5: the E4M3 candidate lost to MCG, while a
research-only FP16 reconstruction endpoint won on all seven measured experts.
The `(K4, K4, K5)` diagnostic must therefore use that FP16 high-rate research
path and charge its table and scale representation. Running K5 with the
production E4M3 endpoint would repeat an established negative control.

Each selected rate needs a few metadata bits for every matrix. Matrices with
different rates also occupy separate dense payload stacks and need an
exact-byte directory. The dense stacks, alignment, and
padding are expected to dominate the metadata cost and must be measured from
serialized files.

A sparse residual plane remains an offline oracle until its exact-byte and
runtime evidence beats mixed K3/K4. One fixed correction in every 256-position
tile needs at least eight position bits and eight value bits, or 0.0625 payload
bpw. A per-tile shared correction exponent can supply range more cheaply than
a scale for every correction; four or five exponent bits are amortized over
the corrections in that tile. A deployable format must still charge the
exponent, tile presence, format maps, offsets, alignment, and padding. Accepted
corrections must enter BlockLDLQ feedback before the remaining path is encoded.

The residual oracle has a numerical kill threshold. A standard-normal
order-statistic calculation puts the expected largest squared residual among
256 values near 9.4. The largest residual carries about 3.7% of total squared
error, and the two largest carry about 6.6%. One extra Gaussian
rate-distortion bit reduces error by about 75%. A one-correction payload must
therefore recover more than `0.0625 × 75% = 4.69%` of tile error; two
corrections at 0.125 bpw must recover more than `0.125 × 75% = 9.38%`.

Build a residual format only if held-out curvature-weighted measurements exceed
6% for the largest residual or 10% for the two largest residuals in a
256-position tile. The complete oracle must also beat K4 per exact added byte.
The concentration thresholds include a screening margin for nonideal
correction values and metadata; they are not claims about the final serialized
rate.

Measure direct sparse corrections against fixed two-dimensional discrete
cosine and Walsh-Hadamard residual modes. Frequency-domain truncation is useful
only if a few modes contain held-out curvature-weighted error. The transformed
weight layout has no image-like locality, and the existing Hadamard transform
may flatten its spectrum. Do not implement a frequency-domain decoder unless a
fixed-mode oracle beats direct sparse corrections and K4 at the same exact
bytes. The Walsh arm is nearly free because the encoder already implements that
basis. It also tests whether a second Walsh representation merely renames
sparsity that was already present before the production transform. Pre-transform
coordinates may be inspected as a diagnostic, but a stored correction must
live in the coding domain; otherwise the decoder must add the inverse-transform
work that fixed modes were meant to avoid.

The EXL3 allocation supplies a useful sampling stratum. It does not supply a
directional answer. The
[GLM-5.2 EXL3 checkpoint model card](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78)
states that its encoder calibrates the down projection on reconstructed gate/up
outputs. A down-heavy EXL3 map therefore cannot be
explained as compensation for upstream reconstruction error that the encoder
did not observe. Pre-register the neutral mechanism test instead: if
curvature-scored QSRT moves bits upstream under matched candidate rates, test
whether the downstream-loss objective explains the shift; if down-heavy
allocation persists, treat that result as evidence of intrinsic down
sensitivity. Neither outcome is implied by conditional down calibration alone.

## Ideas not promoted by the synthetic calculation

- Per-coordinate reciprocal up/down balancing is not promoted. The synthetic
  two-position trellis exaggerated its benefit, while a real Kimi two-bit SQG
  study increased pooled error by 2.250%.
- Exhaustive exact-KLD selection across every gate and up path is not promoted.
  Its candidate product grows beyond production feasibility and does not match
  the additive Viterbi computation.
- The two-step closed trellis is not an experimental proxy. It is retained only
  as a drawing that exposes sixteen legal paths.
- A gate/up pair trellis remains a later representation experiment. It needs a
  production-length CPU reference, CUDA encoder, decoder, coupled feedback,
  exact-byte accounting, and evidence that it recovers more than it loses by
  giving up the qualified scalar transform basis.
- Residual trellis streams and entropy-coded paths remain deferred because they
  change the storage and random-access contracts.

## Promotion criteria

Before a mechanism reaches a complete checkpoint build, require all of the
following:

- bit-exact agreement between CPU reference and CUDA decode;
- the production 256-position path, 128-position cyclic context, scale search,
  transforms, and BlockLDLQ path;
- no fit/report document overlap and no candidate-aware panel selection;
- a source-controlled confirmation-panel list frozen before candidate errors
  are available;
- lower pooled loss with a positive lower clustered confidence bound on the
  pre-registered confirmation panel;
- no unacceptable regression in rare-support experts, heavy-tail tiles, live
  routing, or the pre-registered tail metric;
- agreement in direction between the expert panel and complete-layer runs; and
- an implementation whose computational form and memory use remain feasible
  for every routed expert.

The first promotion gate applies to the measurement itself. Model-loss
curvature must predict held-out forward KLD better than routed squared error on
the frozen confirmation panel. Failure rejects curvature as a damage score and
blocks curvature-weighted residual allocation. Reconstruction-table training
has already failed its independent headroom gate. Down refitting keeps its own
paired-KLD confirmation path because its fitted target does not require the
curvature score.

The sparse-residual gate is also mechanical: reject the format when the
held-out curvature-weighted top-one and top-two shares fail to exceed 6% and
10%, respectively, or when its oracle fails to beat K4 per exact added byte.

The final decision uses complete serialized bytes and paired full-model forward
KLD against the immutable EXL3 checkpoint. Smaller measurements remain
diagnostics even when every diagnostic agrees.

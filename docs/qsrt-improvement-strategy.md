# GLM-5.2 QSRT improvement testing strategy

**Status:** Active execution specification, 2026-08-18.

## Decision that the testing program must support

The Quantile-Stratified Rate-shifted Trellis codec (QSRT) compresses model
weights by storing paths through a labelled reconstruction graph. EXL3 is the
comparison format produced by ExLlamaV3's trellis quantizer. The program seeks
a complete QSRT checkpoint that satisfies two conditions:

1. Its serialized checkpoint occupies fewer bytes than the immutable EXL3
   checkpoint.
2. Its output probabilities have lower forward Kullback-Leibler divergence
   from the same high-precision teacher on document-disjoint inputs.

Forward Kullback-Leibler divergence (forward KLD) measures how much probability
the candidate removes from tokens that the teacher considers likely. Lower
forward KLD means that the candidate preserves the teacher's probability
distribution more closely.

The comparison must keep all non-expert weights identical. Checkpoint size
includes reconstruction tables, scales, format directories, headers, alignment,
padding, and every other stored byte. A panel can compare the bytes of the
expert tensors that it replaces, but only a complete serialized artifact can
establish the model-size claim.

Weight error, isolated expert error, and encoder timing can screen candidate
mechanisms. They cannot establish checkpoint quality. The primary quality
decision uses paired full-model forward KLD on the same documents.

K2, K3, and K4 name scalar trellis rates with two, three, and four stored branch
bits per weight. QSRT uses the Stratified Quantile Graph (SQG) and the same
scalar reconstruction law at all three rates. Rate changes the trellis state
and branch split while the reconstruction law remains fixed.

GLM-5.2 routes each token to eight of 256 learned experts in a mixture layer.
Each expert contains gate, up, and down weight matrices. For input `x`, the
expert computes `down(SiLU(gate(x)) * up(x))`, where SiLU is a coordinatewise
nonlinearity and `*` is coordinatewise multiplication. A complete-expert error
measurement executes all three reconstructed matrices and compares that output
with the source expert output.

## Evidence already available for GLM-5.2

The controlled GLM-5.2 measurement runs one 2,048-token context through an
eager, dense-attention, per-expert EXL3 runtime. An unchanged-model repeat and
an engine intervention that returned the resident expert output without
modification reproduced all 2,047 scored position KLD values and every routed
expert identifier bit for bit. The measurement can reject a large regression
within that context. One context cannot estimate document-level uncertainty or
qualify a checkpoint.

The position-level results expose different mean and tail behavior. The 99th
percentile is the value exceeded by the worst one percent of scored positions.
Conditional value at risk at one percent (CVaR1%) is the average KLD within that
worst one percent.

| Layer-3 eight-expert intervention | Mean KLD | 99th percentile | CVaR1% | Maximum |
|---|---:|---:|---:|---:|
| Resident EXL3 | 0.06107434 | 1.09967 | 1.95576 | 5.57961 |
| Uniform QSRT K3 | 0.06237828 | 1.18850 | 2.04973 | 4.91117 |
| Input-covariance-selected K3 | 0.06384127 | 1.23571 | 2.09884 | 4.79991 |
| K3 with reconstructed-activation down refit | 0.06123869 | 1.05916 | 2.09531 | 5.91416 |
| Down refit with one BF16 rank-four correction on expert 103 | 0.05825746 | 0.96855 | 1.81094 | 4.35530 |
| Down source target encoded with reconstructed-input covariance | 0.06585198 | 1.31937 | 2.20897 | 6.36043 |
| Locally selected identity-metric down refit | 0.06413429 | 1.12366 | 2.18651 | 5.57959 |
| Reconstructed-input covariance with locally selected down refits | 0.06381950 | 1.09715 | 2.02186 | 4.58844 |
| Ten-promotion selection-data K3/K4 control with one down target per expert | 0.06362581 | 1.06116 | 2.02477 | 4.24433 |
| Fixed twelve-promotion K3/K4 control with one down target per expert | 0.06396692 | 1.02369 | 2.15752 | 5.91416 |

These values establish four mechanism decisions.

- **Reconstruction-table training remains rejected for the measured GLM
  residuals.** Values presented to Viterbi after blockwise quantization-error
  feedback (BlockLDLQ) were nearly Gaussian, adjacent correlation was
  negligible, and every matrix used all 4,096 table entries.
  A per-matrix fixed-path oracle with finite eight-bit floating-point centroids,
  using four exponent and three mantissa bits (E4M3), reduced pooled squared
  error by 0.00175%.
- **One-sided input-covariance selection remains rejected.** It reduced
  complete-expert output error substantially, then increased full-model mean
  KLD by 2.3453% relative to uniform K3. A locally accurate reconstruction can
  perturb a downstream probability direction that matters more to the model.
- **The earlier reconstructed-activation down refit remains a candidate, but
  local refit selection is rejected.** The earlier frozen construction
  recovered 87.3960% of uniform K3's excess mean KLD above EXL3. A later rule
  hard-encoded every ridge candidate and required both local mean and row-tail
  improvement. Five expert tensors matched the earlier result, but three new
  local choices raised model mean KLD to 0.06413429. Complete-expert mean and
  local row CVaR therefore cannot authorize ridge or fallback decisions.
- **Reconstructed-input covariance for the down encoder is rejected on this
  panel.** It reduced local complete-expert error by 48.7027% for the source
  target and raised model mean KLD by 5.5688% relative to uniform K3. Refitting
  three experts recovered 42.5415% of that policy's excess KLD above EXL3 but
  did not make the policy competitive.
- **Removing blockwise quantization-error feedback (BlockLDLQ) remains a scoped
  neutral result.** With frozen K3 scales and the identity input metric,
  disabling feedback changed the values presented to the dynamic-programming
  path search (Viterbi search) but changed none of the 24 trellis paths. It
  also changed none of the reconstructed tensor bytes. This result covers
  eight layer-3 experts at K3. It does not establish the result at K2 or under
  a captured dense metric.
- **An activation-weighted low-rank down correction is the first mechanism to
  cross the numerical KLD target.** A BF16 rank-four correction on layer-3
  expert 103 reduced mean KLD to 0.05825746. It also improved p99, CVaR1%, and
  maximum KLD relative to EXL3 on the available context. The rank and expert
  were chosen after inspecting other KLD arms on that same context. The result
  is therefore a mechanism screen and supplies no document-replicated
  evidence. The executed endpoint also materialized the factor product as a dense matrix;
  a factor-aware container and serving branch remain unimplemented.

The logical eight-expert ledger charges 133,791,744 bytes for EXL3 and
113,643,520 bytes for uniform QSRT K3. This 20,148,224-byte margin is large
enough to fund twelve complete K3-to-K4 matrix promotions while retaining a
1,273,856-byte logical margin. The GLM QSRT container does not yet exist, so
these figures support panel allocation only.

The measured mixed-rate intervention does not test a coherent combination. It
promoted twelve source-target matrices over a down-refitted K3 base. Promoted
down matrices replaced their refitted K3 targets with source-target K4 tensors.
Mean KLD rose to 0.06596340. This result rejects that construction. It does not
reject K4 encoding of a refitted target.

A later pool preserved one down-refit target per expert and encoded that target
at both K3 and K4. Complete-expert error selected ten promotions, producing
mean KLD 0.06362581. The matched fixed twelve-promotion control produced mean
KLD 0.06396692. Both regressed relative to EXL3, uniform K3, and K3 down
refitting. Both also worsened CVaR1% despite lowering p99. Reject these
one-target constructions.

The later pool still reused a target fitted from K3/K3 upstream activations
when gate or up changed rate. It therefore does not test the coherent
rate-conditioned construction specified below. That experiment remains the
first admissible mixed-rate test.

The rank-four expert-103 intervention adds 65,536 logical BF16 factor bytes to
the 113,643,520-byte uniform-K3/down-refit panel. Its resulting logical total
is 113,709,056 bytes, which is 20,082,688 bytes below the panel's EXL3 tensors.
This ledger excludes factor headers, alignment, directories, scales, and the
serving implementation. It supports a panel-size screen and cannot establish a
serialized checkpoint-size claim.

## Governing acceptance rule

A QSRT candidate advances from a document-replicated comparison only when all
of the following conditions hold:

- its paired mean forward KLD is lower than EXL3;
- the 95% paired-bootstrap upper bound for candidate-minus-EXL3 mean KLD is
  below zero;
- its document-level CVaR1% is non-inferior within a margin frozen before
  candidate measurement;
- its charged bytes are lower than EXL3 for the comparison scope;
- its candidate construction was frozen before confirmation data were opened;
  and
- the result repeats on documents that did not contribute to target fitting,
  candidate selection, rate allocation, or threshold selection.

The complete-checkpoint claim also requires lower complete serialized bytes,
task-quality evaluation, and production-serving validation. A layer or expert
panel remains a mechanism result even when it passes every panel gate.

Mean forward KLD is the primary quality measure. CVaR1% is a safety constraint
that protects against a candidate buying an average gain by damaging a small
set of positions. The 99th percentile and maximum are descriptive because they
are too unstable to serve as independent pass/fail tests on a small document
set.

## Data separation and uncertainty

The GLM experiments use three document roles.

1. **Activation-fit documents** build input covariances, continuous down
   targets, and closed-form low-rank factors.
2. **Candidate-selection documents** choose ridge strength, expert fallback,
   factor rank and dtype, alternation, rate tuples, and complete panel
   configurations. The rapid KLD selection set must contain eight 2,048-token
   contexts.
3. **Confirmation documents** report the frozen configuration. The first
   confirmation set must contain at least 32 independent 2,048-token contexts
   across the intended deployment domains.

The roles must be disjoint at document level. Splitting adjacent segments from
one document across roles violates the contract. If a reporting result
influences a later candidate, its documents become selection data for that
later decision. A further claim then needs another sealed confirmation set.

Enforce the confirmation boundary through storage and process controls. Create
more than 32 eligible contexts, audit them for overlap, choose the first 32 in
a pre-recorded order, and write their identifiers and content hashes to a
manifest. Store their inputs and reference logits under a separate path. A KLD
process must refuse to read that path until a freeze record names the complete
candidate, byte ledger, runtime, and selection evidence. Mount the sealed path
read-only during confirmation and retain an access log. Replacement contexts
must come from the pre-recorded spare order.

The overlap audit compares source document identifiers, hashes of normalized
documents, and token spans. An exact context hash alone cannot detect two
different windows taken from the same source document. The reference-logit
manifest also records the immutable model and tokenizer revisions, chat
template, token identifiers, attention and mask settings, logits dtype and
vocabulary width, runtime and container identities, and every scored position.

The eight-context set provides fast configuration selection. The 32-context
set provides the first document-replicated acceptance result. A power analysis
after the first eight contexts may increase the confirmation count, but it may
not reduce the frozen minimum.

The rapid screen may use a six-of-eight same-direction rule. Under an
independent symmetric null, the probability of at least six positive results
is 14.453125%. It is a configuration screen and supplies no confirmation
evidence on its own.

Compute the mean difference for each document, then weight documents equally.
Use a paired document bootstrap to estimate sampling uncertainty. Repeated
unchanged runs estimate numerical noise. Their near-zero variation does not
estimate how results vary across documents.

The practical CVaR non-inferiority margin must be recorded before confirmation.
Its record must distinguish three quantities:

- numerical variation from unchanged repeat runs;
- sampling uncertainty from the paired document bootstrap; and
- the largest tail degradation that the release policy considers acceptable.

## Standard candidate report

Every full-model candidate report contains the following measurements.

- paired mean forward KLD for each document;
- the paired bootstrap interval for the mean difference;
- candidate-specific 99th percentile and CVaR1% for each document;
- a CVaR1% non-inferiority verdict;
- the positions entering and leaving the candidate's own worst one percent;
- overlap between the candidate's and comparator's worst-one-percent sets;
- maximum KLD as a descriptive value;
- exact charged bytes for the comparison scope;
- complete-expert output error on fit and selection rows;
- changes in routed expert identifiers and the associated top-k routing
  margins, defined as the score gap between the last selected expert and the
  first rejected expert; and
- every per-expert candidate, acceptance, and fallback decision.

Tail metrics must be recomputed for each candidate. Evaluating every candidate
on positions selected from EXL3's tail confuses movement of the tail with an
improvement in the candidate's own worst positions.

Route changes are diagnostic until repeated evidence ties them to KLD damage
or serving-capacity failures. A route change can improve the model. The testing
program must not impose a zero-change rule without that evidence.

## Test activation-weighted low-rank corrections without training

A low-rank correction adds a small continuous operator to a frozen quantized
matrix. For a source matrix `W`, its QSRT reconstruction `Q`, input rows `X`,
and rank-`r` factors `B` and `A`, fit

```text
minimize ||(W - Q - B A^T) X^T||_F^2.
```

Let `H` be the regularized covariance of `X`. Multiply `W - Q` by the matrix
square root of `H` and keep the leading `r` singular components. Map the right
singular vectors back through the inverse matrix square root of `H` to obtain
the factors. The fit report must record the covariance regularization,
singular values, and held-out recovery curve so a rank choice is reproducible.

The product `B A^T` remains present during inference. It therefore consumes
checkpoint bytes and adds an execution branch. This differs from the
reconstructed-activation down refit: the down refit discards its continuous
target after re-encoding and stores only the resulting trellis matrix.

The construction follows the activation-aware low-rank residual objective in
[CALDERA](https://proceedings.neurips.cc/paper_files/paper/2024/file/a20e8451ffb07ad25282c21945ad4f19-Paper-Conference.pdf).
CALDERA alternates a low-precision backbone with quantized low-rank factors on
dense Llama models. Its result motivates this experiment but does not establish
performance for a routed SQG trellis, GLM-5.2 experts, rank-two factors, or the
KLD acceptance rule used here.

### Measured one-context result

The completed bolt-on screen fitted BF16 rank-two and rank-four corrections to
the down matrices of the eight layer-3 experts. It used each candidate's own
quantized gate and up outputs as down inputs. Rank two reduced pooled routed
complete-expert error by 67.6456% on candidate-selection rows. Rank four
reduced it by 70.7994%.

Those local gains did not predict model quality. Applying rank-two corrections
to all eight experts raised mean KLD to 0.06523264, which is 6.8086% worse than
EXL3. Individual rank-two KLD attribution identified experts 89, 103, and 208
as helpful in isolation, yet their combined correction regressed to
0.06347226. The interactions are not additive.

A later rank-four screen was restricted to those three expert identities.
Expert 103 alone reached mean KLD 0.05825746, 4.6122% below EXL3 and below the
numerical target of 0.059. Its p99 fell from 1.09967 to 0.96855, CVaR1% fell
from 1.95576 to 1.81094, and maximum KLD fell from 5.57961 to 4.35530. Layer-3
routes remained unchanged; downstream routes first changed at layer 4.

The same reporting context influenced the expert and rank restriction. Freeze
expert 103, rank four, BF16 storage, ridge choice, factor hashes, and the
bolt-on construction before opening another context. If no choice changes,
the frozen candidate can proceed directly to the 32-document confirmation
set. If any choice changes after an eight-context screen, the changed candidate
requires a separate sealed confirmation set.

The KLD runtime added the rounded factor product to a dense endpoint. It did
not load serialized factors through a QSRT container. A checkpoint claim
therefore requires deterministic factor serialization, exact byte accounting,
and a factor-aware inference path that reproduces the screened endpoint.

### Use activations from the artifact being corrected

Fit factors against naturally routed rows from the exact checkpoint that will
execute them. Gate and up factors use the expert inputs seen by that artifact.
Down factors use hidden rows reconstructed through that artifact's gate and up
matrices. For the layer-3 panel, use the same uniform-K3 gate and up
reconstructions that will feed the corrected down matrix. Refitting factors
from resident EXL3 hidden rows would repeat the activation-mismatch error that
the down-refit work is designed to avoid.

Capture rows at the input boundary of each routed expert, before gate and up
projections. This boundary is unchanged by a layer-local expert replacement.
For every candidate, derive its down-matrix inputs offline as
`SiLU(candidate_gate(x)) * candidate_up(x)`. Do not use the down-matrix inputs
produced by the resident EXL3 expert to fit a QSRT candidate.

Partition complete documents before fitting. Factor fitting uses the
activation-fit documents. Rank, regularization, factor dtype, expert fallback,
and any alternation decision use candidate-selection documents. Confirmation
documents remain sealed until the complete panel configuration is frozen.

The layer-3 screen has measured down-only ranks two and four. Replicate the
frozen rank-four construction at error-blind early, middle, and late mixture
layers before adopting a model-wide policy. Freeze support-stratified expert
lists before candidate errors are available. Low-rank concentration measured
on another model or another layer supplies a prior for GLM; it does not supply
GLM evidence.

Do not add gate or up factors until the frozen down-only candidate repeats on
document-disjoint contexts. Gate and up factors must pass a complete-expert
functional gate because their errors interact through the coordinatewise
activation and multiplication.

The bounded low-rank candidates correct uniform K3 and remain separate from the
mixed K3/K4 candidates. If a later candidate combines an upstream K4 matrix
with a down correction, rebuild that correction from the K4-specific hidden
rows. Never splice a factor fitted for one upstream reconstruction into
another.

### Compare a bolt-on correction with one residual re-encoding

The bolt-on candidate retains the incumbent trellis matrix `Q0` and fits
`B0 A0^T` to `W - Q0`. The alternating candidate performs one additional
decomposition round.

```text
Q0 = QSRT(W)
B0 A0^T = activation_weighted_rank_r(W - Q0)
Q1 = QSRT(W - B0 A0^T)
B1 A1^T = activation_weighted_rank_r(W - Q1)
```

Retain the pre-registered null prediction that the alternating candidate will not
improve materially over the bolt-on candidate. Direct Viterbi minimizes
transformed coefficient error, while the factor fit minimizes error after
projection through routed activations. The completed GLM screen measures only
the bolt-on construction; it supplies no evidence that residual re-encoding
can exploit the activation-space structure removed by the factor.

The prediction does not remove the experiment. Hard trellis decisions can
change after a small structured target shift. Record five observables:

- the fraction of trellis paths that change;
- the count of reconstructed matrices that remain byte-identical;
- every reconstruction-scale change;
- the incremental held-out activation-weighted error change beyond the
  bolt-on candidate; and
- the paired end-to-end KLD change beyond the bolt-on candidate.

Run a frozen-scale residual re-encode before repeating global scale search.
This control holds the incumbent scales fixed and isolates path changes caused
by subtracting the low-rank correction. If its trellis payload is identical,
refit the factors once and verify complete-candidate closure. Trellis identity
alone does not establish model identity: scales, factor metadata, serialized
factors or their reconstructed product, and complete expert outputs must also
match.

If the frozen-scale trellis remains unchanged while the full alternation
changes, add a fixed-path scale-only control. That control separates scale
drift from path changes induced by the new scales. Do not add the control when
the frozen-scale arm already changes paths.

QSRT's direct Viterbi step and the activation-weighted factor step optimize
different objectives. No shared objective is guaranteed to improve after
every round. Retain the incumbent at each round. Accept an alternating local
candidate only when it improves document-disjoint activation-weighted
complete-expert error. Accept it for model construction only when paired KLD
also improves under the standard tail and byte gates.

Any gate or up change invalidates the downstream fit. Reconstruct the changed
upstream activations, refit the down target or down adapter, and re-encode the
complete expert before comparison.

### Serialize factors deterministically and test their precision

The factorization is unchanged when one factor column is multiplied by a
nonzero scalar and the corresponding column of the other factor is divided by
that scalar. Remove this scale ambiguity before hashing or quantizing factors.
For a singular-value decomposition initialization, order columns by descending
singular value and split each singular value equally as its square root across
both factors. Make the maximum-magnitude entry of each left-factor column
positive, choosing the lowest index on a tie.

After any regression update, balance each factor pair so its two column norms
are equal while preserving their product. Reject a non-finite or zero-norm
column. The serialized schema records matrix role, layer, expert, logical rank,
stored rank, factor dtype, scale dtype and shape, logical shapes, padding,
byte offsets, and content hashes.

Test three stored factor precisions as separate candidates:

1. BF16 factors provide the numerical reference.
2. Eight-bit factors use explicit charged scales and a frozen quantization
   rule.
3. Four-bit factors run only after the eight-bit candidate preserves enough
   functional recovery to justify another precision reduction.

CALDERA's BF16 and four-bit-factor results do not establish that a rank-two
QSRT factor tolerates eight-bit storage. Measure product error,
activation-weighted recovery, complete-expert error, and KLD after factor
quantization. The serving path may pad rank two or four to rank eight in
memory; checkpoint accounting charges the stored representation, while
latency reporting includes the executed padded rank.

Each GLM-5.2 routed expert uses a 6,144 by 2,048 down matrix. A down-only
rank-two correction contains 16,384 factor values per expert. BF16 factors
therefore occupy 32,768 raw bytes per expert and 262,144 raw bytes across the
eight-expert panel. This is about 0.006944 bits per expert-weight equivalent
when charged across the expert's gate, up, and down matrices. Eight-bit values
would halve the raw factor bytes before scales, headers, indices, and
alignment. The logical panel margin below EXL3 is 20,148,224 bytes, so the
BF16 panel candidate fits comfortably. Only a serialized GLM container can
establish checkpoint size.

### Compare against buildable rate allocations under the same byte cap

The low-rank correction must beat the best buildable use of the same bytes.
The GLM panel can already build complete K3 and K4 matrix candidates. Include
the low-rank candidates and the coherent K3/K4 candidates in one charged-byte
comparison. Remove any candidate that another candidate beats in both KLD and
bytes. If the rate format cannot spend an adapter-sized budget because one
matrix promotion is the smallest legal rate change, report that granularity
instead of inventing a fractional promotion.

Select adapter and rate-allocation candidates on the same documents. Compare
complete artifacts with identical non-expert tensors. Nominal bits per weight
cannot substitute for serialized bytes.

### Keep gradient training behind the closed-form result

Gradient training becomes eligible only when the closed-form candidate leaves
a measured deficit that is large enough to matter and small enough for
continuous recovery to plausibly close. Initialize any trained adapter from
the accepted closed-form factors. Report the trained-over-initialized KLD
change separately so the value of training is measured.

Training inputs become recovery-training data even though teacher outputs
supply the labels. Keep those documents disjoint from factor fitting,
candidate selection, and confirmation. Evaluate only factors rounded
to their stored dtype and loaded through the packed serving path.

### Execute the GLM low-rank comparison in a frozen order

| Order | Work | Recorded state or advance rule |
|---:|---|---|
| 1 | Fit BF16 down-only rank-two and rank-four factors for the frozen eight-expert panel | Complete: both ranks improved activation-weighted error on candidate-selection rows |
| 2 | Measure bolt-on factors through complete-expert reconstruction and one-context model KLD | Complete: all-expert rank two failed; rank-four expert 103 reached KLD 0.05825746 |
| 3 | Freeze the exact rank-four expert-103 candidate | Complete in `experiments/glm52_layer3_rank4_expert103_low_rank_down_confirmation_registration.json`; no candidate field may change before confirmation |
| 4 | Evaluate once on at least 32 document-disjoint confirmation contexts | Advance only if paired mean KLD beats EXL3, CVaR1% is non-inferior, and the logical byte advantage survives exact serialization |
| 5 | Implement deterministic factor serialization and a factor-aware inference path | The loaded factors reproduce the screened dense endpoint within a frozen numerical tolerance, and the exact byte ledger remains smaller |
| 6 | Replicate the frozen construction across independently selected layers | Recovery and KLD per byte repeat before model-wide allocation |
| 7 | Compare against coherent K3/K4 configurations at matched exact bytes | The comparator is the best buildable rate allocation; uniform K3 alone is insufficient |
| 8 | Run frozen-scale re-encoding and one guarded alternation only if confirmation leaves material headroom | Retain alternation only when it improves over the bolt-on candidate on separate selection documents |
| 9 | Test eight-bit factors, then conditionally four-bit factors | Retain a smaller dtype only when it passes functional, KLD, tail, and byte gates on separate selection documents |
| 10 | Train factors only if the closed-form result leaves a material, plausibly recoverable deficit | Training must beat its serialized closed-form initialization on disjoint data |

The small expert panel selects the factor rank, dtype, and bolt-on or
alternating construction. It cannot establish a checkpoint win. Only the
complete serialized GLM artifact under the paired confirmation protocol can
establish that QSRT is smaller than EXL3 and has lower KLD.

## GLM-5.2 execution order and advance rules

| Work | Result required before advancement |
|---|---|
| Verify model identity and prepare reference logits | Weight identity is reconciled; at least 32 sealed confirmation contexts remain required for the frozen rank-four expert-103 candidate |
| Freeze the down-construction rule | Reconstructed-input covariance excluded; one identity-metric target, ridge, and fallback rule frozen by measured KLD on eight selection contexts |
| Confirm the frozen low-rank candidate | The registered rank-four expert-103 correction beats EXL3 mean KLD and satisfies CVaR1% non-inferiority on at least 32 document-disjoint contexts |
| Serialize and execute low-rank factors | Factor-aware loading reproduces the screened endpoint; exact serialized bytes remain below EXL3 |
| Build coherent rate-conditioned candidates | Every upstream rate pair has its own down construction |
| Select and confirm one complete panel configuration | The frozen configuration beats EXL3 mean KLD, satisfies CVaR1% non-inferiority, and uses fewer charged panel bytes on confirmation documents |
| Test transfer across layers | The selected construction repeats across error-blind panels from independently chosen layers |
| Build the complete checkpoint | Serialized bytes, forward KLD, task quality, and production-serving checks all pass |

Artifact forensics run alongside candidate construction when their inputs are
available. Their findings select a follow-up investigation when a candidate
fails. They do not delay bounded source transfer, low-rank factor construction,
or construction of unselected rate-conditioned candidates. They cannot replace
the selection-context KLD required to freeze a down rule or rate allocation.

## Verify model and artifact identity

The bfloat16 (BF16) source tensors identify GLM-5.2 revision
`b4734de4facf877f85769a911abafc5283eab3d9`. The published teacher-logit
manifest identifies revision
`4d67f66cc64d3219133b767c253b2ad1425c6c88`. The two immutable revisions have
byte-identical safetensors indexes. Every safetensors object also has the same
Hugging Face large-file SHA-256 identity and size in both revisions. This
metadata closes weight identity without downloading the complete BF16 model.

The configurations are not identical. The source revision explicitly sets
the mixture-of-experts router computation to 32-bit floating point through
`moe_router_dtype: float32`; the teacher revision omits that field. Reference
generation must therefore preserve and report the teacher runtime behavior.
The weight-identity result does not establish runtime identity.

The published reference-logit dataset contains one 2,048-token WikiText
context. It was produced from the teacher revision with tensor parallelism 16,
B12X sparse attention, eight-bit floating-point key/value cache storage, and
32-bit floating-point logits. That single context remains a wiring and
candidate-development control. It cannot supply the eight selection contexts
or the 32 sealed confirmation contexts. Producing those references on an
authorized host that already holds the BF16 teacher is an active dependency;
the GLM program must not download the complete BF16 checkpoint to satisfy it.

Before another GPU run, inventory the artifacts needed by each analysis:

| Analysis | Required retained artifacts |
|---|---|
| KLD tail statistics and migration | Position-level KLD tensors |
| Route-change analysis | Per-layer route identifiers and routing margins |
| Complete-expert error alignment | Routed inputs and reconstructed candidate endpoints |
| Input leverage | Routed inputs and the regularized input covariance used by the encoder |
| Down-refit tail analysis | Routed inputs, source expert outputs, and source/refitted endpoints |

The first analysis is CPU-only from the preserved KLD tensors. The remaining
analyses are CPU-only only when every listed tensor was retained in a directly
readable form. Missing endpoints or routed rows require one bounded replay.
The artifact inventory must record this distinction rather than describing all
forensics as zero-compute work.

## Separate down conditioning from down-target refitting

The down-construction experiment determines why the down intervention improved
mean KLD and how to control its tail. Gate and up remain at uniform K3. Each
expert receives four complete down constructions.

| Input metric used to encode the down matrix | Continuous down target |
|---|---|
| Identity | Original source weights |
| Covariance of activations reconstructed by quantized gate and up matrices | Original source weights |
| Identity | Target fitted to reproduce the source expert output |
| Covariance of activations reconstructed by quantized gate and up matrices | Target fitted to reproduce the source expert output |

The four policies have been encoded and measured on the available reporting
context. Reconstructed-input covariance with the source target reduced local
complete-expert error by 48.7027% across all eight experts, then made model
mean KLD 7.8227% worse than EXL3. Adding three accepted refits recovered
42.5415% of that policy's excess KLD above EXL3, but the combined policy still
lost by 4.4948%.

The locally selected identity-metric refit also failed. It matched five of the
earlier refitted down tensors byte for byte. For the remaining three experts,
it changed two ridge choices and replaced one source fallback with a refit.
Those locally favorable choices moved model mean KLD from 0.06123869 for the
earlier refit to 0.06413429. The four-policy measurement therefore rejects
reconstructed-input covariance and rejects local expert mean plus local
row-CVaR as the refit selector. It does not reject the earlier fixed refit,
which remains unconfirmed beyond one context.

The policies are not a strict tensor-level factorial. The same continuous
target candidates have matching hashes across input metrics, but each metric
performs its own hard encoding and applies its own ridge and fallback choices.
Use the results to accept or reject complete construction policies. Do not
attribute every KLD difference to one matrix operation.

The reconstructed-activation covariance describes the input that the encoded
down matrix will receive. This covariance summarizes the directions and scales
of the hidden activations and weights reconstruction error toward frequently
excited directions. The fitted target solves a regularized least-squares
problem so the complete quantized expert reproduces the source expert output.
The regularization strength is called the ridge strength. It limits how far
the fitted target can move from the source down weights.

Test a small ridge grid within the identity-metric fitted-target policy. Freeze
the grid before candidate measurement. Fit each target on activation-fit
documents and use complete-expert error only to prune clearly dominated hard
encodes. Choose ridge and fallback decisions with measured KLD on the eight
candidate-selection contexts. Evaluate the complete gated expert because gate,
up, and down errors interact.

Retain the source-target encode as an expert-specific fallback. Local mean and
row-tail error may remove a refit that is dominated on both measures, but they
may not select a refit or authorize its fallback decision. Do not derive a
fallback decision from one worst token. Model-level tail safety is judged by
document-level CVaR after the expert choices are assembled.

The eight KLD selection contexts choose one identity-metric down-construction
rule, including its target policy, ridge decisions, and fallback rule. The
chosen rule becomes an input to the mixed-rate experiment. The sealed
confirmation contexts remain unopened during this decision.

## Build complete rate-conditioned expert candidates

The mixed-rate experiment builds eight complete candidates for each expert.

- K3 gate, K3 up, K3 down;
- K3 gate, K3 up, K4 down;
- K3 gate, K4 up, K3 down;
- K3 gate, K4 up, K4 down;
- K4 gate, K3 up, K3 down;
- K4 gate, K3 up, K4 down;
- K4 gate, K4 up, K3 down; and
- K4 gate, K4 up, K4 down.

Each gate/up rate pair creates a different reconstructed hidden activation.
For each of the four gate/up pairs, the encoder must perform these operations:

1. Reconstruct the selected gate and up matrices.
2. Execute them on activation-fit rows.
3. Build the down-input covariance for those reconstructed activations.
4. Fit a separate down target when the selected down rule uses refitting.
5. Encode that pair-specific target independently at K3 and K4.

This construction prevents an upstream rate change from silently invalidating
the down metric or target. The stored candidate contains only ordinary QSRT
matrices; the continuous fitted target adds no checkpoint bytes.

The measured control implementation in
[`qsrt/glm52_down_refit_rate_pool.py`](../qsrt/glm52_down_refit_rate_pool.py)
recomputes one down target from the K3/K3 upstream pair and reuses it when gate
or up changes rate. Its frozen registration is
[`experiments/glm52_layer3_rate_preserving_down_refit_k3_k4_pre_registration.json`](../experiments/glm52_layer3_rate_preserving_down_refit_k3_k4_pre_registration.json).
Preserve that registration and its negative KLD reports as the one-target
control. It cannot govern the rate-conditioned candidate experiment. The
rate-conditioned build needs a new registration and pair-specific down
metrics, targets, hashes, and receipts.

### Select a panel configuration with measured KLD

Complete-expert output error remains useful for pruning. Its earlier inversion
shows that it cannot authorize the final configuration by itself. Use the
following bounded selection procedure for the eight-expert panel:

1. Build all eight coherent candidates for every expert.
2. Remove candidates that another candidate beats in both bytes and
   complete-expert error.
3. Retain the best two or three candidates per expert under the down
   experiment's selected complete-expert metric.
4. Use an exact-byte dynamic program or bounded beam search to produce at most
   sixteen promising complete panel configurations.
5. Measure those configurations on the eight KLD selection contexts.
6. Freeze one configuration and its exact selection rule.
7. Evaluate that configuration once on at least 32 sealed confirmation
   contexts.

The eight selection contexts become training evidence as soon as their KLD
chooses a configuration. Only the confirmation result may close the experiment.
This bounded KLD search is feasible for eight experts. A complete model will
need a validated allocation score and a separate confirmation set because an
exhaustive model-level KLD search does not scale to every expert.

The layer-3 logical budget permits at most twelve K4 matrix promotions:

```text
uniform K3 panel                    113,643,520 bytes
twelve K4 matrix promotions         18,874,368 bytes
mixed QSRT panel                    132,517,888 bytes
EXL3 comparison panel               133,791,744 bytes
logical QSRT margin                   1,273,856 bytes
```

This arithmetic excludes a GLM QSRT container. The panel passes its size screen
only provisionally. The complete artifact must later charge scales, the shared
reconstruction table, headers, directories, alignment, and padding.

### Close the layer-3 experiment

The layer-3 experiment closes only when its frozen configuration has all of the
following evidence:

- paired mean KLD against EXL3 on the confirmation documents;
- document-level bootstrap uncertainty;
- the CVaR1% non-inferiority verdict;
- candidate-specific tail migration and route-change diagnostics;
- the logical panel byte ledger;
- fit, selection, and confirmation document manifests;
- candidate, source, teacher, runtime, and comparison-checkpoint identities;
  and
- reproducible artifact hashes and process controls.

A lower mean on the eight KLD selection contexts does not close the experiment.
A KLD report without the frozen byte ledger also remains incomplete.

## Test transfer across layers before model-wide allocation

A layer-3 result cannot establish behavior in later mixture layers. After the
coherent layer-3 configuration passes, repeat the selected construction on
error-blind expert panels from additional layers.

The external claim that layers 60 and 64 are sensitive supplies a sampling
prior. Layers 52 and 63 provide nearby controls. Freeze expert lists before
candidate errors are available. On separate documents, use the engine hook to
return replacement expert outputs one layer at a time. This sweep produces an
independent layer-damage map instead of treating the external list as ground
truth.

For each document `d`, let `gain(layer, d)` be the reduction in paired mean
KLD from applying the same bounded intervention at that layer. The screening
statistic is

```text
(gain(60, d) + gain(64, d)) / 2
    - (gain(52, d) + gain(63, d)) / 2.
```

The external sensitivity prior passes the rapid screen when the statistic is
positive in at least six of eight selection documents. It passes confirmation
when the lower bound of its equal-document paired-bootstrap interval is above
zero on at least 32 sealed documents. Retain the four individual layer gains
so that one unusually large layer cannot hide the other three. Failure means
that the external map is unconfirmed under this protocol; the independent
layer sweep still proceeds.

The bounded source window for layers 52, 60, 63, and 64 contains 3,072 routed
expert tensors in 17 official shards totaling 91,142,336,944 bytes. Each shard
must match its immutable source SHA-256 identity before encoding begins. This
window is sufficient for the four-layer comparison and does not contain the
complete BF16 model.

Panel manifests and bounded source-shard plans can be prepared while layer-3
confirmation runs. Candidate encoding should begin only after the layer-3
construction passes. Download only the source shards that contain the selected
layers; the transfer test does not require the complete BF16 checkpoint.

Use new selection and confirmation documents when a preceding reporting set
has influenced layer selection or allocation. Compare the direction and
ranking of complete-expert, complete-layer, and full-model results. If the
small panel fails to predict layer behavior, redesign the panel before using it
to allocate checkpoint-wide bytes.

## Build and validate a complete checkpoint

Model-wide allocation chooses complete expert candidates under a complete
serialized-byte budget. The allocator must charge each format directory,
payload stack, scale plane, table, header, alignment gap, and padding region.
Non-expert weights must remain byte-identical to the EXL3 comparison inputs.

Materialize the complete checkpoint into a fresh path. Validate decode,
payload hashes, malformed-input rejection, tensor-parallel ownership, routed
expert outputs, and live routing. Then run paired document-disjoint KLD, task
quality, long-generation checks, and production latency.

Only this complete-artifact result can support the stated objective. A panel
win authorizes model construction; it does not imply the complete model will
win.

## Run reusable forensics in parallel

The following analyses may run alongside identity closure and candidate
construction when their required artifacts are available. They do not block
the down-construction or coherent-pool experiments.

### Align local error, KLD damage, and routing changes

For uniform K3, input-covariance-selected K3, and down-refitted K3, align each
scored position with its routed rows and complete-expert output error.
Classify damaged positions into three cases:

- local expert error also worsens;
- local expert error improves while routes remain unchanged; or
- local expert error improves and a later top-k route changes.

The cases select different follow-up work. The first indicates inadequate
row or tail coverage. The second indicates missing downstream direction or
interaction. The third indicates a routing discontinuity.

### Measure input leverage

Input leverage measures how unusual a routed row is under the regularized
input covariance used by the encoder. For an input row `x` and covariance `H`,
the diagnostic uses the Mahalanobis quantity `x.T H^-1 x` in the same basis and
with the same regularization as candidate construction.

Measure whether the input-covariance candidate's KLD damage increases with
leverage on documents that did not fit `H`. Report gate/up and down leverage
separately. A positive association authorizes a small covariance-eigenvalue
floor sweep. Absence of the association keeps that sweep out of the GPU queue.

### Locate down-refit tail damage

Identify the experts and routed rows that contribute to the down-refit's
candidate-specific worst one percent. Test their relation to ridge strength,
input leverage, the source-target fallback expert, route changes, and
particular co-routed experts. Use the result to define the ridge grid and local
fallback statistic. Do not tune a rule against the sealed confirmation set.

## Trigger expensive investigations from observed failures

| Observed result | Authorized investigation | Decision supplied by the investigation |
|---|---|---|
| KLD damage rises with input leverage | Sweep a small lower bound on the regularized input-covariance eigenvalues | Whether robust input conditioning improves mean KLD without violating CVaR1% |
| Complete-expert error improves, routes remain stable, and full-model KLD worsens | Capture residual-stream output gradients and test two-sided downstream-loss curvature | Whether downstream sensitivity ranks real candidate errors better than complete-expert error |
| Damaging positions coincide with small top-k margins and route changes | Add routing-margin and route-stability features to candidate scoring | Whether routing discontinuities explain enough damage to justify a routing-aware selector |
| Partial expert swaps show unexplained interaction with resident EXL3 errors | Stream the first-order logit cross-term and the teacher-probability-weighted second-order term (Fisher quadratic) alongside the observed KLD change | Whether existing model error amplifies or cancels the candidate perturbation |
| The source of SQG's scalar advantage remains important after the system candidate is competitive | Re-encode matched SQG and ExLlamaV3's computed scalar reconstruction law (MCG) at K3 | Whether SQG's mean scalar advantage extends to high-leverage rows and row-tail error |

The two-sided curvature code already has synthetic and complete real-matrix CUDA
closure. Source-basis identity output curvature and explicit zero-output
feedback reproduce ordinary K3 bit for bit. A real curvature experiment still
needs residual-stream output gradients, matched input-gradient samples,
route-support-aware shrinkage, and validation on full-size K3 and K4 errors.

The factored two-sided score approximates the true sample-averaged curvature.
Routing correlates input rows, output gradients, and router coefficients. A
real experiment must compare the factorized score with matched per-sample
quadratic scores. Complete-expert reconstructed propagation remains the final
local acceptance boundary because separate matrix scores omit gate/up
interactions.

A covariance eigenvalue measures activation variance along one direction. A
lower bound prevents the encoder from treating a low-variance direction as
arbitrarily unimportant. The leverage analysis must show that this failure
pattern exists before the encoder spends GPU time on the sweep.

## Keep rejected and deferred mechanisms outside the critical path

- Reopen reconstruction-table training only when another model, rate, or
  loss-weighted residual diagnosis shows material finite-table headroom.
- Revisit BlockLDLQ removal only with a K2 or captured-metric result that
  changes trellis paths after scale search is held constant.
- Build a sparse residual plane only when held-out curvature-weighted residual
  concentration beats a K4 promotion per charged byte. One indexed correction
  per 256-value tile should explain more than six percent of tile damage; two
  should explain more than ten percent before format work begins.
- Treat discrete cosine and Walsh-Hadamard residual modes as offline controls.
  The transformed weight layout has no established image-like locality.
- Defer a gate/up pair trellis until the scalar, down-conditioning, and
  allocation program is measured. A pair format requires a new encoder,
  decoder, feedback rule, and exact-byte contract.
- Keep K5 outside the production path. The production SQG quantizer supports
  K2, K3, and K4, and prior K5 work found the finite E4M3 endpoint to be the
  limiting error.

## Stop or advance mechanically

Advance the layer-3 candidate to cross-layer testing only when it beats EXL3 in
paired mean KLD, satisfies the frozen CVaR1% constraint, and occupies fewer
charged panel bytes on at least 32 confirmation documents.

If the coherent candidate fails, choose one investigation from the observed
failure pattern:

- leverage-associated damage selects robust input conditioning;
- stable-route local improvement with KLD regression selects output-gradient
  curvature;
- route-boundary damage selects routing-aware scoring; and
- absence of all three patterns selects broader rate and layer allocation
  analysis or termination of the mechanism.

Do not run every branch after a failure. Each added experiment must answer the
specific unresolved cause and must retain the same data-separation and
acceptance contract.

The immediate blocking input for the registered rank-four expert-103 candidate
is at least 32 sealed confirmation contexts. The complete BF16 checkpoint must
not be downloaded to produce them. An authorized host that already holds the
teacher must generate the references. While that work proceeds, kossel may
finish the bounded source windows, implement factor-aware serialization and
execution, and construct unselected coherent rate-conditioned candidates.

Do not change the registered correction after opening a confirmation context.
A different rank, expert, dtype, ridge, factor value, base representation, or
construction creates a different candidate and requires a separate sealed
confirmation set. Any future rate allocation must wait for selection-context
KLD to freeze its identity-metric down rule. Every upstream rate pair gets a
separately reconstructed down input and target.

## Authority and evidence records

The codec specifications govern stored formats and decoder behavior. This
strategy governs experimental order, data separation, and promotion decisions.
The chronological journal records completed operations and rejected runs.

- [GLM-5.2 layer-3 KLD results](glm52-layer3-kld-results.md) records the
  controlled one-context candidate comparisons and artifact hashes.
- [GLM-5.2 experiment journal](glm52-experiment-journal.md) records source,
  teacher, container, capture, and generated-artifact identities.
- [QSRT and EXL3 comparative assessment](qsrt-exl3-comparative-assessment.md)
  records the rate, allocation, and codec evidence boundary.
- [Two-bit trellis research corpus](qsrt-two-bit-research-corpus.md) records
  the academic evidence, implementation costs, and rejected alternatives.

If a proposal-ranking section in another document conflicts with this testing
order, follow this strategy. Preserve the older claim only in a document whose
purpose is to record experimental history.

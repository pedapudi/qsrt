# GLM-5.2 layer-3 QSRT mechanism results

## Evaluation goal

This study asks whether a small set of format-preserving QSRT changes can
reduce the forward Kullback–Leibler divergence (KLD) of a uniform three-bit
trellis replacement for eight routed GLM-5.2 experts.

Forward KLD compares the candidate model's output probabilities with
probabilities saved from the official BF16 model. Lower values are better. The
comparison checkpoint is the immutable GLM-5.2 EXL3 checkpoint stored on
kossel. Only eight experts in mixture layer 3 change; all other weights remain
those of EXL3.

## Evidence boundary

The published BF16 reference contains one 2,048-token context, which produces
2,047 next-token comparisons. Tokens from one document are correlated, so they
cannot provide a document-level confidence interval. The results below can
reject a large regression and compare candidates within the context. They
cannot establish complete-checkpoint quality or generalization to other
documents, layers, experts, or serving kernels.

The bounded source tensors and published reporting logits name different
GLM-5.2 revisions. The source window and sealed tensor inventory name revision
`b4734de4facf877f85769a911abafc5283eab3d9`. The published reference-logit
manifest names teacher revision
`4d67f66cc64d3219133b767c253b2ad1425c6c88`. A metadata-only Hugging Face
inventory on 2026-08-18 proved that the two revisions have the same
safetensors index and the same content SHA-256 and byte count for every
official weight shard. The source weights are therefore byte-identical. The
teacher revision omits the source revision's explicit
`moe_router_dtype: float32` configuration field, so runtime configuration
identity remains a separate requirement. Every comparison below uses the same
reporting teacher and runtime contract.

The measurement uses vLLM's eager, dense-attention, per-expert EXL3 correctness
path on four RTX Pro 6000 GPUs. It does not use the production fused mixture-of-
experts kernel. An engine-side hook computes `KL(BF16 reference || candidate)`
four rows at a time, avoiding a full-vocabulary sort and keeping the saved BF16
reference on CPU-backed storage.

Each candidate process reruns two controls before evaluating the candidate:

- an unchanged resident EXL3 repeat; and
- a direct-return intervention that returns the resident expert output.

Both controls reproduced all 2,047 per-position KLD values bit for bit. They
also reproduced all `2,048 × 78 × 8` routed expert identifiers. Every measured
candidate process produced the same resident EXL3 KLD vector bit for bit.

## Results

| Eight-expert layer-3 representation | Mean forward KLD | Change from EXL3 | Change from uniform K3 | Decision |
|---|---:|---:|---:|---|
| Resident EXL3 | 0.0610743407031 | — | −2.0904% | Comparison checkpoint |
| Uniform QSRT K3 | 0.0623782807651 | +2.1350% | — | Quality control; worse than EXL3 |
| Uniform K3 with BlockLDLQ feedback disabled at frozen scales | 0.0623782807651 | +2.1350% | 0.0000% | Byte-identical to uniform K3 |
| K3 selected with one-sided routed-input covariance | 0.0638412662800 | +4.5304% | +2.3453% | Reject |
| K3 with reconstructed-activation down refit | 0.0612386895257 | +0.2691% | −1.8269% | Confirm on more documents |
| Down refit with BF16 rank-two corrections on all eight experts | 0.0652326383334 | +6.8086% | +4.5759% | Reject the all-expert local selector |
| Down refit with BF16 rank-two corrections on experts 89 and 103 | 0.0601683116025 | −1.4835% | −3.5429% | Exploratory; expert identities came from this reporting context |
| Down refit with a BF16 rank-four correction on expert 103 | 0.0582574646070 | −4.6122% | −6.6062% | Below 0.059 on one context; requires document-disjoint confirmation |
| K3 down encoded with reconstructed-input covariance and source weights | 0.0658519849381 | +7.8227% | +5.5688% | Reject |
| K3 with a locally selected identity-metric down refit | 0.0641342908893 | +5.0102% | +2.8151% | Reject the local selection rule |
| K3 with reconstructed-input covariance and locally selected down refits | 0.0638195014718 | +4.4948% | +2.3105% | Reject |
| Fixed twelve-promotion mixed K3/K4 over the down-refit base | 0.0659634015775 | +8.0051% | +5.7474% | Reject |
| Ten-promotion mixed K3/K4 selected by complete-expert error, with one shared down-refit target per expert | 0.0636258118201 | +4.1776% | +1.9999% | Reject this construction |
| Fixed twelve-promotion mixed K3/K4, with one shared down-refit target per expert | 0.0639669166209 | +4.7362% | +2.5468% | Reject this construction |

The down refit recovered 87.3960% of uniform K3's excess mean KLD above EXL3.
It did not beat EXL3. Its position-level changes were also heavy-tailed: 1,004
positions improved and 1,043 regressed relative to uniform K3, even though the
mean improved.

The fixed mixed-rate candidate was 7.7152% worse than the down-refitted K3
base. Its twelve promoted projections came from an allocation frozen from
EXL3 rates before candidate measurement. Source-target K4 tensors replaced the
base tensor in each promoted cell. A promoted down projection therefore did
not retain the K3 refit's continuous target. The result rejects that fixed
allocation and target combination. It does not test selection-data allocation
or K4 encoding of the reconstructed-activation down target.

The later rate-pool controls preserved each accepted down-refit target when
the down projection changed from K3 to K4. Complete-expert error on frozen
selection documents chose ten K4 projections. That candidate was 3.8981%
worse than the K3 down-refit base and 4.1776% worse than EXL3 on the reporting
context. The fixed twelve-promotion control was 4.4551% worse than the K3
down-refit base and 4.7362% worse than EXL3.

Both rate-pool controls fitted one down target from K3 gate and K3 up
activations, then reused that target when either upstream projection changed
rate. They therefore do not test a coherent rate-conditioned down refit. The
next construction must rebuild the down input, metric, and target separately
for each gate/up rate pair before encoding the down target at K3 and K4.

The one-context tail statistics reinforce the rejection while illustrating why
p99 alone is insufficient:

| Representation | p99 forward KLD | CVaR1% forward KLD |
|---|---:|---:|
| Resident EXL3 | 1.0996727300 | 1.9557645264 |
| Ten-promotion selection-data candidate | 1.0611624646 | 2.0247655369 |
| Fixed twelve-promotion rate-pool control | 1.0236869001 | 2.1575191191 |

Both candidates lowered p99 but worsened the mean loss among their 21
worst-scoring positions. These token-level values come from one document and
cannot supply a document-level non-inferiority verdict.

The completed down-construction comparison produced the following tail
statistics. The locally selected identity-metric refit used the same down
tensor as the earlier refit for five experts. It changed the ridge choice for
experts 106 and 204 and replaced expert 208's source fallback with a refit.
Those three local choices changed mean KLD from `0.0612386895257` to
`0.0641342908893`.

| Down construction | p99 forward KLD | CVaR1% forward KLD | Maximum forward KLD |
|---|---:|---:|---:|
| Resident EXL3 | 1.0996727300 | 1.9557645264 | 5.5796093941 |
| Uniform K3 | 1.1885003185 | 2.0497293302 | 4.9111685753 |
| Earlier reconstructed-activation refit | 1.0591630125 | 2.0953120788 | 5.9141564369 |
| Reconstructed-input covariance with source target | 1.3193669415 | 2.2089717615 | 6.3604264259 |
| Locally selected identity-metric refit | 1.1236556482 | 2.1865061294 | 5.5795865059 |
| Reconstructed-input covariance with locally selected refits | 1.0971495223 | 2.0218569438 | 4.5884408951 |

Reconstructed-input covariance lost on mean, p99, CVaR1%, and maximum when it
encoded the source target. The three accepted refits in the covariance/refit
cell recovered 42.5415% of that cell's excess mean KLD above EXL3 and improved
its tail statistics. The complete covariance/refit policy still lost to EXL3
and uniform K3 on mean KLD. All three down-construction candidates changed
downstream routes beginning at layer 4. The layer-3 routes used to invoke the
intervention did not change.

The four cells compare complete construction policies. They are not a strict
tensor-level factorial because each input metric applies its own hard-encoded
ridge and fallback decisions. Every continuous target candidate had the same
hash under both metric policies, which closes target generation. The selected
materialized target can still differ after metric-specific encoding and local
selection.

## Activation-weighted low-rank down corrections

The low-rank construction starts from the earlier reconstructed-activation
down-refit artifact. It holds each quantized gate and up matrix fixed, derives
the hidden rows that those matrices produce, and fits a small additive
correction to the down matrix. The fit minimizes complete-expert output error
on activation-fit documents. A disjoint candidate-selection split chooses the
ridge coefficient. The model-KLD context does not enter factor fitting.

Rank two stores 32,768 logical BF16 factor bytes per corrected expert. Rank
four stores 65,536 bytes. The bounded runtime screen multiplies the rounded
factors together and adds their product to the dense down endpoint. A
factor-aware QSRT container and two-matrix serving branch remain unimplemented.
The byte counts describe the proposed stored factors rather than the dense
screening artifact.

Across all eight experts, rank two reduced pooled routed complete-expert error
by 67.6456% on candidate-selection rows. Rank four reduced the same error by
70.7994%. Applying every rank-two correction made model mean KLD 6.8086% worse
than EXL3. Local complete-expert error therefore failed again as a model-level
selection rule.

The pre-specified individual rank-two arms separated three helpful experts
from five harmful experts on the reporting context:

| Corrected expert | Mean forward KLD | Change from EXL3 |
|---:|---:|---:|
| 89 | 0.0601812335253 | −1.4623% |
| 103 | 0.0609060687529 | −0.2755% |
| 208 | 0.0603366874259 | −1.2078% |

Combining the three rank-two corrections was not additive. Experts 89 and 103
produced the best rank-two combination at 0.0601683116025. Adding expert 208
raised mean KLD to 0.0634722584106. The pair containing experts 89 and 208
also regressed to 0.0636328828564.

The rank-four screen reused the three identities selected by rank-two KLD.
Rank four on expert 103 produced mean KLD 0.0582574646070, which is 4.6122%
lower than EXL3 and below the numerical target of 0.059. Rank four on expert
208 reached 0.0591001769552. Rank four on expert 89 regressed to
0.0679760778071. The expert-103 candidate improved the measured tail as well
as the mean:

| Representation | p99 forward KLD | CVaR1% forward KLD | Maximum forward KLD |
|---|---:|---:|---:|
| Resident EXL3 | 1.0996727300 | 1.9557645264 | 5.5796093941 |
| Rank-four correction on expert 103 | 0.9685485280 | 1.8109366610 | 4.3553042412 |

Seventeen of the 21 worst-scoring positions were common to both
representations. Four positions entered and four left the candidate's worst
one percent. The expert-103 intervention preserved every layer-3 route and
changed downstream routes beginning at layer 4. It changed 284,052 routed
expert identifiers across the complete `2,048 × 78 × 8` route array.

The 0.0582574646070 result is a mechanism screen. The rank-four expert set was
restricted after inspecting rank-two KLD on the same reporting context. The
one available context cannot estimate document-level uncertainty or correct
this selection bias. A valid promotion requires frozen expert 103, the same
rank, factor dtype, ridge, and construction to repeat on document-disjoint
selection and confirmation contexts. The shipped artifact must also execute
serialized factors rather than the materialized dense product used here.

The exact candidate is frozen in
`experiments/glm52_layer3_rank4_expert103_low_rank_down_confirmation_registration.json`.
The registration records both factor hashes, ridge `0.001`, the base artifact,
the materialized endpoint hash, the logical byte screen, and the independent
confirmation rule. Changing any registered field creates a different
candidate and requires a separate sealed confirmation set.

## Interpretation by mechanism

### Reconstruction-table training

Do not implement table training for the measured residual domain. Production
post-BlockLDLQ values were nearly Gaussian, adjacent correlation was
negligible, and every measured matrix used all 4,096 table entries. A
per-matrix fixed-path oracle with finite E4M3 centroids reduced pooled squared
error by 0.00175%. A shared production table has less freedom than that oracle.

### One-sided routed-input covariance

Reject one-sided input covariance as a path-selection objective. It produced
large complete-expert output squared-error reductions on the separate
candidate-selection documents and then made full-model KLD worse than both
EXL3 and uniform K3. The candidate-selection rows were held out from covariance
fitting, but they chose the stored candidate and therefore do not constitute an
untouched reporting set. The inversion shows that local routed squared error
cannot safely promote candidates.

The rejection does not test two-sided downstream-loss curvature. That method
also needs output gradients and an output-side metric. The repository now
contains the two-sided recurrence, output-metric factorization, bounded factor
format, and frozen-scale GLM candidate encoder. Synthetic and complete
real-matrix CUDA closures pass. The bounded output-gradient capture needed to
construct real GLM factors remains unavailable, so no two-sided KLD result
exists.

The complete-matrix identity audit found and corrected one numerical failure.
For the tested gate matrix, source-basis identity output curvature is
algebraically a scalar identity after the output transform. FP32 Hadamard
round-off first introduced off-diagonal terms up to `1.12593e-8`. The block
factorization turned them into 114,688 nonzero feedback entries, and the hard
trellis path changed. The factorizer now preserves algebraic scalar identity
directly. Both source-basis identity and an explicit zero-output-feedback
control reproduce the ordinary K3 trellis, scales, and dense reconstruction
bit for bit on the complete `2,048 × 6,144` matrix.

### BlockLDLQ feedback removal

Disabling BlockLDLQ at frozen K3 scales does not produce another representation
for this panel. The ablation preserved the source tensors, transforms, graph,
reconstruction table, global scales, and persisted scale vectors. It changed
the floating-point targets sent to Viterbi but changed none of the 24 selected
paths. Every reconstructed projection and all eight complete expert files were
byte-identical to uniform K3.

The byte identity makes another model run redundant: both endpoints must have
the same KLD. The result rules out feedback removal as a hidden improvement for
these eight layer-3 experts under the identity input metric. It does not test
K2, another layer, or a dense captured metric with larger feedback terms.

### Reconstructed-activation down refit

Retain the earlier down-refitted artifact as a candidate for document-replicated
selection, but reject local expert error as the rule that chooses its ridge and
fallback decisions. The encoder freezes the quantized gate and up matrices,
executes them on routed fit rows, fits a new continuous down target against the
source expert output, and re-encodes the target with unchanged K3. The earlier
construction accepted seven of eight refitted targets and kept the source K3
encode for the remaining expert.

The refit adds no payload because the continuous target is discarded. The
stored representation remains an ordinary K3 down matrix. A later construction
encoded every ridge candidate before selection and required both local mean and
local row-CVaR improvement. It accepted all eight refits but made model mean KLD
5.0102% worse than EXL3. Five materialized down tensors matched the earlier
artifact exactly; three changed as described above. The result proves that the
local tail guard does not repair the selector inversion. The earlier
one-context gain still justifies evaluation on more reference documents, but
neither construction defines a model-wide refit rule.

### Reconstructed-input covariance for the down matrix

Reject reconstructed-input covariance as the down-matrix encoding metric for
this panel. With the original source target, it reduced local complete-expert
error by 48.7027% across all eight experts and made model mean KLD 7.8227%
worse than EXL3. It also worsened all reported model-level tail statistics.

Combining the covariance metric with refitted targets improved model KLD by
3.0864% relative to covariance with source targets, but remained 4.4948% worse
than EXL3. This establishes down refitting as a correction within that policy;
it does not rescue the policy. Do not feed reconstructed-input covariance into
the coherent K3/K4 pool unless a later document-replicated experiment supplies
a different model-level result.

### Mixed K3/K4 allocation

Reject the pre-registered fixed rate-stratified allocation. It promoted twelve
projections selected from EXL3's immutable rate map and did not inspect QSRT
candidate measurements. The complete candidate remained 1,273,856 charged
logical bytes smaller than EXL3 across the panel, but its mean KLD was 8.0051%
worse.

The negative result has two narrower boundaries. First, the fixed map copied
EXL3 rate priorities even though QSRT conditions its down target on
reconstructed upstream activations. Second, a promoted down cell used a
source-target K4 tensor and replaced the accepted K3 refit. A later mixed-rate
test must rank complete-expert candidates on the frozen selection documents
and encode accepted down-refit targets at both K3 and K4 before comparing
rates.

The later rate-pool implementation recomputed each accepted continuous down
target, required its repeated K3 encode to equal the stored refit, and encoded
the same target at K4. It scored all eight K3/K4 rate tuples for each complete
expert on frozen selection documents. Both the fixed allocation and the
selection-data allocation failed on the reporting context.

This second failure is narrower than a rejection of mixed rates. The rate pool
held the down target fixed while changing gate or up rate. A coherent pool has
four upstream rate pairs: K3/K3, K3/K4, K4/K3, and K4/K4. Each pair produces a
different reconstructed down input and therefore needs its own down fit. Each
fitted target then needs K3 and K4 encodes, producing eight internally
consistent complete-expert candidates.

## Logical byte comparison

The eight selected experts contain 24 gate, up, and down matrices. Their
charged logical representation is:

| Component | EXL3 bytes | Uniform QSRT K3 bytes |
|---|---:|---:|
| Trellis payloads | 133,693,440 | 113,246,208 |
| Expert or matrix scales | 98,304 | 393,216 |
| Shared QSRT reconstruction table | 0 | 4,096 |
| Total | 133,791,744 | 113,643,520 |

The twelve-promotion mixed candidate occupies 132,517,888 charged logical
bytes. This leaves a 1,273,856-byte margin below EXL3 before a complete
serialized container charges headers, directories, alignment, and padding.
The KLD regression rejects the candidate regardless of that logical margin.

The ten-promotion selection-data candidate occupies 129,372,160 charged
logical bytes and leaves a 4,419,584-byte margin below EXL3. Its KLD regression
also rejects it regardless of the larger logical margin.

Uniform K3 saves 20,148,224 logical bytes across the panel. Uniform K3,
one-sided curvature, both down-refit constructions, and both
reconstructed-input-covariance constructions use the same K3 payload size. The
mixed-rate candidates use the larger totals stated above. A frozen GLM QSRT
container does not yet exist, so headers, alignment, padding, directories,
non-expert weights, and serving caches are absent from this calculation. The
numbers do not establish complete serialized model size.

A BF16 rank-four down correction on one expert adds 65,536 logical factor
bytes. Charged against the uniform-K3 panel and its zero-payload down refit,
the candidate totals 113,709,056 logical bytes. This remains 20,082,688 bytes
below the panel's EXL3 representation. Headers, factor scales, alignment,
directories, and the serving implementation remain uncharged until the GLM
container exists.

## Authoritative artifacts

Paths are relative to `/home/sunil/qsrt-glm52-experiments/` on kossel.

| Measurement | Result directory | Report SHA-256 |
|---|---|---|
| Repeatability control | `results/glm52-layer3-per-expert-exl3-engine-kld-paired-bf16-reference-kld-repeatability-control/` | `b22b39bbb6306519f461acaff9a862085041814e757d9aa8f715e3dc30d75bc0` |
| Uniform K3 | `results/glm52-layer3-frozen8-dense-endpoints-r7-closure-merged-v2-paired-bf16-reference-kld-engine-per-expert-correctness/` | `59dc890d56e1a48814b971836bf1544a86f79d0114043149a607564de8eada6b` |
| One-sided routed-input covariance | `results/glm52-layer3-frozen8-routed-input-curvature-merged-paired-bf16-reference-kld-engine-per-expert-correctness/` | `dc4df5478363582faa7ebca5d088e1d43a85a06d7e600d49c4cefc7c32ee373e` |
| Reconstructed-activation down refit | `results/glm52-layer3-frozen8-reconstructed-activation-down-refit-merged-paired-bf16-reference-kld-engine-per-expert-correctness/` | `d54093ec11d88664419039afa58bb7703a244ec8e0c0aa597db42a5c17cef21a` |
| Reconstructed-input covariance with source target | `results/glm52-layer3-frozen8-down-construction-reconstructed_input_covariance__source_weights-merged-paired-bf16-reference-kld-engine-per-expert-correctness/` | `52305af304917a2b6e2eca917281ea111acc7369c2817c55c05a7ad14c0f76d9` |
| Locally selected identity-metric down refit | `results/glm52-layer3-frozen8-down-construction-identity__reconstructed_activation_refit-merged-paired-bf16-reference-kld-engine-per-expert-correctness/` | `f43deb122e9f8e6152cc6501756c90602baaa398f542056769abf28ca5584dc9` |
| Reconstructed-input covariance with locally selected down refits | `results/glm52-layer3-frozen8-down-construction-reconstructed_input_covariance__reconstructed_activation_refit-merged-paired-bf16-reference-kld-engine-per-expert-correctness/` | `69f513f2f98d8ff76859800ba18de7d95b0be91b799774213d3a76c64ef6e962` |
| Fixed mixed K3/K4 over the down-refit base | `results/glm52-layer3-frozen8-fixed-mixed-k3-k4-down-refit-paired-bf16-reference-kld-engine-per-expert-correctness/` | `c366b25f8e3c4e1f231b0018dba241b25602dea98ad37ae337136756185f34c4` |
| Ten-promotion selection-data rate-pool control | `results/glm52-layer3-frozen8-selection-data-rate-preserving-down-refit-k3-k4-paired-bf16-reference-kld-engine-per-expert-correctness/` | `99a447ed2f0243679b24a98414b632a39c7830010d1b9fc96e8ca90f6d32d07e` |
| Fixed twelve-promotion rate-pool control | `results/glm52-layer3-frozen8-fixed-rate-preserving-down-refit-k3-k4-paired-bf16-reference-kld-engine-per-expert-correctness/` | `41ef7cf67ff0e9fc6bd977ead289c057ec2917fe35ff98cc4f5f41bca3aee6b9` |
| Rank-two low-rank individual attribution | `results/glm52-layer3-frozen8-low-rank-down-reconstructed_activation_down_refit-bf16-rank-2-merged-per-expert-attribution-paired-bf16-reference-kld-engine-per-expert-correctness/` | `f40d7a595535ff983de30316de181ec51da5f95a65af8eda006828bfb47a3ac3` |
| Rank-two attribution-selected combinations | `results/glm52-layer3-frozen8-low-rank-down-reconstructed_activation_down_refit-bf16-rank-2-merged-attribution-selected-subsets-paired-bf16-reference-kld-engine-per-expert-correctness/` | `2151734be97fdb6f28d9424b296c8fa4f16c8999965f218266fe0d373c77c078` |
| Rank-four attribution-selected singletons and combinations | `results/glm52-layer3-frozen8-low-rank-down-reconstructed_activation_down_refit-bf16-rank-4-merged-attribution-selected-subsets-paired-bf16-reference-kld-engine-per-expert-correctness/` | `8cf9d7a7f7332a16ac8accebde102f892dd9bc5e193145b677a12f6e30e0b39b` |

The codec-mechanism reports below do not contain new model KLD vectors.

| Codec mechanism | Report path | Report SHA-256 |
|---|---|---|
| Frozen-scale K3 with BlockLDLQ feedback disabled | `results/glm52-layer3-frozen8-blockldlq-no-feedback-frozen-k3-scale-identity-comparison/report.json` | `026369914fe0e1e9bf868cc21a77d9df19b07a1afb6358545432876661a10153` |
| Source-identity numerical-drift failure | `results/glm52-layer3-expert64-gate-source-identity-curvature-and-zero-output-feedback-bit-equivalence-cuda-closure.json` | `20a6b2a6fce197f6d22cff2a291abced722cb83e9d1e3ee6b2c55a5b8a2ff204` |
| Source-identity and zero-output-feedback bit-equivalence closure | `results/glm52-layer3-expert64-gate-source-identity-and-zero-output-feedback-bit-equivalence-after-scalar-identity-preservation.json` | `c0dc90c21da6c2105ffcb89eff8f8053a23da5785ef4a28b6d2bd9b621d5c398` |

Each result directory contains `report.json`, `measurement-controls.json`, one
per-position KLD tensor for every arm, and complete route arrays for the
resident, identity, and candidate arms. The complete file-by-file inventory is
`/home/sunil/qsrt-glm52-experiments/ARTIFACT_INDEX.json`. The chronological
record is [`glm52-experiment-journal.md`](glm52-experiment-journal.md).

## Next admissible experiments

1. Acquire or produce BF16 reference logits for multiple document-disjoint
   contexts without downloading the full BF16 checkpoint. Evaluate the
   registered rank-four expert-103 correction without changing it.
2. Use at least eight selection contexts to choose between the source target,
   the earlier fixed refit, and a bounded set of hard-encoded ridge/fallback
   policies. Complete-expert error may prune candidates but cannot choose the
   rule.
3. After freezing a down rule with measured selection-context KLD, build a
   coherent rate-conditioned pool. Reconstruct the down input and fit a
   separate down target for each gate/up rate pair, then encode each target at
   K3 and K4.
4. Test the same frozen construction on error-blind panels from layers 52, 60,
   63, and 64. Layers 60 and 64 are an external sensitivity prior; layers 52
   and 63 are nearby controls.
5. Add bounded output-gradient capture only if complete-expert fidelity
   improves with stable routes while document-replicated KLD still worsens.
6. Freeze a GLM QSRT container and count every serialized byte before making a
   model-size comparison.

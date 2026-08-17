# GLM-5.2 layer-3 QSRT mechanism results

## Question

Can a small set of format-preserving QSRT changes reduce the forward
Kullback–Leibler divergence (KLD) of a uniform three-bit trellis replacement
for eight routed GLM-5.2 experts?

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
GLM-5.2 revisions. The five-shard source window and sealed tensor inventory
name revision `b4734de4facf877f85769a911abafc5283eab3d9`. The published
reference-logit manifest names teacher revision
`4d67f66cc64d3219133b767c253b2ad1425c6c88`. Every candidate and EXL3 use the
same source lineage and the same reporting teacher, so their panel comparison
remains paired. The available evidence does not prove that the two official
revisions contain byte-identical weights.

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
| Fixed twelve-promotion mixed K3/K4 over the down-refit base | 0.0659634015775 | +8.0051% | +5.7474% | Reject |

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

Retain down refitting for document-replicated confirmation. The encoder freezes
the quantized gate and up matrices, executes them on routed fit rows, fits a new
continuous down target against the source expert output, and re-encodes the
target with unchanged K3. Seven of eight experts accepted a refitted target;
the remaining expert kept its source-target K3 encode.

The refit adds no payload because the continuous target is discarded. The
stored representation remains an ordinary K3 down matrix. The one-context gain
is large enough to justify more reference documents, but it is not evidence for
a complete checkpoint.

### Fixed mixed K3/K4 allocation

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

The corrective mixed-rate implementation is locally validated. It recomputes
each accepted continuous down target, requires its repeated K3 encode to equal
the stored refit, and encodes the same target at K4. It then scores all eight
K3/K4 rate tuples for each complete expert on the frozen selection documents.
Its immutable registration records the source and reporting-teacher revisions
separately. The four-GPU host was unreachable after the implementation passed
all 697 CPU tests, so no corrective candidate or KLD measurement exists yet.

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

Uniform K3 saves 20,148,224 logical bytes across the panel. Uniform K3,
one-sided curvature, and down refitting use the same K3 payload size. The
mixed-rate candidate uses the larger total stated above. A frozen GLM QSRT
container does not yet exist, so headers, alignment, padding, directories,
non-expert weights, and serving caches are absent from this calculation. The
numbers do not establish complete serialized model size.

## Authoritative artifacts

Paths are relative to `/home/sunil/qsrt-glm52-experiments/` on kossel.

| Measurement | Result directory | Report SHA-256 |
|---|---|---|
| Repeatability control | `results/glm52-layer3-per-expert-exl3-engine-kld-paired-bf16-reference-kld-repeatability-control/` | `b22b39bbb6306519f461acaff9a862085041814e757d9aa8f715e3dc30d75bc0` |
| Uniform K3 | `results/glm52-layer3-frozen8-dense-endpoints-r7-closure-merged-v2-paired-bf16-reference-kld-engine-per-expert-correctness/` | `59dc890d56e1a48814b971836bf1544a86f79d0114043149a607564de8eada6b` |
| One-sided routed-input covariance | `results/glm52-layer3-frozen8-routed-input-curvature-merged-paired-bf16-reference-kld-engine-per-expert-correctness/` | `dc4df5478363582faa7ebca5d088e1d43a85a06d7e600d49c4cefc7c32ee373e` |
| Reconstructed-activation down refit | `results/glm52-layer3-frozen8-reconstructed-activation-down-refit-merged-paired-bf16-reference-kld-engine-per-expert-correctness/` | `d54093ec11d88664419039afa58bb7703a244ec8e0c0aa597db42a5c17cef21a` |
| Fixed mixed K3/K4 over the down-refit base | `results/glm52-layer3-frozen8-fixed-mixed-k3-k4-down-refit-paired-bf16-reference-kld-engine-per-expert-correctness/` | `c366b25f8e3c4e1f231b0018dba241b25602dea98ad37ae337136756185f34c4` |

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

1. Run the prepared rate-preserving K3/K4 pool when the four-GPU host is
   available. Materialize both the frozen EXL3-stratified allocation and the
   allocation selected by complete-expert output error. Measure each frozen
   candidate against EXL3, uniform K3, and K3 down refitting.
2. Acquire or produce BF16 reference logits for multiple document-disjoint
   contexts without downloading the full BF16 checkpoint, then repeat uniform
   K3 and down refitting with clustered document-level uncertainty.
3. Add bounded output-gradient capture, build expert-local two-sided factors,
   and test whether their score predicts held-out KLD better than routed
   squared error before accepting any changed path.
4. Extend a confirmed panel across early, middle, and late mixture layers
   before building a complete candidate.
5. Freeze a GLM QSRT container and count every serialized byte before making a
   model-size comparison.

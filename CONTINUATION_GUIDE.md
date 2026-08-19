# Continuing QSRT quantization experiments

## Research objective

The Quantile-Stratified Rate-shifted Trellis codec (QSRT) stores neural-network
weights as paths through a finite-state reconstruction graph. EXL3 is a
competing variable-rate trellis quantizer and supplies the comparison
checkpoint. The active GLM-5.2 objective has two requirements:

- the serialized QSRT checkpoint must occupy fewer bytes than the immutable
  GLM-5.2 EXL3 comparison checkpoint; and
- the QSRT checkpoint must have lower teacher-to-candidate forward
  Kullback–Leibler divergence (KLD) on the same document-disjoint inputs.

KLD compares the probability distribution produced by a reference model with
the distribution produced by a quantized candidate. Lower values are better.
Weight error, tile error, expert-output error, and a local second-order loss
estimate called curvature can reject a weak idea before a model run. None of
those measurements establishes the checkpoint objective.

This repository is a source snapshot updated at `2026-08-19T04:39:57Z`. It
contains the working implementation, tests, experiment launchers, runtime
adapter source, documentation, and the complete source-controlled experiment
inputs that are small enough for Git. It does not contain model weights,
captured activations, reference logits, generated expert tensors, container
images, or runtime caches.

## Read these files first

1. [`AGENTS.md`](AGENTS.md) defines the codec contracts, repository boundaries,
   protected artifacts, and required validation.
2. [`docs/qsrt-improvement-strategy.md`](docs/qsrt-improvement-strategy.md)
   defines the quality objective, evidence ladder, exact-byte comparison, and
   decision gates for the GLM-5.2 work.
3. [`docs/glm52-layer3-kld-results.md`](docs/glm52-layer3-kld-results.md)
   records the bounded layer-3 KLD results and their evidence limits.
4. [`docs/glm52-experiment-journal.md`](docs/glm52-experiment-journal.md)
   records every remote file operation, container identity, failed launch,
   measurement, digest, and cleanup policy.
5. [`docs/qsrt-two-bit-research-corpus.md`](docs/qsrt-two-bit-research-corpus.md)
   surveys relevant trellis quantization, adaptive rounding, feedback
   quantization, signal-processing, and compression techniques with citations.
6. [`docs/qsrt-exl3-comparative-assessment.md`](docs/qsrt-exl3-comparative-assessment.md)
   compares QSRT and EXL3 mechanisms, rates, evidence, and unresolved risks.
7. [`SOURCE_SNAPSHOT_MANIFEST.json`](SOURCE_SNAPSHOT_MANIFEST.json) records the
   source provenance and SHA-256 digest of every published file that existed
   before the manifest was generated.

The two interactive explanations are
[`docs/qsrt-three-improvements-infographic.html`](docs/qsrt-three-improvements-infographic.html)
and
[`docs/viterbi-trellis-explainer.html`](docs/viterbi-trellis-explainer.html).
They are standalone HTML files and require no server.

## Source snapshot provenance

The source checkout was
`git@github.com:local-inference-lab/qsrt.git` on branch `master`. Its base
commit was:

```text
7902469f071a4f68089eaacfb59d41be0a674e41
```

The source checkout also contained eleven modified tracked files and 114
untracked research files. Those working-tree files are included in this
repository and committed together so a collaborator receives one coherent
state. The machine-readable manifest records every original Git status entry.
The publication adds this guide, a manifest generator and verifier, the
generated manifest, and its checksum. Git history, ignored output, and local
environments were not copied.

## Reproduce the CPU environment

Use Python 3.12 and create an environment from the lock file:

```bash
uv sync --dev
.venv/bin/pytest -q
```

The focused terminal-reference suite passes 34 tests. Repository-wide
collection requires the optional Triton and InstantTensor packages used by
upstream recovery code. It also exposes API mismatches in upstream recovery
tests. A run that excluded the eleven collection blockers passed 964 tests,
skipped 21, and failed one unrelated fixture whose manifest lacks the required
`candidate_codebook` field. Install the optional dependencies or reconcile
those imports before treating a repository-wide result as authoritative. Keep
the collection errors and incompatible fixture visible until their contracts
are repaired.

Verify the published source files before using them:

```bash
python3 tools/verify_source_snapshot.py
```

The verifier rejects a missing file, changed file, unexpected file, symlink,
forbidden generated-artifact suffix, or file larger than ten mebibytes. It
ignores Git metadata and the two manifest files so the manifest does not hash
itself.

## External experiment state

The four-GPU host is `kossel.lan`. The durable experiment root is:

```text
/home/sunil/qsrt-glm52-experiments/
```

The complete artifact inventory is:

```text
/home/sunil/qsrt-glm52-experiments/ARTIFACT_INDEX.json
```

The inventory generated at `2026-08-18T21:21:09Z` contains 20,620 regular-file
records covering 481,515,914,716 bytes, with zero unhashed files. The inventory
SHA-256 is
`7ca6263be32c7db3c31a6b7df2e0b8e11e1754be15cae4a24533c43ddee6718d`.
The general experiment snapshot is:

```text
/home/sunil/qsrt-glm52-experiments/source/qsrt-working-tree/
```

The terminal-reference launchers use a separately verified snapshot:

```text
/home/sunil/qsrt-glm52-experiments/source/qsrt-terminal-teacher-reference-b4734de/
```

Regenerate that snapshot manifest and synchronize the updated repository before
running the screening, freeze, or confirmation launchers in this guide.

The experiment journal records the SHA-256 identities for results that matter
to a scientific conclusion. Treat the remote artifact index and journal as
the authority for locating or removing generated data.

### Immutable model inputs

The EXL3 comparison checkpoint is already present on the GPU host at:

```text
/home/sunil/usb-mnt/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78
```

A verified copy on kossel's internal NVMe is the runtime input:

```text
/home/sunil/qsrt-glm52-experiments/model-cache/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78-nvme
```

Experiment containers mount that internal copy read-only. They do not mount
`/home/sunil/usb-mnt`, which is the external `/dev/sda` disk. Never change the
external copy in place and never download the EXL3 files again.

The first official BF16 source window is the five-shard layer-3 window at:

```text
/home/sunil/usb-mnt/zai-org/GLM-5.2-b4734de-layer003-source-window
```

The second bounded source window contains only the 17 official shards needed
for expert tensors in layers 52, 60, 63, and 64. It is stored on kossel's
internal NVMe at:

```text
/home/sunil/qsrt-glm52-experiments/source-windows/glm52-b4734de-layers-52-60-63-64
```

The 17-shard transfer is complete and verifies every immutable shard hash.
Its frozen manifest is
[`experiments/glm52_layers_52_60_63_64_source_shards.json`](experiments/glm52_layers_52_60_63_64_source_shards.json).
Its receipt records 17 shards and 91,142,336,944 bytes. The transfer used the
internal NVMe and did not consume the external disk's remaining 104 GB.
Do not download the complete GLM-5.2 BF16 checkpoint. The layer-3 window is
sufficient for the eight experts in the frozen mechanism panel. The panel is
defined by
[`experiments/glm52_layer3_rate_pattern_panel.json`](experiments/glm52_layer3_rate_pattern_panel.json).

Every container that mounts model, source, reference, or generated expert
files must use read-only mounts and `--network none`. The launchers under
[`experiments/`](experiments/) encode those restrictions.

### Container source and identities

Container images are absent from Git. The compatible GLM-5.2 R7
runtime on the GPU host is:

```text
verdictai/glm52-exl3-sparkinfer:v39-r28-r7fused-broadcast-cu132-sm120a
sha256:12f86065d7fe64d30dad678585e68c91f47f1f2a32bed45ccaf108382f3928ac
```

The generic encoder image is:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34
sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b
```

The `experiments/create_*.sh` files describe derived container changes. The
`experiments/start_*.sh` files reproduce launches. The small runtime override
in [`runtime/glm52_expert_intervention/sitecustomize.py`](runtime/glm52_expert_intervention/sitecustomize.py)
is included as source. No container layer or image archive is included.

## Established GLM-5.2 results

The scored panel changes eight routed experts in mixture layer 3 while every
other tensor remains from EXL3. `K3` and `K4` denote trellis payloads of three
and four bits per weight. One published BF16 reference contains a
2,048-token context, which yields 2,047 correlated next-token comparisons.
A second public cache supplied 16 eligible untouched documents with 512 tokens
each for an independent auxiliary replication. Neither set satisfies the
frozen 32-document terminal-hidden-state confirmation plan.

The bounded BF16 source windows name revision
`b4734de4facf877f85769a911abafc5283eab3d9`. The published reporting-logit
manifest names teacher revision
`4d67f66cc64d3219133b767c253b2ad1425c6c88`. Metadata inspection proved that
the two revisions have byte-identical safetensors indexes and identical
content SHA-256 and byte count for every official weight shard. The source
config explicitly sets `moe_router_dtype: float32`; the reporting config omits
that field, so runtime identity remains separate from weight identity. The
immutable rate-preserving experiment registration records both roles at
[`experiments/glm52_layer3_rate_preserving_down_refit_k3_k4_pre_registration.json`](experiments/glm52_layer3_rate_preserving_down_refit_k3_k4_pre_registration.json).

| Eight-expert representation | Mean forward KLD | Change from uniform K3 | Decision |
|---|---:|---:|---|
| Resident EXL3 | `0.0610743407031` | `-2.0904%` | Comparison checkpoint |
| Uniform QSRT K3 | `0.0623782807651` | — | Worse than EXL3 |
| Uniform K3 with feedback disabled at frozen scales | `0.0623782807651` | `0.0000%` | Byte-identical control |
| K3 selected with routed-input covariance | `0.0638412662800` | `+2.3453%` | Rejected |
| K3 with reconstructed-activation down refit | `0.0612386895257` | `-1.8269%` | Confirm on more documents |
| Down refit with BF16 rank-two corrections on all eight experts | `0.0652326383334` | `+4.5759%` | Rejected all-expert local selector |
| Down refit with BF16 rank-two corrections on experts 89 and 103 | `0.0601683116025` | `-3.5429%` | Exploratory; identities came from the reporting context |
| Down refit with one BF16 rank-four correction on expert 103, materialized from stored factors at load time | `0.0582574646070` | `-6.6062%` | Below 0.059 on one context; frozen for independent confirmation |
| The same expert-103 factors executed as two inference GEMMs | `0.0606608189028` | `-2.7532%` | Rejected execution path; intermediate rounding erased most of the gain |
| K3 down encoded with reconstructed-input covariance and source weights | `0.0658519849381` | `+5.5688%` | Rejected |
| K3 with a locally selected identity-metric down refit | `0.0641342908893` | `+2.8151%` | Rejected selection rule |
| K3 with reconstructed-input covariance and locally selected down refits | `0.0638195014718` | `+2.3105%` | Rejected |
| Fixed twelve-promotion mixed K3/K4 over the down-refit base | `0.0659634015775` | `+5.7474%` | Rejected |
| Ten-promotion K3/K4 control selected by complete-expert error, using one down target per expert | `0.0636258118201` | `+1.9999%` | Rejected construction |
| Fixed twelve-promotion K3/K4 control using one down target per expert | `0.0639669166209` | `+2.5468%` | Rejected construction |

The down refit trains the down-projection target against activations rebuilt
from quantized gate and up projections. It recovered 87.3960 percent of
uniform K3's excess mean KLD above EXL3, but remained 0.2691 percent worse than
EXL3. It improved 1,004 positions and regressed 1,043 positions. Confirmation
must therefore report mean KLD and a tail measure such as the ninety-ninth
percentile or the mean loss among the worst-scoring positions, known as
conditional value at risk.

The fixed mixed-rate candidate copied twelve promotion priorities from EXL3.
It was 8.0051 percent worse than EXL3 and 7.7152 percent worse than the
down-refitted K3 base. Promoted down projections used ordinary source-target
K4 tensors and therefore replaced their K3 refits. This result rejects the
fixed allocation and target combination. It does not test K4 encoding of the
refitted down target or an allocation selected by complete-expert output error.

The later rate pool preserved a refitted down target when the down projection
changed rate. Its selection-data allocation used ten K4 projections, occupied
129,372,160 logical bytes, and was 4.1776 percent worse than EXL3. Its matched
fixed control used twelve K4 projections and was 4.7362 percent worse than
EXL3. Both lowered p99 while worsening CVaR1%, so p99 alone would have accepted
the wrong tail direction. Both reused one target fitted from K3/K3 upstream
activations when gate or up changed rate. They therefore do not test coherent
rate-conditioned refitting.

The down-construction comparison separated the input metric from the
continuous target. Reconstructed-input covariance with the source target
reduced complete-expert error by 48.7027 percent on candidate-selection rows,
then worsened model mean KLD by 5.5688 percent relative to uniform K3. Adding
three locally accepted refits improved that covariance policy but still lost
to uniform K3. A second local rule accepted eight identity-metric refits yet
also lost to uniform K3. Five of those materialized down tensors matched the
earlier positive refit; three locally preferred ridge or fallback changes
reversed the model-level outcome. Reject reconstructed-input covariance for
this panel and reject local complete-expert mean plus local row-tail loss as a
refit selector.

The activation-weighted low-rank experiment fitted a small additive correction
to each down-refit residual. Candidate-specific quantized gate and up outputs
supplied the down inputs. Rank two reduced pooled candidate-selection
complete-expert error by 67.6456 percent, yet applying it to all eight experts
made model KLD 6.8086 percent worse than EXL3. Local error did not select a
safe model intervention.

Individual model-KLD attribution identified three helpful rank-two experts,
but their combined effects were not additive. A rank-four correction on expert
103 reached mean KLD `0.0582574646070`. Its p99, CVaR1%, and maximum KLD also
improved relative to EXL3 on the available context. The rank and expert were
chosen after inspecting arms on that same context, so this is a mechanism
screen. It is not document-replicated qualification.

The exact candidate is frozen in
[`experiments/glm52_layer3_rank4_expert103_low_rank_down_confirmation_registration.json`](experiments/glm52_layer3_rank4_expert103_low_rank_down_confirmation_registration.json).
The registration binds both factor hashes, BF16 storage, rank four, expert
103, ridge `0.001`, the base representation, the materialized endpoint, and
the confirmation rule. Changing a bound field creates a different candidate
and requires a separate sealed confirmation set.

The runtime qualification is frozen separately in
[`experiments/glm52_layer3_rank4_expert103_low_rank_down_runtime_qualification.json`](experiments/glm52_layer3_rank4_expert103_low_rank_down_runtime_qualification.json).
Executing the correction as two BF16 factor GEMMs reached KLD
`0.0606608189028`. The local output differed from the dense endpoint by only
about `2e-7` relative squared error, but later propagation removed most of the
KLD gain. The accepted runtime reconstructs the FP16 down matrix once from the
stored K3 base and factors. Its 2,047 KLD values and complete route arrays were
bit-identical to the dense screen, restoring KLD `0.0582574646070` without an
additional inference GEMM.

The frozen correction was then measured without modification on 16 public
documents that did not participate in fitting, selection, or the original
screen. Resident EXL3 equal-document mean KLD was `0.092796188456`; the
candidate measured `0.092635107672`, a 0.1736 percent paired reduction. Nine
documents improved and seven regressed. The 20,000-resample document-bootstrap
95 percent interval for `candidate minus resident` was
`[-0.003332554163, 0.002636722739]`. Candidate p99, CVaR1%, and maximum also
improved, but the interval crossed zero. Record this as inconclusive auxiliary
evidence. It does not establish a replicated gain. The compact receipt is
[`experiments/glm52_layer3_rank4_expert103_public_reference_auxiliary_result.json`](experiments/glm52_layer3_rank4_expert103_public_reference_auxiliary_result.json).

The same public documents tested error-blind panels from layers 52, 60, 63,
and 64. Complete down-refit panels from layers 52, 60, and 64 had adverse mean
KLD point estimates. The layer-64 regression had a confidence interval above
zero. Layer 63 produced a favorable but inconclusive point estimate: uniform
K3 measured `0.092774344239`, and down refitting measured `0.092735445026`,
compared with resident EXL3 at `0.092796188456`.

Adding locally selected rank-four corrections produced worse mean KLD than
resident EXL3 at all four high layers. Layer-60 and layer-64 confidence
intervals were above zero. Layer-63 expert 164 improved local expert-output
error by 33.5 percent and worsened mean KLD. Layer-64 expert 253 had a small
favorable KLD point estimate whose confidence interval included zero. The
local activation-weighted objective can propose candidates, but it cannot
authorize an expert correction.

Additional findings constrain the next experiments:

- Finite-E4M3 reconstruction-table fitting reduced pooled fixed-path squared
  error by 0.00175 percent. The measured headroom does not justify a production
  table-training loop.
- Disabling the matrix-aware error feedback method called BlockLDLQ at frozen
  K3 scales changed the values presented to the dynamic-programming path
  solver called Viterbi search. It changed none of the 24 trellis payloads.
  All eight reconstructed experts were byte-identical to uniform K3.
- A complete `2,048 by 6,144` two-sided-curvature CUDA closure changed the
  path, held scales fixed, and used about 1.57 GiB of peak GPU memory. Identity
  curvature in the source output basis changed source-space relative squared
  error by only 0.000039 percent. The output transform and persisted output
  scales map that source-basis identity through a congruence transform, so it
  is not the ordinary encoder's zero-output-feedback metric. No real
  output-gradient factor or two-sided KLD result exists.

## Exact-byte comparison for the eight-expert panel

The EXL3 panel occupies 133,791,744 logical bytes. Uniform QSRT K3 occupies
113,643,520 logical bytes. Promoting one of the 24 equal-size projection
matrices from K3 to K4 adds 1,572,864 bytes.

Twelve K4 promotions produce 132,517,888 logical bytes, which is 1,273,856
bytes smaller than EXL3. Thirteen promotions produce 134,090,752 logical
bytes, which is 299,008 bytes larger than EXL3. The strict smaller-than-EXL3
panel therefore permits at most twelve K4 promotions.

This ledger counts trellis payloads, scale storage, and the shared QSRT table.
The GLM-5.2 QSRT container format does not exist yet, so headers, alignment,
padding, and serving-directory overhead remain unmeasured. A complete
serialized artifact must retain a positive size margin after those costs.

The frozen expert-103 correction adds 65,536 logical BF16 factor bytes to the
113,643,520-byte K3/down-refit panel. The candidate panel therefore occupies
113,709,056 logical bytes, 20,082,688 below EXL3. Factor headers, scales,
alignment, directories, and runtime metadata remain uncharged until a
factor-aware GLM container exists.

## Active GLM-5.2 work

### Generate document-disjoint BF16 reference logits from terminal hidden states

The canonical public Hessian archive contains the official unquantized BF16
output of decoder layer 77 for 1,049,589 captured tokens. The official source
model computes logits by applying its final root-mean-square normalization and
untied language-model head to those rows. Reference generation therefore needs
selected terminal rows and the two endpoint tensors. It does not need the
complete BF16 checkpoint.

The frozen plan is
[`experiments/glm52_terminal_hidden_teacher_reference_plan.json`](experiments/glm52_terminal_hidden_teacher_reference_plan.json).
Its SHA-256 is
`b73690cb3507e64b51c45312d7817ccb1d9d8a0372d05ee3b68c28a3ff1e9519`.
It reserves eight holdout documents for candidate screening and 32 different
holdout documents for confirmation. The screening tier supplies 9,334
next-token comparisons. The confirmation tier supplies 65,482 comparisons;
31 documents have 2,048 tokens and one has 2,026 tokens.

The confirmation tier maximizes the number of paired positions available from
the canonical holdout pool. It contains 22 general documents, one legal
document, nine code or agentic documents, and no reasoning documents. A result
on this set establishes document replication for the canonical capture
distribution. It cannot establish broad legal or reasoning quality.

The range downloader transfers 2,857,330,353 bytes: selected terminal rows,
the official 1.90 GB language-model head, the 12 KB final-normalization vector,
and provenance metadata. It excludes every decoder-layer weight and all
unselected terminal rows. Run:

```bash
bash experiments/download_glm52_terminal_teacher_assets_on_kossel.sh
```

After the download receipt closes, generate only the screening references:

```bash
bash experiments/generate_glm52_screening_teacher_logits_on_kossel.sh
```

The generator uses the existing pinned container image with network access
disabled. It vocabulary-shards the endpoint across four GPUs, checks captured
token hashes, repeats the BF16 calculation bit for bit, and records a 32-bit
floating-point endpoint comparison. The screening outputs occupy about 2.89
GB. The confirmation outputs occupy about 20.28 GB. Their generator refuses to
run until a freeze record binds the candidate construction, factor dtype,
serialized correction bytes, and screening result.

### Confirm the frozen rank-four expert-103 correction

The registered candidate must first pass the eight-document screen:

```bash
bash experiments/evaluate_glm52_layer3_expert103_rank4_on_terminal_screening_references.sh
```

The screen passes only when the candidate lowers the equal-document mean,
improves at least six documents, and does not increase pooled position CVaR1%.
Six or more same-sign outcomes among eight independent symmetric outcomes
occur in 37 out of 256 cases, or 14.453125 percent. A screening pass therefore
authorizes confirmation access but does not establish generalization.

Freeze the registered factors, runtime mode, screening report, and every byte
of the standalone BF16 factor payload:

```bash
bash experiments/freeze_glm52_layer3_expert103_rank4_after_terminal_screening.sh
```

The freeze command fails if the screen did not pass. It also records the
confirmation decision before producing any confirmation logits. Generate and
score the 32-document tier with these explicit commands:

```bash
bash experiments/generate_glm52_confirmation_teacher_logits_on_kossel.sh
bash experiments/evaluate_glm52_layer3_expert103_rank4_on_terminal_confirmation_references.sh
```

The confirmation decision requires the one-sided 95 percent document-bootstrap
upper bound for candidate-minus-EXL3 mean KLD to fall below zero. It also
requires candidate pooled position CVaR1% to remain at or below the resident
value. The zero CVaR1% tolerance is part of the pre-access freeze record.

Do not use the absolute `0.059` value across reference suites. On the public
512-token suite, resident EXL3 itself measured `0.092796188456`. The decisive
quantity is the paired candidate-minus-resident document difference under one
frozen runtime and reference contract.

The stored-factor runtime now reconstructs the screened FP16 endpoint bit for
bit when the expert is loaded. It retains the existing one-GEMM expert path.
Complete the factor-aware container layout and exact byte ledger while the
reference contexts are generated. The freeze charges every byte in the
standalone factor file. The serialized panel and complete checkpoint must
remain smaller than EXL3 after headers, scales, alignment, and directories are
charged.

### Use model KLD to select individual recovery experts

The complete high-layer panels reject blanket down refitting. Their local
selection rule also failed to predict model KLD. The frozen plans under
[`experiments/`](experiments/) cover all eight panel experts at each of layers
52, 60, 63, and 64. They use the first eight public documents for screening
and the remaining eight documents for a separate selection check.

The selector reads documents in the top-level reference-plan execution order;
the hash-sorted nested summary cannot define the two groups. A candidate is
retained only when candidate-minus-resident mean KLD is negative in both
groups. This rule retained layer-52 down-refit expert 36 and layer-52
rank-four expert 186. Expert 186 reduced the all-document point estimate by
0.3679%, although its document-bootstrap interval included zero. No measured
rank-four singleton from layers 60 or 64 passed both groups. The separately
registered 2,048-token screen then rejected expert 186: mean KLD rose from
`0.06107434` to `0.06151378` under bitwise-stable controls. Down-refit expert
36 improved the same context to `0.06090743`. The result is favorable
development evidence but remains above the `0.059` target and lacks
document-level replication.

The registered same-layer composition copied the two retained endpoints and
charged 65,536 logical adapter bytes. It improved the overall 16-document mean
by `0.00006838`, but the first ordered group regressed by `0.00029425`. The
second group improved by `0.00043102`. The composition failed its registered
two-group rule, so the separate 2,048-token development reference remained
unopened. Complete-expert error may reject a candidate that another candidate
dominates under every recorded local criterion, but it cannot authorize a
recovery expert.

### Screen the late-middle GLM layers without downloading the complete source

Four rate-pattern-stratified expert panels are frozen at model layers 55, 56,
57, and 58. The expert lists use immutable EXL3 rate metadata and contain no
QSRT error or KLD input. Sixteen official BF16 shards, totaling
85,783,011,360 bytes, contain every routed-expert tensor required by those
layers. The download manifest excludes the remaining 266 source shards.

After the bounded download verifies every shard, one network-disabled EXL3
load captures the same arm-invariant expert-input documents at all four
layers. The queued candidate builder then performs these operations for each
layer:

1. build the eight-expert uniform QSRT K3 artifact;
2. refit each down projection against that candidate's reconstructed gate and
   up activations;
3. fit a BF16 rank-four down correction against the refitted candidate's own
   down inputs; and
4. measure every singleton and the predeclared complete panel on the 16 public
   candidate-selection documents.

Each construction receives one resident-model load. The selector retains an
arm only when candidate-minus-resident mean KLD is negative in both ordered
groups of eight documents. Any cross-layer composition must recapture and
refit downstream corrections with the selected upstream interventions active.
Copying independently fitted endpoints would violate the candidate-native
input contract.

### Build coherent rate-conditioned down candidates

The measured one-target control implementation is in
[`qsrt/glm52_down_refit_rate_pool.py`](qsrt/glm52_down_refit_rate_pool.py).
It is retained as a negative control. The next builder must reconstruct the
down input and fit an independent down target for each of the K3/K3, K3/K4,
K4/K3, and K4/K4 gate/up pairs. Each target then receives K3 and K4 encodes.
The result is eight internally consistent complete-expert candidates. Build
the pool while reference logits are being prepared, but do not freeze a panel
configuration until the KLD-selected down rule is available. Use
complete-expert error only to shortlist candidates, then use KLD on the eight
selection contexts to freeze one panel configuration.

### Confirm down refitting on multiple documents

The K3 down refit reduced mean KLD relative to uniform K3 but remained 0.2691
percent worse than EXL3 on one context. Obtain additional reference logits
without downloading the complete BF16 checkpoint. Report paired per-document
differences and tail loss before extending the method to more experts or
layers.

### Use the complete-expert inversion to place output curvature

The one-sided routed-input candidate reduced complete-expert routed output
squared error by 93.6575 percent and still worsened full-model KLD. The missing
sensitivity therefore lies downstream of the complete expert on the measured
context. Capture gradients at the expert-output or residual-stream boundary
before building per-matrix output factors.

### Capture gradients at the residual-stream boundary

The useful downstream signal is the gradient of model loss with respect to an
expert's contribution where it is added back to the model's hidden state,
called the residual-stream boundary. That gradient is computed once per token
and layer and is shared by every routed expert at that token. Capture it before
building per-expert or per-matrix output factors.

For sampled real candidate error matrices, compare two predictions:

- the per-sample curvature sum using paired input vectors and output gradients;
  and
- the Kronecker approximation, which replaces the joint input-gradient
  statistic with separate input and output second moments.

Routing correlates inputs and gradients because the same router coefficient
scales both. Reject the factorized selector when its approximation error is as
large as the score differences that determine candidate order. Test real K3
candidate error magnitudes. Small synthetic perturbations do not test whether
the quadratic approximation ranks production candidates.

Gate and up errors interact through multiplication and GLM-5.2's
coordinatewise sigmoid linear unit (SiLU). Use two-sided scores to generate or
rank candidates within one matrix. Final acceptance must reconstruct and
propagate the complete gate, up, and down expert. Include low-support experts
and shrink noisy output factors toward an explicit fallback.

### Add document-level KLD evidence

The repository contains one published 2,048-token BF16-logit context and a
16-document, 512-token selection set. The terminal-hidden-state plan adds a
larger source-derived reference route without obtaining the complete BF16
checkpoint. Reference-logit files contain model outputs and remain outside
Git.

Report paired per-document KLD differences, a clustered bootstrap interval,
repeatability controls, exact artifact hashes, route equality, and tail
statistics. Pre-register a confirmation for the twelve selected promotions to
avoid winner's-curse bias from choosing the highest estimated gains.

## Publication contents and exclusions

The source snapshot includes:

- Python codec, adapter, capture, scoring, allocation, and validation modules;
- C++ and CUDA encoder source;
- unit and integration tests;
- local and remote experiment launchers;
- reproducible container-derivation scripts and runtime overrides;
- frozen small JSON experiment inputs;
- technical specifications, research reviews, citations, result summaries,
  experiment history, and interactive HTML explanations;
- `pyproject.toml`, `uv.lock`, `.gitignore`, and `AGENTS.md`; and
- source provenance, per-file SHA-256 records, and the snapshot verifier.

The source snapshot excludes:

- Git history from the source checkout;
- `.venv`, Python bytecode, pytest and Ruff caches, and generated `out/` data;
- model checkpoints and weight-bearing formats, including Safetensors, PyTorch
  tensors, GGUF files, and QSRT capture or Hessian bundles;
- reference logits, captured activations, generated expert tensors, traces,
  result directories, and remote artifact caches;
- Docker and OCI layers, image archives, mounted runtime filesystems, and
  package caches;
- credentials, authentication state, SSH material, environment-variable
  files, and other secrets; and
- any regular file larger than ten mebibytes.

The generated manifest is the complete file-level authority for this Git
snapshot. The remote artifact index is the complete file-level authority for
the experiment host.

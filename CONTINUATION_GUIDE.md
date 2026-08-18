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

This repository is a source snapshot updated at `2026-08-18T18:50:45Z`. It
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

The source checkout also contained eleven modified tracked files and 76
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

The GLM down-construction, down-refit rate-pool, paired-KLD,
intervention-runtime, and reconstructed-activation refit tests pass. The local
CPU-only environment completed 52 focused tests in 0.54 seconds.
Repository-wide collection requires the optional Triton and InstantTensor
packages used by upstream recovery code. It also exposes API mismatches in
upstream recovery tests. A run that excluded those collection blockers passed
849 tests, skipped 21, and found one unrelated artifact-fixture failure because
the fixture lacked `candidate_codebook`.
Install the optional dependencies or reconcile those imports before treating
a repository-wide result as authoritative; do not hide the collection errors
by deleting tests.

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

The inventory generated at `2026-08-18T18:44Z` contains 16,941 regular-file
records covering 409,590,032,232 bytes. It also records 636 files that the
indexing process could not hash. Do not treat those unhashed files as verified.
The inventory SHA-256 is
`cfff53e5113f89c06e4e3a3f5951a3085190c2145af4620688b65f719cd382e1`.
The synchronized source copy is:

```text
/home/sunil/qsrt-glm52-experiments/source/qsrt-working-tree/
```

The experiment journal records the SHA-256 identities for results that matter
to a scientific conclusion. Treat the remote artifact index and journal as
the authority for locating or removing generated data.

### Immutable model inputs

The EXL3 comparison checkpoint is already present on the GPU host at:

```text
/home/sunil/usb-mnt/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78
```

A verified host-local copy exists below the experiment root. Use the exact path
recorded in the journal. Never change the USB copy in place and never download
the EXL3 files again.

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

The 17-shard transfer is resumable and verifies every immutable shard hash.
Its frozen manifest is
[`experiments/glm52_layers_52_60_63_64_source_shards.json`](experiments/glm52_layers_52_60_63_64_source_shards.json).
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
and four bits per weight. The published BF16 reference contains one
2,048-token context, which yields 2,047 correlated next-token comparisons.
These results can screen mechanisms, but they cannot establish document-level
generalization.

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

Three other findings constrain the next experiments:

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

## Active GLM-5.2 work

### Obtain document-disjoint BF16 reference logits

The available reference-logit set contains one 2,048-token document. It can
reject a large regression but cannot select a refit rule or qualify a model.
An authorized host that can run the official BF16 teacher must generate eight
selection contexts and at least 32 sealed confirmation contexts. The context
documents, tokenizer, chat template, teacher revision, runtime configuration,
and output hashes must be recorded. The documents must be disjoint from fit
and candidate-construction data. Do not download the complete BF16 checkpoint
to kossel or to a developer workstation.

### Select the down-refit rule with model KLD

Retain the earlier fixed down-refit artifact as a candidate because it is the
only tested QSRT intervention that improved mean KLD relative to uniform K3.
Do not adopt it from the one-context result. Use complete-expert error only to
prune clearly dominated candidates. Choose ridge and source-fallback decisions
with measured KLD on the eight selection contexts, then freeze the rule before
opening the confirmation contexts.

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

The repository contains one published BF16-logit context and must not obtain
the full BF16 checkpoint. Obtain additional reference logits or arrange for an
external owner of the official model to generate them from a frozen,
document-disjoint corpus plan. Reference-logit files contain model outputs,
not weights, and remain outside Git.

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

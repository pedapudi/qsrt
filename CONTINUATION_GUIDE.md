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

This repository is a source snapshot updated at `2026-08-17T06:52:00Z`. It
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
453b4834332d2735c5a326ca57fb6a8b36e776bf
```

The base commit and its upstream branch were synchronized when the snapshot
was taken. The source checkout also contained eleven modified tracked files
and 78 untracked research files. Those working-tree files are included in this
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

The exported source passed 689 tests with four skips in the local CPU-only
environment. The complete real-matrix zero-output-feedback control also passed
bit equivalence before the mixed-rate experiment. Model-level KLD still needs
the remote artifacts described below.

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

That inventory contains 19,048 file records, zero unhashed files, a semantic
category for each file, and a cleanup policy. The synchronized source copy is:

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

The only official BF16 weight source allowed by this experiment is the five-
shard layer-3 window at:

```text
/home/sunil/usb-mnt/zai-org/GLM-5.2-b4734de-layer003-source-window
```

Do not download the complete GLM-5.2 BF16 checkpoint. The bounded window is
sufficient for the eight layer-3 experts in the frozen panel. The panel is
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

| Eight-expert representation | Mean forward KLD | Change from uniform K3 | Decision |
|---|---:|---:|---|
| Resident EXL3 | `0.0610743407031` | `-2.0904%` | Comparison checkpoint |
| Uniform QSRT K3 | `0.0623782807651` | — | Worse than EXL3 |
| Uniform K3 with feedback disabled at frozen scales | `0.0623782807651` | `0.0000%` | Byte-identical control |
| K3 selected with routed-input covariance | `0.0638412662800` | `+2.3453%` | Rejected |
| K3 with reconstructed-activation down refit | `0.0612386895257` | `-1.8269%` | Confirm on more documents |
| Fixed twelve-promotion mixed K3/K4 over the down-refit base | `0.0659634015775` | `+5.7474%` | Rejected |

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

## Work that should proceed when the GPU host returns

### Preserve the down-refit target across rate changes

The completed fixed mixed-rate artifact used source-target K4 tensors for
promoted projections. A promoted down cell therefore replaced its accepted K3
refit. Recompute each accepted continuous down target from the captured fit
rows and stored ridge factor, then encode that same target at both K3 and K4.
Keep the continuous matrix outside the checkpoint after both encodes finish.

Measure one-projection changes before allocating the twelve-promotion budget.
The required arms are upstream-only K4, source-target down K4, and refitted-
target down K4. Score complete experts on the frozen candidate-selection
documents. The published reporting context may report a frozen allocation but
must not choose rates.

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

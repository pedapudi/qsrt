# Continuous-parameter recovery tuning for the QSRT-K2 Kimi-K3 checkpoint

Status: **research implementation**. Packed-versus-decoded numerical parity,
router-frequency capture and feedback updates, selective suffix-archive
construction, archive-independent distributed-autograd parity, and the
real-model zero-step gates are implemented and qualified. The all-layer router
bias view is frozen after same-runtime screening qualification. The
4,000,000-token training archive is complete. Shared-expert and normalization
training is research-only: its minimum decoded-BF16 archive-replay result
improves the frozen zero-step result, while distribution-fidelity qualification
remains pending. Every component leaves
the K2 payload bytes, reconstruction law, and decoder untouched. Components
1-4 also leave the serving graph unchanged and deliver only retuned BF16
non-expert tensors as overlays. Component 5 (per-expert low-rank adapters,
section 9) additionally ships new BF16 adapter tensors and adds one additive
branch to expert execution — a serving-format commitment that is explicitly
gated on measured equal-byte results.

## 1. Objective

Let the official Kimi-K3 checkpoint (`moonshotai/Kimi-K3` @
`9f62e4e9fffbd0a83ddd60e1c209d828994b3569`: BF16 non-expert tensors, MXFP4
routed experts) define teacher token distributions `p_t`. Let the student be a
model whose routed experts are the frozen K2 QSRT payloads of the
direct-Viterbi anchor and whose non-expert tensors are BF16 and trainable. The
training objective is the token-mean forward KL at the final logits:

```
J(theta) = (1/T) * sum_t KL(p_t || q_t(theta))
```

where `theta` ranges only over the trainable BF16 tensors. The expert payloads
are constants of the computation.

The baseline to improve on: the direct-Viterbi K2 anchor
(`/data/models/Kimi-K3-QSRT-K2-DIRECT-VITERBI-ALL-LINEARS-v1-model`) measures
mean KL `0.062993208155` and top-1 agreement `93.571003%` on the
distribution-fidelity suite of 768 contexts and 1,572,096 scored positions
(suite manifest SHA-256
`f3a79f7f28365d406a19a82cf210c25adf18974c4b9b607ab3754e9939f941cf`). All
reported results of this effort must use that same suite identity
(`docs/qsrt-fisher-experiment-ledger.md` records the comparable entries).

### Distribution-fidelity evaluation contract

All model-level KLD screening, candidate selection, and qualification use
`/data/datasets/kld/kimi-k3-distribution-fidelity-1024x2048-v1`. Research
decisions use its 768-context analysis partition with 1,572,096 scored
positions. Final qualification uses its disjoint 256-context qualification
partition with 524,032 scored positions after parameters and acceptance rules
are frozen.

The retired 32-context, 65,504-position WikiText/HumanEval/Dolly suite is not
an admissible KLD screen. Its small sample, narrower source coverage, and
runtime variation are too large for the effect sizes targeted here. Existing
archive-replay measurements over those token IDs are historical optimizer
diagnostics only; they cannot select an overlay, qualify a checkpoint, or
support a model-quality claim.

## 2. Why this approach

Two measured facts motivate optimizing the continuous tensors rather than the
expert payloads:

1. **The payload lattice has a tiny usable trust radius.** Gradient-guided
   re-encoding of the K2 payloads (see
   `docs/final-kl-gradient-viterbi-refinement.md`) only tolerated target
   shifts of `alpha = 1/128` of the quantization-error norm before regressing,
   and the measured gain at that step was not statistically resolved. A
   continuous parameter has no lattice: a gradient step is applied exactly,
   line search works, and many steps compound.
2. **Encoder-side conditioning consistently underperformed plain
   reconstruction.** Every curvature- or gradient-weighted encoder arm in
   `docs/qsrt-fisher-experiment-ledger.md` delivered at most +0.3% relative KL
   or regressed against the unconditioned direct-Viterbi control.

Architectural leverage of the trainable set:

- **Shared experts** (two per MoE layer, `num_shared_experts = 2`) run in
  parallel with the routed experts — same input, output summed at the same
  point — so they can learn the input-conditioned mean of the routed experts'
  K2 quantization error. They are the highest-leverage compensation target
  per parameter.
- **Per-expert low-rank adapters** (component 5) parameterize per-expert
  corrections continuously. A shared expert can only learn the mixture-averaged
  correction; an adapter on each expert's own matrices learns that expert's
  error structure, still with no lattice constraint.
- **RMSNorm gains** rescale the residual stream per channel; classical
  quantization compensation.
- **Attention and latent projections** provide additional capacity if the
  cheaper sets plateau.
- **The router correction bias** is the only handle on the routing channel:
  quantized experts perturb hidden states, which flips downstream top-16
  selections; selection is non-differentiable, so no VJP-based method (Fisher,
  KL gradients) can see this damage. The bias is adjusted by a forward-only
  feedback rule (section 8), not by gradient descent.

Why the experts stay frozen: an SGD update to an expert weight leaves the
legal K2 payload set, and projecting it back is the re-encoding problem with
the 1/128 trust radius above. Independently, expert weight gradients at this
scale are infeasible to hold (2.7227e12 parameters: 10.9 TB of FP32 gradients
per step, ~22 TB of Adam state) and have no consumer. Backpropagation does not
require them: at each frozen linear `y = Wx`, backward needs only the input
gradient `W^T delta` to continue the chain to trainable parameters upstream;
the weight gradient `delta x^T` is skipped entirely. The router still receives
its gradient through the differentiable path: the layer output is
`sum_i g_i * E_i(x)` over selected experts, so `dL/dg_i = <delta, E_i(x)>`
flows to the gate weight with no expert weight gradient involved.

## 3. Model geometry and scale facts

From `qsrt/constants.py` (verified against the checkpoint):

| Quantity | Value |
| --- | ---: |
| Decoder layers | 93 (layer 0 dense, layers 1-92 MoE) |
| Routed experts per MoE layer | 896, top-16 routing |
| Hidden dimension | 7168 |
| Expert latent dimension | 3584 |
| Expert intermediate dimension | 3072 |
| Params per expert (w1+w3+w2) | 33,030,144 |
| Expert params per layer | 29.6e9 |
| Expert params total | 2.7227e12 |

Derived sizes the design depends on:

| Quantity | Value |
| --- | ---: |
| One layer's experts, decoded BF16 | 59.2 GB |
| One layer's experts, packed K2 (~2.06 bpw) | ~7.6 GB |
| All expert payloads, packed | ~700 GB (~58 GB/GPU sharded 12 ways) |
| Teacher checkpoint (MXFP4 experts + BF16 rest) | ~1.5 TB |
| Final-boundary hidden state per token (BF16) | 14,336 B |
| Expert rows per batch | `tokens * 16 / 896 = tokens / 56` |

Hardware: 12x RTX PRO 6000 Blackwell, 96 GB each (95.59 GiB usable), one
node; a 30 TB NVMe RAID sustaining ~100 GB/s. The filesystem currently has
6.22 TiB available. The 4.9 TB routed-row capture described in section 9
already occupies this filesystem.

Measured throughput baselines from existing infrastructure:

- Full-model streaming forward with boundary-slab writes: 100,000 tokens in
  1,158 s (~86 tok/s) — recorded in the boundary archive's `forward-run.json`.
  This is the number the waterfall capture (section 5) must beat by orders of
  magnitude.
- Forward+backward through a 9-layer resident residual segment (layer per
  GPU, queue-chained documents): 100,000 tokens in ~26-29 s (~3,500 tok/s) —
  recorded in the dense-gradient archive's `dense-gradient-run.json`. This is
  the throughput basis for the suffix trainer (section 6).

## 4. Frozen-expert execution contract

The forward adapter already implements the required semantics:
`QSRTKimiForwardAdapter.load_layer` (`qsrt/kimi_quantized_forward.py`) decodes
every expert payload through the exact reconstruction law and casts to BF16 at
load time, then executes plain BF16 GEMMs through the grouped expert dispatch.
Two execution modes cover all components:

- **Resident decode** (exists today): decode a layer's experts to BF16 once at
  load (59.2 GB/layer), keep resident, use ordinary autograd. Valid whenever a
  GPU holds at most one MoE layer. The decode is deterministic — every decode
  of the same payload is bit-identical — so recompute-for-checkpointing,
  forward, and backward all see the same weights. No straight-through
  estimator exists anywhere: no gradient flows into the decode.
- **Fused ephemeral decode** (to build only if the full-depth trainer with
  pipeline parallelism is green-lit, section 7): experts stay packed in HBM;
  the GEMM kernel decodes 16x16 trellis tiles in registers/shared memory as it
  streams. At training row counts (an 8k-token microbatch gives 143 rows per
  expert) the GEMM is weight-traffic-bound, and reading 8.3 MB packed beats
  reading 66 MB BF16 — below the arithmetic-intensity crossover the fused
  kernel is faster, not merely smaller. The backward direction needs a
  transpose variant (`W^T delta`); tile-local decode makes transposition a
  within-tile detail. The reverse-replay pipeline already backpropagates
  through quantized layers, so the semantics have precedent.

**Serving-numerics fidelity:** the production packed W4A16 kernel and the
independent decoded-BF16 replay were compared on 48 real routed rows spanning
layers 12, 24, and 84 and TP extents 0, 5, and 11. The comparison covered
10,752 output tiles. Full-output relative L2 error was 0.00267%, 0.03195%, and
0.03610% for the three cases. Across output tiles, relative L2 error had a
0.04130% median, 0.26355% p99, and 0.43739% maximum; max-absolute error had a
1.026e-6 median, 2.641e-5 p99, and 4.055e-5 maximum. These differences are
fused-arithmetic rounding rather than a reconstruction mismatch. Recovery
training therefore uses decoded-BF16 expert weights without emulating the
packed kernel. The complete result is
`/data/kquant/research/qsrt-continuous-recovery-m0/packed-parity-summary.json`.

## 5. Waterfall capture (component 1)

Purpose: produce the two precomputed artifacts training consumes, at
thousands of tokens per second instead of 86.

**Design: weight-stationary megabatch inference.** Load layer `l`'s weights
once, apply them to the entire megabatch (~1M tokens), write the boundary
activations to the RAID, advance to layer `l+1`. One traversal of the
checkpoint (~1.5 TB, ~15 s of I/O at 100 GB/s) amortizes over the megabatch;
compute becomes the limit. Boundary ping-pong needs ~14 GB per boundary per
1M tokens. This is a batching restructure of the existing capture scripts
(`scripts/capture_kimi_k3_boundary_slabs.py` lineage), not new math.
Per-document state (attention over KDA, positions) must be handled exactly as
the existing capture does: process documents as units inside the megabatch,
carrying the same document-extent bookkeeping (`documents.json`).

The authenticated 4,000,000-token teacher and student router-frequency passes
are the scheduling benchmark for this execution design. Their reports record
total wall throughput, embedding-load time, and per-layer weight-load and
compute-lane seconds. Do not extrapolate from the 4,096-token correctness
witness, whose 176-second wall time is dominated by fixed setup and layer-load
costs. Schedule the 50,000,000-token archive only after the 4M measurements
separate weight I/O from layer execution and establish the amortized token
rate on the complete 12-GPU pipeline.

**Artifacts produced:**

1. **Teacher normalized LM-head inputs** for the training corpus and the
   disjoint screening partition, BF16, 14,336 B/token. The capture derives
   each target from the teacher's final decoder boundary, all teacher residual
   prefixes required by the residual mixer, and the frozen teacher final
   RMSNorm. Teacher probabilities are *not* stored; a frozen teacher `lm_head`
   reconstructs them during training. A raw final decoder boundary alone is
   insufficient because it omits the residual prefixes needed to reproduce
   the normalized teacher state. This keeps the student's final norm and
   `lm_head` trainable while targets stay fixed and costs one teacher-head GEMM
   per batch.
2. **Student boundary slabs, selective:** both the training corpus and the
   disjoint screening partition include boundary-84 slabs and matching
   teacher normalized LM-head targets. The periodic suffix-replay evaluator reads
   the screening artifacts directly; a training-only archive is invalid.
   Capturing all 93 boundaries at 50M tokens (~65 TB) is unnecessary. Exact
   replay from layer 84 nevertheless requires more than hidden boundary 84.
   Kimi carries the residual-prefix tensors from boundaries 0, 12, ..., 72;
   `KimiBoundarySlabArchive.reconstruct_block_residual(layer=84)` reads all
   seven of them. The student input therefore occupies eight slabs: hidden
   boundary 84 plus seven residual boundaries. The teacher normalized LM-head
   target adds a ninth slab. At 50M tokens this exact archive is 6.4512 TB
   (5.867 TiB); at 4M tokens it is 516.096 GB (0.469 TiB). These residual slabs
   are execution state, not optional diagnostic boundaries, and cannot be
   omitted without changing suffix replay semantics. Reuse
   `KimiBoundarySlabArchive` formats and read paths.

**Cut-point constraint:** training cut points must lie on residual-segment
boundaries. The architecture carries residual state across
`attn_res_block_size` layers; the final residual segment is layers 84-92 (the
reverse-replay run records `first_layer=84, end_layer=93`). A mid-segment cut
would orphan intra-segment residual state.

**Training populations:** the initial suffix-training run uses the authenticated
4,000,000-token corpus identified by
`/data/datasets/kquant/captures/k3-all-routed-4m-v1-corpus.json`. Its complete
documents are disjoint from both distribution-fidelity suites. Four million
tokens with dense teacher-distribution targets are sufficient to test the
shared-experts-and-norms optimizer and loss path; multiple epochs are allowed.

A broader approximately 50,000,000-token corpus is assembled and captured in
parallel. The best configuration selected by the 4M run is re-evaluated on the
50M population before any full-depth trainer or adapter serving-format
decision. Both populations are tokenized once and stored with document
extents. Their manifests record teacher revision, anchor model identity,
tokenizer, document sources, and verified disjointness from the 768-context
KLD suite and 32-context screening partition.

**50M corpus composition.** The 4M corpus's 11-source mixture cannot scale by
token count alone: two locally curated sources are saturated
(`local-diverse-calib` at ~3.8M usable tokens, `local-deep-calib` at 0.695M)
and several external shards were cut thin for 4M. The 50M plan saturates the
local sources at their full remaining capacity, expands external shards from
their pinned upstreams, adds sources for three distribution gaps (model
continuations, literary prose, human-written raw code), and reallocates the
freed share. Target mixture (shares of 50M):

| Group | Sources | Share |
| --- | --- | ---: |
| Model continuations | anchor-student rollouts 10% + teacher (K3) rollouts 10% | 20% |
| Educational prose | fineweb-edu (50k-document shard) | 16% |
| Chat | ultrachat 9.6% + WildChat-1M 7.2% | 16.8% |
| Literary prose | PG-19 or Common-Pile Gutenberg 5.6% + FinePDFs (English narrative slice) 3.2% | 8.8% |
| Agentic coding trajectories | Open-SWE-Traces 4% + swe-agent 2% + swe-openhands 2% + Nemotron-Terminal-Corpus 2% + SWE-rebench-openhands 1.2% | 11.2% |
| Raw human-written code | the-stack-dedup, permissive subset, language-stratified to the trajectory block's nine languages | 2.4% |
| Math | open-web-math (expanded shard) | 9.6% |
| Local curated | local-diverse-calib + local-deep-calib, both saturated at absolute capacity | ~7.1% |
| Recall | reap_recall_calib | 4% |
| Chinese | fineweb2-cmn_Hani (expanded shard) | 2.1% |
| Tool-calling | apigen-mt + toolace (expanded shards) | 2% |

The continuation group exists because serving contexts are dominated by
model-generated tokens, which no human-text corpus represents: the student's
quantization drift compounds through its own generated context, and
compensation trained only on human text never sees those states (the
on-policy distillation argument). Anchor-student rollouts sample the states
the deployed artifact actually visits; teacher rollouts sample the states it
should visit.

**Continuation generation contract:**

1. **Two modes, weighted like serving traffic.** Chat mode (~70% of the
   group): seeds wrapped in the K3 chat template — real user turns from the
   chat sources, tool-calling task prompts, SWE issue statements, and local
   calibration prompts; multi-turn seeds keep one to three real turns of
   conversation prefix. Completion mode (~30%): raw document prefixes from
   the prose, code, and math sources truncated at natural boundaries,
   continued without a template.
2. **Seed sampling mirrors the mixture.** Seeds are drawn from the human-text
   groups proportional to their corpus shares, deduplicated by hash, and pass
   the same exclusion machinery as every document. The teacher and
   anchor-student rollouts use the *identical* seed set, so the two flavors
   are directly comparable and carry identical domain balance.
3. **Generate long.** Seed prefixes of ~200-1,500 tokens; generation fills
   the remainder of the 4,096-token document cap, so most documents are
   majority-generated — deep self-conditioned positions are the states this
   group exists to cover. A minority of short-generation documents covers
   early-position behavior.
4. **Agentic seeds use assistant-turn resampling.** Real trajectories from
   the trajectory sources keep their scaffold system prompt and real
   environment observations verbatim; only the assistant turns are
   regenerated — decode until a tool call, splice the source trajectory's
   actual observation, continue. Real environment, model policy, no
   execution infrastructure, no hallucinated observations.
5. **Serving-default sampling, near-zero filtering.** Temperature and
   sampling parameters are the serving stack's defaults, identical for both
   models, with recorded RNG seeds. One decode per seed (breadth over
   depth). The only filter is a degeneracy filter (n-gram repetition-loop
   detection); refusals and mediocre output are retained — they are part of
   the model's real distribution, and quality filtering would bias the
   corpus toward atypical generations.
6. **Per-document metadata:** the prompt/continuation boundary offset
   (enables a continuation-only loss-weighting ablation later) and the
   generator identity and mode tag. The chat template must be byte-exact
   with the deployed serving configuration, verified against the serving
   stack rather than a hub default — the corpus is tokenized once, and a
   template mismatch would silently corrupt the entire group.

Both flavors are tagged as distinct sources so per-source screening
attribution can decide which one carries the gain in the next corpus
revision. Generation configs — model identity, sampling parameters, seed
manifest, template hash — are recorded in the corpus plan's provenance.

Long-document sources define their chunking rule before shard pull, since
chunking determines document identity hashes: turn-aligned windows for
conversations and trajectories, chapter- or paragraph-aligned windows for
books, whole files for code (with generated-file filters: maximum line
length, alphanumeric fraction). Trajectory sources are model-generated; the
mixture bounds them at ~11% with generator diversity (at least four generator
models across three scaffolds) so no single model's style imprints, and the
raw-code slice grounds the code-token distribution in the non-trajectory
register. All external shards use pinned immutable upstream revisions with
recorded SHAs. The evaluation suites are natural-text contexts, so the
continuation share trains a distribution the pinned suite under-measures;
a small continuation-context KLD partition may be added as a secondary
diagnostic, but the pinned suite identity remains the only ledger metric.

## 6. Suffix recovery trainer (component 2)

The first trainer, and the gate for everything downstream. It trains only
parameters at or above the final residual segment, so the entire prefix
(layers 0-83) is frozen *and fixed*, and every optimizer step replays from the
stored boundary-84 slabs instead of running the full model.

**Trainable set:** non-expert BF16 tensors of layers 84-92 (attention,
shared experts, norms, latent projections, router gate weights), plus the
final RMSNorm and `lm_head`. The exact allowlist is derived from the
checkpoint's `model.safetensors.index.json` by excluding
`*.block_sparse_moe.experts.*` and anything below layer 84; the implementor
materializes the list explicitly in the run config (tensor names, shapes,
parameter counts) so a reviewer can audit what moved. Measured full-model
inventory: the complete continuous-recovery allowlist is 229 tensors,
6,243,186,496 parameters, 11.63 GiB in BF16, and ~93 GiB with BF16 gradients,
FP32 masters, and Adam state — small enough that optimizer memory is a
non-issue at suffix scale and comfortable at full depth without 8-bit
optimizers or offload.

**Topology:** the dense-gradient replay topology
(`qsrt/kimi_dense_gradient_replay.py` / `KimiPipelinedUpstreamReverse`), which
already sustains ~3,500 tok/s forward+backward through this exact segment:
one decoder layer resident per GPU (9 GPUs, decoded-BF16 experts, 59.2
GB/layer), documents queue-chained through the stages. The remaining 3 GPUs
host the final norm + `lm_head` + frozen teacher-suffix replica + loss, and
the data loader. Additions over the replay:

1. `requires_grad=True` on the allowlisted tensors of each resident layer.
2. A local AdamW per stage: FP32 master weights and moments live on the GPU
   that owns the layer. No optimizer state ever crosses GPUs; the only
   inter-GPU traffic is boundary activations and cotangents (14 KB/token per
   cut).
3. A loss head: student logits from the trainable suffix; teacher
   probabilities from the frozen teacher replica applied to the stored
   teacher hiddens; token-mean forward KL in FP32, chunked over tokens so the
   vocab-sized logit tensors stay bounded.
4. A data loader sampling shuffled document batches from the RAID: stored
   boundary-84 slabs (student input) + teacher hiddens (targets). At 3,500
   tok/s the read rate is ~50 MB/s — I/O is a non-factor.

**Step shape:** minibatch 32-64k tokens (571-1,143 rows per expert — enough
arithmetic intensity for the expert GEMMs), gradient accumulation as needed,
activation checkpointing at decoder-layer granularity (recompute is safe
because the expert decode is deterministic). At the measured replay throughput,
a 4M-token epoch is approximately 19 minutes and a 50M-token epoch is
approximately 4 hours.

**Evaluation loop:** every N steps, evaluate held-out KL by suffix replay on
the screening partition (the candidate differs from the anchor only above
boundary 84, so anchor prefix activations remain valid — evaluation costs a
9-layer replay, not a 93-layer forward). Track train KL, held-out KL,
per-tensor-group update norms, and per-layer top-16 routing agreement
against the teacher (the routing drift is conditional on hidden states, so
recovery training should improve it as a side effect; section 8). Checkpoint the trainable tensors (safetensors,
atomic write, metadata: base checkpoint identity, corpus identity, step,
optimizer config) whenever held-out KL improves.

## 7. Full-depth recovery trainer (component 3, gated)

Built only if the suffix trainer's held-out KL delta justifies it. Trains
non-expert tensors at all depths; no stored-activation shortcut exists because
activations change under the parameters being trained. Two layouts; choose
after the suffix results and a one-day spike of each:

**Option A — 12-stage pipeline parallelism.** ~8 layers per stage; per GPU:
~61 GB packed experts (fused ephemeral decode both directions — this option
requires the transpose-dequant kernel of section 4) + BF16 trainables + local
optimizer state + in-flight microbatch activations. Bubble fraction is
`(S-1)/(S-1+M)`: ~19% at 48 microbatches per step, ~10% at 96. Optimizer
state stays local to stages; only boundary tensors cross GPUs.

**Option B — NVMe waterfall training.** Forward sweep layer-by-layer over a
megabatch with every GPU on the *same* layer (experts sharded 12 ways, ~5
GB/GPU decoded), boundary activations spilled to the RAID (14 GB per boundary
per 1M tokens; a full 93-boundary pass is ~1.3 TB per direction, ~13 s of I/O
against ~10 minutes of compute); then a backward sweep in reverse reading
them back, recomputing within-layer activations from the stored boundary.
Semantically this is gradient accumulation over the megabatch. Zero pipeline
bubbles, no fused-decode kernel needed (decode once per layer per sweep,
amortized over the megabatch), trivially balanced. Its cost is optimizer step
count: one step per megabatch. At 1M tokens/step a 50M-token epoch is 50
steps — starvation. Cap the megabatch at 128-256k tokens (~200-400 steps per
epoch), which keeps I/O amortization intact.

**Trainable-set staging within either layout:** shared experts + norms +
router gate weights first (optimizer state small enough to ignore); extend to
attention and latent projections only if the per-parameter returns from the
first set justify the added optimizer memory (use 8-bit optimizer state or
RAID offload — at 100 GB/s, streaming optimizer state is viable — if the full
set is enabled).

## 8. Router bias frequency matching (component 4)

**Resolved semantics** (code checkpoint
`c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721`, `modeling_kimi_linear.py`,
`KimiGate.forward`): the K3 gate computes `scores = sigmoid(logits)`, selects
top-16 over `scores + e_score_correction_bias`, and takes the gate values
from the *unbiased* scores (`topk_weight = scores.gather(1, topk_idx)`),
renormalized to sum 1 (`moe_renormalize = true`,
`routed_scaling_factor = 1.0`; `num_expert_group = 1`, so the
grouped-selection branch is inactive). The bias is therefore selection-only:
its gradient is zero almost everywhere and gradient descent cannot move it —
the feedback rule below is the required mechanism. The gate weight itself
remains fully differentiable through the sigmoid and the renormalization. The
gate forward also asserts `not self.training`, so trainers keep every module
in eval mode and drive autograd through `requires_grad` alone (correct
regardless: no train-mode behavior is wanted). The feedback rule:

1. Run the student and teacher over a shared token sample (forward only; the
   teacher run can reuse waterfall capture outputs if per-layer selections are
   recorded there).
2. For each MoE layer and expert, compute the selection frequency divergence
   `f_student(l, e) - f_teacher(l, e)`.
3. Update `bias(l, e) -= eta_b * (f_student(l, e) - f_teacher(l, e))` — the
   aux-free load-balancing update, targeted at the teacher's routing
   distribution instead of uniform load.
4. Iterate to convergence of the frequency divergence; then re-measure KL on
   the screening partition. Accept only if KL does not regress: matching
   marginal frequencies is a proxy — the acceptance metric remains KL.

Step-size stability: the sensitivity of selection frequencies to the bias is
proportional to the score density at the selection threshold, and that
density varies by orders of magnitude across layers — at layer 12, 54.28% of
tokens sit within `1e-4` of the threshold (median margin ~`8.8e-5`, roughly
one tenth of its neighbors; see the experiment journal). A single global
`eta_b` sized for quiet layers will overshoot and oscillate exactly where
the loop matters most. Scale `eta_b` per layer, inversely to the measured
near-threshold token fraction, and verify convergence at layer 12 first.
Additionally, respect the sampling-noise floor of the fit population: at 4M
tokens each expert expects ~71k selections (~0.4% relative standard error
per frequency estimate). Suppress an expert update unless the absolute
teacher/student frequency difference exceeds 2.5 times its conservative
independent-binomial standard error. Each iteration re-measures student
frequencies with a complete 4M-token forward pass because bias changes alter
all downstream hidden states. Permit at most four measured student passes;
failure to settle in two to four passes is evidence that the per-layer
scaling is unstable, not that the corpus should be enlarged after observing
the result.

Measured qualification (experiment journal, round 1): capping the largest
bias change at the median selection margin is *over*-damped for a large
marginal displacement — a margin-sized step at layer 12 reduced its
noise-resolved TV by only 0.224 percentage points of 27.1 (0.83% relative),
stable and correctly signed but two orders of magnitude short of closing
the gap within any bounded round budget. The margin is the token-level flip
scale, not the layer-level stability scale; stability is governed by the
frequency-versus-bias response slope, which consecutive measured rounds
estimate directly. A successor update rule should take per-expert secant
steps — `db_e = -(f_e - f_teacher_e) / slope_e`, with `slope_e` estimated
from the two most recent (bias, frequency) measurements for experts whose
movement resolved above the noise floor — clamped to a bounded multiple of
the margin and still noise-suppressed. Any such rule change applies only as
a new pre-registered experiment after the current contract completes; it
never amends a running one.

**Sequencing.** This component is not a prerequisite for anything: no other
component consumes its output, and it is adopted only if the
routing-agreement diagnostic (below) shows a routing channel worth closing.
But if it is adopted, it must run *before* the waterfall capture of student
boundary slabs: bias updates in layers below the suffix cut change the
prefix computation, which invalidates stored boundary-84 activations. The
order is therefore: diagnostic first; if adopted, run the bias loop to
convergence and freeze it; then capture slabs at the bias-corrected anchor;
then train. If the diagnostic shows near-teacher routing agreement, skip the
loop and revisit only at full-depth scope.

**Routing-agreement diagnostic (measured).** Over 65,536 shared token
positions across all 92 routed layers, with router tensors verified identical
between teacher and student: mean top-16 overlap 89.84%, mean exact-set
agreement 18.54%, mean marginal selection-frequency total variation 1.85%.
Overlap degrades with depth (86.09% over layers 70-92; 73.33-80.74% at
layers 89-91) and layer 12 is an isolated sensitivity event (70.07% overlap,
~6.7% boundary relative L2 error). Hidden-state relative error compounds to
~43.7% at boundary 84 and ~60.3% at boundary 92. All routing divergence is
quantization-induced hidden-state drift, not router loading error.

**Measured status.** The four-round margin-scaled contract was stable and
improved every layer but remained far from frequency closure. A subsequent
64x-clamped layer-12 secant step removed 5.17 points of layer-12 marginal TV
(26.23% to 21.05%) in one measurement. That probe used the captured
round-four cumulative all-layer biases as its baseline; only the additional
secant step was layer-local. Its paired same-runtime screening comparison
showed no quality effect: KLD delta +4.6e-6 and top-1 agreement -0.055
percentage points. The result rejects the combination of the four small
all-layer updates and the large layer-12 correction, but it does not determine
whether larger corrections at the other 91 layers matter. Their wider routing
margins can make changed selections less interchangeable than the near-tied
layer-12 selections. The all-layer secant experiment below is the final direct
test of the router-bias channel before the suffix prefix is frozen.

**Interpretation — what the bias can and cannot fix.** A per-expert bias
shifts selection thresholds uniformly across tokens, so it can only move
*marginal* selection frequencies. Globally those are nearly matched
(TV 1.85%), so the loop's average addressable share is small — but the
marginal mismatch is highly nonuniform: layer 12's marginal TV is 28.4%
(`docs/qsrt-continuous-recovery-experiment-journal.md`), which is exactly
the pathology a per-expert selection bias can correct. Evaluate the loop's
effect per layer, with layer 12 as its primary target and success metric,
not the global average. The remaining large disagreement is *per-token
conditional* routing drift caused by perturbed hidden states, which the bias
cannot reach — a small global result must not be read as evidence the
routing channel is small. The conditional drift is addressed indirectly by
recovery training itself — less hidden drift means fewer near-threshold
flips — so per-layer routing agreement is tracked as a secondary metric in
the trainer evaluation loop (section 6).

### Layer-12 secant calibration experiment

**Status: research-only and pre-registered.** The four-pass margin-scaled
frequency-feedback contract remains unchanged. After its fourth measured
student pass, construct one layer-12-only secant probe. Let `b0`, `f0` denote
the baseline layer-12 biases and selection frequencies, and let `b1`, `f1`
denote the values after the first margin-scaled update. Remove the
selection-null common shift from `b1 - b0`. An expert receives a slope estimate
only when its frequency movement exceeds 2.5 times the conservative
independent-binomial standard error and the estimated slope is positive:

```text
s_e = (f1_e - f0_e) / centered(b1_e - b0_e).
```

Let `f4` be the fourth-pass frequencies. Starting from the fourth-pass
cumulative bias, propose

```text
delta_b_e = -(f4_e - f_teacher_e) / s_e.
```

Experts without a resolved positive slope receive zero update. Clamp each
proposal to plus or minus 64 times the layer-12 median selection margin from
the fourth pass, then remove the common mean from the complete layer-12
update. All other layers remain byte-identical to the fourth-pass bias view.
The multiplier 64 is fixed before observing the probe: it tests the empirical
response beyond the margin-sized regime while bounding the intervention well
below an unconstrained secant solution.

Run exactly one 4,000,000-token frequency capture for the probe and report:

- layer-12 marginal and noise-resolved total variation;
- resolved positive-slope expert count;
- unclamped and clamped update distributions;
- the frequency response relative to the fourth-pass state; and
- the fixed 65,504-position, 32-context screening KLD and top-1 agreement.

The unchanged direct-Viterbi uniform-K2 anchor has screening KLD
`0.07834965130622809`. If the probe substantially closes layer-12 marginals but
does not improve screening KLD within the suite's paired variation, freeze the
fourth-pass bias view only if its own screening gate passes; otherwise freeze
the unchanged anchor. If the probe improves screening KLD, specify a separate
multi-layer secant experiment with refreshed two-point slopes, the same noise
suppression, an explicit clamp, and a fixed round cap before executing it.

Any frozen bias view that differs from the anchor requires the complete
768-context distribution-fidelity suite and a ledger entry before suffix
recovery results use it as a baseline. Screening results never substitute for
that suite identity.

### All-layer secant closure experiment

**Status: research-only and pre-registered.** Apply one bounded secant update
simultaneously to all 92 routed layers. The baseline is the bias tensor loaded
and measured by
`router-frequency-student-fit4m-round4-v1.safetensors`: it contains the four
cumulative all-layer updates after runtime BF16 conversion. The experiment
does not start from the unchanged anchor and does not include the separate
layer-12-only secant update.

For every layer and expert, fit one ordinary least-squares response slope over
the five measured round-zero through round-four pairs. Center each round's
896-element bias row before fitting because a common router-bias shift is
selection-null. Let `b_r` and `f_r` be the centered bias and selection
frequency in round `r`. The fitted slope is

```text
s_e = sum_r (b_r,e - mean_r(b_e)) (f_r,e - mean_r(f_e))
      / sum_r (b_r,e - mean_r(b_e))^2.
```

Retain a slope only when it is finite, positive, and its fitted frequency
excursion is resolved beyond 2.5 times the conservative independent-binomial
frequency uncertainty. Independently suppress the round-four residual
`f_4,e - f_teacher,e` unless it exceeds the same 2.5-standard-error threshold.
For eligible experts, construct

```text
delta_b_e = -(f_4,e - f_teacher,e) / s_e
```

and clamp it independently in each layer to plus or minus 64 times that
layer's round-four median 16th-to-17th selection margin. The 64x multiplier is
fixed before constructing the payload and matches the layer-12 calibration
probe. Experts without a resolved positive slope or resolved residual receive
zero update. Apply all layer updates simultaneously.

Run exactly one 4,000,000-token frequency capture and one 65,504-position
screening evaluation. Do not iterate or extend the experiment after observing
the result. Report per layer:

- marginal and noise-resolved total-variation changes;
- predicted and realized total-variation landing;
- resolved positive-slope and updated-expert counts;
- raw and noise-resolved residual sign-flip counts; and
- bounded update magnitude relative to the layer margin.

Under a locally linear response, mean marginal TV should fall from 1.442% to
approximately 0.7-1.0%. The payload-construction report records the exact
per-layer linear predictions before the frequency capture. The risk is
asymmetric: wider-margin layers may replace decisive expert selections, so a
KLD regression is an expected and informative outcome rather than an
implementation failure.

Screening decisions use same-runtime paired measurements. The pinned
`0.07834965130622809` value is historical context, not a gate reference. The
layer-12 paired null established an absolute mean-KLD detection threshold of
`0.000584` on 65,504 positions. A quality gain requires a candidate-minus-
anchor delta below `-0.000584` and a paired window-bootstrap interval wholly
below zero. A delta within that threshold or any regression closes the
router-bias channel, freezes the unchanged anchor, and releases suffix-archive
capture. A detected improvement permits one separately registered bounded
convergence experiment before the freeze; an accepted bias view then requires
the 768-context distribution-fidelity suite and a ledger entry.

Every screening gate must capture or select an anchor produced by the same
vLLM, B12X, QSRT, launcher, and evaluation-tool revisions as its candidate.
The pinned and rebuilt anchor measurements differed by 0.46% relative while
B12X changed from `7eef2ce3d046e81371a63fa8f79dc9f580f0fdba` to
`6714ff09bc5be749c6f674ac8e2ba6a3b6a40ab4`, vLLM changed from
`0578dd057139340cba19740f3bb8c44777a0854a` to
`f0d8e41c735a163fb8379f3447515f68cdef838b`, and the QSRT dirty-tree identity
also changed. The available evidence cannot attribute the evaluator shift to
one component, so unmatched runtime measurements are invalid for promotion.

## 9. Per-expert low-rank compensation adapters (component 5, gated)

Attach a rank-`r` adapter to each frozen expert matrix so the executed
operator becomes `y = W_k2 x + B (A^T x)`, with `A` and `B` in BF16 and
trainable. This gives every routed expert its own continuous correction while
the K2 payload stays frozen, preserving the no-lattice optimization
properties of section 2. The component stands or falls on one measurable
question — is the functionally important part of the K2 quantization error
approximately low-rank? — and a training-free baseline answers it before any
training code exists.

**Cost arithmetic (rank 8):**

| Quantity | Value |
| --- | ---: |
| Adapter params per expert (w1+w3+w2) | 19,968 x r = 159,744 |
| Full model, BF16 | 13.2e9 params, 26.3 GB |
| Effective rate cost | ~0.077 bpw |
| Serving compute overhead | ~0.5% of expert MACs |
| Final-segment (9 layers) trainable params | 1.29e9 (~21 GB optimizer state) |
| Full-depth optimizer state | ~211 GB (~18 GB per stage under 12-way PP) |

The equal-byte benchmark the adapters must beat: ~26 GB also buys a K2-to-K3
re-encode of roughly 7% of experts. Every adapter result is reported against
a matched-byte mixed-rate arm and the shared-expert arm on the same suite.

**Training-free baseline (error truncation).** For each expert matrix in the
final residual segment, compute the truncated SVD of the quantization error
`W_source - W_k2` at the interface the serving kernel executes, install the
top-`r` factors as frozen adapters, and evaluate by suffix replay. Run two
variants — plain Frobenius SVD and input-second-moment-weighted SVD — and let
the evaluation pick: the encoder-side history in
`docs/qsrt-fisher-experiment-ledger.md` warns against assuming the weighted
variant wins. The product is a KL-vs-rank curve from ~24,000 batched
truncated SVDs (hours of GPU time, no training loop). A flat curve kills the
component cheaply; a knee at low rank justifies everything below and provides
the initialization for the trained variant.

For the weighted variant, real routed input rows already exist:
`/data/datasets/kquant/captures/k3-all-routed-4m-v1.kqrows` holds 4M routed
expert-input rows (3584-dim latent, top-16, gate-weight convention recorded
in its manifest) per MoE layer for all 92 layers — ~71k rows per expert per
layer, enough for a rank-16 activation-weighted fit — and
`k3-denseh-broad-v7-4m-train-input-v1.kqsamples` covers the W2 input side.
Caveat: those rows were captured against the
`Kimi-K3-QSRT-3p08-COUPLED-HADAMARD-DRAWS-0-7-v1` checkpoint, not the K2
direct-Viterbi anchor, so the weighting distribution is a labeled proxy —
acceptable for an initialization that training refines; recapture at the K2
anchor for anything load-bearing. The capture's 4M-token corpus manifest is
also a natural seed for the training corpus. The rows are expert-input
snapshots, not boundary states, and are not training data for the trainers.

Measured truncation curves with document-disjoint validation (experiment
journal, 16 experts at layer 84, full routed support for all three
matrices): in plain Frobenius terms the K2 error is essentially
incompressible — rank 16 captures ~1.5%, barely above the random-matrix
baseline — but under activation weighting the held-out capture at rank 16
is ~14% for gate/up and ~40% for down, with down showing a pronounced
rank-2 knee (~36% already at rank 2) while gate/up keep gaining through
rank 16. Weighted factors beat plain factors on the held-out weighted
objective for every expert, matrix, and rank. The structural reading:
trellis encoding leaves error that is white in coefficient space (nothing
left for a fixed-rate encoder to exploit) yet concentrated in activation
space (much for an additive side-channel corrector to exploit, since a
corrector pays no rate cost). Two follow-ups before these factors become
initializations: report the weighted second moment's own eigenspectrum
concentration at matching ranks, so metric concentration is separated from
error concentration (a rank-2 down knee could partly reflect a
near-rank-2 input distribution — functionally equivalent for a corrector,
but different for interpretation and for how the structure generalizes);
and carry a mixed-rank arm (low-rank down, higher-rank gate/up) into the
equal-byte comparison, since the measured knees differ by matrix. KL
relevance still requires suffix-replay evaluation, and the weighting
remains a 3.08-bpw-vintage proxy.

**Trained variant.** The suffix trainer (section 6) runs unchanged with the
trainable set swapped to (or extended with) the adapters, initialized from
the SVD factors. Adapter gradients are rank-`r` projections of the per-expert
row streams the existing hooks already tap: with `z = A^T x` and
`delta = dL/dy`, `dL/dB = delta z^T` and `dL/dA = x (B^T delta)^T`. No dense
expert-weight gradient appears anywhere. Report the trained-over-initialized
delta separately, so end-to-end training's contribution is measured rather
than assumed.

**Consistency check:** the dense final-KL gradient archive for layers 89-92
gives an independent probe — the SVD adapter subspaces should overlap the
dominant singular subspaces of the corresponding expert gradients. Weak
overlap predicts that error truncation and KL descent disagree about which
directions matter, raising the expected value of the trained variant over the
frozen baseline.

**Rank allocation:** uniform `r` first. Adaptive per-expert allocation (by
measured per-expert KL contribution or error energy) only after uniform
results exist.

**Overfitting:** at 50M tokens each expert sees ~890k routed rows per layer
against ~160k adapter params (~5.6:1). Held-out evaluation is load-bearing at
this ratio, and the corpus may need to grow before full-depth adapter
training.

**Serving-format commitment:** unlike components 1-4, this component adds new
tensors and one additive branch (two small grouped GEMMs) to expert
execution. That commitment is accepted or rejected explicitly at the M3 gate
on measured equal-byte results, before any full-depth investment.

**Deferred endgame — fold-back re-encoding (noted, not scheduled).** If
trained adapters produce corrections whose per-entry amplitude is comparable
to the K2 quantization step, the correction can be folded into the payload
itself: re-encode `W* = W_k2 + B A^T` with the direct-Viterbi encoder,
retrain adapters around the new anchor, and iterate — alternating
minimization with a KL-trained oracle in place of LoftQ's SVD step. The
projection is nearly risk-free: the current payload is itself a legal
codeword at distance `||B A^T||` from the target, so re-encode distortion is
bounded by the correction norm, and an accept gate (the adapter-free
re-encoded model must beat the adapter-free anchor on the screening
partition) makes each round a ratchet whose worst case is the status quo.
Because the move is justified by measured training rather than a local
model, it is not subject to the 1/128 linearization radius that bounds
gradient-tilted re-encoding. Suffix-scope iteration leaves boundary-84 slabs
and teacher targets valid across rounds. Decision inputs when attempted:
geometric survival `||Q(W*) - W*||^2 / ||B A^T||^2` per matrix and the KL of
the re-encoded adapter-free model — one 9-layer re-encode plus suffix replay.
A fold with zeroed adapters must reproduce the anchor bit-for-bit. The fixed
point — a payload that has absorbed everything the lattice can express of
the trained correction — is where the ship-with-or-without-adapters decision
properly belongs. Out of scope until M3 produces strong adapter results.

## 10. Optimization recipe

Starting points, to be tuned during the suffix pilot:

- AdamW, beta1=0.9, beta2=0.95, eps=1e-8.
- **Weight decay 0, or decay-toward-init.** Parameters start at pretrained
  values and the goal is a small compensating displacement; decay toward zero
  fights the pretrained function.
- LR 1e-5 to 5e-5; linear warmup ~100 steps; cosine or constant after.
- Gradient clipping at global norm 1.0.
- BF16 compute matching serving numerics; FP32 master weights; FP32 softmax
  and KL. Per-microbatch gradients may be BF16, but accumulate across
  microbatches in FP32 buffers — the extra ~12 GiB is trivial and BF16
  accumulation loses small contributions over long accumulation windows.
- No loss scaler (BF16 range makes it unnecessary).
- Routing will shift as norms and gate weights move; this is ordinary MoE
  training behavior, not an error condition.
- Expect a loss floor well above zero: the student cannot match the teacher
  beyond K2 expert capacity. The metric that matters is the held-out plateau
  relative to `0.062993`.

## 11. Storage budget (30 TB filesystem)

| Artifact | Size |
| --- | ---: |
| Tokenized corpus, 50M tokens + extents | ~0.1 TB |
| Initial exact suffix archive, 4M training tokens (eight-slab student state + teacher normalized LM-head target; section 5) | 0.469 TiB |
| Exact suffix archive, 50M training tokens (eight-slab student state + teacher normalized LM-head target; section 5) | 5.867 TiB |
| Exact screening archive (eight-slab student state + teacher normalized LM-head target) | 7.875 GiB at 65,536 tokens |
| Transient waterfall state for one 4M-token capture shard | 0.427 TiB |
| Full-depth waterfall activation spill (256k-token megabatch) | ~0.35 TB |
| Trainable-tensor checkpoints (keep ~10) | ~1-2 TB |
| Available before capture | 6.22 TiB |

The existing `k3-all-routed-4m-v1.kqrows` routed-input capture (4.9 TB,
consumed read-only by the M2 weighted-SVD fit, section 9) resides on this
filesystem. The 50M exact suffix archive cannot safely coexist with that
capture and normal transient/checkpoint headroom under the available capacity.
Complete the M2 fits, then relocate or remove the routed-row capture before the
50M archive is written. The 4M training and screening archives fit without that
storage transition.

## 12. Deliverable format and materialize extension

Tuned checkpoints ship as a **BF16 side-tensor overlay**: a safetensors file
of exactly the allowlisted tensors, with metadata recording the base
checkpoint identity, training corpus manifest hash, step, and objective. The
materialize step (`scripts/materialize_qsrt_fp32_kl_refinement.py` lineage)
gains an overlay type that swaps these tensors while hardlinking everything
else, with the same atomic-write and completion-record discipline as the
payload overlays. If the adapter component is accepted (section 9), adapter
factors ship as a second overlay type of *new* tensor names, and the serving
loader gains the additive expert branch; that loader change is part of the
component's acceptance, not an incidental patch. Ledger entries and
full-suite evaluations always run against the materialized artifact, never
against in-memory training state.

## 13. Validation gates

1. **Zero-step gate:** the trainer with LR 0 (or step count 0), evaluated on
   the corpus, must reproduce the anchor's KL exactly, and the materialized
   zero-step overlay must reproduce the trainer's initial BF16 runtime tensors
   bit-for-bit. BF16 tensors already stored in that representation must also be
   bit-identical to their checkpoint values. Shared-expert matrices stored as
   MXFP8 are deliberately materialized as BF16 and therefore cannot be raw-byte
   identical to their serialized tensor and scale pair; for those matrices the
   required gate is exact equality to the BF16 tensor produced by the anchor
   loader, followed by execution parity with the anchor. This is the analogue
   of the payload pipeline's alpha-zero bit-exactness gate without confusing
   serialized MXFP8 identity with decoded runtime identity.
2. **Autograd parity:** the trainer's distributed execution must compute the
   same parameter gradients as a plain single-graph autograd reference. The
   reference requires a model small enough to run as one monolithic autograd
   graph, so the test builds a miniature model with the real architecture's
   *shape* — a few decoder layers with routed experts (e.g. 8 experts,
   top-2), shared experts, norms, and a small LM head at toy dimensions —
   and runs the identical inputs through both paths: (a) the actual trainer
   machinery (segment-resident pipeline, queue-chained documents, stored
   boundary inputs, frozen-teacher targets, chunked KL loss head, eval-mode
   modules with `requires_grad`-driven autograd) and (b) one
   `torch.autograd.grad` call on the monolithic graph. Every trainable
   parameter's gradient must match, in FP32 or FP64 so the comparison is
   tight rather than tolerance-fuzzy, and one optimizer step must produce
   matching parameter values. This is the only gate that catches
   silently-wrong-gradient bugs — a contribution dropped at a stage
   boundary, a misplaced `detach`, an unfired hook, loss-chunking or
   accumulation-order errors — the class where training runs and loss falls
   but toward the wrong optimum. Scale- and numerics-dependent behavior is
   deliberately out of scope here; gates 1 and 3 own it. Precedent for the
   pattern: `tests/test_kimi_reverse_pipeline.py` (pipeline cotangents vs
   autograd) and `tests/test_kimi_official_forward_hooks.py` (tap wiring vs
   autograd); this gate extends the same discipline to parameter gradients
   and the optimizer step.
3. **Serving-numerics parity:** the production packed kernel and decoded-BF16
   replay satisfy the measured bound in section 4. Decoded-BF16 is the frozen
   training operator.
4. **Waterfall capture parity:** final-boundary hiddens from the megabatch
   waterfall vs the existing per-document capture on a shared 100k-token
   sample. Bitwise equality is not expected (batched GEMM reduction order
   differs); gate on max-abs hidden difference and on the KL delta it induces
   through the frozen head being below measurement noise of the screening
   partition.
5. **Adapter null gate:** the adapter execution path with zeroed factors must
   reproduce the anchor's outputs bit-identically (adding exact zeros changes
   nothing), so the branch's presence alone is proven inert before any
   nonzero factors are evaluated.
6. **Evaluation discipline:** training corpus and screening partition
   disjoint from the 768-context suite (recorded in manifests); final numbers
   only from the fixed suite identity on materialized artifacts; report
   held-out regressions and the paired-context interval, not just the mean.

## 14. Delivery plan

| Milestone | Contents | Gate to proceed |
| --- | --- | --- |
| M0 diagnostics | Routing-agreement measurement; serving-numerics parity; trainable-set inventory from index.json (router-bias semantics and embedding tying already resolved, sections 8 and 15) | Findings recorded; no code risk |
| M1 waterfall capture | Megabatch weight-stationary mode for teacher hiddens + exact eight-slab student state at layer 84; authenticated 4M training archive and screening archive first; 50M confirmation archive assembled concurrently | Capture parity gate; throughput >= 2,000 tok/s teacher |
| M2 adapter error-truncation baseline | Batched truncated SVD of final-segment quantization error (plain + weighted variants; weighting from the existing `k3-all-routed-4m-v1.kqrows` routed-input capture, section 9); frozen adapters evaluated by suffix replay; KL-vs-rank curve; gradient-subspace consistency check | A knee at low rank continues the adapter track; a flat curve drops component 5 at no further cost |
| M3 suffix trainer | Segment-resident trainer + loss head + eval loop; zero-step, autograd-parity, and adapter-null gates; arms: BF16 side tensors, SVD-initialized adapters, both; matched-byte mixed-rate comparison; initial training on 4M tokens and confirmation of the selected configuration on 50M tokens | Held-out KL deltas on the screening partition, then full-suite eval of the best materialized checkpoints; adapter serving-format decision only after 50M confirmation |
| M4 overlay + materialize | BF16 side-tensor overlay type; adapter overlay type and loader branch if accepted; ledger entries with suite-identity results | Materialized artifact reproduces trainer eval |
| M5 full-depth trainer | Option A vs B spike, then build the winner; staged trainable set | Only if M3's full-suite delta justifies the build |
| Parallel | Router bias frequency-matching loop | Accept on non-regressing screening KL |
| Parallel | Archive-independent suffix-trainer scaffold: toy autograd parity, dense-distribution loss head, and optimizer plumbing | Ready before the 4M boundary slabs complete |
| Deferred | Adapter fold-back re-encoding loop (section 9): fold trained corrections into the payload, retrain, iterate | Only after M3 shows strong adapter results; each round gated on adapter-free screening KL |

M0 through M3 are roughly days of wall time on the fleet; every expensive
decision — M5, fused kernels, extended trainable sets, the adapter
serving-format commitment — is made on measured deltas from M2 and M3.

## 15. Open questions (resolve in M0)

1. KDA/attention per-document state handling in the megabatch waterfall:
   confirm the existing capture's document-unit processing transfers to the
   batched layout unchanged.
2. Adapter attachment interface: the exact coordinates at which the grouped
   expert dispatch executes decoded weights (and therefore where the adapter
   branch and its error-SVD initialization must be defined).

Resolved during design (from the checkpoint config and modeling code):
`gate.e_score_correction_bias` is selection-only (section 8), and
`tie_word_embeddings = false` at both config levels — `lm_head` is untied, so
training it does not touch the embedding table. Resolved by M0 measurement:
routing-drift statistics (section 8) and the trainable-set inventory — 229
tensors, 6.243e9 parameters, ~93 GiB full optimizer footprint (section 6).
The layer-12 sensitivity event is also resolved: aggregate and per-expert
payload distortion are normal, and an eight-draw re-encode reduces aggregate
SSE by only 0.0403%. The layer's median 16th-to-17th biased router-score margin
is 9-12 times smaller than its immediate neighbors, while 16.79% of its
boundary error energy lies in the router row-space against a 12.5% isotropic
expectation. Tight margins and router-aligned hidden error jointly explain its
route instability; a targeted expert re-encode is not indicated.
The production loader accepts both
`kquant_kimi_k3_qsrt_atoms_v2` completion records and
`qsrt_kimi_k3_qsrt_atoms_v2` layer metadata. The packed-kernel parity result in
section 4 resolves the expert-operator choice in favor of decoded-BF16
training.

## 16. Reference identities

- Student anchor:
  `/data/models/Kimi-K3-QSRT-K2-DIRECT-VITERBI-ALL-LINEARS-v1-model`
- Anchor payload pool + overlays: as recorded in
  `docs/qsrt-fisher-experiment-ledger.md`
- Teacher: `moonshotai/Kimi-K3` @
  `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`
- KLD suite manifest SHA-256:
  `f3a79f7f28365d406a19a82cf210c25adf18974c4b9b607ab3754e9939f941cf`
- Throughput baselines: boundary archive `forward-run.json` (86 tok/s
  streaming capture); dense-gradient archive `dense-gradient-run.json`
  (~3,500 tok/s segment replay)
- Geometry constants: `qsrt/constants.py`

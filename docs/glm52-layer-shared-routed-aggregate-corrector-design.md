# GLM-5.2 layer-shared routed-aggregate corrector design

**Status:** Pre-registered technical design. No corrector parameters, serving
kernel, or KLD result exists.

## Purpose and evidence boundary

The corrector is a small trainable network that adds a bounded correction to
the router-weighted output of GLM-5.2's quantized routed experts. It leaves the
QSRT trellis payloads, router parameters, dense shared expert, attention
weights, and other model parameters unchanged.

The design tests one hypothesis: quantization error from the eight active
experts may have a low-dimensional structure after their outputs are weighted
and summed. A layer-specific low-rank network may learn that structure from a
bounded recovery corpus and reduce endpoint Kullback-Leibler divergence (KLD).

The corrector produces a bounded-recovery artifact. A favorable result would
establish the behavior of the tested artifact, training distribution, and
serving implementation. It would not establish that the QSRT codec alone is
better than EXL3. A complete-checkpoint claim still requires lower serialized
bytes, document-disjoint KLD, task quality, and production-serving validation.

The frozen experimental choices live in
[`experiments/glm52_layer_shared_routed_aggregate_corrector_pre_registration.json`](../experiments/glm52_layer_shared_routed_aggregate_corrector_pre_registration.json).
The broader promotion rules live in
[`docs/qsrt-improvement-strategy.md`](qsrt-improvement-strategy.md).

## Place the correction where routed errors become one model contribution

GLM-5.2 sends each token to eight of 256 routed experts in a mixture layer.
For layer `l`, let:

- `x_l` be the residual-stream input shared by the routed experts;
- `R_l` be the selected expert identifiers;
- `p_l,e` be the applied router coefficient for expert `e`; and
- `f_q,l,e` be expert `e` executed with its frozen quantized weights.

The quantized routed-expert contribution is

```text
y_q,l = sum over e in R_l of p_l,e * f_q,l,e(x_l).
```

The corrector adds one vector to `y_q,l`. It does not produce eight separate
full-width corrections. The runtime applies the addition in the routed
branch's 32-bit floating-point accumulator. It then merges the routed branch
with the unchanged shared-expert contribution and performs the final BF16 or
FP16 cast.

```text
residual input x_l
       |
       +-------------------------------+
       |                               |
       v                               v
router selects experts            rank-16 input projection
       |                               |
       v                               v
eight frozen QSRT experts          shared hidden features
       |                               |
       v                               +-- route-weighted expert modulation
router-weighted FP32 sum               |
       |                               v
       +---------------------> rank-16 output projection
                                       |
                                       v
                           add correction in FP32
                                       |
                                       v
                    merge unchanged shared-expert branch
                                       |
                                       v
                                one output cast
```

This boundary matches the model contribution whose downstream damage the GLM
measurements exposed. One-sided input-covariance selection reduced the
complete routed-expert error by 93.6575% on the reporting input and still
worsened full-model KLD. The remaining sensitivity therefore lies after the
complete expert functions, rather than inside a single gate, up, or down
matrix score.

The selected boundary also keeps arithmetic independent of the active-expert
count for the two full-width projections. Only the small modulation reduction
depends on the eight selected experts.

## Use one shared output basis with small expert modulation

Each corrected layer owns three trainable tensors:

| Tensor | Shape | Role |
|---|---:|---|
| `V_l` | `16 × 6,144` | Projects the normalized layer input into 16 shared features |
| `U_l` | `6,144 × 16` | Maps the shared features into one residual-stream correction |
| `a_l` | `256 × 16` | Modulates the shared features according to the selected experts and their router coefficients |

The network computes

```text
z_l       = SiLU(V_l * RMSNorm(x_l))
m_l       = 1 + sum over e in R_l of p_l,e * a_l,e
delta_y_l = U_l * (z_l elementwise-multiplied by m_l)
y_out,l   = y_q,l + delta_y_l.
```

`U_l` and `V_l` receive training signal from every token that reaches the
layer. The modulation row for one expert receives signal only when the router
selects that expert. Initialize all modulation rows at the layer-average
value, regularize them toward that value, and retain it when routed support is
inadequate. Shared projections reduce rare-expert dependence; they do not
eliminate sparse expert-specific supervision.

The output projection `U_l` defines the correction's output space. More hidden
layers, wider FiLM conditioning, or a larger expert embedding cannot represent
an error direction outside that space. A failed output-basis diagnostic must
therefore increase output rank or introduce several expert-clustered bases.

## Close parameter, storage, and arithmetic accounting

One rank-16 layer contains

```text
input projection       6,144 * 16 =  98,304 values
output projection      6,144 * 16 =  98,304 values
expert modulation        256 * 16 =   4,096 values
                                     -------
                                     200,704 values.
```

BF16 parameters occupy 401,408 logical bytes per layer. Twenty-three corrected
layers occupy 9,232,384 logical bytes. Serialized accounting must also charge
headers, tensor names, offsets, alignment, padding, checksums, schema data, and
any stored scales.

The two shared projections perform 196,608 multiply-accumulates per token and
layer. The modulation reduction adds about 128 multiply-accumulates for eight
routes at rank 16. Eight GLM experts perform approximately 301,989,888
multiply-accumulates for their three `6,144 × 2,048` projections. The
corrector therefore adds about 0.067% of routed-expert arithmetic.

This ratio does not predict latency. Small projections may be limited by
kernel launches, synchronization, memory traffic, and tensor-parallel
collectives. Production qualification must measure prefill and decode latency
through the intended serving kernel.

During training, BF16 parameters, gradients, and 32-bit optimizer state occupy
only tens of megabytes for a short suffix. Decoded expert weights and suffix
activations dominate device memory. The corrector therefore uses the bounded
suffix replay architecture defined in the improvement strategy.

## Prove the shared output-space ceiling before training

For one candidate-native input and route, define each routed expert's source
error contribution as

```text
epsilon_l,e(x) = f_source,l,e(x) - f_q,l,e(x).
```

The correction target is the applied aggregate error:

```text
epsilon_l(x) = sum over e in R_l of p_l,e * epsilon_l,e(x).
```

The label requires both complete nonlinear expert functions on the same input.
Subtracting source and quantized weights does not produce this label because
GLM experts contain SiLU and coordinatewise multiplication between gate and up
branches.

Fit a rank-16 singular-vector basis to aggregate errors from activation-fit
documents. Freeze the basis and measure captured energy on candidate-selection
documents. Fit separate rank-two bases for every supported expert on the same
fit documents, then evaluate them on the same selection documents.

Report two weightings:

- production weighting uses natural routes and applied router coefficients;
- expert-balanced weighting gives each supported expert equal influence.

The shared basis passes when it retains at least 80% of the held-out recovery
obtained by the per-expert rank-two collection. The network warm start must
then recover at least 80% of the shared-basis ceiling. The first requirement
tests output-space capacity. The second tests whether the input projection and
expert modulation predict useful coefficients within that space.

## Measure cancellation among co-routed experts

The aggregate target may be lower-rank than the union of individual expert
errors because co-routed contributions can cancel. For each token with a
nonzero denominator, compute

```text
                                      ||sum_e p_l,e * epsilon_l,e||^2
cancellation ratio = ------------------------------------------------------------ .
                     sum_e p_l,e^2 * ||epsilon_l,e||^2
```

A ratio below one means that cross-expert terms reduced aggregate error
energy. A ratio above one means that they reinforced it. Retain:

- the pooled numerator divided by the pooled denominator;
- the distribution of defined per-token ratios;
- route-support-stratified values; and
- document-level values so a few high-energy tokens cannot determine the
  pooled result without being visible.

This measurement supplies a prior for later multi-layer composition tests. It
does not establish whether errors cancel across layers because nonlinear
propagation and routing intervene between mixture blocks.

## Keep local regression and endpoint recovery on separate data contracts

Local aggregate-error regression needs:

- candidate-native layer inputs;
- routed expert identifiers and applied coefficients;
- the frozen quantized expert tensors; and
- bounded BF16 source tensors for every corrected layer.

The complete BF16 checkpoint is unnecessary. Additional documents require a
resident-student forward capture followed by offline execution of the bounded
source and quantized experts.

End-to-end KLD training needs two additional assets for every
recovery-training document:

- suffix boundary states produced by the exact frozen prefix; and
- matched canonical terminal endpoint targets.

Every boundary cache names the complete prefix-artifact hash. Any accepted
change before the suffix boundary invalidates that cache. Routed inputs alone
cannot supply endpoint KLD targets.

The existing layer-3 capture contains 62,148 fit tokens and 15,625 selection
tokens. Individual experts receive 762 to 2,478 routed fit rows. These are the
relevant support counts. The 6,144 output coordinates are correlated and do
not multiply the statistical sample count.

Every added corpus receives immutable document roles before capture. The
roles are recovery training, candidate selection, and sealed confirmation.
Documents cannot cross roles. A parameter-count increase requires a new route-
support report and held-out learning curve before training begins.

## Warm-start locally, then optimize endpoint KLD

The local warm start minimizes aggregate-output squared error on
recovery-training rows. It uses natural router weighting for the primary loss
and reports an expert-balanced diagnostic. Weight decay and a support-aware
penalty shrink expert modulation toward the layer-average value.

After the warm start passes its held-out ceiling-efficiency gate, bounded
suffix replay may optimize the serialized corrector against endpoint KLD.
Compare two training objectives under the same initialization and document
order:

1. document-balanced mean endpoint KLD; and
2. document-balanced mean endpoint KLD plus a smooth tail-risk penalty.

Small-minibatch CVaR is unstable because a minibatch contains too few tokens
from the target tail. The tail-risk arm must use a running quantile or a token
buffer whose size is frozen before selection results are opened. Its quantile,
penalty weight, warm-up, update frequency, and document balancing belong in
the training-run registration.

The optimizer state is a shadow training representation. Every accepted round
serializes the trainable tensors, loads them through the serving path, and
measures that computation. A favorable shadow-model loss cannot replace a
serving-path result.

## Preserve incumbent behavior and serving arithmetic

The disabled corrector bypasses every corrector operation. The closure test
requires bit-identical routed aggregates, logits, and routes relative to the
uncorrected student. Executing a zero projection and adding numerical zero
does not satisfy the bypass contract.

For enabled execution:

1. compute the low-rank input projection with FP32 accumulation;
2. retain the 16-value hidden vector in FP32;
3. reduce the selected modulation rows with the applied router coefficients;
4. compute the output correction with FP32 accumulation;
5. add the correction to the routed-expert FP32 accumulator; and
6. perform one BF16 or FP16 cast after the addition.

The earlier rank-four down correction lost most of its one-context KLD gain
when two BF16 GEMMs introduced a local relative error of about `2e-7`.
Load-time materialization reproduced the favorable dense endpoint bit for bit.
The corrector cannot be folded into expert weights because its output depends
on the token input and selected route. Its serving kernel must therefore close
against the registered training arithmetic before a KLD result advances.

The reference implementation and deterministic arithmetic fixtures live in
this repository. Production vLLM integration belongs in
`/home/luke/projects/vllm`. A fused B12X implementation belongs in
`/home/luke/projects/b12x`. QSRT commits must not stage files from either
sibling repository.

## Respect tensor-parallel ownership

The reference module operates on complete hidden vectors. The serving design
must record whether the layer input is replicated or sharded at the corrector
hook:

- A replicated input allows every rank to compute the 16-value hidden vector
  independently. Each rank then computes only its owned output slice.
- A hidden-sharded input computes a partial `V_l x_l` on each rank and
  all-reduces 16 FP32 values. Each rank still computes only its owned output
  slice.

Replicate the `256 × 16` modulation table because it occupies 8,192 BF16 bytes
per layer. The runtime must use the same complete route identifiers and applied
coefficients on every participating rank. Add a distributed closure that
compares one-rank reference output with the joined tensor-parallel result.

## Serialize enough information to reproduce the computation

The corrector container records:

- schema name and version;
- source, student, tokenizer, and prefix-artifact identities;
- corrected layer identifier and hidden width;
- rank and activation function;
- normalization identity and placement;
- `U_l`, `V_l`, and `a_l` logical shapes, stored dtypes, offsets, byte counts,
  and SHA-256 hashes;
- tensor-parallel ownership rule;
- FP32 accumulation and final-cast rule;
- training-data and candidate-selection manifest hashes;
- optimizer-run receipt and selected checkpoint identity; and
- total payload, header, alignment, padding, and directory bytes.

Reject duplicate layers, unsupported ranks, non-finite values, incorrect
shapes, overlapping offsets, hash mismatches, and unknown arithmetic versions.
The disabled artifact omits the corrector payload and selects the direct
bypass. A zero-filled enabled payload is a different execution case and must
pass its own numerical closure.

## Attribute routing changes without overclaiming

Compare every candidate route with the initialization student on the same
token. Cross two binary observations: whether the route at the corrected layer
changed and whether any later route changed. This creates four mutually
exclusive reporting groups.

| Corrected-layer route | Later routes | Interpretation |
|---|---|---|
| Unchanged | All unchanged | Continuous correction remained within the original routing path |
| Unchanged | At least one changed | The correction altered a later router input |
| Changed | All unchanged | An upstream correction changed the current route, followed by route stability |
| Changed | At least one changed | An upstream correction changed the current route and later routing also changed |

Report scored-position count, document count, paired mean KLD change, and an
uncertainty interval when support permits for every group. Small groups remain
descriptive. Their point estimates cannot support an overall conclusion.

A single corrector cannot change its own layer's route because it runs after
that routing decision. It can compensate for the output produced by the taken
route and can change later routes. Router recalibration directly targets route
selection. The two mechanisms require separate interventions before KLD
recovery is attributed to either one.

## Advance through fixed gates

Run the implementation in the following order:

1. Prove true-bypass identity.
2. Fit the shared rank-16 and per-expert rank-two bases, then apply the 80%
   held-out output-space gate.
3. Freeze one bounded late layer and train one rank-16 local warm start.
4. Require at least 80% realization of the shared-basis ceiling on untouched
   selection documents.
5. Reproduce the correction through the intended FP32 serving arithmetic.
6. Freeze a two-to-four-layer suffix training run and compare mean-KLD and
   smooth-tail-risk objectives.
7. Serialize the selected candidate before screening.
8. Advance only when the serving candidate improves held-out mean KLD,
   satisfies the registered document-level CVaR1% constraint, and beats the
   final-hidden-state recovery baseline by enough to justify measured bytes and
   latency.

The following observations authorize specific design changes:

| Observation | Authorized change |
|---|---|
| Shared rank 16 retains less than 80% of per-expert rank-two recovery | Increase output rank or test expert-clustered output bases |
| The shared basis passes, while the warm start realizes less than 80% of its ceiling | Improve the input projection, normalization, or modulation within the frozen output rank |
| Local aggregate error improves and serving arithmetic closes, while endpoint KLD does not improve | Train through bounded suffix endpoint KLD without widening the local network from that result |
| Damage concentrates where an upstream correction already changed the current route | Measure router restoration before expanding the corrector |
| Shadow training improves and serialized serving regresses | Correct serving arithmetic before another quality run |

A changed rank, basis count, layer set, loss, serialization dtype, or serving
arithmetic creates a new candidate. Freeze those choices before their
candidate-selection KLD is opened.

## Academic precedents and limits of transfer

Several research lines support individual parts of the design. None validates
the complete GLM-5.2 construction.

### Additive networks beside a frozen model

[Side-Tuning](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1104_ECCV_2020_paper.php)
trains a lightweight side network and combines its output with an unchanged
pretrained network by addition. This is the closest architectural precedent
for `y_q,l + delta_y_l`. Its experiments cover adaptation tasks in vision,
reinforcement learning, imitation learning, and question answering. They do
not cover quantization-error correction, routed experts, or language-model
KLD.

[Ladder Side-Tuning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/54801e196796134a2b0ae5e8adef502f-Abstract-Conference.html)
trains a small side network from frozen backbone activations and avoids
backpropagating through the complete backbone. This supports the memory logic
behind cached suffix boundaries and a trainable side path. Its side network
produces task predictions, while the GLM corrector modifies intermediate
routed-expert contributions.

### Small residual modules in frozen transformers

[Parameter-Efficient Transfer Learning for NLP](https://proceedings.mlr.press/v97/houlsby19a.html)
adds small bottleneck adapters to a frozen Transformer and trains only those
modules for downstream tasks. It establishes that low-dimensional residual
modules can adapt a large frozen language representation with few trainable
parameters. The GLM corrector uses a similar bottleneck, but its supervision
targets source-model fidelity rather than a new task.

[Learning Multiple Visual Domains with Residual Adapters](https://proceedings.neurips.cc/paper/2017/hash/e7b24b112a44fdd9ee93bdf998c6ca0e-Abstract.html)
uses residual adapter modules to share most parameters across domains. It
supports the general pattern of a frozen shared computation with small
specialized residual capacity. Its domain-specific adapters do not establish
that GLM expert quantization errors share one output basis.

### Identity-preserving initialization

[ControlNet](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html)
connects a trainable branch to a locked diffusion model through zero-initialized
layers so the branch begins with zero contribution. This is an architectural
analogy for zero-output initialization. The QSRT design requires a stronger
runtime condition: the disabled mode bypasses the corrector and must reproduce
the incumbent bit for bit.

### Low-rank reconstruction of quantization error

[CALDERA](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a20e8451ffb07ad25282c21945ad4f19-Abstract-Conference.html)
approximates a dense weight matrix with a low-precision backbone plus low-rank
factors and optimizes activation-weighted reconstruction. It directly supports
measuring a low-rank correction against the inputs the quantized operator will
receive. CALDERA does not aggregate routed experts or train a nonlinear helper
network against endpoint KLD.

[LQER](https://proceedings.mlr.press/v235/zhang24j.html) uses an
activation-induced scaling matrix to make low-rank quantization-error
reconstruction effective for dense LLM layers. [QERA](https://openreview.net/forum?id=LB5cKhgOTu)
derives a closed-form activation-aware quantization-error reconstruction and
evaluates both post-training correction and low-precision fine-tuning. These
results motivate the aggregate-error warm start and shared-basis ceiling test.
They do not establish the rank, expert sharing, or serving behavior for GLM.

[LoftQ](https://openreview.net/forum?id=LzPWWPAdY4) initializes low-rank
fine-tuning parameters jointly with a quantized model so the low-rank branch
starts closer to the full-precision model. It supports warm-starting bounded
recovery from a measured quantization residual before gradient tuning. LoftQ
operates on weight-space residuals; the GLM design uses complete nonlinear
expert-output errors on candidate-native inputs.

## Full citations

- Houlsby, Neil, et al. "Parameter-Efficient Transfer Learning for NLP."
  *Proceedings of the 36th International Conference on Machine Learning*,
  PMLR 97, 2019, pp. 2790–2799.
- Li, Yixiao, et al. "LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large
  Language Models." *International Conference on Learning Representations*,
  2024.
- Rebuffi, Sylvestre-Alvise, Hakan Bilen, and Andrea Vedaldi. "Learning
  Multiple Visual Domains with Residual Adapters." *Advances in Neural
  Information Processing Systems 30*, 2017.
- Saha, Rajarshi, et al. "Compressing Large Language Models Using Low Rank and
  Low Precision Decomposition." *Advances in Neural Information Processing
  Systems 37*, 2024.
- Sung, Yi-Lin, Jaemin Cho, and Mohit Bansal. "LST: Ladder Side-Tuning for
  Parameter and Memory Efficient Transfer Learning." *Advances in Neural
  Information Processing Systems 35*, 2022.
- Zhang, Cheng, et al. "LQER: Low-Rank Quantization Error Reconstruction for
  LLMs." *Proceedings of the 41st International Conference on Machine
  Learning*, PMLR 235, 2024, pp. 58763–58779.
- Zhang, Cheng, et al. "QERA: An Analytical Framework for Quantization Error
  Reconstruction." *International Conference on Learning Representations*,
  2025.
- Zhang, Jeffrey O., et al. "Side-Tuning: A Baseline for Network Adaptation via
  Additive Side Networks." *European Conference on Computer Vision*, 2020,
  pp. 698–714.
- Zhang, Lvmin, Anyi Rao, and Maneesh Agrawala. "Adding Conditional Control to
  Text-to-Image Diffusion Models." *Proceedings of the IEEE/CVF International
  Conference on Computer Vision*, 2023, pp. 3836–3847.

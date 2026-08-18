# Final-output KL-guided QSRT re-encoding

Status: **research-only**. The method changes only offline payload selection.
The served K2 format, reconstruction law, decoder, and kernel are unchanged.

## Objective

Let the official MXFP4 model define teacher token distributions `p`, and let a
materialized QSRT checkpoint with expert weights `theta_a` define student
distributions `q`. The optimized objective is the token-summed forward KL:

$$
J(\theta_a) = \sum_t \mathrm{KL}(p_t \Vert q_t).
$$

At the exact quantized anchor, reverse replay computes

$$
g = \nabla_\theta J(\theta_a).
$$

For another legal QSRT payload `theta_c`, the first-order model is

$$
J(\theta_c) - J(\theta_a)
\approx
\langle g, \theta_c - \theta_a \rangle.
$$

This is not a Fisher approximation and does not interpret gradient magnitude
as intrinsic weight importance. It measures the effect of changing the
specific quantized anchor, including its existing errors and model trajectory.

## Trellis objective

The direct K2 encoder normally chooses a legal tail-biting SQG path by
minimizing squared reconstruction error in the trellis work basis. If `x` is
the unmodified source target, `r` is a legal reconstruction, and `g_w` is the
exact decoder-adjoint transform of the source-coordinate gradient, the guided
implementation supplies the shifted target

$$
x' = x - \eta g_w.
$$

The ordinary squared-error Viterbi search on `x'` is equivalent, up to terms
independent of `r`, to ranking paths with

$$
\lVert r-x \rVert_2^2
+ 2\eta\langle g_w, r-x\rangle.
$$

The gradient term is therefore present during pathfinding. It is not a
post-hoc path rescore. Each alternative remains a valid K2 SQG path using the
same carry-mixed graph, finite-E4M3 reconstruction table, scales, transform
draws, tail-biting context, and packed payload layout.

The encoder normalizes the step separately for the coupled gate/up matrices
and the down projection:

$$
s = \sqrt{
\frac{\lVert \theta_a-\theta_{src}\rVert_2^2}
     {\lVert g\rVert_2^2}
},
\qquad
\eta = \alpha s.
$$

Scale fitting receives the unmodified source matrix. The gradient changes the
trellis work target but not the scale-search target. An `alpha = 0` encode must
be bit-identical to the direct-Viterbi anchor before nonzero steps are accepted.

No BlockLDLQ feedback, dense input Hessian, output Fisher factor, fitted W2
replacement, or decoder change participates in this arm.

## Gradient capture

The capture is anchored to an identified materialized QSRT checkpoint and an
identified official MXFP4 teacher.

1. A forward pass stores every decoder-boundary activation for the exact
   quantized anchor.
2. The official model supplies teacher boundary activations for the same token
   sequence.
3. At the final LM-head input, the exact logit derivative is

   $$
   \frac{\partial J}{\partial z_t} = q_t-p_t.
   $$

   The LM-head transpose and the suffix normalization map this derivative to
   a final decoder-boundary cotangent.
4. Reverse replay runs the exact quantized suffix and routed expert execution.
   Hooks at gate/up preactivations and down-projection outputs accumulate dense
   FP32 gradients for W1, W3, and W2.
5. The coupled activation-boundary transform is applied to both activations
   and cotangents, so each stored gradient is already in the same matrix
   coordinates used by offline QSRT encoding.

For a routed expert, the accumulated matrix gradients have the ordinary outer
product forms

$$
G_{13} = \sum_t \delta_{13,t}^{T} x_t,
\qquad
G_2 = \sum_t \delta_{2,t}^{T} h_t.
$$

Router weights, SiTU derivatives, W2, residual propagation, later layers,
normalization, the LM head, and the softmax KL all enter through the reverse
cotangents. Gradient files are sharded by contiguous expert ranges and stored
as memory-mappable FP32 arrays.

## Successive refinement

A second update must be relinearized around the first materialized update:

1. materialize the first legal QSRT payload;
2. capture its boundary activations and final KL cotangents;
3. recompute dense expert gradients at that anchor;
4. perform another small legal Viterbi update; and
5. compare complete checkpoints with the same end-to-end KLD suite.

Reusing the first gradient for another step is only a larger step under the
same local linear model. It does not measure successive optimization after the
model trajectory changes.

## Parallel encoding

Each Kimi-K3 MoE layer contains 896 independent expert payloads. A layer encode
is partitioned into 12 contiguous ranges of 74 or 75 experts, one per GPU.
Each worker reads only its gradient shards and emits a disjoint safetensors
overlay. Nonzero ranges run expert zero once as a discarded primer so that the
shared W1/W3 input-scale state is identical to a complete-layer encode.

The merge step requires contiguous coverage of experts 0 through 895,
consistent normalization metadata, no duplicate tensor names, and exactly
8,064 payload tensors. A one-expert nonzero-range check reproduced all nine
W1/W3/W2 payload tensors bit-for-bit. Layer 90 required 830.5 seconds on one
GPU and 75.9 seconds for the slowest of 12 workers, an approximately 11-fold
wall-time reduction before the one-second merge.

## Measured result

The direct-Viterbi K2 anchor and the first four-layer update were evaluated on
the same 768-context, 1,572,096-position distribution-fidelity suite.

| Model | Mean KL | Top-1 agreement |
| --- | ---: | ---: |
| Direct-Viterbi K2 anchor | 0.062993208155 | 93.571003% |
| KL-guided W1/W3/W2 at layers 89-92, `alpha = 1/128` | 0.062976057133 | 93.574883% |

Mean KL decreased by 0.02723% and top-1 agreement increased by 0.00388
percentage points. The candidate improved 399 contexts and regressed 369; the
paired context bootstrap interval for the mean KL change crossed zero. The
signal is directionally positive but not statistically resolved. Larger steps
under the same gradient regressed.

## Implementation map

| File | Responsibility |
| --- | --- |
| `scripts/capture_kimi_k3_boundary_slabs.py` | Streams the exact anchor model and stores decoder-boundary activations. |
| `scripts/capture_kimi_k3_upstream_fisher.py` | Builds final-boundary Fisher samples and, when requested, deterministic teacher-KL cotangents. |
| `qsrt/kimi_official_fisher.py` | Computes `q - p` at the LM head and maps it to the final hidden state. |
| `qsrt/kimi_suffix_pipeline.py` | Runs suffix normalization and LM-head work and writes cotangent slabs. |
| `scripts/capture_qsrt_dense_kl_gradients.py` | Configures exact-anchor reverse replay and creates a sharded dense-gradient archive. |
| `qsrt/kimi_dense_gradient_replay.py` | Keeps one residual segment resident across GPUs while replaying every expert shard. |
| `qsrt/kimi_dense_objective_gradients.py` | Hooks routed expert execution and accumulates FP32 W1/W3/W2 gradients in coupled coordinates. |
| `qsrt/two_sided_qsrt.py` | Applies the reconstruction adjoint and injects the shifted target into direct SQG Viterbi. |
| `qsrt/gradient_guided_viterbi.py` | Defines the linear-objective and complete-square utilities used by gradient-guided encoders. |
| `scripts/encode_qsrt_fp32_kl_refinement_layer.py` | Loads official MXFP4 weights, anchor payloads, and gradients; encodes complete or expert-sharded layer overlays. |
| `scripts/merge_qsrt_fp32_kl_refinement_shards.py` | Validates and merges disjoint expert overlays into one canonical layer overlay. |
| `scripts/materialize_qsrt_fp32_kl_refinement.py` | Replaces selected layer atoms, hardlinks unchanged anchor data, and emits a loadable model directory. |
| `docs/qsrt-fisher-experiment-ledger.md` | Records comparable full-model KLD results and exact artifact identities. |

## Research artifact identities

- Direct-Viterbi anchor:
  `/data/models/Kimi-K3-QSRT-K2-DIRECT-VITERBI-ALL-LINEARS-v1-model`
- First four-layer update:
  `/data/models/Kimi-K3-QSRT-K2-DIRECT-VITERBI-FINAL-KL-GRADIENT-L89-92-v1-model`
- Boundary archive for relinearization:
  `/data/datasets/kquant/captures/k3-qsrt-k2-direct-viterbi-final-kl-gradient-l89-92-a0p0078125-100k-v1-boundaries`
- Final-KL cotangents for relinearization:
  `/data/kquant/research/k3-qsrt-k2-direct-viterbi-final-kl-gradient-l89-92-a0p0078125-100k-v1-objective-cotangents`
- Second-update layer overlays:
  `/data/kquant/research/qsrt-fp32-kl-direct-viterbi-gradient-relinearized-a0p0078125-v1/overlays`

# Audit of the K2 no-feedback quantization claim

## Question and conclusion

The claim under review is that a Kimi-K3 two-bit QSRT encode obtained much
lower full-model Kullback–Leibler divergence (KLD) after replacing BlockLDLQ
with direct Viterbi encoding. K2 means that the trellis stores two branch bits
per weight. BlockLDLQ is the blockwise error-feedback procedure that changes a
later quantization target to compensate for reconstruction errors committed in
already encoded coordinates.

The reported improvement is plausible. The available description does not
establish that removing BlockLDLQ caused it. A production fixed-K2 encode and a
simple K2 encode can use differently fitted global scales, so the comparison
may change scale selection and feedback together. The full-model measurement
also needs a same-runtime identity control because repository KLD results have
changed materially with the non-expert overlay, tensor-parallel configuration,
and evaluation corpus.

The minimum defensible experiment freezes source weights, transforms, scales,
codebook, batch shape, and payload rate while changing only the feedback
multiplier. A second two-by-two experiment crosses feedback on or off with a
scale fitted at K2 or K3.

## The scale-selection confound

The production fixed-K2 configuration deliberately retains `K=3` during
regularization and global-scale search because its allocation has a three-bit
average rate. The configuration is set in
[`qsrt/pack/qsrt_encoder.py`](../qsrt/pack/qsrt_encoder.py#L489).

Global-scale search quantizes sampled, direct pre-feedback tiles using the
unchanged `quant_args["K"]`. The decisive call is in
[`qsrt/exl3_encoder_backend.py`](../qsrt/exl3_encoder_backend.py#L1940). The
batched preparation path supplies the first group's unchanged quantization
arguments to that search in
[`qsrt/exl3_encoder_backend.py`](../qsrt/exl3_encoder_backend.py#L2645).

Mixed-codebook dispatch replaces `K` with the selected codebook's actual width
only after scale selection. The substitution occurs in
[`qsrt/exl3_encoder_backend.py`](../qsrt/exl3_encoder_backend.py#L593) and
reaches traversal through the batched encoder near
[`qsrt/exl3_encoder_backend.py`](../qsrt/exl3_encoder_backend.py#L1115).

The resulting production path can encode with a K2 trellis while using a scale
chosen under K3 quantization. A direct call configured as K2 can instead fit a
K2-specific scale. Such a comparison changes at least two variables:

- whether BlockLDLQ feedback changes later targets; and
- which rate is used to select the global reconstruction scale.

A causal feedback comparison must make `g_scale`, `suh`, and `svh`
byte-identical across its two arms. A separate experiment can then determine
whether K2-native scale selection is itself beneficial.

Scale fitting has another asymmetry. The production scale objective sees
direct weights before BlockLDLQ feedback. It therefore matches the direct
Viterbi arm more closely than the feedback-adjusted targets. A
feedback-aware complete-encode scale sweep is warranted if K2-native scale
selection changes the result.

## Why final quadratic loss does not prove implementation correctness

QSRT evaluates input-covariance-weighted reconstruction error as

\[
\operatorname{tr}(E^\mathsf{T} H E),
\]

where columns of \(E\) are input-coordinate errors and \(H\) is the input
covariance. This is the same scalar as \(\operatorname{tr}(E H E^\mathsf{T})\)
under the conventional output-by-input matrix orientation. The implementation
computes the first form in
[`qsrt/exl3_encoder_backend.py`](../qsrt/exl3_encoder_backend.py#L1646).

BlockLDLQ factors \(H=L D L^\mathsf{T}\) and traverses coordinate blocks in
reverse. The factorization, compensation direction, and reverse traversal are
implemented near these locations:

- [factorization](../qsrt/exl3_encoder_backend.py#L316);
- [reverse traversal and compensation](../qsrt/exl3_encoder_backend.py#L445);
- [batched traversal](../qsrt/exl3_encoder_backend.py#L1077); and
- [FP64 algebraic reference](../qsrt/blockldlq_proof.py#L189).

Two properties prevent the implemented procedure from being a global minimizer
of the final quadratic loss:

1. The Viterbi search minimizes ordinary squared error within each
   256-position tile. It does not place the 16-by-16 block of \(D\) in each
   trellis edge cost. The codec specification states this distinction in
   [`docs/qsrt-2bpw-codec.md`](qsrt-2bpw-codec.md#L442).
2. Reverse error feedback is greedy. A decision that reduces the transformed
   residual for the coordinate being processed can make a later unprocessed
   coordinate's residual larger.

Consequently, a feedback encode can have higher final
\(\operatorname{tr}(E^\mathsf{T} H E)\) than a direct encode even when the
factorization, sign, transpose, traversal order, and arithmetic are correct.
Higher loss remains a reason to investigate. It is not, by itself, proof of an
implementation defect.

### Reproducible three-scalar counterexample

Use one output row with source weights

```text
w = [
  0.1790580742138147,
  0.3024845217623837,
  0.8050946675062194
]
```

and the binary scalar quantizer

```text
Q(x) = 0 when x < 0.5
       1 otherwise.
```

Define \(H=L D L^\mathsf{T}\) with

```text
L = [
  [1.0,                 0.0,                  0.0],
  [1.57839062759425,    1.0,                  0.0],
  [1.0710418696724187, -1.0162027705538617,  1.0]
]

D = diag([
  0.7651626108767694,
  0.39227003911150926,
  4.894930875920982
])
```

This gives

```text
H = [
  [0.7651626108767694, 1.2077254935934389, 0.8195211933568845],
  [1.2077254935934389, 2.2985326389060328, 0.8948986701589741],
  [0.8195211933568845, 0.8948986701589741, 6.1777571316439435]
]
```

The eigenvalues are positive: `0.0866208143`, `2.5795976509`, and
`6.5752339162`. The covariance is therefore positive definite.

Direct quantization produces `q_direct = [0, 0, 1]` and quadratic loss
`0.4376280520`. Reverse greedy feedback uses

\[
t_i=w_i+\sum_{j>i}L_{j,i}(w_j-q_j)
\]

and visits coordinates `2, 1, 0`. It produces targets `0.8050946675`,
`0.5005478606`, and `-1.1306455910`, yielding
`q_feedback = [0, 1, 1]` and loss `1.2619546424`. Feedback raises the final
quadratic objective by a factor of about `2.884` without an algebra error.

The following snippet reproduces the calculation:

```python
import torch

torch.set_default_dtype(torch.float64)

w = torch.tensor([
    0.1790580742138147,
    0.3024845217623837,
    0.8050946675062194,
])
L = torch.tensor([
    [1.0,                 0.0,                  0.0],
    [1.57839062759425,    1.0,                  0.0],
    [1.0710418696724187, -1.0162027705538617,  1.0],
])
D = torch.diag(torch.tensor([
    0.7651626108767694,
    0.39227003911150926,
    4.894930875920982,
]))
H = L @ D @ L.T

def quantize(value):
    return (value >= 0.5).to(value.dtype)

q_direct = quantize(w)
q_feedback = torch.empty_like(w)
for index in range(w.numel() - 1, -1, -1):
    target = w[index] + L[index + 1:, index] @ (
        w[index + 1:] - q_feedback[index + 1:]
    )
    q_feedback[index] = quantize(target)

def evaluate(reconstruction):
    error = w - reconstruction
    return error @ H @ error, L.T @ error

print(torch.linalg.eigvalsh(H))
print("direct", q_direct, *evaluate(q_direct))
print("feedback", q_feedback, *evaluate(q_feedback))
```

## Audit of the proposed explanations

### Measurement artifact is a serious prior

The repository records `0.0851995464` mean KLD for one 32-window,
tensor-parallel-eight, MXFP8-overlay configuration and `0.0653855355` for a
1,024-context, tensor-parallel-16, BF16-overlay configuration of the same
routed-expert payload. These values are not a paired comparison. They show why
configuration identity is part of the measurement, as documented in
[`docs/qsrt-exl3-comparative-assessment.md`](qsrt-exl3-comparative-assessment.md#L268).

A paired 1,024-context receipt attributes `-0.0016353402` KLD to the non-expert
overlay alone, with interval `[-0.0017714323, -0.0015119850]`. A 32-window
uncertainty of about `±0.04` is only an extrapolation from other uncertainty
measurements; the repository does not contain a committed 32-window confidence
interval and must not describe that value as a measured receipt.

The repository contains no measured EXL3 full-model KLD that supports an
expected range of `0.01` to `0.015`. Any such range is speculation rather than
an acceptance baseline.

### A feedback implementation defect remains possible

The algebraic sign, transpose, and reverse order appear consistent in the
inspected implementation. That review does not close CUDA arithmetic,
half-precision accumulated costs, batching, scale interaction, or saturation.
A defect requires a stronger witness than higher final quadratic loss:

- failure to reconstruct \(H=L D L^\mathsf{T}\);
- failure of the encoded targets to match the documented recurrence;
- disagreement between CUDA and a full-precision reference under identical
  targets and scale; or
- path divergence that begins when individual or accumulated Viterbi costs
  clamp.

The encoder warns that candidate-prefix reuse is not a bit-exact batching
contract. That warning does not directly implicate a one-candidate uniform-K2
encode, which needs no cross-candidate prefix reuse. Batch shape and
TensorFloat-32 policy remain valid controls because they can perturb hard path
decisions.

The Cholesky retry loop does not silently double `sigma_reg`. Each failure
prints a warning and adds twice the configured regularization times the
current diagonal mean. It permits ten retries. The effective final damping is
not recorded as structured artifact provenance and should be added to a causal
experiment.

Commit `27d4894` fixes a real trellis-closure defect when half-precision path
costs overflow, but its regression fixture uses target magnitudes of `1000`.
No production receipt shows that Kimi K2 encountered this condition. Commit
`8b0ccf0` adds worker retries and incomplete-build provenance; it does not
identify a BlockLDLQ arithmetic defect.

### Bad covariance can make feedback harmful

The covariance policy differs by projection. Gate and up ordinarily use a
layer-global input covariance. Down uses expert-local routed rows, gate-squared
weights, and adaptive shrinkage with at most `0.75` local weight. A statement
that all three projections use gate-squared expert-local covariance is false.

Canonical identity covariance also does not necessarily remove production
feedback. Nonuniform input scaling followed by the Hadamard congruence can
turn identity in the source basis into a dense covariance in the work basis.
A no-feedback experiment must explicitly set the feedback multiplier to zero
or call the direct Viterbi path.

If feedback wins on the fitted covariance but loses on a document-disjoint
covariance or full-model KLD, the result supports covariance or objective
mismatch. It does not support an arithmetic-defect conclusion.

## Pre-registered causal experiment

### Isolate feedback first

Encode one complete real expert through two arms:

| Arm | Scale plane | Feedback multiplier |
|---|---|---:|
| Production-feedback control | Frozen production `g_scale`, `suh`, and `svh` | 1 |
| Direct-Viterbi comparison | Byte-identical frozen scale plane | 0 |

Both arms must share:

- source tensor paths and hashes;
- captured covariance and final damped-covariance hash;
- coupled transform, draw, signs, and permutation;
- traversal order and batch shape;
- K2 graph, reconstruction table, tail-biting context, and decoder;
- TensorFloat-32 policy; and
- exact serialized byte count.

Only the feedback multiplier and resulting trellis path bytes may differ. The
clean implementation supplies a per-member feedback multiplier to the same
batched traversal. Forcing the emergency `q_fallback` path is invalid because
it changes Hessian finalization and regularization control flow.

### Separate the scale effect

Run the following factorial after the feedback-only comparison:

| Feedback | Frozen K3-fitted scale | Frozen K2-fitted scale |
|---:|---:|---:|
| 1 | Production control | K2-scale effect under feedback |
| 0 | Isolated feedback removal | Described simple K2 encoder |

The scale is frozen within each feedback pair. If the K2-fitted scale helps,
follow with a scale sweep that scores complete feedback encodes rather than
direct pre-feedback sample tiles.

### Required diagnostics

1. Verify production targets against the reverse feedback recurrence and
   verify the covariance congruence plus \(L D L^\mathsf{T}\) reconstruction.
2. Compare sampled CUDA paths against a full-precision Viterbi reference.
3. Count individual branch-cost clamps and accumulated path-cost clamps.
4. Record the final Cholesky damping and score both the damped covariance used
   for encoding and the undamped captured covariance.
5. Measure fit covariance loss, document-disjoint covariance loss, complete
   routed expert-output error, and complete-expert error on a committed
   depth- and support-stratified panel.
6. Advance to full-model KLD only after local recurrence and decoder closure.

### Interpretation

- A recurrence, congruence, or reference-Viterbi failure is an implementation
  defect.
- CUDA divergence that begins only when costs clamp is a numerical range
  defect.
- A no-feedback gain confined to the K2-fitted scale is scale-search
  confounding.
- Feedback losing the fitted covariance after the algebra and path checks pass
  exposes greedy assignment or the unweighted tile objective.
- Feedback winning fitted covariance but losing document-disjoint covariance
  indicates covariance estimation or distribution mismatch.
- Feedback winning linear metrics but losing complete-expert error indicates
  that the one-matrix objective misses gate/up/down interaction.
- A local expert gain that loses full-model KLD indicates downstream
  sensitivity, co-routing, or model-objective mismatch.
- A no-feedback arm that wins repeated paired KLD on an untouched suite and a
  second transform seed establishes a real improvement under the tested
  format. It still motivates damped-feedback and feedback-aware scale controls
  before permanent removal.

## GLM-5.2 frozen-scale K3 measurement

A subsequent GLM-5.2 layer-3 experiment applied the feedback-only part of the
design above to eight complete experts. For every gate, up, and down
projection, the encoder first reproduced the qualified uniform-K3 endpoint
with its recorded global scale. It then set the feedback multiplier to zero
while retaining the source tensor, transforms, K3 graph, T12 table, global
scale, and persisted `suh` and `svh` scale-vector hashes.

BlockLDLQ changed the floating-point targets presented to the trellis but did
not change any of the 24 selected paths. The zero-feedback reconstruction of
every projection was therefore identical, and all eight complete dense expert
files were byte-identical to the uniform-K3 inputs. The independent identity
report is
`results/glm52-layer3-frozen8-blockldlq-no-feedback-frozen-k3-scale-identity-comparison/report.json`
below the indexed kossel experiment root; its SHA-256 is
`026369914fe0e1e9bf868cc21a77d9df19b07a1afb6358545432876661a10153`.

This result rules out BlockLDLQ as a hidden source of KLD differences for that
identity-metric K3 panel: byte-identical endpoints necessarily inherit the
same measured full-model KLD. It does not resolve the original Kimi-K3 K2
claim. K2 uses larger reconstruction errors, and the claimed comparison still
needs the frozen K2/K3 scale factorial described above. A captured dense
curvature metric can also produce larger feedback terms than the identity
control and must be evaluated separately.

## Evidence boundary

The K2 claim remains a code-and-report audit: no causally isolated Kimi-K3 K2
candidate or paired KLD measurement has been supplied. The GLM-5.2 result is a
real GPU encode of eight layer-3 K3 experts with frozen scales, but its identity
input metric and single layer do not establish behavior at K2, under captured
dense curvature, or across a complete model. Neither investigation downloaded
a model or produced a raster image.

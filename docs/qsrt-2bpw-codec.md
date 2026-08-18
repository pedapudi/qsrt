# QSRT two-bit expert-weight codec

## Scope, authority, and implementation status

QSRT is the Quantile-Stratified Rate-shifted Trellis codec used to compress the
routed-expert matrices in Kimi-K3. This document defines the uniform two-bit
reconstruction, transform, covariance, and storage contract. The
[QSRT technical brief](qsrt-technical-brief.md) defines the complete system,
including mixed-rate allocation and the high-quality endpoint. The
[two-bit quality research corpus](qsrt-two-bit-research-corpus.md) separates
implemented behavior from research proposals. The
[interactive trellis-path explainer](viterbi-trellis-explainer.html) provides a
visual introduction to the trellis graph.

As of 2026-08-10, the uniform two-bit encoder, reference decoder, atom layout,
and complete 82,432-expert artifact are implemented. The artifact has passed a
full-model numerical run with eight-way tensor parallelism and 16-bit
activations. Multimodal execution, task quality, production latency, and the
intended twelve-way tensor-parallel runtime still require qualification. The
modules and tests listed under [Code that enforces the specification](#code-that-enforces-the-specification)
bind the serialized format.

The two-bit profile encodes every expert weight with two path bits, plus a
small set of block scales and one transform identifier per expert.

Each Kimi-K3 routed expert contains three matrices:

$$
W_1,W_3\in\mathbb R^{3072\times3584},
\qquad
W_2\in\mathbb R^{3584\times3072}.
$$

For a residual row $z$, the expert computes

$$
g=zW_1^{\mathsf T},
\qquad
u=zW_3^{\mathsf T},
\qquad
h=\mathrm{SiTU}(g,u),
\qquad
y=hW_2^{\mathsf T}.
$$

The coordinatewise activation is

$$
\mathrm{SiTU}(g,u)
=\left[4\tanh(g/4)\,\sigma(g)\right]
 \odot
 \left[25\tanh(u/25)\right].
$$

It combines corresponding gate and up coordinates independently for each of
the 3,072 intermediate neurons.

The codec does not round these matrix coefficients independently. It encodes
each coefficient sequence as a path through a finite-state graph. Two new bits
are chosen at every step. The previous fourteen path bits determine the four
reconstruction values currently available and which four sets of values can
be reached next. An offline dynamic-programming algorithm called Viterbi
search minimizes the sum of scalar squared errors over each 256-coefficient
tile. A surrounding covariance-aware error-feedback procedure adjusts later
coefficient blocks to compensate for earlier reconstruction errors in
activation-sensitive directions.

The terms used elsewhere in the repository follow directly from this
construction:

- `K2` means that each trellis step adds two payload bits and therefore has
  four outgoing branches;
- `L16` means that the current edge is identified by a sixteen-bit window,
  consisting of fourteen retained history bits and two new branch bits; and
- a `K2 record` is a 128-neuron region whose coefficients all use this
  two-bit path law.

The uniform two-bit profile assigns K2 to all 24 records of every routed
expert. It does not use higher-rate records or higher-rate experts. The rate
table reports bits per weight (bpw).

```text
weights per expert                33,030,144
trellis payload per weight        2.0 bits
trellis payload per expert        8,257,536 bytes
stored scales per expert             18,432 bytes
total payload per expert          8,275,968 bytes
total rate including scales        2.004464 bpw
routed experts                       82,432
```

The two-bit representation combines four mechanisms:

1. history-dependent four-way sequence quantization;
2. Gaussian-quantile reconstruction values rounded to an eight-bit
   floating-point alphabet with four exponent bits and three fraction bits
   (E4M3);
3. an exact coupled Hadamard change of basis across all three matrices; and
4. covariance-aware error feedback evaluated on the reconstructed expert.

## How QSRT builds on QTIP and EXL3

QSRT uses the quantization architecture established by Quantization with
Trellises and Incoherence Processing
([QTIP](https://arxiv.org/abs/2406.11235)) and
[EXL3](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md).
It is not an independent reinvention of their trellis encoder.

[QTIP](https://github.com/Cornell-RelaxML/qtip) supplies the fundamental
combination of high-dimensional trellis-coded quantization, the
hardware-efficient bitshift trellis, procedural reconstruction codes, and
orthogonal mixing that spreads concentrated values before quantization
(incoherence processing). Its separation of the trellis memory $L$ from the
per-step rate $K$ is the basis of the `L16` and `K2` terminology used here.

EXL3 is a streamlined QTIP variant that retains procedural codebooks and
tail-biting trellis optimization while providing a practical GPU encoder,
blockwise signed-Hadamard regularization, scale conditioning, and packed
tensor representation. QSRT's offline encoder is derived from that EXL3
implementation and uses the unmodified ExLlamaV3 extension and Hadamard tensor
utilities as dependencies.

QSRT adds a model-specific labelled graph and finite-E4M3 reconstruction law,
an exact expert-level transform that couples the gate, up, and down
projections, reconstructed-upstream conditioning for the down projection, and
a tensor-parallel-independent mixture-of-experts (MoE) payload. These
additions change the graph, reconstruction labels, and expert-level
calibration objective. The bitshift trellis and incoherence methods retain
their QTIP and EXL3 lineage.

## History-dependent sequence quantization

### Four choices with fourteen bits of memory

At one encoding step, let $h$ be the fourteen retained path bits and let
$b\in\{0,1,2,3\}$ be the two new bits. The pair $(h,b)$ identifies one of
$2^{16}=65,536$ directed edges.

Every history exposes four reconstruction candidates. Their unrounded ranks
come from four equal-probability regions of a Gaussian distribution:

```text
stratum 0    one rank from probability interval [0, 1/4)
stratum 1    one rank from probability interval [1/4, 1/2)
stratum 2    one rank from probability interval [1/2, 3/4)
stratum 3    one rank from probability interval [3/4, 1)
```

The physical two-bit branch does not have a fixed stratum. The current history
selects a permutation that maps the four branch values to the four strata. The
history also selects the exact rank within each stratum. Before finite
floating-point rounding, the four candidates cover the lower tail, lower
center, upper center, and upper tail. Ranks near the median can round to the
same zero label, while the lower-tail and upper-tail candidates retain
opposite signs.

Selecting a branch also changes the next fourteen-bit history. Two edges with
the same numerical value can remain useful alternatives because they lead to
different future states. The quantizer is consequently richer than its scalar
reconstruction alphabet.

This labelled finite-state construction is the Stratified Quantile Graph
(SQG).

### Edge-to-quantile assignment

The graph assigns every directed edge a unique integer rank from 0 through
65,535. Those ranks are ordered from the most negative to the most positive
reconstruction region.

For a sixteen-bit edge word $t$, split the retained history and new branch:

$$
h=t\gg2,
\qquad
b=t\mathbin{\mathrm{AND}}3.
$$

With $M=2^{14}-1$, compute

$$
\begin{aligned}
x_0 &= h\oplus(h\gg11),\\
x_1 &= x_0\oplus((x_0\ll11)\mathbin{\mathrm{AND}}M),\\
p &= (\mathtt{0x3FA7D929}\,x_1+
      \mathtt{0xC928FD8E})\bmod2^{32},\\
\phi &= p\mathbin{\mathrm{AND}}M,\\
s &= p\gg30,\\
j &= \mathrm{rev}_2(b)\oplus s,\\
r &= (j\ll14)\mathbin{\mathrm{OR}}\phi.
\end{aligned}
$$

Here $j\in\{0,1,2,3\}$ selects the probability stratum and
$\phi\in\{0,\ldots,16383\}$ selects the position within that stratum.

Both exclusive-or shifts are invertible on fourteen-bit integers. The
multiplier is odd and is therefore invertible modulo $2^{14}$. It follows that
$h\mapsto\phi$ is a permutation. For fixed history, $b\mapsto j$ is also a
permutation.

The construction therefore guarantees:

- every one of the 65,536 ranks is assigned to exactly one directed edge;
- every state exposes one candidate from each probability stratum; and
- all four edges from a state share one fine position within their respective
  strata.

The integer carries and invertible shifts also control how a chosen branch
changes the next fine position. This continuation structure is part of the
code. Matching the one-step value distribution without matching useful future
states produces materially worse quantization.

### Closed-path encoding

The encoder uses tail-biting Viterbi search: the final state must close onto
the initial state. This removes an arbitrary start-state penalty and makes the
stored bitstream a closed finite-state path.

Each 256-coefficient tile is self-contained. With the production context of
128, an unconstrained primer starts halfway through the tile and traverses the
tile's own 256 symbols in cyclic order. A traceback through that primer chooses
one candidate state at the boundary before coefficient zero. The encoder then
runs a complete 256-step Viterbi pass constrained to start and end at that
state. Smaller primer contexts use fewer symbols around the same cyclic
boundary and are encoder-only search approximations. Neither case reads
coefficients from neighboring tiles or changes the decoder.

## Gaussian reconstruction values in finite E4M3

The rank assignment defines probability regions. A separate reconstruction
table assigns a floating-point value to each rank. For rank $r$, the ideal
value is based on the midpoint of its equal-probability Gaussian interval:

$$
u_r=\frac{r+\tfrac12}{65536},
\qquad
z_r=\Phi^{-1}(u_r).
$$

The reconstruction value is scaled and rounded to the nearest finite E4M3
number, with ties resolved toward the even significand (RNE):

$$
Y(r)=\mathrm{RNE}_{\mathrm{E4M3FN}}(1.5z_r).
$$

E4M3 is an eight-bit floating-point reconstruction alphabet with one sign bit,
four exponent bits, and three fraction bits. The decoder produces these
eight-bit values from a path that stores two bits per weight.

The normal quantile law gives every state broad sign and magnitude coverage.
Rounding creates repeated numerical labels, but repeated labels on different
edges retain different successors and therefore remain distinct coding
choices.

### Building the finite staircase offline

Table generation evaluates the Gaussian quantile function directly at all
65,536 midpoint probabilities. The implementation computes
`1.5 * torch.special.ndtri((rank + 0.5) / 65536)`, rounds the results to finite
E4M3 with round-to-nearest-even, and converts both signed-zero encodings to the
same zero byte. The encoder and decoder use the frozen byte table, so neither
evaluates a Gaussian quantile function while processing weights.

The immutable codebook identifier is `sqg_xor_cheb_t12`. Its reconstruction
law is the direct Gaussian-midpoint calculation above; no Chebyshev polynomial
defines or evaluates the table. `tests/test_sqg_e4m3.py` fixes the table and
rate-specific label maps with SHA-256 hashes.

### Shared 4,096-byte staircase

The exact 65,536-rank staircase is reduced to one globally shared 4,096-byte
table. For table index $q$,

$$
Y_{12}(q)=
\mathrm{mode}
\{Y(16q),Y(16q+1),\ldots,Y(16q+15)\},
$$

with the lower unsigned E4M3 byte selected on a tie. The reconstructed value
for an edge is

$$
\widehat Y(h,b)=Y_{12}\!\left(r(h,b)\gg4\right).
$$

The table index uses the upper twelve bits of the full sixteen-bit rank. The
repository therefore calls it the twelve-bit, or `T12`, staircase. One
4,096-byte codebook serves the entire model, while each weight still stores
two path bits.

The approximation changes only the final reconstruction label. It retains the
complete edge permutation, four-way stratification, state transitions, and
Viterbi path search. Frozen hashes identify the shared table and the resulting
65,536-edge K2 label map.

## Exact coupled Hadamard conditioning

Two-bit quantization is highly sensitive to outliers and unequal coordinate
scales. Orthogonal Hadamard transforms spread concentrated energy across a
block, making the sequence presented to the quantizer more homogeneous.

Independent matrix-local transforms are valid only when each transform is
cancelled at that matrix's immediate input or output. A change of basis that
conditions the three-matrix expert as one function must also respect the
nonlinear activation between gate/up and down. QSRT therefore uses a coupled
change of basis whose transforms are explicitly cancelled on the correct side
of the activation.

### Relationship to QTIP and EXL3 incoherence processing

The ordinary Hadamard rotations used by QTIP and EXL3 are matrix-local
incoherence transforms. For one linear map with weight $W$, an input-side and
an output-side orthogonal transform spread the coefficients of that matrix.
Writing those transforms as $R_{\mathrm{in}}$ and $R_{\mathrm{out}}$, the
standard construction has the form

$$
\widetilde W
=R_{\mathrm{out}}WR_{\mathrm{in}}^{\mathsf T},
\qquad
\widetilde x=xR_{\mathrm{in}}^{\mathsf T},
\qquad
\widetilde x\widetilde W^{\mathsf T}
=(xW^{\mathsf T})R_{\mathrm{out}}^{\mathsf T}.
$$

Applying $R_{\mathrm{out}}$ restores the original output. Each linear
operator's coordinate change therefore closes locally. This regularization
makes each matrix more homogeneous for its scalar or trellis codebook, but it
does not by itself create a joint gate/up coordinate system or bind that
system to the down projection.

QSRT retains this matrix-local EXL3 regularization inside each trellis encode.
The coupled Hadamard described here is an additional, expert-function-level
reparameterization applied before that regularization. It differs in three
structural ways:

- gate and up rows are interleaved and rotated as one 6,144-coordinate
  preactivation vector, so the stored upstream halves are not independently
  rotated gate and up matrices;
- the preactivation rotation is inverted before gate and up are separated and
  before SiTU is evaluated, because a general orthogonal transform does not
  commute with the coordinatewise nonlinearity; and
- a separate post-SiTU rotation is applied to the hidden vector and matched by
  the input side of $W_2$, while one model-wide residual transform is matched
  across the inputs of $W_1/W_3$ and the output of $W_2$.

The distinction is therefore one of scope. QTIP/EXL3-style rotations condition
individual linear operators. The coupled transform conditions the complete
three-matrix expert while preserving its unquantized nonlinear function
exactly. Reconstructed whole-expert error selects the transform identifier;
individual matrix error does not make that choice.

### Transforming and recovering an expert

Interleave corresponding gate and up rows:

$$
Q_e=\mathrm{interleave}(W_{1,e},W_{3,e})
\in\mathbb R^{6144\times3584}.
$$

Define three orthogonal block transforms:

- $U_R$: normalized 512-coordinate Hadamard blocks on the residual input and
  expert-output axes, with one fixed all-ones sign vector across the model;
- $U_{A,e}$: normalized signed 128-coordinate Hadamard blocks across the
  interleaved gate/up preactivations of expert $e$; and
- $U_{B,e}$: normalized signed 128-coordinate Hadamard blocks across the
  post-activation intermediate coordinates of expert $e$.

The matrices given to the two-bit encoder are

$$
Q'_e=U_{A,e}^{\mathsf T}Q_eU_R,
\qquad
W'_{2,e}=U_R^{\mathsf T}W_{2,e}U_{B,e}.
$$

For a row vector,

$$
z'=zU_R,
\qquad
q'=z'Q_e'^{\mathsf T}=(zQ_e^{\mathsf T})U_{A,e}.
$$

The inverse of $U_{A,e}$ is applied before the preactivations are split into
gate and up vectors and passed through SiTU. The resulting hidden vector is
then transformed by $U_{B,e}$ before multiplication by the transformed down
matrix. Orthogonality gives

$$
y'_e=(hU_{B,e})W_{2,e}'^{\mathsf T}
     =(hW_{2,e}^{\mathsf T})U_R,
$$

followed by

$$
y_e=y'_eU_R^{\mathsf T}=hW_{2,e}^{\mathsf T}.
$$

The transformation is therefore exactly function-preserving before
quantization. Its only purpose is to present a better-conditioned coordinate
system to the lossy encoder.

### Expert-specific signed transforms

The implementation calls each deterministic sign-pattern choice a draw. The
residual-axis transform always uses draw zero. Draw zero produces an
all-ones sign vector, so $U_R$ is the same unsigned block-Hadamard transform in
every layer and expert. The artifact stores no residual draw or per-layer sign
data.

The two intermediate-axis transforms use deterministic sign patterns selected
separately for each expert. One three-bit identifier generates the two required
sign vectors. The artifact regenerates those vectors instead of storing them.

The format defines eight possible identifiers. The materialized model uses two
validated choices:

```text
transform identifier 0      60,277 experts
transform identifier 6      22,155 experts
logical identifier width     3 bits per expert
serialized format entry      1 byte per expert
```

For a given expert, a fit split proposes identifier six as an alternative to
identifier zero. A document-disjoint confirmation split can accept that
proposal or retain identifier zero. The confirmation split does not search the
remaining identifiers.

The selected transform affects every subsequent encoding operation: transformed
weights, scale fitting, trellis paths, activation covariances, and reconstructed
expert error are all recomputed for that candidate.

## Covariance-aware error feedback

For a linear matrix with input rows $X$, the local activation metric uses the
dense covariance $H=X^{\mathsf T}X$. The encoder factors this matrix as
$LDL^{\mathsf T}$ and quantizes coefficient blocks in an order that feeds each
block's reconstruction error into later blocks. The repository calls the
scalar construction LDLQ and its blockwise form BlockLDLQ. The method lets
later coefficients compensate for earlier errors in directions exercised by
routed activations.

The Viterbi kernel remains a scalar path solver. For the adjusted 256-value
tile that BlockLDLQ supplies, Viterbi accumulates an unweighted squared error
$(w-\widehat w)^2$ on every transition. Dense activation weighting enters
through the surrounding BlockLDLQ pre-compensation and feedback, rather than
through a weighted branch cost inside Viterbi.

QSRT inherits this $LDL^{\mathsf T}$ error-feedback construction through
[QuIP#](https://arxiv.org/abs/2402.04396), QTIP, and EXL3. QSRT changes how the
expert covariances are formed and conditioned, especially for the down
projection. The underlying blockwise feedback rule remains unchanged.

Every candidate receives its own fitted scales and a complete Viterbi and
BlockLDLQ encode. Reconstructed whole-expert output error selects the final
payload rather than the Viterbi tile cost.

### Gate and up covariance

All experts in a layer consume the same residual coordinate system. A common
gate/up input covariance can therefore be accumulated for the layer and
transformed into the residual Hadamard basis. The repository names this matrix
`H13` because it conditions the first and third expert projections, $W_1$ and
$W_3$.

### Down covariance conditioned on reconstructed gate and up

The input to $W_2$ is produced by the quantized gate and up matrices. It is
also expressed in an expert-specific Hadamard basis. Its covariance cannot be
pooled across experts or borrowed from the unquantized source matrices.

For each candidate transform, the encoder therefore:

1. transforms and two-bit encodes $W_1$ and $W_3$;
2. decodes their exact stored reconstruction;
3. replays naturally routed inputs through those reconstructed matrices;
4. cancels the preactivation transform and evaluates SiTU;
5. applies the candidate's post-activation transform;
6. accumulates the resulting expert-specific covariance for $W_2$; and
7. performs a complete two-bit BlockLDLQ encode of $W_2$.

The repository calls this covariance `H2`. With intermediate dimension
$d=3072$, its regularized estimate is

$$
\widehat H_{2,e}
=\alpha_e H^{\mathrm{sample}}_{2,e}
+(1-\alpha_e)
\frac{\mathrm{tr}(H^{\mathrm{sample}}_{2,e})}{d}I.
$$

The sample covariance uses the squared applied router coefficient as the row
weight. Let those nonnegative weights be $w_i$, and define

$$
n_{\mathrm{eff}}=\frac{(\sum_i w_i)^2}{\sum_i w_i^2},
\qquad
\mu=\frac{\mathrm{tr}(H^{\mathrm{sample}}_{2,e})}{d},
\qquad
m_2=\frac{\lVert H^{\mathrm{sample}}_{2,e}\rVert_F^2}{d^2}.
$$

The implementation evaluates a weighted Oracle Approximating Shrinkage
estimate

$$
\lambda_e=
\mathrm{clip}_{[0,1]}
\left(
\frac{m_2+\mu^2}
{(n_{\mathrm{eff}}+1)(m_2-\mu^2/d)}
\right),
\qquad
\alpha_e=\min(0.75,1-\lambda_e).
$$

If the denominator is nonpositive, the implementation sets $\lambda_e=1$.
The identity term stabilizes poorly sampled directions without importing a
coordinate system from another expert. Experts without sufficient routed
support use the unscaled identity matrix because no local trace estimate is
available. A layer-wide post-activation covariance is never used.

### Gradient-conditioned Viterbi proposals

Status: research-only and not implemented in the production encoder. This
facility changes offline path selection only; it does not change the two-bit
payload, reconstruction law, or serving decoder.

The gradient interface is defined relative to an explicitly identified anchor
model, not to a particular QSRT profile. Let $\widehat W_A$ be the decoded
matrix at anchor $A$, let $L_A$ be a scalar model objective evaluated at that
anchor, and let

$$
g_A=\left.\frac{\partial L}{\partial \widehat W}\right|_A.
$$

For a legal trellis path $q$, define its decoded displacement from the anchor
as

$$
d(q)=\widehat W(q)-\widehat W_A.
$$

A fixed gradient supplies the linear proposal objective

$$
J_{\text{proposal}}(q)
=J_{\text{local}}(q)+\lambda\langle g_A,d(q)\rangle.
$$

The second term is additive over reconstructed coefficients. After conversion
to the trellis coordinate system, it can therefore be added directly to each
Viterbi edge cost. The anchor contribution is constant across paths and may be
omitted while finding the minimum.

The capture policy is deliberately not fixed by this interface. A useful
anchor may be a quantized checkpoint evaluated against a higher-quality
teacher, a partially rebuilt model, or a model evaluated under a task loss. If
the anchor and teacher are the same model and the objective is their KL
divergence, the deterministic gradient is zero and supplies no linear term.

The deterministic objective gradient is distinct from sampled score gradients
used to estimate Fisher curvature. Around an arbitrary anchor, a candidate
move is approximated by

$$
\Delta L
\mathrel{\approx}
\langle g_A,d\rangle
+\frac{1}{2}\mathrm{vec}(d)^{\mathsf T}
F_A\mathrm{vec}(d).
$$

For a two-sided Kronecker approximation,

$$
F_A\mathrel{\approx}H_I\otimes H_O,
$$

and the quadratic term is

$$
\frac{1}{2}\mathrm{tr}
\left(H_OdH_Id^{\mathsf T}\right).
$$

The linear term describes a repair direction conditional on the anchor's
aggregate errors. It is not an intrinsic per-weight importance score. The
two-sided term retains interactions that a diagonal Fisher penalty would
discard. Viterbi may use the linear term to generate legal paths, but complete
candidates must be rescored under the dense or two-sided objective and then
qualified by model-level KL divergence.

Gradient conversion must preserve the directional derivative through every
permutation, sign transform, Hadamard transform, scale, and matrix-orientation
change. The required closure is

$$
\langle g_{\text{source}},\delta W_{\text{source}}\rangle
=
\langle g_{\text{trellis}},\delta W_{\text{trellis}}\rangle.
$$

This must be verified on decoded candidate displacements before a gradient can
affect path selection. Transforming a gradient as though it were a weight is
valid only for an orthonormal map with no intervening scale; the general
implementation must apply the adjoint of the complete reconstruction map.

A reusable gradient artifact must identify:

- the anchor checkpoint and decoded payload hashes;
- the teacher, scalar objective, corpus, tokens, and reduction convention;
- the semantic activation point and routing-weight convention;
- the matrix name, orientation, transform plan, scales, and codebook identity;
- whether it contains a deterministic objective gradient or Fisher samples;
  and
- the numerical dtype, normalization, support, and finite-difference closure.

Gradient-guided path selection must retain the ordinary canonical-target path
as a fallback, enforce a bounded local-distortion increase, and validate the
selected payload on data not used to construct the gradient. A materially
changed anchor requires a fresh gradient before further directed updates.
Neither gradient nor curvature artifacts are serialized into the checkpoint.

## Whole-expert selection

The final candidate score reconstructs all three matrices, evaluates the
complete expert function on naturally routed rows, and weights output error by
the square of the router coefficient actually applied to that expert.

This objective captures interactions that independent matrix or weight errors
miss:

- gate and up errors interact through SiTU;
- their reconstructed activations determine the correct down covariance;
- the Hadamard candidate changes all three encoded matrices; and
- dense error feedback can move error between coefficient blocks.

The fit and confirmation splits contain disjoint source documents. A
source-controlled natural-routing capture supplies covariances and transform
proposals. The confirmation documents accept or reject each proposal. A
separate 128,000-token routed capture measures transfer. Final model
Kullback–Leibler divergence (KLD) and task evaluations do not select codec
parameters.

## Tensor-parallel-independent expert storage

The 3,072-neuron intermediate axis is divided into 24 records of 128 neurons.
Every record uses two path bits per coefficient. There is no rate bitmap and no
record- or tile-specific reconstruction table.

Canonical storage groups aligned fragments from $W_1$, $W_3$, and $W_2$ by
32 intermediate neurons. One such group is called an atom. Each expert
contributes 96 atoms, and each atom contains:

- the two-bit trellis payload for its gate fragment;
- the two-bit trellis payload for its up fragment;
- the two-bit trellis payload for its down fragment; and
- the corresponding half-precision scale fragments.

One atom occupies 86,208 bytes. Ninety-six atoms give the 8,275,968-byte
expert payload shown above. The 18,432 bytes above the exact two-bit weight
stream are the stored scales.

Atoms have complete ownership and the canonical file stores no tensor-parallel
degree. This permits the same encoded weights to be divided among different
numbers of devices without re-encoding. It does not alter the quantization
law.

Across 92 mixture-of-experts layers, the serialized expert container occupies
682,207,608,832 bytes. The atom payload in each layer is already divisible by
the 4,096-byte storage alignment, so this profile adds zero atom-row padding.
Each layer also has a fixed 32,768-byte prefix containing the safetensors
header, format section, and shared-scale section. The global 4,096-byte
reconstruction table and three-bit expert transform identifiers are negligible
at model scale.

## Evidence for the implemented coupled transform

`scripts/run_k3_coupled_codec_confirmation.py` implements the bounded
uniform-K2 comparison. Its control re-encodes each matrix with the shared SQG
T12 table, fitted scales, matrix-local regularization, and identity covariance.
The comparison arm adds the draw-zero coupled boundary transform at the same
payload rate. Across 24 official-source experts, the comparison arm reduced
pooled routed post-projection squared error by 3.052%, and 22 experts improved.

The intermediate-draw study covers 28 experts across seven layers. It uses
disjoint fit and confirmation documents, then measures the selected draws on
up to 256 rows per expert from the independent 128,000-token routed capture.
Relative to draw zero, expert-specific selection reduced pooled routed
post-projection error by a further 1.308%. Median expert error fell 0.447%, and
17 experts improved. A single transform for all experts regressed. Choosing
one transform per layer recovered 0.030%. The proposal logic is implemented in
`scripts/run_k3_coupled_expert_study.py`; production-table replay is implemented
in `scripts/run_k3_coupled_codec_confirmation.py`.

The complete representation contains all 82,432 routed experts and closes the
payload for all 92 mixture-of-experts layers. A full-model run used 16-bit
activations (A16) over 32 windows and 65,504 scored token positions. It measured
mean $\mathrm{KL}(\text{official microscaled four-bit floating-point reference}
\,\Vert\,\text{two-bit candidate})$ of 0.08520. The repository abbreviates the
source format as MXFP4. `scripts/run_k3_kld_capture.py` drives the run, and
`qsrt/kld_gate.py` pins the source-reference dataset, revision, token-suite
hash, and KLD direction.

The repository does not commit the exact command lines or aggregate result
manifests for the two expert panels above. The stated values therefore identify
recorded evidence rather than a checkout-only reproducibility guarantee. The
full-model KLD is an absolute divergence from the official MXFP4 source. It
does not establish a complete checkpoint that is both smaller than EXL3 and
lower in held-out KLD.

## Features included in the uniform two-bit profile

The two-bit representation contains:

- one history-dependent four-way trellis construction;
- one shared 4,096-byte Gaussian-derived E4M3 table;
- one fixed model-wide residual-axis Hadamard basis;
- one three-bit intermediate-axis Hadamard identifier per expert;
- candidate-specific scales and covariance-aware trellis paths; and
- one tensor-parallel-independent all-expert payload.

It does not contain mixed two-, three-, or four-bit records, a higher-rate
expert tier, tile-local codebooks, joint gate/up vector symbols, cross-expert
bases, or refinement bit planes.

## Code that enforces the specification

The codec is defined by:

- `qsrt/sqg_e4m3.py`: edge ranks and the shared E4M3 table;
- `qsrt/sqg_quantizer.py` and `qsrt/csrc`: closed-path Viterbi encoding;
- `qsrt/qsrt_coupled.py`: exact coupled Hadamard coordinates;
- `qsrt/qsrt_coupled_plan.py`: expert transform selection;
- `qsrt/candidate_hessian.py`: weighted covariance shrinkage;
- `qsrt/exl3_encoder_backend.py`: covariance-aware BlockLDLQ encoding; and
- `qsrt/qsrt_atoms_v2.py`: canonical payload and byte accounting.

The structural and numerical contracts are covered by
`tests/test_sqg_e4m3.py`, `tests/test_sqg_quantizer.py`,
`tests/test_qsrt_coupled.py`, `tests/test_candidate_hessian.py`, and
`tests/test_qsrt_high_rate_storage.py`.

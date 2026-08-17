# Kimi-K3-QSRT technical brief

Status, 2026-08-15: the fixed 3.083333-trellis-bpw and uniform two-bit all-QSRT
artifacts are sealed in tensor-parallel-independent containers. The uniform
two-bit artifact has passed a tensor-parallel-8 numerical and text-distribution
gate with 16-bit activations. Tensor-parallel-12 multimodal and latency
qualification remain incomplete. The mixed QSRT/X4T allocator is implemented,
but its all-expert X4T cost index, allocation, and materialized mixed artifact
are not sealed.

`QSRT` means **Quantile-Stratified Rate-shifted Trellis codec**. QSRT is a
fixed-payload trellis codec for gated mixture-of-experts (MoE) weights. Its
format decisions are expert-static: each expert keeps one decision across all
tokens. The Kimi-K3 construction combines four independently useful ideas:

1. a 16-bit-transition stratified-quantile graph that reconstructs finite
   E4M3 floating-point values;
2. equal-byte exchanges between two-bit and four-bit trellis records around a
   three-bit baseline, selected separately for fused gate/up weights and for
   down-projection weights;
3. an exact coupled coordinate transform around the nonlinear activation that
   makes a uniform two-bit expert profile more accurate; and
4. an exact high-quality endpoint, X4T, that preserves the official MXFP4
   four-bit values while compressing their per-block scale bytes.

The intended artifact name is `Kimi-K3-QSRT`. A mixed-tier artifact may combine
lossy QSRT experts with exact X4T experts chosen by an exact-byte global
allocator. The uniform two-bit and fixed 3.083333-trellis-bpw artifacts store every
routed expert as QSRT. Neither storage contract contains a raw-MXFP4 keep tier.

## Terms and document authority

Each trellis transition has a 16-bit identity, abbreviated **L16**. At rate
**K2**, **K3**, or **K4**, the encoder stores two, three, or four branch bits per
weight. The remaining 14, 13, or 12 bits are the history state carried between
weights. A **record** is 128 adjacent intermediate coordinates that share one
rate. A physical pair named `Pab` stores its first record at rate Ka and its
second at rate Kb. For example, `P24` exchanges a two-bit donor record for a
four-bit recipient record, while `P33` stores two baseline records.

Kimi-K3 has coupled gate and up projections followed by the coordinatewise
SiTU activation

$$
\operatorname{SiTU}(g,u)
=\left[4\tanh(g/4)\,\sigma(g)\right]
 \left[25\tanh(u/25)\right].
$$

The rate decision shared by the gate and up matrices is stored as `r13`; the
independent down-projection decision is stored as `r2`. Modes `R0`, `R1`, and
`R2` move zero, one, or two record pairs from `P33` to `P24`. An **atom** is the
32-intermediate-coordinate storage unit that keeps a physical record pair,
all three expert matrices, and their local scale fragments under one owner.
Tensor parallelism (TP) partitions complete atoms at load time and is absent
from the checkpoint format.

The **Stratified Quantile Graph (SQG)** is the bijective mapping from L16
transitions to probability ranks. The runtime rounds those ranks to finite
E4M3, an eight-bit floating-point format with one sign bit, four exponent bits,
and three fraction bits. The frozen 4,096-byte rank table is called the
12-bit execution table, or `T12`, because it indexes the upper 12 bits of each
16-bit rank. The identifier `sqg_xor_cheb_t12` is retained for format
compatibility; its `cheb` token does not denote a polynomial evaluator in the
current implementation.

**BlockLDLQ** is the blockwise LDLQ error-feedback encoder. Its dense
sensitivity matrix, written $H$, weights errors by captured activation
covariance. **W4A16** denotes four-bit stored weights evaluated with 16-bit
activations; **A16** is the shorter name for that activation mode. **B12X** is
the serving-kernel library used by the external runtime integration. The
resident calibration teacher is the running EXL3-formatted Kimi-K3 checkpoint
that supplies routed samples. It is distinct from the immutable official
MXFP4 weight source and from the official-MXFP4 logit reference used for model
quality measurements.

The [uniform two-bit codec specification](qsrt-2bpw-codec.md) is authoritative
for the two-bit mathematical and byte-level contract. This brief is
authoritative for Kimi-K3 artifact composition, storage-profile status, and
qualification evidence. The Python and CUDA implementations plus their tests
are the executable authority for serialized bytes and kernel launch geometry.
The [two-bit research corpus](qsrt-two-bit-research-corpus.md) separates
published techniques, local evidence, and unvalidated proposals. The
[interactive Viterbi explainer](viterbi-trellis-explainer.html) illustrates the
scalar path search, while the
[interactive proposals explainer](qsrt-three-improvements-infographic.html)
uses a synthetic benchmark and does not establish model quality.

## Frozen scope

The supported experiments are intentionally narrow:

```text
canonical storage              TP-independent 32-channel balanced atoms
first serving target           TP12; TP is not serialized in the codec
trellis window                 L16
reconstruction family         sqg_xor_cheb_t12 -> finite E4M3, round-to-nearest-even
lossy rate candidates          R0, R1, R2
gate/up decision               one shared r13
down-projection decision       independent r2
uniform low-rate profile       K2 on all 24 records, all routed experts
uniform-K2 conditioning        coupled 512/128-coordinate Hadamard transform
high-quality endpoint          exact X4T
source weights                 official Kimi-K3 MXFP4 checkpoint
calibration teacher            resident EXL3-formatted interim checkpoint
encoder objective              expert-stratified dense-H BlockLDLQ and routed replay
```

The current performance gate is TP12 because that is the primary Kimi-K3
deployment. The uniform-K2 artifact has also completed its first end-to-end
numerical gate at TP8. TP4, TP8, TP16, TP24, and TP32 are storage-valid
direct-load views of the same artifact and require kernel qualification, not
re-encoding.
Alternate companders, wider rate ladders, learned per-layer tables, and
entropy-coded hot streams remain outside the supported surface.

## Encoder ownership

This repository owns QSRT's offline implementation. The mixed-rate dense-H
BlockLDLQ backend is `qsrt/exl3_encoder_backend.py`; SQG label generation,
packed traceback, tail-biting Viterbi, and its CUDA sources live under
`qsrt/sqg_e4m3.py`, `qsrt/sqg_quantizer.py`, and `qsrt/csrc`.

The ExLlamaV3 checkout is an unmodified upstream dependency. It supplies only
the established EXL packing, Hadamard, and tensor utilities used by the
encoder. No QSRT format, rate-selection, SQG, LDLQ, or CUDA change may be
carried as a local ExLlamaV3 patch. The exact upstream-derived source retained
here is covered by `THIRD_PARTY_NOTICES.md`.

## QSRT-E4M3 reconstruction

QSRT's reconstruction mechanism is the **Stratified Quantile Graph (SQG)**.
SQG assigns the $2^L$ directed edges of an $L$-bit de Bruijn trellis
bijectively to equal-probability microcells of a reference distribution. At
rate $K$, each state retains $L-K$ history bits and has $2^K$ outgoing
branches. A history-dependent branch permutation selects one branch from each
of $2^K$ coarse quantile strata, while a bijective state permutation selects
the within-stratum phase. Consequently, every state exposes exactly one
reconstruction candidate from every stratum, and every global probability
rank occurs exactly once across the directed edge set.

The graph and scalar reconstruction law are separate design objects. For a
reference distribution with quantile function $F^{-1}$, microcell $r$ spans

$$
I_r = \left[\frac{r}{2^L},\frac{r+1}{2^L}\right),
$$

The production scalar law evaluates the Gaussian quantile at the midpoint of
each microcell:

$$
c_r=1.5\,\Phi^{-1}\!\left(\frac{r+\tfrac12}{65536}\right).
$$

It then projects $c_r$ with round-to-nearest-even to finite E4M3. Numerically
identical E4M3 labels may
remain on different directed edges and lead to different successors; scalar
label collisions therefore do not collapse the richer trellis geometry.

The public runtime profile is **QSRT-E4M3**. Its definition is the composition
of an SQG rank map and a finite reconstruction staircase:

$$
\text{codeword}
\xrightarrow{G_K}
r
\xrightarrow{Y_{12}}
\text{finite E4M3}.
$$

The graph $G_K$ and scalar law $Y_{12}$ are independent mathematical objects.
In particular, the carry-mixed graph does not approximate an inverse CDF or a
Chebyshev polynomial.

### Carry-mixed SQG rank map

For $L=16$, rate $K\in\{2,3,4\}$, and $w=16-K$, split a codeword $t$ into
history and physical branch:

$$
h=t\mathbin{\gg}K,
\qquad
b=t\mathbin{\mathrm{AND}}(2^K-1).
$$

With $M_w=2^w-1$, define

$$
\begin{aligned}
x_0 &= h\oplus(h\gg11),\\
x_1 &= x_0\oplus((x_0\ll11)\mathbin{\mathrm{AND}}M_w),\\
p &= (\mathtt{0x3FA7D929}\,x_1+\mathtt{0xC928FD8E})\bmod2^{32},\\
\phi &= p\mathbin{\mathrm{AND}}M_w,\\
s_K &= p\gg(32-K),\\
j &= \mathrm{rev}_K(b)\oplus s_K,\\
G_K(h,b)=r &= (j\ll w)\mathbin{\mathrm{OR}}\phi.
\end{aligned}
$$

Here $\mathrm{rev}_K$ reverses the $K$ branch bits. Both xorshifts are
triangular bijections on $w$ bits, and `0x3FA7D929` is odd, so multiplication
by it is invertible modulo $2^w$. Therefore $h\mapsto\phi$ is a permutation.
For fixed $h$, $b\mapsto j$ is also a permutation. It follows that

$$
(h,b)\longleftrightarrow(j,\phi)
$$

is a bijection over all $2^{16}$ directed edges. Every state has exactly one
outgoing branch in each of its $2^K$ strata, and every global rank occurs once.

The rank bijection does not by itself specify sequence behavior. If $P(h)=\phi$
is the phase permutation, $\pi_h(b)=j$ is the branch permutation, and $T$ is
the de Bruijn successor, then logical stratum $j$ induces the continuation map

$$
F_j(\phi)=P\!\left(
T\!\left(P^{-1}(\phi),\pi_{P^{-1}(\phi)}^{-1}(j)\right)
\right).
$$

The family $\{F_j\}$ is the branch-conditioned phase-transition geometry used
by Viterbi. It belongs to $G_K$. The scalar reconstruction law does not define
it.

### Gaussian-quantile finite staircase

The exact normal staircase is defined on every global rank $r$ by

$$
u_r=\frac{r+\tfrac12}{65536},
\qquad
z_r=\Phi^{-1}(u_r),
\qquad
Y(r)=\mathrm{RNE}_{\mathrm{E4M3FN}}(1.5z_r).
$$

The reference implementation evaluates the midpoint probabilities with
`torch.special.ndtri` and rounds the resulting values directly. It does not
fit or evaluate a Chebyshev polynomial. The frozen codebook identifier retains
the older `cheb` token, but the mathematical contract is the Gaussian-quantile
midpoint formula above.

### Twelve-bit execution staircase

QSRT-E4M3 compresses the exact staircase from 65,536 rank labels to 4,096
bytes. For $q\in\{0,\ldots,4095\}$, define

$$
Y_{12}(q)=
\mathrm{mode}\{Y(16q),Y(16q+1),\ldots,Y(16q+15)\},
$$

with the lower unsigned E4M3 byte selected on a modal tie. Runtime
reconstruction is

$$
\widehat Y_K(h,b)=Y_{12}\!\left(G_K(h,b)\gg4\right).
$$

Thus the 12-bit table is a piecewise-constant approximation to the discrete
Gaussian-quantile E4M3 staircase. It leaves the reference distribution
unchanged. The approximation chain is

$$
\text{normal equal-probability rank}
\longrightarrow
\text{Gaussian-midpoint finite-E4M3 label}
\longrightarrow
\text{modal 16-rank execution label}.
$$

The authoritative construction is implemented independently in
`qsrt/sqg_e4m3.py` and B12X. QSRT generates the 4,096-byte staircase and
complete K2/K3/K4 direct encoder labels and passes them through
`qsrt/sqg_quantizer.py`; B12X evaluates the same immutable construction at
runtime. Frozen SHA-256 checks over the T12 table and all three 65,536-byte
direct tables make cross-repository drift fail in unit tests. A payload
encoded under a different graph cannot be relabelled in place because
$\{F_j\}$ and the selected Viterbi paths differ.

## Fixed-payload rate shifting

The common 3,072-neuron intermediate axis is divided into 24 records of 128
neurons.  Each record retains the existing 16x16 coding tiles.  A single
function-preserving permutation is applied as

```text
W1' = P W1
W3' = P W3
W2' = W2 P^T.
```

Because SiTU is coordinatewise, this changes neither the expert function nor
the coordinate presented at the nonlinear boundary.  It makes importance
regions contiguous and makes the record rate derivable from a small mode ID,
without a per-channel rate map or runtime shuffle.

For a matrix family, mode `Rr` assigns

```text
first r records       K2
middle 24 - 2r        K3
last r records        K4.
```

Thus every mode averages exactly three path bits per weight:

```text
R0 =  0 K2 + 24 K3 + 0 K4
R1 =  1 K2 + 22 K3 + 1 K4
R2 =  2 K2 + 20 K3 + 2 K4
```

QSRT redistributes rate over paired 128-channel records without changing
payload size. A `P24` container assigns K2 to a low-priority donor record and
K4 to a high-priority recipient, while a `P33` container assigns K3 to both.
Each consumes six trellis bits per coefficient pair and occupies the same
physical size. Pair placement is rotated over a global 96-slot atom axis by
layer and expert so P24 work remains balanced at every supported TP view.

The mode is `(r13, r2)`. `w1` and `w3` share `r13` for fused execution;
`w2` selects `r2` independently. The common physical neuron permutation does
not require the three matrices to share a rate schedule.

### Allocation coordinate invariant

The neuron permutation and the rate allocator act at different granularities
and must be composed in a fixed order. First freeze one bijection $P$ over the
3,072 intermediate coordinates and apply it identically to `w1`, `w3`, and
`w2` as above. Then define every record or tile rate decision in that encoder
coordinate system. The same $P$ must be used to construct K2/K3/K4 candidate
errors, select the rate map, rerun BlockLDLQ with the selected map, and score
the reconstructed expert after undoing $P$.

The present conditioning search is a restricted subset of those bijections:
it moves indivisible contiguous four-channel groups and forms each 16-channel
neuron band from exactly four such groups. Each neuron band intersects all 224
orthogonal 16-channel bands, producing 224 distinct 16x16 coefficient tiles.
The funding decision belongs to one such intersection; a neuron band alone
does not define it. This restriction preserves the encoder's score and packing unit. It
must not be described as a tile permutation: one group move changes every
orthogonal tile incident on those four channels. The rate allocator sees only
the completed post-permutation tile grid and cannot move, split, or reinterpret
a group.

Conditioning policies span exact sensitivity order, within-record balancing,
source-shape clustering, and joint sensitivity/rate-response clustering. A
rate-response feature for one four-channel group is computed only after a
preliminary K2/K3/K4 encode and contains its regularized K3 error and K2/K3
and K3/K4 error ratios across every one of the 224 incident orthogonal bands,
separately for upstream and down. The current search preserves each
128-channel importance population and clusters only the 32 four-channel groups
inside that record. It does not move a favorable local group into a different
donor or recipient population. This is a fixed fit-only proposal rule. It has
no fitted continuous hyperparameter.

A tile allocator is not allowed to change $P$. Searching a conditioning
permutation is an outer discrete experiment: each proposed $P$ receives a
complete, independently encoded and validated rate-allocation search. Within
a fixed 128-channel record, a conditioning policy may reorder its eight
16-channel stripes without changing record membership. Moving a channel
between records changes both the candidate basis and its donor/recipient
population and therefore requires rebuilding all rate-error surfaces.

Formally, let $A_{13}$ be the shared gate/up tile-rate map and $A_2$ the down
tile-rate map. The experiment is bilevel:

$$
(A_{13}^*(P),A_2^*(P))
=\arg\min_{A_{13},A_2}D_{\mathrm{fit}}(P,A_{13},A_2),
$$

followed by comparison of

$$
D_{\mathrm{confirm}}
\left(P,A_{13}^*(P),A_2^*(P)\right).
$$

The confirmation partition never selects a tile, a permutation, or a prefix.
When selecting the permutation policy itself, aggregate fit evidence across
training experts and report its result on held-out experts; do not choose the
policy from the same experts' confirmation scores. Every reported candidate
uses a complete BlockLDLQ re-encode and an independently fitted scale. Local
tile errors and single-toggle functional deltas are proposal statistics only,
because the channel permutation and BlockLDLQ feedback couple many tiles.

Gate and up share an intermediate-axis rate map. Down uses an independent map
over its matching intermediate axis because its orthogonal 16-channel tile
coordinate has different functional meaning. A tile-funding experiment may
therefore share a bitmap between `w1` and `w3`, but it must not silently reuse
that bitmap for `w2`. All compared candidates record the frozen permutation
identity, and the research encoder rejects implicit or mismatched permutation
bases.

### Constant-payload 3.083333-trellis-bpw allocation

For 24 intermediate-axis records, an all-QSRT allocation with two more K4
records than K2 records satisfies

$$
N K2 + (22-2N)K3 + (N+2)K4 = 74
$$

trellis bits per 24 coefficients, or $74/24=3.083\overline{3}$ trellis bpw.
This equation constrains rate counts rather than physical pairing. One
research selector grammar realizes it as one `P44` pair, $N$ `P24` pairs, and
$11-N$ `P33` pairs. `P44` stores two K4 records. At tile granularity that
grammar enforces the same identity independently for every 16-channel strip
after freezing the neuron permutation. The shared gate/up map and independent
down map must each sum to 74 bits in every strip.

The sealed fixed profile uses a different physical decomposition. It sets
$N=0$ and places each K4 record beside a K3 record. The result is two `P43`
pairs and ten `P33` pairs; `P43` stores K4 first and K3 second. Separating the
two K4 records balances high-rate work across storage rows while retaining the
same 74 trellis bits.

Two fixed-stride selector grammars were evaluated as research controls. The
paired grammar stores one 32-bit word per strip containing eleven P33/P24 bits
for gate/up and eleven for down. A strip fixes one of eight 16-channel offsets
inside a record and one of 224 orthogonal 16-channel bands, giving
$8\times224=1,792$ strips. The selector therefore costs 7,168 bytes per expert,
or 0.001736 bpw over all three expert matrices. A top-two-K4 grammar
also stores one 32-bit word per strip: each of the gate/up and down K4-record
pairs is one of $\binom{24}{2}=276$ possibilities and therefore needs nine
bits. This likewise costs 7,168 bytes, or 0.001736 bpw. Its disposable
rank-local serving view may expand the canonical word into two 24-bit masks,
but those 14,336 prepared bytes are not checkpoint payload. Both grammars have
a trellis-plus-selector rate of 3.085069 bpw.
Scale and container metadata are accounted separately by exact serialized
bytes.

The current numerical gate uses `h2_reverse` as the frozen permutation. A
four-expert layer-24 screen compared two allocation-conditioned permutations
that moved only intact 16-channel bands within fixed 128-channel records. The
P24-pair objective and top-two-K4 objective improved their fit proxies, but
their isolated serial confirmation SSE regressed by 0.277% and 0.293%
respectively. A preceding 24-expert split likewise rejected rate-response
clustering by 0.205% on held-out experts. These policies remain research
controls; tile allocation does not authorize changing the production neuron
ordering.

A 24-expert panel spanning layers 1, 24, and 40 used fit documents for every
schedule decision and document-disjoint confirmation rows for the final
comparison. The isolated confirmation results were:

| 3.083333-bpw schedule | Pooled SSE | Change from K3 |
| --- | ---: | ---: |
| K3 on all 24 records | 3.335706 | baseline |
| Fixed 22 K3 + 2 K4 records | 2.938512 | -11.907% |
| Tile-top-two K4 for gate/up only | 2.941872 | -11.807% |
| Tile-top-two K4 for down only | 2.944253 | -11.735% |
| Tile-top-two K4 for both axes | 2.950014 | -11.563% |

The fixed record schedule is therefore the first qualified 3.083333-bpw profile.
It is slightly better than every tile-top-two selector while requiring no
selector payload, no tile-local rate bookkeeping, and no new kernel grammar.
An exploratory broad schedule search reached 2.932161 pooled SSE, only 0.216%
below the fixed schedule, but its candidate shortlist was formed from batched
fit proxies and is not production evidence. It does not justify the added
metadata or runtime surface.

All 24 fixed-schedule experts improved over K3. On a four-expert scale-control
subset, separately closing both sides left the high-rate schedules 17.79%
below scale-closed K3. The five-point path-aware schedule-specific scale search
regressed the high-rate pooled result by 0.113% and improved K3 by 0.174%, so
the profile retains the source-local scale fitted by the uniform-K3 procedure.
The serialized trellis rate is exactly $74/24=3.083\overline{3}$ bpw before
the existing scale and container metadata.

## TP-independent balanced-atom storage

Tensor parallelism is a load-time view over the checkpoint. The codec does not
serialize it. The canonical sharding unit is a balanced 32-channel atom. For logical mirrored
record pair $i\in\{0,\ldots,11\}$ and 16-channel stripe
$s\in\{0,\ldots,7\}$, define

$$
a=8i+s.
$$

The encoder serializes logical donor/recipient pair $i$ as physical records
$2i$ and $2i+1$. Atom $a$ owns stripe $s$ from both of those physical
records. In mode `Rr`, its rate pair is

$$
(K_\mathrm{low},K_\mathrm{high})=
\begin{cases}
(2,4),&i<r,\\
(3,3),&i\ge r.
\end{cases}
$$

Both cases contain exactly six trellis bits per coefficient pair. For one
matrix, one atom therefore occupies exactly

$$
32\cdot3584\cdot\frac{3}{8}=43{,}008\ \text{bytes}.
$$

The atom bundle stores the fixed trellis fragments for `w1`, `w3`, and `w2`
plus their three 32-value FP16 intermediate-side scale fragments:

$$
B_\mathrm{atom}
=3(43{,}008+32\cdot2)
=129{,}216\ \text{bytes}.
$$

There are 96 atoms per compressed expert, so atomization preserves the exact
payload:

$$
96B_\mathrm{atom}=12{,}404{,}736\ \text{bytes per expert}.
$$

It adds no rate padding and cannot separate coupled coordinates: both sides
of a P24/P33 pair, all three expert matrices, and all three local scale
fragments have one atom owner.

### Physical atom order

Let

$$
\rho_{\ell,e}=(5e+\ell)\bmod12.
$$

The physical slot of logical atom $a$ is

$$
p=(a+8\rho_{\ell,e})\bmod96.
$$

The rotation is defined over the model-global atom axis and contains no TP
rank. It rotates complete record pairs, leaves the stripe index unchanged,
and is bijective for every layer/expert. Both on-disk revisions are
atom-major:

```text
atoms-v1: [96 physical atom slots, compressed experts, 129216 bytes]
atoms_v2 fixed-stride profiles: [96 physical atom rows, common row stride]
atoms_v2 coupled high-rate profile: [96 rows grouped into 12 variable-width record-pair extents]
```

The layer file is a standards-valid safetensors container. Its tensors are the
atom slab, a 4 KiB expert-format section, and a 24 KiB shared-scale section.
The safetensors header itself occupies a fixed 4 KiB. Fixed-stride profiles
round the largest row payload to 4 KiB and give every row that common stride.
This creates both alignment padding and unused space when row populations have
different bundle widths. The fixed `k3x22_k4x2` profile has 2,048 padding bytes
on 32 rows and 23,552 bytes on 64 rows. Its padding is 1,572,864 bytes per
layer, or 144,703,488 bytes across 92 layers. The uniform two-bit profile and
the variable-stride coupled high-rate profile have zero measured atom-slab
padding. These fixed offsets permit direct range loading of physical atom rows
without copying unrelated payload bytes.

Atoms-v1 uses a uniform 129,216-byte bundle for each three-bit QSRT expert.
Atoms-v2 retains the same 96-row container and atom ownership, but permits a
fixed profile to divide each row into compact groups with different bundle
widths. Group membership is a deterministic function of layer, expert, and
physical record pair; it is not serialized as a TP-specific map.

### Shard views

For any TP size $T$ dividing 96, rank $q$ owns

$$
A=96/T
$$

consecutive physical atom slots beginning at $qA$. Its local intermediate
width is $32A=3072/T$. Consequently one rank loads one aligned contiguous
extent per layer; no trellis bit is decoded, shifted, concatenated, or
repacked. All practical Kimi-K3 views through TP32 are exact direct views:

```text
TP = 1, 2, 3, 4, 6, 8, 12, 16, 24, 32
```

TP48 and TP96 are also equal-width views. A shard count that does not divide
96 still receives one aligned, contiguous range of complete atoms. The
quotient/remainder partition covers every atom once and differs by at most one
atom, or 32 intermediate channels, between shards. A runtime requiring equal
local shapes pads only its disposable prepared cache; canonical bytes remain
unchanged. Thus arbitrary resharding never separates a P24/P33 atom or
requires trellis re-encoding.

At TP12, a rank owns eight atoms, exactly 256 intermediate channels. The
canonical layout produces this ownership; the format does not serialize a TP12
contract.

### Load preparation

The loader reads the safetensors metadata and transfers only its atom-row
range. It then performs one GPU preparation pass that removes slot padding and
transposes atom-major
storage into the fused-MoE operand layout. It also derives P24/P33 work queues
from `(layer, expert, physical_slot, r13, r2)`. Rate metadata is per
expert/atom during preparation; the fused coefficient loop has no TP-dependent
addressing and no coefficient-level rate branch. Rank-local prepared buffers
are disposable caches and are never checkpoint files.

The canonical implementation and byte-accounting reference are in
`qsrt/qsrt_storage.py`.

### Fixed 3.083333-trellis-bpw `atoms_v2` profile

The 3.083333-trellis-bpw all-QSRT profile has no P24 pairs or tile selector. In logical
importance order, records 0 through 21 are K3 and records 22 and 23 are K4.
The checkpoint then applies one expert-static permutation of complete
128-channel records, shared by `w1` rows, `w3` rows, and `w2` columns. This is
the exact symmetry

$$
W_1'=PW_1,\qquad W_3'=PW_3,\qquad W_2'=W_2P^\mathsf{T}.
$$

No channel, 16-channel tile, or transformed block is split by this placement.
Because the encoder's intermediate transform is block-128, moving complete
records commutes with that transform. The rate schedule, reconstructed expert
function, and source-local scales are therefore unchanged by physical
balancing.

The two K4 records are placed in distinct physical record pairs. The pair
assignment is rotated by

$$
\rho_{\ell,e}=(5e+\ell)\bmod12,
$$

which balances K4 work across serving shards without serializing a TP count.
The profile uses the `atoms_v2` revision of the canonical atom container. Every
physical record pair contributes eight consecutive atom rows. Within each
row, experts using P33 are stored first in ascending expert order with a
129,216-byte bundle; experts using P43 follow in ascending expert order with a
150,720-byte bundle. The common row stride is the 4-KiB-aligned maximum over
all 96 rows. The layer, expert, and physical-pair rotation determines group
membership exactly, so the file stores neither a TP count nor an expert mode
bitmap.

A rank owns complete contiguous atom rows; at TP12 it reads eight rows, or 256
intermediate channels. The canonical expert-layer container size is
1,051,056,799,744 bytes across 92 layers, including safetensors headers and row
padding. Load preparation removes the group layout into a disposable compact
P33/P43 operand pool. The canonical layout and exact accounting are in
`qsrt/qsrt_atoms_v2.py`.

### Uniform-K2 coupled `atoms_v2` profile

The uniform-K2 profile assigns K2 to all 24 intermediate records of every
routed expert. It uses the same `qsrt_sqg_e4m3` encoding and
`sqg_xor_cheb_t12` codebook as the mixed-rate profile; it changes the exact
coordinate system presented to the quantizer and removes rate allocation:

```text
record bits                    24 x K2
routed experts                 82,432 QSRT; zero X4T
trellis rate                   2.0 bpw
per-expert atom payload        8,275,968 bytes
payload rate including scales  2.004464 bpw
```

The coupled transform is an exact reparameterization of the full-precision
expert. Let

$$
Q=\mathrm{interleave}(W_1,W_3)
$$

contain alternating gate and up rows. Let $U_R$ be the layer-shared normalized
block-H512 residual transform, and let $U_{A,e}$ and $U_{B,e}$ be the
expert-static normalized signed block-H128 transforms on the interleaved
preactivation and post-SiTU axes. With row vectors, the stored source matrices
before lossy encoding are

$$
Q'_e=U_{A,e}^{\mathsf T}Q_eU_R,
\qquad
W'_{2,e}=U_R^{\mathsf T}W_{2,e}U_{B,e}.
$$

Execution applies

$$
z'=zU_R,
\qquad
q'=z'Q_e'^{\mathsf T}=(zQ_e^{\mathsf T})U_{A,e},
$$

undoes $U_{A,e}$ before splitting the alternating coordinates and evaluating
SiTU, applies $U_{B,e}$ to the resulting hidden vector, and finally undoes
$U_R$ after the down projection. Orthogonality gives exact unquantized
closure:

$$
y'_e=(hU_{B,e})W_{2,e}'^{\mathsf T}=(hW_{2,e}^{\mathsf T})U_R,
\qquad
y_e=y'_eU_R^{\mathsf T}=hW_{2,e}^{\mathsf T}.
$$

The residual transform uses draw zero and is shared by every expert in a
layer. The intermediate draw is expert-static and selects deterministic sign
vectors for both $U_{A,e}$ and $U_{B,e}$; the two axes use different vectors
derived from the same draw ID. The frozen family contains eight draws, so the
artifact stores one three-bit ID per expert rather than sign tensors. The
all-expert build evaluated the qualified production portfolio `{0,6}` with a
fit-propose/confirmation-accept rule. Its sealed plan contains 60,277 draw-zero
experts and 22,155 draw-six experts.

Pure K2 uses the compact P22 atom bundle. Each 32-channel atom stores all three
matrix fragments and their FP16 intermediate-side scale fragments in 86,208
bytes. The 96 atom rows retain the same TP-independent ownership as the
mixed-rate layout. The sealed 92-layer expert container is 682,207,608,832
bytes including safetensors headers and aligned row padding. Every supported
TP view reads complete contiguous atom rows; only the disposable prepared
cache is rank-local.

The runtime fuses the activation-boundary transforms with the routed expert
path. It does not materialize dense transformed weights or intermediate
activation tensors, and it does not branch on a rate mode or tile-local
codebook. Load preparation expands the deterministic intermediate signs for
the local experts and converts the atom rows into the W4A16 operand layout.
The canonical implementation and byte accounting are in
`qsrt/qsrt_coupled.py`, `qsrt/qsrt_coupled_plan.py`, and
`qsrt/qsrt_atoms_v2.py`.

## Dense-H encoding and statistical selection

Cheap importance scores only propose the permutation and donor/recipient
records. They do not authorize a rate shift. Down-projection candidates are
evaluated through complete dense-$H$ BlockLDLQ re-encodes so cross-record
covariance feedback is retained. For `w1`/`w3`, the common input covariance is
retained while the selected output-row records receive their assigned rates.

`H13` may be layer-global because all routed experts consume the same residual
coordinate system. `H2` may not be pooled: every expert owns a distinct
post-SiTU coordinate system, and the coupled draw changes that system again.
For each decoded upstream candidate, the encoder replays that expert's routed
rows, constructs $H_{2,e}$ from the reconstructed middle activations, and
shrinks it only toward its own trace-scaled identity:

$$
\widehat H_{2,e}
=\alpha_e H^{\mathrm{sample}}_{2,e}
+(1-\alpha_e)
\frac{\mathrm{tr}(H^{\mathrm{sample}}_{2,e})}{3072}I.
$$

Unsupported experts use identity. A layer-global post-SiTU covariance is
neither a prior nor a fallback. In the uniform-K2 build, each candidate draw
therefore receives its own complete upstream reconstruction, conditional
`H2`, down-projection BlockLDLQ encode, and full-expert routed score.

The encoder then reconstructs the full expert and scores applied-gate-square
weighted routed output error on document-disjoint samples.  A nonzero mode is
accepted only when its paired document-bootstrap lower confidence bound clears
the frozen improvement margin over matched SQG `R0`; uncertain experts fall
back to `(R0,R0)`.

The mixed-rate search evaluates only the 3x3 Cartesian grid

```text
(r13, r2) in {0,1,2} x {0,1,2}.
```

This keeps the all-expert encode operationally viable while retaining the
independent `w2` decisions that earlier studies showed were important. Modes
are expert-static: serving reads a compact format code and never performs
runtime rate selection.

## X4T exact endpoint

Official microscaling four-bit floating point (MXFP4) uses four E2M1 bits per
weight plus one unsigned eight-bit exponent-only (UE8M0) scale byte per 32
weights, or 4.25 bpw. X4T changes no represented value:

- every E2M1 nibble is preserved exactly, including both zero codes;
- each scale row chooses the adjacent UE8M0 pair that covers the most entries;
- selector bits and out-of-pair exceptions reproduce the complete official
  scale plane; and
- load preparation partitions the decoded exact matrix on 32-channel storage
  groups, with the same equal or quotient/remainder shard rule as QSRT.

The selector stream is directly indexable and needs no tile offset table,
prefix sum, or exception search.  X4T is therefore the high-quality
endpoint; uniform K4 remains lossy and is not treated as a substitute for the
official weights. The all-expert X4T index stores each expert's exact tensor
payload contribution rather than relying on a nominal bpw estimate.

The canonical X4T layer is a TP-independent safetensors file. It contains:

```text
expert_ids: int32[E]
w1/w3.packed: uint8[E, output_rows, input_columns / 2]
w2.packed: uint8[E, 32-channel input groups, output_rows, 16]
matrix.scale_fixed: uint8[E, fixed_stream_bytes]
matrix.scale_exceptions: uint8[concatenated exception bytes]
matrix.scale_exception_offsets: int64[E + 1]
```

The safetensors JSON directory is padded to 4 KiB, which makes storage exactly
additive: each layer has 4,128 fixed bytes, and each expert contributes its
three matrix tensor payloads plus 28 bytes for its ID and exception-offset
entries. Artifact SHA-256 closure receipts authenticate the complete files;
there is no private record header, directory, CRC, or padding convention.

The stored matrix is the source of truth; rank-local W4A16 tensors are a
load-time cache. `w1`/`w3` partition on 32-row groups and `w2` on 32-column
groups. Equal divisors produce identical local shapes; other shard counts use
the same bounded uneven partition and optional cache padding as the QSRT atom
reader. The checkpoint never stores a rank count or rank-local X4T copy.

The scalar scale codec remains `qsrt/mxfp4_scale_codec.py`. The existing
full-matrix `qsrt/x4t.py` layer container is the canonical exact endpoint.

### X4T runtime refinement

The compressed X4T scale planes can remain persistent in device memory rather
than being expanded for every expert at model initialization.  Immediately
before the ordinary W4A16 call, one graph-safe launch expands only the routed
experts into a caller-owned packed-scale scratch buffer.  That scratch is
reused across layers on the same stream; there is no per-call allocation, CPU
parsing, prefix scan, exception search, or disk access.

The TP12 implementation reproduces the active packed W4A16 scale bytes, folds
the fused-`w1`/`w3` row rotation and the bfloat16 clamp for decoded UE8M0
scales into the same launch, and survives scratch poisoning followed by CUDA
graph replay. An external B12X study used 1,000 balanced replays on an RTX PRO
6000 Blackwell Max-Q and one routed input row (`M=1`). It reported the following
complete routed-MoE costs:

| Active X4T experts | Dense W4A16 | X4T + W4A16 | Added latency |
| ---: | ---: | ---: | ---: |
| 1 | 22.08 µs | 24.16 µs | 2.08 µs |
| 2 | 22.11 µs | 26.11 µs | 4.00 µs |
| 4 | 22.11 µs | 26.21 µs | 4.10 µs |
| 8 | 24.16 µs | 28.26 µs | 4.10 µs |
| 16 | 30.30 µs | 32.35 µs | 2.05 µs |

Two-row and four-row sweeps also closed exact scale-byte reconstruction; their
added latency ranged from 1.25 to 8.19 µs depending on routed density. Output
differences once four or more experts contribute match the dense kernel's own
repeatability envelope and come from nondeterministic atomic accumulation,
not scale decode. This repository contains neither the external benchmark
source nor an authenticated result file for these measurements. The numbers
therefore provide latency context but do not constitute a reproducible QSRT
qualification gate.

The remaining X4T work for this checkpoint is to build its all-expert
exact-byte index and run an authenticated routed benchmark with
checkpoint-derived selections.

## Global allocation

Rate shifting and high-tier selection solve different problems.

1. For each expert, the candidate pool freezes the statistically selected
   `(r13,r2)` at the same three-bit trellis payload.
2. X4T then competes against that selected lossy candidate.  Promoting an
   expert removes its measured routed damage and incurs that expert's exact X4T
   safetensors payload bytes rather than a fixed nominal four-bit cost.

The following comparison byte cap came from the validated interim EXL3
checkpoint stored at `/models/Kimi-K3-EXL3-3p09`:

```text
target container bytes = 1,058,586,247,168
```

The `3p09` directory token does not establish the checkpoint's measured bits
per weight. Before this cap defines an allocation ceiling, the EXL3 checkpoint
must be recounted from exact serialized bytes under the same accounting rules
as QSRT. A compression-dominant candidate must use a strictly smaller complete
checkpoint budget. The final cap must be restated in canonical atom-container
bytes; TP-rank padding is not a valid budget component. The global
allocator minimizes

```text
sum_e D_e(choice_e) + lambda * sum_e bytes_e(choice_e)
```

and sweeps `lambda` to meet the checkpoint budget. The allocator stores one
boolean `x4t_mask`: a false entry retains the selected QSRT candidate, and a
true entry promotes the complete expert to X4T. The optimizer charges exact
trellis and X4T safetensors bytes. Since X4T sizes vary by expert, candidate
generation and X4T cost indexing remain reusable when the target budget
changes.

## Evidence and qualification state

The initial production-path SQG study used 24 official-source experts across
layers 1, 24, and 40. At fixed K2/K3/K4 endpoints, SQG normal beat two
external EXL encoder controls named `MUL1-E4M3` and `MCG` for every expert and
all 216 matrix/rate comparisons. These API names identify offline scalar
codebooks; they are not QSRT formats. The study is hypothesis-forming because
its down-projection metric used a layer-global post-SiTU covariance whose
coordinate indices are not shared across independently permuted experts.

The matched R0/R1/R2 gate on the same panel found:

```text
SQG selected nonzero r13       6 of 24 experts
SQG selected nonzero r2        2 of 24 experts
SQG proposed nonzero r13       8 of 24 experts
SQG proposed nonzero r2        5 of 24 experts
aggregate SQG R0 vs MUL1 R0    2.2443% lower confirmation SSE
aggregate SQG selected vs
  MUL1 selected                2.1071% lower confirmation SSE
```

The 21-document confirmation fold established execution closure through the
Hadamard transform, LDLQ feedback, Viterbi search, official weights, and routed
replay. It did not establish representative Hessian geometry or model quality.
Any candidate pool built with its layer-global down-projection covariance is
ineligible for a current checkpoint.

Current mixed-rate candidate generation uses the source-controlled
one-million-token training capture at
`/data/kquant/captures/k3-denseh-broad-v6-1m-train.kqcapture`. The gate/up input
covariance, `H13`, remains layer-global because its latent input basis is
shared. The down-projection input covariance, `H2`, is rebuilt from routed
post-SiTU rows for each expert and upstream reconstruction. It is shrunk toward
identity according to sample support, with identity used for unsupported
experts. The fixed high-rate and uniform two-bit transform studies below have
separate four-million-token fit-corpus provenance; their results do not change
the current mixed-rate capture contract. Final-validation corpora remain
document-disjoint from every fit corpus.

The mature B12X W4A16 kernel has one QSRT serving reconstruction: QSRT-E4M3.
Superseded exact-graph and external-control codebook branches have been removed
from that kernel. The external `MUL1` and `MCG` variants remain offline controls
where comparisons require them. QSRT-E4M3 passes dense K2/K3/K4 reconstruction,
P24, P33, dynamic pair selection, and CUDA graph replay closure.

The current mature split-K W4A16 implementation measures about 56.90 µs for
P33 and 65.09 µs for P24 on the production-shaped benchmark and reproduces the
CPU decoder for P33, P24, and dynamic pair layouts. P24 remains above the
current latency target.

The coupled-Hadamard uniform-K2 profile was selected independently of the
mixed-rate pool. On fresh uniform-K2 SQG re-encodes, the fixed coupled boundary
transform reduced pooled routed SSE by 3.052% and improved 22 of 24 experts. The
expert-private draw rule was then fit on the 4M capture and evaluated on a
separate 128K corpus: pooled routed SSE fell 1.308%, the expert median fell
0.447%, and 17 of 28 experts improved. A single global intermediate draw lost;
layer-shared draw selection recovered only 0.030%, establishing that the
useful degree of freedom is expert-static rather than global.

The first complete artifact materializes all 82,432 routed experts under the
uniform-K2 profile, with exact payload closure for all 92 MoE layers and no
X4T experts. Its first full-model A16 gate ran at TP8 over 32 windows and
65,504 scored positions. Mean Kullback-Leibler divergence was 0.08520 in the
direction `KL(reference || candidate)`. The reference logits came from the
official MXFP4 Kimi-K3 checkpoint captured in the pinned dataset
`festr2/kimi-k3-full-mxfp4-kld-reference-32x2048`, revision
`097b2775900c0940d31c6469c2e930be8d17b2f8`. The run's fail-closed audit
observed the QSRT atom reader, hybrid quantization loader, W4A16 implementation,
and post-start repeat check on every rank. This is a successful codec and
runtime integration result at an extreme expert rate; task quality, multimodal
execution, and production latency remain separate gates.

## Supported reconstruction path

QSRT exposes one serving reconstruction profile:

| Profile | Role | Contract |
| --- | --- | --- |
| `qsrt_sqg_e4m3` | sole runtime encoding profile | `sqg_xor_cheb_t12`: the two-round XOR/odd-multiply bijective SQG graph plus the shared 12-bit approximation to the Gaussian-midpoint finite-E4M3 staircase at K2/K3/K4 |

There is no runtime external-control codebook, superseded exact graph,
alternate K2 staircase, or per-expert codebook selector. External API names
may appear in offline research controls, but they are not valid payload or
kernel profile identities.

Three TP-independent `atoms_v2` storage profiles use that reconstruction:

| Storage profile | Record law | Implementation status |
| --- | --- | --- |
| `k3x22_k4x2` | 22 K3 records and 2 K4 records | Sealed all-QSRT high-rate artifact |
| `k2_coupled_h512_h128` | 24 K2 records plus the exact coupled boundary transform | Sealed all-QSRT two-bit artifact |
| `k3x22_k4x2_coupled_h512_h128` | 22 K3 records and 2 K4 records plus the exact coupled boundary transform | Implemented and unit-tested; no sealed whole-model artifact is recorded here |

Rate shifting was checked on 384 unseen experts sampled across routing-support
levels from layers 1, 24, and 40. The study used production Hadamard ordering,
TensorFloat-32 dense-H BlockLDLQ,
decoded-upstream conditional `H2`, the complete `R0/R1/R2` grid, and
document-disjoint confirmation and external validation. The native
`sqg_xor_cheb_t12` law retained 89 shifted experts, including 88 with `w2`
R1+ and 77 with `w2=R2`, at a pooled selected external SSE of 68.5222868.
This establishes the native four-stratum K2 mapping as the sole encoder
contract. No matrix- or rate-specific reconstruction-law selector is used.

### Offline trellis-encoder optimization

The SM120 offline tile encoder chooses its closing state with a 256-step cyclic
primer: the last 128 and first 128 symbols of the tile itself. It does not read
neighboring coefficients outside the 256-value tile. Its optimized
implementation transposes
each state-indexed SQG byte table into predecessor-major groups so one thread
loads all K2 or K3 predecessor labels in one vector transaction and all K4
labels in two. K2 uses 1,024 threads and stores four two-bit traceback decisions
per byte. K3 uses 640 threads while preserving the established 512-thread final
reduction tree. K4 uses 704 threads with the maximum-L1 cache preference.
Paired
half-precision comparisons update both paths together.

On 512 production-codebook tiles at C128, median kernel time changed as
follows on SM120:

| Rate | State-indexed label loads | Predecessor-major packed traceback | Reduction |
| --- | ---: | ---: | ---: |
| K2 | 7.270 ms | 4.825 ms | 33.63% |
| K3 | 6.378 ms | 4.015 ms | 37.05% |
| K4 | 6.054 ms | 3.345 ms | 44.75% |

Safety was checked directly against the state-indexed CUDA reference in 63 cases
covering K2/K3/K4, C1/C32/C128, Gaussian/heavy/structured inputs, and the
production and control E4M3 tables.  Reconstructed values and trellis indices
were bit-identical in every case.  A complete 20-expert layer-24 endpoint
study also produced the identical serialized candidate-payload SHA-256 and
the same selected modes, while wall time fell from about 136 to 89 seconds.

A shorter C32 primer is not part of this optimization.  The initial
20-expert screening study put every C128 confirmation winner in C32's top
three, but that is not sufficient evidence for the all-expert pool or a future
arbitrary-record search.  The current production build remains C128 end to
end; C32 may be revisited only as a shortlist generator followed by exact C128
re-encoding after a substantially broader audit.

### All-expert mode selection

The sealed production candidate pool at
`/data/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-CANDIDATES-v1` contains all 92
MoE layers and 82,432 experts. A nonzero mode is retained only when its paired,
document-clustered confirmation lower bound clears the zero-improvement
margin.

| Selected `(r13,r2)` | Experts | Share |
| --- | ---: | ---: |
| `R0/R0` | 73,053 | 88.622% |
| `R0/R1` | 1,007 | 1.222% |
| `R0/R2` | 1,231 | 1.493% |
| `R1/R0` | 213 | 0.258% |
| `R1/R1` | 1,252 | 1.519% |
| `R1/R2` | 3,428 | 4.159% |
| `R2/R0` | 10 | 0.012% |
| `R2/R1` | 30 | 0.036% |
| `R2/R2` | 2,208 | 2.679% |

In aggregate, 9,379 experts (11.378%) select a confirmed nonzero shift. The
down projection selects R1+ in 9,156 experts (11.107%), the coupled gate/up
pair selects R1+ in 7,141 (8.663%), and R2 appears on at least one axis in
6,907 experts (8.379%). X4T endpoint allocation is a separate exact-byte
optimization over these sealed candidates.

### Extreme-rate K2 research register

Pure or nearly pure K2 operation makes individually small gains relevant. The
following mechanisms remain distinct research candidates; a negative result
for one parameterization does not remove the underlying symmetry or coding
degree of freedom.

The coupled boundary transform has two rotation scopes. Residual-side signs
are selected once per layer and shared by all experts: they act on the
`w1`/`w3` input boundary and the `w2` output boundary. Intermediate-side signs
may be selected per expert, but the selected draw is coupled across the
interleaved gate/up preactivation boundary and the matching post-SiTU `w2`
input boundary. This preserves layer-level reuse at the residual boundary
without forcing one intermediate rotation on all 896 experts.

The real K2 draw screen uses eight deterministic intermediate rotations. For
each expert, the three lowest-SSE draws on 128 fit documents from the 4M
capture are shortlisted, and 128 disjoint confirmation documents choose among
them. The resulting expert-static draws were then evaluated on up to 256 rows
per expert from the separate 128K corpus. Across 28 experts and seven layers,
the selected rotations reduced external routed post-projection SSE by 1.308%
pooled and 0.447% at the expert median, with 17 of 28 experts improving. The expert-mean
bootstrap interval was 0.364% to 1.864%. A single global draw lost, and a
layer-shared draw recovered only 0.030% pooled, so the useful degree of freedom
is expert-private. Three bits per expert identify one of eight deterministic
draws; the sign vectors themselves need not be stored.

The residual-side screen kept this boundary layer-shared and tested draws zero
through seven after expert-private intermediate selection. Six of seven layers
selected identity. Layer 40's only fit-and-confirm survivor regressed 0.606%
on the separate corpus. Residual draw zero therefore remains fixed.

| Mechanism | Mathematical role | Current evidence | Qualification |
| --- | --- | --- | --- |
| Coupled gate/up/down boundary Hadamard | Exact change of basis before the coordinatewise activation boundary | Fresh uniform-K2 SQG re-encodes improved routed error on 22 of 24 experts; pooled routed SSE fell 3.052%. Eight-draw expert-private selection on the 4M capture transferred to a separate 128K corpus: 1.308% pooled and 0.447% median routed improvement, with 17 of 28 experts improving. A single global draw lost and layer-shared selection recovered only 0.030% pooled | Materialized in the pure-K2 profile with residual draw zero and one expert-static three-bit intermediate draw. TP8 A16 model execution and KLD have closed; TP12 multimodal and latency qualification remain |
| Activation-metric W1/W3 pair code | Uses the local 2-by-2 SiTU metric so gate/up errors can cancel | The isolated pair codebook improved 24 of 28 experts; median functional metric improvement was 4.90% | Codebook oracle; needs a joint vector trellis, decoded-payload scoring, and full-expert validation |
| W3/W2 sign gauge | Exact symmetry from the odd up activation | Eight deterministic sign representatives were searched with real K2 SQG. The fit/confirmation-selected gauge transferred by 0.393% pooled and 0.302% median on 7,168 untouched routed rows, with 14 of 28 experts improving. Combining it mechanically with the selected Hadamard draw reduced the Hadamard gain from 1.308% to 0.448% | Retain as an expert-static alternative to the selected intermediate rotation. Baking matched signs into W3 rows and W2 columns costs no payload or runtime work |
| Positive W3/W2 scale gauge | Approximate symmetry while the up branch is linear | The scalar proxy selected a nonidentity gauge for 18 of 28 experts, but those frozen proposals worsened real SQG K2 by 2.250% pooled and 0.840% median on the 4M corpus. Full-precision drift was not the cause: median relative SSE was $8.22\times10^{-9}$ and the worst was $3.59\times10^{-5}$ | Reject the tested RMS/absmax policies. A future gauge requires path-aware SQG selection and must beat identity before external validation |
| Co-routing-aware candidate phase | Chooses among near-equal expert errors to reduce top-16 cross terms | A 32-row layer-24 audit measured a positive cross term equal to 0.929% of diagonal mapped SSE; the linear metric matched exact post-projection SSE within 0.458% | Plausible sub-percent headroom; requires two or more retained trellis candidates per expert and document-disjoint selection |
| Aligned per-neuron shared bases | Stores a small number of layer bases and quantizes only expert residuals | In 32-expert coordinate sketches, rank-four excess capture over an energy-matched isotropic null was 1.18, 1.74, 0.09, and 0.39 percentage points at layers 1, 24, 64, and 92. Global expert coefficients were weaker | Track as a low-rate oracle, but the signal is not stable through depth; require full-coordinate/all-expert factorization and residual K2 encoding before implementation work |
| Reconstructed-activation W2 refit | Compensates upstream quantization before the final K2 encode | Dense refit improved 20 of 28 experts with 1.55% median routed improvement | Upper bound only; the dense fitted matrix must be distilled into a cheap structured correction or used solely as the next W2 encoding target |

The remaining low-rate design space is retained explicitly even where no
production result exists yet:

| Mechanism | Pure K2 | Required decisive experiment |
| --- | --- | --- |
| One- or two-bit tile-local K2 codebook menu | Yes | Rejected for the first pure-K2 format. On 28 experts across seven layers, the four-law fit-selected menu regressed 0.080% pooled on the separate 128K corpus. Confirmation gating recovered only 0.072%, and a conservative normal-versus-alternate-law gate recovered 0.066%; neither result justifies tile metadata, alternate lookup tables, or kernel branches. Use the single Gaussian T12 staircase. |
| Joint gate/up vector trellis | Yes | Incorporate the measured 2-by-2 activation metric into full tail-biting assignment, preserve decoded scale closure, and score the complete reconstructed expert. |
| Successively refinable K2 base | Yes, as the base layer | Jointly train a two-plane base and K3/K4 refinement planes with the production transform, refitted scales, and functional objective; do not infer viability from native MXFP4 bit truncation. |
| Tile-local P33/P24 funding | No | Rerun the actual equal-byte pair allocator after every tile proposal. Keep its selector accounting and kernel grammar separate from pure-K2 quality claims. |
| Gate/up/down rate triples such as 234 permutations | No | Select complete equal-byte projection triplets through decoded whole-expert error; isolated matrix SSE cannot choose which projection receives K2 or K4. |
| Joint low/high K6 vector code | No | Treat as a six-bit pair-allocation oracle and require a real trellis realization plus P24/P33 allocator comparison before considering a runtime format. |

The co-routing objective for retained candidate mode $m_e$ is

$$
\min_{\{m_e\}}
\sum_n\left\|
\sum_{e\in\mathcal R_n}p_{n,e}J_n\epsilon_{n,e,m_e}
\right\|_2^2,
$$

where $J_n$ is the post-aggregate RMSNorm/output-projection Jacobian. Candidate
modes must first pass an expert-local unary-loss bound; otherwise cancellation
can hide an unacceptable individual regression. The repository analysis
solver implements this constrained objective, but the sealed candidate pool
contains only one payload per expert, so alternate paths must be generated in
a fresh research encode rather than inferred from aggregate SSE.

## Implementation and qualification status

The sealed all-QSRT artifacts and the unfinished mixed-tier path have separate
status:

| Artifact or component | Implemented evidence | Remaining qualification |
| --- | --- | --- |
| Fixed 22-K3 and 2-K4 all-QSRT artifact | The complete candidate pool and 92-layer `atoms_v2` materialization are sealed at `/data/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-3p08-CANDIDATES-v1` and `/data/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-3p08-v2` | Checkpoint-level task quality and production TP12 serving gates |
| Uniform two-bit all-QSRT artifact | All 82,432 routed experts are sealed; unquantized transform closure, structural validation, TP8 W4A16 execution, and the 32-window KLD gate are complete | TP12 multimodal loading, image inputs, long-context behavior, and production latency |
| Native `R0/R1/R2` candidate pool | All 82,432 expert decisions are sealed at `/data/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-CANDIDATES-v1` | Untouched-capture scoring and matched-R0 policy audit before mixed-tier allocation |
| X4T exact endpoint | Numerical representation, TP-independent safetensors storage, load-time reconstruction, and graph-safe scale preparation are implemented | All-expert exact-byte index and authenticated checkpoint-derived routed benchmark |
| Mixed QSRT/X4T artifact | Exact-byte allocation and materialization code paths are implemented | Sealed X4T index, frozen allocation, fresh materialization, structural validation, serve packaging, and full-model quality gates |

A mixed-tier release requires streamed comparison with the official checkpoint,
live routing and logit checks, text and multimodal task quality, long-context
behavior, TP12 latency, and document-disjoint KLD confirmation. The completed
32-window KLD gate establishes coherence of the uniform two-bit artifact and
its A16 serving path. It does not establish production task quality.

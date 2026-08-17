# Research corpus for improving two-bit QSRT quality

Status: research design and evidence index, 2026-08-17.

The Quantile-Stratified Rate-shifted Trellis codec (QSRT) is a
tensor-parallel-independent weight codec for the routed experts in Kimi-K3.
The current uniform representation stores two trellis bits per routed expert
weight and 0.004464 bits per weight of scales. Complete model storage and one
end-to-end numerical result exist. No repository result shows better model
quality than the validated ExLlamaV3 trellis checkpoint at
`/models/Kimi-K3-EXL3-3p09` (EXL3) or a named eight-bit weight baseline. The
directory token `3p09` does not establish the checkpoint's exact rate.

This document defines the evidence behind that conclusion, indexes techniques
that could improve the codec, and specifies the experiments needed to compare
QSRT with EXL3. The leading representation hypothesis encodes corresponding
gate and up coefficients as two-dimensional trellis symbols. It trains the
symbol table against the local metric of Kimi-K3's coordinatewise gated
activation (SiTU) and uses downstream loss curvature to choose paths. The
strict trellis stream remains two bits per weight. No result yet shows that
this representation improves a complete layer or model.

The documents have distinct authority:

- [The uniform two-bit codec specification](qsrt-2bpw-codec.md) defines the
  implemented two-bit storage and reconstruction contract.
- [The QSRT technical brief](qsrt-technical-brief.md) defines the complete
  Kimi-K3 codec, allocation, storage, and runtime system.
- [The repository agent guide](../AGENTS.md) defines operational policy and
  records the active artifact state.
- This research corpus indexes evidence, ranks candidate experiments, and
  records their implementation and measurement status.
  It does not override an implemented format.

Executable code and passing tests define implemented behavior when repository
prose disagrees. The reconciliation list below records known documentation
conflicts rather than choosing an unsupported resolution.

## What the repository establishes

Kimi-K3 has 92 mixture-of-experts layers and 896 routed experts in each layer,
for 82,432 layer/expert assignments. Each expert contains gate and up matrices
with shape 3,072 by 3,584 and a down matrix with shape 3,584 by 3,072. The
coordinatewise SiTU activation combines corresponding gate and up coordinates.
For gate value `g` and up value `u`, it computes

```text
[4 tanh(g / 4) sigmoid(g)] * [25 tanh(u / 25)].
```

This elementwise product is why an error pair from one gate/up coordinate has a
meaningful two-by-two local metric.

The uniform two-bit representation has the following implemented properties:

- Each coefficient follows a closed path through a 16-bit edge-window (L16)
  bitshift trellis. The two-bit rate is called K2. At K2, the previous fourteen
  path bits define the state and the two new branch bits choose its outgoing
  edge.
- The carry-mixed Stratified Quantile Graph (SQG) maps all 65,536 directed
  edges bijectively to Gaussian probability ranks.
- A shared 4,096-byte table maps the upper twelve rank bits to finite eight-bit
  floating-point reconstruction values with four exponent bits and three
  mantissa bits (E4M3).
- A coupled Hadamard change of basis conditions the complete gate, up, and down
  matrix triplet while preserving the full-precision expert function.
- Gate and up use one layer-global input covariance. Down uses an expert-specific
  covariance rebuilt from the reconstructed gate and up activations.
- The dense-covariance blockwise error-feedback procedure (BlockLDLQ) feeds
  committed quantization error into later coefficient blocks.
- Complete-expert output errors are scored on naturally routed rows and
  weighted by the square of the applied router coefficient.
- Canonical storage groups complete 32-channel units called atoms. An atom is
  the smallest tensor-parallel-independent ownership and random-access unit.

The current evidence supports engineering correctness and local quality gains:

- Coupled Hadamard conditioning reduced pooled routed expert-output squared
  error by 3.052% on 24 experts, with 22 experts improving.
- Selecting one intermediate Hadamard sign draw per expert reduced pooled
  routed error by a further 1.308% on a separate 128,000-token corpus. The
  median expert error fell 0.447%, and 17 of 28 experts improved.
- The complete uniform representation covers all 82,432 routed experts at
  2.004464 bits per routed-expert weight including stored scales.
- A run using eight-way tensor parallelism (TP8) and 16-bit activations (A16)
  measured mean forward Kullback-Leibler divergence,
  `KL(reference || candidate)`, of 0.0851995464 over 65,504 scored positions.

The repository does not contain a task, perplexity, or paired KL result in
which the uniform two-bit representation beats EXL3, microscaled eight-bit
floating point (MXFP8), another eight-bit floating-point format (FP8), or
eight-bit integer quantization (INT8). The recorded KL value shows that the
uniform two-bit artifact and runtime execute coherently. It does not establish
production quality.

The repository also contains evidence that discourages small scalar changes:

- A four-law tile-local scalar table menu regressed pooled error by 0.080% on a
  separate corpus. Conservative confirmation gates recovered only 0.066% to
  0.072%.
- A positive up/down scale gauge selected by a scalar proxy worsened real
  two-bit SQG by 2.250% pooled and 0.840% at the expert median.
- Applying one intermediate Hadamard sign draw to all experts regressed.
  Layer-shared draw selection recovered only 0.030%.
- Allocation-conditioned permutations regressed confirmation error by 0.277%
  and 0.293%. Rate-response clustering regressed by 0.205%.
- A layer-global post-SiTU covariance is invalid because the coordinates are
  expert-local. Results computed with that covariance are unsuitable as
  production evidence.

Two research oracles show larger, structurally relevant headroom:

- A 16-entry gate/up vector codebook trained with the local two-by-two SiTU
  metric beat a 16-entry codebook trained with Euclidean distance for 24 of 28
  experts when both were scored with the SiTU metric. The median improvement
  was 4.90%. This oracle does not compare either codebook with the scalar QSRT
  codec.
- Refitting the down matrix against reconstructed upstream activations improved
  20 of 28 experts. The median routed improvement was 1.55%.

These results motivate a vector trellis and a reconstructed-activation target.
They do not predict full-model gains by themselves.

The evidence has four distinct levels:

- **Offline oracle.** The pair-codebook and down-refit results use more freedom
  than the proposed stored representation. They measure available headroom and
  cannot qualify a codec.
- **Path-constrained panel.** A real trellis encode on a panel selected to cover
  experts with different routed-sample support (a support-stratified
  panel) can reject a representation or objective. A panel result does not
  establish a complete-layer or model gain.
- **Complete-layer transfer.** A full layer with natural routing tests expert
  interactions and bitwise agreement between the reference and accelerated
  decoders (kernel closure). It remains a screening result for model quality.
- **Paired full-model evaluation.** Only document-aligned forward KL from
  complete artifacts can establish the primary quality claim. Decoder timing
  and synthetic-source distortion are separate runtime and coding
  measurements.

Qualification is scope-specific. Passing a correctness or performance gate
qualifies only the named implementation and configuration. It does not imply
complete-model or production quality. The first serving target uses twelve-way
tensor parallelism (TP12).

Three repository state descriptions need reconciliation before running the
research program:

- [`AGENTS.md`](../AGENTS.md) identifies a one-million-token dense-covariance
  training capture. [`docs/qsrt-technical-brief.md`](qsrt-technical-brief.md)
  attributes coupled-transform selection to a four-million-token capture. The
  documents do not state how these two training roles relate.
- [`AGENTS.md`](../AGENTS.md) records a sealed all-QSRT artifact with a
  3.083333 trellis-bits-per-weight (bpw) schedule. Adding the documented
  0.004464 scale bpw gives 3.087797 bpw before container headers and padding.
  [`docs/qsrt-technical-brief.md`](qsrt-technical-brief.md) still lists scoring,
  exact high-quality X4T allocation, materialization, and packaging work. X4T
  reproduces the official microscaled four-bit floating-point (MXFP4) expert
  weights while compressing their scale plane. The unchecked items appear to
  describe the mixed QSRT/X4T path and need an explicit scope label there.
- Files under `out/` contain generated two- and three-bit normal-float studies
  (NF2 and NF3) that predate the present QSRT representation. They are not
  evidence for the scalar SQG or pair trellis.

## Meaning of the eight-bit target

An unconstrained eight-bit representation has a strictly larger code space
than a two-bit representation and can include every two-bit reconstruction in
its code space. The named Kimi-K3 comparison is stronger because the
repository's MXFP8 procedure appears to preserve every official routed-expert
weight value. Uniformly lower weight distortion than that exact-value
conversion is impossible for a frozen source model.

Kimi-K3 makes the limitation stronger. Its official routed-expert source
already uses MXFP4. Each four-bit E2M1 coefficient contains two exponent bits
and one fraction bit in addition to its sign, and every decoded source value
has the form

```text
E2M1_value * 2^source_exponent.
```

MXFP4 stores each normalized value in the four-bit E2M1 floating-point alphabet
and shares a power-of-two scale across 32 values. The repository's MXFP8 helper
chooses a power-of-two scale for each 32-value block and stores each normalized
coefficient in the E4M3 floating-point format. For the finite E2M1 alphabet and
source-scale exponent range, the helper's arithmetic implies exact E4M3
representation, including the clipped endpoint cases. This implication still
needs an exhaustive alphabet test and a streamed real-expert round trip before
the MXFP8 conversion can serve as the comparator.

An exact MXFP8 conversion has zero stored-weight error relative to the official
source. Zero weight error does not by itself guarantee zero runtime KL because
the MXFP8 and MXFP4 kernels may use different arithmetic. A paired runtime
comparison must measure that difference with the same 16-bit activation
policy. Forward KL cannot fall below zero, so a two-bit codec cannot surpass an
execution path that reproduces the source logits exactly.

Model behavior admits one narrower comparison. A two-bit representation can be
optimized against routed activations, downstream loss curvature, teacher
logits, and task behavior. Quantization-aware recovery can improve selected
task scores through regularization or adaptation even while coefficient error
and teacher KL remain larger. Such a result is a behavioral improvement on the
pre-registered task suite. It is not superior general fidelity to the source
model.

The governing objective is complete-model Pareto dominance over the immutable
EXL3 checkpoint. A qualifying QSRT artifact must occupy fewer exact serialized
bytes and have lower held-out forward KL. The two artifacts may use different
bits per weight. A rate-matched comparison can diagnose codec efficiency, but
it is not a release requirement.

The comparison must use the same activation precision, tokenizer, prompts,
routing implementation, and runtime audit requirements. Keep non-expert
weights hash-identical when the experiment is intended to attribute the result
to routed-expert coding. Charge trellis payloads, scales, tables, alignment,
indices, format metadata, and non-expert storage to each complete artifact.
Neither the `3p09` directory token nor a nominal QSRT bits-per-weight value can
substitute for the exact byte ledger.

The evaluation must contain three references:

- **Official-source reference.** The immutable Kimi-K3 MXFP4 checkpoint defines
  the teacher values and the primary behavioral reference.
- **EXL3 comparison.** The validated EXL3 checkpoint and its exact file
  inventory define the deployed quality and size baseline. The repository's
  1,058,586,247,168-byte allocation cap is historical evidence about that
  build, but the comparison must recount the immutable artifact rather than
  infer its rate from the `3p09` name.
- **Eight-bit comparison.** Decode the official MXFP4 source and requantize the
  routed experts with the repository's pinned MXFP8 E4M3 block-32 procedure.
  Prove bit-exact value reconstruction on all E2M1 block maxima and streamed
  real experts. Use the same non-expert overlay and A16 execution path as the
  QSRT candidate.

The final claim must name the exact eight-bit format. A claim against an
unspecified category called "traditional eight-bit quantization" is not
reproducible. If task-tuned two-bit QSRT beats untuned MXFP8 on the selected
aggregate, report that narrow result. Do not describe it as lower
quantization error or better teacher fidelity.

## Academic technique corpus

The corpus gives priority to peer-reviewed papers and primary preprints. A
reported gain applies to the model, rate, objective, and runtime in the cited
work. Each QSRT use below is a hypothesis until Kimi-K3 experiments confirm it.

### Transfers from classical signal coding

Classical signal codecs contribute optimization principles, but their source
assumptions and storage contracts differ from QSRT. The following mechanisms
have direct, testable counterparts.

| Classical mechanism | Primary source and transferable result | QSRT experiment |
| --- | --- | --- |
| Transform coding and decorrelation | The [discrete cosine transform](https://doi.org/10.1109/T-C.1974.223784) was designed as a fast transform with rate-distortion behavior close to the source-dependent Karhunen-Loève transform for correlated signals. JPEG applies it to spatially correlated image blocks ([Wallace, 1992](https://doi.org/10.1109/30.125072)). | Measure covariance and residual correlation in each legal QSRT coordinate system. Compare identity, the qualified Hadamard transform, and a pair-preserving low-complexity transform at identical payload, table bytes, and TP12 runtime. Require exact full-precision expert closure before quantization. |
| Alternating assignment and reconstruction updates | [Lloyd's least-squares quantizer conditions](https://doi.org/10.1109/TIT.1982.1056489) alternate nearest-region assignment with centroid updates. The same principle extends to state-constrained trellis paths. | Alternate hard tail-biting Viterbi assignment with activation-metric pair-table updates. Compare the trained table with its fixed initialization on disjoint documents, and rematerialize hard E4M3 values after every update. |
| Error-weighted distortion | JPEG permits coefficient-dependent quantization, and [Watson's visually optimized matrices](https://doi.org/10.1109/DCC.1993.253132) minimize a perceptually weighted error rather than raw coefficient error. The transferable principle is to optimize the error that the consumer observes. | Replace visual weights with routed SiTU sensitivity and teacher-derived output curvature. Compare scalar squared error, routed expert error, and the two-sided model-loss proxy by their ability to predict held-out forward KL. Visual frequency weights themselves have no role in this model. |
| Predictive coding and quantization-error feedback | Differential pulse-code modulation predicts a sample and quantizes its innovation; classical analysis treats the feedback path and quantizer jointly ([Jayant, 1974](https://doi.org/10.1109/PROC.1974.9484)). Noise-shaping systems similarly move error into directions that the reconstruction operator attenuates; [Boufounos and Oppenheim](https://doi.org/10.1155/ASP/2006/53807) formulate this for finite frame expansions. | Treat BlockLDLQ as error feedback in the input-covariance geometry. Sweep legal coefficient orderings and block widths, then measure residual autocorrelation and held-out KL. Do not import a frequency-domain noise filter unless a model-loss metric identifies an analogous low-sensitivity subspace. |
| Rate-distortion Lagrangians and bit allocation | [Entropy-constrained vector quantization](https://doi.org/10.1109/29.17498) minimizes distortion plus a multiplier times code length. [Shoham and Gersho](https://doi.org/10.1109/29.90373) allocate an integer bit budget across quantizers with irregular measured rate-distortion curves. | Use `distortion + lambda * exact_bytes` to allocate complete, decoded expert candidates. Sweep the multiplier across exact-byte ceilings below the EXL3 ledger and include the uniform two-bit artifact as a control. Charge fixed tables, scales, padding, and selectors directly; do not substitute empirical entropy for stored bytes. |
| Multistage and residual vector quantization | [Juang and Gray](https://doi.org/10.1109/ICASSP.1982.1171604) reduce vector-codebook complexity by quantizing a residual in successive stages. [Aksu and Salehi's residual trellis-coded vector quantizer](https://doi.org/10.1109/26.705401) applies a multistage construction to trellis source coding. | After the single pair trellis is measured, test a base path plus a deterministic residual stage against a one-stage candidate at the same exact byte count. A second fixed-rate index stream qualifies only if the combined payload remains four bits per gate/up pair and all reconstruction tables are charged. It otherwise defines a separate average-rate endpoint. |
| Overlap and joint source statistics | Lapped orthogonal transforms use basis functions that cross block boundaries and reduce boundary error at added computation ([Malvar and Staelin, 1989](https://doi.org/10.1109/29.17536)). JPEG2000 uses wavelet support across neighborhoods and independent coded blocks for scalable allocation ([Skodras, Christopoulos, and Ebrahimi, 2001](https://doi.org/10.1109/79.952804)). | Measure pair residuals and state occupancy across adjacent trellis windows and atom boundaries. Test overlapping context during path selection while retaining the same stored atom ownership. A stored transform that couples atoms proceeds only after bounded random access, TP independence, and fused-kernel cost close. |

JPEG and JPEG2000 also contain mechanisms whose gains depend on image
structure or variable-length streams. They are poor defaults for QSRT:

- **Block DCT.** JPEG's 8-by-8 DCT exploits local spatial correlation. Matrix
  adjacency in a serialized expert is a layout choice and does not imply the
  same stationary two-dimensional source. QSRT should select transforms from
  measured expert covariance and full-precision closure.
- **Zig-zag order and run-length coding.** JPEG orders DCT coefficients from
  low to high spatial frequency so trailing quantized zeros form long runs.
  Expert weights have no corresponding spatial-frequency order, and the
  repository's production-panel evidence favors the Hessian-derived order.
- **Huffman and arithmetic coding.** JPEG's Huffman stage and JPEG2000's
  context arithmetic coder exploit symbol-probability skew with variable code
  lengths. Variable-length expert streams would remove constant stride,
  complicate bounded random access, and create routed GPU load imbalance.
- **Progressive bitplanes.** JPEG2000's embedded block coding with optimized
  truncation provides rate-distortion scalability and block-level random
  access ([Taubman, 2000](https://doi.org/10.1109/83.847830)). Its embedded
  arithmetic streams require indices and serial context decoding. QSRT can
  transfer the exact-byte truncation objective while retaining fixed-rate
  candidate payloads.
- **Image perceptual models.** Contrast sensitivity, masking, and spatial
  frequency are properties of human vision. Their transferable abstraction is
  error weighting. QSRT must derive those weights from routed activations,
  teacher curvature, and held-out model behavior.

### Trellis structure and reconstruction

| Publication | Demonstrated technique | Relevance to QSRT | Main cost or risk |
| --- | --- | --- | --- |
| [Trellis Coded Quantization of Memoryless and Gauss-Markov Sources](https://doi.org/10.1109/26.46532), IEEE Transactions on Communications 1990 | Uses path constraints and Viterbi search to obtain shaping gain over scalar quantization. | Supplies the alternating assignment and reproduction-value design foundation for training SQG emissions on real residuals. | Classical source models do not include Hessian feedback or neural activations. |
| [Trellis-Coded Vector Quantization](https://doi.org/10.1109/18.104316), IEEE Transactions on Information Theory 1991 | Extends trellis symbols from scalars to vectors. | Direct foundation for encoding aligned gate/up coefficient pairs as one symbol. | Vector symbols change packing, table loads, and Viterbi costs. |
| [Design, Performance, and Complexity Analysis of Residual Trellis-Coded Vector Quantizers](https://doi.org/10.1109/26.705401), IEEE Transactions on Communications 1998 | Uses multiple residual trellis stages to improve source matching and control codebook complexity. | Suggests successively refinable base and residual paths if one vector stage cannot close the quality gap. | Two decoded components increase runtime and do not fit the first strict single-stream experiment. |
| [QTIP: Quantization with Trellises and Incoherence Processing](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6de2e84b8da47bb2eb5e2ac96c63d2b0-Abstract-Conference.html), NeurIPS 2024 | Combines Hadamard incoherence, high-dimensional bitshift trellises, two-dimensional hybrid tables, tail biting, and BlockLDLQ. At two bits on an independent Gaussian source, its trellis codes report 0.068 to 0.069 MSE versus 0.089 for the eight-dimensional QuIP# code and 0.118 for scalar Lloyd-Max. | Shows that a two-dimensional symbol fits the same bitshift architecture and that a small trainable table can retain high effective dimension. | QSRT already inherits much of QTIP. Reusing its Gaussian table without a Kimi-specific objective would add little novelty. |
| [Maximum-Hamming-Distance Convolutional Codes for TCQ](https://arxiv.org/abs/0704.1411) | Relates convolutional generator choice to trellis quantization performance. | Supports a bounded search over graph generators after emission training is measured. | Generator search changes Viterbi paths and requires complete re-encoding plus runtime qualification. |
| [CCQ: Convolutional Code for Extreme Low-bit Quantization in LLMs](https://arxiv.org/abs/2507.07145) | Uses convolutional-code vector codebooks and compact mappings for large-model weights. | Contributes graph candidates and scale-packing ideas. | Its reported two-bit DeepSeek-V3 and ERNIE results remain below its INT8 comparisons, so it does not establish the target quality. |

### Curvature and assignment objectives

| Publication | Demonstrated technique | Relevance to QSRT | Main cost or risk |
| --- | --- | --- | --- |
| [QuIP: 2-Bit Quantization of Large Language Models With Guarantees](https://proceedings.neurips.cc/paper_files/paper/2023/hash/0df38cd13520747e1e64e5b123a78ef8-Abstract-Conference.html), NeurIPS 2023 | Minimizes the input-covariance-weighted error `tr((Wq-W) H (Wq-W)^T)`, where `Wq` is the quantized weight, `W` is the source weight, and `H` is the input covariance, after making the weights and covariance incoherent. | Confirms the dense input-covariance objective already used by QSRT and supplies its theoretical boundary. | The one-sided layer-output objective omits downstream model sensitivity. |
| [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://proceedings.mlr.press/v235/tseng24a.html), ICML 2024 | Adds randomized Hadamard transforms, the eight-dimensional E8 lattice code, and blockwise recovery tuning. | Establishes that recovery tuning and shaped vector codes improve extreme compression beyond scalar rounding. | General hidden rotations do not commute with Kimi-K3's coordinatewise multiplicative expert activation. |
| [GPTVQ: The Blessing of Dimensionality for LLM Quantization](https://arxiv.org/abs/2402.15319) | Interleaves Hessian-aware vector assignment with updates to unquantized columns and refines compressed codebooks. | Supports joint vector assignment and Hessian-weighted table initialization. | Unstructured vector lookup grows exponentially and does not preserve the QSRT decoder. |
| [VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language Models](https://aclanthology.org/2024.emnlp-main.467/), EMNLP 2024 | Uses second-order vector assignment, channel-independent refinement, and optional residual or outlier codes. | Supplies a practical model for channel-wise second-order emission refinement. | Residual and outlier planes violate a strict uniform-rate claim unless every byte is charged. |
| [LeanQuant: Accurate and Scalable Large Language Model Quantization with Loss-Error-Aware Grid](https://arxiv.org/abs/2407.10032) | Learns reconstruction grids against an inverse-Hessian loss proxy. | Suggests training the finite pair table on loss-weighted errors rather than fitting Gaussian marginals. | Diagonal loss weights may miss the gate/up cross term and routed mixture interactions. |
| [Model-Preserving Adaptive Rounding](https://arxiv.org/abs/2505.22988) | Approximates full-model forward KL with a Kronecker product of output and input curvature and applies feedback on both sides. The paper reports about 30% lower KL than LDLQ on its evaluated models and quantizers. | Supports a format-preserving objective experiment that combines expert-local output curvature with the existing input covariance and correct expert-local down basis. | The reported gain does not establish a Kimi-K3 or trellis result. Capturing output gradients and applying two-sided feedback increases encoder memory and time. Mixture-of-experts routing needs an explicit unbiased sampling policy. |
| [GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance](https://arxiv.org/abs/2505.07004) | Adds end-loss gradient information while preserving dependencies between weight errors. | Provides a simpler gradient-aware ablation before full two-sided feedback. | Later model-preserving rounding evidence favors the Kronecker formulation; both need Kimi-specific validation. |
| [GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration](https://arxiv.org/abs/2504.02692) | Calibrates each quantized layer against the exact full-precision output so accumulated upstream error enters the target. | Supports asymmetric targets for the down projection and for later mixture layers. | Layerwise asymmetric calibration can overfit a small corpus and does not alone resolve expert co-routing. |
| [MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance](https://arxiv.org/abs/2505.03804) | Addresses imbalance in routed samples across experts and weights samples by router affinity inside each expert. | Supports separate production-weighted and expert-balanced estimates for rare-expert curvature, table fitting, and confirmation. | Expert balancing changes the optimized traffic distribution. It must remain visible rather than silently replacing production frequency. |

### Learned codebooks and discrete optimization

| Publication | Demonstrated technique | Relevance to QSRT | Main cost or risk |
| --- | --- | --- | --- |
| [Extreme Compression of Large Language Models via Additive Quantization](https://proceedings.mlr.press/v235/egiazarian24a.html), ICML 2024 | Learns additive vector codebooks against activation error and jointly tunes transformer blocks. | Shows that activation-aware learned reproduction values and cross-layer optimization can materially improve two-bit models. | Multiple additive codebooks need extra decoder work and storage. The first experiment should train one trellis table. |
| [PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression](https://proceedings.neurips.cc/paper_files/paper/2024/hash/091166620a04a289c555f411d8899049-Abstract-Conference.html), NeurIPS 2024 | Alternates continuous parameter updates with explicit discrete code reassignment and improves one- and two-bit vector quantization. | Supports alternating pair-table and scale updates with hard Viterbi path reassignment, including bounded behavioral recovery. | Every discrete update requires another encode. Full-model parameter tuning is expensive for Kimi-K3. |
| [BCJR-QAT: A Differentiable Relaxation of Trellis-Coded Weight Quantization](https://arxiv.org/abs/2605.10655), Venugopalan Iyengar, arXiv 2026 | Uses finite-temperature Bahl-Cocke-Jelinek-Raviv (BCJR) forward-backward path marginals to replace hard Viterbi assignment during quantization-aware training (QAT). The preprint reports a 0.084 WikiText-2 perplexity reduction from the QTIP post-training-quantization baseline after tuning one Llama-3.2-1B layer with end-to-end forward-KL distillation. | Offers a differentiable refinement for a small pair table after hard post-training initialization. | The evidence covers single- and dual-layer interventions rather than an all-layer model. Local reconstruction training and a high-temperature schedule regressed model perplexity in the reported studies. |
| [Initialisation Determines the Basin: Efficient Codebook Optimisation for Extreme LLM Quantization](https://arxiv.org/abs/2604.08118), Ian W. Kennedy and Nafise Sadat Moosavi, arXiv 2026 | Uses output-aware expectation-maximization with Hessian-weighted assignments to initialize additive codebooks. The preprint reports that the initialization advantage persists after PV-Tuning on three dense-model families. | Supports a Hessian-weighted pair-table initialization experiment before alternating Viterbi assignment and table updates. | The evidence concerns additive quantization on dense models. It does not establish the same basin behavior for state-constrained trellis paths or routed experts. |

### Transforms, outliers, and model recovery

| Publication | Demonstrated technique | Relevance to QSRT | Main cost or risk |
| --- | --- | --- | --- |
| [AWQ: Activation-aware Weight Quantization](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html), MLSys 2024 | Protects activation-salient channels through searched per-channel scaling. | Motivates sensitivity-aware scale fitting and a controlled comparison with QSRT's rejected scalar scale proxies. | QSRT's tested positive scale gauge regressed. Any retry must use real trellis paths and exact expert closure. |
| [SqueezeLLM: Dense-and-Sparse Quantization](https://proceedings.mlr.press/v235/kim24f.html), ICML 2024 | Uses sensitivity-weighted nonuniform levels and sparse high-precision storage for outliers and sensitive values. | Provides an exact-byte alternative to whole-expert X4T promotion for average-rate artifacts. | Sparse exceptions complicate routed kernels and cannot support a uniform two-bit statement. |
| [SpinQuant: LLM Quantization with Learned Rotations](https://arxiv.org/abs/2405.16406) | Learns function-preserving rotations and shows that random draw quality varies substantially. | Supports learning within QSRT's proven coupled transform family and explains the gain from expert-specific draws. | Only rotations that cancel around SiTU preserve the Kimi expert. Dense learned rotations would add runtime work. |
| [FlatQuant: Flatness Matters for LLM Quantization](https://arxiv.org/abs/2410.09426) | Learns affine transforms and compresses them with Kronecker structure. | Suggests a structured extension of the existing block-Hadamard family if pair coding and curvature still leave transform-limited error. | Affine transforms across the nonlinear boundary need a new closure proof and fused kernel. |
| [FrameQuant: Flexible Low-Bit Quantization for Transformers](https://proceedings.mlr.press/v235/adepu24a.html), ICML 2024 | Quantizes overcomplete fusion-frame representations to spread quantization noise. | Supplies a transform oracle for testing whether redundancy can reduce two-bit expert error. | Redundant representations increase stored or computed coefficients and conflict with the fixed atom payload. |
| [MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design](https://arxiv.org/abs/2505.05799) | Measures different sensitivity across experts and projection roles and combines quality with hardware cost during allocation. | Supports gate/up/down-specific loss accounting and measured TP12 latency constraints in mixed-rate research. | Matrix-granular high-quality tiers conflict with QSRT's qualified whole-expert X4T policy and require a separate format experiment. |

### Training and behavioral recovery

| Publication | Demonstrated technique | Relevance to QSRT | Main cost or risk |
| --- | --- | --- | --- |
| [BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation](https://arxiv.org/abs/2402.10631) | Combines quantization-aware training with a confidence-aware teacher KL objective at two and three bits. | Supports forward-KL tuning of the pair table, scales, or a bounded correction rather than relying only on routed squared error. | Full-model training of Kimi-K3 is likely impractical. Teacher data and confidence weighting can bias evaluation. |
| [ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization](https://proceedings.neurips.cc/paper_files/paper/2025/hash/83b17fb3369b1effa97ca5409526b02e-Abstract-Conference.html), NeurIPS 2025 | Finds a learning transition between two and three bits: two-bit representations change more substantially during recovery than higher-bit models. | Indicates that post-training path selection alone may reach diminishing returns and that behavior-level recovery may be necessary. | Results from trained scalar formats do not establish that a trellis table can recover Kimi-K3. |
| [Unified Progressive Quantization toward 2-bit Instruction-Tuned LLMs](https://arxiv.org/abs/2506.09104) | Uses an intermediate four-bit model and teacher-divergence training before hard two-bit deployment. | Suggests using the exact MXFP4 source and a high-quality QSRT representation as progressive teachers for the pair trellis. | The method changes model parameters and needs a substantial, legally usable training corpus. |

## Proposal ranking by evidence and implementation risk

Two format-preserving mechanisms remain active. Neither has passed a
document-replicated or complete-checkpoint comparison with EXL3. The paired
forward-KL suite still needs multiple reference documents, and the capture path
must record the output gradients needed by the curvature estimator.

1. **Model-loss-aware scalar path selection.** The two-sided recurrence,
   output-metric factorization, bounded factor format, and frozen-scale scalar
   encoder are implemented. Synthetic and complete real-matrix CUDA closures
   pass. The remaining experiment must capture output gradients, build real
   expert-local factors, and test whether the resulting score predicts
   document-disjoint forward KL better than routed expert squared error.
2. **Reconstructed-activation down target.** Re-encoding a down target fitted
   against reconstructed gate/up activations reduced mean KLD by 1.8269%
   relative to uniform K3 on one GLM context. It still lost to EXL3 by 0.2691%.
   The next gate repeats the comparison on multiple documents and then across
   early, middle, and late layers.
3. **Exact-byte allocation after a validated damage score.** If two-sided
   curvature predicts held-out KLD, use its score to compare K3, K4, and bounded
   residual candidates per serialized byte. This experiment depends on the
   score; it does not justify a new payload before the prediction gate passes.

Training the finite-E4M3 table is rejected on the measured GLM residual domain.
The strongest fixed-path per-matrix oracle reduced pooled squared error by only
0.00175%. One-sided routed-input covariance is also rejected because it
increased mean KLD by 2.3453% relative to uniform K3. A gate/up pair trellis
remains a later representation experiment because it needs a new decoder and
may lose the qualified scalar transform basis.

The corresponding GLM-5.2 experiments are specified in
[the improvement strategy](qsrt-improvement-strategy.md). Its mixed-rate panel
must preserve a structural distinction that does not appear in an isolated
rate tuple. The implemented GLM port encodes gate, up, and down as separate
matrices, so it can test unequal gate/up rates. Kimi-K3 interleaves gate and up
rows before its coupled Hadamard transform; those transformed rows cannot be
assigned separate gate and up rates. An unequal-rate Kimi candidate must give
up that transform and charge its measured conditioning benefit. Joint gate/up
trellis symbols impose the same shared-rate constraint.

Sparse corrections remain a measured oracle rather than a promoted format.
For a 256-position independent Gaussian tile, the largest squared residual
carries about 3.7% of total squared error on average, and the two largest carry
about 6.6%. Those shares are too small to justify even the minimum payload of
one or two indexed corrections against K4. The GLM screen therefore requires
held-out curvature-weighted shares above 6% for the largest residual or 10%
for the two largest, followed by an exact-byte oracle win over K4. A
four- or five-bit shared correction exponent per tile is the cheapest range
control worth testing. Any accepted correction must participate in BlockLDLQ
feedback, and any stored Walsh-mode correction must remain in the coding domain
so the decoder does not acquire another inverse transform.

## Pair-trellis representation hypothesis and falsification design

The leading representation change encodes corresponding gate and up
coefficients as one two-dimensional trellis symbol. The descriptive name for
the research object is **layer-trained activation-metric pair trellis**. Its
support comes from an offline codebook oracle on 28 experts. A real
path-constrained encode may erase that gain because the pair basis gives up the
qualified production gate/up row transform. The proposal therefore ranks
behind an objective-only curvature experiment on the existing scalar
bitstream. A format or schema identifier should be assigned only after the
pair representation passes the panel, complete-layer, and paired full-model
gates.

### Payload and state geometry

For preactivation neuron `j` and input coordinate `i`, let one source symbol
be the aligned pair

```text
v_i = [W_gate[j, i], W_up[j, i]].
```

The pair is formed after the layer-shared residual transform on the input
columns and before any transform across preactivation rows. The production
coupled profile applies a 128-coordinate Hadamard transform across the
interleaved preactivation rows. Those transformed rows are mixtures of several
activation coordinates, so the local two-by-two SiTU metric does not apply to
them. The first pair-trellis experiment must use identity on that preactivation
boundary. It can retain the layer residual transform and the matched
post-SiTU/down transform because neither operation destroys the gate/up pair.

This coordinate change gives up a qualified part of the production coupled
conditioning and may lose more than the pair code recovers. The experiment
therefore needs two scalar controls: scalar SQG in the activation-aligned basis
and scalar SQG in the production coupled basis. A later transform may mix
whole gate/up pairs with the Kronecker product `H64 ⊗ I2`, where `H64` is a
64-coordinate Hadamard transform and `I2` preserves each two-coordinate
gate/up pair. Its transformed output metric contains cross-pair blocks. It
requires two-sided feedback and a separate closure and transfer study.

Store four branch bits for each pair. Each pair contains two weights, so the
trellis payload remains two bits per weight. With a 16-bit edge word:

```text
history bits                 12
branch bits per pair          4
states                    4,096
branches per state           16
directed edges           65,536
weights represented per step  2
```

The production scalar K2 trellis has 16,384 states, four branches per state, and
65,536 directed edges. The pair trellis therefore evaluates the same directed
edge count per time slice and half as many time slices. Each edge cost is a
two-dimensional quadratic form rather than one squared residual.

The decoder obtains two E4M3 values from one table entry and writes them to the
fused gate/up operand. A layer-shared table with 4,096 pairs occupies 8,192
bytes. Ninety-two layer tables occupy 753,664 bytes before alignment, which is
negligible against the 682-gigabyte routed-expert payload. A layer table avoids
per-expert cache changes and permits the reconstruction law to follow depth.

Generalize the existing carry-mixed graph at four branch bits. Interpret the
upper four bits of its 16-bit rank as one cell in a four-by-four Cartesian grid
of gate and up marginal quartiles. Interpret the remaining twelve bits as the
phase inside that joint stratum. The runtime pair-table index is

```text
pair_table_index = rank >> 4.
```

Each state then exposes one candidate from every joint gate/up stratum, and
each stratum has 256 trainable pair entries in the 4,096-entry execution table.
During table fitting, constrain each entry to the empirical quantile box for
its assigned stratum. This retains broad sign and magnitude coverage while
letting the pair locations respond to Kimi-K3 activations. Gate and up retain
separate block scales; each table entry supplies their two normalized E4M3
values.

### Activation metric

For neuron `j`, use routed calibration rows to form a positive semidefinite
two-by-two metric `M_j` for aligned gate and up perturbations. It includes the
SiTU derivatives, the reconstructed down column, applied router weights, and a
downstream output metric. For pair residual `d`, the local distortion is

```text
d^T M_j d.
```

This two-by-two matrix is the diagonal block for neuron `j` in the full metric
over gate/up perturbations. It captures cancellation between gate and up errors
for one neuron. Cross-neuron terms remain outside the first Viterbi cost.

Across input coordinates, combine `M_j` with the input covariance shared by
the layer's gate and up projections (`H13`). The resulting pair-local model is
a Kronecker metric. Its output factor
captures gate/up cancellation through SiTU, while its input factor preserves
the dense correlation feedback already used by BlockLDLQ. The model-loss
curvature experiment adds low-rank or block-sparse off-diagonal output terms
through two-sided feedback.

The down projection receives a separate output-curvature factor and the
candidate-specific expert covariance built from reconstructed pair-trellis
activations. A pooled layer post-SiTU covariance remains invalid because
intermediate coordinates are expert-local.

### Layer-trained emission table

Initialize the pair table from weighted samples of aligned gate/up
coefficients after the layer residual transform and BlockLDLQ feedback. Do not
initialize it from an independent Gaussian product alone.

Alternate two operations on training documents:

1. **Hard path assignment.** Hold table entries and scales fixed. Run
   tail-biting Viterbi with the activation metric and dense input feedback.
2. **Emission and scale fitting.** Hold paths fixed. Update each used pair
   entry by weighted least squares, constrain it to its assigned quantile box,
   refit the separate gate and up scales, and round both normalized components
   to finite E4M3. Preserve the previous value for an unused entry, and reject
   an update if any table value is non-finite.

Every completed table update requires a fresh hard assignment. Stop when
held-out proposal error ceases to improve or after a pre-registered pass cap.
The confirmation documents may accept the trained layer table against the
current scalar SQG table. They may not select an iteration or alter an entry.

The first implementation should keep the carry-mixed bitshift transitions.
Graph-generator search becomes justified only if trained pair emissions leave
substantial adjacent-window correlation or path-occupancy imbalance.

### Reconstructed-activation down target

After pair-trellis gate/up reconstruction, denote the source post-SiTU
activation rows by `X` and the reconstructed pair-trellis activation rows by
`Xq`. Fit a regularized target `W2_target` that minimizes

```text
||Xq W2_target^T - X W2_source^T||^2
    + lambda ||W2_target - W2_source||^2.
```

Quantize `W2_target` with the ordinary two-bit down trellis and the
candidate-specific covariance. The fitted dense matrix is an offline target;
it is not stored as a correction. This uses the repository's 1.55% median
refit oracle without adding payload or runtime operations.

The regularization coefficient must be fixed on fit documents and confirmed
on disjoint documents. The original source down matrix remains a required
candidate. A refit that fails confirmation falls back to the source target.

### Model-loss-aware selection

Routed expert-output squared error remains useful for screening. Final path and
table decisions should approximate forward model KL with two-sided curvature:

```text
loss(E) = tr(H_out E H_in E^T),
```

Here, `E` is the candidate-minus-source weight-error matrix. `H_in` is the
applicable layer or expert input covariance. `H_out` is a low-rank or
Kronecker output-gradient covariance from the official teacher. Gate and up
use the local SiTU pair metric derived from `H_out`. Down uses an expert-local
output factor in the residual basis.

During research encoding, the candidate pipeline should retain at least two
pair-trellis paths for frequently co-routed experts. Each retained path must
satisfy an expert-local loss bound before mixture interactions are considered.
An offline coordinate solver may then choose one path per expert under the
routed-mixture cross terms. The artifact stores only the selected path for each
expert and requires no runtime selector.

### Bounded behavioral recovery

If hard post-training optimization improves expert metrics but misses the
pre-registered model-level KL and task targets, tune only the layer pair
tables, scales, and permitted down targets against forward teacher KL.
Alternate continuous updates with hard Viterbi reassignment. This is the
direct PV-Tuning adaptation.

After every discrete update, repack the selected paths, project the table to
runtime E4M3, and evaluate the resulting hard-decoded model. Full-model weight
fine-tuning is outside the first implementation because it would change the
source model and require a much larger training and provenance program.

## Decisive experiment sequence

Each experiment has a semantic name. No experiment number or temporary
codename should become an artifact identity.

### Establish the EXL3 comparison ledger

Inventory the immutable EXL3 artifact and the common non-expert overlay before
testing an alternate codec. Record routed-expert payload, scales, indices,
padding, alignment, and metadata separately from non-expert bytes. Freeze
hashes for the tokenizer, prompt windows, official-source logits, non-expert
weights, and runtime configuration.

Run EXL3 and the sealed all-QSRT artifact with a 3.083333 trellis bpw schedule
on the same document windows as an initial measurement. Its trellis and scale
planes total 3.087797 bpw before container overhead. Build fresh QSRT
allocations at several exact-byte ceilings below the recounted EXL3 total.
Report paired `KL(reference || candidate)` by document and domain. The first
qualifying point must be both smaller and lower-KL than EXL3. This comparison
is the codec-family baseline. It does not validate the two-bit pair proposal.

### Measure scalar SQG residual and path mismatch

Use the source-controlled training capture and a support-stratified expert
panel. Record the following measurements for the production scalar SQG table:

- marginal residual histograms after BlockLDLQ feedback;
- adjacent-window autocorrelation and conditional moments;
- state, branch, edge-label, and E4M3 occupancy;
- scalar squared error, dense-input-covariance error, routed expert error, and
  model-loss-curvature error;
- path closure cost relative to open-path and longer-context controls; and
- measurements split by layer, matrix role, route support, and activation
  saturation.

This diagnosis determines whether the primary limitation is the scalar
emission, transition geometry, local objective, or downstream accumulation.

### Test model-loss curvature on the scalar bitstream

The scalar encoder path is implemented. It accepts input and output metrics,
applies the two-sided anti-diagonal recurrence, freezes the ordinary K3 scale
plane, and emits an ordinary scalar payload. A complete `2,048 × 6,144` GLM
gate-matrix closure changed its path with about 1.57 GiB of peak allocated GPU
memory. Identity output curvature changed source-space SSE by only 0.000039%,
so that run establishes implementation closure rather than downstream quality.

Capture teacher output gradients on document-disjoint fit data. Compare the
existing one-sided dense-input objective with gradient-diagonal, low-rank
output, and Kronecker two-sided objectives while keeping the scalar format and
rate fixed. The selected approximation must predict held-out forward KL better
than routed expert squared error and must preserve the expert-local down basis.
This experiment isolates the objective change from the pair representation.

### Test a payload-matched gate/up pair trellis

On the 28-expert external-validation panel that produced the pair-codebook
oracle result, compare:

- independent scalar gate/up K2 paths in the production coupled basis;
- independent scalar gate/up K2 paths in the activation-aligned basis;
- a pair trellis with the untrained product-Gaussian table;
- a pair trellis with a Euclidean-trained layer table; and
- a pair trellis with the activation-metric-trained layer table.

All activation-aligned candidates must use the same residual and post-SiTU
transforms, scale search budget, tail-biting context, down target, and exact
two-bit stream. Charge the additional layer tables in the all-in rate rather
than calling the representations memory-matched. Report the production coupled
scalar control separately. A promising result must beat both scalar controls.
For each comparison, the lower bound of the expert-clustered bootstrap interval
for loss reduction must exceed zero, pooled loss must decrease, and the
pre-registered tail metric must remain within its regression bound.

### Add the reconstructed-activation down target

Fit the down target only after pair gate/up paths are frozen. Compare the
source target, the dense ridge target, and a structured low-rank target. Since
only the quantized matrix is stored, the dense target is acceptable if it
transfers. Reject any policy that improves fit error while worsening held-out
KL or live routing.

### Test co-routing-aware path choice

Retain two or more paths that satisfy expert-local loss bounds for the experts
present in a bounded set of layers. Select paths under the complete top-16
routed mixture objective. Report the expert-local loss, cross term, aggregate
loss, changed-expert count, and transfer to disjoint documents.

### Run complete-layer and full-model gates

Expand only after the panel result transfers across complete early, middle,
and late layers. A layer gate requires natural routed traffic, complete expert
outputs, and the production decoder. It may reject the proposal but cannot
establish model quality.

Build every changed representation into a fresh candidate pool. A complete
all-expert run must close exact bytes, table hashes, state replay,
official-source provenance, and atom ownership before behavioral evaluation.
The full-model gate uses the frozen EXL3 comparison ledger and paired document
windows. Only that gate can support a claim of lower forward KL.

## Acceptance criteria for the research target

Pre-register thresholds before the complete model is encoded. At minimum, the
candidate must satisfy all of the following:

- **Accounting.** Recount the immutable EXL3 artifact, the two-bit candidate,
  and the shared non-expert overlay with one byte-accounting implementation.
  Report routed-expert and non-expert bytes separately. Hash equality must
  establish that both candidates use the same non-expert weights.
- **Two-bit claim.** A candidate described as a uniform two-bit representation
  uses exactly two trellis payload bits per routed-expert weight. Report its
  all-in rate after charging tables, scales, alignment, indices, and metadata.
  A mixed or allocated checkpoint must report its measured rate rather than
  inherit the uniform representation's label.
- **Compression dominance.** The complete QSRT checkpoint occupies fewer
  bytes than the complete EXL3 checkpoint and has lower paired forward KL on
  the same document-disjoint windows. Both inequalities are required.
- **Structural correctness.** Every trellis path closes, CPU and CUDA decode
  agree bitwise, all E4M3 labels are finite, and malformed tables or paths fail
  closed.
- **Teacher KL.** Mean forward KL is lower than uniform scalar QSRT and EXL3 on
  document-disjoint text and multimodal suites. For each baseline comparison,
  the lower bound of the paired document-bootstrap interval for
  baseline-minus-candidate forward KL must exceed zero. Report the remaining
  positive gap to the measured MXFP8 runtime reference.
- **Task quality.** Relative to the official source, the candidate remains
  within pre-registered noninferiority margins on the task suite. Report
  aggregate and per-domain results against the official source, EXL3, and
  MXFP8. A task gain does not substitute for the paired forward-KL gate.
- **Routing stability.** Top-16 route retention, applied-gate drift, expert
  traffic, and co-routing distributions remain within frozen bounds.
- **Long behavior.** The candidate must meet pre-registered noninferiority
  margins for long-context text, tool use, multilingual prompts, multimodal
  inputs, and multi-token generation. Every comparator must use the same
  frozen behavioral suite.
- **Runtime.** TP12 A16 latency meets the production gate for the fused pair
  decoder. Table loads and pair unpacking must not erase the bandwidth benefit.
- **Repetition.** A second seed or disjoint corpus reproduces the primary KL
  and task conclusion.

Coefficient error, isolated oracle gains, encoder microbenchmarks, and
complete-layer losses are diagnostics. None can substitute for the paired
full-model forward-KL and exact-byte gates.

## Repository implementation map

The repository contains a finalized NumPy falsification benchmark in
[`qsrt/tiny_improvement_benchmark.py`](../qsrt/tiny_improvement_benchmark.py),
with a command-line driver in
[`scripts/benchmark_qsrt_tiny_improvements.py`](../scripts/benchmark_qsrt_tiny_improvements.py)
and focused tests in
[`tests/test_tiny_improvement_benchmark.py`](../tests/test_tiny_improvement_benchmark.py).
It implements a matched-payload scalar K2 control, a 12-history-bit pair path,
a trained pair table, synthetic output-aware path choice, and a quantized down
target. Separate fit and held-out rows can expose regressions. Its synthetic
expert has one input coordinate, two intermediate neurons, and two outputs. It
omits BlockLDLQ, real routed activations, the production coupled transform, and
the production decoder. The benchmark can falsify a mechanism, but it cannot
qualify a Kimi-K3 representation or compare against EXL3.

The repository also contains a per-symbol synthetic-source distortion harness
in
[`qsrt/synthetic_source_distortion.py`](../qsrt/synthetic_source_distortion.py),
with a command-line driver in
[`scripts/measure_synthetic_source_distortion.py`](../scripts/measure_synthetic_source_distortion.py)
and tests in
[`tests/test_synthetic_source_distortion.py`](../tests/test_synthetic_source_distortion.py).
It runs an exact full CPU Viterbi over the shared L16 trellis on an
independent standard-normal source and compares reconstruction labels at one
trellis rate with a fitted global scale per code: the production
`sqg_xor_cheb_t12` labels, exact-rank SQG variants that isolate the T12
reduction and the E4M3 endpoint, a menu-oriented rank control, and CPU
transcriptions of ExLlamaV3's MCG and MUL1 codebooks validated against the
encoder's own codebook constant. At 256 sequences the production SQG labels
measure 2.62% (K2) and 3.00% (K3) lower per-symbol error than MCG, and the
exact-rank ablations bound the T12 and E4M3 costs below 0.15% at both rates.
These are coding measurements on a synthetic source; they rank labels on the
shared graph and do not predict held-out model quality.

### Synthetic stress result

A 256-expert sequential sweep provides a stronger falsification check than the
default eight-expert demonstration. With the pair table and synthetic experts
both using a correlation of `0.7`, replacing scalar K2 with the coefficient-fit
pair table increased pooled held-out forward KL by 0.72%. The pair table
improved 164 experts and regressed 92. Selecting a pair path by fit-row forward
KL then reduced pooled held-out forward KL by 13.41% relative to that pair-table
stage; 191 experts improved, 16 regressed, and 49 selected the same path. The
quantized down-target step reduced the remaining pooled held-out forward KL by
60.24%; 235 experts improved, 5 regressed, and 16 kept the preceding down
matrix. The complete sequence reduced pooled held-out forward KL by 65.33%
relative to scalar K2.

A distribution-shift control trained the pair table at correlation `-0.7`
while retaining the benchmark's positively correlated source experts. The pair
table then increased pooled held-out forward KL by 26.86% and regressed 216 of
256 experts. These results reject a generic marginally trained pair table as a
quality improvement. They do not reject a layer-trained pair table fitted on
the production transformed, BlockLDLQ-feedback domain. They support testing the
model-loss path objective before changing the stored representation.

The results are reproducible with
`run_sweep(256, pair_table_correlation=0.7)` and
`run_sweep(256, pair_table_correlation=-0.7)`. They remain synthetic four-logit
forward-KL measurements, exclude exact table and metadata bytes, and provide no
evidence of full-model or EXL3 superiority.

A production-shaped pair experiment still requires:

- a real-expert pair-table constructor beside `qsrt/sqg_e4m3.py`;
- a CPU reference for production-length pair paths with 12 history bits and 16
  outgoing branches per state in `qsrt/sqg_quantizer.py`;
- an offline CUDA pair-cost and traceback path under `qsrt/csrc`;
- pair-aware dense feedback in `qsrt/exl3_encoder_backend.py`;
- activation-metric and down-target experiments extending
  `qsrt/coupled_expert_study.py`;
- a dedicated real-expert driver under `scripts/` with document-disjoint fit,
  confirmation, and external-validation inputs; and
- tests for graph bijection, pair packing, tail biting, metric equivalence,
  E4M3 table identity, scale closure, and whole-expert replay.

The first research encoder should emit no production candidate schema. After
the panel and cross-layer experiments pass, add an explicit format revision,
fresh candidate-pool schema fields, table hashes, exact byte accounting, and a
pair-trellis atom profile. Existing candidate pools cannot be resumed because
changing the vector dimension, table, objective, or graph changes Viterbi
paths.

## Research priorities

The evidence supports the following order:

1. Produce multiple document-disjoint BF16 reference-logit contexts without
   downloading the complete BF16 checkpoint. Repeat uniform K3 and down
   refitting with document-clustered uncertainty.
2. Capture bounded model-loss output gradients and build real expert-local
   factors for the implemented two-sided scalar encoder. Verify on disjoint
   documents that its score predicts forward KL better than routed expert
   squared error.
3. Repeat reconstructed-activation down refitting across early, middle, and
   late mixture layers after the document-level result passes.
4. Recount EXL3 and QSRT bytes, freeze the common non-expert overlay, and use a
   validated damage score to compare K3 and K4 candidates per serialized byte.
5. Run the pre-registered sparse-residual concentration oracle. Build a
   residual payload only if it exceeds the 6% or 10% concentration threshold
   and beats K4 per added byte.
6. Implement a production-length pair-trellis reference only if scalar
   diagnostics show that transition geometry still limits quality after the
   objective and down-target experiments.
7. Run complete-layer transfer gates, then build a complete artifact for the
   paired compression-dominance comparison with EXL3.
8. Use bounded forward-KL scale or target recovery only if the hard
   post-training representation transfers but misses the model target.

This order establishes the comparator before optimization, tests the
format-preserving objective change before the higher-risk pair representation,
and reserves model-quality claims for paired complete-artifact evaluation.

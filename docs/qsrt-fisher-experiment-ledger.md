# QSRT Fisher experiment ledger

This ledger identifies the data flow and full-model result for QSRT encoders
that use final-logit curvature or gradients.  Entries are comparable only when
they use the same KLD suite identity shown below.

## Common inputs

- Uniform-K2 checkpoint:
  `/data/releases/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-v1`
- Uniform-K2 candidate pool:
  `/data/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-CANDIDATES-v1`
- Dense input factors:
  `/data/datasets/kquant/hessians/k3-denseh-broad-v7-4m-train-h13-identity-qsrt-v1.kqhess`
- Decoded-upstream input sample cache:
  `/data/datasets/kquant/captures/k3-denseh-broad-v7-4m-train-input-v1.kqsamples`
- Final-logit Fisher output factors:
  `/data/datasets/kquant/hessians/k3-official-mxfp4-final-logit-fisher-100k-v1-output-factors`
- Fisher capture provenance:
  `/data/datasets/kquant/hessians/k3-official-mxfp4-final-logit-fisher-100k-v1-output-factors/output-fisher-run.json`
  records 100,000 tokens from 104 documents and an official-MXFP4 forward
  anchor.
- KLD analysis suite: 768 contexts and 1,572,096 scored positions, identified
  by suite manifest SHA-256
  `f3a79f7f28365d406a19a82cf210c25adf18974c4b9b607ab3754e9939f941cf`.
  The reference-hidden manifest SHA-256 is
  `f0ea6a8575d7b64860ddc1ecdb7252ef8aae5e43b99a06dcbb13cd09bf607c2d`.

## Full-model results

| W1/W3 encoder | W2 encoder | Model artifact | Mean KL | Top-1 agreement | Conclusion |
| --- | --- | --- | ---: | ---: | --- |
| Canonical dense-H | Canonical decoded-upstream H2 | `/data/releases/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-K2-v1` | 0.067589265243 | 93.278082% | Uniform-K2 reference |
| Canonical dense-H | Final-logit Fisher, damping ratio 3 | `/data/models/Kimi-K3-QSRT-K2-W2-FINAL-LOGIT-FISHER-FULL-D3-v1-model` | 0.067394354571 | 93.281962% | 0.2884% lower KL; weak positive signal |
| Final-logit Fisher, damping ratio 3 | Final-logit Fisher, damping ratio 3 | `/data/models/Kimi-K3-QSRT-K2-FINAL-LOGIT-FISHER-ALL-LINEARS-D3-v1-model` | 0.067953612547 | 93.331260% | 0.5391% higher KL; W1/W3 Fisher arm rejected |
| One-sweep rank-32 final-KL gradient, strength 40 | Final-logit Fisher, damping ratio 3 | `/data/models/Kimi-K3-QSRT-K2-FINAL-KL-GUIDED-v1-model` | 0.070964324002 | 93.094760% | 4.9935% higher KL; unconstrained linear guidance rejected |
| Canonical dense-H | Inverse of damped final-logit Fisher | `/data/models/Kimi-K3-QSRT-K2-W2-INVERSE-FINAL-LOGIT-FISHER-FULL-D3-v1-model` | 0.071518720125 | 92.993112% | 5.8137% higher KL; Fisher-direction falsification passed |
| Direct tile Viterbi | Direct tile Viterbi | `/data/models/Kimi-K3-QSRT-K2-DIRECT-VITERBI-ALL-LINEARS-v1-model` | 0.062993208155 | 93.571003% | Unconditioned-path control |
| Final-KL-gradient direct Viterbi on layers 89-92 | Final-KL-gradient direct Viterbi on layers 89-92 | `/data/models/Kimi-K3-QSRT-K2-DIRECT-VITERBI-FINAL-KL-GRADIENT-L89-92-v1-model` | 0.062976057133 | 93.574883% | Directionally positive; paired context interval includes zero |

The KLD result files are, in table order:

- `/data/kld/kimi-k3-qsrt-k2-baseline-distribution-fidelity-1024x2048-analysis/analysis-kld.json`
- `/data/kld/kimi-k3-qsrt-k2-w2-final-logit-fisher-full-d3-distribution-fidelity-1024x2048-analysis-v1/analysis-kld.json`
- `/data/kld/kimi-k3-qsrt-k2-final-logit-fisher-all-linears-d3-distribution-fidelity-1024x2048-analysis-v1/analysis-kld.json`
- `/data/kld/kimi-k3-qsrt-k2-final-kl-guided-v1-distribution-fidelity-1024x2048-analysis/analysis-kld.json`
- `/data/kld/kimi-k3-qsrt-k2-w2-inverse-final-logit-fisher-full-d3-distribution-fidelity-1024x2048-analysis-v1/analysis-kld.json`
- `/data/kld/kimi-k3-qsrt-k2-direct-viterbi-all-linears-distribution-fidelity-1024x2048-analysis-v1/analysis-kld.json`
- `/data/kld/kimi-k3-qsrt-k2-direct-viterbi-final-kl-gradient-a0p0078125-l89-92-distribution-fidelity-1024x2048-analysis-v1/analysis-kld.json`

## Inverse-Fisher falsification

The counterfactual applies the spectral inverse to each captured W2 output
factor after adding the same diagonal damping used by the positive arm:

```text
F_damped = F + 3 * mean(diag(F)) * I
F_counterfactual = trace_match(inverse(F_damped), F_damped)
```

The encoder receives `F_counterfactual` without additional damping.  W1/W3,
the coupled transform, reconstruction law, rate, stored payload size, input H2,
and KLD suite remain unchanged.  The overlay build is stored at
`/data/kquant/research/k3-uniform-k2-w2-inverse-final-logit-fisher-full-d3-batched-v1`.
Mean KLD orders the three W2 objectives as normal Fisher
(`0.067394354571`), uniform K2 (`0.067589265243`), then inverse Fisher
(`0.071518720125`). The inverse arm is 5.8137% worse than uniform K2 and
6.1197% worse than normal Fisher. This separation establishes that the weak
positive W2 result is directionally attributable to the captured Fisher
eigensystem rather than arbitrary two-sided perturbation.

## Direct-Viterbi control

The direct-Viterbi control encodes every transformed W1, W3, and W2 16-by-16
tile with the ordinary tail-biting SQG path search.  It uses neither dense-H
target feedback nor an output Fisher factor for any projection.  The coupled
activation-boundary transform uses the exact per-expert draw stored by the
uniform-K2 profile. The fixed internal SQG transform draw, per-matrix global
reconstruction-scale search, K2 graph and scalar law, tail-biting context, and
payload layout remain unchanged.

The overlay destination is
`/data/kquant/research/k3-uniform-k2-direct-viterbi-all-linears-v1`.
Comparing this arm with the uniform-K2 reference and final-logit Fisher
separates unconditioned SQG reconstruction from one-sided and two-sided
curvature guidance.

## Direct final-logit-gradient Viterbi refinement

The direct-gradient encoder augments each legal SQG path cost with the exact
linear term from the token-summed final-output objective while retaining the
direct-Viterbi anchor's reconstruction scales. It does not use BlockLDLQ,
dense-H feedback, Fisher factors, or a fitted replacement target. The gradient
archive contains the dense W1, W3, and W2 gradients for layers 89 through 92
under

```text
KL(official MXFP4 || direct-Viterbi K2)
```

over 100,000 tokens. Its root is
`/data/kquant/research/qsrt-fp32-kl-refinement-layers89-92/gradients`.

The selected step coefficient is `0.0078125`. On the 768-context analysis
partition, covering 1,572,096 scored token positions, it changed mean KL from
`0.062993208155` to `0.062976057133`, a reduction of `0.02723%`. Top-1
agreement increased from `93.571003%` to `93.574883%`. The candidate improved
399 contexts and regressed 369. A paired context bootstrap interval for the
mean KL change was `[-0.00013288, 0.00009888]`, so the measured improvement is
not statistically resolved by this partition.

Larger same-gradient steps of `0.015625`, `0.03125`, and `0.125` regressed on
the 32-context screening partition. A second refinement step therefore
requires a fresh gradient capture around the first refined checkpoint; reusing
the existing gradient is equivalent to a larger step and is not a valid
successive update.

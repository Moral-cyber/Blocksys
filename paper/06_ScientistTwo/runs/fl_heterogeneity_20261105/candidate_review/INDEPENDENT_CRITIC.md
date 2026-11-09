# Independent novelty and mechanism critique

Author: `/root/route_a_citations`, acting as ScientistTwo Novelty Checker and
Independent Critic through actual Codex native collaboration. Reviewed on
2026-11-05. The three candidates in `../roles/seeds.json` were inspected before
any candidate performance score. The `idea-evaluator` skill's fatal-flaw and
nearest-mechanism checks were applied; its subjective score template is omitted
in favor of the requested **Good / Engineer / Bad** decisions. No network request,
new dependency, model training, candidate fit, TEST access or manuscript edit was
performed for this review. A small algebra fixture below is not FL performance
evidence.

**Decision: none of these candidates currently establishes a new core
contribution. Permit only Seed 1's corrected development screen.** Its strongest
rival is ordinary count-weighted ridge, not just the old isotropic predictor.
Seed 2 should not consume the next screen, and Seed 3 should remain parked.
Framework installation and a schema-valid seed do not affect these decisions.

## Bounded verdicts

| Candidate | Verdict now | Reason and what would change the verdict |
|---|---|---|
| `count_geometry_v1` | **Engineer; conditional mechanism screen** | It is an analytic anisotropic L2 penalty derived from a familiar Dirichlet log-ratio covariance. Its precision has a simple weighted-ridge closed form and is close to a no-trigamma count-ridge. Trace matching and a permuted precision do not exclude that simpler explanation. A fixed screen can identify whether geometry adds anything beyond that rival and scalar shrinkage; a numerical win over the old λ=10 alone is insufficient. |
| `residual_marginal_v1` | **Bad as the next core-contribution candidate** | The logarithmic prior ratio is standard label-shift correction, and local TRAIN predictive mass is generally not the source prior required by the correction. At finite λ this is ordinary calibration around a heuristic offset. It might be a useful negative diagnostic, but it has weak novelty and a missing identification argument. Do not run it ahead of Seed 1. |
| `support_scales_v1` | **Engineer; park** | This is support-conditioned partial vector scaling. It is not exactly BCTS, but the distinction is a mask over existing affine-calibration coordinates, not a new heterogeneity principle. Matched-center BCTS can explain a win through lower capacity; an equal-cardinality wrong mask can explain it through generic restriction. No direct duplicate of this exact support rule is established by this bounded review, and no novelty claim is established either. |

Here **Good** would mean a distinct, defensible mechanism worth advancing beyond
its simple rivals. **Engineer** means a plausible useful construction whose
scientific contribution is not yet separated from familiar methods. **Bad** is a
decision about the stated research claim and next expenditure, not proof that a
heuristic can never improve a number. None is labeled Good without evidence.

## Seed 1: trace normalization is necessary but not sufficient

Let `K=10`, `H=I-11'/K`, `alpha=n+1`, `d=trigamma(alpha)`, and
`C=H diag(d) H`. Its Moore-Penrose inverse has the exact closed form

```text
w_c = 1 / trigamma(alpha_c)
Q = diag(w) - w w' / sum(w)
Q_bar = (K-1) Q / trace(Q).
```

For any centered displacement `delta=b-mu`, the penalty is

```text
delta' Q delta = sum_c w_c delta_c^2
                - (sum_c w_c delta_c)^2 / sum_c w_c.
```

This is weighted quadratic shrinkage with a rank-one correction for translation
invariance. It is not a new nonquadratic prior or a new optimizer. The equality
follows because `Q1=0`, `QH=Q`, and `Q D H=H` on the centered subspace. For
large counts, inverse trigamma satisfies `w_c = n_c + 1/2 + O(1/(n_c+1))`.
Consequently, the deterministic simple rival

```text
v = n + 1/2
R = diag(v) - v v' / sum(v)
R_bar = 9 R / trace(R)
```

is almost the same construction without a special function or covariance
pseudoinverse. It must be compared directly. Do not adjust the `1/2`, trace
normalizer, smoothing or λ after seeing candidate scores.

There is also a direct Dirichlet/KL connection. In centered logits `b`, a
Dirichlet density transformed to logits has negative log-density, up to a
constant, `alpha_0 logsumexp(b) - alpha'b`. Its mode is
`mu=H log(alpha)` and its Hessian there is
`diag(alpha)-alpha alpha'/alpha_0`. Thus count-weighted quadratic shrinkage is
already the local curvature of a conventional Dirichlet/KL construction. This
is an algebraic relation, not a claim that the current discriminative objective
is a fully specified Bayesian posterior. In particular, using the same labels
in the CE fit and count prior does not justify a new posterior-calibration
guarantee. The Dirichlet mean of centered log probabilities is `H digamma(alpha)`,
whereas the proposed center is `H log(alpha)`; an inverse-covariance quadratic
around the latter is an approximation, not the exact posterior density.

Normalizing `trace(Q_bar)=9` fixes the arithmetic mean precision over the nine
centered directions. It does **not** match its eigenvalues, alignment with the
CE Hessian, effective degrees of freedom, or shrinkage of the fitted residual.
The fit can behave like a favorable change of regularization strength even with
equal trace. It also removes the absolute concentration scale from `Q`, so the
mechanism is primarily relative within-client class geometry. It must not be
described as preserving the full amount of client count uncertainty.

Permuting `Q_bar` alone is a useful class-alignment control because the center,
spectrum and trace stay fixed. It cannot distinguish the proposed covariance
interpretation from familiar frequency-aligned weighted ridge: the simple rival
uses the same class/count alignment. A successful permutation contrast supports
alignment, not a unique uncertainty mechanism.

An actual FP64 algebra fixture compared this closed form with an independent
matrix pseudoinverse, without fitting any classifier. For count vectors
`[0,0,1,2,5,10,20,50,100,500]`, ten copies of `10`, and
`[450,0,0,0,0,0,0,0,0,0]`, the largest absolute disagreement was
`9.10e-13`. The relative normalized-matrix differences from the simple
`n+1/2` rival were approximately `0.00180`, `0`, and `0.000973`. These invented
count vectors only check the mathematical identity and expose the close rival;
they say nothing about task NLL. Uniform counts reduce both normalized penalties
exactly to `H`, recovering the original isotropic centered-bias penalty.

## Seed 2: the denominator is not identified

Under label shift and a calibrated source posterior,
`p_target(c|x) proportional p_source(c|x) pi_target(c)/pi_source(c)`.
The corresponding logit offset is the standard log prior ratio. In Seed 2,
however, `q_i=E_{x drawn from local target TRAIN} p_source(c|x)` is a mixture
under the client's inputs. Generally it equals a soft confusion mapping applied
to the target prior, not the source prior. Self-consistency can make `q_i`
close to the target counts and collapse the proposed center toward zero. The
existing zero-center comparison is therefore indispensable if this is ever
used diagnostically.

Alexandari, Kundaje and Shrikumar (2020), §3.2, already discusses estimating a
source prior by the mean predicted probability **over the source domain** to
mirror predictor bias. Substituting the local target mixture changes the
assumption, not the familiar ratio algebra. Its §3.1 already defines BCTS and
class-bias correction. A finite quadratic fit around a ratio offset can be
rewritten as `b=mu_ratio+r` with an ordinary L2 penalty on `r`; this does not
create a new generic prior-centered principle. The direct primary full text
strengthens the existing residual-anchor critique; a broader network search is
unnecessary for the present no-priority decision.

## Seed 3: support restriction differs from BCTS but remains a capacity choice

In BCTS, one shared scale and classwise biases are fitted. In Seed 3, observed
classes have separate scales and absent classes keep scale `1`, while all biases
remain fitted. These parameter sets are generally **not nested**: shared scaling
of absent classes is allowed in BCTS but excluded by the mask; different
observed-class scales are allowed by the mask but excluded by BCTS. Therefore it
would be inaccurate to call the seed exactly BCTS, or to attribute a difference
automatically to a strict capacity ordering.

The support mask is nonetheless a direct partial vector-scaling construction.
Absent classes supply negative-class likelihood information; absence of positive
examples does not make their scale mathematically unidentifiable once logits
vary. The justification is variance/bias control, not a proof that those scales
cannot be estimated. A single shared-scale method or a count-blind restriction
can explain a favorable result. If ever revisited, retain a matched-prior-center
BCTS with the same fit data, CE normalization and a clearly declared shared-scale
penalty, plus a wrong mask of the same cardinality. Merely using λ=10 for both
does not match scale regularization: restricting the original ten-scale ridge
to a common scale gives `10 * lambda/2 * (a-1)^2`, while a literal one-coordinate
ridge gives `lambda/2 * (a-1)^2`. State which is being tested. Keep the existing
published-convention BCTS separately named.

At inference all three seeds retain the same 80 B layout and zero additional
uplink; they do not improve that already established persistent budget. A support
mask can reduce fitted degrees of freedom and solver work, but those are separate
claims that need measurement. This small change is not the highest-priority
route to a stronger core contribution.

## Exactly one recommended development screen

Recommend **Seed 1 only**, with this correction to its current plan. Root executes;
this document launches nothing. The candidate code/constants/input hashes and
this interpretation must be frozen before the first candidate score.

Use saved `fedlc_uniform3` TRAIN/development arrays only, fixed original
partitions, all 20 clients, with λ=10 for the candidate geometry and unchanged
scale penalty. No checkpoint loading, new CNN forward, retraining, official TEST
read, deployment claim or manuscript promotion. Use these six reported arms:

| Arm | Purpose |
|---|---|
| Saved zero-center, original λ=10 | Free reference for whether local prior information still helps; not an extra candidate |
| Original prior, isotropic λ=10 | Primary matched reference; replay must reproduce saved states/metrics |
| Original prior, isotropic λ=1 | Fixed simple scalar-ridge rival; the already inspected sensitivity curve shows that scalar strength matters. This is a development comparator chosen before this candidate's scores, not a blind confirmation result. No new λ sweep or choice. |
| Proposed `Q_bar`, λ=10 | Candidate |
| `P Q_bar P'`, λ=10 | Fixed cyclic one-class permutation of precision alone; keep the center unchanged |
| Simple `R_bar` using `n+1/2`, λ=10 | Required no-trigamma count-ridge rival |

First run both tasks with the first predeclared seed of each, all 20 clients,
as a **fail-fast resource screen**. This is narrower than the seed agent's full
three-seed minimum and cannot establish cross-seed benefit. A successful first
seed only permits expansion to all three original development seeds. An early
failure stops expenditure on this version; it is not statistical evidence that
every seed must fail.

Before score inspection, define the numerical equality tolerance from solver and
saved-state replay fixtures, not an arbitrary scientific improvement threshold.
Report projected gradients and all centered eigenvalues; a failure to converge
is indeterminate, not a candidate loss or win. Compare FP32 deployed states in
the same FP64 metric path as the original. Record all six arms and every client,
including adverse accuracy/ECE results.

The first-seed screen can justify expansion only if the proposed geometry beats
the original λ=10 and the precision-permuted control in client-macro NLL on **both
tasks**, beyond that numerical tolerance, and is not weakly dominated across the
two tasks by either the simple count-ridge or fixed scalar λ=1 rival. If it is
numerically equivalent to the count-ridge, record **Engineer / ordinary
count-weighted shrinkage** and do not expand just to manufacture a novel
uncertainty story. If the two tasks split or any claimed alignment contrast is
uninformative, retain the mixed/null result and stop this formulation.

After all three development seeds, any proposed scientific advancement still
requires a mechanism-specific advantage over the simple rivals on both tasks,
complete seed-level results, and a separately frozen confirmation decision.
The current additional partitions are available for confirmation only if their
candidate outcomes have not influenced design; otherwise they are development.
No p-value or internal ScientistTwo score turns the screen into novelty proof.

## Sources inspected and boundaries

The source manifest beside this report records SHA-256 of the exact local files.
Full-text anchors used for mechanism judgments:

- Guo, Pleiss, Sun and Weinberger, *On Calibration of Modern Neural Networks*,
  ICML 2017, local `guo_cal.txt`: fixed-network calibration, vector scaling,
  temperature scaling and capacity/overfitting discussion.
- Alexandari, Kundaje and Shrikumar, *Maximum Likelihood with Bias-Corrected
  Calibration is Hard-To-Beat at Label Shift Adaptation*, ICML 2020, local
  `bcts.txt`, §2.1, §3.1, §3.2: standard ratio adaptation, BCTS, and predictive
  source-marginal estimation. URL: <https://arxiv.org/abs/1901.06852>.
- Menon et al., *Long-Tail Learning via Logit Adjustment*, ICLR 2021, local
  `menon_la.txt`, §3/Eq.8: classic post-hoc log prior adjustment. Its balanced-error
  objective differs from local NLL.
- Shi et al., *PRIOR: Personalized Prior for Reactivating the Information
  Overlooked in Federated Learning*, NeurIPS 2023 (PRIOR/pFedBreD), local
  full-text equation extract Eq.8/11/14: personalized
  Bregman/model-space centers are prior art. URL:
  <https://arxiv.org/html/2310.09183v2>.

The covariance, inverse and Dirichlet-logit density relations above are explicit
mathematical derivations, not invented claims about those papers. No source here
proves the exact count-precision or support-mask proposal has been published.
Conversely, the lack of an exact duplicate in these bounded sources does not
prove originality. LaSCal and FedNPC remain method-unresolved in the supplied
near-neighbor note and are not used to manufacture a non-overlap claim. This
bounded decision is sufficient to choose one cheap discriminating screen now.

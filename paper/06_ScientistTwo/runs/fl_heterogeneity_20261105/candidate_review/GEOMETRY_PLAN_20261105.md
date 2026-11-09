# Count-geometry development screen

Predeclared 2026-11-05, before candidate fits or scores. Root authorized this exact bounded screen and conditional extension; no further approval is needed for these runs. This is the native ScientistTwo research-loop adaptation, not an upstream live campaign or evidence of novelty. Ratio-anchor candidate is not executed.

## Candidate and rivals

Keep the original frozen `fedlc_uniform3` logits, twenty-coordinate affine predictor, prior center `mu=H log(n+1)`, centered bias, scale penalty `10/2||A-1||²`, FP64 Newton/Armijo and FP32 deployment. Let `H=I-11'/10`, `d=trigamma(n+1)`, `C=H diag(d) H`, `Q=C+`, `Qbar=9Q/trace(Q)`. Replace only the bias penalty by `10/2(b-mu)'Qbar(b-mu)`. The closed form `Q=diag(w)-ww'/sum(w)`, `w=1/d`, is checked against the pseudoinverse. This is an established count-dependent weighted ridge construction; numerical success alone does not establish uncertainty-specific novelty.

All six arms are reported without post-result tuning:

1. Original prior-centered vector, both ridge coefficients10: exact old FP32-state and metric replay.
2. Zero-centered vector, both ridge coefficients10: exact old FP32-state and metric replay.
3. Geometry: the analytic Qbar, original mu and scale penalty10.
4. Misaligned geometry: fixed one-class cyclic permutation `Qbar[np.ix_(roll(arange10,1),roll(arange10,1))]`, same eigenvalues and mu, scale penalty10.
5. Count ridge: `w=n+0.5`, `Q_count=diag(w)-ww'/sum(w)`, normalized trace9, same mu and scale penalty10. This strong simple rival targets the trigamma-versus-frequency explanation.
6. Fixed isotropic lambda1: both scale/bias coefficients1, original mu, original numerical schedule. This value is chosen before this screen from the already-run fixed sensitivity, not selected here. It is a reference to a known scalar alternative, with its differing scale coefficient explicit.

The center-specific comparisons have matched information/capacity. Trace normalization matches average curvature, not effective shrinkage relative to the CE Hessian; cyclic misalignment does not by itself rule out ordinary frequency-weighted ridge.

## Frozen data and phases

Read only saved original development TRAIN/holdout logits/labels for `fedlc_uniform3`: CIFAR partition20261217/seeds20261231/32/33; Fashion partition20261219/seeds20261241/42/43. Audit-reject official TEST, predictive_final_test, raw dataset paths and model checkpoints. No new CNN forward, backward, training or uploaded bytes. No new-partition outputs may be read during design or this screen. Fit and score all20clients; no favorable-client selection. Every new score is development evidence, never a substitute for original TEST.

Phase1 is a fail-fast subset: CIFAR seed20261231 and Fashion seed20261241 only (40clients total), four new fits/client; free original/zero replays. Save all six arms and adverse results. Primary endpoint is paired client-macro holdout NLL. Secondary metrics are macro/sample-weighted accuracy/NLL/Brier/ECE15, pooled ECE and clientaccuracy q10; no alternate metric can rescue a failed primary screen.

Before any fit, lock PLAN, source code and exact first-seed input hashes. Numerical fixtures must establish isotropic frozen-solver FP32 exactness, generic-penalty finite-difference derivatives, positive curvature on centered bias space, KKT projection, Q pseudoinverse/closed-form identity and metric known cases. Nominal numerical contrast tolerance is `epsilon_nll=1e-7`; accept a fit only with projected gradient <=1e-5 and objective monotonicity. No numerical repair changes hyperparameters silently.

## Predeclared fail-fast decision

Let N_g,N_p,N_perm,N_c,N_1 be task macro NLL for geometry, original prior10, permutedgeometry, countridge and isotropic1. Extend to the remaining two fixed seeds per task only if:

- Both tasks satisfy `N_g < N_p - 1e-7` and `N_g < N_perm - 1e-7`.
- Geometry is not weakly dominated across both tasks by either simple rival: for each rival r in {count,iso1}, it must not be true that `N_r <= N_g + 1e-7` on both tasks.
- Geometry and count-ridge are not numerically equivalent on both tasks (`abs(N_g-N_c)<=1e-7`). Such equivalence ends the uncertainty-specific mechanism claim as a frequency-ridge engineering observation.

Any failed task or gate stops this candidate after Phase1. A pass triggers only full three-seed development confirmation, never a publication/novelty conclusion. Retain the gate outcome, all contrasts and reversals. Extension uses the same locked code/settings/all20clients and reports every fixed seed. Separate external literature review is needed before promoting any mechanism; generic centered or anisotropic regularization is prior art.

## Outputs and cost

Write only this new revision artifact directory and its plan/conclusion copies in `06_ScientistTwo/.../candidate_review`. Preserve original13 numeric groups, old outputs and all `03_英文论文` manuscript text. Save per-client six-arm metrics, exact FP32 fitted states, counts/geometry eigenvalues, optimizer diagnostics and source/input hashes. Report effective support `exp(entropy((n+1)/(N+10)))`; do not select bins after scores. Deployment remains80B,0extra-upload. Named geometry arrays and peak numerical Python allocation are reported separately; neither is a total-energy or network result.

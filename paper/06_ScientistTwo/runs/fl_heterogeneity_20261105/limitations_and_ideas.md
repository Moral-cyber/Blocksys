# Native limitation extraction and mechanism seeds

Role output date: 2026-11-05, Asia/Shanghai. Author: the active Codex Limitation Extractor / Idea Generator role, not the upstream framework's scripted demo. Status: **candidate generation only; no candidate has been run, selected, or established as novel**.

The `hypothesis-generation` skill was applied to keep observations, proposed explanations, rival explanations and falsification tests distinct. This local role uses the user-authorized paper workspace and saved research outputs. It sends no unpublished material to an external service. The existing paper and protected evidence remain unchanged.

## Reference contract and evidence boundary

The source IDs below refer to task/literature documents. Numerical observations in these source documents have not yet been imported as committed native-controller observations. Therefore `evidence_ids` in every seed is empty. A controller must register these source IDs before ingesting the role JSON; it must not silently reinterpret them as trial IDs.

| Source ID | Supplied local source |
|---|---|
| `task_pr_lfl` | `06_ScientistTwo/runs/fl_heterogeneity_20261105/PROBLEM.md` |
| `prior_pfedbred` | `/private/tmp/pr_lfl_route_a/03_英文论文/revision_artifacts/route_a_20261105/literature/near_neighbors/prior_full.txt` and the associated author full text, <https://arxiv.org/html/2310.09183v2> |
| `near_neighbor_comparison` | `/private/tmp/pr_lfl_route_a/03_英文论文/revision_artifacts/route_a_20261105/literature/near_neighbors/METHOD_COMPARISON.md` |
| `lambda_development` | `/private/tmp/pr_lfl_route_a/03_英文论文/revision_artifacts/route_a_20261105/p1/PLAN.md` and `lambda_result.json` |
| `drift_development` | `/private/tmp/pr_lfl_route_a/03_英文论文/revision_artifacts/route_a_20261105/p1/DRIFT_PLAN.md` and `drift_result.json` |

`novelty_score=0.0` is a required-schema **UNASSESSED transport placeholder**, not a measured score, a scientific ranking, or a conclusion that novelty is zero. These seeds require a separate novelty check before promotion. The supplied comparison establishes real overlap with PRIOR/pFedBreD's generic personalized regularization centers; it does not resolve the full methods of LaSCal or FedNPC. No claim of universal non-overlap is made.

## Limitation extraction

1. **The information source is valuable but not stationary by construction.** The current center uses local TRAIN class counts. In the frozen-predictor class-reallocation diagnostic, the prior-center advantage reverses as the local label distribution moves away from that TRAIN membership. This diagnoses a boundary of a stale predictor; it is not evidence that any proposed adaptation works.
2. **A single quadratic geometry ignores how uncertain different log class frequencies are.** An observed count of zero after smoothing and a large observed count enter the center with the same regularization weight. The existing lambda scan tests one scalar strength, not class-dependent uncertainty or a covariance geometry.
3. **The source of the desired bias correction is not identified by the center alone.** A local class prior and the frozen model's existing class evidence may overlap. Reusing local frequencies as a bias center could help through shrinkage, label-shift correction, or compensation for class-dependent logit defects. Existing comparisons do not isolate those explanations.
4. **Twenty stored coordinates do not imply that twenty freely fitted coordinates are needed.** Client TRAIN sets miss classes, yet every class receives a fitted scale. BCTS is an essential compact rival, and the current ECE comparison does not support a universal calibration advantage for the 20-coordinate predictor.
5. **The present evidence concerns label-skew clients and limited partition realizations.** It cannot establish feature-shift robustness or arbitrary non-IID generality. New partition experiments are running separately; their outcomes must not be anticipated in seed descriptions.
6. **Generic personalized priors are not a new principle here.** PRIOR/pFedBreD already supplies centered regularization in personalized model training. A stronger contribution needs a specific information source or fitting mechanism, its compact budget, and an attribution control against familiar calibration/shrinkage methods.

These limitations are relevant to the lightweight label-heterogeneity research question. They are not an acceptance/rejection decision about the manuscript and do not authorize changing its protected numbers.

## Shared development protocol for all three candidates

- Freeze the existing shared CNN checkpoints. Use only their local TRAIN logits/labels to fit each candidate. Read only the already designated development holdouts for scoring; do not access or select on official TEST. The original TEST results stay fixed.
- First use both tasks, the original partition for each task, all three training seeds and all 20 clients: 120 client fits per method. This is a low-cost frozen-logit development experiment, not a new training matrix. Freeze candidate code, source hashes, all constants and this plan before reading candidate scores.
- Primary contrast: paired client-macro development NLL versus the original prior-center method, with seed-level summaries retained. Report accuracy, sample-weighted metrics, per-client distribution and ECE as secondary trade-offs. Do not choose a mechanism because only one secondary metric improves.
- No p-value promotion threshold is prescribed for these exploratory fits. Report effect sizes by task and seed and a client-cluster descriptive interval that keeps the three training seeds together. Clustering cannot turn one partition into population-level replication.
- A candidate is eligible for a separate confirmation plan only if its primary mechanism-specific prediction survives both tasks, all three seeds are reported, and the attribution control is informative. A numerical win alone is insufficient. The two new partitions, once completed, may provide confirmation **only if no candidate outcomes from those partitions were inspected during design**; otherwise label them further development.
- Retain adverse, null and indeterminate outcomes. Set no post-result sweep over thresholds, normalizers, smoothing constants or permutations. Any changed candidate becomes a new dated seed, not an invisible repair of the old one.
- Every candidate has **0 extra uploaded bytes** and no shared-training modification. Store the fitted ten scales and ten centered biases as FP32 (80 B). Report transient local fitting memory separately; 80 B is not a claim about total training memory, saved logits, or optimizer workspace.

## Seed 1: Count-uncertainty geometry for the bias center

**Question.** Does the geometry of count uncertainty matter beyond a scalar lambda when shrinking a compact predictor toward local label information?

**Mechanism candidate.** Let `alpha_c=n_c+1`, `H=I-11'/K`, and keep the original center `mu=H log(alpha)`, `K=10`, and original scale penalty. Under a Dirichlet count model, the covariance of centered log probabilities is

`C = H diag(trigamma(alpha_c)) H`.

The common `-trigamma(sum(alpha)) 11'` term disappears under centering. Compute the Moore-Penrose inverse `Q=C^+` on the mean-zero subspace, then normalize it as `Q_bar=(K-1) Q / trace(Q)`. Replace only the bias penalty with `lambda/2 (b-mu)' Q_bar (b-mu)`, using the fixed original `lambda=10` and `1'b=0`. The trace normalization matches average curvature rather than selecting lambda from development data. This remains a convex frozen-logit fit; `Q_bar` is positive definite on the centered-bias subspace.

**Falsifiable hypothesis.** Correctly aligning precision with count uncertainty lowers client-macro NLL versus the isotropic prior penalty, especially for clients with smaller effective class support. Improvement should be reduced by a class-misaligned precision control while keeping the center and the same precision eigenvalues fixed. An improvement reproduced equally by a scalar rescaling of the old penalty would favor ordinary shrinkage over the proposed geometry.

**Nearest-work distinction and uncertainty.** Heteroscedastic Bayesian regularization and Dirichlet log-ratio covariance are established ideas. PRIOR/pFedBreD establishes personalized centers in model-parameter training. The candidate here is their specific composition into a 20-coordinate output-space fit with count-derived geometry and zero communication. This combination is a testable mechanism, not an established novel contribution. Search anisotropic calibration, empirical-Bayes output calibration and uncertainty-aware personalized FL before any novelty claim.

**Minimum experiment and attribution controls.** Run the shared protocol with (i) isotropic original prior, (ii) the analytic `Q_bar`, and (iii) a fixed cyclic permutation by one class of `Q_bar` alone, holding `mu` fixed. Also report the already frozen scalar-lambda development curve as context, not as a newly optimized comparator. Bin clients by TRAIN effective support `exp(H(pi))` using fixed taskwise thirds defined without candidate outcomes. Check projected gradient, objective monotonicity and the nine positive centered-subspace eigenvalues.

**Rival explanation.** Geometry merely changes global effective regularization strength, or strong precision hurts because counts do not model representation error. Normalization and the misalignment control target the first rival; a null or negative result under valid optimization supports the second boundary.

**Reject or leave indeterminate.** Reject this seed as a general improvement if task-mean NLL is not lower for both tasks, or if the aligned-versus-misaligned contrast has no consistent direction. If gains are within numerical tolerance or client intervals span material harm, retain an indeterminate result rather than promoting it. A win accompanied by an accuracy/calibration trade-off must be explicitly bounded.

**Budget.** Persistent predictor: 80 B; extra upload: 0 B. Count vector: 80 B int64. Anchor/eigenvalue vectors: 160 B FP64; covariance, precision and eigenvectors: 2,400 B FP64. These named analytical buffers total 2,640 B beyond the baseline fit; measure actual process/array peak because eigensolver temporaries are additional. All geometry is discarded after fitting. No new CNN forward pass is needed.

## Seed 2: Residual marginal-evidence anchor

**Question.** Does subtracting the frozen predictor's existing class evidence improve a class-prior anchor, or does it remove useful local information?

**Mechanism candidate.** From the same local TRAIN inputs, compute `q_c=(sum_j softmax(z_j)_c+1)/(N+K)` and `pi_c=(n_c+1)/(N+K)`. Fit the original convex 20-coordinate model with `mu=H[log(pi)-log(q)]`, fixed `lambda=10`, and every other numerical setting unchanged. No holdout labels define the center.

**Falsifiable hypothesis.** If the existing output marginal carries part of the class prior that would otherwise be counted twice, a residual center lowers NLL relative to both the original prior and zero center. The effect should depend on correctly matching `q` to class identity and the frozen checkpoint. If local `q` is already near local `pi`, the center approaches zero; this could be appropriate or could erase the prior advantage, which is a direct falsification risk.

**Nearest-work distinction and uncertainty.** Prior-ratio label-shift correction, Saerens-style adaptation and bias-corrected temperature scaling are familiar mechanisms. This candidate uses a **local TRAIN prediction marginal**, which is not guaranteed to equal the model's source prior. That distinction is a risk as well as a possible mechanism, not a novelty claim. A global pooled marginal can be nearly uniform in a balanced source and reduce the method to the original prior anchor. The candidate cannot claim valid label-shift correction without testing the assumptions and checking the closest calibration papers.

**Minimum experiment and attribution controls.** Run the shared protocol for prior, zero, local residual, and cyclically misaligned `q` residual centers. Log `||H log(q)||`, `||mu||`, agreement between `q` and `pi`, and class-mass residuals using TRAIN only. Add a synthetic unit fixture with known logits and known injected class offsets to check finite centers and detect exact/near-zero cancellation; do not treat that fixture as task-performance evidence. A separate pooled-TRAIN ratio is only permitted if its information availability and communication budget are explicitly frozen first; it is not part of this zero-upload local seed.

**Rival explanation.** The ratio helps only by moving closer to the already known zero center, or reflects same-TRAIN optimism rather than disentangling a source prior. Retain the zero-center comparator and report the anchor-distance diagnostics. Do not rename a numerical gain as proof of prior decomposition.

**Reject or leave indeterminate.** Reject if residual task-mean NLL fails to beat the original prior on either task, if it cannot improve over the zero center on either task, or if class-misaligned `q` behaves equivalently. A result explained only by proximity to zero is a shrinkage observation rather than evidence for residual evidence alignment. Large between-seed reversals leave the result indeterminate.

**Budget.** Persistent predictor: 80 B; extra upload: 0 B. Local streaming accumulator `sum p`: 80 B FP64; `q`, `pi`, and new center: 240 B FP64, plus at most one 10-probability FP64 row (80 B) beyond the existing saved logits. The named extra buffers total 400 B; vectorized implementations must measure additional temporary arrays. Discard these after fitting. No new network parameters or CNN forward pass is required.

## Seed 3: Observed-support scale fitting

**Question.** Are class-specific scales useful when the local TRAIN data contain no positive example of that class?

**Mechanism candidate.** Let `S={c:n_c>0}` from local TRAIN labels. Keep all ten bias coordinates, the original prior center, both original penalties and the centered-bias constraint. Optimize only scales with `c in S`; fix `a_c=1` for `c not in S`. This changes a structural degree of freedom using an observed support condition rather than choosing a count threshold on development results. The objective stays convex. Store the same FP32 20-slot predictor; the fixed slots cost bytes but no fitted degree of freedom.

**Falsifiable hypothesis.** Removing scales without local positive-label support reduces estimation variance and improves NLL for low-support clients while retaining the prior-center benefit. A count-blind restriction with the same number of fixed scales should be worse if support identity is the cause. Classes absent from local TRAIN still contribute negative-class likelihood information, so unrestricted scales may instead be useful; that is a genuine competing explanation.

**Nearest-work distinction and uncertainty.** BCTS shares one scale and classwise biases, while full vector scaling fits every class scale. Masking scale fitting by local observed support is a specific intermediate parameterization under label heterogeneity. Parameter sharing, partial calibration and sparse-support personalization are established neighboring ideas; a novelty search is required. If the result merely reproduces BCTS's capacity benefit, the proper conclusion is that a smaller known calibrator suffices.

**Minimum experiment and attribution controls.** Run the shared protocol for original full prior, support-constrained prior, BCTS with matched center/penalty convention, and a fixed cyclic shift of the support mask by one class with the same mask cardinality. Report the exact free-scale count for every client. A BCTS implementation must state that it uses the original prior center when serving as this attribution control; keep the existing published-convention BCTS baseline separately named. Verify fixed coordinates remain exactly `1.0` and the reduced projected gradient converges.

**Rival explanation.** Fewer degrees of freedom help regardless of label identity, or one global temperature is already enough. The equal-cardinality misaligned mask and matched-center BCTS distinguish those explanations.

**Reject or leave indeterminate.** Reject as a support-specific mechanism if task-mean NLL does not improve on both tasks, if the misaligned mask gains equally, or if matched-center BCTS matches/exceeds the proposed restriction without a material cost advantage. The last result can still improve the paper's parameter-efficiency story, but it would not establish this seed as a new core contribution.

**Budget.** Persistent predictor: 80 B; extra upload: 0 B. A packed 10-bit support mask needs 2 B if retained, but the prediction state already encodes fixed slots, so no extra persistent mask is required. Count vector: 80 B int64; implementation Boolean mask: 10 B, plus a ten-index vector of at most 80 B int64. Named extra fitting metadata is at most 170 B; reduced Hessian workspace can be smaller than the baseline but must be measured. No CNN change or forward pass is needed.

## Immediate next step and non-claims

Supply these role outputs to an independent verifier and novelty checker. The already assigned ratio-anchor development plan may implement Seed 2 after its plan is frozen; this document does not launch it. Seeds 1 and 3 remain unexecuted proposals. No candidate result is a reason to alter the protected 13 groups, original official TEST evidence, or the manuscript before complete scoring and review.

Procedural acknowledgement: the installed `hypothesis-generation` skill requests citation of Kassis et al. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, <https://doi.org/10.48550/arXiv.2609.00065>. This bibliographic information is supplied by the local skill; a live arXiv metadata fetch was unavailable in this child environment. It documents the procedural aid only and supplies no scientific evidence for these mechanisms. It has not been added to the protected manuscript bibliography.

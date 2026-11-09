# Research contribution and evidence review

Review snapshot: 2026-11-05 15:37 UTC. Reviewer: `/root/route_a_stats`.
Scope: the staged Route A manuscript and completed local evidence. This is an independent argument, mathematics and evidence review, not a venue prediction, a quality score, or a certification of perfection. The manuscript and protected results were not edited for this review.

## Judgment against the user's objective

The completed work now supports a coherent paper built around **a class-aligned local prior as a correctable regularization center for compact post-hoc prediction**. Its strongest result is a repeatable improvement in client-macro NLL over a same-capacity, same-solver zero-center control. The communication setting gives this predictor a useful operating point: 80 B of local prediction state on a shared model trained with 54,067 B uploads.

This meets the practical direction of “an idea + focused validation + strong paper presentation” more closely than a collection of engineering demonstrations. It does not establish a new compression algorithm, a universally superior personalization method, or robustness to changing label distributions. The remaining publication question is the significance and distinctness of this compact prior-centering choice relative to existing score adjustment and calibration methods. More completed runs improve the evidence for that choice; their count alone does not create another conceptual contribution.

The central idea is deliberately modest. Vector scaling, quadratic shrinkage and log-prior adjustment are established ingredients. The stationary identity is a useful explanation of their combination, but follows directly from first-order optimality; it should not be sold as a substantial new optimization theorem. The paper is strongest when its contribution is the informative center, the control that attributes its benefit to class alignment, and the demonstrated communication/prediction operating point.

## Completed evidence and what it establishes

| Evidence | Completed observation | Supported interpretation |
|---|---|---|
| Original official TEST, two tasks and three training seeds per task | Prior-minus-zero client-macro NLL is −0.03575086 on CIFAR-10 and −0.01823567 on FashionMNIST. All 60 client–seed pairs per task improve. | The informative center improves NLL in the protected evaluated setting. These are 60 conditional paired descriptions, not 60 independent training experiments. |
| Fixed-prior and score-family controls | Learned correction improves NLL over fixed prior addition by 0.0282/0.0222. Same-prior BCTS is within 0.0016/0.0036 of the vector predictor. | Correcting the prior is useful, and the useful information is not exclusive to twenty separate scale/bias values. This supports centering across compact score families rather than a necessity claim for class-specific scales. |
| Class-permuted center on development holdouts | Aligned versus cyclically permuted centers differ by 0.071145/0.043893 NLL; all observed paired cells favor alignment. Norm, scalar penalty, capacity and solver are held fixed. | Class assignment carries useful information under this controlled permutation. This is a direct attribution control for the center, not a proof that every possible rival explanation or prior construction has been eliminated. |
| Twelve new dense shared-training runs | Both dense FedAvg and FedProx were completed on both tasks and all three original seeds; all 480 TRAIN-only zero/prior fits were locked before TEST. | Prior centering also works on logits from ordinary CE and proximal shared training. It is not solely an interaction with the original FedLC-inspired loss or low-bit shared training. |
| Three-partition development comparison | All 18 task × partition × seed configurations have lower prior-centered macro NLL; improved client–seed cells are 177/180 CIFAR and 176/180 Fashion. | The direction is not confined to one partition under the same label-skew generator. The new rows are development results and do not turn the original official TEST into a three-partition TEST experiment. |
| Frozen-predictor label reallocation | The original prior advantage reverses as development labels are reassigned to different clients. | Prior stability matters. The completed diagnostic establishes sensitivity of stale predictors; it does not test successful updating of the center. |
| Count-geometry candidate | The first-seed two-task screen failed its continuation gates and was stopped. | There is no supported second contribution based on count uncertainty or trigamma geometry. Retain the negative screen as research evidence without promoting it into the method. |

### New FedAvg/FedProx comparisons

These are matched in-house baseline implementations. “Raw” is the underlying shared algorithm; “+ zero” and “+ prior” are our post-hoc additions. They must remain separate in tables and prose. They are not unmodified author-code reproductions.

| Task / shared training | Raw NLL | + zero NLL | + prior NLL | Prior minus zero NLL |
|---|---:|---:|---:|---:|
| CIFAR-10 / FedAvg | 1.6854 | 0.6794 | 0.6441 | −0.03533653 |
| CIFAR-10 / FedProx, μ=0.1 | 1.7086 | 0.6908 | 0.6551 | −0.03575757 |
| FashionMNIST / FedAvg | 0.7597 | 0.2054 | 0.1868 | −0.01862446 |
| FashionMNIST / FedProx, μ=0.1 | 0.7648 | 0.2078 | 0.1893 | −0.01848190 |

The large raw-to-fitted change is not the isolated contribution of the prior center: zero-centered fitting already produces most of that change. The last column is the matched contrast that identifies the center's additional value. It improves in every training seed and all 60 client–seed NLL comparisons for each task/algorithm combination.

Accuracy and calibration measures do not all improve. CIFAR mean macro accuracy changes by −0.04295/−0.07561 percentage points on FedAvg/FedProx; Fashion FedAvg q10 changes by −0.04726 points. CIFAR FedProx pooled ECE rises by 0.00062091 in raw ECE units, or 0.06209 percentage points. These completed adverse outcomes are consistent with a primary NLL claim and rule out wording such as “improves every quality measure.”

Execution acceptance covers 12 complete twenty-round runs, exact original memberships, full participation and local seed/batch counts. Saved-score replay covers 240 client exports, with zero metric discrepancy, identical original TEST assignments and unchanged frozen inputs. These checks establish the identity and arithmetic of the new evidence; they are not the paper's research contribution.

### Three-partition development comparison

The original partition is combined with two newly trained partitions using the same three model-seed identifiers, fixed λ=10, final shared checkpoints and TRAIN-only fitting. All configurations are retained.

| Task | Partition | Prior-minus-zero mean macro NLL | Improved client–seed cells |
|---|---:|---:|---:|
| CIFAR-10 | 20261217, original | −0.041112 | 60/60 |
| CIFAR-10 | 20261105 | −0.039062 | 57/60 |
| CIFAR-10 | 20261106 | −0.046313 | 60/60 |
| FashionMNIST | 20261219, original | −0.013788 | 60/60 |
| FashionMNIST | 20261105 | −0.010327 | 58/60 |
| FashionMNIST | 20261106 | −0.010856 | 58/60 |

The independent aggregation check verifies 18 configurations and 35 source identities, with per-client summaries agreeing within 1e−12. This is an aggregation check of saved outputs, not an independent model rerun. New partitions contain seven non-improving client–seed cells across the two tasks; the manuscript's original “all 60” statement must remain attached to its actual original-partition experiment. Across all development runs, macro accuracy is adverse in one CIFAR run and weighted accuracy in five CIFAR and one Fashion run. Do not present the NLL consistency as uniform accuracy improvement.

### Failed geometry screen

Count geometry minus the original prior gives −0.001191945 CIFAR NLL but +0.000619258 Fashion NLL; versus the precision-permuted control it gives −0.000938851 and +0.001037785. Fashion fails both predeclared continuation conditions. The simpler count ridge is also slightly better on CIFAR, and fixed isotropic λ=1 is better on Fashion.

The decision `STOP_CANDIDATE` is supported. Geometry and count ridge are close but not numerically equivalent at the fixed 1e−7 comparison tolerance. Independent replay of 240 arm/client rows has maximum metric discrepancy 1.54e−15, and all 253 locked inputs remain unchanged. No remaining-seed extension, official TEST evaluation or result-driven tuning followed the failed gate. This outcome does not invalidate the original isotropic center, but gives no basis for a stronger uncertainty mechanism claim.

## Mathematics of the current method

The objective in `sections/method.tex` is mathematically correct for frozen finite logits, λ>0 and the stated centered-bias constraint. Let θ=(A,b), let the per-example affine map from θ to logits be D_x, and let P_x=diag(p_x)−p_x p_xᵀ. Its Hessian is

    ∇²J(θ) = Σ_x D_xᵀ P_x D_x + λ I.

Each P_x is positive semidefinite. The λI term therefore makes the objective strongly convex on the centered-bias subspace, and coercivity gives a unique finite minimizer. No positive-scale restriction is assumed; unrestricted A remains an affine-logit convex fit and can change rankings.

For the Lagrangian multiplier ν associated with 1ᵀb=0, the bias stationarity equation is

    Σ_x p_c(x) − n_c + λ(b_c−μ_c) + ν = 0.

Summing over classes gives Cν=0: each example's probabilities sum to one, counts sum to |D|, and both b and μ are centered. Thus ν=0, yielding exactly the manuscript's identity

    λ(b_c−μ_c) = n_c − Σ_x p_c(x).

The right side uses **fitted** probability mass, not the unadjusted shared logits. The claim is an identity at the exact optimizer; finite-tolerance and FP32 deployment states satisfy it only approximately. The manuscript currently makes the exact-optimizer qualification.

As λ→∞, compare the optimum with θ₀=(1,μ). Since cross-entropy is nonnegative, λ‖θλ−θ₀‖²/2 ≤ CE(θ₀), so θλ→θ₀. Centered log priors differ from log priors only by a common offset, hence the endpoint is fixed prior addition after softmax. As λ→0 the penalty disappears from the objective. This does not guarantee a finite limiting minimizer: with an absent TRAIN class, its bias can decrease indefinitely, recentering all biases without affecting relative observed-class logits, while strictly decreasing loss. The current wording distinguishes the objective endpoint from a nonexistent finite optimum.

Dividing the sum-loss objective by |D| gives penalty coefficient λ/|D| in mean-loss units. The manuscript's client-size interpretation is correct as a statement about that coefficient; it is not a theorem that every smaller client has a larger realized bias displacement or predictive gain.

For C=10 the stored predictor is two ten-vectors in FP32, exactly 80 B, despite one redundant bias degree being removed by centering. This is persistent prediction state. The TRAIN logit cache, FP64 Hessian/caches, shared model, error-feedback residual and dense downlinks are additional costs already distinguished by the paper. No mathematics correction to the current method is required by this review.

## Work still in progress at this snapshot

| Item | Verified current state | What may be concluded now |
|---|---|---|
| pFedMe | The six-run manifest is `RUNNING`; the three CIFAR training commands have exited 0 and Fashion training is underway. Final model-lock and official-TEST comparison are not yet available in the reviewed manifest. | There is no completed comparative outcome to cite. Source reuse is the author's unmodified optimizer with a matched protocol adapter, not a full unmodified author-code reproduction. |
| CIFAR-100 | Isolated plan, adapter, source checks and numerical fixtures pass. The reviewed source preflight says `actual_data_ready=false`; the reviewed fixtures use invented logits/counts and contain no CNN task run. | There is no completed CIFAR-100 accuracy or NLL result. Synthetic K=100 curvature, gradient and storage checks establish solver readiness only. |

These are state snapshots, not resource-impossibility claims. They may be updated after the root finishes their frozen execution and verifies their outputs. Neither is counted as supporting evidence above. No additional experiment is requested by this review.

## Fit with CCF-style writing and remaining editorial decisions

The current introduction, abstract, method and discussion have an identifiable problem → method → mechanism control → measured operating point. They focus on the prior center and NLL rather than a reproduction checklist. The related-work distinctions between fixed logit adjustment, post-hoc score scaling and learned model-training centers make the intended contribution reviewable. The writing can support a CCF-targeted submission when the integrated final manuscript keeps this precise scope. Acceptance or exceptional novelty cannot be inferred from these completed local results.

The strongest defensible paper sentence is: **Under the evaluated stationary label-skewed client distributions, a local class prior is a useful regularization center for a compact post-hoc predictor; matched controls and new shared-training/partition evidence support an NLL benefit without requiring another model upload.**

The main remaining editorial decisions are concrete:

1. Keep NLL as the central endpoint. Accuracy, tail-client performance and ECE trade-offs remain visible; “calibration improves” should identify its metric rather than imply every calibration measure improves.
2. Make the new shared-training comparison explicitly prior versus zero-centered fitting. Raw FedAvg/FedProx and our fitted variants remain distinct.
3. Integrate completed three-partition development evidence under that split label. Preserve every original TEST number and keep “every client–seed pair” attached to the experiment where it is true.
4. Avoid presenting extremely small dependent-pair Wilcoxon p values as significance evidence. Direction counts, paired effects, seed results and clearly conditional client-resampling intervals convey the completed finding more accurately. Also keep ECE units consistent.
5. Sell 80 B as added persistent prediction state and 54,067 B as the stated compressed upload. Compression ingredients and first-order stationarity are not independent inventions; baseline runtime and source-integrity records belong in supporting material.

The three priorities now align: there is a precise compact-prediction idea, focused completed controls test the claim, and the manuscript can explain those results as one argument. The unresolved judgment is the research value of this specific center relative to close prior work and completed personalization competitors. It should be resolved by accurate comparison and concise positioning of the existing results, not by calling the paper perfect or turning unfinished extensions into evidence.

## Sources actually inspected

- `sections/{front,abstract,method,evaluation,results,appendix,discussion,conclusion}.tex` and `tables/{main,main_full,quality,seeds,training,reference,dense_baselines}.tex` in the staged manuscript.
- `stats/result.json` for protected TEST effects, paired descriptions and inference boundaries.
- `baselines/{summary,paired_outcomes,TRAINING_ACCEPTANCE,VERIFICATION}.json` and the completed baseline outputs reviewed during this task.
- `multipartition/{combined_development.md,AGGREGATION_VERIFICATION.json,MATRIX_STATUS.json}`.
- `scientisttwo_screen/{CONCLUSION.md,SUBSET_RESULT.json,VERIFICATION.json}` and `06_ScientistTwo/runs/fl_heterogeneity_20261105/candidate_review/POST_SCREEN_CRITIC.md`.
- `personalized_baseline/{PLAN.md,matrix_execution/run_20261105T151515412446Z.json}`.
- `cifar100_extension/{PLAN.md,FIXTURES.json,SOURCE_PREFLIGHT.json}`.

Process acknowledgement: this review applied the local `scientific-critical-thinking` skill to separate observations, attribution and scope. Its supplied software-reference metadata identifies Kassis et al. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, https://doi.org/10.48550/arXiv.2609.00065; that bibliographic record was not independently verified here and is not used as scientific evidence for PR-LFL. No manuscript bibliography entry was added.

## Completion addendum — 2026-11-05 16:00 UTC

This addendum supersedes the earlier in-progress pFedMe and source-only CIFAR-100 rows above; earlier observations are retained as timestamped history.

**pFedMe is complete.** All six fixed runs completed twenty rounds, the 43-input model lock preceded official-TEST scoring, and both task aggregates plus all six raw-personalized-theta score reports are DONE. This reviewer independently re-hashed all 43 locked inputs and reconstructed the aggregate mean/sample SD from the six seed reports: maximum aggregate discrepancy is zero. The separate exported-logit scorer replay passes 1,060 checks with maximum discrepancy 1.47793e−12, without CNN/raw TEST access.

| Comparator | CIFAR-10 macro NLL | CIFAR-10 macro accuracy (%) | FashionMNIST macro NLL | FashionMNIST macro accuracy (%) |
|---|---:|---:|---:|---:|
| pFedMe, author optimizer with matched protocol adapter; raw final theta | 1.0072 ± 0.0291 | 64.38 ± 0.69 | 0.3557 ± 0.0359 | 88.86 ± 1.33 |

The protected PR-LFL macro NLL/accuracy is favorable against this configured training-time personalization comparator. This makes the empirical comparison more complete; it does not make generic prior-centered regularization novel or establish state of the art. The adapter uses the unmodified author optimizer but changes outer lr from the author's .005 default to matched .03, fixes K=5 inner gradients, uses dense 547,560 B messages and evaluates uncalibrated final personalized theta. Those choices are declared, frozen and computationally different from the compact post-hoc fit. The conclusion is specific to the tested protocol; it is not a full author-code reproduction or a claim about tuned best pFedMe performance. Six training logs record 2,285.75 seconds and exactly five CE gradients per outer minibatch step.

All ten per-seed/aggregate metrics, scope and suggested paper row are now in `personalized_baseline/{compact_summary.json,result_all_metrics.csv,RESULT_SUMMARY.md,table_row_suggestion.tex}`. `result_pfedme.json` and `BRIDGE_IMPORT_SUGGESTION.md` prepare final-scope evidence for root's actual sequential bridge import; this reviewer has not performed that import.

**CIFAR-100 awaits final matrix completion and verification.** Actual official TRAIN data passed `PASS_DATA_READY`. The live matrix snapshot inspected at 15:59 UTC records seed20261231 and seed20261232 exiting 0 after twenty rounds, with the third fixed seed in progress. The first saved development evaluation is DONE, but no three-seed aggregate or independent final verification is yet used in this review. Data/solver readiness and a partial run are not substituted for the final three-seed result. All observed records retain zero official TEST reads. Do not count an unfinished extension as a positive or negative research outcome.

The primary scientific judgment remains: **prior-centering has a consistent NLL benefit in the completed matched comparisons, with nonuniform accuracy and ECE outcomes.** On dense CIFAR FedAvg/FedProx, prior-minus-zero macro accuracy is adverse by 0.04295/0.07561 percentage points; Fashion FedAvg q10 is adverse by 0.04726 points; CIFAR FedProx pooled ECE increases by 0.00062091 raw units (0.06209 percentage points). The original weighted CIFAR accuracy, Fashion q10 and same-prior BCTS client-ECE comparisons also remain adverse to any “all measures improve” claim. The paper should use the favorable NLL result confidently while keeping these metric-specific trade-offs. The failed geometry candidate remains stopped and supplies no additional core contribution.

## Final CIFAR-100 evidence addendum — 2026-11-05 16:02 UTC

**CIFAR-100 development is complete and independently verified**, superseding the preceding partial-matrix snapshots. All three frozen seeds on partition20261107 completed twenty rounds. `INDEPENDENT_RESULTS_VERIFICATION.json` reports `PASS_INDEPENDENT_SAVED_ARRAY_REPLAY`: three runs, sixty clients and 240 arm/client rows, maximum metric discrepancy3.33067e−16, with source/data/checkpoint identities checked and no new CNN, fit or official TEST read during replay. This reviewer inspected that report together with `formatted_development.json`, `DEVELOPMENT_RESULTS.md` and the completed matrix manifest; the replay itself was performed by the separate p1 agent.

The added dataset supports the **NLL-specific** conclusion: zero→prior client-macro NLL falls **2.4283→2.2896**, with a mean difference−0.138731 and improvement in all three runs and all60client–seed cells. It also supplies an adverse result that must remain visible: macro accuracy falls **35.2671→35.1211%**, or−0.1460percentage points, while macro ECE15 increases **0.04976695→0.05058740**, or+0.00082045raw units (+0.08205percentage points). Macro Brier and pooled ECE improve, so “calibration improves” without naming its measure would be misleading. These results strengthen the evidence that the center can reduce NLL beyond two ten-class tasks; they do not establish uniform accuracy/calibration gains, robustness to drift or another conceptual mechanism.

This is one fixed label-skew partition with three model seeds, TRAIN-only fitting and development holdout scoring; it does not replace or alter the protected original official TEST. The 100-class model stores a 200-coordinate **800 B** predictor and sends **56,376 B** shared updates. The p1 acceptance prose and table caption agree with `formatted_development.json`'s `upload_bytes=56376`. These dimensions and bytes are specific to the 100-class extension and must not replace the original 80 B/54,067 B claims.

## Final source-number inspection — 2026-11-05 16:06 UTC

Read-only inspection of the current staged manuscript finds **no new pFedMe/CIFAR-100 numerical or protocol-scope error**. This checks saved results against the actual reachable sources, not just proposed table snippets. Final compilation, page layout and artifact delivery remain root's responsibility; the check did not edit the manuscript or status file.

| Current source | Inspected statement | Finding |
|---|---|---|
| `sections/results.tex:25` | Nine development configurations per original task improve; CIFAR-100 NLL2.4283→2.2896 and800B local state; Table`tab:partitiondev` | Correct rounded result and development context. The named table is the lower panel of the actual included `tables/mechanism.tex`, with zero/prior means and sample SDs2.4283±0.0116/2.2896±0.0114. |
| `sections/results.tex:37` | Dense FedAvg/FedProx compare prior versus zero-centered fits; matched pFedMe NLL1.0072/0.3557 with five inner gradients per batch and dense communication | Correct comparison identity and frozen pFedMe means. The sentence refers to Appendix`app:complete`, which includes the actual table row in `tables/training.tex:18`. |
| `tables/training.tex:18` | Raw-theta pFedMe accuracy64.38±0.69/88.86±1.33%; NLL1.0072±0.029/0.3557±0.036 | Every mean and displayed sample SD matches the six completed score reports after stated rounding. It is explicitly raw personalized theta, not an extra calibrated pFedMe variant. |
| `sections/appendix.tex:12` | Author optimizer with data/model adapter; K5,lambda15,beta1,inner rate.09,mu.001; full local epochs and outer rate.03 replace author's iterations/default.005; dense547,560B | Accurately discloses actual reuse and algorithm-affecting protocol adaptation, including both local-schedule and outer-rate differences. This does not claim full author-code reproduction, matched computation, tuned best performance or SOTA. |
| `sections/appendix.tex:14` | Original tasks'177/180 and176/180 development pairs; C100 partition20261107,37,500/12,500fit/development rows,C100,lambda10,142,724CNN parameters,56,376B updates,dense570,960B,800B predictor | All match completed records. “Same three seeds” reuses original CIFAR model-seed identifiers and is not a reuse of trained models. Every added-dataset score is explicitly development. |
| `sections/appendix.tex:14` |60/60 C100 NLL pairs improve; accuracy drops.1460pp, client ECE rises.000820raw,Brier.78671→.78520 | The favorable NLL and unfavorable accuracy/client-macro ECE are both retained. ECE units are explicit; rounded values agree with the independently verified output. The Discussion also identifies C100 accuracy/client ECE as favoring zero centering. |

A recursive static walk from `main.tex` finds14reachable TeX sources. Each inspected target occurs exactly once in the included source graph: `tab:partitiondev` in`tables/mechanism.tex`, `tab:dense_baselines` in`tables/training.tex`, and`app:complete` in`sections/appendix.tex`. Unused earlier standalone table files are not counted as duplicate labels in the compiled graph. `dinh2020pfedme` exists in the bibliography. This establishes source-reference identity only; no compiled-PDF cross-reference success is inferred without root's final build.

Source hashes at inspection: `sections/results.tex`=`a04de020d532b10b4fccec46f2f170e08833da179d1b7e5d1799916bf8047264`; `sections/appendix.tex`=`77f258817c0335f5b237f3ce82ccdb54756e44902762e2a6f1369d2ad7ae48b2`; `tables/training.tex`=`ed99e29035f4e3f9e7ce5b7746066f47fa4a9e9ea8d04f9c3076884565163dec`; `tables/mechanism.tex`=`45fb821bed318581465195644e619cd0ae18cf84d9071e24c697048234a335c4`. These identify the checked snapshot and do not assert that later root edits were inspected.

# Route A revision status

Status: **COMPLETE_SELECTED_FL_REVISION — Route A manuscript, selected P0/P1/P2 FL evidence, independent checks and delivery are complete. No active experiment remains. This does not certify perfect novelty, arbitrary heterogeneity or conference acceptance.**

This file records the revision of `论文修改任务书_PR-LFL.docx`, Sections 0–11, on Route A (post-hoc calibration). It does not certify that every research objective is complete, that the manuscript is perfect, or that a CCF venue will accept it. The narrower supported contribution is a class-aligned prior used as a correctable regularization center for compact local prediction, with compressed shared training as its communication setting. The upload implementation is supporting material in the appendix.

## Scope, backups and evidence rules

- Original manuscript: `/Users/wind/Desktop/Codex/paper/03_英文论文/`.
- Before-revision backup: `/Users/wind/Desktop/Codex/paper/01_论文材料/历史论文版本/route_A_before_20261104T235231+0800/03_英文论文/`.
- Working manuscript at drafting: `/private/tmp/pr_lfl_route_a/03_英文论文/`.
- Original lock manifest: `/private/tmp/pr_lfl_route_a/original_lock.json`.
- New analyses and experiments are archived under `revision_artifacts/route_a_20261105/{stats,p1,baselines,multipartition,personalized_baseline,cifar100_extension,scientisttwo_screen}/`; the original `evidence/`, existing `checks/` files and pre-existing figure provenance are not the destination for new results.
- This is the only newly authorized file in `checks/`. Existing check reports describe earlier revisions and are not silently relabelled as checks of this manuscript.
- Updated taskbook SHA256: `6ba87f18803a47be333162220fed47bacaff159d1f6e849e4f4586090b79f407`.
- Byte-level checks of the class/BST, main preamble, original bibliography prefix and17 original evidence/check/provenance/template files: PASS. See the new revision artifact checks, which do not overwrite historical checks.

## Completed evidence work

### P0: existing official-TEST metrics and paired summaries — COMPLETE

Source: unchanged `evidence/individual_seed_metrics.csv`, the six existing official-TEST per-seed result JSON files pinned by `revision_artifacts/route_a_20261105/stats/result.json`, and `stats/analyze_p0.py` in the same archive. This analysis performed no training, fitting or new TEST inference.

| Task | Endpoint | Zero center | Prior center |
|---|---|---:|---:|
| CIFAR-10 | Client-macro NLL | 0.6803677623 | 0.6446169008 |
| CIFAR-10 | Sample-weighted NLL | 0.6787784958 | 0.6570664934 |
| CIFAR-10 | Sample-weighted / global accuracy | 0.7475333333 | 0.7471333333 |
| CIFAR-10 | Client-accuracy q10 | 0.6027078966 | 0.6065036443 |
| FashionMNIST | Client-macro NLL | 0.2062309628 | 0.1879952973 |
| FashionMNIST | Sample-weighted NLL | 0.1588086046 | 0.1486097222 |
| FashionMNIST | Sample-weighted / global accuracy | 0.9461666667 | 0.9472666667 |
| FashionMNIST | Client-accuracy q10 | 0.8577691759 | 0.8566495009 |

Every task has 60/60 negative prior-minus-zero NLL differences and 20/20 negative client means. Median paired differences are −0.02570773 / −0.01185078. Nominal pooled Wilcoxon W=0, two-sided p=1.7347235e−18; these dependent client–seed observations are a descriptive consistency calculation, not sixty independent training runs. Across twenty client means the nominal p is 1.907e−6. Client-cluster bootstrap uses 1,000 draws and PCG64 seed 20261104, retaining all three seeds for a selected client: intervals are [−0.05248377,−0.02328366] / [−0.02734456,−0.01089916]. They condition on the observed partition and trained models and omit new-partition/training-run uncertainty.

The main quality table reports weighted metrics and q10. Unfavorable weighted CIFAR accuracy, Fashion q10, BCTS client-ECE comparisons, and the negative CIFAR seed accuracy difference remain in the manuscript/table material. Macro and sample weighting use different client weights; scarcity alone is not asserted as an experimentally identified cause of their difference.

### P1: fixed-grid penalty sensitivity — COMPLETE, with a nonattained zero endpoint

Archived plan/script/result: `revision_artifacts/route_a_20261105/p1/PLAN.md`, `lambda_sensitivity.py`, `lambda_result.json` and `plot_data.json`.

- Exactly two tasks × three existing seeds × twenty clients; 480 positive-penalty fits at λ∈{0.1,1,10,100}; λ=∞ is fixed prior addition.
- Saved development TRAIN logits fit the predictors; disjoint development holdouts score them. No CNN calls, new training seeds or official TEST reads.
- The λ=10 FP32 states reproduce the original states bitwise, with zero metric discrepancy; λ=10 remains the primary coefficient and is not newly selected or claimed optimal.
- All 120 client–seed TRAIN sets lack classes. At λ=0, lowering an absent-class bias (with common-shift compensation for centering) strictly decreases finite softmax TRAIN cross-entropy. Therefore there is no finite minimizer to evaluate. The figure labels this endpoint and plots no fabricated λ=0 NLL.
- Result metadata records 250 source/input hashes unchanged and no forbidden file-open attempts.

| Holdout client-macro NLL | λ=0.1 | λ=1 | λ=10 | λ=100 | λ=∞ |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 0.610296 | 0.610944 | 0.612447 | 0.619347 | 0.642073 |
| FashionMNIST | 0.165633 | 0.164888 | 0.166384 | 0.172402 | 0.197719 |

### P1: frozen-center label-reallocation diagnostic — COMPLETE at the development diagnostic scope

Archived plan/script/result: `revision_artifacts/route_a_20261105/p1/DRIFT_PLAN.md`, `prior_drift.py`, `drift_result.json` . `drift_preliminary_result.json` is not the cited result and is not the final evidential source.

This tests stale predictors, not successful adaptation to a changed prior. It reuses all original development holdout examples exactly once per scenario, freezes original destination-client prior/zero states and shared logits, and applies label-only classwise movement at ρ∈{0,0.25,0.5,0.75,1}. The fixed target is Dirichlet(0.1) over twenty clients; PCG64 seed 20261105 supplies independent task streams and fixed movement orders. Largest-remainder quotas assign moved examples; all three model seeds share each scenario. No target redraw was needed. At ρ=0 the original membership and metrics are recovered (maximum discrepancy 2.22e−16).

| Prior-minus-zero holdout macro NLL | ρ=0 | ρ=0.25 | ρ=0.5 | ρ=0.75 | ρ=1 |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | −0.041112 | +0.336270 | +0.604060 | +0.857850 | +1.239329 |
| FashionMNIST | −0.013788 | +0.171311 | +0.317382 | +0.450069 | +0.633471 |

The adverse result is reported, not removed. It shows that stale prior-centered predictors lose their NLL advantage under this controlled reallocation. It does not test a re-estimated center, feature shift, multiple drift realizations, or production deployment. It uses development holdouts instead of reopening official TEST for method development; the manuscript labels that split. Metadata records 248 source/input hashes unchanged, zero CNN calls/fits/TEST reads and no forbidden opens.

## P1/P2 execution status

Final execution record: **2026-11-06 (Asia/Shanghai)**. All selected FL rows below are complete. Original TEST, added development evidence and stopped exploratory candidates retain their own scopes.

| Requested item | Current status | Evidence / remaining dependency |
|---|---|---|
| Three partitions × three seeds per task | COMPLETE — development, 18 configurations | `multipartition/MATRIX_STATUS.json` records all twelve new twenty-round training runs exiting 0. `combined_development.{json,csv,md}` combines their holdout scores with six original-partition development runs; `AGGREGATION_VERIFICATION.json` independently checks 18 configurations, 35 source identities and per-client summary agreement within 1e−12. Original partition plus 20261105/20261106, fixed λ=10 and all three original model-seed identifiers are retained. All 9/9 run-level macro NLL differences improve per task; improved client–seed cells are 177/180 CIFAR and 176/180 Fashion. Seven non-improving cells remain. These are development results, not three-partition official TEST results or independent client replicates. |
| FedAvg matched baseline | COMPLETE — six task/seed runs, frozen TEST scoring and replay | `baselines/{PLAN.md,SOURCE_LOCK.json,TRAINING_ACCEPTANCE.json,FIT_LOCK.json,summary.json,paired_outcomes.json,VERIFICATION.json}`. Dense ordinary-CE FedAvg matches the original splits, backbone, seeds, twenty rounds, local epoch, batch and scorer. Raw algorithm predictions and our zero/prior post-hoc additions are separately labelled; this is an in-house matched implementation, not an author-code reproduction. Prior-minus-zero macro NLL is −0.03533653 CIFAR / −0.01862446 Fashion. |
| FedProx matched baseline | COMPLETE — six task/seed runs, frozen TEST scoring and replay | Same `baselines/` artifacts, with fixed μ=0.1 dense FedProx. The local gradient adds μ(w−w_global) to ordinary CE. Prior-minus-zero macro NLL is −0.03575757 CIFAR / −0.01848190 Fashion. Across both algorithms all 12 runs completed; all 480 TRAIN-only fits were locked before TEST. Saved-score replay checks 240 client exports with zero metric discrepancy, exact original TEST assignments and unchanged frozen inputs. CIFAR macro-accuracy losses, Fashion FedAvg q10 loss and CIFAR FedProx pooled-ECE increase remain disclosed. |
| Ditto or pFedMe | COMPLETE — pFedMe six twenty-round runs, final raw-theta TEST scoring and replay | All six fixed runs are DONE20; `personalized_baseline/MODEL_LOCK.json` freezes 43 inputs before TEST, independently re-hashed unchanged. `official_test/{cifar,fashion}/result.json` and all six seed reports are DONE with no TEST fit/backward; raw final personalized theta is the prediction rule. `INDEPENDENT_SCORE_VERIFICATION.json` PASS covers 1,060 checks with maximum discrepancy 1.47793e−12. `compact_summary.json` independently reconstructs task means/sample SDs with zero discrepancy; `result_all_metrics.csv`, `RESULT_SUMMARY.md` and `table_row_suggestion.tex` preserve all metrics and the precise paper-row scope. CIFAR macro NLL/accuracy is 1.007232±0.029098 /64.3825±0.6907%; Fashion is 0.355710±0.035933 /88.8609±1.3304%. The unmodified author optimizer at commit `caf24b1f954d4381bd9b4d104ee7eff389e1489b` uses a matched task adapter, not a full author-code reproduction. Author outer-lr default .005 was explicitly adapted to .03 before training; K=5 gives five CE gradient evaluations per minibatch, with dense 547,560 B uploads/downlinks. These results do not establish state of the art or tuned best pFedMe performance. |
| CIFAR-100 or FEMNIST | CIFAR-100 COMPLETE — three fixed seeds, development scoring and independent replay | `cifar100_extension/MATRIX_STATUS.json` records all three seeds20261231/20261232/20261233 exiting 0 after twenty rounds on partition20261107. Actual official TRAIN data passed MD5/SHA256 and exact disjoint 37,500/12,500 TRAIN/development coverage. `formatted_development.json` is `COMPLETE_VERIFIED_DEVELOPMENT`; `DEVELOPMENT_RESULTS.md` and `table_development_compact.tex` retain every seed. Zero→prior macro NLL is 2.428310→2.289578 (mean difference −0.138731), improving in 3/3 runs and 60/60 client–seed cells. Macro accuracy instead decreases 35.2671→35.1211% (−0.1460 pp), and macro ECE15 increases .04976695→.05058740 (+.00082045 raw, +.08205 pp). `INDEPENDENT_RESULTS_VERIFICATION.json` PASS replays three runs,60clients,240arm/client rows with maximum metric error3.33067e−16; source/data/checkpoint identities and fitted-state diagnostics are retained. This is one-partition development evidence, not an added official TEST result or SOTA claim. The isolated 100-class adapter uses142,724CNN parameters, an800B predictor, **56,376 B** uniform3 updates and a structured200-dimensional solver. All records retain zero official TEST reads. |

The actual host-MPS logs and execution manifests supersede earlier feasibility-only notes. FedAvg/FedProx, new-partition, pFedMe and CIFAR-100 runs all completed. The 12 dense baseline runs record about 1,206.33 training seconds in total; this is measured run cost, not a performance guarantee for pFedMe or CIFAR-100. pFedMe's five inner steps require extra gradient computation and its dense communication is accounted separately.

### ScientistTwo research screen — completed negative decision

`scientisttwo_screen/{PLAN.md,SOURCE_LOCK.json,FIXTURES.json,SUBSET_RESULT.json,CONCLUSION.md,VERIFICATION.json,NATIVE_RECEIPT.json}` records the predeclared first-seed, two-task count-geometry screen. Fashion fails both continuation conditions: geometry-minus-original-prior NLL is +0.000619258 and geometry-minus-precision-permuted NLL is +0.001037785. The decision is **STOP_CANDIDATE**; no remaining-seed extension, TEST evaluation or result-driven retuning follows. All six arms and adverse outcomes remain. Independent saved-state replay covers 240 arm/client rows with maximum metric discrepancy 1.54e−15, and all 253 locked inputs are unchanged. Count-ridge is close but not numerically equivalent at the 1e−7 contrast tolerance. The residual-ratio candidate was reviewed and never executed. Neither candidate is promoted to a new paper contribution.

The real installed package is the **unofficial Open-ScientistTwo 0.4.0**, commit `68ef7c9948b3da091724b5759f8a43888286ab9d`, in `06_ScientistTwo/upstream/Open-ScientistTwo/` with isolated Python 3.12.16 and frozen uv dependencies. The bounded official-source check did not locate an official released framework or complete prompts. The upstream live campaign remains disabled: Docker/Podman is absent, the container GPU path expects CUDA rather than host MPS, and the separate Codex CLI reports `Not logged in`. This is not a reproduction of ScientistTwo's original benchmark results or a claim of an official Google framework installation.

The local native-agent/host-MPS bridge records actual role outputs and existing experiment bytes; it does not generate synthetic answers or claim an upstream live campaign. Active campaign: `06_ScientistTwo/bridge/campaigns/fl_heterogeneity_20261105-v1`. The negative screen was really imported as `geometry-subset-stop-20261105`, with experiment receipt `artifacts/4253154642457d62edc0728c57d5623dfa5438cfe27110812bd5186fa4e90c7b.json`. The final bridge check receipt `artifacts/0a4dee7b5f8155ac85c6d78b48089a383b2ae194c1d2a6c9923966bed710f555.json` reports PASS, 45 artifacts, two roles, five experiments and no errors. It includes the completed dense baselines and pFedMe as final evidence, multipartition/CIFAR-100 and the stopped screen as development evidence. This verifies bridge/source/artifact integrity, not scientific acceptance, external cross-family review or live upstream completion. See `06_ScientistTwo/{README.md,sources/INSTALLATION_STATUS.json,bridge/README.md}`. The installation status and final research assessment now reflect these completed records.

### New near-neighbor sources — primary-source relation checked

- **PRIOR/pFedBreD:** official NeurIPS 2023 identity, DOI `10.52202/075280-1234`, volume 36, pages 28378–28392, is verified in `literature/near_neighbors/{PRIOR_CITATION_VERIFICATION.md,PRIOR_CITATION_SOURCE_RECORD.json}`. The inspected arXiv v2 full text Eq.8/11/14 and Algorithm 1 establish existing personalized Bregman/model-space regularization centers. The paper's distinction is the observed local log-class-frequency center in a frozen convex output-space fit; generic prior-centered regularization is not claimed new. The published author spelling **Jiancheng Lv** is used; the arXiv v2 spelling **Jiangcheng Lv** remains recorded as a version discrepancy.
- **LaSCal:** the initially unresolved method search is superseded by actual official NeurIPS 2024 HTML/PDF inspection, DOI `10.52202/079017-2088`, volume 37, pages 65386–65414. `literature/near_neighbors/{LASCAL_METHOD_OVERLAP.md,LASCAL_SOURCE_RECORD.json,lascal_official.pdf,lascal_official.html}` records method checks on PDF pp.4–7 and 16–17. LaSCal estimates target classwise calibration error from labeled source and unlabeled target data and fits a temperature; its estimator is extensible. PR-LFL instead uses labeled client TRAIN data for an observed-prior-centered affine fit. No universal non-overlap claim or inherited label-shift consistency guarantee is made.

- **FedNPC:** complete official CVF PDF inspection (pp.3–5, Eq.2/3 and Algorithm1), archived in `literature/near_neighbors/{FEDNPC_METHOD_OVERLAP.md,FEDNPC_SOURCE_RECORD.json,fednpc_cvf_full.pdf}`. It freezes the global feature extractor and retrains a server classifier on isotropic synthetic features and uniform labels; PR-LFL fits client TRAIN logits with an observed-prior-centered penalty. Shared post-hoc freezing/no-extra-client-communication properties are prior art, not standalone novelty. Formal citation uses the actually inspected CVPR2026 **Findings** CVF edition, pages7737–7746; conflicting Crossref pages7740–7749/candidate DOI remain recorded separately. No FedNPC performance claim is made.

The generic earlier `METHOD_COMPARISON.md` and `comparison_sources.json` retain historical retrieval state; later source-specific records above control current status. All three verified new entries were appended without changing the original bibliography prefix. Final build:33 cited keys,34 bibliography entries. All new entries and near-neighbor descriptions were independently checked.

## Minimal evidence-required corrections to supplied prose and figures

The updated taskbook allows evidence-required corrections while preserving its argument structure. Each change below corrects a mathematical, empirical or accounting scope.

| Location / supplied wording | Implemented minimum correction | Reason and source |
|---|---|---|
| Abstract: shared model “ranks classes poorly for each of them” | “can rank classes poorly for individual clients” | Aggregate client-macro results do not prove degradation on every client; source: main/individual-seed results. |
| Abstract: stationary identity “keeps the prior where the shared logits already agree …” | Agreement refers to **fitted probability mass** matching observed class counts. | Equation λ(b−μ)=n−Σp contains fitted p=softmax(Az+b), not raw shared logits; method equation and KKT derivation. |
| Abstract and second contribution: permutation “erases the gain on all 60 …” | All sixty comparisons explicitly use the **aligned center** as comparator; task-mean NLL also exceeds zero-center NLL. | `05_核心方法研发/results/core_contribution_probe/result.json`: permuted is worse than aligned in 60/60 pairs per task; worse than zero in only 47/60 CIFAR and 58/60 Fashion pairs. |
| Abstract: three-bit updates “… add 80 B of local state” | Attribute **80 B to the local predictor**, separately from 54,067 B uploads. | Twenty FP32 predictor values are 80 B; training also retains a 547,496 B EF residual. No device-wide memory saving is inferred. |
| First contribution / method / conclusion: λ→0 “recovers unconstrained vector scaling” and estimator endpoint wording | Two **baseline objectives** are penalty endpoints; λ→0 removes the penalty, without promising an attained finite estimator. | Every observed TRAIN set has missing classes; analytical nonattainment plus sensitivity plan/result. |
| Third contribution: “three-bit logits” | “logits from three-bit shared training” | Quantization is applied to communicated updates; the evaluated logits are not three-bit values. |
| Third contribution: “within 0.03 accuracy points of dense training” | “within 0.03 accuracy points of the **reported dense-training means**” | Protected training table rounds CIFAR 75.59 vs 75.57 to 0.02 pp and Fashion 93.00 vs 92.97 to 0.03 pp. The bound applies to displayed means, not unrounded seedwise differences. |
| Figure 1 specified three bars but required 38.96 vs 63.06 contrast | Use **four** bars: Shared, Prior-only, Zero center, Prior center. | 63.06% is prior-only, not zero-centered fitting. The actual four CIFAR client-macro means are 38.96/63.06/75.39/75.57; `motivation_route_a.provenance.json` traces CSV and real client-0 TRAIN counts. |
| Figure 1 illustrative “90% cats” | Actual client-0 histogram; prose says “dominated by one class” without asserting 90% for this split. | Client 0 counts are [1,0,193,0,98,151,0,1,0,6], total 450; left panel is one TRAIN client, right panel is aggregate TEST, explicitly labelled. |
| λ figure requested a plotted 0 endpoint | Label λ=0 “No finite minimum”; show no numeric point/connecting line there. | A numerical zero-penalty fitted endpoint would fabricate a nonexistent finite optimum. The ∞ fixed-prior endpoint remains separately plotted. |
| Conclusion / Discussion drift wording | Re-estimating the center remains a next design step; the **adverse frozen-center diagnostic** is disclosed in Discussion and appendix. | A completed stale-predictor diagnostic does not establish adaptation by an updated-prior method. |

Additional minimum corrections after independent review: the Linear control is explicitly called **one-epoch**; the drift caption identifies the **whole frozen prior-centered predictor**, rather than attributing its failure to the prior alone. The abstract opener omits the redundant word “federated” to avoid a line overflow without changing scope; a discretionary hyphen in “parameter” changes no printed wording. The introduction retains the required four-paragraph argument and opening roles while removing repetition for the12-page limit. Its logit-adjustment comparison now names the fixed unit-strength control: Menon already permits a tunable strength, so generic logit adjustment is not presented as lacking one. The two Wilcoxon calculations explicitly identify sixty pairs versus twenty means, their independence reference and the prohibition on population-significance interpretation. Compact-control forms, C=10 storage scope, accuracy units and appendix navigation were clarified without changing numbers. New publisher URLs are recorded in the citation ledger rather than repeated in the printed new bibliography entries; old entries remain byte-identical. The new pFedMe entry normalizes the first author as `Dinh, Canh T.` from the archived publisher name; the publisher/arXiv third-author discrepancy is disclosed in its source record.

## Protected Section 1.3 numeric groups: source-to-manuscript map

`CSV` below means the unchanged `evidence/individual_seed_metrics.csv` (task × seed × variant; three-seed arithmetic means). Original evidence fields remain unchanged. Existing tables may change layout or location; their protected values do not.

| # | Protected values | Source / exact derivation | Retained location |
|---:|---|---|---|
| 1 | 0.6446 / 0.1880 | CSV `prioranchored_vector.macro_nll` means 0.644616900848 / 0.187995297344 | `sections/results.tex` four-decimal values; `tables/main.tex` rounded display 0.645 / 0.188 |
| 2 | 0.0358 / 0.0182 | CSV paired zero-minus-prior mean NLL = 0.035750861411 / 0.018235665483 | `sections/results.tex` |
| 3 | 75.57% / 92.97% | 100 × CSV prior-vector macro-accuracy means = 75.56805534 / 92.97399369 | `sections/results.tex`, `tables/main.tex`, related controls |
| 4 | +0.1777 / +0.1730 pp | 100 × CSV paired prior-minus-zero macro-accuracy means = 0.1777086253 / 0.1729994971 | `sections/results.tex`; full seed table |
| 5 | 0.0282 / 0.0222 | CSV analytic-prior minus prior-vector mean NLL = 0.028152092484 / 0.022225985805 | `sections/results.tex` |
| 6 | 0.0016 / 0.0036 | CSV prior-BCTS minus prior-vector mean NLL = 0.001599750357 / 0.003611555588 | `sections/results.tex`; prior-BCTS main row |
| 7 | 0.071145 / 0.043893 | Existing mechanism result: permuted-minus-aligned holdout NLL; protected rounded table values 0.683592−0.612447 / 0.210277−0.166384 | `sections/results.tex`, `tables/mechanism.tex` in appendix |
| 8 | 90.13% / 45.06% | 100×(1−54067/547560)=90.12583096%; 100×(1−601627/1095120)=45.06291548% | `sections/results.tex` |
| 9 | 54,067 / 547,560 / 601,627 B | Encoded uniform3 payload / analytical 64+4×136874 dense message / 54067+547560 exchange | Method/results and implementation appendix |
| 10 | 80 / 44 / 2,600 B | 20×4 / 11×4 / 650×4 FP32 deployed predictor values | `sections/results.tex`; method/Discussion preserve scope |
| 11 | 341 / 54,408 / 217,632 B | Existing `05_核心方法研发/results/chain_integration/result.json`: application message 54408; 54408−54067=341; four new messages=217632 | Recoverable-upload paragraph in `sections/appendix.tex` |
| 12 | 54,067 B | 136874×3/8 + 535×(1+4) + 64 = 54066.75, packing rounds to 54067 | Codec calculation in `sections/appendix.tex` |
| 13 | Six original full-seed rows | CSV prior-vector rows with same-seed zero-vector differences; table below matches preserved `tables/seeds.tex` | `tables/seeds.tex` in appendix |

| Seed | Acc. (%) | NLL | ΔAcc. (pp) | −ΔNLL |
|---:|---:|---:|---:|---:|
| 20261231 | 76.0956 | 0.632057 | +0.3741 | 0.034321 |
| 20261232 | 75.6965 | 0.642482 | −0.1473 | 0.037891 |
| 20261233 | 74.9120 | 0.659311 | +0.3063 | 0.035041 |
| 20261241 | 93.4697 | 0.175308 | +0.2781 | 0.016792 |
| 20261242 | 92.6338 | 0.194890 | +0.1653 | 0.019176 |
| 20261243 | 92.8185 | 0.193788 | +0.0756 | 0.018739 |

## Deleted or relocated original sentences and material

Compared against the before-revision backup named above. These are argument/QA removals or relocations, not deletion of unfavorable numerical outcomes. Full old paragraphs remain recoverable in that backup.

| Original location and sentence / identifying excerpt | Revision action |
|---|---|
| `method.tex`: “This identity explains the fitted predictor, rather than asserting a new federated convergence result.” | Removed defensive sentence; the stationary identity is explained affirmatively in the method, abstract and contribution. |
| `method.tex`: “These are disclosed reference implementations.” | Removed; reference table uses neutral implementation scope. |
| `method.tex`: “This is a configured block-depth rule on the local development chain.” and “The trusted server, bridge/EVM endpoint, database, keys, and decoder enforce this route.” | Long route exposition compressed; demonstrated local scope and trust boundary retained in the implementation appendix. |
| `method.tex`: “The external tag authenticates bytes and context; it is not a weight-embedded watermark or proof of honest training.” | Concise factual boundary retained in the appendix; it is not promoted to a watermark deployment. |
| `method.tex`: detailed FedLC formula and codec block/scale/refinement/strict-decoding exposition | Relocated to implementation appendix; original training/codec semantics retained. |
| `evaluation.tex`: “This exploratory analysis uses the existing three seeds and twenty clients per task, without official TEST access or predictor selection.” | Replaced by direct development TRAIN/holdout protocol and fixed controls. |
| `evaluation.tex`: “It uses no official TEST data and adds no predictive-quality replicate.” and “Two manually mined descendant blocks exercise the acceptance rule on this single-node development chain.” | Integration-evaluation paragraph removed from the main paper; relevant arithmetic/system scope condensed in appendix and original artifact retained. |
| `results.tex`: “Thus prior-centered adaptation is useful across these two score families; the twenty-value form is the evaluated instance, rather than the basis for a class-specific-scale necessity claim.” | Replaced by concise cross-score-family finding; same-prior BCTS result retained. |
| `results.tex`: “The diagnostic supports informative regularization in this stable-prior setting, without establishing behavior under prior drift.” | Replaced by completed development reallocation diagnostic and focused Discussion boundary; adverse result retained. |
| `results.tex`: “All 128 assertions pass, alongside six EVM authorization and collision checks.” | Removed internal QA from research narrative; original checks/artifact unchanged. |
| `results.tex`: recovery-fixture/attack-case inventory beginning “Recovery handles connection loss…” | Removed long inventory from main text; implementation paragraph retains registration/incorporation/residual correspondence. |
| `results.tex`: “These measurements demonstrate an executed local contract--learning route; they do not estimate public-chain fees or multi-node consensus cost.” | Original 5.4 removed as a separate research result; message-byte values retained in appendix with local scope. |
| `discussion.tex`: “The evidence favors a focused design principle…” | Replaced by mechanism-oriented discussion. |
| `discussion.tex`: scattered prior-drift/feature-shift/device-cost and chain-deployment caveats | Reorganized into one compact Discussion boundary paragraph plus exact appendix implementation scope. No unperformed studies are described as completed. |
| `appendix.tex`: inference-replay QA paragraph containing “60,000 CNN predictions”, “3.67×10−16” and “291 pinned…” | Removed from paper; original evidence and old checks remain archived, not overwritten. |
| `appendix.tex`: “Software guidance. Figure preparation used the scientific-visualization skill … arXiv:2609.00065…” | Removed irrelevant software-guidance paragraph. |
| `appendix.tex`: “Clients and rounds are not independent training replicates.” | Statistical fact preserved in Evaluation footnote and dependence-aware paired-summary appendix. |
| Original abstract last sentence: “A two-round HTTP integration with a single-node EVM development chain…” | Replaced by prior-center main story and predictor/upload accounting. |
| Original conclusion last sentence ending “…on a single-node development chain.” | Replaced by next-step center re-estimation; implementation remains appendix support. |
| Old main-result rows beyond six; training/reference/mechanism tables | Relocated to appendix tables, not discarded. |
| Old two-point prediction scatter | Replaced by observed twenty-client × three-seed paired ΔNLL figure; original figure/provenance preserved. |

## Task Section 11 acceptance gates

All11 manuscript gates below are complete for the delivered revision. These gates verify the taskbook revision, not state-of-the-art performance or guaranteed acceptance. All selected experiments and their actual evidence scopes are recorded above.

| # | Acceptance item | Status / final evidence |
|---:|---|---|
| 1 | Abstract matches supplied argument with recorded evidence corrections, ≤150 words, no development-chain ending | PASS —136 whitespace-delimited abstract tokens in final source |
| 2 | Three contributions; baseline objectives as penalty endpoints | PASS — final source inspected |
| 3 | Introduction first paragraph includes 38.96% versus 63.06% with client-macro scope | PASS — first paragraph, client-macro scope |
| 4 | Paragraph-ending defensive-language count ≤8 | PASS — literal taskbook pattern hits8 source lines; paragraph-ending screen1 (historical implementation trust boundary) |
| 5 | Main table ≤6 rows; training/reference tables in appendix with one sentence each in main text | PASS — two main table floats; six predictor rows; training/reference appendix tables and one main sentence each |
| 6 | Main paper excludes internal QA numbers and manually-mined wording | PASS — no forbidden QA tokens or banned style words in paper |
| 7 | Discussion limitation paragraph P2, 2–3 sentences | PASS — exactly three sentences in P2 |
| 8 | Conclusion follows supplied argument with objective-endpoint correction; final sentence is next design step | PASS — endpoint correction recorded; last sentence is next design step |
| 9 | Required pdflatex→bibtex→pdflatex×2 succeeds; ≤12 pages; no undefined references or overfull boxes | PASS —required four-command sequence exited0;12pages incl. appendix/references,33cited/34bib; clean final log,27embedded font resources, final visual inspection complete |
| 10 | This status records completed work, outstanding P1/P2 and deleted-sentence list | PASS —record includes every selected completed P1/P2, stopped candidate and deleted-sentence map |
| 11 | Every supplied-prose number and all/both/within claim mapped to source | PASS —13-group map and minimum scope corrections above |

## Root finalization fields

- Build: **COMPLETE**, `/Library/TeX/texbin/pdflatex`→`bibtex`→`pdflatex`×2; actual logs/stdout and `BUILD_RECORD.json` in revision artifacts. Final12-page PDF SHA256 `30a5fa1694810e3e46c6862003e59754825f6af0de71d585854e14f1ff85a0b8`.
- PDF/layout: **PASS**, all12 current pages rendered at130dpi and inspected; table pages5/8/9 individually inspected.27embedded font resources,3vector figures, no clipping/overlap/missing glyphs, no undefined/overfull warnings. `visual_and_source_checks.json` records the exact PDF identity.
- References: **PASS**,33cited keys/34bib records; original byte prefix preserved; PRIOR, LaSCal and FedNPC primary-source relations checked, with metadata discrepancies retained.
- Numbers/captions/scope: **PASS**, `FINAL_PROTECTED_13_PRESENCE_CHECK.json`, `manuscript_checks.json` and `RESEARCH_EVIDENCE_REVIEW.md` independently cover all13protected groups and final added rows. Exactly2main tables/6primary rows;136-word abstract. All adverse evidence retained.
- Protected files: **PASS**,17original evidence/check/provenance/template files plus preamble and old bibliography prefix unchanged; six-seed table byte-identical. New copies verified again on promotion.
- Evidence identities: **PASS**, `FINAL_EVIDENCE_IDENTITY.json`:18blocks,1274references,677distinct current files, no missing/mismatched bytes. No fitting/training during this final identity audit.
- Selected experiment work: **COMPLETE**, including all three-partition and CIFAR-100 development runs, dense baselines and pFedMe. Geometry candidate STOP is a completed negative decision, not an unrun result or paper contribution.
- Delivery: promoted to `/Users/wind/Desktop/Codex/paper/03_英文论文/`; current `main.pdf`, source, results and `revision_artifacts/route_a_20261105/DELIVERY_MANIFEST.json` identify the release. `English_Paper_Complete_Package.zip` is rebuilt from the release with an explicit archive exclusion list. Existing historical `checks/DELIVERY_MANIFEST.json` is untouched and does not describe this release.
- Remaining author actions: author/affiliation and final venue/track-specific formatting before actual submission. No manuscript result is blocked by these metadata choices; no submission has been made.

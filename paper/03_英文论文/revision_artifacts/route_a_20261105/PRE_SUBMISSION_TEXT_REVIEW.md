# Pre-submission text review — Route A, current staged manuscript

Reviewed by `/root/route_a_citations` on 2026-11-05. This is a **read-only review**, not a manuscript rewrite. Applied `pre-submission-reviewer` and its grammar/LaTeX/logic references; applied `pdf` for the actual compiled PDF. The user taskbook overrides generic skill formulas: the four-paragraph Introduction is accepted, no six-paragraph rewrite is requested, and a third dataset is not treated as a universal publication rule. No new literature retrieval was performed. pFedMe and CIFAR-100 results that were still being generated are excluded from this review.

The reviewed source is `/private/tmp/pr_lfl_route_a/03_英文论文/`; the authoritative rendered output is `/private/tmp/pr_lfl_route_a/build/main.pdf`, not the older PDF/log/aux under the staged root or `revision_artifacts`. Source/PDF extraction and the initial full scan are recorded in `text_review_snapshot/SCAN.json` and `text_review_snapshot/compiled_pdf.txt`. The parent is editing concurrently; the exact quoted text below, rather than an assumed current line number, identifies each finding.

## Summary

At first inspection: **CRITICAL 0; MAJOR 3; MINOR 8**. The initial table-spacing MAJOR was subsequently corrected by the parent and visually rechecked; a new 0.62785 pt overflow appeared in Table 2 and is tracked with that same layout finding. The remaining findings are local and do not require a different paper story or additional experiments.

The Route A argument is coherent: local label counts motivate a prior center; the zero-center comparison isolates the center; the cyclic permutation tests class alignment; BCTS and vector scaling support the center across compact score families; communication costs are reported separately from local predictor state. The adverse ECE, weighted-accuracy and one-seed accuracy outcomes remain visible. The current PDF has 12 pages and the current bibliography has 32 cited keys/33 entries, with only the protected `rfc8998` entry uncited. All cited keys resolve.

The three highest-priority fixes are: (1) narrow the Introduction's logit-adjustment contrast to the actual fixed-strength control; (2) remove the nominal Wilcoxon numerical values or explicitly cease treating them as inferential evidence; (3) finish the table-width correction and recompile without the new overflow. No acceptance probability or conference tier is inferred from these text checks.

## Major findings

### M1 — Prior-work contrast overstates what is fixed in logit adjustment

- **Location:** `sections/front.tex:4`, Introduction P2.
- **Actual text:** “Logit adjustment fixes the offset but not its strength; vector scaling frees the fit but supplies no client-prior center”.
- **Why it matters:** “but not its strength” is ambiguous: it can mean that the offset's strength is not fixed, or be read as saying that logit adjustment cannot adapt its strength. The cited Menon paper explicitly includes a scalar `tau` and discusses treating it as a tuning parameter. Thus this is a positioning issue, not merely a preference about wording.
- **Local evidence:** existing archived primary text `/Users/wind/Desktop/Codex/paper/05_核心方法研发/literature/prior_personalization/sources/menon_la.txt`, §4.1, lines 426–457, includes `f_y(x)-tau*log(pi_y)` and “treat tau as a tuning parameter”. No new retrieval was used.
- **Concrete correction:** “Fixed unit-strength prior addition uses a preset offset; vector scaling fits scores without a client-prior center~\cite{menon2021logitadjust,guo2017calibration}.” The next sentence already explains duplicate information and finite local labels. This preserves the taskbook P2 role and distinguishes the actual fixed-prior control without claiming that all logit adjustment lacks a strength parameter.
- **Status at report creation:** sent to parent; verify the latest source before closing.

### M2 — Nominal Wilcoxon numbers still imply an unjustified independent-sample null

- **Location:** `sections/appendix.tex:15`, Paired analysis and diagnostics.
- **Actual text:** “Nominal two-sided Wilcoxon values are $p=1.735\times10^{-18}$ / $1.907\times10^{-6}$, descriptive under shared-model dependence.”
- **Why it matters:** the footnote correctly says the client–seed pairs are dependent, but the numeric `p` values still use the independent-difference reference distribution. Calling them “descriptive” does not validate that null distribution. Moreover, the slash does not tell readers that the two values refer to 60 pair differences versus 20 client means rather than CIFAR versus Fashion, the slash convention used elsewhere. This is a genuine statistical-logic weakness, independently identified by the statistics agent.
- **Concrete correction:** keep the observed facts: “All sixty paired NLL differences and twenty client means favor the prior center ($W=0$). Conditional client-resampling 95\% percentile intervals resample twenty clients 1,000 times (PCG64 seed 20261104), retaining their three seeds; they omit new-partition and training-run uncertainty.” Remove the two numerical `p` values from the paper, preserve their original calculation in artifacts, and record the minimal evidence-based departure from taskbook §5.1(b) in `REVISION_STATUS.md`.
- **Protection:** these nominal `p` values are not among taskbook §1.3's 13 protected groups. The protected NLL changes, all 60/60 facts, interval endpoints and six-seed table need not change.
- **Status at report creation:** pending parent decision; no source edited by reviewer.

### M3 — Table spacing corrected; final width check still needed

- **Original actual text/cells:** `tables/training.tex`, “FedLC-inspired + uniform3” directly adjacent to `$75.57\pm0.60$`; PDF p9 also had nearly joined cross-task numeric columns. Table 2 p6 similarly had insufficient cell separation. These were visible in rasterized page inspection despite a clean earlier TeX log.
- **Parent correction already inspected:** shorter labels `FedLC + u3`, `Prior + TopK`, `Zero`, `P-BCTS`, `P-vector`; abbreviations defined in captions; `tabcolsep` increased to 2.8 pt for Table 5 and 2.2 pt for Table 2. Re-rendered pp6/9 show substantially improved spacing. **The original visible collision is resolved.**
- **Remaining actual log text:** `/private/tmp/pr_lfl_route_a/build/main.log`: “Overfull \\hbox (0.62785pt too wide) in paragraph at lines 3--10”, corresponding to the revised quality table.
- **Concrete correction:** decrease Table 2's `tabcolsep` slightly (2.2 to 2.1 pt is enough in the current six-numeric-column layout) or shorten the predictor label further. Recompile and inspect at final scale; do not shrink fonts, change margins, change table numbers or discard rows.
- **Severity:** the initial collisions were MAJOR because entries became difficult to parse. The residual overflow is small, but taskbook's clean-log acceptance remains unmet until it is removed.

## Minor findings

### m1 — Accuracy unit changes inside Table 2 are unstated

- **Location:** `tables/quality.tex:2,15–16`.
- **Actual caption:** “Upper: Brier/ECE (lower is better), ECE in percentage points. … Lower: three-seed means.” Actual lower rows: “Weighted accuracy & 0.7475 & 0.7471 …”; “Client accuracy $q_{10}$ & 0.6027 …”.
- **Issue:** the main and appendix accuracy columns use percent, but this block uses proportions. The values are correct; the units need to be explicit.
- **Fix:** add “Lower-block accuracy is a proportion.” Alternatively put “(proportion)” in both accuracy row labels. Preserve every numeric cell. Upper-block Brier/ECE values and ECE percentage-point labels are otherwise consistent.

### m2 — Appendix reference text no longer identifies its intended subsection

- **Location:** `sections/method.tex:26,29`; `sections/results.tex:3,25`; `sections/discussion.tex:4`.
- **Actual text:** “(Appendix~\ref{app:implementation})”; “Appendix~\ref{app:sensitivity} reports frozen-center reallocation”; “(Appendix~\ref{app:statistics})”.
- **Actual compiled resolution:** `app:implementation`, `app:sensitivity`, and `app:statistics` all resolve to “A”, because they are under unnumbered subsections/paragraphs inside Appendix A. This is not an undefined reference, but it removes useful navigation.
- **Fix:** keep the compact Appendix A layout and add the heading name: “Appendix A, Implementation”; “Appendix A, Paired analysis and diagnostics”; “Appendix A, Penalty and reallocation diagnostics”. A `\pageref` can be added if helpful without touching the protected preamble. No new appendix sections are required.

### m3 — Undefined class-count symbol and generic-to-ten-class transition

- **Location:** `sections/method.tex:4,16`.
- **Actual text:** “$z(x;w)\in\mathbb{R}^C$”; “Its twenty 32-bit floating-point (FP32) values occupy 80 B.”
- **Issue:** `C` is inferable but never explicitly defined, and the prose jumps from a general `C`-class objective to twenty stored coordinates.
- **Fix:** “For $C$ classes, client $i$ …” and “With $C=10$, twenty … values occupy 80 B.” This also makes clear that the 80 B claim applies to the two ten-class tasks rather than an arbitrary class count. No protected number changes.

### m4 — Compact controls lack their essential functional forms

- **Location:** `sections/method.tex:32`.
- **Actual text:** “Prior-only, fixed-prior, bias-only, temperature, eleven-value BCTS … and a 650-parameter Linear head isolate counts, correction and capacity. TRAIN fits use coefficient 10; Linear runs one epoch.”
- **Issue:** the main objective defines vector scaling well, but the important eleven-value BCTS comparison is described by a name and a storage count only. It is unclear whether its single scale is positive, what the bias anchor is, and whether the coefficient statement includes the untrained prior-only reference/Linear head.
- **Existing source check:** frozen `calibration_baselines/solver.py` uses `az` for temperature and `az+b` for BCTS with `a>=.001`, scale anchor one, centered biases; the wrapper `prior_vector/bcts_control.py` changes the bias anchor to the same prior. The Linear control is a separate one-epoch head.
- **Fix:** replace the generic “TRAIN fits” sentence with a compact explicit definition: “Affine controls use ridge 10: temperature fits $az$, BCTS fits $az+b$ ($a\ge0.001$), and bias-only fits $z+b$; same-prior controls center $b$ at $\mu_i$. Linear runs one epoch.” This clarifies the comparison rather than expanding engineering detail.

### m5 — Misplaced capital after semicolon

- **Location:** `sections/appendix.tex:10`.
- **Actual text:** “Both match the original data, backbone, seeds and training schedule; Fits use TRAIN before TEST.”
- **Rule:** grammar reference G4, sentence/clause connection; a semicolon does not start a new capitalized sentence here.
- **Fix:** “… schedule; fits use TRAIN before TEST.” This is the one unambiguous mechanical English error identified. Present-tense results and present-tense related-work descriptions are acceptable conventions here; no wholesale tense rewrite is recommended.

### m6 — Comparison is grammatically compressed past its intended meaning

- **Location:** `sections/results.tex:16`.
- **Actual text:** “Sample-weighted NLL improves by 0.0217/0.0102, less than macro NLL.”
- **Rule:** grammar reference G4, retain a well-formed and unambiguous comparison.
- **Issue:** reductions, not NLL levels, are being compared.
- **Fix:** “Sample-weighted NLL reductions are 0.0217/0.0102, smaller than under macro averaging.” Evaluation already explains equal client weights, so no causal explanation needs to be invented.

### m7 — Two appendix captions no longer start with the finding

- **Location:** `tables/mechanism.tex:2`, `tables/training.tex:2`.
- **Actual text:** “Alignment and partition checks: development macro NLL.”; “Shared-training controls on original TEST: macro mean $\pm$ sample SD.”
- **Issue:** taskbook §6.1 explicitly requests a conclusion in the first caption sentence. These are correct setup descriptions, but compression removed that required rhetorical function.
- **Fix:** Table 4: “Aligned priors lower development NLL across the checked partitions.” Then the upper/lower definitions. Table 5: “Prior centering lowers NLL on dense FedAvg and FedProx.” Then the matched-implementation and original-TEST definitions. The table values, row counts and comparisons stay intact.

### m8 — SmallCNN's layer shape is not recoverable from its parameter count alone

- **Location:** `sections/evaluation.tex:2`.
- **Actual text:** “CIFAR-10 … and FashionMNIST … use a 136,874-parameter SmallCNN.”
- **Issue:** this is sufficient to follow the argument, but a generic architecture name is less informative than one short layer description; it is not a request for a portability/reproduction project.
- **Verified existing source:** `02_实验工程/tools/fedavg_update_diagnostic.py:24–35`: 3×3 convolutions with 16/32 channels, ReLU and 2×2 max pooling, then a 64-unit hidden layer and the class output.
- **Fix if space permits:** “SmallCNN uses two 3×3 convolutions (16/32 channels) with ReLU/max pooling and a 64-unit hidden layer.” This can live in the existing implementation paragraph. Do not change the backbone or rerun results.

## Full scan, citation and equation checks

The full active manuscript was scanned, not a sample: `main.tex`, every `sections/*.tex`, all five included table files and the bibliography key set. Literal AI-vocabulary substring hits were `exceeds` in the abstract and contribution, plus `yet` in the attribution contribution. These are ordinary precise mathematical/comparison uses, two inherited from the taskbook's prescribed language; they are **not inflated-performance language**. A mechanical ban would conflict with the user's required text. They are recorded in `SCAN.json` and are not counted as defects. Unicode em-dash/TeX semantic em-dash count was zero. No innovative/superior/surpass/notably/pave-the-way/underscore vocabulary was found.

All 32 cited keys exist and use a preceding nonbreaking tilde. The uncited protected `rfc8998` entry is harmless and should remain. This pass did not re-audit protected bibliography metadata or request modifying old entries. The added PRIOR/LaSCal descriptions match the already inspected full-text material: the LaSCal sentence describes the published temperature instance without asserting that its framework can only support temperature. Both numbered equations are referenced and contiguous. Multiple labels intentionally alias the combined Tables 4/5; this is valid, though prose can say “Table 5” once instead of twice in one sentence.

## Figure and PDF assessment

The actual 12-page build was text-extracted in full and pages 2,5,6,7,8,9,10 were inspected through rendered output/text. The three figure files are vector PDF; the provided generation source/provenance verifies embedded TrueType fonts, no image XObjects, 122 mm width and minimum label size 7.6 pt. Figure 1 separates one client's TRAIN counts from the aggregate TEST accuracy panel and labels the two populations. Figure 2 uses color plus marker shapes and black client-mean diamonds. Figure 3 does not fabricate a zero-penalty score and separates the infinity endpoint. No figure-specific MAJOR issue was found. The remaining table issue is described under M3 rather than misclassified as figure quality.

## Taskbook alignment and recommendation

The four functional Introduction paragraphs are present; its first paragraph contains 38.96%/63.06% with the client-macro qualifier. Contributions and conclusion retain the required argument and make the zero-penalty objective, rather than a potentially nonexistent minimizer, the endpoint. Main table has exactly six rows; only the main/quality tables occur before the appendix; all three new figures remain. Byte-accounting groups and the complete six-seed table remain present. The Discussion has three paragraphs and concentrates distribution/metric boundaries in P2. Neither an untested state-of-the-art claim nor a claim that the frozen reallocation diagnostic solves deployment drift appears.

**Provisional text/argument readiness: 7.5/10.** This is not a scientific novelty score or acceptance prediction. Finish the three localized MAJOR items and the minor clarity corrections, then update this review after the parent inserts the final pFedMe/CIFAR-100 evidence. Current recommendation: **needs a final local revision and a fresh final build before submission**. Running experiments are excluded from the rating; their completion or failure must not be inferred from this review.

## Follow-up dispositions — 2026-11-06, before final CIFAR-100 insertion

The parent applied changes while this report was being completed. The following closures are based on an additional direct source read; the original diagnostic quotes above are retained as review history. pFedMe result values newly inserted by the parent were **not independently audited in this text-review pass**; their score-verification record is separate.

| Finding | Freshly observed change | Disposition |
|---|---|---|
| M1, prior-work contrast | “Fixed unit-strength prior addition uses a preset offset; vector scaling fits scores without a client-prior center” | Closed: now limited to the actual fixed-strength control. |
| M2, Wilcoxon interpretation | “Independence-reference Wilcoxon calculations give … for sixty pairs … for twenty client means in each task; shared-model dependence precludes population-significance interpretation.” | Closed for the taskbook's explicitly requested descriptive calculation. The text now identifies the reference assumption, each number's sample aggregation and the invalid population-inference boundary. It must not be cited elsewhere as evidence of valid significance. Removing numerical values remains the statistically cleaner optional alternative; the review does not certify those values as calibrated inferential p-values. |
| M3, table spacing | Short labels plus increased separation; latest quality table uses `tabcolsep=2.1pt` | Original table collision closed by visual reread. Final full-build clean-log acceptance remains the parent's check after new-result insertion. |
| m1, units | “Lower: means; accuracy is a proportion.” | Closed. |
| m2, appendix navigation | Method's first implementation reference now says “Appendix A, Implementation”; remaining references still say Appendix A | Partially addressed; remaining locations are a minor navigation issue, not a wrong appendix number. |
| m3, C/ten-class state | “For $C$ classes …”; “At $C=10$, twenty … values occupy 80 B.” | Closed. |
| m4, controls | “Ridge-10 controls fit bias-only $z+b$, temperature $az$, or eleven-value BCTS $az+b$ ($a\ge0.001$); same-prior controls center $b$ at $\mu_i$.” | Closed. |
| m5, capital after semicolon | “… schedule; fits use TRAIN before TEST.” | Closed. |
| m6, reductions | “Sample-weighted NLL reductions are 0.0217/0.0102, smaller than under macro averaging.” | Closed. |
| m7, caption finding first | The two identified captions still start with setup descriptions | Remains minor taskbook wording compliance. |
| m8, architecture | SmallCNN is still named by parameter count only in the inspected text | Optional minor clarification remains. |

One new small English issue introduced during compression was sent to the parent: the method opening now says “client $i$ fits TRAIN set $D_i$”. Fit the **predictor**, not the set: “client $i$ fits its predictor on TRAIN set $D_i$”. This follows grammar reference G4's clause-meaning requirement and is MINOR.

No unresolved MAJOR text-logic finding remains after these scoped closures; final PDF width/page/reference checks and evaluation of newly inserted results are still outside this follow-up. **Updated text-only readiness: 8.5/10, pending final integration and the remaining small clarity edits.** This is still neither an acceptance prediction nor a novelty assessment.

# LaSCal: verified method overlap and citation recommendation

Independent review by `/root/route_a_citations`, using root-retrieved official
NeurIPS HTML and complete 29-page PDF. The `pdf` skill was applied for local
page-anchored text extraction. No new network request, dependency installation,
experiment or manuscript edit was performed by this reviewer.

**Cite LaSCal as a close post-hoc calibration / label-shift neighbor. The inspected
method does not duplicate PR-LFL's local supervised prior-centered affine fit.**
This resolves the earlier bounded-search “method unresolved” status for LaSCal;
FedNPC remains unresolved and must not borrow this conclusion.

## Verified published identity

- Title: *LaSCal: Label-Shift Calibration without target labels*.
- Authors: Teodora Popordanoska; Gorjan Radevski; Tinne Tuytelaars;
  **Matthew B. Blaschko** (the official middle initial should be retained).
- NeurIPS 2024, *Advances in Neural Information Processing Systems*, volume 37,
  pages 65386–65414, official publication date 2024-12-16.
- DOI: `10.52202/079017-2088`.
- Official page:
  <https://proceedings.neurips.cc/paper_files/paper/2024/hash/783c5986e1d6112cb4688d9b2105609a-Abstract-Conference.html>.
- Official PDF:
  <https://proceedings.neurips.cc/paper_files/paper/2024/file/783c5986e1d6112cb4688d9b2105609a-Paper-Conference.pdf>.

The optional new entry is `lascal_verified_append.bib`, key
`popordanoska2024lascal`. Root may append that new entry without touching the
protected existing bibliography prefix. Do not cite the related 2023 author
preprint as a substitute for this actually inspected published method.

## Reconstructed method from the primary PDF

**PDF p.4, §3, Eq.(1)–(5).** The available data are labeled source samples and
unlabeled target samples. Label shift assumes changing `p(Y)` and invariant
`p(X|Y)`, with target label support contained in source support. The prediction
model `f` is fixed. For each class the method estimates a source-to-target prior
ratio `omega_c=p_t(Y=c)/p_s(Y=c)`, obtained through established label-shift
estimators. A kernel ratio combines source labeled observations, the estimated
class ratio, and target prediction-density observations to estimate the target
conditional class frequency given a confidence coordinate. Its consistency
claim is conditional on its label-shift, support, weight-estimator and kernel
assumptions.

**PDF p.5, Eq.(6)–(7).** This conditional-frequency estimate yields a target
classwise calibration-error estimator. The stated LaSCal calibration procedure
optimizes a **single temperature T** by minimizing the estimated target
classwise calibration error over softmax logits divided by T. The paper also
explicitly says its estimator can be integrated with other post-hoc or trainable
calibration strategies; it is inaccurate to claim the general framework can
only support temperature scaling. Its instantiated positive-temperature scaling
is accuracy preserving; classwise free scales/biases in PR-LFL can change ranking.

**PDF pp.6–7, §4.1; pp.16–17, Appendix A.2–A.3.** The empirical pipeline primarily
uses RLLS for class-importance estimates; additional comparisons include
EM-BCTS, BBSL and ELSA. Appendix A.3 compares source-only vector scaling and
other calibrators against LaSCal's labeled-source plus unlabeled-target access.
The work is therefore directly relevant to calibration under changed label
mixtures, rather than an unrelated temperature-scaling baseline.

| Axis | LaSCal, inspected implementation | Current PR-LFL |
|---|---|---|
| Data used to adapt | Labeled source plus unlabeled target predictions | Client local TRAIN logits and true labels |
| Objective | Estimated target classwise calibration error | Local TRAIN cross-entropy plus quadratic penalties |
| Local label information | Target ratios estimated without target labels | Smoothed observed local TRAIN class counts |
| Calibration parameters | One temperature in Eq.(7); estimator extensible | Ten free scales and ten centered stored biases |
| Center / penalty | Eq.(7) does not use the PR-LFL prior-centered quadratic | Bias shrinks toward centered log local class frequency |
| Claim boundary | Label-shift calibration-error consistency under assumptions | Compact supervised local fit; no unsupervised target-shift consistency guarantee |

This table establishes concrete differences in the methods actually inspected.
It does not prove that every extension of LaSCal excludes a prior-centered
quadratic, nor that combining known output calibration with a prior center is
automatically novel. PR-LFL still needs the narrow information/mechanism and
cost evidence in its existing matched comparisons. A LaSCal performance baseline
would require a separately declared labeled-source/unlabeled-target protocol;
it cannot silently use held-out target labels or replace the existing matched
TRAIN-only controls. A citation and precise relation are appropriate now; no
new experiment is required merely to acknowledge this neighbor.

## Two sentences suitable for root's related-work paragraph

> LaSCal estimates classwise calibration error under label shift from labeled
> source data and unlabeled target predictions, and uses this estimate to fit a
> post-hoc temperature~\cite{popordanoska2024lascal}.
> PR-LFL instead uses observed client TRAIN labels to center a regularized affine
> fit on frozen logits; it does not estimate an unlabeled target shift or inherit
> LaSCal's consistency guarantee.

If the paragraph is particularly short, “LaSCal supports label-free target
calibration under label shift; our fit uses labeled client TRAIN data and an
observed-class-prior regularization center” is a faithful single-sentence
distinction. Avoid “LaSCal cannot use biases”, “our prior principle is new”, or
“we solve arbitrary heterogeneity”; the source does not support those claims.

## Extraction boundary

`pypdf` 6.19.0 extracted all 29 pages with page separators. It reported rotated
text in some figures/tables; no numerical result in those rotated regions was
used. The method judgments above rely on readable prose and equations in §3,
§4.1 and Appendix A.2–A.3, cross-checked against the official abstract. Original
PDF/HTML and extracted text hashes are recorded in `LASCAL_SOURCE_RECORD.json`.

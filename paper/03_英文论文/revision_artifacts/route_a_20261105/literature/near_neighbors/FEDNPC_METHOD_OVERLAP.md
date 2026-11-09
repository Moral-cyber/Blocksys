# FedNPC full-text method overlap and citation review

Independent reviewer: `/root/route_a_citations`; 2026-11-06 (Asia/Shanghai). This review uses the **complete official CVF paper**, not its title, abstract or a search snippet. The ten-page PDF was extracted with bundled Python and `pypdf 6.10.0`; `fitz` was not used. All pages were extracted/read. Some figure text on PDF p8 has font-encoding artifacts; method conclusions use readable prose, equations and algorithm on pp3–5. No quantitative result is copied into PR-LFL, and no new training or official TEST access was performed.

## Source integrity and formal identity

Official PDF URL:

<https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_FedNPC_Stochastic_Noise-driven_Post-hoc_Classifier_Calibration_Method_for_Federated_Long-tailed_CVPRF_2026_paper.pdf>

Official HTML URL, linked in the checked CVF listing:

<https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_FedNPC_Stochastic_Noise-driven_Post-hoc_Classifier_Calibration_Method_for_Federated_Long-tailed_CVPRF_2026_paper.html>

- **Title:** FedNPC: Stochastic Noise-driven Post-hoc Classifier Calibration Method for Federated Long-tailed Learning
- **Authors/order:** Jintong Gao; He Zhao; Yibo Yang; Dandan Guo
- **Venue:** Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) **Findings**, June 2026. Do not silently call this a CVPR main-track paper.
- **CVF pages:** **7737–7746**. The official listing's BibTeX and all PDF printed page footers agree.
- **PDF:** 1,643,740 bytes, 10 pages, SHA-256 `b004153f11ecefd58bdebdb6b70cdcc850ed884ae46163d530bae3c4e335aa34`.
- **Download completeness:** all twelve recorded ranges return 206 with matching Content-Range, size and total; concatenation exactly matches the complete PDF. First range is 0–139999, final range is 1540000–1643739. PDF parser successfully reads ten complete pages and xref/trailer.

### Explicit metadata-version difference

The pre-existing Crossref response matches title, four authors, 2026 Findings event and candidate DOI **10.1109/cvprf71909.2026.00743**, but gives **7740–7749**, not the CVF page range. It identifies the IEEE document URL `https://ieeexplore.ieee.org/document/11697702/`. The older local `fednpc_ieee.html` is **zero bytes**, so it supplies no direct IEEE-page verification.

This may be different proceedings pagination; its cause has **not** been established. Do not guess that either DOI is fake or the CVF pages are erroneous. The optional append-only BibTeX supplied here cites the actually inspected **CVF edition**, uses 7737–7746 and its official URL, and omits the candidate DOI rather than silently combine metadata variants. The candidate DOI/Crossref pages remain preserved in `FEDNPC_SOURCE_RECORD.json` and `fednpc_crossref.json`. No protected old bibliography entry is altered.

## Actual method, anchored to pages and equations

| Evidence | What the full text establishes |
|---|---|
| PDF p1, printed 7737, abstract; p2/7738, Introduction | FedNPC addresses **federated long-tailed learning**: both global class imbalance and client heterogeneity, aiming at an unbiased global model. Its diagnostic concerns class-specific classifier weight norms. |
| PDF p3/7739, §3.1, Eq. (1) | Conventional sample-weighted federated global model training obtains feature extractor `theta_g` and classifier `W_g`. |
| PDF p3/7739, §3.3, Stage 1/2; PDF p4/7740, Eq. (2) | Freeze the learned global feature extractor. On the **server**, generate virtual **feature-space** vectors `z_j ~ N(0,I_d)` and labels `y_j ~ U(C)`. `d` is feature dimension; these `z_j` are not observed examples or the frozen class-output logits used by PR-LFL. Minimize average CE in `W_g` over these pairs; each classifier score is `w_{g,c}^T z_j`. |
| PDF p4/7740, Algorithm 1, lines 5–11 | Repeatedly generate the virtual dataset and apply SGD to `W_g` for a finite number `E` of epochs; return **one calibrated global classifier** with the frozen feature extractor. It is not a per-client posterior prior-center fit. |
| PDF p4/7740, §3.4, Proposition 1, Eq. (3); p5/7741 | Gives expected gradient `E_z[(p(c|z)-1/C) z]` and interprets the resulting dynamics as classifier norm equalization. This does not give PR-LFL's class-count/bias stationarity identity and must not be cited as a guarantee for it. |
| PDF p5/7741, §3.5 | Stage 2 involves no client communication and no auxiliary real data, client feature statistics or gradients. Thus **post-hoc freezing, low adaptation traffic and plug-in calibration are shared prior-art properties**. They alone cannot establish PR-LFL novelty. |
| PDF p5/7741, §4.1; p6–8/7742–7744 | Experiments target CIFAR-10-LT, CIFAR-100-LT and ImageNet-LT; global LT factors plus Dirichlet heterogeneity, predominantly test accuracy. The global-model setting/backbones/training budgets differ from PR-LFL's fixed client-macro prediction protocol. Published results are not numerically comparable to the current PR-LFL table. |

In Eq. (2), the optimization is over the global classifier matrix. It is not a quadratic penalty shrinking a client bias to its observed centered log prior. No local class counts appear in that calibration objective, and its target labels are uniform random labels. The paper calls its update an adaptive norm equalizer; this review records that interpretation without independently proving all subsequent norm claims or endorsing a formal privacy guarantee.

## Exact relationship to PR-LFL

| Axis | FedNPC | PR-LFL evaluated construction |
|---|---|---|
| Calibrated object | One server/global feature-to-class classifier `W_g` | Per-client affine transform of a frozen shared class-logit vector |
| Calibration input | Synthetic isotropic `d`-dimensional feature noise | Observed client TRAIN class logits and true labels |
| Label information | Virtual uniform class labels | Empirical client class counts plus supervised CE labels |
| Objective | CE on virtual balanced pairs, optimized by finite SGD | Sum supervised CE plus lambda-10 scale/bias quadratic with bias center `center(log(n+1))` |
| Changed parameters | General classifier matrix entries | `C` class scales and `C` stored centered biases |
| Intended target | Global long-tail debiasing / classifier norm equalization | Client-specific prediction under label skew |
| Additional client communication | None in post-processing stage | None for local predictor fitting |

**Verdict:** FedNPC is a close and relevant published federated post-hoc calibration neighbor and should be cited. The inspected objective and deployment location differ materially from the current PR-LFL construction. This is **bounded method differentiation**, not proof that PR-LFL is first, that all special cases are disjoint, or that PR-LFL outperforms FedNPC. In particular, adjusting a classifier and scaling output logits can intersect in restricted parameterizations, so avoid an absolute “no overlap” claim. The local observed prior center and matched class-permutation attribution remain the narrower PR-LFL story.

## Ready-to-insert sentence

> FedNPC recalibrates a global classifier on server-generated isotropic features and uniform labels~\cite{gao2026fednpc}; we fit client-specific affine predictors on labeled TRAIN logits with a prior-centered penalty.

This sentence states only the verified mechanism and the actual PR-LFL distinction. No new baseline result, privacy superiority, measured runtime or state-of-the-art claim follows from it. A new numerical FedNPC comparison would require a separately specified global-versus-client endpoint and matched access protocol; it is not silently supplied by the citation.

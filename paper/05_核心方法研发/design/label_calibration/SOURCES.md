# Sources and verification boundary

Checked on 2026-11-04. One primary paper; no third-party implementation read; no training or experiments executed.

- Jie Zhang, Zhiqi Li, Bo Li, Jianghe Xu, Shuang Wu, Shouhong Ding, Chao Wu. Federated Learning with Label Distribution Skew via Logits Calibration. ICML 2022 / PMLR 162:26311–26329.
  - Publisher page: https://proceedings.mlr.press/v162/zhang22p.html
  - Full paper (19 PDF pages): https://proceedings.mlr.press/v162/zhang22p/zhang22p.pdf
  - Saved original: `fedlc_primary.pdf`; extraction: `fedlc_primary.txt` (pypdf, with surrogate replacement); publisher metadata: `fedlc_primary.html`.
  - PDF p3: Definition 2 sets missing-class count n_s=0.
  - PDF p4: learning objective states prediction argmax_y f_y(x).
  - PDF p5: Eqs. 5–7; Eq.7 denominator excludes true class. Confirmed visually from rendered `fedlc_page5.png`.
  - PDF p6: §5.2 implementation defaults, 20 clients, batch128, SGD lr0.01, no weight decay, 5 seeds.
  - PDF p14: Algorithm 1 local model download, calibrated loss update, full local model return.
  - PDF p17: Appendix C probability denominator explicitly sums all K classes, inconsistent with p5 Eq.7.
  - Whole-paper text checked for τ and zero-count implementation prescriptions: no numerical τ default/search range or explicit count smoothing rule located.

Official repository discovery (metadata only, no third-party code fetched):

- https://api.github.com/search/repositories?q=FedLC&per_page=20
  - Saved `github_repo_search.json`. Same-name repositories include unrelated medical segmentation and a different ISPA 2022 logit-calibration paper; these were not treated as official FedLC code.
- https://api.github.com/search/repositories?q=%22Federated%20Learning%20with%20Label%20Distribution%20Skew%22&per_page=10
  - Saved `github_title_search.json`; only FedMGD metadata result, a different paper.
- https://api.github.com/search/repositories?q=%22Federated%20Learning%20with%20Label%20Distribution%20Skew%20via%20Logits%20Calibration%22%20in%3Areadme&per_page=100
  - Saved `github_readme_search.json`, 75 results; no author repository confirmed from inventory. This finite search does not prove no official repository exists. Formal page/PDF contain no source-code repository link found.

Implementation suggestions in LITERATURE_CHECK.md (all-class CE, count floor, τ=0 branch, TopK+EF composition, 2×2 isolation) are explicitly recommendations, not verified author code behavior.

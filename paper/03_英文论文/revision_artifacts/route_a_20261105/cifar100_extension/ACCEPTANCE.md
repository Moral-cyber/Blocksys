# CIFAR-100 extension: complete verified development evidence

All three predeclared model seeds completed 20 rounds on the one fixed partition (20261107), followed by TRAIN-only affine fitting and development-holdout scoring. This completes the **added-dataset development experiment**; no official TEST scoring was performed.

Protocol: official fine-label TRAIN images, 20 clients, Dirichlet-multinomial alpha .1, minimum100examples/client, 75/25 TRAIN/holdout, fixed2000-row stratified monitor; the same SmallCNN trunk with100outputs, one local epoch, batch64,SGD.03,20rounds, explicit FedLC-inspired shift and uniform3+FP32EF. New model size142724parameters; upload56376B, EF residual570896B; prediction state200FP32coordinates=800B. The zero/prior affine models have identical capacity, fitting data, fixed lambda10 and numerical schedule.

| Predictor | Client-macro NLL mean ± sample SD | Client-macro accuracy (%) mean ± sample SD |
|---|---:|---:|
| Shared logits | 3.946613 ± 0.012279 | 10.0541 ± 0.4153 |
| Fixed prior addition | 2.372943 ± 0.013441 | 33.0174 ± 0.3311 |
| Zero-center affine | 2.428310 ± 0.011646 | 35.2671 ± 0.3947 |
| Prior-center affine | 2.289578 ± 0.011366 | 35.1211 ± 0.2845 |

Prior-minus-zero NLL per model seed is −0.138326, −0.141190 and −0.136678; all3runs and all60client-seed cells improve. Fixed prior addition beats the zero-center affine fit in NLL, while fitted prior-centered prediction improves further. Thus this dataset supplies a real class-count-information result, but it does not support universal superiority of fitting over fixed correction for all metrics.

The adverse secondary results are retained. Prior centering lowers mean macro accuracy by0.1460percentage points and sample-weighted accuracy by0.1493points; two of three seeds lose accuracy. Client-mean ECE rises from0.04976695 to0.05058740 and weighted client ECE from0.04856534 to0.04991236. Mean client accuracy q10 falls by0.0172points. Brier and pooled ECE improve, so the appropriate conclusion is **consistent NLL improvement with metric-dependent trade-offs**, not an across-the-board calibration/accuracy win. `formatted_development.json` and `client_metrics.csv` retain all arms, clients and secondary metrics.

Independent verification (`INDEPENDENT_RESULTS_VERIFICATION.json`) read saved TRAIN/development logits, FP32 states and manifests, without fitting or CNN inference. It replayed240arm-client metric rows with maximum absolute disagreement3.33e−16, independently checked the count anchors and fitted FP64 projected gradients (maximum3.47e−6, below1e−5), objective monotonicity, centered-state shape/bytes, exact membership and source/data/checkpoint identities. The structured solver uses a200×200Hessian and no N×K×2Kdesign cache. This is a local implementation/evidence check, not external peer review or a novelty certificate.

Minimal manuscript sentence for root integration, accompanied by the compact appendix table:

> On CIFAR-100, extending the same protocol to a 200-coordinate (800 B) predictor reduces development client-macro NLL from 2.4283 to 2.2896 over three seeds, with 56,376 B shared uploads; mean accuracy decreases by 0.1460 percentage points (Table~\ref{tab:cifar100dev}).

The sentence and table must remain explicitly development evidence. Do not replace original TEST values, alter the protected thirteen numerical groups, claim all data heterogeneity is resolved, or imply that a100-class result fits in80B.

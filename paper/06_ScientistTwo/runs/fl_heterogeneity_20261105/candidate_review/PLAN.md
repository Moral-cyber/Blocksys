# Development-only residual-anchor screen: proposed, not executed

Status: **AWAITING ROOT CONFIRMATION OF EXACT CANDIDATE**. Written before candidate fitting. This plan does not authorize implementation or execution by itself.

## Hypothesis and competing explanation

H1: local TRAIN predicted class mass contains enough information about a shared model's class bias that subtracting its log from the local log-prior regularization center improves out-of-sample NLL. Rival: that mass mainly reflects the client's own label mixture, so subtraction removes helpful information; any gain is standard prior-ratio/logit correction plus familiar shrinkage. Generic ratio correction and regularization centers are prior art. Anticipated novelty is low; a positive screen alone does not justify full promotion.

## Exact candidate proposed for approval

For each frozen `fedlc_uniform3` checkpoint and client, use only its saved TRAIN logits/labels. Let `pi_c=(n_c+1)/(N+10)` as in the original method. Compute TRAIN softmax in FP64 with max subtraction, and define the fixed pseudo-count smoothed predictive marginal `q_c=(sum_j p(c|x_j)+1)/(N+10)`. The same one-pseudo-observation convention keeps q positive and avoids a new epsilon/tuning grid. Set `mu_ratio=center(log(pi)-log(q))`. This is **local predictive-marginal residual centering**, not an estimator of the source prior. No pooled statistics, extra CNN forward, backward, upload, or extra deployed state.

Fit `sum_j CE(A*z_j+b,y_j)+10/2||A-1||²+10/2||b-mu_ratio||²`, with centered b, unrestricted real A, original max50 FP64 Newton/Armijo, then deploy twenty FP32 values. Only the bias center changes. Source files and old states remain read only.

## Fixed inputs and matched controls

Exactly six old **development** runs: CIFAR partition20261217/seeds20261231/32/33; Fashion partition20261219/seeds20261241/42/43; final20 `fedlc_uniform3` only. Read `results/predictive_pfl_confirm/evaluation_{task}_p{partition}_s{seed}/fedlc_uniform3_client*.npz` for TRAIN/holdout arrays and the paired `adaptation_*` states/reports. Process all20 clients and all3 seeds per task; no seed/client selection.

Six predictor definitions, all reported:

1. Existing zero-centered 20d vector, exact saved state/replay.
2. Existing prior-centered 20d vector, exact saved state/replay.
3. New ratio-centered 20d vector.
4. Fixed original log-prior offset `z+center(log pi)`; no fit.
5. Fixed ratio offset `z+mu_ratio`; no fit, to expose classical correction without learned shrinkage.
6. Ratio-centered 20d vector with a fixed one-class cyclic shift of q only: `mu_shift=center(log(pi)-log(q[(c+1) mod10]))`. Keep pi aligned and shift q to test whether its class correspondence matters. This deliberately does not preserve the residual center's norm; report that norm/proximity-to-zero diagnostic so a shrinkage-only explanation remains visible.

All controls have exactly the same TRAIN access. Comparing fitted ratio to fixed offsets does not equalize model capacity; the causal center comparison is against the matched 20d zero/prior vectors. There is no lambda search and no promotion from accuracy alone. The cyclic shift is fixed above, matching the limitation/idea agent's `residual_marginal_v1` rather than selecting a convenient permutation. Lock all input/source hashes before first fit. Candidate runner should audit-reject `predictive_final_test`, `test_batch`, `t10k`, raw dataset paths and model checkpoints; saved development arrays only.

## Outputs and predeclared interpretation

Primary: client-macro holdout NLL per seed/task and paired ratio-minus-original-prior differences. Also report macro and sample-weighted accuracy/NLL/Brier/ECE15, client-accuracy q10, every client/seed result, q/pi values and center norms. Keep all failed/negative outcomes. Sixty client--seed observations are conditional repeated measurements on one partition, not 60 independent runs.

Pass to a **separate discussion**, not automatic full promotion, only if both tasks have lower three-seed mean NLL than both the original prior and zero centers, at least2/3 seeds improve over the original prior on each task, no seed loses >0.5pp macro accuracy relative to the original prior, and aligned ratio has lower mean NLL than the shifted-q control. All thresholds are declared here before fitting, not used to retune q/lambda/permutation. Failure or mixed tasks stops this formulation; no rescue by changing the denominator or selecting a favorable metric. Even a pass remains low novelty unless a new mechanism or information-identification argument survives nearest-work review.

Deliver only `screen_result.json`, per-client CSV, code/hash lock and a concise mechanism decision under this candidate directory. Do not modify `03_英文论文`, old results, official TEST evidence, taskbook-protected files, or existing numeric groups. New-partition confirmation, official TEST scoring and manuscript promotion require a later explicit root decision.

# CIFAR-100 P2 extension: frozen development protocol

Plan frozen before CIFAR-100 model training or predictor scores. This is an isolated extension, not an edit to the protected 10-class method, its 13 numerical groups or original evidence. Official data acquisition is handled by root; this plan/adapter does not download.

## Data and allocation

Use official CIFAR-100 **fine labels**, only the 50,000-row TRAIN file. Require extracted-file MD5 `16019d7e3df5f24257cddd939b257f8d`, shape 50,000×32×32×3 and exactly 500 examples per class. Record SHA-256 and label/order hashes. Reject any open of an official `cifar-100-python/test`, CIFAR-10 `test_batch`, `t10k` or prior final-TEST outputs. The generic CIFAR-100 filename `test` must be guarded explicitly. No TEST score, selection or TEST-membership construction occurs.

New partition seed: **20261107**. Model/local training seeds: **20261231, 20261232, 20261233**, the same three identifiers as original CIFAR; new dataset means new model runs. Reuse the original classwise Dirichlet-multinomial algorithm with alpha .1,20clients,min100 total examples,75/25 local TRAIN/development split, fixed stratified 2,000 holdout monitor. Clone the frozen partition module into a distinct import namespace and change only its explicit `NUM_CLASSES` to100; raw CIFAR-10 loading is never called. Verify unique exact coverage, disjointness and 20 monitor examples/class. Original-row IDs are preserved and embedded in config.

## Shared training

SmallCNN keeps exactly the original feature trunk and uses `SmallCNN(classes=100)` final layer. Expected parameters:142724. Full20client participation,20rounds,one local epoch,batch64,SGD lr.03, local seed `model_seed+100*round_zero_based+client`. Loss is the same explicit FedLC-inspired variant `CE(z-max(n,1)^(-1/4),y)`, with100 local class counts. Sample-count aggregation; same uniform3 quantizer and FP32 error feedback. No checkpoint/round choice by development metric: use final round20 only. Root owns the MPS launch queue; preflight and fixtures never launch CNN training.

Exact per-client full update wire size becomes56376B, including64Bheader and558blocks of up to256coordinates. Set cap56376 so it fits. Dense analytical update=570960B, FP32 EF residual=570896B. Do not reuse54808 cap,136874shape checks,80Bpredictor claim or54067wire value in the new task.

## Post-hoc fit and development scoring

Freeze the final shared model. For every client, forward local TRAIN first; fit zero-center and local `center(log(n+1))`-center affine predictors on TRAIN only, fixed scale/bias ridge10 and unrestricted scales,centered biases. Read development holdouts only after the fits are fixed. Round each state once to FP32 before scoring FP64-transformed logits. Two100-vectors=200FP32coordinates=800B prediction state,0extra-upload. Keep all20clients and all3seeds, including adverse NLL/accuracy. No official TEST; no lambda/method selection.

Primary endpoint is paired client-macro development NLL; report per-seed/sample-weighted NLL, macro/weighted accuracy,q10 and all client deltas. Also report raw shared and fixed-prior scores from the same logits as descriptive reference. No newly claimed state-of-the-art comparison or new mechanism follows from an extra dataset alone.

## Structured convex solver

The `2K`-parameter convex objective is the same sum CE plus ridge. Compute analytic gradients/Hessian blocks directly from `p`, `p*z` and weighted diagonal sums; never form an `N×K×2K` design tensor. Use batches of at most512logit rows for Hessian accumulation and a200×200FP64Hessian (320000B). Keep the original50step Newton/Armijo schedule, tolerance1e-8, fallback at numerical resolution and deployedFP32 states. Center the bias and Newton direction. Log objective monotonicity, projected gradient,eigenvalues,fits/time and Python-traced allocation separately from persistent state.

Before real scores, run K10equivalence against frozen original prior_vector solver for zero/prior centers, FP64objective/gradient/Hessian comparisons, deployed-state/metric equivalence, finite differences and K100positive-curvature/finite fit fixtures. Numerical equality bounds are fixed by fixtures; no hyperparameter repair is allowed after scores. A failed fit is indeterminate and stops scoring rather than becoming a fabricated result.

## Source and launch discipline

Create fixture/source manifest before TRAIN data preflight; create data-ready manifest only after actual root-downloaded TRAIN bytes pass integrity. All runs validate source/data locks before and after. The three-run launcher runs sequentially and resumes only exact matching config/signature. Root launches it only after preflight PASS and a free MPS slot. This child prepares files and checks only; it never starts a CNN run.

# Matched dense federated baselines

Frozen 2026-11-05 before new baseline training or TEST inference.

## Question and claim

Add the basic dense FedAvg and FedProx training references requested in taskbook section 5.3. Report their raw predictors and, separately, the same post-hoc zero-centered and TRAIN-prior-centered vector calibration used by the paper. The calibrated variants are additions to each training baseline, not original FedAvg/FedProx definitions. Results test whether the prior-center behavior extends to these basic dense models; no state-of-the-art claim or acceptance gate is introduced.

## Fixed matrix

- CIFAR-10: partition 20261217, training seeds 20261231/20261232/20261233.
- FashionMNIST: partition 20261219, training seeds 20261241/20261242/20261243.
- Both: original SmallCNN 136874 parameters, 20 clients, full participation, exact original local TRAIN indices/counts and normalization, 20 rounds, one local epoch, batch 64, SGD learning rate 0.03, original initialization and local batch seed `seed + 100*round_index + client`.
- Dense FedAvg: ordinary CE, sample-count-weighted delta aggregation; no quantization or error feedback.
- Dense FedProx: the same CE and aggregation with fixed proximal coefficient mu=0.1, local objective CE + mu/2 * ||w - w_global||^2. No mu search or TEST-based reselection.
- Dense accounting is analytical: 547560 B per upload and per downlink, including the existing 64 B message header.

This is an explicitly disclosed in-house matched implementation. The archived original FedProx paper identifies https://github.com/litian96/FedProx and TensorFlow 1.10.1; that repository/runtime is not present locally. We reuse the frozen project training/data functions for matching rather than claim author-code reproduction.

## Fitting and scoring

Before new TEST access, fit and freeze each client's zero-centered and prior-centered 20-dimensional vector on its own TRAIN logits/labels using the original `prior_vector/solver.py`, ridge 10, max 50 Newton/Armijo, centered bias, unrestricted real scales, FP32 deployment state. Zero center uses b anchor zero; prior center uses centered log(TRAIN counts + 1). No CNN gradient or upload for fitting. All 12 models and their states are frozen before either task's TEST score stage.

Use the original strict official TEST loaders and deterministic TRAIN-count-derived client assignment (assignment seed 20261301), verify assignment hashes match existing official main results. Report raw, zero-centered vector, and prior-centered vector for every run and every client: client-macro and sample-weighted accuracy/NLL/Brier/ECE15, pooled ECE15, client-accuracy q10; aggregate over exactly three seeds as mean and sample SD. Original baseline raw and our calibrated variants remain distinct labels. GT comes only from official dataset labels. No TEST fitting, hyperparameter choice, stopping or seed replacement.

## Run order and checks

1. CPU synthetic fixtures: mu=0 route equals frozen trainer exactly, proximal gradient equals autograd of explicit objective, aggregation identity, metric known cases. No official TEST.
2. First CIFAR seed, both methods; verify 20 complete finite rounds, unchanged global anchors between clients, exact partition and fixed seed/batch settings. Continue based on execution integrity only, regardless of prediction quality.
3. Remaining CIFAR seeds then Fashion seeds, sequential MPS access coordinated with other training agent. New directories only. Do not edit old sources/results.
4. Fit all states in TRAIN-only process; freeze a hash lock over all models, configs, states, PLAN and source identities.
5. Fixed TEST scoring in a separate process; verify before/after hashes and all three seeds. Summarize, retain every result, report adverse results.

Historical 20-round MPS costs are 60–91 s for CIFAR dense and 123–310 s for Fashion dense; 12 runs are estimated at 20–40 min plus fit/scoring. This estimate is not a speed result. Current sandbox cannot expose MPS, so actual training uses normally approved escalation; no CPU fallback.

## Boundaries

This finite run completes basic FedAvg/FedProx references only. It does not implement Ditto/pFedMe or a new dataset. Those remain separate taskbook items. Numeric groups and original evidence files are unchanged. Cross-family integrity review is unavailable unless an actual independent backend is supplied; local fixtures/source checks do not claim it.

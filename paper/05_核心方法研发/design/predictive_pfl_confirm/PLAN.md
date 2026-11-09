# Prospective confirmation: prior-regularized vector adaptation under low-bit FL

Frozen before creating the new partitions or training their models, 2026-11-04.

## Question and provenance

Can a local label-prior anchor reduce personalized predictive log loss at the same vector-calibration capacity while retaining classification accuracy after low-bit federated training?

This question was selected after old-partition results. The earlier ten-dimensional bias and prior-vector accuracy-oriented gates retain their STOP decisions. This is a new prospective question, not a relabeling of those experiments as successes. No atomic novelty is claimed for vector scaling, a class prior, ridge regularization, Newton optimization, FedLC-inspired training, quantization, or error feedback. The proposed contribution is the motivated combination and its measured quality/cost behavior.

## Fixed method

Use the frozen shared SmallCNN (136874 parameters), the exact existing FedLC-inspired tau1 loss, uniform3 block quantization with actual decode before sample-weighted aggregation, and standard FP32 error feedback. Upload cap is 54808 B; actual uniform3 wire is 54067 B. Downlink remains dense. The same exact existing CE-uniform3 and FedLC-dense controls are required.

After the final checkpoint, each client fits A,b using only its own TRAIN logits and labels:

    sum_x CE(A * z(x) + b, y)
    + (10/2) ||A - 1||^2
    + (10/2) ||b - center(log(n_class + 1))||^2.

A is unrestricted real; b is centered. Reuse work/prior_vector/solver.py without alteration: max50 FP64 Newton/Armijo, fixed ridge10 and tolerance. Store A,b as twenty FP32 parameters, 80 B; score by applying that deployed state to original logits in FP64, as for every baseline. There is one extra TRAIN forward, numerical fitting/cache cost, no new CNN backward or upload. These costs are reported. Neither zero CNN backward nor small calibrator state implies a measured energy or end-to-end speed advantage.

## Prospective experiments

- CIFAR-10: strict five TRAIN files only, alpha0.1, 20 clients, fresh partition20261217, training seeds20261231/32/33.
- FashionMNIST: strict two TRAIN gzip only, alpha0.1, 20 clients, fresh partition20261219, training seeds20261241/42/43.
- Three training methods per task/seed: CE-uniform3, FedLC-uniform3, FedLC-dense; 20 rounds, SGD0.03, batch64, one local epoch, full participation, sample-count weights, identical initialization and batch seeds within each paired run.
- Use only each final checkpoint. All client TRAIN-internal holdouts are development confirmation, not official TEST. Official TEST bytes and metrics remain forbidden during this stage.
- There is no loss/lambda/bit/seed/round selection. Do not replace a missing or failed seed. Record actual row counts and all source/data/config hashes. Old p20261203 and Fashion p20261207 results are diagnostic only for this new question.

## Required same-array controls

raw; TRAIN prior only; raw plus fixed TRAIN prior; prior-anchor ten-dimensional bias; zero-anchor bias; positive scalar temperature; BCTS; zero-anchor twenty-dimensional vector scaling; matched one-epoch last-Linear adaptation, with and without fixed prior. Use the frozen implementations, same TRAIN access, same coefficient10 (not claimed to equalize statistical complexity across dimensions). A persistent FedRoD-linear reference and same-cap TopK+EF are required compression/learning references before any final paper claim.

## Primary endpoint and continuation criteria

The primary endpoint is client-macro holdout negative log likelihood (NLL). Accuracy is a preservation constraint. Multiclass Brier (sum over ten classes) and top-label ECE (15 fixed equal-width bins, last including1) must also be reported, without claiming they follow from NLL. Include weighted metrics and client-accuracy q10. Do not use ECE/Brier to select another calibrator.

For the first paired seed of each new task, all these development conditions must hold to continue this fixed formulation on that task:

1. FedLC-uniform3 prior-vector NLL is at least0.01 lower than same-model zero-anchor vector; accuracy is no more than0.5pp lower.
2. Its NLL is at least0.01 lower than same-model fixed TRAIN-prior prediction; accuracy is no more than0.2pp lower.
3. Its NLL is no more than0.005 above same-model prior-anchor ten-dimensional bias; accuracy is no more than0.5pp lower.
4. Relative to CE-uniform3 with the same prior-vector adaptation, NLL is no more than0.01 higher and accuracy no more than0.5pp lower. Passing this does not prove FedLC is necessary.
5. Relative to FedLC-dense with the same adaptation, NLL is no more than0.01 higher and accuracy no more than0.5pp lower.

If any first-seed condition fails, stop extension of this fixed formulation on that task, report it, and do not tune the rule to pass. If it passes, finish exactly the other two planned training seeds. At three seeds report each seed and paired mean/sample SD; a reliable task-level claim requires all three seeds to preserve accuracy within the respective bounds and each seed's NLL improvements in(1–2) to be positive, with mean improvements at least0.01. No post-hoc gate change or per-seed best model is allowed. Failure remains a failure even if another task passes.

## Subsequent evidence and stop boundary

Only if these prospective results support the narrower log-loss question, add one predeclared same-task moderate-heterogeneity setting, longer-training setting, fixed lambda1/10/100 sensitivity (all shown, no reselection), independent official TEST evaluation with all methods locked, and complete cost comparisons. These are follow-up checks, not ways to rescue failed primary criteria. No broad claim of calibration dominance, state-of-the-art accuracy, compression-specific heterogeneity amplification, theoretical FL convergence, poisoning resistance, or real-chain throughput is licensed by this plan.

Use external review only if an actual separate-family backend is available; otherwise record REVIEW_UNAVAILABLE. Local independent numerical checks are a distinct source of verification.

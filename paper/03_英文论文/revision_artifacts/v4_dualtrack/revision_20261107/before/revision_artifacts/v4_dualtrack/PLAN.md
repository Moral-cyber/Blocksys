# v4 dual-track experiment plan

This experiment record implements the user's 2026-11-06 instruction. It does not revise manuscript prose. The existing Route A `COMPLETE` state is historical and does not describe this v4 work.

## Scope and fixed interpretation

- Execution order: finish E2, then immediately start E1. E3 is specified for planning only; running E3 is outside the current authorization.
- All new files live under this directory. `sections/`, `tables/`, `main.tex`, the current PDF and original evidence remain unchanged. `protected_files.json` records their initial SHA-256 identities.
- **“Greater heterogeneity produces greater gain” is an unverified hypothesis, not a conclusion.** Retain unfavorable and non-monotone outcomes. Do not alter settings after inspecting gains.
- Define delta = NLL(prior center) - NLL(zero center), so negative delta favors the prior. Define gain = -delta, so positive gain favors the prior. NLL uses natural logarithms; primary summaries weight clients equally.
- Train/fitting and assessment are separate. E2 reanalyzes previously saved TEST metrics without new inference. E1 evaluates development holdout from official TRAIN only. Original official TEST files and memberships are not reopened or repartitioned.

## E2: client skew versus existing gain (estimated 0.5 day)

Settings: use `../route_a_20261105/stats/result.json`, with its 3 x 20 per-client delta matrix for each of CIFAR-10 and FashionMNIST. Match client IDs to original development config manifests and verify equal TRAIN membership/counts across the three model seeds. Reconstruct saved deltas from the six existing final-result JSON files; this uses saved metrics, not TEST image bytes.

For local TRAIN counts n_k and N = sum n_k, use p_k = n_k/N, maximum-class fraction max(p_k), and normalized entropy H = -sum(p_k log p_k)/log(10), with 0 log 0 = 0. Larger maximum fraction and smaller entropy represent stronger skew. No smoothing is applied to these measures. Main scatter points average gains over the three seeds for each of 20 clients; show seed observations too. Report Pearson and Spearman associations, per-seed sensitivity, sample-size association, and leave-one-client-out correlation ranges. These are conditional descriptive quantities; 60 pairs are not 60 independent training replicates.

Outputs:

- `e2_skew_gain.json`: all counts, distributions, seed-level deltas/gains, associations, input hashes, limitations.
- `e2_client_data.csv`: numeric source table for all 120 client-seed pairs.
- `figures/skew_vs_gain.pdf` and `.png`: two tasks x two skew measures, all client points, explicit gain convention.
- `e2_analyze.py`: transformation/plotting script.

Acceptance: hashes match saved inputs; exact client matching and count consistency; reconstructed deltas agree to 1e-12; all 120 points retained; finite skew and metric values; exported plot inspected. Positive association with maximum fraction and negative association with entropy would support the directional hypothesis descriptively. Mixed signs or outlier sensitivity count against a general claim. E2 cannot establish causality or generalization over new partitions.

## E1: heterogeneity sweep on development data (estimated 2–3 days)

Settings: CIFAR-10 and FashionMNIST; alpha in {0.1, 0.5, 1.0}; 20 clients; 20 rounds; 3 model seeds per task (CIFAR: 20261231/32/33; Fashion: 20261241/42/43). Fixed partition seed per task (CIFAR: 20261217; Fashion: 20261219), reused across alpha. This is one partition per alpha and three training seeds, not three independent partition replications. Run all 18 models freshly within v4.

Use the frozen official-TRAIN loader and original Dirichlet partition algorithm with the alpha argument changed. At alpha=0.1 verify exact original train/holdout/validation memberships. Other settings stay fixed: no mixing, original 75% client TRAIN and 25% client development-holdout split, stratified 2,000-row development validation drawn from holdout as in the existing algorithm; no validation rows in reported client holdouts. Check disjointness and full coverage. Record realized skew and client sizes for every partition.

Use frozen FedLC-inspired local loss plus uniform three-bit encoding/error feedback, batch 64, SGD lr 0.03, one local epoch, full participation, original SmallCNN with 136,874 parameters. Use MPS without silent fallback. Use final round 20, with no checkpoint/alpha/lambda selection based on reported evaluation.

Freeze the shared model. Fit zero-centered and TRAIN-prior-centered 20-FP32-parameter local affine predictors on each client's TRAIN logits/labels only, with the unchanged vector solver and lambda=10 for both scale and bias. Compare both fits on that client's disjoint development holdout. Save the logits, labels, original row IDs, fitted states and solver diagnostics. No original TEST image/label bytes are accessed; enforce an audit guard.

Outputs:

- `e1_run.py`, `e1_aggregate.py`: execution and aggregation scripts.
- `e1_runs/<task>/alpha_<value>/s<seed>/...`: signed configs, round checkpoints, model, TRAIN/development arrays, fits, per-client metrics, timing and provenance.
- `e1_alpha_sweep.json`: all 18 results and per-client outcomes, mean/SD across the three model-seed macro gains, actual skew, and hypothesis assessment.
- `figures/alpha_vs_gain.pdf` and `.png`: seed-level macro gains, means and sample SD (not confidence intervals), exact alpha values and explicit sign convention.
- `STATUS.json`: v4 progress only; failures remain visible.

Acceptance: all 18 runs reach round 20; 20 clients/run; finite metrics and solver diagnostics; exact TRAIN/development separation; alpha=0.1 original membership matches; zero official TEST reads; protected file hashes unchanged. Numerical optimization must finish without unresolved solver failures; preserve FP32 deployment states and FP64 scoring semantics.

Predeclared assessment: report positive mean gain separately at every task/alpha. The stronger-heterogeneity hypothesis receives *descriptive support in this sweep* only if realized heterogeneity follows alpha order and mean gain(0.1) > mean gain(0.5) > mean gain(1.0) for both tasks; also report whether ordering holds within each training seed and how large seed spread is. Three model seeds and one partition per alpha cannot establish a universal monotonic law. If results are mixed or non-monotone, record that result and limit the claim to the tested conditions; do not rerun with selected seeds to manufacture the order.

## E3: ledger scale and cost (plan only; estimated 1–2 days)

Taskbook setting: compare 2 clients x 2 rounds with 5 clients x 3 rounds under the same compressed-message/receipt-gated EF protocol. Measure per-registration and total gas, receipt verification wall time, on/off-chain bytes, successful incorporation, duplicate/retry handling, and EF commit correctness. Keep actual transaction/receipt records and environment identifiers. Exact software/chain configuration and repeat count must be frozen before any future run.

Reserved outputs: `e3_chain_cost.json`, `e3_receipts.jsonl`, `figures/chain_scale_cost.pdf`, `e3_cost_table.csv`.

Acceptance if later authorized: every expected transaction and incorporation is accounted for; bytes/gas are computed from actual records; timings name measurement boundaries and repetitions; no lost/duplicate update changes EF twice. Only two scales provide a limited cost comparison, not proof of asymptotic linear scalability. E3 is `NOT_STARTED_NOT_AUTHORIZED` in this phase and does not gate the FL experiment results.

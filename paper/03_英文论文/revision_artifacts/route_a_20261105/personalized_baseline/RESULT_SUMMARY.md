# pFedMe: completed matched-protocol comparison

Independent saved-result aggregation by `/root/route_a_stats`, 2026-11-05 15:56 UTC. No training, fitting, model inference, raw TEST loading or source changes were performed for this summary.

All six fixed pFedMe task/seed runs completed twenty full-participation rounds. `MODEL_LOCK.json` freezes 43 inputs; this summarizer independently re-hashed all 43 and found no changes. Both official-TEST aggregate reports are `DONE`, keep raw final personalized theta predictions and record zero TEST fits. The original TRAIN-count-defined TEST assignments are unchanged.

`INDEPENDENT_SCORE_VERIFICATION.json` reports PASS for 1,060 exported-logit arithmetic/artifact comparisons, maximum absolute discrepancy 1.4779288903810084e−12. This is a local independent scorer replay, not external model-family review. Recomputing all task-level means and sample SDs from the six seed JSON reports gives zero discrepancy against the official aggregates and agrees with that verification within 1e−12.

## Compact paper row

| Predictor | CIFAR-10 macro NLL | CIFAR-10 macro accuracy (%) | FashionMNIST macro NLL | FashionMNIST macro accuracy (%) |
|---|---:|---:|---:|---:|
| pFedMe, raw final theta | 1.0072 ± 0.0291 | 64.38 ± 0.69 | 0.3557 ± 0.0359 | 88.86 ± 1.33 |

Mean ± sample SD is over three training seeds on the original fixed partition per task. `table_row_suggestion.tex` gives this row in the NLL/accuracy order used by the current compact baseline table. `result_all_metrics.csv` preserves every seed, mean and sample SD for all ten reported metrics. Its accuracy, q10 and ECE values use raw proportions; multiply by 100 to display percentage or percentage-point units.

Additional three-seed endpoints, in their raw units:

| Task | Weighted NLL | Weighted accuracy | Macro Brier | Macro ECE15 | Pooled ECE15 | Accuracy q10 |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 0.962986 ± 0.033950 | 0.640667 ± 0.018231 | 0.489397 ± 0.013171 | 0.147958 ± 0.023162 | 0.051388 ± 0.015025 | 0.440956 ± 0.060589 |
| FashionMNIST | 0.263066 ± 0.021252 | 0.916967 ± 0.006466 | 0.170720 ± 0.016760 | 0.069315 ± 0.004926 | 0.023813 ± 0.006496 | 0.766849 ± 0.053641 |

Suggested caption/implementation sentence:

> pFedMe reuses the author's optimizer with a matched task adapter and predicts with the final personalized theta; it uses dense communication and no post-hoc fit. The comparison fixes twenty rounds, one local data pass, the original partitions/backbone/seeds and the common TEST scorer.

Suggested concise result sentence:

> Under this fixed protocol, raw personalized pFedMe reaches macro NLL 1.007/0.356 and accuracy 64.38%/88.86% on CIFAR-10/FashionMNIST.

The PR-LFL protected macro NLL/accuracy values are lower/higher than this configured comparator. This supports the evaluated compact-prediction operating point, not a state-of-the-art claim or a general ranking of pFedMe across its published settings. The matched zero-center comparison remains the evidence that isolates the added information of the prior center.

## Protocol identity and fair interpretation

- **Actual code reuse:** the unmodified author `pFedMeOptimizer` from `CharlieDinh/pFedMe` commit `caf24b1f954d4381bd9b4d104ee7eff389e1489b`. Dataset preparation, SmallCNN, local scheduling, checkpointing, MPS execution, independent theta export and common scoring use a new adapter. Label the row **“pFedMe, author optimizer with matched protocol adapter”**, not “full author-code reproduction.”
- **Explicit hyperparameter adaptation:** K=5, lambda=15, beta=1, personal learning rate=.09 and optimizer mu=.001 follow the author defaults. The **outer learning rate changes from the author CLI default .005 to .03** to match the manuscript SGD protocol. This choice was fixed before training and not selected from TEST; it is nevertheless an algorithm-affecting adaptation. The observed gap does not establish pFedMe's best attainable performance.
- **Equal data/round schedule, unequal computation:** one complete local data pass uses ceil(n_i/64) outer minibatch steps. Each step makes K=5 personalized CE gradients before its outer update. Saved logs satisfy `gradient_evaluations = 5 × outer_steps` in every round: 59,500 gradients/11,900 outer steps per CIFAR run and 71,200/14,240 per Fashion run. This is exactly five times the CE gradient evaluations per minibatch; it is not a claim of exactly five times wall-clock runtime. Six training logs sum to 2,285.75 seconds of recorded round time on the actual host execution.
- **Prediction semantics:** the primary row is the raw final personalized inner theta for each client, cloned before overwriting training tensors with outer weights. No zero/prior calibrator is added. This preserves a genuine training-time personalization comparator, but does not compare both algorithms with identical post-hoc prediction transformations.
- **Communication/state:** this arm has analytical dense uploads and downlinks of 547,560 B each. A personalized theta has 136,874 FP32 weights, 547,496 B. These are full-model prediction parameters; PR-LFL's 80 B is an additional local predictor on its shared CNN, not the entire model memory. Do not assign uniform3's 54,067 B upload to the dense pFedMe row or claim an equal-communication comparison.
- **Scope of fairness:** task, partitions, initialization seed identifiers, backbone, local data passes, participation, final-round rule and scorer are matched. Algorithm-specific optimization and the raw prediction rule differ as disclosed. No source reimplementation or frozen hyperparameter was repaired after seeing these outcomes; no extra experiment is requested to reinterpret the fixed comparison.

## Saved outputs

- `compact_summary.json`: checked aggregate metrics, source hashes, per-run measured gradient counts/runtime and replay scope.
- `result_all_metrics.csv`: six seed rows plus mean/sample-SD rows per task, ten metrics each.
- `table_row_suggestion.tex`: proposed single row, not a manuscript edit.
- `summarize_saved_results.py`: deterministic aggregation only; validates the 43-input model lock and final score reports without loading CNNs or raw TEST data.
- Numerical sources: `official_test/{cifar,fashion}/result.json`, six per-seed score reports, six `runs/.../result.json`, `MODEL_LOCK.json`, and `INDEPENDENT_SCORE_VERIFICATION.json`.

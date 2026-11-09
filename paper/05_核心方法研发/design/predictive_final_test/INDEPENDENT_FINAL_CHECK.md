# Independent final exported-array check

PASS: cifar, 9/9 completed models.

Checks: `{'client_files': 180, 'state_arrays': 1440, 'variant_score_arrays': 2160, 'client_metric_values': 12960, 'ece_bin_values': 97200, 'summary_values': 1296, 'three_seed_aggregate_values': 864, 'paired_seed_values': 264}`.

Maximum absolute errors: `{'transform': 0.0, 'metric': 1.5649699418307517e-14, 'bins': 1.375610736431554e-11, 'summary': 2.324854718460667e-15, 'aggregate': 2.3247462982434186e-15}`.

All twelve FP64 score transforms reproduce exactly from exported raw/head logits and prelocked FP32 TRAIN states. Metrics, ECE bins, macro/sample-weighted/pooled/q10 summaries and available three-seed aggregates were independently recomputed.

Original IDs cover0..9999 exactly once; per-class1000, deterministic TRAIN-count quotas/RNG assignment, GT original-order hash, locked model/state/source hashes and strict-loader provenance records agree. Official TEST raw files were not reopened. Raw network inference and extraction were not independently rerun.

Local numerical/provenance verification only; external review REVIEW_UNAVAILABLE. No training, fit, backward or prediction inference was performed.

# pFedMe final scoring and independent verification

Preparation only. This document does not start MPS, fit a model, read TEST, or
authorize a parameter change. The fixed model is each client's raw final `theta`,
without a post-hoc predictor. Root controls when scoring starts.

## Required boundary before scoring

1. `run_matrix.py --execute` must finish all six fixed 20-round runs and call the
   unchanged adapter's `--stage model-lock`.
2. `MODEL_LOCK.json` must have status `ALL_SIX_PERSONAL_MODELS_FROZEN`. Verify all
   SHA-256 entries, including the six checkpoint/config/result triplets and 25
   frozen inputs. Do not alter the adapter, parameters, seeds or lock afterwards.
3. Preserve `matrix_execution/run_*.json` as actual execution receipts. A training
   `result.json` is not an official TEST score.

Exact scoring command (the fixed adapter uses CPU inference; no MPS dispatch):

```sh
/opt/anaconda3/envs/Tinyproto/bin/python \
  /private/tmp/pr_lfl_route_a/03_英文论文/revision_artifacts/route_a_20261105/personalized_baseline/adapter.py \
  --stage score
```

That command reads the official TEST inputs through the existing root-lock data
loader, reproduces the original task assignment hash, scores all 20 raw client
models for each fixed seed, and generates:

```text
official_test/cifar/pfedme_dense_s20261231/result.json
official_test/cifar/pfedme_dense_s20261232/result.json
official_test/cifar/pfedme_dense_s20261233/result.json
official_test/cifar/result.json
official_test/fashion/pfedme_dense_s20261241/result.json
official_test/fashion/pfedme_dense_s20261242/result.json
official_test/fashion/pfedme_dense_s20261243/result.json
official_test/fashion/result.json
```

Each seed directory also has `client0.npz` through `client19.npz`, containing
original row IDs, labels and raw-theta logits. The task report is an unweighted
mean and sample SD (`ddof=1`) across the **three training seeds**. Per-seed
client-macro NLL is the unweighted average of 20 client NLLs; sample-weighted NLL
uses client TEST sizes. The same distinction applies to accuracy, Brier and
client ECE. Pooled ECE sums the confidence-bin masses first and is a separate
quantity. Client-accuracy q10 uses the standard linear quantile. Do not describe
60 client/seed rows as 60 independent training replications.

If scoring is interrupted, retain the output and diagnose before restarting:
the frozen adapter skips complete task reports, but a partial task's existing
seed directory can cause `mkdir` to fail. Do not patch the locked adapter or
silently delete partial evidence. Root should preserve the partial output under
a dated sibling snapshot before an explicitly controlled replay, or inspect the
uncompleted task in a separate verification plan.

## Independent verification

After scoring, execute:

```sh
/opt/anaconda3/envs/Tinyproto/bin/python \
  /private/tmp/pr_lfl_route_a/03_英文论文/revision_artifacts/route_a_20261105/personalized_baseline/verify_scores.py
```

This independent read-only numerical replay uses only the newly exported client
NPZ/logit artifacts, reports and model lock; it does not load checkpoints, perform
CNN inference, read raw TEST datasets or call the production scorer. It verifies:

- All model-lock input SHA-256 values remain unchanged after scoring.
- Exactly 20 ordered clients per seed, 10,000 unique row IDs forming `0..9999`,
  finite ten-class logits, and identical row memberships/labels across seeds.
- NLL via stable logsumexp, accuracy, multiclass Brier and 15 equal-width
  top-confidence ECE from the exported logits, versus every per-client report.
- Per-seed client-macro/sample-weighted summaries, q10 and pooled ECE.
- Three-seed task means and sample SD, exact task seed lists and assignment hash,
  and `TEST_fits=0`, `TEST_backwards=0` in seed reports.

It writes `INDEPENDENT_SCORE_VERIFICATION.json` with differences and evidence
hashes. A `PASS` establishes arithmetic/artifact consistency of the new pFedMe
results, not semantic equivalence to the full original author training setup.
Results remain labeled author-optimizer reuse with the declared matched-protocol
adapter. Root then compares them with the manuscript's frozen existing rows;
the original 13 numeric groups and old evidence are not overwritten.

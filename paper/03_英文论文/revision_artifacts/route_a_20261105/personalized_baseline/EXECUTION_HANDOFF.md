# pFedMe execution handoff

Readiness: author optimizer, adapter, CPU fixtures, exact original data memberships and source lock passed. No pFedMe model training and no pFedMe TEST inference have run yet. MPS order agreed by root: matched FedAvg/FedProx, multipartition, then this baseline.

Runtime: `/opt/anaconda3/envs/Tinyproto/bin/python`.
Entry: `personalized_baseline/adapter.py` in this revision artifact folder.

First integrity run (root launches after MPS slot is free):

```text
python adapter.py --stage train --task cifar --seed 20261231
```

Then the remaining predeclared runs, one process at a time:

```text
python adapter.py --stage train --task cifar --seed 20261232
python adapter.py --stage train --task cifar --seed 20261233
python adapter.py --stage train --task fashion --seed 20261241
python adapter.py --stage train --task fashion --seed 20261242
python adapter.py --stage train --task fashion --seed 20261243
```

After all six status DONE and round count20, freeze checkpoints before any TEST:

```text
python adapter.py --stage model-lock
python adapter.py --stage score
```

Each full run checkpoints per round and resumes by exact config/source equality. No hyperparameter choices occur. Final predictor is raw theta; no fitted calibrator is hidden in the pFedMe row. Dense upload/downlink547,560 B; 20rounds; local computation K5 times CE: CIFAR2,975 gradients/round and Fashion3,560 gradients/round on the original fixed partitions. Full local epochs and outerLR.03 differ from the author's iteration-budget/outerLR.005 defaults, as explicitly locked in PLAN/SOURCE_LEDGER. This is author-optimizer reuse with a new matched-protocol adapter, not complete unmodified reproduction of the author's 800-round experiments.

Artifacts available before training:

- `FIXTURES.json`: actual CPU PASS, explicit proximal-update maximum error0, independent theta snapshot, SmallCNN2batches×K5, sample-weighted/beta1, uniform NLL.
- `DATA_PREFLIGHT.json`: exact original TRAIN/holdout/validation IDs and label counts, both tasks, zero TEST reads.
- `SOURCE_LOCK.json`:25 source inputs frozen before training.
- `PLAN.md`, `PLAN_20261105.md`, `SOURCE_LEDGER.json`, `SOURCE_LEDGER.md`: prespecified protocol and source/adaptation boundaries.

If current shell reports MPS unavailable in sandbox, root must use its already working MPS execution route; do not silently fall back to CPU or change the locked protocol. First-run numerical failure, if any, is recorded as a failure; do not revise lambda/K based on TEST. Existing sources/evidence remain read-only.

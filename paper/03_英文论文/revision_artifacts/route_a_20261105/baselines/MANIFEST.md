# Baseline revision artifact manifest

- PLAN_20261105.md and PLAN.md: frozen matched dense FedAvg/FedProx experiment plan.
- adapter.py: new adapter; all original sources/results are read only.
- FIXTURES.json: CPU numerical/data-free fixture outcomes.
- SOURCE_LOCK.json: source/PLAN/fixture SHA256 frozen before training.
- runs/: new final20-round training checkpoints, configs, histories, then TRAIN-only states.
- FIT_LOCK.json: all models and TRAIN states locked before TEST.
- official_test/: fixed official TEST assignments, per-client scores and aggregate results.

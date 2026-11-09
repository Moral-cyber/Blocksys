# Personalized baseline plan: pFedMe optimizer reuse with a matched task adapter

Status: PROTOCOL_FROZEN_BEFORE_TRAINING; CPU fixtures passed; no numerical experiment results claimed.
Date: 2026-11-05. Skill applied: experiment-plan, narrowed to the user's P1 baseline task.

## Claim and scope
This baseline tests whether compact prior-centered post-hoc prediction remains useful relative to training-time personalization under the same task, partitions, backbone, initialization seeds, final-round rule and metrics. It is not needed to attribute the prior-center mechanism; the matched zero-center and class-permutation controls do that. Results may favor either method. No state-of-the-art claim or no-cost equivalence is predeclared.

The implementation directly imports the unmodified pFedMeOptimizer from CharlieDinh/pFedMe commit caf24b1f954d4381bd9b4d104ee7eff389e1489b. Dataset, SmallCNN, local-epoch/batch scheduling, MPS routing, checkpointing, theta export and scoring use a new adapter. Describe it as “pFedMe, author optimizer with a matched protocol adapter”, not a complete unmodified author-code reproduction.

## Frozen run matrix
- CIFAR-10: original partition 20261217; training seeds 20261231/20261232/20261233.
- FashionMNIST: original partition 20261219; training seeds 20261241/20261242/20261243.
- 20 full-participation communication rounds; same 136,874-parameter SmallCNN and preprocessing; no augmentation, momentum or learning-rate scheduling.
- Each round starts from the current server model. Each client's local outer model and inner theta are initialized from it, matching the author's send_parameters. One complete local epoch means ceil(n_i/64) minibatches; batches shuffle using seed+100*round+client, as in the manuscript controls.
- Every minibatch makes K=5 personalized inner steps on ordinary mean CE, calling the unmodified author optimizer, then one local outer step w_i <- w_i - lambda*outer_lr*(w_i-theta). K=5 means five times the CE gradient evaluations per minibatch; report this extra computation.
- Algorithm parameters: lambda=15, beta=1, personal_lr=0.09, optimizer mu=0.001. These match author CLI/optimizer defaults. outer_lr=0.03 matches this manuscript's SGD protocol (author CLI default is 0.005); this is an explicit adaptation, not tuned. The root accepted this explicit choice before SOURCE_LOCK.
- Aggregation weights are exact TRAIN sample fractions, as in author persionalized_aggregate_parameters; beta=1 gives the full weighted aggregate. Uploads/downlinks are dense, analytically 547,560 B per message. No compressed-pFedMe result or 54,067 B accounting will be assigned to this arm.
- Snapshot theta with independent cloned tensors before overwriting the trainable model with local outer weights. Author persionalized_model_bar stores Parameter aliases and can be overwritten by update_parameters; this adapter corrects the export semantics explicitly. Preserve final raw theta predictions as pFedMe; no calibrator is included in its primary row.

## Evaluation and guardrails
Freeze plan/source identities before any training. First pass CPU algebra/aliasing and scoring fixtures, then train only CIFAR seed 20261231 as an integrity/run-cost check after the shared MPS queue is free. Continue all six runs if finite and protocol-matched; never select seeds or parameters from TEST.
TRAIN-only training installs a file-read guard rejecting official TEST paths. Compare every train/holdout index, client count and validation ID to the original manifest. Checkpoint after every complete round, saving server and all twenty theta states. Record batches, K, gradient evaluations, dense bytes, backend, runtime, source/data hashes and unchanged global anchor during client processing.
Only after all six fixed final-round checkpoints are frozen, create a single model lock. Then load official TEST once per task, use the exact existing TRAIN-count-defined client assignment and existing scorer metrics: client-macro/weighted accuracy, NLL, Brier, client/pooled 15-bin ECE, q10 accuracy. Report mean and sample SD over the three predeclared seeds. TEST never fits or chooses anything.

## Stop/go gates
1. Author optimizer algebra fixture equals the explicit CE + lambda/2||theta-w||^2 + mu/2||theta||^2 update.
2. Theta snapshot remains unchanged when the local outer/model tensors mutate; explicit outer beta/sample-weighted arithmetic passes.
3. Original data memberships match byte/index-exactly, source hashes are frozen, final raw-theta scoring has no post-hoc fit.
4. First CIFAR run remains finite for all 20 rounds. A numerical failure is reported without changing lambda, K or seed in response to TEST.

## Remaining work and costs
Implementation: one compact wrapper, CPU fixtures, source lock, one integrity run, five remaining runs, final frozen scoring. K=5 makes local backprop about five times a CE baseline; actual MPS round costs will be measured on the first run instead of inventing an ETA. No new environment or packages will be installed. Existing source/evidence files remain read-only. Unfinished work is recorded in checks/REVISION_STATUS.md by the root, not in manuscript experimental conclusions.

# Core-contribution mechanism diagnostic

Declared 4 November 2026, before computing the new class-permuted-anchor outcomes. This is an exploratory diagnosis on previously inspected development data, not a new prospective final test or a replacement for earlier STOP decisions.

The working contribution is local prior-centered regularization for personalized probability prediction after low-bit shared training. The question here is whether the class correspondence of the prior matters beyond a nonzero regularization center.

- Use only the saved TRAIN and TRAIN-internal holdout arrays of `fedlc_uniform3`, CIFAR partition20261217/seeds20261231–33 and Fashion partition20261219/seeds20261241–43, all20 clients. No model selection.
- Reuse the existing correct-prior and zero-prior FP32 twenty-dimensional states unchanged.
- Fit exactly one new control per client: `np.roll(center(log(TRAIN_counts+1)), 1)`. This preserves the anchor's value multiset and Euclidean norm while changing class correspondence. Do not search permutations.
- Use the unchanged vector solver, ridge10, original Newton/Armijo budget, and FP32 deployed states applied to logits in FP64. Only TRAIN logits and labels enter fitting. Holdout labels enter scoring after fitting.
- Primary descriptive contrast: macro holdout NLL of correct vs permuted anchor for each original training seed. Also report correct vs zero, accuracy, TRAIN NLL, and probability mass on classes absent from TRAIN. Report all outcomes, including adverse ones.
- Interpret a consistent correct-anchor advantage as evidence about class-aligned information in this fixed objective. A contrary result narrows this explanation. No new acceptance threshold, statistical independence claim for clients, or TEST retuning follows.
- No new CNN forward/backward, original images, official TEST, new training seed, network measurement or ledger execution. Save one aggregate JSON, including the120 new states and per-client outcomes, rather than many NPZ files.

This diagnostic supplements the existing fixed two-task results. The core decision will combine mechanism evidence, existing direct controls and accurate comparison with prior calibration work. Portable reproduction and external review are not prerequisites.

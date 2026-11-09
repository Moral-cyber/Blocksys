# Completed dense baseline results

All 12 fixed MPS training runs, 480 TRAIN-only vector fits, and fixed official TEST evaluations completed. The original 13 numeric groups and old evidence were not changed. The baseline labels are **matched in-house implementations**, not author-code reproductions. Each raw predictor is the basic algorithm; zero/prior center adds this paper's post-hoc calibrator.

## Suggested main-text sentence

The same prior-center effect appears with dense FedAvg and FedProx training: compared with their zero-centered post-hoc vectors, macro NLL decreases by 0.0353/0.0358 on CIFAR-10 and 0.0186/0.0185 on FashionMNIST (Appendix Table~\ref{tab:dense_baselines}).

## Suggested appendix paragraph

We additionally run matched dense FedAvg and FedProx with $\mu=0.1$, using the same client partitions, three training seeds, SmallCNN, local SGD schedule, and final-round evaluation as the main study. The raw rows report the base algorithms; the remaining rows apply the separately TRAIN-fitted zero- and prior-centered vectors with the original coefficient 10. Prior centering lowers macro NLL in every training seed for both base algorithms and tasks, with all 60 client--seed differences negative within each task--algorithm setting. The accuracy effect remains small and mixed: on CIFAR-10, prior centering changes macro accuracy by $-0.043$ and $-0.076$ percentage points for FedAvg and FedProx, respectively; on FashionMNIST the changes are $+0.241$ and $+0.222$ points. FedProx's pooled CIFAR-10 ECE increases by 0.0006 despite a lower client-macro ECE. These controls support the NLL effect across basic dense training objectives, with a distinct accuracy and calibration profile for each metric.

## Evidence paths

- `TRAINING_ACCEPTANCE.json`: 12 complete final20 runs; exact client counts, seed formula, batch counts and input hashes.
- `FIT_LOCK.json`: 290 exact model/config/state/source inputs frozen before new TEST access.
- `official_test/{cifar,fashion}/`: all three seeds, both base algorithms, all three predictor variants, every client and raw saved score/GT.
- `summary.json`: all macro/sample-weighted metrics, sample SD, accuracy q10, pooled ECE; unrounded values.
- `per_client_seed_metrics.csv`: all 720 client--seed--variant rows and four metrics.
- `paired_outcomes.json`: all seed-level accuracy changes and mean metric differences, including adverse outcomes.
- `VERIFICATION.json`: replay of 240 saved client exports; metric maximum absolute error 0, original TEST assignment IDs equal, FP32 deployment arithmetic exact, all 290 frozen inputs unchanged. This is local deterministic verification and does not claim a separate-family integrity review.
- `table_compact.tex`: six rows, both tasks side by side, NLL and accuracy.
- `table_all_macro_metrics.tex`: optional 12-row alternative including Brier/ECE.

The final official TEST client assignment hashes equal the original main-study hashes: CIFAR `08d8954ee4da54785f801a4310b3d39f4999bb879c2b74958794b817df1a215f`, Fashion `844576190704f1d995daecee886d896c7225d4d2f46adf818d45e1c65241cecf`.

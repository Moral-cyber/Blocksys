# Count-geometry candidate stopped after the first-seed screen

The predeclared two-task subset completed and **failed the continuation rule**. No remaining-seed extension or official TEST evaluation is performed. These are development results from one original training seed per task and all20clients, not a general multi-seed conclusion.

| Predictor | CIFAR-10 macro NLL | FashionMNIST macro NLL |
|---|---:|---:|
| Original prior, lambda10 | 0.599279485 | 0.152988912 |
| Zero center, lambda10 | 0.638949567 | 0.166182572 |
| Count geometry | 0.598087540 | 0.153608170 |
| Cyclically misaligned geometry | 0.599026390 | 0.152570384 |
| Normalized count ridge, w=n+0.5 | 0.598005057 | 0.153622261 |
| Fixed isotropic lambda1 | 0.598109034 | 0.151196569 |

On CIFAR, geometry improves NLL over the original prior by0.001191945 and the misaligned precision by0.000938851. On Fashion it instead worsens NLL by0.000619258 versus the original prior, and by0.001037785 versus misaligned precision. Thus the proposed class-uncertainty alignment does not satisfy its two-task prediction. The simpler count ridge also slightly outperforms exact trigamma geometry on CIFAR; a fixed isotropic lambda1 is substantially better on Fashion. These outcomes do not justify attributing a new core contribution to Dirichlet count geometry.

The negative decision follows the locked PLAN rather than a changed threshold or chosen seed. Geometry and count ridge are numerically distinct beyond the1e-7 contrast tolerance, but the main gate fails before that distinction could support extension. We retain all six arms, every client, accuracy/Brier/ECE/weighted trade-offs, eigenvalues, fitted FP32 states and optimizer diagnostics in the result files. For example, aligned geometry also lowers macro accuracy slightly on both tasks relative to the original prior; the NLL failure is not rescued by secondary metrics.

Verification: original prior/zero states replayed bitwise for all40clients, and their four metrics matched saved development reports. New fits met the declared KKT bound; maximum reported projected gradient across all arms was7.51e-7. An independent saved-state replay recomputed240arm-client rows with maximum metric difference1.54e-15. All253locked source/input hashes remain unchanged. Audit guards recorded no official TEST or new-partition access; no CNN training, forward, backward, or communication was run. The original manuscript and its13protected numeric groups were not edited.

The20-FP32 predictor remains80B. Tracemalloc peaks for an entire new geometry fit ranged681,016–26,210,680B, including the baseline logit/design/solver workspace. This is not incremental geometry memory or total resident memory. The finite screen neither establishes a new efficiency result nor changes the accepted main method. The residual-ratio candidate was only reviewed and never executed.

Artifacts: `PLAN.md`, `SOURCE_LOCK.json`, `FIXTURES.json`, `SUBSET_RESULT.json`, `runs/{cifar,fashion}/sSEED/{result.json,client*.npz}`, `client_metrics.csv`, and `VERIFICATION.json`. No manuscript promotion is recommended.

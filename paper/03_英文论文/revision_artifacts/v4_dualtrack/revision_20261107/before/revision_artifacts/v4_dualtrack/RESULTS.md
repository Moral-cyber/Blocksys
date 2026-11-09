# v4 experiment results — authorized scope only

Completed: 2026-11-06T23:04:56.616815+08:00. PLAN, E2 and E1 are complete. E3 remains plan-only. No manuscript prose was edited.

Gain = NLL(zero) - NLL(prior), positive is beneficial. E1 uses development holdouts excluding 2,000 monitoring rows; E2 reuses existing saved TEST metrics without new inference.

## E1: all 18 fresh runs

| Task | Alpha | Mean gain ± seed sample SD | Improved client-seed pairs |
|---|---:|---:|---:|
| cifar | 0.1 | 0.04081121 ± 0.00244436 | 60/60 |
| cifar | 0.5 | 0.01012792 ± 0.00015187 | 55/60 |
| cifar | 1.0 | 0.00420221 ± 0.00023688 | 45/60 |
| fashion | 0.1 | 0.01319697 ± 0.00070030 | 59/60 |
| fashion | 0.5 | 0.00471553 ± 0.00025486 | 57/60 |
| fashion | 1.0 | 0.00242967 ± 0.00016625 | 42/60 |

Both tasks and each of the three model seeds satisfy G(0.1)>G(0.5)>G(1.0); realized maximum-class fraction and normalized entropy confirm the nominal skew order. This is descriptive support within one partition per alpha. Alpha also changes client sizes, membership and trained-model quality. It is not evidence for a universal causal monotonic relationship.

## E2: saved-outcome association

| Task | Mean gain | Spearman: max-class fraction | Spearman: entropy | Spearman: TRAIN rows |
|---|---:|---:|---:|---:|
| cifar | 0.03575086 | -0.180 | +0.057 | -0.771 |
| fashion | 0.01823567 | +0.269 | -0.304 | -0.729 |

E2 does not show a consistent cross-task client-level skew association. E1 varies alpha across trained systems; E2 compares clients inside a fixed alpha=0.1 system. These are different comparisons. A monotonic E1 curve does not turn E2 into causal support.

## Observed but unresolved

- The frozen objective is sum(NLL) + 10/2 times squared distance to the anchor. The effective coefficient per example is 10/n. Sample size may partly explain the strong E2 association; it has not been isolated experimentally. Lambda was not changed.
- Mean gains stay positive at all tested alphas, while some client-seed pairs regress. Fashion alpha=0.1 has 59/60 improving pairs, correcting the earlier preliminary 60/60 progress message.
- E1 development and E2 saved TEST values use different assessment samples. Directly comparing their absolute gain values does not isolate training instability.
- E1 uses one partition realization per alpha and three model seeds; it does not estimate uncertainty over new partitions.

## Verification

Independent replay passed all 18 runs, 360 scored client-seed pairs and 720 deployed states. NLL/accuracy maximum absolute difference is 4.44e-16. Solver fits resolved without maximum-step failures. Original official TEST was neither reopened nor repartitioned. All 60 protected paper/evidence files retain their initial hashes.

Files: `e1_alpha_sweep.json`, `figures/alpha_vs_gain.pdf`, `e2_skew_gain.json`, `figures/skew_vs_gain.pdf`, `e1_independent_audit.json`, `e2_independent_audit.json`, `FINAL_CHECK.json`. Old Route A completion status has not been adopted or edited.

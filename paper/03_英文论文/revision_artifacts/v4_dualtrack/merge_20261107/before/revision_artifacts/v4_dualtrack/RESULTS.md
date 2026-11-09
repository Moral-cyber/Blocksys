# v4 dual-track results — current delivery, 2026-11-07

Completed in this pass: the user-confirmed claim lock, redrawn Figures 1–2, measured E3, Table 3 / Figure 4, and Section 6–7 sources with a numbered PDF preview. The original `main.tex` and 12-page `main.pdf` remain unchanged; this pass prepares the system half for review before a full-manuscript merge. The old Route A COMPLETE status describes an earlier scope.

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

The locked claim is: **在本次三点扫描中，异构越强平均收益越大。** Both tasks and each of the three model seeds satisfy G(0.1)>G(0.5)>G(1.0). The qualifier applies whenever the claim is used. Each alpha has one partition realization; alpha also changes client sizes, membership and trained-model quality. Manuscript improvement counts refer explicitly to alpha=0.1: CIFAR 60/60 and Fashion 59/60. Other-alpha counts remain in the evidence table above.

## E2: client TRAIN sample count versus gain

| Task | Mean gain | Spearman rho: TRAIN rows versus gain | Unit |
|---|---:|---:|---|
| cifar | 0.03575086 | -0.7714285714 | 20 client means, 3 seeds each |
| fashion | 0.01823567 | -0.7293233083 | 20 client means, 3 seeds each |

Figure 2 uses logarithmic TRAIN sample count on the horizontal axis. Historical skew/gain correlations stay in the original artifacts and outside manuscript prose and conclusions. E1 compares trained systems across alpha; E2 compares clients within a fixed alpha=0.1 system. The sample-count result is an association.

## E3: measured anchored uploads

Each configuration uses a fresh process and local EVM chain with the final locked instrumentation. The five-client activity trains original CIFAR TRAIN clients 0–4 for three rounds; the two-client activity supplies a matched measurement reference. Ganache 7.9.2, Solidity 0.8.30 and ethers 6.15.0 were used.

| Metric | 2 clients × 2 rounds | 5 clients × 3 rounds |
|---|---:|---:|
| Fresh registrations | 4 | 15 |
| Registration gas, total | 1,140,380 | 4,229,478 |
| Registration gas / upload, mean | 285,095.0 | 281,965.2 |
| Contract deployment gas, separately | 530,120 | 601,869 |
| Local incorporation-receipt verification, mean (ms) | 0.457459 | 0.380150 |
| Ledger readback roundtrip, mean (ms) | 17.911622 | 15.572714 |
| Codec payload / fresh upload (B) | 54,067 | 54,067 |
| Authenticated message / fresh upload (B) | 54,408 | 54,408 |
| Total fresh application message bytes | 217,632 | 816,120 |

The local verification sample is the first successful commit-path INCORPORATED receipt verification per client–round. It includes JSON, HMAC-SM3, session/challenge, context and embedded chain-field checks, and excludes HTTP/RPC, waiting and SQLite commitment. Ledger retrieval is measured separately. Gas records executed registration work; deployment is a separate quantity.

All 15 five-client records have successful canonical registration evidence and authenticated incorporation receipts. Protocol/direct models and error-feedback states match bit for bit after each of the three rounds. Four recovery classes pass in both activities. Exact recovery adds zero encoding calls, registration requests or model contributions.

Outputs: `e3_chain_cost.json`, `e3_receipts.jsonl` (19 records across both configurations), `e3_cost_table.csv`, `tables/table3_chain_cost.tex`, `figures/figure4_chain_cost.pdf`, and final raw results `e3/c2_r2_measured/result.json` / `e3/c5_r3_measured/result.json`. Earlier instrumentation attempts are excluded from the measured comparison.

## Section 6–7 draft

Sources: `../../sections/anchored_uploads.tex` and `../../sections/upload_evaluation.tex`. The draft connects encoded updates, ledger binding, incorporation receipts and residual commitment, then presents measured messages, gas, verification and recovery. Figures 3–4 and Tables 2–4 are included in `system_preview.pdf` (six pages including references). New prose and table captions pass the user-specified banned-term scan; paragraph endings were reviewed for mechanism/result progression. The preview compiles with zero overfull boxes and no undefined references.

## Observed but unresolved

- The frozen objective is sum(NLL) + 10/2 times squared distance to the anchor. The effective coefficient per example is 10/n. Sample size may partly explain the strong E2 association; it has not been isolated experimentally. Lambda was not changed.
- Mean gains stay positive at all tested alphas, while some client-seed pairs regress. Fashion alpha=0.1 has 59/60 improving pairs, correcting the earlier preliminary 60/60 progress message.
- E1 development and E2 saved TEST values use different assessment samples. Directly comparing their absolute gain values does not isolate training instability.
- E1 uses one partition realization per alpha and three model seeds; it does not estimate uncertainty over new partitions.
- E3 verification is slightly faster in the larger measured activity. Each configuration has one execution, with shared host/process conditions, so a causal speedup claim remains unsupported.
- The two E3 configurations change client count and round count together. Their total costs describe the measured workloads rather than a general linear-scaling law.

## Verification

The previous independent replay passed all 18 E1 runs, 360 scored client-seed pairs and 720 deployed states, with NLL/accuracy maximum absolute difference 4.44e-16. That replay was retained rather than repeated. Current checks verify the archived historical PLAN identity, original E1/E2 hashes, new figure hashes, final E3 source/runtime identities, all 19 receipt bindings, raw totals, protocol equivalence and compiled section sources. All 59 pre-existing substantive paper/evidence files retain their initial hashes; operating-system `.DS_Store` metadata is excluded. Original official TEST was neither reopened nor repartitioned for E1/E3.

Current acceptance record: `revision_20261107/FINAL_CHECK.json`. Current status: `STATUS.json`. The older root `FINAL_CHECK.json` and historical E1/E2 locks remain unchanged as records of their original scope. The pre-amendment PLAN is preserved under `revision_20261107/before/` so its historical hash remains verifiable.

# 2026-11-07 delivery — the three authorized items

## 1. Claim lock and figures — complete

- Active plan: `../PLAN.md`.
- Figure 1: `../figures/figure1_delta_nll_vs_alpha.pdf`.
- Figure 2: `../figures/figure2_delta_nll_vs_samples.pdf`; logarithmic x.
- E1 mean gains at alpha 0.1 / 0.5 / 1.0: CIFAR 0.04081121 / 0.01012792 / 0.00420221; Fashion 0.01319697 / 0.00471553 / 0.00242967.
- E2 TRAIN sample-count versus gain rho: −0.7714285714 / −0.7293233083, over 20 client means per task.
- Locked claim: 在本次三点扫描中，异构越强平均收益越大。Improvement counts refer explicitly to E1 alpha=0.1: CIFAR 60/60, Fashion 59/60. Historical skew/gain correlations stay out of manuscript prose.
- Uncertain sentence: “Smaller client datasets cause larger prediction gains.” Size is associated with gain; fixed summed-NLL regularization may contribute, and causality has not been isolated.

## 2. E3 — complete

- Final measurement sources/results: `../e3/`; final settings are `c2_r2_measured` and `c5_r3_measured`.
- Aggregate: `../e3_chain_cost.json`; public receipt evidence: `../e3_receipts.jsonl` (4 + 15 records).
- Table 3: `../tables/table3_chain_cost.tex`; Figure 4: `../figures/figure4_chain_cost.pdf`.
- Five clients × three rounds: 15 registrations, 4,229,478 registration gas, 281,965.2 mean gas per upload; 0.380149933 ms mean local receipt verification; 54,408 B per authenticated message and 816,120 B across fresh uploads.
- Each of three protocol models and committed residual sets matches the corresponding direct path bit for bit. All four recovery classes pass, with once-per-record contribution and residual-generation changes.
- Uncertain sentence: “Total cost scales linearly with client count.” The two measured configurations also differ in round count and provide no general scaling estimate.

## 3. Sections 6–7 — draft complete with measured E3

- Section 6: `../../../sections/anchored_uploads.tex`.
- Section 7: `../../../sections/upload_evaluation.tex`.
- Numbered review preview: `../system_preview.pdf` and `../system_preview.tex` (six pages including references).
- Supporting Figure 3 / Tables 2 and 4: `../figures/figure3_upload_protocol.pdf`, `../tables/table2_upload_fields.tex`, `../tables/table4_upload_recovery.tex`.
- New section prose and captions pass the banned-term check. Paragraph ending review is recorded in `paragraph_endings.md`; compilation reports zero overfull boxes and no undefined references.
- Uncertain sentence: “Receipt verification becomes faster with more clients.” The observed 0.380 ms versus 0.457 ms comparison has one execution per setting; this causal wording was excluded from the manuscript.
- This delivery adds section sources and a separate preview. The original 12-page `main.pdf` and `main.tex` retain their prior hashes; Sections 1–5 and the full-paper integration have not been revised in this pass.

Acceptance: `FINAL_CHECK.json`, produced by `verify_delivery.py`. Active v4 state is recorded in `../STATUS.json`; old Route A COMPLETE remains historical.

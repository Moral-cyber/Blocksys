# v4 dual-track plan — interpretation locked 2026-11-07

This revision implements the user's confirmed E1/E2 interpretation, authorizes E3, and starts manuscript Sections 6–7. Completed E1/E2 raw results remain unchanged. Historical Route A COMPLETE describes that earlier scope.

## Locked claims and sign convention

1. E1 claim: **“在本次三点扫描中，异构越强平均收益越大”** — in this three-point sweep, stronger heterogeneity accompanies larger mean gain. Retain the three-point-sweep qualifier every time; never turn it into a general conclusion or causal law. Each alpha has one partition realization and three model seeds.
2. E2 main result: **client TRAIN sample count versus gain**, Spearman rho **−0.771** (CIFAR-10) / **−0.729** (FashionMNIST), over20 client means per task. Skew/gain correlations remain in historical artifacts and stay out of manuscript conclusions and prose.
3. Figure Delta NLL = **NLL(zero) − NLL(prior)**, positive means improvement. Historical prior-minus-zero JSON fields keep their original sign. Raw results remain immutable.
4. Figure1 = Delta NLL vs alpha, exact x={0.1,0.5,1.0}, three seed observations and sample SD. Figure2 = Delta NLL vs n_i, logarithmic x, all seed observations and client means.
5. Manuscript improvement counts refer explicitly to **E1 alpha=0.1: CIFAR60/60, Fashion59/60**. Use exact counts and avoid universal client-improvement language. Other-alpha counts stay in artifacts.
6. No lambda/seed selection or new TEST inference. E2 reuses saved TEST metrics; E1 scores separate TRAIN-derived development data.

## E1 — completed heterogeneity sweep

CIFAR-10/FashionMNIST, alpha={0.1,0.5,1.0},20clients,20rounds,3 original seeds/task. Partition seeds C20261217/F20261219; model seeds C20261231/32/33,F20261241/42/43. All18 fresh MPS models: frozen FedLC-inspired tau1 + uniform3/EF, SmallCNN136874parameters,batch64,SGDlr0.03,one local epoch,full participation,final20. Fit local20FP32zero/prior predictors with frozen lambda10 on TRAIN logits; score disjoint development holdouts excluding2000 monitor rows. Alpha0.1 memberships match originals exactly.

Outputs retained: `e1_runs/`, `e1_alpha_sweep.json`, `E1_PREFLIGHT.json`, `E1_SOURCE_LOCK.json`, `e1_independent_audit.json`. Figure1: `figures/figure1_delta_nll_vs_alpha.pdf` (+PNG). Transformations: `redraw_claim_figures.py`, `revision_20261107/claim_figures.json`.

Acceptance met:18final20runs,360client-seed metrics independently replayed, resolved fits, no TEST reads. Both tasks and every model seed follow the three-point gain order. New partitions remain outside the measured uncertainty.

## E2 — completed sample-count association

Reuse immutable `e2_skew_gain.json`, `e2_client_data.csv`, and original TRAIN manifests. Main relationship is sample count n_i vs mean gain over3seeds. Spearman rho −0.7714285714/−0.7293233083. All120client-seed observations retained; Figure2 displays20client means/task on logarithmic n_i. No new fits or inference.

Outputs: `e2_sample_gain.json`, `figures/figure2_delta_nll_vs_samples.pdf` (+PNG). Historical skew correlations stay in artifacts and out of manuscript conclusions. Interpretation is association: summed-NLL plus fixed ridge10 gives per-example strength10/n; causality has not been isolated.

## E3 — authorized:5 clients ×3 rounds

Start after claim/figure lock. Use original CIFAR TRAIN clients0–4,init20261231,partition20261217,3 fresh CPU rounds,batch64/lr0.03/one epoch,frozen FedLC-inspired loss/uniform3 codec. Reuse existing chain_integration Solidity contract,durable protocol,HTTP service. Extend bridge account setup through isolated adapter; preserve original scripts/result.

Runtime:Ganache7.9.2,solc0.8.30,ethers6.15.0,local EVM development chain; successful canonical transaction,exact binding readback,two descendant blocks. Freeze source/runtime identities before run. Run a fresh matched2clients×2rounds reference with identical instrumentation; preserve old result as historical evidence.

Measure actual registration gas(15registrations for5×3),deployment gas separately,ledger readback/validation time,local authenticated incorporation-receipt verification time,payload/envelope+tag bytes,HTTP body/wire bytes,transaction/calldata/receipt bytes. Record timing boundaries and all samples. Chain waiting/transport/local verification use separate fields.

Keep15registration/incorporation receipts; secret keys stay in memory. Compare protocol/direct model and EF state eachround. Exact retries must add zero encoding/registration/aggregation/EF commits. Preserve fault/recovery outcomes. No TEST bytes.

Outputs:`e3_chain_cost.json`,`e3_receipts.jsonl`,`e3_cost_table.csv`,`tables/table3_chain_cost.tex`,`figures/figure4_chain_cost.pdf`(+PNG),`e3/`sources/config/rawresults/logs.

Acceptance:5×3completes,15records and15once-committed client-round EF changes,actual gas/bytes,named timing,exact bindings,protocol/direct equality,old evidence unchanged. Two scales provide measured comparisons, not an asymptotic linear-scalability conclusion.

## Sections6–7 — system draft

Section6 Anchored Uploads:causal prediction→transmission transition,message/codec,chain binding,incorporation receipt,EF commit,idempotent recovery; protocolFigure3/fieldTable2. Section7 Upload Evaluation:existing chain_integration evidence,then actual E3 numbers inTable3/Figure4.

New English prose excludes watermark/水印,does not,only,trusted,single-node and taskbook§4related defensive phrases. End each paragraph with a mechanism,result,or forward transition. Accurate measured conditions remain explicit. Original protected numerical evidence stays unchanged; this pass writes system section sources and a numbered preview for review before a full-manuscript merge.

## Historical lock preservation

Pre-amendment PLAN: `revision_20261107/before/revision_artifacts/v4_dualtrack/PLAN.md`. That exact archived copy retains the PLAN identity pinned by completed E1/E2 runs. Active PLAN changes follow user authorization; raw JSON/checkpoints and historical lockfiles stay immutable. Verify historical PLAN hashes through the archive.

## Merge and compression acceptance — 2026-11-07

User authorized mechanical integration and ordered page reduction. System sections are now in main.tex after Results and before Discussion. Legacy3.3/3.4 were already absent; duplicate appendix upload paragraph was removed. Current main.pdf is14pages including appendix/references. Reductions were applied in the requested sequence; no further section drafting follows this merge. E1 Figure1's exact requested caption is stored with its standalone figure; old front-half figures await the user's subsequent Sections4-5 instructions. Current record: merge_20261107/FINAL_CHECK.json and PAGE_LOG.json. Stop at this point.

## Front-half narrative acceptance — 2026-11-07

User-authorized Section3sentence, compactSection4assessmentclarification, Section5four-subsectionheterogeneityrestructure, replacementdual-panelFigure2arecomplete. Table1 andrawE1/E2/E3evidenceunchanged. main.pdf14pages; appendixAunchanged. Figure2graphic42.792mmhigh, bodylinecounts5.1/5.2/5.3/5.4=9/7/5/3. Currentrecord: front_half_20261107/FINAL_CHECK.json. Stopafterdelivery.

## Structure packaging acceptance — 2026-11-07

User-authorized final title/abstract/introduction/contribution/relatedwork packaging complete, with titleB,177-word abstract,fiveintroparagraphs,andFigure2development/TESTpaneltitles. Three minimal factualclarifications preserve the requested abstract narrative. Current main.pdf14pages; noappendixcutsneeded. Acceptedrecord:packaging_20261107/FINAL_CHECK.json. Workstopsafterdelivery.

## Closing revision acceptance — 2026-11-07

Requested Discussion/Conclusion replacement and Section6.1registryrationale complete. Minimalprecisionedits preserve measured authentication/context,ledger-access,local-databaseandreceiptboundaries. Activepaperhasno reversal/favor-zero-centeringphrases. main.pdf14pagesafterappendixnumericcompressionandcompactregistryparagraph. Discussion19bodylines;Conclusion8. Currentrecord:closing_20261107/FINAL_CHECK.json. Stopafterdelivery.

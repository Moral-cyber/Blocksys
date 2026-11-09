# Paper builder read-only review

2026-11-04. Reviewed `work/predictive_manuscript/build_papers.py` as source text only; did not import or execute it, create manuscript outputs, read official TEST raw files, or train/infer a model. Checked current CIFAR final reports, available reference report/schema, method fragments, and existing bibliography. Fashion final task report was absent at the initial snapshot and appeared during the final follow-up; its exported arrays are checked separately by the existing `independent_final_check.py`.

## Material findings

### Resolved — Preserve the direction of NLL differences in Results prose

The initial snapshot defined `delta` using `abs(main_minus_control_mean)`, which could render worse NLL as an improvement. The parent has corrected the generator: current line 301 uses `-d[task][variant][metric]['mean']`, retaining the control-minus-main sign. The abstract also uses signed negation. This arithmetic finding is resolved.

Current “lower/falls/reduction” prose is appropriate for the completed two-task comparisons: all six control-minus-main mean NLL differences are positive. Fashion differences are `0.018235665483165076` versus zero vector, `0.022225985804777443` versus analytic prior, and `0.003611555587884896` versus prior BCTS. No negative difference is currently described as a reduction.

The completed CIFAR data do not currently trigger reversal: mean main-minus-control NLL is `-0.03575086141101125` versus zero vector, `-0.02815209248365491` versus analytic prior, and `-0.001599750356989449` versus prior BCTS. This is therefore an identified directional reporting defect, not a claim that CIFAR's reported benefits are false. Fashion's direction must remain data-dependent when its final reports arrive.

### P2 — BCTS contrast does not uniquely identify the effect of class-specific scales

English line 314 calls the vector/BCTS difference “additional mean NLL reduction from class-specific scales”; Chinese line 348 similarly assigns it to independent class scales. The compared fits differ in scale dimension and hence the total quadratic penalty, even at the same coefficient 10; BCTS additionally constrains its scalar scale to be positive while vector scales are unrestricted. The method correctly discloses these choices. This causal phrasing overstates what the control isolates.

Use “the observed mean NLL difference relative to same-prior BCTS” (with its actual sign), rather than assigning the difference uniquely to class-specific scales. Retain the existing conclusion that their necessity is not established. No new experiment is required.

## Scope checks without further material findings

- The table and per-seed arithmetic use the correct directions: accuracy is multiplied by100; mean/sample SD use exactly3 values with ddof1; accuracy deltas are main-minus-control; the per-seed “NLL reduction” column is control-minus-main. No `abs` is used in table construction.
- For the current `METHOD_ZH.tex`, subsection index5 is the endpoints subsection. Its five blank-line blocks comprise the title, metric definition, fixed development rule, seed rule, and terminal empty block. The last-three-block split moves exactly the last two substantive paragraphs to the appendix and keeps the metric definition in Methods. The English split matches that scope. No Chinese split bug is present in the reviewed fragments.
- The reference contract matches the actual completed `reference_preflight/result.json`: `reports` is a list; `fedlc_topk` has `prioranchored_vector`; `fedrod_linear_uniform3` has `generic_plus_local_residual`; each has exactly seeds20261231/32/33. The adapter does not confuse the residual-only diagnostic with FedRoD's final prediction.
- Current TRAIN manifests confirm the stated counts: CIFAR37,500/12,500 and Fashion45,001/14,999 TRAIN/holdout. The TEST assignment explanation distinguishes TRAIN-derived P(client|class) from exact TRAIN-prior matching.
- The method formula is sum-CE plus fixed ridge10, unrestricted class scales, centered bias, smoothed TRAIN counts, and FP32 deployment-state application in FP64. The network score/argmax and probability metrics are not conflated.
- FedLC and FedRoD are disclosed implementation variants. Durable full-payload evidence is correctly distinguished from authenticated end-to-end training; spec hashes do not prove decoder execution; server/decoder remain trusted; no real chain/network claim is made.
- The 18 front-matter reference keys and six Methods keys resolve against the20-entry bibliography. The separate local reference audit retains secondary-metadata/primary-source limits and exact archived arXiv version choices. No claim of external citation verification is warranted.

## Final Fashion follow-up

Executed the existing `independent_final_check.py` once against the completed `official_fashion` exports. Result: **PASS**, all **9/9** models complete, no pending models. It verified 180 client files, 1,440 state arrays, 2,160 variant-score arrays, 12,960 client metrics, 97,200 ECE-bin values, 1,296 summary values, 864 three-seed aggregates, and 264 paired-seed values. Reconstructed score-transform error is zero; maximum metric error is `1.1337068436811926e-14`. Reports: `work/predictive_final_test/INDEPENDENT_FINAL_CHECK_FASHION.md` and `.json`.

The checker reads exported arrays and provenance records; it does not reopen official TEST raw files, fit a predictor, run model inference, or train. This is a bounded local numerical and argument/schema review. External separate-family review remains `REVIEW_UNAVAILABLE`.

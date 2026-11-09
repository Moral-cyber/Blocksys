# Writing notes and source map

2026-11-04. Method-only draft using the paper-writer skill, read from `/Users/wind/.codex/skills/paper-writer/SKILL.md` together with its section, checklist and verification-ladder guidance. No abstract, introduction, Results, superiority claim, publication action or output-bundle edit is included. Scope is the specified four new files; bibliography work belongs to the separate reference task.

## Delivery checks

- English whitespace-token count: 1576 (including equations and headings), below the 2,200-word ceiling.
- Eight English/Chinese subsections and 24 corresponding nonheading blocks; four displayed equation blocks are byte-identical across languages; citation occurrence lists match exactly.
- METHOD_EN.tex is a literal structural conversion of METHOD_EN.md; automated conversion back to Markdown reproduces the complete English file byte-for-byte. It is an include fragment with no document class/preamble/bibliography environment. Parent document requires amsmath and amssymb (or equivalent math support). No standalone TeX compilation was performed.
- Source-level fact/formula review checked TRAIN counts, all-class CE sign, per-block byte metadata, actual codec/decode aggregation, centered sum-CE/ridge10 objective, unrestricted A, FP32 state/FP64 application, fixed macro-NLL rules, three-seed unit of replication, and trusted local protocol scope.
- No original source, protocol plan, existing result, official dataset, TEST file or outputs/ deliverable was edited or opened for data extraction. File source hashes below record the current writing evidence snapshot; they do not imply this writing task reran the experiments.

## Evidence status and exclusions

The completed `work/predictive_pfl_confirm/CIFAR_THREE_SEED_SUMMARY.json` is DONE for the three CIFAR seeds on one development partition. It was inspected to confirm evidence scope; no numeric predictive outcome or final scientific conclusion is quoted in these Methods. At assignment, Fashion seeds 2/3 and the full strong-reference evidence were pending. The manuscript therefore describes their fixed procedures and attribution questions, without representing incomplete evidence as completed results. State export/official TEST evaluation and a complete cost comparison remain outside this draft. No expansion beyond the three predefined training seeds is proposed.

The frozen PLAN records a new NLL question informed by previous development results. Earlier STOP decisions remain unchanged. This provenance belongs in a study-status or limitations disclosure, not a chronological narrative inside Methods. Existing code text that carries historical gates is not a new decision rule: the primary thresholds in the draft come from the current predictive PLAN. Same-array analytic-prior prediction in those rules means raw z + log(smoothed TRAIN prior), distinct from prior-only probabilities.

The full-payload report is a completed local simulation, with two distinct real training payloads and constructed updates, not an end-to-end authenticated FL run. The draft intentionally separates synchronous training EF (committed immediately after decoding) from durable pending/receipt-dependent EF. No hash-linked public ledger, consensus, real blockchain, measured network throughput, or distributed commit is claimed. REGISTERED/ACCEPTED never authorize a client EF commit; only a matching INCORPORATED receipt does. Keys, server, SQLite authority, receipts and decoder are trusted; a key holder can authenticate arbitrary/poisoned updates.

## Claim-to-source map

| Draft material | Verified local source and locator |
|---|---|
| Primary question, exact ridge objective, frozen parameters, 3 seeds, all five continuation criteria | `work/predictive_pfl_confirm/PLAN.md`, lines 7-47; preregistration identities in `PREREGISTRATION.json` |
| Strict five-file CIFAR TRAIN read and counts; 20 clients/partition | `work/calibrated_lowbit_confirm/common.py`, `load_train_only`, `prepare`; `work/method_followup/run_partitions.py`, `partition_indices`, TRAIN_FRACTION=.75, MIN_CLIENT_SAMPLES=100 |
| Strict two-file Fashion TRAIN read and preprocessing | `work/calibrated_fashion/adapter.py`, `load_train_only`, `prepare`, `runtime_metadata`; `work/method_followup/run_fashion.py`, `partition_indices` |
| FedLC-inspired local loss CE(z-max(n,1)^(-1/4)), not official exact FedLC | `work/balanced_quant/run.py`, `shift_for`, `train`; loss counts are computed from ds.labels[ds.indices] |
| Actual FP32 EF/decoded sample-weighted aggregation; 20r/SGD.03/batch64 | `work/calibrated_lowbit_confirm/train.py`, `run` |
| Block codec byte format, FP32 scale refinements, canonical decode and 54,067 B | `work/mixed_precision/codec.py`, `wire_size`, `_quantize`, `decode`, `encode`; 64 header + 535*5 metadata + 51,328 values |
| Main convex affine fit, Newton/Armijo constants, centered prior, deployment | `work/prior_vector/solver.py`, `Problem`, `train_prior_anchor`, `fit`; `work/calibration_baselines/solver.py`, `Problem`, constants, `transform` |
| Probability metrics, macro/weighted summaries, sum-Brier, ECE bin boundary, q10 | `work/prior_vector/evaluate_saved.py`, `predictive_metrics`, `summarize`, `run` |
| Prior-BCTS same-anchor mechanism comparison and dimensionality caveat | `work/prior_vector/BCTS_CONTROL_PLAN.md`; `work/prior_vector/bcts_control.py`, `Problem`, `fit` |
| Frozen-feature 650-parameter one-epoch head and TRAIN-only prior/bias | `work/calibrated_lowbit_confirm/evaluate.py`, `matched_head`, `evaluate_model` |
| Strict same-cap TopK actual bytes and ordering | `work/calibrated_topk_reference/wire.py`, `encode`, `decode`; profile 64 + 6843*8 = 54,808 B |
| FedRoD derived reference, raw counts, persistent local head, B.2 endpoint choice/costs | `work/fedrod_linear_reference/PLAN.md`, paragraphs anchored to primary-paper Eqs.6-10,13-14; implementation choices explicitly disclosed |
| Packet fields, canonical envelope, HMAC-SM3, receipts, durable local state transitions | `work/durable_integrity_v2/protocol.py`, `Packet`, `Envelope`, `LocalServer.register/accept/finalize`, `ClientStore.prepare/commit_if_incorporated` |
| Actual integration scope and trust boundaries | `work/full_payload_integrity/REPORT_DISTINCT.md`, lines 1-9; different payloads/constructed deltas, local subprocess recovery, simulation_only |

## Citation keys and archived primary sources

Keys identify existing archived primary material; they are not invented-reference placeholders. A separate reference task owns `verified_references.bib` and expanded citation coverage. All six keys in the draft have identifiable local originals. The formulas and implementation settings in this Methods are our recorded implementations and are not attributed to those papers as exact reproduced choices.

| Key | Original and exact attribution scope |
|---|---|
| `zhang2022fedlc` | `work/label_calibration/fedlc_primary.pdf`, `.txt`, `.html`; publisher https://proceedings.mlr.press/v162/zhang22p.html; PDF p5 Eqs.5-7 and p17 Appendix C. `SOURCES.md` records the denominator discrepancy and lack of prescribed tau/zero-count handling. Draft only says inspired by FedLC and explicitly defines our all-class CE/count-floor/tau1 implementation. |
| `karimireddy2019errorfeedback` | `work/paper_draft/reference_sources/error_feedback.html`; https://proceedings.mlr.press/v97/karimireddy19a.html; existing `work/paper_draft/references.bib` key. Used at citation level for established error feedback. The exact implemented equation is sourced from local train.py, not a claimed paper theorem or new convergence result. |
| `guo2017calibration` | `work/prior_bias/sources/guo_cal.html`, `.txt`; arXiv 1706.04599, section 4, matrix/vector scaling Eq.8 and temperature-scaling subsection, extracted lines 378-395. Supports established diagonal vector scaling and scalar temperature; original held-out fitting is not described as our TRAIN-only protocol. |
| `alexandari2020labelshift` | `work/prior_bias/sources/bcts.html`, `.txt`; arXiv 1901.06852, section 3.1, extracted lines 53-54, bias-corrected temperature scaling. Our ridge/positive-bound/prior-anchor settings are local protocol choices. |
| `chen2022fedrod` | `work/prior_personalization/sources/fedrod.html`, `.txt`; arXiv 2107.00778v2, sections 4.1/4.2, Eqs.6-10, Appendix B.2 Eqs.13-14, extracted lines 1295-1307. Supports generic+residual logits and gradient separation. Main text/local versus appendix/global endpoint discrepancy is disclosed; no official-code or exact-hyperparameter reproduction claim. |
| `rfc2104` | `work/paper_draft/reference_sources/rfc2104.txt`, section 2, existing `work/paper_draft/references.bib`; https://www.rfc-editor.org/rfc/rfc2104. Used for shared-secret HMAC. Actual SM3 instantiation is verified from protocol.py; not claimed to be an RFC2104 recommendation or a signature. |

Citation checking used supplied local full texts/primary publisher archives and source-grounded review. No new online literature retrieval or separate-family review was performed by this writing task; external review status remains REVIEW_UNAVAILABLE. Sibling reference auditing is a separate same-family check and should not be labeled external audit PASS.

## Source snapshot SHA256

- `work/predictive_pfl_confirm/PLAN.md`: `703d5956c9edd24d5a03501a75f678a43222d19bd3dd1ede515ed77545b0830b`
- `work/predictive_pfl_confirm/PREREGISTRATION.json`: `1eb7d284c43992a3cca58e31448a4b595b88e67abdf167c24a605342bf2f0568`
- `work/calibrated_lowbit_confirm/common.py`: `3f2e4693488d7e5a35a182e5340edb9d73f5cafc45b7c0e61b1e2cc392a22d9a`
- `work/calibrated_lowbit_confirm/train.py`: `7fe8777db0ab8b2309140b39026a9bdbec42cd09ba9d2b8ef0c89b831a5641bd`
- `work/balanced_quant/run.py`: `b94f6a3a985839e6781582c6203e3dd8154fc97cf7bcc8edfe9d4648240625d4`
- `work/mixed_precision/codec.py`: `46f3601e5c107f192d359227b21aea2734cc5e798674f2dccbbdf46fc796d863`
- `work/method_followup/run_partitions.py`: `ca338c616ff12e4ec4d3a75958292af7166109555baa2bc3262407b64d4776b6`
- `work/calibrated_fashion/adapter.py`: `624109d019433ed1983bdc810063cdd5fe46116f0896c9dd8636214d553915d2`
- `work/method_followup/run_fashion.py`: `f87e281acaf9a46a62eb31be588c9505ea7bb53c67162fd8fcf2ccc3440111d4`
- `work/prior_vector/solver.py`: `9c3196eb3e6ff7f5f32073e3bc4aae8d986ac7e44bdf0d719258343847d4dbae`
- `work/calibration_baselines/solver.py`: `10b11405e6662cb06d08a20549db92e2948b730c426aa1f0b912b958f1211f1c`
- `work/prior_vector/evaluate_saved.py`: `7b45a055df8c3e03421fd9edd6749c1e2bfba1203864e0b41695413db1fb8b05`
- `work/prior_vector/BCTS_CONTROL_PLAN.md`: `9ea93ea8496f33f4f7330a1ea7a3764e3aa101432c04c0bb916903370116d1cc`
- `work/calibrated_lowbit_confirm/evaluate.py`: `afcb8a47dc9c303b3e7d558e4e91b50b560cd10dd0287956969d61ab27e13250`
- `work/calibrated_topk_reference/wire.py`: `6bf9856e73fba6d601bbcd8ce19cb9ec3bdfe09f17e58ca79e0faef000c86693`
- `work/fedrod_linear_reference/PLAN.md`: `c5817d080e4ac55ba9a3ea8a4739b1885ba33a84a26be5056e97aed51eacd895`
- `work/durable_integrity_v2/protocol.py`: `0b40fe49834a57a25c46006022e1161d6e0b3980de49ccca9c0a723f101ebc51`
- `work/full_payload_integrity/REPORT_DISTINCT.md`: `1331985593f685400058e2c78863733d074088cba13ff950622bf361e3388343`
- `work/predictive_pfl_confirm/CIFAR_THREE_SEED_SUMMARY.json`: `702dca88f5527e5d89fa888ab54dde51200ac45ca819fbbf71d159ef70ad0d55`

## Draft SHA256

- `METHOD_EN.md`: `9b4d8403b737e6b42fccd1a88cd4a33223dd477f0fc565c7faf9eaffb3b2cc05`
- `METHOD_ZH.md`: `100b62a4a8f3f28c1b1bc2556944523037905e8d91261d899a4521e8d5278a64`
- `METHOD_EN.tex`: `811684a980dda992b7a4cc31bb104de51c5c56caf4ea2a177e89c5f89fd996a0`

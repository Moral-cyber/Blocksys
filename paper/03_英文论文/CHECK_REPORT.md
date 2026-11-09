> **HISTORICAL** — 本文档描述的是 12 页的 Route A 版本（六行主表）。当前稿件为 14 页、十行主表，以 `checks/CURRENT_STATE.md` 为准。

# Route A manuscript checks

Finalized 2026-11-05T16:14:32.587956+00:00. The required pdfLaTeX → BibTeX → pdfLaTeX ×2 sequence exited 0. The PDF has12 pages including appendix/references,33 cited references from34 bibliography entries,136 abstract whitespace tokens,two main tables,six primary rows and three vector figures. No undefined references/citations or overfull boxes remain; underfull notices are nonfatal.

Root inspected all12 final rendered pages, with individual table-page checks. All27 font resources are embedded. The17 protected files, original preamble, original bibliography byte prefix and all13 numeric groups are preserved; the six-seed table is byte-identical. `manuscript_checks.json`, `BUILD_RECORD.json`, `visual_and_source_checks.json` and final13-group report are under `revision_artifacts/route_a_20261105/`.

All selected P1/P2 evidence work is complete: dense FedAvg/FedProx, pFedMe, three-partition development and CIFAR-100 development. Saved-score replay and independent aggregation passed; final source audit checked1,274 references to677 distinct files. Development results and original TEST remain explicitly separate. Baseline adapters, extra pFedMe computation, negative accuracy/ECE results and frozen-drift failure are disclosed.

New primary-source distinctions for PRIOR, LaSCal and FedNPC were inspected, including FedNPC full text and the CVF/Crossref pagination discrepancy. Original bibliography records remain byte-identical. The ScientistTwo bridge has two actual role outputs and five experiment imports; integrity PASS does not certify scientific novelty or an upstream live campaign.

The only new file under `checks/` is `REVISION_STATUS.md`; historical checks retain their original scope. The previous `final_checks.py` assumes the old layout and is not this revision's validator. See the current delivery manifest in `revision_artifacts/route_a_20261105/DELIVERY_MANIFEST.json`.

Final PDF SHA-256: `30a5fa1694810e3e46c6862003e59754825f6af0de71d585854e14f1ff85a0b8`.

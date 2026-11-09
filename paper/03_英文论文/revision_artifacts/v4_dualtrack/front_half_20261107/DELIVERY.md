# Front-half revision — 2026-11-07

Current main.pdf remains **14 pages including appendix and references**. Figure2 replaces the old client scatter with two panels; Figure3/4 and table numbering are unchanged. Table1 and Table2 retain exact prior hashes. AppendixA was not changed because the page cap was met on the first successful compile.

- Section3: exact user-supplied shrinkage sentence inserted immediately after the mean-loss penalty statement.
- Section4: assessment boundaries added for E1 TRAIN-derived development holdouts and E2 saved originalTEST outcomes.
- Section5: Main results; Heterogeneity sweep; Who benefits: client sample count; Class-alignment control. Old compressed-training subsection removed.
- E1: this three-point sweep qualifier retained; alpha0.1 improvement counts60/60 and59/60.
- E2: rho-.771/-.729,20clientmeans/task; samplecount association.
- Class permutation: one sentence, .071/.044.

Rendered Figure2 image height: **42.792mm** at **109.805mm** width. Including its shared caption and intervening gap: **66.250mm**. The caption starts with the exact saved sentence and identifies left sweep/right client size, their assessment samples, and gain sign.

| Subsection | Body lines | With one heading line |
|---|---:|---:|
|5.1 Main results|9|10|
|5.2 Heterogeneity sweep|7|8|
|5.3 Who benefits: client sample count|5|6|
|5.4 Class-alignment control|3|4|

Counts use visible lines in the compiledPDF, excluding float/table contents, captions and blank lines. Measurements: LAYOUT_METRICS.json. Figure source/transformation record: figure2_provenance.json and draw_figure2.py. Final check: FINAL_CHECK.json. Compilation: compile.log; zero overfull/undefined/warnings. OriginalE1/E2/E3 evidence remains unchanged. Independent read-only audit found no blockers.

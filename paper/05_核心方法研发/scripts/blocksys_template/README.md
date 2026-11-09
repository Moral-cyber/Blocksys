# BlockSys svproc format adaptation

The user selected the 2026 BlockSys kit for their 2027 manuscript and confirmed this choice. `template/svproc.cls` and `template/spmpsci_unsrt.bst` are unchanged copies. See TEMPLATE_SHA256.json and COMPLIANCE_NOTE.md.

No experiments, seed selection, retraining or inference are performed by this workflow. The fixed numerical tables still originate in the completed predictive_final_test result JSONs.

- `adapt.py`: deterministic conversion from the saved generic manuscript to the user-selected template, with a compact abstract, task-stacked tables, compact front/discussion, and final-size artwork. `--from-current` is used only directly after the baseline builder generates fresh generic source; it is not a mode for reapplying to an already adapted table.
- `compact_sections.py`, `FRONT_EN.tex`, `FRONT_ZH.tex`: bilingual prose edits. Method equations and solver settings are retained. Full development gates remain in each paper's evidence directory and the research archive.
- `tables.py`: reflows source cells without rounding or changing metrics.
- `figures.py`: final-size 122 mm vector artwork, source data, alt text and provenance.
- `compile.py`: compiles manuscript sources only. English uses pdfLaTeX/BibTeX; Chinese uses Tectonic (XeTeX), with xeCJK and Fandol. The Chinese source also supports a suitably equipped XeLaTeX/BibTeX installation.
- `render.py`: all-page previews and font/page inventory. It needs pypdfium2, pypdf and Pillow.
- `work/predictive_manuscript/final_checks.py`: frozen numeric-source hashes, exact original numeric cells after reflow, first-citation ordering, unchanged template hashes, abstract length, page count, and reviewed render hashes. It packages the clean sources after checking.

The class controls paper size, text area, fonts and spacing. Only Chinese-language font support and long-URL breakpoints are added; English body type and the class/bibliography style remain intact. Benign underfull spacing notices are recorded rather than hidden by changing template spacing.

From the workspace root, with the original evidence paths available:

```bash
python work/predictive_manuscript/build_papers.py
python work/blocksys_template/compile.py
python work/blocksys_template/render.py
```

Inspect all pages before recording a new VISUAL_CHECK.json and running final_checks.py. Existing visual-check hashes become invalid if the PDF changes. Do not use this workflow to imply a new independent numerical experiment.

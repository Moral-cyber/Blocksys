# Table spacing revision

- Final PDF: `../../../main.pdf`, 14 pages.
- `tables/main.tex`: local `tabcolsep=5pt`; rule width 117.639 mm inside the 122 mm text block. No Table 1 overfull, so the 4pt fallback is unused.
- `tables/quality.tex`: `tabcolsep=3.5pt` as requested. The upper tabular reports one 16.22789pt overfull (about 5.70 mm). No unrequested resizing or font/value edits were applied.
- All remaining table sources match the before snapshot. Numeric values and table fonts are unchanged.
- Page 5 rendered at 200 dpi and visually checked. Table 1 minimum adjacent numeric cell gap is approximately 9.962 bp (3.51 mm or 27.7 pixels at 200 dpi).
- Author, affiliation, email and short-name placeholders are present. Chinese support and local title-block whitespace adjustments keep the new author block within the 14-page total.

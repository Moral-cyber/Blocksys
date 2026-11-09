# PRIOR citation verification

Independently inspected by `/root/route_a_citations`, 2026-11-05, from pages
retrieved by root. No new network request or manuscript edit was performed here.

**The published reference is PRIOR, with final author Jiancheng Lv.** Use
`prior_verified_append.bib` for the new `shi2023prior` entry only. The existing
bibliography prefix must remain untouched.

Official NeurIPS page:
<https://proceedings.neurips.cc/paper_files/paper/2023/hash/5a3674849d6d6d23ac088b9a2552f323-Abstract-Conference.html>

Local page: `/private/tmp/scientisttwo_sources/near/prior_official.html`.
The metadata and visible author/title fields agree:

| Field | Official NeurIPS value |
|---|---|
| Title | PRIOR: Personalized Prior for Reactivating the Information Overlooked in Federated Learning. |
| Authors, in order | Mingjia Shi; Yuhao Zhou; Kai Wang; Huaizheng Zhang; Shudong Huang; Qing Ye; **Jiancheng Lv** |
| Venue | Advances in Neural Information Processing Systems |
| Volume | 36 |
| Year / publication date | 2023 / 2023-12-15 |
| Pages | 28378–28392 |
| Published DOI | 10.52202/075280-1234 |
| PDF | <https://proceedings.neurips.cc/paper_files/paper/2023/file/5a3674849d6d6d23ac088b9a2552f323-Paper-Conference.pdf> |

The official 2023 index independently lists the same title and author sequence.
The previously cached Crossref record matches title, authors, pages, year and DOI.
This is sufficient to use the published reference rather than downgrading to a
metadata-unresolved preprint entry.

ArXiv v2 is separately identified by
<https://arxiv.org/abs/2310.09183v2>, dated 2023-11-10 (initial submission
2023-10-13), DOI **10.48550/arXiv.2310.09183**. Its title is also PRIOR, not
“Personalized Federated Learning via Bregman Divergence”. The latter title appeared
in an intermediate critic draft through reviewer error and was corrected in the
final report before source hashing. It is not a distinct title supported by these
retrieved pages.

The arXiv v2 metadata, visible author list and experimental full-text HTML spell
the last author **Jiangcheng Lv**. That differs from the published official
NeurIPS spelling **Jiancheng Lv**. Preserve this version discrepancy in research
records, but cite the official published spelling when citing the proceedings.
Do not combine arXiv DOI/title metadata with an assumed author spelling, and do
not claim to know why the spelling changed.

The method interpretation was based on the actually inspected arXiv v2 full text
Eq.8/11/14 and Algorithm 1: personalized Bregman/model-space prior centers are
prior art. Matching metadata does not itself prove that every equation is
identical across publication versions; no cross-version equation identity is
asserted. The scientific distinction in the paper should remain about observed
local class information in a frozen output-space fit, not generic prior centers.

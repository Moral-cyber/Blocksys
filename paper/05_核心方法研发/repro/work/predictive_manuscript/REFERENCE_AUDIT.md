# Reference audit for the predictive-personalization manuscript

Audit date: 2026-11-04. Bibliography: `verified_references.bib` (20 entries).

This audit began with existing local archives and was completed by a targeted official-source follow-up for FedPer, QSGD, Fashion-MNIST and the separate software acknowledgement on 2026-11-04. It is not an independent external review. The audit checks bibliographic identity and a bounded attribution role. It does not verify this manuscript's experimental results, reproduce a cited method, or transfer another paper's convergence/security guarantees to the present implementation. External review status: **REVIEW_UNAVAILABLE**.

The file name `verified_references.bib` does not imply that every field was checked against an original publisher page. Nineteen entries now have archived original-paper, publisher/author-metadata, or source-document evidence; CIFAR is identified by its official dataset page. The earlier three VERIFY_PRIMARY items have been resolved at the bibliographic-metadata level. Field-level differences are recorded below. No missing DOI, arXiv identifier, author forename, proceedings detail, or page range was supplied from memory. Existing archives were read without editing them. No training, dataset evaluation or change under `outputs/` was performed for this reference task. The scoped follow-up used urllib to retrieve only official arXiv and NeurIPS literature pages; it did not open official dataset TEST files or modify frozen inference sources.

## Evidence labels and remaining checks

- **LOCAL_PRIMARY**: the paper, author-provided metadata, publisher page, RFC, or Internet-Draft is archived locally and supports identity or attribution. This label applies to the available evidence, not automatically every optional field.
- **LOCAL_PRIMARY_METADATA**: a target paper's official arXiv record or publisher entry has been retrieved and archived directly. FedPer, QSGD and Fashion-MNIST were upgraded to this status in the scoped follow-up. Their metadata are verified; detailed full-text/theorem verification is not implied. No VERIFY_PRIMARY item remains in the 20-entry inventory.
- **LOCAL_DATASET_METADATA**: the original dataset website explicitly identifies the requested citation. The linked report itself is not archived. This applies to CIFAR.
- **VERIFY_VENUE**: a formal conference citation was not established from the inspected local material. BCTS and Menon are therefore cited as the exact archived arXiv versions, with no assumed proceedings fields. This is a complete citation to those versions, rather than a claim that they lack formal publication.

Additional field limits: FedNova's NeurIPS venue is corroborated by FedRoD's bibliography, Guo's volume/pages by BCTS's bibliography, Balanced Meta-Softmax's volume/pages by a locally saved OpenAlex proceedings record, and QSGD's retained page range by FedCOMGATE's bibliography. The retrieved official QSGD BibTeX verifies volume 30 but leaves pages empty. Their original paper identities and method passages are available, but those particular optional fields do not have an archived publisher BibTeX in this audit. These distinctions are retained rather than silently elevating all metadata to publisher verification.

## Entry-by-entry evidence and attribution

All paths below are relative to `/Users/wind/Documents/Codex/2026-11-02/zhe`. Citation keys are stable identifiers; they are not reference numbers.

| Key | Bibliographic identity and local evidence | Status, intended role, and limits |
|---|---|---|
| `mcmahan2017fedavg` | Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas. *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS 2017, PMLR 54:1273–1282. `work/paper_draft/reference_sources/fedavg.html`, publisher-supplied BibTeX. | **LOCAL_PRIMARY**. Iterative federated averaging and communication motivation. Does not establish authentication, poisoning resistance, or this manuscript's implementation results. No DOI is asserted. |
| `li2020fedprox` | Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith. *Federated Optimization in Heterogeneous Networks*. MLSys 2020. `work/literature_followup/baseline_primary_metadata.xml`, arXiv entry 1812.06127v5, ordered authors and `MLSys 2020` comment; original `fedprox.html`/`.txt` in the same folder. | **LOCAL_PRIMARY**. Heterogeneous federated optimization/proximal reference. Conference year 2020 is distinct from the first arXiv posting in 2018. Proceedings volume/pages and DOI are omitted. |
| `karimireddy2020scaffold` | Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, Ananda Theertha Suresh. *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning*. ICML 2020, PMLR 119:5132–5143. `work/head_control/sources/scaffold.html`, official publisher BibTeX. | **LOCAL_PRIMARY**. Control variates/client drift as related optimization work. The cited paper's assumptions and theorem do not prove convergence of the present compressed/personalized system. No DOI is asserted. |
| `wang2020fednova` | Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, H. Vincent Poor. *Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization*. NeurIPS 2020. Original `work/exposure_prior/sources/fednova.html`/`.txt`, arXiv 2007.07481v1; ordered citation and NeurIPS 2020 corroborated in `work/prior_personalization/sources/fedrod.txt`, reference Wang et al. (2020b). | **LOCAL_PRIMARY**, with **secondary venue metadata**. Normalized aggregation/objective inconsistency is related work. No proceedings volume/pages/DOI are supplied. It is not evidence of a FedNova experiment in this manuscript. |
| `chen2022fedrod` | Hong-You Chen, Wei-Lun Chao. *On Bridging Generic and Personalized Federated Learning for Image Classification*. ICLR 2022. `work/prior_personalization/sources/fedrod.html`/`.txt`, arXiv 2107.00778v2; author-repository BibTeX in `work/balanced_quant/sources/fedrod_README.md`, lines 7–11. | **LOCAL_PRIMARY**. Sections 4.1/4.2, Eqs. 6–10 support balanced generic training and persistent private residual personalization. Appendix B.2, Eqs. 13–14 support the selected global-generic-plus-local-residual endpoint and gradient separation. The manuscript discloses this endpoint choice and does not claim an exact official-code/hyperparameter reproduction. Initial arXiv 2021 is not the conference year. |
| `zhang2022fedlc` | Jie Zhang, Zhiqi Li, Bo Li, Jianghe Xu, Shuang Wu, Shouhong Ding, Chao Wu. *Federated Learning with Label Distribution Skew via Logits Calibration*. ICML 2022, PMLR 162:26311–26329. `work/label_calibration/fedlc_primary.html`, official publisher BibTeX; original PDF/TXT alongside it. | **LOCAL_PRIMARY**. Paper p.5, Eqs. 5–7 and Appendix C motivate a label-count-dependent margin. The manuscript explicitly defines its count floor, unit multiplier and all-class CE implementation as FedLC-inspired. The reference is not evidence that these local choices exactly reproduce the original formula or code. |
| `arivazhagan2019fedper` | Manoj Ghuhan Arivazhagan, Vinay Aggarwal, Aaditya Kumar Singh, Sunav Choudhary. *Federated Learning with Personalization Layers*. arXiv:1912.00818, 2019. Direct official record https://arxiv.org/abs/1912.00818 archived as `work/predictive_manuscript/reference_primary_followup/fedper_abs.html`; its citation_title, ordered citation_author, citation_date and submission history verify v1, 2 December 2019. DOI 10.48550/arXiv.1912.00818 is explicitly displayed by arXiv. | **LOCAL_PRIMARY_METADATA; earlier VERIFY_PRIMARY resolved**. The abstract identifies the base + personalization-layer approach, supporting the manuscript's citation-level context. This audit does not establish detailed formula/code reproduction. No formal venue is asserted. |
| `guo2017calibration` | Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. *On Calibration of Modern Neural Networks*. ICML 2017, PMLR 70:1321–1330. Original `work/prior_bias/sources/guo_cal.html`/`.txt`, arXiv 1706.04599v2; proceedings fields independently visible in archived `bcts.txt`, extracted lines 1081–1088. | **LOCAL_PRIMARY**, with **secondary proceedings volume/pages**. Section 4.2, Eq. 8 and the temperature-scaling subsection establish vector scaling and scalar temperature scaling as prior methods. Our TRAIN-only fitting, centering, unrestricted scales, prior anchor and penalties are separate local choices. The HTML's “Supplementary Materials for” wrapper is not included in the canonical main-paper title. |
| `alexandari2020labelshift` | Amr M. Alexandari, Anshul Kundaje, Avanti Shrikumar. *Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation*. arXiv:1901.06852v5, 27 June 2020. `work/prior_bias/sources/bcts.html`/`.txt`, front matter and section 3.1, extracted lines 53–54. | **LOCAL_PRIMARY / VERIFY_VENUE**. Establishes bias-corrected temperature scaling as scalar temperature plus class biases fitted by validation NLL. It does not prescribe our TRAIN fit, positive coefficient bound, ridge value, or prior bias anchor. Label-shift adaptation claims require the paper's conditional-invariance/calibration assumptions. The BibTeX deliberately cites v5, without assuming a venue from the keyword “ICML.” |
| `menon2021logitadjust` | Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, Sanjiv Kumar. *Long-Tail Learning via Logit Adjustment*. arXiv:2007.07314v2, 9 July 2021. `work/prior_personalization/sources/menon_la.html`/`.txt`, ordered front matter and section 3, Eqs. 9–10. | **LOCAL_PRIMARY / VERIFY_VENUE**. Distinguishes training-time from post-hoc prior/logit adjustment. Use as prior art and motivate empirical controls; do not portray adding an empirical client prior as automatically Bayes-correct without distribution assumptions. The exact v2 citation avoids an unverified proceedings field. |
| `ren2020balanced` | Jiawei Ren, Cunjun Yu, Shunan Sheng, Xiao Ma, Haiyu Zhao, Shuai Yi, Hongsheng Li. *Balanced Meta-Softmax for Long-Tailed Visual Recognition*. NeurIPS 2020, 33:4175–4186. Original `work/balanced_quant/sources/balanced_meta_neurips2020.pdf`/`.txt`, p.1 authors and NeurIPS footer; proceedings record W3106060611 in local `02_balanced_meta_softmax_search.json` gives volume/pages. | **LOCAL_PRIMARY**, with **secondary volume/pages**. Balanced Softmax is prior art for class-count adjustment, including the FedRoD generic branch. Balanced Softmax and the complete Balanced Meta-Softmax/meta-sampler pipeline must not be treated as interchangeable implementations. No DOI is asserted. |
| `alistarh2017qsgd` | Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, Milan Vojnovic. *QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding*. NeurIPS 2017, vol.30. Direct https://arxiv.org/abs/1610.02132 (v4, 6 December 2017) archived as `reference_primary_followup/qsgd_abs.html`; official NeurIPS record and BibTeX archived as `qsgd_neurips.html`/`.bib` in that folder verify title, ordered authors, conference, volume and year. The existing pp.1709–1720 remain supported by `work/head_control/sources/fedcomgate_pdf.txt`, reference [3]; the official current BibTeX leaves pages empty. | **LOCAL_PRIMARY_METADATA; earlier VERIFY_PRIMARY resolved**, with **secondary page-range metadata**. Broad quantization/encoding related work is supported by the official abstract. Original theorem and detailed bit-accounting checks were not performed; this is not an identical-codec or convergence claim. No proceedings DOI is asserted; the arXiv identifier is now included. |
| `karimireddy2019errorfeedback` | Sai Praneeth Karimireddy, Quentin Rebjock, Sebastian Stich, Martin Jaggi. *Error Feedback Fixes SignSGD and other Gradient Compression Schemes*. ICML 2019, PMLR 97:3252–3261. `work/paper_draft/reference_sources/error_feedback.html`, official publisher BibTeX. | **LOCAL_PRIMARY**. Supports established error-feedback terminology and accumulated quantization error. Does not prove convergence for the manuscript's non-IID FL setting or establish its receipt-dependent durable protocol. No DOI is asserted. |
| `stich2018sparsified` | Sebastian U. Stich, Jean-Baptiste Cordonnier, Martin Jaggi. *Sparsified SGD with Memory*. NeurIPS 2018, vol.31. `work/paper_draft/reference_sources/sparsified_sgd.html` and publisher `sparsified_sgd.bib`. | **LOCAL_PRIMARY**. Top-k/random-k sparsification with memory is related work. The publisher's current archived BibTeX has no page range or DOI; both are omitted. Our exact 6,843-coordinate serializer, tie rule and wire cost are locally defined. |
| `kim2020blockfl` | Hyesung Kim, Jihong Park, Mehdi Bennis, Seong-Lyun Kim. *Blockchained On-Device Federated Learning*. *IEEE Communications Letters* 24(6):1279–1283, 2020. `work/paper_draft/reference_sources/blockfl_doi.json`, Crossref `message`; corroborating `blockfl_arxiv.html`. DOI 10.1109/LCOMM.2019.2921755. | **LOCAL_PRIMARY_METADATA**. Related blockchain/FL architecture and latency discussion. The formal issue year is 2020 although the DOI contains 2019. This citation does not imply that the present local SQLite/HMAC simulation deploys blockchain, consensus or a public ledger. |
| `rfc2104` | H. Krawczyk, M. Bellare, R. Canetti. *HMAC: Keyed-Hashing for Message Authentication*. RFC 2104, 1997. `work/paper_draft/reference_sources/rfc2104.txt`, section 2; RFC Editor XML in `rfc2104.bib`. DOI 10.17487/RFC2104. | **LOCAL_PRIMARY**. Informational RFC describing the generic shared-secret HMAC construction. It is not a digital-signature/public-verification/nonrepudiation specification and does not specifically recommend this implementation's SM3 choice. |
| `rfc8998` | P. Yang. *ShangMi (SM) Cipher Suites for TLS 1.3*. RFC 8998, March 2021. `work/paper_draft/reference_sources/rfc8998.txt`; RFC Editor XML in `rfc8998.bib`. DOI 10.17487/RFC8998. | **LOCAL_PRIMARY**. **Informational**, not Standards Track. It specifies TLS cipher-suite context and explicitly says it is not endorsed by the IETF. It is not a standalone normative specification of the SM3 algorithm or of our packet/receipt protocol. |
| `shen2018sm3` | Sean Shen, Xiaodong Lee, Ronald Henry Tse, Wai Kit Wong, Paul Y. Yang. *The SM3 Cryptographic Hash Function*. draft-sca-cfrg-sm3-02, 8 January 2018. `work/paper_draft/reference_sources/sm3_draft.txt`, title block and Authors' Addresses. | **LOCAL_PRIMARY**. **Expired Internet-Draft; work in progress; not an RFC.** This is distinct from RFC8998 and from the referenced Chinese national standard. It identifies the archived algorithm description; no DOI or RFC number is invented. |
| `krizhevsky2009tiny` | Alex Krizhevsky. *Learning Multiple Layers of Features from Tiny Images*. Technical report, 2009. `work/paper_draft/reference_sources/cifar10.html`, “Reference” section, extracted lines 357–361, explicitly instructs citing this title/author/year and links the report. | **LOCAL_DATASET_METADATA / VERIFY_REPORT**. Official dataset provenance and its requested citation. The linked report is not archived, so report institution/report number are omitted. Dataset creators listed on the webpage are not substituted for the report's explicitly requested sole author. |
| `xiao2017fashion` | Han Xiao, Kashif Rasul, Roland Vollgraf. *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. arXiv:1708.07747, 2017. Direct official record https://arxiv.org/abs/1708.07747 archived as `work/predictive_manuscript/reference_primary_followup/fashion_abs.html` verifies full ordered author names, title, identifier, first submission 25 August 2017 and current v2, 15 September 2017. DOI 10.48550/arXiv.1708.07747 is explicitly displayed by arXiv. | **LOCAL_PRIMARY_METADATA; earlier VERIFY_PRIMARY resolved**. Author initials were expanded only after reading this primary record. The official abstract identifies 28x28 grayscale images, ten classes and the 60,000/10,000 split; this metadata retrieval is not a current experiment or a dataset TEST read. |

## Integration and checks performed

Use `\bibliographystyle{unsrt}` and `\bibliography{verified_references}` in the main LaTeX document. Do not hand-number references. Only cited entries are printed by the bibliography style; the 20-entry inventory is not a requirement to cite all twenty in the manuscript.

The six keys used in `METHOD_EN.md`, `METHOD_EN.tex` and `METHOD_ZH.md` all resolve: `zhang2022fedlc`, `karimireddy2019errorfeedback`, `guo2017calibration`, `alexandari2020labelshift`, `chen2022fedrod`, `rfc2104`. A read-only check of those Methods passages found no material attribution issue within the above limits. The methods writer retains responsibility for its implementation/source map in `WRITING_NOTES.md`.

Offline checks used the citation-management skill's standard-library, brace-depth-aware BibTeX parser (`scripts/_common.py`) plus local required-field and duplicate checks. Result: **20 raw entries / 20 parsed entries; no missing required fields; no duplicate keys, normalized titles, DOI identifiers or arXiv identifiers; no unresolved Methods citation keys**. Optional metadata omissions are intentional and documented above.

The skill's complete `validate_citations.py` command could not start because this environment's default Python lacks `requests`. No dependency was installed and no full DOI-resolver validation was run. The later targeted urllib retrieval is recorded separately below. The successful local parser/field/citation checks are therefore reported directly, not relabeled as a full validator PASS or as compilation. No standalone BibTeX/LaTeX engine was available on the current command path for this reference task; main-document compilation belongs to manuscript integration.

The machine-readable check and source digests are retained as an intermediate record at `work/predictive_manuscript_reference_validation.json`. Bibliography SHA256 after the primary-source follow-up: `b5ac259c7c015ed7ac7596dbd642240b9beda431d297c2a35bbe4673bae10c26`.

## Source snapshot SHA256

The digests below identify the exact local evidence checked. A digest records file identity and does not itself establish correctness or primary-source status.

- `work/paper_draft/reference_sources/fedavg.html`: `1297343d64424a8234d881d2afff34fe4b080bd44a8fc38945497d8fbb208caa`

- `work/literature_followup/baseline_primary_metadata.xml`: `5d6ddc2b955e87b2defc8e4449c8972f29d0386f2b7dc9b80b9248d10a1f5511`

- `work/head_control/sources/scaffold.html`: `ab8c615349975a58a71f0fd28bca66aa5c4095914c1979ea6377a85a267b76b2`

- `work/exposure_prior/sources/fednova.txt`: `d2ef885147200567c350b0f0d5744948b95380eb3e9c6f201f83e327079c9a02`

- `work/prior_personalization/sources/fedrod.txt`: `b33fbbd8f63cce4aa78067c9f95b79ebd9c01a4e8da3cfdbd287aada0a8025f3`

- `work/balanced_quant/sources/fedrod_README.md`: `b8dc36a73491a30007a4d906403d79d8c3768fb6485a50d60ebeeaac6618c60e`

- `work/label_calibration/fedlc_primary.html`: `f1b2e6e2ffef03e52590ca9ee48f2ea2da59b212835a7d87a07f5b541e907d70`

- `work/prior_bias/sources/guo_cal.txt`: `8d5afe341cc1a2686af2490ef81fd16584023357ac507ae1419d40bdc04ca113`

- `work/prior_bias/sources/bcts.txt`: `452728d1af43080d1155612742fa5f74c80caccce65567a8473de6344979efa4`

- `work/prior_personalization/sources/menon_la.txt`: `1f4a6276320f7fbdc14c7dda1f0d22775e571b422078278fa02d4e87fc07a2a4`

- `work/balanced_quant/sources/balanced_meta_neurips2020.txt`: `109143e1bd2af0094c662d4903a522f124a30e1c86d9d7c09f5c15f94c5d09b8`

- `work/balanced_quant/sources/02_balanced_meta_softmax_search.json`: `fcf821bbee7ff785de4bd3bb66935b20fa6eb2c48d872a7e9474cca75d3592c4`

- `work/weighted_feedback/ef21_w.txt`: `c256c6c007889b931d0155fc75f815f5c7c01de545b713517eb194d4469960c1`

- `work/head_control/sources/fedcomgate_pdf.txt`: `f7e98aabb5d5d573e22aacf2c40901f0f00d7af66547e341466594709f99f599`

- `work/paper_draft/reference_sources/error_feedback.html`: `89dc417d3d9af665159f8a284f5260b22ff60eb67171ad1d3365b11affe9534e`

- `work/paper_draft/reference_sources/sparsified_sgd.bib`: `e0f057dfca815860a17b64d28e290fbff0ba4869c51f9a8fda6819c186f63251`

- `work/paper_draft/reference_sources/blockfl_doi.json`: `63045b73fa7b4f1b36448a4592b485f77c9d21c97309a29c79ffed1a56110d16`

- `work/paper_draft/reference_sources/rfc2104.txt`: `64d5245a9101929025336e470e3737f118704001249d503c85a86e19fe9fbb01`

- `work/paper_draft/reference_sources/rfc8998.txt`: `de2b7d694de1981464dacb6c3e5aaf45583a26e702fe523f25be4fd28eb85d78`

- `work/paper_draft/reference_sources/sm3_draft.txt`: `095dfb60526d1dc1d32be64577ba730da75aa3ef7b43fc72dcb016ee7e3afcd2`

- `work/paper_draft/reference_sources/cifar10.html`: `c95fa60b177f91997e0cb891bafdae1bad862e3a4c1d88621693c8103e0cae93`

## Targeted primary-source and software acknowledgement follow-up

The task owner authorized a bounded online follow-up on 2026-11-04. The ordinary environment initially could not resolve arXiv; the permitted network execution then retrieved the official records successfully. `reference_primary_followup/network_manifest.json` records URL, retrieval time, HTTP status, byte count and SHA256 for the four arXiv pages. The QSGD publisher page was reached through the official 2017 NeurIPS directory and its publisher BibTeX was saved. This follow-up resolves the three previously secondary-only bibliographic identities without expanding the 20-entry research bibliography.

The scientific-visualization skill materially informed the workflow figure, as already disclosed in `figures/workflow.alt.md` and `figures/generate_workflow.py`. Its requested software citation was independently checked against the real https://arxiv.org/abs/2609.00065 record, archived as `reference_primary_followup/scientific_agent_skills_abs.html`. The actual record verifies Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel and Aubrey M. Brueckner; *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*; 2026; current v2 posted 2 September 2026. The page displays DOI 10.48550/arXiv.2609.00065 and has no journal reference/publisher DOI field.

A separately reusable `SOFTWARE_NOTE_EN.tex` / `SOFTWARE_NOTE_ZH.tex` supplies the acknowledgement and full verified software citation with the unsuffixed current-version DOI. It is not an extra research reference in `verified_references.bib`, experimental evidence, or a claim of independent external review. No frozen `reference_inference.py` source or official TEST input was modified by this follow-up.

Follow-up evidence SHA256:

- `work/predictive_manuscript/reference_primary_followup/fashion_abs.html`: `d2c5526f7416b54eb1b145be94106d1b2499103ae0b1c1150fc580e7c08d5bd5`

- `work/predictive_manuscript/reference_primary_followup/fedper_abs.html`: `e456b243c829755827e4637fd39602ef347d478c2d6e56f6377b56300055e6c0`

- `work/predictive_manuscript/reference_primary_followup/network_manifest.json`: `feff496184dbddb98e3a666b8c4c355a4cd9c4909260661a7425f78e54f58f4e`

- `work/predictive_manuscript/reference_primary_followup/neurips2017_index.html`: `c8850a1f2364f9d8eb0ffb46972b0c8721c3fcffb29baf281c890bce3074359e`

- `work/predictive_manuscript/reference_primary_followup/qsgd_abs.html`: `830e518bce94c78d2e1fbefa3c55fc045de7fe9c43f47125d891a0611a8082ae`

- `work/predictive_manuscript/reference_primary_followup/qsgd_neurips.bib`: `8a51554f3bfde379ea548694dd31386bc43ddfe482f60bbf22f0c32dec61fb2b`

- `work/predictive_manuscript/reference_primary_followup/qsgd_neurips.html`: `56153fb5fd89f33b1998bebe383baee83c680a1a0ad04a75bf262c2c029a94dd`

- `work/predictive_manuscript/reference_primary_followup/scientific_agent_skills_abs.html`: `67cc1bf1d1a285943f563765900dcd3bd12c7ecf884903c05468e261b94068ce`

- `work/predictive_manuscript/reference_primary_followup/primary_metadata.json`: `8c23403643b594ad8682f9e50451a7e63a529e4eb022ab2a84f7f63e22094ed7`

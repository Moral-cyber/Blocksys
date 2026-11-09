# ScientistTwo attribution and local adaptation review

Checked 2026-11-06 (Asia/Shanghai) by `/root/route_a_citations`, using **already archived primary sources and the installed checkout only**. No new network retrieval, authentication, model call, experiment or upstream campaign was performed for this review.

## What may accurately be called a Google paper

The original work is **ScientistTwo: Pioneering the Human Knowledge Frontier with Autonomous AI**, Jaehyun Nam, Jinsung Yoon, Yanzhou Pan, Yubo Wang, Rui Meng, Parthasarathy Ranganathan and Tomas Pfister, **arXiv:2609.19644v1**, dated **2026-09-17**. It is accurate in ordinary conversation to call this a **Google-led ScientistTwo preprint** or a **paper by Google Cloud AI Research with a University of Waterloo coauthor**.

The direct evidence is the archived arXiv full-text author block in `paper_html.html:195–235`:

| Author | Affiliation stated in arXiv v1 |
|---|---|
| Jaehyun Nam | Google Cloud AI Research |
| Jinsung Yoon | Google Cloud AI Research |
| Yanzhou Pan | Google Cloud AI Research |
| Yubo Wang | University of Waterloo |
| Rui Meng | Google Cloud AI Research |
| Parthasarathy Ranganathan | Google Cloud AI Research |
| Tomas Pfister | Google Cloud AI Research |

The archived project page (`official.html:730–740`) lists the same seven names and displays “Google Cloud AI Research”. The per-author arXiv block is more specific and preserves the Waterloo affiliation. Neither source identifies this as a Google DeepMind project. The archived arXiv submission history (`paper_abs.html:156–159`) records v1 on 17 September 2026. This review does not establish a conference acceptance or later version.

Recommended short Chinese phrasing:

> Google Cloud AI Research 主导、滑铁卢大学研究者参与的 ScientistTwo 预印本（arXiv:2609.19644v1）。

Recommended English phrasing:

> We drew on the research workflow described in the Google Cloud AI Research–led ScientistTwo preprint (Nam et al., arXiv:2609.19644v1), which includes a University of Waterloo coauthor.

“Google论文” is acceptable shorthand **for that source paper**. It must not be transferred to the independent repository or our local adapter.

## Official-source availability is a bounded finding

`SOURCE_RECORD.json` archives these sources:

- `https://arxiv.org/html/2609.19644v1`
- `https://arxiv.org/abs/2609.19644`
- `https://scientist-two.github.io/`

All five recorded source/extraction hashes were recomputed in this review and matched: `paper_html.html`, `paper_abs.html`, `official.html`, `paper.txt`, `paper_abs.txt`.

The archived official project page links arXiv and “Generated Papers”. Among its literal `href`/`src` links, the only GitHub-hosted destination is the **Nerfies website-template credit**. There is no framework-repository/code/source/zip destination in those link attributes. This supports the existing carefully bounded status:

> No official ScientistTwo framework source repository or complete prompt release was identified in the archived official page checked on 2026-11-05.

It does **not** establish that no official source exists anywhere, that no later release will appear, or that generated-paper artifacts are the original framework implementation. Keep that date-and-scope qualifier.

## Installed code is explicitly unofficial

Installed checkout:

- Repository: `https://github.com/duyvuleo/Open-ScientistTwo`
- Git origin freshly verified as that URL.
- HEAD freshly verified: `68ef7c9948b3da091724b5759f8a43888286ab9d`
- `pyproject.toml` version freshly verified: `0.4.0`
- Git working tree was clean at this review.

`upstream/Open-ScientistTwo/README.md:3–5` states:

> “An independent, experimental implementation of the research workflow described in [ScientistTwo]. This project is unofficial and is not affiliated with the paper's authors.”

`README.md:14–17` says its default uses scripted agents, real CPU experiments and zero live model calls, and does not reproduce the paper's reported benchmark results, model mix, costs or scientific novelty. `README.md:104–107` asks users to identify the implementation as **Open-ScientistTwo**, cite its exact revision and cite the original paper separately. `THIRD_PARTY_NOTICES.md:63–67` repeats that it is not the original authors' released implementation.

Therefore, do not call the installed checkout “Google官方ScientistTwo源码”, “Google提供的Open-ScientistTwo” or an exact reproduction of the Google-led framework. The evidence supports **an independent unofficial implementation inspired by the published method**.

## The local bridge is a second, separately bounded adaptation

`bridge/README.md` and the actual `bridge.py` agree on the executed boundary:

- Reuses upstream `Store`, `ResearchPolicy`, the paper-methodology profile and three output schemas.
- Records actual Codex native-agent JSON, native-origin attestations, immutable artifacts and experiment imports.
- Replaces the upstream CLI/API transport with external/native Codex collaboration and imports genuine host MPS/CPU runs.
- Does not invoke an LLM itself, dispatch agents, launch training, implement the whole `ResearchController`, reproduce the original container/CUDA path or issue publication acceptance.
- Records `upstream_live_enabled: false`, `model_calls_made_by_bridge: 0`, and command receipts with `training_runs_started: 0`.
- Native role receipts attest agent origin plus exact output bytes; they are not cryptographic app signatures.
- The original paper's public loop ceilings are reused; the locally chosen three seed ideas and existing svproc paper template are explicit local choices.

`INSTALLATION_STATUS.json` accurately distinguishes installation/help/schema inspection from a disabled upstream container campaign. Its current scientific status records a count-geometry screen stopped at its frozen gate and imported evidence; it does not claim a successful new method. This reviewer did not rerun those campaign operations.

Recommended combined Chinese description:

> 我们参考 ScientistTwo 论文的研究循环，安装了独立非官方实现 Open-ScientistTwo，并编写本地适配层接入 Codex 原生协作与 Mac MPS 实验。当前实际完成的是原生角色输出、实验执行及证据归档；并未运行 Google 原框架或复现其论文基准。

Recommended combined English description:

> We adapted selected workflow and evidence schemas from the independent Open-ScientistTwo implementation (commit 68ef7c9), inspired by the ScientistTwo paper. A local bridge records Codex native-agent outputs and independently executed host experiments; this is not an execution or benchmark reproduction of the original ScientistTwo system.

## Review conclusion

The existing top-level `06_ScientistTwo/README.md` already uses the correct affiliation and implementation distinction. No correction is required to that README based on the checked evidence. Preserve three names with three scopes: **ScientistTwo paper**, **Open-ScientistTwo unofficial implementation**, **local Codex/native-MPS adaptation**. Source-paper benchmark scores and automated-review outcomes must not be attributed to this local FL project, and an internal review must not be described as external peer review or proof of conference readiness.

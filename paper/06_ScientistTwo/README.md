# ScientistTwo for lightweight federated learning

This workspace applies the publicly described ScientistTwo research loop to the current PR-LFL project. Its purpose is to improve the lightweight FL contribution under data heterogeneity, with real development experiments and clear attribution.

## Installed implementation and execution

- Method source: Google Cloud AI Research and University of Waterloo, **ScientistTwo: Pioneering the Human Knowledge Frontier with Autonomous AI**, [arXiv:2609.19644v1](https://arxiv.org/abs/2609.19644v1), [project page](https://scientist-two.github.io/).
- The checked official page links the paper and generated artifacts. No official framework source repository or complete prompts were identified in this bounded check on 2026-11-05.
- Installed independent implementation: [Open-ScientistTwo](https://github.com/duyvuleo/Open-ScientistTwo), version **0.4.0**, commit **68ef7c9948b3da091724b5759f8a43888286ab9d**. This is an unofficial implementation, not Google's released source or a reproduction of its benchmark results.
- Source is in `upstream/Open-ScientistTwo/`; its isolated `.venv` uses CPython 3.12.16 and the repository's frozen `uv.lock`. The project-local bootstrap contains uv 0.13.7. Existing Tinyproto training dependencies were not upgraded.
- The installed upstream CLI and workflow schemas are usable locally. The upstream container campaign is not enabled: Docker/Podman is absent, its GPU execution expects CUDA rather than Mac MPS, and the separate Codex CLI reports `Not logged in`.
- `bridge/` adapts upstream persistence and schema validation to this chat's actual Codex native agents and the existing MPS experiment runners. It imports genuine role outputs and actual experiment records. It does not claim an upstream live campaign, create synthetic agent answers, or automatically call a paid API.

## Active scientific scope

The user confirmed on 2026-11-05 that watermark/blockchain construction, attack/defence and other blockchain experiments are out of scope. Work now concentrates on lightweight FL. Existing upload material can remain historical appendix support.

The current question is: **Can a compact local predictor correct label-distribution heterogeneity after compressed shared training, and which part of that correction carries information beyond familiar regularization?**

The current supported mechanism is a class-aligned log-prior center in a convex post-hoc predictor on frozen logits. The generic idea of personalized regularization centers already appears in PRIOR/pFedBreD, so novelty must be argued at the specific information source, predictor, cost and attribution level. Neither a polished manuscript nor an internal agent acceptance establishes a new research contribution by itself.

Active run: `runs/fl_heterogeneity_20261105/`. The starting point preserves the original 13 numeric groups, existing official TEST outputs, and the taskbook's protected files. New candidate selection uses TRAIN/development data. Original official TEST outcomes are not repurposed as a development objective or silently replaced by new-partition results.

## Research loop

1. Extract actionable limitations and verify that they motivate a mechanism change.
2. Generate distinct seed ideas and compare them to actual nearby literature.
3. Establish the matched baseline; screen on declared development subsets.
4. Promote a candidate only after a full development comparison; preserve unsuccessful results.
5. Use zero/prior/permuted centers and simple competing explanations to attribute any gain.
6. Draft around the measured contribution. A review concern triggers a targeted experiment when needed.
7. Check score provenance, task compliance, citations and method/code agreement before promoting claims.

The paper's public loop ceilings are recorded separately from local execution choices. Model mixture, original prompts, 107-task performance, research costs and review scores are not reproduced. Internal review is an engineering/scientific aid and is not external peer review.

## Useful paths

- `sources/`: original-paper and official-page snapshots, source hashes and implementation identity.
- `bridge/`: local native-agent/MPS adaptation and command usage.
- `runs/fl_heterogeneity_20261105/`: hypotheses, actual role outputs, experiment imports and campaign state.
- `../03_英文论文/checks/REVISION_STATUS.md`: delivered paper revision status when published.
- `../03_英文论文/main.pdf`: delivered current12-page manuscript,33 cited references and completed selected FL evidence. The temporary stage is retained as execution provenance.

Use the isolated executable `upstream/Open-ScientistTwo/.venv/bin/scientisttwo` for upstream CLI inspection. Do not treat a scripted demonstration or a saved plan as a completed autonomous research campaign.

## Completed current revision

Dense FedAvg/FedProx, pFedMe, three-partition development and CIFAR-100 development are complete and independently checked. The active bridge records two actual roles and five experiment imports (45 artifacts; PASS). Count-geometry failed its frozen two-task development gate and was stopped; no new candidate is promoted. See `runs/fl_heterogeneity_20261105/FINAL_RESEARCH_ASSESSMENT.md` for the research conclusion, negative accuracy/ECE findings and PRIOR/LaSCal/FedNPC boundaries.

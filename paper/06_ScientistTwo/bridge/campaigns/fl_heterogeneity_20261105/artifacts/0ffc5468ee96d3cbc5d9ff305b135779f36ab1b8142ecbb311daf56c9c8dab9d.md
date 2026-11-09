# Lightweight FL under heterogeneous client labels

Run initialized 2026-11-05 (Asia/Shanghai). Execution is a native-Codex/MPS adaptation of the ScientistTwo public methodology.

## Objective and research unit

Improve the mechanism for compact client prediction when clients have different label distributions. A useful outcome is a defensible lightweight contribution, limited decisive validation and clear conference-paper writing. Installing an agent framework is infrastructure for this outcome, not a scientific result.

The current PR-LFL method freezes the shared CNN, fits ten scales and ten centered biases, and shrinks the bias toward a centered log prior from local TRAIN labels. The same 80 B prediction state accompanies compressed shared uploads of 54,067 B. The primary scientific endpoint is client-macro NLL; accuracy, weighted metrics and calibration remain reported to show trade-offs.

The starting evidence covers label heterogeneity. It does not establish a general solution to feature heterogeneity, concept shift or arbitrary non-IID distributions. Candidate hypotheses must state the heterogeneity they address.

## Protected starting point

- Original 13 numeric groups, full six-seed table, old evidence and check files, class/BST, main preamble and bibliography entries remain protected under the taskbook.
- Original official TEST figures remain fixed. Candidate search uses TRAIN/development holdouts. New baselines freeze all fitting before final scoring.
- Preserve adverse results: weighted CIFAR accuracy, Fashion q10, ECE relative to BCTS, one negative CIFAR accuracy seed, and the adverse frozen-predictor reallocation diagnostic.
- Generic personalized prior/regularization centers are prior art. PRIOR/pFedBreD uses centers in model-parameter training; our candidate must explain its specific output-space information and mechanism.
- No watermark, blockchain, chain construction or attack/defence experiments.

## Immediate experiment queue

| Block | Scientific purpose | Execution state at initialization |
|---|---|---|
| Dense FedAvg/FedProx, original splits and seeds | Establish basic shared-training references and separate raw from added calibration | 12/12 training runs completed; TRAIN-only predictor fits underway |
| Two additional partitions × three seeds × two tasks | Test whether prior-center NLL improvements persist across partition realizations | MPS matrix launched; development-only evaluation |
| pFedMe, original splits and seeds | Compare against a published personalized training mechanism at explicit compute cost | Source, optimizer fixture and data preflight passed; queued after partition matrix |
| Mechanism candidates | Address a demonstrated weakness with a compact change and identifiable cause | Native limitation/idea generation underway; no claimed new result |

## Experiment decisions

Do not select a candidate because it wins a single convenient seed or metric. Start with the predeclared two tasks, all existing development seeds and a matched information/budget control. Keep all candidates, including failed ones, in the campaign. A new mechanism must survive an attribution control against familiar shrinkage, parameter count, center alignment and calibration as appropriate. Favor one interpretable improvement over a bundle of training tricks.

The initial paper-revision task remains active. Results are integrated only after complete scoring and review, and page/table constraints still apply. Unexecuted P1/P2 items are recorded in the paper's `checks/REVISION_STATUS.md`, not described as experimental findings in its body.

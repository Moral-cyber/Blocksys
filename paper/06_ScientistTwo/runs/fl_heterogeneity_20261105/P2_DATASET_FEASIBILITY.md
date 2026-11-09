# P2 additional-dataset feasibility

Read-only assessment, 2026-11-05. No data download, checkpoint load, CNN construction/training, candidate fit or manuscript edit was performed for this assessment. Taskbook §5.3 asks for **CIFAR-100 or FEMNIST** as one additional dataset. Neither result presently exists in the inspected workspace, so this remains an uncompleted P2 item until an actual experiment is executed and scored.

## Finding

**CIFAR-100 is the smaller, technically feasible extension, but it is not immediately runnable with the frozen 10-class entry point.** The data are not local in the inspected paper/Codex/DeepSeek Harness/cache file inventories. The installed `torchvision.datasets.CIFAR100` documents an official Toronto archive and integrity checks; HTTP availability and actual download completion were not verified here. A separate K-class adapter, new model runs and a memory-conscious calibration implementation are required. Do not report P2 as complete on the basis of loader availability.

FEMNIST is a larger change because meaningful federated evaluation needs writer identities and an explicit writer/client protocol. Installed torchvision **EMNIST is not FEMNIST**; its class-balanced/byclass image splits do not by themselves reproduce writer-partitioned FEMNIST. No local LEAF/FEMNIST runner, writer metadata, raw data or prepared federated shards were located by the filename/code search. The exact LEAF version, source archive and preprocessing pipeline need primary-source confirmation before use.

## Local evidence and official data route

| Item | Current evidence |
|---|---|
| Local CIFAR-10 | Present under `02_实验工程/dataset/Cifar10/rawdata/cifar-10-batches-py/`; this is not CIFAR-100. |
| Local FashionMNIST | Present in the existing experiment directories; this is not FEMNIST. |
| Local CIFAR-100/FEMNIST | No matching data/runner filenames found in the inspected `/Users/wind/Desktop/Codex`, `/Users/wind/Desktop/DeepSeek Harness` and `/Users/wind/.cache` inventories. This is a scoped search, not a claim about every disk or remote store. |
| CIFAR-100 official archive | Installed source `/opt/anaconda3/envs/Tinyproto/lib/python3.13/site-packages/torchvision/datasets/cifar.py`, `CIFAR100`: <https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz>. `tgz_md5=eb9058c3a382ffc7106e4002c42a8d85`. No network retrieval in this assessment. |
| CIFAR-100 extracted checks | `train` MD5 `16019d7e3df5f24257cddd939b257f8d`; `test` MD5 `f0ef6b0ae62326f3e7ffdfab6717acfc`; metadata MD5 `7973b15100ade9c7d40fb424638fde48`. They are specifications read from the installed loader, not checks performed on downloaded files. |
| FEMNIST route | LEAF's primary repository/project is the source to verify for writer-aware processing. No official package or current download URL was fetched in this read-only assessment. |
| torchvision EMNIST | The installed MNIST module provides NIST `gzip.zip` and six image splits. None is named FEMNIST; substituting it would change the requested dataset. |

## K-class prediction and communication accounting

The existing `SmallCNN` constructor in `02_实验工程/tools/fedavg_update_diagnostic.py` accepts `classes=10` and uses `Linear(64, classes)` at the final layer. The 100-class form is therefore structurally available without changing the feature trunk, but the actual frozen runner still calls `SmallCNN()` and validates exactly 136,874 parameters.

For the same trunk, `P(K)=136224+65K`; these are source-derived calculations, not a trained model or measured payload:

| Quantity | K=10 existing instance | K=100 proposed instance |
|---|---:|---:|
| Shared CNN parameters | 136,874 | 142,724 |
| Stored scale+bias coordinates | 20 | 200 |
| Stored FP32 predictor bytes | 80 B | **800 B** |
| Independent centered-bias free coordinates | 9 | 99 |
| FP32 EF residual bytes | 547,496 B | 570,896 B |
| Analytical dense upload, 64 B header included | 547,560 B | 570,960 B |
| Uniform3 wire-size formula result | 54,067 B | **56,376 B** |

The codec's exact formula is `64 + 5*ceil(P/256) + 96*floor(P/256) + ceil(3*(P mod 256)/8)`. The old fixed `CAP=54808` would reject the 100-class three-bit payload. The model parameter count, cap, residual checks and new communication accounting must be bound to the new adapter. Preserve all original 80 B/54,067 B figures; the extension needs a separate dataset-specific row. A 62-class FEMNIST instance would instead store 124 FP32 values = 496 B, if a verified 62-class protocol is chosen.

## Minimal implementation plan

1. **Add an isolated CIFAR-100 loader and manifest.** Use official `fine_labels` (100 classes), 50,000 training images, the same 32×32 RGB normalization, and an explicit TRAIN-only development path. Verify archive/extracted hashes and row/class counts. The archive contains a TEST file: acquiring it does not authorize using it for selection. The existing guard that only recognizes `test_batch` is insufficient for CIFAR-100's filename `test`; guard exact resolved paths and reject its reads during fitting/search.
2. **Generalize the partition adapter without editing frozen originals.** The classwise Dirichlet-multinomial algorithm, 20 clients, alpha 0.1, minimum 100 total samples, local 75/25 split, exact coverage/disjointness and stratified 2,000-row monitor are reusable concepts. Existing `run_partitions.py` already uses `NUM_CLASSES` in much of its loop, but its loader, label checks, identifiers and original source pins are CIFAR-10-specific. With K=100, stratification targets 20 monitor examples/class; verify feasibility after partitioning instead of assuming every holdout class has enough rows.
3. **Use `SmallCNN(classes=100)` in a fresh runner context.** Reuse the same SGD/local epoch/round protocol, sample-count aggregation and unchanged codec. Generalize all class-count vectors, CE shifts and checkpoint parameter-layout checks. The old sources contain literal 10, 20, slice boundaries and fixed dimensions, so this is not just changing one constructor argument.
4. **Implement K-class affine calibration with structured derivatives.** Store `a,b` with K entries each and `sum(b)=0`; derive the same convex CE plus fixed ridge objective and local `center(log(n+1))`. Do not merely replace `CLASSES=10`: the current implementation explicitly builds an `N×K×2K` FP64 design tensor. At K=100 this is 160,000 bytes per TRAIN example (400 MB for N=2,500), 100 times the K=10 design cost. Use diagonal-scale structure and batched accumulation of the `2K×2K` Hessian; a 200×200 FP64 Hessian itself is only 320,000 B. Replay K=10 fixtures before new scores, plus K=100 finite-difference/centering/FP32-state checks.
5. **Freeze a bounded evidence plan before any new scores.** One new partition, three predeclared model seeds and the main compressed shared training suffice for the requested first additional-dataset check; evaluate raw, fixed prior addition, zero/prior affine fits and a compact BCTS rival on the same checkpoints/information. Report all clients/seeds and real fitting cost. Do not select the backbone, lambda or the retained seed by official TEST. A one-seed preflight can check resources but cannot be reported as a three-seed P2 result.

This is a feasible follow-up, not a promise that 20 rounds of the small trunk will yield a useful 100-class predictor or that the existing NLL advantage will persist. If it is not executed in this revision, record **NOT_RUN: data unavailable locally; K-class adapter and structured calibration required** in `checks/REVISION_STATUS.md`, with no unexecuted outcome in the paper body.

## Later implementation update: source/fixtures ready, model runs pending

After this initial feasibility assessment, root authorized the isolated CIFAR-100 implementation. It is now prepared under `/private/tmp/pr_lfl_route_a/03_英文论文/revision_artifacts/route_a_20261105/cifar100_extension/`: plan, source-locked adapter, K-parameterized structured solver, sequential three-run launcher and fixtures. Six invented K=10 zero/prior cases replay the old deployed FP32 states bitwise; K=100 finite differences, positive Hessian curvature, centered 800 B states and probability-metric fixtures pass. Synthetic partition coverage/disjointness also passes. These are numerical/preflight checks, not CIFAR-100 task results. Actual official TRAIN data preflight and all three CNN runs remain pending root's download/MPS queue at this update.

A deeper hidden-directory search also located a legacy `generate_Cifar100.py` in `/Users/wind/Desktop/DeepSeek Harness/.work/repo/FedCSPACK-main/dataset/`. It is not a ready matched runner: it combines official TRAIN and TEST before repartitioning, so it must **not** be used for this extension's TRAIN-only protocol. No actual CIFAR-100/FEMNIST data were found by that deeper search. This corrects the earlier shallow inventory's lack of a legacy generator; it does not change the absent-data finding.

Root subsequently acquired the official-byte CIFAR-100 archive through the Paddle-documented BOS mirror. `02_实验工程/dataset/Cifar100/rawdata/ACQUISITION.json` records the primary loader source, archive MD5 matching torchvision (`eb9058c3a382ffc7106e4002c42a8d85`) and SHA-256. Only `train` and `meta` were extracted; official TEST remains unextracted. Actual TRAIN data preflight passed: 37,500/12,500 client TRAIN/holdout rows, exact disjoint coverage, 100 fine classes ×500 original rows and 20 monitor rows/class. An independent read of TRAIN labels and source/data hashes confirmed those checks. This supersedes the earlier missing-data condition; all three actual model/evaluation runs are still pending root's MPS queue at this update. A data-ready preflight is not a completed P2 result.

## Completion update, 2026-11-06

All three predeclared CIFAR-100 model runs and development evaluations have now completed, with independent saved-array/state replay passing. Client-macro zero/prior NLL is2.428310/2.289578 (mean difference−0.138731); all3runs and60client-seed cells improve. Mean accuracy decreases by0.1460percentage points and client-mean ECE worsens slightly; complete adverse/secondary metrics are retained. `cifar100_extension/ACCEPTANCE.md`, `formatted_development.json`, `client_metrics.csv`, compact LaTeX and `INDEPENDENT_RESULTS_VERIFICATION.json` contain the actual results and protocol. The additional-dataset development experiment is therefore complete. It remains one partition with TRAIN fitting/development scoring, not official TEST or general data-heterogeneity evidence. Earlier missing-data/adapter descriptions above are dated history rather than current blockers.

# PR-LFL: Prior as Center

Research repository for:

> **Prior as Center: Compact Prediction and Ledger-Anchored Uploads for Heterogeneous Federated Learning**

Two halves of one system: a 20-parameter prior-centered predictor fitted on frozen shared logits, and a ledger-anchored upload path that binds compressed updates to their round context and gates error-feedback residuals on an authenticated incorporation receipt.

当前 `main.pdf`：14 页（含参考文献），正文 p1–12。

## 目录

| 路径 | 内容 |
|---|---|
| `paper/03_英文论文/` | LaTeX 源码、图表、`main.pdf`、`evidence/`、`checks/`、筛选后的 `revision_artifacts/` |
| `paper/02_实验工程/` | 三比特编解码、误差反馈与诊断脚本、小体积结果 |
| `paper/05_核心方法研发/` | 方法脚本、检查记录、设计说明（不含 checkpoint 与 results） |
| `paper/00_使用说明/` | 研究状态说明 |
| `00_文件总导航.md` | 仓库结构导航、排除清单与历史迁移记录 |

## 编译

```bash
cd paper/03_英文论文
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

产出 `main.pdf`，14 页（含参考文献）。本仓库副本已在干净目录中验证可编译，`\input` 与 `\includegraphics` 全部可解析。

## 未收录

| 内容 | 体积 | 原因 |
|---|---|---|
| `paper/02_实验工程/dataset/` | 680 MB | CIFAR 原始数据，可公开下载 |
| `paper/05_核心方法研发/results/` | 2.3 GB | 逐运行 `result.json`（单个 60–70 MB）与 checkpoint |
| `paper/06_ScientistTwo/` | 2.0 MB | 第三方研究工具（Open-ScientistTwo 非官方实现），不属于论文项目 |
| `paper/03_英文论文/English_Paper_Complete_Package.zip` | 330 MB | 打包副本 |
| `paper/03_英文论文/revision_artifacts/**/runs/` | ~470 MB | 训练 checkpoint（`*.pt`）与中间缓存 |
| `对话产物完整归档_20261103/` | 7.6 GB | 完整对话与目录快照（含 6.4 GB 删除前备份） |

这些仍保留在原始 `Desktop/Codex/` 目录；需要版本化时建议用 Git LFS 或 release 附件。

## 说明

- 所有数字来自 `evidence/` 与 `revision_artifacts/` 中的存档，未在本仓库重新生成。
- 链上测量在本地单节点 Ganache 开发链上完成，不代表公链共识或最终性。
- 结果基于 3 个固定种子；60 个 client–seed 对共享同一划分与模型，不是独立训练重复。

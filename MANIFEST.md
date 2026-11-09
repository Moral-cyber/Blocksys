# 整理清单

- 生成时间：2026-11-09
- 源目录：`/Users/wind/Desktop/Codex`
- 目标目录：`/Users/wind/Desktop/Codex_repo`
- 收录：**1041 个文件，17.3 MB**（单文件上限 2 MB）

## 分区统计

| 区域 | 文件数 | 体积 |
|---|---:|---:|
| paper/03_英文论文（源码、图表、evidence、checks） | 62 | 3.4 MB |
| paper/03_英文论文/revision_artifacts（筛选后） | 256 | 5.7 MB |
| paper/02_实验工程（代码与小结果） | 15 | 0.8 MB |
| paper/05_核心方法研发（脚本与检查） | 699 | 7.5 MB |
| paper/00_使用说明 | 3 | 0.0 MB |
| 根目录文件 | 5 | 0.0 MB |

## 收录规则

- 只保留文本与小型二进制：`.tex .bib .cls .bst .md .py .csv .json .txt .pdf .png .svg`
- 排除：`*.pt`、`*.npz`、`*.sqlite`、`*.zip`、`__pycache__`、`runs/`、`e1_runs/`、`dataset/`、`results/`、`upstream/`
- `revision_artifacts/` 仅保留：表格 `.tex`、`figures/` 下的图、脚本、报告，以及白名单摘要 JSON（λ 扫描、α 扫描、链上成本、CIFAR-100、pFedMe、漂移诊断等）
- 保留原目录编号（`02_`、`03_`、`05_`），不重命名，避免破坏文档里的相对路径
- **`paper/06_ScientistTwo/` 整体未收录**：它是第三方研究工具（Open-ScientistTwo，非 Google 官方实现），不属于本论文项目内容
- 已剔除 `paper/06_ScientistTwo/.bootstrap/`（虚拟环境内置 pip）

## 校验

- 在干净目录用仓库副本编译：`Output written on main.pdf (14 pages)`，与原文一致
- `\input` 15 处、`\includegraphics` 4 处全部存在，无缺失
- 关闭 Git 行尾转换（`.gitattributes: * -text`），抽样 `cmp` 与工作区字节一致
- 无 `.DS_Store`、无 `__pycache__`、无构建中间文件

## 最大的入选文件

| 文件 | 体积 |
|---|---:|
| `paper/03_英文论文/evidence/PAPER_NUMBERS.json` | 1.8 MB |
| `paper/03_英文论文/revision_artifacts/route_a_20261105/p1/lambda_result.json` | 0.9 MB |
| `paper/02_实验工程/results/sparse_clean_20c_15r_3s.csv` | 0.7 MB |

## 未收录（仍在原目录）

| 内容 | 体积 |
|---|---:|
| `paper/05_核心方法研发/results/` | 2.3 GB |
| `对话产物完整归档_20261103/09_删除原目录前完整快照_20261104/` | 6.4 GB |
| `paper/02_实验工程/dataset/` | 680 MB |
| `paper/03_英文论文/revision_artifacts/**/runs/` | ~470 MB |
| `paper/03_英文论文/English_Paper_Complete_Package.zip` | 330 MB |
| `paper/06_ScientistTwo/`（第三方工具，含 upstream） | 259 MB |
| `对话产物完整归档_20261103/02_实验工程完整快照/` | 1.2 GB |
| `paper/01_论文材料/历史论文版本/` | 9.7 MB |

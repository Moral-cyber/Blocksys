# PR-LFL: Prior as Center

**Prior as Center: Compact Prediction and Ledger-Anchored Uploads for Heterogeneous Federated Learning**

异构联邦学习的一条双线系统：冻结共享 logits 上的 20 参数先验中心预测器，以及把三比特压缩上传绑定到轮次上下文、以认证回执为门槛提交误差反馈残差的链上锚定路径。

---

## 四个入口

| 想看什么 | 位置 |
|---|---|
| **论文 PDF（14 页）** | [`paper/03_英文论文/main.pdf`](paper/03_英文论文/main.pdf) |
| **上链合约与协议** | [`paper/05_核心方法研发/scripts/chain_integration/`](paper/05_核心方法研发/scripts/chain_integration/) |
| **实验证据（JSON）** | [`paper/03_英文论文/revision_artifacts/`](paper/03_英文论文/revision_artifacts/) |
| **论文源码** | [`paper/03_英文论文/main.tex`](paper/03_英文论文/main.tex) + `sections/` + `tables/` |

## 按需求查

| 查什么 | 去哪 |
|---|---|
| 上链的 8 个字段 | [`UploadRegistry.sol`](paper/05_核心方法研发/scripts/chain_integration/UploadRegistry.sol) 的 `struct Binding` |
| 填这些字段的代码 | [`network_service.py`](paper/05_核心方法研发/scripts/chain_integration/network_service.py) 第 75 行 `binding()` |
| 链上运行入口 | [`run_learning.py`](paper/05_核心方法研发/scripts/chain_integration/run_learning.py) + [`ledger_bridge.cjs`](paper/05_核心方法研发/scripts/chain_integration/ledger_bridge.cjs) |
| 一次真实链上运行的记录（合约地址、交易哈希、gas） | [`results/chain_integration/result.json`](paper/05_核心方法研发/results/chain_integration/result.json) |
| 上传实验（字节 / gas / 恢复） | [`e3_chain_cost.json`](paper/03_英文论文/revision_artifacts/v4_dualtrack/e3_chain_cost.json) |
| 主结果（表 1） | [`stats/result.json`](paper/03_英文论文/revision_artifacts/route_a_20261105/stats/result.json) |
| 异构扫描 E1 | [`e1_alpha_sweep.json`](paper/03_英文论文/revision_artifacts/v4_dualtrack/e1_alpha_sweep.json) |
| 客户端规模 E2 | [`e2_skew_gain.json`](paper/03_英文论文/revision_artifacts/v4_dualtrack/e2_skew_gain.json) |
| 论文图表 | [`paper/03_英文论文/figures/`](paper/03_英文论文/figures/) |
| 编解码、误差反馈与诊断脚本 | [`paper/02_实验工程/`](paper/02_实验工程/) |
| 方法与检查记录 | [`paper/05_核心方法研发/`](paper/05_核心方法研发/) |

## 目录

```
paper/
├── 03_英文论文/          论文源码、图表、main.pdf、证据（revision_artifacts/）、编译检查
├── 02_实验工程/          三比特编解码、误差反馈、诊断脚本与小体积结果
└── 05_核心方法研发/      方法脚本、上链集成、设计说明与检查记录
```

## 状态

- 目标会议：BlockSys 2027 第一轮（Springer CCIS），截稿 2026-11-30
- 论文：14 页，两条线（标签异构 + 链上锚定），34 条被引
- 仓库：1041 个文件，17.3 MB；单文件上限 2 MB，因此不含 checkpoint、数据集与缓存
- 证据文件按实验分组，每个 JSON 内记录输入哈希与统计口径

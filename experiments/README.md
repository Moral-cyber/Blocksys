# experiments — 两条实验线的可复现入口

本目录把论文《Prior as Center: Compact Prediction and Ledger-Anchored Uploads for
Heterogeneous Federated Learning》的实验拆成**两条独立的线**，每条线一个文件夹：

| 目录 | 对应论文 | 一句话 |
|---|---|---|
| [`PR-LFL/`](PR-LFL/README.md) | §3–§5、Table 1、Fig. 2 | 标签异构下，用 20 个参数把预测偏置拉向本地类别先验；收益随异构增强 |
| [`chain/`](chain/README.md) | §6–§7、Table 3–5、Fig. 3–4 | 三比特压缩上传绑定轮次上下文与链上登记，回执后才提交误差反馈残差 |

两条线共用同一个前提：**客户端数据分布不同、上传必须压缩**。除此之外互不依赖——
`chain/` 不需要 PyTorch 和数据集就能跑「登记 + 校验」，可以直接交给外部合作方。

## 快速开始

```bash
# 第一条线：复现表格与图（需要 numpy/scipy/matplotlib，几秒到几分钟）
cd PR-LFL
pip install -r requirements.txt
python t1_main_table.py        # → results/t1_main_table.json
python f2a_alpha_sweep.py      # → results/f2a_alpha_sweep.pdf
python f2b_client_size.py      # → results/f2b_client_size.pdf
python s5_4_permutation.py     # → results/s5_4_permutation.json

# 第二条线：导出上传记录 → 登记到本地链 → 校验（需要 Node 20+）
cd ../chain
node --version                 # 需要 20 以上
npm ci                         # 安装 ganache / solc / ethers（版本已锁定）
python x1_export_uploads.py    # → data/uploads.jsonl（19 条）
node x2_register_on_chain.mjs  # → results/x2_registration.json
python x3_verify_receipts.py   # → results/x3_verification.json（PASS）
```

## 已在本机验证

| 项 | 结果 |
|---|---|
| Table 1 主结果 | NLL 差值 −0.03575086 / −0.01823567，60/60 配对改善，与论文一致 |
| Fig. 2(a) α 扫描 | 0.0408 / 0.0101 / 0.0042（CIFAR），0.0132 / 0.0047 / 0.0024（Fashion），与论文一致 |
| Fig. 2(b) 样本量 | Spearman −0.771（CIFAR）/ −0.729（Fashion），与论文一致 |
| §5.4 置换对照 | 直接复用已保存的诊断结果（上游脚本拒绝覆盖已完成诊断） |
| 链上导出 | 19 条（2×2 共 4 条 + 5×3 共 15 条） |
| 链上登记 | 全部 readback 一致；gas 1,140,380（2×2）/ 4,229,442（5×3，与论文差 36 gas，源于账户与存储状态差异） |

## 约定

- **不修改 `paper/`**。两条线都是「新建目录 + 调用上游实现」，原始论文目录保持原样。
- 每个脚本头部都有注释说明：对应论文哪一段、输入是什么、输出是什么、怎么运行。
- 结果文件都带 `_meta`（时间戳、脚本哈希、输入哈希），见 `PR-LFL/io_schema.py`。

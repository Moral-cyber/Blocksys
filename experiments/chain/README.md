# 链上实验线（压缩上传的链上锚定）

对应论文 §6–§7：Table 3（字段划分）、Table 4（字节与 gas）、Table 5（恢复）、Fig. 3–4。

一句话：**每次上传的三比特更新，把「它是什么、属于哪一轮哪个客户端、进了哪个模型状态」
绑定成 8 个 32 字节字段，登记到链上；客户端只有拿到「已纳入」的回执，才提交误差反馈残差。**

这条线可以**完全脱离训练和数据独立运行**：`data/uploads.jsonl` 里已经放好 19 条真实上传记录
（2 客户端 × 2 轮 4 条，5 客户端 × 3 轮 15 条），外部合作方只需要 Node 就能完成登记与校验。

## 文件清单

| 文件 | 用途 |
|---|---|
| `data/uploads.jsonl` | **核心交付物**：19 条上传记录，每行含 8 个绑定字段 + 链上交易信息 |
| `data/raw/e3_receipts.jsonl` | 原始记录（x1 的输入），由论文 E3 实验产生 |
| `x1_export_uploads.py` | 把原始记录归一化成 `uploads.jsonl`；只读、不连链 |
| `x2_register_on_chain.mjs` | 读 `uploads.jsonl`，在本地 Ganache 部署合约并逐条登记、读回比对 |
| `x3_verify_receipts.py` | 静态校验字段格式/唯一性，并与 x2 的登记结果交叉核对 |
| `contracts/UploadRegistry.sol` | 合约：`struct Binding` 定义 8 个上链字段 + uploader/blockNumber |
| `bridge/ledger_bridge.cjs` | 论文原版桥接脚本（部署 Ganache、代理 JSON-RPC），保留作对照 |
| `common/protocol.py` | 上传包的序列化与 HMAC-SM3（`Packet` / `Envelope`） |
| `common/network_service.py` | `binding()` 生成 8 个字段；回执验证；上传/查询服务 |
| `common/run_learning.py` | 论文原版的端到端入口（真实训练 + HTTP + 链上登记） |
| `results/` | x2/x3 的输出 |

## 快速开始（不需要 Python 环境、不需要数据）

```bash
npm ci                          # 安装 ganache 7.9.2 / solc 0.8.30 / ethers 6.15.0
python x1_export_uploads.py     # → data/uploads.jsonl（已预生成，可跳过）
node x2_register_on_chain.mjs   # → results/x2_registration.json
python x3_verify_receipts.py    # → results/x3_verification.json，应为 PASS
```

本机已验证：19 条全部登记成功、readback 全一致；
gas 总量 1,140,380（2×2）/ 4,229,442（5×3）。论文 Table 4 记的是 1,140,380 / 4,229,478，
36 gas 的差异来自账户编号与合约存储状态，不是数据不一致。

## uploads.jsonl 的一行是什么

```json
{
  "schema": "pr-lfl-upload/1",
  "configuration": "c2_r2_measured",     // 哪次活动：2 客户端 2 轮 / 5 客户端 3 轮
  "round": 1, "client": 0,
  "record_id": "r1-c0",                  // 记录名，合约键 = keccak256(utf8(record_id))
  "record_key_rule": "keccak256(utf8(record_id))",
  "binding": {                           // 8 个上链字段，全部 32 字节
    "activity_hash": "0x…", "client_hash": "0x…", "round_id": 1,
    "nonce_hash": "0x…", "base_model_hash": "0x…",
    "layout_hash": "0x…", "codec_spec_sha256": "0x…", "digest": "0x…"
  },
  "chain": { "tx_hash": "0x…", "block_number": 2, "gas_used": 297926, … },
  "bytes": { "payload": 54067, "envelope_and_tag": 54408 }
}
```

## ⚠️ 字段映射（对接外部链时最容易错的地方）

这些字段的「字符串 → bytes32」不是同一种哈希：

| 合约字段 | 来源 | 编码 |
|---|---|---|
| `record`（mapping key） | `record_id` 字符串 | **keccak256(utf8)**，即 ethers 的 `id()` |
| `activity` | `activity_hash` | 已是 32 字节摘要，**原样写入** |
| `client` | `client_hash` | 同上，原样 |
| `round` | `round_id` | `uint64` |
| `nonce` | `nonce_hash` | 原样 |
| `baseModel` | `base_model_hash` | 原样（模型参数的 SM3） |
| `layout` | `layout_hash` | 原样 |
| `codecSpec` | `codec_spec_sha256` | 原样（编解码器规格文件的 SHA-256） |
| `digest` | `digest` | 原样（`SM3(wire)`，**不需要密钥就能复算**） |
| `uploader` / `blockNumber` | — | 合约派生，调用方必须传 0 |

对接方如果「全部用 keccak」或「全部用 sha256」，两边一定对不上——按上表逐列实现。

## 这套机制证明什么、不证明什么

**证明**：某条上传的字节与上下文在某个区块被登记过；客户端拿到的回执与登记记录、模型状态一致；
重试/重复/丢失响应后状态可恢复（论文 Table 5 的四类故障）。

**不证明**：训练本身正确；模型没有投毒；链上记录独立于服务器（协调方仍签发回执）；
多节点共识或公链最终性（本实验是单节点 Ganache 开发链）。

## 对接到外部链要改什么

1. `x2_register_on_chain.mjs` 当前在进程内启动 Ganache（单节点）。对接外部链时，把
   `new ethers.JsonRpcProvider(...)` 换成对方 RPC，并把 `getSigner(i)` 换成持有私钥的
   `new ethers.Wallet(key, provider)`（改动约 10 行，其余逻辑不变）；
2. 若对方的链不是 EVM 兼容（如 Hyperledger Fabric），只需按上面的字段映射重写登记端，
   `uploads.jsonl` 和字段语义不变；
3. 对方回传交易哈希后，用 `x3_verify_receipts.py --registration <对方的结果>` 做交叉核对。

# 本地持久化登记、可信接纳与 EF 提交测试

本次实现并实际运行了一个有限的 SQLite 本地组件，补充压缩载荷认证与 EF 提交时序的协议证据。**52/52 项 synthetic protocol tests 通过，失败 0 项。**所有输入是极小合成向量，不涉及训练或 ML 性能结论。这部分属于协议与实现证据，不属于仍在研发的 AI 方法贡献。

## 组件与不变量

- `protocol.py`：HMAC-SM3 认证封装、SQLite 模拟登记账本、可信服务端接纳及聚合、独立客户端 SQLite EF pending/commit。SQLite 采用 WAL、`synchronous=FULL`，每次操作独立打开连接。
- 每个认证包绑定 activity、client、round、RecordID、nonce、base model hash、layout、codec 和**完整 payload bytes**。数据包摘要使用 SM3，认证标签使用 32 字节 HMAC-SM3。本地测试密钥随机生成，保存在各用例的本地文件中，文件权限为 0600；结果 JSON 不包含密钥。
- 服务端状态为 `REGISTERED → ACCEPTED → INCORPORATED`。仅登记或已接纳均不足以提交客户端 EF。固定完整参与名单为 c1、c2，已知权重分别为 0.25、0.75，不计算或创新动态聚合权重。
- 最终化使用一个 `BEGIN IMMEDIATE` 事务，同时更新全局模型、累计更新量、轮次结果和记录的 `INCORPORATED` 状态。相同 RecordID 和相同认证字节的重发返回原状态；相同 RecordID、不同字节被拒绝。nonce 对 activity/client 唯一；同一客户端每轮最多一个已接纳贡献。
- 客户端先持久化精确认证字节及 `r_next = update + r_current − decode(payload)`，保持 `r_current` 不变。仅查询到同 RecordID、同 digest 的 `INCORPORATED` 后，在一个本地客户端事务中一次性复制预存 `r_next` 并增加 EF generation。未知状态只允许查询和重发已存字节，同 nonce 不重新生成新贡献。
- base model hash 表示模型版本，layout 表示稳定的坐标语义。轮次推进导致 base hash 改变时，布局相同的 EF 残差保持并继续参与下一次编码。

## 实际执行结果

运行命令：

```sh
/opt/anaconda3/envs/Tinyproto/bin/python work/durable_integrity/test_protocol.py --output-dir work/durable_integrity/results
```

Python 3.13.15，SQLite 3.53.4；测试套件用时约 1.461 秒。该时间仅说明本次测试执行情况，不能作为链、网络或认证吞吐量指标。重复运行请指定新的输出目录，程序会保留已有证据而拒绝覆盖。

| 测试组 | 项数 | 结果 |
|---|---:|---|
| 原标签下改动数值码、索引、位宽、scale、载荷长度或绑定元数据 | 14 | 全部拒绝 |
| 正确重签名但 activity、固定客户端名单、round、base、layout、codec 上下文不符 | 6 | 全部拒绝接纳 |
| 正确重签名但索引越界/重复、位宽/scale 不支持或载荷截断 | 5 | 全部由解码检查拒绝 |
| 有效聚合、三态、pending、重放、碰撞、持久化、事务恢复、并发、实际编码器及信任边界 | 27 | 全部符合断言 |

关键证据：

1. 16 个线程并发重发两份相同贡献并进行接纳、最终化和客户端提交，只有 **1 次新最终化**，两个客户端各 **1 次 EF 提交**；全局模型等于固定已知权重和，不发生二次聚合。
2. 子进程在执行全局模型 SQL 更新、尚未更新记录状态及提交事务时直接 `os._exit(73)`。重开数据库后，模型、累计更新量和最终化次数与中断前相同，记录仍为 `ACCEPTED`；随后重新最终化成功。另有普通异常回滚用例。
3. 服务端最终化后丢弃应答，客户端保留 pending。重新打开服务端及客户端数据库后，精确重发原字节、查询到匹配 digest 的 `INCORPORATED`，提交预存残差一次；再次查询/提交不重复修改残差。
4. 分别重新打开数据库验证 `REGISTERED`、`ACCEPTED`、`INCORPORATED` 及客户端 pending/EF 的持久性。共保存 **158 个 SQLite 文件**；随后独立对每个文件执行 `PRAGMA integrity_check`，158/158 为 `ok`。52 份测试密钥文件的权限均为 0600。
5. 直接只读加载现有 `work/mixed_precision/codec.py`，编码一个 4 维合成向量，得到 **71 字节真实编码器载荷**，完成认证、持久化、严格解码、接纳、聚合及 EF 提交往返。该用例验证接口与实际字节的结合，不证明训练效果。
6. 旧 `integrity_attack_matrix.py` 和冻结 `mixed_precision/codec.py` 在最终测试运行前后的 SHA-256 完全一致；本任务未修改这些源文件。

逐项机器结果见 `results/test_results.json`；所有落盘数据库检查见 `results/durability_audit.json`；数据库及本地测试密钥保存在 `results/synthetic_cases/`。`results_run1/` 保留了补充边界用例之前的 46 项开发运行，正式结果以 `results/` 的 52 项为准。

## 信任边界与论文可用表述

服务端及客户端共享 HMAC 密钥，服务端、其数据库和查询结果均被信任。这保护的是无密钥传输篡改的检测和受信本地状态机下的重复处理约束。专门的边界用例证明：持有密钥的一方可以签出更改后的有害更新并被接纳和纳入；原客户端虽会因 digest 不一致而拒绝提交自己的旧 pending 残差，已经改变的全局模型仍不具备投毒防护。HMAC 也不提供面向共享密钥持有者的不可否认性。

论文可表述为：“在本地 SQLite 持久化协议测试中，认证绑定、三状态提交约束、事务中断恢复及重复消息幂等处理通过合成用例验证。”应把“最多一次纳入”限定在该受信本地数据库和被测重试模型中；完成恢复依赖客户端最终重试查询。

登记账本只是本地模拟。没有区块链节点、共识、网络、真实许可链性能测量，也没有跨服务端与客户端数据库的端到端分布式原子事务。进程中断测试不等于硬件掉电、磁盘故障或任意分布式故障验证。测试不支持隐私、投毒防护、动态掉线聚合、AI 方法有效性或 ML 性能的主张。

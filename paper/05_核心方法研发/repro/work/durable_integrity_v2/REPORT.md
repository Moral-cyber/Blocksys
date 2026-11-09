# v2：绑定 codec 规范身份的本地持久化协议测试

已针对独立核查发现的“同名 codec 配置语义漂移”完成 v2 修复，并实际通过 **62/62 项 synthetic protocol tests，0 失败**。正式机器结果为 `results_final/test_results.json`。本目录独立于 v1；v1 源码、报告、正式结果、旧 toy 和冻结量化编码器均保持原字节，其运行前后摘要记录在正式 JSON 中。

## 修复及身份含义

v1 认证了 codec 名称，客户端与服务端仍可能为同名配置声明不同的解码语义。v2 在 `Packet` 增加 `codec_spec_sha256`，封装版本改为 `DURINT02`；HMAC-SM3 与包摘要均覆盖该身份字段和完整 payload bytes。v2 使用全新 SQLite 数据库，不对 v1 数据库作原地迁移。

两份可读 wire 规范在 `specs/` 中，包含明确的 `wire_spec_version`。其精确文件字节的 SHA256 固定在只读 `SPEC_SHA256` 中；装载规范时重新计算并核对固定值。`Codec` 配置为冻结对象，服务端注册表为只读映射。活动创建时把 codec 名称和 spec SHA256 一起持久化，随后服务端在任何解码前要求以下三者一致：

1. 认证包携带的规范 SHA256；
2. 该活动 SQLite 配置固定的规范 SHA256；
3. 当前本地注册表条目的规范 SHA256。

同名但身份不符会在调用 decoder 前拒绝，包括数据库重新打开后配置身份发生漂移的情况。客户端 pending 保存该身份，重试时也核对它并重发已有完整字节，不重新编码。

**SHA256 是双方约定的语义标识，不是 decoder 执行正确性的证明。** 正确实现规范及配置来源可信仍是前提。专门的负对照让错误 decoder 输出 2 倍数值但仍声称正确身份：本地受信状态机仍会接纳并允许 EF 提交。这明确显示，身份一致不能证明实际代码忠实执行规范，也不能保护恶意持钥服务端。冻结规范的源码参考摘要仅用于标识这次只读接入的参考编码器。

## 其余状态与提交约束

认证包还绑定 activity、client、round、RecordID、nonce、base model hash、layout、codec 名称及完整载荷。服务端状态为 `REGISTERED → ACCEPTED → INCORPORATED`。最终化在一个 `BEGIN IMMEDIATE` SQLite 事务中同时更新全局模型、累计更新量、轮次结果及 `INCORPORATED` 状态。

固定完整参与客户端 c1、c2 使用已知权重 0.25、0.75；没有动态权重贡献。相同 RecordID、相同认证字节重发幂等；不同字节的 ID 碰撞拒绝。nonce 在 activity/client 内唯一，同一客户端每轮最多一个已接纳贡献。

客户端持久化精确认证字节和 `r_next = update + r_current − decode(payload)`，只有查询到同 RecordID、同 digest 的 `INCORPORATED` 才在一个独立本地事务中提交预存残差一次。未知、仅登记及仅接纳状态均不提交 EF。模型 base hash 更新而稳定 layout 不变时，残差保留。

## 实际验证

```sh
/opt/anaconda3/envs/Tinyproto/bin/python work/durable_integrity_v2/test_protocol.py --output-dir work/durable_integrity_v2/results_final
```

本次 Python 为 3.13.15，SQLite 为 3.53.4；正式 JSON 记录测试总耗时约 1.418 秒。该数值只是执行记录，不是链、网络、认证吞吐量或 ML 性能指标。已有结果不覆盖；再次运行应选择新输出目录。

| 测试组 | 项数 | 结果 |
|---|---:|---|
| 旧标签下篡改数值码、索引、位宽、scale、payload、绑定元数据或规范身份 | 15 | 全拒绝 |
| 正确重签但固定轮次上下文或 codec 规范身份不符 | 7 | 全拒绝接纳 |
| 正确重签但索引/位宽/scale/长度不合法 | 5 | 全由严格解码拒绝 |
| 生命周期、身份持久化、恢复、并发、实际编码器、客户端回滚与边界负对照 | 35 | 全符合断言 |

新增身份用例确认：同名不同客户端 spec ID、服务端重开后的不同注册表 spec ID 均拒绝且 decoder 调用次数为 0；正确 ID 可从 wire、pending 和 SQLite 配置恢复并完成聚合及 EF 提交；pending 重试时身份漂移不重新编码；只读配置及 SHA256 格式约束正常。

原有关键套件全部保留并通过：

- 16 线程并发同字节注册、接纳、最终化和 EF 提交，仅 1 次服务端新最终化，两个客户端各 1 次残差提交，模型符合固定已知权重。
- 子进程在模型 SQL 更新后、提交事务前直接 `os._exit(73)`；数据库重开后模型、累计量、最终化次数和记录状态共同回滚，随后可重试完成。
- 最终化应答丢失后重启查询，精确重发原载荷，匹配 digest 后仅一次提交预存残差。
- 新加入客户端第二条 SQL 写入失败用例：pending 状态更新被触发器中断时，之前的 EF 修改一起回滚；去掉故障后仅提交一次。
- 只读加载冻结 `mixed_precision.codec`，其 SHA256 与规范参考值一致；4 维合成更新产生 71 字节真实编码器载荷并完成完整本地往返。

188 个 SQLite 文件实际落盘；独立逐文件执行 `PRAGMA integrity_check`，**188/188 为 ok**。62 份随机合成本地密钥均为 0600，全部在结果目录之外；正式结果树中密钥文件数为 0。见 `results_final/durability_audit.json`。

## 冻结摘要

| 项目 | SHA256 |
|---|---|
| synthetic-sparse-q8-v1 wire/spec | `90543828505430c4a0df3f2671c9f396015f8c0bf1d6943fa4d0566e4ea2c0ec` |
| frozen-mixed-precision-v1 wire/spec | `73e252614fad7fabc4d4921f526c1ff71e7c0d4c18f4528f443aacb39f689226` |
| 冻结 mixed_precision/codec.py | `46f3601e5c107f192d359227b21aea2734cc5e798674f2dccbbdf46fc796d863` |
| v2 protocol.py | `0b40fe49834a57a25c46006022e1161d6e0b3980de49ccca9c0a723f101ebc51` |
| v2 test_protocol.py | `b27894b2e07fbc52b083dbbeece15e0cd871db770af27ab0fe46cb385c7b3974` |

## 复现材料与证据边界

公开复现材料可包含源码、`specs/`、此报告和正式机器结果；若附数据库，它们只有认证字节、标签和状态，不存密钥。**必须排除所有 `*_private_test_keys/` 目录**，包括本次的 `results_final_private_test_keys/` 及开发运行的 `results_private_test_keys/`。复现时测试程序自行生成新的随机合成密钥，无须发布当前密钥。`.gitignore` 也排除了这些目录。`results/` 是首次 v2 开发运行；正式冻结结果采用 `results_final/`。

这仍是本地 SQLite 模拟登记、受信聚合和独立客户端数据库的有限合成测试。服务端持有 HMAC 密钥，数据库与未单独认证的 query/Receipt 调用均须可信；持钥者伪造及投毒负对照仍可被纳入。不存在实际链节点、共识、网络或真实许可链 benchmark。原子性仅分别覆盖各本地 SQLite 事务，不跨服务器/客户端构成分布式事务；进程退出测试也不等于硬件掉电或磁盘故障保证。该部分不证明 AI 方法、ML 性能、隐私或抗投毒能力。

# 本地独立协议核查

范围：**simulation_only**；外部cross-model：**REVIEW_UNAVAILABLE**。未训练ML，未改作者源码/结果。

**复跑52/52通过；发现一个需明确约束的配置语义缺口。** 独立输出为`check_independent/formal_52/test_results.json`；158个数据库integrity_check全ok，52份测试密钥权限0600，protocol/test源码哈希与作者正式结果一致。

## 发现：同名codec语义漂移未检测（P2，配置一致性）

`protocol.py:330–332`只校验codec名称再调用服务端decode；`446–452`客户端用自己的decode预存rnext。名称/HMAC不绑定实现哈希。独立例将服务端同名decoder改为2倍：update=[.7,.2]，客户端decode=[.75,0]，服务端纳入[1.5,0]，客户端仍提交EF=[−.05,.2]；两者和=[1.45,.2]，不守预期账。见`check_independent/INDEPENDENT_EXTRA_EVIDENCE.json`。

应显式要求不可变、两端一致的codec注册表，或绑定并核对实现/规范版本哈希并加拒绝测试。这不是无密钥攻击，也不否定现有相同codec路径；当前报告若称“codec已绑定”，须限定为名称绑定。

## 关键正确性与证据边界

- `protocol.py:208–217,349–378`：finalize在同一BEGIN IMMEDIATE事务内更新model、accumulator、round结果和INCORPORATED；`354–356`返回历史轮结果，后续轮也不重聚合。正式用例`test_protocol.py:218,268,323`覆盖进程中断、并发及旧轮幂等。
- `protocol.py:332–337,365–369`：接纳时严格解码并持久化decoded；聚合确实使用这些值，未用原始update旁路。HMAC覆盖完整序列化元数据及payload（`75–95,128–130,285–290`）；14项旧标签篡改、6项重签上下文错误、5项重签坏载荷均拒绝。
- `protocol.py:429–450,465–485`：重试返回原认证字节，旧r不变；仅匹配RecordID/digest的INCORPORATED可在客户端事务内提交预存rnext。generation与状态复查保证被测重试下最多一次。独立触发第二条pending写失败：第一条EF更新也回滚，去掉触发器后仅提交一次。完成性仍依赖最终查询/重试。
- 查询Receipt本身未认证，调用方query与数据库必须受信。持钥者投毒可被纳入的负对照已存在（`test_protocol.py:372–386`）；不能据认证声称诚实更新或抗投毒。

原子性只覆盖每个本地SQLite事务，不跨服务器/客户端数据库；不是分布式事务、真实链或共识。进程退出不等于硬件掉电/磁盘故障。52项及补测均是有限合成验证，不支持ML收益、网络性能或普遍故障保证。

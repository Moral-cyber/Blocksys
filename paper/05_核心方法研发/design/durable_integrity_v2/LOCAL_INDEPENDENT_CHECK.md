# v2 本地独立复核

**PASS；simulation_only；REVIEW_UNAVAILABLE。** 本次仅为本地独立重跑和同模型静态复核，未实施外部或跨模型审查。

用正式 CLI 在新目录 `check_independent/final_62/` 复跑同名 62 项测试：**62 通过、0 失败**。188 个新 SQLite 文件逐一只读执行完整性检查，188/188 为 ok。运行耗时约 1.373 秒，仅为执行记录。

codec 修补符合报告：`DURINT02` 的规范 wire、HMAC-SM3 及包摘要覆盖 `codec_spec_sha256`；两份版本化 spec 的精确字节摘要与只读固定值一致。活动 SQLite 持久化名称及 spec 身份；pending 保存完整 wire 并在重试时比较身份，不重新编码。

`protocol.py:357–376` 在 decoder 和幂等接纳返回之前比较包、持久化活动、当前注册表三方身份。客户端同名不同 spec、服务端重开后同名注册表 spec 漂移两项均拒绝，**decoder_calls=0**。登记阶段可保存此类包；拒绝点是接纳检查。匹配身份恢复及 pending 漂移用例均通过。

信任边界得到负对照验证：错误 decoder 声称正确 spec 后，仍可把数值放大两倍并被接纳、允许 EF 提交。身份摘要约定语义，不能证明实现忠实执行。持钥服务端、数据库及未单独认证的 Receipt/query 仍须可信；本地事务测试不构成链共识、分布式原子性、ML 效果或抗投毒证明。

运行前后 **203 个受保护文件摘要一致**，7 项 v1/冻结上游基线也一致；v2 冻结源码、specs、报告及正式结果未改。仅核对新密钥文件数量和权限，未读取或复述密钥内容：62 份均为 0600，结果树内为 0。本次无外部共享。

机器摘要见 `LOCAL_INDEPENDENT_CHECK.json`；完整重跑、完整性检查及文件摘要证据分别见新目录的 `test_results.json`、`audit_independent.json` 和其上一级 `rerun_audit.json`。

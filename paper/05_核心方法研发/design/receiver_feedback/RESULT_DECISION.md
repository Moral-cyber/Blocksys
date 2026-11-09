# 接收mask反馈闭环：有限20轮停止结论

2026-11-04。同分区同10轮dense warmup，后继20轮单training seed，均为开发validation。

|方法|last5客户端宏accuracy|last5宏CE|warmup后应用上传B|warmup后unicast下载B|
|---|---:|---:|---:|---:|
|local_sgd|79.681190%|.5625088733|0|0|
|magnitude_gate|78.986920%|.5793509869|10852720|69281120|
|receiver_feedback|78.870861%|.5802046087|10717672|79405040|

反馈相对幅值accuracy -.116059pp，CE+.0008536218，未通过冻结条件。其上传少135048B，但下载多10123920B，warmup后总通信+12.47%。停止此固定规则，不调EMA/quota/阈值救结果。

共同warmup的219004800B是明确公式追记；后继wire由编码器实际生成计长度，主仿真聚合使用原始张量，独立decoder fixture核查等价，未使用真实socket/链。反馈组52包+150B接收反馈，对照53包，两组满足共同上传上限，但实际字节不等；不能说唯一差异仅utility分数，也不能称完全等计算。

60轮汇总、预算、候选选择、TRAIN批次、最终utility独立复算一致，LOCAL_CHECK.md和LOCAL_STATS_CHECK.json保存证据。外部跨模型审计REVIEW_UNAVAILABLE。无官方test选型、无最终test结论。

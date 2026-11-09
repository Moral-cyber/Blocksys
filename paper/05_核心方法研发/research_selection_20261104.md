> 历史阶段记录：以下判断形成于后续固定实验与英文稿之前，不是当前研究状态。现有阶段产物已交付，核心贡献继续凝练；最新目标见 PROJECT_STATE.md 顶部。

# 本轮研发选择与证据边界

日期2026-11-04，主目标active，尚未完成核心方法。用户允许有针对性的已有机制组合，不要求原子首创。

|已完成开发配方|实际决定|主要依据|
|---|---|---|
|梯度敏感性混精度|停止|敏感33.79%低于简单3bit33.96%|
|仅头SCAFFOLD+精确头/低比特body|停止|35.95%比两个低字节对照仅+.16pp|
|已有body/full控制强参考|不扩此固定协议|dense35.82%，body33.09%，full31.57%，非新方法|
|pK曝光先验后处理|停止|两个异构冻结模型CE/accuracy均变差|
|新分区balanced训练+低比特|停止|35.72%未胜FedLC35.84%，CE也更差|
|balanced低比特+解析PFL先验|停止|77.4974%比FedLC同先验仅+.1556pp|
|balanced低比特+TRAIN拟合先验锚定bias|停止|77.8124%比同先验+.315pp，未过+.5|

表内全局与PFL指标不可横比，PFL用每client的TRAIN内部holdout。独立原始复算报告在相应design/checks/results。

事后观察FedLC+低比特+拟合bias78.0894%/CE.5871585，比同模型解析先验+.7476pp；这不是原候选，也不能据此翻原STOP。新探索性确认需未用过partition20261203与seed20261211，事前冻结后再运行，保持平滑/正则/轮数。必须加同资源先验、学习头、零anchor bias与vector calibration等强对照，拆清已有组件收益与组合价值。

偏置校准与免CNN反传已属已有calibration族（Guo vector scaling、BCTS、FedCal及多类后处理FL），Newton与强凸性本身不能称新。可投稿贡献须依赖具体轻量架构、通信/计算资源和可靠实证，尚未成立。

后半本地durable_integrity_v2已正式/独立/导出包62tests通过，codec身份配置误配在decode前拒绝，SQLite crash/replay/EFcommit证据成立于受信单机状态机。不能称真实链/分布式原子性、防投毒或恶意持key服务器防护。

新分区真实37502/12498，旧PLAN名义数偏差透明列WARN。旧torchvision train=True可能MD5读取test_batch字节，但未用于测试记录推理/指标选型；新strict loader仅读五TRAINbatch。

当前旧中英文稿均非新核心最终论文。参考文献次目标在核心方案确认后处理。

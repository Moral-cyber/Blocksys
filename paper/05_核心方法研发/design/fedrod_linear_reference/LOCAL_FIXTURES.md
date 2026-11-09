# FedRoD-linear CPU synthetic 检验：77 PASS

本轮未读原始数据、未联网、未启动正式训练；默认入口 DESIGN_ONLY，数据loader/torchvision均未导入，CIFAR打开尝试0。原13个冻结源SHA256与新增入口/核心/方案在检查前后均一致。

共享136874参数，额外每client Linear(64,10)为650参数/2600B。ERM不能给body/generic梯度；BRM不能给P梯度；联合梯度与独立分支最大误差均为0，通用SGD更新逐位等于同batch BRM-only。P实际更新、跨轮保留，保存/恢复后下一步与不中断训练逐位相同。推理严格为G+P logits。原始BSM缺失类有限、单类loss/梯度为0，全零计数/不可能target拒绝。

两轮synthetic共享delta真实编码均54067B（cap54808B），P不进wire/EF；实际decode用于非等权聚合，EF精确更新。恢复的source/layout/signature/head/EF/nonfinite/history异常均拒绝。一次合成batch各共享/个人forward1次，联合backward/SGD1次，个人梯度650标量；holdout两输出共享一次feature pass。

CPU toy：7张合成图，2预热＋12交错重复。BRM-only中位0.001854s，联合0.001719s，差-0.000135s。只说明此toy的墙钟，不能外推GPU、真实吞吐或准确率；独占head backward时间保持null。2600B不含activation/EF/metadata。

原文终点global/local表述差异、零初始化选择、非官方超参复现及代价取舍见PLAN.md。尚无20轮结果，不能据此主张精度或成本优胜。REVIEW_UNAVAILABLE / LOCAL_CHECK_ONLY。

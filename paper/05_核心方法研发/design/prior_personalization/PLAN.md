# 解析先验个性化头：冻结状态的应用目标诊断

2026-11-04，评估前固定。此处明确改为PFL客户端分布预测，不将local指标冒充全局IID指标。
仍是轻量化FL数据异构问题的子方向。当前全局balanced组合未通过原门槛，不撤销该负结果。

共同checkpoint来自新分区20261193四组已完成20轮的ce_uniform3、fedlc_uniform3、
balanced_uniform3、balanced_dense。官方TRAIN划分37500训练、12500 clientholdout。
本开发诊断使用每client自身holdout（包含先前globaldev的2000样本，明确非新独立test）。
所有clientprior只从其TRAIN counts产生，不读取holdout标签拟合先验。

每模型评估：(1)raw generic；(2)解析Bayes先验头 z+log((n_ic+1)/(n_i+10))；
(3)先验占位baseline只预测该prior；(4)同冻结主体、仅最后Linear用TRAIN再训练1epoch的
learned-head适配。4是额外计算的强对照，方法本身不增加梯度。学习头lr.03batch64，不调参。
learnedhead也评估raw与相同prior，不能故意不给强对照同样后处理。

BSM训练期z+logprior使latent generic对类先验脱敏、推理再恢复localprior是候选解释；
仅在label-shift和优化满足条件时有Bayes含义，不能作为一般保证。FedRoD及LA近邻应准确引用。
不声明解析prior、低比特或EF原子首创。

主要结果为20client宏平均CE/accuracy，次要为按holdout样本加权指标和client10%分位。
BSM量化+prior若宏CE低于CE+prior及FedLC+prior，宏accuracy分别至少+.5pp，
且不低于best learned-head counterpart超过.5pp，才值得新seed/分区确认零适配计算的组合价值。
若不满足，停止此固定解析个性化配方，不调prior强度、学习头lr或选client。
与dense对照报告压缩代价。全部用实际分子分母，不把20client当20个独立训练seed。
独立后处理数组复算与源码检查后才决定；不读official TEST，无新FL训练。

# 有效标签曝光先验：仅冻结模型诊断

2026-11-04，评估前固定。探索性假设由既有class6低准确率、样本权重与本地步数不一致提出，
不是事前确认研究。未评估新修正的结果前固定本文件。

根据TRAIN计数n_ic、n_i、聚合p_i=n_i/N和实际K_i=ceil(n_i/64)，计算
pi_pool,c=Σn_ic/N；pi_eff,c=Σp_i K_i(n_ic/n_i)/Σp_i K_i。
客户端小尾batch使真实等batch曝光略有偏差；本诊断先用上述固定hist近似，不偷称精确梯度先验。
普通FedAvg的一阶展开给出p_i K_i权重，但完整学习动态不等于标签偏移模型。

拟检验输出logit加 d_c=log(pi_pool,c)−log(pi_eff,c)。
这是经典label-prior-shift校正的FL具体估计组合，尚无独立新颖性结论。
若模型学到源先验pi_eff下的后验且类条件不变，则该比值实现目标先验pi_pool；
这两个条件在实际FL中仅是假设，不由数据counts自动保证。

五个既有最终checkpoint：重锚定seed20261171的mix0/1×dense/TopK-EF四组，
以及seed20261191 plain_dense。所有模型20轮已训练完，不修改权重/残差或训练流程。
同TRAIN holdout2000开发集，严禁official TEST。重算仅最终一轮，不能伪造last5修正结果。

四种输出：raw；pooled→uniform(-log pi_pool，含任意常数不影响softmax)；
effective→pooled(d)；反方向(-d)。不拟合温度/系数/截距、不用dev标签估计先验。
保存一次raw logits及真实dev labels、训练元数据与输入文件摘要供独立复算。
这份保存数据仅为TRAIN开发集，非模型生成ground truth。

有限去留：mix0 dense与TopK两状态，d校正最终CE均下降、accuracy均至少+0.5pp，
且两者都优于反方向修正，才值得进一步查精确prior和独立训练/分区；否则停止这个固定规则。
mix1为分布重分配对照，检查先验偏斜是否相应接近0；seed20261191作补充探索性状态，
不以它替换失败的两主状态。无新训练、无网格、无最终性能或首创声明。

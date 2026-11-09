# 新分区确认独立核查
DONE；产物PASS 1505/1505；固定FedLC uniform3＋prior-anchor bias：**FOLLOW_UP**；外部REVIEW_UNAVAILABLE。
宏acc 75.712447%、CE 0.643701922；比同loss prior +1.253417pp、CE减少0.021676961；比CE+bias +0.098237pp、dense+bias -0.035528pp。
六个冻结门槛：same_loss_prior_accuracy_plus_half_pp=通过；same_loss_prior_lower_ce=通过；ce_bias_accuracy_noninferior_half_pp=通过；ce_bias_ce_within_001=通过；matched_head_accuracy_noninferior_half_pp=通过；quant_dense_accuracy_noninferior_half_pp=通过。
实际TRAIN/holdout=37501/12499；60组保存数组重建50000标签hash、300条client指标与75项汇总独立复算。TRAIN-only prior、F/g/H有限差分、H≥10I、零均值及全部Newton/Armijo轨迹核验；独立重新拟合FP64最大差2.7e-09，FP32位级一致60/60。
三组训练日志的共同seed/counts/shift/梯度/字节、checkpoint最终布局/EF norm与首包解析核查。作者bias额外TRAIN forward 112503样本/468批，CNN反传0；matched head 1788梯度；Newton 242步、目标评估320次；bias40B/client。
部分fit以FP64目标分辨率停止，最终梯度∞范数最大1.1e-06，非全部≤1e−8；轨迹等价求和容差2e−10不改效果门槛。
未读原图/official TEST，未做新CNN/训练；仅保存首包，不能从最终产物独立重演所有轮local delta/EF/聚合。一个探索性新split/seed，非最终test，clients不是seeds；旧STOP不变。FOLLOW_UP也须零anchor、BCTS/FedCal、两freshseed、第二任务及归属/总成本对照。源/输入重验不变；完整证据见LOCAL_CHECK.json。

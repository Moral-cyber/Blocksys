# FedLC低比特＋TRAIN拟合bias：探索性新分区确认

2026-11-04，运行前固定。组合为FedLC-inspired tau1 CE、uniform3 EF、prior-anchor十维bias。在查看旧分区后选出，属探索性确认；balanced及旧prior/bias配方STOP不变。已有组件不作原子创新。

新partition20261203、alpha.1、20clients、freshseed20261211。三组ce_uniform3、fedlc_uniform3、fedlc_dense，默认20轮，SGD.03、batch64、epoch1、全参与固定样本权、同SmallCNN136874参数、共同初始化及seed+100r+i的batch流；显式改轮数形成新签名。样本数按实际分区报告。

冻结balanced_quant.train直接复用：CE或CE(z−max(n,1)^−1/4)。冻结codec真实uniform3编码、decode后聚合，EF=delta+旧residual−decoded，cap54808B/client。dense上传及global下行解析547560B/client，非网络/链实测。

仅读五个TRAIN batch，核对固定SHA、labels hash，直接partition_indices＋IndexedCifar10；不用旧prepare。test_batch字节读取审计拒绝。source/PLAN、冻结依赖、数据及IDs/counts/weights进签名；checkpoint含模型、EF、config、轮历史，不兼容拒绝resume。

评估只用final checkpoint和全部clientholdout，不按dev选checkpoint。固定raw、TRAIN prior、prior-anchor bias及同模型最后Linear SGD1epoch的matched head/head+prior。prior仅TRAIN counts；冻结NumPy solver的lambda10、20步/Armijo不调。bias零CNN反传、一次TRAIN forward；学习头梯度另计，记录forward/时间/40B状态和上传、下行。

GO须全部满足：FedLC量化+bias比同loss prior宏accuracy≥+0.5pp且CE更低；比CE+bias accuracy不劣0.5pp、CE不差0.01；比同模型head/head+prior最好accuracy不劣0.5pp；比FedLC dense+bias不劣0.5pp。否则STOP，不调lambda/温度/bit/轮数。宏指标为主，加权与client10%分位为辅；clients不是训练seeds。

单新split/seed达标仅FOLLOW_UP。must-next：独立helper实现同lambda10零anchor bias；vector scaling/BCTS主对照、强FedCal vector、seed20261212/13、第二任务和归属确认。先CPU合成fixture，由主任务决定启动；交付时不运行正式训练。

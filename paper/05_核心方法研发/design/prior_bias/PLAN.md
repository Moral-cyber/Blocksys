# 固定模型的TRAIN拟合十维bias：新有限诊断

2026-11-04，运行前固定。此前解析prior固定配方的停止结论保留。本候选针对模型混淆造成的真实logit偏差与类别prior不匹配，使用已有bias calibration思想；不声明新机制首创，不训练CNN或新分类头。

四个已完成balanced_quant冻结模型全部运行：ce_uniform3、fedlc_uniform3、balanced_uniform3、balanced_dense。只以balanced_uniform3为候选，其余作loss/压缩删除背景，不按结果更换候选。相同partition20261193，实际每模型37502 local TRAIN、12498 clientholdout；holdout包含旧globaldev，非独立新test。

每client对local TRAIN恰一次CNN前向，保存raw logits、GT、original IDs。令b0=log((n+1)/(N+10))再中心化。用纯NumPy float64最小化F(b)=ΣTRAIN CE(z+b,y)+10/2||b−b0||²；grad=Σ(p−onehot)+10(b−b0)，H=Σ(diag p−ppᵀ)+10I。从b0起最多20个Newton step；每次中心化去gauge，Armijo c1=1e−4、初始步长1、折半至多40次，梯度∞范数≤1e−8或FP64目标无可分辨下降时停止。没有温度、λ、client选择或其他网格。

拟合只接收TRAIN数组。holdout复用prior_personalization保存的raw/learned logits、labels/IDs，核验与冻结manifest一致后，仅作评估。全部固定对照raw、analytic_prior、prior_only、learned_head、learned_head+prior与新bias。主要宏平均CE/accuracy，同时报告样本加权值和client10%分位。

新候选必须宏accuracy较同模型prior及四模型既有prior中的最好者各至少+0.5pp、宏CE均更低，且不低于四模型两个learned-head对照的最好accuracy超过0.5pp，才标FOLLOW_UP；否则STOP，不网格挽救。最好对照只构成固定比较包络，不能改变候选模型。

部署bias存FP32十维40B/client；FP64求解轨迹另存为证据。新增CNN backward/learned-head训练/FL训练/uplink均为0；实际额外TRAIN图像forward数、batch调用数、Newton/线搜索次数、缓存与耗时逐项记录。不是零计算或端到端通信实测。所有源/方案、冻结checkpoint/config/NPZ和五个TRAIN batch摘要前后核对；绝不读取official TEST。

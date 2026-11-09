# FedRoD-linear 强对照设计；正式训练尚未运行

2026-11-04固定。候选与既有STOP/confirm结论均不改。只新增独立入口及CPU synthetic fixtures，不读取CIFAR原图，不联网，不启动正式20轮训练。

原文依据是已归档 arXiv:2107.00778v2（11 Jul2022），work/prior_personalization/sources/fedrod.html/.txt：§4.1式6给BSM；§4.2式7–10规定BRM只更新共享特征θ/通用头ψ，ERM只更新本地残差头φ，φ跨轮保留、不上传/聚合；附录B.2式13明确personalized=hG(f(x))+hP(f(x))，式14明确single forward-backward SGD，ERM不对θ/ψ求导。因此代码只做一次共享feature/generic forward，个人分支为hG.detach()+hP(feature.detach())，两loss相加后一次backward和一次SGD step。

原文终点有表述差异：主文§4.2末句、算法1输出包括最后本地θ*/ψ*/φ*；B.2又明确最终barθ/barψ加各φ*进入式13。本入口采用B.2明确的global generic＋持久local residual，避免新增持久完整local body。不是宣称解决原文歧义，也不冒充官方代码逐项复现。没有hypernetwork、meta-BSM、外部meta数据或个性化prior后处理。

训练预算固定：CIFAR同partition p20261203、初始化seed20261211，20clients全参与、20轮、1 local epoch、batch64、SGD lr.03，无momentum/weight decay；batch seed=seed+100r+i，沿用冻结confirm/common.prepare严格TRAIN loader、原SmallCNN（136874参数，64维feature、既有通用650参数头）及量化codec。额外hP=Linear(64,10,bias=True)，650个FP32参数/2600B，零残差初始化（原文未固定初始化，属公开实现选择），跨轮/重启保留。global model初始化与既有confirm一致，初始化local heads不消耗全局模型RNG流。按式6原BSM γ=1直接使用TRAIN计数：缺失类log n=-inf；不加1、不改成现有平滑BSM。全零计数拒绝，batch GT必须属于正计数类，单类BSM为有限零损失。这些及本地训练超参数均明确披露，非官方超参复现。

只压缩共享body＋generic的136874维delta，uniform3＋EF，真实生成/解码wire，payload cap54808B；P-head不进入delta/EF/uplink，固定sample-count权重聚合；最终推理global raw与global+local residual分别报告。原有量化、数据/模型源均哈希锁定，不修改。checkpoint同时保存global、20个EF、20个head及完整source/config签名；不兼容参数布局/状态/源在读数据前拒绝。恢复仅按已完成round续训，不重新初始化head。

成本逐项记录：TRAIN feature/generic/residual forward batch和image数、BRM/ERM调用、一次combined backward与SGD次数、body/generic/residual梯度更新次数；feature+BRM、personal forward+ERM、joint backward+step、CPU拷贝/量化/评估/保存的同步wall time。额外个人分支前向有独立计时，joint backward不伪造“可独立归因的额外head backward时间”；其独立值记null并说明，另CPU同批BRM-only/联合一步仅作toy计时对照，不能外推GPU或性能。报告650梯度标量/step和残差Linear矩阵规模，不称head梯度免费。最终holdout一次共享feature forward同时得generic和personal logits，计评估forward；不按holdout调参/换种类。

状态/通信边界：2600B head是额外持久权重，checkpoint框架/metadata、EF每client547496B、global下行及训练activation峰值另计；压缩包字节是真实wire，downlink沿用解析口径，尚无真实network/chain。bias40B vs vector80B仅省40B，相对EF/模型/下行意义有限，不能独立构成系统贡献。偏置拟合的n×10 FP32 TRAIN logits缓存为40nB，FP64副本、design/Hessian、holdout缓存另计；不能只报部署40B。

既有matched_head已requires_grad_(False)冻结body，仅末Linear650参数训练1epoch，因此偏置“零CNN body backprop”不是相对它的独有优势；应比较650 vs10参数的梯度、额外forward/cache、墙钟、分数和状态。FedRoD在每轮已有feature forward上增加残差head训练，无需最终TRAIN-logit拟合缓存；bias在既有FL后另作TRAIN forward与拟合。对照允许准确率、CE、尾客户端、计算/存储存在取舍，不要求候选全面胜出。

验证先做CPU synthetic：ERM单独不能给body/G梯度；BRM不能给P梯度；联合一次梯度与独立分支逐位/容差一致；通用分支更新与同批BRM-only相同；P持久跨轮及save/reload一致，inference exact logit sum；缺失类/单类有限；真实编码136874维、54067B<=cap，只含共享delta且实际解码聚合；source不改、官方数据打开0。正式入口默认DESIGN_ONLY，必须显式--execute-formal才会加载TRAIN。当前不执行该开关。LOCAL_CHECK_ONLY，外部REVIEW_UNAVAILABLE。

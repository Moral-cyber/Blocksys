范围：2次OpenAlex查询，各取前15、无翻页；核读4篇primary并复用既有原文。有限定位不证明不存在，未判断效果。

[FedRoD](https://arxiv.org/html/2107.00778v2) §4.1式6已用CE(z+γlog n_i,y)；§4.2通用=hG、个性化=hG+hP，hP为ERM学习、本地保留的FC；§4.3可由类频率经可训练超网络生成hP。平衡训练、双路预测及频率驱动适配已有；未见hP固定为log(n_i+1)。

[Menon](https://arxiv.org/html/2007.07314) 式9/10已给推理z−τlogπ、训练CE(z+τlogπ,y)。[FedNoRo](https://www.ijcai.org/proceedings/2023/0492.pdf) §4.1使用本地z+logπ训练，§5.3将LA基线称FedLA；这不是整套解析PFL的证据。候选应归为平滑BSM＋Bayes先验适配＋既有低比特EF组合；未定位全套同式同压缩先例，不能据此声称首创。

令q_ic=(n_ic+1)/(n_i+C)。若BSM raw对应均匀后验，个性化为z+logq_i；CE pooled后验的标准适配为z+logq_i−logπ_pool。BSM再减pool会改变目标，仅pool均匀时等价；近均匀仅近似。共享p(x|y)、校准及先验正确是解释前提，平滑/缺失类无一般保证。

最小强对照：CE/FedLC＋同prior、BSM raw/同prior、同主体lastLinear本地1epoch；补CE标准先验比、FedRoD-linear、prior-only、dense BSM＋prior。统一预算并报告计算/适配梯度/可训练头状态/计数状态/实际字节；1epoch未必同计算。“无新增可训练头”成立仍需本地计数。

范围：2次OpenAlex查询，各审阅前15条、无翻页；核读5篇原文。不读拟合结果、不判效果，有限定位不证明不存在。

[Guo](https://arxiv.org/html/1706.04599) §4式8：冻结网络，NLL拟合softmax(Wz+b)，vector scaling约束W对角。[BCTS](https://arxiv.org/html/1901.06852) §3.1已有z/T+b；候选z+b是T=1特例。缓存固定logits、免网络反传不是新原子；Newton是求解器，仍优化b。

[FedCal](https://arxiv.org/html/2405.15458v2) §5已有本地后处理NLL校准，但用保序MLP(K-64-64-K)，权重匹配后聚合。[联邦多类校准](https://arxiv.org/html/2510.01987) §4.2/算法2已有softmax(Az+b)、FedVector对角A，客户端拟合并聚合；使用独立calibration划分。均非纯本地保留的先验锚定10维偏置。

候选为min_b mean CE(z+b,y)+λ||b−log(n+1)||²/2；λ>0的强凸性属常规正则化性质。未定位“本地TRAIN先验锚定＋Newton＋低比特FL＋链认证”精确全套；应按已有校准族的受限正则化组合探索，不能称首次偏置校准/PFL或首创免反传。

最小强对照：raw、固定prior；相同λ的零中心L2偏置（隔离anchor）；本地TS、BCTS、vector scaling；lastLinear微调；dense＋同偏置（隔离压缩）。FedCal作有额外参数/通信的文献对照。统一可用TRAIN、冻结模型与调参预算，独立评估NLL/准确率/校准；40B仅10个FP32部署偏置，不含拟合缓存和身份元数据。

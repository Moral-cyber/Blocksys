# 非IID稀疏联邦学习：近邻核查与候选定位

检索日期：2026-11-03（Asia/Shanghai）。应用技能：`paper-lookup`，读取其 arXiv、Crossref 和 Semantic Scholar 接口说明后查询公开数据库。本次只将一般领域关键词及已公开论文题名发送至外部，未发送用户的未公开候选公式、代码或数据。

> 补充更新：更直接的RegTopK/GRSGD近邻、MultiGS公开摘要及准确基线公式见 `FOLLOWUP_NEIGHBORS_BASELINES.md`。后者更新本文件初次检索的缺口与基线优先级。

## 结论

1. **FedCSPACK 名称已核实。** Qiantao Yang、Liquan Chen、Mingfu Xue、Songze Li 的 *Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack*，arXiv:2601.01840，2026。它明确定位于个性化 FL，使用参数包、cosine选择、方向/分布距离双权聚合。用户关于“参数是否适合共享”的理解有依据；但该文也明确考虑资源约束，不能说它完全忽略资源问题。全文摘要与正文对“相似度高/低”的文字表述不完全一致，复现时须按公式与官方代码核对。
2. **“预测全局方向＋拟合系数＋压缩创新”本身不是新的方法范式。** Yue等人的2022预测编码、Edin等人的2024最小二乘预测编码及ResFed均是强近邻；尤其Edin明确支持按层系数及Top-L残差稀疏化。将此作为强基线更合理。
3. **“根据全局历史方向选择坐标”也已有直接概念近邻。** FedSGC（2024）明确用上一轮全局模型差分的符号图引导本地剪枝和生长。因此主张必须收窄到具体上传选择规则及可验证性质，不能主张首创全局方向引导、首创方向一致性、首创异构感知稀疏化。
4. 在本次已记录检索边界内，未找到与当前“Top-K能量保护核心＋剩余名额按有符号历史方向投影选择”的完整规则完全一致的论文。**这只是本次检索的未发现结果，不能证明不存在先行工作或直接证明足够新颖。**

## 最接近候选的已核实文献

| 工作 | 已核实出处 | 机制及差异 |
|---|---|---|
| FedSGC | Chris Xing Tian, Yibing Liu, Haoliang Li, Ray C. C. Cheung, Shiqi Wang. *Gradient-Congruity Guided Federated Sparse Training*. 2024. https://arxiv.org/abs/2405.01189 | 用 `sign(global_new-global_old)` 指导动态稀疏训练；优先剪除反向参数、优先生长同向高梯度参数。它修改训练mask；当前候选仅对完整本地训练后EF更新做上传坐标选择，并增加显式能量保护。 |
| FedCSPACK | Yang, Chen, Xue, Li. 2026. https://arxiv.org/abs/2601.01840 | 个性化参数包共享，cosine与KL双权聚合；目标和聚合机制均不同于单全局模型的固定K上传。 |
| Majority Voting | Kerem Ozfatura, Emre Ozfatura, Deniz Gunduz. *Distributed Sparse SGD with Majority Voting*. 2020. https://arxiv.org/abs/2011.06495 | 先让worker就稀疏结构达成共识，以减少稀疏模式错配和索引通信；不是客户端自行用历史全局方向打分。 |
| ConSpar/FedREP | Yi-Rui Yang, Kun Wang, Wu-Jun Li. *FedREP: A Byzantine-Robust, Communication-Efficient and Privacy-Preserving Framework for Federated Learning*. 2023. https://arxiv.org/abs/2303.05206 | 共识稀疏化用于兼容鲁棒聚合和隐私；不得把“稀疏＋安全”笼统称首次结合。 |
| Total-error sparsification | Atal Narayan Sahu et al. *Rethinking gradient sparsification as total error minimization*. 2021. https://arxiv.org/abs/2108.00951 | 指出Top-K在固定每步K预算下最小化局部误差；改为总训练预算的hard-threshold。候选不能声称同K局部L2误差优于Top-K，应明确用部分能量换方向投影。 |
| L-GreCo | Mohammadreza Alimohammadi, Ilia Markov, Elias Frantar, Dan Alistarh. *L-GreCo: Layerwise-Adaptive Gradient Compression for Efficient and Accurate Deep Learning*. 2022预印本. https://arxiv.org/abs/2210.17357 | 在误差约束下动态选择各层压缩参数。“分层自适应＋误差约束”不是新概念。 |
| DAGC | Rongwei Lu, Yutong Jiang, Yinan Mao, Chen Tang, Bin Chen, Laizhong Cui, Zhi Wang. *Data-Aware Gradient Compression for FL in Communication-Constrained Mobile Computing*. arXiv:2311.07324，DOI https://doi.org/10.1109/TMC.2024.3504284 | 根据不同客户端数据量/分布分配压缩率，有DAGC-R/A；不是同一K内按方向选择坐标。仅摘要核查，未完整复现。 |
| Overlap-weighted FL | Zichen Tang et al. *Bandwidth-Aware and Overlap-Weighted Compression for Communication-Efficient Federated Learning*. 2024. https://arxiv.org/abs/2408.14736，DOI https://doi.org/10.1145/3673038.3673142 | 带宽自适应比例＋按坐标重叠修正聚合权重；与候选的选择端机制不同，但关系到异构稀疏更新聚合偏差。 |

## 为什么不采用普通预测压缩作为原创主张

- Kai Yue, Richeng Jin, Chau-Wai Wong, Huaiyu Dai. *Communication-Efficient Federated Learning via Predictive Coding*. IEEE JSTSP 16(3), 369–380, 2022. https://doi.org/10.1109/JSTSP.2022.3142678 ; https://arxiv.org/abs/2108.00918 。全文IV使用共享历史全局权重、多个预测模式、选最小预测误差并传模式信息；实验已有Non-IID CIFAR-10。
- Adrian Edin, Zheng Chen, Michel Kieffer, Mikael Johansson. *Temporal Predictive Coding for Gradient Compression in Distributed Learning*. Allerton 2024. https://doi.org/10.1109/Allerton63246.2024.10735311 ; https://arxiv.org/abs/2410.02478 。III-A最小二乘组合历史梯度，发送系数和压缩预测残差；III-B任意预测器可乘最优scalar；Remark III.1允许分参数子集/层系数；Table II已测Top-L稀疏残差并计系数字节。
- Rui Song, Liguo Zhou, Lingjuan Lyu, Andreas Festag, Alois Knoll. *ResFed: Communication-Efficient Federated Learning With Deep Compressed Residuals*. IEEE IoT Journal，2024卷期；DOI 2023注册。https://doi.org/10.1109/JIOT.2023.3324079 ; https://arxiv.org/abs/2212.05602 。上下行共享预测器、残差稀疏/量化/编码。
- Peter Richtárik et al. *3PC: Three Point Compressors for Communication-Efficient Distributed Training and a Better Theory for Lazy Aggregation*. 2022. https://arxiv.org/abs/2202.00998 。演化压缩统一框架，可恢复EF21；不能把差分压缩泛称新范式。

## 必须区分的其他工作

- **GraVAC**，不是已核实的“FedGraVAC”：Sahil Tyagi、Martin Swany，*GraVAC: Adaptive Compression for Communication-Efficient Distributed DL Training*，IEEE CLOUD 2023，https://doi.org/10.1109/CLOUD60044.2023.00045 ，https://arxiv.org/abs/2305.12201 。其主要场景是DDP动态压缩率，不应当成针对非IID个性化共享的同类方法。
- **FedSparsify**：Stripelis、Gupta、Ver Steeg、Ambite，*Federated Progressive Sparsification (Purge, Merge, Tune)+*，2022，https://arxiv.org/abs/2204.12430 。渐进权重剪枝、模型逐步变小，不仅上传压缩。
- **FedMask**：Ang Li、Jingwei Sun、Xiao Zeng、Mi Zhang、Hai Li、Yiran Chen，*FedMask: Joint Computation and Communication-Efficient Personalized Federated Learning via Heterogeneous Masking*，SenSys 2021，https://doi.org/10.1145/3485730.3485929 。本次核实题名、作者与出版记录；全文机制未充分核查，不据此声称具体性能差异。
- **FedDST**：Sameer Bibikar、Haris Vikalo、Zhangyang Wang、Xiaohan Chen，*Federated Dynamic Sparse Training: Computing Less, Communicating Less, Yet Learning Better*，2021预印本，https://arxiv.org/abs/2112.09824 。动态稀疏子网训练，改变本地计算和参数空间。
- **EF21**：Peter Richtárik、Igor Sokolov、Ilyas Fatkhullin，*EF21: A New, Simpler, Theoretically Better, and Practically Faster Error Feedback*，2021，https://arxiv.org/abs/2106.05203 。不是本工程普通EF的别名，不能把普通Top-K+EF实现标为EF21。
- **FetchSGD**：Daniel Rothchild、Ashwinee Panda、Enayat Ullah、Nikita Ivkin、Ion Stoica、Vladimir Braverman、Joseph Gonzalez、Raman Arora，2020，https://arxiv.org/abs/2007.07682 。Count Sketch可合并，误差与momentum移到服务器；需计入sketch配置/字节并处理其算法假设。
- **SCAFFOLD**：Sai Praneeth Karimireddy et al.，2020正式会议/2019预印本，https://arxiv.org/abs/1910.06378 。控制变量修正client drift；控制变量额外通信不能漏算，也不能把简单方向选择称作SCAFFOLD的等价替代。

## 推荐最小对照与机制验证

主表先限制为三类：

1. 稠密FedAvg参考＋普通Top-K＋Top-K-EF（当前实现可复用）。
2. 方向感知对照：纯历史方向投影选择；幅值与方向固定比例混合；随机替换同样名额。它们是明确标注的机制控制，不冒充FedSGC完整复现。FedSGC至少必须进入相关工作，若主要贡献仍是方向选择，应在扩展实验实施其完整动态稀疏训练，或清晰解释设定差异。
3. 非IID优化强对照：FedProx+Top-K-EF（控制本地训练、同一字节预算）或完整SCAFFOLD并计控制变量开销。若采用预测候选，则必须额外有Edin风格最小二乘预测Top-K，而不是只比普通Top-K。

候选最重要的证据是：**相同模型、相同K、相同随机轨迹和字节预算，在非IID条件下是否稳定优于Top-K-EF，且不损害少数类别/差客户端。** 至少包含IID与多个Dirichlet强度、不同数据划分与训练种子；test只用于冻结方法后的最终报告。

当前能量保护规则可以提出单步能量和对历史方向投影性质，但历史方向不是当前真实全局梯度。必须比较真实聚合误差、当前方向夹角、残差增长及最终学习表现；不能凭对历史g投影更大推出非IID收敛更快。

## 检索边界、原始记录与限制

- arXiv关键词、请求URL、返回数量和条目原始元数据分别保存于 `arxiv_search_results.json`、`predictive_search_results.json`、`neighbors_search_results.json`、`alignment_search_results.json`、`exact_neighbor_search_results.json`；原始Atom XML同目录。
- Crossref一般检索、精确题名与DOI响应保存于 `crossref_search_results.json`、`exact_crossref_results.json`、`alignment_crossref_search_results.json`及原始响应文件。
- 已下载公开HTML并提取可搜索文本：FedCSPACK、FedSGC、Predictive Coding、Temporal Predictive Coding、ResFed、3PC、Majority Voting、Total Error、Overlap-weighted FL等。
- 部分查询只取最相关前12/15/30条，未系统追踪全部引文，未覆盖所有出版社付费全文。此为聚焦近邻核查，不是系统综述，不支持“没人做过”的绝对断言。
- Semantic Scholar仅补充元数据与公开摘要；若限流/无摘要，将记录为缺口，不补写推测。
- `verified_arxiv_sources.json`汇总选取来源；完整作者与原始发布时间可回查数据库响应。原文中“我们首次”“达到SOTA”等为作者主张，本报告没有独立复现其性能。

### 额外核实但尚未取得摘要的近邻缺口

- Jitao Ma, Donglai Liu, Weiying Xie, Haowei Li, Yunsong Li, Leyuan Fang. *MultiGS: Multi-Dimensional Information-Aware Gradient Sparsification*. IEEE TCSVT，Crossref卷期日期2026-04，DOI带2025：https://doi.org/10.1109/TCSVT.2025.3636885 。Crossref确认存在及作者，未返回摘要；Semantic Scholar精确DOI接口HTTP429。本次未核实其信息维度/选择公式，不推断与候选规则等价或不同。
- Semantic Scholar `ARXIV:2405.01189`补充请求也HTTP429；FedSGC已有arXiv原文，所以不影响对其3.2节的判断。

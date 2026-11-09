# 补查：MultiGS、历史全局梯度近邻与两个强基线

日期：2026-11-03。此文补充并更新 `LITERATURE_POSITIONING.md`。本轮使用 `paper-lookup` 的公开API流程，并用 `pdf` 技能读取和视觉核对作者公开的RegTopK PDF。没有运行外部论文代码，没有将外部论文性能当作本工程结果。

## 1. MultiGS公开摘要已取得，全文仍未取得

**书目信息**：Jitao Ma, Donglai Liu, Weiying Xie, Haowei Li, Yunsong Li, Leyuan Fang. *MultiGS: Multi-Dimensional Information-Aware Gradient Sparsification*. IEEE Transactions on Circuits and Systems for Video Technology, 36(4), 5246–5257, 2026. DOI: https://doi.org/10.1109/TCSVT.2025.3636885 。Crossref卷期为2026年4月；OpenAlex记录的首次公开日期为2025-11-25。后一个日期是索引元数据，不能误称本次已读取IEEE原页确认的online日期。

来源层级与获取过程：

- IEEE官方公开落地页：https://ieeexplore.ieee.org/document/11267482/ ，本次请求返回HTTP202、空正文；未取得官方页摘要。
- IEEE公开metadata路径返回404；Semantic Scholar精确DOI重试仍429。
- OpenAlex公开DOI记录：https://api.openalex.org/works/https://doi.org/10.1109/TCSVT.2025.3636885 ，成功取得题名、作者、摘要倒排索引，并按位置无生成地重建摘要。原始记录：`multigs_openalex.json`；重建文字：`multigs_openalex_abstract.txt`。
- OpenAlex标记closed、无已知开放全文库位置；arXiv精确题名查询无匹配。未绕过出版社付费墙，也未根据题名虚构公式。

摘要的具体机制：将**本地梯度高频分量**和**稀疏全局梯度低频分量**组合，重建逐层梯度，处理stale gradients与model bifurcation。摘要没有说明它通过历史方向对上传TopK做打分，也没有说明显式保留TopK能量核心。现有证据支持“相关但不能判定等价”；没有全文，不能声称已完整排除近邻重叠。

## 2. 新找到的两个更直接近邻

### 2.1 RegTopK：必须认真比较的选择端近邻

**准确引用**：Ali Bereyhi, Ben Liang, Gary Boudreau, Ali Afana. *Regularized Top-k: A Bayesian Framework for Gradient Sparsification*. IEEE Transactions on Signal Processing, vol.73, pp.4463–4478, 2025. DOI: https://doi.org/10.1109/TSP.2025.3624791 。

作者开放稿：https://arxiv.org/abs/2501.05633 。本轮读取v2，更新于2026-02-15；首次提交2025-01-10。v2第一页注明正式发表于2025年且当前arXiv版本增加实验。作者公开代码链接由论文第一页提供：https://github.com/AlBerey/RegTopK 。

RegTopK直接利用过去聚合梯度，推断哪些本地累积梯度会在全局聚合中被抵消，并改变上传TopK选择。因此，仅提出“考虑其他客户端、历史全局信息、取消无效大梯度、改进EF选择”并不足以构成清晰差异。

本轮下载并用`pypdf`读取`regularized_topk.pdf`（27页），关键页面用`pypdfium2`渲染后视觉核对：`regtopk_page_9.png`、`regtopk_page_13.png`、`regtopk_page_14.png`、`regtopk_page_18.png`。

原文Notation：本地梯度为 \(g_n^t\)，本地累积梯度为 \(a_n^t=\epsilon_n^t+g_n^t\)，服务器上一轮广播聚合梯度为 \(g^{t-1}\)，上一轮选择mask为 \(s_n^{t-1}\)，聚合权重为 \(\omega_n\)。

**Algorithm2（PDF第14页，已视觉核对）**：

\[
\Delta_n^t=s_n^{t-1}\odot
\left[(g^{t-1}-\omega_n a_n^{t-1})\oslash(\omega_n a_n^t)\right]
+Q(1-s_n^{t-1}),\quad Q\to\infty.
\]

\[
s_n^t=\operatorname{TopKMask}\left(a_n^t\odot
\tanh\left(\frac{|1+\Delta_n^t|}{\mu}\right),K\right).
\]

随后上传原累积梯度 \(\widehat g_n^t=s_n^t\odot a_n^t\)，残差为 \(\epsilon_n^{t+1}=a_n^t-\widehat g_n^t\)。`TopKMask`按输入幅值选择，不按有符号值。

特别核对：分子中的全局梯度和本地累积梯度均为 **t−1**，分母本地累积梯度为 **t**。激活是 `tanh(abs(1+d)/mu)`，没有额外负号，没有`1−tanh`，也不是`tanh(abs(d)/mu)`。原文第13页Eq46对上一轮未选中坐标使用常数C，脚注6说明C=1有效；Algorithm2的Q趋于无穷给出相同因子1。原文第12页Eq45另写了CDF形式`0.5*(1+tanh(x/mu))`，但**复现选择规则应按明确给出的Eq46/Algorithm2，不自行改成CDF偏置形式**。

参数定义和已核实实验值：

- \(\mu>0\)为postulated innovation分布尺度，文中给标准差约\(\sigma\approx0.9\mu\)；应与FedProx近端系数分开命名。
- 第18页§5.3：ImageNette微调对每个模型在\(\mu\in[1,10]\)中网格搜索，未写精确步长；曲线中的\(\mu=0\)表示TopK参考点，不是代入除法的合法值。
- Fig7题注有`λ`字样，但横轴和正文都用`μ`，不能由题注推断额外λ参数。
- §5.1线性回归与§5.2 CIFAR-10正文没有定位到一个明确的固定μ值；不能替作者补写。
- 作为本工程适配基线，应使用本工程验证集选择μ并报告候选集，不能借本文外部论文成绩指定最有利test超参。

### 2.2 GRSGD：历史全局方向修正并非新概念

Wanyi Ning, Haifeng Sun, Xiaoyuan Fu, Xiang Yang, Qi Qi, Jingyu Wang, Jianxin Liao, Zhu Han. *Following the Correct Direction: Renovating Sparsified SGD Towards Global Optimization in Distributed Edge Learning*. IEEE JSAC, 40(2), 499–514, 2022. DOI: https://doi.org/10.1109/JSAC.2021.3118396 。Crossref卷期2022年2月；OpenAlex较早记录日期2021-10-06。

公开摘要（OpenAlex DOI记录）明确：GRSGD用上一轮全局梯度估计当前全局梯度，再renovate当前零稀疏的本地梯度，使方向接近全局优化。它与当前仅上传坐标选择的区别在于重建/修正数值，但没有公开全文公式，不应推断其具体实现。`correct_direction_openalex.json`与`correct_direction_abstract.txt`保存证据。

## 3. 是否找到“历史方向上传TopK＋显式能量下限”的完全相同规则？

截至本次有限检索：没有定位到与当前能量核心＋方向补充规则完全相同的公开算法。但已经有：

- FedSGC：全局模型差分符号引导稀疏训练剪枝/生长；
- RegTopK：过去全局聚合结果引导本地EF上传选择；
- GRSGD：过去全局梯度修正零稀疏梯度；
- L-GreCo：误差约束下跨层压缩分配；
- TopK与误差反馈中的标准contractive compressor/保范数分析。

因此潜在差异应收窄到**同K下的显式能量保护集合与可证明历史投影性质，以及在具体非IID任务的稳定收益**。保能界本身不能作为“全新理论范式”，历史方向投影更大也不等于当前全局损失必然下降。

新增检索：arXiv使用公开一般关键词，包括历史梯度稀疏化、global direction、energy retention、contraction、direction-aware等；原始请求和零匹配也保存在`guarded_neighbor_search_results.json`。OpenAlex用一般关键词补查，保存`guarded_openalex_search_results.json`。OpenAlex宽查询命中数很大，本轮只筛前15条，不能以此证明无先行工作。

## 4. 最新建议：两个最有用的强基线

此前建议FedProx与FedSGC-inspired上传控制；在找到RegTopK后，**正式强基线优先级更新为FedProx+TopK-EF、RegTopK-on-model-deltas**。FedSGC-inspired控制仍应保留为廉价机制消融，但不能标成原FedSGC完整复现。

### A. FedProx + TopK-EF

主来源：Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith. *Federated Optimization in Heterogeneous Networks*. MLSys2020，https://arxiv.org/abs/1812.06127 。本轮取得v5原文`fedprox.html`及元数据`baseline_primary_metadata.xml`。

客户端初始化\(w\leftarrow w_t\)，使用相同本地epoch、batch与SGD配置，优化

\[
F_i(w)+\frac{\mu_p}{2}\|w-w_t\|^2,
\qquad w\leftarrow w-\eta[\nabla\ell_i(w;b)+\mu_p(w-w_t)].
\]

训练后\(\Delta_i=w-w_t\)，\(u_i=\Delta_i+r_i\)，\(s_i=\operatorname{TopK}(u_i,K)\)，\(r_i' =u_i-s_i\)。聚合仍按既有授权样本量权重。

它检验“是否只要现成的client-drift修正就能达到同样效果”。这是FedProx本地目标与现有压缩器的组合基线，不是声称该组合为新算法。

通信：同FP32值/uint32坐标为\(8K\)字节＋相同头部；近端项没有额外客户端上传或每轮新下行。计算多一个逐参数近端项，存储参考模型（本地训练本来已需要当前模型副本）。μp候选集和验证预算须预先记录，test不参与选择。

### B. RegTopK-on-model-deltas（明确为通信规则适配）

将原文每步梯度\(g_n^t\)替换为当前FL多步本地模型差分\(\Delta_i^t\)，因此这不是完整原文distributed gradient-SGD复现。名称和结果表应写`RegTopK (model-delta adaptation)`。

令\(u_i^t=\Delta_i^t+r_i^t\)，\(G_{t-1}\)为上一轮实际全局聚合的稀疏更新。上一轮本客户端选中集合为\(S_i^{t-1}\)，该集合上的原上传值为\(u_i^{t-1}[j]\)。对于\(j\in S_i^{t-1}\)且\(u_i^t[j]\ne0\)：

\[
d_{ij}^t=\frac{G_{t-1}[j]-\omega_i u_i^{t-1}[j]}{\omega_i u_i^t[j]},
\quad q_{ij}^t=|u_i^t[j]|\tanh\left(\frac{|1+d_{ij}^t|}{\mu_r}\right).
\]

未在上一轮选择的坐标令\(q_{ij}^t=|u_i^t[j]|\)。当前值为零直接令score=0，不实际做0除。取score最大的K个坐标，但上传**原\(u_i^t[j]\)**，而非score或乘过tanh的数值；剩余仍进入EF。首轮标准TopK。

在当前全参与、固定样本权重、服务器直接加权更新的设定中，\(\omega_i=n_i/\sum n\)，\(G_{t-1}=w_t-w_{t-1}\)。如使用额外服务器学习率/动量或部分参与，上式的参考与权重需重新定义；不能静默复用。

通信：同样K个FP32数值和uint32坐标，新增上行0；全参与、每轮dense下发且客户端缓存前模型时，可重建\(G_{t-1}\)，新增下行0。否则必须计入取得参考所需的通信。需要保存上一轮选中集合与值（K对）、EF残差，以及获取历史全局差分所需缓存。μr只用验证集选择，不能使用μr=0运行除法。

### 可保留的FedSGC-inspired上传控制（机制消融）

这只是受FedSGC方向优先生长思想启发的上传选择，不改变本地训练mask：令\(J_+=\{j:u_i[j]G_{t-1}[j]>0\}\)，\(m=\min(\lfloor\lambda K\rfloor,|J_+|)\)。先在\(J_+\)中按\(|u_i[j]|\)选择m个位置，再在其余坐标按\(|u_i[j]|\)补满K。上传原u并保留EF。

它与候选共享历史参考，但没有显式能量核心，也不用完整幅值投影\(u_jG_j\)排序。可用它判断候选是否只是重复已有“方向优先＋幅值补足”思想。上行同K同字节；历史参考的下行/缓存口径同上。

## 5. 结果与引用边界

这些来源只是文献、算法和比较设计依据。本轮没有训练这些基线，也没有复现外部论文的准确率、加速比或收敛保证。任何本工程最终主张必须由同配置新结果文件支撑。

## 6. 作者公开代码补充核对：μ=5，但存在实现口径差异

只读下载了作者仓库main分支中的`DistLearn.py`、`DistLearnLinearReg.py`、`ResNet18-CIFAR10.py`、`Toy_Example.py`、`README.md`，保存在`regtopk_author_code_readonly/`。没有运行这些文件。`provenance.json`记录检索日期、Git tree SHA、各文件Git blob SHA与SHA-256，并检查下载内容匹配当次读取的tree。

- 作者代码 https://github.com/AlBerey/RegTopK/blob/main/DistLearn.py ，本次快照第151行：`Mu = 5`。第168行：`post = torch.abs(torch.tanh((Del+1)/Mu)) * torch.abs(a_loc)`。因tanh是奇函数，`abs(tanh((1+d)/mu))`与`tanh(abs(1+d)/mu)`在μ>0时数值等价。此处没有负号或`1−tanh`。
- **μ=5可作为有公开来源的预设pilot值**；它不是本工程验证过的最佳值，也不说明所有原论文实验统一用了5。
- 本次代码快照 `ResNet18-CIFAR10.py` 第160–166行在调用RegTopK之前使用仍为上一轮值的`a_loc[k]`构造`Del_k`，其分母也是旧`a_loc[k]`；但PDF Algorithm2的分母明确为本轮`a_n^t`。这是一个需要报告的源码/论文口径差异。本工程若按当前u作分母，应该标注**按论文Algorithm2的model-delta适配**，不能声称逐行复现作者公开程序。
- 不据这处差异判断作者论文结果无效，也不在未运行两种版本前声称哪个版本更好。需要比较时，将两种口径作为明确区分的额外控制，不能暗中替换。

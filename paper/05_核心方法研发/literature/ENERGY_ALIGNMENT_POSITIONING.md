# 能量约束与历史方向选择：追加近邻核查

日期：2026-11-03。用途：支持当前候选的继续开发与实验定位。本文不把候选预先认定为创新，不把公开论文成绩当作本工程结果。继续使用 `paper-lookup` 技能；查询只发送公开的一般主题词，没有向外发送未公开的完整方法公式、实验数据或代码。

## 当前可支持的结论

本轮取得并筛查了 arXiv 6组、Crossref 5组、OpenAlex 3组一般关键词查询，另补查DAGC精确题名。当前检索材料中，没有确认一个与“固定坐标数、显式保留更新平方范数下限、优先历史聚合方向”完全相同的公开坐标选择规则。

这不是首创证明：arXiv API主要检索元数据而非全文公式，Crossref返回不少弱相关结果，OpenAlex大集合只筛前15条；MultiGS与GRSGD仍缺公开全文。因此论文不能写“首次全局方向感知”“首次误差约束压缩”或“首次将拉格朗日用于梯度压缩”。可继续评估的差异是约束对象、选择规则、可核验的单步性质，以及同预算强基线上的实测收益。

## 具体近邻与差异

### RegTopK仍是最直接选择端近邻

Ali Bereyhi, Ben Liang, Gary Boudreau, Ali Afana. *Regularized Top-k: A Bayesian Framework for Gradient Sparsification*. IEEE TSP73:4463–4478,2025. DOI https://doi.org/10.1109/TSP.2025.3624791 ，作者开放稿 https://arxiv.org/abs/2501.05633 。

它利用上一轮聚合信息，对当前本地累积梯度作Bayesian正则打分，再取TopK；上传原累积值并保留EF。当前已读Algorithm2没有显式施加相对幅值TopK的能量下限，也不是通过约束下的线性方向投影目标求解。共享历史全局参考这一点已有先行工作，不能作为候选的独立创新。准确公式、作者代码与论文分母差异，见 `FOLLOWUP_NEIGHBORS_BASELINES.md`。

### L-GreCo：显式误差上限与参考误差机制已存在

Mohammadreza Alimohammadi, Ilia Markov, Elias Frantar, Dan Alistarh. *L-GreCo: Layerwise-Adaptive Gradient Compression for Efficient and Accurate Deep Learning*. 作者开放稿 https://arxiv.org/abs/2210.17357 。本轮取得作者公开HTML并提取全文：`lgreco.html`、`lgreco.txt`。

原文优化跨层压缩代价：

\[
\min_{c_1,\ldots,c_L}\sum_\ell \mathrm{size}(\ell,c_\ell),\qquad
\mathrm{s.t.}\ \sum_\ell\mathrm{error}(\ell,c_\ell)\leq\mathcal E_{\max}.
\]

误差是L2压缩误差；上限可跟踪一个已知准确率可恢复的参考压缩设置。它将跨层离散选项看成knapsack类问题，对误差离散化后动态规划。原文也明确讨论误差离散化意味着近似精度问题。它没有在同一固定K下最大化历史全局方向坐标投影。

因此“用参考压缩误差作为保护上限”也不宜笼统声称首次；候选的差异应明确到固定载荷的坐标选择目标及具体保障。

### DAGC：数据异构与拉格朗日通信分配已存在

Rongwei Lu, Yutong Jiang, Yinan Mao, Chen Tang, Bin Chen, Laizhong Cui, Zhi Wang. *Data-Aware Gradient Compression for FL in Communication-Constrained Mobile Computing*. DOI https://doi.org/10.1109/TMC.2024.3504284 ，开放稿 https://arxiv.org/abs/2311.07324 。本轮取v3，2024-11-24更新。精确题名API：`dagc_title.xml`；公开HTML与全文：`dagc.html`、`dagc.txt`。

它分析非均匀客户端压缩，针对数据量不同的客户端分配不同压缩比例/阈值；将分配建模为有总通信预算的n变量优化，并使用拉格朗日乘子化成一维求解。DAGC-R倾向对大数据量客户端保守压缩，DAGC-A处理绝对阈值压缩。

本轮核读的拉格朗日式为跨客户端资源分配的标准形式，例如：

\[
\mathbb L=\sum_i b_i/\sqrt{a_i}+\sigma(\sum_i a_i-A).
\]

它不是在每个客户端固定K下，按历史方向与局部平方范数线性组合选择坐标。但这再次说明“拉格朗日＋梯度压缩”本身不是创新。

**引用注意**：arXiv的comment误附`/mm1`补充材料后缀，正式论文DOI应使用上面的主DOI，不用补充PDF DOI。旧会议版本 *DAGC: Data-Aware Adaptive Gradient Compression*，INFOCOM2023，DOI https://doi.org/10.1109/INFOCOM53939.2023.10229053 ，不要把会议与期刊题名混写。

### 其它新命中：目前不足以否定或确认等价

- *A Theoretical Framework for Energy-Aware Gradient Pruning in Federated Learning*，https://arxiv.org/abs/2603.22465 。OpenAlex摘要中的energy指设备的物理传输/内存成本，提出cost-weighted magnitude pruning；这与更新平方范数能量不是同一约束，不能因energy一词命中就视为等价。仅摘要层面筛查。
- *DSFL: Dynamic Sparsification for Federated Learning*，DOI https://doi.org/10.1109/ICCSPA55860.2022.10019204 。摘要说明CKA逐层相似性稀疏化与每客户端可变K，兼顾通信异构；未见当前固定K坐标投影目标，仅摘要层面筛查。
- *Sparsification Under Siege: Dual-Level Defense Against Poisoning in Communication-Efficient Federated Learning*，https://arxiv.org/abs/2505.01454 ，其摘要的Jaccard结构校准与上传符号聚类用于投毒防御；不能把本工程水印认证直接等同于此种恶意更新防御，也不能据摘要当成当前选择端的相同算法。
- *An All-Reduce Compatible Top-K Compressor for Communication-Efficient Distributed Learning*，https://arxiv.org/abs/2510.26709 ，摘要采用轻量sketch协调跨节点mask以支持无索引All-Reduce；尚无证据是同一历史投影＋能量下限规则。仅摘要层面筛查。

## 本地候选的数学口径：拉格朗日族不是通用精确求解器

下面只在本地文档中讨论候选，不发送检索。令更新\(u=\Delta+r\)，历史参考\(G\)，

\[
A(S)=\sum_{j\in S}u_j^2,\quad B(S)=\sum_{j\in S}u_jG_j,
\quad |S|=K,\quad A(S)\ge c.
\]

若取标准幅值TopK集合T，令\(c=\tau A(T)\)，\(0<\tau\le1\)，则T始终提供可行回退。约束给出

\[
\|u-C(u)\|^2\le\|u\|^2-\tau\|\mathrm{TopK}(u)\|^2
\le(1-\tau K/d)\|u\|^2.
\]

后一个不等式是标准TopK平均能量性质的直接推论，不宜包装成全新收敛理论。

以\(\lambda\ge0\)构造

\[
S_\lambda=\operatorname{TopKIndex}_j\{u_jG_j+\lambda u_j^2\}.
\]

该选择精确最大化对应拉格朗日评分，但一般不能保证找到原始0-1约束问题的全局最优：离散双目标的未支持Pareto点可能被所有线性组合跳过。双分λ找到可行解也不等于消除离散gap。正确称谓可以是“高效的拉格朗日候选族近似”或“候选集合中的最好可行解”，应说明回退和并列值规则。

一个可记录的单步证书是：

\[
D_\lambda=\max_{|S|=K}\{B(S)+\lambda[A(S)-c]\}
\]

为原约束最优投影\(B^*\)的上界；若该λ的选中集合可行，则

\[
0\le B^*-B(S_\lambda)\le\lambda[A(S_\lambda)-c].
\]

证明仅需：任意可行S有\(B(S)\le B(S)+\lambda[A(S)-c]\le D_\lambda\)，而Sλ精确最大化该评分。若λ=0且已可行，或能量约束恰好取等，则该证书为0并可认证本次坐标问题最优；一般情况只报告其上界。如果最终返回的是多个λ候选和T中的最好可行集合，可用\(\min_\lambda D_\lambda-B(S_{\rm out})\)作更紧的同类证书。

历史投影更大仍然不代表当前全局目标必然下降：G可能陈旧或错误，当前真梯度未由此观测。应靠同预算跨轮训练结果及当前方向诊断建立实证，而不跳到非IID普遍收敛主张。

## 检索复核材料与边界

主要索引 `energy_alignment_search_results.json` 逐项列出query、精确URL、返回总数、实际检查条目和原始文件名。14个查询均成功返回；原始响应为 `energy_alignment_arxiv_00.xml` 至05、`energy_alignment_crossref_00.json` 至04、`energy_alignment_openalex_00.json` 至02。

查询包含公开的一般词组：gradient compression + energy/Lagrangian，gradient sparsification + alignment/constraint，global-aware + gradient/sparsification，gradient sparsification + distortion，以及inner product。arXiv的distortion组合返回0；这只是对应元数据查询为0，不能外推全文没有该概念。

本轮没有系统综述式检索全部页，没有运行任何论文作者代码，没有复现外部性能；MultiGS、GRSGD全文缺口仍保留。Wangni经典稀疏化论文的HTML端点只返回转换空页，未据此声称读过全文或核实其公式。

# 时变聚合权重与 EF 残差单位：有限近邻核查

核查日期：2026-11-03 UTC。使用 `paper-lookup` 的公开 API 流程，限定两批一般查询、最多四篇正文比较；没有向外部发送本地未发表公式、代码或完整候选协议，没有运行训练或读取 test/确认结果。以下具体公式均在本地依据公开原文或简单代数分析，不是外发检索词。

## 核心结论

**已找到直接的一手先例，候选的核心操作不宜作为新算法主张。** 2026 年预印本 *Communication-Efficient Agnostic Federated Learning via Faster Convergence and Compression* 的 **AFL-Com** 明确指出：当客户端权重随轮次变化时，把权重放在 EF 外面不能直接控制累计加权全局压缩误差；其解决办法正是**把当前权重乘进新梯度，再与 contribution 单位的旧残差相加、压缩并留误差，服务器直接求和**。

这与“在全局贡献单位里维护 EF accumulator”的代数核心直接重合。AFL-Com 的权重来自 AFL 的时变对偶变量，主算法是全参与梯度；本地候选关心部分参与、活跃样本量归一化和模型差分。场景差别需要说明，但不能仅凭把权重来源换成 active-set normalization 就声称核心压缩规则原创。

另外，Fed-EF 的部分参与分析采用固定参与数 m 与常数 1/m 权重，确实可以 telescoping；EF21-W-PP 使用固定 smoothness 权重及持久梯度估计器，不能因名称中有 weighted/partial participation 就认定已写出时变 a 的旧/新比例缩放。MARINA/SCAFFOLD 未在本次四篇上限内另行全文审查，本报告不对它们作覆盖或排除断言。

## 四篇最直接来源

| 论文/算法 | 已读一手机制 | 是否覆盖候选核心 |
|---|---|---|
| **AFL-Com**，arXiv:2609.36610v1，2026 | 当前 dual weight 放在 EF 内；本地维护加权贡献残差，服务器直接求和；正文明确讨论权重在 EF 外的误差问题 | **直接覆盖 contribution 单位 EF 核心**；权重来源、参与和本地步场景不同 |
| **Fed-EF**，arXiv:2211.14292v1，2022 | 固定 m 个客户端、等权 1/m；未活跃客户端保留残差；Appendix D.4 明确 telescoping | 覆盖部分参与 EF，但不是 changing active-sample denominator 重标度 |
| **EF21-W / EF21-W-PP**，arXiv:2402.10774v1，2024 | 固定 \(w_i=L_i/\sum_jL_j\)，压缩梯度估计差；未参与估计器保持，服务器仍对全体估计器固定加权 | 是加权/部分参与近邻，**所读算法没有时变旧/新 a 残差缩放** |
| **BCRS / OPWA**，ICPP 2024，arXiv:2408.14736v1 | 部分参与先按 active 样本总量归一化，并调整客户端/坐标聚合权重 | 证实该归一化及权重变化场景有公开实现，但所读主算法没有 contribution-EF 记账 |

### 1. AFL-Com：与候选代数核心直接相同

Haomin Bai, Junyan Sun, Sifan Yang, Bo Xue, Lijun Zhang. *Communication-Efficient Agnostic Federated Learning via Faster Convergence and Compression*. arXiv:2609.36610v1，2026-09-29。当前只核实为预印本，未核实正式发表。官方全文：<https://arxiv.org/html/2609.36610v1>。

[§4.1](https://arxiv.org/html/2609.36610v1#S4.SS1) 先讨论不合适的直接组合：本地按未加权梯度做 EF，服务器再乘当前 dual weight。原文明确写：

> “since the weights are applied outside the EF recursion, the accumulated uplink error … is not directly controlled by the worker-side EF guarantee”

随后原文写：

> “We address the mismatch by applying EF to the dual-weighted gradients”

[Algorithm 2 lines 5、8](https://arxiv.org/html/2609.36610v1#alg2) 给出：

\[
\Delta_{w,t}^{(i)}=\mathcal C\left(e_t^{(i)}+q_{t,i}g_{w,t}^{(i)}\right),
\qquad
e_{t+1}^{(i)}=e_t^{(i)}+q_{t,i}g_{w,t}^{(i)}-\Delta_{w,t}^{(i)},
\]
\[
\widehat g_{w,t}=\sum_{i=1}^{m}\Delta_{w,t}^{(i)}.
\]

这里 \(q_{t,i}\) 是时变 dual weight；文中再对服务器下行加一层 EF，但不影响上述上行结论。把公开公式里的新梯度换成多步模型差分、把 dual weight 换成选中客户端归一化权重，仍然是同一“先加权、再 EF”的核心。

原文还明确说明：服务器在上一轮末向各客户端发送下一轮所用的权重，每轮额外 m 个 floats。候选若依赖本轮 active sample denominator，也必须保证压缩前客户端已知道该权重；不能无说明地把这项协商/广播当作自动可得。

该直接先例足以压低核心机制创新性，不需要继续寻找更多标题以证明“最早是谁”。未复核该文全部收敛证明，也未把其梯度、全参与理论迁移成对本地多步/部分参与候选的保证。正文未定位到可确认的作者核心代码链接，本轮没有执行外部代码。

保存：`afl_com.html`、`afl_com.txt`；数学锚点包括 `alg2.l5.m1`、`alg2.l5.m2`、`alg2.l8.m2`、`S4.SS1.p1.m4`、`S4.SS1.p2.m7`。

### 2. Fed-EF：部分参与本身不必破坏 telescoping

Xiaoyun Li, Ping Li. *Analysis of Error Feedback in Federated Non-Convex Optimization with Biased Compression*. 本轮使用已有一手稿 arXiv:2211.14292v1（2022-11-25）：<https://arxiv.org/abs/2211.14292v1>。此处按已读版本引用，不追查发表排版信息。

Algorithm 2（PDF p.8）是局部模型差分加 residual 后压缩。§4.3（PDF pp.11–12）在每轮随机选固定 m 个客户端，未参与客户端保留残差。Appendix D.4（PDF p.51）明确使用

\[
x_{t+1}=\theta_{t+1}-\frac{\eta}{m}\sum_{i=1}^{n}e_{t+1,i}
=x_t-\eta\overline\Delta_{t,M_t}.
\]

这里对全体残差的系数始终是常数 1/m；未参与残差不变。因此其 stale error compensation 讨论与“active 样本总数变化导致 a_i 变化”的问题不同。不能说“Fed-EF 的 partial participation 都不 telescope”，也不能用其标题把候选重标度当作已经核实的原算法部分。

保存位置沿用 `work/literature_followup/residual_freshness/fedef.pdf`、`fedef.txt`，来源记录见同目录 `primary_retrievals.json`。

### 3. EF21-W-PP：固定加权的 estimator 记账，不是当前候选的时变 residual 重标度

Peter Richtárik, Elnur Gasanov, Konstantin Burlachenko. *Error Feedback Reloaded: From Quadratic to Arithmetic Mean of Smoothness Constants*. arXiv:2402.10774v1，2024-02-16。官方全文：<https://arxiv.org/html/2402.10774v1>。

[Algorithm 2](https://arxiv.org/html/2402.10774v1#alg2) 的输入权重是固定常数

\[
w_i=\frac{L_i}{\sum_jL_j},\qquad
u_i^t=\mathcal C_i^t\left(\frac{1}{nw_i}\nabla f_i(x^{t+1})-g_i^t\right),\quad
g_i^{t+1}=g_i^t+u_i^t,
\]

服务器维护 \(g^{t+1}=\sum_iw_ig_i^{t+1}\)。[Appendix G / Algorithm 5](https://arxiv.org/html/2402.10774v1#alg5) 的 EF21-W-PP 对任意子集分布采样：活跃客户端更新上述估计，未活跃客户端设消息为零并保持 \(g_i\)；服务器依然对全体持久估计器用固定 \(w_i\) 聚合。

所以它确有 weighted EF 和 partial participation，但所读公式没有 \(a_i^{t-1}/a_i^t\) 型残差缩放，也没有按 \(\sum_{j\in S_t}n_j\) 重设同一残差的单位。该方法本身也不是传统“未发模型差分累积”的一对一实现，不能只把符号 g_i 称为候选 residual。

保存：`ef21_w.html`、`ef21_w.txt`。论文复现声明请读者联系作者取得实验代码；本轮未取得或执行作者代码。

### 4. BCRS / OPWA：active-n 归一化是具体规则，主算法未给该 EF 修正

Zichen Tang et al. *Bandwidth-Aware and Overlap-Weighted Compression for Communication-Efficient Federated Learning*. ICPP 2024，DOI <https://doi.org/10.1145/3673038.3673142>；本轮读取 arXiv:2408.14736v1（2024-08-27）：<https://arxiv.org/html/2408.14736v1>。

Algorithm 1 lines 7–14 先选 \(S_t\)，令 \(n=\sum_{i\in S_t}|D_i|\)，再用 \(f_i=|D_i|/n\) 加权稀疏模型差分。BCRS / OPWA 进一步随压缩率或坐标 overlap 调整权重。因此“活跃集合决定归一化分母”确是公开实践中的一条规则。

但所读主算法是对差分 TopK 后作上述加权，没有给出时变权重 contribution accumulator 或旧/新权重残差缩放。文中把 Error-feedback TopK 列为 baseline，不足以据此推断它已修正 candidate 的残差单位。该文用于界定场景和排除标题误读，不作为比 AFL-Com 更直接的同式先例。

保存：`weighted_compression.html`、`weighted_compression.txt`。

## 本地代数核对：适用范围和竞争替代

这一节是对一般 EF 恒等式的本地分析，不是新理论主张。

按同一客户端的活跃事件编号 \(\ell=1,\ldots,L\)，设其当次实际聚合系数为正数 \(a_\ell\)。若普通局部 EF 使用

\[
s_\ell=C(\delta_\ell+r_{\ell-1}),\qquad
r_\ell=\delta_\ell+r_{\ell-1}-s_\ell,
\]

则累计加权压缩误差为

\[
\sum_{\ell=1}^La_\ell(\delta_\ell-s_\ell)
=a_Lr_L-a_1r_0+
\sum_{\ell=1}^{L-1}(a_\ell-a_{\ell+1})r_\ell.
\]

当 a 改变时，中间项一般不消失；这不意味着必然不收敛，只意味着不能直接套用常权重的简单 telescoping 论证。AFL-Com 已公开指出并处理了这个加权误差不匹配。

contribution 单位的 EF 令 \(v_\ell=a_\ell\delta_\ell+e_{\ell-1}\)、\(q_\ell=C(v_\ell)\)、\(e_\ell=v_\ell-q_\ell\)，则

\[
\sum_{\ell=1}^L(a_\ell\delta_\ell-q_\ell)=e_L-e_0.
\]

对正齐次 TopK，以及一致的并列值处理，令 \(e_\ell=a_\ell r_\ell\)，可等价写成

\[
u_\ell=\delta_\ell+\frac{a_{\ell-1}}{a_\ell}r_{\ell-1},\quad
s_\ell=\operatorname{TopK}(u_\ell),\quad
r_\ell=u_\ell-s_\ell,\quad q_\ell=a_\ell s_\ell.
\]

这里旧系数指**上一次活跃事件所用的系数**。未活跃时 contribution 残差保持，不需要把 a=0 代入除法；首次残差为零。FP32 的不同乘除顺序不保证逐位相等。对不具正齐次性的量化器，该 local-rescale 表达不自动等价，但 contribution-EF 的记账恒等式仍成立。

必须同时写清三种聚合口径：

- **活跃样本量归一化**：\(a_i^t=n_i/\sum_{j\in S_t}n_j\)。只有总 active mass 变化时，重复活跃客户端的该系数才变化；全参与固定数据或每轮 active mass 相同可退化成常数。
- **固定总体权重或固定 HT 权重**：若客户端目标权重 p_i 和包含概率 π_i 都固定，活跃时使用 \(p_i/\pi_i\)（外加一致的固定服务器系数），每个客户端的非零系数固定。此时 contribution-EF 与标准 local-EF 只差固定单位，候选没有额外机制收益。HT 是必须认真比较的简单替代。若 π_i 也随时间变，不能继续把它当固定权重情形。
- **目标/抽样性质**：保持 active-ratio 聚合规则的压缩记账，不会自动消除该规则自身的 ratio-estimator 偏差、client drift 或部分参与方差；换 HT 又改变了聚合估计器和可能的方差。两者不能只用一次瞬时残差误差比较就宣称谁更优。

因此该问题只能定位为**某些时变系数协议下的 EF 单位一致性**，不是一般 FL 或一般 partial participation 的普遍缺陷。AFL-Com 已覆盖核心修正，若做工程适配应标明来源与限定范围，不宜包装成新压缩范式。

对 active-n 规则还可进一步收窄：令 \(p_i=n_i/N\)、\(\kappa_t=N/\sum_{j\in S_t}n_j\)，则 \(a_i^t=\kappa_tp_i\) 是固定客户端权重乘公共时变标量。以 \(e_i=p_ih_i\) 换单位，正齐次 TopK 下有 \(q_i=p_i\operatorname{TopK}(\kappa_t\delta_i+h_i)\)。这与把公共时变缩放放进 EF 输入严格同构；当新输入本身代表梯度步时，也可理解为 update-units/变步长 EF 的记账。该代数等价进一步限制了新意范围；本轮未再检索这类变步长文献，不能把这里的说明冒充另一个已核实文献引用。

## 两批公开检索与证据保存

两批、七个查询，每条仅筛前 15 个 OpenAlex 结果，全部 HTTP200。宽查询噪声较大；命中总量不是已审查论文数。

| 批次 | 公开一般查询 | 返回总量 |
|---|---|---:|
| 1 | `error feedback time varying aggregation weights partial participation federated learning` | 1107 |
| 1 | `importance weighted error feedback weighted residual compression` | 10330 |
| 1 | `EF21 partial participation importance sampling weighted aggregation` | 6 |
| 2 | `"error feedback" "importance sampling"` | 111 |
| 2 | `"EF21" "weighted"` | 81 |
| 2 | `"error feedback" "weighted aggregation"` | 70 |
| 2 | `"error feedback" "time-varying" "weights"` | 1162 |

原始响应：`batch1_openalex_{0,1,2}.json`、`batch2_openalex_{0,1,2,3}.json`。`batch1_log.json`、`batch2_log.json` 保存 exact query、URL、UTC 时间、状态与原始路径。成功取得的三篇官方 HTML 及提取文本在同目录，`primary_log.json` 保存 URL、时间、字节数和 SHA-256。Fed-EF 复用已有第一方稿，没有再次下载。

到此停止检索；不实施实验，不再追查 MARINA/SCAFFOLD 或增加第五篇，以免把本次直接重合核查扩成完整综述。

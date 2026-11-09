# FedALA / FedCAC 有限原文核查

核查日期：2026-11-03。仅核对这两篇 primary paper，不做训练，不据此宣称 novelty。HTML 均从 arXiv 原文获取；FedCAC 对存在歧义的公式另以同版本 PDF 文本复核。

## FedALA

**来源**：Zhang et al., *FedALA: Adaptive Local Aggregation for Personalized Federated Learning*, arXiv:2212.01197v4（AAAI 2023）；[原文 Method / Adaptive Local Aggregation](https://arxiv.org/html/2212.01197v4#Sx3.SSx2)。本地 `fedala.html`, `fedala.txt`。

- 式(3)：`theta_init_i = theta_local_i + (theta_global - theta_local_i) ⊙ W_i`；式(4)仅在最高 p 层学逐元素实数权重 W，其余低层用 1，故完全接收 global 低层。W 经 clip 约束在 `[0,1]`。
- **接收是否有益由 client 的本地目标决定**。式(5)：`W_i ← W_i - eta ∇_W L(theta_init_i, D_i^{s,t})`；client 每轮采样本地数据的 s%，冻结 global/local 模型参数，只学 W，然后做常规本地训练。Alg.1 第9–17行给出 client 学权重、clip、初始化；t=2 先学到收敛，之后通常仅一 epoch。
- **上传不是稀疏上传**：Alg.1 第20行上传完整训练后本地模型，第21行 server 数据量加权聚合；第7行下发一个完整 global。引言明确说上下行各一个模型、通信量等同 FedAvg。
- **与 receiver gradient-block gate 的重合**：式(5)及 Appendix A 式(7)给出 `∇_W L = ∇_{theta_init}L ⊙ (theta_global-theta_local)`。这已是用 receiver 本地梯度与候选位移来调节接收量的机制。若把 W 换成 block 二值 gate / sign 判别，则粒度与求解方式有差异，但“receiver 用本地梯度评估并局部接收”不能作新颖性主张。
- **与 sparse donors + exclude self 的差异**：FedALA 以 server 聚合完整 global 为候选；Alg.1 第21行没有为 receiver 排除 self 的聚合，也没有稀疏 donor 支持。方法中没有 independent local minibatch backtracking 接收检验；这只是这篇原文未包含的机制，不能推出领域空白。

## FedCAC

**来源**：Wu et al., *Bold but Cautious: Unlocking the Potential of Personalized Federated Learning through Cautiously Aggressive Collaboration*, arXiv:2309.11103v1（ICCV 2023）；[§3.3](https://arxiv.org/html/2309.11103v1#S3.SS3)、[§3.4](https://arxiv.org/html/2309.11103v1#S3.SS4)。本地 `fedcac.html`, `fedcac.txt`, `fedcac.pdf`, `fedcac_pdf.txt`。

- **重要参数判据在 client 算**。§3.3 式(5)：`s_k^t = |(theta_k^{t,E} - theta_k^{t,0}) · theta_k^{t,E}|`，以完整本地训练前后变化代替单一末步梯度；式(6)在每层挑 sensitivity 最大的 top-τ 为 critical，得到二值 mask `M_i`。
- **server 算的是协作资格**。§3.4 式(7)(8)用 client 上传的 mask 两两关系、轮次 t、β 计算 threshold 并选择 `C_i`；它没有用 receiver 的本地 loss/gradient 判每个候选反馈对 receiver 是否有益。
- **上传 mask + 完整模型，下发两个完整模型后 client 拼合**。§3.1 Steps1–5、Alg.1 明确上传 `(w_i,M_i)`。式(9)全体均值 `w_bar=(1/N)Σ_i w_i`；式(10)定制模型 `u_i = (1/(|C_i|+1)) Σ_{j∈C_i∪{i}} w_j`；式(11) client 初始化 `w_i^{t+1}=u_i⊙M_i + w_bar⊙(1-M_i)`。故 mask 控制协作来源，不是少传 τ 参数。§3.3 还单独计 mask 约为完整32-bit模型传输量的3%。
- **不排 self**：虽然式(8)定义的“其他协作 client 集”排除 i，式(10)又明确加入 `{i}`，因此不能把它称为 exclude-self donor average。
- **与 receiver gate 的差异**：M 标识 client 自己哪些坐标对本地任务敏感，它不判定 `u_i-w_i` 或 `w_bar-w_i` 是否与 receiver 本地梯度相容；融合时没有 local gradient accept/reject 或 independent-minibatch backtracking。反之，“mask + server反馈 + client部分融合”这一泛化框架已被式(5)–(11)覆盖。
- **原文符号疑点，不替作者修正**：§3.4 将 `O_ij=||M_i-M_j||_1/(2n)` 称为 overlap，式(8)却选 `O_ij≥Threshold`，叙述称其选择更相似 client。对二值 mask，该 L1 式本身是差异量，与叙述方向不一致。相同表达也出现在同版本 PDF（正文第4页附近；`fedcac_pdf.txt` 501–539行），不是本次 HTML 文本抽取造成。这里仅报告各端职责和明确的聚合/融合公式，不据此断言实现实际选相似还是相异 client；未核对作者代码。

## 对有限诊断的直接约束

可尝试的具体组合是：从通信预算允许的 sparse donors 产生**排 self**的候选反馈，在 receiver 用本地小批数据对候选位移做 block 级梯度判别，再以独立本地小批回溯检验接收幅度。上述两篇没有呈现这一完整组合，但：

1. FedALA 已覆盖 receiver 本地目标驱动的逐元素部分融合，并明确使用梯度×候选位移；不能把 gradient gate 本身包装成全新思想。
2. FedCAC 已覆盖 client mask → server 个性化协作反馈 → client 按坐标融合；不能把这个闭环本身包装成新颖性。
3. 排 self、稀疏 donor 的预算核算、block二值门以及独立批回溯，是与这两篇可精确区分的实现条件；在未查尽直接邻居、未做等预算有效性检验前，仅能称待测试设计，不能 claim novelty 或异构性收益。

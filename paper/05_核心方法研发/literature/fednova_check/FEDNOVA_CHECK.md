# FedNova 无 momentum 公式与压缩组合：有限核查

核查日期：2026-11-04。只读一篇 primary：Wang et al., *Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization*, [arXiv:2007.07481v1](https://arxiv.org/html/2007.07481v1)。另严格完成一次精确压缩检索，无追加查询、无第二篇primary全文、无repo、无训练。用户允许合理组合，组件已有先例不构成停止理由。

## 公式：拟用写法正确，但 d 与论文命名不同

设同轮所有client从同一global `x_t` 出发，用固定同一局部学习率η、vanilla SGD、无momentum、无proximal项，实际完成正整数 `τ_i` 次更新。定义 `Δ_i=x_i,end−x_t`，`p_i=n_i/Σ_j n_j`。

论文[§4.1](https://arxiv.org/html/2007.07481v1#S4.SS1)把 `d_i` 定义为**平均梯度**：`a_i=1_(τ_i)`、`||a_i||_1=τ_i`，因此 `d_i^paper=Σ_k g_i,k/τ_i`，且 `Δ_i=−ητ_i d_i^paper`。

[§5式(12)及紧邻下一段](https://arxiv.org/html/2007.07481v1#S5)明确给出等价的**模型增量**写法：

`u_i = Δ_i/τ_i`

`τ_eff = Σ_i p_i τ_i`

`x_(t+1) = x_t + τ_eff Σ_i p_i u_i`

这就是拟用的无momentum版本。若代码把 `u_i` 也命名为d可以，但应注明它是归一化模型增量，**不要在server再次乘η或改变正负号**。论文还允许其他τ_eff，不过上式是它为与FedAvg更新尺度对应而明确推荐的选择。所有τ_i相同时，上式应代数退化为FedAvg（不要求p_i相等）。

局部LR在各步变化、momentum、proximal项会改变累积系数 `a_i`，应按 `||a_i||_1` 归一；不能把“无momentum”单独当成 `/τ_i` 对所有求解器都成立的充分条件（§4.1各solver例子）。

## τ_i：实际优化步数，不是统一的“官方ceil”

[§1与§2](https://arxiv.org/html/2007.07481v1#S2)原文均写：相同本地E epochs、数据量n_i、batch B时，`τ_i=floor(E*n_i/B)`；理论把τ_i视作可变实际局部更新次数。**本次primary不支持“官方τ_i=ceil(n_i/B)*E”的说法。**

实现以实际 `optimizer.step()` 计数为准：每epoch重新遍历完整DataLoader、`drop_last=False`、没有梯度累积/早停/跳过step时，`τ_i=E*ceil(n_i/B)`；`drop_last=True`时为 `E*floor(n_i/B)`。不要把论文的floor近似、实际ceil循环或 `floor(E*n/B)` 与 `E*floor(n/B)` 混用。若τ_i=0不可除，应提前处理空/不足一批的client，并记录实际参与集合。

部分参与时，[附录H Alg.1第6行](https://arxiv.org/html/2007.07481v1#A8)对选中集合显式除以 `Σ_(i∈S_t)p_i`；实现可定义 `q_i=p_i/Σ_(j∈S_t)p_j`，然后两处都用q：`τ_eff=Σq_iτ_i`，`Σq_i u_i`。这只是该轮算法定义；非均匀参与的全局目标期望还依赖采样设计，不宜自动套用全部客户端的理论结论。

## 与 compression 的已知关系和本轮可行定义

原论文§2末段明确称 FedNova **compatible with and complementary to gradient compression/quantization**，引用其[43–48]。但该文没有给出tensor层RMS打分TopK+EF专用算法、压缩误差界或该具体组合实验。因此组合有原文动机支持，性能与理论不能从原版直接继承。

唯一精确检索 `all:FedNova AND (all:compression OR all:compressed OR all:sparsification OR all:quantization)` 返回0条；原始XML和解析JSON保存在本目录，说明见 `COMPRESSION_SEARCH.md`。这只表示该受限查询未命中，不表示组合不存在，不用于novelty判断。

可清楚定义当前压缩变体为：先构造 `u_i=Δ_i/τ_i`；若用EF，则 `v_i=u_i+r_i`、`s_i=C_K(v_i)`、`r_i←v_i−s_i`；server `x←x+τ_effΣp_i s_i`。这是**待检验组合定义**，不是本次primary的原版算法。EF的r与u必须同单位；如果τ_i跨轮变化，不可把旧raw-delta residual直接除以新τ混入。若无EF则明确写r=0。

各client支持不同时，缺失坐标按该client的压缩输出0处理，聚合保持同一组固定p_i；再按“此坐标上传者”重归一会引入坐标相关权重，不再是标准FedNova式聚合。预算须含值、索引、τ_i/normalizer；若server本已由协议知道实际τ_i，可说明无需重复传递。

## 固定2×2及必要条件

| | 普通幅值TopK | tensor层归一化分数TopK |
|---|---|---|
| FedAvg式增量聚合 | `C(Δ)`，server `Σp_i C(Δ_i)` | 同总K、改选择分数，传原增量值 |
| FedNova式归一聚合 | `C(Δ/τ)`，server `τ_effΣp_i C(u_i)` | 同总K、改选择分数，传原归一增量值 |

这个2×2可以分开检验聚合归一与支持选择分数，以及二者交互。固定分区、起点、batch顺序、E、η、K、EF有无及计算/通信口径；无需网格调参。若EF不同单位按各单元定义保留，明确它也是归一空间的一部分。已有dense FedAvg/FedNova结果可作压缩损失参照；若没有，本轮只能比较四个压缩组合，不能独立量化压缩相对dense的损失。

一个代数事实须保留：对单client乘正标量 `1/τ_i`，普通TopK排序不变；若层分数为 `|u_j|/RMS(u_tensor)`，τ也相消（实际固定ε、零层和舍入边界除外）。因此无EF时Nova不会因为该标量本身改变普通/层归一TopK的支持；作用主要在聚合中各client贡献尺度。若EF存在，历史状态会影响支持。层归一分数只用于选择，发送应仍是原u值，除非明确另定义并还原层缩放。

当前旧协议“一client一epoch”但n_i不同造成τ_i不同，与FedNova的动机直接吻合。归一校正的是局部进度诱导的权重偏置，不保证消除全部标签异构性或压缩偏差；最终判断仍用固定2×2有限训练的实测结果。

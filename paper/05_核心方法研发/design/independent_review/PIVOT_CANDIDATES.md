# 换方向：一条值得廉价诊断的路径，一条暂不立项

日期：2026-11-03。范围：一轮独立方法判断，不启动闭环训练、不新增长检索、不访问 test。应用 `hypothesis-generation` 与 `scientific-critical-thinking`：将观测、机制解释、竞争解释和停止条件分开；没有给候选自动评分，也没有把缺少近邻命中当作新颖性。

**判断：目前没有一个已完成近邻核查、可以直接宣布为新核心算法的候选。可以保留一个“改变本地目标”的廉价研究诊断；不建议同时开启第二条完整开发路线。** 历史方向选择已经止损，不能把新鲜参考、τ、sketch 或拉格朗日系数当成新一轮主线。

## 已验证事实给出的方向约束

1. 原 signed 历史方向选择三组 CPU 最后五轮开发验证准确率平均收益约 .337 pp，末轮差值明显波动。它作为小幅经验基线保留，不支持核心贡献定位；本报告不使用 test。
2. 同一真实 EF 状态下，完整 `U=Σp(Δ+r)` 反而比 Top-K 验证 loss 高 .0622；旧 g 的 signed 选择 loss 高 .0181。残差“最终补上”不是当前模型立即需要它的证据。
3. 真 TRAIN 梯度 oracle 在 `w+Q` 与 w 求值，signed 准确率增益分别 +2.15 pp 与 +.05 pp；新鲜 dense `ΣpΔ` 的 signed 控制是 −.45 pp。不能从 oracle 成功推导可部署全局参考会成功。
4. 这支持重新检查**训练出的更新与可上传更新是否存在功能差距**；但未证明差距是异构数据特有的。也没有证据支持“局部大梯度就是无用”“局部熵低就是全局分布偏差”等简化。

来源：`oracle_replays/cpu_topk20_tau90/`、`cpu_topk20_tau90_v2/`、`CORE_DECISION_CRITERIA.md`。同状态 probe 不是独立训练重复。

## 路径一：本地目标显式考虑最终可传模型的功能差距

### 问题与候选假说

普通本地训练优化的是 dense 本地模型 `w+v`，实际上传的是 `C_K(v+r)`。两者在本客户端数据上的预测可以不同。候选假说是：**在保持同一上传预算下，让本地训练同时优化其可传版本，可以比训练完才压缩更少损失任务信息。**

竞争解释至少有三个：只是普通正则化；dense teacher 自己已严重局部过拟合，传其预测反而坏；每客户端的可传模型仍不等于加权全局模型，局部功能保持未必有全局作用。这条路径不能先写成解决了异构聚合。

### 可部署的具体目标与状态

客户端本轮起点 w、历史 EF 残差 r，当前参数差 v。按与最终通信一致的规则每 J 个本地步更新一次 mask：

\[
M=\operatorname{TopKMask}(|v+r|),\qquad
\widetilde w=w+M\odot(v+r).
\]

短时间固定 M（不宣称对离散 Top-K 精确求导），定义可传 shadow 的梯度通过 `M⊙v` 回传：

\[
\mathcal L_i(v)=\frac{\operatorname{CE}(f_{w+v}(x),y)
+\lambda\operatorname{CE}(f_{\widetilde w}(x),y)}{1+\lambda}.
\]

这是最简单、无需 teacher 温度的第一候选。可传 shadow 使用同一真实标签，不能用模型自造标签当任务真值。不要同时加 KL、蒸馏温度、类别权重和多个 λ；第一次固定 λ=1，J 取“一个 epoch 中间与结束各更新一次”或一个预先固定 minibatch 间隔，只选一种。

训练结束依原规则重新编码当前 `u=v+r`，发送 K 个原 FP32 值和索引，残差 `r'=u-C_K(u)`；如果交付失败，继续遵守原待接纳残差协议。训练中的 M 只是优化辅助状态，最终消息仍需认证真正发出的支持和值。

此目标不应称“普通 FedProx + Top-K”：FedProx 的近端项拉向 w；FLARE 的残差相关正则拉向 `w+r` 的参数位置；这里损失直接评价**依照真实码率生成的可传 shadow 模型**。不过这种公式差异仍不足以证明新颖性。

### 成本

- 上行保持同 K、相同索引和值；不额外上传标签直方图或 oracle 梯度。
- 每个采用辅助损失的 minibatch 多一次 shadow 前向/反向，实际计算可能接近加倍；不能只凭字节相同称“更轻量”。可仅最后一部分本地步启用，但第一次诊断不要搜索启用比例。
- O(N) shadow 参数/激活开销；一个 mask O(N bit 或 K indices)，每次刷新增加 Top-K 选择成本。需要函数式模型或临时参数副本，不能在计算图中破坏原参数。
- 当前全参与、信任服务器不变；它没有把本地数据送到服务器。模型上传仍可能泄漏信息，原水印认证不提供额外隐私保证。

### 直接近邻与尚未核实的差异

- 已读 FLARE：*Sparse Training for Federated Learning with Regularized Error Correction*，本地 `residual_freshness/flare.txt`，Eq.3/Algorithm1 前若干步对残差锚点加参数正则，再普通 Top-K。直接挑战“把残差带进本地目标是新颖”的主张。
- 已读 SAEF/SA-PEF：用残差偏移本地训练起点并调整剩余误差；已读 Fed-EF 说明偏压缩与残差的联邦理论已存在。不能把 residual-aware training 泛称新范式。
- compression-aware training、quantization-aware training、prediction/consistency distillation 是明显概念近邻。本轮未对其 federated sparse-shadow 具体版本完成检索，因此**新颖性未解决**。若存在同码率 sparse-shadow CE 直接方法，应转为基线，而不是换个名字坚持原创。

### 最低证伪实验（不先做 20 轮）

先在已保存的一个 Top-K 开发 checkpoint，固定真实 r 和本地批次，只运行**一轮**新本地训练，两组：

1. 原 CE + Top-K-EF；
2. shadow CE 目标（λ=1，固定刷新规则）；

实现 fixture 再核验：将 shadow 参数强制等于 dense 参数时，归一化双 CE 的梯度应与原 CE 相同，排除纯损失缩放；这只是实现校验，不另占一个重复训练名额。主方案保持上式的归一化，所有组学习率一致。

全部相同最终 K、真实残差和模型初始化。先测压缩前后本地 held-out 功能差距，再在共同全局模型下聚合所有实际 sparse 更新，评价统一开发验证 loss/accuracy；报告 codec 与本地时间、类准确率。**本地 gap 下降而全局 loss 无改善，不能算成功。**

只有显示“实际 sparse 全局任务收益”且梯度缩放 fixture 通过，才进入一组不超过 20 轮的 paired pilot：Top-K-EF、shadow、直接强近邻 FLARE（如可正确实现）。只固定一个 λ/刷新规则，不开网格。若一轮诊断无信号，停止；若 20 轮最后五轮不优于 Top-K 或强近邻，停止核心贡献开发。单次 pilot 成功也只意味着值得新的独立划分验证。

异构主张另外需要 skew 与 mixing=1 的差异，不能由本地 shadow 故事直接推出。若同样改善，应定位为一般压缩感知本地优化，而不是非IID专属贡献。

## 路径二：类别条件下的任务敏感压缩几何——目前只值得离线否证，不立项

### 问题与可计算诊断

幅值 Top-K 优化参数欧氏误差，但同样的参数扰动在不同类别/客户端上造成的预测误差可能不同。候选度量是用训练来源的局部敏感度构造对角权重：

\[
h_{ij}=\frac1{|\mathcal C_i|}\sum_{c\in\mathcal C_i}
\frac1{|B_{ic}|}\sum_{(x,y)\in B_{ic}}
\left[\nabla_j\log p_w(y\mid x)\right]^2,
\]

\[
S_i=\operatorname{TopKIndex}_j\{h_{ij}u_{ij}^2\},\qquad
s_i=1_{S_i}\odot u_i.
\]

这是类均衡的经验 Fisher 对角**诊断**，不是 Hessian 等价式，也不保证全局损失。类均衡只改变压缩敏感度评估，不应悄悄改变正式样本加权训练目标。客户端没有的类别无法估计；缺类恰是该规则最严重的局限，不能填成零然后宣称少数类保护。

成本：原值同 K 上传；额外本地梯度测量，精确 per-example Fisher 很贵，不能拿 batch-mean gradient 平方冒充。每类一次小 batch 的梯度平方仅是另一种 proxy，必须换名字并说明。需要 O(N) 权重；多类别反传显著增加时间。若构造共享全局权重，上传 N 个值的成本直接破坏原预算，不能免费假设。

### 为什么暂不建议作为第二主线

importance/Fisher-weighted compression、曲率加权量化/剪枝是显著近邻；本轮只核实过 L-GreCo、DAGC 和 RegTopK，没有核实最直接 Fisher 先例。仅把它们换成“类别感知”不足以构成原创。更重要的是本地 Fisher 可能专门放大本地类偏好，反而加剧异构；当前 oracle 已说明“局部新鲜”不等于“全局有用”。

可允许一次非常廉价的**离线可计算性检查**：在固定共享状态，用极少固定 TRAIN 样本算明确标记的 sensitivity proxy，与 Top-K 比较同 K 后统一验证 loss；加入不分类均衡的 proxy 控制。若支持几乎不变、只改善本地目标或全局损失恶化，当场否定，不进入闭环。即使它赢，也必须先查直接近邻，不能马上写成核心算法。

## 本轮选择

推荐只讨论路径一的一轮 shadow 诊断；路径二暂不实施。若 shadow 也无法在实际 sparse 聚合后的任务损失上给出信号，现有材料确实没有现成可辩护的新 AI 核心贡献。届时需要重新选研究问题，或调整整篇论文贡献重心；不应为了填充“AI 亮点”继续堆小改动。

流程来源：本地 `hypothesis-generation/SKILL.md` 与 `scientific-critical-thinking/SKILL.md`。两项技能列出的方法论参考 Kassis et al., *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, 2026, arXiv:2609.00065；本轮未在线复核该条，仅登记流程来源，不用其支持 FL 方法主张。正式跨模型审计仍未完成。

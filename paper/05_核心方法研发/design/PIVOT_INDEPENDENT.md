# 一轮有限转向构思

日期：2026-11-03。使用 `hypothesis-generation` 区分观察、假设、竞争解释与否证；使用 `scientific-critical-thinking` 检查代理目标、混杂和证据范围。研究材料仅留本地，未读确认或官方测试，未启动新训练。历史 g、score、tau、sketch family 已结束。

**本轮结论：不保留 label-bias 修正作为主路径；只留一个结构不同的压缩后本地修复候选，最多做一次廉价否证。当前没有足够证据称它是可发表新算法。**

## 已完成的 saved-state 结构检查

读取 `checkpoint_probes/cpu_topk20/shared_state.pt`，按参数层计算样本权重加权的残差平方能量；结果保存为 `pivot_layer_geometry.json`。这里只是一个开发状态的描述，不包含因果判断。

| 参数块 | 参数数 | 残差能量占比 |
|---|---:|---:|
| classifier.1.weight | 131072 | 95.1306% |
| features.3.weight | 4608 | 3.8593% |
| classifier.3.weight | 640 | 0.4965% |
| classifier.3.bias | 10 | **0.00725%** |

classifier.1.weight 本身约占全部参数的 95.77%，因此不能把它的残差总量大直接称为“异常集中”；但它确实承载了这个模型绝大部分残差总量。不能用十维最终 bias 的机制直接解释这些总量。

父代理提供的 `||r||≈3.4||delta||`、cos(r,delta)≈.47 作为转向背景；本轮没有将它们升级为所有客户端/状态的一般规律。残差大也不自动证明它已失效，EF 在强稀疏下本来就会累积未发送量。

## 候选一：解析 label-prior bias 中心化——本轮淘汰

设 pi_i 为本地 TRAIN 标签比例、pi=Σ_i p_i pi_i。对最后分类 bias b，CE 梯度为 mean(pred_i)-pi_i。局部修正目标可写成

\[
F_i^{center}(w)=F_i(w)+(\pi_i-\pi)^T b,
\qquad \sum_i p_i F_i^{center}(w)=\sum_i p_iF_i(w).
\]

它在最后 bias 梯度中抵消本地先验偏移，且全局加权目标恒等不变；一次共享 C 维计数及广播 C 维比例，之后额外通信为零，存储 O(C)。这比泛泛把 FedLC 和 TopK 拼接更具体，但 label-aware gradient correction/FedLC 等近邻风险仍然很高。

**淘汰原因：本轮想解释的是漂移与压缩残差的主体耦合，而最终 bias 仅承载 0.00725% 残差能量。** 这不证明 bias 对预测不重要，却足以否定“它能直接消除观察到的大部分残差质量”的窄机制。没有理由现在投入训练，或把它扩成 prototype/control-variate 大方案来挽救。FedLC 可作为现成基线，不能把此构思当成已形成的新贡献。

## 唯一保留候选：压缩后，在将发送的支持上做一次本地目标修复

### 可部署定义

现有普通 local SGD 得到 delta_i，旧 residual 为 r_i。先按普通 TopK 决定支持，随后才修复将发送的值；不使用历史全局方向，也不依赖本轮全局 oracle。

\[
u_i=\delta_i+r_i,\quad M_i=\operatorname{maskTopK}(u_i),\quad s_i^0=M_i u_i.
\]

在客户端可直接构造的 shadow model `w_t+s_i^0` 上，用一个本地 TRAIN 均匀抽样 minibatch B_i 计算梯度：

\[
g_i^q=\nabla\widehat F_{i,B_i}(w_t+s_i^0),\qquad v_i=-\alpha M_i g_i^q.
\]

只允许修改原支持内的数值，不改变 K。固定 theta=1/2，以当前 local learning rate 作为 alpha，不搜索它们：

\[
z_i=v_i\min\left\{1,\frac{\sqrt{(1-\theta)\|s_i^0\|^2}}{\|v_i\|}\right\},
\qquad s_i=s_i^0+z_i,\qquad r_i^{next}=u_i-s_i.
\]

v_i=0 或 s_i^0=0 时令 z_i=0。服务器仍更新 `w_{t+1}=w_t+Σ_i p_i s_i`。发送的 FP32 数值经过本地修复，论文应明确它不再是原 u_i 的未经修改分量。

这个干预把额外计算放在**压缩之后形成的模型**上，意图检查本地学习目标能否修复实际保留值与本地目标的不匹配。它与 FLARE 的参数拉向 `w+r`、SA-PEF 的本地训练前 residual 偏移不同；也不是再换 filler score。

### 可以严格说明的性质与局限

由于 z_i 只在 M_i 上，而 `u_i-s_i^0` 只在补集上，精确算术下

\[
\|r_i^{next}\|^2=\|u_i-s_i^0\|^2+\|z_i\|^2
\le \|u_i\|^2-\theta\|s_i^0\|^2
\le (1-\theta K/P)\|u_i\|^2.
\]

因此可以保留一个弱于 TopK 的收缩界，且 `s_i+r_i^{next}=delta_i+r_i` 守恒；数值实现需另行检查舍入。**修复会增加当前残差平方范数，不能宣传为减小残差的方法。** 它只可能以受限的额外残差换取当前发送方向的作用；累积 EF 仍会将这次修复的相反量记入后续债务，不能声称永久消除了 client drift。

更大的风险是：`w+s_i^0` 不是实际的 `w+Σp_i s_i^0`。本地修复可能强化异质目标、使全局表现更差。残差契约性不代表学习优势；此风险必须通过诊断否定，而不能被公式掩盖。

### 成本

- 每客户端每轮额外一个 64 例 TRAIN minibatch 的前向/反向；同一个模型对象可以重装为 shadow model，无需长期保留第二个模型。
- 稳态仍保留一份 O(P) residual；梯度/模型临时内存属于常规 SGD 量级，原 K 支持可复用，不增控制变量历史。
- 额外上传/下载为 **0**；仍上传 K 个 FP32 值和 uint32 索引。当前 P=136874、K=6843 时，20 客户端 sparse 消息分析总量仍为 1096160 bytes/round。没有把额外梯度上传。
- 若工程实现使用独立 shadow model copy，必须据实计入额外模型内存；不能仅凭上式声称内存为零。

### 最近邻与新颖性风险

已本地阅读的最近邻包括 FLARE（`residual_freshness/flare.txt`，第4–5页）、SA-PEF（arXiv:2601.20738，第2–3页）及 Fed-EF（arXiv:2211.14292）。FLARE/SA-PEF 已明确处理 residual、梯度失配、本地漂移和稀疏化的耦合，因此这些问题本身不是新发现。

本候选还与 compression-aware training、固定稀疏支持上的 fine-tuning、compressed-model distillation、可学习收缩压缩器直接接近。当前没有完成覆盖这些方法的系统查新，**不能宣称首次提出**。DIANA/SCAFFOLD 等控制变量也表明，拿现成漂移修正器再压缩通常不够构成贡献。这里仅有“前后顺序与支持内值修复”的结构差别，尚不足以确认发表价值。

## 一次廉价诊断与停止规则

只用已经保存的 `cpu_topk20` 一个开发共享状态，真实 residual 和 delta 不变；全部20客户端各取一个均匀 TRAIN minibatch，固定新 seed，不更换批次、不调 alpha/theta。

三组就够：

1. **A：原 TopK-EF**，已有 saved delta/r。
2. **B：同等额外计算的普通 local SGD**，在 `w+delta_i` 上用该 minibatch 额外做一步，再 TopK+EF。
3. **C：上述压缩后修复**，使用同一 minibatch、相同步长和 K，修复 `w+s_i^0` 后发送。

比较一次全局聚合后的原始验证 CE/accuracy，并记录真实 residual 范数、支持/K/bytes/守恒、repair 范数和 clipping 是否触发。不得把客户端数当作训练 seed 重复。无需新 epoch、长训练、官方测试或候选网格。

**预先停止条件：C 未同时降低 A、B 的验证 CE，或 accuracy 比 A 下降，就停止本候选，不再调修复系数、增加步数、改 minibatch/class balance 来挽救。** 这只是给当前探索设置的投入门槛，不是统计显著性标准。即便这一状态通过，也只说明值得另行决定是否做一项小型独立验证，不自动启动训练矩阵。

B 是关键竞争解释：若普通多一步 SGD 同样有效，则观察到的效果可能来自额外计算，而非压缩位置感知。若 C 只是降低本地 shadow loss、全局 CE 却无益，就直接否定当前的局部目标代理。一次正结果也不能证明收益专属于异质性；那仍需要以后独立设计识别。

初稿到这里时尚未运行 A/B/C。随后 root 明确选择执行这一次三组诊断；实际结果见下，不再扩展。

对应预测/竞争解释已写 `pivot_prediction_matrix.csv`，并使用 hypothesis-generation 的本地检查器进行结构校验。它只检查声明完整性，不验证科学正确性，也不替人决定候选。

## root 授权后的唯一一次诊断结果

脚本 `probe_sparse_repair.py` 的 15 个 fixture 通过，然后只运行一次真实 CPU 单线程诊断，约 3.38 秒；没有参数、batch 或步数搜索。结果保存在 `sparse_repair_probes/cpu_topk20_fixed_abc/`。

| 组 | 验证 CE | accuracy | 相对 A 的 CE 差 | 相对 A 的 accuracy 差，pp |
|---|---:|---:|---:|---:|
| A：原 TopK-EF | 1.596964630 | 42.2500% | 0 | 0 |
| B：同计算额外普通 SGD | **1.558726578** | **43.8500%** | **-0.038238052** | **+1.599997** |
| C：支持内压缩后修复 | 1.583601044 | 42.6000% | -0.013363586 | +0.349998 |

**按原固定判据停止：C 未胜过 B。** 相对不多算一步的 A 有改善，不足以支持压缩后修复的特殊价值；额外普通 SGD 的竞争解释更强。保留结果，不调整修复参数/批次/步数，不启动此候选的闭环训练。

事后有限存档检查 `sparse_repair_postcheck.json` 通过：核对 60 个 client×variant 状态、全部输入与执行依赖 hash、相同 K/实际序列化 bytes、EF 守恒和固定判据；没有重算梯度或重新训练。C 相对 B 的 CE 高 0.024874466、accuracy 低 1.249999 pp；没有客户端触发 clipping，故不能将这次失败归因于 clip 截断。最大 FP32 守恒绝对误差 1.862645149230957e-9，处于预设容差内。A/B/C 运行文件与输出至此冻结。

label-bias 的能量检查只否定“直接解释主要残差质量”的窄机制，不排除它对任务预测的作用；root 当前也未选择另开该线。本轮没有形成受支持的新算法，不因研究需要而把负结果改写为贡献。

# TRAIN-gradient oracle replay 独立审查

审查日期：2026-11-03。范围仅为冻结的 `replay_oracle_probe.py` 和 `oracle_replays/cpu_topk20_tau90`；读取必要的来源开发状态、来源锁和训练/验证输入。未读 confirmation 或官方测试数据/结果，未修改任何运行脚本、runner、codec。与设计代理已协调：新 v2 及其额外对照由该代理处理，本审查不重复。

## 结论

**本次 oracle 的计算、数据边界和关键结果通过独立复算。它提供了一个具体的机制诊断：在此开发状态下，参考方向选择确实可能改变保留值的学习效果；它尚未提供可部署算法。**

旧全局方向在这个状态下效果差，当前全 TRAIN oracle 明显改善一步验证结果，而已有本地 fresh delta 替代没有显示 accuracy 优势。不能把 oracle 的 +2.15 pp 写成拟部署方法的结果，也不能据此恢复大规模搜索。

本次实际一次性检查记录为 `oracle_probe_check.json`，耗时约 16 秒：独立重算全部 37500 TRAIN 梯度；重算 12 行支持与诊断；复算 TopK、previous signed、TRAIN oracle signed、fresh-local 四行验证指标。没有执行优化器步骤或完整训练。此为独立子代理复核，不声称完成不同模型家族的正式外部审计。

## 实现核验

| 检查项 | 证据与结论 |
|---|---|
| 梯度位置 | `replay_oracle_probe.py:258-263` 先由共享真实 EF candidates 构造 TopK 聚合 Q，再将模型赋值为 **base+Q**，计算 h = -∇F_train(base+Q)。不是 base 处梯度，也不是 validation 或 test 梯度。 |
| 全 TRAIN 平均 | `109-137` 遍历全部 trainsets；每批 CE 用 `reduction="sum"`，累计批梯度后除以全局样本数 N=37500。每例权重 1/N，等价于按 n_i/N 加权各客户端平均目标；不是客户端等权平均。FP32 反传、FP64 梯度累加。 |
| 独立重算 | 审查使用 `torch.autograd.grad` 按参数 FP64 累加，未调用被审核的 `full_train_gradient` 来生成实际复算梯度。遍历 37500 例后与保存的 `gradient_at_topk` **逐位相同**；平均 TRAIN CE=1.591569663213094，也完全相同。模型参数前后逐位不变。 |
| 不等客户端 fixture | 实际运行 `fixture()`；2/5 样本客户端分割所得梯度与全 7 例平均梯度匹配，并明确检出错误的客户端等权平均；三种 scoring 选出不同的预期支持。 |
| 模型模式 | 模型 eval；冻结 SmallCNN 没有 BatchNorm/Dropout，因此 eval 不改变该网络的目标定义。没有 optimizer 或参数更新。 |
| 输入身份 | `228-243,337-358` 绑定来源状态、4 项实际执行源码、helper 和 40 个 NPZ 加来源 config。独立检查上游输入 hash、42 个消费文件 hash、来源锁/成员检查文件 hash、保存 oracle 文件 hash 全匹配。本次 replay 已补上前一次 probe 的 helper/输入字节身份缺口；这仍是 replay 的当前绑定，不应声称最初训练时已记录全部这些 hash。 |
| 数据边界 | `184-207` 的消费文件中目录名为 `test` 的 20 个 NPZ 是官方 TRAIN 内的 local holdouts；来源 membership 记录所有分片等于官方训练集且 train/holdout 内容无交集。本次未解码官方测试。`238-263` 梯度只取 trainsets；validation 只进入 `275` 的结果评价。 |
| 真实 EF | u_i = saved delta_i + saved residual_i，base/g 与先前审核的共享开发状态一致。没有为不同候选重新生成 delta，也没有清零真实 residual。 |
| shared core/q | `51-106,251,308-318` 对每个客户端先固定其幅值 core 和 q，再只换 filler 评分。K=6843；各客户端 q 为 807–927，core+q=K；不同方法共享的是**每个客户端各自的同一 core/q**，不是强制所有客户端 q 相同。20 客户端 sparse values+indices+头部共 1096160 bytes。 |
| 对照含义 | signed：u_j h_j；absolute：|u_j h_j|；sign_reference：u_j sign(h_j)。最后一种保留方向符号、移除参考幅度，并非只按参数符号发送一位值；所有发送值仍为原 FP32 u_j。 |
| 支持与指标 | 独立重选全部 supports，支持 hash、core 包含、K、原值传输、守恒全部通过。12 行梯度内积、误差能量和交叉项匹配；4 行验证指标逐字段匹配。真实数据标签产生 accuracy/CE，没有模型伪标签或预测分数归一化。 |
| 参数舍入 | `269-282` 同时记录理想聚合差 sent-Q 与实际 FP32 参数相加后的差，及两种 gradient dot；二者有约 1e-9 的内积差，已暴露。h/评分为 FP64、上传值仍 FP32，口径清楚。 |

## 已验证的主要数字

全部为同一开发 seed、同一第 20 轮 TopK-EF 终点、同一第 21 轮 saved deltas/真实 EF、tau=.9；差值均相对同状态 TopK。

| 参考与评分 | 验证 CE 差 | accuracy 差，pp | ∇F_train(base+Q)·(sent−Q) |
|---|---:|---:|---:|
| previous_global，signed | +0.018067 | -0.550002 | +0.013773 |
| previous_global，absolute | +0.001762 | -0.150001 | +0.001732 |
| previous_global，sign reference | +0.000674 | -0.100002 | -0.000394 |
| dense U oracle，signed | -0.008993 | -0.100002 | -0.010377 |
| TRAIN h(base+Q) oracle，signed | **-0.054785** | **+2.149999** | **-0.146466** |
| TRAIN h(base+Q) oracle，absolute | -0.000797 | -0.100002 | -0.001326 |
| TRAIN h(base+Q) oracle，sign reference | -0.033688 | +1.249999 | -0.040568 |
| 各客户端 fresh local delta，signed | -0.007910 | -0.200000 | -0.009365 |

previous g 与 h(base+Q) 的 cosine 为 **-0.05205075**；dense U 与 h 为 -0.00445887，当前 dense local-delta 聚合与 h 为 +0.01475178。这些是此状态的几何量，不是 across-round 一般规律或异质性的因果证明。

对 TRAIN h 使用 signed scoring 明显优于其 absolute scoring，sign-only reference 也获得部分改善。这支持“正确的有符号全局参考在此状态有用”。它不能证明参考幅度在所有状态都必要，也不能把单一状态所得差值作为闭环训练、泛化或跨分区优势。

## oracle 能证明什么

给定共享 core/K、同一候选 u，以及 h=-∇F_train(base+Q)，signed filler 在 core 外选择最大的 u_j h_j。由于各客户端权重为正，理想精确算术下，它在这类固定 core 的 supports 中最大化 h 与聚合 sent 的点积，等价于最小化 **TRAIN 目标的一阶线性预测**。这是一个有意义的支持选择 oracle。

它**不是实际 loss 的最优上界**，也不是对所有 sparse 支持的最优解，因为 core 已固定，实际有限变化受曲率和参数舍入影响。脚本实际计算的是 baseline 的 TRAIN loss/gradient、候选的一阶内积和候选的验证指标；没有为所有候选测得实际全 TRAIN loss。因此不能写“已实测 TRAIN loss 下降 .146466”，更不能将 TRAIN 梯度当 validation 下降真值。

此次实际验证集 CE 和 accuracy 同时改善，说明该 oracle 在这个状态并非仅改善自己定义的几何指标。一次探索性状态的效果仍不足以承担论文的算法性能贡献；不同 scoring 行也不是独立重复。

## 能否用当前可用本地信息替代

**现在没有足够证据说可以。** fresh local delta_i 是客户端在压缩前已有的信息，额外通信可以为零，但目前仅 CE 小幅改善而 accuracy 降 .2 pp，无法支持将其作为替代方案。它是客户端多步本地 SGD 结果，受到本地异质目标和本地路径影响，不等于公共模型位置的全局负梯度。

仅凭一个客户端的本地数据、delta、residual 和历史广播，通常也不能保证恢复任意异质数据下的当前全局梯度：其他客户端数据可以改变全局梯度，而该客户端观察保持相同。这排除了无条件保证，不排除某个额外假设下的有用近似；目前必须由新证据建立，不能从 oracle 的效果直接迁移结论。

h(base+Q) 还存在**时序障碍**：Q 是本轮各客户端 TopK 上传聚合后才得到的，而原方案的支持应在上传前决定。实际获得该位置的全 TRAIN 梯度至少需要额外的数据访问、聚合或通信阶段；把它直接写进客户端选择器会改变协议。

当前全局 dense delta 聚合 D=Σp_i delta_i、dense U 也并非原上传前天然广播的免费信息。若 v2 使用它们并改善结果，只能区分参考新鲜度和梯度位置的作用，不能同时证明可部署性；必须另行明确取得方式和全部字节/轮次成本。

不建议现在提出新的通信 sketch、更多配额或混合参考网格。设计代理正在有限加入 D 和 h(base) 的诊断即可；收完这一轮后，应先判断是否确实存在一个无需隐含 oracle 的具体实现，再决定投入闭环训练。若只有集中式 oracle 有效，应保留机制诊断并停止将它直接包装成轻量化算法。

## 一项轻微输出标记问题

`evaluate()` 在 `295` 只根据 reference 名称给 `reference_requires_centralized_train_information` 赋值。`dense_U_diagnostic` 的 reference 为 None，结果该字段为 false，尽管它使用所有客户端的密集 candidates。该行已有 `diagnostic_only=true`，且摘要已经将 dense 参考称为集中式 oracle，所以不影响已核验的计算，但正文或归档解释应明确 **dense_U_diagnostic 同样不是原通信协议下的免费部署方法**。无需为此重跑或改变冻结 v1。

## 决策边界

保留的主张：在一个真实 EF 开发状态、固定同一 core/q/K 的反事实里，TRAIN h(base+Q) 改变 filler 选择后改善验证 CE/accuracy，旧全局参考未改善，说明参考质量是值得核查的机制因素。

尚不成立的主张：已有可部署算法达到 +2.15 pp；本地 fresh delta 可替代全局 oracle；异质性导致该问题已被因果识别；相同 sparse 行字节代表 oracle 的端到端通信预算相同；存在闭环收敛或跨 seed 优势。

审查的 v1 源码 SHA256：`52a0aa814c9b7e69247f375a9565f7955f1d51173e537906d554272cc077e399`。本轮到此结束，不再新增实验。

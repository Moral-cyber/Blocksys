# 本轮确定的核心贡献与论文主线

日期：2026-11-04。当前轻量化核心主线、完整系统连接与英文稿技术工作已完成，可审阅论文及配套证据；这是当前论文的研究定位，不代表已投稿、首次性保证或接收结果。

**核心贡献：针对标签分布异构的轻量联邦预测，将客户端类别先验作为可修正的本地正则化中心，在共享模型保持低比特通信、个性化状态无需上传的条件下改善客户端概率预测。**

对应英文主张：*A client-specific prior is used as a correctable regularization center, allowing a compact local predictor to adapt compressed federated representations without exchanging personalization parameters.*

## 问题与关键设计

压缩共享更新解决上传成本，客户端类别分布差异仍要求不同的预测概率。单纯增加本地可训练参数不回答这些参数应该向什么目标收缩；直接叠加类别先验也无法修正共享logits已经包含的类别信息。

本设计将两个职责分开：低比特共享训练提供共同表示；冻结共享模型后，客户端只用本地TRAIN拟合小型仿射预测器。关键变量是偏置的正则化中心。令 `mu=center(log(n_class+1))`，优化：

`sum TRAIN CE(A*z+b,y) + (lambda/2)||A-1||² + (lambda/2)||b-mu||²`，`sum b=0`。

当前实例保持已冻结的20维、lambda10、三比特量化和误差反馈；不重新选模型或系数。FedLC-inspired与20维形式是具体实现，不要求每个组件单独构成创新。同先验11维BCTS保留为简化与机制对照。

这一目标的偏置驻点满足 `lambda*(b_c-mu_c)=n_c-sum_x p_c(x)`。因此，预测器在先验附近按观测类别计数与预测概率质量之差调整。有限正则允许修正固定先验偏移；这是现有目标的解释性质，不包装为新的联邦收敛定理。求和损失等价于均值损失加 `lambda/n_i` 正则，小客户端受到较强的相对收缩。

## 两项论文主张与证据

| 主张 | 已有支撑 | 论文处理 |
|---|---|---|
| 本地先验作为正则化中心有预测价值 | 官方TEST相对同维零锚点NLL降低0.0358/0.0182；相对直接加先验降低0.0282/0.0222 | 主结果；核心控制保持维度、共享模型与系数相同 |
| 预测个性化可以与共享上传分离 | 真实量化上传54,067 B；20个FP32适配值80 B留在本地；CIFAR同cap TopK及持久FedRoD-linear已有三seed参考 | 支撑性资源取舍；不将适配器大小当整个设备内存 |

两任务为CIFAR-10/FashionMNIST，每任务原固定三个训练种子。官方TEST准确率75.57%/92.97%，NLL0.6446/0.1880。既有固定TEST用于报告结果，后续新分析不据其调参。

## 本轮实际新增的最小机制验证

按 `design/core_contribution/PLAN.md`，只在既有开发TRAIN/holdout数组上新增一次固定类循环错配锚：`mu_bad=np.roll(mu,1)`。锚的数值集合、范数、维度、lambda和求解器保持相同；120次拟合仅接收TRAIN，holdout随后评分，未读取官方TEST或原图、未增加训练种子。正锚和零锚直接复用既有FP32部署状态。

| 开发holdout，三seed宏NLL | 正确本地锚 | 零锚 | 固定错配锚 |
|---|---:|---:|---:|
| CIFAR-10 | 0.612447 | 0.653559 | 0.683592 |
| FashionMNIST | 0.166384 | 0.180172 | 0.210277 |

正确锚相对错配锚的NLL分别降低0.071145/0.043893，两任务各60/60个客户端—seed组合均降低。这是固定划分的配对描述，独立重复单位仍为三个训练种子。该固定错配对照支持当前设置中类别对齐的本地信息有价值；同维度控制参数容量，但这一个错配方式不排除所有其他非零中心。新诊断属于已看过开发数据后的机制分析，不能冒充新的最终测试。

TRAIN缺失类别的平均预测概率质量，正确锚/零锚/错配锚分别为CIFAR 0.002789/0.030758/0.046389，Fashion 0.000871/0.009122/0.016704。对应holdout真实标签落入TRAIN缺失类别的比例约0.019%/0.037%，因此这一解释局限于当前先验较稳定的标签偏斜设置。新结果未证明抑制过拟合或应对先验漂移。

完整数值与输入身份集中保存在 `results/core_contribution_probe/result.json`；新增运行约2.77秒，无额外CNN前反向。

## 与直接先例的区别

| 一手来源 | 已有构造 | 本文问题中的选择 |
|---|---|---|
| Guo等，2017，vector scaling | 在独立validation上拟合对角logit变换 | 算子沿用；偏置以本地TRAIN先验为正则化中心，状态本地保留 |
| Alexandari等，2020，BCTS与label-shift adaptation | 温度与偏置校准，结合无标签目标数据的EM先验估计 | 使用已知本地TRAIN计数作为可修正的软目标，不估计无标签目标先验 |
| Peng等，2024，FedCal | 保序MLP与scaler参数聚合 | 不上传/聚合校准器，目标为客户端概率预测，允许类别排序改变 |
| Maddock等，2025，Private Federated Multiclass Post-hoc Calibration | 联邦拟合并聚合全局校准参数，含FedVector等 | 保留客户端自己的先验中心与预测状态，不声称全局校准或DP |

上述区别来自已保存一手全文 `literature/prior_bias/sources/{guo,bcts,fedcal,private_cal}.txt`。这是具体差异与组合贡献的依据；检索边界为已有本地存档，不据此声称没有其他近邻。

## 系统连接与写作的完成状态

先前本地闭环 `results/learning_protocol_smoke/result.json` 保留119项检查。本次已进一步接通真实回环 HTTP 与本机 EVM 合约，结果见 `results/chain_integration/result.json`：两个原 CIFAR 客户端、两轮完整本地 epoch、22次梯度、4个新编码更新、4笔登记交易，128项集成检查通过；另有6项真实 EVM eth_call 检查，不增加攻击交易。

系统顺序为：真实delta→新编码并持久化pending→HTTP传输→HMAC/上下文校验→登记交易及两个后继块确认→核对chain/contract/uploader/完整readback→SQLite接纳并聚合→认证纳入收据→客户端验证session/challenge/context/result hash→EF提交。链登记确认与模型incorporation是两个事件，登记收据不能授权EF提交。链成功但local register前丢ACK、finalize后丢ACK均经服务/SQLite重建和精确重试恢复，无重新编码、重复登记交易、聚合或EF提交。

direct/protocol均用double residual和math.fsum，CNN/codec为FP32；模型和残差逐位一致对应此诊断配置，不是原主实验FP32/MPS轨迹复现。链集成应用消息54,408 B/包；先前本地训练夹具54,410 B/包、历史夹具54,405 B/包，随上下文长度不同。结果记录实际HTTP与JSON-RPC body字节、调用时延及开发链gas。单节点Ganache的手动两后继块不构成公链、多节点共识或生产最终性证据。

固定全参与缺席时保留OPEN/pending/旧EF，不能finalize或推进下一轮；该客户端返回后完成同轮。永久掉线自动推进与生产部署为可选扩展。外置tag和record identifier沿用项目的水印上链线，不称为权重嵌入水印，也不证明诚实训练或防投毒。

主模型本机独立推理重放亦已完成，见 `results/replay_saved_inference/result.json`：6个固定checkpoint、120个客户端—种子组合、60,000张官方TEST图像；raw logits逐位一致、argmax差异0、主汇总指标最大误差3.66e-16，无训练/拟合/新增种子。它只核验锁定结果，不用于新选型。

英文稿已围绕本主线完成题目、摘要、引言、方法、结果、讨论与结论，纳入先验驻点解释、开发错配诊断和完整系统闭环，入口 `../03_英文论文/main.pdf`。当前14页、摘要116词，已编译无overfull。保留BCTS与CE等关键对照，不再围绕每个实现部件的必要性扩矩阵。核心贡献→完整系统→优秀英文论证→复现可行性的优先顺序保持；便携复现不是要求。

当前技术工作可审阅，待用户提供作者/单位信息和最终track匿名要求后办理实际投稿；不声称已经投稿或接收。

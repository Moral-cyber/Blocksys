# 已有 SCAFFOLD 子空间控制诊断

2026-11-04，启动前固定方案。目的：定位 head-only 小收益是否遗漏主体梯度漂移；这是已有 SCAFFOLD 控制公式的诊断基线，不是新方法，不引入压缩、校准或新启发式。

三组均 fresh seed20261191、20轮：plain_dense 无校正；body_control 只校正前136224参数；full_control 校正全部136874参数。最后 Linear(64,10) 的650参数定义不变。已有 head_control_dense 只作同协议背景，复用相同seed/分区不构成独立重复。

同 Cifar10IndexA0p1、partition20261143、官方TRAIN内部37500训练/2000分层globaldev，20 clients、全参与固定 p_i=n_i/Σn_i；普通CE、SGD .03、batch64、epoch1、共同初始化与 client batch seed=20261191+100r+i。MPS局部、CPU全局，使用相同 SmallCNN、分区和 metrics helper，不读official TEST。

受控子集梯度为 g+c−ci；本轮 c 固定，K_i 为实际batch数。ci'=ci−c−delta_subset/(ηK_i)，server c'=Σp_i ci'，与 −Σp_i delta_subset/(ηK_i) 核对。必须固定全参与与样本权。只上传一次精确dense delta；用delta重建control归SCALLION已有先例，不主张零额外control上行首创。

所有通信为解析字节：上行64+4×136874B/client；下行同尺寸global，受控组另加16+4d B/client。完整保存每client ci、global c、模型、固定权重、实际SGD/梯度次数、不变量及source/PLAN/helper身份；签名或checkpoint不兼容即拒绝复载。

CPU合成fixture先通过，由主任务决定是否启动。四个背景/诊断组完成后统一看round16–20均值、逐类dev表现和控制/通信开销，定位控制范围差异；不按中途结果改控制范围、强度或训练预算。此诊断不给新方法、泛化或核心贡献结论。

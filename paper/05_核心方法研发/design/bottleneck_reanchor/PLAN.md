# 回到可测问题：异构性是否放大压缩代价

2026-11-04。连续候选未胜强对照后，不继续新增组件。先冻结一个最低成本2×2诊断，确定当前协议值得解决的压缩-异构耦合是什么。

同一官方TRAIN索引分区20261143、同client样本数、同全局训练图像多重集、同样本加权目标、同模型seed20261171、SGD.03、batch64、localepoch1、20全参与、20轮、2000分层开发验证、MPS训练/CPU其余。官方test不读取。

两个变量：

1. 保留Dirichlet标签异构(mixing0)，或把全部TRAIN行确定性混池再按原client数量重新分配(mixing1)。原client样本数和梯度步数保持不变，holdout和global验证不变；mixing1不保证有限样本每client标签严格均匀，只是随机重分配对照。
2. 不压缩dense，或全局5% TopK+EF。使用已冻结run_accelerated/run_partitions入口，不写新优化算法。

观测每运行last5 CE/accuracy和完整曲线。定义压缩CE代价=L_sparse-L_dense、accuracy代价=A_dense-A_sparse；计算异构与重分配下的代价差。单seed交互是描述，不作显著性/因果总体/领域普遍结论。

还检查dense本身的异构差、update残差范数、local样本量/step分布。若主要差发生在dense本身且压缩附加代价很小或负，则不能继续用“压缩特别损害异构”作未经证实的论文动机。后续目标可以改为同精度下通信节省、客户端尾部指标或更合适的已知问题，但必须先给测量与对照，避免随机添加方法。

只做四个20轮baseline，不调参数、不命名新算法。已有全部负结果保留。

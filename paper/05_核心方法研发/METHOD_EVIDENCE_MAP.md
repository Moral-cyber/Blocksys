# 核心贡献与证据入口

本轮论文主线已确定：**以本地 TRAIN 类别先验作为可修正的正则化中心，在低比特共享训练下改善客户端概率预测，个性化状态无需上传。** 完整决策与近邻区别见 `CORE_CONTRIBUTION.md`。

- 固定官方TEST主结果：`results/predictive_final_test/`；同维零中心NLL对照降低0.0358/0.0182，三种子/任务。
- 新开发类错配诊断：`results/core_contribution_probe/result.json`；正确/零/错配中心holdout NLL为CIFAR0.612447/0.653559/0.683592、Fashion0.166384/0.180172/0.210277。独立实现重算通过，属探索性开发机制分析，不是新TEST。
- 通信与状态：54,067 B codec上传，80 B本地预测状态，547,496 B FP32残差仍保留。
- 本地真实TRAIN系统闭环：`results/learning_protocol_smoke/result.json`，2客户端×2轮、4个新编码，119项断言通过；与同double参考一致，非原FP32主轨迹复现。真实链尚未执行。
- 当前英文稿：`../03_英文论文/`，主张与关键对照统一组织，作者身份待补。

同先验BCTS与普通CE对照保留，支持先验中心作为主要机制；不要求20维/FedLC-inspired分别成为必要创新。旧STOP、原始数据与官方TEST身份保持不变。便携复现不是默认要求，外部跨模型审查不可用不阻塞推进。

# 实验执行表

更新：2026-11-03。独立单位为训练seed；不同划分、后端分组报告。

|ID|目的|配置|状态|证据|
|---|---|---|---|---|
|M0-CODEC|编码与理论属性|480属性、10畸形输入；Lagrange960属性|DONE|codec_check / lagrangian_check|
|M0-BASELINE|RegTopK、FedProx实现与恢复|显式目标/选择公式、RNG和恢复签名|DONE|controls_check / BASELINE_CODE_REVIEW|
|M0-DATA|源身份与成员核查|官方train/test图像多重集与hash|DONE|checks/source_and_dataset_lock.json、dataset_membership_current.json|
|M0-MPS|真实本机加速一致性小检查|mu0/.01/.1目标等价、可重复、全局不变|DONE|accelerated_check.json / ACCELERATOR_CODE_REVIEW|
|M1-SKEW|dense同轨迹零残差诊断|mix0，12round，rho.05|DONE|snapshot_runs|
|M1-IID|固定图像/样本量的近IID诊断|mix1，12round，rho.05|DONE|snapshot_iid_runs|
|M2-CPU-DEV|开发与强基线|20round，seed20261113，验证选型|DONE|DEVELOPMENT_RESULTS.md|
|M2-MPS-DEV|后端独立开发|10个既定方法/参数，20round|IN_PROGRESS near completion|mps_v1/development_cifar|
|M3-CPU-CONFIRM|原冻结40round配对|seed20261123/24/25；TopK-EF vs prefix.9|2 pairs DONE; third pair IN_PROGRESS|confirmation_fixed_partition；不读test选型|
|M1-ACTUAL-EF|真实残差共同状态反事实|TopK与prefix20round开发状态，各actual/zero residual|DONE; negative mechanism evidence|checkpoint_probes、REVIEW_CHECKPOINT_PROBE.md|
|M1-ORACLE|区分历史参考与选择机制|一次全train目标梯度、同核心/q控制|IN_PROGRESS|replay_oracle_probe.py；非部署算法|
|M3-MPS-MAIN|原全矩阵主实验|新split、3seed、5方法、40round|DEFERRED at user's stop-loss instruction|不以现有小效应自动扩张|
|M4-FASHION|第二任务主实验|核心参数直接迁移，强基线|DEFERRED until core candidate supported|数据与入口已核查，不冒称已训练|
|COST|本地编码成本|median/IQR、状态内存与额外字段|implementation DONE, full timing DEFERRED|benchmark_codec.py / BENCHMARK_PROTOCOL.md|

外部跨模型experiment-audit：REVIEW_UNAVAILABLE；独立同族代码核查不替代它。若当前选择器仍缺稳定和实用价值，止损并换方向，主线保持轻量化FL下的数据异构。

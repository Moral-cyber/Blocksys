# Stage1决定（Stage2结果前记录）

2026-11-04。四组均20轮DONE，seed20261181，同索引partition20261143，只开发验证。

|loss/codec|last5 dev CE|last5 dev accuracy|
|---|---:|---:|
|CE dense|1.8340582886|33.2200%|
|CE TopK-EF|1.8771309235|31.5800%|
|cal dense|1.8191286957|33.8300%|
|cal TopK-EF|1.8627080505|32.0800%|

校准在两个对应分支均改善CE与accuracy，按原计划，Stage2暂保留cal loss。此为开发数据依赖选型，不冒充确认；效果归属已知校准部件，不是新通信算法贡献。全类CE/count-floor1/tau1均按本实验明确规则。

下一步仅cal_uniform3、cal_mixed_mse、cal_mixed_sensitive，保留相同seed/数据/批次/优化协议，和既有cal_topk配对。使用冻结codec哈希46f3601e5c107f192d359227b21aea2734cc5e798674f2dccbbdf46fc796d863，现有11 fixtures通过。按原Stage2门槛比较三个量化方式，不调块长/权重/tau。无officialtest。当前核心尚未成立。

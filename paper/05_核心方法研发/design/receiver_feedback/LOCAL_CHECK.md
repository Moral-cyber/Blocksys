# Receiver feedback：本地独立检查

2026-11-03。性质：本地静态检查、CPU fixture 与保存结果的独立统计复算。**外部跨模型审计：`REVIEW_UNAVAILABLE`**。未改冻结 `run.py` / `PLAN.md`，未重跑训练，未加载官方 TEST。

**未发现会使本次闭环结果失效的数据泄漏、反馈时序、包映射或归一聚合错误。** 数值 fixture 全部 PASS；三方法各20轮已完整，独立复算决定为 **`STOP_THIS_FIXED_FEEDBACK_RULE`**。

## 核心核对

- 共同源为同一 seed20261143 分区的 TRAIN10轮 warmup，checkpoint签名/round/哈希相符。新 runner 重建分区后对 TRAIN 与 dev 索引哈希双向校验。原始索引独立核对为37,500 TRAIN、12,500 dev、交集0且完整覆盖50,000行；没有调用官方TEST工厂的路径。
- 每个客户端持有独立向量，每轮先从自身起点做本地训练，再按该轮delta包能量选支持，上传绝对模型值。服务器只按实际上传位置的train样本权重归一化；无上传处不补零。共享 anchor 后各方法轨迹分化是预期行为。
- magnitude_gate / receiver_feedback 的接收步骤完全相同：同样一份训练梯度、四个固定候选点、另一个不重叠训练批次选最小CE，平局保留local。两方法都包含self；已有单状态候选的排self已删除。选择批次全部来自TRAIN；dev只用于探索性观测和最终筛选。
- 反馈只在接收者最终选中含transfer的候选时标记非零gate包，且要求至少有另一上传者；不是“所有上传包都算被采用”。本轮全部接收完成后才更新utility，下轮上传才能读取新utility，未见同轮反馈前视。
- utility是**聚合包的接收可用性代理**：排除sender自身作为receiver的bit，再对其他19个receiver取均值。全局有上传支持的包做`.75 old+.25 observed`；观测到但无人采用的包衰减，未被上传观测的包保持原值。它不估计某sender的因果贡献，也不承诺贡献归因。代码与计划一致。

## 有实际意义的fixture

`local_check_fixtures.py` / `LOCAL_FIXTURE_EVIDENCE.json`：独立decoder验证全长/短包、乱序chosen规范化、空支持、负数/零值、绝对模型值往返；1070bit反馈的首位/字节边界/末位及2个padding bit验证；接收规则一致/平局行为、utility排自身与缺失观测处理验证。全部PASS。

主仿真路径直接对张量聚合，wire用于生成字节和哈希，没有在运行路径中执行decoder。fixture确认当前编码与独立解码等价，但这不构成端到端网络、故障恢复、认证或链上系统测试。

## 预算与表述边界

- 128参数包，模型136,874参数、1070包，末包42参数。固定公共上传上限是`16+53×(4+128×4)=27,364B/client/round`。
- magnitude53全长包=27,364B；feedback52全长包+16B反馈头+134B bitset=26,998B。若包含短包，则分别为27,020B和26,654B。二者均不超过公共上限，**实际字节不相等**。这是给定上限下完整机制的比较；包数差1是反馈开销设计的一部分，不能声称唯一差别只有utility读取。
- PFL轮实际构造model/feedback bytes并按len计数；服务器下发并集包、按20路unicast计费。总通信须含上传、下载及共同warmup，不能只报模型上传来称整体省通信。
- warmup的219,004,800B是按10轮、20client、双向unicast、16B头、完整FP32模型的**公式追记**，不是旧warmup当时记录的实际序列化或网络测量。它是各方法共同项；local_sgd仅warmup后零通信。
- local_sgd也使用一次梯度和四次候选forward，但其候选是SGD步长回溯，通信方法的候选含transfer与组合。只能称梯度/候选次数匹配；不能称完全相同候选、总计算量或wall-time。每轮seconds包括训练/筛选/评估等全部工作，不是方法内部的分项计时。
- 此次单seed使用整个dev参与探索性选择，不能外推多seed显著性或最终TEST性能。后继FedALA式融合、反馈删除/打乱对照仍是待执行实验，不能填作已有结果。

## 冻结统计规则

独立复算器：`verify_results.py`，证据：`LOCAL_STATS_CHECK.json`。它从每轮20个client行复算宏平均和样本加权统计、检查训练batch位置、解码采用mask、重新核算上传/下发/累计字节，并从所有轮bitset独立重建最终utility。

primary：round16–20的client宏平均CE与accuracy分别取5轮均值。GO定义为`CE_feedback < CE_magnitude AND accuracy_feedback >= accuracy_magnitude + .005`；任一不满足即停止此固定反馈规则。这明确采用“CE更低且accuracy至少高0.5pp”的门槛。五轮和20个客户端均不视为独立训练重复。

结果状态：三方法各20轮均DONE，primary固定使用round16–20；没有挑选中间最好轮。


## 完整结果独立复算

已从保存的每轮20客户端行复算全部60轮的4个汇总指标，均与原结果绝对差<1e−12。逐轮batch位置、选择最小训练CE且平局优先local、采用mask与外部贡献支持、上传/并集下发/累计字节均通过；从全部采用bitset重建的最终utility与三个checkpoint匹配（atol1e−15）。这些校验未重新训练或重新评价模型；不声称独立重现了MPS数值轨迹。

| 方法 | last5 client宏CE | last5 client宏accuracy | last5样本加权CE | last5样本加权accuracy |
|---|---:|---:|---:|---:|
| `local_sgd` | 0.5625088733 | 79.681190% | 0.5728457169 | 79.228800% |
| `magnitude_gate` | 0.5793509869 | 78.986920% | 0.5949930428 | 78.318400% |
| `receiver_feedback` | 0.5802046087 | 78.870861% | 0.5953761585 | 78.060800% |

反馈相对幅值基线的primary差值为：宏CE **+0.0008536218**，宏accuracy **−0.116059pp**。CE严格更低=false；accuracy至少高0.5pp=false；独立决定 **`STOP_THIS_FIXED_FEEDBACK_RULE`**。按每客户端last5均值配对，CE更好/更差/持平为10/9/1，仅为描述统计。local_sgd在此单seed的last5两项宏指标也优于两种通信方法。

| 方法 | warmup后上传B | warmup后下发B（unicast） | warmup后总B | 含共同warmup公式项总B |
|---|---:|---:|---:|---:|
| `local_sgd` | 0 | 0 | 0 | 219,004,800 |
| `magnitude_gate` | 10,852,720 | 69,281,120 | 80,133,840 | 299,138,640 |
| `receiver_feedback` | 10,717,672 | 79,405,040 | 90,122,712 | 309,127,512 |

反馈组比幅值组少上传135,048B，但多下发10,123,920B；warmup后总通信多9,988,872B（约12.47%）。因此不能据上传略少宣称整体通信节省；含共同warmup后总量差约3.34%。这里是仿真中序列化应用层长度与明确warmup公式项，不是实际网络或链上测量。

本次仅停止这条冻结反馈规则，不据此否定所有接收反馈组合。未执行后继seed复核、FedALA式融合或反馈删除/打乱对照；外部跨模型审计仍为 **`REVIEW_UNAVAILABLE`**。

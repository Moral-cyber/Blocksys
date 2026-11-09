# Shadow 两半步训练：有限本地检查

2026-11-04。本地独立静态检查与合成数据CPU fixture；外部跨模型审计 **`REVIEW_UNAVAILABLE`**。未修改冻结源文件，未训练真实数据，未加载官方TEST。当前未发现使三组比较失效的影子点、梯度投影、残差时序或全局模型修改错误。

- **影子点符合公式。** 每batch在实际local上先做`.015`普通SGD，再以`u=theta_half-anchor+old_residual`选择5%支持，影子模型参数为`anchor+M*u`。anchor/残差/支持均不求导。
- **第二半步回写实际theta。** 在独立shadow副本求CE梯度，将同一M内的梯度赋给local.grad，再由local的`.015`SGD更新。没有把影子参数覆盖到实际theta，支持外坐标第二半步不变。当前SmallCNN无BN/dropout或buffer，不存在副本状态未同步的附带更新。
- **旧r轮内只读。** CPU上的`residual.to(cpu)`可能共享storage，但代码没有对rem写入；local/shadow均deepcopy，global未改。轮末基于实际theta唯一一次真实上传：`TopK(theta-anchor+old_r)`，r更新一次；无额外repair。shadow在后端用topk，终编码在CPU用stable argsort，计划明确不要求并列值跨后端一致。
- **对照计算次数符合计划。** A每batch`.03`一次梯度；B与C同一TRAIN batch两次梯度、`.015`两步，第二次均接第一半步之后。B第二次梯度在actual half，C在shadow point，后者只投影支持内梯度；“累计名义步长”和次数匹配不等于优化轨迹/梯度更新量或wall-time相同。C有支持选择及额外模型副本开销。
- **数据口径正确。** fresh seed20261161，同partition20261143；TRAIN37500，holdout池12500，实际global dev为固定分层2000例，不能称client宏平均/12500例评估。本次从头初始化，没有warmup。
- **通信仅解析预算。** 三组最终上传均为5%坐标K6843、FP32值+uint32索引、64B头：每client54,808B，每轮1,096,160B，20轮21,923,200B。没有真实wire/network/下载测量，不称总网络字节。轮seconds包括评估等整个迭代，不是纯训练内核计时。

关键证据：`local_fixture_check.py` / `LOCAL_FIXTURE_EVIDENCE.json`，全部PASS。用带非零旧残差的8参数线性分类器，独立NumPy解析CE梯度核对单batch三组：参数误差均0；C与“覆盖为shadow”和“不投影第二梯度”的错误替代公式不同。再用68条合成数据的2个真实loader batch核对C，误差1.04e−17，global和旧r逐位不变；最终上传EF精确守恒。不把CPU toy通过称作完整MPS训练数值重现。

只读统计复算器：`verify_results.py`；证据：`LOCAL_STATS_CHECK.json`。固定primary为round16–20的dev accuracy与CE均值；GO要求`CE(C)<min(CE(A),CE(B))`且`acc(C)>=max(acc(A),acc(B))+.01`。未通过停止此配方，不从中间最好轮替代last5。

完整结果：三组均DONE20，固定使用round16–20；独立判定为STOP_THIS_SHADOW_RECIPE。


## 完整结果独立复算

核对全部60轮的样本数2000、client批次数与每batch梯度/optimizer步数、解析上传预算及最终checkpoint残差norm，均通过；各组config签名与冻结source/plan相符。从原始保存validation记录独立求last5均值，未重新预测或重训，不能把保存值复算称作重现完整MPS轨迹。

| 方法 | last5 global dev CE | last5 global dev accuracy | 20轮梯度次数 | 20轮迭代时间（含评估） |
|---|---:|---:|---:|---:|
| A standard | 1.8763364685 | 33.190001% | 11,940 | 62.648s |
| B samebatch | 1.8466855621 | 34.850000% | 23,880 | 106.267s |
| C shadow | 1.9392419617 | 30.920000% | 23,880 | 113.809s |

C−A：CE **+0.0629054932**、accuracy **−2.270001pp**；C−B：CE **+0.0925563995**、accuracy **−3.929999pp**。C的CE未同时更低，accuracy也未同时高于A/B各1pp；冻结两个条件均false，独立决定 **`STOP_THIS_SHADOW_RECIPE`**。不得挑中间轮或更换支持/半步配置来声称原计划通过。

B−A：CE −0.0296509064、accuracy +1.659999pp。该额外同batch两步SGD对照在此次筛选中优于A，而C仍显著落后于它；因此不能把两倍梯度次数本身的潜在收益归给shadow。这里“落后”为保存均值的描述，不是多seed统计显著性结论。B/C梯度次数相同、名义累计步长相同，C仍有支持选择开销；表中时间是整个迭代含评估，不能称纯训练内核速度或严格匹配耗时。

三组20轮解析上传均21,923,200B，相同K不能冒称真实wire量。结果只对应当前SmallCNN、分区20261143、seed20261161与20轮开发筛选；不否定全部CrAM类训练/压缩感知方法，不声称原创原子机制或完整复现既有论文。未执行多seed、更长训练或官方TEST验证；外部跨模型审计仍 **`REVIEW_UNAVAILABLE`**。

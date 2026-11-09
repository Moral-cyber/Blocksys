# Stage2量化：本地独立核对

2026-11-04。外部跨模型审计 **`REVIEW_UNAVAILABLE`**。仅检查冻结codec、run量化分支、独立CPU小fixture与固定last5；没有修改源码或另开训练。codec SHA256=`46f3601e5c107f192d359227b21aea2734cc5e798674f2dccbbdf46fc796d863`。

**当前未发现使本次Stage2结果失效的解码、预算或EF单位错误。** encode把实际wire交给decode，返回的sent来自该解码；run取encoded.sent聚合，并额外检查decode(wire)==q。服务器不是聚合预量化浮点候选。EF在FP32模型delta单位更新为u−decoded；浮点加回可能有1ULP误差，不宣称逐位恒等或收缩性。

- **标量量化规则一致。** 2/3/4bit分别使用3/7/15个对称level，最高code保留不用；round-to-nearest、饱和裁剪，零scale只允许零code。候选先经过FP32 scale回写，再按实际FP32解码MSE评价。maxabs初值加四次LS/refit中选MSE最低者；refit可缩小scale并裁剪离群端点，不保证端点精确保存或找到全局最优scale。
- **真实长度计费。** 64B头；每256坐标块1B位宽+4B scale，再加ceil(count×bits/8)个值字节。136874参数共535块、末块170值；每块末padding也入账。没有索引/权重旁路或隐含scale元数据。uniform3=54067B/client；mixed实际长度≤54808B，共同上限不等于完全相等实际字节。这里只计应用payload，不含网络、认证或链上开销。
- **greedy marginal正确。** 从2bit逐级升级，gain=(old decoded MSE−new decoded MSE)×block_length/P×block_weight，除实际新增packed字节；只选正gain且不超预算的升级，0B升级也须正gain。长度因子使短块按坐标数计入全向量目标；固定5B块元数据不随位宽变化。局部贪心不保证全局最优或跨预算单调最优。
- **importance只影响bits。** 每个候选scale/code只读u和bitwidth；sensitive仅将本轮训练批梯度平方proxy在块内取mean，按共同最大值归一用于升级权重。全零importance回退unweighted。它不改训练loss、u值、标量量化器、推理logits或总预算，也不发送权重。proxy不是per-example Fisher/Hessian；无需额外backward。
- **删除对照保留。** uniform3固定3bit；mixed_mse与mixed_sensitive使用相同候选量化器和预算/greedy规则，仅block_weight为1或梯度proxy。校准作用已由Stage1单列并归属FedLC-inspired部件；不能把校准或已有量化思想的收益计作本候选原子首创。

独立证据：`independent_quant_fixtures.py` / `QUANTIZATION_FIXTURE_EVIDENCE.json`，全部PASS。采用整型bitstream解码，不调用codec解包helper；7值端点例独立LS候选得到MSE .11372856，优于maxabs初值.21385713；三不同块的单次32B升级按独立gain/byte选对，importance仅改变位宽，固定bit下scale与解码值相同。全模型三模式独立逐字节核算/解码通过，样例EF最大误差7.45e−9，低于保守FP32容差4.77e−7。未复制作者fixture作为独立证据。

历史训练wire未逐条保存，因此本检查对历史轮次核对保存的全部block元数据/实际len与运行路径解码断言；不虚称重新解析了每个历史payload。`verify_stage2.py` / `QUANTIZATION_STATS_CHECK.json`会检查client计数/shift/梯度次数、每块metadata+padding、升级账、MSE长度加权、EF误差和最终checkpoint，并固定取round16–20。

Stage2门槛：sensitive CE同时低于uniform3和mixed_mse；accuracy比uniform3至少高0.5pp，且不低于TopK超过0.5pp。三组20轮均已完成，以下固定使用round16–20，不用中间最好轮替代last5。


## 完整Stage2独立复算

三个量化条件各20轮全部完成；共1200个量化client-round的block元数据、padding、升级账、MSE长度加权和EF误差检查通过，梯度次数、TRAIN计数/shift、同dev2000与最终checkpoint亦通过。每组11940次梯度，无敏感性额外backward。历史记录中量化EF最大绝对加回误差4.66e−10，低于保守FP32容差；这只验证数值账，不推出收敛保证。last5从保存的validation记录直接复算，并非重算全部预测或复现MPS轨迹。

| 方法 | last5 global dev CE | last5 global dev accuracy | 每client上传B范围 | 20轮上传B |
|---|---:|---:|---:|---:|
| cal_topk | 1.8627080505 | 32.080001% | 54,808–54,808 | 21,923,200 |
| cal_uniform3 | 1.8180330200 | 33.960001% | 54,067–54,067 | 21,626,800 |
| cal_mixed_mse | 1.8197476868 | 33.909999% | 54,782–54,792 | 21,916,760 |
| cal_mixed_sensitive | 1.8198931610 | 33.790001% | 54,782–54,792 | 21,916,760 |

TopK列为解析布局字节，量化列为实际生成并解码的应用payload长度。两个mixed本次累计长度相同；uniform3更少，不声称三个方法实际字节相同，也不称整体网络流量。

sensitive−uniform3：CE **+0.0018601410**、accuracy **−0.170000pp**；sensitive−mixed_mse：CE **+0.0001454742**、accuracy **−0.119998pp**。sensitive−TopK为CE −0.0428148895、accuracy +1.710000pp，但更简单的uniform3已取得更好结果。冻结条件：CE同时胜uniform3/MSE=false；accuracy比uniform3高0.5pp=false；不低于TopK超过0.5pp=true。独立判定 **`STOP_THIS_SENSITIVITY_ALLOCATION_RULE`**。

这组删除对照没有支持梯度proxy作为核心贡献。不能把全部量化相对稀疏TopK的收益、Stage1已知标签校准收益或已有混精度思想归给本敏感性规则；也不把均值细差表述为多seed统计显著性。

三种量化累计codec_seconds分别17.778/47.869/48.091s（uniform3/MSE/sensitive），包含编码、解码与运行重解码断言，不是纯标量量化内核时间；总迭代时间约81.197/114.364/114.149s，含本地训练与dev评估。当前种子仅作有限筛选，不调块长、权重幂或tau挽救结果；无官方TEST和跨模型审计。**`REVIEW_UNAVAILABLE`**。

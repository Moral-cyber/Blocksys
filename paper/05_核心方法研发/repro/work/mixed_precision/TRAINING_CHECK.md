# Stage1训练：本地有限核对

2026-11-04。仅核对冻结run.py的训练与Stage1 dense/TopK路径，以及保存结果的last5；不审尚由另一agent实现的codec，不启动真实数据训练。外部跨模型审计 **`REVIEW_UNAVAILABLE`**。当前未发现客户端计数、校准梯度、推理或EF路径的致命错误。

- **精确校准规则。** counts取`ds.labels[ds.indices]`，即客户端实际TRAIN原始行，而非所有官方TRAIN行、dev或当前batch。完整10类shift=`max(count,1)^(-1/4)`，零类与count1均为1；cal仅在训练把该常数逐类从logits减去，再对原标签做完整全类别CE。没有删样本、改标签、屏蔽不在本地的类或改变归一分母。
- **实现归属。** tau=1与count0截断是本实验明示规则；不是可核实的FedLC官方复现，也不是原子新发明。cal关闭为普通CE；没有在此试tau网格。训练校准的收益只能记作已知FedLC类组件作用，不能算新通信方案贡献。
- **推理rawlogits。** 全局评估走原`r.metrics(model,dev)`，不减shift。CPU Indexed toy对比raw/shifted CE验证结果匹配raw，差异不是口头约定。
- **梯度proxy的准确语义。** 一次正常CE.backward后、optimizer.step前，读取每个参数实际batch均值梯度，平方累加并除本地batch数。没有额外backward或额外样本。末小batch与全batch等权；它是训练路径批梯度的二阶矩proxy，不是per-example Fisher、Hessian或样本加权梯度二阶矩。Stage1也计算该proxy但不用于上传决策。
- **dense与TopK分支正确区分。** dense发送原delta，残差始终清零；TopK对`delta+old_r`选6843个坐标并发原值，r更新为差值。各组全模型更新与压缩，没有分类头豁免。Stage1两种字节均为解析公式，不称wire；dense每client547,560B、TopK54,808B，不能写等总上传字节。
- **数据/预算与次数。** fresh seed20261181、同partition20261143，TRAIN37500、holdout池12500、global dev2000；非client宏平均，未载官方TEST。每local epoch实际batch数ceil(n/64)，每batch一步/一次backward；完整20轮每方法11940梯度。无warmup、新数据或server校准数据。

CPU合成fixture：`local_training_fixtures.py` / `TRAINING_FIXTURE_EVIDENCE.json`，全部PASS。独立NumPy全10类CE梯度核对带原始索引/零类/非均匀count的toy：cal参数误差7.44e−9；CE/cal均计数1次CE forward+1次backward。64+4两个batch的proxy误差5.18e−9，且与按样本加权不同；global模型不变、最终EF守恒。未把CPU toy通过称作完整MPS轨迹重现。

只读复算器：`verify_stage1.py`；保存证据：`TRAINING_STATS_CHECK.json`。逐client核对真实运行记录的train_label_counts与分区manifest、shift逐FP32值、梯度次数、payload和最终residual；固定last5为round16–20的rawlogit global dev CE/accuracy。

完整Stage1统计：四组均DONE20；以下固定使用round16–20。


## 完整Stage1独立复算

80轮全部已保存client计数/shift、实际梯度次数、解析payload和最终残差核对通过；四组每组20轮11940次梯度。cal shift逐FP32值匹配客户端TRAIN标签计数；dense最终残差逐坐标为0。四组共用相同TRAIN/holdout/dev索引；当前config签名、源和计划哈希一致。

| 方法 | last5 rawlogit global dev CE | last5 global dev accuracy | 20轮梯度次数 | 20轮解析上传B |
|---|---:|---:|---:|---:|
| ce_dense | 1.8340582886 | 33.220000% | 11,940 | 219,024,000 |
| ce_topk | 1.8771309235 | 31.580000% | 11,940 | 21,923,200 |
| cal_dense | 1.8191286957 | 33.830000% | 11,940 | 219,024,000 |
| cal_topk | 1.8627080505 | 32.080001% | 11,940 | 21,923,200 |

cal相对CE：dense的CE降低0.0149295929、accuracy提高0.610000pp；TopK的CE降低0.0144228729、accuracy提高0.500001pp。两个对应分支均满足本阶段“CE与accuracy同时改善”的描述筛选条件；本次继续采用cal属于基于该dev结果的数据依赖选择，后续tracker和独立验证须保留此事实。这里没有新seed、置信区间或最终test，不称可靠泛化收益。

这点校准收益归属已有FedLC-inspired标签校准组件及本实验明确的count0/tau规则，**不是mixed precision通信方法的新贡献或官方FedLC复现结果**。下一codec与敏感性分配尚不在本报告的数值审查范围；不能由Stage1通过推出它们已有效。

以上是从保存validation记录独立求last5均值，并非重新运行MPS训练或从全部预测重算CE。官方TEST未加载；外部跨模型审计仍为 **`REVIEW_UNAVAILABLE`**。

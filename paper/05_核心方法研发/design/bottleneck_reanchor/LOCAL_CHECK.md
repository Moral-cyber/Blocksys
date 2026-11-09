# Baseline reanchor：本地有限核对

2026-11-04。仅核对冻结launcher、当前配置/索引和保存统计；没有启动训练、修改优化器或设计新算法。外部跨模型审计 **`REVIEW_UNAVAILABLE`**。

launcher调用原run_accelerated/run_partitions，固定partition20261143、seed20261171、MPS、20轮、全参与、SGD.03/localepoch1/batch64；仅改变mixing0/1与dense/topk_ef。所有已生成config的冻结源哈希及run_signature匹配，test_final=false，当前result没有official_test_final字段。

索引核对：已生成条件均TRAIN37500、holdout池12500、固定global dev2000，train/holdout互斥且覆盖全部50000原始行。各客户端一epoch真实DataLoader(drop_last=False)的批次数枚举为[30,46,4,93,39,8,2,5,27,51,9,62,66,5,44,22,16,3,40,25]。runner每batch一次optimizer.step，因此同client count意味着相同步数；当前runner没有单独记录实际optimizer计数，不把该推导冒称运行时计数日志。

跨mixing索引检查 **PASS**：两边训练原始行multiset及每client count相同，holdout/validation完整索引逐项相同，训练分配确实改变。mixing1实际索引与从mixing0独立执行确定性全量混池所得结果相符；因此同数据规模/步数、同全局经验目标，仅训练客户端归属改变。mixing1不保证每客户端有限样本标签严格均匀。

| 条件 | 状态 | 完成轮数 | last5 dev CE | last5 dev accuracy |
|---|---|---:|---:|---:|
| mixing0/dense | DONE | 20 | 1.7844544312 | 35.440000% |
| mixing0/topk_ef | DONE | 20 | 1.8223577545 | 34.400001% |
| mixing1/dense | DONE | 20 | 1.4913654312 | 46.520000% |
| mixing1/topk_ef | DONE | 20 | 1.5386679123 | 44.650000% |

四组均DONE20；last5严格取round16–20，从保存validation行求均值，未重算模型预测或重现MPS轨迹。2000是global dev，不能称client宏平均。

- mixing0压缩CE代价(sparse−dense)=+0.0379033234；accuracy代价(dense−sparse)=+1.039999pp。
- mixing1压缩CE代价(sparse−dense)=+0.0473024811；accuracy代价(dense−sparse)=+1.870000pp。

压缩代价交互(mixing0−mixing1)：CE=-0.0093991577；accuracy代价=-0.830001pp。

Dense本身mixing0−mixing1：CE=+0.2930889999，accuracy=-11.080000pp。以上仅当前模型/分区/seed/20轮的描述差，不作显著性、总体因果或普遍领域结论。负压缩代价不叫受损。

预算按已有codec解析公式核算，仅上传应用payload，不称真实wire或总网络通信。检查器和证据：verify_results.py / LOCAL_STATS_CHECK.json。此前所有STOP候选保持STOP；本检查不会复活旧候选。

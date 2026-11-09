# 标签先验训练校准＋全坐标低比特反馈：组合式开发

2026-11-04，执行前冻结。按用户允许“有理由地组合已有机制”的标准，
本候选不再要求额外自创压缩评分器，考察本地统计偏斜校准与低字节全坐标传输的实际组合价值。
既有损失、量化与EF准确归属；不能仅换算法名称声称原子创新，也不能凭会议等级保证贡献或录用。

已知balanced softmax/FedRoD采用按local类别数调整logits。
当前明确变体：local CE(z_c+log(n_ic+1),y)，每类加入1个Dirichlet伪计数；
分母n_i+C是softmax公共常数而省略。推理用raw logits。+1为公开固定平滑规则，
不是官方精确复现、不用dev估计先验、无tau/温度网格、无额外backward。
这与已停止的有效先验后处理不同：后者无训练、且用pK假设推断预测先验，已被状态诊断否定。

四组fresh seed20261201、独立分区seed20261193、alpha.1、20clients、TRAIN37500、
TRAIN内部2000分层globaldev、20轮、localepoch1、batch64、SGD.03、固定全参与样本权：

1. ce_uniform3：普通CE＋全模型uniform3 EF。
2. fedlc_uniform3：公开FedLC-inspired CE(z−max(n,1)^(-1/4))tau1＋同uniform3。
3. balanced_uniform3：上述Dirichlet1平滑balanced CE＋同uniform3。
4. balanced_dense：上述balanced CE＋精确dense传输，隔离压缩代价。

统一使用既有SmallCNN136874参数和冻结mixed_precision/codec.py，量化真实编码+解码后聚合；
每client实际54067B≤54808B cap，EF=delta+旧residual−decoded。dense解析547560B，
全模型下行统一dense解析547560B/client，不隐藏下行成本。无真实网络/认证/链测量。
不读取official TEST。数据划分是开发性独立分区，不是独立新数据集或最终确认。

四组必须各完成20轮：round16–20均值balanced_uniform3 CE低于CE及FedLC，accuracy分别至少+.5pp；
与balanced_dense相比accuracy不低超过1pp，才进入有限多seed/第二任务/强近邻与组合归属确认。
失败则保留结果，不调平滑/温度/bit/本地epoch救此固定配方。
通过也仅证明该已知组件组合值得确认，不能直接写成已成立新核心或优于所有算法。

后继最低证据：两新训练seed、另一个数据任务、较温和异构、固定总上传字节对比TopK，
与原始FedLC/FedRoD相关global目标强对照，校准删除与量化删除、通信曲线、真实codec成本；
整套组合直接prior检索与外部可用审查。正式test等最终方法冻结后才访问。

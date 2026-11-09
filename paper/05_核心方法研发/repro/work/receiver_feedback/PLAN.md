# 接收反馈用于后继选包：组合式候选与有限闭环

2026-11-03，探索性研发。组合已有机制允许，算法部件首创不是本项目的必要门槛。

## 与已停止候选的关系

上一单状态“排self+硬门+回溯”未胜同额外计算SGD，依原规则停止。这里不再保留排self，也不以硬门本身作为新核心。新的变量是：本轮各接收客户端实际采用的包，是否能作为下一轮上传选包的有效反馈。两通信方法使用完全相同的接收/本地优化步骤，仅选择器是否读取反馈不同。

## 候选机制

- 每个客户端保存自己的完整模型。每轮1本地epoch后，按相对此轮本地起点的delta包能量选上传包，上传绝对参数值。服务器对每个包按有上传者的train样本数归一平均，没有上传的位置不补零。
- 幅值基线每轮上传53包；反馈候选52包并发送1070bit接收mask（16B反馈头+134B bitset）。两者使用同一53全长包的应用层字节上限，不做填充；逐轮记录实际序列化字节，不能宣称实际字节完全相等。
- 接收端仅用本地TRAIN两批，第一批64例计算梯度g，第二批64例选方案。d为包级 -g·(server-local)>0 的接收方向。两通信方法均在{local, local-.03g, local+.5d, local-.03g+.5d}四个点中按第二TRAIN批CE选最小者；平局优先local。一次backward、4次forward。此固定候选集是算法的一部分，不用dev挑选。
- 被采用的transfer方向中，正收益且有外部贡献的包为1，其余0。server每轮将receiver的bitset反馈汇总，给sender i的utility用其他receiver均值；观测到的位置做 EMA=.75 old+.25 observed，未观测保持。初值.5；上传分数=本地delta包能量×(.1+.9 utility)。不把这个分数叫某个sender的因果贡献；它是聚合包的接收可用性代理。
- 这组成“本地模型保留→稀疏共享→接收筛选→下轮选包”的反馈环。mask有额外隐私暴露可能，本研究不声称形式化隐私保护；不会发送原始样本。

## 固定有限运行

共同源：partial_fusion/dev_seed20261143/warmup.pt，新的索引分区seed20261143，alpha=.1。该warmup仅在相同分区TRAIN训练，不复用其它分区旧checkpoint。所有方法计入共同10轮dense warmup的通信，另外报告warmup后通信。全20client参与；PFL20轮，training seed20261144；每轮SGD .03 batch64 localepoch1；MPS训练/CPU评估。

三组：local_sgd（不通信、同一梯度+4点SGD回溯）、magnitude_gate（53包）、receiver_feedback（52包+bitset）。local_sgd仅在warmup后不通信，不得写成从头零通信。每轮完整12,500 client holdout用于开发评估，均非最终test。官方test不载入。每client每轮同源batch顺序；长期各自模型会分化。

primary比较为feedback-vs-magnitude的最后5轮client宏平均验证CE及accuracy。单seed只作筛选。若feedback CE没有低于magnitude且accuracy至少高0.5pp，停止这条固定反馈规则，不为此调EMA/pack/阈值网格。若通过，先做独立代码/统计检查，再以新训练seed做两配对复核；后续必须加入同一候选的FedALA式融合及反馈删除/打乱对照，才能支持核心组合贡献。

此门槛在结果前记录。FedCSPACK/FedALA/FedCAC/FedLP是相关来源与对照，不声称任何单组件首次提出。所有未完成实验在tracker中单列，不能填成论文结果。

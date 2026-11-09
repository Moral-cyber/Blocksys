# 固定全局balanced-quant组合停止

2026-11-04。四组20轮完成，独立保存产物589/589检查通过；方案数字一致性WARN。
实际TRAIN37502、holdout12498（逐client round(.75n)），不是PLAN名义37500/12500。
四组共用相同数据成员和实际样本权，未改数据或事前去留条件。

|方法|last5 CE|accuracy|
|---|---:|---:|
|CE uniform3|1.7809861389|35.410000%|
|FedLC-inspired uniform3|1.7741681885|35.840001%|
|Dirichlet1 balanced uniform3|1.7814179413|35.720001%|
|Dirichlet1 balanced dense|1.7789119690|35.790000%|

balanced对CE仅+.310001pp且CE更差；对FedLC−.120000pp且CE更差。
相对dense压缩代价小，不替代相对学习强对照的失败。
决定STOP_THIS_FIXED_BALANCED_QUANT_RECIPE，不改平滑、轮数或阈值。

每量化client每轮实际54067B，20轮上传21626800B；含解析dense下行总240650800B。
各方法11920次梯度，误差反馈与首包独立解码检查通过；未重放全部历史wire。
不是完整FedRoD或BALMS复现，不认为+1平滑或低比特EF原子创新。
本结果仅为新开发分区的一次训练seed，未使用official TEST指标。

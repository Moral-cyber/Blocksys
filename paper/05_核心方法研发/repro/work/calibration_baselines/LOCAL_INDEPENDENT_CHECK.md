独立NumPy核查PASS：16083项检查；2组固定数组、7模型×20client，560个校准态、1260组client×variant指标及63组汇总全部复算。所有准确率精确一致，NLL/宏均值、holdout样本加权、client q10误差≤3.33e-16。

映射核对：zero-anchor为z+b(10维)；TS为az、a≥.001(1维)；BCTS为az+b(11维)；VS为A⊙z+b(20维，A允许负值)。目标是ΣTRAIN CE＋10/2||θ−anchor||²，不是mean CE；bias中心化。四类解析grad/Hessian与独立公式、有限差分一致，ridge特征值≥10（舍入容差）。3个预先固定TRAIN案例×4类独立重拟合一致；不使用holdout GT拟合。

560态的FP32舍入与40/4/44/80B部署态完全核对；评分实际使用FP64仿射/NLL。轨迹单调，319次按KKT阈值结束，241次按FP64分辨率结束，实际513/560态满足1e−8，最大投影梯度3.98e-06。保存态未触边界，另2个synthetic边界案例验证正a及KKT。

新分区FedLC uniform3：VS宏准确率76.3145%、CE 0.66331；prior-anchor为75.7124%、0.64370。VS准确率高0.6020pp、CE反而高0.01961；q10也由59.1864%到62.5262%。这是取舍，不能宣称prior-anchor全胜。

289个源/输入/结果文件哈希前后不变；未读取原始数据/TEST、未加载CNN、无联网。官方复现差异：本地TRAIN拟合、ridge10、固定约束/中心化；没有FedCal的MLP、保序或聚合。同λ不同维度及样本量不代表等复杂度；holdout仅评分，无网格选型。LOCAL_INDEPENDENT_CHECK；外部REVIEW_UNAVAILABLE。

# 同TRAIN先验锚的11维BCTS：机制control addendum

2026-11-04，真实数组运行前固定。此为机制control，不是新candidate，不改主PLAN、主方法、原STOP或新阶段gates。主evaluator7b45、主solver9c31保持冻结。

f(z)=a z+b，a为单个正尺度；a0=1，b0=center(log(local TRAIN count+1))，与20维prior-vector相同TRAIN先验锚。目标ΣTRAIN CE(f,y)+10/2(a−1)²+10/2||b−b0||²。a≥0.001，bias中心化；从anchor出发，复用冻结base.Problem(kind=bcts)解析design/grad/Hessian，固定ridge10/max50 Newton、Armijo c1=1e−4、40次折半、原bound/KKT/梯度回退及1e−8终止。无网格/温度/λ/初始化选择。

分别维持FP32部署态11维44B与20维80B；指标统一使用保存FP32状态升FP64应用。拟合只接收保存TRAIN logits/GT；holdout只最终评分。比较同一fresh来源的20维prior-vector、zero-vector、原10维、prior/raw/head/head+prior，完整输出accuracy/NLL/Brier/ECE15和成本；不重拟合20维或替换候选。20维缓存状态必须绑定相同原评估、协议和client原行号。

问题是全局单尺度+b是否足以解释收益。相同ridge系数但scale维数1对10、总罚项和自由度不同，不能把任何差异唯一归因class-specific scales；若11维持平或更好，同样完整报告，不能宣称per-class A必要。

先CPU toy有限差分grad/Hessian、凸性/下降、正尺度边界/KKT、anchor删除与冻结BCTS逐位等价、FP32状态/FP64应用一致性；只交付fixture。随后由主任务在fresh saved arrays运行一次新11维fit，0CNN/0数据集字节/0uplink。此addendum没有通过/失败gate，研究判定由独立主PLAN管理。

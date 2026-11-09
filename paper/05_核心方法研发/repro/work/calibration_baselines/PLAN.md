# 旧冻结logits上的四个固定校准强对照

2026-11-04，运行前固定。只读取prior_bias/diagnostic_v1已保存的四模型、20client TRAIN/holdout logits、GT和original IDs；没有新CNN forward/backward、FL训练或数据加载。旧candidate与STOP结论保持不变，不用旧结果替代新confirm结论。

四个校准器都最小化Σlocal TRAIN CE(fθ(z),y)+10/2||θ−θ0||²，固定ridge10；同系数但维度不同是公开设置，不表示各模型复杂度已等价。①zero_anchor_bias：f=z+b，b0=0，10维；②temperature_scaling：f=az，a0=1，a≥0.001，T=1/a；③BCTS：f=az+b，anchor=(1,0)，a≥0.001，11维；④vector_scaling：f=A⊙z+b，anchor=(1,0)，20维，A允许实数。bias每步中心化，消除CE公共平移gauge；ridge使完整Hessian正定。

纯NumPy FP64解析梯度/Hessian，最多50步投影阻尼Newton，Armijo c1=1e−4，初始步长1，最多40次折半；边界Newton失去下降时固定投影梯度回退，初步长为1/max(10,||H||₂)。KKT投影梯度∞范数≤1e−8或FP64目标分辨率终止。无温度/λ/初始化/迭代网格，无holdout选型。部署FP32状态分别40/4/44/80B；FP64轨迹和缓存另计。

原prior-anchor bias保留冻结lambda10 solver及其已保存state，作为不改动的比较项；同时完整报告raw、prior、matched旧learned_head及head+prior。四旧模型全部输出client宏CE/accuracy、加权指标和client10%分位，不按holdout分数换candidate或选系数。

目的：隔离prior anchor是否必要，以及更灵活校准器是否构成强解释。若BCTS/vector更好，不能宣称prior-anchor性能占优。它们属于已知校准方法；本地独立拟合、不聚合，不冒充官方FedCal精确复现或新原子方法。

先synthetic核对有限差分grad/Hessian、凸ridge、gauge、已知零点/边界/下降，再运行旧数组一次。source/PLAN及所有输入前后摘要相同；saved GT只允许TRAIN进入fit，holdout仅最后评分。记录Newton/线搜索/耗时/缓存与部署态，zero新增CNN计算、uplink，非端到端通信实测。新confirm仍由主任务独立运行和判断。

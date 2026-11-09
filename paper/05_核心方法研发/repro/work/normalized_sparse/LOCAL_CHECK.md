# Global FL 归一化压缩：有限本地检查

2026-11-04。本地独立静态检查与CPU关键fixture；外部跨模型审计 **`REVIEW_UNAVAILABLE`**。未修改冻结源文件、重跑训练或加载官方TEST。当前未发现使本次2×2比较失效的FedNova公式、tau、RMS或EF单位错误；不据此声称复现了完整FedNova论文实验。

- **FedNova特例正确。** 当前无momentum、weight decay、梯度裁剪，固定LR .03、本地1epoch、全参与。`delta`是local−global，Nova组先除实际步数tau；server用`tau_eff=Σp tau`乘`Σp sent`，符号一致。无压缩时恢复所写的无momentum归一更新；相同tau时退化为FedAvg。它是已知FedNova机制，收益不能计作本组合的新贡献。
- **tau来自本实现实际optimizer步数。** MPS adapter的DataLoader batch64、drop_last默认False，逐batch一次step。无训练的实际DataLoader枚举得到20个tau=`[30,46,4,93,39,8,2,5,27,51,9,62,66,5,44,22,16,3,40,25]`，均等于ceil(n/64)，保留末小批。tau_eff=50.3577066667。这是当前数据加载设置的步数，不称FedNova“官方ceil”，也不机械套论文理想记号floor。
- **EF单位一致。** Nova residual与candidate都在delta/tau单位，server乘数只在聚合后施加；非Nova在delta单位。本次p/tau/tau_eff固定，三轮独立telescoping fixture误差8.88e−16。压缩器逐坐标验证sent+residual=u，不把已放大server步长的残差回存到client。
- **RMS只改选择。** 分母是每个参数tensor的`u=normalized_delta+residual`的RMS；全零tensor保持零分数，按全向量稳定排序选相同K=6843坐标，发原始u值。手工fixture确认支持改变、正的逐tensor缩放下支持不变、零tensor无NaN、原值发送与EF守恒。它不是128参数包，也不是固定每层预算，不能声称L2误差最优或层间公平保证。
- **数据与初始化符合计划。** fresh seed20261151，无warmup。分区20261143：TRAIN37500、holdout池12500，实际评估为其中固定分层2000行全局开发子集；训练/开发无交集。这里accuracy是全局dev准确率，不能写成client宏平均或沿用上一PFL的12500行评估口径。四组同初始与本地batch种子，后续模型轨迹各自分化。
- **通信是相同解析上传预算。** 每client64+8×6843=54,808B，每轮1,096,160B，20轮21,923,200B；没有真实encoder/network测量，也没统计dense下发与总网络通信。RMS多做尺度计算，Nova多归一操作，不能称总计算完全相等。
- **保存范围偏差已记录。** 每轮当前checkpoint包含完整model/residual可恢复，并保存全轮原始validation/客户端诊断；checkpoint文件逐轮覆盖，未保留每轮历史权重/残差快照。计划“按轮保存全部模型/残差”若理解为历史快照则未满足，不能据现有产物复核任意历史权重。

关键证据：`local_fixture_check.py` / `LOCAL_FIXTURE_EVIDENCE.json`，全部PASS。只读结果复算器：`verify_results.py` / `LOCAL_STATS_CHECK.json`。结果完成后固定算round16–20的devCE与accuracy均值；GO=CE(D)<min(CE(B),CE(C)) 且acc(D)>=max(acc(B),acc(C))+.005。仅作单seed/当前模型和分区筛选，不从未完轮次改RMS、幂、预算或加新variant。

完整结果：四组均DONE20，固定使用round16–20。


## 完整结果与冻结判定

从保存的原始validation行直接求round16–20均值；未重新训练或重算模型预测，不声称重现MPS数值轨迹。四组当前config/源/计划签名、20轮样本数2000、每client选中坐标count、解析字节和最终checkpoint残差记录均一致。每tensor计数始终合计K=6843；FP32 norm日志只作粗略数值合理性检查，精确EF守恒来自独立运行的生产压缩器fixture。

| 方法 | last5 dev CE | last5 dev accuracy |
|---|---:|---:|
| A TopK+EF | 1.8803710388 | 32.630000% |
| B FedNova+TopK+EF | 1.9122744385 | 31.070000% |
| C RMS-score+EF | 1.8981472382 | 32.450000% |
| D FedNova+RMS-score+EF | 1.9140961395 | 31.460000% |

D−B：CE +0.0018217010、accuracy +0.390000pp；D−C：CE +0.0159489014、accuracy −0.990000pp。D的CE未低于两个组件，accuracy也未达到两个组件中的较高者再加0.5pp；冻结CE与accuracy条件均为false，独立决定 **`STOP_THIS_NORMALIZATION_COMBINATION`**。不追加此组合训练，也不在此状态改RMS幂、预算或新增variant救结果。

2×2描述性差值：B−A（已知FedNova组件）CE +0.0319033997、accuracy −1.560000pp；C−A（RMS组件）CE +0.0177761993、accuracy −0.180000pp。交互D−B−C+A为CE −0.0159544983、accuracy +0.570000pp；交互符号不代表D优于B/C，不能据它虚称新增组合贡献。本次A的两个primary指标均最好；本次也没有“FedNova收益可归到组合”的正向证据。

仅观察last5的100个client-round选择记录，原TopK两组八个tensor均没有整tensor零上传；RMS两组的四个bias tensor出现零上传。C的conv1/conv2/hidden/output bias零选择次数为66/40/55/94，D为69/46/60/93；C/D的output bias平均选中比例仅0.6%/0.7%，A为54.3%。这表明逐tensor RMS并不保证每个tensor被选中，不能称“消除层级饥饿”；它是当前模型、分区、预算和20轮下的诊断，不是该现象造成CE变化的因果证明。

四组20轮解析上传量均为21,923,200B；不含未记录的下发或网络开销。单seed、固定dev子集仅用于筛选；未进行新seed、其它分区/任务或最终TEST复核，结果不能外推否定全部FedNova/尺度归一化方法。外部跨模型审计仍为 **`REVIEW_UNAVAILABLE`**。

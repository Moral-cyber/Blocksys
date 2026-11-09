# TRAIN拟合十维bias：固定诊断结果

2026-11-04。正式结论：**STOP**。预先固定的balanced_uniform3候选没有达到accuracy至少+0.5pp门槛；保留全部结果，不更换候选，不扩λ/温度/迭代或模型网格。此前解析prior固定配方的停止结论不变。

## 结果与门槛

真实使用4个冻结final模型、20client各自TRAIN拟合和既有clientholdout数组评估。主要指标是client宏平均accuracy/CE；holdout实际12498，包含旧globaldev，不是新独立测试集。

| 模型 | 原prior macro accuracy | 原prior macro CE | 拟合bias macro accuracy | 拟合bias macro CE | accuracy变化 |
|---|---:|---:|---:|---:|---:|
| ce_uniform3 | 76.9832% | 0.603131 | 77.9411% | 0.587390 | +0.9579pp |
| fedlc_uniform3 | 77.3418% | 0.598649 | 78.0894% | 0.587159 | +0.7476pp |
| balanced_uniform3 | 77.4974% | 0.594850 | 77.8124% | 0.592504 | +0.3150pp |
| balanced_dense | 77.5983% | 0.593399 | 77.8422% | 0.591207 | +0.2439pp |

固定候选balanced_uniform3：

- 较同模型prior：accuracy从77.4974%到77.8124%，**+0.3150pp**，低于+0.5pp；CE从0.594850降至0.592504。
- 较既有最佳accuracy prior（balanced_dense）：**+0.2142pp**，也低于+0.5pp；CE低于其0.593399。
- 较最好learned-head accuracy（ce_uniform3 learned_head，77.2660%）：候选满足不低于超过0.5pp条件。
- 新bias在ce/fedlc模型的分数同样完整呈现，它们不能替换预先指定候选。当前证据表明既有bias校准可降低TRAIN拟合后的部分偏差，但固定balanced组合没有达到本次复核门槛。

所有raw、prior_only、learned_head、learned_head+prior、样本加权指标和client10%分位保存在result.json中。

## 实施与计算量

每模型准确读取其37502个local TRAIN样本一次，每client按冻结original IDs顺序用既有IndexedCifar10归一化；保存train_logits/labels/IDs。模型eval、requires_grad=False、torch.inference_mode；没有CNN反传、新head训练或FL训练。

固定中心化b0=log((n+1)/(N+10))。NumPy FP64最小化ΣTRAIN CE(z+b,y)+10/2||b−b0||²，使用解析grad/Hessian，最多20步Newton与固定Armijo；只接收TRAIN数组。holdout在bias确定后评分，没有据holdout调整任何参数。部署态为FP32十维bias，**40B/client**；FP64轨迹另作审计证据，不属于部署态。

- 四模型总额外图像forward：**150008**，640次batch forward调用。
- 总接受Newton step：**280**（各client最多20）；每次线搜索目标评估次数及停机原因逐client保存。
- CNN backward：0；额外uplink：0；每模型20client总bias state=800B。
- CPU1实际CNN前向时间16.235s，NumPy拟合0.090s，总记录计算时间16.909s；文件加载校验另有开销。不是端到端通信或零计算声明。

## 先行合成验证

NumPy synthetic fixtures先通过再运行真实forward：解析grad有限差分最大误差2.3626443e-09，Hessian误差2.81364265e-10。Hessian最小特征值≥10；数据CE Hessian的gauge方向、共同logit常数/共同bias-prior平移检查通过。imbalanced/single-class/one-example/uniform-zero-logits四例Armijo接受步均不增目标，中心化部署态40B，FP32解对logit常数平移不变。

可复算：`python -B work/prior_bias/local_fixture_check.py`。真实诊断入口为`probe.py --out <fresh output>`，完整一次forward已执行，无需重跑本次冻结证据。

## 来源与边界

新loader只反序列化五个固定`data_batch_1..5`，分别核验冻结manifest SHA256；原TRAIN标签哈希、original row顺序及归一化保持相同。audit hook对`test_batch`任何字节读取直接报错；本次尝试为0。既有流程的torchvision完整性校验可能MD5读取test_batch字节，这不同于加载test记录进行推理或使用test指标选择模型；本次没有发现后两者证据，也不把历史MD5检查误判为label泄漏。

所有80份holdout NPZ的IDs/labels/TRAIN counts/prior与冻结manifest核对；所有输入文件、模型config/checkpoint、源与PLAN前后哈希相同。4个模型本轮前向前后state完全相同。

- 结果：`work/prior_bias/diagnostic_v1/result.json`，SHA256`fad176a08e6979c7ff7731ef23e5b3017423f69a5d95e4f7d59d75dc5c4fc0e3`。
- 80个每模型/client NPZ：`work/prior_bias/diagnostic_v1/`，包括TRAIN logits/GT/IDs、holdout既有数组与拟合bias。
- 固定输入摘要：`work/prior_bias/diagnostic_v1/inputs_before.json`（也完整嵌入result的before_after_equal字段）。
- synthetic证据：`work/prior_bias/LOCAL_FIXTURES.json`。
- `work/prior_bias/PLAN.md`：`120472270ba4fdff1d5fc211d6b7cb1d9ee352daee013de4321d396aae8b3310`。
- `work/prior_bias/solver.py`：`6b8dda5e86438f78f3918f8601ec3ce9e7099417d3b42ac55bfbeb4014132b96`。
- `work/prior_bias/probe.py`：`527cd00d7c36ac2faa3e3af83a4d15bef09deeb6b361e9743e0737491a0e1113`。

本次是已知bias calibration思想的有限开发诊断，不作新机制、隐私、网络或独立泛化结论。decision=STOP已固定；后续统一复核/归档由主任务处理。

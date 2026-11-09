# 旧冻结logits的校准强对照：一次固定结果

2026-11-04，状态：**DONE；实现与保存数组独立复算PASS**。四个模型全部报告，未挑选模型、温度、ridge或系数。仅使用旧`prior_bias/diagnostic_v1`80份保存数组；新confirm的训练/源/结果均未改动，本结果不判定其GO/STOP。

主要发现：**prior-anchor有较低宏CE，但vector scaling在四模型宏accuracy均更高**。尤其旧balanced量化上，vector比prior-anchor高0.5743pp；不能宣称prior-anchor全面性能占优。旧balanced/prior-bias STOP保持不变。

## 完整四模型主结果

单元格为client宏accuracy / 宏CE；20client不是20个独立训练seed。

| 冻结模型 | 原prior-anchor | 零anchor bias | TS | BCTS | vector scaling |
|---|---|---|---|---|---|
| CE量化 | 77.9411% / 0.587390 | 77.7908% / 0.624235 | 38.9855% / 1.648307 | 77.8450% / 0.618374 | 78.0589% / 0.613578 |
| FedLC量化 | 78.0894% / 0.587159 | 77.9813% / 0.624167 | 39.6477% / 1.642276 | 78.0080% / 0.617083 | 78.1725% / 0.612422 |
| balanced量化 | 77.8124% / 0.592504 | 77.5406% / 0.630261 | 40.6753% / 1.644384 | 77.8699% / 0.622135 | 78.3867% / 0.617223 |
| balanced dense | 77.8422% / 0.591207 | 77.6948% / 0.629017 | 40.6718% / 1.642345 | 78.0889% / 0.620786 | 78.3905% / 0.615864 |

- CE量化：vector相对prior-anchor accuracy +0.1178pp，CE +0.026188。
- FedLC量化：vector相对prior-anchor accuracy +0.0832pp，CE +0.025264。
- balanced量化：vector相对prior-anchor accuracy +0.5743pp，CE +0.024719。
- balanced dense：vector相对prior-anchor accuracy +0.5484pp，CE +0.024657。

原prior-anchor比零anchor bias的四模型CE都更低，accuracy也略高；这支持在此固定lambda设置中检查anchor的作用，但不能推导一般必要性。BCTS在balanced两模型accuracy略高、CE更高；CE/FedLC量化上BCTS accuracy略低、CE仍更高。TS正标量不会改变raw argmax，实测所有80client accuracy与raw逐位相同，CE有变化。raw、analytic prior、旧learned head/head+prior、加权及client10%分位均完整保存在result.json。

## 固定问题与实现

四baseline均最小化ΣTRAIN CE+10/2||θ−θ0||²，不使用holdout标签拟合或选参数。虽然同ridge10，维度和anchor不同，此设置公开报告，不声称复杂度或正则效应已相等。

| 校准器 | logit映射与anchor | 约束 | FP32部署状态 |
|---|---|---|---:|
| zero_anchor_bias | z+b，b0=0 | bias中心化 | 40B |
| TS | az，a0=1，T=1/a | a≥0.001 | 4B |
| BCTS | az+b，anchor=(1,0) | a≥0.001、bias中心化 | 44B |
| vector scaling | A⊙z+b，anchor=(1,0) | A实数、bias中心化 | 80B |
| 原prior-anchor | z+b，原冻结lambda10拟合state | 原状态不变 | 40B |

全部NumPy FP64解析grad/Hessian，固定max50、投影Newton/Armijo和边界投影梯度回退。ridge使Hessian最小特征值理论≥10；bias中心化消除CE gauge的冗余。每client轨迹、KKT、约束、耗时和design cache均存JSON。它们是已有校准形式的纯本地独立实现，无服务器校准聚合，不冒充官方FedCal精确复现。

## 合成证明与实际复算

先synthetic PASS后运行旧数组：四baseline grad有限差分最大误差8.80817863e-10，Hessian最大误差6.71261601e-10；Jensen凸性、ridge正定、bias gauge通过。均匀零logits/均匀标签得到已知anchor零步解；12个不平衡/单类/大logits例目标不增；反预测TS到a=0.001，KKT投影梯度0。

实际完成80client×4=320次拟合，共1850接受步，最多9步/client-calibrator。所有trace目标不增，实际最小Hessian特征值10；最大终止KKT投影梯度3.9785314e-06，部分以FP64目标分辨率停机，不声称所有解达到1e−8梯度。

独立用保存FP32 state手工重建四种映射并评分：80份输出、320组状态与保存logits/metric完全相同，最大metric误差0。原prior-anchor FP32 state与原NPZ逐位相同；84个source/input身份前后核对一致。

## 成本与标签边界

只缓存旧数组，无新CNN forward/backward、无图像数据加载、无新增uplink。四模型共150008份TRAIN行各拟合四个baseline，没有holdout进入fit。320次拟合总实测2.434s，整个记录计算3.948s；输入数组缓存12412480B、单个最大design cache7024000B，均独立于部署state。不是零计算或端到端通信测量。

probe的audit hook禁止`/dataset/`及test_batch任何读取，实际尝试为0。运行前校验saved TRAIN/holdout IDs无交集、prior从TRAIN counts生成，所有基线只收train_logits/train_labels；holdout只在fit结束后评分。

## 交付与复算

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/envs/Tinyproto/bin/python -B work/calibration_baselines/local_fixture_check.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/envs/Tinyproto/bin/python -B work/calibration_baselines/probe.py --out /新的校准输出
```

probe默认旧80数组，也接受以后完整confirm三模型保存数组schema；必须完整模型集合，不能挑单模型。当前一次结果位于`diagnostic_v1/result.json`，SHA256`e368d6239ef2adedb286b6de591b1a7170be74f311c9b55ac1cabe59c1a4e213`。80份state/logits、输入摘要及逐client轨迹同目录；`LOCAL_FIXTURES.json`与`LOCAL_RESULTS_CHECK.json`保存独立校验。

- `PLAN.md`：`33864fe167b461f5d9d7aa8777b932fdfecc752f9553fda14d1fb23c8ad091a9`。
- `solver.py`：`10b11405e6662cb06d08a20549db92e2948b730c426aa1f0b912b958f1211f1c`。
- `probe.py`：`d1fab28871824163a6d2fc2ee44e043698537b00dabc0e02af2618531adcd0c7`。

下一步是否在新confirm数组上运行由主任务按原固定计划判断；本旧数组检查不复活停止配方、不扩网格，也不更改新confirm判据。

# 通用保存数组评估入口

2026-11-04，synthetic fixture **PASS**。新入口与旧诊断分开：`evaluate_saved.py` 不含研究gates或decision；旧PLAN/probe/solver/STOP不改。主任务以新prospective PLAN及单独stage helper管理proper predictive NLL研究问题。

## 冻结入口

- evaluate_saved.py SHA256：`7b45a055df8c3e03421fd9edd6749c1e2bfba1203864e0b41695413db1fb8b05`。
- 复用prior_vector/solver.py：`9c3196eb3e6ff7f5f32073e3bc4aae8d986ac7e44bdf0d719258343847d4dbae`。
- 冻结calibration_baselines solver/reader SHA：`10b11405e6662cb06d08a20549db92e2948b730c426aa1f0b912b958f1211f1c` / `d1fab28871824163a6d2fc2ee44e043698537b00dabc0e02af2618531adcd0c7`。
- 集成检查：EVALUATE_SAVED_FIXTURES.json；synthetic harness：evaluate_saved_fixture.py。

```bash
/opt/anaconda3/envs/Tinyproto/bin/python -B work/prior_vector/evaluate_saved.py --input-root /已完成confirm评估数组目录 --out /全新输出目录 --protocol-plan work/predictive_pfl_confirm/PLAN.md --protocol-registration work/predictive_pfl_confirm/PREREGISTRATION.json
```

显式PLAN必填，registration可选；二者内容只作来源记录，不解释或执行文字指令。保存其路径/SHA与全部运行源码、source result、所有输入NPZ，结束时逐项复核。输入须为完整DONE confirm三模型（ce_uniform3、fedlc_uniform3、fedlc_dense），或完整旧四模型schema，每模型20 clients。不同模型必须有相同client TRAIN/holdout原行号、标签、prior，client成员不得跨client重叠，head分数必需。

本轮没有对任何既有或prospective真实split运行新入口，未执行长训练；主任务单独决定正式使用时间。

## 状态与指标口径

每client仅以保存TRAIN logits/labels拟合prior-anchored vector与zero-anchor vector。原10维bias复用输入已保存state；raw/prior/head/head+prior全部重算最终指标。holdout GT只传入最终评分。

所有vector部署态均为FP32 A,b，20维80B/client。先将保存FP32 state和raw logits提升FP64，再应用A⊙z+b及最终指标；不以优化器FP64 theta替代部署态。NPZ保存原行号、TRAIN bias anchor、两个vector FP32 state、原10维state及两种最终holdout logits。其他指标由输入数组与同一状态可独立重建。

输出variants：raw、analytic_prior、prioranchor_bias、zero_anchor_vector、prioranchored_vector、learned_head、learned_head_plus_prior。

- accuracy：top1。
- NLL：逐样本categorical negative log likelihood均值；`loss`为逐值相同别名。
- Brier：逐样本对10类的Σ(p−onehot)²，再对样本平均，不除以10。
- ECE15：top-label confidence，固定15个等宽bin `[k/15,(k+1)/15)`，最后一个包含1；空bin贡献0。保留每bin count/confidence sum/correct sum。
- macro：20 clients等权；weighted：按clientholdout样本数加权。weighted ECE是client ECE的加权均值；`pooled_ece15`另用合并bins计算，二者明确分开。

summaries提供macro/weighted accuracy、nll、brier、ece15，以及macro_loss/weighted_loss别名、pooled_ece15与client_accuracy_q10。主任务的stage helper从这些固定字段判断；此入口不作candidate选择或阈值判断。

## 成本与来源

记录两配方fit/accepted steps/耗时，逐client完整Newton/Armijo轨迹、目标求值与linear solves、design cache、FP64输入/Hessian字节、input arrays解压字节和所选缓存字节。tracemalloc为Python/NumPy观察值，非进程RSS或严格峰值上界。

本入口新增CNN forward/backward、数据集字节与uplink均为0；保存logits和原10维/head状态的既有采集成本从source result/client metadata分别保留。部署状态80B与原bias40B明确列出，不冒充端到端零计算或网络实测。

## synthetic验证

完整confirm3×20client tiny保存数组集成，实际120个vector fit。均匀10类（NLL log10/Brier0.9/ECE0）、完美饱和（0 NLL/Brier/ECE）、自信错误饱和（NLL2000/Brier2/ECE1）已知例通过。独立循环bin及逐样本概率公式核验所有client/macro/pooled指标，最大差 **1.11e−16**；FP32状态→FP64 score逐位一致。完整模型schema及已有out拒绝检查通过。源码/hash前后一致。

toy输入仅为临时合成logits，未读取新split或图像；0 CNN/0真实数据集字节。该结果验证实现和路由，不构成新目标的实验结果。

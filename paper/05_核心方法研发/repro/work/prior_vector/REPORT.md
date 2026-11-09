# 固定20维先验锚消融：STOP

2026-11-04。预先固定的候选为 **fedlc_uniform3**；旧数组只运行一次。新20维prior-anchored vector 的宏accuracy **78.175095%**、NLL **0.577094316**。它降低NLL，但相对10维prior-anchor bias仅增加 **0.085743pp**，未达到预定+0.5pp，因此此固定accuracy配方判定 **STOP**。六个子gate中五个通过；不改门槛，不换候选，不扩展ridge/family，不用新目标重写此结论。旧10维六gates及原STOP保持。

## 固定机制与评分

相同20维vector_scaling，f=A⊙z+b，A锚为1；唯一机制变化是b锚从0改为中心化local TRAIN log(count+1)。两段ridge均10，目标为sum TRAIN CE，不是mean CE。A允许任意实数；bias中心化；FP64固定Newton/Armijo、最多50步。该组合使用已知vector scaling与先验正则，不声明原子首创。

只读取旧 `prior_bias/diagnostic_v1` 的四模型×20client保存TRAIN/holdout arrays。anchor与fit仅接收TRAIN GT；holdout仅最终评分。旧holdout包含旧globaldev，不能当作新split或独立test。

部署态A,b以 **FP32共20维80B/client** 保存。所有vector最终评分统一先把保存的FP32 A,b提升为FP64，把保存raw logits也提升为FP64，再计算A⊙z+b与softmax/NLL；未用FP64优化器参数冒充部署态。10维、prior和head对照同样以保存状态/分数升FP64评分。优化器FP64参数只作为优化轨迹证据。

## 完整四模型

下表每格为 **宏accuracy百分数 / 宏NLL**。candidate固定FedLC uniform3；其余是预定背景对照。

| 模型 | analytic prior | 原10维prior-anchor | zero-anchor vector | 新20维prior-anchor vector |
|---|---:|---:|---:|---:|
| CE uniform3 | 76.983238 / 0.603130888 | 77.941113 / 0.587389552 | 78.058949 / 0.613577570 | 78.118406 / 0.578561927 |
| **FedLC uniform3** | 77.341794 / 0.598649229 | 78.089352 / 0.587158544 | 78.172517 / 0.612422153 | **78.175095 / 0.577094316** |
| balanced uniform3 | 77.497398 / 0.594850200 | 77.812440 / 0.592503798 | 78.386747 / 0.617223133 | 78.212545 / 0.581068027 |
| balanced dense | 77.598262 / 0.593399203 | 77.842165 / 0.591207163 | 78.390535 / 0.615864350 | 78.239885 / 0.579662780 |

| 模型 | raw | 原learned head | 原head+prior |
|---|---:|---:|---:|
| CE uniform3 | 38.985477 / 1.669165355 | 77.266003 / 0.648595469 | 75.974488 / 0.708809240 |
| FedLC uniform3 | 39.647659 / 1.665569615 | 77.238083 / 0.653190748 | 76.027732 / 0.706876166 |
| balanced uniform3 | 40.675262 / 1.672701622 | 77.100415 / 0.698150525 | 76.238280 / 0.700725845 |
| balanced dense | 40.671762 / 1.670898019 | 77.213722 / 0.696404607 | 76.337969 / 0.699314023 |

所有client指标、样本加权指标、client accuracy10%分位及prior-only基线完整保存在result.json。NLL下降跨四模型出现，但不构成此预定accuracy gate通过。

## 固定候选gates

差值方向为新20维减对照；accuracy单位pp。

| 固定比较 | 实测 | 预定条件 | 结果 |
|---|---:|---:|---|
| vs10维accuracy | +0.085743pp | ≥+0.5pp | **失败** |
| vs10维NLL | −0.010064228 | ≤+0.005 | 通过 |
| vszero-vector accuracy | +0.002578pp | ≥−0.2pp | 通过 |
| vszero-vector NLL | −0.035327837 | ≤−0.01 | 通过 |
| vsprior accuracy | +0.833301pp | ≥+0.5pp | 通过 |
| vsprior NLL | −0.021554913 | <0 | 通过 |

此STOP只针对预先固定的accuracy命题。若另立proper predictive NLL研究问题，须另行前瞻固定新PLAN和未使用split，其证据及判断与本诊断分开。

## 检查与成本

先完成synthetic fixture：grad/Hessian有限差分误差最大1.1804e−9/3.1746e−10；正定、凸性/gauge、单类/单样本下降、均衡零点、负A均通过。删除anchor后，四种toy中新函数与冻结vector solver的FP32 state、FP64参数和完整优化轨迹逐位相同。均衡log-count中心化仅出现小于1e−15的浮点舍入，按数值零点核验。

正式旧数组：80个新prior-vector fits，358个accepted Newton steps，886次目标求值；80个配对zero-vector fits，565 steps。新fit最大5步，最大最终梯度∞范数1.027e−6，部分因FP64目标分辨率终止，完整stop reason/trace留存。每配方TRAIN rows150008；计时新fit1.387秒、配对zero1.965秒，总probe3.804秒，属于本机该次计时。

输入数组未压缩字节12412480B，选定数组缓存12406080B；最大单client FP64 design缓存7024000B（design_flat为共享view）；Hessian20×20 FP64为3200B。tracemalloc观察峰值31818398B，覆盖输入、求解及保存过程中的可追踪Python/NumPy分配；它不是进程RSS或严格峰值上界。部署80B较原10维40B增加40B/client；FP64求解缓存和轨迹另计。

新CNN forward/backward、数据集字节、uplink均为0；这不是零计算或网络实测。guard对dataset/datasets、test_batch和t10k路径均拒绝，真实尝试为0。输入/来源89项SHA前后一致。

独立只读复核80个输出NPZ、160个vector状态、**999840个holdout score元素逐位一致**，全部client和summary指标最大误差0；直接目标公式误差0，梯度∞范数公式差9.24e−14。FP64优化态与部署FP32态应用后的最大score差4.6923e−7；另行全FP32算术敏感性最大score差7.0950e−7、client NLL差8.4115e−9、accuracy差0。正式表格仍统一使用部署FP32状态升FP64应用。

## 冻结与入口

`diagnostic_v1/result.json` SHA256：`3ac6d580eda6462fa23b24855112863536af7cc3c4fb76ad745416ee615dd7b2`。

| 文件 | SHA256 |
|---|---|
| PLAN.md | `2d606bb226c08b73d24270250bf7fe676b56c515afba9e420a1b2f6e8dea5cd0` |
| solver.py | `9c3196eb3e6ff7f5f32073e3bc4aae8d986ac7e44bdf0d719258343847d4dbae` |
| probe.py | `eb3831a0bcfaf602ca691b3b198663ab29d5ecd39d4f36f68a5e01b820dc7011` |
| 冻结calibration_baselines/solver.py | `10b11405e6662cb06d08a20549db92e2948b730c426aa1f0b912b958f1211f1c` |

fixture证据在LOCAL_FIXTURES.json；独立复核在INDEPENDENT_CHECK.json。仅可用新的输出目录重放，原diagnostic_v1不覆盖：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python -B work/prior_vector/local_fixture_check.py
/opt/anaconda3/envs/Tinyproto/bin/python -B work/prior_vector/probe.py --out /新的输出目录
/opt/anaconda3/envs/Tinyproto/bin/python -B work/prior_vector/independent_check.py
```

本报告不发起重放、长训练或publish。后续通用evaluator与新研究目标单独管理，不修改本PLAN/solver/probe/STOP。

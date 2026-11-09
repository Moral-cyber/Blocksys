# 已有控制基线：CPU synthetic fixtures

2026-11-04。结论：**PASS**。`PLAN.md`全文995字符（含空白），满足不超过1000字约束。已创建独立`run.py`，实现plain_dense/body_control/full_control；正式20轮训练由主任务另行启动。

验证命令：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python -B work/control_reference/local_fixture_check.py
```

本次仅CPU单线程、合成TensorDataset。包含67个3×32×32样本的2-batch作用范围检查，3/5个样本的两个客户端×两轮实际run()，以及中断/恢复验证，合计26个微型SGD step。没有调用data.prepare；audit hook禁止任何`/dataset/`文件打开，实际尝试为0。

## 模型、协议与配对

确认SmallCNN共136874参数；最后Linear(64,10)的650参数为head，其前136224为body。三个控制维数分别0/136224/136874。新实现与冻结head_control引用同一SmallCNN、分区prepare和metrics函数。相同seed、相同合成batch流下，新plain本地delta与冻结head runner的零控制delta逐位相等。

三方法实际run()前两轮的client batch seed依次为20261191、20261192、20261291、20261292；每个client的batch顺序、全局anchor均配对相同；零初始control下首轮原始本地delta逐位相同。每次client训练前后全局anchor逐位不变。

## 直接梯度校正与ci恒等式

参数backward hook记录原始梯度，SGD.step前探针记录实际梯度。探针只记录，不改数学运算。body控制只增加前136224坐标，full增加全136874坐标；受控子集之外的直接梯度改变量为0。

| 方法 | 实际batch数 | 校正误差最大值 | 子集外直接改变量 | ci′与原始梯度均值误差 |
|---|---:|---:|---:|---:|
| plain_dense | 2 | 0 | 0.0 | 0 |
| body_control | 2 | 6.51925802e-09 | 0.0 | 2.40746886e-07 |
| full_control | 2 | 2.23517418e-08 | 0.0 | 2.2444874e-07 |

另用非零旧ci、固定权重[0.375,0.625]、不同实际K=[2,5]验证通用服务器恒等式。body/full的Σp ci′与−Σpδ/(ηK)最大误差均为1.49011612e−8。实际两轮run()第二轮已使用非零控制状态；body/full逐轮server不变量最大误差分别为5.58793545e−9、7.45058060e−9，均低于5e−6容差。

## 联通、计数和状态

真实运行训练、固定样本权聚合、metrics、原子checkpoint和JSON结果。两个客户端样本数[3,5]，权重[0.375,0.625]，dev为2个合成样本。每方法两轮共4次实际梯度求值、4次optimizer step；记录各client seed/K/梯度计数、每轮global anchor哈希、控制公式误差、旧/新加权控制不变量。最终metrics用同一helper重算一致。

checkpoint包含版本、完整config及签名、模型、全部client ci、全局c、固定权重、client样本数、完整轮历史。主体和全控制的checkpoint c与保存cis的样本加权和一致。全体通信均为**解析字节**，本实现不编码wire、网络或链。

| 方法 | 控制维数 | 每client单次dense delta上行 | 每client模型下行 | 额外control下行 | 每client ci状态 |
|---|---:|---:|---:|---:|---:|
| plain_dense | 0 | 547560B | 547560B | 0B | 0B |
| body_control | 136224 | 547560B | 547560B | 544912B | 544896B |
| full_control | 136874 | 547560B | 547560B | 547512B | 547496B |

单次上行64B假设头包含K和shape，仅上传精确dense delta；control上行额外字节为0，其重建已有SCALLION先例。config同时记录全局control状态字节与整个模拟的(client ci+global c)状态字节。

## 中断/复载及不兼容拒绝

固定rounds=2签名，在body_control完成round1后于下一client训练入口主动中断。由真实checkpoint复载完成round2后，模型、所有ci、c与连续两轮运行逐位一致，最终dev metrics相同；完成态再次复载不触发训练。以下10种情况都在训练前拒绝：

- `seed_mismatch`
- `backend_mismatch`
- `rounds_mismatch`
- `source_identity_mismatch`
- `tampered_config_with_old_signature`
- `checkpoint_signature`
- `checkpoint_version`
- `checkpoint_weights`
- `checkpoint_control_shape`
- `checkpoint_control_invariant`

负向测试只改临时fixture out，逐项恢复原config/checkpoint。run/PLAN哈希、原helper/runtime身份、数据manifest及固定权重均参与config签名；存储config自身签名也重算，checkpoint另核对版本/config/shape/权重/c不变量。

## 留档及启动交接

- 完整JSON：`work/control_reference/LOCAL_FIXTURES.json`。
- 可重复fixture源码：`work/control_reference/local_fixture_check.py`；SHA256`dce8df5f3255914eca4ea9e90714479bebfd084ebb2f2ef705a590c224c572cb`。
- 原始微型输出：`/var/folders/0t/lt7l05rn1zz3xrpbkj6ym_8m0000gn/T/control_reference_fixture_xn0z5_g3`。
- 检查前后未变的`work/control_reference/run.py`：`5a6f1376c6e5b997cd05a044fbc52060382484e80118314f8599d2ff13490223`。
- 检查前后未变的`work/control_reference/PLAN.md`：`d67374ec7621b9977b345be85c464b23c10192745f27c03d133576b1187b97db`。
- 检查前后未变的`work/head_control/run.py`：`8ee487ea523ee772a99e3555174cc9f41d2c9e1ade300f58b3df778562c63b82`。
- 检查前后未变的`work/head_control/PLAN.md`：`31b014425b47eb7af3fe36c51d755246e57127ebed68034f4e02572db277b48e`。
- 检查前后未变的`work/mixed_precision/codec.py`：`46f3601e5c107f192d359227b21aea2734cc5e798674f2dccbbdf46fc796d863`。

由主任务决定是否在新out启动，默认三方法、seed20261191、20轮：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python -B work/control_reference/run.py --out /新的正式输出目录 --backend mps --rounds 20 --seed 20261191
```

该命令由主任务掌握。现有head_control_dense只作同协议背景，不能作为独立重复。本fixture验证CPU数学实现与复载，未验证MPS数值路径、正式数据上的收益或20轮稳定性；其数据范围由synthetic标记界定。

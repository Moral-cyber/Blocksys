# Calibrated lowbit 新确认入口：CPU synthetic fixtures

2026-11-04。结论：**PASS，源码已冻结并交主任务决定启动**。本地只运行synthetic fixtures，没有调用正式TRAIN加载/训练入口。PLAN全文1356字符，满足≤1400字约束。旧balanced及prior/bias STOP不变。

## 交付入口

```bash
/opt/anaconda3/envs/Tinyproto/bin/python -B work/calibrated_lowbit_confirm/train.py --out /新训练目录 --backend mps --rounds 20 --partition-seed 20261203 --seed 20261211
/opt/anaconda3/envs/Tinyproto/bin/python -B work/calibrated_lowbit_confirm/evaluate.py --model-root /新训练目录 --out /新评估目录
```

三训练组固定ce_uniform3、fedlc_uniform3、fedlc_dense，评估只接收三组相同协议的完整final checkpoint；不能用中间checkpoint。正式启动归主任务，本文件不是已完成真实3×20轮的证据。

可复算fixture：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python -B work/calibrated_lowbit_confirm/local_fixture_check.py
```

## strict TRAIN与分区

common仅反序列化五个data_batch_1..5，并核对旧冻结数据manifest中的SHA256及TRAIN label hash；然后直接调用冻结partition_indices＋IndexedCifar10。没有调用旧data.prepare或torchvision CIFAR10构造器。新入口的test_batch审计钩子用一个不存在的dummy路径验证：在文件打开前抛出ValueError，实际官方文件读取为0。历史test字节MD5检查与test记录推理/指标选型是不同问题，此入口两者都不发生。

用50000个合成labels（每类5000）直接运行partition函数，新seed20261203重复manifest一致、全覆盖/互斥且globaldev2000。该合成类序下实际TRAIN=37501、holdout=12499；正式入口按真实TRAIN行顺序生成并保存实际counts，不硬编码名义37500。

## 三组微型真实联通

CPU单线程、12个随机uint8合成图像，两个client分别3/5 TRAIN样本、各2个holdout样本。数据归一化直接使用同一IndexedCifar10。三组真实train.run各两轮，复用冻结balanced_quant.train和codec；每轮每client一个实际batch。

| 方法 | 实际tiny rounds | 梯度求值总数 | 每client上传 | EF最大误差 |
|---|---:|---:|---:|---:|
| ce_uniform3 | 2 | 4 | 54067B | 0 |
| fedlc_uniform3 | 2 | 4 | 54067B | 0 |
| fedlc_dense | 2 | 4 | 547560B | 0 |

三组同seed的batch index序列一致，初始anchor逐位一致，同轮两个client之间anchor不变；FedLC量化/dense首轮原始local delta逐位一致。每个uniform3实际wire都是54067B≤54808，保存first_payload.bin逐字节匹配；独立decoder+旧residual逐轮重建发送量，样本加权聚合末态模型及EF残差逐位一致。dense无量化残差，上传/下行解析字节明示。

## config与checkpoint严格复载

config自身签名重算通过，完整记录源/PLAN、固定依赖、数据身份、IDs/counts/weights、版本与runtime。checkpoint绑定完整config并验证轮历史、EF tensor shape/dtype/device/finite、模型finite；dense残差必须零。

在FedLC uniform3真实完成round1后于下一client入口主动中断，再由checkpoint恢复：最终模型与EF残差和连续运行逐位一致；完成态重载不触发训练。下列11类不兼容均在训练前拒绝：

- `seed_mismatch`
- `partition_seed_mismatch`
- `rounds_mismatch`
- `backend_mismatch`
- `source_identity_mismatch`
- `config_self_signature_mismatch`
- `checkpoint_signature`
- `checkpoint_version`
- `checkpoint_full_config`
- `checkpoint_ef_shape`
- `checkpoint_ef_nonfinite`

评估另外验证：中间checkpoint、不同partition manifest均在任何forward或创建评估out前拒绝。负向测试只操作临时synthetic输出，原文件逐项恢复。

## 微型最终评估

三组完整final checkpoint进入实际evaluate.run。每client固定raw、analytic_prior、prioranchor_bias、matched_learned_head、matched_learned_head_plus_prior。bias只由保存的TRAIN logits/GT重新拟合，复算bias逐位一致、40B；TRAIN/holdout IDs互斥，holdout不参与bias拟合。matched head只最后650参数更新，主体逐位固定；三个模型使用相同client adaptation seed=trainseed+100000+i、batch64/SGD.03/epoch1。

本fixture新增bias TRAIN图像forward24次、CNN backward0，matched head共6次梯度求值；费用与状态分开记录。所有实际holdout bias metrics由保存数组复算一致。预设GO六项比较均完整写入；synthetic分数只用于测试决策代码，不是科学结果。

主门槛通过后must-next明确包含：同lambda10的零anchor local bias独立helper、vector scaling/BCTS主baseline、强FedCal vector、两fresh seed20261212/13及第二任务。此次没有扩至更多训练组，也未改冻结prior solver。

## 留档与冻结身份

- 机器证据：`work/calibrated_lowbit_confirm/LOCAL_FIXTURES.json`。
- 复算源码：`local_fixture_check.py`；SHA256`062f2c7491b4205f88aec4880ab34be1d30d2d4ef04cfe1fa3977b0c3c4d92e8`。
- 本次微型原始输出：`/var/folders/0t/lt7l05rn1zz3xrpbkj6ym_8m0000gn/T/calibrated_lowbit_confirm_fixture_85can77_`。
- 源身份检查前后相同；冻结balanced_quant.run、mixed codec、prior_bias solver严格pin哈希，未修改任何既有source/PLAN/结果。

| 新正式源码 | SHA256 |
|---|---|
| PLAN.md | `ac6f2f4b9582e95897f9d9dedcd3af3481114c3824674cc1cee02832a3144597` |
| common.py | `3f2e4693488d7e5a35a182e5340edb9d73f5cafc45b7c0e61b1e2cc392a22d9a` |
| train.py | `7fe8777db0ab8b2309140b39026a9bdbec42cd09ba9d2b8ef0c89b831a5641bd` |
| evaluate.py | `afcb8a47dc9c303b3e7d558e4e91b50b560cd10dd0287956969d61ab27e13250` |

主任务已收到PASS与上述source身份，本子任务之后仅补此文档，不再修改正式源。此验证覆盖CPU公式/状态/数据路由；MPS正式训练与新分区结果仍需主任务实际运行与独立复核。

# Balanced quant 独立CPU synthetic fixtures

2026-11-04。结论：**PASS，未发现阻断错误**。未改`run.py`、`PLAN.md`、冻结mixed旧实现或codec；未读取正在运行的正式结果。

可复算入口：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python -B work/balanced_quant/local_fixture_check.py
```

运行环境：Tinyproto Python，CPU单线程。所有图像、标签与计数均合成；未调用data.prepare。Python audit hook禁止任何`/dataset/`文件打开，实际打开尝试为0；没有official TEST读取或长训练。

## Dirichlet1损失与边界

对同一随机logits/labels，独立比较`CE(z+log(n+1),y)`与`CE(z+log((n+1)/(n_total+10)),y)`，计数含缺失类。FP64/FP32的CE差均0，logit梯度最大差分别3.46944695e−17、1.11758709e−8，确认共同归一常数可省略。

每类相同n∈{0,1,7,1000000}时，FP64/FP32均退化普通CE；观察最大CE误差2.38418579e−7、梯度误差9.31322575e−9。全零、单类、缺失类三个计数情形、三种loss均得到有限shift/CE/梯度，使用放大100倍logits仍有限。另三种loss分别真实训练3个单类合成样本一个batch，delta均有限。

FedLC shift逐位等于冻结旧mixed实现的`−max(n,1)^(-1/4)`。进一步使用同模型、同seed、同67个样本的两个batch：新FedLC与冻结旧mixed calibrated路径的完整delta及batch index顺序逐位相同；无额外梯度评分改变训练。

## 配对batch、梯度次数和global anchor

SyntheticIndexDataset保留labels/indices接口；在本地训练集合外放入10个class9 backing rows，检查train_counts只使用indices内样本，排除行的class9计数仍为0。

CE/FedLC/balanced各67个合成3×32×32样本、batch64、2个实际batch。通过记录真实Tensor.backward和SGD.step调用，三组均为**每batch恰1次backward、1次step**，与返回steps/gradient_evaluations一致。三组同seed20261201，batch index流相同，client训练前后全局model/anchor逐位不变。

## 实际四组两轮、uniform3与EF

两个synthetic client分别3/5样本，固定权重[0.375,0.625]，dev为2个合成样本。真实运行四方法各2轮；client种子20261201、20261202、20261301、20261302、batch流一致。四组初始anchor一致，每轮client之间anchor不变；balanced_uniform3/dense首轮同一client原始训练delta逐位相同。后续模型随各方法更新分别演化。

实际encoder输入、wire、decoder输出与本地delta均独立记录，并以客户端旧residual逐轮重建输入/新residual；使用真实decoder重新计算样本加权聚合，最终模型和保存residual逐位一致。各量化组`first_payload.bin`与实际首包逐字节一致，解码长度136874；每包实际54067B≤54808B，余量741B。

| 方法 | 实际wire数 | 每client上传 | FP32 EF最大误差 |
|---|---:|---|---:|
| ce_uniform3 | 4 | 54067B | 0.0 |
| fedlc_uniform3 | 4 | 54067B | 0.0 |
| balanced_uniform3 | 4 | 54067B | 0.0 |
| balanced_dense | 0 | dense解析547560B/client | 0.0 |

两轮共核查12个真实uniform3包；本组合成输入FP32 EF重构误差为0。dense组cap=None且不生成first_payload.bin；所有组下行均按dense解析547560B/client核对。最终raw-logit metrics用同一helper重算一致；未在推理添加训练shift。

## 配置签名与复载

每组的config签名独立重算，与checkpoint/result签名相同；绑定run、PLAN、codec哈希、partition seed20261193、manifest、weights、raw-logit推理规则。固定rounds=2，在balanced_uniform3完成round1后于下一训练入口主动中断，再由真实checkpoint恢复：末态模型、所有client residual与连续运行逐位相同，最终dev metrics相同；完成态复载不触发训练。

以下8种不兼容情况均在训练前拒绝，负向测试只修改临时fixture输出或内存中的身份函数，逐项恢复：

- `seed_mismatch`
- `partition_seed_mismatch`
- `rounds_mismatch`
- `backend_mismatch`
- `run_source_mismatch`
- `plan_source_mismatch`
- `codec_source_mismatch`
- `wrong_checkpoint_signature`

## 留档与范围

- 机器可读完整结果：`work/balanced_quant/LOCAL_FIXTURES.json`。
- 可复算源码：`work/balanced_quant/local_fixture_check.py`；SHA256`a0631d9cc838b4bc3a3f0eaee43792315785fd887280799af130d9142d9465a4`。
- 本次原始微型输出：`/var/folders/0t/lt7l05rn1zz3xrpbkj6ym_8m0000gn/T/balanced_quant_local_fixture_iutzqb0x`。
- protected source前后哈希一致：
- `work/balanced_quant/run.py`：`b94f6a3a985839e6781582c6203e3dd8154fc97cf7bcc8edfe9d4648240625d4`。
- `work/balanced_quant/PLAN.md`：`6c58ed74e04c8b7477faa7fcaee2e45f11f86f02775bd0642c6b08da2a304397`。
- `work/mixed_precision/run.py`：`f08b0fa40ad32a7d486153737f3225ba9e386f1aa2f4b69a125949b679436cd7`。
- `work/mixed_precision/codec.py`：`46f3601e5c107f192d359227b21aea2734cc5e798674f2dccbbdf46fc796d863`。

此次验证CPU实现、等价式、实际字节、EF与复载保护；没有验证MPS数值路径、正式四组20轮收益或泛化。正式实验session由主任务负责，本fixture没有更改其数据、源码、配置、状态或结果。

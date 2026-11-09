# Head-control 本地实现 fixtures

2026-11-04。结论：**PASS，未发现阻断错误**。检查对象为冻结 `PLAN.md` 与 `run.py`；本次没有修改训练实现或 frozen mixed_precision codec。

检查完全使用合成张量。运行环境为 `/opt/anaconda3/envs/Tinyproto/bin/python`，CPU、单线程。独立可重复入口：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python -B work/head_control/local_fixture_check.py
```

该入口生成新的临时 fixture out，真实执行四方法各 2 个 tiny client × 1 round，及 67 个合成 RGB 样本的 2-batch 局部检查。整个完整 fixture 共 14 个微型 SGD step；完成 checkpoint 重载与拒绝错误签名时不训练。`data.prepare` 未被调用。Python audit hook 禁止打开任何 `/dataset/` 文件；实际打开尝试为 0。

## 参数布局与梯度检查

- `SmallCNN` 总参数 136874；末尾确为 `classifier.3.weight`（10×64）及 `classifier.3.bias`（10），合计 head=650、body=136224。
- 使用 67×3×32×32 合成输入，batch64，真实走 2 个 SGD step。以参数 backward hook 捕获校正前梯度，再在 SGD.step 前观察实际梯度；这两个探针只记录张量，不改变数学运算。
- 每个 batch 的主体梯度直接改变量最大值为 **0.0**；头部实际增加量与 `c−ci` 的最大误差 **6.98491931e-09**。后续 batch 的主体梯度可因模型头变化间接受影响，这不属于直接施加主体校正。
- `ci' = ci−c−delta_head/(0.03×K)` 与逐 batch **原始**头梯度均值最大误差 **1.23865902e-07**，低于实现 5e−6 容差；返回的 K=2 与实际 step 一致。
- 局部训练前后全局 model/anchor 逐位相同。相同 seed 重复的 delta/ci 逐位相同；controlled/plain 的 batch index 流一致。

## 实际 wrapper 与 EF

使用完整 136874 维合成更新：normal、全零、body 最后坐标脉冲。每例都通过真实 wrapper 编码再由独立的 wire 路径解码。32B wrapper + 2600B FP32 head 后，body budget 恰为 **52176B**。

| 合成更新 | body 实际 wire | wrapper 总 wire | 54808B cap 余量 | body FP32 EF 最大误差 |
|---|---:|---:|---:|---:|
| normal | 52153B | 54785B | 23B | 1.86264515e-09 |
| zero | 36785B | 39417B | 15391B | 0 |
| body_tail_impulse | 36785B | 39417B | 15391B | 0 |

每例的解码 head 与输入 head 逐位相同、解码 body 与编码发送值逐位相同、actual K 相同。12 类损坏包均拒绝：短头、截断、尾随字节、版本、body/head 维数、K=0、长度、reserved、magic、非有限 head、非法 body bitwidth。

## 固定权重控制不变量

构造两个非零旧 ci，令旧 c=0.375ci₁+0.625ci₂，实际 K 分别为 2、5；头更新经过真实 wrapper 后，分别按接收的 head delta 与 K 重建。

- `Σp_i ci_i'` 与服务器直接式 `−Σp_i delta_hi/(ηK_i)` 的最大误差 **3.7252903e-09**。
- 重建各 ci' 与预设 raw-gradient mean 最大误差 **1.49011612e-08**。

## 实际 run() 一轮联通与 checkpoint

两个客户端各 3、5 个合成样本，dev 为 2 个合成样本；真实运行 run()、train()、encoder、decoder、aggregation、checkpoint。样本权确为 [0.375, 0.625]，四方法共用 seed20261191/20261192、初始化 anchor 和逐 client batch order。零初始控制下，四方法首轮原始本地 delta 逐位相同。

| 方法 | 实际轮数 | 两客户端实际上传 | cap 处理 |
|---|---:|---|---|
| plain_uniform | 1 | 54067B, 54067B | 均≤54808B |
| exact_head | 1 | 54785B, 54785B | 均≤54808B |
| head_control | 1 | 54785B, 54785B | 均≤54808B |
| head_control_dense | 1 | 547560B, 547560B | 配置 cap=None；dense 诊断组 |

两个 controlled 方法存入的 c 均逐位等于存入 cis 的样本加权和；对接收头 delta 的服务器直接重建最大误差为 0。上传合计等于各 client 之和；额外控制下行按照 16+2600B/client 核对。全局 anchor 在每个 client 训练期间保持逐位不变。

实际读取这些 **fixture 自己生成** 的 config/result/checkpoint，并验证签名重算一致，config 绑定 run、PLAN 和 codec 的 SHA256。完成态 checkpoint 同配置重载不触发训练；以下情况均在训练前明确拒绝：seed 改变、backend 改变、内存中模拟 source identity 改变、checkpoint signature 错误。测试仅修改临时 fixture out，完成后恢复其原始 config/checkpoint。

## 留档及证据边界

- 机器可读证据：`work/head_control/LOCAL_FIXTURES.json`。
- 可重复 fixture 源码：`work/head_control/local_fixture_check.py`，SHA256 `1cf78ded0bb0b64e8924c0ac2d17a982a09072a9f1b156223222da557ec095a8`。
- 本次临时原始输出：`/var/folders/0t/lt7l05rn1zz3xrpbkj6ym_8m0000gn/T/head_control_local_fixtures_0kkqb1yf`。
- 检查前后相同的 `work/head_control/run.py` SHA256：`8ee487ea523ee772a99e3555174cc9f41d2c9e1ade300f58b3df778562c63b82`。
- 检查前后相同的 `work/head_control/PLAN.md` SHA256：`31b014425b47eb7af3fe36c51d755246e57127ebed68034f4e02572db277b48e`。
- 检查前后相同的 `work/mixed_precision/codec.py` SHA256：`46f3601e5c107f192d359227b21aea2734cc5e798674f2dccbbdf46fc796d863`。

这些结果验证本地实现、张量恒等式、字节约束与复载拒绝路径；质量判断仍来自冻结四组完整开发试验。这里没有官方数据加载、官方 test 读取、长训练、MPS 数值路径验证，或多轮控制效益结论。临时 run() config 内通用的生产评估描述沿用实现原文；本 fixture 的实际数据范围以本文件与 JSON 的 synthetic 标记为准。

# 部分融合诊断：本地独立代码与数据语义检查

检查日期：2026-11-03。检查类型：独立本地代码检查与 CPU 数值 fixture；**外部跨模型审计：`REVIEW_UNAVAILABLE`**。本报告不能代替外部审计，也不构成新颖性结论。

**结论：未发现当前运行的 train/dev 泄漏、向量错位、缺失上传被当零、排 self 失效或汇总错误。按冻结规则独立复算的决定为 `STOP_THIS_EXACT_EXTERNAL_GATE`。** gate_external 优于 local 与直接稀疏覆盖，但未优于相同额外梯度和回溯候选预算的 extra_sgd，不能转入本候选的后继闭环来宣称成功。

## 范围与可复算证据

- 仅读取本次冻结源文件、当前 `dev_seed20261143/config.json`、`warmup.pt`、`shared_state.pt`、`result.json`，并运行 CPU 小 fixture；未修改冻结源文件、重跑训练或读取历史 official test 结果字段。
- `config.json` 列出的 6 个源文件 SHA256 均与当前文件一致。主脚本 SHA256：`adc720706453793c68bca758e4f8c43296ea10ae6ae5d243449cfaa165c715ae`；计划 SHA256：`1ef5f2a673bb131d9e9c822b8fdee269a4008a7e72b52f29f228ffad13bd46ab`。
- config、warmup、shared_state、result 的签名相同：`9177c84dba22af85424fa58ef9f747bc69675383d5e6ab6cf0c8177098c3ad4f`。warmup checkpoint 记录 round=10，参数与 shared_state 的 anchor 逐位相等。这只校验当前产物一致性，未重新执行其训练轨迹。
- fixture 入口：`local_check_fixtures.py`；机器可读证据：`LOCAL_CHECK_FIXTURES.json`。Python：`/opt/anaconda3/envs/Tinyproto/bin/python`，CPU，单线程。最终状态 `PASS`。

## 数据语义与泄漏

1. 从原始索引独立复算，20 个客户端共 37,500 train、12,500 dev；二者交集为 0，合起来恰好覆盖 original row 0..49,999，每个样本一次。重新计算的分组索引哈希与清单相符。最小客户端 train=104、dev=35。
2. `probe_fusion.py:128–133` 明确以完整客户端 holdout 构造 devsets，训练仅用 trainsets；`run_partitions.py:83–104` 仅加载 `CIFAR10(train=True)`。没有本脚本可执行的官方 TEST 评估入口。当前运行的 `official_test_loaded=false` 与这条代码路径一致。
3. `train_batches`（主脚本 63–74）从客户端 TRAIN 位置做确定性无放回划分。本次 result 的全部 20 对 batch 位置与单独复算的 RNG 一致；映射回 original row 后，fit/check 互斥且与全部 dev 无交集。train=104 的客户端使用 52+52，其余均为 64+64。这些行已参与该客户端本地 epoch，故 check 是训练内回溯批次，不能称独立验证集。
4. `gradient`、整包 gate 和回溯只接触训练批次；dev 仅用于报告与探索性 GO/STOP。开发选择依赖 dev 是本计划明示范围，不能再把本次 dev 指标当未参与选择的最终性能。
5. 继承的清单还含一个 2,000 行分层 `validation_indices` 子集，但本 probe **没有使用它**；实际汇总的样本数是完整 12,500 holdout。后续报告不能照抄 `validation_samples=2000` 作为本次评估规模。每种方法的 20 个 metrics.samples 均与相应客户端 holdout 长度一致。

## 向量、包、聚合和排 self

- 参数展开为 136,874 个 FP32 值，1,070 个连续包；最后短包 42 值，上传 k=floor(1,070×.05)=53 包。fixture 验证每个位置恰好属于一包、assign/vectorize 往返一致、向量与模型无存储别名。
- `SmallCNN` 没有 BN/dropout 或其它状态 buffer；训练 `.train()` 与 CPU `.eval()` 不引入漏传 BN buffer 问题。最后 Linear 的 650 个参数确为向量尾部，所以 `keephead[-650:]` 保留的是计划所述最后分类 Linear。它不是保留整个 classifier 模块。
- 各客户端从共同 anchor 训练私有副本；MPS adapter 返回 delta，不改传入的全局 model。上传按 delta 包能量选支持，但服务器聚合的是 `anchor+delta` 的模型值，与计划一致。
- 手工 fixture 区分“上传值 0”和“没有上传”；缺失位置不进入分母。排 self 同时移除该客户端的分子和分母贡献，余下 train 样本权重在每位置重新归一化；没有 donor 时保持接收者本地值。当前 20 个 receiver 的全部参数又用独立 NumPy FP64 计算核对，最大绝对误差 5.92e-08，符合 FP32 累加误差。
- 当前状态所有 mask 都恰含 53 个完整选中包（含真实短包），选中能量不低于未选中包；result 的 pack_ids、values、payload_bytes 均与保存 mask 一致。重构 delta=local-anchor 会有最后位相减误差，fixture 因此检查支持的能量分离，而不虚称逐位恢复了原始 delta。
- gate 的方向为 available×(donor−local)，整包以 −gᵀdirection>0 保留；手工五维 fixture 验证正、负与无 donor 包的处理。本检查没有发现包索引错位或对缺失值作零覆盖。

## 计划一致性与计算公平性限制

1. **轻微冻结协议偏差：回溯平局顺序。** 计划列 alpha=(1,.5,.25,0)，代码（91–96）用 (0,1,.5,.25)，且 Python min 取首个最小值。零方向 fixture 实际选 alpha=0，计划顺序则会标 alpha=1。当前三种回溯方法的 60 条记录均无精确最小损失平局，因此这次观察到的选点与 GO/STOP 不受此差异影响。保留本次运行并记录偏差；不要事后改冻结源文件后仍沿用同一身份。
2. **梯度/候选预算可比，不能称总计算量或耗时相等。** 三种方法共享同一个训练梯度，各自在相同 check 批次比较 0、1、.5、.25。单独执行一种方法是一次梯度计算（含其 forward/backward）+四次 check forward，其中三次为非零候选，另一次为 baseline。fixture 计数确认 line_search 实际四次 forward；它本身不做 backward，返回的 extra_backward=1 是方法级名义归属。
3. 当前合并运行只实际求一次共享梯度，随后做三个 line_search，共十二次 check forward；baseline 在三次 line_search 中重复计算，并未缓存复用。gate 另有整包内积、稀疏 donor 聚合等工作，SGD 有全向量缩放；预算相同不足以推出这些工作与 wall-time 相同。
4. **耗时记录不完整。** rows.seconds 包含该客户端三种回溯、所有八个 dev 评估等总时间，没有方法级计时。它支持“此客户端诊断总时间”，不支持任何具体方法的加速比或额外训练耗时相等。计划“每种额外工作和耗时记录”在方法级耗时上未完全实现。
5. **通信是解析 payload 核算。** 当前每客户端 52×128+42=6,698 值，64 字节固定头+53×4 字节包 ID+6,698×4 字节 FP32=27,068 字节；20 客户端共 541,360 字节。相同固定头假设下，dense 上传参照为 10,951,200 字节。短包核算正确，但脚本没有编码成真实包、实际网络/链上测量、下载/广播或服务器开销核算，不能将此称实际线缆/链上字节。
6. 上传约 5% 不等于下载约 5%。聚合其他客户端上传的支持后，当前外部 donor 可用参数数为 42,666–44,586（均值 43,754），占全模型 31.17%–32.57%（均值 31.97%）。相应 donor 下发或客户端专属聚合不在所报 payload 里。dense_average 只能称不同通信代价的参照，不能称等字节强对照。
7. checkpoint signature 绑定主脚本/计划/参数，却未把 imported helper 的哈希纳入签名；本次源文件哈希全部已匹配，因此不是本次失效证据。未来若改 helper，应另开身份或升级签名，不能仅凭旧 signature 继续恢复并称同一实现。

## 独立统计复算与去留

未调用 probe.summarize；从当前 result 的 20 个客户端行分别复算均值、加权均值、受益/受损比例，八种方法各字段与保存 summary 的绝对差均 <1e−12。下表“受益/受损”按 CE 相对 local（1e−8 容差）统计；未列入者为持平。

| 方法 | client 宏 CE | client 宏 accuracy | 样本加权 CE | 样本加权 accuracy | 受益/受损客户端 |
|---|---:|---:|---:|---:|---:|
| `local` | 0.93055270 | 67.5291% | 0.88862391 | 66.6960% | 0/0 |
| `dense_average` | 1.99526310 | 29.5764% | 1.95679668 | 29.4880% | 1/19 |
| `sparse_inclusive` | 1.45466386 | 54.7055% | 1.37551411 | 57.2960% | 2/18 |
| `sparse_external` | 1.61344120 | 46.6041% | 1.65117499 | 44.6880% | 2/18 |
| `sparse_external_keephead` | 0.99198536 | 65.8176% | 0.98164016 | 64.4720% | 5/15 |
| `gate_external` | 0.75327170 | 72.0681% | 0.74158159 | 70.9360% | 14/1 |
| `gate_inclusive` | 0.75404656 | 71.9827% | 0.74332318 | 70.8800% | 14/1 |
| `extra_sgd` | 0.72091635 | 73.1269% | 0.72361924 | 71.8960% | 17/2 |

排 self gate 相对 local：宏 CE −0.17728101，accuracy +4.5390pp；14 个客户端 CE 改善、1 个受损、5 个持平。相对 extra_sgd：宏 CE **+0.03235535**，accuracy **−1.0589pp**；仅 6 个客户端 CE 更好，13 个更差，1 个持平。

冻结 GO 要求 gate 的宏 CE 同时严格低于 local 与 extra_sgd，且 accuracy 对二者均不落后超过 0.5pp。本次相对 extra_sgd 同时违反 CE 和 accuracy 条件，所以独立决定与 result 一致：**`STOP_THIS_EXACT_EXTERNAL_GATE`**。不能因为 gate 比直接覆盖明显改善就省略同额外梯度 SGD 对照。

gate_external 与 gate_inclusive 的差异仅宏 CE −0.00077486、accuracy +0.0854pp；当前一个 seed/一个共享状态不能证明排 self 组件必要。由于整个外部 gate 已未过停止线，也不能据此结果挑其它 pack、阈值、回溯在同一状态补救并称通过原计划。

这里只报告一个模型初始化、一个划分、一个共同训练状态的描述性结果。客户端共享同一训练过程，20 个客户端不是 20 次独立训练重复；不据此构造多 seed 置信区间、显著性结论、长程 PFL 优越性或最终 TEST 性能。**外部跨模型审计保持 `REVIEW_UNAVAILABLE`。**

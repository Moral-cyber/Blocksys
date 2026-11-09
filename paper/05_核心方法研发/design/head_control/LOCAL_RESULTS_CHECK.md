# Head control 本地独立结果核验

实验状态：**ALL_DONE**；冻结判据决定：**STOP_THIS_FIXED_HEAD_CONTROL_RECIPE**。保存产物核验：PASS（3111/3111）。外部跨模型审查：`REVIEW_UNAVAILABLE`。

输入仅为 config/result/checkpoint 与首个保存 payload；未导入训练脚本或原 codec，未训练、未读取数据集或 official TEST。

| 方法 | 状态/轮数 | last5 CE | last5 accuracy | 累计梯度次数 | 累计实际/解析上行B | 解析下行B | 总通信B |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain_uniform | DONE / 20 | 1.7989318725585934 | 35.78999996185303% | 11940 | 21626800 | 219024000 | 240650800 |
| exact_head | DONE / 20 | 1.7990290252685548 | 35.78999996185303% | 11940 | 21914000 | 219024000 | 240938000 |
| head_control | DONE / 20 | 1.7730224151611331 | 35.950000286102295% | 11940 | 21914000 | 220070400 | 241984400 |
| head_control_dense | DONE / 20 | 1.7712562713623048 | 36.09000027179718% | 11940 | 219024000 | 220070400 | 439094400 |

固定判据：round16–20 均值，head_control CE 严格低于 plain_uniform 与 exact_head，accuracy 分别至少提高 0.005（0.5pp）；四组全部 DONE 后才判断。
- 相对 plain_uniform：CE降低 0.0259094574；accuracy 增益 0.160000pp；CE条件=True，accuracy条件=False。
- 相对 exact_head：CE降低 0.0260066101；accuracy 增益 0.160000pp；CE条件=True，accuracy条件=False。
- dense仅诊断：压缩组减dense CE=0.0017661438，accuracy差=-0.140000pp。

plain_uniform：实际 batch 步数 [30, 46, 4, 93, 39, 8, 2, 5, 27, 51, 9, 62, 66, 5, 44, 22, 16, 3, 40, 25]；逐 client gradient_evaluations 与 steps 一致。上行范围 [54067, 54067]B/client/round；解析下行 547560B/client/round，其中控制额外 0B=16+650×4。
保存诊断最大 control公式误差=0，加权control invariant误差=0，body EF误差=1.16e-10；最终checkpoint独立加权control误差=None。
首包独立解码：54067B，136874个有限FP32坐标；wrapper=None；保存长度/位宽/尺度/头delta范数与首轮metadata一致。

exact_head：实际 batch 步数 [30, 46, 4, 93, 39, 8, 2, 5, 27, 51, 9, 62, 66, 5, 44, 22, 16, 3, 40, 25]；逐 client gradient_evaluations 与 steps 一致。上行范围 [54785, 54785]B/client/round；解析下行 547560B/client/round，其中控制额外 0B=16+650×4。
保存诊断最大 control公式误差=0，加权control invariant误差=0，body EF误差=4.66e-10；最终checkpoint独立加权control误差=None。
首包独立解码：54785B，136874个有限FP32坐标；wrapper={'bytes': 32, 'body_payload_bytes': 52153, 'head_bytes': 2600, 'steps': 30}；保存长度/位宽/尺度/头delta范数与首轮metadata一致。

head_control：实际 batch 步数 [30, 46, 4, 93, 39, 8, 2, 5, 27, 51, 9, 62, 66, 5, 44, 22, 16, 3, 40, 25]；逐 client gradient_evaluations 与 steps 一致。上行范围 [54785, 54785]B/client/round；解析下行 550176B/client/round，其中控制额外 2616B=16+650×4。
保存诊断最大 control公式误差=4.47e-07，加权control invariant误差=4.66e-09，body EF误差=4.66e-10；最终checkpoint独立加权control误差=0.0。
首包独立解码：54785B，136874个有限FP32坐标；wrapper={'bytes': 32, 'body_payload_bytes': 52153, 'head_bytes': 2600, 'steps': 30}；保存长度/位宽/尺度/头delta范数与首轮metadata一致。

head_control_dense：实际 batch 步数 [30, 46, 4, 93, 39, 8, 2, 5, 27, 51, 9, 62, 66, 5, 44, 22, 16, 3, 40, 25]；逐 client gradient_evaluations 与 steps 一致。上行范围 [547560, 547560]B/client/round；解析下行 550176B/client/round，其中控制额外 2616B=16+650×4。
保存诊断最大 control公式误差=4.62e-07，加权control invariant误差=4.66e-09，body EF误差=0；最终checkpoint独立加权control误差=0.0。

核验包括签名、冻结身份、共享配置/分区、ID哈希和互斥、样本权重、逐轮步数/梯度数、codec字节重算、完整checkpoint/result行及指标一致、控制状态及首包独立解析。

单独完成的只读静态核对（下列源文件位置并非该核验脚本的运行输入）：
- run.py:42-71: one CE backward per actual batch; correction is added only to the last weight/bias gradients. The telescoping identity c_i'=mean(raw head gradients) is tested inside training at 5e-6, not independently replayed here.
- run.py:74-94,132-144: compact arms decode their actual wire before weighted aggregation; head is FP32 exact and has no EF; body alone receives EF for exact_head/control. Wrapper is 32B, head is 2600B, body budget is 52176B.
- run.py:142-156: because c=sum(p_i*c_i), full participation and fixed p_i cancel old control terms, so the server reconstruction sum(-p_i*delta_h/(eta*K_i)) matches weighted new c_i. This argument does not establish partial-participation support.
- run.py:157-166: checkpoint includes the same saved metric rows as result.json; c/control/client residual states are checkpointed. All download counts and dense uploads are analytical.
- run.py:178-181 and PLAN.md:29-32,36-44: shared fixed official-TRAIN development split, no official TEST call on this route; frozen last5 criterion and one exploratory seed/partition.

证据边界：
- Single development seed/partition after repeated development choices; rounds are not independent seeds.
- Training is MPS according to saved metadata; this checker only runs CPU saved-file arithmetic, not an independent MPS trajectory reproduction.
- Checkpoint/result equality checks saved metrics; predictions/CE are not reevaluated because no dataset is read.
- Only first saved quantized payload per arm is independently parsed; historical payloads and local raw gradients are not saved, so EF/control per-step bounds are recorded diagnostics, not independently replayed.
- Dense upload and every download are analytical byte counts; no network, energy, blockchain or distributed deployment is tested.
- Official TEST is not read by this checker. Exclusion in the experiment is supported by saved metadata and separately reviewed static source, not filesystem access tracing.
- No external cross-model reviewer is available: REVIEW_UNAVAILABLE, not a formal external integrity acquittal.

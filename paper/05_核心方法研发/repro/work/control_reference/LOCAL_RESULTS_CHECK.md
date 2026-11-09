# 既有控制基线：本地独立结果核验

实验状态：**ALL_DONE**；保存产物检查 **PASS（444/444）**；外部审查 `REVIEW_UNAVAILABLE`。

输入仅为四组已保存 config/result/checkpoint；不导入训练源文件，不加载数据、不训练。head背景复用同seed/分区，不构成独立重复。

| 组 | 轮数/状态 | last5 CE | last5 acc% | mean round-worst% | min mean-class% | 梯度次数 | 解析总通信B |
|---|---|---:|---:|---:|---:|---:|---:|
| plain_dense | 20/DONE | 1.7985032837 | 35.820000 | 0.000000 | 0.000000 | 11940 | 438048000 |
| body_control | 20/DONE | 1.8455745239 | 33.090000 | 0.000000 | 0.000000 | 11940 | 656012800 |
| full_control | 20/DONE | 1.9033007507 | 31.570000 | 2.700000 | 4.700000 | 11940 | 657052800 |
| head_control_dense_same_seed_background | 20/DONE | 1.7712562714 | 36.090000 | 0.500000 | 0.500000 | 11940 | 439094400 |

逐类 last5 accuracy 均值（class 0–9，%）：

| 组 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plain_dense | 26.4000 | 41.1000 | 15.2000 | 15.0000 | 69.3000 | 52.7000 | 0.0000 | 24.4000 | 72.9000 | 41.2000 |
| body_control | 26.7000 | 48.1000 | 16.6000 | 5.2000 | 69.9000 | 49.9000 | 0.0000 | 14.8000 | 72.0000 | 27.7000 |
| full_control | 31.1000 | 28.0000 | 17.5000 | 8.6000 | 69.5000 | 37.9000 | 4.7000 | 15.3000 | 66.0000 | 37.1000 |
| head_control_dense_same_seed_background | 30.2000 | 41.8000 | 16.7000 | 12.5000 | 68.9000 | 50.2000 | 0.5000 | 26.1000 | 72.6000 | 41.4000 |

mean round-worst = 每轮10类最小值再取5轮均值；min mean-class = 先取每类5轮均值再取最小值，两者分别报告。
- body_control 相对plain：CE降低-0.0470712402，accuracy差-2.730000pp，mean round-worst差0.000000pp。这是已有控制基线的同协议诊断，不支持新算法结论。
- full_control 相对plain：CE降低-0.1047974670，accuracy差-4.250000pp，mean round-worst差2.700000pp。这是已有控制基线的同协议诊断，不支持新算法结论。
- head_control_dense_same_seed_background 相对plain：CE降低0.0272470123，accuracy差0.270001pp，mean round-worst差0.500000pp。这是已有控制基线的同协议诊断，不支持新算法结论。

plain_dense：控制范围=none，维数=0；实际每client每轮steps=[30, 46, 4, 93, 39, 8, 2, 5, 27, 51, 9, 62, 66, 5, 44, 22, 16, 3, 40, 25]；gradient=11940，optimizer=11940（背景未单独记录optimizer）。
解析上行=219024000B，模型下行=219024000B，控制下行=0B；额外控制下行=0B/client/round；每client控制state=0B，全模拟控制state=0B（含server）。
记录的公式/prior invariant/new invariant最大误差=0/0/0；最终checkpoint独立加权control误差=0。

body_control：控制范围=prefix body，维数=136224；实际每client每轮steps=[30, 46, 4, 93, 39, 8, 2, 5, 27, 51, 9, 62, 66, 5, 44, 22, 16, 3, 40, 25]；gradient=11940，optimizer=11940（背景未单独记录optimizer）。
解析上行=219024000B，模型下行=219024000B，控制下行=217964800B；额外控制下行=544912B/client/round；每client控制state=544896B，全模拟控制state=11442816B（含server）。
记录的公式/prior invariant/new invariant最大误差=4.43e-07/0/5.59e-09；最终checkpoint独立加权control误差=0。

full_control：控制范围=all parameters，维数=136874；实际每client每轮steps=[30, 46, 4, 93, 39, 8, 2, 5, 27, 51, 9, 62, 66, 5, 44, 22, 16, 3, 40, 25]；gradient=11940，optimizer=11940（背景未单独记录optimizer）。
解析上行=219024000B，模型下行=219024000B，控制下行=219004800B；额外控制下行=547512B/client/round；每client控制state=547496B，全模拟控制state=11497416B（含server）。
记录的公式/prior invariant/new invariant最大误差=4.77e-07/0/1.21e-08；最终checkpoint独立加权control误差=0。

head_control_dense_same_seed_background：控制范围=suffix head，维数=650；实际每client每轮steps=[30, 46, 4, 93, 39, 8, 2, 5, 27, 51, 9, 62, 66, 5, 44, 22, 16, 3, 40, 25]；gradient=11940，optimizer=None（背景未单独记录optimizer）。
解析上行=219024000B，模型下行=219024000B，控制下行=1046400B；额外控制下行=2616B/client/round；每client控制state=2600B，全模拟控制state=54600B（含server）。
记录的公式/prior invariant/new invariant最大误差=4.62e-07/0/4.66e-09；最终checkpoint独立加权control误差=0。

单独只读静态核对（不属于脚本的运行输入）：
- run.py:43-51,89-98: checked last-650 head layout; none/body/full select 0/136224/136874 coordinate prefixes. Body correction stops before the head.
- run.py:75,85-103: fixed c-ci per round, one CE backward per actual batch, raw gradient accumulated before correction; ci_new=ci-c-delta_subset/(.03*actual_steps) is checked against raw gradient mean at 5e-6.
- run.py:143-146,196-220: with full participation and fixed p, sum(p*ci)=c implies sum(p*ci_new)=-sum(p*delta_subset/(eta*K)); prior and new invariants are checked. This does not establish partial-participation support.
- run.py:161-169,223-236: one analytical dense delta upload, analytical dense model download, plus 16+4d control download; per-client/global controls are each 4d and all 21 simulated control vectors cost 21*4d. The 64B header/K layout is assumed, not serialized.
- run.py:239-244: version/config/signature/weights/client_sizes/control tensors and complete metric rows are checkpointed; result uses those same metric rows.
- run.py:267-271; PLAN.md:3-13: fixed official-TRAIN development split, existing SCAFFOLD/SCALLION references; reused head_control_dense has the same seed/partition and is not an independent repeat or new algorithm.

证据边界：
- Existing SCAFFOLD full/body/head control references, not a new algorithm or independent confirmation of a contribution.
- Outcomes apply to this fixed 20-round, lr=.03, local-epoch1 SmallCNN/CIFAR10 development protocol. They do not establish what longer schedules or other fixed protocols would do, or invalidate SCAFFOLD generally.
- Same seed20261191 and partition20261143; reused head_control_dense adds no independent seed. Rounds are not independent repeats.
- No training or dataset/official TEST load in this checker; saved metrics are checked/reaveraged, not recomputed from predictions.
- Per-step raw gradients and updates are not saved; local telescoping and historical server invariants are recorded training diagnostics. Final control weighting is independently recomputed.
- MPS local training is supported by saved metadata/static source; this CPU checker does not independently reproduce its trajectory.
- All communication values are analytical, including assumed 64B delta header with actual K. No serialization/network/authentication/energy/chain measurement.
- Fixed full participation and fixed sample weights are essential to server delta-only control reconstruction.
- No external cross-model reviewer is available: REVIEW_UNAVAILABLE.

# Balanced quant：独立保存产物核验

实验状态 **ALL_DONE**；保存产物 **PASS（589/589）**；方案数字一致性 **WARN**；冻结判据 **STOP_THIS_FIXED_BALANCED_QUANT_RECIPE**；外部审查 `REVIEW_UNAVAILABLE`。

脚本仅仅读 config/result/checkpoint/首包，不导入训练源文件或原codec，不加载数据、不训练、不读取official TEST。

样本数警告：实际TRAIN37502、holdout12498；PLAN写37500/12500，源于逐client round(.75*n)。四组共享实际分区与样本权，IDs互斥合计50000，不能称精确遵守该PLAN数字。

| 组 | 状态/轮数 | last5 CE | last5 acc% | mean round-worst% | min mean-class% | 梯度次数 | 上行B | 解析下行B | 总通信B |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ce_uniform3 | DONE/20 | 1.7809861389 | 35.410000 | 9.500000 | 11.400000 | 11920 | 21626800 | 219024000 | 240650800 |
| fedlc_uniform3 | DONE/20 | 1.7741681885 | 35.840001 | 12.300000 | 12.500000 | 11920 | 21626800 | 219024000 | 240650800 |
| balanced_uniform3 | DONE/20 | 1.7814179413 | 35.720001 | 13.300000 | 13.300000 | 11920 | 21626800 | 219024000 | 240650800 |
| balanced_dense | DONE/20 | 1.7789119690 | 35.790000 | 14.200000 | 14.200000 | 11920 | 219024000 | 219024000 | 438048000 |

冻结GO：四组各20轮DONE；round16–20 balanced_uniform3对CE/FedLC分别CE更低且accuracy≥+.005；对balanced_dense accuracy≥−.01。全部条件同时成立。
- 相对ce_uniform3：CE降低=-0.0004318024；accuracy差=0.310001pp；CE更低=False；≥+.5pp=False。
- 相对fedlc_uniform3：CE降低=-0.0072497528；accuracy差=-0.120000pp；CE更低=False；≥+.5pp=False。
- 相对balanced_dense：量化减dense CE=0.0025059723；accuracy差=-0.069999pp；≥−1pp=True。

逐类last5均值（class0–9，%）：

| 组 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ce_uniform3 | 50.6000 | 51.6000 | 46.3000 | 39.3000 | 12.0000 | 11.4000 | 40.8000 | 21.6000 | 40.9000 | 39.6000 |
| fedlc_uniform3 | 49.7000 | 52.9000 | 44.1000 | 35.4000 | 17.1000 | 12.5000 | 43.5000 | 23.3000 | 41.2000 | 38.7000 |
| balanced_uniform3 | 48.4000 | 41.9000 | 24.1000 | 13.3000 | 26.1000 | 26.4000 | 54.9000 | 30.3000 | 42.5000 | 49.3000 |
| balanced_dense | 48.1000 | 42.2000 | 23.9000 | 14.2000 | 26.6000 | 26.1000 | 54.8000 | 30.3000 | 42.2000 | 49.5000 |

ce_uniform3：实际K=[47, 23, 10, 25, 6, 7, 45, 21, 33, 69, 25, 17, 14, 23, 34, 5, 43, 58, 53, 38]；梯度次数=11920，optimizer未单独记录（静态每batch一次step）；每client每轮上行=54067B（actual serialized uniform3 payload）；EF记录max=1.86e-09，局部核对容差=5e-06。计数零项=83；独立FP32 shift最大差=0.0。
首包独立解析=54067B=64+2675+51328；535块，tail170，3bit，136874个有限FP32坐标。

fedlc_uniform3：实际K=[47, 23, 10, 25, 6, 7, 45, 21, 33, 69, 25, 17, 14, 23, 34, 5, 43, 58, 53, 38]；梯度次数=11920，optimizer未单独记录（静态每batch一次step）；每client每轮上行=54067B（actual serialized uniform3 payload）；EF记录max=1.86e-09，局部核对容差=5e-06。计数零项=83；独立FP32 shift最大差=0.0。
首包独立解析=54067B=64+2675+51328；535块，tail170，3bit，136874个有限FP32坐标。

balanced_uniform3：实际K=[47, 23, 10, 25, 6, 7, 45, 21, 33, 69, 25, 17, 14, 23, 34, 5, 43, 58, 53, 38]；梯度次数=11920，optimizer未单独记录（静态每batch一次step）；每client每轮上行=54067B（actual serialized uniform3 payload）；EF记录max=4.66e-10，局部核对容差=5e-06。计数零项=83；独立FP32 shift最大差=0.0。
首包独立解析=54067B=64+2675+51328；535块，tail170，3bit，136874个有限FP32坐标。

balanced_dense：实际K=[47, 23, 10, 25, 6, 7, 45, 21, 33, 69, 25, 17, 14, 23, 34, 5, 43, 58, 53, 38]；梯度次数=11920，optimizer未单独记录（静态每batch一次step）；每client每轮上行=547560B（analytical dense FP32 delta+64B）；EF记录max=0，局部核对容差=5e-06。计数零项=83；独立FP32 shift最大差=0.0。

每client完整TRAIN计数及期望FP32 shift矩阵保存于JSON，逐轮逐client与metadata一致。

只读静态核对（不是脚本运行输入）：
- run.py:27-30,40-41: local counts come from client training ds.labels[ds.indices]; CE shift=0, FedLC-inspired shift=-max(n,1)^(-1/4), balanced shift=+log(n+1), all cast to FP32. Balanced is add-one, not max(n,1).
- run.py:42-50,78: batch64 with no dropped tail; one backward/SGD step per actual batch, common seed=20261201+100*r+i; no calibration grids/extra backward.
- run.py:63,88; separately read run_followup.py:120-135: evaluation uses raw model(x), CE(logits,y), argmax(logits) and dataset labels for metrics; no inference count/logit shift.
- run.py:79-87: quantized delta+old residual is serialized with frozen uniform3, decoded, compared with encoder sent, then aggregated with fixed sample weights. EF=u-decoded; dense sends delta and resets residual. Only first wire is retained.
- run.py:89-92: complete metric/client history and residuals stored in checkpoint and result; all downloads and dense uploads are analytical. Resume checks signature only, not independent state-shape validation; no fresh-run blocker was found.
- PLAN.md:26-28: all four arms 20 rounds; round16-20 balanced_uniform3 must have lower CE and accuracy >= each CE/FedLC baseline +.005, and accuracy >= balanced_dense -.01. Failure stops this fixed recipe without rescuing its smoothing/temperature/bits/local epochs.
- Separately read run_partitions.py:145: each client's train cutoff is round(.75*client_total); exact aggregate train total need not be 37500. This run has 37502 train / 12498 holdout, a disclosed numerical deviation from PLAN, shared by all four arms.

证据边界：
- This is one development training seed20261201 and partition20261193 on CIFAR10 official TRAIN only, fixed 20 rounds/SGD.03/epoch1. Rounds are not independent seeds and the new partition is not a new dataset.
- Actual train/holdout totals are 37502/12498 due to each-client round(.75*n); PLAN's 37500/12500 is numerically inaccurate. All arms share actual IDs/counts and sample weights. This is disclosed as a protocol warning, not silently treated as exact PLAN replication.
- Counts are recomputed for consistency from saved TRAIN-count metadata, not from original labels/data. Dataset/official TEST is never loaded by this checker.
- Raw-logit inference, no official TEST access, shared initialization/batch streams and MPS training are supported by metadata/static source. This checker does not independently reproduce the MPS training trajectory or trace filesystem accesses.
- Loss shifts are independently reconstructed and compared bitwise at FP32; +1 balanced loss and FedLC-inspired loss are explicit variants, not official full-method reproductions or new atomic mechanisms.
- Only first saved packet per quantized arm is independently decoded. Historical payloads, local raw deltas and pre-update residuals are absent, so EF bounds are recorded diagnostics, not independently replayed conservation or a contraction guarantee. The 5e-6 local arithmetic tolerance is not a changed performance criterion.
- Checkpoint/result equality checks saved metrics. Last5/class/worst values are arithmetic recomputations; no predictions/CE are reevaluated from data.
- Quantized uploads are actual wire lengths; dense uploads and all downlinks are analytical. No real network/authentication/energy/blockchain performance is tested.
- Failure applies to this fixed add-one balanced recipe and does not invalidate mechanism combination research or the FedLC-inspired low-bit reference. No new-algorithm or all-baselines superiority claim is supported.
- No external cross-model reviewer is available: REVIEW_UNAVAILABLE.

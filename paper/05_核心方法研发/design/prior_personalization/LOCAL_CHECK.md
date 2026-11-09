# 解析先验个性化：独立NumPy数组核验

状态 **DONE**；数组/产物核验 **PASS（1237/1237）**；方案数字一致性 **WARN**；冻结决定 **STOP_THIS_FIXED_PERSONALIZATION_RECIPE**；外部审查 `REVIEW_UNAVAILABLE`。

这是最终轮20的PFL客户端分布开发诊断，不是global IID指标或独立test。实际TRAIN/holdout=37502/12498，原PLAN和result的37500/12500另列WARN；12498包含此前globaldev全部2000样本。未改旧PLAN。

| 模型 | 分数 | macro acc% | macro CE | weighted acc% | weighted CE | client q10 acc% |
|---|---|---:|---:|---:|---:|---:|
| ce_uniform3 | raw | 38.985477 | 1.6691653545 | 37.333973 | 1.7310223439 | 28.978391 |
| ce_uniform3 | analytic_prior | 76.983238 | 0.6031308886 | 74.739958 | 0.6540509645 | 68.263593 |
| ce_uniform3 | prior_only | 61.276766 | 0.9069658069 | 57.241159 | 0.9717330279 | 40.438711 |
| ce_uniform3 | learned_head | 77.266003 | 0.6485954687 | 75.620099 | 0.6732647902 | 68.171181 |
| ce_uniform3 | learned_head_plus_prior | 75.974488 | 0.7088092388 | 73.883821 | 0.7622997517 | 66.096725 |
| fedlc_uniform3 | raw | 39.647659 | 1.6655696148 | 38.086094 | 1.7210989890 | 29.728284 |
| fedlc_uniform3 | analytic_prior | 77.341794 | 0.5986492295 | 75.292047 | 0.6479489076 | 69.081726 |
| fedlc_uniform3 | prior_only | 61.276766 | 0.9069658069 | 57.241159 | 0.9717330279 | 40.438711 |
| fedlc_uniform3 | learned_head | 77.238083 | 0.6531907478 | 75.748120 | 0.6754306609 | 68.572948 |
| fedlc_uniform3 | learned_head_plus_prior | 76.027732 | 0.7068761649 | 73.995839 | 0.7607568630 | 66.109071 |
| balanced_uniform3 | raw | 40.675262 | 1.6727016217 | 38.462154 | 1.7202890254 | 31.277656 |
| balanced_uniform3 | analytic_prior | 77.497398 | 0.5948502001 | 75.796127 | 0.6392232771 | 69.215491 |
| balanced_uniform3 | prior_only | 61.276766 | 0.9069658069 | 57.241159 | 0.9717330279 | 40.438711 |
| balanced_uniform3 | learned_head | 77.100415 | 0.6981505250 | 75.948152 | 0.7050938431 | 69.222650 |
| balanced_uniform3 | learned_head_plus_prior | 76.238280 | 0.7007258427 | 74.243879 | 0.7571174962 | 67.448526 |
| balanced_dense | raw | 40.671762 | 1.6708980186 | 38.582173 | 1.7177126670 | 31.113555 |
| balanced_dense | analytic_prior | 77.598262 | 0.5933992030 | 75.908145 | 0.6377409658 | 69.502726 |
| balanced_dense | prior_only | 61.276766 | 0.9069658069 | 57.241159 | 0.9717330279 | 40.438711 |
| balanced_dense | learned_head | 77.213722 | 0.6964046070 | 76.012162 | 0.7034099337 | 69.141641 |
| balanced_dense | learned_head_plus_prior | 76.337969 | 0.6993140208 | 74.331893 | 0.7557028628 | 68.217019 |

固定门槛和复算：
- 相对ce_uniform3+prior：macro CE降低0.0082806885，macro accuracy差0.514161pp；gate={'ce': True, 'accuracy': True}。
- 相对fedlc_uniform3+prior：macro CE降低0.0037990294，macro accuracy差0.155604pp；gate={'ce': True, 'accuracy': False}。
- best learned低字节对照=ce_uniform3/learned_head，macro accuracy=77.266003%；候选差0.231396pp；不低超过.5pp条件=True。
- 量化减dense：macro accuracy=-0.100863pp、macro CE=0.0014509971，只诊断压缩代价。

400条client score指标、100项summary全部复算；最大CE误差client=3.54e-07、summary=3.15e-08（核对容差5e-06）；accuracy/q10最大误差=1.11e-16；FP32 prior shift最大误差=0.0。CE用独立FP64 logsumexp；score加法保留原FP32。

宏平均每client等权；weighted用真实holdout样本数；q10由排序后的第1.9位置线性插值。JSON保留逐client正确分子、CE总和、真实分母、prior/count和完整summary。
- ce_uniform3：额外head适配记录596次梯度；据代码/实际batch数推导featureforward 596批/37502个TRAIN样本；raw+learned holdout推理116批/24996个样本forward。
- fedlc_uniform3：额外head适配记录596次梯度；据代码/实际batch数推导featureforward 596批/37502个TRAIN样本；raw+learned holdout推理116批/24996个样本forward。
- balanced_uniform3：额外head适配记录596次梯度；据代码/实际batch数推导featureforward 596批/37502个TRAIN样本；raw+learned holdout推理116批/24996个样本forward。
- balanced_dense：额外head适配记录596次梯度；据代码/实际batch数推导featureforward 596批/37502个TRAIN样本；raw+learned holdout推理116批/24996个样本forward。

4模型合计head适配2384次梯度与featureforward 2384批/150008样本；两个模型holdout推理464批/99984样本。解析prior额外梯度0；不能把learned head适配说成零计算。forward为代码与批大小推导，不是硬件FLOPs测量。

只读源码证据：
- probe.py:31,39-44: loads official TRAIN only; prior counts are bincount(labels[client TRAIN IDs]), checked against frozen training manifest, then shift=log((n_ic+1)/(n_i+10)). Holdout labels do not fit the prior.
- probe.py:45-54: head baseline freezes all parameters then unfreezes only final Linear, runs actual CPU forward/backward/SGD .03 for one TRAIN epoch, batch64. The frozen feature extractor is still forwarded on every TRAIN batch.
- probe.py:55-63: raw and adapted models both evaluate every holdout input, then five score arrays use the same holdout GT. Learned head is also offered the same analytic prior.
- probe.py:67-82: macro means give each of 20 clients equal weight; weighted means use actual holdout n_i/total; q10 is linear 10th percentile. Learned comparator takes max accuracy among learned_head/learned_head_plus_prior across three low-byte models; dense remains a diagnostic.
- probe.py:83-89: model/config input hashes are checked and source/PLAN identified; original diagnostic status is DONE. This pure NumPy checker independently recomputes saved-array scores, not the trained heads or frozen logits.
- PLAN.md:6-9 states 37500/12500; actual shared split is 37502/12498 due to per-client round(.75*n). Both original result personal_eval strings and PLAN retain 12500 and need this explicit warning.

证据边界：
- This recomputes all400 client score metrics and100 summary metrics from80 saved NPZ using NumPy FP64 stable CE, preserving FP32 score additions. It does not reevaluate images/model features, rerun adaptation or reproduce FL.
- Checkpoint bytes are hash checked only; pure NumPy never deserializes torch checkpoints. Final20 provenance is supported by source result metadata, exact checkpoint hashes, and the earlier frozen global-run audit.
- Counts match saved TRAIN manifest, every frozen FL round, and reconcile with NPZ holdout GT counts to original group counts. Source computes them from TRAIN IDs. Original training labels are not in these NPZ, so this checker does not independently reconstruct TRAIN counts from original labels. Holdout GT/ID consistency is checked across all four models.
- 12498 clientholdout samples include all2000 previously used globaldev samples; same seed/partition and frozen final-round models. This is non-independent PFL development, not new global IID or official test evidence.
- 20 clients share each global training run and are not20 independent seeds; no seed-level significance or generalization claim is made.
- Analytic prior uses no extra gradient or feature forward beyond ordinary raw inference, but requires TRAIN counts and score addition. Learned-head comparator has actual extra head gradients and whole-body feature forwards; its plus-prior version is included.
- Gate is unchanged: balanced+prior must beat CE+prior and FedLC+prior by >=.5pp macro accuracy and lower macro CE, and stay within.5pp of best low-byte learned counterpart. Winning another comparator cannot rescue a failed gate.
- Source implementation uses learned comparator from three low-byte models; dense is diagnostic. Even including dense learned heads would not change best comparator or this STOP result.
- No official TEST, original dataset load, torch/model import, new FL/head training or source/PLAN change in this checker. External cross-model review is unavailable: REVIEW_UNAVAILABLE.

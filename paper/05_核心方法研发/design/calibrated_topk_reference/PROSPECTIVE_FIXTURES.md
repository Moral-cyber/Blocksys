# Prospective TopK wrapper交付

2026-11-04。**CPU synthetic toy PASS，没有正式TRAIN加载/训练或官方TEST字节。** 原TopK五源及PLAN全部保持原hash；原confirm、codec、local loss和bias/head helper未修改。外部`REVIEW_UNAVAILABLE`。旧STOP不改变，本入口仅强压缩参考，没有新GO。

外部协议要求`work/predictive_pfl_confirm/PLAN.md`的完整SHA256为：

`703d5956c9edd24d5a03501a75f678a43222d19bd3dd1ede515ed77545b0830b`

新CLI固定partition20261217、20轮，只允许训练seed20261231/32/33；`--protocol-plan`必填。它直接复用旧`run(args,...)`和strict `prepare(new_partition)`，不调用旧main的p20261203/seed20261211断言，也不先运行旧分区。

## 显式签名绑定

`prospective.py`复制被冻结run函数的globals字典，把一个新的common/accelerator context交给同一个函数code。原common及原train/evaluate globals均不变。新context的source identities增加两wrapper和外部PLAN；`runtime.prospective_reference`明确记录外部PLAN路径/完整SHA、新partition/seed、允许seed列表、formal rounds、旧默认协议及覆盖机制。这些字段随完整runtime/source进入原config签名，并受checkpoint完整config绑定。

Standalone `prospective_evaluate.py`同样绑定被冻结two-method generic run，不调用旧eval main；先检查每组签名配置的prospective runtime覆盖，再接受新分区完整final checkpoint。输出保留相同TRAIN/holdout数组格式及冻结prior/bias/head协议，`decision=REFERENCE_ONLY`、`gates=null`。默认不加载任何旧分区uniform参考；可在同新分区uniform评估完成后显式传入`--uniform-evaluation`。

## CPU toy证据

- 纯50000个合成标签按p20261217调用冻结分区函数两次，manifest一致；此合成标签次序为TRAIN37500/holdout12500，**不等于正式真实样本计数证明**。
- 12张随机uint8图像，2clients，三个允许seed各CE/FedLC两方法各2轮；toy实际rounds2和formal20明确分开记录。新p/seed、两wrapper及外部PLAN身份、runtime覆盖、完整config/checkpoint签名吻合；各seed同方法签名不同。
- 两方法共同初始化及`seed+100r+i`的真实batch seed/order一致。冻结TopK codec未变，首包均54808B、6843pairs。
- 错PLAN SHA、旧seed、旧partition、伪runtime覆盖均拒绝；伪runtime在任何评估forward和创建输出前拒绝。
- 新standalone evaluation成功接受新p/seed的完整toy，TRAIN-only bias forward16样本，保持原saved-array schema与REFERENCE_ONLY。
- 原common source/runtime函数对象、train的common/wire globals、evaluate的common globals身份均未变；全部源hash前后相同。

机器证据`PROSPECTIVE_FIXTURES.json`；临时toy输出`/var/folders/0t/lt7l05rn1zz3xrpbkj6ym_8m0000gn/T/prospective_topk_fixture_g1zaun8g`。

## 供root启动的新seed1命令

以下命令未在本交付中运行；直接使用新p20261217/seed20261231：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python -B /Users/wind/Documents/Codex/2026-11-02/zhe/work/calibrated_topk_reference/prospective.py --protocol-plan /Users/wind/Documents/Codex/2026-11-02/zhe/work/predictive_pfl_confirm/PLAN.md --seed 20261231 --partition-seed 20261217 --rounds 20 --backend mps --out /Users/wind/Documents/Codex/2026-11-02/zhe/work/predictive_pfl_confirm/topk_cifar_p20261217_s20261231
/opt/anaconda3/envs/Tinyproto/bin/python -B /Users/wind/Documents/Codex/2026-11-02/zhe/work/calibrated_topk_reference/prospective_evaluate.py --protocol-plan /Users/wind/Documents/Codex/2026-11-02/zhe/work/predictive_pfl_confirm/PLAN.md --model-root /Users/wind/Documents/Codex/2026-11-02/zhe/work/predictive_pfl_confirm/topk_cifar_p20261217_s20261231 --out /Users/wind/Documents/Codex/2026-11-02/zhe/work/predictive_pfl_confirm/evaluation_topk_cifar_p20261217_s20261231
```

| 新文件 | SHA256 |
|---|---|
| prospective.py | `af3e654d1c9f7710d10096d12419a92eedb3e962583a4d1c7774fd546d95f1d4` |
| prospective_evaluate.py | `83e97c50d770da661861f8e982164973883e2a45a186bbce602c97a7ca4e7bd3` |
| prospective_fixture_check.py | `a0b72ed40f1d65e058e89ea6b3fe7b48fb24e0ae15b113f50f54991bced91f45` |

正式预算仍是cap54808B、TopK实发54808B、uniform3实发54067B；64B真实header和每pair8B不变。没有调lambda、bit、loss或旧gate。

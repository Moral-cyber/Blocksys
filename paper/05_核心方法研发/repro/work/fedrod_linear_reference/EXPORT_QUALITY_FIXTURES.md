# FedRoD final概率指标导出：CPU toy交付

2026-11-04。**CPU toy PASS；未运行正式图像推理、TRAIN拟合或长训练，未读取官方图像/TEST。** 原reference/train/prospective/PLAN/SOURCE_LOCK均保持冻结身份；外部`REVIEW_UNAVAILABLE`。

## 新入口与正式边界

`export_quality.py`只接受prospective p20261217、允许seed20261231/32/33、完整final20 checkpoint与匹配的DONE history。它重建冻结prospective base config，核对config自签名、全部源、完整checkpoint config/签名、global及20个private P的布局/CPU FP32/有限性和真实样本权，然后才通过原strict adapter读取五个TRAIN文件并按config分区取holdout。官方TEST字节审计拒绝。

每client、每batch只计算一次共享feature，再得到G与P并计算FP32的G+P。导出NPZ包含`holdout_ids`、整型dataset GT `labels`、`generic_raw`、`personal_residual`、`generic_plus_local_residual`。global和20个head的state hash在推理前后核对。

复用已pin的`work/calibration_baselines/score_quality.py`，SHA`5f14f01a25f64c9ed495214aa57f4277370f315ce08b7686b985faa03666fc98`；在调用其内部cast前，显式拒绝非整数或错误shape GT。报告accuracy、自然对数NLL、十类Brier平方误差总和（不除类别数，[0,2]）、15固定等宽bin top-label ECE，保留全部bin、macro/weighted/pooled区别及accuracy q10。较低NLL不自动等于较低ECE/Brier；不新增gate。

成本包含本次实际feature/generic/residual forward图像数、批数与逐client/总耗时。TRAIN image forward、TRAIN fit、CNN backward、optimizer steps和额外上传均为0。此次导出是训练入口已有final评估之外的额外holdout forward，不能把这一成本隐藏或称为零计算。

正式模型完成后由root决定运行；本交付未运行以下命令：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python -B /Users/wind/Documents/Codex/2026-11-02/zhe/work/fedrod_linear_reference/export_quality.py --model-root /Users/wind/Documents/Codex/2026-11-02/zhe/work/fedrod_linear_reference/prospective_p20261217_s20261231 --out /Users/wind/Documents/Codex/2026-11-02/zhe/work/fedrod_linear_reference/quality_p20261217_s20261231
```

## 合成toy证据

构造70张随机uint8图像，40个合成TRAIN IDs仅用于签名/样本权，30张合成holdout图像分20client（10个各1张、10个各2张）。构造合法的synthetic final20历史及checkpoint，没有执行训练；20个private P均有非零随机weight和非零bias。

- 前向hook确认恰好30图像/20个feature batches，G和P共享该feature，没有为G+P重复计算body。
- 独立`F.linear`使用hook保存的feature及checkpoint head验证G、P、G+P均逐位一致，G+P确实不同于G；保存GT和原ID顺序精确一致。
- 不调用作者指标函数的logaddexp、显式onehot Brier和15区间mask ECE手算，最大client指标差`8.88e-16`；全部macro/weighted/pooled汇总最大差`4.44e-16`。
- uniform analytic fixture符合NLL=ln10、Brier=.9、ECE=.15；fractional GT在score内部cast之前拒绝。
- 不完整19轮、非有限private P、checkpoint错签名、config错签名均在任何feature forward前拒绝。
- global/private状态及全部受保护源hash前后相同；不存在的dummy TEST路径在open前拒绝，实际官方文件读取0。

机器证据`EXPORT_QUALITY_FIXTURES.json`；toy输出`/var/folders/0t/lt7l05rn1zz3xrpbkj6ym_8m0000gn/T/fedrod_quality_fixture_i78wo63p`。重算toy：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python -B /Users/wind/Documents/Codex/2026-11-02/zhe/work/fedrod_linear_reference/export_quality_fixture.py
```

冻结导出源SHA`294aa4356fb0b335c83dbf07bab2003c972708f5e129178208088cba332b1dae`；toy harness SHA`71f8f4a324816cd13c6fde39149192ce932d4f8efae193643d1611db6190ee2a`。

## TopK saved prior-vector适配可行性

当前prior_vector/evaluate_saved.py只在模型集合选择分支限制confirm-three/old-four，后续TRAIN-only vector fitting和七score计算对两个TopK method同样成立；TopK评估保存数组字段已兼容冻结read_saved。因此可另薄wrapper支持精确`ce_topk/fedlc_topk`两组，在独立globals/schema分支复用原run、明确输出`topk_two`并加入wrapper/原evaluator身份，不改通用evaluator源、solver、reader或数学，也不做新CNN。当前仅提出可行性，尚未实现此适配。

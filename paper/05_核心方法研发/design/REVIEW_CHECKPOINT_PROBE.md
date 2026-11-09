# Development checkpoint probe 独立复核

日期：2026-11-03。审查范围：`probe_checkpoint.py`、`checkpoint_probes/cpu_topk20`、`checkpoint_probes/cpu_align9020`，以及两者明确指向的 development 来源及执行依赖。未读取 confirmation 目录或官方测试集数据/结果，未启动完整训练。

本报告由另一子代理独立读取代码并复算。已参考 `experiment-audit/SKILL.md` 的证据边界，但本代理与执行者并非已核实的不同模型家族，正式跨模型审计为 **REVIEW_UNAVAILABLE**，不能称作该 skill 的完整跨模型验收。

## 结论

**同状态反事实实现与已有开发数据复算通过；来源记录存在可修补缺口；当前数据不支持把 align90 的方向机制写成学习性能优势。建议有限收尾，不启动大矩阵。**

两个 probe 是同一训练 seed、同一数据分区、两种训练轨迹第 20 轮终点上的第 21 轮单步干预，不是两个独立 seed，也不是九种方法各自完整训练的结果。两组来源状态不能视为独立重复。

复核脚本：`check_checkpoint_probe.py`。实际执行报告：`checkpoint_probe_check.json`。脚本只处理上述两个显式路径；每组重算全部 18 行 sparse 编码及 6 行关键验证结果，并各重算 client0 的一次本地更新。

## 已核实的实现和结果

| 检查项 | 结论与证据 |
|---|---|
| 开发范围 | `probe_checkpoint.py:32-40` 强制 `test_final == false`，限定 Clean 数据和普通本地目标的来源。`runner.prepare` 构造官方训练数据中的固定 local holdout 验证子集；probe 未调用官方测试工厂。NPZ 的局部 holdout 目录虽名为 `test`，这里不代表官方测试集。 |
| 真实来源状态 | `probe_checkpoint.py:58-67` 从来源 checkpoint 取模型、残差、`previous`。独立检查证实：20 轮连续编号，checkpoint rows 与来源 development result rows 相同；保存的 base、reference 和全部 20 个客户端 residuals 与 checkpoint 逐位一致。残差为真实非零状态。 |
| 共同干预起点 | `probe_checkpoint.py:70-83` 只生成一次各客户端 delta；全部候选复用这些 delta、同一 base、同一 reference 及 residual 模式。`105` 每次直接赋值 `base + sent`，没有继承上一候选的模型更新。独立重编码前后 deltas/residuals/reference 均未改变。 |
| 下一轮种子 | 原 runner 使用零起始 `rnd`（`run_followup.py:191-202`）。完成 20 轮后的下一步为 `seed + 20*100 + client_id`，probe `70-73,91-93` 正确。两组 client0 的 seed 均为 **20263113**；独立重训与保存 delta 逐位相同。这只是两次短本地更新检查，不是重新完成训练。 |
| 聚合权重 | `probe_checkpoint.py:68,83,99` 使用 `n_i / sum n_i`；独立加载数据验证所有 weights 与保存值相同，权重和为 1。不是客户端等权。 |
| 相同 K 与字节预算 | P=136874、fraction=.05、K=6843；全部 sparse 候选、两个 residual 模式、两种起点均为每客户端 `64 + 8*K`，20 客户端每轮 **1096160 bytes**。独立重编码验证每个客户端 K、`u = sent + residual` 守恒及字节相同。此为 FP32 值/uint32 索引/64-byte 头部的分析计算，未测网络、水印或账本开销。 |
| 误差分解 | `probe_checkpoint.py:100-112` 计算相对同一 `sum w_i u_i` 的平方误差及 `sum w_i^2 ||u_i-s_i||^2`，两者差为交叉项。两 probe 的全部 36 行 sparse 误差、范数及交叉项均精确复现。此量不是相对真实全局梯度的误差。 |
| 验证数据真实性 | `run_followup.py:77-101,120-135` 用真实 holdout 标签计算 accuracy/cross entropy；没有用模型输出当标签，也未按预测最大值归一化分数。两 probe 各 6 行 dense/TopK/align90 × actual/zero residual 验证指标逐字段复现，loss 差为 0。 |
| 文件身份 | 两组 source config、source checkpoint、shared state、probe 源文件的四项已记录 SHA256 全匹配；六项实际 method_followup 执行依赖 SHA256 当前匹配；独立重算来源 config 的 run_signature 通过。 |

## 来源记录的缺口与使用限制

1. **中等程度的 provenance 缺口。** `probe_checkpoint.py:133` 对 `method_followup/*.py` 广泛计算 hash，却没有记录实际定义 SmallCNN、load_client、vectorize、assign_vector 的 `tinyproto_diag/tools/fedavg_update_diagnostic.py`，也未记录实际 NPZ 的字节 hash。`run_followup.py:89-100` 的 manifest 只 hash 标签排列和验证索引，不会检测图像值改变。独立验证当前 data_manifest 匹配，当前外部 helper SHA256 为 `1307514260e8ec0e741324edc9d1b7ae34076f2e2de3cabadd8c8a11d6766496`，并成功复现 client0 delta 和 12 行验证结果；但不能据此声称所有历史输入的逐字节身份早已封存。

   建议将真实执行依赖、原始 NPZ/来源 config 身份、实际线程数和 torch/numpy/python 版本写入后续 probe 的明确 manifest。为已有 probe 增补审查记录即可；不要把当前补录 hash 冒充运行前已冻结的 hash，也无需因此重跑大规模训练。

2. **清单过宽产生非执行漂移。** `cpu_align9020` 记录的 `benchmark_codec.py`、`run_batch.py`、`summarize_confirmatory.py` 当前 hash 已改变；这三项均未被 probe 的执行路径调用，当前实际编码/训练依赖没有漂移。报告应区分这种编辑与真实依赖漂移；不能因它们变化就宣称 probe 计算已失效，也不能声称清单所有项完全未变化。

3. **输入校验可加强，但现有两组已独立通过。** probe 运行时只核对 DONE/行数，没有自己检查连续轮号、source result 与 checkpoint rows 相同、run_signature 或 config/result seed/variant 一致。独立复核对当前两组补做了这些检查。未来扩充入口可加强这些拒绝条件；不应修改冻结 runner/codec。

4. **无跨线程/后端等价结论。** 本次实际复算为 CPU、2 线程；MPS 或别的线程配置需要独立记录和核验。两 probe 的计时不是纯 codec 耗时。

## 开发结果对机制主张的影响

下表均为同一 checkpoint、同一 residual 模式下 **align90 相对 TopK-EF** 的一步验证差值；loss 越低越好。

| 第 20 轮来源轨迹 | residual 模式 | accuracy 差，pp | cross entropy 差 | 聚合相对重建误差差 |
|---|---|---:|---:|---:|
| TopK-EF | actual_EF | -0.550002 | +0.01806700 | +0.00356797 |
| align90 | actual_EF | -0.049999 | +0.01468363 | +0.00576099 |
| TopK-EF | zero_residual_control | -0.049999 | +0.00493571 | +0.03442299 |
| align90 | zero_residual_control | -0.050002 | +0.00280650 | +0.03345688 |

两个 actual_EF 起点中，align90 的交叉项较 TopK 略低，但独立误差能量增加，最终聚合误差和验证 loss 均更高。因此不能从“交叉项下降”单独推出聚合质量或学习性能改善。两个零残差控制也未显示 align90 的一步 loss 优势；它们改变了 EF 状态，不能作为训练到第 21 轮的独立无 EF 算法。

`dense_candidate_reference` 是 `sum_i w_i(delta_i + residual_i)`，是**压缩重建参照**。它含有历史累积残差，不是当前全局负梯度、最优更新或当前下降真值。实际 EF 中它相对 TopK 的 loss 反而高出约 .0622/.0624，accuracy 低约 2.55/2.60 pp。这说明“更接近 dense u”不能自动转述为“更有利于当前学习”。零残差 dense delta 也只是有限本地 SGD 后的聚合 delta，不能称为全局梯度真值。

previous-reference 与 actual_EF dense candidate 的 cosine 仅约 .0545/.00974；清零残差后约 .367/.366。这支持“累积残差改变了候选与旧全局方向的关系”的有限观察，不足以证明异质性引起了机制、方向校准已经解决了问题或旧方向永远无效。

这两次局部干预会削弱当前机制解释，但无法单凭一步结果证明完整训练永远不可能获益。能量下界、固定 core 下的 alignment 性质仍是编码性质，不能写成经实验证明的准确率、收敛或异质性适应优势。

## 止损与有限确认建议

父代理另报告两组 40 轮 CPU 配对确认的 last5 验证 accuracy 优势约 +.20/+.35 pp，loss 约 -.0023/-.0036。本代理**未读取或独立核验这些确认结果**；这里只将该摘要作为是否继续投入的上下文，不把它纳入上表或来源核验结论。

建议停止 tau × fraction × mixing × dataset × partition × backend 的探索矩阵，也不再从 sign50、align99 或其他单步偶然较高 accuracy 中挑选新赢家。已见结果的增益小，机制支持不足，继续扩大搜索容易把方法设计变成验证集筛选。

如果原先已明确冻结 TopK-EF vs align90 的 3-seed 配对确认，可只完成尚缺的第三个 seed pair，其他设置、tau、预算、主终点保持不变。已经在运行的有限队列可收尾；不要因某个 seed 的结果临时增加配置。若第三对尚未计划冻结，则先写明有限范围与最大运行数，再启动，不追求某个分数出现后才停止。

确认的报告单位是训练 seed，last5 rounds 是每个 run 的一个汇总值，不能把轮数、客户端或验证样本当作额外训练重复。3 seeds 的区间可能很宽，应如实报告。保留官方测试封存，先完成有限设计/结论冻结，再由父代理按既定流程一次性读取最终测试；本审核不参与读取。

若有限确认仍只有微小、跨 seed 不稳定或不胜合理对照的增益，就应停止把当前方向选择器包装成强 AI 核心贡献。此时应基于已得到的负结果重新确定一个可证伪的具体机制问题，而不是再扩大当前选择器网格。现有水印/账本流程可以保留，但无法替代前半部分算法的独立价值证据。

## 文件版本

- 本次审查的 `probe_checkpoint.py` SHA256：`7d448dd031ef6cf7a0f0a3dd3d721782353618a886f97b10adcabacbbc1afdc0`。
- 来源 runner SHA256：`4066929ff317e486576b17e7d347b3437b17ce019f1425a18817c37f8406e18d`。
- 来源 guarded codec SHA256：`3dcb6636f12d6dc17fe023a978a9016ce10593ed3da5472b1ad498a33b8c948e`。
- 以上冻结 runner/codec 均未修改。

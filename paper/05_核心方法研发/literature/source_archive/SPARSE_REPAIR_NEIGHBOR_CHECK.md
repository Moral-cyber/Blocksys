# 固定支持的一步修复：近邻与脚本静态核查

日期：2026-11-03 UTC。范围：最多三批公开一般检索，核对明显机制近邻，并只读核查 `work/method_followup/probe_sparse_repair.py`。本轮未运行训练、梯度计算或 fixture，未读取确认实验/test 文件；执行 agent 的诊断结果由 parent 统一处理，不在本报告重复当作独立复验。

## 结论与停止边界

**支持内修复的宽机制创新风险高。** CrAM 已明确包含“先取得压缩模型点，在该点计算梯度，并投影回同一 Top-K 支持”的操作，其附录也比较了去掉额外梯度扰动、直接在压缩点计算梯度的变体。因此，“压缩后看当前损失、固定支持调整数值”不能独立作为原创核心。

在限定的三批检索和下列已读正文中，没有核实与本地本轮探针的**整条链**完全相同的先例：对 FL 模型差分加 EF 残差做 TopK，在重建端点作一步同支持修复，再将修复的相反量记入 EF。这里的“未核实”只描述证据范围，不能转为“没有前人”或“已确认创新”。不继续扩展这一 family，也不提出新方向。

脚本静态核查未发现破坏 A/B/C 定义、同 batch 计算匹配、支持约束或 EF 记账的明显错误。其裁剪界是**压缩误差的收缩界**，不是最终上传向量的保范数界。一次开发反事实不能证明闭环方法收益；parent 已要求按固定诊断的 STOP 结果收尾，本核查不触发任何后续计算或调参。

## 明显近邻：核对具体操作而非仅看动机

### CrAM：核心操作的直接先例，整链有差异

Alexandra Peste, Adrian Vladu, Eldar Kurtic, Christoph H. Lampert, Dan Alistarh. *CrAM: A Compression-Aware Minimizer*. 本轮读取 arXiv:2207.14200v4（2023-05-04），官方正文：<https://arxiv.org/html/2207.14200>。

Eq.2 与 [Algorithm 1](https://arxiv.org/html/2207.14200#alg1) 先构造

\[
\widetilde\theta_t=C(\theta_t+\rho\nabla L(\theta_t)),
\]

当 C 是 Top-K，Algorithm 1 lines 9–10 令 M 为同一个 Top-K 支持的投影，并使用

\[
\widetilde g_t=M\nabla L(\widetilde\theta_t).
\]

随后用这个梯度更新原模型；CrAM+ 还加原 dense 模型的梯度。[Appendix C.1 Eq.12–13](https://arxiv.org/html/2207.14200#A3.E12) 明确比较去掉 extra-gradient 后，在 \(C(\theta_t)\) 上计算梯度的 Top-K 变体，以及加入压缩支持梯度的 Top-K+ 变体。

这不是仅有“压缩感知”的相同动机，而是**在压缩后的模型点求梯度、对梯度使用该压缩支持**的具体操作已公开。与本地探针的区别也要保持准确：CrAM 面向模型权重压缩和训练鲁棒性；其更新针对原 dense 模型，并未给出这里的 FL 上传差分＋旧 EF 残差＋反向修复债务这一完整协议。不能把两者称为完全等价；也不能忽略这项核心近邻后把支持内一步修复本身称为新。

作者论文给出代码 <https://github.com/IST-DASLab/CrAM>，本轮未下载或执行该代码，结论以官方全文公式为据。保存材料：`sparse_repair/cram.html`、`cram.txt`，URL、UTC 时间和 hash 见 `primary_log.json`。

### 两个容易误认的检索命中：已读正文后排除整链等价

- James O’Neill, Sourav Dutta. *Gradient Sparsification For Masked Fine-Tuning of Transformers*. arXiv:2307.10098v1（2023-07-19），<https://arxiv.org/html/2307.10098>。其 §III GradDrop 是训练过程中的随机梯度 mask，以正则化语言模型微调；不是先压缩 EF 上传差分，再在重建点固定该支持修复数值。它支持“稀疏支持内微调并不新”的一般风险，但不是已确认的本轮完整协议。
- Lucas Grativol, Mathieu Léonardon, Guillaume Muller, Virginie Fresse, Matthieu Arzel. *Federated learning compression designed for lightweight communications*. arXiv:2310.14693v1（2023-10-23），<https://arxiv.org/html/2310.14693>。§III 是在双方发送前对模型权重做非结构化幅值剪枝；没有这里的压缩后梯度修复和 EF 反向记账。标题/摘要中的 compression 不能作为整链重合证据。

上述原始正文均已保存。既有 FLARE、SAEF、SA-PEF 的训练前残差处理证据见 `RESIDUAL_FRESHNESS_NEIGHBORS.md`；本轮不再复述为新的直接先例。

## 脚本静态核查：定义、反向债务和收缩口径

所读脚本 SHA-256：`2ad6ad8d7c4a7856cc7c31628d17e11a90076ede1acbcfe246bd59f80708ade8`。

依赖快照：`guarded_codec.py` 为 `3dcb6636f12d6dc17fe023a978a9016ce10593ed3da5472b1ad498a33b8c948e`；`run_followup.py` 为 `4066929ff317e486576b17e7d347b3437b17ce019f1425a18817c37f8406e18d`。这些是本轮静态读取时的 hash，不代替执行 agent 自己记录的运行依赖 hash。

### A/B/C 与计算口径

脚本 lines 214–234：

- A 对保存的 \(\delta+r\) 用普通 TopK-EF。`encode` 默认 `method="topk_ef"`，因此这里没有意外启用旧方向选择器。
- B 在 \(w+\delta\) 上对固定 TRAIN batch 计算一次梯度，用 \(\delta_B=\delta-\alpha g_B\) 后再 TopK-EF。它是额外一步普通 local SGD 的增量坐标表达；FP32 中不必与先更新逐参数模型再减 base 完全逐位相同，脚本 line 238 记录该舍入差。α=0 保留保存的 delta。
- C 在 \(w+s_0\) 上对同一 batch 计算一次梯度，只在 A 的 indices 上形成修复 z；不重选支持。C 对应每个客户端单独重建的 shadow 点，不是加权全局聚合端点。

同一客户端 B/C 使用相同 `ids` 与相同 batch tensors（lines 215–223），各恰好一次 `gradient_at`。其内部使用私有模型副本、平均 CE，检查共享模型向量未变（lines 50–65）；没有第二条 shadow 本地训练轨迹。既定 SmallCNN 无 dropout/BatchNorm，故这里 `eval()` 不改变其训练层行为。所有客户端按样本量权重聚合（lines 203、246）；验证数据只用于最终 metric，没有进入梯度函数。

### EF 反向债务确实存在，脚本没有漏记

设 \(u=\delta+r\)，\(S=\operatorname{supp}(s_0)\)，\(s_0=P_Su\)，修复 z 只在 S 上非零。脚本 lines 78–102 执行

\[
s=s_0+z,\qquad r'=u-s=(u-s_0)-z.
\]

所以在 S 上有 \(r'_S=-z_S\)：修复不是免费加入未来训练的额外更新，而会在后续 EF 中以相反符号形成待偿债务。初始未上传向量 \(u-s_0\) 在 \(S^c\)，故精确算术下

\[
\|r'\|^2=\|u-s_0\|^2+\|z\|^2.
\]

因此修复会增加即时残差能量；后续可被抵消。脚本的 limitations 已写明这一点。`residual=u-sent`、守恒容差检查及保存的 C residual 与这个定义相符。不能把 C 的单轮验证变化直接当成长期增益。

### 裁剪证明压缩误差收缩，不能称作同样的发送能量保护

脚本将 \(\|z\|^2\) 裁剪到 \((1-\theta)\|s_0\|^2\) 内（lines 81–87），因此

\[
\|u-s\|^2
\le \|u\|^2-\theta\|s_0\|^2
\le (1-\theta K/N)\|u\|^2.
\]

第二步用的是普通 TopK 的 \(\|s_0\|^2\ge(K/N)\|u\|^2\)。代码 lines 94–98 的两个 bound 与此一致，并显式容许 FP32 舍入。该推导在 z 依赖 TRAIN 梯度时仍是逐样本几何界；它不等于收敛证明。

尤其不能把它改写成 \(\|s\|^2\ge\theta\|s_0\|^2\)。若 z 与 s0 反向，发送范数可以下降到 \((1-\sqrt{1-\theta})^2\|s_0\|^2\)，低于 θ 比例。脚本记录的是 `topk_retained_energy` 与 `new_residual_energy`，本身没有作该错误改写；对外应称“误差收缩预算”。

### 通信和范围

脚本 line 110 起序列化原 selected indices 与修复后的 FP32 数值；所选坐标即使修复成零也保留 pair，保持 K 对和 `64+8K` 字节（lines 240–245）。因此“同支持”应理解为 payload indices 相同，实际非零 support 可以因抵消缩小。收缩推导只要求 z 不超出 S，不要求每个选中数值非零。

`locks` 中名为 `test` 的 NPZ 路径按工程声明是 official-TRAIN 内的本地 holdout；此函数只做 hash 锁绑定。脚本禁止官方 test 已访问的来源状态，并仅把 `prepare` 返回的 validation 用于 metric。该静态判断不代替运行时数据锁证据。没有修改本脚本或任何冻结实验代码。

## 已完成的三批公开检索与保存记录

仅使用 OpenAlex 公开 search endpoint，每个查询筛前 15 条。三个 batch、九个公开一般查询已全部结束，不再扩展。返回总量不是检查过的论文数量，也不证明搜索完整性。

| 批次 | 一般公开查询 | 返回总量 |
|---|---|---:|
| 1 | `post compression fine tuning sparse federated learning updates` | 2483 |
| 1 | `support restricted gradient correction sparse federated learning` | 1724 |
| 1 | `compression aware error feedback sparse update optimization` | 11817 |
| 2 | `"gradient sparsification" "fine tuning"` | 350 |
| 2 | `"sparse updates" "error feedback" correction` | 22 |
| 2 | `"federated learning" "compression-aware"` | 104 |
| 3 | `"gradient compression" "fine-tuning" "correction"` | 260 |
| 3 | `"sparsification" "refitting" "federated"` | 15 |
| 3 | `"post-compression" "gradient"` | 1137，以光学后压缩为主，查询噪声高 |

全部响应为 HTTP200。原始结果为 `work/literature_followup/sparse_repair/batch{1,2,3}_openalex_{0,1,2}.json`，每批 `batch*_log.json` 保存 query、URL、UTC 时间、原始路径和响应状态。一手正文请求与 hash 在 `primary_log.json`。未向外部发送本地公式、裁剪规则或研究代码。

最终口径：**宽机制已有 CrAM 的强近邻，完整 FL-EF 协议等价未被本轮核实；脚本静态无明显定义/记账错误，保能应准确称误差收缩；按 parent 的固定 STOP 决定归档，不继续训练、调参或搜索。**

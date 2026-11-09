# FedProx / RegTopK 基线代码独立检查

日期：2026-11-03。范围：当前 `run_controls.py`、`run_followup.py`、`guarded_codec.py`、`check_controls.py`，并核对 RegTopK 原文 Algorithm 2 与 `work/literature_followup/FOLLOWUP_NEIGHBORS_BASELINES.md`。补读导入的模型、数据加载和 clean split 生成工具以核对数据流。

**这是同模型家族的独立代码检查，含一位独立子代理的 RegTopK 逐坐标 oracle 复核；不是 `experiment-audit` 的跨模型审计。没有调用该技能，没有训练完整实验，也没有验证基线的学习性能。** 检查没有改动四份被审源码、现有配置、checkpoint 或结果文件。现有 `check_controls.py` 在临时目录输出检查报告后重新执行；临时目录已自动删除，未覆盖原 `controls_check.json`。

## 结论

在当前 **CPU SmallCNN、20 个客户端固定全参与、固定样本权重、服务器直接相加更新、无服务器学习率或动量** 的设定中，未发现会制造主方法准确率优势的 FedProx / RegTopK 核心实现 bug。可以继续当前冻结版本的基线训练。

必须保留四项限定：

1. RegTopK 是按论文 Algorithm 2 适配到多步本地模型差分的通信选择规则，不能称原论文 distributed gradient-SGD 完整复现，也不能称逐行复现作者公开程序。
2. RegTopK 的客户端身份来自调用次数，当前 runner 满足其假设；未来部分参与、跳过或重排客户端会使历史错配。
3. 当前 RegTopK 实现保存完整上一轮候选向量，实际缓存多于理论最低的 K 对；不能用最低理论状态成本描述当前实测内存。
4. 配置签名锁定了 adapter / runner / codec，尚未完整锁定导入工具、图像内容和运行库。现有 manifest 足以比较当前同一数据副本上的设置，不能作为完整数据溯源或无泄漏证明。

## 审查对象与冻结身份

| 文件 | SHA-256 |
|---|---|
| `run_controls.py` | `0ec1485702bcfe8293f90e99f46bdb9f363cad43b7f457a95e342260b4b0d881` |
| `run_followup.py` | `4066929ff317e486576b17e7d347b3437b17ce019f1425a18817c37f8406e18d` |
| `guarded_codec.py` | `3dcb6636f12d6dc17fe023a978a9016ce10593ed3da5472b1ad498a33b8c948e` |
| `check_controls.py` | `bacf931869a6593c17abec7a936c0aae804af05e6acd0cf513ae7c26d3d98bf3` |

检查前后四份 SHA-256 相同。新执行检查的解释器为 `/opt/anaconda3/envs/Tinyproto/bin/python`，PyTorch `2.13.0`、NumPy `2.5.3`。

补读依赖在本轮的身份：

| 文件 | SHA-256 |
|---|---|
| `work/tinyproto_diag/tools/fedavg_update_diagnostic.py` | `1307514260e8ec0e741324edc9d1b7ae34076f2e2de3cabadd8c8a11d6766496` |
| `work/tinyproto_diag/tools/generate_clean_cifar10_split.py` | `2d68abe9f7151f21cca8e43aa11392c9e5b4f3be0b36a421cc6c85e54ba4e20c` |
| `work/tinyproto_diag/tools/sparse_update_benchmark.py` | `4bb2ecbc9b5ad6353497ac75f70557796e56c53b64a6acdf701045168b2c813a` |

这三个补充哈希由审查记录，尚未写入原运行配置。`regularized_topk.txt` 的 SHA-256 为 `2efc371b1b9b97731b78ae8a87b14ba4cff36c5477867f27e0989c576bd8fa42`。

## FedProx 目标与训练顺序

**核对通过。** `run_controls.py:79–105` 实现每个小批次的

\[
\nabla\ell_i(w;b)+\mu_p(w-w_t),
\]

对应 `CE + mu_p/2 * ||w-w_t||²`。每个客户端先深拷贝同一全局模型，固定全局锚点；每个 batch 清梯度、反向传播 CE、加入近端梯度、再执行 SGD step。没有先压缩后训练、先更新服务器再训练下一个客户端、或把残差放进近端锚点的顺序问题。

`run_followup.py:198–210` 在全部客户端结束后才修改全局模型。因此一轮内所有客户端从同一模型开始，训练后产生 `local-global` 差分，再经过 Top-K EF。`prox001_ef` / `prox01_ef` 的近端系数分别为 `.01` / `.1`；编码路径在 `run_controls.py:210–212` 直接调用原 `topk_ef`，没有额外衰减上传值或更改聚合权重。

`mu_p=0` 直接调用原训练函数，避免副本逻辑偏差。当前 SmallCNN 无 BatchNorm、Dropout 或额外模型状态，向量化参数与模型更新一致。验证函数会将全局模型切换到 eval，但两个本地训练函数都显式 `local.train()`，不会让后续本地训练误留在 eval。

新执行的小检查：

- 原 `check_controls.py` 中 `.01` / `.1` 与显式近端 loss 在三个 mini-batch 上的梯度更新比较通过。
- 独立 fixture 确认单 batch 的第一步近端梯度为零，上传差分与普通 SGD 完全相同；跨多个 batch/epoch 后差分发生变化。
- 全局锚点参数及传入模型的 eval 状态在训练后保持不变。

## RegTopK 公式、状态与权重

**按论文 Algorithm 2 核对通过。** 原文文本 `regularized_topk.txt:779–800` 明确：上一轮全局聚合和上一轮本地累积值在分子，当前本地累积值在分母；先生成选择 mask，再上传原累积值。

`run_controls.py:162–204` 对每个客户端计算 `u = delta + residual`，仅在该客户端上一轮支持集上使用

\[
d_j=\frac{G_{t-1,j}-\omega_i u_{i,j}^{t-1}}{\omega_i u_{i,j}^{t}},\qquad
q_j=|u_{i,j}^{t}|\tanh(|1+d_j|/\mu_r).
\]

上一轮未选中的坐标保留 `|u|` 分数，相当于 Algorithm 2 的 Q→∞ / 因子 1。上一轮支持集与候选值分客户端保存；当前轮为零的坐标直接置零分数，避免除零。首轮使用标准 Top-K。即使上一轮全局聚合恰好为零，有历史时也继续执行 cancellation 规则，没有错误退回 Top-K。

实现使用 `tanh(abs(1+d)/mu_r)`，不是 `1-tanh`、额外负号或 `tanh(abs(d)/mu_r)`。先按非负 score 选 K，再由 `_support_encoding` 发送原 FP32 `u`；没有发送 score 或乘过 tanh 的数值。未发送部分进入 EF。

`run_controls.py:40–42` 的 `mu_r=1,5,10` 均为正，且与 FedProx 的 `mu_p` 分开记录。参数 5 是文献补查中作者实现的预设来源；参数是否最优仍需本工程验证集决定，不能由本代码检查推断。

权重来自 `len(ds)/total`（`run_controls.py:274–279`），与 `run_followup.py:158–159,205` 的聚合权重一致。`previous` 在每轮结束保存**实际稀疏聚合更新**（`run_followup.py:227`），不是本轮 dense 反事实，也不是无权均值。模型差分相对于梯度的整体负号会同时翻转 `G` 与 `u`，不改变 distortion 和支持选择。

独立子代理使用逐坐标标量 Algorithm-2 oracle，在不借用 adapter 公式向量化实现的情况下检查了：`mu_r=1/5/10`、客户端权重 `.2/.8`、三轮变化更新、当前零值、`K=3,N=6`、`K=N=6` 和整体符号翻转。支持、原值上传、残差、client index、weight 和字节数均一致。

### 当前状态机制的两项限制

- `run_controls.py:174` 用 `call_count % n_clients` 确定客户端。当前 `run_followup.py:198–202` 固定按客户端顺序、全参与、每客户端每轮编码一次，因此合法。未来加入部分参与、异步、重试或重排时必须显式传 client id；否则可错误使用其他客户端历史。此项不是当前训练的 bug。
- `run_controls.py:202` 保存完整 `candidate.clone()`，即每客户端 N 个 FP32 值及 K 个 int64 索引；仅在 K 个位置读取旧值，但未只存 K 个值。以 `N=136874,K=6843` 计算，该历史缓存约 `4N+8K=602240` 字节/客户端，20 客户端约 12.0 MB，另有 EF、模型和工作区；不能说当前实现只新增 K 个 FP32/uint32 对。上行字节不受此差异影响。

## 同 K、同上传字节与资源口径

**当前上传预算一致。** 编码器 `guarded_codec.py:96–100,149–152,177–180` 使用 `K=max(1,min(N,int(N*rho)))`，每个选中位置上传原 FP32 数值、uint32 坐标，头部 64 字节；选中值恰好为零时仍按显式一对计费。控制选择还检查 K 个坐标唯一性（`run_controls.py:111–113`）。

所有受审稀疏规则同样 `64+8K`。当前 `N=136874,rho=.05` 时，`K=6843`，每客户端每轮 `54808 B`，20 客户端合计 `1096160 B/round`，20 轮 `21923200 B`。这是编码成本模型，不是网络抓包；没有包含下行、网络包、认证或账本费用。FedProx 不因近端目标新增上传。

RegTopK 和候选规则都可在每轮 dense 模型下发且客户端缓存前模型时重建历史全局增量；当前模拟直接给出 `previous`。如改变下发协议，新增参考获取成本必须重新核算。

`run_controls.py:213` 为自定义控制先计算完整 Top-K baseline，再做自定义选择和诊断。因此当前 elapsed_seconds 包含不同量的基准计算、双精度诊断和 Python 分配。它可以反映这个模拟程序的耗时，不能直接用于“方法本身更快/更轻”的计算性能结论。与上传准确率公平性是两个不同口径。

## 随机性、指标和数据隔离

**训练随机性口径一致。** 每个运行在模型初始化前设置 Python / NumPy / Torch seed（`run_followup.py:153–154`）；客户端训练 seed 为 `seed + rnd*100 + i`（同文件 `:199–202`），20 客户端时同一 seed 内没有轮次碰撞。两个本地训练函数均使用相同 batch size 64、独立显式 seeded DataLoader generator，FedProx 不改变 batch 顺序。随机 filler 与 shuffle reference 使用本地 generator，不消耗训练全局 Torch RNG；原检查脚本本轮重新验证该性质。不同训练 seed 之间未承诺严格独立的随机数子流；没有冻结库/硬件或启用跨平台完全确定性保证。

**指标计算通过静态核对。** `metrics` 用真实标签计算总样本 accuracy 和按样本加权 CE；macro / worst-class 是按类 accuracy 的均值/最小值。默认验证集按类抽样 2000 张图，避免客户端规模直接改变验证权重。若以后选择缺类数据，缺类会按 0 accuracy 进入 macro/worst-class，需另行定义；本次未重新对真实图像内容做缺类或重叠扫描。

`aggregate_relative_error` 的分母是各客户端当前 `delta+EF residual` 的样本加权聚合（`run_followup.py:200,206,212–219`）。它是**当前各自轨迹中 EF 候选的压缩误差**，不是相对同一个独立 dense FedAvg 训练轨迹的误差；FedProx 还改变了其本地目标。不能仅用跨方法较小的该诊断值推断更接近共同真实梯度或更好的泛化。snapshot 才提供同状态一步反事实，并已拒绝 FedProx / RegTopK（`run_controls.py:302–306`）。

**代码流层面的 test 隔离通过。** `prepare` 仅加载客户端 train 与本地 holdout，验证子集使用固定 seed 73021，与预测输出无关（`run_followup.py:53–101`）。clean 生成工具只读取官方 `CIFAR10(train=True)`，再分成客户端 train/holdout（`generate_clean_cifar10_split.py:29,61–69`）。独立官方 test 只在 `test_final` 为真且训练结束后加载（`run_followup.py:236–240`），其 loader 使用 `CIFAR10(train=False)`（`sparse_update_benchmark.py:45–60`）。这轮未对全部真实数据逐图像验重；`test_used_for_selection=False` 是硬编码的声明，不能阻止人后来按 test 指标调参。

抽查当前三份开发配置：`development_runs/{topk_ef,align90}_mix0_rho0.05_s20261113/config.json` 与 `strong_baseline_development/regtopk5_ef_mix0_rho0.05_s20261113/config.json` 的 20 轮、seed、模型参数数、学习率、epoch、线程、fraction、训练样本量 37500、验证样本量 2000、验证索引哈希及标签分配哈希一致；均 `test_final=false`，runner / codec hash 相同。三份配置签名重新计算均匹配。该抽查未把未完成训练当完整结果，也未评估任何准确率优劣。

## Adapter、配置签名与恢复

**机制通过。** `install()` 只修改本 adapter 进程导入的 runner 模块，未修改磁盘源码或其他运行进程。`_with_metadata` 在原 `one_run` 计算签名前加入 adapter hash、控制名称、FedProx/RegTopK 参数和适配说明（`run_controls.py:59–76,281–284`）；原配置签名包含 runner / codec hash、manifest 和其余参数，排除输出目录（`run_followup.py:165–176`）。本轮重新执行了 tiny one_run，确认 adapter hash 改变会改变签名。

FedProx 的 checkpoint 可使用原模型/残差/previous 恢复；本地每次重设 seed，且当前 SmallCNN 无模型外随机状态。RegTopK 新增逐客户端历史未进 checkpoint，adapter 明确拒绝任何未完成已有目录的恢复（`run_controls.py:268–279`）；同配置 DONE 运行交原 runner 验签后跳过。这比静默清空历史继续训练正确。snapshot 也拒绝这两个不兼容控制。

**完整溯源仍有缺口。** 原 config 未保存 `fedavg_update_diagnostic.py`（模型/向量化/数据加载）的 hash，未保存完整图像/分片 hash，也未保存 PyTorch / NumPy / Python 版本。`assignment_label_sha256` 只摘要标签顺序；`validation_ids_sha256` 只摘要相对本地 holdout 拼接顺序的索引。若同标签图片被替换、holdout 内容更换、导入工具改变，原运行签名可能仍相同。此项足以让未来比较在来源变化时悄然失配，但当前审查未发现实际发生替换。已在本报告补记导入工具 hash，后续正式导出/重跑时宜纳入完整来源记录，不能回填后冒称原配置已验证。

## 最可能形成虚假优势的错误与本轮检查结果

| 风险 | 当前发现 |
|---|---|
| FedProx 锚点错误、少了 1/2、梯度放到 step 之后或改变 batch 顺序 | 未发现；显式 loss 与多 batch fixture 通过 |
| RegTopK 把旧 u 作分母、额外负号、传 score 而非原值、漏掉样本权重 | 未发现；论文及不等权多轮 oracle 通过 |
| RegTopK 不同客户端共享历史、零全局聚合错误 fallback | 当前固定全参与路径未发现；zero-cancellation 检查通过 |
| 主方法比基线传更多坐标、漏算索引、零值免费 | 未发现；K 与 64+8K 检查通过 |
| 随机 filler 改变本地训练 RNG | 未发现；本轮全控制 RNG 隔离检查通过 |
| 验证函数误改变后续本地 train mode | 未发现；显式 local.train，当前模型无随机层/缓冲区 |
| 官方 test 被每轮读取、验证子集按预测筛选 | 代码流未发现；人工作法和全量图像身份未审计 |
| 将 EF 自身轨迹误差、模拟耗时或最低理论缓存解释为共同目标精度/算法性能 | 有解释风险；必须使用本报告的指标和资源口径 |
| 数据或依赖发生变化而旧签名仍通过 | 有来源锁定缺口；当前未发现变化证据 |

**最终状态：核心实现检查通过，带上述范围和溯源限制；没有发现需要立即停止当前冻结训练的实现 bug。** 这不是对“基线调参已充分”“主方法优于强基线”“跨种子稳定收益”或“可发表新颖性”的结论。

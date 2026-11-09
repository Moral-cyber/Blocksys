# 本地训练 MPS 加速入口独立代码检查

日期：2026-11-03。审查对象：`run_accelerated.py`、`summarize_accelerated.py`、`check_accelerated.py`，并核对 `launch.py` 与既有 controls / lagrangian / partitions / Fashion / 原 runner 的调用关系。

**本报告是同模型家族的独立静态检查与小型 CPU fixture 复核，不是 `experiment-audit` 跨模型审计。审查者没有运行完整训练、没有读取任何确认实验的官方 test 输出，也没有在本进程执行 MPS。MPS 实际执行检查由主代理在可用上下文另行负责。** 本报告不认证外部观察到的速度倍数或学习优势。

## 结论

当前版本在静态检查和 CPU 小型集成检查范围内通过，未发现新增的客户端状态、全局模型或本地模型写错问题。可以将该入口作为**新的 MPS 数值后端设置**开展后续实验；正式使用 MPS 仍需主代理的实际 GPU 检查通过，并使用新的输出目录。

- 本地训练仅将 deep copy 的模型、近端锚点和当前 minibatch 移至 MPS；原全局模型、编码、残差、加权聚合、验证与 checkpoint 留在 CPU。
- CPU backend 直接调用冻结的 `controls.train_local`，已验证 `mu_p=0/.01/.1` 的逐位一致性。
- FedProx 在本地训练调用时读取同一进程中的 `controls._ACTIVE_MU`；没有在入口安装前捕获错误的零值。
- late hook 位于各 entry 安装完成后的 `runner.main` 边界，保留 Fashion → lagrangian → controls → 原 one_run 的元数据与状态链。
- 加速来源、实际模型/加载/prepare 工具、后端与运行库身份在原配置签名生成之前进入 args。CPU/MPS 不能互相恢复同一旧目录。
- 混合结果必须使用 `summarize_accelerated.py`；冻结旧 `summarize_followup.py` 仍不识别后端，不能直接用于 CPU/MPS 混合目录。

## 本轮审查身份

| 文件 | 本轮审查 SHA-256 |
|---|---|
| `run_accelerated.py` | `f4a9169ab5ca9f2ea149f9bcd96e452bf04b399342b754e9f30eba6f8e01643a` |
| `summarize_accelerated.py` | `1024f4e005d08927a1b9db549d7ffeff90f9df241ab7d674a97664f27b72e54e` |
| `check_accelerated.py` | `9659643dc9bd2fdf632c09c2468d4c77196a62175ec3d3a2e3b423c9f0f98dda` |
| `launch.py` | `6108663ca1e8e0366784875c517b6c91a844719b84f1cf232b42f4bb5a901c27` |

`run_followup.py`、`run_controls.py`、`guarded_codec.py` 与此前基线审查的冻结哈希相同；审查者未编辑任何被审源码。独立 fixture 的运行、配置与 checkpoint 均在自动清理的临时目录中，既有检查报告未被覆盖。

## 训练、设备与状态

`run_accelerated.py:36–71` 先检查 MPS 可用及全局参数/缓冲区在 CPU，然后读取当前 `_ACTIVE_MU`。`base` 是 CPU 参数向量的独立副本，`local=deepcopy(model).to('mps')`。近端锚点在 GPU 私有模型开始训练前克隆，对应与 CPU 相同的本轮全局起点。

每个 minibatch 顺序为 CE backward → 加入 `mu_p*(p-anchor)` → SGD step。该顺序与已审 CPU FedProx 相同。`mu_p=0` 不创建额外 anchors。没有把 EF 残差用于近端锚点，没有在完成所有客户端前修改全局模型，也没有把 tanh/选择分数引入训练目标。

训练完成后将 local 参数展平并复制至 CPU，再减去 CPU `base`。返回值检查为有限 CPU 张量。压缩器和原 `one_run` 因而继续接收 CPU FP32 差分；它们不需要 MPS 状态。未新增跨客户端或跨轮次训练缓存，也没有新增未进入 checkpoint 的 GPU 模型状态。

`run_accelerated.py:74–79` 的 CPU 分支直接调用 `controls.train_local`，未复制训练逻辑。现有 RegTopK per-client history 和中断拒绝规则仍由 controls 管理；加速器没有替换该状态上下文。CPU/MPS 源码与后端签名不同，不能在旧 CPU 目录上静默切到 MPS 恢复。

随机性方面，MPS 分支同时执行 `torch.manual_seed(seed)` 与 `torch.mps.manual_seed(seed)`，DataLoader 使用显式 CPU generator、相同 batch size 64 与 shuffle seed。种子在每次客户端本地训练前重设。此设计保持数据顺序口径，并不保证 CPU 与 MPS 的浮点轨迹逐位相同；当前 SmallCNN 无 Dropout / BatchNorm 等额外随机或缓冲状态。

MPS 不可用会明确报错，不会由本入口静默改成 CPU。若环境显式设置 `PYTORCH_ENABLE_MPS_FALLBACK`，底层库仍可能对不支持的算子执行 CPU fallback；该环境值被记录在运行身份中，应保留原始配置解释该设置。

## 安装顺序与三类数据路由

`run_accelerated.py:156–177` 的 late_main 在选中 entry 安装完成后捕获 `runner.one_run/snapshot_run`，再包装 args 并替换 `runner.train_local`。因此既有 `controls.install()` 不会随后覆盖 GPU patch；Fashion 的专用 one_run wrapper 也没有被绕过。`finally` 恢复该边界的函数引用。

自动路由与实际 prepare 身份对应：

| 数据名 | 实际路由与 prepare |
|---|---|
| `Cifar10Clean*` | lagrangian entry，保留原 `run_followup.prepare`；可显式选择 run_controls |
| `Cifar10Index*` | 将 RAW 绑定到实际 `runner.PROJECT`，清理其 source cache，使用 `run_partitions.prepare`，随后安装 lagrangian |
| `FashionIndex*` | `run_fashion.main`，保留其 prepare 与延迟测试分派，再经 lagrangian |

Fashion 的过程内 metadata 路径替换只修正来源路径及说明，没有修改其样本转换或 test gate。该替换自身由加速入口源码哈希覆盖，且在退出时恢复。CIFAR Index 的 RAW 与 prepare 也在退出时恢复。

最终 CLI 在 parse 之前移除可选 `--` 分隔符，随后统一识别 backend / entry / dataset。显式 run_fashion / run_partitions 且未指定 dataset 时使用对应默认；不兼容 entry 和数据名明确拒绝。`launch.py` 已将 `run_accelerated` 注册为可移植入口。

审查发现并报告过一个中间版本的问题：`--backend cpu -- --dataset FashionIndexA0p1` 曾被追加的默认 CIFAR 数据名覆盖。实现代理修复后，审查者独立重跑该案例以及显式 entry 默认/不兼容案例，均通过。最终审查版本已不存在这一已复现的路由问题。

## 来源元数据与签名覆盖

`runtime_metadata` 在 late_main 中获取**已选定路由后的** prepare。记录覆盖：

- backend、route、加速 adapter 名称/版本/哈希；原 runner、controls、两个 codec、lagrangian、partitions、Fashion、launcher 源码哈希。
- 通过实际函数/类定位的 SmallCNN、load_client、prepare、CPU local train 来源哈希，以及所选项目下 test dispatch 工具哈希。
- Torch / TorchVision / NumPy / Python 版本、Torch git version 与 build configuration hash。
- 平台、架构、MPS 可用/构建状态、确定性算法设置、FP32 matmul precision、三个 MPS 环境设置及实际项目根。
- CPU 全局/编码设备政策、CPU DataLoader 顺序政策、MPS seed 政策及不同数值后端分组说明。

`metadata_args` 先复制 args 并注入这些字段，然后调用捕获的既有 one_run 链。Fashion、lagrangian 和 controls 使用不同字段名，最终原 `run_followup.one_run` 的 `vars(args)` 包含全部加速字段后才计算签名。独立 tiny run 重新计算签名匹配；仅改变 backend 就会改变签名。已有 controls / lagrangian / Fashion metadata 没有被覆盖丢失。

最终版本还在 `metadata_args:141–152` 读取实际 CifarClean 源 `config.json` 的 hash、alpha 和固定 Dirichlet 生成 seed，并与本次 CLI `split_seed` 的 mixing 重分配含义分开。独立最终回归核实当前 `Cifar10CleanA0p1` 的生成 seed 为 `20261102`，CLI mixing seed 为 `20261103`，alpha 为 `.1`；源配置 hash 匹配。新汇总也加入这些源身份，改变源配置或生成 seed 会改变 settings_id。这个补充没有把 CLI split_seed 冒称固定 NPZ 的原始分区 seed。

本入口主要补齐后端与代码来源记录。CifarClean 原有的图像内容哈希缺口仍存在；不能用新增模型/loader 源码哈希冒充完整数据图像溯源。Index / Fashion 延续其已有的原图行索引及原始来源记录。本轮没有重新读取真实数据验重。

## 汇总隔离

`summarize_accelerated.settings` 在原设置身份基础上加入后端及加速源、模型/数据工具、test helper、依赖版本和运行身份；缺少加速元数据的历史 CPU 归入单独 legacy 设置。原分析器的 DONE 完整性、种子单位、重复排除、验证指标与同设置配对规则保留。

独立合成 fixture 写入 CPU/MPS 各 Top-K 与 align90 四份结果，再加入一个改变依赖版本的 CPU align90，共五个 DONE 运行。新汇总生成三个 settings、五个 method groups，只产生两个同后端、同设置的 Top-K 配对；没有跨后端或跨依赖版本配对。此检查不接触真实实验和 test 字段。

此隔离依赖使用新汇总入口；旧 `summarize_followup.py` 的白名单没有 backend。不能把同后端的 Top-K 基线与另一个后端的候选拿来直接计算主张中的配对优势。

## 本轮实际验证

1. CPU passthrough：`mu_p=0/.01/.1`，130 个样本、多个 mini-batch，与冻结 controls 训练输出逐位一致；传入全局模型不变。
2. 三类真实 entry 安装链的 toy integration：保留真实 prepare 函数身份、实际图像加载用临时 toy 数据替代；每类覆盖 prox01 / regtopk5 / lagrange90、两客户端、两轮，共 36 次 local 调用。每次读取正确 `_ACTIVE_MU`，全局参数在 local 调用期间不变，返回有限 CPU delta。
3. tiny checkpoint 中 model、所有 residuals 和 previous 全部为 CPU；配置保留 backend、controls、lagrangian 以及 Fashion 专用 metadata；签名重算通过。
4. 重新执行现有 `check_accelerated.py`，报告写入临时目录：`CPU_PASS_MPS_UNAVAILABLE`。本进程 MPS 不可用，因此没有执行或报告 GPU 检查通过。
5. 参数分隔符、三类自动/显式路由、默认数据名及不兼容 entry 拒绝回归通过；最终新增 launcher/test helper 来源字段存在。
6. 上述五份合成结果的后端/依赖汇总隔离通过。
7. 最终冻结版本的 legacy NPZ 生成/mixing seed 区分、源配置 hash，以及 backend / 源配置 / 生成 seed / test helper hash 的分组隔离回归通过。

**独立审查状态：静态与 CPU 范围通过，未发现剩余实现阻断；MPS 实际运行/重复性/显式近端目标比较需由主代理独立取得结果。** 本报告不将 CPU fixture 或主代理口头观察等同于该 GPU 执行证据。

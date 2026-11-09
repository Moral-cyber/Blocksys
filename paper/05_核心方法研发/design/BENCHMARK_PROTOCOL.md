# CPU 编码成本与存储测量协议

日期：2026-11-03。工具：独立新增 `benchmark_codec.py`。不改动冻结的 runner、controls 或 codec；不加载训练图像、官方 test 或确认实验输出；不启动本地训练。

## 目的与当前完成状态

测量当前 Python/PyTorch 实现的一次本地 CPU 编码成本，比较 `N=136874`、`K=6843/27374` 时的 Top-K EF、prefix alignment（τ=.9/.99）、Lagrangian-family（τ=.9/.99）及 RegTopK（μ=5 的 model-delta 适配）。这些 K 分别对应现有 `floor(.05N)` 与 `floor(.2N)`。

本次只完成了小型功能检查：`N=31/127`、K=1/17/N、六种规则、输入不变、候选/残差守恒、精确 K 与字节、能量/对齐约束、算子计数、存储别名及释放、分位数计算和完整测量控制流程。**没有运行 N=136874 的计时，没有得到目标规模效率结论。** 当前并发训练环境中的小 fixture 数字不作为效率结果发布。

默认 `--mode check` 仅做上述小检查。正式计时必须显式选择 `--mode benchmark` 并给出输出文件。

## 输入来源

默认输入为 `deterministic_synthetic_proxy`，每个 case 的 delta、EF residual、previous_u 和构造的 g 都由显式 CPU generator 产生。g 包含该客户端上一轮所选值的权重贡献以及独立构造项，便于非退化选择；它不是实际 FL 全局聚合。**所有 proxy 时间/内存结果必须标注为代理数据微基准，不能冒称真实训练更新的实测分布。** 同一 case 内各方法使用完全相同的候选和参考。

每个 K 使用相同 seed 的输入生成逻辑，但上一轮支持集大小和 g 的构造随 K 改变；不能把两个 K 当作仅改变压缩率而历史状态完全相同的因果实验。默认三个 case 的结果分别报告，不将其当训练随机种子统计。

后续 root 可选择实际开发轨迹，显式保存并使用 `--tensor-file`。文件要求是一个 `torch.save` 字典：

```python
{
    "delta": ...,              # CPU FP32 [N]，本轮本地模型差分
    "residual": ...,           # CPU FP32 [N]，本轮编码前的 EF 残差
    "reference": ...,          # CPU FP32 [N]，上一轮实际全局稀疏聚合
    "previous_u": ...,         # CPU FP32 [N]，同客户端上一轮 delta+residual
    "previous_support": ...,   # CPU int64 [K]，该客户端上一轮真实所选支持
    "client_weight": 0.05,     # 必须替换为该实际客户端 n_i/sum(n_i)
    "provenance": { ... },     # 开发 run/config hash、轮次、客户端、来源说明
}
```

仅选择一个真实 case 所在的 K 运行；若两个 K 来自不同历史轨迹，分别保存文件并记录来源。脚本严格检查形状、dtype、有限值、支持范围/唯一性与权重，记录文件和各输入张量 SHA-256；caller provenance 本身不会自动被认证为真实训练证据。

**已有 runner checkpoint 不能独立提供完整本轮编码案例。** 它包含模型、EF 残差、previous 聚合等，通常没有当前本地 delta，而且现有 RegTopK checkpoint 不保存新增逐客户端历史。不能将 checkpoint 的 residual 单独改名为 u，不能根据当前模型反推或虚构旧 RegTopK 支持。实际案例需从明确的开发运行中另行捕获，或保持 proxy 标签。本轮没有读取任何真实 checkpoint。

## 调用边界与 RegTopK 的合理单独调用

Top-K / prefix 直接调用冻结 `guarded_codec.encode`，Lagrangian 调用冻结 `lagrangian_codec.encode`。RegTopK 调用冻结 `run_controls.encode_variant`，以单个被测客户端的上一轮状态构造局部 context，权重仍为其真实或 proxy 权重 `.05`，不是强行改成 1。`call_count=0` 仅定位这个被测客户端；其 previous history 已存在，所以测的是稳态编码，不是首轮普通 Top-K。

每次计时前在计时区间外恢复相同输入和 RegTopK 旧 context。实际调用中的 baseline Top-K、score、支持排序/唯一性检查、sent/residual 重建、所有双精度诊断以及新 history 的 `candidate.clone()` **均在计时区间内**。RegTopK 返回元组转为统一结果容器的少量 Python 包装也在区间内。输出检查、旧 context 恢复、张量哈希、计时结果存储和输出对象释放在区间外。

这里测的是 codec API 的实际计算及返回结果成本。未计入本地训练、canonical-pairs 逐对序列化、包头构造、网络、下行、认证、链上写入、磁盘 checkpoint 或端到端延迟。因此结果不能命名为“端到端联邦学习加速比”。

## 正式计时操作

等待机器没有同期训练后，以新 Python 进程执行。保留电源/温度/负载/同期工作情况，写进 `--context-note`。脚本不会终止其他进程或修改系统性能设置，也不自动认证机器空闲；它记录开始时系统 load average。

以下是**尚未执行**的单线程 proxy 正式命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/Tinyproto/bin/python \
  work/method_followup/benchmark_codec.py --mode benchmark \
  --n 136874 --ks 6843,27374 --threads 1 \
  --warmup 10 --repeats 30 --blocks 5 --cases 3 \
  --context-note "记录实际空闲/电源/温度和同期任务情况；这是synthetic proxy" \
  --out work/method_followup/codec_benchmarks/proxy_threads1.json
```

固定四线程另起进程、另存 `proxy_threads4.json`，其余配置相同并设置 `--threads 4`。不要混池不同线程、硬件、PyTorch 版本或源码身份。Torch inter-op 线程固定为 1；脚本记录实际 intra-op / inter-op 线程数，没有设置 CPU affinity。

正式真实案例例式为 `--tensor-file /明确选择的开发case.pt --ks 6843 --cases 1`；其余预热、重复与线程协议相同。不要指向确认 test 输出或把 `.pt` 训练 checkpoint 直接当完整案例。

每个 case / K / method 先完成一次独立检查和内存/算子观察，再预热 10 次。正式计时为 5 blocks × 30 calls，串行运行；每个 block 内的方法顺序按显式 seed 置乱，减轻固定顺序的热状态影响。每次调用使用 `time.perf_counter_ns`，计时阶段关闭 Python GC，退出后恢复原状态。CPU 调用同步完成，不存在未同步 GPU 时间。

JSON 按 case / K / method 保留全部原始 ns、方法 block 顺序、median / Q1 / Q3 / IQR / min / max。分位数使用排序样本的线性插值。重复调用是同输入的测量噪声样本，不是独立训练实验，不提供学习收益的置信区间。不要把多个 case/K/线程直接合并成一个“平均加速比”。

## 排序和多次 Top-K 的成本

脚本在一次**不计时**的调用中临时记录 `torch.topk`、`torch.sort` 和 `Tensor.unique` 的调用次数、输入长度及 K，随后恢复原函数；正式计时没有这些 hook。

| 当前实现 | 选择/排序的关键操作 |
|---|---|
| Top-K EF | 一次 N→K `topk(sorted=True)`，一次 K 索引 `torch.sort` |
| prefix .9/.99 | 上述 Top-K；K 能量累积扫描；有 filler 时再对剩余坐标做一次 signed-product Top-K；K 索引排序 |
| Lagrangian .9/.99 | baseline Top-K；λ=0、最多 24 次 high 扩张、12 次二分的 score Top-K；最坏上界共 38 次 Top-K；结果 build 与可能 numeric fallback 还会排序 K 索引 |
| RegTopK μ=5 | baseline Top-K 加 score Top-K；baseline 和自定义支持各排序 K；支持 `unique` 校验；另有 dense history clone |

`sorted=True` 要求对选中的 K 个结果排序，不能据此声称源码做了完整 N 元素 `torch.sort`。当前受审主路径没有显式全 N 排序；日志专门记录 `full_length_explicit_sort_calls`。PyTorch Top-K 和 unique 底层可能采用不同选择/排序策略及内部 workspace，本协议不为其虚构 O(N) 或严格复杂度。Lagrangian 的实际迭代数/solver_status 随 case 改变，应与时间一起报告，不能把上界 38 当每次实测次数。

## 持久状态、返回结果和上传口径

下面均为张量数据区字节，不含 Python 对象/allocator 元数据。

| 项目 | N=136874, K=6843 | N=136874, K=27374 |
|---|---:|---:|
| 客户端 EF residual：N 个 FP32 | 547496 | 547496 |
| 历史全局参考 g：N 个 FP32 | 547496 | 547496 |
| RegTopK 新增旧 candidate：N 个 FP32/客户端 | 547496 | 547496 |
| RegTopK 新增旧支持：K 个实际 int64/客户端 | 54744 | 218992 |
| RegTopK 相对同 runner Top-K EF 的额外持久状态/客户端 | 602240 | 766488 |
| 当前上传 FP32 values | 27372 | 109496 |
| 当前上传 uint32 indices | 27372 | 109496 |
| 同一固定头部 | 64 | 64 |
| 每客户端每轮总上传表示 | 54808 | 219056 |
| 相对同 K Top-K 的额外上传字段 | 0 | 0 |

当前同步模拟器只持有一份共享 previous g。部署到独立客户端时，客户端保存该 g 就需要每客户端 4N 字节；通过相邻 dense 全局模型重建时还可能需要前模型缓存，不能既不计 g 也不计重建缓存。当前冻结 runner 对 Top-K 也持有 previous 以做一致诊断，所以表中的“额外”相对的是该 runner，而非理论最简 Top-K。

RegTopK 当前实际保存 dense previous_u 和 int64 支持，不是最低理论的 K 对。prefix / Lagrangian 不新增逐客户端跨轮选择历史。所有 codec 在本地用 dense sent 和 residual 重建向量，返回时两者本身占 8N 字节，不能把 sparse wire bytes 当内部 resident 大小。全局/本地模型、SGD、训练激活和数据是其他成本，本协议单独排除，不等于不存在。

因此“额外上传字段 0”只表达同 K 消息格式不增长，不能写成“零计算开销”“零缓存”或“免费利用历史”。

## 临时内存观察及其限制

脚本在一次**独立、不计时**调用中使用 Torch dispatcher 追踪存活 CPU tensor 的唯一底层 storage：别名不重复计费，张量释放时撤销引用。小 fixture 验证了 11 个 FP32 数的 view 不增加 storage，另一个新向量增加 44 B，释放后回落。

结果含输入 storage、peak visible live storage、扣除输入后的增量峰值，以及返回 sent/residual/indices 与新 RegTopK history 的保留 storage。增量峰值包含返回结果和新历史，不能直接全部叫“临时 scratch”。方法特有但不参与调用的 proxy 历史辅助数组不计入该方法的可见输入基线。

该观察是**实际可见张量数据区的下界**，不是完整 resident memory/RSS 峰值：C++ 算子内部 workspace、Python 对象、allocator caching 和 OS 页无法在这个 hook 中全部看到。另记录 process RSS high-water 的前后值，供追溯；它受同进程先前输入构造、其他方法和 allocator 峰值影响，差值为 0 不代表没有分配，也不能称为隔离后的真实临时峰值。若后续需要发布严格峰值 resident 内存，应额外设计独立进程/allocator profiling，保留这两个口径，不替换成未验证的“精确峰值”。

所有正式输出还记录脚本及冻结依赖 SHA-256、运行库/平台、线程、输入张量身份和上下文说明。只有在目标规模、所声明输入来源和合适机器条件下实际执行后，才可把对应结果放入计算成本表；本次小检查不提供这些数值结论。

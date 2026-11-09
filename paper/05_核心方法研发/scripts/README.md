# 方法补充实验：运行与导出说明

本目录实现稀疏联邦学习候选机制、公开方法的适配对照，以及验证集分析工具。实验是同步联邦训练模拟；字节数按 FP32 数值与 uint32 坐标核算，不是网络抓包或区块链端到端测量。

## 当前文件和输入依赖

| 文件 | 用途 |
|---|---|
| `guarded_codec.py` | Top-K EF、能量保护核心及方向/创新/随机填充规则 |
| `run_followup.py` | 原始训练与同状态 snapshot 诊断运行器 |
| `run_controls.py` | 不改原运行器的扩展控制 adapter |
| `run_lagrangian.py` | Lagrangian 可行支持搜索 adapter，不声称原离散约束精确最优 |
| `run_partitions.py` | 从官方训练集原始行号重新生成独立客户端划分 |
| `run_accelerated.py` | 独立 CPU/MPS 本地训练后端；全局模型与所有压缩操作保持 CPU |
| `launch.py` | 导出后使用的入口，解析工程路径并核对实际数据加载器绑定 |
| `summarize_followup.py` | 仅验证集指标的种子级汇总、配对比较和 snapshot 分析 |
| `summarize_accelerated.py` | 额外隔离后端、代码和软件/硬件身份的验证集汇总 |
| `check_guarded_codec.py` | 编码与数学性质的小规模检查 |
| `check_controls.py` | 控制机制、近端梯度、历史状态和签名检查 |

当前已验证的 Python 环境为 `/opt/anaconda3/envs/Tinyproto/bin/python`，核心依赖是 PyTorch、NumPy；独立官方测试评估还依赖 torchvision。检查脚本通过只说明实现性质通过，不证明学习优势或算法新颖性。

**当前源代码存在相对路径依赖。** `run_followup.py` 将 `PROJECT` 定义为 `HERE.parent / 'tinyproto_diag'`，随后从其 `tools` 目录导入模型和数据加载函数。在当前工作区，这对应：

```text
work/
├── method_followup/             # 本目录
└── tinyproto_diag/
    ├── tools/
    │   ├── fedavg_update_diagnostic.py
    │   └── sparse_update_benchmark.py  # 仅官方测试评估需要
    └── dataset/
        ├── Cifar10CleanA0p1/
        │   ├── config.json
        │   ├── train/0.npz ... 19.npz
        │   └── test/0.npz ... 19.npz
        └── Cifar10/rawdata/      # 官方测试评估需要
```

这里 `Cifar10CleanA0p1/test` 是来自官方训练集的本地留出集。开发验证从这些留出图像中取固定分层子集；它与官方 CIFAR-10 test split 不同。训练图像、参与客户端数、样本量和分区来源记录在各运行的 `config.json`。

## 在当前工作区运行

以下命令从 `/Users/wind/Documents/Codex/2026-11-02/zhe` 执行。先运行小规模实现检查：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python work/method_followup/check_guarded_codec.py
/opt/anaconda3/envs/Tinyproto/bin/python work/method_followup/check_controls.py
```

运行一组明确指定的训练配置，例如：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python work/method_followup/run_controls.py \
  --mode train \
  --variants topk_ef,align90,regtopk5_ef \
  --dataset Cifar10CleanA0p1 \
  --mixing 0 --fraction .05 --rounds 20 \
  --seeds 20261113 --validation-size 2000 \
  --lr .03 --local-epochs 1 --threads 4 \
  --out work/method_followup/control_runs
```

`run_controls.py` 要求显式指定 `--variants`，避免无意启动一大组实验。`--mixing 0` 保留原异构分配；`--mixing 1` 将训练图像汇入全局池后随机重新分配，同时保持每个客户端的样本量和全局训练图像不变。中间取值只混合相应比例图像。`--fraction` 是保留坐标比例，不是直接的字节压缩比例。

只有原运行器中的规则可以直接使用 `run_followup.py`；扩展控制必须通过 `run_controls.py` 启动。开发完成前不要加 `--test-final`。它会在该运行最后一次读取官方测试集；不可用这些指标筛选变体、超参数或轮次。已有 DONE 目录的训练配置包含该标志，事后更改标志会改变签名，不能视作原地追加一次测试评估。

## 扩展控制含义

| Variant | 规则与参数 |
|---|---|
| `prox001_ef` | FedProx 本地目标，近端系数 `.01`；上传仍为 Top-K EF |
| `prox01_ef` | FedProx 本地目标，近端系数 `.1`；上传仍为 Top-K EF |
| `sign50_ef` | 最多预留 `floor(.5K)` 个与参考方向同号的坐标，按幅值选，剩余幅值补齐 |
| `fixed50_ef` | 固定幅值核心 `floor(.5K)`，其余按 signed `u*g` 选 |
| `reverse90_ef` | 能量保护系数 `.9`，使用上一轮全局更新的反向参考 |
| `shuffle90_ef` | 能量保护系数 `.9`，使用固定种子 `73023` 置乱参考坐标 |
| `regtopk1_ef` | RegTopK 选择规则适配到模型增量，正则参数 `1` |
| `regtopk5_ef` | RegTopK 选择规则适配到模型增量，正则参数 `5` |
| `regtopk10_ef` | RegTopK 选择规则适配到模型增量，正则参数 `10` |

所有控制都使用相同坐标保留数 `K`、原始候选数值及 EF，编码模型为 `64 + 8K` 字节。FedProx 的近端系数与 RegTopK 的正则参数是不同机制，在配置中分别记为 `fedprox_mu` 和 `regtopk_mu`。

RegTopK 是 **RegTopK-on-model-deltas adaptation**：根据原文 Algorithm 2 的过去聚合信息与逐客户端支持集计算选择分数，但本实验使用多步本地训练的模型增量，不能写成对原始分布式 gradient-SGD 全实验的完整复现。参数 `5` 来自本次核查的作者公开实现 `DistLearn.py` 中 `RegTopK` 的 `Mu = 5`；`1/10` 保留为开发比较。核查的仓库为 [AlBerey/RegTopK](https://github.com/AlBerey/RegTopK)，对应树标识 `1ff4324ede3f6298322987cb19564a4a3e3d008b`；来源副本与检索记录在本工作区的 `work/literature_followup/regtopk_author_code_readonly/`。本 adapter 独立实现选择公式，没有执行作者仓库代码。

`sign50_ef` 只是相关符号引导思想的机制控制，不是某个已发表算法的完整复现。`reverse90_ef` 与 `shuffle90_ef` 是方向负控制。其 `guidance_alignment_gain` 对应变换后的指导方向，常规 `alignment_gain` 仍相对未变换的上一轮全局更新。

## snapshot、恢复与配置签名

snapshot 在同一密集训练轨迹状态上，比较不同压缩规则的一步影响；它不等同于各方法独立训练。以下控制不能运行 snapshot：

- FedProx：它改变本地目标，无法共享同一组 dense 轨迹更新。
- RegTopK：它需要每一轮真实发送的逐客户端支持集历史，不能跳过中间轮次。

普通训练运行每轮保存 checkpoint。精确恢复要求同一源码、参数、variant 列表及数据身份；配置发生变化时需新输出目录，不覆盖旧运行。adapter 的源码哈希、具体控制规则和参数已经纳入配置签名。

**RegTopK 当前不支持中断恢复。** 它新增的逐客户端历史不在原运行器 checkpoint 中，adapter 遇到未完成的已有 RegTopK 目录会明确拒绝恢复，要求使用新 `--out`。同配置的已 DONE 运行可跳过；不能仅凭原 checkpoint 静默清空 RegTopK 历史再继续。

## 汇总结果

```bash
/opt/anaconda3/envs/Tinyproto/bin/python work/method_followup/summarize_followup.py \
  --input work/method_followup \
  --output work/method_followup/current_summary
```

输出包括每运行 CSV、跨种子均值/样本标准差、相对 Top-K EF 的同种子配对差、按轮次分开的 snapshot 表、运行中清单和中文报告。只有完整 DONE 训练轨迹进入正式训练统计。最后五轮均值先在每个运行内求取，再跨种子汇总；不能将五个轮次当五次独立重复。单种子不报告样本标准差。

分析器不访问或筛选官方测试指标。配对要求设置与数据身份兼容；不同数据、轮次、优化配置或原运行器/编码器源码不会混入同一统计组。若同名 adapter 控制跨版本运行，打包与汇总时还应核对各 config 的 `adapter_source_sha256`，不能静默将不同实现当成同一算法。

## 导出后直接运行：使用 launch.py

冻结的 `run_followup.py` 本身仍不读取 `FL_PROJECT_ROOT`。**导出后推荐通过新入口 `launch.py` 运行**，它支持这个变量并在当前进程中绑定工程路径，无需修改冻结源码。仅给直接调用的旧运行器设置环境变量仍不会生效。

路径解析优先级为：

1. `--project-root` 明确指定的工程目录。
2. `FL_PROJECT_ROOT` 环境变量。
3. 三个有限的相对位置：本目录的 `../tinyproto_diag`、`../../02_实验工程`、`../02_实验工程`。

工程根必须直接包含 `tools/fedavg_update_diagnostic.py` 与 `dataset/`。明确指定的路径或环境变量无效时，会报出缺失项，不会静默改用另一个目录。入口不会搜索全盘。

导出目录应保留本表所列脚本在同一目录中，`--entry` 选择 `run_controls`、`run_lagrangian`、`run_partitions` 或 `run_accelerated`。其余训练参数原样转发；可用 `--` 分隔入口参数和训练参数。先做只检查路径的 dry run：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python /导出目录/scripts/launch.py \
  --project-root "/Users/wind/Desktop/Codex/paper/02_实验工程" \
  --entry run_controls --dry-run -- \
  --variants topk_ef,align90,regtopk5_ef \
  --fraction .05 --rounds 20 --seeds 20261113 \
  --out /新的结果目录
```

`--dry-run` 只显示解析出的工程、入口及转发参数，不导入训练模块、不读图像、不创建结果目录。去掉 `--dry-run` 才会实际运行。也可通过环境变量指定路径：

```bash
FL_PROJECT_ROOT="/实际路径/包含tools和dataset的实验工程" \
  /opt/anaconda3/envs/Tinyproto/bin/python /导出目录/scripts/launch.py \
  --entry run_lagrangian -- \
  --variants lagrange90_ef,lagrange99_ef \
  --fraction .05 --rounds 20 --seeds 20261113 \
  --out /新的结果目录
```

若脚本位于 `paper/某个材料目录/scripts/`，而已有实验工程位于 `paper/02_实验工程/`，可以省略显式路径，由上述第二个相对位置解析。仍建议先 dry run 查看实际位置。

真实运行时，入口会先从选定工程的 `tools` 导入数据工具，核对其文件位置及 `ROOT`，然后导入本目录冻结运行器并绑定 `runner.PROJECT`，再次确认 `load_client` 与模型类来自该工程。`run_partitions` 在这之后才导入，其原始图像路径 `RAW` 也在本进程中绑定到选定工程。冻结文件的内容和哈希不变。若当前 Python 进程已经导入另一个工程的数据工具，会明确拒绝复用；应启动新的 Python 进程。

## 导出包的数据依赖

- `run_controls` 与 `run_lagrangian` 的默认 `Cifar10CleanA0p1` 设置，需要选定工程中该数据集的 20 份 `train/*.npz`、20 份本地留出 `test/*.npz` 及 `config.json`。图片仍在原实验工程中，复制这些入口脚本不会自动复制数据。
- `run_partitions` 使用 `--dataset Cifar10IndexA0p1`、`Cifar10IndexA0p5` 或 `Cifar10IndexA1`，从选定工程的 `dataset/Cifar10/rawdata/` 读取已缓存官方 CIFAR-10 数据，按 `--split-seed` 生成新的原始行号划分。无需预先生成对应的 NPZ 数据集目录。缓存需满足 torchvision 的 CIFAR-10 完整性要求；入口不会自动下载。
- `--test-final` 另外依赖工程中的 `tools/sparse_update_benchmark.py` 及原始 CIFAR-10 缓存；它只能用于方法冻结后的最终测试评估。
- `launch.py --dry-run` 检查工程目录标记和入口文件，不对每个数据文件做完整校验；实际运行需要上述对应数据输入齐备。

例如独立分区开发可通过：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python /导出目录/scripts/launch.py \
  --project-root /实际实验工程 --entry run_partitions -- \
  --dataset Cifar10IndexA0p1 --split-seed 20261121 \
  --variants topk_ef,align90 --fraction .05 --rounds 20 \
  --seeds 20261131 --out /新的独立划分结果目录
```

不同方法、数据划分和路径下的结果都需保留实际配置与来源；不能修改冻结源码后继续冒充原版本的恢复运行。launcher 会在真实运行开始时打印其自身及选定数据工具的源码哈希，便于与运行日志一并保存。

## 独立本地训练加速后端

`run_accelerated.py` 在各数据入口安装其覆盖函数之后，才替换本地训练函数。`--backend cpu` 直接调用冻结的 CPU 本地训练函数；`--backend mps` 仅将客户端的私有模型副本、该批图像/标签及 FedProx 锚点放到 MPS。全局模型、验证评估、Top-K/Lagrangian 等编码器、残差、聚合和 checkpoint 始终在 CPU。本地训练返回 CPU 增量，不修改传入的全局模型。

自动路由遵循数据集名：`Cifar10Clean*` 使用原 NPZ 输入和 Lagrangian adapter；`Cifar10Index*` 组合独立原始行号划分与 Lagrangian adapter；`FashionIndex*` 使用 Fashion adapter 的固定输入转换与惰性最终测试工厂。三条路径都支持已经注册的主方法和对照。也可通过加速入口自身的 `--entry run_controls/run_lagrangian/run_partitions/run_fashion` 明确选择兼容入口；不兼容的数据集与入口会报错。

```bash
/opt/anaconda3/envs/Tinyproto/bin/python work/method_followup/run_accelerated.py \
  --backend mps --dataset Cifar10IndexA0p1 \
  --split-seed 20261121 --seeds 20261131 \
  --variants topk_ef,lagrange90_ef,regtopk5_ef \
  --fraction .05 --rounds 20 --out work/method_followup/mps_runs
```

导出包可从 launcher 进入：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python /导出目录/scripts/launch.py \
  --project-root /实际实验工程 --entry run_accelerated -- \
  --backend mps --dataset FashionIndexA0p1 \
  --variants topk_ef,lagrange90_ef --fraction .05 --rounds 20 \
  --seeds 20261131 --out /新的MPS结果目录
```

这一路径的 Fashion 压缩源文件仍位于脚本目录的 `datasets/FashionMNIST/raw/`，需随导出材料保留；来源和官方 MD5 在 Fashion adapter 中核对。它不会将 Fashion 最终测试静默替换成 CIFAR 测试。`--test-final` 仍是方法冻结后才启用的显式最终评估开关。

MPS 在当前 Python 进程不可用时会明确报错，程序不会静默降级成 CPU。不同权限或执行环境可能呈现不同的 MPS 可用性。实现检查命令为：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python work/method_followup/check_accelerated.py --require-mps
```

不加 `--require-mps` 时，若 MPS 不可用，只会输出 `CPU_PASS_MPS_UNAVAILABLE`，不能将其当作 GPU 检查通过。测试使用小模型、三个 minibatch 和极小集成输入，不进行真实数据集的长训练。

每个加速运行会把实际后端、新 adapter 源码、模型/数据加载器/测试工厂与方法依赖源码哈希、PyTorch/NumPy/torchvision/Python 版本、PyTorch 构建身份及运行环境注入原始配置签名。CPU 与 MPS 的浮点训练轨迹不要求逐位一致，正式比较应使用相同后端。加速结果请使用以下汇总器，避免与历史 CPU 结果混池：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python work/method_followup/summarize_accelerated.py \
  --input work/method_followup/mps_runs \
  --output work/method_followup/mps_summary
```

该工具继续执行 DONE 轨迹、种子级重复和同种子配对的规则，并额外隔离后端、加速源码、模型/数据工具、软件依赖与运行身份。缺少加速元数据的历史 CPU 单独分组；不将不同后端或不兼容软件版本的运行配成一对。

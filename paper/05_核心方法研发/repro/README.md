> 当前状态（2026-11-04）：本目录保留历史复现辅助材料，以下“探索性”“driver尚不存在”等文字属于各阶段交付记录。最终推理 driver 已存在于本目录的 `work/predictive_final_test/infer.py`，两任务固定三种子的官方 TEST 结果均为 DONE，实际结果与状态在 [当前 results 目录](/Users/wind/Desktop/Codex/paper/05_核心方法研发/results/predictive_final_test/)，训练模型在 [predictive_pfl_confirm](/Users/wind/Desktop/Codex/paper/05_核心方法研发/results/predictive_pfl_confirm/)。当前稿仅维护 [英文论文](/Users/wind/Desktop/Codex/paper/03_英文论文/)，历史生成脚本不再作为双语交付入口。
>
> 运行路径仍需显式映射：历史 `work/predictive_pfl_confirm/development_*` 对应当前 `05_核心方法研发/results/predictive_pfl_confirm/development_*`；历史 `work/predictive_final_test/train_states_*` 对应当前 `05_核心方法研发/results/predictive_final_test/train_states_*`；CIFAR 原始数据在 `02_实验工程/dataset/Cifar10/rawdata/`，Fashion 原始数据在 `05_核心方法研发/scripts/datasets/FashionMNIST/raw/`。现有 driver 保留历史相对路径，不能据此假定从本目录即可重跑最终推理；迁移时应另作明确路径适配并使用新输出，不改历史数据、来源哈希或结果。核心贡献、系统闭环和英文写作优先，便携复现不是当前论文工作的前置要求。
>
> 当前本机核验入口已接通：[replay_saved_inference.py](/Users/wind/Desktop/Codex/paper/05_核心方法研发/scripts/replay_saved_inference.py)。从论文根目录运行 `python -B 05_核心方法研发/scripts/replay_saved_inference.py` 仅预检；显式路径参数为 `--project-root`、`--method-root`、`--fashion-raw`。添加 `--execute --out /新的绝对路径/result.json` 才重放固定六个主模型的图片推理，只写一个新 JSON，拒绝覆盖结果；不训练、不重新拟合、不改旧锁，也不生成分数副本。已完成两任务三种子、120客户端、60,000张图片重放，raw logits全部逐位一致，主方法各汇总指标最大差3.67e-16。详见 [重放结果](/Users/wind/Desktop/Codex/paper/05_核心方法研发/results/replay_saved_inference/result.json)。本入口核验原始网络和先验向量预测，不重新运行参考方法或本地末层适配头。

> 本轮系统执行也已完成：[合约/HTTP运行说明](/Users/wind/Desktop/Codex/paper/05_核心方法研发/scripts/chain_integration/README.md)，两客户端两轮128项断言、4笔登记交易和6项EVM调用检查通过。此处以下各阶段历史说明不覆盖当前状态；本机单节点开发链不等同于生产多节点或公链最终性。

# 当前有界开发实验的复算入口

本包保存源码和确切的共同warmup输入。实验仍为探索性，未达到核心贡献目标。
原始结果在上级results/对应组件；文献证据位于上级literature/，本包不改原结果。
支持partial_fusion、receiver_feedback、normalized_sparse、shadow_training、mixed_precision、head_control、control_reference、balanced_quant、prior_bias、calibrated_lowbit_confirm、calibration_baselines、calibrated_fashion、prior_vector、predictive_pfl_confirm、calibrated_topk_reference和fedrod_linear_reference。
repro_source_manifest.json逐项记录原始路径、复制后路径及哈希；训练和codec源码保持原样。
另收录predictive_final_test与predictive_manuscript，包括.tex/.bib稿件源；draft不等于finalTEST已运行。
final_stage.py提供单独明确路由：manuscript inventory、final-test CPU fixture及root新增driver。
实际driver若尚不存在，明确报错；不自动选择脚本、不生成root-lock、不补--execute。

依赖Python、PyTorch、torchvision、NumPy。当前验证环境为/opt/anaconda3/envs/Tinyproto/bin/python。
原始CIFAR数据缓存保留在既有实验工程，避免复制大数据。工程须含tools/和dataset/Cifar10/rawdata/。
Fashion入口显式指定--fashion-raw，只要求两个官方TRAIN gzip，正式新路由不读取TEST字节。
calibration_baselines只读取显式--input-root下保存的数组，及prior_bias的NumPy fixture，无需--project-root。

从本目录执行，先检查路径（不训练）：

```bash
python launch.py --project-root "/Users/wind/Desktop/Codex/paper/02_实验工程" --entry receiver_feedback --dry-run
```

复算已完成诊断或有限闭环，请分别使用新的输出目录：

```bash
python launch.py --project-root "/实际实验工程" --entry partial_fusion -- --out /新诊断输出 --backend mps
python launch.py --project-root "/实际实验工程" --entry receiver_feedback -- --out /新闭环输出 --backend mps --rounds 20 --seed 20261144
python launch.py --project-root "/实际实验工程" --entry mixed_precision --dry-run -- --out /新量化输出 --backend mps --rounds 20 --seed 20261181 --methods cal_uniform3,cal_mixed_mse,cal_mixed_sensitive
python launch.py --project-root "/实际实验工程" --entry control_reference --dry-run -- --out /新控制参考输出 --backend mps --rounds 20 --seed 20261191
python launch.py --project-root "/实际实验工程" --entry calibrated_lowbit_confirm --operation train --dry-run -- --out /新CIFAR确认输出 --backend mps --rounds 20 --partition-seed 20261203 --seed 20261211
python launch.py --project-root "/实际实验工程" --entry calibrated_lowbit_confirm --operation evaluate --dry-run -- --model-root /同一便携位置新训练输出 --out /新CIFAR评估输出
python launch.py --project-root "/实际实验工程" --entry calibrated_fashion --fashion-raw /实际FashionMNIST/raw --operation train --dry-run -- --out /新Fashion输出 --backend mps --rounds 20 --partition-seed 20261207 --seed 20261221
python launch.py --project-root "/实际实验工程" --entry calibrated_fashion --fashion-raw /实际FashionMNIST/raw --operation evaluate --dry-run -- --model-root /同一便携位置新Fashion训练输出 --out /新Fashion评估输出
python launch.py --entry calibration_baselines -- --input-root /已完成的prior_bias或confirm评估数组目录 --out /新强校准对照输出
python launch.py --project-root "/实际实验工程" --entry prior_bias --probe-model-root /原冻结balanced模型目录 --probe-baseline-root /原完成prior_personalization目录 --dry-run -- --out /新bias诊断输出
```

mixed_precision的run.py、codec.py和EXPERIMENT_PLAN.md保留在同一目录；启动器优先提供入口所在目录的导入路径。
上例mixed_precision仅检查路由与参数转发；实际复算时移除--dry-run，并始终使用新的--out。

MPS不可用时明确报错，若主动改为cpu将形成另一数值后端，不能合并为原MPS重复。
receiver_feedback读取随包提供的同分区10轮warmup；更改seed只改变后继训练随机性，不重建warmup。
索引划分、源代码/运行时身份及完整数据成员会写到新输出配置。移动路径会改变运行签名，
因此新位置复算应使用新out，不能当原路径checkpoint继续。先核对两helper的固定源码哈希。
calibrated_lowbit_confirm与Fashion的evaluate严格检查完整来源/配置，应使用同一便携位置、同一工程新训练的checkpoint；不能给原路径checkpoint换来源。
prior_bias保留原始基线输入字典中的绝对模型路径校验，显式提供的两个原结果目录必须仍满足该关系；不自动重写旧路径或旧结果。
强校准probe需要完整4组旧模型或3组confirm模型、每组20个client保存数组和DONE result.json；本源码包不复制这些数组/模型。

CPU fixture入口（合成数据与NumPy校验，不是正式ML质量结果）：

```bash
python -B launch.py --project-root "/实际实验工程" --entry calibrated_lowbit_confirm --operation fixture
python -B launch.py --entry prior_bias --operation fixture
python -B launch.py --entry calibration_baselines --operation fixture
python -B launch.py --project-root "/实际实验工程" --entry calibrated_fashion --fashion-raw /实际FashionMNIST/raw --operation fixture
```

Fashion fixture额外只读校验真实TRAIN gzip与分区，真实数据CNN forward/backward为0；正式训练仍需单独明确调用train入口。

## 新prospective路线：必须显式operation

新PLAN、registration、assessment锁以及solver/evaluator/assess保持原SHA。新NLL研究问题与旧accuracy STOP分开。
主路线把冻结CIFAR/Fashion训练与原评估作为明确操作；launcher先校验新PLAN及规定partition/seed/20轮，
不改冻结训练source或其原签名规则。新位置使用fresh out；saved-array适配与assessment分别绑定新PLAN/registration。
下面都是dry-run，验证路由，不读图像/新数组或启动训练：

```bash
python launch.py --project-root /实际实验工程 --entry predictive_pfl_confirm --operation train-cifar --dry-run -- --partition-seed 20261217 --seed 20261231 --rounds 20 --backend mps --out /新CIFARprospective输出
python launch.py --project-root /实际实验工程 --entry predictive_pfl_confirm --operation evaluate-cifar --dry-run -- --model-root /同位置新CIFAR模型 --out /新原评估输出
python launch.py --project-root /实际实验工程 --entry predictive_pfl_confirm --operation train-fashion --fashion-raw /实际FashionMNIST/raw --dry-run -- --partition-seed 20261219 --seed 20261241 --rounds 20 --backend mps --out /新Fashionprospective输出
python launch.py --entry predictive_pfl_confirm --operation adapt --dry-run -- --input-root /已完成新原评估目录 --out /新20维适配目录
python launch.py --entry predictive_pfl_confirm --operation assess --dry-run -- --task cifar --seed 20261231 --evaluation-root /已完成新原评估目录 --adaptation-root /已完成新20维适配目录 --out /新阶段判定目录
python launch.py --entry prior_vector --operation saved-evaluate --dry-run -- --input-root /已完成原评估目录 --out /新通用适配目录 --protocol-plan work/predictive_pfl_confirm/PLAN.md --protocol-registration work/predictive_pfl_confirm/PREREGISTRATION.json
python launch.py --project-root /实际实验工程 --entry calibrated_topk_reference --operation prospective-train --dry-run -- --partition-seed 20261217 --seed 20261231 --rounds 20 --backend mps --out /新TopK输出
python launch.py --project-root /实际实验工程 --entry calibrated_topk_reference --operation prospective-evaluate --dry-run -- --model-root /同位置新TopK模型 --out /新TopK评估
python launch.py --entry fedrod_linear_reference --operation prospective --dry-run -- --partition-seed 20261217 --seed 20261231 --backend cpu --out /新FedRoD设计预检
```

TopK与predictive adapt若未显式传入protocol参数，launcher补充并显示本包的固定PLAN/registration路径。
FedRoD prospective本身默认DESIGN_ONLY；只有另加原入口--execute-formal才启动固定20轮，launcher不会补这个开关。
其formal路线另需--project-root绑定原TRAIN cache。三个新方法入口都没有隐式默认operation。
FedRoD SOURCE_LOCK及两份精确paper来源证据按原work相对布局复制，供原校验使用。
predictive assess沿用冻结旧source/registration检查；不能重写adapted输入的绝对来源关系或用原位置checkpoint换身份。

新增CPU检查（仅合成toy，无真实prospective数组）：

```bash
python -B launch.py --entry prior_vector --operation fixture
python -B launch.py --entry prior_vector --operation saved-fixture
python -B launch.py --entry calibrated_topk_reference --operation prospective-fixture
python -B launch.py --entry fedrod_linear_reference --operation prospective-fixture
```

11维prior-anchored BCTS是单独机制control，不是新candidate/gate。原主solver/evaluator不变。
portable路径改变后，先运行以下纯synthetic fixture以生成当前路径的BCTS_CONTROL_FIXTURES.json，
再显式给出同一原评估和对应已完成20维适配目录；control只拟合11维，复用20维状态，不重拟合主方法。

```bash
python -B launch.py --entry prior_vector --operation bcts-fixture
python -B launch.py --entry prior_vector --operation bcts-control --dry-run -- --input-root /同位置fresh原评估 --vector-root /对应已完成20维适配 --out /新11维control --protocol-plan work/predictive_pfl_confirm/PLAN.md --protocol-registration work/predictive_pfl_confirm/PREREGISTRATION.json
```

BCTS_CONTROL_PLAN.md及control/probe/fixture源码原样复制；现有fixture JSON是来源证据，
不以旧路径fixture冒充便携路径可直接续用。44B/80B状态、FP64评分和不同维度ridge解释见其独立说明。

最终阶段增量（只列清单或dry-run）：

```bash
python -B final_stage.py --component predictive_manuscript --operation inventory
python -B final_stage.py --component predictive_final_test --operation fixture --dry-run
python -B final_stage.py --component predictive_final_test --operation driver --driver-file driver.py --dry-run -- --root-lock /root审阅后生成的锁 --execute
```

driver-file必须是predictive_final_test组件内显式的Python文件且已存在。--dry-run不导入driver、不读TEST。
真正finalTEST由root自行决定；root-lock绑定本包路径下实际PLAN/data/scorer指纹，移动后使用fresh输出，
不可改原签名/旧结果或把synthetic fixtures当真实finalTEST。HELPER_REVIEW.md列明类列序和sumCE/TRAIN-state来源等driver责任。

字节含义必须区分：部分融合旧诊断为解析公式；后继闭环生成真实应用序列化字节计长度，
训练聚合仍使用等价原张量（独立decoder fixture已核查），没有实际网络、链或认证测量。
共同warmup开销按公式追记，不是历史网络抓包。以上入口仅面向TRAIN划分的开发集。

本地持久化协议v2保持work/下的完整相对源码布局，包括specs、固定v1五文件、冻结量化codec和旧toy。
从本repro目录执行以下命令可运行62项synthetic protocol tests，不需要实验工程或ML训练：

```bash
python -B work/durable_integrity_v2/test_protocol.py --output-dir work/durable_integrity_v2/新测试目录
```

新测试目录必须位于work/durable_integrity_v2/内，并且此前不存在synthetic_cases或同名_private_test_keys。
随机测试keys在旁边新生成，发布清单排除*_private_test_keys及local_test_keys.json；原始来源的keys不随包复制。
SQLite协议测试不表示真实网络、链共识或ML质量验证。

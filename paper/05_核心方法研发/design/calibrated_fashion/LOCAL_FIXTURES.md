# Fashion wrapper 本地交付检查

2026-11-04，状态 **PASS**。此次只执行 CPU 合成 toy，以及真实 FashionMNIST TRAIN 的只读校验、分区和单样本预处理检查；未启动正式训练，真实数据 CNN forward/backward 均为 0。

## 独立路由与冻结边界

`adapter.py` 通过 importlib 载入冻结的 `calibrated_lowbit_confirm/common.py`、`train.py`、`evaluate.py` 与 `method_followup/run_fashion.py`，校验固定 SHA。仅在本进程覆盖 prepare、dataset proxy、source identities、runtime metadata 与 TEST guard；不修改冻结源文件。载入 train/evaluate 时短暂替换 `sys.modules['common']`，随后恢复原绑定。训练直接调用原 train.run，评估直接调用原 evaluate.run；六 gates 不变。

数据读取只调用两个 TRAIN gzip 的新 loader。不会调用旧 `run_fashion.load_official_train` 或 `source_file_manifest`，因其会触及 TEST 文件。最终 clientholdout 数据集由 `FashionIndexDataset` proxy 提供。runtime/config 签名包含实际 Fashion task、全部路由覆盖、wrapper 与冻结依赖指纹；继承 helper 中保留的 CIFAR 文字通过明确的实际路由元数据限定。Fashion/CIFAR checkpoint 不互载。

## 真实 TRAIN 只读证据

- 官方 TRAIN 共 60000，10 类各 6000；partition seed 20261207、alpha 0.1、20 clients。
- 实际 TRAIN 为 **44999**、clientholdout 为 **15001**；globaldev2000 为 holdout 内子集，不另加计数。固定原始行号，分区重复一致，完整覆盖且 TRAIN/holdout 无交集。
- 灰度 uint8 28×28 在 uint8 域零填充 2 像素至 32×32，重复三通道，再转 FP32 并映射至 [-1,1]。逐值手算预处理一致；填充值映射 -1，三通道一致。
- 两个压缩 TRAIN 文件的官方 MD5 与既有 SHA256 均核验，原标签以 little-endian int64 编码的 SHA256 为 `e3245b63f7c40d1c8b652835f19744d970c1613b40ddac6bcfc87b9c46e16a0b`。
- TRAIN images：MD5 `8d4fb7e6c68d591d4c3dfef9ec88bf0d`；SHA256 `3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84`。
- TRAIN labels：MD5 `25c81989df183df01b3e8a0aad5dffbe`；SHA256 `a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845`。
- 文件访问审计实际只打开上述两个 TRAIN gzip。真实 TEST 字节读取尝试 **0**；3 次测试专用 dummy `test_batch`/`t10k` 访问均被 guard 拒绝，不是对真实 TEST 的访问。

完整分区行号、类别计数、文件和源指纹保存在 `TRAIN_PARTITION_FIXTURE.json`。

## CPU toy 检查

合成 28×28 uint8 数据，2 clients，各有 3/5 TRAIN 与 2/2 holdout。ce_uniform3、fedlc_uniform3、fedlc_dense 各跑 2 轮，复用原 SmallCNN136874 与 loss/codec/solver。

- 三组相同初始化、batch seed/anchor；首轮 FedLC quant/dense 的本地 delta 逐位相同。
- uniform3 实际首包 54067B，低于 54808B cap；dense 547560B。执行真实 wire encode/decode 与 error feedback。
- 完成 checkpoint 的 resume 不再训练；不同 task 的配置签名被拒绝；config/runtime/source/routes 已签名。
- 三组最终所有评估变体均经过 FashionDataset proxy；TRAIN-only bias 重新拟合结果一致，24 次合成 TRAIN 样本 forward、CNN backward 0；匹配 head 对照执行 6 次梯度评估。bias 状态 10 维 FP32、40B/client。
- toy 的六 gates 会完整输出；其中 +0.5pp accuracy gate 为 false。这只验证失败会正常呈现，合成指标不构成科学结果。

本地完整证据保存在 `LOCAL_FIXTURES.json`，harness SHA256 `95f88a6eff08e36c10802ae2663a0cf4ebd52cd9a38fca1eda00a6f214d28d1a`。fixture 前后 17 项源身份不变。

## 冻结 SHA256

| 文件 | SHA256 |
|---|---|
| PLAN.md（1083 字） | `05963718ab8a59c647099511ae5d0290daf37ecb58b921b52ade85c8db5ffc0a` |
| adapter.py | `624109d019433ed1983bdc810063cdd5fe46116f0896c9dd8636214d553915d2` |
| train.py | `23b8d617c844ce2fc5b34fd3a82da95cd895ecb334c4be0ec3a4296863a3a777` |
| evaluate.py | `6776c9ed0492f7b171f5168fc6d851ee3a35e8d093f60a49790693b8c8713038` |
| frozen confirm/common.py | `3f2e4693488d7e5a35a182e5340edb9d73f5cafc45b7c0e61b1e2cc392a22d9a` |
| frozen confirm/train.py | `7fe8777db0ab8b2309140b39026a9bdbec42cd09ba9d2b8ef0c89b831a5641bd` |
| frozen confirm/evaluate.py | `afcb8a47dc9c303b3e7d558e4e91b50b560cd10dd0287956969d61ab27e13250` |
| frozen run_fashion.py | `f87e281acaf9a46a62eb31be588c9505ea7bb53c67162fd8fcf2ccc3440111d4` |

全部 17 项依赖指纹见两个 JSON。本交付说明不属于实验签名源，不改变已冻结源身份。

## 入口

本地快速检查已执行并通过：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python -B work/calibrated_fashion/local_fixture_check.py
```

由主任务决定正式启动；单 seed 入口如下（22/23 分别使用独立输出目录）：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python -B work/calibrated_fashion/train.py --partition-seed 20261207 --seed 20261221 --rounds 20 --backend mps --out work/calibrated_fashion/development_p20261207_s20261221
/opt/anaconda3/envs/Tinyproto/bin/python -B work/calibrated_fashion/evaluate.py --model-root work/calibrated_fashion/development_p20261207_s20261221 --out work/calibrated_fashion/final_eval_p20261207_s20261221
```

入口一次覆盖三个预定训练组；SGD lr0.03、batch64、本地 epoch1、20 轮。正式六 gates、跨种子和强校准对照结论留给正式运行，旧 STOP 原样保留。

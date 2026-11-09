> 当前状态（2026-11-04）：下文保留 helper 交付时的阶段记录，末尾“driver尚未接入”已被后续工作取代。本目录已有 `infer.py` 和 `reference_inference.py`；两任务固定三种子的官方 TEST 当前均为 DONE，见 [实际结果目录](/Users/wind/Desktop/Codex/paper/05_核心方法研发/results/predictive_final_test/)，主训练模型在 [实际模型目录](/Users/wind/Desktop/Codex/paper/05_核心方法研发/results/predictive_pfl_confirm/)。历史 root-lock 与结果继续作为来源证据，不重写其路径或哈希。
>
> driver 的历史 `ROOT/work/predictive_pfl_confirm/development_*`、`work/predictive_final_test/train_states_*` 分别需要映射到当前 `05_核心方法研发/results/predictive_pfl_confirm/development_*`、`05_核心方法研发/results/predictive_final_test/train_states_*`。CIFAR raw 应明确指向 `02_实验工程/dataset/Cifar10/rawdata/`，Fashion raw 指向 `05_核心方法研发/scripts/datasets/FashionMNIST/raw/`。这些文件存在，但本目录的历史相对路径并未自动接通；重新执行需明确路径适配与新输出，不能直接把下文阶段命令当作现成复现命令。便携复现按研究优先级后置，不是当前论文完善的前置条件。
>
> 已新增独立的当前路径适配入口 [replay_saved_inference.py](/Users/wind/Desktop/Codex/paper/05_核心方法研发/scripts/replay_saved_inference.py)，保留本目录冻结源码及旧锁的全部身份。其默认操作只核验模型、状态和路径，不读取官方 TEST 原始字节；`--execute` 才读取已指定图片并重放全部六个主模型，`--out` 必须指向新的 JSON。支持 `--project-root /Users/wind/Desktop/Codex/paper/02_实验工程 --method-root /Users/wind/Desktop/Codex/paper/05_核心方法研发 --fashion-raw /Users/wind/Desktop/Codex/paper/05_核心方法研发/scripts/datasets/FashionMNIST/raw`。本次 [重放记录](/Users/wind/Desktop/Codex/paper/05_核心方法研发/results/replay_saved_inference/result.json) 为PASS：120客户端、60,000张图片的 raw logits逐位一致，主方法汇总指标最大差3.67e-16；无训练、拟合、新种子或分数副本。参考方法与本地适配头不在这次重放范围内。

# Final TEST plan/helper handoff — CPU synthetic PASS

2026-11-04。本次只固定方案、实现独立helper与CPUtoy；真实TEST字节打开尝试0、CNNforward0、fit0、backward0，未创建root-lock。没有新增训练/seed或改变主方法。独立审阅方案见FINAL_TEST_PLAN.md；最终实际执行由root决定。

## 固定指纹

| 文件 | SHA256 |
|---|---|
| FINAL_TEST_PLAN.md | `a188e4e566e22804df7a08e89fd5ce5e1f23264be48eb1231330d3ca803f40e4` |
| data.py | `0366aa408b22220a4098be8aa14abb672c7480b264a77e95145b10c9ee918412` |
| scorer.py | `9bba7e92e142a6c6c101da99a26734257f0abcb9f46feb9c242ae7aacdf99041` |
| local_fixture_check.py | `9c9dcf3d359d0e211f9aea31781c100f6e87daa0637a7065da155be8c24b31e8` |

数据模块不导入torch/torchvision，不下载或扫描目录；import不读任何文件。仅official loader调用时先验证root-lock与execute，再打开规定TEST源。CIFAR固定test_batch官方MD5、受限pickle/uint8shape；Fashion固定两个t10k gzip官方MD5、IDXmagic/shape/payload长度。保存文件字节MD5/SHA和确切GT来源、原行号标签hash。

## root接driver的最小API

将本目录加入Python导入路径，独立导入data/scorer。raw路径沿用既有布局：CIFAR raw目录包含cifar-10-batches-py/，Fashion raw目录直接包含两个t10k gzip。

```python
images, labels, test_source = data.load_cifar_test(
    cifar_raw, root_lock=lock_path, execute=True)
# 或 data.load_fashion_test(fashion_raw, root_lock=lock_path, execute=True)

groups, assignment = data.assign_clients(
    labels, train_counts_20_by_10, task="cifar",
    train_manifest_sha256=frozen_train_manifest_sha)
batch = data.normalize_batch(images[batch_ids], "cifar")  # FP32 NCHW
```

assign_clients返回20个uint32 original-ID数组与完整manifest。整数largest-remainder低clientID破tie，每类PCG64(SeedSequence([20261301,classid]))；每类1000、全部10000无放回恰一次。manifest有配额、IDs、assignment/labels/TRAINcount hashes、TRAIN与TEST实际client类频率、差异/TV/maxabs。类总量和整数化造成prior变化，明确不叫exact TRAIN-prior matching。所有seed/method应复用同task同一assignment。空client直接报错，不重采样。

主driver负责final模型eval/inference_mode，只从现有TRAIN拟合部署state取值；本helper不导入模型或拟合。raw_logits必须为原ID0..9999顺序10000×10；每个client的head_logits按groups[i]顺序由该client固定TRAIN-adapted head推理，并同时传head_row_ids_by_client，防止形状相同但行序误配。

```python
report = scorer.score_export(
    fresh_out, task="cifar", seed=20261231, method="fedlc_uniform3",
    row_ids=np.arange(10000), raw_logits=raw_logits,
    head_logits_by_client=head_logits20,
    head_row_ids_by_client=groups,
    labels=labels, assignment=assignment, states=client_states20,
    test_provenance=test_source,
    input_snapshot={str(checkpoint_path):checkpoint_sha,
                    str(train_state_export):train_state_sha,
                    str(train_manifest_path):train_manifest_sha},
    root_lock=lock_path, execute=True,
    inference_cost={"cnn_forward_samples":10000,
                    "raw_forward_batches":raw_batches,
                    "head_forward_samples":head_forward_samples,
                    "inference_seconds":measured_seconds})
```

`input_snapshot`由driver在任何TEST打开前固定，列出所有实际checkpoint/TRAIN state/manifest/相关source文件；scorer前后复核每项。root-lock不另设复杂审批结构，仅root审阅后写小JSON并传--execute：

```json
{
  "status": "ROOT_LOCKED_FINAL_TEST",
  "tasks": ["cifar", "fashion"],
  "plan_sha256": "a188e4e566e22804df7a08e89fd5ce5e1f23264be48eb1231330d3ca803f40e4",
  "source_sha256": {
    "data.py": "0366aa408b22220a4098be8aa14abb672c7480b264a77e95145b10c9ee918412",
    "scorer.py": "9bba7e92e142a6c6c101da99a26734257f0abcb9f46feb9c242ae7aacdf99041"
  }
}
```

以上只是schema示例，不代表已授权或生成root-lock。本目录没有真正root-lock，fixture也不创建。driver CLI可直接使用--root-lock/--execute映射这两个参数。

## state契约与输出

20个client各有且仅有下列FP32一维数组：

| key | 维数/bytes | 解释 |
|---|---:|---|
| analytic_prior | 10/40 | local TRAIN log((count+1)/(N+10))，逐位核对TRAIN counts |
| prioranchor_bias | 10/40 | 原固定10维TRAIN拟合 |
| zero_anchor_bias | 10/40 | 冻结zero-bias |
| temperature_scaling | 1/4 | 正尺度a≥0.001 |
| bcts | 11/44 | a、10b，零bias锚 |
| zero_anchor_vector | 20/80 | A、b，零bias锚 |
| prioranchored_vector | 20/80 | 主方法TRAINprior锚 |
| prioranchored_bcts | 11/44 | 同prior锚机制control |

另从raw/head score生成prior_only、raw、head/head+prior。所有FP32部署参数提升FP64应用；没有optimizer FP64 theta替代。scorer只接受完整固定家族，不缺项、不添候选。共12个variant，与既有命名一致。

每个client NPZ保存IDs、GT、raw/head、全部FP32 state和全部最终FP64 scores；manifest保存每NPZ bytes/SHA。JSON给20client metrics、macro/weighted accuracy/NLL/Brier/ECE15、pooledECE及accuracyq10；固定ECE15最后bin闭合包含1。记录transform/metric/export与driver提供的推理cost，0fit/0backward明确列出，不称网络/能源测量。

单方法完成3seed后可`scorer.summarize_three_seeds(reports, task, method)`：校验正好固定3个seed与相同assignment，输出各variant mean/sampleSD(ddof1)和主方法相对各control的配对差；不返回gate或selection。root汇总全部3个训练method，不由scorer挑method。CIFAR TopK/FedRoD按各自冻结预测定义另表，不伪造不存在的calibratorstate。

## CPU验证

`LOCAL_FIXTURES.json`为PASS：1000synthetic/class共10000；配额/整数ties/缺类平滑/覆盖无重复/哈希重放/频率差异/emptyclient拒绝；仅内存CIFAR/IDX/gzip解析及畸形拒绝；两归一化；均匀/正确饱和/错误饱和概率边界和独立随机指标公式；完整FP32 state与12variant export。20NPZ共**1200000个score元素逐位重建**、metricerror0。root-lock/execute拒绝在任何TEST文件打开前发生。

模型推理driver尚未接入；该状态不能写成已运行finalTEST。此交付不读真实TEST、不拟合/重训，不扩大每任务3seed范围。

# Final TEST helper 独立只读复核

2026-11-04。复核PLAN、data、scorer及冻结TRAIN拟合公式，没有打开真实TEST字节/模型推理/新fit。发现的CIFAR10 MD5常数错误已在root明确授权后定点修复，并重跑CPUtoy。此审阅是本地代码/数值复核，不是外部模型审稿或真实finalTEST验证。

## 实质发现与修复

**P1已修复：CIFAR数据集MD5混淆。** 初版写入的f0ef6…属于torchvision CIFAR100.test，而非CIFAR10.test_batch。只读取当前安装的torchvision/datasets/cifar.py源码，CIFAR10.test_list第43–45行明确为 `40351d587109b95175f43aff81a1287e`；CIFAR100.test_list第160–162行为旧误值。data.py与FINAL_TEST_PLAN.md现已改正，README指纹同步。旧错误会在任何解码前MD5拒绝真实CIFAR10，但绝不能作为可用final loader发布。旧92ed/58f9指纹废止，没有真正root-lock依赖旧值。

当前冻结SHA：PLAN `a188e4e566e22804df7a08e89fd5ce5e1f23264be48eb1231330d3ca803f40e4`；data.py `0366aa408b22220a4098be8aa14abb672c7480b264a77e95145b10c9ee918412`；scorer.py `9bba7e92e142a6c6c101da99a26734257f0abcb9f46feb9c242ae7aacdf99041`。Fashion两MD5与冻结run_fashion.EXPECTED_MD5相同。官方TEST本轮仍未读取，不能声称实际文件MD5已通过。

## root-lock与加载边界

data.py:53起先检查execute=True、task和root-lock状态，再核PLAN/data/scorer SHA；official loader只在其后打开TEST。闭锁CPU检查四种调用均在TEST路径open前拒绝。真正root-lock由root审阅后生成，helper/fixture不生成、不隐式授权；不需要额外审批框架。

这个小JSON是本地root控制的执行意图标记，不是密码学签名或不可伪造权限边界。它只锁PLAN/helper/task，不锁每个checkpoint/state/GT文件；这些由driver在TEST前建立完整input_snapshot并前后核验。scorer只确认传入snapshot非空且所列文件未变化，不能自行知道是否遗漏了必要来源。driver须显式包含最终20轮模型、所有TRAIN state导出、TRAIN manifest、协议和本身代码，不能用无关文件满足这个参数。

synthetic=True是CPU fixture的公开测试接口，会跳过root-lock但输出scope=SYNTHETIC_CPU_FIXTURE；它不读取数据。生产driver不得传synthetic=True，也不得把synthetic报告拼进官方结果。

## assignment数学与实现

data.py中的weight=count+1、整除quotient、整数remainder排序以及clientID破tie与PLAN一致，避免浮点近似打破tie。每类独立PCG64(SeedSequence([20261301,c]))，原IDs按class筛选后shuffle，再按clientID连续切片。完整10000分配一次；20client、每类1000及实际quota检查明确。client为0样本直接拒绝，无重采样或模型评分参与。

TEST使用标签仅定义预先固定的client membership和最终评分；这符合新定义label-skew protocol，不能称原benchmark官方clientpartition。P(client|class)来自TRAIN count，但类总量1000和平滑/舍入使P(class|client)可与TRAIN不同；manifest实际保存频率差、TV/maxabs。主driver应跨全部seed/method重用同task一个assignment，不重新抽样。

scorer重算assignment_hash与labels hash并重放assignment规则。train_manifest_sha256字符串本身不解析TRAIN文件，无法独立证明20x10计数来自该manifest；driver须从冻结TRAIN manifest提取并核对TRAINcount hash。manifest包括runtime版本，跨环境manifest_hash可变；assignment_hash与实际IDs才是跨seed复用依据。

## raw列顺序、head行序与transform

raw必须10000×10、原row IDs0..9999；每client head同时提供显式head IDs，拒绝与assignment顺序不同的行。所有部署state只接受finiteFP32确切维数；analytic_prior逐位核对TRAINcounts；TS/BCTS正尺度下界保留。raw/head先升FP64，state也由保存FP32升FP64，再按z、z+b、az+b或A⊙z+b应用；zero-vector与main-vector同口径，没有用优化器FP64theta替代。

**driver必须补齐的语义契约：** helper只能检查10列shape，无法从数字证明第c列就是官方classID c。最终模型、head exporter及所有state必须明确列顺序 `[0,1,2,3,4,5,6,7,8,9]`，保存class_order及训练label mapping来源，禁止TEST准确率驱动列置换。否则一个10列排列错误仍可通过现有shape/hash检查。FP32 raw、head和state均应由同一frozen模型/task/seed输出；head dtype目前只检查shape/finite，生产driver应明确拒绝非float32/float64（尤其complex或object），不依赖隐式转换。

helper只验证传入head ID数组与分配一致，无法证明其数值确来自对应head；正确model.eval/inference_mode、相同预处理、head来源/ID附着、样本forward计数属于driver责任。既有10维bias、vector等冻结公式已核查sumTRAIN CE而非meanCE，ridge10；原10维max20，其余vector/BCTS/TS/zeroBias max50，positive scalar约束沿原实现。

**sumCE/TRAIN来源不能由state数值倒推。** scorer不重新拟合，也不检查优化轨迹/λ/拟合GT来源。fusion的TRAIN exporter和driver应验证所有state元数据为相同TRAIN original IDs、目标sumCE、ridge10、固定步数/约束、FP32最终态，及final模型训练round20，不允许missing-state fallback到TEST fit。原head是固定TRAIN一epoch，head/head+prior非在TEST学习。state字典shape匹配不构成这些实验来源的证据。

## 指标、汇总与实际局限

稳定softmax/NLL、Brier十类平方和、top-labelECE15最后bin含1，weighted-clientECE与pooledECE分开；mean/sampleSD要求固定3seed且同assignment。NLL值的loss别名相同。scorer不挑候选、不作新gate，原STOP不改。仅单个training method的all20client导出；driver还须验证每task全部3训练method×3seed结果完整，参考TopK/FedRoD另按其冻结预测口径报告。

源码CPUtoy覆盖每类1000合成行、strict in-memory解析与畸形拒绝、配额/缺类平滑/覆盖/频率差、两种normalize、uniform/perfect/wrong概率边界、随机独立NLL/Brier/ECE公式、12variants和20NPZ的1200000score逐位重建。修复MD5后CPUtoy再PASS，TESTopen=[]、CNN0、fit0、root-lock未创建。它没有测试真实官方pickle/压缩文件兼容性、GPU模型推理、真实state来源或最终性能。

导出目录在开始时拒绝已存在，但逐client写入不是全目录原子提交；中断可留下没有DONE result的部分目录，必须新out重跑，不能把不完整目录认作成功。inference成本由driver提供，若未传为NOT_SUPPLIED；scorer仅实测transform/metric/export，不把0fit等同零端到端计算/能耗。source/checkpoint/state在完成时复核，GT源文件在loader时记录SHA，driver应将真实TEST文件加入最终前后snapshot（有root-lock后才允许hash读取），保留精确GT来源。

结论：MD5问题已修正；helper足以承接root最小driver，但独立来源与列语义责任必须在driver/TRAIN exporter落实后，才能标记官方final TEST真实完成。本轮未把待实现责任伪称已验证，也未要求新训练、新seed或新研究方向。

## root infer.py追加只读复核

已逐项阅读root最小driver及TRAIN exporter契约。当前driver每task锁3seed×3method=9组主模型、180个clientstate及各export report/manifest/counts；两task合计18组。TEST文件仅在有效root-lock并冻结这些输入后加载，其原始SHA随loader provenance进入scoring_snapshot，scorer逐次与task完成前均复核。assignment在task开始创建一次，跨3seed/3method复用。raw原ID0..9999，head显式附着同client IDs。

实际CIFAR三组export只读核对：每seed60NPZ，report列出的export/file SHA与checkpoint SHA相符，train_counts==manifest.client_label_counts，三method config.data_manifest==export manifest；issues=[]。没有在此审阅加载checkpoint或执行真实训练模型。

缓存feature逻辑与SmallCNN classifier[:-1]定义一致；随机初始化模型每task257合成uint8行验证，driver raw与同批原model.forward逐位相同，重排feature后用相同Linear最大差CIFAR2.98e-8/Fashion1.86e-8，属FP32矩阵乘法舍入。shape257×64且raw/head FP32，无TESTopen、fit或正式训练模型前向。证据保存在INFER_REVIEW.json。

**P2已通知root待定点修：** inputs_for_task仅检查export报告status=DONE，尚未显式绑定报告task/training_seed/partition_seed/rounds/methods及exports/checkpoint SHA、manifest/counts。若完整导出误放另一seed目录而TRAINcounts相同，会被后续同count检查接受；应使用现有元数据校验后再freeze，无需新增框架或修改已运行exporter。class_order成本中保存[0..9]，原TRAIN dataset直接使用官方整数0..9且没有重映射；建议附训练label mapping来源。

**同轮root已修复上述P2。** 当前inputs_for_task已绑定export task/seed/partition/rounds/methods，核report列出的state SHA、checkpoint SHA/config signature、manifest/counts及export provenance。重新只读调用inputs_for_task(cifar)通过，222个输入=9checkpoint+180clientNPZ+其他source/config/report/manifest/counts；TESTopen=[]，未创建lock。当前infer SHA为 `ccc8e3bae5a2a940ba4cc319bbe31595f7bff52c9fc245965ad99dcc5e05df9f`。后续若driver变化，root完成后需再刷新最终包。

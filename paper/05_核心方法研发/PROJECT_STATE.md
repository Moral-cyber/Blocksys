# 持续工作目标与约束

更新时间：2026-11-04T20:27:03+08:00（Asia/Shanghai）。本页是当前状态的统一入口；历史阶段记录不覆盖用户最新要求。

**当前技术工作已完成，可审阅英文论文与配套证据：轻量化核心贡献已确定，真实 TRAIN—HTTP—本机 EVM 登记—聚合纳入—认证收据—EF 提交的完整系统闭环已执行，60,000 张图像的主模型推理已独立重放。英文稿已编译为14页，摘要116词，无 overfull。** 这表示当前研究与论文交付范围已完成，不表示已经投稿或获得接收。

1. **核心贡献：** 针对标签分布异构的轻量联邦预测，将客户端 TRAIN 类别先验作为可修正的本地正则化中心，在低比特共享训练和个性化状态无需上传的条件下改善客户端概率预测。贡献落在先验中心的设计与通信约束下的预测取舍；20维形式和 FedLC-inspired 是实例，不要求每个部件单独创新或必不可少。贡献依据见 `../05_核心方法研发/CORE_CONTRIBUTION.md`。
2. **完整系统链已执行：** 两个原 CIFAR 客户端、两轮完整本地 epoch，共22次梯度计算；4个真实新更新分别新编码，经真实回环 HTTP 传输并完成4笔 EVM 登记交易。128项集成检查和6项 EVM 调用检查通过。链确认要求成功交易、规范包含块和两个后继块，并核对 chain/contract/uploader 与完整绑定；SQLite 聚合纳入后，客户端验证绑定 session、challenge、上下文及结果模型哈希的 HMAC-SM3 收据，再提交 EF。链登记与模型纳入是两个事件。丢 ACK 重建服务/SQLite 后精确重试，不重新编码、登记交易、聚合或重复提交 EF。同一 double 配置下模型与残差对直接参考逐位一致。结果见 `../05_核心方法研发/results/chain_integration/result.json`；范围为真实执行合约的本机单节点 Ganache 开发链，不据此声称公链、多节点最终性或生产性能。
3. **英文论文已完成当前技术范围的写作与编译：** 主线为问题—机制—关键证据，保留 BCTS/CE 等必要对照；14页、摘要116词，编译无 overfull。入口为 `../03_英文论文/main.pdf`，采用用户指定的2026 BlockSys svproc模板准备2027稿。当前技术工作可审阅；作者/单位信息及最终 track 的匿名要求须由用户确定，再办理实际投稿。
4. **复现可行性已完成本机核验，优先级仍排最后：** 6个固定 checkpoint、120个客户端—种子组合、60,000张官方 TEST 图像完成独立 CPU 推理重放；raw logits逐位一致，argmax差异0，主汇总指标最大误差3.66e-16。没有训练、拟合或新增种子。结果见 `../05_核心方法研发/results/replay_saved_inference/result.json`。保留原实验与数据，不默认要求便携复现。

工作中心仍是：**好的想法＋部分有效的数据验证＋优秀的论文表述。** 不以文件数、实验数、审查数作为质量指标，不扩展无界搜索，不增加当前主实验的训练种子。

## 支撑主线的关键证据

| 范围 | CIFAR-10 | FashionMNIST |
|---|---:|---:|
| 固定官方 TEST 准确率 | 75.57 ± 0.60% | 92.97 ± 0.44% |
| 固定官方 TEST 宏NLL | 0.6446 ± 0.0138 | 0.1880 ± 0.0110 |
| TEST相对同维零中心NLL降低 | 0.0358 | 0.0182 |
| 开发诊断：正确/固定错配中心holdout NLL | 0.612447 / 0.683592 | 0.166384 / 0.210277 |

主实验每任务三个固定训练种子，±为种子样本标准差。机制诊断使用已有开发 TRAIN/holdout 数组，一次固定类别循环错配保持中心数值集合与范数；该配对结果支持此设置中类对齐信息的价值，不排除所有其他非零中心。诊断不读取官方 TEST、不调参、不改写旧准确率 STOP。独立实现重算120个客户端—种子组合，最大指标差4.44e-16；它不是新的最终测试或先验漂移证据。后续官方 TEST 推理重放只核验锁定结果，不用于设计或选型。

实际 codec 上传54,067 B，20个本地FP32预测值80 B；上行节省90.13%，计入稠密下行后的往返节省45.06%。每客户端FP32残差547,496 B仍需保留。本次链集成认证应用消息54,408 B/包；先前本地训练夹具54,410 B/包、历史夹具54,405 B/包，上下文字段长度不同，不能混用。HTTP应用流量与JSON-RPC body字节分别实际记录，时延为本机调用观测，gas为开发链执行消耗，不等于共识或生产部署成本。

既有协议62/62、历史载荷集成186/186、本地真实TRAIN闭环119项及本次HTTP/EVM闭环128项检查保留各自范围；6项EVM检查为真实 eth_call，不是额外挖出的攻击交易。固定全参与缺席时保持OPEN、pending和旧EF，不能推进下一轮；返回后完成同轮。永久掉线自动推进、生产部署、多节点链、大实验矩阵及先验漂移均为可选扩展，不挂起当前交付。外置HMAC/记录标识不称为权重嵌入水印，不证明诚实训练或防投毒。

中文论文文件已按用户要求删除，后续只维护英文论文。**当前技术必做项已清空；待用户提供的事项为作者/单位信息与最终track匿名要求。** 模板沿用已由用户确定，不声称已完成实际投稿或接收。

---

## 历史阶段记录（非当前任务状态）

以下保留早期候选、停止决定、阶段运行句柄和来源说明。旧“正在运行”、旧目录、旧主次目标及阶段完成判断仅描述当时状态；当前工作一律以本文件顶部和统一状态页为准。旧负结果不改写成成功，已完成的新阶段结果也不因旧记录而回退。

## 已知证据边界

旧稿采用Top-K与误差反馈，不足以声称独立原创异构感知算法。固定块与能量/novelty配额没有建立优势。旧TinyProto相关含泄漏的早期split不能用于主结果。清洁split来自官方训练集，官方测试集单独保留。

任何新候选先做机制诊断与验证集选型；保留负结果。科学新颖性和准确率优势都由证据判断，不预先承诺一定为正。

## 当前接续要点（接续更新，替代旧运行状态）

- 历史方向prefix .9的CPU40轮3配对全部DONE，last5验证准确率差+.20/.35/.46pp，均值+.3367pp；末轮差-.65/+.60/+.75pp。不再扩大此family。
- MPS开发10方法/参数均DONE，prefix .9 last5+.41pp；Lagrange .9是负结果。CPU/MPS不混池。
- 两真实EF共同状态与v1/v2训练梯度oracle诊断已DONE并独立复算；旧g下一步loss均差，真正global梯度的oracle收益不能部署或免费取得。
- 所有确认official_test_final仍未用于任何设计和选型；后继研发仅用开发训练/holdout。
- 桌面paper/06_核心方法研发已同步477文件，含raw/checkpoint/provenance/脚本与负结果，旧版本保留。后续新增还需同步。
- 新候选仅先做一次压缩后支持内数值修复A/B/C；固定alpha .03 theta .5、一批64TRAIN例、CPU1、samebatch与等计算普通SGD对照。C loss不同时胜A/B或acc<A则止损，不开网格。实现与执行由guarded_codec agent进行；这不是shadow CE本地训练另一候选。
- 新颖性未成立，FLARE/SA-PEF/AdaComp/RegTopK已核查，不能泛称残差陈旧或压缩感知是新；另一文献agent做有限support-repair近邻核查。
- 独立CIFAR索引/Fashion数据入口已核查，但跨任务主实验延期，未冒称训练。goal保持active。

## 2026-11-03 用户追加的止损要求

若当前路线不能成功，应及时换方向，避免不必要的token和时间消耗。总方向始终是FL轻量化处理下的数据异构性问题，不要求保住现有历史方向选择器。

据此收缩执行：不启动原计划的MPS大规模正式确认/跨数据集矩阵；先完成已在运行的CPU第三组TopK-vs-prefix固定确认。用现有验证轨迹与有真实EF的同状态诊断决定去留，不以官方test筛选后继方案。MPS开发基线已近完成，保留作后继候选的同协议对照。仅再做一次全训练目标梯度的oracle同状态诊断，以区分历史参考失效与选择机制缺少作用；oracle不作为可部署方法，也不当免费通信。

当前两组40轮CPU验证的last5准确率差约+.20/+.35pp、loss差约-.00228/-.00359；两个实际EF共同状态的下一步probe均不支持方向补位改善loss。这是收缩资源的已有证据，不把此修订冒充事前预注册。参考文献扩充与顺序修复仍排在主目标之后。

## 追加状态

三配对已经全部收尾，MPS开发也完成。旧历史方向family已停止扩展；完整止损证据见STOPLOSS_AND_EVIDENCE.md。新目标是一次可否证修复诊断，不同时开shadow CE或Fisher路线。

## 追加：第二候选也已及时止损

一次固定A/B/C诊断已完成：A TopK CE1.596964630/acc42.25%；B等计算额外SGD CE1.558726578/acc43.85%；C支持内修复 CE1.583601044/acc42.60%。C未同时胜A/B，按预先停止条件停止，不调alpha/theta/batch救。15fixture与60客户端记录postcheck PASS。CrAM已发现相近压缩点支持内梯度操作，宽机制也不能声称原创。

因此当前没有正在运行的长训练，也没有已成立的新核心算法。继续主目标需要先重新锚定一个具体的异构-压缩问题，再做有信息量的小诊断；不能把被否证的历史方向、修复或普通TopK+EF包装成创新。当前shadow CE与Fisher路线均未实施，不能把repair的结果当它们的结果。引用次目标仍保留在核心贡献之后。

## 新问题核查：时变聚合权重与EF记账

正在限量核查部分客户端参与、按当轮参与样本量归一化时，每个client的a_i^t会变化，普通local residual补传是否保留其原始global贡献份额。候选以global贡献单位保存误差，先检查代数/合成反例/固定权重退化与HT固定权重聚合竞争解释，查直接近邻，再决定是否做真实训练。该问题与统计样本量异构和客户端抽样有关，不以真实设备带宽为研究变量。尚未认定新颖、未运行真实学习实验。

## 时变权重候选最终止损

纯NumPy合成记账检查PASS：普通localEF多出权重变化内部项，global-unit EF消除该项；固定权重时等价。未运行真实学习训练。AFL-Com（arXiv2609.36610v1）§4.1/Alg2已明确同一time-varying weighted EF核心，因此不作为新算法继续；该方向的代数和来源材料仅作实现/理论对照。HT固定权重与公共有效步长重参数化也应作为简单替代。

主目标仍active且尚未达到。目前已经止损history scoring、support repair、global-unit weighted EF三条，下一阶段必须先重新提出具体可部署问题和对直接近邻可辩护的差异，再开训练，不能继续围绕这些已否证/已直接覆盖规则改名或开网格。历史官方test字段没有用于方法选型。

## 用户进一步明确的贡献标准（后续以本段为准）

用户明确允许针对具体问题合理组合已有机制，不要求算法原子部件或某层机制“首次提出”。因此，组件有先例不再自动构成停止条件；应通过准确引用、组合设计的理由、与直接拼接基线的公平比较及可靠实验来评价贡献。不凭会议级别保证接收，也不把完整照搬的方法当自己的算法。既有负结果保留；AFL-Com已有完全相同的weighted-EF核心仍不改名冒充新算法，但可作为正确实现部件。

用户提供的FedCSPACK部分覆盖思路仅是可参考候选，不要求选择该方向，一切以最终核心方法与可靠证据为主。文献核查发现论文与作者代码有差异，须分别标注。

## 部分融合诊断与新组合（当前阶段）

- 新CIFAR索引分区seed20261143、alpha=.1、TRAIN37500/dev12500；新初始化10轮dense共同warmup，只在相同TRAIN训练；官方test未读取。单共同状态共8种方法已完成。
- 本地保留macro acc67.5291%，直接稀疏含self覆盖54.7055%，排self覆盖46.6041%，排self保留最后Linear头65.8176%；排self gate72.0681%/CE.75327170；同额外梯度预算普通SGD73.1269%/CE.72091635。gate未胜强对照，按冻结规则停止该精确候选。
- 独立本地代码/数据/FP64聚合/summary核查PASS；外部跨模型审计REVIEW_UNAVAILABLE。第一诊断的字节为公式核算，非实际传输；其他client mask并集约31.17%–32.57%，不能说5%下载。
- 新候选为接收mask反馈用于下一轮上传包选择，主对比两组用完全相同接收优化，差别是是否使用跨轮feedback。52包+1070bit接收mask（134B加16B头）与53包基线在同应用字节上限内，记录实际序列化长度，无网络/链上测量。
- work/receiver_feedback/PLAN.md与run.py已冻结，20轮、3组、seed20261144闭环正在执行。先以last5宏CE及accuracy差>=.5pp决定是否复核，未达门槛就停止这条固定反馈规则，不调EMA/阈值开网格。尚非核心贡献成立。

## 部分融合与反馈闭环最终状态（2026-11-04）

两组已完成且独立复算。接收反馈last5宏acc78.870861%/CE.58020461，幅值门78.986920%/CE.57935099，本地额外SGD79.681190%/CE.56250887；反馈实际下载增加，warmup后总通信+12.47%。按冻结规则STOP，不开网格。FedCSPACK作者固定commit6bb6d9e已只读审查，论文部分mask下行未在该代码实现，代码实为完整覆盖；阈值全选无固定K，delta上传，abs(cos+KL)，这些差异分别归档。

标签支持强制分类头上传一次双状态诊断也STOP：两实际EF状态均没有同时改善TopK CE/accuracy，保留负结果，无长训练。

当前下一候选为FedNova局部步数归一×tensor-RMS稀疏选择的2×2，work/normalized_sparse/PLAN.md与run.py，fresh seed20261151，新索引partition20261143，20轮四组session46051正在执行。组件可引用组合；不能把FedNova已知归一收益称自己的创新。只使用开发validation，未访问finaltest。

## 归一组合2×2已停止

全部80轮完成、本地独立复算PASS：last5 topk32.63%/CE1.88037104，nova_topk31.07%/1.91227444，rms32.45%/1.89814724，nova_rms31.46%/1.91409614。组合未优于组件，停止此固定组合。RMS组偏置参数频繁零选择只是当前模型描述，不作因果或普遍否定。checkpoint只保留当前权重/残差及全部指标，计划中“每轮保存全部”不代表保留历史快照。外部REVIEW_UNAVAILABLE。

## 当前压缩感知训练候选

work/shadow_training/PLAN.md与run.py已冻结。3x20轮session95578，freshseed20261161，同新partition20261143、2000globaldev。标准SGD.03一步vs同批次两普通半步.015vs普通半步+压缩shadow支持内半步.015。新候选改变完整本地训练轨迹，不是已止损的末端单步repair；仍引用CrAM/QAT/FLARE近邻，允许组合不声称首创。旧残差轮内固定、终点TopK EF正常更新，无额外repair债务。单seed门槛shadow last5 acc同时超两对照1pp且CE均更低，否则停止固定配方。真实CPU小fixture已检查global模型不变与梯度次数；待独立核查。尚未核心成立，无officialtest读取。

## 压缩感知轨迹候选收尾，回到问题诊断

三组20轮全部DONE，本地独立复算PASS。last5 standard33.190001%/CE1.87633647，samebatch34.85%/1.84668556，shadow30.92%/1.93924196。shadow相对强对照-3.93pp，按冻结门槛STOP，不再调整支持/步长/残差比例。无核心贡献成立。

当前只跑问题重锚定的4baseline诊断，work/bottleneck_reanchor/PLAN.md及run_finite.py，session85202：固定同图像全集、各client样本数/训练步数、globaldev、seed20261171与partition20261143，mixing0/1 × dense/TopK-EF20round。要辨别压缩是否真的额外放大统计异构，避免继续无目标添加组件。不是新算法/主实验；官方test不访问。

## 问题重锚定诊断完成，阶段更新

四组20轮DONE，独立verify_results.py复算ALL_DONE：hetero dense35.44%/TopK34.40%，mix1 dense46.52%/TopK44.65%；compression CE代价.0379033 vs.0473025，accuracy代价1.04 vs1.87pp。因此当前单seed不支持“压缩额外放大异构损失”；dense本身有11.08pp异构差。后续研究动机不能继续按原先假定写。

新阶段work/mixed_precision/EXPERIMENT_PLAN.md与run.py：先做CE/明确FedLC-inspired校准变体×dense/TopK四格，freshseed20261181、同分区、20轮，session61990正在跑。校准的已知收益单独归属，不作为自己的创新。FedLC正文/附录分母不同，缺失类平滑与tau没有确认官方可执行规则；明确本实验全classCE(z-max(n,1)^(-1/4))tau1，rawlogits推理。

Stage2待codec冻结和Stage1结果：相同字节上限54808B下uniform3全坐标量化、unweighted mixed、training-grad-square proxy weighted mixed与TopK。实际wire解码参与聚合，scale/位宽/头部/padding计字节，EF正常；proxy不是Fisher/Hessian。是否保留校准由开发证据决定并标记探索性，不强留。FedFQ/FedMPQ已有bit分配先例；按用户允许组合标准，以直接基线和删除对照评价组合，不首创claim。

## 2026-11-04 Stage1/2全部完成，敏感性分配停止（最新）

Stage1 session61990、Stage2 session56613均已退出。独立复算ALL_DONE。
cal_uniform3 last5 CE1.8180330200/acc33.960001%；mixed_mse1.8197476868/33.909999%；
sensitive1.8198931610/33.790001%。敏感性未过冻结CE和+.5pp条件，STOP，
不调块长/代理/tau/bit网格。相对TopK的1.88pp来自已有简单量化，非新规则。
证明成立不替代学习结果，官方test未用于选型。

主目标仍active且未达到。正在限量评价仅分类头SCAFFOLD控制+精确头传输+body低比特，
尚未定方法、尚未训练；已有控制变量部件需准确引用。
独立SQLite持久化完整性组件正在完成真实本地测试，非真实链/网络实验。
材料整理代理补portable mixed_precision入口与primary来源归档。

## 头子空间控制四组开发启动

work/head_control/PLAN.md与run.py已冻结，session26690，freshseed20261191。
plain_uniform / exact_head / head_control / head_control_dense，四组各20轮；
前三组相同上传cap54808B，后两组same exact头+mixed_mse body，隔离校正学习作用。
controlstate650FP32=2600B，额外下行2616B计入，body EF、actualK、固定p不变量均记录。
SCAFFOLD、SCALLION、FedCOMGATE有直接prior，零control上行不作为新意。
未定位完全相同head/body组合不代表不存在；当前只是探索性有限开发，尚未核心成立。

## 2026-11-04 最新接续

头子空间四组session26690已exit0；候选last5 accuracy35.950000%，
plain_uniform和exact_head均约35.79%，差+.16pp，未过冻结+.5pp条件，
停止此固定头校正规则，虽CE下降不更改门槛。完整独立统计记录正收尾。

下一步仅做已有body/full SCAFFOLD强参考诊断，先定位有效学习纠偏是否需要主体，
不是新方法，不复活原head候选或开网格。还没有成立的新核心贡献。

后半work/durable_integrity_v2正式及独立重跑62/62通过，188 SQLite完整性ok。
修补同name不同spec误配：packet/活动配置/注册表specSHA三者匹配后才decode。
负对照明确相同spec谎报下错误decoder仍可执行，不证明执行正确或防恶意持keyserver。
全部为本地simulation_only，无真实链/网络，外部跨模型REVIEW_UNAVAILABLE。

## 已有强控制参考启动（当前活句柄）

work/control_reference/run.py/PLAN.md，CPU合成与checkpoint恢复fixture PASS，
MPS session7854，三组plain_dense/body_control/full_control各20轮，seed20261191。
这是已有SCAFFOLD与单向量重建已有SCALLION先例，不是新方法。
固定同TRAIN内部dev和batch流，官方test未访问。head-only STOP不变。

## 强控制参考完成，转有限标签曝光诊断

三组20轮session7854 exit0，独立444/444通过。plain35.82%/CE1.79850328，
body33.09%/1.84557452，full31.57%/1.90330075，当前protocol不继续扩该控制组合。
不外推SCAFFOLD整体（单seed/分区、20轮、lr.03、epoch1）。

work/exposure_prior/PLAN.md和probe.py：session59490，仅对5个已冻结最终ck做
raw/pooled→uniform/effective→pooled/反方向后处理；counts只来自TRAIN，
实际pK造成class6有效prior.0657 vs pooled.0996、class7 .176 vs .100。
不以该近似冒充真实完整训练后验；无系数网格、不训练、不读取officialtest。
当前无新核心成立。

## 最新：有效先验后处理停止，训练期校准组合新分区开发

冻结状态probe session59490 DONE。hetero dense36.4→32.8%且CE变差，
TopK34.9→32.85%且CE变差，按原条件STOP。pi_eff是曝光代理不能当训练后预测先验；
不反号或调系数救。保留5模型raw logits/GT开发标签、counts、摘要和独立复算。

新work/balanced_quant/PLAN.md/run.py，CPU objective/gradient/codec fixtures PASS，
session73047，4x20轮，fresh partition20261193与seed20261201。
CE+uniform3 / FedLC-inspired+uniform3 / Dirichlet1-smoothed balancedCE+uniform3 / balanced+dense。
已有balanced softmax/FedRoD训练目标与量化/EF准确归属，不要求另添自创评分器。
按用户组合创新标准评价整体实用价值；未成立新核心、未访问officialtest，仍为探索性。

## 最新：全局balanced组合完成，PFL诊断启动

73047已DONE，balanceduniform3 last5 35.7200%/CE1.78141794，CEuniform35.41/1.78098614，
FedLCuniform35.84/1.77416819，balanceddense35.79/1.77891197。候选未过原CE与+.5pp条件，
STOP此固定全局组合，待独立统计归档；不称超过FedLC。

work/prior_personalization/PLAN.md/probe.py，session21309，既有4个最终模型的
PFL客户端12500holdout开发诊断（含global2000dev，非新独立test），
raw/prior/prioronly/额外本地1epoch只训练最后Linear/learned+prior五对照。
prior只取TRAINcounts；量化+balanced+analyticprior无需额外梯度，强learned头有额外计算。
明确不将PFL/local指标混写IID/global，不新训练FL、不访问officialTEST。

## PFL固定解析先验已完成，新校准诊断筹备

21309已exit0。balanced quant+prior宏acc77.4974%/CE.5948502，
CE+prior76.9832/.6031309，FedLC+prior77.3418/.5986492，
候选胜CE+.5142pp但仅胜FedLC+.1556pp，未过两者+.5pp条件，固定规则STOP。
learnedhead为额外TRAIN一轮lastLinear训练的强对照，不等同完整FedRoD。
新分区实际TRAIN37502/holdout12498，为逐client round(.75n)偏差，四组完全相同，
旧PLAN名义37500/12500不追改，单列protocol WARN。

新prior_bias仅筹备：冻结generic模型，每client一次TRAINforward，10维bias
以loglocalprior为anchor解强凸Newton校准；无CNN反传、零extra uplink，
额外forward/solver/localstate真实计入。不能把旧解析规则改名复活，先有新PLAN/fixtures。

精确访问边界：历史torchvision train=True可在MD5完整性检查读test_batch字节，
不载入test记录或用于推理/选型；见OFFICIAL_TEST_ACCESS_SCOPE.md。后继只解包五TRAIN文件。

## 当前活句柄：prior_bias一次前向诊断

work/prior_bias/PLAN.md/solver.py/probe.py，session3337 CPU1。
C=10、10维prioranchor bias、最多20Newton，CPU解析梯度/Hessian/Armijo/gauge fixtures通过。
四模型各37502 TRAIN forward，零CNNbackward、新FL训练、extra上传，40B/client localstate。
严格只读5官方TRAINbatch并校验hash；现有clientholdout只在fit后评估。
候选固定balanceduniform，不按结果换loss；达标也仅FOLLOW_UP，未达STOP不网格。

## prior_bias原固定候选完成STOP（不得翻案）

3337完成。balanceduniform+bias宏acc77.81244%/CE.5925038，
vs同prior+.315pp，vsbest existingprior+.214pp，未过+.5pp，原规则STOP。
4×37502=150008 TRAIN imageforward，0 CNNbackward/extra上传，每client40B。
strict官方TEST字节尝试0。独立保存数组核查正在进行。

观察到FedLC+bias78.08935%/.5871585，相对其prior+.74756pp，
CE+bias77.94111%/.5873895。此是事后看到，不能换candidate翻原结果。
若继续，需新明确探索性目标和未用过分区/seed确认，不把旧开发结果当确认。

## 当前真实运行38357，新分区探索性确认

work/calibrated_lowbit_confirm/PLAN.md1356字与common/train/evaluate已冻结，
CPU合成fixture全部PASS，train SHA7fe8777d...，session38357。
p20261203s20261211，CEuniform/FedLCuniform/FedLCdense三组各20轮。
新strictloader仅5TRAIN文件，actualcounts签名，真实量化wire聚合+标准EF。
最后checkpoint后统一PFLraw/prior/bias/matchedhead评估，原balanced STOP保持不变。
通过只FOLLOW_UP，不等于核心成立；后续需zeroanchor/vector/BCTS/2seeds/2ndtask。

## 新分区第一种子实际通过继续门槛（非核心完成）

训练38357、eval26560均exit0。独立1505/1505核验，6gateTrue，FOLLOW_UP。
FedLCuniform+bias75.712447%/CE.643701922，同prior74.4590/.665378883，
差+1.2534165pp、CE−.02167696；vsCE+bias+.09824pp，vsdense−.03553pp。
37501train/12499holdout，仍为PFL开发。bias无CNNbackward/extra上传，
额外forward与matchedhead1788梯度完整计；公式原子归属已有calibration族。

当前10285第二seed20261212训练三组20轮，8144新confirmsavedarray强calbaseline，
新Fashion入口筹备，所有旧STOP不变。尚无正式最终test、多seed稳定或跨任务核心结论。

## 接续：第二种子未过原门槛，第三种子在跑

训练10285和评估53685完成。seed20261212固定候选75.91819465%/CE.64206443，
同prior75.43509684%/.65017537，差+.48309781pp，略低于冻结+.5pp，
本seed效果STOP；其余5gate true。独立1505/1505产物检查PASS不等于效果过门槛。
不降低门槛、不以第一seed覆盖。39380完成已安排的第三训练seed20261213；
14937仅savedarray校准对照。Fashion只读TRAIN严格wrapper/fixtures完成，待root启动，
当前尚未跑Fashion正式训练。强vector准确率vs十维bias CE/状态取舍如实保留。

## 三训练seed完整，固定十维规则未稳定达门槛

同一新TRAIN内部划分p20261203，三个训练seed，非三个分区或最终测试；固定20轮，不挑checkpoint。

|seed|宏准确率%|宏CE|相对同loss固定先验pp|原六门槛判定|
|---|---:|---:|---:|---|
|20261211|75.71244730|0.6437019223|+1.25341653|FOLLOW_UP|
|20261212|75.91819465|0.6420644328|+0.48309781|STOP|
|20261213|77.06340708|0.6106028485|+0.16907084|STOP|

准确率配对差均值0.63519506pp、样本SD0.55794406pp，仅描述训练随机性，不能补订平均门槛救回。三个seed交叉熵均降低；两个seed未达原+0.5pp，固定准确率贡献不成立。数据与方法不改，负结果保留。

当前继续已启动的Fashion第1seed迁移边界；独立新候选仅做TRAIN输出数组上的先验锚定vector固定诊断，不复活旧规则、不网格。


## 新研究问题：先验约束vector的预测损失价值（不是原准确率翻案）

旧20d诊断原gateSTOP：与10d acc+.08574pp未到+.5，但同维zero-vector
acc+.00258pp、NLL.612422→.577094。root明确纠正机械要求每一组件都acc+.5的过宽标准，
新问题选自旧结果，须新未使用划分确认，不能改名宣布原目标成功。
work/predictive_pfl_confirm/PLAN.md SHA703d5956...和PREREGISTRATION预先固定
p61217/seed61231/32/33、Fashionp61219/seed61241/42/43、主macroNLL及accuracy非劣界，
firstseed强对照达标再做其余两seed。method不调参，10/20dim旧STOP不变。
4567正在新CIFAR三训练，Fashion旧p61207已eval81884完成，97.5676%/.0828981，
vsprior+.4451pp，仍未过旧accuracygate，不扩其旧规则。37870仅补旧Fashion强校准。

## Prospective CIFAR首seed达到新NLL继续条件

新p61217 s61231三训练4567、baseeval91521、vector6146 DONE，
assessment 10true FIRST_SEED_FOLLOW_UP：77.02376785%/NLL.5992794846，
vs同维zero-vector acc+.19243786pp/NLL−.03967008；prior+1.31988943pp/NLL−.02851764，
10dim+.50708807pp/NLL−.01303082；dense−.00761812pp/NLL+.00063186。
未改旧accuracySTOP，新目标PLAN703d...及assessbdb0...在评分前冻结。
head_control_literature正独立核查。按firstfollowup启动二seed44514；Fashionfresh首seed78531，
两独立MPS进程短暂并行，墙钟只作有竞争观测，不作隔离速度结论。

## 用户强调最终交付，进入收尾而非扩大研究

每任务3seed不增；CIFAR三seed已支持预注册NLL问题。Fashion剩2/3，TopK与FedRoD强参考配齐同3；最终TEST读取前锁定源/模型/TRAIN部署state。开始英文/中文方法稿和最终测试入口。goal卡片get_goal显示blocked，但现实无资料/权限阻塞，用户明确继续最终目标，本轮工作继续，现有工具不能resume卡片。不标complete、不停止实际工作。


## 2026-11-04T11:37:32.852471+08:00 最终固定范围完成

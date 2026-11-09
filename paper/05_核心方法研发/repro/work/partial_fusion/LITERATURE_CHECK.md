# 部分融合候选：有限原文核查

核查日期：2026-11-03。范围冻结为 FedCSPACK 锚点及 FedALA、FedCAC、FedLP 三篇可读 primary paper；PerFedMask 做了有限定位但未取得 primary 全文，因此不对其机制下结论。未训练、未核对作者代码，也不复活已停止的三条路线。用户已明确允许合理组合已有机制，以下已有先例用于准确引用和设定强对照，**不构成自动停止理由**。

后续作者代码核查（2026-11-03）：见 `official_fedcspack/CODE_CHECK.md`。固定公开commit实际下行仅完整global覆盖，没有论文描述的mask下行部分覆盖；下述FedCSPACK部分描述的是论文文本，二者不可混同。

## 结论与可测试的精确差异

“mask 上传 → server 聚合反馈 → client 部分融合”已有直接先例；receiver 用本地梯度决定接收量也已有直接先例。仍可测试的具体问题是：**在固定稀疏上传预算下，从其他 client 的可用包构造候选反馈，由接收 client 判断候选位移是否有利，并检验这种组合是否胜过直接覆盖、保留输出层和同计算量本地更新。** 这里的有利是 receiver 本地训练目标支持，不应把 server 观察到的覆盖或权重当作通用质量证明。上述完整组合未在本次四篇可读原文中出现；这只限定了与这些论文的差异，尚不能 claim novelty、异构性收益或长期收敛收益。

## FedCSPACK：mask 与融合到底做什么

来源：[arXiv:2601.01840v1](https://arxiv.org/html/2601.01840v1)，*Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack*；本地既有 `../literature_followup/fedcspack.html`、`.txt`。原文 `p2.1` 给出作者代码地址 **https://github.com/NigeloYang/FedCSPACK**；此次只确认链接存在，未访问仓库。

1. **client 决定上传范围。** “Top-k Based Cosine Parameter Packing”，式(3)(4)：先算整模余弦 `θ_a=cos(FW_i,FW)`，再算包余弦 `θ_i,j=cos(PW_i,j,PW_j)`。式(5)及周围文字选择 `θ_i,j<θ_a` 的包。未分享包留在本地。引言另称选 “highest similarity”，与方法段方向不一致；不能默默替作者统一。
2. **client 决定上传权重。** “Mask Double Weight Aggregation”，式(7)(8)：`β_i,j=Σ PW_i,j log(PW_i,j/PW_j)`；选中处 `M_i,j=θ_i,j+β_i,j`，未选处0。它是含位置和实数权重的 mask，非单纯二值筛选。判据是参数空间 proxy；没有 server 另用验证数据判“优良包”。
3. **server 对齐并归一聚合。** 式(9)前段令 `M_j^{t+1}=Σ_i M_i,j^t`；式(9)按 `M_i,j/M_j` 聚合上传包，未覆盖位置为0；紧邻文字把该聚合结果与旧 global 包相加。server 不在这一步为不同 receiver 验证反馈质量，也没有排 self 的个性化聚合。
4. **下行是部分应用，不等于稀疏传输。** Alg.1 第8行向所有 client 发送整个 `W^{t+1},M^{t+1}`。“Top-k...”首段说 client 仅在 `M[j]=Valid` 处执行 `PW_i[j]←PW_global[j]`，其余本地包保留。这是共同 global mask 引导覆盖，未用 receiver 本地 loss/gradient 决定是否接收。

复现边界：式(7)直接对参数包写 KL，附近未交代正值化、概率归一或平滑；Alg.1第15行虽计算 ΔW，式(3)–(9)仍用模型参数包，随后又写“加回”global。连同上述选包方向，必须查作者实现后才能声称 faithful baseline。当前按 delta 能量选包、传模型值、coverage 平均的诊断只是受其启发，不能标为原论文复现。

## 三个已核实直接邻居

| 方法及原文定位 | 上传 / server | client 接收与“优良”判据 | 与当前组合的关系 |
|---|---|---|---|
| [FedALA, arXiv:2212.01197v4](https://arxiv.org/html/2212.01197v4#Sx3.SSx2)，Method / ALA 式(3)–(5)，Alg.1，Appendix A式(7) | 上传完整本地模型、FedAvg聚合，下发完整global（Alg.1第7、20、21行） | `w_init=w_i+(w_global-w_i)⊙W_i`；最高p层学 `[0,1]` 逐元素 W。式(5)在本地子集 loss 上学习；附录式(7)给出 `∇_W L=∇_{w_init}L⊙(w_global-w_i)`。判据在 receiver client | “本地梯度×候选位移来调整部分接收”已有先例。原文没有 sparse donors、真正排self或独立批回溯，但换成block二值gate并不使这一思想成为首创 |
| [FedCAC, arXiv:2309.11103v1](https://arxiv.org/html/2309.11103v1#S3.SS3)，§3.1、§3.3式(5)(6)、§3.4式(7)–(11) | client 用 `|Δw·w_final|` 敏感度，每层top-τ得critical mask；上传完整模型+mask。server按mask关系选协作集合并生成global `w_bar` 与定制 `u_i` | 式(11)：`w_i^{t+1}=u_i⊙M_i+w_bar⊙(1-M_i)`。client 的mask表达本地重要性；server判断协作资格，均未检验该候选位移对receiver当前loss的改善 | 完整覆盖“mask→server个性化反馈→client部分融合”。式(8)的协作集合虽排i，式(10)又把 `{i}` 加入平均，故不是排self。mask并不使模型值上传稀疏 |
| [FedLP, arXiv:2303.06360v1](https://arxiv.org/html/2303.06360v1#S3)，§III-A式(2)(3)、III-B式(4)/Alg.1、III-C式(5)/Alg.2 | 式(3)已有支持集归一：`w_bar^l=Σ_k I_k^l ω_k w_k^l / Σ_m I_m^l ω_m`。同构版按Bernoulli概率选上传层，非本地质量筛选 | 同构版下发完整global覆盖。异构版共享前L_k层，个性输出层既不上行也不被下行覆盖（III-C式(5)后段） | coverage归一与保留private output head都是已知部件；其基本规则未对receiver候选位移作梯度质量门控。异构版还有模型结构裁剪，不能等同于当前同结构保留head基线 |

FedCAC 文本疑点：§3.4 把 `||M_i-M_j||_1/(2n)` 称为 overlap，又用 `≥Threshold` 选择“更相似”client。HTML和同版本PDF一致。此次不替作者修正方向，不据此复现协作集合，未审计代码。FedALA / FedCAC 的详细摘录和来源在 `ALA_CAC_CHECK.md`、`ALA_CAC_SOURCES.json`。

PerFedMask：FedCAC参考文献给出 Setayesh、Li、Wong，*PerFedMask: Personalized Federated Learning with Optimized Masking Vectors*，ICLR。此次 arXiv 缩写检索0条，Crossref缩写检索0条；题名关键词arXiv前20条未发现目标；OpenAlex前3条未定位目标；OpenReview按准确题名读取返回HTTP403。以上是**定位/访问未成功**，不是不存在或方法缺失的证据；FedCAC对它的二手描述不作本报告机制结论。

## 组合式基线与后续必要对照

当前 `EXPERIMENT_PLAN.md` 的单状态探索可用于决定是否继续该具体组合；它不能替代端到端PFL和论文基线。若通过既定去留条件，下一阶段应优先做少量可解释对照：

- **同一 sparse candidate + FedALA式本地融合。** 在相同上行包、同一候选来源下，对比原计划的block二值一阶门与本地loss学习接收权重，清楚标成“稀疏候选+FedALA融合”的改编组合；报告额外forward/backward和实际时间。还需准确原版FedALA作非等通信预算参照，不能把改编版冒充原版。
- **直接覆盖 / 保留head / 本地SGD。** 原计划已有这些对照；它们分别隔离接收门、本地输出层保护、额外计算本身的作用。前两项已有FedCSPACK/FedLP式先例，应引用。仅gate获益但不胜同计算SGD，不能解释为外部知识有效。
- **排self、门控、独立批回溯逐项删除。** 同一候选上至少比较含self与排self、仅回溯无gate、gate固定步幅与gate+回溯，否则无法辨别收益来自何处。排self只移除receiver当前上传值，其共同warmup历史仍共享，不应称完全独立外部知识。
- **FedCAC作为反馈结构的直接邻居。** 若后续贡献强调server定制协作，应查作者实现并加入准确FedCAC；若强调固定字节的稀疏反馈，则把“FedCAC式反馈+相同稀疏预算”的改编组合作补充对照，明确它与原版完整上传/双模型下发的差别。无需在本次单状态诊断中展开全部算法。

进入闭环前先核清 FedCSPACK 作者实现，上行模型值/增量、未覆盖位处理、真实mask/索引字节以及下行广播字节必须一致。凡准确实现/资源预算尚不可比，报告为启发式组件或改编baseline，不声称严格复现。所有已知机制可以合理组合；要证明的是具体组合对该问题的增量价值。

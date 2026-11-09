# Head-only control：有限近邻核查

截至 2026-11-04，本轮完成 **3 次精确检索**及已定位原论文读取；未定位到可确认的“仅最后 Linear 650 参数用 SCAFFOLD、head 不量化、body 低比特加 EF”的完全相同方案。**未找到不等于不存在**：只审阅检索返回的前 15/20 条及下列原论文，不能据此宣称首创。

| 原始来源 | 已核实内容与候选边界 |
|---|---|
| [SCAFFOLD，ICML 2020](https://proceedings.mlr.press/v119/karimireddy20a.html)，Alg.1，PDF 第 4–5 页 | 局部方向 `g+c−c_i`、Option II 的 `c_i'=c_i−c−delta/(ηK)` 均为已有公式。原算法维护全模型控制变量；所读算法未提出本候选的 head/body 分工。 |
| [Huang 等，SCALLION/SCAFCOM，arXiv:2308.08165v2](https://arxiv.org/html/2308.08165v2#S4.SS2)，§4.2 Eq.(4–6)、Alg.1–2 | 已明确给出 SCAFFOLD 的等价单向量上行实现：模型差分与控制增量可互推，上行减半；SCALLION 用无偏压缩，SCAFCOM 用动量处理有偏压缩。**零额外 control 向量上行已有直接 prior**。两者压缩全模型增量，不能直接等同于本候选的 body EF。 |
| [FedCOMGATE，AISTATS 2021](https://proceedings.mlr.press/v130/haddadpour21a.html)，§4、Alg.2、Remark 6，PDF 第 4–5、7 页 | 已组合局部梯度跟踪与压缩；Remark 6 明言无需 control variate 上行、下行因跟踪增加。它并非 SCAFFOLD 的同一递推；不能把已有单上行结构当候选新意。其采样版本有实验，正文未给采样收敛证明。 |

候选可定位为**按参数块配置已有纠偏与压缩的组合方案**：650 维控制状态与下发各为 2,600 B/客户端；head FP32 上行本身亦为 2,600 B，必须纳入同一预算。若 `c=Σp_i c_i` 且 `p_i` 固定，服务器可从已收 head delta 计算

`c' = c + Σ_(i∈S) p_i [−c−delta_h,i/(ηK_i)]`。

该不变量要求使用全局 `p_i`，不能换成当轮归一化权重；`K_i,η` 须已知或计入元数据。body 持续变化使 head 梯度也变化，不能直接继承完整 SCAFFOLD/SCALLION 的收敛结论。本核查未运行训练，不判断效果。

检索端点与原始响应位于 `sources/queries_manifest.json`：arXiv 邻居合并查询返回 1 篇；OpenAlex 宽查询共 12,156 条、读取前 15 条；带 federated/head-only 限定查询共 126 条、读取前 20 条。后两者含背景性提及，命中数不能当作精确同方案数量。未使用 Crossref，已按 paper-lookup 阅读其接口说明；现有来源已满足本次有限核查。

# 混精度近邻核查（2026-11-04；仅两篇原文，未训练）

结论：按重要性分配位宽、预算约束混精度均有直接先例。当前 grad² 加权块失真可作为组合方案检验，不能称混精度 FL 或敏感性分配首创。

**FedFQ**：[原文](https://arxiv.org/pdf/2408.08977)，2024预印本。第3–4页 Eq.(12),(17) 对本地更新 h=θ_local−θ_global 定义 q_f=Σ_j d|h_j|²/(4^{b_j}‖h‖²)，并在 Σ_j b_j=B 下最小化 q_f，即量化方差上界。Algorithm 1 用 CGSA 分配坐标级 {0,2,4,8}bit，优先为幅值大者加位宽。已有“更新幅值平方决定精度”的明确依据；它用的是更新值，不是 minibatch 梯度平方。随机无偏量化的理论不能直接移给确定性 min-max+EF。

**FedMPQ**：[CVPR2024原文](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Mixed-Precision_Quantization_for_Federated_Learning_on_Resource-Constrained_Heterogeneous_Devices_CVPR_2024_paper.pdf)。第4页 Eq.(11)–(13)：L_local=L_task+λΣ_l(M_l/M)Σ_i‖B_{l,i}‖₂；δ_{l,i}=‖B_{l,i}‖₀/M_l，若最高有效位 δ≤ε 则裁剪该位。第6页 Algorithm 2 按 m⊙(Δb+1) 降序确定 pruning/growing 顺序，围绕按参数数加权的平均位宽预算调整。正文把保留位宽解释为层敏感性，但没有用 grad² 或 Hessian/Fisher 评分。它包含低位宽本地 QAT、位平面稀疏正则、服务器再分配，且聚合权重 p_n∝v_n|D_n|，与只压缩全精度本地更新有不同目标和训练过程。

**对当前候选的归属。** s_g=本地梯度平方均值，最小化 Σ_g s_g‖u_g−Q_{b_g}(u_g)‖²（u=Δ+e），256坐标块、2/3/4bit、按失真减少/实际增量字节 greedy，可以称训练信号代理加权的预算混精度+EF。两篇未给出这个完全相同的规则；这不构成全领域首创证明。grad² 是二阶矩代理，不能叫已验证 Fisher、曲率或损失增量保证。

**公平对照。** 保留 uniform3、unweighted mixed（s_g=1）及 TopK5%，全部使用同一完整坐标 EF、模型/切分/训练/聚合/seed。以同一实际 bytes cap 比较，并报告各自真实使用量；不足预算无需填充。计入 bitpack、位宽表、尺度/零点、头部及padding，TopK计入索引和值；不能用 Σb 或平均3bit代替封包字节。校准CE单列因子，无额外 dense 分类头豁免。可再加同256块/同量化器/同字节约束的 FedFQ-inspired Σ_g‖u_g‖²4^{−b_g} 分配，明确是适配对照而非精确复现。FedMPQ应作方法定位；若要复现则须保留其QAT等组件，不能只借其名字标注本方案。没有原文支持当前组合必然胜出。

# 曝光先验：有限近邻核查

3 次检索、3 篇原文；未训练或读数据，不判断效果。

- [FedNova](https://arxiv.org/html/2007.07481)：§4.1 Eq6 及 vanilla SGD 段明确 `w_i=p_iK_i/Σp_jK_j`。候选 `π_eff,c=Σw_i(n_ic/n_i)` 是已有有效目标权重的类别边际推论；FedNova 通过训练期更新归一化处理目标不一致，原文未见该类别先验的后处理修正。
- [FedLC](https://arxiv.org/html/2209.00189v2)：§4 Eq4 已写 `f_y−logγ_y` 的先验校准判决；最终方法是训练期 local CE 的类别margin校准，非实际 K_i 加权的源先验后处理。
- [FedNoRo](https://www.ijcai.org/proceedings/2023/0492.pdf)：§5.3 的 FedLA 指引用 Menon-LA 的 FL baseline；§4.1 明确用 local prior 做训练期 `f+logπ`。FedLA 有重名，须明确所指。

**定位**：`logits+logπ_pool−logπ_eff` 属于经典 prior-shift 修正；具体组合是把 FedNova 型有效权重用于源先验代理。本轮未定位精确同方案；未找到不等于不存在。

π_eff 是更新曝光代理，未被证明等于压缩模型的预测先验。该修正还依赖 logits 近似源后验、类别条件分布稳定、先验为正及目标确为 pooled 分布；末批、局部漂移、压缩/EF 均可能破坏近似。按实际 K_i 记账有意义，但不是效果证明；π_pool 近均匀也不消除 p_iK_i 曝光差异。

原始响应与定位见 `sources/`；未用 dev 或拟合校准器。

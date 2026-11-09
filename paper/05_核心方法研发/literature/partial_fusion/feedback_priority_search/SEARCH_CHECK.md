# 一次有限检索：下行反馈影响下一轮上传支持

日期：2026-11-03。遵守一次查询上限：只调用了一次 arXiv search；随后只读该次命中的一篇 primary paper，不扩引文、不读 repo、不训练。已读 paper-lookup 及 arXiv API reference。

## 查询记录

- 数据库：arXiv Atom API。
- 查询：`all:federated AND (all:sparsification OR all:sparsity) AND (all:feedback OR all:mask)`。
- Endpoint：`https://export.arxiv.org/api/query?search_query=all:federated%20AND%20%28all:sparsification%20OR%20all:sparsity%29%20AND%20%28all:feedback%20OR%20all:mask%29&start=0&max_results=40&sortBy=relevance&sortOrder=descending`。
- 返回：总计26条，本次取得26条。原始 Atom XML：`arxiv_raw.xml`；解析后的全部题名、摘要、版本日期：`parsed_results.json`。
- 这是一个受控词项检索，不能证明所有相关工作已覆盖。并非逐字匹配“next upload priority”。

## 明确命中的近邻：TCS-H

[Sun et al., *Time-Correlated Sparsification for Efficient Over-the-Air Model Aggregation in Wireless Federated Learning*, arXiv:2202.08420v1](https://arxiv.org/html/2202.08420v1#S2)。原文 `tcsh.html`、含完整数学alttext的 `tcsh.txt`。

**§II，Algorithm1第3、8–11、14–15行；同节“2) Local Model Compression”**给出精确循环：

1. server/BS 广播上一轮聚合位移 `g_hat_{t-1}`（第3行），并非另发一个二值mask。
2. 每个client从该反馈生成相同全局上传支持：`m_t^g = TopK(|g_hat_{t-1}|, K_g)`（第8行；原文写为 `S_top(g_hat_{t-1},K_g)`，S_top按绝对值取top-K）。
3. 该支持强制保留当前本地误差补偿更新中的相应坐标：`g_t,n^g = m_t^g ⊙ g_t,n^ec`（第9行）。
4. 另留本地探索预算：`m_t,n^l = S_top((1-m_t^g)⊙g_t,n^ec,K_l)`（第10行），在全局支持之外选本地重要坐标。
5. server 聚合全局支持分量（over-the-air）及本地探索分量（数字正交传输），结果成为下一轮的广播位移（第14–15行及下一轮第3行）。

**结论**：虽然它没有以“server显式下发mask”作为消息格式，却已经明确实现了“上一轮server聚合反馈 → 当前/下一轮client上传支持”的闭环，并保留 global exploitation + local exploration 两部分预算。因此“反馈帮助安排下一轮上传”这一泛化思想已有先例，不能仅把下行换成mask、或把坐标换成包就宣称首创。该先例不自动否决合理组合，也不验证当前设计是否有效。

精确差异应按反馈语义写清：TCS-H 根据**上一轮聚合位移幅值**统一决定共同支持，目的包括使无线叠加可行；它在上述公式中没有用“哪些包上轮被其他client覆盖/不足覆盖”的计数反馈给不同receiver分配上传优先级，也没有receiver的候选收益检验。若当前设计拟用覆盖/贡献/接收结果作为不同的反馈信号，需要明确其定义、预算与对照；本次检索不足以把这种具体信号判为文献空白。

## 本次返回的其他已知邻居（仅摘要，未展开）

- **FedCSPACK**, arXiv:2601.01840v1：本次再次返回；其mask聚合与client部分覆盖已在主报告核查，不读取repo。
- **SSFL**, arXiv:2405.09037v2，及 **SalientGrads**, arXiv:2304.07488v1：摘要描述聚合本地saliency以初始化共同稀疏支持，不能据摘要将其称为逐轮反馈优先级机制。
- **FedDST**, arXiv:2112.09824v1；**FLASH**, arXiv:2208.13092v3；**SparsyFed**, arXiv:2504.05153v1：摘要涉及动态/共享稀疏mask与适应异构，故属于相关背景；仅凭摘要无法确认是否存在目标的优先级反馈规则。
- **Bandwidth-Aware and Overlap-Weighted Compression**, arXiv:2408.14736v1：摘要描述保留支持的非重叠及参数级聚合权重，未从摘要确认下一轮client上传优先级反馈。

以上记录不把普通local error feedback与server反馈混同，不作novelty或性能主张。

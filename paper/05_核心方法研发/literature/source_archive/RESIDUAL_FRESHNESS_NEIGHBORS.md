# 残差陈旧性与当前本地收益：六篇近邻的有限检索

检索与核对日期：2026-11-03 UTC。本轮使用 `paper-lookup` 的公开 API 检索流程及 `pdf` 技能，外部查询仅含公开一般关键词或已公开论文标识；没有把未发表公式、研究代码或实验结果发到外部。没有启动训练，没有运行下载的作者代码，也没有读取本工程确认实验或 test 输出。

## 核心判断

**“EF 累积残差会陈旧、不能只看残差幅值、应利用新梯度改善其使用”已经有直接先例。** 其中 AdaComp 尤其接近：它早在 2017/AAAI 2018 就用当前梯度改变累积残差的上传选择，并明确讨论 gradient staleness。RegTopK 则在固定 Top-k 选择端利用历史聚合信息抑制可能发生全局抵消的坐标。不能把这两类已有思想重新概括成宽泛原创。

本轮取得的六篇原文支持一个较窄的机制区分：**直接用当前本地目标在指定模型点的梯度，对待上传累积向量逐坐标的一阶收益排序**，不同于 AdaComp 的分箱软阈值、RegTopK 的聚合抵消后验、SAEF/SA-PEF 的梯度计算点预移，以及 FLARE 的本地损失正则。但这只是对这六篇已核对算法的区分，**不是“此前没有相同方法”的检索结论**。公开检索中的 loss-aware / first-order 查询覆盖有限且噪声较大，本轮没有建立足够证据排除这条更窄规则的其他先行工作。

| 论文 | 已核实的主要作用点 | 与当前问题的关系 | 一手证据 |
|---|---|---|---|
| AdaComp（AAAI 2018） | 以最新梯度调整累积残差的分箱上传阈值 | **最直接的当前梯度＋残差选择近邻**；不能只以“新梯度指导旧残差”作差异 | PDF pp.3–4，Algorithm 2；算法页视觉核对 |
| RegTopK（IEEE TSP 2025） | 以历史稀疏聚合推断全局抵消，并重排累积梯度 Top-k | **最直接的固定 K、只改选择、仍传原累积值近邻** | PDF p.14，Algorithm 2；既有作者代码只读快照 |
| SAEF（AAAI 2021） | 训练前预移残差，使梯度评估点接近 EF 的有效更新点；可周期平均误差 | 梯度失配/陈旧动机高度重合，未以当前收益重排支持 | 官方 arXiv HTML，Algorithm 1、Eq.9–11 |
| SA-PEF（2026 预印本） | 部分残差预移＋部分残差 carry，兼容外置压缩器 | 将上述动机明确带入非 IID、多本地步、部分参与 FL | PDF pp.2–5，Algorithm 1；算法页视觉核对 |
| FLARE（IEEE JSTSP 2024） | 用未发送残差定义本地训练的正则目标；上传仍按累积幅值 | 明确覆盖“残差陈旧＋当前本地训练目标”的结合，作用点主要在训练 | PDF pp.4–5，Eq.3–4、Algorithm 1；作者核心代码只读 |
| Fed-EF analysis（2022 arXiv 稿） | 分析非 IID FL 的 EF；解释部分参与的 stale error compensation；讨论按年龄清零 | 提供陈旧残差的理论/机制背景，未提出当前收益选择器 | PDF p.8 Algorithm 2、pp.11–12 §4.3、p.18 Error restarting/Remark 5 |

## 1. AdaComp：当前梯度已经进入残差选择，不能漏比

Chia-Yu Chen, Jungwook Choi, Daniel Brand, Ankur Agrawal, Wei Zhang, Kailash Gopalakrishnan. *AdaComp: Adaptive Residual Gradient Compression for Data-Parallel Distributed Training*. AAAI 2018. DOI: <https://doi.org/10.1609/aaai.v32i1.11728>。作者开放稿：<https://arxiv.org/abs/1712.02679v1>，首次发布 2017-12-07。

原文 pp.2–3 指出：只按残差幅值可能漏掉与高活动输入特征相关的较小但重要的残差；累积残差与 feature activations 相关性弱。p.3 还明确将残差膨胀与 gradient staleness 作为自适应方案的处理对象。

Algorithm 2（PDF p.4）给出：设旧残差为 `r`、最新反传梯度为 `dW`，

\[
G=r+dW,\qquad H=G+dW=r+2dW.
\]

把 `G` 分成固定长度的 bins，每个 bin 的阈值为

\[
M_b=\max_{j\in b}|G_j|.
\]

当 \(|H_j|\ge M_b\) 时，发送 \(Q(G_j)\)，并令残差为 \(G_j-Q(G_j)\)；否则保留 \(G_j\)。它用 `H` 选择，但传的是量化后的 `G`，不是 `H`。正文试过最新梯度的 scale factor 1.5–3，最后取 2；每个 bin 的发送个数会变，因此不是严格全局固定 K。

这与“新梯度指导累积残差选择”有实质重合。进一步从其公开公式可见，\(|G_j+dW_j|^2=G_j^2+2G_jdW_j+dW_j^2\)，已经含新梯度与累积向量的对齐交叉项。不过它不是单独按当前本地损失的一阶变化 \(-\nabla F_i(w)_jG_j\) 排序；其主要规则是分箱、相对幅值阈值和自适应发送数量。不能把二者说成完全相同，也不能把 AdaComp 降成“普通 magnitude Top-k”。本轮未定位到作者 AdaComp 核心实现链接，算法结论以论文为据。

本地一手材料：`residual_freshness/adacomp.pdf`、`adacomp.txt`、`adacomp_page_4.png`、`adacomp_retrieval.json`。

## 2. RegTopK：固定 K 的选择端强近邻

Ali Bereyhi, Ben Liang, Gary Boudreau, Ali Afana. *Regularized Top-k: A Bayesian Framework for Gradient Sparsification*. IEEE Transactions on Signal Processing 73, 4463–4478, 2025. DOI: <https://doi.org/10.1109/TSP.2025.3624791>。作者开放稿：<https://arxiv.org/abs/2501.05633>；本地已有 v2（2026-02-15 更新）及作者代码 <https://github.com/AlBerey/RegTopK> 的只读快照。

Algorithm 2（PDF p.14）先形成 \(a_n^t=\epsilon_n^t+g_n^t\)，对上一轮选中位置构造

\[
\Delta_n^t=s_n^{t-1}\odot
\frac{g^{t-1}-\omega_na_n^{t-1}}{\omega_na_n^t}
+Q(1-s_n^{t-1}),\quad Q\to\infty,
\]

以 \(a_n^t\odot\tanh(|1+\Delta_n^t|/\mu)\) 的幅值选 K 个，**仍传原累积值** \(s_n^t\odot a_n^t\)，未发部分留作 EF。它的当前信息包含当前累积值作为分母，但判断来源是上一轮全局聚合与本客户端贡献所揭示的跨客户端抵消，不是直接估计当前本地损失的梯度收益。

因此，“只改支持、不改所传累积值”“过滤看似大却可能无效的残差”均不足以与 RegTopK 拉开差异。若使用多步模型差分而非原文每步梯度，应明确称 model-delta adaptation。既有源码/论文分母口径差异及 μ=5 证据见 `FOLLOWUP_NEIGHBORS_BASELINES.md` §6；本轮不重做该审计，也不将作者实现的外部性能当成本工程结果。

本地一手材料：`regularized_topk.pdf`、`regularized_topk.txt` p.14，以及 `regtopk_author_code_readonly/`。

## 3. SAEF：残差导致的梯度评估点失配已有明确表述

An Xu, Zhouyuan Huo, Heng Huang. *Step-Ahead Error Feedback for Distributed Training with Compressed Gradient*. AAAI 2021. DOI: <https://doi.org/10.1609/aaai.v35i12.17254>。作者开放稿：<https://arxiv.org/abs/2008.05823v3>，首次发布 2020-08-13；本轮核对官方 HTML：<https://arxiv.org/html/2008.05823v3#alg1>。

原文 Local Error Feedback 一节假定不同 worker 的数据分布相同。Resolving Gradient Mismatch Problem 一节与 Eq.9–11 明确解释：当前模型计算的梯度用于一个受残差位移的辅助迭代点，会产生 gradient mismatch，类似一轮过时梯度。

Algorithm 1 的 SAEF 分支（lines 6–12）在计算梯度前执行

\[
x_{t+1/2}^{(k)}=x_t^{(k)}-e_t^{(k)},\qquad
m_{t+1}^{(k)}=\mu m_t^{(k)}+\nabla f(x_{t+1/2}^{(k)};\xi_t^{(k)}).
\]

之后形成其本地差分、经通用压缩器 \(\mathcal C\) 传输。可每隔 p 轮平均本地误差，作者明确把这项额外通信计入比较。其机制重点是**改变梯度计算点**；Top-k/SignSGD 是所用压缩器，不是以当前目标收益重写支持集打分。

两份 PDF 下载均因限时而不完整，不能把 `saef.pdf` 或 `saef_arxiv.pdf` 当作已完整读取的 PDF；上述公式来自成功取得的官方 HTML，含 LaTeX annotations 与 Algorithm 1 锚点。没有依据本文虚构一个已核实的作者核心代码仓库。

本地一手材料：`residual_freshness/saef_html.html`、`saef_html.txt`；失败/成功记录见 `primary_retrievals.json`、`saef_arxiv_retrieval.json`、`supplemental_primary_log.json`。

## 4. SA-PEF：非 IID FL 的残差失配/衰减动机也已有近邻

Dawit Kiros Redie, Reza Arablouei, Stefan Werner. *SA-PEF: Step-Ahead Partial Error Feedback for Efficient Federated Learning*. arXiv:2601.20738v1，2026-01-28，**预印本，PDF 标注 Under review；未核实正式录用**。<https://arxiv.org/abs/2601.20738v1>。

原文 pp.1–3 明确讨论非 IID、多本地步、压缩和部分参与共同作用下，残差衰减慢、gradient mismatch、早期进展停滞；也指出本地目标漂移。Algorithm 1（PDF p.4）按原文写为：

\[
w_{r+1/2,0}^{(k)}=w_r^{(k)}-\alpha_re_r^{(k)},\qquad
g_r^{(k)}=w_{r+1/2,0}^{(k)}-w_{r+1/2,T}^{(k)},
\]
\[
u_{r+1}^{(k)}=(1-\alpha_r)e_r^{(k)}+g_r^{(k)},\quad
e_{r+1}^{(k)}=u_{r+1}^{(k)}-\mathcal C(u_{r+1}^{(k)}).
\]

它在残差预移和 carry 上加权，压缩器仍为外置 \(\mathcal C\)，未定义逐坐标当前损失收益 score。原文自述 α=0/1 对应 EF/SAEF，但本报告只按其明确算法比较，不把作者的端点等价或全部收敛论证当作本轮独立证明。PDF 未给出可确认为该方法作者实现的核心仓库链接；其中 CIFAR10-fast 链接是既有模型/训练参考，不是该方法实现证据。

本地一手材料：`residual_freshness/sapef.pdf`、`sapef.txt`、`sapef_page_4.png`。

## 5. FLARE：用残差改变当前本地目标已有公开算法与代码

Ran Greidi, Kobi Cohen. *Sparse Training for Federated Learning With Regularized Error Correction*. IEEE Journal of Selected Topics in Signal Processing, 2024. DOI: <https://doi.org/10.1109/JSTSP.2024.3463505>。作者稿：<https://arxiv.org/abs/2312.13795v2>，2024-07-16 更新；正式发表信息由公开 DOI 索引核对，所读作者稿首页仍保留投稿声明。

Eq.3–4（PDF p.4）以客户端累积残差 \(A_k^i\) 为目标偏移，前 p 个本地优化步使用

\[
\widetilde f_k^i(x,w)
=f(x,w)+\tau_k\sum_j
\mathbf 1\{|A_{k,j}^i|>a_0\}
\left|w_j-(w_{k,j}+A_{k,j}^i)\right|.
\]

其余本地步回到原任务损失。Algorithm 1（PDF p.5）再把新模型差分加到 accumulator，仍按其**绝对幅值**选择上传。FLARE 因而不是以新梯度对旧残差支持排序；但“残差包含未使用的优化信息，应让当前本地优化吸收残差、应处理陈旧性”已是其明确目标。

作者仓库由论文 Reference [1] 给出：<https://github.com/RanGreidi/FLARE>。本轮只读抓取 `main` 的 tree SHA `32a3ae69099dddb177e31ad90737876cbc19d1b2`，没有假定其为当前默认分支（公开 repo 元数据的默认分支为 `develop`）。该快照中：

- `src/FLARE_regulerizer.py` lines 15–17 以 accumulator 绝对幅值的百分位阈值做 mask，并计算以 `accumulator+server_weights` 为目标的 masked L1 正则。其阈值实现与论文固定 \(a_0\) 参数表达有细节差异，应以明确版本标注。
- `src/FLARE_utils.py` lines 51–63 的 `sparsify_layer` 用 `tf.nn.top_k(tf.abs(flat_layer), ...)`，lines 108–125 对 `local_difference+accumulator` 调用该函数，再把未发送部分留为残差。因此作者实现证实其上传选择仍是幅值准则。代码有取整/阈值细节，本轮没有执行，也不把它声称为严格全局恰好 K 的参考实现。

本地一手材料：`residual_freshness/flare.pdf`、`flare.txt`、`flare_page_4.png`、`flare_page_5.png`、`flare_repo.json`、`flare_tree.json`、`flare_author_code_readonly/`。下载代码均有 URL、时间、文件 hash；未执行。

## 6. Fed-EF：部分参与的 stale error compensation 及清零权衡

Xiaoyun Li, Ping Li. *Analysis of Error Feedback in Federated Non-Convex Optimization with Biased Compression*. 本轮读取 arXiv:2211.14292v1（2022-11-25），<https://arxiv.org/abs/2211.14292v1>。未在本轮另外核实其正式出版版本，引用这个已读版本即可。

Algorithm 2（PDF p.8）为局部模型差分加误差累积再压缩，使用一般偏置压缩器。§4.3（pp.11–12，Theorem 4 及后续解释）专门指出：部分参与时，某客户端残差来自若干轮以前，因此有 stale error compensation；该年龄与客户端采样有关。这与**每轮全参与但方向已不适合当前目标**并非同一个陈旧性定义，引用时需要区分。

§5 的 Error restarting（PDF p.18）给出示意性策略：若某客户端累积器超过 S 轮未更新，则设为零。Remark 5 同时指出清零会引入偏差，未必改善理论渐近率，展示了陈旧信息与压缩偏差之间的权衡。它没有提出当前局部目标的逐坐标收益排序。

本地一手材料：`residual_freshness/fedef.pdf`、`fedef.txt`。本轮未定位可确认为该版本作者核心实现的链接。

## 一阶收益说法的三条必要边界

这些是根据标准泰勒展开与 EF 假设作出的解释，不是上述论文已证明的新方法保证。

1. **先固定梯度评估点和更新符号。** 若上传向量是加到模型上的位移 s，则小步下本地损失变化近似为 \(\nabla F_i(w)^\top s\)，相应“收益”带负号。若以多步本地模型差分代替负梯度，它只是在额外条件下的方向代理，不能称作在当前模型点计算的一阶损失下降量。
2. **当前本地收益不等于全局收益。** 非 IID 下 \(\nabla F_i(w)\) 与 \(\nabla F(w)=\sum_i p_i\nabla F_i(w)\) 可以方向相反。即便每个客户端按本地有利坐标选，聚合更新对全局损失的影响仍需独立核对。这个差异也是 RegTopK 跨客户端抵消信息值得比较的原因。
3. **仅保留正收益坐标不能自动继承标准 EF 保证。** 若非零累积向量所有坐标被判为非正收益而全部拒发，则 \(\|u-\mathcal C(u)\|^2=\|u\|^2\)，不存在该样本上的正收缩常数。即使补满固定 K，按收益选也未必保留 Top-k 的统一能量比例；引用 contractive-compressor 定理前必须核实它的前提。Fed-EF 对 error restarting 的偏差提醒同样表明，处理陈旧残差与保持 EF 性质需要分开论证。

## 检索范围、保存证据与未解之处

检索仅筛选以下公开查询的前 15 条结果；文献关系由以上一手正文核对，检索排序本身不是完整性证明。

| 来源 | 公开查询 | 索引返回总量 | 本地原始文件 |
|---|---|---:|---|
| OpenAlex | `error feedback residual staleness gradient compression` | 552 | `public_openalex_00.json` |
| OpenAlex | `loss aware gradient sparsification` | 5401 | `public_openalex_01.json` |
| OpenAlex | `federated learning residual correction gradient sparsification` | 365 | `public_openalex_02.json` |
| Crossref | `gradient sparsification first order loss aware` | 2358568，结果噪声高 | `public_crossref_03.json` |
| OpenAlex | `"gradient sparsification" "loss" "first"` | 951 | `loss_gradient.json` |
| OpenAlex | `"gradient compression" "descent" "residual"` | 893 | `fresh_gradient.json` |

另外用 arXiv API 按已经公开的五个论文标识取得第一方书目和摘要：`1712.02679,2008.05823,2601.20738,2312.13795,2211.14292`，原始 XML 为 `metadata.xml`。RegTopK 使用本地本轮任务既有的原文和作者快照，不重复下载。

全部新原始文件位于 `work/literature_followup/residual_freshness/`。`public_search_log.json` 与 `focused_search_log.json` 保存每条 URL、UTC 时间、原始响应路径与结果摘要；`search_log.json` 保留最初普通网络查询的 DNS 失败。随后公开查询在获准网络访问后成功。`primary_retrievals.json`、`adacomp_retrieval.json`、`saef_arxiv_retrieval.json`、`supplemental_primary_log.json` 保存原文请求结果；FLARE 代码的单独 provenance 文件保存快照、URL 和 hash。

SAEF 最初连续 PDF 下载出现长时间缓慢读取，已停止；其 AAAI 与 arXiv 的后续限时 PDF 请求仍为不完整文件，最终以成功的官方 HTML 核对公式。没有把这些失败当作“全文不存在”。本轮没有对付费墙作绕过。

**可支撑的表述**：残差陈旧与梯度失配的动机已有大量相邻工作；当前梯度进入残差选择至少已见 AdaComp；固定 K 的全局感知选择至少已见 RegTopK；当前本地损失的直接逐坐标一阶评分，需要与这些明确机制逐项比较。本轮有限检索留下该更窄规则的先行工作覆盖缺口，不能据此声明“没有前人”或“已确认创新”。

# 两项 primary 核查：Balanced Softmax 训练损失组件

2026-11-04。限两篇指定论文；仅两次不同的精确标题检索，随后沿原文和作者 README 链接读取证据。未运行下载代码，未加载训练/开发/official TEST 数据，未训练。目前没有本项目效果 claim。

## 已确认的公式与来源

| Primary | 直接证据 | 对当前组件的含义 |
|---|---|---|
| Hong-You Chen, Wei-Lun Chao, *On Bridging Generic and Personalized Federated Learning for Image Classification*, ICLR 2022 | [作者 arXiv](https://arxiv.org/abs/2107.00778)，本次读取 v2；PDF p.5 Eq.(6)、p.21 Table 5；本地 `sources/fedrod_arxiv.txt:307–324,1441–1445` | 对客户端 i，以本地训练类别计数 n_ic 构造 BSM：−log[n_iy^γ exp(z_y) / Σ_c n_ic^γ exp(z_c)]。Table 5 的普通 BSM 固定 γ=1，所以等价于 **CE(z+log n_i,y)**。这里是逐类向量相加，随后按标签 y 计算 CE；不是给所有 logit 加同一个 log n_iy。 |
| Jiawei Ren et al., *Balanced Meta-Softmax for Long-Tailed Visual Recognition*, NeurIPS 2020 | [会议正式 PDF](https://proceedings.neurips.cc/paper/2020/file/2ba61cc3a8f44143e1f2f13b2b729ab3-Paper.pdf)，p.3 Eq.(3)–(4)；本地 `sources/balanced_meta_neurips2020.txt:152–172` | 原始 Balanced Softmax 的训练概率是 n_c exp(z_c)/Σ_k n_k exp(z_k)，同样对应 **CE(z+log n,y)**。Balanced Meta-Softmax/BALMS 另含 Meta Sampler；只用该损失不等于复现完整 BALMS。 |

Fed-RoD §4.2（PDF p.6，本地文本390–402行）说明 generic prediction 直接使用 generic head 输出；其完整方法还有个性化残差 head 与第二项 ERM。因此，单一全局网络使用此类训练损失、raw-logit 推理，应称借用 BRM/BSM 组件，不能称官方完整 Fed-RoD 精确复现。原始 Balanced Softmax §3.1 也分别以普通 softmax 定义均衡目标分布、以计数加权 softmax 定义训练分布。

## 零计数与作者代码边界

原文链接的 [BalancedMetaSoftmax 总仓库](https://github.com/jiawei-ren/BalancedMetaSoftmax) README 指向作者分类实现。核查固定提交 `34a61e432881816c2da14d577d6ed63501288f5f` 的 [BalancedSoftmaxLoss.py](https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification/blob/34a61e432881816c2da14d577d6ed63501288f5f/loss/BalancedSoftmaxLoss.py#L26-L50)：

- 26–31行从频数 JSON 读取并保存 `sample_per_class`；47–50行是 `logits + spc.log()`，然后 `F.cross_entropy`。
- **该损失函数没有 +1、epsilon、clamp 或针对0的分支**；本地只读副本为 `sources/balanced_softmax_loss_source.txt`。本次未运行作者实现或核验其实际频数文件，不能把函数外的预处理推断为已核实。
- 直接令某类 n_ic=0 时，log n_ic=−∞；对实际观察到的训练标签 n_iy>0，数学上该未观察类不进入分母，且其 logit 梯度为0。这不等于加一平滑，亦不能误称只要出现0计数整个 CE 必然无定义。

Fed-RoD 论文 p.10 给出的 [作者仓库](https://github.com/hongyouc/Fed-RoD) 在本次固定 HEAD `b5050f43026ac6b203eb615248aeab0a0f43bd58` 的完整 tree 中只有 README；README 也没有损失实现或平滑规则。论文 Eq.(6)/Table 5 未写 +1。因此本次**未确认到 Fed-RoD 作者的官方 zero-count smoothing 实现**；不能据此断言历史版本或其他发布位置绝不存在代码，也不能把拟用 +1 宣称为官方规则。tree、commit、README 均已留档。

## 当前公开固定的变体

仅使用每客户端 **TRAIN** 计数 n_ic，固定所有 C 个类别的计数向量；训练采用

\[
\ell_i(z,y)=\operatorname{CE}\bigl(z+\log(n_i+1),y\bigr)
=-z_y-\log(n_{iy}+1)+\log\sum_{c=1}^{C}(n_{ic}+1)e^{z_c}.
\]

计数不参与求导。令 q_ic=(n_ic+1)e^{z_c}/Σ_k(n_ik+1)e^{z_k}，则 ∂ℓ/∂z_c=q_ic−1[c=y]。这是对全部类计数加一，**不是** `max(n_ic,1)`；零计数类也保留有限 logit 与梯度通路。推理固定使用原始 z/argmax z，不加本地计数项。

将 (n_ic+1)/(N_i+C) 解释为对称 **Dirichlet(1) 先验的后验均值/后验预测概率**，亦即 Laplace 加一平滑；不是 MAP。分母 N_i+C 对所有类相同，在 softmax 中抵消。这个平滑解释是数学说明，**并非两篇原文提出或验证了当前固定规则**。

准确命名建议：**“采用本地训练计数加一平滑的 Balanced Softmax 风格训练损失（公开固定变体），推理使用原始 logits。”** BSM/BRM 本身已有明确先例，+1 规则不包装成新原子创新。已有组件仍可与压缩/量化机制组合研究；是否有组合价值必须由后续固定对照检验，当前不作效果、协同增益或新算法成立的结论。

## 留档

`sources/manifest.json` 保存两次精确检索、primary/作者链接、固定提交、下载失败及替代读取说明，并记录文件 SHA256。PDF、逐页文本、原始 API 响应与只读损失文本均在 `sources/`；所有网络内容只作为核查材料。

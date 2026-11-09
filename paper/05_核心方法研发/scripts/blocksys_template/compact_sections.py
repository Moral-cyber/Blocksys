"""Prose-only proceedings edits; all equations and numeric tables retained."""
from pathlib import Path
import re

def apply(folder, chinese):
    here=Path(__file__).resolve().parent
    for suffix in ('EN','ZH'):
        if chinese != (suffix=='ZH'): continue
        source=here/f'FRONT_{suffix}.tex'
        if source.exists(): (folder/'sections/front.tex').write_text(source.read_text())
    p=folder/'sections/method.tex';t=p.read_text()
    # The six-step restatement duplicates the fully specified procedure.
    marker='\\paragraph{PR-LFL algorithm summary.}' if not chinese else '\\paragraph{PR-LFL 算法摘要。}'
    if marker in t:t=t[:t.index(marker)].rstrip()+'\n'
    elif chinese:
        t=re.sub(r'\\paragraph\{[^}]*算法[^}]*\}.*','',t,flags=re.S)
    if not chinese:
        t=t.replace('maximum-magnitude initialization and four least-squares/reassignment refinements',
            'initialization from the maximum magnitude and four refinements of least squares and code reassignment')
        t=t.replace('For probabilities $p_i(x)$ and an evaluation set $E_i$, client NLL is $-|E_i|^{-1}\\sum_{(x,y)\\in E_i}\\log p_{iy}(x)$; the primary endpoint averages it equally over clients.',
            'For probabilities $p_i(x)$ on evaluation set $E_i$, define\n\\begin{equation}\n\\operatorname{NLL}_i=-|E_i|^{-1}\\sum_{(x,y)\\in E_i}\\log p_{iy}(x).\n\\end{equation}\nThe primary endpoint averages this equally over clients.')
    p.write_text(t)
    p=folder/'sections/discussion.tex'
    p.write_text(r'''\section{讨论}
先验锚点在保持仿射维度不变的对照中改善了客户端概率预测。同先验 BCTS 的结果接近且宏平均 ECE 更低，因此收益主要支持本地先验正则化，尚不足以说明类别独立尺度必不可少。NLL、准确率和 ECE 分别刻画概率、分类决策和分箱校准，不应相互替代。

轻量化收益来自实际上传字节和较小的本地预测状态。完整部署仍包含共享模型、误差反馈残差、TRAIN 前向、缓存与求解，不能据此推断能耗或端到端速度。持久上传协议解决回执前后残差的一致性；其证据局限于独立的本地恢复仿真。共享密钥持有者仍可认证恶意更新，服务器和正确解码仍受信任。两个任务、固定划分和三个训练种子限定了经验范围；真实链与经认证的端到端训练仍需单独验证。
''' if chinese else r'''\section{Discussion}
Prior anchoring improves local probability prediction at fixed affine dimensionality. The competitive same-prior BCTS results and its lower macro ECE place the explanatory weight on local prior regularization; they do not establish that independent class scales are essential. NLL, accuracy, and ECE describe different predictive properties and should be interpreted separately.

The measured lightweight benefit concerns uploaded bytes and local prediction state. Complete deployment also retains the shared model, error-feedback residuals, TRAIN forward passes, caches, and fitting costs, so energy and end-to-end speed remain open. Durable upload handling addresses residual consistency after incorporation; its evidence is a separate local recovery simulation. The key-holding server and correct decoder remain trusted, and a key holder can authenticate a poisoned update. Two tasks, fixed partitions, and three training seeds define the empirical scope. Real-ledger execution and authenticated end-to-end training require separate validation.
''')
    p=folder/'sections/appendix.tex';t=p.read_text()
    marker='\\paragraph{软件' if chinese else '\\paragraph{Software'
    if marker in t:
        t=t[:t.index(marker)]+(r'''\paragraph{软件指导。}
图形制作参考 Scientific Agent Skills 的 scientific-visualization 技能：Timothy Kassis、Vinayak Agarwal、Yuhuan He、Darshil Patel 与 Aubrey M. Brueckner，\emph{Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents}（2026），arXiv:2609.00065，\url{https://doi.org/10.48550/arXiv.2609.00065}（v2，2026年11月4日核查）。此项致谢不构成实验证据。
''' if chinese else r'''\paragraph{Software guidance.}
Figure preparation used scientific-visualization from Scientific Agent Skills: Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner, \emph{Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents} (2026), arXiv:2609.00065, \url{https://doi.org/10.48550/arXiv.2609.00065} (v2, checked 4 November 2026). This acknowledgement supplies no experimental evidence.
''')
    p.write_text(t)
    # Avoid unsplittable compound names at the narrow class text width.
    for p in (folder/'sections').glob('*.tex'):
        t=p.read_text().replace('persistent-personalization references','references for persistent personalization')
        if chinese:
            t=t.replace('固定三个种子的 TopK+EF 与持久 FedRoD-linear 实现','TopK+EF 与持久 FedRoD-linear 的固定三种子实现')
        t=t.replace('class-frequency-aware constructions','constructions informed by class frequency').replace('count-dependent local logit margins','local logit margins based on counts')
        t=t.replace('reaches CIFAR-10 accuracy','reaches an accuracy on CIFAR-10 of').replace('three predefined training seeds: CIFAR-10 partition','three predefined training seeds. The CIFAR-10 partition is')
        p.write_text(t)

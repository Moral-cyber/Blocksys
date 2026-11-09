"""Adapt frozen manuscripts to the user-supplied svproc kit; no model fitting."""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import re
import shutil

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
ORIGINAL=Path('/Users/wind/Desktop/DeepSeek Harness/BlockSys2027-template')
SNAPSHOT=OUT/'previous_revision/pre_blocksys_20261104'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--from-current',action='store_true');args=parser.parse_args()
    kit=HERE/'template';kit.mkdir(exist_ok=True)
    for name in ('svproc.cls','spmpsci_unsrt.bst'):
        if ORIGINAL.joinpath(name).exists():
            shutil.copy2(ORIGINAL/name,kit/name)
        assert (kit/name).is_file()
    manifest={name:sha(kit/name) for name in ('svproc.cls','spmpsci_unsrt.bst')}
    (HERE/'TEMPLATE_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if not SNAPSHOT.exists():
        for lang in ('english','chinese'):
            shutil.copytree(OUT/f'{lang}_paper',SNAPSHOT/f'{lang}_paper')
    for lang in ('english','chinese'):
        folder=OUT/f'{lang}_paper'
        if not args.from_current:
            # Reproduce the adaptation from the preserved generic sources.
            for name in ('sections','tables'):
                shutil.copytree(SNAPSHOT/f'{lang}_paper'/name,folder/name,dirs_exist_ok=True)
        for name in manifest: shutil.copy2(kit/name,folder/name)
        (folder/'checks/TEMPLATE_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
        chinese=lang=='chinese'
        title=('先验正则化轻量化联邦预测与上下文绑定的上传认证' if chinese else
               'Prior-Regularized Lightweight Federated Prediction with Context-Bound Upload Authentication')
        running='先验正则化轻量化联邦预测' if chinese else 'Prior-Regularized Lightweight Federated Prediction'
        preamble=r'''\documentclass{svproc}
'''+(r'''\usepackage{xeCJK}
\setmainfont{lmroman10-regular.otf}[BoldFont=lmroman10-bold.otf,ItalicFont=lmroman10-italic.otf]
\setCJKmainfont{FandolSong-Regular.otf}[BoldFont=FandolSong-Bold.otf,ItalicFont=FandolKai-Regular.otf]
\setCJKsansfont{FandolHei-Regular.otf}[BoldFont=FandolHei-Bold.otf]'''+ '\n' if chinese else '')+r'''\usepackage{url}
\usepackage{amsmath,amssymb,booktabs,array,graphicx}
\def\UrlFont{\rmfamily}
% Permit long source URLs to wrap without changing the class or bibliography.
\makeatletter
\g@addto@macro\UrlBreaks{\do\a\do\b\do\c\do\d\do\e\do\f\do\g\do\h\do\i\do\j\do\k\do\l\do\m\do\n\do\o\do\p\do\q\do\r\do\s\do\t\do\u\do\v\do\w\do\x\do\y\do\z\do\0\do\1\do\2\do\3\do\4\do\5\do\6\do\7\do\8\do\9}
\makeatother
\begin{document}
\mainmatter
\title{'''+title+r'''}
\titlerunning{'''+running+r'''}
% Author and affiliation are intentionally empty until supplied by the author.
% This does not assert a conference anonymity policy.
\author{}
\authorrunning{}
\tocauthor{}
\institute{}
\maketitle
'''
        main=preamble+r'''\input{sections/abstract}
\input{sections/front}
\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{figures/workflow.pdf}
\caption{'''+('学习、本地适配及独立测试的持久上传路径。' if chinese else
            'Learning, local adaptation, and the separately tested durable upload route.')+r'''}\label{fig:workflow}
\end{figure}
\input{sections/method}
\input{sections/evaluation}
\input{sections/results}
\input{sections/discussion}
\input{sections/conclusion}
\input{sections/appendix}
\bibliographystyle{spmpsci_unsrt}
\bibliography{references}
\end{document}
'''
        (folder/'main.tex').write_text(main)
        abstract=(r'''\begin{abstract}
标签偏斜使通信受限的联邦客户端需要不同的概率预测器。PR-LFL 将三比特分块量化与误差反馈结合，并在冻结共享网络后，仅用本地 TRAIN 数据拟合以平滑标签先验为偏置锚点的正则仿射预测器。十分类客户端保存二十个 FP32 适配值。在固定的标签偏斜官方 TEST 划分上，每任务三个训练种子的 CIFAR-10 与 FashionMNIST 宏平均 NLL 分别为 0.6446 和 0.1880，相对同维零锚点对照降低 0.0358 和 0.0182；准确率分别为 75.57\% 和 92.97\%。实际上传载荷为 54,067 B。同先验 BCTS 解释了大部分收益。独立的上下文绑定 HMAC 协议在本地协议及全载荷测试中分别通过 62/62 和 186/186 项检查；真实账本和经认证的端到端学习尚待验证。
\keywords{轻量化联邦学习；标签偏斜；个性化预测；先验正则化；上传认证}
\end{abstract}
''' if chinese else r'''\begin{abstract}
Label-skewed clients require different probability predictors under communication constraints. We combine three-bit quantization and error feedback with a TRAIN-only affine predictor regularized toward a smoothed local label prior. Ten-class clients store twenty FP32 adaptation values. On fixed label-skew assignments of official TEST data, three training seeds per task yield client-macro NLLs of 0.6446 on CIFAR-10 and 0.1880 on FashionMNIST, reductions of 0.0358 and 0.0182 against same-dimensional zero-anchor adaptation. Accuracies are 75.57\% and 92.97\%. Actual uploads occupy 54,067 B. A same-prior BCTS control explains much of the benefit. A separate context-bound HMAC protocol couples residual commitment to durable incorporation, passing 62/62 local protocol and 186/186 full-payload checks. Real-ledger execution and authenticated end-to-end learning remain untested.
\keywords{Federated learning; label skew; prior regularization; upload authentication}
\end{abstract}
''')
        (folder/'sections/abstract.tex').write_text(abstract)
        # Preserve the development history outside the manuscript, then remove
        # process-only gates from the proceedings appendix.
        original=(folder/'sections/appendix.tex').read_text()
        (folder/'evidence/DEVELOPMENT_RULES_AND_PRIOR_APPENDIX.tex.txt').write_text(original)
        if chinese:
            software=original[original.index('\\paragraph{软件'):] if '\\paragraph{软件' in original else ''
            appendix=r'''\appendix
\section{逐种子结果与复现信息}
\input{tables/seeds}
每任务使用一个固定划分和三个预先固定的训练种子：CIFAR-10 划分为 20261217，种子为 20261231--20261233；FashionMNIST 划分为 20261219，种子为 20261241--20261243。模型、适配规则、惩罚系数和最终检查点均在官方 TEST 推断前冻结。客户端和轮次不作为独立训练重复。

复现包记录数据文件身份、TRAIN/留出/TEST 成员、三个种子的检查点、FP32 适配状态、逐样本分数导出及编解码器和求解器摘要；本地协议密钥不包含在内。开发阶段的完整判据保存在随附证据材料中。本地独立数值核验与不可用的外部跨模型审查分别记录。

'''+software
        else:
            software=original[original.index('\\paragraph{Software'):]
            appendix=r'''\appendix
\section{Complete Seed Results and Reproducibility}
\input{tables/seeds}
Each task uses one fixed partition and three predefined training seeds: CIFAR-10 partition 20261217 with seeds 20261231--20261233, and FashionMNIST partition 20261219 with seeds 20261241--20261243. Models, adaptation rules, penalties, and final checkpoints were frozen before official TEST inference. Clients and rounds are not independent training replicates.

The bundle records file identities, TRAIN/holdout/TEST membership, all seed checkpoints, FP32 adaptation states, score exports, and codec/solver digests; local protocol keys are excluded. Full development criteria remain in the accompanying evidence materials. Local independent numerical checks and unavailable external cross-model review are recorded separately.

'''+software
        appendix=appendix.replace(r'\texttt{10.48550/arXiv.2609.00065}',r'\url{https://doi.org/10.48550/arXiv.2609.00065}')
        (folder/'sections/appendix.tex').write_text(appendix)
        method=(folder/'sections/method.tex').read_text()
        if not chinese:
            method=method.replace('The primary objective is lower client-macro holdout negative log likelihood (NLL), subject to predefined accuracy-preservation constraints.',
                'Client-macro negative log likelihood (NLL) is the primary endpoint: development uses holdouts, and the fixed final assessment uses official TEST data.')
            method=method.replace("For probabilities $p_i(x)$, client NLL is $-|H_i|^{-1}\\sum_{(x,y)\\in H_i}\\log p_{iy}(x)$; the primary endpoint averages it equally over clients.",
                "For probabilities $p_i(x)$ and an evaluation set $E_i$, client NLL is $-|E_i|^{-1}\\sum_{(x,y)\\in E_i}\\log p_{iy}(x)$; the primary endpoint averages it equally over clients. Development uses $E_i=H_i$; final results use the assigned official TEST examples.")
            method=method.replace(r'$|H_i|^{-1}\sum_{(x,y)\in H_i}\sum_c',r'$|E_i|^{-1}\sum_{(x,y)\in E_i}\sum_c')
        else:
            method=method.replace('主要目标是在满足预设准确率保持约束的同时，降低客户端宏平均留出负对数似然（NLL）。',
                '主要终点为客户端宏平均负对数似然（NLL）：开发使用留出集，固定的最终评估使用官方 TEST 数据。')
            method=method.replace('给定概率 $p_i(x)$，客户端 NLL 为 $-|H_i|^{-1}\\sum_{(x,y)\\in H_i}\\log p_{iy}(x)$；主要终点对客户端等权平均。',
                '给定概率 $p_i(x)$ 和评估集 $E_i$，客户端 NLL 为 $-|E_i|^{-1}\\sum_{(x,y)\\in E_i}\\log p_{iy}(x)$；主要终点对客户端等权平均。开发时 $E_i=H_i$，最终结果使用分配给客户端的官方 TEST 样本。')
            method=method.replace(r'$|H_i|^{-1}\sum_{(x,y)\in H_i}\sum_c',r'$|E_i|^{-1}\sum_{(x,y)\in E_i}\sum_c')
        (folder/'sections/method.tex').write_text(method)
        for name in ('main','training'):
            p=folder/f'tables/{name}.tex';t=p.read_text()
            t=t.replace(r'\begin{tabular}{lrrrr}',r'\begin{tabular}{@{}lrrrr@{}}')
            first='Predictor' if name=='main' else 'Training'
            t=t.replace(first+r' & CIFAR Acc. & CIFAR NLL & Fashion Acc. & Fashion NLL\\',
                r' & \multicolumn{2}{c}{CIFAR-10} & \multicolumn{2}{c}{FashionMNIST}\\'+'\n'+
                r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}'+'\n'+first+r' & Acc. (\%) & NLL & Acc. (\%) & NLL\\')
            t=t.replace('Shared + TRAIN prior','Shared + prior')
            if name=='training':
                t=t.replace('FedLC-inspired + uniform3',r'\shortstack[l]{FedLC-inspired\\+ uniform3}')
                t=t.replace('FedLC-inspired + dense',r'\shortstack[l]{FedLC-inspired\\+ dense}')
            p.write_text(t)
        p=folder/'tables/quality.tex';t=p.read_text().replace(r'\begin{tabular}{llrrr}',r'\begin{tabular}{@{}llrrr@{}}')
        t=t.replace('Macro ECE & Pooled ECE','ECE$_{\\rm macro}$ & ECE$_{\\rm pooled}$')
        t=t.replace('FashionMNIST &',r'Fashion &')
        p.write_text(t)
        p=folder/'tables/seeds.tex';t=p.read_text().replace(r'\begin{tabular}{llrrrr}',r'\begin{tabular}{@{}llrrrr@{}}')
        t=t.replace('FashionMNIST &','Fashion &').replace('NLL reduction',r'$-\Delta$NLL').replace(r'$\Delta$Acc. (pp)',r'$\Delta$Acc.')
        p.write_text(t)
        from compact_sections import apply
        apply(folder,chinese)
        from tables import apply as reflow_tables
        reflow_tables(folder)
        for path in [folder/'main.tex',*list((folder/'sections').glob('*.tex')),*list((folder/'tables').glob('*.tex'))]:
            path.write_text(path.read_text().replace('[htbp]','[!ht]'))
        # New artwork is generated at the actual 122 mm destination width.
        from figures import generate
        generate(folder/'figures')
        (folder/'README.md').write_text('''# BlockSys 模板适配稿

main.pdf 为阅读入口；main.tex、sections/、tables/、figures/、references.bib、svproc.cls 和 spmpsci_unsrt.bst 构成完整 LaTeX 来源。英文为投稿工作稿；中文为对应阅读稿。

版式以用户指定的 BlockSys2027-template 文件夹为准：原版 svproc 类和 spmpsci_unsrt 参考样式原字节保留，无 geometry、字号或版心覆盖。参考文献按首次引用编号。英文用 pdfLaTeX + BibTeX；中文用 XeLaTeX + BibTeX（Fandol 中文字体）。作者、单位和邮箱尚待作者填写；当前空值不代表已确认匿名要求。

按用户明确要求，2027 年投稿沿用这套 2026 年模板，模板年份无需另行确认。当前按含参考文献及附录的整份 PDF 不超过14页准备，实际页数见 CHECK_REPORT.md。作者单位与投稿系统信息需作者最终填写。

所有方法定义和冻结实验数值保留。数字来源为 evidence/PAPER_NUMBERS.json；尚未执行的真实链和经认证端到端系统实验见 EXPERIMENT_CHECKLIST.md。旧通用版已保存到 outputs/previous_revision/pre_blocksys_20261104，并保留桌面历史版本。

本轮应用 venue-templates、paper-writer、scientific-visualization 技能。模板来源与核验范围见 checks/TEMPLATE_COMPLIANCE.md，模板文件摘要见 checks/TEMPLATE_SHA256.json。
''')
        shutil.copy2(HERE/'COMPLIANCE_NOTE.md',folder/'checks/TEMPLATE_COMPLIANCE.md')
        current=json.loads((folder/'CURRENT_REVISION.json').read_text())
        current.update(status='TEMPLATE_ADAPTED_COMPILE_PENDING',format='user-supplied BlockSys svproc',template_sha256=manifest,
            authors='not supplied; empty fields; anonymity policy unverified')
        (folder/'CURRENT_REVISION.json').write_text(json.dumps(current,indent=2)+'\n')
    print(json.dumps({'status':'ADAPTED_TO_USER_TEMPLATE','snapshot':str(SNAPSHOT),'template_sha256':manifest}))

if __name__=='__main__':main()

"""Build bilingual evidence-bound manuscripts from completed frozen results."""
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil

import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'plot_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FINAL=ROOT/'work/predictive_final_test'
OUTPUT=ROOT/'outputs'
TASKS={'cifar':(20261217,(20261231,20261232,20261233),'CIFAR-10'),
       'fashion':(20261219,(20261241,20261242,20261243),'FashionMNIST')}
MAIN='prioranchored_vector'
LABELS={'raw':'Shared model','prior_only':'TRAIN prior only','analytic_prior':'Shared + TRAIN prior',
    'prioranchor_bias':'Prior bias (10d)','zero_anchor_bias':'Zero bias (10d)',
    'temperature_scaling':'Temperature (1d)','bcts':'Zero BCTS (11d)',
    'zero_anchor_vector':'Zero vector (20d)','prioranchored_vector':'Prior vector (20d)',
    'learned_head':'Local Linear (650d)','learned_head_plus_prior':'Linear + prior',
    'prioranchored_bcts':'Prior BCTS (11d)'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
SOURCES={}
def load(path):
    path=Path(path)
    value=json.loads(path.read_text())
    SOURCES[str(path.resolve())]=sha(path)
    if value.get('status')!='DONE':raise ValueError('Completed evidence required: '+str(path))
    return value

def stat(values):
    a=np.asarray(values,dtype=float)
    if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError('Exactly three finite seed metrics required')
    return float(a.mean()),float(a.std(ddof=1))

def fmt(values,scale=1,digits=3):
    mean,sd=stat(np.asarray(values)*scale)
    return f'${mean:.{digits}f}\\pm{sd:.{digits}f}$'

def records(task,method='fedlc_uniform3'):
    return [load(FINAL/f'official_{task}/{method}_s{seed}/result.json') for seed in TASKS[task][1]]

def values(rows,variant,metric):return [row['summaries'][variant][metric] for row in rows]

def table(caption,label,header,rows,columns):
    return ('\\begin{table}[htbp]\n\\centering\\small\n\\caption{'+caption+'}\n\\label{'+label+'}\n'
       '\\begin{tabular}{'+columns+'}\n\\toprule\n'+header+'\\\\\n\\midrule\n'
       +'\n'.join(' & '.join(row)+'\\\\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

def scientific_assets(rows,out):
    out.mkdir(parents=True,exist_ok=True)
    colors=['#0072B2','#D55E00','#007A5E']
    variants=('zero_anchor_vector','prioranchored_bcts',MAIN)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,2,figsize=(7.1,2.8),layout='constrained')
        for ax,(task,items) in zip(axes,TASKS.items()):
            for i,(variant,color) in enumerate(zip(variants,colors)):
                acc=np.asarray(values(rows[task],variant,'macro_accuracy'))*100
                nll=np.asarray(values(rows[task],variant,'macro_nll'))
                ax.scatter(acc,nll,color=color,marker=('s','^','o')[i],s=26,label=LABELS[variant])
                ax.errorbar(acc.mean(),nll.mean(),xerr=acc.std(ddof=1),yerr=nll.std(ddof=1),
                    color=color,fmt=('s','^','o')[i],markersize=7,capsize=3,alpha=.8)
            ax.set(title=items[2],xlabel='Client-macro accuracy (%)',ylabel='Client-macro NLL')
            ax.grid(alpha=.2)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=7)
        for extension in ('pdf','svg','png'):fig.savefig(out/f'prediction.{extension}',dpi=300,facecolor='white')
        plt.close(fig)
    for ext in ('pdf','svg','png'):
        source=HERE/f'figures/workflow.{ext}'
        if source.exists():shutil.copy2(source,out/source.name)
    (out/'FIGURE_PROVENANCE.json').write_text(json.dumps({
        'prediction':{'source_results':SOURCES,'unit':'independent training seed, exactly3/task',
            'encoding':'small points are individual seeds; large points mean; horizontal/vertical bars sample SD (ddof1)',
            'transformations':'accuracy times100; no smoothing or rescaling of NLL',
            'alt':'Accuracy and NLL on each task for zero vector, prior BCTS, and prior vector; three seed observations and mean with sample SD.',
            'scope':'one fixed new TEST client assignment/task, 20rounds; provisional generic conference figure format'},
        'workflow':'separately evaluated learning and durable-protocol routes; see source alt/provenance'},indent=2)+'\n')

def main():
    final={task:load(FINAL/f'official_{task}/result.json') for task in TASKS}
    refs=load(FINAL/'reference_preflight/result.json')
    rows={task:records(task) for task in TASKS}
    controls=('raw','prior_only','analytic_prior','prioranchor_bias','temperature_scaling','bcts',
              'zero_anchor_vector','prioranchored_bcts',MAIN,'learned_head')
    summary={task:{} for task in TASKS}
    per_seed=[]
    for task in TASKS:
        for row in rows[task]:
            for variant,metrics in row['summaries'].items():
                per_seed.append({'task':task,'seed':row['seed'],'method':'fedlc_uniform3','variant':variant,**metrics})
        for variant in rows[task][0]['summaries']:
            summary[task][variant]={metric:dict(zip(('mean','sample_sd'),stat(values(rows[task],variant,metric))))
                for metric in ('macro_accuracy','macro_nll','macro_brier','macro_ece15','weighted_accuracy','weighted_nll','client_accuracy_q10','pooled_ece15')}
    differences={task:{variant:{metric:dict(zip(('mean','sample_sd'),stat(
        np.asarray(values(rows[task],MAIN,metric))-np.asarray(values(rows[task],variant,metric)))))
        for metric in ('macro_accuracy','macro_nll','macro_brier','macro_ece15')}
        for variant in controls if variant!=MAIN} for task in TASKS}
    # Reference script's report remains a separately labeled artifact. Its
    # complete fixed-method data are rendered by the small adapter below.
    reference_rows=reference_table_rows(refs)
    for lang,folder in (('en','english_paper'),('zh','chinese_paper')):
        out=OUTPUT/folder
        if out.exists() and not (out/'CURRENT_REVISION.json').exists():
            previous=OUTPUT/'previous_revision'/folder
            previous.parent.mkdir(exist_ok=True)
            if previous.exists():raise ValueError('Previous revision already exists, inspect before moving')
            out.rename(previous)
        out.mkdir(parents=True,exist_ok=True)
        for name in ('sections','tables','figures','evidence','checks'): (out/name).mkdir(exist_ok=True)
        for src,dst in ((f'FRONT_{lang.upper()}.tex','front.tex'),(f'DISCUSSION_{lang.upper()}.tex','discussion.tex')):
            shutil.copy2(HERE/src,out/'sections'/dst)
        front=(out/'sections/front.tex').read_text()
        front=front.replace('The resulting procedure combines','We propose prior-regularized lightweight federated learning (PR-LFL), which combines')
        front=front.replace('据此形成的流程结合了','本文提出先验正则化轻量化联邦学习（PR-LFL），结合')
        (out/'sections/front.tex').write_text(front)
        method=(HERE/f'METHOD_{lang.upper()}.tex').read_text()
        # Keep the fitting and evaluation definitions in the main Methods;
        # move development continuation criteria to the appendix.
        begin=method.index('The fixed development rule' if lang=='en' else '固定的开发规则') if lang=='en' else -1
        if lang=='en':
            end=method.index('\\subsection{Controls and attribution}')
            gate_text=method[begin:end]
            method=method[:begin]+method[end:]
        else:
            # Chinese subsection has exactly the same block ordering; split
            # the last two paragraphs of its endpoints subsection.
            sections=method.split('\\subsection{')
            endpoint=sections[5]
            blocks=endpoint.split('\n\n')
            gate_text='\n\n'.join(blocks[-3:])
            sections[5]='\n\n'.join(blocks[:-3])+'\n\n'
            method='\\subsection{'.join(sections)
        (out/'sections/method.tex').write_text(method)
        (out/'sections/method.tex').write_text(method+algorithm_text(lang))
        shutil.copy2(HERE/'verified_references.bib',out/'references.bib')
        mainrows=[]
        for variant in controls:
            mainrows.append([LABELS[variant],*[fmt(values(rows[task],variant,metric),100 if metric=='macro_accuracy' else 1,2 if metric=='macro_accuracy' else 3)
                for task in TASKS for metric in ('macro_accuracy','macro_nll')]])
        caption=('Official TEST results under the fixed TRAIN-count-defined client assignment. Mean $\\pm$ sample SD over three training seeds. Accuracy is client-macro percentage; NLL is client-macro. All rows use the same FedLC-inspired uniform3 backbone.'
            if lang=='en' else '固定的 TRAIN 计数定义客户端划分上的官方 TEST 结果。三训练种子的均值与样本标准差。准确率为客户端宏平均百分数，NLL 为客户端宏平均。所有行使用相同的 FedLC 启发式 uniform3 共享模型。')
        (out/'tables/main.tex').write_text(table(caption,'tab:main','Predictor & CIFAR Acc. & CIFAR NLL & Fashion Acc. & Fashion NLL',mainrows,'lrrrr'))
        training=[]
        for method,label in (('ce_uniform3','CE + uniform3'),('fedlc_uniform3','FedLC-inspired + uniform3'),('fedlc_dense','FedLC-inspired + dense')):
            training.append([label,*[fmt(values(records(task,method),MAIN,metric),100 if metric=='macro_accuracy' else 1,2 if metric=='macro_accuracy' else 3)
                for task in TASKS for metric in ('macro_accuracy','macro_nll')]])
        (out/'tables/training.tex').write_text(table(
            'Training-loss and compression controls; every row uses the same prior-vector adaptation.' if lang=='en' else '训练损失与压缩对照，所有行使用相同的先验向量适配。',
            'tab:training','Training & CIFAR Acc. & CIFAR NLL & Fashion Acc. & Fashion NLL',training,'lrrrr'))
        (out/'tables/reference.tex').write_text(table(
            'CIFAR-10 official TEST references, three paired seeds. These are disclosed implementation variants; the TopK budget cap is shared but actual bytes differ.' if lang=='en' else 'CIFAR-10 官方 TEST 强参考，三个配对种子。均为公开说明的实现变体；TopK 与 uniform3 共享预算上限，但实际字节数不同。',
            'tab:reference','Method & Acc. (\\%) & NLL & Upload (B)',reference_rows,'lrrr'))
        quality=[]
        for task in TASKS:
            for variant in ('zero_anchor_vector','prioranchored_bcts',MAIN):
                quality.append([TASKS[task][2],LABELS[variant],fmt(values(rows[task],variant,'macro_brier')),
                    fmt(values(rows[task],variant,'macro_ece15'),100,2),fmt(values(rows[task],variant,'pooled_ece15'),100,2)])
        (out/'tables/quality.tex').write_text(table('Probability-quality endpoints; lower is better. ECE values are percentage points.' if lang=='en' else '概率质量指标，越低越好。ECE 数值单位为百分点。',
            'tab:quality','Task & Predictor & Brier & Macro ECE & Pooled ECE',quality,'llrrr'))
        allrows=[]
        for task in TASKS:
            for row in rows[task]:
                z=row['summaries']['zero_anchor_vector'];p=row['summaries'][MAIN]
                allrows.append([TASKS[task][2],str(row['seed']),f"{100*p['macro_accuracy']:.4f}",f"{p['macro_nll']:.6f}",
                    f"{100*(p['macro_accuracy']-z['macro_accuracy']):+.4f}",f"{z['macro_nll']-p['macro_nll']:.6f}"])
        (out/'tables/seeds.tex').write_text(table('Every fixed training seed; differences compare prior vector with same-dimensional zero vector.' if lang=='en' else '全部固定种子；差值比较先验向量与同维零锚点向量。',
            'tab:seeds','Task & Seed & Acc. (\\%) & NLL & $\\Delta$Acc. (pp) & NLL reduction',allrows,'llrrrr'))
        scientific_assets(rows,out/'figures')
        (out/'sections/evaluation.tex').write_text(evaluation_text(lang))
        (out/'sections/results.tex').write_text(results_text(lang,summary,differences,rows))
        appendix=appendix_text(lang,gate_text)
        software=HERE/f'SOFTWARE_NOTE_{lang.upper()}.tex'
        if software.exists():appendix+='\n'+software.read_text()
        (out/'sections/appendix.tex').write_text(appendix)
        (out/'sections/abstract.tex').write_text(abstract_text(lang,summary,differences))
        (out/'main.tex').write_text(preamble(lang)+r'''
\begin{document}
\maketitle
\input{sections/abstract}
\input{sections/front}
\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{figures/workflow.pdf}
\caption{'''+('Learning, local adaptation, and the separately tested durable upload route.' if lang=='en' else '学习、本地适配及独立测试的持久上传路径。')+r'''}\label{fig:workflow}
\end{figure}
\input{sections/method}
\input{sections/evaluation}
\input{sections/results}
\input{sections/discussion}
\input{sections/conclusion}
\input{sections/appendix}
\bibliographystyle{unsrt}
\bibliography{references}
\end{document}
''')
        (out/'sections/conclusion.tex').write_text(conclusion_text(lang))
        evidence={'status':'DONE','scope':'fixed official TEST evaluations; no TEST-driven retuning',
            'primary_summaries':summary,'paired_main_minus_control':differences,'source_sha256':SOURCES,
            'references':refs,'seed_unit':'three independent training seeds/task, one fixed partition/task'}
        (out/'evidence/PAPER_NUMBERS.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
        with (out/'evidence/individual_seed_metrics.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(per_seed[0]));writer.writeheader();writer.writerows(per_seed)
        (out/'EXPERIMENT_CHECKLIST.md').write_text(checklist_text(lang))
        (out/'README.md').write_text(readme_text(lang))
        shutil.copy2(HERE/'REFERENCE_AUDIT.md',out/'checks/REFERENCE_AUDIT.md')
        (out/'CURRENT_REVISION.json').write_text(json.dumps({'status':'GENERATED_COMPILE_PENDING','method':'prior-regularized vector + uniform3/EF',
            'language':lang,'previous_revision':'../previous_revision/','source_sha256':SOURCES},indent=2)+'\n')
    # Keep subsequent rebuilds in the user's selected template. This is a
    # formatting pass only; all numeric tables above still come from the
    # completed frozen outputs.
    import subprocess
    import sys
    subprocess.run([sys.executable,str(ROOT/'work/blocksys_template/adapt.py'),'--from-current'],check=True)
    print(json.dumps({'status':'BILINGUAL_SOURCES_WRITTEN','source_results':len(SOURCES),'format':'user-supplied svproc'}))

def reference_table_rows(refs):
    # Written against the reference exporter contract, verified by the root
    # before this builder is executed. Exact field adapter is below.
    rows=[]
    for method,label,upload in (
        ('fedlc_uniform3','Prior vector + uniform3',54067),
        ('fedlc_topk','Prior vector + TopK',54808),
        ('fedrod','FedRoD-linear + uniform3',54067)):
        if method=='fedlc_uniform3':
            report=records('cifar');acc=values(report,MAIN,'macro_accuracy');nll=values(report,MAIN,'macro_nll')
        else:
            method_key=method if method=='fedlc_topk' else 'fedrod_linear_uniform3'
            key=MAIN if method=='fedlc_topk' else 'generic_plus_local_residual'
            items=sorted((row for row in refs['reports'] if row['method']==method_key),key=lambda row:row['seed'])
            acc=[row['summaries'][key]['macro_accuracy'] for row in items]
            nll=[row['summaries'][key]['macro_nll'] for row in items]
        rows.append([label,fmt(acc,100,2),fmt(nll),f'{upload:,}'])
    return rows

def preamble(lang):
    prefix=(r'\documentclass[10pt,a4paper]{article}\usepackage[T1]{fontenc}\usepackage{lmodern}' if lang=='en'
        else r'\documentclass[UTF8,fontset=fandol,10pt,a4paper]{ctexart}')
    title=('Prior-Regularized Lightweight Federated Prediction\\\\with Context-Bound Upload Authentication' if lang=='en'
        else '先验正则化轻量化联邦预测\\\\与上下文绑定的上传认证')
    return prefix+r'''
\usepackage[margin=22mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,array,graphicx,longtable}
\usepackage[expansion=false]{microtype}
\usepackage[hidelinks]{hyperref}
\setlength{\emergencystretch}{3em}
\title{'''+title+r'''}
\author{}\date{}
'''

def evaluation_text(lang):
    if lang=='en':return r'''
\section{Evaluation Protocol}
Development data comprise 37,500 TRAIN and 12,500 client-holdout examples on CIFAR-10, and 45,001 TRAIN and 14,999 client-holdout examples on FashionMNIST. All models, score families, penalties, final checkpoints, and three training seeds were fixed before official TEST inference. Each task uses its 10,000 official TEST examples, exactly once across twenty clients. Because these datasets do not provide the client assignment used here, we define and report a separate label-skew TEST protocol.

For each class $c$, define $P(i\mid c)=(n_{ic}+1)/\sum_j(n_{jc}+1)$ from local TRAIN counts. Allocate the 1,000 official TEST examples of that class by integer largest remainder, breaking equal remainders by lower client identifier. Independently shuffle original row identifiers per class with PCG64/SeedSequence$(20261301,c)$ and assign consecutive quota-sized slices. The assignment is identical across all seeds, methods, and predictors of a task. It contains no overlap or omitted examples. This preserves a TRAIN-derived client-given-class allocation, not an exact match of each client's TRAIN class prior; class balancing and rounding change the realized prior. We save every realized prior difference and assignment hash. This is an explicit study partition rather than an official federated benchmark split.

The primary endpoint is TEST client-macro NLL, with client-macro accuracy reported alongside it. Three training seeds are the unit of replication; all tables use mean $\pm$ sample standard deviation with denominator $3-1$. We show individual paired differences rather than treating clients or rounds as independent seeds. Brier, client-macro ECE15, pooled ECE15, weighted accuracy/NLL, and the lower client-accuracy decile are separate endpoints. No significance or broad-distribution robustness claim is inferred from three runs on one fixed partition per task.

The calibration controls all fit the same TRAIN logits with the specified fixed penalties; they are regularized TRAIN-fitted variants of known score families. The matched Linear control trains one local epoch only after the shared backbone is frozen. The FedRoD-linear reference instead retains its private head throughout federated training and uses an explicitly documented generic-plus-residual endpoint. TopK+EF uses the same 54,808 B cap and the same prior-vector fit, with 741 B more actual payload than uniform3. These two stronger references are run on CIFAR-10 only. We do not present them as exact official implementations.

The upload study is separate. It tests canonical packets, replay, durable state transitions, crash recovery, and full-dimensional decoded aggregation locally. The end-to-end FL learning runs assume reliable synchronous delivery. A real ledger, network, and consensus are outside the executed experiment scope.
'''
    return r'''
\section{评估协议}
开发数据包括 CIFAR-10 的 37,500 条 TRAIN 与 12,500 条客户端留出数据，以及 FashionMNIST 的 45,001 条 TRAIN 与 14,999 条客户端留出数据。模型、分数变换族、正则项、最终轮次和每任务三个训练种子均在官方 TEST 推理之前固定。每任务使用其全部 10,000 条官方 TEST 数据，每条恰好分配给二十个客户端中的一个。由于原数据集不提供本文使用的客户端划分，我们明确规定独立的标签偏斜 TEST 协议。

对类别 $c$，由本地 TRAIN 计数定义 $P(i\mid c)=(n_{ic}+1)/\sum_j(n_{jc}+1)$。该类别的 1,000 条官方 TEST 样本按整数最大余数法分配，相同余数优先分给编号较小的客户端。每类别以 PCG64/SeedSequence$(20261301,c)$ 独立打乱原始行号，再按配额连续切分。同任务的全部种子、方法和预测器复用完全相同的分配，不重复、不遗漏。该规则保持的是 TRAIN 导出的给定类别的客户端分配比例，并不使每个客户端的 TEST 类先验精确等于 TRAIN；全局类别平衡及整数化都会产生差异。我们保存实际先验差异与划分哈希。这是本文显式定义的研究划分，不是官方联邦基准划分。

主要指标为 TEST 客户端宏平均 NLL，同时报告宏平均准确率。独立重复单位是三个训练种子，表格报告均值与分母为 $3-1$ 的样本标准差。我们给出全部种子的配对差值，不将客户端或训练轮次当作独立训练重复。Brier、客户端宏平均 ECE15、汇总 ECE15、样本加权准确率/NLL 与客户端准确率下十分位数均单独计算。一个固定划分上的三次训练不支持广泛分布鲁棒性或统计显著性结论。

校准对照均以相同 TRAIN logits 和固定正则系数拟合，属于已有分数变换族的 TRAIN 正则化实现。匹配的 Linear 对照在冻结共享特征后仅训练本地一轮；FedRoD-linear 则在联邦训练全程保留私有头，并采用明确说明的通用分支加残差预测终点。TopK+EF 使用相同 54,808 B 预算上限及相同先验向量适配，实际载荷比 uniform3 多 741 B。这两项强参考仅在 CIFAR-10 运行，均不宣称精确复现官方实现。

上传协议实验独立开展，检查规范化报文、重放、持久状态转换、崩溃恢复及全维解码聚合。实际 FL 学习实验假定可靠同步交付。真实账本、网络与共识不属于已执行范围。
'''

def algorithm_text(lang):
    if lang=='en':return r'''
\paragraph{PR-LFL algorithm summary.}
\begin{enumerate}
\item Count each client's TRAIN labels and initialize a common shared model and zero FP32 residuals.
\item For each fixed federated round, train locally with the defined loss, add the committed residual to the model difference, and encode uniform3 blocks.
\item Strictly decode actual payload bytes, aggregate sample-weighted updates, and retain the quantization difference as the next residual in the reliable-delivery learning experiment.
\item Freeze the final shared network. Forward each local TRAIN example once, compute the smoothed prior anchor, and solve the fixed regularized affine problem.
\item Store twenty FP32 values locally; predict with $\operatorname{softmax}(A_i\odot z+b_i)$ without uploading the fitted parameters.
\item In the separately evaluated durable route, persist pending bytes and residual, register/authenticate context, and replace the committed residual only on a matching incorporated receipt.
\end{enumerate}
'''
    return r'''
\paragraph{PR-LFL 算法概要。}
\begin{enumerate}
\item 统计各客户端 TRAIN 类别计数，初始化相同共享模型及全零 FP32 残差。
\item 每个固定联邦轮次先进行规定损失的本地训练，将已提交残差加入模型差值，再编码为 uniform3 分块报文。
\item 对实际载荷字节严格解码，按样本数加权聚合。在可靠交付的学习实验中，将量化差值保留为下一轮残差。
\item 冻结最终共享网络，对每个本地 TRAIN 样本增加一次前向，计算平滑先验锚点并求解固定正则仿射目标。
\item 本地保存二十个 FP32 数值，用 $\operatorname{softmax}(A_i\odot z+b_i)$ 预测，无需上传适配参数。
\item 独立评估的持久协议路径先保存 pending 字节及候选残差，登记并认证上下文；仅在匹配的 incorporated 回执到达后替换已提交残差。
\end{enumerate}
'''

def results_text(lang,s,d,rows):
    c=s['cifar'][MAIN];f=s['fashion'][MAIN]
    delta=lambda task,variant,metric,scale=1:f"{-d[task][variant][metric]['mean']*scale:.4f}"
    if lang=='en':
        text=r'''
\section{Results}
\subsection{Personalized probability prediction}
\input{tables/main}
On the locked official TEST assignment, the prior-vector predictor reaches CIFAR-10 accuracy %s\%% and NLL %s; FashionMNIST reaches %s\%% and NLL %s. Compared with same-dimensional zero-anchor vector scaling, mean NLL is lower by %s and %s, respectively. The corresponding mean accuracy differences are %s and %s percentage points. Table~\ref{tab:seeds} reports every paired seed. This comparison holds the deployed dimensionality and shared model fixed, locating the measured benefit in the count-derived bias anchor rather than additional affine parameters.

\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{figures/prediction.pdf}
\caption{Individual training seeds (small markers) and means with sample SD (large markers and bars). The axes show accuracy and NLL separately; uncertainty is across exactly three training seeds.}\label{fig:prediction}
\end{figure}

The anchor is also more effective than simply adding the smoothed TRAIN log prior: NLL falls by %s on CIFAR-10 and %s on FashionMNIST. However, a prior-anchored scalar-scale BCTS control recovers much of the result. The observed mean NLL reduction relative to same-prior BCTS is only %s and %s. This difference also involves scale constraints and regularized dimensionality. The evidence therefore supports prior-regularized local score adaptation, while the narrower claim that independent class scales are essential is not established.

\input{tables/quality}
Table~\ref{tab:quality} keeps Brier and ECE separate from NLL. Prior vector has lower mean NLL but higher mean client ECE than same-prior BCTS on both tasks. A lower log loss is not used to infer better calibration under every metric, and pooled ECE is not substituted for average client ECE. The same-prior BCTS result is retained even when its smaller state is competitive.

\subsection{Training and compression attribution}
\input{tables/training}
CE training followed by the same prior-vector fit is close to the FedLC-inspired version. The data do not establish that count-adjusted local training is necessary for the final predictor. The dense reference similarly gives a direct accuracy/NLL comparison at the same adaptation rule. The lightweight claim is based on measured serialized uploads and the resulting predictive tradeoff, rather than a claim that quantization improves optimization.

\input{tables/reference}
The CIFAR-10 reference table compares fixed three-seed TopK+EF and persistent FedRoD-linear implementations on exactly the same TEST assignment. TopK uses the same capacity limit but its actual transmitted payload is larger. FedRoD stores and trains a persistent local residual head; it answers a different personalization question from post-training affine fitting. These references support a scoped empirical comparison, without converting implementation variants into an all-method state-of-the-art claim.

\subsection{Bytes, state, and durable integrity}
Uniform3 generates 54,067 B per upload, including block scales and headers. Relative to the analytical dense profile of 547,560 B, the uplink reduction is 90.13\%%. With an unchanged dense downlink, a complete upload--download exchange decreases from 1,095,120 B to 601,627 B, a 45.06\%% reduction. The residual still occupies 547,496 B per client. The local predictor adds 80 B, compared with 44 B for prior BCTS and 2,600 B for the last-Linear or residual-head state. The difference between 44 B and 80 B is negligible compared with the complete model and residual; it is not an overall-memory-halving claim. No adaptation fit requires a new CNN backward, but the TRAIN forward pass, logit cache, and numerical solve remain real costs. The matched head also avoids feature-extractor backpropagation, so this property alone does not establish an end-to-end runtime or energy advantage.

The durable protocol suite passes 62 of 62 local checks. A separate full-dimensional integration using two distinct real 54,067 B training payloads passes 186 of 186 checks, including 32 mutation/context rejection cases, actual subprocess crashes, lost acknowledgements, replay, and idempotent residual commitment. Independently recomputed weighted aggregation agrees bit for bit; an incorrect equal-weight counterexample differs by 0.0153089 at the largest coordinate. A complete authenticated application message is 54,405 B: 54,067 B payload, 306 B context, and a 32 B HMAC tag, adding 0.6252\%% to the codec payload. These counts exclude ledger/network framing and consensus traffic. The three fixture databases occupy 14,045,184 B in total; that fixture storage is not a production blockchain-cost estimate.

The integration uses constructed updates around replayed training payloads and tests local transaction recovery. It is not an authenticated end-to-end training run, and SQLite does not supply a distributed ledger. A negative control also confirms that a matching codec-specification digest cannot establish correct decoder execution. Correct decoding and the key-holding server remain trusted.
'''%(f"{100*c['macro_accuracy']['mean']:.2f}",f"{c['macro_nll']['mean']:.4f}",f"{100*f['macro_accuracy']['mean']:.2f}",f"{f['macro_nll']['mean']:.4f}",
         delta('cifar','zero_anchor_vector','macro_nll'),delta('fashion','zero_anchor_vector','macro_nll'),
         f"{100*d['cifar']['zero_anchor_vector']['macro_accuracy']['mean']:+.4f}",f"{100*d['fashion']['zero_anchor_vector']['macro_accuracy']['mean']:+.4f}",
         delta('cifar','analytic_prior','macro_nll'),delta('fashion','analytic_prior','macro_nll'),delta('cifar','prioranchored_bcts','macro_nll'),delta('fashion','prioranchored_bcts','macro_nll'))
        return text
    return r'''
\section{实验结果}
\subsection{个性化概率预测}
\input{tables/main}
在冻结的官方 TEST 客户端划分上，先验向量预测器的 CIFAR-10 宏平均准确率为 %s\%%，NLL 为 %s；FashionMNIST 分别为 %s\%% 和 %s。与相同维数的零锚点向量适配相比，两任务的平均 NLL 分别降低 %s 和 %s，准确率平均变化分别为 %s 与 %s 个百分点。表~\ref{tab:seeds} 给出每个种子的配对结果。该对照固定共享模型和部署维数，因此测得的差异来自计数导出的偏置锚点，而不是增加仿射参数。

\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{figures/prediction.pdf}
\caption{小标记为独立训练种子，大标记及误差条为均值与样本标准差。准确率和 NLL 分别展示，不确定性来自每任务恰好三个训练种子。}\label{fig:prediction}
\end{figure}

与直接将平滑 TRAIN 对数先验加到共享 logits 相比，先验向量在 CIFAR-10 与 FashionMNIST 上的平均 NLL 分别降低 %s 与 %s。不过，相同先验锚点的标量尺度 BCTS 已恢复大部分效果，相对同先验 BCTS 观察到的平均 NLL 降幅仅为 %s 与 %s。该差异同时涉及尺度约束与正则化维数。因此证据支持先验正则化本地分数适配，但不能据此宣称独立类别尺度不可或缺。

\input{tables/quality}
表~\ref{tab:quality} 将 Brier 和 ECE 与 NLL 分开。两个任务上，先验向量比同先验 BCTS 的平均 NLL 更低，但客户端平均 ECE 更高。较低对数损失不能自动推出全部校准指标改善，汇总 ECE 也不替代客户端平均 ECE。具有更小状态的同先验 BCTS 对照始终保留。

\subsection{训练与压缩归因}
\input{tables/training}
普通 CE 训练后采用同一先验向量适配，与 FedLC 启发式版本接近。现有数据不能证明计数调整训练损失是最终预测器的必要组成。稠密参考以同一适配规则直接比较准确率和 NLL。轻量化主张基于真实序列化上传字节及相应预测权衡，不宣称量化改善优化本身。

\input{tables/reference}
CIFAR-10 强参考表在完全相同的 TEST 划分上比较固定三个种子的 TopK+EF 与持久 FedRoD-linear 实现。TopK 具有相同预算上限，但实际载荷更大。FedRoD 在训练全程存储并训练本地残差头，研究的个性化方式与训练后仿射适配不同。该结果支持限定范围内的实证比较，不将实现变体包装为全面优于现有方法的结论。

\subsection{字节、状态与持久完整性}
Uniform3 每次实际上传 54,067 B，已含分块尺度和报头。相对 547,560 B 的解析稠密格式，上行减少 90.13\%%。若下行仍为稠密，一次上传加下载从 1,095,120 B 降至 601,627 B，减少 45.06\%%。每客户端残差仍占 547,496 B。先验向量新增 80 B 本地部署状态，先验 BCTS 为 44 B，末层 Linear 或残差头为 2,600 B。44 B 与 80 B 的差距相对完整模型及残差很小，不能写成整体内存减半。适配无需新增 CNN 反向传播，但 TRAIN 前向、logit 缓存和数值求解均有实际成本。匹配头也不反传特征提取器，因而该属性本身不能证明端到端速度或能耗优势。

持久协议测试通过全部 62 项本地检查。使用两个不同真实训练载荷的全维集成另通过全部 186 项检查，包括 32 项篡改/上下文拒绝，以及真实子进程崩溃、丢失确认、重放和幂等残差提交。独立重算的加权聚合逐位一致，错误等权反例的最大坐标差为 0.0153089。完整认证应用报文为 54,405 B：载荷 54,067 B、上下文 306 B、HMAC 标签 32 B，比编解码载荷增加 0.6252\%%。该计数不含账本/网络封装及共识流量。三个测试数据库共占 14,045,184 B，仅为测试存储，不能视为生产区块链成本。

该集成围绕重放的训练载荷构造更新，验证本地事务恢复，不是经认证的端到端联邦训练；SQLite 不提供分布式账本。负对照还表明相同 codec 规范摘要不能证明解码程序执行正确。解码器与持钥服务器仍属于可信边界。
'''%(f"{100*c['macro_accuracy']['mean']:.2f}",f"{c['macro_nll']['mean']:.4f}",f"{100*f['macro_accuracy']['mean']:.2f}",f"{f['macro_nll']['mean']:.4f}",
         delta('cifar','zero_anchor_vector','macro_nll'),delta('fashion','zero_anchor_vector','macro_nll'),
         f"{100*d['cifar']['zero_anchor_vector']['macro_accuracy']['mean']:+.4f}",f"{100*d['fashion']['zero_anchor_vector']['macro_accuracy']['mean']:+.4f}",
         delta('cifar','analytic_prior','macro_nll'),delta('fashion','analytic_prior','macro_nll'),delta('cifar','prioranchored_bcts','macro_nll'),delta('fashion','prioranchored_bcts','macro_nll'))

def abstract_text(lang,s,d):
    c=s['cifar'][MAIN];f=s['fashion'][MAIN]
    reductions=[-d[t]['zero_anchor_vector']['macro_nll']['mean'] for t in TASKS]
    if lang=='en':return r'''\begin{abstract}
Label-skewed federated clients can require different probability predictors even when they share a communication-constrained backbone. We study prior-regularized lightweight federated prediction: three-bit block-quantized uploads with error feedback, followed by a TRAIN-only local affine transformation whose bias is anchored to a smoothed label prior. The transformation stores twenty FP32 values per ten-class client and requires no additional CNN backward or upload. Under a fixed, explicitly defined label-skew assignment of official TEST data, three training seeds per task give mean NLL %s on CIFAR-10 and %s on FashionMNIST, reductions of %s and %s against same-dimensional zero-anchor vector adaptation. Mean accuracies are %s\%% and %s\%%. A same-prior scalar-scale control explains much of the benefit, bounding the role of class-specific scales. Actual codec payloads are 54,067 B versus an analytical dense profile of 547,560 B. A separate context-bound HMAC protocol couples residual commitment to durable incorporation; local protocol and full-payload suites pass 62/62 and 186/186 checks. The executed integrity evidence covers local crash/replay recovery, while real-ledger and authenticated end-to-end training experiments remain future integration work.
\end{abstract}
\noindent\textbf{Keywords:} lightweight federated learning; label skew; personalized prediction; prior regularization; error feedback; upload authentication.
'''%(f"{c['macro_nll']['mean']:.4f}",f"{f['macro_nll']['mean']:.4f}",f'{reductions[0]:.4f}',f'{reductions[1]:.4f}',f"{100*c['macro_accuracy']['mean']:.2f}",f"{100*f['macro_accuracy']['mean']:.2f}")
    return r'''\begin{abstract}
标签分布偏斜的联邦客户端即便共享通信受限的主干模型，也可能需要不同的概率预测器。本文研究先验正则化轻量化联邦预测：以三比特分块量化与误差反馈压缩上传，再用纯 TRAIN 数据拟合本地仿射变换，将偏置正则化到平滑标签先验。十分类客户端仅存储二十个 FP32 适配值，无需新增 CNN 反向传播或上传。在显式定义并冻结的官方 TEST 标签偏斜划分上，每任务三个训练种子的 CIFAR-10 与 FashionMNIST 平均 NLL 分别为 %s 和 %s，相对同维零锚点向量适配降低 %s 与 %s；平均准确率分别为 %s\%% 与 %s\%%。相同先验的标量尺度对照解释了大部分收益，限定了类别独立尺度的作用。实际编解码载荷为 54,067 B，解析稠密格式为 547,560 B。另行设计的上下文绑定 HMAC 协议将残差提交与持久纳入聚合关联，本地协议与全载荷测试分别通过 62/62 与 186/186 项检查。已执行证据覆盖本地崩溃与重放恢复，真实账本及经认证的端到端训练仍需集成验证。
\end{abstract}
\noindent\textbf{关键词：}轻量化联邦学习；标签偏斜；个性化预测；先验正则化；误差反馈；上传认证。
'''%(f"{c['macro_nll']['mean']:.4f}",f"{f['macro_nll']['mean']:.4f}",f'{reductions[0]:.4f}',f'{reductions[1]:.4f}',f"{100*c['macro_accuracy']['mean']:.2f}",f"{100*f['macro_accuracy']['mean']:.2f}")

def appendix_text(lang,gate):
    title='Supplementary Protocol and Complete Seed Results' if lang=='en' else '补充协议与完整种子结果'
    return '\\appendix\n\\section{'+title+'}\n\\input{tables/seeds}\n'+gate+'\n'+(
        r'''The development question was prospectively fixed after earlier development work. Earlier accuracy-oriented candidate failures were not reversed by changing thresholds; the current study tests a separately declared NLL question. The official TEST experiment is a final fixed-state assessment, with no further tuning. The complete raw metrics, source identities, memberships, optimizer states, actual payloads, and negative controls accompany the experimental bundle.

\paragraph{Data, code, and reproducibility.} Public CIFAR-10 and FashionMNIST provide image labels. The bundle records official file identities, TRAIN/holdout/TEST membership, all three seed results, checkpoints, FP32 adaptation states, score exports, solver/codec source digests, and the exact scripts. Local protocol keys are excluded. The figures show raw seeds and sample SD. Generic article formatting is used in this draft; the final venue template and page limit need a separate check. Local independent numerical checks are distinct from unavailable external cross-model review.
''' if lang=='en' else r'''当前开发问题是在此前开发工作之后前瞻固定的 NLL 问题。此前以准确率为目标的候选失败结论没有通过改阈值翻转。官方 TEST 作为固定状态的最终评估，不再据此调参。实验包保留完整指标、来源身份、样本成员关系、优化状态、实际载荷及负对照。

\paragraph{数据、代码与复现。} 图像标签来自公开 CIFAR-10 与 FashionMNIST。实验包记录官方文件身份、TRAIN/留出/TEST 成员关系、三个种子结果、检查点、FP32 适配状态、分数导出、求解器/编解码来源摘要与完整脚本；不含本地协议私钥。图中显示原始种子与样本标准差。本文当前采用通用 article 排版，正式会议模板和页数限制需另外核对。本地独立数值检查与尚不可用的外部跨模型审查明确区分。
''')

def checklist_text(lang):
    return '''# 实验完成范围与后续清单

已完成：两任务各三个固定训练种子；主三训练配置、TRAIN-only 适配及机制对照；官方 TEST 固定状态评估；CIFAR TopK+EF 与 FedRoD-linear 三种子参考；真实编解码载荷；本地持久协议 62/62 与不同真实首包全维集成 186/186；中英文稿与数值来源表。

后续真实系统实验（不能写成已完成）：

1. 选定实际许可链/测试网，实现登记、交易回执与上传验证；保留交易标识、链上字段、合约版本、节点数、共识及最终性设置。
2. 将真实 FL 每轮训练、量化、pending 状态、链上登记、聚合和 EF receipt 提交接成端到端，比较启用/关闭协议的模型轨迹与失败恢复。
3. 测量真实网络与链的吞吐、p50/p95 登记/确认时延、总字节/交易费用/存储，区分 codec、认证报文、RPC 与共识成本。
4. 在真实接入流程重做篡改、冒充、重放、跨轮/跨模型替换、丢包、超时、掉线与重启；验证错误拒绝及正常报文接受。

可选扩展：更多异构程度、部分参与、长期训练、设备/能源及分布漂移。这些不属于当前固定两任务三种子研究的已完成范围，也不作为当前结论前提。无须为扩展再增加当前种子数量。

安全边界：当前是外置 HMAC 完整性认证与本地事务恢复，非权重嵌入水印、公开签名、不可否认、防投毒或诚实训练证明。正确解码、密钥服务、服务器及回执可信。62/186 均为本地验证案例数，非统计安全概率。
'''

def conclusion_text(lang):
    if lang=='en':return r'''
\section{Conclusion}
PR-LFL combines decoded low-bit federated updates with a small TRAIN-fitted prior-regularized predictor. Fixed three-seed evaluations on two label-skew tasks support lower client-macro NLL relative to the same-dimensional zero-anchor control while preserving classification performance. The comparison with same-prior BCTS places much of the explanatory weight on local prior regularization. Exact byte accounting and separately tested receipt-dependent residual handling connect this predictive workflow to an auditable upload interface. Actual ledger execution and authenticated end-to-end learning are the next integration steps.
'''
    return r'''
\section{结论}
PR-LFL 将经解码的低比特联邦更新与仅由 TRAIN 拟合的小型先验正则预测器结合。两个标签偏斜任务的固定三种子评估支持：相对同维零锚点对照，客户端宏平均 NLL 降低，同时保持分类性能。同先验 BCTS 对照表明，本地先验正则化解释了大部分收益。精确字节计量及独立测试的回执依赖残差提交，将预测流程与可审计的上传接口联系起来。真实账本执行和经认证的端到端学习是下一步集成工作。
'''

def readme_text(lang):
    return '''# 当前论文版本

打开 main.pdf 阅读；main.tex 与 sections/、tables/、figures/、references.bib 为完整 LaTeX 来源。英文/中文对应的证据使用同一组冻结结果。

本稿替换早期 TopK 15 轮稿。早期版本保存在 outputs/previous_revision，对应桌面统一 paper 内归档，不应与当前结果混用。

数字来源见 evidence/PAPER_NUMBERS.json 和 individual_seed_metrics.csv。EXPERIMENT_CHECKLIST.md 明确真实上链与端到端训练尚未执行的部分。本文当前为通用可编译论文版式，未声称符合某一届 BlockSys 的正式模板/页数。

参考文献按首次引用顺序由 BibTeX unsrt 自动编号；20条文献元数据及核验范围见 checks/REFERENCE_AUDIT.md。此前三项次级来源条目已用官方 arXiv/会议资料核实；未补造出版信息。

本版本使用 paper-writer、scientific-visualization 等已列入技能接续记录的学术技能。本地独立数值/来源检查不冒充外部跨模型审查。
'''

if __name__=='__main__':main()

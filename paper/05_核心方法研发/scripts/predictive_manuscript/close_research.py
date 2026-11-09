"""Close the fixed experiment scope and prepare reviewable delivery state."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'
BASE=ROOT/'work/predictive_pfl_confirm'
FINAL=ROOT/'work/predictive_final_test'
STAMP=datetime.now().astimezone().isoformat()

def main():
    checks=json.loads((OUT/'PAPER_DELIVERY_CHECKS.json').read_text())
    assert checks['status']=='PASS'
    for name in ('INDEPENDENT_FINAL_CHECK.json','INDEPENDENT_FINAL_CHECK_FASHION.json'):
        assert json.loads((FINAL/name).read_text())['status']=='PASS'
    confirmations={}
    for task,partition,seeds in (('cifar',20261217,(20261231,20261232,20261233)),('fashion',20261219,(20261241,20261242,20261243))):
        reports=[json.loads((BASE/f'assessment_{task}_p{partition}_s{s}/result.json').read_text()) for s in seeds]
        zero=np.array([r['differences']['zero_vector']['nll_reduction'] for r in reports])
        prior=np.array([r['differences']['prior']['nll_reduction'] for r in reports])
        all_accuracy=all(all(value for key,value in r['first_seed_conditions'].items() if 'accuracy' in key) for r in reports)
        supported=bool((zero>0).all() and (prior>0).all() and zero.mean()>=.01 and prior.mean()>=.01 and all_accuracy)
        assert supported
        confirmations[task]={'status':'DEVELOPMENT_TASK_SUPPORTED','seeds':list(seeds),
            'accuracy_preserved_each_seed':all_accuracy,'nll_reduction_vs_zero_each_seed':zero.tolist(),
            'nll_reduction_vs_prior_each_seed':prior.tolist(),'mean_nll_reduction_vs_zero':float(zero.mean()),
            'mean_nll_reduction_vs_prior':float(prior.mean()),'scope':'development only; official TEST reported separately'}
    (BASE/'TWO_TASK_DEVELOPMENT_SUMMARY.json').write_text(json.dumps(confirmations,indent=2)+'\n')
    report='''# 当前成果与阅读入口

本轮“前半部分核心方法设计 + 可靠实验 + 中英文论文材料”已完成，停止新增方向、种子与参数搜索。所有结果采用每任务固定三个训练种子。论文不宣称已经完成真实链部署。

核心方法为 PR-LFL：共享模型采用真实三比特分块量化与误差反馈上传；训练结束后冻结共享网络，以各客户端 TRAIN 类别计数导出的先验为偏置锚，拟合本地正则仿射预测器。十分类保存20个 FP32 参数（80 B），无需上传这些适配值。其优势主要来自异构客户端的先验正则化，而不是一个新量化器或算法原子首创。

| 官方 TEST，三个训练种子 | 准确率（%） | NLL | 同维零锚点向量的 NLL | NLL 降低 |
|---|---:|---:|---:|---:|
| CIFAR-10 | 75.57 ± 0.60 | 0.6446 ± 0.0138 | 0.6804 ± 0.0140 | 0.0358 |
| FashionMNIST | 92.97 ± 0.44 | 0.1880 ± 0.0110 | 0.2062 ± 0.0123 | 0.0182 |

均为客户端宏平均，±是三个独立训练种子的样本标准差。官方 TEST 原图像和标签来自数据集；客户端划分由冻结的 TRAIN 计数规则生成，属于本文定义的标签偏斜协议，不是官方联邦划分。相对零锚点向量，平均准确率分别变化 +0.1777 与 +0.1730 个百分点。

CIFAR 强参考也完成相同三个种子：TopK+EF+相同先验向量为74.27% / NLL0.6699，FedRoD-linear为74.30% / NLL0.6772。均为说明清楚的实现变体，不宣称精确复现作者代码。TopK实际54808 B，主方法54067 B，两者预算上限相同而实际字节不同。

更简单的同先验 BCTS 与主方法接近，且两任务平均客户端 ECE 更低；不能把20维类别尺度说成必要突破。普通 CE 训练配合相同适配也接近，不能声称 FedLC-inspired 损失必不可少。论文已保留这些强对照并解释其含义。

资源方面，上行54067 B相对解析稠密547560 B减少90.13%；含不变稠密下行的一次往返仅减少45.06%。80 B适配状态不含547496 B残差、共享模型、TRAIN前向、缓存和求解，未宣称能耗或端到端速度优势。

后半部分已完成本地持久完整性协议62/62检查，以及两个不同真实训练载荷的全维集成186/186检查；含篡改、上下文替换、重放、真实崩溃、丢失确认及幂等残差提交。它是外置HMAC认证与本地事务恢复，不是权重嵌入水印，也不证明诚实训练。真实链、网络及经认证的完整FL训练未执行，已单列后续实验清单。

阅读顺序：

1. 桌面 paper/04_中文论文/main.pdf：中文完整稿，10页。
2. paper/03_英文论文/main.pdf：英文完整稿，10页；同目录含LaTeX完整来源和可复制打包文件。
3. paper/06_核心方法研发：实际源代码、训练检查点、适配状态、逐种子结果、官方TEST分数导出、协议实验与复现材料。
4. 各论文目录的 EXPERIMENT_CHECKLIST.md：真实系统尚未执行事项。

参考库20条，正文实际引用18条，按首次出现顺序自动编号。两语言数值表逐字相同；编译通过，无未定义引用、缺字或越界排版警告，最终逐页渲染已检查。两任务各9个主模型的导出分数与指标已独立重算核验；外部跨模型审查仍不可用，不冒称通过。

目前采用通用article版式，正式BlockSys年份/track模板、页数与投稿信息还需最终核对。这不影响前半部分方法和实验结果用于写作；不把当前草稿称为已满足全部投稿要求。
'''
    (OUT/'当前成果与阅读入口.md').write_text(report)
    state=OUT/'method_followup'
    old=state/'00_当前研究状态.md'
    backup=state/'checks/研究状态历史_最终交付前.md'
    backup.parent.mkdir(parents=True,exist_ok=True)
    if old.exists() and not backup.exists():shutil.copy2(old,backup)
    old.write_text(report+'\n更新时间：'+STAMP+'\n')
    running=json.loads((state/'RUNNING_STATE.json').read_text())
    running.update(updated_at=STAMP,active=[],active_agents=[],
        research_phase='COMPLETE_FIXED_SCOPE: method, two tasks x three seeds, official TEST, strong references, bilingual papers',
        objective_status='completed for front-half core method and evidence; real-chain integration explicitly separate',
        remaining_required_for_current_objective=[],
        later_system_experiments=['real ledger/network measurements','authenticated end-to-end FL integration'],
        seed_policy='exactly three predefined training seeds/task; no further seeds',
        final_results={'cifar':'work/predictive_final_test/official_cifar/result.json',
            'fashion':'work/predictive_final_test/official_fashion/result.json',
            'references':'work/predictive_final_test/reference_preflight/result.json'},
        paper_delivery_checks='outputs/PAPER_DELIVERY_CHECKS.json')
    running['completed_sessions']=sorted(set(running.get('completed_sessions',[])+[47740,50453,17992,26022,35447,69490,38349,34717,21968]))
    (state/'RUNNING_STATE.json').write_text(json.dumps(running,indent=2,ensure_ascii=False)+'\n')
    with (state/'PROJECT_STATE.md').open('a') as handle:
        handle.write('\n\n## '+STAMP+' 最终固定范围完成\n\n'+report+'\n')
    with (state/'技能使用与任务接续.md').open('a') as handle:
        handle.write('''\n\n## 最终交付阶段（2026-11-04）

继续使用 paper-writer、experiment-audit、scientific-visualization 及 PDF 渲染核查。参考文献用 citation-management 工作流做元数据核对；首次引用使用 unsrt 自动编号。experiment-audit 的外部跨模型后端不可用，因此只报告实际独立数值/来源核查，不标外审PASS。流程图的软件指导文献已核实并在中英文软件说明处致谢。

当前方法与实验已冻结并完成，不复活旧STOP路线、不新增种子、不使用TEST调参。下一阶段若继续，仅以用户新要求处理真实链集成或正式投稿排版。
''')
    (state/'METHOD_EVIDENCE_MAP.md').write_text('''# 当前核心贡献与证据

先验锚定的20维本地仿射适配：两任务各三个固定训练种子，开发规则均满足；官方TEST由预先冻结模型/适配状态/协议执行。完整结果和独立检查在 results/predictive_final_test；旧准确率命题STOP不改写。

主要支持：相对同维零锚点向量的NLL降低并保持准确率；CIFAR同预算TopK与持久FedRoD-linear参考。相同先验BCTS接近，FedLC-inspired与CE适配接近，因此不能宣称这些组件分别必需。

完整性：durable_integrity_v2协议62/62，full_payload_integrity不同载荷集成186/186，均为本地simulation；不声称真实链/网络/端到端FL或防投毒。上下文身份不证明decoder执行。

最终论文数值和来源SHA见英文/中文论文evidence/PAPER_NUMBERS.json；编译与引用顺序检查见checks/FINAL_CHECKS.json。真实系统未做部分见EXPERIMENT_CHECKLIST.md。
''')
    print(json.dumps({'status':'CURRENT_OBJECTIVE_COMPLETE_PREPARED_FOR_PUBLISH','active_experiments':0,'paper_checks':'PASS'}))

if __name__=='__main__':main()

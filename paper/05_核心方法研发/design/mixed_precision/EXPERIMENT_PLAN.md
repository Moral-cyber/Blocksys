# 标签校准与预算内量化：分阶段组合式开发

2026-11-04。先前控制诊断显示当前标签异构的dense代价约11.08pp，而TopK-EF附加代价约1.04pp；不以压缩额外放大异构作为既定动机。目标改为在明确字节上限下改善异构客户端的信息保留与全局学习。

## 已有部件与精确定义

标签校准参考FedLC，但原文正文/附录分母不同、count0与tau无可执行官方规则。本实验明确使用附录式全类别交叉熵变体：训练logit z'_c=z_c-max(n_c,1)^(-1/4)，tau=1；推理用raw z。max(n,1)与固定tau为本实验规则，不能写成官方复现或新发明。tau0路径等于原CE但本次不调tau。完整模型仍同样更新/压缩，没有分类头豁免。

通信候选：固定256连续坐标块，FP32块scale，2/3/4bit对称量化，全坐标发送，正常EF。uniform3与mixed_mse、mixed_sensitive使用相同标量量化器；mixed从2bit开始，按实际decoded失真降低/额外字节贪心升级，预算54808B/client，与5% FP32+uint32坐标TopK预算相同上限，实际长度分别报告。uniform3为54067B，不声称实际字节完全相等。header、scale、width和尾部padding全部计入。

mixed_sensitive块权重为本轮所有实际local SGD batch中各坐标梯度平方的均值，再在块内取mean。这是训练路径梯度二阶矩proxy，不是per-example Fisher或Hessian，也不证明全局任务收益。无需额外backward或额外上传权重。mixed_mse权重为1，隔离梯度proxy的作用。固定相同量化scale拟合规则，避免bit分配与量化器混改。

FedFQ已有固定bits下update幅值导向的量化方差分配，FedMPQ已有FL混精度；本候选是已有校准、量化、EF部件的具体组合，不作原子首创声明。必要对照为校准删除、量化/稀疏比较、敏感性删除。是否值得作为核心由实际结果决定。

## Stage1：校准的基础作用

CE/calibratedCE × dense/TopK-EF四组。从头初始化seed20261181，same partition20261143、20clients、alpha.1、37500train、2000stratified globaldev，20轮全参与SGD.03，batch64，localepoch1。MPS本地训练，CPU聚合/编码/评估。全模型、全参数同预算；官方test不读取。没有额外数据、额外梯度、私有validation训练或server数据。

只用last5开发CE和accuracy比较。校准若未比相应CE基线改善CE且accuracy，后续不强留它；量化比较可以用CE目标，必须在tracker准确标注数据依赖选择。校准有效也只能记为已知组件收益。

## Stage2：固定上限下的通信机制

保留同一seed/分区/batch流，所选loss的TopK（Stage1已有）、uniform3、mixed_mse、mixed_sensitive四组。先做编码fixture和运行开销检查；若量化器过慢则记录问题并修实现，不能静默降低训练数据或变更bit语义。

具体保留条件：mixed_sensitive last5 CE低于uniform3和mixed_mse；accuracy至少比uniform3高0.5pp，且不低于TopK超过0.5pp。若未过，停止此敏感性分配规则，不调块长/权重幂/tau/候选bit网格。若unweighted或uniform已解释全部收益，优先简单方案并准确归属，不把已有量化算法收益当自己的新算法。

## 通过后才做的确认

独立数值/代码审查；2个新训练seed与1个独立分区；更温和异构与另一图像任务；同等编码预算/通信曲线；FedFQ类分配与FedLC-TopK强对照；梯度proxy去除/置换与校准删除。最终方法冻结后才访问独立test。当前单seed不冒称可靠泛化、收敛或已成立核心贡献。

所有实际结果、源代码版本、配置、负结果和变化均保留。学术引用整理在核心确认后完成。

# 步数归一与相对尺度压缩：组合式2×2开发对照

2026-11-04。用户允许有理由地组合已有机制，不要求各原子组件首次提出。接收融合/反馈已停止，标签支持强制头上传一次诊断也未改善，不继续这些规则。

## 问题与机制

一个local epoch不等于相同步数。样本数加权FedAvg的delta同时受数据分布和local step数量影响，FedNova已经提出local update归一化。本研究不把FedNova称原创。另一个可部署问题是全模型幅值TopK受参数张量的尺度差异影响；用每张量候选RMS归一的分数选择同K坐标，使相对层内变化参与跨层竞争。这是既有归一思想的组合候选，不预称新算法。

四组：A TopK+EF，B FedNova+TopK+EF，C RMS-score+EF，D FedNova+RMS-score+EF。

无momentum、weightdecay、梯度裁剪的local SGD下tau_i=ceil(n_i/64)。FedNova d_i=delta_i/tau_i，tau_eff=Σp_i tau_i；u_i=d_i+r_i，sent=C(u_i)，r'=u_i-sent，server w+=tau_eff Σp_i sent_i。非Nova组不除步数且server系数1。固定全参与、固定本地数据使tau和p恒定；不声称处理设备速度或dropout。

RMS-score对每个参数tensor l使用score_j=|u_j|/sqrt(mean(u_l²))；全零tensor score为0。全向量按分数取K=6843，发送原始u坐标值与索引，不发送归一分数或尺度。K、FP32/uint32、64B应用头相同，EF守恒。RMS-score不保证TopK最小L2误差；若下游无收益就删除。每张量的score排序可等价为对其局部欧氏尺度缩放。

## 冻结开发协议

清洁索引CIFAR10，20clients alpha=.1、partition20261143，train37500/holdout12500，固定分层2000开发验证。官方test不载入。fresh model seed20261151，20rounds，localepoch1、SGD .03、batch64，MPS本地训练CPU其余。四组相同初始和batch种子，独立训练轨迹。按轮保存全部模型/残差与原始验证。

主读最后5轮验证CE与accuracy，单seed仅筛选。D若不优于B和C的验证CE且accuracy超过至少一个组件0.5pp，则不追加该组合训练；保留组件对照，不调epsilon/RMS幂/预算。此GO门槛定义为CE(D)<min(CE(B),CE(C))且acc(D)>=max(acc(B),acc(C))+.005。若单一组件有效但组合无效，优先简化，并仅为明确组合价值再设计新对照；不能把FedNova本身收益冒充我们的新贡献。

所需后续（仅GO后）：独立实现核查、两新训练seed配对、另一独立分区/异构程度、第二任务、全局TopK与layerwiseTopK、已知尺度敏感方案、通信效率曲线、机制删除。最终test只有方法冻结后才访问；不能用当前dev效果作为最终test精度。

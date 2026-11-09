# TRAIN先验锚定vector：固定一次机制消融

2026-11-04，运行前固定。zero-anchor vector的accuracy较好而NLL较差，10维prior-anchor相反；只在相同20维vector_scaling中把b锚从0改为中心化TRAIN log(count+1)，A锚仍为1。此为已知vector scaling与先验正则组合，不称原子首创，不增评分器。

令f=A⊙z+b，最小化Σlocal TRAIN CE(f,y)+10/2||A−1||²+10/2||b−center(log(n+1))||²。A允许任意实数；bias每步中心化；从anchor起，FP64解析梯度/Hessian，固定ridge10、max50 Newton、Armijo c1=1e−4、步长1/最多40次折半、固定梯度回退及1e−8终止，完全继承冻结calibration_baselines求解公式/常数。不调λ/温度/初始化/步数，不网格。

仅旧prior_bias/diagnostic_v1四模型×20clients保存TRAIN logits/GT/IDs，固定partition20261193；holdout仅最后评分，旧globaldev包含其中，非新test。候选固定fedlc_uniform3，不按分数换模型。完整报告ce_uniform3、fedlc_uniform3、balanced_uniform3、balanced_dense的宏accuracy/NLL、加权与client10%分位；对照同源zero-anchor vector、已存10维prior-anchor、analytic prior、raw及旧head/head+prior。独立wrapper固定SHA复用冻结解析design/读数组/stats，用显式可配anchor新函数，不改旧solver/STOP。

固定diagnostic gates（宏平均）：候选vs10维prior-anchor accuracy≥+.5pp且NLL≤+.005；vs同源zero-vector accuracy≥−.2pp且NLL降低≥.01；vsanalytic prior accuracy≥+.5pp且NLL降低。全过仅FOLLOW_UP探索线索，须新split确认；任一失败STOP此固定配方。原10维六gates完全不变。

先toy有限差分grad/Hessian、凸性/优化下降、TRAIN先验构造、A可为负、anchor删除等价冻结vector fixture；通过后旧数组只跑一次，80个新fit，同时复算80个zero-vector作同源配对核验。冻结全部source/PLAN/input前后SHA。0CNNforward/backward、0数据集字节、0uplink；部署FP32 20维80B/client（相比10维40B）；FP64 design/中间缓存、步数、线搜索、耗时另计，不称零计算或网络实测。

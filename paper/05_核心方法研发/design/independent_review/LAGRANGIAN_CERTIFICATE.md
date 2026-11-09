# Lagrangian 支持选择的数学与数值证书

日期：2026-11-03。范围：固定 u、历史参考 g、K 和 tau 的坐标选择子问题。未修改运行源码，未读取任何确认实验的 test 字段，未运行训练。本文不认定文献新颖性。

## 1. 问题与两项性质

对 |S|=K 定义

\[
A(S)=\sum_{j\in S}u_j^2,\qquad B(S)=\sum_{j\in S}u_jg_j.
\]

令 T 为能量 TopK，E_top=A(T)，c=tau E_top，0<tau≤1。原约束目标为

\[
B^*=\max_{|S|=K,\ A(S)\ge c}B(S).
\]

T 保证可行。对 lambda≥0，假设 S_lambda **精确最大化** B(S)+lambda A(S)。与 T 比较得到

\[
B(S_\lambda)\ge B(T)+\lambda[E_{top}-A(S_\lambda)]\ge B(T),
\]

其中第二个不等式来自 A(S_lambda)≤E_top。

若 S_lambda 还满足能量约束，则按原值保留的输出 s=u masked by S_lambda 满足

\[
\|u-s\|^2=\|u\|^2-A(S_\lambda)
\le(1-\tau K/N)\|u\|^2.
\]

前者是相对固定 T 的历史点积保证，后者是 uniform contraction。二者都不是任务损失或 FL 收敛保证。

## 2. 弱对偶上界与可行解差距

对任意可行 S，因为 A(S)≥c 且 lambda≥0，

\[
\begin{aligned}
B(S)
&\le B(S)+\lambda[A(S)-c]\\
&\le \max_{|R|=K}\{B(R)+\lambda[A(R)-c]\}\\
&=B(S_\lambda)+\lambda[A(S_\lambda)-c]
\equiv D_\lambda.
\end{aligned}
\]

所以 **D_lambda 是 B* 的上界**，不是下界。若 S_lambda 本身可行，B(S_lambda) 是原问题下界，于是

\[
\boxed{0\le B^*-B(S_\lambda)
\le D_\lambda-B(S_\lambda)
=\lambda[A(S_\lambda)-c].}
\]

若 S_lambda 不可行，D_lambda 仍是上界，但 B(S_lambda) 不能充作可行下界。此时可对任意已知可行集合 I（如 T）使用 B(I)≤B*≤D_lambda。

两个充分最优条件：

- **lambda=0 且 S_0 可行**：它已经最大化无约束 B，因此也是约束全局最优，允许正能量松弛。
- **S_lambda 可行且 A(S_lambda)=c**：互补项为零，仍有 B(S_lambda)=B*。

更一般地，lambda[A(S_lambda)-c]=0 足够。条件不是必要的：原问题已经最优时，这个上界仍可能为正。离散问题可能有对偶间隙，因此一般情形不能称为精确约束求解器。

## 3. 当前归一化与浮点限制

代码使用 b_scale=max|u_jg_j|、a_scale=max u_j²，评分为 b_j/b_scale+lambda_norm a_j/a_scale。换回原始 B、A 单位必须使用

\[
\lambda_{raw}=\lambda_{norm}\frac{b_{scale}}{a_{scale}},\qquad
D=B(S_\lambda)+\lambda_{raw}[A(S_\lambda)-c].
\]

不可将 lambda_norm 直接乘以原始能量松弛。零 scale 需要单独处理；当前实现回退 TopK。

上述证明以精确实数和精确标量化 argmax 为前提。若仅知其 argmax 次优误差不超过 eta≥0，安全上界需再加 eta。当前实现使用 FP32 输入与 FP64 计算，没有向外舍入或形式化实数误差认证，因此实践中只能称**带声明容差的数值证书**。实际标量化得分、能量和历史点积仍须重新核验。

有限搜索回退 T 时，返回集合未必是任何所记录 lambda 的标量化 argmax，不能机械套用 eta=0 的证书；保能和相对 T 点积相等仍成立。tau=1 的 TopK 回退也不自动保证在能量并列集合中 B 最大。

## 4. 实际小枚举核验

使用冻结 codec SHA256 `2fbbe6a7afc8664ae6e40516e7978b4adf30830773e55899cdc41bdcb394eb2e`，枚举 N=3/4/5/7、K≤3 的全部 K 坐标集合，核验 720 个随机实例及下列四例。没有加载真实数据或实验 test 输出。

| 情形 | u；g；K；tau | 实际 raw lambda | 选中 B | 枚举 B* | 差距上界 |
|---|---|---:|---:|---:|---:|
| lambda=0，可行且不等于 TopK | (10,9,8)；(0,0,1)；2；.9 | 0 | 8 | 8 | 0 |
| 正 lambda，零松弛 | (4,2,1)；(0,2.25,5)；1；.25 | .16670227 | 4.5 | 4.5 | 0 |
| 离散非精确例 | (sqrt10,sqrt6,1)；(0,4/sqrt6,10)；1；.59 | 1.11132810 | 0 | 约4.00000029 | 4.55644531 |
| 已最优但界不紧 | (10,9,8)；(0,0,1)；1；.9 | .22222656 | 0 | 0 | 2.22226563 |

第二例 A_selected=c=4；第一例能量松弛为 1.1，但 lambda=0，故两例的互补项都为零。第三例明确显示数值证书可以有效却不紧，不能将求解器称为原约束最优。第四例说明上界非零不等于实际解次优。

随机实例中，未加容差的最大上界超出量为 4.44e-16；全部在 `1e-12 × max(1,|相关目标值|)` 的尺度容差内通过。这一舍入现象也是不应宣称形式化精确证书的原因。

完整实际数值与容差：`work/method_design_review/LAGRANGIAN_CERTIFICATE_CHECK.json`。

## 5. 证书不覆盖的主张

- B 仅为固定历史方向投影。即使精确最大化 B，也不能证明当前损失下降、全局精度提高、少数类保护或 FedAvg 收敛；历史误差、曲率、更新幅度和客户端聚合交叉项仍然存在。
- 账本/HMAC 认证的是上传字节和上下文。它并不向服务器提供完整 u、正确 g、lambda 求解轨迹或本地诚实训练证明。这里没有实现链上可核验的 honest-compute、零知识证明或可验证计算协议。
- 离线复核该数值证书需要相关完整向量或另行设计的证明系统；不能把它当成已经随稀疏包免费获得的链上安全性质。
- Lagrangian 弱对偶是标准数学工具。这份证明及小枚举均不建立方法的文献优先权，仍需独立核对相关稀疏压缩研究。

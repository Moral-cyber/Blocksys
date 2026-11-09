# Lagrangian 开发实现只读核查

日期：2026-11-03。结论：**没有发现阻断同快照或单个开发 seed 试跑的问题。** 本次不修改源码、不运行长训练、不读取官方测试结果。属于本地独立代码 review 与小 N 枚举，不是外部跨模型审计。

核对哈希：

- codec：`2fbbe6a7afc8664ae6e40516e7978b4adf30830773e55899cdc41bdcb394eb2e`
- adapter：`b1df038d83adcce18e0aef1315f960f691a29c658d5ef03e8fe13e9811562178`
- checks：`25ddc2cd1681015c817905bbd14d1f1b144f24c49a52005fce5a1c99c0891d24`

## 核查结果

1. **归一化及有效 lambda 正确。** 设 B=max|u*g|、A=max u²，则代码的分数为 b/B+λe/A，等价于 (b+λ_raw e)/B，λ_raw=λB/A。`solver_lambda_effective_raw` 使用的正是该换算。
2. **搜索维护了正确的括区。** lambda=0 实际计算所选能量，不因乘积平局而假设可行；若不可行则有限倍增寻找 high。二分只在实际满足能量时替换 high_support，最后返回保存的可行端点。
3. **数值失效有保守回退。** 有限步找不到可行 high 时回退 TopK。返回前重新构造完整 sent，使用实际 double 能量及对原 reference 的点积检查；没有容忍低于能量阈值或低于 TopK 点积的负裕量，任何失守都回退。严格比较可能导致边界舍入的额外回退，但不会默许违反记录的性质。
4. **指标对象正确。** alignment_gain 相对原始 g 与原 TopK 计算，不是仅比较归一化得分；保能比也使用重建向量的真实平方和。历史点积性质仍不是任务性能证据。
5. **适配器链路正确。** 自己的两个 lagrange 变体先由专用 encode 截获；其他变体透传 controls。原训练与 snapshot 包装保留，codec、adapter 哈希及搜索预算进入 vars(args)，从而进入现有 run_signature。
6. **已有 check 的范围表述恰当。** 随机性质检查、强制回退、离散非最优反例及 tiny train/snapshot 都是实现测试。tiny fixture 复用训练/验证数据是集成测试用法，不可把其中的 accuracy 报作泛化结果；其报告目前明确排除了学习效果主张。

## 独立枚举验证

独立生成 N=3/4/5/7、K≤3、tau=.5/.9/.99 共 540 个案例，对每例枚举全部 K 坐标集合：

- exact K、实际能量下限、实际历史点积不降全部通过；
- 返回支持在所记录 lambda 下均达到标量化得分最大值，最大归一化 argmax gap 为 0；
- raw lambda 换算的最大相对误差约 3.52e-16；
- 记录的二分 low 不可行、high 可行均通过；
- 52 例可行解未达到原始能量约束问题的全局最优投影，说明不能省略“feasible approximation”限定。

证据：`work/method_design_review/LAGRANGIAN_ENUMERATION.json`。没有使用真实训练数据或测试数据，也没有运行训练。

## 非阻断的报告注意事项

- Lag 的 core/filler 字段是“原 TopK 支持的重叠/替换量”，不是 prefix 方法的“固定核心/预留名额”。跨方法图表应使用清楚的标签。
- lambda 原始量纲随客户端 u/g 的尺度改变，不能直接把它当作可比较的异构程度。
- 最坏求解调用预算是 1+24+12 次支持选择；应报告 solver 调用次数、回退率及独立 codec 耗时。
- codec 直接 API 中 tau 由参数决定，方法别名不会自动覆盖 tau；当前 adapter 对 .9/.99 均显式传参，未发现配置错误。
- 通过这些检查只允许进入探索性开发，不支持精确约束最优、FL 收敛、创新优先权或学习优势结论。

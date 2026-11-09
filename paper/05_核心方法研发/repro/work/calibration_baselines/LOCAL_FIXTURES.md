# Calibration baseline synthetic fixtures

2026-11-04，**PASS**。仅NumPy synthetic logits，未打开数据、checkpoint、confirm或旧结果文件。

| 类型 | 维数 | grad FD误差 | Hessian FD误差 | 最小特征值 |
|---|---:|---:|---:|---:|
| zero_anchor_bias | 10 | 8.80817863e-10 | 7.2623435e-11 | 10 |
| temperature_scaling | 1 | 2.46673793e-10 | 1.27887034e-10 | 23.5046305 |
| bcts | 11 | 8.09544209e-10 | 6.71261601e-10 | 10 |
| vector_scaling | 20 | 7.97129918e-10 | 8.52757864e-11 | 10 |

四种均通过score映射独立比对、Jensen凸性、ridge正定。所有含bias形式的共同bias平移不改变CE，目标恰增加对应ridge能量；数据Hessian的gauge方向为零，完整Hessian方向为10倍gauge。均匀零logits、均匀标签下返回anchor且接受步0，state字节40/4/44/80一致。

12个不平衡、单类、大logits拟合例保持有限与单调下降，最多50步且bias中心化。TS反预测已知边界解a=0.001，KKT投影梯度0。完整逐例见LOCAL_FIXTURES.json。

源码身份：
- `PLAN.md`：`33864fe167b461f5d9d7aa8777b932fdfecc752f9553fda14d1fb23c8ad091a9`。
- `solver.py`：`10b11405e6662cb06d08a20549db92e2948b730c426aa1f0b912b958f1211f1c`。
- `local_fixture_check.py`：`8acc5d9cabeb0094b2311f1b61e8dc5b401b0db8b8300fd967b2e8370285165a`。

复算入口为local_fixture_check.py。此处验证数学实现，不作正式数据效果结论。

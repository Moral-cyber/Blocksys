# 11维prior-anchored BCTS control：CPU交付PASS

2026-11-04。只新增control addendum、实现、probe和fixture，主solver9c31/evaluator7b45、主方法和所有gates保持原样。此control没有candidate/decision；真实fresh arrays留给主任务运行一次，不新增种子，每任务三种子规模不变。

固定数学见BCTS_CONTROL_PLAN.md。单正尺度a≥0.001，a锚1，b锚为与20维相同的center(log(TRAINcount+1))，sum TRAIN CE、两ridge10。通过私有function globals绑定**冻结base.fit的原代码对象**与新Problem factory，完整执行原BCTS bound/KKT/Newton/Armijo代码；不修改base全局变量或主vector源。FP32部署11维44B，提升FP64应用与评分。

CPU toy已通过：有限差分grad/Hessian误差9.5191e−10/8.0817e−10；正定/gauge/仅anchor gradient改变；5种case删除anchor后，state、FP64参数与完整轨迹均与冻结BCTS逐位一致。正尺度边界a≈0.0010000000475，KKT投影梯度4.855e−15。原solver globals与主evaluator VARIANTS前后不变。

synthetic confirm3×20clients完整集成通过：实际60个新11维fit，cached20维状态不重拟合，全部7种旧baseline指标与主evaluator相同；44B状态→FP64 score及指标重建误差0。0真实数组fit、0图像、0CNN、0正式训练。证据BCTS_CONTROL_FIXTURES.json/BCTS_CONTROL_INTEGRATION.json。

正式入口（由主任务执行，out必须新建）：

```bash
/opt/anaconda3/envs/Tinyproto/bin/python -B work/prior_vector/bcts_control_probe.py --input-root /fresh原评估数组 --vector-root /同一来源已完成20维适配 --out /新11维control目录 --protocol-plan work/predictive_pfl_confirm/PLAN.md --protocol-registration work/predictive_pfl_confirm/PREREGISTRATION.json
```

输入必须完整3模型×20clients；vector result须为固定7b45 schema，绑定相同原评估、协议、registration路径、所有原行号和原10维state。保留并复核完整paired vector来源/input SHA，不重写原路径。输出原7个对照与prioranchored_bcts的accuracy/NLL/Brier/ECE15、macro/weighted/pooled、44B状态、cost与NPZ。

机制解释：1个scale对10个scale还改变自由度和总ridge罚项。无论11维更好、持平或较差都完整报告，不能把主方法准确率收益自动归因class-specific scales。

| 文件 | SHA256 |
|---|---|
| BCTS_CONTROL_PLAN.md | `9ea93ea8496f33f4f7330a1ea7a3764e3aa101432c04c0bb916903370116d1cc` |
| bcts_control.py | `1dd83808fb0685e7136a24d8c4f9ff711abba754c5a06b0ef17ff40d83c9130f` |
| bcts_control_probe.py | `4f474299a27be96b58d39dd78b2ea5736f4d675fc56b455576f1696c799cb5a8` |
| bcts_control_fixture.py | `9595aafa1bf38a62cdebd0916a3e8549fdb52c7dfc1b5024246b08cc0f2026bd` |
| 主solver.py | `9c3196eb3e6ff7f5f32073e3bc4aae8d986ac7e44bdf0d719258343847d4dbae` |
| 主evaluate_saved.py | `7b45a055df8c3e03421fd9edd6749c1e2bfba1203864e0b41695413db1fb8b05` |

portable复制源码不改SHA。换位置后，先使用`launch.py --entry prior_vector --operation bcts-fixture`生成当前路径的fixture身份；再用`--operation bcts-control`显式参数。当前原工作区fixture不冒充便携路径的新fixture。

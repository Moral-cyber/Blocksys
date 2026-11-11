# PR-LFL 实验线（数据异构 + 紧凑个性化）

对应论文 §3–§5：Table 1、Fig. 2、§5.4。核心主张是「把本地类别先验当作可纠正的正则中心，
用 20 个参数（80 B）改善客户端预测，且异构越强收益越大」。

## 文件清单（每个文件的第一段注释都写明了用途）

| 文件 | 复现论文的哪一部分 | 输入 | 输出 | 耗时 |
|---|---|---|---|---|
| `t1_main_table.py` | Table 1 第 1–5 行；§5.1 的 0.6446/0.1880、0.0358/0.0182、60/60 | `paper/…/evidence/individual_seed_metrics.csv` | `results/t1_main_table.json`、`.md` | 秒级 |
| `f2a_alpha_sweep.py` | Fig. 2(a)；§5.2 | `revision_artifacts/v4_dualtrack/e1/` 的 18 次运行 | `results/f2a_alpha_sweep.json`、`.pdf` | 秒级（聚合） |
| `f2b_client_size.py` | Fig. 2(b)；§5.3 | `route_a_20261105/stats/result.json` | `results/f2b_client_size.json`、`.pdf` | 秒级 |
| `s5_4_permutation.py` | §5.4 类别置换对照 | `05_核心方法研发/results/predictive_pfl_confirm/` | `results/s5_4_permutation.json` | 秒级（复用已存结果） |
| `e1_alpha_training.py` | Fig. 2(a) 的**训练**部分（生成 18 次运行） | CIFAR-10 / FashionMNIST 原始数据 | `revision_artifacts/v4_dualtrack/e1/` | **数小时** |
| `io_schema.py` | 统一的配置读取 / 结果写出（带输入哈希） | — | — | — |
| `download_data.py` | 下载 CIFAR-10 / FashionMNIST | 网络 | `data/` | 视网速 |

## 两条使用路径

**A. 只想核对论文里的数字（推荐）**

```bash
python t1_main_table.py
python f2a_alpha_sweep.py
python f2b_client_size.py
python s5_4_permutation.py
```

四个脚本只读已保存的结果，不做训练，几秒钟出结果。

**B. 想从头重跑训练**

```bash
python download_data.py                 # 下载数据（或复用 paper/02_实验工程/dataset）
python e1_alpha_training.py --stage preflight
python e1_alpha_training.py --stage matrix   # 18 次训练，数小时
python f2a_alpha_sweep.py                    # 再聚合
```

## 为什么脚本是「调用上游」而不是「复制逻辑」

`paper/03_英文论文/revision_artifacts/` 和 `paper/05_核心方法研发/scripts/` 里的实现是论文
证据链的一部分（被 `REVISION_STATUS.json` 记录了哈希）。复制一份逻辑出来，两边会逐渐漂移，
反而说不清哪个才是论文用的版本。

所以这里的脚本只做三件事：**定位上游脚本 → 用正确的参数和路径执行 → 把产物复制到 `results/`**。
上游脚本按 `__file__` 解析相对路径，因此必须用它的真实路径执行（`runpy.run_path`）。

## 依赖

见 `requirements.txt`。核对数字只需要 numpy / scipy / matplotlib；
重跑训练额外需要 torch。

## 已知限制

- `s5_4_permutation.py` 与 `f2a_alpha_sweep.py` 在上游结果已存在时直接复用，不重算
  （上游脚本有「拒绝覆盖已完成诊断」的保护）。要重算，请先手动移走上游的输出文件。
- 数据默认路径写死在上游脚本里（`paper/02_实验工程/dataset/`）。换机器复现时，
  把数据放到该路径，或自行修改上游脚本的 `RAW` 常量。

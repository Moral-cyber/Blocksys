#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""e1_alpha_training.py — 生成 E1 异构扫描所需的 18 次训练（重，数小时）。

对应论文
--------
Fig. 2(a) 的训练部分。跑完之后再用 f2a_alpha_sweep.py 聚合出图。

为什么单独放
------------
f2a 只做聚合，几秒钟；真正耗时的是这里的训练：
  3 个 α（0.1 / 0.5 / 1.0）× 2 个任务 × 3 个种子 = 18 次；
  每次 20 轮联邦训练 + 逐客户端零中心/先验中心拟合。

输入
----
CIFAR-10 / FashionMNIST 原始数据（默认在 paper/02_实验工程/dataset/ 下）
paper/03_英文论文/revision_artifacts/route_a_20261105/baselines/adapter.py

输出
----
paper/03_英文论文/revision_artifacts/v4_dualtrack/e1/<task>/<alpha>/<seed>/ 下的训练与评分结果

用法
----
    python e1_alpha_training.py --stage preflight          # 检查环境与输入
    python e1_alpha_training.py --stage matrix             # 按矩阵逐个跑（推荐）
    python e1_alpha_training.py --stage run --task cifar --alpha 0.1 --seed 20261231

环境
----
需要 PyTorch + numpy（见 requirements.txt），单次运行占用 CPU/GPU 视机器而定。
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[1]
UPSTREAM_REL = "paper/03_英文论文/revision_artifacts/v4_dualtrack/e1_run.py"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the E1 alpha-sweep training matrix")
    ap.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    # 透传给上游脚本的参数
    for flag in ("--stage", "--task", "--alpha", "--seed"):
        ap.add_argument(flag, required=False)
    args, unknown = ap.parse_known_args()

    upstream = args.repo_root / UPSTREAM_REL
    if not upstream.exists():
        raise SystemExit(f"upstream script not found: {upstream}")

    forwarded = [str(upstream)]
    for flag in ("--stage", "--task", "--alpha", "--seed"):
        value = getattr(args, flag[2:].replace("-", "_"), None)
        if value is not None:
            forwarded += [flag, str(value)]
    forwarded += unknown

    sys.argv = forwarded
    runpy.run_path(str(upstream), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

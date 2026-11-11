#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""f2a_alpha_sweep.py — 复现论文 Fig. 2(a)：异构强度扫描（E1）。

对应论文
--------
Fig. 2(a)、§5.2。“α 从 1.0 到 0.1，平均收益从 0.004 升到 0.041”。

输入
----
paper/03_英文论文/revision_artifacts/v4_dualtrack/e1/ 下的 18 次开发实验输出
（3 个 α × 2 个任务 × 3 个种子）。若该目录不存在，说明还没跑训练，
需要先用 e1_alpha_training.py 生成（耗时较长）。

输出
----
results/f2a_alpha_sweep.json   每个 α 的收益均值与种子值
results/f2a_alpha_sweep.pdf    与论文 Fig. 2(a) 同源的图

做法
----
调用上游 e1_aggregate.py（不复制逻辑），跑完后把产物复制到本目录 results/。

用法
----
    python f2a_alpha_sweep.py
"""

from __future__ import annotations

import argparse
import runpy
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[1]
UPSTREAM_REL = "paper/03_英文论文/revision_artifacts/v4_dualtrack/e1_aggregate.py"


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproduce Fig. 2(a) heterogeneity sweep")
    ap.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    args = ap.parse_args()

    upstream = args.repo_root / UPSTREAM_REL
    if not upstream.exists():
        raise SystemExit(f"upstream script not found: {upstream}")

    src_dir = upstream.parent
    if not (src_dir / "e1_alpha_sweep.json").exists():
        sys.argv = [str(upstream)]
        runpy.run_path(str(upstream), run_name="__main__")
    else:
        # 上游聚合脚本会校验冻结的 PLAN.md 哈希；已生成过就直接复用，不改动 paper/。
        print("[reuse] existing e1_alpha_sweep.json")

    HERE.joinpath("results").mkdir(exist_ok=True)
    for name, dst_name in (("e1_alpha_sweep.json", "f2a_alpha_sweep.json"),
                           ("figures/alpha_vs_gain.pdf", "f2a_alpha_sweep.pdf")):
        src = src_dir / name
        if src.exists():
            shutil.copyfile(src, HERE / "results" / dst_name)
            print(f"copied: results/{dst_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

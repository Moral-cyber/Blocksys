#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""t1_main_table.py — 复现论文 Table 1 的主结果行（PR-LFL vs 零中心）。

对应论文
--------
Table 1 的第 1–5 行（Shared / Shared+prior / Zero vector / Prior BCTS / Prior vector），
以及 §5.1 的 0.6446/0.1880、0.0358/0.0182、60/60 配对等数字。

输入
----
paper/03_英文论文/evidence/individual_seed_metrics.csv
  三个种子 × 两个任务 × 各变体的逐客户端指标（原始 TEST 评分结果汇总）。

输出
----
results/t1_main_table.json
  * summary_by_variant：每个变体的均值
  * paired_nll：prior − zero 的 60 个配对、均值/中位数、bootstrap 区间
results/t1_main_table.md：可直接对照论文表格的文字摘要

做法
----
调用 paper/…/revision_artifacts/route_a_20261105/stats/analyze_p0.py 的同一套计算
（不复制逻辑，避免两份代码漂移），跑完后把输出搬到本目录 results/。

用法
----
    python t1_main_table.py
    python t1_main_table.py --repo-root /path/to/Codex
"""

from __future__ import annotations

import argparse
import runpy
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[1]
UPSTREAM_REL = "paper/03_英文论文/revision_artifacts/route_a_20261105/stats/analyze_p0.py"
UPSTREAM_OUT = Path("/private/tmp/pr_lfl_route_a/stats")


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproduce Table 1 main rows")
    ap.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    args = ap.parse_args()

    upstream = args.repo_root / UPSTREAM_REL
    if not upstream.exists():
        raise SystemExit(f"upstream script not found: {upstream}")

    # 上游脚本把结果写到固定的临时目录，先确保存在（该目录被系统清理过也不影响）。
    UPSTREAM_OUT.mkdir(parents=True, exist_ok=True)

    # 上游脚本按 __file__ 解析相对路径，因此必须以它的真实路径执行。
    sys.argv = [str(upstream)]
    runpy.run_path(str(upstream), run_name="__main__")

    HERE.joinpath("results").mkdir(exist_ok=True)
    copied = []
    for name in ("result.json", "prose.md"):
        src = UPSTREAM_OUT / name
        if src.exists():
            dst = HERE / "results" / ("t1_main_table.md" if name == "prose.md" else "t1_main_table.json")
            shutil.copyfile(src, dst)
            copied.append(str(dst))
    print(f"copied: {copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""f2b_client_size.py — 复现论文 Fig. 2(b)：收益与客户端样本量的相关性（E2）。

对应论文
--------
Fig. 2(b)、§5.3。“Spearman ρ = −0.771 / −0.729”。

输入
----
paper/03_英文论文/revision_artifacts/route_a_20261105/stats/result.json
  （逐客户端 prior/zero 的 NLL 均值，无需重新训练）
paper/03_英文论文/evidence/individual_seed_metrics.csv
  （逐种子指标，用于复算一致性检查）

输出
----
results/f2b_client_size.json   相关系数、每客户端数据点
results/f2b_client_size.pdf    与论文 Fig. 2(b) 同源的图

做法
----
调用上游 e2_analyze.py（不复制逻辑），跑完后复制产物。

用法
----
    python f2b_client_size.py
"""

from __future__ import annotations

import argparse
import runpy
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[1]
UPSTREAM_REL = "paper/03_英文论文/revision_artifacts/v4_dualtrack/e2_analyze.py"


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproduce Fig. 2(b) client-size correlation")
    ap.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    args = ap.parse_args()

    upstream = args.repo_root / UPSTREAM_REL
    if not upstream.exists():
        raise SystemExit(f"upstream script not found: {upstream}")

    sys.argv = [str(upstream)]
    runpy.run_path(str(upstream), run_name="__main__")

    src_dir = upstream.parent
    HERE.joinpath("results").mkdir(exist_ok=True)
    for name, dst_name in (("e2_skew_gain.json", "f2b_client_size.json"),
                           ("figures/skew_vs_gain.pdf", "f2b_client_size.pdf")):
        src = src_dir / name
        if src.exists():
            shutil.copyfile(src, HERE / "results" / dst_name)
            print(f"copied: results/{dst_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

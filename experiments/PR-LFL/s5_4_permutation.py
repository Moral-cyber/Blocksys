#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""s5_4_permutation.py — 复现论文 §5.4：类别置换对照（先验的“类别对齐”是否真的重要）。

对应论文
--------
§5.4，“Permuting class labels in the prior raises development NLL by 0.071/0.044”。

思路
----
把先验向量的 10 个分量整体循环移位（数值和范数完全不变，只是类别对错了），
如果 NLL 明显变差，说明收益来自“类别对得上”，而不是“多了一点正则”。

输入
----
paper/05_核心方法研发/results/predictive_pfl_confirm/ 下已保存的开发集 logits
（不重新训练）

输出
----
results/s5_4_permutation.json

做法
----
调用上游 core_contribution_probe.py，跑完后复制产物。

用法
----
    python s5_4_permutation.py
"""

from __future__ import annotations

import argparse
import runpy
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[1]
UPSTREAM_REL = "paper/05_核心方法研发/scripts/core_contribution_probe.py"
UPSTREAM_OUT_REL = "paper/05_核心方法研发/results/core_contribution_probe/result.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproduce §5.4 class-permutation control")
    ap.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    args = ap.parse_args()

    upstream = args.repo_root / UPSTREAM_REL
    if not upstream.exists():
        raise SystemExit(f"upstream script not found: {upstream}")

    src = args.repo_root / UPSTREAM_OUT_REL
    if src.exists():
        # 上游脚本拒绝覆盖已完成的诊断结果，这里直接复用，避免改动 paper/ 下的文件。
        print(f"[reuse] existing diagnostic: {src}")
    else:
        sys.argv = [str(upstream)]
        runpy.run_path(str(upstream), run_name="__main__")
    HERE.joinpath("results").mkdir(exist_ok=True)
    if src.exists():
        shutil.copyfile(src, HERE / "results/s5_4_permutation.json")
        print("copied: results/s5_4_permutation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

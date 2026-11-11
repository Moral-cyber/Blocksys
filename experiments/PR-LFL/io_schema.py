#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""io_schema.py — 统一的「参数导入 / 结果导出」约定。

为什么要有它
------------
每个实验脚本都应该是 `输入配置 → 计算 → 输出结果`，否则别人复现时不知道该改哪里、
结果里也没法追溯是哪份输入算出来的。本文件只做三件事：

  1. load_config(path)：读 JSON 配置，套默认值，校验未知字段；
  2. save_result(path, obj, inputs=[...])：写 JSON 结果，自动附带时间戳、
     脚本哈希、输入文件哈希；
  3. repo_root()：从脚本位置推回仓库根目录，避免写死绝对路径。

用法（在任意实验脚本里）
------------------------
    from io_schema import load_config, save_result

    cfg = load_config("configs/t1_main_table.json", defaults={"repo_root": ".", "out": "results/t1_main_table.json"})
    ...
    save_result(cfg["out"], payload, inputs=[cfg["csv_path"]])

结果文件里固定会多出三个键
--------------------------
    "_meta": {created_at, script, script_sha256, inputs: {path: sha256}, host}
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def repo_root() -> Path:
    """experiments/<line>/<script>.py → 仓库根（含 paper/ 与 experiments/ 的那一层）。"""
    return Path(__file__).resolve().parents[2]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: str | Path, defaults: dict | None = None) -> dict:
    """读配置并套默认值；出现未知字段直接报错，防止笔误被静默忽略。"""
    defaults = defaults or {}
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    unknown = set(raw) - set(defaults)
    if unknown:
        raise ValueError(f"unknown config keys in {p}: {sorted(unknown)}")
    cfg = {**defaults, **raw}
    cfg.setdefault("_config_path", str(p))
    return cfg


def save_result(path: str | Path, payload: dict, inputs: list[str | Path] | None = None) -> Path:
    """写结果 JSON，并附带可追溯的 _meta 块。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    script = Path(sys.argv[0]).resolve()
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": str(script),
        "script_sha256": sha256_of(script) if script.exists() else None,
        "inputs": {str(i): sha256_of(Path(i)) for i in (inputs or []) if Path(i).exists()},
        "python": platform.python_version(),
    }
    out.write_text(json.dumps({**payload, "_meta": meta}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    # 自检：读一个不存在的配置也能给出默认值。
    print(json.dumps(load_config("__none__", {"repo_root": ".", "out": "results/x.json"}), ensure_ascii=False, indent=2))

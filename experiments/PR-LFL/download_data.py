#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""download_data.py — 下载 PR-LFL 实验所需的数据集。

用途
----
仓库不存数据集（CIFAR-10 约 170 MB，FashionMNIST 约 30 MB）。本脚本把它们下到
`experiments/PR-LFL/data/` 下，并打印每个文件的 SHA-256 供你记录。

注意
----
论文里的实验使用的是 `paper/02_实验工程/dataset/` 下已经准备好的数据；
本脚本只用于「换一台机器复现」的场景。跑完请把 `--data-root` 指到新目录，
或把数据复制到 `paper/02_实验工程/dataset/` 以匹配上游脚本的默认路径。

用法
----
    python download_data.py                  # 两个数据集都下
    python download_data.py --only cifar10
    python download_data.py --dest /path/to/data
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_ARCHIVE = "cifar-10-python.tar.gz"
FASHION_BASE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FASHION_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already exists")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {url}")
    with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as fh:
        fh.write(resp.read())
    print(f"[ok  ] {dest.name}  {dest.stat().st_size} B  sha256={sha256(dest)[:16]}…")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description="Download CIFAR-10 / FashionMNIST")
    ap.add_argument("--dest", type=Path, default=HERE / "data")
    ap.add_argument("--only", choices=("cifar10", "fashion", "both"), default="both")
    args = ap.parse_args()

    if args.only in ("cifar10", "both"):
        archive = fetch(CIFAR_URL, args.dest / "cifar10" / CIFAR_ARCHIVE)
        target = args.dest / "cifar10"
        if not (target / "cifar-10-batches-py").exists():
            print(f"[ext ] {archive.name}")
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(target)
        print(f"[dir ] {target/'cifar-10-batches-py'}")

    if args.only in ("fashion", "both"):
        target = args.dest / "fashion" / "raw"
        for name in FASHION_FILES:
            fetch(FASHION_BASE + name, target / name)
        print(f"[dir ] {target}")

    print("\n把 --data-root 指向上面的目录，或复制到 paper/02_实验工程/dataset/ 以匹配上游默认路径。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

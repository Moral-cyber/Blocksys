#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x1_export_uploads.py — 把一次链上集成实验的原始记录，归一化成可交给外部链的 uploads.jsonl。

用途
----
本脚本是「链上实验」与外部合作方之间的数据出口。它不训练、不连链，
只读取一次集成实验留下的原始 JSONL（每行一次真实上传 + 一次链上登记），
把字段整理成稳定、可被对方合约直接消费的格式。

输入
----
默认读取 data/raw/e3_receipts.jsonl —— 由论文 E3 实验产生，19 条记录：
  * c2_r2_measured：2 客户端 × 2 轮 = 4 条
  * c5_r3_measured：5 客户端 × 3 轮 = 15 条
每条记录包含：8 个绑定字段、交易哈希/区块/gas、纳入回执、字节数。

输出
----
data/uploads.jsonl，一行一条上传，字段见文件末尾 SCHEMA 注释。

对应论文
--------
Table 3（上传字段与链上/链下划分）、Table 4（字节与 gas）、§7.1–7.3。

重要约定（对方实现时必须遵守）
------------------------------
合约里的 record 键 = keccak256(utf8(record_id))，由调用方（Node/ethers 的 id()）计算；
其余 *_hash 字段已经是 32 字节摘要，直接写入即可，不要再哈希一次。
digest = SM3(wire)，是不需要密钥就能复算的公开摘要。

用法
----
    python x1_export_uploads.py                     # 默认输入输出
    python x1_export_uploads.py --input data/raw/e3_receipts.jsonl --out data/uploads.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "data/raw/e3_receipts.jsonl"
DEFAULT_OUT = HERE / "data/uploads.jsonl"
SCHEMA = "pr-lfl-upload/1"

# 只保留与链上登记、审计直接相关的字段；其余（时延等）留在原始文件里。
BINDING_FIELDS = (
    "activity_hash",
    "client_hash",
    "round_id",
    "nonce_hash",
    "base_model_hash",
    "layout_hash",
    "codec_spec_sha256",
    "digest",
)
CHAIN_FIELDS = (
    "tx_hash",
    "block_number",
    "block_hash",
    "chain_id",
    "contract_address",
    "uploader",
    "status",
    "confirmations",
    "gas_used",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(row: dict) -> dict:
    reg = row["registration"]
    receipt = row.get("incorporation_receipt") or {}
    return {
        "schema": SCHEMA,
        "configuration": row["configuration"],
        "round": row["round"],
        "client": row["client"],
        "record_id": reg["record_id"],
        "record_key_rule": "keccak256(utf8(record_id))",
        "binding": {k: reg[k] for k in BINDING_FIELDS},
        "chain": {k: reg.get(k) for k in CHAIN_FIELDS},
        "incorporation_receipt": {
            k: receipt.get(k)
            for k in ("state", "digest", "model_digest", "session", "challenge")
            if k in receipt
        },
        "bytes": {
            "payload": row.get("payload_bytes"),
            "envelope_and_tag": row.get("application_envelope_and_tag_bytes"),
        },
        "verification_ms": row.get("verification_ms"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize chain-integration receipts into uploads.jsonl")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--indent", action="store_true", help="write pretty JSON per line (for inspection)")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")

    rows = [normalize(json.loads(line)) for line in args.input.read_text().splitlines() if line.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, indent=2 if args.indent else None) + "\n")

    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.input),
        "source_sha256": sha256(args.input),
        "output_sha256": sha256(args.out),
        "records": len(rows),
        "configurations": sorted({r["configuration"] for r in rows}),
        "script_sha256": sha256(Path(__file__)),
        "notes": [
            "record 键由调用方按 keccak256(utf8(record_id)) 计算",
            "binding 中的 32 字节摘要直接上链，不要二次哈希",
            "digest = SM3(wire)，公开可复算",
        ],
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/exports_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in ("records", "configurations", "output_sha256")}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# SCHEMA（一行一条）
# {
#   "schema": "pr-lfl-upload/1",
#   "configuration": "c2_r2_measured" | "c5_r3_measured",
#   "round": int, "client": int,
#   "record_id": "r1-c0",
#   "record_key_rule": "keccak256(utf8(record_id))",
#   "binding": { activity_hash, client_hash, round_id, nonce_hash,
#                base_model_hash, layout_hash, codec_spec_sha256, digest },
#   "chain":   { tx_hash, block_number, block_hash, chain_id, contract_address,
#                uploader, status, confirmations, gas_used },
#   "incorporation_receipt": { ... },
#   "bytes": { "payload": 54067, "envelope_and_tag": 54408 },
#   "verification_ms": float
# }
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())

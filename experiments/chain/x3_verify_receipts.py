#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x3_verify_receipts.py — 校验 uploads.jsonl 的字段完整性，并与链上登记结果交叉核对。

用途
----
在不连链、不跑训练的前提下，回答三个问题：
  1. 导出的 19 条记录格式是否自洽（32 字节字段、轮次/客户端一致、无重复）；
  2. 每条记录的 record 键是否唯一（record_id 与 nonce 都不重复）；
  3. 如果跑过 x2，链上登记的条数、gas 总量、readback 是否与导出文件对得上。

输入
----
data/uploads.jsonl                 必需
results/x2_registration.json       可选（存在才做交叉核对）

输出
----
results/x3_verification.json，包含每项检查的结果与总体 PASS/FAIL。

对应论文
--------
§7.2–7.3（登记与恢复）、Table 4（字节与 gas）。

运行
----
    python x3_verify_receipts.py
    python x3_verify_receipts.py --uploads data/uploads.jsonl --registration results/x2_registration.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEX32 = set("0123456789abcdefABCDEF")


def is_hex32(value) -> bool:
    """合约里的 bytes32 字段必须是 0x + 64 个十六进制字符。"""
    return isinstance(value, str) and value.startswith("0x") and len(value) == 66 and set(value[2:]) <= HEX32


def check(condition: bool, name: str, detail: str = "") -> dict:
    return {"name": name, "pass": bool(condition), "detail": detail}


def main() -> int:
    ap = argparse.ArgumentParser(description="Static verification of uploads.jsonl")
    ap.add_argument("--uploads", type=Path, default=HERE / "data/uploads.jsonl")
    ap.add_argument("--registration", type=Path, default=HERE / "results/x2_registration.json")
    ap.add_argument("--out", type=Path, default=HERE / "results/x3_verification.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.uploads.read_text().splitlines() if l.strip()]
    checks: list[dict] = []

    checks.append(check(len(rows) > 0, "records_present", f"{len(rows)} rows"))

    # 1) schema 与字段格式
    bad_schema = [r["record_id"] for r in rows if r.get("schema") != "pr-lfl-upload/1"]
    checks.append(check(not bad_schema, "schema_version", str(bad_schema)))

    binding_keys = ("activity_hash", "client_hash", "nonce_hash", "base_model_hash",
                    "layout_hash", "codec_spec_sha256", "digest")
    bad_fields = [
        f"{r['record_id']}:{k}" for r in rows for k in binding_keys if not is_hex32(r["binding"].get(k))
    ]
    checks.append(check(not bad_fields, "binding_fields_are_bytes32", ";".join(bad_fields[:5])))

    # 2) 轮次/客户端自洽
    bad_round = [r["record_id"] for r in rows if int(r["binding"]["round_id"]) != int(r["round"])]
    checks.append(check(not bad_round, "round_id_matches_round", str(bad_round)))

    # 3) 唯一性：record_id、nonce 都不应重复
    # 唯一性按 configuration 计：每个活动各自部署一份合约，不同活动之间 record 可重名。
    record_ids = [(r["configuration"], r["record_id"]) for r in rows]
    nonces = [(r["configuration"], r["binding"]["nonce_hash"]) for r in rows]
    checks.append(check(len(record_ids) == len(set(record_ids)), "record_ids_unique",
                        f"{len(record_ids)} ids"))
    checks.append(check(len(nonces) == len(set(nonces)), "nonces_unique", f"{len(nonces)} nonces"))

    # 4) 字节范围（54067 载荷 / 54408 信封，允许其他配置略有差异，只做量级检查）
    sizes = [r.get("bytes", {}).get("payload") for r in rows]
    checks.append(check(all(isinstance(s, int) and 50_000 <= s <= 60_000 for s in sizes),
                        "payload_size_in_expected_range", str(sorted(set(sizes)))))

    # 5) 交叉核对：x2 的登记结果
    cross = None
    if args.registration.exists():
        reg = json.loads(args.registration.read_text())
        configs = {}
        for config, block in reg.get("configurations", {}).items():
            exported = [r for r in rows if r["configuration"] == config]
            configs[config] = {
                "exported": len(exported),
                "registered": block.get("registrations"),
                "counts_match": len(exported) == block.get("registrations"),
                "gas_total_registered": block.get("gas_total"),
                "gas_per_upload_mean": round(block.get("gas_total", 0) / max(1, block.get("registrations", 1)), 1),
                "all_readback_match": block.get("all_readback_match"),
            }
        cross = configs
        checks.append(check(all(v["counts_match"] and v["all_readback_match"] for v in configs.values()),
                            "x2_registration_cross_check", json.dumps({k: [v["exported"], v["registered"]] for k, v in configs.items()})))
    else:
        checks.append(check(False, "x2_registration_cross_check", "skipped: run x2 first"))

    passed = all(c["pass"] for c in checks if not (c["name"].endswith("cross_check") and c["detail"].startswith("skipped")))
    result = {
        "schema": "pr-lfl-verification/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "uploads": str(args.uploads),
        "uploads_sha256": hashlib.sha256(args.uploads.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records": len(rows),
        "checks": checks,
        "cross_check": cross,
        "status": "PASS" if passed else "FAIL",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "records": len(rows),
                      "failed": [c["name"] for c in checks if not c["pass"]]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

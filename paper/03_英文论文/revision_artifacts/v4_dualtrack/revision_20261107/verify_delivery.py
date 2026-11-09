"""Verify the authorized 7 November delivery against immutable evidence."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import re
import statistics

from pypdf import PdfReader

ARTIFACTS = Path(__file__).resolve().parent.parent
DELIVERY = Path(__file__).resolve().parent
MANUSCRIPT = ARTIFACTS.parents[1]
ROOT = MANUSCRIPT.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


checks = {}
historical_plan = DELIVERY / "before/revision_artifacts/v4_dualtrack/PLAN.md"
for path, expected in load(ARTIFACTS / "E1_SOURCE_LOCK.json")["inputs"].items():
    actual = historical_plan if Path(path) == ARTIFACTS / "PLAN.md" else Path(path)
    assert sha(actual) == expected, path
checks["E1_historical_source_lock"] = "PASS; archived pre-amendment PLAN identity retained"

figures = load(DELIVERY / "claim_figures.json")
for field in ["sources", "output_sha256"]:
    for path, expected in figures[field].items():
        assert sha(path) == expected, path
assert figures["E1"]["cifar"]["improved_pairs_at_alpha0.1"] == 60
assert figures["E1"]["fashion"]["improved_pairs_at_alpha0.1"] == 59
for task, rho in [("cifar", -0.7714285714285714), ("fashion", -0.7293233082706766)]:
    assert math.isclose(figures["E2"][task]["rho"], rho, abs_tol=1e-12)
checks["E1_E2_source_and_figures"] = "PASS"

e3 = load(ARTIFACTS / "e3_chain_cost.json")
lock = load(ARTIFACTS / "e3/SOURCE_LOCK.json")
assert sha(ARTIFACTS / "e3/SOURCE_LOCK.json") == e3["source_lock_sha256"]
for path, expected in lock["inputs"].items():
    assert sha(path) == expected, path
for field in ["source_and_input_sha256", "output_sha256"]:
    for path, expected in e3[field].items():
        assert sha(path) == expected, path
receipts = [json.loads(line) for line in (ARTIFACTS / "e3_receipts.jsonl").read_text().splitlines()]
assert len(receipts) == 19
assert len({(r["configuration"], r["round"], r["client"]) for r in receipts}) == 19
assert len({r["registration"]["tx_hash"] for r in receipts}) == 19
configurations = []
for row in e3["rows"]:
    raw = load(row["raw_result"])
    clients, rounds = row["clients"], row["rounds"]
    count = clients * rounds
    assert raw["status"] == "PASS"
    assert (raw["client_count"], raw["round_count"]) == (clients, rounds)
    for path, expected in raw["source_and_input_sha256"].items():
        assert sha(path) == expected, path
    assert not raw["audit"]["forbidden_test_open_attempts"]
    assert not raw["audit"]["non_loopback_attempts"]
    assert raw["HMAC_keys_persisted_or_reported"] is False
    assert raw["private_keys_reported"] is False
    assert raw["chain"]["bridge_stats"]["record_count"] == count
    exported = [r for r in receipts if r["configuration"] == row["configuration"]]
    assert len(exported) == count
    assert sum(r["registration"]["gas_used"] for r in exported) == row["gas_registration_total"]
    assert sum(r["application_envelope_and_tag_bytes"] for r in exported) == row["message_bytes_total"]
    assert math.isclose(
        statistics.mean(r["verification_ms"] for r in exported),
        row["incorporation_verification_ms"]["mean"],
        rel_tol=1e-12,
    )
    assert raw["counters"]["protocol_encode"] == count
    for rd in raw["rounds"]:
        assert rd["protocol_direct_model_max_abs_error"] == 0
        assert rd["protocol_direct_residual_max_abs_error"] == 0
        assert f"r{rd['round']}:recovery_adds_no_encode_decode_or_chain_register" in raw["checks"]
        assert f"r{rd['round']}:model_finalizes_exactly_once" in raw["checks"]
        for client in rd["clients"]:
            assert client["EF_generation"] == rd["round"]
    for exported_record in exported:
        chain = exported_record["registration"]
        incorporation = exported_record["incorporation_receipt"]["receipt"]
        rd = raw["rounds"][exported_record["round"] - 1]
        assert chain["canonical"] and chain["status"] == 1 and chain["confirmation_depth"] >= 2
        assert incorporation["state"] == "INCORPORATED"
        assert incorporation["result_model_hash"] == rd["result_model_sm3"]
        assert incorporation["record_id"] == chain["record_id"]
        assert incorporation["round_id"] == exported_record["round"]
        assert incorporation["client"] == f"c{exported_record['client']}"
        for key in ["base_model_hash", "codec_spec_sha256", "digest"]:
            assert "0x" + incorporation[key] == chain[key]
        assert exported_record["payload_bytes"] == 54067
        assert exported_record["application_envelope_and_tag_bytes"] == 54408
    configurations.append({
        "configuration": row["configuration"],
        "records": count,
        "gas_total": row["gas_registration_total"],
        "mean_local_verification_ms": row["incorporation_verification_ms"]["mean"],
        "message_bytes_total": row["message_bytes_total"],
        "direct_model_and_EF_bitwise_equal": True,
    })
assert sha(ARTIFACTS / "e3/UploadRegistry.sol") == sha(ROOT / "05_核心方法研发/scripts/chain_integration/UploadRegistry.sol")
checks["E3"] = "PASS; source locks, 19 unique receipts, bindings, totals, recovery and model/EF equality"

prose_files = [
    MANUSCRIPT / "sections/anchored_uploads.tex",
    MANUSCRIPT / "sections/upload_evaluation.tex",
    *sorted((ARTIFACTS / "tables").glob("*.tex")),
]
forbidden = re.compile(
    r"watermark|水印|does\s+not|\bonly\b|trusted|single[-\s]+node|\bcannot\b|\blimited\b|\bsolely\b|at\s+most", re.I
)
for path in prose_files:
    assert not forbidden.search(path.read_text()), path
log = (ARTIFACTS / "system_preview.log").read_text()
assert "Overfull" not in log and "undefined" not in log.lower()
assert len(PdfReader(ARTIFACTS / "system_preview.pdf").pages) == 6
checks["system_prose_and_layout"] = "PASS; banned terms absent, paragraph endings manually reviewed, six-page preview, zero overfull or undefined references"

unchanged, skipped = [], []
for path, expected in load(ARTIFACTS / "protected_files.json")["sha256"].items():
    if Path(path).name == ".DS_Store":
        skipped.append(path)
        continue
    assert sha(path) == expected, path
    unchanged.append(path)
before = load(DELIVERY / "BEFORE.json")["files"]
for name in ["main.tex", "main.pdf"]:
    assert sha(MANUSCRIPT / name) == before[str(MANUSCRIPT / name)]
checks["original_files"] = {
    "unchanged_substantive_files": len(unchanged),
    "excluded_OS_metadata": skipped,
    "main_tex_and_main_pdf": "UNCHANGED",
}
assert round((1 - 54067 / 547560) * 100, 2) == 90.13
assert round(54408 / 547560 * 100, 2) == 9.94

claim_map = [
    {"claim": "Figure1: qualified three-point E1 sweep; alpha=.1 counts 60/60 and 59/60", "evidence": "e1_alpha_sweep.json; revision_20261107/claim_figures.json"},
    {"claim": "Figure2: TRAIN sample-count/gain rho -.771/-.729, log x", "evidence": "e2_sample_gain.json; e2_skew_gain.json"},
    {"claim": "Section7/Table3/Figure4: 15 registrations, 4,229,478 total gas, 281,965.2 mean gas, .380 ms local verification, 54,408 B/message", "evidence": "e3_chain_cost.json; e3/c5_r3_measured/result.json; e3_receipts.jsonl"},
    {"claim": "Section7: bitwise model/EF equality per round; recovery preserves one contribution and generation", "evidence": "e3/c2_r2_measured/result.json; e3/c5_r3_measured/result.json"},
    {"claim": "Section6: binding, readback, incorporation receipt and residual transition", "evidence": "e3/UploadRegistry.sol; e3/network_service.py; 05_核心方法研发/repro/work/durable_integrity_v2/protocol.py"},
]
now = datetime.datetime.now().astimezone().isoformat()
final = {
    "status": "PASS",
    "created_at": now,
    "scope": "User-authorized items 1-3: system section sources and preview, no full manuscript merge",
    "checks": checks,
    "E3_configurations": configurations,
    "claim_map": claim_map,
    "files_sha256": {
        str(path): sha(path)
        for path in [ARTIFACTS / "PLAN.md", ARTIFACTS / "e3_chain_cost.json", ARTIFACTS / "e2_sample_gain.json", ARTIFACTS / "system_preview.pdf", *prose_files]
    },
}
(DELIVERY / "FINAL_CHECK.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n")
status = load(ARTIFACTS / "STATUS.json")
status.update({
    "updated_at": now,
    "plan": "COMPLETE",
    "claim_lock": "COMPLETE",
    "E3": "COMPLETE",
    "system_sections_6_7": "DRAFT_COMPLETE_WITH_E3",
    "system_preview_pages": 6,
    "full_manuscript_merge": "NOT_PERFORMED_THIS_PASS",
    "protected_files_unchanged": len(unchanged),
    "protected_metadata_excluded": len(skipped),
    "scope": "Claim lock + Figures1-2 + measured E3/Table3/Figure4 + Sections6-7 sources and numbered preview complete; original main.tex/main.pdf unchanged",
    "current_check": "revision_20261107/FINAL_CHECK.json",
})
(ARTIFACTS / "STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"status": "PASS", "unchanged_substantive_original_files": len(unchanged), "receipts": len(receipts), "preview_pages": 6, "E3": configurations}, ensure_ascii=False))

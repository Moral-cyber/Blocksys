"""Deterministic integrity-layer attack matrix for sparse update packets.

This is a local protocol test, not a blockchain benchmark.  It mirrors the
frozen wire-format rules in the project notes: little-endian fields, sorted
coordinates, an HMAC-SM3 tag, round/client binding, and one-time nonce use.
The output is suitable for the paper's verification table and keeps the
compression experiment independent from the ledger implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import struct
import time
from pathlib import Path


def _u8(value: int) -> bytes:
    return struct.pack("<B", int(value))


def _u16(value: int) -> bytes:
    return struct.pack("<H", int(value))


def _u32(value: int) -> bytes:
    return struct.pack("<I", int(value))


def _u64(value: int) -> bytes:
    return struct.pack("<Q", int(value))


def canonical_encode(packet: dict) -> bytes:
    """Encode metadata, sorted sparse coordinates and float32 values.

    The values are encoded as raw little-endian float32 bytes.  Sorting is by
    coordinate and is performed together with the corresponding value, so a
    receiver cannot change the order without changing the authenticated bytes.
    """

    pairs = sorted(zip(packet["indices"], packet["values"]), key=lambda x: int(x[0]))
    out = bytearray()
    out += _u8(packet["protocol_version"])
    out += _u8(packet["dtype_code"])
    out += _u16(packet["selected_count"])
    out += _u32(packet["num_elements"])
    out += _u32(packet["round_id"])
    out += _u64(packet["activity_id"])
    out += _u64(packet["client_id"])
    out += _u64(packet["wm_id"])
    out += _u64(packet["nonce"])
    for index, value in pairs:
        out += _u32(index)
        out += struct.pack("<f", float(value))
    return bytes(out)


def tag(packet: dict, key: bytes) -> bytes:
    return hmac.new(key, canonical_encode(packet), digestmod="sm3").digest()


class Ledger:
    def __init__(self, key: bytes):
        self.key = key
        self.records: dict[int, tuple[bytes, dict]] = {}
        self.used_nonces: set[tuple[int, int, int]] = set()

    def register(self, packet: dict) -> bytes:
        value = tag(packet, self.key)
        self.records[int(packet["wm_id"])] = (value, dict(packet))
        return value

    def verify(self, packet: dict, supplied_tag: bytes, *, consume: bool = True) -> bool:
        record = self.records.get(int(packet["wm_id"]))
        if record is None:
            return False
        expected, metadata = record
        if not hmac.compare_digest(expected, supplied_tag):
            return False
        # Recompute over the bytes that actually arrived.  Comparing only the
        # ledger tag with a supplied tag would accept value/index mutations.
        if not hmac.compare_digest(tag(packet, self.key), expected):
            return False
        for field in ("round_id", "activity_id", "client_id", "wm_id", "nonce"):
            if int(packet[field]) != int(metadata[field]):
                return False
        nonce_key = (int(packet["activity_id"]), int(packet["client_id"]), int(packet["nonce"]))
        if nonce_key in self.used_nonces:
            return False
        if consume:
            self.used_nonces.add(nonce_key)
        return True


def base_packet() -> dict:
    indices = [4, 18, 73, 91, 204, 901]
    values = [0.125, -0.25, 0.5, -0.75, 1.0, -1.25]
    return {
        "protocol_version": 1,
        "dtype_code": 1,  # float32
        "selected_count": len(indices),
        "num_elements": 1024,
        "round_id": 7,
        "activity_id": 11,
        "client_id": 3,
        "wm_id": 7003,
        "nonce": 9000007,
        "indices": indices,
        "values": values,
    }


def mutate(packet: dict, case: str) -> dict:
    out = dict(packet)
    out["indices"] = list(packet["indices"])
    out["values"] = list(packet["values"])
    if case == "value_tamper":
        out["values"][2] += 0.001
    elif case == "index_tamper":
        out["indices"][2] += 1
    elif case == "round_substitution":
        out["round_id"] += 1
    elif case == "client_substitution":
        out["client_id"] += 1
    elif case == "truncation":
        out["indices"] = out["indices"][:-1]
        out["values"] = out["values"][:-1]
        out["selected_count"] -= 1
    elif case == "reordered_payload":
        out["indices"] = list(reversed(out["indices"]))
        out["values"] = list(reversed(out["values"]))
    elif case == "pair_mismatch":
        out["indices"] = list(reversed(out["indices"]))
    elif case in ("valid", "replay"):
        pass
    else:
        raise ValueError(f"unknown case: {case}")
    return out


def run(case: str, repeats: int, key: bytes) -> dict:
    ledger = Ledger(key)
    original = base_packet()
    registered_tag = ledger.register(original)
    accepted = 0
    latencies_us = []
    for repeat in range(repeats):
        if case == "replay":
            candidate = original
        else:
            # Each non-replay trial uses a fresh record so that the measured
            # rejection is caused by the mutation rather than nonce reuse.
            trial = base_packet()
            trial["wm_id"] += repeat + 1
            trial["nonce"] += repeat + 1
            trial_tag = ledger.register(trial)
            candidate = mutate(trial, case)
            registered_tag = trial_tag
        started = time.perf_counter_ns()
        result = ledger.verify(candidate, registered_tag)
        latencies_us.append((time.perf_counter_ns() - started) / 1000.0)
        accepted += int(result)
    return {
        "case": case,
        "attempts": repeats,
        "accepted": accepted,
        "rejected": repeats - accepted,
        "acceptance_rate": accepted / repeats,
        "rejection_rate": (repeats - accepted) / repeats,
        "latency_us_mean": sum(latencies_us) / len(latencies_us),
        "latency_us_p95": sorted(latencies_us)[max(0, int(0.95 * len(latencies_us)) - 1)],
        "tag_bytes": len(registered_tag),
        "canonical_bytes": len(canonical_encode(original)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output-stem", default="results/integrity_attack_matrix")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    key = bytes.fromhex("00112233445566778899aabbccddeeff" * 2)
    cases = [
        "valid", "value_tamper", "index_tamper", "round_substitution",
        "client_substitution", "truncation", "pair_mismatch", "reordered_payload", "replay",
    ]
    rows = [run(case, args.repeats, key) for case in cases]
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_stem)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output.with_suffix(".csv"))


if __name__ == "__main__":
    main()

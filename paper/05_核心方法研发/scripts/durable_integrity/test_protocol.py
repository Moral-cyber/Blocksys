"""Actual synthetic protocol tests. No training, chain, or network benchmark."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import sqlite3
import struct
import subprocess
import sys
import threading
import time
import unittest

from protocol import (ACCEPTED, INCORPORATED, REGISTERED, AuthenticationError,
                      ClientStore, Codec, CollisionError, Envelope, LocalServer,
                      Packet, ProtocolError, Receipt, StateError, synthetic_sparse_codec)


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
FROZEN_CODEC = WORKSPACE / "work/mixed_precision/codec.py"
ARTIFACT_ROOT = ROOT / "results/synthetic_cases"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = ARTIFACT_ROOT / self._testMethodName
        self.directory.mkdir(parents=True, exist_ok=False)
        self.keys = {client: secrets.token_bytes(32) for client in ("c1", "c2", "c3")}
        self.key_path = self.directory / "local_test_keys.json"
        descriptor = os.open(self.key_path, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            json.dump({client: key.hex() for client, key in self.keys.items()}, handle)
        self.codec = synthetic_sparse_codec()
        self.server_path = self.directory / "server.sqlite"
        self.server = LocalServer(self.server_path, self.keys, {self.codec.name: self.codec})
        self.base = self.server.create_activity("act", [1.0, 2.0, 3.0, 4.0], layout="layout4-v1",
                                                codec=self.codec.name, fixed_weights={"c1": 0.25, "c2": 0.75})
        self.server.open_round("act", 1)
        self.clients = {client: self.reopen_client(client) for client in ("c1", "c2")}
        self.envelopes = {client: self.clients[client].prepare(round_id=1, record_id="record-" + client,
                         nonce="nonce-" + client, base_model_hash=self.base, codec=self.codec,
                         update=[0.7, -0.2, 1.2, 0.5] if client == "c1" else [-0.4, 0.3, -0.8, -0.6])
                          for client in self.clients}
        self.evidence = {"scope": "synthetic protocol test", "sqlite_file": str(self.server_path.relative_to(ROOT))}

    def reopen_server(self):
        return LocalServer(self.server_path, self.keys, {self.codec.name: self.codec})

    def reopen_client(self, client):
        return ClientStore(self.directory / (client + ".sqlite"), activity="act", client=client,
                           key=self.keys[client], layout="layout4-v1", initial_residual=[0.0] * 4)

    def accept_all(self):
        for envelope in self.envelopes.values():
            self.server.register(envelope)
            self.server.accept(envelope)

    def assertVector(self, first, second):
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            self.assertAlmostEqual(a, b, places=12)

    def expected_model(self):
        decoded = {client: self.codec.decode(env.packet.payload) for client, env in self.envelopes.items()}
        return tuple(base + 0.25 * a + 0.75 * b for base, a, b in
                     zip([1, 2, 3, 4], decoded["c1"], decoded["c2"]))

    def test_valid_three_states_fixed_weight_aggregate_and_ef(self):
        envelope = self.envelopes["c1"]
        self.assertEqual(self.server.register(envelope).state, REGISTERED)
        self.assertEqual(self.server.accept(envelope).state, ACCEPTED)
        self.server.register(self.envelopes["c2"])
        self.server.accept(self.envelopes["c2"])
        result = self.server.finalize("act", 1)
        self.assertTrue(result["newly_finalized"])
        self.assertVector(result["model"], self.expected_model())
        self.assertEqual(self.server.query("record-c1").state, INCORPORATED)
        self.assertTrue(self.clients["c1"].commit_if_incorporated("record-c1", self.server.query))
        self.assertVector(self.clients["c1"].snapshot()["residual"], [-0.05, -0.2, -0.05, 0.5])
        self.evidence.update(states=[REGISTERED, ACCEPTED, INCORPORATED], known_weights={"c1": .25, "c2": .75},
                             model=list(result["model"]), ef_generation=1)

    def test_registered_only_cannot_commit_ef(self):
        self.server.register(self.envelopes["c1"])
        self.assertFalse(self.clients["c1"].commit_if_incorporated("record-c1", self.server.query))
        self.assertEqual(self.clients["c1"].snapshot()["generation"], 0)
        self.assertVector(self.clients["c1"].snapshot()["residual"], [0] * 4)

    def test_accepted_only_cannot_commit_ef(self):
        self.accept_all()
        self.assertFalse(self.clients["c1"].commit_if_incorporated("record-c1", self.server.query))
        self.assertEqual(self.clients["c1"].snapshot()["generation"], 0)
        self.assertEqual(self.server.snapshot("act")["finalized_count"], 0)

    def test_unknown_status_retains_exact_pending_bytes(self):
        self.assertFalse(self.clients["c1"].commit_if_incorporated("record-c1", self.server.query))
        reopened = self.reopen_client("c1")
        self.assertEqual(reopened.pending(), self.envelopes["c1"])
        self.assertEqual(reopened.snapshot()["generation"], 0)

    def test_pending_retry_never_regenerates_same_nonce_contribution(self):
        def forbidden_encoder(_):
            self.fail("encoder must not run for a pending retry")
        retry_codec = Codec(self.codec.name, forbidden_encoder, self.codec.decode)
        retry = self.reopen_client("c1").prepare(round_id=1, record_id="record-c1", nonce="nonce-c1",
                  base_model_hash=self.base, codec=retry_codec, update=[9000] * 4)
        self.assertEqual(retry, self.envelopes["c1"])
        with self.assertRaises(StateError):
            self.clients["c1"].prepare(round_id=1, record_id="new-record", nonce="nonce-c1",
                base_model_hash=self.base, codec=retry_codec, update=[9000] * 4)

    def test_unregistered_packet_cannot_be_accepted(self):
        with self.assertRaises(StateError):
            self.server.accept(self.envelopes["c1"])

    def test_missing_fixed_participant_prevents_finalize(self):
        self.server.register(self.envelopes["c1"])
        self.server.accept(self.envelopes["c1"])
        with self.assertRaises(StateError):
            self.server.finalize("act", 1)
        self.assertEqual(self.server.snapshot("act")["finalized_count"], 0)
        self.assertEqual(self.server.query("record-c1").state, ACCEPTED)

    def test_receipt_digest_mismatch_prevents_ef_commit(self):
        self.accept_all()
        self.server.finalize("act", 1)
        with self.assertRaises(AuthenticationError):
            self.clients["c1"].commit_if_incorporated("record-c1", lambda rid: Receipt(rid, INCORPORATED, "0" * 64))
        self.assertEqual(self.clients["c1"].snapshot()["generation"], 0)

    def test_receipt_record_mismatch_prevents_ef_commit(self):
        self.assertFalse(self.clients["c1"].commit_if_incorporated("record-c1",
            lambda rid: Receipt("different-record", INCORPORATED, self.envelopes["c1"].digest)))
        self.assertEqual(self.clients["c1"].snapshot()["generation"], 0)

    def test_nonce_reuse_new_record_rejected(self):
        original = self.envelopes["c1"]
        self.server.register(original)
        new = Envelope.sign(replace(original.packet, record_id="new-record"), self.keys["c1"])
        with self.assertRaises(CollisionError):
            self.server.register(new)

    def test_record_id_collision_different_validly_signed_bytes_rejected(self):
        original = self.envelopes["c1"]
        self.server.register(original)
        packet = replace(original.packet, payload=self.codec.encode([1, 2, 3, 4]), nonce="different-nonce")
        changed = Envelope.sign(packet, self.keys["c1"])
        with self.assertRaises(CollisionError):
            self.server.register(changed)
        with self.assertRaises(CollisionError):
            self.server.accept(changed)
        self.assertEqual(self.server.query("record-c1").digest, original.digest)

    def test_second_accepted_contribution_same_client_round_rejected(self):
        original = self.envelopes["c1"]
        self.server.register(original)
        self.server.accept(original)
        changed = Envelope.sign(replace(original.packet, record_id="extra", nonce="extra"), self.keys["c1"])
        self.server.register(changed)
        with self.assertRaises(CollisionError):
            self.server.accept(changed)

    def test_same_bytes_replay_is_idempotent_after_incorporation(self):
        self.accept_all()
        initial = self.server.finalize("act", 1)
        for _ in range(8):
            for envelope in self.envelopes.values():
                self.assertEqual(self.server.register(envelope).state, INCORPORATED)
                self.assertEqual(self.server.accept(envelope).state, INCORPORATED)
            self.assertFalse(self.server.finalize("act", 1)["newly_finalized"])
        self.assertEqual(self.server.snapshot("act")["finalized_count"], 1)
        self.assertEqual(self.server.snapshot("act")["model"], initial["model"])

    def test_closed_round_replay_with_new_nonce_rejected(self):
        self.accept_all()
        self.server.finalize("act", 1)
        changed = Envelope.sign(replace(self.envelopes["c1"].packet, record_id="late", nonce="late"), self.keys["c1"])
        self.server.register(changed)
        with self.assertRaises(StateError):
            self.server.accept(changed)

    def test_invalid_tag_registration_rejected(self):
        envelope = self.envelopes["c1"]
        with self.assertRaises(AuthenticationError):
            self.server.register(Envelope(envelope.wire, bytes(32)))
        self.assertEqual(self.server.query("record-c1").state, "UNKNOWN")

    def test_malformed_envelope_truncation_and_trailing_bytes_rejected(self):
        envelope = self.envelopes["c1"]
        for wire in [envelope.wire[:-1], envelope.wire + b"x", b"garbage"]:
            with self.assertRaises(ProtocolError):
                self.server.register(Envelope(wire, envelope.tag))

    def test_exception_rolls_back_model_and_incorporation_together(self):
        self.accept_all()
        before = self.server.snapshot("act")
        def crash():
            raise RuntimeError("synthetic transaction interruption")
        with self.assertRaises(RuntimeError):
            self.server.finalize("act", 1, after_model_write=crash)
        reopened = self.reopen_server()
        self.assertEqual(reopened.snapshot("act"), before)
        self.assertEqual(reopened.query("record-c1").state, ACCEPTED)
        self.assertTrue(reopened.finalize("act", 1)["newly_finalized"])

    def test_process_crash_before_sql_commit_recovers_atomic_rollback(self):
        self.accept_all()
        before = self.server.snapshot("act")
        program = ("import json,os,sys; from pathlib import Path; "
                   "from protocol import LocalServer,synthetic_sparse_codec; "
                   "keys={k:bytes.fromhex(v) for k,v in json.loads(Path(sys.argv[2]).read_text()).items()}; "
                   "codec=synthetic_sparse_codec(); "
                   "server=LocalServer(sys.argv[1],keys,{codec.name:codec}); "
                   "server.finalize('act',1,after_model_write=lambda:os._exit(73))")
        process = subprocess.run([sys.executable, "-c", program, str(self.server_path), str(self.key_path)],
                                 cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 73, process.stderr)
        reopened = self.reopen_server()
        self.assertEqual(reopened.snapshot("act"), before)
        self.assertEqual(reopened.query("record-c1").state, ACCEPTED)
        self.assertEqual(reopened.query("record-c2").state, ACCEPTED)
        self.assertTrue(reopened.finalize("act", 1)["newly_finalized"])
        self.evidence.update(crashed_process_exit_code=73, rollback_model_matches=True,
                             states_after_crash=[ACCEPTED, ACCEPTED], successful_retry=True)

    def test_lost_finalize_ack_and_client_restart_commit_saved_rnext_once(self):
        self.accept_all()
        self.server.finalize("act", 1)  # Deliberately discard response: lost ACK.
        self.assertEqual(self.clients["c1"].snapshot()["generation"], 0)
        recovered_client, recovered_server = self.reopen_client("c1"), self.reopen_server()
        self.assertEqual(recovered_client.pending(), self.envelopes["c1"])
        self.assertEqual(recovered_server.accept(recovered_client.pending()).state, INCORPORATED)
        self.assertTrue(recovered_client.commit_if_incorporated("record-c1", recovered_server.query))
        self.assertFalse(self.reopen_client("c1").commit_if_incorporated("record-c1", recovered_server.query))
        self.assertEqual(self.reopen_client("c1").snapshot()["generation"], 1)
        self.assertVector(self.reopen_client("c1").snapshot()["residual"], [-0.05, -0.2, -0.05, .5])
        self.assertEqual(recovered_server.snapshot("act")["finalized_count"], 1)
        self.evidence.update(lost_ack_simulated=True, reopened_server=True, reopened_client=True,
                             ef_commit_count=1, server_finalization_count=1)

    def test_sqlite_reopen_proves_all_three_states_and_pending_persistence(self):
        original = self.envelopes["c1"]
        self.server.register(original)
        self.assertEqual(self.reopen_server().query("record-c1").state, REGISTERED)
        self.assertEqual(self.reopen_client("c1").pending(), original)
        self.reopen_server().accept(original)
        self.assertEqual(self.reopen_server().query("record-c1").state, ACCEPTED)
        self.reopen_server().register(self.envelopes["c2"])
        self.reopen_server().accept(self.envelopes["c2"])
        self.reopen_server().finalize("act", 1)
        self.assertEqual(self.reopen_server().query("record-c1").state, INCORPORATED)
        self.assertTrue(self.reopen_client("c1").commit_if_incorporated("record-c1", self.reopen_server().query))
        self.assertEqual(self.reopen_client("c1").snapshot()["generation"], 1)
        self.assertGreater(self.server_path.stat().st_size, 0)

    def test_concurrent_same_bytes_accept_finalize_and_client_commit_exactly_once(self):
        barrier = threading.Barrier(16)
        def worker(number):
            client = "c1" if number % 2 == 0 else "c2"
            server = self.reopen_server()
            envelope = self.envelopes[client]
            server.register(envelope)
            server.accept(envelope)
            barrier.wait(timeout=30)
            finalized = server.finalize("act", 1)["newly_finalized"]
            committed = self.reopen_client(client).commit_if_incorporated(envelope.packet.record_id, server.query)
            return client, finalized, committed
        with ThreadPoolExecutor(max_workers=16) as executor:
            outcomes = list(executor.map(worker, range(16)))
        self.assertEqual(sum(row[1] for row in outcomes), 1)
        for client in ("c1", "c2"):
            self.assertEqual(sum(row[2] for row in outcomes if row[0] == client), 1)
            self.assertEqual(self.reopen_client(client).snapshot()["generation"], 1)
        self.assertEqual(self.reopen_server().snapshot("act")["finalized_count"], 1)
        self.assertVector(self.reopen_server().snapshot("act")["model"], self.expected_model())
        self.evidence.update(concurrent_workers=16, successful_finalizations=1,
                             successful_client_commits={"c1": 1, "c2": 1}, model_matches_known_weight_sum=True)

    def test_base_model_advance_preserves_residual_with_same_layout(self):
        self.accept_all()
        self.server.finalize("act", 1)
        for client in self.clients:
            self.clients[client].commit_if_incorporated("record-" + client, self.server.query)
        residual = self.clients["c1"].snapshot()["residual"]
        new_base = self.server.open_round("act", 2)
        self.assertNotEqual(new_base, self.base)
        envelope = self.clients["c1"].prepare(round_id=2, record_id="r2-c1", nonce="n2-c1",
                       base_model_hash=new_base, codec=self.codec, update=[0] * 4)
        self.assertEqual(envelope.packet.payload, self.codec.encode(residual))
        self.assertEqual(self.clients["c1"].snapshot()["residual"], residual)
        self.assertEqual(envelope.packet.layout, "layout4-v1")
        self.server.register(envelope)
        self.server.accept(envelope)
        self.evidence.update(base_hash_advanced=True, layout_unchanged=True, ef_was_not_reset=True)

    def test_client_reopen_wrong_layout_rejected(self):
        with self.assertRaises(StateError):
            ClientStore(self.directory / "c1.sqlite", activity="act", client="c1", key=self.keys["c1"],
                        layout="different-schema", initial_residual=[0] * 4)

    def test_reused_committed_nonce_does_not_invoke_encoder(self):
        self.accept_all()
        self.server.finalize("act", 1)
        self.clients["c1"].commit_if_incorporated("record-c1", self.server.query)
        def forbidden_encoder(_):
            self.fail("a committed nonce cannot trigger a new encoding")
        with self.assertRaises(CollisionError):
            self.clients["c1"].prepare(round_id=2, record_id="extra", nonce="nonce-c1",
                base_model_hash="new-base", codec=Codec(self.codec.name, forbidden_encoder, self.codec.decode), update=[0] * 4)

    def test_old_finalization_result_stable_after_later_round(self):
        self.accept_all()
        first = self.server.finalize("act", 1)
        for client in self.clients:
            self.clients[client].commit_if_incorporated("record-" + client, self.server.query)
        base2 = self.server.open_round("act", 2)
        for client in self.clients:
            envelope = self.clients[client].prepare(round_id=2, record_id="r2-" + client, nonce="n2-" + client,
                 base_model_hash=base2, codec=self.codec, update=[1] * 4)
            self.server.register(envelope)
            self.server.accept(envelope)
        second = self.server.finalize("act", 2)
        self.assertNotEqual(first["model"], second["model"])
        self.assertEqual(self.server.finalize("act", 1)["model"], first["model"])
        self.assertEqual(self.server.snapshot("act")["finalized_count"], 2)

    def test_real_frozen_mixed_precision_codec_payload_roundtrip(self):
        sys.path.insert(0, str(FROZEN_CODEC.parent))
        import importlib.util
        import torch
        spec = importlib.util.spec_from_file_location("frozen_mixed_precision_for_durable_tests", FROZEN_CODEC)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        before = sha256(FROZEN_CODEC)
        encoded = []
        def encode(values):
            result = module.encode(torch.tensor(values, dtype=torch.float32), budget=module.wire_size(4, 4), mode="mixed_mse")
            encoded.append(result)
            return result.wire
        real = Codec("frozen-mixed-precision-v1", encode, lambda wire: module.decode(wire).tolist())
        server = LocalServer(self.directory / "mp-server.sqlite", self.keys, {real.name: real})
        base = server.create_activity("mp", [0] * 4, layout="layout4-v1", codec=real.name, fixed_weights={"c1": 1.0})
        server.open_round("mp", 1)
        client = ClientStore(self.directory / "mp-client.sqlite", activity="mp", client="c1", key=self.keys["c1"],
                             layout="layout4-v1", initial_residual=[0] * 4)
        envelope = client.prepare(round_id=1, record_id="mp1", nonce="mp-nonce", base_model_hash=base,
                                  codec=real, update=[0.25, -.5, 1.0, -.75])
        self.assertEqual(envelope.packet.payload, encoded[0].wire)
        self.assertVector(real.decode(envelope.packet.payload), encoded[0].sent.tolist())
        server.register(envelope)
        server.accept(envelope)
        model = server.finalize("mp", 1)["model"]
        self.assertVector(model, real.decode(envelope.packet.payload))
        self.assertTrue(client.commit_if_incorporated("mp1", server.query))
        self.assertEqual(before, sha256(FROZEN_CODEC))
        self.evidence.update(payload_bytes=len(envelope.packet.payload), codec="frozen mixed_precision.codec",
                             frozen_codec_sha256=before, source_unchanged=True, synthetic_dimension=4)

    def test_key_holder_can_authenticate_poisoned_update_trust_boundary(self):
        # An entity with c1's HMAC key can create a different, valid message.
        # The server intentionally has no poisoning detector in this fixture.
        original = self.envelopes["c1"]
        forged = Envelope.sign(replace(original.packet, payload=self.codec.encode([100, 0, -100, 0])), self.keys["c1"])
        self.server.register(forged)
        self.assertEqual(self.server.accept(forged).state, ACCEPTED)
        self.server.register(self.envelopes["c2"])
        self.server.accept(self.envelopes["c2"])
        poisoned = self.server.finalize("act", 1)["model"]
        self.assertNotEqual(poisoned, self.expected_model())
        with self.assertRaises(AuthenticationError):
            self.clients["c1"].commit_if_incorporated("record-c1", self.server.query)
        self.evidence.update(authorized_key_holder_forgery_accepted=True, poisoning_detection=False,
                             original_pending_digest_still_blocks_wrong_ef_commit=True)


def make_unsigned_mutation_test(name, mutate):
    def test(self):
        original = self.envelopes["c1"]
        self.server.register(original)
        mutated = Envelope(mutate(original.packet).to_bytes(), original.tag)
        with self.assertRaises(AuthenticationError):
            self.server.accept(mutated)
        self.assertEqual(self.server.query("record-c1").state, REGISTERED)
        self.assertEqual(self.server.snapshot("act")["finalized_count"], 0)
        self.evidence.update(mutation=name, original_tag_reused=True, rejected_before_incorporation=True)
    return test


def flip_payload(packet, offset):
    payload = bytearray(packet.payload)
    payload[offset] ^= 1
    return replace(packet, payload=bytes(payload))


UNSIGNED_MUTATIONS = {
    "numeric_code": lambda p: flip_payload(p, 24),
    "sparse_index": lambda p: flip_payload(p, 20),
    "bitwidth": lambda p: flip_payload(p, 16),
    "scale": lambda p: flip_payload(p, 12),
    "payload_truncate": lambda p: replace(p, payload=p.payload[:-1]),
    "payload_append": lambda p: replace(p, payload=p.payload + b"x"),
    "activity": lambda p: replace(p, activity="other"),
    "client": lambda p: replace(p, client="c2"),
    "round": lambda p: replace(p, round_id=2),
    "record_id": lambda p: replace(p, record_id="other"),
    "nonce": lambda p: replace(p, nonce="other"),
    "base_model": lambda p: replace(p, base_model_hash="0" * 64),
    "layout": lambda p: replace(p, layout="other-layout"),
    "codec": lambda p: replace(p, codec="other-codec"),
}
for mutation_name, mutation in UNSIGNED_MUTATIONS.items():
    setattr(ProtocolTests, "test_hmac_rejects_mutated_" + mutation_name,
            make_unsigned_mutation_test(mutation_name, mutation))


def make_signed_context_test(name, fields):
    def test(self):
        packet = replace(self.envelopes["c1"].packet, **fields)
        envelope = Envelope.sign(packet, self.keys[packet.client])
        self.server.register(envelope)
        with self.assertRaises(StateError):
            self.server.accept(envelope)
        self.assertEqual(self.server.query(packet.record_id).state, REGISTERED)
        self.evidence.update(context_replacement=name, hmac_is_valid=True, rejected_by_fixed_server_context=True)
    return test


for context_name, replacements in {
    "activity": {"activity": "other-activity"},
    "client_outside_fixed_roster": {"client": "c3"},
    "round": {"round_id": 2},
    "stale_base_model": {"base_model_hash": "0" * 64},
    "layout": {"layout": "other-layout"},
    "codec": {"codec": "other-codec"},
}.items():
    setattr(ProtocolTests, "test_signed_context_rejects_" + context_name,
            make_signed_context_test(context_name, replacements))


def make_signed_invalid_payload_test(name, mutate):
    def test(self):
        original = self.envelopes["c1"]
        invalid = Envelope.sign(replace(original.packet, payload=mutate(original.packet.payload)), self.keys["c1"])
        self.server.register(invalid)
        with self.assertRaises(ProtocolError):
            self.server.accept(invalid)
        self.assertEqual(self.server.query("record-c1").state, REGISTERED)
        self.evidence.update(payload_error=name, hmac_is_valid=True, decoder_rejected=True)
    return test


def overwrite(payload, offset, value):
    return payload[:offset] + value + payload[offset + len(value):]


for invalid_name, invalid_mutate in {
    "out_of_bounds_index": lambda p: overwrite(p, 20, struct.pack("<I", 4)),
    "duplicate_index": lambda p: overwrite(p, 25, struct.pack("<I", 0)),
    "unsupported_bitwidth": lambda p: overwrite(p, 16, b"\x07"),
    "unsupported_scale": lambda p: overwrite(p, 12, struct.pack("<f", 0.5)),
    "truncated_payload": lambda p: p[:-1],
}.items():
    setattr(ProtocolTests, "test_signed_invalid_payload_" + invalid_name,
            make_signed_invalid_payload_test(invalid_name, invalid_mutate))


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = []

    def startTest(self, test):
        self.started = time.perf_counter()
        super().startTest(test)

    def _record(self, test, passed, error=None):
        row = {"test": test._testMethodName, "passed": passed,
               "elapsed_seconds": time.perf_counter() - self.started,
               "evidence": getattr(test, "evidence", {})}
        if error:
            row["failure"] = self._exc_info_to_string(error, test)
        self.rows.append(row)

    def addSuccess(self, test):
        self._record(test, True)
        super().addSuccess(test)

    def addFailure(self, test, err):
        self._record(test, False, err)
        super().addFailure(test, err)

    def addError(self, test, err):
        self._record(test, False, err)
        super().addError(test, err)


def main():
    global ARTIFACT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    result_dir = args.output_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT = result_dir / "synthetic_cases"
    if ARTIFACT_ROOT.exists():
        raise SystemExit("Choose a fresh output directory; existing evidence is preserved.")
    frozen_hash_before = sha256(FROZEN_CODEC)
    old_toy = WORKSPACE / "work/tinyproto_diag/tools/integrity_attack_matrix.py"
    old_toy_hash_before = sha256(old_toy)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests)
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
    report = {
        "scope": "synthetic protocol tests; local SQLite simulated registration and trusted aggregation",
        "not_measured": ["ML performance", "consensus", "network latency", "real permissioned-chain throughput",
                         "end-to-end distributed atomicity", "poisoning resistance"],
        "python": sys.executable, "python_version": platform.python_version(),
        "sqlite_version": sqlite3.sqlite_version, "hmac": "HMAC-SM3", "tag_bytes": 32,
        "frozen_codec_source": str(FROZEN_CODEC), "frozen_codec_sha256_before": frozen_hash_before,
        "frozen_codec_sha256_after": sha256(FROZEN_CODEC),
        "old_toy_source_sha256_before": old_toy_hash_before, "old_toy_source_sha256_after": sha256(old_toy),
        "protocol_source_sha256": sha256(ROOT / "protocol.py"),
        "test_source_sha256": sha256(Path(__file__)),
        "tests_run": result.testsRun, "passed": sum(row["passed"] for row in result.rows),
        "failed": len(result.failures) + len(result.errors),
        "elapsed_seconds": time.perf_counter() - started,
        "database_files": len(list(ARTIFACT_ROOT.rglob("*.sqlite"))), "cases": result.rows,
    }
    (result_dir / "test_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("JSON results:", result_dir / "test_results.json")
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()

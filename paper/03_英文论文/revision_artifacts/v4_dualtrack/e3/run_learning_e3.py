"""Real TRAIN -> HTTP -> local EVM binding -> durable model/EF diagnostic.

Run with the existing Tinyproto environment. This adds no main-quality seed,
reads no official TEST data, and claims no public-chain/multi-node finality.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import secrets
import select
import sqlite3
import subprocess
import sys
import tempfile
import time
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
import numpy as np
import torch
from torch.utils.data import TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = Path('/Users/wind/Desktop/Codex/paper')
METHOD = ROOT / "05_核心方法研发"
OUTPUT = None


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def main():
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--runtime", default="/private/tmp/paper-chain-runtime")
    arguments.add_argument("--clients", type=int, choices=(2,5), required=True)
    arguments.add_argument("--rounds", type=int, choices=(2,3), required=True)
    arguments.add_argument("--out", type=Path, required=True)
    args = arguments.parse_args()
    global OUTPUT
    OUTPUT = args.out / "result.json"
    if (args.clients,args.rounds) not in ((2,2),(5,3)): raise ValueError("fixed E3 matrix")
    source_lock=json.loads((HERE/'SOURCE_LOCK.json').read_text())
    for path,digest in source_lock['inputs'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError('E3 locked input changed: '+path)
    if OUTPUT.exists():
        raise ValueError("Preserve the existing successful diagnostic result")
    started = time.perf_counter()
    smoke = module("_chain_learning_training", METHOD / "scripts/learning_protocol_smoke.py")
    smoke.CLIENTS = tuple(range(args.clients))
    net = module("_chain_learning_network", HERE / "network_service.py")
    receipt_verifications=[]
    original_verify_receipt=net.verify_receipt
    def timed_receipt(*a,**kw):
        began=time.perf_counter_ns(); result=original_verify_receipt(*a,**kw)
        elapsed_ms=(time.perf_counter_ns()-began)/1e6
        raw=a[1]; body=json.loads(raw)["receipt"]
        receipt_verifications.append({"record_id":body["record_id"],"state":body["state"],"caller":sys._getframe(1).f_code.co_name,"verification_ms":elapsed_ms,"receipt_body_bytes":len(raw),"raw_receipt":json.loads(raw)})
        return result
    net.verify_receipt=timed_receipt
    protocol = module("_chain_learning_protocol", smoke.PROTOCOL)
    codec_module = module("_chain_learning_codec", smoke.CODEC)
    torch.set_num_threads(1)
    audit = {"forbidden_test_open_attempts": [], "official_train_files_opened": [],
             "loopback_connection_attempts": 0, "non_loopback_attempts": []}
    def guard(event, values):
        if event == "open" and values and isinstance(values[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(values[0]))
            if path.name == "test_batch" or path.name.startswith("t10k"):
                audit["forbidden_test_open_attempts"].append(str(path))
                raise AssertionError("Official TEST access forbidden")
            if path.name.startswith("data_batch_") and str(smoke.RAW) in str(path.absolute()):
                audit["official_train_files_opened"].append(path.name)
        if event == "socket.getaddrinfo":
            if values[0] not in ("127.0.0.1", "localhost", b"127.0.0.1", b"localhost"):
                audit["non_loopback_attempts"].append(str(values[0]))
                raise AssertionError("Only loopback network endpoints allowed")
        if event == "socket.connect":
            address = values[1]
            if not isinstance(address, tuple) or address[0] not in ("127.0.0.1", "::1"):
                audit["non_loopback_attempts"].append(str(address))
                raise AssertionError("Only loopback network endpoints allowed")
            audit["loopback_connection_attempts"] += 1
    sys.addaudithook(guard)
    checks = []
    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)
    def rejected(name, function, error=protocol.AuthenticationError):
        try:
            returned = function()
        except error:
            check(name, True)
        else:
            raise AssertionError("Expected rejection: " + name + "; actual=" + str(returned)[:1000])
    def transport_dropped(name, function):
        rejected(name, function, net.TransportError)
    configuration = json.loads(smoke.CONFIG.read_text())
    manifest = configuration["data_manifest"]
    check("original_seed_and_partition", configuration["seed"] == 20261231 and configuration["partition_seed"] == 20261217)
    paths = [Path(__file__), HERE / "network_service.py", HERE / "ledger_bridge.cjs",
             METHOD / "scripts/learning_protocol_smoke.py", Path(args.runtime)/"package-lock.json",
             HERE / "UploadRegistry.sol", smoke.CONFIG, smoke.PROTOCOL, smoke.CODEC, smoke.MODEL,
             smoke.PROTOCOL.parent / "specs/frozen-mixed-precision-v1.json"]
    identities = {str(path): smoke.file_sha(path) for path in paths}
    spec = json.loads(paths[-1].read_text())
    check("frozen_codec_source_pinned", smoke.file_sha(smoke.CODEC) == spec["reference_source_sha256"])
    images, labels = [], []
    for row in manifest["official_train_batch_files"]:
        path = smoke.RAW / row["name"]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        check("TRAIN_hash:" + row["name"], digest == row["sha256"])
        identities[str(path)] = digest
        batch = pickle.loads(raw, encoding="latin1")
        images.append(np.asarray(batch["data"], dtype=np.uint8))
        labels.extend(batch["labels"])
    images = np.concatenate(images).reshape(50000, 3, 32, 32)
    labels = np.asarray(labels, dtype=np.int64)
    check("TRAIN_label_hash", hashlib.sha256(labels.astype("<i8").tobytes()).hexdigest() == manifest["official_train_labels_sha256"])
    datasets, counts = {}, {}
    for client in smoke.CLIENTS:
        indices = np.asarray(manifest["train_indices"][client], dtype=np.int64)
        counts[client] = np.bincount(labels[indices], minlength=10)
        check(f"c{client}:original_TRAIN_partition", counts[client].tolist() == manifest["client_label_counts"][client]
              and not set(indices) & set(manifest["holdout_indices"][client]))
        x = (torch.from_numpy(images[indices].astype(np.float32)) / 255 - .5) / .5
        datasets[client] = TensorDataset(x, torch.from_numpy(labels[indices].copy()))
    del images, labels
    torch.manual_seed(configuration["seed"])
    model = smoke.load_model_class()().eval()
    current = tuple(float(v) for v in smoke.flat(model).tolist())
    size = len(current)
    check("original_SmallCNN_dimension", size == 136874)
    layout_rows, cursor = [], 0
    for name, parameter in model.named_parameters():
        layout_rows.append({"name": name, "shape": list(parameter.shape), "dtype": "float32",
                            "offset": cursor, "numel": parameter.numel()})
        cursor += parameter.numel()
    layout = hashlib.sha256(net.canonical(layout_rows)).hexdigest()
    weights = {f"c{c}": len(datasets[c]) / sum(len(datasets[i]) for i in smoke.CLIENTS) for c in smoke.CLIENTS}
    counters = {"protocol_encode": 0, "protocol_decode": 0, "reference_encode": 0, "reference_decode": 0}
    def encode(values, arm):
        counters[arm + "_encode"] += 1
        wire = codec_module.encode(torch.tensor(values, dtype=torch.float64), smoke.CAP, mode="uniform3").wire
        check(f"{arm}:payload_length:{counters[arm + '_encode']}", len(wire) == 54067)
        return wire
    def decode(payload, arm):
        counters[arm + "_decode"] += 1
        return tuple(float(v) for v in codec_module.decode(payload).tolist())
    codec = protocol.Codec(smoke.CODEC_NAME, protocol.frozen_spec_identity(smoke.CODEC_NAME),
                            lambda values: encode(values, "protocol"), lambda wire: decode(wire, "protocol"))
    keys = {name: secrets.token_bytes(32) for name in weights}
    session = secrets.token_hex(24)
    activity = 'real-train-anchored-uploads-v1----'
    direct_residuals = {c: tuple(0. for _ in range(size)) for c in smoke.CLIENTS}
    prior_wires, old_receipts, round_reports = {}, {}, []
    app_metrics, server_responses, service_events = [], [], []
    bridge_process, service = None, None
    initial_hash = smoke.vector_sha(current)
    with tempfile.TemporaryDirectory(prefix="chain-learning-", dir="/private/tmp") as folder:
        temporary_path = Path(folder)
        server_path = temporary_path / "server.sqlite"
        try:
            bridge_process = subprocess.Popen(["node", str(HERE / "ledger_bridge.cjs"),
                    "--runtime", args.runtime, "--workdir", str(temporary_path / "evm"), "--clients", str(args.clients)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            readable, _, _ = select.select([bridge_process.stdout], [], [], 45)
            if not readable:
                raise RuntimeError("EVM bridge did not become ready within 45 seconds")
            ready = json.loads(bridge_process.stdout.readline())
            check("real_local_EVM_bridge_ready", ready["status"] == "READY")
            ledger = net.LedgerClient(ready["url"], ready["chain_id"], ready["contract_address"], ready["client_addresses"])
            def start_service():
                value = net.LearningService(protocol, server_path, keys, {codec.name: codec}, ledger, session)
                value.start()
                return value
            def stop_service():
                if service:
                    service.stop()
                    server_responses.extend(service.responses)
                    service_events.extend(service.events)
            def reopen_client(client):
                return protocol.ClientStore(temporary_path / f"c{client}.sqlite", activity=activity,
                    client=f"c{client}", key=keys[f"c{client}"], layout=layout, initial_residual=[0.] * size)
            def wrappers(stores):
                return {c: net.NetworkClient(protocol, stores[c], keys[f"c{c}"], session,
                                             ledger, service.url, app_metrics) for c in smoke.CLIENTS}
            service = start_service()
            service.server.create_activity(activity, current, layout=layout, codec=codec.name, fixed_weights=weights)
            clients = {c: reopen_client(c) for c in smoke.CLIENTS}
            network_clients = wrappers(clients)
            for round_id in range(1,args.rounds+1):
                round_start = time.perf_counter()
                base_hash = service.server.open_round(activity, round_id)
                check(f"r{round_id}:server_direct_anchor_equal", service.server.snapshot(activity)["model"] == current)
                smoke.assign(model, current)
                envelopes, decoded, next_residuals, training, chain_records = {}, {}, {}, [], {}
                before_round_counts = dict(counters)
                for client in smoke.CLIENTS:
                    name = f"c{client}"
                    seed = configuration["seed"] + 100 * (round_id - 1) + client
                    delta, training_stats = smoke.train(model, datasets[client], counts[client], seed)
                    training.append({"client": client, **training_stats})
                    effective = tuple(a + float(b) for a, b in zip(direct_residuals[client], delta))
                    reference_wire = encode(effective, "reference")
                    decoded[client] = decode(reference_wire, "reference")
                    next_residuals[client] = tuple(a - b for a, b in zip(effective, decoded[client]))
                    old_state = clients[client].snapshot()
                    envelope = clients[client].prepare(round_id=round_id, record_id=f"r{round_id}-{name}",
                        nonce=f"nonce-r{round_id}-{name}", base_model_hash=base_hash, codec=codec, update=delta.tolist())
                    envelopes[client] = envelope
                    check(f"r{round_id}:{name}:new_payload_matches_direct", envelope.packet.payload == reference_wire)
                    check(f"r{round_id}:{name}:prepare_keeps_old_EF", clients[client].snapshot() == old_state)
                    if round_id >= 2:
                        check(f"r2:{name}:round2_uses_nonzero_EF", any(direct_residuals[client]))
                        check(f"r2:{name}:new_wire_differs", prior_wires[client] != reference_wire)
                        raw_old, challenge_old = old_receipts[client]
                        rejected(f"r2:{name}:old_incorporation_receipt_rejected_for_new_pending",
                            lambda: network_clients[client].commit(envelope, raw_old, challenge_old))
                        check(f"r2:{name}:old_receipt_rejection_preserves_EF", clients[client].snapshot() == old_state)
                    prior_wires[client] = reference_wire
                    _, unknown_raw, unknown, unknown_challenge = network_clients[client].request(envelope)
                    check(f"r{round_id}:{name}:authenticated_UNKNOWN_cannot_commit", unknown.state == "UNKNOWN"
                          and not network_clients[client].commit(envelope, unknown_raw, unknown_challenge))
                    before_rejection = dict(counters)
                    ledger_calls = len(ledger.metrics)
                    mutation = bytearray(envelope.wire)
                    mutation[-1] ^= 1
                    status, _, _, _ = network_clients[client].request(protocol.Envelope(bytes(mutation), envelope.tag), "/upload")
                    check(f"r{round_id}:{name}:payload_mutation_rejected_before_chain_and_decode",
                          status == 400 and counters == before_rejection and len(ledger.metrics) == ledger_calls)
                    wrong_packet = replace(envelope.packet, record_id=f"wrong-r{round_id}-{name}",
                              nonce=f"wrong-nonce-r{round_id}-{name}", base_model_hash="0" * 64)
                    wrong_envelope = protocol.Envelope.sign(wrong_packet, keys[name])
                    status, _, _, _ = network_clients[client].request(wrong_envelope, "/upload")
                    check(f"r{round_id}:{name}:signed_wrong_context_rejected_before_chain_and_decode",
                          status == 400 and counters == before_rejection and len(ledger.metrics) == ledger_calls)
                    if round_id == 1 and client == 0:
                        transport_dropped("chain_committed_ACK_lost_before_local_register", lambda:
                            network_clients[client].request(envelope, "/upload",
                                headers={"X-Diagnostic-Drop-After-Ledger": "1"}))
                        check("chain_success_does_not_imply_local_registration_or_EF_commit",
                              service.server.query(envelope.packet.record_id).state == "UNKNOWN"
                              and clients[client].snapshot() == old_state)
                        stop_service()
                        del network_clients, clients
                        service = start_service()
                        clients = {c: reopen_client(c) for c in smoke.CLIENTS}
                        network_clients = wrappers(clients)
                        retry_before = dict(counters)
                        retry = clients[client].prepare(round_id=round_id, record_id=envelope.packet.record_id,
                            nonce=envelope.packet.nonce, base_model_hash=base_hash, codec=codec, update=[999.] * size)
                        check("pre_register_reopen_retry_is_original_pending_bytes_no_encode",
                              retry == envelope and counters == retry_before)
                    if client == smoke.CLIENTS[-1]:
                        transport_dropped(f"r{round_id}:model_finalized_ACK_lost", lambda:
                            network_clients[client].request(envelope, "/upload", headers={"X-Diagnostic-Drop-ACK": "1"}))
                    else:
                        status, accepted_raw, accepted, accepted_challenge = network_clients[client].request(envelope, "/upload")
                        check(f"r{round_id}:first_client_is_ACCEPTED", status == 200 and accepted.state == protocol.ACCEPTED)
                        check(f"r{round_id}:chain_registered_ACCEPTED_cannot_commit_EF",
                              not network_clients[client].commit(envelope, accepted_raw, accepted_challenge))
                        forged_state = json.loads(accepted_raw)
                        forged_state["receipt"]["state"] = protocol.INCORPORATED
                        forged_state["receipt"]["result_model_hash"] = "1" * 64
                        rejected(f"r{round_id}:unauthenticated_state_upgrade_cannot_commit_EF", lambda:
                            network_clients[client].commit(envelope, net.canonical(forged_state), accepted_challenge))
                        state_before_stop = service.server.snapshot(activity)
                        rejected(f"r{round_id}:missing_client_safe_stop_no_finalize",
                                 lambda: service.server.finalize(activity, round_id), protocol.StateError)
                        rejected(f"r{round_id}:missing_client_prevents_next_round",
                                 lambda: service.server.open_round(activity, round_id + 1), protocol.StateError)
                        check(f"r{round_id}:safe_stop_preserves_model_and_old_EF",
                              service.server.snapshot(activity) == state_before_stop
                              and state_before_stop["model"] == current
                              and clients[client].snapshot() == old_state and clients[client].pending() == envelope)
                    chain_records[client] = ledger.readback(net.binding(protocol, envelope))
                    invalid_depth = dict(chain_records[client], confirmation_depth=1)
                    rejected(f"r{round_id}:{name}:insufficient_confirmation_depth_rejected", lambda:
                        ledger.verify(net.binding(protocol, envelope), invalid_depth), ValueError)
                    invalid_binding = dict(chain_records[client], readback=dict(chain_records[client]["readback"], digest="0" * 64))
                    rejected(f"r{round_id}:{name}:wrong_chain_readback_rejected", lambda:
                        ledger.verify(net.binding(protocol, envelope), invalid_binding), ValueError)
                direct_delta = tuple(math.fsum(weights[f"c{c}"] * decoded[c][i] for c in smoke.CLIENTS) for i in range(size))
                expected_model = tuple(a + b for a, b in zip(current, direct_delta))
                check(f"r{round_id}:aggregate_model_equals_direct_bitwise", service.server.snapshot(activity)["model"] == expected_model)
                check(f"r{round_id}:lost_ACK_keeps_EF_uncommitted",
                      all(clients[c].snapshot()["generation"] == round_id - 1 and clients[c].pending() is not None for c in smoke.CLIENTS))
                stop_service()
                del network_clients, clients
                service = start_service()
                clients = {c: reopen_client(c) for c in smoke.CLIENTS}
                network_clients = wrappers(clients)
                before_recovery = dict(counters)
                register_calls_before = sum(row["label"] == "chain_register" for row in ledger.metrics)
                for client in smoke.CLIENTS:
                    envelope = envelopes[client]
                    retry = clients[client].prepare(round_id=round_id, record_id=envelope.packet.record_id,
                        nonce=envelope.packet.nonce, base_model_hash=base_hash, codec=codec, update=[123.] * size)
                    check(f"r{round_id}:c{client}:post_finalize_reopen_exact_retry", retry == envelope)
                    status, retry_raw, retry_receipt, retry_challenge = network_clients[client].request(retry, "/upload")
                    check(f"r{round_id}:c{client}:duplicate_upload_returns_INCORPORATED",
                          status == 200 and retry_receipt.state == protocol.INCORPORATED)
                    status, raw, recovered, challenge = network_clients[client].request(envelope)
                    check(f"r{round_id}:c{client}:authenticated_query_recovers_INCORPORATED",
                          status == 200 and recovered.state == protocol.INCORPORATED)
                    tampered = json.loads(raw)
                    tampered["receipt"]["result_model_hash"] = "0" * 64
                    rejected(f"r{round_id}:c{client}:tampered_receipt_rejected",
                        lambda: network_clients[client].commit(envelope, net.canonical(tampered), challenge))
                    rejected(f"r{round_id}:c{client}:old_challenge_receipt_rejected",
                        lambda: network_clients[client].commit(envelope, raw, secrets.token_hex(32)))
                    check(f"r{round_id}:c{client}:verified_recovery_commits_EF_once",
                          network_clients[client].commit(envelope, raw, challenge))
                    check(f"r{round_id}:c{client}:second_commit_is_idempotent",
                          not network_clients[client].commit(envelope, raw, challenge))
                    state = clients[client].snapshot()
                    check(f"r{round_id}:c{client}:EF_equals_direct_bitwise_and_generation_once",
                          state["residual"] == next_residuals[client] and state["generation"] == round_id and clients[client].pending() is None)
                    old_receipts[client] = raw, challenge
                check(f"r{round_id}:recovery_adds_no_encode_decode_or_chain_register",
                      counters == before_recovery and register_calls_before == sum(row["label"] == "chain_register" for row in ledger.metrics))
                repeated = service.server.finalize(activity, round_id)
                check(f"r{round_id}:duplicate_finalize_is_idempotent", not repeated["newly_finalized"] and repeated["model"] == expected_model)
                check(f"r{round_id}:model_finalizes_exactly_once", service.server.snapshot(activity)["finalized_count"] == round_id)
                round_reports.append({"round": round_id, "training": training,
                    "gradient_evaluations": sum(row["gradient_evaluations"] for row in training),
                    "base_model_sha256_fp64": smoke.vector_sha(current),
                    "result_model_sha256_fp64": smoke.vector_sha(expected_model),
                    "result_model_sm3": repeated["model_hash"], "aggregate_sha256_fp64": smoke.vector_sha(direct_delta),
                    "protocol_direct_model_max_abs_error": 0.0, "protocol_direct_residual_max_abs_error": 0.0,
                    "clients": [{"client": c, "payload_bytes": len(envelopes[c].packet.payload),
                        "application_envelope_and_tag_bytes": len(envelopes[c].wire) + len(envelopes[c].tag),
                        "payload_sha256": hashlib.sha256(envelopes[c].packet.payload).hexdigest(),
                        "envelope_sha256": hashlib.sha256(envelopes[c].wire).hexdigest(),
                        "committed_residual_sha256_fp64": smoke.vector_sha(next_residuals[c]),
                        "EF_generation": clients[c].snapshot()["generation"], "chain": chain_records[c]} for c in smoke.CLIENTS],
                    "codec_calls_in_round": {key: counters[key] - before_round_counts[key] for key in counters},
                    "seconds": time.perf_counter() - round_start})
                current, direct_residuals = expected_model, next_residuals
                print(json.dumps({"round": round_id, "status": "PASS", "gradient_evaluations": round_reports[-1]["gradient_evaluations"]}), flush=True)
            check("all_fresh_real_training_updates_encoded_once", counters["protocol_encode"] == args.clients*args.rounds and counters["protocol_decode"] == 2*args.clients*args.rounds)
            check("exactly_clients_times_rounds_register_HTTP_requests", sum(row["label"] == "chain_register" for row in ledger.metrics) == args.clients*args.rounds)
            status, adversarial = ledger.request("POST", "/adversarial-checks", {}, "contract_adversarial_checks")
            check("contract_adversarial_checks_pass", status == 200
                  and adversarial.get("status", "PASS") == "PASS"
                  and adversarial["checks_passed"] == len(adversarial["checks"])
                  and adversarial["checks_passed"] >= 6)
            status, bridge_stats = ledger.request("GET", "/stats", label="bridge_final_stats")
            check("bridge_stats_available", status == 200)
            sqlite_integrity = {}
            for path in sorted(temporary_path.glob("*.sqlite")):
                with sqlite3.connect(path) as connection:
                    sqlite_integrity[path.name] = connection.execute("PRAGMA integrity_check").fetchone()[0]
            check("all_client_and_server_SQLite_integrity_checks", len(sqlite_integrity) == args.clients+1 and set(sqlite_integrity.values()) == {"ok"})
            stop_service()
            service = None
        finally:
            if service:
                service.stop()
            if bridge_process is not None:
                if bridge_process.poll() is None:
                    bridge_process.terminate()
                    try:
                        bridge_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        bridge_process.kill()
                        bridge_process.wait(timeout=5)
                if bridge_process.returncode not in (0, -15):
                    # The bridge implementation never writes private keys to logs.
                    error_text = bridge_process.stderr.read()
                    if error_text:
                        print(error_text[-3000:], file=sys.stderr)
    check("temporary_databases_and_chain_state_removed", not temporary_path.exists())
    check("source_and_config_unchanged", all(smoke.file_sha(Path(path)) == digest
        for path, digest in identities.items() if Path(path).parent != smoke.RAW))
    check("only_original_five_TRAIN_files_read", sorted(audit["official_train_files_opened"]) == [f"data_batch_{i}" for i in range(1, 6)])
    check("zero_TEST_and_external_network_access", not audit["forbidden_test_open_attempts"] and not audit["non_loopback_attempts"])
    check("actual_HTTP_body_bytes_received_equal_sent", sum(row["request_body_bytes"] for row in app_metrics)
          == sum(row["request_body_bytes_received"] for row in server_responses))
    check("actual_HTTP_response_body_bytes_received_equal_sent", sum(row["response_body_bytes_received"] for row in app_metrics)
          == sum(row["response_body_bytes_sent"] for row in server_responses))
    result = {"status": "PASS", "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": "Original CIFAR clients and actual CPU TRAIN rounds over loopback HTTP and local EVM development chain",
        "client_count":args.clients,"round_count":args.rounds,
        "profile": {"model": "SmallCNN", "parameters": size, "initialization_seed": configuration["seed"],
            "partition_seed": configuration["partition_seed"], "client_ids": list(smoke.CLIENTS),
            "client_sizes": [len(datasets[c]) for c in smoke.CLIENTS], "activity_weights": weights,
            "rounds": args.rounds, "initial_model_sha256_fp64": initial_hash,
            "arithmetic": "Both direct/protocol: FP32 local CNN/codec, float64 effective residual/subtraction/server model; math.fsum aggregation; FP32 projection into next-round CNN",
            "comparison": "One actual local-training delta per client-round supplies independently encoded direct and HTTP/protocol arms; system equivalence, not independent quality replicates",
            "loss": configuration["loss_scope"], "session": session},
        "receipt_verifications": receipt_verifications,
        "timing_boundaries": {"local_receipt_verification_ms":"JSON parsing + HMAC-SM3 + session/challenge/full-context + embedded chain-field verification; excludes HTTP/RPC/wait/SQLite commit", "ledger_readback_ms":"actual GET /record roundtrip including contract read/event/transaction/canonical-block/readback verification; includes loopback transport", "registration_ms":"actual POST /register including submission, inclusion, two descendant blocks, readback and loopback transport"},
        "rounds": round_reports, "counters": counters, "checks_passed": len(checks), "checks": checks,
        "total_gradient_evaluations": sum(row["gradient_evaluations"] for row in round_reports),
        "fresh_training_payload_bytes": sum(c["payload_bytes"] for row in round_reports for c in row["clients"]),
        "fresh_training_envelope_and_tag_bytes": sum(c["application_envelope_and_tag_bytes"] for row in round_reports for c in row["clients"]),
        "network": {"client_HTTP_requests": app_metrics, "server_HTTP_responses": server_responses,
            "bridge_HTTP_requests": ledger.metrics,
            "total_client_request_body_bytes": sum(row["request_body_bytes"] for row in app_metrics),
            "total_client_request_HTTP_bytes_sent": sum(row["request_http_bytes_sent"] for row in app_metrics),
            "total_server_response_body_bytes": sum(row["response_body_bytes_sent"] for row in server_responses),
            "total_server_response_HTTP_bytes_sent": sum(row["response_http_bytes_sent"] for row in server_responses),
            "accounting_boundary": "Actual serialized HTTP body bytes in both directions; client request wire HTTP headers counted by send(), server response exact serialized header/body writes. Excludes TCP/IP headers, TLS, retransmissions and remote-network/consensus cost. Includes diagnostic attacks, exact retries and receipt queries. Bridge RPC metrics are separately reported by Node."},
        "chain": {"chain_id": ready["chain_id"], "contract_address": ready["contract_address"],
            "client_addresses": ready["client_addresses"], "confirmation_rule": "status=1, canonical transaction block hash, exact contract/chain/readback, head number minus inclusion block >=2 (two descendants; confirmations=3)",
            "bridge_stats": bridge_stats, "adversarial_checks": adversarial},
        "recovery_events": service_events, "sqlite_integrity": sqlite_integrity,
        "temporary_databases_and_chain_state_removed": True, "HMAC_keys_persisted_or_reported": False,
        "private_keys_reported": False, "audit": audit, "source_and_input_sha256": identities,
        "limitations": ["This diagnostic is not another main predictive-quality seed or twenty-client experiment.",
            "Single-node Ganache development EVM with manually mined descendants does not establish PoW, multi-node consensus or public-chain finality.",
            "The registry attests a binding of upload metadata/digest, not honest training, poisoning resistance or an embedded model watermark.",
            "The server, its HMAC keys, SQLite incorporation authority, bridge/EVM endpoint and matching decoder remain trusted.",
            "Fixed full participation retains OPEN/pending/old EF during a missing client; it resumes the same round when that client returns. Permanent dropout advancement is not implemented.",
            "A chain registration never authorizes EF commit; only a fresh context-bound HMAC-SM3 INCORPORATED receipt does.",
            "Recovery rebuilds the HTTP service and SQLite objects after deliberate connection closure; no process kill or OS crash was injected.",
            "Float64 reference equivalence applies to this system profile, not to the original FP32/MPS main-training trajectory.",
            "Observed loopback latency and gas are development diagnostic measurements, not production latency, throughput, network fees or consensus cost."],
        "elapsed_seconds": time.perf_counter() - started}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "checks_passed": len(checks), "result": str(OUTPUT),
                      "gradient_evaluations": result["total_gradient_evaluations"], "seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()

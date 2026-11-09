"""Two-client, two-round real-TRAIN learning/durable-upload integration smoke.

This is a local system diagnostic, not another predictive-quality seed or a
blockchain experiment. Training/codec use FP32; durable residuals, aggregation,
and the direct reference both use the existing protocol's double arithmetic.
No historical checkpoint, final TEST file, external network, or private key is
written. SQLite databases live only in a temporary directory and are removed.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import secrets
import sys
import tempfile
import time

sys.dont_write_bytecode = True
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
METHOD = ROOT / '05_核心方法研发'
CONFIG = METHOD / 'results/predictive_pfl_confirm/development_cifar_p20261217_s20261231/fedlc_uniform3/config.json'
PROTOCOL = METHOD / 'repro/work/durable_integrity_v2/protocol.py'
CODEC = METHOD / 'scripts/mixed_precision/codec.py'
MODEL = ROOT / '02_实验工程/tools/fedavg_update_diagnostic.py'
RAW = ROOT / '02_实验工程/dataset/Cifar10/rawdata/cifar-10-batches-py'
OUTPUT = METHOD / 'results/learning_protocol_smoke/result.json'
CLIENTS = (0, 1)
ROUNDS = 2
CAP = 54808
ACTIVITY = 'real-train-two-client-local-smoke-v1'
CODEC_NAME = 'frozen-mixed-precision-v1'


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector_sha(values):
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()]).cpu()


def assign(model, values):
    cursor = 0
    source = torch.tensor(values, dtype=torch.float32)
    with torch.no_grad():
        for parameter in model.parameters():
            count = parameter.numel()
            parameter.copy_(source[cursor:cursor + count].view_as(parameter))
            cursor += count
    assert cursor == len(values)


def load_model_class():
    tree = ast.parse(MODEL.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SmallCNN')
    scope = {'nn': nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(MODEL), 'exec'), scope)
    return scope['SmallCNN']


def train(model, dataset, counts, seed):
    torch.manual_seed(seed)
    local = copy.deepcopy(model).train()
    before = flat(model).clone()
    shift = -torch.tensor(np.maximum(counts, 1) ** (-.25), dtype=torch.float32)
    loader = DataLoader(dataset, batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.SGD(local.parameters(), lr=.03)
    losses = []
    for images, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(local(images) + shift, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    delta = flat(local) - before
    assert torch.equal(flat(model), before)
    assert torch.isfinite(delta).all() and torch.count_nonzero(delta)
    return delta.numpy().astype(np.float64), {
        'training_seed_for_client_round': seed, 'train_samples': len(dataset),
        'gradient_evaluations': len(loader), 'optimizer_steps': len(loader),
        'local_epochs': 1, 'batch_size': 64, 'mean_training_batch_loss': float(np.mean(losses)),
        'delta_sha256_fp64_cast': vector_sha(delta.tolist()),
        'delta_l2': float(delta.norm()), 'global_training_anchor_unchanged': True,
    }


def main():
    started = time.perf_counter()
    if OUTPUT.exists():
        raise ValueError('Diagnostic already exists; preserve original result rather than overwrite')
    torch.set_num_threads(1)
    audit = {'forbidden_test_open_attempts': [], 'network_attempts': 0, 'official_train_files_opened': []}
    def guard(event, arguments):
        if event == 'open' and arguments and isinstance(arguments[0], (str, bytes, os.PathLike)):
            value = os.fsdecode(arguments[0])
            name = Path(value).name
            if name == 'test_batch' or name.startswith('t10k'):
                audit['forbidden_test_open_attempts'].append(value)
                raise AssertionError('Official TEST access is forbidden')
            if name.startswith('data_batch_') and str(RAW) in str(Path(value).absolute()):
                audit['official_train_files_opened'].append(name)
        if event in ('socket.connect', 'socket.getaddrinfo'):
            audit['network_attempts'] += 1
            raise AssertionError('No network in a local integration diagnostic')
    sys.addaudithook(guard)
    configuration = json.loads(CONFIG.read_text())
    manifest = configuration['data_manifest']
    assert configuration['seed'] == 20261231 and configuration['partition_seed'] == 20261217
    identities = {str(p): file_sha(p) for p in (Path(__file__), CONFIG, PROTOCOL, CODEC, MODEL)}
    protocol = module('_learning_smoke_durable_protocol', PROTOCOL)
    codec_module = module('_learning_smoke_frozen_codec', CODEC)
    spec_path = PROTOCOL.parent / 'specs' / (CODEC_NAME + '.json')
    identities[str(spec_path)] = file_sha(spec_path)
    wire_spec = json.loads(spec_path.read_text())
    assert file_sha(CODEC) == wire_spec['reference_source_sha256']
    assert protocol.frozen_spec_identity(CODEC_NAME) == file_sha(spec_path)

    images, labels = [], []
    for record in manifest['official_train_batch_files']:
        path = RAW / record['name']
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == record['sha256']
        identities[str(path)] = digest
        batch = pickle.loads(raw, encoding='latin1')
        images.append(np.asarray(batch['data'], dtype=np.uint8))
        labels.extend(batch['labels'])
    images = np.concatenate(images).reshape(50000, 3, 32, 32)
    labels = np.asarray(labels, dtype=np.int64)
    assert hashlib.sha256(labels.astype('<i8').tobytes()).hexdigest() == manifest['official_train_labels_sha256']
    datasets, counts = {}, {}
    for client in CLIENTS:
        indices = np.asarray(manifest['train_indices'][client], dtype=np.int64)
        assert not set(indices) & set(manifest['holdout_indices'][client])
        counts[client] = np.bincount(labels[indices], minlength=10)
        assert counts[client].tolist() == manifest['client_label_counts'][client]
        x = torch.from_numpy(images[indices].astype(np.float32)) / 255
        x = (x - .5) / .5
        datasets[client] = TensorDataset(x, torch.from_numpy(labels[indices].copy()))
    assert not set(manifest['train_indices'][0]) & set(manifest['train_indices'][1])
    del images, labels

    torch.manual_seed(configuration['seed'])
    model = load_model_class()()
    model.eval()
    current = tuple(float(v) for v in flat(model).tolist())
    size = len(current)
    assert size == 136874
    layout_rows, cursor = [], 0
    for name, parameter in model.named_parameters():
        layout_rows.append({'name': name, 'shape': list(parameter.shape), 'dtype': 'float32',
                            'offset': cursor, 'numel': parameter.numel()})
        cursor += parameter.numel()
    layout = hashlib.sha256(json.dumps(layout_rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    weights = {'c' + str(c): len(datasets[c]) / sum(len(datasets[i]) for i in CLIENTS) for c in CLIENTS}
    counters = {'protocol_encode': 0, 'protocol_decode': 0, 'reference_encode': 0, 'reference_decode': 0}
    def encode_values(values, arm):
        counters[arm + '_encode'] += 1
        encoded = codec_module.encode(torch.tensor(values, dtype=torch.float64), CAP, mode='uniform3')
        assert len(encoded.wire) == 54067
        return encoded.wire
    def decode_values(payload, arm):
        counters[arm + '_decode'] += 1
        return tuple(float(v) for v in codec_module.decode(payload).tolist())
    codec = protocol.Codec(CODEC_NAME, protocol.frozen_spec_identity(CODEC_NAME),
                           lambda values: encode_values(values, 'protocol'),
                           lambda payload: decode_values(payload, 'protocol'))
    keys = {name: secrets.token_bytes(32) for name in weights}
    checks = []
    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)
    def rejected(name, function, error):
        before = dict(counters)
        try:
            function()
        except error:
            check(name, True)
        else:
            raise AssertionError('Expected rejection: ' + name)
        check(name + ':no_codec_execution', counters == before)

    direct_residuals = {c: tuple(0. for _ in range(size)) for c in CLIENTS}
    round_reports, prior_wires = [], {}
    initial_hash = vector_sha(current)
    temporary_path = None
    with tempfile.TemporaryDirectory(prefix='learning-protocol-smoke-', dir='/private/tmp') as folder:
        temporary_path = Path(folder)
        server_path = temporary_path / 'server.sqlite'
        server = protocol.LocalServer(server_path, keys, {codec.name: codec})
        server.create_activity(ACTIVITY, current, layout=layout, codec=codec.name, fixed_weights=weights)
        def reopen_client(c):
            return protocol.ClientStore(temporary_path / ('c' + str(c) + '.sqlite'), activity=ACTIVITY,
                    client='c' + str(c), key=keys['c' + str(c)], layout=layout, initial_residual=[0.] * size)
        clients = {c: reopen_client(c) for c in CLIENTS}
        for round_id in range(1, ROUNDS + 1):
            round_start = time.perf_counter()
            base_hash = server.open_round(ACTIVITY, round_id)
            check(f'r{round_id}:server_equals_direct_before_training', server.snapshot(ACTIVITY)['model'] == current)
            check(f'r{round_id}:base_hash_binds_double_model', base_hash == protocol.sm3(protocol.vector_bytes(current)))
            assign(model, current)
            transmitted, decoded, next_residuals, training = {}, {}, {}, []
            protocol_encode_before = counters['protocol_encode']
            model_before = server.snapshot(ACTIVITY)
            for client in CLIENTS:
                name = 'c' + str(client)
                seed = configuration['seed'] + 100 * (round_id - 1) + client
                delta, training_stats = train(model, datasets[client], counts[client], seed)
                training.append({'client': client, **training_stats})
                effective = tuple(a + float(b) for a, b in zip(direct_residuals[client], delta))
                reference_wire = encode_values(effective, 'reference')
                q = decode_values(reference_wire, 'reference')
                decoded[client] = q
                next_residuals[client] = tuple(a - b for a, b in zip(effective, q))
                snapshot = clients[client].snapshot()
                check(f'r{round_id}:{name}:preparation_uses_committed_residual', snapshot['residual'] == direct_residuals[client])
                envelope = clients[client].prepare(round_id=round_id, record_id=f'r{round_id}-{name}',
                            nonce=f'nonce-r{round_id}-{name}', base_model_hash=base_hash, codec=codec, update=delta.tolist())
                transmitted[client] = envelope
                check(f'r{round_id}:{name}:actual_new_encoder_equals_reference_wire', envelope.packet.payload == reference_wire)
                check(f'r{round_id}:{name}:pending_does_not_commit_EF', clients[client].snapshot() == snapshot)
                check(f'r{round_id}:{name}:unknown_cannot_commit', not clients[client].commit_if_incorporated(envelope.packet.record_id, server.query))
                registration = server.register(envelope)
                check(f'r{round_id}:{name}:registered_state', registration.state == protocol.REGISTERED)
                check(f'r{round_id}:{name}:registered_cannot_commit', not clients[client].commit_if_incorporated(envelope.packet.record_id, server.query))
                mutated = bytearray(envelope.wire)
                mutated[-1] ^= 1
                rejected(f'r{round_id}:{name}:mutated_payload_rejected',
                         lambda: server.accept(protocol.Envelope(bytes(mutated), envelope.tag)), protocol.AuthenticationError)
                wrong = replace(envelope.packet, record_id=f'wrong-r{round_id}-{name}',
                                nonce=f'wrong-nonce-r{round_id}-{name}', base_model_hash='0' * 64)
                wrong_envelope = protocol.Envelope.sign(wrong, keys[name])
                server.register(wrong_envelope)
                rejected(f'r{round_id}:{name}:context_substitution_rejected',
                         lambda: server.accept(wrong_envelope), protocol.StateError)
                check(f'r{round_id}:{name}:rejection_preserves_server_model_and_EF',
                      server.snapshot(ACTIVITY) == model_before and clients[client].snapshot() == snapshot)
                accepted = server.accept(envelope)
                check(f'r{round_id}:{name}:accepted_state', accepted.state == protocol.ACCEPTED)
                check(f'r{round_id}:{name}:accepted_cannot_commit', not clients[client].commit_if_incorporated(envelope.packet.record_id, server.query))
                if client == CLIENTS[0]:
                    rejected(f'r{round_id}:incomplete_participation_cannot_finalize',
                             lambda: server.finalize(ACTIVITY, round_id), protocol.StateError)
                if round_id > 1:
                    check(f'r{round_id}:{name}:new_training_payload_not_replay', reference_wire != prior_wires[client])
                    check(f'r{round_id}:{name}:nonzero_committed_EF_reused', any(direct_residuals[client]))
                prior_wires[client] = reference_wire
            direct_delta = tuple(math.fsum(weights['c' + str(c)] * decoded[c][i] for c in CLIENTS) for i in range(size))
            expected_model = tuple(a + b for a, b in zip(current, direct_delta))
            finalized = server.finalize(ACTIVITY, round_id)
            check(f'r{round_id}:first_finalize_changes_model_once', finalized['newly_finalized'])
            check(f'r{round_id}:protocol_aggregate_equals_direct_bitwise', finalized['model'] == expected_model)
            check(f'r{round_id}:model_really_changes', finalized['model'] != current)
            check(f'r{round_id}:unacknowledged_clients_still_pending',
                  all(clients[c].snapshot()['generation'] == round_id - 1 and clients[c].pending() is not None for c in CLIENTS))

            # Simulate acknowledgement loss, discard objects, then reopen the real DBs.
            del server, clients
            server = protocol.LocalServer(server_path, keys, {codec.name: codec})
            clients = {c: reopen_client(c) for c in CLIENTS}
            before_recovery_counters = dict(counters)
            for client in CLIENTS:
                envelope = transmitted[client]
                packet = envelope.packet
                retry = clients[client].prepare(round_id=round_id, record_id=packet.record_id,
                        nonce=packet.nonce, base_model_hash=base_hash, codec=codec, update=[123.] * size)
                check(f'r{round_id}:c{client}:reopen_retry_returns_identical_pending_bytes', retry == envelope)
                check(f'r{round_id}:c{client}:duplicate_register_is_incorporated', server.register(retry).state == protocol.INCORPORATED)
                check(f'r{round_id}:c{client}:duplicate_accept_is_incorporated', server.accept(retry).state == protocol.INCORPORATED)
                check(f'r{round_id}:c{client}:query_recovers_lost_ACK', server.query(packet.record_id).state == protocol.INCORPORATED)
                check(f'r{round_id}:c{client}:EF_commits_after_recovery', clients[client].commit_if_incorporated(packet.record_id, server.query))
                check(f'r{round_id}:c{client}:EF_commit_is_idempotent', not clients[client].commit_if_incorporated(packet.record_id, server.query))
                state = clients[client].snapshot()
                check(f'r{round_id}:c{client}:committed_EF_equals_direct_bitwise', state['residual'] == next_residuals[client])
                check(f'r{round_id}:c{client}:generation_advances_exactly_once', state['generation'] == round_id and clients[client].pending() is None)
            repeated = server.finalize(ACTIVITY, round_id)
            check(f'r{round_id}:finalize_is_idempotent', not repeated['newly_finalized'] and repeated['model'] == expected_model)
            check(f'r{round_id}:recovery_has_no_new_encode_or_decode', counters == before_recovery_counters)
            check(f'r{round_id}:one_protocol_encoding_per_new_training_update', counters['protocol_encode'] - protocol_encode_before == len(CLIENTS))
            check(f'r{round_id}:finalized_count_advances_once', server.snapshot(ACTIVITY)['finalized_count'] == round_id)
            round_reports.append({
                'round': round_id, 'base_model_hash_sm3': base_hash,
                'base_model_sha256_fp64': vector_sha(current),
                'result_model_hash_sm3': finalized['model_hash'],
                'result_model_sha256_fp64': vector_sha(expected_model),
                'aggregate_sha256_fp64': vector_sha(direct_delta),
                'protocol_direct_model_max_abs_error': 0.0,
                'protocol_direct_residual_max_abs_error': 0.0,
                'training': training,
                'clients': [{'client': c, 'payload_bytes': len(transmitted[c].packet.payload),
                    'application_envelope_and_tag_bytes': len(transmitted[c].wire) + len(transmitted[c].tag),
                    'payload_sha256': hashlib.sha256(transmitted[c].packet.payload).hexdigest(),
                    'envelope_sha256': hashlib.sha256(transmitted[c].wire).hexdigest(),
                    'committed_residual_sha256_fp64': vector_sha(next_residuals[c]),
                    'EF_generation': clients[c].snapshot()['generation']} for c in CLIENTS],
                'gradient_evaluations': sum(row['gradient_evaluations'] for row in training),
                'protocol_encodes_for_fresh_updates': counters['protocol_encode'] - protocol_encode_before,
                'recovery_encodes': 0, 'recovery_decodes': 0,
                'lost_acknowledgement_recovery': 'SQLite objects reopened; exact original bytes retried, query recovered incorporation, EF committed once',
                'seconds': time.perf_counter() - round_start,
            })
            current, direct_residuals = expected_model, next_residuals
            print(json.dumps({'round': round_id, 'status': 'PASS', 'gradient_evaluations': round_reports[-1]['gradient_evaluations'],
                              'protocol_encodes': counters['protocol_encode'] - protocol_encode_before,
                              'seconds': round_reports[-1]['seconds']}), flush=True)
        check('two_rounds_finalize_exactly_once', server.snapshot(ACTIVITY)['finalized_count'] == ROUNDS)
        check('four_new_training_updates_encoded_exactly_once_each', counters['protocol_encode'] == ROUNDS * len(CLIENTS))
        sqlite_integrity = {}
        for path in sorted(temporary_path.glob('*.sqlite')):
            import sqlite3
            with sqlite3.connect(str(path)) as connection:
                sqlite_integrity[path.name] = connection.execute('PRAGMA integrity_check').fetchone()[0]
        check('all_three_database_integrity_checks', set(sqlite_integrity.values()) == {'ok'} and len(sqlite_integrity) == 3)
    check('temporary_databases_removed', not temporary_path.exists())
    # TRAIN file hashes are already bound to the bytes actually loaded. Recheck
    # source/config identities without reopening the official data unnecessarily.
    check('source_and_configuration_unchanged', all(file_sha(Path(p)) == value for p, value in identities.items() if Path(p).parent != RAW))
    check('only_five_official_TRAIN_files_read', sorted(audit['official_train_files_opened']) == [f'data_batch_{i}' for i in range(1, 6)])
    check('no_official_TEST_or_network_access', not audit['forbidden_test_open_attempts'] and audit['network_attempts'] == 0)
    result = {
        'status': 'PASS', 'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'scope': 'Two original CIFAR clients, two fresh CPU full-local-epoch rounds with real TRAIN data; local durable-learning integration diagnostic',
        'profile': {'model': 'SmallCNN', 'parameters': size, 'initialization_seed': configuration['seed'],
                    'partition_seed': configuration['partition_seed'], 'client_ids': list(CLIENTS),
                    'client_sizes': [len(datasets[c]) for c in CLIENTS], 'activity_weights': weights,
                    'rounds': ROUNDS, 'loss': configuration['loss_scope'],
                    'local_training': 'FP32 FedLC-inspired tau1, lr0.03, batch64, one epoch; original deterministic client-round seed rule',
                    'arithmetic': 'Direct and protocol: Python float64 effective residual, subtraction, math.fsum weighted aggregation and float64 server model. Existing codec casts effective update to FP32. Shared SmallCNN is projected to FP32 before each round.',
                    'comparison': 'One actual local-training computation per client-round supplied to both direct-reference and durable-protocol paths, isolating protocol handling; reference/protocol encodes executed separately.',
                    'initial_model_sha256_fp64': initial_hash},
        'rounds': round_reports, 'checks_passed': len(checks), 'checks': checks,
        'counters': counters, 'total_gradient_evaluations': sum(row['gradient_evaluations'] for row in round_reports),
        'total_protocol_payload_bytes': sum(c['payload_bytes'] for row in round_reports for c in row['clients']),
        'total_protocol_application_bytes': sum(c['application_envelope_and_tag_bytes'] for row in round_reports for c in row['clients']),
        'sqlite_integrity': sqlite_integrity, 'temporary_databases_removed': True,
        'private_keys_persisted_or_reported': False, 'audit': audit,
        'source_and_input_sha256': identities,
        'limitations': [
            'This is not a new predictive-quality seed/result and does not update any official TEST table.',
            'Two-client renormalized participation is a system diagnostic, not the original twenty-client experiment.',
            'Both arms deliberately use the existing durable protocol double arithmetic; no bitwise equivalence with original FP32/MPS main-experiment trajectories is claimed.',
            'The smoke has real learning deltas and new quantization each round, rather than replayed historical payloads.',
            'Reopen recovery discards Python objects and reopens SQLite; this run does not inject process kills or operating-system crashes.',
            'No real network, ledger, consensus, embedded model watermark, poisoning resistance or honest-training proof is established.',
            'HMAC keys, server, query/receipts and correct decoder remain trusted; record_id is an external upload identifier.',
            'Time is observed CPU diagnostic wall time, not an isolated deployment performance benchmark.'
        ],
        'elapsed_seconds': time.perf_counter() - started,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'checks_passed': len(checks), 'result': str(OUTPUT),
                      'gradient_evaluations': result['total_gradient_evaluations'], 'seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()

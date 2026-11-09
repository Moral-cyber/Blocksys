"""CPU synthetic fixtures only; never call strict real TRAIN preparation."""
from pathlib import Path
from types import SimpleNamespace
import contextlib
import hashlib
import io
import json
import struct
import tempfile
import numpy as np
import torch
import common as c
import train
import evaluate
import wire

torch.set_num_threads(1)
OUT = Path(tempfile.mkdtemp(prefix='strict_topk_reference_fixture_'))
sources = c.source_identities()
report = {'scope': 'CPU synthetic vectors/images and saved toy checkpoints only; no real-data preparation/training',
          'fixture_out': str(OUT), 'source_identities': sources, 'checks': {}}


def independent_parse(packet):
    # Separate struct-per-pair parser; never calls wire.decode.
    magic, version, n, k, width, reserved = struct.unpack_from('<8sIIII40s', packet)
    assert magic == b'TOPKEF01' and version == 1 and width == 8 and not any(reserved)
    assert len(packet) == 64 + k * 8
    pairs = [struct.unpack_from('<If', packet, 64 + 8 * i) for i in range(k)]
    assert [p[0] for p in pairs] == sorted(set(p[0] for p in pairs))
    assert all(0 <= i < n and np.isfinite(v) for i, v in pairs)
    decoded = np.zeros(n, dtype=np.float32)
    for i, value in pairs:
        decoded[i] = value
    return np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs], dtype=np.float32), torch.from_numpy(decoded)


# Fixed full-profile wire count, values and stable tie selection.
rng = np.random.default_rng(54008)
candidate = torch.from_numpy(rng.normal(size=c.PARAMETERS).astype(np.float32))
encoded = wire.encode(candidate)
ids, values, decoded = independent_parse(encoded.wire)
expected = sorted(sorted(range(c.PARAMETERS), key=lambda j: (-abs(float(candidate[j])), j))[:c.K])
assert ids.tolist() == expected
assert np.array_equal(values.view(np.uint32), candidate[encoded.indices].numpy().view(np.uint32))
assert torch.equal(decoded, encoded.sent) and len(encoded.wire) == 64 + 6843 * 8 == 54808
assert torch.equal(wire.decode(encoded.wire), decoded)
assert wire.encode(candidate).wire == encoded.wire
tie = torch.tensor([1., -1., 2., -2., 2., -2., 0., -0.], dtype=torch.float32)
tied = wire.encode(tie, 88)
assert independent_parse(tied.wire)[0].tolist() == [2, 3, 4]
zeros = torch.zeros(10, dtype=torch.float32)
zeros[1] = -0.
zero_wire = wire.encode(zeros, 88)
zids, zvalues, _ = independent_parse(zero_wire.wire)
assert zids.tolist() == [0, 1, 2] and zvalues.view(np.uint32)[1] == np.uint32(0x80000000)
report['checks']['full_wire'] = {'header_bytes': 64, 'pairs': c.K, 'pair_bytes': 8,
    'actual_wire_bytes': len(encoded.wire), 'indices_exact_global_topk': True,
    'values_FP32_bit_exact': True, 'ties_lower_index_first': True,
    'wire_indices_sorted': True, 'zero_pairs_and_negative_zero_preserved': True,
    'independent_struct_decoder_equal': True, 'deterministic_repeated_wire': True,
    'same_cap_uniform3_actual_bytes': len(c.base.codec.encode(candidate, c.CAP, mode='uniform3').wire)}


def overwrite(packet, offset, value):
    return packet[:offset] + value + packet[offset + len(value):]


small = wire.encode(torch.arange(1., 9.), 96).wire
bad_wires = {
    'short_header': small[:63], 'wrong_magic': overwrite(small, 0, b'BADMAGIC'),
    'wrong_version': overwrite(small, 8, struct.pack('<I', 2)),
    'zero_length': overwrite(small, 12, struct.pack('<I', 0)),
    'too_large_model': overwrite(small, 12, struct.pack('<I', c.PARAMETERS + 1)),
    'zero_k': overwrite(small, 16, struct.pack('<I', 0)),
    'wrong_pair_size': overwrite(small, 20, struct.pack('<I', 4)),
    'reserved_nonzero': overwrite(small, 24, b'\x01'),
    'duplicate_index': overwrite(small, 72, small[64:68]),
    'unsorted_indices': small[:64] + small[72:80] + small[64:72] + small[80:],
    'out_of_range': overwrite(small, 64, struct.pack('<I', 8)),
    'nan_value': overwrite(small, 68, struct.pack('<f', float('nan'))),
    'inf_value': overwrite(small, 68, struct.pack('<f', float('inf'))),
    'truncated_pair': small[:-1], 'trailing_bytes': small + b'\x00',
    'over_cap': encoded.wire + b'\x00', 'not_bytes': [0] * 64}
rejected_wire = []
for name, packet in bad_wires.items():
    try:
        wire.decode(packet)
    except ValueError:
        rejected_wire.append(name)
    else:
        raise AssertionError('accepted invalid wire: ' + name)
for name, value, budget in [('nan_input', torch.tensor([float('nan')]), c.CAP),
        ('wrong_dtype', torch.zeros(8, dtype=torch.float64), c.CAP),
        ('wrong_shape', torch.zeros(2, 4), c.CAP),
        ('empty', torch.zeros(0), c.CAP), ('bool_budget', torch.zeros(8), True),
        ('small_budget', torch.zeros(8), 71), ('over_cap_budget', torch.zeros(8), c.CAP + 1)]:
    try:
        wire.encode(value, budget)
    except ValueError:
        rejected_wire.append(name)
    else:
        raise AssertionError('accepted invalid encoder input: ' + name)
report['checks']['malformed_inputs'] = {'rejected': rejected_wire, 'all_rejected': True}

attempts = c.install_test_guard()
try:
    with Path('/tmp/topk_fixture_dummy_test_batch').open('rb'):
        raise AssertionError('audit guard failed')
except ValueError as error:
    assert 'TEST byte read forbidden' in str(error)
assert len(attempts) == 1
report['checks']['official_test_guard'] = {'dummy_path_rejected_before_open': True,
    'official_files_read': 0, 'actual_test_file_opened': False}

images = rng.integers(0, 256, size=(12, 32, 32, 3), dtype=np.uint8)
labels = np.array([0, 0, 1, 2, 2, 2, 4, 4, 0, 1, 2, 4], dtype=np.int64)
training_ids, holdout_ids = [[0, 1, 2], [3, 4, 5, 6, 7]], [[8, 9], [10, 11]]
manifest = {'fixture_only': True, 'partition_seed': 20261203,
    'train_indices': training_ids, 'holdout_indices': holdout_ids,
    'validation_indices': [8, 9, 10, 11],
    'client_label_counts': [np.bincount(labels[ids], minlength=10).tolist() for ids in training_ids]}


class LoggedDataset(c.partitions.IndexedCifar10):
    def __init__(self, *arguments):
        super().__init__(*arguments)
        self.order = []

    def __getitem__(self, index):
        self.order.append(int(self.indices[index]))
        return super().__getitem__(index)


datasets = [LoggedDataset(images, labels, ids) for ids in training_ids]
validation = c.partitions.IndexedCifar10(images, labels, [8, 9, 10, 11])
args = SimpleNamespace(out=OUT / 'complete', seed=20261211, partition_seed=20261203, rounds=2, backend='cpu')
original_train, original_encode = c.balanced.train, wire.encode
captured, packets, current = {}, {}, {}


def observed_train(model, dataset, seed, backend, loss):
    before = c.runner.vectorize(model).clone()
    dataset.order.clear()
    delta, counters = original_train(model, dataset, seed, backend, loss)
    assert torch.equal(before, c.runner.vectorize(model))
    current['clients'].append({'anchor': before, 'delta': delta.clone(), 'seed': seed, 'order': dataset.order.copy()})
    return delta, counters


def observed_encode(value, budget):
    packet = original_encode(value, budget)
    current['packets'].append({'input': value.clone(), 'wire': packet.wire})
    return packet


c.balanced.train, wire.encode = observed_train, observed_encode
try:
    for method in c.METHODS:
        current['clients'], current['packets'] = [], []
        with contextlib.redirect_stdout(io.StringIO()):
            train.run(args, method, datasets, validation, manifest, {})
        captured[method], packets[method] = current['clients'], current['packets']
finally:
    c.balanced.train, wire.encode = original_train, original_encode
assert [(r['seed'], r['order']) for r in captured['ce_topk']] == [(r['seed'], r['order']) for r in captured['fedlc_topk']]
assert torch.equal(captured['ce_topk'][0]['anchor'], captured['fedlc_topk'][0]['anchor'])
tiny_results = []
for method in c.METHODS:
    folder = args.out / method
    config = json.loads((folder / 'config.json').read_text())
    checkpoint = torch.load(folder / 'checkpoint.pt', weights_only=True)
    model = c.validate_checkpoint(checkpoint, config)
    assert config['weights'] == [.375, .625]
    residuals = [torch.zeros(c.PARAMETERS), torch.zeros(c.PARAMETERS)]
    for round_id in range(2):
        aggregate = torch.zeros(c.PARAMETERS)
        for client, weight in enumerate(config['weights']):
            index = 2 * round_id + client
            local, packet = captured[method][index], packets[method][index]
            assert torch.equal(local['delta'] + residuals[client], packet['input'])
            ids, values, q = independent_parse(packet['wire'])
            assert len(packet['wire']) == c.CAP and len(ids) == c.K
            residuals[client] = packet['input'] - q
            assert torch.equal(q + residuals[client], packet['input'])
            aggregate.add_(q, alpha=weight)
        assert torch.equal(captured[method][2 * round_id]['anchor'], captured[method][2 * round_id + 1]['anchor'])
        predicted = captured[method][2 * round_id]['anchor'] + aggregate
        if round_id == 0:
            assert torch.equal(predicted, captured[method][2]['anchor'])
    assert torch.equal(c.runner.vectorize(model), predicted)
    assert all(torch.equal(a, b) for a, b in zip(residuals, checkpoint['residuals']))
    assert (folder / 'first_payload.bin').read_bytes() == packets[method][0]['wire']
    assert all(row['actual_gradient_evaluations'] == 2 and row['upload_bytes'] == 2 * c.CAP for row in checkpoint['rows'])
    tiny_results.append({'method': method, 'rounds': 2, 'actual_gradients': 4,
        'wire_per_client': c.CAP, 'decoded_aggregation_EF_bitwise_equal': True})
report['checks']['tiny_training'] = {'methods': tiny_results, 'same_seed_batch_order_initialization': True,
    'fixed_global_anchor_between_clients': True}

# Interrupt after complete round1 and resume the same signed round2 protocol.
resume_args = SimpleNamespace(**{**vars(args), 'out': OUT / 'interrupted'})
calls = []


class Interrupted(Exception):
    pass


def interrupt(*arguments):
    calls.append(True)
    if len(calls) == 3:
        raise Interrupted()
    return original_train(*arguments)


c.balanced.train = interrupt
try:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            train.run(resume_args, 'fedlc_topk', datasets, validation, manifest, {})
    except Interrupted:
        pass
    else:
        raise AssertionError('expected synthetic interruption')
finally:
    c.balanced.train = original_train
with contextlib.redirect_stdout(io.StringIO()):
    train.run(resume_args, 'fedlc_topk', datasets, validation, manifest, {})
full_path = args.out / 'fedlc_topk/checkpoint.pt'
full_bytes = full_path.read_bytes()
full = torch.load(io.BytesIO(full_bytes), weights_only=True)
resumed = torch.load(resume_args.out / 'fedlc_topk/checkpoint.pt', weights_only=True)
assert all(torch.equal(v, resumed['model'][name]) for name, v in full['model'].items())
assert all(torch.equal(a, b) for a, b in zip(full['residuals'], resumed['residuals']))
config_path = full_path.parent / 'config.json'
config_bytes = config_path.read_bytes()
negative, training_attempts = {}, []


def forbid_train(*arguments):
    training_attempts.append(True)
    raise AssertionError('rejected/complete resume must not train')


def expect_reject(name, selected=args):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            train.run(selected, 'fedlc_topk', datasets, validation, manifest, {})
    except ValueError:
        negative[name] = True
    else:
        raise AssertionError('invalid resume accepted: ' + name)


c.balanced.train = forbid_train
try:
    with contextlib.redirect_stdout(io.StringIO()):
        train.run(args, 'fedlc_topk', datasets, validation, manifest, {})
    for key, value in [('seed', 20261212), ('partition_seed', 20261204), ('rounds', 3), ('backend', 'mps')]:
        expect_reject(key + '_mismatch', SimpleNamespace(**{**vars(args), key: value}))
    original_sources = c.source_identities
    c.source_identities = lambda: {**sources, 'fixture_changed_source': '0' * 64}
    try:
        expect_reject('source_identity_mismatch')
    finally:
        c.source_identities = original_sources
    broken = json.loads(config_bytes)
    broken['weights'] = [.5, .5]
    config_path.write_text(json.dumps(broken))
    expect_reject('config_self_signature')
    config_path.write_bytes(config_bytes)
    for name, mutation in {'checkpoint_signature': lambda x: x.update(signature='wrong'),
            'checkpoint_version': lambda x: x.update(version=999),
            'checkpoint_full_config': lambda x: x['config'].update(seed=0),
            'checkpoint_EF_shape': lambda x: x.update(residuals=[torch.zeros(3)] * 2),
            'checkpoint_EF_nonfinite': lambda x: x['residuals'][0].fill_(float('nan'))}.items():
        broken = torch.load(io.BytesIO(full_bytes), weights_only=True)
        mutation(broken)
        torch.save(broken, full_path)
        expect_reject(name)
        full_path.write_bytes(full_bytes)
finally:
    c.balanced.train = original_train
    full_path.write_bytes(full_bytes)
    config_path.write_bytes(config_bytes)
assert not training_attempts
report['checks']['resume'] = {'round1_resume_final_model_EF_bitwise_equal': True,
    'complete_resume_no_training': True, 'negative_rejections': negative}

# Frozen final-only local evaluation; synthetic arrays only, no reference files.
incomplete = torch.load(io.BytesIO(full_bytes), weights_only=True)
incomplete['rows'] = incomplete['rows'][:1]
torch.save(incomplete, full_path)
try:
    try:
        evaluate.run(args.out, OUT / 'must_not_create', images, labels, manifest, {})
    except ValueError as error:
        assert 'final complete' in str(error)
    else:
        raise AssertionError('accepted incomplete final evaluation')
finally:
    full_path.write_bytes(full_bytes)
assert not (OUT / 'must_not_create').exists()
with contextlib.redirect_stdout(io.StringIO()):
    result = evaluate.run(args.out, OUT / 'evaluation', images, labels, manifest, {})
assert result['decision'] == 'REFERENCE_ONLY' and result['gates'] is None
assert result['bias_total_train_image_forwards'] == 16 and result['bias_total_cnn_backward'] == 0
assert result['matched_head_total_gradient_evaluations'] == 4
for method in c.METHODS:
    for i, client in enumerate(result['results'][method]['clients']):
        with np.load(OUT / 'evaluation' / f'{method}_client{i}.npz', allow_pickle=False) as arrays:
            assert np.array_equal(arrays['train_ids'], training_ids[i])
            assert np.array_equal(arrays['holdout_ids'], holdout_ids[i])
            assert not set(map(int, arrays['train_ids'])) & set(map(int, arrays['holdout_ids']))
            bias, diagnostic = c.bias_solver.fit(arrays['train_logits'], arrays['train_labels'])
            assert np.array_equal(bias, arrays['fitted_bias']) and bias.nbytes == 40
            logits, y = arrays['raw'].astype(np.float64) + bias, arrays['labels']
            loss = np.logaddexp.reduce(logits, axis=1) - logits[np.arange(len(y)), y]
            assert abs(loss.mean() - client['metrics']['prioranchor_bias']['loss']) < 1e-12
            assert float(np.mean(logits.argmax(1) == y)) == client['metrics']['prioranchor_bias']['accuracy']
report['checks']['tiny_evaluation'] = {'only_final_complete_accepted': True,
    'frozen_local_TRAIN_only_prior_bias_head': True, 'bias_train_forwards': 16,
    'bias_CNN_backward': 0, 'matched_head_gradients': 4,
    'decision': 'REFERENCE_ONLY', 'gates': None, 'new_candidate': False}
c.verify_identities(sources)
assert len(attempts) == 1
report.update({'status': 'PASS', 'source_identities_unchanged': True,
    'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'formal_ML_training_started': False, 'review': 'REVIEW_UNAVAILABLE'})
(c.HERE / 'LOCAL_FIXTURES.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': 'PASS', 'fixture_out': str(OUT), 'checks': list(report['checks']),
                  'formal_ML_training_started': False}, ensure_ascii=False, indent=2))

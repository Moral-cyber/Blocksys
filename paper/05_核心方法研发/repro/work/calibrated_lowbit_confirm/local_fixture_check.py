"""Synthetic CPU fixtures; never launch the real-data confirmation entries."""
from pathlib import Path
from types import SimpleNamespace
import contextlib
import hashlib
import io
import json
import tempfile

import numpy as np
import torch
import common as c
import train
import evaluate

torch.set_num_threads(1)
OUT = Path(tempfile.mkdtemp(prefix='calibrated_lowbit_confirm_fixture_'))
sources = c.source_identities()
report = {'scope': 'synthetic CPU single-thread fixtures only; no official data or real training entry',
          'fixture_out': str(OUT), 'source_identities': sources, 'checks': {}}
attempts = c.install_test_guard()
try:
    with Path('/tmp/fixture_forbidden_test_batch').open('rb'):
        raise AssertionError('audit hook failed')
except ValueError as error:
    assert 'TEST byte read forbidden' in str(error)
assert len(attempts) == 1
report['checks']['test_byte_guard'] = {'intentional_dummy_path_rejected': True,
    'official_files_read': 0, 'actual_test_file_opened': False}

# Same real partition function on fabricated balanced labels; no image loading.
synthetic_labels = np.repeat(np.arange(10), 5000)
first = c.partitions.partition_indices(synthetic_labels, alpha=.1, split_seed=20261203,
                                     mixing=0., validation_size=2000)
second = c.partitions.partition_indices(synthetic_labels, alpha=.1, split_seed=20261203,
                                      mixing=0., validation_size=2000)
assert first[3] == second[3]
assert sum(map(len, first[0])) + sum(map(len, first[1])) == 50000
assert not np.intersect1d(np.concatenate(first[0]), np.concatenate(first[1])).size
report['checks']['synthetic_partition'] = {'seed': 20261203, 'deterministic': True,
    'train_actual': sum(map(len, first[0])), 'holdout_actual': sum(map(len, first[1])),
    'validation_actual': len(first[2]), 'unique_disjoint_full_coverage': True,
    'same_frozen_partition_function_and_normalization': True}

rng = np.random.default_rng(1203)
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


trainsets = [LoggedDataset(images, labels, ids) for ids in training_ids]
validation = c.partitions.IndexedCifar10(images, labels, [8, 9, 10, 11])
args = SimpleNamespace(out=OUT / 'complete', seed=20261211, partition_seed=20261203,
                       rounds=2, backend='cpu')
original_train, original_encode = c.balanced.train, c.codec.encode
captured, encoded, current = {}, {}, {}


def observed_train(model, dataset, seed, backend, loss):
    before = c.runner.vectorize(model).clone()
    dataset.order.clear()
    result = original_train(model, dataset, seed, backend, loss)
    assert torch.equal(before, c.runner.vectorize(model))
    current['clients'].append({'seed': seed, 'order': dataset.order.copy(),
        'anchor': before, 'delta': result[0].clone()})
    return result


def observed_encode(value, *arguments, **keywords):
    result = original_encode(value, *arguments, **keywords)
    current['wire'].append({'input': value.clone(), 'wire': result.wire})
    return result


c.balanced.train, c.codec.encode = observed_train, observed_encode
try:
    for method in c.METHODS:
        current['clients'], current['wire'] = [], []
        with contextlib.redirect_stdout(io.StringIO()):
            train.run(args, method, trainsets, validation, manifest, {})
        captured[method], encoded[method] = current['clients'], current['wire']
finally:
    c.balanced.train, c.codec.encode = original_train, original_encode
reference = captured['ce_uniform3']
for method, rows in captured.items():
    assert [(row['seed'], row['order']) for row in rows] == [(row['seed'], row['order']) for row in reference]
    assert torch.equal(rows[0]['anchor'], reference[0]['anchor'])
    assert torch.equal(rows[0]['anchor'], rows[1]['anchor']) and torch.equal(rows[2]['anchor'], rows[3]['anchor'])
assert all(torch.equal(left['delta'], right['delta']) for left, right in
           zip(captured['fedlc_uniform3'][:2], captured['fedlc_dense'][:2]))
training_rows = []
for method in c.METHODS:
    folder = args.out / method
    config = json.loads((folder / 'config.json').read_text())
    checkpoint = torch.load(folder / 'checkpoint.pt', weights_only=False)
    loaded = c.validate_checkpoint(checkpoint, config)
    unsigned = {key: value for key, value in config.items() if key != 'signature'}
    assert c.signature(unsigned) == config['signature']
    assert config['weights'] == [.375, .625]
    residuals, ef_error = [torch.zeros(c.PARAMETERS), torch.zeros(c.PARAMETERS)], 0.
    for round_id in range(2):
        aggregate = torch.zeros(c.PARAMETERS)
        for client, weight in enumerate(config['weights']):
            index = round_id * 2 + client
            local = captured[method][index]
            if method.endswith('dense'):
                q = local['delta']
            else:
                packet = encoded[method][index]
                assert torch.equal(local['delta'] + residuals[client], packet['input'])
                assert len(packet['wire']) == 54067 <= c.CAP
                q = c.codec.decode(packet['wire'])
                residuals[client] = packet['input'] - q
                ef_error = max(ef_error, float((q + residuals[client] - packet['input']).abs().max()))
            aggregate.add_(q, alpha=weight)
        predicted = captured[method][2 * round_id]['anchor'] + aggregate
    assert torch.equal(c.runner.vectorize(loaded), predicted)
    assert all(torch.equal(left, right) for left, right in zip(residuals, checkpoint['residuals']))
    assert all(row['actual_gradient_evaluations'] == 2 for row in checkpoint['rows'])
    if not method.endswith('dense'):
        assert (folder / 'first_payload.bin').read_bytes() == encoded[method][0]['wire']
    training_rows.append({'method': method, 'rounds': 2, 'signature': config['signature'],
        'actual_gradient_evaluations': 4, 'wire_per_client': 547560 if method.endswith('dense') else 54067,
        'ef_max_error': ef_error, 'decoder_aggregation_and_residual_state_match': True})
report['checks']['tiny_training'] = {'methods': training_rows, 'same_seed_batch_initial_anchor': True,
    'fedlc_quant_dense_first_local_deltas_bitwise_equal': True,
    'new_real_data_training_started': False}

# Real checkpoint interruption and deterministic continuation.
resume_args = SimpleNamespace(**{**vars(args), 'out': OUT / 'interrupted'})
calls = []


class Interrupted(Exception):
    pass


def interrupted_train(*arguments):
    calls.append(True)
    if len(calls) == 3:
        raise Interrupted()
    return original_train(*arguments)


c.balanced.train = interrupted_train
try:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            train.run(resume_args, 'fedlc_uniform3', trainsets, validation, manifest, {})
    except Interrupted:
        pass
    else:
        raise AssertionError('expected deliberate interruption')
finally:
    c.balanced.train = original_train
with contextlib.redirect_stdout(io.StringIO()):
    train.run(resume_args, 'fedlc_uniform3', trainsets, validation, manifest, {})
resumed = torch.load(resume_args.out / 'fedlc_uniform3/checkpoint.pt', weights_only=False)
complete_path = args.out / 'fedlc_uniform3/checkpoint.pt'
complete = torch.load(complete_path, weights_only=False)
assert all(torch.equal(value, complete['model'][name]) for name, value in resumed['model'].items())
assert all(torch.equal(left, right) for left, right in zip(resumed['residuals'], complete['residuals']))

negative, attempts_train = {}, []
checkpoint_bytes = complete_path.read_bytes()
config_path = complete_path.parent / 'config.json'
config_bytes = config_path.read_bytes()


def forbid_train(*arguments):
    attempts_train.append(True)
    raise AssertionError('unexpected training during rejection')


def rejected(name, selected=args):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            train.run(selected, 'fedlc_uniform3', trainsets, validation, manifest, {})
    except ValueError:
        negative[name] = True
    else:
        raise AssertionError('accepted invalid checkpoint/config: ' + name)


c.balanced.train = forbid_train
try:
    with contextlib.redirect_stdout(io.StringIO()):
        train.run(args, 'fedlc_uniform3', trainsets, validation, manifest, {})
    for key, value in [('seed', args.seed + 1), ('partition_seed', args.partition_seed + 1),
                       ('rounds', 3), ('backend', 'mps')]:
        rejected(key + '_mismatch', SimpleNamespace(**{**vars(args), key: value}))
    original_sources = c.source_identities
    c.source_identities = lambda: {**sources, 'fixture_changed_source': '0' * 64}
    try:
        rejected('source_identity_mismatch')
    finally:
        c.source_identities = original_sources
    broken_config = json.loads(config_bytes)
    broken_config['weights'] = [.5, .5]
    config_path.write_text(json.dumps(broken_config))
    rejected('config_self_signature_mismatch')
    config_path.write_bytes(config_bytes)
    for name, mutation in {
            'checkpoint_signature': lambda value: value.update(signature='wrong'),
            'checkpoint_version': lambda value: value.update(version=999),
            'checkpoint_full_config': lambda value: value['config'].update(seed=0),
            'checkpoint_ef_shape': lambda value: value.update(residuals=[torch.zeros(3)] * 2),
            'checkpoint_ef_nonfinite': lambda value: value['residuals'][0].fill_(float('nan'))}.items():
        broken = torch.load(io.BytesIO(checkpoint_bytes), weights_only=False)
        mutation(broken)
        torch.save(broken, complete_path)
        rejected(name)
        complete_path.write_bytes(checkpoint_bytes)
finally:
    c.balanced.train = original_train
    complete_path.write_bytes(checkpoint_bytes)
    config_path.write_bytes(config_bytes)
assert not attempts_train
report['checks']['resume'] = {'round1_interrupted_resume_model_EF_bitwise_equal': True,
    'completed_resume_without_training': True, 'negative_rejections': negative}

# Evaluation must refuse an intermediate checkpoint and mismatched data
# before it creates output or performs any forward/adaptation.
evaluation_rejections = {}
try:
    incomplete = torch.load(io.BytesIO(checkpoint_bytes), weights_only=False)
    incomplete['rows'] = incomplete['rows'][:1]
    torch.save(incomplete, complete_path)
    try:
        evaluate.run(args.out, OUT / 'must_not_create_incomplete_eval', images, labels, manifest, {})
    except ValueError as error:
        assert 'final complete' in str(error)
        evaluation_rejections['incomplete_checkpoint'] = True
    else:
        raise AssertionError('accepted non-final checkpoint')
finally:
    complete_path.write_bytes(checkpoint_bytes)
try:
    evaluate.run(args.out, OUT / 'must_not_create_mismatch_eval', images, labels,
                 {**manifest, 'partition_seed': 0}, {})
except ValueError as error:
    assert 'source/data differs' in str(error)
    evaluation_rejections['different_partition'] = True
else:
    raise AssertionError('accepted mismatched evaluation partition')
assert not (OUT / 'must_not_create_incomplete_eval').exists()
assert not (OUT / 'must_not_create_mismatch_eval').exists()
report['checks']['final_only_evaluation_rejections'] = evaluation_rejections

# Full evaluation on tiny synthetic data, including TRAIN-only frozen solver
# and the matched head strong baseline; never touch official dataset files.
with contextlib.redirect_stdout(io.StringIO()):
    evaluation = evaluate.run(args.out, OUT / 'evaluation', images, labels, manifest, {})
evaluation_rows = []
for method, result in evaluation['results'].items():
    for client, value in enumerate(result['clients']):
        with np.load(OUT / 'evaluation' / f'{method}_client{client}.npz') as arrays:
            assert np.array_equal(arrays['train_ids'], training_ids[client])
            assert np.array_equal(arrays['holdout_ids'], holdout_ids[client])
            assert not np.intersect1d(arrays['train_ids'], arrays['holdout_ids']).size
            bias, diagnostic = c.bias_solver.fit(arrays['train_logits'], arrays['train_labels'])
            assert np.array_equal(bias, arrays['fitted_bias']) and bias.nbytes == 40
            metrics = c.bias_solver.stats(arrays['raw'].astype(np.float64) + bias, arrays['labels'])
            assert metrics == value['metrics']['prioranchor_bias']
        assert value['bias_new_cnn_backward'] == 0
        assert value['matched_head_training']['gradient_evaluations'] == 1
    evaluation_rows.append({'method': method, 'all_variants': list(result['summaries']),
        'holdout_fitting_absent_and_bias_reproduces': True, 'matched_head_body_frozen': True})
assert evaluation['bias_total_train_image_forwards'] == 24
assert evaluation['bias_total_cnn_backward'] == 0
assert evaluation['matched_head_total_gradient_evaluations'] == 6
assert 'zero-anchor' in evaluation['must_next_if_passed'][0]
report['checks']['tiny_evaluation'] = {'methods': evaluation_rows,
    'bias_train_image_forwards': 24, 'bias_cnn_backward': 0,
    'matched_head_gradient_evaluations': 6, 'state_bytes_per_client': 40,
    'gates_recorded_without_hyperparameter_selection': evaluation['gates'],
    'fixture_decision_not_a_scientific_result': evaluation['decision'],
    'must_next_zero_anchor_vector_BCTS_preserved': True}
c.verify_identities(sources)
assert len(attempts) == 1  # only intentional dummy path, no real test data
report.update({'status': 'PASS', 'source_identities_unchanged': True,
    'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'formal_ML_training_started': False})
(c.HERE / 'LOCAL_FIXTURES.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'status': 'PASS', 'fixture_out': str(OUT), 'checks': list(report['checks']),
    'formal_ML_training_started': False}, indent=2))

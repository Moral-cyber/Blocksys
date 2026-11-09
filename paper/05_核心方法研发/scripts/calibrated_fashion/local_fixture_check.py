"""CPU toy routing plus read-only real Fashion TRAIN partition fixture."""
from pathlib import Path
from types import SimpleNamespace
import contextlib
import hashlib
import io
import json
import sys
import tempfile

import numpy as np
import torch
import adapter as a

torch.set_num_threads(1)
OUT = Path(tempfile.mkdtemp(prefix='calibrated_fashion_fixture_'))
before = a.source_identities()
attempts = a.install_test_guard()
opened_gzip = []


def audit(event, arguments):
    if event == 'open' and isinstance(arguments[0], (str, bytes)):
        name = str(arguments[0])
        if name.endswith('.gz'):
            opened_gzip.append(name)


sys.addaudithook(audit)
for name in ('test_batch', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'):
    try:
        with Path('/tmp/intentional_forbidden_' + name).open('rb'):
            raise AssertionError('guard failed')
    except ValueError as error:
        assert 'official TEST byte read forbidden' in str(error)
assert len(attempts) == 3
report = {'scope': 'CPU toy plus strict real TRAIN parsing/partition only; no real-data optimization',
    'fixture_out': str(OUT), 'source_identities': before, 'checks': {}}

# This reads and hashes only two real TRAIN gzip files. No CNN inference or
# optimization is performed on any real image in this fixture.
sets, validation, real_manifest, data_ids, real_images, real_labels = a.prepare(20261207)
repeated = a.fashion.partition_indices(real_labels, alpha=.1, split_seed=20261207,
                                      mixing=0., validation_size=2000)
assert repeated[3]['train_indices'] == real_manifest['train_indices']
assert repeated[3]['holdout_indices'] == real_manifest['holdout_indices']
assert real_images.shape == (60000, 28, 28) and real_labels.shape == (60000,)
assert hashlib.sha256(real_labels.astype('<i8').tobytes()).hexdigest() == a.LABELS_SHA256
training_ids = np.concatenate(real_manifest['train_indices'])
holdout_ids = np.concatenate(real_manifest['holdout_indices'])
assert len(training_ids) + len(holdout_ids) == 60000 and not np.intersect1d(training_ids, holdout_ids).size
assert np.array_equal(np.sort(np.concatenate((training_ids, holdout_ids))), np.arange(60000))
sample, target = a.fashion.FashionIndexDataset(real_images, real_labels, [0])[0]
expected = np.pad(real_images[0], 2).astype(np.float32) / 255.
expected = (expected - .5) / .5
assert sample.shape == (3, 32, 32)
assert np.array_equal(sample[0].numpy(), expected)
assert torch.equal(sample[0], sample[1]) and torch.equal(sample[0], sample[2])
assert torch.all(sample[:, :2, :] == -1.) and torch.all(sample[:, :, :2] == -1.)
report['checks']['real_TRAIN_readonly_partition'] = {
    'source_total': 60000, 'train_actual': len(training_ids), 'holdout_actual': len(holdout_ids),
    'globaldev': len(validation), 'partition_seed': 20261207,
    'labels_sha256': a.LABELS_SHA256, 'data_identities': data_ids,
    'official_md5_sha_verified': True, 'deterministic_membership': True,
    'full_coverage_disjoint': True, 'normalization_manual_bitwise': True,
    'pad_zero_maps_minus_one_repeat3': True, 'real_cnn_forwards': 0, 'real_backwards': 0}
a.common.atomic_json(a.HERE / 'TRAIN_PARTITION_FIXTURE.json', {
    'status': 'PASS', 'manifest': real_manifest, 'data_identities': data_ids,
    'source_identities': before, 'real_cnn_forwards': 0, 'real_backwards': 0})
del real_images, real_labels, sets, validation

rng = np.random.default_rng(20261207)
images = rng.integers(0, 256, (12, 28, 28), dtype=np.uint8)
labels = np.array([0, 0, 1, 2, 2, 2, 4, 4, 0, 1, 2, 4], dtype=np.int64)
trainids, holdoutids = [[0, 1, 2], [3, 4, 5, 6, 7]], [[8, 9], [10, 11]]
manifest = {'fixture_only': True, 'task': 'FashionMNIST', 'partition_seed': 20261207,
    'train_indices': trainids, 'holdout_indices': holdoutids,
    'validation_indices': [8, 9, 10, 11], 'route_overrides': a.ROUTES,
    'client_label_counts': [np.bincount(labels[ids], minlength=10).tolist() for ids in trainids]}


class Logged(a.fashion.FashionIndexDataset):
    def __init__(self, *args):
        super().__init__(*args)
        self.order = []

    def __getitem__(self, index):
        self.order.append(int(self.indices[index]))
        return super().__getitem__(index)


trainsets = [Logged(images, labels, ids) for ids in trainids]
dev = a.fashion.FashionIndexDataset(images, labels, [8, 9, 10, 11])
args = SimpleNamespace(out=OUT / 'toy_training', seed=20261221, partition_seed=20261207,
                       rounds=2, backend='cpu')
old_train, captured, current = a.common.balanced.train, {}, {}


def observed_train(model, dataset, seed, backend, loss):
    anchor = a.common.runner.vectorize(model).clone()
    dataset.order.clear()
    result = old_train(model, dataset, seed, backend, loss)
    assert torch.equal(anchor, a.common.runner.vectorize(model))
    current['records'].append({'seed': seed, 'order': dataset.order.copy(),
        'anchor': anchor, 'delta': result[0].clone()})
    return result


a.common.balanced.train = observed_train
try:
    for method in a.common.METHODS:
        current['records'] = []
        with contextlib.redirect_stdout(io.StringIO()):
            a.training.run(args, method, trainsets, dev, manifest, {})
        captured[method] = current['records']
finally:
    a.common.balanced.train = old_train
reference = captured['ce_uniform3']
training_rows = []
for method in a.common.METHODS:
    records = captured[method]
    assert [(record['seed'], record['order']) for record in records] == [(record['seed'], record['order']) for record in reference]
    assert torch.equal(records[0]['anchor'], reference[0]['anchor'])
    folder = args.out / method
    config = json.loads((folder / 'config.json').read_text())
    checkpoint = torch.load(folder / 'checkpoint.pt', weights_only=False)
    a.common.validate_checkpoint(checkpoint, config)
    assert config['runtime']['dataset_task'] == 'FashionMNIST'
    assert config['runtime']['route_overrides'] == a.ROUTES
    assert config['runtime']['wrapper_source_identities'] == before
    assert config['source_identities'] == before
    assert all(row['actual_gradient_evaluations'] == 2 for row in checkpoint['rows'])
    if not method.endswith('dense'):
        wire = (folder / 'first_payload.bin').read_bytes()
        assert len(wire) == 54067 <= 54808 and a.common.codec.decode(wire).shape == (136874,)
    training_rows.append({'method': method, 'tiny_rounds': 2,
        'signature': config['signature'], 'source_route_runtime_signed': True,
        'wire_or_dense_bytes': 547560 if method.endswith('dense') else 54067})
assert all(torch.equal(left['delta'], right['delta']) for left, right in
           zip(captured['fedlc_uniform3'][:2], captured['fedlc_dense'][:2]))
report['checks']['toy_training'] = {'methods': training_rows,
    'same_seed_batch_anchor': True, 'fedlc_quant_dense_initial_delta_bitwise_equal': True}

# Same frozen source functions also exercise strict completed resume and the
# source/task boundary. No physical source file is changed for this test.
forbidden_calls = []


def forbidden_train(*arguments):
    forbidden_calls.append(True)
    raise AssertionError('unexpected training in completed resume')


a.common.balanced.train = forbidden_train
try:
    with contextlib.redirect_stdout(io.StringIO()):
        a.training.run(args, 'fedlc_uniform3', trainsets, dev, manifest, {})
    try:
        a.training.run(args, 'fedlc_uniform3', trainsets, dev, {**manifest, 'task': 'CIFAR10'}, {})
    except ValueError as error:
        assert 'incompatible output' in str(error)
    else:
        raise AssertionError('CIFAR task signature was accepted')
finally:
    a.common.balanced.train = old_train
assert not forbidden_calls
with contextlib.redirect_stdout(io.StringIO()):
    evaluated = a.evaluation.run(args.out, OUT / 'toy_evaluation', images, labels, manifest, {})
assert evaluated['bias_total_train_image_forwards'] == 24
assert evaluated['bias_total_cnn_backward'] == 0
assert evaluated['matched_head_total_gradient_evaluations'] == 6
for method, entry in evaluated['results'].items():
    assert entry['frozen_global_model_unchanged']
    for client, record in enumerate(entry['clients']):
        with np.load(OUT / 'toy_evaluation' / f'{method}_client{client}.npz') as saved:
            bias, _ = a.common.bias_solver.fit(saved['train_logits'], saved['train_labels'])
            assert np.array_equal(bias, saved['fitted_bias'])
            assert np.array_equal(saved['train_ids'], trainids[client])
            assert np.array_equal(saved['holdout_ids'], holdoutids[client])
        assert record['bias_local_state_bytes'] == 40
report['checks']['toy_evaluation_resume'] = {'completed_resume_no_training': True,
    'different_task_resume_rejected': True, 'final_fashion_dataset_proxy': True,
    'three_methods_all_variants': True, 'TRAIN_only_bias_recomputed': True,
    'bias_forwards': 24, 'bias_backwards': 0, 'head_gradient_evaluations': 6,
    'synthetic_gates_not_a_scientific_result': evaluated['gates']}
assert a.source_identities() == before
assert len(attempts) == 3
assert {Path(path).name for path in opened_gzip} <= set(a.TRAIN_FILES)
a.common.verify_identities(data_ids)
report.update({'status': 'PASS', 'sources_before_after_equal': True,
    'actual_gzip_paths_opened': sorted(set(opened_gzip)),
    'intentional_guard_dummy_rejections': 3, 'real_test_byte_read_attempts': 0,
    'formal_ML_training_started': False,
    'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
a.common.atomic_json(a.HERE / 'LOCAL_FIXTURES.json', report)
print(json.dumps({'status': 'PASS', 'checks': list(report['checks']),
    'real_partition_train': report['checks']['real_TRAIN_readonly_partition']['train_actual'],
    'real_partition_holdout': report['checks']['real_TRAIN_readonly_partition']['holdout_actual'],
    'real_test_byte_read_attempts': 0, 'formal_ML_training_started': False,
    'fixture_out': str(OUT)}, indent=2))

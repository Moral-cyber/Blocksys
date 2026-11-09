"""Independent tiny CPU fixtures; no official data or development-run access."""
from pathlib import Path
from types import SimpleNamespace
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
protected = (HERE / 'run.py', HERE / 'PLAN.md',
             HERE.parent / 'mixed_precision/run.py', HERE.parent / 'mixed_precision/codec.py')
before = {str(path.relative_to(ROOT)): sha(path) for path in protected}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m = load('_balanced_quant_independent_fixture', HERE / 'run.py')
old = load('_frozen_mixed_precision_loss_fixture', HERE.parent / 'mixed_precision/run.py')
torch.set_num_threads(1)
OUT = Path(tempfile.mkdtemp(prefix='balanced_quant_local_fixture_'))
report = {'scope': 'CPU single-thread synthetic implementation fixtures only',
          'fixture_out': str(OUT), 'source_identities': before,
          'official_dataset_open_attempts': [], 'checks': {}}


def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)) and '/dataset/' in str(args[0]):
        report['official_dataset_open_attempts'].append(str(args[0]))
        raise AssertionError('official data access forbidden in synthetic fixture')


sys.addaudithook(audit)


def maximum(value):
    return float(value.abs().max()) if value.numel() else 0.


def ce_gradient(logits, labels, shift):
    value = logits.clone().requires_grad_()
    loss = nn.functional.cross_entropy(value + shift, labels)
    gradient, = torch.autograd.grad(loss, value)
    return loss.detach(), gradient.detach()


generator = torch.Generator().manual_seed(20261201)
labels = torch.randint(0, 10, (11,), generator=generator)
logits64 = torch.randn(11, 10, generator=generator, dtype=torch.float64)
counts = np.array([0, 1, 2, 10, 99, 4, 0, 30, 5, 0], dtype=np.int64)
equivalences = []
for dtype, loss_tol, grad_tol in ((torch.float64, 1e-12, 1e-12), (torch.float32, 2e-6, 2e-7)):
    logits = logits64.to(dtype)
    unnormalized = torch.tensor(m.shift_for(counts, 'balanced'), dtype=dtype)
    normalized = torch.tensor(np.log((counts + 1) / (counts.sum() + 10)), dtype=dtype)
    left, left_gradient = ce_gradient(logits, labels, unnormalized)
    right, right_gradient = ce_gradient(logits, labels, normalized)
    assert abs(float(left - right)) < loss_tol
    assert maximum(left_gradient - right_gradient) < grad_tol
    equivalences.append({'dtype': str(dtype), 'ce_error': abs(float(left - right)),
                         'logit_gradient_error': maximum(left_gradient - right_gradient)})
uniform_rows = []
for count in (0, 1, 7, 1000000):
    uniform = np.repeat(count, 10)
    for dtype in (torch.float64, torch.float32):
        plain, plain_grad = ce_gradient(logits64.to(dtype), labels, torch.zeros(10, dtype=dtype))
        balanced, balanced_grad = ce_gradient(logits64.to(dtype), labels,
            torch.tensor(m.shift_for(uniform, 'balanced'), dtype=dtype))
        assert abs(float(plain - balanced)) < 2e-6
        assert maximum(plain_grad - balanced_grad) < 2e-7
        uniform_rows.append({'count_per_class': count, 'dtype': str(dtype),
            'ce_error': abs(float(plain - balanced)), 'gradient_error': maximum(plain_grad - balanced_grad)})
boundary_rows = []
for name, value in [('all_zero', np.zeros(10, dtype=np.int64)),
                    ('single_class', np.array([0, 0, 0, 31, 0, 0, 0, 0, 0, 0])),
                    ('missing_classes', counts)]:
    for loss_name in ('ce', 'fedlc', 'balanced'):
        shift = torch.tensor(m.shift_for(value, loss_name), dtype=torch.float32)
        loss, gradient = ce_gradient(logits64.float() * 100, labels, shift)
        assert torch.isfinite(shift).all() and torch.isfinite(loss) and torch.isfinite(gradient).all()
        boundary_rows.append({'case': name, 'loss': loss_name, 'finite_shift_ce_gradient': True})
    old_shift = np.maximum(value, 1) ** (-.25)
    assert np.array_equal(m.shift_for(value, 'fedlc'), -old_shift)
report['checks']['loss_identities'] = {'dirichlet1_normalized_equivalence': equivalences,
    'uniform_counts_reduce_to_ce': uniform_rows, 'boundary_cases': boundary_rows,
    'fedlc_shift_matches_frozen_formula_bitwise': True}


class SyntheticIndexDataset(Dataset):
    """Implement the same labels/indices interface, with excluded backing rows."""
    def __init__(self, count, seed):
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(count + 10, 3, 32, 32, generator=generator)
        self.labels = np.array(([0, 1, 4] * ((count + 2) // 3))[:count] + [9] * 10, dtype=np.int64)
        self.indices = np.arange(count, dtype=np.int64)
        self.order = []

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        actual = int(self.indices[index])
        self.order.append(actual)
        return self.images[actual], torch.tensor(self.labels[actual], dtype=torch.long)


torch.manual_seed(20261201)
model = m.r.SmallCNN()
anchor = m.r.vectorize(model).clone()
assert anchor.numel() == 136874
assert m.r.SmallCNN is old.r.SmallCNN and m.r.metrics is old.r.metrics
train_rows, local_deltas = [], {}
for loss_name in ('ce', 'fedlc', 'balanced'):
    dataset = SyntheticIndexDataset(67, 1201)
    events = {'backward': 0, 'optimizer_step': 0}
    previous_backward, previous_step = torch.Tensor.backward, torch.optim.SGD.step

    def tracked_backward(value, *args, **kwargs):
        events['backward'] += 1
        return previous_backward(value, *args, **kwargs)

    def tracked_step(optimizer, *args, **kwargs):
        events['optimizer_step'] += 1
        return previous_step(optimizer, *args, **kwargs)

    torch.Tensor.backward, torch.optim.SGD.step = tracked_backward, tracked_step
    try:
        delta, counters = m.train(model, dataset, 20261201, 'cpu', loss_name)
    finally:
        torch.Tensor.backward, torch.optim.SGD.step = previous_backward, previous_step
    assert counters['steps'] == counters['gradient_evaluations'] == events['backward'] == events['optimizer_step'] == 2
    assert counters['train_counts'] == np.bincount(dataset.labels[dataset.indices], minlength=10).tolist()
    assert counters['train_counts'][9] == 0  # backing rows excluded from local data
    assert torch.equal(m.r.vectorize(model), anchor)
    local_deltas[loss_name] = delta
    train_rows.append({'loss': loss_name, 'examples': 67, 'batches': 2,
        'backward_calls': events['backward'], 'optimizer_step_calls': events['optimizer_step'],
        'global_anchor_bitwise_unchanged': True, 'counts_ignore_excluded_rows': True,
        'order': dataset.order})
assert all(row['order'] == train_rows[0]['order'] for row in train_rows)
old_dataset = SyntheticIndexDataset(67, 1201)
old_delta, _, old_counters = old.train(model, old_dataset, 20261201, 'cpu', True)
assert torch.equal(old_delta, local_deltas['fedlc'])
assert old_dataset.order == train_rows[0]['order']
new_shift = np.array(m.shift_for(np.array(old_counters['train_label_counts']), 'fedlc'), dtype=np.float32)
assert np.array_equal(new_shift, -np.array(old_counters['calibration_shift'], dtype=np.float32))
report['checks']['local_training'] = {'methods': train_rows,
    'same_seed_batch_order': True, 'fedlc_matches_frozen_mixed_local_delta_bitwise': True,
    'fedlc_matches_frozen_mixed_shift_bitwise': True, 'global_anchor_unchanged': True}
single_class_rows = []
for loss_name in ('ce', 'fedlc', 'balanced'):
    single = SyntheticIndexDataset(3, 1202)
    single.labels[single.indices] = 7
    delta, counters = m.train(model, single, 20261201, 'cpu', loss_name)
    assert torch.isfinite(delta).all() and torch.equal(m.r.vectorize(model), anchor)
    assert counters['train_counts'] == [0, 0, 0, 0, 0, 0, 0, 3, 0, 0]
    single_class_rows.append({'loss': loss_name, 'finite_local_delta': True,
                              'single_class': 7, 'steps': counters['steps']})
report['checks']['local_training']['single_class_local_train'] = single_class_rows

# Actual four-group two-round run, including actual encoder wires and EF state.
trainsets, dev = [SyntheticIndexDataset(3, 1211), SyntheticIndexDataset(5, 1212)], SyntheticIndexDataset(2, 1213)
args = SimpleNamespace(out=OUT / 'complete', seed=20261201, partition_seed=20261193, rounds=2, backend='cpu')
manifest = {'fixture_only': True, 'client_examples': [3, 5], 'dev_examples': 2, 'official_data_loaded': False}
original_train, original_encode = m.train, m.codec.encode
captured, encoded, current = {}, {}, {}


def observed_train(model, dataset, seed, backend, loss):
    before_anchor = m.r.vectorize(model).clone()
    dataset.order.clear()
    result = original_train(model, dataset, seed, backend, loss)
    assert torch.equal(before_anchor, m.r.vectorize(model))
    current['training'].append({'seed': seed, 'order': dataset.order.copy(),
        'anchor': before_anchor, 'delta': result[0].clone()})
    return result


def observed_encode(value, *args, **kwargs):
    result = original_encode(value, *args, **kwargs)
    current['encoded'].append({'u': value.clone(), 'wire': result.wire, 'sent': result.sent.clone()})
    return result


m.train, m.codec.encode = observed_train, observed_encode
try:
    for method in m.METHODS:
        current['training'], current['encoded'] = [], []
        with contextlib.redirect_stdout(io.StringIO()):
            m.run(args, method, trainsets, dev, manifest)
        captured[method], encoded[method] = current['training'], current['encoded']
finally:
    m.train, m.codec.encode = original_train, original_encode
reference = captured['ce_uniform3']
assert all(torch.equal(rows[0]['anchor'], reference[0]['anchor']) for rows in captured.values())
for rows in captured.values():
    assert [(row['seed'], row['order']) for row in rows] == [(row['seed'], row['order']) for row in reference]
    assert all(torch.equal(rows[index]['anchor'], rows[index + 1]['anchor']) for index in (0, 2))
for left, right in zip(captured['balanced_uniform3'][:2], captured['balanced_dense'][:2]):
    assert torch.equal(left['delta'], right['delta'])

run_rows = []
for method in m.METHODS:
    folder = args.out / method
    config = json.loads((folder / 'config.json').read_text())
    checkpoint = torch.load(folder / 'checkpoint.pt', weights_only=False)
    result = json.loads((folder / 'result.json').read_text())
    unsigned = {key: value for key, value in config.items() if key != 'signature'}
    assert hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest() == config['signature'] == checkpoint['signature'] == result['signature']
    assert config['source_sha256'] == before['work/balanced_quant/run.py']
    assert config['plan_sha256'] == before['work/balanced_quant/PLAN.md']
    assert config['codec_sha256'] == before['work/mixed_precision/codec.py']
    assert config['partition_seed'] == 20261193 and config['inference'] == 'raw logits'
    assert config['weights'] == [.375, .625] and not config['official_test_loaded']
    residuals = [torch.zeros(136874), torch.zeros(136874)]
    wire_sizes, max_ef = [], 0.
    predicted_model = None
    for round_id in range(2):
        aggregate = torch.zeros(136874)
        for client_id, weight in enumerate(config['weights']):
            index = 2 * round_id + client_id
            training = captured[method][index]
            if method.endswith('dense'):
                q = training['delta']
                residuals[client_id].zero_()
            else:
                packet = encoded[method][index]
                assert torch.equal(packet['u'], training['delta'] + residuals[client_id])
                q = m.codec.decode(packet['wire'])
                assert q.numel() == 136874 and torch.equal(q, packet['sent'])
                residuals[client_id] = packet['u'] - q
                max_ef = max(max_ef, maximum(q + residuals[client_id] - packet['u']))
                assert len(packet['wire']) <= 54808
                wire_sizes.append(len(packet['wire']))
            aggregate.add_(q, alpha=weight)
        predicted_model = captured[method][round_id * 2]['anchor'] + aggregate
        assert result['rows'][round_id]['download_bytes_analytical'] == 2 * 547560
        assert all(client['steps'] == client['gradient_evaluations'] == 1 for client in result['rows'][round_id]['clients'])
    assert all(torch.equal(left, right) for left, right in zip(residuals, checkpoint['residuals']))
    final_model = m.r.SmallCNN()
    final_model.load_state_dict(checkpoint['model'])
    assert torch.equal(m.r.vectorize(final_model), predicted_model)
    assert m.r.metrics(final_model, dev) == result['rows'][-1]['validation']
    if method.endswith('dense'):
        assert not (folder / 'first_payload.bin').exists() and config['upload_cap_bytes'] is None
        assert all(client['upload_bytes'] == 547560 for row in result['rows'] for client in row['clients'])
    else:
        assert (folder / 'first_payload.bin').read_bytes() == encoded[method][0]['wire']
        assert wire_sizes == [54067] * 4 and max_ef < 1e-7
    run_rows.append({'method': method, 'rounds': 2, 'signature': config['signature'],
        'actual_uniform3_wire_sizes': wire_sizes, 'body_and_head_full_vector_ef_max_error': max_ef,
        'saved_residuals_and_decoder_aggregation_match': True,
        'raw_logits_metrics_recomputed': True, 'source_plan_codec_bound': True})
report['checks']['four_group_integration'] = {'synthetic_client_examples': [3, 5],
    'dev_examples': 2, 'seeds': [row['seed'] for row in reference],
    'same_seed_batch_initial_anchor': True, 'global_anchor_unchanged_between_clients': True,
    'balanced_uniform_dense_first_local_deltas_match': True, 'methods': run_rows}

# Interrupt a real two-round run after round one; resume the exact same config.
resume_args = SimpleNamespace(**{**vars(args), 'out': OUT / 'interrupted'})
calls = []


class DeliberateInterruption(Exception):
    pass


def interrupted_train(*arguments):
    calls.append(True)
    if len(calls) == 3:
        raise DeliberateInterruption()
    return original_train(*arguments)


m.train = interrupted_train
try:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            m.run(resume_args, 'balanced_uniform3', trainsets, dev, manifest)
    except DeliberateInterruption:
        pass
    else:
        raise AssertionError('expected fixture interruption')
finally:
    m.train = original_train
with contextlib.redirect_stdout(io.StringIO()):
    m.run(resume_args, 'balanced_uniform3', trainsets, dev, manifest)
resumed = torch.load(resume_args.out / 'balanced_uniform3/checkpoint.pt', weights_only=False)
completed = torch.load(args.out / 'balanced_uniform3/checkpoint.pt', weights_only=False)
assert resumed['signature'] == completed['signature']
assert all(torch.equal(value, completed['model'][name]) for name, value in resumed['model'].items())
assert all(torch.equal(left, right) for left, right in zip(resumed['residuals'], completed['residuals']))
assert resumed['rows'][-1]['validation'] == completed['rows'][-1]['validation']

folder = args.out / 'balanced_uniform3'
checkpoint_path, config_path = folder / 'checkpoint.pt', folder / 'config.json'
checkpoint_bytes, config_bytes = checkpoint_path.read_bytes(), config_path.read_bytes()
negative, unexpected_training = {}, []


def forbid_train(*arguments):
    unexpected_training.append(True)
    raise AssertionError('training during signature rejection fixture')


def rejected(name, selected_args=args):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            m.run(selected_args, 'balanced_uniform3', trainsets, dev, manifest)
    except ValueError:
        negative[name] = True
    else:
        raise AssertionError('accepted incompatible fixture: ' + name)


m.train = forbid_train
try:
    with contextlib.redirect_stdout(io.StringIO()):
        m.run(args, 'balanced_uniform3', trainsets, dev, manifest)
    for name, value in [('seed', args.seed + 1), ('partition_seed', args.partition_seed + 1), ('rounds', 3), ('backend', 'mps')]:
        rejected(name + '_mismatch', SimpleNamespace(**{**vars(args), name: value}))
    original_sha = m.sha
    for name, source in [('run_source', HERE / 'run.py'), ('plan_source', HERE / 'PLAN.md'),
                         ('codec_source', HERE.parent / 'mixed_precision/codec.py')]:
        m.sha = lambda path, source=source: '0' * 64 if Path(path).resolve() == source.resolve() else original_sha(path)
        try:
            rejected(name + '_mismatch')
        finally:
            m.sha = original_sha
    bad = torch.load(io.BytesIO(checkpoint_bytes), weights_only=False)
    bad['signature'] = 'wrong-fixture-signature'
    torch.save(bad, checkpoint_path)
    rejected('wrong_checkpoint_signature')
finally:
    m.train = original_train
    checkpoint_path.write_bytes(checkpoint_bytes)
    config_path.write_bytes(config_bytes)
assert not unexpected_training
report['checks']['resume'] = {'interrupted_round1_resume_model_residuals_bitwise_equal': True,
    'resumed_dev_metrics_equal': True, 'completed_resume_no_training': True,
    'signature_rejections_before_training': negative}
assert before == {str(path.relative_to(ROOT)): sha(path) for path in protected}
assert not report['official_dataset_open_attempts']
report['source_identities_unchanged'] = True
report['harness'] = {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}
report['status'], report['failures'] = 'PASS', []
(HERE / 'LOCAL_FIXTURES.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'fixture_out': str(OUT),
    'checks': list(report['checks']), 'official_dataset_open_attempts': report['official_dataset_open_attempts'],
    'frozen_run_sha256': before['work/balanced_quant/run.py']}, indent=2))

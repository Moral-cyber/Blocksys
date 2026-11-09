"""Tiny CPU implementation fixtures; never load official data or start a benchmark."""
from pathlib import Path
from types import SimpleNamespace
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
SOURCE_PATHS = (HERE / 'run.py', HERE / 'PLAN.md', HERE.parent / 'head_control/run.py',
                HERE.parent / 'head_control/PLAN.md', HERE.parent / 'mixed_precision/codec.py')
before = {str(path.relative_to(ROOT)): sha(path) for path in SOURCE_PATHS}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m = load('_control_reference_fixture', HERE / 'run.py')
background = load('_frozen_head_control_background_fixture', HERE.parent / 'head_control/run.py')
torch.set_num_threads(1)
OUT = Path(tempfile.mkdtemp(prefix='control_reference_fixture_'))
report = {'scope': 'CPU single-thread synthetic implementation fixture only',
          'fixture_out': str(OUT), 'source_identities': before,
          'official_dataset_open_attempts': [], 'checks': {}}


def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)) and '/dataset/' in str(args[0]):
        report['official_dataset_open_attempts'].append(str(args[0]))
        raise AssertionError('official data access forbidden in synthetic fixture')


sys.addaudithook(audit)


class LoggedDataset(TensorDataset):
    def __init__(self, *values):
        super().__init__(*values)
        self.order = []

    def __getitem__(self, index):
        self.order.append(int(index))
        return super().__getitem__(index)


def dataset(count, seed):
    generator = torch.Generator().manual_seed(seed)
    return LoggedDataset(torch.randn(count, 3, 32, 32, generator=generator),
                         torch.randint(0, 10, (count,), generator=generator))


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def maximum(value):
    return float(value.abs().max()) if value.numel() else 0.


torch.manual_seed(20261191)
model = m.r.SmallCNN()
anchor = m.r.vectorize(model).clone()
assert m.r.SmallCNN is background.r.SmallCNN and m.r.metrics is background.r.metrics
assert m.data.prepare is background.data.prepare
assert [m.control_size(model, method) for method in m.METHODS] == [0, 136224, 136874]
assert model.classifier[3].weight.numel() + model.classifier[3].bias.numel() == 650
report['checks']['layout_helpers'] = {'parameters': 136874, 'body': 136224, 'head': 650,
    'control_sizes': [0, 136224, 136874], 'same_model_partition_metrics_helpers': True}

# Observe raw backward gradients, then the actual gradients passed to SGD.
local_rows = []
for method in m.METHODS:
    size = m.control_size(model, method)
    ci = torch.linspace(-.01, .01, size) if size else torch.empty(0)
    c = torch.linspace(.003, -.003, size) if size else torch.empty(0)
    raw, observations, holder = {}, [], {}
    oldcopy, oldstep = copy.deepcopy, torch.optim.SGD.step

    def tracked_copy(obj, *args, **kwargs):
        value = oldcopy(obj, *args, **kwargs)
        if obj is model:
            holder['model'] = value
            for name, parameter in value.named_parameters():
                parameter.register_hook(lambda grad, name=name: raw.__setitem__(name, grad.detach().clone()))
        return value

    def tracked_step(optimizer, *args, **kwargs):
        parameters = list(holder['model'].named_parameters())
        rawvector = torch.cat([raw[name].flatten() for name, _ in parameters])
        adjusted = torch.cat([parameter.grad.flatten() for _, parameter in parameters])
        expected = torch.zeros_like(rawvector)
        expected[:size] = c - ci
        observations.append({'raw': rawvector[:size].clone(),
            'correction_error': maximum(adjusted - rawvector - expected),
            'outside_added': maximum((adjusted - rawvector)[size:])})
        return oldstep(optimizer, *args, **kwargs)

    copy.deepcopy, torch.optim.SGD.step = tracked_copy, tracked_step
    ds = dataset(67, 901)
    try:
        delta, newci, steps, error = m.train(model, ds, 20261191, 'cpu', ci, c, size)
    finally:
        copy.deepcopy, torch.optim.SGD.step = oldcopy, oldstep
    assert steps == 2 and len(observations) == 2
    assert max(row['correction_error'] for row in observations) < 5e-8
    assert max(row['outside_added'] for row in observations) == 0.
    rawmean = torch.stack([row['raw'] for row in observations]).mean(0)
    assert maximum(newci - rawmean) < m.TOLERANCE
    assert torch.equal(m.r.vectorize(model), anchor)
    local_rows.append({'method': method, 'steps': steps, 'control_size': size,
        'gradient_correction_error': max(row['correction_error'] for row in observations),
        'outside_subset_gradient_added': max(row['outside_added'] for row in observations),
        'ci_raw_gradient_mean_error': maximum(newci - rawmean),
        'global_anchor_bitwise_unchanged': True, 'order': ds.order})
assert all(row['order'] == local_rows[0]['order'] for row in local_rows)
plain_dataset, head_dataset = dataset(67, 901), dataset(67, 901)
plain_delta = m.train(model, plain_dataset, 20261191, 'cpu', torch.empty(0), torch.empty(0), 0)[0]
head_delta = background.train(model, head_dataset, 20261191, 'cpu', None, None)[0]
assert torch.equal(plain_delta, head_delta) and plain_dataset.order == head_dataset.order
report['checks']['local_gradient'] = {'examples': 67, 'batch': 64, 'methods': local_rows,
    'same_seed_batch_order_all_scopes': True,
    'plain_matches_frozen_head_runner_zero_control_delta_and_batch_order': True}

# Check general nonzero old controls with unequal actual step counts, beyond
# the tiny integrated clients whose datasets each occupy one batch.
identity_rows = []
generator = torch.Generator().manual_seed(914)
for size in (m.BODY, m.TOTAL):
    weights = [.375, .625]
    oldcis = [torch.randn(size, generator=generator) * .01 for _ in weights]
    oldc = weights[0] * oldcis[0] + weights[1] * oldcis[1]
    rawmeans = [torch.randn(size, generator=generator) * .03 for _ in weights]
    counts = [2, 5]
    deltas = [-m.LR * count * (raw + oldc - ci)
              for count, raw, ci in zip(counts, rawmeans, oldcis)]
    newcis = [ci - oldc - delta / (m.LR * count)
              for ci, delta, count in zip(oldcis, deltas, counts)]
    direct = sum((weight * delta / (-m.LR * count)
                  for weight, delta, count in zip(weights, deltas, counts)), torch.zeros(size))
    weighted = weights[0] * newcis[0] + weights[1] * newcis[1]
    assert maximum(weighted - direct) < m.TOLERANCE
    identity_rows.append({'control_size': size, 'weights': weights, 'steps': counts,
        'weighted_new_ci_vs_server_direct_error': maximum(weighted - direct),
        'new_ci_vs_raw_gradient_mean_error': max(maximum(ci - raw) for ci, raw in zip(newcis, rawmeans))})
report['checks']['nonzero_control_unequal_steps'] = identity_rows

# Actual run(): two rounds make the second round use nonzero stored controls.
trainsets, dev = [dataset(3, 911), dataset(5, 912)], dataset(2, 913)
manifest = {'fixture_only': True, 'synthetic_client_sizes': [3, 5],
            'synthetic_dev_size': 2, 'official_data_loaded': False}
args = SimpleNamespace(out=OUT / 'complete', seed=20261191, rounds=2, backend='cpu')
original_train, captured, current = m.train, {}, {}


def observed_train(model, ds, seed, backend, ci, c, size):
    before_anchor = m.r.vectorize(model).clone()
    ds.order.clear()
    result = original_train(model, ds, seed, backend, ci, c, size)
    assert torch.equal(before_anchor, m.r.vectorize(model))
    current['rows'].append({'seed': seed, 'order': ds.order.copy(),
        'anchor_sha256': tensor_sha(before_anchor), 'delta': result[0].clone(),
        'ci_new': result[1].clone(), 'steps': result[2]})
    return result


m.train = observed_train
try:
    for method in m.METHODS:
        current['rows'] = []
        with contextlib.redirect_stdout(io.StringIO()):
            m.run(args, method, trainsets, dev, manifest)
        captured[method] = current['rows']
finally:
    m.train = original_train
reference = captured['plain_dense']
for method, rows in captured.items():
    assert [(row['seed'], row['order'], row['anchor_sha256']) for row in rows] == [
        (row['seed'], row['order'], row['anchor_sha256']) for row in reference]
    assert all(torch.equal(left['delta'], right['delta']) for left, right in zip(rows[:2], reference[:2]))

run_rows = []
for method in m.METHODS:
    folder = args.out / method
    config = json.loads((folder / 'config.json').read_text())
    result = json.loads((folder / 'result.json').read_text())
    checkpoint = torch.load(folder / 'checkpoint.pt', weights_only=False)
    signed = {key: value for key, value in config.items() if key != 'signature'}
    assert m.signature(signed) == config['signature'] == checkpoint['signature'] == result['signature']
    assert config['weights'] == [3/8, 5/8]
    assert config['source_sha256'] == before['work/control_reference/run.py']
    assert config['plan_sha256'] == before['work/control_reference/PLAN.md']
    size = config['control_size']
    for round_id, row in enumerate(result['rows']):
        assert row['actual_gradient_evaluations'] == row['actual_optimizer_steps'] == 2
        assert row['upload_bytes_analytical'] == 2 * (64 + 4 * m.TOTAL)
        assert row['download_model_bytes_analytical'] == 2 * (64 + 4 * m.TOTAL)
        assert row['download_control_bytes_analytical'] == (2 * (16 + 4 * size) if size else 0)
        if size:
            records = captured[method][round_id * 2:round_id * 2 + 2]
            weighted = sum((weight * value['ci_new'] for weight, value in zip(config['weights'], records)), torch.zeros(size))
            direct = sum((weight * value['delta'][:size] / (-m.LR * value['steps']) for weight, value in zip(config['weights'], records)), torch.zeros(size))
            assert maximum(weighted - direct) < m.TOLERANCE
    assert maximum(m.weighted_controls(config['weights'], checkpoint['cis'], size) - checkpoint['c']) < m.TOLERANCE
    checked_model = m.r.SmallCNN()
    checked_model.load_state_dict(checkpoint['model'])
    assert m.r.metrics(checked_model, dev) == result['rows'][-1]['validation']
    run_rows.append({'method': method, 'rounds': len(result['rows']),
        'control_size': size, 'signature': config['signature'],
        'max_invariant_error': max(row['control_invariant_error'] for row in result['rows']),
        'max_local_formula_error': max(client['control_formula_max_error'] for row in result['rows'] for client in row['clients']),
        'gradient_evaluations_total': sum(row['actual_gradient_evaluations'] for row in result['rows']),
        'upload_analytical_per_client': config['upload_bytes_analytical_per_client'],
        'download_model_analytical_per_client': config['download_model_bytes_analytical_per_client'],
        'download_control_analytical_per_client': config['download_control_bytes_analytical_per_client'],
        'control_state_per_client_bytes': config['control_state_per_client_bytes'],
        'complete_model_ci_c_weights_state': True, 'same_metrics_helper_recompute': True})
report['checks']['two_round_integration'] = {'synthetic_client_sizes': [3, 5],
    'synthetic_dev_size': 2, 'weights': [3/8, 5/8], 'methods': run_rows,
    'same_seed_batch_anchor_all_groups': True,
    'first_round_raw_local_deltas_bitwise_equal': True,
    'seeds': [row['seed'] for row in reference]}

# A real interrupted checkpoint must continue to the same exact final tensors.
resume_args = SimpleNamespace(**{**vars(args), 'out': OUT / 'interrupted'})
calls = []


class DeliberateInterruption(Exception):
    pass


def stop_after_first_round(*arguments):
    calls.append(True)
    if len(calls) == 3:
        raise DeliberateInterruption()
    return original_train(*arguments)


m.train = stop_after_first_round
try:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            m.run(resume_args, 'body_control', trainsets, dev, manifest)
    except DeliberateInterruption:
        pass
    else:
        raise AssertionError('expected fixture interruption')
finally:
    m.train = original_train
checkpoint_path = resume_args.out / 'body_control/checkpoint.pt'
assert len(torch.load(checkpoint_path, weights_only=False)['rows']) == 1
with contextlib.redirect_stdout(io.StringIO()):
    m.run(resume_args, 'body_control', trainsets, dev, manifest)
resumed = torch.load(checkpoint_path, weights_only=False)
complete = torch.load(args.out / 'body_control/checkpoint.pt', weights_only=False)
assert resumed['signature'] == complete['signature']
assert all(torch.equal(value, complete['model'][name]) for name, value in resumed['model'].items())
assert torch.equal(resumed['c'], complete['c'])
assert all(torch.equal(left, right) for left, right in zip(resumed['cis'], complete['cis']))
assert resumed['rows'][-1]['validation'] == complete['rows'][-1]['validation']

# Deliberately damage only temporary fixture outputs; every case must fail
# before a client train call. Restore bytes after each negative check.
folder = args.out / 'body_control'
config_path, checkpoint_path = folder / 'config.json', folder / 'checkpoint.pt'
config_bytes, checkpoint_bytes = config_path.read_bytes(), checkpoint_path.read_bytes()
negative, train_attempts = {}, []


def forbid_train(*arguments):
    train_attempts.append(True)
    raise AssertionError('unexpected train during resume/signature rejection')


def rejected(name, selected_args=args):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            m.run(selected_args, 'body_control', trainsets, dev, manifest)
    except ValueError:
        negative[name] = True
    else:
        raise AssertionError('accepted invalid fixture: ' + name)


m.train = forbid_train
try:
    with contextlib.redirect_stdout(io.StringIO()):
        m.run(args, 'body_control', trainsets, dev, manifest)
    for key, value in [('seed', args.seed + 1), ('backend', 'mps'), ('rounds', 3)]:
        rejected(key + '_mismatch', SimpleNamespace(**{**vars(args), key: value}))
    oldsha = m.sha
    m.sha = lambda path: '0' * 64 if Path(path).resolve() == (HERE / 'run.py').resolve() else oldsha(path)
    try:
        rejected('source_identity_mismatch')
    finally:
        m.sha = oldsha
    changed = json.loads(config_bytes)
    changed['weights'] = [.5, .5]
    config_path.write_text(json.dumps(changed))
    rejected('tampered_config_with_old_signature')
    config_path.write_bytes(config_bytes)
    mutations = {
        'checkpoint_signature': lambda ck: ck.update(signature='wrong'),
        'checkpoint_version': lambda ck: ck.update(version=999),
        'checkpoint_weights': lambda ck: ck.update(weights=[.5, .5]),
        'checkpoint_control_shape': lambda ck: ck.update(c=torch.zeros(3)),
        'checkpoint_control_invariant': lambda ck: ck.update(c=ck['c'] + 1),
    }
    for name, mutation in mutations.items():
        changed = torch.load(io.BytesIO(checkpoint_bytes), weights_only=False)
        mutation(changed)
        torch.save(changed, checkpoint_path)
        rejected(name)
        checkpoint_path.write_bytes(checkpoint_bytes)
finally:
    m.train = original_train
    config_path.write_bytes(config_bytes)
    checkpoint_path.write_bytes(checkpoint_bytes)
assert not train_attempts
report['checks']['resume'] = {'interrupted_after_round': 1,
    'resumed_final_model_ci_c_bitwise_equal': True,
    'resumed_final_dev_metrics_equal': True,
    'completed_resume_no_training': True, 'negative_rejections_before_training': negative}
assert not report['official_dataset_open_attempts']
assert before == {str(path.relative_to(ROOT)): sha(path) for path in SOURCE_PATHS}
report['source_identities_unchanged'] = True
report['harness'] = {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}
report['status'], report['failures'] = 'PASS', []
(HERE / 'LOCAL_FIXTURES.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'fixture_out': str(OUT),
    'checks': list(report['checks']), 'official_data_open_attempts': report['official_dataset_open_attempts'],
    'run_sha256': before['work/control_reference/run.py']}, indent=2))

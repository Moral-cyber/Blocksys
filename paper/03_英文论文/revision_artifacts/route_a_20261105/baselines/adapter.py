"""New matched dense FedAvg/FedProx runs. Original sources/results read only."""
from pathlib import Path
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PAPER = Path('/Users/wind/Desktop/Codex/paper')
METHOD = PAPER / '05_核心方法研发'
WORK = METHOD / 'repro/work'
TASKS = {'cifar': (20261217, (20261231, 20261232, 20261233)),
         'fashion': (20261219, (20261241, 20261242, 20261243))}
ARMS = {'fedavg_dense': 0., 'fedprox01_dense': .1}
VARIANTS = ('raw', 'zero_anchor_vector', 'prioranchored_vector')
METRICS = ('accuracy', 'nll', 'brier', 'ece15')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def signed(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dependencies(task):
    if task == 'cifar':
        common = load('_baseline_cifar_common', WORK / 'calibrated_lowbit_confirm/common.py')
        raw = PAPER / '02_实验工程/dataset/Cifar10/rawdata'
        prepare = lambda p: common.prepare(p, raw)
    else:
        fashion = load('_baseline_fashion_adapter', WORK / 'calibrated_fashion/adapter.py')
        common = fashion.common
        raw = METHOD / 'scripts/datasets/FashionMNIST/raw'
        prepare = lambda p: fashion.prepare(p, raw)
    return common, prepare


def original_manifest(task, seed):
    partition = TASKS[task][0]
    path = METHOD / f'results/predictive_pfl_confirm/development_{task}_p{partition}_s{seed}/fedlc_uniform3/config.json'
    return json.loads(path.read_text())['data_manifest'], path


def checked_data(task, seed):
    c, prepare = dependencies(task)
    sets, validation, manifest, data_ids, images, labels = prepare(TASKS[task][0])
    expected, expected_path = original_manifest(task, seed)
    for key in ('train_indices', 'holdout_indices', 'validation_indices', 'client_label_counts', 'client_sizes'):
        if manifest[key] != expected[key]:
            raise ValueError('new baseline partition differs from original: ' + key)
    # Use the exact original manuscript manifest for assignment identity.
    return c, sets, validation, expected, {**data_ids, str(expected_path): sha(expected_path)}, images, labels


def test_guard():
    attempts = []
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(args[0])
            if 'test_batch' in path or 't10k' in path:
                attempts.append(path)
                raise ValueError('TRAIN-only baseline stage cannot read official TEST')
    sys.addaudithook(audit)
    return attempts


def sources(c):
    result = c.source_identities()
    for path in (Path(__file__), HERE / 'PLAN.md', WORK / 'prior_vector/solver.py',
                 WORK / 'calibration_baselines/solver.py'):
        result[str(path)] = sha(path)
    return result


def verify(mapping):
    if any(sha(p) != h for p, h in mapping.items()):
        raise ValueError('locked baseline input changed')


def local_train(c, model, dataset, seed, backend, mu):
    if mu == 0:
        # Exactly the frozen main-protocol CE local training.
        return c.balanced.train(model, dataset, seed, backend, 'ce')
    if mu != .1:
        raise ValueError('only predeclared mu=.1 allowed')
    torch.manual_seed(seed)
    if backend == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; no fallback')
        torch.mps.manual_seed(seed)
    local = copy.deepcopy(model).to(backend).train()
    anchor = c.runner.vectorize(model).clone()
    parameter_anchors = [p.detach().clone() for p in local.parameters()]
    loader = DataLoader(dataset, batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.SGD(local.parameters(), lr=.03)
    for x, y in loader:
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(local(x.to(backend)), y.to(backend)).backward()
        with torch.no_grad():
            for p, base in zip(local.parameters(), parameter_anchors):
                if p.grad is not None:
                    p.grad.add_(p - base, alpha=mu)
        optimizer.step()
    delta = torch.cat([p.detach().flatten() for p in local.parameters()]).cpu() - anchor
    if not torch.isfinite(delta).all() or not torch.equal(c.runner.vectorize(model), anchor):
        raise ValueError('nonfinite proximal update or altered global model')
    counts = np.bincount(dataset.labels[dataset.indices], minlength=10)
    return delta, {'steps': len(loader), 'gradient_evaluations': len(loader),
                   'train_counts': counts.tolist(), 'proximal_mu': mu}


def folder(task, seed, arm):
    return HERE / 'runs' / task / f's{seed}' / arm


def train(task, seed):
    if seed not in TASKS[task][1]:
        raise ValueError('fixed original seed only')
    attempts = test_guard()
    torch.set_num_threads(1)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no fallback')
    c, sets, validation, manifest, data_ids, _, _ = checked_data(task, seed)
    identities = {**sources(c), **data_ids}
    weights = np.asarray([len(ds) for ds in sets], dtype=np.float64)
    weights /= weights.sum()
    for arm, mu in ARMS.items():
        out = folder(task, seed, arm)
        config = {'version': 1, 'task': task, 'method': arm, 'mu': mu, 'seed': seed,
                  'partition_seed': TASKS[task][0], 'rounds': 20, 'backend': 'mps',
                  'batch': 64, 'lr': .03, 'local_epochs': 1, 'parameters': 136874,
                  'weights': weights.tolist(), 'data_manifest': manifest,
                  'source_and_data_identities': identities,
                  'implementation': 'in-house matched baseline; not author-code reproduction',
                  'objective': 'ordinary CE' if mu == 0 else 'CE + .1/2 * squared distance from global anchor',
                  'communication': 'dense analytical 64+4*136874 bytes per upload/downlink',
                  'selection': 'fixed original seeds/final20; no TEST choice',
                  'torch_version': str(torch.__version__)}
        config['signature'] = signed(config)
        if (out / 'result.json').exists():
            prior = json.loads((out / 'result.json').read_text())
            old = json.loads((out / 'config.json').read_text())
            if old != config:
                raise ValueError('existing baseline source/config mismatch')
            if prior.get('status') == 'DONE':
                verify(identities)
                print(json.dumps({'status': 'ALREADY_DONE', 'task': task, 'seed': seed, 'arm': arm}), flush=True)
                continue
        out.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)
        model = c.runner.SmallCNN()
        rows = []
        if (out / 'checkpoint.pt').exists():
            old = torch.load(out / 'checkpoint.pt', map_location='cpu', weights_only=False)
            if old['config'] != config or old['signature'] != config['signature']:
                raise ValueError('checkpoint mismatch')
            model.load_state_dict(old['model']); rows = old['rows']
        write(out / 'config.json', config)
        for rnd in range(len(rows), 20):
            began = time.perf_counter()
            anchor = c.runner.vectorize(model).clone()
            aggregate = torch.zeros_like(anchor)
            clients = []
            for client, ds in enumerate(sets):
                local_seed = seed + 100 * rnd + client
                delta, counts = local_train(c, model, ds, local_seed, 'mps', mu)
                if not torch.equal(c.runner.vectorize(model), anchor):
                    raise ValueError('client changed global anchor')
                aggregate.add_(delta, alpha=float(weights[client]))
                clients.append({'client': client, 'seed': local_seed, **counts})
            c.runner.assign_vector(model, anchor + aggregate)
            if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
                raise ValueError('nonfinite global model')
            validation_metrics = c.runner.metrics(model, validation)
            rows.append({'round': rnd + 1, 'validation': validation_metrics, 'clients': clients,
                         'global_anchor_unchanged_between_clients': True,
                         'upload_bytes_analytical': len(sets) * 547560,
                         'download_bytes_analytical': len(sets) * 547560,
                         'seconds': time.perf_counter() - began})
            temporary = out / 'checkpoint.tmp'
            torch.save({'version': 1, 'config': config, 'signature': config['signature'],
                        'model': model.state_dict(), 'rows': rows}, temporary)
            temporary.replace(out / 'checkpoint.pt')
            write(out / 'result.json', {'status': 'DONE' if len(rows) == 20 else 'RUNNING',
                                      'rows': rows, 'signature': config['signature'],
                                      'official_test_reads': 0})
            print(json.dumps({'task': task, 'seed': seed, 'arm': arm, 'round': rnd + 1,
                              'validation': validation_metrics, 'seconds': rows[-1]['seconds']}), flush=True)
        verify(identities)
    if attempts:
        raise ValueError('forbidden TEST access attempted')


def forward(model, dataset):
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for x, y in DataLoader(dataset, batch_size=256, shuffle=False):
            logits.append(model(x).numpy().copy()); labels.append(y.numpy().copy())
    return np.concatenate(logits), np.concatenate(labels)


def load_model(c, task, seed, arm):
    root = folder(task, seed, arm)
    config = json.loads((root / 'config.json').read_text())
    report = json.loads((root / 'result.json').read_text())
    if config['signature'] != signed({k: v for k, v in config.items() if k != 'signature'}):
        raise ValueError('baseline signature invalid')
    verify(config['source_and_data_identities'])
    checkpoint = torch.load(root / 'checkpoint.pt', map_location='cpu', weights_only=False)
    if (report['status'] != 'DONE' or len(checkpoint['rows']) != 20 or
        report['rows'] != checkpoint['rows'] or checkpoint['config'] != config or
        checkpoint['signature'] != config['signature'] or config['seed'] != seed or
        config['task'] != task or config['method'] != arm):
        raise ValueError('only exact complete final baseline checkpoint allowed')
    model = c.runner.SmallCNN(); model.load_state_dict(checkpoint['model']); model.eval()
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('baseline state nonfinite')
    return model, config


def fit():
    attempts = test_guard(); torch.set_num_threads(1)
    solver = load('_baseline_vector_solver', WORK / 'prior_vector/solver.py')
    all_outputs = {}
    for task, (_, seeds) in TASKS.items():
        c, sets, _, manifest, data_ids, _, _ = checked_data(task, seeds[0])
        for seed in seeds:
            for arm in ARMS:
                model, config = load_model(c, task, seed, arm)
                out = folder(task, seed, arm) / 'train_states'
                if (out / 'result.json').exists():
                    old = json.loads((out / 'result.json').read_text())
                    if old.get('status') == 'DONE':
                        verify(old['outputs']); all_outputs.update(old['outputs']); continue
                    raise ValueError('incomplete states require inspection')
                out.mkdir(parents=True)
                clients = []
                for client, ds in enumerate(sets):
                    logits, labels = forward(model, ds)
                    if not np.array_equal(labels, ds.labels[ds.indices]):
                        raise ValueError('TRAIN row order changed')
                    counts = np.bincount(labels, minlength=10)
                    if counts.tolist() != manifest['client_label_counts'][client]:
                        raise ValueError('TRAIN counts differ')
                    prior = solver.train_prior_anchor(labels)
                    zero_state, zero_info = solver.fit(logits, labels, np.zeros(10))
                    prior_state, prior_info = solver.fit(logits, labels, prior)
                    path = out / f'client{client}.npz'
                    np.savez_compressed(path, train_logits=logits, train_labels=labels,
                                        train_ids=np.asarray(ds.indices, dtype=np.uint32),
                                        train_counts=counts, zero_anchor_vector=zero_state,
                                        prioranchored_vector=prior_state)
                    all_outputs[str(path)] = sha(path)
                    clients.append({'client': client, 'train_samples': len(ds),
                                    'zero_fit': zero_info, 'prior_fit': prior_info})
                output_hashes = {str(out / f'client{i}.npz'): sha(out / f'client{i}.npz') for i in range(20)}
                write(out / 'result.json', {'status': 'DONE', 'task': task, 'seed': seed, 'method': arm,
                                           'clients': clients, 'outputs': output_hashes,
                                           'checkpoint_sha256': sha(folder(task, seed, arm) / 'checkpoint.pt'),
                                           'fit_source_sha256': sha(WORK / 'prior_vector/solver.py'),
                                           'official_test_reads': 0})
                print(json.dumps({'status': 'STATES_DONE', 'task': task, 'seed': seed, 'arm': arm}), flush=True)
        verify(data_ids)
    if attempts:
        raise ValueError('forbidden TEST access attempted')
    snapshot = {str(HERE / 'PLAN.md'): sha(HERE / 'PLAN.md'), str(Path(__file__)): sha(Path(__file__))}
    for task, (_, seeds) in TASKS.items():
        for seed in seeds:
            for arm in ARMS:
                root = folder(task, seed, arm)
                for name in ('config.json', 'result.json', 'checkpoint.pt', 'train_states/result.json'):
                    snapshot[str(root / name)] = sha(root / name)
    snapshot.update(all_outputs)
    path = HERE / 'FIT_LOCK.json'
    lock = {'status': 'ALL_BASELINE_MODELS_AND_TRAIN_STATES_FROZEN',
            'created_at': datetime.now(timezone.utc).isoformat(), 'inputs': snapshot,
            'task_seed_matrix': TASKS, 'arms': ARMS, 'variants': VARIANTS,
            'selection': 'none; all fits completed before any new baseline TEST access'}
    if path.exists() and json.loads(path.read_text())['inputs'] != snapshot:
        raise ValueError('cannot replace existing FIT lock')
    if not path.exists():
        write(path, lock)
    print(json.dumps({'status': 'ALL_FITS_LOCKED', 'inputs': len(snapshot)}), flush=True)


def final_modules():
    path = WORK / 'predictive_final_test'
    sys.path.insert(0, str(path))
    import data as d
    import scorer as s
    return d, s


def summary(clients):
    weights = np.asarray([v['test_rows'] for v in clients], dtype=np.float64)
    if len(clients) != 20 or np.any(weights <= 0) or weights.sum() != 10000:
        raise ValueError('exact original 20-client TEST assignment required')
    weights /= weights.sum()
    result = {}
    for variant in VARIANTS:
        values = {}
        for key in METRICS:
            numbers = np.asarray([v['metrics'][variant][key] for v in clients])
            values['macro_' + key] = float(numbers.mean())
            values['weighted_' + key] = float(numbers @ weights)
        values['client_accuracy_q10'] = float(np.quantile([v['metrics'][variant]['accuracy'] for v in clients], .1))
        conf = np.sum([v['metrics'][variant]['ece_bins']['confidence_sum'] for v in clients], axis=0)
        corr = np.sum([v['metrics'][variant]['ece_bins']['correct_sum'] for v in clients], axis=0)
        values['pooled_ece15'] = float(np.abs(corr - conf).sum() / 10000)
        result[variant] = values
    return result


def score():
    torch.set_num_threads(1)
    d, s = final_modules()
    locked = json.loads((HERE / 'FIT_LOCK.json').read_text())
    if locked.get('status') != 'ALL_BASELINE_MODELS_AND_TRAIN_STATES_FROZEN':
        raise ValueError('all models/states must freeze before TEST')
    verify(locked['inputs'])
    for task, (_, seeds) in TASKS.items():
        root = HERE / 'official_test' / task
        if (root / 'result.json').exists():
            print(json.dumps({'status': 'ALREADY_SCORED', 'task': task}), flush=True); continue
        c, _ = dependencies(task)
        bundles = [(seed, arm, *load_model(c, task, seed, arm)) for seed in seeds for arm in ARMS]
        original, original_path = original_manifest(task, seeds[0])
        counts = np.asarray(original['client_label_counts'], dtype=np.int64)
        if any(config['data_manifest'] != original for _, _, _, config in bundles):
            raise ValueError('baseline manifests must match original')
        lock = {'status': 'ROOT_LOCKED_FINAL_TEST', 'tasks': [task],
                'plan_sha256': d.file_sha(WORK / 'predictive_final_test/FINAL_TEST_PLAN.md'),
                'source_sha256': {name: d.file_sha(WORK / 'predictive_final_test' / name) for name in ('data.py', 'scorer.py')},
                'baseline_plan_sha256': sha(HERE / 'PLAN.md'), 'baseline_fit_lock_sha256': sha(HERE / 'FIT_LOCK.json'),
                'authorization': 'root delegated fixed baseline execution; no selection'}
        root.mkdir(parents=True, exist_ok=True)
        lock_path = root / 'ROOT_LOCK.json'; write(lock_path, lock)
        if task == 'cifar':
            images, labels, provenance = d.load_cifar_test(PAPER / '02_实验工程/dataset/Cifar10/rawdata', root_lock=lock_path, execute=True)
        else:
            images, labels, provenance = d.load_fashion_test(METHOD / 'scripts/datasets/FashionMNIST/raw', root_lock=lock_path, execute=True)
        groups, assignment = d.assign_clients(labels, counts, task=task, train_manifest_sha256=d.file_sha(original_path))
        # Ordered assignment depends on counts and the fixed seed, not manifest annotation.
        old_assignment = json.loads((METHOD / f'results/predictive_final_test/official_{task}/assignment.json').read_text())
        if assignment['assignment_hash'] != old_assignment['assignment_hash']:
            raise ValueError('new baseline TEST membership differs from original main results')
        write(root / 'assignment.json', assignment)
        reports = []
        for seed, arm, model, config in bundles:
            began = time.perf_counter(); chunks = []
            with torch.inference_mode():
                for offset in range(0, 10000, 256):
                    x = torch.from_numpy(d.normalize_batch(images[offset:offset + 256], task))
                    chunks.append(model(x).numpy().copy())
            logits = np.concatenate(chunks)
            clients = []
            out = root / f'{arm}_s{seed}'; out.mkdir()
            for client, ids in enumerate(groups):
                state_path = folder(task, seed, arm) / f'train_states/client{client}.npz'
                with np.load(state_path, allow_pickle=False) as saved:
                    if not np.array_equal(saved['train_counts'], counts[client]):
                        raise ValueError('frozen states counts differ')
                    states = {name: saved[name].copy() for name in VARIANTS if name != 'raw'}
                scores = {'raw': logits[ids].astype(np.float64)}
                for variant, state in states.items():
                    if state.dtype != np.float32 or state.shape != (20,) or not np.isfinite(state).all():
                        raise ValueError('finite FP32 deployment state required')
                    p = state.astype(np.float64)
                    scores[variant] = scores['raw'] * p[:10] + p[10:]
                measured = {name: s.metrics(value, labels[ids]) for name, value in scores.items()}
                np.savez_compressed(out / f'client{client}.npz', row_ids=np.asarray(ids, dtype=np.uint32),
                                    labels=labels[ids], **{'score_' + k: v for k, v in scores.items()}, **states)
                clients.append({'client': client, 'test_rows': len(ids), 'metrics': measured})
            report = {'status': 'DONE', 'task': task, 'seed': seed, 'method': arm,
                      'clients': clients, 'summaries': summary(clients), 'official_label_provenance': provenance,
                      'assignment_hash': assignment['assignment_hash'], 'inference_seconds': time.perf_counter() - began,
                      'selection': 'none; 0 TEST fits, 0 TEST backward, fixed all variants'}
            write(out / 'result.json', report); reports.append(report)
            print(json.dumps({'status': 'SCORE_DONE', 'task': task, 'seed': seed, 'arm': arm,
                              'summaries': report['summaries']}), flush=True)
        aggregates = {}
        for arm in ARMS:
            runs = [r for r in reports if r['method'] == arm]
            if [r['seed'] for r in runs] != list(seeds):
                raise ValueError('all three fixed seeds must be present')
            aggregates[arm] = {}
            for variant in VARIANTS:
                aggregates[arm][variant] = {key: {'mean': float(np.mean([r['summaries'][variant][key] for r in runs])),
                                                 'sample_sd': float(np.std([r['summaries'][variant][key] for r in runs], ddof=1))}
                                            for key in runs[0]['summaries'][variant]}
        verify(locked['inputs'])
        write(root / 'result.json', {'status': 'DONE', 'task': task, 'seeds': seeds,
                                    'summaries': aggregates, 'assignment_hash': assignment['assignment_hash'],
                                    'all_model_state_hashes_unchanged': True, 'official_test_fits': 0})


def fixtures():
    test_guard(); torch.set_num_threads(1)
    c, _ = dependencies('cifar')
    class Dataset(torch.utils.data.Dataset):
        def __init__(self):
            rng = np.random.default_rng(195)
            self.images = rng.normal(size=(71, 3, 32, 32)).astype(np.float32)
            self.labels = np.arange(71, dtype=np.int64) % 10; self.indices = np.arange(71)
        def __len__(self): return len(self.indices)
        def __getitem__(self, i): return torch.from_numpy(self.images[i]), int(self.labels[i])
    torch.manual_seed(719); model = c.runner.SmallCNN(); ds = Dataset()
    d0, _ = local_train(c, model, ds, 123, 'cpu', 0.)
    expected, _ = c.balanced.train(model, ds, 123, 'cpu', 'ce')
    if not torch.equal(d0, expected):
        raise ValueError('mu0 must equal frozen CE trainer bitwise')
    anchor = c.runner.vectorize(model).clone()
    proximal, _ = local_train(c, model, ds, 123, 'cpu', .1)
    if not torch.equal(c.runner.vectorize(model), anchor) or not torch.isfinite(proximal).all() or torch.equal(proximal, d0):
        raise ValueError('two-minibatch proximal fixture must preserve anchor and affect update')
    aggregate = torch.zeros_like(d0)
    aggregate.add_(d0, alpha=.25); aggregate.add_(proximal, alpha=.75)
    if not torch.allclose(aggregate, .25 * d0 + .75 * proximal, rtol=0, atol=2e-9):
        raise ValueError('sample-weighted aggregation fixture failed')
    torch.manual_seed(42); a = nn.Linear(3, 2); base = [p.detach().clone() for p in a.parameters()]
    with torch.no_grad():
        for p in a.parameters(): p.add_(.2)
    x = torch.randn(5, 3); y = torch.tensor([0, 1, 0, 0, 1])
    loss = nn.functional.cross_entropy(a(x), y)
    loss.backward(); manual = [p.grad.clone() + .1 * (p.detach() - b) for p, b in zip(a.parameters(), base)]
    a.zero_grad(); objective = nn.functional.cross_entropy(a(x), y) + .05 * sum(((p - b) ** 2).sum() for p, b in zip(a.parameters(), base))
    objective.backward(); error = max(float((p.grad - g).abs().max()) for p, g in zip(a.parameters(), manual))
    if error > 1e-7: raise ValueError('proximal gradient differs from explicit objective')
    d, s = final_modules(); z = np.zeros((10, 10)); labels = np.arange(10)
    measured = s.metrics(z, labels)
    if abs(measured['nll'] - np.log(10)) > 1e-14 or abs(measured['accuracy'] - .1) > 1e-14:
        raise ValueError('metric known case failed')
    write(HERE / 'FIXTURES.json', {'status': 'PASS', 'mu0_frozen_train_bitwise_equal': True,
                                 'proximal_gradient_max_error': error, 'metric_uniform_nll': measured['nll'],
                                 'proximal_update_differs_after_two_minibatches': True,
                                 'private_client_preserves_global_anchor': True, 'sample_weighted_aggregation': True,
                                 'official_test_reads': 0, 'source_sha256': sha(Path(__file__))})
    print(json.dumps({'status': 'FIXTURES_PASS', 'gradient_error': error}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--stage', choices=('fixtures', 'train', 'fit', 'score'), required=True)
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.stage != 'fixtures':
        source_lock = json.loads((HERE / 'SOURCE_LOCK.json').read_text())
        if source_lock.get('status') != 'FROZEN_BEFORE_BASELINE_TRAINING':
            raise ValueError('pre-training source/PLAN lock required')
        verify(source_lock['inputs'])
    if args.stage == 'fixtures': fixtures()
    elif args.stage == 'fit': fit()
    elif args.stage == 'score': score()
    elif args.task and args.seed: train(args.task, args.seed)
    else: parser.error('train requires --task and --seed')


if __name__ == '__main__':
    main()

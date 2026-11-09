"""Existing SCAFFOLD full/body control references; dense development only."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'method_followup'))
import run_followup as r
import run_partitions as data
import run_accelerated as accel

METHODS = ('plain_dense', 'body_control', 'full_control')
LR = .03
BATCH = 64
HEAD = 650
TOTAL = 136874
BODY = TOTAL - HEAD
TOLERANCE = 5e-6
CHECKPOINT_VERSION = 1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def vector_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def control_size(model, method):
    names = list(model.named_parameters())
    if ([name for name, _ in names[-2:]] != ['classifier.3.weight', 'classifier.3.bias']
            or sum(p.numel() for _, p in names[-2:]) != HEAD
            or sum(p.numel() for _, p in names) != TOTAL):
        raise ValueError('unexpected frozen SmallCNN parameter layout')
    if method not in METHODS:
        raise ValueError('unknown reference method')
    return {'plain_dense': 0, 'body_control': BODY, 'full_control': TOTAL}[method]


def weighted_controls(weights, controls, size):
    return sum((float(weight) * ci for weight, ci in zip(weights, controls)), torch.zeros(size))


def max_error(left, right):
    return float((left - right).abs().max()) if left.numel() else 0.


def train(model, dataset, seed, backend, ci, c, size):
    torch.manual_seed(seed)
    if backend == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; no fallback')
        torch.mps.manual_seed(seed)
    if size not in (0, BODY, TOTAL):
        raise ValueError('unsupported control subset')
    if size and (ci.shape != (size,) or c.shape != (size,)
                 or not torch.isfinite(ci).all() or not torch.isfinite(c).all()):
        raise ValueError('invalid local control state')
    local = copy.deepcopy(model).to(backend).train()
    anchor = r.vectorize(model).clone()
    correction = (c - ci).to(backend) if size else None
    sumgrad = torch.zeros(size, device=backend)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    if not len(loader):
        raise ValueError('empty client dataset')
    optimizer = torch.optim.SGD(local.parameters(), lr=LR)
    steps = 0
    for x, y in loader:
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(local(x.to(backend)), y.to(backend)).backward()
        with torch.no_grad():
            if size:
                offset = 0
                for parameter in local.parameters():
                    count = min(parameter.numel(), size - offset)
                    if count <= 0:
                        break
                    gradient = parameter.grad.view(-1)[:count]
                    sumgrad[offset:offset + count].add_(gradient)
                    gradient.add_(correction[offset:offset + count])
                    offset += count
                if offset != size:
                    raise ValueError('control subset does not match model')
        optimizer.step()
        steps += 1
    delta = torch.cat([p.detach().flatten() for p in local.parameters()]).cpu() - anchor
    newci = ci - c - delta[:size] / (LR * steps) if size else torch.empty(0)
    error = max_error(newci, sumgrad.cpu() / steps) if size else 0.
    if (not torch.isfinite(delta).all() or not torch.isfinite(newci).all()
            or error > TOLERANCE or not torch.equal(r.vectorize(model), anchor)):
        raise ValueError('local control identity, finite update, or global anchor failure')
    return delta, newci, steps, error


def validate_checkpoint(checkpoint, config, weights, client_sizes, size):
    if checkpoint.get('version') != CHECKPOINT_VERSION:
        raise ValueError('wrong checkpoint version')
    if checkpoint.get('signature') != config['signature']:
        raise ValueError('wrong checkpoint signature')
    if checkpoint.get('config') != config:
        raise ValueError('wrong checkpoint source/config identity')
    if (checkpoint.get('weights') != weights.tolist()
            or checkpoint.get('client_sizes') != client_sizes):
        raise ValueError('wrong checkpoint participation weights')
    rows = checkpoint.get('rows')
    if (not isinstance(rows, list) or len(rows) > config['rounds']
            or [row.get('round') for row in rows] != list(range(1, len(rows) + 1))):
        raise ValueError('invalid checkpoint round history')
    cis, c = checkpoint.get('cis'), checkpoint.get('c')
    if not isinstance(cis, list) or len(cis) != len(weights):
        raise ValueError('wrong checkpoint client control count')
    for value in [*cis, c]:
        if (not isinstance(value, torch.Tensor) or value.shape != (size,)
                or value.dtype != torch.float32 or value.device.type != 'cpu'
                or not torch.isfinite(value).all()):
            raise ValueError('invalid checkpoint control tensor')
    if max_error(weighted_controls(weights, cis, size), c) > TOLERANCE:
        raise ValueError('checkpoint weighted control invariant broken')


def run(args, method, trainsets, dev, manifest):
    torch.manual_seed(args.seed)
    model = r.SmallCNN()
    size = control_size(model, method)
    client_sizes = [len(dataset) for dataset in trainsets]
    if not client_sizes or any(count <= 0 for count in client_sizes):
        raise ValueError('full participation requires nonempty fixed clients')
    weights = np.array(client_sizes, dtype=float)
    weights /= weights.sum()
    if not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12):
        raise ValueError('sample weights do not sum to one')
    identities = {'run.py': sha(__file__), 'PLAN.md': sha(HERE / 'PLAN.md')}
    config = {'method': method, 'seed': args.seed, 'rounds': args.rounds,
        'backend': args.backend, 'source_sha256': identities['run.py'],
        'plan_sha256': identities['PLAN.md'], 'source_identities': identities,
        'runtime': accel.runtime_metadata(args.backend, 'existing_scaffold_control_reference'),
        'manifest': manifest, 'parameters': TOTAL, 'body': BODY, 'head': HEAD,
        'control_size': size, 'control_subset': 'prefix' if size else 'none',
        'lr': LR, 'batch': BATCH, 'local_epochs': 1,
        'objective': 'ordinary all-class cross entropy; no label calibration',
        'aggregation': 'full participation, fixed sample weights',
        'client_sizes': client_sizes, 'weights': weights.tolist(),
        'batch_seed': 'seed + 100 * zero_based_round + zero_based_client',
        'control_formula': 'ci_new = ci - c - delta_subset / (lr * actual_steps)',
        'control_prior': 'existing SCAFFOLD; delta reconstruction has SCALLION prior',
        'control_state_per_client_bytes': 4 * size,
        'global_control_state_bytes': 4 * size,
        'simulation_control_state_bytes': 4 * size * (len(trainsets) + 1),
        'upload_bytes_analytical_per_client': 64 + 4 * TOTAL,
        'upload_control_bytes': 0,
        'upload_layout': 'one FP32 dense delta; assumed 64B header includes actual K and shape',
        'download_model_bytes_analytical_per_client': 64 + 4 * TOTAL,
        'download_control_bytes_analytical_per_client': 16 + 4 * size if size else 0,
        'communication': 'all uploads/downlinks analytical; no wire/network/authentication/chain measurement',
        'comparison_scope': 'existing strong-control diagnostic; same-seed head_control_dense background is not an independent repeat',
        'official_test_loaded': False,
        'evaluation': 'official TRAIN stratified 2000 global development holdout',
        'invariant_tolerance': TOLERANCE, 'checkpoint_version': CHECKPOINT_VERSION}
    config['signature'] = signature(config)
    out = args.out / method
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists():
        previous = json.loads((out / 'config.json').read_text())
        stored_signature = previous.pop('signature', None)
        if stored_signature != signature(previous) or stored_signature != config['signature']:
            raise ValueError('incompatible source/config output')
    cis = [torch.zeros(size) for _ in trainsets]
    c, rows = torch.zeros(size), []
    if (out / 'checkpoint.pt').exists():
        checkpoint = torch.load(out / 'checkpoint.pt', weights_only=False)
        validate_checkpoint(checkpoint, config, weights, client_sizes, size)
        model.load_state_dict(checkpoint['model'])
        if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError('nonfinite checkpoint model')
        cis, c, rows = checkpoint['cis'], checkpoint['c'], checkpoint['rows']
    r.atomic_json(out / 'config.json', config)
    for round_id in range(len(rows), args.rounds):
        began = time.perf_counter()
        anchor = r.vectorize(model).clone()
        aggregate = torch.zeros_like(anchor)
        prior_error = max_error(weighted_controls(weights, cis, size), c)
        if prior_error > TOLERANCE:
            raise ValueError('prior weighted control invariant broken')
        newcis, clients = [], []
        direct_c = torch.zeros(size)
        for client_id, dataset in enumerate(trainsets):
            seed = args.seed + 100 * round_id + client_id
            delta, newci, steps, error = train(model, dataset, seed, args.backend,
                                               cis[client_id], c, size)
            newcis.append(newci)
            if size:
                direct_c.add_(delta[:size] / (-LR * steps), alpha=float(weights[client_id]))
            aggregate.add_(delta, alpha=float(weights[client_id]))
            clients.append({'client': client_id, 'seed': seed, 'samples': len(dataset),
                'weight': float(weights[client_id]), 'steps': steps,
                'gradient_evaluations': steps, 'optimizer_steps': steps,
                'control_formula_max_error': error,
                'control_state_norm': float(newci.norm()),
                'delta_norm': float(delta.norm()),
                'upload_bytes_analytical': 64 + 4 * TOTAL})
        weighted_c = weighted_controls(weights, newcis, size)
        invariant = max_error(weighted_c, direct_c)
        if invariant > TOLERANCE or not torch.equal(r.vectorize(model), anchor):
            raise ValueError('server control invariant or global anchor broken')
        cis, c = newcis, weighted_c
        r.assign_vector(model, anchor + aggregate)
        metrics = r.metrics(model, dev)
        upload = len(trainsets) * (64 + 4 * TOTAL)
        down_model = len(trainsets) * (64 + 4 * TOTAL)
        down_control = len(trainsets) * (16 + 4 * size) if size else 0
        rows.append({'round': round_id + 1, 'validation': metrics, 'clients': clients,
            'global_anchor_sha256': vector_sha(anchor),
            'global_anchor_unchanged_during_clients': True,
            'actual_gradient_evaluations': sum(client['gradient_evaluations'] for client in clients),
            'actual_optimizer_steps': sum(client['optimizer_steps'] for client in clients),
            'prior_control_invariant_error': prior_error,
            'control_invariant_error': invariant, 'control_norm': float(c.norm()),
            'upload_bytes_analytical': upload,
            'download_model_bytes_analytical': down_model,
            'download_control_bytes_analytical': down_control,
            'download_bytes_analytical': down_model + down_control,
            'seconds': time.perf_counter() - began})
        temporary = out / 'checkpoint.tmp'
        torch.save({'version': CHECKPOINT_VERSION, 'signature': config['signature'],
            'config': config, 'model': model.state_dict(), 'cis': cis, 'c': c,
            'weights': weights.tolist(), 'client_sizes': client_sizes, 'rows': rows}, temporary)
        temporary.replace(out / 'checkpoint.pt')
        r.atomic_json(out / 'result.json', {'status': 'DONE' if len(rows) == args.rounds else 'RUNNING',
            'rows': rows, 'signature': config['signature'], 'official_test_loaded': False})
        print(json.dumps({'method': method, 'round': round_id + 1,
            'accuracy': metrics['accuracy'], 'ce': metrics['loss'],
            'control_invariant_error': invariant, 'upload_bytes_analytical': upload,
            'seconds': rows[-1]['seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261191)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--backend', choices=('mps', 'cpu'), default='mps')
    parser.add_argument('--methods', default=','.join(METHODS))
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds must be positive')
    methods = args.methods.split(',')
    if len(methods) != len(set(methods)) or any(method not in METHODS for method in methods):
        parser.error('unknown or duplicate method')
    torch.set_num_threads(1)
    if args.backend == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no fallback')
    trainsets, dev, manifest = data.prepare('Cifar10IndexA0p1', 0, 20261143, 2000)
    if len(trainsets) != 20:
        raise ValueError('reference plan requires exactly twenty fixed clients')
    for method in methods:
        run(args, method, trainsets, dev, manifest)


if __name__ == '__main__':
    main()

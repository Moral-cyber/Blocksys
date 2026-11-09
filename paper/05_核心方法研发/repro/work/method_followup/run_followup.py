"""Controlled follow-up for energy-guarded sparse federated updates.

Development uses only a fixed subset of held-out official TRAIN images.
The official test set is opened only with the explicit --test-final flag.
Runs are simulated synchronous FL, not network/blockchain measurements.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from guarded_codec import encode

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / 'tinyproto_diag'
sys.path.insert(0, str(PROJECT / 'tools'))
from fedavg_update_diagnostic import SmallCNN, assign_vector, load_client, vectorize


VARIANTS = {
    'dense': ('dense', 1.0),
    'topk': ('topk', 1.0),
    'topk_ef': ('topk_ef', 1.0),
    'align90': ('guard_alignment_ef', .90),
    'align99': ('guard_alignment_ef', .99),
    'align999': ('guard_alignment_ef', .999),
    'random90': ('guard_random_ef', .90),
    'innov90': ('guard_innovation_ef', .90),
    'agree90': ('guard_agreement_ef', .90),
    'align50': ('guard_alignment_ef', .50),
    'random99': ('guard_random_ef', .99),
    'innov99': ('guard_innovation_ef', .99),
}


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def prepare(dataset, mixing, split_seed, validation_size):
    trainsets = [load_client(dataset, i, True) for i in range(20)]
    valsets = [load_client(dataset, i, False) for i in range(20)]
    sizes = [len(t) for t in trainsets]
    # Keep client sample counts, global images, and sample-weighted objective
    # constant. Mixing changes only label/image assignment among clients.
    if mixing > 0:
        rng = np.random.default_rng(split_seed)
        kept, pool_x, pool_y, needed = [], [], [], []
        for ds in trainsets:
            idx = rng.permutation(len(ds))
            n = int(round(mixing * len(ds)))
            moved, retained = idx[:n], idx[n:]
            x, y = ds.tensors
            kept.append((x[retained], y[retained]))
            pool_x.append(x[moved]); pool_y.append(y[moved]); needed.append(n)
        px, py = torch.cat(pool_x), torch.cat(pool_y)
        order = rng.permutation(len(py)); start = 0
        trainsets = []
        for (kx, ky), n in zip(kept, needed):
            idx = order[start:start+n]; start += n
            trainsets.append(TensorDataset(torch.cat((kx, px[idx])), torch.cat((ky, py[idx]))))
    vx = torch.cat([x.tensors[0] for x in valsets])
    vy = torch.cat([x.tensors[1] for x in valsets])
    rng = np.random.default_rng(73021)  # fixed across all methods and seeds
    # Stratify approximately, then fill without using any predicted outputs.
    ids = []
    per_class = validation_size // 10
    for c in range(10):
        available = np.flatnonzero(vy.numpy() == c)
        ids.extend(rng.permutation(available)[:per_class].tolist())
    ids = np.asarray(ids, dtype=np.int64)
    rest = np.setdiff1d(np.arange(len(vy)), ids)
    if len(ids) < validation_size:
        ids = np.concatenate((ids, rng.permutation(rest)[:validation_size-len(ids)]))
    validation = TensorDataset(vx[ids], vy[ids])
    digest = hashlib.sha256()
    counts = []
    for ds in trainsets:
        digest.update(ds.tensors[1].numpy().tobytes())
        counts.append(torch.bincount(ds.tensors[1], minlength=10).tolist())
    return trainsets, validation, {
        'source': dataset, 'mixing': mixing, 'split_seed': split_seed,
        'train_samples': sum(sizes), 'client_sizes': sizes,
        'validation_source': 'fixed class-stratified subset of official-train local holdouts',
        'validation_samples': len(validation),
        'validation_ids_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
        'client_label_counts': counts, 'assignment_label_sha256': digest.hexdigest(),
        'test_used_for_selection': False,
    }


def train_local(model, dataset, seed, lr, epochs):
    torch.manual_seed(seed)
    local = copy.deepcopy(model)
    local.train()
    loader = DataLoader(dataset, batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.SGD(local.parameters(), lr=lr)
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(local(x), y).backward()
            opt.step()
    return vectorize(local) - vectorize(model)


@torch.no_grad()
def metrics(model, dataset):
    model.eval()
    hits = torch.zeros(10, dtype=torch.long)
    counts = torch.zeros(10, dtype=torch.long)
    loss_sum = 0.
    for x, y in DataLoader(dataset, batch_size=256):
        logits = model(x)
        loss_sum += float(nn.functional.cross_entropy(logits, y, reduction='sum'))
        pred = logits.argmax(1)
        counts += torch.bincount(y, minlength=10)
        hits += torch.bincount(y[pred == y], minlength=10)
    accs = hits.double() / counts.clamp_min(1)
    return {'accuracy': float(hits.sum() / counts.sum()), 'loss': loss_sum / len(dataset),
            'class_accuracy': accs.tolist(), 'macro_accuracy': float(accs.mean()),
            'worst_class_accuracy': float(accs.min()), 'samples': len(dataset)}


def encode_variant(delta, residual, reference, fraction, variant, seed):
    method, tau = VARIANTS[variant]
    if method == 'dense':
        return delta, torch.zeros_like(delta), {'encoded_bytes': 64 + 4*delta.numel(),
                'filler_fraction': 0., 'retained_topk_energy_ratio': 1., 'alignment_gain': 0.}
    if method == 'topk':
        result = encode(delta, torch.zeros_like(delta), fraction=fraction,
                        method='topk_ef', reference=reference, seed=seed)
        return result.sent, torch.zeros_like(delta), result.diagnostics
    result = encode(delta, residual, fraction=fraction, method=method,
                    reference=reference, tau=tau, seed=seed)
    return result.sent, result.residual, result.diagnostics


def one_run(args, trainsets, validation, manifest, variant, seed, out):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = SmallCNN()
    base = vectorize(model)
    residuals = [torch.zeros_like(base) for _ in trainsets]
    previous = None
    weights = np.asarray([len(ds) for ds in trainsets], dtype=float)
    weights /= weights.sum()
    rows, client_rows, total_bytes = [], [], 0
    start_round = 0
    checkpoint_path = out / 'checkpoint.pt'
    result_path = out / 'result.json'
    config_path = out / 'config.json'
    run_config = {**vars(args), 'variant': variant, 'seed': seed,
                'data_manifest': manifest, 'parameter_count': base.numel(),
                'evaluation_type': 'real_gt; synchronous FL simulation',
                'codec_source_sha256': hashlib.sha256((HERE/'guarded_codec.py').read_bytes()).hexdigest(),
                'runner_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    signature_config = {k:v for k,v in run_config.items() if k != 'out'}
    signature = hashlib.sha256(json.dumps(signature_config, sort_keys=True).encode()).hexdigest()
    run_config['run_signature'] = signature
    if config_path.exists():
        old_config = json.loads(config_path.read_text())
        if old_config.get('run_signature') != signature:
            raise ValueError(f'incompatible existing run at {out}; use a new output directory')
    elif result_path.exists() or checkpoint_path.exists():
        raise ValueError(f'existing artifacts without matching configuration at {out}')
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old.get('status') == 'DONE':
            print(f'skip complete {out.name}', flush=True)
            return
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(state['model'])
        residuals, previous = state['residuals'], state['previous']
        rows, client_rows, total_bytes = state['rows'], state['client_rows'], state['total_bytes']
        start_round = len(rows)
    atomic_json(config_path, run_config)
    for rnd in range(start_round, args.rounds):
        began = time.perf_counter()
        base = vectorize(model)
        aggregate = torch.zeros_like(base)
        dense_aggregate = torch.zeros_like(base)
        round_bytes, diagnostics = 0, []
        weighted_error_energy = 0.
        for i, ds in enumerate(trainsets):
            delta = train_local(model, ds, seed + rnd*100 + i, args.lr, args.local_epochs)
            candidate = delta + residuals[i] if variant not in ('dense','topk') else delta
            sent, next_residual, diag = encode_variant(delta, residuals[i], previous,
                                                       args.fraction, variant, seed + rnd*100 + i)
            # The simulation assumes unconditional delivery and completed aggregation.
            residuals[i] = next_residual
            aggregate.add_(sent, alpha=float(weights[i]))
            dense_aggregate.add_(candidate, alpha=float(weights[i]))
            weighted_error_energy += float(weights[i])**2 * float((candidate-sent).double().square().sum())
            round_bytes += diag['encoded_bytes']; diagnostics.append(diag)
            client_rows.append({'round': rnd+1, 'client': i, **diag})
        assign_vector(model, base + aggregate)
        total_bytes += round_bytes
        error_energy = float((dense_aggregate-aggregate).double().square().sum())
        dense_energy = float(dense_aggregate.double().square().sum())
        reference_cosine = 0. if previous is None else float(torch.dot(previous, dense_aggregate) /
                    (torch.linalg.vector_norm(previous)*torch.linalg.vector_norm(dense_aggregate)+1e-12))
        row = {'round': rnd+1, 'validation': metrics(model, validation),
               'round_bytes': round_bytes, 'cumulative_bytes': total_bytes,
               'aggregate_relative_error': (error_energy/(dense_energy+1e-30))**.5,
               'aggregate_error_squared': error_energy,
               'error_cross_term': error_energy-weighted_error_energy,
               'previous_dense_cosine': reference_cosine,
               'mean_filler_fraction': float(np.mean([d['filler_fraction'] for d in diagnostics])),
               'mean_retained_topk_energy_ratio': float(np.mean([d['retained_topk_energy_ratio'] for d in diagnostics])),
               'mean_alignment_gain': float(np.mean([d['alignment_gain'] for d in diagnostics])),
               'elapsed_seconds': time.perf_counter()-began}
        rows.append(row)
        previous = aggregate.detach().clone()
        temp = checkpoint_path.with_suffix('.tmp')
        torch.save({'model': model.state_dict(), 'residuals': residuals, 'previous': previous,
                    'rows': rows, 'client_rows': client_rows, 'total_bytes': total_bytes}, temp)
        temp.replace(checkpoint_path)
        atomic_json(result_path, {'status': 'IN_PROGRESS', 'variant': variant, 'seed': seed,
                    'rows': rows, 'client_rows': client_rows})
        print(f'{out.name} round {rnd+1}/{args.rounds} val={row["validation"]["accuracy"]:.4f} '
              f'loss={row["validation"]["loss"]:.4f} {row["elapsed_seconds"]:.1f}s', flush=True)
    final = {'status': 'DONE', 'variant': variant, 'seed': seed, 'rows': rows, 'client_rows': client_rows}
    if args.test_final:
        from sparse_update_benchmark import load_global_testset
        final['official_test_final'] = metrics(model, load_global_testset(args.dataset))
    atomic_json(result_path, final)


def snapshot_run(args, trainsets, validation, manifest, seed, out):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = SmallCNN(); previous = None; rows = []
    weights = np.asarray([len(ds) for ds in trainsets], dtype=float); weights /= weights.sum()
    atomic_json(out/'config.json', {**vars(args), 'seed': seed, 'data_manifest': manifest,
                'interpretation': 'one-step counterfactuals at identical dense-trajectory states; zero residuals'})
    for rnd in range(args.rounds):
        began = time.perf_counter(); base = vectorize(model)
        deltas = [train_local(model, ds, seed + rnd*100+i, args.lr, args.local_epochs)
                  for i, ds in enumerate(trainsets)]
        dense = sum(float(w)*d for w,d in zip(weights,deltas))
        is_probe = (rnd+1) in args.probe_rounds
        if is_probe:
            for variant in args.variants:
                encoded = [encode_variant(d,torch.zeros_like(d),previous,args.fraction,variant,seed+rnd*100+i)
                           for i,d in enumerate(deltas)]
                sent = sum(float(w)*e[0] for w,e in zip(weights,encoded))
                assign_vector(model, base+sent)
                error = float((dense-sent).double().square().sum())
                independent_error = sum(float(w)**2*float((d-e[0]).double().square().sum())
                                        for w,d,e in zip(weights,deltas,encoded))
                rows.append({'round':rnd+1,'variant':variant,'validation':metrics(model,validation),
                    'aggregate_error_squared':error,
                    'aggregate_relative_error':(error/(float(dense.double().square().sum())+1e-30))**.5,
                    'error_cross_term':error-independent_error,
                    'mean_filler_fraction':float(np.mean([e[2]['filler_fraction'] for e in encoded])),
                    'mean_retained_topk_energy_ratio':float(np.mean([e[2]['retained_topk_energy_ratio'] for e in encoded])),
                    'mean_alignment_gain':float(np.mean([e[2]['alignment_gain'] for e in encoded]))})
        assign_vector(model,base+dense)
        previous=dense.detach().clone()
        atomic_json(out/'result.json',{'status':'IN_PROGRESS','rows':rows})
        print(f'snapshot seed={seed} mix={args.mixing} round={rnd+1}/{args.rounds} '
              f'probe={is_probe} seconds={time.perf_counter()-began:.1f}',flush=True)
    atomic_json(out/'result.json',{'status':'DONE','rows':rows})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['train','snapshot'],default='train')
    p.add_argument('--dataset',default='Cifar10CleanA0p1')
    p.add_argument('--mixing',type=float,default=0.)
    p.add_argument('--split-seed',type=int,default=20261103)
    p.add_argument('--validation-size',type=int,default=2000)
    p.add_argument('--seeds',default='20261113')
    p.add_argument('--variants',default='topk_ef,align90,align99,random90,innov90')
    p.add_argument('--rounds',type=int,default=20)
    p.add_argument('--fraction',type=float,default=.05)
    p.add_argument('--lr',type=float,default=.03)
    p.add_argument('--local-epochs',type=int,default=1)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--test-final',action='store_true')
    p.add_argument('--probe-rounds',default='3,5,10,15')
    p.add_argument('--out',default=str(HERE/'runs'))
    args=p.parse_args()
    if not 0<=args.mixing<=1 or args.rounds<1 or args.validation_size<10:
        p.error('invalid mixing, rounds, or validation size')
    if not 0 < args.fraction <= 1 or not 0 < args.lr or args.local_epochs < 1 or args.threads < 1:
        p.error('invalid fraction, learning rate, local epochs, or threads')
    if args.validation_size > 12500:
        p.error('validation size exceeds the available clean holdout')
    args.variants=args.variants.split(',')
    unknown=set(args.variants)-set(VARIANTS)
    if unknown: p.error(f'unknown variants {unknown}')
    args.probe_rounds=[int(s) for s in args.probe_rounds.split(',')]
    torch.set_num_threads(args.threads)
    trainsets,validation,manifest=prepare(args.dataset,args.mixing,args.split_seed,args.validation_size)
    for seed in map(int,args.seeds.split(',')):
        if args.mode=='snapshot':
            out=Path(args.out)/f'snapshot_mix{args.mixing:g}_rho{args.fraction:g}_s{seed}'
            out.mkdir(parents=True,exist_ok=True)
            snapshot_run(args,trainsets,validation,manifest,seed,out)
        else:
            for variant in args.variants:
                out=Path(args.out)/f'{variant}_mix{args.mixing:g}_rho{args.fraction:g}_s{seed}'
                out.mkdir(parents=True,exist_ok=True)
                one_run(args,trainsets,validation,manifest,variant,seed,out)


if __name__=='__main__':
    main()

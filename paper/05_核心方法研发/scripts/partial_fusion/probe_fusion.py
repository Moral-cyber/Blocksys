"""Finite exploratory PFL fusion probe. No official test access or PFL claim."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'method_followup'))
import run_followup as base
import run_partitions as data
import run_accelerated as accel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def pack_slices(p, size):
    return [slice(start, min(start+size, p)) for start in range(0, p, size)]


def donor_average(vectors, masks, weights, exclude=None):
    numerator = torch.zeros_like(vectors[0])
    denominator = torch.zeros_like(vectors[0])
    for i, (v, m, w) in enumerate(zip(vectors, masks, weights)):
        if i == exclude:
            continue
        numerator[m] += float(w) * v[m]
        denominator[m] += float(w)
    valid = denominator > 0
    out = torch.zeros_like(numerator)
    out[valid] = numerator[valid] / denominator[valid]
    return out, valid


def ce(model, point, batch):
    base.assign_vector(model, point)
    with torch.no_grad():
        return float(nn.functional.cross_entropy(model(batch[0]), batch[1]))


def gradient(model, point, batch):
    base.assign_vector(model, point)
    loss = nn.functional.cross_entropy(model(batch[0]), batch[1])
    grads = torch.autograd.grad(loss, tuple(model.parameters()))
    return torch.cat([g.detach().flatten() for g in grads])


def train_batches(ds, seed):
    ids = np.random.default_rng(seed).permutation(len(ds))
    size = min(64, len(ids)//2)
    if size < 1:
        raise ValueError('at least two training samples required')
    fit, check = ids[:size], ids[size:2*size]
    if set(fit.tolist()) & set(check.tolist()):
        raise AssertionError('gradient/line search training batches overlap')
    def tensor(ids):
        rows = [ds[int(i)] for i in ids]
        return torch.stack([x for x,y in rows]), torch.stack([y for x,y in rows])
    return tensor(fit), tensor(check), {'gradient_positions': fit.tolist(), 'check_positions': check.tolist()}


def gate_direction(local, donor, available, grad, packs):
    direction = torch.where(available, donor-local, torch.zeros_like(local))
    selected = 0
    predicted = 0.
    for section in packs:
        gain = -float(torch.dot(grad[section].double(), direction[section].double()))
        if gain > 0:
            selected += int(available[section].any())
            predicted += gain
        else:
            direction[section] = 0
    return direction, {'selected_packs': selected, 'predicted_train_gain': predicted}


def line_search(model, local, direction, batch):
    alphas = (0., 1., .5, .25)
    losses = [ce(model, local + alpha*direction, batch) for alpha in alphas]
    selected = min(range(len(alphas)), key=lambda i: losses[i])
    return local + alphas[selected]*direction, {'alpha': alphas[selected], 'train_check_ce': losses,
        'extra_backward': 1, 'candidate_forward_excluding_shared_baseline': 3}


def summarize(rows):
    out = {}
    for name in rows[0]['metrics']:
        acc = np.array([r['metrics'][name]['accuracy'] for r in rows])
        loss = np.array([r['metrics'][name]['loss'] for r in rows])
        n = np.array([r['metrics'][name]['samples'] for r in rows])
        origin = np.array([r['metrics']['local']['loss'] for r in rows])
        out[name] = {'client_macro_accuracy': float(acc.mean()), 'client_macro_ce': float(loss.mean()),
            'sample_weighted_accuracy': float(np.average(acc, weights=n)),
            'sample_weighted_ce': float(np.average(loss, weights=n)),
            'fraction_clients_ce_improved_vs_local': float((loss < origin-1e-8).mean()),
            'fraction_clients_ce_harmed_vs_local': float((loss > origin+1e-8).mean())}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu','mps'), default='mps')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261143)
    parser.add_argument('--split-seed', type=int, default=20261143)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.backend == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('requested MPS unavailable; no fallback')
    args.out.mkdir(parents=True, exist_ok=True)
    images, labels, provenance = data.load_official_train()
    train_ids, holdout_ids, _, manifest = data.partition_indices(labels, alpha=.1,
        split_seed=args.split_seed, validation_size=2000)
    trainsets = [data.IndexedCifar10(images, labels, ids) for ids in train_ids]
    devsets = [data.IndexedCifar10(images, labels, ids) for ids in holdout_ids]
    manifest.update(provenance)
    config = {**vars(args), 'out': str(args.out.resolve()), 'alpha': .1, 'fraction': .05, 'pack_size':128,
        'data_manifest': manifest, 'source_sha256': {str(p.resolve()):sha(p) for p in (
        Path(__file__), HERE/'EXPERIMENT_PLAN.md', Path(data.__file__), Path(accel.__file__),
        Path(base.__file__), Path(sys.modules[base.SmallCNN.__module__].__file__))},
        'evaluation': 'real dataset labels; synchronous simulation; TRAIN holdout for development only',
        'official_test_loaded': False, 'runtime': accel.runtime_metadata(args.backend, 'partial_fusion_probe')}
    signature = sha(HERE/'EXPERIMENT_PLAN.md') + sha(__file__) + str(args.seed)+str(args.split_seed)+args.backend+str(args.warmup)
    config['signature'] = hashlib.sha256(signature.encode()).hexdigest()
    if (args.out/'config.json').exists():
        old = json.loads((args.out/'config.json').read_text())
        if old['signature'] != config['signature']:
            raise ValueError('existing incompatible run; choose new output')
    atomic(args.out/'config.json', config)
    torch.manual_seed(args.seed)
    model = base.SmallCNN()
    weights = np.array([len(ds) for ds in trainsets], dtype=float)
    weights /= weights.sum()
    warm = args.out/'warmup.pt'
    if warm.exists():
        ck = torch.load(warm, weights_only=False)
        if ck['signature'] != config['signature']:
            raise ValueError('checkpoint signature mismatch')
        model.load_state_dict(ck['model'])
        start = ck['round']
    else:
        start = 0
    for rnd in range(start, args.warmup):
        began = time.perf_counter()
        anchor = base.vectorize(model)
        aggregate = torch.zeros_like(anchor)
        for i, ds in enumerate(trainsets):
            delta = accel.train_local(model, ds, args.seed+rnd*100+i, .03, 1, backend=args.backend)
            aggregate += float(weights[i])*delta
        base.assign_vector(model, anchor+aggregate)
        torch.save({'model':model.state_dict(), 'round':rnd+1, 'signature':config['signature']}, warm)
        print(json.dumps({'stage':'warmup', 'round':rnd+1,'seconds':time.perf_counter()-began}), flush=True)
    anchor = base.vectorize(model)
    packs = pack_slices(len(anchor),128)
    k = max(1, int(len(packs)*.05))
    vectors, masks, uploads = [], [], []
    for i, ds in enumerate(trainsets):
        delta = accel.train_local(model, ds, args.seed+args.warmup*100+i, .03, 1, backend=args.backend)
        local = anchor+delta
        energy = torch.tensor([float(delta[s].double().square().sum()) for s in packs], dtype=torch.float64)
        chosen = torch.argsort(energy, descending=True, stable=True)[:k]
        mask = torch.zeros_like(anchor, dtype=torch.bool)
        for index in chosen.tolist():
            mask[packs[index]] = True
        vectors.append(local)
        masks.append(mask)
        uploads.append({'client':i,'pack_ids':chosen.tolist(),'values':int(mask.sum()),
            'payload_bytes':64+4*k+4*int(mask.sum())})
    torch.save({'anchor':anchor,'vectors':vectors,'masks':masks,'weights':weights,
        'signature':config['signature']}, args.out/'shared_state.pt')
    dense = sum(float(w)*v for w,v in zip(weights,vectors))
    inclusive, incvalid = donor_average(vectors,masks,weights)
    last_head_count = sum(p.numel() for p in model.classifier[-1].parameters())
    rows=[]
    evaluator = copy.deepcopy(model).cpu().eval()
    for i, local in enumerate(vectors):
        began=time.perf_counter()
        donor,valid = donor_average(vectors,masks,weights,exclude=i)
        fit,check,batch_ids = train_batches(trainsets[i], args.seed+10000+i)
        grad = gradient(evaluator,local,fit)
        direct = torch.where(valid,donor,local)
        direct_inc = torch.where(incvalid,inclusive,local)
        keephead=direct.clone()
        keephead[-last_head_count:]=local[-last_head_count:]
        dg,gd = gate_direction(local,donor,valid,grad,packs)
        gated,ls = line_search(evaluator,local,dg,check)
        dig,gid = gate_direction(local,inclusive,incvalid,grad,packs)
        gated_inc,lsi = line_search(evaluator,local,dig,check)
        sgd,lsg = line_search(evaluator,local,-.03*grad,check)
        points={'local':local,'dense_average':dense,'sparse_inclusive':direct_inc,
            'sparse_external':direct,'sparse_external_keephead':keephead,
            'gate_external':gated,'gate_inclusive':gated_inc,'extra_sgd':sgd}
        metrics={}
        for name,point in points.items():
            base.assign_vector(evaluator,point)
            metrics[name]=base.metrics(evaluator,devsets[i])
        rows.append({'client':i,'metrics':metrics,'gate_external':{**gd,**ls},
            'gate_inclusive':{**gid,**lsi},'extra_sgd':lsg,'train_batches':batch_ids,
            'available_external':int(valid.sum()),'seconds':time.perf_counter()-began})
        atomic(args.out/'partial.json', {'completed_clients':len(rows),'rows':rows})
        print(json.dumps({'stage':'fusion','client':i,'local_ce':metrics['local']['loss'],
            'gate_ce':metrics['gate_external']['loss'],'sgd_ce':metrics['extra_sgd']['loss']}),flush=True)
    summary=summarize(rows)
    c=summary['gate_external']
    controls=[summary[x] for x in ('local','extra_sgd')]
    go=all(c['client_macro_ce'] < x['client_macro_ce'] and
        c['client_macro_accuracy'] >= x['client_macro_accuracy']-.005 for x in controls)
    result={'status':'DONE','rows':rows,'summary':summary,'uploads':uploads,
        'decision':'GO_TO_INDEPENDENT_CHECK' if go else 'STOP_THIS_EXACT_EXTERNAL_GATE',
        'scope':'one shared development state, not long-run PFL or final test',
        'signature':config['signature']}
    atomic(args.out/'result.json',result)
    print(json.dumps({'summary':summary,'decision':result['decision']}),flush=True)


if __name__=='__main__':
    main()

"""Bounded 2x2 global-FL normalization/compression pilot. Development only."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'method_followup'))
import run_followup as r
import run_partitions as data
import run_accelerated as accel


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sparse(u,k,sections,normalize):
    score=u.abs().clone()
    if normalize:
        for s in sections:
            rms=u[s].double().square().mean().sqrt()
            if float(rms)>0:
                score[s]/=float(rms)
    ids=torch.argsort(score,descending=True,stable=True)[:k]
    sent=torch.zeros_like(u)
    sent[ids]=u[ids]
    residual=u-sent
    if not torch.equal(sent+residual,u):
        raise AssertionError('EF conservation')
    counts=[int(((ids>=s.start)&(ids<s.stop)).sum()) for s in sections]
    return sent,residual,counts


def run(args,method,train,dev,manifest):
    out=args.out/method
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed)
    model=r.SmallCNN()
    anchor=r.vectorize(model)
    p=len(anchor)
    k=int(p*.05)
    offset=0
    sections=[]
    names=[]
    for name,param in model.named_parameters():
        sections.append(slice(offset,offset+param.numel()))
        names.append(name)
        offset+=param.numel()
    weights=np.array([len(x) for x in train],dtype=float)
    weights/=weights.sum()
    taus=np.array([math.ceil(len(x)/64) for x in train],dtype=int)
    nova=method.startswith('nova')
    rms=method.endswith('rms')
    eff=float(weights@taus) if nova else 1.
    conf={'method':method,'rounds':args.rounds,'seed':args.seed,'backend':args.backend,
        'source_sha256':sha(__file__),'plan_sha256':sha(HERE/'PLAN.md'),
        'runtime':accel.runtime_metadata(args.backend,'normalized_sparse'),
        'data_manifest':manifest,'local_steps':taus.tolist(),'server_multiplier':eff,
        'parameter_count':p,'selected_coordinates':k,'lr':.03,'local_epoch':1,'batch':64,
        'normalization':'RMS per parameter tensor' if rms else 'none',
        'ef_units':'delta/local_steps' if nova else 'delta',
        'official_test_loaded':False,'scope':'single global model, simulated synchronous FL, development only'}
    sig=hashlib.sha256(json.dumps(conf,sort_keys=True).encode()).hexdigest()
    conf['signature']=sig
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())['signature']!=sig:
        raise ValueError('incompatible run')
    r.atomic_json(out/'config.json',conf)
    residuals=[torch.zeros_like(anchor) for _ in train]
    rows=[]
    if (out/'checkpoint.pt').exists():
        state=torch.load(out/'checkpoint.pt',weights_only=False)
        if state['signature']!=sig:
            raise ValueError('incompatible checkpoint')
        model.load_state_dict(state['model'])
        residuals,rows=state['residuals'],state['rows']
    for rnd in range(len(rows),args.rounds):
        began=time.perf_counter()
        anchor=r.vectorize(model)
        aggregate=torch.zeros_like(anchor)
        clients=[]
        for i,ds in enumerate(train):
            delta=accel.train_local(model,ds,args.seed+rnd*100+i,.03,1,backend=args.backend)
            if nova:
                delta/=int(taus[i])
            u=delta+residuals[i]
            sent,residuals[i],counts=sparse(u,k,sections,rms)
            aggregate+=float(weights[i])*sent
            clients.append({'client':i,'selected_by_tensor':counts,'candidate_norm':float(u.norm()),
                'residual_norm':float(residuals[i].norm()),'sent_norm':float(sent.norm())})
        r.assign_vector(model,anchor+eff*aggregate)
        metrics=r.metrics(model,dev)
        row={'round':rnd+1,'validation':metrics,'clients':clients,'tensor_names':names,
            'analytical_upload_bytes':len(train)*(64+8*k),
            'seconds':time.perf_counter()-began}
        rows.append(row)
        tmp=out/'checkpoint.tmp'
        torch.save({'model':model.state_dict(),'residuals':residuals,'rows':rows,'signature':sig},tmp)
        tmp.replace(out/'checkpoint.pt')
        r.atomic_json(out/'result.json',{'status':'DONE' if len(rows)==args.rounds else 'RUNNING',
            'rows':rows,'signature':sig,'test_evaluated':False})
        print(json.dumps({'method':method,'round':rnd+1,'acc':metrics['accuracy'],'ce':metrics['loss'],
            'seconds':row['seconds']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=20)
    p.add_argument('--seed',type=int,default=20261151)
    p.add_argument('--backend',choices=('cpu','mps'),default='mps')
    p.add_argument('--methods',default='topk,nova_topk,rms,nova_rms')
    args=p.parse_args()
    torch.set_num_threads(1)
    if args.backend=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no fallback')
    train,dev,manifest=data.prepare('Cifar10IndexA0p1',0,20261143,2000)
    for method in args.methods.split(','):
        if method not in ('topk','nova_topk','rms','nova_rms'):
            raise ValueError('unknown method')
        run(args,method,train,dev,manifest)


if __name__=='__main__':
    main()

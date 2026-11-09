"""Compression-aware two-half-step local training, finite global-FL pilot."""
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

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'method_followup'))
import run_followup as r
import run_partitions as data
import run_accelerated as accel


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def train(model,ds,seed,residual,k,method,backend):
    if backend=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no fallback')
    device=torch.device(backend)
    torch.manual_seed(seed)
    if backend=='mps':
        torch.mps.manual_seed(seed)
    anchor=r.vectorize(model).clone().to(device)
    rem=residual.to(device)
    local=copy.deepcopy(model).to(device).train()
    shadow=copy.deepcopy(model).to(device).train() if method=='shadow' else None
    lr=.03 if method=='standard' else .015
    opt=torch.optim.SGD(local.parameters(),lr=lr)
    loader=DataLoader(ds,batch_size=64,shuffle=True,
        generator=torch.Generator(device='cpu').manual_seed(seed))
    steps=gradients=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        opt.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(local(x),y).backward()
        opt.step();steps+=1;gradients+=1
        if method=='samebatch':
            opt.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(local(x),y).backward()
            opt.step();steps+=1;gradients+=1
        elif method=='shadow':
            current=torch.cat([p.detach().flatten() for p in local.parameters()])
            u=current-anchor+rem
            ids=torch.topk(u.abs(),k=k,sorted=False).indices
            mask=torch.zeros_like(u,dtype=torch.bool)
            mask[ids]=True
            q=torch.where(mask,u,torch.zeros_like(u))
            point=anchor+q
            offset=0
            with torch.no_grad():
                for p in shadow.parameters():
                    n=p.numel()
                    p.copy_(point[offset:offset+n].view_as(p))
                    offset+=n
            shadow.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(shadow(x),y).backward()
            opt.zero_grad(set_to_none=True)
            offset=0
            for p,sp in zip(local.parameters(),shadow.parameters()):
                n=p.numel()
                p.grad=sp.grad.detach().clone()*mask[offset:offset+n].view_as(p)
                offset+=n
            opt.step();steps+=1;gradients+=1
    delta=torch.cat([p.detach().flatten() for p in local.parameters()]).cpu()-anchor.cpu()
    if not torch.isfinite(delta).all():
        raise ValueError('nonfinite local delta')
    return delta,{'optimizer_steps':steps,'gradient_evaluations':gradients,'batches':len(loader)}


def run(args,method,trainsets,dev,manifest):
    out=args.out/method
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed)
    model=r.SmallCNN()
    anchor=r.vectorize(model)
    k=int(len(anchor)*.05)
    weights=np.array([len(x) for x in trainsets],dtype=float)
    weights/=weights.sum()
    conf={'method':method,'seed':args.seed,'rounds':args.rounds,'backend':args.backend,
        'source_sha256':sha(__file__),'plan_sha256':sha(HERE/'PLAN.md'),
        'data_manifest':manifest,'runtime':accel.runtime_metadata(args.backend,'shadow_training'),
        'local_epoch':1,'batch':64,'lr_per_step':.03 if method=='standard' else .015,
        'gradient_evaluations_per_batch':1 if method=='standard' else 2,
        'compressor':'global magnitude TopK, original values, EF in unnormalized model-delta units',
        'shadow_point':'anchor + TopK(theta_after_dense_half - anchor + previous_round_residual)',
        'shadow_gradient_projection':'current shadow support only; only shadow method uses this',
        'selected_coordinates':k,'parameter_count':len(anchor),'official_test_loaded':False,
        'evaluation':'stratified2000 officialTRAIN-holdout global development, simulation only'}
    sig=hashlib.sha256(json.dumps(conf,sort_keys=True).encode()).hexdigest()
    conf['signature']=sig
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())['signature']!=sig:
        raise ValueError('incompatible run')
    r.atomic_json(out/'config.json',conf)
    residuals=[torch.zeros_like(anchor) for _ in trainsets]
    rows=[]
    if (out/'checkpoint.pt').exists():
        ck=torch.load(out/'checkpoint.pt',weights_only=False)
        if ck['signature']!=sig:
            raise ValueError('checkpoint signature')
        model.load_state_dict(ck['model'])
        residuals,rows=ck['residuals'],ck['rows']
    for rnd in range(len(rows),args.rounds):
        began=time.perf_counter()
        anchor=r.vectorize(model)
        aggregate=torch.zeros_like(anchor)
        clients=[]
        for i,ds in enumerate(trainsets):
            delta,counts=train(model,ds,args.seed+rnd*100+i,residuals[i],k,method,args.backend)
            u=delta+residuals[i]
            ids=torch.argsort(u.abs(),descending=True,stable=True)[:k]
            q=torch.zeros_like(u)
            q[ids]=u[ids]
            residuals[i]=u-q
            if not torch.equal(q+residuals[i],u):
                raise AssertionError('EF conservation')
            aggregate+=float(weights[i])*q
            clients.append({'client':i,**counts,'residual_norm':float(residuals[i].norm()),
                'dense_delta_norm':float(delta.norm())})
        r.assign_vector(model,anchor+aggregate)
        m=r.metrics(model,dev)
        row={'round':rnd+1,'validation':m,'clients':clients,
            'analytical_upload_bytes':len(trainsets)*(64+8*k),'seconds':time.perf_counter()-began}
        rows.append(row)
        tmp=out/'checkpoint.tmp'
        torch.save({'model':model.state_dict(),'residuals':residuals,'rows':rows,'signature':sig},tmp)
        tmp.replace(out/'checkpoint.pt')
        r.atomic_json(out/'result.json',{'status':'DONE' if len(rows)==args.rounds else 'RUNNING',
            'rows':rows,'signature':sig,'test_evaluated':False})
        print(json.dumps({'method':method,'round':rnd+1,'acc':m['accuracy'],'ce':m['loss'],
            'seconds':row['seconds']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=20)
    p.add_argument('--seed',type=int,default=20261161)
    p.add_argument('--backend',choices=('cpu','mps'),default='mps')
    p.add_argument('--methods',default='standard,samebatch,shadow')
    args=p.parse_args()
    torch.set_num_threads(1)
    if args.backend=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no fallback')
    trainsets,dev,manifest=data.prepare('Cifar10IndexA0p1',0,20261143,2000)
    for method in args.methods.split(','):
        if method not in ('standard','samebatch','shadow'):
            raise ValueError('unknown method')
        run(args,method,trainsets,dev,manifest)


if __name__=='__main__':
    main()

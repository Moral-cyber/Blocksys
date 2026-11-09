"""Label calibration x fixed-byte sparse/quantized global-FL development."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib
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


def train(model,ds,seed,backend,calibrated):
    torch.manual_seed(seed)
    if backend=='mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; no fallback')
        torch.mps.manual_seed(seed)
    local=copy.deepcopy(model).to(backend).train()
    anchor=r.vectorize(model)
    counts=np.bincount(ds.labels[ds.indices],minlength=10)
    shift=torch.tensor(np.maximum(counts,1)**(-.25),dtype=torch.float32,device=backend)
    importance=torch.zeros(len(anchor),dtype=torch.float32,device=backend)
    loader=DataLoader(ds,batch_size=64,shuffle=True,
        generator=torch.Generator(device='cpu').manual_seed(seed))
    opt=torch.optim.SGD(local.parameters(),lr=.03)
    for x,y in loader:
        x,y=x.to(backend),y.to(backend)
        opt.zero_grad(set_to_none=True)
        logits=local(x)
        if calibrated:
            logits=logits-shift
        nn.functional.cross_entropy(logits,y).backward()
        with torch.no_grad():
            grad=torch.cat([p.grad.detach().flatten() for p in local.parameters()])
            importance.add_(grad.square())
        opt.step()
    delta=torch.cat([p.detach().flatten() for p in local.parameters()]).cpu()-anchor
    importance=(importance/len(loader)).cpu()
    if not torch.isfinite(delta).all() or not torch.isfinite(importance).all():
        raise ValueError('nonfinite delta or gradient second moment')
    return delta,importance,{'optimizer_steps':len(loader),'gradient_evaluations':len(loader),
        'train_label_counts':counts.tolist(),'calibration_shift':shift.cpu().tolist() if calibrated else None}


def run(args,method,trainsets,dev,manifest):
    calibrated=method.startswith('cal_')
    compression=method.split('_',1)[1]
    quant=compression in ('uniform3','mixed_mse','mixed_sensitive')
    codec=importlib.import_module('codec') if quant else None
    if quant and Path(codec.__file__).resolve()!=HERE/'codec.py':
        raise RuntimeError('wrong quantizer module')
    out=args.out/method
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed)
    model=r.SmallCNN()
    anchor=r.vectorize(model)
    k=int(.05*len(anchor));budget=64+8*k
    weights=np.array([len(ds) for ds in trainsets],dtype=float);weights/=weights.sum()
    conf={'method':method,'seed':args.seed,'rounds':args.rounds,'backend':args.backend,
        'source_sha256':sha(__file__),'plan_sha256':sha(HERE/'EXPERIMENT_PLAN.md'),
        'quantizer_sha256':sha(HERE/'codec.py') if quant else None,
        'runtime':accel.runtime_metadata(args.backend,'calibrated_mixed_precision'),
        'data_manifest':manifest,'local_epoch':1,'batch':64,'lr':.03,
        'calibration':'all-class CE(logits-max(count,1)^(-1/4)), tau1; raw logits inference' if calibrated else 'standard CE',
        'calibration_scope':'FedLC-inspired explicit implementation variant, not official replication',
        'importance':'mean across local batches of squared batch-mean SGD gradient; not per-example Fisher/Hessian',
        'parameter_count':len(anchor),'topk':k,'application_budget_bytes':budget,
        'communication':'quantizer actual generated+decoded wire; TopK/dense analytical bytes; no actual network/chain',
        'official_test_loaded':False,'evaluation':'2000 stratified TRAIN-holdout global development only'}
    sig=hashlib.sha256(json.dumps(conf,sort_keys=True).encode()).hexdigest();conf['signature']=sig
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())['signature']!=sig:
        raise ValueError('incompatible output')
    r.atomic_json(out/'config.json',conf)
    residuals=[torch.zeros_like(anchor) for _ in trainsets];rows=[]
    if (out/'checkpoint.pt').exists():
        ck=torch.load(out/'checkpoint.pt',weights_only=False)
        if ck['signature']!=sig:raise ValueError('wrong checkpoint signature')
        model.load_state_dict(ck['model']);residuals,rows=ck['residuals'],ck['rows']
    for rnd in range(len(rows),args.rounds):
        began=time.perf_counter();anchor=r.vectorize(model);aggregate=torch.zeros_like(anchor)
        clients=[];wirebytes=0
        for i,ds in enumerate(trainsets):
            delta,importance,counters=train(model,ds,args.seed+rnd*100+i,args.backend,calibrated)
            u=delta+residuals[i]
            encbegan=time.perf_counter()
            if compression=='dense':
                q=delta;residuals[i]=torch.zeros_like(delta);nbytes=64+4*len(anchor);diag={}
            elif compression=='topk':
                ids=torch.argsort(u.abs(),descending=True,stable=True)[:k]
                q=torch.zeros_like(u);q[ids]=u[ids];residuals[i]=u-q;nbytes=budget;diag={}
            else:
                encoded=codec.encode(u,budget,mode=compression,
                    importance=importance if compression=='mixed_sensitive' else None)
                q,residuals[i]=encoded.sent,encoded.residual;nbytes=len(encoded.wire);diag=encoded.diagnostics
                if nbytes>budget or not torch.equal(codec.decode(encoded.wire),q):
                    raise AssertionError('quantizer bytes or wire reconstruction')
            encsec=time.perf_counter()-encbegan
            aggregate+=float(weights[i])*q;wirebytes+=nbytes
            clients.append({'client':i,**counters,'payload_bytes':nbytes,'codec_seconds':encsec,
                'residual_norm':float(residuals[i].norm()),'delta_norm':float(delta.norm()),
                'gradient_second_moment_mean':float(importance.mean()),'codec':diag})
        r.assign_vector(model,anchor+aggregate);m=r.metrics(model,dev)
        row={'round':rnd+1,'validation':m,'clients':clients,'upload_bytes':wirebytes,
            'byte_accounting':'generated decoded quantizer bytes' if quant else 'analytical fixed layout',
            'seconds':time.perf_counter()-began}
        rows.append(row);tmp=out/'checkpoint.tmp'
        torch.save({'model':model.state_dict(),'residuals':residuals,'rows':rows,'signature':sig},tmp)
        tmp.replace(out/'checkpoint.pt')
        r.atomic_json(out/'result.json',{'status':'DONE' if len(rows)==args.rounds else 'RUNNING',
            'rows':rows,'signature':sig,'test_evaluated':False})
        print(json.dumps({'method':method,'round':rnd+1,'acc':m['accuracy'],'ce':m['loss'],
            'bytes':wirebytes,'seconds':row['seconds']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=20261181)
    p.add_argument('--rounds',type=int,default=20);p.add_argument('--backend',choices=('cpu','mps'),default='mps')
    p.add_argument('--methods',default='ce_dense,ce_topk,cal_dense,cal_topk')
    args=p.parse_args();torch.set_num_threads(1)
    if args.backend=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no fallback')
    trainsets,dev,manifest=data.prepare('Cifar10IndexA0p1',0,20261143,2000)
    allowed={prefix+'_'+mode for prefix in ('ce','cal') for mode in ('dense','topk','uniform3','mixed_mse','mixed_sensitive')}
    for method in args.methods.split(','):
        if method not in allowed:raise ValueError('unknown method')
        run(args,method,trainsets,dev,manifest)


if __name__=='__main__':main()

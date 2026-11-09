"""Explicit local label-prior loss + frozen low-bit EF components, development."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
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
spec=importlib.util.spec_from_file_location('_balanced_frozen_codec',HERE.parent/'mixed_precision/codec.py')
codec=importlib.util.module_from_spec(spec);sys.modules[spec.name]=codec;spec.loader.exec_module(codec)
METHODS=('ce_uniform3','fedlc_uniform3','balanced_uniform3','balanced_dense')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def shift_for(counts,loss):
    if loss=='ce':return np.zeros(10)
    if loss=='fedlc':return -np.maximum(counts,1)**(-.25)
    if loss=='balanced':return np.log(counts+1)
    raise ValueError('unknown local objective')


def train(model,ds,seed,backend,loss):
    torch.manual_seed(seed)
    if backend=='mps':
        if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no fallback')
        torch.mps.manual_seed(seed)
    local=copy.deepcopy(model).to(backend).train();anchor=r.vectorize(model)
    counts=np.bincount(ds.labels[ds.indices],minlength=10)
    shift=torch.tensor(shift_for(counts,loss),dtype=torch.float32,device=backend)
    loader=DataLoader(ds,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
    opt=torch.optim.SGD(local.parameters(),lr=.03)
    for x,y in loader:
        opt.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(local(x.to(backend))+shift,y.to(backend)).backward();opt.step()
    delta=torch.cat([p.detach().flatten() for p in local.parameters()]).cpu()-anchor
    if not torch.isfinite(delta).all() or not torch.equal(r.vectorize(model),anchor):raise ValueError('nonfinite update or altered anchor')
    return delta,{'steps':len(loader),'gradient_evaluations':len(loader),'train_counts':counts.tolist(),
        'logit_shift':shift.cpu().tolist()}


def run(args,method,trainsets,dev,manifest):
    loss,compression=method.split('_',1);dense=compression=='dense'
    torch.manual_seed(args.seed);model=r.SmallCNN();anchor=r.vectorize(model)
    p=np.array([len(ds) for ds in trainsets],dtype=float);p/=p.sum()
    conf={'method':method,'seed':args.seed,'partition_seed':args.partition_seed,'rounds':args.rounds,
        'backend':args.backend,'source_sha256':sha(__file__),'plan_sha256':sha(HERE/'PLAN.md'),
        'codec_sha256':sha(HERE.parent/'mixed_precision/codec.py'),'runtime':accel.runtime_metadata(args.backend,'balanced_lowbit_components'),
        'data_manifest':manifest,'weights':p.tolist(),'batch':64,'lr':.03,'local_epochs':1,
        'loss_rule':{'ce':'ordinary CE','fedlc':'CE(z-max(n,1)^(-1/4)); inspired explicit variant, tau1',
            'balanced':'CE(z+log(n+1)); Dirichlet1-smoothed balanced softmax explicit variant'}[loss],
        'inference':'raw logits','parameter_count':len(anchor),'upload_cap_bytes':None if dense else 54808,
        'communication':'quantized actual wire decoded for aggregation; dense upload/all downlinks analytical; no network/chain',
        'official_test_loaded':False,'evaluation':'stratified official TRAIN-holdout global development'}
    sig=hashlib.sha256(json.dumps(conf,sort_keys=True).encode()).hexdigest();conf['signature']=sig
    out=args.out/method;out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())['signature']!=sig:raise ValueError('incompatible output')
    residuals=[torch.zeros_like(anchor) for _ in trainsets];rows=[]
    if (out/'checkpoint.pt').exists():
        ck=torch.load(out/'checkpoint.pt',weights_only=False)
        if ck['signature']!=sig:raise ValueError('incompatible checkpoint')
        model.load_state_dict(ck['model']);residuals,rows=ck['residuals'],ck['rows']
    r.atomic_json(out/'config.json',conf)
    for rnd in range(len(rows),args.rounds):
        began=time.perf_counter();anchor=r.vectorize(model);aggregate=torch.zeros_like(anchor);clients=[]
        for i,ds in enumerate(trainsets):
            delta,counters=train(model,ds,args.seed+rnd*100+i,args.backend,loss)
            if dense:q=delta;residuals[i].zero_();nbytes=64+4*len(anchor);eferr=0.
            else:
                u=delta+residuals[i];enc=codec.encode(u,54808,mode='uniform3');q=codec.decode(enc.wire)
                if not torch.equal(q,enc.sent):raise ValueError('decoder mismatch')
                residuals[i]=u-q;nbytes=len(enc.wire);eferr=float((q+residuals[i]-u).abs().max())
                if nbytes>54808:raise ValueError('over byte cap')
                if rnd==0 and i==0:(out/'first_payload.bin').write_bytes(enc.wire)
            aggregate.add_(q,alpha=float(p[i]));clients.append({'client':i,**counters,
                'upload_bytes':nbytes,'ef_max_error':eferr,'residual_norm':float(residuals[i].norm())})
        r.assign_vector(model,anchor+aggregate);m=r.metrics(model,dev)
        rows.append({'round':rnd+1,'validation':m,'clients':clients,'upload_bytes':sum(v['upload_bytes'] for v in clients),
            'download_bytes_analytical':len(trainsets)*(64+4*len(anchor)), 'seconds':time.perf_counter()-began})
        tmp=out/'checkpoint.tmp';torch.save({'model':model.state_dict(),'residuals':residuals,'rows':rows,'signature':sig},tmp);tmp.replace(out/'checkpoint.pt')
        r.atomic_json(out/'result.json',{'status':'DONE' if len(rows)==args.rounds else 'RUNNING','rows':rows,'signature':sig,'official_test_loaded':False})
        print(json.dumps({'method':method,'round':rnd+1,'accuracy':m['accuracy'],'ce':m['loss'],'seconds':rows[-1]['seconds']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--backend',choices=('mps','cpu'),default='mps')
    p.add_argument('--seed',type=int,default=20261201);p.add_argument('--partition-seed',type=int,default=20261193)
    p.add_argument('--rounds',type=int,default=20);p.add_argument('--methods',default=','.join(METHODS));args=p.parse_args()
    torch.set_num_threads(1)
    if args.backend=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no fallback')
    trainsets,dev,manifest=data.prepare('Cifar10IndexA0p1',0,args.partition_seed,2000)
    for method in args.methods.split(','):
        if method not in METHODS:raise ValueError('unknown method')
        run(args,method,trainsets,dev,manifest)


if __name__=='__main__':main()

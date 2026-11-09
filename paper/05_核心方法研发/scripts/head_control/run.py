"""Bounded head-subspace SCAFFOLD + exact head / quantized body diagnostic."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import struct
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
spec=importlib.util.spec_from_file_location('_frozen_head_body_codec',HERE.parent/'mixed_precision/codec.py')
codec=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=codec
spec.loader.exec_module(codec)
HEADER=struct.Struct('<8sIIIIII')
MAGIC=b'HEADQ001'
CAP=54808
LR=.03
METHODS=('plain_uniform','exact_head','head_control','head_control_dense')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def head_size(model):
    names=list(model.named_parameters())
    if [name for name,_ in names[-2:]]!=['classifier.3.weight','classifier.3.bias']:
        raise ValueError('unexpected head layout')
    return sum(p.numel() for _,p in names[-2:])


def train(model,ds,seed,backend,ci,c):
    torch.manual_seed(seed)
    if backend=='mps':
        if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no fallback')
        torch.mps.manual_seed(seed)
    local=copy.deepcopy(model).to(backend).train()
    anchor=r.vectorize(model);h=head_size(model)
    correction=(c-ci).to(backend) if c is not None else None
    sumgrad=torch.zeros(h,device=backend)
    loader=DataLoader(ds,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
    opt=torch.optim.SGD(local.parameters(),lr=LR)
    for x,y in loader:
        opt.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(local(x.to(backend)),y.to(backend)).backward()
        with torch.no_grad():
            head=list(local.parameters())[-2:]
            raw=torch.cat([p.grad.flatten() for p in head]);sumgrad.add_(raw)
            if correction is not None:
                offset=0
                for p in head:
                    p.grad.add_(correction[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
        opt.step()
    delta=torch.cat([p.detach().flatten() for p in local.parameters()]).cpu()-anchor
    k=len(loader)
    rawmean=sumgrad.cpu()/k
    newci=ci-c-delta[-h:]/(LR*k) if c is not None else None
    discrepancy=float((newci-rawmean).abs().max()) if newci is not None else 0.
    if not torch.isfinite(delta).all() or discrepancy>5e-6:
        raise ValueError('nonfinite update or head-control telescoping mismatch')
    return delta,newci,k,discrepancy


def encode_head(u,dh,k,budget=CAP):
    enc=codec.encode(u,budget-HEADER.size-4*len(dh),mode='mixed_mse')
    head=dh.numpy().astype('<f4',copy=False).tobytes()
    wire=HEADER.pack(MAGIC,1,len(u),len(dh),k,len(enc.wire),0)+enc.wire+head
    qb,qh,decoded_k=decode_head(wire)
    if len(wire)>budget or decoded_k!=k or not torch.equal(qh,dh) or not torch.equal(qb,enc.sent):
        raise AssertionError('wire size or reconstruction failure')
    return torch.cat((qb,qh)),u-qb,wire,enc.diagnostics


def decode_head(wire):
    if len(wire)<HEADER.size:raise ValueError('short header')
    magic,version,p,h,k,n,reserved=HEADER.unpack_from(wire)
    if magic!=MAGIC or version!=1 or reserved or k<1 or h!=650 or p!=136224:
        raise ValueError('wrong wrapper metadata')
    if len(wire)!=HEADER.size+n+4*h:raise ValueError('wrong packet length')
    body=codec.decode(wire[HEADER.size:HEADER.size+n])
    if body.numel()!=p:raise ValueError('wrong body shape')
    head=torch.from_numpy(np.frombuffer(wire,dtype='<f4',offset=HEADER.size+n,count=h).copy())
    if not torch.isfinite(head).all():raise ValueError('nonfinite head')
    return body,head,k


def run(args,method,trainsets,dev,manifest):
    controlled=method.startswith('head_control');dense=method.endswith('_dense')
    torch.manual_seed(args.seed);model=r.SmallCNN();anchor=r.vectorize(model)
    h=head_size(model);body=len(anchor)-h
    out=args.out/method;out.mkdir(parents=True,exist_ok=True)
    weights=np.array([len(ds) for ds in trainsets],dtype=float);weights/=weights.sum()
    conf={'method':method,'seed':args.seed,'rounds':args.rounds,'backend':args.backend,
        'source_sha256':sha(__file__),'plan_sha256':sha(HERE/'PLAN.md'),
        'codec_sha256':sha(HERE.parent/'mixed_precision/codec.py'),
        'runtime':accel.runtime_metadata(args.backend,'head_subspace_control'),
        'manifest':manifest,'body':body,'head':h,'lr':LR,'batch':64,'local_epochs':1,
        'aggregation':'full participation, fixed sample weights','weights':weights.tolist(),
        'upload_cap_bytes':None if dense else CAP,'official_test_loaded':False,
        'evaluation':'official TRAIN stratified 2000 global development holdout',
        'communication':'actual serialized+decoded quantized upload; dense uploads and all downlinks analytical, no network/chain',
        'control_scope':'last classifier weight+bias only; existing SCAFFOLD/SCALLION principle',
        'control_client_state_bytes':h*4 if controlled else 0}
    sig=hashlib.sha256(json.dumps(conf,sort_keys=True).encode()).hexdigest();conf['signature']=sig
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())['signature']!=sig:
        raise ValueError('incompatible source/config output')
    r.atomic_json(out/'config.json',conf)
    residuals=[torch.zeros(len(anchor) if method=='plain_uniform' else body) for _ in trainsets]
    cis=[torch.zeros(h) for _ in trainsets];c=torch.zeros(h);rows=[]
    if (out/'checkpoint.pt').exists():
        ck=torch.load(out/'checkpoint.pt',weights_only=False)
        if ck['signature']!=sig:raise ValueError('wrong checkpoint signature')
        model.load_state_dict(ck['model']);residuals,cis,c,rows=ck['residuals'],ck['cis'],ck['c'],ck['rows']
    for rnd in range(len(rows),args.rounds):
        began=time.perf_counter();anchor=r.vectorize(model);aggregate=torch.zeros_like(anchor)
        clients=[];wirebytes=0;newcis=[];direct_c=torch.zeros(h)
        for i,ds in enumerate(trainsets):
            delta,ci,k,err=train(model,ds,args.seed+rnd*100+i,args.backend,cis[i] if controlled else None,c if controlled else None)
            codec_start=time.perf_counter()
            if dense:
                q=delta;residuals[i].zero_();nbytes=64+4*len(anchor);diag={};eferr=0.
            elif method=='plain_uniform':
                u=delta+residuals[i];enc=codec.encode(u,CAP,mode='uniform3')
                q=codec.decode(enc.wire);residuals[i]=u-q;nbytes=len(enc.wire);diag=enc.diagnostics
                eferr=float((q+residuals[i]-u).abs().max())
                wire=enc.wire
            else:
                u=delta[:body]+residuals[i]
                q,residuals[i],wire,diag=encode_head(u,delta[body:],k)
                nbytes=len(wire);eferr=float((q[:body]+residuals[i]-u).abs().max())
            if not dense and nbytes>CAP:raise ValueError('upload cap exceeded')
            if controlled:
                newcis.append(ci);direct_c.add_(q[-h:]/(-LR*k),alpha=float(weights[i]))
            aggregate.add_(q,alpha=float(weights[i]));wirebytes+=nbytes
            # A real packet example per method, not a claim of saving all historical wire.
            if rnd==0 and i==0 and not dense:(out/'first_payload.bin').write_bytes(wire)
            clients.append({'client':i,'steps':k,'gradient_evaluations':k,'upload_bytes':nbytes,
                'codec_seconds':time.perf_counter()-codec_start,'control_formula_max_error':err,
                'body_ef_max_error':eferr,'residual_norm':float(residuals[i].norm()),
                'head_delta_norm':float(delta[-h:].norm()),'codec':diag})
        invariant=0.
        if controlled:
            weighted_c=sum((float(w)*ci for w,ci in zip(weights,newcis)),torch.zeros(h))
            invariant=float((weighted_c-direct_c).abs().max())
            if invariant>5e-6:raise ValueError('weighted control invariant broken')
            cis,c=newcis,weighted_c
        r.assign_vector(model,anchor+aggregate);m=r.metrics(model,dev)
        down_per_client=64+4*len(anchor)+(16+4*h if controlled else 0)
        rows.append({'round':rnd+1,'validation':m,'clients':clients,'upload_bytes':wirebytes,
            'download_bytes_analytical':len(trainsets)*down_per_client,
            'control_invariant_error':invariant,'control_norm':float(c.norm()),
            'seconds':time.perf_counter()-began})
        tmp=out/'checkpoint.tmp';torch.save({'model':model.state_dict(),'residuals':residuals,
            'cis':cis,'c':c,'rows':rows,'signature':sig},tmp);tmp.replace(out/'checkpoint.pt')
        r.atomic_json(out/'result.json',{'status':'DONE' if len(rows)==args.rounds else 'RUNNING',
            'rows':rows,'signature':sig,'official_test_loaded':False})
        print(json.dumps({'method':method,'round':rnd+1,'accuracy':m['accuracy'],'ce':m['loss'],
            'upload_bytes':wirebytes,'seconds':rows[-1]['seconds']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=20261191)
    p.add_argument('--rounds',type=int,default=20);p.add_argument('--backend',choices=('mps','cpu'),default='mps')
    p.add_argument('--methods',default=','.join(METHODS))
    args=p.parse_args();torch.set_num_threads(1)
    if args.backend=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no fallback')
    trainsets,dev,manifest=data.prepare('Cifar10IndexA0p1',0,20261143,2000)
    for method in args.methods.split(','):
        if method not in METHODS:raise ValueError('unknown method')
        run(args,method,trainsets,dev,manifest)


if __name__=='__main__':main()

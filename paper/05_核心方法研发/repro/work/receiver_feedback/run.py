"""Receiver-feedback pack selection: finite exploratory PFL runs, no test access."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'partial_fusion'))
import probe_fusion as probe
base,data,accel=probe.base,probe.data,probe.accel


def encode_model(vector,chosen,packs,rnd):
    chosen=sorted(int(x) for x in chosen)
    output=bytearray(struct.pack('<4I',rnd,len(vector),128,len(chosen)))
    for j in chosen:
        output.extend(struct.pack('<I',j))
        output.extend(vector[packs[j]].numpy().astype('<f4',copy=False).tobytes())
    return bytes(output)


def encode_feedback(bits,rnd):
    packed=np.packbits(np.asarray(bits,dtype=np.uint8),bitorder='little')
    return struct.pack('<4I',rnd,len(bits),1,len(packed))+packed.tobytes()


def choose(model,local,grad,direction,check,method):
    step=-.03*grad
    if method=='local_sgd':
        points=[local,local+step,local+.5*step,local+.25*step]
    else:
        points=[local,local+step,local+.5*direction,local+step+.5*direction]
    losses=[probe.ce(model,p,check) for p in points]
    selected=min(range(4),key=lambda j:losses[j])
    return points[selected],selected,losses


def run_one(args,method,trainsets,devsets,anchor,packs,weights,manifest):
    out=args.out/method
    out.mkdir(parents=True,exist_ok=True)
    warm=HERE.parent/'partial_fusion/dev_seed20261143/warmup.pt'
    conf={'method':method,'seed':args.seed,'rounds':args.rounds,'backend':args.backend,
        'warmup_sha256':probe.sha(warm),'source_sha256':probe.sha(__file__),
        'plan_sha256':probe.sha(HERE/'PLAN.md'),'probe_source_sha256':probe.sha(probe.__file__),
        'runtime':accel.runtime_metadata(args.backend,'receiver_feedback'),
        'data_manifest':manifest,'protocol':'personalized models, absolute-value packs, no EF',
        'evaluation':'real labels, synchronous simulation, full client TRAIN holdout for development',
        'final_test_loaded':False,'transmission':'actual serialized application bytes, no sockets/blockchain/authentication',
        'pack_size':128,'model_packages':52 if method=='receiver_feedback' else (53 if method=='magnitude_gate' else 0),
        'feedback_ema':.25,'utility_prior':.5,'utility_score_floor':.1,
        'local_epoch':1,'lr':.03,'batch':64}
    signature=hashlib.sha256(json.dumps(conf,sort_keys=True).encode()).hexdigest()
    conf['signature']=signature
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())['signature']!=signature:
        raise ValueError('incompatible output '+str(out))
    probe.atomic(out/'config.json',conf)
    model=base.SmallCNN().cpu().eval()
    models=[anchor.clone() for _ in trainsets]
    utility=np.full((len(trainsets),len(packs)),.5,dtype=np.float64)
    rows=[]
    warm_bytes=10*len(trainsets)*2*(16+4*len(anchor))
    total_up=total_down=0
    if (out/'checkpoint.pt').exists():
        ck=torch.load(out/'checkpoint.pt',weights_only=False)
        if ck['signature']!=signature:
            raise ValueError('checkpoint signature mismatch')
        models,utility,rows=ck['models'],ck['utility'],ck['rows']
        total_up,total_down=ck['total_up'],ck['total_down']
    for rnd in range(len(rows),args.rounds):
        began=time.perf_counter()
        trained,masks,uploads=[],[],[]
        payload_hashes=[]
        up=down=0
        for i,ds in enumerate(trainsets):
            base.assign_vector(model,models[i])
            delta=accel.train_local(model,ds,args.seed+100000+rnd*100+i,.03,1,backend=args.backend)
            local=models[i]+delta
            trained.append(local)
            if method=='local_sgd':
                continue
            energy=np.array([float(delta[s].double().square().sum()) for s in packs])
            score=energy*(.1+.9*utility[i]) if method=='receiver_feedback' else energy
            k=52 if method=='receiver_feedback' else 53
            chosen=np.argsort(-score,kind='stable')[:k]
            mask=torch.zeros_like(anchor,dtype=torch.bool)
            for j in chosen:
                mask[packs[int(j)]]=True
            masks.append(mask)
            wire=encode_model(local,chosen,packs,rnd)
            up+=len(wire)
            uploads.append(chosen.tolist())
            payload_hashes.append(hashlib.sha256(wire).hexdigest())
        if method!='local_sgd':
            donor,valid=probe.donor_average(trained,masks,weights)
            validpacks=np.array([bool(valid[s].any()) for s in packs])
            union=np.flatnonzero(validpacks)
            wire=encode_model(donor,union,packs,rnd)
            down=len(wire)*len(trainsets)
        else:
            donor=torch.zeros_like(anchor)
            valid=torch.zeros_like(anchor,dtype=torch.bool)
            validpacks=np.zeros(len(packs),dtype=bool)
        accepted=np.zeros((len(trainsets),len(packs)),dtype=bool)
        client_rows=[]
        for i,local in enumerate(trained):
            fit,check,batch_ids=probe.train_batches(trainsets[i],args.seed+200000+rnd*100+i)
            grad=probe.gradient(model,local,fit)
            if method=='local_sgd':
                direction=torch.zeros_like(local)
            else:
                direction,_=probe.gate_direction(local,donor,valid,grad,packs)
            chosen,choice,losses=choose(model,local,grad,direction,check,method)
            models[i]=chosen.detach().clone()
            if method!='local_sgd' and choice in (2,3):
                for j,s in enumerate(packs):
                    other=any(bool(m[s].any()) for z,m in enumerate(masks) if z!=i)
                    accepted[i,j]=other and bool(direction[s].abs().sum()>0)
            if method=='receiver_feedback':
                up+=len(encode_feedback(accepted[i],rnd))
            base.assign_vector(model,models[i])
            m=base.metrics(model,devsets[i])
            client_rows.append({'client':i,'accuracy':m['accuracy'],'ce':m['loss'],'samples':m['samples'],
                'choice':choice,'training_candidate_ce':losses,'accepted_packs':int(accepted[i].sum()),
                'train_batches':batch_ids})
        if method=='receiver_feedback':
            for i in range(len(trainsets)):
                obs=(accepted.sum(axis=0)-accepted[i].astype(int))/(len(trainsets)-1)
                utility[i,validpacks]=.75*utility[i,validpacks]+.25*obs[validpacks]
        total_up+=up
        total_down+=down
        n=np.array([r['samples'] for r in client_rows])
        acc=np.array([r['accuracy'] for r in client_rows])
        loss=np.array([r['ce'] for r in client_rows])
        row={'round':rnd+1,'client_macro_accuracy':float(acc.mean()),'client_macro_ce':float(loss.mean()),
            'sample_weighted_accuracy':float(np.average(acc,weights=n)),
            'sample_weighted_ce':float(np.average(loss,weights=n)),
            'upload_bytes':up,'download_bytes_unicast':down,'cumulative_upload_bytes':total_up,
            'cumulative_download_bytes_unicast':total_down,'common_warmup_bytes_unicast':warm_bytes,
            'clients':client_rows,'uploads':uploads,'upload_payload_sha256':payload_hashes,
            'accepted_masks_packbits_hex':[np.packbits(x,bitorder='little').tobytes().hex() for x in accepted],
            'seconds':time.perf_counter()-began}
        rows.append(row)
        tmp=out/'checkpoint.tmp'
        torch.save({'models':models,'utility':utility,'rows':rows,'total_up':total_up,
            'total_down':total_down,'signature':signature},tmp)
        tmp.replace(out/'checkpoint.pt')
        probe.atomic(out/'result.json',{'status':'DONE' if len(rows)==args.rounds else 'RUNNING','rows':rows,
            'signature':signature,'scope':'single partition/seed exploratory PFL; no final test'})
        print(json.dumps({'method':method,**{k:row[k] for k in ('round','client_macro_accuracy','client_macro_ce',
            'upload_bytes','download_bytes_unicast','seconds')}}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--backend',choices=('cpu','mps'),default='mps')
    p.add_argument('--rounds',type=int,default=20)
    p.add_argument('--seed',type=int,default=20261144)
    p.add_argument('--methods',default='local_sgd,magnitude_gate,receiver_feedback')
    args=p.parse_args()
    torch.set_num_threads(1)
    if args.backend=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable; no fallback')
    original=HERE.parent/'partial_fusion/dev_seed20261143'
    source=json.loads((original/'config.json').read_text())
    images,labels,provenance=data.load_official_train()
    tr,dev,_,manifest=data.partition_indices(labels,alpha=.1,split_seed=20261143,validation_size=2000)
    if manifest['train_indices_sha256']!=source['data_manifest']['train_indices_sha256']:
        raise ValueError('warmup/new run TRAIN membership mismatch')
    if manifest['holdout_indices_sha256']!=source['data_manifest']['holdout_indices_sha256']:
        raise ValueError('warmup/new run dev membership mismatch')
    manifest.update(provenance)
    manifest['development_evaluation']='entire 12500 client holdout, inherited 2000 subset unused'
    ck=torch.load(original/'warmup.pt',weights_only=False)
    if ck['signature']!=source['signature'] or ck['round']!=10:
        raise ValueError('wrong common warmup identity')
    model=base.SmallCNN()
    model.load_state_dict(ck['model'])
    anchor=base.vectorize(model)
    packs=probe.pack_slices(len(anchor),128)
    trainsets=[data.IndexedCifar10(images,labels,x) for x in tr]
    devsets=[data.IndexedCifar10(images,labels,x) for x in dev]
    weights=np.array([len(ds) for ds in trainsets],dtype=float)
    weights/=weights.sum()
    for method in args.methods.split(','):
        if method not in ('local_sgd','magnitude_gate','receiver_feedback'):
            raise ValueError('unknown method')
        run_one(args,method,trainsets,devsets,anchor,packs,weights,manifest)


if __name__=='__main__':
    main()

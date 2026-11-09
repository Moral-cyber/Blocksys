"""Frozen-state exposure-prior probe on official TRAIN development only."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'method_followup'))
import run_followup as r
import run_partitions as data

MODELS={
 'hetero_dense':'bottleneck_reanchor/mixing0/dense_mix0_rho0.05_s20261171',
 'hetero_topk':'bottleneck_reanchor/mixing0/topk_ef_mix0_rho0.05_s20261171',
 'mixed_dense':'bottleneck_reanchor/mixing1/dense_mix1_rho0.05_s20261171',
 'mixed_topk':'bottleneck_reanchor/mixing1/topk_ef_mix1_rho0.05_s20261171',
 'additional_dense':'control_reference/development_s20261191/plain_dense'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def priors(manifest):
    counts=np.array(manifest['client_label_counts'],dtype=float)
    n=np.array(manifest['client_sizes'],dtype=float)
    if not np.array_equal(counts.sum(1),n):raise ValueError('class and client sample count mismatch')
    p=n/n.sum();k=np.ceil(n/64)
    pooled=counts.sum(0)/n.sum()
    effective=((p*k)[:,None]*counts/n[:,None]).sum(0)/(p*k).sum()
    if min(pooled.min(),effective.min())<=0:raise ValueError('globally missing class')
    d=np.log(pooled/effective)
    return pooled,effective,d


def metrics(z,y):
    loss=float(torch.nn.functional.cross_entropy(z,y))
    pred=z.argmax(1);counts=torch.bincount(y,minlength=10)
    hits=torch.bincount(y[pred==y],minlength=10)
    acc=hits.double()/counts
    return {'loss':loss,'accuracy':float((pred==y).double().mean()),
        'class_accuracy':acc.tolist(),'worst_class_accuracy':float(acc.min()),'samples':len(y)}


def main():
    torch.set_num_threads(1)
    out=HERE/'diagnostic_v1'
    if out.exists():raise ValueError('fresh output required; existing evidence retained')
    out.mkdir()
    inputs={};results={}
    with torch.no_grad():
        for name,relative in MODELS.items():
            source=HERE.parent/relative
            config=json.loads((source/'config.json').read_text())
            manifest=config.get('manifest',config.get('data_manifest'))
            for file in ('config.json','checkpoint.pt'):inputs[str(source/file)]=sha(source/file)
            _,dev,current=data.prepare('Cifar10IndexA0p1',manifest['mixing'],20261143,2000)
            for key in ('train_indices_sha256','validation_ids_sha256','assignment_label_sha256'):
                if current[key]!=manifest[key]:raise ValueError('data identity mismatch')
            ck=torch.load(source/'checkpoint.pt',weights_only=False)
            model=r.SmallCNN();model.load_state_dict(ck['model']);model.eval()
            zs=[];ys=[]
            for x,y in DataLoader(dev,batch_size=256):zs.append(model(x));ys.append(y)
            z=torch.cat(zs);y=torch.cat(ys)
            pooled,effective,d=priors(manifest)
            shifts={'raw':np.zeros(10),'pooled_to_uniform':-np.log(pooled),
                'effective_to_pooled':d,'wrong_direction':-d}
            values={key:metrics(z+torch.tensor(shift,dtype=z.dtype),y) for key,shift in shifts.items()}
            np.savez_compressed(out/(name+'.npz'),logits=z.numpy(),labels=y.numpy(),
                pooled=pooled,effective=effective,shift=d)
            results[name]={'source':str(source),'config':str(source/'config.json'),
                'evaluation':'official TRAIN-holdout development, final round only',
                'pooled':pooled.tolist(),'effective':effective.tolist(),'shift':d.tolist(),
                'shift_norm':float(np.linalg.norm(d)), 'metrics':values,
                'raw_saved_final':ck['rows'][-1]['validation'],
                'development_id_sha256':current['validation_ids_sha256']}
    changed=[p for p,h in inputs.items() if sha(p)!=h]
    if changed:raise ValueError('frozen model/config changed')
    gates={}
    for name in ('hetero_dense','hetero_topk'):
        a=results[name]['metrics'];raw=a['raw'];new=a['effective_to_pooled'];wrong=a['wrong_direction']
        gates[name]={'ce':new['loss']<raw['loss'],'accuracy':new['accuracy']>=raw['accuracy']+.005,
            'direction':new['loss']<wrong['loss'] and new['accuracy']>wrong['accuracy']}
    go=all(all(g.values()) for g in gates.values())
    result={'status':'DONE','source_sha256':sha(__file__),'plan_sha256':sha(HERE/'PLAN.md'),
        'official_test_loaded':False,'training_started':False,'results':results,
        'input_sha256':inputs,'frozen_inputs_unchanged':True,'gates':gates,
        'decision':'FOLLOW_UP_ONLY' if go else 'STOP_THIS_FIXED_PRIOR_RULE'}
    r.atomic_json(out/'result.json',result)
    print(json.dumps({'decision':result['decision'],'gates':gates,
        'summary':{name:{k:{m:round(v[m],8) for m in ('accuracy','loss')} for k,v in item['metrics'].items()} for name,item in results.items()}},indent=2))


if __name__=='__main__':main()

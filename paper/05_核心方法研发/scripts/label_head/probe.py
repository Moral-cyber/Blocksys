"""Frozen finite global-FL head-support diagnostic; development states only."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'method_followup'))
import run_followup as r


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def select(u,k,mandatory):
    mask=torch.zeros_like(u,dtype=torch.bool)
    mask[mandatory]=True
    h=int(mask.sum())
    if h>=k:
        raise ValueError('mandatory quota must be below K')
    scores=u.abs().clone()
    scores[mask]=-1
    rest=torch.argsort(scores,descending=True,stable=True)[:k-h]
    mask[rest]=True
    if int(mask.sum())!=k:
        raise AssertionError('K mismatch')
    sent=torch.where(mask,u,torch.zeros_like(u))
    residual=u-sent
    if not torch.equal(sent+residual,u):
        raise AssertionError('EF conservation failed')
    bound=(k-h)/(len(u)-h)
    if float(sent.double().square().sum())+1e-10 < bound*float(u.double().square().sum()):
        raise AssertionError('energy contraction bound failed')
    return sent,mask


def main():
    torch.set_num_threads(1)
    train,validation,manifest=r.prepare('Cifar10CleanA0p1',0,20261103,2000)
    model=r.SmallCNN()
    p=r.vectorize(model).numel()
    k=int(p*.05)
    head=torch.arange(p-650,p)
    headclass=torch.cat([torch.arange(10).repeat_interleave(64),torch.arange(10)])
    labels=[torch.bincount(ds.tensors[1],minlength=10) for ds in train]
    sources=HERE.parent/'method_followup/checkpoint_probes'
    out=HERE/'diagnostic_v1'
    out.mkdir(exist_ok=True)
    if (out/'result.json').exists():
        raise FileExistsError('finite diagnostic already completed; do not rerun as new evidence')
    rows=[]
    for name in ('cpu_topk20','cpu_align9020'):
        path=sources/name
        meta=json.loads((path/'result.json').read_text())
        if meta.get('data_manifest')!=manifest or meta.get('official_test_evaluated') is not False:
            raise ValueError('source data identity or test gate mismatch')
        state=torch.load(path/'shared_state.pt',weights_only=False)
        base=state['base']
        candidates=[d+rem for d,rem in zip(state['deltas'],state['residuals'])]
        diagnostics=[]
        aggregates={v:torch.zeros_like(base) for v in ('topk','all_head','supported_head','random_head_matched')}
        for i,u in enumerate(candidates):
            support=labels[i]>0
            mandatory=head[support[headclass]]
            gen=torch.Generator().manual_seed(20261144+i)
            randomhead=head[torch.randperm(650,generator=gen)[:len(mandatory)]]
            selections={'topk':torch.tensor([],dtype=torch.long),'all_head':head,
                'supported_head':mandatory,'random_head_matched':randomhead}
            record={'client':i,'observed_labels':support.nonzero().flatten().tolist(),
                'mandatory_supported_coordinates':len(mandatory),'variants':{}}
            top_energy=None
            for variant,forced in selections.items():
                sent,mask=select(u,k,forced)
                aggregates[variant]+=float(state['weights'][i])*sent
                energy=float(sent.double().square().sum())
                if variant=='topk':
                    top_energy=energy
                record['variants'][variant]={'head_selected':int(mask[head].sum()),
                    'supported_head_selected':int(mask[mandatory].sum()),
                    'sent_energy':energy,'retained_topk_energy':energy/(top_energy+1e-30),
                    'mandatory':len(forced),'analytical_bytes':64+8*k}
            diagnostics.append(record)
        for variant,aggregate in aggregates.items():
            r.assign_vector(model,base+aggregate)
            rows.append({'source':name,'variant':variant,'validation':r.metrics(model,validation),
                'analytical_upload_bytes':20*(64+8*k)})
        r.atomic_json(out/(name+'_clients.json'),diagnostics)
    comparisons=[]
    for name in ('cpu_topk20','cpu_align9020'):
        a=next(x['validation'] for x in rows if x['source']==name and x['variant']=='topk')
        for x in rows:
            if x['source']==name:
                x['delta_ce_vs_topk']=x['validation']['loss']-a['loss']
                x['delta_acc_pp_vs_topk']=100*(x['validation']['accuracy']-a['accuracy'])
        s=next(x for x in rows if x['source']==name and x['variant']=='supported_head')
        comparisons.append(s['delta_ce_vs_topk']<0 and s['delta_acc_pp_vs_topk']>0)
    result={'status':'DONE','rows':rows,'decision':'CHECK_SIMPLICITY_CONTROLS' if any(comparisons) else 'STOP_THIS_SUPPORT_RULE',
        'source_sha256':sha(__file__),'plan_sha256':sha(HERE/'PLAN.md'),
        'data_manifest':manifest,'source_states':{name:sha(sources/name/'shared_state.pt') for name in ('cpu_topk20','cpu_align9020')},
        'official_test_evaluated':False,'scope':'two shared CPU development states, not independent training/replicates'}
    r.atomic_json(out/'result.json',result)
    print(json.dumps({'rows':rows,'decision':result['decision']},indent=2))


if __name__=='__main__':
    main()

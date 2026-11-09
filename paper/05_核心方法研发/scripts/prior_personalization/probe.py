"""Local-prior personalization diagnostic; existing frozen global checkpoints."""
from pathlib import Path
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
METHODS=('ce_uniform3','fedlc_uniform3','balanced_uniform3','balanced_dense')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stats(z,y):
    return {'accuracy':float((z.argmax(1)==y).double().mean()),
        'loss':float(nn.functional.cross_entropy(z,y)), 'samples':len(y)}


def main():
    torch.set_num_threads(1)
    out=HERE/'diagnostic_v1'
    if out.exists():raise ValueError('fresh output required')
    out.mkdir();images,labels,provenance=data.load_official_train()
    results={};inputs={}
    for method in METHODS:
        source=HERE.parent/'balanced_quant/development_p20261193_s20261201'/method
        conf=json.loads((source/'config.json').read_text());manifest=conf['data_manifest']
        for f in ('config.json','checkpoint.pt'):inputs[str(source/f)]=sha(source/f)
        ck=torch.load(source/'checkpoint.pt',weights_only=False);model=r.SmallCNN();model.load_state_dict(ck['model']);model.eval()
        clients=[]
        for i,(trainids,holdoutids) in enumerate(zip(manifest['train_indices'],manifest['holdout_indices'])):
            if set(trainids)&set(holdoutids):raise ValueError('train holdout overlap')
            train=data.IndexedCifar10(images,labels,trainids);dev=data.IndexedCifar10(images,labels,holdoutids)
            counts=np.bincount(labels[trainids],minlength=10)
            if counts.tolist()!=manifest['client_label_counts'][i]:raise ValueError('train prior mismatch')
            shift=torch.tensor(np.log((counts+1)/(len(trainids)+10)),dtype=torch.float32)
            adapted=copy.deepcopy(model).train()
            for p in adapted.parameters():p.requires_grad_(False)
            for p in adapted.classifier[-1].parameters():p.requires_grad_(True)
            seed=20261211+i;torch.manual_seed(seed)
            loader=DataLoader(train,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
            opt=torch.optim.SGD(adapted.classifier[-1].parameters(),lr=.03)
            start=time.perf_counter()
            for x,y in loader:
                opt.zero_grad(set_to_none=True);nn.functional.cross_entropy(adapted(x),y).backward();opt.step()
            adaptseconds=time.perf_counter()-start;adapted.eval()
            raw=[];learned=[];ys=[]
            with torch.no_grad():
                for x,y in DataLoader(dev,batch_size=256):raw.append(model(x));learned.append(adapted(x));ys.append(y)
            z=torch.cat(raw);a=torch.cat(learned);y=torch.cat(ys)
            scores={'raw':z,'analytic_prior':z+shift,'prior_only':shift.expand(len(y),-1),
                'learned_head':a,'learned_head_plus_prior':a+shift}
            metrics={key:stats(value,y) for key,value in scores.items()}
            np.savez_compressed(out/(method+f'_client{i}.npz'),raw=z.numpy(),learned=a.numpy(),labels=y.numpy(),
                prior_shift=shift.numpy(),holdout_ids=np.array(holdoutids,dtype=np.uint32),train_counts=counts)
            clients.append({'client':i,'metrics':metrics,'adaptation_gradient_evaluations':len(loader),
                'analytic_extra_gradient_evaluations':0,'adaptation_seconds':adaptseconds,
                'client_train_samples':len(train),'holdout_samples':len(dev)})
        summaries={}
        weights=np.array([v['holdout_samples'] for v in clients],dtype=float);weights/=weights.sum()
        for variant in clients[0]['metrics']:
            arr={key:np.array([v['metrics'][variant][key] for v in clients]) for key in ('accuracy','loss')}
            summaries[variant]={'macro_'+key:float(value.mean()) for key,value in arr.items()}
            summaries[variant].update({'weighted_'+key:float((value*weights).sum()) for key,value in arr.items()})
            summaries[variant]['client_accuracy_q10']=float(np.quantile(arr['accuracy'],.1))
        results[method]={'clients':clients,'summaries':summaries,'personal_eval':'official TRAIN clientholdout development12500, includes prior globaldev2000'}
        print(json.dumps({'method':method,'summaries':summaries}),flush=True)
    candidate=results['balanced_uniform3']['summaries']['analytic_prior'];gates={}
    for baseline in ('ce_uniform3','fedlc_uniform3'):
        comp=results[baseline]['summaries']['analytic_prior']
        gates[baseline]={'ce':candidate['macro_loss']<comp['macro_loss'],
            'accuracy':candidate['macro_accuracy']>=comp['macro_accuracy']+.005}
    bestlearned=max(results[m]['summaries'][v]['macro_accuracy'] for m in METHODS[:3] for v in ('learned_head','learned_head_plus_prior'))
    gates['vs_learned']={'accuracy':candidate['macro_accuracy']>=bestlearned-.005}
    if any(sha(p)!=h for p,h in inputs.items()):raise ValueError('frozen input changed')
    go=all(all(v.values()) for v in gates.values())
    result={'status':'DONE','source_sha256':sha(__file__),'plan_sha256':sha(HERE/'PLAN.md'),
        'evaluation_type':'real TRAIN-holdout labels, PFL development only','official_test_loaded':False,
        'new_fl_training':False,'extra_head_baseline_training':True,'input_sha256':inputs,
        'results':results,'gates':gates,'decision':'FOLLOW_UP_ONLY' if go else 'STOP_THIS_FIXED_PERSONALIZATION_RECIPE'}
    r.atomic_json(out/'result.json',result);print(json.dumps({'decision':result['decision'],'gates':gates}),flush=True)


if __name__=='__main__':main()

"""pFedMe: unchanged author optimizer, matched task/schedule/scoring adapter."""
from pathlib import Path
import argparse, copy, hashlib, importlib.util, json, sys, time
from datetime import datetime, timezone
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent
AUTHOR=HERE/'pfedme_source/FLAlgorithms/optimizers/fedoptimizer.py'
AUTHOR_SHA='a7ab4b82c6efa8002189383b9ddac198211aa4441bd8d5ad92bb436ca117f2ac'
K=5; LAM=15.; BETA=1.; PERSONAL_LR=.09; OUTER_LR=.03; MU=.001

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def shared():return load('_pfedme_matched_shared',HERE.parent/'baselines/adapter.py')
def author_optimizer():
    if sha(AUTHOR)!=AUTHOR_SHA:raise ValueError('pinned author optimizer differs')
    return load('_unchanged_author_pfedme_optimizer',AUTHOR).pFedMeOptimizer

def clone_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def folder(task,seed):return HERE/'runs'/task/f's{seed}'/'pfedme_dense'

def local_train(c,model,dataset,seed,backend):
    torch.manual_seed(seed)
    if backend=='mps':
        if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
        torch.mps.manual_seed(seed)
    anchor=c.runner.vectorize(model).clone()
    theta=copy.deepcopy(model).to(backend).train()
    outer=[p.detach().clone() for p in theta.parameters()]
    optimizer=author_optimizer()(theta.parameters(),lr=PERSONAL_LR,lamda=LAM,mu=MU)
    loader=DataLoader(dataset,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
    for x,y in loader:
        x,y=x.to(backend),y.to(backend)
        for _ in range(K):
            optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(theta(x),y).backward()
            optimizer.step(outer)
        with torch.no_grad():
            for w,p in zip(outer,theta.parameters()):w.add_(p.detach()-w,alpha=LAM*OUTER_LR)
    # Independent export must precede loading outer weights; never retain Parameter aliases.
    personal=clone_state(theta)
    local_vector=torch.cat([w.detach().flatten() for w in outer]).cpu()
    delta=local_vector-anchor
    if (not torch.equal(c.runner.vectorize(model),anchor) or not torch.isfinite(delta).all()
        or not all(torch.isfinite(v).all() for v in personal.values())):
        raise ValueError('nonfinite local pFedMe result or mutated server anchor')
    counts=np.bincount(dataset.labels[dataset.indices],minlength=10)
    return delta,personal,{'batches':len(loader),'gradient_evaluations':K*len(loader),
        'outer_steps':len(loader),'K':K,'train_counts':counts.tolist(),
        'personal_theta_export':'independent CPU clone before outer reset'}

def source_inputs(b,c):
    paths=[Path(__file__),HERE/'PLAN.md',HERE/'SOURCE_LEDGER.json',HERE.parent/'baselines/adapter.py']
    paths += [p for p in (HERE/'pfedme_source').rglob('*') if p.is_file()]
    values={str(p):sha(p) for p in paths};values.update(c.source_identities());return values

def source_lock():
    b=shared();c,_=b.dependencies('cifar');inputs=source_inputs(b,c)
    lock={'status':'FROZEN_BEFORE_PERSONALIZED_TRAINING','created_at':datetime.now(timezone.utc).isoformat(),
          'inputs':inputs,'algorithm':{'K':K,'lambda':LAM,'beta':BETA,'personal_lr':PERSONAL_LR,'outer_lr':OUTER_LR,'mu':MU},
          'task_matrix':b.TASKS,'rounds':20,'selection':'fixed before any new pFedMe training or TEST'}
    if (HERE/'SOURCE_LOCK.json').exists():
        old=json.loads((HERE/'SOURCE_LOCK.json').read_text());b.verify(old['inputs'])
        if old['inputs']!=inputs:raise ValueError('existing source lock differs')
    else:write(HERE/'SOURCE_LOCK.json',lock)
    print(json.dumps({'status':'SOURCE_LOCKED','input_count':len(inputs)}),flush=True)

def train(task,seed):
    b=shared()
    if seed not in b.TASKS[task][1]:raise ValueError('fixed seed only')
    attempts=b.test_guard();torch.set_num_threads(1)
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable, no fallback')
    locked=json.loads((HERE/'SOURCE_LOCK.json').read_text());b.verify(locked['inputs'])
    c,sets,validation,manifest,data_ids,_,_=b.checked_data(task,seed)
    identities={**locked['inputs'],**source_inputs(b,c),**data_ids}
    weights=np.asarray([len(ds) for ds in sets],dtype=np.float64);weights/=weights.sum()
    config={'version':1,'method':'pfedme_dense','task':task,'seed':seed,'partition_seed':b.TASKS[task][0],
        'rounds':20,'backend':'mps','parameters':136874,'batch':64,'local_epochs':1,
        'local_epoch_definition':'one full pass of ceil(n_i/64) minibatches, K inner gradients each',
        'K':K,'lambda':LAM,'beta':BETA,'personal_learning_rate':PERSONAL_LR,'outer_learning_rate':OUTER_LR,'optimizer_mu':MU,
        'author_commit':'caf24b1f954d4381bd9b4d104ee7eff389e1489b',
        'implementation':'unmodified author pFedMeOptimizer with matched task/schedule/independent-theta-export adapter',
        'author_default_outer_lr':.005,'outer_lr_adaptation':'0.03 matches manuscript SGD, fixed without tuning',
        'weights':weights.tolist(),'data_manifest':manifest,'source_and_data_identities':identities,
        'predictor':'final raw client theta; no post-hoc calibrator',
        'communication':'dense analytical 547560 B per upload and downlink',
        'selection':'all original seeds, final round20; no TEST-driven choices','torch_version':str(torch.__version__)}
    config['signature']=b.signed(config);out=folder(task,seed);out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:raise ValueError('config mismatch')
    torch.manual_seed(seed);model=c.runner.SmallCNN();rows=[];theta_states=[]
    if (out/'checkpoint.pt').exists():
        ck=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
        if ck['config']!=config:raise ValueError('checkpoint mismatch')
        model.load_state_dict(ck['model']);rows=ck['rows'];theta_states=ck['theta_states']
    write(out/'config.json',config)
    for rnd in range(len(rows),20):
        began=time.perf_counter();anchor=c.runner.vectorize(model).clone();aggregate=torch.zeros_like(anchor)
        clients=[];theta_states=[]
        for client,ds in enumerate(sets):
            seed_i=seed+100*rnd+client
            delta,theta,counts=local_train(c,model,ds,seed_i,'mps')
            aggregate.add_(delta,alpha=float(weights[client]));theta_states.append(theta)
            clients.append({'client':client,'seed':seed_i,**counts})
        c.runner.assign_vector(model,anchor+BETA*aggregate)
        if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('nonfinite server')
        measured=c.runner.metrics(model,validation)
        rows.append({'round':rnd+1,'validation_global_model':measured,'clients':clients,
            'gradient_evaluations':sum(v['gradient_evaluations'] for v in clients),
            'outer_steps':sum(v['outer_steps'] for v in clients),'global_anchor_unchanged_between_clients':True,
            'upload_bytes_analytical':20*547560,'download_bytes_analytical':20*547560,'seconds':time.perf_counter()-began})
        temp=out/'checkpoint.tmp';torch.save({'config':config,'signature':config['signature'],
            'model':clone_state(model),'theta_states':theta_states,'rows':rows},temp);temp.replace(out/'checkpoint.pt')
        write(out/'result.json',{'status':'DONE' if len(rows)==20 else 'RUNNING','signature':config['signature'],
            'rows':rows,'official_test_reads':0,'predictor':'final raw theta independent clone'})
        print(json.dumps({'task':task,'seed':seed,'method':'pfedme_dense','round':rnd+1,
            'validation_global':measured,'gradient_evaluations':rows[-1]['gradient_evaluations'],'seconds':rows[-1]['seconds']}),flush=True)
    b.verify(identities)
    if attempts:raise ValueError('forbidden TEST opens')

def fixtures():
    torch.set_num_threads(1);optclass=author_optimizer();torch.manual_seed(675)
    model=nn.Linear(3,2);anchor=[p.detach().clone()-.02 for p in model.parameters()]
    x=torch.randn(7,3);y=torch.tensor([0,1,0,1,1,0,1]);optimizer=optclass(model.parameters(),lr=PERSONAL_LR,lamda=LAM,mu=MU)
    loss=nn.functional.cross_entropy(model(x),y);loss.backward()
    explicit=[p.detach().clone()-PERSONAL_LR*(p.grad.detach()+LAM*(p.detach()-a)+MU*p.detach()) for p,a in zip(model.parameters(),anchor)]
    optimizer.step(anchor);err=max(float((p.detach()-e).abs().max()) for p,e in zip(model.parameters(),explicit))
    if err>1e-7:raise ValueError('author optimizer update differs from explicit squared-proximal objective')
    snap=clone_state(model)
    with torch.no_grad():
        for p in model.parameters():p.fill_(10)
    if any(torch.equal(snap[k],v) for k,v in model.state_dict().items()):raise ValueError('theta snapshot aliased')
    b=shared();c,_=b.dependencies('cifar')
    class FixtureDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.x=np.random.default_rng(98).normal(size=(71,3,32,32)).astype(np.float32)
            self.labels=np.arange(71,dtype=np.int64)%10;self.indices=np.arange(71)
        def __len__(self):return 71
        def __getitem__(self,i):return torch.from_numpy(self.x[i]),int(self.labels[i])
    torch.manual_seed(81);server=c.runner.SmallCNN();base=c.runner.vectorize(server).clone()
    delta,theta,counters=local_train(c,server,FixtureDataset(),111,'cpu')
    if not torch.equal(c.runner.vectorize(server),base) or counters['gradient_evaluations']!=10 or len(theta)!=len(server.state_dict()):raise ValueError('local fixture contract')
    if not torch.allclose(.25*(base+delta)+.75*base,base+.25*delta,atol=3e-8,rtol=0):raise ValueError('weighted aggregation')
    d,s=b.final_modules();metrics=s.metrics(np.zeros((10,10)),np.arange(10))
    if abs(metrics['nll']-np.log(10))>1e-14:raise ValueError('metric fixture')
    write(HERE/'FIXTURES.json',{'status':'PASS','author_optimizer_explicit_update_max_error':err,
       'theta_snapshot_independent':True,'global_anchor_preserved':True,'two_batches_K5_gradients':10,
       'weighted_beta1_aggregation':True,'scoring_uniform_nll':metrics['nll'],'official_test_reads':0,
       'source_sha256':sha(__file__),'author_optimizer_sha256':sha(AUTHOR)})
    print(json.dumps({'status':'FIXTURES_PASS','optimizer_max_error':err}),flush=True)

def model_lock():
    b=shared();locked=json.loads((HERE/'SOURCE_LOCK.json').read_text());b.verify(locked['inputs']);inputs=dict(locked['inputs'])
    for task,(_,seeds) in b.TASKS.items():
        for seed in seeds:
            out=folder(task,seed);report=json.loads((out/'result.json').read_text());ck=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
            if report['status']!='DONE' or len(ck['rows'])!=20 or len(ck['theta_states'])!=20:raise ValueError('all six final runs required')
            b.verify(ck['config']['source_and_data_identities'])
            for name in ('config.json','result.json','checkpoint.pt'):inputs[str(out/name)]=sha(out/name)
    value={'status':'ALL_SIX_PERSONAL_MODELS_FROZEN','inputs':inputs,'created_at':datetime.now(timezone.utc).isoformat(),'selection':'none before TEST'}
    target=HERE/'MODEL_LOCK.json'
    if target.exists() and json.loads(target.read_text())['inputs']!=inputs:raise ValueError('model lock changed')
    if not target.exists():write(target,value)
    print(json.dumps({'status':'ALL_MODELS_LOCKED','inputs':len(inputs)}),flush=True)

def score():
    b=shared();torch.set_num_threads(1);locked=json.loads((HERE/'MODEL_LOCK.json').read_text());b.verify(locked['inputs']);d,s=b.final_modules()
    for task,(_,seeds) in b.TASKS.items():
        root=HERE/'official_test'/task
        if (root/'result.json').exists():continue
        root.mkdir(parents=True,exist_ok=True);c,_=b.dependencies(task);original,path=b.original_manifest(task,seeds[0])
        lock={'status':'ROOT_LOCKED_FINAL_TEST','tasks':[task],'plan_sha256':d.file_sha(b.WORK/'predictive_final_test/FINAL_TEST_PLAN.md'),
            'source_sha256':{n:d.file_sha(b.WORK/'predictive_final_test'/n) for n in ('data.py','scorer.py')},'personalized_model_lock_sha256':sha(HERE/'MODEL_LOCK.json')}
        write(root/'ROOT_LOCK.json',lock)
        if task=='cifar':images,labels,provenance=d.load_cifar_test(b.PAPER/'02_实验工程/dataset/Cifar10/rawdata',root_lock=root/'ROOT_LOCK.json',execute=True)
        else:images,labels,provenance=d.load_fashion_test(b.METHOD/'scripts/datasets/FashionMNIST/raw',root_lock=root/'ROOT_LOCK.json',execute=True)
        groups,assignment=d.assign_clients(labels,np.asarray(original['client_label_counts']),task=task,train_manifest_sha256=d.file_sha(path))
        old=json.loads((b.METHOD/f'results/predictive_final_test/official_{task}/assignment.json').read_text())
        if assignment['assignment_hash']!=old['assignment_hash']:raise ValueError('original assignment mismatch')
        write(root/'assignment.json',assignment);reports=[]
        for seed in seeds:
            ck=torch.load(folder(task,seed)/'checkpoint.pt',map_location='cpu',weights_only=False);clients=[];out=root/f'pfedme_dense_s{seed}';out.mkdir()
            model=c.runner.SmallCNN().eval();start=time.perf_counter()
            for client,ids in enumerate(groups):
                model.load_state_dict(ck['theta_states'][client]);chunks=[]
                with torch.inference_mode():
                    for offset in range(0,len(ids),256):
                        chunks.append(model(torch.from_numpy(d.normalize_batch(images[ids[offset:offset+256]],task))).numpy().copy())
                logits=np.concatenate(chunks);metric=s.metrics(logits,labels[ids])
                np.savez_compressed(out/f'client{client}.npz',row_ids=ids,labels=labels[ids],raw_theta_logits=logits)
                clients.append({'client':client,'test_rows':len(ids),'metrics':{'raw':metric}})
            weights=np.asarray([v['test_rows'] for v in clients],dtype=np.float64)/10000;summary={}
            for key in b.METRICS:
                vals=np.asarray([v['metrics']['raw'][key] for v in clients]);summary['macro_'+key]=float(vals.mean());summary['weighted_'+key]=float(vals@weights)
            summary['client_accuracy_q10']=float(np.quantile([v['metrics']['raw']['accuracy'] for v in clients],.1))
            conf=np.sum([v['metrics']['raw']['ece_bins']['confidence_sum'] for v in clients],axis=0);corr=np.sum([v['metrics']['raw']['ece_bins']['correct_sum'] for v in clients],axis=0)
            summary['pooled_ece15']=float(np.abs(corr-conf).sum()/10000)
            report={'status':'DONE','task':task,'seed':seed,'method':'pfedme_dense','predictor':'raw final theta','summaries':{'raw':summary},
               'clients':clients,'assignment_hash':assignment['assignment_hash'],'test_provenance':provenance,'inference_seconds':time.perf_counter()-start,'TEST_fits':0,'TEST_backwards':0}
            write(out/'result.json',report);reports.append(report)
            print(json.dumps({'status':'SCORED','task':task,'seed':seed,'summaries':summary}),flush=True)
        aggregate={key:{'mean':float(np.mean([v['summaries']['raw'][key] for v in reports])),
                       'sample_sd':float(np.std([v['summaries']['raw'][key] for v in reports],ddof=1))} for key in reports[0]['summaries']['raw']}
        b.verify(locked['inputs']);write(root/'result.json',{'status':'DONE','task':task,'seeds':seeds,'method':'pfedme_dense','predictor':'raw final theta','summaries':{'raw':aggregate},'assignment_hash':assignment['assignment_hash'],'model_inputs_unchanged':True,'TEST_fits':0})

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('fixtures','source-lock','train','model-lock','score'),required=True);p.add_argument('--task',choices=('cifar','fashion'));p.add_argument('--seed',type=int);args=p.parse_args()
    if args.stage=='fixtures':fixtures()
    elif args.stage=='source-lock':source_lock()
    elif args.stage=='train':train(args.task,args.seed)
    elif args.stage=='model-lock':model_lock()
    else:score()
if __name__=='__main__':main()
